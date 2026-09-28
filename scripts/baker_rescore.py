"""
scripts/baker_rescore.py — reproduction diagnostic for Baker et al.'s validation BA 0.82.

No retraining: re-scores every run saved by scripts/baker_e0.py.

Check 1 — which validation set V?  The paper's two versions define V differently:
    arXiv v2   V from the single-transaction conditions (U±, B±)      → 'arxiv_single'
    AAAI 2025  V from the multi-transaction conditions (Ũ±, B̃±)       → 'aaai_multi'
  The released arrays contain one Xval ('released'). The two alternative pools are
  built from the TEST sets (balanced, fixed seed), so they are optimistic and purely
  interpretive: they show what validation BA each definition of V would give,
  not a reportable test result.

Check 2 — scoring resolution.  The notebook scores every 4th point with a fresh
  linspace(0.1, 2π, 45); 'full' scores all 180 points on the training time grid.

Validation BA = balanced accuracy of V vs Xte_norm at the best threshold on the
notebook's grid (ζ ∈ linspace(0, 1, 10000)), as in the paper.

Usage:
    python scripts/baker_rescore.py --runs results/e0/baker results/e0/baker_paper_ry \
        results/e0/baker_paper_iter results/e0/baker_nb_tau5 --out results/e0/baker_rescore
    python scripts/baker_rescore.py --self-test
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qvr import circuit as C                                   # noqa: E402
from qvr.data import roc_auc                                   # noqa: E402
from qvr.datasets import baker                                 # noqa: E402
from qvr.infer import infer                                    # noqa: E402
from baker_e0 import T_TRAIN, STRIDE, best_threshold           # noqa: E402

GRID = np.linspace(0, 1, 10000)
MULTI = ('Xte_dirty_usdt_pos', 'Xte_dirty_usdt_neg', 'Xte_dirty_btc_pos', 'Xte_dirty_btc_neg')
SINGLE = ('Xte_clean_usdt', 'Xte_clean_btc')


def balanced_pool(data, names, n_total=60, seed=0):
    names = [n for n in names if n in data]
    per = min(n_total // len(names), min(len(data[n]) for n in names))
    rng = np.random.default_rng(seed)
    return torch.cat([data[n][torch.as_tensor(rng.choice(len(data[n]), per, replace=False))] for n in names])


def score(params, cfg, X, hist, resolution, seed=0):
    if resolution == 'stride4':
        Xs = X[:, ::STRIDE]
        t = torch.linspace(0.1, 2 * math.pi, Xs.shape[1], dtype=torch.float64)
    else:
        Xs, t = X, T_TRAIN
    raw = infer(params, cfg, Xs, t, mode=cfg.draw_mode, seed=seed)['score']
    a_x = np.abs(2 * min(hist) - 2 * float(C.arctan_penalty(params, cfg)) - raw)
    return {'raw': raw, 'a_x': a_x}


def run(run_dirs, data_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = baker.load_all(data_dir)
    V = {'released': data['Xval'],
         'aaai_multi': balanced_pool(data, MULTI),
         'arxiv_single': balanced_pool(data, SINGLE)}
    print('=' * 78)
    print('Baker re-scoring diagnostic — V sizes: ' + ', '.join(f'{k} {len(v)}' for k, v in V.items()))
    print('  (aaai_multi / arxiv_single are drawn from TEST sets: interpretive, optimistic)')
    print('=' * 78)

    report = {'v_sizes': {k: len(v) for k, v in V.items()}, 'variants': {}}
    for rd in run_dirs:
        rep = json.loads((Path(rd) / 'baker_e0.json').read_text())
        for var, q in rep['qvr'].items():
            cfg = C.QVRConfig(**q['cfg'])
            key = q['score_key']
            res = {r: {v: [] for v in V} for r in ('stride4', 'full')}
            aucs = {r: [] for r in ('stride4', 'full')}
            for run_ in q['runs']:
                p = {k: torch.tensor(v, dtype=torch.float64) for k, v in run_['params'].items()}
                for r in ('stride4', 'full'):
                    sn = score(p, cfg, data['Xte_norm'], run_['hist'], r)[key]
                    for vname, Xv in V.items():
                        _, ba = best_threshold(sn, score(p, cfg, Xv, run_['hist'], r)[key], GRID)
                        res[r][vname].append(ba)
                    su = score(p, cfg, data['Xte_dirty_usdt_pos'], run_['hist'], r)[key]
                    aucs[r].append(roc_auc(np.r_[np.zeros(len(sn)), np.ones(len(su))], np.r_[sn, su]))
            report['variants'][var] = {'n_runs': len(q['runs']), 'score_key': key, 'val_ba': res, 'auc_usdt_pos': aucs}
            print(f"\n{var} ({len(q['runs'])} runs, score {key})")
            for r in ('stride4', 'full'):
                line = '  '.join(f"{v}: best {max(b):.3f} / median {np.median(b):.3f}" for v, b in res[r].items())
                print(f"  {r:7s} val BA  {line}   | Ũ₊ AUC median {np.median(aucs[r]):.3f}")
    all_best = {(r, v): max(max(report['variants'][k]['val_ba'][r][v]) for k in report['variants'])
                for r in ('stride4', 'full') for v in V}
    print('\n' + '=' * 78)
    print('Best validation BA over ALL runs and variants (published: 0.82)')
    for (r, v), b in all_best.items():
        print(f'  {r:7s} {v:13s} {b:.3f}')
    report['best_overall'] = {f'{r}/{v}': b for (r, v), b in all_best.items()}
    (out / 'baker_rescore.json').write_text(json.dumps(report, indent=1))
    print(f'saved → {out}/baker_rescore.json')
    return report


def _self_test():
    import pickle
    import tempfile
    from baker_e0 import run as e0_run
    print('baker_rescore self-test: fake data, tiny e0 run, then re-score')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = rng.normal(0, 0.6, (1, 2, 180))
        sets = [('Xtr', 60, 0), ('Xval', 30, 0.6), ('Xte_norm', 30, 0)] + \
               [(n, 30, 1.0) for n in MULTI] + [(n, 20, 0.2) for n in SINGLE]
        for name, n, sh in sets:
            with open(Path(tmp) / f'{name}.pickle', 'wb') as f:
                pickle.dump(np.clip(base + rng.normal(0, 0.4, (n, 2, 180)) + sh, -np.pi, np.pi), f)
        e0_run(tmp, Path(tmp) / 'e0', seeds=2, restarts=2, maxfev_nb=30, maxfev_paper=30, n_boot=20)
        rep = run([Path(tmp) / 'e0'], tmp, Path(tmp) / 'out')
        checks = {
            'both variants re-scored': set(rep['variants']) == {'notebook', 'paper'},
            'three V definitions × two resolutions': all(
                set(rep['variants'][v]['val_ba'][r]) == {'released', 'aaai_multi', 'arxiv_single'}
                for v in rep['variants'] for r in ('stride4', 'full')),
            'easy multi-transaction V ≥ hard single-transaction V (fake data)':
                rep['best_overall']['stride4/aaai_multi'] >= rep['best_overall']['stride4/arxiv_single'],
            'json written': (Path(tmp) / 'out' / 'baker_rescore.json').exists(),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+')
    ap.add_argument('--data-dir', default=str(baker.DATA_DIR))
    ap.add_argument('--out', default='results/e0/baker_rescore')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not a.runs:
            ap.error('--runs required')
        run(a.runs, a.data_dir, a.out)