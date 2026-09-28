"""
scripts/e4_temporal.py — EXPLORATORY time-aware subtype probe (ANALYSIS_PLAN §11, 2026-09-28).

E4 proper uses p̄_b and six Fourier moments, which are order-invariant by construction, so
they cannot separate subtypes that differ only in temporal order (e.g. synthetic growing
vs falling). This add-on asks whether QVR's per-timepoint outputs carry that information.

Features per series (fixed model, D mode 'mc', inference seed 0):
  z_t      per-timepoint ⟨Z_q⟩, flattened (T × 3)       — time-aware, QVR
  z_mean   ⟨Z_q⟩ averaged over time (3)                  — order-invariant, QVR
  pbar     eigenbasis fingerprint p̄_b (8)                — order-invariant, QVR
  raw      the normalized series itself (T)              — time-aware, classical analog
  moments  six time-averaged Fourier moments             — order-invariant, classical

Same probe protocol as E4 (scripts/run_suite.py::e4_probe): fit on Xprobe, evaluate on the
pooled anomalous test sets (and on Xval), eligible classes only, logistic (C = 0.1) and
gradient-boosted trees; unit-level permutation null (--n-perm) for the reference N_E,
first seed, logistic probe.

Runs after scripts/run_suite.py has trained the models (reads <suite>/models/).

Usage:
    python scripts/e4_temporal.py --data-dir data_splits/synthetic --suite-dir results/suite/synthetic
    python scripts/e4_temporal.py --self-test
"""

import argparse
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qvr.data import t_grid                                                   # noqa: E402
from qvr.infer import infer                                                   # noqa: E402
from run_suite import REF_NE, anomaly_sets, e4_probe, fourier_moments, load_split  # noqa: E402

FEATURES = ('z_t', 'z_mean', 'pbar', 'raw', 'moments')


def features(params, cfg, X):
    t = t_grid(X.shape[1])
    o = infer(params, cfg, X, t, mode='mc', seed=0, keep_time=True)
    return {'z_t': o['z_t'].reshape(len(X), -1), 'z_mean': o['zrepr'], 'pbar': o['pbar'],
            'raw': X.numpy(), 'moments': fourier_moments(X)}


def run(data_dir, suite_dir, out=None, channel=0, n_perm=1000):
    from qvr.train import load_run
    suite_dir = Path(suite_dir)
    out = Path(out) if out else suite_dir / 'e4_temporal'
    out.mkdir(parents=True, exist_ok=True)
    cfgj = json.loads((suite_dir / 'config.json').read_text())
    nes, seeds = cfgj['nes'], cfgj['seeds']
    ref = REF_NE if REF_NE in nes else nes[-1]
    X, units, sub, man = load_split(data_dir, channel)
    sets = anomaly_sets(X)
    elig = man.get('e4_eligibility') or {}
    classes = [c for c, v in elig.items() if v.get('eligible')] if elig else sorted(set(sub['Xprobe']))
    y_te = np.concatenate([sub.get(k, np.array([k[4:]] * len(X[k]))) for k in sets])
    X_te = torch.cat([X[k] for k in sets])
    rep = {'exploratory': True, 'classes': classes, 'reference_ne': ref, 'per_ne': {}}
    print('=' * 88)
    print(f'EXPLORATORY time-aware E4 — {Path(data_dir).name}, classes {classes}, ref N_E {ref}')
    print('=' * 88)
    for ne in nes:
        rows = []
        for s in seeds:
            params, cfg, _ = load_run(suite_dir / 'models' / f'ne{ne}_s{s}')
            Fp, Ft = features(params, cfg, X['Xprobe']), features(params, cfg, X_te)
            Fv = features(params, cfg, X['Xval']) if 'Xval' in X and 'Xval' in sub else None
            r = {}
            for f in FEATURES:
                perm = n_perm if (ne == ref and s == seeds[0]) else 0
                r[f] = e4_probe(Fp[f], sub['Xprobe'], units['Xprobe'], Ft[f], y_te, classes, perm)
                if Fv is not None:
                    r[f + '_to_val'] = e4_probe(Fp[f], sub['Xprobe'], units['Xprobe'], Fv[f], sub['Xval'], classes)
            rows.append(r)
        rep['per_ne'][str(ne)] = rows
        line = '  '.join(f"{f} {np.mean([r[f]['logreg']['macro_auc'] for r in rows if r[f]]):.3f}/"
                         f"{np.mean([r[f]['hgb']['macro_auc'] for r in rows if r[f]]):.3f}" for f in FEATURES)
        print(f'  N_E={ne:<3d} macro AUC logreg/hgb (mean over seeds): {line}')
    r0 = rep['per_ne'][str(ref)][0]
    print(f'\n  reference N_E={ref}, seed {seeds[0]}, logistic probe vs unit-level permutation null:')
    for f in FEATURES:
        m = r0[f]['logreg'] if r0[f] else None
        if m and 'p_value' in m:
            print(f"    {f:8s} macro AUC {m['macro_auc']:.3f}  bal.acc {m['bal_acc']:.3f}  κ {m['kappa']:.3f}  "
                  f"| null p95 {m['null_p95']:.3f}  p = {m['p_value']:.3f}")
    (out / 'e4_temporal.json').write_text(json.dumps(rep, indent=1, default=float))
    print(f'\nsaved → {out}/e4_temporal.json')
    return rep


def _self_test():
    import tempfile
    from prepare_dataset import SYN_SPLITS, prepare_synthetic
    from run_suite import run as suite_run
    print('e4_temporal self-test: tiny synthetic suite (train only), then time-aware probes')
    SYN_SPLITS['normal'].update({'Xtr': 120, 'Xval_norm': 30, 'Xte_norm': 40})
    SYN_SPLITS['anomaly'].update({'Xval': 20, 'Xprobe': 60, 'Xte': 40})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prepare_synthetic(Path(tmp) / 'syn', seq_len=12)
        suite_run(Path(tmp) / 'syn', Path(tmp) / 'suite', nes=[1], seeds=[0], stages=('train',), max_samples=80)
        rep = run(Path(tmp) / 'syn', Path(tmp) / 'suite', n_perm=20)
        r = rep['per_ne']['1'][0]
        # growing vs falling only: order-invariant features must fail, raw series must succeed
        from run_suite import load_split as ls
        X, units, sub, _ = ls(Path(tmp) / 'syn')
        gf = ['falling', 'growing']
        y_te = np.concatenate([sub[k] for k in ('Xte_falling', 'Xte_growing')])
        Xte = torch.cat([X['Xte_falling'], X['Xte_growing']])
        mom = e4_probe(fourier_moments(X['Xprobe']), sub['Xprobe'], units['Xprobe'], fourier_moments(Xte), y_te, gf)
        raw = e4_probe(X['Xprobe'].numpy(), sub['Xprobe'], units['Xprobe'], Xte.numpy(), y_te, gf)
        checks = {
            'all five feature sets probed': all(r[f] is not None for f in FEATURES),
            'permutation null at reference': 'p_value' in r['z_t']['logreg'],
            'growing vs falling: moments ≈ chance (order-invariant)': abs(mom['logreg']['macro_auc'] - 0.5) < 0.15,
            'growing vs falling: raw series separates (time-aware)': raw['logreg']['macro_auc'] > 0.95,
            'json written': (Path(tmp) / 'suite' / 'e4_temporal' / 'e4_temporal.json').exists(),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir')
    ap.add_argument('--suite-dir')
    ap.add_argument('--out')
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not (a.data_dir and a.suite_dir):
            ap.error('--data-dir and --suite-dir are required')
        run(a.data_dir, a.suite_dir, a.out, a.channel, a.n_perm)