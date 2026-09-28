"""
scripts/baker_incremental.py — does QVR carry information beyond the one-number baseline?

For a saved E0 run (default: the normals-only-scaling follow-up), per anomaly set and
pooled over the four multi-transaction sets, test normals vs anomalies:

  H   one-number baseline (vol_mean: |window-mean volume − train median| / MAD)
  Q   QVR anomaly score (the run's score key)
  H+Q logistic combination of standardized H and Q

  AUC(H), AUC(Q), AUC(H+Q) from out-of-fold predictions (5-fold stratified CV,
  repeated 5×; the logistic model only learns the 2 combination weights),
  ΔAUC(H+Q − H) and ΔAUC(H+Q − Q) with 2,000-resample bootstrap CIs over windows,
  and Spearman ρ(H, Q).

Done for the best restart (selected on validation, as published) and summarized over
all restarts (distribution of ΔAUC(H+Q − H)). The other prespecified baselines are
reported alongside from the run file.

Usage:
    python scripts/baker_incremental.py \
        --run-json results/e0/baker_fu_bi_train782_normscale/baker_e0.json \
        --data-dir results/e0/baker_sets/bi_train782_normscale \
        --out results/e0/baker_incremental
    python scripts/baker_incremental.py --self-test
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
from qvr import circuit as C                         # noqa: E402
from qvr.data import roc_auc                         # noqa: E402
from qvr.datasets import baker                       # noqa: E402
from baker_e0 import baseline_scores, qvr_scores     # noqa: E402

MULTI = ('Xte_dirty_usdt_pos', 'Xte_dirty_usdt_neg', 'Xte_dirty_btc_pos', 'Xte_dirty_btc_neg')


def oof_combined(H, Q, y, reps=5, folds=5, seed=0):
    """Out-of-fold probabilities of a logistic model on standardized [H, Q]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    X = np.c_[H, Q]
    p = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            p[te] += m.predict_proba(sc.transform(X[te]))[:, 1] / reps
    return p


def compare(H, Q, y, n_boot=2000, seed=0):
    from scipy.stats import spearmanr
    HQ = oof_combined(H, Q, y, seed=seed)
    a = {'H': roc_auc(y, H), 'Q': roc_auc(y, Q), 'HQ': roc_auc(y, HQ)}
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    dH, dQ = [], []
    for _ in range(n_boot):
        ix = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        yy = y[ix]
        hq = roc_auc(yy, HQ[ix])
        dH.append(hq - roc_auc(yy, H[ix]))
        dQ.append(hq - roc_auc(yy, Q[ix]))
    ci = lambda v: ([float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
                    if len(v) else [float('nan'), float('nan')])
    return {'auc': a, 'delta_HQ_minus_H': a['HQ'] - a['H'], 'ci_HQ_minus_H': ci(dH),
            'delta_HQ_minus_Q': a['HQ'] - a['Q'], 'ci_HQ_minus_Q': ci(dQ),
            'spearman_H_Q': float(spearmanr(H, Q).statistic), 'n': [int(len(i0)), int(len(i1))]}


def run(run_json, data_dir, out, variant='paper', n_boot=2000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads(Path(run_json).read_text())
    q = rep['qvr'][variant]
    cfg = C.QVRConfig(**q['cfg'])
    data = baker.load_all(data_dir)
    sets = [s for s in baker.anomaly_sets(data)]
    H = baseline_scores(data, 'vol_mean')

    def qscores(run_):
        p = {k: torch.tensor(v, dtype=torch.float64) for k, v in run_['params'].items()}
        return {k: qvr_scores(p, cfg, data[k], run_['hist'])[q['score_key']] for k in ['Xte_norm'] + sets}

    def per_set(Qs, boot):
        res = {}
        for s in sets + ['pooled_multi']:
            names = [n for n in MULTI if n in sets] if s == 'pooled_multi' else [s]
            if not names:
                continue
            y = np.r_[np.zeros(len(H['Xte_norm'])), np.ones(sum(len(H[n]) for n in names))]
            h = np.r_[H['Xte_norm'], np.concatenate([H[n] for n in names])]
            qq = np.r_[Qs['Xte_norm'], np.concatenate([Qs[n] for n in names])]
            res[s] = compare(h, qq, y, boot)
        return res

    best = q['runs'][q['best_run']]
    report = {'run_json': str(run_json), 'data_dir': str(data_dir), 'variant': variant,
              'score_key': q['score_key'], 'best_run': q['best_run'],
              'best': per_set(qscores(best), n_boot), 'all_runs': {},
              'other_baselines': {k: {s: v['sets'][s]['auc'] for s in sets}
                                  for k, v in rep['baselines'].items()}}
    deltas = {s: [] for s in list(report['best'])}
    for run_ in q['runs']:
        r = per_set(qscores(run_), 0)          # point estimates only for the restart summary
        for s in deltas:
            deltas[s].append(r[s]['delta_HQ_minus_H'])
    report['all_runs'] = {s: {'median_delta_HQ_minus_H': float(np.median(v)),
                              'frac_positive': float(np.mean(np.array(v) > 0)),
                              'q25_q75': [float(np.quantile(v, 0.25)), float(np.quantile(v, 0.75))]}
                          for s, v in deltas.items()}

    print('=' * 92)
    print(f'Incremental information — {Path(run_json).parent.name}, best restart {q["best_run"]} '
          f'(score {q["score_key"]}), {len(q["runs"])} restarts')
    print('=' * 92)
    print(f"{'set':20s} {'AUC H':>7s} {'AUC Q':>7s} {'AUC H+Q':>8s}  {'Δ(H+Q−H) [95% CI]':>24s}  "
          f"{'Δ(H+Q−Q) [95% CI]':>24s} {'ρ(H,Q)':>7s} | all restarts: median Δ(H+Q−H), % > 0")
    for s, r in report['best'].items():
        a = report['all_runs'][s]
        print(f"{s[4:] if s.startswith('Xte_') else s:20s} {r['auc']['H']:7.3f} {r['auc']['Q']:7.3f} {r['auc']['HQ']:8.3f}  "
              f"{r['delta_HQ_minus_H']:+.3f} [{r['ci_HQ_minus_H'][0]:+.3f}, {r['ci_HQ_minus_H'][1]:+.3f}]  "
              f"{r['delta_HQ_minus_Q']:+.3f} [{r['ci_HQ_minus_Q'][0]:+.3f}, {r['ci_HQ_minus_Q'][1]:+.3f}] "
              f"{r['spearman_H_Q']:+7.3f} | {a['median_delta_HQ_minus_H']:+.3f}, {a['frac_positive']:.0%}")
    print('\nOther prespecified baselines (AUC, same data):')
    for k, v in report['other_baselines'].items():
        print(f"  {k:13s} " + '  '.join(f"{s[4:]} {x:.3f}" for s, x in v.items()))
    (out / 'baker_incremental.json').write_text(json.dumps(report, indent=1))
    print(f'\nsaved → {out}/baker_incremental.json')
    return report


def _self_test():
    import pickle
    import tempfile
    from baker_e0 import run as e0_run
    print('baker_incremental self-test: fake data, tiny e0 run, then incremental test')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = rng.normal(0, 0.6, (1, 2, 180))
        for name, n, sh in [('Xtr', 60, 0), ('Xval', 30, 0.6), ('Xte_norm', 30, 0)] + [(m, 30, 0.8) for m in MULTI]:
            with open(Path(tmp) / f'{name}.pickle', 'wb') as f:
                pickle.dump(np.clip(base + rng.normal(0, 0.4, (n, 2, 180)) + sh, -np.pi, np.pi), f)
        e0_run(tmp, Path(tmp) / 'e0', restarts=3, maxfev_paper=30, n_boot=20, variants=('paper',))
        r = run(Path(tmp) / 'e0' / 'baker_e0.json', tmp, Path(tmp) / 'out', n_boot=200)
        b = r['best']['pooled_multi']
        checks = {
            'all sets + pooled evaluated': set(r['best']) == set(MULTI) | {'pooled_multi'},
            'AUCs in [0, 1]': all(0 <= v <= 1 for v in b['auc'].values()),
            'CI brackets point estimate': b['ci_HQ_minus_H'][0] <= b['delta_HQ_minus_H'] + 1e-9
                                          or b['ci_HQ_minus_H'][1] >= b['delta_HQ_minus_H'] - 1e-9,
            'all-restart summary present': len(r['all_runs']) == 5,
            'json written': (Path(tmp) / 'out' / 'baker_incremental.json').exists(),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-json', default='results/e0/baker_fu_bi_train782_normscale/baker_e0.json')
    ap.add_argument('--data-dir', default='results/e0/baker_sets/bi_train782_normscale')
    ap.add_argument('--variant', default='paper')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--out', default='results/e0/baker_incremental')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        run(a.run_json, a.data_dir, a.out, a.variant, a.n_boot)