"""
scripts/score_audit.py — what is the QVR score using? Dataset-agnostic.

Reads a split directory in the standard layout (README "Drop-in dataset layout"):
X*.pickle arrays (N, d, T): Xtr (training normals), Xte_norm (test normals),
Xte_<name> (anomalous test sets). QVR scores come either from a baker_e0 run file
(--run-json, best restart) or from a scores file (--scores-npz with one array per
set name, e.g. written by a later pipeline).

All comparators are fit on Xtr only, on the same time grid as the QVR scores
(--stride), and scored two-sided (robust z against training normals):

  prespecified   vol_mean      window mean of the primary channel
                 trig_moments  per-channel mean cos/sin(kx), k = 1, 2 → Mahalanobis
  POST HOC       final_value   last value of the primary channel
  (added after   late_mean     mean of the last third of the window, primary channel
   seeing the    tz2_primary   mean over t of per-timepoint z² (train mean / SD per t),
   Baker          primary channel — same additive-over-time structure as QVR
   results)      tz2_all       the same over all channels

Per anomaly set and pooled over all sets:
  - AUC of QVR and every comparator, and a paired bootstrap CI of AUC(QVR) − AUC(comparator)
  - rank combination: logistic on within-pool ranks of [vol_mean, QVR], out-of-fold,
    Δ vs each component (fixes the heavy-tail problem of combining raw scores)
  - channel knockout (run-json only): each channel replaced by its per-timepoint
    training-normal mean, QVR re-scored

Usage:
    python scripts/score_audit.py --run-json results/e0/baker_fu_bi_train782_normscale/baker_e0.json \
        --data-dir results/e0/baker_sets/bi_train782_normscale --out results/e0/baker_score_audit
    python scripts/score_audit.py --self-test
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
from qvr.data import roc_auc                  # noqa: E402
from qvr.datasets import baker as splitdir    # noqa: E402  (generic X*.pickle loader)

PREREG = ('vol_mean', 'trig_moments')
POSTHOC = ('final_value', 'late_mean', 'tz2_primary', 'tz2_all')


# ---------------------------------------------------------------------------
# Comparators (fit on Xtr only)
# ---------------------------------------------------------------------------

def _robust_z(train, x):
    med = np.median(train)
    mad = np.median(np.abs(train - med)) or 1.0
    return np.abs(x - med) / mad


def _mahal(train, x):
    mu = train.mean(0)
    P = np.linalg.inv(np.cov(train, rowvar=False) + 1e-6 * np.eye(train.shape[1]))
    return np.sqrt(np.einsum('ij,jk,ik->i', x - mu, P, x - mu))


def comparator_scores(data, c, stride):
    """{method: {set: score}}; data values (N, T, d)."""
    D = {k: v[:, ::stride].numpy() for k, v in data.items()}
    tr = D['Xtr']
    T = tr.shape[1]
    late = slice(T - T // 3, T)
    mu_t, sd_t = tr.mean(0), tr.std(0) + 1e-9                     # (T, d)
    out = {}
    f = lambda X: X[..., c].mean(1)                                # noqa: E731
    out['vol_mean'] = {k: _robust_z(f(tr), f(X)) for k, X in D.items()}
    trig = lambda X: np.concatenate([g(k * X).mean(1) for k in (1, 2) for g in (np.cos, np.sin)], 1)  # noqa: E731
    out['trig_moments'] = {k: _mahal(trig(tr), trig(X)) for k, X in D.items()}
    f = lambda X: X[:, -1, c]                                      # noqa: E731
    out['final_value'] = {k: _robust_z(f(tr), f(X)) for k, X in D.items()}
    f = lambda X: X[:, late, c].mean(1)                            # noqa: E731
    out['late_mean'] = {k: _robust_z(f(tr), f(X)) for k, X in D.items()}
    out['tz2_primary'] = {k: (((X[..., c] - mu_t[:, c]) / sd_t[:, c]) ** 2).mean(1) for k, X in D.items()}
    out['tz2_all'] = {k: (((X - mu_t) / sd_t) ** 2).mean((1, 2)) for k, X in D.items()}
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _ranks(v):
    from scipy.stats import rankdata
    return rankdata(v) / len(v)


def rank_combo(a, b, y, reps=5, folds=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    X = np.c_[_ranks(a), _ranks(b)]
    p = np.zeros(len(y))
    for r in range(reps):
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            p[te] += LogisticRegression(max_iter=1000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / reps
    return p


def paired_boot(y, s1, s2, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    d = []
    for _ in range(n_boot):
        ix = np.r_[rng.choice(i0, len(i0)), rng.choice(i1, len(i1))]
        d.append(roc_auc(y[ix], s1[ix]) - roc_auc(y[ix], s2[ix]))
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))] if d else [float('nan')] * 2


# ---------------------------------------------------------------------------
# QVR scores
# ---------------------------------------------------------------------------

def qvr_from_run(run_json, data, variant=None, knockout=None):
    from qvr import circuit as C
    from baker_e0 import qvr_scores
    rep = json.loads(Path(run_json).read_text())
    variant = variant or next(iter(rep['qvr']))
    q = rep['qvr'][variant]
    run = q['runs'][q['best_run']]
    cfg = C.QVRConfig(**q['cfg'])
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in run['params'].items()}
    X = dict(data)
    if knockout is not None:
        mu = data['Xtr'][..., knockout].mean(0)
        for k in X:
            X[k] = X[k].clone()
            X[k][..., knockout] = mu
    return {k: qvr_scores(p, cfg, v, run['hist'])[q['score_key']] for k, v in X.items() if k != 'Xtr'}, variant


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, out, run_json=None, scores_npz=None, variant=None, primary=None, stride=4, n_boot=2000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = splitdir.load_all(data_dir)
    d = data['Xtr'].shape[2]
    c = primary if primary is not None else (1 if d > 1 else 0)
    sets = splitdir.anomaly_sets(data)
    if run_json:
        Q, variant = qvr_from_run(run_json, data, variant)
    else:
        z = np.load(scores_npz)
        Q = {k: z[k] for k in z.files}
    comp = comparator_scores(data, c, stride)
    groups = {s: [s] for s in sets}
    groups['pooled_all'] = sets
    rep = {'data_dir': str(data_dir), 'primary_channel': c, 'stride': stride, 'variant': variant,
           'posthoc_methods': list(POSTHOC), 'sets': {}}

    print('=' * 100)
    print(f'Score audit — {Path(data_dir).name}  (primary channel {c}, stride {stride}, d = {d})')
    print('  post hoc comparators (added after seeing Baker results): ' + ', '.join(POSTHOC))
    print('=' * 100)
    print(f"{'set':18s} {'QVR':>6s} | " + ' '.join(f'{m:>12s}' for m in PREREG + POSTHOC)
          + ' | rank combo (vol_mean + QVR): Δ vs QVR, Δ vs vol_mean')
    for g, names in groups.items():
        y = np.r_[np.zeros(len(Q['Xte_norm'])), np.ones(sum(len(Q[n]) for n in names))]
        cat = lambda S: np.r_[S['Xte_norm'], np.concatenate([S[n] for n in names])]  # noqa: E731
        q = cat(Q)
        r = {'qvr_auc': roc_auc(y, q), 'methods': {}}
        for m in PREREG + POSTHOC:
            s = cat(comp[m])
            r['methods'][m] = {'auc': roc_auc(y, s), 'ci_qvr_minus': paired_boot(y, q, s, n_boot)}
        combo = rank_combo(cat(comp['vol_mean']), q, y)
        r['rank_combo'] = {'auc': roc_auc(y, combo),
                           'ci_combo_minus_qvr': paired_boot(y, combo, q, n_boot),
                           'ci_combo_minus_vol_mean': paired_boot(y, combo, cat(comp['vol_mean']), n_boot)}
        rep['sets'][g] = r
        cells = []
        for m in PREREG + POSTHOC:
            lo, hi = r['methods'][m]['ci_qvr_minus']
            flag = '+' if lo > 0 else ('-' if hi < 0 else ' ')
            cells.append(f"{r['methods'][m]['auc']:.3f}{flag:>1s}".rjust(12))
        rc = r['rank_combo']
        print(f"{g.replace('Xte_', ''):18s} {r['qvr_auc']:6.3f} | " + ' '.join(cells) +
              f" | {rc['auc'] - r['qvr_auc']:+.3f} [{rc['ci_combo_minus_qvr'][0]:+.3f},{rc['ci_combo_minus_qvr'][1]:+.3f}]"
              f", {rc['auc'] - r['methods']['vol_mean']['auc']:+.3f}")
    print("  flag after a comparator AUC: '+' QVR significantly higher, '-' comparator significantly higher "
          '(paired bootstrap 95% CI excludes 0)')

    if run_json:
        rep['knockout'] = {}
        print('\nChannel knockout (channel replaced by its per-timepoint training-normal mean), QVR AUC:')
        for k in range(d):
            Qk, _ = qvr_from_run(run_json, data, variant, knockout=k)
            y = np.r_[np.zeros(len(Qk['Xte_norm'])), np.ones(sum(len(Qk[n]) for n in sets))]
            a = roc_auc(y, np.r_[Qk['Xte_norm'], np.concatenate([Qk[n] for n in sets])])
            rep['knockout'][k] = a
            print(f"  channel {k} removed: pooled AUC {a:.3f}   (intact {rep['sets']['pooled_all']['qvr_auc']:.3f})")
    (out / 'score_audit.json').write_text(json.dumps(rep, indent=1))
    print(f'\nsaved → {out}/score_audit.json')
    return rep


def _self_test():
    import pickle
    import tempfile
    from baker_e0 import run as e0_run
    print('score_audit self-test: fake 2-channel data, signal only late in channel 1')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        def mk(n, late_shift):
            X = rng.normal(-2.0, 0.2, (n, 2, 180))
            X[:, 1, 120:] += late_shift
            return np.clip(X, -np.pi, np.pi)
        for name, n, sh in (('Xtr', 60, 0), ('Xval', 30, 1.0), ('Xte_norm', 30, 0), ('Xte_a', 30, 1.0), ('Xte_b', 30, 0.5)):
            with open(Path(tmp) / f'{name}.pickle', 'wb') as f:
                pickle.dump(mk(n, sh), f)
        e0_run(tmp, Path(tmp) / 'e0', restarts=2, maxfev_paper=40, n_boot=20, variants=('paper',))
        rep = run(tmp, Path(tmp) / 'out', run_json=Path(tmp) / 'e0' / 'baker_e0.json', n_boot=100)
        np.savez(Path(tmp) / 's.npz', **{k: rng.normal(size=30) for k in ('Xte_norm', 'Xte_a', 'Xte_b')})
        rep2 = run(tmp, Path(tmp) / 'out2', scores_npz=Path(tmp) / 's.npz', n_boot=50)
        m = rep['sets']['pooled_all']['methods']
        checks = {
            'late-window twin beats window mean on late-only signal': m['late_mean']['auc'] >= m['vol_mean']['auc'],
            'per-timepoint z² twin detects signal': m['tz2_primary']['auc'] > 0.9,
            'knockout of channel 1 ran for both channels': set(rep['knockout']) == {0, 1},
            'scores-npz path works (random scores ≈ chance)': abs(rep2['sets']['pooled_all']['qvr_auc'] - 0.5) < 0.2,
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
    ap.add_argument('--run-json')
    ap.add_argument('--scores-npz')
    ap.add_argument('--variant')
    ap.add_argument('--primary-channel', type=int)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--out', default='results/score_audit')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not a.data_dir or not (a.run_json or a.scores_npz):
            ap.error('--data-dir and one of --run-json / --scores-npz are required')
        run(a.data_dir, a.out, a.run_json, a.scores_npz, a.variant, a.primary_channel, a.stride, a.n_boot)