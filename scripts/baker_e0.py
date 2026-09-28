"""
scripts/baker_e0.py — ANALYSIS_PLAN.md E0 for Baker et al. (arXiv:2210.16438).

E0.1 Reproduce the bivariate QVR with the authors' configuration:
       2 qubits, RX AngleEmbedding (one channel per qubit), 3 StronglyEntanglingLayers,
       k = 2 diagonal D, N_E = 10, observable = mean Z, Powell on 10 series × 10
       timepoints per function evaluation.
     'notebook' variant (QVR_example.ipynb): τ = 15, cost squared OUTSIDE the draw mean,
       fresh draws per timepoint, one Powell run with maxfev = 500; seeds 0..S−1.
     'paper' variant (Eqs. 4–8, SM pseudocode): τ = 5, cost squared INSIDE the draw mean,
       one draw per series shared over timepoints, R restarts × maxfev 2000,
       best restart by validation balanced accuracy.
     Scoring as the notebook: every 4th timepoint (45) with t = linspace(0.1, 2π, 45);
     score = raw time-series cost (notebook) and a_X (paper Eq. 9).
E0.2 Prespecified baselines, fit on Xtr only (no labels):
       vol_mean      |median-centred mean of channel 1| / MAD
       chan_mean_sd  Mahalanobis on per-channel mean and SD
       trig_moments  Mahalanobis on per-channel mean cos(kx), sin(kx), k = 1, 2
E0.4 Two protocols for every method:
       paper  threshold scanned on Xte_norm vs Xval, then tested on Xte_norm vs each set
              (the same normals are used for tuning and testing, as published)
       clean  AUC with bootstrap 95% CI, and 2-fold balanced accuracy: threshold on half the
              normals + Xval, tested on the other half + the set (and swapped)
E0.3 Mechanism on the reproduced models: D modes, time permutation (common random
       numbers), and an exact function-class check of the per-timepoint term.

Usage:
    python scripts/baker_e0.py --out results/e0/baker            # full run
    python scripts/baker_e0.py --restarts 10 --seeds 3 --out ...   # quicker look
    python scripts/baker_e0.py --self-test
"""

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C                 # noqa: E402
from qvr.data import roc_auc                 # noqa: E402
from qvr.datasets import baker               # noqa: E402
from qvr.infer import infer, permute_time    # noqa: E402
from qvr.train import git_state              # noqa: E402

T_TRAIN = torch.linspace(0.1, 2 * math.pi, 180, dtype=torch.float64)
STRIDE = 4
NOTEBOOK_TARGET = {'val_ba': 0.675, 'dirty_usdt_pos_ba': 0.7167, 'dirty_usdt_pos_f1': 0.7606}
PAPER_TARGET = {'val_ba': 0.82}


_BASE = dict(n_qubits=2, n_layers=3, k=2, N_E=10, sigma_lambda=1.0, ansatz='sel')
_NB = dict(tau=15.0, cost='outside', draw_mode='mc', embedding='rx_per_qubit')
_PAPER = dict(tau=5.0, cost='inside', draw_mode='shared', embedding='rx_per_qubit')

# name → (cfg overrides, n_runs key, maxfev, maxiter, selection score)
# 'notebook' and 'paper' are E0.1 as prespecified. The last three are the prespecified
# reproduction diagnostics for the paper's 0.82 (ANALYSIS_PLAN §11, 2026-09-25):
#   paper_ry    paper maths with RY embedding (paper text) instead of RX (notebook code)
#   paper_iter  paper maths, Powell capped by iterations (2000) with a 20,000-evaluation budget
#   nb_tau5     notebook maths (square outside, per-timepoint draws) with paper settings (τ=5, 2000 evals)
VARIANTS = {
    'notebook':   ({**_NB},                              'seeds',    500,    None, 'raw'),
    'paper':      ({**_PAPER},                           'restarts', 2000,   None, 'a_x'),
    'paper_ry':   ({**_PAPER, 'embedding': 'ry_per_qubit'}, 'restarts', 2000, None, 'a_x'),
    'paper_iter': ({**_PAPER},                           'iter_restarts', 20000, 2000, 'a_x'),
    'nb_tau5':    ({**_NB, 'tau': 5.0},                  'restarts', 2000,   None, 'a_x'),
}


def variant_cfg(name: str) -> C.QVRConfig:
    return C.QVRConfig(**_BASE, **VARIANTS[name][0])


# ---------------------------------------------------------------------------
# Training (Powell, stochastic minibatch objective — as the notebook)
# ---------------------------------------------------------------------------

def _unflat(theta, cfg):
    a = cfg.n_layers * cfg.n_qubits * 3
    th = torch.as_tensor(theta, dtype=torch.float64)
    return {'alpha': th[:a], 'mu': th[a:a + cfg.n_terms],
            'sigma': th[a + cfg.n_terms:a + 2 * cfg.n_terms], 'eta_0': th[-1]}


def train_powell(Xtr, cfg, seed, maxfev, n_series=10, n_t=10, maxiter=None):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    p0 = C.init_params(cfg, rng)
    theta0 = np.concatenate([p0[k].detach().numpy().ravel() for k in ('alpha', 'mu', 'sigma', 'eta_0')])
    N, T = Xtr.shape[:2]
    state = {'s': [], 't': []}

    def next_idx(key, n, size):          # shuffled cycling, like the notebook DataGetter
        if len(state[key]) < size:
            state[key] = list(rng.permutation(n)) + state[key]
        out, state[key] = state[key][-size:], state[key][:-size]
        return np.array(out)

    hist = []

    def f(theta):
        p = _unflat(theta, cfg)
        si, ti = next_idx('s', N, n_series), next_idx('t', T, n_t)
        with torch.no_grad():
            sc, _ = C.series_score(p, Xtr[si][:, ti], T_TRAIN[ti], cfg, mode=cfg.draw_mode, generator=gen)
            loss = float(0.5 * sc.mean() + C.arctan_penalty(p, cfg))
        hist.append(loss)
        return loss

    res = minimize(f, theta0, method='Powell', options={'maxfev': maxfev, 'maxiter': maxiter or maxfev})
    return _unflat(res.x, cfg), hist


# ---------------------------------------------------------------------------
# Scoring and protocols
# ---------------------------------------------------------------------------

def strided(X):
    return X[:, ::STRIDE]


def qvr_scores(params, cfg, X, hist, seed=0, mode=None):
    Xs = strided(X)
    t = torch.linspace(0.1, 2 * math.pi, Xs.shape[1], dtype=torch.float64)
    raw = infer(params, cfg, Xs, t, mode=mode or cfg.draw_mode, seed=seed)['score']
    pen = float(C.arctan_penalty(params, cfg))
    a_x = np.abs(2 * min(hist) - 2 * pen - raw)
    return {'raw': raw, 'a_x': a_x}


def _ba(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    return 0.5 * ((pred[y]).mean() + (~pred[~y]).mean())


def _f1(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def best_threshold(s_norm, s_anom, grid=None):
    """Threshold maximizing balanced accuracy (anomalous if score ≥ ζ); first maximum."""
    if grid is None:
        u = np.unique(np.r_[s_norm, s_anom])
        grid = np.r_[u[0] - 1, (u[:-1] + u[1:]) / 2, u[-1] + 1]
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
    s = np.r_[s_norm, s_anom]
    accs = [_ba(y, s >= z) for z in grid]
    i = int(np.argmax(accs))
    return float(grid[i]), float(accs[i])


def boot_auc(s_norm, s_anom, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.r_[np.zeros(len(s_norm)), np.ones(len(s_anom))]
    a = [roc_auc(y, np.r_[rng.choice(s_norm, len(s_norm)), rng.choice(s_anom, len(s_anom))])
         for _ in range(n)]
    return [float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))]


def evaluate(scores: dict, sets: list, paper_grid=None, n_boot=2000):
    """scores: {'Xte_norm', 'Xval', <sets>} → per-set paper and clean metrics."""
    zeta, val_ba = best_threshold(scores['Xte_norm'], scores['Xval'], paper_grid)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(scores['Xte_norm']))
    halves = (idx[: len(idx) // 2], idx[len(idx) // 2:])
    out = {'val_ba': val_ba, 'zeta': zeta, 'sets': {}}
    for s in sets:
        n, a = scores['Xte_norm'], scores[s]
        y = np.r_[np.zeros(len(n)), np.ones(len(a))]
        pred = np.r_[n, a] >= zeta
        clean = []
        for tune, test in (halves, halves[::-1]):
            z, _ = best_threshold(n[tune], scores['Xval'])
            yy = np.r_[np.zeros(len(test)), np.ones(len(a))]
            clean.append(_ba(yy, np.r_[n[test], a] >= z))
        out['sets'][s] = {'paper_ba': float(_ba(y, pred)), 'paper_f1': _f1(y, pred),
                          'auc': roc_auc(y, np.r_[n, a]), 'auc_ci': boot_auc(n, a, n_boot),
                          'clean_ba': float(np.mean(clean))}
    return out


# ---------------------------------------------------------------------------
# Baselines (fit on Xtr only)
# ---------------------------------------------------------------------------

def _feat(X, kind):
    Xs = strided(X)
    if kind == 'vol_mean':
        return Xs[..., 1].mean(1, keepdim=True).numpy()
    if kind == 'chan_mean_sd':
        return torch.cat([Xs.mean(1), Xs.std(1)], 1).numpy()
    if kind == 'trig_moments':
        return torch.cat([f(k * Xs).mean(1) for k in (1, 2) for f in (torch.cos, torch.sin)], 1).numpy()
    raise ValueError(kind)


BASELINES = ('vol_mean', 'chan_mean_sd', 'trig_moments')


def baseline_scores(data, kind):
    F = {k: _feat(v, kind) for k, v in data.items()}
    tr = F['Xtr']
    if kind == 'vol_mean':
        med = np.median(tr[:, 0])
        mad = np.median(np.abs(tr[:, 0] - med)) or 1.0
        return {k: np.abs(v[:, 0] - med) / mad for k, v in F.items()}
    mu = tr.mean(0)
    cov = np.cov(tr, rowvar=False) + 1e-6 * np.eye(tr.shape[1])
    P = np.linalg.inv(cov)
    return {k: np.sqrt(np.einsum('ij,jk,ik->i', v - mu, P, v - mu)) for k, v in F.items()}


# ---------------------------------------------------------------------------
# Mechanism
# ---------------------------------------------------------------------------

def mechanism(params, cfg, data, sets, hist):
    res = {'d_modes': {}, 'permutation': {}}
    for mode in ('mc', 'shared', 'mean', 'identity'):
        sc = {k: qvr_scores(params, cfg, data[k], hist, mode=mode)['raw'] for k in ['Xte_norm'] + sets}
        res['d_modes'][mode] = {s: roc_auc(np.r_[np.zeros(len(sc['Xte_norm'])), np.ones(len(sc[s]))],
                                           np.r_[sc['Xte_norm'], sc[s]]) for s in sets}
    orig = {k: qvr_scores(params, cfg, data[k], hist)['raw'] for k in ['Xte_norm'] + sets}
    perm = {k: qvr_scores(params, cfg, permute_time(data[k], seed=7, keep_first=False), hist)['raw']
            for k in ['Xte_norm'] + sets}
    for s in sets:
        y = np.r_[np.zeros(len(orig['Xte_norm'])), np.ones(len(orig[s]))]
        a0, a1 = roc_auc(y, np.r_[orig['Xte_norm'], orig[s]]), roc_auc(y, np.r_[perm['Xte_norm'], perm[s]])
        r = float(np.corrcoef(np.r_[orig['Xte_norm'], orig[s]], np.r_[perm['Xte_norm'], perm[s]])[0, 1])
        res['permutation'][s] = {'auc': a0, 'auc_permuted': a1, 'delta': a0 - a1, 'shuf_r': r}
    # exact function class of the per-timepoint term (D = μ, t fixed): degree ≤ 2 per channel
    if cfg.n_qubits != 2:
        res['function_class_max_resid'] = float('nan')     # grid check implemented for 2 channels
        return res
    g = torch.linspace(-math.pi, math.pi, 41, dtype=torch.float64)
    x1, x2 = torch.meshgrid(g, g, indexing='ij')
    xg = torch.stack([x1.ravel(), x2.ravel()], -1).unsqueeze(1)          # (G, 1, 2)
    tt = torch.tensor([2.0], dtype=torch.float64)
    with torch.no_grad():
        v, _ = C.series_score(params, xg, tt, cfg, mode='mean')
    basis = lambda u: [torch.ones_like(u), torch.cos(u), torch.sin(u), torch.cos(2 * u), torch.sin(2 * u)]  # noqa: E731
    Fm = torch.stack([a * b for a in basis(xg[:, 0, 0]) for b in basis(xg[:, 0, 1])], 1).numpy()
    coef = np.linalg.lstsq(Fm, v.numpy(), rcond=None)[0]
    res['function_class_max_resid'] = float(np.abs(Fm @ coef - v.numpy()).max())
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, out, seeds=5, restarts=50, maxfev_nb=500, maxfev_paper=2000, n_boot=2000,
        variants=('notebook', 'paper'), iter_restarts=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = baker.load_all(data_dir)
    sets = baker.anomaly_sets(data)
    grid = np.linspace(0, 1, 10000)
    print('=' * 72)
    print('E0 Baker audit — ' + ', '.join(f'{k} {tuple(v.shape)}' for k, v in data.items()))
    print('=' * 72)
    report = {'git': git_state(), 'sets': sets, 'targets': {'notebook': NOTEBOOK_TARGET, 'paper': PAPER_TARGET},
              'qvr': {}, 'baselines': {}, 'mechanism': {}}

    for kind in BASELINES:
        sc = baseline_scores(data, kind)
        report['baselines'][kind] = evaluate(sc, sets, None, n_boot)

    for var in variants:
        d = data['Xtr'].shape[2]                       # 2 = bivariate, 3 = trivariate
        cfg = dataclasses.replace(variant_cfg(var), n_qubits=d, k=d)
        _, nkey, maxfev, maxiter, score_key = VARIANTS[var]
        n_runs = {'seeds': seeds, 'restarts': restarts, 'iter_restarts': iter_restarts}[nkey]
        if var == 'notebook':
            maxfev = maxfev_nb
        elif maxiter is None:
            maxfev = maxfev_paper
        runs = []
        t0 = time.time()
        for s in range(n_runs):
            params, hist = train_powell(data['Xtr'], cfg, seed=s, maxfev=maxfev, maxiter=maxiter)
            sc = {k: qvr_scores(params, cfg, data[k], hist, seed=0) for k in ['Xte_norm', 'Xval'] + sets}
            ev = {sk: evaluate({k: v[sk] for k, v in sc.items()}, sets, grid, n_boot) for sk in ('raw', 'a_x')}
            runs.append({'seed': s, 'final_loss': hist[-1], 'min_loss': min(hist), 'eval': ev,
                         'params': {k: v.tolist() for k, v in params.items()}, 'hist': hist})
            print(f"  {var:8s} run {s:2d}: val BA raw {ev['raw']['val_ba']:.3f} / a_x {ev['a_x']['val_ba']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        best = int(np.argmax([r['eval'][score_key]['val_ba'] for r in runs]))
        report['qvr'][var] = {'cfg': cfg.to_dict(), 'maxfev': maxfev, 'maxiter': maxiter,
                              'n_fev_mean': float(np.mean([len(r['hist']) for r in runs])),
                              'score_key': score_key,
                              'best_run': best, 'runs': runs}
        bp = {k: torch.tensor(v, dtype=torch.float64) for k, v in runs[best]['params'].items()}
        report['mechanism'][var] = mechanism(bp, cfg, data, sets, runs[best]['hist'])

    (out / 'baker_e0.json').write_text(json.dumps(report, indent=1))
    _summary(report, sets)
    print(f'saved → {out}/baker_e0.json')
    return report


def _summary(rep, sets):
    print('\n' + '=' * 72)
    print('SUMMARY — per anomaly set: paper BA / F1 | clean AUC [95% CI] | clean BA')
    rows = []
    for var, q in rep['qvr'].items():
        r = q['runs'][q['best_run']]['eval'][q['score_key']]
        rows.append((f"QVR {var} (best of {len(q['runs'])}, {q['score_key']})", r))
        vb = [x['eval'][q['score_key']]['val_ba'] for x in q['runs']]
        aucs = {s: [x['eval'][q['score_key']]['sets'][s]['auc'] for x in q['runs']] for s in sets}
        print(f"  QVR {var:10s} {len(q['runs'])} runs (~{q['n_fev_mean']:.0f} evals each): val BA best "
              f"{max(vb):.3f}, median {np.median(vb):.3f} | median AUC " +
              ', '.join(f"{s[4:]} {np.median(v):.3f}" for s, v in aucs.items()))
    for k, r in rep['baselines'].items():
        rows.append((f'baseline {k}', r))
    for name, r in rows:
        print(f"\n  {name}   val BA {r['val_ba']:.3f}")
        for s in sets:
            m = r['sets'][s]
            print(f"    {s[4:]:18s} paper {m['paper_ba']:.3f}/{m['paper_f1']:.3f} | "
                  f"AUC {m['auc']:.3f} [{m['auc_ci'][0]:.3f}, {m['auc_ci'][1]:.3f}] | clean BA {m['clean_ba']:.3f}")
    for var, m in rep['mechanism'].items():
        print(f"\n  mechanism ({var}): function-class residual {m['function_class_max_resid']:.1e} "
              f"(n_qubits {rep['qvr'][var]['cfg']['n_qubits']})")
        for s in sets:
            d = m['d_modes']
            print(f"    {s[4:]:18s} AUC mc {d['mc'][s]:.3f} shared {d['shared'][s]:.3f} mean {d['mean'][s]:.3f} "
                  f"identity {d['identity'][s]:.3f} | permuted {m['permutation'][s]['auc_permuted']:.3f} "
                  f"(shuf_r {m['permutation'][s]['shuf_r']:.3f})")
    print(f"\n  targets: notebook {rep['targets']['notebook']}; paper {rep['targets']['paper']}")


def _self_test():
    import pickle
    import tempfile
    print('baker_e0 self-test: fake pickles, tiny training')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = rng.normal(0, 0.6, (1, 2, 180))
        def mk(n, shift):
            return np.clip(base + rng.normal(0, 0.4, (n, 2, 180)) + shift, -np.pi, np.pi)
        for name, n, sh in (('Xtr', 60, 0), ('Xval', 30, 0.8), ('Xte_norm', 30, 0), ('Xte_dirty_usdt_pos', 30, 1.0)):
            with open(Path(tmp) / f'{name}.pickle', 'wb') as f:
                pickle.dump(mk(n, sh), f)
        rep = run(tmp, Path(tmp) / 'out', seeds=2, restarts=2, maxfev_nb=40, maxfev_paper=40, n_boot=50)
        VARIANTS['paper_iter'] = (VARIANTS['paper_iter'][0], 'iter_restarts', 60, 3, 'a_x')
        rep2 = run(tmp, Path(tmp) / 'out2', restarts=2, maxfev_paper=40, n_boot=50, iter_restarts=1,
                   variants=('paper_ry', 'paper_iter', 'nb_tau5'))
        checks = {
            'both variants trained': set(rep['qvr']) == {'notebook', 'paper'},
            'baselines evaluated': set(rep['baselines']) == set(BASELINES),
            'shifted anomalies detected by trig_moments (AUC > 0.9)':
                rep['baselines']['trig_moments']['sets']['Xte_dirty_usdt_pos']['auc'] > 0.9,
            'function class exact (residual < 1e-10)':
                all(m['function_class_max_resid'] < 1e-10 for m in rep['mechanism'].values()),
            'qubits follow channel count': rep['qvr']['paper']['cfg']['n_qubits'] == 2,
            'json written': (Path(tmp) / 'out' / 'baker_e0.json').exists(),
            'diagnostic variants run': set(rep2['qvr']) == {'paper_ry', 'paper_iter', 'nb_tau5'},
            'paper_ry uses RY embedding': rep2['qvr']['paper_ry']['cfg']['embedding'] == 'ry_per_qubit',
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=str(baker.DATA_DIR))
    ap.add_argument('--out', default='results/e0/baker')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--restarts', type=int, default=50)
    ap.add_argument('--maxfev-notebook', type=int, default=500)
    ap.add_argument('--maxfev-paper', type=int, default=2000)
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--variants', nargs='+', default=['notebook', 'paper'], choices=list(VARIANTS))
    ap.add_argument('--iter-restarts', type=int, default=10)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        run(a.data_dir, a.out, a.seeds, a.restarts, a.maxfev_notebook, a.maxfev_paper, a.n_boot,
            a.variants, a.iter_restarts)