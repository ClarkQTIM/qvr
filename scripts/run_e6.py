"""
scripts/run_e6.py — E6 (ANALYSIS_PLAN §12.2–12.3, EXPLORATORY): does a Hamiltonian that
carries the series' history help? Runs on any split directory (univariate).

Trained models (seeds --seeds, same budget as §2: Adam, batch 8, ≤ --max-samples,
monitor early stopping):
  sqvr     sequential Hamiltonian-memory model (qvr/sequential.py), NLL score
  gru4     GRU, hidden 4 (≈ 120 parameters, comparable to sQVR's 43), 8-bin softmax, NLL score
  gru32    GRU, hidden 32 (strong small classical sequence model), NLL score
  wqvr     QVR with windowed embedding (x_t, x_{t−1}, x_{t−2} on the three qubits), fixed
           model otherwise (§2); scored natively and by per-timepoint z² of ⟨Z_q⟩_t (Test A)
Untrained comparators (fit on Xtr only):
  hist     memoryless per-timepoint tail-bin histogram (−log mass)
  tz2      per-timepoint z² of the raw series
  lag1     lag-1 autocorrelation, robust z against training normals (two-sided)
  msd      mean squared first difference, robust z (two-sided)

Reported: AUC per anomaly set and pooled (seed-rank-averaged, 95% unit CI); mean NLL on
test normals for sqvr / gru4 / gru32 / hist; sQVR order sensitivity (shuf_r, ΔAUC under
time permutation); score-level growing-vs-falling AUC when those sets exist.

Usage:
    python scripts/run_e6.py --data-dir data_splits/memory --out results/e6/memory --workers 8
    python scripts/run_e6.py --self-test
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qvr import circuit as C                                          # noqa: E402
from qvr import sequential as S                                       # noqa: E402
from qvr.data import roc_auc, t_grid                                  # noqa: E402
from qvr.infer import infer, permute_time                             # noqa: E402
from run_suite import _rank_avg, _unit_boot, anomaly_sets, load_split  # noqa: E402

TRAINED = ('sqvr', 'gru4', 'gru32', 'wqvr')
UNTRAINED = ('hist', 'tz2', 'lag1', 'msd')


def lagged(X):
    """(N, T) → (N, T, 3): x_t, x_{t−1}, x_{t−2}; the first values are repeated at the start."""
    x1 = torch.cat([X[:, :1], X[:, :-1]], 1)
    x2 = torch.cat([x1[:, :1], x1[:, :-1]], 1)
    return torch.stack([X, x1, x2], -1)


def wqvr_cfg():
    return C.QVRConfig(n_qubits=3, n_layers=3, N_E=10, tau=1.0, sigma_lambda=1e-2,
                       embedding='ry_per_qubit', ansatz='chain', cost='outside', draw_mode='mc')


# ---------------------------------------------------------------------------
# GRU next-bin model
# ---------------------------------------------------------------------------

class GRUBins(nn.Module):
    def __init__(self, hidden, n_bins=8):
        super().__init__()
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_bins)

    def forward(self, X):                       # predict bin(x_t) from x_{<t}; zero start token
        inp = torch.cat([torch.zeros(len(X), 1), X[:, :-1]], 1).unsqueeze(-1).float()
        h, _ = self.gru(inp)
        return torch.log_softmax(self.head(h), -1)          # (B, T, n_bins)


def gru_nll(model, X, edges):
    return -model(X).gather(-1, S.bin_index(X, edges).unsqueeze(-1)).squeeze(-1).double()


def train_gru(X, hidden, seed, out_dir, max_samples, batch=8, lr=0.01, eval_every=50, patience=10):
    from qvr.train import git_state
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    edges = S.fit_edges(X, S.SeqConfig())
    model = GRUBins(hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    perm = rng.permutation(len(X))
    M = min(256, max(1, len(X) // 5))
    mon, pool = X[perm[:M]], perm[M:]
    best, best_m, bad, nb, ns = None, math.inf, 0, 0, 0
    t0 = time.time()
    while ns + batch <= max_samples:
        idx = rng.choice(pool, batch, replace=False)
        opt.zero_grad()
        loss = gru_nll(model, X[idx], edges).mean()
        loss.backward()
        opt.step()
        nb, ns = nb + 1, ns + batch
        if nb % eval_every == 0:
            with torch.no_grad():
                m = float(gru_nll(model, mon, edges).mean())
            if m < best_m - 1e-6:
                best_m, bad, best = m, 0, {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
    if best is not None:
        model.load_state_dict(best)
    torch.save({'state': model.state_dict(), 'hidden': hidden, 'edges': edges, 'seed': seed}, out_dir / 'params.pt')
    (out_dir / 'manifest.json').write_text(json.dumps({'model': f'gru{hidden}', 'seed': seed, 'git': git_state(),
                                                      'n_params': sum(p.numel() for p in model.parameters()),
                                                      'best_monitor': best_m, 'n_batches': nb,
                                                      'elapsed_s': time.time() - t0}, indent=1))


# ---------------------------------------------------------------------------
# Train / score workers
# ---------------------------------------------------------------------------

def _train(args):
    method, data_dir, out, channel, seed, max_samples, strict = args
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    from qvr.train import git_state, train
    g = git_state()
    if strict and (g['commit'] is None or g['dirty']):
        raise RuntimeError(f'refusing to train: git state {g}')
    run_dir = Path(out) / 'models' / f'{method}_s{seed}'
    if (run_dir / 'params.pt').exists():
        return f'{method}_s{seed} (exists)'
    X, *_ = load_split(data_dir, channel)
    if method == 'sqvr':
        S.train_seq(X['Xtr'], S.SeqConfig(), seed, run_dir, max_samples=max_samples, strict_git=strict)
    elif method.startswith('gru'):
        train_gru(X['Xtr'], int(method[3:]), seed, run_dir, max_samples)
    elif method == 'wqvr':
        train(lagged(X['Xtr']), t_grid(X['Xtr'].shape[1]), wqvr_cfg(), run_seed=seed, out_dir=run_dir,
              max_samples=max_samples, strict_git=strict, verbose=False)
    return f'{method}_s{seed}'


def _score(args):
    method, data_dir, out, channel, seed = args
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    path = Path(out) / 'scores' / f'{method}_s{seed}.npz'
    if path.exists():
        return f'{method}_s{seed} (exists)'
    X, *_ = load_split(data_dir, channel)
    run_dir = Path(out) / 'models' / f'{method}_s{seed}'
    keys = ['Xte_norm'] + anomaly_sets(X)
    res = {}
    if method == 'sqvr':
        p, cfg, edges, _ = S.load_seq(run_dir)
        for k in keys:
            res[f'score__{k}'] = S.score(p, X[k], edges, cfg)
            res[f'perm__{k}'] = S.score(p, permute_time(X[k], 7, keep_first=False), edges, cfg)
    elif method.startswith('gru'):
        ck = torch.load(run_dir / 'params.pt', weights_only=False)
        m = GRUBins(ck['hidden'])
        m.load_state_dict(ck['state'])
        with torch.no_grad():
            for k in keys:
                res[f'score__{k}'] = gru_nll(m, X[k], ck['edges']).mean(1).numpy()
    elif method == 'wqvr':
        from qvr.train import load_run
        p, cfg, _ = load_run(run_dir)
        T = X['Xtr'].shape[1]
        t = t_grid(T)
        rng = np.random.default_rng(0)
        fit = torch.as_tensor(np.sort(rng.choice(len(X['Xtr']), min(5000, len(X['Xtr'])), replace=False)))
        o = infer(p, cfg, lagged(X['Xtr'][fit]), t, mode='mc', seed=0, keep_time=True)
        mu, sd = o['z_t'].mean(0), o['z_t'].std(0) + 1e-9
        for k in keys:
            o = infer(p, cfg, lagged(X[k]), t, mode='mc', seed=0, keep_time=True)
            res[f'score__{k}'] = o['score']
            res[f'tz2__{k}'] = (((o['z_t'] - mu) / sd) ** 2).mean((1, 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return f'{method}_s{seed}'


def _pool(fn, jobs, workers):
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            print(f'  {fn.__name__[1:]}: {fn(j)}  ({time.time() - t0:.0f}s)', flush=True)
    else:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(fn, jobs):
                print(f'  {fn.__name__[1:]}: {r}  ({time.time() - t0:.0f}s)', flush=True)


# ---------------------------------------------------------------------------
# Untrained comparators
# ---------------------------------------------------------------------------

def _robust_z(train, x):
    med = np.median(train)
    mad = np.median(np.abs(train - med)) or 1.0
    return np.abs(x - med) / mad


def untrained_scores(X):
    tr = X['Xtr'].numpy()
    keys = ['Xte_norm'] + anomaly_sets(X)
    cfg = S.SeqConfig()
    edges = S.fit_edges(X['Xtr'], cfg)
    mu, sd = tr.mean(0), tr.std(0) + 1e-9

    def lag1(A):
        a, b = A[:, :-1] - A[:, :-1].mean(1, keepdims=True), A[:, 1:] - A[:, 1:].mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1) + 1e-12)

    msd = lambda A: (np.diff(A, axis=1) ** 2).mean(1)  # noqa: E731
    out = {m: {} for m in UNTRAINED}
    for k in keys:
        A = X[k].numpy()
        out['hist'][k] = S.histogram_nll(X[k], edges, cfg).numpy()
        out['tz2'][k] = (((A - mu) / sd) ** 2).mean(1)
        out['lag1'][k] = _robust_z(lag1(tr), lag1(A))
        out['msd'][k] = _robust_z(msd(tr), msd(A))
    return out


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze(data_dir, out, channel, seeds, methods, n_boot):
    X, units, sub, man = load_split(data_dir, channel)
    sets = anomaly_sets(X)
    groups = {s: [s] for s in sets}
    groups['pooled'] = sets
    U = untrained_scores(X)
    scores = {}                                     # name → list over seeds of {set: score}
    for m in methods:
        Z = [dict(np.load(Path(out) / 'scores' / f'{m}_s{s}.npz')) for s in seeds]
        scores[m] = [{k: z[f'score__{k}'] for k in ['Xte_norm'] + sets} for z in Z]
        if m == 'wqvr':
            scores['wqvr_tz2'] = [{k: z[f'tz2__{k}'] for k in ['Xte_norm'] + sets} for z in Z]
        if m == 'sqvr':
            scores['sqvr_perm'] = [{k: z[f'perm__{k}'] for k in ['Xte_norm'] + sets} for z in Z]
    for m in UNTRAINED:
        scores[m] = [U[m]]
    rep = {'exploratory': True, 'data_dir': str(data_dir), 'seeds': seeds, 'auc': {}, 'nll_test_normals': {}}
    for g, names in groups.items():
        y = np.r_[np.zeros(len(X['Xte_norm'])), np.ones(sum(len(X[k]) for k in names))]
        u = np.r_[units['Xte_norm'], np.concatenate([units[k] for k in names])]
        rep['auc'][g] = {}
        for name, per in scores.items():
            vecs = [np.r_[d['Xte_norm'], np.concatenate([d[k] for k in names])] for d in per]
            avg = _rank_avg(vecs) if len(vecs) > 1 else vecs[0]
            rep['auc'][g][name] = {'auc': roc_auc(y, avg), 'ci': _unit_boot(y, avg, u, n_boot),
                                   'auc_seeds': [roc_auc(y, v) for v in vecs]}
    for name in ('sqvr', 'gru4', 'gru32', 'hist'):
        if name in scores:
            rep['nll_test_normals'][name] = float(np.mean([d['Xte_norm'].mean() for d in scores[name]]))
    if 'sqvr' in scores:
        a = [np.concatenate([d[k] for k in ['Xte_norm'] + sets]) for d in scores['sqvr']]
        b = [np.concatenate([d[k] for k in ['Xte_norm'] + sets]) for d in scores['sqvr_perm']]
        rep['sqvr_order'] = {'shuf_r': float(np.mean([np.corrcoef(x, z)[0, 1] for x, z in zip(a, b)])),
                             'dAUC_shuf': rep['auc']['pooled']['sqvr']['auc'] - rep['auc']['pooled']['sqvr_perm']['auc']}
    if {'Xte_growing', 'Xte_falling'} <= set(X):
        y = np.r_[np.zeros(len(X['Xte_growing'])), np.ones(len(X['Xte_falling']))]
        rep['growing_vs_falling'] = {
            name: roc_auc(y, _rank_avg([np.r_[d['Xte_growing'], d['Xte_falling']] for d in per]) if len(per) > 1
                          else np.r_[per[0]['Xte_growing'], per[0]['Xte_falling']])
            for name, per in scores.items() if not name.endswith('_perm')}
    (Path(out) / 'e6_summary.json').write_text(json.dumps(rep, indent=1, default=float))
    md = _markdown(rep, groups)
    (Path(out) / 'e6_summary.md').write_text(md)
    print(md)
    return rep


def _markdown(rep, groups):
    names = [n for n in ('sqvr', 'gru4', 'gru32', 'wqvr', 'wqvr_tz2', 'hist', 'tz2', 'lag1', 'msd')
             if n in rep['auc']['pooled']]
    L = [f"# E6 (EXPLORATORY) — {Path(rep['data_dir']).name}", '',
         'AUC (seed-rank-averaged for trained models) with 95% unit CI', '',
         '| set | ' + ' | '.join(names) + ' |', '| --- |' + ' --- |' * len(names)]
    for g in groups:
        L.append(f"| {g.replace('Xte_', '')} | " + ' | '.join(
            f"{rep['auc'][g][n]['auc']:.3f} [{rep['auc'][g][n]['ci'][0]:.2f}, {rep['auc'][g][n]['ci'][1]:.2f}]" for n in names) + ' |')
    if rep['nll_test_normals']:
        L += ['', 'Mean NLL on test normals (lower = better prediction; memoryless histogram = no-memory reference): '
              + ', '.join(f'{k} {v:.3f}' for k, v in rep['nll_test_normals'].items())]
    if 'sqvr_order' in rep:
        o = rep['sqvr_order']
        L += ['', f"sQVR order sensitivity: shuf_r {o['shuf_r']:.3f}, ΔAUC_shuf (pooled) {o['dAUC_shuf']:+.3f}"]
    if 'growing_vs_falling' in rep:
        L += ['', 'Score-level growing vs falling AUC: ' + ', '.join(f'{k} {v:.3f}' for k, v in rep['growing_vs_falling'].items())]
    return '\n'.join(L) + '\n'


def run(data_dir, out, seeds, methods, stages, workers=1, channel=0, max_samples=50_000, n_boot=1000, strict_git=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'data_dir': str(data_dir), 'seeds': seeds, 'methods': methods,
                                                 'max_samples': max_samples}, indent=1))
    jobs = [(m, s) for m in methods for s in seeds]
    if 'train' in stages:
        print(f'== train: {len(jobs)} models')
        _pool(_train, [(m, str(data_dir), str(out), channel, s, max_samples, strict_git) for m, s in jobs], workers)
    if 'score' in stages:
        print(f'== score: {len(jobs)} models')
        _pool(_score, [(m, str(data_dir), str(out), channel, s) for m, s in jobs], workers)
    if 'analyze' in stages:
        print('== analyze')
        return analyze(data_dir, out, channel, seeds, methods, n_boot)


def _self_test():
    import tempfile
    from prepare_dataset import MEM_SPLITS, prepare_memory
    print('run_e6 self-test: tiny memory dataset, all methods, tiny budgets')
    MEM_SPLITS['normal'].update({'Xtr': 300, 'Xval_norm': 40, 'Xte_norm': 80})
    MEM_SPLITS['anomaly'].update({'Xval': 20, 'Xprobe': 30, 'Xte': 60})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prepare_memory(Path(tmp) / 'mem', seq_len=16)
        rep = run(Path(tmp) / 'mem', Path(tmp) / 'e6', seeds=[0, 1], methods=list(TRAINED),
                  stages=('train', 'score', 'analyze'), max_samples=400, n_boot=50)
        a = rep['auc']
        checks = {
            'all methods scored': set(TRAINED + UNTRAINED + ('wqvr_tz2',)) <= set(a['pooled']),
            'per-timepoint models blind on memory data (hist, tz2 within 0.5 ± 0.15)':
                all(abs(a['pooled'][m]['auc'] - 0.5) < 0.15 for m in ('hist', 'tz2')),
            'classical lag statistic detects the anti class (lag1 > 0.9)': a['Xte_anti']['lag1']['auc'] > 0.9,
            'NLL reported for sqvr / gru / hist': {'sqvr', 'gru4', 'gru32', 'hist'} <= set(rep['nll_test_normals']),
            'sQVR order sensitivity computed': 'sqvr_order' in rep,
            'summary written': (Path(tmp) / 'e6' / 'e6_summary.md').exists(),
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
    ap.add_argument('--out')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--methods', nargs='+', default=list(TRAINED), choices=list(TRAINED))
    ap.add_argument('--stages', nargs='+', default=['train', 'score', 'analyze'], choices=['train', 'score', 'analyze'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--max-samples', type=int, default=50_000)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--strict-git', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not (a.data_dir and a.out):
            ap.error('--data-dir and --out are required')
        run(a.data_dir, a.out, a.seeds, a.methods, a.stages, a.workers, a.channel, a.max_samples, a.n_boot, a.strict_git)