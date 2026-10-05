"""
scripts/run_suite.py — the full per-dataset suite (ANALYSIS_PLAN §2–§7) for any split directory.

Fixed model (§2): 3 qubits, 3 layers, single-pass RY(x) embedding on every qubit, CNOT-chain
ansatz, k = 3, cost squared outside the draw mean, fresh draws per timepoint ('mc'),
σ_λ = 1e-2, τ = 1, Adam lr 0.01, batch 8, ≤ 50,000 samples, monitor early stopping.
t = linspace(0, 2π, T). Univariate: multichannel data use one channel (--channel).

Stages (each resumable; outputs under --out):
  train    models/ne{N}_s{S}/  via qvr.train.train, for N_E ∈ --ne × seed ∈ --seeds
  score    scores/ne{N}_s{S}.npz: per set, score under D modes mc / mean / identity / averaged,
           mc score of the time-permuted copy (same draws), p̄_b; inference seed 0
  analyze  summary.json + summary.md:
    E1 detection AUC per anomaly set and pooled; reference N_E: seed-rank-averaged scores
       with 95% bootstrap CI over independent units
    E2 shuf_r and ΔAUC_shuf per model; Spearman(log N_E, shuf_r) over all models
    E3 AUC per D mode; ΔAUC(mean − mc), ΔAUC(mc − identity); unit bootstrap at reference N_E
    E4 subtype probes (logistic C = 0.1 and gradient-boosted trees) on p̄_b and on six Fourier
       moments, fit on Xprobe, evaluated on the pooled test anomalies and on Xval; macro OvR AUC,
       balanced accuracy, κ (unweighted, nominal) for eligible classes only; unit-level
       permutation null (--n-perm) at the reference N_E
    E5 eigenvalues of H = ½Σμ_P P, minimum gap, interaction-order weights S_r, |σ|
  Also writes scores/ref_mc_scores.npz for scripts/score_audit.py (--scores-npz, --stride 1).

Usage:
    python scripts/run_suite.py --data-dir data_splits/synthetic --out results/suite/synthetic --workers 8
    python scripts/run_suite.py --self-test
"""

import argparse
import json
import math
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C                      # noqa: E402
from qvr.data import roc_auc, t_grid              # noqa: E402
from qvr.infer import infer, permute_time         # noqa: E402

MODES = ('mc', 'mean', 'identity', 'averaged')
REF_NE = 10


def fixed_cfg(ne):
    return C.QVRConfig(n_qubits=3, n_layers=3, k=None, N_E=ne, tau=1.0, sigma_lambda=1e-2,
                       embedding='ry_all', ansatz='chain', cost='outside', draw_mode='mc')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_split(data_dir, channel=0):
    """{set: (N, T) float64 tensor}, {set: units}, {set: subtypes}, manifest."""
    d = Path(data_dir)
    X, units, sub = {}, {}, {}
    for p in sorted(d.glob('X*.pickle')):
        a = pickle.load(open(p, 'rb'))
        name = p.stem
        if name.endswith('_units'):
            units[name[:-6]] = np.asarray(a).astype(str)
        elif name.endswith('_subtype'):
            sub[name[:-8]] = np.asarray(a).astype(str)
        elif np.asarray(a).ndim == 3:
            X[name] = torch.as_tensor(np.asarray(a, dtype=np.float64)[:, channel, :])
    man = json.loads((d / 'manifest.json').read_text()) if (d / 'manifest.json').exists() else {}
    for k in X:
        units.setdefault(k, np.arange(len(X[k])).astype(str))
    return X, units, sub, man


def anomaly_sets(X):
    return sorted(k for k in X if k.startswith('Xte_') and k != 'Xte_norm')


# ---------------------------------------------------------------------------
# Train / score (one model per worker)
# ---------------------------------------------------------------------------

def _train_one(args):
    data_dir, out, channel, ne, seed, max_samples, strict = args
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    from qvr.train import train
    X, *_ = load_split(data_dir, channel)
    run_dir = Path(out) / 'models' / f'ne{ne}_s{seed}'
    if (run_dir / 'params.pt').exists():
        return f'ne{ne}_s{seed} (exists)'
    t = t_grid(X['Xtr'].shape[1])
    train(X['Xtr'], t, fixed_cfg(ne), run_seed=seed, out_dir=run_dir, max_samples=max_samples,
          meta={'data_dir': str(data_dir), 'channel': channel}, strict_git=strict, verbose=False)
    return f'ne{ne}_s{seed}'


def _score_one(args):
    data_dir, out, channel, ne, seed = args
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 4)))
    from qvr.train import load_run
    path = Path(out) / 'scores' / f'ne{ne}_s{seed}.npz'
    if path.exists():
        return f'ne{ne}_s{seed} (exists)'
    X, *_ = load_split(data_dir, channel)
    params, cfg, _ = load_run(Path(out) / 'models' / f'ne{ne}_s{seed}')
    t = t_grid(X['Xtr'].shape[1])
    res = {}
    for k, Xk in X.items():
        if k == 'Xtr':
            continue
        for m in MODES:
            o = infer(params, cfg, Xk, t, mode=m, seed=0)
            res[f'{m}__{k}'] = o['score']
            if m == 'mc':
                res[f'pbar__{k}'] = o['pbar']
        res[f'perm__{k}'] = infer(params, cfg, permute_time(Xk, seed=7, keep_first=False), t, mode='mc', seed=0)['score']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **res)
    return f'ne{ne}_s{seed}'


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
# Analysis helpers
# ---------------------------------------------------------------------------

def _pair(S, names, key):
    """(y, s) for Xte_norm vs the union of `names`, from score dict S with key prefix."""
    n = S[f'{key}__Xte_norm']
    a = np.concatenate([S[f'{key}__{k}'] for k in names])
    return np.r_[np.zeros(len(n)), np.ones(len(a))], np.r_[n, a]


def _unit_boot(y, s, u, n_boot=1000, seed=0, s2=None):
    """Cluster bootstrap over units: CI of AUC(s) or of AUC(s) − AUC(s2)."""
    rng = np.random.default_rng(seed)
    uu = np.unique(u)
    idx = {k: np.flatnonzero(u == k) for k in uu}
    vals = []
    for _ in range(n_boot):
        ix = np.concatenate([idx[k] for k in rng.choice(uu, len(uu))])
        if len(np.unique(y[ix])) < 2:
            continue
        v = roc_auc(y[ix], s[ix])
        if s2 is not None:
            v -= roc_auc(y[ix], s2[ix])
        vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))] if vals else [float('nan')] * 2


def _rank_avg(list_of_arrays):
    from scipy.stats import rankdata
    return np.mean([rankdata(a) / len(a) for a in list_of_arrays], axis=0)


def fourier_moments(X):
    return torch.stack([f(k * X).mean(1) for k in (1, 2, 3) for f in (torch.cos, torch.sin)], 1).numpy()


def e4_probe(F_fit, y_fit, u_fit, F_eval, y_eval, classes, n_perm=0, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mf, me = np.isin(y_fit, classes), np.isin(y_eval, classes)
    Ff, yf, uf, Fe, ye = F_fit[mf], y_fit[mf], u_fit[mf], F_eval[me], y_eval[me]
    if len(np.unique(yf)) < 2 or len(np.unique(ye)) < 2:
        return None
    labels = sorted(np.unique(yf))

    def metrics(model, y_train):
        model.fit(Ff, y_train)
        P = model.predict_proba(Fe)
        cls = list(model.classes_)
        pred = np.array(cls)[P.argmax(1)]
        if len(cls) == 2:
            auc = roc_auc_score(ye == cls[1], P[:, 1])
        else:
            auc = float(np.mean([roc_auc_score(ye == c, P[:, i]) for i, c in enumerate(cls) if (ye == c).any()]))
        return {'macro_auc': float(auc), 'bal_acc': float(balanced_accuracy_score(ye, pred)),
                'kappa': float(cohen_kappa_score(ye, pred))}

    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))  # noqa: E731
    out = {'n_fit': int(len(yf)), 'n_eval': int(len(ye)), 'classes': labels,
           'logreg': metrics(lr(), yf),
           'hgb': metrics(HistGradientBoostingClassifier(max_iter=200, early_stopping=True, random_state=0), yf)}
    if n_perm:
        # Unit-level null: units are permuted; each unit's series receive labels resampled from
        # the label mix of the unit it is mapped to. Keeps within-unit label clustering, breaks
        # the link between a unit's series and its labels. (One label per unit → plain shuffle.)
        rng = np.random.default_rng(seed)
        uu = np.unique(uf)
        idx = {k: np.flatnonzero(uf == k) for k in uu}
        null = []
        for _ in range(n_perm):
            perm = dict(zip(uu, rng.permutation(uu)))
            yp = np.empty_like(yf)
            for k in uu:
                yp[idx[k]] = rng.choice(yf[idx[perm[k]]], len(idx[k]), replace=True)
            if len(np.unique(yp)) < 2:
                continue
            null.append(metrics(lr(), yp)['macro_auc'])
        null = np.array(null)
        out['logreg']['null_mean'] = float(null.mean()) if len(null) else float('nan')
        out['logreg']['null_p95'] = float(np.quantile(null, 0.95)) if len(null) else float('nan')
        out['logreg']['p_value'] = float((1 + (null >= out['logreg']['macro_auc']).sum()) / (1 + len(null)))
    return out


def _S_r(mu, cfg):
    mu = np.abs(np.asarray(mu))
    orders = [len(c) for c in cfg.terms]
    return {r: float(mu[[i for i, o in enumerate(orders) if o == r]].sum() / mu.sum()) for r in sorted(set(orders))}


def e5_hamiltonian(params, cfg, init=None):
    """Eigenvalue gaps and interaction-order weights; compared with the initialization, because
    S_r of a random init is already ≈ (terms of order r) / (all terms), e.g. 3/7, 3/7, 1/7."""
    lam = C.eigenvalues(params['mu'], cfg).numpy()
    out = {'min_gap': float(np.diff(np.sort(lam)).min()), 'S_r': _S_r(params['mu'].numpy(), cfg),
           'abs_sigma': params['sigma'].abs().tolist()}
    if init is not None:
        out['S_r_init'] = _S_r(init['mu'].numpy(), cfg)
        out['mean_abs_dmu'] = float((params['mu'] - init['mu']).abs().mean())
        out['mean_abs_sigma_init'] = float(init['sigma'].abs().mean())
    return out


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------

def analyze(data_dir, out, channel, nes, seeds, n_boot, n_perm):
    from scipy.stats import spearmanr
    from qvr.train import load_run
    X, units, sub, man = load_split(data_dir, channel)
    sets = anomaly_sets(X)
    groups = {s: [s] for s in sets}
    groups['pooled'] = sets
    S = {(ne, s): dict(np.load(Path(out) / 'scores' / f'ne{ne}_s{s}.npz')) for ne in nes for s in seeds}
    ref = REF_NE if REF_NE in nes else nes[-1]
    rep = {'data_dir': str(data_dir), 'channel': channel, 'nes': nes, 'seeds': seeds, 'reference_ne': ref,
           'E1': {}, 'E2': {}, 'E3': {}, 'E4': {}, 'E5': {}}
    u_pool = lambda names: np.r_[units['Xte_norm'], np.concatenate([units[k] for k in names])]  # noqa: E731

    # E1 / E3 per model
    per = {}
    for (ne, s), Sc in S.items():
        per[(ne, s)] = {g: {m: roc_auc(*_pair(Sc, n, m)) for m in MODES + ('perm',)} for g, n in groups.items()}
    for g in groups:
        rep['E1'][g] = {str(ne): {'auc_mean': float(np.mean([per[(ne, s)][g]['mc'] for s in seeds])),
                                  'auc_sd': float(np.std([per[(ne, s)][g]['mc'] for s in seeds]))} for ne in nes}
        rep['E3'][g] = {str(ne): {m: float(np.mean([per[(ne, s)][g][m] for s in seeds])) for m in MODES} for ne in nes}
    for g, names in groups.items():
        y, _ = _pair(S[(ref, seeds[0])], names, 'mc')
        u = u_pool(names)
        sc = {m: _rank_avg([_pair(S[(ref, s)], names, m)[1] for s in seeds]) for m in MODES}
        rep['E1'][g]['ref_seed_avg'] = {'auc': roc_auc(y, sc['mc']), 'ci_units': _unit_boot(y, sc['mc'], u, n_boot)}
        rep['E3'][g]['ref_contrasts'] = {
            'mean_minus_mc': [roc_auc(y, sc['mean']) - roc_auc(y, sc['mc']), _unit_boot(y, sc['mean'], u, n_boot, s2=sc['mc'])],
            'mc_minus_identity': [roc_auc(y, sc['mc']) - roc_auc(y, sc['identity']), _unit_boot(y, sc['mc'], u, n_boot, s2=sc['identity'])]}

    # E2
    shuf = {}
    for (ne, s), Sc in S.items():
        a = np.concatenate([Sc[f'mc__{k}'] for k in ['Xte_norm'] + sets])
        b = np.concatenate([Sc[f'perm__{k}'] for k in ['Xte_norm'] + sets])
        shuf[(ne, s)] = {'shuf_r': float(np.corrcoef(a, b)[0, 1]),
                         'dAUC': per[(ne, s)]['pooled']['mc'] - per[(ne, s)]['pooled']['perm']}
    xs = [math.log(ne) for (ne, s) in shuf]
    rho = spearmanr(xs, [v['shuf_r'] for v in shuf.values()]).statistic if len(set(xs)) > 1 else float('nan')
    rep['E2'] = {'per_ne': {str(ne): {'shuf_r': float(np.mean([shuf[(ne, s)]['shuf_r'] for s in seeds])),
                                      'dAUC_shuf': float(np.mean([shuf[(ne, s)]['dAUC'] for s in seeds]))} for ne in nes},
                 'spearman_logNE_shuf_r': float(rho), 'prediction': 'negative (cervical)'}

    # E4
    elig = man.get('e4_eligibility') or {}
    classes = [c for c, v in elig.items() if v.get('eligible')] if elig else sorted(set(sub.get('Xprobe', [])))
    rep['E4']['eligible_classes'] = classes
    rep['E4']['eligibility'] = elig
    if 'Xprobe' in X and len(classes) >= 2:
        y_te = np.concatenate([sub.get(k, np.array([k[4:]] * len(X[k]))) for k in sets])
        mom = {'Xprobe': fourier_moments(X['Xprobe']), 'test': fourier_moments(torch.cat([X[k] for k in sets])),
               'Xval': fourier_moments(X['Xval']) if 'Xval' in X else None}
        for ne in nes:
            rows = []
            for s in seeds:
                Sc = S[(ne, s)]
                pb_te = np.concatenate([Sc[f'pbar__{k}'] for k in sets])
                r = {'pbar': e4_probe(Sc['pbar__Xprobe'], sub['Xprobe'], units['Xprobe'], pb_te, y_te, classes,
                                      n_perm if (ne == ref and s == seeds[0]) else 0)}
                if 'Xval' in X and 'Xval' in sub:
                    r['pbar_to_val'] = e4_probe(Sc['pbar__Xprobe'], sub['Xprobe'], units['Xprobe'],
                                                Sc['pbar__Xval'], sub['Xval'], classes)
                rows.append(r)
            rep['E4'][str(ne)] = rows
        rep['E4']['moments'] = e4_probe(mom['Xprobe'], sub['Xprobe'], units['Xprobe'], mom['test'], y_te, classes, n_perm)
        if mom['Xval'] is not None and 'Xval' in sub:
            rep['E4']['moments_to_val'] = e4_probe(mom['Xprobe'], sub['Xprobe'], units['Xprobe'], mom['Xval'], sub['Xval'], classes)

    # E5
    for ne in nes:
        rows = []
        for s in seeds:
            p, _, ck = load_run(Path(out) / 'models' / f'ne{ne}_s{s}')
            rows.append(e5_hamiltonian(p, fixed_cfg(ne), ck.get('init_params')))
        rep['E5'][str(ne)] = rows

    # score_audit hand-off (reference N_E, seed-rank-averaged mc scores)
    np.savez(Path(out) / 'scores' / 'ref_mc_scores.npz',
             **{k: _rank_avg([S[(ref, s)][f'mc__{k}'] for s in seeds]) for k in ['Xte_norm'] + sets})
    (Path(out) / 'summary.json').write_text(json.dumps(rep, indent=1, default=float))
    md = _markdown(rep, groups, nes)
    (Path(out) / 'summary.md').write_text(md)
    print(md)
    return rep


def _markdown(rep, groups, nes):
    L = [f"# Suite summary — {Path(rep['data_dir']).name} (channel {rep['channel']}, reference N_E = {rep['reference_ne']})", '']
    L += ['## E1 detection AUC (mc, mean ± SD over seeds)', '', '| set | ' + ' | '.join(f'N_E={n}' for n in nes) + ' | ref, seed-avg [95% unit CI] |',
          '| --- |' + ' --- |' * (len(nes) + 1)]
    for g in groups:
        e = rep['E1'][g]
        r = e['ref_seed_avg']
        L.append(f"| {g.replace('Xte_', '')} | " + ' | '.join(f"{e[str(n)]['auc_mean']:.3f} ± {e[str(n)]['auc_sd']:.3f}" for n in nes)
                 + f" | {r['auc']:.3f} [{r['ci_units'][0]:.3f}, {r['ci_units'][1]:.3f}] |")
    L += ['', f"## E2 temporal sensitivity — Spearman(log N_E, shuf_r) = {rep['E2']['spearman_logNE_shuf_r']:+.3f} (predicted negative)", '',
          '| N_E | shuf_r | ΔAUC_shuf (pooled) |', '| --- | --- | --- |']
    for n in nes:
        e = rep['E2']['per_ne'][str(n)]
        L.append(f"| {n} | {e['shuf_r']:.3f} | {e['dAUC_shuf']:+.3f} |")
    L += ['', '## E3 D-mode AUC (pooled, mean over seeds)', '', '| N_E | ' + ' | '.join(MODES) + ' |', '| --- |' + ' --- |' * len(MODES)]
    for n in nes:
        e = rep['E3']['pooled'][str(n)]
        L.append(f'| {n} | ' + ' | '.join(f'{e[m]:.3f}' for m in MODES) + ' |')
    c = rep['E3']['pooled']['ref_contrasts']
    L += ['', f"Reference contrasts (pooled, seed-avg, unit CI): mean − mc {c['mean_minus_mc'][0]:+.3f} "
          f"[{c['mean_minus_mc'][1][0]:+.3f}, {c['mean_minus_mc'][1][1]:+.3f}]; mc − identity {c['mc_minus_identity'][0]:+.3f} "
          f"[{c['mc_minus_identity'][1][0]:+.3f}, {c['mc_minus_identity'][1][1]:+.3f}]"]
    e4 = rep['E4']
    L += ['', f"## E4 subtype probes — eligible classes: {e4.get('eligible_classes')}", '']
    if 'moments' in e4 and e4['moments']:
        L += ['| features | probe | macro AUC | bal. acc | κ | null p95 | p |', '| --- | --- | --- | --- | --- | --- | --- |']
        pb = e4[str(rep['reference_ne'])][0]['pbar']
        for name, r in (('p̄_b (ref N_E, seed 0)', pb), ('Fourier moments', e4['moments'])):
            for pr in ('logreg', 'hgb'):
                m = r[pr]
                L.append(f"| {name} | {pr} | {m['macro_auc']:.3f} | {m['bal_acc']:.3f} | {m['kappa']:.3f} | "
                         f"{m.get('null_p95', float('nan')):.3f} | {m.get('p_value', float('nan')):.3f} |")
    else:
        L.append('Not run (fewer than two eligible classes or no Xprobe).')
    L += ['', '## E5 Hamiltonian (mean over seeds; S_r at initialization in parentheses)', '',
          '| N_E | min gap | S_1 | S_2 | S_3 | mean abs Δμ from init | mean abs σ (init → trained) |', '| --- | --- | --- | --- | --- | --- | --- |']
    for n in nes:
        rows = rep['E5'][str(n)]
        has0 = all('S_r_init' in r for r in rows)
        cell = lambda k: (f"{np.mean([r['S_r'][k] for r in rows]):.2f}" +  # noqa: E731
                          (f" ({np.mean([r['S_r_init'][k] for r in rows]):.2f})" if has0 else ''))
        dmu = f"{np.mean([r['mean_abs_dmu'] for r in rows]):.3f}" if has0 else '—'
        sig = (f"{np.mean([r['mean_abs_sigma_init'] for r in rows]):.2f} → {np.mean([np.mean(r['abs_sigma']) for r in rows]):.2f}"
               if has0 else f"{np.mean([np.mean(r['abs_sigma']) for r in rows]):.2f}")
        L.append(f"| {n} | {np.mean([r['min_gap'] for r in rows]):.3f} | " + ' | '.join(cell(k) for k in (1, 2, 3))
                 + f' | {dmu} | {sig} |')
    return '\n'.join(L) + '\n'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(data_dir, out, nes, seeds, stages, workers=1, channel=0, max_samples=50_000, n_boot=1000,
        n_perm=1000, strict_git=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'data_dir': str(data_dir), 'nes': nes, 'seeds': seeds,
                                                 'channel': channel, 'max_samples': max_samples,
                                                 'model': fixed_cfg(REF_NE).to_dict()}, indent=1))
    jobs = [(ne, s) for ne in nes for s in seeds]
    if 'train' in stages:
        print(f'== train: {len(jobs)} models')
        _pool(_train_one, [(str(data_dir), str(out), channel, ne, s, max_samples, strict_git) for ne, s in jobs], workers)
    if 'score' in stages:
        print(f'== score: {len(jobs)} models')
        _pool(_score_one, [(str(data_dir), str(out), channel, ne, s) for ne, s in jobs], workers)
    if 'analyze' in stages:
        print('== analyze')
        return analyze(data_dir, out, channel, nes, seeds, n_boot, n_perm)


def _self_test():
    import tempfile
    print('run_suite self-test: synthetic split dir, 2 N_E × 2 seeds, tiny training')
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_dataset import prepare_synthetic, SYN_SPLITS
    SYN_SPLITS['normal'].update({'Xtr': 120, 'Xval_norm': 30, 'Xte_norm': 40})
    SYN_SPLITS['anomaly'].update({'Xval': 15, 'Xprobe': 30, 'Xte': 25})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prepare_synthetic(Path(tmp) / 'syn', seq_len=12)
        rep = run(Path(tmp) / 'syn', Path(tmp) / 'out', nes=[1, 3], seeds=[0, 1],
                  stages=('train', 'score', 'analyze'), workers=1, max_samples=80, n_boot=50, n_perm=20)
        checks = {
            'models trained': len(list((Path(tmp) / 'out' / 'models').glob('ne*_s*'))) == 4,
            'scores written': len(list((Path(tmp) / 'out' / 'scores').glob('ne*_s*.npz'))) == 4,
            'E1 AUCs in [0, 1]': all(0 <= rep['E1']['pooled'][str(n)]['auc_mean'] <= 1 for n in (1, 3)),
            'E3 identity pooled AUC present': 'identity' in rep['E3']['pooled']['3'],
            'E4 ran with a permutation null': rep['E4'].get('moments') is not None
                                              and 'p_value' in rep['E4']['moments']['logreg'],
            'E5 S_r sums to 1, init comparison present': all(abs(sum(r['S_r'].values()) - 1) < 1e-9 and 'S_r_init' in r
                                                             for r in rep['E5']['1']),
            'summary.md + score_audit hand-off written': (Path(tmp) / 'out' / 'summary.md').exists()
                                                        and (Path(tmp) / 'out' / 'scores' / 'ref_mc_scores.npz').exists(),
            'resumable (second run skips)': 'exists' in _score_one((str(Path(tmp) / 'syn'), str(Path(tmp) / 'out'), 0, 1, 0)),
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
    ap.add_argument('--ne', type=int, nargs='+', default=[1, 3, 5, 10, 20])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--stages', nargs='+', default=['train', 'score', 'analyze'], choices=['train', 'score', 'analyze'])
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--max-samples', type=int, default=50_000)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--strict-git', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not (a.data_dir and a.out):
            ap.error('--data-dir and --out are required')
        run(a.data_dir, a.out, a.ne, a.seeds, a.stages, a.workers, a.channel, a.max_samples, a.n_boot,
            a.n_perm, a.strict_git)