"""
scripts/readout_test.py — EXPLORATORY Test A (ANALYSIS_PLAN §11, 2026-10-05).

Is QVR's weakness its scoring rule? Same trained models, same circuit outputs, four ways
to turn them into an anomaly score (all fit on training normals only):

  native        QVR's own cost: mean_t (η₀ − mean_q ⟨Z_q⟩_t)² / 4
  tz2_z         per-timepoint z² of the measured ⟨Z_q⟩_t (mean / SD per t and q from
                training normals), averaged over t and q
  mahal_zmean   Mahalanobis distance of time-averaged ⟨Z_q⟩ (3) from training normals
  mahal_pbar    Mahalanobis distance of the fingerprint p̄_b (8) from training normals

D mode 'mc', inference seed 0. Per N_E: AUC per model, and seed-rank-averaged AUC with a
95% bootstrap CI over independent units (records). The classical per-timepoint z² on the raw
series (scripts/score_audit.py, 'tz2_primary') is printed for reference when available.

Registered prediction (§11): tz2_z ≫ native on MIT-BIH, approaching the raw-series tz2
(pooled ≈ 0.72, VEB ≈ 0.90); mahal_pbar ≈ the trig-moment comparator (pooled ≈ 0.70).

Usage:
    python scripts/readout_test.py --data-dir data_splits/mitbih --suite-dir results/suite/mitbih
    python scripts/readout_test.py --self-test
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
from qvr.data import roc_auc, t_grid                                    # noqa: E402
from qvr.infer import infer                                             # noqa: E402
from run_suite import _rank_avg, _unit_boot, anomaly_sets, load_split   # noqa: E402

READOUTS = ('native', 'tz2_z', 'mahal_zmean', 'mahal_pbar')


def _mahal(train, x):
    mu = train.mean(0)
    P = np.linalg.inv(np.cov(train, rowvar=False) + 1e-6 * np.eye(train.shape[1]))
    return np.sqrt(np.einsum('ij,jk,ik->i', x - mu, P, x - mu))


def readouts(params, cfg, X, n_fit, seed=0):
    t = t_grid(X['Xtr'].shape[1])
    rng = np.random.default_rng(seed)
    fit_idx = torch.as_tensor(np.sort(rng.choice(len(X['Xtr']), min(n_fit, len(X['Xtr'])), replace=False)))
    o_tr = infer(params, cfg, X['Xtr'][fit_idx], t, mode='mc', seed=0, keep_time=True)
    mu_t, sd_t = o_tr['z_t'].mean(0), o_tr['z_t'].std(0) + 1e-9
    out = {r: {} for r in READOUTS}
    for k in ['Xte_norm'] + anomaly_sets(X):
        o = infer(params, cfg, X[k], t, mode='mc', seed=0, keep_time=True)
        out['native'][k] = o['score']
        out['tz2_z'][k] = (((o['z_t'] - mu_t) / sd_t) ** 2).mean((1, 2))
        out['mahal_zmean'][k] = _mahal(o_tr['zrepr'], o['zrepr'])
        out['mahal_pbar'][k] = _mahal(o_tr['pbar'], o['pbar'])
    return out


def run(data_dir, suite_dir, nes=(1, 10), channel=0, n_fit=5000, n_boot=1000):
    from qvr.train import load_run
    suite_dir = Path(suite_dir)
    seeds = json.loads((suite_dir / 'config.json').read_text())['seeds']
    X, units, sub, man = load_split(data_dir, channel)
    sets = anomaly_sets(X)
    groups = {s: [s] for s in sets}
    groups['pooled'] = sets
    rep = {'exploratory': True, 'per_ne': {}}
    print('=' * 96)
    print(f'EXPLORATORY Test A — readouts on trained QVR models — {Path(data_dir).name}')
    print('=' * 96)
    for ne in nes:
        R = {s: readouts(*load_run(suite_dir / 'models' / f'ne{ne}_s{s}')[:2], X, n_fit) for s in seeds}
        res = {}
        for g, names in groups.items():
            y = np.r_[np.zeros(len(X['Xte_norm'])), np.ones(sum(len(X[k]) for k in names))]
            u = np.r_[units['Xte_norm'], np.concatenate([units[k] for k in names])]
            res[g] = {}
            for r in READOUTS:
                per_seed = [np.r_[R[s][r]['Xte_norm'], np.concatenate([R[s][r][k] for k in names])] for s in seeds]
                avg = _rank_avg(per_seed)
                res[g][r] = {'auc_seeds': [roc_auc(y, v) for v in per_seed], 'auc_seed_avg': roc_auc(y, avg),
                             'ci_units': _unit_boot(y, avg, u, n_boot)}
        rep['per_ne'][str(ne)] = res
        print(f'\nN_E = {ne}: seed-rank-averaged AUC [95% unit CI]')
        print(f"  {'set':18s} " + ''.join(f'{r:>26s}' for r in READOUTS))
        for g in groups:
            print(f"  {g.replace('Xte_', ''):18s} " + ''.join(
                f"{res[g][r]['auc_seed_avg']:.3f} [{res[g][r]['ci_units'][0]:.2f},{res[g][r]['ci_units'][1]:.2f}]".rjust(26)
                for r in READOUTS))
    sa = suite_dir / 'score_audit' / 'score_audit.json'
    if sa.exists():
        a = json.loads(sa.read_text())['sets']
        print('\nClassical reference (score_audit, raw series): ' + ', '.join(
            f"{g.replace('Xte_', '').replace('pooled_all', 'pooled')} tz2 {v['methods']['tz2_primary']['auc']:.3f} / "
            f"trig {v['methods']['trig_moments']['auc']:.3f}" for g, v in a.items()))
    out = suite_dir / 'readout_test'
    out.mkdir(exist_ok=True)
    (out / 'readout_test.json').write_text(json.dumps(rep, indent=1, default=float))
    print(f'\nsaved → {out}/readout_test.json')
    return rep


def _self_test():
    import tempfile
    from prepare_dataset import SYN_SPLITS, prepare_synthetic
    from run_suite import run as suite_run
    print('readout_test self-test: tiny synthetic suite, then readouts')
    SYN_SPLITS['normal'].update({'Xtr': 150, 'Xval_norm': 30, 'Xte_norm': 40})
    SYN_SPLITS['anomaly'].update({'Xval': 15, 'Xprobe': 30, 'Xte': 30})
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prepare_synthetic(Path(tmp) / 'syn', seq_len=12)
        suite_run(Path(tmp) / 'syn', Path(tmp) / 'suite', nes=[1], seeds=[0, 1], stages=('train',), max_samples=80)
        rep = run(Path(tmp) / 'syn', Path(tmp) / 'suite', nes=(1,), n_fit=100, n_boot=50)
        r = rep['per_ne']['1']['pooled']
        checks = {
            'all readouts evaluated': set(r) == set(READOUTS),
            'AUCs in [0, 1]': all(0 <= r[k]['auc_seed_avg'] <= 1 for k in READOUTS),
            'best density readout ≥ native score': max(r[k]['auc_seed_avg'] for k in READOUTS[1:]) >= r['native']['auc_seed_avg'],
            'json written': (Path(tmp) / 'suite' / 'readout_test' / 'readout_test.json').exists(),
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
    ap.add_argument('--ne', type=int, nargs='+', default=[1, 10])
    ap.add_argument('--channel', type=int, default=0)
    ap.add_argument('--n-fit', type=int, default=5000)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not (a.data_dir and a.suite_dir):
            ap.error('--data-dir and --suite-dir are required')
        run(a.data_dir, a.suite_dir, a.ne, a.channel, a.n_fit, a.n_boot)