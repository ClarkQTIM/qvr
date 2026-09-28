"""
scripts/explore_fingerprint.py — EXPLORATORY (cervical is the discovery dataset).

Two questions about the eigenbasis fingerprint p̄_b:

  1. What does each eigenstate respond to?  p_b(x) = |<b|W E(x)|0>|^2 are 8
     nonnegative curves over normalized intensity that sum to 1, so p̄_b is a
     learned soft histogram. Plot them per seed, over the data distribution.

  2. Does p̄_b carry anything beyond the six Fourier moments
     m̄ = mean_t (cos kx_t, sin kx_t), k = 1..3?  Fact 1 says p̄_b = A_W m̄, so a
     linear probe cannot differ. A nonlinear probe can. Compare both.

Data: the TRAINING split's segments. QVR was trained on none-voted segments only,
so the any-voted (lesion) segments of the training split were never seen by the
model; the normalizer uses training normals only. Val and test stay untouched.
Evaluation: 5-fold patient-grouped, class-stratified CV.

Tasks
  detect   any-voted vs none-voted (none subsampled per patient)
  cin2p    CIN2+ vs Neg/CIN1, any-voted segments of main-grade patients
  grade    Neg / CIN1 / CIN2 / CIN3 (linear-weighted κ), same segments

Usage:
    python scripts/explore_fingerprint.py --grade-csv <csv> --list-columns
    python scripts/explore_fingerprint.py \
        --checkpoints <old trained_model.pt> [<more seeds> ...] \
        --grade-csv /scratch90/chris_/quantum/qvr_cervix/csvs/20210331_ai_dataset_0.1.csv \
        --pid-col <col> --grade-col <col> --out results/explore/fingerprint
    python scripts/explore_fingerprint.py --self-test
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Cap threads BEFORE numpy/torch/sklearn load: small problems on many-core
# machines slow down badly when every library grabs all cores.
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C                      # noqa: E402
from qvr.data import Normalizer                   # noqa: E402
from qvr.datasets import cervical                 # noqa: E402
from qvr.train import git_state                   # noqa: E402

GRADES = ['Neg', 'CIN1', 'CIN2', 'CIN3']


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def _grade_code(v) -> int:
    s = str(v).strip().lower().replace(' ', '').replace('-', '')
    if s.startswith('neg') or s in ('normal', 'benign', 'none'):
        return 0
    for i, g in ((1, 'cin1'), (2, 'cin2'), (3, 'cin3')):
        if s.startswith(g) or s == g[-1:]:
            return i
    return -1          # AIS, invasive, unknown → excluded from grade tasks


def load_grades(csv_path, pid_col, grade_col) -> dict:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = {}
    for pid, g in zip(df[pid_col], df[grade_col]):
        if pd.isna(pid):
            continue
        out[str(int(pid)).zfill(5) if str(pid).replace('.', '').isdigit() else str(pid)] = _grade_code(g)
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def fourier_moments(Xn: torch.Tensor, K: int = 3) -> np.ndarray:
    """(N, 2K) time-averaged cos(kx), sin(kx), k = 1..K."""
    feats = [f(k * Xn).mean(dim=1) for k in range(1, K + 1) for f in (torch.cos, torch.sin)]
    return torch.stack(feats, dim=1).numpy()


@torch.no_grad()
def pbar(params, cfg, Xn, chunk=65536) -> np.ndarray:
    return np.concatenate([C.probs_after_W(params, Xn[i:i + chunk], cfg).mean(dim=1).numpy()
                           for i in range(0, len(Xn), chunk)])


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        'logreg': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        'hgb': lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                      max_leaf_nodes=15, early_stopping=True,
                                                      random_state=0),
    }


def grouped_cv(F, y, groups, task, folds=5, seed=0) -> dict:
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    res = {}
    for name, make in _models().items():
        oof = np.zeros((len(y), len(np.unique(y))))
        for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(F, y, groups):
            m = make().fit(F[tr], y[tr])
            oof[te] = m.predict_proba(F[te])
        if task == 'grade':
            pred = oof.argmax(1)
            res[name] = {'kappa_lin': float(cohen_kappa_score(y, pred, weights='linear')),
                         'acc': float((pred == y).mean())}
        else:
            p1 = oof[:, 1]
            r = {'auc_seg': float(roc_auc_score(y, p1))}
            if task == 'cin2p':
                u = np.unique(groups)
                pp = np.array([p1[groups == g].mean() for g in u])
                py = np.array([y[groups == g][0] for g in u])
                r['auc_patient'] = float(roc_auc_score(py, pp)) if len(np.unique(py)) == 2 else float('nan')
                r['n_patients'] = int(len(u))
            res[name] = r
    return res


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(checkpoints, data_dir, grades, out, none_per_patient=200, folds=5, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = cervical.load_split('train', data_dir)
    rng = np.random.default_rng(seed)
    gcode = np.array([grades.get(u, -1) for u in ds.unit])

    # detection subset: all any-voted + up to N none-voted per patient
    keep = ds.y == 1
    for u in np.unique(ds.unit[ds.y == 0]):
        idx = np.flatnonzero((ds.unit == u) & (ds.y == 0))
        keep[rng.choice(idx, min(none_per_patient, len(idx)), replace=False)] = True
    det = np.flatnonzero(keep)
    les = np.flatnonzero((ds.y == 1) & (gcode >= 0))
    print('=' * 64)
    print('EXPLORATORY fingerprint analysis — cervical TRAIN split (lesions unseen by QVR)')
    print(f'  detect: {len(det):,} segments ({ds.y[det].sum():,} lesion), '
          f'{len(np.unique(ds.unit[det]))} patients')
    print(f'  grade : {len(les):,} lesion segments, {len(np.unique(ds.unit[les]))} main-grade patients, '
          f'per grade (patients): ' + ', '.join(
              f'{g}={len(np.unique(ds.unit[les][gcode[les] == i]))}' for i, g in enumerate(GRADES)))
    print('=' * 64)

    xs = torch.linspace(-math.pi, math.pi, 721, dtype=C.DTYPE)
    report = {'exploratory': True, 'split': 'train', 'git': git_state(),
              'n_detect': int(len(det)), 'n_grade': int(len(les)), 'seeds': []}
    rows, curves = [], {}

    for ck in checkpoints:
        t0 = time.time()
        params, cfg, norm = C.load_legacy_checkpoint(ck)
        nm = Normalizer.from_legacy(norm)
        name = Path(ck).parent.parent.name
        idx_all = np.union1d(det, les)
        Xn = nm.transform(ds.X[idx_all])
        pos = {i: k for k, i in enumerate(idx_all)}
        P = pbar(params, cfg, Xn)
        M = fourier_moments(Xn)
        # Fact 1 on real data with trained W
        A = np.c_[np.ones(len(M)), M]
        resid = float(np.abs(A @ np.linalg.lstsq(A, P, rcond=None)[0] - P).max())
        curves[name] = C.probs_after_W(params, xs, cfg).detach().numpy()
        seed_rep = {'run': name, 'fact1_max_resid': resid, 'tasks': {}}
        print(f'{name}\n  Fact 1 on real segments (trained W): max residual {resid:.1e}')

        sel = lambda ids: np.array([pos[i] for i in ids])      # noqa: E731
        tasks = {
            'detect': (det, ds.y[det]),
            'cin2p': (les, (gcode[les] >= 2).astype(int)),
            'grade': (les, gcode[les]),
        }
        for task, (ids, y) in tasks.items():
            g = ds.unit[ids]
            seed_rep['tasks'][task] = {}
            for feat, F in (('pbar', P[sel(ids)]), ('moments', M[sel(ids)])):
                r = grouped_cv(F, y, g, task, folds, seed)
                seed_rep['tasks'][task][feat] = r
                for model, metrics in r.items():
                    rows.append({'run': name, 'task': task, 'features': feat, 'model': model, **metrics})
            line = '  '.join(
                f"{feat}/{m}: " + ', '.join(f'{k}={v:.3f}' for k, v in seed_rep['tasks'][task][feat][m].items()
                                           if k != 'n_patients')
                for feat in ('pbar', 'moments') for m in ('logreg', 'hgb'))
            print(f'  {task:6s} {line}')
        report['seeds'].append(seed_rep)
        print(f'  ({time.time() - t0:.0f}s)')

    np.savez(out / 'bin_curves.npz', x=xs.numpy(), **curves)
    (out / 'fingerprint_explore.json').write_text(json.dumps(report, indent=1))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / 'fingerprint_explore.csv', index=False)
    except ImportError:
        pass
    _plot_curves(xs.numpy(), curves, nm.transform(ds.X[det]), ds.y[det], out / 'bin_curves.png')
    print(f'saved → {out}/ (fingerprint_explore.json, .csv, bin_curves.npz, bin_curves.png)')
    return report


def _plot_curves(x, curves, Xn_det, y_det, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(curves)
    fig, axes = plt.subplots(n, 1, figsize=(8, 2.6 * n), sharex=True, squeeze=False)
    vals_n = Xn_det[y_det == 0, 1:].numpy().ravel()
    vals_a = Xn_det[y_det == 1, 1:].numpy().ravel()
    for ax, (name, P) in zip(axes[:, 0], curves.items()):
        for b in range(P.shape[1]):
            ax.plot(x, P[:, b], lw=1.4, label=f'|{b:03b}⟩')
        ax2 = ax.twinx()
        bins = np.linspace(-math.pi, math.pi, 121)
        ax2.hist(vals_n, bins, density=True, alpha=0.18, color='k', label='normal values')
        ax2.hist(vals_a, bins, density=True, alpha=0.18, color='r', label='lesion values')
        ax2.set_yticks([])
        ax.set_ylabel('p_b(x)')
        ax.set_title(name, fontsize=9)
    axes[0, 0].legend(ncol=8, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, 1.35))
    axes[-1, 0].set_xlabel('normalized value x (frames 1–16); shaded: data distribution')
    fig.suptitle('EXPLORATORY — eigenstate response curves (learned soft-histogram bins)', y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile
    import pandas as pd
    print('explore_fingerprint self-test: fake checkpoints, data and grade CSV')
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        pids = [f'{i:05d}' for i in range(40)]
        grade_of = {p: GRADES[i % 4] for i, p in enumerate(pids)}
        none, npid, anyv, apid = [], [], [], []
        for i, p in enumerate(pids):
            none.append(torch.randn(300, 17, generator=g) * 0.05); npid += [p] * 300
            shift = 0.02 * (i % 4)
            anyv.append(torch.randn(40, 17, generator=g) * 0.08 + shift); apid += [p] * 40
        none, anyv = torch.cat(none), torch.cat(anyv)
        none[:, 0] = anyv[:, 0] = 0
        torch.save({'none_time_series': none, 'any_time_series': anyv,
                    'none_patient_ids': np.array(npid), 'any_patient_ids': np.array(apid)},
                   cervical.pt_path('train', tmp))
        pd.DataFrame({'pid': [int(p) for p in pids], 'hist': [grade_of[p] for p in pids]}).to_csv(
            Path(tmp) / 'grades.csv', index=False)
        lo, hi = none[:, 1:].min(0).values, none[:, 1:].max(0).values
        cks = []
        for s in range(2):
            cfg = C.QVRConfig()
            p = C.init_params(cfg, np.random.default_rng(s))
            ck = Path(tmp) / f'run_s{s}' / 'training' / 'trained_model.pt'
            ck.parent.mkdir(parents=True)
            torch.save({'model_state_dict': {k: v.detach() for k, v in p.items()},
                        'model_config': {'n_qubits': 3, 'n_layers': 3, 'N_E': 10, 'k': 3,
                                         'embedding_strategy': 'fixed'},
                        'norm_config': {'norm': 'per_timepoint', 'norm_params':
                                        {'per_tp_min': lo.tolist(), 'per_tp_max': hi.tolist()}}}, ck)
            cks.append(ck)
        grades = load_grades(Path(tmp) / 'grades.csv', 'pid', 'hist')
        ok = len(grades) == 40 and set(grades.values()) == {0, 1, 2, 3}
        print(f"  [{'PASS' if ok else 'FAIL'}] grade CSV parsed → codes {sorted(set(grades.values()))}")
        rep = run(cks, tmp, grades, Path(tmp) / 'out', none_per_patient=50, folds=3)
        checks = {
            'Fact 1 residual < 1e-10 (both seeds)': all(s['fact1_max_resid'] < 1e-10 for s in rep['seeds']),
            'all tasks × features × models present': all(
                set(s['tasks'][t][f]) == {'logreg', 'hgb'} for s in rep['seeds']
                for t in ('detect', 'cin2p', 'grade') for f in ('pbar', 'moments')),
            'outputs written': all((Path(tmp) / 'out' / f).exists() for f in
                                   ('fingerprint_explore.json', 'fingerprint_explore.csv',
                                    'bin_curves.npz', 'bin_curves.png')),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        good = ok and all(checks.values())
        print('RESULT:', 'ALL PASS' if good else 'FAIL')
        if not good:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoints', nargs='+')
    ap.add_argument('--data-dir', default=str(cervical.DATA_DIR))
    ap.add_argument('--grade-csv')
    ap.add_argument('--pid-col')
    ap.add_argument('--grade-col')
    ap.add_argument('--list-columns', action='store_true')
    ap.add_argument('--none-per-patient', type=int, default=200)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', default='results/explore/fingerprint')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    elif a.list_columns:
        import pandas as pd
        df = pd.read_csv(a.grade_csv)
        print(df.columns.tolist())
        print(df.head().to_string())
    else:
        if not (a.checkpoints and a.grade_csv and a.pid_col and a.grade_col):
            ap.error('--checkpoints, --grade-csv, --pid-col and --grade-col are required')
        run(a.checkpoints, a.data_dir, load_grades(a.grade_csv, a.pid_col, a.grade_col),
            a.out, a.none_per_patient, a.folds)