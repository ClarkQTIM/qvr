"""
scripts/replicate_grade.py — reproduce the old paper's grade result, then take it apart.

Old result (seed 0, unfiltered_repr_results.json): spectral p̄_b probe fit on VAL
smooth main-grade lesion segments, applied cold to TEST:
    CIN2+ AUC 0.7401 (segment), 0.8133 (patient, 28 pts); grade κ 0.1877 (segment)

Steps, per seed:
  A  old caches   rebuild row order from the .pt files, refit the probe on the OLD
                  val cache, score the OLD test cache → must reproduce the old numbers
  B  new code     p̄_b from qvr.circuit vs the old caches (max |Δ|) → same numbers
  C  moments      same val→test protocol on the six Fourier moments (Fact 1)
  D  null         permute grades across VAL PATIENTS (keeps segments together),
                  refit, score test with true labels → where does the real AUC fall?
  E  more data    fit on TRAIN-split lesions (78 patients, unseen by QVR) → test / val
  F  composition  per-patient grades and segment counts in val

Touches test labels. This re-examines the old paper's already-reported test
result and chooses nothing for the new analysis; log as a deviation in
ANALYSIS_PLAN.md §11.

Usage:
    python scripts/replicate_grade.py --runs <old run dir> [...] \
        --grade-csv <csv> --out results/explore/replicate_grade
    python scripts/replicate_grade.py --self-test
"""

import argparse
import json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qvr import circuit as C                                    # noqa: E402
from qvr.data import Normalizer                                 # noqa: E402
from qvr.datasets import cervical                               # noqa: E402
from qvr.train import git_state                                 # noqa: E402
from explore_fingerprint import fourier_moments, load_grades, pbar   # noqa: E402

OLD_SEED0 = {'seg_auc': 0.7401130962928154, 'pt_auc': 0.8133333333333334,
             'kappa_seg': 0.18772858802942582}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def lesion_rows(split, data_dir, grades):
    """Smooth any-voted, main-grade (0–3) segments in .pt order: X, pids, grade."""
    ds = cervical.load_split(split, data_dir)
    g = np.array([grades.get(u, -1) for u in ds.unit])
    m = (ds.y == 1) & (g >= 0) & (g <= 3)
    return ds.X[torch.as_tensor(m)], ds.unit[m], g[m]


# ---------------------------------------------------------------------------
# Probe protocol (matches build_unfiltered_repr_probes.py)
# ---------------------------------------------------------------------------

def fit_eval(F_tr, g_tr, F_te, g_te, pid_te, seed=0, grade=True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    y_tr, y_te = (g_tr >= 2).astype(int), (g_te >= 2).astype(int)
    sc = StandardScaler().fit(F_tr)
    lr = LogisticRegression(max_iter=2000, C=0.1, random_state=seed).fit(sc.transform(F_tr), y_tr)
    p = lr.predict_proba(sc.transform(F_te))[:, 1]
    u = np.unique(pid_te)
    pp = np.array([p[pid_te == k].mean() for k in u])
    py = np.array([y_te[pid_te == k][0] for k in u])
    out = {'seg_auc': float(roc_auc_score(y_te, p)),
           'pt_auc': float(roc_auc_score(py, pp)) if len(set(py)) == 2 else float('nan'),
           'n_pt': int(len(u))}
    if grade:
        gl = LogisticRegression(max_iter=2000, C=0.1, random_state=seed).fit(sc.transform(F_tr), g_tr)
        out['kappa_seg'] = float(cohen_kappa_score(g_te, gl.predict(sc.transform(F_te)), weights='linear'))
    return out


def patient_permutation_null(F_tr, g_tr, pid_tr, F_te, g_te, pid_te, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(pid_tr)
    pg = np.array([g_tr[pid_tr == k][0] for k in u])
    idx = {k: i for i, k in enumerate(u)}
    row = np.array([idx[k] for k in pid_tr])
    null = []
    for _ in range(n_perm):
        null.append(fit_eval(F_tr, rng.permutation(pg)[row], F_te, g_te, pid_te, seed, grade=False))
    return {k: np.array([n[k] for n in null]) for k in ('seg_auc', 'pt_auc')}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(runs, data_dir, grades, out, n_perm=1000, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Xv, pv, gv = lesion_rows('val', data_dir, grades)
    Xt, pt, gt = lesion_rows('test', data_dir, grades)
    Xr, pr, gr = lesion_rows('train', data_dir, grades)

    print('=' * 70)
    print('Grade replication — smooth any-voted main-grade segments')
    for name, p, g in (('train', pr, gr), ('val', pv, gv), ('test', pt, gt)):
        cnt = ', '.join(f'{k}={len(np.unique(p[g == i]))}pt/{(g == i).sum():,}seg'
                        for i, k in enumerate(['Neg', 'CIN1', 'CIN2', 'CIN3']))
        print(f'  {name:5s} {len(g):6,} segs, {len(np.unique(p)):3d} pts  ({cnt})')
    print('=' * 70)

    comp = [{'pid': k, 'grade': int(gv[pv == k][0]), 'n_seg': int((pv == k).sum())}
            for k in np.unique(pv)]
    print('F  val composition (patients with the most segments):')
    for c in sorted(comp, key=lambda c: -c['n_seg'])[:8]:
        print(f"     {c['pid']}  grade {c['grade']}  {c['n_seg']:5,} segs")

    report = {'exploratory': True, 'touches_test': True, 'git': git_state(),
              'counts': {'val': int(len(gv)), 'test': int(len(gt)), 'train': int(len(gr))},
              'val_composition': comp, 'seeds': []}
    rows = []
    for rd in runs:
        t0 = time.time()
        rd = Path(rd)
        params, cfg, norm = C.load_legacy_checkpoint(rd / 'training' / 'trained_model.pt')
        nm = Normalizer.from_legacy(norm)
        P = {s: pbar(params, cfg, nm.transform(X)) for s, X in (('val', Xv), ('test', Xt), ('train', Xr))}
        M = {s: fourier_moments(nm.transform(X)) for s, X in (('val', Xv), ('test', Xt), ('train', Xr))}
        rep = {'run': rd.name}
        print(f'\n{rd.name}')

        # A + B: old caches
        uv = rd / 'unfiltered_val'
        oldv, oldt = uv / 'val_spectral_cache_smooth.npz', uv / 'test_spectral_cache_smooth.npz'
        if oldv.exists() and oldt.exists():
            cv, ct = np.load(oldv)['probs'], np.load(oldt)['probs']
            aligned = cv.shape == P['val'].shape and ct.shape == P['test'].shape
            rep['B_cache_rows'] = [list(cv.shape), list(ct.shape)]
            if aligned:
                rep['B_max_diff'] = float(max(np.abs(cv - P['val']).max(), np.abs(ct - P['test']).max()))
                print(f"  B  new p̄_b vs old caches: max |Δ| {rep['B_max_diff']:.1e}")
            else:
                print(f'  B  row mismatch: old caches {cv.shape}/{ct.shape} vs rebuilt '
                      f"{P['val'].shape}/{P['test'].shape} — ordering differs, stop and check")
            rep['A_old_cache'] = fit_eval(cv, gv, ct, gt, pt, seed) if aligned else None
            if aligned:
                a = rep['A_old_cache']
                print(f"  A  old caches, val→test:  seg {a['seg_auc']:.4f}  pt {a['pt_auc']:.4f}  "
                      f"κ {a['kappa_seg']:.4f}")
        else:
            print('  A/B skipped: old caches not found')

        # new code, same protocol; moments; more training data
        for key, (Ftr, gtr, Fte, gte, pte) in {
            'val_to_test_pbar': (P['val'], gv, P['test'], gt, pt),
            'val_to_test_moments': (M['val'], gv, M['test'], gt, pt),
            'train_to_test_pbar': (P['train'], gr, P['test'], gt, pt),
            'train_to_val_pbar': (P['train'], gr, P['val'], gv, pv),
            'train_to_test_moments': (M['train'], gr, M['test'], gt, pt),
        }.items():
            rep[key] = fit_eval(Ftr, gtr, Fte, gte, pte, seed)
            r = rep[key]
            rows.append({'run': rd.name, 'analysis': key, **r})
            print(f"  {key:22s} seg {r['seg_auc']:.4f}  pt {r['pt_auc']:.4f} ({r['n_pt']} pts)  "
                  f"κ {r['kappa_seg']:.4f}")

        # D: patient-level permutation null for the val→test transfer
        null = patient_permutation_null(P['val'], gv, pv, P['test'], gt, pt, n_perm, seed)
        obs = rep['val_to_test_pbar']
        rep['D_null'] = {k: {'mean': float(v.mean()), 'sd': float(v.std()),
                             'p95': float(np.quantile(v, 0.95)),
                             'p_value': float((1 + (v >= obs[k]).sum()) / (1 + len(v)))}
                         for k, v in null.items()}
        for k in ('seg_auc', 'pt_auc'):
            d = rep['D_null'][k]
            print(f"  D  null ({n_perm} val-patient perms) {k}: mean {d['mean']:.3f} ± {d['sd']:.3f}, "
                  f"95th pct {d['p95']:.3f}; observed {obs[k]:.3f} → p = {d['p_value']:.3f}")
        report['seeds'].append(rep)
        print(f'  ({time.time() - t0:.0f}s)')

    (out / 'replicate_grade.json').write_text(json.dumps(report, indent=1))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / 'replicate_grade.csv', index=False)
    except ImportError:
        pass
    print(f'\nsaved → {out}/replicate_grade.json, .csv')
    return report


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile
    import pandas as pd
    print('replicate_grade self-test: fake splits, fake old run with fake caches')
    g = torch.Generator().manual_seed(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        grade_of, pid_base = {}, 0
        for split, n_pt in (('train', 24), ('val', 12), ('test', 12)):
            none, npid, anyv, apid = [], [], [], []
            for i in range(n_pt):
                p = f'{pid_base + i:05d}'
                grade_of[p] = ['Negative', 'CIN1', 'CIN2', 'CIN3'][i % 4]
                none.append(torch.randn(50, 17, generator=g) * 0.05); npid += [p] * 50
                anyv.append(torch.randn(30, 17, generator=g) * 0.08 + 0.03 * (i % 4)); apid += [p] * 30
            pid_base += n_pt
            none, anyv = torch.cat(none), torch.cat(anyv)
            none[:, 0] = anyv[:, 0] = 0
            torch.save({'none_time_series': none, 'any_time_series': anyv,
                        'none_patient_ids': np.array(npid), 'any_patient_ids': np.array(apid)},
                       cervical.pt_path(split, tmp))
            if split == 'train':
                lo, hi = none[:, 1:].min(0).values, none[:, 1:].max(0).values
        pd.DataFrame({'GUID': [int(k) for k in grade_of], 'H': list(grade_of.values())}).to_csv(
            Path(tmp) / 'g.csv', index=False)
        grades = load_grades(Path(tmp) / 'g.csv', 'GUID', 'H')
        cfg = C.QVRConfig()
        params = C.init_params(cfg, np.random.default_rng(0))
        rd = Path(tmp) / 'q3_l3_fake_s0'
        (rd / 'training').mkdir(parents=True)
        (rd / 'unfiltered_val').mkdir()
        norm = {'norm': 'per_timepoint', 'norm_params': {'per_tp_min': lo.tolist(), 'per_tp_max': hi.tolist()}}
        torch.save({'model_state_dict': {k: v.detach() for k, v in params.items()},
                    'model_config': {'n_qubits': 3, 'n_layers': 3, 'N_E': 10, 'k': 3,
                                     'embedding_strategy': 'fixed'}, 'norm_config': norm},
                   rd / 'training' / 'trained_model.pt')
        nm = Normalizer.from_legacy(norm)
        for split in ('val', 'test'):
            X, _, _ = lesion_rows(split, tmp, grades)
            np.savez(rd / 'unfiltered_val' / f'{split}_spectral_cache_smooth.npz',
                     probs=pbar(params, cfg, nm.transform(X)))
        rep = run([rd], tmp, grades, Path(tmp) / 'out', n_perm=30)
        s = rep['seeds'][0]
        checks = {
            'B: new p̄_b equals cache': s.get('B_max_diff', 1) < 1e-12,
            'A: old-cache probe equals new-code probe': abs(s['A_old_cache']['seg_auc']
                                                           - s['val_to_test_pbar']['seg_auc']) < 1e-12,
            'C: moments ≈ p̄_b under logreg': abs(s['val_to_test_moments']['seg_auc']
                                                 - s['val_to_test_pbar']['seg_auc']) < 0.02,
            'D: null p-values in (0, 1]': all(0 < s['D_null'][k]['p_value'] <= 1 for k in s['D_null']),
            'outputs written': (Path(tmp) / 'out' / 'replicate_grade.json').exists(),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', help='old run dirs (contain training/ and unfiltered_val/)')
    ap.add_argument('--data-dir', default=str(cervical.DATA_DIR))
    ap.add_argument('--grade-csv')
    ap.add_argument('--pid-col', default='GUID')
    ap.add_argument('--grade-col', default='Overall Histology')
    ap.add_argument('--n-perm', type=int, default=1000)
    ap.add_argument('--out', default='results/explore/replicate_grade')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not (a.runs and a.grade_csv):
            ap.error('--runs and --grade-csv are required')
        run(a.runs, a.data_dir, load_grades(a.grade_csv, a.pid_col, a.grade_col), a.out, a.n_perm)