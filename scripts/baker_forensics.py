"""
scripts/baker_forensics.py — what is in Baker et al.'s data, and why detection works.

No training. Four analyses:

  1  Scaling reference population. The README says all data were rescaled per the
     SM (per-timepoint min-max to [-π, π] across instances). For each feature and
     timepoint: does the POOLED data (all conditions) hit exactly -π and +π, while
     each condition alone does not? If so, anomalous windows set the scale.

  2  Split provenance. Match every row of separated_data/X*.pickle (N, 2, 180)
     against large_data_sets (N, 181 per feature): which condition and index each
     row came from, which timepoint was dropped, channel order, and whether
     X (train), N (test normals), V (validation) and the test sets are disjoint.

  3  Aliasing. RX/RY embeddings are 2π-periodic, so a value near +π embeds almost
     like one near -π. Per condition: fraction of values near +π, and how many
     anomalous windows are closer to the normal region after wraparound than
     before. With --e0-json: QVR vs the one-number baseline, AUC within strata
     of anomaly extremeness (window-mean volume).

  4  Transactions. Rank correlation of |transaction value| with the QVR score and
     with the baseline score, and AUC within strata of |transaction value|
     (the paper's Fig. 3 claim: detection rises with transaction value).

Usage:
    python scripts/baker_forensics.py \
        --repo-data /scratch90/chris_/quantum/QuantumVariationalRewinding/data \
        --e0-json results/e0/baker_paper_ry/baker_e0.json --variant paper_ry \
        --out results/e0/baker_forensics
    python scripts/baker_forensics.py --self-test
"""

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qvr.data import roc_auc                                          # noqa: E402

FEATURES = ('open', 'vol', 'tbba')
CONDITIONS = ('normal', 'clean_btc', 'clean_usdt', 'dirty_btc', 'dirty_usdt')
PI = math.pi


def _load(p):
    with open(p, 'rb') as f:
        return np.asarray(pickle.load(f), dtype=np.float64)


def load_large(repo_data):
    d = Path(repo_data) / 'large_data_sets'
    return {c: {f: _load(d / f'{f}_{c}.pickle') for f in FEATURES} for c in CONDITIONS}


def load_separated(repo_data):
    d = Path(repo_data) / 'separated_data'
    X = {p.stem: _load(p) for p in sorted(d.glob('X*.pickle')) if 'transactions' not in p.stem}
    tx = {p.stem.replace('_transactions', ''): _load(p) for p in sorted(d.glob('*_transactions.pickle'))}
    return X, tx


# ---------------------------------------------------------------------------
# 1. scaling reference population
# ---------------------------------------------------------------------------

def scaling_audit(large, tol=1e-3):
    out = {}
    for f in FEATURES:
        pooled = np.concatenate([large[c][f] for c in CONDITIONS])
        lo, hi = pooled.min(0), pooled.max(0)
        per = {c: {'min': float(large[c][f].min()), 'max': float(large[c][f].max()),
                   'frac_t_reaching_+pi': float(np.mean(np.abs(large[c][f].max(0) - PI) < tol)),
                   'frac_t_reaching_-pi': float(np.mean(np.abs(large[c][f].min(0) + PI) < tol))}
               for c in CONDITIONS}
        out[f] = {'pooled_frac_t_min_is_-pi': float(np.mean(np.abs(lo + PI) < tol)),
                  'pooled_frac_t_max_is_+pi': float(np.mean(np.abs(hi - PI) < tol)),
                  'n_timepoints': int(pooled.shape[1]), 'per_condition': per}
    return out


# ---------------------------------------------------------------------------
# 2. split provenance
# ---------------------------------------------------------------------------

def provenance(large, sep, tol=1e-8):
    from scipy.spatial.distance import cdist
    names, rows = [], []
    for c in CONDITIONS:
        names += [(c, i) for i in range(len(large[c]['open']))]
        rows.append(np.stack([large[c]['open'], large[c]['vol']], -1))      # (N, 181, 2)
    pool = np.concatenate(rows)
    best = None
    for drop in ('first', 'last'):
        P = (pool[:, 1:] if drop == 'first' else pool[:, :-1]).reshape(len(pool), -1)
        res = {}
        for k, X in sep.items():
            S = np.transpose(X, (0, 2, 1)).reshape(len(X), -1)              # (n, 180*2) open,vol
            D = cdist(S, P, metric='chebyshev')
            j = D.argmin(1)
            res[k] = {'min_dist': D.min(1), 'match': [names[m] for m in j]}
        n_exact = sum(int((r['min_dist'] < tol).sum()) for r in res.values())
        if best is None or n_exact > best[0]:
            best = (n_exact, drop, res)
    n_exact, drop, res = best
    out = {'dropped_timepoint': drop, 'n_rows': int(sum(len(v) for v in sep.values())),
           'n_exact_matches': n_exact, 'sets': {}}
    idx = {}
    for k, r in res.items():
        exact = r['min_dist'] < tol
        conds = {}
        for (c, i), e in zip(r['match'], exact):
            if e:
                conds[c] = conds.get(c, 0) + 1
        idx[k] = {(c, i) for (c, i), e in zip(r['match'], exact) if e}
        out['sets'][k] = {'n': int(len(exact)), 'exact': int(exact.sum()),
                          'max_residual': float(r['min_dist'].max()), 'from_conditions': conds}
    keys = sorted(idx)
    out['overlaps'] = {f'{a}&{b}': len(idx[a] & idx[b]) for i, a in enumerate(keys)
                       for b in keys[i + 1:] if idx[a] & idx[b]}
    # channel order check: channel 1 should be volume (normal vol sits near -π)
    return out


# ---------------------------------------------------------------------------
# 3. aliasing
# ---------------------------------------------------------------------------

def _circ(a, b):
    d = np.abs(a - b) % (2 * PI)
    return np.minimum(d, 2 * PI - d)


def aliasing_audit(large, near=0.5, gap=1.0):
    out = {}
    for f in FEATURES:
        ref = np.median(large['normal'][f], axis=0)                          # per-timepoint normal centre
        out[f] = {}
        for c in CONDITIONS:
            A = large[c][f]
            lin = np.abs(A - ref).mean(1)
            circ = _circ(A, ref).mean(1)
            out[f][c] = {'frac_values_within_{}_of_+pi'.format(near): float(np.mean(A > PI - near)),
                         'median_linear_dist_to_normal': float(np.median(lin)),
                         'median_circular_dist_to_normal': float(np.median(circ)),
                         f'frac_windows_aliased_(linear-circular>{gap})': float(np.mean(lin - circ > gap))}
    return out


# ---------------------------------------------------------------------------
# 3b / 4. QVR vs baseline by strata (needs a saved E0 model)
# ---------------------------------------------------------------------------

def _strata_auc(s_norm, s_anom, key, n_bins=3):
    edges = np.quantile(key, np.linspace(0, 1, n_bins + 1))
    res = []
    for b in range(n_bins):
        m = (key >= edges[b]) & (key <= edges[b + 1]) if b == n_bins - 1 else (key >= edges[b]) & (key < edges[b + 1])
        y = np.r_[np.zeros(len(s_norm)), np.ones(int(m.sum()))]
        res.append({'bin': b, 'range': [float(edges[b]), float(edges[b + 1])], 'n': int(m.sum()),
                    'auc': roc_auc(y, np.r_[s_norm, s_anom[m]])})
    return res


def model_strata(e0_json, variant, sep, tx):
    from scipy.stats import spearmanr
    from qvr import circuit as C
    from baker_e0 import baseline_scores, qvr_scores
    rep = json.loads(Path(e0_json).read_text())
    q = rep['qvr'][variant]
    run = q['runs'][q['best_run']]
    cfg = C.QVRConfig(**q['cfg'])
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in run['params'].items()}
    data = {k: torch.as_tensor(v).permute(0, 2, 1).contiguous() for k, v in sep.items()}   # (N, T, d)
    sets = [k for k in data if k.startswith('Xte_') and k != 'Xte_norm']
    qs = {k: qvr_scores(p, cfg, data[k], run['hist'])[q['score_key']] for k in ['Xte_norm'] + sets}
    bs = baseline_scores(data, 'vol_mean')
    dirty = [k for k in sets if 'dirty' in k]
    anom_q = np.concatenate([qs[k] for k in dirty])
    anom_b = np.concatenate([bs[k] for k in dirty])
    vol_level = np.concatenate([data[k][:, ::4, 1].mean(1).numpy() for k in dirty])
    out = {'variant': variant, 'score_key': q['score_key'],
           'extremeness_strata': {'qvr': _strata_auc(qs['Xte_norm'], anom_q, vol_level),
                                  'vol_mean': _strata_auc(bs['Xte_norm'], anom_b, vol_level)},
           'transactions': {}}
    tx_sets = [k for k in sets if k in tx and len(tx[k]) == len(qs[k])]
    if tx_sets:
        v = np.abs(np.concatenate([tx[k] for k in tx_sets]))
        aq = np.concatenate([qs[k] for k in tx_sets])
        ab = np.concatenate([bs[k] for k in tx_sets])
        out['transactions'] = {
            'sets': tx_sets,
            'spearman_abs_value_vs_qvr': float(spearmanr(v, aq).statistic),
            'spearman_abs_value_vs_vol_mean': float(spearmanr(v, ab).statistic),
            'value_strata': {'qvr': _strata_auc(qs['Xte_norm'], aq, v),
                             'vol_mean': _strata_auc(bs['Xte_norm'], ab, v)}}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(repo_data, out, e0_json=None, variant='paper_ry'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    large = load_large(repo_data)
    sep, tx = load_separated(repo_data)
    rep = {'sizes': {c: int(len(large[c]['open'])) for c in CONDITIONS}}
    print('=' * 78)
    print('Baker data forensics — large sets: ' + ', '.join(f'{c} {n}' for c, n in rep['sizes'].items()))
    print('=' * 78)

    rep['scaling'] = s = scaling_audit(large)
    print('\n1  Scaling reference population (per-timepoint min-max to [-π, π])')
    for f in FEATURES:
        print(f"  {f:5s} pooled: min=-π at {s[f]['pooled_frac_t_min_is_-pi']:.0%} of t, "
              f"max=+π at {s[f]['pooled_frac_t_max_is_+pi']:.0%} of t")
        for c in CONDITIONS:
            pc = s[f]['per_condition'][c]
            print(f"        {c:11s} range [{pc['min']:+.3f}, {pc['max']:+.3f}]   reaches +π at "
                  f"{pc['frac_t_reaching_+pi']:.0%} of t, -π at {pc['frac_t_reaching_-pi']:.0%}")

    rep['provenance'] = pv = provenance(large, sep)
    print(f"\n2  Split provenance — dropped timepoint: {pv['dropped_timepoint']}; "
          f"exact matches {pv['n_exact_matches']}/{pv['n_rows']} rows (channels assumed open, vol)")
    for k, v in pv['sets'].items():
        print(f"  {k:20s} {v['exact']:3d}/{v['n']:3d} exact  from {v['from_conditions']}  (max residual {v['max_residual']:.1e})")
    print('  overlaps between released sets: ' + (json.dumps(pv['overlaps']) if pv['overlaps'] else 'none'))

    rep['aliasing'] = al = aliasing_audit(large)
    print('\n3  Aliasing (circular distance to the normal per-timepoint median)')
    for f in ('vol', 'tbba'):
        for c in CONDITIONS[1:]:
            a = al[f][c]
            k_near = [x for x in a if x.startswith('frac_values')][0]
            k_al = [x for x in a if x.startswith('frac_windows')][0]
            print(f"  {f:5s} {c:11s} values near +π {a[k_near]:6.1%} | window dist linear "
                  f"{a['median_linear_dist_to_normal']:.2f} vs circular {a['median_circular_dist_to_normal']:.2f} | "
                  f"aliased windows {a[k_al]:6.1%}")

    if e0_json:
        rep['model'] = m = model_strata(e0_json, variant, sep, tx)
        print(f"\n3b QVR ({m['variant']}, best run) vs one-number baseline — AUC by anomaly extremeness "
              '(window-mean volume tertile, pooled dirty test sets vs Xte_norm)')
        for b_q, b_b in zip(m['extremeness_strata']['qvr'], m['extremeness_strata']['vol_mean']):
            print(f"  tertile {b_q['bin']} (vol {b_q['range'][0]:+.2f}..{b_q['range'][1]:+.2f}, n={b_q['n']}): "
                  f"QVR {b_q['auc']:.3f}   vol_mean {b_b['auc']:.3f}")
        t = m['transactions']
        if t:
            print(f"\n4  Transactions ({len(t['sets'])} sets): Spearman(|value|, score) QVR "
                  f"{t['spearman_abs_value_vs_qvr']:+.3f}, vol_mean {t['spearman_abs_value_vs_vol_mean']:+.3f}")
            for b_q, b_b in zip(t['value_strata']['qvr'], t['value_strata']['vol_mean']):
                print(f"  |value| tertile {b_q['bin']} ({b_q['range'][0]:.2e}..{b_q['range'][1]:.2e}, n={b_q['n']}): "
                      f"QVR {b_q['auc']:.3f}   vol_mean {b_b['auc']:.3f}")
    (out / 'baker_forensics.json').write_text(json.dumps(rep, indent=1, default=str))
    print(f'\nsaved → {out}/baker_forensics.json')
    return rep


def _self_test():
    import tempfile
    from baker_e0 import run as e0_run
    print('baker_forensics self-test: fake repo with pooled scaling and subset splits')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / 'data'
        (root / 'large_data_sets').mkdir(parents=True)
        (root / 'separated_data').mkdir()
        n = {'normal': 200, 'clean_btc': 34, 'clean_usdt': 19, 'dirty_btc': 150, 'dirty_usdt': 150}
        shift = {'normal': 0, 'clean_btc': 0.3, 'clean_usdt': 0.3, 'dirty_btc': 2.0, 'dirty_usdt': 3.0}
        raw = {c: {f: np.cumsum(rng.gamma(1 + shift[c], 1, (n[c], 181)), 1) for f in FEATURES} for c in CONDITIONS}
        scaled = {c: {} for c in CONDITIONS}
        for f in FEATURES:
            pooled = np.concatenate([raw[c][f] for c in CONDITIONS])
            lo, hi = pooled.min(0), pooled.max(0)
            for c in CONDITIONS:
                scaled[c][f] = 2 * PI * (raw[c][f] - lo) / (hi - lo) - PI
                with open(root / 'large_data_sets' / f'{f}_{c}.pickle', 'wb') as fh:
                    pickle.dump(scaled[c][f], fh)
        def sep_rows(c, idx):
            return np.stack([scaled[c]['open'][idx, 1:], scaled[c]['vol'][idx, 1:]], 1)   # (n, 2, 180)
        perm = rng.permutation(200)
        sets = {'Xtr': sep_rows('normal', perm[:80]), 'Xte_norm': sep_rows('normal', perm[80:120]),
                'Xval': np.concatenate([sep_rows('dirty_btc', np.arange(20)), sep_rows('dirty_usdt', np.arange(20))]),
                'Xte_dirty_usdt_pos': sep_rows('dirty_usdt', np.arange(20, 60)),
                'Xte_dirty_btc_pos': sep_rows('dirty_btc', np.arange(20, 60)),
                'Xte_clean_btc': sep_rows('clean_btc', np.arange(34))}
        for k, v in sets.items():
            with open(root / 'separated_data' / f'{k}.pickle', 'wb') as fh:
                pickle.dump(v, fh)
        for k in ('Xte_dirty_usdt_pos', 'Xte_dirty_btc_pos', 'Xte_clean_btc'):
            with open(root / 'separated_data' / f'{k}_transactions.pickle', 'wb') as fh:
                pickle.dump(rng.lognormal(18, 1, len(sets[k])), fh)
        e0_run(root / 'separated_data', Path(tmp) / 'e0', seeds=1, restarts=2, maxfev_nb=30,
               maxfev_paper=30, n_boot=20, variants=('paper',))
        rep = run(root, Path(tmp) / 'out', e0_json=Path(tmp) / 'e0' / 'baker_e0.json', variant='paper')
        pv = rep['provenance']
        checks = {
            'pooled scaling detected (vol hits ±π only when pooled)':
                rep['scaling']['vol']['pooled_frac_t_max_is_+pi'] > 0.9
                and rep['scaling']['vol']['per_condition']['normal']['frac_t_reaching_+pi'] < 0.1,
            'dropped timepoint = first': pv['dropped_timepoint'] == 'first',
            'all rows matched exactly': pv['n_exact_matches'] == pv['n_rows'],
            'Xval traced to dirty conditions': set(pv['sets']['Xval']['from_conditions']) == {'dirty_btc', 'dirty_usdt'},
            'train/test normals disjoint': 'Xte_norm&Xtr' not in pv['overlaps'],
            'strata + transaction analyses ran': len(rep['model']['extremeness_strata']['qvr']) == 3
                                                 and bool(rep['model']['transactions']),
        }
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        ok = all(checks.values())
        print('RESULT:', 'ALL PASS' if ok else 'FAIL')
        if not ok:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-data', default='/scratch90/chris_/quantum/QuantumVariationalRewinding/data')
    ap.add_argument('--e0-json', default=None)
    ap.add_argument('--variant', default='paper_ry')
    ap.add_argument('--out', default='results/e0/baker_forensics')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        run(a.repo_data, a.out, a.e0_json, a.variant)