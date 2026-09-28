"""
scripts/baker_build_sets.py — rebuild Baker et al.'s released splits from large_data_sets.

Every released row (separated_data, (N, 2, 180): open, vol) is matched exactly to its
source row in large_data_sets ((N, 181) per feature; the last timepoint was dropped).
Using those indices, each output directory holds pickles in the released layout
(N, d, 180) that scripts/baker_e0.py reads with --data-dir:

  bi_train100            open, vol        released Xtr (100 normals)   → must equal the release
  bi_train782            open, vol        all normals not in Xte_norm (782)
  tri_train100           open, vol, tbba  released Xtr indices
  tri_train782           open, vol, tbba  782 normals
  bi_train782_normscale  open, vol        782 normals; global min-max refit on TRAINING
                                          normals only, then clipped to [-π, π]
                                          (exploratory: clipping is an added choice)

Val, test normals and anomaly sets are always the released rows (Xval, Xte_*).
The released data use GLOBAL min-max per feature over the pooled data; min-max is
affine-invariant, so refitting on released values equals refitting on raw data.

Usage:
    python scripts/baker_build_sets.py --out-root results/e0/baker_sets
    python scripts/baker_build_sets.py --self-test
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baker_forensics import CONDITIONS, FEATURES, load_large, load_separated  # noqa: E402

PI = math.pi


def match_indices(large, sep, tol=1e-8):
    """{set name: [(condition, row index), ...]} for exact matches (last timepoint dropped)."""
    from scipy.spatial.distance import cdist
    names, rows = [], []
    for c in CONDITIONS:
        names += [(c, i) for i in range(len(large[c]['open']))]
        rows.append(np.stack([large[c]['open'][:, :-1], large[c]['vol'][:, :-1]], -1))
    P = np.concatenate(rows).reshape(len(names), -1)
    out = {}
    for k, X in sep.items():
        S = np.transpose(X, (0, 2, 1)).reshape(len(X), -1)
        D = cdist(S, P, metric='chebyshev')
        j = D.argmin(1)
        if (D.min(1) > tol).any():
            raise RuntimeError(f'{k}: {(D.min(1) > tol).sum()} rows have no exact source row')
        out[k] = [names[m] for m in j]
    return out


def gather(large, idx, feats):
    """(n, d, 180) from (condition, index) pairs, last timepoint dropped."""
    return np.stack([np.stack([large[c][f][i, :-1] for f in feats]) for c, i in idx])


def build(repo_data, out_root):
    out_root = Path(out_root)
    large = load_large(repo_data)
    sep, tx = load_separated(repo_data)
    idx = match_indices(large, sep)
    test_norm = {i for c, i in idx['Xte_norm'] if c == 'normal'}
    train782 = [('normal', i) for i in range(len(large['normal']['open'])) if i not in test_norm]
    specs = {
        'bi_train100': (('open', 'vol'), idx['Xtr'], None),
        'bi_train782': (('open', 'vol'), train782, None),
        'tri_train100': (('open', 'vol', 'tbba'), idx['Xtr'], None),
        'tri_train782': (('open', 'vol', 'tbba'), train782, None),
        'bi_train782_normscale': (('open', 'vol'), train782, 'train_normals_global'),
    }
    manifest = {}
    for name, (feats, train_idx, scaling) in specs.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        sets = {'Xtr': gather(large, train_idx, feats)}
        for k in sep:
            if k != 'Xtr':
                sets[k] = gather(large, idx[k], feats)
        info = {'features': feats, 'n_train': len(train_idx), 'scaling': scaling or 'released (global, pooled)'}
        if scaling:
            lo = sets['Xtr'].min(axis=(0, 2), keepdims=True)[0]
            hi = sets['Xtr'].max(axis=(0, 2), keepdims=True)[0]
            clipped = {}
            for k, v in sets.items():
                z = 2 * PI * (v - lo) / (hi - lo) - PI
                clipped[k] = float(np.mean((z < -PI) | (z > PI)))
                sets[k] = np.clip(z, -PI, PI)
            info['fraction_clipped'] = clipped
        for k, v in sets.items():
            with open(d / f'{k}.pickle', 'wb') as fh:
                pickle.dump(v, fh)
        for k, v in tx.items():
            with open(d / f'{k}_transactions.pickle', 'wb') as fh:
                pickle.dump(v, fh)
        info['shapes'] = {k: list(v.shape) for k, v in sets.items()}
        if name == 'bi_train100':
            info['equals_release'] = all(np.array_equal(sets[k], sep[k]) for k in sep)
        (d / 'manifest.json').write_text(json.dumps(info, indent=1))
        manifest[name] = info
        extra = f", equals release: {info['equals_release']}" if 'equals_release' in info else ''
        clip = (f", clipped: Xtr {info['fraction_clipped']['Xtr']:.1%}, "
                f"Xte_dirty_usdt_pos {info['fraction_clipped'].get('Xte_dirty_usdt_pos', 0):.1%}") if scaling else ''
        print(f"  {name:22s} d={len(feats)} train={len(train_idx):4d}{extra}{clip}")
    (out_root / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    return manifest


def _self_test():
    import tempfile
    print('baker_build_sets self-test: fake repo, rebuild, compare with release')
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root = Path(tmp) / 'data'
        (root / 'large_data_sets').mkdir(parents=True)
        (root / 'separated_data').mkdir()
        n = {'normal': 120, 'clean_btc': 10, 'clean_usdt': 8, 'dirty_btc': 40, 'dirty_usdt': 40}
        L = {c: {f: rng.uniform(-PI, PI, (n[c], 181)) for f in FEATURES} for c in CONDITIONS}
        for c in CONDITIONS:
            for f in FEATURES:
                with open(root / 'large_data_sets' / f'{f}_{c}.pickle', 'wb') as fh:
                    pickle.dump(L[c][f], fh)
        rel = lambda c, ii: np.stack([np.stack([L[c]['open'][i, :-1], L[c]['vol'][i, :-1]]) for i in ii])  # noqa: E731
        sets = {'Xtr': rel('normal', range(30)), 'Xte_norm': rel('normal', range(30, 50)),
                'Xval': np.concatenate([rel('dirty_btc', range(10)), rel('dirty_usdt', range(10))]),
                'Xte_dirty_usdt_pos': rel('dirty_usdt', range(10, 30)), 'Xte_clean_btc': rel('clean_btc', range(10))}
        for k, v in sets.items():
            with open(root / 'separated_data' / f'{k}.pickle', 'wb') as fh:
                pickle.dump(v, fh)
        with open(root / 'separated_data' / 'Xte_dirty_usdt_pos_transactions.pickle', 'wb') as fh:
            pickle.dump(rng.normal(size=20), fh)
        m = build(root, Path(tmp) / 'sets')
        tri = pickle.load(open(Path(tmp) / 'sets' / 'tri_train100' / 'Xval.pickle', 'rb'))
        ns = pickle.load(open(Path(tmp) / 'sets' / 'bi_train782_normscale' / 'Xtr.pickle', 'rb'))
        checks = {
            'bi_train100 reproduces the release exactly': m['bi_train100']['equals_release'],
            'expanded training = all normals minus test normals': m['bi_train782']['n_train'] == 100,
            'trivariate adds tbba as channel 2': tri.shape == (20, 3, 180)
                and np.allclose(tri[:, 2], np.stack([L['dirty_btc']['tbba'][i, :-1] for i in range(10)]
                                                   + [L['dirty_usdt']['tbba'][i, :-1] for i in range(10)])),
            'normal-refit training normals span exactly [-π, π]': np.isclose(ns.min(), -PI) and np.isclose(ns.max(), PI),
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
    ap.add_argument('--out-root', default='results/e0/baker_sets')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        print(f'Building from {a.repo_data} → {a.out_root}')
        build(a.repo_data, a.out_root)