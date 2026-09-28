"""
scripts/prepare_dataset.py — convert a dataset into the standard split-directory layout
(README "Drop-in dataset layout"). One subcommand per dataset.

mitbih  — MIT-BIH Arrhythmia Database (PhysioNet mitdb 1.0.0), beat windows.
  * Inter-patient split (de Chazal et al. 2004): DS1 records for fitting, DS2 records
    for testing; paced records (102, 104, 107, 217) excluded. DS1 is further split by
    record into train / val (fixed seed). No record appears in two splits.
  * Lead chosen by name ('MLII'); records without MLII are skipped and listed.
  * Whole-record band-pass filter (zero-phase Butterworth, order 2, default 0.5–40 Hz)
    before windowing: removes baseline wander and high-frequency noise, uses no labels.
  * Beat window around each R-peak annotation: `before` + `after` samples at 360 Hz
    (default 90 + 162 = 0.25 s + 0.45 s, covering P, QRS and T), then every `downsample`-th
    sample (default 2 → 180 Hz, T = 126). The old PoC window (50 + 50) covers little more
    than the QRS complex. Single-beat windows carry no RR-interval (prematurity) information.
    AAMI classes: normal (N L R e j), supraventricular (A a J S),
    ventricular (V E), fusion (F). Q-class / unclassifiable beats excluded.
  * Normalization: per-timepoint train-normal quantiles (0.5%, 99.5%) → [-π, π], clipped
    (ANALYSIS_PLAN §2), fit on Xtr only.
  Outputs: Xtr (normal, DS1-train), Xval_norm + Xval (normal / anomalous, DS1-val),
    Xprobe (anomalous beats of DS1-train records: unseen by QVR, for E4 probes),
    Xte_norm, Xte_supraventricular, Xte_ventricular, Xte_fusion (DS2),
    plus <set>_units (record id) and <set>_subtype, and manifest.json.

synthetic — the qvr_repr PoC generator (normal flat noise; growing, falling, oscillating),
  reproduced exactly. One batch per class is generated and split by index (the original
  functions reuse one seed per call). Normalization is a FIXED affine map known from the
  generator (x · π / 1.5, clipped), not fitted — normals are pure noise, so any fit on them
  would saturate every anomaly. Growing and falling contain the same values in opposite
  order: order-invariant features (p̄_b, moments) cannot separate them by construction.

Usage:
    python scripts/prepare_dataset.py mitbih --db-dir <mitdb dir> --out data_splits/mitbih
    python scripts/prepare_dataset.py synthetic --out data_splits/synthetic
    python scripts/prepare_dataset.py --self-test
"""

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr.data import Normalizer  # noqa: E402

PI = math.pi
DS1 = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124',
       '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2 = ['100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212',
       '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']
AAMI = {'N': 'normal', 'L': 'normal', 'R': 'normal', 'e': 'normal', 'j': 'normal',
        'A': 'supraventricular', 'a': 'supraventricular', 'J': 'supraventricular',
        'S': 'supraventricular', 'V': 'ventricular', 'E': 'ventricular', 'F': 'fusion'}
ANOM = ('supraventricular', 'ventricular', 'fusion')


def units_summary(sets, fit_set='Xprobe', min_units=5):
    """Independent units per subtype in each set, and E4 eligibility per subtype:
    at least `min_units` units in the probe-fitting set AND in the evaluation (test) sets."""
    upc = {}
    for k in sets:
        if k.endswith('_subtype') and f"{k[:-8]}_units" in sets:
            base, y, u = k[:-8], sets[k], sets[f"{k[:-8]}_units"]
            upc[base] = {str(c): int(len(np.unique(u[y == c]))) for c in np.unique(y)}
    fit = upc.get(fit_set, {})
    test = {}
    for base, d in upc.items():
        if base.startswith('Xte_') and base != 'Xte_norm':
            for c, n in d.items():
                test[c] = test.get(c, 0) + n
    elig = {c: {'fit_units': fit.get(c, 0), 'test_units': test.get(c, 0),
                'eligible': fit.get(c, 0) >= min_units and test.get(c, 0) >= min_units}
            for c in sorted(set(fit) | set(test))}
    return upc, elig


def make_figures(out, sets, t=None, t_label='timepoint', n_examples=6, seed=0):
    """figures/class_means.png, examples.png, value_hist.png for Xtr, Xte_norm and every Xte_<class>."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = Path(out) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = ['Xtr', 'Xte_norm'] + sorted(k for k in sets if k.startswith('Xte_') and k != 'Xte_norm'
                                          and not k.endswith(('_units', '_subtype')))
    names = [k for k in names if k in sets]
    T = sets[names[0]].shape[-1]
    t = np.arange(T) if t is None else t
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, k in enumerate(names):
        X = sets[k][:, 0]
        m, sd = X.mean(0), X.std(0)
        ax.plot(t, m, color=colors[i % 10], label=f'{k} (n={len(X):,})', lw=1.6)
        ax.fill_between(t, m - sd, m + sd, color=colors[i % 10], alpha=0.08)
    ax.set_xlabel(t_label); ax.set_ylabel('normalized value'); ax.set_ylim(-np.pi - 0.2, np.pi + 0.2)
    ax.legend(fontsize=8); ax.set_title('Class means ± SD')
    fig.tight_layout(); fig.savefig(fig_dir / 'class_means.png', dpi=130); plt.close(fig)

    fig, axes = plt.subplots(len(names), 1, figsize=(8, 1.6 * len(names)), sharex=True, squeeze=False)
    for i, k in enumerate(names):
        X = sets[k][:, 0]
        for j in rng.choice(len(X), min(n_examples, len(X)), replace=False):
            axes[i, 0].plot(t, X[j], color=colors[i % 10], lw=0.9, alpha=0.8)
        axes[i, 0].set_ylabel(k.replace('Xte_', ''), fontsize=8); axes[i, 0].set_ylim(-np.pi - 0.2, np.pi + 0.2)
    axes[-1, 0].set_xlabel(t_label); fig.suptitle(f'{n_examples} random examples per set')
    fig.tight_layout(); fig.savefig(fig_dir / 'examples.png', dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-np.pi, np.pi, 61)
    for i, k in enumerate(names):
        ax.hist(sets[k][:, 0].ravel(), bins, density=True, histtype='step', lw=1.4, color=colors[i % 10], label=k)
    ax.set_xlabel('normalized value (all timepoints pooled)'); ax.set_ylabel('density'); ax.legend(fontsize=8)
    ax.set_title('Value distributions (what order-invariant features see)')
    fig.tight_layout(); fig.savefig(fig_dir / 'value_hist.png', dpi=130); plt.close(fig)
    return [str(fig_dir / f) for f in ('class_means.png', 'examples.png', 'value_hist.png')]


def _write(out, sets, manifest, t=None, t_label='timepoint'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    upc, elig = units_summary(sets)
    manifest['units_per_subtype'] = upc
    manifest['e4_eligibility'] = elig
    manifest['figures'] = make_figures(out, sets, t, t_label)
    hashes = {}
    for k, v in sets.items():
        with open(out / f'{k}.pickle', 'wb') as f:
            pickle.dump(v, f)
        hashes[k] = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
    manifest['sha256_16'] = hashes
    manifest['shapes'] = {k: list(np.shape(v)) for k, v in sets.items()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------------------
# MIT-BIH
# ---------------------------------------------------------------------------

def extract_record(path, seq_len, before, downsample=1, band=(0.5, 40.0)):
    import wfdb
    rec = wfdb.rdrecord(str(path))
    if 'MLII' not in rec.sig_name:
        return None
    sig = rec.p_signal[:, rec.sig_name.index('MLII')]
    if band:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, band, btype='bandpass', fs=rec.fs, output='sos')
        sig = sosfiltfilt(sos, sig)
    ann = wfdb.rdann(str(path), 'atr')
    beats, labels = [], []
    for s, sym in zip(ann.sample, ann.symbol):
        cls = AAMI.get(sym)
        if cls is None or s - before < 0 or s - before + seq_len > len(sig):
            continue
        beats.append(sig[s - before:s - before + seq_len:downsample])
        labels.append(cls)
    T = len(range(0, seq_len, downsample))
    return np.asarray(beats, dtype=np.float64).reshape(-1, T), np.asarray(labels)


def prepare_mitbih(db_dir, out, before=90, after=162, downsample=2, n_val_records=5, seed=0,
                   ds1=DS1, ds2=DS2, band=(0.5, 40.0)):
    import wfdb
    db_dir = Path(db_dir)
    seq_len = before + after
    T = len(range(0, seq_len, downsample))
    data, skipped = {}, []
    for r in ds1 + ds2:
        if not (db_dir / f'{r}.hea').exists():
            skipped.append(f'{r} (missing)')
            continue
        res = extract_record(db_dir / r, seq_len, before, downsample, band)
        if res is None:
            skipped.append(f'{r} (no MLII)')
            continue
        data[r] = res
    ds1_ok = [r for r in ds1 if r in data]
    rng = np.random.default_rng(seed)
    val_recs = sorted(rng.choice(ds1_ok, min(n_val_records, len(ds1_ok) - 1), replace=False).tolist())
    train_recs = [r for r in ds1_ok if r not in val_recs]
    test_recs = [r for r in ds2 if r in data]

    def collect(recs, classes):
        X, y, u = [], [], []
        for r in recs:
            b, lab = data[r]
            m = np.isin(lab, classes)
            X.append(b[m]); y.append(lab[m]); u.append(np.full(m.sum(), r))
        if not X:
            return np.empty((0, T)), np.array([]), np.array([])
        return np.concatenate(X), np.concatenate(y), np.concatenate(u)

    raw = {
        'Xtr': collect(train_recs, ['normal']),
        'Xval_norm': collect(val_recs, ['normal']),
        'Xval': collect(val_recs, list(ANOM)),
        'Xprobe': collect(train_recs, list(ANOM)),
        'Xte_norm': collect(test_recs, ['normal']),
    }
    for c in ANOM:
        raw[f'Xte_{c}'] = collect(test_recs, [c])
    nm = Normalizer.fit(torch.as_tensor(raw['Xtr'][0]), 'per_timepoint_q', skip_first=False, clip=False)
    sets, oor = {}, {}
    for k, (X, y, u) in raw.items():
        if len(X) == 0:
            continue
        Z = nm.transform(torch.as_tensor(X)).numpy()
        oor[k] = float(np.mean(np.abs(Z) > PI))
        sets[k] = np.clip(Z, -PI, PI)[:, None, :]
        sets[f'{k}_units'] = u
        sets[f'{k}_subtype'] = y
    nm.clip = True
    manifest = {
        'dataset': 'MIT-BIH Arrhythmia Database (PhysioNet mitdb 1.0.0)', 'source': str(db_dir),
        'wfdb': wfdb.__version__, 'preprocessing': 'band-pass (zero-phase) on whole record → window → downsample → normalize', 'lead': 'MLII (by name)', 'before_r_peak': before, 'after_r_peak': after,
        'window_s': [-before / 360, after / 360], 'downsample': downsample, 'fs_hz_effective': 360 / downsample,
        'T': T, 'bandpass_hz': list(band) if band else None, 'aami_map': AAMI, 'split': 'inter-patient (de Chazal 2004 DS1/DS2)',
        'records': {'train': train_recs, 'val': val_recs, 'test': test_recs}, 'skipped': skipped,
        'val_seed': seed, 'normalizer': nm.to_dict(), 'fraction_clipped': oor,
        'counts': {k: {c: int((v[1] == c).sum()) for c in ['normal', *ANOM] if (v[1] == c).any()}
                   for k, v in raw.items()},
    }
    _write(out, sets, manifest, t=np.arange(T) * downsample / 360 - before / 360, t_label='time from R-peak (s)')
    return manifest


# ---------------------------------------------------------------------------
# Synthetic (qvr_repr PoC generator, reproduced)
# ---------------------------------------------------------------------------

SYN_SPLITS = {'normal': {'Xtr': 1000, 'Xval_norm': 200, 'Xte_norm': 300},
              'anomaly': {'Xval': 100, 'Xprobe': 200, 'Xte': 200}}


def _gen(cls, n, seq_len, rng):
    if cls == 'normal':
        return rng.normal(0.0, 0.1, (n, seq_len))
    noise = rng.normal(0.0, 0.05, (n, seq_len))
    if cls == 'growing':
        return np.tile(np.linspace(0.0, 1.0, seq_len), (n, 1)) + noise
    if cls == 'falling':
        return np.tile(np.linspace(1.0, 0.0, seq_len), (n, 1)) + noise
    if cls == 'oscillating':
        return np.tile(np.sin(np.linspace(0.0, 4.0 * np.pi, seq_len)), (n, 1)) + noise
    raise ValueError(cls)


def prepare_synthetic(out, seq_len=16, seed=1989, scale=PI / 1.5):
    rng = np.random.default_rng(seed)
    anoms = ('growing', 'falling', 'oscillating')
    sets, counts = {}, {}
    Xn = _gen('normal', sum(SYN_SPLITS['normal'].values()), seq_len, rng)
    i = 0
    for k, n in SYN_SPLITS['normal'].items():
        sets[k], i = Xn[i:i + n], i + n
        sets[f'{k}_subtype'] = np.array(['normal'] * n)
    parts = {'Xval': [], 'Xprobe': []}
    for c in anoms:
        Xa = _gen(c, sum(SYN_SPLITS['anomaly'].values()), seq_len, rng)
        a, b = SYN_SPLITS['anomaly']['Xval'], SYN_SPLITS['anomaly']['Xprobe']
        parts['Xval'].append((Xa[:a], c))
        parts['Xprobe'].append((Xa[a:a + b], c))
        sets[f'Xte_{c}'] = Xa[a + b:]
        sets[f'Xte_{c}_subtype'] = np.array([c] * (len(Xa) - a - b))
    for k, lst in parts.items():
        sets[k] = np.concatenate([x for x, _ in lst])
        sets[f'{k}_subtype'] = np.concatenate([[c] * len(x) for x, c in lst])
    oor = {}
    for k in list(sets):
        if k.endswith('_subtype'):
            continue
        z = sets[k] * scale
        oor[k] = float(np.mean(np.abs(z) > PI))
        sets[k] = np.clip(z, -PI, PI)[:, None, :]
        sets[f'{k}_units'] = np.arange(len(z)).astype(str)          # each series independent
        counts[k] = len(z)
    manifest = {'dataset': 'synthetic (qvr_repr PoC generator)', 'seq_len': seq_len, 'seed': seed,
                'classes': {'normal': 'N(0, 0.1) flat', 'growing': 'ramp 0→1 + N(0, 0.05)',
                            'falling': 'ramp 1→0 + N(0, 0.05)', 'oscillating': 'sin, 2 periods + N(0, 0.05)'},
                'normalization': f'fixed affine x·{scale:.6f} (π/1.5), clipped to [-π, π]; not fitted',
                'fraction_clipped': oor, 'counts': counts,
                'note': 'growing and falling share the same multiset of values: order-invariant features cannot separate them'}
    _write(out, sets, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile
    import wfdb
    from qvr.datasets import baker as splitdir
    print('prepare_dataset self-test: fake mitdb records (wfdb) + synthetic')
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / 'mitdb'
        db.mkdir()
        ds1, ds2 = ['101', '106', '108', '114'], ['100', '103']
        for r in ds1 + ds2 + ['999']:
            n = 360 * 30
            sig = rng.normal(0, 0.05, (n, 2))
            peaks = np.arange(200, n - 200, 300)
            syms = rng.choice(['N', 'N', 'N', 'V', 'A', 'F', 'Q'], len(peaks)).tolist()
            for p, s in zip(peaks, syms):
                sig[p - 5:p + 5, 0 if r != '114' else 1] += {'N': 1.0, 'V': 2.0, 'A': 1.3, 'F': 1.6, 'Q': 0.5}[s]
            names = ['MLII', 'V1'] if r != '114' else ['V5', 'MLII']
            if r == '106':
                names = ['V1', 'V2']                                   # no MLII → skipped
            wfdb.wrsamp(r, fs=360, units=['mV', 'mV'], sig_name=names, p_signal=sig,
                        fmt=['212', '212'], write_dir=str(db))
            wfdb.wrann(r, 'atr', sample=peaks, symbol=syms, write_dir=str(db))
        m = prepare_mitbih(db, Path(tmp) / 'mit', n_val_records=1, ds1=ds1, ds2=ds2)
        rec = m['records']
        check('record 106 without MLII skipped', any(s.startswith('106') for s in m['skipped']), str(m['skipped']))
        check('no record in two splits', not (set(rec['train']) & set(rec['val']) | set(rec['train']) & set(rec['test'])
                                               | set(rec['val']) & set(rec['test'])), str(rec))
        tr_u = pickle.load(open(Path(tmp) / 'mit' / 'Xtr_units.pickle', 'rb'))
        te_u = pickle.load(open(Path(tmp) / 'mit' / 'Xte_norm_units.pickle', 'rb'))
        check('train and test units disjoint', not (set(tr_u) & set(te_u)))
        check('Q beats excluded', all('Q' not in v for v in m['counts'].values()))
        d = splitdir.load_all(Path(tmp) / 'mit')
        check('loader reads split dir (T = 126)', d['Xtr'].shape[1:] == (126, 1) and 'Xte_ventricular' in d,
              f"Xtr {tuple(d['Xtr'].shape)}")
        check('training normals within [-π, π]', float(d['Xtr'].abs().max()) <= PI + 1e-12)
        check('MLII picked from channel 1 for record 114',
              '114' in rec['train'] + rec['val'] and m['counts']['Xtr'].get('normal', 0) > 0)

        s = prepare_synthetic(Path(tmp) / 'syn')
        ds = splitdir.load_all(Path(tmp) / 'syn')
        g, f = ds['Xte_growing'][:, :, 0], ds['Xte_falling'][:, :, 0]
        check('synthetic sets present', {'Xtr', 'Xte_norm', 'Xte_growing', 'Xte_falling', 'Xte_oscillating',
                                         'Xval', 'Xprobe'} <= set(ds), str(sorted(ds)))
        check('growing and falling are mirror images in time (means)',
              torch.allclose(g.mean(0), f.mean(0).flip(0), atol=0.05))
        check('train and test normals are different draws', not torch.equal(ds['Xtr'][:5], ds['Xte_norm'][:5]))
        check('figures written (both datasets)', all(Path(f).exists() for f in m['figures'] + s['figures']))
        check('E4 eligibility computed per subtype', set(m['e4_eligibility']) == {'fusion', 'supraventricular', 'ventricular'},
              str({c: v['eligible'] for c, v in m['e4_eligibility'].items()}))
        check('fixed scaling: nothing clipped', max(s['fraction_clipped'].values()) == 0.0,
              f"max {max(s['fraction_clipped'].values()):.3f}")
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='?', choices=['mitbih', 'synthetic'])
    ap.add_argument('--db-dir')
    ap.add_argument('--out')
    ap.add_argument('--seq-len', type=int, help='synthetic only')
    ap.add_argument('--before', type=int, default=90, help='mitbih: samples before R (360 Hz)')
    ap.add_argument('--after', type=int, default=162, help='mitbih: samples after R (360 Hz)')
    ap.add_argument('--downsample', type=int, default=2, help='mitbih: keep every k-th sample')
    ap.add_argument('--bandpass', type=float, nargs=2, default=[0.5, 40.0], help='mitbih: Hz; 0 0 disables')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    elif a.dataset == 'mitbih':
        if not (a.db_dir and a.out):
            ap.error('mitbih needs --db-dir and --out')
        band = None if a.bandpass == [0.0, 0.0] else tuple(a.bandpass)
        m = prepare_mitbih(a.db_dir, a.out, before=a.before, after=a.after, downsample=a.downsample, band=band)
        print(json.dumps({k: m[k] for k in ('window_s', 'T', 'records', 'skipped', 'counts', 'fraction_clipped',
                                            'units_per_subtype', 'e4_eligibility', 'figures')}, indent=1))
    elif a.dataset == 'synthetic':
        m = prepare_synthetic(a.out or 'data_splits/synthetic', seq_len=a.seq_len or 16)
        print(json.dumps({k: m[k] for k in ('counts', 'fraction_clipped', 'e4_eligibility', 'figures', 'note')}, indent=1))
    else:
        ap.error('choose a dataset or --self-test')