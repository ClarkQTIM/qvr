"""
qvr/datasets/cervical.py — DYSIS cervical acetowhitening adapter.

Reads the holdout-experiment .pt splits from qvr_cervix. These contain the
SMOOTHNESS-FILTERED segments only (|step| < 0.10), already preprocessed
(÷255 → baseline-subtract → filter → EWMA span 5). Frame 0 is 0.

    none_time_series  (N_none, 17)  none-voted  → y = 0 (normal)
    any_time_series   (N_any, 17)   any-voted   → y = 1 (lesion)
    none_patient_ids / any_patient_ids          → unit (patient)

The unfiltered population (jagged segments included; used for the old paper's
Table 1 detection numbers) comes from the PKL path and is NOT loaded here,
pending the voting-alignment check.

Self-test:  python -m qvr.datasets.cervical   (uses a fake .pt; no real data needed)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qvr.data import SeriesSet

DATA_DIR = Path('/scratch90/chris_/quantum/qvr_cervix/data/cervix/holdout_experiment')
SPLITS = ('train', 'val', 'test')
REQUIRED = ('none_time_series', 'any_time_series', 'none_patient_ids', 'any_patient_ids')


def pt_path(split: str, data_dir: Path = DATA_DIR) -> Path:
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    return Path(data_dir) / f'cervical_{split}.pt'


def _load(path: Path) -> dict:
    d = torch.load(path, map_location='cpu', weights_only=False)
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise KeyError(f'{path}: missing keys {missing}; found {sorted(d)}')
    return d


def load_split(split: str, data_dir: Path = DATA_DIR, normals_only: bool = False) -> SeriesSet:
    d = _load(pt_path(split, data_dir))
    Xn = torch.as_tensor(d['none_time_series'], dtype=torch.float64)
    un = np.asarray(d['none_patient_ids']).astype(str)
    if normals_only:
        return SeriesSet(Xn, np.zeros(len(Xn), int), un, name=f'cervical_{split}_normals')
    Xa = torch.as_tensor(d['any_time_series'], dtype=torch.float64)
    ua = np.asarray(d['any_patient_ids']).astype(str)
    return SeriesSet(torch.cat([Xn, Xa]),
                     np.r_[np.zeros(len(Xn), int), np.ones(len(Xa), int)],
                     np.r_[un, ua], name=f'cervical_{split}')


def _self_test() -> None:
    import tempfile

    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    print('=' * 64)
    print('qvr.datasets.cervical self-test (fake .pt with the real key layout)')
    print('=' * 64)
    g = torch.Generator().manual_seed(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for split, (nn, na) in {'train': (500, 20), 'val': (200, 10), 'test': (300, 15)}.items():
            none = torch.randn(nn, 17, generator=g) * 0.05
            anyv = torch.randn(na, 17, generator=g) * 0.10
            none[:, 0] = anyv[:, 0] = 0
            torch.save({'none_time_series': none, 'any_time_series': anyv,
                        'none_patient_ids': np.array([f'{i % 7:05d}' for i in range(nn)]),
                        'any_patient_ids': np.array([f'{i % 3:05d}' for i in range(na)]),
                        'per_patient_summary': {}}, pt_path(split, tmp))
        s = load_split('val', tmp)
        print('  ' + s.summary())
        check('val N', len(s.y) == 210, f'{len(s.y)}')
        check('labels: normals first, then lesions', s.y[:200].sum() == 0 and s.y[200:].all())
        check('X float64 (N, 17)', s.X.dtype == torch.float64 and s.X.shape == (210, 17))
        check('units are patient-id strings', s.unit.dtype.kind == 'U', f'{s.unit[:3]}')
        tr = load_split('train', tmp, normals_only=True)
        check('train normals only', len(tr.y) == 500 and tr.y.sum() == 0)
        bad = Path(tmp) / 'cervical_test.pt'
        torch.save({'none_time_series': torch.zeros(2, 17)}, bad)
        try:
            load_split('test', tmp)
            check('missing keys raise', False)
        except KeyError as e:
            check('missing keys raise', True, str(e).split(';')[0][-40:])
    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()