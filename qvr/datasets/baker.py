"""
qvr/datasets/baker.py — Baker et al. (arXiv:2210.16438) released cryptocurrency arrays.

Pickles in `<repo>/data/separated_data/` (Zenodo 7258627), already rescaled per time
point to [−π, π] by the authors (their SM Eq. S2). Layout on disk: (N, d, T) with
d = 2 (bivariate: 0 = mean deviation of open price, 1 = cumulative volume), T = 180.

Returned as float64 tensors (N, T, d) — our backend's layout.

  Xtr         normal training windows
  Xval        mixed anomalous validation windows (threshold tuning in the paper)
  Xte_norm    normal test windows (used for BOTH threshold tuning and testing in the paper)
  Xte_<name>  anomalous test sets (e.g. dirty_usdt_pos = Ũ₊)

Self-test:  python -m qvr.datasets.baker
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path('/scratch90/chris_/quantum/QuantumVariationalRewinding/data/separated_data')


def _load(path: Path) -> torch.Tensor:
    with open(path, 'rb') as f:
        a = np.asarray(pickle.load(f), dtype=np.float64)
    if a.ndim != 3:
        raise ValueError(f'{path.name}: expected (N, d, T), got {a.shape}')
    return torch.as_tensor(a).permute(0, 2, 1).contiguous()


def load_all(data_dir: Path = DATA_DIR) -> dict[str, torch.Tensor]:
    """{'Xtr', 'Xval', 'Xte_norm', 'Xte_<set>'...} → (N, T, d) float64. Non-3D files skipped."""
    data_dir = Path(data_dir)
    out = {}
    for p in sorted(data_dir.glob('X*.pickle')):
        try:
            out[p.stem] = _load(p)
        except ValueError:
            continue
    for req in ('Xtr', 'Xval', 'Xte_norm'):
        if req not in out:
            raise FileNotFoundError(f'{req}.pickle missing in {data_dir}; found {sorted(out)}')
    return out


def anomaly_sets(data: dict) -> list[str]:
    return [k for k in data if k.startswith('Xte_') and k != 'Xte_norm']


def _self_test() -> None:
    import tempfile
    print('=' * 64)
    print('qvr.datasets.baker self-test (fake pickles, real layout)')
    print('=' * 64)
    ok = True
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for name, n in (('Xtr', 100), ('Xval', 60), ('Xte_norm', 60), ('Xte_dirty_usdt_pos', 60)):
            with open(Path(tmp) / f'{name}.pickle', 'wb') as f:
                pickle.dump(rng.uniform(-np.pi, np.pi, (n, 2, 180)), f)
        with open(Path(tmp) / 'Xval_transactions.pickle', 'wb') as f:
            pickle.dump(rng.normal(size=60), f)
        d = load_all(tmp)
        checks = {
            'loads 3-D sets, skips metadata': sorted(d) == ['Xte_dirty_usdt_pos', 'Xte_norm', 'Xtr', 'Xval'],
            'layout (N, T, d)': d['Xtr'].shape == (100, 180, 2) and d['Xtr'].dtype == torch.float64,
            'anomaly_sets': anomaly_sets(d) == ['Xte_dirty_usdt_pos'],
        }
        for k, v in checks.items():
            ok &= v
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()