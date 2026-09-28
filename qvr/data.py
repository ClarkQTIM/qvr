"""
qvr/data.py — dataset-agnostic data layer.

SeriesSet   raw series + labels + independent-unit ids (patient, record, window)
Normalizer  fit on TRAINING NORMALS only; transform any split. Reports how many
            values land outside [-π, π] (p_b is 2π-periodic, so those alias).
t_grid      the time grid fed to D(ε, t): linspace(0, 2π, T), as in qvr_cervix.
roc_auc     rank-based AUC (ties averaged), no sklearn needed.

Dataset adapters (qvr/datasets/*.py) only have to return SeriesSets.

Normalizer kinds
  'per_timepoint'  legacy-exact qvr_cervix: each frame t≥1 mapped with its own
                   train min/max to [-π, π]; frame 0 left as-is (0 after
                   baseline subtraction). Use from_legacy() for old checkpoints.
  'per_timepoint_q' same, but with train quantiles (lo_q, hi_q) instead of
                   min/max, and optional clipping — for new datasets where
                   outliers would otherwise compress the range.

Self-test:  python -m qvr.data
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import rankdata

DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class SeriesSet:
    X: torch.Tensor                 # (N, T) raw (un-normalized) series, float64
    y: np.ndarray                   # (N,) int: 0 = normal, 1 = anomalous (task-level)
    unit: np.ndarray                # (N,) independent unit id (patient / record / window)
    name: str = ''
    extra: dict = field(default_factory=dict)   # optional per-series arrays (e.g. subtype)

    def __post_init__(self):
        self.X = torch.as_tensor(self.X, dtype=DTYPE)
        self.y = np.asarray(self.y).astype(int)
        self.unit = np.asarray(self.unit)
        n = self.X.shape[0]
        if len(self.y) != n or len(self.unit) != n:
            raise ValueError(f'{self.name}: X has {n} rows, y {len(self.y)}, unit {len(self.unit)}')
        for k, v in self.extra.items():
            if len(v) != n:
                raise ValueError(f'{self.name}: extra[{k!r}] has {len(v)} rows, expected {n}')

    @property
    def T(self) -> int:
        return self.X.shape[1]

    def normals(self) -> torch.Tensor:
        return self.X[self.y == 0]

    def subset(self, mask) -> 'SeriesSet':
        mask = np.asarray(mask)
        return SeriesSet(self.X[torch.as_tensor(mask)], self.y[mask], self.unit[mask],
                         self.name, {k: np.asarray(v)[mask] for k, v in self.extra.items()})

    def summary(self) -> str:
        n1 = int(self.y.sum())
        return (f'{self.name}: N={len(self.y):,} (normal {len(self.y) - n1:,}, anomalous {n1:,}), '
                f'T={self.T}, units={len(np.unique(self.unit)):,}')


def t_grid(T: int) -> torch.Tensor:
    return torch.linspace(0, 2 * math.pi, T, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    def __init__(self, kind: str, lo=None, hi=None, skip_first: bool = True,
                 clip: bool = False, q: tuple[float, float] | None = None):
        self.kind, self.skip_first, self.clip, self.q = kind, skip_first, clip, q
        self.lo = None if lo is None else torch.as_tensor(lo, dtype=DTYPE)
        self.hi = None if hi is None else torch.as_tensor(hi, dtype=DTYPE)

    @classmethod
    def fit(cls, X_train_normals: torch.Tensor, kind: str = 'per_timepoint',
            skip_first: bool = True, clip: bool = False,
            q: tuple[float, float] = (0.005, 0.995)) -> 'Normalizer':
        X = torch.as_tensor(X_train_normals, dtype=DTYPE)
        Xs = X[:, 1:] if skip_first else X
        if kind == 'per_timepoint':
            return cls(kind, Xs.min(0).values, Xs.max(0).values, skip_first, clip)
        if kind == 'per_timepoint_q':
            lo = torch.quantile(Xs, q[0], dim=0)
            hi = torch.quantile(Xs, q[1], dim=0)
            return cls(kind, lo, hi, skip_first, clip, q)
        raise ValueError(f'unknown normalizer kind {kind!r}')

    @classmethod
    def from_legacy(cls, norm_config: dict) -> 'Normalizer':
        """From an old checkpoint's norm_config (norm='per_timepoint')."""
        if norm_config.get('norm') != 'per_timepoint':
            raise ValueError(f"legacy norm {norm_config.get('norm')!r} not supported")
        p = norm_config['norm_params']
        return cls('per_timepoint', p['per_tp_min'], p['per_tp_max'], skip_first=True)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(X, dtype=DTYPE)
        out = X.clone()
        sl = slice(1, None) if self.skip_first else slice(None)
        z = 2 * math.pi * (X[:, sl] - self.lo) / (self.hi - self.lo) - math.pi
        if self.clip:
            z = z.clamp(-math.pi, math.pi)
        out[:, sl] = z
        return out

    def out_of_range(self, Xn: torch.Tensor, tol: float = 1e-12) -> float:
        """Fraction of normalized values with |x| > π (aliasing risk)."""
        return float((Xn.abs() > math.pi + tol).double().mean())

    def to_dict(self) -> dict:
        return {'kind': self.kind, 'lo': self.lo.tolist(), 'hi': self.hi.tolist(),
                'skip_first': self.skip_first, 'clip': self.clip, 'q': self.q}

    @classmethod
    def from_dict(cls, d: dict) -> 'Normalizer':
        return cls(d['kind'], d['lo'], d['hi'], d['skip_first'], d['clip'],
                   tuple(d['q']) if d.get('q') else None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def roc_auc(y, s) -> float:
    """AUC with average ranks for ties. y ∈ {0,1}; higher s = more anomalous."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(np.asarray(s, dtype=np.float64))
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    g = torch.Generator().manual_seed(0)
    N, T = 1000, 17
    X = torch.randn(N, T, generator=g, dtype=DTYPE) * 0.1
    X[:, 0] = 0.0
    y = np.r_[np.zeros(800), np.ones(200)].astype(int)
    unit = np.repeat(np.arange(50), 20)

    print('=' * 64)
    print('qvr.data self-test')
    print('=' * 64)

    print('SeriesSet')
    s = SeriesSet(X, y, unit, name='fake', extra={'grade': np.arange(N) % 4})
    print('  ' + s.summary())
    check('normals shape', s.normals().shape == (800, T), f'{tuple(s.normals().shape)}')
    sub = s.subset(s.y == 1)
    check('subset keeps extra aligned', len(sub.extra['grade']) == 200 and sub.y.all())
    try:
        SeriesSet(X, y[:10], unit)
        check('length mismatch raises', False)
    except ValueError:
        check('length mismatch raises', True)
    check('t_grid', torch.allclose(t_grid(17)[[0, -1]], torch.tensor([0, 2 * math.pi], dtype=DTYPE)))

    print('Normalizer (per_timepoint, legacy-exact)')
    nm = Normalizer.fit(s.normals(), 'per_timepoint')
    Xn = nm.transform(s.normals())
    check('frame 0 untouched', torch.equal(Xn[:, 0], s.normals()[:, 0]))
    check('train normals span exactly [-π, π] per frame',
          torch.allclose(Xn[:, 1:].min(0).values, torch.full((T - 1,), -math.pi, dtype=DTYPE))
          and torch.allclose(Xn[:, 1:].max(0).values, torch.full((T - 1,), math.pi, dtype=DTYPE)))
    lo, hi = s.normals()[:, 1:].min(0).values, s.normals()[:, 1:].max(0).values
    legacy = 2 * math.pi * (X[:, 1:] - lo) / (hi - lo) - math.pi
    check('matches legacy formula 2π(x−min)/(max−min)−π', torch.allclose(nm.transform(X)[:, 1:], legacy))
    leg = Normalizer.from_legacy({'norm': 'per_timepoint',
                                  'norm_params': {'per_tp_min': lo.tolist(), 'per_tp_max': hi.tolist()}})
    check('from_legacy == fit', torch.allclose(leg.transform(X), nm.transform(X)))
    rt = Normalizer.from_dict(nm.to_dict())
    check('to_dict/from_dict round trip', torch.equal(rt.transform(X), nm.transform(X)))
    X_big = X * 3
    frac = nm.out_of_range(nm.transform(X_big))
    check('out_of_range detects values past ±π', frac > 0, f'{frac:.3f} of values for 3× scaled data')
    nq = Normalizer.fit(s.normals(), 'per_timepoint_q', clip=True)
    check("'per_timepoint_q' + clip stays in [-π, π]",
          nq.out_of_range(nq.transform(X_big)) == 0)

    print('roc_auc')
    check('perfect separation = 1', roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0)
    check('reversed = 0', roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0)
    check('all ties = 0.5', roc_auc([0, 1, 0, 1], [1, 1, 1, 1]) == 0.5)
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 2, 5000)
    ss = rng.normal(size=5000) + yy
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in ss[yy == 1][:300] for b in ss[yy == 0][:300]])
    check('matches pairwise definition', abs(roc_auc(np.r_[np.ones(300), np.zeros(300)],
                                                      np.r_[ss[yy == 1][:300], ss[yy == 0][:300]]) - brute) < 1e-12,
          f'{brute:.4f}')

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()