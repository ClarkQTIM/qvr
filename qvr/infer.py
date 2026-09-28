"""
qvr/infer.py — scores and representations for a trained QVR model.

One pass returns, per series (N of them):
    score  (N,)          reconstruction cost  mean_t (eta_0 - mean_q <Z_q>)^2 / 4
    zrepr  (N, Q)        per-qubit <Z_q>, averaged over time
    pbar   (N, 2^Q)      eigenbasis fingerprint p̄_b (independent of D mode)
  optional (keep_time=True):
    z_t    (N, T, Q)     per-timepoint <Z_q>
    p_t    (N, T, 2^Q)   per-timepoint p_b

D mode (see circuit.py): 'mc' (default, matches training), 'averaged' (old
paper inference), 'mean', 'identity'.

Common random numbers: the eps draws for series i depend only on
(seed, i) — via fixed-size CRN blocks — never on batch size or on the input
values. So an original set and its time-permuted copy, scored with the same
seed, see identical draws: score differences reflect the inputs only.

Time permutation (`permute_time`) acts on already-NORMALIZED series, i.e. the
representation-level shuffle. Frame 0 stays in place by default.

Cache: `infer_cached` keys results on params + cfg + inference settings +
data hash + t + git commit; a stale cache from other weights cannot be reused.

Self-test:  python -m qvr.infer
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from qvr import circuit as C
from qvr.train import git_state, sha256_tensor

CRN_BLOCK = 1024   # eps are generated per block of this many series; never change casually


# ---------------------------------------------------------------------------
# Common random numbers
# ---------------------------------------------------------------------------

def _eps_block(block_id: int, seed: int, mode: str, cfg: C.QVRConfig,
               S: int, T: int) -> torch.Tensor:
    ss = np.random.SeedSequence([seed, block_id])
    g = torch.Generator().manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0] >> 1))
    shape = (CRN_BLOCK, S, T, cfg.n_terms) if mode == 'mc' else (CRN_BLOCK, S, cfg.n_terms)
    return C.sample_eps(shape, g)


def crn_eps(start: int, stop: int, seed: int, mode: str, cfg: C.QVRConfig,
            S: int, T: int) -> torch.Tensor | None:
    """eps for series indices [start, stop), identical however the range is batched."""
    if mode not in ('mc', 'shared', 'averaged'):
        return None
    b0, b1 = start // CRN_BLOCK, (stop - 1) // CRN_BLOCK
    blocks = torch.cat([_eps_block(b, seed, mode, cfg, S, T) for b in range(b0, b1 + 1)])
    off = start - b0 * CRN_BLOCK
    return blocks[off: off + (stop - start)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(params: dict, cfg: C.QVRConfig, X: torch.Tensor, t: torch.Tensor,
          mode: str = 'mc', n_draws: int | None = None, seed: int = 0,
          batch_size: int = 4096, keep_time: bool = False,
          verbose: bool = False) -> dict[str, np.ndarray]:
    X, t = X.to(C.DTYPE), t.to(C.DTYPE)
    N, T = X.shape[:2]
    S = n_draws if n_draws is not None else cfg.N_E
    params = {k: v.detach().to(C.DTYPE) for k, v in params.items()}
    out = {'score': np.empty(N), 'zrepr': np.empty((N, cfg.n_qubits)),
           'pbar': np.empty((N, cfg.dim))}
    if keep_time:
        out['z_t'] = np.empty((N, T, cfg.n_qubits))
        out['p_t'] = np.empty((N, T, cfg.dim))

    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        xb = X[start:stop]
        eps = crn_eps(start, stop, seed, mode, cfg, S, T)
        sc, z = C.series_score(params, xb, t, cfg, mode=mode, n_draws=S, eps=eps)  # (b,), (b, T, Q)
        p = C.probs_after_W(params, xb, cfg)                                        # (b, T, dim)
        out['score'][start:stop] = sc.numpy()
        out['zrepr'][start:stop] = z.mean(dim=1).numpy()
        out['pbar'][start:stop] = p.mean(dim=1).numpy()
        if keep_time:
            out['z_t'][start:stop] = z.numpy()
            out['p_t'][start:stop] = p.numpy()
        if verbose:
            print(f'  infer [{mode}] {stop:,}/{N:,}', flush=True)
    return out


def permute_time(X: torch.Tensor, seed: int, keep_first: bool = True) -> torch.Tensor:
    """Independent random time permutation per series (on normalized data).
    X (N, T) or (N, T, C); for multichannel series all channels share the permutation."""
    N, T = X.shape[:2]
    g = torch.Generator().manual_seed(seed)
    lo = 1 if keep_first else 0
    perm = torch.argsort(torch.rand(N, T - lo, generator=g), dim=1) + lo
    if keep_first:
        perm = torch.cat([torch.zeros(N, 1, dtype=perm.dtype), perm], dim=1)
    if X.dim() == 3:
        perm = perm.unsqueeze(-1).expand(-1, -1, X.shape[2])
    return torch.gather(X, 1, perm)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_key(params: dict, cfg: C.QVRConfig, X: torch.Tensor, t: torch.Tensor,
              settings: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(params[k].detach().cpu().to(C.DTYPE).numpy().tobytes())
    h.update(json.dumps({'cfg': cfg.to_dict(), 'settings': settings,
                         'data': sha256_tensor(X), 't': sha256_tensor(t),
                         'crn_block': CRN_BLOCK, 'git': git_state()['commit']},
                        sort_keys=True).encode())
    return h.hexdigest()[:24]


def infer_cached(params, cfg, X, t, cache_dir, **kw) -> dict[str, np.ndarray]:
    settings = {k: kw.get(k) for k in ('mode', 'n_draws', 'seed', 'keep_time')}
    path = Path(cache_dir) / f'infer_{cache_key(params, cfg, X, t, settings)}.npz'
    if path.exists():
        return dict(np.load(path))
    out = infer(params, cfg, X, t, **kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    (path.with_suffix('.json')).write_text(json.dumps(
        {'settings': settings, 'cfg': cfg.to_dict(), 'data_hash': sha256_tensor(X),
         'git': git_state(), 'n': int(X.shape[0])}, indent=1))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile
    import time

    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    cfg = C.QVRConfig(n_qubits=3, n_layers=3, N_E=10)
    params = C.init_params(cfg, np.random.default_rng(0))
    N, T = 2500, 17
    g = torch.Generator().manual_seed(0)
    X = (torch.rand(N, T, generator=g, dtype=C.DTYPE) * 2 - 1) * math.pi
    X[:, 0] = 0.0
    t = torch.linspace(0, 2 * math.pi, T, dtype=C.DTYPE)

    print('=' * 64)
    print(f'qvr.infer self-test   N={N} T={T}  cfg={cfg.to_dict()}')
    print('=' * 64)

    print('Shapes')
    out = infer(params, cfg, X, t, mode='mc', seed=0, keep_time=True)
    for k, shp in {'score': (N,), 'zrepr': (N, 3), 'pbar': (N, 8),
                   'z_t': (N, T, 3), 'p_t': (N, T, 8)}.items():
        check(f'{k} shape', out[k].shape == shp, f'{out[k].shape}')
    check('scores finite and >= 0', np.isfinite(out['score']).all() and (out['score'] >= 0).all(),
          f"range [{out['score'].min():.4f}, {out['score'].max():.4f}]")
    check('pbar rows sum to 1', np.allclose(out['pbar'].sum(1), 1))
    check('pbar = time-average of p_t', np.allclose(out['pbar'], out['p_t'].mean(1)))

    print('Common random numbers / reproducibility')
    small = infer(params, cfg, X, t, mode='mc', seed=0, batch_size=97)
    check('batch size does not change results', np.array_equal(small['score'], out['score']))
    again = infer(params, cfg, X, t, mode='mc', seed=0)
    check('same seed → identical', np.array_equal(again['score'], out['score']))
    other = infer(params, cfg, X, t, mode='mc', seed=1)
    check("different seed → different 'mc' scores", not np.array_equal(other['score'], out['score']))
    sub = infer(params, cfg, X[1500:1600], t, mode='mc', seed=0)
    check('draws depend on index, not data (subset ≠ slice)',
          not np.array_equal(sub['score'], out['score'][1500:1600]),
          'expected: subset re-indexes from 0')

    print('D modes')
    m1 = infer(params, cfg, X, t, mode='mean', seed=0)
    m2 = infer(params, cfg, X, t, mode='mean', seed=5)
    check("'mean' is deterministic", np.array_equal(m1['score'], m2['score']))
    p0 = {k: v.detach().clone() for k, v in params.items()}
    p0['sigma'] = torch.zeros_like(p0['sigma'])
    a0 = infer(p0, cfg, X, t, mode='averaged', seed=3)
    check("'averaged' with sigma=0 equals 'mean'", np.allclose(a0['score'], m1['score'], atol=1e-14))
    idm = infer(params, cfg, X, t, mode='identity')
    analytic = ((params['eta_0'].item() - torch.cos(X)) ** 2 / 4).mean(1).numpy()
    check("'identity' score = mean_t (eta0 - cos x)^2/4", np.allclose(idm['score'], analytic, atol=1e-12))
    check('pbar identical across D modes', np.allclose(m1['pbar'], out['pbar'])
          and np.allclose(idm['pbar'], out['pbar']))

    print('Time permutation (representation level)')
    Xp = permute_time(X, seed=11)
    check('frame 0 unchanged', torch.equal(Xp[:, 0], X[:, 0]))
    check('same values per series', torch.equal(Xp.sort(1).values, X.sort(1).values))
    check('order actually changed', not torch.equal(Xp, X))
    sh_mc = infer(params, cfg, Xp, t, mode='mc', seed=0)
    check('pbar invariant under permutation', np.allclose(sh_mc['pbar'], out['pbar'], atol=1e-12))
    sh_id = infer(params, cfg, Xp, t, mode='identity')
    check("'identity' score invariant under permutation", np.allclose(sh_id['score'], idm['score'], atol=1e-12))
    r = np.corrcoef(out['score'], sh_mc['score'])[0, 1]
    check("'mc' score changes under permutation", r < 1 - 1e-6, f'shuf_r={r:.4f} (random weights)')

    print('Cache')
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        c1 = infer_cached(params, cfg, X, t, tmp, mode='mc', seed=0)
        n_files = len(list(Path(tmp).glob('*.npz')))
        c2 = infer_cached(params, cfg, X, t, tmp, mode='mc', seed=0)
        check('second call hits cache', n_files == 1 and len(list(Path(tmp).glob('*.npz'))) == 1)
        check('cached result identical', np.array_equal(c1['score'], c2['score']))
        infer_cached(p0, cfg, X, t, tmp, mode='mc', seed=0)
        check('different weights → new cache entry', len(list(Path(tmp).glob('*.npz'))) == 2)

    print('Timing')
    Xbig = X.repeat(40, 1)
    t0 = time.perf_counter()
    infer(params, cfg, Xbig, t, mode='mc', seed=0)
    dt = time.perf_counter() - t0
    print(f"  'mc' N_E=10: {len(Xbig):,} series in {dt:.1f}s ({len(Xbig) / dt:,.0f} series/s)")

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()