"""
qvr/train.py — train QVR on normal-only series. Dataset-agnostic.

Input is an already-normalized (N, T) float64 tensor of NORMAL training series
and a (T,) time grid. Normalization, splits and labels belong to the dataset
adapter; their parameters are passed through `meta` and saved with the run.

One integer `run_seed` fully determines a run. It is split with
np.random.SeedSequence into independent streams:
    init    → initial alpha, mu, sigma, eta_0
    batches → which series form each minibatch (+ the held-out monitor set)
    draws   → training-time D samples
    monitor → fixed D samples for the monitor set (common random numbers)

Objective (unchanged from qvr_cervix): 0.5 * mean_batch(cost, 'mc', N_E draws)
+ sigma_lambda/pi * mean(arctan(2π τ |sigma|)).  Adam, batch 8, lr 0.01.

Early stopping: every `eval_every` batches the loss is evaluated on a fixed
monitor set drawn from the TRAINING split (never val) with fixed draws; stop
after `patience_evals` evaluations without improvement; restore the best.
Defaults (eval_every=50, patience_evals=10) ≈ the old 500-batch patience, but
on a stable signal instead of single noisy minibatches.

Outputs in out_dir:
    params.pt      params + cfg + run_seed + meta   (load with load_run)
    history.json   batch losses, monitor losses, stopping info
    manifest.json  git commit/dirty, versions, config hash, data hash, timing

Self-test:  python -m qvr.train
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from qvr import circuit as C


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def git_state(repo_dir: Path | None = None) -> dict:
    cwd = str(repo_dir or Path(__file__).resolve().parents[1])
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=cwd,
                                             stderr=subprocess.DEVNULL).decode().strip())
        return {'commit': commit, 'dirty': dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {'commit': None, 'dirty': None}


def sha256_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def seed_streams(run_seed: int) -> dict:
    init_ss, batch_ss, draw_ss, mon_ss = np.random.SeedSequence(run_seed).spawn(4)
    def _torch_gen(ss):
        return torch.Generator().manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0] >> 1))
    return {
        'init':    np.random.default_rng(init_ss),
        'batches': np.random.default_rng(batch_ss),
        'draws':   _torch_gen(draw_ss),
        'monitor': _torch_gen(mon_ss),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: torch.Tensor,
    t: torch.Tensor,
    cfg: C.QVRConfig,
    run_seed: int,
    out_dir,
    max_samples: int = 50_000,
    batch_size: int = 8,
    lr: float = 0.01,
    eval_every: int = 50,
    patience_evals: int = 10,
    min_delta: float = 1e-6,
    monitor_size: int = 256,
    meta: dict | None = None,
    strict_git: bool = False,
    verbose: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = X.to(C.DTYPE)
    t = t.to(C.DTYPE)
    N, T = X.shape[:2]
    if t.shape != (T,):
        raise ValueError(f't has shape {tuple(t.shape)}, expected ({T},)')

    git = git_state()
    if strict_git and (git['commit'] is None or git['dirty']):
        raise RuntimeError(f'refusing to train: git state {git} (commit your changes)')

    rs = seed_streams(run_seed)
    params = C.init_params(cfg, rs['init'])
    init_params = {k: v.detach().clone() for k, v in params.items()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)

    # held-out monitor set from the training split, with fixed draws
    M = min(monitor_size, max(1, N // 5))
    perm = rs['batches'].permutation(N)
    mon_idx, pool = perm[:M], perm[M:]
    X_mon = X[mon_idx]
    shape = ((M, cfg.N_E, T, cfg.n_terms) if cfg.draw_mode == 'mc' else (M, cfg.N_E, cfg.n_terms))
    eps_mon = C.sample_eps(shape, rs['monitor'])

    def monitor_loss() -> float:
        with torch.no_grad():
            sc, _ = C.series_score(params, X_mon, t, cfg, mode=cfg.draw_mode, eps=eps_mon)
            return float(0.5 * sc.mean() + C.arctan_penalty(params, cfg))

    hist = {'batch_loss': [], 'monitor': [], 'best_monitor': math.inf,
            'best_batch': 0, 'stopped': 'max_samples'}
    best = {k: v.detach().clone() for k, v in params.items()}
    hist['monitor'].append({'batch': 0, 'loss': monitor_loss()})
    hist['best_monitor'] = hist['monitor'][0]['loss']
    bad_evals, n_batches, n_samples = 0, 0, 0
    t_start = time.time()

    while n_samples + batch_size <= max_samples:
        idx = rs['batches'].choice(pool, size=batch_size, replace=False)
        opt.zero_grad()
        loss = C.training_loss(params, X[idx], t, cfg, generator=rs['draws'])
        loss.backward()
        opt.step()
        n_batches += 1
        n_samples += batch_size
        hist['batch_loss'].append(float(loss.detach()))

        if n_batches % eval_every == 0:
            m = monitor_loss()
            hist['monitor'].append({'batch': n_batches, 'loss': m})
            if m < hist['best_monitor'] - min_delta:
                hist['best_monitor'], hist['best_batch'], bad_evals = m, n_batches, 0
                best = {k: v.detach().clone() for k, v in params.items()}
            else:
                bad_evals += 1
            if verbose:
                print(f'  batch {n_batches:6d}  samples {n_samples:7d}  '
                      f'train {float(loss):.6f}  monitor {m:.6f}  '
                      f'best {hist["best_monitor"]:.6f}@{hist["best_batch"]}', flush=True)
            if bad_evals >= patience_evals:
                hist['stopped'] = 'early_stop'
                break

    elapsed = time.time() - t_start
    for k in params:
        params[k].data.copy_(best[k])
    hist.update({'n_batches': n_batches, 'n_samples': n_samples, 'elapsed_s': elapsed})

    hyper = {'max_samples': max_samples, 'batch_size': batch_size, 'lr': lr,
             'eval_every': eval_every, 'patience_evals': patience_evals,
             'min_delta': min_delta, 'monitor_size': M}
    torch.save({'params': {k: v.detach().clone() for k, v in params.items()},
                'init_params': init_params, 'cfg': cfg.to_dict(), 'hyper': hyper,
                'run_seed': run_seed, 't': t, 'meta': meta or {}},
               out_dir / 'params.pt')
    (out_dir / 'history.json').write_text(json.dumps(hist, indent=1))
    manifest = {
        'run_seed': run_seed, 'git': git,
        'config_hash': sha256_json({'cfg': cfg.to_dict(), 'hyper': hyper}),
        'data_hash': sha256_tensor(X), 'data_shape': [N, T],
        't_hash': sha256_tensor(t),
        'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                     'pennylane': qml.__version__, 'numpy': np.__version__},
        'finished': time.strftime('%Y-%m-%d %H:%M:%S'), 'elapsed_s': elapsed,
        'stopped': hist['stopped'], 'best_batch': hist['best_batch'],
        'best_monitor': hist['best_monitor'],
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    if verbose:
        print(f'  done: {hist["stopped"]} after {n_batches} batches ({elapsed:.1f}s), '
              f'best monitor {hist["best_monitor"]:.6f} at batch {hist["best_batch"]}')
    return {'params': params, 'history': hist, 'manifest': manifest}


def load_run(out_dir) -> tuple[dict, C.QVRConfig, dict]:
    """Load a run saved by train(): (params, cfg, full checkpoint dict)."""
    ck = torch.load(Path(out_dir) / 'params.pt', map_location='cpu', weights_only=False)
    return ck['params'], C.QVRConfig(**ck['cfg']), ck


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile

    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    cfg = C.QVRConfig(n_qubits=3, n_layers=3, N_E=2)
    N, T = 300, 17
    g = torch.Generator().manual_seed(0)
    X = (torch.rand(N, T, generator=g, dtype=C.DTYPE) * 2 - 1) * math.pi
    X[:, 0] = 0.0
    t = torch.linspace(0, 2 * math.pi, T, dtype=C.DTYPE)
    kw = dict(max_samples=160, batch_size=8, eval_every=5, patience_evals=100,
              meta={'note': 'self-test'}, verbose=False)

    print('=' * 64)
    print(f'qvr.train self-test   N={N} T={T}  cfg={cfg.to_dict()}')
    print(f'  training: {kw["max_samples"]} samples, batch 8, eval every 5 batches')
    print('=' * 64)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r1 = train(X, t, cfg, run_seed=0, out_dir=Path(tmp) / 'a', **kw)
        h = r1['history']
        print('Loop')
        check('20 batches run', h['n_batches'] == 20, f"{h['n_batches']} batches")
        check('batch losses finite', all(math.isfinite(v) for v in h['batch_loss']),
              f"first {h['batch_loss'][0]:.5f}, last {h['batch_loss'][-1]:.5f}")
        check('monitor evaluated 5× (incl. batch 0)', len(h['monitor']) == 5,
              ', '.join(f"{m['loss']:.5f}" for m in h['monitor']))
        ck = torch.load(Path(tmp) / 'a' / 'params.pt', weights_only=False)
        moved = max((ck['params'][k] - ck['init_params'][k]).abs().max().item()
                    for k in ck['params'])
        check('parameters changed from init', moved > 0 or h['best_batch'] == 0,
              f'max |Δ| {moved:.2e}, best batch {h["best_batch"]}')
        for k, v in ck['params'].items():
            check(f'saved {k} shape', v.shape == C.init_params(cfg, np.random.default_rng(0))[k].shape,
                  f'{tuple(v.shape)}')

        print('Files / provenance')
        for f in ('params.pt', 'history.json', 'manifest.json'):
            check(f'{f} written', (Path(tmp) / 'a' / f).exists())
        man = json.loads((Path(tmp) / 'a' / 'manifest.json').read_text())
        print(f"  manifest: git={man['git']}  data_hash={man['data_hash'][:12]}…  "
              f"config_hash={man['config_hash'][:12]}…")

        print('Reload / determinism')
        p_load, cfg_load, _ = load_run(Path(tmp) / 'a')
        check('reload cfg matches', cfg_load == cfg)
        z1 = C.expvals(r1['params'], X[:5], t, cfg, mode='mean')
        z2 = C.expvals(p_load, X[:5], t, cfg_load, mode='mean')
        check('reloaded params give identical outputs', torch.equal(z1, z2))
        r2 = train(X, t, cfg, run_seed=0, out_dir=Path(tmp) / 'b', **kw)
        same = all(torch.equal(r1['params'][k], r2['params'][k]) for k in r1['params'])
        check('same run_seed → bit-identical run', same)
        r3 = train(X, t, cfg, run_seed=1, out_dir=Path(tmp) / 'c', **kw)
        ck3 = torch.load(Path(tmp) / 'c' / 'params.pt', weights_only=False)
        diff_init = not torch.equal(ck['init_params']['alpha'], ck3['init_params']['alpha'])
        check('different run_seed → different init', diff_init)
        print(f"  timing: {h['elapsed_s'] / h['n_batches'] * 1e3:.1f} ms/batch "
              f"(N_E={cfg.N_E}, batch 8)")

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()