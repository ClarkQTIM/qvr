"""
qvr/sequential.py — sQVR: a sequential Hamiltonian-memory model (ANALYSIS_PLAN §12.2).

In QVR every timepoint is a fresh circuit, so nothing propagates across time. sQVR keeps
ONE state and evolves it through the series, so the Hamiltonian carries the history:

    ψ_0 = |0…0⟩
    for t = 1..T:
        p_t(b) = |⟨b| V ψ_{t−1}⟩|²                 predictive distribution over 2^Q bins
        ψ_t    = D(Δt) · W · U_enc(x_t) · ψ_{t−1}  write x_t, mix, evolve

  U_enc(x) = RY(x)^{⊗Q} applied to the running state; W, V: CNOT-chain ansätze
  (circuit.transform_ops); D(Δt) = exp(−i Δt H), H = ½ Σ_P μ_P P (Pauli-Z strings up to
  order k), deterministic; Δt = 2π / T.

  Bins: per-timepoint quantile edges of the TRAINING NORMALS at (1, 5, 25, 50, 75, 95, 99)%
  → 8 bins with tail masses 1%, 4%, 20%, 25%, 25%, 20%, 4%, 1%. A memoryless per-timepoint
  histogram therefore scores rarity (NLL = −log mass) and is the no-memory baseline.

  Loss / anomaly score: mean over t of −log p_t(bin(x_t)).
  Generation (follow-on paper): sample a bin from p_t, then x uniformly inside the bin.

Matrix backend as in circuit.py: W, V from qml.matrix (single source of the ansatz);
everything else batched torch. `reference_probs` rebuilds the whole sequence gate by gate
in PennyLane for the equivalence test.

Self-test:  python -m qvr.sequential
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

from qvr import circuit as C

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


@dataclass(frozen=True)
class SeqConfig:
    n_qubits: int = 3
    n_layers: int = 3
    k: int | None = None
    quantiles: tuple = QUANTILES

    @property
    def circ(self) -> C.QVRConfig:
        return C.QVRConfig(n_qubits=self.n_qubits, n_layers=self.n_layers, k=self.k, ansatz='chain')

    @property
    def n_bins(self) -> int:
        return len(self.quantiles) + 1

    def __post_init__(self):
        if self.n_bins != 2 ** self.n_qubits:
            raise ValueError(f'{len(self.quantiles) + 1} bins need {math.log2(self.n_bins):.0f} qubits')

    def to_dict(self):
        d = asdict(self)
        d['quantiles'] = list(self.quantiles)
        return d


def init_params(cfg: SeqConfig, rng: np.random.Generator) -> dict:
    n = cfg.n_layers * cfg.n_qubits * 3
    t = lambda a: torch.tensor(a, dtype=C.DTYPE, requires_grad=True)  # noqa: E731
    return {'alpha_W': t(rng.uniform(0, 2 * np.pi, n)), 'alpha_V': t(rng.uniform(0, 2 * np.pi, n)),
            'mu': t(rng.uniform(0, 2 * np.pi, cfg.circ.n_terms))}


# ---------------------------------------------------------------------------
# Bins
# ---------------------------------------------------------------------------

def fit_edges(X_train: torch.Tensor, cfg: SeqConfig) -> torch.Tensor:
    """(T, n_bins − 1) per-timepoint quantile edges of the training normals."""
    q = torch.tensor(cfg.quantiles, dtype=C.DTYPE)
    return torch.quantile(X_train.to(C.DTYPE), q, dim=0).T.contiguous()


def bin_index(X: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """(N, T) bin of each value: number of edges strictly below it."""
    return (X.unsqueeze(-1) > edges.unsqueeze(0)).sum(-1)


def bin_masses(cfg: SeqConfig) -> torch.Tensor:
    q = torch.tensor((0.0,) + tuple(cfg.quantiles) + (1.0,), dtype=C.DTYPE)
    return q[1:] - q[:-1]


def histogram_nll(X: torch.Tensor, edges: torch.Tensor, cfg: SeqConfig) -> torch.Tensor:
    """Memoryless baseline: (N,) mean over t of −log mass(bin(x_t))."""
    return -torch.log(bin_masses(cfg))[bin_index(X, edges)].mean(1)


# ---------------------------------------------------------------------------
# Matrix backend
# ---------------------------------------------------------------------------

def enc_matrix(x: torch.Tensor, Q: int) -> torch.Tensor:
    """(B, 2^Q, 2^Q) RY(x)^{⊗Q}, qubit 0 = most significant (PennyLane order)."""
    c, s = torch.cos(x / 2), torch.sin(x / 2)
    r = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)     # (B, 2, 2)
    m = r
    for _ in range(Q - 1):
        m = torch.einsum('bij,bkl->bikjl', m, r).reshape(len(x), m.shape[1] * 2, m.shape[2] * 2)
    return m


def step_operators(params, cfg: SeqConfig, T: int):
    cc = cfg.circ
    W = C.W_matrix(params['alpha_W'], cc)
    V = C.W_matrix(params['alpha_V'], cc)
    lam = C.eigenvalues(params['mu'], cc)
    D = torch.exp(-1j * lam * (2 * math.pi / T)).to(C.CDTYPE)
    return W, V, D


def predictive(params, X: torch.Tensor, cfg: SeqConfig) -> torch.Tensor:
    """(B, T, 2^Q) predictive distributions p_t(b | x_{<t})."""
    B, T = X.shape
    W, V, D = step_operators(params, cfg, T)
    psi = torch.zeros(B, cfg.circ.dim, dtype=C.CDTYPE)
    psi[:, 0] = 1.0
    out = []
    for t in range(T):
        out.append((psi @ V.T).abs() ** 2)
        U = enc_matrix(X[:, t], cfg.n_qubits).to(C.CDTYPE)
        psi = torch.einsum('bij,bj->bi', U, psi) @ W.T * D
    return torch.stack(out, 1)


def nll(params, X: torch.Tensor, edges: torch.Tensor, cfg: SeqConfig) -> torch.Tensor:
    """(B, T) −log p_t(bin(x_t))."""
    P = predictive(params, X, cfg)
    b = bin_index(X, edges)
    return -torch.log(P.gather(-1, b.unsqueeze(-1)).squeeze(-1) + 1e-12)


@torch.no_grad()
def score(params, X: torch.Tensor, edges: torch.Tensor, cfg: SeqConfig, batch: int = 4096) -> np.ndarray:
    params = {k: v.detach() for k, v in params.items()}
    return np.concatenate([nll(params, X[i:i + batch], edges, cfg).mean(1).numpy() for i in range(0, len(X), batch)])


@torch.no_grad()
def sample(params, n: int, T: int, edges: torch.Tensor, cfg: SeqConfig, generator=None) -> torch.Tensor:
    """(n, T) generated series: bin ~ p_t, x ~ Uniform(bin), fed back. (Follow-on paper.)"""
    params = {k: v.detach() for k, v in params.items()}
    W, V, D = step_operators(params, cfg, T)
    psi = torch.zeros(n, cfg.circ.dim, dtype=C.CDTYPE)
    psi[:, 0] = 1.0
    lo = torch.cat([torch.full((T, 1), -math.pi, dtype=C.DTYPE), edges], 1)
    hi = torch.cat([edges, torch.full((T, 1), math.pi, dtype=C.DTYPE)], 1)
    xs = []
    for t in range(T):
        p = ((psi @ V.T).abs() ** 2).real
        b = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=generator).squeeze(1)
        u = torch.rand(n, dtype=C.DTYPE, generator=generator)
        x = lo[t, b] + u * (hi[t, b] - lo[t, b])
        xs.append(x)
        psi = torch.einsum('bij,bj->bi', enc_matrix(x, cfg.n_qubits).to(C.CDTYPE), psi) @ W.T * D
    return torch.stack(xs, 1)


# ---------------------------------------------------------------------------
# PennyLane reference (tests only)
# ---------------------------------------------------------------------------

def reference_probs(params, x_prefix, cfg: SeqConfig, T: int):
    """Predictive distribution after the prefix, built gate by gate."""
    cc = cfg.circ
    Q, L = cc.n_qubits, cc.n_layers
    dev = qml.device('default.qubit', wires=Q)
    dt = 2 * math.pi / T

    def _diag(angles):
        for j, comb in enumerate(cc.terms):
            pairs = [comb[i:i + 2] for i in range(len(comb) - 1)]
            for pr in pairs:
                qml.CNOT(wires=list(pr))
            qml.RZ(angles[j], wires=comb[-1])
            for pr in reversed(pairs):
                qml.CNOT(wires=list(pr))

    @qml.qnode(dev, interface='torch')
    def circuit():
        for x in x_prefix:
            for q in range(Q):
                qml.RY(x, wires=q)
            C.transform_ops(params['alpha_W'], Q, L, 'chain')
            _diag(params['mu'] * dt)
        C.transform_ops(params['alpha_V'], Q, L, 'chain')
        return qml.probs(wires=range(Q))

    return circuit()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_seq(X: torch.Tensor, cfg: SeqConfig, run_seed: int, out_dir, max_samples=50_000, batch_size=8,
              lr=0.01, eval_every=50, patience_evals=10, min_delta=1e-6, monitor_size=256, meta=None,
              strict_git=False, verbose=False) -> dict:
    from qvr.train import git_state, seed_streams, sha256_json, sha256_tensor
    git = git_state()
    if strict_git and (git['commit'] is None or git['dirty']):
        raise RuntimeError(f'refusing to train: git state {git} (commit your changes)')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = X.to(C.DTYPE)
    N, T = X.shape
    rs = seed_streams(run_seed)
    edges = fit_edges(X, cfg)
    params = init_params(cfg, rs['init'])
    init = {k: v.detach().clone() for k, v in params.items()}
    opt = torch.optim.Adam(list(params.values()), lr=lr)
    M = min(monitor_size, max(1, N // 5))
    perm = rs['batches'].permutation(N)
    mon, pool = X[perm[:M]], perm[M:]

    def mon_loss():
        with torch.no_grad():
            return float(nll(params, mon, edges, cfg).mean())

    hist = {'batch_loss': [], 'monitor': [{'batch': 0, 'loss': mon_loss()}], 'stopped': 'max_samples'}
    best_m, best_b, bad, nb, ns = hist['monitor'][0]['loss'], 0, 0, 0, 0
    best = {k: v.detach().clone() for k, v in params.items()}
    t0 = time.time()
    while ns + batch_size <= max_samples:
        idx = rs['batches'].choice(pool, size=batch_size, replace=False)
        opt.zero_grad()
        loss = nll(params, X[idx], edges, cfg).mean()
        loss.backward()
        opt.step()
        nb, ns = nb + 1, ns + batch_size
        hist['batch_loss'].append(float(loss.detach()))
        if nb % eval_every == 0:
            m = mon_loss()
            hist['monitor'].append({'batch': nb, 'loss': m})
            if m < best_m - min_delta:
                best_m, best_b, bad = m, nb, 0
                best = {k: v.detach().clone() for k, v in params.items()}
            else:
                bad += 1
            if verbose:
                print(f'  batch {nb:6d} train {float(loss):.4f} monitor {m:.4f} best {best_m:.4f}@{best_b}', flush=True)
            if bad >= patience_evals:
                hist['stopped'] = 'early_stop'
                break
    for k in params:
        params[k].data.copy_(best[k])
    hyper = {'max_samples': max_samples, 'batch_size': batch_size, 'lr': lr, 'eval_every': eval_every,
             'patience_evals': patience_evals, 'monitor_size': M}
    torch.save({'params': {k: v.detach().clone() for k, v in params.items()}, 'init_params': init,
                'cfg': cfg.to_dict(), 'edges': edges, 'hyper': hyper, 'run_seed': run_seed, 'meta': meta or {}},
               out_dir / 'params.pt')
    hist.update({'n_batches': nb, 'best_batch': best_b, 'best_monitor': best_m, 'elapsed_s': time.time() - t0,
                 'memoryless_nll_monitor': float(histogram_nll(mon, edges, cfg).mean())})
    (out_dir / 'history.json').write_text(json.dumps(hist, indent=1))
    (out_dir / 'manifest.json').write_text(json.dumps({
        'model': 'sQVR', 'run_seed': run_seed, 'git': git, 'config_hash': sha256_json({'cfg': cfg.to_dict(), 'hyper': hyper}),
        'data_hash': sha256_tensor(X), 'elapsed_s': hist['elapsed_s'], 'stopped': hist['stopped'],
        'best_batch': best_b, 'best_monitor': best_m}, indent=1))
    return {'params': params, 'edges': edges, 'history': hist}


def load_seq(run_dir):
    ck = torch.load(Path(run_dir) / 'params.pt', map_location='cpu', weights_only=False)
    d = dict(ck['cfg'])
    d['quantiles'] = tuple(d['quantiles'])
    return ck['params'], SeqConfig(**d), ck['edges'], ck


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    cfg = SeqConfig()
    g = torch.Generator().manual_seed(0)
    p = init_params(cfg, np.random.default_rng(0))
    B, T = 6, 10
    X = (torch.rand(B, T, generator=g, dtype=C.DTYPE) * 2 - 1) * math.pi
    print('=' * 64)
    print(f'qvr.sequential self-test   cfg={cfg.to_dict()}')
    print('=' * 64)

    print('Shapes / invariants')
    P = predictive(p, X, cfg)
    check('predictive shape', P.shape == (B, T, 8), f'{tuple(P.shape)}')
    check('distributions sum to 1', torch.allclose(P.sum(-1), torch.ones(B, T, dtype=C.DTYPE)))
    check('t = 1 prediction ignores the data (same for every series)', torch.allclose(P[:, 0], P[:1, 0].expand(B, -1)))
    E = enc_matrix(X[:, 0], 3)
    check('encoder unitary (real orthogonal)', torch.allclose(E @ E.transpose(1, 2), torch.eye(8, dtype=C.DTYPE).expand(B, 8, 8)))

    print('Equivalence with gate-by-gate PennyLane (whole sequence)')
    err = 0.0
    with torch.no_grad():
        pd = {k: v.detach() for k, v in p.items()}
        for t in range(5):
            ref = reference_probs(pd, X[0, :t], cfg, T)
            err = max(err, float((ref - P[0, t].detach()).abs().max()))
    check('p_t matches reference for t = 1..5', err < 1e-10, f'max |Δ| {err:.1e}')

    print('Bins / baseline')
    Xtr = torch.randn(4000, T, generator=g, dtype=C.DTYPE)
    edges = fit_edges(Xtr, cfg)
    frac = torch.bincount(bin_index(Xtr, edges).flatten(), minlength=8).double() / Xtr.numel()
    check('training normals fill bins at the nominal masses', torch.allclose(frac, bin_masses(cfg), atol=0.01),
          ' '.join(f'{v:.3f}' for v in frac))
    h_norm = histogram_nll(Xtr[:500], edges, cfg).mean()
    h_far = histogram_nll(Xtr[:500] + 4.0, edges, cfg).mean()
    check('memoryless histogram scores extreme values as rare', h_far > h_norm + 2, f'{h_norm:.2f} → {h_far:.2f}')

    print('Gradients')
    loss = nll(p, X, edges, cfg).mean()
    loss.backward()
    for k, v in p.items():
        check(f'grad {k}', v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0,
              f'|grad| {v.grad.abs().sum():.2e}')

    print('Order sensitivity (memory)')
    s0 = score(p, X, edges, cfg)
    s1 = score(p, X.flip(1), edges, cfg)
    check('score changes when the series is reversed', np.abs(s0 - s1).max() > 1e-6, f'max |Δ| {np.abs(s0 - s1).max():.3f}')

    print('Learning: train on growing ramps, test growing vs falling (time-reversed twins)')
    Tg = 12
    ramp = torch.linspace(-2.0, 2.0, Tg, dtype=C.DTYPE)
    Xg = ramp + 0.15 * torch.randn(600, Tg, generator=g, dtype=C.DTYPE)
    Xf = ramp.flip(0) + 0.15 * torch.randn(200, Tg, generator=g, dtype=C.DTYPE)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r = train_seq(Xg[:400], cfg, run_seed=0, out_dir=tmp, max_samples=4000, eval_every=25, patience_evals=8)
        h = r['history']
        check('monitor NLL decreases during training', h['best_monitor'] < h['monitor'][0]['loss'] - 0.05,
              f"{h['monitor'][0]['loss']:.3f} → {h['best_monitor']:.3f} (memoryless {h['memoryless_nll_monitor']:.3f})")
        pr, cfg2, ed, _ = load_seq(tmp)
        sg, sf = score(pr, Xg[400:], ed, cfg2), score(pr, Xf, ed, cfg2)
        auc = float(np.mean(sf[:, None] > sg[None, :]))
        check('score separates falling from growing (AUC > 0.9)', auc > 0.9, f'AUC {auc:.3f}')
        x_gen = sample(pr, 50, Tg, ed, cfg2, generator=torch.Generator().manual_seed(1))
        check('sampling: shape and range', x_gen.shape == (50, Tg) and float(x_gen.abs().max()) <= math.pi + 1e-9)
        slope = float((x_gen[:, -3:].mean() - x_gen[:, :3].mean()))
        check('generated series rise on average (learned direction)', slope > 0, f'end − start {slope:+.2f}')
    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()