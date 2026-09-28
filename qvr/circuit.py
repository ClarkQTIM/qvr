"""
qvr/circuit.py — the ONE definition of the QVR circuit.

Circuit (fixed embedding only):
    |0>^Q --RY(x)^{⊗Q}--> W(alpha) --> D(eps, t) --> W(alpha)^† --> measure Z_q

  W(alpha)  : n_layers × [RX, RY, RZ per qubit] + CNOT chain between layers
  D(eps, t) : diagonal, one RZ-type term per Pauli-Z string of order 1..k,
              angle_j = D_j * t,  D_j = mu_j + |sigma_j| * eps_j
              PennyLane RZ(θ) = exp(-iθZ/2)  →  mean Hamiltonian
              H = ½ Σ_j mu_j P_j,  eigenvalues λ_b = ½ Σ_j mu_j ∏_{q∈j} z_q(b)

Two backends, one definition:
  * `transform_ops` is the single source of W. The PennyLane reference circuit
    and the matrix backend both use it (the matrix backend via qml.matrix).
  * The matrix backend does everything else as batched torch linear algebra
    (embedding in closed form, D as diagonal phases). It is what train/infer use.
  * `reference_*` qnodes build the circuit gate-by-gate, like the old model.py,
    and exist only to test the matrix backend.

D modes (how eps is handled):
  'mc'        : independent draws per (series, timepoint, draw); circuit outputs
                averaged over draws. Matches the training objective.
  'averaged'  : N_E draws averaged BEFORE the circuit, one D̄ per series shared
                across timepoints. Reproduces old predict_with_representations.
  'mean'      : D = mu  (deterministic mean dynamics)
  'identity'  : D = 0   (no evolution; W†W = I)

Parameter names and alpha layout match the old checkpoints
(alpha reshaped to (n_layers, n_qubits, 3)), so old weights load unchanged.

Self-test:  python -m qvr.circuit
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
import pennylane as qml
import torch

DTYPE = torch.float64
CDTYPE = torch.complex128
D_MODES = ('mc', 'shared', 'averaged', 'mean', 'identity')
EMBEDDINGS = ('ry_all', 'ry_first', 'ry_per_qubit', 'rx_per_qubit')
ANSATZE = ('chain', 'sel')
COSTS = ('outside', 'inside')


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QVRConfig:
    n_qubits: int = 3
    n_layers: int = 3
    k: int | None = None          # max Pauli-string order in D; None → n_qubits
    N_E: int = 10                 # draws per evaluation (training + 'mc'/'averaged')
    tau: float = 1.0              # arctan penalty scale
    sigma_lambda: float = 1e-2    # arctan penalty weight
    embedding: str = 'ry_all'     # 'ry_all' (same x on every qubit, qvr_cervix), 'ry_first'
                                  # (x on qubit 0 only), 'ry_per_qubit' / 'rx_per_qubit'
                                  # (channel c on qubit c; rx = PennyLane AngleEmbedding default)
    ansatz: str = 'chain'         # 'chain' (RX·RY·RZ + CNOT chain, qvr_cervix) or
                                  # 'sel' (StronglyEntanglingLayers, Baker notebook)
    cost: str = 'outside'         # 'outside': (η₀ − E_ε⟨O⟩)²  (qvr_cervix, Baker notebook)
                                  # 'inside' : E_ε[(η₀ − ⟨O⟩)²] (Baker paper Eq. 4)
    draw_mode: str = 'mc'         # training draws: 'mc' (fresh per timepoint) or
                                  # 'shared' (one draw per series, Baker paper pseudocode)

    def __post_init__(self):
        for name, val, ok in (('embedding', self.embedding, EMBEDDINGS),
                              ('ansatz', self.ansatz, ANSATZE),
                              ('cost', self.cost, COSTS),
                              ('draw_mode', self.draw_mode, ('mc', 'shared'))):
            if val not in ok:
                raise ValueError(f'{name}={val!r}; choose from {ok}')

    @property
    def per_qubit_input(self) -> bool:
        return self.embedding in ('ry_per_qubit', 'rx_per_qubit')

    @property
    def order(self) -> int:
        return self.n_qubits if self.k is None else self.k

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    @property
    def terms(self) -> list[tuple[int, ...]]:
        """Pauli-Z strings in D, in the same order as the old model.py."""
        return [c for i in range(1, self.order + 1)
                for c in combinations(range(self.n_qubits), i)]

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params(cfg: QVRConfig, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Same initial distributions as the old model.py, drawn from `rng`."""
    def _t(a):
        return torch.tensor(a, dtype=DTYPE, requires_grad=True)
    return {
        'alpha': _t(rng.uniform(0, 2 * np.pi, cfg.n_layers * cfg.n_qubits * 3)),
        'mu':    _t(rng.uniform(0, 2 * np.pi, cfg.n_terms)),
        'sigma': _t(rng.uniform(0, 2 * np.pi, cfg.n_terms)),
        'eta_0': _t(rng.uniform(-1, 1)),
    }


def load_legacy_checkpoint(path) -> tuple[dict, QVRConfig, dict]:
    """
    Load an old qvr_cervix `trained_model.pt` (fixed embedding only).
    Returns (params, cfg, norm_config). sigma_lambda / tau were not stored in
    old checkpoints; they only matter for training, so defaults are used.
    """
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    mc = ckpt['model_config']
    strategy = mc.get('embedding_strategy', 'fixed')
    if strategy != 'fixed':
        raise ValueError(f"only embedding='fixed' is supported, got {strategy!r}")
    cfg = QVRConfig(n_qubits=mc['n_qubits'], n_layers=mc['n_layers'],
                    k=mc.get('k'), N_E=mc['N_E'])
    params = {name: torch.as_tensor(ckpt['model_state_dict'][name], dtype=DTYPE)
              .detach().clone()
              for name in ('alpha', 'mu', 'sigma', 'eta_0')}
    norm_config = ckpt.get('norm_config', {'norm': 'none', 'norm_params': {}})
    return params, cfg, norm_config


# ---------------------------------------------------------------------------
# Single source of W
# ---------------------------------------------------------------------------

def transform_ops(alpha, n_qubits: int, n_layers: int, ansatz: str = 'chain') -> None:
    """Apply W(alpha). The only place the variational ansatz is defined."""
    a = alpha.reshape(n_layers, n_qubits, 3)
    if ansatz == 'sel':
        qml.StronglyEntanglingLayers(a, wires=range(n_qubits))
        return
    for layer in range(n_layers):
        for q in range(n_qubits):
            qml.RX(a[layer, q, 0], wires=q)
            qml.RY(a[layer, q, 1], wires=q)
            qml.RZ(a[layer, q, 2], wires=q)
        if layer < n_layers - 1:
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])


def W_matrix(alpha: torch.Tensor, cfg: QVRConfig) -> torch.Tensor:
    """(dim, dim) complex unitary of W, differentiable in alpha."""
    W = qml.matrix(transform_ops, wire_order=list(range(cfg.n_qubits)))(
        alpha, cfg.n_qubits, cfg.n_layers, cfg.ansatz)
    return torch.as_tensor(W).to(CDTYPE)


# ---------------------------------------------------------------------------
# Basis bookkeeping (MSB = qubit 0, matching PennyLane)
# ---------------------------------------------------------------------------

def basis_bits(cfg: QVRConfig) -> torch.Tensor:
    """(dim, n_qubits) bit of qubit q in basis state b."""
    b = torch.arange(cfg.dim).unsqueeze(1)
    shifts = torch.arange(cfg.n_qubits - 1, -1, -1).unsqueeze(0)
    return ((b >> shifts) & 1).to(DTYPE)


def z_signs(cfg: QVRConfig) -> torch.Tensor:
    """(dim, n_qubits) eigenvalue of Z_q on basis state b: +1 / -1."""
    return 1.0 - 2.0 * basis_bits(cfg)


def term_parities(cfg: QVRConfig) -> torch.Tensor:
    """(n_terms, dim) eigenvalue of each Pauli-Z string on basis state b."""
    z = z_signs(cfg)
    return torch.stack([z[:, list(c)].prod(dim=1) for c in cfg.terms])


def eigenvalues(mu: torch.Tensor, cfg: QVRConfig) -> torch.Tensor:
    """(dim,) eigenvalues of the mean Hamiltonian H = ½ Σ_j mu_j P_j."""
    return 0.5 * (mu.to(DTYPE) @ term_parities(cfg))


# ---------------------------------------------------------------------------
# Matrix backend
# ---------------------------------------------------------------------------

def embed_state(x: torch.Tensor, cfg: QVRConfig) -> torch.Tensor:
    """
    Embedded product state, (..., dim) complex, closed form.
      'ry_all'       x (...)     RY(x) on every qubit            (qvr_cervix)
      'ry_first'     x (...)     RY(x) on qubit 0, others |0>    (GW paper)
      'ry_per_qubit' x (..., Q)  RY(x_q) on qubit q              (Baker paper text)
      'rx_per_qubit' x (..., Q)  RX(x_q) on qubit q              (Baker notebook AngleEmbedding)
    Each qubit: cos(x/2)|0> + a·sin(x/2)|1>, a = 1 (RY) or −i (RX).
    """
    Q = cfg.n_qubits
    if cfg.per_qubit_input:
        if x.shape[-1] != Q:
            raise ValueError(f'{cfg.embedding} needs x[..., {Q}], got {tuple(x.shape)}')
        xq = x
    elif cfg.embedding == 'ry_all':
        xq = x.unsqueeze(-1).expand(*x.shape, Q)
    else:  # ry_first
        xq = torch.zeros(*x.shape, Q, dtype=x.dtype)
        xq[..., 0] = x
    c = torch.cos(xq / 2).to(CDTYPE).unsqueeze(-2)            # (..., 1, Q)
    s = torch.sin(xq / 2).to(CDTYPE).unsqueeze(-2)
    if cfg.embedding == 'rx_per_qubit':
        s = -1j * s
    bits = basis_bits(cfg)                                   # (dim, Q)
    return torch.where(bits == 0, c, s).prod(dim=-1)         # (..., dim)


def probs_after_W(params: dict, x: torch.Tensor, cfg: QVRConfig) -> torch.Tensor:
    """p_b(x) = |<b| W E(x)|0>|^2 : (..., dim). Time-average over T for p̄_b."""
    W = W_matrix(params['alpha'], cfg)
    phi = embed_state(x, cfg).to(CDTYPE) @ W.T
    return phi.abs() ** 2


def expvals_from_D(params: dict, x: torch.Tensor, t: torch.Tensor,
                   D: torch.Tensor, cfg: QVRConfig, keep_draws: bool = False) -> torch.Tensor:
    """
    Per-qubit <Z_q> for explicit D values.

    x : (B, T) or (B, T, Q) for per-qubit embeddings    t : (T,)
    D : broadcastable to (B, S, T, n_terms)
    returns (B, T, n_qubits) averaged over draws, or (B, S, T, n_qubits) if keep_draws
    """
    W = W_matrix(params['alpha'], cfg)                       # (dim, dim)
    psi = embed_state(x, cfg).to(CDTYPE)                     # (B, T, dim)
    phi = psi @ W.T                                          # W ψ
    angles = D * t.view(1, 1, -1, 1)                         # (B, S, T, n_terms)
    phase = -0.5 * (angles @ term_parities(cfg))             # (B, S, T, dim)
    phi_d = phi.unsqueeze(1) * torch.exp(1j * phase)         # D W ψ
    chi = phi_d @ W.conj()                                   # W† D W ψ
    z = (chi.abs() ** 2) @ z_signs(cfg)                      # (B, S, T, Q)
    return z if keep_draws else z.mean(dim=1)


def sample_eps(shape, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(*shape, dtype=DTYPE, generator=generator)


def build_D(params: dict, mode: str, cfg: QVRConfig, B: int, T: int,
            n_draws: int | None = None, eps: torch.Tensor | None = None,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """
    D values for a D mode, broadcastable to (B, S, T, n_terms).
      'mc'       eps shape (B, S, T, n_terms)
      'shared'   eps shape (B, S, n_terms)  → one draw per (series, draw), shared across T
      'averaged' eps shape (B, S, n_terms)  → averaged over S, shared across T
    Pass `eps` for common random numbers; otherwise it is drawn from `generator`.
    """
    mu, sig = params['mu'], params['sigma'].abs()
    S = n_draws if n_draws is not None else cfg.N_E
    if mode == 'mc':
        if eps is None:
            eps = sample_eps((B, S, T, cfg.n_terms), generator)
        return mu + sig * eps
    if mode == 'shared':
        if eps is None:
            eps = sample_eps((B, S, cfg.n_terms), generator)
        return (mu + sig * eps).view(B, S, 1, cfg.n_terms)
    if mode == 'averaged':
        if eps is None:
            eps = sample_eps((B, S, cfg.n_terms), generator)
        return (mu + sig * eps.mean(dim=1)).view(B, 1, 1, cfg.n_terms)
    if mode == 'mean':
        return mu.view(1, 1, 1, -1)
    if mode == 'identity':
        return torch.zeros(1, 1, 1, cfg.n_terms, dtype=DTYPE)
    raise ValueError(f'unknown D mode {mode!r}; choose from {D_MODES}')


def expvals(params: dict, x: torch.Tensor, t: torch.Tensor, cfg: QVRConfig,
            mode: str = 'mc', n_draws: int | None = None,
            eps: torch.Tensor | None = None,
            generator: torch.Generator | None = None) -> torch.Tensor:
    """(B, T, n_qubits) per-qubit <Z_q> under a D mode."""
    B, T = x.shape[:2]
    D = build_D(params, mode, cfg, B, T, n_draws, eps, generator)
    return expvals_from_D(params, x, t, D, cfg)


def series_score(params: dict, x: torch.Tensor, t: torch.Tensor, cfg: QVRConfig,
                 mode: str = 'mc', n_draws: int | None = None,
                 eps: torch.Tensor | None = None,
                 generator: torch.Generator | None = None):
    """
    Per-series cost under cfg.cost, plus draw-averaged <Z_q>.
      'outside': mean_t (η₀ − mean_S mean_q <Z_q>)² / 4
      'inside' : mean_t mean_S (η₀ − mean_q <Z_q>)² / 4
    returns (score (B,), z (B, T, Q))
    """
    B, T = x.shape[:2]
    D = build_D(params, mode, cfg, B, T, n_draws, eps, generator)
    zd = expvals_from_D(params, x, t, D, cfg, keep_draws=True)   # (B, S, T, Q)
    m = zd.mean(dim=-1)                                          # (B, S, T)
    if cfg.cost == 'outside':
        score = ((params['eta_0'] - m.mean(dim=1)) ** 2 / 4).mean(dim=-1)
    else:
        score = ((params['eta_0'] - m) ** 2 / 4).mean(dim=(1, 2))
    return score, zd.mean(dim=1)


def series_cost(params: dict, z: torch.Tensor) -> torch.Tensor:
    """Reconstruction score per series: mean_t (eta_0 - mean_q <Z_q>)^2 / 4 → (B,)."""
    return ((params['eta_0'] - z.mean(dim=-1)) ** 2 / 4).mean(dim=-1)


def arctan_penalty(params: dict, cfg: QVRConfig) -> torch.Tensor:
    return cfg.sigma_lambda / math.pi * torch.arctan(
        2 * math.pi * cfg.tau * params['sigma'].abs()).mean()


def training_loss(params: dict, x: torch.Tensor, t: torch.Tensor, cfg: QVRConfig,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """0.5 * mean_batch(cost under cfg.draw_mode / cfg.cost, N_E draws) + penalty.
    Defaults ('mc', 'outside') are the qvr_cervix / Baker-notebook objective."""
    score, _ = series_score(params, x, t, cfg, mode=cfg.draw_mode, generator=generator)
    return 0.5 * score.mean() + arctan_penalty(params, cfg)


# ---------------------------------------------------------------------------
# PennyLane reference (tests only)
# ---------------------------------------------------------------------------

def reference_qnodes(cfg: QVRConfig):
    """Gate-by-gate circuits mirroring old model.py, for equivalence tests."""
    dev = qml.device('default.qubit', wires=cfg.n_qubits)
    Q, L = cfg.n_qubits, cfg.n_layers

    def _diag(angles):
        for j, comb in enumerate(cfg.terms):
            pairs = [comb[i:i + 2] for i in range(len(comb) - 1)]
            for p in pairs:
                qml.CNOT(wires=list(p))
            qml.RZ(angles[j], wires=comb[-1])
            for p in reversed(pairs):
                qml.CNOT(wires=list(p))

    def _embed(x):
        if cfg.embedding == 'ry_all':
            for q in range(Q):
                qml.RY(x, wires=q)
        elif cfg.embedding == 'ry_first':
            qml.RY(x, wires=0)
        elif cfg.embedding == 'ry_per_qubit':
            for q in range(Q):
                qml.RY(x[q], wires=q)
        else:
            for q in range(Q):
                qml.RX(x[q], wires=q)

    @qml.qnode(dev, interface='torch')
    def full(x, alpha, angles):
        _embed(x)
        transform_ops(alpha, Q, L, cfg.ansatz)
        _diag(angles)
        qml.adjoint(transform_ops)(alpha, Q, L, cfg.ansatz)
        return [qml.expval(qml.PauliZ(q)) for q in range(Q)]

    @qml.qnode(dev, interface='torch')
    def after_W(x, alpha):
        _embed(x)
        transform_ops(alpha, Q, L, cfg.ansatz)
        return qml.state()

    return full, after_W


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import time

    torch.set_printoptions(precision=4)
    ok = True

    def check(name, cond, detail=''):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f'  ({detail})' if detail else ''))

    cfg = QVRConfig(n_qubits=3, n_layers=3, N_E=10)
    params = init_params(cfg, np.random.default_rng(0))
    B, T = 4, 17
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(B, T, generator=g, dtype=DTYPE) * 2 - 1) * math.pi
    x[:, 0] = 0.0
    t = torch.linspace(0, 2 * math.pi, T, dtype=DTYPE)

    print('=' * 64)
    print(f'qvr.circuit self-test   config={cfg.to_dict()}')
    print(f'  n_terms={cfg.n_terms}  terms={cfg.terms}')
    print(f'  params: ' + ', '.join(f'{k}{tuple(v.shape)}' for k, v in params.items()))
    print('=' * 64)

    # ---- shapes and basic invariants -------------------------------------
    print('Shapes / invariants')
    W = W_matrix(params['alpha'], cfg)
    eye = torch.eye(cfg.dim, dtype=CDTYPE)
    check('W is unitary', torch.allclose(W.conj().T @ W, eye, atol=1e-12),
          f'shape {tuple(W.shape)}')
    lam = eigenvalues(params['mu'], cfg)
    check('eigenvalues shape', lam.shape == (cfg.dim,), f'{tuple(lam.shape)}')
    p = probs_after_W(params, x, cfg)
    check('p_b shape', p.shape == (B, T, cfg.dim), f'{tuple(p.shape)}')
    check('p_b sums to 1', torch.allclose(p.sum(-1), torch.ones(B, T, dtype=DTYPE)))
    for mode in D_MODES:
        z = expvals(params, x, t, cfg, mode=mode, generator=g)
        in_range = bool((z.abs() <= 1 + 1e-12).all())
        check(f"expvals '{mode}'", z.shape == (B, T, cfg.n_qubits) and in_range,
              f'shape {tuple(z.shape)}, |Z|<=1: {in_range}')
    cost = series_cost(params, expvals(params, x, t, cfg, mode='mean'))
    check('series_cost shape', cost.shape == (B,), f'{cost.detach().numpy().round(4)}')

    # ---- analytic checks ---------------------------------------------------
    print('Analytic checks')
    z_id = expvals(params, x, t, cfg, mode='identity')
    check("'identity' gives <Z_q> = cos(x)",
          torch.allclose(z_id, torch.cos(x).unsqueeze(-1).expand_as(z_id), atol=1e-12))
    z_t0 = expvals(params, x[:, :1], t[:1], cfg, mode='mc', generator=g)
    check('t=0 gives <Z_q> = cos(x) in any mode',
          torch.allclose(z_t0, torch.cos(x[:, :1]).unsqueeze(-1).expand_as(z_t0), atol=1e-12))
    xs = torch.linspace(-math.pi, math.pi, 400, dtype=DTYPE)
    P = probs_after_W(params, xs, cfg).detach().numpy()
    F = np.stack([np.ones(400)] + [f(k * xs.numpy()) for k in (1, 2, 3)
                                    for f in (np.cos, np.sin)], axis=1)
    resid = np.abs(F @ np.linalg.lstsq(F, P, rcond=None)[0] - P).max()
    check('p_b(x) is a degree-3 trig polynomial (Fact 1)', resid < 1e-10,
          f'max residual {resid:.1e}')

    # ---- equivalence with gate-by-gate PennyLane reference ----------------
    print('Equivalence with PennyLane reference (gate-by-gate)')
    full, after_W = reference_qnodes(cfg)
    D = build_D(params, 'mc', cfg, B, T, n_draws=1, generator=g)   # (B,1,T,nt)
    z_mat = expvals_from_D(params, x, t, D, cfg).detach()
    z_ref = torch.stack([torch.stack(full(x[b, i], params['alpha'].detach(),
                                          D[b, 0, i].detach() * t[i]))
                         for b in range(B) for i in range(T)]).view(B, T, -1)
    err = (z_mat - z_ref).abs().max().item()
    check('<Z_q> matches reference (random D draw)', err < 1e-10, f'max |Δ| {err:.1e}')
    st = torch.stack([after_W(x[0, i], params['alpha'].detach()) for i in range(T)])
    err_p = ((st.abs() ** 2) - p[0].detach()).abs().max().item()
    check('p_b matches reference statevector', err_p < 1e-10, f'max |Δ| {err_p:.1e}')

    # ---- other configurations (Baker notebook / paper, GW embedding) ---------
    print('Other configurations vs PennyLane reference')
    for kw in (dict(n_qubits=2, n_layers=3, k=2, embedding='rx_per_qubit', ansatz='sel'),
               dict(n_qubits=2, n_layers=3, k=2, embedding='ry_per_qubit', ansatz='sel'),
               dict(n_qubits=2, n_layers=3, k=2, embedding='ry_first', ansatz='chain')):
        c2 = QVRConfig(**kw)
        p2 = init_params(c2, np.random.default_rng(1))
        x2 = (torch.rand(3, 9, 2, generator=g, dtype=DTYPE) * 2 - 1) * math.pi
        if not c2.per_qubit_input:
            x2 = x2[..., 0]
        t2 = torch.linspace(0.1, 2 * math.pi, 9, dtype=DTYPE)
        D2 = build_D(p2, 'mc', c2, 3, 9, n_draws=1, generator=g).detach()
        zm = expvals_from_D(p2, x2, t2, D2, c2).detach()
        f2, _ = reference_qnodes(c2)
        zr = torch.stack([torch.stack(f2(x2[b, i], p2['alpha'].detach(), D2[b, 0, i] * t2[i]))
                          for b in range(3) for i in range(9)]).view(3, 9, -1)
        e2 = (zm - zr).abs().max().item()
        check(f"{kw['embedding']}/{kw['ansatz']} matches reference", e2 < 1e-10, f'max |Δ| {e2:.1e}')
    cb = QVRConfig(n_qubits=2, n_layers=3, k=2, embedding='rx_per_qubit', ansatz='sel')
    pb = init_params(cb, np.random.default_rng(2))
    xb2 = (torch.rand(4, 9, 2, generator=g, dtype=DTYPE) * 2 - 1) * math.pi
    tb = torch.linspace(0.1, 2 * math.pi, 9, dtype=DTYPE)
    eb = sample_eps((4, 10, 9, cb.n_terms), g)
    s_out, _ = series_score(pb, xb2, tb, cb, mode='mc', eps=eb)
    s_in, _ = series_score(pb, xb2, tb, QVRConfig(**{**cb.to_dict(), 'cost': 'inside'}), mode='mc', eps=eb)
    check("cost 'inside' ≥ 'outside' (Jensen: adds draw variance)",
          bool((s_in >= s_out - 1e-15).all()), f'mean gap {(s_in - s_out).mean().item():.2e}')
    p0 = {k: v.detach().clone() for k, v in pb.items()}
    p0['sigma'] = torch.zeros_like(p0['sigma'])
    sh, _ = series_score(p0, xb2, tb, cb, mode='shared', generator=g)
    mn, _ = series_score(p0, xb2, tb, cb, mode='mean')
    check("'shared' with sigma=0 equals 'mean'", torch.allclose(sh, mn, atol=1e-14))

    # ---- gradients ----------------------------------------------------------
    print('Gradients')
    loss = training_loss(params, x, t, cfg, generator=g)
    loss.backward()
    for name, v in params.items():
        good = v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
        check(f'grad {name}', good,
              f'|grad|={v.grad.abs().sum().item():.3e}' if v.grad is not None else 'None')
    print(f'  training loss = {loss.item():.6f}')

    # ---- timing -------------------------------------------------------------
    print('Timing (N_E=10, mode=mc)')
    xb = x.repeat(64, 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        expvals(params, xb, t, cfg, mode='mc', generator=g)
    dt = time.perf_counter() - t0
    print(f'  matrix backend: {xb.shape[0]} series × {T} timepoints × {cfg.N_E} draws '
          f'in {dt * 1e3:.1f} ms  ({xb.shape[0] / dt:,.0f} series/s)')

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _self_test()