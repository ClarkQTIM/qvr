"""
tests/test_baker_notebook_equivalence.py — new backend vs the Baker et al. notebook.

The notebook functions below are copied verbatim from QVR_example.ipynb
(cells 9, 11, 13, 17, 21), with only the covalent decorators removed and the
device switched from lightning.qubit to default.qubit (same exact statevector
maths). M_sample_func is replaced by a queue so both implementations see the
same D draws.

Checks, for the bivariate model (2 qubits, AngleEmbedding=RX, 3
StronglyEntanglingLayers, k=2):
  1. per-point expectation  get_anomaly_expec
  2. single-point cost      get_single_point_cost   (square OUTSIDE the draw mean)
  3. time-series cost       get_time_series_cost    (t = linspace(0.1, 2π, T))
  4. arctan penalty         arctan_penalty

Usage:  python tests/test_baker_notebook_equivalence.py
"""

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C  # noqa: E402

n_qubits = 2


# ------------------------- notebook code (verbatim) -------------------------

@qml.qnode(qml.device('default.qubit', wires=n_qubits, shots=None), interface='torch')
def get_anomaly_expec(x, t, D, alpha, wires, k, embed_func, transform_func, diag_func, observable,
                      embed_func_params={}, transform_func_params={}):

    embed_func(x, wires=wires, **embed_func_params) # U[x_i(t_j)]
    transform_func(alpha, wires, **transform_func_params) # W(\alpha)
    diag_func(D * t, n_qubits, k=k) # D(\epsilon, t_j)
    qml.adjoint(transform_func)(alpha, wires=range(n_qubits), **transform_func_params
        ) # W^{\dagger}(\alpha)

    # plug in \hat{O}_{\eta}
    coeffs = np.ones(len(wires))/len(wires) # scale by 1/n
    H = qml.Hamiltonian(coeffs, observable)
     # calculate expecation
    return qml.expval(H)


def create_diagonal_circuit(D, n, k=None):
    # D is a sub-set of the exponential number of eigenvalues.
    if k is None:
        k = n
    cnt = 0
    # Note there NO explicit loop over all 2^n eigenvalues
    for i in range(1, k + 1):
        for comb in combinations(range(n), i):
            if len(comb) == 1:
                qml.RZ(D[cnt], wires=[comb[0]])
                cnt += 1
            elif len(comb) > 1:
                cnots = [comb[i : i + 2] for i in range(len(comb) - 1)]
                for j in cnots:
                    qml.CNOT(wires=j)
                qml.RZ(D[cnt], wires=[comb[-1]])
                cnt += 1
                for j in cnots[::-1]:
                    qml.CNOT(wires=j)


def get_single_point_cost(x, t, alpha, eta_0, M_sample_func, sigma, mu, N_E, wires,
                          k, embed_func, transform_func, diag_func, observable,
                          embed_func_params={}, transform_func_params={}):
    expecs = torch.zeros(N_E)
    for i in range(N_E):
        D = M_sample_func(sigma, mu)
        expec = get_anomaly_expec(x, t, D, alpha, wires, k, embed_func, transform_func, diag_func, observable,
                                   embed_func_params={}, transform_func_params={})
        expecs[i] = expec
    mean = expecs.mean()
    # eta_0 separated from expectation expression since it was an identity matrix prefactor
    single_point_cost = (eta_0 - mean)**2/4 # 1/4 factor fom L = 4 in article
    return single_point_cost


def get_time_series_cost(xt, alpha, eta_0, M_sample_func, sigma, mu, N_E, wires, k, embed_func,
                         transform_func, diag_func, observable, t_cycler, embed_func_params={},
                         transform_func_params={}, start=0.1, end=2*np.pi):
    if t_cycler is None:
        t_idxs = np.arange(xt.shape[1])
        ts = np.linspace(start, end, xt.shape[1], endpoint=True)
    else:
        t_idxs = next(t_cycler)
        ts = np.linspace(start, end, xt.shape[1], endpoint=True)[t_idxs]
    xt_batch = xt[:, t_idxs]
    xfunct = zip(xt_batch.T, ts)
    a_func_t = \
        [get_single_point_cost(x, t, alpha, eta_0, M_sample_func, sigma, mu, N_E, wires,
                               k, embed_func, transform_func, diag_func, observable) for x, t in xfunct]
    single_time_series_cost = torch.tensor(a_func_t, requires_grad=True).mean()
    return single_time_series_cost


def arctan_penalty(sigma, contraction_hyperparameter):
    prefac = 1/(np.pi)
    sum_terms = torch.arctan(2*np.pi*contraction_hyperparameter*torch.abs(sigma))
    mean = sum_terms.mean()
    return prefac*mean

# -----------------------------------------------------------------------------


def main():
    ok = True

    def check(name, err, tol=1e-6):
        nonlocal ok
        good = err < tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}  (max |Δ| {err:.1e})")

    cfg = C.QVRConfig(n_qubits=2, n_layers=3, k=2, N_E=10, tau=15, sigma_lambda=1.0,
                      embedding='rx_per_qubit', ansatz='sel')
    p = C.init_params(cfg, np.random.default_rng(3))
    alpha_nb = p['alpha'].detach().reshape(3, 2, 3)
    mu, sigma, eta0 = p['mu'].detach(), p['sigma'].detach(), p['eta_0'].detach()
    g = torch.Generator().manual_seed(0)
    T = 12
    xt = (torch.rand(2, T, generator=g, dtype=torch.float64) * 2 - 1) * math.pi   # notebook layout (d, p)
    ts = torch.tensor(np.linspace(0.1, 2 * np.pi, T, endpoint=True), dtype=torch.float64)
    kw = dict(wires=range(2), k=2, embed_func=qml.templates.AngleEmbedding,
              transform_func=qml.templates.StronglyEntanglingLayers,
              diag_func=create_diagonal_circuit, observable=[qml.PauliZ(i) for i in range(2)])

    print('=' * 64)
    print('Baker notebook equivalence (bivariate: RX AngleEmbedding, 3 SEL layers, k=2)')
    print('=' * 64)

    # 1. per-point expectation
    D = mu + sigma * torch.randn(cfg.n_terms, generator=g, dtype=torch.float64)
    nb = float(get_anomaly_expec(xt[:, 4], ts[4], D, alpha_nb, **kw))
    new = C.expvals_from_D(p, xt.T[None, 4:5], ts[4:5], D.view(1, 1, 1, -1), cfg).mean().item()
    check('1. get_anomaly_expec', abs(nb - new), 1e-12)

    # 2 + 3. costs with shared draws (queue order: timepoint-major, then draw)
    eps = torch.randn(1, cfg.N_E, T, cfg.n_terms, generator=g, dtype=torch.float64)
    queue = iter([mu + sigma.abs() * eps[0, s, j] for j in range(T) for s in range(cfg.N_E)])
    sampler = lambda s_, m_: next(queue)  # noqa: E731
    nb1 = float(get_single_point_cost(xt[:, 0], ts[0], alpha_nb, eta0, sampler, sigma, mu,
                                      cfg.N_E, **kw))
    s1, _ = C.series_score(p, xt.T[None, :1], ts[:1], cfg, mode='mc', eps=eps[:, :, :1])
    # notebook stores expectations with torch.zeros(N_E) → float32, so agreement is ~1e-8
    check('2. get_single_point_cost (outside; notebook float32)', abs(nb1 - s1.item()), 1e-7)

    queue = iter([mu + sigma.abs() * eps[0, s, j] for j in range(T) for s in range(cfg.N_E)])
    nbT = float(get_time_series_cost(xt, alpha_nb, eta0, sampler, sigma, mu, cfg.N_E,
                                     t_cycler=None, **kw))
    sT, _ = C.series_score(p, xt.T[None], ts, cfg, mode='mc', eps=eps)
    check('3. get_time_series_cost (notebook float32)', abs(nbT - sT.item()), 1e-7)

    # 4. penalty
    check('4. arctan_penalty (tau=15)',
          abs(float(arctan_penalty(sigma, 15)) - C.arctan_penalty(p, cfg).item()), 1e-12)

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()