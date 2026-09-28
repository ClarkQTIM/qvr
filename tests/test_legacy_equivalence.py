"""
tests/test_legacy_equivalence.py — new circuit vs the old qvr_cervix code.

Imports the OLD model.py and spectral_utils.py from a directory (default: the
old repo's src/) and checks, on identical parameters and inputs, that the new
matrix backend reproduces:

  1. old training QNode output for an explicit D           (circuit definition)
  2. old training loss AND gradients, same eps draws       (training objective)
  3. old fast-inference scores + Z-repr (predict_with_representations,
     sigma=0 so its unseeded D̄ draw is deterministic)     (paper inference path)
  4. old spectral_utils eigenvalues and p_b                (fingerprint path)
  5. optionally, a real trained checkpoint                 (--checkpoint)

Usage:
    python tests/test_legacy_equivalence.py --old-src /path/to/qvr_cervix/src \
        [--checkpoint /path/to/run/training/trained_model.pt]
"""

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-src', required=True, help='dir containing old model.py')
    ap.add_argument('--checkpoint', default=None, help='old trained_model.pt (optional)')
    args = ap.parse_args()
    sys.path.insert(0, args.old_src)
    import model as old_model            # noqa: E402
    import spectral_utils as old_spec    # noqa: E402

    ok = True

    def check(name, err, tol=1e-10):
        nonlocal ok
        good = err < tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}  (max |Δ| {err:.1e})")

    cfg = C.QVRConfig(n_qubits=3, n_layers=3, N_E=4, sigma_lambda=1e-2)
    B, T = 3, 17
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(B, T, generator=g, dtype=C.DTYPE) * 2 - 1) * math.pi
    x[:, 0] = 0.0
    t = torch.linspace(0, 2 * math.pi, T, dtype=C.DTYPE)

    def make_old(params):
        m = old_model.QuantumVariationalRewindingModel(
            n_qubits=cfg.n_qubits, n_layers=cfg.n_layers, N_E=cfg.N_E,
            embedding_strategy='fixed', diff_method='backprop',
            sigma_lambda=cfg.sigma_lambda)
        for k, v in params.items():
            m.parameters[k] = v.detach().clone().requires_grad_(True)
        return m

    new_p = C.init_params(cfg, np.random.default_rng(7))
    old = make_old(new_p)
    print('=' * 64)
    print('Legacy equivalence: new qvr.circuit vs old model.py / spectral_utils.py')
    print('=' * 64)

    # 1. circuit output for explicit D -------------------------------------
    D = C.build_D(new_p, 'mc', cfg, B, T, n_draws=1, generator=g).detach()
    z_new = C.expvals_from_D(new_p, x, t, D, cfg).mean(-1).detach()
    z_old = torch.stack([old.qnode(x[b, i], t[i], D[b, 0, i]) for b in range(B)
                         for i in range(T)]).view(B, T).detach()
    check('1. mean-Z for explicit D (old training QNode)', (z_new - z_old).abs().max().item())

    # 2. training loss and gradients with shared eps -------------------------
    eps = torch.randn(B, cfg.N_E, T, cfg.n_terms, dtype=C.DTYPE, generator=g)
    queue = iter([eps[b, s] for b in range(B) for s in range(cfg.N_E)])
    old._sample_D_reparam = lambda n: (old.parameters['mu'].unsqueeze(0)
                                       + old.parameters['sigma'].abs().unsqueeze(0)
                                       * next(queue))
    loss_old = old.forward(x, t)
    loss_old.backward()
    p2 = {k: v.detach().clone().requires_grad_(True) for k, v in new_p.items()}
    z = C.expvals(p2, x, t, cfg, mode='mc', eps=eps)
    loss_new = 0.5 * C.series_cost(p2, z).mean() + C.arctan_penalty(p2, cfg)
    loss_new.backward()
    check('2a. training loss', abs(loss_new.item() - loss_old.item()))
    for k in p2:
        check(f'2b. gradient {k}',
              (p2[k].grad - old.parameters[k].grad).abs().max().item())

    # 3. old fast inference path (paper numbers), sigma = 0 --------------------
    p3 = {k: v.detach().clone() for k, v in new_p.items()}
    p3['sigma'] = torch.zeros_like(p3['sigma'])
    old3 = make_old(p3)
    s_old, r_old = old3.predict_with_representations(x, t)
    z3 = C.expvals(p3, x, t, cfg, mode='averaged', generator=g)
    check('3a. predict_with_representations scores',
          np.abs(C.series_cost(p3, z3).numpy() - s_old).max())
    check('3b. predict_with_representations Z-repr',
          np.abs(z3.mean(dim=1).numpy() - r_old).max())

    # 4. spectral_utils --------------------------------------------------------
    cache = old_spec.extract_model_cache(old)
    lam_old = old_spec.compute_eigenvalues(cache['mu'], cfg.n_qubits, cfg.order)
    check('4a. eigenvalues', np.abs(C.eigenvalues(new_p['mu'], cfg).detach().numpy()
                                    - lam_old).max())
    qn = old_spec.build_intermediate_state_qnode(cache)
    p_old = np.stack([np.abs(qn(float(v))) ** 2 for v in x[0]])
    check('4b. p_b (intermediate-state qnode)',
          np.abs(C.probs_after_W(new_p, x[0], cfg).detach().numpy() - p_old).max())

    # 5. real checkpoint ---------------------------------------------------------
    if args.checkpoint:
        print(f'Checkpoint: {args.checkpoint}')
        pl, cfg_l, norm = C.load_legacy_checkpoint(args.checkpoint)
        print(f'  config={cfg_l.to_dict()}  norm={norm.get("norm")}')
        sys.path.insert(0, args.old_src)
        import train as old_train  # noqa: E402
        oldm = old_train.load_trained_model(args.checkpoint)
        oldm.parameters['sigma'] = torch.zeros_like(oldm.parameters['sigma'])
        pl0 = copy.deepcopy(pl)
        pl0['sigma'] = torch.zeros_like(pl0['sigma'])
        s_o, r_o = oldm.predict_with_representations(x, t)
        zl = C.expvals(pl0, x, t, cfg_l, mode='mean')
        check('5a. checkpoint scores (sigma=0)',
              np.abs(C.series_cost(pl0, zl).numpy() - s_o).max())
        check('5b. checkpoint Z-repr (sigma=0)', np.abs(zl.mean(1).numpy() - r_o).max())

    print('=' * 64)
    print('RESULT:', 'ALL PASS' if ok else 'FAILURES ABOVE')
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    main()