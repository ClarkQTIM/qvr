"""
scripts/gate_cervical_val.py — AUC-level reproduction gate on real cervical data.

Loads an old qvr_cervix checkpoint, normalizes the val split with the
checkpoint's own norm params, scores it with the new pipeline under several
D modes, and prints recon AUC next to a reference number.

'averaged' is the estimator that produced the old paper numbers (one D̄ per
segment, N_E draws averaged); with a fixed seed it is now reproducible.
'mean' and 'mc' show how much the estimator choice moves the AUC.

Usage (from repo root):
    python scripts/gate_cervical_val.py \
        --checkpoint .../q3_l3_fixed_ne10_s0_norm_per_timepoint_sl1.00e-2/training/trained_model.pt \
        --reference 0.778 --out results/gates/cervical_val_q3l3_s0.json

    python scripts/gate_cervical_val.py --self-test     # fake checkpoint + data
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvr import circuit as C                       # noqa: E402
from qvr.data import Normalizer, roc_auc, t_grid   # noqa: E402
from qvr.datasets import cervical                  # noqa: E402
from qvr.infer import infer                        # noqa: E402
from qvr.train import git_state                    # noqa: E402


def run_gate(checkpoint, data_dir, modes, seed, reference=None, out=None, split='val'):
    params, cfg, norm_config = C.load_legacy_checkpoint(checkpoint)
    nm = Normalizer.from_legacy(norm_config)
    ds = cervical.load_split(split, data_dir)
    Xn = nm.transform(ds.X)
    t = t_grid(ds.T)

    print('=' * 64)
    print(f'Gate: {Path(checkpoint).parent.parent.name}')
    print(f'  cfg={cfg.to_dict()}')
    print('  ' + ds.summary())
    oor = {c: nm.out_of_range(Xn[ds.y == c]) for c in (0, 1)}
    print(f'  values outside [-π, π]: normal {oor[0]:.4%}, lesion {oor[1]:.4%}')
    print('=' * 64)

    res = {'checkpoint': str(checkpoint), 'split': split, 'cfg': cfg.to_dict(),
           'n': int(len(ds.y)), 'n_pos': int(ds.y.sum()),
           'n_units': int(len(np.unique(ds.unit))), 'out_of_range': oor,
           'seed': seed, 'reference': reference, 'git': git_state(), 'auc': {}}
    for mode in modes:
        t0 = time.time()
        o = infer(params, cfg, Xn, t, mode=mode, seed=seed)
        auc = roc_auc(ds.y, o['score'])
        res['auc'][mode] = auc
        ref = f'   (reference {reference:.3f}, Δ {auc - reference:+.3f})' if reference else ''
        print(f'  recon AUC [{mode:8s}] = {auc:.4f}   {time.time() - t0:5.1f}s{ref}', flush=True)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=1))
        print(f'  saved → {out}')
    return res


def _self_test():
    import tempfile
    print('gate self-test: fake checkpoint + fake data (checks plumbing, not numbers)')
    cfg = C.QVRConfig(n_qubits=3, n_layers=3, N_E=10)
    p = C.init_params(cfg, np.random.default_rng(0))
    g = torch.Generator().manual_seed(0)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        none = torch.randn(400, 17, generator=g) * 0.05
        anyv = torch.randn(40, 17, generator=g) * 0.10 + 0.05
        none[:, 0] = anyv[:, 0] = 0
        torch.save({'none_time_series': none, 'any_time_series': anyv,
                    'none_patient_ids': np.array([f'{i % 9:05d}' for i in range(400)]),
                    'any_patient_ids': np.array([f'{i % 4:05d}' for i in range(40)])},
                   cervical.pt_path('val', tmp))
        lo, hi = none[:, 1:].min(0).values, none[:, 1:].max(0).values
        ck = Path(tmp) / 'run' / 'training' / 'trained_model.pt'
        ck.parent.mkdir(parents=True)
        torch.save({'model_state_dict': {k: v.detach() for k, v in p.items()},
                    'model_config': {'n_qubits': 3, 'n_layers': 3, 'N_E': 10, 'k': 3,
                                     'embedding_strategy': 'fixed'},
                    'norm_config': {'norm': 'per_timepoint',
                                    'norm_params': {'per_tp_min': lo.tolist(),
                                                    'per_tp_max': hi.tolist()}}}, ck)
        r = run_gate(ck, tmp, ['averaged', 'mean', 'mc', 'identity'], seed=0,
                     reference=0.5, out=Path(tmp) / 'gate.json')
        good = (all(0 <= v <= 1 for v in r['auc'].values()) and (Path(tmp) / 'gate.json').exists()
                and r['out_of_range'][0] == 0.0)
        print('RESULT:', 'ALL PASS' if good else 'FAIL')
        if not good:
            raise SystemExit(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint')
    ap.add_argument('--data-dir', default=str(cervical.DATA_DIR))
    ap.add_argument('--split', default='val', choices=['val'])
    ap.add_argument('--modes', nargs='+', default=['averaged', 'mean', 'mc'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reference', type=float, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test()
    else:
        if not a.checkpoint:
            ap.error('--checkpoint is required')
        run_gate(a.checkpoint, a.data_dir, a.modes, a.seed, a.reference, a.out, a.split)