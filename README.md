# qvr_final — cross-dataset QVR analysis

Lean rewrite of the Quantum Variational Rewinding (QVR) pipeline from `qvr_cervix`,
built to run the same set of analyses on several time-series datasets and ask
whether findings such as eigenstate fingerprinting recur beyond cervical imaging.
Framed as an analysis of QVR's potential, not a performance claim.

- **Repo (server):** `/scratch90/chris_/quantum/qvr_final` (conda env `qml`)
- **Old code (read-only reference):** `/scratch90/chris_/quantum/qvr_cervix`
- **Porting record:** [`docs/project_progress_092326_porting.md`](docs/project_progress_092326_porting.md)
- **Grade replication:** [`docs/project_progress_092326_grade_replication.md`](docs/project_progress_092326_grade_replication.md)
- **Analysis plan:** [`docs/ANALYSIS_PLAN.md`](docs/ANALYSIS_PLAN.md)

## Layout

```
qvr_final/
├── qvr/                      core package
│   ├── circuit.py            the ONE circuit definition + fast matrix backend
│   ├── train.py              training loop (seeded, early stopping, manifest)
│   ├── infer.py              scores, Z-repr, p̄_b; seeded draws; cache
│   ├── data.py               SeriesSet, Normalizer, t_grid, roc_auc
│   └── datasets/
│       └── cervical.py       DYSIS cervical adapter (.pt splits)
├── scripts/
│   ├── gate_cervical_val.py  reproduction gate: old checkpoint → val AUC
│   ├── explore_fingerprint.py  exploratory: response curves, p̄_b vs moments
│   └── replicate_grade.py    old grade result: reproduce + dissect
│   ├── prepare_dataset.py    converters → split dirs (mitbih, synthetic)
│   ├── run_suite.py          full E1–E5 suite for any split dir
│   └── score_audit.py        QVR vs classical comparators
├── tests/
│   └── test_legacy_equivalence.py   new code vs old model.py / spectral_utils.py
├── docs/                     plans, decisions, progress records
└── results/                  run outputs (gitignored except small JSON summaries)
```

## Modules

| File | What it does |
| --- | --- |
| `qvr/circuit.py` | Fixed-embedding QVR circuit. W is defined once (`transform_ops`, PennyLane); everything else is batched torch linear algebra. D modes `mc`, `averaged`, `mean`, `identity`. Also p_b, eigenvalues, loss, legacy checkpoint loader. |
| `qvr/train.py` | Trains on normalized normal-only series. One `run_seed` → independent init / batch / draw / monitor streams. Early stopping on a fixed monitor set from the training split. Writes `params.pt`, `history.json`, `manifest.json`. |
| `qvr/infer.py` | One pass per dataset: `score`, `zrepr`, `pbar` (+ per-timepoint arrays). Draws keyed on (seed, series index), so permuted copies share draws. `permute_time` for shuffle tests. Hash-keyed cache. |
| `qvr/data.py` | `SeriesSet` (series, labels, unit ids). `Normalizer` fit on training normals (`per_timepoint` legacy-exact; `per_timepoint_q` quantile + clip). `t_grid` = linspace(0, 2π, T). Rank-based `roc_auc`. |
| `qvr/datasets/cervical.py` | Loads `cervical_{train,val,test}.pt` (smoothness-filtered). None-voted → 0, any-voted → 1, patient as unit. |
| `scripts/gate_cervical_val.py` | Loads an old checkpoint, normalizes val with its own params, reports recon AUC per D mode vs a reference. Val only. |
| `scripts/explore_fingerprint.py` | EXPLORATORY. Eigenstate response curves; p̄_b vs Fourier moments with patient-grouped CV on training-split lesions. |
| `scripts/replicate_grade.py` | Reproduces the old val→test grade result, then dissects it: moments, patient-level permutation null, train→test transfer. Reads test labels (logged). |
| `scripts/run_suite.py` | The full per-dataset suite for any split directory: trains the fixed model over N_E × seeds, scores every set (D modes, time-permuted copies, p̄_b), and runs E1–E5 → `summary.md` / `summary.json`; resumable, parallel (`--workers`). |
| `scripts/prepare_dataset.py` | Converters into the split-dir layout: `mitbih` (inter-patient DS1/DS2 split, MLII by name, train-normal normalization) and `synthetic` (qvr_repr PoC generator, fixed scaling). |
| `scripts/score_audit.py` | Dataset-agnostic score audit: QVR vs prespecified and post hoc classical comparators (paired bootstrap), rank combination, channel knockout. |
| `tests/test_legacy_equivalence.py` | Checks circuit output, loss, gradients, paper inference path, eigenvalues and p_b against the old code; optional real checkpoint. |

## D modes

| Mode | Meaning |
| --- | --- |
| `mc` | Fresh draws per series × timepoint × draw; outputs averaged. Matches the training objective. |
| `averaged` | N_E draws averaged before the circuit, one D̄ per series. The old paper's inference estimator. |
| `mean` | D = μ (deterministic). |
| `identity` | D = 0 (no evolution). |

## Self-tests

Every module prints PASS/FAIL checks on fake data. Run from the repo root:

```bash
python -m qvr.circuit
python -m qvr.train
python -m qvr.infer
python -m qvr.data
python -m qvr.datasets.cervical
python scripts/gate_cervical_val.py --self-test
python tests/test_legacy_equivalence.py --old-src /scratch90/chris_/quantum/qvr_cervix/src \
    [--checkpoint <old trained_model.pt>]
```

If temp-folder cleanup fails on `/scratch90`, set `export TMPDIR=/tmp`.

## Conventions

- **One circuit.** Nothing outside `circuit.py` builds a circuit or re-implements the embedding.
- **Provenance.** Every run records git commit, config hash, data hash, seed and versions. Train with `strict_git=True` once the repo is committed.
- **Splits.** Normalizers are fit on training normals only. Test data is not touched until analyses are prespecified.
- **Shuffle semantics.** Time permutation acts on normalized data (representation level); frame 0 stays in place.
- **Self-test per script.** New scripts ship with a `_self_test()` that runs the main path on fake data.


## Drop-in dataset layout

Any dataset enters the pipeline as a **split directory** of pickles, each an array
`(N, d, T)` (series, channels, timepoints), float64:

| File | Contents | Required |
| --- | --- | --- |
| `Xtr.pickle` | normal training series (the only data models and comparators are fit on) | yes |
| `Xte_norm.pickle` | normal test series | yes |
| `Xte_<name>.pickle` | one file per anomalous test set (e.g. `Xte_ventricular`) | ≥ 1 |
| `Xval.pickle` | anomalous validation series (threshold tuning, paper protocol) | for E0-style runs |
| `Xval_norm.pickle` | normal validation series (clean threshold tuning) | recommended |
| `Xprobe.pickle` | anomalous series from TRAINING units (never seen by QVR, which trains on normals); E4 probe fitting | for E4 |
| `<set>_units.pickle` | `(N,)` independent-unit id per series (patient, record, window) | recommended |
| `<set>_subtype.pickle` | `(N,)` subtype label per series (E4 probes) | for E4 |
| `manifest.json` | source, version, preprocessing, normalizer parameters, hashes | yes |

Values are normalized to [−π, π] with a normalizer fit on `Xtr` only (see `qvr/data.py`).
`qvr/datasets/baker.py::load_all` reads any such directory (returns `(N, T, d)` tensors).

## Data

| Dataset | Location | Status |
| --- | --- | --- |
| Cervical (DYSIS) | `/scratch90/chris_/quantum/qvr_cervix/data/cervix/holdout_experiment/` | Adapter done |
| MIT-BIH | TBD | Planned |
| Baker crypto | `/scratch90/chris_/quantum/QuantumVariationalRewinding` | Planned |
| Gravitational-wave | TBD | Planned |
| Best-case synthetic | generated | Planned |