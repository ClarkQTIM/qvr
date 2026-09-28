# Project progress — 2026-09-23 — Porting `qvr_cervix` → `qvr_final`

Record of the soft-reset port: what was carried over, what was cut, how
equivalence with the old code was verified, the first reproduction on real
data, and what is still open. Status labels: **Verified** (tested on the
server), **Observed** (one run, not yet replicated), **Open** (not done).

---

## 1. Goal of the port

Extract only what is needed to run the same analyses on new datasets, with
provenance tight enough that any cross-dataset difference can be attributed to
the data rather than to the code. The old repo stays untouched as the reference.

Old repo: `/scratch90/chris_/quantum/qvr_cervix` · New repo: `/scratch90/chris_/quantum/qvr_final`

## 2. What was ported

| Old (`qvr_cervix/src`) | New (`qvr_final`) | Change |
| --- | --- | --- |
| `model.py` (fixed embedding path) | `qvr/circuit.py` | Rewritten as one circuit definition + batched matrix backend; W still defined in PennyLane via `qml.matrix` |
| `spectral_utils.py` (`build_intermediate_state_qnode`, `compute_eigenvalues`, `compute_probs`) | `qvr/circuit.py` (`probs_after_W`, `eigenvalues`) | Same maths, vectorized; no second circuit copy |
| `train.py` (loop, save) + `run_sweep.py` (seeding) | `qvr/train.py` | Seed split into independent streams; monitor-based early stopping; full config + manifest saved |
| `evaluate.py` / `validate.py` (`_inference_worker`) | `qvr/infer.py` | Single pass for score / Z-repr / p̄_b; named D mode; seeded common random numbers; cache keyed by weights |
| `data_loaders.py` (`apply_norm`, `.pt` loading) | `qvr/data.py`, `qvr/datasets/cervical.py` | Generic container + normalizer; cervical adapter; out-of-range reporting |

## 3. What was cut

- Learned, per-layer and re-uploading embeddings (only `fixed` is used)
- Spread loss, L1/L2 representation training, `WithState` + von Neumann loss (`losses.py` was missing)
- Sequential per-timepoint QNodes, parallel worker pools, heartbeat/status JSON, worker logs
- Plots, timing banners, legacy `.npy` loaders, embedded sweep grid
- Dynamics features (V_λ, revival, F(τ)) — functions of p̄_b and Λ; recoverable from saved outputs if needed
- Baselines, spatial Dice, MedSAM, biopsy, presentation scripts (belong to the parked cervical paper)

## 4. Old-code findings that shaped the port

| Finding | Where | Consequence in new code |
| --- | --- | --- |
| Paper inference averages N_E draws of D **before** the circuit (one D̄ per segment); training averages circuit **outputs** over per-timepoint draws | `model.predict_with_representations` vs `_get_time_series_cost_batched` | Both available as named modes: `averaged` (paper) and `mc` (training) |
| Inference RNG unseeded (`np.random.default_rng()`) | `model.py` | Draws keyed on (seed, series index); reruns identical; permuted copies share draws |
| Embedding circuit duplicated in ≥5 places (source of the July circuit bug) | `model.py`, `spectral_utils.py`, analysis scripts | Single `transform_ops`; all consumers import `circuit.py` |
| Early stopping on single 8-sample batch loss | `train.py` | Fixed monitor set from the training split with fixed draws |
| Checkpoints omit σ_λ, τ, seed | `train.save_training_results` | Full config, hyperparameters and seed saved; manifest with hashes |
| Spectral cache reused if file exists | `validate.py` | Cache key includes weights, config, data hash, git commit |
| Per-timepoint scaling uses raw min/max, no clipping | `data_loaders.py` | Legacy-exact kept for cervical; quantile + clip option for new datasets; out-of-range fraction reported |
| `ts_all[pid][:-1]` drops a row and silently skips mismatched patients | `_UnfilteredNoneVotedDataset` | Unfiltered population not ported yet; flagged as likely source of the voting-alignment bug |
| Seeds 0–4 differ in init **and** data subset | `run_sweep.run_one` | Old seed spreads are genuine restarts; existing weights kept |
| Circuit order embed → W → D → W† (operator W†DW) | `model.py` | Matches Baker; walkthrough's "W·D·W† deviation" was notation |
| t-grid `linspace(0, 2π, 17)` in training and inference | `data_loaders.py`, `evaluate.py` | `data.t_grid(T)` |

## 5. Verification (all run on the server, 2026-09-23)

### 5.1 Self-tests — **Verified**

| Module | Result | Notes |
| --- | --- | --- |
| `qvr.circuit` | ALL PASS | W unitary; D=I and t=0 give ⟨Z⟩=cos(x); p_b(x) is a degree-3 trig polynomial (residual 4.4e-16); matches gate-by-gate PennyLane (5.3e-15); all gradients finite; 2,916 series/s at N_E=10 |
| `qvr.train` | ALL PASS | 20-batch loop; files + manifest; reload identical; same seed bit-identical; different seed different init; 69.9 ms/batch at N_E=2 |
| `qvr.infer` | ALL PASS | Batch-size invariant; seed reproducible; D modes consistent; p̄_b and `identity` score invariant under permutation; cache hit/miss; 6,773 series/s at N_E=10 |
| `qvr.data` | ALL PASS | Normalizer matches legacy formula and `from_legacy`; AUC matches pairwise definition |
| `qvr.datasets.cervical` | ALL PASS | After temp-dir cleanup fix (network filesystem) |
| `gate_cervical_val --self-test` | ALL PASS | Plumbing only |

### 5.2 Legacy equivalence — **Verified**

`tests/test_legacy_equivalence.py` against the old `model.py` and `spectral_utils.py`:

| Check | Max abs difference |
| --- | --- |
| Circuit output for explicit D (old training QNode) | 3.4e-15 |
| Training loss, same draws | 0 |
| Gradients α / μ / σ / η₀ | ≤ 6.6e-17 |
| Paper inference path scores / Z-repr (σ=0) | 5.6e-17 / 3.5e-16 |
| Eigenvalues (½ convention) | 0 |
| p_b vs intermediate-state QNode | 6.1e-16 |
| Real checkpoint `q2_l3_fixed_ne10_s0` scores / Z-repr | 8.3e-17 / 8.3e-16 |

The new code computes the same quantities as the old code, including training gradients.

### 5.3 Speed

| Task | Old | New |
| --- | --- | --- |
| Training step (batch 8, N_E=10) | ~1.26 s | 14.5 ms (sandbox); server N_E=10 timing pending |
| Full cervical val, 735,668 segments | ~20 min with 32 workers at N_E=1 (test set) | 34 s (`averaged`), 115 s (`mc`, N_E=10), single process |

Why: the circuit is 3 qubits (8 amplitudes). W depends only on α, so it is built once per step; the embedding has a closed form; D is diagonal (phases). Each evaluation is two 8×8 products, batched over segments × timepoints × draws. Simulation only; hardware deployment still uses the PennyLane circuit.

## 6. First reproduction on real data

`scripts/gate_cervical_val.py`, checkpoint `q3_l3_fixed_ne10_s0_norm_per_timepoint_sl1.00e-2`,
filtered val split (735,668 segments; 22,903 lesion; 39 patients), inference seed 0.
Two runs gave identical numbers.

| D mode | Val recon AUC |
| --- | --- |
| `averaged` (paper estimator) | 0.7431 |
| `mc` (training objective) | 0.7498 |
| `mean` (D = μ) | 0.8237 |

- **Verified:** values outside [−π, π] after normalization are 0.0002% (normal) and 0.0113% (lesion) — aliasing is negligible for cervical.
- **Verified — gate PASSED.** The old pipeline's own full-population result
  (`unfiltered_val/summary.json`, `smooth_Any_vs_smooth_None`) is **0.74315**; the new
  pipeline gives **0.7431** on the same checkpoint, same population, same estimator
  (difference < 0.0001; old draws unseeded, new seeded). Legacy equivalence also passes
  on this checkpoint (`q3_l3_fixed_ne10_s0`).

### 6.1 All five seeds (new pipeline, filtered val)

| Seed | `averaged` | `mc` | `mean` | mean − mc | learned \|σ\| |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.743 | 0.750 | 0.824 | +0.074 | 3 of 7 terms at 2–6 rad, rest ≈ 0 |
| 1 | 0.722 | 0.751 | 0.762 | +0.012 | 5 of 7 large |
| 2 | 0.742 | 0.764 | 0.822 | +0.058 | 5 of 7 large |
| 3 | 0.756 | 0.776 | 0.822 | +0.046 | 3 of 7 large |
| 4 | 0.776 | 0.776 | 0.776 | 0 | all ≈ 0.001–0.003 (collapsed) |
| Mean ± SD | 0.748 ± 0.020 | 0.763 ± 0.013 | 0.801 ± 0.030 | +0.038 | |

- **Verified:** with σ collapsed (seed 4) all estimators coincide, so the between-mode gap is caused by σ.
- **Observed:** σ is bimodal (≈ 0 or 2–6 rad); D = μ beats `mc` on 4 of 5 seeds. The gap size is not explained by which terms dephase (seeds 1 and 2 share the same large-σ terms, gaps +0.012 vs +0.058). Cervical is the discovery dataset; ΔAUC(mean − mc) is a prespecified primary contrast for the others.

### 6.2 Where the old reference numbers came from

| Old number | Source | Meaning |
| --- | --- | --- |
| 0.778 (paper "val AUC") | not in any results file; seed 0 `all_Any_vs_all_None` = 0.776 | Most likely the 5-seed mean on the **unfiltered** population (jagged included), which the new cervical adapter does not load |
| AUC_Δ +0.037 | `temporal_sensitivity_s{0..4}.json`: 0.040, 0.022, 0.015, 0.039, 0.069 | Mean over seeds, 5,000-segment subsample, unseeded draws |
| shuf_r 0.963 | same files: 0.976, 0.934, 0.948, 0.971, 0.989 | Mean over seeds; only seeds 1–2 individually pass shuf_r < 0.97 |
| Subsample AUC 0.739 … 0.784 | same files | 5,000 segments (~150 lesions) → SE ≈ 0.02 |

- **Jaggedness confound confirmed** (seed 0, old summary): jagged lesion vs smooth normal AUC 0.938; smooth vs smooth 0.743; all vs all 0.776.
- **Observed:** seed 4 (σ ≈ 0, no draw noise) has the highest old shuf_r (0.989). Consistent with unseeded draws deflating old shuf_r for the other seeds; E2 with common random numbers will measure this cleanly.

## 7. Open items

1. ~~Old seed-0 reference~~ — found (§6.2); gate passed.
2. ~~Gate seeds 1–4~~ — done (§6.1).
3. ~~Learned μ, |σ|, η₀~~ — done (§6.1).
4. ~~Legacy test on `q3_l3_fixed_ne10_s0`~~ — passed.
5. Server timing: 49.8 ms/batch at N_E=10 with 96 threads; re-time with 4 threads for the parallel grid.
6. `git init`, commit, push; tag `plan-v1`; then enable `strict_git=True`.
7. Leftover `/scratch90/chris_/tmp_old_delete` (background delete exited 1; likely files held open).
8. Unfiltered population and the voting-alignment check before any unfiltered numbers are used.
9. Prespecify the D-mode comparison and estimator choice in `ANALYSIS_PLAN.md` before new datasets run.

## 8. Next

Dataset adapters for MIT-BIH and Baker crypto on the `SeriesSet` interface, then the per-dataset analysis set: N_E sweep with shuffle sensitivity, class-conditional fingerprints, D-mode ablation, Hamiltonian interaction-order summary.