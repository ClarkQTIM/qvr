# ANALYSIS_PLAN.md — cross-dataset QVR analysis (v1)

**Status:** draft for freeze · **Freeze:** commit + tag `plan-v1` before any new-dataset QVR run.
**Pre-freeze revision (2026-09-23):** E4 strengthened after the cervical grade replication
(`docs/project_progress_092326_grade_replication.md`); cervical grade dropped as an outcome.
After the tag, every change is a dated entry in §11 (Deviations), never a silent edit.

---

## 1. Question and framing

Does the structure QVR learns on cervical acetowhitening — in particular the
eigenbasis fingerprint p̄_b and its class dependence — recur on other time-series
datasets, and how much of QVR's behaviour depends on its temporal evolution D(ε, t)?

Framed as an analysis of QVR's potential and mechanism, not a claim of superiority.
No classical baselines in v1. Cervical is the **discovery** dataset; the others test
whether its findings replicate.

## 2. Fixed model (no architecture search)

| Setting | Value |
| --- | --- |
| Circuit | 3 qubits, 3 layers, fixed single-pass RY(x) embedding, all Pauli-Z terms k = 3 |
| Rationale | Q = 3 is the smallest register with 1-, 2- and 3-body terms (§5 E5); matches the published cervical model. Not chosen by performance. |
| Objective | 0.5 · mean cost (`mc`, N_E draws) + σ_λ/π · mean arctan(2π τ \|σ\|); σ_λ = 1e-2, τ = 1 |
| Optimizer | Adam, lr 0.01, batch 8, max 50,000 samples |
| Early stopping | fixed monitor set (≤ 256 training normals, fixed draws), every 50 batches, patience 10 evaluations |
| Time grid | t = linspace(0, 2π, T) |
| Training data | normal-class series from the training split only |
| Code | `qvr_final` at the tagged commit; runs refuse a dirty tree (`strict_git=True`) |

**Normalization.** Cervical: legacy `per_timepoint` (min/max), to stay identical to the
published model. New datasets: `per_timepoint_q` with train-normal quantiles (0.005, 0.995)
and clipping to [−π, π]. Fraction of clipped values reported per split and class.

**Multichannel data.** One univariate model per channel. Each dataset's primary channel
is fixed in its Table 1 row before its first run; other channels are secondary.

## 3. Design axes

| Axis | Levels |
| --- | --- |
| Training N_E | 1, 3, 5, 10, 20 |
| Seeds | 0–4 (each seed = new init + new training subset) |
| Inference D mode | primary `mc` at the model's N_E; secondary `mean`, `identity`, `averaged` |
| Inference seed | 0 (common random numbers keyed on series index) |

Per dataset: 5 N_E × 5 seeds = 25 trained models. Cervical reuses the existing
`results_final_circuit_fixed` weights (same architecture and objective).

**Why `mc` is primary.** It is the quantity the model is trained to minimise. The
cervical seed-0 gate showed `mean` > `mc` on val; choosing `mean` as primary after
seeing that would be selection on val, so `mean` is a prespecified secondary contrast.

## 4. Data rules

1. Splits and independent units are fixed in the dataset's Table 1 row (§9) before any run.
2. Normalizers are fit on training normals only.
3. No model selection: every (N_E, seed) model is reported.
4. Probes are fit on val and applied cold to test.
5. Test scores are computed only after this plan is tagged; test is never used to choose anything.
6. Shuffles act on normalized data (representation level). A frame that is constant by
   construction (cervical frame 0) stays in place; otherwise all frames are permuted.

## 5. Experiments (same set for every dataset)

**E0 — Original-claim audit (Baker crypto, gravitational-wave).** Runs before E1–E5 on
those two datasets, with the ORIGINAL authors' model configuration (not the fixed model of §2).

*Sources:* Baker et al., arXiv:2210.16438 (released pickles, notebook, Zenodo 7258627);
Rodrigues de Miranda et al., Quantum Mach. Intell. 7:17 (2025), code `brodriguesdemiranda/QVR4GW`.

*Configurations to implement (checked against each paper and its code before running):*

| | Baker bivariate / trivariate | GW |
| --- | --- | --- |
| Qubits / embedding | 2 / 3; RY(x^c) on qubit c (one channel per qubit) | 2; RY(x) on qubit 0 only |
| W | 3 layers, Schuld et al. ansatz (exact gates from notebook) | 3 layers, RX·RY·RZ + one CNOT (control = encoded qubit) |
| D | k-local diagonal, Q terms | 3 diagonal entries (Walsh), last fixed at 0 |
| Draws | ε per (series, draw), shared across timepoints | ε per input, 10 draws |
| Cost | E_ε[Ω²] (expectation of the squared distance) | E_ε[Ω²] |
| τ, N_E | paper τ = 5 (notebook 15), N_E = 10 | τ = 15, N_E = 10 |
| Normalization | per-timepoint min/max to [−π, π] across instances | per-timestamp across series to [0, 2π] |
| Score | a_X (paper) and raw C₂ (notebook) — both | mean C_single over 4,096 tuples |

Note: both papers square inside the ε-expectation, E[Ω²] = (η₀ − E⟨O⟩)² + Var_ε⟨O⟩, which
penalizes draw variance. The qvr_cervix objective squares outside, (η₀ − E⟨O⟩)², which does not.
Both are implemented and reported (relevant to the σ-dephasing observation, E3).

*E0.1 Reproduce.* Train with each paper's protocol (Baker: 50 restarts, best by validation
balanced accuracy; GW: 5 runs), then with ours (Adam, monitor early stopping, 5 seeds).
Targets — Baker: validation balanced accuracy 0.82 (bivariate) / 0.77 (trivariate) and per-subset
test balanced accuracy / F1; GW: filtered accuracy 1.0 in 4/5 runs (one 0.596), unfiltered best
0.833, 0.904 at perfect recall. A reproduction within ±0.03 counts as reproduced; otherwise the
discrepancy is reported with its likely cause.

*E0.2 Simple baselines, prespecified, fit on normal training series only.* No baseline is tuned
on test labels; no baseline is added after results are seen.
- Baker: mean of cumulative volume (channel 1, as preprocessed); per-channel mean and SD;
  per-channel trig moments (cos, sin of x, k = 1–2) scored by Mahalanobis distance from training normals.
- GW: signal energy (mean x² of the raw, filtered strain); max |x|; per-series SD; trig moments of
  normalized values (k = 1–2) scored by Mahalanobis distance from training normals.

*E0.3 Mechanism.* On the reproduced models: D modes (`mc`, `mean`, `identity`, paper-shared);
time permutation after normalization with common random numbers; fit of the QVR score by a
per-timepoint additive trigonometric model (R², to confirm the function-class bound).

*E0.4 Protocol.* Every method is evaluated twice:
(a) the paper's own protocol (threshold tuned as published, including GW's reuse of test anomalies);
(b) a clean protocol — threshold-free AUC plus balanced accuracy with the threshold chosen on
anomalies disjoint from the test anomalies (GW: 2-fold split over the 78 events; Baker: released
validation set), with bootstrap 95% CIs over independent units (windows / events).

*Interpretation (fixed in advance):*

| Outcome under the clean protocol | Reading |
| --- | --- |
| Simple baseline ≥ QVR (CI overlapping or above) | The benchmark does not require QVR |
| QVR > all baselines, and `identity` or permutation destroys the gap | Evidence for a temporal/dynamical mechanism |
| QVR > baselines, ablations leave the gap | QVR helps, but not through its dynamics |
| Paper-protocol result not reproduced | Report the discrepancy first; E0.2–E0.4 then use our reproduction |

Findings are reported neutrally as an audit of what these benchmarks test; authors may be
contacted with results before submission.

**E1 — Detection.** Recon AUC of `score` (normal vs anomalous), val and test, per model.

**E2 — Temporal sensitivity.** For each model, score the normalized split and a
time-permuted copy (`permute_time`, seed 0) with the same inference seed.
Report `shuf_r` = Pearson r(ordered, permuted scores) and ΔAUC_shuf = AUC(ordered) − AUC(permuted).
Trend with N_E is the replication test of the cervical "N_E drives temporality" finding.

**E3 — D-mode ablation.** AUC under `mc`, `mean`, `identity`, `averaged`.
Prespecified contrasts: ΔAUC(mean − mc) and ΔAUC(mc − identity).

**E4 — Fingerprint.** p̄_b per series (8-dim).
- Class-conditional mean p̄_b with unit-level 95% CIs, per class and per subtype
  (subtypes defined in Table 1: arrhythmia class, crypto behaviour type, etc.).
- Eigenstate response curves p_b(x) per seed, over the data distribution (which states are used).
- Subtype probe: StandardScaler + multinomial LogisticRegression (C = 0.1), fit on the
  largest pool of units unseen by QVR (normal-only training leaves the anomalous units of the
  training split unseen), applied to test; also reported for a second transfer direction
  (e.g. train→val). Macro one-vs-rest AUC, balanced accuracy and confusion matrix; κ
  (linear-weighted for ordinal subtypes, unweighted for nominal ones such as AAMI classes)
  only where every class has ≥ 5 units in both fitting and evaluation sets. Eligibility is
  computed by the converter (`manifest.json` → `e4_eligibility`) before any run; ineligible
  classes are reported descriptively only.
- Unit-level permutation null for every probe result: 1,000 shuffles of subtype labels across
  fitting units (segments stay with their unit), refit, score with true labels; report the
  null mean, 95th percentile and p-value.
- Co-primary comparison: the identical probe on the six Fourier moments
  m̄ = mean_t(cos kx_t, sin kx_t), k = 1–3, reported next to every p̄_b result. Fact 1 predicts
  a tie for linear probes; one nonlinear probe (gradient-boosted trees) tests whether W adds
  anything beyond m̄.

**E5 — Hamiltonian.** From learned μ: eigenvalues λ_b = ½ Σ μ_j ∏ z_q(b); minimum
eigenvalue gap (degeneracy check); interaction-order weights
S_r = Σ_{|j| = r} |μ_j| / Σ_j |μ_j| for r = 1, 2, 3. Compared across seeds and datasets.
Learned |σ| reported alongside (dephasing context for E3).

**Best-case synthetic.** Constructed to sit in QVR's representable niche; generator,
parameters and expected outcome are written into its Table 1 row before generation.
Reported as a constructed illustration, not evidence about real data.

## 6. Statistics

- **Seed level:** mean ± SD over the 5 seeds for each (dataset, N_E).
- **Unit-level uncertainty:** 2,000 bootstrap resamples of independent units (patient,
  record, window) on seed-averaged per-series scores; 95% percentile CIs for AUC,
  ΔAUC (paired, same resample), κ.
- **N_E trend (E2):** Spearman ρ between log N_E and shuf_r over all 25 models per
  dataset; direction predicted from cervical: negative.
- **Multiplicity:** Holm across the primary contrasts within each dataset
  (E2 trend, E3 ΔAUC(mean − mc), E3 ΔAUC(mc − identity), E4 subtype probe vs its permutation null).
  Everything else is descriptive.
- Segment- or beat-level p-values are never reported.

## 7. Primary and secondary outcomes

| Primary (per dataset) | Secondary |
| --- | --- |
| E0 (Baker, GW): QVR vs prespecified baselines under the clean protocol | E0 paper-protocol reproduction; E0.3 mechanism |
| E2 N_E trend in shuf_r | E1 AUC per N_E; `averaged` mode |
| E3 ΔAUC(mean − mc), ΔAUC(mc − identity) | E4 class means; S_r across seeds |
| E4 subtype probe AUC on test vs permutation null, with the moments probe alongside | E4 nonlinear probe; response curves |

## 8. Claims we will not make

- QVR outperforms classical methods (not tested).
- The fingerprint carries temporal-order information (it is permutation-invariant by construction).
- F(τ) is a sufficient statistic for p̄_b; "additive R² ≈ 1 signals collapse".
- Seed stability of p̄_b probes implies stable learned quantum structure (predicted by Fact 1).
- p̄_b is a richer representation than low-order moments (p̄_b = A_W m̄; tested, ties).
- A probe result is significant on the strength of a bootstrap over evaluation units alone.
- "Eigenbasis" naming without the degeneracy check.

## 9. Table 1 — dataset specifications

Each row is completed and committed before that dataset's first QVR run.

| Field | Cervical | MIT-BIH | Baker crypto | Gravitational-wave | Best-case synthetic |
| --- | --- | --- | --- | --- | --- |
| Source / version | DYSIS holdout `.pt` | PhysioNet mitdb 1.0.0 (wfdb) | Baker et al. released pickles (Zenodo 7258627) | LIGO Livingston O3 via `gwqml` / GWOSC, per QVR4GW | generated |
| Series | 5×5-px segment, 17 frames | beat window around R-peak: 0.25 s before, 0.45 s after (90 + 162 samples at 360 Hz), downsampled ×2 → 180 Hz | 3-h window, 1-min bars | 1-s strain segment, 4,096 Hz | TBD |
| T | 17 | 126 | 180 (notebook scores every 4th → 45) | 4,096 | TBD |
| Primary channel | intensity | MLII (selected by name) | TBD | TBD (detector) | — |
| Normal class | none-voted | AAMI normal (N, L, R, e, j) | windows with no Whale Alert | background noise (5,000 segments) | TBD |
| Anomalous class | any-voted | AAMI supraventricular / ventricular / fusion (one test set each) | windows around large BTC/USDT transfers | 78 GWTC events (confident, marginal, auxiliary) | TBD |
| Subtypes (E4) | none (grade dropped; see grade-replication record) | AAMI class | asset × direction × single/multi | TBD | TBD |
| Independent unit | patient | record | window | segment / event | series |
| Split | 150/39/49 patients, seed 42 | inter-patient: DS1 → train/val by record (5 val records, seed 0); DS2 → test | released train/val/test | 4,000 train / 78 val / 78 test normals + 78 events (paper); clean 2-fold over events (E0.4) | TBD |
| Normalizer | legacy per_timepoint | per_timepoint_q | per_timepoint_q | per_timepoint_q | per_timepoint_q |
| Population note | smoothness-filtered | — | — | — | — |

## 10. Reproduction gate (precondition)

Before new-dataset runs: new-pipeline cervical val recon AUC under `averaged` matches the
old pipeline's value for the same seed and population within 0.005, and
`tests/test_legacy_equivalence.py` passes on `q3_l3_fixed_ne10_s0`.

## 11. Deviations

| Date | Section | Change | Reason |
| --- | --- | --- | --- |
| 2026-09-23 | §4 rule 5 | Cervical test labels read by `scripts/replicate_grade.py` (before tag) | Re-examines the old paper's already-reported val→test grade result; chooses nothing for the new analysis. Exploratory. |
| 2026-09-25 | §5 E0.1 | Three reproduction diagnostics added for Baker's paper-protocol result (val BA 0.82 not reproduced; best of 50 = 0.692): `paper_ry` (RY embedding per paper text), `paper_iter` (Powell capped at 2,000 iterations, 20,000-evaluation budget, 10 restarts), `nb_tau5` (notebook maths with τ = 5). All three reported regardless of outcome. | Fairness to the original claim. The E0.2–E0.4 comparisons and interpretation table are unchanged; baselines untouched. |
| 2026-09-25 | §5 E0.3 | Added data forensics for Baker (`scripts/baker_forensics.py`, no training): scaling reference population, split provenance against `data/large_data_sets`, embedding aliasing near ±π, QVR vs baseline AUC by anomaly extremeness and by transaction value. | `large_data_sets` found in the authors' repo (842 normal, 5,485 anomalous, three features); per-condition ranges suggest pooled scaling. Diagnostic; no model selection. |
| 2026-09-25 | §5 E0.1 | Follow-up reproduction on splits rebuilt from `large_data_sets` (`scripts/baker_build_sets.py`), paper maths (RX, square inside, shared draws, τ = 5, 50 × 2,000): bivariate and trivariate (3 qubits, k = 3) × training normals 100 (released) or 782 (all normals not in `Xte_norm`), plus one exploratory bivariate-782 run with global min-max refit on training normals and clipping to [−π, π]. Released `Xval`, `Xte_norm` and test sets throughout; baselines and interpretation table unchanged. | Forensics showed 682 unused normals, the third channel available, and global pooled scaling (not the SM's per-timepoint scaling). Last reproduction attempt for the 0.82 / 0.77; all five reported. |
| 2026-09-25 | §5 E0.2–E0.3 | POST HOC comparators added after seeing the Baker follow-up (QVR > vol_mean under normals-only scaling): final value, late-window mean, per-timepoint z² (primary channel, all channels); rank-based combination replaces the raw-score logistic combination (heavy tails made the latter uninformative); channel knockout. `scripts/score_audit.py`. | To identify what QVR's edge consists of. Labelled post hoc in every output; prespecified comparisons unchanged. |
| 2026-09-28 | §2, §9 | Synthetic dataset (qvr_repr PoC generator: normal, growing, falling, oscillating; T = 16) uses a FIXED affine scaling x·π/1.5 known from the generator instead of `per_timepoint_q`; MIT-BIH Table 1 row filled (inter-patient DS1/DS2). | Synthetic normals are pure noise, so any fit on them saturates every anomaly; a generator-defined scale involves no fitting and no leakage. |
| 2026-09-28 | §5 E4 | κ clarified: unweighted for nominal subtypes (AAMI), linear-weighted for ordinal; E4 eligibility (≥ 5 units per class in fitting and evaluation sets) computed by the converter before any run. | MIT-BIH subtypes are nominal; fusion beats cluster in few records. Set before any MIT-BIH run. || 2026-09-28 | tag | `plan-v1` re-created on the first commit that actually contains the plan and code. The original tag (4c204c7) held only `.gitignore` and result JSONs because a failed `git add` (nonexistent `docs/` path) staged nothing else. Plan text unchanged. | Provenance repair; no analysis affected. |
| 2026-09-28 | §5 E4 add-on | Correction: z_mean (time-averaged ⟨Z_q⟩) was labelled order-invariant and predicted ≈ chance on synthetic growing vs falling. It is measured after the time-dependent D(ε, t), so it is order-sensitive; observed macro AUC 0.91–0.98 (rising with N_E), 1.000 at the reference. Only p̄_b (before D) is order-invariant. Registered prediction failed for z_mean; held for p̄_b and moments. | Honest record of a failed prediction. |
