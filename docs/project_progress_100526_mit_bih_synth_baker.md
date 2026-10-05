# Project progress — 2026-10-05 — Full suite on MIT-BIH and synthetic data

First runs of the complete per-dataset suite (`ANALYSIS_PLAN.md` §2–§7) on two new
datasets, plus the score audit and the exploratory time-aware probe. Status labels:
**Verified** (run on the server, prespecified), **Exploratory** (logged in §11 as such),
**Open** (not done).

**Bottom line.** On real ECG beats, QVR is a weak anomaly detector. Every reasonable
classical detector beats it significantly (pooled AUC 0.70–0.72 vs 0.57), and its own
untrained cosine formula (0.65) beats the trained circuit. More random draws make its
score strongly order-sensitive, as predicted, but that sensitivity does not help. Its
representations lose subtype information relative to the raw beat, and the fingerprint
again equals six Fourier moments. On the synthetic data, detection is trivially perfect;
the order-invariant features cannot separate time-reversed classes (by construction),
while QVR's time-averaged ⟨Z⟩ can, because it is measured after the time-dependent
evolution.

---

## 1. Infrastructure added

| Component | Purpose |
| --- | --- |
| `scripts/prepare_dataset.py` | Converters into the split-directory layout, with figures (`class_means`, `examples`, `value_hist`) and E4 eligibility per subtype |
| `scripts/run_suite.py` | Full suite for any split directory: 25 models (N_E ∈ {1, 3, 5, 10, 20} × seeds 0–4), scoring under four D modes plus time-permuted copies, E1–E5, hand-off to the score audit; resumable, parallel, `--strict-git` |
| `scripts/score_audit.py` | QVR vs classical comparators with paired bootstrap, rank combination, channel knockout |
| `scripts/e4_temporal.py` | Exploratory time-aware subtype probe (§11, 2026-09-28) |

Fixed model (§2): 3 qubits, 3 layers, single-pass RY(x) on every qubit, CNOT chain, k = 3,
cost squared outside the draw mean, fresh draws per timepoint, σ_λ = 1e-2, τ = 1, Adam,
batch 8, ≤ 50,000 samples, monitor early stopping, t = linspace(0, 2π, T). Every model
records commit and `dirty = False`.

## 2. Data

### 2.1 MIT-BIH Arrhythmia Database — **Verified**

- **Source:** PhysioNet mitdb 1.0.0 (wfdb 4.x), `qvr_repr/data/mit-bih-arrhythmia-database-1.0.0`.
- **Split:** inter-patient (de Chazal 2004). DS1 → 17 train + 5 val records (seed 0);
  DS2 → 22 test records; paced records excluded; no record in two splits.
- **Preprocessing:**
  - MLII lead selected by name;
  - whole-record zero-phase band-pass 0.5–40 Hz (removes baseline wander, which dominated the
    first conversion);
  - beat window 0.25 s before to 0.45 s after the R-peak, downsampled ×2 → T = 126 at 180 Hz;
  - per-timepoint train-normal quantiles (0.5%, 99.5%) → [−π, π], clipped.
- **Classes:** AAMI normal / supraventricular (SVEB) / ventricular (VEB) / fusion.

| | Ours | Standard (de Chazal 2004) |
| --- | --- | --- |
| DS1 N / SVEB / VEB / F | 45,848 / 944 / 3,788 / 414 | 45,866 / 944 / 3,788 / 415 |
| DS2 N / SVEB / VEB / F | 44,240 / 1,837 / 3,220 / 388 | 44,259 / 1,837 / 3,221 / 388 |

The small normal shortfall is beats too close to a record edge for the window.

- **Clipped values:** 1.0% train normals, 1.5% test normals, 15.1% VEB, 4.3% fusion, 0.4% SVEB.
- **Limitation:** single-beat windows carry no RR-interval (prematurity) information, the main
  marker of SVEB, so SVEB detection is weak for every method.
- **Fixes relative to the old `qvr_repr` loader:**
  1. random beat-level split replaced by an inter-patient split;
  2. pooled scaling (normals and anomalies) replaced by scaling fit on training normals only;
  3. channel 0 replaced by MLII selected by name.

### 2.2 Synthetic (old qvr_repr generator) — **Verified**

Normal flat noise N(0, 0.1); growing and falling ramps 0↔1 + N(0, 0.05) (exact time-mirrors
with the same multiset of values); oscillating sine (2 periods). T = 16. Fixed scaling
x·π/1.5, not fitted, nothing clipped (§11). 1,000 training normals, 200 per test class.

## 3. Synthetic results — **Verified**

- **E1 and E3:** AUC 1.000 for every class, N_E, seed and D mode. Detection is trivial on
  this generator and says nothing about QVR.
- **E2:** shuf_r 0.999–1.000 at every N_E (Spearman with log N_E −0.14). The scalar score
  ignores order here.
- **E4 (p̄_b vs moments):** macro AUC 0.835 vs 0.839, both p = 0.001. Balanced accuracy
  ≈ 0.667 = (1 + ½ + ½)/3, exactly what perfect oscillating plus chance between
  growing and falling gives. Order-invariant features cannot separate time-reversed twins,
  as predicted.
- **E5:** σ collapses with more draws (3.19 → 0.26 for N_E ≥ 5); μ barely moves
  (mean |Δμ| 0.09–0.20).

**Time-aware probe — Exploratory.**

| Features | macro AUC (logreg, reference) | p |
| --- | --- | --- |
| `z_t` per-timepoint ⟨Z_q⟩ | 1.000 | 0.001 |
| `z_mean` time-averaged ⟨Z_q⟩ | 1.000 | 0.010 |
| `raw` series | 1.000 | 0.001 |
| `pbar` | 0.835 | 0.001 |
| `moments` | 0.839 | 0.001 |

`z_mean` separation rises with N_E (0.910, 0.952, 0.970, 0.967, 0.977). **Registered
prediction failed for `z_mean`:** it was labelled order-invariant, but it is measured after
the time-dependent D(ε, t), so each timepoint is weighted differently. Only p̄_b (before D)
is order-invariant. QVR's dynamics give its Z-representation access to order, though any
time-weighted average does the same (`raw` = 1.000). Correction logged in §11.

## 4. MIT-BIH results — **Verified**

### 4.1 E1 detection (D mode `mc`, AUC mean ± SD over seeds)

| Set | N_E = 1 | 3 | 5 | 10 | 20 | Reference, seed-avg [95% record CI] |
| --- | --- | --- | --- | --- | --- | --- |
| fusion | 0.802 ± 0.047 | 0.809 | 0.812 | 0.821 | 0.824 | 0.823 [0.552, 0.877] |
| SVEB | 0.411 ± 0.019 | 0.453 | 0.453 | 0.490 | 0.512 | 0.484 [0.301, 0.574] |
| VEB | 0.786 ± 0.050 | 0.658 | 0.619 | 0.588 | 0.569 | 0.593 [0.468, 0.723] |
| pooled | 0.660 ± 0.028 | 0.600 | 0.577 | 0.572 | 0.568 | 0.572 [0.467, 0.677] |

Detection of ventricular beats, morphologically the most distinct class, falls as N_E
grows. CIs are wide because there are only 22 test records.

### 4.2 E2 order sensitivity

| N_E | shuf_r | ΔAUC_shuf (pooled) |
| --- | --- | --- |
| 1 | 0.646 | −0.009 |
| 3 | 0.378 | −0.046 |
| 5 | 0.287 | −0.063 |
| 10 | 0.174 | −0.052 |
| 20 | 0.123 | −0.042 |

Spearman(log N_E, shuf_r) = **−0.965**: the prespecified prediction from cervical is
confirmed. But ΔAUC is negative, so scrambled beats are detected slightly *better*. The
order sensitivity is real and unhelpful.

### 4.3 E3 D modes (pooled AUC, mean over seeds)

| N_E | mc | mean | identity | averaged |
| --- | --- | --- | --- | --- |
| 1 | 0.660 | 0.693 | 0.651 | 0.636 |
| 3 | 0.600 | 0.604 | 0.645 | 0.613 |
| 5 | 0.577 | 0.613 | 0.645 | 0.600 |
| 10 | 0.572 | 0.613 | 0.649 | 0.600 |
| 20 | 0.568 | 0.563 | 0.651 | 0.602 |

Prespecified contrasts at N_E = 10 (seed-avg, record CI): mean − mc +0.044 [−0.095, +0.178];
mc − identity −0.077 [−0.306, +0.145]. The direction matches cervical and Baker (noise and
dynamics do not help), but the record-level CIs span zero.

With D = 0, W cancels and the score is mean_t (η₀ − cos x_t)² / 4, a one-parameter
classical formula. It (≈ 0.65) beats the trained stochastic circuit (0.57) at N_E ≥ 3.

### 4.4 Score audit: QVR vs classical comparators (reference, seed-avg)

| Set | QVR | window mean | trig moments | per-t z² * | late mean * | final value * |
| --- | --- | --- | --- | --- | --- | --- |
| VEB | 0.592 | **0.801** − | **0.890** − | **0.904** − | 0.735 − | 0.427 + |
| fusion | **0.823** | 0.746 + | 0.633 + | **0.946** − | 0.502 + | 0.730 + |
| SVEB | 0.484 | 0.504 − | 0.386 + | 0.356 + | 0.558 − | 0.293 + |
| pooled | 0.572 | **0.697** − | **0.702** − | **0.722** − | 0.659 − | 0.403 + |

`−`: comparator significantly higher; `+`: QVR significantly higher (paired bootstrap 95% CI
excludes 0). \* post hoc comparators (added after the Baker results). Rank combination
QVR + window mean vs window mean alone: +0.004 pooled (fusion +0.099).

**Interpretation.**
- Pooled, every reasonable classical detector significantly beats QVR. QVR adds almost
  nothing to the window mean.
- The "no headroom" hypothesis (the periodic embedding folds extreme values back toward
  normal) is **weakened**: trig moments see the same clipped, periodic values and still
  reach 0.890 on VEB.
- The likelier cause is QVR's scoring: one ⟨Z⟩ per timepoint compared with a single learned
  constant η₀, rather than a distance from the normals' distribution.

### 4.5 E4 subtype probes (SVEB / VEB / fusion; fit on training-record anomalies, evaluated on DS2)

All three classes are eligible (≥ 5 records in fitting and evaluation sets).

| Features | macro AUC (logreg) | p (record-level null) |
| --- | --- | --- |
| `raw` beat * | **0.945** | 0.003 |
| `z_t` * | 0.826 | 0.010 |
| Fourier moments | 0.815 | 0.032 |
| p̄_b | 0.810 | 0.038 |
| `z_mean` * | 0.706 | 0.184 |

\* exploratory time-aware probe. Across N_E, `z_t` is flat within noise (0.76, 0.75, 0.78,
0.80, 0.79). κ is unreliable here (`raw` has the best AUC and the lowest κ, 0.31): test
anomalies are mostly VEB and class mixes differ between DS1 and DS2, so argmax decisions are
poorly calibrated. Macro AUC is the metric to read.

- The fingerprint equals the moments (fourth dataset).
- Every QVR representation loses subtype information relative to the raw beat.

### 4.6 E5

| N_E | min gap | S₁ / S₂ / S₃ (init) | mean abs Δμ | mean abs σ init → trained |
| --- | --- | --- | --- | --- |
| 1 | 0.175 | 0.45/0.43/0.12 (0.45/0.42/0.12) | 0.42 | 3.19 → 3.14 |
| 10 | 0.234 | 0.45/0.44/0.11 (0.45/0.42/0.12) | 3.48 | 3.19 → 3.17 |
| 20 | 0.447 | 0.44/0.46/0.10 (0.45/0.42/0.12) | 4.67 | 3.19 → 3.46 |

- μ moves a lot, but S_r stays at its initialization (≈ term counts 3/7, 3/7, 1/7). Because μ
  enters as a phase (μ·t), |μ| is not an interaction strength. **E5's S_r is uninformative
  as designed**; report it as such.
- σ stays near 3 rad on MIT-BIH (phases fully scrambled over t ∈ [0, 2π]) but collapses on the
  easy synthetic data.
- No exact eigenvalue degeneracies, so the eigenbasis label stays defined.

## 5. Problems found and fixed during these runs

| Problem | Effect | Fix |
| --- | --- | --- |
| First commit staged nothing but results: `git add` listed a nonexistent `docs/` and git refused the whole command | `plan-v1` tag pointed at a commit without plan or code | Docs moved into `docs/`, full project committed, tag re-created; §11 entry |
| `train()` creates the run folder before the git check | Refused runs leave empty folders | **Open:** move the check first (harmless; skip by `params.pt`) |
| Score hand-off ranked each test set separately | Score-audit QVR AUC = 0.5 by construction | Joint ranking; self-test now checks it reproduces E1 |
| E5 had no initialization reference | S_r looked meaningful but equals the init proportions | Init S_r, Δμ and σ change reported |
| `z_mean` mislabelled order-invariant | Failed registered prediction | Correction logged in §11 |

## 6. Cross-dataset status

| Finding | Cervical | Baker | Synthetic | MIT-BIH |
| --- | --- | --- | --- | --- |
| Fingerprint = six Fourier moments | ✓ | ✓ (function class) | ✓ | ✓ |
| Dynamics / noise do not help detection | ✓ | ✓ | ceiling | ✓ (direction; CIs span 0) |
| Simple classical ≥ QVR | ✓ (grade: moments tie) | ✓ (published scaling) | ceiling | ✓ (significant) |
| More draws → more order-sensitive score | ✓ (old code) | — | ✗ (shuf_r ≈ 1) | ✓ (ρ = −0.97) |
| Order sensitivity helps detection | — | ✗ | — | ✗ (ΔAUC < 0) |

## 7. Decisions

- **Headroom check dropped.** Trig moments, equally periodic and clipped, reach 0.890 on VEB,
  so headroom is not the main explanation (§11).
- Next: GW audit (E0) with a registered prediction, then the best-case synthetic, then the
  write-up.

## 8. Open

1. `train.py`: run the git check before creating the run folder.
2. Per-term σ for MIT-BIH (the means here can hide the "off or fully dephased" split seen on
   cervical and Baker).
3. GW audit; best-case synthetic; write-up.

## 9. Files

| Output | Location |
| --- | --- |
| Split directories + figures + manifests | `data_splits/{mitbih,synthetic}/` (gitignored; hashes in manifests) |
| Suite outputs | `results/suite/{mitbih,synthetic}/` (`summary.md`, `summary.json`, `models/`, `scores/`) |
| Score audit | `results/suite/mitbih/score_audit/score_audit.json` |
| Time-aware probe | `results/suite/{mitbih,synthetic}/e4_temporal/e4_temporal.json` |