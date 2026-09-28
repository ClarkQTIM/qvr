# Project progress — 2026-09-23 — Cervical fingerprint and grade replication

Exploratory. Cervical is the discovery dataset. Status labels: **Verified**
(reproduced on the server), **Observed** (consistent across seeds, not yet
formally tested), **Open** (not done).

**Bottom line.** The old paper's grade result reproduces exactly, so the
computation was correct. It does not hold up as evidence:

- The fingerprint p̄_b carries no information beyond six Fourier moments of the
  normalized intensities.
- The headline CIN2+ AUC is not distinguishable from a patient-level permutation
  null (p ≈ 0.07).
- The 4-grade κ falls to zero when the probe is trained on 4× more patients.

What survives: detection, a modest CIN2+ signal (≈ 0.65–0.67 segment AUC across
independent transfers), and an interpretable "one normal eigenstate, lesions
leak out" picture.

---

## 1. The claim under test

From the parked cervical paper (`unfiltered_repr_results.json`,
`q3_l3_fixed_ne10`, 5 seeds). A spectral p̄_b probe fit on **val** smooth
main-grade lesion segments and applied cold to **test** gave:

- CIN2+ AUC 0.740 ± 0.001 (segment) and 0.827 ± 0.012 (patient)
- Grade κ 0.184 ± 0.005 (linear-weighted, segment)

This was described as beating all classical baselines on grade.

Probe protocol (`build_unfiltered_repr_probes.py`): StandardScaler +
LogisticRegression(C = 0.1, max_iter = 2000), no class weights. CIN2+ means grade ≥ 2.
Patient AUC uses the mean probability per patient. Labels come from `GUID`
(zero-padded to 5 digits) → `Overall Histology`. The CSV has 5,485 rows with
unique GUIDs.

## 2. Populations

Smooth any-voted segments of main-grade patients (Negative, CIN1, CIN2, CIN3).
Training-split lesions were never seen by QVR, which is trained on none-voted
segments only.

| Split | Segments | Patients | Negative | CIN1 | CIN2 | CIN3 |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 70,553 | 78 | 7 pt / 11,205 | 9 / 6,822 | 17 / 9,850 | 45 / 42,676 |
| Val | 20,979 | 19 | **1 pt / 4,983** | 3 / 3,014 | 4 / 3,344 | 11 / 9,638 |
| Test | 16,343 | 28 | **1 pt / 423** | 2 / 3,215 | 6 / 2,640 | 19 / 10,065 |

In val, the non-CIN2+ class is only 4 patients. One Negative patient (04081)
supplies every Negative segment, 24% of val.

## 3. Fingerprint exploration (`scripts/explore_fingerprint.py`)

Training split, 5-fold patient-grouped stratified CV, 5 seeds. Probes: the
plan's logistic regression and gradient-boosted trees (HGB).

### 3.1 Fact 1 on real data — **Verified**

p̄_b is an exact linear function of the six time-averaged moments
(cos kx, sin kx, k = 1–3) on real segments with each seed's trained W.
The maximum residual is 8.9e-16 to 1.3e-15.

### 3.2 p̄_b vs moments — **Verified**

| Task | p̄_b logreg | moments logreg | p̄_b HGB | moments HGB |
| --- | --- | --- | --- | --- |
| Detection AUC (segment) | 0.854–0.861 | 0.861 | 0.865–0.869 | 0.864 |
| CIN2+ AUC, in-train CV (segment) | 0.300–0.305 | 0.299 | 0.353–0.360 | 0.360 |
| Grade κ, in-train CV | −0.022 to −0.012 | −0.020 | −0.032 to −0.018 | −0.020 |

Ranges span the 5 seeds. Moment results are identical across seeds because
they do not depend on W.

- Under a linear probe, p̄_b and the moments tie, as the algebra requires.
  Under HGB, the learned basis adds at most ≈ 0.003.
- The in-train CV AUCs below 0.5 are consistent with the known negative bias of
  grouped CV when there is little or no signal and few units per class
  (16 non-CIN2+ patients). The out-of-split transfers in §4 point the same way.

### 3.3 Eigenstate response curves (`bin_curves.png`) — **Observed**

- Each seed learns one dominant "normal" eigenstate (p ≈ 0.85–0.9 near x = 0):
  |011⟩ in seeds 0 and 2, |100⟩ in seeds 1 and 3, |000⟩ in seed 4.
- Lesion values (≈ 0 to 2) sit where the dominant state falls off and one or two
  others rise. Anomaly shows up as leakage out of the normal state.
- The data occupy roughly [−1, 2] of [−π, π]. Most curve structure lies at
  |x| > 2, where there is almost no data, so effectively 2–3 of the 8 states
  are used. This is the per-timepoint min/max compression predicted from
  `data_loaders.py`.

## 4. Replication and dissection (`scripts/replicate_grade.py`)

Steps: **A** refit the old probe on the old caches; **B** new-code p̄_b vs old
caches; **C** the same protocol on moments; **D** a patient-level permutation
null (1,000 shuffles of grades across val patients, refit, score test with true
labels); **E** fit on training-split patients; **F** val composition.

### 4.1 Exact reproduction — **Verified**

- **B:** new p̄_b equals the old caches, max |Δ| 5.6e-16 to 1.1e-15 across seeds.
- **A:** the old numbers reproduce to 4 decimals, e.g. seed 0 gives
  0.7401 / 0.8133 / κ 0.1877.

### 4.2 Results per seed — **Verified**

CIN2+ AUC as segment / patient; κ is 4-grade linear-weighted.

| Seed | val→test p̄_b | val→test moments | train→test p̄_b | train→val p̄_b | Null seg AUC (mean ± SD, 95th) | p seg / p patient |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.740 / 0.813 / κ 0.188 | 0.742 / 0.840 / κ 0.182 | 0.668 / 0.773 / κ −0.005 | 0.642 / 0.650 / κ −0.012 | 0.498 ± 0.176, 0.741 | 0.074 / 0.291 |
| 1 | 0.740 / 0.840 / κ 0.184 | same | 0.666 / 0.773 / κ −0.006 | 0.639 / 0.650 / κ −0.010 | 0.498 ± 0.176, 0.741 | 0.076 / 0.204 |
| 2 | 0.742 / 0.840 / κ 0.182 | same | 0.667 / 0.760 / κ −0.005 | 0.640 / 0.650 / κ −0.014 | 0.500 ± 0.175, 0.743 | 0.072 / 0.227 |
| 3 | 0.740 / 0.827 / κ 0.190 | same | 0.659 / 0.760 / κ −0.008 | 0.631 / 0.650 / κ −0.005 | 0.497 ± 0.173, 0.741 | 0.069 / 0.263 |
| 4 | 0.739 / 0.813 / κ 0.177 | same | 0.671 / 0.787 / κ −0.005 | 0.644 / 0.650 / κ −0.013 | 0.500 ± 0.176, 0.740 | 0.074 / 0.300 |

Train→test on moments: 0.669 / 0.773 / κ −0.005 on every seed.

## 5. Interpretation

**What does not hold:**

- **"Grade κ 0.18."** A probe trained on 78 patients gives κ ≈ 0 on both test
  and val. The 0.18 came from a probe trained on val, whose class balance is
  set by very few patients. Test has only one Negative patient, so any 4-grade
  κ there rests on very few units.
- **"CIN2+ AUC 0.740 is real signal."** At that training size it cannot be told
  apart from chance: p ≈ 0.07 (segment) and 0.2–0.3 (patient). The old
  cluster bootstrap resampled test patients only. That gives uncertainty for
  the fitted probe, not for whether the probe reflects signal.
- **"QVR spectral beats all classical baselines on grade."** Six
  training-free moments tie it (0.742 / 0.840 / κ 0.182). The old baseline set
  had no moment or histogram summary. Significance held only against VAE and
  VAE-GAN, whose probes had collapsed to κ = 0.
- **"The fingerprint is a new, richer representation space."** By construction
  and on real data, p̄_b = A_W m̄. Seed stability of the probes (±0.001) follows
  from this, not from stable learned structure.

**What survives:**

- **A modest CIN2+ signal.** It replicates across two independent transfers:
  train→test 0.66–0.67 per segment (≈ 0.77 per patient) and train→val
  0.63–0.64. It lives in the low-order intensity distribution.
- **Detection.** Supervised moment/p̄_b probes reach ≈ 0.86; unsupervised
  reconstruction gives 0.74–0.82 on val depending on the D mode.
- **An interpretable basis.** One normal eigenstate, with anomalies leaking
  into one or two others. This is a readable coordinate system for the
  moments, not extra information.

## 6. What went well and what changes

**Went well:**

- The computations were correct.
- The val→test probe protocol had no leakage, and the old bootstrap language
  was honest ("numerically higher").
- Everything now reproduces to the fourth decimal.

**Changes, adopted into `ANALYSIS_PLAN.md` before the tag:**

1. Every probe result carries a unit-level permutation null, so uncertainty
   from the training side is included.
2. Probes are fit on the largest pool of units unseen by QVR, and transfers are
   reported in more than one direction.
3. The Fourier-moment probe is reported next to every fingerprint result, as a
   co-primary comparison rather than a footnote.

**For the parked cervical paper.** Withdraw the 4-grade κ and "beats classical
on grade" claims. The defensible statement is: a modest CIN2+ signal exists in
the low-order intensity distribution, and QVR's eigenbasis expresses it in an
interpretable form.

## 7. Decision

Keep cervical in the cross-dataset analysis as the **discovery dataset for
mechanism and detection**. Drop histological grade as an outcome.

## 8. Open

1. Fit-free patient-level associations (patient-mean moments vs CIN2+ and
   grade, 78 training patients). This would confirm the §3.2 negative-bias
   reading.
2. A permutation null for κ (only AUC was tested).
3. The unfiltered population (jagged segments), pending the voting-alignment
   check.

## 9. Files

| Output | Location |
| --- | --- |
| Fingerprint exploration | `results/explore/fingerprint/` (`fingerprint_explore.json`, `.csv`, `bin_curves.npz`, `bin_curves.png`) |
| Grade replication | `results/explore/replicate_grade/` (`replicate_grade.json`, `.csv`) |
| Scripts | `scripts/explore_fingerprint.py`, `scripts/replicate_grade.py` |
| Old sources | `qvr_cervix/results_final_circuit_fixed/q3_l3_fixed_ne10_s{0..4}_…/unfiltered_val/` |

Test labels were read for this re-examination; logged in `ANALYSIS_PLAN.md` §11.