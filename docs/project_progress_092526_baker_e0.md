# Project progress — 2026-09-25 — E0 audit of Baker et al. (crypto QVR benchmark)

Original claim audit per `ANALYSIS_PLAN.md` §5 E0. Paper: Baker et al., *Quantum
Variational Rewinding for Time Series Anomaly Detection*, arXiv:2210.16438
(AAAI Symposium Series). Data: authors' released arrays (Zenodo 7258627).
Code: `QVR_example.ipynb`.

**Bottom line.**

- On the released data, a one-number summary of the cumulative-volume channel reaches the
  paper's reported test performance on its strongest set (BA 0.808 vs published ≈ 0.8).
- The notebook's result reproduces. The paper's headline validation balanced
  accuracy (0.82) does not: across 5 configurations and 165 training runs, the
  best is 0.717.
- Under a clean protocol, the mean of a single channel (cumulative volume),
  with no training, matches QVR's best restart within overlapping CIs and beats
  its typical restart.
- Time order contributes little.
- QVR's stochastic ensemble either collapses to zero noise (paper objective) or
  dephases and lowers detection (notebook objective), as on cervical.

Status labels: **Verified** (run on the server), **Observed** (consistent, not
formally tested).

---

## 1. Code-level findings from the notebook — **Verified**

| Item | Paper text | Notebook code (what was released) |
| --- | --- | --- |
| Embedding | R_y per channel | `AngleEmbedding`, default **RX** |
| W | Schuld et al. ansatz | `StronglyEntanglingLayers`, 3 layers |
| Cost | Eq. 4: E_ε[Ω²] (square **inside**) | (η₀ − mean_ε⟨O⟩)² / 4 (square **outside**) |
| Draws | one ε per series, shared over time (SM pseudocode) | fresh ε per (timepoint, draw) |
| τ | 5 | 15 |
| Training | 50 runs × 2,000 mini-batch iterations, best by val BA | one Powell run, maxfev = 500 |
| Score | a_X (Eq. 9) | raw time-series cost (a_X computed, then discarded) |
| Time grid | — | train `linspace(0.1, 2π, 180)`; score every 4th point with a fresh `linspace(0.1, 2π, 45)` |
| Threshold | tuned on V and N | tuned on `Xte_norm` vs `Xval`; the **same** `Xte_norm` is used for testing |
| Precision | — | expectations stored in float32 (`torch.zeros(N_E)`) |

**Equivalence.** `tests/test_baker_notebook_equivalence.py` runs the notebook
functions verbatim against the new backend:

| Quantity | Max \|Δ\| |
| --- | --- |
| Per-point expectation | 4.4e-16 |
| Single-point cost | 3.5e-10 (notebook float32 limit) |
| Time-series cost | 3.2e-10 (notebook float32 limit) |
| Arctan penalty | 0 |

Only the bivariate arrays were released: `Xtr` 100, `Xval` 60, `Xte_norm` 60,
four dirty sets of 60, `clean_btc` 34, `clean_usdt` 19. The trivariate model
cannot be audited.

## 2. Reproduction — **Verified**

`scripts/baker_e0.py`. Every variant uses Powell on 10 series × 10 timepoints
per evaluation, N_E = 10, 2 qubits, k = 2.

| Variant | What it tests | Runs | Val BA best | Val BA median |
| --- | --- | --- | --- | --- |
| `notebook` | Notebook as released (τ = 15, outside, per-point draws, 500 evals) | 5 seeds | 0.633 | — |
| `paper` | Paper maths (τ = 5, inside, shared draws, 2,000 evals), RX | 50 | 0.692 | — |
| `paper_ry` | Paper maths with RY embedding (paper text) | 50 | 0.700 | 0.642 |
| `paper_iter` | Paper maths, Powell capped by 2,000 iterations (converged at ≈ 2,760 evals) | 10 | 0.692 | 0.629 |
| `nb_tau5` | Notebook maths with paper settings (τ = 5, 2,000 evals) | 50 | 0.717 | 0.596 |
| **Published** | | | **notebook 0.675; paper 0.82** | |

- The notebook reproduces within run-to-run noise. Ũ₊ test BA / F1 is
  0.700 / 0.640 against the published 0.717 / 0.761.
- **The paper's 0.82 is not reproduced** under any reading tried. The maximum
  over all 165 runs is 0.717. Since 0.82 is itself a best-of-50 on the same
  validation set it is reported on, part of the gap may be selection luck. It
  remains a discrepancy between the paper and its released code and data.
- The three diagnostics (`paper_ry`, `paper_iter`, `nb_tau5`) were prespecified
  and logged (§11, 2026-09-25) before running.

### 2.1 Discrepancies between the paper's two versions and the released data

| Item | arXiv v2 (2022) | AAAI Symposium (2025) | Released arrays |
| --- | --- | --- | --- |
| Validation set V | same conditions as U+, U−, B+, B− (single transaction) | same conditions as Ũ+, Ũ−, B̃+, B̃− (multiple transactions) | one `Xval`, 60 series, recipe not stated |
| Data volume | U and B: 1,900 series each | 5,432 anomalous and 842 normal series | `Xtr` 100, `Xte_norm` 60, `Xval` 60, test sets 19–60 |
| Embedding | R_y in main text; SM figure notes R_x | R_y | notebook: RX |
| η₀ | bounded to [−1, 1] | bounded to [−1, 1] | unbounded in notebook |

Both versions report the same validation balanced accuracy, 0.82 (bivariate) and
0.77 (trivariate). The released normals (160) are about 19% of the 842 the AAAI
version describes. The training-set size behind 0.82 is not stated.

### 2.2 Re-scoring diagnostic (`scripts/baker_rescore.py`) — **Verified**

All 165 saved models were re-scored with no retraining, under:
- three definitions of V:
  - `released`: the released `Xval`;
  - `aaai_multi`: 60 series, balanced from the four multi-transaction test sets;
  - `arxiv_single`: 38 series, balanced from the two single-transaction test sets;
- two scoring resolutions: the notebook's stride-4 grid, and all 180 points on
  the training time grid.

The alternative pools are drawn from test sets, so they are optimistic and
interpretive only.

Best validation BA over all runs and variants (published: 0.82):

| Resolution | released | aaai_multi | arxiv_single |
| --- | --- | --- | --- |
| stride 4 | 0.717 | 0.733 | 0.639 |
| full 180 | 0.708 | 0.733 | 0.643 |

- **Neither the definition of V nor the scoring resolution closes the gap.**
  Even an optimistic V taken from the easiest test conditions tops out at 0.733.
- The released `Xval` behaves closer to the multi-transaction pool than the
  single-transaction one.
- Full-resolution scoring helps only the notebook-objective models, whose noise
  averages out over more points: median Ũ₊ AUC rises 0.714 → 0.765 (`notebook`)
  and 0.662 → 0.706 (`nb_tau5`). Paper-objective models (σ ≈ 0) are unchanged.
- **Remaining explanations, not testable with the released materials:**
  1. the larger unreleased data described in the AAAI version;
  2. selection luck, since 0.82 is a best-of-50 reported on its own selection set.

  We report 0.82 as not reproducible from the released code and data under 5
  model configurations × 3 definitions of V × 2 scoring resolutions (165 trained
  models).

### 2.3 Remaining paper ambiguities — **Verified**

- **η₀ bound (paper: −1 ≤ η₀ ≤ 1).** No run in any variant exceeded it, in 0 of 165
  runs, so the bound would never have been active. Irrelevant.
- **W ansatz.** The paper cites Schuld et al. (2020), whose circuit uses trainable
  controlled single-qubit gates. The notebook uses PennyLane `StronglyEntanglingLayers`,
  which uses fixed CNOTs. The model sees W only through its eigenprojectors
  W†|b⟩⟨b|W, a 12-dimensional space for 2 qubits.

  | SEL layers | Parameters | Jacobian rank of α → eigenprojectors (of 12) |
  | --- | --- | --- |
  | 1 | 6 | 4 |
  | 2 | 12 | 8 |
  | 3 | 18 | 12 |
  | 4 | 24 | 12 |

  With 3 layers, SEL is locally full rank, so no other ansatz can enlarge the
  bivariate model's function class; only the optimization landscape could differ.
  (Local rank at random points, not a proof of global surjectivity.) The trivariate
  model (3 qubits) could differ, but its data were not released.
- **Time grid (unstated; one hint of hours).** A rescaling of t is absorbed by μ and σ,
  except through the penalty and initialization ranges. Not run.

## 3. QVR vs prespecified baselines — **Verified**

Baselines are fit on `Xtr` only, with no labels and no selection:
- `vol_mean`: the median-centred mean of channel 1 over MAD;
- `chan_mean_sd`: Mahalanobis on per-channel mean and SD;
- `trig_moments`: Mahalanobis on per-channel mean cos/sin(kx), k = 1, 2.

Clean protocol, Ũ₊ (`dirty_usdt_pos`, the strongest set):

| Method | AUC [95% CI] | Clean BA | Median over runs (AUC) |
| --- | --- | --- | --- |
| `vol_mean` (one number) | 0.832 [0.751, 0.902] | 0.738 | — |
| `trig_moments` | 0.800 [0.712, 0.874] | 0.738 | — |
| QVR `notebook`, best of 5 | 0.792 [0.714, 0.866] | 0.692 | 0.636 ± 0.166 (mean ± SD) |
| QVR `paper`, best of 50 | 0.853 [0.784, 0.916] | 0.717 | 0.804 |
| QVR `paper_ry`, best of 50 | 0.873 [0.810, 0.927] | 0.775 | 0.805 |
| QVR `paper_iter`, best of 10 | 0.868 [0.804, 0.924] | 0.733 | 0.829 |
| QVR `nb_tau5`, best of 50 | 0.855 [0.790, 0.917] | 0.783 | 0.662 |

Across all six anomaly sets the QVR and `vol_mean` CIs overlap. `vol_mean` or
`trig_moments` is ahead on BTC-neg for several variants. The two single-transaction
("clean") sets sit near chance for every method, as the paper itself notes.

The QVR rows are best restarts selected on validation, taken from five
configurations. The baselines had no selection. Median restarts sit at or below
the one-number baseline.

**Against the published test result.** Under the paper's own protocol, the one-number
baseline `vol_mean` scores balanced accuracy 0.808 / F1 0.813 on Ũ₊. The paper reports
A_B close to 0.8 for Ũ₊ with both models. Our QVR reproductions reach 0.700–0.775 under
the same protocol.

**Prespecified reading (interpretation table, row 1):** a simple baseline is ≥ QVR
with overlapping CIs → **this benchmark does not require QVR.** Row 4 also applies:
the paper-protocol headline was not reproduced.

## 4. Mechanism — **Verified**

| | Paper objective (`paper`, `paper_ry`, `paper_iter`) | Notebook objective (`notebook`, `nb_tau5`) |
| --- | --- | --- |
| Learned \|σ\| | ≈ 1e-4 on all terms (collapsed) | bimodal: 0.003, 4.3, 5.1 rad (`notebook`) |
| AUC `mc` vs `mean` | identical | `mean` higher on every set (Ũ₊: 0.792 → 0.837; `nb_tau5`: 0.855 → 0.887) |
| Time permutation, Ũ₊ AUC | 0.873 → 0.845, 0.868 → 0.846, 0.853 → 0.849 | 0.792 → 0.777, 0.855 → 0.804 |
| D = 0 (`identity`) | AUC well below 0.5 (ranking inverts; information is in the values) | same |
| Per-timepoint term | exact degree-≤ 2 trig polynomial per channel (residual ≤ 5e-16) | same |

1. **Squaring inside the draw average penalizes draw variance, so training
   drives σ to 0** and the random-Hamiltonian ensemble switches off.
2. **Squaring outside leaves σ free.** It settles "off or fully dephased", the
   same split seen in the cervical seeds, and the surviving noise lowers
   detection. This is observed on two datasets, two circuits and two optimizers.
3. **Order contributes little.** The score is additive over timepoints by
   construction, and each term is a low-degree trig polynomial of that
   timepoint's values.

## 5. Protocol notes

- The paper protocol reuses the test normals (`Xte_norm`) for threshold tuning.
  The clean protocol tunes on half the normals plus `Xval` and tests on the
  other half (2-fold), alongside threshold-free AUC with 2,000 bootstrap
  resamples.
- 0.82 is a best-of-50 on validation BA, reported on that same validation set.

## 6. Conclusion for E0 (Baker)

The released code reproduces. The paper's headline number does not. The benchmark
is solved about as well by the mean of one preprocessed channel. QVR's temporal
and stochastic machinery contributes little or negatively. This is consistent
with the structural analysis (additive per-timepoint trig polynomial scores) and
with the cervical findings. The authors framed their work as a proof of concept
and did not claim an advantage. What was missing was a same-data simple baseline
and a check of what the circuit computes.

## 7. Files

| Output | Location |
| --- | --- |
| Main run (`notebook`, `paper`) | `results/e0/baker/baker_e0.json` |
| Diagnostics | `results/e0/baker_{paper_ry,paper_iter,nb_tau5}/baker_e0.json`, `.log` |
| Scripts / tests | `scripts/baker_e0.py`, `qvr/datasets/baker.py`, `tests/test_baker_notebook_equivalence.py` |

---

## 8. Follow-up: full data, preprocessing, and what QVR actually uses — **Verified**

### 8.1 Data forensics (`scripts/baker_forensics.py`)

- **Full data exist in the authors' repo** (`data/large_data_sets`): 842 normal and
  5,485 anomalous windows (1,912 dirty BTC, 3,520 dirty USDT, 34 clean BTC, 19 clean
  USDT), three features (open, vol, tbba), 181 timepoints. The trivariate model is
  therefore auditable.
- **Provenance:** all 513 released rows match rows of the large sets exactly (last
  timepoint dropped; channels = open, vol). `Xval` = 30 dirty BTC + 30 dirty USDT, i.e.
  the AAAI definition of V. `Xtr` uses 100 of 842 normals and `Xte_norm` another 60,
  leaving 682 normals unused. Hygiene: 3 series are shared between `Xval` and test sets;
  2 test series also appear under a different condition's pool.
- **Scaling is global, not per-timepoint.** One min and one max per feature over all
  timepoints and all series pooled, anomalies included: the pooled per-timepoint max of
  cumulative volume climbs from −3.06 (t = 0) to +3.14 (t = 180), and ±π are reached only
  at 1–4% of timepoints. This contradicts SM Eq. S2. Normal volume occupies
  [−3.14, −1.46]; the anomalies set the scale.
- **Aliasing near ±π: not present** (0.0% of values near +π; ≤ 0.4% of windows). The
  earlier hypothesis is refuted.
- **Detection tracks volume and transaction size for both methods.** AUC by anomaly
  volume tertile: QVR 0.527 / 0.837 / 0.990, vol_mean 0.385 / 0.843 / 0.991. AUC by
  |transaction| tertile: QVR 0.682 / 0.762 / 0.837, vol_mean 0.641 / 0.696 / 0.789
  (the paper's Fig. 3 claim holds, for both).

### 8.2 Reproduction on the full data (`scripts/baker_build_sets.py`)

Paper maths, 50 restarts × 2,000 evaluations, released `Xval` / `Xte_norm` / test sets.

| Variant | Best val BA | Median val BA | Ũ₊ AUC best / median restart |
| --- | --- | --- | --- |
| bivariate, 100 normals (control; rebuild = release) | 0.683 | 0.621 | 0.867 / 0.756 |
| bivariate, 782 normals | 0.692 | 0.621 | 0.868 / 0.779 |
| trivariate, 100 normals | 0.683 | 0.592 | 0.743 / 0.709 |
| trivariate, 782 normals | 0.675 | 0.588 | 0.864 / 0.652 |
| bivariate, 782, normals-only global scaling (exploratory) | 0.692 | 0.658 | 0.862 / 0.849 |
| published | 0.82 (bi) / 0.77 (tri) | | |

Neither more training data nor the third channel approaches the published numbers.
Over all configurations (~400 trained models), the maximum validation BA is 0.717.

Reproducibility note: the control gave 0.683 vs 0.692 in the first run with identical
data, because the thread count differed (4 vs 8). Floating-point summation order
changes with threads and Powell amplifies it; runs are bit-identical only at a fixed
thread count, which is now recorded.

### 8.3 What QVR uses (`scripts/score_audit.py`, normals-only scaling, best restart)

Comparators are fit on training normals; post hoc comparators are labelled as such
(ANALYSIS_PLAN §11).

| Set | QVR | vol_mean | trig_moments | final value* | late mean* | per-t z² (vol)* | per-t z² (all)* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pooled | **0.758** | 0.677 | 0.664 | 0.667 | 0.665 | 0.564 | 0.645 |
| Ũ₊ | **0.862** | 0.811 | 0.778 | 0.805 | 0.804 | 0.698 | 0.804 |

\* post hoc. QVR is significantly higher (paired bootstrap 95% CI excludes 0) than every
comparator on every set.

- **Not time weighting:** final value and late-window mean match the window mean; the
  per-timepoint z² model (same additive structure as QVR) is worse.
- **Not price:** knocking out price leaves QVR at 0.754 (intact 0.758); knocking out
  volume drops it to 0.628.
- **Not complementary to volume:** the rank combination of QVR with vol_mean is slightly
  below QVR alone (−0.008 pooled, CI [−0.013, −0.004]).
- **Not the dynamics:** with D = 0 (`identity`), AUC equals the full model's
  (Ũ₊ 0.871 vs 0.866; all sets within ±0.02). Time permutation leaves AUC unchanged
  (shuf_r 0.977–0.992).

With D = 0, W cancels (W†W = I) and the RX embedding gives ⟨Z_q⟩ = cos x_q, so the
score is exactly

    s(x) = mean_t ( η₀ − ½ [cos x_price(t) + cos x_vol(t)] )² / 4

a closed-form classical detector with a single parameter, η₀. **QVR's advantage over
the simple baselines under normals-only scaling is reproduced exactly by this
untrained cosine transform of the values.** The trained unitary and the
random-Hamiltonian dynamics contribute nothing measurable.

## 9. Final conclusion (Baker)

1. The published validation accuracy (0.82 / 0.77) is not reproducible from the released
   code and full data.
2. Under the published preprocessing, the mean of one channel matches QVR and reaches the
   published test accuracy (prespecified result).
3. The published preprocessing (global min-max over pooled data, anomalies included)
   handicaps QVR, and differs from the paper's description.
4. With normals-only scaling, QVR significantly outperforms all simple comparators
   (exploratory), and that advantage is fully explained by the cosine embedding readout:
   a one-parameter classical function. Neither the trained circuit nor the dynamics nor
   time order contributes.