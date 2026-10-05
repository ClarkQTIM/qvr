"""
One-off: add §12 (post-freeze exploratory extensions, 2026-10-05) and a §11 row to
docs/ANALYSIS_PLAN.md in place, keeping every existing §11 entry.

    python scripts/patch_plan_e6.py && git diff --stat docs/ANALYSIS_PLAN.md
"""
from pathlib import Path

p = Path('docs/ANALYSIS_PLAN.md')
s = p.read_text()
if '## 12. Post-freeze exploratory extensions' in s:
    raise SystemExit('already patched')

sec12 = r"""## 12. Post-freeze exploratory extensions (added 2026-10-05)

Added after the cervical, Baker, synthetic and MIT-BIH results. Everything here is
EXPLORATORY and is reported as such. Predictions are registered before any run.

### 12.1 Test A — readout of trained QVR models (`scripts/readout_test.py`)

Same trained models (fixed model, D mode `mc`, N_E ∈ {1, 10}), four scores fit on training
normals: QVR's native cost; per-timepoint z² of ⟨Z_q⟩_t; Mahalanobis of time-averaged
⟨Z_q⟩; Mahalanobis of p̄_b. Seed-rank-averaged AUC with record-level CIs.
**Prediction (MIT-BIH):** per-timepoint z² of ⟨Z_q⟩_t ≫ native, approaching the raw-series
per-timepoint z² (pooled ≈ 0.72, VEB ≈ 0.90); Mahalanobis of p̄_b ≈ trig moments (pooled ≈ 0.70).

### 12.2 E6 — sequential Hamiltonian-memory model (sQVR)

Motivation: in QVR every timepoint is a fresh circuit, so nothing propagates across time
(additive score; D only reweights by the time label). sQVR evolves ONE state through the
series so that the Hamiltonian carries the history.

- **State:** 3 qubits, ψ₀ = |000⟩.
- **Step t:** predictive distribution p(b | x_{<t}) = |⟨b| V ψ_{t−1}⟩|², b ∈ {0..7}, read without
  collapse (simulation; on hardware by repetition). Then update
  ψ_t = D(Δt) · W · U_enc(x_t) · ψ_{t−1}, with U_enc(x) = RY(x)^{⊗3} applied to the running
  state, W and V CNOT-chain ansätze (3 layers each), D = exp(−i Δt Σ_P μ_P P) deterministic
  (σ = 0; the stochastic ensemble did not help in any dataset), Δt = 2π / T.
- **Bins:** x_t is discretized into 8 bins, equiprobable per timepoint under training normals
  (per-timepoint quantiles), so a memoryless uniform predictor has NLL log 8.
- **Training:** mean negative log-likelihood of bin(x_t) over t on training normals; Adam,
  same budget, seeds and early stopping as §2.
- **Anomaly score:** per-series mean NLL.
- **Generation (deferred to a follow-on paper):** sample b, then x uniformly within the bin,
  feed back. Not evaluated here.

**Comparators (same 8-bin output, same data):**
1. per-timepoint histogram (independent bins, no memory: the additive baseline);
2. QVR with windowed embedding (x_t, x_{t−1}, x_{t−2} on the three qubits), a classical-style
   control for local shape without memory;
3. a small GRU with an 8-way softmax output and a comparable parameter count;
4. the classical detectors of `scripts/score_audit.py`.

**Registered predictions:**
- (a) Synthetic growing vs falling: sQVR's scalar score separates them (score AUC ≥ 0.9;
  QVR: 0.554), and its score is order-sensitive (shuf_r < 0.9).
- (b) MIT-BIH: sQVR NLL score beats the per-timepoint histogram on VEB and pooled (memory
  helps), and beats QVR's native score.
- (c) Relative to the windowed control and the GRU: no directional prediction.
- (d) sQVR does not beat the best classical detector by a margin whose CI excludes 0.

Generation is out of scope for this paper.
"""

anchor = '## 11. Deviations'
assert anchor in s, 'section 11 heading not found'
s = s.replace(anchor, sec12 + '\n' + anchor, 1)
row = ('| 2026-10-05 | §12 (new) | Added post-freeze exploratory extensions: Test A (readouts of trained QVR models) '
       'and E6 (sequential Hamiltonian-memory model with likelihood score, windowed-QVR / GRU / histogram comparators), '
       'with registered predictions. Generation deferred to a follow-on paper. | Mechanistic findings show QVR never '
       'propagates information across timepoints and its scoring rule discards information; §12 tests both fixes. |\n')
s = s.rstrip('\n') + '\n' + row
p.write_text(s)
print('patched: §12 inserted before §11; §11 row appended')