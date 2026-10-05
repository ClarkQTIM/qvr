"""
One-off: add §12.3 (pre-run amendments to E6) and a §11 row to docs/ANALYSIS_PLAN.md in place.

    python scripts/patch_plan_e6b.py && git diff --stat docs/ANALYSIS_PLAN.md
"""
from pathlib import Path

p = Path('docs/ANALYSIS_PLAN.md')
s = p.read_text()
if '### 12.3 Pre-run amendments' in s:
    raise SystemExit('already patched')

sec = r"""### 12.3 Pre-run amendments to E6 (2026-10-05, before any sQVR run on real or test data)

Found while building and self-testing `qvr/sequential.py` (self-test only, synthetic ramps):

1. **Bins.** Equiprobable bins cannot express rarity: an extreme value lands in a bin that
   holds 12.5% of normals, so the memoryless baseline is constant. Replaced by tail bins at
   the training normals' per-timepoint (1, 5, 25, 50, 75, 95, 99)% quantiles (masses
   1, 4, 20, 25, 25, 20, 4, 1%). The memoryless per-timepoint histogram then scores rarity
   (NLL = −log mass) and is a meaningful no-memory baseline.
2. **Prediction (a) is not a memory test.** Growing vs falling ramps differ in their
   per-timepoint values, so any per-timepoint model separates them (memoryless histogram,
   per-timepoint z²). (a) is kept and reported, but memory is tested on a new dataset:
3. **Memory-test dataset** (`scripts/prepare_dataset.py memory`): normals AR(1), φ = 0.95;
   anomalies with the identical N(0, 1) marginal at every timepoint: iid, AR(1) φ = −0.95
   ("anti"), and time-permuted normal series ("shuffled"); T = 32, fixed scale x·π/3.
4. **Added comparators** (classical memory baselines): lag-1 autocorrelation and mean squared
   first difference, each scored two-sided against training normals.
5. **Reporting:** sQVR's NLL on test normals is always reported next to the memoryless
   histogram's NLL (predictive quality), not only AUC.

**Registered predictions on the memory-test dataset:**
- (e) QVR native score, memoryless histogram and per-timepoint z²: AUC within 0.5 ± 0.1 for
  every anomaly class (blind by construction).
- (f) sQVR NLL score: pooled AUC ≥ 0.8.
- (g) Classical lag statistics: pooled AUC ≥ 0.95 (they target exactly this structure).
  sQVR is not predicted to match them.
"""

anchor = '## 11. Deviations'
assert anchor in s
s = s.replace(anchor, sec + '\n' + anchor, 1)
row = ('| 2026-10-05 | §12.3 (new) | Pre-run amendments to E6: tail bins; growing-vs-falling recognised as not a memory '
       'test; memory-test dataset (same marginals, different dependence); classical lag baselines; NLL reported vs '
       'memoryless histogram; predictions (e)–(g). | Found while self-testing qvr/sequential.py; no sQVR run on any '
       'evaluation data yet. |\n')
s = s.rstrip('\n') + '\n' + row
p.write_text(s)
print('patched: §12.3 inserted before §11; §11 row appended')