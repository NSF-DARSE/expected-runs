# Sinker / cutter gate loop — ledger

Goal: get SI and FC past `coach_incremental_gate.py` (`P(blend gain > 0) >= 0.95`, 50/50
z-blend vs prior results, 200 cluster-bootstrap refits) without touching the bar, the
criterion, the pool, or the bootstrap. Baseline from `coach_incremental_gate.json`:
SI p_gain_positive = 0.29 (n=272, stuff_r +0.082, results_r +0.241), FC = 0.775
(n=190, stuff_r +0.137, results_r +0.127).

Sign convention reminder: adjT and every grade here are expected run value, lower = better;
a predictor is correctly oriented when it correlates POSITIVELY with the future run-value
criterion (both are in the same lower-is-better frame).

| iter | hypothesis | change | SI p_gain | FC p_gain | verdict |
|------|-----------|--------|-----------|-----------|---------|
| 0 | Task zero: is the criterion reliable enough to be predictable at all? The 0.008-vs-0.24 discrepancy between `coach_sinker_why.py` and the gate had to be resolved before anything else. | New `coach_crit_reliability_panel.py`: 2x2 of year pair (2024→2025 vs 2025→2026) x panel (25-min "why" panel vs 15-min + 10%-share gate panel), plus reliability-vs-floor curves, 2000-rep pitcher bootstrap CIs. Output: `coach_crit_reliability_panel.json` in the score workdir. | 0.29 (unchanged — no model change) | 0.775 (unchanged) | Both published numbers are right on their own definitions; the `coach_sinker_why.py` docstring conflated them. SI crit reliability is year-pair dependent: 2024→2025 ≈ 0 at low floors (0.013, the "0.008") but +0.25 at floor≥40, and 2025→2026 is +0.195 [+0.08,+0.29] (25-min) / +0.241 [+0.13,+0.35] (gate panel). The gate's operative pair is 2025→2026, where the criterion is real. **SI ceiling ≈ sqrt(0.77 × 0.24) ≈ 0.43 vs current stuff_r 0.082 → large headroom, worth pursuing.** FC crit reliability is indistinguishable from 0 at every floor on both pairs (2025→2026 gate panel +0.127 [−0.04,+0.27]; within-build controls +0.09/+0.05/+0.01 at floors 15/25/40). **FC verdict: criterion-unreliable at this sample (96–244 pairs); no feature work can be validated against it. Park until ~2027 data doubles the panel.** Cross-build join artifact ruled out: identical r within the crit build alone. |

## Iteration notes

### Iter 0 (task zero) — 2026-08-20

- Panel definitions stated: "why" panel = pitcher-seasons with ≥25 pitches of the type in
  both years, no usage condition, within one build. "Gate" panel = ≥15 pitches AND ≥10% of
  the pitcher's total pitches that season, both sides, score-build eval year joined to
  crit-build eval year (real 2025 → real 2026).
- The gate's `results_r` column IS the criterion's 2025→2026 reliability on the gate panel.
  `coach_sinker_why.py`'s `crit_yoy` is the 2024→2025 reliability on the 25-min panel.
  Different year pairs, different floors, both correct; the sinker_why docstring's
  "results predicts it at +0.24 while physics predicts +0.08, so it can't be noise"
  argument mixed the 2025→2026 figure with 2024→2025 panel conclusions.
- Why 2024→2025 looks dead for SI at low floors but not high: small-sample attenuation
  (15–25 pitch season means of adjT are mostly noise) plus thinner 2024 D1 coverage
  (n pairs 181 vs 296). At floor≥40 the 2024→2025 SI reliability is +0.249 [+0.07,+0.42].
- Implied ceilings (sqrt(rel_grade × rel_crit), grade YoY from coach_sinker_why.json):
  SI sqrt(0.770 × 0.241) ≈ 0.43. FC sqrt(0.835 × 0.127) ≈ 0.33 at the point estimate but
  the reliability CI straddles 0, so the ceiling is not established at all.
- Literature scan (parallel agent, primary sources): seam-shifted-wake residuals are NOT
  computable from our TrackMan fields — TrackMan's SpinAxis is itself inferred from
  movement, so observed-minus-spin-predicted movement is 0 by construction; needs Hawkeye
  or optical (Rapsodo) axis. Height-adjusted VAA IS computable (VertApprAngle +
  PlateLocHeight + handedness; FanGraphs "VAA above average" normalizes VAA at fixed plate
  height). Sinker grade unreliability is a known public result (basetunnel/SMOKE: sinkers
  mix a whiff job and a contact-management job under one tag; tjStuff+ documented
  sinker/four-seam feature cross-contamination as a failure mode). BP StuffPro is the one
  public model with a launch-angle-conditioned contact target (KNN on EV/LA/spray) — that
  is a criterion change, frozen here, noted only for the record.
