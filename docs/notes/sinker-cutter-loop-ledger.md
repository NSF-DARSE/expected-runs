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

| 1 | Height-adjusted VAA: a sinker's approach-angle steepness at a fixed reference height is physics the linear ridge can't see (queued from the coach meeting). | New `coach_si_feature_gate.py` (SI-only clone of the gate, same statistic/panel/seed; "base" candidate replicates the official SI row exactly: 0.290). Added `vaa_flat` = kinematic approach angle at a fixed 2.0 ft plate height from RelSpeed/RelHeight/Extension/IVB — fixed reference, so no location leakage. | 0.290 → 0.295 | — (parked) | No movement. stuff_r +0.082→+0.083, semipartial +0.062→+0.065. As a smooth function of four features already in the ridge, it adds only nonlinearity, and that nonlinearity carries ~nothing at this sample. |

| 2 | Sinker identity is the break-vector's DIRECTION, which linear IVB+HB terms encode only as two magnitudes; also test slot-deviation as the computable stand-in for SSW. | Added to `coach_si_feature_gate.py`: `mov_angle` (atan2 of arm-side HB over IVB), `mov_mag` (total break), `slot_dev` (movement direction minus release-slot direction). | 0.290 → **0.460** (movgeo); slot_dev alone 0.265; movgeo+slotdev 0.460 | — (parked) | First real movement: stuff_r +0.082 → +0.116, semipartial +0.062 → +0.101. Movement geometry carries signal; slot_dev carries none (its info is already in the slot + movement terms). Keep movgeo, drop slot_dev. |

| 3 | Nonlinearity on top of movement geometry: "heavy sinker" velocity×break interactions and a break-direction sweet spot (quadratic angle). | Added `velo_x_ivb`, `velo_x_movang`, `mov_angle_sq` candidates. | 0.460 → 0.520 (movgeo+angsq; all3 0.495) | — (parked) | Marginal: stuff_r +0.116 → +0.121, semipartial +0.101 → +0.109. The direction-squared term is the only interaction that helps; velocity interactions add ~nothing. |
| 4 | Complete the nonlinearity family: full movement-plane quadratic (ivb², hb², ivb×hb) — value as a smooth non-monotone surface over the break vector. | Added `movquad` candidates. Also ran the tag-heterogeneity diagnostic: Sinker vs TwoSeamFastBall gaps are 0.02–0.17 pooled SD across velocity/IVB/HB/spin/slot, and within the 213 pitchers carrying both tags at 15+ the mean gaps are ±0.3 (|1.6|") — tag noise, pooling is correct. | 0.520 → 0.505 (movquad); movgeo+movquad **0.360** (overfits) | — (parked) | No movement; two flat iterations → stop. Nonlinear break terms are one shared effect already captured by the angle terms; stacking them subtracts. |

## Conclusion — 2026-08-20

**Sinker does not clear the gate with the frozen criterion, and the residual gap is
structural, not a missing feature.** The asymptote: base P(gain>0)=0.290, best candidate
(movement geometry + direction²) P=0.520, and every further physics term is flat or negative.

The arithmetic of the pre-registered statistic makes the required jump explicit. Prior
results predict the criterion at r=+0.241; the 50/50 z-blend beats prior alone only when the
grade's own validity approaches the prior's. Passing at P≥0.95 with the observed bootstrap
spread (gain SD ≈ 0.04, n=272) needs a mean blend gain ≈ +0.07, which needs stuff_r ≈ +0.24 —
roughly double the best achieved (+0.124), which itself is +50% over the shipping baseline
(+0.082). Task zero put the theoretical ceiling at ≈0.43, so a passing grade is not
impossible in principle — but every in-scope physics avenue is now spent:

- Height-adjusted VAA (fixed-reference kinematic proxy): nothing (iter 1).
- Movement-vector geometry (direction, magnitude): the one real gain (iter 2).
- Slot-deviation SSW stand-in: nothing; true SSW residuals not computable from this extract
  (TrackMan SpinAxis is movement-inferred; needs Hawkeye/optical axis).
- Velocity interactions, direction², movement-plane quadratic: marginal to negative (3–4).
- Tag split (Sinker vs TwoSeamFastBall): tags don't separate physically; pooling correct.
- Fastball differentials: excluded for SI by design (Jack, 2026-08-17) — a sinker IS the
  fastball; construct decision, not revisited.

The mechanism that eats the rest is the one `coach_sinker_why.py` flagged: 52.6% of sinker
balls in play are ground balls, and the criterion values every ball in play through a pooled
EV/LA map that by construction averages away what separates one ground ball from another.
The skill that makes sinker results repeat at r=0.24 lives largely in the part of the target
we deliberately smoothed. Fixing that means changing the criterion, which is frozen —
correctly, since a criterion tuned until the pitch passes is no gate at all.

**Recommendation:** ship nothing for SI now. Two honest paths, both requiring Jack:
(a) revisit the criterion's contact valuation (GB-conditioned map) as a *pre-registered*
criterion change evaluated on all six types before looking at SI's result; or (b) wait for
the 2026→2027 pair, where D1 coverage growth (~2x/yr) roughly doubles the panel and shrinks
the gain CI enough that a true stuff_r ≈ 0.15–0.18 might clear honestly.

FC remains parked per task zero: criterion reliability indistinguishable from 0 at this
sample; no feature work is validatable against it.

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
