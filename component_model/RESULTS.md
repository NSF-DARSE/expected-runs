# Results: the Stuff+ / Location+ / Pitching+ stack (four-seam, 2024 -> 2025)

Everything below is reproducible from `analysis/` (see its README for setup and
run order). Scope: four-seam fastballs, pitchers with 100+ qualifying FF in both
2024 and 2025 (n = 699), trained on 2024, evaluated out-of-sample on 2025.
One season-pair: every conclusion is strong-but-provisional pending 2026.

SE on a single correlation at n = 699 is ~0.037. Differences between predictors
are judged by paired bootstrap over pitchers (4000 reps, seed 42); under ~1 SE
of the bootstrapped difference is treated as a tie.

## The fair criterion (script 01)

Raw run value allowed is mostly noise (year-over-year r = 0.185). The evaluation
target for everything below is C2: each ball in play replaced by its exit-speed /
launch-angle expected run value (defense and batted-ball luck stripped), then
opponent-adjusted with batter effects shrunk toward league means (K = 200).

Anchor table. Script 01 must reproduce these before any other script is trusted:

| criterion | reliability | own 24 -> C_25 | ridge 24 -> C_25 | 50/50 blend |
|---|---|---|---|---|
| C0 raw | 0.185 | 0.185 | 0.175 | 0.250 |
| C1 xT | 0.303 | 0.303 | 0.171 | 0.321 |
| C2 adjusted | 0.304 | 0.304 | 0.235 | 0.355 |

Stuff+ Ridge reliability 0.908. FF rows with complete features 1,567,979;
panel pitchers 699.

## Location is a real, separate skill (scripts 02-03)

Location traits pass the reliability gate: scatter (sd_x 0.57, sd_z 0.59) and
mean height (0.53) repeat at nearly stuff level; the pooled location run-value
descriptor repeats at 0.485, far above run-value noise (0.18) and above whiff
(0.35). Zone rate (0.37) is mediocre; shadow rate (0.08) is noise.

Location+ (2024-trained 0.25 ft binned (x,z) -> xT map, 0.5 ft and overall-mean
fallbacks) has reliability 0.479 and alone predicts C2_25 at 0.290, on par with
the full last-year stat line (0.304) and ahead of Stuff+ (0.235). It is nearly
orthogonal to Stuff+ across pitchers (r = 0.158): location and stuff are
different skills.

Ablation vs C2_25 (equal-weight z blends): results alone 0.304, + Location+
0.352, + Stuff+ 0.392. Both increments survive the bootstrap (Location+ over
results +0.048, CI [+0.008, +0.089]; Stuff+ over both +0.040, CI [+0.007,
+0.075]). Equal weights beat split-half OLS-fitted weights (0.392 vs 0.380);
use equal weights.

## Count-conditioning: right idea, and a trap (script 04)

Motivation: a pitch well out of the zone on 0-2 should not be penalized like the
same pitch on 3-0. A count-conditioned map (12 discrete counts, each count-cell
shrunk toward the pooled cell value, prior m = 5 tuned on a 2024-internal
holdout) fixes exactly that: high-chase xT flips from -0.05 on 0-2 to +0.11 on
3-0, and a grooved heart pitch on 3-0 is the best cell on the board.

Scored raw (E[xT | loc, count]), it beats the pooled map standalone: reliability
0.508 vs 0.479, validity 0.313 vs 0.290, both ~2 SE. But decomposing the score
shows the entire gain is count occupancy - credit for living in favorable counts
- not better location measurement:

| score | reliability | validity vs C2_25 |
|---|---|---|
| pooled | 0.479 | 0.290 |
| count, raw | 0.508 | 0.313 |
| count, location-given-count only | 0.474 | 0.300 |
| count occupancy only | 0.397 | 0.199 |

The location-given-count variant ties pooled (both diffs under 1 SE). Occupancy
is a real, repeatable skill (0.397), but it is results-shaped: it correlates
0.385 with the criterion and only 0.108 with Stuff+, and in the three-way blend
the raw variant's advantage collapses to +0.005 (tie) because last-year results
already carry count mix.

Decision (per FRAMEWORK.md "Score design principles"): pitcher-level Location+
stays pooled / count-relative; the count-conditioned map is used for pitch-level
explanation, where it scores individual pitches fairly; count occupancy gets its
own label if surfaced. The null on location-given-count is four-seam only and
should not be generalized to breaking/offspeed, where the ideal target moves
much more with count.

## Scaling: the MLB convention inverts at this level (script 05)

MLB reporting has Stuff+ separating pitchers ~3.6x more than Location+. On our
data the ratio is 0.80 - location separates college pitchers MORE than stuff
(pitcher-level SD in runs/100 pitches: stuff 0.72, location 0.90-1.06, results
1.9). Signal SDs (SD x sqrt(reliability)) are nearly equal: 0.69 vs 0.63-0.75.

Consequences: showing both scores at 100 +/- 15 is roughly honest here (a
strictly honest Location+ scale would be ~100 +/- 19), and the equal-weight
1/3 z blend remains the best Pitching+ combine (0.392) against the natural-unit
run sum (0.380, P = 0.88 it is worse) and reliability-weighted z (0.364,
P = 0.98). Keep 100 +/- 15 for both and equal weights.

## Sample floor for a Location+ read (script 06)

Split-half reliability with Spearman-Brown correction gives n0 ~ 52 FF
(pitches at which reliability reaches 0.5), with n0 drifting up in larger
samples, so the curve is optimistic at high n. Measured full-sample reliability:
~0.50 at 30-60 FF, ~0.60 at 60-100, ~0.65-0.67 at 100-150. This is far below
the ~400-pitch MLB guidance: coarser cells plus much larger between-pitcher
location variance in college make the score stabilize faster.

Recommendation: full read at 100+ FF, caution tag at 50-99, small-sample flag
below ~50 (mirroring the whiff column's flag). Per-team flags: run script 06
with `--team`.

## The models out-predict the stat line (script 07)

Location+ + Stuff+ combined, with no access to a pitcher's own results, predict
his next-season fair criterion better than his actual results do:

| predictor (2024 -> C2_25) | Pearson | Spearman |
|---|---|---|
| raw results (luck included) | 0.197 | 0.163 |
| adjusted results (luck-stripped) | 0.304 | 0.251 |
| Location+ + Stuff+ (no results) | 0.345 | 0.352 |
| all three | 0.392 | 0.375 |

Against raw results the margin is decisive (+0.150, CI [+0.059, +0.240],
P = 1.000). Against the luck-stripped baseline it is a point-estimate win
(+0.042, ~1.1 SE) with a clearly better ordering (Spearman 0.35 vs 0.25).
The three-part blend still beats models-only, so last year's results retain
real incremental information.

Note the models are trained on league-wide outcomes; what they never see is the
scored pitcher's own results.

## Secondary pitch types: the pattern reverses (script 09)

Running the identical protocol per pitch type (per-type Ridge, per-type maps,
per-type criterion; panel = 100+ of the type in both years):

| pitch (n) | criterion rel | Stuff+ validity | Location+ rel | Location+ validity |
|---|---|---|---|---|
| four-seam (699) | 0.304 | 0.235 | 0.479 | 0.290 |
| slider (216) | 0.174 | 0.126 | 0.434 | 0.020 |
| changeup (103) | 0.181 | 0.219 | 0.501 | 0.013 |
| curveball (33) | -0.09 | 0.347 | 0.393 | 0.109 |

Two reversals of the four-seam picture. First, secondary-pitch Location+ is
reliable but NOT valid: where a pitcher throws his slider repeats year to year
(0.43), but it predicts next-year slider outcomes at ~0. Adding it to the blend
makes prediction worse. This is not a handedness artifact: batter-mirrored and
platoon-conditioned maps score the same (slider 0.020 -> 0.032; changeup stays
at ~0), while the four-seam map is unaffected by the same variants. Second,
Stuff+ gains importance on secondaries - movement quality predicts breaking-ball
outcomes better than season-average location does.

The count-conditioning hypothesis (that location-given-count would matter more
for breaking/offspeed) is moot on this data: no count variant rescues a location
score that has no validity to begin with (all diffs within ~1 SE; the curveball
panel is too small to read at all).

Working interpretation, pending 2026 data: secondary-pitch value lives in the
pitch's physical quality and its usage/sequencing context, not in where it sits
on average over a season. Production implication: show Location+ for fastballs;
do not surface a season-level Location+ for secondary pitches, and score
secondary pitch quality from Stuff+ (which is MORE predictive there).

## Secondary pitches: a deployment component (script 10, EXPLORATORY)

If season-average location does not predict secondary-pitch outcomes, what
does? A reliability-gated screen of usage and fastball-relative location traits
found four whose directions replicate across the slider and changeup panels.
Read against the run-value criterion (lower = better), a secondary pitch does
BETTER next year when it is used more overall, used more at two strikes, thrown
with MORE horizontal separation off the pitcher's fastball line, and not buried
far below the fastball band. The equal-weight composite of the four:

| pitch (n) | composite rel | composite validity | adds over results+Stuff+ |
|---|---|---|---|
| slider (216) | 0.603 | 0.300 | +0.121, CI [+0.053, +0.186], P = 1.000 |
| changeup (103) | 0.548 | 0.256 | +0.074, CI [-0.019, +0.165], P = 0.94 |
| curveball (33) | 0.507 | -0.04 | panel too small to read |

On sliders this is the largest single increment measured anywhere in this
project, it is essentially orthogonal to Stuff+ (r = 0.04), and it is more
reliable than fastball Location+.

Interpretation (settled by the causality probes below): this is mostly
REVEALED TRUST, not a deception skill. Pitchers and coaches allocate usage
based on quality they observe and our features do not, so usage is a proxy for
unmeasured pitch quality. Consistent with that reading: (1) the usage trait's
effect mostly disappears when the pitcher's own-fastball quality and own-pitch
stuff are partialled out (0.16 -> 0.06); (2) within-pitcher first differences
are null - changing deployment from 2024 to 2025 did not change outcomes
(r = -0.04) - which is what a stable proxy produces and a causal lever does
not; (3) the composite predicts through next-year whiff rate, the channel a
quality proxy should flow through. The one exception is horizontal separation
off the fastball line (sep_x): pitchers who increased it saw outcomes improve
in both panels (r = -0.25 and -0.22 with improvement), the only trait with a
causal-looking within-pitcher signal.

Practical consequence: the composite is a legitimate FORECASTING component and
must not be read as coaching advice. "Throw it more and it will get better" is
exactly backwards - the causality runs from quality to usage. Only sep_x has
any prescriptive standing, and one first-difference result is not enough to
coach from.

The revealed-trust reading has one productive internal use: usage the measured
quality CANNOT explain flags where staff knowledge and the model disagree.
Script 11 computes it - an arsenal-relative share model (a pitch competes with
the pitcher's own other pitches, never the league; fastball identity is modeled
as primary/secondary slots so sinker-first pitchers are not fake flags) fit on
3,556 D1 pitchers, R^2 0.68, residual sd ~10 share pts. The residual feeds a
coach-facing disagreement queue ("Usage Gap Board"), where each resolved case
is logged and scored against next season - both a conversation tool and a
curated sample of what the current features miss.

A prior version of this section described the trait directions inverted (as a
tunneling/overexposure story). The predictive numbers were unaffected -
orientation cancels in correlations and blends - but the narrative was wrong;
it was caught by an inverse-selection question in review.

Label it honestly: this is a THIRD construct - deployment/revealed trust - not
stuff and not location (FRAMEWORK.md, one construct per score). And treat it as
DISCOVERED, NOT CONFIRMED: the traits were screened on the same season-pair
they are evaluated on. Direction replication across two pitch types is the only
internal validation available; 2026 is the real test. Do not ship a score from
this yet.

## Production summary

- Report three scores: adjusted results, Stuff+ (Ridge), Location+ (pooled map),
  plus Pitching+ = equal-weight z blend, all at 100 +/- 15. Script 08 emits the
  staff scoresheet (scores, whiff, fingerprints, flags, explanation grids).
- Pitch-level explanations use the count-conditioned map.
- Location+ carries a small-sample flag below ~50 FF, caution to 99.
- Location+ is a fastball score: do not surface it for secondary pitches
  (reliable but zero validity there); secondaries are graded by Stuff+.
- Ceiling check: criterion reliability caps attainable validity at
  ~sqrt(0.304) = 0.55; the stack reaches 0.392 (~70% of ceiling).
- Open items: what does predict secondary-pitch outcomes beyond stuff
  (usage/sequencing context), joint model vs blend, 2026 replication.

## 2026 replication

Everything above was one season-pair (2024->2025, n=699). The 2026 season
(FTP mirror -> API-validated build, frozen 2024-25 ExpectedRuns table,
inclusive RunsRemaining matching the original build) allows the promised
replication: the identical suite re-run on 2025->2026 via
`--years 2025,2026` (year-role relabeling in `fair_criterion.py`; train/eval
labels in script output then mean 2025/2026). Because the national feed's
level mix shifted (D2 share roughly doubled), all headline calls below are
read from D1-only control runs (`--level D1`; panels n=649 old pair, n=825
new pair); all-levels runs agree directionally except where noted.

REPLICATED (adopt as standing findings, no longer provisional):
- Criterion structure: C0 < C1 < C2 ordering for reliability and blend
  validity; C1-C2 cross-corr 0.986/0.987.
- Stuff+ increment over results+Location+ (D1: P=1.000 old, 0.998 new) and
  Location+ increment over results (D1: P=0.994 old, 0.983 new). The
  all-levels Location+ CI includes zero in the new pair -- the increment is a
  D1 signal diluted by feed expansion, not a failed effect.
- Models-only (no results information) beat raw results (D1: +0.195 P=1.000
  old; +0.114, CI [+0.040,+0.189], P=0.999 new); the margin over adjusted
  results stays positive but borderline in both pairs (P=0.992 old D1,
  0.957 new D1).
- Count-conditioning decomposition: raw count map beats pooled on
  reliability (P=0.999), the gain is count occupancy, location-given-count
  ties pooled, and the 3-way blend advantage collapses (+0.006 new vs +0.005
  old) -- the one-construct-per-score decision to keep pitcher-level
  Location+ count-relative is confirmed. (Tuned shrinkage moved m=5 -> m=2;
  scheme winner count12 unchanged.)
- Scaling: the college stuff-vs-location SD inversion (ratio 0.78 vs 0.80),
  honest Location+ scale (~100+/-19 both pairs), equal-weight z blend still
  beats reliability weighting (P=0.987).
- Sample floor: n0 = 51 vs ~52; identical flag tiers.
- Expected-usage share model: R^2 0.647 vs 0.68, residual sd ~10 pts.
- Secondary pitches: adding Location+ to slider/changeup blends still makes
  prediction worse; Location+ remains a fastball score.

ATTENUATED (real, not composition -- persists under D1-only):
- Absolute levels are ~10% lower across the board in the new pair: C2
  reliability 0.304 -> 0.268, Stuff+ Ridge reliability 0.901 -> 0.828, ridge
  validity 0.282 -> 0.214 (all D1). With SE ~0.037 these are 1-2 SE moves,
  consistent in direction across every score, so treated as environment/
  measurement drift to monitor rather than a design problem.
- Secondary-pitch Location+ validities are no longer ~0 (slider 0.058,
  changeup 0.150, curveball 0.245 all-levels) but remain well below the
  four-seam level; the curveball criterion-reliability sign flip (-0.09 ->
  +0.34) says the old n=33 panel was noise, as flagged.

REFUTED (withdraw):
- The deployment/revealed-trust composite. Its 2024->2025 D1 numbers (slider
  validity 0.299, increment over results+Stuff+ +0.116, CI [+0.044,+0.188],
  P=0.999; all-levels: 0.300, +0.121, P=1.000) collapse in 2025->2026 to
  validity 0.006/0.027 (slider/changeup, D1) with the increment flipping
  negative (slider -0.053, changeup -0.014, curveball -0.131 with CI
  excluding zero). Individual trait signs (usage, sep_x) are
  unstable across pitch types. The composite is still reliable (~0.6) -- a
  stable pitcher attribute -- but it does not predict future run value. The
  DISCOVERED-NOT-CONFIRMED label and the do-not-ship gate did their job;
  deployment traits stay out of every score. The Usage Gap Board is
  unaffected (it rests on the share model, which replicated).

Replication artifacts: `--years`/`--level` parameterization in
`fair_criterion.py`; per-script logs and the full metric-by-metric comparison
(replication_comparison.md) in the analysis workdir (Level II, not committed).

Post-replication addendum (2026-07-23): collaborator review asked whether the
location-scatter penalty weakens for elite arms (batters timing predictable
locations). Tested on both D1 panels: the sd_z x stuff-quality interaction is
null in both pairs (t=+0.39 and t=-0.52, opposite signs), and the elite
tercile's scatter-criterion correlation (~+0.20) matches the full panel.
Within college, location consistency is good at every stuff tier; any
predictability cost starts above the college ceiling. MLB context for the
reliability gap (FanGraphs primer): MLB year-over-year reliability is ~0.73
Stuff+ / ~0.48 Location+ -- college Location+ reliability (0.479) matches MLB
almost exactly; what inverts at college is location's VALIDITY, not its
repeatability.

Follow-up (same day): is scatter bad in itself, or only as a proxy for bad
misses? Conditioning on control (waste-rate terciles) and partialling waste
out, on both D1 pairs: horizontal scatter is mostly wildness by another name
(corr with waste 0.65-0.71; partial validity collapses to +0.04/+0.08).
Vertical scatter keeps a modest residual penalty (partial +0.09/+0.11, ~2-3
SE) but among the cleanest tercile it is inconsistent across pairs (-0.03
then +0.11). Nowhere does more varied placement HELP -- even for clean
pitchers, variation is at best neutral. Coaching translation: the actionable
sin is the waste pitch, not variation per se; sd_x adds little beyond waste
rate, sd_z a little.

Follow-up 2 (2026-07-23): waste-type decomposition + stability of individual
location features (both D1 pairs, pitchers with >=100 FF in both years,
n=649 / n=825; script: stuffplus_replication/waste_and_feature_stability.py).

Waste as the coach target. Waste rate is more predictive than either scatter
axis (validity +0.258 / +0.210 vs sd_x +0.21/+0.19 which collapses when
waste is partialled) and decently reliable (0.405 / 0.342, vs criterion
self-reliability 0.304 / 0.268). It is also directly countable per outing.

Waste TYPE matters, asymmetrically. Volume: horizontal misses are the bulk
of waste (6.2-6.6% of all FF vs 1.2-1.5% low, 2.2-2.6% high). Per-pitch cost
is roughly equal across types (+0.11 to +0.13 runs vs base). But as a
pitcher TRAIT the types diverge sharply, replicated in both pairs:
  waste-low   rel 0.37/0.34  validity +0.198/+0.198  (most validity-dense)
  waste-horiz rel 0.49/0.46  validity +0.173/+0.151
  waste-high  rel 0.45/0.39  validity +0.037/+0.001  (stable but PREDICTS NOTHING)
High-miss rate correlates with BETTER stuff (corr with ridge_pred -0.07/-0.06,
lower=better) while horizontal-miss rate goes with worse stuff (+0.17/+0.19):
missing high is partly a byproduct of ride/velo arms attacking up, which
offsets its per-pitch cost at the trait level. Second live example of
stable-but-not-predictive (after the deployment composite). Coach board
implication: count waste and split it; side-spray and buried misses are the
concern, occasional overthrow-high is not a red flag.

Stability of individual Location+ features (rel pair1/pair2, validity
pair1/pair2; validity sign: positive = predicts worse outcomes):
  sd_z        0.60/0.56   +0.17/+0.18
  sd_x        0.57/0.52   +0.21/+0.19  (proxy for waste)
  mean_z      0.54/0.53   -0.14/-0.11  (partial|waste -0.12/-0.13: living
                                        higher helps INDEPENDENT of waste)
  waste_pct   0.41/0.34   +0.26/+0.21
  heart_pct   0.29/0.27   -0.27/-0.15  (partial|waste -0.15/-0.04: its
                                        independent contribution is shaky)
  chase_pct   0.18/0.16   +0.15/+0.14
  shadow_pct  0.09/0.13   -0.08/-0.14  (near-noise as a trait despite being
                                        41% of pitches)
Design implication: Location+'s trait signal concentrates in waste
avoidance, average height, and (as proxies) the scatter SDs; shadow% and
chase% are mostly noise at the pitcher level and any attribution built on
them attributes noise. The mixed reliability of these ingredients is
consistent with Location+'s overall 0.48 -- the measurement-improvement
lever is to weight the stable+valid features (waste, mean_z) and downweight
the unstable ones.

Portal buy-low backtest (2026-07-23): does bad-results + good-model-grade
identify pitchers who outperform their line the next year? Within each D1
pair, take the worst tercile on year-1 surface results (mean Target) and
split by year-1 model grade (mean ridge_pred). Next-year mean Target:
  2024->2025: BUY (good model) -0.0074 vs AVOID (bad model) +0.0043
  2025->2026: BUY -0.0040 vs AVOID +0.0037
The BUY group crosses to better-than-average despite a bottom-tercile prior
line; the gap (~0.008-0.012 runs/pitch) is roughly a win over a full season
of fastballs. Symmetric sell-high check: best-results + bad-model pitchers
regress hard in pair 1 (-0.030 -> -0.001 vs -0.018 for the real group);
attenuated but same direction in pair 2. Caveats: same-school population
(portal transfers change park/defense/level context), FF-only panel, and
portal candidates often have thinner samples than the >=100-pitch floor
used here.

Correction + coach-metric version (same day): a first coach-facing "control"
(worst-line tercile split by grade tercile) was confounded -- the groups
started 0.84 RA9 apart and ended 0.85 apart, so it displayed nothing beyond
the stat line (caught by the coach-consumer review agent). The clean
evidence, 2025->2026 D1, grade = Pitching+ (equal-z Stuff+ ridge +
script-03 pooled-map Location+), coach metric RA9 (runs while on mound
per 9, outs = OutsOnPlay + K):
  Matched pairs (n=74, matched on 2025 RA9, caliper 0.30): baselines
  8.27 vs 8.39; next year 6.24 (liked) vs 6.65 (disliked); K% 19.1->21.6
  vs 16.5->18.1. Suggestive at this n (paired diff -0.41, SE 0.42).
  Regression (n=543, bootstrap 4000): holding 2025 RA9 fixed, +1 SD of
  grade = 0.39 runs better next-year RA9, 95% CI [0.19, 0.60]; holding
  RA9 + K% + BB% all fixed, 0.30 +/- 0.11 (P=0.999). The model's edge
  beyond the full surface line is real in coach units: ~0.3-0.4 RA9 per
  SD, on top of near-universal regression-to-mean improvement.
Scripts: stuffplus_replication/build_portal_data.py (board data) and the
matched/regression ad-hoc runs logged in the session transcript.

Arsenal grade beats FF-only (2026-07-23, both D1 pairs): extending the
grade to SL/CH/CB via the script-09 protocol (per-type ridge + per-type
pooled location map; per-pitch predictions are in run units so the
arsenal grade is just the pitcher's mean predicted value across all
graded pitches - usage-weighted by construction; median arsenal
coverage 94% vs ~65% FF-only). Effect on next-year RA9 per SD, holding
RA9+K%+BB% fixed:
  2025->2026 (n=543): FF-only +0.30 [+0.10,+0.53]; arsenal +0.42
    [+0.20,+0.65]; joint model FF -0.00, arsenal +0.42 - the arsenal
    grade SUBSUMES the FF-only grade.
  2024->2025 (n=394): FF-only +0.18 [-0.10,+0.47]; arsenal +0.24
    [-0.06,+0.54]; joint FF -0.05, arsenal +0.28. Same ordering and
    same subsumption pattern, attenuated on the smaller pool.
Verdict: adopt the arsenal grade for portal evaluation (theorized
construct, direction replicated in both pairs, subsumption replicated).
Scripts: stuffplus_replication/arsenal_grade_test.py / _2425.py.

Arsenal weighting choice (same day): four schemes compared, learned
weights fit ONLY on 2024->2025 and tested out-of-sample on 2025->2026
(screening discipline). Effect per SD holding RA9+K+BB fixed
(2425 / 2526):
  A usage-weighted raw predictions      +0.25 / +0.42
  B mix-neutral (quality vs type mean)  +0.31 / +0.44
  C = B + n/(n+51) shrinkage            +0.31 / +0.43
  D learned per-type weights            (fit) / +0.29  <- worst OOS;
    fitted weights unstable (negative slider-stuff weight)
Adopted B: within 1 SE of A predictively but better in BOTH pairs, and
the cleaner construct - each pitch graded against its own type's
population mean, so pitch MIX (an occupancy-like skill) is not absorbed
into the quality score, mirroring the count-conditioning decision.
Board rebuilt on arsenal-B: regression 0.50/SD holding RA9
[CI 0.31,0.70], 0.43 holding full line [0.22,0.67]; matched pairs
(n=75) 8.22->6.09 vs 8.35->7.01 (0.92-run gap, 59 vs 53 improved);
survivorship now equal (33% vs 34% return) so attrition does not
explain the gap. Script: stuffplus_replication/arsenal_weighting_test.py.

Blend weight check (2026-07-23): should the arsenal grade weight Stuff
and Location equally? Sweep of w*z(stuff)+(1-w)*z(loc) on both D1
pairs: flat plateau across w=0.4-0.75 (2526 effects 0.43/0.44/0.44/
0.42; 2425 0.28-0.34), equal weight on the plateau both times. The
2425-fitted tilt (0.80 stuff) applied out-of-sample to 2526 gives
+0.42 vs equal's +0.44 - fitted weights don't transfer, and the
components' relative importance is itself unstable (joint coefs
stuff/loc +0.33/+0.09 in 2425, +0.36/+0.25 in 2526) while being
uncorrelated (r~0.01). KEEP EQUAL WEIGHTS - consistent with the
script-05 finding that equal-z beat reliability-weighted blends.
Script: component_model/portal/blend_weight_test.py.

Extended pitch coverage (2026-07-23): does grading Sinker (incl.
two-seam), Cutter, and Splitter on top of FF/SL/CH/CB improve the
arsenal grade? Same protocol (mix-neutral B, equal stuff/loc weights,
effect = next-year RA9 per SD holding RA9+K+BB fixed), paired
bootstrap on the difference so the comparison shares resamples:
  2425: 4-type +0.31, 7-type +0.33, paired diff +0.03 SE 0.03
  2526: 4-type +0.44, 7-type +0.48, paired diff +0.04 SE 0.02
Small but consistent (never worse, ~2 SE on 2526), and it closes a
systematic blind spot: sinker/cutter-primary pitchers are the
soft-contact profile the v1 whiff model missed, and they were being
graded on partial arsenals. Sweeper excluded (zero 2024 pitches, no
training data in the 2425 pair). ADOPTED the 7-type grade. Board
rebuilt: regression 0.54/SD holding RA9 [CI 0.35,0.74], 0.48 holding
full line [0.26,0.71]; matched pairs (n=72) 8.21->6.05 vs 8.33->7.10
(1.05-run gap, 57 vs 50 improved); liked top-50 9.41->6.16 (44/50
improved); survivorship still equal (32.6% vs 33.2%). Board JSON now
also carries a per-pitcher `detail` payload (per-type within-type
Stuff+/Location+ and top-3 exact ridge feature contributions,
context features excluded from display) for the board's "why we like
him" tooltip. Scripts: component_model/portal/extended_types_test.py,
build_portal_data.py (v3).

## Noise vs missing skill (script 12, 2026-07-27)

How much of our year-over-year unpredictability is irreducible and how
much is skill Pitching+ fails to encode? Three buckets, only the third
of which is a model problem: measurement noise (a college season is too
few pitches), true drift (pitchers really change), and missing skill
(persistent, year-1-visible talent the physical scores do not carry).

Panel: 1,377 D1 pitchers, 2,945 pitcher-seasons, 2024-2026, four-seam,
100+ FF per season and 2+ qualified seasons, median 221 FF. Criterion
xT, fit once on all three seasons pooled so a given EV/LA outcome maps
to the same run value in every year. Built from the existing D1 pitch
caches; note the 2025-2026 cache stores role-relabeled years, so
calendar years are re-derived from Date.

WITHIN-SEASON RELIABILITY of mean xT, split by whole GAME and
Spearman-Brown corrected to full length: 0.283 (2024), 0.362 (2025),
0.295 (2026), mean 0.314. Stuff+ anchor 0.983-0.988 on the same split,
confirming the split itself. This sits essentially on top of the
across-season figure (C1/C2 reliability 0.27-0.30), which is the
diagnostic: the criterion is noisy, not drifting. Note this 0.314 is
(stable + drift) / total, i.e. "how much of a season repeats within
itself" -- NOT the stable-skill-only share below (24.4%). Conflating
the two is the mistake to avoid: asked live where the 24.4% figure came
from, the instinctive answer is to point back at this number, and it's
wrong -- 24.4% comes from the separate cross-season covariance measure
further down.

Game parity matters. Script 06 splits by PITCH parity, which leaves
within-game shared variance (batter, park, umpire, day) in both halves
and reads high; its n0 ~= 51 floor is optimistic for that reason. The
same effect makes a pitch-level variance understate the error in a
season mean by 1.11x here (design effect from the half-split), and on
the naive scale Part B disagreed with Part A by 0.06 reliability, with
the difference being credited to skill. The clustered scale is used
throughout.

BUCKET SHARES of single-season variance in mean xT (cluster bootstrap
over pitchers, 1000 reps):

| bucket | share | 95% CI |
|---|---|---|
| measurement noise | 69.7% | [66.0, 74.0] |
| true drift | 6.0% | [0.4, 10.9] |
| stable skill | 24.4% | [19.3, 29.3] |

Persistence (stable / stable+drift) = 0.80, CI [0.65, 0.99]. Nearly
everything that repeats within a season also survives to the next one.
The ceiling on year-ahead forecasting is set by how little of a college
season is signal, not by pitchers changing.

MISSING SKILL, stable variance surviving the physical scores (results
deliberately excluded: explaining a results-based criterion with a
results-based predictor is circular):

| predictors | captured share of stable skill |
|---|---|
| Stuff+ only | 12.2% |
| Location+ only | 50.1% |
| Stuff+ and Location+ | 57.5% |

Physical Pitching+ captures 57.5% of stable skill (bootstrap 58.0%, CI
[48.0, 68.9]); 42.5% is MISSING, which is 10.3% of total single-season
variance. Coefficient signs both positive as required (stuff +0.0029,
loc +0.0082 on z-scored predictors against a lower-is-better criterion).
Cross-check: the decomposition implies a year-ahead validity of
sqrt(0.575 x 0.244) = 0.375 for a predictor carrying exactly the captured
part. Script 07 measures 0.345 for Location+ + Stuff+ with no access to
the pitcher's own results, the directly comparable quantity. The two
routes to the same ceiling agree, and the small gap sits in the expected
direction given the lag decay below.

Absolute ceiling (100% of stable skill captured, zero missing): sqrt(0.244)
= 0.494. Current Pitching+ (0.345) is 70% of that CORRELATION ceiling but
only 49% of it in VARIANCE (R^2) terms (0.345^2 / 0.244) -- these two
percentages differ because r compresses ratios via a square root relative
to R^2. State one or the other explicitly; "about half" (the R^2 framing)
and "most of the way there" (the r framing) are both defensible readings
of the same number and will contradict each other if used loosely in the
same conversation.

"Stable" means persists to next season, not permanent. Covariance by
lag: 0.000091 at one year, 0.000043 at two, retention 0.48 with CI
[0.06, 0.95] and P(lag2 < lag1) = 0.981. About half the one-year signal
is gone by two years, so the three-bucket model's permanent-plus-
independent-shock structure is a simplification of something closer to
a decaying component.

VERDICT: measurement noise dominates at roughly 70%, and true drift is
small (6%, CI barely excluding zero). Of the ~24% that is stable skill,
the physical scores already carry well over half. The missing skill is
real but bounded: ~10% of single-season variance, which caps what any
better feature set can add. Two implications. Improving the features has
much less headroom than the low pitch-level R2 suggests, and the larger
lever is precision, since noise is a function of sample size and can be
attacked with more pitches, multi-season pooling, or shrinkage rather
than with new physics. Location+ carries 4x what Stuff+ does here
(captured share of stable skill: 50.1% vs 12.2%), which is worth
weighing against the fastball-only restriction in the production rule.
Metric note: that 4x is NOT the same comparison as script 07's raw
next-year predictive correlation (Location+ 0.290 vs Stuff+ 0.235, only
~1.2x) -- raw correlation is diluted by noise both scores are equally
powerless against, which compresses the visible gap between a strong
predictor and a weak one. Cite the captured-share ratio (4x) when
talking about how much of true skill each score explains; cite the
script-07 ratio (1.2x) when talking about how much better one score is
as a practical forecasting tool. Don't use one number under the other's
label.

Caveats: D1 only (the sole all-levels 2024 frame lacks a Level column;
the feed's D2 share doubled, so all-levels would be composition-diluted
anyway). Four-seam only. Three seasons, so drift and stable skill are
separated but not precisely, and the lag-2 covariance rests on 285
pitchers. The clustered noise scale is treated as known in the bootstrap.
Same decomposition on adjT tracks closely (noise 68.2%, drift 5.5%,
stable 26.3%, captured 64.1%). Part C, naming the missing channel, was
scoped out pending this result.
Script: component_model/analysis/12_reliability_decomposition.py
(estimator + tests: variance_components.py, tests/).
