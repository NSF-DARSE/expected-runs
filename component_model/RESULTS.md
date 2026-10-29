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
C2 was replaced on 2026-10-29 by a league-strength-adjusted version; the current anchors
are in "Adopted: league-strength-adjusted adjT" at the end of this file.

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

## Conference strength and the fair criterion (script 17, 2026-10-29)

Question: adjT (C2) subtracts each game's League mean xT and a batter effect measured
against the batter's own league mean. Within a closed conference those means cannot see
how strong the conference is, so C2 may grade a weak-conference pitcher too kindly. Four-
seam, D1, 2024 to 2026. Nothing shipped changes here; add_adjusted is untouched.

What League holds. It is a property of the game: one value per game (one exception in
16,422), and it is the home team's conference. A team's home games sit under one League
98% of the time (median 100%), while its away games scatter across opponents (median
55%). A team's conference is therefore the modal League of its home games. Teams with no
TrackMan home park borrow their conference from another season's home games, then from a
confident away-game mode, and otherwise go to an unknown group (3.5 to 4.0% of pitches by
batting team). Conference names are TrackMan's codes.

Linkage. Non-conference games are what separate "this conference hits well" from "this
conference pitches badly". There are plenty: 2,135 of 4,540 games in 2024, 2,463 of 5,570
in 2025 and 2,715 of 6,312 in 2026 are cross-conference, over 239 to 280 linked
conference pairs. The conference graph is a single connected component in every season.
The thinnest links are SWAC (6 cross-conference games in 2024, 19 in 2025) and NEC (13 in
2026), and their intervals below are correspondingly wide.

Method 2, the joint fit. Pitch-level xT on pitcher, pitcher conference, batter, batter
conference, park (home team) and batting-at-home effects, all against one global mean,
ridge-penalized (individual and park effects K = 100 as in BATTER_K; conference terms
penalty 10, so they are estimated almost freely from cross-conference games). Fit
separately on each season, using only that season. The conference terms are essential:
with only individual effects the ridge penalty pulls every conference-wide shift toward
zero, which the synthetic tests in tests/test_league_strength.py show directly. The
candidate criterion adjT_global (G) is xT minus the fitted batter, batter-conference,
park and home effects, recentered on mean xT. Lower is better, as for every criterion.

Conference strength, batter side, in run value per 100 pitches against the D1 average
(higher means tougher hitters). Three-season average with 95% game-cluster bootstrap
intervals (200 refits per season), with the method 3 cross-check alongside:

| conference | joint fit (m2) | 95% CI | Pitching+ residual (m3) | 95% CI |
|---|---|---|---|---|
| SEC | +0.94 | [+0.83, +1.03] | +0.77 | [+0.70, +0.88] |
| ACC | +0.75 | [+0.62, +0.86] | +0.90 | [+0.78, +1.01] |
| PAC12 | +0.73 | [+0.37, +1.02] | +0.35 | [+0.02, +0.65] |
| BIG12 | +0.56 | [+0.43, +0.69] | +0.71 | [+0.60, +0.82] |
| BIG10 | +0.31 | [+0.18, +0.41] | +0.45 | [+0.34, +0.54] |
| BW | +0.06 | [-0.14, +0.22] | -0.38 | [-0.53, -0.22] |
| AMER | -0.01 | [-0.16, +0.11] | -0.12 | [-0.26, +0.01] |
| SBELT | -0.01 | [-0.15, +0.12] | -0.21 | [-0.31, -0.09] |
| CUSA | -0.02 | [-0.14, +0.14] | +0.08 | [-0.05, +0.24] |
| SOCON | -0.03 | [-0.21, +0.17] | -0.11 | [-0.29, +0.09] |
| SLAND | -0.22 | [-0.44, +0.03] | -0.40 | [-0.63, -0.21] |
| WAC | -0.25 | [-0.43, -0.07] | -0.32 | [-0.47, -0.15] |
| CAA | -0.38 | [-0.53, -0.22] | -0.13 | [-0.26, +0.00] |
| WCC | -0.40 | [-0.57, -0.25] | -0.67 | [-0.81, -0.55] |
| BEAST | -0.41 | [-0.61, -0.23] | -0.25 | [-0.44, -0.07] |
| MVC | -0.42 | [-0.56, -0.24] | -0.11 | [-0.25, +0.05] |
| MW | -0.43 | [-0.66, -0.23] | -0.45 | [-0.62, -0.30] |
| A10 | -0.45 | [-0.62, -0.31] | -0.36 | [-0.50, -0.24] |
| BSOU | -0.46 | [-0.63, -0.29] | -0.38 | [-0.55, -0.18] |
| ASUN | -0.51 | [-0.66, -0.33] | -0.29 | [-0.45, -0.14] |
| MAC | -0.59 | [-0.79, -0.43] | -0.39 | [-0.62, -0.19] |
| OVC | -0.65 | [-0.87, -0.44] | -0.65 | [-0.88, -0.50] |
| AMEAST | -0.75 | [-1.03, -0.44] | -0.25 | [-0.57, +0.03] |
| SWAC | -0.77 | [-1.42, +0.02] | -0.97 | [-1.63, -0.31] |
| HORZ | -0.87 | [-1.09, -0.63] | -0.81 | [-1.03, -0.56] |
| PAT | -1.02 | [-1.30, -0.73] | -0.66 | [-0.88, -0.43] |
| SUM | -1.07 | [-1.50, -0.59] | -0.69 | [-1.07, -0.27] |
| IVY | -1.09 | [-1.28, -0.86] | -0.54 | [-0.74, -0.34] |
| NEC | -1.39 | [-1.79, -1.05] | -1.30 | [-1.77, -0.91] |
| MAAC (2026 only) | -1.39 | [-1.89, -0.92] | -1.74 | [-2.11, -1.27] |

PAC12 is not one conference across the three seasons: after 2024 it is two programs, and
its 2025 value (+1.61) rests on very few teams. The spread is about 0.6 to 0.65 per 100
pitches (sd across conferences) in every season, and the SEC-to-NEC gap is about 2.3 per
100 pitches. The fitted pitcher-side conference effects run the other way (correlation
-0.73 to -0.88 with the batter side): conferences that hit well also pitch well, which is
why conference mean xT carries little of the gap.

Method 3 cross-check. Mean residual of four-seam xT after a pooled fit on the fixed
Stuff+ ridge and pooled Location+ (both coefficients positive, as required), by batter
conference. Rank agreement with method 2 is Spearman 0.79, 0.80 and 0.74 by season
(paired bootstrap intervals roughly [0.52, 0.88]) and 0.82 on the three-season average.
The slope of method 3 on method 2 is 0.74 to 0.83: method 3 compresses the gap, as
expected when strong programs also recruit pitching skill Pitching+ does not measure.
Two methods with different identifying sources agree on the order. Method 3 is a check
only; grading Pitching+ against a criterion built from Pitching+ would be circular.

The premise is partly wrong. League mean xT is not flat: its sd across conferences is
0.26 to 0.41 per 100 and it correlates +0.14 to +0.26 with batter-side strength. The
calibration below puts a number on it: C2 removes 37% (2025) and 45% (2026) of opponent
conference strength and leaves the rest in.

Out-of-season calibration (added because the mover test below is underpowered). Take the
season-t conference strengths, then look at season t+1 pitches with pitcher-season fixed
effects and season-t park and home controls, so the only variation left is one pitcher
facing different conferences. Nothing from t+1 informs the strengths. The slope of xT on
opponent strength is +0.80 (SE 0.04) for 2024 strengths on 2025 results and +0.97 (SE
0.04) for 2025 on 2026, so the strengths hold up out of sample at roughly the right
scale. C2's slope is +0.51 and +0.53 (SE 0.04): about half of opponent conference
strength is still in the shipped criterion. G's slope is +0.01 and +0.00, close to zero
by construction and shown only as a check.

Reliability and validity. Panel is pitchers with 100+ qualifying four-seams in both
seasons (650 for 2024 to 2025, 828 for 2025 to 2026). C2 is refit per season so no
criterion sees another season; the pooled-frame C2 that scripts 01 and 12 use tracks it
within 0.01 everywhere. Paired bootstrap over pitchers, 2000 reps; values are the mean
over both pairs.

| metric | xT | C2 | G | G minus C2 (SE) |
|---|---|---|---|---|
| reliability r(C_t, C_t+1) | 0.265 | 0.283 | 0.318 | +0.034 (0.006) |
| Stuff+ validity r(ridge_t, C_t+1) | 0.207 | 0.253 | 0.291 | +0.037 (0.005) |
| Location+ validity r(loc_t, C_t+1) | 0.256 | 0.273 | 0.289 | +0.016 (0.005) |
| Stuff+ increment over own-criterion baseline | 0.040 | 0.060 | 0.066 | +0.006 (0.004) |

Each pair goes the same direction (reliability +0.041 and +0.028, Stuff+ validity +0.033
and +0.042). Reliability rises when a stable context term is removed, the opposite of
what removing context usually does. That fits the opposing signs above: in C2, a good
pitcher in a strong conference partly cancels against his tough schedule, and G
separates the two again.

Decomposition audit. Most of the validity gain is between conferences. Among conference
stayers, Stuff+ validity within conference moves only from 0.199 to 0.207 (2024 to 2025)
and 0.167 to 0.186 (2025 to 2026), against 0.272 to 0.303 and 0.247 to 0.298 overall.
Stuff+ correlates -0.43 and -0.36 with the conference strength a pitcher faces next
season, meaning better physical stuff sits in tougher conferences. The gain comes from G
no longer charging those pitchers for their schedule. That is a valid gain only if the
strength estimates are right, which is what the next two tests ask. The gain does not
make Stuff+ more useful relative to results: last season's own criterion improves by
about as much, and Stuff+'s increment over that baseline is a tie (+0.006, SE 0.004).

The deciding test: conference movers. 242 panel pitcher-pairs change conference (103 for
2024 to 2025, 139 for 2025 to 2026): 163 changed team and 79 stayed with a team that
realigned. 156 moved to tougher hitters and 86 to easier ones; the opponent strength they
faced changed by a mean of +0.16 per 100 (sd 0.41). The residual is next season's
criterion minus what the stayers' year-to-year regression predicts, which absorbs
regression to the mean. A correct adjustment should leave no relationship between that
residual and the change in strength faced.

| movers | xT | C2 | G | G minus C2 (SE) |
|---|---|---|---|---|
| mean residual, to tougher hitters (per 100) | +0.13 | -0.07 | -0.21 | -0.14 (0.03) |
| mean residual, to easier hitters (per 100) | -0.55 | -0.42 | -0.36 | +0.07 (0.04) |
| slope of residual on change in strength | +0.29 | -0.14 | -0.47 | -0.33 (0.05) |
| same, controlling change in Stuff+ and Location+ | +0.36 | -0.07 | -0.40 | -0.34 (0.05) |
| RMSE predicting movers / sd of criterion | 0.973 | 0.972 | 0.959 | -0.013 (0.014) |
| year-over-year r, movers | 0.274 | 0.307 | 0.337 | +0.030 (0.015) |

The bootstrap SE of each slope on its own is about 0.28 to 0.29. The paired SE on G minus
C2 is small only because the two criteria differ by a nearly fixed adjustment; it does
not say which one is right. Read alone, the test leans against G. C2's slope is 0.5 SE
from zero, G's is 1.6 SE below zero (it over-corrects), and G's slope is closer to zero
in only 14.5% of bootstrap reps. G predicts movers slightly better in correlation, but
stayers gain more (+0.038), so the mover-specific gain is -0.007 (SE 0.017), none.
Splitting movers into team changes and realignment, or controlling for the change in
physical stuff, does not change the picture.

The discrepancy that matters. Within a season, one pitcher's xT moves about one-for-one
with opponent strength (calibration slope 0.80 to 0.97). Across seasons, a mover's season
mean moves only about 0.3 per unit of change in strength (SE 0.29), about 2 SE below
that. Selection on a lucky or unlucky season would push the mover slope up, not down, so
it does not explain the gap. Two explanations fit and the data cannot yet separate them.
Pitchers who move up may also be improving in ways the physical scores do not capture,
for example development at a stronger program. Or non-conference games, which identify
the strengths, differ systematically from conference play (early-season dates, midweek
arms, neutral sites), so the strengths fit those games and carry only partly to a full
season.

VERDICT: do not adopt G yet. Two pieces of evidence with different sources (the joint
fit's cross-conference identification and the Pitching+ residual cross-check) agree on
the conference order. Out-of-sample calibration says C2 still carries about half of
opponent conference strength. G is 3 to 7 SE better on reliability and on Stuff+ and
Location+ validity. But the gain is almost entirely between conferences, it gives Stuff+
no advantage over results, and the one test that uses pitchers as their own control
across seasons does not confirm it: at the point estimate G over-corrects. That test has
n = 242 and a slope SE near 0.29, so it is underpowered rather than decisive. Needs more
data: the 2027 movers, read on this same slope as a fixed pre-registered test, would
roughly double n. If the mover slope stays near 0.3, a partial adjustment (subtract a
fraction of the conference term) is the natural candidate. Choosing that fraction from
this mover sample would be fitting to the test, so it was not tried here.

Caveats. Run on the current-schema caches (workdir_ext3 for 2024 and 2025, ext3_crit for
2025 and 2026). The older workdir_2425_d1 and workdir_2526_d1 caches predate RelSpeed and
cannot fit the shipped Stuff+ ridge; the script stops with a message if handed one.
Conference membership is inferred from home games; teams without a TrackMan park borrow
another season's conference, which assumes they did not realign in between. Park is the
home team, so neutral-site games are charged to the host. The batter key is name plus
team, as in add_adjusted.
Script: component_model/analysis/17_league_strength.py (helper and synthetic-recovery
tests: league_strength.py, tests/test_league_strength.py).

## Adopted: league-strength-adjusted adjT (2026-10-29)

Jack adopted G as the project's C2 on 2026-10-29, overriding the "do not adopt yet"
verdict above. The reliability and validity gains are 3 to 7 SE and consistent across
both pairs. The conference-mover test that leaned against G has n = 242 and a slope SE
near 0.29, which he judged too small to block adoption. The mover test stays the open
check: read it again on the 2027 movers as planned above.

What changed in code. fair_criterion.add_adjusted now returns adjT from the per-season
joint fit (league_strength.add_global_criterion, through league_strength.fit_season,
which script 17 also calls, so the research and the scorers share one code path). The
column is still adjT and the orientation is unchanged: expected run value from the
pitcher's view, lower is better. Two other changes come with it. Each season in a frame
is fit on its own rows only, where the old version pooled batter effects across every
season in the frame. A batter is shrunk toward his conference effect instead of his
league mean, with the same K = 100. The old criterion is kept as
add_adjusted_within_league, and STUFFPLUS_ADJ_METHOD=within_league switches a whole run
back to it (a banner prints when it does). ff_panel's cache name now carries the method,
so a panel cached before today cannot serve the old adjT under the new name.

Script 17 rerun on the shared code path reproduces the table above to three decimals
(reliability +0.034, SE 0.006; Stuff+ validity +0.037, SE 0.005; Location+ validity
+0.016, SE 0.005), so moving the fit into league_strength changed nothing. The rerun used
20 game-cluster refits per season instead of 200, which only affects the conference
interval and calibration SE columns. Calibration slopes match (C2 +0.51 and +0.53, G
+0.01 and +0.00), and so does the mover slope (G -0.47, C2 -0.14).

Script 01 anchors on the current caches (D1, panel 100+ FF in both seasons). The old row
is the within-league C2 as script 01 has always built it (one fit over the two-season
frame); the new row is the adopted adjT.

| pair | criterion | reliability | ridge -> C_t+1 | 50/50 blend -> C_t+1 |
|---|---|---|---|---|
| 2024 -> 2025 (n = 649) | C1 xT | 0.283 | 0.219 | 0.325 |
| | C2 within-league (old) | 0.306 | 0.260 | 0.360 |
| | C2 adopted | 0.349 | 0.297 | 0.407 |
| 2025 -> 2026 (n = 828) | C1 xT | 0.231 | 0.200 | 0.276 |
| | C2 within-league (old) | 0.251 | 0.235 | 0.308 |
| | C2 adopted | 0.277 | 0.271 | 0.345 |

C0 and C1 are unchanged by construction. The C1 to C2 cross-correlation falls from 0.984
to 0.956 and 0.958: the adopted criterion moves further from raw xT than the old one
did, which is the conference term it now removes. These are the anchors script 01 must
reproduce from now on. The 2024 to 2025 table at the top of this file is the original
build (n = 699, K = 200) and is kept as history.

Incremental-validity gate (coach_incremental_gate.py, 2024+2025 grades vs the 2025 to
2026 criterion build, 200 refits, bar P = 0.95). Mean blend gain over prior results and
P(gain > 0), old then new:

| type | old gain | old P | new gain | new P | verdict |
|---|---|---|---|---|---|
| FF | +0.068 | 1.000 | +0.074 | 1.000 | PASS, unchanged |
| SI (pooled) | +0.046 | 0.890 | +0.060 | 0.925 | no, unchanged |
| FC | +0.036 | 0.795 | +0.049 | 0.860 | no, unchanged |
| SL | +0.056 | 1.000 | +0.067 | 1.000 | PASS, unchanged |
| CB | +0.143 | 1.000 | +0.143 | 1.000 | PASS, unchanged |
| CH | +0.060 | 1.000 | +0.063 | 1.000 | PASS, unchanged |

Every verdict holds, so composite_eligible and display_eligible in the Pitching+
contract do not move, and the provisional sinker ruling still applies (the gate still
fails it). The old-criterion rerun reproduces the 2026-09-11 gate log exactly.

Pitching+ contract (coach_pitching_plus_weights.py). The old-criterion rerun matches the
published v4 contract to the last digit. Under the adopted criterion the display weights
move by at most 0.030 (FF), 0.067 (SI), 0.022 (FC), 0.059 (SL), 0.037 (CB) and 0.042 (CH)
anywhere on the n grid, always toward Stuff+. Four-seam at 250
pitches goes from stuff 0.39 / location 0.45 / results 0.16 to 0.41 / 0.43 / 0.16. The
pitch count at which results reach a quarter of the weight rises for FF (750 to 1000), SI
(60 to 80) and SL (120 to 175). The closed-form blend R^2 rises for every type, by 0.006
to 0.023 at the pitch counts checked (40 to 1000). The contract's generated block now records which criterion built it
("criterion": "global"). CONTRACT_VERSION is unchanged because no field changed meaning.

Shipped scores, not republished. 08_staff_scores.py and 14_pitcher_pages.py were rerun on
the publish build (all levels, 2025 and 2026, the cache behind the 2026-09-10 bundle) with
each criterion. Stuff+ and Location+ are identical. On the 18-arm DEL_BLU staff board,
Adj Results moves by 1.6 points on average (max 5.6) on the 100 +/- 15 scale with Spearman
0.988 old vs new; Pitch+ moves 1.0 on average (max 3.6), Spearman 0.997, and four arms
swap one place on the board. The Runs Allowed and Expected Runs Allowed ladder rungs move
about 0.5 points because they ride Adj Results' population mean and sd. On the pitcher
pages, per-type Adj Results moves 1.6 to 2.0 points on average for FF, SL, CB, CH and FC
and 3.5 for the sinker (max 7.0), Spearman 0.90 to 1.00 within type. Publishing is Jack's
call; nothing was uploaded.

Caveats. The evidence above is D1. The publish build includes D2, D3, JUCO and NAIA, and
there the conference graph is not fully connected: 3 components in 2025 and 2 in 2026,
with some lower-level conferences playing no cross-conference games in the feed. For
those conferences the split between hitting and pitching strength is not identified, and
the conference penalty splits the difference toward the global mean. That affects only
non-D1 rows and, through them, the all-levels display scale. Scripts 12 and 13 and every
dated section above this one were measured on the within-league criterion and were not
rerun. The sinker's blind 2026 to 2027 read was locked as "same gate" when the gate used
the within-league criterion; which criterion that read uses needs a ruling before the
data arrives, and the clean option is to report both with the pre-registered one
deciding. The BATTER_K sweep was run on the within-league version and was not repeated.
The prose of the section above quotes 2,135 / 2,463 cross-conference games and 4,540 /
5,570 games; the script 17 log (before and after this change) prints 2,129 / 2,474 /
2,705 cross-conference games of 4,540 / 5,595 / 6,313. The table rows are unaffected.
Outputs and logs: workdir_adopt_adjt_global (Level II, outside the repo).
