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
