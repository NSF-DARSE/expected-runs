# Two-season Pitching+ forecast, 2026-09: ledger

Scope: does a returning pitcher's forecast get better when it uses their last two seasons instead
of one? Everything in "Pre-registration" was written and committed BEFORE any forecast was
correlated with the criterion. Results are appended below in later commits; the
pre-registration text is not edited after that commit.

Sign convention: adjT, ridge_pred, Location+ values and the criterion are expected run value
from the pitcher's perspective, LOWER = BETTER. A forecast is correctly oriented when it
correlates POSITIVELY with the criterion. None of these are "runs"; they are run value relative
to an average pitcher.

## Why this, why now

Script 12 (RESULTS.md, "Noise vs missing skill") found measurement noise is about 70% of a
single season's variance in the fair criterion, and concluded: "the larger lever is precision,
since noise is a function of sample size and can be attacked with more pitches, multi-season
pooling, or shrinkage rather than with new physics." Shrinkage by sample size is already in
Pitching+ (the contract's weights move with pitch count). Multi-season pooling has never been
built or read. Every grade on the pitcher page comes from one season, while a coach reading a
returning arm already looks at two.

Every other lever on the current season pairs has been read: four-seam Stuff+ features and
learners, four-seam Location+ conditioning (count, batter frame, pitch shape), the sinker's
features, pooling and target, and the secondary per-type Location+. This is a different axis:
the same models, more of each pitcher's own data.

Expectations, written before the read. Results need hundreds of pitches to stabilise and
Stuff+ about a dozen, so the gain should come almost entirely from pooling results (R1 or RH).
Pooling Stuff+ could cost a little, because a winter's velocity or shape change makes last
year's stuff stale while this year's is already precise. Script 12's lag-2 retention of 0.48
says the older season carries about half the signal, which argues for lambda 0.5 over 1. A
plausible gain on returning four-seam pitchers is +0.02 to +0.05 in r.

## Pre-registration (committed before any read)

### Harness and data

`component_model/analysis/coach_two_season_forecast.py` (tests:
`tests/test_two_season_forecast.py`). Caches read, never rebuilt: score build `workdir_ext3`
(2024 and 2025 D1), criterion build `workdir_ext3_crit` (2025 and 2026 D1, role-relabeled).
Environment as in the secondary-research ledger.

The pool and criterion are the frozen gate's pair C exactly: 15+ pitches and 10%+ share in
2025 (score build) and in 2026 (criterion build), criterion = 2026 mean adjT on the type. The
dry run reproduces the gate's n for every type: FF 2007, SL 1263, CB 301, CH 622, SI 272.

Components per pitcher-season:
- results: mean adjT over the type's graded pitches (the gate's prior);
- Stuff+: mean ridge_pred from the model that ships for the type (`fc.ridge_for_group`,
  trained on 2024 score-build rows; its 2024 predictions are in-sample, measured inflation
  -0.001);
- Location+ (four-seam only): the shipped pooled catcher-frame map on xT trained on 2024; 2024
  values are cross-fitted over 5 pitcher folds (seed 20260924).

A pitcher is **returning** if he has 15+ graded pitches of the type in 2024. Dry run, returning
share of each pool: FF 646 (32%), SL 362 (29%), CB 60 (20%), CH 161 (26%), SI 49 (18%).

### Forecasts

The incumbent **ONE** is the equal-z blend RESULTS.md validated (scripts 05/07): z(results) +
z(Stuff+) + z(Location+) for the four-seam, z(results) + z(Stuff+) for the other types, all
from 2025. A two-season component is the pitch-weighted mean of 2025 and 2024 with the older
season discounted by lambda, (n25 x25 + lambda n24 x24) / (n25 + lambda n24); a pitcher without
a qualifying 2024 keeps their 2025 value. Four fixed candidates, nothing fitted:

| candidate | pooled components | lambda |
|---|---|---|
| P1 | all | 1 |
| PH | all | 0.5 |
| R1 | results only | 1 |
| RH | results only | 0.5 |

### The read

**Scope.** Primary: **returning pitchers** of each type, the only ones whose forecast changes.
Secondary: the full pool. A scope with fewer than 100 pitchers is reported and not read, so
the returning scope is read for FF, SL and CH only; CB and SI are read on the full pool only.

**Procedure, one cross-fitted read per scope.** Pitchers are split into two fixed halves
(seed 20260925 permutation of sorted ids, blind to outcomes). Each half's two-season forecast
uses the candidate with the higher forecast r on the OTHER half, so no pitcher's own criterion
chooses their forecast. Delta = r(cross-fitted forecast, criterion) - r(ONE, criterion) over the
scope. 2000 bootstrap resamples of pitchers within each half (seed 20260817) repeat the choice
and the scoring, so the selection sits inside the uncertainty.

**Decision rule.** For a type and scope, the two-season forecast **improves** iff
P(Delta > 0) >= 0.95. The confirmatory tests are the three returning scopes (FF, SL, CH);
under a global null about a 14% chance that at least one clears 0.95. |Delta| under 1 SE is a
tie whatever P says.

**Audit, reported, no bar.** Do the physical grades still add once results cover two seasons?
r(z R_pooled + z Stuff+ [+ z Location+]) - r(R_pooled), with results pooled at lambda 1 and
physics from 2025, next to the one-season analogue, with P(> 0). If physics stops adding over
two seasons of results, that is a finding about what the grades are worth for returning
pitchers, and it would be reported as plainly as a gain.

**Reads.** 1 (every type, both scopes, one invocation). Cap 2: a second read only for a bug
found after the first, registered here as a new candidate first.

**Nothing ships from this pass.** An improvement is a recommendation to Jack: the publisher
would need the prior season's components, and the contract's pitch-count weights would be
re-derived on pooled pitch counts.

### Contamination, stated

- The one-season equal-z blend has been read on pair C for the four-seam (script 07 on the
  100+ panel; the shape ledger's three-way blend on this pool). ONE is the incumbent, not a
  test.
- No multi-season forecast of any kind has been read by any script.
- lambda 0.5 is motivated by script 12's lag-2 retention (0.48), which was estimated on the
  2024-2026 panel and so saw 2026 xT. lambda 1 uses no information from 2026 at all. Both are
  round numbers fixed here, not tuned.
- Before registering, the harness ran with `--dry` only: pool sizes, returning shares, and
  median pitch counts. No correlation with any outcome was computed.
