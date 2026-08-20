# Why hand-assigned weights tie a fitted ridge on four-seam Stuff+

Recorded 2026-08-14 so this is not relitigated. A domain expert built a Stuff+
scorecard by assigning linear weights to TrackMan features by judgment, with no
fitting. We compared it to our per-pitch-type Ridge on four-seam fastballs and could
not separate them. **This was the expected result, not a failure of either model, and
not evidence that fitting is pointless.**

## What was measured

- The two scores correlate **r = 0.73 to 0.75** at the pitcher-season level.
- Across seven criterion variants (next-year RA9; next-year four-seam `Target`, `xT`,
  and `adjT`; two sample floors), the sign of the difference changed four times and
  **no comparison cleared the project's P >= 0.95 bar in either direction**.
- Power required to resolve a gap that size: **~9,100 pitcher-seasons**. Available with
  both year pairs pooled: **~940**. Off by an order of magnitude.
- The criterion choice does reliably move the comparison (difference-of-differences on
  shared resamples, P = 0.994 at the 51-pitch floor and P = 0.972 at 100), decomposing
  to roughly three quarters batted-ball luck and one quarter opponent/league strength.
  So the flip is real while neither model is ever established as better. Both halves of
  that sentence have to be reported together; either alone misleads.

## Why it should have been expected

Two nearly-linear functions of the same handful of physical features cannot be
separated by a noisy criterion at this sample size. Almost all of the predictive power
in a linear scorecard comes from **which features enter and in which direction**, not
from the precision of the coefficients. This is the "improper linear models" result
(Dawes 1979): expert-assigned unit or judgment weights on correctly-chosen predictors
perform close to optimally-fitted weights, and the gap shrinks further as the criterion
gets noisier. Our ridge and the hand card are drawing on largely the same feature set,
so they are largely the same measurement.

Corollary that matters for how we spend effort: **fitting buys much less on a single
component than adding a component the other model does not have at all.** Location has
no representation in a stuff-only scorecard, and that is a structural difference rather
than a coefficient difference, so it is testable at this sample size where the
stuff-vs-stuff seam is not.

## What was ruled out along the way

- **In-sample inflation of our ridge.** Hypothesised, then measured: **-0.001**. Ridge
  at alpha=10 over ~700k pitches does not meaningfully overfit its training season.
  Reason about the regularisation before blaming the split.
- **Controlling for RA9 while scoring a results-derived measure.** Doing so drove the
  adjusted-results component to an apparent -0.22 ("negative skill"), which is a
  suppression artifact: adjusted results *is* de-noised RA9. Quality scores and
  results-containing scores need different control sets and must never be described as
  measuring the same thing.
- **Joint models over collinear scores.** Coefficients across three scores correlated
  0.70-0.94 were read as an ordering; they are not identified and the reading was
  withdrawn.

## What survived

- Our Stuff+ **clearly beats plain average velocity** on the fastball-only luck-adjusted
  criterion (P = 0.001 at n = 825, replicated at the 51-pitch floor). This is the one
  clean, replicated win in the exercise, and it is ours against a radar gun rather than
  ours against the hand card.
- **Location+ is the strongest single non-structural component** against actual next-year
  outcomes: spread +1.34 runs vs Stuff+'s +0.46, P = 0.999.
- Quality scores add roughly +0.35 to +0.39 runs per standard deviation **on top of** the
  prior box-score line. They do not beat the box score on their own.
- A recorded but un-actioned defect: our composite carries the **four-seam** adjusted
  results component, which is inert for next-year prediction (spread +0.04, r = 0.027),
  while season-wide results sort better than anything else we have. One sample is not
  enough to change the composite (composite vs Stuff+ is P = 0.214), so this is logged
  for a future look rather than fixed here.

## Method notes worth keeping

- Evaluate on year-over-year reliability plus predictive validity against a fixed fair
  criterion. Never pitch-level R-squared.
- Paired bootstrap on the **difference**, same resamples, terciles recomputed inside each
  resample so bands stay resample-relative. A gap under ~1 SE is a tie and gets reported
  as a tie.
- Fix the criterion before looking at any result, and state its limitation up front. A
  fastball-only criterion is better matched to a fastball score but is no longer neutral
  between our score and an external one; the RA9 version is neutral but diluted. Neither
  alone settles a comparison.
- `pd.qcut` assigns labels in **ascending** value order. On a higher-is-better score the
  first label is the worst third. Getting this backwards inverted a whole table once.
