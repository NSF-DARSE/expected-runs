# Component Model Framework (Draft for Discussion)

This is a rough framework for the next version of the Stuff+ model. The goal is unchanged:
grade the physical quality of a pitch (velocity, movement, release) in a way that predicts a
pitcher's future effectiveness. What changes is how we get there. Instead of regressing run
value directly on pitch characteristics, we model the components of a swing outcome separately
and combine them through the run expectancy framework we already built.

## Why change the approach

Two measurements from our own data (pitchers with 100+ four-seam fastballs in both 2024 and
2025) drive this redesign:

1. A pitcher's average run value allowed barely repeats year to year (correlation ~0.17).
   Single-pitch run value is dominated by the batter, the defense, and batted-ball luck.
   A regression trained directly on it mostly fits noise, which is why v2 R-squared was
   near zero and rankings were unstable.
2. The physical inputs repeat strongly (correlations ~0.84 to 0.97 for velocity, movement,
   spin). The stable signal lives in the pitch characteristics, not the outcomes.

Whiff rate sits in between (~0.33 year over year), which makes it learnable. But a pure whiff
model undervalues pitchers who succeed through weak contact. That was the known flaw of the
v1 swing-and-miss model, and it is the specific problem the contact-quality component fixes.

## Model structure

Scope for v1: pitches that were swung at, one model per pitch type, four-seam fastballs first.
Conditioning on swings keeps location and command out of the metric, which is intentional.

Every swing ends exactly one of three ways. The expected run value of a pitch, given a swing,
is the probability-weighted sum over those branches:

    E[RV | swing] = P(whiff)   * RV_whiff(count)
                  + P(foul)    * RV_foul(count)
                  + P(in play) * E[RV | contact]

Note that P(in play) is not 1 - P(whiff). Fouls are roughly 40% of all swings and must be
their own branch. Getting this wrong overweights the contact term by about a factor of two.

### Where each piece comes from

Branch probabilities: a model (multinomial, or two stacked binary models) that predicts
whiff / foul / in-play from stuff features. This is the natural extension of the existing
swing-only whiff model: same framing, same features, three classes instead of two.

Branch run values: computed from our data, not modeled. RV_whiff(count) is the average run
value of a swinging strike in each ball-strike count. It is small early in the count and large
in two-strike counts (where the whiff is a strikeout). RV_foul(count) matches RV_whiff below
two strikes and is roughly zero at two strikes (a foul cannot strike a batter out). The script
`compute_component_inputs.py` generates these tables from the final dataset.

Contact quality, E[RV | contact], has two stages:

1. A league-wide physical map from (exit speed, launch angle) to average run value, built by
   bucketing all batted balls in our data. This is the same construction as public
   "expected outcome" statistics. It is fixed and shared across all pitchers.
2. A stuff model that predicts, from pitch characteristics, the run value of the contact a
   pitch tends to give up (train it on the stage-1 value of each batted ball, not on the
   realized outcome, so defense and luck are averaged out).

### Data gap for contact quality

The current final dataset does not carry ExitSpeed or Angle through the pipeline, so stage 1
cannot be built from it yet. This draft updates REQUIRED_COLS in
`target_and_calculated_pipeline.py` to add ExitSpeed, Angle, Direction, Distance, HangTime,
GameID, and PitchUID (the last two make per-game validation and duplicate detection possible);
the dataset needs one regeneration before the contact model can be trained. Until then, TaggedHitType (ground ball / line drive /
fly ball / popup) works as a crude interim contact-quality label, and the input script computes
run values by hit type for that purpose.

## From pitch grades to pitcher scores

Average E[RV | swing] over a pitcher's pitches of the given type, then normalize to the
familiar 100 +/- 15 scale. Normalize against the full population of qualifying D1 pitchers in
our data, not a single roster, so that 100 means an average D1 four-seam fastball.

## How we judge success

Not by pitch-level R-squared. The evaluation protocol for every candidate model:

1. Out-of-time split: train on 2024, evaluate on 2025. No pitcher's season appears in both.
2. Reliability: correlate each pitcher's 2024 model score with his 2025 model score. A stuff
   metric should be far more stable than the outcomes it predicts (target: comparable to the
   ~0.9 stability of the physical inputs themselves).
3. Predictive validity: correlate 2024 model scores with actual 2025 results. The baselines to
   beat are the year-over-year correlations of the raw stats themselves: ~0.17 if predicting
   run value, ~0.33 if predicting whiff rate. A model that cannot beat the matching baseline
   adds nothing over last year's stat line.

## Suggested work plan

1. Regenerate the final dataset with ExitSpeed and Angle included (pipeline change in this
   draft).
2. Run `compute_component_inputs.py` to produce the branch probability and run value tables.
   Review the two-strike sanity checks it prints.
3. Build the three-class swing outcome model (extend the existing whiff model).
4. Build the (exit speed, launch angle) to run value map, then the contact-quality stuff model.
5. Combine per `combine.py` and run the evaluation protocol above.
6. Compare against the current models (whiff-only, direct run value regression, coach's
   weights) on the same protocol.

## Data note

This repository is public and the underlying TrackMan data is licensed. Do not commit
generated input tables, model outputs, or any derived data values. Scripts here read local
data paths supplied at run time and write outputs outside the repository by default.
