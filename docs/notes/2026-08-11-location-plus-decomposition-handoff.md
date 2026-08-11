# Handoff: Location+ sub-aspect decomposition (2026-08-11)

Design conversation only. No code was written, no analysis was run, nothing was
modified outside this file.

## State

Branch `component-model-framework`. This note is the only thing this session
committed.

The working tree was already dirty when this session started and **none of it is
this session's work** — do not commit it on the assumption that it is:

```
?? -                                                  (a file literally named "-")
?? .vscode/
?? CLAUDE.md                                           (untracked, though it is the project instructions file)
?? Coach_Linear_Regression_Model.xlsx
?? Trackman_-_License_Agreement_(version_3)_(002)_-_TM_Edits.docx
?? University of Delaware Renewal Addendum 2026 (5 Year).docx
?? meeting-2026.07.30.vtt
?? share/
?? webapp_publisher/.env.example
```

Two of those are executed/negotiated contract documents and one is a meeting
recording. Leave them alone; committing them is Jack's call, not a cleanup task.

## The question

Location+ is less reliable than Stuff+ overall (0.479 vs 0.908). Are some
*aspects* of it more reliable and more predictive than others, such that two
pitchers with the same Location+ built from different components should be
ranked differently? And how should that show up in the visuals?

## What is already measured (do not re-derive)

`component_model/portal/waste_and_feature_stability.py`, written up in
`component_model/RESULTS.md` lines 326-370. Both D1 pairs, pitchers with >=100 FF
in both years, n=649 / n=825. Validity sign: **positive = predicts worse
outcomes** (runs from the pitcher's perspective, lower is better).

| feature | rel p1/p2 | validity p1/p2 |
|---|---|---|
| sd_z | 0.60/0.56 | +0.17/+0.18 |
| sd_x | 0.57/0.52 | +0.21/+0.19 (collapses when waste is partialled) |
| mean_z | 0.54/0.53 | -0.14/-0.11 (partial\|waste -0.12/-0.13) |
| waste_pct | 0.41/0.34 | +0.26/+0.21 |
| heart_pct | 0.29/0.27 | -0.27/-0.15 (partial\|waste -0.15/-0.04, shaky) |
| chase_pct | 0.18/0.16 | +0.15/+0.14 |
| shadow_pct | 0.09/0.13 | -0.08/-0.14 (near-noise despite being 41% of pitches) |

Waste split by type, replicated in both pairs:

- waste-low: rel 0.37/0.34, validity +0.198/+0.198 — most validity-dense
- waste-horiz: rel 0.49/0.46, validity +0.173/+0.151
- waste-high: rel 0.45/0.39, validity +0.037/+0.001 — **stable but predicts nothing**

Missing high correlates with *better* stuff (-0.07/-0.06 with ridge_pred);
missing horizontally goes with worse stuff (+0.17/+0.19). So the per-pitch cost
of a high miss is real but offset at the trait level by who throws them.

**The headline: reliability and validity dissociate.** "Most reliable aspect" is
the wrong ranking rule. waste-high is the counterexample.

## What has NOT been tested

Everything above is marginal across the population. Nobody has asked whether,
**holding Location+ constant**, the composition of that score carries incremental
signal about next year. That is the open empirical question and it gates the
whole design. It may come back null, in which case the composite already absorbed
it and the display becomes explanatory only. Do not design the visual before
running it.

## Decisions made

- **Test first, then design.** Jack chose this over building a portal-ranking
  input or a player-development view directly. Do not skip to the visual.
- **Both audiences eventually,** portal ranking and UD player development, but
  the residual test decides whether composition is a ranking input at all.
- **For UD players the coaching question is a 2x2, not a ranking.** Jack's
  framing: some of what explains a bad season is descriptive but does not need
  fixing. Note the vocabulary correction agreed in-thread — a bad Location+ is
  never outcome luck, because he genuinely threw those pitches there. What varies
  is whether the *pattern* repeats.

  |  | valid | invalid |
  |---|---|---|
  | **reliable** | fix it (waste-low, waste-horiz, mean_z) | leave it (waste-high: a real habit that costs nothing) |
  | **unreliable** | it cost him, but one season can't attribute it to him | ignore (shadow%, chase%) |

- **Settled elsewhere, do not relitigate** (all in RESULTS.md): count-conditioned
  Location+ scores better but the entire gain is count *occupancy*, not better
  location measurement, so pitcher-level Location+ stays count-relative;
  Location+ is a fastball-only score and must not be surfaced for secondaries;
  equal-weight z blends beat OLS-fitted weights.

## In flight — the open question, unanswered

Jack was asked how to define "aspects" and had not answered when the session
ended. Three options were put to him, with (1) recommended:

1. **Exact additive decomposition of the map score by zone region.** Location+ is
   a mean of per-cell map values, so it partitions exactly:
   `Location+ = sum over regions of (share_region * mean_value_region)`, with
   heart / shadow / chase / waste, waste split low/horiz/high. Sums to the score
   with zero residual; each region splits further into a *how often* and a *how
   bad when it happens* term; each gets its own reliability and validity from the
   existing `component_model/analysis/reliability_curves.py` machinery.
2. **Trait panel** (sd_x, sd_z, mean_z, waste%, heart%…) — what the existing
   script measures and more coach-legible, but the traits are collinear and do
   not sum to Location+, so attribution would be regression-based and the panel
   could contradict the headline score.
3. **Per-pitch SHAP over the location map** — matches the Stuff+ explanation
   layer, but the map is a lookup table, not a fitted model, so this means
   fitting a surrogate model to explain a lookup.

Recommendation attached to (1): keep mean_z and sd_z as coaching context rather
than as score components, since (1)'s regions already absorb what sd_x proxies.

**Nothing here is approved. Get Jack's answer before writing any code.**

Also unaddressed: per-pitcher shrinkage. The reliabilities above are population
parameters, but how much to shrink an individual depends on his n. A pitcher with
40 FF and one with 400 should not get the same confidence on a waste-low read,
and `component_model/analysis/06_sample_floor.py` already covers the sample floor
for the composite.

## Where this would surface

`webapp_publisher/build_bundle.py` currently ships one `loc` value and a
`locFlag` per pitcher (lines 49 and 58). Any decomposition means a schema change
there and in `webapp_publisher/schema.py`, plus the frontend. Out of scope until
the residual test says there is something to show.

## Needs from Jack

1. Answer the aspects-definition question above (1, 2, or 3).
2. Confirm the residual test is worth running before any display work.

Nothing here is blocked on a credential, an account, or an external party.
