# Location+ composition: does the mix carry signal beyond the score?

**Date:** 2026-08-12
**Status:** design, pending user review
**Scope:** the residual test only. No display work, no schema change, no adoption.
**Supersedes the open question in:** `docs/notes/2026-08-11-location-plus-decomposition-handoff.md`

## Question

Location+ is less reliable than Stuff+ (0.479 vs 0.908, `RESULTS.md`). Holding a
pitcher's Location+ **constant**, does the *composition* of that score carry
incremental information about next year?

Everything measured so far is marginal across the population: each location trait's
reliability and validity taken one at a time (`RESULTS.md` lines 326-370). Nobody has
asked whether two pitchers with the same Location+, built differently, should be
ranked differently. That is the question here, and it gates all display work.

It may come back null. If it does, the composite already absorbed the composition and
the decomposition becomes explanatory only. That branch is pre-committed below so it
does not get relitigated after the fact.

## Aspect definition (decided)

**Exact additive partition of the map score by zone region.** Rejected: a trait panel
(sd_x, sd_z, mean_z, waste%…) because the traits are collinear and do not sum to
Location+, so attribution would be regression-based and could contradict the headline
score; and surrogate SHAP over the location map, because the map is a lookup table and
a fitted surrogate would inject approximation error into the one score that currently
has none.

`mean_z` and `sd_z` stay as **coaching context, not score components.** The regions
already absorb what `sd_x` proxies (`RESULTS.md`: sd_x's validity collapses when waste
is partialled).

### The algebra

Pitcher Location+ is a mean of per-pitch map lookups, verified in source:
`ff["loc"] = pooled.apply(ff)` per pitch (`08_staff_scores.py:41`), aggregated
`loc=("loc","mean")` (`:47`). So for mutually exclusive, exhaustive regions `r`:

```
loc = sum_r ( share_r * val_r )      where c_r := share_r * val_r
```

with `share_r` the fraction of the pitcher's FF in region `r` and `val_r` the mean map
value of his pitches in that region. Zero residual, by construction.

`Loc100 = 100 + 15 * z_loc` (`:58`) is **affine** in `loc`, so the display score
decomposes too: region `r` contributes `(c_r - mean_pop(c_r)) * (-15 / sd_loc)`.

### Regions

Bands as defined in `waste_and_feature_stability.py` (themselves copied from script
02). Do not redefine them; import or copy verbatim so this is comparable to the
published marginal table.

| region | definition (`ax` = `abs(PlateLocSide)`, `z` = `PlateLocHeight`) |
|---|---|
| heart | `ax <= 0.558`, `1.83 <= z <= 3.17` |
| shadow | `ax <= 1.108`, `1.17 <= z <= 3.83`, not heart |
| chase | `ax <= 1.658`, `0.50 <= z <= 4.50`, not heart/shadow |
| waste-low | waste and `z < 0.5` |
| waste-high | waste and `z > 4.5` |
| waste-horiz | waste, not low/high (i.e. `ax > 1.658` within the height band) |

Two constructs per region: `share_r` is **occupancy** (how often he goes there),
`val_r` is **within-region precision** (how costly it is when he does). These are
different skills and the framework requires occupancy be labeled, not absorbed.

## Gating pre-checks (run before the residual test; either can reshape it)

**PC1 — exhaustivity / the seventh bucket.** In `features()`, `waste` excludes
non-finite locations and heart/shadow/chase are all False for NaN, so a NaN-location
pitch is in **no** region. But `PooledLocationMap.apply` fills those with
`self.fallback` (`fair_criterion.py:254`), so they still contribute to `loc`. Left
alone this produces a nonzero residual and destroys the one advantage this aspect
definition has.

Resolution: add an explicit **`unassigned`** bucket for non-finite-location pitches and
report its share. Assert `sum_r c_r == loc` to floating tolerance per pitcher; the
test does not proceed until that assertion passes. If the unassigned share is
negligible, drop those pitches from both `loc` and the partition and note it.

**PC2 — is `val_r` real or is it the fallback constant?** The map needs 50+ pitches per
0.25ft bin, falls back to 0.5ft bins, then to a single overall mean. Waste regions are
where bins are sparsest, so waste pitches disproportionately draw the coarse or the
flat fallback value. If `val_r` in waste-low is mostly `self.fallback`, the
share x value split is fictional there and all real variance is in `share_r`.

Resolution: report, per region, the share of pitches resolved at fine / coarse /
fallback tier, and the cross-pitcher SD of `val_r` against the cross-pitcher SD of
`share_r`. **If a region's `val_r` is effectively constant, collapse that region to
share-only** and say so, rather than shipping a precision term that cannot vary.

## The residual test

### Nested form (this is why the test is clean)

Because `loc = sum_r c_r` exactly, regressing next-year outcome on the composite *plus
all six/seven* `c_r` is **exactly singular** — not merely collinear. But the
composite-only model **is** the constrained model in which every region's contribution
carries an equal coefficient:

- **Constrained (composite):** `y2 ~ beta * sum_r c_r`
- **Free (composition):** `y2 ~ sum_r beta_r * c_r`

Composition carries signal iff the `beta_r` differ. 5 df (6 if `unassigned` survives
PC1). No reference category to drop, no log-ratio reparameterization, no residualizing
of compositional shares.

### Decision rule: out-of-sample, not the F-test

With n=649/825 and 5 df, an in-sample F will very likely clear significance while
adding nothing forward. The F-test is reported as a **diagnostic only**.

The decision is out-of-sample validity, both directions, with replication required:

1. Fit free-model coefficients on pair 1 (2024->2025). Freeze them. Apply to pair 2.
2. Compare validity against the composite on pair 2.
3. Reverse (fit on pair 2, apply to pair 1).
4. **Paired bootstrap on the difference**, both variants sharing resamples.
5. **Both directions must win.** Under ~1 SE is a tie, and a tie is a null.

### Target and conventions

- **Criterion: the fair criterion, year 2**, per `fair_criterion.py`. Same construction
  as the published validity column so results are comparable to `RESULTS.md`.
- **Sign convention, shipped wrong once.** `Target`/`xT`/`adjT`/`ridge_pred` and map
  values are expected runs from the pitcher's perspective, **lower = better**. A trait
  predicts *better* outcomes when its correlation with the criterion is **negative**.
  Display negation happens once, at the display layer. Read the `fair_criterion.py`
  module docstring before interpreting any sign.
- **Population: >=100 FF in both years** (`MIN_P = 100`), matching the published table.
  n=649 (2024->2025) and n=825 (2025->2026).
- **D1 only** for headline calls, consistent with prior work.
- Read source from `STUFFPLUS_DATA`, write only under `STUFFPLUS_WORKDIR`. Never commit
  derived values or per-pitcher output (Level II).

### Sample-size attenuation

The six `c_r` are estimated per pitcher with more noise than the composite is, and that
attenuation biases the test **toward** the composite. So report a **>=200 FF subset**
alongside the >=100 headline. A null at 100 that becomes a win at 200 is informative,
not contradictory.

Do not build new per-pitcher shrinkage machinery for this test.
`06_sample_floor.py` covers the composite's floor; a per-aspect shrinkage rule is a
follow-on and is only worth writing if the test comes back positive.

## Preregistered checks that can turn a "win" into a rejection

**C1 — Stuff+ confound. This is a kill condition, not a footnote.**

`RESULTS.md` explains waste-high's null validity by noting high-miss rate correlates
with *better* stuff (-0.07/-0.06 with `ridge_pred`) while horizontal-miss goes with
worse stuff (+0.17/+0.19). Region shares therefore demonstrably carry Stuff+
information. A free composition model can beat composite Location+ by partially
**rebuilding Stuff+**.

This matters more than mislabeling. `Pitch100` is an equal-weight blend of
`z_ridge + z_loc + z_adj` (`08_staff_scores.py:60`), so a Location+ that absorbs Stuff+
**double-counts stuff inside Pitching+** and silently reweights the blend away from
equal thirds, degrading Pitching+ precisely for pitchers whose stuff and location
diverge.

Threshold, fixed in advance: with `ridge_pred` added as a control, the gain must keep
its sign, retain **at least half its magnitude**, and still clear ~1 SE on the paired
bootstrap.

**Routing rule if it fails C1:** the finding is not discarded. Failing C1 means those
region shares carry stuff information `ridge_pred` currently misses. That is a
**Stuff+ feature lead**, filed against Stuff+, and Location+ keeps the composite. The
gain gets attributed to the construct that owns it, per the framework's decomposition
audit.

**C2 — coefficient pattern must match the marginal table.** If composition is real, the
free model should place a near-zero `beta` on waste-high and a large one on waste-low,
reproducing +0.04/+0.00 vs +0.198/+0.198. A win whose coefficient pattern *contradicts*
the marginal table is overfitting, not discovery, and is rejected.

**C3 — survives next to the results component**, per `FRAMEWORK.md`. Add `adjres` and
confirm the gain persists.

**C4 — Pitching+ gate.** Anything surviving C1-C3 at the Location+ level must be shown
to improve **`Pitch100` itself**, not just Location+ in isolation. A Location+ gain
that does not move the blend is not worth a schema change.

## Outcome branches (pre-committed)

| result | action |
|---|---|
| Wins both directions, survives C1-C4 | Adopt into Location+. Then, and only then, design the display and the `build_bundle.py` / `schema.py` change. |
| Wins but fails C1 | Do **not** touch Location+. File as a Stuff+ feature lead. |
| Wins C1 but fails C4 | Explanatory only. No schema change. |
| Tie / null (<~1 SE either direction) | Composition is explanatory only. Goes in the 2x2 coaching frame for UD player development, never becomes a portal ranking input. `build_bundle.py` and `schema.py` unchanged. |

The 2x2 coaching frame, for the null and explanatory-only branches (from the handoff;
note the vocabulary correction: a bad Location+ is never outcome luck, he genuinely
threw those pitches there, what varies is whether the *pattern* repeats):

|  | valid | invalid |
|---|---|---|
| **reliable** | fix it | leave it (a real habit that costs nothing) |
| **unreliable** | it cost him, but one season can't attribute it to him | ignore |

## Out of scope

- Any display or frontend work. `webapp_publisher/build_bundle.py` ships one `loc` and
  one `locFlag` per pitcher today (`:49`, `:58`); a decomposition means a schema change
  there, in `schema.py`, and in the frontend. **Not until the test says there is
  something to show,** and `webapp_publisher/` is claimed by the pitcher-page session.
- Per-aspect shrinkage rules (follow-on, positive result only).
- Secondaries. Location+ is fastball-only by prior decision and must not be surfaced
  for non-FF.
- Count-conditioning. Settled: the gain is count *occupancy*, not better location
  measurement, so pitcher-level Location+ stays count-relative.

## Settled elsewhere, do not relitigate

All in `RESULTS.md`: count-conditioned Location+ scores better but the entire gain is
occupancy; Location+ is fastball-only; equal-weight z blends beat OLS-fitted weights.

## Files

**Deliverable:** one new standalone script under `component_model/analysis/`. Compose
`fair_criterion.py` and the reliability machinery in `reliability_curves.py`; do not
modify either (both are fixed references).

**Hands off** (owned by concurrent sessions as of 2026-08-12): `arsenal.py`,
`14_pitcher_pages.py`, `15_recent_change_floor.py`, `webapp_publisher/`,
`coach_model_comparison.py`.

**Repo hygiene:** the main checkout's working tree carries untracked licensed and
Level II files, including two contract documents, a meeting recording, and a file
literally named `-`. This repo is public. Never `git add .`.
