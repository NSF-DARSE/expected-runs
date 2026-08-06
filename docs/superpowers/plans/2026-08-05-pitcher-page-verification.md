# Pitcher page data layer: real-data verification

Date: 2026-08-05
Plan: `2026-08-05-pitcher-page-data-layer.md`, Task 9
Spec: `../specs/2026-08-05-pitcher-development-page-design.md`

One run, one caches pair. Per the replication discipline in `component_model/analysis/README.md`,
treat every number here as a first pass rather than a citable project finding until it is
reproduced.

## Run configuration

```
source : C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv
workdir: C:/Users/jackdav/stuffplus_replication/workdir_webapp
years  : 2025,2026   (so year role 2024 = actual 2025, role 2025 = actual 2026)
team   : DEL_BLU
level  : unset, matching how webapp_publisher/publish.py actually invokes the scorer
```

The workdir's `pitches_cache_2025_2026.parquet` was reused, so `load_pitches` did not re-read
the 1.4 GB source CSV.

## Per-type fit

| type | pitches | qualified pitchers (100+) |
|---|---|---|
| FF | 1,004,906 | 3,220 |
| Slider | 459,314 | 1,375 |
| ChangeUp | 240,952 | 600 |
| Curveball | 163,322 | 393 |
| Sinker | 145,188 | 428 |
| Cutter | 132,497 | 373 |
| Splitter | 14,752 | 19 |

All seven types cleared the two-pitcher minimum needed to set a display scale. Splitter is thin
at 19 qualifying pitchers; its scale is technically valid but rests on a small reference
population, which is worth remembering before reading a splitter grade closely.

18 DEL_BLU pitchers were emitted.

## Additivity: exact

The spec's central claim is that one affine transform at every level keeps the numbers additive.
On real data the mean of a pitcher's per-pitch four-seam grades equals his four-seam arsenal
grade with a worst-case violation of **0.000000000** across all 18 pitchers.

This is the property that makes the single scale honest rather than merely convenient: a coach
can add up what he sees.

## Decision 2's open question, answered

The spec deferred one measurement: whether per-pitch grades, calibrated against the same
population moments as pitcher grades, would spread too widely to present.

Measured four-seam per-pitch standard deviation, by pitcher: **8.9 to 16.0 points, median about
11.5**, against the pitcher-level scale's 15 points by construction. Observed range across all
pitchers roughly 20 to 147, with the bulk between 45 and 140.

Per-pitch spread is therefore **tighter** than pitcher-to-pitcher spread, not wider. Within-pitcher
variation in velocity and movement really is modest next to the differences between pitchers,
which is what the spec predicted but had not measured. Individual pitch grades sit comfortably on
the shared scale.

**The spec's fallback is not needed.** No band display, no rescaling, no second reference.

## Staff Board agreement

A pitcher must not read one fastball number on the Staff Board and a different one on his own
page. Comparing script 08's `staff_scores.json` against script 14's `pitcher_pages.json` from the
same source and workdir:

- 18 pitchers compared, 18 matched by name, none present on one side only
- **max absolute difference 0.05 points**, mean +0.004

The residual is rounding: script 08 stores its records at one decimal place, the pitcher page
carries the full float. Both scales derive from the same qualified population moments, so this is
agreement rather than coincidence.

## Payload sizes

| file | size |
|---|---|
| largest pitcher file | 192 KB (860 pitches across 5 types) |
| median pitcher file | 68 KB |
| smallest pitcher file | 7 KB |
| `location_maps.json` | 49 KB |
| `model_artifacts.json` | 8 KB |
| 22 files total | manifest, staff board, 2 shared, 18 pitchers |

Comfortably inside browser range. The frontend's planned lazy per-pitcher fetch with query-client
caching stands; no pagination and no combined-file fallback needed.

## Constraint checks on real output

- Every four-seam arsenal row carries a numeric Location+; no secondary type carries one. Zero
  violations across 18 pitchers.
- `percentiles` and `typical` are length 12 on every row, matching `featureOrder`.
- The manifest's pitcher index is keyed by TrackMan `PitcherId` (stable across seasons), not by the
  staff board's positional id.
- The publisher's dry run wrote all 22 files including the nested `pitchers/` directory, and both
  validators passed.

## What the real run caught that nothing else did

One bug, and it had survived three review passes and 116 unit tests.

`schema.py`'s `validate_pitcher_bundle` rejected any feature label identical to its field name,
treating that as proof of a missing label. But `Extension`'s correct plain-English label **is**
"Extension" — there is nothing to translate. The validator therefore refused a correct bundle and
aborted the publish:

```
ValueError: feature Extension has no plain-English label
```

It survived review because the test fixture's only feature was `SpinRate`, whose label
"Spin rate" differs from the field name, so the identity branch never executed. Fixed by requiring
the label to be present and non-empty and dropping the identity requirement; enforcing that a
label exists for every feature stays with `build_pitcher_bundle`, which owns the label map.

The general lesson matches the script-13 experience recorded in `component_model/analysis/METHODS.md`:
tests written from the same document as the code inherit that document's blind spots. Two earlier
defects in this plan had the same character — both sat in the seam between two components rather
than inside either one, and both were caught by review or real data rather than by tests.

## Deferred, not blocking

- `build_grids` still selects its training rows with `year == 2024` rather than `!= season_year`.
  Not broken today, because it reads the full frame rather than a season-filtered subset, but it is
  the same latent hazard that produced this plan's first Critical (a location map trained on an
  empty frame). Worth normalizing.
- Secondary-pitch sample floors remain unmeasured and reuse the four-seam floor as a conservative
  stand-in, as the spec states.
- Splitter's display scale rests on 19 qualifying pitchers.
