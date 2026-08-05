# Pitcher development page — design

Date: 2026-08-05
Status: approved, ready for implementation planning
Supersedes nothing. Extends `2026-07-24-pitching-app-design.md` (this is that spec's "Pitcher page", rescoped around player development and designed against the existing Power BI app).

## Why

The pitching staff has a working six-page Power BI app (Strike Zone, Pitcher Summary, Stuff+ Timeline, Feature Timeline, Roster Leaderboard, Dictionary). The coach confirmed that the first four all earn their keep, but reading them requires hopping between tabs, holding filter state in your head, and translating raw TrackMan field names. The web app that replaces it currently has one page, the Staff Board, which is the Roster Leaderboard's successor.

This spec covers the first player-development deliverable: a single pitcher page a coach opens *with the player*, which answers three questions in one view. How does each pitch grade? Why does it grade that way? Is it moving?

It also covers routine data refresh and closes two monitoring gaps on the already-live app.

## Users and the job

Primary user is the pitching coach, with the player present. That has two consequences the design must honor.

The page is read aloud, so every label is plain English and every number states what it is measured against. Raw model field names (`effectivevelo`, `relheight`, `horzbreakdiff`) never appear.

The tone is objective and minimal. No generated narrative paragraphs, no praise, no prescription. The page reports what is measured, hedges what is thin, and stops. Where the model cannot support a claim, the page says so in as few words as possible rather than filling the space.

## Scope

In scope:

- One pitcher page covering the full arsenal, reached by clicking a pitcher on the existing Staff Board.
- Per-pitch-type grade, trait attribution, location-in-zone, and trend over time.
- Individual-pitch drill-in.
- New publisher stages producing per-pitcher bundles, the shared location maps, and the model artifact.
- A scheduled refresh that reads the game-file tree, rescores, rebuilds the bundle, and uploads.
- Application Insights wiring for both the frontend and the managed functions.

Out of scope, deferred to later specs: Usage Gap Board, Portal Board, Opponent Scouting, Pitch Design. The Dictionary does not become a page; definitions live in tooltips on the terms themselves, and any label that needs a glossary entry to be legible is treated as a label defect.

Also out of scope: hardening the TrackMan API client. That proceeds as an independent track (see Refresh architecture).

## Page anatomy

### Arsenal summary

One row per pitch type, sorted by usage descending. Columns: pitch type, usage share, Stuff+, pitch count, and recent change.

Recent change is defined explicitly: the pitch type's Stuff+ over the trailing 30 days of appearances minus its Stuff+ over the 30 days before that, in Stuff+ points. It renders only when both windows clear the sample floor, and is otherwise blank rather than zero, because a blank reads as "not enough to say" while a zero reads as "no change."

Location+ appears only on the four-seam row. Secondary rows show an em-dash with a tooltip explaining that secondary-pitch Location+ repeats year over year but does not predict next-year outcomes, so it is not surfaced. This follows `RESULTS.md` ("Location+ is a fastball score: do not surface it for secondary pitches") and the framework's one-construct-per-score rule.

Selecting a row drives the three panels below. Default selection is the highest-usage pitch type.

### Panel 1 — why it grades this way

Horizontal bars, one per trait, in Stuff+ points, sorted by absolute contribution. Plain labels throughout.

Each row also carries the trait's measured value and its percentile against the qualified population for that pitch type — the three columns the Power BI's attribute table showed, for the same reason. The contribution says how much the trait moves the grade; the value and percentile say whether the trait itself is any good. A coach needs both to have the conversation: "your extension is 7 feet, which is 85th percentile, and it is worth +9 points to you."

Two baselines, depending on what is selected:

- **Pitch-type average selected (default).** Contributions are measured against the qualified population mean. Reads as "why his fastball grades 124 rather than 100."
- **An individual pitch selected.** Contributions are measured against *his own* typical pitch of that type. Reads as "why this pitch differs from his usual one."

Both are exact, not approximate: ridge on standardized features is additive, so contributions sum to the score gap identically.

### Panel 2 — where it goes

A strike-zone scatter of his actual pitches of the selected type, with a count selector.

For four-seams, the scatter is drawn over the count-conditioned run-value map that `08_staff_scores.py` already emits as `grids`. This answers a real question: is he living in a part of the zone that is expensive *in this count*.

For secondary pitches, locations are plotted descriptively with no value surface underneath, labeled as showing where pitches went rather than what those locations are worth. Same reason as the arsenal table's dash.

Dots can optionally be colored by each pitch's Stuff+, as the existing Power BI does. That is worth keeping, but it needs labeling for what it actually shows. Stuff+ contains no location term — it is built from velocity, movement, spin, and release — so coloring by it does not tell you which locations are valuable. What it can show is whether his *shape* varies with where he is aiming, for instance losing perceived velo when reaching for a corner. That is a genuine development question, and a different one from what the value surface underneath answers. The two are labeled accordingly, so location value reads off the map and shape reads off the color.

Clicking a plotted pitch re-renders panel 1 for that pitch.

### Panel 3 — trend

Stuff+ by outing for the selected pitch type, plus trait lines over time.

The trait lines default to the two or three traits panel 1 identified as the largest contributors for this pitcher, with a control to show the remaining traits. This is the improvement over the Power BI's fixed six small multiples: the panels inform each other rather than each showing everything.

Trends below the sample floor for their pitch type are rendered as thin points without a connecting line and labeled as too few pitches to read, rather than being hidden or drawn as a confident line.

## Data contract

Three new bundle files, alongside the existing `manifest.json` and `staff_board.json`. All are served through the existing `/api/bundle/{*path}` proxy and need no API changes.

### Stable pitcher keys

`build_bundle.assign_ids` currently derives ids by index into the sorted unique-name list. Those ids shift whenever the roster changes, since inserting one name alphabetically renumbers everyone after it. That is acceptable for a row key inside a single bundle but not for a filename that persists across refreshes or appears in a URL.

Per-pitcher files are therefore keyed by TrackMan's `PitcherId`, which is already the grouping key in `fair_criterion.py` and is stable across seasons. The existing sequential `id` stays on the staff-board row so the live page keeps working; a `pitcherId` field is added alongside it.

### `pitchers/{pitcherId}.json`

```json
{
  "pitcherId": 1000123,
  "name": "Last, First",
  "hand": "R",
  "season": 2026,
  "arsenal": [
    {
      "type": "FourSeamFastBall",
      "label": "Fastball",
      "n": 412,
      "usage": 0.58,
      "stuff": 124,
      "loc": 103,
      "trendStuff": -6.2,
      "aboveFloor": true,
      "typical": [2350, 6.8, -13.7, 18.0, 89.2, 5.4, -1.1, 2.5, 1.2, -3.0, 0, 0]
    }
  ],
  "outings": [
    { "date": "2026-03-15", "type": "FourSeamFastBall", "n": 42, "stuff": 118 }
  ],
  "traits": [
    { "date": "2026-03-15", "type": "FourSeamFastBall", "f": [2340, 6.9, -14.1, 17.6, 89.0, 5.4, -1.1, 2.4, 1.1, -3.1, 0, 0] }
  ],
  "pitches": [
    { "d": "2026-03-15", "t": "FourSeamFastBall", "x": -0.42, "z": 2.31, "c": "0-2", "g": 131,
      "f": [2350, 6.8, -13.7, 18.0, 89.2, 5.4, -1.1, 2.5, 1.2, -3.0, 0, 0] }
  ]
}
```

`typical` and every `f` array are ordered per `model_artifacts.json`'s `featureOrder`. Short keys and a positional feature array keep the per-pitch payload small; a pitcher-season is on the order of 500 to 1500 pitches across all types, so a per-pitcher file lands in the low hundreds of kilobytes.

`g` is the pitch's own Stuff+, on the same scale as every other Stuff+ on the page (Decision 2). `c` is the 12-way count. `x`/`z` are `PlateLocSide`/`PlateLocHeight`.

### `location_maps.json`

The `grids` object `08_staff_scores.py` already produces, passed through: thirteen keys (`pooled` plus each of the twelve counts), each a flat list of 120 `{x, z, v}` cells on the fixed 10 × 12 grid. `v` keeps the pitcher's-perspective sign convention, unnegated; the frontend negates at the display layer only. Shared across all pitchers, so it ships once.

### `model_artifacts.json`

```json
{
  "featureOrder": ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak", "EffectiveVelo",
                   "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
                   "velocity_differential", "is_lhp", "is_lhb"],
  "labels": {
    "SpinRate": "Spin rate",
    "Extension": "Extension",
    "HorzBreak": "Horizontal break",
    "InducedVertBreak": "Vertical break",
    "EffectiveVelo": "Perceived velo",
    "RelHeight": "Release height",
    "RelSide": "Release side",
    "vertbreakdiff": "Vertical break vs his fastball",
    "horzbreakdiff": "Horizontal break vs his fastball",
    "velocity_differential": "Velo vs his fastball",
    "is_lhp": "Throws left",
    "is_lhb": "Batter hits left"
  },
  "byPitchType": {
    "FourSeamFastBall": {
      "coef": [], "scalerMean": [], "scalerScale": [],
      "populationMeanZ": [],
      "displayMu": 0.0,
      "displaySd": 0.0,
      "sampleFloor": 100
    }
  }
}
```

The four array fields each hold twelve entries, ordered per `featureOrder`.

The last two labels need care on a page a player reads. `is_lhp` and `is_lhb` are handedness terms inside the model, and their contribution is not something a pitcher can act on. The existing Staff Board already handles this with an "Include handedness impact" toggle, defaulting on; the pitcher page inherits that toggle and the matching `stuff_nohand` fields rather than inventing a second convention.

`coef`, `scalerMean`, and `scalerScale` come straight off the fitted pipeline. `populationMeanZ` is the qualified-population mean of standardized features (`Zq` in `08_staff_scores.py`), the baseline for the default waterfall. `displayMu` and `displaySd` are the population moments of `ridge_pred` behind the existing `100 + 15z` transform — the same pair used for pitcher-season Stuff+, reused unchanged at every level per Decision 2. `sampleFloor` is the minimum pitch count for a readable trend.

### Extension to `manifest.json`

Adds a pitcher index so the frontend can route without fetching every pitcher file: a list of `{pitcherId, name, hand}`.

## Modeling decisions and constraints

**Decision 1: attribution is computed in the browser.** The publisher ships coefficients, the scaler, and each pitch's raw feature values; the frontend derives contributions on demand.

The formula is the one already in `08_staff_scores.py`, unchanged:

```
z_i          = (value_i − scalerMean) / scalerScale
contribution = −15 × (z_i − z_baseline) × coef / displaySd
```

where `z_baseline` is `populationMeanZ` for the default view and the pitcher's own standardized typical values for a selected pitch. The leading minus sign is the display-layer negation of the lower-is-better convention, and the `15` and `displaySd` put contributions in Stuff+ points. Because the model is linear in standardized features, the contributions sum exactly to the difference in Stuff+ between the subject and its baseline.

This is exact rather than approximate, avoids shipping twelve precomputed numbers per pitch for no accuracy gain, and produces `model_artifacts.json`, which is the single artifact a future Pitch Design page requires.

**Decision 2: one Stuff+ scale everywhere, same model, same features.** A pitch, an outing, a pitch type, and a pitcher are all graded by the same ridge on the same twelve features, then put on the display scale by the same `100 + 15z` transform using the same population moments.

This works because the transform is affine, so it commutes with averaging. A pitcher's individual pitch grades average exactly to his pitch-type Stuff+, and those usage-weight up to his arsenal grade. The number means one thing wherever it appears, and a coach can check it by addition.

An earlier draft of this spec proposed a separately-named pitch-level scale, calibrated against the distribution of individual pitches rather than of pitchers. That was wrong, and the reasoning behind it was imported from the wrong metric. For `adjT`, a single pitch's value is overwhelmingly noise, so a per-pitch number genuinely is a different quantity from a season aggregate. Stuff+ is not that: it is a deterministic function of measured physical properties with no outcome luck in it, and a pitcher's own pitch-to-pitch variation in velocity and movement is modest next to the spread between pitchers. Two differently-named scales would have broken additivity and introduced the very confusion the separation was meant to avoid.

One quantity to measure during implementation, before this ships: the observed spread of individual pitch grades. Dividing a single pitch's deviation by the pitcher-level standard deviation is only presentable if within-pitcher variation is in fact modest, which is the expectation above but is not yet measured. If the resulting spread is wide enough to look broken, the response is to show the distribution honestly (for instance a band around the pitcher's typical value) rather than to rescale and reintroduce two references.

**Constraint: Location+ stays fastball-only**, per `RESULTS.md`. Applies to the arsenal table and to panel 2's value surface.

**Constraint: the fair criterion and the Stuff+ ridge are not modified.** Both are fixed references; changing them invalidates every published comparison. This spec only reads them.

**Constraint: sign convention.** `Target`, `ridge_pred`, `adjT`, and location-map `v` are expected runs from the pitcher's perspective, where lower is better. Negation to higher-is-better happens once, at the display layer. Scores arriving from `08_staff_scores.py` are already on the display scale and must not be re-flipped.

**Honest gap: measured sample floors exist for four-seams, not for secondaries.** Script 06 measured the four-seam floor. Secondary-pitch floors have not been measured. Until they are, secondary `sampleFloor` values use the four-seam floor as a conservative stand-in, and the field is documented as unmeasured for those types rather than presented as derived. Measuring them is follow-up work, not a blocker.

## Refresh architecture

Ingestion and publication stay decoupled, which is what lets the API land later without touching the app.

Ingestion writes the `year/month/day/CSV` game-file tree. Nothing downstream knows or cares whether that tree was filled by the FTP mirror or by `trackman_api/backfill.py --refresh`, because both write the same layout — a property already verified by `golden_diff.py`. The page therefore ships on the FTP-sourced data that exists today, and the API becomes a drop-in source once hardened.

There is a gap in the current chain worth naming, because it is real scope. `publish.py`'s `run_scorer()` shells out only to `08_staff_scores.py`, which reads a prebuilt `Final_Target_Calc_*.csv` via `STUFFPLUS_DATA`. Nothing in the scheduled path regenerates that CSV from the game tree — today it is produced by a separate manual run of `python_files/target_and_calculated_pipeline.py`. A refresh that picks up new games therefore needs that stage added to the chain, not merely a new bundle stage.

The full scheduled chain becomes: resolve new game files from the tree, run the target pipeline to refresh `Final_Target_Calc`, run the scorer, build all bundles, validate against schema, upload. This extends the existing `run_refresh.ps1`, which already has a bounded retry limit, exponential backoff, and an orchestration timeout. A stage failure fails the run loudly and leaves the previously published bundle in place, so the live page keeps serving the last good data rather than breaking.

The target-pipeline stage must use `resolve_latest_game_files` when collecting game files, so that games re-pulled into later day folders do not produce duplicate pitches. That resolution logic already exists and is tested; the requirement here is only that the scheduled path actually call it.

The API track, proceeding independently: merge `trackman-api-slice` onto current main (it needs the rebase for the `RunsRemaining` sign fix), resolve the egress-IP whitelist with the TrackMan rep, run a real end-to-end `backfill.py --refresh`, complete the historical backfill, and add tests. None of this gates the page.

## Production readiness

Two gaps on the already-live app, both org hard gates rather than improvements.

`src/services/appInsights.ts` exists but is imported nowhere, so the app has recorded zero telemetry since launch. It gets wired into `src/main.tsx`, with `setAuthenticatedUserContext` called after auth resolves.

The managed functions have no telemetry path at all, because `VITE_APPINSIGHTS_CONNECTION_STRING` is a build-time frontend variable. `APPLICATIONINSIGHTS_CONNECTION_STRING` gets added as an SWA app setting so function-side failures are visible. This is the direct reason the bundle API being dead for two weeks produced no signal anywhere.

## Testing

**Additivity is an equality test, not a tolerance test.** Trait contributions must sum to the score gap exactly, because the underlying model is linear in standardized features. Asserted for both baselines.

**One fixture, asserted on both sides.** Attribution is derived in Python but computed in the browser, so a shared fixture is asserted against in the Python suite and in the frontend suite. This is the test that catches the two implementations drifting apart, which is the main risk Decision 1 introduces.

**Scale coherence is also an equality test.** Because one affine transform is used at every level, the mean of a pitcher's per-pitch Stuff+ for a pitch type must equal that pitch type's published Stuff+. Asserting that catches any accidental reintroduction of a second reference.

Beyond that: unit tests for the trait percentile function, the per-outing and per-date aggregation, and the recent-change window (including that it stays blank below the floor); schema validation for every new bundle file, extending `webapp_publisher/schema.py`; a duplicate-pitch regression test on the refreshed target pipeline, since that bug has bitten before; component tests for each panel against synthetic fixtures; and a test that secondary pitch types never emit a Location+ value, since that constraint is a correctness property rather than a styling choice.

No test depends on live Azure or on network access.

**Level II discipline.** Real pitcher names appear in the published bundle, which is acceptable because the blob container is private and the app is auth-gated. Bundles are never committed to git, and test fixtures use synthetic names.

## Risks and open items

The browser-side attribution could drift from the Python derivation. Mitigated by the shared fixture test; if that test proves awkward to maintain, the fallback is precomputing attribution in the publisher at the cost of payload size.

Per-pitcher files scale with roster size, so a full staff means a dozen or so fetches if the coach browses several pitchers. Mitigated by fetching lazily per pitcher and caching in the existing query client. If this becomes slow, the fallback is one combined file for the staff.

Secondary-pitch sample floors are unmeasured, as noted above.

The spread of individual pitch grades under the single shared scale is not yet measured (Decision 2). If within-pitcher variation turns out to be larger than expected, individual pitch grades will range more widely than pitcher grades do. The committed response is to display that spread honestly rather than to rescale, since rescaling would reintroduce two references for one name.

## Success criteria

A coach opens a pitcher from the Staff Board and, without navigating anywhere else, can say what each pitch grades, which two or three physical traits drive that grade, whether the pitch is being located in expensive parts of the zone for the count, and whether any of it is moving. Every number on the page states what it is measured against. Anything too thin to read says so. The page refreshes on a schedule without manual steps, and a refresh failure leaves the last good data live.
