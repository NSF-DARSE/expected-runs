# Pitcher development page (frontend): real-data verification

Date: 2026-08-05
Plan: `2026-08-05-pitcher-page-frontend.md`, Task 11
Spec: `../specs/2026-08-05-pitcher-development-page-design.md`
Companion: `2026-08-05-pitcher-page-verification.md` (the data layer's pass)

One run, one cache pair. Per the replication discipline in
`component_model/analysis/README.md`, treat every number here as a first pass rather
than a citable project finding until it is reproduced.

## Run configuration

```
source : C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv   (1.4 GB, full D1)
workdir: C:/Users/jackdav/stuffplus_replication/workdir_webapp
years  : 2025,2026   (so season = 2026)
team   : DEL_BLU
command: python -m webapp_publisher.publish --dry-run
```

`pitches_cache_2025_2026.parquet` was reused, so `load_pitches` did not re-read the
source CSV. Both schema validators passed and the dry run wrote 22 files, including
the nested `pitchers/` directory.

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

18 DEL_BLU pitchers emitted, 63 arsenal rows across them. Splitter's scale still rests
on 19 qualifying pitchers, unchanged from the data layer's pass and still worth
remembering before reading a splitter grade closely.

## The two equalities, on real data

Both are equality properties of a linear model, not tolerances, and both hold exactly.

**Scale coherence.** For every one of the 63 arsenal rows, the mean of that pitch
type's per-pitch grades equals the row's published Stuff+. Worst violation across all
63: **0.000000000000**.

**Additivity.** For every arsenal row with a model, the browser's own attribution
formula — recomputed independently in the check script, not called through the
frontend — sums to `stuff - 100`. Worst violation: **0.000000000000**.

The second number is the one that matters for Decision 1. Attribution is derived in
Python and computed in TypeScript, and the plan's mitigation was a shared fixture
asserted in both languages. This run confirms the arithmetic also holds against the
real coefficients at every row the page will render, not only at the fixture's three
synthetic features.

## Staff Board agreement

A pitcher must not read one number on the board and a different one on his own page.

- 18 of 18 pitchers matched **by `pitcherId`**, the stable TrackMan key this plan
  added, rather than by name. The name join the publisher previously had to rely on is
  no longer on the read path.
- FF Stuff+: max absolute difference **0.050**, mean +0.027
- Location+: max absolute difference **0.057**, mean +0.024
- FF pitch counts agree **exactly** on all 18 rows

The residuals are rounding: script 08 stores its records at one decimal place while the
pitcher page carries the full float. Both scales derive from the same qualified
population moments, so this is agreement rather than coincidence.

## Constraint checks on real output

Eight seam checks, all passing. They target the space *between* components, because
every defect this plan found sat there rather than inside any one file.

1. Every staff-board row carries a `pitcherId`; all 18 are non-null and every one names
   a file that exists under `pitchers/`. Zero unlinked rows, so the null-`pitcherId`
   branch the frontend renders unlinked is untested against real data — see Deferred.
2. The manifest index and the files on disk agree in both directions; symmetric
   difference 0.
3. Scale coherence, above.
4. Additivity, above.
5. Every `typical`, `percentiles`, and per-pitch `f` array is length 12, matching
   `featureOrder`. Zero disagreements.
6. Location+ appears on all 18 FF rows and on **zero** secondary rows, ranging 54.4 to
   111.6 — inside the plausible display band and comfortably clear of the raw
   expected-run values (~0.00x) whose accidental emission the data layer's pass caught.
   No non-FF type carries `displayLocMu`/`displayLocSd`.
7. All 12 counts are present in `location_maps.json` at 120 cells each, and all 12 are
   thrown. See Finding C.
8. `recentChange` is null on 61 rows and numeric on 2. See Finding B.

## Findings

### A. `dataThrough` described the wrong population — fixed

`derive_data_through` took the maximum `Date` across the entire source CSV for the
season. That source is all of Division I, so the value described the D1 dataset rather
than the team the bundle is about.

Measured: the manifest said **2026-06-22** while DEL_BLU's last pitch was
**2026-05-16**, a five-week gap. The frontend renders that value in the shared header
on every page, pitcher page included, as "data through Jun 22, 2026". A coach reading
that beside his own staff's rows would reasonably conclude either that five weeks of
his games were missing from the page or that recent outings were included when they
were not.

Fixed in `e1d143a`: the derivation is now scoped to the bundle's team, still fails
loudly rather than falling back to today or to the population maximum, and is pinned by
a regression test in which the *other* team has later dates — the test that would have
caught this. Re-verified against the real source: it now returns **2026-05-16**,
matching the team's last outing date computed independently from the emitted bundle.

This one is worth naming precisely because nothing was broken in the ordinary sense.
Every component did what it said. The label was simply measured against a different
population than the rows beneath it, which is the failure mode the project's
"every number states what it is measured against" criterion exists to catch.

### B. The "Last 30 days" column is blank on 61 of 63 rows — NOT fixed, deliberately

The arsenal table's recent-change column renders blank unless both 30-day windows clear
the sample floor. On real data almost nothing clears it.

DEL_BLU's FF outings run 2026-02-13 to 2026-05-16, and the publisher's as-of date is the
team's last game, 2026-05-16. Median trailing-30-day volume per (pitcher, pitch type) is
**20 pitches**; median prior-30-day volume is **30**. The floor is **100**.

Only two rows in the entire staff clear it:

| type | season pitches | trailing 30d | prior 30d | change |
|---|---|---|---|---|
| FF | 405 | 131 | 138 | −3.4 |
| Sinker | 361 | 100 | 121 | −7.3 |

Floor sensitivity across all 63 rows: 100 → 2 filled, 60 → 6, 50 → 8, 40 → 13,
30 → 20, 25 → 22, 20 → 24.

So a coach opens the page and a column that promises to tell him whether a pitch is
moving tells him nothing on 97% of rows. That is a real design problem, and the blank is
at least honest — it reads as "not enough to say" rather than as a zero claiming "no
change", which is exactly why the plan forbade rendering 0 there.

**It is deliberately not fixed here.** The 100-pitch floor is script 06's measured value
for *season-level* four-seam Stuff+ reliability. Reusing it for a 30-day window is a
stand-in, and lowering it to make the column populate would be a modeling decision
disguised as a UI tweak — it needs a reliability measurement on 30-day windows, which is
follow-up work, not something to guess at while shipping a page. Changing it silently is
precisely how a score starts absorbing noise.

The honest interim options, in preference order: measure the 30-day-window floor and use
it; or widen the window to something a college season supports (a trailing N outings
rather than N days) and re-measure; or drop the column until one of those is done. All
three are decisions for the next pass.

### C. The pooled-fallback path is never exercised by real data

All 12 counts have their own map in this bundle, so `ZonePanel`'s fallback to the pooled
surface — and the "All counts pooled" copy that goes with it — is currently proven only
by a synthetic fixture built to force it. The fallback exists because a count with too
little training data can legitimately be absent, and that is a real possibility on a
smaller or earlier-season dataset, so the path is worth keeping. But nothing here
demonstrates it firing in production, and this record should not be read as though it
did.

## What the real run caught that the tests did not

One defect, Finding A, and it is the same shape as all four the data layer's pass found:
it sat in a seam between components rather than inside any of them, it was authored in a
plan rather than introduced by an implementer, and no unit test would have caught it
because every component was individually correct. `derive_data_through` correctly
returned the maximum date of what it was given; `build_bundle` correctly passed it
through; the header correctly rendered it. Only the meaning was wrong, and meaning lives
between components.

Findings B and C are not defects the tests missed. B is a measured consequence of a
documented stand-in, surfaced by putting the page in front of real volumes for the first
time. C is an honest statement of what this run does and does not demonstrate.

## Not done: visual and coach review

The plan's Task 11 also called for building `dist/`, viewing the page against this
bundle, and putting it in front of the `consumer-coach-baseball` reviewer. **Neither was
done.** The Playwright MCP server disconnected partway through this session, so there is
no way to render the page, capture screenshots, or drive the interactions from here.

This matters, and the gap should not be papered over: everything above is a check on
*numbers*, and none of it establishes that the page is legible to a coach reading it
aloud with a player present, which is the spec's actual success criterion. Layout,
label wording, chart readability, and whether the arsenal table's mostly-empty
recent-change column reads as broken rather than as honest are all unverified.

What is needed before this goes in front of the staff:

```bash
# frontend repo; the prebuild App Insights guard now blocks a bare local build
SKIP_APPINSIGHTS_CHECK=1 npm run build
# then serve dist/ with the dry-run bundle mounted at /api/bundle/
```

Then walk the checks the plan lists (usage summing to 100%, the fastball row's
Location+ against every secondary row's em-dash, a blank recent-change rendering blank
rather than 0, the waterfall total matching Stuff+ − 100, click-to-rebaseline dropping
percentiles, the count selector moving both dots and surface, a secondary type losing
the surface, a thin type saying so), and run the coach review.

## Deferred

- **The 30-day sample floor is unmeasured** and is currently script 06's season-level
  number. Finding B.
- **Secondary-pitch season floors remain unmeasured** and reuse the four-seam 100 as a
  conservative stand-in, as the spec states.
- **Splitter's display scale rests on 19 qualifying pitchers**, and the page does not
  currently say so. `nQualified` already ships in the model artifact, so a per-type note
  is the cheap fix.
- **The unlinked-pitcher branch is untested against real data** — all 18 rows linked
  here. It is covered by fixture only.
- **The pooled-fallback path is untested against real data.** Finding C.
- ~~`build_grids` still selects training rows with `year == 2024`.~~ **Struck: this was
  false.** `14_pitcher_pages.py:146` already reads
  `train[(train["year"] != SEASON_ROLE_YEAR) & train["xT"].notna()]`, and has since
  before this branch. The item was copied verbatim out of the data layer's deferred list
  without being checked against the code, which is the same failure the data layer's own
  record had to correct once — a note asserting a defect that no longer exists invites
  someone to "fix" working code. Verified by reading the line.
- **The managed Functions still have no telemetry path.** Adding
  `APPLICATIONINSIGHTS_CONNECTION_STRING` as a Static Web App setting is the last item of
  the plan's Task 10 and was deliberately left for the repo owner: on some CLI versions
  `az staticwebapp appsettings set` replaces the entire setting collection, and dropping
  `STORAGE_CONNECTION_STRING` would break the live bundle API. Capture the existing list
  first, re-supply every name, then verify `STORAGE_CONNECTION_STRING` and
  `BUNDLE_CONTAINER` survived.
