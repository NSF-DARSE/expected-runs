# Pitcher development page: coach-review prep — handoff

Written 2026-08-11. Covers two repos: `baseball-stuff-plus` (this one) and
`ud-athletics-baseball-pitching` (the React/TS frontend, `~/repos/ud-athletics-baseball-pitching`).

## Read this first: things deliberately left in an odd state

**1. `recentChange` ships from the publisher and is rendered nowhere.**
`ArsenalRow.recentChange` is still emitted, still typed, and no component reads
it. This is intentional, not an oversight. Measured against the real bundle, no
arsenal row's 30-day Stuff+ change clears two standard errors of the
detectability floor and only about a quarter clear one, so the column was
showing noise with a sign attached. Do not "fix" the missing column by putting
it back, and do not delete the field from the publisher to tidy up. It returns
only when the publisher can emit a per-row standard error so the display can
say which movements are readable. The reasoning is duplicated in the
`ArsenalTable` header comment and the `types.ts` field doc.

**2. The `explainable` fallback in `TraitPanel` never fires on real data.**
It looks like dead code and is not. It compares the trait bars' exact sum
against the gap between the two displayed grades and, past one point, refuses
to apportion. That threshold is principled: two independent roundings each move
at most half a point. Measured worst case against the real bundle is 0.490
across 63 arsenal rows and 0.976 across 6,787 pitch rows, so it currently never
trips. Its job is to catch a model artifact that disagrees with its own
published grade, which would otherwise be distributed invisibly across the
trait bars where nobody would find it. Deleting the branch removes the only
detector for that fault.

**3. App Insights is not wired on the SWA.** The org standard treats this as a
hard pre-deploy gate and the app is currently on the wrong side of it. Caveat
when fixing: setting the connection string on the Static Web App requires
re-supplying all four existing settings in the same call (`AAD_CLIENT_ID`,
`AAD_CLIENT_SECRET`, `BUNDLE_CONTAINER`, `STORAGE_CONNECTION_STRING`) or they
are wiped.

## Tree state

Nothing of this session's work is uncommitted. Both trees verified with
`git status --short`.

- `ud-athletics-baseball-pitching` on `main`: clean. **5 commits unpushed.**
- `baseball-stuff-plus` on `component-model-framework`: clean apart from
  pre-existing untracked files (see "Needs a human"). **3 commits unpushed.**

## What changed

`ud-athletics-baseball-pitching`, all on `main`, none pushed:

- `bb3e668` — pulled the "Last 30 days" column; removed the now-dead
  `formatChange`
- `2a3df87` — thin-sample rows no longer render a bold colored grade beside
  their own "too few pitches to read" caveat
- `8701f38` — run-value surface stands down while the dots carry color, so the
  zone is never showing two color scales at once
- `12ae98f` — the trait table's own arithmetic now closes (see decisions)
- `ae113ee` — chart axis labels

181 tests passing, `tsc -b` clean.

`baseball-stuff-plus`, on `component-model-framework`, none pushed:

- `c7435c3` — ignore `trackman_api/`
- `ef3ef0f` — ignore `*.png` and `.playwright-mcp/`
- `1a38113` — measure the recent-change detectability floor

The two ignore commits closed live Level II exposures. About 20 screenshots
rendering real pitcher names, and roughly 27,000 raw licensed TrackMan CSVs,
bat-tracking JSON, and positioning files, were sitting untracked and unignored.
Nothing was ever tracked, so nothing was published, but one `git add -A` would
have committed the licensed source data. Untracked count went from 27,633 to 11.

## Decisions settled — do not relitigate

- **The displayed trait total is defined by the two grades on screen**, not by
  the exact sum of the contributions, and the bars are apportioned to it by
  largest remainder. Rounding each bar independently produced a column adding
  to 28 beside a total reading 29, with 104 and 132 bracketing it. The number a
  coach can check by hand wins over the one that matches an unrounded value
  they cannot see.
- **Rejected: absorbing any discrepancy into the bars.** Apportioning without
  the one-point bound would hide an upstream artifact/grade mismatch. Past the
  bound the panel shows the model's own total and lets it visibly disagree.
- **Real velocity is not a model feature.** The 12 features use `EffectiveVelo`
  (perceived velo) plus `velocity_differential`, which is a difference of raw
  velos, so the level cancels. Nothing in the ridge sees `RelSpeed`.
  Consequence: real velo cannot be a trait row with a percentile and a Worth
  column, because it has no coefficient. Agreed direction is context beside the
  pitch type ("Fastball, 93.1 mph avg"). `RelSpeed` already flows through
  `python_files/target_and_calculated_pipeline.py`, so this is a publisher
  pass-through, not new math. Not built yet.
- **The which-way-helps phrase is stacked under the percentile, not inline.**
  Inline, the widest phrase set the column's width and pushed the zone panel in
  the next grid column off the right edge of the page.
- **The direction comes off the same coefficient that produces the points**, so
  the two can never drift into disagreement. `coef < 0` means more is better,
  because expected runs are pitcher's-perspective and lower is better.

## Approaches already tried and rejected

- **jsdom alone does not verify this page.** Two defects this session were
  invisible to a passing 182-test suite: the column-width overflow above, and
  the arithmetic not closing. Render in a real browser at 1440x900 before
  believing a layout or display-number change.
- **Sweeping all 18 pitchers via in-app SPA navigation to check the totals.**
  It read half-rendered DOM across route transitions and reported three false
  failures. The tell was that those same rows reported zero direction phrases.
  Replaced with a bundle-level check that has no race.
- **A test fixture that mutates `coef` but inherits the fixture's original
  `stuff`.** The artifact then contradicts its own published grade, the guard
  takes the fallback path, and the assertion silently tests the wrong branch.
  Fixture values have to be derived from the fixture's own model.

## Needs a human, not code

- **Push.** 8 commits across the two repos, none pushed.
- **Real velo:** build it? Direction is settled, work is not started.
- **App Insights connection string** on the SWA (see the four-settings caveat
  above).
- **Untracked files in the `baseball-stuff-plus` root** that are nobody's
  deliberate state and may want a home or a delete: two contract `.docx` files,
  `Coach_Linear_Regression_Model.xlsx`, `meeting-2026.07.30.vtt`, `share/`,
  `.vscode/`, `webapp_publisher/.env.example`, and a file literally named `-`.
  The `.env.example` is placeholders only and is safe to commit if wanted.
- **Tier 3, post-review:** LHH/RHH matchup split (needs per-type
  handedness-excluded grades on the model side, the same work that would
  restore the dropped toggle) and a "what to work on" summary.

## Known gaps, carried

- Secondary-pitch season floors are unmeasured.
- Splitter's scale rests on 19 qualifying pitchers and the page does not say so.
- The pitch dots in the zone panel are not keyboard reachable.
