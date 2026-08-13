# Pitcher Development Page (Frontend) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the coach-facing pitcher development page in the React app — arsenal table, trait attribution waterfall, strike-zone scatter over the count-conditioned run-value surface, and a trend panel — against the bundle the data layer already ships, and close the two Application Insights gaps on the live app.

**Architecture:** The data layer (plan `2026-08-05-pitcher-page-data-layer.md`, complete and pushed) publishes `location_maps.json`, `model_artifacts.json`, and `pitchers/{pitcherId}.json` to the private blob container, served through the existing `/api/bundle/{*path}` managed function. This plan adds one small publisher change (a stable `pitcherId` on each staff-board row, so the board can link to a pitcher file) and everything on the frontend: three React Query hooks, two pure-logic modules (`attribution.ts`, `zone.ts`), one route, and four presentational panels. Trait attribution is computed in the browser from the shipped coefficients (spec Decision 1), so the arithmetic is duplicated across Python and TypeScript and is pinned by a fixture asserted in both languages.

**Tech Stack:** React 19, TypeScript 6, Vite 8, Tailwind 3, `@tanstack/react-query` 5, `react-router-dom` 7, `recharts` 3 (installed, currently unused — this plan is its first use), Vitest 4 + `@testing-library/react`, `lucide-react`. Publisher side: Python 3, pytest.

**Two repos.** Unless a task says otherwise, work happens in `C:\Users\jackdav\repos\ud-athletics-baseball-pitching` (branch `main`). Tasks 1 and 2 happen in `C:\Users\jackdav\repos\baseball-stuff-plus` (branch `component-model-framework`). Each task states its repo explicitly. Never mix repos inside one commit.

---

## Global Constraints

- **Sign convention.** Every Stuff+ / Location+ number arriving in the bundle is already on the 100 ± 15 display scale, higher = better. Do NOT negate it again. Location-map `v` values are the one exception: they are raw expected runs from the pitcher's perspective, LOWER = better, and the frontend negates them for color only.
- **One Stuff+ scale.** A pitch, an outing, a pitch type, and a pitcher are all on the same scale. Never rescale, never re-center, never introduce a second reference population. The mean of a pitch type's per-pitch `g` values equals that type's `stuff` exactly; a test asserts it.
- **Additivity is an equality assertion, not a tolerance assertion** on the attribution sum, in both languages. Floating-point comparison uses `toBeCloseTo(x, 10)` in Vitest and `pytest.approx(x, abs=1e-10)` in Python — that is the precision floor for "exact," not a modeling tolerance.
- **Location+ is fastball-only.** No Location+ value and no value surface for any pitch type other than `FF`. This is a correctness property with its own test, not styling.
- **Plain English only.** No raw model field name (`effectivevelo`, `relheight`, `horzbreakdiff`) ever renders. Labels come from `model_artifacts.json`'s `labels` map.
- **Objective, minimal copy.** No generated narrative, no praise, no prescription, as few words as possible. Every number states what it is measured against. Anything below its sample floor says so rather than being hidden or drawn confidently.
- **Level II data.** Never commit a bundle, a pitcher file, or a real pitcher name. All fixtures use synthetic names (`Test-Pitcher, Alpha` style, matching the existing suite) and synthetic ids (101, 102, 1000101…).
- **No AI attribution** in commit messages. No `Co-Authored-By`, no "Generated with".
- **`git commit` and `git push` are separate commands.** Never amend a pushed commit, never force-push.
- **Do not modify** `component_model/analysis/fair_criterion.py`, `component_model/analysis/08_staff_scores.py`, or `component_model/portal/build_portal_data.py`. They are fixed references.
- **Brand.** UD Blue `#00539f` is the primary data color, Cool Gray `#bdbdbd` for context, UD Gold `#ffd200` at most once per visual. No rainbow palettes. White backgrounds, minimal gridlines, direct labels over legends. Components use Tailwind theme keys (`text-ud-navy`), never raw hex, except inside SVG `fill`/`stroke` attributes where Tailwind classes do not apply.
- **Existing conventions to match:** relative imports only (no `@/` alias exists); `tabular-nums` on every numeric cell; `font-display` on headings; colocated `.test.ts`/`.test.tsx` beside the file under test; `vi.stubGlobal('fetch', …)` for data tests (no MSW); `fireEvent` for interaction (`@testing-library/user-event` is not installed). `strict` is NOT enabled in tsconfig — do not rely on strict-mode inference; annotate explicitly.

---

## The as-built data contract

The spec's JSON examples were illustrative. These are the field names and types the publisher **actually emits**, verified against a real run. Where this section and the spec disagree, this section governs.

### `manifest.json`

```json
{
  "built": "2026-08-05T14:02:11Z",
  "season": 2026,
  "dataThrough": "2026-06-22",
  "bundleVersion": "2026-08-05T14:02:11Z",
  "pitchers": [{ "pitcherId": 1000101, "name": "Test-Pitcher, Alpha", "hand": "R" }]
}
```

### `staff_board.json`

Unchanged from today except for the field Task 1 adds: each row in `pitchers[]` gains `"pitcherId": 1000101 | null`. Everything else (`id`, `name`, `hand`, `ff`, `stuff`, `loc`, `adjres`, `pitch`, `whiff`, `zone`, `heart`, `meanHeight`, `locFlag`, `stuffAttr`, `stuffNoHand`, `pitchNoHand`, `stuffAttrNoHand`) stays as-is.

### `model_artifacts.json`

```json
{
  "featureOrder": ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak", "EffectiveVelo",
                   "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
                   "velocity_differential", "is_lhp", "is_lhb"],
  "labels": { "SpinRate": "Spin rate", "...": "..." },
  "byPitchType": {
    "FF": {
      "coef": [12 floats], "scalerMean": [12 floats], "scalerScale": [12 floats],
      "populationMeanZ": [12 floats],
      "displayMu": -0.0231, "displaySd": 0.0142,
      "displayLocMu": -0.0061, "displayLocSd": 0.0038,
      "sampleFloor": 100, "nQualified": 3220
    }
  }
}
```

- **Pitch-type keys are `FF`, `Slider`, `ChangeUp`, `Curveball`, `Sinker`, `Cutter`, `Splitter`.** NOT `FourSeamFastBall`. The spec used the long form; the code uses the short form.
- `displayLocMu` / `displayLocSd` are `null` for every type except `FF`.
- A pitch type absent from `byPitchType` was skipped for having too few qualifying pitchers. The page must tolerate that.

### `pitchers/{pitcherId}.json`

```json
{
  "pitcherId": 1000101,
  "name": "Test-Pitcher, Alpha",
  "hand": "R",
  "season": 2026,
  "arsenal": [
    { "type": "FF", "label": "Fastball", "n": 412, "usage": 0.58,
      "stuff": 124.3, "loc": 103.1, "recentChange": -6.2, "aboveFloor": true,
      "typical": [12 floats], "percentiles": [12 ints] }
  ],
  "outings": [{ "date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.4 }],
  "pitches": [{ "d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31, "c": "0-2",
                "g": 131.2, "f": [12 floats] }]
}
```

- The field is **`recentChange`**, not `trendStuff`. It is `number | null`; `null` means at least one 30-day window was below the sample floor and the cell must render blank, never `0`.
- `loc` is `null` for every non-`FF` row.
- `usage` shares sum to 1 across the *included* types only (types below 25 pitches for that pitcher are omitted entirely).
- `arsenal` arrives sorted by `usage` descending.
- **There is no `traits` array.** The spec proposed one; the publisher does not emit it. Panel 3's trait lines are derived in the browser by averaging each pitch's `f` per date per type. This is exact, not an approximation, and saves duplicating the data.
- `c` is the 12-way count as `"balls-strikes"`: `"0-0"` … `"3-2"`.
- `x` is `PlateLocSide` in feet (catcher's view, positive = toward a right-handed batter's box), `z` is `PlateLocHeight` in feet.

### `location_maps.json`

```json
{ "pooled": [{ "x": -1.25, "z": 1.0, "v": -0.0184 }, "… 120 cells"], "0-0": ["…"], "3-2": ["…"] }
```

Thirteen keys: `pooled` plus each count present in the training data. Each is 120 cells on a fixed grid. **`x` and `z` are the lower-left corner of a 0.25 ft cell**, so a cell covers `[x, x+0.25] × [z, z+0.25]`. `x` runs −1.25 to 1.00, `z` runs 1.00 to 3.75. `v` is raw expected runs, pitcher's perspective, lower = better.

---

## Deviations from the spec, and why

Three, all deliberate. Each is called out again at the task that implements it.

1. **No `traits` array in the bundle** — derived in the browser from `pitches` instead. Same numbers, smaller payload.
2. **Pitch-type keys are short (`FF`)**, not long (`FourSeamFastBall`). Follows the code.
3. **The pitcher page does not carry the Staff Board's "Include handedness impact" toggle.** The spec (line 173) said it would, inheriting the `stuff_nohand` fields. Those fields do not exist per pitch type in the bundle, and they cannot be derived in the browser: `08_staff_scores.py` produces them by zeroing the handedness features and **re-standardizing against the population**, which needs the population. Subtracting the two handedness contributions in the browser gives a different number, so a toggle built that way would disagree with the Staff Board — the exact confusion the one-scale rule exists to prevent.

   What this plan does instead: every Stuff+ on the pitcher page includes handedness, matching the Staff Board's default (toggle on). The waterfall shows the two handedness terms collapsed into one grey row labeled `Handedness (context, not coachable)`, which keeps the contributions summing exactly to the gap. The page header states `Includes handedness impact`. A coach who switches the Staff Board toggle off and then opens a pitcher page will see the with-handedness number; the header line is what tells him so.

   Closing this properly is follow-up work in the data layer (emit per-type `stuffNoHand` and `populationMeanZNoHand`), recorded in the plan's Deferred section.

---

## File structure

**`baseball-stuff-plus`** (Tasks 1–2):

| Path | Responsibility |
|---|---|
| `webapp_publisher/publish.py` (modify) | Stamp `pitcherId` onto staff-board rows from the pitcher index |
| `webapp_publisher/build_pitcher_bundle.py` (modify) | New `stamp_pitcher_ids(bundle, pages)` helper |
| `webapp_publisher/schema.py` (modify) | Require `pitcherId` on staff-board rows |
| `webapp_publisher/tests/test_build_pitcher_bundle.py` (modify) | Tests for the stamp |
| `component_model/analysis/tests/fixtures/attribution_fixture.json` (create) | The canonical cross-language attribution fixture |
| `component_model/analysis/tests/test_attribution_fixture.py` (create) | Python side of the cross-language assertion |

**`ud-athletics-baseball-pitching`** (Tasks 3–11):

| Path | Responsibility |
|---|---|
| `src/lib/types.ts` (modify) | Add `PitcherPage`, `ModelArtifacts`, `LocationMaps`, arsenal/pitch/outing row types; extend `Manifest` and `PitcherRow` |
| `src/lib/attribution.ts` (create) | Pure attribution math: `contributions`, `standardize`, `waterfallRows` |
| `src/lib/derive.ts` (create) | Pure derivations: per-date trait series, outing series, thin-sample flags, formatting |
| `src/lib/zone.ts` (create) | Strike-zone SVG geometry and the value-surface color ramp |
| `src/lib/featureLabels.ts` (modify) | Align wording with the bundle's `labels`; stays the fallback |
| `src/hooks/usePitcherPage.ts` (create) | Lazy per-pitcher fetch + the two shared artifacts |
| `src/pages/PitcherPage.tsx` (create) | Route-level: header, arsenal table, selection state, the three panels |
| `src/components/pitcher/ArsenalTable.tsx` (create) | Arsenal rows, selection |
| `src/components/pitcher/TraitPanel.tsx` (create) | Panel 1 — attribution waterfall |
| `src/components/pitcher/ZonePanel.tsx` (create) | Panel 2 — scatter over the value surface, count selector |
| `src/components/pitcher/TrendPanel.tsx` (create) | Panel 3 — Stuff+ by outing plus trait lines |
| `src/components/ErrorBoundary.tsx` (create) | Catch render failures instead of a white screen |
| `src/services/appInsights.ts` (modify) | Add `trackPageView` helper |
| `src/main.tsx` (modify) | Import App Insights, resolve `/.auth/me`, wrap in the boundary |
| `src/test/fixtures/*.json` (create) | Synthetic bundle fixtures shared by component tests |

---

## Task 1: Stable pitcher key on the staff-board row

**Repo:** `baseball-stuff-plus`, branch `component-model-framework`.

**Why:** the Staff Board row is what the coach clicks, but its `id` is a positional index that shifts when the roster changes, and it is not the key the pitcher files are named by. The publisher already computes a `{pitcherId, name, hand}` index; this task joins it onto the board rows by name so the frontend can route without a second fetch or a fragile client-side join.

**Files:**
- Modify: `webapp_publisher/build_pitcher_bundle.py` (add `stamp_pitcher_ids`)
- Modify: `webapp_publisher/publish.py:124` area (call it)
- Modify: `webapp_publisher/schema.py:2-4` (`REQUIRED_ROW_KEYS`)
- Modify: `webapp_publisher/tests/test_build_pitcher_bundle.py`

**Interfaces:**
- Consumes: `pitcher_index(pages) -> list[{pitcherId, name, hand}]` (exists), `bundle["staff_board.json"]["pitchers"]` rows keyed by `name`.
- Produces: `stamp_pitcher_ids(bundle: dict, pages: dict) -> None`, mutating in place. Every staff-board row gains `pitcherId: int | None`.

- [ ] **Step 1: Write the failing tests**

Append to `webapp_publisher/tests/test_build_pitcher_bundle.py`:

```python
from webapp_publisher.build_pitcher_bundle import stamp_pitcher_ids


def _bundle(names):
    return {"staff_board.json": {"pitchers": [{"name": n, "pitch": 100.0} for n in names]}}


def test_stamp_matches_by_name():
    pages = {"pitchers": [{"pitcherId": 1000101, "name": "Test-Pitcher, Alpha", "hand": "R"},
                          {"pitcherId": 1000102, "name": "Test-Pitcher, Bravo", "hand": "L"}]}
    bundle = _bundle(["Test-Pitcher, Bravo", "Test-Pitcher, Alpha"])
    stamp_pitcher_ids(bundle, pages)
    got = {r["name"]: r["pitcherId"] for r in bundle["staff_board.json"]["pitchers"]}
    assert got == {"Test-Pitcher, Alpha": 1000101, "Test-Pitcher, Bravo": 1000102}


def test_stamp_leaves_none_when_no_pitcher_file():
    # A pitcher on the board with no graded arsenal has no pitcher file. The row
    # must still validate; the frontend renders it unlinked.
    pages = {"pitchers": [{"pitcherId": 1000101, "name": "Test-Pitcher, Alpha", "hand": "R"}]}
    bundle = _bundle(["Test-Pitcher, Alpha", "Test-Pitcher, Charlie"])
    stamp_pitcher_ids(bundle, pages)
    rows = {r["name"]: r["pitcherId"] for r in bundle["staff_board.json"]["pitchers"]}
    assert rows["Test-Pitcher, Charlie"] is None


def test_stamp_rejects_duplicate_names():
    # Two pitcher files claiming one board name means the name join is unsafe and
    # a coach could be routed to the wrong player. Fail loudly rather than pick one.
    pages = {"pitchers": [{"pitcherId": 1000101, "name": "Test-Pitcher, Alpha", "hand": "R"},
                          {"pitcherId": 1000199, "name": "Test-Pitcher, Alpha", "hand": "L"}]}
    with pytest.raises(ValueError, match="more than one pitcher file"):
        stamp_pitcher_ids(_bundle(["Test-Pitcher, Alpha"]), pages)
```

Make sure `import pytest` is present at the top of that test file; add it if not.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest webapp_publisher/tests/test_build_pitcher_bundle.py -v`
Expected: FAIL with `ImportError: cannot import name 'stamp_pitcher_ids'`.

- [ ] **Step 3: Implement**

Add to `webapp_publisher/build_pitcher_bundle.py`, after `pitcher_index`:

```python
def stamp_pitcher_ids(bundle: dict, pages: dict) -> None:
    """Join the stable TrackMan PitcherId onto each staff-board row, by name.

    The board's own `id` is a positional index into the sorted name list, so it
    shifts whenever the roster changes and cannot name a file or appear in a URL.
    Name is the only column the two sides share -- 08_staff_scores.py is a fixed
    reference and does not emit PitcherId -- so a duplicate name is a hard error
    rather than a coin flip that could route a coach to the wrong player.
    """
    by_name: dict[str, int] = {}
    for p in pages["pitchers"]:
        name = p["name"]
        if name in by_name:
            raise ValueError(f"more than one pitcher file claims the name {name!r}")
        by_name[name] = int(p["pitcherId"])
    for row in bundle["staff_board.json"]["pitchers"]:
        row["pitcherId"] = by_name.get(row["name"])
```

In `webapp_publisher/publish.py`, alongside the existing `bundle["manifest.json"]["pitchers"] = pitcher_index(pages)` line, add the stamp. Import it from the same module the index comes from:

```python
    bundle["manifest.json"]["pitchers"] = pitcher_index(pages)
    stamp_pitcher_ids(bundle, pages)
```

Both lines must stay **before** the `validate_bundle(bundle)` call, since validation now requires the field.

In `webapp_publisher/schema.py`, add `"pitcherId"` to `REQUIRED_ROW_KEYS`.

- [ ] **Step 4: Run the full publisher suite**

Run: `python -m pytest webapp_publisher/tests -v`
Expected: PASS. Existing `validate_bundle` tests that build board rows by hand will fail on the new required key — fix each by adding `"pitcherId": 1000101` (or `None`) to the fixture row, which is the correct change: the field is now part of the contract.

- [ ] **Step 5: Commit**

```bash
git add webapp_publisher/build_pitcher_bundle.py webapp_publisher/publish.py webapp_publisher/schema.py webapp_publisher/tests
git commit -m "Give the staff board a stable pitcher key to link on"
```

---

## Task 2: The canonical attribution fixture, asserted in Python

**Repo:** `baseball-stuff-plus`, branch `component-model-framework`.

**Why:** spec Decision 1 puts the attribution arithmetic in the browser while the derivation lives in Python. That duplication is the main risk the decision introduces. The mitigation is one fixture file, byte-identical in both repos, whose expected outputs are asserted by both suites. This task creates it and asserts the Python side; Task 4 copies it and asserts the TypeScript side.

The fixture uses **three synthetic features, not twelve.** It is testing the arithmetic, not the bundle shape, and three features with round numbers make a hand-checkable expectation. One coefficient is positive so the fixture exercises a negative contribution.

**Files:**
- Create: `component_model/analysis/tests/fixtures/attribution_fixture.json`
- Create: `component_model/analysis/tests/test_attribution_fixture.py`

**Interfaces:**
- Consumes: `arsenal.contributions(feature_values, scaler_mean, scaler_scale, coef, baseline_z, sd)`.
- Produces: the fixture file, whose exact bytes Task 4 copies into `ud-athletics-baseball-pitching/src/test/fixtures/attribution_fixture.json`.

- [ ] **Step 1: Write the fixture**

Create `component_model/analysis/tests/fixtures/attribution_fixture.json`. The numbers are chosen so every expected value is exact in binary floating point.

```json
{
  "_comment": "Cross-language attribution fixture. A byte-identical copy lives at ud-athletics-baseball-pitching/src/test/fixtures/attribution_fixture.json. Edit BOTH or the drift this fixture exists to catch goes uncaught. Synthetic 3-feature model; not a real pitch type.",
  "featureOrder": ["velo", "spin", "extension"],
  "labels": { "velo": "Velocity", "spin": "Spin rate", "extension": "Extension" },
  "model": {
    "coef": [0.01, -0.02, -0.03],
    "scalerMean": [2200.0, 6.0, 88.0],
    "scalerScale": [200.0, 0.5, 2.0],
    "populationMeanZ": [0.0, 0.0, 0.0],
    "displaySd": 0.05
  },
  "cases": [
    {
      "name": "pitch type vs population mean",
      "featureValues": [2400.0, 6.5, 90.0],
      "baselineZ": [0.0, 0.0, 0.0],
      "expectedZ": [1.0, 1.0, 1.0],
      "expectedContributions": [-3.0, 6.0, 9.0],
      "expectedSum": 12.0
    },
    {
      "name": "one pitch vs his own typical pitch",
      "featureValues": [2400.0, 6.5, 90.0],
      "typicalValues": [2300.0, 6.2, 89.0],
      "baselineZ": [0.5, 0.4, 0.5],
      "expectedZ": [1.0, 1.0, 1.0],
      "expectedContributions": [-1.5, 3.6, 4.5],
      "expectedSum": 6.6
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

Create `component_model/analysis/tests/test_attribution_fixture.py`:

```python
"""Python side of the cross-language attribution fixture.

The browser computes trait attribution from shipped coefficients (spec Decision 1),
so the same arithmetic exists twice. This fixture is the pin: a byte-identical copy
in the frontend repo is asserted against the same expected values by the Vitest
suite. If either implementation drifts, one of the two suites goes red.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import arsenal as ar

FIXTURE = Path(__file__).parent / "fixtures" / "attribution_fixture.json"


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text())


def test_standardization_matches_fixture(fx):
    m = fx["model"]
    for case in fx["cases"]:
        z = (np.array(case["featureValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
        assert z == pytest.approx(case["expectedZ"], abs=1e-12), case["name"]


def test_contributions_match_fixture(fx):
    m = fx["model"]
    for case in fx["cases"]:
        got = ar.contributions(case["featureValues"], m["scalerMean"], m["scalerScale"],
                               m["coef"], case["baselineZ"], m["displaySd"])
        assert list(got) == pytest.approx(case["expectedContributions"], abs=1e-10), case["name"]


def test_contributions_sum_to_the_gap_exactly(fx):
    """Additivity is an equality property of a linear model, not a tolerance."""
    m = fx["model"]
    for case in fx["cases"]:
        got = ar.contributions(case["featureValues"], m["scalerMean"], m["scalerScale"],
                               m["coef"], case["baselineZ"], m["displaySd"])
        assert float(got.sum()) == pytest.approx(case["expectedSum"], abs=1e-10), case["name"]


def test_sum_equals_the_display_gap_the_model_would_predict(fx):
    """The independent check: the summed contributions must equal the difference in
    to_display() between the subject's prediction and the baseline's, so the
    waterfall cannot disagree with the number above it."""
    m = fx["model"]
    coef, mu, sd = np.array(m["coef"]), 0.0, m["displaySd"]
    for case in fx["cases"]:
        z = (np.array(case["featureValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
        pred_subject = float(coef @ z)
        pred_baseline = float(coef @ np.array(case["baselineZ"]))
        gap = float(ar.to_display(pred_subject, mu, sd) - ar.to_display(pred_baseline, mu, sd))
        assert gap == pytest.approx(case["expectedSum"], abs=1e-10), case["name"]


def test_typical_values_standardize_to_the_second_case_baseline(fx):
    """The 'own typical pitch' baseline must be the standardized typical values,
    not a separately authored array. Catches a fixture that quietly disagrees
    with itself."""
    m = fx["model"]
    case = next(c for c in fx["cases"] if "typicalValues" in c)
    z = (np.array(case["typicalValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
    assert list(z) == pytest.approx(case["baselineZ"], abs=1e-12)
```

- [ ] **Step 3: Run it**

Run: `python -m pytest component_model/analysis/tests/test_attribution_fixture.py -v`
Expected: all five PASS. `arsenal.contributions` already exists and already implements this formula, so this task is pinning behavior rather than adding it. If any assertion fails, the fixture's arithmetic is wrong — recompute it by hand from `contribution = -15 * (z - baselineZ) * coef / displaySd`; do not change `arsenal.py`.

- [ ] **Step 4: Run the whole analysis suite**

Run: `python -m pytest component_model/analysis/tests -q`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/tests/fixtures/attribution_fixture.json component_model/analysis/tests/test_attribution_fixture.py
git commit -m "Pin the attribution arithmetic with a fixture the browser will share"
```

---

## Task 3: Bundle types and the pitcher-page hooks

**Repo:** `ud-athletics-baseball-pitching`, branch `main`. Every remaining task is in this repo.

**Files:**
- Modify: `src/lib/types.ts`
- Create: `src/hooks/usePitcherPage.ts`
- Create: `src/hooks/usePitcherPage.test.ts`
- Create: `src/test/fixtures/pitcherPage.ts`

**Interfaces:**
- Produces, in `src/lib/types.ts`: `PitcherIndexEntry`, `ArsenalRow`, `OutingRow`, `PitchRow`, `PitcherPage`, `TypeArtifact`, `ModelArtifacts`, `LocationCell`, `LocationMaps`; `Manifest` gains `pitchers`; `PitcherRow` gains `pitcherId`.
- Produces, in `src/hooks/usePitcherPage.ts`: `usePitcherPage(pitcherId: number | null)` returning `{ data: { page, model, maps } | undefined, isLoading, isError }`; and `getJson<T>(path: string): Promise<T>` exported for reuse.
- Produces, in `src/test/fixtures/pitcherPage.ts`: `samplePage`, `sampleModel`, `sampleMaps`, `sampleManifest`, `sampleBoard` — the synthetic fixtures every later task's component tests import.

- [ ] **Step 1: Add the types**

Append to `src/lib/types.ts` (keep the existing declarations untouched except the two noted additions):

```ts
export interface PitcherIndexEntry { pitcherId: number; name: string; hand: string }

/** Pitch-type keys as the publisher emits them. Short form, not `FourSeamFastBall`. */
export type PitchType = 'FF' | 'Slider' | 'ChangeUp' | 'Curveball' | 'Sinker' | 'Cutter' | 'Splitter';

export interface ArsenalRow {
  type: PitchType;
  label: string;
  n: number;
  usage: number;
  stuff: number;
  /** Location+ on the 100±15 scale. Null for every type except FF, by design. */
  loc: number | null;
  /** Trailing-30-day Stuff+ minus the 30 days before. Null = a window was below
   *  the sample floor. Render blank, never 0: a 0 claims "no change". */
  recentChange: number | null;
  aboveFloor: boolean;
  /** His typical value for each feature, ordered per ModelArtifacts.featureOrder. */
  typical: number[];
  /** Percentile of each typical value vs that pitch type's qualified population. */
  percentiles: number[];
}

export interface OutingRow { date: string; type: PitchType; n: number; stuff: number }

export interface PitchRow {
  d: string;
  t: PitchType;
  /** PlateLocSide, feet. */
  x: number;
  /** PlateLocHeight, feet. */
  z: number;
  /** 12-way count, "balls-strikes". */
  c: string;
  /** This pitch's own Stuff+, same scale as every other Stuff+ on the page. */
  g: number;
  f: number[];
}

export interface PitcherPage {
  pitcherId: number;
  name: string;
  hand: string;
  season: number;
  arsenal: ArsenalRow[];
  outings: OutingRow[];
  pitches: PitchRow[];
}

export interface TypeArtifact {
  coef: number[];
  scalerMean: number[];
  scalerScale: number[];
  populationMeanZ: number[];
  displayMu: number;
  displaySd: number;
  displayLocMu: number | null;
  displayLocSd: number | null;
  sampleFloor: number;
  nQualified: number;
}

export interface ModelArtifacts {
  featureOrder: string[];
  labels: Record<string, string>;
  /** A pitch type absent here was skipped for too few qualifying pitchers. */
  byPitchType: Partial<Record<PitchType, TypeArtifact>>;
}

/** x and z are the LOWER-LEFT corner of a 0.25 ft cell. v is raw expected runs,
 *  pitcher's perspective, LOWER = better. Negate for color only. */
export interface LocationCell { x: number; z: number; v: number }

/** Keys: 'pooled' plus each 12-way count present in training. */
export type LocationMaps = Record<string, LocationCell[]>;
```

Then modify the two existing interfaces:

```ts
export interface Manifest {
  built: string; season: number; dataThrough: string; bundleVersion: string;
  /** Added by the pitcher-page publisher stage. Absent on an older bundle. */
  pitchers?: PitcherIndexEntry[];
}
```

and add one field to `PitcherRow`:

```ts
  /** Stable TrackMan id; null when the pitcher has no graded arsenal, in which
   *  case the board row does not link anywhere. */
  pitcherId: number | null;
```

- [ ] **Step 2: Write the synthetic fixtures**

Create `src/test/fixtures/pitcherPage.ts`. Three features would not match the 12-feature bundle shape, so these fixtures use a **3-feature synthetic model** consistent throughout — every `typical`, `percentiles`, and `f` array is length 3, and `featureOrder` is length 3. Component tests do not care how many features there are, only that the arrays agree with `featureOrder`.

```ts
import type { LocationMaps, Manifest, ModelArtifacts, PitcherPage, StaffBoard } from '../../lib/types';

export const sampleModel: ModelArtifacts = {
  featureOrder: ['velo', 'spin', 'is_lhp'],
  labels: { velo: 'Velocity', spin: 'Spin rate', is_lhp: 'Throws left' },
  byPitchType: {
    FF: {
      coef: [-0.02, -0.01, 0.005],
      scalerMean: [88, 2200, 0.3], scalerScale: [2, 200, 0.5],
      populationMeanZ: [0, 0, 0],
      displayMu: 0, displaySd: 0.05,
      displayLocMu: -0.006, displayLocSd: 0.004,
      sampleFloor: 100, nQualified: 3220,
    },
    Slider: {
      coef: [-0.03, -0.01, 0.004],
      scalerMean: [80, 2400, 0.3], scalerScale: [2, 200, 0.5],
      populationMeanZ: [0, 0, 0],
      displayMu: 0, displaySd: 0.05,
      displayLocMu: null, displayLocSd: null,
      sampleFloor: 100, nQualified: 1375,
    },
  },
};

export const samplePage: PitcherPage = {
  pitcherId: 1000101, name: 'Test-Pitcher, Alpha', hand: 'R', season: 2026,
  arsenal: [
    { type: 'FF', label: 'Fastball', n: 300, usage: 0.6, stuff: 118, loc: 103,
      recentChange: -6.2, aboveFloor: true, typical: [90, 2300, 0], percentiles: [78, 61, 0] },
    { type: 'Slider', label: 'Slider', n: 200, usage: 0.4, stuff: 96, loc: null,
      recentChange: null, aboveFloor: false, typical: [82, 2500, 0], percentiles: [40, 55, 0] },
  ],
  outings: [
    { date: '2026-03-15', type: 'FF', n: 40, stuff: 121 },
    { date: '2026-03-22', type: 'FF', n: 38, stuff: 115 },
    { date: '2026-03-15', type: 'Slider', n: 12, stuff: 99 },
  ],
  pitches: [
    { d: '2026-03-15', t: 'FF', x: -0.4, z: 2.3, c: '0-0', g: 124, f: [91, 2320, 0] },
    { d: '2026-03-15', t: 'FF', x: 0.5, z: 3.1, c: '0-2', g: 112, f: [89, 2280, 0] },
    { d: '2026-03-22', t: 'FF', x: 0.1, z: 1.9, c: '1-1', g: 118, f: [90, 2300, 0] },
    { d: '2026-03-15', t: 'Slider', x: -0.7, z: 1.6, c: '0-2', g: 99, f: [82, 2500, 0] },
  ],
};

export const sampleMaps: LocationMaps = {
  pooled: [
    { x: -0.25, z: 2.0, v: -0.02 }, { x: 0.0, z: 2.0, v: 0.01 },
    { x: -0.25, z: 2.25, v: 0.0 }, { x: 0.0, z: 2.25, v: 0.03 },
  ],
  '0-2': [
    { x: -0.25, z: 2.0, v: -0.05 }, { x: 0.0, z: 2.0, v: -0.01 },
    { x: -0.25, z: 2.25, v: -0.02 }, { x: 0.0, z: 2.25, v: 0.02 },
  ],
};

export const sampleManifest: Manifest = {
  built: '2026-08-05T14:02:11Z', season: 2026, dataThrough: '2026-06-22',
  bundleVersion: '2026-08-05T14:02:11Z',
  pitchers: [{ pitcherId: 1000101, name: 'Test-Pitcher, Alpha', hand: 'R' }],
};

export const sampleBoard: StaffBoard = {
  population: 3220, team: 'DEL_BLU',
  pitchers: [
    { id: 1, pitcherId: 1000101, name: 'Test-Pitcher, Alpha', hand: 'R', ff: 300,
      stuff: 118, loc: 103, adjres: 104, pitch: 112, whiff: 0.24, zone: 0.51,
      heart: 0.22, meanHeight: 2.4, locFlag: '', stuffAttr: [['velo', 8]],
      stuffNoHand: 111, pitchNoHand: 107, stuffAttrNoHand: [['velo', 8]] },
    { id: 2, pitcherId: null, name: 'Test-Pitcher, Bravo', hand: 'L', ff: 40,
      stuff: 94, loc: 98, adjres: 97, pitch: 95, whiff: null, zone: 0.44,
      heart: 0.18, meanHeight: 2.6, locFlag: 'small sample', stuffAttr: [['velo', -4]],
      stuffNoHand: 96, pitchNoHand: 96, stuffAttrNoHand: [['velo', -4]] },
  ],
};
```

- [ ] **Step 3: Write the failing hook test**

Create `src/hooks/usePitcherPage.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import React from 'react';
import { usePitcherPage } from './usePitcherPage';
import { samplePage, sampleModel, sampleMaps } from '../test/fixtures/pitcherPage';

function bodyFor(url: string): unknown {
  if (url.includes('model_artifacts')) return sampleModel;
  if (url.includes('location_maps')) return sampleMaps;
  return samplePage;
}

const wrap = ({ children }: { children: React.ReactNode }) =>
  React.createElement(QueryClientProvider,
    { client: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }, children);

describe('usePitcherPage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(bodyFor(url)) } as Response)));
  });
  afterEach(() => vi.unstubAllGlobals());

  it('fetches the pitcher file plus both shared artifacts', async () => {
    const { result } = renderHook(() => usePitcherPage(1000101), { wrapper: wrap });
    await waitFor(() => expect(result.current.data).toBeDefined());
    expect(result.current.data!.page.name).toBe('Test-Pitcher, Alpha');
    expect(result.current.data!.model.featureOrder).toHaveLength(3);
    expect(Object.keys(result.current.data!.maps)).toContain('pooled');
    const urls = (fetch as unknown as ReturnType<typeof vi.fn>).mock.calls.map((c) => c[0]);
    expect(urls).toContain('/api/bundle/pitchers/1000101.json');
  });

  it('does not fetch at all for a null pitcher id', () => {
    renderHook(() => usePitcherPage(null), { wrapper: wrap });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('reports an error when the pitcher file is missing', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      Promise.resolve(url.includes('pitchers/')
        ? ({ ok: false, status: 404 } as Response)
        : ({ ok: true, json: () => Promise.resolve(bodyFor(url)) } as Response))));
    const { result } = renderHook(() => usePitcherPage(1000101), { wrapper: wrap });
    await waitFor(() => expect(result.current.isError).toBe(true));
  });
});
```

- [ ] **Step 4: Run it to verify it fails**

Run: `npx vitest run src/hooks/usePitcherPage.test.ts`
Expected: FAIL — the module does not exist.

- [ ] **Step 5: Implement the hook**

Create `src/hooks/usePitcherPage.ts`:

```ts
import { useQuery } from '@tanstack/react-query';
import type { LocationMaps, ModelArtifacts, PitcherPage } from '../lib/types';

export async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`/api/bundle/${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

/**
 * One pitcher's page data plus the two artifacts shared across all pitchers.
 *
 * The shared artifacts get their own long-lived query keys so browsing a second
 * pitcher refetches only that pitcher's file (the largest real one measured at
 * 192 KB, median 68 KB). Passing null disables the query, which is how the route
 * handles a board row with no graded arsenal.
 */
export function usePitcherPage(pitcherId: number | null) {
  return useQuery({
    queryKey: ['pitcherPage', pitcherId],
    enabled: pitcherId !== null,
    queryFn: async () => {
      const [page, model, maps] = await Promise.all([
        getJson<PitcherPage>(`pitchers/${pitcherId}.json`),
        getJson<ModelArtifacts>('model_artifacts.json'),
        getJson<LocationMaps>('location_maps.json'),
      ]);
      return { page, model, maps };
    },
  });
}
```

- [ ] **Step 6: Run tests and typecheck**

Run: `npx vitest run src/hooks/usePitcherPage.test.ts` — Expected: 3 PASS.
Run: `npx tsc -b` — Expected: no errors. The `StaffBoard` fixture now sets `pitcherId`, which the modified `PitcherRow` requires; if `tsc` complains about the existing `src/test/fixtures/*` used by `StaffBoard.test.tsx`, add `pitcherId` there too.
Run: `npx vitest run` — Expected: the whole suite passes.

- [ ] **Step 7: Commit**

```bash
git add src/lib/types.ts src/hooks/usePitcherPage.ts src/hooks/usePitcherPage.test.ts src/test/fixtures
git commit -m "Add the pitcher-page bundle types and its data hook"
```

---

## Task 4: Browser-side attribution

**Repo:** `ud-athletics-baseball-pitching`.

**Why:** spec Decision 1. This is the module the cross-language fixture pins.

**Files:**
- Create: `src/test/fixtures/attribution_fixture.json` (byte-identical copy of Task 2's file)
- Create: `src/lib/attribution.ts`
- Create: `src/lib/attribution.test.ts`
- Modify: `src/lib/featureLabels.ts`

**Interfaces:**
- Consumes: `TypeArtifact`, `ModelArtifacts`, `ArsenalRow`, `PitchRow` from `../lib/types`.
- Produces:
  - `standardize(values: number[], mean: number[], scale: number[]): number[]`
  - `contributions(values: number[], m: Pick<TypeArtifact,'scalerMean'|'scalerScale'|'coef'|'displaySd'>, baselineZ: number[]): number[]`
  - `HANDEDNESS_FEATURES: readonly string[]` — `['is_lhp', 'is_lhb']`
  - `WaterfallRow` — `{ key: string; label: string; points: number; value: number | null; percentile: number | null; grouped: boolean }`
  - `waterfallRows(args): WaterfallRow[]` — sorted by `|points|` descending, handedness collapsed into one trailing grouped row.

- [ ] **Step 1: Copy the fixture**

Copy `C:\Users\jackdav\repos\baseball-stuff-plus\component_model\analysis\tests\fixtures\attribution_fixture.json` to `src/test/fixtures/attribution_fixture.json` **without editing it**. The `_comment` field in it says both copies must change together; that is the point.

- [ ] **Step 2: Write the failing tests**

Create `src/lib/attribution.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import fixture from '../test/fixtures/attribution_fixture.json';
import { standardize, contributions, waterfallRows } from './attribution';
import { sampleModel, samplePage } from '../test/fixtures/pitcherPage';

interface Case {
  name: string; featureValues: number[]; baselineZ: number[];
  typicalValues?: number[]; expectedZ: number[];
  expectedContributions: number[]; expectedSum: number;
}
const cases = fixture.cases as Case[];
const m = fixture.model;

describe('attribution, against the shared cross-language fixture', () => {
  it.each(cases)('standardizes correctly: $name', (c) => {
    expect(standardize(c.featureValues, m.scalerMean, m.scalerScale))
      .toEqual(c.expectedZ);
  });

  it.each(cases)('reproduces the expected contributions: $name', (c) => {
    const got = contributions(c.featureValues, m, c.baselineZ);
    got.forEach((v, i) => expect(v).toBeCloseTo(c.expectedContributions[i], 10));
  });

  it.each(cases)('sums exactly to the score gap: $name', (c) => {
    const sum = contributions(c.featureValues, m, c.baselineZ).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(c.expectedSum, 10);
  });
});

describe('waterfallRows', () => {
  const ff = sampleModel.byPitchType.FF!;
  const row = samplePage.arsenal[0];

  it('sorts by absolute contribution, largest first, handedness last', () => {
    const rows = waterfallRows({
      values: row.typical, artifact: ff, baselineZ: ff.populationMeanZ,
      featureOrder: sampleModel.featureOrder, labels: sampleModel.labels,
      values2: row.typical, percentiles: row.percentiles,
    });
    const coachable = rows.filter((r) => !r.grouped);
    const mags = coachable.map((r) => Math.abs(r.points));
    expect([...mags].sort((a, b) => b - a)).toEqual(mags);
    expect(rows[rows.length - 1].grouped).toBe(true);
  });

  it('collapses every handedness feature into one grouped row', () => {
    const rows = waterfallRows({
      values: row.typical, artifact: ff, baselineZ: ff.populationMeanZ,
      featureOrder: sampleModel.featureOrder, labels: sampleModel.labels,
      values2: row.typical, percentiles: row.percentiles,
    });
    expect(rows.filter((r) => r.grouped)).toHaveLength(1);
    expect(rows.some((r) => r.key === 'is_lhp')).toBe(false);
    expect(rows.find((r) => r.grouped)!.label).toBe('Handedness (context, not coachable)');
  });

  it('keeps the total exact after grouping', () => {
    const rows = waterfallRows({
      values: row.typical, artifact: ff, baselineZ: ff.populationMeanZ,
      featureOrder: sampleModel.featureOrder, labels: sampleModel.labels,
      values2: row.typical, percentiles: row.percentiles,
    });
    const grouped = rows.reduce((a, r) => a + r.points, 0);
    const raw = contributions(row.typical, ff, ff.populationMeanZ).reduce((a, b) => a + b, 0);
    expect(grouped).toBeCloseTo(raw, 10);
  });

  it('omits percentiles when none are supplied, as for a single selected pitch', () => {
    const rows = waterfallRows({
      values: samplePage.pitches[0].f, artifact: ff,
      baselineZ: standardize(samplePage.arsenal[0].typical, ff.scalerMean, ff.scalerScale),
      featureOrder: sampleModel.featureOrder, labels: sampleModel.labels,
      values2: samplePage.pitches[0].f, percentiles: null,
    });
    expect(rows.every((r) => r.percentile === null)).toBe(true);
  });
});
```

- [ ] **Step 3: Run it to verify it fails**

Run: `npx vitest run src/lib/attribution.test.ts`
Expected: FAIL — `./attribution` does not exist.

- [ ] **Step 4: Implement**

Create `src/lib/attribution.ts`:

```ts
import type { TypeArtifact } from './types';

/** Display spread. The single negation of the lower-is-better convention lives in
 *  `contributions` below and in the publisher's `to_display`; nowhere else. */
const DISPLAY_SPREAD = 15;

/**
 * Handedness terms. They are inside the model and their contribution is real, but
 * a pitcher cannot act on them, so the waterfall groups them into one labeled row
 * instead of presenting them as development targets. Grouping preserves the exact
 * sum; dropping them would not.
 */
export const HANDEDNESS_FEATURES: readonly string[] = ['is_lhp', 'is_lhb'];

export const HANDEDNESS_LABEL = 'Handedness (context, not coachable)';

export function standardize(values: number[], mean: number[], scale: number[]): number[] {
  return values.map((v, i) => (v - mean[i]) / scale[i]);
}

type ContribModel = Pick<TypeArtifact, 'scalerMean' | 'scalerScale' | 'coef' | 'displaySd'>;

/**
 * Per-feature contribution to Stuff+, in display points.
 *
 * Mirrors arsenal.contributions in baseball-stuff-plus exactly; the shared
 * fixture at src/test/fixtures/attribution_fixture.json pins the two together.
 *
 *   z            = (value - scalerMean) / scalerScale
 *   contribution = -15 * (z - baselineZ) * coef / displaySd
 *
 * Because the ridge is linear in standardized features, these sum to the Stuff+
 * difference between subject and baseline exactly, not approximately.
 */
export function contributions(values: number[], m: ContribModel, baselineZ: number[]): number[] {
  if (m.displaySd <= 0) throw new Error(`display sd must be positive, got ${m.displaySd}`);
  const z = standardize(values, m.scalerMean, m.scalerScale);
  return z.map((zi, i) => (-DISPLAY_SPREAD * (zi - baselineZ[i]) * m.coef[i]) / m.displaySd);
}

export interface WaterfallRow {
  key: string;
  label: string;
  points: number;
  /** The measured trait value. Null on the grouped handedness row. */
  value: number | null;
  /** Percentile vs the qualified population, or null when not applicable —
   *  a single selected pitch has no pitch-level reference population. */
  percentile: number | null;
  grouped: boolean;
}

export function waterfallRows(args: {
  values: number[];
  artifact: TypeArtifact;
  baselineZ: number[];
  featureOrder: string[];
  labels: Record<string, string>;
  /** The values to display in the value column; same as `values` in every current
   *  caller, kept separate so a caller can show typical values beside a selected
   *  pitch's contributions if that is ever wanted. */
  values2: number[];
  percentiles: number[] | null;
}): WaterfallRow[] {
  const pts = contributions(args.values, args.artifact, args.baselineZ);
  const coachable: WaterfallRow[] = [];
  let handPoints = 0;
  let sawHandedness = false;

  args.featureOrder.forEach((key, i) => {
    if (HANDEDNESS_FEATURES.includes(key)) {
      handPoints += pts[i];
      sawHandedness = true;
      return;
    }
    coachable.push({
      key,
      label: args.labels[key] ?? key,
      points: pts[i],
      value: args.values2[i],
      percentile: args.percentiles ? args.percentiles[i] : null,
      grouped: false,
    });
  });

  coachable.sort((a, b) => Math.abs(b.points) - Math.abs(a.points));
  if (!sawHandedness) return coachable;
  return [...coachable, {
    key: 'handedness', label: HANDEDNESS_LABEL, points: handPoints,
    value: null, percentile: null, grouped: true,
  }];
}
```

- [ ] **Step 5: Align the label vocabulary**

The bundle now ships labels, and `src/lib/featureLabels.ts` disagrees with them in wording (it says `Ride (IVB)` where the bundle says `Vertical break`). Two vocabularies for one feature across two pages is a label defect. Make `featureLabels.ts` match the bundle exactly, and leave it in place as the Staff Board's fallback (the board's `stuffAttr` ships raw feature names and has no label map of its own).

Replace the map's entries with these, keyed as the existing file keys them (lowercased raw names) and preserving the file's existing `featureLabel(name)` fall-through:

```ts
const LABELS: Record<string, string> = {
  spinrate: 'Spin rate',
  extension: 'Extension',
  horzbreak: 'Horizontal break',
  inducedvertbreak: 'Vertical break',
  effectivevelo: 'Perceived velo',
  relheight: 'Release height',
  relside: 'Release side',
  vertbreakdiff: 'Vertical break vs his fastball',
  horzbreakdiff: 'Horizontal break vs his fastball',
  velocity_differential: 'Velo vs his fastball',
  is_lhp: 'Throws left',
  is_lhb: 'Batter hits left',
};
```

Keep whatever export signature the file already has. If its existing test asserts the old wording, update those assertions — the bundle is now authoritative.

- [ ] **Step 6: Run tests, lint, typecheck**

Run: `npx vitest run src/lib` — Expected: PASS.
Run: `npx vitest run` — Expected: PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/lib/attribution.ts src/lib/attribution.test.ts src/lib/featureLabels.ts src/test/fixtures/attribution_fixture.json
git commit -m "Compute trait attribution in the browser from the shipped coefficients"
```

---

## Task 5: Derivations and formatting

**Repo:** `ud-athletics-baseball-pitching`.

**Why:** the panels need per-date series, thin-sample flags, and consistent number formatting. Keeping them as pure functions with their own tests means the component tests can stay about rendering.

**Files:**
- Create: `src/lib/derive.ts`
- Create: `src/lib/derive.test.ts`

**Interfaces:**
- Produces:
  - `outingsFor(page: PitcherPage, type: PitchType): OutingRow[]` — sorted by date ascending
  - `pitchesFor(page: PitcherPage, type: PitchType, count?: string): PitchRow[]`
  - `traitSeries(page, type, featureIndex): { date: string; n: number; value: number }[]` — per-date mean of that feature, date ascending
  - `meanPitchGrade(page, type): number | null`
  - `formatPoints(n: number): string` — signed, zero decimals (`+9`, `-3`, `0`)
  - `formatChange(n: number | null): string` — same, but `''` for null
  - `formatUsage(share: number): string` — `'58%'`
  - `formatValue(n: number): string` — one decimal, or zero decimals when `|n| >= 100`
  - `thinLabel(n: number, floor: number): string | null` — `'too few pitches to read'` below the floor, else `null`

- [ ] **Step 1: Write the failing tests**

Create `src/lib/derive.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { samplePage } from '../test/fixtures/pitcherPage';
import {
  outingsFor, pitchesFor, traitSeries, meanPitchGrade,
  formatPoints, formatChange, formatUsage, formatValue, thinLabel,
} from './derive';

describe('selection helpers', () => {
  it('keeps only the selected type and sorts outings by date', () => {
    const o = outingsFor(samplePage, 'FF');
    expect(o.map((r) => r.date)).toEqual(['2026-03-15', '2026-03-22']);
    expect(outingsFor(samplePage, 'Slider')).toHaveLength(1);
  });

  it('filters pitches by type and optionally by count', () => {
    expect(pitchesFor(samplePage, 'FF')).toHaveLength(3);
    expect(pitchesFor(samplePage, 'FF', '0-2')).toHaveLength(1);
    expect(pitchesFor(samplePage, 'FF', '3-2')).toHaveLength(0);
  });
});

describe('traitSeries', () => {
  it('averages the feature per date, with the pitch count behind it', () => {
    // FF velo: 2026-03-15 has 91 and 89 -> 90; 2026-03-22 has 90 -> 90.
    expect(traitSeries(samplePage, 'FF', 0)).toEqual([
      { date: '2026-03-15', n: 2, value: 90 },
      { date: '2026-03-22', n: 1, value: 90 },
    ]);
  });

  it('returns an empty series for a type with no pitches', () => {
    expect(traitSeries(samplePage, 'Curveball', 0)).toEqual([]);
  });
});

describe('meanPitchGrade', () => {
  it('averages the per-pitch grades, which is the scale-coherence check', () => {
    // (124 + 112 + 118) / 3
    expect(meanPitchGrade(samplePage, 'FF')).toBeCloseTo(118, 10);
  });

  it('is null when there are no pitches of that type', () => {
    expect(meanPitchGrade(samplePage, 'Cutter')).toBeNull();
  });
});

describe('formatting', () => {
  it('signs point values and drops decimals', () => {
    expect(formatPoints(8.7)).toBe('+9');
    expect(formatPoints(-3.2)).toBe('-3');
    expect(formatPoints(0)).toBe('0');
  });

  it('renders a null recent change as blank, never as zero', () => {
    expect(formatChange(null)).toBe('');
    expect(formatChange(-6.2)).toBe('-6');
  });

  it('formats usage as a whole percent', () => {
    expect(formatUsage(0.58)).toBe('58%');
  });

  it('drops decimals on values large enough not to need them', () => {
    expect(formatValue(6.83)).toBe('6.8');
    expect(formatValue(2320.4)).toBe('2320');
  });
});

describe('thinLabel', () => {
  it('says so below the floor and says nothing above it', () => {
    expect(thinLabel(40, 100)).toBe('too few pitches to read');
    expect(thinLabel(100, 100)).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/derive.test.ts` — Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

Create `src/lib/derive.ts`:

```ts
import type { OutingRow, PitchRow, PitchType, PitcherPage } from './types';

export function outingsFor(page: PitcherPage, type: PitchType): OutingRow[] {
  return page.outings.filter((o) => o.type === type)
    .sort((a, b) => a.date.localeCompare(b.date));
}

export function pitchesFor(page: PitcherPage, type: PitchType, count?: string): PitchRow[] {
  return page.pitches.filter((p) => p.t === type && (count === undefined || p.c === count));
}

/**
 * Per-date mean of one feature for one pitch type.
 *
 * The bundle ships no per-date trait table; it ships every pitch with its feature
 * vector, so the series is derived here. Averaging the raw values is exact, not an
 * approximation of some other quantity.
 */
export function traitSeries(page: PitcherPage, type: PitchType, featureIndex: number):
  { date: string; n: number; value: number }[] {
  const byDate = new Map<string, { sum: number; n: number }>();
  for (const p of page.pitches) {
    if (p.t !== type) continue;
    const acc = byDate.get(p.d) ?? { sum: 0, n: 0 };
    acc.sum += p.f[featureIndex];
    acc.n += 1;
    byDate.set(p.d, acc);
  }
  return [...byDate.entries()]
    .map(([date, a]) => ({ date, n: a.n, value: a.sum / a.n }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/** Mean of the per-pitch grades. Equals that type's published Stuff+ by
 *  construction, because one affine transform is used at every level. */
export function meanPitchGrade(page: PitcherPage, type: PitchType): number | null {
  const g = page.pitches.filter((p) => p.t === type).map((p) => p.g);
  if (g.length === 0) return null;
  return g.reduce((a, b) => a + b, 0) / g.length;
}

export function formatPoints(n: number): string {
  const r = Math.round(n);
  return r > 0 ? `+${r}` : String(r);
}

/** Blank for null. A null recent change means a window was below the sample
 *  floor; a "0" would claim no change, which is a different and wrong claim. */
export function formatChange(n: number | null): string {
  return n === null ? '' : formatPoints(n);
}

export function formatUsage(share: number): string {
  return `${Math.round(share * 100)}%`;
}

export function formatValue(n: number): string {
  return Math.abs(n) >= 100 ? String(Math.round(n)) : n.toFixed(1);
}

export const THIN_SAMPLE_LABEL = 'too few pitches to read';

export function thinLabel(n: number, floor: number): string | null {
  return n < floor ? THIN_SAMPLE_LABEL : null;
}
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run src/lib/derive.test.ts` — Expected: all PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/lib/derive.ts src/lib/derive.test.ts
git commit -m "Derive the per-date trait series and page formatting from the bundle"
```

---

## Task 6: Route, page shell, and the arsenal table

**Repo:** `ud-athletics-baseball-pitching`.

**Files:**
- Create: `src/components/pitcher/ArsenalTable.tsx`
- Create: `src/components/pitcher/ArsenalTable.test.tsx`
- Create: `src/pages/PitcherPage.tsx`
- Create: `src/pages/PitcherPage.test.tsx`
- Modify: `src/App.tsx` (add the route)
- Modify: `src/pages/StaffBoard.tsx` (link the pitcher name)
- Modify: `src/pages/StaffBoard.test.tsx` (assert the link)

**Interfaces:**
- Consumes: `usePitcherPage`, `ArsenalRow`, `PitchType`, `formatUsage`/`formatChange`/`thinLabel`, `ScoreCell`.
- Produces:
  - `ArsenalTable` props: `{ rows: ArsenalRow[]; selected: PitchType; onSelect: (t: PitchType) => void; floorByType: Partial<Record<PitchType, number>> }`
  - `PitcherPage` — default export, no props; reads `:pitcherId` from the route.
  - Route: `/pitcher/:pitcherId`.

The three panels do not exist yet. Render a placeholder `<p>` per panel with a stable `data-testid` (`panel-traits`, `panel-zone`, `panel-trend`) so Tasks 7–9 each replace exactly one placeholder without touching the shell.

- [ ] **Step 1: Write the failing ArsenalTable test**

Create `src/components/pitcher/ArsenalTable.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ArsenalTable from './ArsenalTable';
import { samplePage } from '../../test/fixtures/pitcherPage';

const floors = { FF: 100, Slider: 100 } as const;

describe('ArsenalTable', () => {
  it('shows one row per pitch type with plain labels and usage', () => {
    render(<ArsenalTable rows={samplePage.arsenal} selected="FF" onSelect={() => {}} floorByType={floors} />);
    expect(screen.getByText('Fastball')).toBeInTheDocument();
    expect(screen.getByText('Slider')).toBeInTheDocument();
    expect(screen.getByText('60%')).toBeInTheDocument();
  });

  it('shows Location+ only on the fastball row', () => {
    render(<ArsenalTable rows={samplePage.arsenal} selected="FF" onSelect={() => {}} floorByType={floors} />);
    expect(screen.getByText('103')).toBeInTheDocument();
    // The secondary row's Location+ cell carries an em-dash with an explanation.
    const dash = screen.getByTitle(/does not predict next-year outcomes/i);
    expect(dash).toHaveTextContent('—');
  });

  it('leaves recent change blank rather than zero when it is null', () => {
    render(<ArsenalTable rows={samplePage.arsenal} selected="FF" onSelect={() => {}} floorByType={floors} />);
    const cell = screen.getByTestId('recent-Slider');
    expect(cell).toHaveTextContent('');
    expect(cell).not.toHaveTextContent('0');
  });

  it('flags a pitch type below its sample floor', () => {
    render(<ArsenalTable rows={samplePage.arsenal} selected="FF" onSelect={() => {}} floorByType={floors} />);
    expect(screen.getByText(/too few pitches to read/i)).toBeInTheDocument();
  });

  it('selects a row on click', () => {
    const onSelect = vi.fn();
    render(<ArsenalTable rows={samplePage.arsenal} selected="FF" onSelect={onSelect} floorByType={floors} />);
    fireEvent.click(screen.getByRole('button', { name: /Slider/ }));
    expect(onSelect).toHaveBeenCalledWith('Slider');
  });

  it('marks the selected row for assistive tech', () => {
    render(<ArsenalTable rows={samplePage.arsenal} selected="Slider" onSelect={() => {}} floorByType={floors} />);
    expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Slider');
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/components/pitcher/ArsenalTable.test.tsx` — Expected: FAIL, module missing.

- [ ] **Step 3: Implement ArsenalTable**

Create `src/components/pitcher/ArsenalTable.tsx`:

```tsx
import type { ArsenalRow, PitchType } from '../../lib/types';
import ScoreCell from '../ui/ScoreCell';
import { formatChange, formatUsage, thinLabel } from '../../lib/derive';

const SECONDARY_LOC_NOTE =
  'Location+ is shown for fastballs only. For secondary pitches it repeats year to '
  + 'year but does not predict next-year outcomes, so it is not surfaced.';

const RECENT_NOTE =
  'Stuff+ over the last 30 days minus the 30 days before. Blank when either window '
  + 'has too few pitches.';

export default function ArsenalTable({ rows, selected, onSelect, floorByType }: {
  rows: ArsenalRow[];
  selected: PitchType;
  onSelect: (t: PitchType) => void;
  floorByType: Partial<Record<PitchType, number>>;
}) {
  return (
    <div className="max-w-3xl overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left border-b border-ud-gray">
            <th className="py-2 pr-2">Pitch</th>
            <th className="py-2 px-2 text-right font-normal text-ud-gray">Usage</th>
            <th className="py-2 px-2 text-right font-normal text-ud-gray">Pitches</th>
            <th className="py-2 px-2">Stuff+</th>
            <th className="py-2 px-2">Location+</th>
            <th className="py-2 px-2" title={RECENT_NOTE}>Last 30 days</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const floor = floorByType[r.type] ?? 0;
            const thin = thinLabel(r.n, floor);
            const isSelected = r.type === selected;
            return (
              <tr key={r.type} aria-selected={isSelected}
                className={`border-b border-gray-100 ${isSelected ? 'bg-ud-cream/40' : 'hover:bg-gray-50'}`}>
                <td className="py-2 pr-2">
                  <button type="button" onClick={() => onSelect(r.type)}
                    className="cursor-pointer select-none bg-transparent border-0 p-0 text-left font-[inherit]">
                    {r.label}
                  </button>
                  {thin && <span className="ml-2 text-xs text-ud-gray">{thin}</span>}
                </td>
                <td className="py-2 px-2 text-right tabular-nums text-ud-gray">{formatUsage(r.usage)}</td>
                <td className="py-2 px-2 text-right tabular-nums text-ud-gray">{r.n}</td>
                <td className="py-2 px-2"><ScoreCell value={r.stuff} /></td>
                <td className="py-2 px-2">
                  {r.loc === null
                    ? <span className="text-ud-gray cursor-help" title={SECONDARY_LOC_NOTE}>—</span>
                    : <ScoreCell value={r.loc} />}
                </td>
                <td className="py-2 px-2 tabular-nums" data-testid={`recent-${r.type}`}
                  title={r.recentChange === null ? RECENT_NOTE : undefined}>
                  {formatChange(r.recentChange)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: Run the ArsenalTable test**

Run: `npx vitest run src/components/pitcher/ArsenalTable.test.tsx` — Expected: PASS.

- [ ] **Step 5: Write the failing page test**

Create `src/pages/PitcherPage.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import PitcherPage from './PitcherPage';
import { samplePage, sampleModel, sampleMaps } from '../test/fixtures/pitcherPage';

function bodyFor(url: string): unknown {
  if (url.includes('model_artifacts')) return sampleModel;
  if (url.includes('location_maps')) return sampleMaps;
  return samplePage;
}

function renderAt(path: string) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes><Route path="/pitcher/:pitcherId" element={<PitcherPage />} /></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe('PitcherPage', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn((url: string) =>
      Promise.resolve({ ok: true, json: () => Promise.resolve(bodyFor(url)) } as Response)));
  });
  afterEach(() => vi.unstubAllGlobals());

  it('shows the pitcher, the season, and that handedness is included', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => expect(screen.getByText('Test-Pitcher, Alpha')).toBeInTheDocument());
    expect(screen.getByText(/2026/)).toBeInTheDocument();
    expect(screen.getByText(/includes handedness impact/i)).toBeInTheDocument();
  });

  it('defaults the selection to the highest-usage pitch type', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Fastball'));
  });

  it('changes selection when another pitch type is clicked', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => screen.getByText('Slider'));
    fireEvent.click(screen.getByRole('button', { name: /Slider/ }));
    expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Slider');
  });

  it('renders all three panels', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => screen.getByTestId('panel-traits'));
    expect(screen.getByTestId('panel-zone')).toBeInTheDocument();
    expect(screen.getByTestId('panel-trend')).toBeInTheDocument();
  });

  it('links back to the staff board', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => expect(screen.getByRole('link', { name: /staff board/i })).toHaveAttribute('href', '/'));
  });

  it('reports a load failure plainly', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: false, status: 404 } as Response)));
    renderAt('/pitcher/1000101');
    await waitFor(() => expect(screen.getByText(/something went wrong loading this pitcher/i)).toBeInTheDocument());
  });
});
```

- [ ] **Step 6: Run it to verify it fails**

Run: `npx vitest run src/pages/PitcherPage.test.tsx` — Expected: FAIL, module missing.

- [ ] **Step 7: Implement the page shell**

Create `src/pages/PitcherPage.tsx`. The panel bodies are placeholders that Tasks 7–9 replace one at a time.

```tsx
import { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import { usePitcherPage } from '../hooks/usePitcherPage';
import type { PitchType } from '../lib/types';
import ArsenalTable from '../components/pitcher/ArsenalTable';

export default function PitcherPage() {
  const { pitcherId } = useParams<{ pitcherId: string }>();
  const id = pitcherId ? Number(pitcherId) : NaN;
  const { data, isLoading, isError } = usePitcherPage(Number.isFinite(id) ? id : null);
  const [selected, setSelected] = useState<PitchType | null>(null);

  const floorByType = useMemo(() => {
    const out: Partial<Record<PitchType, number>> = {};
    if (!data) return out;
    for (const [t, m] of Object.entries(data.model.byPitchType)) {
      if (m) out[t as PitchType] = m.sampleFloor;
    }
    return out;
  }, [data]);

  if (isLoading) return <p className="text-ud-gray">Loading pitcher…</p>;
  if (isError || !data) {
    return <p className="text-ud-error">Something went wrong loading this pitcher. Try refreshing.</p>;
  }

  const { page, model, maps } = data;
  // Arsenal arrives sorted by usage descending, so the first row is the default.
  const type: PitchType = selected ?? page.arsenal[0].type;
  const row = page.arsenal.find((a) => a.type === type)!;
  const artifact = model.byPitchType[type];

  return (
    <section>
      <Link to="/" className="text-sm text-ud-blue inline-flex items-center gap-1 mb-2">
        <ArrowLeft className="w-4 h-4" /> Staff Board
      </Link>
      <h1 className="font-display text-2xl text-ud-navy mb-1">{page.name}</h1>
      <p className="text-sm text-ud-gray mb-1">
        {page.hand}HP · {page.season} season · 100 = D1 average, higher is better
      </p>
      <p className="text-xs text-ud-gray mb-4">Includes handedness impact</p>

      <h2 className="font-display text-lg text-ud-navy mb-2">Arsenal</h2>
      <ArsenalTable rows={page.arsenal} selected={type} onSelect={setSelected} floorByType={floorByType} />

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div data-testid="panel-traits">
          <h2 className="font-display text-lg text-ud-navy mb-2">Why it grades this way</h2>
          <p className="text-sm text-ud-gray">Panel pending.</p>
        </div>
        <div data-testid="panel-zone">
          <h2 className="font-display text-lg text-ud-navy mb-2">Where it goes</h2>
          <p className="text-sm text-ud-gray">Panel pending.</p>
        </div>
        <div data-testid="panel-trend" className="lg:col-span-2">
          <h2 className="font-display text-lg text-ud-navy mb-2">Trend</h2>
          <p className="text-sm text-ud-gray">Panel pending.</p>
        </div>
      </div>

      {!artifact && (
        <p className="mt-4 text-xs text-ud-gray">
          No model for {row.label} this season; too few qualifying pitchers to grade it.
        </p>
      )}
      <p className="sr-only" data-testid="maps-loaded">{Object.keys(maps).length} location maps</p>
    </section>
  );
}
```

The `maps-loaded` line exists only so `maps` is referenced before Task 8 consumes it; Task 8 deletes it.

- [ ] **Step 8: Add the route and the Staff Board link**

In `src/App.tsx`, add the route inside the existing `<Routes>`:

```tsx
import PitcherPage from './pages/PitcherPage';
...
          <Route path="/" element={<StaffBoard />} />
          <Route path="/pitcher/:pitcherId" element={<PitcherPage />} />
```

In `src/pages/StaffBoard.tsx`, the pitcher-name cell currently holds one button that toggles the drivers row. Keep that button for the chevron and add a link on the name itself, so the row still expands in place and the name navigates. Replace the name span inside `FragmentRow`'s first `<td>`:

```tsx
          <button type="button" className="cursor-pointer select-none bg-transparent border-0 p-0 text-left"
            onClick={onToggle} aria-expanded={open} aria-controls={`drivers-${row.id}`}
            aria-label={`Show Stuff+ drivers for ${row.name}`}>
            {open ? <ChevronDown className="inline w-4 h-4" /> : <ChevronRight className="inline w-4 h-4" />}
          </button>
          {row.pitcherId === null
            ? <span className="ml-1">{row.name}</span>
            : <Link to={`/pitcher/${row.pitcherId}`} className="ml-1 text-ud-blue hover:underline">{row.name}</Link>}
```

Add `import { Link } from 'react-router-dom';` at the top of `StaffBoard.tsx`.

- [ ] **Step 9: Extend the Staff Board test**

Add to `src/pages/StaffBoard.test.tsx`. Its existing render helper wraps in `QueryClientProvider` only; a `<Link>` needs a router, so wrap in `MemoryRouter` too — update the shared `wrap` helper rather than each test:

```tsx
const wrap = (ui: React.ReactNode) => (
  <QueryClientProvider client={new QueryClient()}>
    <MemoryRouter>{ui}</MemoryRouter>
  </QueryClientProvider>
);
```

and add:

```tsx
  it('links a pitcher name to his page', async () => {
    render(wrap(<StaffBoard />));
    await waitFor(() => screen.getByText('Test-Pitcher, Alpha'));
    expect(screen.getByRole('link', { name: 'Test-Pitcher, Alpha' }))
      .toHaveAttribute('href', '/pitcher/1000101');
  });

  it('does not link a pitcher with no graded arsenal', async () => {
    render(wrap(<StaffBoard />));
    await waitFor(() => screen.getByText('Test-Pitcher, Bravo'));
    expect(screen.queryByRole('link', { name: 'Test-Pitcher, Bravo' })).toBeNull();
  });
```

The existing fixtures that back `StaffBoard.test.tsx` must gain `pitcherId` (`1000101` on the first row, `null` on the second) to match `sampleBoard`. If those fixtures are inline in the test file, edit them there.

- [ ] **Step 10: Run everything**

Run: `npx vitest run` — Expected: all PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean.

- [ ] **Step 11: Commit**

```bash
git add src/pages src/components/pitcher src/App.tsx
git commit -m "Open a pitcher's page from the staff board with his arsenal"
```

---

## Task 7: Panel 1 — the trait attribution waterfall

**Repo:** `ud-athletics-baseball-pitching`.

**Files:**
- Create: `src/components/pitcher/TraitPanel.tsx`
- Create: `src/components/pitcher/TraitPanel.test.tsx`
- Modify: `src/pages/PitcherPage.tsx` (replace the `panel-traits` placeholder, add selected-pitch state)

**Interfaces:**
- Consumes: `waterfallRows`, `standardize`, `contributions` from `../../lib/attribution`; `formatPoints`, `formatValue` from `../../lib/derive`.
- Produces: `TraitPanel` props
  ```ts
  {
    row: ArsenalRow;               // the selected pitch type
    artifact: TypeArtifact | undefined;
    featureOrder: string[];
    labels: Record<string, string>;
    selectedPitch: PitchRow | null;  // null = pitch-type view
    onClearPitch: () => void;
  }
  ```
- Produces on `PitcherPage`: `selectedPitch` state, cleared whenever the pitch type changes.

Bars are hand-rolled divs, not a chart library: a horizontal bar per row with a value and a percentile beside it is a table with a bar in one cell, and recharts would fight the alignment. Bar width is `|points| / maxAbs` as a percentage. Positive bars extend right from a centered axis in UD Blue; negative bars extend left in UD Error. Direct labels, no legend.

- [ ] **Step 1: Write the failing test**

Create `src/components/pitcher/TraitPanel.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TraitPanel from './TraitPanel';
import { samplePage, sampleModel } from '../../test/fixtures/pitcherPage';
import { contributions, standardize } from '../../lib/attribution';

const ff = sampleModel.byPitchType.FF!;
const row = samplePage.arsenal[0];
const base = {
  row, artifact: ff, featureOrder: sampleModel.featureOrder,
  labels: sampleModel.labels, selectedPitch: null, onClearPitch: () => {},
};

describe('TraitPanel', () => {
  it('uses plain-English labels, never raw field names', () => {
    render(<TraitPanel {...base} />);
    expect(screen.getByText('Velocity')).toBeInTheDocument();
    expect(screen.queryByText('velo')).toBeNull();
    expect(screen.queryByText('is_lhp')).toBeNull();
  });

  it('states what the contributions are measured against', () => {
    render(<TraitPanel {...base} />);
    expect(screen.getByText(/vs the average qualified D1 pitcher/i)).toBeInTheDocument();
  });

  it('shows the value and the percentile beside each contribution', () => {
    render(<TraitPanel {...base} />);
    const velo = screen.getByTestId('trait-velo');
    expect(velo).toHaveTextContent('90.0');
    expect(velo).toHaveTextContent('78th');
  });

  it('shows a total that equals the sum of the bars', () => {
    render(<TraitPanel {...base} />);
    const expected = contributions(row.typical, ff, ff.populationMeanZ).reduce((a, b) => a + b, 0);
    expect(screen.getByTestId('trait-total')).toHaveTextContent(
      expected > 0 ? `+${Math.round(expected)}` : String(Math.round(expected)));
  });

  it('groups handedness into one non-coachable row', () => {
    render(<TraitPanel {...base} />);
    expect(screen.getByText(/handedness \(context, not coachable\)/i)).toBeInTheDocument();
  });

  it('rebaselines against his own typical pitch when one pitch is selected', () => {
    render(<TraitPanel {...base} selectedPitch={samplePage.pitches[0]} />);
    expect(screen.getByText(/vs his own typical fastball/i)).toBeInTheDocument();
  });

  it('drops percentiles for a selected pitch', () => {
    render(<TraitPanel {...base} selectedPitch={samplePage.pitches[0]} />);
    expect(screen.getByTestId('trait-velo')).not.toHaveTextContent('th');
  });

  it('sums exactly to the selected pitch grade minus his typical grade', () => {
    const pitch = samplePage.pitches[0];
    const baselineZ = standardize(row.typical, ff.scalerMean, ff.scalerScale);
    const sum = contributions(pitch.f, ff, baselineZ).reduce((a, b) => a + b, 0);
    // The pitch's own grade minus the pitch type's grade is the same quantity.
    expect(sum).toBeCloseTo(pitch.g - row.stuff, 6);
  });

  it('offers a way back to the pitch-type view', () => {
    const onClearPitch = vi.fn();
    render(<TraitPanel {...base} selectedPitch={samplePage.pitches[0]} onClearPitch={onClearPitch} />);
    fireEvent.click(screen.getByRole('button', { name: /back to all fastballs/i }));
    expect(onClearPitch).toHaveBeenCalled();
  });

  it('says so plainly when the pitch type has no model', () => {
    render(<TraitPanel {...base} artifact={undefined} />);
    expect(screen.getByText(/not enough qualifying pitchers to grade this pitch/i)).toBeInTheDocument();
  });
});
```

Note on the eighth assertion: it holds because both sides are the same affine transform of the same ridge, so it is a genuine coherence check on the fixture. It uses `toBeCloseTo(…, 6)` rather than 10 because `sampleModel`'s hand-written `g` and `stuff` values are rounded, not machine-derived. Real bundle data is checked against the tighter bound in Task 11. **If this assertion fails, adjust the fixture's `g`/`stuff` values so the identity holds — do not loosen the tolerance further and do not change `attribution.ts`.**

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/components/pitcher/TraitPanel.test.tsx` — Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

Create `src/components/pitcher/TraitPanel.tsx`:

```tsx
import { useMemo } from 'react';
import type { ArsenalRow, PitchRow, TypeArtifact } from '../../lib/types';
import { standardize, waterfallRows } from '../../lib/attribution';
import { formatPoints, formatValue } from '../../lib/derive';

const ORDINAL = (p: number): string => {
  const s = ['th', 'st', 'nd', 'rd'];
  const v = p % 100;
  return `${p}${s[(v - 20) % 10] ?? s[v] ?? s[0]}`;
};

export default function TraitPanel({ row, artifact, featureOrder, labels, selectedPitch, onClearPitch }: {
  row: ArsenalRow;
  artifact: TypeArtifact | undefined;
  featureOrder: string[];
  labels: Record<string, string>;
  selectedPitch: PitchRow | null;
  onClearPitch: () => void;
}) {
  const rows = useMemo(() => {
    if (!artifact) return [];
    // Two baselines. Pitch-type view measures against the qualified population
    // mean ("why his fastball grades 124 rather than 100"). Single-pitch view
    // measures against his own typical pitch ("why this pitch differs from his
    // usual one"). Both are exact: the ridge is linear in standardized features.
    const baselineZ = selectedPitch
      ? standardize(row.typical, artifact.scalerMean, artifact.scalerScale)
      : artifact.populationMeanZ;
    const values = selectedPitch ? selectedPitch.f : row.typical;
    return waterfallRows({
      values, artifact, baselineZ, featureOrder, labels,
      values2: values,
      // A percentile ranks his typical pitch against other pitchers. Ranking a
      // single pitch would need a pitch-level reference population, which is a
      // second reference for one word. So: no percentile on a selected pitch.
      percentiles: selectedPitch ? null : row.percentiles,
    });
  }, [row, artifact, featureOrder, labels, selectedPitch]);

  if (!artifact) {
    return <p className="text-sm text-ud-gray">Not enough qualifying pitchers to grade this pitch.</p>;
  }

  const total = rows.reduce((a, r) => a + r.points, 0);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.points)), 1);

  return (
    <div>
      <p className="text-xs text-ud-gray mb-1">
        {selectedPitch
          ? `Stuff+ points vs his own typical ${row.label.toLowerCase()}`
          : `Stuff+ points vs the average qualified D1 pitcher's ${row.label.toLowerCase()}`}
      </p>
      {selectedPitch && (
        <p className="text-xs mb-2">
          <span className="text-ud-gray mr-2">{selectedPitch.d} · {selectedPitch.c} count · graded {Math.round(selectedPitch.g)}</span>
          <button type="button" onClick={onClearPitch}
            className="text-ud-blue bg-transparent border-0 p-0 cursor-pointer font-[inherit]">
            Back to all {row.label.toLowerCase()}s
          </button>
        </p>
      )}
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left text-xs text-ud-gray border-b border-gray-100">
            <th className="py-1 pr-2 font-normal">Trait</th>
            <th className="py-1 px-2 font-normal text-right">His value</th>
            <th className="py-1 px-2 font-normal text-right">Rank</th>
            <th className="py-1 pl-2 font-normal">Worth</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const pct = (Math.abs(r.points) / maxAbs) * 50;
            return (
              <tr key={r.key} data-testid={`trait-${r.key}`} className="border-b border-gray-50">
                <td className={`py-1 pr-2 ${r.grouped ? 'text-ud-gray italic' : ''}`}>{r.label}</td>
                <td className="py-1 px-2 text-right tabular-nums text-ud-gray">
                  {r.value === null ? '' : formatValue(r.value)}
                </td>
                <td className="py-1 px-2 text-right tabular-nums text-ud-gray">
                  {r.percentile === null ? '' : ORDINAL(r.percentile)}
                </td>
                <td className="py-1 pl-2">
                  <div className="flex items-center">
                    <div className="w-1/2 flex justify-end">
                      {r.points < 0 && (
                        <div style={{ width: `${pct * 2}%`, backgroundColor: r.grouped ? '#bdbdbd' : '#dc2626' }}
                          className="h-3 rounded-l" />
                      )}
                    </div>
                    <div className="w-1/2">
                      {r.points >= 0 && (
                        <div style={{ width: `${pct * 2}%`, backgroundColor: r.grouped ? '#bdbdbd' : '#00539f' }}
                          className="h-3 rounded-r" />
                      )}
                    </div>
                    <span className="ml-2 tabular-nums text-xs w-8 text-right">{formatPoints(r.points)}</span>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="border-t border-ud-gray">
            <td className="py-1 pr-2 font-semibold" colSpan={3}>Total</td>
            <td className="py-1 pl-2 tabular-nums font-semibold" data-testid="trait-total">
              {formatPoints(total)}
            </td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: Wire it into the page**

In `src/pages/PitcherPage.tsx`, add selected-pitch state and clear it whenever the type changes, then replace the `panel-traits` placeholder body:

```tsx
  const [selectedPitch, setSelectedPitch] = useState<PitchRow | null>(null);

  const selectType = (t: PitchType) => { setSelected(t); setSelectedPitch(null); };
```

Pass `selectType` to `ArsenalTable`'s `onSelect` instead of `setSelected`. Then:

```tsx
        <div data-testid="panel-traits">
          <h2 className="font-display text-lg text-ud-navy mb-2">Why it grades this way</h2>
          <TraitPanel row={row} artifact={artifact} featureOrder={model.featureOrder}
            labels={model.labels} selectedPitch={selectedPitch}
            onClearPitch={() => setSelectedPitch(null)} />
        </div>
```

Import `TraitPanel` and `PitchRow`. `setSelectedPitch` has no producer until Task 8; that is expected.

- [ ] **Step 5: Run everything**

Run: `npx vitest run` — Expected: all PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean. `oxlint` may flag `setSelectedPitch` as set-but-unread if it only sees the clear path; it is read by `TraitPanel`'s prop, so this should not fire. If it does, leave the code as-is and note it in the report rather than adding a suppression.

- [ ] **Step 6: Commit**

```bash
git add src/components/pitcher/TraitPanel.tsx src/components/pitcher/TraitPanel.test.tsx src/pages/PitcherPage.tsx
git commit -m "Explain a pitch grade trait by trait, in Stuff+ points"
```

---

## Task 8: Panel 2 — the strike zone

**Repo:** `ud-athletics-baseball-pitching`.

**Files:**
- Create: `src/lib/zone.ts`
- Create: `src/lib/zone.test.ts`
- Create: `src/components/pitcher/ZonePanel.tsx`
- Create: `src/components/pitcher/ZonePanel.test.tsx`
- Modify: `src/pages/PitcherPage.tsx` (replace the `panel-zone` placeholder, delete the `maps-loaded` line)

**Geometry.** Hand-rolled SVG, not recharts: the panel is a 120-cell heatmap with a scatter and a zone outline on top, which is three layers of custom marks. Coordinates in feet, `x` from −1.5 to 1.5 (left to right, catcher's view), `z` from 0.75 to 4.0 (bottom to top). SVG `y` increases downward, so `z` inverts. The nominal strike zone drawn as a rectangle: `x` from −0.83 to 0.83, `z` from 1.5 to 3.5. Label it "nominal strike zone" — it is a fixed rectangle, not each batter's real zone.

**Interfaces:**
- Produces in `src/lib/zone.ts`:
  - `ZONE = { xMin: -1.5, xMax: 1.5, zMin: 0.75, zMax: 4.0, sxMin: -0.83, sxMax: 0.83, szMin: 1.5, szMax: 3.5, cell: 0.25 }`
  - `project(x: number, z: number, w: number, h: number): { px: number; py: number }`
  - `valueColor(v: number, maxAbs: number): string` — diverging blue→white→red on the **negated** value, so blue = good for the pitcher
  - `maxAbsValue(cells: LocationCell[]): number`
  - `gradeColor(g: number): string` — the scatter's optional Stuff+ coloring, reusing the Staff Board's thresholds
- Produces `ZonePanel` props:
  ```ts
  {
    pitches: PitchRow[];        // already filtered to the selected type
    maps: LocationMaps;         // ignored for non-FF
    showSurface: boolean;       // true only for FF
    typeLabel: string;
    onSelectPitch: (p: PitchRow) => void;
    selectedPitch: PitchRow | null;
  }
  ```

- [ ] **Step 1: Write the failing geometry test**

Create `src/lib/zone.test.ts`:

```ts
import { describe, it, expect } from 'vitest';
import { ZONE, project, valueColor, maxAbsValue, gradeColor } from './zone';
import { sampleMaps } from '../test/fixtures/pitcherPage';

describe('project', () => {
  it('puts the plate center in the horizontal middle', () => {
    expect(project(0, ZONE.zMin, 300, 400).px).toBeCloseTo(150, 6);
  });

  it('inverts height, because SVG y grows downward', () => {
    const low = project(0, ZONE.zMin, 300, 400);
    const high = project(0, ZONE.zMax, 300, 400);
    expect(high.py).toBeLessThan(low.py);
    expect(high.py).toBeCloseTo(0, 6);
    expect(low.py).toBeCloseTo(400, 6);
  });

  it('maps the horizontal extremes to the edges', () => {
    expect(project(ZONE.xMin, 2, 300, 400).px).toBeCloseTo(0, 6);
    expect(project(ZONE.xMax, 2, 300, 400).px).toBeCloseTo(300, 6);
  });
});

describe('valueColor', () => {
  it('is blue where expected runs are low, which is good for the pitcher', () => {
    expect(valueColor(-0.05, 0.05)).toMatch(/^rgb\(/);
    const good = valueColor(-0.05, 0.05);
    const bad = valueColor(0.05, 0.05);
    expect(good).not.toEqual(bad);
  });

  it('is near-white at zero', () => {
    expect(valueColor(0, 0.05)).toBe('rgb(255, 255, 255)');
  });

  it('does not blow up on an all-zero map', () => {
    expect(valueColor(0, 0)).toBe('rgb(255, 255, 255)');
  });
});

describe('maxAbsValue', () => {
  it('finds the largest magnitude in the map', () => {
    expect(maxAbsValue(sampleMaps['0-2'])).toBeCloseTo(0.05, 10);
  });
});

describe('gradeColor', () => {
  it('reuses the staff board thresholds', () => {
    expect(gradeColor(120)).not.toEqual(gradeColor(90));
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/lib/zone.test.ts` — Expected: FAIL, module missing.

- [ ] **Step 3: Implement the geometry**

Create `src/lib/zone.ts`:

```ts
import type { LocationCell } from './types';

/** Plot extent and the nominal strike zone, in feet. The zone rectangle is fixed
 *  rather than per-batter, so it is labeled "nominal" wherever it is drawn. */
export const ZONE = {
  xMin: -1.5, xMax: 1.5, zMin: 0.75, zMax: 4.0,
  sxMin: -0.83, sxMax: 0.83, szMin: 1.5, szMax: 3.5,
  /** Location-map cell size. A cell's (x, z) is its LOWER-LEFT corner. */
  cell: 0.25,
} as const;

export function project(x: number, z: number, w: number, h: number): { px: number; py: number } {
  const px = ((x - ZONE.xMin) / (ZONE.xMax - ZONE.xMin)) * w;
  // SVG y grows downward while height grows upward, so invert.
  const py = h - ((z - ZONE.zMin) / (ZONE.zMax - ZONE.zMin)) * h;
  return { px, py };
}

export function maxAbsValue(cells: LocationCell[]): number {
  return cells.reduce((m, c) => Math.max(m, Math.abs(c.v)), 0);
}

const GOOD = [0, 83, 159];    // UD Blue #00539f
const BAD = [220, 38, 38];    // UD Error #dc2626

/**
 * Diverging surface color for a location's run value.
 *
 * `v` is raw expected runs from the pitcher's perspective, LOWER = better, so the
 * ramp is applied to -v: blue where the location is cheap for the pitcher, red
 * where it is expensive, white at neutral. This is the one place the location map's
 * sign convention is flipped, and it is for color only -- no number on the page is
 * negated here.
 */
export function valueColor(v: number, maxAbs: number): string {
  if (maxAbs <= 0) return 'rgb(255, 255, 255)';
  const t = Math.max(-1, Math.min(1, -v / maxAbs));
  const end = t >= 0 ? GOOD : BAD;
  const k = Math.abs(t);
  const mix = end.map((c) => Math.round(255 + (c - 255) * k));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

/** Same thresholds as src/lib/scoreColor.ts, as hex for SVG fills. */
export function gradeColor(g: number): string {
  if (g >= 115) return '#16a34a';
  if (g >= 105) return '#00539f';
  if (g >= 95) return '#bdbdbd';
  return '#dc2626';
}
```

- [ ] **Step 4: Write the failing panel test**

Create `src/components/pitcher/ZonePanel.test.tsx`:

```tsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ZonePanel from './ZonePanel';
import { samplePage, sampleMaps } from '../../test/fixtures/pitcherPage';
import { pitchesFor } from '../../lib/derive';

const ffPitches = pitchesFor(samplePage, 'FF');
const base = {
  pitches: ffPitches, maps: sampleMaps, showSurface: true,
  typeLabel: 'Fastball', onSelectPitch: () => {}, selectedPitch: null,
};

describe('ZonePanel', () => {
  it('plots one dot per pitch', () => {
    render(<ZonePanel {...base} />);
    expect(screen.getAllByTestId(/^pitch-dot-/)).toHaveLength(3);
  });

  it('draws the value surface for a fastball and says what it means', () => {
    render(<ZonePanel {...base} />);
    expect(screen.getAllByTestId(/^value-cell-/).length).toBeGreaterThan(0);
    expect(screen.getByText(/blue = cheap for the pitcher/i)).toBeInTheDocument();
  });

  it('omits the surface for a secondary pitch and says why', () => {
    render(<ZonePanel {...base} showSurface={false} typeLabel="Slider"
      pitches={pitchesFor(samplePage, 'Slider')} />);
    expect(screen.queryAllByTestId(/^value-cell-/)).toHaveLength(0);
    expect(screen.getByText(/shows where pitches went, not what those locations are worth/i))
      .toBeInTheDocument();
  });

  it('filters to a count and switches the surface with it', () => {
    render(<ZonePanel {...base} />);
    fireEvent.change(screen.getByLabelText(/count/i), { target: { value: '0-2' } });
    expect(screen.getAllByTestId(/^pitch-dot-/)).toHaveLength(1);
  });

  it('says so when a count has no pitches rather than showing an empty plot silently', () => {
    render(<ZonePanel {...base} />);
    fireEvent.change(screen.getByLabelText(/count/i), { target: { value: '3-2' } });
    expect(screen.getByText(/no fastballs in this count/i)).toBeInTheDocument();
  });

  it('can color dots by Stuff+, labeled for what that actually shows', () => {
    render(<ZonePanel {...base} />);
    fireEvent.click(screen.getByLabelText(/color dots by stuff\+/i));
    expect(screen.getByText(/stuff\+ has no location term/i)).toBeInTheDocument();
  });

  it('reports a clicked pitch', () => {
    const onSelectPitch = vi.fn();
    render(<ZonePanel {...base} onSelectPitch={onSelectPitch} />);
    fireEvent.click(screen.getAllByTestId(/^pitch-dot-/)[0]);
    expect(onSelectPitch).toHaveBeenCalledWith(ffPitches[0]);
  });

  it('falls back to the pooled map when a count is missing from the bundle', () => {
    render(<ZonePanel {...base} maps={{ pooled: sampleMaps.pooled }} />);
    fireEvent.change(screen.getByLabelText(/count/i), { target: { value: '1-1' } });
    expect(screen.getByText(/all counts pooled/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 5: Run it to verify it fails**

Run: `npx vitest run src/components/pitcher/ZonePanel.test.tsx` — Expected: FAIL, module missing.

- [ ] **Step 6: Implement the panel**

Create `src/components/pitcher/ZonePanel.tsx`:

```tsx
import { useMemo, useState } from 'react';
import type { LocationMaps, PitchRow } from '../../lib/types';
import { ZONE, gradeColor, maxAbsValue, project, valueColor } from '../../lib/zone';

const W = 300;
const H = 380;
const ALL = 'all';
const COUNTS = ['0-0', '0-1', '0-2', '1-0', '1-1', '1-2', '2-0', '2-1', '2-2', '3-0', '3-1', '3-2'];

const SURFACE_NOTE =
  'Blue = cheap for the pitcher, red = expensive, in expected runs for this count.';
const DESCRIPTIVE_NOTE =
  'Shows where pitches went, not what those locations are worth. Location value is '
  + 'a fastball measure only.';
const COLOR_NOTE =
  'Stuff+ has no location term; it is velocity, movement, spin, and release. Coloring '
  + 'by it shows whether his shape changes with where he aims, not which spots are valuable.';

export default function ZonePanel({ pitches, maps, showSurface, typeLabel, onSelectPitch, selectedPitch }: {
  pitches: PitchRow[];
  maps: LocationMaps;
  showSurface: boolean;
  typeLabel: string;
  onSelectPitch: (p: PitchRow) => void;
  selectedPitch: PitchRow | null;
}) {
  const [count, setCount] = useState<string>(ALL);
  const [colorByGrade, setColorByGrade] = useState(false);

  const shown = count === ALL ? pitches : pitches.filter((p) => p.c === count);

  // A count with too little training data is absent from the bundle; fall back to
  // the pooled surface and say so rather than drawing a blank grid.
  const usingPooled = count === ALL || !(count in maps);
  const cells = showSurface ? (usingPooled ? maps.pooled ?? [] : maps[count]) : [];
  const maxAbs = useMemo(() => maxAbsValue(cells), [cells]);
  const cellW = (ZONE.cell / (ZONE.xMax - ZONE.xMin)) * W;
  const cellH = (ZONE.cell / (ZONE.zMax - ZONE.zMin)) * H;

  const sz = project(ZONE.sxMin, ZONE.szMax, W, H);
  const szEnd = project(ZONE.sxMax, ZONE.szMin, W, H);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-4 mb-2 text-sm">
        <span>
          <label htmlFor="count-select" className="text-ud-gray mr-1">Count</label>
          <select id="count-select" value={count} onChange={(e) => setCount(e.target.value)}
            className="border border-ud-gray rounded px-1 py-0.5">
            <option value={ALL}>All</option>
            {COUNTS.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </span>
        <span>
          <input type="checkbox" id="color-by-grade" checked={colorByGrade}
            onChange={(e) => setColorByGrade(e.target.checked)} className="mr-1 align-middle" />
          <label htmlFor="color-by-grade" className="align-middle select-none text-ud-navy">
            Color dots by Stuff+
          </label>
        </span>
        <span className="text-ud-gray tabular-nums">{shown.length} {typeLabel.toLowerCase()}s</span>
      </div>

      <svg width={W} height={H} role="img"
        aria-label={`${typeLabel} locations, catcher's view`} className="bg-white">
        {cells.map((c) => {
          const { px, py } = project(c.x, c.z + ZONE.cell, W, H);
          return <rect key={`${c.x},${c.z}`} data-testid={`value-cell-${c.x}-${c.z}`}
            x={px} y={py} width={cellW} height={cellH} fill={valueColor(c.v, maxAbs)} />;
        })}
        <rect x={sz.px} y={sz.py} width={szEnd.px - sz.px} height={szEnd.py - sz.py}
          fill="none" stroke="#003c71" strokeWidth={1.5} />
        {shown.map((p, i) => {
          const { px, py } = project(p.x, p.z, W, H);
          const isSel = selectedPitch !== null && selectedPitch.d === p.d
            && selectedPitch.x === p.x && selectedPitch.z === p.z && selectedPitch.c === p.c;
          return (
            <circle key={`${p.d}-${i}`} data-testid={`pitch-dot-${i}`}
              cx={px} cy={py} r={isSel ? 6 : 4}
              fill={colorByGrade ? gradeColor(p.g) : '#00539f'}
              fillOpacity={colorByGrade ? 0.9 : 0.55}
              stroke={isSel ? '#ffd200' : 'none'} strokeWidth={isSel ? 2 : 0}
              className="cursor-pointer"
              onClick={() => onSelectPitch(p)} />
          );
        })}
      </svg>

      <p className="text-xs text-ud-gray mt-1">Catcher's view. Box is the nominal strike zone.</p>
      {showSurface
        ? <p className="text-xs text-ud-gray">{SURFACE_NOTE}{usingPooled && ' All counts pooled.'}</p>
        : <p className="text-xs text-ud-gray">{DESCRIPTIVE_NOTE}</p>}
      {colorByGrade && <p className="text-xs text-ud-gray mt-1">{COLOR_NOTE}</p>}
      {shown.length === 0 && (
        <p className="text-xs text-ud-gray mt-1">No {typeLabel.toLowerCase()}s in this count.</p>
      )}
    </div>
  );
}
```

- [ ] **Step 7: Wire it into the page**

In `src/pages/PitcherPage.tsx`, replace the `panel-zone` placeholder body and delete the `maps-loaded` paragraph:

```tsx
        <div data-testid="panel-zone">
          <h2 className="font-display text-lg text-ud-navy mb-2">Where it goes</h2>
          <ZonePanel pitches={pitchesFor(page, type)} maps={maps} showSurface={type === 'FF'}
            typeLabel={row.label} onSelectPitch={setSelectedPitch} selectedPitch={selectedPitch} />
        </div>
```

Import `ZonePanel` and `pitchesFor`.

- [ ] **Step 8: Add the fastball-only assertion to the page test**

Add to `src/pages/PitcherPage.test.tsx`:

```tsx
  it('never draws a value surface for a secondary pitch', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => screen.getByText('Slider'));
    fireEvent.click(screen.getByRole('button', { name: /Slider/ }));
    expect(screen.queryAllByTestId(/^value-cell-/)).toHaveLength(0);
  });

  it('rebaselines the trait panel when a plotted pitch is clicked', async () => {
    renderAt('/pitcher/1000101');
    await waitFor(() => screen.getAllByTestId(/^pitch-dot-/));
    fireEvent.click(screen.getAllByTestId(/^pitch-dot-/)[0]);
    expect(screen.getByText(/vs his own typical fastball/i)).toBeInTheDocument();
  });
```

- [ ] **Step 9: Run everything**

Run: `npx vitest run` — Expected: all PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean.

- [ ] **Step 10: Commit**

```bash
git add src/lib/zone.ts src/lib/zone.test.ts src/components/pitcher/ZonePanel.tsx src/components/pitcher/ZonePanel.test.tsx src/pages
git commit -m "Plot pitch locations over the count's run-value surface"
```

---

## Task 9: Panel 3 — the trend

**Repo:** `ud-athletics-baseball-pitching`.

**Files:**
- Create: `src/components/pitcher/TrendPanel.tsx`
- Create: `src/components/pitcher/TrendPanel.test.tsx`
- Modify: `src/pages/PitcherPage.tsx` (replace the `panel-trend` placeholder)

This is the repo's **first use of recharts**, so it sets the pattern: named imports from `recharts`, a `ResponsiveContainer` with an explicit pixel height, `CartesianGrid` with `stroke="#f3f4f6"` only (minimal gridlines per the data-viz standard), and `Tooltip` with a `labelFormatter`. Recharts renders nothing in jsdom without a container width; give the container a fixed `width={520}` rather than `"100%"` so the component is testable, and let the parent grid clip on small screens via `overflow-x-auto`.

**Interfaces:**
- Consumes: `outingsFor`, `traitSeries`, `thinLabel` from `../../lib/derive`; `waterfallRows` output for defaulting which traits to show.
- Produces: `TrendPanel` props
  ```ts
  {
    page: PitcherPage;
    type: PitchType;
    typeLabel: string;
    sampleFloor: number;
    featureOrder: string[];
    labels: Record<string, string>;
    /** Feature keys of the largest contributors, largest first. The panel shows
     *  the first two by default. */
    topTraits: string[];
  }
  ```

- [ ] **Step 1: Write the failing test**

Create `src/components/pitcher/TrendPanel.test.tsx`:

```tsx
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import TrendPanel from './TrendPanel';
import { samplePage, sampleModel } from '../../test/fixtures/pitcherPage';

const base = {
  page: samplePage, type: 'FF' as const, typeLabel: 'Fastball', sampleFloor: 100,
  featureOrder: sampleModel.featureOrder, labels: sampleModel.labels,
  topTraits: ['velo', 'spin'],
};

describe('TrendPanel', () => {
  it('charts Stuff+ by outing', () => {
    render(<TrendPanel {...base} />);
    expect(screen.getByText(/stuff\+ by outing/i)).toBeInTheDocument();
  });

  it('defaults to the two largest contributors, by plain label', () => {
    render(<TrendPanel {...base} />);
    expect(screen.getByLabelText('Velocity')).toBeChecked();
    expect(screen.getByLabelText('Spin rate')).toBeChecked();
  });

  it('never offers a handedness feature as a trait line', () => {
    render(<TrendPanel {...base} />);
    expect(screen.queryByLabelText('Throws left')).toBeNull();
  });

  it('lets the remaining traits be turned on', () => {
    render(<TrendPanel {...base} topTraits={['velo']} />);
    expect(screen.getByLabelText('Spin rate')).not.toBeChecked();
    fireEvent.click(screen.getByLabelText('Spin rate'));
    expect(screen.getByLabelText('Spin rate')).toBeChecked();
  });

  it('labels outings below the sample floor rather than drawing a confident line', () => {
    render(<TrendPanel {...base} />);
    expect(screen.getByText(/outings under 100 pitches are too few to read/i)).toBeInTheDocument();
  });

  it('says so plainly when the pitch type has no outings', () => {
    render(<TrendPanel {...base} type="Cutter" typeLabel="Cutter" />);
    expect(screen.getByText(/no cutters this season/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run src/components/pitcher/TrendPanel.test.tsx` — Expected: FAIL, module missing.

- [ ] **Step 3: Implement**

Create `src/components/pitcher/TrendPanel.tsx`:

```tsx
import { useMemo, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import type { PitchType, PitcherPage } from '../../lib/types';
import { HANDEDNESS_FEATURES } from '../../lib/attribution';
import { outingsFor, traitSeries } from '../../lib/derive';

const TRAIT_COLORS = ['#00539f', '#00a0df', '#003c71', '#bdbdbd'];

export default function TrendPanel({ page, type, typeLabel, sampleFloor, featureOrder, labels, topTraits }: {
  page: PitcherPage;
  type: PitchType;
  typeLabel: string;
  sampleFloor: number;
  featureOrder: string[];
  labels: Record<string, string>;
  topTraits: string[];
}) {
  // Handedness is not a trait a pitcher develops, so it is never a trend line.
  const selectable = useMemo(
    () => featureOrder.filter((f) => !HANDEDNESS_FEATURES.includes(f)),
    [featureOrder],
  );
  const defaults = useMemo(
    () => topTraits.filter((t) => selectable.includes(t)).slice(0, 2),
    [topTraits, selectable],
  );
  const [shown, setShown] = useState<string[]>(defaults);

  const outings = outingsFor(page, type);
  if (outings.length === 0) {
    return <p className="text-sm text-ud-gray">No {typeLabel.toLowerCase()}s this season.</p>;
  }

  const thin = outings.some((o) => o.n < sampleFloor);
  const toggle = (f: string) =>
    setShown((s) => (s.includes(f) ? s.filter((x) => x !== f) : [...s, f]));

  const traitRows = useMemo(() => {
    const byDate = new Map<string, Record<string, number | string>>();
    for (const o of outings) byDate.set(o.date, { date: o.date });
    for (const f of shown) {
      const idx = featureOrder.indexOf(f);
      for (const pt of traitSeries(page, type, idx)) {
        const row = byDate.get(pt.date) ?? { date: pt.date };
        row[f] = pt.value;
        byDate.set(pt.date, row);
      }
    }
    return [...byDate.values()].sort((a, b) => String(a.date).localeCompare(String(b.date)));
  }, [outings, shown, featureOrder, page, type]);

  return (
    <div className="overflow-x-auto">
      <h3 className="text-sm text-ud-navy mb-1">Stuff+ by outing</h3>
      <LineChart width={520} height={200} data={outings}>
        <CartesianGrid stroke="#f3f4f6" vertical={false} />
        <XAxis dataKey="date" tick={{ fontSize: 11 }} />
        <YAxis domain={['dataMin - 5', 'dataMax + 5']} tick={{ fontSize: 11 }} />
        <Tooltip formatter={(v: number) => Math.round(v)} />
        <Line type="linear" dataKey="stuff" stroke="#00539f" strokeWidth={2}
          dot={{ r: 3 }} name={`${typeLabel} Stuff+`} />
      </LineChart>
      {thin && (
        <p className="text-xs text-ud-gray">
          Outings under {sampleFloor} pitches are too few to read on their own.
        </p>
      )}

      <h3 className="text-sm text-ud-navy mt-4 mb-1">Traits over time</h3>
      <div className="flex flex-wrap gap-3 mb-2 text-xs">
        {selectable.map((f) => (
          <span key={f}>
            <input type="checkbox" id={`trait-line-${f}`} checked={shown.includes(f)}
              onChange={() => toggle(f)} className="mr-1 align-middle" />
            <label htmlFor={`trait-line-${f}`} className="align-middle select-none text-ud-navy">
              {labels[f] ?? f}
            </label>
          </span>
        ))}
      </div>
      {shown.length === 0
        ? <p className="text-xs text-ud-gray">No traits selected.</p>
        : (
          <LineChart width={520} height={200} data={traitRows}>
            <CartesianGrid stroke="#f3f4f6" vertical={false} />
            <XAxis dataKey="date" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            {shown.map((f, i) => (
              <Line key={f} type="linear" dataKey={f} name={labels[f] ?? f}
                stroke={TRAIT_COLORS[i % TRAIT_COLORS.length]} strokeWidth={2} dot={{ r: 3 }} />
            ))}
          </LineChart>
        )}
      <p className="text-xs text-ud-gray mt-1">Each point is one outing's average.</p>
    </div>
  );
}
```

`ResponsiveContainer` is imported but the charts use fixed widths so jsdom can measure them; drop the unused import rather than leaving it (oxlint's `noUnusedLocals` will flag it).

- [ ] **Step 4: Wire it into the page**

In `src/pages/PitcherPage.tsx`, compute the top traits from the same attribution the waterfall uses, so the two panels agree, and replace the `panel-trend` placeholder:

```tsx
  const topTraits = useMemo(() => {
    if (!artifact) return [];
    return waterfallRows({
      values: row.typical, artifact, baselineZ: artifact.populationMeanZ,
      featureOrder: model.featureOrder, labels: model.labels,
      values2: row.typical, percentiles: row.percentiles,
    }).filter((r) => !r.grouped).map((r) => r.key);
  }, [artifact, row, model]);
```

```tsx
        <div data-testid="panel-trend" className="lg:col-span-2">
          <h2 className="font-display text-lg text-ud-navy mb-2">Trend</h2>
          <TrendPanel page={page} type={type} typeLabel={row.label}
            sampleFloor={artifact?.sampleFloor ?? 0} featureOrder={model.featureOrder}
            labels={model.labels} topTraits={topTraits} />
        </div>
```

`artifact`, `row`, and `model` are computed after the early returns, so `topTraits` must be declared after them — a `useMemo` after a conditional return violates the hooks rule. Move the early returns' bodies into the render path instead: compute `artifact`/`row`/`type` guarded by `data`, keep all hooks above the returns. Concretely, hoist the derivations into one `useMemo` that returns `null` when `data` is undefined, and keep the `isLoading`/`isError` returns after every hook call.

- [ ] **Step 5: Run everything**

Run: `npx vitest run` — Expected: all PASS.
Run: `npx tsc -b && npx oxlint` — Expected: clean, including the react-hooks rule.

- [ ] **Step 6: Commit**

```bash
git add src/components/pitcher/TrendPanel.tsx src/components/pitcher/TrendPanel.test.tsx src/pages/PitcherPage.tsx
git commit -m "Chart Stuff+ and its top traits by outing"
```

---

## Task 10: Application Insights and an error boundary

**Repo:** `ud-athletics-baseball-pitching`, plus one Azure app setting.

**Why:** an org hard gate, and the direct reason the bundle API being dead for two weeks produced no signal. `src/services/appInsights.ts` exists but is imported nowhere.

**Files:**
- Modify: `src/services/appInsights.ts`
- Create: `src/services/appInsights.test.ts`
- Create: `src/components/ErrorBoundary.tsx`
- Create: `src/components/ErrorBoundary.test.tsx`
- Modify: `src/main.tsx`
- Modify: `api/src/functions/bundle.js` (log failures with enough detail to be findable)

**Interfaces:**
- Produces: `resolveAuthenticatedUser(): Promise<string | null>` in `appInsights.ts` — fetches `/.auth/me`, returns `userDetails` or null; never throws.
- Produces: `ErrorBoundary` — a class component with `{ children }`, rendering a plain fallback and calling `appInsights?.trackException` on catch.

- [ ] **Step 1: Write the failing tests**

Create `src/services/appInsights.test.ts`:

```ts
import { describe, it, expect, vi, afterEach } from 'vitest';
import { resolveAuthenticatedUser } from './appInsights';

describe('resolveAuthenticatedUser', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('returns the signed-in email from the SWA auth endpoint', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ clientPrincipal: { userDetails: 'coach@udel.edu' } }),
    } as Response)));
    await expect(resolveAuthenticatedUser()).resolves.toBe('coach@udel.edu');
  });

  it('returns null when nobody is signed in', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({
      ok: true, json: () => Promise.resolve({ clientPrincipal: null }),
    } as Response)));
    await expect(resolveAuthenticatedUser()).resolves.toBeNull();
  });

  it('never throws when the endpoint is unavailable, since telemetry must not break the page', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('offline'))));
    await expect(resolveAuthenticatedUser()).resolves.toBeNull();
  });
});
```

Create `src/components/ErrorBoundary.test.tsx`:

```tsx
import { describe, it, expect, vi, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import ErrorBoundary from './ErrorBoundary';

function Boom(): never { throw new Error('kaboom'); }

describe('ErrorBoundary', () => {
  afterEach(() => vi.restoreAllMocks());

  it('renders a plain fallback instead of a white screen', () => {
    vi.spyOn(console, 'error').mockImplementation(() => {});
    render(<ErrorBoundary><Boom /></ErrorBoundary>);
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
  });

  it('renders children when nothing throws', () => {
    render(<ErrorBoundary><p>fine</p></ErrorBoundary>);
    expect(screen.getByText('fine')).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `npx vitest run src/services src/components/ErrorBoundary.test.tsx`
Expected: FAIL — `resolveAuthenticatedUser` and `ErrorBoundary` do not exist.

- [ ] **Step 3: Implement**

Append to `src/services/appInsights.ts`:

```ts
/**
 * The signed-in user's email, from the Static Web Apps auth endpoint.
 *
 * Telemetry must never break the page, so every failure path returns null rather
 * than throwing. The app is infra-gated by AAD; this call exists to attribute
 * telemetry to a user, not to enforce access.
 */
export async function resolveAuthenticatedUser(): Promise<string | null> {
  try {
    const res = await fetch('/.auth/me');
    if (!res.ok) return null;
    const body = (await res.json()) as { clientPrincipal?: { userDetails?: string } | null };
    return body.clientPrincipal?.userDetails ?? null;
  } catch {
    return null;
  }
}
```

Create `src/components/ErrorBoundary.tsx`:

```tsx
import { Component, type ErrorInfo, type ReactNode } from 'react';
import { appInsights } from '../services/appInsights';

interface State { failed: boolean }

export default class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    appInsights?.trackException({ exception: error, properties: { componentStack: info.componentStack } });
  }

  render(): ReactNode {
    if (this.state.failed) {
      return (
        <p className="text-ud-error p-6">
          Something went wrong. Try refreshing; if it keeps happening, let Jack know.
        </p>
      );
    }
    return this.props.children;
  }
}
```

Rewrite `src/main.tsx`:

```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import { appInsights, resolveAuthenticatedUser, setAuthenticatedUser } from './services/appInsights';
import './styles/brand.css';

// Importing the service is what loads App Insights; it was previously imported
// nowhere, so the app recorded zero telemetry from launch until 2026-08.
if (appInsights) {
  appInsights.trackPageView();
  void resolveAuthenticatedUser().then((email) => { if (email) setAuthenticatedUser(email); });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary><App /></ErrorBoundary>
  </React.StrictMode>,
);
```

In `api/src/functions/bundle.js`, the existing `context.error` line already names the path; add the status so a 404 storm is distinguishable from a 500 in the logs:

```js
    context.error(`bundle fetch failed for ${path}: status=${err.statusCode ?? 'none'} ${err.message}`);
```

- [ ] **Step 4: Run tests**

Run: `npx vitest run` — Expected: all PASS.
Run: `cd api && npm test && cd ..` — Expected: the API suite passes.
Run: `npx tsc -b && npx oxlint` — Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/main.tsx src/services src/components/ErrorBoundary.tsx src/components/ErrorBoundary.test.tsx api/src/functions/bundle.js
git commit -m "Turn on telemetry and stop render failures from blanking the page"
```

- [ ] **Step 6: Add the function-side connection string in Azure**

The frontend's connection string is a build-time `VITE_*` variable injected by the GitHub workflow. The managed functions run server-side and need a runtime app setting. Run this from Git Bash, calling `az` by its full path so it does not shadow python (per the project's environment notes), and note that `MSYS2_ARG_CONV_EXCL` must be exported for any `/subscriptions/...` argument:

```bash
export MSYS2_ARG_CONV_EXCL="/subscriptions"
CS=$(/c/az/venv/Scripts/az monitor app-insights component show \
  --app appi-ud-athletics-baseball-pitching \
  --resource-group rg-ud-athletics-baseball-pitching \
  --query connectionString -o tsv)
/c/az/venv/Scripts/az staticwebapp appsettings set \
  --name swa-ud-athletics-baseball-pitching \
  --resource-group rg-ud-athletics-baseball-pitching \
  --setting-names "APPLICATIONINSIGHTS_CONNECTION_STRING=$CS"
```

This adds a setting to an existing resource and creates nothing billable, so it needs no cost approval. Verify with:

```bash
/c/az/venv/Scripts/az staticwebapp appsettings list \
  --name swa-ud-athletics-baseball-pitching \
  --resource-group rg-ud-athletics-baseball-pitching -o table
```

`STORAGE_CONNECTION_STRING` and `BUNDLE_CONTAINER` must still be listed. **`appsettings set` on Static Web Apps replaces the whole set on some CLI versions** — capture the existing list first and re-supply every name if the verify step shows anything missing. Report the before/after lists in the task report.

- [ ] **Step 7: Commit nothing further**

Step 6 changes Azure, not the repo. Record the outcome in the task report.

---

## Task 11: Verify against real data and a real build

**Repo:** both. This task writes one document and no product code.

**Why:** every defect the data-layer plan shipped past its unit tests was caught by a real run or a whole-branch read, and each sat in a seam between components. The fixtures in this plan encode intended values; this task is where the page meets the actual bundle.

**Files:**
- Create: `docs/superpowers/plans/2026-08-05-pitcher-page-frontend-verification.md` (in `baseball-stuff-plus`)

- [ ] **Step 1: Produce a real bundle locally**

From `baseball-stuff-plus`, with the system python (not the az venv):

```bash
export STUFFPLUS_DATA=C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv
export STUFFPLUS_WORKDIR=C:/Users/jackdav/stuffplus_replication/workdir_webapp
export STUFFPLUS_YEARS=2025,2026
python -m webapp_publisher.publish --dry-run
```

Expected: 22-ish files under `$STUFFPLUS_WORKDIR/bundle/`, including `manifest.json`, `staff_board.json`, `model_artifacts.json`, `location_maps.json`, and `pitchers/*.json`. Both validators pass.

- [ ] **Step 2: Check the seams this plan created**

Write a throwaway script under the scratchpad directory (not the repo) that loads the dry-run bundle and asserts:

1. Every `staff_board.json` row has a `pitcherId`, and every non-null one names a file that exists under `pitchers/`.
2. `manifest.json`'s `pitchers` index and the set of `pitchers/*.json` files agree exactly, both directions.
3. For every pitcher and every arsenal row: the mean of that type's per-pitch `g` equals the row's `stuff` to within `1e-9`. **This is the scale-coherence equality**, on real data rather than a fixture.
4. For every pitcher and every arsenal row with a model: `contributions(typical, artifact, populationMeanZ)` summed equals `stuff - 100` to within `1e-9`. That is the additivity claim as the page actually renders it, since the page's baseline is the population mean and the population mean grades 100 by construction.
5. Every `f` array, `typical`, and `percentiles` has `len(featureOrder)` entries.
6. No non-`FF` arsenal row has a non-null `loc`; every `FF` row's `loc` is inside 40–160.
7. Every count key referenced by any pitch's `c` is either present in `location_maps.json` or absent — record which counts are missing, since the page falls back to `pooled` for those and the fallback should be exercised, not theoretical.

Record every measured number. If item 3 or 4 fails, stop and report it: that is a Critical, and the fault is in this plan or the data layer, not in the page's styling.

- [ ] **Step 3: Build and view the page against that bundle**

The live app is AAD-gated and headless browsers cannot sign in, so verify locally. From `ud-athletics-baseball-pitching`:

```bash
npm run build
```

Serve `dist/` with the dry-run bundle mounted at `/api/bundle/`. Any static server that can alias a directory works; a three-line node script in the scratchpad is fine. Then open the staff board, click a pitcher, and check:

- The arsenal shows every graded type with plain labels, usage summing to 100%.
- The fastball row shows a Location+; every secondary row shows an em-dash with its tooltip.
- A `recentChange` of null renders blank, not `0`.
- The waterfall's total matches the arsenal row's Stuff+ minus 100.
- Clicking a plotted pitch rebaselines the waterfall and drops the percentiles.
- Switching the count changes both the dots and the surface.
- Selecting a secondary type removes the surface and shows the descriptive note.
- The trend chart draws, and a thin type says so.

Capture screenshots into the scratchpad (not the repo) for the write-up.

- [ ] **Step 4: Read the page as the coach**

Dispatch the `consumer-coach-baseball` agent against the rendered page or its screenshots. Its job is to catch what would lose a real coach: buried takeaways, jargon, unreadable charts, no clear action. Record its findings verbatim in the verification document, and fix anything that is a label or copy defect (those are cheap and in scope). Anything that asks for a new measure or a new panel is out of scope for this plan — record it as follow-up, do not build it.

- [ ] **Step 5: Write the verification document**

Create `docs/superpowers/plans/2026-08-05-pitcher-page-frontend-verification.md` in `baseball-stuff-plus`, following the structure of `2026-08-05-pitcher-page-verification.md`: run configuration, every measured number, the constraint checks with their actual values, what the real bundle caught that the fixtures did not, the coach review's findings and which were fixed, and a deferred list. Include the same discipline note: one run, treat the numbers as a first pass until reproduced.

Be specific about anything that passed for the wrong reason. The data-layer verification's most useful paragraph was the one admitting a check had passed while the data was wrong.

- [ ] **Step 6: Commit**

In `baseball-stuff-plus`:

```bash
git add docs/superpowers/plans/2026-08-05-pitcher-page-frontend-verification.md
git commit -m "Record what the pitcher page looks like on real data"
```

---

## Deferred, and named rather than silently dropped

- **Per-type handedness-excluded grades.** The pitcher page has no handedness toggle because the bundle ships no per-type `stuffNoHand`, and it cannot be derived in the browser (see Deviations). Closing it means the data layer emitting `stuffNoHand` and `populationMeanZNoHand` per pitch type, the way `08_staff_scores.py` does for the four-seam board.
- **Secondary-pitch sample floors are unmeasured** and reuse the four-seam floor of 100. The page therefore flags secondary types as thin using a stand-in number. Measuring them is script-06 follow-up.
- **Splitter's display scale rests on 19 qualifying pitchers.** A splitter grade is technically on the same scale as everything else but rests on a thin reference population. The page does not currently say so; a per-type `nQualified` note is the cheap fix and is already in the bundle.
- **`build_grids` selects training rows with `year == 2024`** rather than `!= season_year`, carried over from the data-layer plan. Not broken today; the same latent hazard that produced that plan's first Critical.
- **The refresh chain re-runs the whole target build on a publish blip**, since the retry wraps every stage.
- **`recentChange`'s `asof - 30` boundary is untested** on the Python side.

---

## Self-review notes

Checked against the spec section by section.

- Arsenal summary, both panels' baselines, the fastball-only Location+ rule, the count selector, the Stuff+-colored dots with their labeling caveat, the trend's top-contributor defaults and thin-sample rendering, the browser-side attribution with its cross-language fixture, the additivity and scale-coherence equalities, the secondary-Location+ correctness test, App Insights on both sides: each maps to a task above.
- The spec's refresh-architecture section was implemented by the data-layer plan (Task 8 there) and is not re-planned here.
- The spec's Decision 2 open question — the observed spread of individual pitch grades — was measured during the data-layer plan (8.9 to 16.0 points, tighter than the pitcher-level 15), so its fallback is not needed and no task implements one.
- One spec requirement is deliberately not implemented: the handedness toggle on the pitcher page. It is documented in Deviations with the reason and the follow-up.
- Type names used in later tasks (`ArsenalRow`, `PitchRow`, `TypeArtifact`, `PitchType`, `WaterfallRow`) all trace to Task 3 or Task 4. `getJson` is exported from `usePitcherPage.ts` rather than duplicated. `pitchesFor`/`outingsFor`/`traitSeries` are defined in Task 5 and consumed in Tasks 8 and 9.
- Task 9 flags a real hazard it creates: adding a `useMemo` after `PitcherPage`'s early returns breaks the hooks rule. The step says how to resolve it rather than leaving the implementer to discover it at lint time.
