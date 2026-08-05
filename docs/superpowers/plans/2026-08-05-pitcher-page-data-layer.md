# Pitcher Page Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and publish the JSON bundle that the pitcher development page consumes — per-pitcher arsenal grades, per-pitch rows, per-outing trends, count-conditioned location maps, and the model artifact — on a schedule that picks up new games automatically.

**Architecture:** A new importable module `component_model/analysis/arsenal.py` holds all the math (per-pitch-type ridge fitting, one shared display transform, additive attribution, percentiles, per-outing aggregation) and is unit-tested against synthetic panels with known ground truth. A new script `14_pitcher_pages.py` composes that module into `pitcher_pages.json`. The publisher gains a bundle builder for the new files and a scorer stage; `run_refresh.ps1` gains the target-pipeline stage that currently does not exist in the scheduled path.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn (Ridge + StandardScaler pipeline), pytest. Azure Blob upload via `azure-storage-blob`. PowerShell 5+ for the scheduled wrapper.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-05-pitcher-development-page-design.md`. Read it before starting.
- **Do not modify** `component_model/analysis/fair_criterion.py` or `component_model/analysis/08_staff_scores.py`. Both are fixed references feeding the live Staff Board; changing them invalidates published comparisons. This plan only imports and reads them.
- **Do not modify** `component_model/portal/build_portal_data.py`. Its per-type approach is the model to copy, not to refactor.
- **Sign convention:** `Target`, `ridge_pred`, `adjT`, and location-map `v` are expected runs from the pitcher's perspective, where **lower is better**. Negation to higher-is-better happens exactly once, inside `arsenal.to_display`. Never negate twice.
- **One display transform.** Every Stuff+ number at every aggregation level (pitch, outing, pitch type, pitcher) goes through `arsenal.to_display`. Introducing a second scale is a spec violation.
- **Location+ is fastball-only.** Never emit a `loc` value for a non-FF pitch type.
- **Level II data.** Never commit `Final_Target_Calc_*.csv`, pitch caches, `staff_scores.json`, `pitcher_pages.json`, or any bundle output. Test fixtures use synthetic pitcher names only (`"Test-Pitcher, Alpha"` style, matching the existing frontend fixtures).
- **Pitch type names** are exactly: `FF`, `Slider`, `ChangeUp`, `Curveball`, `Sinker`, `Cutter`, `Splitter`. FF is identified by `df["is_ff"]` (which covers the three source spellings `Fastball`, `FourSeamFastBall`, `FourSeamFastball`); the others by `TaggedPitchType.isin(...)` with `Sinker` covering `{"Sinker", "TwoSeamFastBall"}`.
- **The twelve model features, in order** (`fc.FEATS`): `SpinRate`, `Extension`, `HorzBreak`, `InducedVertBreak`, `EffectiveVelo`, `RelHeight`, `RelSide`, `vertbreakdiff`, `horzbreakdiff`, `velocity_differential`, `is_lhp`, `is_lhb`.
- Analysis tests run: `cd component_model/analysis && python -m pytest tests/ -v`
- Publisher tests run: `python -m pytest webapp_publisher/tests/ -v` (from repo root)
- Commit after every task. Never chain `git commit` and `git push` in one command (repo hook enforces this). Commit messages lead with why, never narrate the diff, no AI attribution.

---

### Task 1: `arsenal.py` — the shared display transform

**Files:**
- Create: `component_model/analysis/arsenal.py`
- Test: `component_model/analysis/tests/test_arsenal.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `to_display(value, mu, sd) -> float | np.ndarray`, `display_scale(pitcher_means, floor_mask) -> tuple[float, float]`, and module constant `PITCH_TYPES: list[tuple[str, set[str] | None]]`.

- [ ] **Step 1: Write the failing test**

Create `component_model/analysis/tests/test_arsenal.py`:

```python
"""Unit tests for arsenal.py.

Synthetic-recovery style, matching test_reliability_curves.py: build data with
known properties, assert the estimator recovers them. Nothing in real data
reveals the true display scale, so these are the only correctness check.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arsenal as ar


def test_to_display_negates_the_run_value_convention():
    """Lower expected runs must map to a HIGHER display score."""
    better = ar.to_display(-0.02, mu=0.0, sd=0.01)
    worse = ar.to_display(0.02, mu=0.0, sd=0.01)
    assert better > 100 > worse


def test_to_display_puts_one_sd_at_fifteen_points():
    assert ar.to_display(-0.01, mu=0.0, sd=0.01) == pytest.approx(115.0)
    assert ar.to_display(0.01, mu=0.0, sd=0.01) == pytest.approx(85.0)


def test_to_display_commutes_with_averaging():
    """The whole point of one affine scale: per-pitch grades must average to
    the grade of the average pitch, exactly. This is the property that keeps
    pitch, outing, type, and pitcher numbers additive and mutually consistent.
    Asserted exactly, not approximately -- affine maps have no slack here.
    """
    values = np.array([-0.031, 0.004, 0.017, -0.009, 0.022])
    mu, sd = 0.002, 0.013
    mean_of_grades = ar.to_display(values, mu, sd).mean()
    grade_of_mean = ar.to_display(values.mean(), mu, sd)
    assert mean_of_grades == pytest.approx(grade_of_mean, abs=1e-12)


def test_display_scale_uses_only_rows_above_the_floor():
    """Pitchers below the sample floor must not influence the scale, or a few
    tiny-sample outliers would widen sd and compress everyone's score.
    """
    means = np.array([0.0, 0.01, -0.01, 5.0])
    floor_mask = np.array([True, True, True, False])
    mu, sd = ar.display_scale(means, floor_mask)
    assert mu == pytest.approx(0.0)
    assert sd == pytest.approx(np.std([0.0, 0.01, -0.01], ddof=1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'arsenal'`

- [ ] **Step 3: Write minimal implementation**

Create `component_model/analysis/arsenal.py`:

```python
"""Per-pitch-type Stuff+ grading, shared by the pitcher-page scorer.

SIGN CONVENTION: ridge_pred is expected runs from the pitcher's perspective,
LOWER = better. to_display() is the single place that negation happens. Do not
negate anywhere else.

ONE SCALE: every Stuff+ number the pitcher page shows -- for a single pitch, an
outing, a pitch type, or a pitcher -- goes through to_display() with the same
(mu, sd) for that pitch type. Because to_display is affine, per-pitch grades
average exactly to the grade of the average pitch, so the numbers stay additive
and a coach can check them by addition. Introducing a second scale calibrated on
a different population breaks that and is a spec violation.

The per-type model protocol here follows component_model/portal/build_portal_data.py,
whose arsenal grade was adopted 2026-07-23 after beating FF-only on both D1 year
pairs. That script is left untouched; this module re-expresses the same protocol
in testable form.
"""
from __future__ import annotations

import numpy as np

# (display name, TaggedPitchType values). None means "use the frame's is_ff flag",
# which already covers the three source spellings of four-seam.
PITCH_TYPES: list[tuple[str, set[str] | None]] = [
    ("FF", None),
    ("Slider", {"Slider"}),
    ("ChangeUp", {"ChangeUp"}),
    ("Curveball", {"Curveball"}),
    ("Sinker", {"Sinker", "TwoSeamFastBall"}),
    ("Cutter", {"Cutter"}),
    ("Splitter", {"Splitter"}),
]

DISPLAY_CENTER = 100.0
DISPLAY_SPREAD = 15.0


def to_display(value, mu: float, sd: float):
    """Map an expected-run value onto the 100 +/- 15 display scale.

    Accepts a scalar or an array. Affine by construction, so it commutes with
    averaging -- see test_to_display_commutes_with_averaging.
    """
    if sd <= 0:
        raise ValueError(f"display sd must be positive, got {sd}")
    return DISPLAY_CENTER - DISPLAY_SPREAD * (np.asarray(value, dtype=float) - mu) / sd


def display_scale(pitcher_means, floor_mask) -> tuple[float, float]:
    """Population moments for the display scale, from qualifying pitchers only.

    pitcher_means: one mean ridge_pred per pitcher for a single pitch type.
    floor_mask: boolean, True where that pitcher cleared the sample floor.
    """
    vals = np.asarray(pitcher_means, dtype=float)[np.asarray(floor_mask, dtype=bool)]
    if vals.size < 2:
        raise ValueError(f"need 2+ qualifying pitchers to set a scale, got {vals.size}")
    return float(vals.mean()), float(vals.std(ddof=1))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/arsenal.py component_model/analysis/tests/test_arsenal.py
git commit -m "Add the single display transform the pitcher page grades everything with

One affine scale per pitch type, so a pitch, an outing, a pitch type and a
pitcher all read on the same number and stay additive. The commuting test is
what keeps a second, differently-calibrated scale from creeping back in."
```

---

### Task 2: additive attribution and percentiles

**Files:**
- Modify: `component_model/analysis/arsenal.py`
- Test: `component_model/analysis/tests/test_arsenal.py`

**Interfaces:**
- Consumes: `to_display`, `display_scale` from Task 1.
- Produces: `contributions(feature_values, scaler_mean, scaler_scale, coef, baseline_z, sd) -> np.ndarray` (one value per feature, in Stuff+ points) and `percentile(reference_values, value) -> int`.

- [ ] **Step 1: Write the failing test**

Append to `component_model/analysis/tests/test_arsenal.py`:

```python
def _toy_model(n_feats=4, seed=3):
    """A standardizer + linear model whose parameters we control exactly."""
    rng = np.random.default_rng(seed)
    scaler_mean = rng.normal(0, 1, n_feats)
    scaler_scale = rng.uniform(0.5, 2.0, n_feats)
    coef = rng.normal(0, 0.01, n_feats)
    return scaler_mean, scaler_scale, coef


def test_contributions_sum_to_the_display_gap_exactly():
    """The load-bearing property. Ridge on standardized features is linear, so
    the per-trait contributions must account for the ENTIRE difference in Stuff+
    between the subject and its baseline -- no residual, no rounding slack.
    Exact equality, because any tolerance here would hide a real bug.
    """
    scaler_mean, scaler_scale, coef = _toy_model()
    sd = 0.02
    subject = np.array([1.4, -0.3, 0.8, 2.1])
    baseline = np.array([0.2, 0.1, -0.4, 0.9])

    baseline_z = (baseline - scaler_mean) / scaler_scale
    contrib = ar.contributions(subject, scaler_mean, scaler_scale, coef, baseline_z, sd)

    # Ridge prediction is intercept + z @ coef, so the intercept cancels in a gap.
    subject_pred = ((subject - scaler_mean) / scaler_scale) @ coef
    baseline_pred = baseline_z @ coef
    display_gap = ar.to_display(subject_pred, 0.0, sd) - ar.to_display(baseline_pred, 0.0, sd)

    assert contrib.sum() == pytest.approx(display_gap, abs=1e-10)


def test_contributions_are_zero_when_subject_equals_baseline():
    scaler_mean, scaler_scale, coef = _toy_model()
    subject = np.array([1.4, -0.3, 0.8, 2.1])
    baseline_z = (subject - scaler_mean) / scaler_scale
    contrib = ar.contributions(subject, scaler_mean, scaler_scale, coef, baseline_z, 0.02)
    assert np.allclose(contrib, 0.0, atol=1e-12)


def test_contribution_sign_follows_the_display_convention():
    """A feature that LOWERS expected runs must show a POSITIVE contribution,
    because the display scale is higher-is-better.
    """
    scaler_mean = np.array([0.0])
    scaler_scale = np.array([1.0])
    coef = np.array([-0.01])  # more of this feature => fewer runs => better
    contrib = ar.contributions(np.array([2.0]), scaler_mean, scaler_scale, coef,
                               baseline_z=np.array([0.0]), sd=0.02)
    assert contrib[0] > 0


def test_percentile_ranks_against_the_reference_population():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert ar.percentile(ref, 0.5) == 0
    assert ar.percentile(ref, 2.5) == 50
    assert ar.percentile(ref, 5.0) == 100


def test_percentile_ignores_missing_reference_values():
    ref = np.array([1.0, np.nan, 3.0, np.nan])
    assert ar.percentile(ref, 2.0) == 50
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: FAIL with `AttributeError: module 'arsenal' has no attribute 'contributions'`

- [ ] **Step 3: Write minimal implementation**

Append to `component_model/analysis/arsenal.py`:

```python
def contributions(feature_values, scaler_mean, scaler_scale, coef, baseline_z, sd):
    """Per-feature contribution to Stuff+, in display points.

    This is the formula already used in 08_staff_scores.py, kept identical so the
    Staff Board and the pitcher page explain a grade the same way:

        z            = (value - scaler_mean) / scaler_scale
        contribution = -15 * (z - baseline_z) * coef / sd

    baseline_z is the standardized baseline the gap is measured against: the
    qualified-population mean for the default view, or the pitcher's own typical
    pitch when a single pitch is selected.

    Because the model is linear in standardized features, these sum exactly to the
    Stuff+ difference between subject and baseline.
    """
    if sd <= 0:
        raise ValueError(f"display sd must be positive, got {sd}")
    z = (np.asarray(feature_values, dtype=float) - np.asarray(scaler_mean, dtype=float)) / np.asarray(
        scaler_scale, dtype=float
    )
    return -DISPLAY_SPREAD * (z - np.asarray(baseline_z, dtype=float)) * np.asarray(coef, dtype=float) / sd


def percentile(reference_values, value) -> int:
    """Percentile rank of value within reference_values, 0-100.

    Reference is the qualifying population for one pitch type. NaNs are dropped
    rather than propagated, since a feature can be missing for some pitches.
    """
    ref = np.asarray(reference_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        raise ValueError("percentile needs a non-empty reference population")
    return int(round(100.0 * float((ref < value).mean())))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/arsenal.py component_model/analysis/tests/test_arsenal.py
git commit -m "Add additive trait attribution and percentile ranking

Reuses 08_staff_scores.py's exact contribution formula so the Staff Board and
the pitcher page explain a grade identically. The sum-to-the-gap test is an
equality rather than a tolerance, since the model is linear and any residual
would mean a real error."
```

---

### Task 3: per-outing and per-pitch-type aggregation

**Files:**
- Modify: `component_model/analysis/arsenal.py`
- Test: `component_model/analysis/tests/test_arsenal.py`

**Interfaces:**
- Consumes: `to_display` from Task 1.
- Produces: `outing_table(pitches, mu, sd) -> pd.DataFrame` with columns `date`, `n`, `stuff`; and `recent_change(outings, sd_floor_n, asof) -> float | None`.

- [ ] **Step 1: Write the failing test**

Append to `component_model/analysis/tests/test_arsenal.py`:

```python
import pandas as pd


def _toy_pitches():
    """Two dates, known ridge_pred values, so aggregates are hand-checkable."""
    return pd.DataFrame({
        "Date": ["2026-03-01", "2026-03-01", "2026-03-08", "2026-03-08", "2026-03-08"],
        "ridge_pred": [0.00, 0.02, -0.01, -0.03, 0.01],
    })


def test_outing_table_groups_by_date_and_grades_each_outing():
    out = ar.outing_table(_toy_pitches(), mu=0.0, sd=0.02)
    assert list(out["date"]) == ["2026-03-01", "2026-03-08"]
    assert list(out["n"]) == [2, 3]
    # First outing mean ridge_pred is 0.01 -> 100 - 15*(0.01/0.02) = 92.5
    assert out.loc[0, "stuff"] == pytest.approx(92.5)


def test_outing_grades_average_to_the_overall_grade_when_outings_are_equal_size():
    """Sanity check that outing grades live on the same scale as everything else."""
    pitches = pd.DataFrame({
        "Date": ["2026-03-01", "2026-03-01", "2026-03-08", "2026-03-08"],
        "ridge_pred": [0.00, 0.02, -0.01, -0.03],
    })
    out = ar.outing_table(pitches, mu=0.0, sd=0.02)
    overall = ar.to_display(pitches["ridge_pred"].mean(), 0.0, 0.02)
    assert out["stuff"].mean() == pytest.approx(overall, abs=1e-12)


def test_recent_change_is_none_when_a_window_is_below_the_floor():
    """A blank reads as 'not enough to say'; a zero would wrongly read as
    'no change'. So below the floor must return None, never 0.0.
    """
    outings = pd.DataFrame({
        "date": ["2026-01-05", "2026-03-01"],
        "n": [40, 5],
        "stuff": [110.0, 95.0],
    })
    assert ar.recent_change(outings, floor_n=30, asof="2026-03-10") is None


def test_recent_change_differences_the_two_thirty_day_windows():
    outings = pd.DataFrame({
        "date": ["2026-01-20", "2026-03-01"],   # prior window, then recent window
        "n": [50, 50],
        "stuff": [100.0, 112.0],
    })
    got = ar.recent_change(outings, floor_n=30, asof="2026-03-10")
    assert got == pytest.approx(12.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: FAIL with `AttributeError: module 'arsenal' has no attribute 'outing_table'`

- [ ] **Step 3: Write minimal implementation**

Add to the imports at the top of `component_model/analysis/arsenal.py`:

```python
import pandas as pd
```

Append to `component_model/analysis/arsenal.py`:

```python
RECENT_WINDOW_DAYS = 30


def outing_table(pitches: pd.DataFrame, mu: float, sd: float) -> pd.DataFrame:
    """One row per date the pitcher threw this pitch type.

    Grades each outing with the same transform used at every other level, so
    outing numbers are directly comparable to the pitch-type number.

    Date is normalized to a YYYY-MM-DD string first: load_pitches leaves the
    source Date column as-is, so it can arrive as either a string or a datetime,
    and a datetime would stringify with a spurious " 00:00:00" into the bundle.
    """
    dates = pd.to_datetime(pitches["Date"]).dt.strftime("%Y-%m-%d")
    g = pitches.assign(_date=dates).groupby("_date")["ridge_pred"].agg(["size", "mean"]).reset_index()
    g.columns = ["date", "n", "mean_ridge"]
    g = g.sort_values("date").reset_index(drop=True)
    g["stuff"] = to_display(g["mean_ridge"].values, mu, sd)
    return g[["date", "n", "stuff"]]


def recent_change(outings: pd.DataFrame, floor_n: int, asof: str) -> float | None:
    """Stuff+ over the trailing 30 days minus the 30 days before that.

    Returns None when either window is below the sample floor, so the UI can
    render a blank. A zero would be read as "no change", which is a different
    and wrong claim.
    """
    asof_ts = pd.Timestamp(asof)
    dates = pd.to_datetime(outings["date"])
    recent = outings[(dates > asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS)) & (dates <= asof_ts)]
    prior_lo = asof_ts - pd.Timedelta(days=2 * RECENT_WINDOW_DAYS)
    prior = outings[(dates > prior_lo) & (dates <= asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS))]
    if recent["n"].sum() < floor_n or prior["n"].sum() < floor_n:
        return None
    recent_mean = np.average(recent["stuff"].values, weights=recent["n"].values)
    prior_mean = np.average(prior["stuff"].values, weights=prior["n"].values)
    return float(recent_mean - prior_mean)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: PASS, 13 tests

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/arsenal.py component_model/analysis/tests/test_arsenal.py
git commit -m "Aggregate grades by outing and define the recent-change window

Outings grade through the same transform as everything else so the numbers stay
comparable. Recent change returns None rather than zero below the sample floor,
because a blank says 'not enough to tell' while a zero claims 'no change'."
```

---

### Task 4: per-type model fitting

**Files:**
- Modify: `component_model/analysis/arsenal.py`
- Test: `component_model/analysis/tests/test_arsenal.py`

**Interfaces:**
- Consumes: `PITCH_TYPES`, `display_scale` from Task 1.
- Produces: `type_mask(pit, tags) -> pd.Series` and `fit_type(pit, tags, floor_n, fc_module, season_year) -> dict` returning keys `pitches` (DataFrame with `ridge_pred`), `model`, `scaler_mean`, `scaler_scale`, `coef`, `mu`, `sd`, `population_mean_z`, `reference_features` (DataFrame of qualifying pitchers' feature means, used for percentile ranking), `n_qualified`.

- [ ] **Step 1: Write the failing test**

Append to `component_model/analysis/tests/test_arsenal.py`:

```python
def test_type_mask_uses_the_is_ff_flag_for_four_seams():
    """FF must come from is_ff, which already unifies the three source spellings,
    rather than from a literal string match that would silently drop two of them.
    """
    pit = pd.DataFrame({
        "TaggedPitchType": ["Fastball", "FourSeamFastBall", "FourSeamFastball", "Slider"],
        "is_ff": [True, True, True, False],
    })
    assert list(ar.type_mask(pit, None)) == [True, True, True, False]


def test_type_mask_treats_two_seam_as_sinker():
    pit = pd.DataFrame({
        "TaggedPitchType": ["Sinker", "TwoSeamFastBall", "Slider"],
        "is_ff": [False, False, False],
    })
    assert list(ar.type_mask(pit, {"Sinker", "TwoSeamFastBall"})) == [True, True, False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: FAIL with `AttributeError: module 'arsenal' has no attribute 'type_mask'`

- [ ] **Step 3: Write minimal implementation**

Append to `component_model/analysis/arsenal.py`:

```python
def type_mask(pit: pd.DataFrame, tags: set[str] | None) -> pd.Series:
    """Row mask selecting one pitch type.

    tags=None means four-seam, taken from the frame's is_ff flag because that
    flag already unifies the three spellings the source data uses.
    """
    if tags is None:
        return pit["is_ff"]
    return pit["TaggedPitchType"].isin(tags)


def fit_type(pit: pd.DataFrame, tags: set[str] | None, floor_n: int, fc_module, season_year: int) -> dict:
    """Fit the ridge for one pitch type and derive its display scale.

    Protocol copied from build_portal_data.py (arsenal grade, adopted 2026-07-23):
    one ridge per pitch type via fc.stuff_ridge(pitch_mask=...), then a display
    scale from that type's qualifying pitchers.

    season_year is the canonical year role to grade (fair_criterion relabels the
    year pair to 2024/2025 roles, so pass 2025 for the later season).

    Raises ValueError if the type has too few qualifying pitchers to scale.
    """
    mask = type_mask(pit, tags)
    pp, model = fc_module.stuff_ridge(pit, pitch_mask=mask, return_model=True)
    pp = pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()].copy()
    season = pp[pp["year"] == season_year].copy()

    per_pitcher = season.groupby("PitcherId")["ridge_pred"].agg(["size", "mean"])
    mu, sd = display_scale(per_pitcher["mean"].values, (per_pitcher["size"] >= floor_n).values)

    scaler = model.named_steps["standardscaler"]
    coef = model.named_steps["ridge"].coef_
    feats = fc_module.FEATS
    qualified = per_pitcher.index[per_pitcher["size"] >= floor_n]
    feature_means = season.groupby("PitcherId")[feats].mean()
    population_mean_z = ((feature_means.loc[qualified].values - scaler.mean_) / scaler.scale_).mean(axis=0)

    return {
        "pitches": season,
        "model": model,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "coef": coef,
        "mu": mu,
        "sd": sd,
        "population_mean_z": population_mean_z,
        "reference_features": feature_means.loc[qualified],
        "n_qualified": int(len(qualified)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd component_model/analysis && python -m pytest tests/test_arsenal.py -v`
Expected: PASS, 15 tests

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/arsenal.py component_model/analysis/tests/test_arsenal.py
git commit -m "Fit one ridge per pitch type, following the adopted arsenal protocol

Mirrors build_portal_data.py's per-type approach, which beat fastball-only on
both D1 year pairs, but in an importable and tested form rather than inline in
a standalone script. Four-seam selection goes through is_ff so none of the
three source spellings gets silently dropped."
```

---

### Task 5: `14_pitcher_pages.py` — compose the scorer output

**Files:**
- Create: `component_model/analysis/14_pitcher_pages.py`
- Modify: `component_model/analysis/README.md` (add the script-14 row)
- Test: `component_model/analysis/tests/test_pitcher_pages_output.py`

**Interfaces:**
- Consumes: everything from `arsenal.py` (Tasks 1-4).
- Produces: `<workdir>/pitcher_pages.json` with top-level keys `team`, `season`, `pitchTypes`, `pitchers`, `grids`, `model`. Consumed by Task 6's bundle builder.
- Module-level functions the tests call directly: `build_pitcher_records(fitted_by_type, feats, floor_n, asof, min_type_pitches=MIN_TYPE_PITCHES) -> list[dict]`, `build_model_artifact(fitted_by_type, feats) -> dict`, `build_grids(pit) -> dict`, `main() -> int`.

The output dict shape, which Task 6 depends on exactly:

```python
{
  "team": "DEL_BLU",
  "season": 2026,
  "pitchTypes": ["FF", "Slider", ...],          # types that had enough data to scale
  "model": {
    "featureOrder": [...12 names...],
    "byPitchType": {
      "FF": {"coef": [...], "scalerMean": [...], "scalerScale": [...],
             "populationMeanZ": [...], "displayMu": 0.0, "displaySd": 0.0,
             "sampleFloor": 100, "nQualified": 543},
    },
  },
  "grids": {"pooled": [{"x":..,"z":..,"v":..}], "0-0": [...], ...},
  "pitchers": [
    {
      "pitcherId": 1000123, "name": "Last, First", "hand": "R",
      "arsenal": [
        {"type": "FF", "n": 412, "usage": 0.58, "stuff": 124.0, "loc": 103.0,
         "recentChange": -6.2, "aboveFloor": True,
         "typical": [...12 raw values...], "percentiles": [...12 ints 0-100...]},
      ],
      "outings": [{"date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.0}],
      "pitches": [{"d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31,
                   "c": "0-2", "g": 131.0, "f": [...12 raw values...]}],
    },
  ],
}
```

`loc` is present only when `type == "FF"`; it is `None` for every other type.

- [ ] **Step 1: Write the failing test**

This task's script needs real licensed data to run end to end, so the test asserts the *contract properties* of an output dict rather than running the script. Create `component_model/analysis/tests/test_pitcher_pages_output.py`:

```python
"""Contract tests for 14_pitcher_pages.py's output shape.

The script itself needs licensed TrackMan data, so these tests validate the
assembly helpers against a synthetic frame instead of running the full script.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arsenal as ar

PAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "14_pitcher_pages.py")


def _load_pages_module():
    """Import the numerically-named script by path, since `import 14_...` is
    not valid Python syntax."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("pitcher_pages", PAGES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fitted():
    """A minimal fitted-type dict shaped like arsenal.fit_type's return."""
    feats = ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak", "EffectiveVelo",
             "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
             "velocity_differential", "is_lhp", "is_lhb"]
    n = 6
    rng = np.random.default_rng(11)
    pitches = pd.DataFrame({
        "PitcherId": [1, 1, 1, 2, 2, 2],
        "Pitcher": ["A, A"] * 3 + ["B, B"] * 3,
        "PitcherThrows": ["Right"] * 3 + ["Left"] * 3,
        "Date": ["2026-03-01", "2026-03-01", "2026-03-08"] * 2,
        "ridge_pred": [-0.01, 0.00, 0.01, 0.02, -0.02, 0.00],
        "PlateLocSide": rng.normal(0, 0.5, n),
        "PlateLocHeight": rng.normal(2.5, 0.5, n),
        "count12": ["0-0"] * n,
        "loc": [0.0] * n,
    })
    for f in feats:
        pitches[f] = rng.normal(0, 1, n)
    return feats, {
        "pitches": pitches,
        "scaler_mean": np.zeros(12), "scaler_scale": np.ones(12),
        "coef": rng.normal(0, 0.01, 12),
        "mu": 0.0, "sd": 0.02,
        "population_mean_z": np.zeros(12),
        "reference_features": pitches.groupby("PitcherId")[feats].mean(),
        "n_qualified": 2,
    }


def test_pitcher_records_carry_one_arsenal_row_per_type_with_data():
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    assert {r["pitcherId"] for r in records} == {1, 2}
    for r in records:
        assert [a["type"] for a in r["arsenal"]] == ["FF"]


def test_pitch_grades_average_to_the_arsenal_row_grade():
    """The additivity guarantee, asserted on real assembled output rather than
    on the transform in isolation."""
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    for r in records:
        row = r["arsenal"][0]
        grades = [p["g"] for p in r["pitches"] if p["t"] == "FF"]
        assert np.mean(grades) == pytest.approx(row["stuff"], abs=1e-9)


def test_secondary_types_never_carry_a_location_score():
    """Location+ is a fastball score. Emitting it for a slider would be a
    construct leak, so this is a correctness test, not a formatting preference.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"Slider": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    for r in records:
        assert r["arsenal"][0]["loc"] is None


def test_usage_shares_sum_to_one_per_pitcher():
    mod = _load_pages_module()
    feats, fitted = _fitted()
    other = {**fitted, "pitches": fitted["pitches"].copy()}
    records = mod.build_pitcher_records({"FF": fitted, "Slider": other}, feats,
                                        floor_n=1, asof="2026-03-10", min_type_pitches=1)
    for r in records:
        assert sum(a["usage"] for a in r["arsenal"]) == pytest.approx(1.0)


def test_a_pitch_type_below_the_minimum_is_dropped_entirely():
    """A three-pitch slider is not a graded pitch; including it would put a
    meaningless number in front of a coach.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=100)
    assert records == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd component_model/analysis && python -m pytest tests/test_pitcher_pages_output.py -v`
Expected: FAIL with `FileNotFoundError` on `14_pitcher_pages.py`

- [ ] **Step 3: Write minimal implementation**

Create `component_model/analysis/14_pitcher_pages.py`:

```python
"""Pitcher development page data: arsenal grades, pitch rows, outing trends.

Question: for one team's staff, how does each pitch type grade, why, where is it
being thrown, and is it moving?

Reads the same source as script 08 but grades the FULL ARSENAL (per-type ridge
models, the protocol adopted 2026-07-23) rather than four-seams only, and keeps
per-pitch rows instead of collapsing to pitcher means.

Writes <workdir>/pitcher_pages.json. Level II data: never commit that file.

SIGN CONVENTION: everything stays in pitcher's-perspective expected runs until
arsenal.to_display negates once. See arsenal.py's module docstring.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import arsenal as ar
import fair_criterion as fc

# Four-seam floor is script 06's measured value. Secondary floors are UNMEASURED;
# they reuse the four-seam number as a conservative stand-in. See the spec's
# "Honest gap" note. Do not present these as derived for non-FF types.
SAMPLE_FLOOR = 100
MIN_TYPE_PITCHES = 25   # skip a pitch type for a pitcher below this
SEASON_ROLE_YEAR = 2025  # fair_criterion relabels the year pair to 2024/2025 roles


def build_pitcher_records(fitted_by_type: dict, feats: list[str], floor_n: int, asof: str,
                          min_type_pitches: int = MIN_TYPE_PITCHES) -> list[dict]:
    """Assemble one record per pitcher from the per-type fitted results.

    min_type_pitches is a parameter rather than a module constant so tests can
    exercise the assembly on small synthetic frames.
    """
    all_ids: set = set()
    for state in fitted_by_type.values():
        all_ids.update(state["pitches"]["PitcherId"].unique())

    records = []
    for pid in sorted(all_ids):
        # First pass: which types clear the per-type minimum for this pitcher.
        # Usage is shared out over the INCLUDED types only, so the shares always
        # sum to 1 and read as "of the pitches we grade, this is the mix."
        included = []
        for tname, state in fitted_by_type.items():
            sub = state["pitches"]
            sub = sub[sub["PitcherId"] == pid]
            if len(sub) >= min_type_pitches:
                included.append((tname, state, sub))
        if not included:
            continue
        graded_total = sum(len(sub) for _, _, sub in included)

        arsenal_rows, outings, pitch_rows = [], [], []
        name = str(included[0][2]["Pitcher"].iloc[0])
        hand = str(included[0][2]["PitcherThrows"].iloc[0])[0]
        for tname, state, sub in included:
            mu, sd = state["mu"], state["sd"]
            per_outing = ar.outing_table(sub, mu, sd)
            change = ar.recent_change(per_outing, floor_n=floor_n, asof=asof)

            arsenal_rows.append({
                "type": tname,
                "n": int(len(sub)),
                "usage": float(len(sub) / graded_total),
                "stuff": float(ar.to_display(sub["ridge_pred"].mean(), mu, sd)),
                # Location+ is a fastball score only -- never emit it elsewhere.
                "loc": float(sub["loc"].mean()) if tname == "FF" else None,
                "recentChange": change,
                "aboveFloor": bool(len(sub) >= floor_n),
                "typical": [float(v) for v in sub[feats].mean().values],
                # Percentile of each of his typical trait values against the
                # qualifying pitchers for this type. Computed here rather than in
                # the browser because it needs the reference population, which is
                # far larger than the page and is not worth shipping.
                "percentiles": [
                    ar.percentile(state["reference_features"][f].values, float(sub[f].mean()))
                    for f in feats
                ],
            })
            for _, o in per_outing.iterrows():
                outings.append({"date": str(o["date"]), "type": tname,
                                "n": int(o["n"]), "stuff": float(o["stuff"])})
            grades = ar.to_display(sub["ridge_pred"].values, mu, sd)
            dates = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d").values
            for (_, p), g, d in zip(sub.iterrows(), grades, dates):
                pitch_rows.append({
                    "d": str(d), "t": tname,
                    "x": round(float(p["PlateLocSide"]), 3),
                    "z": round(float(p["PlateLocHeight"]), 3),
                    "c": str(p["count12"]), "g": float(g),
                    "f": [float(p[f]) for f in feats],
                })
        arsenal_rows.sort(key=lambda r: -r["usage"])
        records.append({"pitcherId": int(pid), "name": name, "hand": hand,
                        "arsenal": arsenal_rows, "outings": outings, "pitches": pitch_rows})
    return records


def build_model_artifact(fitted_by_type: dict, feats: list[str]) -> dict:
    return {
        "featureOrder": list(feats),
        "byPitchType": {
            tname: {
                "coef": [float(v) for v in s["coef"]],
                "scalerMean": [float(v) for v in s["scaler_mean"]],
                "scalerScale": [float(v) for v in s["scaler_scale"]],
                "populationMeanZ": [float(v) for v in s["population_mean_z"]],
                "displayMu": float(s["mu"]),
                "displaySd": float(s["sd"]),
                "sampleFloor": SAMPLE_FLOOR,
                "nQualified": s["n_qualified"],
            }
            for tname, s in fitted_by_type.items()
        },
    }


def build_grids(pit: pd.DataFrame) -> dict:
    """Count-conditioned run-value surface, same construction as script 08.

    The training frame needs location bins before PooledLocationMap can use it,
    and binning requires non-null plate coordinates -- script 08 filters and bins
    before constructing the map, so do the same here.
    """
    train = pit[pit["PlateLocSide"].notna() & pit["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(train)
    train = train[(train["year"] == 2024) & train["xT"].notna()]
    pooled = fc.PooledLocationMap(train)
    cmap = fc.CountLocationMap(train, "count12", 5)
    xs = np.arange(-1.25, 1.25, 0.25)
    zs = np.arange(1.0, 4.0, 0.25)
    grid = pd.DataFrame([(gx, gz) for gx in xs for gz in zs],
                        columns=["PlateLocSide", "PlateLocHeight"])
    grid["PlateLocSide"] += 0.01   # land inside the intended cell when binning
    grid["PlateLocHeight"] += 0.01
    fc.add_loc_bins(grid)
    pv = pooled.apply(grid)
    out = {"pooled": [{"x": round(float(a), 2), "z": round(float(b), 2), "v": round(float(v), 4)}
                      for a, b, v in zip(grid["gx"], grid["gz"], pv)]}
    for cnt in sorted(train["count12"].unique()):
        s = grid.copy()
        s["count12"] = cnt
        out[cnt] = [{"x": round(float(a), 2), "z": round(float(b), 2), "v": round(float(v), 4)}
                    for a, b, v in zip(grid["gx"], grid["gz"], cmap.apply(s, pv))]
    return out


def main() -> int:
    args = fc.paths()
    pit = fc.load_pitches(args)
    fc.add_xt(pit)
    fc.add_count_cols(pit)

    fitted = {}
    for tname, tags in ar.PITCH_TYPES:
        try:
            state = ar.fit_type(pit, tags, SAMPLE_FLOOR, fc, SEASON_ROLE_YEAR)
        except ValueError as err:
            print(f"skipping {tname}: {err}")
            continue
        # Location+ only matters for four-seams; attach it there and nowhere else.
        if tname == "FF":
            ff = state["pitches"]
            fc.add_loc_bins(ff)
            train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
            state["pitches"]["loc"] = fc.PooledLocationMap(train).apply(ff)
        else:
            state["pitches"]["loc"] = np.nan
        fitted[tname] = state
        print(f"{tname}: {len(state['pitches'])} pitches, {state['n_qualified']} qualified")

    if "FF" not in fitted:
        print("no four-seam model: cannot build pages")
        return 1

    team_ids = set(fitted["FF"]["pitches"].loc[
        fitted["FF"]["pitches"]["PitcherTeam"] == args.team, "PitcherId"].unique())
    for state in fitted.values():
        state["pitches"] = state["pitches"][state["pitches"]["PitcherId"].isin(team_ids)].copy()

    asof = str(pd.to_datetime(fitted["FF"]["pitches"]["Date"]).max().date())
    records = build_pitcher_records(fitted, fc.FEATS, SAMPLE_FLOOR, asof)
    print(f"{len(records)} pitchers on {args.team}")

    payload = {
        "team": args.team,
        "season": int(args.year_pair[1]),
        "pitchTypes": list(fitted.keys()),
        "model": build_model_artifact(fitted, fc.FEATS),
        "grids": build_grids(pit),
        "pitchers": records,
    }
    out = f"{args.workdir}/pitcher_pages.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd component_model/analysis && python -m pytest tests/test_pitcher_pages_output.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Run the whole analysis suite to confirm nothing regressed**

Run: `cd component_model/analysis && python -m pytest tests/ -v`
Expected: PASS, all tests (existing variance-components and reliability-curves tests plus the new ones)

- [ ] **Step 6: Add the script to the run-order table**

In `component_model/analysis/README.md`, add this row to the run-order table immediately after the script-13 row:

```markdown
| `14_pitcher_pages.py` | Pitcher development page data: full-arsenal grades, per-pitch rows, outing trends (`--team`) | 01 pass |
```

- [ ] **Step 7: Commit**

```bash
git add component_model/analysis/14_pitcher_pages.py component_model/analysis/tests/test_pitcher_pages_output.py component_model/analysis/README.md
git commit -m "Produce the pitcher development page dataset

Grades a staff's full arsenal rather than four-seams only, and keeps per-pitch
rows so the page can explain an individual pitch instead of only a season mean.
Script 08 is left untouched: it still feeds the Staff Board, and its numbers
stay the published reference.

The additivity test runs against assembled output, not just the transform, so
pitch grades are verified to average to the arsenal row a coach reads."
```

---

### Task 6: bundle builder for the new files

**Files:**
- Create: `webapp_publisher/build_pitcher_bundle.py`
- Test: `webapp_publisher/tests/test_build_pitcher_bundle.py`

**Interfaces:**
- Consumes: the `pitcher_pages.json` dict from Task 5.
- Produces: `build_pitcher_bundle(pages: dict) -> dict[str, dict]` keyed by blob name — `"location_maps.json"`, `"model_artifacts.json"`, and one `"pitchers/{pitcherId}.json"` per pitcher. Also `pitcher_index(pages) -> list[dict]` returning `[{"pitcherId", "name", "hand"}]` for the manifest.

- [ ] **Step 1: Write the failing test**

Create `webapp_publisher/tests/test_build_pitcher_bundle.py`:

```python
import pytest

from webapp_publisher.build_pitcher_bundle import build_pitcher_bundle, pitcher_index

PAGES = {
    "team": "DEL_BLU",
    "season": 2026,
    "pitchTypes": ["FF"],
    "model": {"featureOrder": ["SpinRate"], "byPitchType": {"FF": {
        "coef": [0.01], "scalerMean": [2200.0], "scalerScale": [180.0],
        "populationMeanZ": [0.0], "displayMu": 0.0, "displaySd": 0.02,
        "sampleFloor": 100, "nQualified": 400}}},
    "grids": {"pooled": [{"x": 0.0, "z": 2.5, "v": 0.01}]},
    "pitchers": [{
        "pitcherId": 1000123, "name": "Test-Pitcher, Alpha", "hand": "R",
        "arsenal": [{"type": "FF", "n": 412, "usage": 1.0, "stuff": 124.0,
                     "loc": 103.0, "recentChange": -6.2, "aboveFloor": True,
                     "typical": [2350.0], "percentiles": [78]}],
        "outings": [{"date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.0}],
        "pitches": [{"d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31,
                     "c": "0-2", "g": 131.0, "f": [2350.0]}],
    }],
}


def test_bundle_has_one_file_per_pitcher_plus_the_shared_files():
    out = build_pitcher_bundle(PAGES)
    assert set(out) == {"location_maps.json", "model_artifacts.json",
                        "pitchers/1000123.json"}


def test_pitcher_file_is_keyed_by_trackman_id_not_a_positional_index():
    """Positional ids shift when the roster changes, which would repoint every
    pitcher's URL. TrackMan's PitcherId is stable across seasons.
    """
    out = build_pitcher_bundle(PAGES)
    assert out["pitchers/1000123.json"]["pitcherId"] == 1000123


def test_pitcher_file_carries_arsenal_outings_and_pitches():
    body = build_pitcher_bundle(PAGES)["pitchers/1000123.json"]
    assert body["arsenal"][0]["stuff"] == 124.0
    assert body["outings"][0]["date"] == "2026-03-15"
    assert body["pitches"][0]["g"] == 131.0


def test_model_artifact_preserves_feature_order():
    """Every per-pitch feature array is positional against this list, so a
    reordering here silently mislabels every trait on the page.
    """
    out = build_pitcher_bundle(PAGES)
    assert out["model_artifacts.json"]["featureOrder"] == ["SpinRate"]


def test_model_artifact_includes_plain_english_labels_for_every_feature():
    out = build_pitcher_bundle(PAGES)
    labels = out["model_artifacts.json"]["labels"]
    for feat in out["model_artifacts.json"]["featureOrder"]:
        assert feat in labels and labels[feat] != feat


def test_pitcher_index_is_manifest_sized_not_the_whole_payload():
    idx = pitcher_index(PAGES)
    assert idx == [{"pitcherId": 1000123, "name": "Test-Pitcher, Alpha", "hand": "R"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp_publisher/tests/test_build_pitcher_bundle.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'webapp_publisher.build_pitcher_bundle'`

- [ ] **Step 3: Write minimal implementation**

Create `webapp_publisher/build_pitcher_bundle.py`:

```python
"""Transform 14_pitcher_pages.py's output into the pitcher-page bundle files.

Input is the dict written by component_model/analysis/14_pitcher_pages.py.
Scores are already on the 100±15 display scale (higher = better); do not re-flip.
"""
from __future__ import annotations

# Absolute import, matching publish.py's existing style in this package.
from webapp_publisher.build_bundle import to_native

# Plain-English labels for the coach-facing page. A label that needs a glossary
# entry to be legible is a label defect -- fix the label, not the glossary.
FEATURE_LABELS = {
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
    "is_lhb": "Batter hits left",
}

PITCH_TYPE_LABELS = {
    "FF": "Fastball",
    "Slider": "Slider",
    "ChangeUp": "Changeup",
    "Curveball": "Curveball",
    "Sinker": "Sinker",
    "Cutter": "Cutter",
    "Splitter": "Splitter",
}


def pitcher_index(pages: dict) -> list[dict]:
    """Small index for the manifest, so routing does not need every pitcher file."""
    return [{"pitcherId": p["pitcherId"], "name": p["name"], "hand": p["hand"]}
            for p in pages["pitchers"]]


def build_pitcher_bundle(pages: dict) -> dict[str, dict]:
    model = dict(pages["model"])
    missing = [f for f in model["featureOrder"] if f not in FEATURE_LABELS]
    if missing:
        raise ValueError(f"no plain-English label for features {missing}")
    model["labels"] = {f: FEATURE_LABELS[f] for f in model["featureOrder"]}

    files: dict[str, dict] = {
        "location_maps.json": to_native(pages["grids"]),
        "model_artifacts.json": to_native(model),
    }
    for p in pages["pitchers"]:
        body = to_native({
            "pitcherId": p["pitcherId"],
            "name": p["name"],
            "hand": p["hand"],
            "season": pages["season"],
            "arsenal": [{**a, "label": PITCH_TYPE_LABELS.get(a["type"], a["type"])}
                        for a in p["arsenal"]],
            "outings": p["outings"],
            "pitches": p["pitches"],
        })
        files[f"pitchers/{p['pitcherId']}.json"] = body
    return files
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp_publisher/tests/test_build_pitcher_bundle.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add webapp_publisher/build_pitcher_bundle.py webapp_publisher/tests/test_build_pitcher_bundle.py
git commit -m "Build the pitcher-page bundle files

Keys per-pitcher files by TrackMan's PitcherId rather than the staff board's
positional id, which shifts whenever the roster changes and would repoint every
pitcher's URL. Fails the build if any model feature lacks a plain-English
label, since raw field names are exactly what the page exists to remove."
```

---

### Task 7: schema validation and publisher wiring

**Files:**
- Modify: `webapp_publisher/schema.py`
- Modify: `webapp_publisher/publish.py`
- Modify: `webapp_publisher/README.md`
- Test: `webapp_publisher/tests/test_schema_pitcher.py`

**Interfaces:**
- Consumes: `build_pitcher_bundle`, `pitcher_index` from Task 6.
- Produces: `validate_pitcher_bundle(files: dict) -> None` in `schema.py`; `run_pitcher_scorer(data, workdir, team) -> dict` in `publish.py`.

- [ ] **Step 1: Write the failing test**

Create `webapp_publisher/tests/test_schema_pitcher.py`:

```python
import copy

import pytest

from webapp_publisher.schema import validate_pitcher_bundle

GOOD = {
    "location_maps.json": {"pooled": [{"x": 0.0, "z": 2.5, "v": 0.01}]},
    "model_artifacts.json": {
        "featureOrder": ["SpinRate"], "labels": {"SpinRate": "Spin rate"},
        "byPitchType": {"FF": {"coef": [0.01], "scalerMean": [2200.0],
                               "scalerScale": [180.0], "populationMeanZ": [0.0],
                               "displayMu": 0.0, "displaySd": 0.02,
                               "sampleFloor": 100, "nQualified": 400}}},
    "pitchers/1000123.json": {
        "pitcherId": 1000123, "name": "Test-Pitcher, Alpha", "hand": "R", "season": 2026,
        "arsenal": [{"type": "FF", "label": "Fastball", "n": 412, "usage": 1.0,
                     "stuff": 124.0, "loc": 103.0, "recentChange": -6.2,
                     "aboveFloor": True, "typical": [2350.0], "percentiles": [78]}],
        "outings": [{"date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.0}],
        "pitches": [{"d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31,
                     "c": "0-2", "g": 131.0, "f": [2350.0]}],
    },
}


def test_valid_bundle_passes():
    validate_pitcher_bundle(copy.deepcopy(GOOD))


def test_missing_shared_file_is_rejected():
    bad = copy.deepcopy(GOOD)
    del bad["model_artifacts.json"]
    with pytest.raises(ValueError, match="model_artifacts.json"):
        validate_pitcher_bundle(bad)


def test_bundle_with_no_pitchers_is_rejected():
    bad = {k: v for k, v in GOOD.items() if not k.startswith("pitchers/")}
    with pytest.raises(ValueError, match="no pitcher files"):
        validate_pitcher_bundle(bad)


def test_secondary_pitch_type_carrying_a_location_score_is_rejected():
    """Location+ on a slider is a construct leak, so the publisher refuses to
    ship it rather than leaving it to the frontend to hide.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["type"] = "Slider"
    with pytest.raises(ValueError, match="Location\\+"):
        validate_pitcher_bundle(bad)


def test_feature_array_length_mismatch_is_rejected():
    """Feature arrays are positional against featureOrder; a length mismatch
    would mislabel every trait rather than fail visibly.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["pitches"][0]["f"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="feature array"):
        validate_pitcher_bundle(bad)


def test_unlabeled_feature_is_rejected():
    bad = copy.deepcopy(GOOD)
    bad["model_artifacts.json"]["labels"] = {}
    with pytest.raises(ValueError, match="label"):
        validate_pitcher_bundle(bad)


def test_out_of_range_percentile_is_rejected():
    """A percentile outside 0-100 means the reference population was wrong,
    which would put a nonsense rank in front of a coach.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["percentiles"] = [140]
    with pytest.raises(ValueError, match="percentile"):
        validate_pitcher_bundle(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp_publisher/tests/test_schema_pitcher.py -v`
Expected: FAIL with `ImportError: cannot import name 'validate_pitcher_bundle'`

- [ ] **Step 3: Write minimal implementation**

Append to `webapp_publisher/schema.py`:

```python
REQUIRED_ARSENAL_KEYS = {"type", "label", "n", "usage", "stuff", "loc",
                         "recentChange", "aboveFloor", "typical", "percentiles"}
REQUIRED_PITCH_KEYS = {"d", "t", "x", "z", "c", "g", "f"}


def validate_pitcher_bundle(files: dict) -> None:
    """Fail loudly before upload. Mirrors validate_bundle's style: plain
    ValueErrors naming the offending file and key.
    """
    for name in ("location_maps.json", "model_artifacts.json"):
        if name not in files:
            raise ValueError(f"pitcher bundle missing {name}")

    model = files["model_artifacts.json"]
    order = model["featureOrder"]
    n_feats = len(order)
    for feat in order:
        if model.get("labels", {}).get(feat) in (None, feat):
            raise ValueError(f"feature {feat} has no plain-English label")
    for tname, m in model["byPitchType"].items():
        for key in ("coef", "scalerMean", "scalerScale", "populationMeanZ"):
            if len(m[key]) != n_feats:
                raise ValueError(f"{tname}.{key} feature array is {len(m[key])}, expected {n_feats}")
        if m["displaySd"] <= 0:
            raise ValueError(f"{tname} displaySd must be positive, got {m['displaySd']}")

    pitcher_files = [k for k in files if k.startswith("pitchers/")]
    if not pitcher_files:
        raise ValueError("pitcher bundle has no pitcher files")

    for key in pitcher_files:
        body = files[key]
        if not body.get("arsenal"):
            raise ValueError(f"{key} has no arsenal rows")
        for a in body["arsenal"]:
            missing = REQUIRED_ARSENAL_KEYS - set(a)
            if missing:
                raise ValueError(f"{key} arsenal row missing {missing}")
            if a["type"] != "FF" and a["loc"] is not None:
                raise ValueError(f"{key} emits Location+ for {a['type']}; it is a fastball score only")
            for arr in ("typical", "percentiles"):
                if len(a[arr]) != n_feats:
                    raise ValueError(f"{key} arsenal {arr} feature array is {len(a[arr])}, expected {n_feats}")
            if any(not 0 <= p <= 100 for p in a["percentiles"]):
                raise ValueError(f"{key} has a percentile outside 0-100")
        for p in body["pitches"]:
            missing = REQUIRED_PITCH_KEYS - set(p)
            if missing:
                raise ValueError(f"{key} pitch row missing {missing}")
            if len(p["f"]) != n_feats:
                raise ValueError(f"{key} pitch feature array is {len(p['f'])}, expected {n_feats}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp_publisher/tests/test_schema_pitcher.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Wire the new stage into `publish.py`**

In `webapp_publisher/publish.py`, add next to the existing `SCORER` constant:

```python
PITCHER_SCORER = REPO / "component_model" / "analysis" / "14_pitcher_pages.py"
```

Add this function immediately after `run_scorer`:

```python
def run_pitcher_scorer(data: str, workdir: str, team: str) -> dict:
    workdir_p = pathlib.Path(workdir)
    cmd = [sys.executable, str(PITCHER_SCORER), "--data", data, "--workdir", workdir, "--team", team]
    subprocess.run(cmd, check=True)  # raises CalledProcessError -> loud failure
    out = workdir_p / "pitcher_pages.json"
    if not out.exists():
        raise FileNotFoundError(f"pitcher scorer did not produce {out}")
    return json.loads(out.read_text())
```

Update the imports at the top of `publish.py`. Note the existing style is absolute (`from webapp_publisher.x import y`), not relative — add the new line and extend the existing `schema` import:

```python
from webapp_publisher.build_pitcher_bundle import build_pitcher_bundle, pitcher_index
from webapp_publisher.schema import validate_bundle, validate_pitcher_bundle
```

In `main()`, immediately after the existing `bundle = build_bundle(...)` line, insert:

```python
    pages = run_pitcher_scorer(args.data, args.workdir, args.team)
    pitcher_files = build_pitcher_bundle(pages)
    validate_pitcher_bundle(pitcher_files)
    bundle["manifest.json"]["pitchers"] = pitcher_index(pages)
    bundle.update(pitcher_files)
```

This must sit before the existing `validate_bundle(bundle)` call so the manifest is validated after the pitcher index is attached.

- [ ] **Step 6: Run the full publisher suite**

Run: `python -m pytest webapp_publisher/tests/ -v`
Expected: PASS, all tests including the existing `test_publish.py`, `test_upload.py`, `test_build_bundle.py`

- [ ] **Step 7: Document the new outputs**

In `webapp_publisher/README.md`, add to the section describing bundle contents:

```markdown
The bundle also carries the pitcher development page files: `location_maps.json`
(count-conditioned run-value surface, shared), `model_artifacts.json` (ridge
coefficients, scaler, display moments, plain-English labels), and one
`pitchers/{pitcherId}.json` per pitcher. These come from
`component_model/analysis/14_pitcher_pages.py` and are keyed by TrackMan
PitcherId, which is stable across seasons.
```

- [ ] **Step 8: Commit**

```bash
git add webapp_publisher/schema.py webapp_publisher/publish.py webapp_publisher/README.md webapp_publisher/tests/test_schema_pitcher.py
git commit -m "Validate and publish the pitcher-page files

The publisher now refuses to ship Location+ on a secondary pitch type or a
feature array whose length disagrees with featureOrder. Both would otherwise
fail silently: the first as a construct leak the frontend would have to know to
hide, the second by mislabeling every trait on the page rather than erroring."
```

---

### Task 8: put the target pipeline in the scheduled chain

**Files:**
- Modify: `python_files/target_and_calculated_pipeline.py`
- Modify: `webapp_publisher/run_refresh.ps1`
- Modify: `webapp_publisher/README.md`
- Test: `python_files/tests/test_pipeline_cli.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: a CLI on `target_and_calculated_pipeline.py` accepting `--base-path`, `--years`, `--summary-path`, `--out-dir`, `--out-name`, and writing a deterministically-named CSV.

Why this task exists: `publish.py`'s `run_scorer` reads a prebuilt `Final_Target_Calc_*.csv` via `--data`. Nothing in the scheduled path regenerates that CSV from the game-file tree, so a refresh currently cannot pick up new games. `target_and_calculated_pipeline.py` has no CLI at all — its `__main__` block hardcodes another machine's absolute paths — and it names output `Final_Target_Calc_{HHMM}.csv`, a clock-dependent name a scheduled job cannot predict.

- [ ] **Step 1: Write the failing test**

Create `python_files/tests/test_pipeline_cli.py`:

```python
"""The scheduled refresh needs a predictable entry point and a predictable
output filename. The existing __main__ block has neither.
"""
import os
import subprocess
import sys

PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "python_files", "target_and_calculated_pipeline.py")
if not os.path.exists(PIPELINE):
    PIPELINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "target_and_calculated_pipeline.py")


def test_module_imports_without_running_a_build():
    """Importing must stay side-effect free so the module is testable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcp", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "build_final_dataset")
    assert hasattr(mod, "main")


def test_cli_reports_usage_and_exits_nonzero_without_required_args():
    proc = subprocess.run([sys.executable, PIPELINE], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "--base-path" in (proc.stderr + proc.stdout)


def test_out_name_is_deterministic_when_supplied():
    """A clock-based filename cannot be handed to the next stage of a scheduled
    run, so --out-name must win over the HHMM default.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcp", PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.output_path("/tmp", None).startswith("/tmp")
    assert mod.output_path("/tmp", "Final_Target_Calc_current.csv") == os.path.join(
        "/tmp", "Final_Target_Calc_current.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest python_files/tests/test_pipeline_cli.py -v`
Expected: FAIL — `main` and `output_path` do not exist, and the bare invocation currently either runs a full build against hardcoded paths or raises

- [ ] **Step 3: Write minimal implementation**

In `python_files/target_and_calculated_pipeline.py`, add near the top imports:

```python
import argparse
```

Add these two functions above the existing `if __name__ == "__main__":` block:

```python
def output_path(out_dir, out_name=None):
    """Where the final dataset lands.

    A scheduled run must pass --out-name so the next stage can predict the path.
    The clock-based default is kept only for interactive one-off builds.
    """
    if out_name:
        return os.path.join(out_dir, out_name)
    return os.path.join(out_dir, f"Final_Target_Calc_{datetime.now().strftime('%H%M')}.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build Final_Target_Calc from the game-file tree.")
    ap.add_argument("--base-path", required=True,
                    help="root of the year/month/day/CSV game tree")
    ap.add_argument("--years", required=True,
                    help="comma-separated years, e.g. 2025,2026")
    ap.add_argument("--summary-path", required=True,
                    help="game-state summary parquet/csv path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-name", default=None,
                    help="deterministic output filename; required for scheduled runs")
    args = ap.parse_args(argv)

    years = [y.strip() for y in args.years.split(",") if y.strip()]
    out = output_path(args.out_dir, args.out_name)
    df = build_final_dataset(args.base_path, years, args.summary_path, args.out_dir, save=False)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    return 0
```

Replace the entire existing `if __name__ == "__main__":` block (the one with hardcoded `/Users/suma/...` paths) with:

```python
if __name__ == "__main__":
    sys.exit(main())
```

Confirm `import sys` and `from datetime import datetime` are already present at the top of the file; add whichever is missing.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest python_files/tests/test_pipeline_cli.py -v`
Expected: PASS, 3 tests

- [ ] **Step 5: Confirm the existing pipeline tests still pass**

Run: `python -m pytest python_files/tests/ -v`
Expected: PASS, including the existing `test_game_file_resolution.py` and `test_pipeline_revisions.py`

- [ ] **Step 6: Add the pipeline stage to `run_refresh.ps1`**

Replace the existing `param(...)` block with:

```powershell
param(
  [int]$MaxRetries = 4,
  [int]$TimeoutMinutes = 30,
  [string]$GameTree = $env:STUFFPLUS_GAME_TREE,
  [string]$SummaryPath = $env:STUFFPLUS_SUMMARY,
  [string]$Years = $env:STUFFPLUS_YEARS,
  [switch]$SkipPipeline
)
```

Then, inside the `try` block, insert this immediately after the two existing `$stdoutLog`/`$stderrLog` assignments and before the existing `$proc = Start-Process ...` line for publish. It reuses the same `$remainingSec` deadline arithmetic and the same kill-on-timeout pattern, so a hung pipeline is covered by the existing overall timeout rather than needing its own:

```powershell
    if (-not $SkipPipeline) {
      if (-not $GameTree -or -not $SummaryPath -or -not $Years) {
        throw "GameTree, SummaryPath and Years are required unless -SkipPipeline is passed"
      }
      if (-not $env:STUFFPLUS_WORKDIR) { throw "STUFFPLUS_WORKDIR must be set" }
      # Deterministic filename so the scorer stage can predict the path. The
      # pipeline's own default is wall-clock based and unusable from a schedule.
      $targetCsv = Join-Path $env:STUFFPLUS_WORKDIR "Final_Target_Calc_current.csv"
      $pipeOut = Join-Path $logDir "refresh-$dateStamp-attempt$attempt-pipeline.log"
      $pipeErr = Join-Path $logDir "refresh-$dateStamp-attempt$attempt-pipeline.err.log"
      $pipeProc = Start-Process -FilePath "python" -ArgumentList @(
        "python_files\target_and_calculated_pipeline.py",
        "--base-path", $GameTree,
        "--years", $Years,
        "--summary-path", $SummaryPath,
        "--out-dir", $env:STUFFPLUS_WORKDIR,
        "--out-name", "Final_Target_Calc_current.csv"
      ) -NoNewWindow -PassThru -RedirectStandardOutput $pipeOut -RedirectStandardError $pipeErr
      if (-not $pipeProc.WaitForExit($remainingSec * 1000)) {
        try { $pipeProc.Kill() } catch {}
        throw "target pipeline timed out after ${remainingSec}s (attempt $attempt)"
      }
      if ($pipeProc.ExitCode -ne 0) {
        throw "target pipeline exited with code $($pipeProc.ExitCode) (attempt $attempt)"
      }
      # publish.py reads STUFFPLUS_DATA; point it at what we just built.
      $env:STUFFPLUS_DATA = $targetCsv
      # Recompute the remaining budget so publish gets the time actually left.
      $remainingSec = [int]([Math]::Floor(($deadline - (Get-Date)).TotalSeconds))
      if ($remainingSec -le 0) { throw "no time left for publish after pipeline (attempt $attempt)" }
    }
```

- [ ] **Step 7: Verify the script parses**

Run: `powershell -NoProfile -Command "[void][System.Management.Automation.Language.Parser]::ParseFile('webapp_publisher/run_refresh.ps1', [ref]$null, [ref]$null); 'parsed ok'"`
Expected: prints `parsed ok`

- [ ] **Step 8: Verify the skip path still works without the new variables**

Run: `powershell -NoProfile -File webapp_publisher/run_refresh.ps1 -SkipPipeline -MaxRetries 1`
Expected: behaves exactly as before this task — it runs publish only. It may still fail for unrelated reasons (missing `STUFFPLUS_DATA` or storage credentials); what matters is that it does **not** fail with the "GameTree, SummaryPath and Years are required" message.

- [ ] **Step 9: Document the new environment variables**

In `webapp_publisher/README.md`, add to the environment-variable section:

```markdown
Scheduled runs also need `STUFFPLUS_GAME_TREE` (root of the `year/month/day/CSV`
game tree, filled by either the FTP mirror or `trackman_api/backfill.py --refresh`),
`STUFFPLUS_SUMMARY` (game-state summary path), and `STUFFPLUS_YEARS`. The refresh
regenerates `Final_Target_Calc_current.csv` in the workdir and points the scorer at
it, so new games reach the page without a manual rebuild. Pass `-SkipPipeline` to
reuse an existing `STUFFPLUS_DATA` CSV instead.
```

- [ ] **Step 10: Commit**

```bash
git add python_files/target_and_calculated_pipeline.py python_files/tests/test_pipeline_cli.py webapp_publisher/run_refresh.ps1 webapp_publisher/README.md
git commit -m "Let a scheduled refresh actually pick up new games

The publish chain scored a prebuilt CSV that nothing in the chain rebuilt, so
new games never reached the page without a manual run. The pipeline had no CLI
at all (its entry point hardcoded another machine's paths) and named output by
wall-clock minute, which the next stage cannot predict. Both fixed, and the new
stage runs inside the existing attempt and timeout machinery so a hang is
already covered.

Game-file collection still goes through resolve_latest_game_files, which is
what keeps re-pulled games from double-counting pitches."
```

---

### Task 9: end-to-end verification against real data

**Files:**
- Create: `docs/superpowers/plans/2026-08-05-pitcher-page-verification.md` (findings record)

No new production code. This task runs the chain against the real caches and records what it found, in the same spirit as the script-13 real-data run that surfaced three bugs no synthetic test caught.

- [ ] **Step 1: Run the new scorer against real data**

```bash
cd component_model/analysis
python 14_pitcher_pages.py \
  --data "C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv" \
  --workdir "C:/Users/jackdav/stuffplus_replication/workdir_webapp" \
  --years 2025,2026 --level D1 --team DEL_BLU
```

Expected: prints a per-type pitch/qualified count, a pitcher count for DEL_BLU, and writes `pitcher_pages.json`.

- [ ] **Step 2: Check the Decision 2 open question — the spread of per-pitch grades**

The spec commits to measuring this before shipping. Run:

Write this to `scratch_pitch_spread.py` in the workdir (not the repo) and run it:

```python
import json
import statistics as st

WD = r"C:/Users/jackdav/stuffplus_replication/workdir_webapp"
d = json.load(open(WD + "/pitcher_pages.json"))
for p in d["pitchers"]:
    ff = [q["g"] for q in p["pitches"] if q["t"] == "FF"]
    if len(ff) < 30:
        continue
    row = next(a for a in p["arsenal"] if a["type"] == "FF")
    print(f"n={len(ff):4d} type_grade={row['stuff']:6.1f} pitch_mean={st.mean(ff):6.1f} "
          f"pitch_sd={st.stdev(ff):5.1f} min={min(ff):6.1f} max={max(ff):6.1f}")
```

Two things to read off it. `pitch_mean` must equal `type_grade` for every pitcher — that is the additivity guarantee holding on real data. And `pitch_sd` with `min`/`max` is the actual answer to the spec's open question.

Record the observed per-pitch standard deviation and range. Per the spec: if the spread is wide enough that individual pitch grades look broken, the committed response is to display the distribution honestly (a band around the typical value), **not** to introduce a second scale.

- [ ] **Step 3: Verify the pitcher page's fastball Stuff+ agrees with the Staff Board's**

The same pitcher must not read one number on the Staff Board and a different one on his page. First run script 08 against the same data and workdir:

```bash
cd component_model/analysis
python 08_staff_scores.py \
  --data "C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv" \
  --workdir "C:/Users/jackdav/stuffplus_replication/workdir_webapp" \
  --years 2025,2026 --level D1 --team DEL_BLU
```

Then write this to `scratch_compare_ff.py` in the workdir and run it:

```python
import json

WD = r"C:/Users/jackdav/stuffplus_replication/workdir_webapp"
staff = {s["name"]: s["stuff"] for s in json.load(open(WD + "/staff_scores.json"))["staff"]}
pages = json.load(open(WD + "/pitcher_pages.json"))
diffs = []
for p in pages["pitchers"]:
    ff = next((a for a in p["arsenal"] if a["type"] == "FF"), None)
    if ff and p["name"] in staff:
        diff = ff["stuff"] - staff[p["name"]]
        diffs.append(diff)
        print(f"{p['name']:28s} board={staff[p['name']]:6.1f} page={ff['stuff']:6.1f} diff={diff:+5.1f}")
if diffs:
    print(f"\nmax abs diff: {max(abs(d) for d in diffs):.2f} over {len(diffs)} pitchers")
```

Expected: differences near zero. They will not be exactly zero, because script 08 scales against the four-seam qualified population with `n_ff >= 100` over its own year pair while the new scorer scales per type over the graded season. **If any difference exceeds about 3 points, stop and reconcile before building the frontend** — a coach seeing two different fastball grades for one pitcher is the exact confusion this design is meant to prevent. Record the observed distribution of differences either way.

- [ ] **Step 4: Run the publisher end to end in dry-run**

```bash
python -m webapp_publisher.publish \
  --data "C:/Users/jackdav/stuffplus_replication/source_2025_2026.csv" \
  --workdir "C:/Users/jackdav/stuffplus_replication/workdir_webapp" \
  --team DEL_BLU --dry-run
```

Expected: writes `manifest.json`, `staff_board.json`, `location_maps.json`, `model_artifacts.json`, and one `pitchers/{id}.json` per pitcher under `<workdir>/bundle/`, and passes both validators. Record the byte size of the largest pitcher file — the spec assumed low hundreds of kilobytes, and a much larger file means the frontend needs pagination.

- [ ] **Step 5: Record findings**

Write `docs/superpowers/plans/2026-08-05-pitcher-page-verification.md` documenting: per-type pitch and qualified counts, the per-pitch grade spread from Step 2 and the decision it drove, the Staff Board agreement distribution from Step 3, the largest pitcher-file size, and any bug the real run surfaced that no synthetic test caught. Note explicitly that this is one run against one caches pair, per the replication discipline in `component_model/analysis/README.md`.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-08-05-pitcher-page-verification.md
git commit -m "Record what the real-data run of the pitcher-page chain found

Answers the spec's open question on per-pitch grade spread with a measurement
rather than the expectation, and confirms whether a pitcher's fastball grade
reads the same on his page as on the Staff Board."
```

---

## What this plan does not cover

The React page itself is Plan 2, written after this plan's Task 9 produces a real bundle, so the frontend is built against measured payload sizes and the confirmed contract rather than assumptions. Plan 2 covers the arsenal table, the three panels, routing, the browser-side attribution with its shared cross-language fixture test, and the Application Insights wiring for both the frontend and the managed functions.

Hardening the TrackMan API client stays an independent track: merge `trackman-api-slice`, resolve the egress-IP whitelist with the TrackMan rep, run a real end-to-end `backfill.py --refresh`, complete the historical backfill, and add tests. Nothing in this plan depends on it, because ingestion only has to leave files in the game tree.
