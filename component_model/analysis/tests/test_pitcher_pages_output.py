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


class _FakeMap:
    """Stands in for fc.PooledLocationMap: records the training frame it was
    given and returns a constant, so a test can assert what it was trained on."""
    trained_rows = None

    def __init__(self, train):
        _FakeMap.trained_rows = len(train)

    def apply(self, sub):
        return np.full(len(sub), 0.25)


class _FakeFC:
    FEATS = []

    @staticmethod
    def add_loc_bins(df):
        df["gx"] = 0.0
        df["gz"] = 0.0
        return df

    PooledLocationMap = _FakeMap


def _two_season_frame():
    """Both year roles present: 2024 (earlier, trains the map) and 2025 (graded)."""
    return pd.DataFrame({
        "PitcherId": [1, 1, 1, 1],
        "year": [2024, 2024, 2025, 2025],
        "is_ff": [True, True, True, True],
        "TaggedPitchType": ["FourSeamFastBall"] * 4,
        "PlateLocSide": [0.1, -0.1, 0.2, -0.2],
        "PlateLocHeight": [2.4, 2.6, 2.5, 2.5],
        "xT": [0.01, -0.01, 0.02, -0.02],
    })


def test_attach_location_trains_on_the_earlier_season_not_the_graded_one():
    """Regression: the graded-season frame alone contains no earlier-season rows,
    so training off it selected zero rows and produced an all-NaN Location+.
    """
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, None, _FakeFC, 2025)
    assert _FakeMap.trained_rows == 2          # the two 2024 rows, not zero
    assert state["pitches"]["loc"].notna().all()
    assert not state["pitches"]["loc"].isna().any()


def test_attach_location_leaves_secondary_types_without_a_value():
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, {"Slider"}, _FakeFC, 2025)
    assert state["pitches"]["loc"].isna().all()


def test_attach_location_fails_loudly_when_no_earlier_season_exists():
    """Better to stop than to publish a null Location+ for fastballs."""
    mod = _load_pages_module()
    pit = _two_season_frame()
    pit = pit[pit["year"] == 2025]
    state = {"pitches": pit.copy()}
    with pytest.raises(ValueError, match="train the location map"):
        mod.attach_location(pit, state, None, _FakeFC, 2025)
