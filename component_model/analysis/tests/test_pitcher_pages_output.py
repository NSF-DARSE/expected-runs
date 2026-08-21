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
        # Raw location run values, pitcher's perspective (lower = better).
        # Pitcher 1 locates better than pitcher 2.
        "loc": [-0.01] * 3 + [0.01] * 3,
    })
    for f in feats:
        pitches[f] = rng.normal(0, 1, n)
    return feats, {
        "pitches": pitches,
        "scaler_mean": np.zeros(12), "scaler_scale": np.ones(12),
        "coef": rng.normal(0, 0.01, 12),
        "mu": 0.0, "sd": 0.02,
        "loc_mu": 0.0, "loc_sd": 0.01,
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


def test_fastball_location_score_is_on_the_display_scale_and_polarised():
    """Regression: `loc` was emitted as the bare mean expected-run value (~0.00x,
    LOWER = better), so the page showed a raw run value with reversed polarity
    where the Staff Board showed a 100 +/- 15 score. It must go through
    arsenal.to_display, which negates exactly once: the pitcher with the LOWER
    mean raw loc has to end up with the HIGHER Location+.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    by_id = {r["pitcherId"]: r["arsenal"][0]["loc"] for r in records}
    raw = fitted["pitches"].groupby("PitcherId")["loc"].mean()
    assert raw[1] < raw[2]                      # pitcher 1 locates better (lower runs)
    assert by_id[1] > by_id[2]                  # ...so his display score is higher
    for pid, expected in ((1, 115.0), (2, 85.0)):
        assert by_id[pid] == pytest.approx(expected)
        assert 40.0 <= by_id[pid] <= 160.0      # the band the schema enforces


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
    given and returns a per-pitcher value, so a test can assert both what it was
    trained on and that the display moments come out of a real spread."""
    trained_rows = None

    def __init__(self, train):
        _FakeMap.trained_rows = len(train)

    def apply(self, sub):
        return np.where(sub["PitcherId"].values == 1, 0.20, 0.30)


class _FakeFC:
    FEATS = []

    @staticmethod
    def add_loc_bins(df):
        df["gx"] = 0.0
        df["gz"] = 0.0
        return df

    PooledLocationMap = _FakeMap


def _two_season_frame():
    """Both year roles present: 2024 (earlier, trains the map) and 2025 (graded).

    Two pitchers in each season, because the display moments attach_location now
    derives need at least two qualifying pitchers to define a scale.
    """
    return pd.DataFrame({
        "PitcherId": [1, 2, 1, 2],
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
    mod.attach_location(pit, state, None, _FakeFC, 2025, floor_n=1)
    assert _FakeMap.trained_rows == 2          # the two 2024 rows, not zero
    assert state["pitches"]["loc"].notna().all()
    assert not state["pitches"]["loc"].isna().any()


def test_attach_location_derives_display_moments_from_qualifying_pitchers():
    """build_pitcher_records needs these to route Location+ through to_display.
    Without them it fell back to emitting the raw run value.
    """
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, None, _FakeFC, 2025, floor_n=1)
    assert state["loc_mu"] == pytest.approx(0.25)   # mean of 0.20 and 0.30
    assert state["loc_sd"] > 0


def test_attach_location_leaves_secondary_types_without_a_value():
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, {"Slider"}, _FakeFC, 2025, floor_n=1)
    assert state["pitches"]["loc"].isna().all()
    assert state["loc_mu"] is None and state["loc_sd"] is None


def test_attach_location_fails_loudly_when_no_earlier_season_exists():
    """Better to stop than to publish a null Location+ for fastballs."""
    mod = _load_pages_module()
    pit = _two_season_frame()
    pit = pit[pit["year"] == 2025]
    state = {"pitches": pit.copy()}
    with pytest.raises(ValueError, match="train the location map"):
        mod.attach_location(pit, state, None, _FakeFC, 2025, floor_n=1)


# ---------------- trend block (windowed_delta + build_trend) ----------------

def _trend_frame(n_recent=20, n_prior=20, velo_col="EffectiveVelo"):
    """Pitches spread over two adjacent 30-day windows ending at the asof date."""
    rng = np.random.default_rng(3)
    n = n_recent + n_prior
    dates = (["2026-04-20"] * n_prior) + (["2026-05-20"] * n_recent)
    return pd.DataFrame({
        "Date": dates,
        "is_lhp": [0] * n,
        "HorzBreak": rng.normal(14, 1, n),
        "InducedVertBreak": rng.normal(8, 1, n),
        velo_col: np.concatenate([rng.normal(90, 0.5, n_prior),
                                  rng.normal(92, 0.5, n_recent)]),
    })


def test_windowed_delta_returns_none_below_floor():
    df = _trend_frame(n_recent=5, n_prior=40)
    out = ar.windowed_delta(df["EffectiveVelo"], df["Date"], "2026-05-25", floor_n=15)
    assert out is None


def test_windowed_delta_measures_the_shift_with_a_sane_se():
    df = _trend_frame()
    out = ar.windowed_delta(df["EffectiveVelo"], df["Date"], "2026-05-25", floor_n=15)
    assert out is not None
    assert out["delta"] == pytest.approx(2.0, abs=0.5)
    assert 0 < out["se"] < 0.5
    assert out["nRecent"] == 20 and out["nPrior"] == 20


def test_build_trend_directions_are_tiered_by_validation():
    """Arrow metadata ships only where the direction claim was validated:
    velo 'up' for FF and Sinker, never for other types, and break-shape
    metrics carry no direction anywhere."""
    mod = _load_pages_module()
    df = _trend_frame()
    grades = np.full(len(df), 100.0)
    for tname, want_velo in [("FF", "up"), ("Sinker", "up"),
                             ("Cutter", None), ("Slider", None)]:
        trend = mod.build_trend(df, grades, tname, "2026-05-25", floor_n=15)
        assert trend["velo"]["direction"] == want_velo, tname
        assert trend["movAngle"]["direction"] is None
        assert trend["movMag"]["direction"] is None
    # Stuff+ rows always say "up" (display scale); the frontend gates WHICH
    # types render that row at all via isStuffPlusConfirmed.
    trend = mod.build_trend(df, grades, "Sinker", "2026-05-25", floor_n=15)
    assert trend["stuff"]["direction"] == "up"


def test_build_trend_prefers_relspeed_and_falls_back_to_effectivevelo():
    mod = _load_pages_module()
    df = _trend_frame(velo_col="RelSpeed")
    df["EffectiveVelo"] = 0.0  # would flatten the delta if wrongly chosen
    grades = np.full(len(df), 100.0)
    trend = mod.build_trend(df, grades, "FF", "2026-05-25", floor_n=15)
    assert trend["velo"]["delta"] == pytest.approx(2.0, abs=0.5)
    df2 = _trend_frame(velo_col="EffectiveVelo")
    trend2 = mod.build_trend(df2, grades, "FF", "2026-05-25", floor_n=15)
    assert trend2["velo"]["delta"] == pytest.approx(2.0, abs=0.5)
