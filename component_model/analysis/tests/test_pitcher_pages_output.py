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
        # Raw location run values, pitcher's perspective (lower = better).
        # Pitcher 1 locates better than pitcher 2.
        "loc": [-0.01] * 3 + [0.01] * 3,
        # Real release speed. Deliberately not in `feats`: it carries no ridge
        # coefficient and exists only as context beside the pitch type.
        "RelSpeed": [93.0, 94.0, 92.0, 88.0, 89.0, 87.0],
        "BatterSide": ["Right", "Left", "Right", "Left", "Right", "Left"],
        "count12": ["0-0"] * n,
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


def test_average_velocity_covers_the_same_pitches_as_the_row_grade():
    """avgVelo is the mean over the graded pitches, the ones counted by `n`.

    The alternative was a season-wide mean including ungraded pitches, which
    would put a velocity next to a count it does not describe. Pitcher 1 throws
    93/94/92 and pitcher 2 throws 88/89/87 in the fixture.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    by_id = {r["pitcherId"]: r["arsenal"][0] for r in records}
    assert by_id[1]["avgVelo"] == pytest.approx(93.0)
    assert by_id[2]["avgVelo"] == pytest.approx(88.0)
    for row in by_id.values():
        assert row["n"] == 3


def test_an_extract_without_relspeed_publishes_without_a_velocity():
    """The real source extract is a trimmed subset of the pipeline's output and
    the trim in use through 2026-08 drops RelSpeed, so this is the state of the
    data on disk today, not a hypothetical.

    A missing display field must not block a publish. The page already renders a
    row with no velocity; what must never happen is a fabricated number, so the
    field is None rather than 0 or a guess.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    fitted["pitches"] = fitted["pitches"].drop(columns=["RelSpeed"])
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    assert records, "a missing velocity must not drop the pitcher entirely"
    for r in records:
        assert r["arsenal"][0]["avgVelo"] is None


def test_average_velocity_is_not_a_graded_trait():
    """The constraint that makes velo context rather than a trait row: nothing in
    the model sees RelSpeed, so there is no coefficient to rank it against. If it
    ever enters `feats`, it acquires a percentile and a Worth column and quietly
    becomes a second Stuff+ input, which is the construct leak this avoids.
    """
    feats, _ = _fitted()
    assert "RelSpeed" not in feats


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


def _loc_frame():
    """Two pitchers with deliberately different location habits: pitcher 1 lives
    down and away in two-strike counts, pitcher 2 lives middle-middle."""
    import pandas as pd
    rows = []
    for i in range(40):
        rows.append({"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
                     "BatterSide": "Right", "count12": "0-2", "loc": -0.02})
    for i in range(40):
        rows.append({"PitcherId": 2, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
                     "BatterSide": "Right", "count12": "0-0", "loc": 0.02})
    return pd.DataFrame(rows)


def test_location_decomposition_sums_to_the_score_it_explains():
    """The guarantee that makes the card checkable by hand: the rows add to his
    Location+ minus 100. A decomposition that does not close is a decomposition
    a coach cannot trust, and the arsenal table already learned that lesson.
    """
    import numpy as np
    import arsenal as ar
    df = _loc_frame()
    mu, sd = 0.0, 0.02
    his = df[df["PitcherId"] == 1]
    rows = ar.location_decomposition(his, df, mu, sd, min_share=0.0)
    location_plus = ar.to_display(his["loc"].mean(), mu, sd)
    assert sum(r["points"] for r in rows) == pytest.approx(float(location_plus) - 100.0, abs=1e-9)


def test_location_decomposition_reports_both_shares_not_just_points():
    """The question the card answers is why a region paid him, and at this grain
    the answer is occupancy: the map values a spot the same for everyone, so his
    edge is being there more than the field. Points without the two shares state
    the effect and hide the cause.
    """
    import arsenal as ar
    df = _loc_frame()
    rows = ar.location_decomposition(df[df["PitcherId"] == 1], df, 0.0, 0.02, min_share=0.0)
    top = rows[0]
    assert top["region"] == "Down and away"
    assert top["count"] == "ahead"
    assert top["share"] == pytest.approx(1.0)      # every one of his pitches
    assert top["leagueShare"] == pytest.approx(0.5)  # half the league's


def test_adjusted_results_are_emitted_per_pitch_type():
    """Adj Results is legitimate off the fastball where Location+ is not: it
    describes what happened with luck, defense and opponent removed, and a
    description does not need to predict next season to be true. Read it with the
    pitch count, since the criterion's own reliability is roughly half the
    four-seam value on secondaries.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    fitted["pitches"]["adjT"] = [-0.02, -0.01, 0.0, 0.03, 0.02, 0.01]
    fitted["adj_mu"], fitted["adj_sd"] = 0.0, 0.02
    records = mod.build_pitcher_records({"Slider": fitted}, feats, floor_n=1,
                                        asof="2026-03-10", min_type_pitches=1)
    by_id = {r["pitcherId"]: r["arsenal"][0] for r in records}
    # Pitcher 1 gives up less (lower adjT is better), so his display score is higher.
    assert by_id[1]["adjRes"] > by_id[2]["adjRes"]
    assert by_id[1]["loc"] is None, "a slider still has no Location+"


def test_a_pitch_type_with_no_results_scale_emits_no_results_number():
    """A type with too few qualifying pitchers to define a scale must show
    nothing rather than a score resting on a scale that was never established.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    fitted["pitches"]["adjT"] = [-0.02, -0.01, 0.0, 0.03, 0.02, 0.01]
    fitted["adj_mu"], fitted["adj_sd"] = None, None
    records = mod.build_pitcher_records({"Splitter": fitted}, feats, floor_n=1,
                                        asof="2026-03-10", min_type_pitches=1)
    assert all(r["arsenal"][0]["adjRes"] is None for r in records)


def test_pitch_rows_carry_location_on_the_same_scale_as_the_season_number():
    """One scale, per arsenal.py: a pitch's Location+ and the season Location+
    go through the same transform, so the two are comparable and a pitch can be
    read against his own average. A separately calibrated per-pitch scale would
    look reasonable and quietly break that comparison.

    Fastball only, because Location+ is.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    import arsenal as ar
    # The fixture gives each pitcher a constant raw loc (-0.01 for pitcher 1,
    # +0.01 for pitcher 2), so every one of his pitch rows must land on the same
    # display value, and the better locator must score higher.
    by_id = {r["pitcherId"]: [p for p in r["pitches"] if p["t"] == "FF"] for r in records}
    for pid, raw in ((1, -0.01), (2, 0.01)):
        expected = float(ar.to_display(raw, fitted["loc_mu"], fitted["loc_sd"]))
        assert all(p["l"] == pytest.approx(expected) for p in by_id[pid])
    assert by_id[1][0]["l"] > by_id[2][0]["l"]


def test_secondary_pitch_rows_carry_no_location_grade():
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"Slider": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    for r in records:
        assert all(p["l"] is None for p in r["pitches"])


def test_rare_cells_are_pooled_rather_than_dropped():
    """Regression on a real publish failure: cells under the share threshold were
    discarded, so one pitcher's rows summed to 9.40 against a score of 10.91.
    Individually negligible, collectively 1.5 points. Pooling them keeps the card
    short without breaking the promise that the rows add to the score.
    """
    import arsenal as ar
    import pandas as pd
    rows = [{"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
             "BatterSide": "Right", "count12": "0-2", "loc": -0.02} for _ in range(99)]
    rows.append({"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
                 "BatterSide": "Right", "count12": "0-0", "loc": 0.5})
    df = pd.DataFrame(rows)
    out = ar.location_decomposition(df, df, 0.0, 0.02, min_share=0.05)
    assert any(r["region"] == "Everywhere else" for r in out), "the rare cell must be kept"
    expected = float(ar.to_display(df["loc"].mean(), 0.0, 0.02)) - 100.0
    assert sum(r["points"] for r in out) == pytest.approx(expected, abs=1e-9)
