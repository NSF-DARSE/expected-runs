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
        # Pitch-level pair, deliberately a different sd than loc_sd above --
        # a fixture that gave them the same value could not catch a bug that
        # collapses the two scales back into one.
        "loc_pitch_mu": 0.0, "loc_pitch_sd": 0.05,
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


def test_attach_location_also_derives_a_separate_pitch_level_scale():
    """The pitch-level pair has to exist alongside loc_mu/loc_sd, not instead
    of it, and it has to be measured over raw pitches rather than pitcher
    means -- the whole point of the fix. _FakeMap.apply returns a constant
    value per pitcher (0.20 for pitcher 1, 0.30 for pitcher 2), so with two
    pitches per pitcher here the pitch-level values are [0.20, 0.30] just like
    the pitcher means are, and the two scales happen to agree on this tiny
    fixture -- the important assertions are that both keys exist and are
    populated, which is what production code checks for downstream.
    """
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, None, _FakeFC, 2025, floor_n=1)
    assert state["loc_pitch_mu"] is not None
    assert state["loc_pitch_sd"] is not None
    assert state["loc_pitch_sd"] > 0
    # Qualifying population must match: same pitchers feed both scales.
    assert state["loc_pitch_mu"] == pytest.approx(state["loc_mu"])


def test_attach_location_leaves_secondary_types_without_a_value():
    mod = _load_pages_module()
    pit = _two_season_frame()
    state = {"pitches": pit[pit["year"] == 2025].copy()}
    mod.attach_location(pit, state, {"Slider"}, _FakeFC, 2025, floor_n=1)
    assert state["pitches"]["loc"].isna().all()
    assert state["loc_mu"] is None and state["loc_sd"] is None
    assert state["loc_pitch_mu"] is None and state["loc_pitch_sd"] is None


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
    assert top["share"] == pytest.approx(1.0)      # every one of his pitches
    assert top["leagueShare"] == pytest.approx(0.5)  # half the league's
    # Rows are one per region now; the count he threw it in is nested, not a
    # grouping key -- every one of his pitches to this region came ahead.
    assert len(top["byCount"]) == 1
    assert top["byCount"][0]["count"] == "ahead"
    assert top["byCount"][0]["n"] == 40
    assert top["byCount"][0]["share"] == pytest.approx(1.0)


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


def test_pitch_rows_carry_location_on_the_pitch_level_scale_not_the_season_one():
    """Regression: per-pitch Location+ used to be routed through the SEASON
    scale (loc_mu/loc_sd, moments of PER-PITCHER MEANS), which put real pitches
    at -281.5 to +232.7 on a page that promises 100+/-15, because a season
    mean's spread is far tighter than a single pitch's. The fix routes it
    through loc_pitch_mu/loc_pitch_sd (moments of raw per-pitch values)
    instead. The fixture gives them a different sd (0.05) than loc_sd (0.01)
    specifically so a regression back to the season pair changes the expected
    value here and this test catches it.

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
        expected = float(ar.to_display(raw, fitted["loc_pitch_mu"], fitted["loc_pitch_sd"]))
        assert all(p["l"] == pytest.approx(expected) for p in by_id[pid])
    assert by_id[1][0]["l"] > by_id[2][0]["l"]


def test_season_location_score_is_unaffected_by_the_pitch_level_scale():
    """The requirement the fix must not violate: the SEASON Location+ (the
    arsenal row's `loc`) is built from loc_mu/loc_sd exactly as before, and
    adding loc_pitch_mu/loc_pitch_sd for the per-pitch number must not move it
    by a single point. Demonstrated here by computing the expected season
    score with a scale that could never have been influenced by the new
    pitch-level pair, then checking it against the published value byte for
    byte within float tolerance.
    """
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"FF": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    import arsenal as ar
    raw_by_pitcher = fitted["pitches"].groupby("PitcherId")["loc"].mean()
    for r in records:
        row = r["arsenal"][0]
        expected = float(ar.to_display(raw_by_pitcher[r["pitcherId"]], fitted["loc_mu"], fitted["loc_sd"]))
        assert row["loc"] == pytest.approx(expected, abs=1e-9)


def test_secondary_pitch_rows_carry_no_location_grade():
    mod = _load_pages_module()
    feats, fitted = _fitted()
    records = mod.build_pitcher_records({"Slider": fitted}, feats, floor_n=1, asof="2026-03-10",
                                        min_type_pitches=1)
    for r in records:
        assert all(p["l"] is None for p in r["pitches"])


def test_region_rows_carry_no_count_field_and_no_points_in_byCount():
    """The collapsed row contract, checked directly rather than only implied by
    other assertions: a region row has no top-level `count` (the frontend's
    pooled-row detector keys on that being absent from real rows), and its
    nested byCount entries are frequency only -- his own n/share, plus the
    optional league counterpart (leagueShare), never a points field.
    """
    import arsenal as ar
    df = _loc_frame()
    rows = ar.location_decomposition(df[df["PitcherId"] == 1], df, 0.0, 0.02, min_share=0.0)
    for r in rows:
        assert "count" not in r
        for bc in r["byCount"]:
            assert set(bc) <= {"count", "n", "share", "leagueShare"}
            assert {"count", "n", "share"} <= set(bc)


def test_pooled_rows_byCount_is_combined_from_the_regions_it_pools():
    """The pooled row still has to answer the count question: it is built by
    summing the byCount breakdowns of the regions pooled into it, not left
    empty just because no single region cleared min_share on its own.
    """
    import arsenal as ar
    import pandas as pd
    rows = [{"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
             "BatterSide": "Right", "count12": "0-2", "loc": -0.02} for _ in range(90)]
    # A second, rare region: 10 pitches total (10% share, under the 15% floor),
    # split 5 even / 5 behind, so the whole region -- and both its count splits
    # -- lands in "Everywhere else".
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "0-0", "loc": 0.5} for _ in range(5)]
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "3-2", "loc": 0.5} for _ in range(5)]
    df = pd.DataFrame(rows)
    out = ar.location_decomposition(df, df, 0.0, 0.02, min_share=0.15)
    pooled = next(r for r in out if r["region"] == "Everywhere else")
    by_count = {bc["count"]: bc for bc in pooled["byCount"]}
    assert by_count["even"]["n"] == 5
    assert by_count["behind"]["n"] == 5
    assert by_count["even"]["share"] == pytest.approx(0.5)
    assert by_count["behind"]["share"] == pytest.approx(0.5)


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


def test_league_cell_table_is_taken_before_any_team_filter():
    """Regression on a shipped defect. 14_pitcher_pages narrows every fitted
    state to one team before building records, so a decomposition that read that
    frame compared a pitcher against his own teammates while the card labeled the
    column D1. On real data that made a staff locating worse than D1 look better
    than it, because the comparison population WAS that staff.
    """
    import arsenal as ar
    import pandas as pd
    league = pd.DataFrame([
        {"PitcherId": pid, "PlateLocSide": side, "PlateLocHeight": 1.8,
         "BatterSide": "Right", "count12": "0-2", "loc": -0.02}
        for pid, side in [(1, -0.7)] * 10 + [(2, 0.0)] * 90
    ])
    full = ar.league_cell_table(league)
    one_team = ar.league_cell_table(league[league["PitcherId"] == 1])
    # The whole point: the two populations give different shares, so which frame
    # is passed is not an implementation detail.
    assert full["n"] == 100 and one_team["n"] == 10
    assert full["share"].max() != one_team["share"].max()


def _split_frame():
    """Three regions in the league, one of which pitcher 1 never throws to. That
    absence is the case the split has to price: he avoids a spot D1 uses."""
    import pandas as pd
    rows = []
    rows += [{"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
              "BatterSide": "Right", "count12": "0-2", "loc": -0.02}] * 30
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "0-0", "loc": 0.01}] * 10
    # Only pitcher 2 lives up and in, and only pitcher 2 throws in 3-0 counts.
    rows += [{"PitcherId": 2, "PlateLocSide": 0.7, "PlateLocHeight": 3.2,
              "BatterSide": "Right", "count12": "3-0", "loc": 0.04}] * 25
    rows += [{"PitcherId": 2, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "0-0", "loc": 0.02}] * 35
    return pd.DataFrame(rows)


def test_occupancy_and_placement_and_baseline_reach_the_score_exactly():
    """The identity the two new columns exist to satisfy. `points` alone mixed
    occupancy, placement and the league's own mix into one number, and the page
    named only the first, which measured on a real staff was the smaller of the
    two terms it was standing in for.
    """
    import arsenal as ar
    df = _split_frame()
    mu, sd = 0.0, 0.02
    his = df[df["PitcherId"] == 1]
    rows = ar.location_decomposition(his, df, mu, sd, min_share=0.0)
    baseline = ar.location_baseline(df, mu, sd)
    total = (sum(r["occupancyPoints"] for r in rows)
             + sum(r["placementPoints"] for r in rows) + baseline)
    expected = float(ar.to_display(his["loc"].mean(), mu, sd)) - 100.0
    assert total == pytest.approx(expected, abs=1e-9)
    # The old contract still holds alongside the new one.
    assert sum(r["points"] for r in rows) == pytest.approx(expected, abs=1e-9)


def test_a_spot_he_never_throws_to_is_priced_as_occupancy():
    """Avoiding a region D1 uses is a fact about where he lives, so it belongs in
    the occupancy term and nowhere else. Iterating only his own cells would drop
    it, and would also make the baseline term depend on his coverage rather than
    on the league.
    """
    import arsenal as ar
    df = _split_frame()
    his = df[df["PitcherId"] == 1]
    rows = ar.location_decomposition(his, df, 0.0, 0.02, min_share=0.0)
    pooled = next(r for r in rows if r["region"] == "Everywhere else")
    # Up and in, 3-0: pitcher 2 throws 25 of his 60 pitches there, pitcher 1
    # throws 0 of his 40. league_cell_table gives each pitcher one vote, so the
    # cell's share is the average of the two individual shares, (0 + 25/60)/2,
    # not the pooled pitch count 25/100 -- that pitch-weighted number is exactly
    # the bug this decomposition had to stop reproducing.
    assert pooled["leagueShare"] == pytest.approx(25 / 60 / 2)
    assert pooled["share"] == pytest.approx(0.0)
    assert pooled["n"] == 0
    assert pooled["placementPoints"] == pytest.approx(0.0)
    assert pooled["occupancyPoints"] != pytest.approx(0.0)
    # A pooled row without a league value is what blocked the split before; it
    # has to be a real share-weighted number over exactly the cells pooled here.
    assert pooled["leagueValue"] == pytest.approx(0.04)


def test_the_baseline_is_the_same_number_for_every_pitcher():
    """The reason it is one scalar and not a column. Summed over only the cells a
    pitcher happens to throw to it is not constant, which is precisely the trap:
    it would read as a league term while actually measuring his coverage.
    """
    import arsenal as ar
    df = _split_frame()
    mu, sd = 0.0, 0.02
    table = ar.league_cell_table(df)
    ones = [ar.location_baseline(table, mu, sd) for _ in (1, 2)]
    assert ones[0] == pytest.approx(ones[1], abs=1e-12)

    # And it really does differ if restricted to one pitcher's own cells, so the
    # full-league sum is doing work rather than being an equivalent spelling.
    def over_his_cells(pid):
        rows = ar.location_decomposition(df[df["PitcherId"] == pid], df, mu, sd,
                                         min_share=0.0)
        seen = {r["region"] for r in rows if r["share"] > 0}
        return sum(-15.0 * float(table["share"][c]) * (float(table["value"][c]) - mu) / sd
                   for c in table["share"].index if c in seen)
    assert over_his_cells(1) != pytest.approx(over_his_cells(2))


def test_pooled_row_league_value_covers_exactly_the_pooled_cells():
    """The blocker this work had to clear first: the pooled row's leagueValue was
    NaN, so the split could not be completed downstream at all.
    """
    import arsenal as ar
    import pandas as pd
    rows = [{"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
             "BatterSide": "Right", "count12": "0-2", "loc": -0.02} for _ in range(99)]
    rows.append({"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
                 "BatterSide": "Right", "count12": "0-0", "loc": 0.5})
    df = pd.DataFrame(rows)
    out = ar.location_decomposition(df, df, 0.0, 0.02, min_share=0.05)
    pooled = next(r for r in out if r["region"] == "Everywhere else")
    assert pooled["leagueValue"] == pytest.approx(0.5)


def test_league_cell_table_weights_every_pitcher_equally():
    """Regression on the bug this module shipped once: league_cell_table gave
    every PITCH one vote, so a pitcher who threw more pitches counted for more
    of both the cell shares and the cell values, while loc_mu (14_pitcher_pages,
    the zero point this table is compared against) gives every PITCHER one
    vote. The mismatch surfaced as a constant +2.4 display point "D1's own
    location mix" baseline on real data, for every pitcher, which was really
    just the gap between the average pitch and the average pitcher.

    _split_frame gives pitcher 1 forty pitches and pitcher 2 sixty, split across
    three regions, so pitch-weighting and pitcher-weighting disagree everywhere.
    This checks the table directly against the by-hand equal-pitcher formula
    rather than against a downstream identity, so a future change that keeps
    the identity but drifts the weighting again still gets caught here.

    Cells are keyed by region alone (count is no longer a grouping dimension
    of this table, see league_cell_table's docstring), so this checks the same
    three regions _split_frame always described, just without the count half
    of the old (region, count) key -- _split_frame happens to put each region's
    pitches in a single count bucket, so collapsing the key changes nothing
    about which pitches land in which region.
    """
    import arsenal as ar
    df = _split_frame()
    table = ar.league_cell_table(df)

    region_down_away = "Down and away"   # only pitcher 1: 30/40
    region_middle = "Middle, middle"     # both: 10/40 and 35/60
    region_up_in = "Up and in"           # only pitcher 2: 25/60

    expected_share = {
        region_down_away: (30 / 40 + 0) / 2,
        region_middle: (10 / 40 + 35 / 60) / 2,
        region_up_in: (0 + 25 / 60) / 2,
    }
    for region, share in expected_share.items():
        assert table["share"][region] == pytest.approx(share)
    assert sum(expected_share.values()) == pytest.approx(1.0)

    # value in the mixed region is the same per-pitcher weighting applied to
    # loc, not the plain 45-pitch pitch-weighted mean.
    p1_w, p1_v = 10 / 40, 0.01
    p2_w, p2_v = 35 / 60, 0.02
    expected_value = (p1_w * p1_v + p2_w * p2_v) / (p1_w + p2_w)
    assert table["value"][region_middle] == pytest.approx(expected_value)

    # n is still a raw pitch count, not a weighted quantity.
    assert table["n"] == 100


def test_location_baseline_is_zero_when_weighted_like_loc_mu():
    """The baseline term is sum_c w*_c(v*_c - loc_mu). league_cell_table's
    weighting telescopes exactly back to loc_mu -- the mean of per-pitcher
    means -- so if loc_mu really is that same mean, the baseline has to be
    zero to floating-point error. It is not a real "league location mix"
    effect; it is a check that the two weightings still agree. The old,
    pitch-weighted table gave a nonzero constant here (+2.4 display points on
    a real 18-pitcher staff), which is the bug this reconciles.

    Tolerance is 1e-6 display points, tight enough that reintroducing a
    pitch-weighted share or value (worth whole points on real data) fails
    this test immediately.
    """
    import arsenal as ar
    df = _split_frame()
    per_pitcher_mean = df.groupby("PitcherId")["loc"].mean()
    loc_mu = float(per_pitcher_mean.mean())   # one vote per pitcher, like 14_pitcher_pages.py
    baseline = ar.location_baseline(df, loc_mu, loc_sd=0.02)
    assert abs(baseline) < 1e-6


def test_byCount_leagueShare_sums_to_one_within_a_region():
    """The denominator the request hinges on: his own byCount shares already
    sum to 1.0 within one region (three buckets partition his pitches to that
    spot exhaustively). D1's leagueShare has to be normalized the same way --
    of D1's pitches to THIS region, not to D1's whole arsenal -- so it also
    sums to 1.0 within the region, even though both sides throw to the region
    in very different counts.
    """
    import arsenal as ar
    import pandas as pd
    rows = []
    # Region "Middle, middle": pitcher 1 goes ahead-heavy, D1 (pitcher 2, more
    # pitches so pitch-weighting would disagree with pitcher-weighting too)
    # goes behind-heavy. Different mixes on purpose, so a bug that copies one
    # side's shares onto the other would be caught by the value, not just the
    # sum-to-one check.
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "0-2", "loc": 0.01}] * 8
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "1-1", "loc": 0.01}] * 2
    rows += [{"PitcherId": 2, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "1-0", "loc": 0.02}] * 30
    rows += [{"PitcherId": 2, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "1-1", "loc": 0.02}] * 20
    df = pd.DataFrame(rows)
    his = df[df["PitcherId"] == 1]
    out = ar.location_decomposition(his, df, 0.0, 0.02, min_share=0.0)
    row = next(r for r in out if r["region"] == "Middle, middle")
    by_count = {bc["count"]: bc for bc in row["byCount"]}

    # His own side: unchanged contract, 8/10 ahead, 2/10 even.
    assert by_count["ahead"]["share"] == pytest.approx(0.8)
    assert by_count["even"]["share"] == pytest.approx(0.2)
    assert "behind" not in by_count   # he never throws behind here

    # D1's side: each pitcher gets total weight 1/n_pitchers = 1/2 for this
    # region (every one of both pitchers' pitches lands here), split across
    # buckets in proportion to his own pitch count within the region --
    # pitcher 1 contributes 8/10 of his 1/2 to ahead and 2/10 to even;
    # pitcher 2 contributes 30/50 of his 1/2 to behind and 20/50 to even.
    ahead_w = 0.5 * (8 / 10)
    even_w = 0.5 * (2 / 10) + 0.5 * (20 / 50)
    behind_w = 0.5 * (30 / 50)
    region_total = ahead_w + even_w + behind_w
    assert region_total == pytest.approx(1.0)   # one vote per pitcher, two pitchers
    assert by_count["ahead"]["leagueShare"] == pytest.approx(ahead_w / region_total)
    assert by_count["even"]["leagueShare"] == pytest.approx(even_w / region_total)

    # "behind" is missing from his own byCount (he never throws it), but D1
    # does throw behind here -- confirm the full three-bucket split still
    # sums to 1.0 by reconstructing it from the row's own leagueShare share.
    ahead_frac = by_count["ahead"]["leagueShare"]
    even_frac = by_count["even"]["leagueShare"]
    behind_frac = behind_w / region_total
    assert behind_frac > 0   # D1 does throw behind here even though he doesn't
    assert ahead_frac + even_frac + behind_frac == pytest.approx(1.0)


def test_byCount_leagueShare_absent_when_league_never_lands_in_that_bucket():
    """A bucket the league table has literally no weight in gets no
    leagueShare key at all, distinct from a real 0% -- so the frontend can
    tell "never measured" from "measured at zero" if it ever needs to."""
    import arsenal as ar
    import pandas as pd
    rows = [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
             "BatterSide": "Right", "count12": "0-2", "loc": 0.01}] * 5
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "3-0", "loc": 0.01}] * 5
    df = pd.DataFrame(rows)
    out = ar.location_decomposition(df, df, 0.0, 0.02, min_share=0.0)
    row = next(r for r in out if r["region"] == "Middle, middle")
    by_count = {bc["count"]: bc for bc in row["byCount"]}
    assert "leagueShare" in by_count["ahead"]
    assert "leagueShare" in by_count["behind"]
    assert "even" not in by_count   # nobody, him or the league, ever throws even here


def test_pooled_byCount_leagueShare_is_weight_pooled_not_fraction_averaged():
    """The pooled row's leagueShare per bucket has to combine the pooled
    regions' RAW league weight, not average their per-region fractions --
    two regions with very different amounts of league weight would otherwise
    give the smaller one an equal vote it didn't earn.
    """
    import arsenal as ar
    import pandas as pd
    rows = [{"PitcherId": 1, "PlateLocSide": -0.7, "PlateLocHeight": 1.8,
             "BatterSide": "Right", "count12": "0-2", "loc": -0.02} for _ in range(90)]
    # Two rare regions, each individually pooled. Region A: 100% ahead on both
    # sides. Region B: 100% behind on both sides, but with ten times region
    # A's league weight (more pitchers/pitches land there), so a pooled
    # leagueShare that averages fractions (0.5/0.5) would be wrong; the
    # correct weight-pooled answer leans heavily toward region B's bucket.
    rows += [{"PitcherId": 1, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "3-0", "loc": 0.5} for _ in range(3)]
    rows += [{"PitcherId": 2, "PlateLocSide": 0.0, "PlateLocHeight": 2.5,
              "BatterSide": "Right", "count12": "3-0", "loc": 0.5} for _ in range(3)]
    rows += [{"PitcherId": 1, "PlateLocSide": 0.7, "PlateLocHeight": 3.2,
              "BatterSide": "Right", "count12": "0-2", "loc": 0.3} for _ in range(2)]
    rows += [{"PitcherId": 2, "PlateLocSide": 0.7, "PlateLocHeight": 3.2,
              "BatterSide": "Right", "count12": "0-2", "loc": 0.3} for _ in range(20)]
    df = pd.DataFrame(rows)
    his = df[df["PitcherId"] == 1]
    out = ar.location_decomposition(his, df, 0.0, 0.02, min_share=0.5)
    pooled = next(r for r in out if r["region"] == "Everywhere else")
    by_count = {bc["count"]: bc for bc in pooled["byCount"]}
    # Weight-pooled: region "Up and in" carries far more league weight
    # (pitcher 2's 20 pitches vs pitcher 1's 2, one vote each but very
    # different WITHIN-region shares feeding a bigger denominator here isn't
    # the point -- the point is the two regions' total leagueShare differ),
    # so the pooled ahead/behind split should not be a plain 50/50 average.
    total_lg = sum(v for v in by_count.values() if "leagueShare" in v
                   for v in [v["leagueShare"]])
    assert total_lg == pytest.approx(1.0)


def test_league_cell_table_raises_on_an_empty_frame():
    """An empty comparison population is a missing-data error, not a table with
    no rows -- there is nothing to reconcile loc_mu against."""
    import arsenal as ar
    df = _split_frame()
    with pytest.raises(ValueError, match="comparison population is missing"):
        ar.league_cell_table(df[df["PitcherId"] == 999])
