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
                     "avgVelo": 93.1, "aboveFloor": True,
                     # points sum to loc - 100, and occupancy + placement +
                     # locBaseline reach the same 3.0 by the other route.
                     # Rows are one per region; the count split lives only in
                     # each row's byCount (frequency, no points -- see
                     # arsenal.location_decomposition).
                     "locWhere": [{"region": "Down and away", "n": 200,
                                   "share": 0.5, "leagueShare": 0.3, "points": 3.0,
                                   "occupancyPoints": 2.0, "placementPoints": 0.5,
                                   "value": -0.01, "leagueValue": -0.005,
                                   # leagueShare here is optional (see schema.py) --
                                   # present on this row to cover the common case,
                                   # while the other row below omits it to keep
                                   # covering a bundle from before this field
                                   # existed, or a bucket the league never lands in.
                                   "byCount": [{"count": "ahead", "n": 200, "share": 1.0,
                                                "leagueShare": 0.9}]},
                                  {"region": "Up, middle", "n": 212,
                                   "share": 0.5, "leagueShare": 0.4, "points": 0.0,
                                   "occupancyPoints": 0.0, "placementPoints": 0.0,
                                   "value": 0.0, "leagueValue": 0.0,
                                   "byCount": [{"count": "even", "n": 212, "share": 1.0}]}],
                     "locBaseline": 0.5,
                     "typical": [2350.0], "percentiles": [78]}],
        "outings": [{"date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.0}],
        "pitches": [{"d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31,
                     "c": "0-2", "g": 131.0, "f": [2350.0]}],
    },
    "staff_by_type.json": {
        "types": [
            {"type": "FF", "label": "Fastball", "nQualified": 400, "sampleFloor": 100,
             "pitchers": [{"pitcherId": 1000123, "name": "Test-Pitcher, Alpha",
                          "hand": "R", "n": 412, "usage": 1.0, "stuff": 124.0,
                          "avgVelo": 93.1, "aboveFloor": True}]},
        ],
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


def test_fastball_missing_its_location_score_is_rejected():
    """The converse of the secondary-pitch check, and the gap a real bug already
    slipped through: a NaN Location+ becomes None by the time it reaches here, and
    a fastball silently missing its headline column must not ship.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["loc"] = None
    with pytest.raises(ValueError, match="numeric Location"):
        validate_pitcher_bundle(bad)


def test_raw_run_value_location_score_is_rejected():
    """Regression: `loc` shipped once as the bare mean expected-run value (~0.00x,
    lower = better) instead of the negated 100 +/- 15 display score. Numeric-ness
    alone accepted it, so the range is what has to catch it.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["loc"] = 0.0042
    with pytest.raises(ValueError, match="outside the plausible"):
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


def test_label_identical_to_the_field_name_is_accepted():
    """Regression: a real run failed here. Some features are already plain
    English -- "Extension" needs no translation -- so requiring the label to
    differ from the field name rejected a correct bundle and blocked publish.
    """
    good = copy.deepcopy(GOOD)
    good["model_artifacts.json"]["featureOrder"] = ["Extension"]
    good["model_artifacts.json"]["labels"] = {"Extension": "Extension"}
    validate_pitcher_bundle(good)   # must not raise


def test_empty_label_is_rejected():
    bad = copy.deepcopy(GOOD)
    bad["model_artifacts.json"]["labels"] = {"SpinRate": "   "}
    with pytest.raises(ValueError, match="no plain-English label"):
        validate_pitcher_bundle(bad)


def test_out_of_range_percentile_is_rejected():
    """A percentile outside 0-100 means the reference population was wrong,
    which would put a nonsense rank in front of a coach.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["percentiles"] = [140]
    with pytest.raises(ValueError, match="percentile"):
        validate_pitcher_bundle(bad)


def test_raw_run_value_stuff_score_is_rejected():
    """Regression guard mirroring the loc bug: `stuff` sits on the same
    100+/-15 display scale and is just as exposed to a raw expected-run value
    (~0.00x, lower = better) shipping unscaled.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["stuff"] = -0.0231
    with pytest.raises(ValueError, match="Stuff"):
        validate_pitcher_bundle(bad)


def test_raw_run_value_pitch_grade_is_rejected():
    """Same failure mode as the stuff regression above, but for the per-pitch
    grade `g`.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["pitches"][0]["g"] = 0.0038
    with pytest.raises(ValueError, match="pitch grade"):
        validate_pitcher_bundle(bad)


def test_pitch_grade_at_measured_real_spread_edges_is_accepted():
    """Regression against a band tight enough to break a real publish:
    per-pitch four-seam grades measured on a real bundle range ~20-147 with
    per-pitcher standard deviations of 8.9-16.0, well outside DISPLAY_BAND.
    Both edges of that measured range must pass.
    """
    good_low = copy.deepcopy(GOOD)
    good_low["pitchers/1000123.json"]["pitches"][0]["g"] = 20.0
    validate_pitcher_bundle(good_low)  # must not raise

    good_high = copy.deepcopy(GOOD)
    good_high["pitchers/1000123.json"]["pitches"][0]["g"] = 147.0
    validate_pitcher_bundle(good_high)  # must not raise


def test_pitch_grade_band_admits_an_awful_grade_but_rejects_an_unscaled_one():
    """Pins what the band is actually for, so nobody tightens it back.

    A raw ridge_pred is |v| < ~0.2, so the band's only real job is catching an
    unscaled value; a polarity flip is undetectable here either way, since
    100 + 15z occupies the same range as 100 - 15z. That makes every point of
    extra tightness pure downside. One real team's worst pitch already grades
    19.9, so a genuinely dreadful pitch on some other staff must not abort a
    publish.
    """
    awful = copy.deepcopy(GOOD)
    awful["pitchers/1000123.json"]["pitches"][0]["g"] = 5.0
    validate_pitcher_bundle(awful)  # bad pitch, still a real score: must not raise

    unscaled = copy.deepcopy(GOOD)
    unscaled["pitchers/1000123.json"]["pitches"][0]["g"] = 0.02
    with pytest.raises(ValueError, match="pitch grade"):
        validate_pitcher_bundle(unscaled)


def test_arsenal_pitch_type_missing_from_model_artifacts_is_rejected():
    """The frontend treats a byPitchType entry for every arsenal type as an
    unstated precondition (missing artifact reads a sample floor as 0 and a
    tooltip as "Fewer than 0 pitches this season"). Today it holds by
    construction; this checks it.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["type"] = "SL"
    bad["pitchers/1000123.json"]["arsenal"][0]["loc"] = None
    with pytest.raises(ValueError, match="SL"):
        validate_pitcher_bundle(bad)


def test_missing_average_velocity_is_rejected():
    """avgVelo is required rather than optional. It is the pitch type's own
    context line, so a row without it renders a pitch with no velocity beside it
    and nothing anywhere says the field went missing.
    """
    bad = copy.deepcopy(GOOD)
    del bad["pitchers/1000123.json"]["arsenal"][0]["avgVelo"]
    with pytest.raises(ValueError, match="avgVelo"):
        validate_pitcher_bundle(bad)


def test_average_velocity_in_wrong_units_is_rejected():
    """The failure this band exists for: RelSpeed arriving in m/s. A 93.1 mph
    fastball reads 41.6, which is numeric, positive, and completely wrong on a
    page a coach reads in mph.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["avgVelo"] = 41.6
    with pytest.raises(ValueError, match="mph"):
        validate_pitcher_bundle(bad)


def test_nan_average_velocity_is_rejected():
    """A mean over an all-null RelSpeed slice is NaN, which passes every range
    comparison by failing all of them, and would reach the page as "NaN mph".
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["avgVelo"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_pitcher_bundle(bad)


def test_average_velocity_across_the_real_college_range_is_accepted():
    """The band is a units check, not a quality check. A soft changeup and a
    plus fastball must both publish.
    """
    for velo in (68.0, 98.0):
        good = copy.deepcopy(GOOD)
        good["pitchers/1000123.json"]["arsenal"][0]["avgVelo"] = velo
        validate_pitcher_bundle(good)  # must not raise


def test_non_positive_display_sd_is_rejected():
    """A zero or negative display sd means the scale is degenerate; every score
    derived from it would be garbage or a division error.
    """
    bad = copy.deepcopy(GOOD)
    bad["model_artifacts.json"]["byPitchType"]["FF"]["displaySd"] = 0.0
    with pytest.raises(ValueError, match="displaySd"):
        validate_pitcher_bundle(bad)


def test_type_board_entry_with_no_pitchers_is_rejected():
    """An empty pitchers list for a type would render a staff board section with
    nothing in it; the regroup upstream should never produce this, so a bundle
    that has it is a build-time bug, not a legitimately quiet pitch type.
    """
    bad = copy.deepcopy(GOOD)
    bad["staff_by_type.json"]["types"][0]["pitchers"] = []
    with pytest.raises(ValueError, match="no pitchers for"):
        validate_pitcher_bundle(bad)


def test_type_board_entry_missing_from_model_artifacts_is_rejected():
    """staff_by_type's per-type scale (nQualified, sampleFloor, displaySd) lives
    in model_artifacts.json's byPitchType, keyed by the same type string. A type
    on the board with no matching artifact would render the same broken defaults
    the arsenal check above guards against: a sample floor of 0 and a tooltip
    claiming "Fewer than 0 pitches this season".
    """
    bad = copy.deepcopy(GOOD)
    bad["staff_by_type.json"]["types"][0]["type"] = "SL"
    with pytest.raises(ValueError, match="SL"):
        validate_pitcher_bundle(bad)


def test_unscaled_staff_stuff_value_is_rejected():
    """Regression guard mirroring the arsenal `stuff` check: staff_by_type rows
    carry the same 100+/-15 display Stuff+ and are just as exposed to a raw
    expected-run value (~0.00x, lower = better) shipping unscaled onto the board.
    """
    bad = copy.deepcopy(GOOD)
    bad["staff_by_type.json"]["types"][0]["pitchers"][0]["stuff"] = -0.0231
    with pytest.raises(ValueError, match="Stuff"):
        validate_pitcher_bundle(bad)


def test_location_decomposition_that_does_not_sum_to_its_score_is_rejected():
    """The rows are an exact split of the same mean, so a gap is a bug rather
    than rounding: a mismatched population, a dropped cell, or a lost sign. The
    page presents these as adding up, so a publish that does not add up must
    not ship.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["locWhere"][0]["points"] = 40.0
    with pytest.raises(ValueError, match="do not explain the number"):
        validate_pitcher_bundle(bad)


def test_secondary_pitch_type_carrying_a_location_decomposition_is_rejected():
    """Same construct leak the `loc` check catches, one level down: a slider has
    no Location+, so it cannot have a breakdown of one either.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["type"] = "Slider"
    bad["pitchers/1000123.json"]["arsenal"][0]["loc"] = None
    bad["model_artifacts.json"]["byPitchType"]["Slider"] =         copy.deepcopy(bad["model_artifacts.json"]["byPitchType"]["FF"])
    bad["staff_by_type.json"]["types"][0]["type"] = "Slider"
    with pytest.raises(ValueError, match="fastball score only"):
        validate_pitcher_bundle(bad)


def test_a_bad_but_real_adjusted_results_value_still_publishes():
    """Measured on a real bundle, per-type adjusted results run 28.7 to 140.1:
    the number can rest on under 30 pitches where the fastball board rests on
    100+, so it spreads much wider. DISPLAY_BAND rejected a legitimate changeup
    and aborted a whole publish, which is the failure this pins.
    """
    good = copy.deepcopy(GOOD)
    good["staff_by_type.json"]["types"][0]["pitchers"][0]["adjRes"] = 28.7
    validate_pitcher_bundle(good)  # must not raise


def test_an_unscaled_adjusted_results_value_is_still_rejected():
    """Widening the band must not give up what the band is for: a raw run value
    (|v| < ~0.2) that never went through to_display.
    """
    bad = copy.deepcopy(GOOD)
    bad["staff_by_type.json"]["types"][0]["pitchers"][0]["adjRes"] = -0.031
    with pytest.raises(ValueError, match="Adj Results"):
        validate_pitcher_bundle(bad)


def test_occupancy_plus_placement_plus_baseline_that_misses_the_score_is_rejected():
    """The page prints occupancy and placement as separate columns under the
    score, so the three terms have to reach it. `points` summing correctly is not
    enough: the split is a different arrangement of the same algebra and can be
    wrong on its own.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["locWhere"][0]["occupancyPoints"] = 9.0
    with pytest.raises(ValueError, match="does not explain the number"):
        validate_pitcher_bundle(bad)


def test_fastball_decomposition_without_a_baseline_is_rejected():
    """The baseline is not recoverable from the rows, so a bundle missing it
    leaves the two columns short of the score with nothing to name the gap.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["locBaseline"] = None
    with pytest.raises(ValueError, match=r"no numeric Location\+ baseline"):
        validate_pitcher_bundle(bad)


def test_location_row_missing_a_split_column_is_rejected():
    """A bundle published before the split existed still validates its `points`,
    so the per-row key check is the only thing that catches it.
    """
    bad = copy.deepcopy(GOOD)
    del bad["pitchers/1000123.json"]["arsenal"][0]["locWhere"][0]["placementPoints"]
    with pytest.raises(ValueError, match="missing"):
        validate_pitcher_bundle(bad)


def test_byCount_without_leagueShare_still_validates():
    """leagueShare is optional on a byCount entry: a bundle built between the
    region-collapse and this change, and a fresh bundle's bucket the league
    never lands in, both look like this -- REQUIRED_BY_COUNT_KEYS must not
    grow to include it.
    """
    good = copy.deepcopy(GOOD)
    assert "leagueShare" not in good["pitchers/1000123.json"]["arsenal"][0]["locWhere"][1]["byCount"][0]
    validate_pitcher_bundle(good)


def test_byCount_leagueShare_outside_0_1_is_rejected():
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["locWhere"][0]["byCount"][0]["leagueShare"] = 1.4
    with pytest.raises(ValueError, match="leagueShare outside 0-1"):
        validate_pitcher_bundle(bad)
