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
                     "trend": {"stuff": {"recent": 121.0, "prior": 118.4, "delta": 2.6,
                                         "se": 1.9, "nRecent": 160, "nPrior": 140,
                                         "direction": "up"},
                               "velo": {"recent": 92.1, "prior": 91.4, "delta": 0.7,
                                        "se": 0.15, "nRecent": 160, "nPrior": 140,
                                        "direction": "up"},
                               "movAngle": None, "movMag": None},
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


def test_non_positive_display_sd_is_rejected():
    """A zero or negative display sd means the scale is degenerate; every score
    derived from it would be garbage or a division error.
    """
    bad = copy.deepcopy(GOOD)
    bad["model_artifacts.json"]["byPitchType"]["FF"]["displaySd"] = 0.0
    with pytest.raises(ValueError, match="displaySd"):
        validate_pitcher_bundle(bad)
