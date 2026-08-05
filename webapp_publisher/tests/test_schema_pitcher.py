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


def test_fastball_missing_its_location_score_is_rejected():
    """The converse of the secondary-pitch check, and the gap a real bug already
    slipped through: a NaN Location+ becomes None by the time it reaches here, and
    a fastball silently missing its headline column must not ship.
    """
    bad = copy.deepcopy(GOOD)
    bad["pitchers/1000123.json"]["arsenal"][0]["loc"] = None
    with pytest.raises(ValueError, match="numeric Location"):
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
