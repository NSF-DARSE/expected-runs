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
