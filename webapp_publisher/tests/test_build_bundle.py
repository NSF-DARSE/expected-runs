import json, pathlib
from webapp_publisher.build_bundle import build_bundle, to_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "staff_scores.json"

def test_build_bundle_shapes_manifest_and_board():
    staff_scores = json.loads(FIX.read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="2026-07-24T02:00:00Z")
    # stamp_pitcher_ids is called in publish.py before validation, so we don't test it here.
    # This test validates build_bundle's output shape only.
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101

    manifest = bundle["manifest.json"]
    assert manifest == {"built": "2026-07-24T02:00:00Z", "season": 2026,
                        "dataThrough": "2026-03-15", "bundleVersion": "2026-07-24T02:00:00Z"}

    board = bundle["staff_board.json"]
    assert board["team"] == "DEL_BLU"
    assert board["population"] == 543
    row = board["pitchers"][0]
    assert set(row) == {"id","name","hand","ff","stuff","loc","adjres","pitch",
                        "whiff","zone","heart","meanHeight","locFlag","stuffAttr",
                        "stuffNoHand","pitchNoHand","stuffAttrNoHand","pitcherId"}
    assert isinstance(row["id"], int)
    assert row["locFlag"] in ("", "caution", "small sample")
    assert isinstance(row["stuffAttr"], list) and isinstance(row["stuffAttr"][0], list)
    assert isinstance(row["stuffNoHand"], (int, float))
    assert isinstance(row["pitchNoHand"], (int, float))
    assert isinstance(row["stuffAttrNoHand"], list) and isinstance(row["stuffAttrNoHand"][0], list)
    assert len(row["stuffAttrNoHand"][0]) == 2

def test_ids_are_stable_across_calls():
    staff_scores = json.loads(FIX.read_text())
    b1 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    b2 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    ids1 = {r["name"]: r["id"] for r in b1["staff_board.json"]["pitchers"]}
    ids2 = {r["name"]: r["id"] for r in b2["staff_board.json"]["pitchers"]}
    assert ids1 == ids2

from webapp_publisher.schema import validate_bundle
import pytest

def _stub_stuff_attr_detail(row: dict) -> dict:
    # validate_bundle is exercised here without going through publish.py's
    # enrich_stuff_attr_detail, so these tests build the same all-null shape
    # that step produces for a pitcher with no matching arsenal, by hand.
    names = {f for f, _ in row["stuffAttr"]} | {f for f, _ in row["stuffAttrNoHand"]}
    return {n: {"value": None, "percentile": None} for n in names}

def test_validate_bundle_rejects_bad_flag():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    bundle["staff_board.json"]["pitchers"][0]["locFlag"] = "nope"
    with pytest.raises(ValueError):
        validate_bundle(bundle)

def test_validate_bundle_rejects_missing_manifest_key():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    del bundle["manifest.json"]["season"]
    with pytest.raises(ValueError):
        validate_bundle(bundle)

def test_validate_bundle_rejects_empty_pitchers():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    bundle["staff_board.json"]["pitchers"] = []
    with pytest.raises(ValueError):
        validate_bundle(bundle)


def test_validate_bundle_rejects_stuff_attr_detail_missing_a_named_feature():
    """Every feature named in stuffAttr/stuffAttrNoHand must have a detail
    entry, even a null one; a gap here means the enrichment step skipped a
    trait rather than nulling it out.
    """
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    bundle["staff_board.json"]["pitchers"][0]["stuffAttrDetail"] = {}
    with pytest.raises(ValueError, match="stuffAttrDetail missing"):
        validate_bundle(bundle)


def test_validate_bundle_rejects_a_value_with_no_matching_percentile():
    """value and percentile come from the same positional lookup, so one being
    present without the other means the join half-ran, not that data is
    genuinely absent.
    """
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    row = bundle["staff_board.json"]["pitchers"][0]
    any_feature = next(iter(row["stuffAttrDetail"]))
    row["stuffAttrDetail"][any_feature] = {"value": 91.2, "percentile": None}
    with pytest.raises(ValueError, match="null together"):
        validate_bundle(bundle)


def test_validate_bundle_rejects_out_of_range_stuff_attr_percentile():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    row = bundle["staff_board.json"]["pitchers"][0]
    any_feature = next(iter(row["stuffAttrDetail"]))
    row["stuffAttrDetail"][any_feature] = {"value": 91.2, "percentile": 140}
    with pytest.raises(ValueError, match="outside 0-100"):
        validate_bundle(bundle)


def test_to_native_handles_numpy_scalars_and_arrays():
    np = pytest.importorskip("numpy")

    i = to_native(np.int64(5))
    assert i == 5 and isinstance(i, int)

    f = to_native(np.float64(1.5))
    assert f == 1.5 and isinstance(f, float)

    assert to_native(np.float64("nan")) is None
    assert to_native(np.float64("inf")) is None

    arr = to_native(np.array([1, 2]))
    assert arr == [1, 2] and isinstance(arr, list)

    nested = to_native({"a": np.int64(3), "b": [np.float64("nan")]})
    assert nested == {"a": 3, "b": [None]}

def test_validate_bundle_rejects_a_board_decomposition_with_no_baseline():
    """The hover card adds occupancy, placement and the league's own mix up to
    the score beside it, and the third term is not recoverable from the rows. A
    board row carrying the decomposition without it would come up short.
    """
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    for r in bundle["staff_board.json"]["pitchers"]:
        r["pitcherId"] = 1000101
        r["stuffAttrDetail"] = _stub_stuff_attr_detail(r)
    bundle["staff_board.json"]["pitchers"][0]["locWhere"] = [
        {"region": "Down and away", "count": "ahead", "n": 41, "share": 0.31,
         "leagueShare": 0.18, "points": 14.0, "occupancyPoints": 9.0,
         "placementPoints": 4.5, "value": -0.004, "leagueValue": -0.002}]
    with pytest.raises(ValueError, match="no numeric locBaseline"):
        validate_bundle(bundle)
