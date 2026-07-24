import json, pathlib
from webapp_publisher.build_bundle import build_bundle, to_native

FIX = pathlib.Path(__file__).parent / "fixtures" / "staff_scores.json"

def test_build_bundle_shapes_manifest_and_board():
    staff_scores = json.loads(FIX.read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="2026-07-24T02:00:00Z")

    manifest = bundle["manifest.json"]
    assert manifest == {"built": "2026-07-24T02:00:00Z", "season": 2026,
                        "dataThrough": "2026-03-15", "bundleVersion": "2026-07-24T02:00:00Z"}

    board = bundle["staff_board.json"]
    assert board["team"] == "DEL_BLU"
    assert board["population"] == 543
    row = board["pitchers"][0]
    assert set(row) == {"id","name","hand","ff","stuff","loc","adjres","pitch",
                        "whiff","zone","heart","meanHeight","locFlag","stuffAttr"}
    assert isinstance(row["id"], int)
    assert row["locFlag"] in ("", "caution", "small sample")
    assert isinstance(row["stuffAttr"], list) and isinstance(row["stuffAttr"][0], list)

def test_ids_are_stable_across_calls():
    staff_scores = json.loads(FIX.read_text())
    b1 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    b2 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    ids1 = {r["name"]: r["id"] for r in b1["staff_board.json"]["pitchers"]}
    ids2 = {r["name"]: r["id"] for r in b2["staff_board.json"]["pitchers"]}
    assert ids1 == ids2

from webapp_publisher.schema import validate_bundle
import pytest

def test_validate_bundle_rejects_bad_flag():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    bundle["staff_board.json"]["pitchers"][0]["locFlag"] = "nope"
    with pytest.raises(ValueError):
        validate_bundle(bundle)

def test_validate_bundle_rejects_missing_manifest_key():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    del bundle["manifest.json"]["season"]
    with pytest.raises(ValueError):
        validate_bundle(bundle)

def test_validate_bundle_rejects_empty_pitchers():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    bundle["staff_board.json"]["pitchers"] = []
    with pytest.raises(ValueError):
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
