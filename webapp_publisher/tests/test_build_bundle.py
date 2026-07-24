import json, pathlib
from webapp_publisher.build_bundle import build_bundle

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
import pytest, json, pathlib

def test_validate_bundle_rejects_bad_flag():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    bundle["staff_board.json"]["pitchers"][0]["locFlag"] = "nope"
    with pytest.raises(ValueError):
        validate_bundle(bundle)
