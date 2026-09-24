"""The Command+ bundle section: sourcing, refusal of a half-configured run, schema."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from webapp_publisher.build_command_bundle import build_command_section, resolve_tags_dir
from webapp_publisher.schema import validate_command_pairs

T0 = datetime(2026, 9, 2, 18, 0, 0, tzinfo=timezone.utc)
STEP = timedelta(seconds=12)


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fixture(tmp_path, n=5):
    tags = tmp_path / "tags" / "2026-09-02" / "1000000001"
    tags.mkdir(parents=True)
    session = {"sessionId": "2026-09-02__1000000001__T180000Z", "pitcherId": "1000000001",
               "pitcherName": "Test-Pitcher, Alpha", "date": "2026-09-02", "rev": n,
               "tags": [{"seq": i + 1, "zone": (i % 5) + 1,
                         "pitchClass": "fastball" if i % 2 == 0 else "offspeed",
                         "at": _iso(T0 + i * STEP), "skipped": False} for i in range(n)]}
    (tags / f"{session['sessionId']}.json").write_text(json.dumps(session))
    csv = tmp_path / "tree" / "2026" / "09" / "02" / "CSV"
    csv.mkdir(parents=True)
    pd.DataFrame([{"PitchNo": i + 1, "Date": "2026-09-02", "UTCDateTime": _iso(T0 + i * STEP),
                   "PitcherId": 1000000001, "PitcherThrows": "Left", "BatterSide": "Right",
                   "TaggedPitchType": "Undefined", "PlateLocSide": 0.1 * i,
                   "PlateLocHeight": 2.5, "PitchUID": f"u{i}", "Level": "TeamExclusive"}
                  for i in range(n)]).to_csv(csv / "pen.csv", index=False)
    return str(tmp_path / "tags"), str(tmp_path / "tree")


def test_no_tag_source_ships_an_empty_section():
    out = build_command_section(tags_dir=None, pitch_roots=None)
    validate_command_pairs(out)
    assert out["sessions"] == [] and "no tag source" in out["note"]


def test_tags_without_pitch_roots_is_refused(tmp_path):
    tags, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="COMMAND_PITCH_ROOTS"):
        build_command_section(tags_dir=tags, pitch_roots=None)


def test_pull_without_connection_string_is_refused(tmp_path):
    with pytest.raises(ValueError, match="CONNECTION_STRING"):
        build_command_section(tags_dir=str(tmp_path), pitch_roots=None, pull=True)


def test_pull_defaults_the_mirror_into_the_workdir(tmp_path):
    assert resolve_tags_dir(None, str(tmp_path), pull=True).endswith("session_tags")
    assert resolve_tags_dir(None, str(tmp_path), pull=False) is None
    assert resolve_tags_dir("x", str(tmp_path), pull=True) == "x"


def test_full_path_builds_a_valid_section(tmp_path):
    tags, tree = _fixture(tmp_path)
    out = build_command_section(tags_dir=tags, pitch_roots=tree)
    validate_command_pairs(out)
    s = out["sessions"][0]
    assert s["status"] == "joined" and s["throws"] == "Left" and s["batterSide"] == "Right"
    assert len(s["pairs"]) == 5
    assert "Test-Pitcher" not in json.dumps(out)


@pytest.mark.parametrize("mutate, match", [
    (lambda p: p.update(zone=6), "zone"),
    (lambda p: p.update(pitchClass="Fastball"), "pitchClass"),
    (lambda p: p.update(side="0.3"), "non-numeric side"),
    (lambda p: p.update(height=float("nan")), "non-numeric height"),
])
def test_schema_rejects_unscorable_pairs(tmp_path, mutate, match):
    tags, tree = _fixture(tmp_path)
    out = build_command_section(tags_dir=tags, pitch_roots=tree)
    mutate(out["sessions"][0]["pairs"][0])
    with pytest.raises(ValueError, match=match):
        validate_command_pairs(out)


def test_schema_rejects_an_unknown_batter_side(tmp_path):
    tags, tree = _fixture(tmp_path)
    out = build_command_section(tags_dir=tags, pitch_roots=tree)
    out["sessions"][0]["batterSide"] = "Undefined"
    with pytest.raises(ValueError, match="batterSide"):
        validate_command_pairs(out)


def test_schema_rejects_an_unjoined_session_with_pairs(tmp_path):
    tags, tree = _fixture(tmp_path)
    out = build_command_section(tags_dir=tags, pitch_roots=tree)
    out["sessions"][0]["status"] = "unjoined"
    with pytest.raises(ValueError, match="reason"):
        validate_command_pairs(out)
