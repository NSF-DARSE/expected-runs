"""command_pairs.py: tagged bullpen pitches -> (intended zone, actual location) pairs.

Synthetic tags laid out exactly as the app's api stores them, synthetic TrackMan
practice files laid out as the practice tree is. The cases are the ways a bullpen is
not clean: warm-ups before the first tag, every pen under one placeholder pitcher id,
two pens on one day, files with no timestamp, and a session the join refuses. None of
them may produce pairs that look fine and are not, and none may stop the other
sessions from building.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import command_pairs as cp

DAY = "2026-09-02"
T0 = datetime(2026, 9, 2, 18, 0, 0, tzinfo=timezone.utc)
CADENCE = timedelta(seconds=12)
PID = "1000000001"


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _pitches(n, *, start=T0, pitcher_id=PID, throws="Right", level="TeamExclusive",
             with_utc=True, uid_prefix="u", side0=0.0, batter="Left"):
    rows = []
    for i in range(n):
        t = start + i * CADENCE
        row = {
            "PitchNo": i + 1, "Date": DAY, "Pitcher": "Test-Pitcher, Alpha",
            "PitcherId": int(pitcher_id), "PitcherThrows": throws,
            "BatterSide": batter(i) if callable(batter) else batter,
            "TaggedPitchType": "Undefined", "PitchCall": "Undefined",
            "PlateLocSide": side0 + 0.1 * i, "PlateLocHeight": 2.0 + 0.05 * i,
            "PitchUID": f"{uid_prefix}-{i}", "Level": level,
        }
        if with_utc:
            row["UTCDateTime"] = _iso(t)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(root, df, name):
    d = root / "2026" / "09" / "02" / "CSV"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / name, index=False)


def _session(tags, pitcher_id=PID, sid=None):
    return {"sessionId": sid or f"{DAY}__{pitcher_id}__T180000Z", "pitcherId": pitcher_id,
            "pitcherName": "Test-Pitcher, Alpha", "date": DAY, "tags": tags, "rev": len(tags),
            "taggedBy": "manager@example.edu", "savedAt": _iso(T0)}


def _tag(seq, at, zone=1, pitch_class="fastball", skipped=False):
    return {"seq": seq, "pitchClass": pitch_class, "at": _iso(at), "skipped": skipped,
            "zone": None if skipped else zone}


def _write_session(tags_root, session):
    d = tags_root / session["date"] / session["pitcherId"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{session['sessionId']}.json").write_text(json.dumps(session))


def test_reads_the_api_blob_layout_and_ignores_settings(tmp_path):
    tags = tmp_path / "tags"
    _write_session(tags, _session([_tag(1, T0)]))
    (tags / "settings").mkdir()
    (tags / "settings" / "command-plus.json").write_text(json.dumps({"tags": []}))
    bad = tags / DAY / PID / "broken.json"
    bad.write_text("{not json")
    sessions = cp.read_tag_sessions(str(tags))
    assert [s["sessionId"] for s in sessions] == [f"{DAY}__{PID}__T180000Z"]


def test_warmups_skipped_and_skips_dropped(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(10), "pen.csv")
    first = T0 + 3 * CADENCE  # three warm-ups before the manager started tagging
    tags = [_tag(1, first, zone=1), _tag(2, first + CADENCE, zone=4),
            _tag(3, first + 2 * CADENCE, skipped=True),
            _tag(4, first + 3 * CADENCE, zone=5, pitch_class="offspeed")]
    out = cp.build_command_pairs([_session(tags)], [str(tree)])
    s = out["sessions"][0]
    assert s["status"] == "joined" and s["pitcherMatch"] == "tagged-id"
    assert s["throws"] == "Right"
    assert s["batterSide"] == "Left"  # the hitter side the operator recorded for the pen
    assert [p["pitchNo"] for p in s["pairs"]] == [4, 5, 7]
    assert [p["zone"] for p in s["pairs"]] == [1, 4, 5]
    assert s["pairs"][2]["pitchClass"] == "offspeed"
    assert s["pairs"][0]["side"] == pytest.approx(0.3)
    assert s["pairs"][0]["height"] == pytest.approx(2.15)
    assert s["nSkipped"] == 1 and s["nTags"] == 4


def test_placeholder_pitcher_id_falls_back_to_the_time_anchor(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(6, pitcher_id="999"), "pen.csv")
    tags = [_tag(i + 1, T0 + i * CADENCE) for i in range(6)]
    s = cp.build_command_pairs([_session(tags)], [str(tree)])["sessions"][0]
    assert s["status"] == "joined"
    assert s["pitcherMatch"] == "anchor-only"
    assert len(s["pairs"]) == 6


def test_two_pens_one_day_use_only_the_anchored_file(tmp_path):
    tree = tmp_path / "tree"
    # An earlier pen under the same placeholder id, in its own file. PitchNo restarts,
    # so without the file restriction its rows would interleave with this pen's.
    _write_csv(tree, _pitches(8, pitcher_id="999", start=T0 - timedelta(minutes=30),
                              uid_prefix="early", side0=-1.0), "pen1.csv")
    _write_csv(tree, _pitches(8, pitcher_id="999", uid_prefix="late"), "pen2.csv")
    tags = [_tag(i + 1, T0 + i * CADENCE) for i in range(8)]
    s = cp.build_command_pairs([_session(tags)], [str(tree)])["sessions"][0]
    assert s["status"] == "joined"
    assert [p["side"] for p in s["pairs"]][:2] == [pytest.approx(0.0), pytest.approx(0.1)]


def test_files_without_timestamps_are_reported_not_guessed(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(6, with_utc=False), "pen.csv")
    s = cp.build_command_pairs([_session([_tag(1, T0)])], [str(tree)])["sessions"][0]
    assert s["status"] == "unjoined" and "UTCDateTime" in s["reason"]
    assert s["pairs"] == []


def test_refused_join_does_not_stop_other_sessions(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(4), "pen.csv")
    other = "1000000002"
    _write_csv(tree, _pitches(3, pitcher_id=other, start=T0 + timedelta(hours=1),
                              uid_prefix="b"), "pen_b.csv")
    ok = _session([_tag(i + 1, T0 + i * CADENCE) for i in range(4)])
    too_many = _session([_tag(i + 1, T0 + timedelta(hours=1) + i * CADENCE) for i in range(5)],
                        pitcher_id=other)
    out = cp.build_command_pairs([ok, too_many], [str(tree)])
    status = {s["pitcherId"]: s["status"] for s in out["sessions"]}
    assert status == {PID: "joined", other: "unjoined"}
    assert out["nJoined"] == 1 and out["nSessions"] == 2


def test_game_rows_are_never_candidates(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(6, level="D1"), "game.csv")
    s = cp.build_command_pairs([_session([_tag(1, T0)])], [str(tree)])["sessions"][0]
    assert s["status"] == "unjoined"


def test_payload_carries_no_names_no_outcomes_no_tagger(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(3), "pen.csv")
    out = cp.build_command_pairs([_session([_tag(i + 1, T0 + i * CADENCE) for i in range(3)])],
                                 [str(tree)])
    text = json.dumps(out)
    assert "Test-Pitcher" not in text
    assert "manager@example.edu" not in text
    assert "PitchCall" not in text and "Undefined\"" in text  # taggedType kept, call not
    cp.bj.assert_no_outcome_fields(out)


def test_empty_payload_shape():
    e = cp.empty_payload("no tag source configured")
    assert e["sessions"] == [] and e["nSessions"] == 0 and e["note"]


def test_batter_side_that_changes_mid_session_is_not_guessed(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(4, batter=lambda i: "Left" if i < 2 else "Right"), "pen.csv")
    s = cp.build_command_pairs([_session([_tag(i + 1, T0 + i * CADENCE) for i in range(4)])],
                               [str(tree)])["sessions"][0]
    assert s["status"] == "joined" and s["batterSide"] is None


def test_missing_batter_side_column_is_none(tmp_path):
    tree = tmp_path / "tree"
    _write_csv(tree, _pitches(3).drop(columns=["BatterSide"]), "pen.csv")
    s = cp.build_command_pairs([_session([_tag(i + 1, T0 + i * CADENCE) for i in range(3)])],
                               [str(tree)])["sessions"][0]
    assert s["batterSide"] is None and s["throws"] == "Right"
