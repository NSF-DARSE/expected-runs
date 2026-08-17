"""Contract tests for 15_bullpen_scores.py.

The real practice tree is licensed TrackMan (Level II), so these drive the
module on synthetic frames written to a tmp_path tree. Three things are being
protected: an empty session file must not raise, a pitcher with a handful of
pitches must still be gradeable and honestly counted, and no Adj Results or
Location+ value may ever reach the payload.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BULLPEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "15_bullpen_scores.py")


def _load():
    spec = importlib.util.spec_from_file_location("bullpen_scores", BULLPEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load()

COLUMNS = ["PitchUID", "Date", "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
           "BatterSide", "Balls", "Strikes", "TaggedPitchType", "AutoPitchType",
           "PitchCall", "RelSpeed", "SpinRate", "Extension", "HorzBreak",
           "InducedVertBreak", "EffectiveVelo", "RelHeight", "RelSide",
           "PlateLocHeight", "PlateLocSide", "GameID", "Level", "League"]


def _pitches(n, pitcher="Doe, John", pid=1, date="2026-07-22", auto="Four-Seam",
             game="G1", velo=90.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "PitchUID": [f"{pid}-{game}-{auto}-{i}" for i in range(n)],
        "Date": date, "Pitcher": pitcher, "PitcherId": pid, "PitcherThrows": "Right",
        "PitcherTeam": "DEL_BLU", "BatterSide": "Right", "Balls": 0, "Strikes": 0,
        "TaggedPitchType": "Undefined", "AutoPitchType": auto, "PitchCall": "Undefined",
        "RelSpeed": velo + rng.normal(0, 1, n), "SpinRate": 2200 + rng.normal(0, 50, n),
        "Extension": 6.2, "HorzBreak": 8.0, "InducedVertBreak": 15.0,
        "EffectiveVelo": velo - 1.5, "RelHeight": 5.9, "RelSide": 1.8,
        "PlateLocHeight": 2.5, "PlateLocSide": 0.0, "GameID": game,
        "Level": "TeamExclusive", "League": "NCAA",
    })[COLUMNS]


def _model():
    """A minimal artifact in the real shape: featureOrder + per-type coef,
    scaler moments, populationMeanZ and display moments."""
    feats = ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak", "EffectiveVelo",
             "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
             "velocity_differential", "is_lhp", "is_lhb"]
    k = len(feats)
    per_type = {
        "coef": list(np.linspace(-0.002, 0.002, k)),
        "scalerMean": [2200, 6.2, 8.0, 15.0, 88.0, 5.9, 1.8, 0.0, 0.0, 0.0, 0.4, 0.5],
        "scalerScale": [200, 0.5, 4.0, 4.0, 3.0, 0.3, 0.5, 5.0, 5.0, 5.0, 0.5, 0.5],
        "populationMeanZ": [0.0] * k,
        "displayMu": -0.002, "displaySd": 0.0076,
        "sampleFloor": 100, "nQualified": 3218,
    }
    return {"featureOrder": feats,
            "byPitchType": {t: dict(per_type) for t in ("FF", "Slider", "ChangeUp")}}


def _tree(tmp_path, frames_by_relpath):
    for rel, df in frames_by_relpath.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if df is None:
            path.write_text("")           # truly empty file
        elif df.empty:
            df.to_csv(path, index=False)  # header only, zero rows
        else:
            df.to_csv(path, index=False)
    return str(tmp_path)


# ------------------------------------------------------------- empty files ---

def test_empty_session_file_does_not_raise(tmp_path):
    root = _tree(tmp_path, {
        "2026/07/22/CSV/empty.csv": None,
        "2026/07/23/CSV/headers_only.csv": pd.DataFrame(columns=COLUMNS),
        "2026/07/24/CSV/real.csv": _pitches(40),
    })
    raw = bp.read_practice_tree(root)
    assert len(raw) == 40, "the two empty files should be skipped, not fatal"


def test_tree_with_only_empty_files_returns_empty_frame(tmp_path):
    root = _tree(tmp_path, {"2026/07/22/CSV/empty.csv": None})
    assert bp.read_practice_tree(root).empty


def test_main_exits_nonzero_on_an_all_empty_tree(tmp_path):
    root = _tree(tmp_path, {"2026/07/22/CSV/empty.csv": None})
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(_model()))
    rc = bp.main(["--practice-tree", root, "--model", str(model_path),
                  "--out", str(tmp_path / "out.json")])
    assert rc == 1


# --------------------------------------------------------- few-pitch arms ----

def test_pitcher_with_very_few_pitches_is_still_graded_and_counted(tmp_path):
    root = _tree(tmp_path, {
        "2026/07/22/CSV/a.csv": _pitches(300, pitcher="Big, Sample", pid=1, game="G1"),
        "2026/08/10/CSV/b.csv": _pitches(3, pitcher="Tiny, Sample", pid=2, game="G2",
                                         date="2026-08-10", velo=84.0, seed=7),
    })
    scored = bp.score_pitches(bp.prepare(bp.read_practice_tree(root)), _model())
    recs = {r["name"]: r for r in bp.build_records(scored)}
    tiny = recs["Tiny, Sample"]
    assert tiny["nPitches"] == 3 and tiny["nGraded"] == 3
    assert tiny["stuff"] is not None and np.isfinite(tiny["stuff"])
    assert len(tiny["sessions"]) == 1 and tiny["sessions"][0]["nPitches"] == 3
    assert len(tiny["pitches"]) == 3


def test_unclassified_pitches_are_ungraded_not_folded_into_another_type(tmp_path):
    root = _tree(tmp_path, {
        "2026/07/22/CSV/a.csv": pd.concat([
            _pitches(10, auto="Four-Seam", game="G1"),
            _pitches(5, auto="Other", game="G1", seed=3),
        ], ignore_index=True),
    })
    scored = bp.score_pitches(bp.prepare(bp.read_practice_tree(root)), _model())
    assert scored["stuff"].notna().sum() == 10
    rec = bp.build_records(scored)[0]
    assert rec["nUngraded"] == 5
    assert [r["type"] for r in rec["byType"]] == ["FF"]


def test_scale_is_the_artifact_scale_not_refit_on_the_pen(tmp_path):
    """Two pens differing only in population must not be re-centred on each
    other: the same pitch scores the same Stuff+ in both."""
    model = _model()
    solo = _tree(tmp_path / "solo", {"2026/07/22/CSV/a.csv": _pitches(50, pid=1)})
    mixed_root = tmp_path / "mixed"
    mixed = _tree(mixed_root, {
        "2026/07/22/CSV/a.csv": _pitches(50, pid=1),
        "2026/07/22/CSV/b.csv": _pitches(50, pitcher="Other, Guy", pid=2,
                                         game="G9", velo=99.0, seed=11),
    })
    a = bp.build_records(bp.score_pitches(bp.prepare(bp.read_practice_tree(solo)), model))
    b = bp.build_records(bp.score_pitches(bp.prepare(bp.read_practice_tree(mixed)), model))
    b_first = [r for r in b if r["pitcherId"] == 1][0]
    assert a[0]["stuff"] == b_first["stuff"]


# ------------------------------------ no results / Location+, ever -----------

def test_payload_never_contains_adj_results_or_location_plus(tmp_path):
    root = _tree(tmp_path, {
        "2026/07/22/CSV/a.csv": pd.concat([
            _pitches(60, auto="Four-Seam", game="G1"),
            _pitches(30, auto="Slider", game="G1", velo=82.0, seed=2),
        ], ignore_index=True),
    })
    payload = bp.build_payload(
        bp.score_pitches(bp.prepare(bp.read_practice_tree(root)), _model()),
        _model(), "model_artifacts.json")

    seen = []

    def walk(node):
        if isinstance(node, dict):
            seen.extend(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(payload)
    lowered = {k.lower() for k in seen}
    for forbidden in bp.FORBIDDEN_KEYS:
        assert forbidden.lower() not in lowered, f"{forbidden} leaked into the payload"
    # And the guard itself has to be live, not a no-op.
    assert bp.FORBIDDEN_KEYS


def test_guard_rejects_a_smuggled_outcome_field():
    with pytest.raises(bp.BullpenOutcomeError):
        bp.assert_no_outcome_fields({"pitchers": [{"name": "X", "adjRes": 101.4}]})
    with pytest.raises(bp.BullpenOutcomeError):
        bp.assert_no_outcome_fields({"pitchers": [{"byType": [{"type": "FF", "loc": 0.0}]}]})
    # Case-insensitive: the bundle's camelCase spelling must be caught too.
    with pytest.raises(bp.BullpenOutcomeError):
        bp.assert_no_outcome_fields({"locationPlus": None})


def test_every_record_is_marked_as_bullpen(tmp_path):
    root = _tree(tmp_path, {"2026/07/22/CSV/a.csv": _pitches(20)})
    payload = bp.build_payload(
        bp.score_pitches(bp.prepare(bp.read_practice_tree(root)), _model()),
        _model(), "model_artifacts.json")
    assert payload["context"] == bp.CONTEXT
    for rec in payload["pitchers"]:
        assert rec["context"] == bp.CONTEXT
        for s in rec["sessions"]:
            assert s["context"] == bp.CONTEXT
        for p in rec["pitches"]:
            assert p["context"] == bp.CONTEXT
    assert payload["unavailableConstructs"]


def test_primary_fastball_flag_reports_a_non_fastball_baseline(tmp_path):
    """A pen with no fastball at all: the differentials get computed off a
    breaking ball, and the session has to say so."""
    root = _tree(tmp_path, {
        "2026/07/22/CSV/a.csv": pd.concat([
            _pitches(20, auto="Slider", game="G1", velo=84.0),
            _pitches(20, auto="Curveball", game="G1", velo=78.0, seed=4),
        ], ignore_index=True),
    })
    scored = bp.score_pitches(bp.prepare(bp.read_practice_tree(root)), _model())
    rec = bp.build_records(scored)[0]
    assert rec["sessions"][0]["primaryTypeIsFastball"] is False
