"""End-to-end check that the build reads the corrected copy of a revised game.

Guards the actual failure this fix addresses: the old build read every day folder,
so a game re-pulled after a revision was loaded two to five times AND the
first-kept copy was the pre-correction one.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from target_and_calculated_pipeline import generate_target_for_years_df


GAME_HEADER = [
    "PitchNo", "Date", "PAofInning", "PitchofPA", "Pitcher", "PitcherId",
    "PitcherThrows", "PitcherTeam", "Batter", "BatterId", "BatterSide",
    "BatterTeam", "Inning", "Top/Bottom", "Outs", "Balls", "Strikes",
    "TaggedPitchType", "AutoPitchType", "PitchCall", "TaggedHitType",
    "PlayResult", "OutsOnPlay", "RunsScored", "RelSpeed", "SpinRate",
    "Extension", "HorzBreak", "InducedVertBreak", "SpinAxis", "EffectiveVelo",
    "RelHeight", "RelSide", "VertBreak", "PlateLocHeight", "PlateLocSide",
    "GameID", "PitchUID", "Level", "League",
]


def game_frame(batter_side, play_result):
    """Three pitches in one half-inning, ending on a ball in play."""
    rows = []
    for i in range(3):
        rows.append({
            "PitchNo": i + 1, "Date": "2026-01-23", "PAofInning": 1,
            "PitchofPA": i + 1, "Pitcher": "Arm, Test", "PitcherId": 501,
            "PitcherThrows": "Right", "PitcherTeam": "TST_AAA",
            "Batter": "Bat, Test", "BatterId": 901, "BatterSide": batter_side,
            "BatterTeam": "TST_BBB", "Inning": 1, "Top/Bottom": "Top",
            "Outs": 0, "Balls": 0, "Strikes": 0,
            "TaggedPitchType": "Fastball", "AutoPitchType": "Four-Seam",
            "PitchCall": "InPlay" if i == 2 else "StrikeCalled",
            "TaggedHitType": "LineDrive" if i == 2 else None,
            "PlayResult": play_result if i == 2 else "Undefined",
            "OutsOnPlay": 0, "RunsScored": 0,
            "RelSpeed": 92.0 + i, "SpinRate": 2200, "Extension": 6.2,
            "HorzBreak": 8.0, "InducedVertBreak": 16.0, "SpinAxis": 210,
            "EffectiveVelo": 92.5, "RelHeight": 5.9, "RelSide": 1.8,
            "VertBreak": -12.0, "PlateLocHeight": 2.8, "PlateLocSide": 0.1,
            "GameID": "G-2026-01-23-TST", "PitchUID": f"uid-{i + 1}",
            "Level": "D1", "League": "TestLeague",
        })
    return pd.DataFrame(rows)[GAME_HEADER]


def write_copy(root, month, day, frame, name="20260123-TestPark-1.csv"):
    folder = os.path.join(root, "2026", month, day, "CSV")
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, name), index=False)


def write_summary(tmp_path):
    path = os.path.join(str(tmp_path), "GameState_Summary.csv")
    states = []
    for outs in range(3):
        for balls in range(4):
            for strikes in range(3):
                for bases in ["000", "100", "010", "001", "110", "101", "011", "111"]:
                    states.append({
                        "GameState": f"{bases}-O{outs}-B{balls}-S{strikes}",
                        "ExpectedRuns": 0.5,
                    })
    pd.DataFrame(states).to_csv(path, index=False)
    return path


def test_revised_copy_wins_and_no_pitch_is_double_counted(tmp_path):
    root = os.path.join(str(tmp_path), "source")
    summary_path = write_summary(tmp_path)

    # Same game pulled three times. The last pull carries the corrections.
    stale = game_frame(batter_side="Right", play_result="Double")
    write_copy(root, "01", "24", stale)
    write_copy(root, "01", "27", stale)
    corrected = game_frame(batter_side="Left", play_result="Single")
    write_copy(root, "03", "11", corrected)

    built = generate_target_for_years_df(root, ["2026"], summary_path)

    assert built is not None
    assert len(built) == 3, "each pitch should appear exactly once"
    assert built["PitchUID"].is_unique
    assert set(built["BatterSide"]) == {"Left"}, "corrected handedness must win"
    assert "Single" in set(built["PlayResult"]), "corrected PlayResult must win"
    assert "Double" not in set(built["PlayResult"])


def test_unrevised_build_is_unchanged(tmp_path):
    """A tree with no duplicate pulls must produce exactly what it always did."""
    root = os.path.join(str(tmp_path), "source")
    summary_path = write_summary(tmp_path)
    write_copy(root, "01", "24", game_frame("Right", "Single"))

    built = generate_target_for_years_df(root, ["2026"], summary_path)

    assert built is not None
    assert len(built) == 3
    assert built["PitchUID"].tolist() == ["uid-1", "uid-2", "uid-3"]


def test_suspended_game_keeps_the_reissued_file(tmp_path):
    """The real 20260221-RiddlePaceField-1 case.

    A suspended game is first exported as a standalone continuation, then later
    re-issued as the whole game under its original date with PitchNo renumbered.
    The same PitchUIDs appear under two different filenames, so file-level
    resolution cannot see it. The later pull must win.
    """
    root = os.path.join(str(tmp_path), "source")
    summary_path = write_summary(tmp_path)

    fragment = game_frame("Right", "Single")
    fragment["GameID"] = "20260222-Park-1"
    fragment["PitchNo"] = [56, 57, 58]
    write_copy(root, "02", "23", fragment, name="20260222-Park-1.csv")

    reissued = game_frame("Right", "Single")
    reissued["GameID"] = "20260221-Park-1"
    reissued["PitchNo"] = [125, 126, 127]
    write_copy(root, "03", "23", reissued, name="20260221-Park-1.csv")

    built = generate_target_for_years_df(root, ["2026"], summary_path)

    assert len(built) == 3, "the fragment's pitches must not be counted twice"
    assert built["PitchUID"].is_unique
    assert set(built["GameID"]) == {"20260221-Park-1"}, "re-issued GameID must win"
    assert built["PitchNo"].tolist() == [125, 126, 127], "continuous PitchNo must win"


def test_two_distinct_games_both_survive(tmp_path):
    root = os.path.join(str(tmp_path), "source")
    summary_path = write_summary(tmp_path)

    write_copy(root, "01", "24", game_frame("Right", "Single"))
    second = game_frame("Left", "Double")
    second["PitchUID"] = ["uid-4", "uid-5", "uid-6"]
    second["GameID"] = "G-2026-01-23-TST-2"
    write_copy(root, "01", "24", second, name="20260123-TestPark-2.csv")

    built = generate_target_for_years_df(root, ["2026"], summary_path)

    assert len(built) == 6
    assert built["PitchUID"].is_unique
