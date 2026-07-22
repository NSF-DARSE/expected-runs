"""Golden diff: API-flattened game vs the pipeline-built final dataset.

Acceptance test for flatten.py. Pulls one game from the TrackMan Data API,
flattens it, runs it through the SAME pipeline logic that built the final
dataset from the raw Drive CSVs (Helpers.add_runner_states / add_game_state /
add_runs_remaining + the ExpectedRuns/Target computation), and diffs the
result row-for-row against that game's rows in Final_Target_Calc_*.csv.

Because the downstream code is the same deterministic function, everything
through the Target stage should match EXACTLY. Per-pitcher calculated
features (FastestPitchType, MaxRelSpeed, Avg_*_FastestType, *diff,
velocity_differential) are computed over the full multi-season dataset, so
they are out of scope for a one-game diff and are skipped.

Two documented departures from a byte-for-byte diff:
  - ExpectedRuns map: the GameState_Summary.csv on the Drive share is not the
    vintage used to build the final dataset (every ER value differs), so the
    GameState -> ExpectedRuns map is read off the final dataset itself
    (verified single-valued per state first). This still tests everything the
    flattener owns: GameState construction, row order, the shift(-1) Target
    logic, and rounding.
  - RunsRemaining: the final dataset was built with an INCLUSIVE future-runs
    sum (current pitch's runs counted) -- a deliberate July 2026 correction
    agreed during the target-definition fix, and the right convention for run
    expectancy (Target = RS + ER_next - ER assumes it). Repo Helpers.py still
    carries the older exclusive sum (runs[i+1:]); it should be updated to
    match the dataset. Until then, mismatches are accepted only if they
    exactly match that signature (csv == repo_value + RunsScored, on scoring
    pitches only) and are reported as KNOWN-DIFF. RunsRemaining feeds only
    the ER-table build, not Target.

Usage:
    python trackman_api/golden_diff.py --data-dir "<folder with Final_Target_Calc_*.csv and GameState_Summary.csv>"
    python trackman_api/golden_diff.py --data-dir ... --game-id 20250516-BobHannahStadium-1

Output is aggregate only (row counts, per-column match verdicts, max numeric
deltas) -- no licensed pitch-level values are printed.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python_files"))

from auth import get_token
from config import load_config
from flatten import flatten_game
from smoke_test import discover_game_sessions, get_json
from Helpers import add_runner_states, add_game_state, add_runs_remaining

# Columns compared exactly as strings (identity/categorical/integer-like).
EXACT_COLS = [
    "PitchNo", "Date", "PAofInning", "PitchofPA", "Pitcher", "PitcherId",
    "PitcherThrows", "PitcherTeam", "Batter", "BatterSide", "BatterTeam",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "TaggedPitchType",
    "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", "OutsOnPlay",
    "RunsScored", "RunnerOn1B", "RunnerOn2B", "RunnerOn3B", "GameState",
    "GameID", "PitchUID", "Level", "League",
]

# Float columns: same underlying measurements, but the raw CSVs and the API
# serialize them at slightly different float precision (observed max relative
# delta ~1e-7), so these compare with a relative tolerance.
FLOAT_COLS = [
    "ExpectedRuns", "Target", "RelSpeed", "SpinRate", "Extension",
    "HorzBreak", "InducedVertBreak", "SpinAxis", "EffectiveVelo",
    "RelHeight", "RelSide", "VertBreak", "PlateLocHeight", "PlateLocSide",
    "ExitSpeed", "Angle", "Direction", "Distance", "HangTime",
]

_ATOL = 1e-6
_RTOL = 1e-6


def run_pipeline_on_game(raw: pd.DataFrame, gamestate_to_er: dict) -> pd.DataFrame:
    """Apply the per-game pipeline stages to one flattened game.

    Mirrors generate_target_for_month() in
    python_files/target_and_calculated_pipeline.py (which cannot be imported
    because it executes its full build at module import), plus the row-drop
    rules of add_calculated_features() so the surviving rows match the final
    dataset. Runner-state / game-state / runs-remaining logic is imported,
    not copied.
    """
    df = raw[raw["Inning"] < 9].copy()

    df = add_runner_states(df)
    df = add_game_state(df)
    df = add_runs_remaining(df)

    df = df[(df["Outs"] <= 2) & (df["Balls"] <= 3) & (df["Strikes"] <= 2)]

    df["ExpectedRuns"] = df["GameState"].map(gamestate_to_er).round(4)
    df["ExpectedRuns_Next"] = df["ExpectedRuns"].shift(-1)
    df["Top/Bottom_Next"] = df["Top/Bottom"].shift(-1)

    df["RunsScored"] = df["RunsScored"].fillna(0)
    df["Target"] = df.apply(
        lambda r: round(r["RunsScored"] - r["ExpectedRuns"], 4)
        if r["Top/Bottom"] != r["Top/Bottom_Next"]
        else round(r["RunsScored"] + r["ExpectedRuns_Next"] - r["ExpectedRuns"], 4),
        axis=1,
    )
    df.drop(columns=["ExpectedRuns_Next", "Top/Bottom_Next"], inplace=True)

    # Row-drop rules from add_calculated_features(): untracked pitches and
    # excluded reference pitch types never reach the final dataset.
    df["TaggedPitchType"] = df["TaggedPitchType"].replace({"Changeup": "ChangeUp"})
    df = df.dropna(subset=["PitcherId", "TaggedPitchType", "RelSpeed",
                           "InducedVertBreak", "HorzBreak"])
    df = df[~df["TaggedPitchType"].isin(
        {"Undefined", "Other", "Knuckleball", "OneSeamFastBall"})]
    return df.reset_index(drop=True)


def fetch_game(game_id: str):
    """Discover the session whose gameID matches, pull plays + balls."""
    cfg = load_config()
    token = get_token(cfg)
    day = datetime.strptime(game_id.split("-", 1)[0], "%Y%m%d").replace(tzinfo=timezone.utc)
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    sessions = discover_game_sessions(
        cfg, token,
        (day - timedelta(days=1)).strftime(fmt),
        (day + timedelta(days=2)).strftime(fmt),
    )
    matches = [s for s in sessions if s.get("gameID") == game_id]
    if not matches:
        raise SystemExit(f"No API session with gameID {game_id} "
                         f"({len(sessions)} sessions in window).")
    session = matches[0]
    sid = session["sessionId"]
    plays = get_json(cfg, token, f"data/game/plays/{sid}")
    balls = get_json(cfg, token, f"data/game/balls/{sid}")
    return session, plays, balls


def compare(api_df: pd.DataFrame, csv_df: pd.DataFrame) -> bool:
    """Column-by-column diff; prints a verdict table, returns overall pass."""
    print(f"rows: api={len(api_df)} csv={len(csv_df)}", end="")
    if len(api_df) != len(csv_df):
        print("  -> ROW COUNT MISMATCH")
        api_only = set(api_df["PitchUID"]) - set(csv_df["PitchUID"])
        csv_only = set(csv_df["PitchUID"]) - set(api_df["PitchUID"])
        print(f"  PitchUIDs only in api: {len(api_only)}, only in csv: {len(csv_only)}")
        return False
    print()

    order_ok = list(api_df["PitchUID"]) == list(csv_df["PitchUID"])
    print(f"row order (PitchUID sequence): {'MATCH' if order_ok else 'MISMATCH'}")
    if not order_ok:
        return False

    all_ok = True
    print(f"\n{'column':<22}{'verdict':<12}{'mismatches':<12}{'max |delta|'}")
    for col in EXACT_COLS:
        a = api_df[col].astype(str).str.strip()
        c = csv_df[col].astype(str).str.strip()
        bad = int((a != c).sum())
        ok = bad == 0
        all_ok &= ok
        print(f"{col:<22}{'ok' if ok else 'DIFF':<12}{bad:<12}")
    for col in FLOAT_COLS:
        a = pd.to_numeric(api_df[col], errors="coerce")
        c = pd.to_numeric(csv_df[col], errors="coerce")
        nan_match = a.isna() == c.isna()
        close = np.isclose(a, c, rtol=_RTOL, atol=_ATOL) | (a.isna() & c.isna())
        bad = int((~nan_match).sum() + (~close & nan_match).sum())
        ok = bad == 0
        all_ok &= ok
        both = a.notna() & c.notna()
        mx = f"{(a[both] - c[both]).abs().max():.2e}" if both.any() else "n/a"
        print(f"{col:<22}{'ok' if ok else 'DIFF':<12}{bad:<12}{mx}")

    # RunsRemaining: pass only as the documented provenance diff (see module
    # docstring) -- csv == repo value + that pitch's own RunsScored.
    a = pd.to_numeric(api_df["RunsRemaining"], errors="coerce")
    c = pd.to_numeric(csv_df["RunsRemaining"], errors="coerce")
    rs = pd.to_numeric(csv_df["RunsScored"], errors="coerce").fillna(0)
    diff_rows = a != c
    signature = (c == a + rs) & (rs > 0)
    if not diff_rows.any():
        print(f"{'RunsRemaining':<22}{'ok':<12}0")
    elif bool((~signature[diff_rows]).sum() == 0):
        print(f"{'RunsRemaining':<22}{'KNOWN-DIFF':<12}{int(diff_rows.sum()):<12}"
              "(dataset built with inclusive future-runs sum)")
    else:
        print(f"{'RunsRemaining':<22}{'DIFF':<12}{int(diff_rows.sum()):<12}"
              "(does NOT match the known inclusive-sum signature)")
        all_ok = False
    return all_ok


def _to_exact_str(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize integer-like exact columns so 1.0 (CSV float) == 1 (API int)."""
    out = df.copy()
    for col in ["PitchNo", "PAofInning", "PitchofPA", "PitcherId", "Inning",
                "Outs", "Balls", "Strikes", "OutsOnPlay", "RunsScored",
                "RunnerOn1B", "RunnerOn2B", "RunnerOn3B", "RunsRemaining"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Golden diff: API flatten vs final dataset")
    p.add_argument("--data-dir", required=True,
                   help="Folder containing Final_Target_Calc_*.csv and GameState_Summary.csv")
    p.add_argument("--game-id", default="20250516-BobHannahStadium-1")
    args = p.parse_args()

    finals = sorted(glob.glob(os.path.join(args.data_dir, "Final_Target_Calc_*.csv")))
    if not finals:
        raise SystemExit(f"No Final_Target_Calc_*.csv in {args.data_dir}")
    final_path = finals[-1]

    print(f"final dataset: {os.path.basename(final_path)}")
    print(f"game:          {args.game_id}")

    # Implied GameState -> ExpectedRuns map, read off the final dataset itself
    # (the summary CSV on the share is a different vintage; see docstring).
    full = pd.read_csv(final_path, usecols=["GameID", "GameState", "ExpectedRuns"])
    per_state = full.groupby("GameState")["ExpectedRuns"].nunique()
    ambiguous = per_state[per_state > 1]
    if len(ambiguous):
        raise SystemExit(f"Implied ER map is not single-valued for "
                         f"{len(ambiguous)} GameStates; cannot proceed.")
    gamestate_to_er = full.groupby("GameState")["ExpectedRuns"].first().to_dict()
    print(f"implied ER map: {len(gamestate_to_er)} states, single-valued")

    print("\nPulling game from the API...")
    session, plays, balls = fetch_game(args.game_id)
    raw = flatten_game(session, plays, balls)
    print(f"flattened: {len(raw)} pitches")

    api_df = run_pipeline_on_game(raw, gamestate_to_er)

    csv_df = pd.read_csv(final_path)
    csv_df = csv_df[csv_df["GameID"] == args.game_id].reset_index(drop=True)

    ok = compare(_to_exact_str(api_df), _to_exact_str(csv_df))
    print(f"\n{'GOLDEN DIFF PASS' if ok else 'GOLDEN DIFF FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
