from collections import defaultdict
from datetime import datetime
import os
import pandas as pd

from Helpers import (
    add_runner_states,
    add_game_state,
    add_runs_remaining,
    calculate_zero_run_probabilities,
    count_superseded_copies,
    resolve_latest_game_files,
)


def build_gamestate_summary_all_years(data_root, save_path):
    """
    Builds a run expectancy summary across ALL available years of data.

    For each GameState, this function calculates:
        - Count: total occurrences
        - TotalRunsRemaining: total future runs scored after that state
        - ExpectedRuns: average runs remaining from that state
        - ZeroRunsCount: number of times RunsRemaining == 0
        - ZeroRunProbability: probability of zero runs from that state

    It loops through:
        Year → Month → Day → CSV files
    """
    summary = defaultdict(lambda: {"Count": 0, "TotalRunsRemaining": 0, "ZeroRunsCount": 0})

    # A revised game is re-pulled into a later day folder, so walking every folder
    # counted the same half-innings two to five times and weighted the run-expectancy
    # table toward whichever games happened to be revised. Resolve to the newest copy
    # per game first. See Helpers.resolve_latest_game_files.
    file_paths = resolve_latest_game_files(data_root)
    superseded = count_superseded_copies(data_root)
    print(
        f"Resolved {len(file_paths)} games under {data_root}; "
        f"skipped {superseded} superseded copies."
    )

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)

            required = {
                "Inning", "Top/Bottom", "Outs", "Balls",
                "Strikes", "RunsScored", "PlayResult"
            }
            if not required.issubset(df.columns):
                continue

            df = df[df["Inning"] < 9]
            if df.empty:
                continue

            df = add_runner_states(df)
            df = add_game_state(df)
            df = add_runs_remaining(df)

            df = df[
                (df["Outs"] <= 2) &
                (df["Balls"] <= 3) &
                (df["Strikes"] <= 2)
            ]

            agg = df.groupby("GameState")["RunsRemaining"].agg(["count", "sum"])
            for state, row in agg.iterrows():
                summary[state]["Count"] += row["count"]
                summary[state]["TotalRunsRemaining"] += row["sum"]

            zero_stats = calculate_zero_run_probabilities(df)
            for state, val in zero_stats.items():
                summary[state]["ZeroRunsCount"] += val["ZeroRunsCount"]

        except Exception:
            continue

    combined_df = pd.DataFrame(
        [
            {
                "GameState": s,
                "Count": d["Count"],
                "TotalRunsRemaining": d["TotalRunsRemaining"],
                "ExpectedRuns": d["TotalRunsRemaining"] / d["Count"] if d["Count"] else 0,
                "ZeroRunsCount": d["ZeroRunsCount"],
                "ZeroRunProbability": d["ZeroRunsCount"] / d["Count"] if d["Count"] else 0,
            }
            for s, d in summary.items()
        ]
    )

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"GameState_Summary_ALL_prob_{timestamp}.csv"
    out_path = os.path.join(save_path, filename)

    combined_df.to_csv(out_path, index=False)
    print(f"\nAll-year summary saved: {out_path}")

    return combined_df


if __name__ == "__main__":
    data_root = "/Users/suma/Downloads/Baseball_Project/v3"
    save_path = "/Users/suma/Downloads/Baseball_Project/CSV_files"

    summary_df = build_gamestate_summary_all_years(data_root, save_path)
    print(summary_df.head())