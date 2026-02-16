import os
import calendar
from datetime import datetime

import pandas as pd
from Helpers import add_runner_states, add_game_state, add_runs_remaining


# Final schema (same list you already use)
REQUIRED_COLS = [
    "PitchNo", "Date", "PAofInning", "PitchofPA", "Pitcher", "PitcherId",
    "PitcherThrows", "PitcherTeam", "Batter", "BatterSide", "BatterTeam",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "TaggedPitchType",
    "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", "OutsOnPlay",
    "RunsScored", "RunnerOn1B", "RunnerOn2B", "RunnerOn3B", "GameState",
    "RunsRemaining", "ExpectedRuns", "Target", "RelSpeed", "SpinRate",
    "Extension", "HorzBreak", "InducedVertBreak", "SpinAxis", "EffectiveVelo",
    "RelHeight", "RelSide", "FastestPitchType", "MaxRelSpeed",
    "Avg_InducedVertBreak_FastestType", "Avg_HorzBreak_FastestType",
    "Avg_RelSpeed_FastestType", "vertbreakdiff", "horzbreakdiff",
    "velocity_differential", "VertBreak", "PlateLocHeight", "PlateLocSide",
    "Level", "League"
]


def load_gamestate_to_er(summary_path):
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"GameState summary file not found at: {summary_path}")
    df = pd.read_csv(summary_path)
    return dict(zip(df["GameState"], df["ExpectedRuns"]))


def generate_target_for_month(base_path, year, month, summary_path):
    gamestate_to_er = load_gamestate_to_er(summary_path)
    all_dfs = []

    total_days = calendar.monthrange(int(year), int(month))[1]

    for day in range(1, total_days + 1):
        day_str = str(day).zfill(2)
        folder_path = os.path.join(base_path, year, month, day_str, "CSV")

        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if not file.endswith(".csv") or "unverified" in file.lower():
                continue

            file_path = os.path.join(folder_path, file)

            try:
                df = pd.read_csv(file_path)

                if df.empty or not {
                    'Inning', 'Top/Bottom', 'Outs', 'Balls',
                    'Strikes', 'RunsScored', 'PlayResult'
                }.issubset(df.columns):
                    continue

                df = df[df['Inning'] < 9]

                df = add_runner_states(df)
                df = add_game_state(df)
                df = add_runs_remaining(df)

                df = df[(df['Outs'] <= 2) & (df['Balls'] <= 3) & (df['Strikes'] <= 2)]

                df["ExpectedRuns"] = df["GameState"].map(gamestate_to_er).round(4)
                df["ExpectedRuns_Next"] = df["ExpectedRuns"].shift(-1)
                df["Top/Bottom_Next"] = df["Top/Bottom"].shift(-1)

                df["Target"] = df.apply(
                    lambda r: round(0 - r["ExpectedRuns"], 4)
                    if r["Top/Bottom"] != r["Top/Bottom_Next"]
                    else round(r["ExpectedRuns_Next"] - r["ExpectedRuns"], 4),
                    axis=1
                )

                df.drop(columns=["ExpectedRuns_Next", "Top/Bottom_Next"], inplace=True)

                insert_cols = [
                    'RunnerOn1B', 'RunnerOn2B', 'RunnerOn3B',
                    'GameState', 'RunsRemaining', 'ExpectedRuns', 'Target'
                ]

                idx = df.columns.get_loc('RunsScored') + 1
                for col in reversed(insert_cols):
                    if col in df.columns:
                        df.insert(idx, col, df.pop(col))

                df = df.loc[:, [c for c in REQUIRED_COLS if c in df.columns]]
                all_dfs.append(df)

            except Exception as e:
                print(f"Error in {file_path}: {e}")

    if not all_dfs:
        return None

    return pd.concat(all_dfs, ignore_index=True)


def generate_target_for_years_df(base_path, years, summary_path):
    all_year_dfs = []

    for year in years:
        print(f"Processing year {year}")
        for month in [f"{m:02d}" for m in range(1, 13)]:
            df_m = generate_target_for_month(base_path, year, month, summary_path)
            if df_m is not None and not df_m.empty:
                all_year_dfs.append(df_m)

    if not all_year_dfs:
        print("No valid files processed.")
        return None

    return pd.concat(all_year_dfs, ignore_index=True)


def add_calculated_features(df):
    """
    Same calculated-feature logic, but works directly on the dataframe
    (no intermediate CSV required).
    """

    required_for_calc = ["PitcherId", "TaggedPitchType", "RelSpeed", "InducedVertBreak", "HorzBreak"]
    df_clean = df.dropna(subset=required_for_calc).copy()

    fastest_pitch = (
        df_clean.loc[df_clean.groupby("PitcherId")["RelSpeed"].idxmax(), ["PitcherId", "TaggedPitchType"]]
        .rename(columns={"TaggedPitchType": "FastestPitchType"})
    )

    df_merged = df_clean.merge(fastest_pitch, on="PitcherId", how="left")

    max_relspeed = (
        df_clean.groupby("PitcherId")["RelSpeed"]
        .max()
        .reset_index()
        .rename(columns={"RelSpeed": "MaxRelSpeed"})
    )
    df_merged = df_merged.merge(max_relspeed, on="PitcherId", how="left")

    fastest_type_stats = (
        df_merged[df_merged["TaggedPitchType"] == df_merged["FastestPitchType"]]
        .groupby(["PitcherId", "FastestPitchType"])
        .agg({
            "InducedVertBreak": "mean",
            "HorzBreak": "mean",
            "RelSpeed": "mean"
        })
        .reset_index()
        .rename(columns={
            "InducedVertBreak": "Avg_InducedVertBreak_FastestType",
            "HorzBreak": "Avg_HorzBreak_FastestType",
            "RelSpeed": "Avg_RelSpeed_FastestType"
        })
    )

    df_merged = df_merged.merge(
        fastest_type_stats,
        on=["PitcherId", "FastestPitchType"],
        how="left"
    )

    df_merged["vertbreakdiff"] = df_merged["InducedVertBreak"] - df_merged["Avg_InducedVertBreak_FastestType"]
    df_merged["horzbreakdiff"] = df_merged["HorzBreak"] - df_merged["Avg_HorzBreak_FastestType"]
    df_merged["velocity_differential"] = df_merged["RelSpeed"] - df_merged["Avg_RelSpeed_FastestType"]

    cols_to_keep = [c for c in REQUIRED_COLS if c in df_merged.columns]
    df_final = df_merged.loc[:, cols_to_keep]

    return df_final


def build_final_dataset(base_path, years, summary_path, out_dir, save=True):
    """
    One callable function:
      1) Build target-integrated dataframe for years
      2) Add calculated features
      3) Save ONE final CSV (optional)
      4) Return final dataframe
    """

    target_integrated_df = generate_target_for_years_df(base_path, years, summary_path)
    if target_integrated_df is None or target_integrated_df.empty:
        return None

    final_df = add_calculated_features(target_integrated_df)

    if save:
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M")
        years_tag = "_".join(years)
        out_path = os.path.join(out_dir, f"Final_Target_Calc_{timestamp}.csv")
        final_df.to_csv(out_path, index=False)
        print(f"Final dataset saved: {out_path}")

    return final_df


# ---------------- RUN ----------------
final_df = build_final_dataset(
    base_path="/Users/suma/Downloads/Baseball_Project/v3",
    years=["2024", "2025"],
    summary_path="/Users/suma/Downloads/Baseball_Project/CSV_files/game_state_summary_file/GameState_Summary.csv",
    out_dir="/Users/suma/Downloads/Baseball_Project/CSV_files",
    save=True
)

final_df