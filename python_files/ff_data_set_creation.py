"""
Create the four-seam fastball dataset used for modeling.

This script is the Python-file version of the ff_data_set_creation.ipynb
workflow. It creates df_ff.csv and the team-specific df_del_blu_ff.csv using
the same logic as the notebook.
"""

from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/Final Data Set/Final_Target_Calc.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_ff.csv"
)
DEFAULT_TEAM_OUTPUT_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_del_blu_ff.csv"
)
DEFAULT_TEAM_CODE = "DEL_BLU"


FEATURE_COLUMNS = [
    "SpinRate",
    "Extension",
    "HorzBreak",
    "InducedVertBreak",
    "EffectiveVelo",
    "RelHeight",
    "RelSide",
    "Is_Left_Handed_Pitcher",
    "Is_Left_Handed_Batter",
    "vertbreakdiff",
    "horzbreakdiff",
    "velocity_differential",
]


def assign_bucket(row):
    """Map the notebook's TaggedPitchType values into broader pitch buckets."""
    pitch = row["TaggedPitchType"]

    if pitch == "Slider" and row["InducedVertBreak"] <= -3:
        return "Slider"
    elif pitch == "Slider" and row["InducedVertBreak"] > -3:
        return "Gyro/Sweeper"
    elif pitch == "Sweeper":
        return "Gyro/Sweeper"
    elif pitch in ["Fastball", "FourSeamFastBall"]:
        return "FourSeamFastball"
    elif pitch in ["TwoSeamFastBall", "Sinker"]:
        return "Sinker"
    elif pitch == "Cutter":
        return "Cutter"
    elif pitch == "ChangeUp":
        return "ChangeUp"
    elif pitch == "Curveball":
        return "Curveball"
    elif pitch == "Splitter":
        return "Splitter"
    else:
        return "Exclude"


def create_ff_dataset(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH):
    """
    Creates the four-seam fastball dataset from Final_Target_Calc.csv.

    Steps:
        1. Load the final target/calculated dataset.
        2. Standardize TaggedPitchType naming.
        3. Create PitchBucket using the notebook's bucket logic.
        4. Keep only FourSeamFastball rows.
        5. Remove bad/undefined handedness rows.
        6. Add binary handedness features.
        7. Reorder handedness features after RelSide.
        8. Drop rows missing Extension or SpinRate.
        9. Save df_ff.csv and return the dataframe.

    Args:
        input_path: Path to Final_Target_Calc.csv.
        output_path: Path where df_ff.csv should be saved.

    Returns:
        pandas.DataFrame: The cleaned four-seam fastball dataset.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    # Match the exact pitch-type naming cleanup used in the notebook.
    df.loc[
        df["TaggedPitchType"] == "Changeup",
        "TaggedPitchType",
    ] = "ChangeUp"

    # Bucket pitches, then keep only the four-seam fastball family.
    df["PitchBucket"] = df.apply(assign_bucket, axis=1)
    df_ff = df[df["PitchBucket"] == "FourSeamFastball"].copy()

    # Remove handedness rows the notebook treated as bad values.
    df_ff = df_ff[
        (df_ff["BatterSide"].notna())
        & (df_ff["BatterSide"] != "Undefined")
    ].copy()

    df_ff = df_ff[df_ff["PitcherThrows"] != "Both"].copy()

    # Convert handedness to binary model features.
    df_ff["Is_Left_Handed_Pitcher"] = (
        df_ff["PitcherThrows"] == "Left"
    ).astype(int)
    df_ff["Is_Left_Handed_Batter"] = (
        df_ff["BatterSide"] == "Left"
    ).astype(int)

    # Place handedness features next to the other physical/modeling features.
    cols = df_ff.columns.tolist()
    cols.remove("Is_Left_Handed_Pitcher")
    cols.remove("Is_Left_Handed_Batter")

    insert_pos = cols.index("RelSide") + 1
    cols[insert_pos:insert_pos] = [
        "Is_Left_Handed_Pitcher",
        "Is_Left_Handed_Batter",
    ]

    df_ff = df_ff[cols]

    # Final notebook cleanup before saving df_ff.csv.
    df_ff = df_ff.dropna(subset=["Extension", "SpinRate"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_ff.to_csv(output_path, index=False)

    return df_ff


def create_team_ff_dataset(
    df_ff=None,
    team_code=DEFAULT_TEAM_CODE,
    ff_input_path=DEFAULT_OUTPUT_PATH,
    output_path=DEFAULT_TEAM_OUTPUT_PATH,
):
    """
    Creates the team-specific four-seam fastball dataset from df_ff.

    This follows the df_del_blu_ff section from the notebook:
        1. Keep rows where PitcherTeam matches the team code.
        2. Save df_del_blu_ff.csv.
        3. Clean Pitcher spacing around commas.
        4. Return the cleaned team dataframe.

    Args:
        df_ff: Optional four-seam fastball dataframe created by
            create_ff_dataset(). If not provided, ff_input_path is loaded.
        team_code: PitcherTeam value to keep.
        ff_input_path: Path to df_ff.csv when df_ff is not provided.
        output_path: Path where df_del_blu_ff.csv should be saved.

    Returns:
        pandas.DataFrame: Team-specific four-seam fastball dataset.
    """
    if df_ff is None:
        df_ff = pd.read_csv(ff_input_path)

    output_path = Path(output_path)

    df_del_blu_ff = df_ff[df_ff["PitcherTeam"] == team_code].copy()

    df_del_blu_ff["Pitcher"] = (
        df_del_blu_ff["Pitcher"]
        .str.strip()
        .str.replace(r"\s+,", ",", regex=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_del_blu_ff.to_csv(output_path, index=False)

    return df_del_blu_ff


if __name__ == "__main__":
    final_df = create_ff_dataset()
    team_df = create_team_ff_dataset(df_ff=final_df)
    print(f"df_ff.csv created with shape: {final_df.shape}")
    print(f"df_del_blu_ff.csv created with shape: {team_df.shape}")
