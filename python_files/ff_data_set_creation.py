"""
Create the four-seam fastball dataset used for modeling.

This script is the Python-file version of the ff_data_set_creation.ipynb
workflow. It creates timestamped df_ff and team-specific df_del_blu_ff files
using the same logic as the notebook.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/corrected_target_outputs/Final_Target_Calc_2109.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/corrected_target_outputs"
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


def get_timestamped_output_paths(output_dir=DEFAULT_OUTPUT_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    output_dir = Path(output_dir)
    ff_output_path = output_dir / f"df_ff_{timestamp}.csv"
    team_output_path = output_dir / f"df_del_blu_ff_{timestamp}.csv"

    return ff_output_path, team_output_path


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


def create_ff_dataset(input_path=DEFAULT_INPUT_PATH, output_path=None):
    """
    Creates the four-seam fastball dataset from the final target/calculated dataset.

    Steps:
        1. Load the final target/calculated dataset.
        2. Standardize TaggedPitchType naming.
        3. Create PitchBucket using the notebook's bucket logic.
        4. Keep only FourSeamFastball rows.
        5. Remove bad/undefined handedness rows.
        6. Add binary handedness features.
        7. Reorder handedness features after RelSide.
        8. Drop rows missing Extension or SpinRate.
        9. Optionally save df_ff and return the dataframe.

    Args:
        input_path: Path to Final_Target_Calc.csv.
        output_path: Optional path where df_ff should be saved.

    Returns:
        pandas.DataFrame: The cleaned four-seam fastball dataset.
    """
    input_path = Path(input_path)

    if output_path is not None:
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

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_ff.to_csv(output_path, index=False)

    return df_ff


def create_team_ff_dataset(
    df_ff=None,
    team_code=DEFAULT_TEAM_CODE,
    ff_input_path=None,
    output_path=None,
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
        ff_input_path: Path to df_ff when df_ff is not provided.
        output_path: Optional path where df_del_blu_ff should be saved.

    Returns:
        pandas.DataFrame: Team-specific four-seam fastball dataset.
    """
    if df_ff is None:
        if ff_input_path is None:
            raise ValueError("Either df_ff or ff_input_path must be provided.")
        df_ff = pd.read_csv(ff_input_path)

    if output_path is not None:
        output_path = Path(output_path)

    df_del_blu_ff = df_ff[df_ff["PitcherTeam"] == team_code].copy()

    df_del_blu_ff["Pitcher"] = (
        df_del_blu_ff["Pitcher"]
        .str.strip()
        .str.replace(r"\s+,", ",", regex=True)
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_del_blu_ff.to_csv(output_path, index=False)

    return df_del_blu_ff


if __name__ == "__main__":
    ff_output_path, team_output_path = get_timestamped_output_paths()

    final_df = create_ff_dataset(output_path=ff_output_path)
    team_df = create_team_ff_dataset(
        df_ff=final_df,
        output_path=team_output_path,
    )

    print(f"df_ff created: {ff_output_path}")
    print(f"df_ff shape: {final_df.shape}")

    print(f"df_del_blu_ff created: {team_output_path}")
    print(f"df_del_blu_ff shape: {team_df.shape}")
