"""
Run SHAP analysis and create normalized pitcher scores for DEL_BLU.

This script combines the SHAP workflow from modeling_ff.ipynb with the score
scaling workflow from scaling_shap.ipynb. It:
    1. Loads the DEL_BLU four-seam fastball dataset.
    2. Loads the saved Random Forest model.
    3. Calculates pitch-level SHAP values.
    4. Generates SHAP summary plots.
    5. Averages SHAP values by pitcher.
    6. Creates raw SHAP sums and normalized 100-scale pitcher scores.

It intentionally stops before the feature-contribution calculations.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


DEFAULT_DEL_BLU_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_del_blu_ff.csv"
)
DEFAULT_MODEL_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/model_files/rf_full_model_ff.pkl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/shap_analysis"
)


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


FEATURE_NAMES = {
    "horzbreakdiff": "Horizontal Break Diff",
    "velocity_differential": "Velocity Diff",
    "HorzBreak": "Horizontal Break",
    "InducedVertBreak": "Induced Vertical Break",
    "vertbreakdiff": "Vertical Break Diff",
    "RelSide": "Release Side",
    "RelHeight": "Release Height",
    "EffectiveVelo": "Effective Velocity",
    "Extension": "Extension",
    "SpinRate": "Spin Rate",
    "Is_Left_Handed_Batter": "Left-Handed Batter",
    "Is_Left_Handed_Pitcher": "Left-Handed Pitcher",
}


def create_shap_pitch_level_data(
    del_blu_path=DEFAULT_DEL_BLU_PATH,
    model_path=DEFAULT_MODEL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Calculates pitch-level SHAP values for the DEL_BLU dataset.

    Args:
        del_blu_path: Path to df_del_blu_ff.csv.
        model_path: Path to the saved Random Forest model.
        output_dir: Folder where SHAP outputs should be saved.

    Returns:
        tuple: DEL_BLU dataframe, feature dataframe, SHAP dataframe.
    """
    del_blu_path = Path(del_blu_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_del_blu_ff = pd.read_csv(del_blu_path)
    X_del_blu = df_del_blu_ff[FEATURE_COLUMNS]

    rf = joblib.load(model_path)
    explainer = shap.TreeExplainer(rf)

    shap_values = explainer.shap_values(
        X_del_blu,
        approximate=True,
    )

    shap_df = pd.DataFrame(
        shap_values,
        columns=FEATURE_COLUMNS,
    )
    shap_df["Pitcher"] = df_del_blu_ff["Pitcher"].values

    cols = ["Pitcher"] + [col for col in shap_df.columns if col != "Pitcher"]
    shap_df = shap_df[cols].round(5)

    shap_df.to_csv(output_dir / "pitch_level_shap.csv", index=False)

    return df_del_blu_ff, X_del_blu, shap_df


def save_shap_plots(X_del_blu, shap_df, output_dir=DEFAULT_OUTPUT_DIR):
    """
    Saves SHAP summary plots for the DEL_BLU pitch-level SHAP values.

    Args:
        X_del_blu: Feature dataframe used for SHAP analysis.
        shap_df: Pitch-level SHAP dataframe with Pitcher plus feature columns.
        output_dir: Folder where plots should be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_values = shap_df[FEATURE_COLUMNS].values
    X_plot = X_del_blu.rename(columns=FEATURE_NAMES)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="dot",
        cmap="coolwarm",
        show=False,
    )
    plt.title("Feature Impact on Pitch Effectiveness (SHAP)", fontsize=14)
    plt.xlabel("Impact on Expected Run Change", fontsize=12)
    plt.ylabel("Pitch Features", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_plot,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_normalized_pitcher_scores(
    shap_df,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Creates pitcher-level normalized scores from pitch-level SHAP values.

    This follows scaling_shap.ipynb through the normalized-score step:
        1. Average SHAP values by pitcher.
        2. Sum averaged feature SHAP values into raw_shap_sum.
        3. Calculate mean and standard deviation of raw_shap_sum.
        4. Scale scores so mean = 100 and standard deviation = 15.

    Args:
        shap_df: Pitch-level SHAP dataframe.
        output_dir: Folder where normalized scores should be saved.

    Returns:
        pandas.DataFrame: Pitcher-level normalized SHAP score table.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shap_cols = [col for col in shap_df.columns if col != "Pitcher"]

    pitcher_avg_df = shap_df.groupby("Pitcher")[shap_cols].mean().reset_index()
    pitcher_avg_df.to_csv(
        output_dir / "pitcher_average_shap.csv",
        index=False,
    )

    pitcher_avg_df["raw_shap_sum"] = pitcher_avg_df[shap_cols].sum(axis=1)

    std_val = pitcher_avg_df["raw_shap_sum"].std()
    mean_val = pitcher_avg_df["raw_shap_sum"].mean()
    scaling_factor = 15 / std_val

    pitcher_avg_df["normalized_score"] = (
        100 - (pitcher_avg_df["raw_shap_sum"] - mean_val) * scaling_factor
    )

    pitcher_avg_df.to_csv(
        output_dir / "pitcher_normalized_scores.csv",
        index=False,
    )

    return pitcher_avg_df


def run_shap_scaling_analysis(
    del_blu_path=DEFAULT_DEL_BLU_PATH,
    model_path=DEFAULT_MODEL_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Runs the full SHAP and normalized-score workflow.

    Returns:
        tuple: pitch-level SHAP dataframe and pitcher-level score dataframe.
    """
    _, X_del_blu, shap_df = create_shap_pitch_level_data(
        del_blu_path=del_blu_path,
        model_path=model_path,
        output_dir=output_dir,
    )
    save_shap_plots(
        X_del_blu=X_del_blu,
        shap_df=shap_df,
        output_dir=output_dir,
    )
    pitcher_avg_df = create_normalized_pitcher_scores(
        shap_df=shap_df,
        output_dir=output_dir,
    )

    return shap_df, pitcher_avg_df


if __name__ == "__main__":
    pitch_level_shap, pitcher_scores = run_shap_scaling_analysis()

    print(f"Pitch-level SHAP rows: {pitch_level_shap.shape}")
    print(f"Pitcher score rows: {pitcher_scores.shape}")
    print(f"Outputs saved to: {DEFAULT_OUTPUT_DIR}")
    print("\nTop normalized pitcher scores:")
    print(
        pitcher_scores.sort_values("normalized_score", ascending=False)
        .head()
        .to_string(index=False)
    )
