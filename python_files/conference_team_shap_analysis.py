"""
Run conference-team linear-regression SHAP scoring for four-seam fastballs.

This script converts the final conference-team workflow from conf_teams.ipynb
into reusable functions. It filters four-seam fastball data to selected
conference matchups, trains the same Linear Regression model used for SHAP in
the notebook, and creates DEL_BLU pitcher-level normalized SHAP scores.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_FF_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_ff.csv"
)
DEFAULT_DEL_BLU_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_del_blu_ff.csv"
)
DEFAULT_COACH_FEATURE_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/conf_teams/Stuff+ Linear Regressoin.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/conf_teams"
)


CUSA_TEAMS = [
    "DAL_PAT",
    "WES_HIL",
    "KEN_OWL",
    "JAC_GAM",
    "LOU_BUL",
    "FLO_PAN",
    "NMS_AGG",
    "LIB_FLA",
    "MTSU_BLU",
    "SAM_BEA",
    "UTS_ROA",
    "CHA_FOR",
    "FAU_OWL",
    "RIC_OWL",
    "UAB_BLA",
]

SUNBELT_TEAMS = [
    "APP_MOU",
    "ASU_RED",
    "COA_CHA",
    "GEO_EAG",
    "GEO_PAN",
    "JMU_DUK",
    "LOU_CAJ",
    "MAR_THU",
    "OLD_MON",
    "SAL_JAG",
    "SOU_GOL",
    "TEX_BOB",
    "TRO_TRJ",
    "ULM_WAR",
]

AMERICAN_TEAMS = [
    "ECU_PIR",
    "HOU_COU",
    "WIC_SHO",
    "UCF_KNI",
    "CIN_BEA",
    "MT",
    "TUL_GRE",
    "USF_BUL",
    "UTS_ROA",
    "UAB_BLA",
    "FAU_OWL",
    "CHA_FOR",
    "RIC_OWL",
]

CONFERENCE_TEAMS = [
    # C-USA
    "DAL_PAT",
    "WES_HIL",
    "KEN_OWL",
    "JAC_GAM",
    "LOU_BUL",
    "FLO_PAN",
    "NMS_AGG",
    "LIB_FLA",
    "MTSU_BLU",
    "SAM_BEA",
    # Shared C-USA / American
    "UTS_ROA",
    "CHA_FOR",
    "FAU_OWL",
    "RIC_OWL",
    "UAB_BLA",
    # Sun Belt
    "APP_MOU",
    "ASU_RED",
    "COA_CHA",
    "GEO_EAG",
    "GEO_PAN",
    "JMU_DUK",
    "LOU_CAJ",
    "MAR_THU",
    "OLD_MON",
    "SAL_JAG",
    "SOU_GOL",
    "TEX_BOB",
    "TRO_TRJ",
    "ULM_WAR",
    # American only
    "ECU_PIR",
    "HOU_COU",
    "WIC_SHO",
    "UCF_KNI",
    "CIN_BEA",
    "MT",
    "TUL_GRE",
    "USF_BUL",
]

MODEL_FEATURES = [
    "HorzBreak",
    "InducedVertBreak",
    "RelHeight",
    "RelSide",
    "velocity_differential",
    "EffectiveVelo",
    "SpinRate",
]

TARGET_COLUMN = "Target"


def evaluate_predictions(y_true, y_pred):
    """Calculates MAE, RMSE, and R2 for model predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def filter_conference_matchups(
    ff_path=DEFAULT_FF_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Loads df_ff.csv and keeps only selected conference matchups.

    A row is kept only when both PitcherTeam and BatterTeam are in the
    combined conference team list.

    Returns:
        pandas.DataFrame: Conference-filtered four-seam dataset.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_ff = pd.read_csv(ff_path)
    df_ff_conf = df_ff[
        df_ff["PitcherTeam"].isin(CONFERENCE_TEAMS)
        & df_ff["BatterTeam"].isin(CONFERENCE_TEAMS)
    ].copy()

    outside_teams = (
        set(df_ff_conf["PitcherTeam"]).union(df_ff_conf["BatterTeam"])
        - set(CONFERENCE_TEAMS)
    )
    if outside_teams:
        raise ValueError(f"Outside conference teams found: {outside_teams}")

    df_ff_conf.to_csv(output_dir / "df_ff_conf.csv", index=False)

    return df_ff_conf


def load_coach_feature_file(coach_feature_path=DEFAULT_COACH_FEATURE_PATH):
    """
    Loads the coach Stuff+ spreadsheet used to confirm model features.

    The notebook inspected this file, then trained with MODEL_FEATURES.
    """
    return pd.read_excel(coach_feature_path, engine="openpyxl")


def train_linear_regression_model(df_ff_conf):
    """
    Trains the same Linear Regression model used for SHAP in conf_teams.ipynb.

    Returns:
        tuple: trained model, X_train, train metrics dict, test metrics dict.
    """
    X = df_ff_conf[MODEL_FEATURES]
    y = df_ff_conf[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_train_pred = lr_model.predict(X_train)
    y_test_pred = lr_model.predict(X_test)

    train_metrics = evaluate_predictions(y_train, y_train_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    return lr_model, X_train, train_metrics, test_metrics


def create_del_blu_linear_shap_scores(
    lr_model,
    X_train,
    del_blu_path=DEFAULT_DEL_BLU_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Runs Linear Regression SHAP and creates normalized DEL_BLU pitcher scores.

    This follows the notebook logic:
        1. Load df_del_blu_ff.csv.
        2. Select the coach-model features.
        3. Run shap.LinearExplainer.
        4. Create pitch-level SHAP values.
        5. Average SHAP values by pitcher.
        6. Create raw_shap_sum.
        7. Normalize with mean=100 and std=15, using lower raw_shap_sum as
           better.
        8. Round normalized scores to whole numbers.

    Returns:
        tuple: pitch-level SHAP dataframe and pitcher-level score dataframe.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_del_blu_ff = pd.read_csv(del_blu_path)
    X_del_blu = df_del_blu_ff[MODEL_FEATURES]

    explainer = shap.LinearExplainer(lr_model, X_train)
    shap_values_del_blu = explainer(X_del_blu)

    shap_df = pd.DataFrame(
        shap_values_del_blu.values,
        columns=MODEL_FEATURES,
        index=df_del_blu_ff.index,
    )
    shap_df["Pitcher"] = df_del_blu_ff["Pitcher"].values

    cols = ["Pitcher"] + [col for col in shap_df.columns if col != "Pitcher"]
    shap_df = shap_df[cols]
    shap_df.to_csv(output_dir / "delaware_lr_shap_pitch_level.csv", index=False)

    pitcher_shap_del = shap_df.groupby("Pitcher").mean()
    pitcher_shap_del["raw_shap_sum"] = pitcher_shap_del.sum(axis=1)

    mean_val = pitcher_shap_del["raw_shap_sum"].mean()
    std_val = pitcher_shap_del["raw_shap_sum"].std()
    scaling_factor = 15 / std_val

    pitcher_shap_del["normalized_score"] = (
        100 - (pitcher_shap_del["raw_shap_sum"] - mean_val) * scaling_factor
    )
    pitcher_shap_del["normalized_score"] = (
        pitcher_shap_del["normalized_score"].round(0).astype(int)
    )

    pitcher_shap_del = pitcher_shap_del.sort_values(
        "normalized_score",
        ascending=False,
    )
    pitcher_shap_del.to_csv(output_dir / "delaware_lr_pitcher_scores.csv")

    return shap_df, pitcher_shap_del


def run_conference_team_shap_analysis(
    ff_path=DEFAULT_FF_PATH,
    coach_feature_path=DEFAULT_COACH_FEATURE_PATH,
    del_blu_path=DEFAULT_DEL_BLU_PATH,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """
    Runs the full conference-team Linear Regression SHAP workflow.

    Returns:
        tuple: filtered conference dataset, model metrics, pitch-level SHAP,
        and pitcher-level scores.
    """
    df_ff_conf = filter_conference_matchups(
        ff_path=ff_path,
        output_dir=output_dir,
    )

    # Load the spreadsheet to preserve the notebook's coach-feature check.
    load_coach_feature_file(coach_feature_path)

    lr_model, X_train, train_metrics, test_metrics = train_linear_regression_model(
        df_ff_conf
    )
    shap_df, pitcher_scores = create_del_blu_linear_shap_scores(
        lr_model=lr_model,
        X_train=X_train,
        del_blu_path=del_blu_path,
        output_dir=output_dir,
    )

    metrics = {
        "train": train_metrics,
        "test": test_metrics,
    }

    return df_ff_conf, metrics, shap_df, pitcher_scores


if __name__ == "__main__":
    conf_df, model_metrics, pitch_shap, pitcher_scores = (
        run_conference_team_shap_analysis()
    )

    print(f"Conference-filtered rows: {conf_df.shape}")
    print(f"Pitch-level SHAP rows: {pitch_shap.shape}")
    print(f"Pitcher score rows: {pitcher_scores.shape}")

    print("\nTrain Metrics:")
    for metric, value in model_metrics["train"].items():
        print(f"{metric}: {value}")

    print("\nTest Metrics:")
    for metric, value in model_metrics["test"].items():
        print(f"{metric}: {value}")

    print("\nTop normalized pitcher scores:")
    print(
        pitcher_scores[["raw_shap_sum", "normalized_score"]]
        .head()
        .to_string()
    )
