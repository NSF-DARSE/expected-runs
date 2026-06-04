"""
Train the finalized Random Forest model for four-seam fastball data.

This script converts the Random Forest modeling section of modeling_ff.ipynb
into reusable functions. It trains the same full Random Forest model that was
used later for permutation importance and SHAP analysis in the notebook.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_INPUT_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/CSV_files/df_ff_new/df_ff.csv"
)
DEFAULT_MODEL_PATH = Path(
    "/Users/suma/Downloads/Baseball_Project/model_files/rf_full_model_ff.pkl"
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

TARGET_COLUMN = "Target"


def evaluate_predictions(y_true, y_pred):
    """
    Calculates the same evaluation metrics used in modeling_ff.ipynb.

    Args:
        y_true: Actual target values.
        y_pred: Model predictions.

    Returns:
        dict: MAE, RMSE, and R2.
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_random_forest_model(
    input_path=DEFAULT_INPUT_PATH,
    model_path=DEFAULT_MODEL_PATH,
):
    """
    Trains and saves the finalized full Random Forest model.

    This follows the same logic as modeling_ff.ipynb:
        1. Load df_ff.csv.
        2. Select the same feature columns and Target column.
        3. Split into train/test sets with test_size=0.2 and random_state=42.
        4. Train RandomForestRegressor with the notebook's final parameters.
        5. Evaluate train and test predictions.
        6. Save the trained model with joblib.

    Args:
        input_path: Path to df_ff.csv.
        model_path: Path where the trained Random Forest model should be saved.

    Returns:
        tuple: trained model, train metrics dict, test metrics dict.
    """
    input_path = Path(input_path)
    model_path = Path(model_path)

    df_ff = pd.read_csv(input_path)

    X = df_ff[FEATURE_COLUMNS]
    y = df_ff[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    train_metrics = evaluate_predictions(y_train, y_train_pred)
    test_metrics = evaluate_predictions(y_test, y_test_pred)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, model_path)

    return rf, train_metrics, test_metrics


if __name__ == "__main__":
    model, train_results, test_results = train_random_forest_model()

    print("Train Metrics:")
    for metric, value in train_results.items():
        print(f"{metric}: {value}")

    print("\nTest Metrics:")
    for metric, value in test_results.items():
        print(f"{metric}: {value}")

    print(f"\nModel saved to: {DEFAULT_MODEL_PATH}")
