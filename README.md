# Baseball Pitch Physical Parameters Against Expected Runs Model

This project builds an expected-runs based pitch evaluation workflow for baseball TrackMan data. It creates pitch-level run value targets, engineers physical pitch features, trains models on four-seam fastballs, and uses SHAP values to explain pitcher-level performance.

The current workflow has moved beyond the original run-expectancy scripts. It now includes four-seam fastball dataset creation, Random Forest modeling, SHAP feature impact analysis, normalized pitcher scores, and conference-filtered linear regression SHAP scoring.

## Project Files

| File | Description |
| --- | --- |
| `Helpers.py` | Helper functions for reconstructing base runner states, creating game-state labels, calculating runs remaining, and computing zero-run probabilities. |
| `generate_gamestate_summary.py` | Builds an all-year game-state summary from TrackMan CSV files, including expected runs and zero-run probability for each game state. |
| `target_and_calculated_pipeline.py` | Creates the final pitch-level modeling dataset by adding expected-runs targets and calculated pitch features such as velocity differential, vertical break difference, and horizontal break difference. |
| `ff_data_set_creation.py` | Filters the final target/calculated dataset to four-seam fastballs, cleans handedness fields, creates binary handedness features, and saves both the full four-seam dataset and the DEL_BLU pitcher-team subset. |
| `random_forest_modeling.py` | Trains the finalized Random Forest regression model for four-seam fastball target prediction and saves the fitted model file. |
| `shap_scaling_analysis.py` | Runs Random Forest SHAP analysis for DEL_BLU four-seam fastballs, creates pitch-level SHAP values, saves SHAP summary plots, and produces normalized pitcher scores. |
| `conference_team_shap_analysis.py` | Filters four-seam data to selected C-USA, Sun Belt, and American Athletic Conference teams, trains a Linear Regression model, and creates DEL_BLU SHAP-based pitcher scores for comparison with coach-style Stuff+ scoring. |
| `conf_teams.ipynb` | Notebook version of the conference filtering, linear regression modeling, and SHAP comparison workflow. |

## Workflow Overview

1. `Helpers.py` reconstructs runner states, game states, and runs remaining after each pitch.
2. `generate_gamestate_summary.py` scans TrackMan CSV files and creates a run expectancy table for each count/base/out game state.
3. `target_and_calculated_pipeline.py` maps expected-runs values back to pitch-level data and calculates the target value for each pitch.
4. `ff_data_set_creation.py` filters the final dataset to four-seam fastballs and creates model-ready handedness features.
5. `random_forest_modeling.py` trains the main Random Forest model for expected run change prediction.
6. `shap_scaling_analysis.py` explains the Random Forest model with SHAP and converts pitcher-level SHAP summaries into normalized scores.
7. `conference_team_shap_analysis.py` runs a separate conference-team Linear Regression SHAP workflow for comparing DEL_BLU pitchers against selected conference competition.

## Main Outputs

The workflow supports creation of:

- game-state expected-runs summary files
- final pitch-level target and calculated feature datasets
- four-seam fastball modeling datasets
- DEL_BLU four-seam fastball subsets
- trained Random Forest model files
- pitch-level SHAP value exports
- SHAP summary plots
- pitcher average SHAP tables
- normalized pitcher scoring tables
- conference-filtered four-seam datasets
- conference Linear Regression SHAP pitcher score tables

## Model Features

The Random Forest four-seam model uses physical pitch, handedness, and calculated differential features:

- `SpinRate`
- `Extension`
- `HorzBreak`
- `InducedVertBreak`
- `EffectiveVelo`
- `RelHeight`
- `RelSide`
- `Is_Left_Handed_Pitcher`
- `Is_Left_Handed_Batter`
- `vertbreakdiff`
- `horzbreakdiff`
- `velocity_differential`

The conference Linear Regression SHAP workflow uses:

- `HorzBreak`
- `InducedVertBreak`
- `RelHeight`
- `RelSide`
- `velocity_differential`
- `EffectiveVelo`
- `SpinRate`

## Requirements

The scripts use Python with the following main packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `shap`
- `joblib`
- `matplotlib`
- `openpyxl`

## Notes

Some scripts currently use absolute local paths under `/Users/suma/Downloads/Baseball_Project/`. If this project is run on another machine, update those default paths or pass custom paths into the script functions.

Generated data files, model files, virtual environments, `__pycache__/`, and system files such as `.DS_Store` should not be committed unless they are intentionally part of a release.

Project documentation can be found at: https://NSF-DARSE.github.io/expected-runs
