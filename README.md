Pitch's physical parameters against expected runs model
=======================================================
# Python Files

This folder contains the main Python scripts and notebook used for building expected-runs targets, calculated pitch features, conference-based filtering, model training, and SHAP-based pitcher analysis.

## Files

| File | Description |
|---|---|
| `Helpers.py` | Contains helper functions for reconstructing base runner states, creating game-state labels, calculating runs remaining, and computing zero-run probabilities. |
| `generate_gamestate_summary.py` | Builds a game-state summary table across available TrackMan CSV files, including expected runs and zero-run probability for each game state. |
| `target_and_calculated_pipeline.py` | Creates the final modeling dataset by adding expected-runs targets and calculated pitch features such as velocity differential and movement differences. |
| `conf_teams.ipynb` | Notebook for conference team shortcut mapping, filtering four-seam fastball data to selected conferences, training/evaluating models, and comparing SHAP-based pitcher scores with coach scores. |

## Workflow Overview

1. `Helpers.py` defines reusable functions for runner states, game states, and run expectancy calculations.
2. `generate_gamestate_summary.py` uses those helper functions to create a game-state expected-runs summary.
3. `target_and_calculated_pipeline.py` applies expected-runs values to pitch-level data and creates the final target/calculated feature dataset.
4. `conf_teams.ipynb` filters four-seam fastball data for C-USA, Sun Belt, and American Athletic Conference teams, trains regression models, runs SHAP analysis, and creates normalized pitcher scores.

## Main Outputs

The workflow supports creation of:

- game-state expected-runs summary files
- pitch-level target datasets
- calculated feature datasets
- conference-filtered four-seam datasets
- SHAP value exports
- normalized pitcher scoring tables



The documentation for the project can be found at https://NSF-DARSE.github.io/expected-runs
