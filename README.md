Pitch's physical parameters against expected runs model
=======================================================
## Create Virtual Environment

```bash
python -m venv .aiml
```

### Activate Environment

Linux/macOS:

```bash
source .aiml/bin/activate
```

Windows PowerShell:

```powershell
.aiml\Scripts\Activate.ps1
```

## Installation

```bash
pip install -e ".[dev]"
```

## Example
How to run the code is shown in the the EXAMPLES directory

## Modules Overview

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
