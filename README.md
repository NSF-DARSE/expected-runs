# Expected Runs: pitch evaluation from TrackMan physics

A pitch-evaluation research project built on college baseball TrackMan data. Every
pitch is assigned a run-value target from an expected-runs table, and pitcher-level
scores are built from pitch physics and location rather than from outcomes, so they
stay readable on the small samples a college season provides.

The project reports three scores, all on a 100 +/- 15 scale:

- **Stuff+** — a Ridge model on release and movement features (velocity, spin,
  extension, break, and differentials off the pitcher's own fastball).
- **Location+** — a binned (plate x, plate z) run-value map. A fastball-only score:
  it is reliable but has no predictive validity on secondary pitches.
- **Pitching+** — an equal-weight z-score blend of the two. Equal weights beat every
  fitted alternative we tried.

## Start here

| Document | What it covers |
|---|---|
| [`component_model/FRAMEWORK.md`](component_model/FRAMEWORK.md) | Why the approach changed, how the model is structured, and the score design principles. |
| [`component_model/RESULTS.md`](component_model/RESULTS.md) | Every measured result, including the negative ones, plus the 2026 replication verdicts. |
| [`component_model/analysis/README.md`](component_model/analysis/README.md) | How to reproduce all of it: setup, run order, and what each script answers. |

## What the evidence says

Findings are held to a replication gate: a result discovered on the 2024→2025
season pair is not adopted until it survives 2025→2026. `RESULTS.md` records the
outcome either way.

- Location is a real skill, separate from stuff, and the two are close to
  orthogonal. Both earn their place on top of a pitcher's own prior results.
- The models beat the stat line. Scores built with no access to a pitcher's own
  results out-predict those results as a forecast of next season.
- Contact quality is not visible in location-blind stuff at this level, and a
  promising "deployment" component was **refuted** on replication and withdrawn.
  Both are documented rather than buried.
- Most of what looks like year-to-year pitcher variation is noise, not skill. A
  variance decomposition puts measurement noise at ~70% of pitcher-season
  variance, stable skill at ~24%, and real drift at ~6%. Pitching+ captures
  about 58% of the stable part.
- That bounds the work: criterion reliability caps attainable validity near 0.55,
  and the current stack reaches ~0.39. Precision (sample size, pooling,
  shrinkage) is a bigger lever than additional features.

## Layout

| Path | Contents |
|---|---|
| `component_model/analysis/` | The analysis suite. `fair_criterion.py` holds all shared math; numbered scripts `01`–`15` each answer one question and import it. `tests/` covers the estimators. |
| `component_model/portal/` | Arsenal-level grading and the transfer-portal evaluation board. |
| `python_files/` | Dataset construction: runner-state reconstruction, the game-state expected-runs table, and the pitch-level target pipeline. |
| `webapp_publisher/` | Publishes the precomputed JSON bundle consumed by the coach-facing web application. |
| `docs/` | Design specs, plans, and working notes. |

Run `01_fair_criterion_anchors.py` first, every time the source data changes. If its
output does not match the anchor table in `RESULTS.md`, stop and reconcile before
trusting anything downstream.

## Data

The source data is licensed TrackMan and is **never committed to this repository**.
No pitch-level values, derived player grades, or licensed documentation belong in
version control; `.gitignore` enforces this and it should stay that way. Scripts
locate data through the `STUFFPLUS_DATA` and `STUFFPLUS_WORKDIR` environment
variables (or `--data` / `--workdir`), both pointing outside the repository.
Published results in `RESULTS.md` are aggregates only.

Anyone reproducing this work needs their own TrackMan license and data access.

## Requirements

Python with `pandas`, `numpy`, `scipy`, `scikit-learn`, and `pyarrow`. The legacy
pipeline additionally uses `shap`, `joblib`, `matplotlib`, and `openpyxl`. Tests run
under `pytest` from the repository root.

## Legacy pipeline

The original workflow trained an unregularized Random Forest on four-seam fastballs
and explained it with SHAP. It is **superseded**: its pitcher scores were measured at
the noise floor, meaning they carried no repeatable signal. The scripts remain in
`python_files/` because the dataset-construction half is still in use, but the Random
Forest scoring path should not be used for evaluation. Some of those scripts still
carry absolute local paths and need path arguments to run elsewhere.

Project documentation: https://NSF-DARSE.github.io/expected-runs
