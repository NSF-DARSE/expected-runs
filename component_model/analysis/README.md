# Location+ / Stuff+ analysis suite

Reproducible pipeline behind the results in [`../RESULTS.md`](../RESULTS.md). All
shared math (the fair criterion, the fixed Stuff+ Ridge, the location maps, the
pitcher panel) lives in `fair_criterion.py`; the numbered scripts only import it.

## Setup

The source data is licensed TrackMan (Level II). Point the scripts at the final
target CSV and a working directory **outside the repository** (or gitignored):

```
set STUFFPLUS_DATA=<path to Final_Target_Calc_####.csv>
set STUFFPLUS_WORKDIR=<cache/output dir outside the repo>
```

or pass `--data` / `--workdir` to any script. The first run builds parquet caches
in the workdir; delete them after regenerating the source CSV.

Requires: pandas, pyarrow, numpy, scipy, scikit-learn.

## Run order

| script | question | depends on |
|---|---|---|
| `01_fair_criterion_anchors.py` | Does the fair criterion reproduce the anchor numbers? | — |
| `02_location_gate.py` | Do location traits repeat year over year? | 01 pass |
| `03_location_plus.py` | Does Location+ add value over results, and Stuff+ over both? | 01 pass |
| `04_count_conditioning.py` | Does count-conditioning improve Location+, and where does the gain come from? | 01 pass |
| `05_scaling_combine.py` | Are the 100±15 scales and the equal-weight blend honest? | 04 (writes count_scores.parquet) |
| `06_sample_floor.py` | How many FF before a Location+ read is trustworthy? (`--team` for staff flags) | 01 pass |
| `07_models_vs_results.py` | Do the models alone out-predict the stat line? | 04 |
| `08_staff_scores.py` | Staff scoresheet: scores, flags, explanation grids (`--team`) | 01 pass |
| `09_secondary_pitches.py` | Does the stack transfer to sliders/changeups/curves? | 01 pass |

**Run 01 first, every time the source data changes.** If its prints do not match
the anchor table in RESULTS.md, stop and reconcile before trusting anything else.

## Rules

- Never commit workdir contents, derived TrackMan values, or per-pitcher output.
- The fair criterion (xT/adjT) and the Stuff+ Ridge are FIXED references. Changing
  them invalidates every comparison; do it deliberately and re-anchor everything.
- Scope: four-seam fastballs, 2024→2025, n=699 qualified pitchers. Conclusions are
  provisional pending 2026 replication.
- Evaluation protocol and score-design principles: `../FRAMEWORK.md`.
