"""15: How large must a 30-day recent-Stuff+ change be before it beats
execution-to-execution sampling noise?

The coach-facing page shows a pitch type's Stuff+ over the trailing 30 days
minus the 30 days before that, gated on both windows clearing SAMPLE_FLOOR =
100 pitches (14_pitcher_pages.py). That floor was measured for a DIFFERENT
metric at a DIFFERENT window length: script 06 measured Location+ reliability
at SEASON length (100+ FF => usable Location+ read). Reused verbatim as a
30-day Stuff+ floor, it has never been checked against what a 30-day window
can actually detect, and on real data it blanks 61 of 63 arsenal rows because
the median 30-day volume per pitcher per pitch type is ~20 pitches.

There is also a real statistical gap, not just a wrong constant: every
reliability number in this suite (06, 12, 13) is built for a single window's
MEAN. The page displays a DIFFERENCE of two window means, whose error
variance is the SUM of the two windows' variances -- a quantity this suite
has never estimated. change_detectability.py supplies that extension; see its
docstring for the independence assumption and which direction it biases.

Stuff+ is deterministic given measured physical properties -- no outcome luck
enters it -- so the only source of window-to-window variation is genuine
pitch-to-pitch variation in execution, corrected for within-game clustering
(pitches in one outing share the day's release, fatigue, etc.), which is what
variance_components.effective_noise_scale measures and this script reuses.

Part 1  per-pitch variance of the DISPLAY-scale Stuff+ grade, by pitch type,
        naive and game-clustered, plus the design effect (clustered/naive).
        The FF number is sanity-checked against a previously measured range
        (per-pitcher SD 8.9-16.0, median ~11.5 -- see webapp_publisher/
        schema.py's PITCH_GRADE_BAND comment and its introducing commit).
Part 2  the detectability curve: SE of the 30-day recent-change and the
        minimum detectable change at 1 and 2 SE, over a window-size grid that
        spans what real data shows (a handful of pitches up to ~140), NOT
        restricted to n >= 100.
Part 3  the practical question: of the 63 real arsenal rows in the live
        bundle, how many would display under the current rule, under smaller
        fixed floors, and under an uncertainty rule (display when the
        observed |change| exceeds 1 or 2 SE of itself).

Input: STUFFPLUS_DATA / --data (source CSV, for Parts 1-2) and
STUFFPLUS_WORKDIR / --workdir, exactly as script 06. Part 3 additionally reads
--pages, the emitted pitcher_pages.json (Level II, licensed): read-only, never
copied into the repo, never printed by pitcher name -- only pitch-type-level
counts and changes leave this script.

Writes the aggregate detectability grid (numbers, not per-pitcher rows) to
--workdir. Never commit that file or anything under --workdir.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arsenal as ar
import change_detectability as cd
import fair_criterion as fc

# fair_criterion relabels the year pair to 2024/2025 roles; 2025 is the
# graded ("eval") season, matching 14_pitcher_pages.py's SEASON_ROLE_YEAR.
SEASON_ROLE_YEAR = 2025

# Matches 14_pitcher_pages.py's SAMPLE_FLOOR exactly, NOT the value this
# script is trying to replace: this floor only picks which pitchers count
# toward the qualifying-population mu/sd that sets the display scale, and
# reusing it here means the display-scale variance measured below is on the
# SAME scale the live page actually shows a coach, not a different one.
DISPLAY_SCALE_FLOOR = 100

MIN_HALF = 25  # same game-cluster half-split minimum as scripts 06/12/13

# Real 30-day windows run from a handful of pitches to ~140 (script 06's
# season-length floor of 100 is far outside that); this grid is NOT
# restricted to n >= 100, since the whole point is what happens below it.
WINDOW_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 140]
CANDIDATE_FLOORS = [15, 20, 25, 30, 40, 50, 75, 100]

FF_PRIOR_SD_RANGE = (8.9, 16.0)  # webapp_publisher/schema.py PITCH_GRADE_BAND comment
FF_PRIOR_SD_MEDIAN = 11.5


def extra_cli():
    """--pages and --min-half, layered on top of fc.paths() the way scripts
    12/13 layer their own extra flags without re-implementing --data/--workdir."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages",
                    default=r"C:/Users/jackdav/stuffplus_replication/workdir_webapp/pitcher_pages.json",
                    help="Emitted pitcher_pages.json for Part 3. Level II data: "
                         "read-only, never copied into the repo.")
    ap.add_argument("--min-half", type=int, default=MIN_HALF)
    args, _ = ap.parse_known_args()
    return args


def build_pitches(args):
    df = fc.load_pitches(args)
    fc.add_xt(df)
    return df


def per_type_variance(df, min_half):
    """Naive/clustered per-pitch variance of the DISPLAY grade, by pitch type.

    Fits each type's ridge + display scale exactly as 14_pitcher_pages.py
    does (arsenal.fit_type, same DISPLAY_SCALE_FLOOR), so the variance
    measured here is in the same units the live page shows. Also reports the
    per-PITCHER sd distribution (median, p10-p90) as a sanity check against a
    previously measured FF range -- a different summary than the pooled
    naive/clustered variance used for the floor math (see module docstring).
    """
    out = {}
    for tname, tags in ar.PITCH_TYPES:
        try:
            state = ar.fit_type(df, tags, DISPLAY_SCALE_FLOOR, fc, SEASON_ROLE_YEAR)
        except ValueError as err:
            print(f"skipping {tname}: {err}")
            continue
        season = state["pitches"].copy()
        season["grade"] = ar.to_display(season["ridge_pred"].values, state["mu"], state["sd"])
        vres = cd.per_pitch_variance(season, "grade", ["PitcherId"], min_half=min_half)
        per_pitcher_sd = season.groupby("PitcherId")["grade"].std(ddof=1).dropna()
        vres["n_pitchers_sd"] = int(len(per_pitcher_sd))
        if len(per_pitcher_sd) > 0:
            vres["sd_median"] = float(per_pitcher_sd.median())
            vres["sd_p10"] = float(per_pitcher_sd.quantile(0.10))
            vres["sd_p90"] = float(per_pitcher_sd.quantile(0.90))
        out[tname] = vres
    return out


def part1(var_by_type):
    print("\n" + "=" * 78)
    print("PART 1 -- PER-PITCH VARIANCE OF STUFF+ (display scale), BY PITCH TYPE")
    print("=" * 78)
    print(f"{'type':<10}{'pitchers':>9}{'naive var':>11}{'naive sd':>9}"
          f"{'clust var':>10}{'clust sd':>9}{'design eff':>11}")
    for t, v in var_by_type.items():
        print(f"{t:<10}{v['n_groups']:>9}{v['naive']:>11.2f}{np.sqrt(v['naive']):>9.2f}"
              f"{v['clustered']:>10.2f}{np.sqrt(v['clustered']):>9.2f}{v['design_effect']:>10.2f}x")

    ff = var_by_type.get("FF")
    if ff and ff.get("n_pitchers_sd", 0) > 0:
        lo, hi = FF_PRIOR_SD_RANGE
        print(f"\nFF per-pitcher SD sanity check (previously measured {lo}-{hi}, "
              f"median ~{FF_PRIOR_SD_MEDIAN}):")
        print(f"  measured median={ff['sd_median']:.1f}  "
              f"p10-p90=[{ff['sd_p10']:.1f}, {ff['sd_p90']:.1f}]  n={ff['n_pitchers_sd']} pitchers")
        if not (lo - 2 <= ff["sd_median"] <= hi + 2):
            print("  FLAG: median per-pitcher SD sits outside the previously measured band. "
                  "Treat this run's numbers with caution before using them for a floor -- "
                  "something about this bundle (data window, ridge refit, pitcher mix) may "
                  "differ from the run that produced 8.9-16.0.")
        else:
            print("  Consistent with the previously measured band.")
    return ff


def part2(var_by_type, workdir):
    print("\n" + "=" * 78)
    print("PART 2 -- DETECTABILITY CURVE (SE of the 30-day recent-change, by window size)")
    print("=" * 78)
    frames = []
    for t, v in var_by_type.items():
        grid = cd.detectability_grid(v["clustered"], WINDOW_GRID)
        grid.insert(0, "type", t)
        frames.append(grid)
        print(f"\n  {t}  (clustered per-pitch var={v['clustered']:.2f}, "
              f"design effect {v['design_effect']:.2f}x):")
        print(f"    {'n/window':>9}{'SE(diff)':>10}{'MDC@1SE':>10}{'MDC@2SE':>10}")
        for _, r in grid.iterrows():
            print(f"    {int(r['n_per_window']):>9}{r['se_diff']:>10.2f}"
                  f"{r['mdc_1se']:>10.2f}{r['mdc_2se']:>10.2f}")
    full = pd.concat(frames, ignore_index=True)
    path = os.path.join(workdir, "recent_change_detectability_grid.csv")
    full.to_csv(path, index=False)
    print(f"\nFull grid (all pitch types) written to {path} "
          "(aggregate numbers only, not per-pitcher rows).")
    return full


def load_real_rows(pages_path):
    """Level II data: read-only, never copied into the repo or printed by name.

    Returns (asof, rows) where rows is a list of {type, n_recent, n_prior,
    change, current_display} dicts -- one per arsenal entry, no pitcher
    identifier retained past this function.

    asof is not persisted in pitcher_pages.json (14_pitcher_pages.py computes
    it in-memory from the team's FF frame and never writes it out), so it is
    approximated here as the latest outing date across ALL pitch types in the
    bundle. That equals the true asof whenever the pitcher's last outing
    included at least one four-seam, which is the common case; a pitcher whose
    final outing was breaking-balls-only would make this approximation a few
    days late for that pitcher only.
    """
    with open(pages_path) as f:
        d = json.load(f)
    all_dates = [o["date"] for p in d["pitchers"] for o in p["outings"]]
    if not all_dates:
        raise ValueError("bundle has no outings; cannot approximate asof")
    asof = max(all_dates)

    rows = []
    for p in d["pitchers"]:
        outings = pd.DataFrame(p["outings"], columns=["date", "type", "n", "stuff"])
        for r in p["arsenal"]:
            sub = outings[outings["type"] == r["type"]]
            n_recent, n_prior, change = cd.window_counts_and_change(sub, asof)
            rows.append({"type": r["type"], "n_recent": n_recent, "n_prior": n_prior,
                        "change": change, "current_display": bool(r.get("aboveFloor"))})
    return asof, rows


def part3(asof, rows, var_by_type):
    print("\n" + "=" * 78)
    print(f"PART 3 -- WOULD IT DISPLAY? (real DEL_BLU bundle, asof~{asof})")
    print("=" * 78)
    tab = pd.DataFrame(rows)
    n = len(tab)
    print(f"  {n} arsenal rows total\n")

    def rule_both_floor(floor):
        return int(((tab["n_recent"] >= floor) & (tab["n_prior"] >= floor)).sum())

    def rule_se(k):
        cnt = 0
        for _, r in tab.iterrows():
            if r["change"] is None:
                continue
            v = var_by_type.get(r["type"])
            if v is None:
                continue
            se = cd.diff_se(v["clustered"], r["n_recent"], r["n_prior"])
            if np.isfinite(se) and abs(r["change"]) > k * se:
                cnt += 1
        return cnt

    print(f"  {'rule':<40}{'rows displayed':>16}")
    print(f"  {'current (both windows >= 100)':<40}{rule_both_floor(100):>12} / {n}")
    for floor in CANDIDATE_FLOORS:
        print(f"  {'both windows >= ' + str(floor):<40}{rule_both_floor(floor):>12} / {n}")
    print(f"  {'uncertainty rule: |change| > 1 SE':<40}{rule_se(1):>12} / {n}")
    print(f"  {'uncertainty rule: |change| > 2 SE':<40}{rule_se(2):>12} / {n}")
    print("\n  Row counts by pitch type (context for how thin non-FF windows are):")
    print(tab["type"].value_counts().to_string())
    return tab


def main():
    args = fc.paths()
    extra = extra_cli()
    df = build_pitches(args)
    var_by_type = per_type_variance(df, extra.min_half)
    part1(var_by_type)
    part2(var_by_type, args.workdir)
    asof, rows = load_real_rows(extra.pages)
    part3(asof, rows, var_by_type)


if __name__ == "__main__":
    main()
