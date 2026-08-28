"""Both criteria in one unit: runs per 9, deserved vs actual. Emits chart JSON.

The correlation comparison (coach_model_comparison.py) and the runs comparison
(coach_model_coach_units.py) disagreed, and the cause is the CRITERION, not
in-sample fitting (verified: in-sample inflation on the ridge is -0.001). This
script puts both criteria in the same unit so they can sit side by side on one
coach-facing page without asking the reader to hold two yardsticks.

  DESERVED runs/9  = total adjT over the season / innings * 9. adjT is expected
                     runs with batted-ball luck and defense stripped out, so this
                     is "what he should have allowed".
  ACTUAL runs/9    = RA9, what actually crossed the plate.

Same three scores, same controls, same paired bootstrap; only the outcome changes.
That isolates the criterion, which is the whole point of the exercise.

SIGN CONVENTION: adjT and RA9 are both runs allowed, LOWER = BETTER. Scores are in
higher-is-better display frames (`_hi`). Reported effects are RUNS OF IMPROVEMENT
per +1 SD, so positive always means the score is working.

Data rules: reads the workdir caches, writes chart JSON to the workdir only. The
JSON carries no pitcher names. Never committed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import coach_model_paired as cp
import fair_criterion as fc

N_BOOT = 4000
SCORES = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi", "adjres_hi", "pitch_hi"]
LABELS = {"velo_hi": "Velocity only", "coach_hi": "Coach's card", "stuff_hi": "Our Stuff+",
          "loc_hi": "Our Location+", "adjres_hi": "Our adj. results",
          "pitch_hi": "Our Pitching+"}
BLURB = {
    "velo_hi": "Average four-seam velocity, nothing else",
    "coach_hi": "His scorecard: velo, break, release",
    "stuff_hi": "The physical pitch, 12 measurements",
    "loc_hi": "Where the four-seam actually went",
    "adjres_hi": "What happened, adjusted for hitter and league",
    "pitch_hi": "All three of ours blended equally",
}
# TWO DIFFERENT QUESTIONS, and mixing them on one axis produced a garbage number once
# (adjusted results came out at -0.22 runs/9, i.e. "negative skill", which is a
# collinearity artifact and not a finding).
#
# QUALITY scores are built only from the physics of the pitch. For these, controlling
# for the prior season's line is the right test and the project standard: "what does
# this measurement add beyond the box score?" (RESULTS.md, portal section).
#
# INFORMATION scores already CONTAIN the prior season's results. Controlling for RA9
# then asks what last year's runs add beyond last year's runs, and the answer is a
# suppression artifact. These are scored WITHOUT the RA9 control, so their number
# answers a different question -- "how well does this rank pitchers for next season
# at all?" -- and must never be printed on the same axis as the quality scores.
QUALITY_SCORES = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi"]
INFO_SCORES = ["adjres_hi", "pitch_hi"]
CONTROLS_QUALITY = ["ra9_graded", "k_pct_graded", "bb_pct_graded"]
CONTROLS_INFO = ["k_pct_graded", "bb_pct_graded"]
CONTROLS = ["ra9_graded", "k_pct_graded", "bb_pct_graded"]
# The all-pitch cache for the 2025/2026 build. Passed in rather than hardcoded, and
# deliberately NOT defaulted to STUFFPLUS_WORKDIR_CRIT: the two are usually the same build
# but not always, and silently reading a different one would change the criterion without
# changing the output's label.
SEASON_CACHE_HELP = ("parquet holding the 2025/2026 all-pitch cache, e.g. "
                     "<workdir>_crit/pitches_cache_2025_2026_D1.parquet")


def deserved_runs(pool: pd.DataFrame, season_cache: str) -> pd.DataFrame:
    """Attach deserved (luck/defense-stripped) runs per 9 for both seasons.

    Uses ALL of a pitcher's pitches, not just four-seams: the criterion should be his
    whole season, the same way RA9 is. adjT is rebuilt here from the 2025/2026 cache
    because that is the build holding real 2026 (year role 2025).
    """
    if not os.path.exists(season_cache):
        sys.exit("no 2025/2026 all-pitch cache at %s. Pass --season-cache (%s)."
                 % (season_cache, SEASON_CACHE_HELP))
    df = pd.read_parquet(season_cache, columns=["PitcherId", "year", "Target", "ExitSpeed",
                                                "Angle", "PitchCall", "TaggedHitType",
                                                "League", "Batter", "BatterTeam"])
    df["is_inplay"] = df["PitchCall"] == "InPlay"
    fc.add_xt(df)
    fc.add_adjusted(df)
    g = df.groupby(["PitcherId", "year"])["adjT"].agg(["sum", "size"]).unstack()
    # year role 2024 = real 2025 (graded), 2025 = real 2026 (follow-up)
    out = pd.DataFrame({
        "xruns_graded": g[("sum", 2024)], "npitch_graded": g[("size", 2024)],
        "xruns_next": g[("sum", 2025)], "npitch_next": g[("size", 2025)]})
    j = pool.join(out, how="inner")
    j["xra9_graded"] = j["xruns_graded"] / j["ip_graded"] * 9
    j["xra9_next"] = j["xruns_next"] / j["ip_next"] * 9
    return j.dropna(subset=["xra9_graded", "xra9_next"])


def component_scores(args, pool: pd.DataFrame) -> pd.DataFrame:
    """Add our three component scores plus the Pitching+ blend to the pool.

    All four are four-seam scores on the graded season, matching the Staff Board so the
    page and the app cannot disagree. The location map is trained on the EARLIER season
    (year role 2024 = real 2024 in this pair), so Location+ is out-of-sample on 2025.

    Everything arrives in the lower-is-better run frame and is flipped ONCE here into
    the higher-is-better `_hi` display frame, consistent with the rest of this module.
    """
    df = fc.load_pitches(args)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)
    ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
    ff["loc"] = fc.PooledLocationMap(train).apply(ff)

    season = ff[ff["year"] == 2025]
    g = season.groupby("PitcherId").agg(n_ff=("ridge_pred", "size"),
                                        ridge=("ridge_pred", "mean"),
                                        loc=("loc", "mean"), adj=("adjT", "mean"))
    g = g[g["n_ff"] >= 100]
    g["stuff_hi"] = -g["ridge"]
    g["loc_hi"] = -g["loc"]
    g["adjres_hi"] = -g["adj"]
    # Pitching+ is the equal-weight z blend (RESULTS.md: fitted weights overfit three
    # separate times; equal weight is settled). z'd over this pool, not the D1
    # population, because only relative ordering enters the regression.
    g["pitch_hi"] = sum(cp.z(g[c].values) for c in ["stuff_hi", "loc_hi", "adjres_hi"]) / 3
    keep = ["stuff_hi", "loc_hi", "adjres_hi", "pitch_hi"]
    j = pool.drop(columns=[c for c in keep if c in pool.columns]).join(g[keep], how="inner")
    print(f"  component scores joined: {len(j)} of {len(pool)} pool pitchers")
    return j


def fit(s: pd.DataFrame, score: str, outcome: str, controls: list[str]) -> float:
    X = np.column_stack([np.ones(len(s))] + [s[c].values for c in controls]
                        + [cp.z(s[score].values)])
    beta, *_ = np.linalg.lstsq(X, s[outcome].values, rcond=None)
    return -beta[-1]


def panel(f: pd.DataFrame, outcome: str, rng, scores: list[str], controls: list[str]) -> dict:
    idx = f.index.values
    draws = {sc: [] for sc in scores}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for sc in scores:
            draws[sc].append(fit(s, sc, outcome, controls))
    d = {k: np.array(v) for k, v in draws.items()}
    bars = [dict(key=sc, label=LABELS[sc], blurb=BLURB[sc],
                 effect=round(float(d[sc].mean()), 3), se=round(float(d[sc].std()), 3),
                 lo=round(float(np.percentile(d[sc], 2.5)), 3),
                 hi=round(float(np.percentile(d[sc], 97.5)), 3)) for sc in scores]
    diffs = []
    for i, a in enumerate(scores):
        for b in scores[i + 1:]:
            dd = d[a] - d[b]
            diffs.append(dict(a=LABELS[a], b=LABELS[b], mean=round(float(dd.mean()), 3),
                              se=round(float(dd.std()), 3),
                              lo=round(float(np.percentile(dd, 2.5)), 3),
                              hi=round(float(np.percentile(dd, 97.5)), 3),
                              p=round(float((dd > 0).mean()), 3)))
    return dict(bars=bars, diffs=diffs)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--season-cache", required=True, help=SEASON_CACHE_HELP)
    known, _ = ap.parse_known_args()

    args = fc.paths()
    pool = pd.read_parquet(os.path.join(args.workdir, "coach_compare_pool.parquet"))
    print(f"pool: {len(pool)}")
    pool = component_scores(args, pool)
    f = deserved_runs(pool, known.season_cache)
    print(f"with deserved-runs criterion: {len(f)}")
    print(f"  graded season: deserved {f.xra9_graded.mean():.2f} vs actual "
          f"{f.ra9_graded.mean():.2f} runs/9")
    print(f"  next season:   deserved {f.xra9_next.mean():.2f} vs actual "
          f"{f.ra9_next.mean():.2f} runs/9")
    print(f"  criterion spread (SD): deserved {f.xra9_next.std():.2f}, "
          f"actual {f.ra9_next.std():.2f}")

    rng = np.random.default_rng(23)
    out = {}
    # ACTUAL first: it is the criterion that counts (what a coach lives with) and the
    # only one not built from adjT. DESERVED follows as the diagnostic that explains
    # why the two disagree.
    for name, outcome in [("actual", "ra9_next"), ("deserved", "xra9_next")]:
        out[name] = {}
        for group, scores, controls, question in [
            ("quality", QUALITY_SCORES, CONTROLS_QUALITY,
             "beyond this year's runs, K% and BB%, what does the measurement add?"),
            ("information", INFO_SCORES, CONTROLS_INFO,
             "how well does it rank pitchers for next season (K%/BB% held, RA9 NOT "
             "held -- these scores already contain it)?"),
        ]:
            res = panel(f, outcome, rng, scores, controls)
            res["question"] = question
            res["controls"] = controls
            out[name][group] = res
            print(f"\n{name.upper()} runs/9 next season -- {group.upper()}: {question}")
            for b in res["bars"]:
                print(f"  {b['label']:<18}{b['effect']:+.2f}  SE {b['se']:.2f}  "
                      f"CI [{b['lo']:+.2f},{b['hi']:+.2f}]")
            if len(res["diffs"]) > 1 or group == "quality":
                print("  paired differences (same resamples):")
                for dd in res["diffs"]:
                    verdict = ("TIE (<1 SE)" if abs(dd["mean"]) < dd["se"]
                               else f"P={dd['p']:.3f}")
                    print(f"    {dd['a']} - {dd['b']:<18} {dd['mean']:+.3f} "
                          f"SE {dd['se']:.3f}  {verdict}")

    # How many pitcher-seasons would resolve coach-vs-ours on the criterion that counts?
    # SE scales as 1/sqrt(n): n_needed = n * (SE / target)^2 for a 2-SE separation.
    # Both pairs pooled is about 940 (this pair 543 + the 2024->25 pair's 394), so the
    # verdict text is computed against that, not asserted.
    POOLED_N = 940
    key = next(d for d in out["actual"]["quality"]["diffs"]
               if d["a"] == LABELS["coach_hi"] and d["b"] == LABELS["stuff_hi"])
    target = abs(key["mean"]) / 2
    n_needed = int(round(len(f) * (key["se"] / target) ** 2)) if target > 1e-9 else None
    power = dict(pair=f"{key['a']} vs {key['b']}", observed_diff=key["mean"],
                 se=key["se"], n_now=len(f), n_for_2se=n_needed, pooled_n=POOLED_N,
                 resolvable=bool(n_needed and n_needed <= POOLED_N))
    print(f"\npower: separating {power['pair']} on ACTUAL runs")
    print(f"  observed {key['mean']:+.3f}, SE {key['se']:.3f} at n={len(f)}")
    if n_needed:
        print(f"  a 2-SE separation would need about {n_needed:,} pitcher-seasons; "
              f"both year pairs pooled is about {POOLED_N:,}")
        print("  => " + ("REACHABLE by pooling both pairs"
                         if power["resolvable"] else
                         "NOT reachable with the data that exists"))

    payload = dict(n=len(f), season_graded=2025, season_next=2026,
                   panels=out, power=power,
                   cross={f"{a}|{b}": round(float(fc.R(f[a], f[b])), 3)
                          for i, a in enumerate(QUALITY_SCORES + INFO_SCORES)
                          for b in (QUALITY_SCORES + INFO_SCORES)[i + 1:]},
                   control_note=("adjusted results and Pitching+ contain prior-season "
                                 "results, so they are scored without the RA9 control "
                                 "and their numbers answer a different question. Do not "
                                 "plot them on the same axis as the quality scores."))
    dest = os.path.join(args.workdir, "coach_two_panel.json")
    with open(dest, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
