"""16: How many pitches of each type before a season Stuff+ read is trustworthy?

The pitcher page flags a pitch type as "small sample" below SAMPLE_FLOOR = 100
pitches (14_pitcher_pages.py). That number was measured by script 06 for a
DIFFERENT score: four-seam Location+, whose per-pitch values are run-value cell
means and therefore noisy. Stuff+ is a ridge prediction from physical
measurements that barely vary within a pitcher, so it should converge far
faster, and the same borrowed 100 is applied to every secondary type without
ever having been measured for any of them.

This script measures, per pitch type and per score, how much of a full-season
reading a sample of n pitches already carries, and where a small-sample flag
stops being informative.

Method (one estimator, no scan-and-pick):
  * Grade every pitch of the graded season with the SAME model the page uses
    (fc.stuff_ridge on the type's validated feature list, trained on the earlier
    season). Four-seam Location+ is graded with the pooled location map, as
    script 06 did, so its floor can be re-read on the same footing.
  * Split each pitcher's season into two halves by WHOLE GAME (alternating
    games in date order), never by pitch: pitches in one game share batter,
    park, umpire and day, and a pitch-parity split leaves that variance in both
    halves and reads high (see variance_components.effective_noise_scale).
  * Noise scale sigma2_eff from the half difference; true-score variance tau2 =
    var(season means) - mean(sigma2_eff / N). Reliability of an n-pitch mean is
    rel(n) = tau2 / (tau2 + sigma2_eff / n) = n / (n + n0), n0 = sigma2_eff/tau2.
  * "Share of a season's signal" at n = rel(n) / rel(N_typ), N_typ the median
    season count among pitchers in the panel. The floor reported is the
    smallest n at which that share reaches 75% (and, for reference, where
    absolute reliability reaches 0.5 and 0.75).
  * Empirical check, not a second estimator: correlation of the grade on the
    FIRST n pitches (date order, what a coach sees mid-season) with the grade on
    the rest of the season, against what the n0 curve predicts.

Data rules: reads the source CSV / workdir cache only, writes one aggregate JSON
to the workdir. No pitcher names, no per-pitcher output, no absolute paths.
Never commit anything under the workdir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc  # noqa: E402
import variance_components as vc  # noqa: E402

SEASON_ROLE_YEAR = 2025          # graded season after fair_criterion's role relabel
GROUPS = ["FF", "SI", "FC", "SL", "CB", "CH"]
N_GRID = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
CHECK_N = [10, 25, 50]
SHARE_TARGET = 0.75


def extra_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-half", type=int, default=20,
                    help="Minimum pitches in EACH game-split half for a pitcher to enter "
                         "the noise-scale panel.")
    args, _ = ap.parse_known_args()
    return args


def assign_halves(season: pd.DataFrame) -> pd.DataFrame:
    """Alternate a pitcher's games (date order) into two halves; never split a game."""
    s = season.sort_values(["PitcherId", "Date", "GameID"]).copy()
    game_rank = (s.drop_duplicates(["PitcherId", "GameID"])
                  .groupby("PitcherId").cumcount())
    game_rank.index = pd.MultiIndex.from_frame(
        s.drop_duplicates(["PitcherId", "GameID"])[["PitcherId", "GameID"]])
    key = pd.MultiIndex.from_frame(s[["PitcherId", "GameID"]])
    s["half"] = (game_rank.reindex(key).values % 2).astype(int)
    return s


def reliability_curve(season: pd.DataFrame, value_col: str, min_half: int) -> dict:
    s = assign_halves(season)
    sigma2_eff, n_panel = vc.effective_noise_scale(s, value_col, "PitcherId", "half", min_half)
    halves = (s.groupby(["PitcherId", "half"])[value_col].agg(["mean", "size"])
                .unstack("half").dropna())
    halves.columns = ["mA", "mB", "nA", "nB"]
    halves = halves[(halves["nA"] >= min_half) & (halves["nB"] >= min_half)]
    r_half = float(np.corrcoef(halves["mA"], halves["mB"])[0, 1]) if len(halves) > 2 else float("nan")

    per = s.groupby("PitcherId")[value_col].agg(["mean", "size"])
    per = per.loc[halves.index]
    tau2 = float(per["mean"].var(ddof=1) - (sigma2_eff / per["size"]).mean())
    n0 = sigma2_eff / tau2 if tau2 > 0 else float("inf")
    n_typ = float(per["size"].median())

    def rel(n):
        return n / (n + n0)

    rel_typ = rel(n_typ)
    # Smallest n with rel(n) / rel(N_typ) >= SHARE_TARGET, solved in closed form.
    target = SHARE_TARGET * rel_typ
    n_share = int(np.ceil(target * n0 / (1 - target))) if target < 1 else None

    checks = {}
    for n in CHECK_N:
        eligible = per.index[per["size"] >= n + 2 * min_half]
        sub = s[s["PitcherId"].isin(eligible)]
        first = sub.groupby("PitcherId").head(n).groupby("PitcherId")[value_col].mean()
        rest = sub.groupby("PitcherId")[value_col].apply(lambda v: v.iloc[n:].mean())
        if len(first) > 10:
            obs = float(np.corrcoef(first, rest.loc[first.index])[0, 1])
            rest_n = float((per.loc[first.index, "size"] - n).mean())
            pred = float(np.sqrt(rel(n) * rel(rest_n)))
            checks[n] = {"pitchers": int(len(first)), "observed_r": round(obs, 3),
                         "predicted_r": round(pred, 3)}

    return {
        "pitchers": int(n_panel),
        "half_r": round(r_half, 3),
        "sigma2_eff": sigma2_eff,
        "tau2": tau2,
        "n0": round(float(n0), 1),
        "N_typical": n_typ,
        "rel_at_N_typical": round(rel_typ, 3),
        "rel_by_n": {n: round(rel(n), 3) for n in N_GRID},
        "share_by_n": {n: round(rel(n) / rel_typ, 3) for n in N_GRID},
        "n_for_75pct_of_season_signal": n_share,
        "n_for_rel_0.5": int(np.ceil(n0)),
        "n_for_rel_0.75": int(np.ceil(3 * n0)),
        "first_n_check": checks,
    }


def stuff_frame(df: pd.DataFrame, group: str) -> pd.DataFrame:
    mask = df["is_ff"] if group == "FF" else fc.pitch_mask(df, group)
    graded = fc.stuff_ridge(df, pitch_mask=mask, feats=fc.feats_for(group))
    season = graded[graded["year"] == SEASON_ROLE_YEAR]
    return season[["PitcherId", "GameID", "Date", "ridge_pred"]]


def location_frame(df: pd.DataFrame) -> pd.DataFrame:
    ff = df[df["is_ff"] & df["PlateLocSide"].notna() & df["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
    ff["loc"] = fc.PooledLocationMap(train).apply(ff)
    season = ff[ff["year"] == SEASON_ROLE_YEAR]
    return season[["PitcherId", "GameID", "Date", "loc"]]


def print_block(label: str, res: dict) -> None:
    print(f"\n{label}: panel {res['pitchers']} pitchers, game-split half r {res['half_r']}, "
          f"n0 {res['n0']}, typical season N {res['N_typical']:.0f} "
          f"(rel {res['rel_at_N_typical']})")
    print("   n:      " + "".join(f"{n:>7}" for n in N_GRID))
    print("   rel:    " + "".join(f"{res['rel_by_n'][n]:>7.2f}" for n in N_GRID))
    print("   share:  " + "".join(f"{res['share_by_n'][n]:>7.2f}" for n in N_GRID))
    print(f"   -> 75% of season signal at n = {res['n_for_75pct_of_season_signal']}; "
          f"rel 0.5 at {res['n_for_rel_0.5']}, rel 0.75 at {res['n_for_rel_0.75']}")
    for n, c in res["first_n_check"].items():
        print(f"   check first {n:>3}: observed r {c['observed_r']:.2f} vs predicted "
              f"{c['predicted_r']:.2f} ({c['pitchers']} pitchers)")


def main() -> int:
    args = fc.paths()
    xargs = extra_cli()
    df = fc.load_pitches(args)
    fc.add_xt(df)
    if "GameID" not in df.columns:
        sys.exit("GameID missing from the cache; rebuild it before running this script")

    out = {"season_role_year": SEASON_ROLE_YEAR, "level": args.level, "years": args.years,
           "min_half": xargs.min_half, "share_target": SHARE_TARGET, "stuff": {}, "location": {}}
    print("=" * 78)
    print("SEASON SAMPLE FLOOR BY PITCH TYPE (game-split reliability, graded season)")
    print("=" * 78)
    for g in GROUPS:
        try:
            season = stuff_frame(df, g)
            res = reliability_curve(season, "ridge_pred", xargs.min_half)
        except Exception as err:  # noqa: BLE001 -- report and keep going per type
            print(f"\nStuff+ {g}: skipped ({err})")
            continue
        out["stuff"][g] = res
        print_block(f"Stuff+ {g}", res)

    res = reliability_curve(location_frame(df), "loc", xargs.min_half)
    out["location"]["FF"] = res
    print_block("Location+ FF (pooled map, script 06 footing)", res)

    path = os.path.join(args.workdir, "season_floor.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {os.path.basename(path)} to the workdir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
