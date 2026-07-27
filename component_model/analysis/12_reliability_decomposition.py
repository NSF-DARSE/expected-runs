"""12: How much of year-over-year unpredictability is noise vs missing skill?

Splits the spread in pitcher-season fair-criterion performance three ways:
  bucket 1  measurement noise  -- a college season is too few pitches
  bucket 2  true drift         -- pitchers really change year to year
  bucket 3  missing skill      -- persistent talent Pitching+ does not encode
Only bucket 3 is a Pitching+ problem. Buckets 1 and 2 set the ceiling that no
static physical model can beat.

Part A  within-season split-half reliability of xT, split by GAME parity and
        Spearman-Brown corrected. Separates bucket 1 from bucket 2, which the
        across-season number alone cannot do.
Part B  method-of-moments variance decomposition over the three-season panel,
        then the same fit with physical Stuff+/Location+ added; the stable
        variance that survives is missing skill.

Game parity, not pitch parity: consecutive pitches in one game share batter,
park, umpire, and day effects, so a pitch-parity split (script 06) leaves that
shared variance in BOTH halves and overstates reliability. Expect a lower and
more honest number here.

Criterion is xT (defense/luck-stripped expected runs, LOWER = BETTER); adjT
prints as a robustness row. Four-seam, D1, 2024-2026.

Input: pitch-cache parquets written by fc.load_pitches (--caches). Regenerate
them by running script 01 once per year pair with --level D1 if absent. Years
are re-derived from Date, so role-relabeled caches are safe to pass.
Writes nothing outside --workdir. Never commit output: Level II per-pitcher data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import variance_components as vc

SEASONS = [2024, 2025, 2026]
TRAIN_YEAR = 2024


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"),
                    help="Comma-separated pitch-cache parquet paths. Years are "
                         "re-derived from Date, so relabeled caches are fine.")
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-ff", type=int, default=fc.PANEL_MIN_FF)
    ap.add_argument("--min-half", type=int, default=25)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches (comma-separated pitch-cache parquets) and "
                 "--workdir (outside the repo), or STUFFPLUS_CACHES / "
                 "STUFFPLUS_WORKDIR.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


def load_seasons(paths):
    """Concat pitch caches -> deduped frame with TRUE calendar years.

    fc.load_pitches ROLE-RELABELS year for non-default pairs (a 2025-2026 cache
    stores year=2024 for real 2025), so year is recomputed from Date here. Dedup
    runs across the concatenation because each cache deduped only within itself.
    """
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        print(f"  read {os.path.basename(p)}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    before = len(df)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    print(f"  concat {before:,} -> deduped, in-scope {len(df):,} rows")
    print("  rows by TRUE year: %s" % df["year"].value_counts().sort_index().to_dict())
    if "Level" in df.columns:
        print("  levels present: %s" % df["Level"].value_counts().to_dict())
    return df


def add_game_parity(ff):
    """half = parity of the game's chronological index within pitcher-season.

    GameID begins yyyymmdd, so sorting it orders games in time. Alternating
    whole games keeps the halves balanced while making them independent of
    within-game shared effects.
    """
    keys = ff[["PitcherId", "year", "GameID"]].drop_duplicates()
    keys = keys.sort_values(["PitcherId", "year", "GameID"])
    keys["half"] = keys.groupby(["PitcherId", "year"]).cumcount() % 2
    return ff.merge(keys, on=["PitcherId", "year", "GameID"], how="left")


def build_panel(args):
    """Three-season qualifying-FF frame plus the pitcher-season table.

    xT is fit ONCE on all three seasons pooled, so a given EV/LA outcome maps to
    the same run value in every season -- required for cross-season comparability
    (the spec's frozen reference vintage). The Stuff+ ridge and the location map
    are trained on TRAIN_YEAR only, matching the suite's fixed references.
    """
    df = load_seasons(args.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)

    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    lmap = fc.PooledLocationMap(ff[(ff["year"] == TRAIN_YEAR) & ff["xT"].notna()])
    ff["loc"] = lmap.apply(ff)

    n_by = ff.groupby(["PitcherId", "year"]).size().rename("n_ff")
    ok = n_by[n_by >= args.min_ff].reset_index()[["PitcherId", "year"]]
    ff = ff.merge(ok, on=["PitcherId", "year"], how="inner")
    # A pitcher needs two qualified seasons for cross-season covariance to exist.
    seasons_per = ff.groupby("PitcherId")["year"].nunique()
    ff = ff[ff["PitcherId"].isin(seasons_per[seasons_per >= 2].index)].copy()
    ff = add_game_parity(ff)

    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({
        "mean": g["xT"].mean(),
        "mean_adjT": g["adjT"].mean(),
        "n": g["xT"].size(),
        "stuff": g["ridge_pred"].mean(),
        "loc": g["loc"].mean(),
    }).reset_index().rename(columns={"PitcherId": "pitcher", "year": "season"})

    print(f"\nPANEL: {tab['pitcher'].nunique()} pitchers, {len(tab)} pitcher-seasons "
          f"({args.min_ff}+ FF, 2+ qualified seasons, D1)")
    print("  qualified pitcher-seasons by year: %s"
          % tab["season"].value_counts().sort_index().to_dict())
    print("  median FF per pitcher-season: %.0f" % tab["n"].median())
    return ff, tab


def part_a(ff, args):
    """Within-season game-parity split-half reliability, Spearman-Brown corrected."""
    print("\n" + "=" * 78)
    print("PART A -- WITHIN-SEASON RELIABILITY (game parity, Spearman-Brown)")
    print("=" * 78)
    print("Measures bucket 1 alone: same pitcher, same season, different games.")
    print(f"{'season':<8}{'metric':<14}{'pitchers':>9}{'mean FF':>9}"
          f"{'half r':>9}{'SB full':>9}")
    out = {}
    for col, label in [("xT", "xT"), ("ridge_pred", "Stuff+ anchor")]:
        for year in SEASONS:
            sub = ff[ff["year"] == year]
            piv = sub.groupby(["PitcherId", "half"])[col].agg(["mean", "size"]).unstack("half")
            piv = piv.dropna()
            if piv.empty:
                continue
            piv.columns = ["mA", "mB", "nA", "nB"]
            piv = piv[(piv["nA"] >= args.min_half) & (piv["nB"] >= args.min_half)]
            if len(piv) < 30:
                print(f"{year:<8}{label:<14}{len(piv):>9}  too few pitchers, skipped")
                continue
            r = float(pearsonr(piv["mA"], piv["mB"])[0])
            sb = vc.spearman_brown(r)
            n_mean = float((piv["nA"] + piv["nB"]).mean())
            print(f"{year:<8}{label:<14}{len(piv):>9}{n_mean:>9.0f}{r:>9.3f}{sb:>9.3f}")
            if col == "xT":
                out[year] = {"rho_half": r, "rho_full": sb, "n": len(piv),
                             "mean_ff": n_mean}
    print("\nNote: script 06 splits by PITCH parity, which shares within-game")
    print("effects across both halves and reads higher. Game parity is the")
    print("honest unit for a noise estimate.")
    return out


if __name__ == "__main__":
    args = cli()
    ff, tab = build_panel(args)
    a_out = part_a(ff, args)
