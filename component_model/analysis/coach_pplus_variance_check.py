"""Is the variance decomposition behind the Pitching+ weights actually in bounds?

WHY THIS EXISTS. coach_pitching_plus_weights.py returned persistence > 1 for sinker (1.322),
curveball (1.262) and slider (1.153). persistence is s2_stable / (s2_stable + s2_drift), a
proportion, so a value above 1 means s2_drift came out NEGATIVE. variance_components returns
drift raw and its docstring allows a slightly negative value as honest sampling error near
zero, but -24% of the denominator is not slight, and it is not cosmetic:

    s2_drift  = total_observed - s2_stable - s2_noise
    signal_var = total_observed - s2_noise          (= s2_stable + s2_drift by definition)

so negative drift means signal_var < s2_stable. reliability_curves.optimal_blend puts
signal_var on Sigma's diagonal and cov(a_i, a_y) ~ s2_stable in the numerator, so the results
component gets a diagonal SMALLER than its own numerator and its weight is biased UP at high
pitch count. That is the opposite of conservative, and it lands on the three types where
results already dominate.

THE HYPOTHESIS: the per-season floor is too low. s2_noise = noise_per_pitch * mean(1/n), and
mean(1/n) is dominated by the smallest qualifying seasons -- at a 40-pitch floor a single
40-pitch season contributes 1/40 while a 400-pitch season contributes 1/400. Overstate
s2_noise and drift absorbs the error as a negative number. 13_reliability_vs_sample_size.py
uses PANEL_MIN_FF = 100 for four-seams; this script asks what the equivalent floor is per type.

THE DESIGN THAT MAKES THIS CHEAP AND CLEAN: noise_per_pitch is a PER-PITCH scale, so it should
not move materially with the floor. It is therefore estimated ONCE per type at the lowest floor
and reused across every floor, which both saves the expensive game-split loop and isolates the
variable actually under test -- if the decomposition comes back into bounds at a higher floor,
that is the floor doing it and not a re-estimated noise term moving underneath.

Reports, per type and floor: the three variance pieces, total observed, persistence, and
whether signal_var >= s2_stable (the invariant that must hold). Decides nothing on its own;
the floor choice goes back into coach_pitching_plus_weights.py.

Data rules: reads the pitch-cache parquets only; prints only. No pitcher names, no per-pitcher
rows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import reliability_curves as rc
import variance_components as vc

SEASONS = [2024, 2025, 2026]
ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
FLOORS = [40, 75, 100, 150, 250]
COLS = [("ridge_pred", "Stuff+"), ("adjT", "Recent results")]


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"))
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--min-half", type=int, default=15)
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()
    if not a.caches:
        sys.exit("Set --caches or STUFFPLUS_CACHES (both pitch-cache parquets).")
    a.caches = [p.strip() for p in a.caches.split(",") if p.strip()]
    return a


def load_seasons(paths):
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    return df


def panel(ff, min_n):
    n_by = ff.groupby(["PitcherId", "year"]).size().rename("n")
    ok = n_by[n_by >= min_n].reset_index()[["PitcherId", "year"]]
    p = ff.merge(ok, on=["PitcherId", "year"], how="inner")
    if p.empty:
        return p
    per = p.groupby("PitcherId")["year"].nunique()
    return p[p["PitcherId"].isin(per[per >= 2].index)].copy()


def season_table(p, cols):
    g = p.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({"n": g["adjT"].size(),
                        **{c: g[c].mean() for c in cols}}).reset_index()
    tab = tab.rename(columns={"PitcherId": "pitcher", "year": "season"})
    for c in cols:
        tab[c] = tab[c] - tab.groupby("season")[c].transform("mean")
    return tab


def noise_scale(p, col, splits, min_half, rng):
    vals = []
    for _ in range(splits):
        s = rc.random_game_half(p, rng)
        tmp = s[[col, "half"]].copy()
        tmp["ps"] = (s["PitcherId"].astype(str) + "|" + s["year"].astype(str)).values
        eff, _ = vc.effective_noise_scale(tmp, col, "ps", min_half=min_half)
        if not np.isnan(eff):
            vals.append(eff)
    return float(np.mean(vals)) if vals else float("nan")


def main() -> int:
    t0 = time.time()
    a = cli()
    rng = np.random.default_rng(a.seed)
    df = load_seasons(a.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)
    print("  frame ready in %.0fs" % (time.time() - t0))

    for grp in ORDER:
        ff = fc.stuff_ridge(df, pitch_mask=fc.pitch_mask(df, grp), feats=fc.feats_for(grp))
        ff = ff[ff["xT"].notna()].copy()
        base = panel(ff, min(FLOORS))
        if base.empty:
            print("")
            print("=== %s: empty at the lowest floor" % grp)
            continue
        # Per-pitch scale, estimated once at the lowest floor and held fixed across floors --
        # see the module docstring for why that is the point rather than a shortcut.
        npp = {c: noise_scale(base, c, a.splits, a.min_half, rng) for c, _ in COLS}
        print("")
        print("=== %s   noise/pitch  %s" % (grp, "  ".join(
            "%s=%.6f" % (lab, npp[c]) for c, lab in COLS)))
        print("    %-16s%6s%7s%12s%12s%12s%12s%8s%7s"
              % ("component", "floor", "arms", "total_obs", "s2_stable", "s2_drift",
                 "s2_noise", "persis", "ok"))
        for c, lab in COLS:
            for f in FLOORS:
                p = panel(ff, f)
                if p.empty or p["PitcherId"].nunique() < vc.MIN_PAIR_N:
                    print("    %-16s%6d   thin" % (lab, f))
                    continue
                tab = season_table(p, [c for c, _ in COLS])
                try:
                    d = vc.variance_components(
                        tab[["pitcher", "season", c, "n"]].rename(columns={c: "mean"}),
                        npp[c])
                except ValueError as e:
                    print("    %-16s%6d   %s" % (lab, f, e))
                    continue
                sig = rc.signal_variance(tab, c, npp[c])
                # The invariant: signal_var is stable+drift, so it can never sit below the
                # stable part alone. This is the check the whole script exists to run.
                ok = "yes" if sig >= d["s2_stable"] else "NO"
                print("    %-16s%6d%7d%12.6f%12.6f%12.6f%12.6f%8.3f%7s"
                      % (lab, f, d["n_pitchers"], d["total_observed"], d["s2_stable"],
                         d["s2_drift"], d["s2_noise"], d["persistence"], ok))
    print("")
    print("  total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
