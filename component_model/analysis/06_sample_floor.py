"""06: How many four-seamers before a Location+ read is trustworthy?

Split-half (even/odd pitch) reliability within each year, by FF-count bin,
Spearman-Brown corrected to the full sample; year-over-year r by bin as the
conservative check (it adds true season-to-season change). Flags the requested
team's current-season arms below the floor. Names print to stdout only --
never commit or redistribute per-pitcher output from this script.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import fair_criterion as fc

BINS = [(30, 60), (60, 100), (100, 150), (150, 250), (250, 400), (400, 10000)]
M_SHRINK = 5

args = fc.paths()
ff = fc.ff_panel(args)
ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
fc.add_loc_bins(ff)
fc.add_count_cols(ff)
train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
cmap = fc.CountLocationMap(train, "count12", M_SHRINK)
ff["loc"] = cmap.apply(ff)

def split_half(year):
    sub = ff[ff["year"] == year].copy()
    sub["half"] = sub.groupby("PitcherId").cumcount() % 2
    piv = sub.groupby(["PitcherId", "half"])["loc"].agg(["mean", "size"]).unstack()
    piv.columns = ["mA", "mB", "nA", "nB"]
    piv["n"] = piv["nA"] + piv["nB"]
    return piv.dropna()

n0s = []
print("SPLIT-HALF RELIABILITY vs FF sample (SB = Spearman-Brown full-sample estimate):")
for year in [2024, 2025]:
    piv = split_half(year)
    print(f"\n{year}:  bin        pitchers  mean_FF  half_r   SB   implied n0")
    for lo, hi in BINS:
        m = piv[(piv["n"] >= lo) & (piv["n"] < hi)]
        if len(m) < 25:
            continue
        r = pearsonr(m["mA"], m["mB"])[0]
        n0 = (m["n"].mean() / 2) * (1 - r) / r if r > 0 else np.nan
        print(f"  {f'{lo}-{hi if hi < 10000 else chr(43)}':<12}{len(m):>6}{m['n'].mean():>9.0f}"
              f"{r:>8.3f}{2 * r / (1 + r):>7.3f}{n0:>9.0f}")
        n0s += [n0] * len(m)
n0 = float(np.median(n0s))
print(f"\nmedian n0 = {n0:.0f} FF (n0 = pitches for reliability 0.5; grows with sample, so"
      f" treat the curve as optimistic at high n)")
for n in [30, 50, 100, 200, 400]:
    print(f"  n={n:<4} implied reliability {n / (n + n0):.2f}")

mrg = ff.groupby(["PitcherId", "year"])["loc"].agg(["mean", "size"]).unstack()
mrg.columns = ["m24", "m25", "n24", "n25"]
mrg = mrg.dropna()
mrg["nmin"] = mrg[["n24", "n25"]].min(axis=1)
print("\nYEAR-OVER-YEAR r by min(FF24, FF25) bin (includes true skill change):")
for lo, hi in BINS:
    m = mrg[(mrg["nmin"] >= lo) & (mrg["nmin"] < hi)]
    if len(m) < 25:
        continue
    print(f"  {f'{lo}-{hi if hi < 10000 else chr(43)}':<10} n={len(m):<5} r={pearsonr(m['m24'], m['m25'])[0]:.3f}")

print(f"\nRECOMMENDED FLOOR: full read at 100+ FF, caution 50-99, flag below ~{n0:.0f}.")
d = ff[(ff["year"] == 2025) & (ff["PitcherTeam"] == args.team)]
st = d.groupby("PitcherId").agg(name=("Pitcher", "first"), n_ff=("loc", "size")).query("n_ff>=30")
print(f"\n{args.team} 2025 arms (30+ FF):")
for _, r_ in st.sort_values("n_ff").iterrows():
    n = int(r_["n_ff"])
    tag = "FLAG: small sample" if n < n0 else ("caution" if n < 100 else "ok")
    print(f"  {r_['name']:<24} FF={n:>4}  implied rel={n / (n + n0):.2f}  {tag}")
