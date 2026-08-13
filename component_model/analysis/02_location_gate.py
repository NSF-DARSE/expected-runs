"""02: Reliability gate for location signals.

Before modeling location we require that location traits repeat year over year
better than run-value noise (~0.18) and ideally at whiff level (~0.35) or above.
The locRV descriptor here pools 2024+2025 for map construction -- acceptable for
a reliability gate only, never for prediction (script 03 trains on 2024 only).
"""
import numpy as np
import pandas as pd

import fair_criterion as fc

args = fc.paths()
ff = fc.ff_panel(args)
ff = ff[ff["in_panel"] & ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
fc.add_loc_bins(ff)

x, z = ff["PlateLocSide"], ff["PlateLocHeight"]
heart = (x.abs() <= 0.558) & (z >= 1.83) & (z <= 3.17)
shadow = (x.abs() <= 1.108) & (z >= 1.17) & (z <= 3.83) & ~heart
chase = (x.abs() <= 1.658) & (z >= 0.5) & (z <= 4.5) & ~heart & ~shadow
ff["in_zone"] = ((x.abs() <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float)
ff["heart"] = heart.astype(float)
ff["shadow"] = shadow.astype(float)
ff["chase"] = chase.astype(float)
ff["waste"] = (~(heart | shadow | chase)).astype(float)
ff["absx"] = x.abs()

ff["locRV"] = fc.PooledLocationMap(fc.add_loc_bins(ff[ff["xT"].notna()].copy())).apply(ff)

g = ff.groupby(["PitcherId", "year"])
tab = pd.DataFrame({
    "zone_rate": g["in_zone"].mean(), "heart_rate": g["heart"].mean(),
    "shadow_rate": g["shadow"].mean(), "chase_rate": g["chase"].mean(),
    "waste_rate": g["waste"].mean(), "mean_x": g["PlateLocSide"].mean(),
    "mean_absx": g["absx"].mean(), "mean_z": g["PlateLocHeight"].mean(),
    "sd_x": g["PlateLocSide"].std(), "sd_z": g["PlateLocHeight"].std(),
    "locRV": g["locRV"].mean(),
}).reset_index()
ids = sorted(set(tab[tab["year"] == 2024]["PitcherId"]) & set(tab[tab["year"] == 2025]["PitcherId"]))
a, b = fc.year_split(tab, ids)

print(f"LOCATION TRAIT RELIABILITY (2024 vs 2025, n={len(ids)})")
print("ladder: physical stuff 0.84-0.97 | whiff 0.35 | raw run value 0.18")
for c in tab.columns.drop(["PitcherId", "year"]):
    print(f"  {c:<12} r = {fc.R(a[c], b[c]):.3f}")
