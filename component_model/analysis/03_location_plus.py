"""03: Location+ (pooled map, 2024-trained) and the three-component ablation.

Location+ = mean 2024-map value over a pitcher's FF, applied out-of-sample to 2025.
Ablation vs the fixed fair criterion C2_25: last-year adjusted results alone, then
+Location+, then +Stuff+, as equal-weight z blends (equal weights beat split-half
OLS-fitted weights on this data; see RESULTS.md). Paired bootstrap on increments.
"""
import numpy as np
import pandas as pd

import fair_criterion as fc

args = fc.paths()
ff = fc.ff_panel(args)
ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
fc.add_loc_bins(ff)

lmap = fc.PooledLocationMap(ff[(ff["year"] == 2024) & ff["xT"].notna()])
ff["loc"] = lmap.apply(ff)

panel = ff[ff["in_panel"]]
ids = panel["PitcherId"].unique()
g = panel.groupby(["PitcherId", "year"])
tab = pd.DataFrame({"C2": g["adjT"].mean(), "ridge": g["ridge_pred"].mean(),
                    "loc": g["loc"].mean()}).reset_index()
a, b = fc.year_split(tab, sorted(ids))
N = len(a)

print(f"SCORES (n={N}):  reliability | validity vs C2_25 (Pearson/Spearman)")
for c in ["C2", "ridge", "loc"]:
    print(f"  {c:<6} rel={fc.R(a[c], b[c]):.3f}  val={fc.R(a[c], b['C2']):.3f}/{fc.RS(a[c], b['C2']):.3f}")
print("2024 cross-corrs: ridge-loc=%.3f  ridge-C2=%.3f  loc-C2=%.3f"
      % (fc.R(a["ridge"], a["loc"]), fc.R(a["ridge"], a["C2"]), fc.R(a["loc"], a["C2"])))

blends = [("results alone", fc.z(a["C2"])),
          ("+ Location+ (equal z)", 0.5 * fc.z(a["C2"]) + 0.5 * fc.z(a["loc"])),
          ("+ Stuff+ (equal 1/3)", (fc.z(a["C2"]) + fc.z(a["loc"]) + fc.z(a["ridge"])) / 3)]
print("\nABLATION vs C2_25:")
for name, p in blends:
    print(f"  {name:<24} P={fc.R(p, b['C2']):.3f}  S={fc.RS(p, b['C2']):.3f}")

rng = np.random.default_rng(42)
M = np.column_stack([a["C2"], a["loc"], a["ridge"], b["C2"]])
d_loc, d_stuff = [], []
for _ in range(4000):
    c2, loc, rid, y = M[rng.choice(N, N, replace=True)].T
    def zz(v): return (v - v.mean()) / v.std()
    r1 = np.corrcoef(zz(c2), y)[0, 1]
    r2 = np.corrcoef(0.5 * zz(c2) + 0.5 * zz(loc), y)[0, 1]
    r3 = np.corrcoef((zz(c2) + zz(loc) + zz(rid)) / 3, y)[0, 1]
    d_loc.append(r2 - r1)
    d_stuff.append(r3 - r2)
print("\nPAIRED BOOTSTRAP (4000 reps over pitchers):")
fc.boot_report("Location+ over results", d_loc)
fc.boot_report("Stuff+ over results+Location+", d_stuff)
