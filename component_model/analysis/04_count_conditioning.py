"""04: Count-conditioned Location+ -- build, tune, and decompose.

Shrinkage prior m is tuned on a 2024-internal even/odd pitch holdout (never on
2025). Three count representations are compared; count12 (balls x strikes) wins.
The decomposition is the point of this script (see FRAMEWORK.md, "Score design
principles"):
  raw = E[xT | loc, count]          -> includes count-occupancy skill
  rel = raw - E[xT | count] + E[xT] -> location-given-count only
  mix = raw - rel                   -> pure count occupancy
Verdict on four-seams: raw beats pooled standalone (~2 SE) but the entire gain is
occupancy; rel ties pooled; the gain vanishes next to results in the blend.
Pitcher-level Location+ therefore stays pooled/rel; the count map is used for
pitch-level explanation only.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import fair_criterion as fc

MS = [1, 2, 5, 10, 25, 50, 100, 200, 400, 800]
SCHEMES = ["count12", "bucket4", "state5"]

args = fc.paths()
ff = fc.ff_panel(args)
ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
fc.add_loc_bins(ff)
fc.add_count_cols(ff)

# ---- tune m on 2024 even/odd holdout ----
t24 = ff[(ff["year"] == 2024) & ff["xT"].notna()].copy()
t24["half"] = t24.groupby("PitcherId").cumcount() % 2
A, B = t24[t24["half"] == 0], t24[t24["half"] == 1]
pooledA = fc.PooledLocationMap(A)
pooled_B = pooledA.apply(B)
print(f"HOLDOUT (train evens, test odds, n={len(B)}): pooled r={pearsonr(pooled_B, B['xT'])[0]:.4f} "
      f"mse={((B['xT'] - pooled_B) ** 2).mean():.6f}")
best = {}
for scheme in SCHEMES:
    scores = {}
    for m in MS:
        cm = fc.CountLocationMap(A, scheme, m)
        v = cm.apply(B, pooled_B)
        scores[m] = ((B["xT"] - v) ** 2).mean()
    best[scheme] = min(scores, key=scores.get)
    m = best[scheme]
    print(f"  {scheme:<8} best m={m:<4} holdout mse={scores[m]:.6f} "
          f"r={pearsonr(fc.CountLocationMap(A, scheme, m).apply(B, pooled_B), B['xT'])[0]:.4f}")

# ---- final scores: full-2024 maps ----
train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
pooled = fc.PooledLocationMap(train)
ff["loc_pooled"] = pooled.apply(ff)
cmap = fc.CountLocationMap(train, "count12", best["count12"])
ff["loc_raw"] = cmap.apply(ff, ff["loc_pooled"])
ff["loc_rel"] = cmap.apply_relative(ff, ff["loc_pooled"])

panel = ff[ff["in_panel"]]
g = panel.groupby(["PitcherId", "year"])
tab = pd.DataFrame({"C2": g["adjT"].mean(), "ridge": g["ridge_pred"].mean(),
                    "loc_pooled": g["loc_pooled"].mean(), "loc_raw": g["loc_raw"].mean(),
                    "loc_rel": g["loc_rel"].mean(),
                    "rate_2K": g["s"].agg(lambda s: (s == 2).mean()),
                    "rate_3B": g["b"].agg(lambda v: (v == 3).mean())}).reset_index()
tab["mix"] = tab["loc_raw"] - tab["loc_rel"]
tab.to_parquet(f"{args.workdir}/count_scores.parquet", index=False)
ids = sorted(panel["PitcherId"].unique())
a, b = fc.year_split(tab, ids)
N = len(ids)

print(f"\nMAIN TABLE (n={N}): reliability | validity vs C2_25 (P/S)")
for c in ["loc_pooled", "loc_raw", "loc_rel", "mix"]:
    print(f"  {c:<12} rel={fc.R(a[c], b[c]):.3f}  val={fc.R(a[c], b['C2']):.3f}/{fc.RS(a[c], b['C2']):.3f}")
print("mix corr with: C2=%.3f  ridge=%.3f  loc_rel=%.3f"
      % (fc.R(a["mix"], a["C2"]), fc.R(a["mix"], a["ridge"]), fc.R(a["mix"], a["loc_rel"])))

print("\nABLATION vs C2_25 (equal-weight z blends):")
for name, col in [("pooled", "loc_pooled"), ("count raw", "loc_raw"), ("count rel", "loc_rel")]:
    p3 = (fc.z(a["C2"]) + fc.z(a[col]) + fc.z(a["ridge"])) / 3
    print(f"  res + loc({name}) + stuff   P={fc.R(p3, b['C2']):.3f}  S={fc.RS(p3, b['C2']):.3f}")

rng = np.random.default_rng(42)
M = np.column_stack([a["C2"], a["ridge"], a["loc_pooled"], a["loc_raw"], a["loc_rel"],
                     b["C2"], b["loc_pooled"], b["loc_raw"], b["loc_rel"]])
d = {k: [] for k in ["rel_raw", "rel_rel", "val_raw", "val_rel", "abl_raw"]}
for _ in range(4000):
    c2, rid, lp, lr, lv, y, lp2, lr2, lv2 = M[rng.choice(N, N, replace=True)].T
    def zz(v): return (v - v.mean()) / v.std()
    d["rel_raw"].append(np.corrcoef(lr, lr2)[0, 1] - np.corrcoef(lp, lp2)[0, 1])
    d["rel_rel"].append(np.corrcoef(lv, lv2)[0, 1] - np.corrcoef(lp, lp2)[0, 1])
    d["val_raw"].append(np.corrcoef(lr, y)[0, 1] - np.corrcoef(lp, y)[0, 1])
    d["val_rel"].append(np.corrcoef(lv, y)[0, 1] - np.corrcoef(lp, y)[0, 1])
    d["abl_raw"].append(np.corrcoef((zz(c2) + zz(lr) + zz(rid)) / 3, y)[0, 1]
                        - np.corrcoef((zz(c2) + zz(lp) + zz(rid)) / 3, y)[0, 1])
print("\nPAIRED BOOTSTRAP (count12 vs pooled):")
fc.boot_report("reliability: raw - pooled", d["rel_raw"])
fc.boot_report("reliability: rel - pooled", d["rel_rel"])
fc.boot_report("validity:    raw - pooled", d["val_raw"])
fc.boot_report("validity:    rel - pooled", d["val_rel"])
fc.boot_report("3-way blend: raw - pooled", d["abl_raw"])

# ---- mechanism: sign inversion out of zone, and who moves ----
spots = pd.DataFrame({"label": ["heart (0, 2.5)", "high chase (0, 3.9)", "low chase (0, 1.1)",
                                "wide chase (1.3, 2.5)", "waste high (0, 4.4)"],
                      "PlateLocSide": [0.0, 0.0, 0.0, 1.3, 0.0],
                      "PlateLocHeight": [2.5, 3.9, 1.1, 2.5, 4.4]})
fc.add_loc_bins(spots)
pv = pooled.apply(spots)
print("\nMECHANISM: count12 map values (xT, lower = better for the pitcher):")
print(f"{'location':<22}{'pooled':>9}" + "".join(f"{c:>9}" for c in ["0-2", "1-2", "0-0", "2-0", "3-0", "3-1"]))
for i, row in spots.iterrows():
    line = f"{row['label']:<22}{pv[i]:>9.4f}"
    for cnt in ["0-2", "1-2", "0-0", "2-0", "3-0", "3-1"]:
        s = spots.loc[[i]].copy()
        s["count12"] = cnt
        line += f"{cmap.apply(s, pv[[i]]).iloc[0]:>9.4f}"
    print(line)
move = fc.z(a["loc_raw"]) - fc.z(a["loc_pooled"])
print("\ncorr(score move, 2K-rate minus 3B-rate) = %.3f  (negative = count-mix pitchers improve)"
      % fc.R(move, a["rate_2K"] - a["rate_3B"]))
print("count-mix reliability: %.3f" % fc.R(a["mix"], b["mix"]))
