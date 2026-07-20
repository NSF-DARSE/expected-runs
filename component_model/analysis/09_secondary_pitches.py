"""09: Extend the stack to secondary pitch types and re-test count-conditioning.

Runs the four-seam protocol (scripts 01/03/04 condensed) per pitch type:
per-type Stuff+ Ridge, per-type pooled location map, per-type fair criterion
(adjT over that type's pitches), and the count-conditioned map with the
location-given-count decomposition. The open hypothesis from RESULTS.md is
that count-conditioning improves LOCATION MEASUREMENT (the rel variant) for
breaking/offspeed pitches, where the ideal target moves with count -- it did
not for four-seams. Panel = 100+ pitches of the type in both years.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import fair_criterion as fc

TYPES = ["Slider", "ChangeUp", "Curveball"]
MS = [1, 2, 5, 10, 25, 100]

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)
fc.add_adjusted(df)

for ptype in TYPES:
    pp = fc.stuff_ridge(df, pitch_mask=df["TaggedPitchType"] == ptype)
    pp = pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(pp)
    fc.add_count_cols(pp)
    ids = fc.panel_ids(pp)
    panel_rows = pp["PitcherId"].isin(ids)
    print(f"\n{'=' * 70}\n{ptype}: rows={len(pp)}  panel pitchers (100+ both years)={len(ids)}")

    # tune m on 2024 even/odd holdout (same protocol as 04)
    t24 = pp[(pp["year"] == 2024) & pp["xT"].notna()].copy()
    t24["half"] = t24.groupby("PitcherId").cumcount() % 2
    A, B = t24[t24["half"] == 0], t24[t24["half"] == 1]
    pooled_B = fc.PooledLocationMap(A).apply(B)
    mses = {m: ((B["xT"] - fc.CountLocationMap(A, "count12", m).apply(B, pooled_B)) ** 2).mean()
            for m in MS}
    best_m = min(mses, key=mses.get)
    print(f"holdout: pooled mse={((B['xT'] - pooled_B) ** 2).mean():.6f}  "
          f"count12 best m={best_m} mse={mses[best_m]:.6f}")

    train = pp[(pp["year"] == 2024) & pp["xT"].notna()]
    pooled = fc.PooledLocationMap(train)
    pp["loc_pooled"] = pooled.apply(pp)
    cmap = fc.CountLocationMap(train, "count12", best_m)
    pp["loc_raw"] = cmap.apply(pp, pp["loc_pooled"])
    pp["loc_rel"] = cmap.apply_relative(pp, pp["loc_pooled"])

    g = pp[panel_rows].groupby(["PitcherId", "year"])
    tab = pd.DataFrame({"C2": g["adjT"].mean(), "ridge": g["ridge_pred"].mean(),
                        "loc_pooled": g["loc_pooled"].mean(), "loc_raw": g["loc_raw"].mean(),
                        "loc_rel": g["loc_rel"].mean()}).reset_index()
    a, b = fc.year_split(tab, sorted(ids))
    N = len(ids)

    print(f"scores (n={N}): reliability | validity vs C2_25 (P/S)")
    for c in ["C2", "ridge", "loc_pooled", "loc_raw", "loc_rel"]:
        print(f"  {c:<12} rel={fc.R(a[c], b[c]):.3f}  "
              f"val={fc.R(a[c], b['C2']):.3f}/{fc.RS(a[c], b['C2']):.3f}")
    print("2024 cross-corrs: ridge-loc=%.3f  loc-C2=%.3f  ridge-C2=%.3f"
          % (fc.R(a["ridge"], a["loc_pooled"]), fc.R(a["loc_pooled"], a["C2"]),
             fc.R(a["ridge"], a["C2"])))

    print("ablation vs C2_25 (equal z):")
    for name, cols in [("results", ["C2"]), ("res+loc", ["C2", "loc_pooled"]),
                       ("res+loc+stuff", ["C2", "loc_pooled", "ridge"]),
                       ("res+loc(rel)+stuff", ["C2", "loc_rel", "ridge"])]:
        p = sum(fc.z(a[c]) for c in cols) / len(cols)
        print(f"  {name:<20} P={fc.R(p, b['C2']):.3f}  S={fc.RS(p, b['C2']):.3f}")

    rng = np.random.default_rng(42)
    M = np.column_stack([a["loc_pooled"], a["loc_raw"], a["loc_rel"],
                         b["loc_pooled"], b["loc_raw"], b["loc_rel"], b["C2"]])
    d = {k: [] for k in ["rel_rel", "val_rel", "rel_raw", "val_raw"]}
    for _ in range(4000):
        lp, lr, lv, lp2, lr2, lv2, y = M[rng.choice(N, N, replace=True)].T
        d["rel_rel"].append(np.corrcoef(lv, lv2)[0, 1] - np.corrcoef(lp, lp2)[0, 1])
        d["val_rel"].append(np.corrcoef(lv, y)[0, 1] - np.corrcoef(lp, y)[0, 1])
        d["rel_raw"].append(np.corrcoef(lr, lr2)[0, 1] - np.corrcoef(lp, lp2)[0, 1])
        d["val_raw"].append(np.corrcoef(lr, y)[0, 1] - np.corrcoef(lp, y)[0, 1])
    print("paired bootstrap (count12 vs pooled):")
    fc.boot_report("reliability: rel - pooled  [the hypothesis]", d["rel_rel"])
    fc.boot_report("validity:    rel - pooled  [the hypothesis]", d["val_rel"])
    fc.boot_report("reliability: raw - pooled", d["rel_raw"])
    fc.boot_report("validity:    raw - pooled", d["val_raw"])

    # mechanism spots: does the ideal target move with count for this pitch?
    spots = pd.DataFrame({"label": ["heart (0, 2.5)", "low chase (0, 1.1)",
                                    "buried (0, 0.7)", "high chase (0, 3.9)"],
                          "PlateLocSide": [0.0, 0.0, 0.0, 0.0],
                          "PlateLocHeight": [2.5, 1.1, 0.7, 3.9]})
    fc.add_loc_bins(spots)
    pv = pooled.apply(spots)
    print(f"{'location':<18}{'pooled':>9}" + "".join(f"{c:>9}" for c in ["0-2", "0-0", "3-0", "3-1"]))
    for i, row in spots.iterrows():
        line = f"{row['label']:<18}{pv[i]:>9.4f}"
        for cnt in ["0-2", "0-0", "3-0", "3-1"]:
            s = spots.loc[[i]].copy()
            s["count12"] = cnt
            line += f"{cmap.apply(s, pv[[i]]).iloc[0]:>9.4f}"
        print(line)
