"""01: Build the fair criterion and verify the anchor numbers.

Criteria per pitcher-year (mean over qualifying FF):
  C0 = raw Target (luck included)
  C1 = xT (defense/luck stripped)
  C2 = adjT (opponent-adjusted xT) -- the FIXED fair criterion for all later scripts
Prints the reliability ladder and the Stuff+ Ridge vs matched stat-line baselines.
Anchor values for the current source CSV are recorded in RESULTS.md; if these
prints do not match, stop and reconcile before trusting any downstream script.
"""
import pandas as pd

import fair_criterion as fc

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)
fc.add_adjusted(df)
ff = fc.stuff_ridge(df)
ids = fc.panel_ids(ff)
print(f"rows={len(df)}  FF rows with complete features={len(ff)}  panel pitchers={len(ids)}")

g = ff[ff["PitcherId"].isin(ids)].groupby(["PitcherId", "year"])
tab = pd.DataFrame({"C0": g["Target"].mean(), "C1": g["xT"].mean(),
                    "C2": g["adjT"].mean(), "ridge": g["ridge_pred"].mean()}).reset_index()
a, b = fc.year_split(tab, ids)

print(f"\nRELIABILITY + VALIDITY (Pearson, n={len(ids)}; SE on one r ~ 0.037):")
print(f"{'criterion':<10}{'rel(24,25)':>11}{'own24->C_25':>13}{'ridge24->C_25':>14}{'blend->C_25':>12}")
for c in ["C0", "C1", "C2"]:
    bl = 0.5 * fc.z(a["ridge"]) + 0.5 * fc.z(a[c])
    print(f"{c:<10}{fc.R(a[c], b[c]):>11.3f}{fc.R(a[c], b[c]):>13.3f}"
          f"{fc.R(a['ridge'], b[c]):>14.3f}{fc.R(bl, b[c]):>12.3f}")

print(f"\nStuff+ Ridge reliability: {fc.R(a['ridge'], b['ridge']):.3f}")
print("Spearman (C2 row): baseline %.3f  ridge %.3f"
      % (fc.RS(a["C2"], b["C2"]), fc.RS(a["ridge"], b["C2"])))
print("criterion cross-corrs 2024: C0-C1=%.3f  C0-C2=%.3f  C1-C2=%.3f"
      % (fc.R(a["C0"], a["C1"]), fc.R(a["C0"], a["C2"]), fc.R(a["C1"], a["C2"])))
