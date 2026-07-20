"""07: Can the models alone out-predict the stat line?

Location+ + Stuff+ (equal z, no results input) vs two last-year baselines:
raw run value allowed (C0, luck included) and the luck-stripped opponent-adjusted
version (C2). Requires 04 to have written count_scores.parquet.
"""
import numpy as np
import pandas as pd

import fair_criterion as fc

args = fc.paths()
tab = pd.read_parquet(f"{args.workdir}/count_scores.parquet")
ff = fc.ff_panel(args)
c0 = ff[ff["in_panel"]].groupby(["PitcherId", "year"])["Target"].mean().rename("C0").reset_index()
tab = tab.merge(c0, on=["PitcherId", "year"])
ids = sorted(set(tab[tab["year"] == 2024]["PitcherId"]) & set(tab[tab["year"] == 2025]["PitcherId"]))
a, b = fc.year_split(tab, ids)
N = len(ids)

models = 0.5 * fc.z(a["loc_pooled"]) + 0.5 * fc.z(a["ridge"])
rows = [("C0: raw results (luck included)", a["C0"]),
        ("C2: adjusted results (luck-stripped)", a["C2"]),
        ("Location+ + Stuff+ (no results)", models),
        ("all three (equal 1/3)", (fc.z(a["C2"]) + fc.z(a["loc_pooled"]) + fc.z(a["ridge"])) / 3)]
print(f"PREDICTING C2_25 (n={N}):")
for name, p in rows:
    print(f"  {name:<40} P={fc.R(p, b['C2']):.3f}  S={fc.RS(p, b['C2']):.3f}")

rng = np.random.default_rng(42)
M = np.column_stack([a["C0"], a["C2"], a["loc_pooled"], a["ridge"], b["C2"]])
d0, d2 = [], []
for _ in range(4000):
    c0_, c2_, loc, rid, y = M[rng.choice(N, N, replace=True)].T
    def zz(v): return (v - v.mean()) / v.std()
    rm = np.corrcoef(0.5 * zz(loc) + 0.5 * zz(rid), y)[0, 1]
    d0.append(rm - np.corrcoef(c0_, y)[0, 1])
    d2.append(rm - np.corrcoef(c2_, y)[0, 1])
print("\nPAIRED BOOTSTRAP (models-only minus baseline):")
fc.boot_report("vs raw results", d0)
fc.boot_report("vs adjusted results", d2)
