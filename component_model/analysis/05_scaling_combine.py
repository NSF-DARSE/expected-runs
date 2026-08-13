"""05: Honest scaling of Stuff+ vs Location+ and the Pitching+ combine.

Tests whether the MLB convention (stuff spreads pitchers ~3.6x more than
location; both shown 100+/-15) holds at the college level, and whether the
equal-weight z blend survives when components are combined on their natural
run-value scales instead. Requires 04 to have written count_scores.parquet.
"""
import numpy as np
import pandas as pd

import fair_criterion as fc

args = fc.paths()
tab = pd.read_parquet(f"{args.workdir}/count_scores.parquet")
ids = sorted(set(tab[tab["year"] == 2024]["PitcherId"]) & set(tab[tab["year"] == 2025]["PitcherId"]))
a, b = fc.year_split(tab, ids)
N = len(ids)

print(f"PITCHER-LEVEL SPREADS, runs/100 pitches (n={N}):")
print(f"{'score':<12}{'sd24':>8}{'sd25':>8}{'rel':>7}{'signal sd':>11}")
for c in ["ridge", "loc_pooled", "loc_raw", "C2"]:
    rel = fc.R(a[c], b[c])
    sig = 100 * a[c].std() * np.sqrt(max(rel, 0))
    print(f"{c:<12}{100 * a[c].std():>8.2f}{100 * b[c].std():>8.2f}{rel:>7.3f}{sig:>11.2f}")
ratio = a["ridge"].std() / a["loc_pooled"].std()
print(f"\nSD ratio stuff:location = {ratio:.2f}  [MLB primer ~3.6 -- inverted here]")
print(f"Honest Location+ display scale if Stuff+ is 100+/-15: 100+/-{15 / ratio:.1f}")

combos = [
    ("equal z 1/3 (production)", (fc.z(a["C2"]) + fc.z(a["loc_pooled"]) + fc.z(a["ridge"])) / 3),
    ("natural-unit run sum", (a["C2"] - a["C2"].mean()) + (a["loc_pooled"] - a["loc_pooled"].mean())
     + (a["ridge"] - a["ridge"].mean())),
    ("reliability-weighted z", sum(fc.R(a[c], b[c]) * fc.z(a[c]) for c in ["C2", "loc_pooled", "ridge"])),
]
print("\nCOMBOS vs C2_25 (weights use 2024 info only):")
for name, p in combos:
    print(f"  {name:<28} P={fc.R(p, b['C2']):.3f}  S={fc.RS(p, b['C2']):.3f}")

rng = np.random.default_rng(42)
rels = {c: fc.R(a[c], b[c]) for c in ["C2", "loc_pooled", "ridge"]}
M = np.column_stack([a["C2"], a["loc_pooled"], a["ridge"], b["C2"]])
d_nat, d_rw = [], []
for _ in range(4000):
    c2, loc, rid, y = M[rng.choice(N, N, replace=True)].T
    def zz(v): return (v - v.mean()) / v.std()
    eq = np.corrcoef((zz(c2) + zz(loc) + zz(rid)) / 3, y)[0, 1]
    nat = np.corrcoef((c2 - c2.mean()) + (loc - loc.mean()) + (rid - rid.mean()), y)[0, 1]
    rw = np.corrcoef(rels["C2"] * zz(c2) + rels["loc_pooled"] * zz(loc) + rels["ridge"] * zz(rid), y)[0, 1]
    d_nat.append(eq - nat)
    d_rw.append(eq - rw)
print("\nPAIRED BOOTSTRAP:")
fc.boot_report("equal-z minus natural-sum", d_nat)
fc.boot_report("equal-z minus reliability-weighted", d_rw)
