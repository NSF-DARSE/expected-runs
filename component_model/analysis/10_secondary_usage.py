"""10: What predicts secondary-pitch outcomes beyond stuff? A usage/deception
composite.

Script 09 showed season-average location has no predictive validity for
secondary pitches. This script screens season-level usage and fastball-relative
location traits (reliability gate first, per protocol), then tests an
equal-weight composite of the four traits whose SIGNS replicated across the
slider and changeup panels:
  + sep_z  : thrown below the pitcher's own fastball band (depth off the FB line)
  - sep_x  : horizontal separation off the fastball's line (tunneling penalty)
  - usage  : share of all pitches (overexposure penalty)
  - two_strike_share : two-strike-only predictability penalty
Composite is a new labeled component (usage/deception), NOT stuff and NOT
location -- see FRAMEWORK.md score design principles. EXPLORATORY: traits were
selected on this same season-pair; treat as discovered-not-confirmed until 2026.
"""
import numpy as np
import pandas as pd

import fair_criterion as fc

TYPES = ["Slider", "ChangeUp", "Curveball"]

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)
fc.add_adjusted(df)

df = df.sort_values(["GameID", "Inning", "Top/Bottom", "PAofInning", "PitchofPA"])
pa = ["GameID", "Inning", "Top/Bottom", "PAofInning"]
df["prev_type"] = df.groupby(pa, observed=True)["TaggedPitchType"].shift(1)
df["prev_is_fb"] = df["prev_type"].isin(fc.FF_TYPES | {"Sinker", "TwoSeamFastBall"})
df["is_first_of_pa"] = df["PitchofPA"] == 1

ffm = (df[df["is_ff"] & df["PlateLocSide"].notna()]
       .groupby(["PitcherId", "year"])[["PlateLocSide", "PlateLocHeight"]].mean()
       .rename(columns={"PlateLocSide": "ff_x", "PlateLocHeight": "ff_z"}))
tot = df.groupby(["PitcherId", "year"]).size().rename("n_all")

for ptype in TYPES:
    pp = fc.stuff_ridge(df, pitch_mask=df["TaggedPitchType"] == ptype)
    pp = pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()].copy()
    fc.add_count_cols(pp)
    ids = fc.panel_ids(pp)
    pp = pp[pp["PitcherId"].isin(ids)].join(ffm, on=["PitcherId", "year"])
    g = pp.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({
        "C2": g["adjT"].mean(), "ridge": g["ridge_pred"].mean(),
        "usage": g.size() / tot,
        "two_strike_share": g["s"].agg(lambda s: (s == 2).mean()),
        "prev_fb_rate": g["prev_is_fb"].mean(),
        "first_pitch_rate": g["is_first_of_pa"].mean(),
        "sep_x": g.apply(lambda d: (d["PlateLocSide"] - d["ff_x"]).abs().mean(), include_groups=False),
        "sep_z": g.apply(lambda d: (d["ff_z"] - d["PlateLocHeight"]).mean(), include_groups=False),
        "sd_x": g["PlateLocSide"].std(), "sd_z": g["PlateLocHeight"].std(),
    }).reset_index()
    a, b = fc.year_split(tab, sorted(ids))
    N = len(ids)
    small = " [PANEL TOO SMALL -- descriptive only]" if N < 60 else ""
    print(f"\n{'=' * 74}\n{ptype} (n={N}){small}")
    print(f"{'trait':<18}{'rel':>7}{'val P':>8}{'val S':>8}{'corr w/ ridge24':>17}")
    for c in ["ridge", "usage", "two_strike_share", "prev_fb_rate", "first_pitch_rate",
              "sep_x", "sep_z", "sd_x", "sd_z", "C2"]:
        av, bv = a[c].astype(float), b[c].astype(float)
        ok = av.notna() & bv.notna()
        print(f"{c:<18}{fc.R(av[ok], bv[ok]):>7.3f}{fc.R(av[ok], b['C2'][ok]):>8.3f}"
              f"{fc.RS(av[ok], b['C2'][ok]):>8.3f}{fc.R(av[ok], a['ridge'][ok]):>17.3f}")

    comp24 = (fc.z(a["sep_z"]) - fc.z(a["sep_x"]) - fc.z(a["usage"]) - fc.z(a["two_strike_share"])) / 4
    comp25 = (fc.z(b["sep_z"]) - fc.z(b["sep_x"]) - fc.z(b["usage"]) - fc.z(b["two_strike_share"])) / 4
    print(f"\nusage/deception composite: rel={fc.R(comp24, comp25):.3f}  "
          f"val={fc.R(comp24, b['C2']):.3f}/{fc.RS(comp24, b['C2']):.3f}  "
          f"corr w/ ridge={fc.R(comp24, a['ridge']):.3f}")
    for nm, p in [("results", fc.z(a["C2"])), ("ridge", fc.z(a["ridge"])),
                  ("res+ridge", (fc.z(a["C2"]) + fc.z(a["ridge"])) / 2),
                  ("res+ridge+composite", (fc.z(a["C2"]) + fc.z(a["ridge"]) + fc.z(comp24)) / 3)]:
        print(f"  {nm:<20} P={fc.R(p, b['C2']):.3f}  S={fc.RS(p, b['C2']):.3f}")

    rng = np.random.default_rng(42)
    M = np.column_stack([a["C2"], a["ridge"], comp24, b["C2"]])
    d = []
    for _ in range(4000):
        c2, rid, cp, y = M[rng.choice(N, N, replace=True)].T
        def zz(v):
            return (v - v.mean()) / v.std()
        d.append(np.corrcoef((zz(c2) + zz(rid) + zz(cp)) / 3, y)[0, 1]
                 - np.corrcoef((zz(c2) + zz(rid)) / 2, y)[0, 1])
    fc.boot_report("composite over res+ridge", d)
