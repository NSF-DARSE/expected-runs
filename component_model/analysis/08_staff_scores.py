"""08: Staff scoresheet -- the production scoring path (four-seam).

Emits, for one team's current-season arms (30+ FF), the three model scores on
the 100 +/- 15 scale (population = current-season pitchers with 100+ FF):
  AdjRes100  last-season-to-date adjusted results (C2)
  Stuff100   fixed Ridge
  Loc100     pooled location map (per RESULTS.md, the pitcher-level Location+)
  Pitch100   equal-weight z blend of the three
plus whiff rate, location fingerprint rates, a Location+ sample flag
(<50 FF flagged, 50-99 caution, per script 06), and a Stuff+ feature
attribution vs the qualified-population mean.

Also emits the Adj Results LADDER: the same C2 quantity, but shown at the two
earlier stages of the fair-criterion pipeline it is built from (RESULTS.md,
"The fair criterion") --

  Runs Allowed           raw results, luck and bad defense still in (C0)
  Expected Runs Allowed  luck/defense stripped via the EV/LA map (C1, xT)
  Adj Results            opponent quality subtracted too (C2, adjT) -- AdjRes100

A coach asked to see "how much luck and defense were worth" separately from
the score; C0 minus C1 is exactly that quantity, already computed and already
thrown away downstream (adjT never carries it). This does not add a third
component to the score -- it decomposes on the way it was BUILT, not on what
it is made of, and Adj Results still equals its own bottom rung exactly.

Also writes <workdir>/staff_scores.json with the staff records and the
count-conditioned location grids (pooled + all 12 counts) for pitch-level
explanation displays -- an 0-2 chase pitch must render as a good pitch.
Names go to stdout and the workdir only; never commit either.
"""
import json

import numpy as np
import pandas as pd

import arsenal as ar
import fair_criterion as fc

MIN_STAFF_FF = 30
FLAG_FF, CAUTION_FF = 50, 100
M_SHRINK = 5
SWING = {"StrikeSwinging", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)
fc.add_adjusted(df)
ff, model = fc.stuff_ridge(df, return_model=True)
ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
fc.add_loc_bins(ff)
fc.add_count_cols(ff)

train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
pooled = fc.PooledLocationMap(train)
ff["loc"] = pooled.apply(ff)
cmap = fc.CountLocationMap(train, "count12", M_SHRINK)

# ---- population scale: 2025 pitchers with 100+ FF ----
f25 = ff[ff["year"] == 2025]
g = f25.groupby("PitcherId")
pt = g.agg(n_ff=("Target", "size"), ridge=("ridge_pred", "mean"), loc=("loc", "mean"),
           adj=("adjT", "mean"), target=("Target", "mean"), xt=("xT", "mean"),
           name=("Pitcher", "first"), team=("PitcherTeam", "first"),
           hand=("PitcherThrows", "first"),
           **{f: (f, "mean") for f in fc.FEATS}).reset_index()
qual = pt[pt["n_ff"] >= 100]
print(f"2025 qualified population (100+ FF): {len(qual)}")
mu = {c: qual[c].mean() for c in ["ridge", "loc", "adj"]}
sd = {c: qual[c].std() for c in ["ridge", "loc", "adj"]}
for c in ["ridge", "loc", "adj"]:  # lower run value = better, so negate
    pt["z_" + c] = -(pt[c] - mu[c]) / sd[c]
pt["Stuff100"] = 100 + 15 * pt["z_ridge"]
pt["Loc100"] = 100 + 15 * pt["z_loc"]
pt["AdjRes100"] = 100 + 15 * pt["z_adj"]
blend = (pt["z_ridge"] + pt["z_loc"] + pt["z_adj"]) / 3
qb = blend[pt["n_ff"] >= 100]
pt["Pitch100"] = 100 + 15 * (blend - qb.mean()) / qb.std()

# ---- Adj Results ladder: Target -> xT -> adjT, ONE shared scale ----
# Runs Allowed and Expected Runs Allowed are deliberately NOT given their own
# population mean/sd. They ride mu["adj"]/sd["adj"] -- the exact pair AdjRes100
# itself is built on. Summing a level plus the two gaps back to the far
# endpoint is telescoping algebra (a + (b-a) + (c-b) == c) and would hold no
# matter what (mu, sd) fed each level -- that part can never break, with a
# shared scale or without one. What the shared scale actually buys is that a
# gap MEANS ONLY ONE THING: on one (mu, sd), "Defense & Luck" collapses to
# -15*(xt-target)/sd, a pure function of the raw xt-target difference, so the
# same physical luck swing prices identically for every pitcher on the board.
# Giving Target and xT their own moments (the "obvious" per-quantity choice,
# and worth resisting the next time this reads odd) drags each level's own mu
# and sd into the gap, so the SAME luck swing prices differently depending on
# a pitcher's unrelated absolute level -- every individual card still adds up,
# but the board as a whole is quietly inconsistent, which is worse because
# nothing about it looks wrong. See test_arsenal.py's shared-vs-per-level
# tests for the worked case.
pt["RunsAllowed100"] = ar.to_display(pt["target"].values, mu["adj"], sd["adj"])
pt["ExpRunsAllowed100"] = ar.to_display(pt["xt"].values, mu["adj"], sd["adj"])

# whiff + location fingerprint
sw = f25[f25["PitchCall"].isin(SWING)].groupby("PitcherId")["PitchCall"]
wt = sw.agg(n_sw="size", whiff=lambda s: (s == "StrikeSwinging").mean())
pt = pt.merge(wt, on="PitcherId", how="left")
x, z = f25["PlateLocSide"], f25["PlateLocHeight"]
fp = f25.assign(zone=((x.abs() <= 0.83) & (z >= 1.5) & (z <= 3.5)).astype(float),
                heart=((x.abs() <= 0.558) & (z >= 1.83) & (z <= 3.17)).astype(float),
                mh=z).groupby("PitcherId")[["zone", "heart", "mh"]].mean()
pt = pt.merge(fp, on="PitcherId", how="left")

# Stuff+ feature attribution vs qualified-pop mean (display points)
scaler = model.named_steps["standardscaler"]
coefs = model.named_steps["ridge"].coef_
Zq = ((qual[fc.FEATS].values - scaler.mean_) / scaler.scale_).mean(axis=0)

# ---- handedness-excluded variant (feeds the web app's "Include handedness
# impact" toggle) -- removes the is_lhp/is_lhb contribution from the raw ridge
# prediction and RE-Z-SCORES against the qualified population, so the display
# mean/sd shift too. Orientation unchanged: run value is lower = better, so the
# z negates (higher display = better). Existing scores are untouched. ----
# Derived from FEATS rather than naming both terms outright. This asked for
# is_lhb's index unconditionally, so the day it left the feature set the whole
# scorer stopped running -- and the failure was a ValueError deep in a
# subprocess, not anything pointing at the model change that caused it. The
# toggle's meaning is "take out whatever handedness terms the model has", which
# is exactly one of them now and was two before.
HAND = [i for i, f in enumerate(fc.FEATS) if f in ("is_lhp", "is_lhb")]
KEEP = [i for i in range(len(fc.FEATS)) if i not in HAND]
Zrow = (pt[fc.FEATS].values - scaler.mean_) / scaler.scale_
hand_raw = Zrow[:, HAND] @ coefs[HAND]  # per-pitcher handedness piece of ridge_pred
ridge_nohand = pd.Series(pt["ridge"].values - hand_raw, index=pt.index)
qual_mask = pt["n_ff"] >= 100  # same qualified population as mu/sd above
mu_nh = ridge_nohand[qual_mask].mean()
sd_nh = ridge_nohand[qual_mask].std()  # pandas ddof=1, matches sd["ridge"]
z_ridge_nohand = -(ridge_nohand - mu_nh) / sd_nh
pt["Stuff100_nohand"] = 100 + 15 * z_ridge_nohand
blend_nh = (z_ridge_nohand + pt["z_loc"] + pt["z_adj"]) / 3
qb_nh = blend_nh[qual_mask]
pt["Pitch100_nohand"] = 100 + 15 * (blend_nh - qb_nh.mean()) / qb_nh.std()
Zq_nh = Zq[KEEP]
feats_keep = [fc.FEATS[i] for i in KEEP]

staff = pt[(pt["team"] == args.team) & (pt["n_ff"] >= MIN_STAFF_FF)].sort_values("Pitch100", ascending=False)
print(f"\n{args.team} 2025 staff (30+ FF), scores vs D1 qualified pop:")
records = []
for _, r in staff.iterrows():
    n = int(r["n_ff"])
    flag = "small sample" if n < FLAG_FF else ("caution" if n < CAUTION_FF else "")
    Zi = (r[fc.FEATS].values.astype(float) - scaler.mean_) / scaler.scale_
    contrib = -15 * (Zi - Zq) * coefs / sd["ridge"]
    attr = sorted(zip(fc.FEATS, contrib.round(1)), key=lambda t: -t[1])
    # no-handedness driver attribution over the kept features only, reconciling
    # to Stuff100_nohand (re-z-scored, so it divides by sd_nh not sd["ridge"])
    contrib_nh = -15 * (Zi[KEEP] - Zq_nh) * coefs[KEEP] / sd_nh
    attr_nh = sorted(zip(feats_keep, contrib_nh.round(1)), key=lambda t: -t[1])
    records.append(dict(name=r["name"], hand=r["hand"][0], ff=n,
                        adjres=round(r["AdjRes100"], 1), stuff=round(r["Stuff100"], 1),
                        loc=round(r["Loc100"], 1), pitch=round(r["Pitch100"], 1),
                        whiff=round(float(r["whiff"]), 3) if pd.notna(r["whiff"]) else None,
                        zone=round(float(r["zone"]), 3), heart=round(float(r["heart"]), 3),
                        mean_height=round(float(r["mh"]), 2), loc_flag=flag,
                        stuff_attr=[(f, float(v)) for f, v in attr[:3] + attr[-2:]],
                        stuff_nohand=round(r["Stuff100_nohand"], 1),
                        pitch_nohand=round(r["Pitch100_nohand"], 1),
                        stuff_attr_nohand=[(f, float(v)) for f, v in attr_nh[:3] + attr_nh[-2:]],
                        # Adj Results ladder. Display levels are full precision
                        # (not pre-rounded to 1dp like adjres/stuff/loc above):
                        # the board card apportions them against its own
                        # rounded score cells the same way it already does for
                        # Location+'s locWhere rows, and handing it a
                        # pre-rounded input would just add a second rounding
                        # step ahead of that one.
                        #
                        # Raw units are runs per 100 pitches (mean Target/xT *
                        # 100): the mean-per-pitch value the model actually
                        # carries is ~0.00x and unreadable on a card, while
                        # runs/100 lands in the +/-10ish range for a real
                        # qualified pitcher (script 01's own C0 population sd
                        # is ~2.8 runs/100), which a coach can read at a
                        # glance.
                        res_runs_allowed=float(r["RunsAllowed100"]),
                        res_exp_runs_allowed=float(r["ExpRunsAllowed100"]),
                        res_runs_allowed_raw=round(float(r["target"]) * 100, 2),
                        res_exp_runs_allowed_raw=round(float(r["xt"]) * 100, 2),
                        res_adj_results_raw=round(float(r["adj"]) * 100, 2)))
    print(f"  {r['name']:<24}({r['hand'][0]}) FF={n:>4} AdjRes={r['AdjRes100']:>5.0f} "
          f"Stuff+={r['Stuff100']:>5.0f} Loc+={r['Loc100']:>5.0f} Pitch+={r['Pitch100']:>5.0f} "
          f"whiff={('%.0f%%' % (r['whiff'] * 100)) if pd.notna(r['whiff']) else ' NA'} {flag}")

# ---- location grids for pitch-level explanation (pooled + per count) ----
cells = []
xs = np.arange(-1.25, 1.25, 0.25)
zs = np.arange(1.0, 4.0, 0.25)
grid = pd.DataFrame([(gx, gz) for gx in xs for gz in zs], columns=["PlateLocSide", "PlateLocHeight"])
grid["PlateLocSide"] += 0.01  # cell interior so binning lands in the intended cell
grid["PlateLocHeight"] += 0.01
fc.add_loc_bins(grid)
pv = pooled.apply(grid)
out_grids = {"pooled": [dict(x=round(float(a), 2), z=round(float(b), 2), v=round(float(v), 4))
                        for a, b, v in zip(grid["gx"], grid["gz"], pv)]}
for cnt in sorted(train["count12"].unique()):
    s = grid.copy()
    s["count12"] = cnt
    out_grids[cnt] = [dict(x=round(float(a), 2), z=round(float(b), 2), v=round(float(v), 4))
                      for a, b, v in zip(grid["gx"], grid["gz"], cmap.apply(s, pv))]

with open(f"{args.workdir}/staff_scores.json", "w") as f:
    json.dump(dict(population=len(qual), team=args.team, staff=records, grids=out_grids), f)
print(f"\nwrote {args.workdir}/staff_scores.json "
      f"(staff records + pooled and per-count location grids)")
