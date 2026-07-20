"""11: Arsenal-relative expected usage and the trust residual (Usage Gap Board data).

Expected usage MUST be arsenal-relative: a pitch competes for usage with the
pitcher's own other pitches, not with the league. A pitch leaned on because
everything else in the arsenal is worse is NOT a disagreement. So the share
model works within pitcher: each graded pitch type's usage share is predicted
from a type base rate plus how the pitch's measured quality (per-type Stuff+,
results, whiff) stands RELATIVE to the pitcher's other graded pitches. The
residual (actual minus expected share) is trust the measurements cannot explain
-- the conversation queue. Descriptive, current-season, cross-sectional; this
is a conversation tool, not a score (see RESULTS.md deployment caveats).

Writes <workdir>/usage_gap.json (per-team cases + population context).
Level II: names in workdir/stdout only; never commit outputs.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import fair_criterion as fc

TYPE_MAP = {"Fastball": "FF", "FourSeamFastBall": "FF", "FourSeamFastball": "FF",
            "Slider": "SL", "ChangeUp": "CH", "Curveball": "CB",
            "Cutter": "CT", "Sinker": "SI", "TwoSeamFastBall": "SI"}
TYPE_NAME = {"FF": "four-seam", "SL": "slider", "CH": "changeup", "CB": "curveball",
             "CT": "cutter", "SI": "sinker"}
SWING = {"StrikeSwinging", "FoulBall", "FoulBallFieldable", "FoulBallNotFieldable", "InPlay"}
MIN_FIT_PITCHES = 150   # pitcher-year total to enter the share-model fit
MIN_TYPE_QUAL = 50      # pitches of a type to enter that type's 100+/-15 scale pop
MIN_BOARD = 30          # pitches of a type to appear on the board

args = fc.paths()
df = fc.load_pitches(args)
fc.add_xt(df)
fc.add_adjusted(df)
df["ptype"] = df["TaggedPitchType"].map(TYPE_MAP)

# ---- per-type Stuff+ ridge (trained 2024, identical form to the FF reference) ----
frames = []
for t in TYPE_NAME:
    pp = fc.stuff_ridge(df, pitch_mask=df["ptype"] == t)
    pp["ptype"] = t
    frames.append(pp)
    print(f"ridge {t}: rows={len(pp)}")
allp = pd.concat(frames)

# ---- FF Location+ (pooled map, for fastball evidence tiles only) ----
ffp = allp[allp["ptype"] == "FF"].copy()
ffp = ffp[ffp["PlateLocSide"].notna() & ffp["PlateLocHeight"].notna()]
fc.add_loc_bins(ffp)
lmap = fc.PooledLocationMap(ffp[(ffp["year"] == 2024) & ffp["xT"].notna()])
allp["locv"] = np.nan
allp.loc[ffp.index, "locv"] = lmap.apply(ffp)

# ---- pitcher-year-type table, 2025 ----
p25 = allp[allp["year"] == 2025].copy()
p25["is_swing"] = p25["PitchCall"].isin(SWING)
p25["is_whiff"] = p25["PitchCall"] == "StrikeSwinging"
bip = p25["PitchCall"] == "InPlay"
p25["is_hh"] = np.where(bip & p25["ExitSpeed"].notna(), (p25["ExitSpeed"] >= 95).astype(float), np.nan)
tot = allp[allp["year"] == 2025].groupby("PitcherId").size().rename("n_all")

g = p25.groupby(["PitcherId", "ptype"])
tt = pd.DataFrame({
    "n": g.size(), "ridge": g["ridge_pred"].mean(), "adj": g["adjT"].mean(),
    "locv": g["locv"].mean(),
    "swings": g["is_swing"].sum(), "whiffs": g["is_whiff"].sum(),
    "bip_ev": g["is_hh"].count(), "hh": g["is_hh"].mean(),
    "name": g["Pitcher"].agg("first"), "team": g["PitcherTeam"].agg("first"),
    "hand": g["PitcherThrows"].agg("first"),
}).reset_index().join(tot, on="PitcherId")
tt["share"] = tt["n"] / tt["n_all"]
tt["whiff"] = tt["whiffs"] / tt["swings"].replace(0, np.nan)

# per-type scales (pitchers with 50+ of the type)
qual = tt[tt["n"] >= MIN_TYPE_QUAL]
mus = qual.groupby("ptype")[["ridge", "adj", "whiff"]].mean()
sds = qual.groupby("ptype")[["ridge", "adj", "whiff"]].std()
wavg = qual.groupby("ptype")["whiff"].mean()
hhavg = qual.groupby("ptype")["hh"].mean()
for c in ["ridge", "adj", "whiff"]:
    tt[f"z_{c}"] = (tt[c] - tt["ptype"].map(mus[c])) / tt["ptype"].map(sds[c])
# quality orientation: lower run value = better => negate ridge/adj z; whiff higher = better
tt["q_stuff"] = -tt["z_ridge"]
tt["q_res"] = -tt["z_adj"]
tt["q_whiff"] = tt["z_whiff"]
tt["Stuff100"] = 100 + 15 * tt["q_stuff"]
# FF Location+ scale
ffq = qual[qual["ptype"] == "FF"]
tt["Loc100"] = np.where(tt["ptype"] == "FF",
                        100 - 15 * (tt["locv"] - ffq["locv"].mean()) / ffq["locv"].std(), np.nan)

# ---- arsenal-relative share model (fit on all D1 pitcher-years with 150+ pitches) ----
fit = tt[(tt["n_all"] >= MIN_FIT_PITCHES) & (tt["n"] >= 10)].copy()
# Fastball identity is a choice, not a disagreement: model FF/SI as primary-vs-secondary
# fastball SLOTS (by within-pitcher count), so sinker-first pitchers don't top the queue
# with fake four-seam deficits.
fam = fit["ptype"].isin(["FF", "SI"])
fbmax = fit[fam].groupby("PitcherId")["n"].transform("max")
fit["slot"] = fit["ptype"]
fit.loc[fam, "slot"] = np.where(fit.loc[fam, "n"] == fbmax, "PFB", "SFB")
SLOTS = ["PFB", "SFB", "SL", "CH", "CB", "CT"]
# relative quality: this pitch's quality minus the mean quality of the pitcher's OTHER graded pitches
def rel_within(col):
    s = fit.groupby("PitcherId")[col].transform("sum")
    k = fit.groupby("PitcherId")[col].transform("count")
    other = (s - fit[col]) / (k - 1).replace(0, np.nan)
    return (fit[col] - other).fillna(0.0)
for col in ["q_stuff", "q_res", "q_whiff"]:
    fit[f"rel_{col}"] = rel_within(col).clip(-4, 4)
X = pd.get_dummies(fit["slot"])[SLOTS].astype(float)
for col in ["rel_q_stuff", "rel_q_res", "rel_q_whiff"]:
    X[col] = fit[col].fillna(0.0)
share_model = LinearRegression(fit_intercept=False).fit(X.values, fit["share"].values)
fit["exp_raw"] = share_model.predict(X.values)
print("\nSHARE MODEL (n=%d pitcher-type rows, %d pitchers): R2=%.3f" %
      (len(fit), fit["PitcherId"].nunique(), share_model.score(X.values, fit["share"].values)))
print("  slot base shares:", {t: round(c, 3) for t, c in zip(SLOTS, share_model.coef_[:len(SLOTS)])})
print("  rel-quality coefs (share pts per z of relative quality):",
      {c: round(v, 4) for c, v in zip(["stuff", "results", "whiff"], share_model.coef_[len(SLOTS):])})

# normalize expected within pitcher to the pitcher's total graded share, then residual
grp = fit.groupby("PitcherId")
fit["exp"] = fit["exp_raw"].clip(lower=0.02) * grp["share"].transform("sum") / grp["exp_raw"].transform(
    lambda s: s.clip(lower=0.02).sum())
fit["gap"] = (fit["share"] - fit["exp"]) * 100
print("residual (share pts): sd=%.1f  p5=%.1f  p95=%.1f" %
      (fit["gap"].std(), fit["gap"].quantile(.05), fit["gap"].quantile(.95)))

# ---- board payload for the requested team ----
d = fit[(fit["team"] == args.team) & (fit["n"] >= MIN_BOARD)].copy()
d["absgap"] = d["gap"].abs()
d = d.sort_values("absgap", ascending=False)
cases = []
for _, r in d.iterrows():
    others = fit[(fit["PitcherId"] == r["PitcherId"]) & (fit["ptype"] != r["ptype"])]
    if len(others):
        cp = others.loc[(others["gap"] * (-np.sign(r["gap"]))).idxmax()]
        trade = (f"{'More' if r['gap']<0 else 'Fewer'} {TYPE_NAME[r['ptype']]}s would mostly "
                 f"{'come off' if r['gap']<0 else 'go to'} his {TYPE_NAME[cp['ptype']]} "
                 f"({cp['share']*100:.0f}% usage, grades {cp['Stuff100']:.0f}).")
    else:
        trade = "Only one graded pitch type at this sample."
    cases.append(dict(
        nm=r["name"], pt=r["ptype"], hand=r["hand"][0], gap=round(r["gap"], 1),
        stuff=round(r["Stuff100"]), loc=(round(r["Loc100"]) if pd.notna(r["Loc100"]) else None),
        whiff=(round(float(r["whiff"]), 3) if pd.notna(r["whiff"]) else None),
        swings=int(r["swings"]), hh=(round(float(r["hh"]) * 100) if pd.notna(r["hh"]) else None),
        hhavg=round(float(hhavg[r["ptype"]]) * 100), wavg=round(float(wavg[r["ptype"]]) * 100),
        n=int(r["n"]), use=round(r["share"] * 100), exp=round(r["exp"] * 100), trade=trade))
    print(f"{r['name']:<24}{r['ptype']} n={int(r['n']):>4} share={r['share']*100:>4.0f}% "
          f"exp={r['exp']*100:>4.0f}% gap={r['gap']:>+5.1f} Stuff+={r['Stuff100']:>4.0f} "
          f"whiff={r['whiff']*100 if pd.notna(r['whiff']) else float('nan'):>3.0f}%")

with open(f"{args.workdir}/usage_gap.json", "w") as f:
    json.dump(dict(team=args.team, season=2025, fit_pitchers=int(fit["PitcherId"].nunique()),
                   resid_sd=round(float(fit["gap"].std()), 1),
                   whiff_avg={t: round(float(v), 3) for t, v in wavg.items()},
                   cases=cases), f, indent=1)
print(f"\nwrote {args.workdir}/usage_gap.json ({len(cases)} team cases)")
