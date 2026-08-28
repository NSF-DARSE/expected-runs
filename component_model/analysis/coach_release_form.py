"""Is his 'unusual either way' form more insightful than our monotone one?

PRE-REGISTERED before looking at any result:
  Criterion  = next season's four-seam run value (2026), pitcher level, 100+ FF both
               seasons. Same fixed fair criterion the rest of the work uses. NOT pitch R^2.
  Statistic  = r(pitcher mean prediction, 2026 criterion). Both are expected runs, lower =
               better, so a POSITIVE r means the score is valid.
  Decision   = paired bootstrap over pitchers on the DIFFERENCE vs baseline. Under ~1 SE
               is a tie. His form "captures real nuance we lost" only if adding it to our
               model IMPROVES validity by more than 1 SE.
  Hand means are taken from TRAIN data, not his hardcoded constants, so this tests the
  FORM (deviation-from-typical vs monotone) and not his particular centring values.
"""
import numpy as np, pandas as pd
import fair_criterion as fc, coach_model_ff_criterion as ffc
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DEV = {"RelHeight": "dev_relheight", "RelSide_arm": "dev_relside"}

def add_dev(df, means):
    for col, out in DEV.items():
        df[out] = (df[col] - df["is_lhp"].map(means[col])).abs()
    return df

score = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025").dropna(subset=fc.FEATS).copy()
crit  = ffc._frame(ffc.CRIT_WORKDIR,  "2025,2026")
tr_raw = score[(score["year"] == 2024) & score["Target"].notna()]
means = {c: tr_raw.groupby("is_lhp")[c].mean() for c in DEV}
print("  train hand means: " + ", ".join(
    f"{c} RHP {means[c][0]:.2f} / LHP {means[c][1]:.2f}" for c in DEV))

score = add_dev(score, means)
tr = score[(score["year"] == 2024) & score["Target"].notna()]
ev = score[score["year"] == 2025].copy()
k = ev.groupby("PitcherId").size(); ev = ev[ev["PitcherId"].isin(k[k >= 100].index)]

c26 = crit[crit["year"] == 2025].groupby("PitcherId").agg(
    n=("adjT", "size"), adjT=("adjT", "mean"), Target=("Target", "mean"))
c26 = c26[c26["n"] >= 100]

VARIANTS = {
    "baseline (ours, monotone)": fc.FEATS,
    "ours + his deviation terms": fc.FEATS + list(DEV.values()),
    "his form instead of ours": [f for f in fc.FEATS if f not in DEV] + list(DEV.values()),
    "his form, release HEIGHT only": [f for f in fc.FEATS if f != "RelHeight"] + ["dev_relheight"],
    "his form, release SIDE only": [f for f in fc.FEATS if f != "RelSide_arm"] + ["dev_relside"],
}

preds = {}
for name, cols in VARIANTS.items():
    m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(tr[cols].values, tr["Target"].values)
    p = pd.Series(m.predict(ev[cols].values), index=ev["PitcherId"].values)
    preds[name] = p.groupby(level=0).mean()

for crit_name in ("adjT", "Target"):
    j = pd.DataFrame({k2: v for k2, v in preds.items()}).join(c26[crit_name], how="inner").dropna()
    print(f"\n=== criterion: 2026 four-seam {crit_name} (n={len(j)} pitchers) ===")
    base = "baseline (ours, monotone)"
    r = {k2: float(fc.R(j[k2], j[crit_name])) for k2 in VARIANTS}
    for k2 in VARIANTS:
        print(f"    {k2:<32s} validity r = {r[k2]:+.4f}")
    rng = np.random.default_rng(7)
    idx = j.index.values
    B = {k2: [] for k2 in VARIANTS}
    for _ in range(2000):
        s = j.loc[rng.choice(idx, len(idx))]
        for k2 in VARIANTS:
            B[k2].append(float(fc.R(s[k2], s[crit_name])))
    B = {k2: np.array(v) for k2, v in B.items()}
    print("    paired bootstrap, variant minus baseline (same resamples):")
    for k2 in VARIANTS:
        if k2 == base: continue
        d = B[k2] - B[base]
        print(f"      {k2:<32s} mean={d.mean():+.4f}  SE={d.std():.4f}  "
              f"P(better)={float((d>0).mean()):.3f}  {'>1 SE' if abs(d.mean())>d.std() else 'tie'}")
