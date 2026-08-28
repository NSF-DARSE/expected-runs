"""How much of the v1-v2 agreement is just handedness? Refit v2 without it and re-measure.

THE QUESTION (Jack, 2026-08-16): our ridge carries `is_lhp` as an additive feature, so a
left-hander gets a constant level bump. The coach's card has no handedness FEATURE, but it
is not handedness-blind either -- it holds a separate weight table per hand, which changes
how every other measurement is read without shifting the level between hands. Those are
different designs, and `is_lhp` correlates +0.43 with the part of our score his card does
not contain. So: how much of the r=0.75 agreement, and of our score generally, is carried
by handedness?

WHAT IS REFIT: the ridge is retrained from scratch on the reduced feature set, NOT scored
with the coefficient zeroed out. Zeroing would leave the other coefficients fit in the
presence of handedness and understate how much the model reorganises without it.

THREE VARIANTS:
  full        all FEATS, the shipping model
  no_lhp      drop is_lhp (pitcher handedness)
  no_hand     drop is_lhp AND is_lhb (batter handedness)

`is_lhb` is included as a third variant because it is even further from a property of the
pitch: aggregated to a pitcher-season it credits or blames a pitcher for the handedness mix
he happened to face. FRAMEWORK.md's "one construct per score" is the standard both are
being measured against.

WHAT THIS DOES NOT SETTLE: whether dropping them is right. A drop that costs validity means
Stuff+ is partly a pitcher-value score rather than a pitch-quality score, which is a
framework decision, not a tuning one. Validity against next season is reported here so the
cost is visible, but nothing is adopted.

SIGN CONVENTION (fair_criterion.py): ridge_pred and the criterion are expected run value
from the pitcher's perspective, LOWER = BETTER, relative to an average pitcher. Scores are
negated ONCE into `_hi` display frame. Correlation of a score with the criterion is
reported in run-value orientation, where NEGATIVE means the score predicts better outcomes.

Data rules: reads workdir caches; writes one JSON to the workdir. Scatter points are two
derived grades per pitcher with NO identifiers. Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coach_model_comparison as cm
import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import fair_criterion as fc

FLOOR = 100
VARIANTS = {
    "full":    [f for f in fc.FEATS],
    "no_lhp":  [f for f in fc.FEATS if f != "is_lhp"],
    "no_hand": [f for f in fc.FEATS if f not in ("is_lhp", "is_lhb")],
    "platoon": [f for f in fc.FEATS] + ["lhp_x_lhb"],
}
LABELS = {"full": "With handedness", "no_lhp": "Without pitcher handedness",
          "no_hand": "Without either handedness",
          "platoon": "With the platoon interaction"}
N_BOOT = 3000


def refit(ff: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Retrain the fixed Stuff+ reference on `feats` and score every pitch.

    Mirrors fair_criterion.stuff_ridge exactly -- Ridge(alpha=10) inside a StandardScaler
    pipeline, TRAINED ON THE 2024-ROLE YEAR ONLY, target = Target -- so the only thing
    varying across variants is the feature set."""
    train = ff[(ff["year"] == 2024) & ff["Target"].notna()]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train[feats].values, train["Target"].values)
    return model.predict(ff[feats].values)


def main() -> int:
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})

    ff = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025").dropna(subset=used + fc.FEATS).copy()
    ff["coach_raw"] = cm.coach_score(ff, terms, 1.0)
    # The platoon effect lives in the INTERACTION, not in either flag: an additive model
    # can price "lefty pitcher" and "lefty batter" but cannot say lefty-on-lefty is
    # especially good, which is the asymmetry that actually exists (and is larger than
    # righty-on-righty). Note the league is majority RHB, so a lefty faces his best
    # matchup least often -- is_lhp alone cannot be credit for a favourable mix.
    ff["lhp_x_lhb"] = ff["is_lhp"].astype(float) * ff["is_lhb"].astype(float)
    for name, feats in VARIANTS.items():
        ff[f"pred_{name}"] = refit(ff, feats)

    # criterion: the pitcher's next-season four-seam run value (unadjusted), per 100
    c = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]

    season = ff[ff["year"] == 2025]
    agg = {"n": ("coach_raw", "size"), "coach_hi": ("coach_raw", "mean"),
           "lhp": ("is_lhp", "mean")}
    for name in VARIANTS:
        agg[name] = (f"pred_{name}", "mean")
    g = season.groupby("PitcherId").agg(**agg)
    g = g[g["n"] >= FLOOR].join(k, how="inner")
    g["crit100"] = g["crit"] * 100
    print(f"  pool: {len(g)} pitchers ({int(g.lhp.round().sum())} left-handed), "
          f"{FLOOR}+ four-seams in both seasons")

    out = {"n": int(len(g)), "n_lhp": int(g.lhp.round().sum()), "floor": FLOOR,
           "variants": {}}
    idx = g.index.values
    rng = np.random.default_rng(19)
    boots = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]

    print(f"\n  {'variant':<28}{'r vs v1':>10}{'validity':>11}{'lhp gap':>10}")
    for name in VARIANTS:
        hi = -g[name]                      # negate once into higher-is-better
        r_v1 = float(fc.R(pd.Series(cp.z(g["coach_hi"].values)),
                          pd.Series(cp.z(hi.values))))
        # Both sides are already in run-value orientation -- ridge_pred is a PREDICTION of
        # run value, not a raw trait -- so a correctly oriented model correlates
        # POSITIVELY here and larger is better. The "negative means better" rule applies
        # to raw traits (more velo -> fewer runs), not to a fitted prediction.
        val = float(fc.R(g[name], g["crit100"]))
        lh = g.lhp.round() == 1
        gap = float(cp.z(hi.values)[lh.values].mean() - cp.z(hi.values)[~lh.values].mean())
        bs = np.array([float(np.corrcoef(cp.z(g.loc[b, "coach_hi"].values),
                                         cp.z(-g.loc[b, name].values))[0, 1])
                       for b in boots])
        print(f"  {LABELS[name]:<28}{r_v1:>10.3f}{val:>11.3f}{gap:>+10.2f}")
        out["variants"][name] = dict(
            label=LABELS[name], features=VARIANTS[name], r_vs_v1=round(r_v1, 4),
            r_vs_v1_se=round(float(bs.std()), 4),
            validity_runvalue=round(val, 4), lhp_minus_rhp_z=round(gap, 4),
            points=[[round(float(a), 2), round(float(b), 2)]
                    for a, b in zip(100 + 15 * cp.z(g["coach_hi"].values),
                                    100 + 15 * cp.z(hi.values))])

    print("\n  paired change in agreement with v1 (same resamples):")
    base = np.array([float(np.corrcoef(cp.z(g.loc[b, "coach_hi"].values),
                                       cp.z(-g.loc[b, "full"].values))[0, 1])
                     for b in boots])
    for name in ("no_lhp", "no_hand"):
        alt = np.array([float(np.corrcoef(cp.z(g.loc[b, "coach_hi"].values),
                                          cp.z(-g.loc[b, name].values))[0, 1])
                        for b in boots])
        fc.boot_report(f"{LABELS[name]} - full", alt - base)
        out["variants"][name]["delta_r_vs_full"] = round(float((alt - base).mean()), 4)
        out["variants"][name]["delta_r_se"] = round(float((alt - base).std()), 4)

    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_handedness.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
