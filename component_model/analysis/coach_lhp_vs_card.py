"""Does our grade still beat his card WITHOUT the pitcher-handedness flag?

The question a coach will ask, and the only version of it that settles anything: not "how
much is is_lhp worth" but "if you take it out, are you still ahead of me?" So all three
scores are scored on ONE pool against ONE criterion, and the two differences that matter
are bootstrapped as paired differences on the same resamples:

    shipped  - his card     does our grade beat his at all
    no_lhp   - his card     does it still beat his with handedness removed

FRAMING NOTE, because the premise of the question is not quite right: his card ALREADY
accounts for handedness. Every term on it is per-hand, with its own average and its own
direction (coach_model_comparison.load_coach_terms), so a lefty is scored against lefty
norms. Our is_lhp is not "we know about handedness and he doesn't" -- it is an extra
lefty/righty LEVEL shift on top of geometry that is already in the arm-side frame. Both
models handle handedness; only ours prices a residual.

Bootstrap refits the ridge on resampled train pitchers and re-scores on resampled criterion
pitchers, matching coach_release_gate.py so numbers are comparable to the adoption evidence.
His card has nothing to fit, so it is re-scored on the same resampled pitchers.

Data rules: reads workdir caches only; writes JSON to the workdir. No pitcher names.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coach_model_comparison as cm
import coach_model_ff_criterion as ffc
import fair_criterion as fc

N_BOOT = 200
FLOOR = 100


def main() -> int:
    t0 = time.time()
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})
    score = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    crit = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    print(f"  frames loaded in {time.time() - t0:.0f}s")

    score = score.dropna(subset=sorted(set(fc.FEATS) | set(used))).copy()
    # his card, per pitch, in run-value orientation (lower = better) to match ridge_pred
    score["coach_pred"] = -cm.coach_score(score, terms, 1.0)

    tr = score[(score["year"] == 2024) & score["Target"].notna()]
    ev = score[score["year"] == 2025].copy()
    k = ev.groupby("PitcherId").size()
    ev = ev[ev["PitcherId"].isin(k[k >= FLOOR].index)]
    c = crit[crit["year"] == 2025].groupby("PitcherId").agg(
        n=("adjT", "size"), adjT=("adjT", "mean"), Target=("Target", "mean"))
    c = c[c["n"] >= FLOOR]

    RIDGE = {"shipped": list(fc.FEATS),
             "no_lhp": [f for f in fc.FEATS if f != "is_lhp"]}

    def ridge_validity(cols, train_df, eval_df, crit_col, w=None):
        kw = {"ridge__sample_weight": w} if w is not None else {}
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            train_df[cols].values, train_df["Target"].values, **kw)
        p = pd.Series(m.predict(eval_df[cols].values), index=eval_df["PitcherId"].values)
        return _r(p.groupby(level=0).mean(), crit_col)

    def card_validity(eval_df, crit_col):
        p = pd.Series(eval_df["coach_pred"].values, index=eval_df["PitcherId"].values)
        return _r(p.groupby(level=0).mean(), crit_col)

    def _r(per_p, crit_col):
        j = pd.DataFrame({"p": per_p}).join(c[crit_col], how="inner").dropna()
        return float(fc.R(j["p"], j[crit_col])), len(j)

    # ---- is handedness LURKING inside the other features? Three readings, cheapest first.
    # (a) Do lefties and righties differ on each feature? Standardised mean difference, so
    #     it is comparable across features regardless of units.
    # (b) How predictable is is_lhp FROM the other features? If R^2 is high, the flag is
    #     largely redundant and the other coefficients are already carrying handedness.
    # (c) Do the other coefficients MOVE when is_lhp is removed? That is the direct test of
    #     absorption: a feature whose coefficient jumps was standing in for handedness.
    print("\n=== is pitcher handedness lurking in the other features? ===")
    others = [f for f in fc.FEATS if f != "is_lhp"]
    tr_c = tr.dropna(subset=fc.FEATS)
    lhp = tr_c["is_lhp"].astype(bool)
    print(f"  (a) standardised mean difference, LHP minus RHP (|d| > 0.2 is a real gap):")
    smd = {}
    for f in others:
        a, b = tr_c.loc[lhp, f], tr_c.loc[~lhp, f]
        sd = tr_c[f].std()
        smd[f] = float((a.mean() - b.mean()) / sd) if sd else 0.0
        flag = "  <-- gap" if abs(smd[f]) > 0.2 else ""
        print(f"      {f:<20s} {smd[f]:+.3f}{flag}")

    from sklearn.linear_model import LinearRegression
    Xo = StandardScaler().fit_transform(tr_c[others].values)
    r2 = float(LinearRegression().fit(Xo, tr_c["is_lhp"].values).score(
        Xo, tr_c["is_lhp"].values))
    print(f"  (b) R^2 predicting is_lhp from the other {len(others)} features: {r2:.3f}"
          f"  (1.000 would mean the flag is pure redundancy)")

    def coefs(cols):
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            tr_c[cols].values, tr_c["Target"].values)
        return dict(zip(cols, m[-1].coef_))
    cw, cwo = coefs(list(fc.FEATS)), coefs(others)
    print("  (c) coefficient shift when is_lhp is removed (standardised units):")
    shifts = {}
    for f in others:
        shifts[f] = float(cwo[f] - cw[f])
        rel = abs(shifts[f]) / (abs(cw[f]) or 1e-12)
        flag = "  <-- absorbs handedness" if rel > 0.25 else ""
        print(f"      {f:<20s} {cw[f]:+.5f} -> {cwo[f]:+.5f}  ({shifts[f]:+.5f}){flag}")

    out = {"n_boot": N_BOOT, "floor": FLOOR,
           "confound": dict(smd=smd, r2_is_lhp_from_others=round(r2, 4),
                            coef_with={k: round(v, 6) for k, v in cw.items()},
                            coef_without={k: round(v, 6) for k, v in cwo.items()}),
           "by_criterion": {}}
    for crit_col in ("adjT", "Target"):
        pt = {k2: ridge_validity(v, tr, ev, crit_col) for k2, v in RIDGE.items()}
        pt["card"] = card_validity(ev, crit_col)
        print(f"\n=== criterion 2026 four-seam {crit_col} (n={pt['card'][1]}) ===")
        for k2 in ("card", "no_lhp", "shipped"):
            print(f"    {k2:<10s} validity r = {pt[k2][0]:+.4f}")

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(20260817)
        B = {k2: [] for k2 in ("shipped", "no_lhp", "card")}
        for b in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            w = cnt[tr_codes]
            rs_ev = ev[ev["PitcherId"].isin(set(rng.choice(ev_ids, len(ev_ids))))]
            for k2, cols in RIDGE.items():
                B[k2].append(ridge_validity(cols, tr, rs_ev, crit_col, w=w)[0])
            B["card"].append(card_validity(rs_ev, crit_col)[0])
            if (b + 1) % 100 == 0:
                print(f"      refit boot {b + 1}/{N_BOOT}")
        B = {k2: np.array(v) for k2, v in B.items()}
        res = {}
        for k2 in ("shipped", "no_lhp"):
            d = B[k2] - B["card"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            res[k2] = dict(point=round(pt[k2][0], 4), mean=round(float(d.mean()), 4),
                           se=round(float(d.std()), 4),
                           ci=[round(float(lo), 4), round(float(hi), 4)],
                           p_beats_card=float((d > 0).mean()))
            print(f"    {k2} - his card: mean={d.mean():+.4f} SE={d.std():.4f} "
                  f"CI=[{lo:+.4f},{hi:+.4f}] P(beats card)={res[k2]['p_beats_card']:.3f}")
        out["by_criterion"][crit_col] = dict(n=pt["card"][1],
                                             card=round(pt["card"][0], 4), variants=res)

    dest = ffc.SCORE_WORKDIR + r"\coach_lhp_vs_card.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote {dest}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
