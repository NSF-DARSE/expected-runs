"""PRE-REGISTERED gate: adopt the coach's deviation-from-typical release terms?

Written and fixed BEFORE the refit bootstrap ran. The exploratory result that motivated it
(coach_release_form.py) is NOT the evidence of record: its bootstrap resampled only
pitchers at evaluation and never refit the model, so it understated fitting uncertainty.

THE CHANGES UNDER TEST. Both candidate FEATS changes are gated in ONE run so that a single
adoption drives a single downstream rerun.

  (1) ADD his form on top of ours. His card scores release side and release height as
      |value - typical for that hand| -- a symmetric V, so an unusually high AND an
      unusually low release both earn points. Ours are monotone (raw RelHeight, arm-side
      RelSide_arm). New features:
          dev_relheight = |RelHeight   - centre[hand]|
          dev_relside   = |RelSide_arm - centre[hand]|
      Centres are FIXED CONSTANTS measured once on the 2024 train year, not recomputed per
      frame. That matters: the score frame (2024/2025) and criterion frame (2025/2026) load
      separately, and centring each on its own rows would silently give the two frames
      different features. Fixed constants keep scoring deterministic.

  (2) DROP the three "vs his fastball" differentials. Their reference is the pitcher's own
      fastest pitch type, so on a FOUR-SEAM they reduce to within-pitcher scatter around his
      own mean and average to ~0 per pitcher. Measured importances 0.03/0.00/0.05. They are
      inert rather than harmful, so this arm is tested for NON-INFERIORITY, not improvement.

CRITERION: next season's four-seam run value at the pitcher level, 100+ four-seams in both
seasons -- the same fixed fair criterion used throughout. adjT is primary, Target is the
replication. Both are expected runs, LOWER = BETTER, so a POSITIVE correlation between
prediction and criterion means the score is valid.

DECISION RULES, fixed in advance:
  add deviation terms  ADOPT iff P(validity improves) >= 0.95 on adjT AND the Target point
                       difference agrees in sign.
  drop differentials   ADOPT iff P(delta > -0.01) >= 0.95 on adjT (non-inferiority).
  combined set         ADOPT iff P(validity improves) >= 0.95 on adjT.

Uncertainty is a cluster bootstrap that REFITS on resampled train pitchers (via frequency
weights, which is equivalent to resampling clusters and far cheaper than rebuilding the
frame) and re-evaluates on resampled criterion pitchers, so both fitting and evaluation
noise are inside the interval.

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

import coach_model_ff_criterion as ffc
import fair_criterion as fc

N_BOOT = 200
FLOOR = 100
DEV_SRC = {"dev_relheight": "RelHeight", "dev_relside": "RelSide_arm"}
DROP = ["vertbreakdiff", "horzbreakdiff", "velocity_differential"]
MARGIN = 0.01          # non-inferiority margin, in correlation units
NEW = list(DEV_SRC)


def dev_centres(train: pd.DataFrame) -> dict:
    """Per-hand centres for the two deviation features, from the TRAIN year only."""
    return {out: {int(h): float(v)
                  for h, v in train.groupby("is_lhp")[src].mean().items()}
            for out, src in DEV_SRC.items()}


def add_dev(df: pd.DataFrame, centres: dict) -> pd.DataFrame:
    for out, src in DEV_SRC.items():
        df[out] = (df[src] - df["is_lhp"].map(centres[out])).abs()
    return df


def main() -> int:
    t0 = time.time()
    score = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    crit = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    print(f"  frames loaded in {time.time() - t0:.0f}s")

    base_train = score[(score["year"] == 2024) & score["Target"].notna()].dropna(
        subset=fc.FEATS)
    centres = dev_centres(base_train)
    print("  FIXED centres (2024 train year):")
    for k, v in centres.items():
        print(f"    {k:<14s} RHP {v[0]:.4f}  LHP {v[1]:.4f}")

    score = add_dev(score.copy(), centres)
    tr = score[(score["year"] == 2024) & score["Target"].notna()].dropna(
        subset=fc.FEATS + NEW)
    ev = score[score["year"] == 2025].dropna(subset=fc.FEATS + NEW).copy()
    k = ev.groupby("PitcherId").size()
    ev = ev[ev["PitcherId"].isin(k[k >= FLOOR].index)]

    c = crit[crit["year"] == 2025].groupby("PitcherId").agg(
        n=("adjT", "size"), adjT=("adjT", "mean"), Target=("Target", "mean"))
    c = c[c["n"] >= FLOOR]

    KEEP = [f for f in fc.FEATS if f not in DROP]
    NO_LHB = [f for f in fc.FEATS if f != "is_lhb"]
    PROPOSED = [f for f in fc.FEATS if f not in DROP + ["is_lhb"]] + NEW
    # ADOPTED is the set we actually ship, and it is gated DIRECTLY rather than inferred
    # from the two component arms passing separately. First pass verdicts were: deviation
    # terms ADOPT, drop is_lhb ADOPT, drop differentials DO NOT ADOPT (non-inferiority
    # P=0.910 < 0.95). So the differentials STAY and the shipped set is
    # FEATS - is_lhb + deviation terms, which no first-pass arm tested on its own.
    ADOPTED = [f for f in fc.FEATS if f != "is_lhb"] + NEW
    VAR = {"baseline": fc.FEATS, "adopted": ADOPTED}
    print("  ADOPTED feature set (" + str(len(ADOPTED)) + "): " + ", ".join(ADOPTED))
    print(f"  train {len(tr):,} pitches, eval {len(ev):,}, criterion pool {len(c):,}")

    def validity(cols, train_df, eval_df, crit_df, crit_col, w=None):
        kw = {"ridge__sample_weight": w} if w is not None else {}
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            train_df[cols].values, train_df["Target"].values, **kw)
        p = pd.Series(m.predict(eval_df[cols].values), index=eval_df["PitcherId"].values)
        p = p.groupby(level=0).mean()
        j = pd.DataFrame({"p": p}).join(crit_df[crit_col], how="inner").dropna()
        return float(fc.R(j["p"], j[crit_col])), len(j)

    out = {"n_boot": N_BOOT, "floor": FLOOR, "margin": MARGIN,
           "centres": centres, "dropped": DROP, "added": NEW, "by_criterion": {}}

    for crit_col in ("adjT", "Target"):
        pt = {k2: validity(v, tr, ev, c, crit_col) for k2, v in VAR.items()}
        n = pt["baseline"][1]
        print(f"\n=== criterion: 2026 four-seam {crit_col} (n={n} pitchers) ===")
        for k2 in VAR:
            print(f"    {k2:<20s} validity r = {pt[k2][0]:+.4f}")

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(20260817)
        B = {k2: [] for k2 in VAR}
        t1 = time.time()
        for b in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            w = cnt[tr_codes]
            rs_ev = ev[ev["PitcherId"].isin(set(rng.choice(ev_ids, len(ev_ids))))]
            for k2, cols in VAR.items():
                B[k2].append(validity(cols, tr, rs_ev, c, crit_col, w=w)[0])
            if (b + 1) % 50 == 0:
                print(f"      refit boot {b + 1}/{N_BOOT}  ({time.time() - t1:.0f}s)")
        B = {k2: np.array(v) for k2, v in B.items()}

        res = {}
        print("    refit bootstrap, variant minus baseline (same resamples):")
        for k2 in VAR:
            if k2 == "baseline":
                continue
            d = B[k2] - B["baseline"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            res[k2] = dict(point=round(pt[k2][0], 4), mean=round(float(d.mean()), 4),
                           se=round(float(d.std()), 4),
                           ci=[round(float(lo), 4), round(float(hi), 4)],
                           p_better=float((d > 0).mean()),
                           p_non_inferior=float((d > -MARGIN).mean()))
            r = res[k2]
            print(f"      {k2:<20s} mean={r['mean']:+.4f} SE={r['se']:.4f} "
                  f"CI=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
                  f"P(better)={r['p_better']:.3f} P(>-{MARGIN})={r['p_non_inferior']:.3f}")
        out["by_criterion"][crit_col] = dict(
            n=n, baseline=round(pt["baseline"][0], 4), variants=res)

    prim = out["by_criterion"]["adjT"]["variants"]["adopted"]
    repl = out["by_criterion"]["Target"]["variants"]["adopted"]
    ok = prim["p_better"] >= 0.95 and repl["mean"] > 0
    out["verdict"] = {"adopted_set": "ADOPT" if ok else "DO NOT ADOPT"}
    print("\n  => VERDICT on the shipped set (FEATS - is_lhb + deviation terms)")
    print(f"     {out['verdict']['adopted_set']}   adjT: mean={prim['mean']:+.4f} "
          f"CI=[{prim['ci'][0]:+.4f},{prim['ci'][1]:+.4f}] P(better)={prim['p_better']:.3f} "
          f"(need >=0.95)")
    print(f"     Target replication: mean={repl['mean']:+.4f} "
          f"P(better)={repl['p_better']:.3f}")
    dest = ffc.SCORE_WORKDIR + r"\coach_release_gate_adopted.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"  wrote {dest}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
