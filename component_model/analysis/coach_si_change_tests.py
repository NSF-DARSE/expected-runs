"""Two tests that decide the sinker development story. Neither touches the shipping gate.

TEST A -- CHANGE-ON-CHANGE. "Your sinker is improving" is a within-pitcher claim: when a
pitcher's physics move, do his concurrent results move with them? That is a different
statistic from the cross-pitcher incremental gate (which SI fails), and it is the one that
licenses trend arrows on a development panel. For each pitcher with the pitch in both 2024
and 2025: dg = mean ridge grade 2025 - 2024 (model trained on 2024, applied to both years),
da = mean adjT 2025 - 2024. Report r(dg, da) with a pitcher-bootstrap CI. FF runs as the
positive control: if the method is sound it must show there first. SI runs on the shipping
feature set and on the movement-geometry set from coach_si_feature_gate.py.

Orientation: grades and adjT are both expected run value, lower = better, so a POSITIVE
r(dg, da) means physics improvement travels with results improvement.

TEST B -- GB-RESIDUAL KILL TEST. The criterion-widening premise is that pitcher skill hides
inside ground balls the pooled EV/LA map averages away. Direct check: per ground ball,
resid = Target - xT (what actually happened minus what the map paid for that EV/LA). If a
pitcher's mean GB residual repeats year over year, the map is smoothing away real skill and
a GB-conditioned criterion has something to recover. If it does not repeat, the residual is
defense/spray luck as designed, and widening the criterion cannot help. Run on all pitches
pooled (max sample) and on SI only.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fair_criterion as fc

FLOORS = [25, 40]
N_BOOT = 2000

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def add_derived(df):
    df = df.copy()
    df["RelSide_arm"] = df["RelSide"] * (1 - 2 * df["is_lhp"])
    df["HorzBreak_arm"] = df["HorzBreak"] * (1 - 2 * df["is_lhp"])
    for o, s_ in fc.DEV_SRC.items():
        df[o] = (df[s_] - df["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
    df["mov_angle"] = np.degrees(np.arctan2(df["HorzBreak_arm"], df["InducedVertBreak"]))
    df["mov_mag"] = np.hypot(df["HorzBreak_arm"], df["InducedVertBreak"])
    df["mov_angle_sq"] = df["mov_angle"] ** 2
    return df


def boot_ci_r(x, y, rng, n_boot=N_BOOT):
    """Pearson r with a pitcher-resample bootstrap CI (x, y aligned per pitcher)."""
    r = float(fc.R(pd.Series(x), pd.Series(y)))
    n = len(x)
    rs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        xi, yi = x[i], y[i]
        if np.std(xi) == 0 or np.std(yi) == 0:
            continue
        rs.append(np.corrcoef(xi, yi)[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, float(lo), float(hi), float((np.array(rs) > 0).mean())


def change_on_change(base, feats, floor, rng):
    """r(delta grade, delta adjT) on pitchers with >= floor pitches in both years."""
    b = base.dropna(subset=feats + ["adjT"])
    tr = b[(b["year"] == 2024) & b["Target"].notna()]
    m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    m.fit(tr[feats].values, tr["Target"].values)
    b = b.assign(pred=m.predict(b[feats].values))
    per = b.groupby(["PitcherId", "year"]).agg(
        n=("adjT", "size"), g=("pred", "mean"), a=("adjT", "mean"))
    per = per[per["n"] >= floor].reset_index()
    w = per.pivot(index="PitcherId", columns="year", values=["g", "a"]).dropna()
    if len(w) < 30:
        return {"n": int(len(w)), "skipped": "panel too small"}
    dg = (w[("g", 2025)] - w[("g", 2024)]).values
    da = (w[("a", 2025)] - w[("a", 2024)]).values
    r, lo, hi, p_pos = boot_ci_r(dg, da, rng)
    return {"n": int(len(w)), "r": round(r, 4), "ci": [round(lo, 4), round(hi, 4)],
            "p_positive": round(p_pos, 4)}


def gb_residual_yoy(df, floor, rng, mask=None):
    """YoY reliability of pitcher mean (Target - xT) on ground balls."""
    gb = df[df["is_inplay"] & (df["TaggedHitType"] == "GroundBall")
            & df["Target"].notna() & df["xT"].notna()]
    if mask is not None:
        gb = gb[mask.reindex(gb.index, fill_value=False)]
    gb = gb.assign(resid=gb["Target"] - gb["xT"])
    per = gb.groupby(["PitcherId", "year"]).agg(n=("resid", "size"),
                                                m=("resid", "mean"))
    per = per[per["n"] >= floor].reset_index()
    w = per.pivot(index="PitcherId", columns="year", values="m").dropna()
    if len(w) < 30:
        return {"n": int(len(w)), "skipped": "panel too small"}
    r, lo, hi, p_pos = boot_ci_r(w[2024].values, w[2025].values, rng)
    return {"n": int(len(w)), "r": round(r, 4), "ci": [round(lo, 4), round(hi, 4)],
            "p_positive": round(p_pos, 4)}


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    print("  frame loaded in %.0fs" % (time.time() - t0), flush=True)
    rng = np.random.default_rng(20260821)
    out = {"n_boot": N_BOOT, "change_on_change": {}, "gb_residual_yoy": {}}

    MOVGEO = ["mov_angle", "mov_mag", "mov_angle_sq"]
    runs = [("FF", "base", fc.feats_for("FF")),
            ("SI", "base", fc.feats_for("SI")),
            ("SI", "base+movgeo", fc.feats_for("SI") + MOVGEO)]
    print("")
    print("=== TEST A: change-on-change, r(delta grade, delta adjT), 2024 -> 2025 ===")
    for grp, name, feats in runs:
        base = add_derived(score[fc.pitch_mask(score, grp)])
        for floor in FLOORS:
            res = change_on_change(base, feats, floor, rng)
            key = "%s/%s/floor%d" % (grp, name, floor)
            out["change_on_change"][key] = res
            if "skipped" in res:
                print("    %-28s n=%d  skipped" % (key, res["n"]))
            else:
                print("    %-28s n=%4d  r=%+.3f  CI [%+.3f,%+.3f]  P(r>0)=%.3f"
                      % (key, res["n"], res["r"], res["ci"][0], res["ci"][1],
                         res["p_positive"]))

    print("")
    print("=== TEST B: GB residual (Target - xT) YoY reliability ===")
    si_mask = fc.pitch_mask(score, "SI")
    for label, mask in [("all_pitches", None), ("SI_only", si_mask)]:
        for floor in ([15, 25, 40] if label == "SI_only" else FLOORS):
            res = gb_residual_yoy(score, floor, rng, mask)
            key = "%s/floor%d" % (label, floor)
            out["gb_residual_yoy"][key] = res
            if "skipped" in res:
                print("    %-28s n=%d  skipped" % (key, res["n"]))
            else:
                print("    %-28s n=%4d  r=%+.3f  CI [%+.3f,%+.3f]  P(r>0)=%.3f"
                      % (key, res["n"], res["r"], res["ci"][0], res["ci"][1],
                         res["p_positive"]))

    dest = os.path.join(SCORE_WORKDIR, "coach_si_change_tests.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
