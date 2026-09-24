"""PRE-REGISTERED shadow analysis: what would a ground-ball-conditioned criterion change?

The frozen fair criterion values every ball in play through one pooled EV/LA map (fine 5 mph x
10 deg, coarse fallback). The sinker loop (docs/notes/sinker-cutter-loop-ledger.md) concluded
that the skill making sinker results repeat lives largely in ground balls that this map values
identically, and named a GB-conditioned contact valuation as path (a) -- a CRITERION change,
which is frozen and stays frozen. The post-loop kill test (coach_si_change_tests.py, TEST B)
then found the pitcher-mean GB residual (Target - xT) does not repeat year over year at this
sample, which bounds what any GB-conditioned map can recover.

This script is a SHADOW: it evaluates two alternative criteria on all six pitch types at once,
blind, so the question "would reopening the criterion be worth it" has numbers when Jack decides.
NOTHING here changes fair_criterion, the gate, or any shipped score. No shadow result is adopted.

SHADOW CRITERIA (both leave non-GB pitches exactly as the frozen criterion values them):
  gb_realized  ground balls keep their realized Target instead of the map value. This is the
               CEILING of any GB-conditioned map: it credits the entire GB residual as skill,
               defense and spray luck included. If even this does not help, no GB map can.
  gb_fine      ground balls valued on a GB-only EV/LA grid at 2.5 mph x 5 deg (cell n >= 30,
               fallback to the frozen value). A map that actually separates one ground ball
               from another on the fields we have; a realistic implementation, not a ceiling.
Each shadow xT then passes through the SAME opponent adjustment (fc.add_adjusted) to a shadow
adjT, on both frames (score 2024/2025 and criterion 2025/2026), so prior results and the
criterion are redefined consistently.

READOUT RULE, fixed in advance, read in this order:
  1. Eligible types first (FF, SL, CB, CH). A criterion change is carried forward for discussion
     ONLY if, on each of them, it does not reduce criterion reliability (results_r) or shipping-
     grade validity (stuff_r) by more than 0.01 with P >= 0.95 (paired non-inferiority on the
     shared bootstrap draws). A criterion that helps sinkers by degrading the four types that
     already work is not a better criterion.
  2. Only then SI and FC: delta stuff_r, delta results_r, delta blend gain, and the gate
     P(gain>0) under the shadow. Reported as deltas against the frozen criterion on the same
     draws, never as a standalone pass/fail -- a shadow verdict is not a gate verdict.
The grade itself never changes: the ridge trains on realized Target with the shipping feature
list, exactly as deployed. Only what it is measured against varies.

Sign convention: Target, xT, adjT and every shadow variant are expected run value, pitcher's
perspective, LOWER = BETTER; a correctly oriented grade correlates POSITIVELY with each.

Data rules: reads workdir caches only (STUFFPLUS_DATA, STUFFPLUS_WORKDIR, STUFFPLUS_WORKDIR_CRIT);
writes one JSON to the score workdir. No pitcher names, no per-pitcher output, no absolute paths.
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

GROUPS = ["FF", "SL", "CB", "CH", "SI", "FC"]     # readout order: eligible types first
CRITERIA = ["frozen", "gb_realized", "gb_fine"]
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95
MARGIN = 0.01
SEED = 20260817
GB_EV_BIN, GB_LA_BIN, GB_MIN_CELL = 2.5, 5.0, 30

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def gb_mask(df):
    return df["is_inplay"] & (df["TaggedHitType"] == "GroundBall") & df["Target"].notna()


def add_shadow_criteria(df):
    """Adds adjT_gb_realized and adjT_gb_fine in place; adjT_frozen aliases adjT."""
    gb = gb_mask(df)
    # gb_realized: the ceiling.
    xt1 = df["xT"].copy()
    xt1[gb] = df.loc[gb, "Target"]
    # gb_fine: GB-only fine grid, fallback to the frozen value.
    has = gb & df["ExitSpeed"].notna() & df["Angle"].notna()
    src = df[has]
    e = (np.floor(src["ExitSpeed"] / GB_EV_BIN) * GB_EV_BIN)
    l = (np.floor(src["Angle"] / GB_LA_BIN) * GB_LA_BIN)
    g = src.assign(e=e, l=l).groupby(["e", "l"])["Target"].agg(["mean", "count"])
    grid = g[g["count"] >= GB_MIN_CELL]["mean"]
    v = pd.Series(list(zip(e, l)), index=src.index).map(grid)
    xt2 = df["xT"].copy()
    xt2[has] = v.fillna(df.loc[has, "xT"])
    covered = float(v.notna().mean()) if len(v) else 0.0
    for name, xt in (("gb_realized", xt1), ("gb_fine", xt2)):
        slim = df[["League", "Batter", "BatterTeam"]].copy()
        slim["xT"] = xt.values
        fc.add_adjusted(slim)
        df["adjT_" + name] = slim["adjT"].values
    df["adjT_frozen"] = df["adjT"]
    return {"gb_rows": int(gb.sum()), "gb_fine_cells": int(len(grid)),
            "gb_fine_coverage": round(covered, 4)}


def add_derived(df):
    df = df.copy()
    df["RelSide_arm"] = df["RelSide"] * (1 - 2 * df["is_lhp"])
    df["HorzBreak_arm"] = df["HorzBreak"] * (1 - 2 * df["is_lhp"])
    for o, s_ in fc.DEV_SRC.items():
        df[o] = (df[s_] - df["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
    return df


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j, c):
    rs = float(fc.R(j["grade"], j["crit_" + c]))
    rp = float(fc.R(j["prior_" + c], j["crit_" + c]))
    rb = float(fc.R(_z(j["grade"]) + _z(j["prior_" + c]), j["crit_" + c]))
    return rs, rp, rb


def _ci(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"mean": round(float(a.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_positive": round(float((a > 0).mean()), 3),
            "p_non_inferior": round(float((a > -MARGIN).mean()), 3)}


def run_group(grp, score, crit, tot, ctot):
    feats = fc.feats_for(grp)
    base = add_derived(score[fc.pitch_mask(score, grp)]).dropna(subset=feats + ["Target"])
    tr = base[base["year"] == 2024]
    ev = base[base["year"] == 2025]
    c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
    agg = {"cn": ("adjT", "size")}
    agg.update({"crit_" + k: ("adjT_" + k, "mean") for k in CRITERIA})
    k = c.groupby("PitcherId").agg(**agg).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

    def build(train_df, eval_df, w=None):
        kw = {"ridge__sample_weight": w} if w is not None else {}
        m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
        m.fit(train_df[feats].values, train_df["Target"].values, **kw)
        p = pd.Series(m.predict(eval_df[feats].values), index=eval_df["PitcherId"].values)
        s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
        a2 = {"sn": ("adjT", "size")}
        a2.update({"prior_" + kk: ("adjT_" + kk, "mean") for kk in CRITERIA})
        g = eval_df.groupby("PitcherId").agg(**a2).join(tot)
        g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
        return s.join(g, how="inner").join(k, how="inner").dropna()

    j = build(tr, ev)
    out = {"n": int(len(j)), "feats": feats, "by_criterion": {}}
    if len(j) < MIN_PITCHERS:
        out["skipped"] = "pool too small"
        return out
    for cr in CRITERIA:
        rs, rp, rb = _stats(j, cr)
        out["by_criterion"][cr] = {"stuff_r": round(rs, 4), "results_r": round(rp, 4),
                                   "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4)}
        print("  %-12s n=%4d  stuff_r %+.4f  results_r %+.4f  gain %+.4f"
              % (cr, len(j), rs, rp, rb - rp), flush=True)

    draws = {cr: {"rs": [], "rp": [], "gain": []} for cr in CRITERIA}
    tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
    ev_ids = ev["PitcherId"].unique()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for bi in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                          minlength=len(tr_ids)).astype(float)
        w = cnt[tr_codes]
        keep = set(rng.choice(ev_ids, len(ev_ids)))
        jb = build(tr, ev[ev["PitcherId"].isin(keep)], w=w)
        if len(jb) < MIN_PITCHERS:
            continue
        for cr in CRITERIA:
            rs, rp, rb = _stats(jb, cr)
            draws[cr]["rs"].append(rs)
            draws[cr]["rp"].append(rp)
            draws[cr]["gain"].append(rb - rp)
        if bi % 50 == 49:
            print("    %d/%d refits  %.0fs" % (bi + 1, N_BOOT, time.time() - t0), flush=True)
    f_rs, f_rp, f_gain = (np.array(draws["frozen"][k_]) for k_ in ("rs", "rp", "gain"))
    for cr in CRITERIA:
        d = out["by_criterion"][cr]
        gains = np.array(draws[cr]["gain"])
        d["n_boot_used"] = int(len(gains))
        d["p_gain_positive"] = round(float((gains > 0).mean()), 3)
        d["gate_verdict_under_this_criterion"] = "PASS" if d["p_gain_positive"] >= PASS_BAR else "no"
        if cr == "frozen":
            continue
        d["vs_frozen"] = {"delta_stuff_r": _ci(np.array(draws[cr]["rs"]) - f_rs),
                          "delta_results_r": _ci(np.array(draws[cr]["rp"]) - f_rp),
                          "delta_blend_gain": _ci(gains - f_gain)}
        v = d["vs_frozen"]
        d["non_inferior_on_this_type"] = bool(
            v["delta_stuff_r"]["p_non_inferior"] >= PASS_BAR
            and v["delta_results_r"]["p_non_inferior"] >= PASS_BAR)
        print("  %-12s d.stuff_r %+.4f [%+.4f,%+.4f] P>-.01 %.3f | d.results_r %+.4f P>-.01 %.3f"
              " | d.gain %+.4f P>0 %.3f | gate P %.3f | non-inferior: %s"
              % (cr, v["delta_stuff_r"]["mean"], *v["delta_stuff_r"]["ci"],
                 v["delta_stuff_r"]["p_non_inferior"], v["delta_results_r"]["mean"],
                 v["delta_results_r"]["p_non_inferior"], v["delta_blend_gain"]["mean"],
                 v["delta_blend_gain"]["p_positive"], d["p_gain_positive"],
                 d["non_inferior_on_this_type"]), flush=True)
    return out


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    meta = {"score_frame": add_shadow_criteria(score), "crit_frame": add_shadow_criteria(crit)}
    print("  shadow criteria built: %s" % json.dumps(meta), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    out = {"share": SHARE, "abs_min": ABS_MIN, "n_boot": N_BOOT, "pass_bar": PASS_BAR,
           "margin": MARGIN, "seed": SEED, "gb_grid": [GB_EV_BIN, GB_LA_BIN, GB_MIN_CELL],
           "shadow_build": meta, "by_pitch": {}}
    dest = os.path.join(SCORE_WORKDIR, "coach_gb_criterion_shadow.json")
    for grp in GROUPS:
        print("")
        print("=== %s ===" % grp, flush=True)
        out["by_pitch"][grp] = run_group(grp, score, crit, tot, ctot)
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=1)
    elig = [g for g in ("FF", "SL", "CB", "CH") if "by_criterion" in out["by_pitch"][g]]
    out["readout"] = {cr: {"carry_forward": all(
        out["by_pitch"][g]["by_criterion"][cr]["non_inferior_on_this_type"] for g in elig)}
        for cr in CRITERIA if cr != "frozen"}
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  readout (step 1, eligible types non-inferior under shadow): %s"
          % json.dumps(out["readout"]))
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
