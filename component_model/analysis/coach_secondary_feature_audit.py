"""PRE-REGISTERED secondary-pitch feature audit: CH, SL, CB, SW on the UNCHANGED gate machinery.

Written and fixed before any candidate ran. Motivation: the pitching coach read one changeup /
splitter as graded well below where he ranks it, and asked whether the "vs his fastball"
differentials are the reason. One-pitch patches are not a model change, so the question is
posed as a RULE and asked of every secondary type at once: which feature families carry
construct-valid signal on a secondary pitch, and which are inert or harmful?

WHAT VARIES: only the ridge's feature list, per type. The bar, the criterion (next season's
adjT on the 10%-share / 15-pitch gate pool), the pool, the seed and the bootstrap are the
shipping gate's, verbatim (coach_incremental_gate.py): "base" for each type must replicate its
official row exactly. That replication is the harness check, not a result.

CANDIDATE FAMILIES (all physics; nothing that needs location, batter or usage):
  -diff        drop the three fastball differentials            (the coach's hypothesis)
  -breakdiff   drop the two break differentials, keep velocity_differential
  -dev         drop the coach's release-deviation V terms       (do they carry off the FF?)
  +movgeo      add break-vector direction and magnitude         (the one SI gain, iter 2)
  +movgeo+angsq add the direction-squared sweet-spot term        (SI iter 3)
  -diff+movgeo  the coach's drop and the SI gain together
  +spin (CH)   restore SpinRate, removed 2026-08-17 on the coach's construct reading; run so
               the cost of that decision is a number, NOT a proposal to reverse it.

STATISTICS, all on one shared set of 200 cluster-bootstrap draws so every candidate is PAIRED
with base on the same resample (train pitchers via frequency weights, eval pitchers by set):
  gate       stuff_r, prior-results r, 50/50 z-blend gain over prior, P(gain>0)  -- the gate
  paired     delta stuff_r and delta semipartial (grade | prior results) vs base, mean, 95% CI,
             P(delta>0), P(delta > -0.01)                    -- release-gate decision statistic
  replication  the same delta stuff_r against next season's Target (unadjusted), sign only
  coefficients standardized ridge coefficients of the full fit, pitcher frame (negative = more
             of the feature predicts fewer expected runs, i.e. better)

DECISION RULES (fixed here; the ledger records the verdict, nothing ships from this script):
  ADD arm    recommend iff P(delta stuff_r > 0) >= 0.95 on adjT AND the Target delta agrees in
             sign AND P(delta semipartial > 0) >= 0.95 (the gain survives next to prior
             results, FRAMEWORK.md decomposition audit).
  DROP arm   recommend iff P(delta stuff_r > -0.01) >= 0.95 on adjT (non-inferiority, the
             release-gate margin). A drop that costs measurable validity is reported as such;
             whether an unexplainable feature is worth its validity is a construct call and
             stays with the coach and Jack.
  Anything else: no recommendation. Two-way results (helps one type, hurts another) are
  reported per type; the rule is not forced to be uniform.

Sign convention: adjT, Target and every grade are expected run value, pitcher's perspective,
LOWER = BETTER; a correctly oriented grade correlates POSITIVELY with the criterion.

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

GROUPS = ["CH", "SL", "CB", "SW"]
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95
MARGIN = 0.01
SEED = 20260817          # the gate's seed; base must replicate coach_incremental_gate.json

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()

DIFF = list(fc.DIFF_FEATS)
BREAKDIFF = ["vertbreakdiff", "horzbreakdiff"]
DEV = list(fc.DEV_SRC)
MOVGEO = ["mov_angle", "mov_mag"]
ANGSQ = ["mov_angle_sq"]


def candidates(grp):
    base = fc.feats_for(grp)
    c = {
        "base": base,
        "-diff": [f for f in base if f not in DIFF],
        "-breakdiff": [f for f in base if f not in BREAKDIFF],
        "-dev": [f for f in base if f not in DEV],
        "+movgeo": base + MOVGEO,
        "+movgeo+angsq": base + MOVGEO + ANGSQ,
        "-diff+movgeo": [f for f in base if f not in DIFF] + MOVGEO,
    }
    if grp == "CH":
        c["+spin"] = base + ["SpinRate"]
    return c


ARM = {"-diff": "drop", "-breakdiff": "drop", "-dev": "drop",
       "+movgeo": "add", "+movgeo+angsq": "add", "-diff+movgeo": "add", "+spin": "add"}


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


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j, crit_col="crit"):
    rs = float(fc.R(j["grade"], j[crit_col]))
    rp = float(fc.R(j["prior"], j[crit_col]))
    rb = float(fc.R(_z(j["grade"]) + _z(j["prior"]), j[crit_col]))
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j[crit_col]))
    return rs, rp, rb, rsp


def _ci(a):
    a = np.asarray(a, dtype=float)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return {"mean": round(float(a.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_positive": round(float((a > 0).mean()), 3),
            "p_non_inferior": round(float((a > -MARGIN).mean()), 3)}


def run_group(grp, score, crit, tot, ctot):
    base = add_derived(score[fc.pitch_mask(score, grp)])
    cands = candidates(grp)
    all_feats = sorted({f for fs in cands.values() for f in fs})
    base = base.dropna(subset=all_feats + ["Target"])
    tr = base[(base["year"] == 2024) & base["Target"].notna()]
    ev = base[base["year"] == 2025]
    if len(tr) < 100 or len(ev) < 100:
        # Sweeper is the live case: the tag enters the D1 feed in 2025, so there is no
        # train year for it yet. Nothing to fit; recorded rather than crashed.
        print("  skipped: %d train rows, %d eval rows" % (len(tr), len(ev)), flush=True)
        return {"n_train_rows": int(len(tr)), "n_eval_rows": int(len(ev)),
                "skipped": "no train/eval sample"}

    c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean"),
                                   crit_T=("Target", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

    def build(feats, train_df, eval_df, w=None):
        kw = {"ridge__sample_weight": w} if w is not None else {}
        m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
        m.fit(train_df[feats].values, train_df["Target"].values, **kw)
        p = pd.Series(m.predict(eval_df[feats].values), index=eval_df["PitcherId"].values)
        s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
        g = eval_df.groupby("PitcherId").agg(sn=("adjT", "size"),
                                             prior=("adjT", "mean")).join(tot)
        g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
        j = s.join(g, how="inner").join(k, how="inner").dropna(
            subset=["grade", "prior", "crit", "crit_T"])
        return j, m

    out = {"n_train_rows": int(len(tr)), "by_candidate": {}}
    point = {}
    for name, feats in cands.items():
        j, m = build(feats, tr, ev)
        if len(j) < MIN_PITCHERS:
            out["by_candidate"][name] = {"n": int(len(j)), "skipped": "pool too small"}
            continue
        rs, rp, rb, rsp = _stats(j)
        rsT = _stats(j, "crit_T")[0]
        coef = dict(zip(feats, [round(float(x), 5) for x in m[-1].coef_]))
        point[name] = (rs, rp, rb, rsp, rsT)
        out["by_candidate"][name] = {
            "n": int(len(j)), "arm": ARM.get(name, "base"), "feats": feats,
            "stuff_r": round(rs, 4), "results_r": round(rp, 4), "blend_r": round(rb, 4),
            "blend_gain": round(rb - rp, 4), "semipartial": round(rsp, 4),
            "stuff_r_target": round(rsT, 4), "std_coef_pitcher_frame": coef}
        print("  %-14s n=%4d  stuff_r %+.4f  results_r %+.4f  gain %+.4f  semip %+.4f  "
              "stuff_r(Target) %+.4f" % (name, len(j), rs, rp, rb - rp, rsp, rsT), flush=True)
    if "base" not in point:
        out["skipped"] = "pool too small"
        return out

    names = [n for n in cands if n in point]
    draws = {n: {"gain": [], "rs": [], "rsp": [], "rsT": []} for n in names}
    tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
    ev_ids = ev["PitcherId"].unique()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    for bi in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                          minlength=len(tr_ids)).astype(float)
        w = cnt[tr_codes]
        keep = set(rng.choice(ev_ids, len(ev_ids)))
        evb = ev[ev["PitcherId"].isin(keep)]
        row = {}
        for n in names:
            jb, _ = build(cands[n], tr, evb, w=w)
            if len(jb) < MIN_PITCHERS:
                row = None
                break
            rs, rp, rb, rsp = _stats(jb)
            row[n] = (rs, rp, rb, rsp, _stats(jb, "crit_T")[0])
        if row is None:
            continue
        for n in names:
            rs, rp, rb, rsp, rsT = row[n]
            draws[n]["gain"].append(rb - rp)
            draws[n]["rs"].append(rs)
            draws[n]["rsp"].append(rsp)
            draws[n]["rsT"].append(rsT)
        if bi % 50 == 49:
            print("    %d/%d refits  %.0fs" % (bi + 1, N_BOOT, time.time() - t0), flush=True)

    b_rs, b_rsp, b_rsT = (np.array(draws["base"][k_]) for k_ in ("rs", "rsp", "rsT"))
    for n in names:
        d = out["by_candidate"][n]
        gains = np.array(draws[n]["gain"])
        d["n_boot_used"] = int(len(gains))
        d["gate"] = {"gain_mean": round(float(gains.mean()), 4),
                     "gain_ci": [round(float(x), 4) for x in np.percentile(gains, [2.5, 97.5])],
                     "p_gain_positive": round(float((gains > 0).mean()), 3)}
        d["gate"]["verdict"] = "PASS" if d["gate"]["p_gain_positive"] >= PASS_BAR else "no"
        if n == "base":
            continue
        drs = np.array(draws[n]["rs"]) - b_rs
        drsp = np.array(draws[n]["rsp"]) - b_rsp
        drsT = np.array(draws[n]["rsT"]) - b_rsT
        d["vs_base"] = {"delta_stuff_r": _ci(drs), "delta_semipartial": _ci(drsp),
                        "delta_stuff_r_target": _ci(drsT)}
        arm = ARM[n]
        p_rs, p_sp = d["vs_base"]["delta_stuff_r"]["p_positive"], \
            d["vs_base"]["delta_semipartial"]["p_positive"]
        sign_ok = np.sign(point[n][4] - point["base"][4]) == np.sign(point[n][0] - point["base"][0])
        if arm == "add":
            rec = bool(p_rs >= PASS_BAR and p_sp >= PASS_BAR and sign_ok)
            d["recommend"] = "ADD" if rec else "no"
        else:
            rec = bool(d["vs_base"]["delta_stuff_r"]["p_non_inferior"] >= PASS_BAR)
            d["recommend"] = "DROP-OK" if rec else "no (costs validity)"
        v = d["vs_base"]
        print("  %-14s d.stuff_r %+.4f [%+.4f,%+.4f] P>0 %.3f P>-.01 %.3f | d.semip %+.4f P>0 %.3f"
              " | d.Target %+.4f | gate P %.3f | %s -> %s"
              % (n, v["delta_stuff_r"]["mean"], *v["delta_stuff_r"]["ci"],
                 v["delta_stuff_r"]["p_positive"], v["delta_stuff_r"]["p_non_inferior"],
                 v["delta_semipartial"]["mean"], v["delta_semipartial"]["p_positive"],
                 v["delta_stuff_r_target"]["mean"], d["gate"]["p_gain_positive"], arm,
                 d["recommend"]), flush=True)
    return out


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    dest = os.path.join(SCORE_WORKDIR, "coach_secondary_feature_audit.json")
    out = {"share": SHARE, "abs_min": ABS_MIN, "n_boot": N_BOOT, "pass_bar": PASS_BAR,
           "margin": MARGIN, "seed": SEED, "by_pitch": {}}
    # AUDIT_GROUPS (comma list) reruns a subset and merges into an existing JSON, so one
    # type can be redone without repeating the others' 200 refits.
    groups = GROUPS
    if os.environ.get("AUDIT_GROUPS"):
        groups = os.environ["AUDIT_GROUPS"].split(",")
        if os.path.exists(dest):
            with open(dest) as fh:
                out = json.load(fh)
    for grp in groups:
        print("")
        print("=== %s ===" % grp, flush=True)
        out["by_pitch"][grp] = run_group(grp, score, crit, tot, ctot)
        with open(dest, "w") as fh:
            json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
