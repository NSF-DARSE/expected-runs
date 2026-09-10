"""Pooled fastball model and secondary-sinker differentials against the UNCHANGED SI gate.

The August loop (docs/notes/sinker-cutter-loop-ledger.md) spent every physics term the
sinker's OWN ridge could carry: base P(gain>0)=0.290, best 0.520 with movement geometry.
Two levers were never measured, both excluded by construct on 2026-08-17 ("a sinker IS the
fastball"). A literature pass on 2026-09-10 found every public model (Sarris/Bay, Driveline,
PitchingBot, BP StuffPro) pulls both, and Jack lifted the exclusion for the POOLED form only:

  pooled     train ONE ridge on four-seams AND sinkers with a sinker indicator, so the
             sinker's coefficients are shrunk toward the four-seam's instead of estimated
             from ~1/7 of the data. Interaction terms let the slopes a sinker is expected
             to invert (release height, ride, run) differ where the data supports it.
             Only the SINKER rows are graded and gated here; the four-seam keeps its own
             shipped model regardless of what this shows.
  secdiff    "versus own primary fastball" differentials for the sinker that is a
             SECONDARY fastball (anchor_type == "_FF": the pitcher also throws a four-seam,
             which add_fastball_diffs anchors on). For the sinker-primary pitcher the
             differentials are forced to exactly zero and is_secondary_si = 0, so "slower
             than your fastball" is never a term for a pitcher whose fastball it is.

Same statistic, panel, criterion, bootstrap and seed as coach_incremental_gate.py and
coach_si_feature_gate.py: 50/50 z-blend of grade and prior results, P(blend gain > 0) over
200 cluster-bootstrap refits resampling train pitchers and criterion pitchers separately,
10%-share / 15-pitch pool, bar 0.95. "si_own_base" replicates the official SI row (0.290)
as the harness check; "si_own_movgeo_angsq" replicates the loop's best (0.520).

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs(). The
score frame is loaded from STUFFPLUS_DATA and the criterion frame from STUFFPLUS_DATA_CRIT
when set (the two builds come from different extracts), else from STUFFPLUS_DATA.
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

GRP = "SI"
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95
SEED = 20260817

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()
DATA_CRIT = os.environ.get("STUFFPLUS_DATA_CRIT", DATA)

# Slopes a sinker is expected to carry differently from a four-seam (fair_criterion
# FEATS_BY_PITCH note: release height inverts; ride and run are the pitch's identity).
INTERACT_WITH_SI = ["RelHeight", "InducedVertBreak", "HorzBreak_arm", "RelSpeed"]
MOVGEO = ["mov_angle", "mov_mag", "mov_angle_sq"]
SECDIFF = ["is_secondary_si", "vertbreakdiff_sec", "horzbreakdiff_sec",
           "velocity_differential_sec"]


def add_derived(df):
    """Arm-side frame, deviation terms, movement geometry, indicator/interaction and
    secondary-sinker differential columns. Physics only; no location, batter or usage."""
    df["RelSide_arm"] = df["RelSide"] * (1 - 2 * df["is_lhp"])
    df["HorzBreak_arm"] = df["HorzBreak"] * (1 - 2 * df["is_lhp"])
    for o, s_ in fc.DEV_SRC.items():
        df[o] = (df[s_] - df["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
    df["mov_angle"] = np.degrees(np.arctan2(df["HorzBreak_arm"], df["InducedVertBreak"]))
    df["mov_mag"] = np.hypot(df["HorzBreak_arm"], df["InducedVertBreak"])
    df["mov_angle_sq"] = df["mov_angle"] ** 2
    df["is_si"] = fc.pitch_mask(df, GRP).astype(float)
    for f in INTERACT_WITH_SI:
        df[f"si_x_{f}"] = df["is_si"] * df[f]
    # Secondary-sinker differentials: live only when the anchor is a real four-seam group.
    sec = (df["is_si"] == 1) & (df["anchor_type"] == "_FF")
    df["is_secondary_si"] = sec.astype(float)
    for c in fc.DIFF_FEATS:
        df[f"{c}_sec"] = np.where(sec, df[c], 0.0)
    return df


BASE = list(fc.BASE_FEATS)
SI_X = [f"si_x_{f}" for f in INTERACT_WITH_SI]

# name -> (train_groups, feature list). Train rows come from the listed PITCH_GROUPS keys;
# graded rows are always the sinker rows.
CANDIDATES = {
    "si_own_base":                 (["SI"],       BASE),
    "si_own_movgeo_angsq":         (["SI"],       BASE + MOVGEO),
    "si_own_secdiff":              (["SI"],       BASE + SECDIFF),
    "si_own_movgeo_secdiff":       (["SI"],       BASE + MOVGEO + SECDIFF),
    "pooled_ind":                  (["FF", "SI"], BASE + ["is_si"]),
    "pooled_ind_interact":         (["FF", "SI"], BASE + ["is_si"] + SI_X),
    "pooled_ind_interact_movgeo":  (["FF", "SI"], BASE + ["is_si"] + SI_X + MOVGEO),
    "pooled_ind_interact_secdiff": (["FF", "SI"], BASE + ["is_si"] + SI_X + SECDIFF),
    "pooled_all":                  (["FF", "SI"], BASE + ["is_si"] + SI_X + MOVGEO + SECDIFF),
}


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j):
    rs = float(fc.R(j["grade"], j["crit"]))
    rp = float(fc.R(j["prior"], j["crit"]))
    rb = float(fc.R(_z(j["grade"]) + _z(j["prior"]), j["crit"]))
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j["crit"]))
    return rs, rp, rb, rsp


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    fam = score[fc.pitch_mask(score, "FF") | fc.pitch_mask(score, "SI")].copy()
    fam = add_derived(fam)
    si_mask = fam["is_si"] == 1
    print("  train-year rows: FF %d  SI %d;  SI pitches with a four-seam anchor %.1f%%"
          % (((fam["year"] == 2024) & ~si_mask).sum(), ((fam["year"] == 2024) & si_mask).sum(),
             100 * fam.loc[si_mask, "is_secondary_si"].mean()), flush=True)

    c = crit[fc.pitch_mask(crit, GRP) & (crit["year"] == 2025)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

    out = {"share": SHARE, "n_boot": N_BOOT, "pass_bar": PASS_BAR, "seed": SEED,
           "interact_with_si": INTERACT_WITH_SI, "by_candidate": {}}
    for name, (groups, feats) in CANDIDATES.items():
        gmask = np.zeros(len(fam), dtype=bool)
        for g in groups:
            gmask |= fc.pitch_mask(fam, g).values
        b2 = fam[gmask].dropna(subset=feats + ["Target"])
        tr = b2[b2["year"] == 2024]
        ev = b2[(b2["year"] == 2025) & (b2["is_si"] == 1)]

        def build(train_df, eval_df, w=None):
            kw = {"ridge__sample_weight": w} if w is not None else {}
            m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
            m.fit(train_df[feats].values, train_df["Target"].values, **kw)
            p = pd.Series(m.predict(eval_df[feats].values), index=eval_df["PitcherId"].values)
            s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
            g = eval_df.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                 prior=("adjT", "mean")).join(tot)
            g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
            return s.join(g, how="inner").join(k, how="inner").dropna(
                subset=["grade", "prior", "crit"])

        j = build(tr, ev)
        print("")
        print("=== %s  train=%s rows=%d  n=%d pitchers ===" % (name, "+".join(groups),
                                                               len(tr), len(j)), flush=True)
        if len(j) < MIN_PITCHERS:
            out["by_candidate"][name] = {"n": int(len(j)), "skipped": "pool too small"}
            continue
        rs, rp, rb, rsp = _stats(j)
        print("    stuff r        %+.4f" % rs)
        print("    prior-results  %+.4f" % rp)
        print("    blend 50/50    %+.4f   gain over results %+.4f" % (rb, rb - rp))
        print("    semipartial    %+.4f" % rsp, flush=True)

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(SEED)
        gains, sps = [], []
        for _ in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            keep = set(rng.choice(ev_ids, len(ev_ids)))
            jb = build(tr, ev[ev["PitcherId"].isin(keep)], w=cnt[tr_codes])
            if len(jb) < MIN_PITCHERS:
                continue
            s2 = _stats(jb)
            gains.append(s2[2] - s2[1])
            sps.append(s2[3])
        gains, sps = np.array(gains), np.array(sps)
        p_gain = float((gains > 0).mean())
        lo, hi = np.percentile(gains, [2.5, 97.5])
        verdict = "PASS" if p_gain >= PASS_BAR else "no"
        print("    blend gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  "
              "P(gain>0)=%.3f  -> %s" % (len(gains), gains.mean(), lo, hi, p_gain, verdict),
              flush=True)
        out["by_candidate"][name] = {
            "n": int(len(j)), "train_groups": groups, "train_rows": int(len(tr)),
            "feats": feats, "stuff_r": round(rs, 4), "results_r": round(rp, 4),
            "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4),
            "semipartial": round(rsp, 4), "n_boot_used": int(len(gains)),
            "gain_mean": round(float(gains.mean()), 4),
            "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_gain_positive": p_gain,
            "p_semipartial_positive": float((sps > 0).mean()),
            "verdict": verdict}

    dest = os.path.join(SCORE_WORKDIR, "coach_si_pooled_gate.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
