"""Does a pitch type's Stuff+ ADD to last season's results? The question that gates shipping.

WHY HEAD-TO-HEAD WAS THE WRONG CONTEST. coach_model_by_type.py scored every pitch type as
"Stuff+ vs prior results", which was the right bar for the four-seam page (does physics beat
the scoreboard) and is the wrong bar for a shipping decision, because nobody chooses between
them -- a coach page can show both. Two reasons the head-to-head misleads:

  1. The "prior results" column IS the criterion's own year-over-year reliability. prior =
     this season's mean run value, criterion = next season's, same pitch, same pitcher. So
     "beat results" literally means "predict the criterion better than the criterion predicts
     itself", which is a strange bar to make a product decision on.
  2. Stuff+ is available at 15 pitches; results needs hundreds. A grade that ties on a
     high-usage pool can still be the only usable signal on the pitches a coach actually asks
     about.

So the shipping question is INCREMENTAL: given last season's results, does the physical grade
add anything? Reported two ways, because they answer slightly different questions:
  semipartial   corr(stuff, criterion) after removing prior results from stuff. "What does the
                grade know that the scoreboard does not."
  blend gain    r(blend, criterion) - r(prior, criterion), where blend is an equal-weight
                average of the two z-scored predictors. Equal weights ON PURPOSE: fitted
                weights on the same pool that scores them is how a spurious gain gets
                manufactured, and the four-seam work already showed optimal weighting buys
                ~0.004 over 50/50.

The gate is a CLUSTER BOOTSTRAP matching coach_release_gate.py: resample TRAIN pitchers and
refit the ridge on them (frequency weights), resample CRITERION pitchers and re-score. A
pitch type passes when P(blend gain > 0) >= 0.95. Pre-registered before the run: that bar,
that statistic, on the usage-share pool, for every pitch type with enough sample -- not
chosen after seeing which types looked good.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fair_criterion as fc

ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95


DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j):
    """(stuff r, prior r, blend r, semipartial) on one joined pitcher table."""
    rs = float(fc.R(j["grade"], j["crit"]))
    rp = float(fc.R(j["prior"], j["crit"]))
    # blend: equal weight on z-scored predictors. Both are already oriented as expected run
    # value (lower = better), so no sign flip is needed before averaging.
    blend = _z(j["grade"]) + _z(j["prior"])
    rb = float(fc.R(blend, j["crit"]))
    # semipartial: stuff with prior projected out, still against the raw criterion
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j["crit"]))
    return rs, rp, rb, rsp


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    out = {"share": SHARE, "n_boot": N_BOOT, "pass_bar": PASS_BAR, "by_pitch": {}}
    for grp in ORDER:
        ms = fc.pitch_mask(score, grp)
        if int((ms & (score["year"] == 2024)).sum()) < 2000:
            print("")
            print("=== %s: skipped, no usable train year ===" % grp)
            continue
        feats = fc.feats_for(grp)
        base = score[ms].copy()
        base["RelSide_arm"] = base["RelSide"] * (1 - 2 * base["is_lhp"])
        base["HorzBreak_arm"] = base["HorzBreak"] * (1 - 2 * base["is_lhp"])
        for o, s_ in fc.DEV_SRC.items():
            base[o] = (base[s_] - base["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
        base = base.dropna(subset=feats + ["Target"])
        tr = base[(base["year"] == 2024) & base["Target"].notna()]
        ev = base[base["year"] == 2025]

        c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
        k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

        def build(train_df, eval_df, w=None):
            kw = {"ridge__sample_weight": w} if w is not None else {}
            m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
            m.fit(train_df[feats].values, train_df["Target"].values, **kw)
            p = pd.Series(m.predict(eval_df[feats].values),
                          index=eval_df["PitcherId"].values)
            s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
            g = eval_df.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                 prior=("adjT", "mean")).join(tot)
            g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
            return s.join(g, how="inner").join(k, how="inner").dropna(
                subset=["grade", "prior", "crit"])

        j = build(tr, ev)
        print("")
        print("=== %s  tags %s  n=%d pitchers ===" % (grp, sorted(fc.PITCH_GROUPS[grp]),
                                                      len(j)))
        if len(j) < MIN_PITCHERS:
            print("    too few pitchers on the share pool to gate")
            out["by_pitch"][grp] = {"n": int(len(j)), "skipped": "pool too small"}
            continue
        rs, rp, rb, rsp = _stats(j)
        print("    stuff r        %+.4f" % rs)
        print("    prior-results  %+.4f   (this IS the criterion's YoY reliability)" % rp)
        print("    blend 50/50    %+.4f   gain over results %+.4f" % (rb, rb - rp))
        print("    semipartial    %+.4f   (what the grade knows that results does not)" % rsp)

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(20260817)
        gains, sps = [], []
        for b in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            w = cnt[tr_codes]
            keep = set(rng.choice(ev_ids, len(ev_ids)))
            jb = build(tr, ev[ev["PitcherId"].isin(keep)], w=w)
            if len(jb) < MIN_PITCHERS:
                continue
            s2 = _stats(jb)
            gains.append(s2[2] - s2[1])
            sps.append(s2[3])
            if (b + 1) % 100 == 0:
                print("      refit boot %d/%d" % (b + 1, N_BOOT), flush=True)
        gains, sps = np.array(gains), np.array(sps)
        p_gain = float((gains > 0).mean())
        lo, hi = np.percentile(gains, [2.5, 97.5])
        verdict = "PASS" if p_gain >= PASS_BAR else "no"
        print("    blend gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  "
              "P(gain>0)=%.3f  -> %s" % (len(gains), gains.mean(), lo, hi, p_gain, verdict))
        print("    semipartial P(>0)=%.3f" % float((sps > 0).mean()))
        out["by_pitch"][grp] = {
            "n": int(len(j)), "stuff_r": round(rs, 4), "results_r": round(rp, 4),
            "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4),
            "semipartial": round(rsp, 4), "n_boot_used": int(len(gains)),
            "gain_mean": round(float(gains.mean()), 4),
            "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_gain_positive": p_gain,
            "p_semipartial_positive": float((sps > 0).mean()),
            "verdict": verdict}

    dest = os.path.join(SCORE_WORKDIR, "coach_incremental_gate.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
