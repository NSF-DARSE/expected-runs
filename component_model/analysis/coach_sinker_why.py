"""Why does Stuff+ predict sinkers worse than last season's results does?

The shape of the failure rules out the boring explanation. If the sinker criterion were simply
noisy, NOTHING would predict it, results included. Results predicts it at +0.24 while our
physics predicts it at +0.08, so sinker outcomes carry persistent skill our features cannot
see. Four candidate explanations, cheapest first.

  (1) RELIABILITY CEILING. Year-over-year correlation of the grade and of the criterion, per
      pitch type. Validity cannot exceed what the criterion's own reliability allows, so this
      sets the ceiling the other theories are measured against.
  (2) CONTACT vs WHIFF. Share of each pitch type's value that arrives through a batted ball.
      Our target values a ball in play through a POOLED EV/LA map, which by construction
      averages away everything distinguishing one ground ball from another at the same exit
      speed and angle: double plays, infield positioning, where it was hit. A pitch whose job
      is weak contact therefore routes much of its value through the part of the target we
      deliberately smoothed. If validity tracks inversely with in-play share ACROSS pitch
      types, that is the mechanism, and it is the same failure recorded for v1, which
      undervalued soft-contact and ground-ball pitchers.
  (3) RANGE RESTRICTION. Feature spread among sinker throwers vs four-seam throwers. A
      selected population with compressed inputs gives a compressed correlation for reasons
      unrelated to the model being wrong.
  (4) TAG HETEROGENEITY. Sinker and TwoSeamFastBall apart, and sinker-primary vs
      sinker-secondary pitchers apart. Pooling two pitches, or two roles for one pitch, would
      blunt any model.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

import fair_criterion as fc

ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
SHARE = 0.10
ABS_MIN = 15


DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def _pool(score, crit, mask_s, mask_c, feats, tot, ctot):
    ff, _ = fc.stuff_ridge(score, return_model=True, pitch_mask=mask_s, feats=feats)
    ev = ff[ff["year"] == 2025]
    s = ev.groupby("PitcherId").agg(sn=("adjT", "size"), grade=("ridge_pred", "mean"),
                                    prior=("adjT", "mean")).join(tot)
    c = crit[mask_c & (crit["year"] == 2025)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    j = s.join(k, how="inner").dropna(subset=["grade", "prior", "crit"])
    return j


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")
    out = {"share": SHARE, "by_pitch": {}}

    print("")
    print("=== (1) reliability ceiling and (2) how value arrives ===")
    print("    %-6s%11s%10s%9s%8s%9s" % ("", "grade YoY", "crit YoY", "inplay%",
                                         "whiff%", "GB-ish%"))
    for grp in ORDER:
        m = fc.pitch_mask(score, grp)
        if int((m & (score["year"] == 2024)).sum()) < 2000:
            continue
        ff, _ = fc.stuff_ridge(score, return_model=True, pitch_mask=m,
                               feats=fc.feats_for(grp))
        per = ff.groupby(["PitcherId", "year"]).agg(
            n=("adjT", "size"), g=("ridge_pred", "mean"), a=("adjT", "mean"))
        per = per[per["n"] >= 25].reset_index()
        w = per.pivot(index="PitcherId", columns="year", values=["g", "a"]).dropna()
        gy = float(fc.R(w[("g", 2024)], w[("g", 2025)])) if len(w) > 20 else float("nan")
        ay = float(fc.R(w[("a", 2024)], w[("a", 2025)])) if len(w) > 20 else float("nan")
        sub = score[m]
        inplay = float(sub["is_inplay"].mean())
        whiff = float((sub["PitchCall"] == "StrikeSwinging").mean())
        gb = float(sub["TaggedHitType"].isin(["GroundBall"]).sum()
                   / max(sub["is_inplay"].sum(), 1))
        print("    %-6s%+11.3f%+10.3f%7.1f%%%6.1f%%%7.1f%%"
              % (grp, gy, ay, inplay * 100, whiff * 100, gb * 100))
        out["by_pitch"][grp] = {"grade_yoy": round(gy, 4), "crit_yoy": round(ay, 4),
                                "inplay_pct": round(inplay * 100, 2),
                                "whiff_pct": round(whiff * 100, 2),
                                "gb_share_of_inplay": round(gb * 100, 2),
                                "n_yoy": int(len(w))}

    print("")
    print("=== (3) range restriction: feature SD, sinker vs four-seam throwers ===")
    si = score[fc.pitch_mask(score, "SI")]
    fff = score[fc.pitch_mask(score, "FF")]
    print("    %-20s%9s%9s%8s" % ("feature", "SI sd", "FF sd", "ratio"))
    rr = {}
    for f in ["RelSpeed", "InducedVertBreak", "HorzBreak", "SpinRate", "Extension",
              "RelHeight", "RelSide"]:
        a, b = float(si[f].std()), float(fff[f].std())
        rr[f] = round(a / b, 3) if b else None
        print("    %-20s%9.2f%9.2f%8.2f" % (f, a, b, a / b))
    out["range_ratio_si_over_ff"] = rr

    print("")
    print("=== (4) tag heterogeneity: is the sinker one pitch and one role? ===")
    for label, tags in [("Sinker only", {"Sinker"}),
                        ("TwoSeam only", {"TwoSeamFastBall"}),
                        ("pooled (shipped)", {"Sinker", "TwoSeamFastBall"})]:
        ms = score["TaggedPitchType"].isin(tags)
        if int((ms & (score["year"] == 2024)).sum()) < 2000:
            print("    %-20s skipped, thin train year" % label)
            continue
        j = _pool(score, crit, ms, crit["TaggedPitchType"].isin(tags),
                  fc.feats_for("SI"), tot, ctot)
        j = j[(j["sn"] / j["tot"] >= SHARE) & (j["cn"] / j["ctot"] >= SHARE)
              & (j["sn"] >= ABS_MIN) & (j["cn"] >= ABS_MIN)]
        if len(j) < 30:
            print("    %-20s n=%d too few" % (label, len(j)))
            continue
        print("    %-20s n=%-5d stuff r=%+.4f   prior-results r=%+.4f"
              % (label, len(j), fc.R(j["grade"], j["crit"]), fc.R(j["prior"], j["crit"])))

    j = _pool(score, crit, fc.pitch_mask(score, "SI"), fc.pitch_mask(crit, "SI"),
              fc.feats_for("SI"), tot, ctot)
    j["shr"] = j["sn"] / j["tot"]
    print("    by sinker usage share (his fastball, or a change of pace?):")
    for lo, hi in [(0.10, 0.25), (0.25, 0.45), (0.45, 1.01)]:
        b = j[(j["shr"] >= lo) & (j["shr"] < hi) & (j["cn"] >= ABS_MIN)]
        if len(b) < 30:
            print("      %.0f%%-%.0f%%: n=%d too few" % (lo * 100, hi * 100, len(b)))
            continue
        print("      %.0f%%-%.0f%%: n=%-5d stuff r=%+.4f   prior-results r=%+.4f"
              % (lo * 100, hi * 100, len(b), fc.R(b["grade"], b["crit"]),
                 fc.R(b["prior"], b["crit"])))

    dest = os.path.join(SCORE_WORKDIR, "coach_sinker_why.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
