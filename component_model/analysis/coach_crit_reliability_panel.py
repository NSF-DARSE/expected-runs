"""Which criterion-reliability number is real: 0.008 or 0.24? The 2x2 that decides task zero.

Two scripts report "the criterion's year-over-year reliability" for sinkers and disagree by
an order of magnitude. coach_sinker_why.py says 0.008; coach_incremental_gate.py's
prior-results column says 0.241. Both are corr(pitcher mean adjT year t, year t+1), but they
differ on TWO axes at once:

  year pair   sinker_why uses 2024 -> 2025 (within the score workdir); the gate uses
              2025 -> 2026 (score workdir eval year against crit workdir, whose "2025"
              label is the role-relabeled 2026 season).
  panel       sinker_why keeps pitcher-seasons with n >= 25 pitches of the type, no usage
              condition; the gate keeps n >= 15 AND >= 10% usage share, on both sides.

This script computes the full 2x2 per pitch type, with pitcher-cluster bootstrap CIs, so the
disagreement is attributed rather than argued about. It also traces reliability against the
minimum-pitch threshold (15/25/40/60/100) on both year pairs: if reliability climbs with the
floor, the criterion is sample-limited and a stricter panel has headroom; if it stays flat at
zero, there is no persistent skill for ANY model to find and the shipping question is closed.

The ceiling that matters: observed validity <= sqrt(rel_grade * rel_crit). Grade YoY is ~0.77
(SI) / ~0.84 (FC) from coach_sinker_why.json, so the criterion side decides everything.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import fair_criterion as fc

ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
SHARE = 0.10
ABS_MIN = 15
WHY_MIN = 25
N_BOOT = 2000
MIN_FLOORS = [15, 25, 40, 60, 100]

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def _boot_r(a, b, rng, n_boot=N_BOOT):
    """Point r and percentile CI, resampling pitchers (rows) with replacement."""
    r = float(fc.R(a, b))
    n = len(a)
    if n < 20:
        return r, float("nan"), float("nan")
    av, bv = a.values, b.values
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        x, y = av[idx], bv[idx]
        if x.std() == 0 or y.std() == 0:
            continue
        rs.append(np.corrcoef(x, y)[0, 1])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r, float(lo), float(hi)


def _pair_table(df, mask, y0, y1, min_n, share=None, tot0=None, tot1=None):
    """Pitcher table of mean adjT in year y0 vs y1 under a panel rule.

    share=None reproduces the sinker_why panel (absolute floor only); a float reproduces the
    gate panel (floor AND usage share of the pitcher's total pitches that year).
    """
    sub = df[mask]
    per = sub.groupby(["PitcherId", "year"]).agg(n=("adjT", "size"), a=("adjT", "mean"))
    per = per.reset_index()
    keep = per["n"] >= min_n
    if share is not None:
        tots = {y0: tot0, y1: tot1}
        t = per.apply(lambda r: tots[r["year"]].get(r["PitcherId"], np.nan), axis=1)
        keep &= (per["n"] / t) >= share
    per = per[keep]
    w = per.pivot(index="PitcherId", columns="year", values="a").dropna()
    if y0 not in w.columns or y1 not in w.columns:
        return None
    return w[[y0, y1]].rename(columns={y0: "prior", y1: "crit"})


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(20260820)
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)

    # totals per pitcher-year for the share condition, per frame
    tot_s = {y: score[score["year"] == y].groupby("PitcherId").size().to_dict()
             for y in (2024, 2025)}
    tot_c = {y: crit[crit["year"] == y].groupby("PitcherId").size().to_dict()
             for y in (2024, 2025)}

    # the cross-frame pair (real 2025 -> real 2026): score's eval year joined to crit's
    # eval year. Both labeled 2025 in their own frames.
    def cross_table(mask_s, mask_c, min_n, share=None):
        s = score[mask_s & (score["year"] == 2025)]
        c = crit[mask_c & (crit["year"] == 2025)]
        g = s.groupby("PitcherId").agg(sn=("adjT", "size"), prior=("adjT", "mean"))
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean"))
        g, k = g[g["sn"] >= min_n], k[k["cn"] >= min_n]
        if share is not None:
            g = g[g.index.map(lambda p: g.loc[p, "sn"] / tot_s[2025].get(p, np.inf))
                  >= share]
            k = k[k.index.map(lambda p: k.loc[p, "cn"] / tot_c[2025].get(p, np.inf))
                  >= share]
        return g.join(k, how="inner").dropna(subset=["prior", "crit"])

    out = {"n_boot": N_BOOT, "by_pitch": {}}
    for grp in ORDER:
        ms = fc.pitch_mask(score, grp)
        mc = fc.pitch_mask(crit, grp)
        cells = {}

        # 2x2: (year pair) x (panel rule)
        w = _pair_table(score, ms, 2024, 2025, WHY_MIN)
        if w is not None and len(w) >= 20:
            r, lo, hi = _boot_r(w["prior"], w["crit"], rng)
            cells["y2425_whypanel"] = {"n": len(w), "r": round(r, 4),
                                       "ci": [round(lo, 4), round(hi, 4)]}
        w = _pair_table(score, ms, 2024, 2025, ABS_MIN, SHARE,
                        tot_s[2024], tot_s[2025])
        if w is not None and len(w) >= 20:
            r, lo, hi = _boot_r(w["prior"], w["crit"], rng)
            cells["y2425_gatepanel"] = {"n": len(w), "r": round(r, 4),
                                        "ci": [round(lo, 4), round(hi, 4)]}
        j = cross_table(ms, mc, WHY_MIN)
        if len(j) >= 20:
            r, lo, hi = _boot_r(j["prior"], j["crit"], rng)
            cells["y2526_whypanel"] = {"n": len(j), "r": round(r, 4),
                                       "ci": [round(lo, 4), round(hi, 4)]}
        j = cross_table(ms, mc, ABS_MIN, SHARE)
        if len(j) >= 20:
            r, lo, hi = _boot_r(j["prior"], j["crit"], rng)
            cells["y2526_gatepanel"] = {"n": len(j), "r": round(r, 4),
                                        "ci": [round(lo, 4), round(hi, 4)]}

        # reliability vs minimum-pitch floor, both year pairs, absolute floor only
        floors = {}
        for mn in MIN_FLOORS:
            row = {}
            w = _pair_table(score, ms, 2024, 2025, mn)
            if w is not None and len(w) >= 20:
                r, lo, hi = _boot_r(w["prior"], w["crit"], rng)
                row["y2425"] = {"n": len(w), "r": round(r, 4),
                                "ci": [round(lo, 4), round(hi, 4)]}
            j = cross_table(ms, mc, mn)
            if len(j) >= 20:
                r, lo, hi = _boot_r(j["prior"], j["crit"], rng)
                row["y2526"] = {"n": len(j), "r": round(r, 4),
                                "ci": [round(lo, 4), round(hi, 4)]}
            floors[mn] = row

        out["by_pitch"][grp] = {"cells": cells, "floors": floors}
        print("")
        print("=== %s ===" % grp)
        for k, v in cells.items():
            print("    %-18s n=%4d  r=%+.3f  CI [%+.3f, %+.3f]"
                  % (k, v["n"], v["r"], v["ci"][0], v["ci"][1]))
        for mn, row in floors.items():
            for yk, v in row.items():
                print("    floor>=%-4d %-6s n=%4d  r=%+.3f  CI [%+.3f, %+.3f]"
                      % (mn, yk, v["n"], v["r"], v["ci"][0], v["ci"][1]))

    dest = os.path.join(SCORE_WORKDIR, "coach_crit_reliability_panel.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
