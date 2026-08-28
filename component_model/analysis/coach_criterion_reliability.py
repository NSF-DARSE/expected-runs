"""Does each pitch type's run value persist season to season? One pool, both year pairs.

WHY THIS RUNS BEFORE ANYTHING ELSE. coach_sinker_why.py measured sinker criterion
reliability at +0.008 while coach_model_floors.py had prior results predicting the sinker
criterion at +0.241. Those cannot both describe the same world: observed validity between two
measures is bounded above by sqrt(rel_x * rel_y), so a criterion with ~0 reliability cannot be
predicted at 0.24 by ANYTHING, our model or last year's results. One of those two numbers is
an artifact of how its pool was built, and every sinker conclusion from tonight inherits from
whichever survives.

The two differed in pool AND year pair, which is the bug in the comparison rather than a
finding. Here both are held fixed:
  pool       the usage-share rule (>= SHARE of the pitcher's season in BOTH years of the pair,
             plus ABS_MIN pitches), applied identically to every pitch type and both pairs.
  year pairs score frame = real 2024 -> 2025; criterion frame = real 2025 -> 2026. Two
             independent pairs, so a low figure in one is year-specific and a low figure in
             both is structural.
  criteria   adjT AND Target. This distinction is the one that could pay off: adjT replaces
             batted balls with a POOLED exit-speed/angle map value, so if a pitch type's real
             skill lives in contact quality beyond EV/LA -- ground balls, double plays,
             where the ball was hit -- then luck-stripping REMOVES that skill and adjT will
             look less reliable than Target. Sinkers are the obvious candidate. If sinker
             reliability is low on adjT and decent on Target, the problem is in the criterion,
             not the features, and the fix is a different target for contact pitches rather
             than more physics.

Reliability of a season mean also rises with pitch count by construction, so the share rule is
doing real work here: comparing a 70-pitch curveball to a 400-pitch fastball at a flat floor
would confound persistence with sample size.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs(). Committed
deliberately: this is the ruler every later refinement is measured with, and two sessions
measuring with different rulers is the failure this prevents.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import pandas as pd

import fair_criterion as fc

ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 30


DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def yoy(df, grp, col):
    """Correlation of a pitcher's mean `col` for one pitch type across the frame's two years.

    Returns (r, n). The frame's years are always labelled 2024/2025 (train/eval role), whatever
    the real seasons are -- see fair_criterion.load_pitches.
    """
    tot = df.groupby(["PitcherId", "year"]).size().rename("tot")
    sub = df[fc.pitch_mask(df, grp)]
    per = sub.groupby(["PitcherId", "year"]).agg(n=(col, "size"), v=(col, "mean"))
    per = per.join(tot)
    per = per[(per["n"] >= ABS_MIN) & (per["n"] / per["tot"] >= SHARE)].reset_index()
    w = per.pivot(index="PitcherId", columns="year", values="v").dropna()
    if len(w) < MIN_PITCHERS or 2024 not in w.columns or 2025 not in w.columns:
        return float("nan"), len(w)
    return float(fc.R(w[2024], w[2025])), len(w)


def main() -> int:
    t0 = time.time()
    frames = {"2024->2025": fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025"),
              "2025->2026": fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")}
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)

    out = {"share": SHARE, "abs_min": ABS_MIN, "by_pitch": {}}
    print("")
    print("=== year-over-year reliability of a pitcher's pitch-type run value ===")
    print("    pool: pitch is >=%.0f%% of his season in both years, >=%d pitches"
          % (SHARE * 100, ABS_MIN))
    print("")
    print("    %-5s%22s%22s" % ("", "adjT (luck-stripped)", "Target (unadjusted)"))
    print("    %-5s%11s%11s%11s%11s" % ("", "24->25", "25->26", "24->25", "25->26"))
    for grp in ORDER:
        cells, rec = [], {}
        for col in ("adjT", "Target"):
            for pair, df in frames.items():
                r, n = yoy(df, grp, col)
                rec["%s_%s" % (col, pair)] = (None if math.isnan(r) else round(r, 4))
                rec["n_%s_%s" % (col, pair)] = int(n)
                cells.append("%11s" % ("n=%d" % n if math.isnan(r) else "%+.3f" % r))
        print("    %-5s%s" % (grp, "".join(cells)))
        out["by_pitch"][grp] = rec

    print("")
    print("=== what those ceilings allow ===")
    print("    A validity of r between grade and criterion is bounded by")
    print("    sqrt(rel_grade * rel_crit). Grade reliability measured ~0.77-0.89")
    print("    (coach_sinker_why.py), so with grade rel = 0.80:")
    print("")
    print("    %-5s%14s%16s" % ("", "adjT 25->26", "implied ceiling"))
    for grp in ORDER:
        r = out["by_pitch"][grp].get("adjT_2025->2026")
        if r is None:
            print("    %-5s%14s%16s" % (grp, "--", "--"))
            continue
        ceil = math.sqrt(0.80 * r) if r > 0 else float("nan")
        print("    %-5s%+14.3f%16s"
              % (grp, r, "--" if r <= 0 else "%.3f" % ceil))

    dest = os.path.join(SCORE_WORKDIR, "coach_criterion_reliability.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
