"""Where should the pitch-count floor sit, per pitch type? Two floors, not one.

coach_model_by_type.py used ONE floor for both sides of the comparison, which conflates two
different jobs and made its floor-to-floor flips partly an artifact:

  SCORE floor    how many pitches of that type we need to know a pitcher's GRADE. Stuff+ is a
                 mean of per-pitch physical predictions and physical features barely vary
                 within a pitcher, so this converges fast and should matter little.
  CRITERION floor  how many pitches we need for NEXT season's run value to be less noisy. Run
                 value per pitch is extremely noisy, so this is where a floor earns its keep.

The score floor also drives the precision of the "prior results" arm, because that arm is
itself an outcome measure on the graded season. Holding both at one number therefore degrades
the baseline and the model together and hides which one the floor is actually buying.

READ THE WHOLE SURFACE, DO NOT PICK THE MAXIMUM. Scanning floors and keeping the best is
exactly the p-hacking Jack ruled out in the 2026-08-17 meeting. The output is here to show
whether a conclusion is STABLE across floors, not to select one. A pitch type whose winner
depends on the floor has no established winner.

The USAGE-SHARE rule below was chosen for scale-freeness BEFORE this surface was read, on
the argument in the next paragraph, and is not the winning cell of a scan. That ordering is
the whole reason it can be used as a decision rule rather than a description.

Also reported: a USAGE-SHARE floor. 100 sliders and 100 four-seams are not comparable
commitments -- 100 four-seams is a couple of starts, 100 sliders is most of a season for a
reliever. A share rule ("this pitch is at least X% of his season") is scale-free across pitch
types and is the version a coach can state back to you.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs(). Committed
deliberately: this is the ruler every later refinement is measured with, and two sessions
measuring with different rulers is the failure this prevents.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

import fair_criterion as fc

SCORE_FLOORS = (10, 25, 50, 100, 150)
CRIT_FLOORS = (25, 50, 100, 200)
SHARE_FLOORS = (0.05, 0.10, 0.20)
ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
MIN_PITCHERS = 30


DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print(f"  frames loaded in {time.time() - t0:.0f}s", flush=True)
    # season totals for the usage-share rule
    score_tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    crit_tot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    out = {"score_floors": list(SCORE_FLOORS), "crit_floors": list(CRIT_FLOORS),
           "share_floors": list(SHARE_FLOORS), "min_pitchers": MIN_PITCHERS,
           "by_pitch": {}}
    for grp in ORDER:
        m = fc.pitch_mask(score, grp)
        if int((m & (score["year"] == 2024)).sum()) < 2000:
            continue
        ff, _ = fc.stuff_ridge(score, return_model=True, pitch_mask=m,
                               feats=fc.feats_for(grp))
        ev = ff[ff["year"] == 2025]
        s = ev.groupby("PitcherId").agg(sn=("adjT", "size"), grade=("ridge_pred", "mean"),
                                        prior=("adjT", "mean")).join(score_tot)
        c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")
                                       ).join(crit_tot)
        j0 = s.join(k, how="inner").dropna(subset=["grade", "prior", "crit"])

        print("")
        print(f"===== {grp}  tags {sorted(fc.PITCH_GROUPS[grp])} =====")
        print("  stuff r / prior-results r, blank where fewer than "
              f"{MIN_PITCHERS} pitchers survive")
        print(f"    {'score>=':<9}" + "".join(f"{'crit>=' + str(c):>20}"
                                              for c in CRIT_FLOORS))
        rec = {"grid": {}, "share": {}}
        for sf in SCORE_FLOORS:
            cells = []
            for cf in CRIT_FLOORS:
                j = j0[(j0["sn"] >= sf) & (j0["cn"] >= cf)]
                if len(j) < MIN_PITCHERS:
                    cells.append(f"{'n=' + str(len(j)):>20}")
                    continue
                rv = float(fc.R(j["grade"], j["crit"]))
                rr = float(fc.R(j["prior"], j["crit"]))
                cells.append(f"{rv:+.3f}/{rr:+.3f} n={len(j):<4}".rjust(20))
                rec["grid"][f"{sf}_{cf}"] = {"n": int(len(j)), "stuff_r": round(rv, 4),
                                             "results_r": round(rr, 4)}
            print(f"    {sf:<9}" + "".join(cells))
        # usage share, applied to BOTH seasons, with a small absolute floor for stability
        print("  usage-share rule (pitch is X% of his season, both years, plus 15 pitches):")
        for sh in SHARE_FLOORS:
            j = j0[(j0["sn"] / j0["tot"] >= sh) & (j0["cn"] / j0["ctot"] >= sh)
                   & (j0["sn"] >= 15) & (j0["cn"] >= 15)]
            if len(j) < MIN_PITCHERS:
                print(f"    >={sh:.0%}: n={len(j)} too few")
                continue
            rv = float(fc.R(j["grade"], j["crit"]))
            rr = float(fc.R(j["prior"], j["crit"]))
            print(f"    >={sh:.0%}: n={len(j):<5} stuff r={rv:+.4f}  "
                  f"prior-results r={rr:+.4f}  median pitches {int(j['sn'].median())}")
            rec["share"][f"{sh}"] = {"n": int(len(j)), "stuff_r": round(rv, 4),
                                     "results_r": round(rr, 4),
                                     "median_pitches": int(j["sn"].median())}
        out["by_pitch"][grp] = rec

    dest = os.path.join(SCORE_WORKDIR, "coach_model_floors.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print(f"  wrote {dest}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
