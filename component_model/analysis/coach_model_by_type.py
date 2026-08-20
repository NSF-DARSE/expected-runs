"""Per-pitch-type Stuff+ models: are they viable, and what do they lean on?

The four-seam model was validated against next season's four-seam run value. This does the
same thing for every other pitch type, one model per type on its own feature set
(fc.FEATS_BY_PITCH), so a refinement pass has a number to move instead of an opinion.

WHAT THIS IS AND IS NOT. Measurement, not a gate. No pre-registered decision rule is
attached, because the question here is "does this pitch type support a model at all", which
has to be answered before any change to it can be gated. Treat a validity figure on 30
pitchers as a reason to collect more, never as evidence a feature works.

Reported per pitch type:
  n_train / n_score / n_crit   sample at each stage, so a thin type is visible immediately
  validity r                   pitcher-level corr(grade, next-season run value of that type),
                               positive = the grade orders pitchers correctly
  results r                    the same criterion predicted by that type's OWN prior-season
                               run value. This is the bar that matters: a Stuff+ model that
                               cannot beat "what happened last year" is not yet useful.
  coefficient signs            in HIGHER = BETTER FOR THE PITCHER orientation, to check
                               physical stories directly (does release height invert on a
                               sinker versus a four-seam, as Dan expects?)

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

import numpy as np
import pandas as pd

import fair_criterion as fc

FLOORS = (100, 50, 25)
ORDER = ["FF", "SI", "FC", "SL", "SW", "CB", "CH"]


DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print(f"  frames loaded in {time.time() - t0:.0f}s", flush=True)

    out = {"floors": list(FLOORS), "by_pitch": {}}
    MIN_TRAIN = 2000
    for grp in ORDER:
        feats = fc.feats_for(grp)
        # A tag can exist in the eval year and not the train year: "Sweeper" was barely used
        # in 2024 and StandardScaler raises on an empty fit rather than returning something
        # useless, which is the right behaviour. Skip loudly instead of pretending.
        m = fc.pitch_mask(score, grp)
        n_tr_raw = int((m & (score["year"] == 2024)).sum())
        if n_tr_raw < MIN_TRAIN:
            print("")
            print(f"===== {grp}  tags {sorted(fc.PITCH_GROUPS[grp])} =====")
            print(f"  SKIPPED: only {n_tr_raw:,} train-year pitches "
                  f"(need {MIN_TRAIN:,}); this tag has no usable 2024 history")
            out["by_pitch"][grp] = {"tags": sorted(fc.PITCH_GROUPS[grp]),
                                    "skipped": "insufficient train-year sample",
                                    "n_train_raw": n_tr_raw}
            continue
        ff, model = fc.stuff_ridge(score, return_model=True, pitch_mask=m, feats=feats)
        tr = ff[ff["year"] == 2024]
        ev = ff[ff["year"] == 2025]
        # criterion: the SAME pitch type's next-season luck-adjusted run value
        c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
        cg = c.groupby("PitcherId").agg(n=("adjT", "size"), crit=("adjT", "mean"))
        # prior-season results on the same pitch type, as the bar to beat
        pri = ev.groupby("PitcherId").agg(pn=("adjT", "size"), prior=("adjT", "mean"))
        grade = ev.groupby("PitcherId")["ridge_pred"].mean().rename("grade")

        print(f"\n===== {grp}  tags {sorted(fc.PITCH_GROUPS[grp])}  "
              f"{len(feats)} features =====")
        print(f"  train {len(tr):,}  score {len(ev):,}  criterion {len(c):,}")
        rec = {"tags": sorted(fc.PITCH_GROUPS[grp]), "n_feats": len(feats),
               "n_train": int(len(tr)), "n_score": int(len(ev)),
               "n_crit": int(len(c)), "by_floor": {}}
        for f in FLOORS:
            j = (pd.DataFrame(grade).join(pri).join(cg)
                 .query("n >= @f and pn >= @f").dropna())
            if len(j) < 10:
                print(f"    floor {f:>3}: {len(j)} pitchers -- too few to report")
                rec["by_floor"][f] = {"n": int(len(j))}
                continue
            rv = float(fc.R(j["grade"], j["crit"]))
            rr = float(fc.R(j["prior"], j["crit"]))
            print(f"    floor {f:>3}: n={len(j):>4}  stuff r={rv:+.4f}   "
                  f"prior-results r={rr:+.4f}   {'STUFF+ AHEAD' if rv > rr else 'results ahead'}")
            rec["by_floor"][f] = {"n": int(len(j)), "stuff_r": round(rv, 4),
                                  "results_r": round(rr, 4)}
        # coefficients, negated once into higher = better for the pitcher
        coefs = dict(zip(feats, model[-1].coef_))
        rec["better_when_more"] = {k: bool(-v > 0) for k, v in coefs.items()}
        rec["coefs_hi"] = {k: round(float(-v), 5) for k, v in coefs.items()}
        top = sorted(coefs, key=lambda k: -abs(coefs[k]))[:6]
        print("    strongest inputs (higher = better for pitcher):")
        for k in top:
            print(f"      {k:<22}{-coefs[k]:+.5f}  "
                  f"{'more is better' if -coefs[k] > 0 else 'less is better'}")
        out["by_pitch"][grp] = rec

    dest = os.path.join(SCORE_WORKDIR, "coach_model_by_type.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote {dest}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
