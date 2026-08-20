"""Does Location+'s ceiling advantage over Stuff+ survive a criterion it does not share?

WHY THIS EXISTS. coach_pitching_plus_weights.py measures, on four-seams, an asymptotic
ceiling of 0.83 for Location+ against 0.50 for Stuff+, and that gap is the reason Location+
outweighs Stuff+ from roughly 100 pitches on. It would be the headline of any coach-facing
validation page, so it needs to be a fact about pitching and not an artefact of how the two
numbers are built.

THE CONCERN, stated precisely. The criterion is adjT, which is xT with league and a shrunk
batter effect removed, and xT is itself a batted-ball map applied to launch conditions.
Location+ is the pooled location map's expected value, ALSO fit on xT. So Location+ and the
criterion share an estimation stage that Stuff+ does not: Stuff+ is a ridge on physical
release and movement features. If a slice of that 0.83 is the two quantities agreeing because
they were built from the same map rather than because location predicts run prevention, the
gap is partly circular, and a page built on it would overstate location.

WHAT SEPARATES THE TWO EXPLANATIONS. Re-measure every ceiling against criteria that peel the
shared machinery back one stage at a time, holding the panel, the floor, the noise estimator
and the seed fixed so the criterion is the only thing that moves:

    adjT    xT minus league and batter effects -- the incumbent, most shared machinery
    xT      the batted-ball map alone, no opponent adjustment
    Target  realised run value. Shares NOTHING with either grade: no map, no model.

Target is the decisive column. It is far noisier, so every ceiling drops in absolute terms and
absolute levels across criteria are not comparable. The comparable quantity is the RATIO
Location+ : Stuff+ within each criterion. If that ratio holds up against Target, the advantage
is real and the page can lead with it. If it collapses toward 1 while surviving against xT and
adjT, the advantage is substantially shared-map agreement and the page must not lead with it.

WHAT THIS DOES NOT SETTLE. A surviving ratio does not make Location+ a purely physical skill,
and it says nothing about whether the four-seam map should be applied to other pitch types
(it should not -- see coach_pitching_plus_weights.py, limitation 1). It answers one question:
whether the measured gap is an artefact of a shared estimation stage.

SIGN CONVENTION: all four columns are expected run value from the pitcher's perspective,
LOWER = BETTER. A ceiling here is a correlation between two such columns, so a POSITIVE
ceiling means the grade tracks the criterion in the right direction. Do not negate.

Data rules: reads the pitch-cache parquets only; prints only. Four-seams only, no pitcher
names, no per-pitcher rows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coach_pitching_plus_weights as w
import fair_criterion as fc
import reliability_curves as rc

# Grades on the left, criteria on the right. adjT appears on both sides on purpose: as a grade
# it is the pitcher's own prior results, and its ceiling against a criterion is that
# criterion's own year-over-year reliability, which is the yardstick the other two are read
# against.
GRADES = [("ridge_pred", "Stuff+"), ("loc", "Location+"), ("adjT", "Prior results")]
CRITERIA = [("adjT", "adjT (incumbent)"), ("xT", "xT (map only)"),
            ("Target", "Target (realised)")]


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"))
    ap.add_argument("--min-n", type=int, default=40)
    ap.add_argument("--min-half", type=int, default=15)
    ap.add_argument("--splits", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()
    if not a.caches:
        sys.exit("Set --caches or STUFFPLUS_CACHES (both pitch-cache parquets).")
    a.caches = [p.strip() for p in a.caches.split(",") if p.strip()]
    return a


def main() -> int:
    t0 = time.time()
    a = cli()
    rng = np.random.default_rng(a.seed)
    df = w.load_seasons(a.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)

    ff = fc.stuff_ridge(df)
    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    lmap = fc.PooledLocationMap(ff[(ff["year"] == w.TRAIN_YEAR) & ff["xT"].notna()])
    ff["loc"] = lmap.apply(ff)
    ff = w.panel(ff, "FF", a.min_n)
    print("  frame ready in %.0fs" % (time.time() - t0))

    cols = sorted({c for c, _ in GRADES} | {c for c, _ in CRITERIA})
    ff = ff.dropna(subset=cols).copy()
    tab = w.season_table(ff, cols)
    print("  panel: %d pitchers, %d pitcher-seasons (%d+ four-seams, 2+ seasons)"
          % (tab["pitcher"].nunique(), len(tab), a.min_n))

    # One noise estimate per column, shared across every criterion, so a criterion swap cannot
    # move a number through a re-estimated noise term underneath it.
    noise = {}
    for c in cols:
        m, lo, hi, used = w.noise_scale(ff, c, a.splits, a.min_half, rng)
        noise[c] = m
        print("    noise/pitch  %-11s %.6f  [%.6f, %.6f]  on %d pitcher-seasons"
              % (c, m, lo, hi, used))

    print("")
    print("  %-18s%-15s%9s%9s%9s%9s" % ("criterion", "grade", "raw r", "ceiling",
                                        "rel x", "rel y"))
    ratios = {}
    for ycol, ylab in CRITERIA:
        ceils = {}
        for xcol, xlab in GRADES:
            r = rc.cross_metric_ceiling(tab, xcol, ycol, noise[xcol], noise[ycol])
            ceils[xlab] = r["ceiling"]
            print("  %-18s%-15s%9.3f%9.3f%9.3f%9.3f"
                  % (ylab, xlab, r["raw_r"], r["ceiling"], r["rel_x"], r["rel_y"]))
        st = ceils["Stuff+"]
        ratios[ylab] = ceils["Location+"] / st if st != 0 else float("nan")
        print("  %-18s%-15s%9s%9.2fx" % ("", "Location+ : Stuff+", "", ratios[ylab]))
        print("")

    print("  ratio by criterion:  %s" % "   ".join(
        "%s=%.2fx" % (k.split(" ")[0], v) for k, v in ratios.items()))
    print("  READ THIS AS: the ratio against Target is the one that is not shared machinery.")
    print("  total %.0fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# RESULT, run 2026-08-20 on the three-season D1 panel (2598 pitchers, 5640 pitcher-seasons,
# 40-pitch floor, seed 20260820). The concern above is REFUTED:
#
#   criterion            Stuff+   Location+   ratio
#   adjT (incumbent)      0.496       0.829    1.67x
#   xT (map only)         0.440       0.808    1.84x
#   Target (realised)     0.416       0.744    1.79x
#
# The ratio does not collapse against Target; it comes out slightly HIGHER (1.79x) than
# against the incumbent adjT (1.67x). So the gap is not shared-map agreement, and adjT is if
# anything understating Location+ rather than flattering it. Every ceiling falls in absolute
# terms as the criterion gets noisier, exactly as expected, which is why the ratio and not the
# level is the comparable quantity.
#
# Consequence: a coach-facing validation page may lead with the Location+ ceiling. Re-run this
# if the location map's estimation stage changes (per-type maps, batter-hand mirroring), since
# both are changes to the machinery this check exists to test.
