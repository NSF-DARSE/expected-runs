"""How many pitches does each measure need? Validity against next season vs sample size.

THE QUESTION, which is not the one the other scripts answer: not "which measure is best
given a full season", but "how fast does each measure reach its own ceiling". A physical
grade is a smooth function of pitch characteristics, so its pitcher-level mean stabilises
after very few pitches. Realized run value is dominated by rare high-value events, so its
mean needs a great many. If the grade reaches its ceiling at a few dozen pitches while
results need a full season, the grade is usable in April on a freshman, a transfer, or a
guy who just changed a grip -- which is a capability claim, not a "we beat results" claim.

DESIGN, and the trap it avoids: the pitcher POOL IS HELD FIXED across every sample size.
Pitchers are required to have N_MAX+ graded four-seams up front, then n pitches are
SUBSAMPLED from each. Filtering by n at each step instead would change the population as n
grew (only high-volume arms survive a large n), which measures pitcher type and calls it a
sample-size effect. Cost of doing it correctly: the pool is a high-volume subpopulation,
mostly starters, so the SHAPE of these curves generalises but the absolute heights need
not match the full-pool numbers elsewhere in this suite. Stated, not hidden.

ESTIMAND: the expected correlation of ONE n-pitch sample. Averaging is over CORRELATIONS,
one computed inside each shuffle -- never over the per-pitcher values themselves.
Averaging the values first was the original bug here: pooling 40 independent 10-pitch
draws approximates a whole season, which erased the small-sample penalty the curve exists
to show and made every line flat. The tell was results off 10 pitches scoring +0.101
against results off 392 pitches at +0.102. If a future edit makes these curves flat again,
check this first.

UNCERTAINTY: each bootstrap draw takes a fresh shuffle AND a pitcher resample, so the band
carries both sources. The paired difference against the full-season reference is formed
inside each draw, which is what licenses any claim that a grade matches a full season.

EFFICIENCY: each rep shuffles a pitcher's pitches ONCE and reads running means off the
cumulative sum at every grid point, so one pass yields the whole curve.

CRITERION: next season's mean raw `Target`. Ground truth, deliberately -- adjT was tested
as a criterion (coach_measure_validity.py) and bought no predictive power, so using it here
would be criterion-shopping.

SIGN CONVENTION: ridge_pred, the location value, and Target are ALL expected runs from the
pitcher's perspective, LOWER = BETTER, and all three are used UNNEGATED as predictors.
A better measure therefore has a LARGER POSITIVE correlation with next season's Target.
Nothing is negated, so nothing can be double-negated.

Data rules: reads workdir caches; writes one JSON to the workdir. No pitcher names.
Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import fair_criterion as fc

N_MAX = 250          # pitchers must have at least this many graded four-seams
CRIT_FLOOR = 100     # and this many in the following season
GRID = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250]
REPS = 60            # shuffles for the point estimate; r is averaged, never the values
N_BOOT = 300         # draws for the band: each is a fresh shuffle + a pitcher resample
MEASURES = ["stuff", "loc", "blend", "results"]
LABELS = {"stuff": "Stuff+", "loc": "Location+", "blend": "Stuff+ and Location+",
          "results": "Actual results"}


def load_pool():
    """Per-pitcher arrays of the three per-pitch quantities, plus the next-season
    criterion, for pitchers clearing both floors."""
    s = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    s = s[(s["year"] == 2025) & s["ridge_pred"].notna() & s["loc"].notna()
          & s["Target"].notna()]

    c = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= CRIT_FLOOR]

    counts = s.groupby("PitcherId").size()
    ids = counts[counts >= N_MAX].index.intersection(k.index)
    s = s[s["PitcherId"].isin(ids)]
    arrays = {pid: (g["ridge_pred"].to_numpy(), g["loc"].to_numpy(),
                    g["Target"].to_numpy())
              for pid, g in s.groupby("PitcherId")}
    crit = k.loc[list(arrays), "crit"].to_numpy() * 100.0
    print(f"  pool: {len(arrays)} pitchers with {N_MAX}+ graded four-seams and "
          f"{CRIT_FLOOR}+ the next season")
    # full-season reference: each pitcher's realized results over ALL graded four-seams
    full = np.array([a[2].mean() for a in arrays.values()]) * 100.0
    return arrays, crit, full


def rep_means(arrays, rng):
    """One shuffle per pitcher; running means read off at every grid point.
    Returns array [pitcher, measure_raw(3), grid]."""
    out = np.empty((len(arrays), 3, len(GRID)))
    idx = np.array(GRID) - 1
    for i, (ridge, loc, tgt) in enumerate(arrays.values()):
        perm = rng.permutation(len(ridge))[:N_MAX]
        for j, arr in enumerate((ridge[perm], loc[perm], tgt[perm])):
            out[i, j] = np.cumsum(arr)[idx] / np.array(GRID)
    return out


def main() -> int:
    arrays, crit, full = load_pool()
    if len(arrays) < 60:
        return print(f"pool of {len(arrays)} is too small for this curve; raise "
                     f"CRIT_FLOOR or lower N_MAX rather than reporting it") or 1

    def z(v):
        sd = v.std()
        return (v - v.mean()) / sd if sd > 0 else v * 0.0

    def measures_at(vals, gi, idx):
        """The four measures for one grid point, on one (possibly resampled) pitcher set.
        All stay in run-value orientation (lower = better), matching the criterion."""
        st, lo, re = vals[idx, 0, gi], vals[idx, 1, gi], vals[idx, 2, gi]
        return dict(stuff=st, loc=lo, blend=(z(st) + z(lo)) / 2, results=re)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    n_p = len(arrays)
    all_idx = np.arange(n_p)
    n_full = float(np.mean([len(a[2]) for a in arrays.values()]))

    # ---- point estimates: average the CORRELATION over independent shuffles
    rng = np.random.default_rng(11)
    acc = {m: np.zeros(len(GRID)) for m in MEASURES}
    for _ in range(REPS):
        vals = rep_means(arrays, rng)
        for gi in range(len(GRID)):
            mv = measures_at(vals, gi, all_idx)
            for m in MEASURES:
                acc[m][gi] += corr(mv[m], crit)
    curves = {m: (acc[m] / REPS) for m in MEASURES}
    ref_r = corr(full, crit)

    # ---- band + paired difference vs the full-season reference, both sources of noise
    rb = np.random.default_rng(77)
    bse = {m: np.zeros((N_BOOT, len(GRID))) for m in MEASURES}
    bdiff = {m: np.zeros((N_BOOT, len(GRID))) for m in MEASURES}
    for b in range(N_BOOT):
        vals = rep_means(arrays, rb)
        idx = rb.integers(0, n_p, n_p)
        c = crit[idx]
        r_ref_b = corr(full[idx], c)
        for gi in range(len(GRID)):
            mv = measures_at(vals, gi, idx)
            for m in MEASURES:
                r = corr(mv[m], c)
                bse[m][b, gi] = r
                bdiff[m][b, gi] = r - r_ref_b

    print("\n  validity vs next season's raw Target (r, positive = sorts correctly)")
    print(f"  {'pitches':>8}" + "".join(f"{LABELS[m]:>22}" for m in MEASURES))
    for gi, n in enumerate(GRID):
        line = f"  {n:>8}"
        for m in MEASURES:
            line += f"{curves[m][gi]:>+16.3f}+/-{bse[m][:, gi].std():.3f}"
        print(line)

    print(f"\n  reference: full graded season of actual results "
          f"(mean {n_full:.0f} four-seams): r={ref_r:+.3f}")

    print("\n  first sample size that MATCHES the full-season reference "
          "(paired, P(grade >= results) >= 0.05, i.e. not shown to be worse):")
    crossing = {}
    for m in MEASURES:
        hit = None
        for gi, n in enumerate(GRID):
            if (bdiff[m][:, gi] < 0).mean() <= 0.95:   # not significantly below
                hit = n
                break
        crossing[m] = hit
        d = bdiff[m][:, GRID.index(hit)] if hit else None
        extra = (f"   (diff {d.mean():+.3f} +/-{d.std():.3f})" if hit is not None else "")
        print(f"    {LABELS[m]:<24}{hit if hit else 'not within ' + str(N_MAX)}{extra}")

    out = dict(pool=len(arrays), n_max=N_MAX, crit_floor=CRIT_FLOOR, grid=GRID,
               reps=REPS, n_boot=N_BOOT, mean_graded_pitches=round(n_full, 1),
               curves={m: dict(
                   label=LABELS[m],
                   r=[round(float(v), 4) for v in curves[m]],
                   se=[round(float(bse[m][:, gi].std()), 4) for gi in range(len(GRID))],
                   diff_vs_reference=[round(float(bdiff[m][:, gi].mean()), 4)
                                      for gi in range(len(GRID))],
                   diff_se=[round(float(bdiff[m][:, gi].std()), 4)
                            for gi in range(len(GRID))],
                   p_below=[round(float((bdiff[m][:, gi] < 0).mean()), 4)
                            for gi in range(len(GRID))])
                   for m in MEASURES},
               reference=dict(label="Full season of actual results", r=round(ref_r, 4),
                              pitches=round(n_full, 1)),
               crossing=crossing)
    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_sample_curve.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
