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

CRITERION: next season's mean `adjT`, the project's fair criterion (adjT_global since PR
#38: league-strength-aware, joint pitcher+batter fit vs a global mean). The "results"
measure is the same quantity built from the graded season's own pitches, so the chart
asks how fast each grade reaches the criterion's own year-over-year signal. The August
version used raw Target for both; STUFFPLUS_ADJ_METHOD=within_league reproduces the
pre-#38 adjT for a before/after.

CROSSOVERS: for each pair of measures, the pitch count at which the slower-starting
curve overtakes the other, interpolated on log(n). Reported as the point-estimate
crossing plus the 2.5/50/97.5 percentiles across bootstrap draws (each draw a fresh
shuffle and pitcher resample), with the share of draws that never cross within N_MAX.

SIGN CONVENTION: ridge_pred, the location value, and adjT are ALL expected runs from the
pitcher's perspective, LOWER = BETTER, and all three are used UNNEGATED as predictors.
A better measure therefore has a LARGER POSITIVE correlation with next season's adjT.
Nothing is negated, so nothing can be double-negated.

Data rules: reads workdir caches; writes one JSON to the workdir. No pitcher names.
Never committed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import fair_criterion as fc

N_MAX = 250          # pitchers must have at least this many graded four-seams
CRIT_FLOOR = 100     # and this many in the following season
GRID = [10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 250]
REPS = 400           # shuffles for the point estimate; r is averaged, never the values
N_BOOT = 1000        # draws for the band: each is a fresh shuffle + a pitcher resample
# The August run used 60 / 300; METHODS.md Part C asked for more before citing.
VALUE = "adjT"       # per-pitch run value used for both the criterion and "results"
MEASURES = ["stuff", "loc", "blend", "results", "pitch"]
LABELS = {"stuff": "Stuff+", "loc": "Location+", "blend": "Stuff+ and Location+",
          "results": "2025 adjusted run value", "pitch": "Pitching+"}
# Pitching+ is the shipped composite exactly as 08_staff_scores.py builds Pitch100: an
# EQUAL-weight mean of the three z-scored components (Stuff+, Location+, adjusted
# results), each from the same n-pitch subsample. Nothing is fit, so nothing can be fit
# on the 2026 criterion it is scored against. The z-scaling is within the pool at each n
# (08 scales on the 100+ FF population; with equal weights the difference is only which
# population sets each component's SD).

# Extending past N_MAX. The fixed pool stops at N_MAX because every pitcher must be able
# to supply n pitches. Higher thresholds are DIFFERENT, selected pools (high-volume arms:
# starters, healthier, plausibly better), so they are reported separately and never
# spliced onto the fixed-pool curves. Below MIN_EXT_POOL pitchers a correlation's CI is too
# wide to read and the threshold is reported as a count only.
EXT_T = [300, 350, 400, 450, 500, 600, 700, 800]
MIN_EXT_POOL = 60
# Projection for the fixed pool: R^2(n) = A * n / (n + k), the reliability_curves.py form
# (A = ceiling^2 * persistence * share, k = noise/signal), fit to the observed curve over
# FIT_FROM..N_MAX and to every bootstrap draw for the band.
FIT_FROM = 20
PROJ_GRID = [250, 300, 400, 500, 600, 800]


def load_pool():
    """Per-pitcher arrays of the three per-pitch quantities, plus the next-season
    criterion, for pitchers clearing both floors."""
    data, score_wd, crit_wd = fc.workdirs()
    s = ffc._frame(data, score_wd, "2024,2025")
    s = s[(s["year"] == 2025) & s["ridge_pred"].notna() & s["loc"].notna()
          & s[VALUE].notna()]

    # eval role 2025 of the 2025/2026 pair is the real 2026 season
    c = ffc._frame(data, crit_wd, "2025,2026")
    c = c[c[VALUE].notna()]
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=(VALUE, "size"), crit=(VALUE, "mean"))
    k = k[k["n26"] >= CRIT_FLOOR]

    counts = s.groupby("PitcherId").size()
    ids = counts[counts >= N_MAX].index.intersection(k.index)
    s = s[s["PitcherId"].isin(ids)]
    arrays = {pid: (g["ridge_pred"].to_numpy(), g["loc"].to_numpy(),
                    g[VALUE].to_numpy())
              for pid, g in s.groupby("PitcherId")}
    crit = k.loc[list(arrays), "crit"].to_numpy() * 100.0
    print(f"  pool: {len(arrays)} pitchers with {N_MAX}+ graded four-seams and "
          f"{CRIT_FLOOR}+ the next season")
    full = np.array([a[2].mean() for a in arrays.values()]) * 100.0
    return arrays, crit, full


PAIRS = [("loc", "stuff"), ("results", "stuff"), ("results", "loc")]


def crossing(r_up, r_down):
    """First n (log-interpolated) where r_up rises to meet r_down; None if it never does.
    0 if it already starts at or above."""
    d = np.asarray(r_up) - np.asarray(r_down)
    if d[0] >= 0:
        return float(GRID[0])
    for i in range(1, len(GRID)):
        if d[i] >= 0:
            t = -d[i - 1] / (d[i] - d[i - 1])
            lo, hi = np.log(GRID[i - 1]), np.log(GRID[i])
            return float(np.exp(lo + t * (hi - lo)))
    return None


def sb_fit(r, grid=None):
    """Fit r(n) = a * sqrt(n / (n + k)) (Spearman-Brown attenuation of a mean of n noisy
    pitches) by least squares of 1/r^2 on 1/n. Returns (a^2, k), or None if any r <= 0."""
    r = np.asarray(r, dtype=float)
    grid = GRID if grid is None else grid
    if (r <= 0).any():
        return None
    slope, icpt = np.polyfit(1.0 / np.array(grid), 1.0 / r ** 2, 1)
    if icpt <= 0:
        return None
    return 1.0 / icpt, max(slope / icpt, 0.0)


def sb_crossing(r_up, r_down, n_cap=20000):
    """Pitch count where the fitted curves meet, extrapolating past N_MAX if needed.
    Returns None when the fitted ceilings say the slower curve never catches up."""
    fu, fd = sb_fit(r_up), sb_fit(r_down)
    if fu is None or fd is None:
        return None
    (a2u, ku), (a2d, kd) = fu, fd
    ns = np.exp(np.linspace(np.log(1), np.log(n_cap), 4000))
    d = a2u * ns / (ns + ku) - a2d * ns / (ns + kd)
    hit = np.nonzero(d >= 0)[0]
    return float(ns[hit[0]]) if len(hit) else None


def rep_means(arrays, rng, grid=None):
    """One shuffle per pitcher; running means read off at every grid point.
    Returns array [pitcher, measure_raw(3), grid]."""
    grid = GRID if grid is None else grid
    out = np.empty((len(arrays), 3, len(grid)))
    idx = np.array(grid) - 1
    for i, (ridge, loc, tgt) in enumerate(arrays.values()):
        perm = rng.permutation(len(ridge))[:grid[-1]]
        for j, arr in enumerate((ridge[perm], loc[perm], tgt[perm])):
            out[i, j] = np.cumsum(arr)[idx] / np.array(grid)
    return out


def projection(point_r, boot_r):
    """Fixed-pool projection past N_MAX from the R^2 = A n/(n+k) form, fit on the
    observed grid from FIT_FROM. Band = 2.5/97.5 pct of the per-draw fits; draws whose fit
    fails (a non-positive r, or a non-positive asymptote) are counted, not dropped."""
    import reliability_curves as rc
    g = [n for n in GRID if n >= FIT_FROM]
    sl = slice(GRID.index(g[0]), None)

    def curve(r):
        f = sb_fit(np.asarray(r)[sl], g)
        if f is None:
            return None, None
        A, k = f
        return np.sqrt(rc.reliability_curve(PROJ_GRID, np.sqrt(A), 1.0, 1.0, k, 1.0)), (A, k)

    pt, par = curve(point_r)
    draws = [curve(b)[0] for b in boot_r]
    ok = np.array([d for d in draws if d is not None])
    fail = 1 - len(ok) / len(draws)
    lo, hi = (np.percentile(ok, [2.5, 97.5], axis=0) if len(ok) else (None, None))
    return dict(grid=PROJ_GRID,
                r=None if pt is None else [round(float(v), 4) for v in pt],
                lo=None if lo is None else [round(float(v), 4) for v in lo],
                hi=None if hi is None else [round(float(v), 4) for v in hi],
                asymptote_r=None if par is None else round(float(np.sqrt(par[0])), 4),
                k_noise_over_signal=None if par is None else round(float(par[1]), 1),
                fit_fail_share=round(float(fail), 4))


def main() -> int:
    global REPS, N_BOOT, VALUE
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default=VALUE, choices=["adjT", "Target"],
                    help="Target reproduces the August raw-run-value version")
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default=None, help="output dir (default: the score workdir)")
    a = ap.parse_args()
    REPS, N_BOOT, VALUE = a.reps, a.boot, a.value
    if VALUE != "adjT":
        LABELS["results"] = "2025 run value"
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
        return dict(stuff=st, loc=lo, blend=(z(st) + z(lo)) / 2, results=re,
                    pitch=(z(st) + z(lo) + z(re)) / 3)

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

    print(f"\n  validity vs next season's {VALUE} [{fc.resolve_adj_method()}] (r, positive = sorts correctly)")
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
    match = {}
    for m in MEASURES:
        hit = None
        for gi, n in enumerate(GRID):
            if (bdiff[m][:, gi] < 0).mean() <= 0.95:   # not significantly below
                hit = n
                break
        match[m] = hit
        d = bdiff[m][:, GRID.index(hit)] if hit else None
        extra = (f"   (diff {d.mean():+.3f} +/-{d.std():.3f})" if hit is not None else "")
        print(f"    {LABELS[m]:<24}{hit if hit else 'not within ' + str(N_MAX)}{extra}")

    cross = {}
    print("\n  crossovers (pitches where the first measure overtakes the second):")
    for up, dn in PAIRS:
        pt = crossing(curves[up], curves[dn])
        draws = [crossing(bse[up][b], bse[dn][b]) for b in range(N_BOOT)]
        within = float(np.mean([v is not None for v in draws]))
        # Fitted-curve crossing, extrapolated past N_MAX. Draws where the slower curve's
        # fitted ceiling stays below the other's never cross; they enter the percentiles
        # as +inf so the upper bound reads "never" rather than being silently dropped.
        sb_pt = sb_crossing(curves[up], curves[dn])
        sb = np.array([sb_crossing(bse[up][b], bse[dn][b]) for b in range(N_BOOT)],
                      dtype=object)
        sbv = np.array([np.inf if v is None else v for v in sb], dtype=float)
        q = [float(v) for v in np.percentile(sbv, [2.5, 50, 97.5])]
        cross[f"{up}_over_{dn}"] = dict(
            grid_point=None if pt is None else round(pt, 1),
            p_cross_within_n_max=round(within, 4),
            fit_point=None if sb_pt is None else round(sb_pt, 1),
            fit_q025_q50_q975=[None if not np.isfinite(v) else round(v, 1) for v in q],
            fit_p_never=round(float(np.isinf(sbv).mean()), 4))
        print(f"    {LABELS[up]} over {LABELS[dn]}: on grid {pt}, "
              f"P(cross by {N_MAX}) {within:.1%}; fitted {sb_pt} "
              f"[95%: {q[0]:.0f}, {q[2]:.0f}], P(never) {np.isinf(sbv).mean():.1%}")

    # ---- projection past N_MAX on the fixed pool
    proj = {}
    print(f"\n  fixed-pool projection (R^2 = A n/(n+k), fit on {FIT_FROM}..{N_MAX}):")
    for m in MEASURES:
        proj[m] = projection(curves[m], bse[m])
        pj = proj[m]
        band = ("unusable: fit failed" if pj["r"] is None else
                "  ".join(f"{n}: {r:+.3f} [{l:+.3f},{h:+.3f}]" for n, r, l, h in
                          zip(PROJ_GRID, pj["r"], pj["lo"], pj["hi"])))
        print(f"    {LABELS[m]:<24} k={pj['k_noise_over_signal']}  "
              f"fail {pj['fit_fail_share']:.1%}  {band}")

    # ---- higher-volume pools, reported separately (different population)
    ext = []
    print("\n  higher-volume pools (NOT the fixed pool; selected high-volume arms):")
    for T in EXT_T:
        sub = {pid: v for pid, v in arrays.items() if len(v[0]) >= T}
        e = dict(threshold=T, pitchers=len(sub))
        if len(sub) >= MIN_EXT_POOL:
            keep = np.array([len(v[0]) >= T for v in arrays.values()])
            cs, g2 = crit[keep], [N_MAX, T]
            re_ = np.random.default_rng(1000 + T)
            pa = {m: np.zeros(2) for m in MEASURES}
            for _ in range(REPS):
                vals = rep_means(sub, re_, g2)
                for gi in range(2):
                    mv = measures_at(vals, gi, np.arange(len(sub)))
                    for m in MEASURES:
                        pa[m][gi] += corr(mv[m], cs) / REPS
            bb = {m: np.zeros((N_BOOT, 2)) for m in MEASURES}
            for b in range(N_BOOT):
                vals = rep_means(sub, re_, g2)
                ix = re_.integers(0, len(sub), len(sub))
                for gi in range(2):
                    mv = measures_at(vals, gi, ix)
                    for m in MEASURES:
                        bb[m][b, gi] = corr(mv[m], cs[ix])
            e.update(at=g2, curves={m: dict(r=[round(float(v), 4) for v in pa[m]],
                                             se=[round(float(bb[m][:, i].std()), 4)
                                                 for i in range(2)]) for m in MEASURES})
            print(f"    {T}+: {len(sub)} pitchers  " + "  ".join(
                f"{LABELS[m]} {pa[m][0]:+.3f}->{pa[m][1]:+.3f}" for m in MEASURES))
        else:
            print(f"    {T}+: {len(sub)} pitchers (below {MIN_EXT_POOL}; not estimated)")
        ext.append(e)

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
               reference=dict(label="Full 2025 season of the criterion value",
                              r=round(ref_r, 4), pitches=round(n_full, 1)),
               crossing=match, crossovers=cross, criterion=VALUE,
               projection=proj, extension_pools=ext, min_ext_pool=MIN_EXT_POOL,
               adj_method=fc.resolve_adj_method())
    dest = os.path.join(a.out or fc.workdirs()[1], "coach_sample_curve.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
