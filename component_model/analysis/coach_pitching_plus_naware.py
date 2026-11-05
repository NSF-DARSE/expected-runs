"""Can four-seam Pitching+ be read below 200 four-seams if each part counts only as much as it can
be trusted at the pitcher's pitch count? Pre-registration and verdicts:
docs/notes/pitching-plus-naware-ledger-2026-09.md. Read that first.

WHY. The page mutes Pitching+ below 200 four-seams (the app's SAMPLE_FLOORS.pitch), and the
median D1 arm throws about 120, so for most of a staff the headline number is greyed out. The
reason is structural, not a lack of data: Pitching+ is script 08's equal-z blend, (z Stuff+ +
z Location+ + z results) / 3, with the same one-third on results at 40 pitches as at 400, and
results are the component that needs hundreds of pitches. The Pitching+ mix card already tells a
coach that on a small sample "this is mostly the pitch's shape"; the number does not do that.

THE CANDIDATE, NAWARE. Each component's z is scaled by
    k_c(n) = min(1, rel_c(n) / rel_c(N_REF)),   rel_c(n) = n / (n + n0_c),   N_REF = 200,
where n is the pitcher's four-seam count and n0_c is the component's game-split noise ratio,
measured on the graded season alone with script 16's estimator. At n >= 200, k = 1 for every
component and NAWARE is script 08's number exactly, so no Pitching+ a coach can see today
moves. Below 200 each component is pulled toward the population mean in proportion to how much
less it can be trusted than at 200: Stuff+ (n0 of a few pitches) barely moves, results move
most. Nothing is fitted to an outcome: n0 comes from within-season splits, N_REF is the app's
existing floor.

THE INCUMBENT, SHIPPED, is script 08 reproduced: the shipped four-seam ridge, the pooled
location map trained on 2024, mean adjT; each z'd against 2025 pitchers with 100+ four-seams;
equal thirds.

THE TEST. Pool and criterion are the frozen gate's pair C: 15+ four-seams and 10%+ share in 2025
(score build) and 2026 (criterion build), criterion = 2026 mean four-seam adjT. Primary stratum:
pool pitchers under N_REF four-seams, the only ones whose forecast changes.
Delta = r(NAWARE, criterion) - r(SHIPPED, criterion), 2000 paired bootstrap resamples of
pitchers (seed 20260817). "Improves" iff P(Delta > 0) >= 0.95 and |Delta| >= 1 SE.

READABILITY (no criterion). For fixed weights a blend of component means is the mean of a
per-pitch composite, so script 16's reliability estimator applies to it directly. SHIPPED's
curve is one composite; NAWARE's at count n is the composite with weights k_c(n). The proposed
floor is the smallest n (step 5) at which NAWARE's reliability reaches SHIPPED's reliability at
200: the same standard as today's floor, met sooner.

SIGN CONVENTION: every component and the criterion are expected run value from the pitcher's
perspective, LOWER = BETTER, and z-scores here keep that orientation, so a working forecast
correlates POSITIVELY with the criterion. The display layer negates once (script 08); this file
never does.

Data rules: reads the existing D1 pitch caches only and refuses to run if one is missing or would
be rebuilt; writes one JSON of aggregates to the score workdir. No pitcher names, no per-pitcher
output, no absolute paths. Usage, from component_model/analysis with STUFFPLUS_DATA,
STUFFPLUS_DATA_CRIT, STUFFPLUS_WORKDIR and STUFFPLUS_WORKDIR_CRIT set:
    python coach_pitching_plus_naware.py --dry   # n0, reliability, floor, pools, power
    python coach_pitching_plus_naware.py
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import fair_criterion as fc

sf = importlib.import_module("16_season_floor")

# ---- frozen gate constants (coach_incremental_gate.py) ----
SHARE = 0.10
ABS_MIN = 15
SEED = 20260817
# ---- pre-registered design constants (ledger) ----
N_BOOT = 2000
N_REF = 200          # the app's Pitching+ floor today (derive.ts SAMPLE_FLOORS.pitch)
POP_MIN = 100        # script 08's scale population: 2025 pitchers with 100+ four-seams
MIN_HALF = 20        # script 16's default game-split half minimum
COMPS = [("ridge", "ridge_pred"), ("loc", "loc"), ("adj", "adjT")]
BINS = [(15, 50), (50, 100), (100, 200), (200, None)]
REL_GRID = [15, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200]
FLOOR_STEP = 5
R_ASSUMED = 0.24     # one-season forecast r on this pool, for the power calculation only


# ---------------- pure pieces (tested) ----------------

def rel(n, n0):
    n = np.asarray(n, dtype=float)
    return n / (n + n0) if np.isfinite(n0) else np.zeros_like(n)


def k_factor(n, n0, n_ref=N_REF):
    """Weight on a component at count n relative to its weight at n_ref, capped at 1."""
    r_ref = rel(n_ref, n0)
    if r_ref <= 0:
        return np.zeros_like(np.asarray(n, dtype=float))
    return np.minimum(1.0, rel(n, n0) / r_ref)


def blend(zs, n, n0s, aware):
    """(sum_c k_c z_c) / len(zs); k = 1 when not aware. zs and n0s keyed by component."""
    out = 0.0
    for c, z in zs.items():
        k = k_factor(n, n0s[c]) if aware else 1.0
        out = out + k * np.asarray(z, dtype=float)
    return out / len(zs)


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def dep_corr_se(r, r23, n):
    """SE of r12 - r13 for two predictors of one criterion with r12 = r13 = r and
    corr(predictors) = r23 (Pearson-Filon / Steiger 1980 asymptotic covariance)."""
    psi = r23 * (1 - 2 * r * r) - 0.5 * r * r * (1 - 2 * r * r - r23 * r23)
    var = (2 * (1 - r * r) ** 2 - 2 * psi) / n
    return float(np.sqrt(max(var, 0.0)))


def floor_for(rel_fn, target, lo=ABS_MIN, hi=N_REF, step=FLOOR_STEP):
    """Smallest n on the step grid in [lo, hi] with rel_fn(n) >= target; None if never."""
    for n in range(lo, hi + 1, step):
        if rel_fn(n) >= target:
            return n
    return None


# ---------------- data ----------------

def _refuse_rebuild(data, workdir, years):
    tag = "" if years == "2024,2025" else "_" + years.replace(",", "_")
    cache = os.path.join(workdir, f"pitches_cache{tag}_D1.parquet")
    if not os.path.exists(cache):
        sys.exit(f"refusing to run: no cache for {years}; this harness never builds one")
    import pyarrow.parquet as pq
    have = set(pq.read_schema(cache).names)
    header = pd.read_csv(data, nrows=0).columns
    want = fc.USECOLS + [c for c in fc.OPTIONAL_COLS if c in header] + fc.ANCHOR_COLS
    stale = [c for c in want if c not in have]
    if stale:
        sys.exit(f"refusing to run: the {years} cache would be REBUILT (missing {stale}); "
                 "check STUFFPLUS_DATA / STUFFPLUS_DATA_CRIT")


def ff_rows(score):
    """Script 08's four-seam rows: shipped ridge, plate location present, pooled map on 2024."""
    ff, _model, feats = fc.ridge_for_group(score, "FF")
    ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
    ff["loc"] = fc.PooledLocationMap(train).apply(ff)
    s25 = ff[ff["year"] == 2025].dropna(subset=[col for _, col in COMPS])
    return s25, feats


def scale(s25):
    """Script 08's population moments: pitcher means over 2025 pitchers with 100+ four-seams."""
    g = s25.groupby("PitcherId")
    pt = g.agg(n=("adjT", "size"), **{c: (col, "mean") for c, col in COMPS})
    qual = pt[pt["n"] >= POP_MIN]
    return pt, {c: float(qual[c].mean()) for c, _ in COMPS}, \
        {c: float(qual[c].std()) for c, _ in COMPS}


def composite_rel(s25, mu, sd, weights):
    """Script 16 reliability curve for the per-pitch composite sum_c w_c (x_c - mu_c)/sd_c."""
    v = sum(w * (s25[col] - mu[c]) / sd[c] for (c, col), w in zip(COMPS, weights))
    frame = s25[["PitcherId", "GameID", "Date"]].assign(v=v.values)
    return sf.reliability_curve(frame, "v", MIN_HALF)


def pool(score, crit, pt):
    tot = score[score["year"] == 2025].groupby("PitcherId").size()
    p = pt[(pt["n"] >= ABS_MIN) & (pt["n"] / tot.reindex(pt.index) >= SHARE)]
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")
    c = crit[fc.pitch_mask(crit, "FF") & (crit["year"] == 2025)]   # the gate's criterion mask
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]
    return p.join(k[["crit"]], how="inner").dropna()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="n0, reliability, floor, pools and power; no criterion correlation")
    args = ap.parse_args()
    t0 = time.time()
    data, score_wd, crit_wd = fc.workdirs()
    data_crit = os.environ.get("STUFFPLUS_DATA_CRIT", data)
    _refuse_rebuild(data, score_wd, "2024,2025")
    _refuse_rebuild(data_crit, crit_wd, "2025,2026")
    score = fc.load_frame(data, score_wd, "2024,2025")
    crit = fc.load_frame(data_crit, crit_wd, "2025,2026")
    for col in ("GameID", "Date"):
        if col not in score.columns:
            sys.exit(f"{col} missing from the score cache; the game-split estimator needs it")
    s25, feats = ff_rows(score)
    if list(feats) != list(fc.FEATS):
        sys.exit("the shipped four-seam feature list is not fc.FEATS; script 08 and this "
                 "harness would grade different models")
    pt, mu, sd = scale(s25)
    print("  frames and grades in %.0fs; 2025 four-seam rows %d, scale population %d" % (
        time.time() - t0, len(s25), int((pt["n"] >= POP_MIN).sum())), flush=True)

    # ---- component noise ratios, graded season only ----
    n0s = {}
    for c, col in COMPS:
        res = sf.reliability_curve(s25[["PitcherId", "GameID", "Date", col]], col, MIN_HALF)
        n0s[c] = float(res["n0"])
        print("    n0 %-5s %7.1f   (panel %d, half r %.3f)" % (c, n0s[c], res["pitchers"],
                                                          res["half_r"]))
    k_grid = {c: {n: round(float(k_factor(n, n0s[c])), 3) for n in REL_GRID} for c in n0s}
    print("    k at n: " + "  ".join("%s %s" % (c, [k_grid[c][n] for n in (30, 50, 100, 150)])
                                    for c in n0s) + "   (n = 30, 50, 100, 150)")

    # ---- readability: SHIPPED's composite, and NAWARE's at each count ----
    ship = composite_rel(s25, mu, sd, [1 / 3] * 3)
    n0_ship = float(ship["n0"])
    target = float(rel(N_REF, n0_ship))
    cache = {}

    def rel_naware(n):
        if n not in cache:
            w = [float(k_factor(n, n0s[c])) / 3 for c, _ in COMPS]
            cache[n] = float(rel(n, float(composite_rel(s25, mu, sd, w)["n0"])))
        return cache[n]

    rel_curve = {n: {"shipped": round(float(rel(n, n0_ship)), 3),
                     "naware": round(rel_naware(n), 3)} for n in REL_GRID}
    floor = floor_for(rel_naware, target)
    print("    SHIPPED Pitching+ n0 %.1f, reliability at %d = %.3f" % (n0_ship, N_REF, target))
    print("    reliability by n (shipped / naware): " + "  ".join(
        "%d %.2f/%.2f" % (n, v["shipped"], v["naware"]) for n, v in rel_curve.items()))
    print("    proposed NAWARE floor (reliability >= SHIPPED at %d): %s" % (N_REF, floor),
          flush=True)

    # ---- the pool, forecasts, and power ----
    tab = pool(score, crit, pt)
    n = tab["n"].values
    zs = {c: (tab[c].values - mu[c]) / sd[c] for c, _ in COMPS}
    f_ship = blend(zs, n, n0s, aware=False)
    f_aw = blend(zs, n, n0s, aware=True)
    strata = {"under_ref": n < N_REF, "full_pool": np.ones(len(n), bool)}
    for lo, hi in BINS:
        strata["n%d_%s" % (lo, hi or "up")] = (n >= lo) & (n < (hi or np.inf))
    power = {}
    for name, m in strata.items():
        r23 = _r(f_ship[m], f_aw[m]) if m.sum() > 2 else float("nan")
        se = dep_corr_se(R_ASSUMED, r23, int(m.sum())) if np.isfinite(r23) else float("nan")
        power[name] = {"n": int(m.sum()), "r_shipped_naware": round(r23, 4),
                       "expected_se": round(se, 4), "mde_at_095": round(1.645 * se, 4)}
        print("    [%s] n=%d  r(shipped, naware) %.4f  expected SE %.4f  MDE %.4f" % (
            name, m.sum(), r23, se, 1.645 * se))

    out = {"n_boot": N_BOOT, "seed": SEED, "n_ref": N_REF, "pop_min": POP_MIN,
           "min_half": MIN_HALF, "n0": n0s, "k_grid": k_grid, "n0_shipped": n0_ship,
           "rel_target": target, "rel_curve": rel_curve, "proposed_floor": floor,
           "power": power}
    if args.dry:
        print("\n  dry run: nothing written   total %.0fs" % (time.time() - t0))
        return 0

    # ---- the read ----
    crit_v = tab["crit"].values
    variants = {"results_only": {"ridge": 0.0, "loc": 0.0}, "loc_only": {"ridge": 0.0, "adj": 0.0}}
    reads = {}
    rng = np.random.default_rng(SEED)
    for name, m in strata.items():
        idx = np.where(m)[0]
        if name.startswith("n200"):
            reads[name] = {"n": int(len(idx)), "note": "identical by construction (k = 1)",
                           "r_shipped": _r(f_ship[idx], crit_v[idx])}
            continue
        rs, ra = _r(f_ship[idx], crit_v[idx]), _r(f_aw[idx], crit_v[idx])
        d = []
        for _ in range(N_BOOT):
            b = rng.choice(idx, len(idx))
            d.append(_r(f_aw[b], crit_v[b]) - _r(f_ship[b], crit_v[b]))
        d = np.array(d)
        rec = {"n": int(len(idx)), "r_shipped": rs, "r_naware": ra, "delta": ra - rs,
               "delta_boot_mean": float(d.mean()), "delta_se": float(d.std(ddof=1)),
               "delta_ci": [float(x) for x in np.percentile(d, [2.5, 97.5])],
               "p_delta_positive": float((d > 0).mean())}
        # descriptive decomposition: which component's down-weighting carries the change
        for vname, frozen in variants.items():
            n0v = {c: (0.0 if c in frozen else n0s[c]) for c in n0s}  # n0 0 -> k 1
            rec["r_" + vname] = _r(blend(zs, n, n0v, aware=True)[idx], crit_v[idx])
        reads[name] = rec
        print("  [%s n=%d] shipped r %+.4f  naware r %+.4f  Delta %+.4f  SE %.4f  "
              "CI [%+.4f,%+.4f]  P(Delta>0)=%.3f   (results-only %+.4f, loc-only %+.4f)" % (
                  name, len(idx), rs, ra, ra - rs, rec["delta_se"], rec["delta_ci"][0],
                  rec["delta_ci"][1], rec["p_delta_positive"], rec["r_results_only"],
                  rec["r_loc_only"]), flush=True)
    out["reads"] = reads
    dest = os.path.join(score_wd, "coach_pitching_plus_naware.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n  wrote %s   total %.0fs" % (os.path.basename(dest), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
