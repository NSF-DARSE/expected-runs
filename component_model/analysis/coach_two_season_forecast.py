"""Does a second season of data make Pitching+ a better forecast for a returning pitcher?
Pre-registration and verdicts: docs/notes/two-season-forecast-ledger-2026-09.md. Read that first.

WHY. Every grade on the pitcher page is built from ONE season, yet most of a college staff
returns, and script 12 found that measurement noise is ~70% of a single season's variance and
named multi-season pooling as the unused lever ("the larger lever is precision"). A coach
already reads two seasons of results for a returning arm. This asks whether the forecast
should too, and which part of it gains: results are the noisiest component (hundreds of
pitches to stabilise), Stuff+ the least (a dozen), so pooling should help results most and
may even hurt Stuff+ if a pitcher's stuff changed over the winter.

THE FORECAST is the equal-z blend RESULTS.md validated (script 05/07): z(results) + z(Stuff+)
+ z(Location+) for the four-seam, z(results) + z(Stuff+) for every other graded type (Location+
is four-seam only). Components per pitcher-season, all expected run value from the pitcher's
perspective, LOWER = BETTER:
  results  mean adjT over the type's graded pitches (the frozen gate's "prior")
  Stuff+   mean ridge_pred from the model that ships for the type (fc.ridge_for_group,
           trained on the 2024 score-build rows; its 2024 predictions are in-sample, measured
           inflation -0.001 in coach_model_two_panel.py, so accepted)
  Location+ four-seam only, the shipped pooled catcher-frame map on xT trained on 2024; the
           2024 values are CROSS-FITTED (5 pitcher folds, seed 20260924) so a pitcher's own
           pitches never value their own 2024 locations.
Pool and criterion are the frozen gate's pair C exactly: 15+ pitches and 10%+ share in 2025
(score build) and in 2026 (criterion build), criterion = 2026 mean adjT on the type. A pitcher
is RETURNING with 15+ graded pitches of the type in 2024.

A TWO-SEASON component is the pitch-weighted mean of 2025 and 2024 with the older season
discounted by lambda: (n25 x25 + lambda n24 x24) / (n25 + lambda n24); a non-returning pitcher
keeps their 2025 value. Four fixed candidates, nothing fitted:
  P1  every component pooled, lambda 1      PH  every component pooled, lambda 0.5
  R1  results pooled only, lambda 1         RH  results pooled only, lambda 0.5
ONE is the incumbent: every component from 2025 only.

SCOPE. The primary test is on RETURNING pitchers, the only ones whose forecast changes; the
full pool is reported beside it. A scope with fewer than MIN_READ pitchers is not read.

THE TEST is one cross-fitted read per scope. Pitchers are split into two fixed halves (seed 20260925,
blind to outcomes). The candidate with the higher forecast r on one half is applied to the
other, so every pitcher's two-season forecast was chosen without their own criterion.
Delta = r(cross-fitted forecast, criterion) - r(ONE, criterion) over the scope. 2000
bootstrap resamples of pitchers within each half repeat the choice and the scoring, so the
selection is inside the uncertainty. "Improves" iff P(Delta > 0) >= 0.95.

AUDIT (no bar): do the physical grades still add once results cover two seasons?
r(z R_pooled + z Stuff+ [+ z Location+]) - r(R_pooled), next to the one-season analogue.

SIGN CONVENTION: a correctly oriented forecast correlates POSITIVELY with the criterion (both
lower-is-better run value). Nothing is negated.

Data rules: reads the existing D1 pitch caches only and refuses to run if a cache is missing or
would be rebuilt; writes one JSON of aggregates to the score workdir. No pitcher names, no
per-pitcher output, no absolute paths. Usage, from component_model/analysis with
STUFFPLUS_DATA, STUFFPLUS_DATA_CRIT, STUFFPLUS_WORKDIR and STUFFPLUS_WORKDIR_CRIT set:
    python coach_two_season_forecast.py --dry   # pools and returning shares, no criterion
    python coach_two_season_forecast.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import fair_criterion as fc

# ---- frozen gate constants (coach_incremental_gate.py) ----
SHARE = 0.10
ABS_MIN = 15
SEED = 20260817
# ---- pre-registered design constants (ledger) ----
N_BOOT = 2000
N_FOLDS = 5
FOLD_SEED = 20260924
HALF_SEED = 20260925
GROUPS = ["FF", "SL", "CB", "CH", "SI"]
MIN_READ = 100   # a scope with fewer pitchers is reported, not read
CANDIDATES = {"P1": ("all", 1.0), "PH": ("all", 0.5), "R1": ("R", 1.0), "RH": ("R", 0.5)}


# ---------------- pure pieces (tested) ----------------

def pooled(x25, n25, x24, n24, lam):
    """Pitch-weighted two-season mean with the older season discounted by lam. A 2024 value
    counts only when n24 >= ABS_MIN and it is finite; otherwise the 2025 value stands."""
    x25, n25, x24, n24 = (np.asarray(a, dtype=float) for a in (x25, n25, x24, n24))
    ok = np.isfinite(x24) & np.isfinite(n24) & (n24 >= ABS_MIN)
    w24 = np.where(ok, lam * np.nan_to_num(n24), 0.0)
    return (n25 * x25 + w24 * np.where(ok, x24, 0.0)) / (n25 + w24)


def _z(x):
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd > 0 else x * 0.0


def composite(tab, comps, which, lam):
    """Equal-z forecast. which = 'none' (ONE), 'all', or a component name to pool alone."""
    parts = []
    for c in comps:
        if which == "all" or which == c:
            x = pooled(tab[c + "25"], tab["n" + c + "25"], tab[c + "24"], tab["n" + c + "24"], lam)
        else:
            x = tab[c + "25"].values
        parts.append(_z(x))
    return np.sum(parts, axis=0)


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def crossfit(tab, comps, half):
    """(cross-fitted forecast, picks): each half's forecast uses the candidate that scored
    best on the OTHER half. Composites are z-scored over the whole table (predictors only)."""
    crit = tab["crit"].values
    fc_all = {k: composite(tab, comps, w, lam) for k, (w, lam) in CANDIDATES.items()}
    out = np.empty(len(tab))
    picks = {}
    for h in (0, 1):
        other = half != h
        best = max(CANDIDATES, key=lambda k: _r(fc_all[k][other], crit[other]))
        picks[h] = best
        out[half == h] = fc_all[best][half == h]
    return out, picks


def halves(pids, seed=HALF_SEED):
    """Fixed half label per pitcher from a seeded permutation of the sorted ids."""
    ids = np.sort(pd.unique(pids))
    lab = np.random.default_rng(seed).permutation(len(ids)) % 2
    return dict(zip(ids, lab))


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


def location_values(score):
    """Per-row pooled Location+ value for four-seams in 2024 (cross-fitted) and 2025."""
    ff = score[score["is_ff"] & score["year"].isin([2024, 2025])
               & score["PlateLocSide"].notna() & score["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
    vals = pd.Series(np.nan, index=ff.index)
    e25 = ff["year"] == 2025
    vals[e25] = fc.PooledLocationMap(train).apply(ff[e25]).values
    all_p = np.sort(score["PitcherId"].dropna().unique())
    perm = np.random.default_rng(FOLD_SEED).permutation(len(all_p))
    fold = ff["PitcherId"].map(dict(zip(all_p, perm % N_FOLDS)))
    for f in range(N_FOLDS):
        m = (ff["year"] == 2024) & (fold == f)
        vals[m] = fc.PooledLocationMap(train[fold.loc[train.index] != f]).apply(ff[m]).values
    return ff[["PitcherId", "year"]].assign(L=vals)


def season_table(score, crit, grp, loc):
    """One row per pool pitcher: components by season, pitch counts, criterion."""
    graded, _, _ = fc.ridge_for_group(score, grp)
    mask = score["is_ff"] if grp == "FF" else fc.pitch_mask(score, grp)
    graded = graded[mask.reindex(graded.index).fillna(False).astype(bool)]
    g = graded.groupby(["PitcherId", "year"]).agg(n=("adjT", "size"), R=("adjT", "mean"),
                                                  S=("ridge_pred", "mean"))
    tab = pd.DataFrame(index=g.xs(2025, level="year").index)
    for y, t in ((2025, "25"), (2024, "24")):
        gy = g.xs(y, level="year") if y in g.index.get_level_values("year") else g.iloc[0:0]
        for c in ("R", "S"):
            tab[c + t] = gy[c].reindex(tab.index)
            tab["n" + c + t] = gy["n"].reindex(tab.index)
    if grp == "FF":
        lg = loc.groupby(["PitcherId", "year"])["L"].agg(["size", "mean"])
        for y, t in ((2025, "25"), (2024, "24")):
            ly = lg.xs(y, level="year")
            tab["L" + t] = ly["mean"].reindex(tab.index)
            tab["nL" + t] = ly["size"].reindex(tab.index)
    tot = score[score["year"] == 2025].groupby("PitcherId").size()
    tab = tab[(tab["nR25"] >= ABS_MIN) & (tab["nR25"] / tot.reindex(tab.index) >= SHARE)]
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")
    c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]
    comps = ["R", "S", "L"] if grp == "FF" else ["R", "S"]
    tab = tab.join(k[["crit"]], how="inner").dropna(subset=[c + "25" for c in comps] + ["crit"])
    tab["returning"] = tab["nR24"].fillna(0) >= ABS_MIN
    return tab, comps


# ---------------- the read ----------------

def read(tab, comps, half):
    crit = tab["crit"].values
    one = composite(tab, comps, "none", 1.0)
    cf, picks = crossfit(tab, comps, half)
    rpool_1 = pooled(tab["R25"], tab["nR25"], tab["R24"], tab["nR24"], 1.0)
    phys = [_z(tab[c + "25"].values) for c in comps if c != "R"]
    point = {"r_one": _r(one, crit), "r_crossfit": _r(cf, crit), "picks": picks,
             "by_candidate": {k: _r(composite(tab, comps, w, lam), crit)
                              for k, (w, lam) in CANDIDATES.items()},
             "r_results_one": _r(tab["R25"].values, crit),
             "r_results_pooled": _r(rpool_1, crit),
             "phys_adds_one": _r(_z(tab["R25"].values) + np.sum(phys, axis=0), crit)
             - _r(tab["R25"].values, crit),
             "phys_adds_pooled": _r(_z(rpool_1) + np.sum(phys, axis=0), crit) - _r(rpool_1, crit)}
    point["delta"] = point["r_crossfit"] - point["r_one"]
    rng = np.random.default_rng(SEED)
    idx_h = [np.where(half == h)[0] for h in (0, 1)]
    deltas, pa1, pap = [], [], []
    for _ in range(N_BOOT):
        ix = np.concatenate([rng.choice(i, len(i)) for i in idx_h])
        tb, hb = tab.iloc[ix], half[ix]
        cb = tb["crit"].values
        o = composite(tb, comps, "none", 1.0)
        f, _ = crossfit(tb, comps, hb)
        deltas.append(_r(f, cb) - _r(o, cb))
        rp = pooled(tb["R25"], tb["nR25"], tb["R24"], tb["nR24"], 1.0)
        ph = np.sum([_z(tb[c + "25"].values) for c in comps if c != "R"], axis=0)
        pa1.append(_r(_z(tb["R25"].values) + ph, cb) - _r(tb["R25"].values, cb))
        pap.append(_r(_z(rp) + ph, cb) - _r(rp, cb))
    d, pa1, pap = np.array(deltas), np.array(pa1), np.array(pap)
    point.update({
        "delta_boot_mean": float(d.mean()), "delta_se": float(d.std(ddof=1)),
        "delta_ci": [float(x) for x in np.percentile(d, [2.5, 97.5])],
        "p_delta_positive": float((d > 0).mean()),
        "p_phys_adds_one": float((pa1 > 0).mean()),
        "p_phys_adds_pooled": float((pap > 0).mean()),
        "phys_adds_pooled_ci": [float(x) for x in np.percentile(pap, [2.5, 97.5])]})
    return point


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="pools and returning shares only; touches no criterion correlation")
    args = ap.parse_args()
    t0 = time.time()
    data, score_wd, crit_wd = fc.workdirs()
    data_crit = os.environ.get("STUFFPLUS_DATA_CRIT", data)
    _refuse_rebuild(data, score_wd, "2024,2025")
    _refuse_rebuild(data_crit, crit_wd, "2025,2026")
    score = fc.load_frame(data, score_wd, "2024,2025")
    crit = fc.load_frame(data_crit, crit_wd, "2025,2026")
    loc = location_values(score)
    print("  frames and location values in %.0fs" % (time.time() - t0), flush=True)
    out = {"n_boot": N_BOOT, "seed": SEED, "half_seed": HALF_SEED, "abs_min": ABS_MIN,
           "share": SHARE, "candidates": CANDIDATES, "by_pitch": {}}
    for grp in GROUPS:
        tab, comps = season_table(score, crit, grp, loc)
        lab = halves(tab.index.values)
        half = np.array([lab[p] for p in tab.index])
        ret = tab["returning"]
        print("\n=== %s  n=%d pitchers, returning %d (%.0f%%), halves %d/%d, components %s ===" % (
            grp, len(tab), ret.sum(), 100 * ret.mean(), (half == 0).sum(), (half == 1).sum(),
            comps), flush=True)
        if args.dry:
            print("    [dry] median 2025 pitches %d; returning median 2024 pitches %d" % (
                tab["nR25"].median(), tab.loc[ret, "nR24"].median() if ret.any() else 0))
            continue
        rec = {"n": int(len(tab)), "n_returning": int(ret.sum())}
        for scope, sel in (("returning", ret.values), ("full_pool", np.ones(len(tab), bool))):
            if sel.sum() < MIN_READ:
                rec[scope] = {"skipped": "fewer than %d pitchers" % MIN_READ, "n": int(sel.sum())}
                print("  [%s n=%d] below %d, not read" % (scope, sel.sum(), MIN_READ))
                continue
            p = read(tab[sel], comps, half[sel])
            rec[scope] = dict(p, n=int(sel.sum()))
            print("  [%s n=%d] one-season r %+.4f   two-season (cross-fitted, picks %s) r %+.4f"
                  % (scope, sel.sum(), p["r_one"], p["picks"], p["r_crossfit"]))
            print("    Delta %+.4f  boot mean %+.4f  SE %.4f  CI [%+.4f,%+.4f]  P(Delta>0)=%.3f"
                  % (p["delta"], p["delta_boot_mean"], p["delta_se"], p["delta_ci"][0],
                     p["delta_ci"][1], p["p_delta_positive"]))
            print("    candidates (descriptive): %s" % {
                k: round(v, 4) for k, v in p["by_candidate"].items()})
            print("    results alone: one-season %+.4f, two-season %+.4f" % (
                p["r_results_one"], p["r_results_pooled"]))
            print("    physics adds over results: one-season %+.4f (P %.3f), over two-season "
                  "results %+.4f (P %.3f)" % (p["phys_adds_one"], p["p_phys_adds_one"],
                                              p["phys_adds_pooled"], p["p_phys_adds_pooled"]),
                  flush=True)
        out["by_pitch"][grp] = rec
    if args.dry:
        print("\n  dry run: nothing written   total %.0fs" % (time.time() - t0))
        return 0
    dest = os.path.join(score_wd, "coach_two_season_forecast.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n  wrote %s   total %.0fs" % (os.path.basename(dest), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
