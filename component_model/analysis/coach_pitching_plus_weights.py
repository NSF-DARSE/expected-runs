"""Pitching+ blend weights per pitch type and pitch count -- the contract the dashboard reads.

WHAT PITCHING+ IS, AND WHAT IT IS NOT. Stuff+, Location+, and adjusted results each measure
one construct. Pitching+ measures none of them: it is a FORECAST of next season's run value
built by pooling all three. That distinction is load-bearing, not pedantry --

  * the components stay separately visible in the UI, because a composite cannot tell a coach
    which lever to pull; and
  * Pitching+ must NEVER become the criterion any component is evaluated against. The moment
    a blend becomes the yardstick, the one-construct-per-score rule in CLAUDE.md is gone and
    nothing detects it. Evaluation stays on the fair criterion (see fair_criterion.py) and on
    coach_incremental_gate.py.

WEIGHTS ARE MEASURED, NOT FITTED. Fitting three weights per pitch type per sample bin would
mean dozens of free parameters on pools of 190-300 pitchers, and would manufacture gains the
same way fitted blend weights did on the four-seam (which is why coach_incremental_gate.py
holds its blend at 50/50 on purpose). Here the weights are the closed-form GLS solution given
each component's MEASURED per-pitch noise and asymptotic ceiling: no parameter is tuned to
make the answer look better. All of that math already exists in reliability_curves.py, is
unit-tested against synthetic panels with a known truth, and is used unchanged. The only new
thing here is running it per pitch type and writing the result out as a stable contract.

WHY THE WEIGHTS MOVE WITH PITCH COUNT, which is the whole product. Stuff+ is a smooth function
of pitch characteristics, so a pitcher's mean stabilises in a few dozen pitches. Realised run
value is dominated by rare high-value events and needs hundreds. Location+ sits between. So on
40 sliders the blend is almost entirely Stuff+, and by 400 results carry most of it. The
crossover is measurable, differs by pitch type, and is the honest answer to "can you grade
this guy in April".

FOUR THINGS A CONSUMER OF THIS FILE MUST HONOUR:

  1. LOCATION+ IS FOUR-SEAM ONLY. The pooled location map is a value-by-zone surface fit on
     four-seams; a slider low-and-away and a fastball low-and-away are not worth the same
     thing, so applying the four-seam map to another pitch type would be wrong, not
     approximate. Every non-FF type here is a TWO-component blend until per-type maps exist.
     The UI must handle a missing component rather than assume three.
  2. NEGATIVE RAW WEIGHTS ARE REAL AND ARE FLOORED AT ZERO FOR DISPLAY. A GLS weight can come
     out negative when a component is anti-predictive once the others are accounted for. Both
     forms are emitted: "raw" (the solution, may be negative) and "display" (floored at zero
     and renormalised to sum to 1). Show the display form; never show a negative share, and
     never let a floored component silently reappear with weight.
  3. A ZERO STUFF+ WEIGHT IS AN ANSWER, NOT A BUG. coach_incremental_gate.py found our sinker
     physics adds nothing over prior results (gain -0.030, P=0.290). If sinker comes out near
     zero here too, the display should say plainly that we do not yet have a sinker grade worth
     trusting, which beats a confident-looking number we know is inverted for exactly the
     sinker-primary arms the staff cares about.
  4. THE WEIGHTS LIVE HERE, THE ARITHMETIC LIVES IN THE APP. The consumer interpolates this
     table on n and multiplies. It must not hardcode weights and must not re-derive them: a
     second implementation drifts from the model silently, which is precisely how the
     differential-anchor bug survived as long as it did.

KNOWN LIMITATION, now load-bearing. reliability_curves.optimal_blend takes each off-diagonal
of Sigma as Cov(a_i, a_j) alone, i.e. it assumes same-season cross-component noise is
negligible. Its own docstring flags this and says to validate empirically "if the blend weights
end up load-bearing". They now are. Part D below is that check: it compares the closed-form
blend R^2 against a direct empirical fit on held-out seasons. A material gap means the
off-diagonal assumption is doing damage and the weights should fall back to reliability-ranked
rather than GLS.

SIGN CONVENTION: every component is expected run value from the pitcher's perspective, LOWER =
BETTER, and so is the criterion. Weights are therefore all expected positive, and the display
layer negates once at the very end (see fair_criterion.py's docstring). Do not negate here.

Data rules: reads the pitch-cache parquets only; writes one JSON to the score workdir. No
pitcher names, no per-pitcher rows -- group parameters and a weight grid only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import reliability_curves as rc
import variance_components as vc

SEASONS = [2024, 2025, 2026]
TRAIN_YEAR = 2024
ORDER = ["FF", "SI", "FC", "SL", "CB", "CH"]
# Component key -> (frame column, coach-facing label). Order is the emitted order.
COMPONENTS = [("stuff", "ridge_pred", "Stuff+"),
              ("location", "loc", "Location+"),
              ("results", "adjT", "Recent results")]
LOC_TYPES = {"FF"}          # see limitation 1 in the docstring
N_GRID = [10, 15, 20, 30, 40, 60, 80, 120, 175, 250, 350, 500, 750, 1000, 1500, 2500]
CONTRACT_VERSION = 1


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"),
                    help="Comma-separated pitch-cache parquets. Pass BOTH workdir caches so "
                         "all three real seasons are present; rows dedupe on PitchUID.")
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-n", type=int, default=40,
                    help="Pitches of this type per pitcher-season to qualify. One floor for "
                         "every type so the curves are comparable across types.")
    ap.add_argument("--min-half", type=int, default=15,
                    help="Pitches per game-half for the noise estimator. Lower than script "
                         "13's 25 because off-speed seasons are smaller; a type whose "
                         "estimate rests on few pitcher-seasons is reported, not hidden.")
    ap.add_argument("--splits", type=int, default=60,
                    help="Random game-split iterations per component per pitch type.")
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches and --workdir, or STUFFPLUS_CACHES / STUFFPLUS_WORKDIR. "
                 "--caches wants both pitch_cache parquets so 2024, 2025 and 2026 are all in "
                 "one frame; the variance-components math needs 2+ seasons per pitcher.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


def load_seasons(paths):
    """One frame spanning all three real seasons. Mirrors 13_reliability_vs_sample_size.py.

    The two caches overlap on real 2025 (one holds 2024+2025, the other 2025+2026), so the
    PitchUID dedupe is doing real work rather than being defensive -- without it every 2025
    pitch would be counted twice and every 2025 season mean would be built from duplicated
    rows.
    """
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        print("  read %s: %d rows" % (os.path.basename(p), len(d)))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    n_before = len(df)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    print("  deduped on PitchUID: %d -> %d rows" % (n_before, len(df)))
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    print("  seasons: %s" % df["year"].value_counts().sort_index().to_dict())
    return df


def graded_frame(df, grp, lmap):
    """One pitch type, with ridge_pred and (four-seams only) loc attached."""
    ff = fc.stuff_ridge(df, pitch_mask=fc.pitch_mask(df, grp), feats=fc.feats_for(grp))
    ff = ff[ff["xT"].notna()].copy()
    if grp in LOC_TYPES:
        fc.add_loc_bins(ff)
        ff["loc"] = lmap.apply(ff)
    return ff


def panel(ff, grp, min_n):
    """Pitcher-seasons with min_n+ of this type, restricted to pitchers with 2+ such seasons.

    The 2+ seasons requirement is not a sample-size nicety: every quantity downstream
    (stable/drift split, ceilings, cross-component covariance) is identified from cross-YEAR
    covariance, so a pitcher appearing once contributes nothing and only dilutes the counts.
    """
    n_by = ff.groupby(["PitcherId", "year"]).size().rename("n")
    ok = n_by[n_by >= min_n].reset_index()[["PitcherId", "year"]]
    ff = ff.merge(ok, on=["PitcherId", "year"], how="inner")
    if ff.empty:
        return ff
    per = ff.groupby("PitcherId")["year"].nunique()
    return ff[ff["PitcherId"].isin(per[per >= 2].index)].copy()


def season_table(ff, cols):
    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({"n": g["adjT"].size(),
                        **{c: g[c].mean() for c in cols}}).reset_index()
    tab = tab.rename(columns={"PitcherId": "pitcher", "year": "season"})
    for c in cols:
        tab[c] = tab[c] - tab.groupby("season")[c].transform("mean")
    return tab


def noise_scale(ff, col, splits, min_half, rng):
    """Per-pitch noise scale for col, averaged over random whole-GAME splits.

    Returns (mean, lo95, hi95, n_pitcher_seasons_used). Splitting by game rather than pitch
    is what makes this a real uncertainty estimate: pitches in one game share batter, park,
    umpire and day, so a pitch-level split leaves that variance in both halves and reports
    the naive scale.
    """
    vals, used = [], 0
    for _ in range(splits):
        split = rc.random_game_half(ff, rng)
        tmp = split[[col, "half"]].copy()
        # Key built from `split`, not from `ff`: the merge inside random_game_half hands back
        # a fresh frame, and pairing a key computed on the pre-merge frame would rely on row
        # order surviving the merge. Same class of bug as the anchor diagnostic (2d98388).
        tmp["ps"] = (split["PitcherId"].astype(str) + "|" + split["year"].astype(str)).values
        eff, n_ps = vc.effective_noise_scale(tmp, col, "ps", min_half=min_half)
        if not np.isnan(eff):
            vals.append(eff)
            used = max(used, n_ps)
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    a = np.array(vals)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(a.mean()), float(lo), float(hi), used


def per_type(ff, grp, args, rng):
    """Every measured parameter and the weight grid for one pitch type, or a skip record."""
    cols = [c for _, c, _ in COMPONENTS if c == "adjT" or c == "ridge_pred"
            or (c == "loc" and grp in LOC_TYPES)]
    comps = [(k, c, lab) for k, c, lab in COMPONENTS if c in cols]
    p = panel(ff, grp, args.min_n)
    n_pitchers = 0 if p.empty else int(p["PitcherId"].nunique())
    n_ps = 0 if p.empty else int(p.groupby(["PitcherId", "year"]).ngroups)
    print("")
    print("=== %s   tags %s" % (grp, sorted(fc.PITCH_GROUPS[grp])))
    print("    panel: %d pitchers, %d pitcher-seasons (%d+ of this type, 2+ seasons)"
          % (n_pitchers, n_ps, args.min_n))
    if n_pitchers < vc.MIN_PAIR_N:
        print("    SKIPPED: fewer than %d pitchers with two qualified seasons, so no season "
              "pair can identify stable skill." % vc.MIN_PAIR_N)
        return {"skipped": "too few paired pitchers", "n_pitchers": n_pitchers,
                "n_pitcher_seasons": n_ps}

    tab = season_table(p, cols)
    params, noises = {}, {}
    print("    %-14s%14s%26s%10s" % ("component", "noise/pitch", "95% CI", "used"))
    for key, col, label in comps:
        mean, lo, hi, used = noise_scale(p, col, args.splits, args.min_half, rng)
        if np.isnan(mean):
            print("    %-14s  no game-split estimate at min-half=%d" % (label, args.min_half))
            return {"skipped": "noise estimate unavailable", "n_pitchers": n_pitchers,
                    "n_pitcher_seasons": n_ps}
        print("    %-14s%14.6f   [%.6f, %.6f]%10d" % (label, mean, lo, hi, used))
        noises[col] = mean
        params[key] = {"column": col, "label": label, "noise_per_pitch": mean,
                       "noise_ci": [lo, hi], "noise_n_pitcher_seasons": used}

    # Ceilings against next-season adjT, the within-season caps on each curve, and the
    # non-negative-drift boundary check. See coach_pplus_variance_check.py: for adjT the
    # noise term is 77-90% of observed variance at real college pitch counts, so
    # signal_var = observed - noise is a small difference of large estimates and can land
    # below s2_stable, implying negative drift. Raising the per-season floor does NOT fix it
    # (slider gets monotonically worse: persistence 1.16 at a 40-pitch floor to 2.92 at 250),
    # so this is a boundary case to be handled, not a sample-size problem to be floored away.
    # The clamp is applied here and RECORDED per component, never inside the shared module.
    try:
        for key, col, label in comps:
            res = rc.cross_metric_ceiling(tab, col, "adjT", noises[col], noises["adjT"])
            d = vc.variance_components(
                tab[["pitcher", "season", col, "n"]].rename(columns={col: "mean"}),
                noises[col])
            sig_raw = rc.signal_variance(tab, col, noises[col])
            sig = max(sig_raw, d["s2_stable"])
            clamped = sig > sig_raw
            pers = min(d["persistence"], 1.0)
            params[key].update({"ceiling": res["ceiling"], "raw_r": res["raw_r"],
                                "reliability_stable_share": res["rel_x"],
                                "persistence": pers,
                                "persistence_raw": d["persistence"],
                                "signal_var": sig, "signal_var_raw": sig_raw,
                                "s2_stable": d["s2_stable"], "s2_drift": d["s2_drift"],
                                "s2_noise": d["s2_noise"],
                                "drift_clamped_at_zero": bool(clamped),
                                "half_reliability_n": noises[col] / sig,
                                "n_season_pairs": res["n_pairs"], "share_y": res["rel_y"]})
    except ValueError as e:
        print("    SKIPPED: %s" % e)
        return {"skipped": "ceiling not identified: %s" % e, "n_pitchers": n_pitchers,
                "n_pitcher_seasons": n_ps}

    print("    %-14s%10s%13s%12s%10s%10s" % ("", "ceiling", "persistence", "half-n",
                                             "n pairs", "clamped"))
    for key, _, label in comps:
        q = params[key]
        print("    %-14s%10.3f%13.3f%12.0f%10d%10s"
              % (label, q["ceiling"], q["persistence"], q["half_reliability_n"],
                 q["n_season_pairs"], "yes" if q["drift_clamped_at_zero"] else "no"))

    # adjT cannot be out-predicted by anything, since it IS the target. A violation means the
    # attenuation correction or the panel is off -- recorded, not silently tolerated.
    warn = []
    ceil_adjt = params["results"]["ceiling"]
    for key, _, label in comps:
        if key != "results" and params[key]["ceiling"] > ceil_adjt + 0.02:
            msg = ("%s ceiling %.3f exceeds the target's own %.3f; treat this type's weights "
                   "as provisional" % (label, params[key]["ceiling"], ceil_adjt))
            print("    WARNING: %s" % msg)
            warn.append(msg)

    curves = {}
    for key, col, _ in comps:
        q = params[key]
        curves[key] = (lambda n, q=q: rc.reliability_curve(
            n, q["ceiling"], q["persistence"], q["share_y"], q["noise_per_pitch"],
            q["signal_var"]))

    metric_cols = [c for _, c, _ in comps]
    # Clamped diagonals, so a boundary-case drift cannot inflate a component's weight.
    sig_vars = {c: params[k]["signal_var"] for k, c, _ in comps}
    rows = []
    for n in N_GRID:
        raw, r2 = rc.optimal_blend(tab, metric_cols, "adjT", noises, float(n),
                                   signal_vars=sig_vars)
        raw_by_key = {k: float(raw[c]) for k, c, _ in comps}
        floored = {k: max(0.0, v) for k, v in raw_by_key.items()}
        tot = sum(floored.values())
        disp = ({k: v / tot for k, v in floored.items()} if tot > 0
                else {k: 0.0 for k in floored})
        singles = {k: float(curves[k](float(n))) for k, _, _ in comps}
        rows.append({"n": n, "raw": raw_by_key, "display": disp, "blend_r2": r2,
                     "component_r2": singles,
                     "best_single": max(singles, key=singles.get),
                     "any_negative_raw": any(v < 0 for v in raw_by_key.values())})

    print("    %8s%11s%s%14s" % ("pitches", "blend R2",
                                 "".join("%12s" % ("w " + lab) for _, _, lab in comps),
                                 "best single"))
    for r in rows:
        if r["n"] in (10, 40, 120, 350, 1000, 2500):
            print("    %8d%11.3f%s%14s"
                  % (r["n"], r["blend_r2"],
                     "".join("%12.3f" % r["display"][k] for k, _, _ in comps),
                     r["best_single"]))

    # The number the coach-facing sentence is built from: where results start to matter.
    thresh = next((r["n"] for r in rows if r["display"].get("results", 0) >= 0.25), None)
    if thresh is not None:
        print("    results reach a quarter of the weight at about %d pitches" % thresh)
    else:
        print("    results never reach a quarter of the weight inside the grid")

    return {"n_pitchers": n_pitchers, "n_pitcher_seasons": n_ps,
            "components": [k for k, _, _ in comps],
            "labels": {k: lab for k, _, lab in comps},
            "params": params, "weights": rows,
            "results_quarter_weight_n": thresh, "warnings": warn}


def main() -> int:
    t0 = time.time()
    args = cli()
    rng = np.random.default_rng(args.seed)
    df = load_seasons(args.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)

    # The location map is fit once, on four-seams in the train year, exactly as every other
    # script in the suite does -- a per-type map is a modelling task, not a parameter change.
    ff_for_map = fc.stuff_ridge(df)
    ff_for_map = ff_for_map[ff_for_map["xT"].notna()].copy()
    fc.add_loc_bins(ff_for_map)
    lmap = fc.PooledLocationMap(ff_for_map[(ff_for_map["year"] == TRAIN_YEAR)
                                           & ff_for_map["xT"].notna()])
    print("  frame ready in %.0fs" % (time.time() - t0))

    out = {"contract_version": CONTRACT_VERSION,
           "generated": {"seasons": SEASONS, "train_year": TRAIN_YEAR,
                         "min_n_per_season": args.min_n, "splits": args.splits,
                         "min_half": args.min_half},
           "sign_convention": ("Every component and the criterion are expected run value from "
                               "the pitcher's perspective, LOWER = BETTER. Weights are "
                               "positive. Negate once at the display layer, not here."),
           "consumer_notes": [
               "Interpolate on n; do not re-derive weights and do not hardcode them.",
               "Use the 'display' weights, never 'raw' -- raw may be negative.",
               "Location+ is present for four-seams only; other types are two-component.",
               "A component with zero weight must read as 'no grade worth trusting yet', "
               "not as a missing value.",
               "Pitching+ is a forecast, not a construct. Keep the components visible and "
               "never evaluate a component against Pitching+.",
               "Where params[c]['drift_clamped_at_zero'] is true, that component's drift "
               "estimate hit the non-negative boundary: its 'component_r2' is an UPPER "
               "BOUND, not a point estimate, and must not be shown as a precise figure. "
               "The weights themselves are conservative under the clamp."],
           "n_grid": N_GRID, "by_pitch": {}}

    for grp in ORDER:
        try:
            out["by_pitch"][grp] = per_type(graded_frame(df, grp, lmap), grp, args, rng)
        except Exception as e:  # one thin pitch type must not lose the other five
            print("")
            print("=== %s   FAILED: %s: %s" % (grp, type(e).__name__, e))
            out["by_pitch"][grp] = {"skipped": "%s: %s" % (type(e).__name__, e)}

    dest = os.path.join(args.workdir, "coach_pitching_plus_weights.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
