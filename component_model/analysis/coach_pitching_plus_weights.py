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

SIX THINGS A CONSUMER OF THIS FILE MUST HONOUR:

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
  5. COMPOSITE ELIGIBILITY IS A GATE VERDICT, NOT A DISPLAY PREFERENCE. Every pitch type
     carries "composite_eligible" plus a printable "composite_ineligible_reason", read from
     coach_incremental_gate.json at build time so re-running the gate is what moves the flag.
     An ineligible type withholds BOTH its blended score AND its Stuff+ -- see item 6 for why
     the second half of that is not obvious. Weights and params are still emitted, because the
     remaining components (Location+ where it exists, recent results) are still shown and still
     need them. It FAILS CLOSED: an unreadable gate file marks every type ineligible, because
     the wrong default here ships a confident composite for a pitch whose Stuff+ was never
     shown to add anything (see item 3, and the sinker).
  6. AN INELIGIBLE TYPE SHOWS NO STUFF+ AT ALL. Through v2 this contract said to show Stuff+
     and recent results side by side for an ineligible type, on the reasoning that withholding
     only the blend keeps the components honest and visible. Jack ruled otherwise on
     2026-08-20: a Stuff+ we could not show adds anything over a pitcher's own results is not
     information a coach should be handed next to the results it failed to beat, because it
     will be read as a second opinion rather than as noise. The app implements the ruling
     (isStuffPlusConfirmed, src/lib/pitchingPlusMix.ts) and this contract now states it, so a
     consumer reading only this file does not reintroduce the display that was removed.

KNOWN LIMITATION, now load-bearing. reliability_curves.optimal_blend takes each off-diagonal
of Sigma as Cov(a_i, a_j) alone, i.e. it assumes same-season cross-component noise is
negligible. Its own docstring flags this and says to validate empirically "if the blend weights
end up load-bearing". They now are, and that check has NOT been run. Until it is, treat
"blend_r2" as the closed form's own estimate of itself rather than as validated out-of-sample
performance. The check to run is a direct empirical fit on held-out seasons compared against
the closed form; a material gap means the off-diagonal assumption is doing damage and the
weights should fall back to reliability-ranked rather than GLS. The weights themselves do not
depend on this being clean, but the reported R^2 does.

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
CONTRACT_VERSION = 3   # v3: an ineligible type withholds its Stuff+, not just the blend


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
    ap.add_argument("--gate", default=None,
                    help="coach_incremental_gate.json. Defaults to the one in --workdir. "
                         "Its verdicts decide composite_eligible per pitch type; if the file "
                         "is absent every type is marked INELIGIBLE (fail closed).")
    ap.add_argument("--seed", type=int, default=20260820)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches and --workdir, or STUFFPLUS_CACHES / STUFFPLUS_WORKDIR. "
                 "--caches wants both pitch_cache parquets so 2024, 2025 and 2026 are all in "
                 "one frame; the variance-components math needs 2+ seasons per pitcher.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    if not args.gate:
        args.gate = os.path.join(args.workdir, "coach_incremental_gate.json")
    return args


def load_gate(path):
    """The incremental-validity gate's verdicts, or None if they cannot be read.

    Composite eligibility is NOT a judgement this script makes. coach_incremental_gate.py
    already asks the only question that matters -- does Stuff+ add anything over what the
    pitcher's own prior results already say -- against a pre-registered bar, and re-running
    that gate must be what moves the flag, not an edit here.

    FAILS CLOSED on purpose. A missing, unparseable, or structurally wrong gate file yields
    None, and every pitch type is then marked ineligible. The alternative default would ship
    a blended coach-facing number for a pitch type whose Stuff+ has never been shown to add
    anything, which is the exact failure the gate exists to prevent.
    """
    try:
        with open(path) as fh:
            g = json.load(fh)
        by = g["by_pitch"]
        if not isinstance(by, dict):
            raise TypeError("by_pitch is %s, expected object" % type(by).__name__)
    except Exception as e:
        print("  gate NOT read (%s: %s) -- every type marked composite-ineligible"
              % (type(e).__name__, e))
        return None, {"available": False, "path": path,
                      "error": "%s: %s" % (type(e).__name__, e)}
    meta = {"available": True, "path": path,
            "share": g.get("share"), "n_boot": g.get("n_boot"),
            "pass_bar": g.get("pass_bar")}
    print("  gate read: %s" % ", ".join(
        "%s=%s" % (k, v.get("verdict")) for k, v in by.items()))
    return by, meta


def eligibility(grp, gate, meta):
    """Per-type composite_eligible plus a reason a coach-facing app can print verbatim."""
    if gate is None:
        return {"composite_eligible": False,
                "composite_ineligible_reason":
                    "The incremental-validity gate has not been run, so no pitch type has a "
                    "Stuff+ cleared for display yet.",
                "gate": {"verdict": None}}
    row = gate.get(grp)
    if not isinstance(row, dict) or "verdict" not in row:
        return {"composite_eligible": False,
                "composite_ineligible_reason":
                    "This pitch type has never been tested against the incremental-validity "
                    "gate, so its Stuff+ is withheld.",
                "gate": {"verdict": None}}
    seen = {"verdict": row.get("verdict"), "n": row.get("n"),
            "p_gain_positive": row.get("p_gain_positive"),
            "p_semipartial_positive": row.get("p_semipartial_positive"),
            "blend_gain": row.get("blend_gain"),
            "pass_bar": meta.get("pass_bar")}
    if row.get("verdict") == "PASS":
        return {"composite_eligible": True, "composite_ineligible_reason": None, "gate": seen}
    p, bar = row.get("p_gain_positive"), meta.get("pass_bar")
    detail = ("" if p is None or bar is None
              else " (%.0f%% confident it helps, against a %.0f%% bar)"
                   % (100 * p, 100 * bar))
    return {"composite_eligible": False,
            "composite_ineligible_reason":
                "Stuff+ has not been shown to add anything for this pitch type beyond what "
                "the pitcher's own recent results already say%s, so it is withheld rather "
                "than shown alongside them." % detail,
            "gate": seen}


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
    gate, gate_meta = load_gate(args.gate)
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
                         "min_half": args.min_half, "gate": gate_meta},
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
               "The weights themselves are conservative under the clamp.",
               "Gate on by_pitch[t]['composite_eligible'] before showing any blended score. "
               "It is false for a type whose Stuff+ never cleared the incremental-validity "
               "bar, and false for EVERY type when the gate file could not be read, so an "
               "absent gate withholds composites rather than allowing them.",
               "An ineligible type shows NO Stuff+ -- not the blend and not the component "
               "on its own. Show its other components (Location+ where present, recent "
               "results) with the sample each one has. Withholding only the blend and "
               "leaving Stuff+ beside the results was the v2 guidance and was reversed: a "
               "grade that did not beat the pitcher's own results reads to a coach as a "
               "second opinion rather than as noise.",
               "A withheld Stuff+ must render as absent, never as a number. Rounding a null "
               "to 0 ships '0' as a confident grade, which is worse than the grade this rule "
               "was written to withhold."],
           "n_grid": N_GRID, "by_pitch": {}}

    for grp in ORDER:
        try:
            row = per_type(graded_frame(df, grp, lmap), grp, args, rng)
        except Exception as e:  # one thin pitch type must not lose the other five
            print("")
            print("=== %s   FAILED: %s: %s" % (grp, type(e).__name__, e))
            row = {"skipped": "%s: %s" % (type(e).__name__, e)}
        # Stamped even on the skipped path: a type with no weights must not read as eligible.
        row.update(eligibility(grp, gate, gate_meta))
        out["by_pitch"][grp] = row

    dest = os.path.join(args.workdir, "coach_pitching_plus_weights.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
