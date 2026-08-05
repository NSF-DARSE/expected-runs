"""13: Which metric is most predictive of next-season adjT, as a function of
how many pitches you have?

Motivation: Pitching+ explains ~24% of stable skill (script 12) using full
qualifying seasons (100+ FF). At a portal-evaluation sample size (50-300
pitches), a metric's usable predictive power isn't its asymptotic ceiling --
it's ceiling penalized by how much sampling noise still contaminates a mean
built from that few pitches. adjT itself has the highest ceiling (it IS the
target) but the most per-pitch noise; Stuff+ has the least noise but the
lowest ceiling; Location+ sits between both. The most predictive metric at
a given n therefore depends on n, and the reliability curves cross.

Part A  per-pitch noise scale for adjT, Stuff+ (ridge_pred), and Location+
        (loc), via the same game-clustered half-split estimator as script 12,
        repeated over many random game splits for a bootstrap CI (script 12's
        Part A uses one deterministic parity split; this asked for a
        distribution over random splits instead).
Part B  each metric's asymptotic predictive ceiling for next-season adjT:
        cross-year covariance identifies the stable component (noise and
        drift are fresh, independent draws each season and contribute ~0 in
        expectation), then an attenuation correction removes the residual
        bias from finite per-pitcher-season sample size.
Part C  R(n) reliability curves for all three metrics, plus numerically
        solved pairwise crossover pitch counts.

Reuses fair_criterion.py's xT/adjT/ridge/location primitives and
variance_components.py's noise-scale estimator verbatim. The new math (random
game splits, cross-metric ceiling, R(n)/crossover) lives in
reliability_curves.py, unit-tested there against synthetic panels with a
known truth -- same reason variance_components.py is a separate, importable
module rather than inline in a numbered script.

Input: pitch-cache parquets written by fc.load_pitches (--caches), same as
script 12. Writes nothing outside --workdir except a reliability-curve grid
(numbers, not a rendered chart). Never commit output: Level II per-pitcher data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import reliability_curves as rc

SEASONS = [2024, 2025, 2026]
TRAIN_YEAR = 2024
METRICS = [("adjT", "adjT"), ("ridge_pred", "Stuff+"), ("loc", "Location+")]


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-ff", type=int, default=fc.PANEL_MIN_FF)
    ap.add_argument("--min-half", type=int, default=25)
    ap.add_argument("--splits", type=int, default=200,
                     help="Random game-split iterations for the noise-scale bootstrap.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-grid", default="10,3000",
                     help="Pitch-count range (log-spaced) for the reliability chart.")
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches and --workdir, or STUFFPLUS_CACHES / STUFFPLUS_WORKDIR.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


# ---------------- panel construction (mirrors script 12's build_panel) ----------------

def load_seasons(paths):
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        print(f"  read {os.path.basename(p)}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    return df


def build_panel(args):
    """Same qualifying-FF frame as script 12's build_panel, plus 'loc' and GameID
    retained for the random-split noise estimator. xT fit once, pooled across all
    three seasons; Stuff+ ridge and the location map trained on TRAIN_YEAR only,
    matching the suite's fixed references (see fair_criterion.py docstring)."""
    df = load_seasons(args.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)
    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    lmap = fc.PooledLocationMap(ff[(ff["year"] == TRAIN_YEAR) & ff["xT"].notna()])
    ff["loc"] = lmap.apply(ff)

    n_by = ff.groupby(["PitcherId", "year"]).size().rename("n_ff")
    ok = n_by[n_by >= args.min_ff].reset_index()[["PitcherId", "year"]]
    ff = ff.merge(ok, on=["PitcherId", "year"], how="inner")
    seasons_per = ff.groupby("PitcherId")["year"].nunique()
    ff = ff[ff["PitcherId"].isin(seasons_per[seasons_per >= 2].index)].copy()

    print(f"PANEL: {ff['PitcherId'].nunique()} pitchers, "
          f"{ff.groupby(['PitcherId','year']).ngroups} pitcher-seasons "
          f"({args.min_ff}+ FF, 2+ qualified seasons)")
    return ff


# ---------------- Part A: random-split noise scale, bootstrapped ----------------

def noise_scale_bootstrap(ff, value_col, args, rng):
    """Per-pitch noise scale for value_col, averaged over --splits random game
    splits. Returns (mean, lo95, hi95, per_split_array).

    Each split reuses vc.effective_noise_scale (script 12's exact estimator:
    n * gap^2 / 4, averaged across pitcher-seasons) via rc.random_game_half --
    nothing about the underlying formula changes, only that which games fall
    in which half is redrawn each iteration instead of fixed once.
    """
    import variance_components as vc
    key = ff["PitcherId"].astype(str) + "|" + ff["year"].astype(str)
    vals = []
    for _ in range(args.splits):
        split = rc.random_game_half(ff, rng)
        tmp = split[[value_col, "half"]].copy()
        tmp["ps"] = key.values
        eff, n_ps = vc.effective_noise_scale(tmp, value_col, "ps", min_half=args.min_half)
        if not np.isnan(eff):
            vals.append(eff)
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi), vals


def part_a(ff, args, rng):
    print("\n" + "=" * 78)
    print(f"PART A -- PER-PITCH NOISE SCALE ({args.splits} random game splits)")
    print("=" * 78)
    print(f"{'metric':<12}{'noise/pitch':>13}{'95% CI':>22}")
    out = {}
    for col, label in METRICS:
        mean, lo, hi, _ = noise_scale_bootstrap(ff, col, args, rng)
        print(f"{label:<12}{mean:>13.6f}   [{lo:.6f}, {hi:.6f}]")
        out[label] = {"noise_per_pitch": mean, "ci": (lo, hi)}
    return out


# ---------------- Part B: cross-metric ceiling ----------------

def season_table(ff):
    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({
        "n": g["adjT"].size(),
        "adjT": g["adjT"].mean(),
        "ridge_pred": g["ridge_pred"].mean(),
        "loc": g["loc"].mean(),
    }).reset_index().rename(columns={"PitcherId": "pitcher", "year": "season"})
    for col, _ in METRICS:
        tab[col] = tab[col] - tab.groupby("season")[col].transform("mean")
    return tab


def part_b(tab, noise, args):
    print("\n" + "=" * 78)
    print("PART B -- ASYMPTOTIC CEILING vs NEXT-SEASON adjT")
    print("=" * 78)
    print(f"{'metric':<12}{'raw r':>9}{'rel(x)':>9}{'rel(adjT)':>11}{'ceiling':>10}{'n pairs':>9}")
    out = {}
    noise_adjt = noise["adjT"]["noise_per_pitch"]
    for col, label in METRICS:
        res = rc.cross_metric_ceiling(tab, col, "adjT", noise[label]["noise_per_pitch"], noise_adjt)
        print(f"{label:<12}{res['raw_r']:>9.3f}{res['rel_x']:>9.3f}"
              f"{res['rel_y']:>11.3f}{res['ceiling']:>10.3f}{res['n_pairs']:>9}")
        out[label] = res
    if out["adjT"]["ceiling"] < out["Location+"]["ceiling"] - 0.02:
        print("  NOTE: adjT's own ceiling should be >= every other metric's, since "
              "no metric can out-predict the thing it's predicting. A material "
              "violation here means the attenuation correction or the panel is off.")
    return out


# ---------------- Part C: reliability curves + crossovers ----------------

def part_c(noise, ceilings, tab, args):
    print("\n" + "=" * 78)
    print("PART C -- RELIABILITY CURVES AND CROSSOVERS")
    print("=" * 78)
    curves = {}
    for col, label in METRICS:
        npp = noise[label]["noise_per_pitch"]
        sig_var = rc.signal_variance(tab, col, npp)
        ceil = ceilings[label]["ceiling"]
        persistence_x = rc.persistence_of(tab, col, npp)
        # rel_y from part_b is share_stable_of("adjT"), the SAME quantity for
        # every metric since Y is always next-season adjT -- reused here
        # rather than recomputed.
        share_y = ceilings[label]["rel_y"]
        curves[label] = lambda n, npp=npp, sig_var=sig_var, ceil=ceil, \
            persistence_x=persistence_x, share_y=share_y: rc.reliability_curve(
            n, ceil, persistence_x, share_y, npp, sig_var)
        print(f"{label:<12}ceiling={ceil:.3f}  persistence_x={persistence_x:.3f}  "
              f"share_y={share_y:.3f}  noise/pitch={npp:.6f}  signal_var={sig_var:.6f}  "
              f"half-reliability n={npp/sig_var:.0f}")

    lo, hi = [float(x) for x in args.n_grid.split(",")]
    labels = [l for _, l in METRICS]
    print("\nPairwise crossovers (pitch count where the two curves are equal):")
    crossovers = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            xo = rc.solve_crossover(curves[a], curves[b], lo, hi)
            crossovers[(a, b)] = xo
            if xo is None:
                print(f"  {a} vs {b}: no crossover in [{lo:.0f}, {hi:.0f}]")
            else:
                print(f"  {a} vs {b}: n = {xo:.0f}")

    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    chart = pd.DataFrame({"n": grid})
    for label in labels:
        chart[label] = curves[label](grid)
    chart_path = os.path.join(args.workdir, "reliability_vs_n.csv")
    chart.to_csv(chart_path, index=False)
    print(f"\nCurve grid written to {chart_path} (plot with any tool; kept as "
          f"data, not a rendered image, per the no-derived-output-in-repo rule).")
    return curves, crossovers


# ---------------- Part D: precision-weighted blend ----------------

def part_d(tab, noise, curves, args):
    """GLS blend of all three metrics vs. each one alone, across the same
    pitch-count grid as Part C. 'Out-of-sample' here is the closed-form
    population R^2 the GLS solution implies, NOT an empirical train/test
    split -- this machine has no real held-out data to check against (see the
    module docstring). Treat these numbers as a plausibility check on the
    blend's structure, and validate the actual weights/gains empirically once
    real STUFFPLUS_CACHES are available.
    """
    print("\n" + "=" * 78)
    print("PART D -- PRECISION-WEIGHTED BLEND")
    print("=" * 78)
    metric_cols = [col for col, _ in METRICS]
    noises = {col: noise[label]["noise_per_pitch"] for col, label in METRICS}

    lo, hi = [float(x) for x in args.n_grid.split(",")]
    grid = np.logspace(np.log10(lo), np.log10(hi), 30)
    rows = []
    for n in grid:
        weights, r2_blend = rc.optimal_blend(tab, metric_cols, "adjT", noises, n)
        row = {"n": n, "blend_R2": r2_blend}
        for col, label in METRICS:
            row[f"weight_{label}"] = weights[col]
            row[f"{label}_R2"] = float(curves[label](n))
        rows.append(row)
    blend_tab = pd.DataFrame(rows)

    print(f"{'n':>8}{'blend R2':>10}" + "".join(f"{l+' R2':>12}" for _, l in METRICS)
          + "".join(f"{'w('+l+')':>12}" for _, l in METRICS))
    for _, r in blend_tab.iloc[[0, len(blend_tab) // 4, len(blend_tab) // 2,
                                3 * len(blend_tab) // 4, -1]].iterrows():
        print(f"{r['n']:>8.0f}{r['blend_R2']:>10.3f}"
              + "".join(f"{r[l+'_R2']:>12.3f}" for _, l in METRICS)
              + "".join(f"{r['weight_'+l]:>12.4f}" for _, l in METRICS))

    gain = blend_tab["blend_R2"] - blend_tab[[f"{l}_R2" for _, l in METRICS]].max(axis=1)
    print(f"\nBlend gain over the best single metric: min {gain.min():.4f}, "
          f"max {gain.max():.4f} (should be >= 0 everywhere by construction).")

    out_path = os.path.join(args.workdir, "blend_vs_n.csv")
    blend_tab.to_csv(out_path, index=False)
    print(f"Full weights-vs-n table written to {out_path}.")
    return blend_tab


PORTAL_PITCH_COUNTS = [50, 150, 300, 600, 1200, 2500]


def part_e(tab, noise, curves, args):
    """The coach-facing deliverable: at realistic portal sample sizes, which
    single metric (or the blend) is most predictive, and how confident should
    that read be. Same closed-form-population caveat as Part D.
    """
    print("\n" + "=" * 78)
    print("PART E -- TRANSFER PORTAL SAMPLE-SIZE TABLE")
    print("=" * 78)
    metric_cols = [col for col, _ in METRICS]
    labels = [l for _, l in METRICS]
    noises = {col: noise[label]["noise_per_pitch"] for col, label in METRICS}

    print(f"{'pitches':>9}" + "".join(f"{l+' R2':>12}" for l in labels)
          + f"{'blend R2':>10}{'best option':>16}")
    rows = []
    for n in PORTAL_PITCH_COUNTS:
        r2_by_label = {label: float(curves[label](n)) for label in labels}
        _, r2_blend = rc.optimal_blend(tab, metric_cols, "adjT", noises, n)
        best = max(list(r2_by_label.items()) + [("Blend", r2_blend)], key=lambda kv: kv[1])
        print(f"{n:>9}" + "".join(f"{r2_by_label[l]:>12.3f}" for l in labels)
              + f"{r2_blend:>10.3f}{best[0]:>16}")
        rows.append({"pitches": n, **{f"{l}_R2": r2_by_label[l] for l in labels},
                     "blend_R2": r2_blend, "best_option": best[0]})

    print("\nRead as: at this many pitches, this option explains this share of "
          "next-season adjT variance. 'Best option' is whichever is highest,\n"
          "not necessarily by a meaningful margin -- check the full table before "
          "treating a close call as a real difference.")
    out_path = os.path.join(args.workdir, "portal_sample_size_table.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Table written to {out_path}.")
    return pd.DataFrame(rows)


def main():
    args = cli()
    rng = np.random.default_rng(args.seed)
    ff = build_panel(args)
    noise = part_a(ff, args, rng)
    tab = season_table(ff)
    ceilings = part_b(tab, noise, args)
    curves, _ = part_c(noise, ceilings, tab, args)
    part_d(tab, noise, curves, args)
    part_e(tab, noise, curves, args)


if __name__ == "__main__":
    main()
