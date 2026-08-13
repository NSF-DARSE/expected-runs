"""12: How much of year-over-year unpredictability is noise vs missing skill?

Splits the spread in pitcher-season fair-criterion performance three ways:
  bucket 1  measurement noise  -- a college season is too few pitches
  bucket 2  true drift         -- pitchers really change year to year
  bucket 3  missing skill      -- persistent talent Pitching+ does not encode
Only bucket 3 is a Pitching+ problem. Buckets 1 and 2 set the ceiling that no
static physical model can beat.

Part A  within-season split-half reliability of xT, split by GAME parity and
        Spearman-Brown corrected. Separates bucket 1 from bucket 2, which the
        across-season number alone cannot do.
Part B  method-of-moments variance decomposition over the three-season panel,
        then the same fit with physical Stuff+/Location+ added; the stable
        variance that survives is missing skill.

Game parity, not pitch parity: consecutive pitches in one game share batter,
park, umpire, and day effects, so a pitch-parity split (script 06) leaves that
shared variance in BOTH halves and overstates reliability. Expect a lower and
more honest number here.

Criterion is xT (defense/luck-stripped expected runs, LOWER = BETTER); adjT
prints as a robustness row. Four-seam, D1, 2024-2026.

Input: pitch-cache parquets written by fc.load_pitches (--caches). Regenerate
them by running script 01 once per year pair with --level D1 if absent. Years
are re-derived from Date, so role-relabeled caches are safe to pass.
Writes nothing outside --workdir. Never commit output: Level II per-pitcher data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import variance_components as vc

SEASONS = [2024, 2025, 2026]
TRAIN_YEAR = 2024


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"),
                    help="Comma-separated pitch-cache parquet paths. Years are "
                         "re-derived from Date, so relabeled caches are fine.")
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-ff", type=int, default=fc.PANEL_MIN_FF)
    ap.add_argument("--min-half", type=int, default=25)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches (comma-separated pitch-cache parquets) and "
                 "--workdir (outside the repo), or STUFFPLUS_CACHES / "
                 "STUFFPLUS_WORKDIR.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


def load_seasons(paths):
    """Concat pitch caches -> deduped frame with TRUE calendar years.

    fc.load_pitches ROLE-RELABELS year for non-default pairs (a 2025-2026 cache
    stores year=2024 for real 2025), so year is recomputed from Date here. Dedup
    runs across the concatenation because each cache deduped only within itself.
    """
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        print(f"  read {os.path.basename(p)}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    before = len(df)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    print(f"  concat {before:,} -> deduped, in-scope {len(df):,} rows")
    print("  rows by TRUE year: %s" % df["year"].value_counts().sort_index().to_dict())
    if "Level" in df.columns:
        print("  levels present: %s" % df["Level"].value_counts().to_dict())
    return df


def add_game_parity(ff):
    """half = parity of the game's chronological index within pitcher-season.

    GameID begins yyyymmdd, so sorting it orders games in time. Alternating
    whole games keeps the halves balanced while making them independent of
    within-game shared effects.
    """
    keys = ff[["PitcherId", "year", "GameID"]].drop_duplicates()
    keys = keys.sort_values(["PitcherId", "year", "GameID"])
    keys["half"] = keys.groupby(["PitcherId", "year"]).cumcount() % 2
    return ff.merge(keys, on=["PitcherId", "year", "GameID"], how="left")


def build_panel(args):
    """Three-season qualifying-FF frame plus the pitcher-season table.

    xT is fit ONCE on all three seasons pooled, so a given EV/LA outcome maps to
    the same run value in every season -- required for cross-season comparability
    (the spec's frozen reference vintage). The Stuff+ ridge and the location map
    are trained on TRAIN_YEAR only, matching the suite's fixed references.
    """
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
    # A pitcher needs two qualified seasons for cross-season covariance to exist.
    seasons_per = ff.groupby("PitcherId")["year"].nunique()
    ff = ff[ff["PitcherId"].isin(seasons_per[seasons_per >= 2].index)].copy()
    ff = add_game_parity(ff)

    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({
        "mean": g["xT"].mean(),
        "mean_adjT": g["adjT"].mean(),
        "n": g["xT"].size(),
        "stuff": g["ridge_pred"].mean(),
        "loc": g["loc"].mean(),
    }).reset_index().rename(columns={"PitcherId": "pitcher", "year": "season"})

    print(f"\nPANEL: {tab['pitcher'].nunique()} pitchers, {len(tab)} pitcher-seasons "
          f"({args.min_ff}+ FF, 2+ qualified seasons, D1)")
    print("  qualified pitcher-seasons by year: %s"
          % tab["season"].value_counts().sort_index().to_dict())
    print("  median FF per pitcher-season: %.0f" % tab["n"].median())
    return ff, tab


def part_a(ff, args):
    """Within-season game-parity split-half reliability, Spearman-Brown corrected."""
    print("\n" + "=" * 78)
    print("PART A -- WITHIN-SEASON RELIABILITY (game parity, Spearman-Brown)")
    print("=" * 78)
    print("Measures bucket 1 alone: same pitcher, same season, different games.")
    print(f"{'season':<8}{'metric':<14}{'pitchers':>9}{'mean FF':>9}"
          f"{'half r':>9}{'SB full':>9}")
    out = {}
    for col, label in [("xT", "xT"), ("ridge_pred", "Stuff+ anchor")]:
        for year in SEASONS:
            sub = ff[ff["year"] == year]
            piv = sub.groupby(["PitcherId", "half"])[col].agg(["mean", "size"]).unstack("half")
            piv = piv.dropna()
            if piv.empty:
                continue
            piv.columns = ["mA", "mB", "nA", "nB"]
            piv = piv[(piv["nA"] >= args.min_half) & (piv["nB"] >= args.min_half)]
            if len(piv) < 30:
                print(f"{year:<8}{label:<14}{len(piv):>9}  too few pitchers, skipped")
                continue
            r = float(pearsonr(piv["mA"], piv["mB"])[0])
            sb = vc.spearman_brown(r)
            n_mean = float((piv["nA"] + piv["nB"]).mean())
            print(f"{year:<8}{label:<14}{len(piv):>9}{n_mean:>9.0f}{r:>9.3f}{sb:>9.3f}")
            if col == "xT":
                out[year] = {"rho_half": r, "rho_full": sb, "n": len(piv),
                             "mean_ff": n_mean}
    print("\nNote: script 06 splits by PITCH parity, which shares within-game")
    print("effects across both halves and reads higher. Game parity is the")
    print("honest unit for a noise estimate.")
    return out


def residualize(tab, cols):
    """Season-center the criterion, then project out physical predictors.

    Returns (residual table shaped for vc.variance_components, coefficients).
    Predictors are z-scored so coefficients are comparable and sign-readable.

    ORIENTATION GATE: stuff/loc are expected runs (lower = better) and so is the
    criterion, so a correctly oriented predictor gets a POSITIVE coefficient. An
    inverted trait narrative shipped once in this project before review caught
    it; the caller prints these signs.
    """
    t = tab.copy()
    t["c"] = t["mean"] - t.groupby("season")["mean"].transform("mean")
    X = np.column_stack([fc.z(t[c]).values for c in cols] + [np.ones(len(t))])
    beta, *_ = np.linalg.lstsq(X, t["c"].values, rcond=None)
    t["resid"] = t["c"].values - X @ beta
    out = t[["pitcher", "season", "n"]].copy()
    out["mean"] = t["resid"].values
    return out, dict(zip(cols, beta[:len(cols)]))


def noise_scales(ff, value_col, args):
    """Naive vs game-clustered noise scale for a pitcher-season mean.

    The naive pitch-level variance assumes independent pitches. Pitches in one
    game share batter, park, umpire, and day effects, so it understates the
    uncertainty in a season mean; the half-split scale measures that directly.
    Using the naive scale here left Part B disagreeing with Part A by 0.06
    reliability, in exactly this direction.
    """
    key = ff["PitcherId"].astype(str) + "|" + ff["year"].astype(str)
    naive = vc.pooled_within_variance(ff[value_col], key)
    tmp = ff[[value_col, "half"]].copy()
    tmp["ps"] = key
    eff, n_ps = vc.effective_noise_scale(tmp, value_col, "ps",
                                        min_half=args.min_half)
    print(f"\nNOISE SCALE for {value_col} (var of a season mean ~= scale / n):")
    print(f"  naive, pitches independent   {naive:.5f}   (sd {np.sqrt(naive):.3f})")
    print(f"  game-clustered half-split    {eff:.5f}   "
          f"(from {n_ps} pitcher-seasons)")
    print(f"  design effect                {eff / naive:.2f}x   "
          f"<- how much within-game shared variance inflates the real error")
    print("  The clustered scale is used below; the naive one would understate")
    print("  noise and hand the difference to skill.")
    return naive, eff


def part_b(ff, tab, sigma2_w, args):
    """Three-bucket decomposition, then the same fit net of physical Pitching+."""
    print("\n" + "=" * 78)
    print("PART B -- VARIANCE DECOMPOSITION")
    print("=" * 78)

    base = vc.variance_components(tab, sigma2_w)
    print(f"\nBUCKET SHARES of single-season observed variance in mean xT "
          f"(n={base['n_pitchers']} pitchers, {base['n_pitcher_seasons']} pitcher-seasons):")
    print(f"  bucket 1  measurement noise   {base['share_noise']:>7.1%}   "
          f"(var {base['s2_noise']:.6f})")
    print(f"  bucket 2  true drift          {base['share_drift']:>7.1%}   "
          f"(var {base['s2_drift']:.6f})")
    print(f"  stable skill                  {base['share_stable']:>7.1%}   "
          f"(var {base['s2_stable']:.6f})")
    if base["s2_drift"] < 0:
        print("  NOTE: drift estimated slightly negative -- consistent with a true")
        print("        value near zero. Reported raw, not clamped.")
    print("\n  stable-skill covariance by season pair (lag diagnostic):")
    for (s1, s2), (cov, n) in sorted(base["pairs"].items()):
        lag = s2 - s1
        print(f"    {s1}-{s2}  lag {lag}   cov={cov:.6f}   n={n}")
    print("    If lag-2 covariance sits below lag-1, part of what looks stable")
    print("    at one year of separation is slow drift, not permanent talent.")

    print("\nMISSING SKILL -- stable variance surviving physical Pitching+:")
    print("  Results-based predictors are deliberately EXCLUDED: explaining a")
    print("  results-based criterion with a results-based predictor would be")
    print("  circular. Physical traits only (Stuff+ ridge, pooled Location+).")
    print(f"{'model':<28}{'stable var':>12}{'captured':>10}{'MISSING':>10}"
          f"{'of total':>10}")
    print(f"{'none (raw criterion)':<28}{base['s2_stable']:>12.6f}"
          f"{0.0:>10.1%}{1.0:>10.1%}{base['share_stable']:>10.1%}")
    variants = [("Stuff+ only", ["stuff"]),
                ("Location+ only", ["loc"]),
                ("Stuff+ and Location+", ["stuff", "loc"])]
    results = {}
    for label, cols in variants:
        rtab, coefs = residualize(tab, cols)
        out = vc.variance_components(rtab, sigma2_w)
        captured = 1.0 - out["s2_stable"] / base["s2_stable"]
        print(f"{label:<28}{out['s2_stable']:>12.6f}{captured:>10.1%}"
              f"{1 - captured:>10.1%}{out['s2_stable'] / base['total']:>10.1%}")
        results[label] = {"out": out, "captured": captured, "coefs": coefs}

    print("\n  coefficient signs (POSITIVE = correctly oriented; both predictor")
    print("  and criterion are expected runs, lower = better):")
    for label, res in results.items():
        sig = "  ".join(f"{k}={v:+.4f}" for k, v in res["coefs"].items())
        bad = [k for k, v in res["coefs"].items() if v < 0]
        flag = f"   <-- CHECK: {','.join(bad)} negative" if bad else ""
        print(f"    {label:<26}{sig}{flag}")

    full = results["Stuff+ and Location+"]
    print(f"\n  HEADLINE: physical Pitching+ captures {full['captured']:.1%} of stable "
          f"skill;\n            {1 - full['captured']:.1%} of stable skill is MISSING "
          f"(= {full['out']['s2_stable'] / base['total']:.1%} of total\n            "
          f"single-season variance in mean xT).")
    return base, results


def lag_means(pairs):
    """Pitcher-count-weighted stable covariance at one and two years of lag.

    A permanent trait contributes equally at every lag. If the lag-2 covariance
    sits below lag-1, some of what the three-bucket model books as stable is
    really slow decay, and 'stable' has to be read as 'persists to next season'
    rather than 'permanent'.
    """
    num = {1: 0.0, 2: 0.0}
    den = {1: 0.0, 2: 0.0}
    for (s1, s2), (cov, n) in pairs.items():
        lag = s2 - s1
        if lag in num:
            num[lag] += cov * n
            den[lag] += n
    return tuple(num[k] / den[k] if den[k] else float("nan") for k in (1, 2))


def cluster_bootstrap(tab, sigma2_w, reps, seed=17):
    """Resample PITCHERS with replacement; whole careers move together.

    sigma2_w is held fixed: it is estimated from millions of pitches, so its
    sampling error is negligible next to the between-pitcher terms.
    """
    rng = np.random.default_rng(seed)
    pids = tab["pitcher"].unique()
    base = tab.reset_index(drop=True)
    rows_by_p = [base.index[base["pitcher"] == p].to_numpy() for p in pids]
    keys = ["share_noise", "share_drift", "share_stable", "persistence"]
    acc = {k: [] for k in keys}
    acc["captured"] = []
    lag1s, lag2s = [], []
    for _ in range(reps):
        draw = rng.integers(0, len(pids), len(pids))
        take = np.concatenate([rows_by_p[k] for k in draw])
        # Relabel so a pitcher drawn twice acts as two independent pitchers.
        newpid = np.repeat(np.arange(len(draw)),
                           [len(rows_by_p[k]) for k in draw])
        bt = base.take(take).copy()
        bt["pitcher"] = newpid
        try:
            b = vc.variance_components(bt, sigma2_w)
            rtab, _ = residualize(bt, ["stuff", "loc"])
            r = vc.variance_components(rtab, sigma2_w)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for k in keys:
            acc[k].append(b[k])
        acc["captured"].append(1.0 - r["s2_stable"] / b["s2_stable"])
        l1, l2 = lag_means(b["pairs"])
        if not (np.isnan(l1) or np.isnan(l2)) and l1 > 0:
            lag1s.append(l1)
            lag2s.append(l2)
    print("\nCLUSTER BOOTSTRAP over pitchers "
          f"({len(acc['share_stable'])} usable of {reps} reps):")
    for k in keys + ["captured"]:
        fc.boot_report(k, acc[k])

    if lag1s:
        l1 = np.array(lag1s)
        l2 = np.array(lag2s)
        ratio = l2 / l1
        print("\n  IS 'STABLE' PERMANENT OR SLOWLY DECAYING? A permanent trait")
        print("  contributes the same covariance at every lag.")
        print(f"    lag-1 covariance  {l1.mean():.6f}  95% CI="
              f"[{np.percentile(l1, 2.5):.6f},{np.percentile(l1, 97.5):.6f}]")
        print(f"    lag-2 covariance  {l2.mean():.6f}  95% CI="
              f"[{np.percentile(l2, 2.5):.6f},{np.percentile(l2, 97.5):.6f}]")
        print(f"    retention lag2/lag1 = {np.median(ratio):.2f}  95% CI="
              f"[{np.percentile(ratio, 2.5):.2f},{np.percentile(ratio, 97.5):.2f}]"
              f"   P(lag2 < lag1) = {(ratio < 1).mean():.3f}")
    return acc


def consistency(a_out, base):
    """Do Part A and Part B agree? They measure overlapping quantities."""
    print("\n" + "=" * 78)
    print("A vs B CONSISTENCY CHECK")
    print("=" * 78)
    obs = [v["rho_full"] for v in a_out.values()]
    rho_obs = float(np.mean(obs)) if obs else float("nan")
    print(f"{'quantity':<44}{'observed':>10}{'B predicts':>12}")
    print(f"{'within-season reliability (A, mean of seasons)':<44}"
          f"{rho_obs:>10.3f}{base['rho_within_pred']:>12.3f}")
    print("  (A measures stable+drift over stable+drift+noise; B predicts the same)")
    print(f"\n{'persistence = stable / (stable + drift)':<44}"
          f"{'':>10}{base['persistence']:>12.3f}")
    print("  Fraction of what repeats within a season that also survives to the")
    print("  next one. This, not the model, is what caps year-over-year forecasting.")
    gap = abs(rho_obs - base["rho_within_pred"])
    verdict = "AGREE" if gap < 0.05 else "DISAGREE -- reconcile before reporting"
    print(f"\n  gap = {gap:.3f}  ->  {verdict}")


if __name__ == "__main__":
    args = cli()
    ff, tab = build_panel(args)
    a_out = part_a(ff, args)
    naive_w, sigma2_w = noise_scales(ff, "xT", args)
    base, results = part_b(ff, tab, sigma2_w, args)
    cluster_bootstrap(tab, sigma2_w, args.boot)
    consistency(a_out, base)

    print("\n  Same decomposition on the naive pitch-independent scale, for")
    print("  reference only -- it is the version that disagrees with Part A:")
    nb = vc.variance_components(tab, naive_w)
    print(f"    noise {nb['share_noise']:.1%}  drift {nb['share_drift']:.1%}  "
          f"stable {nb['share_stable']:.1%}  "
          f"implied within-season rel {nb['rho_within_pred']:.3f}")

    print("\n" + "=" * 78)
    print("ROBUSTNESS: same decomposition on adjT (opponent-adjusted criterion)")
    print("=" * 78)
    alt = tab.drop(columns=["mean"]).rename(columns={"mean_adjT": "mean"})
    _, s2w_alt = noise_scales(ff, "adjT", args)
    b_alt = vc.variance_components(alt, s2w_alt)
    r_alt, _ = residualize(alt, ["stuff", "loc"])
    o_alt = vc.variance_components(r_alt, s2w_alt)
    print(f"  noise {b_alt['share_noise']:.1%}  drift {b_alt['share_drift']:.1%}  "
          f"stable {b_alt['share_stable']:.1%}  "
          f"captured {1 - o_alt['s2_stable'] / b_alt['s2_stable']:.1%}")
    print("  xT and adjT correlate ~0.985, so these should track the headline")
    print("  closely; a large divergence means something is wrong.")
