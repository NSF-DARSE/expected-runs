"""Synthetic-recovery tests for reliability_curves.py.

Same reasoning as test_variance_components.py: the cross-metric ceiling and
the R(n) machinery have no ground truth to check against in real data, so
they're verified here against simulated panels whose true noise, signal, and
ceiling are known by construction.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reliability_curves as rc
import variance_components as vc


def simulate_cross_metric_panel(n_pitchers=1500, seasons=(2024, 2025, 2026),
                                s2_stable_x=3e-4, s2_stable_y=4e-4, rho=0.6,
                                s2_drift=1e-4, noise_x=0.09, noise_y=0.105,
                                n_lo=100, n_hi=400, seed=5):
    """Two metrics, X and Y ('adjT'), sharing a correlated stable component.

    X's permanent skill and Y's permanent skill are drawn jointly with
    correlation rho -- rho IS the true asymptotic ceiling this module has to
    recover. Each metric gets its own independent drift and its own
    independent per-pitch noise, exactly as the identification argument
    assumes (noise and drift are fresh draws each season, contributing ~0 to
    cross-year covariance in expectation).
    """
    rng = np.random.default_rng(seed)
    cov = rho * np.sqrt(s2_stable_x * s2_stable_y)
    stable = rng.multivariate_normal([0, 0], [[s2_stable_x, cov], [cov, s2_stable_y]],
                                     n_pitchers)
    rows = []
    for i in range(n_pitchers):
        ax, ay = stable[i]
        for s in seasons:
            n = int(rng.integers(n_lo, n_hi))
            bx = rng.normal(0, np.sqrt(s2_drift))
            by = rng.normal(0, np.sqrt(s2_drift))
            ex = rng.normal(0, np.sqrt(noise_x / n))
            ey = rng.normal(0, np.sqrt(noise_y / n))
            rows.append({"pitcher": i, "season": s, "n": n,
                        "x": ax + bx + ex, "adjT": ay + by + ey})
    truth = {"ceiling": rho, "noise_x": noise_x, "noise_y": noise_y,
             "s2_stable_x": s2_stable_x, "s2_stable_y": s2_stable_y,
             "s2_drift": s2_drift}
    return pd.DataFrame(rows), truth


def test_cross_metric_ceiling_recovers_known_rho():
    """abs=0.12: the ceiling divides by sqrt(share_x * share_y), so it inherits
    and compounds the estimation noise of two separate stable_share_of calls,
    each individually accurate only to ~15-20% relative per the
    variance_components tolerances -- checked at n_pitchers=4000 across five
    seeds this clusters at 0.57-0.65 around a true 0.6, confirming it's
    unbiased noise, not a systematic error."""
    tab, truth = simulate_cross_metric_panel(n_pitchers=3000)
    out = rc.cross_metric_ceiling(tab, "x", "adjT", truth["noise_x"], truth["noise_y"])
    assert out["ceiling"] == pytest.approx(truth["ceiling"], abs=0.12)


def test_ceiling_exceeds_raw_correlation():
    """The attenuation correction must inflate, not shrink, the raw estimate --
    noise can only bias a correlation toward zero, never away from it."""
    tab, truth = simulate_cross_metric_panel()
    out = rc.cross_metric_ceiling(tab, "x", "adjT", truth["noise_x"], truth["noise_y"])
    assert out["ceiling"] > out["raw_r"]


def test_same_metric_ceiling_is_near_one():
    """A metric predicting ITSELF next season has ceiling ~1: its own stable
    component is perfectly correlated with itself, by construction."""
    tab, truth = simulate_cross_metric_panel(seed=9)
    out = rc.cross_metric_ceiling(tab, "adjT", "adjT", truth["noise_y"], truth["noise_y"])
    assert out["ceiling"] == pytest.approx(1.0, abs=0.1)


def test_zero_shared_stable_component_gives_near_zero_ceiling():
    tab, truth = simulate_cross_metric_panel(rho=0.0, seed=17)
    out = rc.cross_metric_ceiling(tab, "x", "adjT", truth["noise_x"], truth["noise_y"])
    assert abs(out["ceiling"]) < 0.15


def test_stable_share_of_matches_variance_components_directly():
    """stable_share_of is a thin pass-through to vc.variance_components's own
    share_stable -- check it actually agrees, not just that it's plausible."""
    tab, truth = simulate_cross_metric_panel(seed=23)
    got = rc.stable_share_of(tab, "adjT", truth["noise_y"])
    vc_tab = tab.rename(columns={"adjT": "mean"})[["pitcher", "season", "mean", "n"]]
    expected = vc.variance_components(vc_tab, truth["noise_y"])["share_stable"]
    assert got == pytest.approx(expected, rel=1e-9)


def test_stable_share_of_is_bounded():
    tab, truth = simulate_cross_metric_panel(seed=31)
    rel = rc.stable_share_of(tab, "adjT", truth["noise_y"])
    assert 0.0 < rel <= 1.0


def test_stable_share_of_recovers_known_share():
    tab, truth = simulate_cross_metric_panel(seed=33)
    got = rc.stable_share_of(tab, "adjT", truth["noise_y"])
    total_var = tab["adjT"].var(ddof=1)
    expected = truth["s2_stable_y"] / total_var
    assert got == pytest.approx(expected, rel=0.25)


def test_pooled_cross_covariance_matches_true_stable_covariance():
    tab, truth = simulate_cross_metric_panel(seed=41)
    cov, n_pairs = rc.pooled_cross_covariance(tab, "x", "adjT")
    true_cov = truth["ceiling"] * np.sqrt(truth["s2_stable_x"] * truth["s2_stable_y"])
    assert cov == pytest.approx(true_cov, rel=0.25)
    assert n_pairs > 2000


# ---------------- reliability_curve / solve_crossover ----------------

def test_reliability_curve_approaches_capped_plateau_at_large_n():
    """The n -> infinity plateau is ceiling^2 * persistence_x * share_y, NOT
    ceiling^2 alone -- X's own drift and Y's own noise both cap it below the
    idealized ceiling, however many pitches X has."""
    got = rc.reliability_curve(1e9, ceiling=0.4, persistence_x=0.7, share_y=0.5,
                               noise_per_pitch=0.1, sig_var=0.0005)
    assert got == pytest.approx(0.4 ** 2 * 0.7 * 0.5, rel=1e-3)


def test_reliability_curve_is_zero_at_zero_pitches():
    got = rc.reliability_curve(0, ceiling=0.4, persistence_x=0.7, share_y=0.5,
                               noise_per_pitch=0.1, sig_var=0.0005)
    assert got == pytest.approx(0.0)


def test_reliability_curve_half_plateau_point():
    """At n = noise_per_pitch / sig_var, the bracket is exactly n/(n+n) = 0.5,
    so R^2(n) is exactly half the plateau value."""
    noise_per_pitch, sig_var = 0.1, 0.0005
    half_n = noise_per_pitch / sig_var
    plateau = 0.4 ** 2 * 0.7 * 0.5
    got = rc.reliability_curve(half_n, ceiling=0.4, persistence_x=0.7, share_y=0.5,
                               noise_per_pitch=noise_per_pitch, sig_var=sig_var)
    assert got == pytest.approx(plateau * 0.5, rel=1e-6)


def test_reliability_curve_is_monotone_increasing():
    ns = np.array([10, 50, 200, 1000, 5000])
    vals = rc.reliability_curve(ns, ceiling=0.5, persistence_x=0.6, share_y=0.5,
                                noise_per_pitch=0.09, sig_var=0.0004)
    assert np.all(np.diff(vals) > 0)


def test_reliability_curve_matches_brute_force_simulation():
    """Regression test for the derivation itself: simulate a panel where X's
    OWN pitch count is actually varied and Y (next-season adjT) is held at a
    realistic, finite pitch count, then check the closed-form R^2(n) matches
    the empirical squared correlation at several pitch counts. This is the
    check that caught the original formula dropping the share_y term (it
    otherwise overstated R^2 by 2x+ at large n).
    """
    rng = np.random.default_rng(0)
    s2_stable_x, s2_drift_x = 3e-4, 2e-4
    s2_stable_y, s2_drift_y = 4e-4, 1e-4
    rho, noise_x, noise_y, n_y, n_pitchers = 0.6, 0.09, 0.105, 250, 20000

    cov = rho * np.sqrt(s2_stable_x * s2_stable_y)
    stable = rng.multivariate_normal([0, 0], [[s2_stable_x, cov], [cov, s2_stable_y]],
                                     n_pitchers)
    ax, ay = stable[:, 0], stable[:, 1]
    Y = (ay + rng.normal(0, np.sqrt(s2_drift_y), n_pitchers)
         + rng.normal(0, np.sqrt(noise_y / n_y), n_pitchers))
    var_y = s2_stable_y + s2_drift_y + noise_y / n_y

    D_x = s2_stable_x + s2_drift_x
    persistence_x = s2_stable_x / D_x
    share_y = s2_stable_y / var_y

    for n in [50, 500, 5000]:
        X = (ax + rng.normal(0, np.sqrt(s2_drift_x), n_pitchers)
             + rng.normal(0, np.sqrt(noise_x / n), n_pitchers))
        r2_empirical = np.corrcoef(X, Y)[0, 1] ** 2
        r2_formula = rc.reliability_curve(n, ceiling=rho, persistence_x=persistence_x,
                                          share_y=share_y, noise_per_pitch=noise_x,
                                          sig_var=D_x)
        assert r2_formula == pytest.approx(r2_empirical, abs=0.02)


def test_solve_crossover_finds_known_analytic_crossing():
    """Two curves with the same ceiling but different noise/pitch never cross
    except in the trivial limit; use different ceilings so a real interior
    crossover exists, and check the solver lands where the two curves are
    numerically equal (not against a closed-form solution, since the
    crossover of two saturating curves has no clean algebraic form)."""
    # persistence_x=1.0, share_y=1.0: neutral values so the crossover-solving
    # logic itself is what's under test, not the persistence/share caps.
    curve_a = lambda n: rc.reliability_curve(n, ceiling=0.30, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.05, sig_var=0.001)
    curve_b = lambda n: rc.reliability_curve(n, ceiling=0.45, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.20, sig_var=0.001)
    xo = rc.solve_crossover(curve_a, curve_b, lo=1, hi=20000)
    assert xo is not None
    assert curve_a(xo) == pytest.approx(curve_b(xo), abs=1e-6)


def test_solve_crossover_returns_none_when_no_crossing_in_range():
    """Identical curves scaled apart by a constant factor never cross."""
    curve_a = lambda n: rc.reliability_curve(n, ceiling=0.50, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.05, sig_var=0.001)
    curve_b = lambda n: rc.reliability_curve(n, ceiling=0.20, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.05, sig_var=0.001)
    xo = rc.solve_crossover(curve_a, curve_b, lo=1, hi=20000)
    assert xo is None


def test_solve_crossover_higher_ceiling_wins_at_large_n():
    """Sanity check on curve ordering: whichever curve has the higher ceiling
    must be on top well past the crossover point."""
    curve_lo_noise = lambda n: rc.reliability_curve(n, ceiling=0.25, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.02, sig_var=0.001)
    curve_hi_ceiling = lambda n: rc.reliability_curve(n, ceiling=0.45, persistence_x=1.0, share_y=1.0, noise_per_pitch=0.30, sig_var=0.001)
    xo = rc.solve_crossover(curve_lo_noise, curve_hi_ceiling, lo=1, hi=20000)
    assert xo is not None
    assert curve_hi_ceiling(20000) > curve_lo_noise(20000)
    assert curve_lo_noise(1) > curve_hi_ceiling(1)


# ---------------- optimal_blend ----------------

def simulate_three_metric_panel(n_pitchers=1500, seasons=(2024, 2025, 2026),
                                stable_cov=None, s2_drift=1e-4,
                                noises=(0.05, 0.09, 0.105), n_lo=150, n_hi=350,
                                seed=7):
    """adjT, Stuff+ ('stuff'), Location+ ('loc') sharing a known 3x3 stable
    covariance matrix. Each gets its own independent drift and per-pitch
    noise. Returns (tab, truth) where truth carries the raw 3x3 stable
    covariance matrix and per-metric D_i (stable+drift) and noise, everything
    optimal_blend needs to be checked against a hand-computed answer.
    """
    if stable_cov is None:
        # order: stuff, loc, adjT -- stuff/loc lightly correlated (mirrors the
        # project's own "nearly orthogonal" finding), both correlated with adjT.
        s_stuff, s_loc, s_adjt = 2e-4, 3e-4, 4e-4
        stable_cov = np.array([
            [s_stuff, 0.15 * np.sqrt(s_stuff * s_loc), 0.5 * np.sqrt(s_stuff * s_adjt)],
            [0.15 * np.sqrt(s_stuff * s_loc), s_loc, 0.6 * np.sqrt(s_loc * s_adjt)],
            [0.5 * np.sqrt(s_stuff * s_adjt), 0.6 * np.sqrt(s_loc * s_adjt), s_adjt],
        ])
    cols = ["stuff", "loc", "adjT"]
    rng = np.random.default_rng(seed)
    stable = rng.multivariate_normal([0, 0, 0], stable_cov, n_pitchers)
    rows = []
    for i in range(n_pitchers):
        for s in seasons:
            n = int(rng.integers(n_lo, n_hi))
            row = {"pitcher": i, "season": s, "n": n}
            for k, col in enumerate(cols):
                b = rng.normal(0, np.sqrt(s2_drift))
                e = rng.normal(0, np.sqrt(noises[k] / n))
                row[col] = stable[i, k] + b + e
            rows.append(row)
    truth = {"cols": cols, "stable_cov": stable_cov, "s2_drift": s2_drift,
             "noises": dict(zip(cols, noises)),
             "D": {col: stable_cov[k, k] + s2_drift for k, col in enumerate(cols)}}
    return pd.DataFrame(rows), truth


def test_single_predictor_blend_matches_reliability_curve_exactly():
    """optimal_blend with ONE metric predicting itself must reproduce
    reliability_curve's own number for that metric, essentially exactly (both
    describe the identical quantity via two independently-written code
    paths). This is a real regression test, not a synthetic-recovery check:
    it caught two genuine bugs on a real STUFFPLUS_CACHES run that no
    synthetic test surfaced --
      1. pooled_cross_covariance pooled only ADJACENT season pairs, while
         persistence_of/stable_share_of (via vc.variance_components) pool
         ALL pairs (lag-1 and lag-2). Silently diverging estimates of the
         "same" cross-year covariance made the same-metric ceiling read 1.08
         instead of the required 1.0.
      2. signal_variance used noise_per_pitch/mean(n) while
         vc.variance_components uses noise_per_pitch*mean(1/n) -- these
         differ under Jensen's inequality whenever pitch counts vary across
         pitcher-seasons (they do, on real data), so the two "same" D_x
         quantities disagreed even after fixing (1).
    Neither showed up in any prior synthetic test because the synthetic
    panels either had near-uniform pitch counts (masking bug 2) or were
    checked against vc.variance_components directly rather than against this
    module's OWN independently-implemented covariance pooling (masking bug
    1). Uses simulate_three_metric_panel with deliberately wide, skewed n
    (n_lo=80, n_hi=900) specifically so bug 2 would resurface if reintroduced.
    """
    tab, truth = simulate_three_metric_panel(n_lo=80, n_hi=900, seed=51)
    noise_adjt = truth["noises"]["adjT"]
    for n in [30, 300, 5000]:
        _, r2_blend = rc.optimal_blend(tab, ["adjT"], "adjT", {"adjT": noise_adjt}, n)
        ceil = rc.cross_metric_ceiling(tab, "adjT", "adjT", noise_adjt, noise_adjt)["ceiling"]
        persistence_x = rc.persistence_of(tab, "adjT", noise_adjt)
        share_y = rc.stable_share_of(tab, "adjT", noise_adjt)
        sig_var = rc.signal_variance(tab, "adjT", noise_adjt)
        r2_curve = rc.reliability_curve(n, ceil, persistence_x, share_y, noise_adjt, sig_var)
        assert r2_blend == pytest.approx(r2_curve, abs=1e-9)
    assert ceil == pytest.approx(1.0, abs=1e-9)


def test_optimal_blend_matches_hand_computed_gls_at_known_parameters():
    """Validates the closed-form formula itself: build Sigma(n)/c directly
    from TRUE (not estimated) parameters, solve by hand, and check
    rc.optimal_blend's weights/R^2 match -- isolates the formula from
    estimation noise, the same separation used for the R(n) brute-force test.
    """
    tab, truth = simulate_three_metric_panel(n_pitchers=6000, seed=11)
    metric_cols = ["stuff", "loc"]
    n = 200
    stable_cov, cols = truth["stable_cov"], truth["cols"]
    idx = {c: i for i, c in enumerate(cols)}

    k = len(metric_cols)
    Sigma_true = np.zeros((k, k))
    c_true = np.zeros(k)
    for a, ma in enumerate(metric_cols):
        Sigma_true[a, a] = truth["D"][ma] + truth["noises"][ma] / n
        c_true[a] = stable_cov[idx[ma], idx["adjT"]]
        for b in range(a + 1, k):
            mb = metric_cols[b]
            Sigma_true[a, b] = Sigma_true[b, a] = stable_cov[idx[ma], idx[mb]]
    # Full OBSERVED variance (stable + drift + noise at the panel's typical n),
    # matching what rc.optimal_blend actually divides by -- var(tab["adjT"]).
    # A noise-free D_y here would systematically inflate r2_true relative to
    # what optimal_blend computes, not just add estimation noise.
    var_y_true = (stable_cov[idx["adjT"], idx["adjT"]] + truth["s2_drift"]
                  + truth["noises"]["adjT"] / tab["n"].mean())
    w_true = np.linalg.solve(Sigma_true, c_true)
    r2_true = float(c_true @ w_true / var_y_true)

    weights, r2 = rc.optimal_blend(tab, metric_cols, "adjT",
                                   {m: truth["noises"][m] for m in metric_cols}, n)
    # abs, not rel: weights are small covariance ratios (~0.2-0.25), each
    # built from an independently-estimated covariance and variance, so the
    # compounded estimation noise is naturally larger than a single
    # covariance's own ~15-20% relative error (same reasoning as the widened
    # tolerance on test_cross_metric_ceiling_recovers_known_rho above).
    assert weights["stuff"] == pytest.approx(w_true[0], abs=0.05)
    assert weights["loc"] == pytest.approx(w_true[1], abs=0.05)
    assert r2 == pytest.approx(r2_true, abs=0.03)


def test_optimal_blend_beats_or_matches_every_single_metric():
    """A GLS blend can always fall back to weighting only the best single
    predictor, so it must never do WORSE than the best of the individual
    metrics at the same n."""
    tab, truth = simulate_three_metric_panel(seed=13)
    metric_cols = ["stuff", "loc"]
    noises = {m: truth["noises"][m] for m in metric_cols}
    n = 200
    _, r2_blend = rc.optimal_blend(tab, metric_cols, "adjT", noises, n)

    for m in metric_cols:
        ceil = cross_metric_ceiling_helper(tab, m, "adjT", noises[m], truth["noises"]["adjT"])
        persistence_x = rc.persistence_of(tab, m, noises[m])
        share_y = rc.stable_share_of(tab, "adjT", truth["noises"]["adjT"])
        sig_var = rc.signal_variance(tab, m, noises[m])
        r2_single = rc.reliability_curve(n, ceil, persistence_x, share_y, noises[m], sig_var)
        assert r2_blend >= r2_single - 0.03  # small slack for estimation noise


def cross_metric_ceiling_helper(tab, x_col, y_col, noise_x, noise_y):
    return rc.cross_metric_ceiling(tab, x_col, y_col, noise_x, noise_y)["ceiling"]


def test_optimal_blend_r2_increases_with_n():
    tab, truth = simulate_three_metric_panel(seed=17)
    metric_cols = ["stuff", "loc"]
    noises = {m: truth["noises"][m] for m in metric_cols}
    r2s = [rc.optimal_blend(tab, metric_cols, "adjT", noises, n)[1]
           for n in [20, 100, 500, 5000]]
    assert all(b >= a - 1e-9 for a, b in zip(r2s, r2s[1:]))


def test_optimal_blend_weights_shift_toward_lower_noise_metric_at_small_n():
    """stuff has less noise than loc in the default simulation; at very small
    n that advantage should dominate the weight split more than at large n,
    where the metrics' relative ceilings matter more than their noise."""
    tab, truth = simulate_three_metric_panel(seed=19)
    metric_cols = ["stuff", "loc"]
    noises = {m: truth["noises"][m] for m in metric_cols}
    w_small, _ = rc.optimal_blend(tab, metric_cols, "adjT", noises, 20)
    w_large, _ = rc.optimal_blend(tab, metric_cols, "adjT", noises, 5000)
    ratio_small = w_small["stuff"] / (abs(w_small["stuff"]) + abs(w_small["loc"]))
    ratio_large = w_large["stuff"] / (abs(w_large["stuff"]) + abs(w_large["loc"]))
    assert ratio_small > ratio_large


# ---------------- random_game_half ----------------

def _toy_ff(n_pitcher_seasons=40, games_per=10, seed=2):
    rng = np.random.default_rng(seed)
    rows = []
    for ps in range(n_pitcher_seasons):
        pitcher, year = ps // 2, 2024 + (ps % 2)
        for g in range(games_per):
            for _ in range(rng.integers(3, 8)):
                rows.append({"PitcherId": pitcher, "year": year,
                            "GameID": f"{pitcher}_{year}_{g}",
                            "v": rng.normal()})
    return pd.DataFrame(rows)


def test_random_game_half_keeps_whole_games_together():
    ff = _toy_ff()
    rng = np.random.default_rng(1)
    split = rc.random_game_half(ff, rng)
    per_game_halves = split.groupby("GameID")["half"].nunique()
    assert (per_game_halves == 1).all()


def test_random_game_half_is_balanced_per_pitcher_season():
    ff = _toy_ff(games_per=20)
    rng = np.random.default_rng(1)
    split = rc.random_game_half(ff, rng)
    games_per_half = (split[["PitcherId", "year", "GameID", "half"]]
                      .drop_duplicates()
                      .groupby(["PitcherId", "year", "half"]).size())
    by_ps = games_per_half.groupby(level=[0, 1])
    for _, counts in by_ps:
        assert abs(counts.iloc[0] - counts.iloc[-1]) <= 1


def test_random_game_half_varies_across_calls():
    """Different rng draws should produce different assignments -- this is
    the whole point of replacing script 12's fixed parity split."""
    ff = _toy_ff(games_per=20, seed=3)
    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(200)
    split_a = rc.random_game_half(ff, rng_a)
    split_b = rc.random_game_half(ff, rng_b)
    games_a = split_a[["GameID", "half"]].drop_duplicates().set_index("GameID")["half"]
    games_b = split_b[["GameID", "half"]].drop_duplicates().set_index("GameID")["half"]
    assert not games_a.equals(games_b)
