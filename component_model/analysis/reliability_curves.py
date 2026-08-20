"""Cross-metric ceiling estimation and reliability-vs-sample-size curves.

Shared math for script 13 (which metric predicts next-season adjT best, as a
function of pitch count). Factored out of the numbered script, the same way
variance_components.py is factored out of script 12 -- numbered scripts are
CLI entry points, not importable modules, and this math is unit-testable
against synthetic panels with a known truth, which real data can't provide.
"""
import numpy as np
import pandas as pd

import variance_components as vc


def random_game_half(ff, rng, keys=("PitcherId", "year"), game_col="GameID"):
    """Randomly assign each pitcher-season's whole games to half A/B (~evenly).

    Splitting by whole GAME, not pitch, matters: pitches within one game share
    batter, park, umpire, and day effects, so a pitch-level split would leave
    that shared variance in both halves and understate the noise. This differs
    from a deterministic chronological-parity split only in that WHICH games
    land in which half is redrawn every call, so it can be repeated to get a
    distribution over the noise estimate rather than one fixed split.
    """
    keys = list(keys)
    game_keys = ff[keys + [game_col]].drop_duplicates().copy()
    game_keys["rand"] = rng.random(len(game_keys))
    game_keys["half"] = (game_keys.groupby(keys)["rand"]
                          .rank(method="first").astype(int) % 2)
    return ff.merge(game_keys[keys + [game_col, "half"]], on=keys + [game_col], how="left")


def pooled_cross_covariance(tab, x_col, y_col, pitcher_col="pitcher", season_col="season"):
    """Precision-weighted cross-year covariance of x_col(t) with y_col(t'),
    pooled over EVERY season pair (t' > t) sharing pitchers -- lag-1 (year1/
    year2) AND lag-2 (year1/year3), not adjacent pairs only.

    Mirrors vc.variance_components's internal covariance pooling exactly (cov
    per season pair, weighted by that pair's shared-pitcher count, summed
    over ALL pairs with t' > t) rather than concatenating pairs into one array
    and taking a single covariance -- a pitcher present in three seasons
    would otherwise contribute to two overlapping pairs at once, correlating
    rows a plain covariance assumes are independent.

    Using only adjacent (lag-1) pairs here was a real bug, not a stylistic
    choice: it silently diverged from vc.variance_components's all-lag
    pooling (used inside stable_share_of/persistence_of), which pools lag-1
    AND lag-2 -- exactly what the task spec asked for ("year1/year2 and
    year1/year3"). Because stable skill decays somewhat by lag-2 (see
    RESULTS.md's retention discussion), a lag-1-only estimate reads
    systematically HIGHER than the all-lag estimate the rest of this module
    uses, which showed up as a same-metric-predicting-itself ceiling of 1.08
    instead of the required 1.0 on real data, and let optimal_blend score
    below a single metric nested inside it. Caught on a real STUFFPLUS_CACHES
    run; the synthetic tests didn't catch it because they only cross-check
    stable_share_of against vc.variance_components directly (which already
    shares its implementation), not this function's independent pooling
    against vc.variance_components's for the same (x, x) pair.

    Season-centers x_col and y_col internally (matching vc.variance_components,
    which always does) rather than assuming the caller already did -- a
    league-wide year-to-year shift would otherwise leak into the covariance,
    and this function previously only agreed with vc.variance_components when
    the caller happened to pre-center first.

    Built via two independent single-column frames, never a frame with both
    x_col and y_col together -- when x_col == y_col (the same-metric case,
    e.g. checking a metric's ceiling against itself), selecting the same
    column name twice into one frame silently produces two duplicate-named
    columns, and pivot_table on a duplicate-named source then doubles up the
    season columns too, corrupting the pairing downstream. Caught by a
    same-metric test case; two separate frames sidesteps the issue entirely.

    Returns (pooled_cov, total_pitcher_seasons_used).
    """
    def _centered_wide(col):
        s = tab[col] - tab.groupby(season_col)[col].transform("mean")
        return pd.DataFrame({pitcher_col: tab[pitcher_col], season_col: tab[season_col],
                             "v": s}).pivot_table(index=pitcher_col, columns=season_col,
                                                  values="v")

    wide_x = _centered_wide(x_col)
    wide_y = _centered_wide(y_col)
    seasons = sorted(set(wide_x.columns) & set(wide_y.columns))
    pairs = []
    for i, s1 in enumerate(seasons):
        for s2 in seasons[i + 1:]:
            both = pd.DataFrame({"x": wide_x[s1], "y": wide_y[s2]}).dropna()
            if len(both) >= vc.MIN_PAIR_N:
                cov = float(np.cov(both["x"].values, both["y"].values, ddof=1)[0, 1])
                pairs.append((cov, len(both)))
    if not pairs:
        raise ValueError(f"no season pair has enough shared pitchers for "
                          f"({x_col}, {y_col})")
    den = sum(n for _, n in pairs)
    pooled = sum(cov * n for cov, n in pairs) / den
    return pooled, den


def centered_variance(tab, col, season_col="season"):
    """Season-centered variance of col -- the one "observed variance" every
    other function in this module means, and vc.variance_components always
    computes internally. Shared here so cross_metric_ceiling and
    optimal_blend can't independently drift from that convention the way
    signal_variance briefly did (caught via a same-metric consistency test
    that requires machine-precision agreement, not just closeness)."""
    centered = tab[col] - tab.groupby(season_col)[col].transform("mean")
    return float(centered.var(ddof=1))


def stable_share_of(tab, col, noise_per_pitch, n_col="n", pitcher_col="pitcher",
                     season_col="season"):
    """Stable-ONLY share of observed variance for one metric: Var(permanent
    component) / Var(observed).

    This, not (stable+drift)/total, is the correct attenuation-correction
    term for cross_metric_ceiling below. The numerator there (cross-year
    covariance) captures only the permanent component -- drift is a fresh,
    independent draw each season and washes out of a cross-year covariance in
    expectation, even though it's still part of THIS season's observed
    variance. Correcting with (stable+drift)/total therefore under-corrects
    for noise and understates the ceiling by roughly the persistence ratio
    (stable/(stable+drift)) -- caught by a same-metric-predicting-itself
    synthetic test, where the true ceiling is exactly 1.0 by construction.

    Delegates to vc.variance_components for the actual stable/drift/noise
    split (its own cross-season covariance identification, precision-weighted
    across season pairs) and uses only its share_stable.
    """
    vc_tab = (tab[[pitcher_col, season_col, col, n_col]]
              .rename(columns={pitcher_col: "pitcher", season_col: "season",
                               col: "mean", n_col: "n"}))
    return vc.variance_components(vc_tab, noise_per_pitch)["share_stable"]


def persistence_of(tab, col, noise_per_pitch, n_col="n", pitcher_col="pitcher",
                    season_col="season"):
    """persistence_x = s2_stable / (s2_stable + s2_drift) for one metric --
    how much of what repeats within a season also survives to next season,
    i.e. the cap reliability_curve's R(n) hits at n -> infinity within one
    season (X's own drift never averages away, however many pitches you have).
    Thin pass-through to vc.variance_components's own "persistence" key.
    """
    vc_tab = (tab[[pitcher_col, season_col, col, n_col]]
              .rename(columns={pitcher_col: "pitcher", season_col: "season",
                               col: "mean", n_col: "n"}))
    return vc.variance_components(vc_tab, noise_per_pitch)["persistence"]


def cross_metric_ceiling(tab, x_col, y_col, noise_x, noise_y, n_col="n",
                          pitcher_col="pitcher", season_col="season"):
    """Attenuation-corrected asymptotic correlation between metric x_col (this
    season) and y_col (next season) -- the ceiling as pitch count -> infinity.

    Only the permanent component of x_col can covary with next season's
    y_col: noise and drift are independent draws each season and contribute
    ~0 to cov(x_t, y_t+1) in expectation, so the pooled cross-year covariance
    is already a clean estimate of the numerator, cov(a_x, a_y). The observed
    VARIANCES of x_t and y_t+1 include noise AND drift on top of that
    permanent component, both of which bias a naive Pearson correlation
    toward zero -- dividing by sqrt of each metric's stable-ONLY share
    removes that attenuation: r_true = r_observed / sqrt(share_x * share_y).
    """
    pooled_cov, n_pairs = pooled_cross_covariance(tab, x_col, y_col, pitcher_col, season_col)
    var_x = centered_variance(tab, x_col, season_col)
    var_y = centered_variance(tab, y_col, season_col)
    raw_r = pooled_cov / np.sqrt(var_x * var_y)
    share_x = stable_share_of(tab, x_col, noise_x, n_col, pitcher_col, season_col)
    share_y = stable_share_of(tab, y_col, noise_y, n_col, pitcher_col, season_col)
    ceiling = raw_r / np.sqrt(share_x * share_y)
    return {
        "raw_r": raw_r, "ceiling": ceiling,
        "rel_x": share_x, "rel_y": share_y,
        "raw_cov": pooled_cov, "n_pairs": n_pairs,
    }


def signal_variance(tab, col, noise_per_pitch, n_col="n", season_col="season"):
    """Var(true per-pitcher-season value) = observed variance minus the mean
    noise variance -- stable + drift together (drift doesn't vanish with more
    pitches WITHIN one season; only noise does). This is D_x in
    reliability_curve's derivation below, not the stable-only quantity
    cross_metric_ceiling uses.

    Matches vc.variance_components's conventions in the two places that
    caused real, non-negligible disagreement on an actual STUFFPLUS_CACHES
    run (a single-predictor optimal_blend call should exactly reproduce
    reliability_curve's own number for that metric, and didn't, until both
    were fixed):

    1. Season-centers col before computing variance. vc.variance_components
       always centers; if col arrives uncentered, any season-to-season mean
       shift (even pure finite-sample noise in how each season's mean lands)
       inflates this function's total variance relative to
       vc.variance_components's centered total_obs.
    2. Uses mean(1/n), NOT 1/mean(n), for the noise term (sigma2_w *
       (1/n).mean(), matching s2_noise exactly). These differ under Jensen's
       inequality whenever pitch counts vary across pitcher-seasons, which
       they do on real data (~100 to 1000+ FF); 1/mean(n) systematically
       understates the noise term and inflates D.
    """
    mean_inv_n = float((1.0 / tab[n_col]).mean())
    total_var = centered_variance(tab, col, season_col)
    return max(1e-9, total_var - noise_per_pitch * mean_inv_n)


def reliability_curve(n, ceiling, persistence_x, share_y, noise_per_pitch, sig_var):
    """R-squared(n) = ceiling^2 * persistence_x * share_y * n / (n + noise_per_pitch / sig_var).

    R-squared, not r: matches the chart's stated axis ("share of next-season
    adjT variance explained"), and the algebra only closes cleanly in R^2
    terms -- see the derivation below for why.

    Two multiplicative caps sit ABOVE the usual noise-vs-n bracket, and
    dropping either one overstates the curve, sometimes by a large factor:

    - persistence_x = s2_stable_x / (s2_stable_x + s2_drift_x). X's OWN drift
      never averages away no matter how many pitches you have in ONE season
      (drift is a real per-season effect, not sampling error), so R(n) can
      never reach the idealized "ceiling" from cross_metric_ceiling (which
      assumes X's drift is also zero) -- only ceiling^2 * persistence_x, as
      n -> infinity within a single season.
    - share_y = s2_stable_y / var(y), Y's (next-season adjT's) own stable
      share at whatever pitch count Y actually has. Y's own noise and drift
      cap how correlated ANYTHING can be with it, independent of how good X
      is -- this term is easy to drop by mistake since it has nothing to do
      with X, but it isn't optional: skipping it overstated a numeric check
      here by more than 2x.

    Full derivation, with D_x = signal_variance(x) = s2_stable_x + s2_drift_x
    and var_y = Y's observed variance:
      r(n) = cov(a_x, a_y) / sqrt((D_x + noise_x/n) * var_y)
           = r(infinity) * sqrt(D_x / (D_x + noise_x/n))
      r(infinity) = cov(a_x, a_y) / sqrt(D_x * var_y)
                  = ceiling * sqrt(persistence_x) * sqrt(share_y)
      => R^2(n) = r(n)^2 = ceiling^2 * persistence_x * share_y * n/(n + noise_x/D_x)
    Verified numerically against a brute-force simulation that actually varies
    X's pitch count (see tests/test_reliability_curves.py).
    """
    n = np.asarray(n, dtype=float)
    return ceiling ** 2 * persistence_x * share_y * (n / (n + noise_per_pitch / sig_var))


def optimal_blend(tab, metric_cols, target_col, noises, n, pitcher_col="pitcher",
                   season_col="season", n_col="n", signal_vars=None):
    """Precision-weighted (GLS) blend of metric_cols predicting target_col next
    season, at pitch count n. Returns (weights dict, R^2).

    signal_vars: optional {metric_col: D_i} overriding the internally computed
    signal_variance for Sigma's diagonal. Omit it and behaviour is exactly as
    before. It exists because signal_variance is a DIFFERENCE of two estimated
    quantities (observed variance minus the noise term), and for a metric whose
    observed variance is mostly noise -- adjT season means are 77-90% noise at
    real college pitch counts -- that difference can land BELOW the metric's own
    stable component, i.e. an implied negative drift. That is out of bounds: D_i
    is s2_stable + s2_drift and drift cannot be negative, so D_i < s2_stable is
    impossible in truth. Left uncorrected it puts a diagonal smaller than its own
    numerator Cov(a_i, a_y) ~ s2_stable, which inflates that metric's weight at
    high n rather than shrinking it -- the opposite of conservative. Callers that
    hit the boundary should pass max(signal_variance, s2_stable) here and RECORD
    that they clamped (see variance_components.variance_components, which returns
    drift raw precisely so the clamp is a visible caller decision and never a
    silent one).

    Each metric this season is Z_i(n) = a_i + b_i,t + e_i(n). The target next
    season is Y = a_y + b_y,t+1 + e_y (fixed n_y, not varied here). Standard
    GLS result for combining several correlated noisy predictors of a common
    target:
      Sigma_ij(n) = Cov(Z_i(n), Z_j(n))  [3x3, i=j: signal_variance(i) + noise_i/n
                                                i!=j: Cov(a_i, a_j), n-independent]
      c_i         = Cov(Z_i(n), Y) = Cov(a_i, a_y)  [n-independent: only the
                    permanent component survives to next season]
      weights(n)  = Sigma(n)^-1 c
      R^2(n)      = c^T Sigma(n)^-1 c / Var(Y)

    Off-diagonal Sigma terms use ONLY Cov(a_i, a_j) -- the stable-skill
    cross-covariance between two DIFFERENT metrics in the SAME season,
    identified the same way as everywhere else in this module: via cross-YEAR
    covariance, since only the permanent component survives across years,
    regardless of which two metrics are being compared. This assumes
    same-season cross-metric noise and drift are negligible (e.g. Stuff+ and
    Location+ noise both come from the same pitches and could plausibly
    correlate a little) -- the same class of simplifying assumption already
    flagged for reliability_curve; validate empirically if the blend weights
    end up load-bearing.
    """
    k = len(metric_cols)
    Sigma = np.zeros((k, k))
    c = np.zeros(k)
    var_y = centered_variance(tab, target_col, season_col)

    for i, mi in enumerate(metric_cols):
        d_i = (signal_variance(tab, mi, noises[mi], n_col) if signal_vars is None
               else float(signal_vars[mi]))
        Sigma[i, i] = d_i + noises[mi] / n
        cov_iy, _ = pooled_cross_covariance(tab, mi, target_col, pitcher_col, season_col)
        c[i] = cov_iy
        for j in range(i + 1, k):
            mj = metric_cols[j]
            cov_ij, _ = pooled_cross_covariance(tab, mi, mj, pitcher_col, season_col)
            Sigma[i, j] = Sigma[j, i] = cov_ij

    weights_arr = np.linalg.solve(Sigma, c)
    r2 = float(c @ weights_arr / var_y)
    weights = dict(zip(metric_cols, weights_arr.tolist()))
    return weights, r2


def solve_crossover(curve_a, curve_b, lo=1.0, hi=20000.0):
    """First n in [lo, hi] where curve_a(n) == curve_b(n), or None if the two
    curves don't cross there. Reliability curves of this form are monotone
    increasing and concave, so at most one crossover exists in practice; this
    finds the single sign change via bisection, not every root.
    """
    from scipy.optimize import brentq
    f = lambda n: curve_a(n) - curve_b(n)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None
    return float(brentq(f, lo, hi))
