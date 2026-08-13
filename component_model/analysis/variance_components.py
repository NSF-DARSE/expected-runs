"""Method-of-moments decomposition of pitcher-season variance.

Splits the observed spread in a pitcher-season mean into three buckets:

    y_it = mu_t + a_i + b_it + e_it

    a_i    stable skill             var = s2_stable   (persists across seasons)
    b_it   true year-to-year change var = s2_drift    (real, but unknowable in advance)
    e_it   sampling noise           var = sigma2_w / n_it

Identification: a_i is the only term shared across a pitcher's seasons, so the
cross-season covariance of centered season means estimates s2_stable directly.
sigma2_w comes from pitch-level scatter within a pitcher-season, so the noise
term is known rather than assumed. s2_drift is what is left over.

Closed-form in numpy on purpose: a mixed-model fit would need statsmodels (not a
project dependency) and would hide the one step worth seeing, which is that
stable skill IS the cross-season covariance.

ORIENTATION: this module operates on variances and covariances, which are
sign-free, so the lower-is-better run-value convention does not apply to
anything returned here. Callers regressing predictors against the criterion must
check coefficient signs themselves (see fair_criterion.py docstring).
"""
import numpy as np
import pandas as pd

MIN_PAIR_N = 30


def pooled_within_variance(values, groups):
    """sigma2_w: pooled within-group variance of pitch-level values.

    Pooled across pitcher-seasons rather than averaged, so that groups with more
    pitches carry more weight in the estimate.
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float),
                       "g": np.asarray(groups)}).dropna()
    grp = df.groupby("g")["v"]
    means = grp.transform("mean")
    ss = float(((df["v"] - means) ** 2).sum())
    dof = float(len(df) - grp.ngroups)
    if dof <= 0:
        return float("nan")
    return ss / dof


def spearman_brown(r):
    """Reliability of a full-length measure from its split-half correlation."""
    return 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else float("nan")


def effective_noise_scale(df, value_col, group_col, half_col="half", min_half=25):
    """Sampling-noise scale for a group mean, estimated from independent halves.

    Returns (sigma2_eff, n_groups) where var(group mean) ~= sigma2_eff / n.

    Why not just use pooled_within_variance: that treats pitches as independent.
    They are not. Pitches in one game share batter, park, umpire, and day
    effects, so the real uncertainty in a season mean exceeds sigma2_w / n. Here
    the two halves share no games, so their difference carries the clustering
    too, and nothing has to be assumed about it. sigma2_eff / sigma2_w is the
    design effect: the factor by which a pitch-level variance understates the
    uncertainty in a season mean.

    Algebra: halves of equal size with independent noise give var(mA - mB) = 2v,
    where v is each half-mean's noise variance; the full mean averages the two
    halves, so its noise variance is v/2 = var(mA - mB) / 4. Multiplying by n
    expresses that as a per-pitch scale comparable to a pitch-level variance.

    Requires a half assignment that splits by CLUSTER (whole games), not by
    pitch; a pitch-level split leaves the shared variance in both halves and
    returns the naive scale.
    """
    g = df.groupby([group_col, half_col])[value_col].agg(["mean", "size"]).unstack(half_col)
    g = g.dropna()
    if g.empty:
        return float("nan"), 0
    g.columns = ["mA", "mB", "nA", "nB"]
    g = g[(g["nA"] >= min_half) & (g["nB"] >= min_half)]
    if g.empty:
        return float("nan"), 0
    n = g["nA"] + g["nB"]
    est = n * (g["mA"] - g["mB"]) ** 2 / 4.0
    return float(est.mean()), int(len(g))


def variance_components(tab, sigma2_w, value_col="mean"):
    """Split pitcher-season variance into stable / drift / noise.

    tab: one row per pitcher-season, columns 'pitcher', 'season', value_col, 'n'.
    sigma2_w: pooled within-pitcher-season pitch-level variance.

    Season means are removed first, so environment-level shifts (feed growth,
    level-mix drift, scoring environment) never land in any bucket.

    s2_drift is returned RAW and may come out slightly negative when the true
    value is near zero; that is honest sampling error in the estimator, not a
    bug. Do not clamp it silently -- callers print it as-is and note it.
    """
    t = tab[["pitcher", "season", value_col, "n"]].dropna().copy()
    t["c"] = t[value_col] - t.groupby("season")[value_col].transform("mean")

    wide = t.pivot_table(index="pitcher", columns="season", values="c")
    seasons = sorted(wide.columns)
    pairs = {}
    for i, s1 in enumerate(seasons):
        for s2 in seasons[i + 1:]:
            both = wide[[s1, s2]].dropna()
            if len(both) < MIN_PAIR_N:
                continue
            cov = float(np.cov(both[s1].values, both[s2].values, ddof=1)[0, 1])
            pairs[(s1, s2)] = (cov, int(len(both)))
    if not pairs:
        raise ValueError("no season pair has enough shared pitchers to identify "
                         "stable skill; need %d+ pitchers in two seasons" % MIN_PAIR_N)

    # Precision-weight the pairwise covariances by the number of shared pitchers.
    den = sum(n for _, n in pairs.values())
    s2_stable = sum(cov * n for cov, n in pairs.values()) / den

    s2_noise = float(sigma2_w * (1.0 / t["n"]).mean())
    total_obs = float(t["c"].var(ddof=1))
    s2_drift = total_obs - s2_stable - s2_noise

    total = s2_stable + s2_drift + s2_noise
    return {
        "s2_stable": s2_stable,
        "s2_drift": s2_drift,
        "s2_noise": s2_noise,
        "total": total,
        "total_observed": total_obs,
        "share_stable": s2_stable / total,
        "share_drift": s2_drift / total,
        "share_noise": s2_noise / total,
        "rho_within_pred": (s2_stable + s2_drift) / total,
        "r_across_pred": s2_stable / total,
        "persistence": s2_stable / (s2_stable + s2_drift),
        "pairs": pairs,
        "n_pitcher_seasons": int(len(t)),
        "n_pitchers": int(t["pitcher"].nunique()),
    }
