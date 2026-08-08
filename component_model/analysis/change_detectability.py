"""Detectability of a recent-Stuff+ change: the difference-of-means extension
to this repo's reliability protocol.

Every existing reliability estimate in this suite (scripts 06, 12, 13) measures
the noise in a single window's MEAN -- how many pitches before you trust one
number. The pitcher page displays a DIFFERENCE of two window means (trailing
30 days minus the preceding 30 days), whose error variance is the SUM of the
two windows' error variances. That is a different question, and no existing
script answers it; script 06's SAMPLE_FLOOR (season-length Location+) was
reused for it by accident, at the wrong statistic and the wrong window length.

This module supplies the missing piece: a game-clustered, design-effect-
corrected per-pitch variance (via variance_components.effective_noise_scale,
reused verbatim -- see its docstring for why a naive pitch-independent
variance understates the true noise in a window mean), propagated to the
variance of a DIFFERENCE of two such means.

SCALE: everything here operates on whatever column the caller passes in.
Pass a DISPLAY-scale column (arsenal.to_display output) to get a floor in the
units a coach reads; to_display is affine, so this is just a fixed rescale of
whatever the raw expected-run variance would have been.

INDEPENDENCE ASSUMPTION -- Var(recent_mean - prior_mean) = Var(recent_mean) +
Var(prior_mean) requires the two windows' errors to be uncorrelated. They are
non-overlapping calendar windows sharing no games, so there is no shared
SAMPLING noise. But if a pitcher's true level drifts smoothly across the
30/30 boundary (e.g. an in-season velocity build or a fatigue fade), the two
window means pick up a shared trend and become POSITIVELY correlated, which
would make the true Var(diff) somewhat SMALLER than the independence estimate
here. This module does not model drift, so diff_se is a conservative (slightly
too large) floor: it will occasionally call a real change "not yet
distinguishable" but should not manufacture a false positive out of drift.
"""
import numpy as np
import pandas as pd

import variance_components as vc


def add_group_halves(df, group_cols, game_col="GameID"):
    """Assign a game-parity half within each group (whole games, not pitches).

    Generalizes 12_reliability_decomposition.py's add_game_parity to an
    arbitrary group key -- there the group was pitcher x season, here it is
    pitcher x pitch type -- because clustering by whole GAME, not pitch, is
    what keeps shared batter/park/umpire/day effects out of both halves. A
    pitch-level split would leave that shared variance in both halves and
    understate the noise (see variance_components.effective_noise_scale).
    """
    keys = df[group_cols + [game_col]].drop_duplicates()
    keys = keys.sort_values(group_cols + [game_col])
    keys["half"] = keys.groupby(group_cols).cumcount() % 2
    return df.merge(keys, on=group_cols + [game_col], how="left")


def per_pitch_variance(df, value_col, group_cols, min_half=25):
    """Naive vs game-clustered per-pitch variance of value_col within group_cols.

    Returns {naive, clustered, design_effect, n_groups}. design_effect =
    clustered / naive is the factor by which ignoring within-game clustering
    would understate the true noise in a window mean -- report it, per
    variance_components.effective_noise_scale's docstring, so a reader can see
    the honest floor was used and not the optimistic one.
    """
    key = df[group_cols].astype(str).agg("|".join, axis=1)
    naive = vc.pooled_within_variance(df[value_col], key)
    halved = add_group_halves(df.assign(_grp=key.values), group_cols=["_grp"])
    clustered, n_groups = vc.effective_noise_scale(halved, value_col, "_grp", min_half=min_half)
    de = clustered / naive if naive > 0 else float("nan")
    return {"naive": float(naive), "clustered": float(clustered),
            "design_effect": float(de), "n_groups": int(n_groups)}


def diff_se(sigma2_eff, n_recent, n_prior):
    """SE of (recent_mean - prior_mean) for two non-overlapping windows.

    Var(recent_mean) + Var(prior_mean) under the independence assumption
    stated in the module docstring; each window's variance is
    sigma2_eff / n, the same per-pitch-scale convention as
    variance_components.effective_noise_scale.
    """
    if n_recent <= 0 or n_prior <= 0:
        return float("inf")
    return float(np.sqrt(sigma2_eff / n_recent + sigma2_eff / n_prior))


def min_detectable_change(sigma2_eff, n_recent, n_prior, k):
    """Smallest |change| distinguishable from noise at k standard errors."""
    return k * diff_se(sigma2_eff, n_recent, n_prior)


def detectability_grid(sigma2_eff, ns):
    """Symmetric-window detectability curve: SE and MDC at 1/2 SE, per n.

    Symmetric (n_recent == n_prior == n) because that is the common case on
    real data and the simplest curve to read; the real-data table uses
    diff_se directly with each row's actual (possibly asymmetric) n_recent
    and n_prior.
    """
    rows = []
    for n in ns:
        se = diff_se(sigma2_eff, n, n)
        rows.append({"n_per_window": n, "se_diff": se,
                     "mdc_1se": min_detectable_change(sigma2_eff, n, n, 1),
                     "mdc_2se": min_detectable_change(sigma2_eff, n, n, 2)})
    return pd.DataFrame(rows)


def window_counts_and_change(outings, asof, window_days=30):
    """Recent/prior window pitch counts and observed change, from an outing table.

    Mirrors arsenal.recent_change's windowing exactly (same boundary math:
    recent = (asof - window_days, asof], prior = (asof - 2*window_days,
    asof - window_days]) but returns the RAW window pitch counts too, which
    recent_change discards after applying its floor. Needed here to evaluate
    candidate floors and the uncertainty rule against the exact windows the
    live page would compute.

    outings needs columns 'date', 'n', 'stuff' (arsenal.outing_table's shape).
    Returns (n_recent, n_prior, change_or_None_if_either_window_is_empty).
    """
    asof_ts = pd.Timestamp(asof)
    dates = pd.to_datetime(outings["date"])
    recent = outings[(dates > asof_ts - pd.Timedelta(days=window_days)) & (dates <= asof_ts)]
    prior_lo = asof_ts - pd.Timedelta(days=2 * window_days)
    prior = outings[(dates > prior_lo) & (dates <= asof_ts - pd.Timedelta(days=window_days))]
    n_recent = int(recent["n"].sum())
    n_prior = int(prior["n"].sum())
    if n_recent == 0 or n_prior == 0:
        return n_recent, n_prior, None
    recent_mean = float(np.average(recent["stuff"].values, weights=recent["n"].values))
    prior_mean = float(np.average(prior["stuff"].values, weights=prior["n"].values))
    return n_recent, n_prior, recent_mean - prior_mean
