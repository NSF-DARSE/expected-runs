"""Synthetic-recovery tests for change_detectability.py.

Same style as test_variance_components.py and test_reliability_curves.py:
simulate data with a KNOWN per-pitch variance and game-clustering effect,
then check the estimator recovers it. Real data has no observable ground
truth for a within-pitcher per-pitch variance, so this is the only place the
estimator's correctness can be checked directly; the real-bundle numbers in
the script are compared only against a previously measured plausible range
(see the script's docstring), not asserted exact.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import change_detectability as cd


def simulate_clustered_arsenal(n_pitchers=200, games_per_pitcher=16, pitches_per_game=8,
                                tau2=6.0, sigma2_e=100.0, seed=5):
    """Pitch-level 'display grade' data with a per-pitcher level, a per-game
    shared effect (tau2), and pitch-level scatter (sigma2_e), grouped into
    whole games so a game-parity half split can be taken.

    True per-pitch scale (matches variance_components' own convention,
    verified in test_variance_components.py): sigma2_eff = tau2 * ppg + sigma2_e
    where ppg = pitches_per_game. The naive (pitch-independent) variance
    should recover close to sigma2_e alone, understating the true noise by
    the design effect tau2 * ppg / sigma2_e + 1.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_pitchers):
        level = rng.normal(100.0, 8.0)
        for gm in range(games_per_pitcher):
            game_eff = rng.normal(0, np.sqrt(tau2))
            vals = rng.normal(level + game_eff, np.sqrt(sigma2_e), pitches_per_game)
            rows.append(pd.DataFrame({
                "grade": vals,
                "PitcherId": pid,
                "GameID": pid * 1000 + gm,
            }))
    return pd.concat(rows, ignore_index=True)


def test_per_pitch_variance_recovers_known_clustered_truth():
    tau2, sigma2_e, ppg = 6.0, 100.0, 8
    df = simulate_clustered_arsenal(tau2=tau2, sigma2_e=sigma2_e, pitches_per_game=ppg)
    out = cd.per_pitch_variance(df, "grade", ["PitcherId"], min_half=3)
    true_naive = sigma2_e
    true_clustered = tau2 * ppg + sigma2_e
    assert out["naive"] == pytest.approx(true_naive, rel=0.10)
    assert out["clustered"] == pytest.approx(true_clustered, rel=0.15)
    assert out["design_effect"] == pytest.approx(true_clustered / true_naive, rel=0.15)


def test_per_pitch_variance_design_effect_is_one_without_clustering():
    df = simulate_clustered_arsenal(tau2=0.0, sigma2_e=100.0, seed=9)
    out = cd.per_pitch_variance(df, "grade", ["PitcherId"], min_half=3)
    assert out["design_effect"] == pytest.approx(1.0, rel=0.10)


def test_diff_se_matches_direct_simulation_of_two_window_means():
    """The formula, not just its algebra: draw many (recent, prior) window
    pairs from independent normals and check the empirical SD of their
    difference matches diff_se's closed form.
    """
    rng = np.random.default_rng(11)
    sigma2_eff = 144.0  # sd 12, in the previously-measured FF ballpark
    n_recent, n_prior = 20, 35
    reps = 20000
    recent_means = rng.normal(100.0, np.sqrt(sigma2_eff / n_recent), reps)
    prior_means = rng.normal(100.0, np.sqrt(sigma2_eff / n_prior), reps)
    empirical_se = float((recent_means - prior_means).std(ddof=1))
    assert empirical_se == pytest.approx(cd.diff_se(sigma2_eff, n_recent, n_prior), rel=0.03)


def test_diff_se_is_symmetric_in_the_two_windows():
    assert cd.diff_se(100.0, 20, 50) == pytest.approx(cd.diff_se(100.0, 50, 20))


def test_diff_se_shrinks_as_windows_grow():
    se_small = cd.diff_se(100.0, 10, 10)
    se_large = cd.diff_se(100.0, 100, 100)
    assert se_large < se_small
    # SE scales as 1/sqrt(n): 10x the pitches should shrink SE by sqrt(10).
    assert se_small / se_large == pytest.approx(np.sqrt(10), rel=0.02)


def test_diff_se_is_infinite_with_an_empty_window():
    assert cd.diff_se(100.0, 0, 30) == float("inf")


def test_min_detectable_change_scales_with_k():
    se = cd.diff_se(144.0, 20, 20)
    assert cd.min_detectable_change(144.0, 20, 20, 2) == pytest.approx(2 * se)
    assert cd.min_detectable_change(144.0, 20, 20, 1) == pytest.approx(se)


def test_detectability_grid_is_monotone_decreasing_in_n():
    grid = cd.detectability_grid(144.0, [10, 20, 50, 100])
    assert grid["se_diff"].is_monotonic_decreasing
    assert grid["mdc_2se"].is_monotonic_decreasing
    assert np.allclose(grid["mdc_2se"].to_numpy(), 2 * grid["mdc_1se"].to_numpy())


def test_window_counts_and_change_matches_arsenal_recent_change_when_both_qualify():
    """Cross-check against arsenal.recent_change's own windowing on a case
    where both windows are non-empty: the two functions must agree on the
    observed change (recent_change just adds a floor gate on top).
    """
    import arsenal as ar
    outings = pd.DataFrame({
        "date": ["2026-01-20", "2026-03-01"],
        "n": [50, 50],
        "stuff": [100.0, 112.0],
    })
    n_recent, n_prior, change = cd.window_counts_and_change(outings, asof="2026-03-10")
    assert n_recent == 50
    assert n_prior == 50
    assert change == pytest.approx(12.0)
    assert change == pytest.approx(ar.recent_change(outings, floor_n=30, asof="2026-03-10"))


def test_window_counts_and_change_is_none_but_counts_still_reported_when_a_window_is_empty():
    outings = pd.DataFrame({
        "date": ["2026-01-05"],
        "n": [40],
        "stuff": [110.0],
    })
    n_recent, n_prior, change = cd.window_counts_and_change(outings, asof="2026-03-10")
    assert n_recent == 0
    assert n_prior == 0
    assert change is None
