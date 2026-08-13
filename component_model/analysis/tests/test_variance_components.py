"""Synthetic-recovery tests for the variance-components estimator.

The estimator's job is to split observed pitcher-season variance into stable
skill, true drift, and sampling noise. On real data there is no way to check
that it did so correctly, so it is verified here against simulated panels whose
true components are known by construction.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import variance_components as vc


def simulate_panel(n_pitchers=1200, seasons=(2024, 2025, 2026), s2_stable=4e-4,
                   s2_drift=2e-4, sigma2_w=0.105, n_lo=100, n_hi=400, seed=7):
    """Panel with known components. Returns (pitcher-season table, truth dict).

    y_it = a_i + b_it + e_it, var(e_it) = sigma2_w / n_it. Scales mimic real xT:
    pitch-level sd ~0.32 (var ~0.105), so with ~200 FF the noise term is the
    same order as stable skill -- the regime that makes this question hard.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, np.sqrt(s2_stable), n_pitchers)
    rows = []
    for i in range(n_pitchers):
        for s in seasons:
            n = int(rng.integers(n_lo, n_hi))
            b = rng.normal(0, np.sqrt(s2_drift)) if s2_drift > 0 else 0.0
            e = rng.normal(0, np.sqrt(sigma2_w / n))
            rows.append({"pitcher": i, "season": s, "mean": a[i] + b + e, "n": n})
    truth = {"s2_stable": s2_stable, "s2_drift": s2_drift, "sigma2_w": sigma2_w}
    return pd.DataFrame(rows), truth


def test_pooled_within_variance_recovers_pitch_level_variance():
    rng = np.random.default_rng(3)
    frames = []
    for g in range(400):
        n = int(rng.integers(100, 400))
        frames.append(pd.DataFrame({"v": rng.normal(rng.normal(0, 0.02), 0.32, n),
                                    "g": g}))
    df = pd.concat(frames, ignore_index=True)
    got = vc.pooled_within_variance(df["v"], df["g"])
    assert got == pytest.approx(0.32 ** 2, rel=0.05)


def test_variance_components_recovers_known_truth():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["s2_stable"] == pytest.approx(truth["s2_stable"], rel=0.15)
    assert out["s2_drift"] == pytest.approx(truth["s2_drift"], rel=0.20)
    assert out["s2_noise"] > 0


def test_shares_sum_to_one():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    total = out["share_stable"] + out["share_drift"] + out["share_noise"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_zero_drift_is_detected():
    """With no true season-to-season change, drift must estimate near zero."""
    tab, truth = simulate_panel(s2_drift=0.0, seed=11)
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert abs(out["s2_drift"]) < 0.25 * out["s2_stable"]


def test_all_noise_panel_has_no_stable_skill():
    """Pure sampling noise: stable and drift both estimate near zero."""
    tab, truth = simulate_panel(s2_stable=0.0, s2_drift=0.0, seed=13)
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert abs(out["s2_stable"]) < 0.15 * out["s2_noise"]
    assert out["share_noise"] > 0.8


def test_derived_ratios_match_definitions():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["rho_within_pred"] == pytest.approx(
        (out["s2_stable"] + out["s2_drift"]) / out["total"], abs=1e-9)
    assert out["r_across_pred"] == pytest.approx(out["s2_stable"] / out["total"], abs=1e-9)
    assert out["persistence"] == pytest.approx(
        out["s2_stable"] / (out["s2_stable"] + out["s2_drift"]), abs=1e-9)


def test_pairwise_covariances_reported_per_lag():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert set(out["pairs"]) == {(2024, 2025), (2024, 2026), (2025, 2026)}
    for cov, n in out["pairs"].values():
        assert n > 1000
        assert cov == pytest.approx(truth["s2_stable"], rel=0.30)


def test_two_season_panel_still_identified():
    tab, truth = simulate_panel(seasons=(2024, 2025))
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["s2_stable"] == pytest.approx(truth["s2_stable"], rel=0.20)


def simulate_clustered_pitches(n_groups=1500, games=12, ppg=20, tau2=0.02,
                               sigma2_e=0.09, seed=5):
    """Pitch-level data with a game effect, split into halves by whole game.

    var(group mean) = tau2 / games + sigma2_e / n, so expressed as a per-pitch
    scale, sigma2_eff = tau2 * ppg + sigma2_e. That is the prediction the
    estimator has to reproduce.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for gid in range(n_groups):
        level = rng.normal(0, 0.03)
        for gm in range(games):
            game_eff = rng.normal(0, np.sqrt(tau2))
            vals = rng.normal(level + game_eff, np.sqrt(sigma2_e), ppg)
            rows.append(pd.DataFrame({"v": vals, "ps": gid, "half": gm % 2}))
    return pd.concat(rows, ignore_index=True)


def test_effective_noise_scale_includes_cluster_variance():
    """The half-split scale must capture the game effect, not just pitch scatter."""
    tau2, sigma2_e, ppg = 0.02, 0.09, 20
    df = simulate_clustered_pitches(tau2=tau2, sigma2_e=sigma2_e, ppg=ppg)
    got, n = vc.effective_noise_scale(df, "v", "ps", min_half=25)
    assert n > 1000
    assert got == pytest.approx(tau2 * ppg + sigma2_e, rel=0.10)


def test_effective_noise_scale_exceeds_naive_when_clustered():
    """Design effect > 1: pitch-level variance understates a season mean's error."""
    df = simulate_clustered_pitches(tau2=0.02, sigma2_e=0.09)
    eff, _ = vc.effective_noise_scale(df, "v", "ps", min_half=25)
    naive = vc.pooled_within_variance(df["v"], df["ps"])
    assert eff > 1.5 * naive


def test_effective_noise_scale_matches_naive_without_clustering():
    """With no game effect the two scales agree, so the correction is inert."""
    df = simulate_clustered_pitches(tau2=0.0, sigma2_e=0.09, seed=21)
    eff, _ = vc.effective_noise_scale(df, "v", "ps", min_half=25)
    naive = vc.pooled_within_variance(df["v"], df["ps"])
    assert eff == pytest.approx(naive, rel=0.10)


def test_spearman_brown_doubles_length():
    assert vc.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert vc.spearman_brown(0.0) == pytest.approx(0.0)
    assert vc.spearman_brown(1.0) == pytest.approx(1.0)
