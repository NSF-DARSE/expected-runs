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


def test_spearman_brown_doubles_length():
    assert vc.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert vc.spearman_brown(0.0) == pytest.approx(0.0)
    assert vc.spearman_brown(1.0) == pytest.approx(1.0)
