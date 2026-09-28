"""Tests for coach_two_season_forecast: the pooled mean, the candidates, and the cross-fit that
must never let a pitcher's own criterion choose their forecast."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coach_two_season_forecast as ts  # noqa: E402


def test_pooled_weights_and_fallbacks():
    x = ts.pooled([1.0, 1.0, 1.0, 1.0], [100, 100, 100, 100],
                  [3.0, 3.0, np.nan, 3.0], [100, 100, 100, 10], 1.0)
    assert x[0] == 2.0
    assert x[2] == 1.0          # no 2024 value
    assert x[3] == 1.0          # 2024 below ABS_MIN
    half = ts.pooled([1.0], [100], [3.0], [100], 0.5)
    assert np.isclose(half[0], (100 + 0.5 * 300) / 150)


def synth(n=3000, seed=1, noise_r=3.0):
    """Stable talent T; results are very noisy, stuff is precise but only partly predictive."""
    rng = np.random.default_rng(seed)
    T = rng.normal(size=n)
    n25 = rng.integers(30, 300, n).astype(float)
    n24 = np.where(rng.random(n) < 0.6, rng.integers(30, 300, n), np.nan)
    tab = pd.DataFrame({
        "R25": T + rng.normal(size=n) * noise_r * np.sqrt(100 / n25),
        "R24": T + rng.normal(size=n) * noise_r * np.sqrt(100 / np.nan_to_num(n24, nan=100)),
        "S25": 0.5 * T + rng.normal(size=n) * 0.9,
        "crit": T + rng.normal(size=n) * 2.0,
        "nR25": n25, "nR24": n24, "nS25": n25, "nS24": n24,
    })
    tab["S24"] = tab["S25"] + rng.normal(size=n) * 0.05
    tab.loc[np.isnan(n24), ["R24", "S24"]] = np.nan
    return tab


def test_pooling_noisy_results_helps_on_synthetic_truth():
    tab = synth()
    crit = tab["crit"].values
    one = ts._r(ts.composite(tab, ["R", "S"], "none", 1.0), crit)
    r1 = ts._r(ts.composite(tab, ["R", "S"], "R", 1.0), crit)
    assert r1 > one + 0.01


def test_crossfit_uses_the_other_half_to_choose():
    tab = synth(seed=4)
    half = np.array([i % 2 for i in range(len(tab))])
    cf, picks = ts.crossfit(tab, ["R", "S"], half)
    crit = tab["crit"].values
    for h in (0, 1):
        other = half != h
        scores = {k: ts._r(ts.composite(tab, ["R", "S"], w, lam)[other], crit[other])
                  for k, (w, lam) in ts.CANDIDATES.items()}
        assert picks[h] == max(scores, key=scores.get)
        own = ts.composite(tab, ["R", "S"], *ts.CANDIDATES[picks[h]])
        assert np.allclose(cf[half == h], own[half == h])


def test_crossfit_choice_is_blind_to_own_half_outcomes():
    tab = synth(seed=7)
    half = np.array([i % 2 for i in range(len(tab))])
    _, picks = ts.crossfit(tab, ["R", "S"], half)
    scrambled = tab.copy()
    rng = np.random.default_rng(0)
    m = half == 0
    scrambled.loc[m, "crit"] = rng.permutation(scrambled.loc[m, "crit"].values)
    _, picks2 = ts.crossfit(scrambled, ["R", "S"], half)
    assert picks2[0] == picks[0]   # half 0's forecast was chosen on half 1 only


def test_halves_are_fixed_and_balanced():
    ids = np.arange(1000, 1400)
    a, b = ts.halves(ids), ts.halves(ids[::-1])
    assert a == b
    assert abs(sum(a.values()) - 200) <= 1
