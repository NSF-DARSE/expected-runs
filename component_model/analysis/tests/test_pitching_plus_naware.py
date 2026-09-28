"""Tests for coach_pitching_plus_naware: the reliability-relative weight, the promise that no
Pitching+ at or above the floor moves, the power formula, and the fixed-weight identity the
readability curve rests on."""
import os
import sys

import numpy as np
import pandas as pd

for _k in ("STUFFPLUS_DATA", "STUFFPLUS_DATA_CRIT", "STUFFPLUS_WORKDIR", "STUFFPLUS_WORKDIR_CRIT"):
    os.environ.setdefault(_k, "placeholder")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import coach_pitching_plus_naware as nw  # noqa: E402


def test_k_is_one_at_and_above_the_floor_and_rises_below_it():
    n = np.array([10, 50, 100, 199, 200, 400, 2000])
    k = nw.k_factor(n, 300.0)
    assert np.all(k[4:] == 1.0)
    assert np.all(np.diff(k[:5]) > 0)
    assert np.isclose(k[1], (50 / 350) / (200 / 500))
    assert nw.k_factor(np.array([15]), 2.0)[0] > 0.88   # a Stuff+-like n0 barely moves


def test_naware_equals_shipped_for_every_pitcher_at_or_above_the_floor():
    rng = np.random.default_rng(0)
    n = rng.integers(15, 800, 500)
    zs = {c: rng.normal(size=500) for c in ("ridge", "loc", "adj")}
    n0s = {"ridge": 3.0, "loc": 96.0, "adj": 400.0}
    ship, aw = nw.blend(zs, n, n0s, False), nw.blend(zs, n, n0s, True)
    hi = n >= nw.N_REF
    assert np.allclose(ship[hi], aw[hi])
    assert not np.allclose(ship[~hi], aw[~hi])
    assert np.allclose(ship, (zs["ridge"] + zs["loc"] + zs["adj"]) / 3)


def test_dependent_correlation_se_matches_simulation():
    rng = np.random.default_rng(1)
    n, r, r23, reps = 400, 0.25, 0.9, 3000
    # criterion y and two predictors with corr(y, x) = r each and corr(x1, x2) = r23
    cov = np.array([[1, r, r], [r, 1, r23], [r, r23, 1]])
    L = np.linalg.cholesky(cov)
    d = []
    for _ in range(reps):
        x = rng.normal(size=(n, 3)) @ L.T
        d.append(np.corrcoef(x[:, 0], x[:, 1])[0, 1] - np.corrcoef(x[:, 0], x[:, 2])[0, 1])
    emp = float(np.std(d, ddof=1))
    assert abs(nw.dep_corr_se(r, r23, n) - emp) / emp < 0.1


def test_blend_of_means_is_the_mean_of_the_per_pitch_composite():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"PitcherId": rng.integers(0, 30, 3000),
                       "a": rng.normal(size=3000), "b": rng.normal(size=3000)})
    w = (0.2, 0.5)
    g = df.groupby("PitcherId")
    lhs = w[0] * g["a"].mean() + w[1] * g["b"].mean()
    rhs = (w[0] * df["a"] + w[1] * df["b"]).groupby(df["PitcherId"]).mean()
    assert np.allclose(lhs.values, rhs.values)


def test_naware_helps_on_a_synthetic_truth_where_results_are_noisy_at_small_n():
    rng = np.random.default_rng(3)
    m = 6000
    T = rng.normal(size=m)
    n = rng.integers(15, 400, m).astype(float)
    zs = {"ridge": 0.5 * T + rng.normal(size=m) * 0.9,
          "loc": 0.5 * T + rng.normal(size=m) * np.sqrt(0.75 + 96 / n),
          "adj": T + rng.normal(size=m) * np.sqrt(400 / n)}
    crit = T + rng.normal(size=m) * 2.0
    n0s = {"ridge": 1.0, "loc": 96.0, "adj": 400.0}
    lo = n < nw.N_REF
    ship, aw = nw.blend(zs, n, n0s, False), nw.blend(zs, n, n0s, True)
    assert nw._r(aw[lo], crit[lo]) > nw._r(ship[lo], crit[lo]) + 0.005


def test_floor_is_the_first_grid_point_meeting_the_target():
    assert nw.floor_for(lambda n: n / (n + 50.0), 0.5) == 50
    assert nw.floor_for(lambda n: 0.1, 0.5) is None
