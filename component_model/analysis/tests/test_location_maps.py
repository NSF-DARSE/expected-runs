"""Tests for location_maps: the weighted map must equal the shipped map where they overlap,
the batter frame must mirror the right way, the count-relative value must carry no count
occupancy, and the shrinkage estimators must recover known truth on simulated pitchers.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fair_criterion as fc
import location_maps as lm


def synth_pitches(n=40000, seed=11):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "PlateLocSide": rng.normal(0, 0.9, n),
        "PlateLocHeight": rng.normal(2.3, 0.9, n),
        "Balls": rng.integers(0, 4, n),
        "Strikes": rng.integers(0, 3, n),
        "BatterSide": rng.choice(["Left", "Right"], n),
        "PitcherThrows": rng.choice(["Left", "Right"], n, p=[0.3, 0.7]),
        "PitcherId": rng.integers(0, 400, n),
    })
    # Value: worse near the heart, better low.
    df["xT"] = (0.05 * np.exp(-(df["PlateLocSide"] ** 2 + (df["PlateLocHeight"] - 2.5) ** 2))
                - 0.01 * (df["PlateLocHeight"] < 1.5) + rng.normal(0, 0.2, n))
    return df


def test_catcher_frame_reproduces_pooled_location_map():
    df = synth_pitches()
    fr = lm.frame_columns(df, "catcher")
    codes = lm.cell_codes(fr)
    mine = lm.CellMap(codes, df["xT"].values).value(codes)
    ref_df = fc.add_loc_bins(df.copy())
    ref = fc.PooledLocationMap(ref_df).apply(ref_df).values
    assert np.allclose(mine, ref, atol=1e-12)


def test_integer_weights_equal_duplicated_rows():
    df = synth_pitches(n=20000)
    rng = np.random.default_rng(5)
    w = rng.integers(0, 4, len(df)).astype(float)
    fr = lm.frame_columns(df, "batter_platoon")
    codes = lm.cell_codes(fr)
    weighted = lm.CellMap(codes, df["xT"].values, w=w, count=True, m=5)
    dup_idx = np.repeat(np.arange(len(df)), w.astype(int))
    dup = lm.CellMap(lm.subset_codes(codes, dup_idx), df["xT"].values[dup_idx],
                     count=True, m=5)
    assert np.allclose(weighted.value(codes), dup.value(codes), atol=1e-12)
    assert np.allclose(weighted.relative(codes), dup.relative(codes), atol=1e-12)


def test_batter_frame_positive_x_is_inside():
    df = pd.DataFrame({"PlateLocSide": [0.8, 0.8, -0.8, -0.8],
                       "PlateLocHeight": [2.0] * 4, "Balls": [0] * 4, "Strikes": [0] * 4,
                       "BatterSide": ["Right", "Left", "Right", "Left"],
                       "PitcherThrows": ["Right", "Right", "Left", "Left"]})
    fr = lm.frame_columns(df, "batter_platoon")
    # +0.8 is the right-handed batter's side: inside to a righty, away from a lefty.
    assert list(fr["x"]) == [0.8, -0.8, -0.8, 0.8]
    assert list(fr["p"]) == [1, 0, 0, 1]


def test_batter_frame_drops_unknown_hands():
    df = pd.DataFrame({"PlateLocSide": [0.1, 0.1, 0.1], "PlateLocHeight": [2.0] * 3,
                       "Balls": [0] * 3, "Strikes": [0] * 3,
                       "BatterSide": ["Right", None, "Left"],
                       "PitcherThrows": ["Right", "Right", "Both"]})
    assert list(lm.frame_columns(df, "batter_platoon")["ok"]) == [True, False, False]


def test_platoon_map_recovers_side_specific_value():
    # Same-side: away (x_in < 0) is good. Opposite-side: inside (x_in > 0) is good.
    rng = np.random.default_rng(2)
    n = 60000
    df = pd.DataFrame({"PlateLocSide": rng.uniform(-1.5, 1.5, n),
                       "PlateLocHeight": rng.uniform(1, 4, n),
                       "Balls": 0, "Strikes": 0,
                       "BatterSide": rng.choice(["Left", "Right"], n),
                       "PitcherThrows": rng.choice(["Left", "Right"], n)})
    fr = lm.frame_columns(df, "batter_platoon")
    truth = np.where(fr["p"] == 1, 0.05 * fr["x"], -0.05 * fr["x"])
    y = truth + rng.normal(0, 0.05, n)
    codes = lm.cell_codes(fr)
    got = lm.CellMap(codes, y).value(codes)
    assert np.corrcoef(got, truth)[0, 1] > 0.95


def test_count_relative_carries_no_occupancy():
    """Two pitchers hit identical locations; one lives in 0-2, the other in 3-0. Count moves
    xT a lot, location moves it a little. Their count-relative grades must match."""
    rng = np.random.default_rng(9)
    n = 80000
    loc_x = rng.normal(0, 0.8, n)
    loc_z = rng.normal(2.3, 0.8, n)
    balls = rng.integers(0, 4, n)
    strikes = rng.integers(0, 3, n)
    count_eff = 0.04 * balls - 0.03 * strikes
    y = count_eff + 0.02 * (np.abs(loc_x) < 0.5) + rng.normal(0, 0.1, n)
    train = pd.DataFrame({"PlateLocSide": loc_x, "PlateLocHeight": loc_z, "Balls": balls,
                          "Strikes": strikes, "BatterSide": "Right", "PitcherThrows": "Right"})
    m_x, m_z = rng.normal(0, 0.8, 3000), rng.normal(2.3, 0.8, 3000)
    a = pd.DataFrame({"PlateLocSide": m_x, "PlateLocHeight": m_z, "Balls": 0, "Strikes": 2,
                      "BatterSide": "Right", "PitcherThrows": "Right"})
    b = a.assign(Balls=3, Strikes=0)
    allrows = pd.concat([train, a, b], ignore_index=True)
    codes = lm.cell_codes(lm.frame_columns(allrows, "batter_platoon"))
    tr = lm.subset_codes(codes, np.arange(n))
    mp = lm.CellMap(tr, y, count=True, m=5)
    ra = mp.relative(lm.subset_codes(codes, np.arange(n, n + 3000))).mean()
    rb = mp.relative(lm.subset_codes(codes, np.arange(n + 3000, n + 6000))).mean()
    raw_gap = (mp.value(lm.subset_codes(codes, np.arange(n + 3000, n + 6000))).mean()
               - mp.value(lm.subset_codes(codes, np.arange(n, n + 3000))).mean())
    assert raw_gap > 0.1          # the raw value does carry the count
    assert abs(ra - rb) < 0.004   # the relative value does not


def test_sparse_count_cell_relative_value_is_pooled_location():
    # A count-cell with no train rows must fall back to pooled(location) - platoon mean.
    df = synth_pitches(n=30000)
    df.loc[:, "Balls"] = 0
    df.loc[:, "Strikes"] = 0
    probe = df.iloc[:200].copy()
    probe["Balls"], probe["Strikes"] = 3, 2
    allrows = pd.concat([df, probe], ignore_index=True)
    codes = lm.cell_codes(lm.frame_columns(allrows, "batter_platoon"))
    tr = lm.subset_codes(codes, np.arange(len(df)))
    mp = lm.CellMap(tr, df["xT"].values, count=True, m=5)
    pc = lm.subset_codes(codes, np.arange(len(df), len(allrows)))
    expect = mp.pooled(pc) - mp.p_mean[pc["p"]]
    assert np.allclose(mp.relative(pc), expect, atol=1e-12)


def test_tune_m_prefers_pooling_when_count_is_irrelevant():
    df = synth_pitches(n=40000)
    codes = lm.cell_codes(lm.frame_columns(df, "batter_platoon"))
    best, errs = lm.tune_m(codes, df["xT"].values, df["PitcherId"].values)
    assert best == 100
    assert errs[100] <= errs[1]


def simulate_means(n_pitchers=3000, s2=0.09, tau2=4e-4, seed=4):
    rng = np.random.default_rng(seed)
    true = rng.normal(0.0, np.sqrt(tau2), n_pitchers)
    ns = rng.integers(15, 400, n_pitchers)
    vals = np.concatenate([rng.normal(t, np.sqrt(s2), k) for t, k in zip(true, ns)])
    grp = np.repeat(np.arange(n_pitchers), ns)
    return vals, grp, true, ns


def test_eb_moments_recover_known_variances():
    vals, grp, _, _ = simulate_means()
    s2, tau2, tab = lm.eb_moments(vals, grp)
    assert s2 == pytest.approx(0.09, rel=0.02)
    assert tau2 == pytest.approx(4e-4, rel=0.15)
    assert len(tab) == 3000


def test_eb_shrink_beats_raw_means_on_truth():
    vals, grp, true, ns = simulate_means()
    s2, tau2, tab = lm.eb_moments(vals, grp)
    raw = tab["mean"].values
    post = lm.eb_shrink(raw, tab["n"].values, np.full(len(raw), vals.mean()), s2, tau2)
    assert np.mean((post - true) ** 2) < 0.8 * np.mean((raw - true) ** 2)
    # Heavily sampled pitchers are barely moved; thin ones are pulled hard.
    w = tab["n"].values / (tab["n"].values + s2 / tau2)
    assert w.max() > 0.5 and w.min() < 0.1


def test_eb_shrink_zero_tau2_returns_prior():
    out = lm.eb_shrink([1.0, 2.0], [10, 100], [0.5, 0.5], 0.1, 0.0)
    assert list(out) == [0.5, 0.5]


def test_tau2_floor_applies():
    # No between-pitcher signal: tau2 must hit the floor, not go negative.
    # Pitcher means spread far less than sampling noise alone would produce.
    rng = np.random.default_rng(8)
    grp = np.repeat(np.arange(200), 100)
    noise = rng.normal(0, 0.3, 200 * 100)
    noise -= pd.Series(noise).groupby(grp).transform("mean").values
    vals = noise + np.repeat(rng.normal(0, 0.001, 200), 100)
    s2, tau2, tab = lm.eb_moments(vals, grp)
    assert tau2 == pytest.approx(lm.TAU2_FLOOR_FRAC * tab["mean"].var(ddof=1))


def test_hier_prior_recovers_cross_pitch_slope():
    rng = np.random.default_rng(6)
    n = 5000
    command = rng.normal(0, 1, n)
    O = command + rng.normal(0, 0.3, n)
    L = 0.01 + 0.004 * command + rng.normal(0, 0.001, n)
    has = rng.random(n) > 0.2
    prior, b = lm.hier_prior(L, O, has, mu=0.01)
    assert b == pytest.approx(0.004 / (1 + 0.09), rel=0.08)   # attenuated by noise in O
    assert np.all(prior[~has] == 0.01)
