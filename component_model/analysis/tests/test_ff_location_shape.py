"""Tests for coach_ff_location_shape_gate's shape binning: cut points come from train rows only,
missing inputs get their own bin, and folding the bin into the platoon code gives one surface
per bin whose relative value carries no shape level (a better pitch earns nothing by itself).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# The gate modules resolve their workdirs at import; nothing here reads them.
for _v in ("STUFFPLUS_DATA", "STUFFPLUS_WORKDIR", "STUFFPLUS_WORKDIR_CRIT"):
    os.environ.setdefault(_v, "unused-by-these-tests")
import location_maps as lm  # noqa: E402
from coach_ff_location_shape_gate import shape_bins, shape_columns  # noqa: E402


def test_cut_points_use_train_rows_only():
    v = np.r_[np.arange(300, dtype=float), np.full(300, 1e6)]
    train = np.r_[np.ones(300, bool), np.zeros(300, bool)]
    b, cuts = shape_bins(pd.DataFrame({"a": v}), ("a",), train)
    assert cuts["a"] == [np.quantile(np.arange(300.0), 1 / 3), np.quantile(np.arange(300.0), 2 / 3)]
    assert np.bincount(b[:300]).tolist() == [100, 100, 100]
    assert (b[300:] == 2).all()


def test_two_columns_combine_and_missing_gets_own_bin():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=900), "h": rng.normal(size=900)})
    df.loc[5, "h"] = np.nan
    b, _ = shape_bins(df, ("a", "h"), np.ones(900, bool))
    assert b[5] == 9
    assert set(np.delete(b, 5)) == set(range(9))
    # First column is the slow index.
    lo_a = df["a"] < df["a"].quantile(0.2)
    assert (b[lo_a.values & (np.arange(900) != 5)] < 3).all()


def test_no_columns_is_one_bin():
    b, cuts = shape_bins(pd.DataFrame({"a": [1.0, 2.0]}), (), np.ones(2, bool))
    assert b.tolist() == [0, 0] and cuts == {}


def test_bin_in_platoon_code_removes_shape_level():
    rng = np.random.default_rng(3)
    n = 60000
    df = pd.DataFrame({"PlateLocSide": rng.normal(0, 0.8, n),
                       "PlateLocHeight": rng.normal(2.5, 0.8, n),
                       "Balls": 0, "Strikes": 0})
    shape = rng.integers(0, 3, n)
    # A better pitch (bin 2) is 0.05 cheaper everywhere; location adds the same surface.
    xt = -0.05 * shape + 0.03 * (df["PlateLocHeight"] > 3) + rng.normal(0, 0.1, n)
    fr = lm.frame_columns(df, "catcher")
    fr["p"] = fr["p"].values + shape
    codes = lm.cell_codes(fr)
    rel = lm.CellMap(codes, xt.values).relative(codes)
    # Each bin's relative values centre on zero, so the 0.05-per-bin shape level is gone. Not
    # exactly zero: cells that fall back to the coarse grid average over neighbouring rows.
    assert max(abs(rel[shape == k].mean()) for k in range(3)) < 0.002
    # The location surface survives inside each bin.
    hi = df["PlateLocHeight"].values > 3
    for k in range(3):
        d = rel[(shape == k) & hi].mean() - rel[(shape == k) & ~hi].mean()
        assert abs(d - 0.03) < 0.01


def test_shape_columns_mirror_arm_side():
    rows = pd.DataFrame({"RelSpeed": [92.0, 92.0], "Extension": [6.0, 6.0],
                         "InducedVertBreak": [17.0, 17.0], "RelHeight": [5.8, 5.8],
                         "HorzBreak": [9.0, -9.0], "is_lhp": [0.0, 1.0]})
    s = shape_columns(rows)
    assert s["HorzBreak_arm"].tolist() == [9.0, 9.0]
    assert s["vaa_flat"].iloc[0] == s["vaa_flat"].iloc[1] < 0
