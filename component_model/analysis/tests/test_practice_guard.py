"""Bullpen and intrasquad pitches (Level == 'TeamExclusive') never reach training.

fair_criterion.load_pitches is the one door every training and qualifying-population
script walks through (08, 14, 16, the coach_* studies all call it). Before this guard
the only thing keeping practice rows out was that the extract builder happens to skip
`_unverified` files. These tests pin the deliberate gate: a fresh read drops them, a
cache written before the guard existed drops them, and --level (which defaults to "all
levels") cannot let them back in. Synthetic data only.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fair_criterion as fc


def _rows(level, year, pid, n, rng):
    out = []
    for i in range(n):
        out.append({
            "PitchUID": f"{level}-{year}-{pid}-{i}", "Date": f"{year}-04-{1 + i % 27:02d}",
            "Pitcher": f"Test-Pitcher, {pid}", "PitcherId": pid, "PitcherThrows": "Right",
            "PitcherTeam": "TST_A", "Batter": "Test-Batter, One", "BatterSide": "Right",
            "BatterTeam": "TST_B", "Balls": 0, "Strikes": 0,
            "TaggedPitchType": "Fastball", "PitchCall": "BallCalled",
            "TaggedHitType": None, "ExitSpeed": np.nan, "Angle": np.nan, "Target": 0.0,
            "SpinRate": rng.normal(2200, 100), "Extension": 6.0,
            "HorzBreak": rng.normal(8, 2), "InducedVertBreak": rng.normal(16, 2),
            "EffectiveVelo": 90.0, "RelHeight": 5.8, "RelSide": 1.5,
            "vertbreakdiff": 0.0, "horzbreakdiff": 0.0, "velocity_differential": 0.0,
            "PlateLocSide": 0.0, "PlateLocHeight": 2.5, "League": "TST",
            "Level": level, "GameID": f"g-{level}-{year}", "Inning": 1,
            "Top/Bottom": "Top", "PAofInning": 1, "PitchofPA": 1,
            "RelSpeed": rng.normal(91, 1),
        })
    return out


def _source(tmp_path):
    rng = np.random.default_rng(3)
    rows = []
    for year in (2025, 2026):
        rows += _rows("D1", year, 1, 20, rng)
        # a pen for the same pitcher, and one for an arm who never pitched a game
        rows += _rows(fc.PRACTICE_LEVEL, year, 1, 15, rng)
        rows += _rows(fc.PRACTICE_LEVEL, year, 2, 15, rng)
    path = tmp_path / "source.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _args(data, workdir, level=None):
    return argparse.Namespace(data=str(data), workdir=str(workdir),
                              year_pair=(2025, 2026), level=level)


def test_fresh_read_drops_practice_rows(tmp_path):
    df = fc.load_pitches(_args(_source(tmp_path), tmp_path))
    assert len(df) == 40
    assert (df["Level"] != fc.PRACTICE_LEVEL).all()
    assert 2 not in set(df["PitcherId"])  # the pen-only arm is not in any population


def test_stale_cache_with_practice_rows_is_filtered(tmp_path):
    src = _source(tmp_path)
    clean = fc.load_pitches(_args(src, tmp_path))
    # Simulate a cache written before the guard: practice rows present in the parquet.
    leaked = clean.iloc[:5].copy()
    leaked["Level"] = fc.PRACTICE_LEVEL
    leaked["PitchUID"] = leaked["PitchUID"] + "-pen"
    cache = tmp_path / "pitches_cache_2025_2026.parquet"
    assert cache.exists()
    pd.concat([clean, leaked], ignore_index=True).to_parquet(cache, index=False)

    again = fc.load_pitches(_args(src, tmp_path))
    assert len(again) == len(clean)
    assert (again["Level"] != fc.PRACTICE_LEVEL).all()


def test_level_filter_cannot_readmit_practice(tmp_path):
    df = fc.load_pitches(_args(_source(tmp_path), tmp_path, level=fc.PRACTICE_LEVEL))
    assert df.empty


def test_exclude_practice_passes_frames_without_level():
    df = pd.DataFrame({"PitchUID": ["a", "b"]})
    assert fc.exclude_practice(df) is df
