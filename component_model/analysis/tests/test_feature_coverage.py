"""The season-wide feature-gap guard in fair_criterion.check_feature_coverage.

The failure it exists for: source_2025_2026_relspeed.csv carried RelSpeed for
the 2026 season only, and the symptom was a StandardScaler "0 sample(s)" error
per pitch type, three scripts away from the cause. The guard must name the
column and the real season instead.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fair_criterion as fc


def _frame(relspeed_2024):
    n = 8
    df = pd.DataFrame({
        "year": [2024] * (n // 2) + [2025] * (n // 2),
        "SpinRate": np.full(n, 2200.0),
        "RelSpeed": np.concatenate([relspeed_2024, np.full(n // 2, 92.0)]),
    })
    return df


def test_a_feature_missing_for_a_whole_season_is_refused_with_the_real_year():
    df = _frame(np.full(4, np.nan))
    with pytest.raises(RuntimeError) as excinfo:
        fc.check_feature_coverage(df, (2025, 2026))
    message = str(excinfo.value)
    assert "RelSpeed" in message
    # The role year is 2024 but the coach-facing season is 2025; the message
    # must name the real one, or it sends whoever reads it to the wrong data.
    assert "2025" in message


def test_partial_coverage_within_a_season_passes():
    # Scattered nulls are normal TrackMan data, not an extract defect; the
    # guard is for the all-or-nothing case only.
    df = _frame(np.array([92.0, np.nan, 91.0, np.nan]))
    fc.check_feature_coverage(df, (2025, 2026))  # must not raise


def test_a_feature_column_absent_entirely_is_not_this_guards_problem():
    # Extracts predating RelSpeed load without the column (OPTIONAL_COLS) and
    # fail on the KeyError in stuff_ridge, as designed. The guard must skip
    # what is not there rather than crash on it.
    df = _frame(np.full(4, 90.0)).drop(columns=["RelSpeed"])
    fc.check_feature_coverage(df, (2025, 2026))  # must not raise
