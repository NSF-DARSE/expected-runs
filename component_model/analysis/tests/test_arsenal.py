"""Unit tests for arsenal.py.

Synthetic-recovery style, matching test_reliability_curves.py: build data with
known properties, assert the estimator recovers them. Nothing in real data
reveals the true display scale, so these are the only correctness check.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arsenal as ar


def test_to_display_negates_the_run_value_convention():
    """Lower expected runs must map to a HIGHER display score."""
    better = ar.to_display(-0.02, mu=0.0, sd=0.01)
    worse = ar.to_display(0.02, mu=0.0, sd=0.01)
    assert better > 100 > worse


def test_to_display_puts_one_sd_at_fifteen_points():
    assert ar.to_display(-0.01, mu=0.0, sd=0.01) == pytest.approx(115.0)
    assert ar.to_display(0.01, mu=0.0, sd=0.01) == pytest.approx(85.0)


def test_to_display_commutes_with_averaging():
    """The whole point of one affine scale: per-pitch grades must average to
    the grade of the average pitch, exactly. This is the property that keeps
    pitch, outing, type, and pitcher numbers additive and mutually consistent.
    Asserted exactly, not approximately -- affine maps have no slack here.
    """
    values = np.array([-0.031, 0.004, 0.017, -0.009, 0.022])
    mu, sd = 0.002, 0.013
    mean_of_grades = ar.to_display(values, mu, sd).mean()
    grade_of_mean = ar.to_display(values.mean(), mu, sd)
    assert mean_of_grades == pytest.approx(grade_of_mean, abs=1e-12)


def test_display_scale_uses_only_rows_above_the_floor():
    """Pitchers below the sample floor must not influence the scale, or a few
    tiny-sample outliers would widen sd and compress everyone's score.
    """
    means = np.array([0.0, 0.01, -0.01, 5.0])
    floor_mask = np.array([True, True, True, False])
    mu, sd = ar.display_scale(means, floor_mask)
    assert mu == pytest.approx(0.0)
    assert sd == pytest.approx(np.std([0.0, 0.01, -0.01], ddof=1))


def _toy_model(n_feats=4, seed=3):
    """A standardizer + linear model whose parameters we control exactly."""
    rng = np.random.default_rng(seed)
    scaler_mean = rng.normal(0, 1, n_feats)
    scaler_scale = rng.uniform(0.5, 2.0, n_feats)
    coef = rng.normal(0, 0.01, n_feats)
    return scaler_mean, scaler_scale, coef


def test_contributions_sum_to_the_display_gap_exactly():
    """The load-bearing property. Ridge on standardized features is linear, so
    the per-trait contributions must account for the ENTIRE difference in Stuff+
    between the subject and its baseline -- no residual, no rounding slack.
    Exact equality, because any tolerance here would hide a real bug.
    """
    scaler_mean, scaler_scale, coef = _toy_model()
    sd = 0.02
    subject = np.array([1.4, -0.3, 0.8, 2.1])
    baseline = np.array([0.2, 0.1, -0.4, 0.9])

    baseline_z = (baseline - scaler_mean) / scaler_scale
    contrib = ar.contributions(subject, scaler_mean, scaler_scale, coef, baseline_z, sd)

    # Ridge prediction is intercept + z @ coef, so the intercept cancels in a gap.
    subject_pred = ((subject - scaler_mean) / scaler_scale) @ coef
    baseline_pred = baseline_z @ coef
    display_gap = ar.to_display(subject_pred, 0.0, sd) - ar.to_display(baseline_pred, 0.0, sd)

    assert contrib.sum() == pytest.approx(display_gap, abs=1e-10)


def test_contributions_are_zero_when_subject_equals_baseline():
    scaler_mean, scaler_scale, coef = _toy_model()
    subject = np.array([1.4, -0.3, 0.8, 2.1])
    baseline_z = (subject - scaler_mean) / scaler_scale
    contrib = ar.contributions(subject, scaler_mean, scaler_scale, coef, baseline_z, 0.02)
    assert np.allclose(contrib, 0.0, atol=1e-12)


def test_contribution_sign_follows_the_display_convention():
    """A feature that LOWERS expected runs must show a POSITIVE contribution,
    because the display scale is higher-is-better.
    """
    scaler_mean = np.array([0.0])
    scaler_scale = np.array([1.0])
    coef = np.array([-0.01])  # more of this feature => fewer runs => better
    contrib = ar.contributions(np.array([2.0]), scaler_mean, scaler_scale, coef,
                               baseline_z=np.array([0.0]), sd=0.02)
    assert contrib[0] > 0


def test_percentile_ranks_against_the_reference_population():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert ar.percentile(ref, 0.5) == 0
    assert ar.percentile(ref, 2.5) == 50
    assert ar.percentile(ref, 5.0) == 100


def test_percentile_ignores_missing_reference_values():
    ref = np.array([1.0, np.nan, 3.0, np.nan])
    assert ar.percentile(ref, 2.0) == 50


import pandas as pd


def _toy_pitches():
    """Two dates, known ridge_pred values, so aggregates are hand-checkable."""
    return pd.DataFrame({
        "Date": ["2026-03-01", "2026-03-01", "2026-03-08", "2026-03-08", "2026-03-08"],
        "ridge_pred": [0.00, 0.02, -0.01, -0.03, 0.01],
    })


def test_outing_table_groups_by_date_and_grades_each_outing():
    out = ar.outing_table(_toy_pitches(), mu=0.0, sd=0.02)
    assert list(out["date"]) == ["2026-03-01", "2026-03-08"]
    assert list(out["n"]) == [2, 3]
    # First outing mean ridge_pred is 0.01 -> 100 - 15*(0.01/0.02) = 92.5
    assert out.loc[0, "stuff"] == pytest.approx(92.5)


def test_outing_grades_average_to_the_overall_grade_when_outings_are_equal_size():
    """Sanity check that outing grades live on the same scale as everything else."""
    pitches = pd.DataFrame({
        "Date": ["2026-03-01", "2026-03-01", "2026-03-08", "2026-03-08"],
        "ridge_pred": [0.00, 0.02, -0.01, -0.03],
    })
    out = ar.outing_table(pitches, mu=0.0, sd=0.02)
    overall = ar.to_display(pitches["ridge_pred"].mean(), 0.0, 0.02)
    assert out["stuff"].mean() == pytest.approx(overall, abs=1e-12)


def test_recent_change_is_none_when_a_window_is_below_the_floor():
    """A blank reads as 'not enough to say'; a zero would wrongly read as
    'no change'. So below the floor must return None, never 0.0.
    """
    outings = pd.DataFrame({
        "date": ["2026-01-05", "2026-03-01"],
        "n": [40, 5],
        "stuff": [110.0, 95.0],
    })
    assert ar.recent_change(outings, floor_n=30, asof="2026-03-10") is None


def test_recent_change_differences_the_two_thirty_day_windows():
    outings = pd.DataFrame({
        "date": ["2026-01-20", "2026-03-01"],   # prior window, then recent window
        "n": [50, 50],
        "stuff": [100.0, 112.0],
    })
    got = ar.recent_change(outings, floor_n=30, asof="2026-03-10")
    assert got == pytest.approx(12.0)
