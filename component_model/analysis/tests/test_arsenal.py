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
