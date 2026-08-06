"""Python side of the cross-language attribution fixture.

The browser computes trait attribution from shipped coefficients (spec Decision 1),
so the same arithmetic exists twice. This fixture is the pin: a byte-identical copy
in the frontend repo is asserted against the same expected values by the Vitest
suite. If either implementation drifts, one of the two suites goes red.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arsenal as ar

FIXTURE = Path(__file__).parent / "fixtures" / "attribution_fixture.json"


@pytest.fixture(scope="module")
def fx():
    return json.loads(FIXTURE.read_text())


def test_standardization_matches_fixture(fx):
    m = fx["model"]
    for case in fx["cases"]:
        z = (np.array(case["featureValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
        assert z == pytest.approx(case["expectedZ"], abs=1e-12), case["name"]


def test_contributions_match_fixture(fx):
    m = fx["model"]
    for case in fx["cases"]:
        got = ar.contributions(case["featureValues"], m["scalerMean"], m["scalerScale"],
                               m["coef"], case["baselineZ"], m["displaySd"])
        assert list(got) == pytest.approx(case["expectedContributions"], abs=1e-10), case["name"]


def test_contributions_sum_to_the_gap_exactly(fx):
    """Additivity is an equality property of a linear model, not a tolerance."""
    m = fx["model"]
    for case in fx["cases"]:
        got = ar.contributions(case["featureValues"], m["scalerMean"], m["scalerScale"],
                               m["coef"], case["baselineZ"], m["displaySd"])
        assert float(got.sum()) == pytest.approx(case["expectedSum"], abs=1e-10), case["name"]


def test_sum_equals_the_display_gap_the_model_would_predict(fx):
    """The independent check: the summed contributions must equal the difference in
    to_display() between the subject's prediction and the baseline's, so the
    waterfall cannot disagree with the number above it."""
    m = fx["model"]
    coef, mu, sd = np.array(m["coef"]), 0.0, m["displaySd"]
    for case in fx["cases"]:
        z = (np.array(case["featureValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
        pred_subject = float(coef @ z)
        pred_baseline = float(coef @ np.array(case["baselineZ"]))
        gap = float(ar.to_display(pred_subject, mu, sd) - ar.to_display(pred_baseline, mu, sd))
        assert gap == pytest.approx(case["expectedSum"], abs=1e-10), case["name"]


def test_typical_values_standardize_to_the_second_case_baseline(fx):
    """The 'own typical pitch' baseline must be the standardized typical values,
    not a separately authored array. Catches a fixture that quietly disagrees
    with itself."""
    m = fx["model"]
    case = next(c for c in fx["cases"] if "typicalValues" in c)
    z = (np.array(case["typicalValues"]) - np.array(m["scalerMean"])) / np.array(m["scalerScale"])
    assert list(z) == pytest.approx(case["baselineZ"], abs=1e-12)
