"""Per-pitch-type Stuff+ grading, shared by the pitcher-page scorer.

SIGN CONVENTION: ridge_pred is expected runs from the pitcher's perspective,
LOWER = better. to_display() is the single place that negation happens. Do not
negate anywhere else.

ONE SCALE: every Stuff+ number the pitcher page shows -- for a single pitch, an
outing, a pitch type, or a pitcher -- goes through to_display() with the same
(mu, sd) for that pitch type. Because to_display is affine, per-pitch grades
average exactly to the grade of the average pitch, so the numbers stay additive
and a coach can check them by addition. Introducing a second scale calibrated on
a different population breaks that and is a spec violation.

The per-type model protocol here follows component_model/portal/build_portal_data.py,
whose arsenal grade was adopted 2026-07-23 after beating FF-only on both D1 year
pairs. That script is left untouched; this module re-expresses the same protocol
in testable form.
"""
from __future__ import annotations

import numpy as np

# (display name, TaggedPitchType values). None means "use the frame's is_ff flag",
# which already covers the three source spellings of four-seam.
PITCH_TYPES: list[tuple[str, set[str] | None]] = [
    ("FF", None),
    ("Slider", {"Slider"}),
    ("ChangeUp", {"ChangeUp"}),
    ("Curveball", {"Curveball"}),
    ("Sinker", {"Sinker", "TwoSeamFastBall"}),
    ("Cutter", {"Cutter"}),
    ("Splitter", {"Splitter"}),
]

DISPLAY_CENTER = 100.0
DISPLAY_SPREAD = 15.0


def to_display(value, mu: float, sd: float):
    """Map an expected-run value onto the 100 +/- 15 display scale.

    Accepts a scalar or an array. Affine by construction, so it commutes with
    averaging -- see test_to_display_commutes_with_averaging.
    """
    if sd <= 0:
        raise ValueError(f"display sd must be positive, got {sd}")
    return DISPLAY_CENTER - DISPLAY_SPREAD * (np.asarray(value, dtype=float) - mu) / sd


def display_scale(pitcher_means, floor_mask) -> tuple[float, float]:
    """Population moments for the display scale, from qualifying pitchers only.

    pitcher_means: one mean ridge_pred per pitcher for a single pitch type.
    floor_mask: boolean, True where that pitcher cleared the sample floor.
    """
    vals = np.asarray(pitcher_means, dtype=float)[np.asarray(floor_mask, dtype=bool)]
    if vals.size < 2:
        raise ValueError(f"need 2+ qualifying pitchers to set a scale, got {vals.size}")
    return float(vals.mean()), float(vals.std(ddof=1))


def contributions(feature_values, scaler_mean, scaler_scale, coef, baseline_z, sd):
    """Per-feature contribution to Stuff+, in display points.

    This is the formula already used in 08_staff_scores.py, kept identical so the
    Staff Board and the pitcher page explain a grade the same way:

        z            = (value - scaler_mean) / scaler_scale
        contribution = -15 * (z - baseline_z) * coef / sd

    baseline_z is the standardized baseline the gap is measured against: the
    qualified-population mean for the default view, or the pitcher's own typical
    pitch when a single pitch is selected.

    Because the model is linear in standardized features, these sum exactly to the
    Stuff+ difference between subject and baseline.
    """
    if sd <= 0:
        raise ValueError(f"display sd must be positive, got {sd}")
    z = (np.asarray(feature_values, dtype=float) - np.asarray(scaler_mean, dtype=float)) / np.asarray(
        scaler_scale, dtype=float
    )
    return -DISPLAY_SPREAD * (z - np.asarray(baseline_z, dtype=float)) * np.asarray(coef, dtype=float) / sd


def percentile(reference_values, value) -> int:
    """Percentile rank of value within reference_values, 0-100.

    Reference is the qualifying population for one pitch type. NaNs are dropped
    rather than propagated, since a feature can be missing for some pitches.
    """
    ref = np.asarray(reference_values, dtype=float)
    ref = ref[~np.isnan(ref)]
    if ref.size == 0:
        raise ValueError("percentile needs a non-empty reference population")
    return int(round(100.0 * float((ref < value).mean())))
