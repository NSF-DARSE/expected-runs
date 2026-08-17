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


def test_type_mask_uses_the_is_ff_flag_for_four_seams():
    """FF must come from is_ff, which already unifies the three source spellings,
    rather than from a literal string match that would silently drop two of them.
    """
    pit = pd.DataFrame({
        "TaggedPitchType": ["Fastball", "FourSeamFastBall", "FourSeamFastball", "Slider"],
        "is_ff": [True, True, True, False],
    })
    assert list(ar.type_mask(pit, None)) == [True, True, True, False]


def test_type_mask_treats_two_seam_as_sinker():
    pit = pd.DataFrame({
        "TaggedPitchType": ["Sinker", "TwoSeamFastBall", "Slider"],
        "is_ff": [False, False, False],
    })
    assert list(ar.type_mask(pit, {"Sinker", "TwoSeamFastBall"})) == [True, True, False]


# ---- Adj Results ladder (08_staff_scores.py): Target -> xT -> adjT --------
#
# The ladder shows the same C2 quantity (adjT, Adj Results) at the two earlier
# stages it is built from (RESULTS.md, "The fair criterion"): C0 raw Target,
# C1 xT. Summing a level plus the two gaps back to the far endpoint is pure
# telescoping algebra (a + (b-a) + (c-b) == c) and holds no matter what (mu,
# sd) went into a, b and c individually -- that part can never break. What
# shared moments actually buy is that a GAP VALUE MEANS ONLY ONE THING: with
# one (mu, sd) for every level, "Defense & Luck" collapses to
# -15*(xt-target)/sd, a pure function of the raw xt-target difference and
# nothing else, so the same physical luck swing prices identically everywhere
# on the board. Per-level moments (the "obvious" per-quantity choice a future
# reader will be tempted to make) drag in each level's own mu and sd, so the
# SAME raw luck swing prices differently depending on the pitcher's unrelated
# absolute level -- a real inconsistency a coach could eventually notice, even
# though every individual card still closes arithmetically. That is the
# "lying in a way nobody could see": each card is locally consistent, and the
# whole board is not.

def test_shared_scale_ladder_telescopes_exactly():
    mu, sd = 0.0016, 0.0194  # the adjT (C2) population moments, per RESULTS.md
    target, xt, adj = 0.0055, 0.0011, -0.0009
    runs_allowed = ar.to_display(target, mu, sd)
    exp_runs_allowed = ar.to_display(xt, mu, sd)
    adj_results = ar.to_display(adj, mu, sd)
    gap_defense_luck = exp_runs_allowed - runs_allowed
    gap_opponent = adj_results - exp_runs_allowed
    assert runs_allowed + gap_defense_luck + gap_opponent == pytest.approx(adj_results)


def test_shared_scale_prices_the_same_luck_swing_identically():
    """The property a shared scale buys: two pitchers with the IDENTICAL
    (xt - target) gap -- the same physical luck/defense swing -- must get the
    identical 'Defense & Luck' point value, regardless of where their raw
    numbers otherwise sit.
    """
    mu, sd = 0.0016, 0.0194
    pairs = [(0.0055, 0.0011), (0.0002, -0.0042)]  # both: xt - target == -0.0044
    gaps = [ar.to_display(xt, mu, sd) - ar.to_display(target, mu, sd) for target, xt in pairs]
    assert gaps[0] == pytest.approx(gaps[1])


def test_per_level_scale_prices_the_same_luck_swing_differently():
    """Regression guard for the change a future reader will be tempted to
    make: giving Target and xT their OWN population (mu, sd) instead of
    reusing adjT's. Each individual card still closes (see the telescoping
    test above -- that identity cannot break), but the SAME raw luck swing
    now prices differently depending on unrelated absolute level, which is
    exactly the inconsistency shared scale exists to rule out.
    """
    mu_t, sd_t = 0.002, 0.028    # Target's own population moments
    mu_x, sd_x = 0.0016, 0.020   # xT's own population moments
    pairs = [(0.0055, 0.0011), (0.0002, -0.0042)]  # both: xt - target == -0.0044
    gaps = [ar.to_display(xt, mu_x, sd_x) - ar.to_display(target, mu_t, sd_t) for target, xt in pairs]
    assert gaps[0] != pytest.approx(gaps[1])


# ---- out-of-zone banding -----------------------------------------------------

def _region_at(plate_side, height, batter_side="Right"):
    """Region label for a single pitch, through the production path."""
    import pandas as pd
    df = pd.DataFrame({"PlateLocSide": [plate_side], "PlateLocHeight": [height],
                       "BatterSide": [batter_side]})
    return ar.region_series(df).iloc[0]


def test_in_zone_regions_are_untouched_by_the_out_of_zone_split():
    """Splitting the miss into bands must not move a single in-zone label."""
    assert _region_at(0.0, 2.5) == "Middle, middle"
    # Away is negative PlateLocSide for a RHH, low is under the middle third.
    assert _region_at(-0.7, 1.8) == "Down and away"
    assert _region_at(0.7, 3.3) == "Up and in"
    # A pitch on the zone edge is still in the zone, not "just off" it.
    assert _region_at(-ar.ZONE_HALF_WIDTH, 2.5) == "Middle and away"
    assert _region_at(0.0, ar.ZONE_TOP) == "Up, middle"


@pytest.mark.parametrize("inches_out,expected", [
    (0.5, "Just off the zone"),
    (2.0, "Just off the zone"),    # exactly on the first boundary
    (2.01, "Off the zone"),
    (5.9, "Off the zone"),
    (6.0, "Off the zone"),         # exactly on the second boundary
    (6.1, "Way off the zone"),
    (18.0, "Way off the zone"),
])
def test_out_of_zone_bands_split_by_distance_horizontally(inches_out, expected):
    side = -(ar.ZONE_HALF_WIDTH + inches_out / 12.0)   # away from a RHH
    assert _region_at(side, 2.5) == expected


@pytest.mark.parametrize("inches_out,expected", [
    (1.0, "Just off the zone"),
    (4.0, "Off the zone"),
    (10.0, "Way off the zone"),
])
def test_out_of_zone_bands_apply_to_height_misses_too(inches_out, expected):
    """A miss over the hitter's head is banded the same way as one off the
    side; the band names must not read as side-only."""
    assert _region_at(0.0, ar.ZONE_TOP + inches_out / 12.0) == expected
    assert _region_at(0.0, ar.ZONE_BOTTOM - inches_out / 12.0) == expected


def test_out_of_zone_distance_is_the_corner_diagonal_when_both_bands_miss():
    """A pitch missing in BOTH directions is measured to the zone's corner, not
    to the nearer edge -- 3-4-5, so 3in wide and 4in high is 5in out."""
    side = -(ar.ZONE_HALF_WIDTH + 3.0 / 12.0)
    height = ar.ZONE_TOP + 4.0 / 12.0
    assert ar._zone_distance_inches([side], [height])[0] == pytest.approx(5.0)
    assert _region_at(side, height) == "Off the zone"


def test_zone_distance_is_zero_inside_the_zone():
    assert ar._zone_distance_inches([0.0], [2.5])[0] == pytest.approx(0.0)


def test_out_of_zone_bands_are_handedness_symmetric():
    """The band depends on distance from the rectangle, which is symmetric in
    side, so the mirrored pitch to a LHH lands in the same band."""
    inside_out = ar.ZONE_HALF_WIDTH + 4.0 / 12.0
    assert _region_at(-inside_out, 2.5, "Right") == _region_at(inside_out, 2.5, "Left")
