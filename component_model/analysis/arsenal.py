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
import pandas as pd

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


RECENT_WINDOW_DAYS = 30


def outing_table(pitches: pd.DataFrame, mu: float, sd: float) -> pd.DataFrame:
    """One row per date the pitcher threw this pitch type.

    Grades each outing with the same transform used at every other level, so
    outing numbers are directly comparable to the pitch-type number.

    Date is normalized to a YYYY-MM-DD string first: load_pitches leaves the
    source Date column as-is, so it can arrive as either a string or a datetime,
    and a datetime would stringify with a spurious " 00:00:00" into the bundle.
    """
    dates = pd.to_datetime(pitches["Date"]).dt.strftime("%Y-%m-%d")
    g = pitches.assign(_date=dates).groupby("_date")["ridge_pred"].agg(["size", "mean"]).reset_index()
    g.columns = ["date", "n", "mean_ridge"]
    g = g.sort_values("date").reset_index(drop=True)
    g["stuff"] = to_display(g["mean_ridge"].values, mu, sd)
    return g[["date", "n", "stuff"]]


def recent_change(outings: pd.DataFrame, floor_n: int, asof: str) -> float | None:
    """Stuff+ over the trailing 30 days minus the 30 days before that.

    Returns None when either window is below the sample floor, so the UI can
    render a blank. A zero would be read as "no change", which is a different
    and wrong claim.
    """
    asof_ts = pd.Timestamp(asof)
    dates = pd.to_datetime(outings["date"])
    recent = outings[(dates > asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS)) & (dates <= asof_ts)]
    prior_lo = asof_ts - pd.Timedelta(days=2 * RECENT_WINDOW_DAYS)
    prior = outings[(dates > prior_lo) & (dates <= asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS))]
    if recent["n"].sum() < floor_n or prior["n"].sum() < floor_n:
        return None
    recent_mean = np.average(recent["stuff"].values, weights=recent["n"].values)
    prior_mean = np.average(prior["stuff"].values, weights=prior["n"].values)
    return float(recent_mean - prior_mean)


def windowed_delta(values: pd.Series, dates: pd.Series, asof: str,
                   floor_n: int) -> dict | None:
    """Trailing 30 days vs the 30 days before, as a mean difference WITH its SE.

    This is the re-entry contract for the arsenal table's removed change column:
    the display may only put a sign on a movement it can also say is readable,
    which needs a per-row standard error, not just a delta.

    Returns None when either window is under floor_n pitches, so the UI renders
    a blank -- a zero would claim "no change", which is a different statement.

    The SE is Welch on pitch-level values. Pitches within one outing are not
    independent, so this runs slightly small on metrics with real outing-level
    drift; the 2-SE display gate is the margin that absorbs it.
    """
    asof_ts = pd.Timestamp(asof)
    d = pd.to_datetime(dates)
    v = pd.Series(np.asarray(values, dtype=float), index=d.index)
    recent = v[(d > asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS)) & (d <= asof_ts)]
    prior_lo = asof_ts - pd.Timedelta(days=2 * RECENT_WINDOW_DAYS)
    prior = v[(d > prior_lo) & (d <= asof_ts - pd.Timedelta(days=RECENT_WINDOW_DAYS))]
    if len(recent) < floor_n or len(prior) < floor_n:
        return None
    se = float(np.sqrt(recent.var(ddof=1) / len(recent) + prior.var(ddof=1) / len(prior)))
    return {
        "recent": round(float(recent.mean()), 3),
        "prior": round(float(prior.mean()), 3),
        "delta": round(float(recent.mean() - prior.mean()), 3),
        "se": round(se, 4),
        "nRecent": int(len(recent)),
        "nPrior": int(len(prior)),
    }


def type_mask(pit: pd.DataFrame, tags: set[str] | None) -> pd.Series:
    """Row mask selecting one pitch type.

    tags=None means four-seam, taken from the frame's is_ff flag because that
    flag already unifies the three spellings the source data uses.
    """
    if tags is None:
        return pit["is_ff"]
    return pit["TaggedPitchType"].isin(tags)


def fit_type(pit: pd.DataFrame, tags: set[str] | None, floor_n: int, fc_module, season_year: int) -> dict:
    """Fit the ridge for one pitch type and derive its display scale.

    Protocol copied from build_portal_data.py (arsenal grade, adopted 2026-07-23):
    one ridge per pitch type via fc.stuff_ridge(pitch_mask=...), then a display
    scale from that type's qualifying pitchers.

    season_year is the canonical year role to grade (fair_criterion relabels the
    year pair to 2024/2025 roles, so pass 2025 for the later season).

    Raises ValueError if the type has too few qualifying pitchers to scale.
    """
    mask = type_mask(pit, tags)
    pp, model = fc_module.stuff_ridge(pit, pitch_mask=mask, return_model=True)
    pp = pp[pp["PlateLocSide"].notna() & pp["PlateLocHeight"].notna()].copy()
    season = pp[pp["year"] == season_year].copy()

    per_pitcher = season.groupby("PitcherId")["ridge_pred"].agg(["size", "mean"])
    mu, sd = display_scale(per_pitcher["mean"].values, (per_pitcher["size"] >= floor_n).values)

    scaler = model.named_steps["standardscaler"]
    coef = model.named_steps["ridge"].coef_
    feats = fc_module.FEATS
    qualified = per_pitcher.index[per_pitcher["size"] >= floor_n]
    feature_means = season.groupby("PitcherId")[feats].mean()
    population_mean_z = ((feature_means.loc[qualified].values - scaler.mean_) / scaler.scale_).mean(axis=0)

    return {
        "pitches": season,
        "model": model,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "coef": coef,
        "mu": mu,
        "sd": sd,
        "population_mean_z": population_mean_z,
        "reference_features": feature_means.loc[qualified],
        "n_qualified": int(len(qualified)),
    }
