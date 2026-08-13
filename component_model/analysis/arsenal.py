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

    # Adjusted results for THIS pitch type, scaled on this type's own qualified
    # population, exactly as Stuff+ is. Legitimate off the fastball in a way
    # Location+ is not: adjT describes what happened to the pitch with luck,
    # defense and opponent quality removed, and a description does not need to
    # predict next season to be true. Read the number with its pitch count,
    # though: script 09 measured the criterion's own year-over-year reliability
    # at 0.304 for four-seams against 0.174 slider and 0.181 changeup.
    adj_mu = adj_sd = None
    if "adjT" in season.columns and season["adjT"].notna().any():
        per_adj = season.groupby("PitcherId")["adjT"].agg(["size", "mean"])
        try:
            adj_mu, adj_sd = display_scale(per_adj["mean"].values,
                                           (per_adj["size"] >= floor_n).values)
        except ValueError:
            # Too few qualifying pitchers to define a scale for this type. The
            # type still grades on Stuff+; it just carries no results column.
            adj_mu = adj_sd = None

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
        "adj_mu": adj_mu,
        "adj_sd": adj_sd,
        "n_qualified": int(len(qualified)),
    }


# ---------------- Location+ decomposition: where the score came from ----------

# The nominal zone the page draws, so a region named here is the region a coach
# sees on the plot. Half-plate plus a ball, and the conventional 1.5-3.5 ft band.
ZONE_HALF_WIDTH = 0.83
ZONE_BOTTOM, ZONE_TOP = 1.5, 3.5

# Away is NEGATIVE PlateLocSide for a right-handed hitter and POSITIVE for a
# left-handed one. Measured rather than assumed: on 727k four-seams the mean
# PlateLocSide is -0.214 against RHH and +0.277 against LHH, and the split holds
# inside each pitcher hand (RHP/LHH +0.286, LHP/LHH +0.245), so it is a property
# of which box the hitter stands in and not a pitcher-side artifact.
def _side_relative(plate_side, batter_side):
    """PlateLocSide re-expressed so POSITIVE is always away from the hitter."""
    return np.where(batter_side == "Right", -plate_side, plate_side)


def _height_band(z):
    third = (ZONE_TOP - ZONE_BOTTOM) / 3
    return np.select(
        [z < ZONE_BOTTOM, z < ZONE_BOTTOM + third, z < ZONE_BOTTOM + 2 * third, z <= ZONE_TOP],
        ["off", "Down", "Middle", "Up"], default="off")


def _side_band(s):
    third = (2 * ZONE_HALF_WIDTH) / 3
    return np.select(
        [s < -ZONE_HALF_WIDTH, s < -ZONE_HALF_WIDTH + third,
         s < -ZONE_HALF_WIDTH + 2 * third, s <= ZONE_HALF_WIDTH],
        ["off", "in", "middle", "away"], default="off")


def _region_label(hband, sband):
    if hband == "off" or sband == "off":
        return "Off the plate"
    if sband == "middle":
        return f"{hband}, middle"
    return f"{hband} and {sband}"


def count_bucket(count12):
    """Pitcher ahead, even, or behind. Three buckets rather than twelve counts:
    the comparison against a league share needs enough pitches in each cell to
    mean anything, and a coach reads put-away counts as one situation."""
    balls, strikes = (int(x) for x in str(count12).split("-"))
    if strikes > balls:
        return "ahead"
    if strikes == balls:
        return "even"
    return "behind"


def league_cell_table(league, qualified_ids=None):
    """Share and mean location value per region and count, over the comparison
    population.

    A separate step because of WHEN it has to run: 14_pitcher_pages narrows every
    fitted state down to one team before assembling records, so a decomposition
    that reads that frame compares a pitcher against his own teammates while
    labeling the column D1. This must be called before that narrowing.

    qualified_ids restricts it to the pitchers the display scale was built from,
    so the population a pitcher is compared against is the population his score
    is measured against.
    """
    df = league.copy()
    if qualified_ids is not None:
        df = df[df["PitcherId"].isin(qualified_ids)]
    if df.empty:
        raise ValueError("league frame is empty; the comparison population is missing")
    rel = _side_relative(df["PlateLocSide"].values, df["BatterSide"].values)
    hh = _height_band(df["PlateLocHeight"].values)
    bb = _side_band(rel)
    df["region"] = [_region_label(a, b) for a, b in zip(hh, bb)]
    df["bucket"] = [count_bucket(c) for c in df["count12"]]
    g = df.groupby(["region", "bucket"])
    return {"share": g.size() / len(df), "value": g["loc"].mean(), "n": int(len(df))}


def location_baseline(league, loc_mu, loc_sd):
    """The league's own location mix, in Location+ points. Identical for every
    pitcher, which is exactly why it is reported once instead of per row.

    It is the third term of the split in `location_decomposition`, summed over
    the FULL league cell set: sum over cells of w*(v* - mu). Restricted to only
    the cells one pitcher happens to throw to it is NOT constant (measured on a
    real staff it ranged -24 to -2 points, entirely because coverage differs),
    so a per-pitcher "baseline" computed that way would be a coverage artifact
    dressed as a league constant. Computing it here, from the league table
    alone, makes that mistake unavailable to a caller.
    """
    table = league if isinstance(league, dict) else league_cell_table(league)
    lg_share, lg_value = table["share"], table["value"]
    total = sum(
        float(lg_share[cell]) * (float(lg_value[cell]) - loc_mu)
        for cell in lg_share.index
    )
    return float(-DISPLAY_SPREAD * total / loc_sd)


def location_decomposition(sub, league, loc_mu, loc_sd, min_share=0.01):
    """Split a pitcher's Location+ into where he threw, against the league mix.

    Location+ is a mean of per-pitch location values, and a mean splits
    additively, so each cell's points are exact and they sum to his score minus
    100. `sub` is his pitches, `league` every pitch of that type this season.

    A row's `points` is the FULL term, share x (his value - the scale's zero),
    which mixes three things a reader will assume are separated:

      w(v - mu) = (w - w*)(v* - mu)   occupancy: he is here more, or less
                + w(v - v*)           placement: where he sits INSIDE the cell
                + w*(v* - mu)         the league's own mix, common to everyone

    So the first two are emitted per row as `occupancyPoints` and
    `placementPoints`, and the third is `location_baseline`, one scalar.

    Measured on a real 18-pitcher staff, occupancy correlates 0.82 with
    Location+ and placement 0.81, with standard deviations of 8.7 and 8.6 points
    against 14.1 for the score. Neither is the term that explains it. An earlier
    reading put occupancy at 0.40, but that was computed over only the cells a
    pitcher throws to, which drops every spot he AVOIDS -- most of the occupancy
    signal. That is the mistake the union below exists to make impossible.

    Cells are the UNION of his and the league's. A cell D1 throws to and he
    never does is an occupancy fact about him -- he avoids it -- worth
    -w*(v* - mu), and it is what makes the baseline term above come out
    constant across pitchers. Such a cell contributes nothing to `points`, and
    is pooled rather than given a line of its own: a spot he never throws to is
    not a spot on his card.

    A cell HE throws to that the league never does has no v*, so there is no
    inside-the-region comparison to make; v* falls back to his own value, which
    puts the whole term in occupancy where it belongs.

    Identity, exact to float error:
      sum(occupancyPoints) + sum(placementPoints) + baseline == score - 100
    """
    def cells(df):
        s = _side_relative(df["PlateLocSide"].values, df["BatterSide"].values)
        h = _height_band(df["PlateLocHeight"].values)
        b = _side_band(s)
        return pd.Series([_region_label(hh, bb) for hh, bb in zip(h, b)], index=df.index)

    sub = sub.copy()
    sub["region"] = cells(sub)
    sub["bucket"] = [count_bucket(c) for c in sub["count12"]]

    # Accepts a prebuilt table (production, snapshotted before the team filter)
    # or a raw frame (tests, where the two populations are the same rows).
    table = league if isinstance(league, dict) else league_cell_table(league)
    lg_share, lg_value = table["share"], table["value"]

    n = len(sub)
    his = {}
    for (region, bucket), g in sub.groupby(["region", "bucket"]):
        his[(region, bucket)] = (len(g) / n, float(g["loc"].mean()), int(len(g)))

    def terms(cell, share, his_value):
        """Occupancy, placement and points for one cell, all in display points."""
        lg_w = float(lg_share.get(cell, 0.0))
        lg_v = float(lg_value.get(cell, np.nan))
        if lg_w == 0.0 or not np.isfinite(lg_v):
            lg_v = his_value          # no league version of this spot to sit inside
        k = -DISPLAY_SPREAD / loc_sd
        return (k * (share - lg_w) * (lg_v - loc_mu),   # occupancy
                k * share * (his_value - lg_v),         # placement
                k * share * (his_value - loc_mu),       # the full term
                lg_w, lg_v)

    rows, dropped = [], []
    for cell in sorted(set(his) | set(lg_share.index)):
        share, his_value, count = his.get(cell, (0.0, float("nan"), 0))
        if share == 0.0:
            # He never throws here but D1 does. Real occupancy, no placement,
            # and no line of its own.
            his_value = float(lg_value.get(cell, loc_mu))
        occ, plac, points, lg_w, lg_v = terms(cell, share, his_value)
        row = {
            "region": cell[0],
            "count": cell[1],
            "n": count,
            "share": float(share),
            "leagueShare": lg_w,
            "points": float(points),
            "occupancyPoints": float(occ),
            "placementPoints": float(plac),
            "value": float(his_value),
            "leagueValue": float(lg_v),
        }
        if share < min_share or share == 0.0:
            dropped.append(row)
        else:
            rows.append(row)
    rows.sort(key=lambda r: -abs(r["points"]))

    # Everything too rare to earn its own line, pooled into one. Dropping those
    # cells outright is what made a real pitcher's rows sum to 9.40 against a
    # score of 10.91: individually negligible, collectively 1.5 points. The card
    # stays short and still adds up, which is the contract the trait table holds
    # itself to as well.
    #
    # The three point columns are summed from the per-cell terms, never
    # recomputed from the pooled shares and values: (W - W*)(V* - mu) over
    # aggregates is not the sum of its parts, and quietly would not add up.
    if dropped:
        share = sum(r["share"] for r in dropped)
        lg_w = sum(r["leagueShare"] for r in dropped)
        rows.append({
            "region": "Everywhere else",
            "count": "all",
            "n": int(sum(r["n"] for r in dropped)),
            "share": float(share),
            "leagueShare": float(lg_w),
            "points": float(sum(r["points"] for r in dropped)),
            "occupancyPoints": float(sum(r["occupancyPoints"] for r in dropped)),
            "placementPoints": float(sum(r["placementPoints"] for r in dropped)),
            # Display only. Share-weighted over exactly the cells pooled here,
            # so the pair is comparable; falls back to the other side when one
            # population has no pitches in any of them.
            "value": _pooled_mean(dropped, "value", "share", fallback_w="leagueShare"),
            "leagueValue": _pooled_mean(dropped, "leagueValue", "leagueShare",
                                        fallback_w="share"),
        })
    return rows


def _pooled_mean(rows, value_key, weight_key, fallback_w):
    """Share-weighted mean of one side of the pooled row, falling back to the
    other side's weights when this side has no pitches in any pooled cell."""
    w = [r[weight_key] for r in rows]
    if sum(w) <= 0:
        w = [r[fallback_w] for r in rows]
    if sum(w) <= 0:
        return float("nan")
    return float(np.average([r[value_key] for r in rows], weights=w))
