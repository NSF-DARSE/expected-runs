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


def pitch_display_scale(values) -> tuple[float, float]:
    """Population moments for a PITCH-LEVEL display scale.

    This is the sibling of display_scale, and the two must never be confused
    for one another. display_scale takes one MEAN PER PITCHER and describes
    the spread of pitcher averages -- that spread is small (a season of pitches
    averages out), so it is the right zero point and divisor for a SEASON
    score, where "100" means "an average pitcher's average pitch location."

    A single pitch is not a season average. Individual pitch locations vary
    enormously -- a pitcher who commands the ball still misses the target on
    any given throw -- so the spread of raw per-pitch values is much wider
    than the spread of pitcher means for the same population. Reusing
    display_scale's (mu, sd) for a per-pitch number divides by a divisor built
    for averages, which is why the shipped bug put individual four-seams
    anywhere from -281.5 to +232.7 on a scale that was supposed to be 100+/-15:
    the pitcher-mean sd (0.008769) was 7.8x too small for the spread of single
    pitches (0.0682).

    values: the raw run values (pitcher's perspective, lower = better) of
    EVERY QUALIFYING PITCHER'S PITCHES -- already filtered to the same
    qualifying population display_scale's pitcher_means/floor_mask select, so
    the season scale and the pitch scale describe the same set of pitchers.
    No separate floor_mask parameter here: filtering happens once, by the
    caller, against loc_qualified_ids, rather than being re-derived from a
    per-pitch frame that has no pitcher-level sample-size column of its own.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size < 2:
        raise ValueError(f"need 2+ pitches to set a pitch-level scale, got {vals.size}")
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
        return "Out of the zone"
    if sband == "middle":
        return f"{hband}, middle"
    return f"{hband} and {sband}"


# Order to display a pitcher's own count-bucket frequency in, everywhere a
# byCount list gets built (a real region row or the pooled "Everywhere else"
# row). Reading order a coach expects: ahead counts first, since those are the
# ones a pitcher chooses to expand a location to.
COUNT_BUCKET_ORDER = ("ahead", "even", "behind")


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
    """Share and mean location value per region, over the comparison
    population.

    A separate step because of WHEN it has to run: 14_pitcher_pages narrows every
    fitted state down to one team before assembling records, so a decomposition
    that reads that frame compares a pitcher against his own teammates while
    labeling the column D1. This must be called before that narrowing.

    qualified_ids restricts it to the pitchers the display scale was built from,
    so the population a pitcher is compared against is the population his score
    is measured against.

    COUNT COLLAPSED, ON PURPOSE. This table used to key on (region, count
    bucket), on the reasonable-sounding theory that a location map might value
    a spot differently depending on the count it was thrown in. Checked against
    the published bundle: it doesn't. The map is fit POOLED across counts, so
    in-zone league values come out identical to four decimals across
    ahead/even/behind. The occupancy term (w - w*)(v* - mu), summed over count
    buckets against a common v*, telescopes to the same term computed once on
    the combined region share -- the count split re-sliced rows without moving
    a single point of the grade. It also did real damage: dividing each
    region's share three ways pushed many (region, count) cells below
    min_share and into the pooled "Everywhere else" row. Collapsing to one row
    per region took the number of named spots falling under that threshold to
    ZERO across a full 18-pitcher staff, and rows per pitcher from a median of
    22 down to 10.

    A coach who wants the count breakdown back still gets it: see
    `location_decomposition`'s `byCount`, nested under each region row as a
    frequency-only breakdown of the pitcher's OWN usage. It is no longer a
    grouping key of this table or of the region rows, because the model does
    not condition on it and re-adding it here would just reopen the same
    min_share problem this collapse fixed. Do not restore the (region, bucket)
    grouping to "add the count dimension back" -- it was never carrying a
    location-value distinction, only slicing rows thinner.

    WEIGHTING, and why it has to match loc_mu exactly. loc_mu (14_pitcher_pages.py)
    is the mean of PER-PITCHER means: one vote per pitcher, regardless of how many
    pitches he threw. A plain pitch-level share/mean here would instead give more
    say to pitchers who threw more, and `location_baseline` sums w*(v* - loc_mu)
    over this table -- if w* and v* are pitch-weighted while mu is pitcher-weighted,
    that sum is not zero by construction, it is the gap between "the average pitch"
    and "the average pitcher", which on a real staff of 18 pitchers priced out to a
    constant +2.4 display points for everyone, masquerading as a location effect.

    So every pitcher gets one vote here too, split across the cells he threw to in
    proportion to his own pitch count. With P pitchers, pitcher p's share of cell c
    written s_pc and his mean loc in that cell v_pc:

        w*_c = (1/P) * sum_p s_pc              (mean per-pitcher share of the cell)
        v*_c = (sum_p s_pc v_pc) / (sum_p s_pc) (that same weighting applied to loc)

    which is implemented by giving every pitch a weight of 1/(P * n_p) (n_p = that
    pitcher's total pitches in this frame), summing the weight within a cell for
    share and the weight-times-loc within a cell (divided by the cell's weight) for
    value. That reduces to the two expressions above: summing 1/(P*n_p) over a
    pitcher's rows in cell c gives s_pc/P, so summing over pitchers gives w*_c;
    the weighted mean of loc with those same weights is v*_c by definition. Both
    share and value have to change together -- reweighting only `share` and
    leaving `value` a plain per-pitch mean does not make sum(w* v*) equal loc_mu.

    `n` stays a raw pitch count (not weighted): it is sample-size display, not a
    term in the identity.

    COUNT_WEIGHT, the league counterpart of a pitcher's own `byCount` share.
    location_decomposition nests each region row's byCount with the pitcher's
    OWN count-bucket frequency (his share of pitches to that region thrown
    ahead/even/behind). A coach asking "is 45% behind unusual" needs D1's own
    split of the SAME region the same way, or the frequency has nothing to sit
    next to.

    The denominator has to match: his share is per-region (the three buckets
    sum to 1.0 for one region, not across his whole arsenal), so the league
    figure has to be normalized the same way -- of D1's one-vote-per-pitcher
    weight in this region, what fraction sits in each count bucket. That is
    `count_weight[(region, bucket)] / share[region]`, and because
    `count_weight` is grouped by the same (region, bucket) partition of the
    same weighted rows that `share` sums over, the three bucket weights for one
    region always sum to exactly `share[region]` -- so the normalized fractions
    always sum to 1.0 for that region, same as his own.

    `count_weight` is returned RAW (not normalized here) rather than as
    already-divided shares, because the pooled "Everywhere else" row has to
    combine several regions' count weights before normalizing (sum raw weight
    per bucket across the pooled regions, then divide by the pooled region's
    total leagueShare) -- dividing per-region first and averaging the
    quotients would not equal that.
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

    # One vote per pitcher: 1/(P * n_p) per pitch, so a pitcher's rows sum to
    # 1/P no matter how many pitches he threw. No row can divide by zero here
    # -- pitcher_n is a count over rows actually present in df, so it is at
    # least 1 for every pitcher that appears at all.
    n_pitchers = df["PitcherId"].nunique()
    pitcher_n = df.groupby("PitcherId")["PitcherId"].transform("size")
    df["_w"] = 1.0 / (n_pitchers * pitcher_n)
    df["_wl"] = df["_w"] * df["loc"]

    g = df.groupby("region")
    share = g["_w"].sum()
    value = g["_wl"].sum() / share
    count_weight = df.groupby(["region", "bucket"])["_w"].sum()
    return {"share": share, "value": value, "n": int(len(df)), "count_weight": count_weight}


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

    When mu is loc_mu -- the zero point 14_pitcher_pages.py actually publishes,
    the mean of per-pitcher means -- this must come out to (approximately) zero.
    sum_c w*_c v*_c telescopes back to that same mean of per-pitcher means (see
    league_cell_table's docstring), so sum_c w*_c(v*_c - loc_mu) is loc_mu minus
    itself. A nonzero result here means the table's weighting and loc_mu's
    weighting have drifted apart again, not that D1 has a real "own location
    mix" effect: see test_location_baseline_is_zero_when_weighted_like_loc_mu.
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

    ONE ROW PER REGION, COUNT NESTED. This used to emit one row per (region,
    count bucket), median 22 rows per pitcher. See league_cell_table's
    docstring for why that was removed: the location map is pooled across
    counts, so the split re-sliced the same points into thinner rows and
    pushed real spots below min_share for no benefit. Collapsing to one row
    per region does not touch a single point total -- occ/plac/points are
    recomputed at the region grain the same way they were at the (region,
    count) grain, and the underlying algebra is exactly the telescoping sum
    league_cell_table's docstring describes, so the identity above is
    unaffected by the collapse.

    A coach who wants to know how a region's usage splits by count still can:
    each row carries `byCount`, a list of {count, n, share} nested under it,
    where `share` is that count's fraction of the region's OWN pitches (not of
    his whole arsenal). `byCount` is deliberately frequency-only -- no points.
    The model returns the same value for a spot regardless of the count it was
    thrown in, so "per-count points" would just be this row's `points`
    re-apportioned by frequency; showing that would look like the grade knows
    about counts, when it does not. If that changes (a count-aware location
    model), byCount is where a real per-count value belongs -- not before.

    Each byCount entry ALSO carries `leagueShare` when the league table has any
    weight in that (region, bucket) cell: D1's own count-bucket split of that
    same region, normalized the same way his `share` is (the three buckets of
    one region sum to 1.0 on both sides). It is omitted, not zero, for a bucket
    where the league table simply has no row -- see league_cell_table's
    count_weight docstring for why that is different from a real 0%. A bundle
    built before this field existed, or a byCount entry where the league truly
    never lands, both just render his own figure with nothing to compare it
    against.
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
    for region, g in sub.groupby("region"):
        his[region] = (len(g) / n, float(g["loc"].mean()), int(len(g)))

    def by_count(region, region_n, lg_w):
        """His own count-bucket frequency within one region: {count, n,
        share}, share as a fraction of region_n. No points -- see the
        docstring above. Empty when he never throws to the region at all.

        `lg_w` is this region's own leagueShare (already computed by `terms`),
        the denominator that normalizes count_weight into D1's own per-region
        count split -- see league_cell_table's count_weight docstring for why
        that has to be the region-level weight and not a re-lookup here.
        """
        if region_n == 0:
            return []
        counts = sub.loc[sub["region"] == region, "bucket"].value_counts()
        lg_shares = _league_by_count_share(table, region, lg_w)
        result = []
        for b in COUNT_BUCKET_ORDER:
            if b not in counts.index:
                continue
            entry = {"count": b, "n": int(counts[b]), "share": float(counts[b]) / region_n}
            if b in lg_shares:
                entry["leagueShare"] = lg_shares[b]
            result.append(entry)
        return result

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
    for region in sorted(set(his) | set(lg_share.index)):
        share, his_value, count = his.get(region, (0.0, float("nan"), 0))
        if share == 0.0:
            # He never throws here but D1 does. Real occupancy, no placement,
            # and no line of its own.
            his_value = float(lg_value.get(region, loc_mu))
        occ, plac, points, lg_w, lg_v = terms(region, share, his_value)
        row = {
            "region": region,
            "n": count,
            "share": float(share),
            "leagueShare": lg_w,
            "points": float(points),
            "occupancyPoints": float(occ),
            "placementPoints": float(plac),
            "value": float(his_value),
            "leagueValue": float(lg_v),
            "byCount": by_count(region, count, lg_w),
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
            # Combined from the pooled regions' own byCount breakdowns, rather
            # than left empty, so "Everywhere else" still answers the count
            # question -- a region only lands here because it is rare, not
            # because its count breakdown stopped mattering.
            "byCount": _pool_by_count(dropped),
        })
    return rows


def _league_by_count_share(table, region, region_lg_share):
    """D1's own count-bucket split of one region, normalized to that region's
    own leagueShare so the three buckets sum to 1.0 -- the league counterpart
    of `by_count`'s his-own-share. `region_lg_share` is the already-computed
    leagueShare for this region (0.0 when the league never lands there at
    all, or when the region isn't in the league table's index), never
    re-derived from the table, so this agrees exactly with the row's own
    leagueShare field rather than a second, possibly-KeyError-prone lookup.

    Returns a plain {bucket: fraction} dict rather than the list shape
    `by_count` builds, since callers merge it bucket-by-bucket into rows they
    already own.
    """
    if region_lg_share <= 0:
        return {}
    cw = table.get("count_weight")
    if cw is None:
        return {}
    out = {}
    for b in COUNT_BUCKET_ORDER:
        key = (region, b)
        if key in cw.index:
            out[b] = float(cw[key]) / region_lg_share
    return out


def _pool_by_count(dropped):
    """Sum the byCount breakdowns of every pooled region into one frequency
    table, keyed by count bucket rather than region. Combines both sides:
    his own n/share (raw pitch counts, summed directly) and D1's leagueShare
    (recovered as raw weight -- bc's per-region leagueShare times that row's
    own leagueShare -- summed across the pooled regions, then renormalized by
    the pooled row's total leagueShare). Renormalizing the raw weight sum
    rather than averaging the per-region fractions is required: the pooled
    regions carry different amounts of league weight, so an unweighted average
    of fractions would not equal the true combined split.
    """
    n_totals: dict[str, int] = {}
    lg_totals: dict[str, float] = {}
    for r in dropped:
        for bc in r["byCount"]:
            n_totals[bc["count"]] = n_totals.get(bc["count"], 0) + bc["n"]
            if "leagueShare" in bc:
                raw = bc["leagueShare"] * r["leagueShare"]
                lg_totals[bc["count"]] = lg_totals.get(bc["count"], 0.0) + raw
    total_n = sum(n_totals.values())
    total_lg = sum(r["leagueShare"] for r in dropped)
    if total_n == 0:
        return []
    result = []
    for b in COUNT_BUCKET_ORDER:
        if b not in n_totals:
            continue
        entry = {"count": b, "n": n_totals[b], "share": float(n_totals[b]) / total_n}
        if b in lg_totals and total_lg > 0:
            entry["leagueShare"] = float(lg_totals[b]) / total_lg
        result.append(entry)
    return result


def _pooled_mean(rows, value_key, weight_key, fallback_w):
    """Share-weighted mean of one side of the pooled row, falling back to the
    other side's weights when this side has no pitches in any pooled cell."""
    w = [r[weight_key] for r in rows]
    if sum(w) <= 0:
        w = [r[fallback_w] for r in rows]
    if sum(w) <= 0:
        return float("nan")
    return float(np.average([r[value_key] for r in rows], weights=w))
