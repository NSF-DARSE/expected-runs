"""Shared machinery for the Location+ / Stuff+ analysis suite.

Everything numerically load-bearing is defined here exactly once:
  - data loading (dedup on PitchUID, keep first) with a local parquet cache
  - xT: luck/defense-stripped expected run value (EV/LA map for balls in play)
  - adjT: opponent-adjusted xT (league means + batter effects shrunk toward league)
  - the fixed Stuff+ reference (Ridge alpha=10 on 12 physical features, trained 2024)
  - the (x,z) plate-location run-value maps, pooled and count-conditioned
  - the qualified pitcher panel (100+ four-seam FF in both 2024 and 2025)

The numbered scripts in this directory import from this module and must not
re-implement any of it. Anchor values for the current source data are recorded
in RESULTS.md; script 01 verifies them.

Data rules (licensed TrackMan, Level II): scripts read the source CSV from
STUFFPLUS_DATA (or --data) and cache/write only under STUFFPLUS_WORKDIR
(or --workdir), which must live outside the repository or stay gitignored.
Never commit the cache, any derived values, or per-pitcher output.

SIGN CONVENTIONS -- do not guess; an inverted trait narrative shipped once
already (see RESULTS.md, deployment section) before review caught it:
  - Target, xT, adjT, ridge_pred, and every location-map value are EXPECTED
    RUNS from the PITCHER's perspective: LOWER = BETTER for the pitcher.
  - A predictor is oriented consistently when it correlates POSITIVELY with a
    future run-value criterion. A raw trait predicts BETTER outcomes when its
    correlation with the criterion is NEGATIVE. Read every trait-screen sign
    against this before writing a word of interpretation.
  - Display scores negate into higher-is-better: X100 = 100 + 15 * z where
    z = -(value - mu) / sd. Convert frames ONLY at the display layer, never
    mid-analysis, and never mix frames in one table.
  - Whiff rate is the lone higher-is-better raw quantity; hard-hit rate is
    lower-is-better.
When defining any new trait, score, or composite, state its orientation in a
comment at the definition site.

NAMING -- what these quantities are CALLED, which the sign block above does not
cover and which has since produced its own repeat errors (three mislabels in one
session, each caught in review):
  - Target, xT and adjT are EXPECTED RUN VALUE, RELATIVE TO AN AVERAGE PITCHER.
    Never call any of them "runs", "actual runs", or "runs allowed". Each pitch
    is charged the change in run expectancy plus any runs that scored on it
    (Target = RunsScored + ER_next - ER), so a double with nobody on costs about
    +0.6 whether or not that runner ever scores.
  - Pool means sit near ZERO by construction. Writing "runs per 100" without
    "vs average" implies an absolute rate and is wrong.
  - RA9 is the ONLY literal runs-allowed quantity here; "actual runs" is correct
    for it and for nothing else.
  - The three differ only in how much is replaced by expectation: Target = the
    realized event valued by the RE table (luck included); xT = batted balls
    replaced by the EV/LA-map value; adjT = xT with league mean and a shrunk
    batter effect also removed. Distinguish Target as "unadjusted", not "actual".
  - Prefer "run value" in any user-facing text: accurate, standard, and it does
    not imply runs crossed the plate.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# RelSpeed (real release velocity) replaced EffectiveVelo on 2026-08-17 after the
# pre-registered gate passed (coach_extension_fix.py, n=826): EffectiveVelo is computed
# FROM release speed and extension, so conditioning on it blocked the path extension works
# through and left its coefficient at zero (P(more is better)=0.545). On RelSpeed the
# ridge learns the tradeoff itself: Extension established more-is-better at P=1.000 over
# 200 cluster-bootstrap refits, validity non-inferior (+0.002 +/- 0.003). EffectiveVelo
# stays in USECOLS: it remains the "Velocity only" display baseline (the radar-gun
# reference the models are measured against), just not a model input.
# HorzBreak_arm / RelSide_arm are ARM-SIDE MIRRORED copies (LHP flipped), created in
# stuff_ridge. The raw columns keep their names and values because downstream consumers
# need them raw -- above all the coach-card scorer, whose per-hand weights expect the
# native frame. Mirroring the raw columns in place corrupted his card's scores and turned
# the v1-v2 tie into a fake rout (caught 2026-08-17 before shipping).
# dev_relheight / dev_relside ADDED 2026-08-17 and are the COACH'S construct, adopted after
# it beat ours on the fixed criterion. His hand-built card scores release side and height as
# |value - typical for that hand|: a symmetric V, so an unusually high AND an unusually low
# release earn credit, where our terms were monotone. Gated over 200 cluster-bootstrap
# refits: the shipped set improves validity +0.0307 (adjT, CI [+0.0079,+0.0514], P=0.995)
# and +0.0406 (Target, P=1.000) over the previous 12-feature set. Our monotone RelHeight and
# RelSide_arm are KEPT alongside them -- adding his form beat swapping ours out, so the two
# carry different information. Centres are the fixed DEV_CENTRES constants below, never
# recomputed per frame.
#
# DROPPED 2026-08-17 for CONSTRUCT reasons, not predictive ones, and the distinction matters
# because the numbers alone would not justify it:
#   the three "vs his fastball" differentials -- their reference is the pitcher's own FASTEST
#     pitch type, so on a four-seam they collapse to within-pitcher scatter about his own
#     mean and average to ~0 per pitcher. Their non-inferiority test actually came in at
#     P=0.910 against a 0.95 bar, i.e. we could NOT show removing them was free. They are
#     dropped anyway: a feature that cannot be explained to a coach looking at a fastball
#     does not belong in the fastball model, and a small unprovable predictive cost is an
#     acceptable price for a feature list that means what it says. They stay in USECOLS for
#     the multi-pitch work, where the same columns become meaningful against a real second
#     pitch.
#   is_lhb -- opponent context inside a pitch-quality score, which credits a pitcher for the
#     batters he happened to face. Removing it also happened to help slightly (+0.004 on
#     both criteria, P(better)=1.000), so this one cost nothing.
FEATS = ["SpinRate", "Extension", "HorzBreak_arm", "InducedVertBreak", "RelSpeed",
         "RelHeight", "RelSide_arm", "is_lhp", "dev_relheight", "dev_relside"]

# Per-hand centres for the two deviation features, measured ONCE on the 2024 train year
# (coach_release_gate.py) and frozen here. Keys are is_lhp (0 = RHP, 1 = LHP), values are
# feet. These must NOT be recomputed from whatever frame is being scored: the score frame
# (2024/2025) and the criterion frame (2025/2026) load separately, and centring each on its
# own rows would silently give the two frames different features.
DEV_CENTRES = {
    "dev_relheight": {0: 5.781186546815588, 1: 5.7260303576615526},
    "dev_relside": {0: 1.6516521778868811, 1: 1.807404618575344},
}
DEV_SRC = {"dev_relheight": "RelHeight", "dev_relside": "RelSide_arm"}
# "FastBall" and "Four-Seam" added 2026-08-17: both appear as TrackMan tags in the D1
# extracts and were previously unrecognised, so those pitches were not counted as fastballs
# at all -- neither for the is_ff model filter nor for the differential anchor below.
FF_TYPES = {"Fastball", "FourSeamFastBall", "FourSeamFastball", "FastBall", "Four-Seam"}

# ---- the "vs primary fastball" anchor, recomputed at load (see add_fastball_diffs) ----
# Source column -> the differential column it feeds.
DIFF_COLS = {"InducedVertBreak": "vertbreakdiff", "HorzBreak": "horzbreakdiff",
             "RelSpeed": "velocity_differential"}
# A fallback anchor group must clear this many pitches before it can win on mean velocity,
# so a single mis-tagged 95mph "changeup" cannot become a pitcher's reference pitch.
ANCHOR_MIN_N = 5
# Columns written by add_fastball_diffs. Their presence is the cache-freshness marker: a
# parquet written before this function existed carries the OLD differentials under the same
# three names, and the column-presence check alone would happily serve them.
ANCHOR_COLS = ["anchor_type", "anchor_n"]

# ---------------- per-pitch-type models ----------------
# One model per pitch type, because the same measurement means different things on different
# pitches. Added 2026-08-17 after the coaching staff walked the four-seam page; every entry
# below traces to a decision recorded there, not to a search over feature sets.
#
# GROUPS pool the TrackMan tags that are one pitch for modelling purposes. Display naming is
# NOT changed by this: the dashboard still shows a pitcher the tag he was thrown.
PITCH_GROUPS = {
    "FF": set(FF_TYPES),
    # A sinker and a two-seam are the same pitch under two names.
    "SI": {"Sinker", "TwoSeamFastBall"},
    "SL": {"Slider"},
    "SW": {"Sweeper"},
    "CB": {"Curveball"},
    "FC": {"Cutter"},
    # Splitter is pooled with ChangeUp for TRAINING ONLY. Measured 2026-08-17 on D1: for the
    # 174 pitcher-seasons carrying both tags at 15+ each, the two are the same pitch for that
    # pitcher (velo apart 0.22 mph, IVB 0.83 in, HorzBreak 0.23 in -- 0.07, 0.18 and 0.02 of
    # the between-pitcher SD; mean Target +0.0032 vs +0.0033; which tag is slower flips 58/42).
    # Splitter alone has ZERO pitchers at 100+ in both seasons, so the alternative to pooling
    # is not a splitter model, it is no splitter grade at all. Dan has not yet ruled on the
    # taxonomy; if he separates them, delete "Splitter" here and splitters go ungraded until
    # the sample exists.
    "CH": {"ChangeUp", "Splitter"},
}

# The physical inputs every pitch type gets. This IS the shipped four-seam list (see the FEATS
# note above); FEATS stays bound to it so existing callers keep working unchanged.
BASE_FEATS = list(FEATS)
# "versus his own primary fastball" -- only meaningful for a pitch trying to look like the
# fastball and behave differently. Anchored by add_fastball_diffs.
DIFF_FEATS = ["vertbreakdiff", "horzbreakdiff", "velocity_differential"]

FEATS_BY_PITCH = {
    # No differentials: a four-seam IS the anchor, so its own differentials are ~0 by
    # construction, and they were dropped for interpretability in 114cae5.
    "FF": BASE_FEATS,
    # No differentials either, per Jack 2026-08-17: a sinker is a fastball, you want both
    # hard, and "slower than your fastball" is not a virtue you would coach into one. It
    # still gets its OWN model, because release height is expected to invert against the
    # four-seam -- a low slot means flat-to-the-top on a four-seam and steep-to-the-bottom
    # on a sinker, both good, for opposite reasons.
    "SI": BASE_FEATS,
    "SL": BASE_FEATS + DIFF_FEATS,
    "SW": BASE_FEATS + DIFF_FEATS,
    "CB": BASE_FEATS + DIFF_FEATS,
    "FC": BASE_FEATS + DIFF_FEATS,
    # SpinRate removed on Dan's reading, 2026-08-17: the term carries the wrong sign, and it
    # rewards high spin because spin correlates with break while the speed differential
    # already carries the mechanism. He wants LOW spin on a cambio. Keeping a feature that is
    # confidently backwards is worse than dropping a feature that is merely weak.
    "CH": [f for f in BASE_FEATS if f != "SpinRate"] + DIFF_FEATS,
}


def pitch_mask(df, group):
    """Rows of one PITCH_GROUPS key. Raises on an unknown key rather than silently empty."""
    if group not in PITCH_GROUPS:
        raise KeyError(f"unknown pitch group {group!r}; have {sorted(PITCH_GROUPS)}")
    return df["TaggedPitchType"].isin(PITCH_GROUPS[group])


def feats_for(group):
    if group not in FEATS_BY_PITCH:
        raise KeyError(f"no feature set for {group!r}; have {sorted(FEATS_BY_PITCH)}")
    return list(FEATS_BY_PITCH[group])
USECOLS = ["PitchUID", "Date", "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
           "Batter", "BatterSide", "BatterTeam", "Balls", "Strikes",
           "TaggedPitchType", "PitchCall", "TaggedHitType", "ExitSpeed", "Angle",
           "Target", "SpinRate", "Extension", "HorzBreak", "InducedVertBreak",
           "EffectiveVelo", "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
           "velocity_differential", "PlateLocSide", "PlateLocHeight", "League",
           "Level", "GameID", "Inning", "Top/Bottom", "PAofInning", "PitchofPA"]

# Read when the extract has them, skipped when it does not (see load_pitches).
OPTIONAL_COLS = ["RelSpeed"]

RIDGE_ALPHA = 10
BATTER_K = 200
PANEL_MIN_FF = 100


def paths():
    """Resolve source CSV and working directory from CLI args or environment."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("STUFFPLUS_DATA"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--team", default="DEL_BLU")
    ap.add_argument("--level", default=os.environ.get("STUFFPLUS_LEVEL"),
                    help="Optional Level filter (e.g. D1). Default: all levels, "
                         "matching the original 2024-2025 runs.")
    ap.add_argument("--years", default=os.environ.get("STUFFPLUS_YEARS", "2024,2025"),
                    help="Comma-separated train,eval year pair (default 2024,2025). "
                         "A non-default pair is ROLE-RELABELED at load: the earlier "
                         "year takes the 2024 (train) role and the later the 2025 "
                         "(eval) role, so every downstream script applies the same "
                         "method to the new pair unchanged. Downstream prints that "
                         "say 2024/2025 then mean train-year/eval-year.")
    args, _ = ap.parse_known_args()
    if not args.data or not args.workdir:
        sys.exit("Set STUFFPLUS_DATA (source CSV) and STUFFPLUS_WORKDIR "
                 "(cache/output dir outside the repo), or pass --data/--workdir.")
    args.year_pair = tuple(int(y) for y in args.years.split(","))
    if len(args.year_pair) != 2 or args.year_pair[0] >= args.year_pair[1]:
        sys.exit("--years must be two ascending years, e.g. 2025,2026")
    os.makedirs(args.workdir, exist_ok=True)
    return args


def _year_suffix(args):
    pair = getattr(args, "year_pair", (2024, 2025))
    tag = "" if pair == (2024, 2025) else f"_{pair[0]}_{pair[1]}"
    level = getattr(args, "level", None)
    if level:
        tag += f"_{level}"
    return tag


def load_pitches(args):
    """Source CSV -> deduped pitch frame for the year pair, cached as parquet.

    For a non-default pair the 'year' column is ROLE-RELABELED (earlier year ->
    2024, later -> 2025) so the numbered scripts' hardcoded train/eval years
    replicate the method on the new pair verbatim. The real years are printed
    loudly here and preserved in the cache filename.
    """
    pair = getattr(args, "year_pair", (2024, 2025))
    cache = os.path.join(args.workdir, f"pitches_cache{_year_suffix(args)}.parquet")
    # RelSpeed is a MODEL feature (FEATS) but optional at load so extracts that predate it
    # still open; stuff_ridge then fails loudly on the missing column rather than here. A
    # cache written before RelSpeed joined the read must be rebuilt, not served -- serving
    # it would surface as a KeyError far from the cause. (Backported from
    # real-velo-context, where the same column was display-only.)
    header = pd.read_csv(args.data, nrows=0).columns
    available = USECOLS + [c for c in OPTIONAL_COLS if c in header]
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        stale = [c for c in available if c not in cached.columns]
        # ANCHOR_COLS are the freshness marker for the recomputed differentials. Without this
        # a pre-2026-08-17 cache passes the column check and silently serves the OLD
        # single-fastest-pitch anchor under the same three column names.
        stale += [c for c in ANCHOR_COLS if c not in cached.columns]
        if not stale:
            return cached
        print(f"*** CACHE REBUILD: {cache} predates {', '.join(stale)} ***")
    df = pd.read_csv(args.data, usecols=available)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    df = df[df["year"].isin(pair)].copy()
    df["year"] = df["year"].astype(int)
    level = getattr(args, "level", None)
    if level:
        before = len(df)
        df = df[df["Level"] == level].copy()
        print(f"*** LEVEL FILTER: {level} keeps {len(df)}/{before} rows ***")
    if pair != (2024, 2025):
        print(f"*** YEAR ROLE RELABELING: {pair[0]} -> '2024' (train role), "
              f"{pair[1]} -> '2025' (eval role). All downstream 2024/2025 labels "
              f"mean train/eval year. ***")
        df["year"] = df["year"].map({pair[0]: 2024, pair[1]: 2025})
    df["is_lhp"] = (df["PitcherThrows"] == "Left").astype(float)
    df["is_lhb"] = (df["BatterSide"] == "Left").astype(float)
    df["is_inplay"] = df["PitchCall"] == "InPlay"
    df["is_ff"] = df["TaggedPitchType"].isin(FF_TYPES)
    df = add_fastball_diffs(df)
    df.to_parquet(cache, index=False)
    return df


def add_fastball_diffs(df):
    """Recompute the three 'vs primary fastball' differentials on a robust anchor.

    OVERRIDES vertbreakdiff / horzbreakdiff / velocity_differential as they arrive from the
    source CSV. target_and_calculated_pipeline.py builds them against `FastestPitchType`,
    which it defines as the tag containing the pitcher's SINGLE fastest pitch. Two things go
    wrong with that, both measured on 2026 D1 (2026-08-17):

      1. The anchor is tag-scoped, and 2436 of 5714 pitchers spread their fastballs over two
         or more tags ("Fastball" and "FourSeamFastBall" being the usual pair). One radar
         reading decides which tag wins, so for 409 pitchers the anchor was the MINORITY
         fastball tag, covering under half their fastballs for 19% of them. Every
         differential for every one of that pitcher's pitch types then measures against a
         partial, unrepresentative fastball.
      2. A max over noisy readings is not a robust statistic. 693 pitchers had a non-fastball
         anchor; sinker (580) and two-seam (70) are defensible, but 11 changeups, 8 sliders
         and a sweeper are not -- those are single hot readings or mis-tags.

    The fix pools ALL of FF_TYPES into one fastball group per pitcher-year and anchors on its
    mean. Only when a pitcher-year has no fastball at all does it fall back to the hardest
    other group by MEAN velocity, and that group must clear ANCHOR_MIN_N pitches first.

    TWO JUDGEMENT CALLS worth knowing about, both flagged to Jack 2026-08-17:
      - The anchor is per pitcher-YEAR, not per pitcher. The pipeline pooled both seasons,
        which mixes arsenals across a year in which a pitcher may have added or dropped a
        pitch. Within-season is the right reference for a within-season grade, and it also
        keeps the score frame and the criterion frame from sharing an anchor.
      - Sinkers and two-seams are NOT pooled into the fastball anchor, so "differential vs
        fastball" keeps meaning "vs the four-seam family". A sinker-primary pitcher with no
        four-seam still gets a sinker anchor through the fallback. Dan's "a sinker is a
        fastball" argues the other way and would change what the feature means for every
        off-speed pitch, so it is deliberately not done here.
    """
    src = list(DIFF_COLS)
    d = df[["PitcherId", "year", "TaggedPitchType"] + src].copy()
    # one pooled fastball group per pitcher-year; every other tag stays its own group
    d["_grp"] = np.where(d["TaggedPitchType"].isin(FF_TYPES), "_FF", d["TaggedPitchType"])
    g = d.groupby(["PitcherId", "year", "_grp"], dropna=False).agg(
        _ivb=("InducedVertBreak", "mean"), _hb=("HorzBreak", "mean"),
        _velo=("RelSpeed", "mean"), _n=("RelSpeed", "count")).reset_index()
    g = g[g["_n"] > 0]
    # priority: a real fastball group always wins; otherwise the hardest group that clears
    # ANCHOR_MIN_N; otherwise the hardest group at all. Sorting ascending and taking the last
    # row per pitcher-year applies that order, with mean velocity as the within-tier rank.
    g["_pri"] = np.where(g["_grp"] == "_FF", 2, np.where(g["_n"] >= ANCHOR_MIN_N, 1, 0))
    anchor = (g.sort_values(["_pri", "_velo"]).groupby(["PitcherId", "year"]).tail(1)
              .rename(columns={"_grp": "anchor_type", "_n": "anchor_n"}))
    # .to_numpy(), NOT a Series copy: the merge below hands back a fresh RangeIndex, while df
    # arrives here with the holes left by the dropna and level filters in load_pitches.
    # Subtracting the two Series then aligns on index, pairs up unrelated rows, and reports a
    # mean |change| of ~9.7 inches for what is really ~0.42. Positional comparison only.
    before = {c: df[c].to_numpy(copy=True) for c in DIFF_COLS.values() if c in df.columns}
    df = df.merge(anchor[["PitcherId", "year", "anchor_type", "anchor_n",
                          "_ivb", "_hb", "_velo"]], on=["PitcherId", "year"], how="left")
    for s, out, ref in (("InducedVertBreak", "vertbreakdiff", "_ivb"),
                        ("HorzBreak", "horzbreakdiff", "_hb"),
                        ("RelSpeed", "velocity_differential", "_velo")):
        df[out] = df[s] - df[ref]
    df = df.drop(columns=["_ivb", "_hb", "_velo"])
    moved = [f"{c} mean |change| {float(np.abs(df[c].to_numpy() - before[c]).mean()):.3f}"
             for c in before if c in df.columns]
    print(f"*** DIFFERENTIAL ANCHOR REBUILT: {(df['anchor_type'] == '_FF').mean():.1%} of "
          f"pitches anchored on a pooled fastball group; {'; '.join(moved)} ***")
    return df


# ---------------- xT: luck/defense-stripped expected run value ----------------

def _evla_grid(src, evb, lab):
    e = (np.floor(src["ExitSpeed"] / evb) * evb).astype(int)
    l = (np.floor(src["Angle"] / lab) * lab).astype(int)
    g = src.assign(e=e, l=l).groupby(["e", "l"])["Target"].agg(["mean", "count"])
    return g[g["count"] >= 50]["mean"]


def add_xt(df):
    """Adds xT in place: balls in play get the pooled EV/LA map value
    (fine 5mph x 10deg, coarse 10x20 fallback, hit-type fallback when EV/LA is
    missing); every other pitch keeps its realized Target."""
    ip = df["is_inplay"]
    has = df["ExitSpeed"].notna() & df["Angle"].notna()
    src = df[ip & has & df["Target"].notna()]
    overall = src["Target"].mean()
    fine, coarse = _evla_grid(src, 5, 10), _evla_grid(src, 10, 20)
    m1 = ip & has
    sub = df.loc[m1]
    fe = (np.floor(sub["ExitSpeed"] / 5) * 5).astype(int)
    fa = (np.floor(sub["Angle"] / 10) * 10).astype(int)
    v = pd.Series(list(zip(fe, fa)), index=sub.index).map(fine)
    ce = (np.floor(sub["ExitSpeed"] / 10) * 10).astype(int)
    ca = (np.floor(sub["Angle"] / 20) * 20).astype(int)
    v2 = pd.Series(list(zip(ce, ca)), index=sub.index).map(coarse)
    df["xT"] = np.nan
    df.loc[m1, "xT"] = v.fillna(v2).fillna(overall)
    htype = df.loc[m1].groupby("TaggedHitType", observed=True)["xT"].mean()
    m2 = ip & ~has
    df.loc[m2, "xT"] = df.loc[m2, "TaggedHitType"].map(htype).fillna(overall).values
    df.loc[~ip, "xT"] = df.loc[~ip, "Target"]
    return df


def add_adjusted(df, K=BATTER_K):
    """Adds adjT in place: xT minus league mean and shrunk batter effect,
    recentered on the grand mean. Leagues under 5000 rows fall back to grand mean."""
    val = df["xT"].notna()
    grand = df.loc[val, "xT"].mean()
    counts = df.loc[val, "League"].value_counts()
    good = set(counts[counts >= 5000].index)
    lg = df.loc[val & df["League"].isin(good)].groupby("League", observed=True)["xT"].mean()
    df["league_mean"] = df["League"].map(lg).fillna(grand)
    df["batter_key"] = np.where(df["Batter"].notna(),
                                df["Batter"].astype(str) + "|" + df["BatterTeam"].astype(str),
                                np.nan)
    bsrc = df[val & df["batter_key"].notna()]
    bg = bsrc.groupby("batter_key")
    st = pd.DataFrame({"n": bg.size(), "mean_xt": bg["xT"].mean(),
                       "mean_lg": bg["league_mean"].mean()})
    eff = (st["n"] / (st["n"] + K)) * (st["mean_xt"] - st["mean_lg"])
    df["adjT"] = df["xT"] - (df["league_mean"] + df["batter_key"].map(eff).fillna(0.0)) + grand
    return df


# ---------------- fixed Stuff+ reference ----------------

def stuff_ridge(df, return_model=False, pitch_mask=None, feats=None):
    """Rows of one pitch type with complete features; adds ridge_pred
    (Ridge alpha=10, trained 2024).

    pitch_mask defaults to four-seams and feats to FEATS, so every pre-2026-08-17
    caller gets exactly the four-seam model it asked for. For any other pitch type
    pass BOTH, from pitch_mask()/feats_for() -- passing a mask without its feature
    list would grade, say, changeups on the four-seam feature set.
    """
    mask = df["is_ff"] if pitch_mask is None else pitch_mask
    feats = list(FEATS) if feats is None else list(feats)
    ff = df[mask].copy()
    # Arm-side frame for the two handedness-mirrored geometry features (see FEATS note):
    # one estimable slope per feature instead of a pooled average over two opposite
    # relationships (RelSide: RHP -0.0019 vs LHP +0.0034, P=1.000 they differ; HorzBreak:
    # RHP mean +10.6 vs LHP -11.4, two separated modes). New columns, raw left intact.
    ff["RelSide_arm"] = ff["RelSide"] * (1 - 2 * ff["is_lhp"])
    ff["HorzBreak_arm"] = ff["HorzBreak"] * (1 - 2 * ff["is_lhp"])
    # The coach's deviation-from-typical release terms (see FEATS note). Must come AFTER
    # RelSide_arm exists, since dev_relside is centred in the arm-side frame, and BEFORE the
    # dropna so rows missing a source column are cut once on the final feature list.
    for out, src in DEV_SRC.items():
        ff[out] = (ff[src] - ff["is_lhp"].map(DEV_CENTRES[out])).abs()
    ff = ff.dropna(subset=feats + ["Target"])
    train = ff[ff["year"] == 2024]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[feats].values, train["Target"].values)
    ff["ridge_pred"] = model.predict(ff[feats].values)
    return (ff, model) if return_model else ff


def panel_ids(ff, min_n=PANEL_MIN_FF):
    """Pitchers with min_n+ qualifying FF in both 2024 and 2025."""
    q = ff.groupby(["PitcherId", "year"]).size().unstack(fill_value=0)
    return q[(q.get(2024, 0) >= min_n) & (q.get(2025, 0) >= min_n)].index


def ff_panel(args):
    """Full chain with caching: slim FF pitch frame with xT, adjT, ridge_pred,
    plate location, count, and panel membership. This is what scripts 02-07 load."""
    cache = os.path.join(args.workdir, f"ff_panel{_year_suffix(args)}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = load_pitches(args)
    add_xt(df)
    add_adjusted(df)
    ff = stuff_ridge(df)
    ids = panel_ids(ff)
    keep = ["PitchUID", "PitcherId", "Pitcher", "PitcherTeam", "PitcherThrows", "year",
            "Balls", "Strikes", "PlateLocSide", "PlateLocHeight", "xT", "adjT",
            "ridge_pred", "is_lhb", "PitchCall", "Target"]
    slim = ff[keep].copy()
    slim["in_panel"] = slim["PitcherId"].isin(ids)
    slim.to_parquet(cache, index=False)
    return slim


# ---------------- plate-location run-value maps ----------------

def add_loc_bins(df):
    df["gx"] = (np.floor(df["PlateLocSide"] / 0.25) * 0.25).round(3)
    df["gz"] = (np.floor(df["PlateLocHeight"] / 0.25) * 0.25).round(3)
    df["cx"] = (np.floor(df["PlateLocSide"] / 0.5) * 0.5).round(3)
    df["cz"] = (np.floor(df["PlateLocHeight"] / 0.5) * 0.5).round(3)
    return df


class PooledLocationMap:
    """(x,z) -> xT, 0.25ft bins (min 50), 0.5ft fallback, overall-mean fallback."""

    def __init__(self, train):
        f = train.groupby(["gx", "gz"])["xT"].agg(["mean", "count"])
        self.fine = f[f["count"] >= 50]["mean"]
        c = train.groupby(["cx", "cz"])["xT"].agg(["mean", "count"])
        self.coarse = c[c["count"] >= 50]["mean"]
        self.fallback = train["xT"].mean()

    def apply(self, sub):
        v = pd.Series(list(zip(sub["gx"], sub["gz"])), index=sub.index).map(self.fine)
        v2 = pd.Series(list(zip(sub["cx"], sub["cz"])), index=sub.index).map(self.coarse)
        return v.fillna(v2).fillna(self.fallback)


class CountLocationMap:
    """Count-conditioned map: each (count, fine-cell) mean is shrunk toward the
    pooled map value at that location, v = (n*mean_cg + m*pooled) / (n+m).
    scheme_col is a column holding the count representation (e.g. '0-2')."""

    def __init__(self, train, scheme_col, m):
        self.pooled = PooledLocationMap(train)
        self.g = train.groupby([scheme_col, "gx", "gz"])["xT"].agg(["mean", "count"])
        self.count_means = train.groupby(scheme_col)["xT"].mean()
        self.overall = train["xT"].mean()
        self.scheme_col = scheme_col
        self.m = m

    def apply(self, sub, pooled_vals=None):
        if pooled_vals is None:
            pooled_vals = self.pooled.apply(sub)
        key = pd.MultiIndex.from_arrays([sub[self.scheme_col], sub["gx"], sub["gz"]])
        mean_cg = pd.Series(self.g["mean"].reindex(key).values, index=sub.index).fillna(0.0)
        n_cg = pd.Series(self.g["count"].reindex(key).values, index=sub.index).fillna(0.0)
        return (n_cg * mean_cg + self.m * pooled_vals) / (n_cg + self.m)

    def apply_relative(self, sub, pooled_vals=None):
        """Location-given-count only: subtract the count baseline E[xT|count].
        This is the variant safe to aggregate into a pitcher-level Location+."""
        raw = self.apply(sub, pooled_vals)
        return raw - sub[self.scheme_col].map(self.count_means).fillna(self.overall) + self.overall


def add_count_cols(df):
    """Legal-count clip plus the three count representations tested."""
    df["b"] = df["Balls"].clip(0, 3).astype(int)
    df["s"] = df["Strikes"].clip(0, 2).astype(int)
    df["count12"] = df["b"].astype(str) + "-" + df["s"].astype(str)
    df["bucket4"] = np.where(df["s"] == 2, "2K", np.where(df["b"] > df["s"], "behind",
                    np.where(df["s"] > df["b"], "ahead", "even")))
    df["state5"] = np.where(df["s"] == 2, "2K", np.where(df["b"] == 3, "3B",
                   np.where(df["b"] > df["s"], "behind",
                   np.where(df["s"] > df["b"], "ahead", "even"))))
    return df


# ---------------- evaluation helpers ----------------

def R(u, v):
    return pearsonr(u, v)[0]


def RS(u, v):
    return spearmanr(u, v)[0]


def z(s):
    return (s - s.mean()) / s.std()


def year_split(tab, ids):
    """Pitcher-year table -> aligned (2024, 2025) frames over the panel."""
    w24 = tab[tab["year"] == 2024].set_index("PitcherId").loc[ids]
    w25 = tab[tab["year"] == 2025].set_index("PitcherId").loc[ids]
    return w24, w25


def boot_report(name, d):
    d = np.asarray(d)
    print(f"  {name:<44} mean={d.mean():+.3f}  SE={d.std():.3f}  "
          f"95% CI=[{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]  "
          f"P(d>0)={(d > 0).mean():.3f}")
