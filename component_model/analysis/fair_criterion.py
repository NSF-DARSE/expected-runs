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

FEATS = ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak", "EffectiveVelo",
         "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
         "velocity_differential", "is_lhp", "is_lhb"]
FF_TYPES = {"Fastball", "FourSeamFastBall", "FourSeamFastball"}
USECOLS = ["PitchUID", "Date", "Pitcher", "PitcherId", "PitcherThrows", "PitcherTeam",
           "Batter", "BatterSide", "BatterTeam", "Balls", "Strikes",
           "TaggedPitchType", "PitchCall", "TaggedHitType", "ExitSpeed", "Angle",
           "Target", "SpinRate", "Extension", "HorzBreak", "InducedVertBreak",
           "EffectiveVelo", "RelHeight", "RelSide", "vertbreakdiff", "horzbreakdiff",
           "velocity_differential", "PlateLocSide", "PlateLocHeight", "League"]

RIDGE_ALPHA = 10
BATTER_K = 200
PANEL_MIN_FF = 100


def paths():
    """Resolve source CSV and working directory from CLI args or environment."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("STUFFPLUS_DATA"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--team", default="DEL_BLU")
    args, _ = ap.parse_known_args()
    if not args.data or not args.workdir:
        sys.exit("Set STUFFPLUS_DATA (source CSV) and STUFFPLUS_WORKDIR "
                 "(cache/output dir outside the repo), or pass --data/--workdir.")
    os.makedirs(args.workdir, exist_ok=True)
    return args


def load_pitches(args):
    """Source CSV -> deduped 2024/25 pitch frame, cached as parquet in workdir."""
    cache = os.path.join(args.workdir, "pitches_cache.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    df = pd.read_csv(args.data, usecols=USECOLS)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    df = df[df["year"].isin([2024, 2025])].copy()
    df["year"] = df["year"].astype(int)
    df["is_lhp"] = (df["PitcherThrows"] == "Left").astype(float)
    df["is_lhb"] = (df["BatterSide"] == "Left").astype(float)
    df["is_inplay"] = df["PitchCall"] == "InPlay"
    df["is_ff"] = df["TaggedPitchType"].isin(FF_TYPES)
    df.to_parquet(cache, index=False)
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

def stuff_ridge(df, return_model=False, pitch_mask=None):
    """Rows of one pitch type with complete features; adds ridge_pred
    (Ridge alpha=10, trained 2024). pitch_mask defaults to four-seams."""
    mask = df["is_ff"] if pitch_mask is None else pitch_mask
    ff = df[mask].dropna(subset=FEATS + ["Target"]).copy()
    train = ff[ff["year"] == 2024]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[FEATS].values, train["Target"].values)
    ff["ridge_pred"] = model.predict(ff[FEATS].values)
    return (ff, model) if return_model else ff


def panel_ids(ff, min_n=PANEL_MIN_FF):
    """Pitchers with min_n+ qualifying FF in both 2024 and 2025."""
    q = ff.groupby(["PitcherId", "year"]).size().unstack(fill_value=0)
    return q[(q.get(2024, 0) >= min_n) & (q.get(2025, 0) >= min_n)].index


def ff_panel(args):
    """Full chain with caching: slim FF pitch frame with xT, adjT, ridge_pred,
    plate location, count, and panel membership. This is what scripts 02-07 load."""
    cache = os.path.join(args.workdir, "ff_panel.parquet")
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
