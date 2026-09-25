"""Weighted, frame-aware plate-location value maps and empirical-Bayes shrinkage.

WHY THIS EXISTS. fair_criterion.PooledLocationMap and CountLocationMap are the shipped
four-seam Location+ machinery. Two things they cannot do are needed to put a SECONDARY-pitch
Location+ through the frozen incremental gate (coach_location_gate.py):

  1. Frequency weights. The gate is a cluster bootstrap that refits the model on resampled
     TRAIN pitchers by weighting their rows. A groupby map has no weight argument, so it
     cannot be refit the way the Stuff+ ridge is. CellMap takes a per-row weight and treats it
     as a frequency: integer weights give exactly the map of the duplicated rows (tested).
  2. A batter-relative, platoon-split frame. A slider's location value depends on which side
     of the plate is the batter's and whether he hits from the pitcher's side. The catcher's
     frame pools a right-hander's back-foot slider to a lefty with a right-hander's backdoor
     slider to a righty. The "batter_platoon" frame mirrors x so positive is always toward the
     batter (inside) and fits separate surfaces for same-side and opposite-side batters.

Everything is done with integer cell codes and np.bincount, so a refit costs milliseconds and
a 200-refit bootstrap over several fold maps is affordable.

FRAMES
  catcher         x = PlateLocSide as recorded, one surface. With count=False and unit weights
                  this reproduces fair_criterion.PooledLocationMap exactly (tested): same
                  0.25 ft cells, 50-pitch minimum, 0.5 ft fallback, overall-mean fallback.
  batter_platoon  x_in = PlateLocSide for a right-handed batter and -PlateLocSide for a
                  left-handed one, so POSITIVE x_in IS INSIDE for every batter. Orientation
                  is from data (coach_location_map.py): positive PlateLocSide is the
                  right-handed batter's side, from hit-by-pitch locations. platoon = 1 when
                  pitcher and batter throw/hit from the same side. Rows with a batter side or
                  pitcher hand other than Left/Right are not graded (switch pitchers, blanks).

COUNT-RELATIVE VALUE (one construct per score, FRAMEWORK.md). With count=True a pitch's value
is E[xT | location, count, platoon] and the pitcher-level quantity is that value MINUS the
context baseline E[xT | count, platoon], so living in favourable counts or facing more
same-side batters earns nothing. The count-cell mean is shrunk toward
pooled(location) + count_effect with prior weight m, an ADDITIVE prior. The shipped
CountLocationMap shrinks toward pooled(location) alone; subtracting the count baseline from
that re-introduces occupancy with the opposite sign wherever a count-cell is sparse. With the
additive prior a sparse count-cell's relative value is just the pooled location value minus
its platoon mean, which carries no count information at all (tested).

SIGN CONVENTION: every value here is expected run value from the PITCHER's perspective,
LOWER = BETTER, in the same frame as xT. Relative values are centred near zero; a negative
relative value is a location that is better than average for its count and platoon. Nothing
is negated in this module.

EMPIRICAL-BAYES SHRINKAGE. A pitcher's season mean of per-pitch values from n pitches has
sampling variance s2 / n around his true mean, which varies across pitchers with variance
tau2 around a prior mean. The posterior mean is prior + n / (n + s2 / tau2) * (mean - prior).
eb_moments estimates s2 (pooled within-pitcher variance) and tau2 (between-pitcher variance of
means minus the average sampling variance) by method of moments. Nothing here sees an outcome
the grade is later scored against: the inputs are location values only.

Data rules: pure functions over arrays; no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_CELL = 50          # PooledLocationMap's minimum pitches per cell
FINE = 0.25            # ft
COARSE = 0.5           # ft
N_COUNTS = 12          # legal counts 0-0 .. 3-2
FRAMES = ("catcher", "batter_platoon")
TAU2_FLOOR_FRAC = 0.01  # pre-registered floor on tau2, as a fraction of var(pitcher means)


# ---------------- frame and cell codes ----------------

def frame_columns(df: pd.DataFrame, frame: str) -> pd.DataFrame:
    """Per-row x, z, platoon, count index and a validity flag for one frame.

    Returns a frame aligned to df.index with columns x, z, p (int platoon code, 0 in the
    catcher frame), k (0..11 count index, balls*3 + strikes) and ok (row can be graded).
    """
    if frame not in FRAMES:
        raise ValueError(f"unknown frame {frame!r}; have {FRAMES}")
    x = df["PlateLocSide"].astype(float)
    z = df["PlateLocHeight"].astype(float)
    b = df["Balls"].clip(0, 3)
    s = df["Strikes"].clip(0, 2)
    ok = x.notna() & z.notna() & b.notna() & s.notna()
    k = (b.fillna(0).astype(int) * 3 + s.fillna(0).astype(int))
    if frame == "catcher":
        p = pd.Series(0, index=df.index)
    else:
        bs, pt = df["BatterSide"], df["PitcherThrows"]
        ok &= bs.isin(["Left", "Right"]) & pt.isin(["Left", "Right"])
        # POSITIVE x IS INSIDE: PlateLocSide > 0 is the right-handed batter's side.
        x = np.where(bs == "Left", -x, x)
        x = pd.Series(x, index=df.index)
        p = (bs == pt).astype(int)
    return pd.DataFrame({"x": x, "z": z, "p": p.astype(int), "k": k.astype(int), "ok": ok},
                        index=df.index)


def cell_codes(fr: pd.DataFrame) -> dict:
    """Integer codes for every key a CellMap uses, factorised over the rows given.

    Codes are only meaningful within one call, so compute them ONCE over every row that will
    be fitted or scored (train and eval together) and slice afterwards.
    """
    ix = np.floor(fr["x"].values / FINE).astype(np.int64)
    iz = np.floor(fr["z"].values / FINE).astype(np.int64)
    cx = np.floor(fr["x"].values / COARSE).astype(np.int64)
    cz = np.floor(fr["z"].values / COARSE).astype(np.int64)
    p = fr["p"].values.astype(np.int64)
    k = fr["k"].values.astype(np.int64)
    # Offsets keep every component non-negative before packing; plate coordinates are
    # bounded well inside +-2^20 cells.
    off = 1 << 20
    fine = (p << 42) | ((ix + off) << 21) | (iz + off)
    coarse = (p << 42) | ((cx + off) << 21) | (cz + off)
    pk = p * N_COUNTS + k
    pkf = (pk << 42) | ((ix + off) << 21) | (iz + off)
    out = {}
    for name, key in (("fine", fine), ("coarse", coarse), ("pkf", pkf)):
        codes, uniq = pd.factorize(key)
        out[name] = codes.astype(np.int64)
        out["n_" + name] = len(uniq)
    out["p"] = p
    out["n_p"] = int(p.max()) + 1 if len(p) else 1
    out["pk"] = pk
    out["n_pk"] = out["n_p"] * N_COUNTS
    return out


def subset_codes(codes: dict, idx) -> dict:
    """Row subset of a cell_codes dict (sizes kept, so bincount lengths still line up)."""
    return {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in codes.items()}


# ---------------- the map ----------------

class CellMap:
    """Frequency-weighted location value map over precomputed cell codes.

    Fit on train rows (codes, y, w); score any rows coded in the same cell_codes call.
    pooled value: fine-cell mean if its weight >= MIN_CELL, else coarse-cell mean if >=
    MIN_CELL, else the platoon's overall mean. count=True adds the count-conditioned value
    and the count-relative value (see module docstring).
    """

    def __init__(self, codes, y, w=None, count=False, m=5.0):
        y = np.asarray(y, dtype=float)
        w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
        self.count, self.m = count, float(m)
        wy = w * y
        nf = np.bincount(codes["fine"], w, codes["n_fine"])
        sf = np.bincount(codes["fine"], wy, codes["n_fine"])
        nc = np.bincount(codes["coarse"], w, codes["n_coarse"])
        sc = np.bincount(codes["coarse"], wy, codes["n_coarse"])
        npl = np.bincount(codes["p"], w, codes["n_p"])
        spl = np.bincount(codes["p"], wy, codes["n_p"])
        with np.errstate(invalid="ignore", divide="ignore"):
            self.fine_mean = np.where(nf >= MIN_CELL, sf / np.where(nf > 0, nf, 1), np.nan)
            self.coarse_mean = np.where(nc >= MIN_CELL, sc / np.where(nc > 0, nc, 1), np.nan)
            overall = sw_div(spl, npl)
        # A platoon with no train rows falls back to the grand mean.
        grand = wy.sum() / w.sum() if w.sum() > 0 else 0.0
        self.p_mean = np.where(np.isfinite(overall), overall, grand)
        if count:
            npk = np.bincount(codes["pk"], w, codes["n_pk"])
            spk = np.bincount(codes["pk"], wy, codes["n_pk"])
            pk_mean = sw_div(spk, npk)
            p_of_pk = np.arange(codes["n_pk"]) // N_COUNTS
            base = self.p_mean[np.minimum(p_of_pk, len(self.p_mean) - 1)]
            self.pk_mean = np.where(np.isfinite(pk_mean), pk_mean, base)
            self.n_pkf = np.bincount(codes["pkf"], w, codes["n_pkf"])
            self.s_pkf = np.bincount(codes["pkf"], wy, codes["n_pkf"])

    def pooled(self, codes):
        v = self.fine_mean[codes["fine"]]
        c = self.coarse_mean[codes["coarse"]]
        v = np.where(np.isnan(v), c, v)
        return np.where(np.isnan(v), self.p_mean[codes["p"]], v)

    def value(self, codes):
        """E[xT | location (, count), platoon] per row, lower = better for the pitcher."""
        pooled = self.pooled(codes)
        if not self.count:
            return pooled
        base = self.pk_mean[codes["pk"]]
        prior = pooled + (base - self.p_mean[codes["p"]])
        n = self.n_pkf[codes["pkf"]]
        s = self.s_pkf[codes["pkf"]]
        return (s + self.m * prior) / (n + self.m)

    def relative(self, codes):
        """Value minus the context baseline: count-relative with count=True, else the value
        minus the platoon mean (so a catcher-frame map's relative value is centred, not
        re-ordered: it is the value minus one constant)."""
        v = self.value(codes)
        base = self.pk_mean[codes["pk"]] if self.count else self.p_mean[codes["p"]]
        return v - base


def sw_div(s, n):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.where(n > 0, n, 1), np.nan)


def tune_m(codes, y, pitcher, ms=(1, 2, 5, 10, 25, 100)):
    """Script-09 holdout: alternate each pitcher's pitches into halves A/B, fit the count map
    on A, return the m minimising squared error on B's xT, and the per-m errors. Uses the
    training rows' own xT only; no criterion."""
    y = np.asarray(y, dtype=float)
    half = pd.Series(np.asarray(pitcher)).groupby(np.asarray(pitcher)).cumcount().values % 2
    a, b = np.where(half == 0)[0], np.where(half == 1)[0]
    ca, cb = subset_codes(codes, a), subset_codes(codes, b)
    errs = {}
    for m in ms:
        mp = CellMap(ca, y[a], count=True, m=m)
        errs[m] = float(np.mean((y[b] - mp.value(cb)) ** 2))
    return min(errs, key=errs.get), errs


# ---------------- empirical-Bayes shrinkage ----------------

def eb_moments(values, groups, min_n=1):
    """(s2, tau2, table) from per-pitch values grouped by pitcher.

    s2    pooled within-pitcher variance (weighted by each pitcher's pitches)
    tau2  var(pitcher means) - mean(s2 / n), over pitchers with n >= min_n, floored at
          TAU2_FLOOR_FRAC * var(pitcher means) so the shrinkage weight stays defined
    table per-pitcher n and mean (all pitchers, not only those used for the moments)
    """
    d = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(groups)}).dropna()
    grp = d.groupby("g")["v"]
    tab = pd.DataFrame({"n": grp.size(), "mean": grp.mean()})
    dev2 = (d["v"] - grp.transform("mean")) ** 2
    ss = dev2.groupby(d["g"]).sum()
    dof = (tab["n"] - 1).clip(lower=0)
    s2 = float(ss.sum() / dof.sum()) if dof.sum() > 0 else 0.0
    use = tab[tab["n"] >= min_n]
    return s2, tau2_between(use["mean"].values, use["n"].values, s2), tab


def tau2_between(means, n, s2, prior=None):
    """Between-pitcher variance of true means around `prior` (default: their own mean):
    var(means - prior) - mean(s2 / n), floored at TAU2_FLOOR_FRAC * var(means - prior)."""
    means, n = np.asarray(means, dtype=float), np.asarray(n, dtype=float)
    if len(means) < 2:
        return 0.0
    dev = means if prior is None else means - np.asarray(prior, dtype=float)
    vb = float(np.var(dev, ddof=1))
    if vb <= 0:
        return 0.0
    return max(vb - float(np.mean(s2 / n)), TAU2_FLOOR_FRAC * vb)


def eb_shrink(mean, n, prior, s2, tau2):
    """Posterior mean prior + n/(n+k) * (mean - prior), k = s2 / tau2. tau2 <= 0 gives the
    prior (no between-pitcher signal to keep)."""
    mean, n, prior = (np.asarray(a, dtype=float) for a in (mean, n, prior))
    if tau2 <= 0:
        return prior.copy()
    w = n / (n + s2 / tau2)
    return prior + w * (mean - prior)


def hier_prior(L, O, has_o, mu):
    """Prior mean for a pitcher's type grade from his other-pitch location score.

    Fits L = a + b * (O - mean O) by OLS over the pitchers that have O (has_o); pitchers
    without it get mu. Returns (prior array, b). O is standardised location value on the
    pitcher's OTHER pitch types, so b is how much of his command carries across pitches.
    Noise in O attenuates b toward zero, which errs toward less borrowing.
    """
    L, O = np.asarray(L, dtype=float), np.asarray(O, dtype=float)
    has_o = np.asarray(has_o, dtype=bool)
    prior = np.full(len(L), float(mu))
    if has_o.sum() < 3:
        return prior, 0.0
    o = O[has_o] - O[has_o].mean()
    a = float(L[has_o].mean())
    denom = float(np.dot(o, o))
    b = float(np.dot(o, L[has_o] - a) / denom) if denom > 0 else 0.0
    prior[has_o] = a + b * o
    return prior, b
