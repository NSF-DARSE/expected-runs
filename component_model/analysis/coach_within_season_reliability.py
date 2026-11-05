"""How many four-seams until each measure is STABLE within a season? Companion panel to
coach_sample_curve.py, which asks how well a measure PREDICTS next season.

The two questions differ. Within-season reliability is capped only by sampling noise, so it
climbs toward 1 with more pitches. Next-season prediction is also capped by year-to-year
drift (pitchers really change), which more pitches in one season cannot remove.

MODEL. reliability(n) = n / (n + n0), n0 = noise_per_pitch / signal_variance, where
  noise_per_pitch  game-clustered split-half scale (vc.effective_noise_scale's estimator:
                   n * (mean_A - mean_B)^2 / 4, games alternated into halves, averaged over
                   pitcher-seasons with min_half+ pitches in each half)
  signal_variance  season-centered variance of pitcher-season means minus the noise term
                   (rc.signal_variance: stable + drift, i.e. everything that is real within
                   a season)
on the variance-components panel of script 12: D1 four-seams 2024-2026, pitcher-seasons with
PANEL_MIN_FF+, pitchers with 2+ qualified seasons. adjT is the adopted criterion
(adjT_global since PR #38). Stabilization point = n0 (reliability 0.5). CIs: pitcher
cluster bootstrap (whole careers resampled together).

COMPOSITES. For fixed weights a blend of component means is the mean of a per-pitch
composite, so the same estimator applies. Both estimators are quadratic forms in the weight
vector, so each bootstrap draw stores three 3x3 matrices and any weighting is evaluated
exactly from them.
  Pitching+ (equal weights)   script 08: thirds of each component z'd by the SD of pitcher
                              means over 2025 pitchers with 100+ four-seams.
  Pitching+ (sample-size aware) PR #35: component weights scaled by k_c(n), which depend on n,
                              so reliability is evaluated at each n with that n's weights.
                              k_c uses PR #35's n0 (2025 graded season only). No criterion
                              enters any weight.

EMPIRICAL CHECK. Pitcher-seasons from 2024 and 2025 (calendar years), all D1 arms with
enough four-seams: games alternated into halves, the first h pitches (chronological) of each
half kept, half means correlated across pitcher-seasons (season-centered), Spearman-Brown to
2h. Plotted at n = 2h against the model curve. These are higher-volume pitchers than the
panel, which is stated, not hidden.

SIGN CONVENTION: all components are run value from the pitcher's view, LOWER = better.
Reliability is sign-free.

Data rules: reads the D1 pitch caches only (STUFFPLUS_WORKDIR, STUFFPLUS_WORKDIR_CRIT), writes
one JSON of aggregates to --out. No names, no per-pitcher output. Never commit output.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd

import fair_criterion as fc
import variance_components as vc
from coach_sample_curve import k_factor, N_REF

s12 = importlib.import_module("12_reliability_decomposition")
sf = importlib.import_module("16_season_floor")

COMPS = [("ridge", "ridge_pred"), ("loc", "loc"), ("adj", "adjT")]
MEASURES = {"stuff": [1, 0, 0], "loc": [0, 1, 0], "results": [0, 0, 1]}
LABELS = {"stuff": "Stuff+", "loc": "Location+", "results": "Adjusted run value",
          "pitch": "Pitching+ (equal weights)", "naware": "Pitching+ (sample-size aware)"}
MIN_HALF = 25        # script 12's default
N_GRID = list(range(5, 501, 5))
CHECK_H = [50, 100, 200]
CHECK_YEARS = [2024, 2025]
N_BOOT = 1000


def season_table(ff):
    """One row per pitcher-season: n, full means, and per-half means/counts of each
    component. Halves are script 12's game parity."""
    cols = [c for _, c in COMPS]
    g = ff.groupby(["PitcherId", "year"])
    full = g[cols].mean()
    full["n"] = g.size()
    h = ff.groupby(["PitcherId", "year", "half"])[cols].agg(["mean", "size"])
    out = full.copy()
    for k in (0, 1):
        hk = h.xs(k, level="half")
        for c in cols:
            out[f"{c}_h{k}"] = hk[(c, "mean")]
        out[f"n_h{k}"] = hk[(cols[0], "size")]
    return out.reset_index()


def moments(t, rows_noise):
    """3x3 matrices so that for weights w: noise = w'Nw, centered var of means = w'Cw."""
    cols = [c for _, c in COMPS]
    M = t[cols].to_numpy()
    M = M - t.groupby("year")[cols].transform("mean").to_numpy()
    C = np.cov(M, rowvar=False, ddof=1)
    tn = t[rows_noise]
    D = tn[[f"{c}_h0" for c in cols]].to_numpy() - tn[[f"{c}_h1" for c in cols]].to_numpy()
    n = (tn["n_h0"] + tn["n_h1"]).to_numpy()
    N = (D * (n / 4.0)[:, None]).T @ D / len(D)
    return N, C, float((1.0 / t["n"]).mean())


def n0_of(w, N, C, inv_n):
    w = np.asarray(w, dtype=float)
    noise = float(w @ N @ w)
    sig = max(1e-12, float(w @ C @ w) - noise * inv_n)
    return noise / sig


def curves(N, C, inv_n, sd, n0_pr35):
    out = {}
    for m, w in MEASURES.items():
        n0 = n0_of(w, N, C, inv_n)
        out[m] = (n0, np.array(N_GRID) / (np.array(N_GRID) + n0))
    base = np.array([1 / sd[c] for c, _ in COMPS]) / 3
    n0 = n0_of(base, N, C, inv_n)
    out["pitch"] = (n0, np.array(N_GRID) / (np.array(N_GRID) + n0))
    rel = []
    for n in N_GRID:
        k = np.array([float(k_factor(n, n0_pr35[c])) for c, _ in COMPS])
        rel.append(n / (n + n0_of(base * k, N, C, inv_n)))
    rel = np.array(rel)
    # stabilization: first n on the grid where reliability reaches 0.5, interpolated
    i = int(np.argmax(rel >= 0.5)) if (rel >= 0.5).any() else None
    if i is None:
        n50 = float("inf")
    elif i == 0:
        n50 = float(N_GRID[0])
    else:
        t = (0.5 - rel[i - 1]) / (rel[i] - rel[i - 1])
        n50 = N_GRID[i - 1] + t * (N_GRID[i] - N_GRID[i - 1])
    out["naware"] = (n50, rel)
    return out


def empirical(ff, sd, n0_pr35, rng):
    """Split-half check at h pitches per half, Spearman-Brown to 2h."""
    cols = [c for _, c in COMPS]
    f = ff[ff["year"].isin(CHECK_YEARS)].sort_values(
        ["PitcherId", "year", "Date", "GameID", "Inning", "PAofInning", "PitchofPA"])
    f = s12.add_game_parity(f.drop(columns=["half"], errors="ignore"))
    f = f.sort_values(["PitcherId", "year", "Date", "GameID", "Inning", "PAofInning",
                       "PitchofPA"])
    f["k"] = f.groupby(["PitcherId", "year", "half"]).cumcount()
    base = np.array([1 / sd[c] for c, _ in COMPS]) / 3
    res = {}
    for h in CHECK_H:
        sub = f[f["k"] < h]
        m = sub.groupby(["PitcherId", "year", "half"])[cols].agg(["mean"])
        cnt = sub.groupby(["PitcherId", "year", "half"]).size().unstack("half")
        ok = cnt[(cnt[0] >= h) & (cnt[1] >= h)].index
        A = m.xs(0, level="half").loc[ok]
        B = m.xs(1, level="half").loc[ok]
        yrs = A.index.get_level_values("year")
        A = (A - A.groupby(yrs).transform("mean")).to_numpy()
        B = (B - B.groupby(yrs).transform("mean")).to_numpy()
        pids = np.asarray(ok.get_level_values("PitcherId"))
        k = np.array([float(k_factor(2 * h, n0_pr35[c])) for c, _ in COMPS])
        W = {**{m_: np.array(w, float) for m_, w in MEASURES.items()},
             "pitch": base, "naware": base * k}

        def sb(idx):
            out = {}
            for m_, w in W.items():
                r = np.corrcoef(A[idx] @ w, B[idx] @ w)[0, 1]
                out[m_] = 2 * r / (1 + r)
            return out

        pt = sb(np.arange(len(A)))
        up = np.unique(pids)
        rows = {p: np.nonzero(pids == p)[0] for p in up}
        boots = {m_: [] for m_ in W}
        for _ in range(N_BOOT // 2):
            draw = rng.choice(up, len(up))
            idx = np.concatenate([rows[p] for p in draw])
            for m_, v in sb(idx).items():
                boots[m_].append(v)
        res[2 * h] = dict(per_half=h, pitcher_seasons=int(len(A)),
                          pitchers=int(len(up)),
                          rel={m_: round(float(pt[m_]), 4) for m_ in W},
                          lo={m_: round(float(np.percentile(boots[m_], 2.5)), 4) for m_ in W},
                          hi={m_: round(float(np.percentile(boots[m_], 97.5)), 4) for m_ in W})
        print(f"  check n={2 * h} ({h}/half): {len(A)} pitcher-seasons, {len(up)} pitchers  "
              + "  ".join(f"{m_} {pt[m_]:.3f}" for m_ in W))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    caches = [os.path.join(os.environ["STUFFPLUS_WORKDIR"], "pitches_cache_D1.parquet"),
              os.path.join(os.environ["STUFFPLUS_WORKDIR_CRIT"],
                           "pitches_cache_2025_2026_D1.parquet")]
    print(f"adjT method: {fc.resolve_adj_method()}")
    df = s12.load_seasons(caches)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)
    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    ff["loc"] = fc.PooledLocationMap(ff[(ff["year"] == s12.TRAIN_YEAR)
                                        & ff["xT"].notna()]).apply(ff)
    ff = ff.dropna(subset=[c for _, c in COMPS])
    ff = s12.add_game_parity(ff)

    # script 08's scale (2025, 100+ FF) and PR #35's n0 (2025 graded season, game split)
    s25 = ff[ff["year"] == 2025]
    pm = s25.groupby("PitcherId").agg(n=("adjT", "size"),
                                      **{c: (col, "mean") for c, col in COMPS})
    sd = {c: float(pm.loc[pm["n"] >= 100, c].std()) for c, _ in COMPS}
    n0_pr35 = {c: float(sf.reliability_curve(s25[["PitcherId", "GameID", "Date", col]],
                                              col, 20)["n0"]) for c, col in COMPS}
    print("PR #35 n0:", {c: round(v, 1) for c, v in n0_pr35.items()})

    t = season_table(ff)
    t = t[t["n"] >= fc.PANEL_MIN_FF]
    per = t.groupby("PitcherId")["year"].nunique()
    t = t[t["PitcherId"].isin(per[per >= 2].index)].reset_index(drop=True)
    noise_rows = ((t["n_h0"] >= MIN_HALF) & (t["n_h1"] >= MIN_HALF)).to_numpy()
    print(f"PANEL: {t['PitcherId'].nunique()} pitchers, {len(t)} pitcher-seasons")

    N, C, inv_n = moments(t, noise_rows)
    # cross-check the quadratic form against the library estimator for one component
    key = ff["PitcherId"].astype(str) + "|" + ff["year"].astype(str)
    keep = set((t["PitcherId"].astype(str) + "|" + t["year"].astype(str)))
    tmp = ff.loc[key.isin(keep), ["ridge_pred", "half"]].assign(ps=key[key.isin(keep)])
    lib, _ = vc.effective_noise_scale(tmp, "ridge_pred", "ps", min_half=MIN_HALF)
    assert abs(lib - N[0, 0]) / lib < 1e-6, (lib, N[0, 0])

    base = curves(N, C, inv_n, sd, n0_pr35)
    rng = np.random.default_rng(17)
    pids = t["PitcherId"].unique()
    rows = t.groupby("PitcherId").indices
    bn0 = {m: [] for m in base}
    brel = {m: [] for m in base}
    for _ in range(a.boot):
        draw = rng.choice(pids, len(pids))
        idx = np.concatenate([rows[p] for p in draw])
        bt = t.iloc[idx].reset_index(drop=True)
        try:
            cb = curves(*moments(bt, noise_rows[idx]), sd, n0_pr35)
        except np.linalg.LinAlgError:
            continue
        for m, (n0, rel) in cb.items():
            bn0[m].append(n0)
            brel[m].append(rel)

    out = dict(panel_pitchers=int(t["PitcherId"].nunique()), panel_seasons=int(len(t)),
               adj_method=fc.resolve_adj_method(), grid=N_GRID, n_boot=a.boot,
               min_half=MIN_HALF, n0_pr35=n0_pr35, measures={})
    print("\nstabilization point (reliability 0.5), 95% pitcher-bootstrap CI:")
    for m, (n0, rel) in base.items():
        lo, hi = np.percentile(bn0[m], [2.5, 97.5])
        rl, rh = np.percentile(np.array(brel[m]), [2.5, 97.5], axis=0)
        out["measures"][m] = dict(label=LABELS[m], n50=round(n0, 1),
                                  n50_ci=[round(float(lo), 1), round(float(hi), 1)],
                                  rel=[round(float(v), 4) for v in rel],
                                  lo=[round(float(v), 4) for v in rl],
                                  hi=[round(float(v), 4) for v in rh])
        at = {n: rel[N_GRID.index(n)] for n in (100, 400)}
        print(f"  {LABELS[m]:<32} n50 {n0:7.1f}  [{lo:.1f}, {hi:.1f}]   "
              f"rel@100 {at[100]:.3f}  rel@400 {at[400]:.3f}")

    print("\nempirical split-half check (2024 and 2025 pitcher-seasons):")
    out["empirical"] = empirical(ff, sd, n0_pr35, np.random.default_rng(29))
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, "within_season_reliability.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
