"""Coach's scorecard vs velo-only vs our Stuff+, in runs per 9 innings.

Companion to coach_model_comparison.py, which answers the same question in
correlations. This one answers it in the unit the project already proves the portal
board in: next-season RA9 improvement per 1 SD of grade, holding the current line
fixed (RESULTS.md, portal section).

THREE SCORES, all four-seam, all graded on the 2025 season:
  velo    plain mean EffectiveVelo. The radar-gun baseline his card has to beat.
          (His own weighted velo term is r=0.944 with this; the difference is that
          he centers per handedness. Plain velo is used because it needs no
          explanation to a coach.)
  coach   his hand-weighted scorecard, "*Off Average" read as deviation-is-good,
          which is the reading his file needs (see coach_model_comparison.py).
  ours    the Stuff+ ridge.

FAIRNESS: the ridge is fitted and his card is not, so grading a season the ridge
trained on would hand us a free advantage. Hence the 2024/2025 pair: ridge trains on
2024, both models grade 2025, and both are scored against what happened in 2026.
Out-of-sample for us, unchanged for him.

SIGN CONVENTIONS: fair_criterion keeps everything in expected runs, lower = better.
For presentation each score is flipped ONCE here into a higher-is-better display
frame (`_hi` suffix), and the reported effect is stated as RUNS OF RA9 IMPROVEMENT
per +1 SD, so a POSITIVE number always means the score is doing its job. RA9 itself
is lower = better throughout.

Data rules: source CSVs are read read-only; nothing is written. Per-pitcher output
stays in stdout and is never committed.
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import pandas as pd

import coach_model_comparison as cm
import fair_criterion as fc

# The two season line-stat pulls. Both are untracked (large, and licensed TrackMan), so
# these are defaults against the checkout root rather than guarantees; --ra9-2025 and
# --ra9-2026 override, and a missing file names the flag instead of raising a bare
# FileNotFoundError three frames down. The 2109 in the 2025 filename is how the file is
# actually named on disk.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
RA9_2025_DEFAULT = str(_ROOT / "Final_Target_Calc_2109.csv")
RA9_2026_DEFAULT = str(_ROOT / "trackman_api" / "2026_build" / "Final_Target_Calc_2026.csv")
USECOLS = ["Date", "Pitcher", "PitcherId", "PitcherTeam", "Balls", "Strikes", "PitchofPA",
           "PitchCall", "OutsOnPlay", "RunsScored", "Level"]

MIN_FF = 100        # four-seams in the graded season
MIN_IP_GRADED = 20  # innings in the graded season (portal board threshold)
MIN_IP_NEXT = 15    # innings in the follow-up season
N_BOOT = 4000
CALIPER = 0.30      # RA9 matching caliper, same as the portal board
SCORES = ["velo_hi", "coach_hi", "ours_hi"]
LABELS = {"velo_hi": "Velo only", "coach_hi": "Coach's card", "ours_hi": "Our Stuff+"}


def line_stats(path: str, year: int) -> pd.DataFrame:
    """Pitcher-season surface line: RA9, K%, BB%, IP. Mirrors build_portal_data.stats."""
    if not os.path.exists(path):
        raise SystemExit("no %d line data at %s. Pass --ra9-%d with its path."
                         % (year, path, year))
    print(f"  loading {year} line data (large file, minutes)...", flush=True)
    df = pd.read_csv(path, usecols=USECOLS, low_memory=False)
    df = df[df["Level"] == "D1"]
    df = df[pd.to_datetime(df["Date"], errors="coerce").dt.year == year]
    df["PitcherId"] = pd.to_numeric(df["PitcherId"], errors="coerce").astype("Int64")
    df = df[df["PitcherId"].notna()]
    d = df.assign(
        is_k=(df["Strikes"] == 2) & df["PitchCall"].isin(["StrikeSwinging", "StrikeCalled"]),
        is_bb=(df["Balls"] == 3) & (df["PitchCall"] == "BallCalled"),
        is_pa=df["PitchofPA"] == 1,
    )
    d["outs"] = d["OutsOnPlay"].fillna(0) + d["is_k"].astype(int)
    g = d.groupby("PitcherId").agg(runs=("RunsScored", "sum"), outs=("outs", "sum"),
                                   k=("is_k", "sum"), bb=("is_bb", "sum"), pa=("is_pa", "sum"))
    g["ip"] = g["outs"] / 3
    g["ra9"] = np.where(g["ip"] > 0, g["runs"] * 9 / g["ip"], np.nan)
    g["k_pct"] = g["k"] / g["pa"]
    g["bb_pct"] = g["bb"] / g["pa"]
    print(f"    {year}: {len(g)} D1 pitchers", flush=True)
    return g[["ip", "ra9", "k_pct", "bb_pct"]]


def graded_season_scores(args) -> pd.DataFrame:
    """Pitcher-level four-seam scores for the graded season (eval role = 2025)."""
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})
    df = fc.load_pitches(args)
    ff = fc.stuff_ridge(df).dropna(subset=used).copy()   # ridge trains on year role 2024
    ff["coach_raw"] = cm.coach_score(ff, terms, 1.0)      # his frame: higher = better
    season = ff[ff["year"] == 2025]                       # eval role year
    g = season.groupby("PitcherId").agg(
        n_ff=("ridge_pred", "size"), ridge=("ridge_pred", "mean"),
        coach_hi=("coach_raw", "mean"), velo_hi=("EffectiveVelo", "mean"),
        name=("Pitcher", "first"), team=("PitcherTeam", "first"))
    g["ours_hi"] = -g["ridge"]        # ridge is lower = better; flip once for display
    return g[g["n_ff"] >= MIN_FF]


def z(s: pd.Series) -> np.ndarray:
    return ((s - s.mean()) / s.std()).values


def effect_per_sd(f: pd.DataFrame, score: str, controls: list[str], rng) -> dict:
    """Bootstrap RA9 improvement per +1 SD of score, holding `controls` fixed.

    Fits ra9_next ~ 1 + controls + z(score). The fitted slope is negative when a
    better score means fewer runs, so it is negated into "runs better per SD".
    """
    out = []
    idx = f.index.values
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        X = np.column_stack([np.ones(len(s))] + [s[c].values for c in controls] + [z(s[score])])
        beta, *_ = np.linalg.lstsq(X, s["ra9_next"].values, rcond=None)
        out.append(-beta[-1])
    a = np.array(out)
    return dict(effect=a.mean(), se=a.std(),
                lo=np.percentile(a, 2.5), hi=np.percentile(a, 97.5), p=(a > 0).mean())


def matched_pairs(f: pd.DataFrame, liked_col: str, disliked_col: str) -> dict:
    """Disagreement test: pitchers one model likes far more than the other.

    Ranks on both scores, takes the quartile where `liked_col` is most favourable
    relative to `disliked_col` and the quartile where the reverse holds, then
    nearest-neighbour matches on the GRADED season's RA9 so both groups start level.
    Matching is what makes this readable: an unmatched tercile split on this data was
    already shown to be confounded (RESULTS.md, portal correction).
    """
    d = f.copy()
    d["gap"] = d[liked_col].rank(pct=True) - d[disliked_col].rank(pct=True)
    hi = d[d["gap"] >= d["gap"].quantile(0.75)].sort_values("ra9_graded")
    lo = d[d["gap"] <= d["gap"].quantile(0.25)].sort_values("ra9_graded")
    used, pairs = set(), []
    for i, r in hi.iterrows():
        cand = lo[~lo.index.isin(used)]
        diffs = (cand["ra9_graded"] - r["ra9_graded"]).abs()
        if len(diffs) and diffs.min() <= CALIPER:
            j = diffs.idxmin()
            used.add(j)
            pairs.append((i, j))
    A = hi.loc[[i for i, _ in pairs]]
    B = lo.loc[[j for _, j in pairs]]
    return dict(n=len(pairs),
                a_graded=A["ra9_graded"].mean(), a_next=A["ra9_next"].mean(),
                b_graded=B["ra9_graded"].mean(), b_next=B["ra9_next"].mean(),
                a_improved=int((A["ra9_next"] < A["ra9_graded"]).sum()),
                b_improved=int((B["ra9_next"] < B["ra9_graded"]).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ra9-2025", default=RA9_2025_DEFAULT)
    ap.add_argument("--ra9-2026", default=RA9_2026_DEFAULT)
    known, _ = ap.parse_known_args()

    args = fc.paths()
    if args.year_pair != (2024, 2025):
        raise SystemExit("run with --years 2024,2025 so the ridge is out-of-sample "
                         "on the graded season (see module docstring)")
    print("four-seam scores for the graded season (2025)...", flush=True)
    g = graded_season_scores(args)
    print(f"  {len(g)} D1 pitchers with {MIN_FF}+ four-seams in 2025", flush=True)

    s25 = line_stats(known.ra9_2025, 2025)
    s26 = line_stats(known.ra9_2026, 2026)

    f = g.join(s25, how="inner").join(s26, how="inner", lsuffix="_graded", rsuffix="_next")
    f = f[(f["ip_graded"] >= MIN_IP_GRADED) & (f["ip_next"] >= MIN_IP_NEXT)].copy()
    f = f.dropna(subset=["ra9_graded", "ra9_next", "k_pct_graded", "bb_pct_graded"])
    print(f"\npool: {len(f)} pitchers graded in 2025 with a 2026 follow-up "
          f"({MIN_IP_GRADED}+ IP then, {MIN_IP_NEXT}+ IP after)")
    print(f"  they allowed {f['ra9_graded'].mean():.2f} runs/9 in 2025, "
          f"{f['ra9_next'].mean():.2f} in 2026 (everyone regresses toward the mean)")

    print("\nHow much is each score worth? RUNS PER 9 BETTER NEXT SEASON per 1 SD of")
    print("grade. Positive = the score is telling you something. Bootstrap "
          f"{N_BOOT} resamples.\n")
    rng = np.random.default_rng(7)
    for controls, note in [(["ra9_graded"], "holding this year's runs allowed fixed"),
                           (["ra9_graded", "k_pct_graded", "bb_pct_graded"],
                            "holding runs, strikeout rate AND walk rate fixed")]:
        print(f"  {note}:")
        for sc in SCORES:
            r = effect_per_sd(f, sc, controls, rng)
            print(f"    {LABELS[sc]:<14} {r['effect']:+.2f} runs/9  "
                  f"(SE {r['se']:.2f}, 95% CI [{r['lo']:+.2f},{r['hi']:+.2f}], "
                  f"P(>0)={r['p']:.3f})")
        print()

    print("Head to head, all three in one model (who still adds something once the")
    print("others are in?), holding runs + K + BB fixed:")
    X = np.column_stack([np.ones(len(f)), f["ra9_graded"].values,
                         f["k_pct_graded"].values, f["bb_pct_graded"].values]
                        + [z(f[sc]) for sc in SCORES])
    beta, *_ = np.linalg.lstsq(X, f["ra9_next"].values, rcond=None)
    for sc, bcoef in zip(SCORES, beta[4:]):
        print(f"    {LABELS[sc]:<14} {-bcoef:+.2f} runs/9 per SD")

    print("\nWhen the two models disagree, who is right?")
    print("Top quartile of disagreement each way, matched on 2025 runs allowed")
    print(f"(caliper {CALIPER}) so both groups start level:\n")
    m = matched_pairs(f, "ours_hi", "coach_hi")
    print(f"  {m['n']} matched pairs")
    print(f"    we like him, his card does not : {m['a_graded']:.2f} -> {m['a_next']:.2f} runs/9 "
          f"({m['a_improved']}/{m['n']} improved)")
    print(f"    his card likes him, we do not  : {m['b_graded']:.2f} -> {m['b_next']:.2f} runs/9 "
          f"({m['b_improved']}/{m['n']} improved)")
    print(f"    gap next season: {m['b_next'] - m['a_next']:+.2f} runs/9 in our favour")

    print("\nThe backwards break sign, in plain outcomes. Right-handed four-seams in")
    print("2025 by horizontal break, and the runs they actually allowed next season.")
    print("His card scores MORE arm-side run as better; watch which way this goes.\n")
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})
    ff = fc.stuff_ridge(fc.load_pitches(args)).dropna(subset=used)
    rh = ff[(ff["year"] == 2025) & (ff["PitcherThrows"] == "Right")]
    hb = rh.groupby("PitcherId")["HorzBreak"].agg(["size", "mean"])
    hb = hb[hb["size"] >= MIN_FF].join(f[["ra9_next"]], how="inner")
    hb["bucket"] = pd.qcut(hb["mean"], 5)
    print(f"    {'arm-side run (in)':<22}{'pitchers':>9}{'next-yr runs/9':>16}")
    for bucket, grp in hb.groupby("bucket", observed=True):
        print(f"    {str(bucket):<22}{len(grp):>9}{grp['ra9_next'].mean():>16.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
