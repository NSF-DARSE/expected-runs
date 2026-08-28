"""Paired bootstrap on DIFFERENCES between the three four-seam scores, in RA9 units.

coach_model_coach_units.py reported three separate effects with three separate CIs.
That is the wrong test for ranking correlated scores, and FRAMEWORK.md says so:
prefer a paired bootstrap on the difference, treat <~1 SE as a tie. The three scores
here correlate 0.73-0.94 with each other, so separate CIs badly overstate how
distinguishable they are, and a joint regression on all three is too collinear for
its coefficients to be read as importance.

Also attaches an uncertainty estimate to the disagreement/matched-pairs gap, which
the earlier run reported as a bare point estimate.

Reuses the graded pool, cached as parquet in the workdir on first run so the two
~1.2 GB line-data CSVs are parsed once rather than per experiment.

SIGN CONVENTION: every score is in a higher-is-better display frame (`_hi`), and
effects are stated as RUNS OF RA9 IMPROVEMENT per +1 SD, so positive = the score
works. Differences are stated as A minus B, positive = A better.

Data rules: writes only the pool cache inside the workdir. Never committed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import coach_model_coach_units as cu
import fair_criterion as fc

N_BOOT = 4000
SCORES = ["velo_hi", "coach_hi", "ours_hi"]
LABELS = {"velo_hi": "velo only", "coach_hi": "coach's card", "ours_hi": "our Stuff+"}
CONTROLS = ["ra9_graded", "k_pct_graded", "bb_pct_graded"]


def build_pool(args) -> pd.DataFrame:
    cache = os.path.join(args.workdir, "coach_compare_pool.parquet")
    if os.path.exists(cache):
        print(f"pool from cache: {cache}")
        return pd.read_parquet(cache)
    g = cu.graded_season_scores(args)
    f = (g.join(cu.line_stats(cu.RA9_2025, 2025), how="inner")
          .join(cu.line_stats(cu.RA9_2026, 2026), how="inner",
                lsuffix="_graded", rsuffix="_next"))
    f = f[(f["ip_graded"] >= cu.MIN_IP_GRADED) & (f["ip_next"] >= cu.MIN_IP_NEXT)]
    f = f.dropna(subset=["ra9_graded", "ra9_next"] + CONTROLS[1:]).copy()
    f.drop(columns=["name", "team"]).to_parquet(cache)  # no names in the cache
    print(f"pool cached to {cache}")
    return f.drop(columns=["name", "team"])


def z(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def fit_effect(s: pd.DataFrame, score: str) -> float:
    """RA9 improvement per +1 SD of score, holding the full line fixed."""
    X = np.column_stack([np.ones(len(s))] + [s[c].values for c in CONTROLS]
                        + [z(s[score].values)])
    beta, *_ = np.linalg.lstsq(X, s["ra9_next"].values, rcond=None)
    return -beta[-1]


def main() -> int:
    args = fc.paths()
    f = build_pool(args)
    print(f"pool: {len(f)} pitchers\n")

    print("pairwise correlation between the three scores (why separate CIs mislead):")
    for i, a in enumerate(SCORES):
        for b in SCORES[i + 1:]:
            print(f"  {LABELS[a]:<14} vs {LABELS[b]:<14} r={fc.R(f[a], f[b]):+.3f}")

    rng = np.random.default_rng(11)
    idx = f.index.values
    draws = {s: [] for s in SCORES}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for sc in SCORES:
            draws[sc].append(fit_effect(s, sc))
    d = {k: np.array(v) for k, v in draws.items()}

    print(f"\neffect per SD, holding runs + K + BB fixed (bootstrap {N_BOOT}):")
    for sc in SCORES:
        a = d[sc]
        print(f"  {LABELS[sc]:<14} {a.mean():+.2f} runs/9  SE {a.std():.2f}")

    print("\nPAIRED differences, same resamples (positive = first one better):")
    for i, a in enumerate(SCORES):
        for b in SCORES[i + 1:]:
            fc.boot_report(f"{LABELS[a]} - {LABELS[b]}", d[a] - d[b])

    print("\ndisagreement / matched pairs, with uncertainty:")
    base = cu.matched_pairs(f.assign(ra9_graded=f["ra9_graded"]), "ours_hi", "coach_hi")
    print(f"  point estimate: {base['n']} pairs, "
          f"ours {base['a_graded']:.2f}->{base['a_next']:.2f}, "
          f"his {base['b_graded']:.2f}->{base['b_next']:.2f}, "
          f"gap {base['a_next'] - base['b_next']:+.2f} runs/9 (positive = ours worse)")
    gaps = []
    for _ in range(500):  # matching is O(n^2), so fewer resamples than the regressions
        s = f.loc[rng.choice(idx, len(idx))].reset_index(drop=True)
        try:
            m = cu.matched_pairs(s, "ours_hi", "coach_hi")
        except (ValueError, KeyError):
            continue
        if m["n"] >= 30:
            gaps.append(m["a_next"] - m["b_next"])
    fc.boot_report("matched-pair gap (ours minus his, + = ours worse)", np.array(gaps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
