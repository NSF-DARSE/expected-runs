"""Same-season fit vs next-season fit, per score. The luck-stripping diagnostic.

The point of stripping batted-ball luck out of the criterion is NOT to explain the
current season's runs better -- it is to explain NEXT season's runs better. So the
informative quantity is not either fit alone, it is the DROP-OFF from same-season to
next-season:

  a score that tracks luck        -> strong same-season, weak next-season (big drop)
  a score that is luck-free       -> weaker same-season, holds up next season

Adjusted results is included as the reference case. It IS the graded season's results
with luck removed, so it should be very strong same-season and near zero next season.
If our physical scores show a smaller drop-off than his card, that is the luck-stripping
working as intended, and it is the argument that survives the head-to-head being a tie.

No controls here, deliberately: the question is raw explanatory power against runs in
each season, so adding controls would change the question.

SIGN CONVENTION: scores are higher-is-better (`_hi`); RA9 is lower-is-better. Effects
are reported as RUNS PER 9 BETTER per +1 SD, so positive always means the score works.

Data rules: reads the cached pool and the workdir parquet only. Writes nothing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import coach_model_paired as cp
import coach_model_two_panel as tp
import fair_criterion as fc

N_BOOT = 4000
ORDER = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi", "adjres_hi", "pitch_hi"]


def effect(s: pd.DataFrame, score: str, outcome: str) -> float:
    """Runs/9 better per +1 SD, no controls. Slope of outcome on z(score), negated."""
    X = np.column_stack([np.ones(len(s)), cp.z(s[score].values)])
    beta, *_ = np.linalg.lstsq(X, s[outcome].values, rcond=None)
    return -beta[-1]


def main() -> int:
    args = fc.paths()
    pool = pd.read_parquet(os.path.join(args.workdir, "coach_compare_pool.parquet"))
    f = tp.component_scores(args, pool)
    print(f"pool: {len(f)} pitchers, graded 2025, follow-up 2026\n")

    rng = np.random.default_rng(31)
    idx = f.index.values
    draws = {(sc, w): [] for sc in ORDER for w in ("same", "next")}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for sc in ORDER:
            draws[(sc, "same")].append(effect(s, sc, "ra9_graded"))
            draws[(sc, "next")].append(effect(s, sc, "ra9_next"))
    d = {k: np.array(v) for k, v in draws.items()}

    print("RUNS PER 9 explained per +1 SD of score, no controls:")
    print(f"  {'score':<18}{'same season':>13}{'next season':>13}{'drop-off':>11}"
          f"{'retained':>10}")
    rows = []
    for sc in ORDER:
        same, nxt = d[(sc, "same")], d[(sc, "next")]
        drop = same - nxt
        retained = nxt.mean() / same.mean() if abs(same.mean()) > 1e-9 else float("nan")
        rows.append((sc, same.mean(), nxt.mean(), drop, retained))
        print(f"  {tp.LABELS[sc]:<18}{same.mean():>+13.2f}{nxt.mean():>+13.2f}"
              f"{drop.mean():>+11.2f}{retained:>9.0%}")

    print("\ndrop-off with uncertainty (positive = loses power next season):")
    for sc, _, _, drop, _ in rows:
        fc.boot_report(f"{tp.LABELS[sc]}", drop)

    print("\nHIS CARD vs OUR STUFF+, paired on the same resamples:")
    for w, label in [("same", "same-season fit"), ("next", "next-season fit")]:
        fc.boot_report(f"{label}: coach - ours",
                       d[("coach_hi", w)] - d[("stuff_hi", w)])
    fc.boot_report("drop-off: coach - ours  [the luck-stripping test]",
                   (d[("coach_hi", "same")] - d[("coach_hi", "next")])
                   - (d[("stuff_hi", "same")] - d[("stuff_hi", "next")]))

    print("\nsame reading in plain correlations (r with RA9, negated so + = good):")
    print(f"  {'score':<18}{'same':>8}{'next':>8}")
    for sc in ORDER:
        print(f"  {tp.LABELS[sc]:<18}{-fc.R(f[sc], f['ra9_graded']):>8.3f}"
              f"{-fc.R(f[sc], f['ra9_next']):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
