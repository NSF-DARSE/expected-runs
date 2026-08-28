"""Coach's hand-weighted scorecard vs our Stuff+ ridge, four-seams.

The pitching coach built a linear pitch-quality scorecard (Coach_Linear_Regression_
Model.xlsx, Level II, untracked at repo root). Despite the filename nothing in it is
fitted: the "Coefficient" column is hand-set integers and the file contains no target
variable. It is a hand-weighted linear scorecard over five four-seam features, keyed
by pitch type x pitcher handedness, referenced to per-hand D1 averages.

This script scores it on our feed and compares it to the incumbent Stuff+ ridge using
the project's standing protocol (FRAMEWORK.md): year-over-year reliability and
predictive validity against the fair criterion, paired bootstrap on the differences.
NOT pitch-level R^2.

SIGN CONVENTIONS (fair_criterion.py): Target/xT/adjT/ridge_pred are expected runs from
the pitcher's perspective, LOWER = BETTER. The coach's scorecard is built the other way
(HIGHER = BETTER). To keep one frame end to end, his score is negated once, here, into
`coach_rv`, so every quantity below is lower-is-better and correlations are read the
same direction for both models. Nothing is negated again downstream.

His spec leaves one thing undefined: the RelHeight/RelSide rows are marked
"*Off Average" / CalcType Absolute, rewarding DEVIATION from the D1 average, but the
file never says which direction of deviation is good. Both readings are scored
(UNIQUE = far from average is good, CONFORM = close to average is good) rather than
guessed.

Data rules: reads the source CSV via fair_criterion.paths(); writes nothing but stdout.
Per-pitcher output is never committed.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook

import fair_criterion as fc

# The coach's workbook is not in version control (it is his file, and it carries his
# unpublished grades), so it is located rather than hardcoded: COACH_WEIGHTS_XLSX if set,
# otherwise the checkout root, where it sits in a normal clone. Resolved at import rather
# than by argparse because other scripts import load_coach_terms directly.
WEIGHTS_XLSX = os.environ.get(
    "COACH_WEIGHTS_XLSX",
    str(pathlib.Path(__file__).resolve().parents[2] / "Coach_Linear_Regression_Model.xlsx"),
)
COACH_TYPE = "FourSeamFastBall"  # his label for the four-seam block

# his feature spellings -> our columns
FEATURE_MAP = {
    "effectivevelo": "EffectiveVelo",
    "horzbreak": "HorzBreak",
    "inducedvertbreak": "InducedVertBreak",
    "spinrate": "SpinRate",
    "velocity_differential": "velocity_differential",
    "RelHeight": "RelHeight",
    "RelSide": "RelSide",
}
N_BOOT = 4000


def load_coach_terms(pitch_type: str) -> list[dict]:
    """Parse one pitch type's rows out of his workbook."""
    if not os.path.exists(WEIGHTS_XLSX):
        sys.exit("the coach's workbook is not at %s. It is untracked, so a fresh worktree "
                 "will not have it: copy it in, or set COACH_WEIGHTS_XLSX to its path."
                 % WEIGHTS_XLSX)
    ws = load_workbook(WEIGHTS_XLSX, data_only=True)["Sheet1"]
    terms = []
    for hand, ptype, feat, coef, note, avg, calc, direction in ws.iter_rows(min_row=2, values_only=True):
        if ptype != pitch_type or hand is None:
            continue
        if feat not in FEATURE_MAP:
            raise ValueError(f"unmapped feature in his workbook: {feat!r}")
        if avg is None or calc is None:
            # Cutter/LHP horzbreak is blank in his file. Not reachable for four-seams;
            # raise rather than silently score a term as zero if it ever is.
            raise ValueError(f"incomplete row: {hand} {ptype} {feat} (note={note!r})")
        terms.append(dict(hand=hand, col=FEATURE_MAP[feat], coef=float(coef), avg=float(avg),
                          relative=(calc == "Relative"),
                          direction=float(direction) if direction is not None else None))
    if not terms:
        raise ValueError(f"no rows for {pitch_type!r}")
    return terms


def coach_score(ff: pd.DataFrame, terms: list[dict], absolute_sign: float) -> pd.Series:
    """His scorecard, evaluated per pitch. HIGHER = BETTER in his frame.

    absolute_sign: +1 scores "*Off Average" rows as deviation-is-good (UNIQUE),
    -1 as deviation-is-bad (CONFORM). Relative rows are unaffected.
    """
    total = pd.Series(0.0, index=ff.index)
    for t in terms:
        rows = ff["PitcherThrows"] == t["hand"]
        gap = ff.loc[rows, t["col"]] - t["avg"]
        if t["relative"]:
            total.loc[rows] += t["coef"] * t["direction"] * gap
        else:
            total.loc[rows] += absolute_sign * t["coef"] * gap.abs()
    return total


def main() -> int:
    args = fc.paths()
    terms = load_coach_terms(COACH_TYPE)
    used = sorted({t["col"] for t in terms})
    print(f"his {COACH_TYPE} block: {len(terms)} rows over {len(used)} features {used}")
    print(f"  ours uses {len(fc.FEATS)} features; his omits "
          f"{sorted(set(fc.FEATS) - set(used))}\n")

    df = fc.load_pitches(args)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)  # four-seams with complete FEATS + ridge_pred
    ff = ff.dropna(subset=used).copy()
    train_yr, eval_yr = 2024, 2025  # role labels; real years printed by load_pitches

    for variant, sign in [("UNIQUE (off-average = good)", 1.0),
                          ("CONFORM (off-average = bad)", -1.0)]:
        ff[f"coach_{sign:+.0f}"] = -coach_score(ff, terms, sign)  # negate once: lower = better

    ids = fc.panel_ids(ff)
    g = ff[ff["PitcherId"].isin(ids)].groupby(["PitcherId", "year"])
    tab = pd.DataFrame({"C2": g["adjT"].mean(), "ours": g["ridge_pred"].mean(),
                        "coach_unique": g["coach_+1"].mean(),
                        "coach_conform": g["coach_-1"].mean(),
                        "n": g.size()}).reset_index()
    a, b = fc.year_split(tab, sorted(ids))
    N = len(ids)
    print(f"panel: {N} pitchers with 100+ four-seams in both seasons "
          f"(median {int(tab.n.median())} FF/season)\n")

    scores = ["ours", "coach_unique", "coach_conform"]
    print("Every column below is in the LOWER = BETTER expected-runs frame, so a")
    print("POSITIVE validity correlation means the score predicts better outcomes.\n")
    print(f"{'score':<16}{'reliability':>13}{'validity (P)':>14}{'validity (S)':>14}")
    for c in scores:
        print(f"{c:<16}{fc.R(a[c], b[c]):>13.3f}{fc.R(a[c], b['C2']):>14.3f}"
              f"{fc.RS(a[c], b['C2']):>14.3f}")
    print(f"{'C2 (criterion)':<16}{fc.R(a['C2'], b['C2']):>13.3f}"
          f"{'--':>14}{'--':>14}")

    print("\ncross-correlation, train season (how much is the same measurement?):")
    for c in scores[1:]:
        print(f"  ours vs {c:<14} r={fc.R(a['ours'], a[c]):+.3f}")

    # paired bootstrap on the differences: same resample for both models
    rng = np.random.default_rng(42)
    M = np.column_stack([a[c] for c in scores] + [b[c] for c in scores] + [b["C2"]])
    K = len(scores)
    d = {f"{m}_{c}": [] for m in ("rel", "val") for c in scores[1:]}
    for _ in range(N_BOOT):
        s = M[rng.choice(N, N, replace=True)]
        A, B, y = s[:, :K], s[:, K:2 * K], s[:, -1]
        for j, c in enumerate(scores[1:], start=1):
            d[f"rel_{c}"].append(np.corrcoef(A[:, 0], B[:, 0])[0, 1]
                                 - np.corrcoef(A[:, j], B[:, j])[0, 1])
            d[f"val_{c}"].append(np.corrcoef(A[:, 0], y)[0, 1] - np.corrcoef(A[:, j], y)[0, 1])
    print(f"\npaired bootstrap ({N_BOOT}), OURS minus HIS (positive = ours better):")
    for c in scores[1:]:
        fc.boot_report(f"reliability: ours - {c}", d[f"rel_{c}"])
        fc.boot_report(f"validity:    ours - {c}", d[f"val_{c}"])

    # what is his score actually made of? velo carries coef 10 of his ~30 total weight
    print("\nhis score decomposed, train season (pitcher-level r vs his full score):")
    sub = ff[ff["year"] == train_yr]
    full = -coach_score(sub, terms, 1.0)
    for t in sorted({t["col"] for t in terms}):
        only = [x for x in terms if x["col"] == t]
        part = -coach_score(sub, only, 1.0)
        pv = pd.DataFrame({"f": full, "p": part, "id": sub["PitcherId"]}).groupby("id").mean()
        print(f"  {t:<22} r={fc.R(pv['f'], pv['p']):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
