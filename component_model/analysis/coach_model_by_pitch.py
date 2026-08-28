"""Run the coach-vs-ours comparison for ANY pitch type, not just four-seams.

Everything before this was four-seam only. His workbook covers eight pitch types and
our arsenal seven, so the question "does the fastball tie hold on a secondary?" needs
the same protocol applied per type.

WHAT CHANGES ON A SECONDARY, and why the numbers are not directly comparable to the
four-seam run:
  - Pool size collapses. Four-seams support a 100-pitch floor; a changeup averages
    ~67 per pitcher-season, so the floor has to drop and the pool shrinks anyway.
    Less power, so a tie here is weaker evidence of similarity than the FF tie was.
  - Location+ is absent by decision (fastball-only score, RESULTS.md).
  - The single-feature baseline changes. For a four-seam it is velocity. For a
    changeup his own card has NO velocity term at all, only velocity_differential, so
    the baseline is separation off the fastball. The baseline feature is therefore
    chosen per type from his own weights rather than fixed.

MAPPING HAZARDS in his workbook, verified, and why some types are refused outright:
  - Splitter: his rows are a byte-for-byte copy of ChangeUp (same D1 averages to four
    decimals). Uncalibrated. REFUSED.
  - Cutter: LHP horzbreak has a blank D1 average and blank CalcType. REFUSED until a
    rule is chosen.
  - Slider: his taxonomy has Gyro/Sweeper and Slurve and no plain Slider, while our
    feed's dominant label IS Slider. REFUSED pending an explicit mapping decision;
    scoring it on either of his slider variants is an assumption, not a translation.

SIGN CONVENTION: scores end up higher-is-better (`_hi`); RA9 is lower-is-better.
Nothing is negated twice.

Data rules: caches all-pitcher line stats and per-type pools inside the workdir. No
names in any cache. Never committed.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import coach_model_band_table as bt
import coach_model_coach_units as cu
import coach_model_comparison as cm
import coach_model_paired as cp
import fair_criterion as fc

# our feed's labels for each of his pitch-type names
TYPE_MAP = {
    "FourSeamFastBall": None,              # None = use is_ff (covers all 3 spellings)
    "ChangeUp": {"ChangeUp"},
    "Curveball": {"Curveball"},
    "Sinker": {"Sinker", "TwoSeamFastBall"},
}
REFUSED = {
    "Splitter": "his Splitter rows are a copy of ChangeUp (identical D1 averages) -- "
                "uncalibrated",
    "Cutter": "his Cutter LHP horzbreak has no D1 average and no CalcType",
    "Gyro/Sweeper": "our feed labels sliders 'Slider'; mapping to his slider split is "
                    "a modelling decision, not a translation",
    "Slurve": "same slider-mapping problem as Gyro/Sweeper",
}
# single-feature baseline per type, taken from his own weights: the feature he leans on
BASELINE = {"FourSeamFastBall": ("EffectiveVelo", "Velo only", +1),
            "ChangeUp": ("velocity_differential", "Velo separation only", -1),
            "Curveball": ("EffectiveVelo", "Velo only", +1),
            "Sinker": ("EffectiveVelo", "Velo only", +1)}
N_BOOT = 3000
LAB = ["worst third", "middle", "best third"]  # qcut ascending on a higher=better score


def all_line_stats(workdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Line stats for every D1 pitcher in both seasons, cached once for all types."""
    cache = os.path.join(workdir, "line_stats_all.parquet")
    if os.path.exists(cache):
        d = pd.read_parquet(cache)
        return d[d.season == 2025].drop(columns="season"), d[d.season == 2026].drop(columns="season")
    a = cu.line_stats(cu.RA9_2025, 2025).assign(season=2025)
    b = cu.line_stats(cu.RA9_2026, 2026).assign(season=2026)
    pd.concat([a, b]).to_parquet(cache)
    print(f"  cached {cache}")
    return a.drop(columns="season"), b.drop(columns="season")


def build(args, coach_type: str, floor: int) -> pd.DataFrame:
    if coach_type in REFUSED:
        sys.exit(f"refusing {coach_type}: {REFUSED[coach_type]}")
    terms = cm.load_coach_terms(coach_type)
    used = sorted({t["col"] for t in terms})
    feat, feat_label, feat_dir = BASELINE[coach_type]
    print(f"{coach_type}: his block uses {used}")
    print(f"  baseline column = {feat} ({feat_label}), his direction {feat_dir:+d}")

    df = fc.load_pitches(args)
    fc.add_xt(df)
    fc.add_adjusted(df)
    mask = df["is_ff"] if TYPE_MAP[coach_type] is None else \
        df["TaggedPitchType"].isin(TYPE_MAP[coach_type])
    pp = fc.stuff_ridge(df, pitch_mask=mask).dropna(subset=used).copy()
    pp["coach_raw"] = cm.coach_score(pp, terms, 1.0)

    season = pp[pp["year"] == 2025]
    g = season.groupby("PitcherId").agg(n=("ridge_pred", "size"),
                                        ridge=("ridge_pred", "mean"),
                                        coach_hi=("coach_raw", "mean"),
                                        base=(feat, "mean"))
    print(f"  pitchers with any {coach_type}: {len(g)}; with {floor}+: "
          f"{int((g.n >= floor).sum())}")
    g = g[g["n"] >= floor]
    g["stuff_hi"] = -g["ridge"]
    g["base_hi"] = feat_dir * g["base"]   # his direction, so higher = better

    s25, s26 = all_line_stats(args.workdir)
    f = g.join(s25, how="inner").join(s26, how="inner", lsuffix="_graded", rsuffix="_next")
    f = f[(f["ip_graded"] >= cu.MIN_IP_GRADED) & (f["ip_next"] >= cu.MIN_IP_NEXT)]
    f = f.dropna(subset=["ra9_graded", "ra9_next"]).copy()
    f.attrs["baseline_label"] = feat_label
    print(f"  pool with a 2026 follow-up: {len(f)}")
    return f


def spread(d: pd.DataFrame, col: str) -> float:
    return (d.loc[d[col + "_b"] == "worst third", "ra9_next"].mean()
            - d.loc[d[col + "_b"] == "best third", "ra9_next"].mean())


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pitch", default="ChangeUp")
    ap.add_argument("--floor", type=int, default=50)
    known, _ = ap.parse_known_args()
    args = fc.paths()

    f = build(args, known.pitch, known.floor)
    if len(f) < 90:
        print(f"\npool of {len(f)} is too small for thirds of 30+; stopping rather than "
              "reporting cell means over a handful of pitchers")
        return 1

    cols = [("base_hi", f.attrs["baseline_label"]), ("coach_hi", "Coach's card"),
            ("stuff_hi", "Our Stuff+")]
    for c, _ in cols:
        f[c + "_b"] = pd.qcut(bt.display(f[c]), 3, labels=LAB)

    print(f"\n{known.pitch}: 2026 RA9 by 2025 grade third (floor {known.floor} pitches, "
          f"n={len(f)})")
    print(f"  {'band':<16}" + "".join(f"{lbl:>24}" for _, lbl in cols))
    for lab in ["best third", "middle", "worst third"]:
        line = f"  {lab:<16}"
        for c, _ in cols:
            grp = f[f[c + "_b"] == lab]
            line += (f"{grp.ra9_next.mean():>14.2f} +/-"
                     f"{grp.ra9_next.std() / np.sqrt(len(grp)):.2f} ({len(grp):>3})")
        print(line)
    print("\n  best-to-worst spread:")
    for c, lbl in cols:
        print(f"    {lbl:<24}{spread(f, c):+.2f} runs/9")

    rng = np.random.default_rng(17)
    idx = f.index.values
    B = {c: [] for c, _ in cols}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c, _ in cols:
            s[c + "_b"] = pd.qcut(bt.display(s[c]), 3, labels=LAB)
            B[c].append(spread(s, c))
    B = {k: np.array(v) for k, v in B.items()}
    print("\n  paired bootstrap on spread differences (same resamples):")
    for i, (a, la) in enumerate(cols):
        for b, lb in cols[i + 1:]:
            fc.boot_report(f"{la} - {lb}", B[a] - B[b])

    print("\n  his card, per-term contribution to its own score (pitcher level):")
    terms = cm.load_coach_terms(known.pitch)
    sub = fc.stuff_ridge(fc.load_pitches(args),
                         pitch_mask=(fc.load_pitches(args)["is_ff"]
                                     if TYPE_MAP[known.pitch] is None else
                                     fc.load_pitches(args)["TaggedPitchType"]
                                     .isin(TYPE_MAP[known.pitch])))
    sub = sub.dropna(subset=sorted({t["col"] for t in terms}))
    sub = sub[(sub["year"] == 2025) & sub["PitcherId"].isin(f.index)]
    full = -cm.coach_score(sub, terms, 1.0)
    for t in sorted({t["col"] for t in terms}):
        part = -cm.coach_score(sub, [x for x in terms if x["col"] == t], 1.0)
        pv = pd.DataFrame({"f": full, "p": part, "id": sub["PitcherId"]}).groupby("id").mean()
        merged = pv.join(f[["ra9_next"]], how="inner")
        print(f"    {t:<22} r vs his score {fc.R(pv['f'], pv['p']):+.3f}   "
              f"r vs 2026 RA9 {fc.R(merged['p'], merged['ra9_next']):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
