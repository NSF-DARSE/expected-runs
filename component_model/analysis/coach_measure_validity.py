"""Which run-value MEASURE best predicts next season's realized runs: T, xT, or adjT?

This validates the CRITERION, not any Stuff+ model. It is the design that validated
xFIP/SIERA against future ERA, and its virtue over everything else run today is that the
criterion -- next season's raw `Target` -- is GROUND TRUTH rather than a model output, so
the circularity that undercut the model-vs-model comparisons does not apply. All three
predictors face identical criterion noise, so the comparison between them is fair.

THE LADDER, and why xT is the interesting rung:
  Target  realized run value. Ground truth, includes fielding and bounces.
  xT      batted balls replaced by the mean run value of their EV/LA bucket. Strips
          defense and luck ONLY. Opponent/league quality is still in it.
  adjT    xT minus league mean minus a shrunk batter effect. Also strips opponent.
Opponent quality is STICKY year over year (same conference, similar schedule), so adjT
discards information that genuinely helps predict next season. Expect xT to win, and adjT
to land between xT and Target or possibly below Target. That would NOT be a mark against
adjT as a skill measure; it is the two goals coming apart (best forecast of future
outcomes keeps sticky context; best measure of a pitcher's own contribution removes it).
If adjT beats Target anyway, that is a STRONGER result, not a weaker one.

WHAT THIS CANNOT SHOW, stated up front:
  1. Nothing about v1 vs v2. It compares measures, not models.
  2. Not that the EV/LA grid is UNBIASED. A less noisy measure correlates better with any
     future realization almost automatically, and a smooth-but-biased measure can beat a
     noisy unbiased one. Complementary to a by-arsenal-type calibration check, not a
     substitute for it.

SIGN CONVENTION -- differs from the `_hi` score columns and this is where a table got
inverted once today. Target/xT/adjT are expected runs from the PITCHER's perspective,
LOWER = BETTER, and they are used here UNNEGATED as predictors. Consequences:
  - A better measure has a LARGER POSITIVE correlation with next season's Target (high
    graded runs go with high next-season runs).
  - pd.qcut assigns labels in ASCENDING value order, so on a lower-is-better column the
    FIRST label is the BEST third. That is the opposite of every `_hi` table in this
    directory. Labels are spelled out in MEASURE_TERCILES to keep it explicit.
  - Spread is still worst-third minus best-third of the next-season criterion, so bigger
    = sorts better, matching the rest of the suite.

Data rules: reads workdir caches; writes one JSON to the workdir. No pitcher names.
Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import fair_criterion as fc

MEASURES = ["Target", "xT", "adjT"]
LABELS = {"Target": "Raw results", "xT": "Luck and defense removed",
          "adjT": "Luck, defense and opponent removed"}
# ascending qcut on a LOWER-IS-BETTER predictor: first label is the best third
MEASURE_TERCILES = ["best third", "middle", "worst third"]
N_BOOT = 4000
RUNS_PER = 100


def build(floor: int) -> pd.DataFrame:
    """One row per pitcher: graded-season (2025) mean T/xT/adjT, and next-season (2026)
    mean T/xT/adjT, both in runs per 100 four-seams."""
    s = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    g = s[s["year"] == 2025].groupby("PitcherId").agg(
        n=("Target", "size"), g_Target=("Target", "mean"),
        g_xT=("xT", "mean"), g_adjT=("adjT", "mean"))

    c = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=("Target", "size"), n_Target=("Target", "mean"),
        n_xT=("xT", "mean"), n_adjT=("adjT", "mean"))

    j = g.join(k, how="inner")
    j = j[(j["n"] >= floor) & (j["n26"] >= floor)].copy()
    for m in MEASURES:
        j["g_" + m] *= RUNS_PER
        j["n_" + m] *= RUNS_PER
    print(f"  floor {floor}: graded {len(g)} -> joined+floored {len(j)}")
    return j


def spread(d: pd.DataFrame, m: str) -> float:
    """Worst-third minus best-third of NEXT season's raw Target. Bigger = sorts better."""
    t = d[m + "_t"]
    return (d.loc[t == "worst third", "n_Target"].mean()
            - d.loc[t == "best third", "n_Target"].mean())


def add_terciles(d: pd.DataFrame) -> pd.DataFrame:
    for m in MEASURES:
        # NOT negated: lower runs = better, so ascending qcut puts the best third first
        d[m + "_t"] = pd.qcut(d["g_" + m], 3, labels=MEASURE_TERCILES)
    return d


def main() -> int:
    out = {}
    for floor in (51, 100):
        print(f"\n=== floor {floor} ===")
        f = add_terciles(build(floor))

        print("  predictive validity: graded measure vs NEXT season's raw Target")
        print("  (r positive and larger = better; both are runs, lower = better)")
        point = {}
        for m in MEASURES:
            r = fc.R(f["g_" + m], f["n_Target"])
            sp = spread(f, m)
            point[m] = dict(r=round(float(r), 4), spread=round(float(sp), 3),
                            reliability=round(float(fc.R(f["g_" + m], f["n_" + m])), 4))
            print(f"    {LABELS[m]:<38} r={r:+.3f}   thirds spread {sp:+.2f} runs/100"
                  f"   own year-to-year r={point[m]['reliability']:+.3f}")

        rng = np.random.default_rng(73)
        idx = f.index.values
        B = {m: {"r": [], "sp": []} for m in MEASURES}
        for _ in range(N_BOOT):
            s = f.loc[rng.choice(idx, len(idx))]
            for m in MEASURES:
                s[m + "_t"] = pd.qcut(s["g_" + m], 3, labels=MEASURE_TERCILES)
                B[m]["r"].append(float(fc.R(s["g_" + m], s["n_Target"])))
                B[m]["sp"].append(float(spread(s, m)))
        B = {m: {k: np.array(v) for k, v in d.items()} for m, d in B.items()}

        print("  paired bootstrap on differences (same resamples):")
        pairs = {}
        for a, b in (("xT", "Target"), ("adjT", "Target"), ("xT", "adjT")):
            dr, ds = B[a]["r"] - B[b]["r"], B[a]["sp"] - B[b]["sp"]
            fc.boot_report(f"r:      {a} - {b}", dr)
            fc.boot_report(f"spread: {a} - {b}", ds)
            pairs[f"{a}-{b}"] = dict(
                r=dict(mean=round(float(dr.mean()), 4), se=round(float(dr.std()), 4),
                       p_gt0=round(float((dr > 0).mean()), 4)),
                spread=dict(mean=round(float(ds.mean()), 3), se=round(float(ds.std()), 3),
                            p_gt0=round(float((ds > 0).mean()), 4)))

        rows = []
        for lab in MEASURE_TERCILES:
            cells = {}
            for m in MEASURES:
                grp = f.loc[f[m + "_t"] == lab, "n_Target"]
                cells[m] = dict(n=int(len(grp)), mean=round(float(grp.mean()), 3),
                                se=round(float(grp.std() / np.sqrt(len(grp))), 3))
            rows.append(dict(band=lab, cells=cells))

        print(f"  next-season raw Target by graded third (runs/100, lower = better):")
        print(f"    {'band':<14}" + "".join(f"{LABELS[m]:>30}" for m in MEASURES))
        for r in rows:
            line = f"    {r['band']:<14}"
            for m in MEASURES:
                line += f"{r['cells'][m]['mean']:>20.2f} +/-{r['cells'][m]['se']:.2f}"
            print(line)

        out[str(floor)] = dict(n=int(len(f)), floor=floor, point=point, pairs=pairs,
                               rows=rows,
                               pool_next_mean=round(float(f["n_Target"].mean()), 3))

    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_measure_validity.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
