"""Thirds table against a FASTBALL-ONLY criterion: next season's four-seam run value.

Fixes a mismatch in every earlier run today. The scores grade a pitcher's FOUR-SEAMS,
but the criterion was RA9 -- runs he allowed on every pitch he threw. For a pitcher who
throws 35% four-seams, most of that criterion is noise with respect to the thing being
measured, which suppresses every four-seam score and suppresses the narrowest ones most.

CRITERION, fixed BEFORE looking at any result (agreed 2026-08-12): mean adjT over the
pitcher's 2026 four-seams, expressed as RUNS PER 100 FOUR-SEAMS so it is readable.
RA9 is deliberately not reported here.

KNOWN LIMITATION, stated up front: adjT is our own run model. A fastball-only criterion
is better matched to the measurement but is no longer neutral between our score and an
external one. The RA9 version (neutral, diluted) lives in coach_model_band_table.py;
this is the targeted, non-neutral counterpart. Neither alone settles the comparison.

TWO COLUMNS ARE STRUCTURALLY ADVANTAGED HERE and are marked in the output:
  adjusted results  = this season's four-seam adjT, predicting next season's four-seam
                      adjT. That is year-over-year RELIABILITY of one measure, not
                      validity. RESULTS.md puts it near 0.268.
  Pitching+         = carries adjusted results as an equal third, so it inherits a
                      share of the same advantage.
Do not read either as evidence that our stack beats his card.

SIGN CONVENTION: adjT is expected runs from the pitcher's perspective, LOWER = BETTER,
and the criterion keeps that frame (a good third shows FEWER runs per 100). Scores are
in higher-is-better display frames (`_hi`). Nothing is negated twice.

Data rules: reads workdir caches only; writes JSON to the workdir, no names. Never
committed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import coach_model_comparison as cm
import coach_model_paired as cp
import fair_criterion as fc

# The three paths come from STUFFPLUS_DATA / STUFFPLUS_WORKDIR / STUFFPLUS_WORKDIR_CRIT via
# fc.workdirs(), which exits loudly on a missing one rather than defaulting to whoever built
# the script's machine. Point them at the RelSpeed-era caches (the extension gate's
# workdir_ext3 pair): as of the 2026-08-17 EffectiveVelo->RelSpeed swap, FEATS needs
# RelSpeed, which the older workdir_2425_d1 / workdir_2526_d1 caches do not carry.

COLUMNS = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi", "adjres_hi", "pitch_hi"]
LABELS = {"velo_hi": "Velo only", "coach_hi": "Coach's card", "stuff_hi": "Our Stuff+",
          "loc_hi": "Our Location+", "adjres_hi": "Our adj. results",
          "pitch_hi": "Our Pitching+"}
STRUCTURAL = {"adjres_hi", "pitch_hi"}
TERCILES = ["worst third", "middle", "best third"]  # qcut ascending, higher=better score
N_BOOT = 3000
RUNS_PER = 100  # criterion unit: runs per 100 four-seams


def _frame(data: str, workdir: str, years: str):
    """Load one build. Returns the FF frame with ridge_pred, adjT, loc, coach score."""
    import sys
    saved = sys.argv
    sys.argv = ["x", "--data", data, "--workdir", workdir, "--years", years, "--level", "D1"]
    args = fc.paths()
    sys.argv = saved
    df = fc.load_pitches(args)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)
    ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    ff["loc"] = fc.PooledLocationMap(ff[(ff["year"] == 2024) & ff["xT"].notna()]).apply(ff)
    return ff


def build(score_floor: int, crit_floor: int) -> pd.DataFrame:
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})
    data, score_workdir, crit_workdir = fc.workdirs()

    # scores: ridge trained on 2024, graded on real 2025 (eval role 2025 of this pair)
    s = _frame(data, score_workdir, "2024,2025").dropna(subset=used)
    s["coach_raw"] = cm.coach_score(s, terms, 1.0)
    # g_Target is the GRADED season's own realized four-seam run value. It is not a score;
    # it is carried so a "last year's actual results" reference column can be built from
    # the same pool. Additive -- existing consumers ignore it.
    g = s[s["year"] == 2025].groupby("PitcherId").agg(
        n=("ridge_pred", "size"), ridge=("ridge_pred", "mean"), locv=("loc", "mean"),
        adj=("adjT", "mean"), coach_hi=("coach_raw", "mean"), velo_hi=("EffectiveVelo", "mean"),
        g_Target=("Target", "mean"))
    g = g[g["n"] >= score_floor]
    g["stuff_hi"] = -g["ridge"]
    g["loc_hi"] = -g["locv"]
    g["adjres_hi"] = -g["adj"]
    g["pitch_hi"] = sum(cp.z(g[c].values) for c in ["stuff_hi", "loc_hi", "adjres_hi"]) / 3

    # criterion: real 2026 four-seams (eval role 2025 of the 2025/2026 pair), in three
    # variants so the luck-stripping claim itself is testable rather than assumed:
    #   Target = the run value that ACTUALLY happened. Not a model output, so it is the
    #            least contaminated criterion available -- and the noisiest.
    #   xT     = batted-ball luck stripped via the EV/LA map.
    #   adjT   = xT with opponent/league effects also removed.
    # If the adjustment only removes noise, the ORDERING of scores should survive on
    # Target and only the spreads should shrink.
    c = _frame(data, crit_workdir, "2025,2026")
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=("adjT", "size"), crit_adjT=("adjT", "mean"),
        crit_xT=("xT", "mean"), crit_Target=("Target", "mean"))
    k = k[k["n26"] >= crit_floor]
    j = g.join(k, how="inner")
    for v in ("Target", "xT", "adjT"):
        j[f"crit100_{v}"] = j[f"crit_{v}"] * RUNS_PER
    print(f"  scores {score_floor}+ FF in 2025: {len(g)};  criterion {crit_floor}+ FF in "
          f"2026: {len(k)};  joined: {len(j)}")
    return j


def spread(d: pd.DataFrame, col: str, crit: str) -> float:
    """Worst third minus best third, in runs per 100. Positive = sorts correctly."""
    return (d.loc[d[col + "_t"] == "worst third", crit].mean()
            - d.loc[d[col + "_t"] == "best third", crit].mean())


def one_criterion(f: pd.DataFrame, variant: str, rng) -> dict:
    crit = f"crit100_{variant}"
    print(f"\n  --- criterion: 2026 four-seam {variant} "
          f"(runs per 100, lower = better; pool mean {f[crit].mean():+.2f}) ---")
    print(f"  {'band':<14}" + "".join(f"{LABELS[c]:>22}" for c in COLUMNS))
    for lab in ["best third", "middle", "worst third"]:
        line = f"  {lab:<14}"
        for c in COLUMNS:
            grp = f[f[c + "_t"] == lab]
            line += f"{grp[crit].mean():>12.2f} +/-{grp[crit].std() / np.sqrt(len(grp)):.2f}"
        print(line)
    print(f"  spread, worst minus best (bigger = sorts better), n={len(f)}:")
    for c in COLUMNS:
        tag = "   [structural]" if c in STRUCTURAL and variant != "Target" else ""
        print(f"    {LABELS[c]:<18}{spread(f, c, crit):+.2f} runs/100{tag}")

    idx = f.index.values
    B = {c: [] for c in COLUMNS}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c in COLUMNS:
            s[c + "_t"] = pd.qcut(pd.Series(cp.z(s[c].values), index=s.index), 3,
                                  labels=TERCILES)
            B[c].append(spread(s, c, crit))
    B = {k: np.array(v) for k, v in B.items()}
    print("  paired bootstrap vs Our Stuff+ (same resamples):")
    for c in COLUMNS:
        if c != "stuff_hi":
            note = " [structural]" if c in STRUCTURAL and variant != "Target" else ""
            fc.boot_report(f"{LABELS[c]} - Our Stuff+{note}", B[c] - B["stuff_hi"])
    return dict(pool_mean=round(float(f[crit].mean()), 3),
                spreads={c: round(float(spread(f, c, crit)), 3) for c in COLUMNS},
                rows=[dict(band=lab, cells={
                    c: dict(n=int((f[c + "_t"] == lab).sum()),
                            value=round(float(f.loc[f[c + "_t"] == lab, crit].mean()), 3),
                            se=round(float(f.loc[f[c + "_t"] == lab, crit].std()
                                           / np.sqrt((f[c + "_t"] == lab).sum())), 3))
                    for c in COLUMNS}) for lab in ["best third", "middle", "worst third"]],
                vs_stuff={c: dict(mean=round(float((B[c] - B["stuff_hi"]).mean()), 3),
                                  se=round(float((B[c] - B["stuff_hi"]).std()), 3),
                                  structural=c in STRUCTURAL and variant != "Target")
                          for c in COLUMNS if c != "stuff_hi"})


def run(score_floor: int, crit_floor: int, label: str) -> dict:
    print(f"\n=== {label} ===")
    f = build(score_floor, crit_floor)
    for c in COLUMNS:
        f[c + "_t"] = pd.qcut(100 + 15 * pd.Series(cp.z(f[c].values), index=f.index),
                              3, labels=TERCILES)
    rng = np.random.default_rng(41)
    return dict(n=int(len(f)), score_floor=score_floor, crit_floor=crit_floor,
                by_criterion={v: one_criterion(f, v, rng) for v in ("Target", "xT", "adjT")})


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--floor", type=int, default=100)
    ap.add_argument("--sens", type=int, default=51)
    known, _ = ap.parse_known_args()

    out = {"primary": run(known.floor, known.floor, f"floor {known.floor} both seasons"),
           "sensitivity": run(known.sens, known.sens,
                              f"sensitivity: floor {known.sens} (measured n0) both seasons")}
    dest = os.path.join(fc.workdirs()[1], "coach_ff_criterion.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
