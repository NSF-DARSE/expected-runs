"""Two-component blend (Stuff+ and Location+ only, no results) vs everything else.

WHY: our standing three-component Pitching+ (arsenal.py, owned by another session and
NOT touched here) equal-weight z-blends Stuff+, Location+, and adjusted results. On the
four-seam-only comparison the adjusted-results third has been measured elsewhere in this
project as INERT for predicting next season (tercile spread +0.04 runs/9, r=0.027), so
the three-component blend is carrying a dead third and mixes "what he did" into a
"how good are his pitches" number, which is also harder to explain to a coach.

This script defines an analysis-local two-component blend --

    pitch2_hi = (z(stuff_hi) + z(loc_hi)) / 2

-- and evaluates it head to head against velo_hi, coach_hi, stuff_hi, loc_hi, and the
existing pitch_hi (3-component blend), on BOTH standing criteria at BOTH sample floors
(51, 100 four-seams). pitch2_hi is defined here ONLY, per the hard constraint against
editing arsenal.py / coach_model_two_panel.py / coach_model_band_table.py. Weights are
equal by construction -- fitted blend weights have overfit three separate times in this
project's history (see FRAMEWORK.md); this script does not fit anything.

CRITERION 1 (coach_model_ff_criterion.build): next season's four-seam run value,
reported in three variants (Target/xT/adjT), at score+criterion floors 51 and 100. Reuses
ffc.build() unmodified; adds pitch2_hi to its output and reruns the tercile/bootstrap
machinery with an extended column set.

CRITERION 2: next season's RA9, thirds table via the cached pool
(coach_compare_pool.parquet) and a LOCAL variant of coach_model_two_panel.component_scores.
That function is not reused directly because it hardcodes n_ff >= 100 and this analysis
also needs floor 51; duplicating the ~15 lines that build stuff_hi/loc_hi/adjres_hi/coach_hi
here is the only way to get both floors without editing an owned file. The pitch-location
join, ridge, and coach-scorecard machinery are still imported, not reimplemented.

STRUCTURAL-ADVANTAGE FLAG (carried over from coach_model_ff_criterion.py): pitch_hi and
adjres_hi contain the GRADED season's own results, which on a run-value criterion gives
them a year-over-year reliability boost that is not predictive validity. That set is
printed with a [structural] / [contains results] tag on every table. pitch2_hi has no
such advantage -- it is built only from the physical pitch (Stuff+) and where it went
(Location+) -- so a win it posts is more meaningful than a loss to pitch_hi.

SIGN CONVENTION (fair_criterion.py): Target/xT/adjT/ridge_pred/location values are
expected runs from the pitcher's perspective, LOWER = better. RA9 is runs allowed,
LOWER = better. Every column ending `_hi` (including pitch2_hi) has already been
negated once into higher-is-better display frame; never negate twice. A well-sorting
column shows FEWER runs in "best third" and MORE in "worst third", so worst-minus-best
spread is POSITIVE when a score sorts correctly.

HONESTY: the working hypothesis is that dropping the dead adjusted-results third helps.
That is a hypothesis. If pitch2_hi does not beat Stuff+ alone, does not beat the coach's
card, or loses to pitch_hi, this script reports exactly that. Differences under about
1 SE are ties, called as ties, not ranked by point estimate.

Data rules: reads only via the constants/paths already defined in fair_criterion.py,
coach_model_ff_criterion.py, and coach_model_coach_units.py. Writes only into the
STUFFPLUS_WORKDIR (--workdir). No pitcher names anywhere. Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_band_table as bt
import coach_model_coach_units as cu
import coach_model_comparison as cm
import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import fair_criterion as fc

N_BOOT = 3000
FLOORS = [51, 100]

COLUMNS = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi", "pitch2_hi", "pitch_hi"]
LABELS = {**ffc.LABELS, "pitch2_hi": "Stuff+ & Location+ only"}
# adjres_hi is not in COLUMNS (the whole point is testing life without it), but pitch_hi
# still carries the graded season's own results through its third of the blend.
STRUCTURAL = {"pitch_hi"}
TERCILES = ["worst third", "middle", "best third"]  # qcut ascending, higher = better score
DIFF_TARGETS = ["stuff_hi", "pitch_hi", "coach_hi"]


def add_pitch2(g: pd.DataFrame) -> pd.DataFrame:
    """pitch2_hi: equal-weight z-blend of Stuff+ and Location+ only. Higher = better,
    consistent with every other _hi column. z-scored within whatever population `g`
    holds at call time (a floor-filtered pool), matching how pitch_hi is z-scored
    upstream."""
    g = g.copy()
    g["pitch2_hi"] = (cp.z(g["stuff_hi"].values) + cp.z(g["loc_hi"].values)) / 2
    return g


# ---------------- Criterion 1: fastball-only next-year run value ----------------

def one_criterion_ff(f: pd.DataFrame, variant: str, rng) -> dict:
    crit = f"crit100_{variant}"
    for c in COLUMNS:
        f[c + "_t"] = pd.qcut(pd.Series(cp.z(f[c].values), index=f.index), 3, labels=TERCILES)

    print(f"\n  --- criterion: 2026 four-seam {variant} "
          f"(runs per 100, lower = better; pool mean {f[crit].mean():+.2f}) ---")
    print(f"  {'band':<14}" + "".join(f"{LABELS[c]:>26}" for c in COLUMNS))
    for lab in ["best third", "middle", "worst third"]:
        line = f"  {lab:<14}"
        for c in COLUMNS:
            grp = f[f[c + "_t"] == lab]
            line += f"{grp[crit].mean():>20.2f} +/-{grp[crit].std() / np.sqrt(len(grp)):.2f}"
        print(line)

    print(f"  spread, worst minus best (bigger = sorts better), n={len(f)}:")
    spreads = {}
    for c in COLUMNS:
        sp = ffc.spread(f, c, crit)
        spreads[c] = sp
        tag = "   [structural]" if c in STRUCTURAL and variant != "Target" else ""
        print(f"    {LABELS[c]:<28}{sp:+.2f} runs/100{tag}")

    idx = f.index.values
    B = {c: [] for c in COLUMNS}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c in COLUMNS:
            s[c + "_t"] = pd.qcut(pd.Series(cp.z(s[c].values), index=s.index), 3,
                                  labels=TERCILES)
            B[c].append(ffc.spread(s, c, crit))
    B = {k: np.array(v) for k, v in B.items()}

    print("  paired bootstrap, Stuff+&Location+ blend minus each other score "
          "(same resamples):")
    vs = {}
    for other in DIFF_TARGETS:
        dd = B["pitch2_hi"] - B[other]
        note = " [structural]" if other in STRUCTURAL and variant != "Target" else ""
        fc.boot_report(f"{LABELS['pitch2_hi']} - {LABELS[other]}{note}", dd)
        vs[other] = dict(mean=round(float(dd.mean()), 4), se=round(float(dd.std()), 4),
                         p=round(float((dd > 0).mean()), 4),
                         structural=other in STRUCTURAL and variant != "Target",
                         tie=bool(abs(dd.mean()) < dd.std()))

    return dict(pool_mean=round(float(f[crit].mean()), 3),
                spreads={c: round(float(v), 3) for c, v in spreads.items()},
                vs_pitch2_hi=vs)


def eval_ff(floor: int) -> dict:
    print(f"\n=== Criterion 1: fastball-only next-year run value, floor {floor} ===")
    f = ffc.build(floor, floor)
    f = add_pitch2(f)
    rng = np.random.default_rng(41)
    by_criterion = {v: one_criterion_ff(f, v, rng) for v in ("Target", "xT", "adjT")}
    return dict(n=int(len(f)), floor=floor, by_criterion=by_criterion)


# ---------------- Criterion 2: next-year RA9 ----------------

def build_ra9_base(args) -> pd.DataFrame:
    """Pitcher-level scores (stuff_hi, loc_hi, adjres_hi, coach_hi, velo_hi, n_ff) joined
    to graded/next-season RA9 lines, WITHOUT a floor filter -- floors are applied per
    call in slice_floor() so the (slow) pitch load and (slower) line-CSV loads happen
    once. This mirrors coach_model_two_panel.component_scores plus
    coach_model_coach_units.graded_season_scores, kept local only because those hardcode
    n_ff >= 100 and this analysis needs 51 too (see module docstring)."""
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})
    df = fc.load_pitches(args)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)
    ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(ff)
    train = ff[(ff["year"] == 2024) & ff["xT"].notna()]
    ff["loc"] = fc.PooledLocationMap(train).apply(ff)
    ff = ff.dropna(subset=used).copy()
    ff["coach_raw"] = cm.coach_score(ff, terms, 1.0)

    season = ff[ff["year"] == 2025]
    g = season.groupby("PitcherId").agg(
        n_ff=("ridge_pred", "size"), ridge=("ridge_pred", "mean"), locv=("loc", "mean"),
        adj=("adjT", "mean"), coach_hi=("coach_raw", "mean"), velo_hi=("EffectiveVelo", "mean"))
    g["stuff_hi"] = -g["ridge"]
    g["loc_hi"] = -g["locv"]
    g["adjres_hi"] = -g["adj"]

    s25 = cu.line_stats(cu.RA9_2025, 2025)
    s26 = cu.line_stats(cu.RA9_2026, 2026)
    f = g.join(s25, how="inner").join(s26, how="inner", lsuffix="_graded", rsuffix="_next")
    f = f[(f["ip_graded"] >= cu.MIN_IP_GRADED) & (f["ip_next"] >= cu.MIN_IP_NEXT)]
    return f.dropna(subset=["ra9_graded", "ra9_next"]).copy()


def slice_floor(f_all: pd.DataFrame, floor: int) -> pd.DataFrame:
    """Apply the n_ff floor, then z-score pitch_hi/pitch2_hi WITHIN this floor's pool
    (matching how the upstream blends are z-scored relative to their own population)."""
    f = f_all[f_all["n_ff"] >= floor].copy()
    f["pitch_hi"] = sum(cp.z(f[c].values) for c in ["stuff_hi", "loc_hi", "adjres_hi"]) / 3
    return add_pitch2(f)


def tspread(d: pd.DataFrame, col: str) -> float:
    return (d.loc[d[col + "_t"] == "worst third", "ra9_next"].mean()
            - d.loc[d[col + "_t"] == "best third", "ra9_next"].mean())


def one_ra9(f: pd.DataFrame, rng) -> dict:
    for c in COLUMNS:
        f[c + "_t"] = pd.qcut(bt.display(f[c]), 3, labels=TERCILES)

    print(f"  {'band':<14}" + "".join(f"{LABELS[c]:>26}" for c in COLUMNS))
    for lab in ["best third", "middle", "worst third"]:
        line = f"  {lab:<14}"
        for c in COLUMNS:
            grp = f[f[c + "_t"] == lab]
            line += f"{grp.ra9_next.mean():>20.2f} +/-{grp.ra9_next.std() / np.sqrt(len(grp)):.2f}"
        print(line)

    print("  best-to-worst spread (bigger = sorts better):")
    spreads = {}
    for c in COLUMNS:
        sp = tspread(f, c)
        spreads[c] = sp
        tag = "  [contains graded-season results]" if c in STRUCTURAL else ""
        print(f"    {LABELS[c]:<28}{sp:+.2f} runs/9{tag}")

    idx = f.index.values
    B = {c: [] for c in COLUMNS}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c in COLUMNS:
            s[c + "_t"] = pd.qcut(bt.display(s[c]), 3, labels=TERCILES)
            B[c].append(tspread(s, c))
    B = {k: np.array(v) for k, v in B.items()}

    print("  paired bootstrap, Stuff+&Location+ blend minus each other score "
          "(same resamples):")
    vs = {}
    for other in DIFF_TARGETS:
        dd = B["pitch2_hi"] - B[other]
        note = " [contains graded-season results]" if other in STRUCTURAL else ""
        fc.boot_report(f"{LABELS['pitch2_hi']} - {LABELS[other]}{note}", dd)
        vs[other] = dict(mean=round(float(dd.mean()), 4), se=round(float(dd.std()), 4),
                         p=round(float((dd > 0).mean()), 4), structural=other in STRUCTURAL,
                         tie=bool(abs(dd.mean()) < dd.std()))

    return dict(spreads={c: round(float(v), 3) for c, v in spreads.items()}, vs_pitch2_hi=vs)


def eval_ra9(f_all: pd.DataFrame, floor: int) -> dict:
    f = slice_floor(f_all, floor)
    print(f"\n=== Criterion 2: next-year RA9, floor {floor} (n={len(f)}) ===")
    rng = np.random.default_rng(5)
    res = one_ra9(f, rng)
    res["n"] = int(len(f))
    res["floor"] = floor
    return res


def main() -> int:
    args = fc.paths()
    out = {"ff_criterion": {}, "ra9_criterion": {}}

    for floor in FLOORS:
        out["ff_criterion"][str(floor)] = eval_ff(floor)

    print("\nloading RA9 lines for both seasons (large files, minutes)...", flush=True)
    f_all = build_ra9_base(args)
    for floor in FLOORS:
        out["ra9_criterion"][str(floor)] = eval_ra9(f_all, floor)

    dest = os.path.join(args.workdir, "coach_model_two_blend.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
