"""Emit the complete cell-level tables the coach-facing page needs, in one pass.

The existing scripts print their thirds tables and persist only the SPREADS, so the
per-cell means/SEs a rendered table needs were lost when the process exited. This
rebuilds them once and writes everything the page reads, so the HTML never carries a
hand-transcribed number.

COLUMN SET is deliberately narrower than the analysis scripts': velo / coach card /
Stuff+ / Location+ / the two-component blend. `adjres_hi` and the three-component
`pitch_hi` are omitted because the two-component blend was measured as their equal or
better (coach_model_two_blend.py) and because a score containing the graded season's own
results is not a pitch-quality score and confuses the thing being explained.

EFFICIENCY: ffc.build(51, 51) is called ONCE and the floor-100 pool is derived by
filtering its `n`/`n26` columns, which is equivalent to build(100, 100) -- same join,
same source -- while halving the heavy parquet loads. Terciles and z-scores are
recomputed per floor, as they must be.

SIGN CONVENTION (fair_criterion.py): Target/xT/adjT and location values are expected
runs from the PITCHER's perspective, LOWER = BETTER. Columns ending `_hi` are already
negated once into higher-is-better. Criterion values are run value RELATIVE TO AVERAGE
per 100 pitches (pool means sit near zero by construction), so the page must say
"vs average", never "runs allowed".

Data rules: reads workdir caches; writes one JSON to the workdir. No pitcher names.
Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_band_table as bt
import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import coach_model_two_blend as tb
import fair_criterion as fc

# prior_hi is NOT a pitch-quality score: it is the graded season's own realized results,
# carried as the reference a grade has to beat. Negated into higher-is-better like every
# other `_hi` column so the shared tercile/spread/bootstrap machinery applies unchanged.
# Ordered immediately before the blend so the comparison that matters is adjacent.
COLUMNS = ["velo_hi", "coach_hi", "stuff_hi", "loc_hi", "prior_hi", "pitch2_hi"]
LABELS = {"velo_hi": "Velocity only", "coach_hi": "Coach's card",
          "stuff_hi": "Stuff+", "loc_hi": "Location+",
          "prior_hi": "Last year's actual runs",
          "pitch2_hi": "Stuff+ and Location+"}
BANDS = ["best third", "middle", "worst third"]
N_BOOT = 3000


def cells(f: pd.DataFrame, value_col: str) -> dict:
    """Per-column, per-third n / mean / SE of `value_col`, plus worst-minus-best spread."""
    out = {}
    for c in COLUMNS:
        rows = {}
        for lab in BANDS:
            g = f.loc[f[c + "_t"] == lab, value_col]
            rows[lab] = dict(n=int(len(g)), mean=round(float(g.mean()), 3),
                             se=round(float(g.std() / np.sqrt(len(g))), 3))
        out[c] = dict(bands=rows,
                      spread=round(float(rows["worst third"]["mean"]
                                         - rows["best third"]["mean"]), 3))
    return out


def grid(f: pd.DataFrame, row_key: str, col_key: str, value_cols) -> dict:
    """3x3 cross-tab of two scores' terciles with the mean criterion in each cell.

    Both scores are higher-is-better and the criterion is lower-is-better, so the
    best cell is (best third, best third) and should read most negative. Cells carry
    their n because a 3x3 over ~800 pitchers leaves roughly 90 per cell, and a mean
    over fewer than 15 is an anecdote rather than a trend.
    """
    out = {"rows": ffc.TERCILES, "cols": ffc.TERCILES, "cells": {}}
    for r in ffc.TERCILES:
        for c in ffc.TERCILES:
            m = (f[row_key + "_t"] == r) & (f[col_key + "_t"] == c)
            g = f.loc[m]
            cell = dict(n=int(len(g)), thin=bool(len(g) < 15))
            for vc in value_cols:
                cell[vc] = (round(float(g[vc].mean()), 3) if len(g) else None)
            out["cells"][f"{r}|{c}"] = cell
    return out


def terciles(f: pd.DataFrame) -> pd.DataFrame:
    for c in COLUMNS:
        f[c + "_t"] = pd.qcut(pd.Series(cp.z(f[c].values), index=f.index), 3,
                              labels=ffc.TERCILES)
    return f


def paired_vs(f: pd.DataFrame, value_col: str, seed: int) -> tuple[dict, dict]:
    """Paired bootstrap over the same resamples. Returns (differences, per-column spread).

    The per-column spread bootstrap is what the page's error bars need: a bar drawn
    without it implies an ordering the data does not support. The difference bootstrap is
    still the thing that decides a win, because the two spreads are positively correlated
    across resamples and SEs read off individual bars therefore OVERSTATE the uncertainty
    of their difference. Both are emitted so neither is inferred from the other."""
    rng = np.random.default_rng(seed)
    idx = f.index.values
    B = {c: [] for c in COLUMNS}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c in COLUMNS:
            s[c + "_t"] = pd.qcut(pd.Series(cp.z(s[c].values), index=s.index), 3,
                                  labels=ffc.TERCILES)
            g = s.groupby(c + "_t", observed=True)[value_col].mean()
            B[c].append(float(g.get("worst third", np.nan) - g.get("best third", np.nan)))
    B = {k: np.array(v) for k, v in B.items()}
    out = {}
    for c in COLUMNS:
        if c == "pitch2_hi":
            continue
        d = B["pitch2_hi"] - B[c]
        out[c] = dict(mean=round(float(d.mean()), 3), se=round(float(d.std()), 3),
                      p_gt0=round(float((d > 0).mean()), 3),
                      tie=bool(abs(d.mean()) < d.std()))
    per_col = {c: dict(mean=round(float(B[c].mean()), 3), se=round(float(B[c].std()), 3),
                       lo=round(float(np.percentile(B[c], 2.5)), 3),
                       hi=round(float(np.percentile(B[c], 97.5)), 3))
               for c in COLUMNS}
    return out, per_col


def main() -> int:
    args = fc.paths()
    out = {"columns": [dict(key=c, label=LABELS[c]) for c in COLUMNS], "ff": {}, "ra9": {}}

    print("building fastball criterion pool once at floor 51...", flush=True)
    base = tb.add_pitch2(ffc.build(51, 51))
    # graded-season realized four-seam run value, negated so higher = better
    base["prior_hi"] = -base["g_Target"]

    for floor in (51, 100):
        sub = base[(base["n"] >= floor) & (base["n26"] >= floor)].copy()
        # re-derive the blend inside this floor's population, as the analysis scripts do
        sub = tb.add_pitch2(sub)
        sub = terciles(sub)
        entry = {"n": int(len(sub)), "by_criterion": {}}
        for variant in ("Target", "adjT"):
            col = f"crit100_{variant}"
            vs, per_col = paired_vs(sub, col, seed=41 + floor)
            entry["by_criterion"][variant] = dict(
                pool_mean=round(float(sub[col].mean()), 3),
                table=cells(sub, col), vs_blend=vs, spread_boot=per_col)
            print(f"  floor {floor} {variant}: n={len(sub)} done", flush=True)
        # 3x3 grids: Location+ against each Stuff+ version. The v1 grid is what carries
        # the "the scorecard cannot see location" point; the v2 grid shows our own two
        # components are complementary. Same format so they are directly comparable.
        vcols = ["crit100_Target", "crit100_adjT"]
        entry["grids"] = {
            "stuff_x_loc": grid(sub, "stuff_hi", "loc_hi", vcols),
            "coach_x_loc": grid(sub, "coach_hi", "loc_hi", vcols),
        }
        out["ff"][str(floor)] = entry

    print("building RA9 pool (large line files, minutes)...", flush=True)
    f_all = tb.build_ra9_base(args)
    for floor in (51, 100):
        sub = tb.slice_floor(f_all, floor)
        # RA9 counterpart of the same reference: last season's own runs allowed per 9
        sub["prior_hi"] = -sub["ra9_graded"]
        sub = terciles(sub)
        vs, per_col = paired_vs(sub, "ra9_next", seed=5 + floor)
        out["ra9"][str(floor)] = dict(
            n=int(len(sub)), pool_mean=round(float(sub["ra9_next"].mean()), 3),
            table=cells(sub, "ra9_next"), vs_blend=vs, spread_boot=per_col)
        print(f"  RA9 floor {floor}: n={len(sub)} done", flush=True)

    dest = os.path.join(args.workdir, "coach_page_data.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
