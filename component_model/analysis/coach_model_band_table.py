"""Band table: sort 2025 pitchers by each score, show what they allowed in 2026.

The coach-facing form of the comparison. No correlations, no runs-per-SD: just
"pitchers who graded here went on to allow this much." The column that sorts best is
the one whose runs climb steadily from the top band to the bottom.

Each score is put on the same 100 +/- 15 display scale (z over this pool) so a band
means the same thing in every column. A given pitcher lands in different rows in
different columns -- that IS the comparison.

Reads as: of the pitchers whose 2025 four-seam graded 120-130 on this score, here is
their mean 2026 RA9. Cell counts are shown because the end bands are thin, and a mean
over eight pitchers is not a trend.

SIGN CONVENTION: scores are higher-is-better (`_hi`) and display as 100 + 15z. RA9 is
runs allowed, LOWER = BETTER, so a well-sorting column runs LOW at the top band and
HIGH at the bottom.

Data rules: reads the cached pool and workdir parquet; writes table JSON to the workdir
only, with no pitcher names. Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_paired as cp
import coach_model_two_panel as tp
import fair_criterion as fc

COLUMNS = ["velo_hi", "coach_hi", "stuff_hi", "adjres_hi", "pitch_hi"]
EDGES = [130, 120, 110, 100, 90, 80]
MIN_CELL = 10  # below this a cell mean is reported but marked thin
N_BOOT = 3000
TERCILES = ["worst third", "middle", "best third"]  # qcut ascending on higher=better

# Pitching+ and adjusted results contain the graded season's RESULTS. In the regression
# tests that made them uninterpretable, because RA9 was a control (see
# coach_model_two_panel.py). Here there is NO control: the question is only "does this
# score sort pitchers by next season's runs?", which is a fair question for a
# results-containing score. Flagged so the two groups are never described as measuring
# the same thing -- these are not pitch-quality scores.
CONTAINS_RESULTS = {"adjres_hi", "pitch_hi"}


def display(score: pd.Series) -> pd.Series:
    """100 +/- 15, higher = better. Affine, so band boundaries are just z cutoffs."""
    return 100 + 15 * pd.Series(cp.z(score.values), index=score.index)


def band_label(i: int) -> str:
    if i == 0:
        return f"{EDGES[0]}+"
    if i == len(EDGES):
        return f"under {EDGES[-1]}"
    return f"{EDGES[i]}-{EDGES[i - 1]}"


def assign_band(v: pd.Series) -> pd.Series:
    """Band index, 0 = top. EDGES is descending, so walking it upward from the lowest
    edge and overwriting leaves each pitcher in the highest band he clears."""
    out = pd.Series(len(EDGES), index=v.index, dtype=int)
    for i in range(len(EDGES) - 1, -1, -1):
        out[v >= EDGES[i]] = i
    return out


def main() -> int:
    args = fc.paths()
    pool = pd.read_parquet(os.path.join(args.workdir, "coach_compare_pool.parquet"))
    f = tp.component_scores(args, pool)
    print(f"pool: {len(f)} pitchers graded on 2025 four-seams, 2026 follow-up")
    print(f"pool mean 2026 RA9: {f.ra9_next.mean():.2f}\n")

    for c in COLUMNS:
        f[c + "_disp"] = display(f[c])
        f[c + "_band"] = assign_band(f[c + "_disp"])

    rows, payload_rows = [], []
    for i in range(len(EDGES) + 1):
        cells = []
        for c in COLUMNS:
            grp = f[f[c + "_band"] == i]
            if len(grp):
                cells.append(dict(n=len(grp), ra9=float(grp.ra9_next.mean()),
                                  se=float(grp.ra9_next.std() / np.sqrt(len(grp)))
                                  if len(grp) > 1 else float("nan"),
                                  thin=len(grp) < MIN_CELL))
            else:
                cells.append(None)
        rows.append((band_label(i), cells))
        payload_rows.append(dict(band=band_label(i),
                                 cells={c: cell for c, cell in zip(COLUMNS, cells)}))

    hdr = f"  {'2025 grade band':<18}" + "".join(
        f"{tp.LABELS[c]:>22}" for c in COLUMNS)
    print(hdr)
    print(f"  {'':<18}" + "".join(f"{'2026 RA9 (n)':>22}" for _ in COLUMNS))
    for label, cells in rows:
        line = f"  {label:<18}"
        for cell in cells:
            if cell is None:
                line += f"{'--':>22}"
            else:
                mark = "*" if cell["thin"] else " "
                line += f"{cell['ra9']:>16.2f}{mark}({cell['n']:>3})"
        print(line)
    print(f"\n  * fewer than {MIN_CELL} pitchers -- read as an individual, not a trend")

    print("\n  top-band minus bottom-band spread (bigger = sorts better):")
    spreads = {}
    for j, c in enumerate(COLUMNS):
        filled = [cells[j] for _, cells in rows if cells[j] is not None]
        if len(filled) >= 2:
            spread = filled[-1]["ra9"] - filled[0]["ra9"]
            spreads[c] = spread
            print(f"    {tp.LABELS[c]:<18}{spread:+.2f} runs/9")

    # Monotonicity: how often does the next band down allow MORE runs? A well-behaved
    # column steps the same direction every time. Reported because a big top-to-bottom
    # spread can hide a non-monotonic middle, which is what a coach would notice first.
    print("\n  steps in the right direction (band to band, runs going up):")
    for j, c in enumerate(COLUMNS):
        seq = [cells[j]["ra9"] for _, cells in rows if cells[j] is not None]
        ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
        print(f"    {tp.LABELS[c]:<18}{ups}/{len(seq) - 1}")

    # ---- terciles: the presentable version. Fixed-width bands above are unstable at
    # the ends (a 4-pitcher cell drove a spurious ranking once), so thirds with a
    # standard error are what actually goes in front of anyone.
    for c in COLUMNS:
        f[c + "_t"] = pd.qcut(f[c + "_disp"], 3, labels=TERCILES)

    print("\n\n=== THIRDS (the presentable form) ===")
    print(f"  {'band':<14}" + "".join(f"{tp.LABELS[c]:>24}" for c in COLUMNS))
    for lab in ["best third", "middle", "worst third"]:
        line = f"  {lab:<14}"
        for c in COLUMNS:
            grp = f[f[c + "_t"] == lab]
            line += (f"{grp.ra9_next.mean():>14.2f} +/-"
                     f"{grp.ra9_next.std() / np.sqrt(len(grp)):.2f} ({len(grp):>3})")
        print(line)

    def tspread(d: pd.DataFrame, col: str) -> float:
        return (d.loc[d[col + "_t"] == "worst third", "ra9_next"].mean()
                - d.loc[d[col + "_t"] == "best third", "ra9_next"].mean())

    print("\n  best-to-worst spread (bigger = sorts better):")
    for c in COLUMNS:
        tag = "  [contains results, not a pitch-quality score]" if c in CONTAINS_RESULTS else ""
        print(f"    {tp.LABELS[c]:<18}{tspread(f, c):+.2f} runs/9{tag}")

    rng = np.random.default_rng(5)
    idx = f.index.values
    B = {c: [] for c in COLUMNS}
    for _ in range(N_BOOT):
        s = f.loc[rng.choice(idx, len(idx))]
        for c in COLUMNS:
            s[c + "_t"] = pd.qcut(bt_display_local(s, c), 3, labels=TERCILES)
            B[c].append(tspread(s, c))
    B = {k: np.array(v) for k, v in B.items()}
    print("\n  paired bootstrap on spread differences vs Our Stuff+ (same resamples):")
    for c in COLUMNS:
        if c != "stuff_hi":
            fc.boot_report(f"{tp.LABELS[c]} - Our Stuff+", B[c] - B["stuff_hi"])

    tercile_rows = [dict(band=lab, cells={
        c: dict(n=int((f[c + "_t"] == lab).sum()),
                ra9=round(float(f.loc[f[c + "_t"] == lab, "ra9_next"].mean()), 3),
                se=round(float(f.loc[f[c + "_t"] == lab, "ra9_next"].std()
                               / np.sqrt((f[c + "_t"] == lab).sum())), 3))
        for c in COLUMNS}) for lab in ["best third", "middle", "worst third"]]

    dest = os.path.join(args.workdir, "coach_band_table.json")
    with open(dest, "w") as fh:
        json.dump(dict(n=len(f), pool_ra9_next=round(float(f.ra9_next.mean()), 3),
                       columns=[dict(key=c, label=tp.LABELS[c],
                                     contains_results=c in CONTAINS_RESULTS)
                                for c in COLUMNS],
                       rows=payload_rows, spreads={k: round(v, 3) for k, v in spreads.items()},
                       terciles=tercile_rows,
                       tercile_spreads={c: round(float(tspread(f, c)), 3) for c in COLUMNS},
                       min_cell=MIN_CELL), fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


def bt_display_local(s: pd.DataFrame, col: str) -> pd.Series:
    """Re-standardise inside a bootstrap resample, so bands are resample-relative."""
    return display(s[col])


if __name__ == "__main__":
    raise SystemExit(main())
