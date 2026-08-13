"""Rebuild a flat pitch extract from the game-date tree and diff it against
the hand-assembled extract.

The 2025 half of source_2025_2026_relspeed.csv came from a collaborator's CSV
that predates the RelSpeed column, so real velocity exists only for 2026. This
script rebuilds the same flat shape straight from the API pulls so the two can
be compared: row counts, date coverage, RelSpeed coverage, game/pitcher/pitch
overlap, and which columns exist on only one side.

Derived columns (Target, the break/velo differentials) are pipeline outputs,
not raw fields, so they are reported as expected-missing rather than diffed.

Usage:
    python trackman_api/rebuild_extract.py --tree <game-date dir> \
        --extract <path to source csv> --out <rebuilt csv> [--year 2025]

Data note: licensed Level II data in, licensed Level II data out. --out must
stay on local storage; this script prints aggregates only.
"""

from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

# Columns of the hand-assembled extract that the pipeline derives rather than
# reads from the raw game CSVs.
DERIVED_COLS = {"Target", "vertbreakdiff", "horzbreakdiff", "velocity_differential"}


def tree_files(tree: str, year: str | None) -> list[str]:
    pattern = os.path.join(tree, year or "*", "*", "*", "CSV", "*.csv")
    return sorted(f for f in glob.glob(pattern) if "_unverified" not in f)


def load_tree(files: list[str], columns: list[str]) -> pd.DataFrame:
    frames = []
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, low_memory=False)
        frames.append(df[[c for c in columns if c in df.columns]])
        if i % 1000 == 0:
            print(f"  read {i}/{len(files)} game files", flush=True)
    out = pd.concat(frames, ignore_index=True)
    # Keep-first dedup on PitchUID, matching the pipeline's own rule.
    return out.drop_duplicates(subset="PitchUID", keep="first")


def summarize(name: str, df: pd.DataFrame) -> None:
    print(f"\n[{name}] rows={len(df):,} "
          f"dates={df.Date.min()}..{df.Date.max()} "
          f"games={df.GameID.nunique():,} pitchers={df.PitcherId.nunique():,}")
    if "RelSpeed" in df.columns:
        n = int(df.RelSpeed.notna().sum())
        print(f"[{name}] RelSpeed present on {n:,} rows ({n / max(len(df), 1):.1%})")
    print(f"[{name}] Level: {df.Level.value_counts().to_dict()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True)
    p.add_argument("--extract", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--year", help="Restrict to one season year folder, e.g. 2025")
    args = p.parse_args()

    extract_cols = pd.read_csv(args.extract, nrows=0).columns.tolist()
    raw_cols = [c for c in extract_cols if c not in DERIVED_COLS]

    files = tree_files(args.tree, args.year)
    print(f"game files in tree: {len(files):,}")
    rebuilt = load_tree(files, raw_cols)
    if args.year:
        rebuilt = rebuilt[rebuilt.Date.astype(str).str[:4] == args.year]
    rebuilt.to_csv(args.out, index=False)
    summarize("rebuilt", rebuilt)

    keep = [c for c in extract_cols if c in ("Date", "GameID", "PitcherId",
                                             "PitchUID", "Level", "RelSpeed")]
    chunks = []
    for ch in pd.read_csv(args.extract, usecols=keep, low_memory=False,
                          chunksize=500_000):
        if args.year:
            ch = ch[ch.Date.astype(str).str[:4] == args.year]
        chunks.append(ch)
    extract = pd.concat(chunks, ignore_index=True)
    summarize("extract", extract)

    only_extract = [c for c in extract_cols if c not in rebuilt.columns]
    only_rebuilt = [c for c in rebuilt.columns if c not in extract_cols]
    print(f"\ncolumns only in extract: {only_extract}")
    print(f"columns only in rebuilt: {only_rebuilt}")

    a, e = set(rebuilt.GameID), set(extract.GameID)
    print(f"\ngames: rebuilt={len(a):,} extract={len(e):,} "
          f"extract-only={len(e - a):,} rebuilt-only={len(a - e):,}")
    pa, pe = set(rebuilt.PitchUID), set(extract.PitchUID)
    print(f"pitches: extract-only={len(pe - pa):,} rebuilt-only={len(pa - pe):,}")

    for level in ("D1",):
        ra = rebuilt[rebuilt.Level == level]
        ea = extract[extract.Level == level]
        print(f"{level}: rebuilt rows={len(ra):,} pitchers={ra.PitcherId.nunique():,} | "
              f"extract rows={len(ea):,} pitchers={ea.PitcherId.nunique():,}")


if __name__ == "__main__":
    main()
