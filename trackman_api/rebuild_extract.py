"""Rebuild a flat pitch extract from the game-date tree(s) and diff it against
the hand-assembled extract.

The 2025 half of source_2025_2026_relspeed.csv came from a collaborator's CSV
that predates the RelSpeed column, so real velocity exists only for 2026. This
script rebuilds the same flat shape straight from the API pulls so the two can
be compared: row counts, date coverage, RelSpeed coverage, game/pitcher/pitch
overlap, and which columns exist on only one side.

Derived columns (Target, the break/velo differentials) are pipeline outputs,
not raw fields, so they are reported as expected-missing rather than diffed.

IMPORTANT (folder-vs-game-date mismatch): the on-disk trees are organized by
FETCH date, not game date. A meaningful number of files under the 2026 folder
contain 2025 (or 2024) games, and some files under 2025 contain 2024 games. A
naive `--year 2025` run that globs only the 2025 folder silently drops those
misfiled games. This script therefore ALWAYS globs every year folder handed to
it via --tree (plus the optional --sftp-tree) and partitions rows into season
buckets using the actual `Date` column read from each file, never the folder
a file happens to live in. Every game file observed here contains exactly one
game date, so partitioning by the first Date value per file is equivalent to
partitioning by row and much cheaper.

Multiple source trees can be combined (e.g. the API pull tree and a separate
SFTP pull tree that partially overlaps it) because rows are deduplicated on
PitchUID (keep-first, in the order trees/files are processed) after all trees
are read for a season.

Memory handling: this does NOT concatenate every file across all seasons into
one DataFrame. Files are bucketed by season first (cheap, header-only-ish
reads restricted to needed columns), then each season is processed as its own
batch: read in file-count chunks, deduped, and written straight to its own
per-season output CSV before the next season starts. Peak memory is therefore
one season's worth of raw rows at a time, not the whole multi-year corpus.

Usage:
    python trackman_api/rebuild_extract.py \
        --tree C:/Users/jackdav/repos/baseball-stuff-plus/trackman_api \
        --sftp-tree C:/Users/jackdav/trackman_games \
        --extract C:/Users/jackdav/stuffplus_replication/source_2025_2026_relspeed.csv \
        --out-dir C:/Users/jackdav/stuffplus_replication/rebuilt \
        --seasons 2025 2026

Data note: licensed Level II data in, licensed Level II data out. --out-dir
must stay on local storage; this script prints aggregates only, never raw
rows or pitcher names.
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict

import pandas as pd

# Columns of the hand-assembled extract that the pipeline derives rather than
# reads from the raw game CSVs.
DERIVED_COLS = {"Target", "vertbreakdiff", "horzbreakdiff", "velocity_differential"}

FILE_BATCH_SIZE = 1500  # files read into memory per incremental-write batch


def tree_files(tree: str) -> list[str]:
    """Every game CSV under a tree, across ALL year folders, skipping
    _unverified pulls (matching the pipeline's existing rule: an unverified
    pull is superseded by its verified counterpart when one exists)."""
    pattern = os.path.join(tree, "*", "*", "*", "CSV", "*.csv")
    return sorted(f for f in glob.glob(pattern) if "_unverified" not in f)


def file_season(path: str) -> str | None:
    """Read a file's Date column (not just row 1) and return its game-date
    year. Every file here is a single game, so almost all rows share one
    Date value, but a meaningful number of files carry one or more leading
    or scattered rows with a blank Date (a stray artifact row, not a second
    game) -- reading only nrows=1, as an earlier version of this function
    did, misclassified 9 of 11 test files as fully unreadable when in fact
    >95% of their rows had a perfectly good, consistent Date. Only treat a
    file as unreadable when EVERY row lacks a Date."""
    try:
        d = pd.read_csv(path, usecols=["Date"], low_memory=False)
    except (ValueError, pd.errors.EmptyDataError):
        return None
    dates = d.Date.dropna()
    if dates.empty:
        return None
    return str(dates.mode().iloc[0])[:4]


def bucket_files_by_season(files: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    unreadable = []
    for i, f in enumerate(files, 1):
        season = file_season(f)
        if season is None:
            unreadable.append(f)
        else:
            buckets[season].append(f)
        if i % 5000 == 0:
            print(f"  bucketed {i}/{len(files)} files by season", flush=True)
    if unreadable:
        print(f"  WARNING: {len(unreadable)} files unreadable / no Date value, "
              f"excluded from all seasons:")
        for f in unreadable[:20]:
            print(f"    {f}")
        if len(unreadable) > 20:
            print(f"    ... and {len(unreadable) - 20} more")
    return buckets


def load_files(files: list[str], columns: list[str], seen_uids: set) -> pd.DataFrame:
    """Read files in batches, restrict to the requested columns, and dedup on
    PitchUID keep-first against everything already seen for this season
    (across trees). Returns only the NEW rows for this batch of files."""
    frames = []
    total_dropped_no_key = 0
    for start in range(0, len(files), FILE_BATCH_SIZE):
        batch = files[start:start + FILE_BATCH_SIZE]
        batch_frames = []
        for f in batch:
            try:
                df = pd.read_csv(f, low_memory=False)
            except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
                print(f"  WARNING: failed to read {f}: {e}")
                continue
            df = df[[c for c in columns if c in df.columns]]
            # A handful of raw files carry trailing/blank rows with no
            # PitchUID or Date (not real pitches). Drop them here so mixed
            # str/NaN dtypes never reach downstream min/max or dedup logic.
            before = len(df)
            df = df.dropna(subset=[c for c in ("PitchUID", "Date") if c in df.columns])
            total_dropped_no_key += before - len(df)
            # Some individual game files have no missing PitcherId (parsed
            # int64) while others do (parsed float64, e.g. "814215.0"). Left
            # unnormalized, concatenating batches of files with different
            # per-batch dtypes, then round-tripping through CSV, turns one
            # physical PitcherId into two distinct string values ("123" vs
            # "123.0") and silently inflates pitcher counts. Force a single
            # nullable-integer dtype per file, before any concatenation.
            if "PitcherId" in df.columns:
                df["PitcherId"] = pd.to_numeric(df["PitcherId"], errors="coerce").astype("Int64")
            batch_frames.append(df)
        if not batch_frames:
            continue
        batch_df = pd.concat(batch_frames, ignore_index=True)
        batch_df = batch_df.drop_duplicates(subset="PitchUID", keep="first")
        new_mask = ~batch_df.PitchUID.isin(seen_uids)
        new_rows = batch_df[new_mask]
        seen_uids.update(new_rows.PitchUID)
        frames.append(new_rows)
        done = min(start + FILE_BATCH_SIZE, len(files))
        print(f"  read {done}/{len(files)} files "
              f"({len(seen_uids):,} unique PitchUIDs so far)", flush=True)
    if total_dropped_no_key:
        print(f"  dropped {total_dropped_no_key:,} rows with missing PitchUID/Date "
              f"(blank/malformed rows in source files)")
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def summarize(name: str, df: pd.DataFrame) -> None:
    dates = df.Date.dropna().astype(str)
    date_range = f"{dates.min()}..{dates.max()}" if len(dates) else "n/a"
    n_missing_date = int(df.Date.isna().sum())
    print(f"\n[{name}] rows={len(df):,} "
          f"dates={date_range} "
          f"games={df.GameID.nunique():,} pitchers={df.PitcherId.nunique():,}")
    if n_missing_date:
        print(f"[{name}] WARNING: {n_missing_date:,} rows with missing Date")
    if "RelSpeed" in df.columns:
        n = int(df.RelSpeed.notna().sum())
        print(f"[{name}] RelSpeed present on {n:,} rows ({n / max(len(df), 1):.1%})")
    if "Level" in df.columns:
        print(f"[{name}] Level: {df.Level.value_counts().to_dict()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tree", required=True, help="API pull tree root "
                   "(contains year folders directly, e.g. .../trackman_api)")
    p.add_argument("--sftp-tree", help="Optional second tree root to merge in "
                   "(e.g. the separate SFTP pull), same year-folder shape")
    p.add_argument("--extract", required=True)
    p.add_argument("--out-dir", required=True, help="Directory to write "
                   "per-season rebuilt CSVs into")
    p.add_argument("--seasons", nargs="+", default=None,
                   help="Season years to build/report, e.g. 2025 2026. "
                        "Default: every season found in the trees.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    extract_cols = pd.read_csv(args.extract, nrows=0).columns.tolist()
    raw_cols = [c for c in extract_cols if c not in DERIVED_COLS]

    print("=== gathering file lists ===")
    api_files = tree_files(args.tree)
    print(f"API tree files (non-unverified): {len(api_files):,}")
    sftp_files = tree_files(args.sftp_tree) if args.sftp_tree else []
    if args.sftp_tree:
        print(f"SFTP tree files (non-unverified): {len(sftp_files):,}")

    print("\n=== bucketing API tree files by actual game-date year ===")
    api_buckets = bucket_files_by_season(api_files)
    sftp_buckets = bucket_files_by_season(sftp_files) if sftp_files else {}

    seasons = args.seasons or sorted(set(api_buckets) | set(sftp_buckets))
    print(f"\nSeasons to build: {seasons}")

    rebuilt_paths = {}
    sftp_unique_counts = {}
    for season in seasons:
        print(f"\n=== season {season} ===")
        seen_uids: set = set()

        api_season_files = api_buckets.get(season, [])
        print(f"API files for {season}: {len(api_season_files):,}")
        api_df = load_files(api_season_files, raw_cols, seen_uids)
        api_uid_count = len(seen_uids)

        sftp_season_files = sftp_buckets.get(season, [])
        if sftp_season_files:
            print(f"SFTP files for {season}: {len(sftp_season_files):,}")
            sftp_df = load_files(sftp_season_files, raw_cols, seen_uids)
            sftp_unique_counts[season] = len(seen_uids) - api_uid_count
            print(f"SFTP tree contributed {sftp_unique_counts[season]:,} "
                  f"PitchUIDs not already present from the API tree for {season}")
            season_df = pd.concat([api_df, sftp_df], ignore_index=True)
        else:
            sftp_unique_counts[season] = 0
            season_df = api_df

        out_path = os.path.join(args.out_dir, f"rebuilt_{season}.csv")
        season_df.to_csv(out_path, index=False)
        rebuilt_paths[season] = out_path
        print(f"wrote {out_path} ({len(season_df):,} rows)")
        summarize(f"rebuilt {season}", season_df)
        del api_df, season_df
        if sftp_season_files:
            del sftp_df

    # ---- diff against the hand-assembled extract, per season ----
    keep = [c for c in extract_cols if c in
            ("Date", "GameID", "PitcherId", "PitchUID", "Level", "RelSpeed")]
    print(f"\n=== loading extract (columns: {keep}) ===")
    extract_chunks = []
    for ch in pd.read_csv(args.extract, usecols=keep, low_memory=False,
                           chunksize=500_000):
        extract_chunks.append(ch)
    extract = pd.concat(extract_chunks, ignore_index=True)
    extract["season"] = extract.Date.astype(str).str[:4]

    only_extract = [c for c in extract_cols if c not in raw_cols]
    only_rebuilt_note = "(none expected; rebuilt is a subset of extract's raw columns)"
    print(f"\ncolumns only in extract (derived, expected missing from rebuilt): {only_extract}")

    for season in seasons:
        print(f"\n########## DIFF: season {season} ##########")
        rebuilt = pd.read_csv(rebuilt_paths[season], low_memory=False,
                               usecols=lambda c: c in keep,
                               dtype={"PitcherId": "Int64"} if "PitcherId" in keep else None)
        ext = extract[extract.season == season].copy()
        if "PitcherId" in ext.columns:
            ext["PitcherId"] = pd.to_numeric(ext["PitcherId"], errors="coerce").astype("Int64")

        summarize(f"rebuilt {season}", rebuilt)
        summarize(f"extract {season}", ext)

        r_dates, e_dates = set(rebuilt.Date), set(ext.Date)
        print(f"\ndates only in extract: {sorted(e_dates - r_dates)}")
        print(f"dates only in rebuilt: {sorted(r_dates - e_dates)}")

        a, e = set(rebuilt.GameID), set(ext.GameID)
        print(f"\ngames: rebuilt={len(a):,} extract={len(e):,} "
              f"extract-only={len(e - a):,} rebuilt-only={len(a - e):,}")
        if e - a:
            print(f"  sample extract-only GameIDs: {sorted(e - a)[:15]}")

        pa, pe = set(rebuilt.PitchUID), set(ext.PitchUID)
        print(f"pitches: extract={len(pe):,} rebuilt={len(pa):,} "
              f"extract-only={len(pe - pa):,} rebuilt-only={len(pa - pe):,}")

        for level in sorted(set(rebuilt.Level.dropna()) | set(ext.Level.dropna())):
            ra = rebuilt[rebuilt.Level == level]
            ea = ext[ext.Level == level]
            print(f"Level={level}: rebuilt rows={len(ra):,} pitchers={ra.PitcherId.nunique():,} | "
                  f"extract rows={len(ea):,} pitchers={ea.PitcherId.nunique():,}")

        print(f"\nSFTP-tree-unique PitchUID contribution for {season}: "
              f"{sftp_unique_counts.get(season, 0):,}")


if __name__ == "__main__":
    main()
