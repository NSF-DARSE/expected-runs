import os
import re

import pandas as pd
from collections import defaultdict


EXCLUDED_FILE_MARKERS = ("unverified", "playerpositioning")

_GAME_DATE_PREFIX = re.compile(r"^(\d{4})\d{4}")


def _is_game_csv(filename):
    lowered = filename.lower()
    if not lowered.endswith(".csv"):
        return False
    return not any(marker in lowered for marker in EXCLUDED_FILE_MARKERS)


def _game_year(filename, folder_year):
    """Year the game was PLAYED, from the TrackMan filename date prefix.

    Falls back to the folder year when a file does not carry the usual
    YYYYMMDD- prefix.
    """
    match = _GAME_DATE_PREFIX.match(filename)
    return match.group(1) if match else folder_year


def resolve_latest_game_files(data_root, years=None, verbose=False):
    """Return one path per distinct game CSV, choosing the most recent copy.

    TrackMan API pulls land a game's CSV in the folder for the day it was
    FETCHED, not the day it was played, so a game that is revised and re-pulled
    appears in several day folders. In the 2026 tree, 642 of 8,338 game
    filenames sit in more than one folder, which is 216k duplicate rows out of
    2.73M, and 413 of those 642 differ in CONTENT between copies. The later copy
    is the corrected one: observed revisions replace placeholder BatterIds with
    real ones, fix batter names, flip BatterSide (65 pitches in one game), and
    change PlayResult. So the earlier copy is not a harmless duplicate, and
    dropping the *second* occurrence of a PitchUID keeps the stale values.

    Resolving at the file level instead handles the identical copies and the
    revisions in one step, and it has to span the whole tree rather than a
    single month: a January game can be re-pulled in March.

    `years` filters on the year the game was played (from the filename date
    prefix), not on the folder the copy was fetched into, so a 2024 game
    re-pulled in 2025 still resolves to its corrected copy.

    This does not catch one pitch appearing under two different game FILENAMES,
    which happens with suspended games: 20260221-RiddlePaceField-1 was first
    exported as a 178-pitch Feb 22 continuation and later re-issued as the full
    247-pitch game under its original Feb 21 date, with PitchNo renumbered. The
    returned order is ascending by fetch date so the caller can resolve that by
    letting the newest file win.
    """
    if not os.path.isdir(data_root):
        return []

    wanted = {str(year) for year in years} if years is not None else None
    newest = {}

    for year in sorted(os.listdir(data_root)):
        year_path = os.path.join(data_root, year)
        if not (year.isdigit() and os.path.isdir(year_path)):
            continue

        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not (month.isdigit() and os.path.isdir(month_path)):
                continue

            for day in sorted(os.listdir(month_path)):
                csv_path = os.path.join(month_path, day, "CSV")
                if not (day.isdigit() and os.path.isdir(csv_path)):
                    continue

                fetched = (int(year), int(month), int(day))

                for filename in sorted(os.listdir(csv_path)):
                    if not _is_game_csv(filename):
                        continue
                    if wanted is not None and _game_year(filename, year) not in wanted:
                        continue

                    key = filename.lower()
                    path = os.path.join(csv_path, filename)
                    current = newest.get(key)
                    if current is None or fetched > current[0]:
                        newest[key] = (fetched, path)

    # Ascending fetch order, so a later pull always lands after an earlier one. The
    # concatenated build relies on this to let the newer file win any leftover
    # cross-file PitchUID collision (see the suspended-game case below).
    resolved = [path for _, path in sorted(newest.values())]

    if verbose:
        print(
            f"Resolved {len(resolved)} game files under {data_root}"
            + (f" for years {sorted(wanted)}" if wanted else "")
        )

    return resolved


def count_superseded_copies(data_root, years=None):
    """How many game-CSV copies are superseded by a newer pull. Diagnostic only."""
    if not os.path.isdir(data_root):
        return 0

    wanted = {str(year) for year in years} if years is not None else None
    total = 0

    for year in sorted(os.listdir(data_root)):
        year_path = os.path.join(data_root, year)
        if not (year.isdigit() and os.path.isdir(year_path)):
            continue
        for month in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month)
            if not (month.isdigit() and os.path.isdir(month_path)):
                continue
            for day in sorted(os.listdir(month_path)):
                csv_path = os.path.join(month_path, day, "CSV")
                if not (day.isdigit() and os.path.isdir(csv_path)):
                    continue
                for filename in os.listdir(csv_path):
                    if not _is_game_csv(filename):
                        continue
                    if wanted is not None and _game_year(filename, year) not in wanted:
                        continue
                    total += 1

    return total - len(resolve_latest_game_files(data_root, years=years))


def add_runner_states(df):
    """
    Reconstructs base runner states (1B, 2B, 3B) sequentially
    for each pitch within an inning-half.

    This function:
        • Tracks runners on first, second, and third base
        • Resets runners at new inning-half or after 3 outs
        • Updates runners based on play result and runs scored
        • Returns the dataframe with RunnerOn1B, RunnerOn2B, RunnerOn3B columns added
    """

    # Initialize base states (0 = empty, 1 = occupied)
    r1 = r2 = r3 = 0

    # Track outs within inning-half
    outs = 0

    # Track previous inning-half to detect inning transitions
    prev_inning_half = None

    # Store runner states for each row
    runner_states = []

    # Iterate row-by-row (sequential pitch logic)
    for i, row in df.iterrows():

        # Identify current inning-half (e.g., 5-Top or 5-Bottom)
        inning_half = f"{row['Inning']}-{row['Top/Bottom']}"

        # Reset bases if new inning-half OR 3 outs reached
        if inning_half != prev_inning_half or outs >= 3:
            r1 = r2 = r3 = 0
            outs = 0

        # Store current runner configuration BEFORE this pitch
        runner_states.append((r1, r2, r3))

        # Determine how many runs were scored on this play
        runs_scored = int(row['RunsScored']) if not pd.isna(row['RunsScored']) else 0

        # Remove runners who scored (starting from third base)
        while runs_scored > 0:
            if r3:
                r3 = 0
            elif r2:
                r2 = 0
            elif r1:
                r1 = 0
            runs_scored -= 1

        # Identify walk or hit-by-pitch
        is_walk = (row.get('KorBB') == 'Walk') or (row.get('PitchCall') == 'HitByPitch')

        # Update base states based on play result
        if row['PlayResult'] == 'Single':
            r3, r2, r1 = r2, r1, 1

        elif row['PlayResult'] == 'Double':
            r3, r2, r1 = r1, 1, 0

        elif row['PlayResult'] == 'Triple':
            r3, r2, r1 = 1, 0, 0

        elif row['PlayResult'] == 'HomeRun':
            # All runners including batter score
            r1 = r2 = r3 = 0

        elif is_walk:
            # Force advancement logic for walks/HBP
            if r1 and r2 and r3:
                pass  # Bases loaded walk already handled by run logic
            elif r1 and r2:
                r3, r2, r1 = 1, 1, 1
            elif r1:
                r2, r1 = 1, 1
            else:
                r1 = 1

        # Update outs based on play
        outs += int(row['OutsOnPlay']) if not pd.isna(row['OutsOnPlay']) else 0

        # Update inning tracker
        prev_inning_half = inning_half

    # Add reconstructed runner columns to dataframe
    df[['RunnerOn1B', 'RunnerOn2B', 'RunnerOn3B']] = pd.DataFrame(runner_states, index=df.index)

    return df


def add_game_state(df):
    """
    Creates a unique GameState string representing the full pitch context.

    GameState format:
        [RunnerOn1B][RunnerOn2B][RunnerOn3B]-O[Outs]-B[Balls]-S[Strikes]

    Example:
        101-O2-B1-S2
        → Runners on 1B and 3B
        → 2 outs
        → 1 ball
        → 2 strikes

    This state is later used to map Expected Runs.
    """

    # Construct GameState string row-wise using base occupancy,
    # outs, balls, and strikes
    df['GameState'] = df.apply(
        lambda row: f"{row['RunnerOn1B']}{row['RunnerOn2B']}{row['RunnerOn3B']}"
                    f"-O{row['Outs']}-B{row['Balls']}-S{row['Strikes']}",
        axis=1
    )

    return df


def add_runs_remaining(df):
    """
    Calculates the number of runs that will be scored later
    in the same half-inning after each pitch.

    For each (Inning, Top/Bottom) group:
        - Looks at RunsScored column
        - Computes how many runs occur AFTER the current row
        - Stores that value in a new column: RunsRemaining

    This is used later for run expectancy calculations.
    """

    # Initialize RunsRemaining column with default value 0
    df['RunsRemaining'] = 0

    # Group data by inning and half-inning (Top or Bottom)
    for (inning, half), group in df.groupby(['Inning', 'Top/Bottom'], sort=False):

        # Convert RunsScored to integers, replacing NaN with 0
        runs = group['RunsScored'].fillna(0).astype(int).tolist()

        # For each row, calculate total runs scored AFTER that row
        # i+1 ensures we exclude current pitch's runs
        future_runs = [sum(runs[i+1:]) for i in range(len(runs))]

        # Assign computed future run totals back to original dataframe
        df.loc[group.index, 'RunsRemaining'] = future_runs

    return df

def calculate_zero_run_probabilities(df):
    """
    Calculates, for each GameState:
      - How many times RunsRemaining = 0 occurred
      - The total number of occurrences for that GameState
      - The probability that RunsRemaining = 0

    Returns:
        dict: {
            GameState: {
                "ZeroRunsCount": int,
                "TotalCount": int,
                "ZeroRunProbability": float
            }
        }
    """
    result = defaultdict(lambda: {"ZeroRunsCount": 0, "TotalCount": 0})

    for state, group in df.groupby("GameState"):
        total = len(group)
        zero_runs = (group["RunsRemaining"] == 0).sum()
        prob = zero_runs / total if total > 0 else 0

        result[state]["ZeroRunsCount"] = zero_runs
        result[state]["TotalCount"] = total
        result[state]["ZeroRunProbability"] = round(prob, 4)

    return result