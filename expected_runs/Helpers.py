import pandas as pd
from collections import defaultdict

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