import pandas as pd
from collections import defaultdict

def add_runner_states(df):
    r1 = r2 = r3 = 0
    outs = 0
    prev_inning_half = None
    runner_states = []

    for i, row in df.iterrows():
        inning_half = f"{row['Inning']}-{row['Top/Bottom']}"
        if inning_half != prev_inning_half or outs >= 3:
            r1 = r2 = r3 = 0
            outs = 0

        runner_states.append((r1, r2, r3))

        runs_scored = int(row['RunsScored']) if not pd.isna(row['RunsScored']) else 0
        while runs_scored > 0:
            if r3: r3 = 0
            elif r2: r2 = 0
            elif r1: r1 = 0
            runs_scored -= 1

        is_walk = (row.get('KorBB') == 'Walk') or (row.get('PitchCall') == 'HitByPitch')

        if row['PlayResult'] == 'Single':
            r3, r2, r1 = r2, r1, 1
        elif row['PlayResult'] == 'Double':
            r3, r2, r1 = r1, 1, 0
        elif row['PlayResult'] == 'Triple':
            r3, r2, r1 = 1, 0, 0
        elif row['PlayResult'] == 'HomeRun':
            r1 = r2 = r3 = 0
        elif is_walk:
            if r1 and r2 and r3:
                pass
            elif r1 and r2:
                r3, r2, r1 = 1, 1, 1
            elif r1:
                r2, r1 = 1, 1
            else:
                r1 = 1

        outs += int(row['OutsOnPlay']) if not pd.isna(row['OutsOnPlay']) else 0
        prev_inning_half = inning_half

    df[['RunnerOn1B', 'RunnerOn2B', 'RunnerOn3B']] = pd.DataFrame(runner_states, index=df.index)
    return df


def add_game_state(df):
    df['GameState'] = df.apply(
        lambda row: f"{row['RunnerOn1B']}{row['RunnerOn2B']}{row['RunnerOn3B']}-O{row['Outs']}-B{row['Balls']}-S{row['Strikes']}",
        axis=1
    )
    return df


def add_runs_remaining(df):
    df['RunsRemaining'] = 0
    for (inning, half), group in df.groupby(['Inning', 'Top/Bottom'], sort=False):
        runs = group['RunsScored'].fillna(0).astype(int).tolist()
        future_runs = [sum(runs[i+1:]) for i in range(len(runs))]
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