TrackMan Baseball Code Documentation
====================================

This document describes helper functions used to derive game states and run expectancy features from raw baseball play-by-play data.

helpers.py
----------

Function: ``add_runner_states(df)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``add_runner_states()`` function tracks base occupancy for first, second, and third base throughout each play of a baseball game.
It dynamically updates base runner positions based on the play outcome and number of outs.

This function is used to construct a detailed game state for every play, which is later used to compute run expectancy.

Inputs
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Type
     - Description
   * - ``df``
     - ``pandas.DataFrame``
     - Play-by-play dataset containing at least the following columns: ``Inning``, ``Top/Bottom``, ``PlayResult``, ``RunsScored``, ``OutsOnPlay``, ``KorBB``, and ``PitchCall``.

Outputs
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column Name
     - Description
   * - ``RunnerOn1B``
     - Indicates whether there is a runner on first base. ``1`` = yes, ``0`` = no.
   * - ``RunnerOn2B``
     - Indicates whether there is a runner on second base. ``1`` = yes, ``0`` = no.
   * - ``RunnerOn3B``
     - Indicates whether there is a runner on third base. ``1`` = yes, ``0`` = no.

Core Logic Explanation
^^^^^^^^^^^^^^^^^^^^^^

1. **Initialize base and inning state.**

   - ``r1``, ``r2``, and ``r3`` track whether first, second, and third base are occupied.
   - ``outs`` tracks outs in the current half-inning.
   - ``prev_inning_half`` tracks inning transitions.
   - ``runner_states`` stores the base configuration before each play.

2. **Iterate through each play.**

   - For each row, determine the current half-inning using ``Inning`` and ``Top/Bottom``.
   - If a new half-inning begins or three outs occur, reset all bases and outs to zero.

3. **Record the pre-play base state.**

   - Append the current base occupancy, ``r1``, ``r2``, and ``r3``, before updating for the new play.

4. **Handle runs scored.**

   - For every run scored in ``RunsScored``, clear bases starting from third to first to represent runners reaching home.

5. **Handle walks and hit-by-pitch events.**

   - Detect walks or hit-by-pitches using:

     .. code-block:: python

        is_walk = (row.get("KorBB") == "Walk") or (row.get("PitchCall") == "HitByPitch")

   - Advance runners according to forced-advance rules.

6. **Update bases after each play.**

   - ``Single``: runners advance one base.
   - ``Double``: runners advance two bases.
   - ``Triple``: runners advance three bases.
   - ``Home Run``: all runners score and bases are cleared.

   Example logic:

   .. code-block:: python

      if row["PlayResult"] == "Single":
          r3, r2, r1 = r2, r1, 1

7. **Track outs.**

   - Update outs using ``OutsOnPlay``.
   - Reset bases and outs after three outs or an inning change.

8. **Finalize output.**

   - Combine recorded base states into the new columns ``RunnerOn1B``, ``RunnerOn2B``, and ``RunnerOn3B``.

Function: ``add_game_state(df)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``add_game_state()`` function creates a unique string identifier for the state of the game at every pitch or play.
This identifier, called ``GameState``, encodes:

- Base occupancy: runners on first, second, and third.
- Number of outs.
- Ball count.
- Strike count.

This unified string is useful for grouping similar game scenarios in run expectancy models and advanced baseball analytics.

Input
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Type
     - Description
   * - ``df``
     - ``pandas.DataFrame``
     - DataFrame containing ``RunnerOn1B``, ``RunnerOn2B``, ``RunnerOn3B``, ``Outs``, ``Balls``, and ``Strikes``. These columns are either generated earlier using ``add_runner_states()`` or present in the raw data.

Output
^^^^^^

- Adds a new column named ``GameState``.
- Each ``GameState`` value is a string representing the complete game situation at a specific moment.

GameState Format
^^^^^^^^^^^^^^^^

The ``GameState`` string uses the following format:

.. code-block:: text

   <runners>-O<outs>-B<balls>-S<strikes>

Where:

- ``<runners>`` is a three-digit binary string:

  - First digit: ``1`` if there is a runner on first base, otherwise ``0``.
  - Second digit: ``1`` if there is a runner on second base, otherwise ``0``.
  - Third digit: ``1`` if there is a runner on third base, otherwise ``0``.

- ``O<outs>`` is the number of outs, from ``0`` to ``2``.
- ``B<balls>`` is the ball count, from ``0`` to ``3``.
- ``S<strikes>`` is the strike count, from ``0`` to ``2``.

Why This Matters
^^^^^^^^^^^^^^^^

Encoding the key components of a play situation into a single field allows:

- Grouping and aggregation.
- Computation of expected runs from a given state.
- Identification of the most frequent or valuable game states.

Function: ``add_runs_remaining(df)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Purpose
^^^^^^^

The ``add_runs_remaining()`` function calculates future runs scored after each pitch or play within the same half-inning.
This metric is essential for run expectancy modeling because it quantifies how many runs are eventually scored after the current pitch or event.

Input
^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Parameter
     - Type
     - Description
   * - ``df``
     - ``pandas.DataFrame``
     - DataFrame containing ``Inning``, ``Top/Bottom``, and ``RunsScored``.

Output
^^^^^^

- Adds a new column named ``RunsRemaining``.
- Each row's value represents the total number of runs scored later in the same half-inning after that play.

Logic Breakdown
^^^^^^^^^^^^^^^

1. Initialize ``RunsRemaining`` with zeroes.
2. Group the DataFrame by half-inning using ``Inning`` and ``Top/Bottom``.
3. For each half-inning group:

   - Convert ``RunsScored`` to a list of integers.
   - For every play ``i``, compute the sum of runs from ``i + 1`` onward.
   - Assign that value as the future runs after play ``i``.

4. Assign the computed values back to ``RunsRemaining`` for the corresponding row indices.

Example
^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 15 25 25 35

   * - Inning
     - Top/Bottom
     - RunsScored
     - RunsRemaining
   * - 1
     - Top
     - 0
     - 2
   * - 1
     - Top
     - 1
     - 1
   * - 1
     - Top
     - 1
     - 0
   * - 1
     - Bottom
     - 0
     - 1
   * - 1
     - Bottom
     - 1
     - 0

For the first row in the top of the first inning, two more runs were scored later in that half-inning, so ``RunsRemaining = 2``.

Why This Matters
^^^^^^^^^^^^^^^^

This future-looking metric is useful for:

- Calculating expected runs for game states.
- Modeling decisions such as bunt vs. swing, pitch changes, or steal attempts.
- Supporting reinforcement-learning or decision-tree-based simulations in baseball analytics.

Generate_summary.ipynb
----------------------

Function: ``build_gamestate_summary_all_years(data_root, save_path)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Purpose
^^^^^^^

The ``build_gamestate_summary_all_years()`` function processes all game event CSV files across all years and months in the specified ``data_root`` directory.
It generates a comprehensive game state summary by computing total occurrences and expected runs for every unique base-out-ball-strike situation.

Input Parameters
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Parameter
     - Type
     - Description
   * - ``data_root``
     - ``str``
     - Base path to the baseball dataset directory. The directory should be organized as ``data_root/year/month/day/CSV/*.csv``.
   * - ``save_path``
     - ``str``
     - Output folder path where the final summary CSV will be saved.

Output
^^^^^^

Returns a DataFrame containing:

- ``GameState``: string such as ``010-O2-B3-S1`` representing base runners, outs, balls, and strikes.
- ``Count``: number of times that game state appeared in valid plays.
- ``TotalRunsRemaining``: total number of runs scored after that game state within the same inning half.
- ``ExpectedRuns``: average number of runs expected from that state, calculated as ``TotalRunsRemaining / Count``.

The function also saves the summary as a timestamped CSV file with a name like:

.. code-block:: text

   GameState_Summary_ALL_YYYYMMDD_HHMM.csv

Expected Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   data_root/
   |-- 2024/
   |   |-- 01/
   |   |   |-- 01/
   |   |   |   |-- CSV/
   |   |   |   |   |-- *.csv
   |   |-- ...
   |-- 2025/
   |   |-- ...

Filtering and Validation Logic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each file is included only if:

1. The filename does not contain ``_unverified`` or ``playerpositioning``.
2. The file extension is ``.csv``.
3. The required columns are present:

   .. code-block:: python

      {"Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "RunsScored", "PlayResult"}

4. The file contains at least one row with ``Inning < 9``.

Processing Steps
^^^^^^^^^^^^^^^^

1. Traverse all years and months inside ``data_root``.
2. For every valid CSV file:

   - Load the file into a DataFrame.
   - Filter out invalid games.
   - Apply transformations:

     - ``add_runner_states(df)`` tracks runner positions before the current play.
     - ``add_game_state(df)`` encodes the game situation as a string.
     - ``add_runs_remaining(df)`` calculates future runs within the same half-inning.

   - Aggregate total occurrences and total future runs for each ``GameState``.

3. Save the final summary as a timestamped CSV file in ``save_path``.

Example Usage
^^^^^^^^^^^^^

.. code-block:: python

   data_root = "/Users/suma/Downloads/Baseball_Project/v3"
   save_path = "/Users/suma/Downloads/Baseball_Project/CSV_files"

   summary_df = build_gamestate_summary_all_years(data_root, save_path)
   summary_df.head()
