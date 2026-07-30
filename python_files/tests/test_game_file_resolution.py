"""Tests for resolve_latest_game_files.

The bug being guarded against: TrackMan API pulls land a game's CSV in the folder
for the day it was FETCHED, not the day it was played. A game that is revised and
re-pulled therefore appears in several day folders, and the earlier copies are
pre-correction. Reading every folder both double-counts pitches and keeps stale
values for the ~64% of duplicated files whose contents actually changed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Helpers import resolve_latest_game_files


def write_game(root, year, month, day, name, body="PitchUID\nabc\n"):
    folder = os.path.join(root, year, month, day, "CSV")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return path


def test_single_copy_is_returned(tmp_path):
    root = str(tmp_path)
    path = write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")

    assert resolve_latest_game_files(root) == [path]


def test_keeps_newest_copy_within_a_month(tmp_path):
    root = str(tmp_path)
    write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv", "PitchUID\nstale\n")
    newest = write_game(root, "2026", "02", "10", "20260206-Lamar-1.csv", "PitchUID\nfresh\n")

    assert resolve_latest_game_files(root) == [newest]


def test_keeps_newest_copy_across_months(tmp_path):
    """The real failure mode: a January game revised in March.

    Any month-scoped resolution misses this, which is why discovery has to span
    the whole tree before any file is read.
    """
    root = str(tmp_path)
    write_game(root, "2026", "01", "24", "20260123-Cypress-1.csv", "PitchUID\nstale\n")
    write_game(root, "2026", "02", "03", "20260123-Cypress-1.csv", "PitchUID\nstale\n")
    newest = write_game(root, "2026", "03", "11", "20260123-Cypress-1.csv", "PitchUID\nfresh\n")

    assert resolve_latest_game_files(root) == [newest]


def test_keeps_newest_copy_across_years(tmp_path):
    root = str(tmp_path)
    write_game(root, "2024", "05", "20", "20240520-Home-1.csv", "PitchUID\nstale\n")
    newest = write_game(root, "2025", "02", "01", "20240520-Home-1.csv", "PitchUID\nfresh\n")

    assert resolve_latest_game_files(root) == [newest]


def test_day_ordering_is_numeric_not_lexicographic(tmp_path):
    root = str(tmp_path)
    newest = write_game(root, "2026", "02", "10", "20260206-Lamar-1.csv")
    write_game(root, "2026", "02", "09", "20260206-Lamar-1.csv")

    assert resolve_latest_game_files(root) == [newest]


def test_distinct_games_are_all_kept(tmp_path):
    root = str(tmp_path)
    a = write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")
    b = write_game(root, "2026", "02", "07", "20260206-Pomona-1.csv")

    assert sorted(resolve_latest_game_files(root)) == sorted([a, b])


def test_doubleheader_game_numbers_are_distinct_games(tmp_path):
    root = str(tmp_path)
    g1 = write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")
    g2 = write_game(root, "2026", "02", "07", "20260206-Lamar-2.csv")

    assert sorted(resolve_latest_game_files(root)) == sorted([g1, g2])


@pytest.mark.parametrize(
    "name",
    [
        "20260206-Lamar-1_unverified.csv",
        "20260206-Lamar-1_UNVERIFIED.csv",
        "20260206-playerpositioning.csv",
        "20260206-PlayerPositioning.csv",
        "notes.txt",
    ],
)
def test_excluded_files_are_skipped(tmp_path, name):
    root = str(tmp_path)
    write_game(root, "2026", "02", "07", name)

    assert resolve_latest_game_files(root) == []


def test_unverified_copy_does_not_supersede_a_verified_one(tmp_path):
    """An unverified pull in a later folder must not win over a verified game."""
    root = str(tmp_path)
    verified = write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")
    write_game(root, "2026", "02", "20", "20260206-Lamar-1_unverified.csv")

    assert resolve_latest_game_files(root) == [verified]


def test_years_filter_restricts_the_tree(tmp_path):
    root = str(tmp_path)
    kept = write_game(root, "2024", "05", "20", "20240520-Home-1.csv")
    write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")

    assert resolve_latest_game_files(root, years=["2024"]) == [kept]


def test_years_filter_accepts_ints(tmp_path):
    root = str(tmp_path)
    kept = write_game(root, "2024", "05", "20", "20240520-Home-1.csv")
    write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")

    assert resolve_latest_game_files(root, years=[2024]) == [kept]


def test_years_filter_does_not_hide_a_cross_year_revision(tmp_path):
    """A 2024 game re-pulled in 2025 is still a 2024 game.

    Restricting discovery to the requested years would silently keep the stale
    copy, so revision resolution scans the whole tree and the years filter is
    applied to the game, not to the folder it was fetched into.
    """
    root = str(tmp_path)
    write_game(root, "2024", "05", "20", "20240520-Home-1.csv", "PitchUID\nstale\n")
    newest = write_game(root, "2025", "02", "01", "20240520-Home-1.csv", "PitchUID\nfresh\n")

    assert resolve_latest_game_files(root, years=["2024"]) == [newest]


def test_order_is_ascending_by_fetch_date(tmp_path):
    """The build relies on this to let a later pull win a cross-file collision."""
    root = str(tmp_path)
    third = write_game(root, "2026", "03", "23", "20260221-Riddle-1.csv")
    first = write_game(root, "2026", "02", "09", "20260208-Lamar-1.csv")
    second = write_game(root, "2026", "02", "23", "20260222-Riddle-1.csv")

    assert resolve_latest_game_files(root) == [first, second, third]


def test_result_is_deterministic(tmp_path):
    root = str(tmp_path)
    write_game(root, "2026", "02", "07", "20260206-Pomona-1.csv")
    write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")
    write_game(root, "2026", "03", "01", "20260228-Cypress-1.csv")

    assert resolve_latest_game_files(root) == resolve_latest_game_files(root)


def test_missing_root_returns_empty(tmp_path):
    assert resolve_latest_game_files(os.path.join(str(tmp_path), "nope")) == []


def test_non_date_folders_are_ignored(tmp_path):
    root = str(tmp_path)
    kept = write_game(root, "2026", "02", "07", "20260206-Lamar-1.csv")
    stray = os.path.join(root, "2026", "notes", "readme", "CSV")
    os.makedirs(stray)
    with open(os.path.join(stray, "20260206-Lamar-1.csv"), "w", encoding="utf-8") as handle:
        handle.write("PitchUID\nstray\n")

    assert resolve_latest_game_files(root) == [kept]
