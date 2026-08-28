"""Whitespace-variant pitcher names collapse to one display string at load.

The 2024 season carries 17 names with a space before the comma (plus stray
doubled/trailing spaces). PitcherId is consistent across the variants, so this
is display hygiene only -- but a name-keyed consumer downstream must never see
two spellings of one pitcher.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fair_criterion as fc


def test_space_before_comma_and_stray_whitespace_collapse():
    df = pd.DataFrame({"Pitcher": [
        "Test-Pitcher , Alpha",
        "Test-Pitcher, Alpha",
        "Test-Pitcher ,  Bravo ",
        "Test-Pitcher,   Bravo",
    ]})
    fc.normalize_pitcher_names(df)
    assert set(df["Pitcher"]) == {"Test-Pitcher, Alpha", "Test-Pitcher, Bravo"}


def test_clean_names_pass_through_unchanged():
    df = pd.DataFrame({"Pitcher": ["Test-Pitcher, Alpha"]})
    fc.normalize_pitcher_names(df)
    assert df["Pitcher"].iloc[0] == "Test-Pitcher, Alpha"
