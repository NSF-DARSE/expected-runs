import os
import pathlib

import pandas as pd
import pytest

import webapp_publisher.publish as pub
from webapp_publisher.publish import default_season, derive_data_through


def test_default_season_uses_later_stuffplus_years_value(monkeypatch):
    monkeypatch.setenv("STUFFPLUS_YEARS", "2024,2025")
    assert default_season() == 2025


def test_default_season_falls_back_when_env_unset(monkeypatch):
    monkeypatch.delenv("STUFFPLUS_YEARS", raising=False)
    assert default_season() == 2025


def test_default_season_tracks_a_non_default_pair(monkeypatch):
    monkeypatch.setenv("STUFFPLUS_YEARS", "2025,2026")
    assert default_season() == 2026


def test_derive_data_through_picks_max_date_for_season(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": ["2025-03-01", "2025-05-16", "2024-11-01", "2025-04-10"],
        "PitcherTeam": ["FAKE_TEAM"] * 4,
    }).to_csv(csv, index=False)
    assert derive_data_through(str(csv), 2025, "FAKE_TEAM") == "2025-05-16"


def test_derive_data_through_ignores_other_teams_later_dates(tmp_path):
    """Regression for the shipped defect: dataThrough must reflect the target
    team's own last game date, not the max across the whole population. The
    other team here is given a date over a month later so the assertion is
    unambiguous if the team filter is dropped.
    """
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": ["2025-03-01", "2025-04-10", "2025-05-16", "2025-06-22"],
        "PitcherTeam": ["FAKE_TEAM", "FAKE_TEAM", "FAKE_TEAM", "OTHER_TEAM"],
    }).to_csv(csv, index=False)
    assert derive_data_through(str(csv), 2025, "FAKE_TEAM") == "2025-05-16"


def test_derive_data_through_handles_yyyymmdd_ints(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": [20250301, 20250516, 20241101],
        "PitcherTeam": ["FAKE_TEAM", "FAKE_TEAM", "FAKE_TEAM"],
    }).to_csv(csv, index=False)
    assert derive_data_through(str(csv), 2025, "FAKE_TEAM") == "2025-05-16"


def test_derive_data_through_raises_when_no_rows_match_season(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": ["2024-03-01", "2024-05-16"],
        "PitcherTeam": ["FAKE_TEAM", "FAKE_TEAM"],
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        derive_data_through(str(csv), 2025, "FAKE_TEAM")


def test_derive_data_through_raises_when_no_rows_match_team(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": ["2025-03-01", "2025-05-16"],
        "PitcherTeam": ["OTHER_TEAM", "OTHER_TEAM"],
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError) as excinfo:
        derive_data_through(str(csv), 2025, "FAKE_TEAM")
    message = str(excinfo.value)
    assert "FAKE_TEAM" in message
    assert "2025" in message


def test_derive_data_through_excludes_team_rows_in_a_different_season(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({
        "Date": ["2024-03-01", "2024-05-16"],
        "PitcherTeam": ["FAKE_TEAM", "FAKE_TEAM"],
    }).to_csv(csv, index=False)
    with pytest.raises(ValueError) as excinfo:
        derive_data_through(str(csv), 2025, "FAKE_TEAM")
    message = str(excinfo.value)
    assert "FAKE_TEAM" in message
    assert "2025" in message


def test_derive_data_through_raises_when_date_column_missing(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"NotDate": ["2025-03-01"], "PitcherTeam": ["FAKE_TEAM"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        derive_data_through(str(csv), 2025, "FAKE_TEAM")


def test_dry_run_writes_nested_bundle_keys(tmp_path, monkeypatch):
    """Bundle keys like "pitchers/1000123.json" are nested, and the dry-run
    writer previously created only <workdir>/bundle -- so the first pitcher
    file raised FileNotFoundError. This drives the real main() dry-run path
    (with the scorer subprocesses stubbed out) so a regression there, not
    just in the write-loop pattern, would be caught.
    """
    staff_scores = {
        "population": {"n": 1},
        "team": "DEL_BLU",
        "staff": [{
            "name": "Test Pitcher", "hand": "R", "ff": 90.0, "stuff": 100.0,
            "loc": 100.0, "adjres": 100.0, "pitch": 100.0,
            "stuff_nohand": 100.0, "pitch_nohand": 100.0, "whiff": 0.25,
            "zone": 0.45, "heart": 0.1, "mean_height": 2.5, "loc_flag": "",
            "stuff_attr": [("SpinRate", 0.1)],
            "stuff_attr_nohand": [("SpinRate", 0.1)],
        }],
    }
    pages = {
        "season": 2026,
        "model": {
            "featureOrder": ["SpinRate"],
            "byPitchType": {"FF": {"coef": [0.01], "scalerMean": [2200.0],
                                    "scalerScale": [180.0], "populationMeanZ": [0.0],
                                    "displayMu": 0.0, "displaySd": 0.02,
                                    "sampleFloor": 100, "nQualified": 400}},
        },
        "grids": {"pooled": [{"x": 0.0, "z": 2.5, "v": 0.01}]},
        "pitchers": [{
            "pitcherId": 1000123, "name": "Test-Pitcher, Alpha", "hand": "R",
            "arsenal": [{"type": "FF", "n": 412, "usage": 1.0, "stuff": 124.0,
                         "loc": 103.0, "recentChange": -6.2,
                         "trend": {"stuff": None, "velo": None,
                                   "movAngle": None, "movMag": None},
                         "aboveFloor": True,
                         "typical": [2350.0], "percentiles": [78]}],
            "outings": [{"date": "2026-03-15", "type": "FF", "n": 42, "stuff": 118.0}],
            "pitches": [{"d": "2026-03-15", "t": "FF", "x": -0.42, "z": 2.31,
                         "c": "0-2", "g": 131.0, "f": [2350.0]}],
        }],
    }

    monkeypatch.setattr(pub, "run_scorer", lambda data, workdir, team: staff_scores)
    monkeypatch.setattr(pub, "run_pitcher_scorer", lambda data, workdir, team: pages)
    monkeypatch.setattr(pub.sys, "argv", [
        "publish.py", "--data", "unused.csv", "--workdir", str(tmp_path),
        "--season", "2026", "--data-through", "2026-03-15", "--dry-run",
    ])

    assert pub.main() == 0
    assert (tmp_path / "bundle" / "pitchers" / "1000123.json").exists()
    assert (tmp_path / "bundle" / "manifest.json").exists()
