import os
import pathlib

import pandas as pd
import pytest

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
    pd.DataFrame({"Date": ["2025-03-01", "2025-05-16", "2024-11-01", "2025-04-10"]}).to_csv(csv, index=False)
    assert derive_data_through(str(csv), 2025) == "2025-05-16"


def test_derive_data_through_handles_yyyymmdd_ints(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"Date": [20250301, 20250516, 20241101]}).to_csv(csv, index=False)
    assert derive_data_through(str(csv), 2025) == "2025-05-16"


def test_derive_data_through_raises_when_no_rows_match_season(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"Date": ["2024-03-01", "2024-05-16"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        derive_data_through(str(csv), 2025)


def test_derive_data_through_raises_when_date_column_missing(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"NotDate": ["2025-03-01"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        derive_data_through(str(csv), 2025)
