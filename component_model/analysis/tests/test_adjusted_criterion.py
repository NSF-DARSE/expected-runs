"""The adopted fair criterion: fair_criterion.add_adjusted builds adjT from the joint
league-strength fit (2026-10-29), and the retired within-league version stays callable.

The synthetic seasons come from test_league_strength.simulate_season, reshaped into the
TrackMan columns add_adjusted reads (Top/Bottom, PitcherTeam, BatterTeam, GameID, League,
Batter, PitcherId, year, xT). Conferences there have equally strong hitters and pitchers,
so every conference's mean xT is the same and only non-conference games carry the gap.

ORIENTATION: xT and adjT are expected run value from the pitcher's view, LOWER = BETTER.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fair_criterion as fc  # noqa: E402
from test_league_strength import CONF_B, simulate_season  # noqa: E402


def trackman_frame(seed=3, year=2025):
    """A simulated season in the column layout add_adjusted reads."""
    sim, _ = simulate_season(seed=seed)
    return pd.DataFrame({
        "GameID": [f"{year}-{g}" for g in sim["GameID"]],
        # In the Top the AWAY team bats, so the batting team is home in the Bottom.
        "Top/Bottom": np.where(sim["bat_home"] == 1.0, "Bottom", "Top"),
        "PitcherTeam": "T" + sim["PitcherTeam"].astype(str),
        "BatterTeam": "T" + sim["BatterTeam"].astype(str),
        "League": sim["League"],
        "PitcherId": sim["pitcher"],
        "Batter": "B" + sim["batter"].astype(str),
        "year": year,
        "xT": sim["xT"],
        # Kept for assertions only; add_adjusted never reads these.
        "true_bat_conf": sim["bat_conf"],
        "true_pit_conf": sim["pit_conf"],
        "truth_ctx": sim["truth_ctx"],
    })


@pytest.fixture(scope="module")
def adjusted():
    df = trackman_frame()
    fc.add_adjusted(df, method="global")
    return df


def test_conferences_come_from_home_games(adjusted):
    assert (adjusted["bat_conf"] == adjusted["true_bat_conf"]).all()
    assert (adjusted["pit_conf"] == adjusted["true_pit_conf"]).all()


def test_default_is_the_global_criterion(monkeypatch):
    monkeypatch.delenv("STUFFPLUS_ADJ_METHOD", raising=False)
    assert fc.resolve_adj_method() == "global"
    a, b = trackman_frame(), trackman_frame()
    fc.add_adjusted(a)
    fc.add_adjusted(b, method="global")
    np.testing.assert_allclose(a["adjT"], b["adjT"])


def test_global_sees_conference_strength_and_within_league_does_not(adjusted):
    """Graded by pitcher conference, the adopted adjT recovers the true pitching gap
    between the strongest and weakest conference; the retired version cannot."""
    old = trackman_frame()
    fc.add_adjusted_within_league(old)
    true_gap = -CONF_B[0] - (-CONF_B[3])          # C0 minus C3 pitchers: negative, better
    new_by = adjusted.groupby("true_pit_conf")["adjT"].mean()
    old_by = old.groupby("true_pit_conf")["adjT"].mean()
    assert abs((new_by["C0"] - new_by["C3"]) - true_gap) < 0.25 * abs(true_gap)
    assert abs(old_by["C0"] - old_by["C3"]) < 0.5 * abs(true_gap)


def test_sign_tougher_hitters_lower_adjT(adjusted):
    """Facing tougher hitters (positive context) must LOWER adjT relative to xT, crediting
    the pitcher. An inverted sign shipped once in this project; pin it."""
    removed = adjusted["xT"] - adjusted["adjT"]
    assert np.corrcoef(removed, adjusted["truth_ctx"])[0, 1] > 0.8
    assert abs(adjusted["adjT"].mean() - adjusted["xT"].mean()) < 1e-10


def test_each_season_is_fit_on_itself_only():
    """A second season in the frame must not move the first season's adjT."""
    alone = trackman_frame(seed=3, year=2024)
    fc.add_adjusted(alone, method="global")
    both = pd.concat([trackman_frame(seed=3, year=2024), trackman_frame(seed=5, year=2025)],
                     ignore_index=True)
    fc.add_adjusted(both, method="global")
    first = both[both["year"] == 2024]
    np.testing.assert_allclose(first["adjT"].values, alone["adjT"].values, atol=1e-8)


def test_missing_xT_stays_missing():
    df = trackman_frame()
    df.loc[df.index[:50], "xT"] = np.nan
    fc.add_adjusted(df, method="global")
    assert df.loc[df.index[:50], "adjT"].isna().all()
    assert df.loc[df.index[50:], "adjT"].notna().all()


def test_env_override_selects_the_retired_version(monkeypatch):
    monkeypatch.setenv("STUFFPLUS_ADJ_METHOD", "within_league")
    a, b = trackman_frame(), trackman_frame()
    fc.add_adjusted(a)
    fc.add_adjusted_within_league(b)
    np.testing.assert_allclose(a["adjT"], b["adjT"])


def test_unknown_method_is_loud():
    with pytest.raises(ValueError):
        fc.add_adjusted(trackman_frame(), method="pooled")
