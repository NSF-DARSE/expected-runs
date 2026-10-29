"""Synthetic-recovery tests for the league-strength joint model.

On real data the true conference gap is unknown, so the estimator is checked here
against simulated seasons whose conference strengths are known by construction. The
central case is the one that motivates the module: conferences whose hitters and
pitchers are equally strong, so every conference's mean xT is the same and the
league-mean adjustment in fair_criterion.add_adjusted cannot see any gap at all.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import league_strength as ls

CONF_B = np.array([0.04, 0.015, -0.015, -0.04])   # true batter-conference effects
NOISE_SD = 0.30


def simulate_season(conf_b=CONF_B, offset_pitching=True, teams_per_conf=8,
                    batters=12, pitchers=8, conf_games=24, nonconf_games=8,
                    pitches_per_half=30, home_eff=0.006, seed=3):
    """One season of pitch rows with known effects.

    Pitcher-conference effects equal MINUS the batter-conference effects when
    offset_pitching is True, so within-conference play gives every conference the same
    mean xT. Only non-conference games carry the gap. Returns (rows, truth).
    """
    rng = np.random.default_rng(seed)
    k = len(conf_b)
    conf_p = -conf_b if offset_pitching else np.zeros(k)
    teams = [(c, t) for c in range(k) for t in range(teams_per_conf)]
    tid = {tm: i for i, tm in enumerate(teams)}
    nt = len(teams)
    b_eff = rng.normal(0, 0.02, (nt, batters))
    p_eff = rng.normal(0, 0.02, (nt, pitchers))
    park = rng.normal(0, 0.01, nt)
    games = []
    for c in range(k):
        mem = [tid[(c, t)] for t in range(teams_per_conf)]
        for _ in range(conf_games * teams_per_conf // 2):
            h, a = rng.choice(mem, 2, replace=False)
            games.append((h, a))
    for _ in range(nonconf_games * nt // 2):
        h, a = rng.choice(nt, 2, replace=False)
        if teams[h][0] == teams[a][0]:
            continue
        games.append((h, a))
    rows = {k_: [] for k_ in ["GameID", "home", "away", "League", "PitcherTeam",
                              "BatterTeam", "pitcher", "batter", "bat_home", "xT", "truth_ctx"]}
    for gid, (h, a) in enumerate(games):
        for top in (True, False):
            pt, bt = (h, a) if top else (a, h)
            pi = rng.integers(0, pitchers, pitches_per_half)
            bi = rng.integers(0, batters, pitches_per_half)
            bh = 0.0 if top else 1.0
            ctx = (conf_b[teams[bt][0]] + b_eff[bt, bi] + park[h] + home_eff * bh)
            mean = conf_p[teams[pt][0]] + p_eff[pt, pi] + ctx
            y = mean + rng.normal(0, NOISE_SD, pitches_per_half)
            n = pitches_per_half
            rows["GameID"] += [gid] * n
            rows["home"] += [h] * n
            rows["away"] += [a] * n
            rows["League"] += [f"C{teams[h][0]}"] * n
            rows["PitcherTeam"] += [pt] * n
            rows["BatterTeam"] += [bt] * n
            rows["pitcher"] += list(pt * 100 + pi)
            rows["batter"] += list(bt * 100 + bi)
            rows["bat_home"] += [bh] * n
            rows["xT"] += list(y)
            rows["truth_ctx"] += list(ctx)
    df = pd.DataFrame(rows)
    df["bat_conf"] = df["BatterTeam"].map(lambda t: f"C{teams[t][0]}")
    df["pit_conf"] = df["PitcherTeam"].map(lambda t: f"C{teams[t][0]}")
    truth = {"conf_b": pd.Series(conf_b - conf_b.mean(), index=[f"C{i}" for i in range(k)])}
    return df, truth


def blocks_for(df, with_conf=True, K=100.0, conf_lam=10.0):
    out = []
    if with_conf:
        out.append(ls.Block("pconf", ls.encode(df["pit_conf"])[0], conf_lam))
        out.append(ls.Block("bconf", ls.encode(df["bat_conf"])[0], conf_lam))
    out += [ls.Block("pitcher", ls.encode(df["pitcher"])[0], K),
            ls.Block("batter", ls.encode(df["batter"])[0], K),
            ls.Block("park", ls.encode(df["home"])[0], K),
            ls.Block("home", ls.encode(df["bat_home"])[0], 1.0)]
    return out


def batter_side_strength(df, fit, blocks, names):
    eff = ls.context_effect(fit, blocks, names)
    return ls.group_mean_centered(eff, df["bat_conf"])


@pytest.fixture(scope="module")
def season():
    return simulate_season()


def test_league_means_are_blind_to_the_gap(season):
    """The add_adjusted premise fails: conference mean xT carries no strength signal."""
    df, truth = season
    lg = df.groupby("League")["xT"].mean()
    true_spread = truth["conf_b"].max() - truth["conf_b"].min()
    assert lg.max() - lg.min() < 0.25 * true_spread


def test_joint_fit_recovers_conference_gap(season):
    df, truth = season
    blocks = blocks_for(df)
    fit = ls.fit_joint(df["xT"].values, blocks)
    est = batter_side_strength(df, fit, blocks, ["bconf", "batter"])
    t = truth["conf_b"].reindex(est.index)
    assert np.corrcoef(est, t)[0, 1] > 0.97
    gap_est = est["C0"] - est["C3"]
    gap_true = t["C0"] - t["C3"]
    assert abs(gap_est - gap_true) < 0.25 * gap_true


def test_ridge_without_conference_terms_shrinks_the_gap(season):
    """Why the conference blocks exist: individual-only ridge pulls the gap toward zero."""
    df, truth = season
    with_c = blocks_for(df)
    without = blocks_for(df, with_conf=False)
    f1 = ls.fit_joint(df["xT"].values, with_c)
    f0 = ls.fit_joint(df["xT"].values, without)
    g1 = batter_side_strength(df, f1, with_c, ["bconf", "batter"])
    g0 = batter_side_strength(df, f0, without, ["batter"])
    true_gap = truth["conf_b"]["C0"] - truth["conf_b"]["C3"]
    assert (g0["C0"] - g0["C3"]) < 0.75 * true_gap
    assert (g1["C0"] - g1["C3"]) > (g0["C0"] - g0["C3"])


def test_global_criterion_removes_context(season):
    df, _ = season
    blocks = blocks_for(df)
    fit = ls.fit_joint(df["xT"].values, blocks)
    ctx = ls.context_effect(fit, blocks, ["bconf", "batter", "park", "home"])
    g = ls.global_criterion(df["xT"].values, ctx)
    assert abs(g.mean() - df["xT"].mean()) < 1e-12
    assert np.corrcoef(ctx, df["truth_ctx"])[0, 1] > 0.8
    # The point of the criterion: graded by PITCHER conference, raw xT shows under half the
    # true gap (each conference's pitchers face mostly its own equally-strong hitters),
    # while the global criterion recovers it (pitcher-conference effect = -batter).
    raw = df.groupby("pit_conf")["xT"].mean()
    by_pconf = pd.Series(g).groupby(df["pit_conf"].values).mean()
    true_p = -CONF_B[0] - (-CONF_B[3])                      # C0 minus C3, negative
    assert abs(raw["C0"] - raw["C3"]) < 0.5 * abs(true_p)
    assert abs((by_pconf["C0"] - by_pconf["C3"]) - true_p) < 0.25 * abs(true_p)


def test_fit_matches_dense_closed_form():
    rng = np.random.default_rng(11)
    n = 400
    a = rng.integers(0, 7, n)
    b = rng.integers(0, 5, n)
    a[:10] = -1                                    # rows with no level in block a
    y = rng.normal(0, 1, n) + 0.3 * (a == 2) - 0.2 * (b == 4)
    w = rng.uniform(0.2, 2.0, n)
    blocks = [ls.Block("a", a, 3.0), ls.Block("b", b, 0.5)]
    fit = ls.fit_joint(y, blocks, weights=w, rtol=1e-12)
    X, offs = ls.design(blocks, n)
    X = X.toarray()
    pen = np.zeros(X.shape[1])
    pen[offs[0][0]:offs[0][1]] = 3.0
    pen[offs[1][0]:offs[1][1]] = 0.5
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    sol = np.linalg.solve(A, X.T @ (w * y))
    assert np.allclose(fit.solution, sol, atol=1e-7)
    assert abs(fit.mu - sol[0]) < 1e-7


def test_unit_weights_equal_unweighted(season):
    df, _ = season
    sub = df.iloc[:20000]
    blocks = blocks_for(sub)
    f0 = ls.fit_joint(sub["xT"].values, blocks)
    f1 = ls.fit_joint(sub["xT"].values, blocks, weights=np.ones(len(sub)))
    assert np.allclose(f0.solution, f1.solution, atol=1e-9)


def test_nonconvergence_is_loud(season):
    df, _ = season
    blocks = blocks_for(df)
    with pytest.raises(RuntimeError):
        ls.fit_joint(df["xT"].values, blocks, maxiter=1)


def test_residual_strength_recovers_group_offsets():
    rng = np.random.default_rng(5)
    n = 60000
    x = rng.normal(0, 1, (n, 2))
    grp = rng.choice(["A", "B", "C"], n)
    off = pd.Series({"A": 0.05, "B": 0.0, "C": -0.05})
    y = 0.4 * x[:, 0] + 0.2 * x[:, 1] + off.reindex(grp).values + rng.normal(0, 0.3, n)
    est, beta = ls.residual_strength(y, x, grp)
    assert beta[1] > 0 and beta[2] > 0
    assert abs((est["A"] - est["C"]) - 0.10) < 0.02


def test_group_mean_excludes_unknown():
    v = np.array([1.0, 1.0, 0.0, 0.0, 100.0])
    g = np.array(["X", "X", "Y", "Y", ls.UNKNOWN_CONF])
    m = ls.group_mean_centered(v, g)
    assert ls.UNKNOWN_CONF not in m.index
    assert m["X"] == pytest.approx(0.5) and m["Y"] == pytest.approx(-0.5)


def test_add_home_away():
    df = pd.DataFrame({"Top/Bottom": ["Top", "Bottom"], "PitcherTeam": ["H", "A"],
                       "BatterTeam": ["A", "H"]})
    ls.add_home_away(df)
    assert list(df["home_team"]) == ["H", "H"]
    assert list(df["away_team"]) == ["A", "A"]
    assert list(df["bat_home"]) == [0.0, 1.0]


def test_team_conferences_rules():
    rows = []
    gid = 0
    # H1 hosts in conf X all season.
    for _ in range(6):
        rows.append((2024, gid, "X", "H1", "H2")); gid += 1
    # H2 hosts in 2025 only, in conf X; in 2024 it is away-only.
    for _ in range(3):
        rows.append((2025, gid, "X", "H2", "H1")); gid += 1
    # R1 is away-only both seasons: 5 of 6 road games in conf Y parks (purity 0.83).
    for lg in ["Y"] * 5 + ["X"]:
        rows.append((2024, gid, lg, "H9" if lg == "Y" else "H1", "R1")); gid += 1
    # R2 is away-only with scattered road games: no confident mode.
    for lg in ["X", "Y", "Z", "W", "V"]:
        rows.append((2024, gid, lg, "H1", "R2")); gid += 1
    games = pd.DataFrame(rows, columns=["year", "GameID", "League", "home", "away"])
    c = ls.team_conferences(games).set_index(["year", "team"])
    assert c.loc[(2024, "H1"), "conf"] == "X" and c.loc[(2024, "H1"), "source"] == "home"
    assert c.loc[(2024, "H2"), "conf"] == "X" and c.loc[(2024, "H2"), "source"] == "home_other"
    assert c.loc[(2024, "R1"), "conf"] == "Y" and c.loc[(2024, "R1"), "source"] == "away"
    assert c.loc[(2024, "R2"), "conf"] == ls.UNKNOWN_CONF


def test_edges_components():
    pc = pd.Series({("A", "B"): 3, ("C", "D"): 1})
    n_comp, n_iso, min_n, min_node = ls.edges_components(["A", "B", "C", "D", "E"], pc)
    assert n_comp == 3 and n_iso == 1 and min_n == 0 and min_node == "E"
    pc2 = pd.Series({("A", "B"): 3, ("B", "C"): 2})
    assert ls.edges_components(["A", "B", "C"], pc2)[0] == 1
