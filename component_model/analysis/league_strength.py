"""League-strength machinery: conference assignment, linkage, and a joint ridge fit.

Why this exists. fair_criterion.add_adjusted builds adjT by subtracting each game's
League mean xT and a batter effect measured against the batter's OWN league mean. In a
closed league every run a pitcher allows is a run that league's hitters score, so league
mean xT comes out roughly equal across leagues whatever their strength, and a batter
effect measured within his league is blind to the level of the league itself. adjT
therefore corrects opponent mix WITHIN a conference but cannot see conference STRENGTH.

The fix here is a joint additive model on pitch-level xT,

    xT = mu + Pconf[pitcher conf] + p[pitcher] + Bconf[batter conf] + b[batter]
            + park[home team] + home[batter is home] + e,

with every effect measured against ONE global mean. Within a closed conference the split
between "its hitters are good" and "its pitchers are bad" is not identified; only
non-conference games separate them. The conference terms carry a tiny penalty so that
split is estimated almost freely from those cross-league games, while individual players
are shrunk toward their conference with the same K as add_adjusted (BATTER_K). Without
the conference terms, a ridge penalty on hundreds of individual effects would pull every
conference-wide shift toward zero and hide the very gap being measured; the synthetic
tests show that failure directly.

Solver: penalized least squares on a sparse indicator design, solved by Jacobi-
preconditioned conjugate gradient on the normal equations. statsmodels is not a project
dependency; this needs only numpy/scipy. Convergence is checked and failure is loud.

ORIENTATION (see the fair_criterion.py docstring): xT is expected run value from the
pitcher's view, LOWER = BETTER for the pitcher. A POSITIVE batter or batter-conference
effect therefore means TOUGHER hitters (they generate more run value against an average
pitcher). A positive pitcher effect means a WORSE pitcher. Conference strength reported
from this module is the batter side: higher = tougher lineup.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

UNKNOWN_CONF = "UNK"


# ---------------- home/away and conference assignment ----------------

def add_home_away(df):
    """Adds home_team, away_team, bat_home (1.0 if the batting team is at home), in place.

    TrackMan's Top/Bottom is the half-inning: in the Top the AWAY team bats, so the
    pitcher's team is home.
    """
    top = df["Top/Bottom"] == "Top"
    df["home_team"] = np.where(top, df["PitcherTeam"], df["BatterTeam"])
    df["away_team"] = np.where(top, df["BatterTeam"], df["PitcherTeam"])
    df["bat_home"] = (~top).astype(float)
    return df


def game_table(df, year_col="year"):
    """One row per (season, game): League, home team, away team."""
    g = df.groupby([year_col, "GameID"], observed=True)
    return pd.DataFrame({"League": g["League"].first(), "home": g["home_team"].first(),
                         "away": g["away_team"].first()}).reset_index()


def _mode_and_purity(s):
    vc = s.value_counts()
    return vc.index[0], float(vc.iloc[0] / vc.sum()), int(vc.sum())


def team_conferences(games, year_col="year", min_away_games=5, min_away_purity=0.5):
    """Conference per (season, team), derived from the games themselves.

    TrackMan's League column is a property of the GAME, and it is the HOME team's
    conference (verified on the D1 caches: one League per game, and a team's home games
    carry one League ~98% of the time while its away games scatter across its opponents'
    conferences). So a team's conference is the modal League of its HOME games. Rules, in
    order:
      1. "home":        modal League over this season's home games.
      2. "home_other":  no home games this season (the park has no TrackMan feed), but home
                        games in another season: modal League there, nearest season first.
                        Assumes no realignment for that program between the seasons.
      3. "away":        modal League over away games, only if at least min_away_games and
                        the mode holds min_away_purity of them. Road conference games land
                        at conference rivals' parks, so the mode can be right, but it is the
                        weakest rule.
      4. UNKNOWN_CONF otherwise.
    Returns a frame: year, team, conf, source, purity, n_games.
    """
    rows = []
    home = {}
    for (yr, team), s in games.groupby([year_col, "home"]):
        c, p, n = _mode_and_purity(s["League"])
        home[(yr, team)] = (c, p, n)
    home_years = {}
    for (yr, team) in home:
        home_years.setdefault(team, []).append(yr)
    away_groups = {k: s for k, s in games.groupby([year_col, "away"])}
    teams = pd.concat([games[[year_col, "home"]].rename(columns={"home": "team"}),
                       games[[year_col, "away"]].rename(columns={"away": "team"})])
    for yr, team in teams.drop_duplicates().itertuples(index=False):
        if (yr, team) in home:
            c, p, n = home[(yr, team)]
            rows.append((yr, team, c, "home", p, n))
            continue
        others = sorted(home_years.get(team, []), key=lambda y: (abs(y - yr), y))
        if others:
            c, p, n = home[(others[0], team)]
            rows.append((yr, team, c, "home_other", p, n))
            continue
        s = away_groups.get((yr, team))
        if s is not None:
            c, p, n = _mode_and_purity(s["League"])
            if n >= min_away_games and p >= min_away_purity:
                rows.append((yr, team, c, "away", p, n))
                continue
            rows.append((yr, team, UNKNOWN_CONF, "unknown", p, n))
            continue
        rows.append((yr, team, UNKNOWN_CONF, "unknown", float("nan"), 0))
    return pd.DataFrame(rows, columns=[year_col, "team", "conf", "source", "purity",
                                       "n_games"])


def conference_links(games, confs, year_col="year"):
    """Cross-conference linkage per season.

    A game links two conferences when its home and away teams are assigned different,
    known conferences. Returns (summary frame per season, dict season -> pair counts).
    The summary reports games, cross-conference games, conferences, linked pairs,
    connected components of the conference graph, and the fewest cross-conference
    games any one conference has (the weakest link).
    """
    cmap = confs.set_index([year_col, "team"])["conf"]
    out, pairs_by = [], {}
    for yr, g in games.groupby(year_col):
        hc = cmap.reindex(list(zip([yr] * len(g), g["home"]))).values
        ac = cmap.reindex(list(zip([yr] * len(g), g["away"]))).values
        known = (hc != UNKNOWN_CONF) & (ac != UNKNOWN_CONF) & pd.notna(hc) & pd.notna(ac)
        cross = known & (hc != ac)
        cnames = sorted(set(hc[known]) | set(ac[known]))
        idx = {c: i for i, c in enumerate(cnames)}
        pairs = pd.Series([tuple(sorted((a, b))) for a, b in zip(hc[cross], ac[cross])])
        pc = pairs.value_counts() if len(pairs) else pd.Series(dtype=int)
        n_comp, n_isolated, min_cross, min_conf = edges_components(cnames, pc)
        per_conf = {c: 0 for c in cnames}
        for (a, b), n in pc.items():
            per_conf[a] += n
            per_conf[b] += n
        out.append({year_col: yr, "games": len(g), "known_games": int(known.sum()),
                    "cross_games": int(cross.sum()), "conferences": len(cnames),
                    "linked_pairs": len(pc), "components": n_comp,
                    "min_cross_games": min_cross, "weakest_conf": min_conf})
        pairs_by[yr] = (pc, per_conf, idx)
    return pd.DataFrame(out), pairs_by


def edges_components(nodes, pair_counts):
    """Connected components of an undirected graph given nodes and {(a, b): n} edges.

    Returns (n_components, n_isolated_nodes, fewest edge-games on any node, that node).
    """
    idx = {c: i for i, c in enumerate(nodes)}
    k = len(nodes)
    if k == 0:
        return 0, 0, 0, None
    deg = np.zeros(k)
    r, c = [], []
    for (a, b), n in pair_counts.items():
        r.append(idx[a])
        c.append(idx[b])
        deg[idx[a]] += n
        deg[idx[b]] += n
    adj = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(k, k))
    n_comp, _ = connected_components(adj, directed=False)
    j = int(np.argmin(deg))
    return int(n_comp), int((deg == 0).sum()), int(deg[j]), nodes[j]


# ---------------- the joint ridge fit ----------------

@dataclass
class Block:
    """One factor of the additive model.

    codes: int array, one per row, in [0, n_levels); -1 means the row has no level here.
    lam:   ridge penalty in pitch-equivalents (shrinkage factor n / (n + lam) for a level
           with n rows and no competing effects, the same convention as BATTER_K).
    """
    name: str
    codes: np.ndarray
    lam: float
    levels: object = None

    @property
    def n_levels(self):
        valid = self.codes[self.codes >= 0]
        return int(valid.max()) + 1 if len(valid) else 0


@dataclass
class JointFit:
    mu: float
    effects: dict = field(default_factory=dict)
    iterations: int = 0
    solution: np.ndarray = None

    def row_effect(self, block):
        """Fitted effect of one block on every row (0 where the row has no level)."""
        e = self.effects[block.name]
        out = np.zeros(len(block.codes))
        ok = block.codes >= 0
        out[ok] = e[block.codes[ok]]
        return out


def encode(values):
    """Factorize to (codes, levels); missing values get -1."""
    codes, levels = pd.factorize(pd.Series(values), sort=True)
    return codes.astype(np.int64), np.asarray(levels)


def design(blocks, n):
    """Sparse indicator design [intercept | block 1 | block 2 | ...] and column offsets."""
    cols, offsets, start = [sp.csr_matrix(np.ones((n, 1)))], [], 1
    for b in blocks:
        k = b.n_levels
        ok = np.flatnonzero(b.codes >= 0)
        m = sp.csr_matrix((np.ones(len(ok)), (ok, b.codes[ok])), shape=(n, k))
        cols.append(m)
        offsets.append((start, start + k))
        start += k
    return sp.hstack(cols, format="csr"), offsets


def fit_joint(y, blocks, weights=None, rtol=1e-10, maxiter=20000, x0=None, design_cache=None):
    """Weighted ridge fit of y on an intercept plus additive factor blocks.

    Minimizes sum_i w_i (y_i - mu - sum_b e_b[code_bi])^2 + sum_b lam_b * sum_j e_bj^2.
    The intercept is unpenalized, so every effect is a deviation from one global mean.
    Raises RuntimeError if conjugate gradient does not converge.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if design_cache is None:
        X, offsets = design(blocks, n)
    else:
        X, offsets = design_cache
    Xw = X.T.multiply(w).tocsr()          # p x n, columns scaled by w
    A = (Xw @ X).tocsr()
    pen = np.zeros(X.shape[1])
    for b, (s, e) in zip(blocks, offsets):
        pen[s:e] = b.lam
    A = A + sp.diags(pen)
    rhs = Xw @ y
    diag = A.diagonal()
    diag[diag <= 0] = 1.0
    M = sp.diags(1.0 / diag)
    it = [0]

    def cb(_):
        it[0] += 1

    sol, info = cg(A, rhs, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=cb)
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge (info={info}, "
                           f"{it[0]} iterations, maxiter={maxiter})")
    fit = JointFit(mu=float(sol[0]), iterations=it[0], solution=sol)
    for b, (s, e) in zip(blocks, offsets):
        fit.effects[b.name] = sol[s:e]
    return fit


def context_effect(fit, blocks, names):
    """Sum of the named blocks' fitted effects on every row."""
    by = {b.name: b for b in blocks}
    return sum(fit.row_effect(by[nm]) for nm in names)


def global_criterion(xt, context, weights=None):
    """xT minus the fitted context, recentered so its mean equals the mean of xT.

    ORIENTATION: same as xT, LOWER = BETTER for the pitcher.
    """
    xt = np.asarray(xt, dtype=float)
    w = np.ones(len(xt)) if weights is None else np.asarray(weights, dtype=float)
    return xt - context + np.average(context, weights=w)


def group_mean_centered(values, groups, weights=None, exclude=(UNKNOWN_CONF,)):
    """Weighted mean of values per group, minus the weighted mean over all included rows.

    Used for conference strength: values = each pitch's fitted batter-side effect (or a
    residual), groups = the batter's conference. Rows in `exclude` groups are dropped
    from both the group means and the centering mean.
    """
    v = np.asarray(values, dtype=float)
    g = pd.Series(np.asarray(groups))
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=float)
    keep = ~g.isin(list(exclude)).values & np.isfinite(v)
    d = pd.DataFrame({"g": g.values[keep], "vw": v[keep] * w[keep], "w": w[keep]})
    s = d.groupby("g")[["vw", "w"]].sum()
    means = s["vw"] / s["w"]
    overall = d["vw"].sum() / d["w"].sum()
    return means - overall


def residual_strength(y, X, groups, weights=None, exclude=(UNKNOWN_CONF,)):
    """Method 3 (cross-check): mean residual of y given predictors X, by group.

    Fits weighted OLS y ~ 1 + X pooled over all groups, then returns
    (group means of the residual centered on the overall mean, coefficients).
    With X = Pitching+ (Stuff+ ridge and Location+), a POSITIVE value means hitters in
    that group produced more run value than the pitches' physics and location predict:
    tougher hitters. Coefficients on run-value predictors should be POSITIVE (both sides
    lower = better for the pitcher); callers print them.
    """
    y = np.asarray(y, dtype=float)
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] != len(y):
        X = X.T
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    D = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(D * sw[:, None], y * sw, rcond=None)
    resid = y - D @ beta
    return group_mean_centered(resid, groups, w, exclude), beta


def poisson_game_weights(game_codes, n_games, rng):
    """Cluster-bootstrap weights: one Poisson(1) draw per game, shared by its pitches.

    The Poisson bootstrap approximates resampling whole games with replacement and lets a
    refit reuse the same sparse design (only the weights change).
    """
    gw = rng.poisson(1.0, n_games).astype(float)
    return gw[game_codes]
