"""17: Can the fair criterion see conference strength? A joint global-mean adjustment.

The problem. add_adjusted (fair_criterion.py) builds C2 = adjT by subtracting each game's
League mean xT and a batter effect measured against the batter's OWN league mean. Inside
a closed conference every run a pitcher allows is a run that conference's hitters score,
so conference mean xT is roughly equal everywhere whatever the conference's strength, and
within-league batter effects cannot see the level of the league. adjT corrects opponent
mix WITHIN a conference but not BETWEEN them, so a pitcher from a weak conference grades
better than he is. (Measured, the premise is only partly true: League means are not flat,
and C2 removes roughly 40% of out-of-season opponent strength. See the calibration
section and RESULTS.md.)

Three criteria, four-seam, D1, 2024-2026:
  C2      the shipped adjT, fit per season (and the pooled-frame variant script 12 uses)
  G       adjT_global: xT minus a JOINT fit of batter, batter-conference, park and
          batting-at-home effects, all measured against ONE global mean, with pitcher and
          pitcher-conference effects in the same fit (league_strength.py). Non-conference
          games are what identify the conference terms; their counts and the
          connectivity of the conference graph are printed first.
  method 3  CROSS-CHECK ONLY, not a criterion: mean residual of FF xT given Pitching+
          (fixed Stuff+ ridge + pooled Location+), by batter conference. Physics is
          league-independent, so the same Pitching+ allowing more run value against one
          conference's hitters means tougher hitters. Building a criterion from Pitching+
          and grading Pitching+ against it would be circular, so it only checks method 2.

Fit choice (leakage): every adjustment is fit on the season it adjusts and nothing else,
so no evaluation season informs the criterion it is graded against. C2 is re-fit per
season here for the same reason; the pooled-frame C2 (batter effects shared across the
seasons in the frame, as scripts 01 and 12 build it) prints as a robustness row.

Evaluation (project rules): year-over-year reliability and predictive validity of the
fixed 2024-trained Stuff+ ridge on the panel of pitchers with PANEL_MIN_FF+ qualifying FF
in both seasons, for 2024->2025 and 2025->2026; paired bootstrap over pitchers on the
DIFFERENCE between criteria. Never pitch-level R^2. The deciding test is pitchers whose
conference changes between seasons.

ORIENTATION: xT, C2, G, ridge and loc are expected run value from the pitcher's view,
LOWER = BETTER. Conference strength is the batter side: HIGHER = tougher hitters. A
correctly oriented predictor correlates POSITIVELY with a next-season criterion.

Input: the pitch caches written by fc.load_pitches (--caches), read through script 12's
load_seasons, which re-derives the true year from Date and dedupes across caches.
Writes only under --workdir, which must be outside the repository. Level II data: never
commit output. Conference names are public; pitcher names are never printed.
"""
import argparse
import importlib
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import league_strength as ls

s12 = importlib.import_module("12_reliability_decomposition")

SEASONS = [2024, 2025, 2026]
PAIRS = [(2024, 2025), (2025, 2026)]
TRAIN_YEAR = 2024
CONF_LAM = 10.0     # near-free conference terms: identified by cross-league games
HOME_LAM = 1.0
CONTEXT = ["bconf", "batter", "park", "home"]   # what G subtracts; pitcher terms stay
CRITS = ["xT", "C2", "C2p", "G"]


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-ff", type=int, default=fc.PANEL_MIN_FF)
    ap.add_argument("--K", type=float, default=fc.BATTER_K,
                    help="ridge penalty on individual pitcher/batter/park effects")
    ap.add_argument("--boot-fit", type=int, default=200,
                    help="game-cluster bootstrap refits per season for conference CIs")
    ap.add_argument("--boot", type=int, default=2000,
                    help="pitcher bootstrap reps for reliability/validity/transfer")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches (comma-separated pitch-cache parquets) and --workdir "
                 "(outside the repo), or STUFFPLUS_CACHES / STUFFPLUS_WORKDIR.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


def rep_rng(seed, season, rep):
    """Generator for bootstrap rep `rep` of `season`; both methods draw from it."""
    return np.random.default_rng([seed, season, rep])


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------- data and conferences ----------------

def build_frame(args):
    df = s12.load_seasons(args.caches)
    df = fc.exclude_practice(df, where="17 load")
    missing = [c for c in fc.FEATS if c not in fc.add_derived_feats(df.head(5).copy())]
    if missing:
        sys.exit(f"caches predate model features {missing}; rebuild them with the "
                 "current fair_criterion.load_pitches before running this script")
    fc.add_xt(df)                       # pooled over all seasons, as script 12
    ls.add_home_away(df)
    games = ls.game_table(df)

    hr("WHAT THE League COLUMN HOLDS")
    per_game = df.groupby(["year", "GameID"])["League"].nunique()
    print(f"  games: {len(per_game):,}; games carrying more than one League value: "
          f"{int((per_game > 1).sum())}")
    print("  distinct League values by season: %s"
          % df.groupby("year")["League"].nunique().to_dict())
    hp = games.groupby(["year", "home"])["League"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0])
    ap = games.groupby(["year", "away"])["League"].agg(
        lambda s: s.value_counts(normalize=True).iloc[0])
    print(f"  share of a team's HOME games under its modal League: median "
          f"{hp.median():.2f}, mean {hp.mean():.2f}")
    print(f"  share of a team's AWAY games under its modal League: median "
          f"{ap.median():.2f}, mean {ap.mean():.2f}")
    print("  -> League is a property of the game and is the HOME team's conference.")
    print("     A team's conference is the modal League of its home games")
    print("     (league_strength.team_conferences documents the fallbacks).")

    confs = ls.team_conferences(games)
    cm = confs.set_index(["year", "team"])["conf"]
    df["bat_conf"] = cm.reindex(pd.MultiIndex.from_arrays(
        [df["year"], df["BatterTeam"]])).fillna(ls.UNKNOWN_CONF).values
    df["pit_conf"] = cm.reindex(pd.MultiIndex.from_arrays(
        [df["year"], df["PitcherTeam"]])).fillna(ls.UNKNOWN_CONF).values
    src = confs.set_index(["year", "team"])["source"]
    df["bat_src"] = src.reindex(pd.MultiIndex.from_arrays(
        [df["year"], df["BatterTeam"]])).fillna("unknown").values
    print("\n  team-seasons by assignment rule:")
    print(confs.groupby(["year", "source"]).size().unstack(fill_value=0).to_string())
    print("  share of PITCHES by the batting team's assignment rule:")
    print(pd.crosstab(df["year"], df["bat_src"], normalize="index").round(3).to_string())

    hr("CROSS-CONFERENCE LINKAGE (what identifies conference strength)")
    summ, pairs_by = ls.conference_links(games, confs)
    print(summ.to_string(index=False))
    for yr, (pc, per_conf, _) in pairs_by.items():
        lo = sorted(per_conf.items(), key=lambda kv: kv[1])[:4]
        print(f"  {yr}: fewest cross-conference games: "
              + ", ".join(f"{c} {n}" for c, n in lo))
    print("  components = 1 means every conference is linked to every other through")
    print("  some chain of non-conference games, so one global scale exists.")
    return df, games, confs, summ


# ---------------- criteria ----------------

def add_c2(df):
    """C2 per season (the leakage-safe version) and the pooled-frame C2p."""
    df["C2"] = np.nan
    for yr in SEASONS:
        sub = df[df["year"] == yr].copy()
        fc.add_adjusted(sub)
        df.loc[sub.index, "C2"] = sub["adjT"]
    fc.add_adjusted(df)
    df["C2p"] = df["adjT"]

    hr("THE PREMISE: what C2 subtracts between conferences")
    val = df["xT"].notna()
    for yr in SEASONS:
        m = val & (df["year"] == yr)
        lg = df[m].groupby("League")["xT"].mean()
        print(f"  {yr}: sd across conferences of League mean xT = "
              f"{100 * lg.std():.2f} per 100 pitches (range {100 * lg.min():+.2f} to "
              f"{100 * lg.max():+.2f})")


def season_blocks(sub, K):
    bk = sub["Batter"].astype(str) + "|" + sub["BatterTeam"].astype(str)
    return [ls.Block("pconf", ls.encode(sub["pit_conf"])[0], CONF_LAM),
            ls.Block("bconf", ls.encode(sub["bat_conf"])[0], CONF_LAM),
            ls.Block("pitcher", ls.encode(sub["PitcherId"])[0], K),
            ls.Block("batter", ls.encode(bk)[0], K),
            ls.Block("park", ls.encode(sub["home_team"])[0], K),
            ls.Block("home", ls.encode(sub["bat_home"])[0], HOME_LAM)]


def add_global(df, args):
    """Method 2: per-season joint fit, adjT_global, conference strength point estimates
    plus game-cluster bootstrap refits. Returns {season: (point, reps, game weights)}."""
    hr("METHOD 2: JOINT GLOBAL-MEAN FIT (per season)")
    df["G"] = np.nan
    df["opp_S"] = np.nan
    df["gcode"] = -1
    out = {}
    for yr in SEASONS:
        m = (df["year"] == yr) & df["xT"].notna()
        sub = df[m]
        blocks = season_blocks(sub, args.K)
        X = ls.design(blocks, len(sub))
        fit = ls.fit_joint(sub["xT"].values, blocks, design_cache=X)
        ctx = ls.context_effect(fit, blocks, CONTEXT)
        df.loc[sub.index, "G"] = ls.global_criterion(sub["xT"].values, ctx)
        bat = ls.context_effect(fit, blocks, ["bconf", "batter"])
        pit = ls.context_effect(fit, blocks, ["pconf", "pitcher"])
        S = ls.group_mean_centered(bat, sub["bat_conf"])
        P = ls.group_mean_centered(pit, sub["pit_conf"])
        df.loc[sub.index, "opp_S"] = sub["bat_conf"].map(S).values
        home_eff = fit.effects["home"]
        print(f"  {yr}: {len(sub):,} pitches, {[b.n_levels for b in blocks]} levels "
              f"(pconf, bconf, pitcher, batter, park, home); CG {fit.iterations} its")
        print(f"        batting-at-home effect {100 * (home_eff[-1] - home_eff[0]):+.2f} "
              f"per 100 pitches; park sd {100 * fit.effects['park'].std():.2f}")
        print(f"        sd of conference strength (batter side) {100 * S.std():.2f}, "
              f"(pitcher side) {100 * P.std():.2f} per 100 pitches")
        print(f"        corr(batter-side, pitcher-side) across conferences "
              f"{np.corrcoef(S, P.reindex(S.index))[0, 1]:+.2f} "
              f"(NEGATIVE = strong-hitting conferences also pitch well, since lower "
              f"pitcher effect = better)")
        gcodes, glev = ls.encode(sub["GameID"])
        df.loc[sub.index, "gcode"] = gcodes
        reps = []
        for b in range(args.boot_fit):
            w = ls.poisson_game_weights(gcodes, len(glev), rep_rng(args.seed, yr, b))
            fb = ls.fit_joint(sub["xT"].values, blocks, weights=w, design_cache=X,
                              x0=fit.solution)
            bb = ls.context_effect(fb, blocks, ["bconf", "batter"])
            reps.append(ls.group_mean_centered(bb, sub["bat_conf"], w))
        park = pd.Series(fit.effects["park"], index=ls.encode(sub["home_team"])[1])
        home = pd.Series(fit.effects["home"], index=ls.encode(sub["bat_home"])[1])
        out[yr] = {"S": S, "P": P, "reps": pd.DataFrame(reps), "n_games": len(glev),
                   "park": park, "home": home}
        lg = sub.groupby("League")["xT"].mean()
        both = S.index.intersection(lg.index)
        slope = np.polyfit(S[both].values, lg[both].values - lg[both].mean(), 1)[0]
        print(f"        League mean xT (what C2 subtracts) vs batter-side strength: "
              f"r = {np.corrcoef(S[both], lg[both])[0, 1]:+.2f}, slope {slope:+.2f}")
    return out


def _wdemean(v, codes, w):
    num = np.bincount(codes, weights=w * v)
    den = np.bincount(codes, weights=w)
    return v - (num / np.where(den > 0, den, 1.0))[codes]


def calibration(df, m2, args):
    """Do season-t conference strengths predict within-pitcher results in season t+1?

    For every season-(t+1) pitch against a batter whose conference was rated in season t,
    regress the criterion on the season-t strength of the batter's conference, with
    pitcher-season fixed effects (within-pitcher demeaning) and the season-t park and
    batting-at-home effects as controls. Pitcher fixed effects remove who is good; the
    variation left is the same pitcher facing different conferences, almost all of it in
    non-conference games. Nothing from season t+1 informs the strengths, so for xT a
    slope near 1 means the strengths are right in scale out of sample. For C2 the slope is
    the share of opponent strength C2 leaves in its criterion. G at t+1 is built from
    season-(t+1) strengths, so its slope is near zero almost by construction and prints
    only as a check.
    """
    hr("OUT-OF-SEASON CALIBRATION: season-t strengths vs within-pitcher results in t+1")
    rows = []
    for t0, t1 in PAIRS:
        S0, park0, home0 = m2[t0]["S"], m2[t0]["park"], m2[t0]["home"]
        s = df[(df["year"] == t1) & df["xT"].notna() & df["bat_conf"].isin(S0.index)]
        pc = ls.encode(s["PitcherId"])[0]
        x = s["bat_conf"].map(S0).values
        pk = s["home_team"].map(park0).fillna(0.0).values
        hm = s["bat_home"].map(home0).fillna(0.0).values
        gc = s["gcode"].values.astype(int)
        n_games = m2[t1]["n_games"]

        def slopes(w):
            Xd = np.column_stack([_wdemean(v, pc, w) for v in (x, pk, hm)])
            sw = np.sqrt(w)
            out = {}
            for c in ["xT", "C2", "G"]:
                yd = _wdemean(s[c].values, pc, w)
                beta, *_ = np.linalg.lstsq(Xd * sw[:, None], yd * sw, rcond=None)
                out[c] = beta[0]
            return out

        pt = slopes(np.ones(len(s)))
        reps = [slopes(ls.poisson_game_weights(gc, n_games, rep_rng(args.seed + 9, t1, b)))
                for b in range(args.boot_fit)]
        xd = _wdemean(x, pc, np.ones(len(s)))
        print(f"  strengths from {t0}, results from {t1}: {len(s):,} pitches; within-"
              f"pitcher sd of opponent strength {100 * xd.std():.2f} per 100")
        for c in ["xT", "C2", "G"]:
            se = np.std([r[c] for r in reps])
            print(f"    {c:<3} slope on season-{t0} opponent strength {pt[c]:+.3f}  "
                  f"(game-cluster SE {se:.3f})")
            rows.append({"pair": f"{t0}-{t1}", "criterion": c, "slope": pt[c], "se": se})
        d = np.array([r["xT"] - r["C2"] for r in reps])
        share = (pt["xT"] - pt["C2"]) / pt["xT"]
        print(f"    share of out-of-sample opponent strength C2 removes: {share:.2f}  "
              f"(xT-C2 slope {pt['xT'] - pt['C2']:+.3f}, SE {d.std():.3f})")
    print("  Read: xT near 1 = the joint fit's strengths are right in scale out of")
    print("  sample; the C2 slope is opponent strength the shipped criterion still carries.")
    return pd.DataFrame(rows)


def method3(df, ff, m2, args):
    """Cross-check: residual of FF xT given Pitching+, by batter conference.

    Bootstrap rep b reuses exactly the game weights of method 2's rep b, so the two
    methods' reps are paired and the rank-correlation interval is a paired one."""
    hr("METHOD 3 (CROSS-CHECK): residual given Pitching+, by batter conference")
    out = {}
    for yr in SEASONS:
        s = ff[(ff["year"] == yr) & ff["xT"].notna() & ff["loc"].notna()]
        X = s[["ridge_pred", "loc"]].values
        S3, beta = ls.residual_strength(s["xT"].values, X, s["bat_conf"])
        flag = "" if (beta[1] > 0 and beta[2] > 0) else "   <-- CHECK orientation"
        print(f"  {yr}: {len(s):,} FF; xT ~ {beta[0]:+.4f} + {beta[1]:+.3f} Stuff "
              f"+ {beta[2]:+.3f} Loc (both should be POSITIVE){flag}")
        n_games = m2[yr]["n_games"]
        gcodes = s["gcode"].values
        reps = []
        for b in range(args.boot_fit):
            w = ls.poisson_game_weights(gcodes, n_games, rep_rng(args.seed, yr, b))
            reps.append(ls.residual_strength(s["xT"].values, X, s["bat_conf"], w)[0])
        out[yr] = {"S": S3, "reps": pd.DataFrame(reps)}
    return out


def strength_tables(m2, m3):
    hr("CONFERENCE STRENGTH (batter side, run value per 100 pitches vs D1 average;"
       " HIGHER = tougher hitters)")
    rows = {}
    for yr in SEASONS:
        for name, m in (("m2", m2), ("m3", m3)):
            S = m[yr]["S"]
            lo = m[yr]["reps"].quantile(0.025)
            hi = m[yr]["reps"].quantile(0.975)
            for c in S.index:
                rows.setdefault(c, {})[(name, yr)] = (S[c], lo.get(c), hi.get(c))
    # Three-season average where present; CI from the per-rep averages.
    avg = []
    for c, d in rows.items():
        rec = {"conf": c}
        for name, m in (("m2", m2), ("m3", m3)):
            yrs = [y for y in SEASONS if (name, y) in d]
            rec[f"{name}_n"] = len(yrs)
            rec[name] = 100 * np.mean([d[(name, y)][0] for y in yrs])
            k = min(len(m[y]["reps"]) for y in yrs)
            stacked = np.mean([m[y]["reps"][c].values[:k] for y in yrs], axis=0)
            rec[f"{name}_lo"] = 100 * np.nanpercentile(stacked, 2.5)
            rec[f"{name}_hi"] = 100 * np.nanpercentile(stacked, 97.5)
        for y in SEASONS:
            rec[f"m2_{y}"] = 100 * d[("m2", y)][0] if ("m2", y) in d else np.nan
        avg.append(rec)
    t = pd.DataFrame(avg).sort_values("m2", ascending=False)
    pd.set_option("display.width", 200)
    cols = ["conf", "m2_n", "m2", "m2_lo", "m2_hi", "m3", "m3_lo", "m3_hi",
            "m2_2024", "m2_2025", "m2_2026"]
    print(t[cols].round(2).to_string(index=False))
    print("  m2 = joint global fit, m3 = Pitching+ residual (cross-check); 95% game-")
    print("  cluster bootstrap intervals on the seasons' average; m2_n = seasons present.")

    print("\n  RANK AGREEMENT, method 2 vs method 3 (Spearman across conferences):")
    for yr in SEASONS:
        a, b = m2[yr]["S"], m3[yr]["S"].reindex(m2[yr]["S"].index)
        k = min(len(m2[yr]["reps"]), len(m3[yr]["reps"]))
        bs = [spearmanr(m2[yr]["reps"].iloc[i], m3[yr]["reps"].iloc[i].reindex(
              m2[yr]["reps"].columns))[0] for i in range(k)]
        sl = np.polyfit(a.values, b.values, 1)[0]
        print(f"    {yr}: rho = {spearmanr(a, b)[0]:.3f}  (bootstrap 95% "
              f"[{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]); "
              f"slope m3 on m2 = {sl:.2f}; sd m2 {100 * a.std():.2f} vs m3 "
              f"{100 * b.std():.2f}")
    both = t[(t["m2_n"] == 3) & (t["m3_n"] == 3)]
    print(f"    three-season average, {len(both)} conferences present all three "
          f"seasons: rho = {spearmanr(both['m2'], both['m3'])[0]:.3f}")
    print("    Method 3 is expected to UNDERSTATE the spread: strong programs also")
    print("    recruit pitching skill Pitching+ does not measure, which lowers their")
    print("    opponents' residuals. Agreement on ORDER is the evidence that matters.")
    return t


# ---------------- evaluation ----------------

def pitcher_seasons(ff, min_ff):
    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({"n": g.size(), "ridge": g["ridge_pred"].mean(),
                        "loc": g["loc"].mean(), "opp_S": g["opp_S"].mean(),
                        "team": g["PitcherTeam"].agg(lambda s: s.value_counts().index[0]),
                        "conf": g["pit_conf"].agg(lambda s: s.value_counts().index[0])})
    for c in CRITS:
        tab[c] = g[c].mean()
    tab = tab[tab["n"] >= min_ff].reset_index()
    return tab


def pair_frames(tab, t0, t1):
    a = tab[tab["year"] == t0].set_index("PitcherId")
    b = tab[tab["year"] == t1].set_index("PitcherId")
    ids = a.index.intersection(b.index)
    return a.loc[ids], b.loc[ids]


def corr(u, v):
    return float(np.corrcoef(u, v)[0, 1])


def pair_metrics(a, b):
    out = {}
    for c in CRITS:
        out[("rel", c)] = corr(a[c], b[c])
        out[("stuff", c)] = corr(a["ridge"], b[c])
        out[("loc", c)] = corr(a["loc"], b[c])
        bl = 0.5 * fc.z(a["ridge"]) + 0.5 * fc.z(a[c])
        out[("stuff_incr", c)] = corr(bl, b[c]) - out[("rel", c)]
    return out


def eval_table(tab, args):
    hr("RELIABILITY AND VALIDITY (four-seam panel, Pearson; paired bootstrap on "
       "differences)")
    rng = np.random.default_rng(args.seed + 2)
    frames = {p: pair_frames(tab, *p) for p in PAIRS}
    point = {p: pair_metrics(*frames[p]) for p in PAIRS}
    boots = {p: [] for p in PAIRS}
    for p in PAIRS:
        a, b = frames[p]
        n = len(a)
        av, bv = a.reset_index(drop=True), b.reset_index(drop=True)
        for _ in range(args.boot):
            i = rng.integers(0, n, n)
            boots[p].append(pair_metrics(av.iloc[i], bv.iloc[i]))
    for p in PAIRS:
        print(f"  {p[0]}->{p[1]}: n = {len(frames[p][0])} pitchers "
              f"({args.min_ff}+ FF both seasons)")
    metrics = [("rel", "reliability r(C_t, C_t+1)"),
               ("stuff", "Stuff+ validity r(ridge_t, C_t+1)"),
               ("loc", "Location+ validity r(loc_t, C_t+1)"),
               ("stuff_incr", "Stuff+ increment over own-criterion baseline")]
    rows = []
    for key, label in metrics:
        print(f"\n  {label}")
        print(f"    {'pair':<11}" + "".join(f"{c:>8}" for c in CRITS)
              + f"{'G-C2':>9}{'SE':>7}{'G-C2p':>9}{'SE':>7}")
        both_d, both_dp = [], []
        for p in PAIRS:
            pt = point[p]
            d = np.array([bt[(key, "G")] - bt[(key, "C2")] for bt in boots[p]])
            dp = np.array([bt[(key, "G")] - bt[(key, "C2p")] for bt in boots[p]])
            both_d.append(d)
            both_dp.append(dp)
            dd = pt[(key, "G")] - pt[(key, "C2")]
            ddp = pt[(key, "G")] - pt[(key, "C2p")]
            print(f"    {p[0]}->{str(p[1])[2:]:<5}" + "".join(f"{pt[(key, c)]:>8.3f}"
                                                    for c in CRITS)
                  + f"{dd:>+9.3f}{d.std():>7.3f}{ddp:>+9.3f}{dp.std():>7.3f}")
            rows.append({"metric": key, "pair": f"{p[0]}-{p[1]}",
                         **{c: pt[(key, c)] for c in CRITS}, "G_minus_C2": dd,
                         "se": d.std()})
        avg = np.mean(both_d, axis=0)
        avgp = np.mean(both_dp, axis=0)
        ptavg = np.mean([point[p][(key, "G")] - point[p][(key, "C2")] for p in PAIRS])
        ptavgp = np.mean([point[p][(key, "G")] - point[p][(key, "C2p")] for p in PAIRS])
        print(f"    {'mean':<11}" + "".join(
            f"{np.mean([point[p][(key, c)] for p in PAIRS]):>8.3f}" for c in CRITS)
            + f"{ptavg:>+9.3f}{avg.std():>7.3f}{ptavgp:>+9.3f}{avgp.std():>7.3f}"
            + f"   P(G>C2)={(avg > 0).mean():.3f}")
        rows.append({"metric": key, "pair": "mean", **{c: np.mean(
            [point[p][(key, c)] for p in PAIRS]) for c in CRITS},
            "G_minus_C2": ptavg, "se": avg.std()})
    return frames, pd.DataFrame(rows)


def decomposition_audit(frames):
    hr("DECOMPOSITION AUDIT: where does any validity change live?")
    for p, (a, b) in frames.items():
        stay = (a["conf"] == b["conf"]) & (a["conf"] != ls.UNKNOWN_CONF)
        aa, bb = a[stay], b[stay]
        print(f"  {p[0]}->{p[1]} (conference stayers, n={int(stay.sum())}):")
        cm = aa["conf"]
        for c in ["C2", "G"]:
            r_all = corr(aa["ridge"], bb[c])
            ra = aa["ridge"] - aa.groupby("conf")["ridge"].transform("mean")
            rb = bb[c] - bb.groupby(cm.values)[c].transform("mean").values
            print(f"    {c:<3} Stuff+ validity: all {r_all:.3f}   within conference "
                  f"{corr(ra, rb):.3f}")
        d = b["G"] - b["C2"]
        print(f"    r(Stuff+_t, conference strength faced t+1) = "
              f"{corr(a['ridge'], b['opp_S']):+.3f}  (NEGATIVE = better-stuff pitchers"
              f" face tougher hitters)")
        print(f"    r(Stuff+_t, G-C2 at t+1) = {corr(a['ridge'], d):+.3f};  sd(G-C2) "
              f"{100 * d.std():.2f} vs sd(C2) {100 * b['C2'].std():.2f} per 100")


def transfer_test(tab, args):
    hr("THE DECIDING TEST: pitchers whose conference changes")
    rng = np.random.default_rng(args.seed + 3)
    parts = []
    for p in PAIRS:
        a, b = pair_frames(tab, *p)
        d = pd.DataFrame({"pid": a.index, "pair": f"{p[0]}-{p[1]}",
                          "conf0": a["conf"].values, "conf1": b["conf"].values,
                          "team_changed": (a["team"].values != b["team"].values),
                          "dS": (b["opp_S"] - a["opp_S"]).values,
                          # physical change, to control for movers whose skill changed
                          "dR": (b["ridge"] - a["ridge"]).values,
                          "dL": (b["loc"] - a["loc"]).values})
        for c in CRITS:
            d[f"{c}_0"] = a[c].values
            d[f"{c}_1"] = b[c].values
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    known = (d["conf0"] != ls.UNKNOWN_CONF) & (d["conf1"] != ls.UNKNOWN_CONF)
    d = d[known].reset_index(drop=True)
    d["mover"] = d["conf0"] != d["conf1"]
    mv = d[d["mover"]]
    print(f"  panel pitcher-pairs with known conferences: {len(d)}; movers {len(mv)} "
          f"({int(mv['team_changed'].sum())} changed team, "
          f"{int((~mv['team_changed']).sum())} realignment with the same team)")
    print("  by pair: %s" % mv.groupby("pair").size().to_dict())
    print(f"  conference strength faced, change for movers (per 100): mean "
          f"{100 * mv['dS'].mean():+.2f}, sd {100 * mv['dS'].std():.2f}; "
          f"up {int((mv['dS'] > 0).sum())}, down {int((mv['dS'] < 0).sum())}")
    print(f"  stayers' change in strength faced: sd {100 * d.loc[~d['mover'], 'dS'].std():.2f}")

    def stats(dd):
        out = {}
        for c in CRITS:
            st = dd[~dd["mover"]]
            res = np.empty(len(dd))
            # Stayer model per pair: next season from this season (absorbs regression
            # to the mean and any season-level shift).
            for pr, g in dd.groupby("pair"):
                s = st[st["pair"] == pr]
                coef = np.polyfit(s[f"{c}_0"], s[f"{c}_1"], 1)
                res[g.index.values] = (g[f"{c}_1"].values
                                       - np.polyval(coef, g[f"{c}_0"].values))
            m = dd["mover"].values
            up = m & (dd["dS"].values > 0)
            dn = m & (dd["dS"].values < 0)
            out[("up", c)] = res[up].mean()
            out[("down", c)] = res[dn].mean()
            out[("slope", c)] = np.polyfit(dd["dS"].values[m], res[m], 1)[0]
            Xc = np.column_stack([np.ones(m.sum()), dd["dS"].values[m],
                                  dd["dR"].values[m], dd["dL"].values[m]])
            out[("slope_ctl", c)] = np.linalg.lstsq(Xc, res[m], rcond=None)[0][1]
            # RMSE in units of the criterion's own spread: the criteria differ in scale
            # (G un-cancels schedule from skill and is wider), so raw RMSE is not comparable.
            out[("rmse_mv", c)] = (np.sqrt(np.mean(res[m] ** 2))
                                   / dd.loc[~dd["mover"], f"{c}_1"].std())
            out[("r_mv", c)] = corr(dd[f"{c}_0"].values[m], dd[f"{c}_1"].values[m])
            out[("r_st", c)] = corr(dd[f"{c}_0"].values[~m], dd[f"{c}_1"].values[~m])
        return out

    pt = stats(d)
    pids = d["pid"].unique()
    rows_by = {p: np.flatnonzero(d["pid"].values == p) for p in pids}
    boots = []
    for _ in range(args.boot):
        draw = rng.integers(0, len(pids), len(pids))
        take = np.concatenate([rows_by[pids[k]] for k in draw])
        dd = d.iloc[take].reset_index(drop=True)
        if dd["mover"].sum() < 10:
            continue
        boots.append(stats(dd))
    labels = [("up", "mean residual, movers to TOUGHER hitters (per 100)", 100),
              ("down", "mean residual, movers to EASIER hitters (per 100)", 100),
              ("slope", "slope of residual on change in strength faced", 1),
              ("slope_ctl", "  same, controlling change in Stuff+ and Location+", 1),
              ("rmse_mv", "RMSE predicting movers, / sd of criterion", 1),
              ("r_mv", "year-over-year r, movers", 1),
              ("r_st", "year-over-year r, stayers", 1)]
    print("\n  Residual = next-season criterion minus what the stayers' model predicts")
    print("  from this season. LOWER = better, so a POSITIVE residual for movers to")
    print("  tougher hitters means they got worse than expected. A correct strength")
    print("  adjustment drives both direction means and the slope to zero.")
    print(f"\n    {'quantity':<52}" + "".join(f"{c:>8}" for c in CRITS)
          + f"{'G-C2':>9}{'SE':>7}")
    rows = []
    for key, label, sc in labels:
        diffs = np.array([b[(key, "G")] - b[(key, "C2")] for b in boots])
        dpt = pt[(key, "G")] - pt[(key, "C2")]
        ses = {c: np.std([b[(key, c)] for b in boots]) for c in CRITS}
        print(f"    {label:<52}" + "".join(f"{sc * pt[(key, c)]:>8.3f}" for c in CRITS)
              + f"{sc * dpt:>+9.3f}{sc * diffs.std():>7.3f}")
        print(f"    {'  (bootstrap SE of each)':<52}" + "".join(
            f"{sc * ses[c]:>8.3f}" for c in CRITS))
        rows.append({"quantity": key, **{c: sc * pt[(key, c)] for c in CRITS},
                     "G_minus_C2": sc * dpt, "se": sc * diffs.std()})
    closer = np.mean([abs(b[("slope", "G")]) < abs(b[("slope", "C2")]) for b in boots])
    print(f"\n    P(G's slope closer to zero than C2's) over the bootstrap = {closer:.3f}")
    gap_c2 = np.array([b[("r_mv", "C2")] - b[("r_st", "C2")] for b in boots])
    gap_g = np.array([b[("r_mv", "G")] - b[("r_st", "G")] for b in boots])
    dd = gap_g - gap_c2
    print(f"\n    mover-minus-stayer reliability gap: C2 {pt[('r_mv', 'C2')] - pt[('r_st', 'C2')]:+.3f}"
          f", G {pt[('r_mv', 'G')] - pt[('r_st', 'G')]:+.3f}; difference "
          f"{(pt[('r_mv', 'G')] - pt[('r_st', 'G')]) - (pt[('r_mv', 'C2')] - pt[('r_st', 'C2')]):+.3f}"
          f" (SE {dd.std():.3f})")
    for label, msk in (("changed team (portal)", d["mover"] & d["team_changed"]),
                       ("same team, conference realigned", d["mover"] & ~d["team_changed"])):
        sub = d[msk]
        st = d[~d["mover"]]
        sp_ = {}
        for c in CRITS:
            res = []
            for pr, g in sub.groupby("pair"):
                s = st[st["pair"] == pr]
                coef = np.polyfit(s[f"{c}_0"], s[f"{c}_1"], 1)
                res.append(g[f"{c}_1"] - np.polyval(coef, g[f"{c}_0"]))
            res = pd.concat(res).reindex(sub.index)
            sp_[c] = (np.polyfit(sub["dS"], res, 1)[0], 100 * res.mean(),
                      100 * sub["dS"].mean())
        print(f"\n  {label}, n={len(sub)}: mean change in strength faced "
              f"{sp_['xT'][2]:+.2f} per 100")
        print("    " + "   ".join(f"{c}: slope {v[0]:+.2f}, mean resid {v[1]:+.2f}"
                                  for c, v in sp_.items()))
    return pd.DataFrame(rows), len(mv)


def main():
    args = cli()
    df, games, confs, summ = build_frame(args)
    add_c2(df)
    m2 = add_global(df, args)

    ff = fc.stuff_ridge(df)
    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    lmap = fc.PooledLocationMap(ff[ff["year"] == TRAIN_YEAR])
    ff["loc"] = lmap.apply(ff)

    m3 = method3(df, ff, m2, args)
    strength = strength_tables(m2, m3)

    tab = pitcher_seasons(ff, args.min_ff)
    frames, ev = eval_table(tab, args)
    decomposition_audit(frames)
    tr, n_movers = transfer_test(tab, args)
    cal = calibration(df, m2, args)

    # Level II: per-pitcher table stays in the workdir, outside the repo.
    tab.to_parquet(os.path.join(args.workdir, "league_strength_pitcher_seasons.parquet"),
                   index=False)
    strength.to_csv(os.path.join(args.workdir, "conference_strength.csv"), index=False)
    ev.to_csv(os.path.join(args.workdir, "reliability_validity.csv"), index=False)
    tr.to_csv(os.path.join(args.workdir, "transfer_test.csv"), index=False)
    summ.to_csv(os.path.join(args.workdir, "linkage.csv"), index=False)
    cal.to_csv(os.path.join(args.workdir, "calibration.csv"), index=False)
    print(f"\n  wrote outputs under {args.workdir}")


if __name__ == "__main__":
    main()
