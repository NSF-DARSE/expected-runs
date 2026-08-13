"""Pitcher development page data: arsenal grades, pitch rows, outing trends.

Question: for one team's staff, how does each pitch type grade, why, where is it
being thrown, and is it moving?

Reads the same source as script 08 but grades the FULL ARSENAL (per-type ridge
models, the protocol adopted 2026-07-23) rather than four-seams only, and keeps
per-pitch rows instead of collapsing to pitcher means.

Writes <workdir>/pitcher_pages.json. Level II data: never commit that file.

SIGN CONVENTION: everything stays in pitcher's-perspective expected runs until
arsenal.to_display negates once. See arsenal.py's module docstring.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import arsenal as ar
import fair_criterion as fc

# Four-seam floor is script 06's measured value. Secondary floors are UNMEASURED;
# they reuse the four-seam number as a conservative stand-in. See the spec's
# "Honest gap" note. Do not present these as derived for non-FF types.
SAMPLE_FLOOR = 100
MIN_TYPE_PITCHES = 25   # skip a pitch type for a pitcher below this
SEASON_ROLE_YEAR = 2025  # fair_criterion relabels the year pair to 2024/2025 roles


def build_pitcher_records(fitted_by_type: dict, feats: list[str], floor_n: int, asof: str,
                          min_type_pitches: int = MIN_TYPE_PITCHES) -> list[dict]:
    """Assemble one record per pitcher from the per-type fitted results.

    min_type_pitches is a parameter rather than a module constant so tests can
    exercise the assembly on small synthetic frames.
    """
    all_ids: set = set()
    for state in fitted_by_type.values():
        all_ids.update(state["pitches"]["PitcherId"].unique())

    records = []
    for pid in sorted(all_ids):
        # First pass: which types clear the per-type minimum for this pitcher.
        # Usage is shared out over the INCLUDED types only, so the shares always
        # sum to 1 and read as "of the pitches we grade, this is the mix."
        included = []
        for tname, state in fitted_by_type.items():
            sub = state["pitches"]
            sub = sub[sub["PitcherId"] == pid]
            if len(sub) >= min_type_pitches:
                included.append((tname, state, sub))
        if not included:
            continue
        graded_total = sum(len(sub) for _, _, sub in included)

        arsenal_rows, outings, pitch_rows = [], [], []
        name = str(included[0][2]["Pitcher"].iloc[0])
        hand = str(included[0][2]["PitcherThrows"].iloc[0])[0]
        for tname, state, sub in included:
            mu, sd = state["mu"], state["sd"]
            per_outing = ar.outing_table(sub, mu, sd)
            change = ar.recent_change(per_outing, floor_n=floor_n, asof=asof)

            arsenal_rows.append({
                "type": tname,
                "n": int(len(sub)),
                "usage": float(len(sub) / graded_total),
                "stuff": float(ar.to_display(sub["ridge_pred"].mean(), mu, sd)),
                # Location+ is a fastball score only -- never emit it elsewhere.
                # It routes through to_display like every other display number, so
                # the raw run value (lower = better) is scaled and negated exactly
                # once. Emitting sub["loc"].mean() directly shipped a ~0.00x run
                # value with reversed polarity onto a 100 +/- 15 page.
                "loc": (float(ar.to_display(sub["loc"].mean(), state["loc_mu"], state["loc_sd"]))
                        if tname == "FF" else None),
                "recentChange": change,
                # Real release velocity, as context beside the pitch type -- not a
                # grade and not a trait. No ridge coefficient touches RelSpeed (the
                # model sees EffectiveVelo and a differential whose level cancels),
                # so this can never carry a percentile or a Worth column. It is the
                # mean over the same graded pitches that produce "n" and "stuff", so
                # the number a coach reads always describes the sample beside it.
                # None when the source extract has no RelSpeed column, which is
                # the state of the trimmed extract in use through 2026-08. A hard
                # failure here would block every publish over a display field the
                # page already renders as absent, so the value is optional and
                # only its correctness is enforced downstream.
                "avgVelo": (float(sub["RelSpeed"].mean())
                            if "RelSpeed" in sub.columns else None),
                # Where his Location+ came from. Fastball only, because that is
                # the only type with a Location+ at all.
                "locWhere": (ar.location_decomposition(
                    sub, state["league_cells"], state["loc_mu"], state["loc_sd"])
                    if tname == "FF" and state.get("league_cells") else None),
                # What actually happened to this pitch type, on the same display
                # scale. None when the type has too few qualifying pitchers to
                # define a scale, rather than a number resting on nothing.
                "adjRes": (float(ar.to_display(sub["adjT"].mean(),
                                               state["adj_mu"], state["adj_sd"]))
                           if state.get("adj_sd") and "adjT" in sub.columns
                           and sub["adjT"].notna().any() else None),
                "aboveFloor": bool(len(sub) >= floor_n),
                "typical": [float(v) for v in sub[feats].mean().values],
                # Percentile of each of his typical trait values against the
                # qualifying pitchers for this type. Computed here rather than in
                # the browser because it needs the reference population, which is
                # far larger than the page and is not worth shipping.
                "percentiles": [
                    ar.percentile(state["reference_features"][f].values, float(sub[f].mean()))
                    for f in feats
                ],
            })
            for _, o in per_outing.iterrows():
                outings.append({"date": str(o["date"]), "type": tname,
                                "n": int(o["n"]), "stuff": float(o["stuff"])})
            grades = ar.to_display(sub["ridge_pred"].values, mu, sd)
            # Per-pitch Location+, through the SAME transform as the season
            # number, per the one-scale rule in arsenal.py: a second scale
            # calibrated on single pitches would make a pitch and its season
            # average incomparable. Like the Stuff+ grade `g`, it spreads much
            # wider than the season figure, because the scale's moments come
            # from pitcher means. Fastball only, since Location+ is.
            loc_grades = (ar.to_display(sub["loc"].values, state["loc_mu"], state["loc_sd"])
                          if tname == "FF" else [None] * len(sub))
            dates = pd.to_datetime(sub["Date"]).dt.strftime("%Y-%m-%d").values
            for (_, p), g, lg, d in zip(sub.iterrows(), grades, loc_grades, dates):
                pitch_rows.append({
                    "d": str(d), "t": tname,
                    "x": round(float(p["PlateLocSide"]), 3),
                    "z": round(float(p["PlateLocHeight"]), 3),
                    "c": str(p["count12"]), "g": float(g),
                    "l": None if lg is None else float(lg),
                    "f": [float(p[f]) for f in feats],
                })
        arsenal_rows.sort(key=lambda r: -r["usage"])
        records.append({"pitcherId": int(pid), "name": name, "hand": hand,
                        "arsenal": arsenal_rows, "outings": outings, "pitches": pitch_rows})
    return records


def build_model_artifact(fitted_by_type: dict, feats: list[str]) -> dict:
    return {
        "featureOrder": list(feats),
        "byPitchType": {
            tname: {
                "coef": [float(v) for v in s["coef"]],
                "scalerMean": [float(v) for v in s["scaler_mean"]],
                "scalerScale": [float(v) for v in s["scaler_scale"]],
                "populationMeanZ": [float(v) for v in s["population_mean_z"]],
                "displayMu": float(s["mu"]),
                "displaySd": float(s["sd"]),
                # Location+ display moments, four-seam only. None elsewhere so the
                # per-type artifact keeps a uniform shape.
                "displayLocMu": s.get("loc_mu"),
                "displayLocSd": s.get("loc_sd"),
                "sampleFloor": SAMPLE_FLOOR,
                "nQualified": s["n_qualified"],
            }
            for tname, s in fitted_by_type.items()
        },
    }


def build_grids(pit: pd.DataFrame) -> dict:
    """Count-conditioned run-value surface, same construction as script 08.

    The training frame needs location bins before PooledLocationMap can use it,
    and binning requires non-null plate coordinates -- script 08 filters and bins
    before constructing the map, so do the same here.
    """
    train = pit[pit["PlateLocSide"].notna() & pit["PlateLocHeight"].notna()].copy()
    fc.add_loc_bins(train)
    # Earlier-season rows, expressed the same way attach_location expresses it.
    # load_pitches relabels the year pair to 2024/2025 roles, so "!= 2025" and
    # "== 2024" select identical rows today; the asymmetry was the hazard.
    train = train[(train["year"] != SEASON_ROLE_YEAR) & train["xT"].notna()]
    pooled = fc.PooledLocationMap(train)
    cmap = fc.CountLocationMap(train, "count12", 5)
    xs = np.arange(-1.25, 1.25, 0.25)
    zs = np.arange(1.0, 4.0, 0.25)
    grid = pd.DataFrame([(gx, gz) for gx in xs for gz in zs],
                        columns=["PlateLocSide", "PlateLocHeight"])
    grid["PlateLocSide"] += 0.01   # land inside the intended cell when binning
    grid["PlateLocHeight"] += 0.01
    fc.add_loc_bins(grid)
    pv = pooled.apply(grid)
    out = {"pooled": [{"x": round(float(a), 2), "z": round(float(b), 2), "v": round(float(v), 4)}
                      for a, b, v in zip(grid["gx"], grid["gz"], pv)]}
    for cnt in sorted(train["count12"].unique()):
        s = grid.copy()
        s["count12"] = cnt
        out[cnt] = [{"x": round(float(a), 2), "z": round(float(b), 2), "v": round(float(v), 4)}
                    for a, b, v in zip(grid["gx"], grid["gz"], cmap.apply(s, pv))]
    return out


def attach_location(pit: pd.DataFrame, state: dict, tags, fc_module, season_year: int,
                    floor_n: int = SAMPLE_FLOOR) -> None:
    """Attach Location+ run values to a fitted type's graded-season pitches.

    Only four-seams get a value; Location+ is a fastball score, and secondary-pitch
    location repeats year over year without predicting next-year outcomes.

    The pooled map must be TRAINED on the earlier season (year role 2024) from the
    FULL frame, then applied to the graded season. state["pitches"] holds only the
    graded season, so training off it selects zero rows and yields an all-NaN map --
    which is how this silently produced null Location+ for fastballs before.

    Also derives that type's Location+ DISPLAY MOMENTS from the qualifying
    population and stores them on state, so build_pitcher_records can route the
    raw run value through arsenal.to_display instead of emitting it bare. These
    are the same qualified-population moments 08_staff_scores.py uses for Loc100
    (its `n_ff >= 100` is this module's SAMPLE_FLOOR).

    floor_n is a parameter rather than the module constant only so tests can drive
    this on small synthetic frames; production passes SAMPLE_FLOOR.
    """
    season = state["pitches"]
    if tags is not None:
        season["loc"] = np.nan
        state["loc_mu"] = state["loc_sd"] = None
        return
    ff_all = pit[ar.type_mask(pit, tags)].copy()
    ff_all = ff_all[ff_all["PlateLocSide"].notna() & ff_all["PlateLocHeight"].notna()]
    fc_module.add_loc_bins(ff_all)
    train = ff_all[(ff_all["year"] != season_year) & ff_all["xT"].notna()]
    if train.empty:
        raise ValueError(
            "no earlier-season four-seams with xT to train the location map; "
            "Location+ would be all-NaN"
        )
    fc_module.add_loc_bins(season)
    season["loc"] = fc_module.PooledLocationMap(train).apply(season)
    if season["loc"].isna().all():
        raise ValueError("location map produced all-NaN values for four-seams")

    per_pitcher = season.groupby("PitcherId")["loc"].agg(["size", "mean"])
    loc_mu, loc_sd = ar.display_scale(
        per_pitcher["mean"].values, (per_pitcher["size"] >= floor_n).values
    )
    state["loc_mu"], state["loc_sd"] = loc_mu, loc_sd
    # The population a pitcher is compared against has to be the one the scale's
    # zero point came from, or the breakdown carries a constant offset that is
    # about the population gap rather than about him.
    state["loc_qualified_ids"] = set(per_pitcher.index[per_pitcher["size"] >= floor_n])


def main() -> int:
    args = fc.paths()
    pit = fc.load_pitches(args)
    fc.add_xt(pit)
    # Needed for per-type adjusted results: xT with the league mean and a shrunk
    # batter effect removed, so the column reflects the pitcher rather than who
    # he happened to face.
    fc.add_adjusted(pit)
    fc.add_count_cols(pit)

    fitted = {}
    for tname, tags in ar.PITCH_TYPES:
        try:
            state = ar.fit_type(pit, tags, SAMPLE_FLOOR, fc, SEASON_ROLE_YEAR)
        except ValueError as err:
            print(f"skipping {tname}: {err}")
            continue
        attach_location(pit, state, tags, fc, SEASON_ROLE_YEAR)
        if tname == "FF":
            # Snapshot the D1 comparison population HERE. The team filter further
            # down rewrites state["pitches"] to one staff, and a decomposition
            # built after that compares a pitcher against his own teammates while
            # the card calls the column D1.
            state["league_cells"] = ar.league_cell_table(
                state["pitches"], state.get("loc_qualified_ids"))
        fitted[tname] = state
        print(f"{tname}: {len(state['pitches'])} pitches, {state['n_qualified']} qualified")

    if "FF" not in fitted:
        print("no four-seam model: cannot build pages")
        return 1

    team_ids = set(fitted["FF"]["pitches"].loc[
        fitted["FF"]["pitches"]["PitcherTeam"] == args.team, "PitcherId"].unique())
    for state in fitted.values():
        state["pitches"] = state["pitches"][state["pitches"]["PitcherId"].isin(team_ids)].copy()

    asof = str(pd.to_datetime(fitted["FF"]["pitches"]["Date"]).max().date())
    records = build_pitcher_records(fitted, fc.FEATS, SAMPLE_FLOOR, asof)
    print(f"{len(records)} pitchers on {args.team}")

    payload = {
        "team": args.team,
        "season": int(args.year_pair[1]),
        "pitchTypes": list(fitted.keys()),
        "model": build_model_artifact(fitted, fc.FEATS),
        "grids": build_grids(pit),
        "pitchers": records,
    }
    out = f"{args.workdir}/pitcher_pages.json"
    with open(out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
