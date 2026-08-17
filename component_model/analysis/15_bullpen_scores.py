"""15: Stuff+ for BULLPEN / PRACTICE pitches (no outcomes, no results, no Location+).

Scores TrackMan practice-tree sessions (Level == "TeamExclusive") against the
SAME league scale the games are scored on, by loading the already-trained model
artifacts (model_artifacts.json, produced by 14_pitcher_pages.build_model_artifact)
and applying them. Nothing here fits, refits, or retrains anything, and this data
must never enter model training: a bullpen has no hitter, no outcome, and a
velocity distribution that is not the game distribution.

WHAT CAN AND CANNOT BE COMPUTED HERE
------------------------------------
Stuff+ CAN be: it is a function of physical release/movement features only, so a
pitch thrown in a pen has a well-defined Stuff+ the moment TrackMan measures it.

Adj Results CANNOT be. There is no PlayResult, no PitchCall other than
"Undefined", no Target, no xT and no adjT. There is nothing to impute from and a
default would be a fabricated result. This script never emits one.

Location+ MUST NOT be. It is a count-conditioned run-value map fitted on game
pitches; in a pen the count is a placeholder attached to no live hitter, so a
value read off that map describes nothing that happened. This script never
emits one, and `assert_no_outcome_fields` fails the payload loudly if a field
named for either construct ever appears (see FORBIDDEN_KEYS).

SIGN CONVENTION (project-wide, verified not assumed): the ridge prediction is
expected runs from the pitcher's perspective, LOWER = better. The display
transform negates exactly once, in arsenal.to_display, so higher = better on the
100 +/- 15 scale. No negation happens anywhere else in this file.

SCALE: display moments (displayMu / displaySd / populationMeanZ) come from the
trained artifact, i.e. from the qualified D1 game population. They are NOT
recomputed on the bullpen population. If they were, a pitcher's bullpen Stuff+
would be measured against his own pen peers (here: mostly himself), which is not
what the number means. The ridge intercept is not needed and is not in the
artifact: Stuff+ = 100 - 15 * (z - populationMeanZ) . coef / displaySd, and the
intercept cancels in the (z - populationMeanZ) difference. This is the same
algebra arsenal.contributions uses, so per-feature contributions still sum to
the grade.

PITCH TYPE: practice files carry TaggedPitchType == "Undefined" for every pitch,
so there is no human tag to route the per-type model with. AutoPitchType (the
TrackMan classifier) is used instead and mapped onto the artifact's type names.
Pitches the classifier calls "Other", or whose type has no trained model, are
counted and reported as ungraded rather than folded into some other type's model.
Treat auto-tags in a pen with suspicion: classifiers key partly on velocity, and
a pen thrown at less than full effort mislabels toward slower types.

PRIMARY-FASTBALL DIFFERENTIALS (vertbreakdiff / horzbreakdiff /
velocity_differential): three of the twelve features are measured against the
pitcher's own primary fastball, defined upstream (python_files/
target_and_calculated_pipeline.py) as the pitch TYPE of his single fastest pitch,
then averaged over his pitches of that type. That baseline is recomputed HERE,
from the bullpen pitches only, per pitcher, using the identical rule.

Why bullpen-internal rather than borrowed from his game data:
  - The differentials are meant to be relative. If a pen is thrown at 90% effort,
    the fastball baseline drops with the secondary pitch, so a within-pen
    baseline largely cancels the effort scaling and keeps the differential
    describing pitch shape rather than pen intensity. Borrowing the GAME
    fastball baseline would price every pen secondary as ~5 mph softer than it
    should be and drag Stuff+ down for a reason that is about effort, not stuff.
  - It keeps this script standalone: it needs no game extract, so it can score a
    pen for an arm who has never appeared in a game.
The cost is real and is reported, not hidden: the baseline moves with the pen, so
a pitcher who did not throw his fastball that day has a baseline defined by
whatever his fastest pitch happened to be, and his differentials are not
comparable to the game ones. Sessions where the auto-classified fastest type is
not a fastball are flagged `primaryTypeIsFastball: false` on each session record.

Level II licensed TrackMan. The output file carries per-pitch rows and pitcher
names: treat it exactly like pitcher_pages.json and never commit it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import arsenal as ar

# Marker written onto every record at every level of the payload, so no consumer
# can pick up a session, a pitcher, or a single pitch and mistake it for a game.
CONTEXT = "bullpen"

# TrackMan's AutoPitchType spellings -> the artifact's per-type model names.
# "Other" is deliberately absent: an unclassified pitch is left ungraded.
AUTO_TYPE_MAP = {
    "Four-Seam": "FF",
    "FourSeamFastBall": "FF",
    "Fastball": "FF",
    "Slider": "Slider",
    "Changeup": "ChangeUp",
    "ChangeUp": "ChangeUp",
    "Curveball": "Curveball",
    "Sinker": "Sinker",
    "Two-Seam": "Sinker",
    "Cutter": "Cutter",
    "Splitter": "Splitter",
}

# Types that count as a fastball for the primary-fastball baseline sanity flag.
FASTBALL_TYPES = {"FF", "Sinker", "Cutter"}

# Physical columns TrackMan must supply for a pitch to be gradeable at all.
RAW_FEATURE_COLS = ["SpinRate", "Extension", "HorzBreak", "InducedVertBreak",
                    "EffectiveVelo", "RelHeight", "RelSide", "RelSpeed"]

# Any key whose presence in the payload would mean an outcome-derived construct
# leaked into data that has no outcomes. Checked recursively before the file is
# written, so this fails at build time rather than on a coach's screen.
FORBIDDEN_KEYS = frozenset({
    "loc", "locPlus", "locationPlus", "loc100", "locWhere", "locBaseline",
    "locFlag", "adjRes", "adjResults", "adjRes100", "adjT", "xT", "target",
    "resLadder", "runsAllowed", "expRunsAllowed", "whiff", "pitch100", "pitch",
})


class BullpenOutcomeError(ValueError):
    """Raised when a code path would emit a construct that has no meaning here."""


def assert_no_outcome_fields(node, path: str = "$") -> None:
    """Recursively reject any Adj Results / Location+ / results key.

    Case-insensitive on the key name, because the bundle uses camelCase and the
    analysis scripts use snake_case for the same quantities.
    """
    if isinstance(node, dict):
        lowered = {k.lower(): k for k in node}
        for forbidden in FORBIDDEN_KEYS:
            if forbidden.lower() in lowered:
                raise BullpenOutcomeError(
                    f"{path}.{lowered[forbidden.lower()]} is an outcome-derived field; "
                    "bullpen pitches have no outcomes, so it must not be emitted"
                )
        for k, v in node.items():
            assert_no_outcome_fields(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_outcome_fields(v, f"{path}[{i}]")


# ---------------------------------------------------------------- loading ----

def read_practice_tree(root: str) -> pd.DataFrame:
    """Every CSV under <root>/<year>/<month>/<day>/CSV/*.csv, concatenated.

    Empty or header-only files are skipped rather than raised on: a session that
    recorded nothing is a normal thing to find in a practice tree, and one of
    them must not take the whole run down. Returns an empty frame (with the
    expected columns absent) when the tree has no usable rows at all.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            print(f"skipping unreadable/empty file: {os.path.basename(path)}")
            continue
        if df.empty:
            print(f"skipping empty session: {os.path.basename(path)}")
            continue
        df["sourceFile"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Dedup, map pitch types, and engineer the 12 model features in FEATS order.

    Nothing here consults an outcome column; the only columns touched are
    identity, date and the physical measurements.
    """
    if df.empty:
        return df
    if "PitchUID" in df.columns:
        df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["ptype"] = df["AutoPitchType"].map(AUTO_TYPE_MAP)
    df["is_lhp"] = (df["PitcherThrows"] == "Left").astype(float)
    # A pen has no live hitter. BatterSide is whatever placeholder the operator
    # left in the file, so is_lhb is a nominal input here, not a fact about an
    # opponent. It stays in the vector because dropping a feature the ridge was
    # fitted with would silently change the model; its effect is reported as a
    # caveat instead.
    df["is_lhb"] = (df.get("BatterSide") == "Left").astype(float)
    df = _add_primary_fastball_diffs(df)
    return df


def _add_primary_fastball_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """vertbreakdiff / horzbreakdiff / velocity_differential, bullpen-internal.

    Same rule as python_files/target_and_calculated_pipeline.py: per pitcher,
    take the TYPE of his single fastest pitch, average that type's
    InducedVertBreak / HorzBreak / RelSpeed, and subtract. Computed over the whole
    practice tree per pitcher (not per session) so a pitcher who skipped his
    fastball on one day still has a baseline from his other pens; the session
    record flags whether that baseline is actually a fastball.
    """
    df = df.copy()
    typed = df[df["ptype"].notna() & df["RelSpeed"].notna()]
    df["primaryType"] = np.nan
    for col in ("vertbreakdiff", "horzbreakdiff", "velocity_differential"):
        df[col] = np.nan
    if typed.empty:
        return df
    fastest = typed.loc[typed.groupby("PitcherId")["RelSpeed"].idxmax(), ["PitcherId", "ptype"]]
    fastest = fastest.rename(columns={"ptype": "primaryType"}).set_index("PitcherId")["primaryType"]
    df["primaryType"] = df["PitcherId"].map(fastest)
    base = (typed.assign(primaryType=typed["PitcherId"].map(fastest))
                 .query("ptype == primaryType")
                 .groupby("PitcherId")[["InducedVertBreak", "HorzBreak", "RelSpeed"]].mean())
    df["vertbreakdiff"] = df["InducedVertBreak"] - df["PitcherId"].map(base["InducedVertBreak"])
    df["horzbreakdiff"] = df["HorzBreak"] - df["PitcherId"].map(base["HorzBreak"])
    df["velocity_differential"] = df["RelSpeed"] - df["PitcherId"].map(base["RelSpeed"])
    return df


# ---------------------------------------------------------------- scoring ----

def score_pitches(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Attach a Stuff+ column using the trained per-type artifacts.

    stuff = 100 - 15 * (z - populationMeanZ) . coef / displaySd, with z from the
    artifact's own scalerMean/scalerScale. The scaler is applied, never refitted.
    Rows whose type has no trained model, or that are missing any feature, come
    back with stuff = NaN and are reported as ungraded.
    """
    feats = list(model["featureOrder"])
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"engineered frame is missing model features: {missing}")
    out = df.copy()
    out["stuff"] = np.nan
    for tname, art in model["byPitchType"].items():
        mask = out["ptype"] == tname
        if not mask.any():
            continue
        X = out.loc[mask, feats].values.astype(float)
        z = (X - np.asarray(art["scalerMean"], float)) / np.asarray(art["scalerScale"], float)
        delta = (z - np.asarray(art["populationMeanZ"], float)) @ np.asarray(art["coef"], float)
        sd = float(art["displaySd"])
        if sd <= 0:
            raise ValueError(f"{tname}: display sd must be positive, got {sd}")
        out.loc[mask, "stuff"] = ar.DISPLAY_CENTER - ar.DISPLAY_SPREAD * delta / sd
    return out


def _type_rows(sub: pd.DataFrame) -> list[dict]:
    rows = []
    for tname, g in sub[sub["stuff"].notna()].groupby("ptype"):
        rows.append({
            "type": str(tname),
            "n": int(len(g)),
            "stuff": round(float(g["stuff"].mean()), 1),
            "avgVelo": round(float(g["RelSpeed"].mean()), 1),
        })
    rows.sort(key=lambda r: -r["n"])
    return rows


def build_records(scored: pd.DataFrame, include_pitches: bool = True) -> list[dict]:
    """One record per pitcher: session rollups, per-type rollups, per-pitch rows.

    Every level carries `context: "bullpen"`. No level carries a results or
    Location+ field; see assert_no_outcome_fields.
    """
    records = []
    for pid, sub in scored.groupby("PitcherId"):
        graded = sub[sub["stuff"].notna()]
        sessions = []
        for (date, gid), g in sub.groupby(["date", "GameID"], dropna=False):
            gg = g[g["stuff"].notna()]
            primary = g["primaryType"].dropna()
            sessions.append({
                "context": CONTEXT,
                "date": str(date),
                "sessionId": str(gid),
                "nPitches": int(len(g)),
                "nGraded": int(len(gg)),
                "stuff": round(float(gg["stuff"].mean()), 1) if len(gg) else None,
                "avgVelo": round(float(g["RelSpeed"].mean()), 1),
                "maxVelo": round(float(g["RelSpeed"].max()), 1),
                "byType": _type_rows(g),
                # Whether the primary-fastball baseline the differentials are
                # measured against is actually a fastball. False means the three
                # differential features describe a gap from a breaking ball.
                "primaryTypeIsFastball": bool(
                    len(primary) and str(primary.iloc[0]) in FASTBALL_TYPES),
            })
        sessions.sort(key=lambda s: s["date"])
        rec = {
            "context": CONTEXT,
            "pitcherId": int(pid),
            "name": str(sub["Pitcher"].iloc[0]),
            "hand": str(sub["PitcherThrows"].iloc[0])[0],
            "nPitches": int(len(sub)),
            "nGraded": int(len(graded)),
            "nUngraded": int(len(sub) - len(graded)),
            "stuff": round(float(graded["stuff"].mean()), 1) if len(graded) else None,
            "avgVelo": round(float(sub["RelSpeed"].mean()), 1),
            "byType": _type_rows(sub),
            "sessions": sessions,
        }
        if include_pitches:
            rec["pitches"] = [
                {"context": CONTEXT, "d": str(r.date), "t": str(r.ptype),
                 "v": round(float(r.RelSpeed), 1), "g": round(float(r.stuff), 1)}
                for r in graded.itertuples()
            ]
        records.append(rec)
    records.sort(key=lambda r: -r["nPitches"])
    return records


def build_payload(scored: pd.DataFrame, model: dict, model_path: str,
                  include_pitches: bool = True) -> dict:
    records = build_records(scored, include_pitches=include_pitches)
    payload = {
        "context": CONTEXT,
        "dataSource": "TrackMan practice tree (Level == TeamExclusive)",
        "generatedUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelArtifact": os.path.basename(model_path),
        "featureOrder": list(model["featureOrder"]),
        # Named here as strings, never as null-valued fields, so nothing
        # downstream can read a placeholder as a computed zero.
        "unavailableConstructs": [
            "Adj Results (no PlayResult / Target / adjT in practice data)",
            "Location+ (count-conditioned run-value map has no meaning with no live hitter)",
        ],
        "scaleNote": ("Stuff+ is on the trained D1 game population scale "
                      "(displayMu/displaySd/populationMeanZ from the artifact); "
                      "no scaler or scale was refit on bullpen pitches."),
        "pitchers": records,
    }
    assert_no_outcome_fields(payload)
    return payload


# ------------------------------------------------------------------- main ----

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--practice-tree", required=True,
                    help="root of the <year>/<month>/<day>/CSV tree")
    ap.add_argument("--model", default=os.environ.get("STUFFPLUS_MODEL_ARTIFACT"),
                    help="model_artifacts.json from the deployed bundle "
                         "(or $STUFFPLUS_MODEL_ARTIFACT)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--no-pitches", action="store_true",
                    help="omit per-pitch rows (session/type rollups only)")
    args = ap.parse_args(argv)
    if not args.model:
        ap.error("--model (or STUFFPLUS_MODEL_ARTIFACT) is required; this script "
                 "applies an existing trained model and never fits one")

    with open(args.model) as f:
        model = json.load(f)

    raw = read_practice_tree(args.practice_tree)
    if raw.empty:
        print("no usable practice rows found")
        return 1
    df = prepare(raw)
    scored = score_pitches(df, model)
    n_ungraded = int(scored["stuff"].isna().sum())
    print(f"{len(scored)} practice pitches, {len(scored) - n_ungraded} graded, "
          f"{n_ungraded} ungraded (unclassified or missing measurements)")

    payload = build_payload(scored, model, args.model,
                            include_pitches=not args.no_pitches)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    for rec in payload["pitchers"]:
        print(f"  {rec['name']:<20} n={rec['nPitches']:>5} graded={rec['nGraded']:>5} "
              f"Stuff+={rec['stuff']} avgVelo={rec['avgVelo']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
