"""Flatten TrackMan Data API JSON into the raw-CSV-shaped pitch frame.

Turns one game session's plays + balls (plus its discovery/session metadata)
into a pandas DataFrame shaped like TrackMan's flat V3 game CSV -- the format
the existing model pipeline (python_files/target_and_calculated_pipeline.py)
already consumes. The pipeline itself is reused unchanged downstream; this
module only reproduces the raw per-pitch columns.

Join model (verified against a live game, 2025-05-16 DEL_BLU vs CAM_CAM):
  - plays: one record per tagged pitch, keyed by "playID", ordered here by
    taggerBehavior.pitchNo (contiguous 1..N within a game -- the raw CSV's
    row order).
  - balls: tracking records keyed by "playId" (lowercase d), one per tracked
    object. kind == "Pitch" carries pitch physics; kind == "Hit" carries
    batted-ball launch/landing. Both left-join onto plays; a play with no
    ball record (untagged/untracked pitch) keeps NaN physics, exactly like an
    untracked row in the raw CSV.
  - Extra balls with playIds matching no play (warmup/between-inning tracks)
    are ignored.

KorBB is not in the model's final 60-column schema but IS consumed by
Helpers.add_runner_states (walk force-advance logic), so the raw CSV shape
includes it and this flattener must too.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

# Raw per-pitch columns sourced directly from the API, in raw-CSV order.
# Everything else in the pipeline's REQUIRED_COLS is derived downstream
# (runner states, GameState, ExpectedRuns, Target, per-pitcher aggregates).
RAW_COLS = [
    "PitchNo", "Date", "PAofInning", "PitchofPA", "Pitcher", "PitcherId",
    "PitcherThrows", "PitcherTeam", "Batter", "BatterSide", "BatterTeam",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "TaggedPitchType",
    "AutoPitchType", "PitchCall", "KorBB", "TaggedHitType", "PlayResult",
    "OutsOnPlay", "RunsScored", "RelSpeed", "SpinRate", "Extension",
    "HorzBreak", "InducedVertBreak", "SpinAxis", "EffectiveVelo",
    "RelHeight", "RelSide", "VertBreak", "PlateLocHeight", "PlateLocSide",
    "ExitSpeed", "Angle", "Direction", "Distance", "HangTime",
    "GameID", "PitchUID", "Level", "League",
]


def _get(obj: dict, *path, default=None):
    """Nested dict lookup that tolerates missing intermediate keys."""
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return default
        obj = obj[key]
    return obj


def game_date(session: dict) -> str:
    """The game's local date, YYYY-MM-DD, matching the raw CSV Date column.

    Taken from the yyyymmdd prefix of the session gameID (e.g.
    "20250516-BobHannahStadium-1") because the API's per-play localDateTime
    just repeats utcDateTime and can roll past local midnight mid-game.
    """
    prefix = str(session.get("gameID", "")).split("-", 1)[0]
    return datetime.strptime(prefix, "%Y%m%d").strftime("%Y-%m-%d")


def flatten_game(session: dict, plays: list[dict], balls: list[dict]) -> pd.DataFrame:
    """Build the raw-CSV-shaped frame for one game session.

    Args:
        session: the discovery record for this game (gameID, level, league).
        plays: response of GET data/game/plays/<sessionId>.
        balls: response of GET data/game/balls/<sessionId>.

    Returns:
        DataFrame with RAW_COLS, one row per tagged pitch, ordered by PitchNo.
    """
    pitch_by_play = {}
    hit_by_play = {}
    for ball in balls:
        pid = ball.get("playId")
        if pid is None:
            continue
        # First track wins if a playId ever repeats within a kind.
        if ball.get("kind") == "Pitch":
            pitch_by_play.setdefault(pid, ball)
        elif ball.get("kind") == "Hit":
            hit_by_play.setdefault(pid, ball)

    date = game_date(session)
    game_id = session.get("gameID")
    level = _get(session, "level", "name")
    league = _get(session, "league", "shortName")

    rows = []
    for play in plays:
        pitch = _get(pitch_by_play.get(play.get("playID"), {}), "pitch", default={})
        hit = _get(hit_by_play.get(play.get("playID"), {}), "hit", default={})
        rows.append({
            "PitchNo": _get(play, "taggerBehavior", "pitchNo"),
            "Date": date,
            "PAofInning": _get(play, "taggerBehavior", "pAofinning"),
            "PitchofPA": _get(play, "taggerBehavior", "pitchofPA"),
            "Pitcher": _get(play, "pitcher", "name"),
            "PitcherId": _get(play, "pitcher", "id"),
            "PitcherThrows": _get(play, "pitcher", "throws"),
            "PitcherTeam": _get(play, "pitcher", "team"),
            "Batter": _get(play, "batter", "name"),
            "BatterSide": _get(play, "batter", "side"),
            "BatterTeam": _get(play, "batter", "team"),
            "Inning": _get(play, "gameState", "inning"),
            "Top/Bottom": _get(play, "gameState", "topBottom"),
            "Outs": _get(play, "gameState", "outs"),
            "Balls": _get(play, "gameState", "balls"),
            "Strikes": _get(play, "gameState", "strikes"),
            "TaggedPitchType": _get(play, "pitchTag", "taggedPitchType"),
            "AutoPitchType": _get(play, "pitchTag", "autoPitchType"),
            "PitchCall": _get(play, "pitchTag", "pitchCall"),
            "KorBB": play.get("korBB"),
            "TaggedHitType": _get(play, "hitTag", "taggedHitType"),
            "PlayResult": _get(play, "playResult", "playResult"),
            "OutsOnPlay": _get(play, "playResult", "outsOnPlay"),
            "RunsScored": _get(play, "playResult", "runsScored"),
            "RelSpeed": _get(pitch, "release", "relSpeed"),
            "SpinRate": _get(pitch, "release", "spinRate"),
            "Extension": _get(pitch, "release", "extension"),
            "HorzBreak": _get(pitch, "movement", "horzBreak"),
            "InducedVertBreak": _get(pitch, "movement", "inducedVertBreak"),
            "SpinAxis": _get(pitch, "movement", "spinAxis"),
            "EffectiveVelo": pitch.get("effectiveVelo"),
            "RelHeight": _get(pitch, "release", "relHeight"),
            "RelSide": _get(pitch, "release", "relSide"),
            "VertBreak": _get(pitch, "movement", "vertBreak"),
            "PlateLocHeight": _get(pitch, "location", "plateLocHeight"),
            "PlateLocSide": _get(pitch, "location", "plateLocSide"),
            "ExitSpeed": _get(hit, "launch", "exitSpeed"),
            "Angle": _get(hit, "launch", "angle"),
            "Direction": _get(hit, "launch", "direction"),
            "Distance": _get(hit, "landingFlat", "distance"),
            "HangTime": _get(hit, "landingFlat", "hangTime"),
            "GameID": game_id,
            "PitchUID": play.get("pitchUID"),
            "Level": level,
            "League": league,
        })

    df = pd.DataFrame(rows, columns=RAW_COLS)
    # Raw CSV row order: PitchNo ascending within the game.
    return df.sort_values("PitchNo", ignore_index=True)
