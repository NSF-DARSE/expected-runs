"""Command+ inputs: every tagged bullpen pitch paired with where it actually crossed the plate.

This module produces the per-pitch pairs the web app scores. It does NOT score them.
The coach edits the target points and tolerance ellipses in the app, so scoring lives
client-side and re-runs the moment a setting changes; a score baked in here would go
stale on the first edit. What ships per pitch is: intended zone (1-5), the tag's
pitch class, the actual PlateLocSide / PlateLocHeight, and the TrackMan tagged pitch
type (usually "Undefined" in a pen, shown only when it is not). Per session it also
carries the PitcherThrows and BatterSide TrackMan recorded, because the app's default
reading of Inside/Away is "relative to the hitter side recorded for that pen".

INPUTS
------
1. Tag sessions, read exactly as the app's api stores them (saveSessionTags.js):
   one JSON document per session at `<date>/<pitcherId>/<sessionId>.json` in the
   `session-tags` blob container. `read_tag_sessions` walks a local directory with
   that same layout; `pull_tag_sessions` mirrors the container into one. Anything
   whose first path segment is not a YYYY-MM-DD date (the app keeps its Command+
   settings under `settings/`) is ignored.
2. TrackMan practice pitches with a per-pitch timestamp, under one or more roots laid
   out `<year>/<month>/<day>/**/*.csv` (the practice tree, or a folder of TrackMan
   portal exports). The time anchor in bullpen_tag_join needs `UTCDateTime`; API
   pulls flattened before 2026-09-24 do not carry it and must be re-pulled.

PAIRING
-------
bullpen_tag_join does the alignment (time anchor on the first tag, then sequence).
Two things have to be settled before it can run, because bullpen tags are not clean
(all pens can land under one TrackMan pitcher id; warm-ups are not excluded):
  - Which pitches are the candidate stream. Rows under the tagged pitcher id are
    preferred. When TrackMan has none (the one-placeholder-id case), every practice
    pitch that day is a candidate and the record says so (`pitcherMatch`).
  - Which single file and pitcher id the stream is. PitchNo restarts per file and two
    mounds can interleave, so the stream is the (file, PitcherId) of the pitch nearest
    the first tag. The join's drift bound then catches an interleaved stream.
A session the join refuses stays in the output with `status: "unjoined"` and the
reason, and no pairs. One bad session never stops a publish, and it never turns into
pairs that look fine and are not.

GUARDS
------
Practice rows only (Level == TeamExclusive); nothing here reads an outcome column, and
the payload is checked with bullpen_tag_join.assert_no_outcome_fields before it is
returned. None of this may enter training: fair_criterion.exclude_practice drops these
rows on every model load. No pitcher names are emitted; the app joins pitcherId to the
manifest. The output is Level II licensed data and is never committed.

Command+ itself (0-100 per pitch, averaged; 100 = inside the tolerance band) is
defined in the app's src/lib/commandPlus.ts.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import date as _date, timedelta
from typing import Iterable, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bullpen_tag_join as bj  # noqa: E402

VERSION = 1
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
PRACTICE_LEVEL = "TeamExclusive"

# Columns read from each candidate CSV. Nothing outcome-derived: PitchCall and
# PlayResult are deliberately absent, so they cannot leak into a pair by accident.
PITCH_COLS = ["PitchNo", "Date", "UTCDateTime", "PitcherId", "PitcherThrows", "BatterSide",
              "TaggedPitchType", "PlateLocSide", "PlateLocHeight", "PitchUID", "Level"]


# ------------------------------------------------------------------ tag input --

def read_tag_sessions(root: str) -> list[dict]:
    """Every session JSON under `root`, in the saveSessionTags blob layout.

    Only `<YYYY-MM-DD>/<pitcherId>/<sessionId>.json` is read. A file that does not
    parse is reported and skipped rather than raised on, since one corrupt record
    must not take down every other pitcher's Command+.
    """
    sessions = []
    if not root or not os.path.isdir(root):
        return sessions
    for day in sorted(os.listdir(root)):
        if not DATE_RE.match(day):
            continue
        for path in sorted(glob.glob(os.path.join(root, day, "*", "*.json"))):
            try:
                with open(path, encoding="utf-8") as f:
                    doc = json.load(f)
            except (OSError, json.JSONDecodeError) as err:
                print(f"[command] skipping unreadable tag record {os.path.basename(path)}: {err}")
                continue
            if isinstance(doc, dict) and isinstance(doc.get("tags"), list):
                sessions.append(doc)
    return sessions


def pull_tag_sessions(connection_string: str, container: str, dest: str) -> int:
    """Mirror the api's tag container into `dest`, same relative paths. Returns count.

    Read-only against Azure. Only date-prefixed session blobs are copied; the app's
    settings blob is not a session and stays in the container.
    """
    from azure.storage.blob import ContainerClient  # lazy: tests never need Azure

    client = ContainerClient.from_connection_string(connection_string, container)
    n = 0
    for blob in client.list_blobs():
        first = blob.name.split("/", 1)[0]
        if not DATE_RE.match(first) or not blob.name.endswith(".json"):
            continue
        target = os.path.join(dest, *blob.name.split("/"))
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(client.download_blob(blob.name).readall())
        n += 1
    return n


# ---------------------------------------------------------------- pitch input --

def _day_dirs(root: str, day: str) -> list[str]:
    d = _date.fromisoformat(day)
    out = []
    # The tag date is the phone's local calendar day; the tree is keyed by the
    # session's own date. Look one day either side so a late-evening pen near
    # midnight UTC still finds its file. The UTC time anchor picks the real one.
    for k in (-1, 0, 1):
        x = d + timedelta(days=k)
        out.append(os.path.join(root, f"{x:%Y}", f"{x:%m}", f"{x:%d}"))
    return out


def read_pitches_for_dates(roots: Iterable[str], days: Iterable[str]) -> pd.DataFrame:
    """Practice pitches from every CSV filed under any of `days` (+/- 1) in any root."""
    files = set()
    for root in roots:
        for day in set(days):
            for d in _day_dirs(root, day):
                files.update(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
    frames = []
    for path in sorted(files):
        try:
            head = pd.read_csv(path, nrows=0).columns
            df = pd.read_csv(path, usecols=[c for c in PITCH_COLS if c in head],
                             low_memory=False)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError):
            print(f"[command] skipping unreadable file {os.path.basename(path)}")
            continue
        if df.empty:
            continue
        df["sourceFile"] = os.path.basename(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=PITCH_COLS + ["sourceFile"])
    df = pd.concat(frames, ignore_index=True)
    if "PitchUID" in df.columns:
        has_uid = df["PitchUID"].notna()
        df = pd.concat([df[has_uid].drop_duplicates(subset="PitchUID", keep="first"),
                        df[~has_uid]], ignore_index=True)
    if "Level" in df.columns:
        df = df[df["Level"] == PRACTICE_LEVEL].reset_index(drop=True)
    return df


# ------------------------------------------------------------------- pairing --

def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)  # NaN -> None


def _str(v) -> Optional[str]:
    return None if v is None or (isinstance(v, float) and v != v) else str(v)


def _session_side(stream: pd.DataFrame, col: str) -> Optional[str]:
    """The one Left/Right value a session's rows carry in `col`, else None.

    TrackMan records one BatterSide per bullpen: the operator picks the hitter side
    for the pen (40 sessions checked 2026-09-24: RHP/RHB 26, RHP/LHB 12, LHP/LHB 2),
    so it is the recorded answer to what Inside and Away meant that day. A session
    whose rows disagree, or that carries no usable value, returns None rather than a
    majority vote; the app then falls back to the pitcher's arm side and flags it.
    """
    if col not in stream.columns:
        return None
    vals = {str(v) for v in stream[col].dropna() if str(v) in ("Left", "Right")}
    return vals.pop() if len(vals) == 1 else None


def _unjoined(base: dict, reason: str) -> dict:
    return {**base, "status": "unjoined", "reason": reason, "pairs": []}


def build_session(session: dict, pitches: pd.DataFrame, **join_kw) -> dict:
    """One session record: joined pairs, or the reason it could not be joined."""
    tags = session.get("tags") or []
    base = {
        "sessionId": str(session.get("sessionId", "")),
        "pitcherId": str(session.get("pitcherId", "")),
        "date": session.get("date"),
        "nTags": len(tags),
        "nSkipped": sum(1 for t in tags if t.get("skipped")),
        "throws": None,
        "batterSide": None,
        "pitcherMatch": None,
        "verdict": None,
        "maxDriftSeconds": None,
    }
    if not tags:
        return _unjoined(base, "session has no tags")
    if pitches.empty:
        return _unjoined(base, "no TrackMan practice pitches found for this date")
    if "UTCDateTime" not in pitches.columns or pitches["UTCDateTime"].isna().all():
        return _unjoined(base, "TrackMan pitches for this date carry no UTCDateTime, so the "
                               "tags cannot be time-anchored (re-pull or use a portal export)")

    pool = pitches[pitches["UTCDateTime"].notna()]
    same_id = pool[pool["PitcherId"].astype(str) == base["pitcherId"]]
    if not same_id.empty:
        candidates, base["pitcherMatch"] = same_id, "tagged-id"
    else:
        candidates, base["pitcherMatch"] = pool, "anchor-only"

    try:
        tag_records = bj.tags_from_session(session)
        records = bj.pitches_from_csv(candidates)
        anchor, _ = bj._find_anchor(records, tag_records[0].at,
                                    join_kw.get("anchor_window_seconds",
                                                bj.ANCHOR_WINDOW_SECONDS_DEFAULT))
        a = anchor.row
        stream = candidates[(candidates["sourceFile"] == a["sourceFile"])
                            & (candidates["PitcherId"].astype(str) == str(a["PitcherId"]))]
        result = bj.join_bullpen_tags(session, stream, **join_kw)
    except (bj.TagJoinError, KeyError, ValueError) as err:
        return _unjoined(base, str(err))

    by_no = {int(r["PitchNo"]): r for r in stream.to_dict("records")}
    base["throws"] = _session_side(stream, "PitcherThrows")
    base["batterSide"] = _session_side(stream, "BatterSide")
    base["verdict"] = result.report.verdict
    base["maxDriftSeconds"] = round(result.report.max_abs_drift_seconds, 2)

    pairs = []
    for p in result.pairs:
        if p.skipped or p.zone is None:
            continue
        row = by_no.get(p.pitch_no, {})
        pairs.append({
            "seq": p.seq,
            "zone": int(p.zone),
            "pitchClass": p.pitch_class,
            "side": _num(row.get("PlateLocSide")),
            "height": _num(row.get("PlateLocHeight")),
            "taggedType": _str(row.get("TaggedPitchType")),
            "pitchNo": p.pitch_no,
        })
    return {**base, "status": "joined", "reason": None, "pairs": pairs}


def build_command_pairs(sessions: list[dict], pitch_roots: list[str], **join_kw) -> dict:
    """The `command_pairs.json` bundle file for every tagged session found."""
    days = [s["date"] for s in sessions if isinstance(s.get("date"), str) and DATE_RE.match(s["date"])]
    pitches = read_pitches_for_dates(pitch_roots, days) if days else pd.DataFrame()
    out = []
    for s in sorted(sessions, key=lambda s: (str(s.get("date")), str(s.get("sessionId")))):
        if not (isinstance(s.get("date"), str) and DATE_RE.match(s["date"])):
            continue
        out.append(build_session(s, pitches, **join_kw))
    payload = {
        "version": VERSION,
        "context": "bullpen",
        "nSessions": len(out),
        "nJoined": sum(1 for s in out if s["status"] == "joined"),
        "sessions": out,
    }
    bj.assert_no_outcome_fields(payload)
    return payload


def empty_payload(note: str) -> dict:
    return {"version": VERSION, "context": "bullpen", "nSessions": 0, "nJoined": 0,
            "sessions": [], "note": note}


# ---------------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tags-dir", default=os.environ.get("COMMAND_TAGS_DIR"),
                    help="local mirror of the session-tags container")
    ap.add_argument("--pitch-roots", default=os.environ.get("COMMAND_PITCH_ROOTS"),
                    help=f"'{os.pathsep}'-separated roots of <year>/<month>/<day> TrackMan CSVs")
    ap.add_argument("--out", required=True, help="output JSON path (outside the repo)")
    args = ap.parse_args(argv)
    if not args.tags_dir or not args.pitch_roots:
        ap.error("--tags-dir and --pitch-roots (or COMMAND_TAGS_DIR / COMMAND_PITCH_ROOTS) required")
    payload = build_command_pairs(read_tag_sessions(args.tags_dir),
                                  args.pitch_roots.split(os.pathsep))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[command] {payload['nSessions']} sessions, {payload['nJoined']} joined -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
