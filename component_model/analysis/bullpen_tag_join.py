"""Join a coach's intended-location tags to the TrackMan pitches of a bullpen.

This is the missing link before Command+ (intended location vs actual
location) can be computed at all: it produces `(tag, pitch)` pairs, nothing
more. It does not compute Command+, does not fit anything, and never touches a
model, a feature set, or a score.

THE PROBLEM THIS SOLVES
------------------------
A manager tags each pitch's intended location from a phone during a bullpen
(zone 1-5, see ZONE constants below). TrackMan independently records every
pitch thrown, including warm-ups the manager never tags. So tag `seq == 1` is
NOT necessarily `PitchNo == 1` in the TrackMan file: a naive positional join
(tag n <-> row n) silently mislabels the entire session the moment there is a
single warm-up pitch, and nothing about the output would look wrong.

THE JOIN, IN TWO STEPS
----------------------
1. ANCHOR: the wall-clock time of the FIRST tag is matched to the nearest
   TrackMan pitch within a bounded window (`anchor_window_seconds`). This is a
   nearest-match, not an equality test, because the tap happens near the
   pitch, not on it. If no pitch falls inside the window, or more than one
   pitch is equally close, the anchor is ambiguous and this code refuses
   (`AmbiguousAnchorError`) rather than picking one.
2. SEQUENTIAL: from the anchor pitch onward, tag `seq == n` maps to the nth
   TrackMan pitch at or after the anchor, in `PitchNo` order. A `skipped` tag
   still consumes that pitch position; it carries no intended zone.

TIMEZONES -- do not guess; read the CSV before assuming
---------------------------------------------------------
Tag `at` values are ISO-8601 UTC. TrackMan's practice CSV was checked (see
component_model/analysis/tests/test_bullpen_tag_join.py, the fixture built
from a real bullpen export) and, in this export, carries a fully-populated
`UTCDateTime` column (e.g. "2026-08-18T14:42:31.6000000Z") alongside the local
`Date`/`Time` columns -- so no offset needs to be guessed at all when that
column is present, and it is used by default.

If a future export is missing `UTCDateTime`, this module does NOT silently
assume a timezone for `Date`/`Time`. Instead the caller must pass an explicit
`local_utc_offset_hours` (e.g. -4.0 for EDT) to `pitch_timestamps_from_csv`;
omitting it in that situation raises `TagJoinError` naming exactly what is
missing. There is no silent default offset anywhere in this module.

BOUNDS, AND WHY THESE NUMBERS
------------------------------
Pens run roughly one pitch per 12 seconds (per the task's stated real bullpen
characterization). Two constants follow from that cadence, chosen to be wide
enough for ordinary tap latency but narrow enough to still catch a real
misalignment:

  ANCHOR_WINDOW_SECONDS_DEFAULT = 20
      Covers a tap up to ~1.5 pitch-intervals late/early, while staying under
      2 full intervals (24s) so it can't reach past an adjacent pitch and
      create a false tie with the intended one.

  MAX_DRIFT_SECONDS_DEFAULT = 30
      Per-pair drift (tag time minus its paired pitch's time, after anchoring)
      is expected to stay near the anchor's own offset if the sequential
      pairing is correct. A drift beyond 2.5 pitch-intervals from the anchor
      offset means some tap in the middle of the sequence landed on the wrong
      pitch (e.g. a missed tap that was not marked `skipped`), and this code
      refuses rather than reporting a plausible-looking but wrong join.

Both are keyword parameters with the defaults above; callers with a different
cadence should pass their own.

OUTCOME-DATA GUARD
-------------------
A tag record carries no outcome data, and this join must never introduce any
(follows the FORBIDDEN_KEYS / assert_no_outcome_fields precedent in
`15_bullpen_scores.py`). `assert_no_outcome_fields` here is a thin re-export of
the same idea, applied to the pair records this module builds.

PRACTICE-DATA IDENTIFICATION
------------------------------
Practice data is identified by `Level == "TeamExclusive"`, exactly as
`15_bullpen_scores.read_practice_tree` / scoring already assume. This module
does not repeat that filtering logic; callers are expected to have already
selected practice rows (or `join_bullpen_tags` will do it for them if a `Level`
column is present -- see `_practice_only`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

# ---------------------------------------------------------------- constants --

# Zone codes, mutually exclusive, as tagged from the coach's phone.
ZONE_INSIDE = 1
ZONE_MIDDLE = 2
ZONE_AWAY = 3
ZONE_OFF = 4       # deliberate chase pitch the coach called for, NOT a miss
ZONE_ELEVATED_OR_BURIED = 5  # Elevated if fastball, Buried if offspeed

VALID_ZONES = frozenset({ZONE_INSIDE, ZONE_MIDDLE, ZONE_AWAY, ZONE_OFF,
                          ZONE_ELEVATED_OR_BURIED})

ANCHOR_WINDOW_SECONDS_DEFAULT = 20.0
MAX_DRIFT_SECONDS_DEFAULT = 30.0

# Same intent as 15_bullpen_scores.FORBIDDEN_KEYS: a joined tag<->pitch record
# must never acquire a result/outcome field. Tag records have no outcome data
# to begin with; this exists so a future edit to this module can't quietly
# start attaching one (e.g. by merging in PlayResult "for convenience").
FORBIDDEN_KEYS = frozenset({
    "loc", "locPlus", "locationPlus", "loc100", "locWhere", "locBaseline",
    "locFlag", "adjRes", "adjResults", "adjRes100", "adjT", "xT", "target",
    "resLadder", "runsAllowed", "expRunsAllowed", "whiff", "pitch100", "pitch",
    "playresult", "pitchcall", "korbb", "outsonplay", "runsscored",
})


class TagJoinError(ValueError):
    """Base class: this join is refusing rather than guessing."""


class AmbiguousAnchorError(TagJoinError):
    """No TrackMan pitch was uniquely nearest the first tag's timestamp."""


class InsufficientPitchesError(TagJoinError):
    """More tags were recorded than TrackMan pitches are available to pair."""


class ExcessiveDriftError(TagJoinError):
    """A tag/pitch pair's timing drift exceeded the sane bound."""


class BullpenTagOutcomeError(ValueError):
    """A joined record would carry an outcome-derived field. See FORBIDDEN_KEYS."""


def assert_no_outcome_fields(node: Any, path: str = "$") -> None:
    """Recursively reject any Adj Results / Location+ / result key.

    Case-insensitive on the key name, matching the 15_bullpen_scores.py
    precedent, since this module's records are camelCase and the rest of the
    codebase mixes camelCase and snake_case for the same quantities.
    """
    if isinstance(node, dict):
        lowered = {k.lower(): k for k in node}
        for forbidden in FORBIDDEN_KEYS:
            if forbidden.lower() in lowered:
                raise BullpenTagOutcomeError(
                    f"{path}.{lowered[forbidden.lower()]} is an outcome-derived "
                    "field; a tag join has no outcomes, so it must not appear"
                )
        for k, v in node.items():
            assert_no_outcome_fields(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            assert_no_outcome_fields(v, f"{path}[{i}]")


# ------------------------------------------------------------ data classes --

@dataclass(frozen=True)
class TagRecord:
    seq: int
    zone: Optional[int]
    zone_label: Optional[str]
    pitch_class: Optional[str]
    at: datetime           # tz-aware UTC
    skipped: bool


@dataclass(frozen=True)
class PitchRecord:
    pitch_no: int
    pitch_uid: Any
    timestamp: datetime    # tz-aware UTC
    row: dict              # raw CSV row, kept opaque; never inspected for outcomes


@dataclass(frozen=True)
class TagPitchPair:
    seq: int
    zone: Optional[int]
    zone_label: Optional[str]
    pitch_class: Optional[str]
    skipped: bool
    pitch_no: int
    pitch_uid: Any
    tag_at: datetime
    pitch_at: datetime
    drift_seconds: float   # tag_at - pitch_at, signed


@dataclass(frozen=True)
class JoinReport:
    """Quality evidence for one join. Inspect this before trusting the pairs."""
    session_id: str
    pitcher_id: Any
    anchor_pitch_no: int
    anchor_offset_seconds: float       # tag_at - pitch_at at the anchor
    anchor_window_seconds: float
    n_tags: int
    n_skipped_tags: int
    n_pitches_available: int           # TrackMan pitches at/after the anchor
    n_pairs: int
    n_tags_without_pitch: int
    n_pitches_without_tag: int         # untagged pitches after the last tag
    max_abs_drift_seconds: float
    drift_seconds_by_seq: dict         # {seq: drift_seconds}
    drift_growing: bool                # monotonic-increase signature
    verdict: str                       # human-readable, see _build_verdict


@dataclass(frozen=True)
class JoinResult:
    report: JoinReport
    pairs: list  # list[TagPitchPair]


# ------------------------------------------------------------- CSV parsing --

def pitch_timestamps_from_csv(
    df: pd.DataFrame,
    local_utc_offset_hours: Optional[float] = None,
) -> pd.Series:
    """Return a tz-aware UTC timestamp Series aligned to df's index.

    Prefers the CSV's own `UTCDateTime` column (ISO-8601, already UTC) when
    present and non-null for every row, since that requires no assumption
    about timezone at all. Falls back to `Date` + `Time` combined with an
    explicit `local_utc_offset_hours` ONLY if the caller supplies one; there is
    no silent default offset.
    """
    if "UTCDateTime" in df.columns and df["UTCDateTime"].notna().all():
        return pd.to_datetime(df["UTCDateTime"], utc=True)

    if local_utc_offset_hours is None:
        raise TagJoinError(
            "no usable UTCDateTime column in this CSV, and no "
            "local_utc_offset_hours was supplied; refusing to guess a "
            "timezone offset for the Date/Time columns. Pass the offset "
            "explicitly (e.g. -4.0 for EDT) once you have confirmed it from "
            "the export's own metadata."
        )
    local = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    return local.dt.tz_localize(
        timezone(timedelta(hours=local_utc_offset_hours))
    ).dt.tz_convert(timezone.utc)


def _practice_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to practice rows, matching 15_bullpen_scores' Level check.

    A no-op (returns df unchanged) if the CSV carries no Level column, so
    already-filtered frames pass through untouched.
    """
    if "Level" not in df.columns:
        return df
    return df[df["Level"] == "TeamExclusive"]


def pitches_from_csv(
    df: pd.DataFrame,
    local_utc_offset_hours: Optional[float] = None,
) -> list:
    """Build ordered PitchRecords (by PitchNo) from a raw TrackMan CSV frame."""
    practice = _practice_only(df)
    ts = pitch_timestamps_from_csv(practice, local_utc_offset_hours=local_utc_offset_hours)
    ordered = practice.assign(_ts=ts).sort_values("PitchNo")
    pitches = []
    for row, t in zip(ordered.to_dict("records"), ordered["_ts"]):
        pitches.append(PitchRecord(
            pitch_no=int(row["PitchNo"]),
            pitch_uid=row.get("PitchUID"),
            timestamp=t.to_pydatetime(),
            row=row,
        ))
    return pitches


# -------------------------------------------------------------- tag parsing --

def tags_from_session(session: dict) -> list:
    """Parse the tags list of one session JSON document into TagRecords."""
    out = []
    for t in session["tags"]:
        at = pd.to_datetime(t["at"], utc=True).to_pydatetime()
        out.append(TagRecord(
            seq=int(t["seq"]),
            zone=(None if t.get("skipped") else t.get("zone")),
            zone_label=(None if t.get("skipped") else t.get("zoneLabel")),
            pitch_class=(None if t.get("skipped") else t.get("pitchClass")),
            at=at,
            skipped=bool(t.get("skipped", False)),
        ))
    out.sort(key=lambda r: r.seq)
    _validate_seq_contiguous(out)
    return out


def _validate_seq_contiguous(tags: list) -> None:
    expected = list(range(1, len(tags) + 1))
    actual = [t.seq for t in tags]
    if actual != expected:
        raise TagJoinError(
            f"tag seq values must be 1-based and contiguous; got {actual}"
        )


# ------------------------------------------------------------------ anchor --

def _find_anchor(
    pitches: list,
    first_tag_at: datetime,
    window_seconds: float,
) -> tuple:
    """Nearest pitch to first_tag_at within window_seconds. Refuses on a tie
    or on nothing in range. Returns (pitch, offset_seconds)."""
    candidates = []
    for p in pitches:
        offset = (first_tag_at - p.timestamp).total_seconds()
        if abs(offset) <= window_seconds:
            candidates.append((p, offset))
    if not candidates:
        raise AmbiguousAnchorError(
            f"no TrackMan pitch found within {window_seconds:.0f}s of the "
            f"first tag's timestamp ({first_tag_at.isoformat()}); cannot "
            "anchor this session without guessing"
        )
    best_abs = min(abs(off) for _, off in candidates)
    tied = [(p, off) for p, off in candidates if math.isclose(
        abs(off), best_abs, abs_tol=1e-6)]
    if len(tied) > 1:
        pitch_nos = [p.pitch_no for p, _ in tied]
        raise AmbiguousAnchorError(
            f"first tag's timestamp is equally close to {len(tied)} TrackMan "
            f"pitches ({pitch_nos}); anchor is ambiguous, refusing to guess"
        )
    return tied[0]


# ------------------------------------------------------------------- join --

def join_bullpen_tags(
    session: dict,
    pitch_df: pd.DataFrame,
    anchor_window_seconds: float = ANCHOR_WINDOW_SECONDS_DEFAULT,
    max_drift_seconds: float = MAX_DRIFT_SECONDS_DEFAULT,
    local_utc_offset_hours: Optional[float] = None,
) -> JoinResult:
    """Join one session's intended-location tags to its TrackMan pitches.

    Pure function: no file I/O, no fitting. `session` is the tag JSON document
    (dict, already parsed); `pitch_df` is the raw (or pre-filtered) TrackMan
    CSV loaded into a DataFrame for one pitcher's bullpen.

    Raises TagJoinError (or a subclass) rather than returning a join whose
    quality cannot be trusted -- see AmbiguousAnchorError,
    InsufficientPitchesError, ExcessiveDriftError.
    """
    tags = tags_from_session(session)
    if not tags:
        raise TagJoinError("session has no tags; nothing to join")

    pitches = pitches_from_csv(pitch_df, local_utc_offset_hours=local_utc_offset_hours)
    if not pitches:
        raise TagJoinError("no TrackMan pitches available to join against")

    anchor_pitch, anchor_offset = _find_anchor(
        pitches, tags[0].at, window_seconds=anchor_window_seconds)

    # Pitches at/after the anchor, in PitchNo order: the sequential pool tag
    # seq n draws its nth element from.
    pool = [p for p in pitches if p.pitch_no >= anchor_pitch.pitch_no]
    pool.sort(key=lambda p: p.pitch_no)

    n_tags_without_pitch = max(0, len(tags) - len(pool))
    if n_tags_without_pitch > 0:
        raise InsufficientPitchesError(
            f"{len(tags)} tags but only {len(pool)} TrackMan pitches at/after "
            f"the anchor (PitchNo {anchor_pitch.pitch_no}); "
            f"{n_tags_without_pitch} tag(s) have no pitch to pair with. "
            "Refusing to emit a partial join."
        )

    pairs = []
    drift_by_seq = {}
    for tag, pitch in zip(tags, pool[: len(tags)]):
        drift = (tag.at - pitch.timestamp).total_seconds()
        drift_by_seq[tag.seq] = drift
        pairs.append(TagPitchPair(
            seq=tag.seq, zone=tag.zone, zone_label=tag.zone_label,
            pitch_class=tag.pitch_class, skipped=tag.skipped,
            pitch_no=pitch.pitch_no, pitch_uid=pitch.pitch_uid,
            tag_at=tag.at, pitch_at=pitch.timestamp, drift_seconds=drift,
        ))

    max_abs_drift = max(abs(d) for d in drift_by_seq.values())
    if max_abs_drift > max_drift_seconds:
        worst_seq = max(drift_by_seq, key=lambda s: abs(drift_by_seq[s]))
        raise ExcessiveDriftError(
            f"pair at seq={worst_seq} drifted {drift_by_seq[worst_seq]:.1f}s "
            f"from its paired pitch, exceeding the {max_drift_seconds:.0f}s "
            "bound; a tap likely landed on the wrong pitch somewhere in this "
            "sequence. Refusing to emit this join."
        )

    n_pitches_without_tag = len(pool) - len(tags)
    drift_growing = _is_drift_growing(drift_by_seq)
    n_skipped = sum(1 for t in tags if t.skipped)

    verdict = _build_verdict(
        max_abs_drift, max_drift_seconds, drift_growing, n_pitches_without_tag)

    report = JoinReport(
        session_id=str(session.get("sessionId", "")),
        pitcher_id=session.get("pitcherId"),
        anchor_pitch_no=anchor_pitch.pitch_no,
        anchor_offset_seconds=anchor_offset,
        anchor_window_seconds=anchor_window_seconds,
        n_tags=len(tags),
        n_skipped_tags=n_skipped,
        n_pitches_available=len(pool),
        n_pairs=len(pairs),
        n_tags_without_pitch=0,
        n_pitches_without_tag=n_pitches_without_tag,
        max_abs_drift_seconds=max_abs_drift,
        drift_seconds_by_seq=drift_by_seq,
        drift_growing=drift_growing,
        verdict=verdict,
    )

    result = JoinResult(report=report, pairs=pairs)
    assert_no_outcome_fields(_pairs_as_dicts(pairs))
    return result


def _pairs_as_dicts(pairs: list) -> list:
    return [
        {"seq": p.seq, "zone": p.zone, "zoneLabel": p.zone_label,
         "pitchClass": p.pitch_class, "skipped": p.skipped,
         "pitchNo": p.pitch_no, "pitchUid": p.pitch_uid,
         "driftSeconds": round(p.drift_seconds, 3)}
        for p in pairs
    ]


def _is_drift_growing(drift_by_seq: dict) -> bool:
    """True when drift trends monotonically away from its starting value
    across the session -- the signature of a missed tap that was never
    corrected, rather than ordinary jitter around a fixed offset.

    Requires at least 3 points to say anything. Uses a simple non-decreasing
    (or non-increasing) check on |drift|, so a single noisy sample can't flip
    the verdict.
    """
    seqs = sorted(drift_by_seq)
    if len(seqs) < 3:
        return False
    mags = [abs(drift_by_seq[s]) for s in seqs]
    non_decreasing = all(b >= a - 1e-9 for a, b in zip(mags, mags[1:]))
    grew_meaningfully = (mags[-1] - mags[0]) > 3.0  # more than a jitter's worth
    return non_decreasing and grew_meaningfully


def _build_verdict(max_abs_drift: float, bound: float, drift_growing: bool,
                    n_pitches_without_tag: int) -> str:
    if drift_growing:
        return (f"WARNING: drift grows monotonically up to "
                 f"{max_abs_drift:.1f}s (bound {bound:.0f}s) -- looks like a "
                 "missed tap that was never corrected; inspect before trusting")
    if max_abs_drift > bound * 0.5:
        return (f"OK but watch: max drift {max_abs_drift:.1f}s is over half "
                 f"the {bound:.0f}s bound")
    note = ""
    if n_pitches_without_tag:
        note = f"; {n_pitches_without_tag} pitch(es) thrown after tagging stopped"
    return f"OK: max drift {max_abs_drift:.1f}s, well inside the {bound:.0f}s bound{note}"
