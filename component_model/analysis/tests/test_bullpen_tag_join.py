"""Contract tests for bullpen_tag_join.py.

This module is the missing link before Command+ can exist: it pairs a coach's
intended-location tags (phone taps during a bullpen) to the TrackMan pitches
of that same bullpen. The whole point of the module is that a naive
tag-n-to-pitch-n join is wrong the moment TrackMan recorded a warm-up pitch
the coach never tagged, and that a bad join must be loud (raise) rather than
merely look fine. These tests protect exactly that: warm-up skipping, skipped
taps, insufficient/ambiguous/excessive-drift refusals, the no-outcome-fields
guard, and that an off-by-one alignment leaves visible evidence rather than
disappearing into an apparently-clean join.

One test (`test_real_export_anchors_past_warmups`) drives the module against
the real bullpen CSV extracted for this task, if it is present on disk at the
absolute scratch path recorded below. It is skipped (not failed) when that
path is unavailable, since the file is licensed TrackMan data (Level II) and
lives outside the repo by design -- it must never be copied into the repo or
committed.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bullpen_tag_join as bj

REAL_BP_CSV = (
    r"C:\Users\jackdav\AppData\Local\Temp\claude\C--Users-jackdav-repos-baseball-stuff-plus"
    r"\ac4efe49-1c39-4819-a422-9b522379fc23\scratchpad\bp.csv"
)

BASE_T = datetime(2026, 8, 18, 14, 42, 31, tzinfo=timezone.utc)
CADENCE = timedelta(seconds=12)


# ---------------------------------------------------------------- builders --

def _pitch_df(n, start=BASE_T, cadence=CADENCE, start_pitch_no=1, level="TeamExclusive"):
    """n evenly-spaced TrackMan pitches, PitchNo starting at start_pitch_no."""
    rows = []
    for i in range(n):
        t = start + i * cadence
        pitch_no = start_pitch_no + i
        rows.append({
            "PitchNo": pitch_no,
            "Date": t.strftime("%Y-%m-%d"),
            "Time": t.strftime("%H:%M:%S.%f")[:-4],
            "UTCDateTime": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "PitchUID": f"uid-{pitch_no}",
            "PitcherId": 823910,
            "AutoPitchType": "Fastball",
            "RelSpeed": 90.0,
            "PlateLocSide": 0.0,
            "PlateLocHeight": 2.5,
            "Level": level,
        })
    return pd.DataFrame(rows)


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _session(tags, session_id="2026-08-18__823910__T104231Z"):
    return {
        "sessionId": session_id, "pitcherId": "823910",
        "pitcherName": "Callaway, Andrew", "date": "2026-08-18",
        "tags": tags,
    }


def _tag(seq, at, zone=1, zone_label="Inside", pitch_class="fastball", skipped=False):
    return {"seq": seq, "zone": (None if skipped else zone),
            "zoneLabel": (None if skipped else zone_label),
            "pitchClass": (None if skipped else pitch_class),
            "at": _iso(at), "skipped": skipped}


# ------------------------------------------------------------------- tests --

def test_clean_join_no_warmups():
    """Every pitch is tagged, timestamps line up exactly: zero drift, no
    warm-ups to skip, one pair per tag."""
    df = _pitch_df(6)
    tags = [_tag(i + 1, BASE_T + i * CADENCE) for i in range(6)]
    result = bj.join_bullpen_tags(_session(tags), df)

    assert result.report.anchor_pitch_no == 1
    assert result.report.n_pairs == 6
    assert result.report.max_abs_drift_seconds == pytest.approx(0.0, abs=1e-3)
    assert result.report.n_pitches_without_tag == 0
    assert [p.pitch_no for p in result.pairs] == [1, 2, 3, 4, 5, 6]
    assert result.report.verdict.startswith("OK")


def test_anchor_skips_warmups_before_first_tag():
    """TrackMan recorded 3 warm-ups before the coach started tagging. The
    first tag's timestamp must anchor to the 4th TrackMan pitch, not the 1st,
    and every following tag must map sequentially from there."""
    df = _pitch_df(10)  # PitchNo 1..10, pitches[0..2] are warm-ups
    first_tagged_pitch_time = BASE_T + 3 * CADENCE  # pitch #4
    tags = [_tag(i + 1, first_tagged_pitch_time + i * CADENCE) for i in range(7)]
    result = bj.join_bullpen_tags(_session(tags), df)

    assert result.report.anchor_pitch_no == 4
    assert [p.pitch_no for p in result.pairs] == [4, 5, 6, 7, 8, 9, 10]
    assert result.report.max_abs_drift_seconds == pytest.approx(0.0, abs=1e-3)
    assert result.report.n_pitches_without_tag == 0


def test_skipped_tag_consumes_position_without_zone():
    """A `skipped` tag holds its seq position (still consumes a pitch) but
    contributes no intended zone."""
    df = _pitch_df(5)
    tags = [
        _tag(1, BASE_T + 0 * CADENCE),
        _tag(2, BASE_T + 1 * CADENCE, skipped=True),
        _tag(3, BASE_T + 2 * CADENCE),
        _tag(4, BASE_T + 3 * CADENCE),
        _tag(5, BASE_T + 4 * CADENCE),
    ]
    result = bj.join_bullpen_tags(_session(tags), df)

    assert result.report.n_skipped_tags == 1
    skipped_pair = next(p for p in result.pairs if p.seq == 2)
    assert skipped_pair.skipped is True
    assert skipped_pair.zone is None
    assert skipped_pair.zone_label is None
    assert skipped_pair.pitch_no == 2  # position still consumed
    others = [p for p in result.pairs if p.seq != 2]
    assert all(p.zone is not None for p in others)


def test_more_tags_than_pitches_raises():
    df = _pitch_df(3)
    tags = [_tag(i + 1, BASE_T + i * CADENCE) for i in range(5)]
    with pytest.raises(bj.InsufficientPitchesError):
        bj.join_bullpen_tags(_session(tags), df)


def test_ambiguous_anchor_raises():
    """Two TrackMan pitches equally close to the first tag's timestamp: the
    anchor cannot be chosen without guessing, so this must refuse."""
    df = _pitch_df(4)
    # First tag lands exactly between pitch #1 (BASE_T) and pitch #2
    # (BASE_T + CADENCE): both are 6s away, a genuine tie.
    midpoint = BASE_T + CADENCE / 2
    tags = [_tag(1, midpoint), _tag(2, midpoint + CADENCE)]
    with pytest.raises(bj.AmbiguousAnchorError):
        bj.join_bullpen_tags(_session(tags), df)


def test_no_pitch_in_window_raises():
    df = _pitch_df(4)
    tags = [_tag(1, BASE_T + timedelta(minutes=10))]
    with pytest.raises(bj.AmbiguousAnchorError):
        bj.join_bullpen_tags(_session(tags), df)


def test_excessive_drift_raises():
    """One tap in the middle of the sequence lands nowhere near its paired
    pitch (a mis-tap that was never marked `skipped`); this must exceed the
    drift bound and refuse rather than emit a broken join."""
    df = _pitch_df(6)
    tags = [_tag(i + 1, BASE_T + i * CADENCE) for i in range(6)]
    # Corrupt tag seq=4's timestamp to be way off (60s from its paired pitch).
    tags[3]["at"] = _iso(BASE_T + 3 * CADENCE + timedelta(seconds=60))
    with pytest.raises(bj.ExcessiveDriftError):
        bj.join_bullpen_tags(_session(tags), df)


def test_growing_drift_is_surfaced_not_averaged_away():
    """Drift that grows monotonically across the session (the signature of a
    missed tap that was never corrected) must be flagged in the report even
    when it stays inside the hard bound -- it must not be silently averaged
    into a clean-looking mean."""
    df = _pitch_df(8)
    tags = []
    for i in range(8):
        # Each tap arrives a little later relative to its pitch than the last:
        # 0s, 2s, 4s, ... 14s of creeping drift, well under the 30s bound but
        # unmistakably a trend rather than jitter.
        drift = i * 2.0
        tags.append(_tag(i + 1, BASE_T + i * CADENCE + timedelta(seconds=drift)))
    result = bj.join_bullpen_tags(_session(tags), df)

    assert result.report.drift_growing is True
    assert "WARNING" in result.report.verdict
    assert result.report.max_abs_drift_seconds < bj.MAX_DRIFT_SECONDS_DEFAULT


def test_joined_pairs_never_carry_outcome_fields():
    """The tag<->pitch join must never acquire an outcome-derived field. The
    guard function must actually reject one if it were ever added."""
    df = _pitch_df(3)
    tags = [_tag(i + 1, BASE_T + i * CADENCE) for i in range(3)]
    result = bj.join_bullpen_tags(_session(tags), df)

    clean = bj._pairs_as_dicts(result.pairs)
    bj.assert_no_outcome_fields(clean)  # must not raise

    poisoned = [dict(clean[0])]
    poisoned[0]["adjT"] = -0.02
    with pytest.raises(bj.BullpenTagOutcomeError):
        bj.assert_no_outcome_fields(poisoned)


def test_off_by_one_would_be_detected():
    """Mid-session, TrackMan records one extra pitch that nobody tapped
    `skipped` for (a genuine missed tap, not a deliberately-marked one). From
    that point on, the coach's app keeps numbering sequentially unaware of
    the gap, so every tag from there on is one pitch behind where the naive
    sequential rule assigns it.

    The anchor and the first few pairs are still exactly correct (drift 0);
    the point of this test is that the report does not average that away --
    it shows a sharp, sustained step up to a full cadence interval of drift
    for every pair after the missed pitch, which is what an off-by-one looks
    like in the evidence rather than in a human's gut feeling about the
    rankings.
    """
    df = _pitch_df(9)  # PitchNo 1..9, one real pitch every 12s
    # Correct correspondence to pitch 2,3,4,5 (pitch 1 is a warm-up).
    tags = [_tag(i + 1, BASE_T + (i + 1) * CADENCE) for i in range(4)]
    # Pitch 6 is thrown and never tagged (missed tap, no `skipped` entry for
    # it) -- so real taps 5, 6, 7 actually happened at pitch 7, 8, 9, but the
    # sequential rule (unaware of the gap) assigns them pool positions 5, 6, 7
    # counting from the anchor (pitch 2), i.e. pitch 6, 7, 8.
    tags += [_tag(5 + i, BASE_T + (6 + i) * CADENCE) for i in range(3)]

    result = bj.join_bullpen_tags(_session(tags), df)

    early_drift = [result.report.drift_seconds_by_seq[s] for s in (1, 2, 3, 4)]
    late_drift = [result.report.drift_seconds_by_seq[s] for s in (5, 6, 7)]
    assert all(d == pytest.approx(0.0, abs=1e-3) for d in early_drift)
    assert all(
        d == pytest.approx(CADENCE.total_seconds(), abs=1e-3) for d in late_drift
    )
    # The step is visible in the headline max-drift number too, not just by
    # inspecting every pair by hand.
    assert result.report.max_abs_drift_seconds == pytest.approx(
        CADENCE.total_seconds(), abs=1e-3)


def test_missing_timezone_offset_refuses_to_guess():
    """A CSV with no usable UTCDateTime column and no explicit offset must
    refuse rather than silently assume a timezone for Date/Time."""
    df = _pitch_df(3)
    df = df.drop(columns=["UTCDateTime"])
    with pytest.raises(bj.TagJoinError):
        bj.pitch_timestamps_from_csv(df)

    # With an explicit offset it works.
    ts = bj.pitch_timestamps_from_csv(df, local_utc_offset_hours=-4.0)
    assert len(ts) == 3


def test_seq_must_be_contiguous_and_one_based():
    df = _pitch_df(3)
    tags = [_tag(1, BASE_T), _tag(3, BASE_T + CADENCE)]  # missing seq=2
    with pytest.raises(bj.TagJoinError):
        bj.join_bullpen_tags(_session(tags), df)


@pytest.mark.skipif(not os.path.exists(REAL_BP_CSV),
                     reason="real bullpen export not present at the scratch path "
                            "for this session; licensed TrackMan data is never "
                            "committed to the repo")
def test_real_export_anchors_past_warmups():
    """Drive the join against the actual bullpen CSV extracted for this task:
    42 real pitches, one pitcher, Level == TeamExclusive, ~12s cadence. Builds
    a tag session whose first tag matches a pitch a few rows in (simulating
    TrackMan-recorded warm-ups the coach didn't tag) and checks the anchor
    lands past them with clean drift.
    """
    df = pd.read_csv(REAL_BP_CSV, low_memory=False)
    assert (df["Level"] == "TeamExclusive").all()
    assert df["PitcherId"].nunique() == 1

    pitches = bj.pitches_from_csv(df)
    assert len(pitches) == 42

    n_warmups = 3
    n_tagged = 10
    tags = [_tag(i + 1, pitches[n_warmups + i].timestamp) for i in range(n_tagged)]
    result = bj.join_bullpen_tags(_session(tags), df)

    assert result.report.anchor_pitch_no == pitches[n_warmups].pitch_no
    assert result.report.n_pairs == n_tagged
    assert result.report.max_abs_drift_seconds == pytest.approx(0.0, abs=1e-3)
    assert [p.pitch_no for p in result.pairs] == [
        pitches[n_warmups + i].pitch_no for i in range(n_tagged)
    ]
