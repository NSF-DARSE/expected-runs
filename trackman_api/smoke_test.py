"""End-to-end smoke test for the TrackMan Data API.

Authenticates, discovers recent game sessions, pulls one session's plays and
balls, and reports how the real JSON lines up with the 60 columns the model
pipeline expects. Output is intentionally small (shapes, field names, a 2-row
sample, a coverage checklist) so it is safe to paste back and never dumps a full
licensed session.

Usage:
    python trackman_api/smoke_test.py                     # last 7 days
    python trackman_api/smoke_test.py --days 14
    python trackman_api/smoke_test.py --from 2024-02-01 --to 2024-02-15
    python trackman_api/smoke_test.py --session-id <guid>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import requests

from auth import Token, get_token
from config import TrackManConfig, load_config

_TIMEOUT = 60  # seconds; ball data for a full game can be large
_MAX_WINDOW_DAYS = 30  # API caps discovery windows at 30 consecutive dates

# Source of truth: REQUIRED_COLS in python_files/target_and_calculated_pipeline.py.
# Kept here as a literal so the smoke test has no dependency on the model pipeline.
REQUIRED_COLS = [
    "PitchNo", "Date", "PAofInning", "PitchofPA", "Pitcher", "PitcherId",
    "PitcherThrows", "PitcherTeam", "Batter", "BatterSide", "BatterTeam",
    "Inning", "Top/Bottom", "Outs", "Balls", "Strikes", "TaggedPitchType",
    "AutoPitchType", "PitchCall", "TaggedHitType", "PlayResult", "OutsOnPlay",
    "RunsScored", "RunnerOn1B", "RunnerOn2B", "RunnerOn3B", "GameState",
    "RunsRemaining", "ExpectedRuns", "Target", "RelSpeed", "SpinRate",
    "Extension", "HorzBreak", "InducedVertBreak", "SpinAxis", "EffectiveVelo",
    "RelHeight", "RelSide", "FastestPitchType", "MaxRelSpeed",
    "Avg_InducedVertBreak_FastestType", "Avg_HorzBreak_FastestType",
    "Avg_RelSpeed_FastestType", "vertbreakdiff", "horzbreakdiff",
    "velocity_differential", "VertBreak", "PlateLocHeight", "PlateLocSide",
    "ExitSpeed", "Angle", "Direction", "Distance", "HangTime",
    "GameID", "PitchUID", "Level", "League",
]

# Columns the model computes downstream, not sourced directly from the API.
DERIVED_COLS = {
    "GameState", "RunsRemaining", "ExpectedRuns", "Target",
    "FastestPitchType", "MaxRelSpeed",
    "Avg_InducedVertBreak_FastestType", "Avg_HorzBreak_FastestType",
    "Avg_RelSpeed_FastestType", "vertbreakdiff", "horzbreakdiff",
    "velocity_differential",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TrackMan Data API smoke test")
    p.add_argument("--days", type=int, default=7,
                   help="Look back this many days from now (default 7).")
    p.add_argument("--from", dest="date_from",
                   help="UTC start date YYYY-MM-DD (overrides --days).")
    p.add_argument("--to", dest="date_to",
                   help="UTC end date YYYY-MM-DD (overrides --days).")
    p.add_argument("--session-id",
                   help="Pull this session instead of the first discovered.")
    return p.parse_args()


def resolve_window(args: argparse.Namespace) -> tuple[str, str]:
    """Return (utcDateFrom, utcDateTo) ISO strings, clamped to 30 days."""
    if args.date_from and args.date_to:
        start = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc)
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    if (end - start).days > _MAX_WINDOW_DAYS:
        raise SystemExit(
            f"Date window exceeds the API's {_MAX_WINDOW_DAYS}-day limit. "
            "Narrow --from/--to or --days."
        )
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    return start.strftime(fmt), end.strftime(fmt)


def auth_header(token: Token) -> dict:
    return {"Authorization": f"Bearer {token.access_token}"}


def discover_game_sessions(cfg: TrackManConfig, token: Token,
                           date_from: str, date_to: str) -> list[dict]:
    resp = requests.post(
        f"{cfg.data_base}/discovery/game/sessions",
        headers={**auth_header(token), "Content-Type": "application/json"},
        json={"sessionType": "All", "utcDateFrom": date_from, "utcDateTo": date_to},
        timeout=_TIMEOUT,
    )
    _raise_for_status(resp, "discovery/game/sessions")
    return resp.json()


def get_json(cfg: TrackManConfig, token: Token, path: str) -> object:
    resp = requests.get(f"{cfg.data_base}/{path}",
                        headers=auth_header(token), timeout=_TIMEOUT)
    _raise_for_status(resp, path)
    return resp.json()


def _raise_for_status(resp: requests.Response, label: str) -> None:
    if not resp.ok:
        raise RuntimeError(
            f"TrackMan {label} failed ({resp.status_code}): {resp.text[:500]}"
        )


def leaf_keys(obj: object, out: set[str]) -> None:
    """Collect all leaf (scalar-valued) key names, recursing into dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                leaf_keys(v, out)
            else:
                out.add(k.lower())
    elif isinstance(obj, list):
        for item in obj:
            leaf_keys(item, out)


def describe(label: str, records: object) -> set[str]:
    """Print count, top-level fields, a 2-row sample; return all leaf keys."""
    print(f"\n=== {label} ===")
    if not isinstance(records, list):
        records = [records]
    print(f"records: {len(records)}")
    if not records:
        return set()
    top = sorted(records[0].keys()) if isinstance(records[0], dict) else []
    print(f"top-level fields: {top}")
    sample = records[:2]
    print("sample (2 rows):")
    print(json.dumps(sample, indent=2, default=str)[:4000])
    keys: set[str] = set()
    for rec in records[:50]:  # scan a few for keys that only appear on some rows
        leaf_keys(rec, keys)
    return keys


def report_coverage(all_keys: set[str]) -> None:
    """For each model column, note whether a same-named leaf key was found."""
    print("\n=== column coverage vs REQUIRED_COLS ===")
    print("(match = a leaf key with the same name exists in plays/balls JSON)")
    found, missing, derived = [], [], []
    for col in REQUIRED_COLS:
        if col in DERIVED_COLS:
            derived.append(col)
        elif col.lower().replace("/", "") in {k.replace("/", "") for k in all_keys} \
                or col.lower() in all_keys:
            found.append(col)
        else:
            missing.append(col)
    print(f"\nDIRECT MATCH ({len(found)}): {found}")
    print(f"\nDERIVED downstream ({len(derived)}): {derived}")
    print(f"\nNO OBVIOUS SOURCE ({len(missing)}): {missing}")
    print("\nNote: 'no obvious source' often just means a different JSON name "
          "or a nested path. Use the field lists above to map them by hand.")


def main() -> None:
    args = parse_args()
    cfg = load_config()
    date_from, date_to = resolve_window(args)

    print("Authenticating...")
    token = get_token(cfg)
    print("OK, got access token.")

    print(f"\nDiscovering game sessions {date_from} .. {date_to}")
    sessions = discover_game_sessions(cfg, token, date_from, date_to)
    print(f"found {len(sessions)} session(s)")
    for s in sessions[:20]:
        home = s.get("homeTeam", {}).get("shortName", "?")
        away = s.get("awayTeam", {}).get("shortName", "?")
        print(f"  {s.get('gameDateLocal', '?')}  {away} @ {home}  id={s.get('sessionId')}")

    if not sessions and not args.session_id:
        print("\nNo sessions in this window; try a wider --days or a known --from/--to.")
        return

    session_id = args.session_id or sessions[0]["sessionId"]
    print(f"\nPulling data for session {session_id}")

    plays = get_json(cfg, token, f"data/game/plays/{session_id}")
    balls = get_json(cfg, token, f"data/game/balls/{session_id}")

    all_keys: set[str] = set()
    all_keys |= describe("PLAYS", plays)
    all_keys |= describe("BALLS", balls)
    report_coverage(all_keys)


if __name__ == "__main__":
    main()
