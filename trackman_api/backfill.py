"""Backfill game CSVs from the TrackMan Data API.

Discovers verified game sessions over a date range (auto-chunked into the
API's 30-day discovery windows), flattens each game with flatten.py, and
writes one CSV per game into the same folder layout the model pipeline
already reads (base/year/month/day/CSV/<gameID>.csv). The pipeline therefore
consumes an API backfill with zero changes -- and because each game is
written exactly once, the double-loaded-game duplication that affected the
Drive source cannot occur.

Resumable: a game whose CSV already exists is skipped, so an interrupted
backfill just gets re-run with the same arguments.

Usage:
    python trackman_api/backfill.py --from 2025-05-10 --to 2025-05-18 --out <dir>
    python trackman_api/backfill.py --from 2025-02-01 --to 2025-06-30 --out <dir> --team DEL_BLU
    python trackman_api/backfill.py ... --dry-run          # discovery + counts only

Data note: TrackMan data is licensed (Level II). The output directory must be
local/UD-controlled storage, and this script prints only session counts and
gameIDs, never pitch-level values.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime, timedelta, timezone

import requests

from auth import Token, get_token
from config import TrackManConfig, load_config
from flatten import flatten_game

_TIMEOUT = 60          # seconds per request
_MAX_ATTEMPTS = 5      # per request, then fail loudly
_BACKOFF_BASE = 2.0    # exponential backoff: 2, 4, 8, 16 s (+ jitter)
_WINDOW_DAYS = 30      # API cap on a discovery window
_RETRYABLE = {429, 500, 502, 503, 504}


class ApiClient:
    """Requests wrapper with token auto-renewal and bounded retry/backoff."""

    def __init__(self, cfg: TrackManConfig):
        self.cfg = cfg
        self.token: Token = get_token(cfg)

    def _headers(self) -> dict:
        if self.token.expired:
            self.token = get_token(self.cfg)
        return {"Authorization": f"Bearer {self.token.access_token}"}

    def _request(self, method: str, path: str, **kwargs):
        url = f"{self.cfg.data_base}/{path}"
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = requests.request(method, url, headers={**self._headers(),
                                        **kwargs.pop("headers", {})},
                                        timeout=_TIMEOUT, **kwargs)
            except requests.RequestException as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise RuntimeError(f"{path}: network error after "
                                       f"{_MAX_ATTEMPTS} attempts: {exc}") from exc
                self._sleep(attempt, None)
                continue
            if resp.ok:
                return resp.json()
            if resp.status_code in _RETRYABLE and attempt < _MAX_ATTEMPTS:
                self._sleep(attempt, resp)
                continue
            raise RuntimeError(
                f"TrackMan {path} failed ({resp.status_code}) after "
                f"{attempt} attempt(s): {resp.text[:300]}")
        raise AssertionError("unreachable")

    @staticmethod
    def _sleep(attempt: int, resp) -> None:
        delay = _BACKOFF_BASE ** attempt + random.uniform(0, 1)
        retry_after = resp.headers.get("Retry-After") if resp is not None else None
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)

    def discover(self, date_from: str, date_to: str) -> list[dict]:
        return self._request(
            "POST", "discovery/game/sessions",
            json={"sessionType": "All", "utcDateFrom": date_from,
                  "utcDateTo": date_to},
            headers={"Content-Type": "application/json"})

    def get(self, path: str):
        return self._request("GET", path)


def windows(start: datetime, end: datetime):
    """Yield (from, to) ISO pairs covering [start, end] in <=30-day chunks."""
    fmt = "%Y-%m-%dT%H:%M:%S.000Z"
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=_WINDOW_DAYS), end)
        yield cur.strftime(fmt), nxt.strftime(fmt)
        cur = nxt


def wanted(session: dict, team: str | None) -> bool:
    """Sessions to backfill: verified games with a non-private gameID.

    Matches what the final dataset actually contains (checked against
    Final_Target_Calc_1535): unverified sessions and "Private" captures are
    excluded; sessionType is NOT filtered (game 2 of a doubleheader can be
    Adhoc, e.g. 20250510-MonmouthU-2); all levels (D1..NAIA) are kept.
    Optional team filter keeps games where either side matches (shortName,
    e.g. DEL_BLU).
    """
    if not session.get("verified"):
        return False
    game_id = session.get("gameID")
    if not game_id or "Private" in game_id:
        return False
    if team:
        names = {session.get("homeTeam", {}).get("shortName"),
                 session.get("awayTeam", {}).get("shortName")}
        if team not in names:
            return False
    return True


def out_path(base: str, game_id: str) -> str:
    """base/year/month/day/CSV/<gameID>.csv from the gameID's date prefix."""
    day = datetime.strptime(game_id.split("-", 1)[0], "%Y%m%d")
    return os.path.join(base, f"{day:%Y}", f"{day:%m}", f"{day:%d}",
                        "CSV", f"{game_id}.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Backfill game CSVs from the TrackMan Data API")
    p.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--out", required=True, help="Output base directory (local storage)")
    p.add_argument("--team", help="Only games involving this team shortName (e.g. DEL_BLU)")
    p.add_argument("--dry-run", action="store_true",
                   help="Discover and count; write nothing")
    args = p.parse_args()

    start = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc)
    if start >= end:
        raise SystemExit("--from must be before --to")

    client = ApiClient(load_config())

    sessions: dict[str, dict] = {}
    for w_from, w_to in windows(start, end):
        found = client.discover(w_from, w_to)
        kept = [s for s in found if wanted(s, args.team)]
        print(f"window {w_from[:10]} .. {w_to[:10]}: "
              f"{len(found)} sessions, {len(kept)} to backfill")
        for s in kept:
            # Rare doubleheader re-captures share a gameID; first session wins,
            # matching the pipeline's keep-first PitchUID dedup.
            sessions.setdefault(s["gameID"], s)

    print(f"\ntotal games: {len(sessions)}")
    if args.dry_run:
        return

    done = skipped = failed = 0
    t0 = time.time()
    for i, (game_id, session) in enumerate(sorted(sessions.items()), 1):
        path = out_path(args.out, game_id)
        if os.path.exists(path):
            skipped += 1
            continue
        try:
            plays = client.get(f"data/game/plays/{session['sessionId']}")
            balls = client.get(f"data/game/balls/{session['sessionId']}")
            df = flatten_game(session, plays, balls)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            done += 1
        except Exception as exc:  # keep going; report at the end
            failed += 1
            print(f"  FAILED {game_id}: {exc}")
        if i % 25 == 0 or i == len(sessions):
            rate = (done + skipped + failed) / max(time.time() - t0, 1)
            print(f"  {i}/{len(sessions)} (written {done}, skipped {skipped}, "
                  f"failed {failed}, {rate:.1f} games/s)")

    print(f"\nwritten {done}, skipped (already present) {skipped}, failed {failed}")
    if failed:
        raise SystemExit(f"{failed} game(s) failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
