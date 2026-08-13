"""Pull TrackMan game CSVs into a game-date tree.

Same job as backfill.py, with the three things a season-scale backfill needs
that a week-scale one does not:

  * Concurrency, paced. Sequential pulling runs at roughly 10 seconds per game,
    so a full season (~6.5k games) is an overnight job; the data GETs are the
    slow part, so they are issued from a small thread pool. The data endpoints
    do have an hourly quota, though (measured: an unpaced 12-worker run pulled
    ~1,600 games, ~3,200 requests, in 27 minutes and then hit a wall of 429s
    that no per-request backoff could ride out). --requests-per-hour throttles
    the whole pool to stay under it, which is faster end to end than running
    flat out and then stalling.
  * An overall deadline. --timeout-hours bounds the whole run, not just each
    request, and exits non-zero when it is hit so an unattended run cannot
    stall silently. Per-request retry stays bounded with exponential backoff
    (inherited from backfill.ApiClient).
  * Idempotent writes. A game's CSV path is derived from the GAME date, never
    the fetch date, so re-pulling a date lands on the same file. Existing files
    are skipped by default and replaced (atomically, via a temp file) under
    --force. Re-running a range can therefore never produce a second copy of a
    game, which is what the older fetch-date tree under trackman_api/2026 does
    produce and what Helpers.resolve_latest_game_files exists to undo.

Layout (do not mix with the fetch-date tree):

    <out>/YYYY/MM/DD/CSV/<gameID>.csv     YYYY/MM/DD = the game's own date

Usage:
    python trackman_api/pull.py --from 2025-01-01 --to 2025-07-01 --out <dir>
    python trackman_api/pull.py --from ... --to ... --out <dir> --workers 8
    python trackman_api/pull.py --from ... --to ... --out <dir> --dry-run
    python trackman_api/pull.py --from ... --to ... --out <dir> --force

Data note: TrackMan data is licensed (Level II). --out must be local or
UD-controlled storage and must not be a tracked path in git. This script
prints counts and gameIDs only, never pitch-level values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from backfill import ApiClient, out_path, wanted, windows
from config import load_config
from flatten import flatten_game

_DISCOVERY_PACE = 5.0  # seconds between discovery windows; its quota is per-hour


class Deadline:
    """Wall-clock budget for the whole run."""

    def __init__(self, hours: float):
        self.expires_at = time.time() + hours * 3600
        self.hours = hours

    @property
    def expired(self) -> bool:
        return time.time() >= self.expires_at

    def remaining_str(self) -> str:
        return f"{(self.expires_at - time.time()) / 3600:.1f}h"


class RateLimiter:
    """Fixed-interval gate shared by every worker thread.

    Not a token bucket: a bucket lets a burst through, and a burst is exactly
    what trips TrackMan's hourly quota at the start of a run.
    """

    def __init__(self, per_hour: float):
        self.interval = 3600.0 / per_hour if per_hour > 0 else 0.0
        self._lock = threading.Lock()
        self._next_at = time.monotonic()

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.interval
        if wait:
            time.sleep(wait)


class ThreadSafeClient(ApiClient):
    """ApiClient with token renewal serialized and every call rate-limited."""

    def __init__(self, cfg, limiter: RateLimiter | None = None):
        self._token_lock = threading.Lock()
        self.limiter = limiter
        super().__init__(cfg)

    def _headers(self) -> dict:
        with self._token_lock:
            return super()._headers()

    def _request(self, method: str, path: str, **kwargs):
        if self.limiter is not None:
            self.limiter.acquire()
        return super()._request(method, path, **kwargs)


def write_game(client: ThreadSafeClient, session: dict, base: str) -> str:
    """Fetch one game and write it atomically to its game-date path."""
    game_id = session["gameID"]
    path = out_path(base, game_id)
    plays = client.get(f"data/game/plays/{session['sessionId']}")
    balls = client.get(f"data/game/balls/{session['sessionId']}")
    df = flatten_game(session, plays, balls)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part{os.getpid()}"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic: a re-pull overwrites, never accumulates
    return game_id


def manifest_path(base: str, start: datetime, end: datetime, team: str | None) -> str:
    tag = f"{start:%Y%m%d}_{end:%Y%m%d}" + (f"_{team}" if team else "")
    return os.path.join(base, "_manifest", f"{tag}.json")


def discover_range(client: ThreadSafeClient, base: str, start: datetime,
                   end: datetime, team: str | None,
                   rediscover: bool = False) -> dict[str, dict]:
    """Discover the games in a range, caching the result next to the tree.

    Discovery has a far tighter quota than the data endpoints, and a run that
    dies partway through the fetch would otherwise have to spend that quota
    again just to learn what it already knew. The cache makes a resume cost
    zero discovery calls.
    """
    cache = manifest_path(base, start, end, team)
    if not rediscover and os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            sessions = json.load(fh)
        print(f"discovery manifest reused: {len(sessions)} games from {cache}",
              flush=True)
        return sessions

    sessions: dict[str, dict] = {}
    for i, (w_from, w_to) in enumerate(windows(start, end)):
        if i:
            time.sleep(_DISCOVERY_PACE)
        found = client.discover(w_from, w_to)
        kept = [s for s in found if wanted(s, team)]
        print(f"window {w_from[:10]} .. {w_to[:10]}: "
              f"{len(found)} sessions, {len(kept)} to pull", flush=True)
        for s in kept:
            sessions.setdefault(s["gameID"], s)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = f"{cache}.part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(sessions, fh)
    os.replace(tmp, cache)
    return sessions


def main() -> None:
    p = argparse.ArgumentParser(description="Pull TrackMan games into a game-date tree")
    p.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--out", required=True, help="Output base directory (local storage)")
    p.add_argument("--team", help="Only games involving this team shortName (e.g. DEL_BLU)")
    p.add_argument("--workers", type=int, default=8, help="Concurrent game fetches")
    p.add_argument("--requests-per-hour", type=float, default=4000.0,
                   help="Global request ceiling; two requests per game. 0 disables.")
    p.add_argument("--timeout-hours", type=float, default=12.0,
                   help="Overall deadline; the run fails loudly when it is hit")
    p.add_argument("--force", action="store_true",
                   help="Re-pull and overwrite games already on disk")
    p.add_argument("--rediscover", action="store_true",
                   help="Ignore the cached discovery manifest and re-run discovery")
    p.add_argument("--dry-run", action="store_true", help="Discover and count; write nothing")
    args = p.parse_args()

    start = datetime.fromisoformat(args.date_from).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.date_to).replace(tzinfo=timezone.utc)
    if start >= end:
        raise SystemExit("--from must be before --to")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    deadline = Deadline(args.timeout_hours)
    limiter = RateLimiter(args.requests_per_hour)
    client = ThreadSafeClient(load_config(), limiter)
    if limiter.interval:
        print(f"pacing: {args.requests_per_hour:.0f} requests/hour "
              f"(~{args.requests_per_hour / 2:.0f} games/hour)", flush=True)

    sessions = discover_range(client, args.out, start, end, args.team,
                              rediscover=args.rediscover)
    print(f"\ntotal games discovered: {len(sessions)}", flush=True)
    if args.dry_run:
        return

    todo = []
    skipped = 0
    for game_id, session in sorted(sessions.items()):
        if not args.force and os.path.exists(out_path(args.out, game_id)):
            skipped += 1
            continue
        todo.append(session)
    print(f"to fetch: {len(todo)} (already on disk: {skipped})", flush=True)

    done = failed = 0
    failures: list[str] = []
    timed_out = False
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(write_game, client, s, args.out): s["gameID"]
                   for s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            game_id = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as exc:
                failed += 1
                failures.append(game_id)
                print(f"  FAILED {game_id}: {type(exc).__name__}: {exc}", flush=True)
            if i % 50 == 0 or i == len(futures):
                rate = i / max(time.time() - t0, 1)
                print(f"  {i}/{len(futures)} (written {done}, failed {failed}, "
                      f"{rate:.2f} games/s, budget left {deadline.remaining_str()})",
                      flush=True)
            if deadline.expired and not timed_out:
                timed_out = True
                print(f"\nOVERALL TIMEOUT after {deadline.hours}h; cancelling "
                      "remaining work. Re-run the same command to resume.",
                      flush=True)
                for f in futures:
                    f.cancel()

    print(f"\nwritten {done}, skipped {skipped}, failed {failed}", flush=True)
    if failures:
        print("failed gameIDs: " + ", ".join(failures[:50])
              + (" ..." if len(failures) > 50 else ""))
    if timed_out:
        sys.exit(f"Run hit its {deadline.hours}h deadline before finishing.")
    if failed:
        sys.exit(f"{failed} game(s) failed; re-run the same command to retry them.")


if __name__ == "__main__":
    main()
