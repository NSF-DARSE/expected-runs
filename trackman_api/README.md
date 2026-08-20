# TrackMan Data API client

Pulls UD Baseball pitch data from the TrackMan Data API instead of the Google
Drive CSV folder. Two slices so far:

1. **Smoke test** (`smoke_test.py`) — authenticates, discovers game sessions,
   pulls one session's plays + balls, and reports how the JSON lines up with
   the 60 columns the model pipeline expects.
2. **Backfill** (`backfill.py`) — discovers verified games over a date range
   (auto-chunked into 30-day windows) and writes one CSV per game into the
   `base/year/month/day/CSV/` layout the pipeline already reads, so a
   backfill is a drop-in replacement for the Drive folder. Resumable
   (re-running skips games already on disk), bounded retry with exponential
   backoff, token auto-renewal. Validated on a DEL_BLU week: all six games
   pass the golden diff after an on-disk CSV round trip.
3. **Flattener** (`flatten.py` + `golden_diff.py`) — joins plays to balls and
   flattens the JSON into the raw-CSV-shaped frame the existing pipeline
   consumes unchanged. Validated by golden diff: two 2025 games (home and
   away) flattened from the API, run through the pipeline's own
   runner-state/game-state/Target logic, and compared row-for-row against the
   same games in `Final_Target_Calc_*.csv` — every column exact through the
   Target stage (physics float dust < 1e-6 relative; one documented
   provenance diff on RunsRemaining, see `golden_diff.py` docstring).

Remaining before production cutover: full 2024-2025 backfill (~11.7k games,
a long overnight run), a parallel rebuild of the final dataset from the
backfilled tree, and the script 01 anchor check against the Drive-sourced
build. Design: `docs/superpowers/specs/2026-07-21-trackman-api-slice-design.md`.

## Setup

1. Install deps:
   ```
   pip install -r trackman_api/requirements.txt
   ```
2. Copy `.env.example` (repo root) to `.env` and fill in the two secrets from
   the TrackMan portal's "Data integration clients" page (create a dedicated
   client for this integration; never reuse or rotate the TruMedia client --
   generating a new secret invalidates the old one on first use):
   ```
   TRACKMAN_CLIENT_ID=
   TRACKMAN_CLIENT_SECRET=
   ```
   `.env` is gitignored. These are licensed (Level II) credentials; keep them
   out of git and out of shared logs.

   Auth note: the API uses the OAuth client_credentials grant at
   `https://login.trackman.com/connect/token` (per `swagger.json`). The
   password-grant flow in the Quick Start Guide v2.5 PDF is the older scheme
   and does not apply to portal-issued clients. There is no refresh token in
   this flow; renewal is just requesting a new token.

## Run the smoke test

```
python trackman_api/smoke_test.py                 # last 7 days
python trackman_api/smoke_test.py --days 14
python trackman_api/smoke_test.py --from 2024-02-01 --to 2024-02-15
python trackman_api/smoke_test.py --session-id <guid>
```

Output is small on purpose (session list, field names, a 2-row sample, and a
coverage checklist), so it is safe to paste back for review. Discovery windows
cannot exceed 30 days (an API limit).

## IP whitelist caveat

TrackMan can optionally restrict which IPs may call the Data API. If the smoke
test authenticates but discovery/data calls return an authorization/forbidden
error, UD's egress IP likely needs to be registered with your TrackMan rep.

## Run a backfill

```
python trackman_api/backfill.py --from 2025-05-10 --to 2025-05-18 --out <local dir> --team DEL_BLU
python trackman_api/backfill.py --from 2025-02-01 --to 2025-07-01 --out <local dir>   # all teams
python trackman_api/backfill.py --refresh --out <local dir>   # new games since latest on disk
python trackman_api/backfill.py ... --dry-run    # discovery + counts only
```

`--refresh` is the scheduled-incremental mode: it pulls from 7 days before
the latest game on disk (the lookback catches games verified late;
skip-existing makes the overlap free) through tomorrow.

Output must stay on local/UD-controlled storage (licensed Level II data).
Interrupted runs resume with the same command. Budget roughly 10 seconds per
game; a full season, all teams, is an overnight job. The discovery endpoint
has a much stricter rate quota than the data GETs (seemingly per-hour) --
many-window discovery sweeps in quick succession will 429 even after the
built-in minutes-scale backoff; wait and re-run.

## Run the golden diff

```
python trackman_api/golden_diff.py --data-dir "<folder with Final_Target_Calc_*.csv>"
python trackman_api/golden_diff.py --data-dir ... --game-id 20250510-MonmouthU-2
```

`--data-dir` is the Drive share folder holding the final dataset (ask Jack).
Output is aggregate-only (row counts, per-column verdicts, max deltas) — no
licensed pitch-level values are printed.

## Files

- `config.py` — loads/validates the two secrets and base URLs.
- `auth.py` — OAuth client_credentials grant.
- `smoke_test.py` — end-to-end proof: auth, discovery, one session's data.
- `flatten.py` — API JSON -> raw-CSV-shaped pitch frame (one game).
- `golden_diff.py` — acceptance test: API-flattened game vs the final dataset.
- `swagger.json` — the API's OpenAPI spec (source of truth for endpoints/auth).
