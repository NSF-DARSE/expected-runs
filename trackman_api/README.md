# TrackMan Data API client

Pulls UD Baseball pitch data from the TrackMan Data API instead of the Google
Drive CSV folder. Two slices so far:

1. **Smoke test** (`smoke_test.py`) — authenticates, discovers game sessions,
   pulls one session's plays + balls, and reports how the JSON lines up with
   the 60 columns the model pipeline expects.
2. **Flattener** (`flatten.py` + `golden_diff.py`) — joins plays to balls and
   flattens the JSON into the raw-CSV-shaped frame the existing pipeline
   consumes unchanged. Validated by golden diff: two 2025 games (home and
   away) flattened from the API, run through the pipeline's own
   runner-state/game-state/Target logic, and compared row-for-row against the
   same games in `Final_Target_Calc_*.csv` — every column exact through the
   Target stage (physics float dust < 1e-6 relative; one documented
   provenance diff on RunsRemaining, see `golden_diff.py` docstring).

Remaining before production cutover: season backfill via 30-day discovery
windows, output-format decision, a parallel run against the Drive source with
the script 01 anchor check, and retry/backoff + token-renewal hardening.
Design: `docs/superpowers/specs/2026-07-21-trackman-api-slice-design.md`.

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
