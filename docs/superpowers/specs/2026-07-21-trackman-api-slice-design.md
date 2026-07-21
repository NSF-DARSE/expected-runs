# TrackMan Data API vertical slice, design

Date: 2026-07-21
Status: approved for implementation

## Purpose

Prove, end to end, that we can pull UD Baseball pitch data from the TrackMan
Data API instead of the current Google Drive CSV folder. This slice
authenticates, discovers game sessions, and pulls one session's data, then
reports how the real JSON compares to what the model pipeline needs. It is a
de-risking step, not the production ingestion path.

## Background

The current model pipeline (`python_files/target_and_calculated_pipeline.py`)
reads a Google Drive folder tree (`base_path/year/month/day/CSV/*.csv`), one row
per pitch in TrackMan's flat V3 CSV schema (60 columns, `REQUIRED_COLS`).

The API is not a drop-in for that CSV:

- Auth is OAuth 2.0 password grant; access tokens live one hour and are renewed
  with a refresh token (`https://login.trackmanbaseball.com/connect/token`).
- Pitch data is split across two endpoints per session:
  - `POST/GET .../data/game/plays/<sessionId>` — pitcher, batter, count,
    tagging, result (the "who / what").
  - `GET .../data/game/balls/<sessionId>` — physical measurements (`release`,
    `movement`, `location`).
- Sessions are found via `POST .../discovery/game/sessions` with a UTC date
  window that cannot span more than 30 days.
- An optional IP whitelist may block requests until UD's egress IP is
  registered with TrackMan.

Data classification: licensed TrackMan data is Level II. Credentials live in
`.env` (already gitignored). Pulls stay scoped; no bulk dataset leaves the
local machine or enters the conversation.

## Scope

In scope:

1. `.env` / `.env.example` for the four secrets (client id, client secret,
   username, password).
2. OAuth client: obtain token (password grant) + renew (refresh token).
3. Smoke test: authenticate, discover recent game sessions, pull one session's
   plays + balls, print shapes / field names / a small sample, and report which
   of the 60 model columns can be populated from the JSON.

Explicitly out of scope (deferred until we have seen the real data):

- The plays to balls join and flatten-to-60-columns mapping.
- Writing the folder-tree CSVs / replacing the Google Drive source.
- Token auto-refresh scheduling and production retry/backoff hardening.
- Practice sessions, media, and video endpoints.

## Components

Everything lives under `trackman_api/`. Nothing imports from or modifies the
existing model pipeline.

### 1. `trackman_api/config.py`

Loads the four secrets from `.env` (via `python-dotenv`) plus the base URLs.
Exposes a small config object. Fails loudly with a clear message naming any
missing variable.

### 2. `trackman_api/auth.py`

- `get_token(config)` — POST password grant, returns access token, refresh
  token, and an absolute expiry time.
- `refresh_token(config, refresh_token)` — POST refresh grant, same return
  shape.

Uses `requests`. No retry loop in the slice: a failed call raises with the
response body so auth / whitelist errors are visible.

### 3. `trackman_api/smoke_test.py`

Runnable script. Flow:

1. Load config, get a token.
2. `POST discovery/game/sessions` for a date window (default: last 7 days;
   overridable via CLI args, clamped to the 30-day API limit).
3. Print the session list: local date, home/away, sessionId.
4. Pick one session (first, or a `--session-id` override).
5. GET its `plays` and `balls`.
6. Print, for each: record count, top-level field names, and a 2-row sample.
7. Report coverage: for each of the 60 `REQUIRED_COLS`, whether a source field
   was found in the plays/balls JSON.

Output is intentionally small and safe to paste back: shapes, field names,
2-row samples, and a coverage checklist. No full session dump.

### 4. `trackman_api/requirements.txt` and `trackman_api/README.md`

`requests` and `python-dotenv`. README documents the four secrets, how to fill
`.env`, how to run the smoke test, and the IP whitelist caveat.

## Error handling

- Missing env vars: config raises a clear message before any network call.
- Auth failure (bad creds): surfaced with the API's `error_description`.
- Whitelist / network block: the failing request raises with status + body so
  the cause is legible; remedy is to register UD's egress IP with the TrackMan
  rep.
- No sessions in the window: reported plainly, script exits without error.

## Verification

Jack fills the four secrets into `.env`. Then either Jack runs
`python trackman_api/smoke_test.py` and pastes the (small) output, or it is run
locally. Success = a token is obtained, at least one session is listed, and the
plays/balls samples plus coverage report print. The coverage report is the real
deliverable: it tells us how far the full flattener will have to go.

## Follow-on work (next slice, not now)

Given a passing smoke test: design the plays to balls join, the field mapping to
`REQUIRED_COLS`, the output format (CSV folder tree vs parquet vs DB), and
production auth refresh + retry hardening.
