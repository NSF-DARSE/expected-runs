# TrackMan Data API client (vertical slice)

De-risking slice that proves we can pull UD Baseball pitch data from the
TrackMan Data API instead of the Google Drive CSV folder. It authenticates,
discovers game sessions, pulls one session's plays + balls, and reports how the
JSON lines up with the 60 columns the model pipeline expects.

This is not the production ingestion path. The plays-to-balls join, the
flatten-to-60-columns mapping, and the Google Drive replacement come in a later
slice, once we have seen the real data shape. Design:
`docs/superpowers/specs/2026-07-21-trackman-api-slice-design.md`.

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

## Files

- `config.py` — loads/validates the two secrets and base URLs.
- `auth.py` — OAuth client_credentials grant.
- `smoke_test.py` — the end-to-end proof.
- `swagger.json` — the API's OpenAPI spec (source of truth for endpoints/auth).
