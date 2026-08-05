# webapp_publisher

Turns the current-season staff scoresheet into the frontend bundle contract
and publishes it to Blob storage on a schedule, so the web app always has a
fresh `manifest.json` + `staff_board.json` without anyone running the
notebook by hand.

## Data flow

1. `run_refresh.ps1` (scheduled task) invokes `python -m webapp_publisher.publish`.
2. `publish.py` shells out to `component_model/analysis/08_staff_scores.py`
   (the validated scorer — never reimplemented here) with `--data`,
   `--workdir`, `--team`, and waits for it to write `<workdir>/staff_scores.json`.
   A non-zero exit or a missing output file fails the job loudly
   (`subprocess.run(..., check=True)` + an explicit existence check).
3. `build_bundle.build_bundle()` (Task 5) transforms `staff_scores.json` into
   the two bundle files (`manifest.json`, `staff_board.json`), converting
   numpy types and NaN/inf to native JSON-safe values (`to_native`).
4. `schema.validate_bundle()` (Task 5) checks the bundle shape before
   anything is published — required manifest keys, non-empty pitcher list,
   valid `locFlag` values, required per-row keys.
5. `upload.upload_bundle()` serializes each file with
   `json.dumps(to_native(payload), allow_nan=False, ...)` (NaN becomes
   `null`, never the invalid JSON token `NaN`) and uploads it to the
   `WEBAPP_BUNDLE_CONTAINER` container, setting `content-type:
   application/json` and `cache-control: no-cache` so the web app never
   serves a stale cached copy.

The bundle also carries the pitcher development page files: `location_maps.json`
(count-conditioned run-value surface, shared), `model_artifacts.json` (ridge
coefficients, scaler, display moments, plain-English labels), and one
`pitchers/{pitcherId}.json` per pitcher. These come from
`component_model/analysis/14_pitcher_pages.py` and are keyed by TrackMan
PitcherId, which is stable across seasons.

## Running it

### Season and data-through labels

`season` and `dataThrough` in the published manifest are derived, not
free-typed, so the board's labels always match what the scorer actually
graded:

- **season** = the later year in `STUFFPLUS_YEARS` (default `2024,2025`,
  so season defaults to `2025`). `08_staff_scores.py`'s population is
  always the later year of that train/eval pair, relabeled internally to
  the "2025" role regardless of the literal year — see
  `component_model/analysis/fair_criterion.py`. Until 2026 is validated,
  keep `STUFFPLUS_YEARS=2024,2025` in `.env` so the board represents 2025.
- **dataThrough** = the latest game date present in `--data` whose year
  matches `season` (read from the `Date` column only). If the column is
  missing or no rows fall in that season, `publish.py` fails loudly rather
  than silently stamping today's date.

Both can still be overridden explicitly with `--season` / `--data-through`
for backfills or one-off runs.

### Dry run (no Azure credentials needed)

```
python -m webapp_publisher.publish --data <csv> --workdir <dir> --team DEL_BLU --dry-run
```

Writes `manifest.json` and `staff_board.json` under `<workdir>/bundle/`
instead of uploading, so you can inspect the output or feed it to the
frontend locally.

### Real run (uploads to Blob)

Requires `WEBAPP_STORAGE_CONNECTION_STRING` (and optionally
`WEBAPP_BUNDLE_CONTAINER`, default `bundles`) set in the environment, plus
`STUFFPLUS_DATA` / `STUFFPLUS_WORKDIR` (or pass `--data`/`--workdir`
explicitly):

```
python -m webapp_publisher.publish
```

### Env file

Copy `.env.example` to `webapp_publisher/.env` and fill in the real
connection string (and `STUFFPLUS_YEARS`, `STUFFPLUS_DATA`,
`STUFFPLUS_WORKDIR`, etc). `publish.py` auto-loads `webapp_publisher/.env`
at startup via `python-dotenv` (best-effort: if `python-dotenv` isn't
installed, or the file doesn't exist, it degrades gracefully and falls
back to whatever is already in the process environment — e.g. from the
scheduler). `.env` is git-ignored and blocked from being staged by a repo
deny rule — **Jack stages/creates `.env` himself**; this repo only ever
tracks `.env.example` with placeholders.

### Scheduled task

`run_refresh.ps1` wraps the publish CLI with bounded retries (default 4),
exponential backoff (5s, 10s, 20s, ...), and an overall timeout (default 30
minutes) so a transient failure doesn't retry forever and a hung run doesn't
block the next day's schedule. It fails loudly (non-zero exit, `Write-Error`)
if all retries are exhausted or the timeout is exceeded. It does NOT pass
`--season`/`--data-through` — `publish.py` derives both (see above); set
`STUFFPLUS_YEARS` in `webapp_publisher/.env` to control the season. Each
attempt's stdout/stderr is redirected to
`webapp_publisher/logs/refresh-<date>-attempt<N>.log` / `.err.log` so an
unattended scheduled run (including any failed retries) leaves a trace.

Register it once (run by Jack, not part of this repo's test suite):

```powershell
schtasks /Create /TN "PitchingAppRefresh" /TR "powershell -File C:\Users\jackdav\repos\baseball-stuff-plus\webapp_publisher\run_refresh.ps1" /SC DAILY /ST 23:30
```

## Tests

```
python -m pytest webapp_publisher/tests/ -v
```

`test_upload.py` injects a fake `upload_fn` so the test suite never touches
real Azure Blob storage or requires credentials.
