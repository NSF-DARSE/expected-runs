# UD Pitching Analytics Web App — Design

**Date:** 2026-07-24
**Status:** Approved design, pending implementation plan
**Replaces:** Power BI Stuff+ dashboard (retired v2 RF/SHAP model)

## Purpose

One web app that answers the pitching staff's questions from validated models — who's good, why, what's changing, and what should we do differently — and replaces Power BI as the single surface for pitching analytics. Primary user: pitching coach. Secondary: analytics staff. Pitcher-facing access is deferred but designed for.

Every displayed number comes from the validated component-model framework (Stuff+, Location+, adjusted results, arsenal Pitching+). Every score carries sample-size honesty (<50 FF unreliable, 50–99 caution) and a "data through {date}" stamp. No score is shown where it isn't validated (e.g. no Location+ on secondary pitches).

## Analytical core

The app is built entirely on the component model (`component_model/FRAMEWORK.md`, RESULTS.md):

- **Stuff+** — per-pitch-type ridge on physical features, all pitch types.
- **Location+** — pooled (x,z)→xT map score, fastballs only at the pitcher level; count-conditioned maps used for pitch-level explanation only.
- **Adjusted results** — opponent-adjusted, luck-stripped expected run value.
- **Pitching+ / arsenal grade** — equal-weight z-blend, mix-neutral per-type weighting, 100±15 display scale (higher = better at display layer only; internal run values are lower = better).

The old RF/SHAP model is not reproduced (score reliability at the noise floor; retired). Its *functionality* — per-pitch explanation, timelines, leaderboard — is rebuilt on the new scores. Per-pitch explanation uses exact additive ridge contributions (linear model) instead of SHAP.

## Pages

1. **Staff Board** (landing) — roster leaderboard: Pitching+, Stuff+, Location+, adjusted results, arsenal grade, trend sparkline, sample-size badges.
2. **Pitcher page** — arsenal cards per pitch type (Stuff+ all types; Location+ FF only; usage; whiff with swing counts and D1 type average; hard-hit). Score and feature timelines. Pitch explorer: strike-zone plot over the count-conditioned location map; clicking a pitch shows its grade decomposed into additive ridge contributions.
3. **Usage Gap Board** — actual-vs-expected usage disagreement queue (share model, script 11), with resolve buttons writing a labeled log (scored against the following season). Language rules from the replication review carry over: unbacked positive gaps read as over-trust questions; evidence-gated branches only.
4. **Portal Board** — buy-low list on the arsenal grade with matched-pairs and regression evidence.
5. **Opponent Scouting** — upcoming opponent's staff graded on the same scales (national feed supports it).
6. **Pitch Design** — sliders for velo/break/release; Stuff+ responds live, scored client-side from published ridge coefficients.
7. **Methodology / Dictionary** — plain-English score definitions, what each measures and does not, reliability context, sample floors.

## Phasing

- **Phase 1 — Replace Power BI:** platform (SWA, Entra auth, App Insights), Staff Board, Pitcher pages. Refresh via the local job pushing JSON bundles to Blob.
- **Phase 2 — Automation + decisions:** cloud game-evening refresh job (full hybrid), Usage Gap Board with Cosmos resolve log.
- **Phase 3 — Beyond the roster:** Portal Board, Opponent Scouting.
- **Phase 4 — Pitch Design.**

Each phase passes a `consumer-coach-baseball` agent review of the rendered app before being shown to the real staff.

## Architecture

Approach chosen: **Azure Static Web App + Blob-backed JSON + Cosmos for interactive writes.**

- **Frontend:** React + TypeScript + Vite on Azure Static Web Apps (org conventions).
- **Data serving:** all model outputs are precomputed JSON bundles in Azure Blob storage (no public access). A thin SWA API function streams the requested bundle file to the authenticated browser — data refreshes without app redeploys.
- **Interactive state:** Cosmos DB serverless, two containers: `resolutions` (usage-gap resolve log: case, verdict, note, user, timestamp) and `scenarios` (saved pitch-design what-ifs, Phase 4). Nothing model-computed lives in Cosmos.
- **New private repo** for the app (the modeling repo is a public upstream fork and must not carry UD app code, infra config, or licensed-data-derived JSON).

### Data flow

1. **Ingest** — two golden-diff-verified interchangeable sources writing the same `year/month/day/CSV` game tree: local FTP mirror (bulk/history) and `trackman_api/backfill.py --refresh` (incremental API pulls).
2. **Build + score** — existing pipeline (frozen 2024–25 ER map, inclusive RunsRemaining, PitchUID dedup) → `Final_Target_Calc` → component-model scoring scripts (per-type ridges, pooled location maps, adjusted results, arsenal Pitching+, usage-gap share model). A new **publisher script** converts scored outputs into a versioned JSON bundle and validates it against a schema before upload.
3. **Serve** — bundle uploaded to Blob; app fetches through the SWA API function.

### JSON bundle contract (per-page granularity)

- `manifest.json` — build timestamp, season, data-through date, bundle version (cache-buster and staleness source)
- `staff_board.json` — roster scores, flags, trend series
- `pitchers/{id}.json` — arsenal cards, timelines, pitch-level rows (grade + ridge contributions per pitch)
- `location_maps.json` — count-conditioned map grids for the strike-zone underlay
- `usage_gap.json` (Phase 2), `portal_board.json`, `opponents/{team}.json` (Phase 3)
- `model_artifacts.json` — ridge coefficients + type means for client-side scoring (Phase 4)

### Hybrid refresh

- **Local job (Windows scheduled task):** weekly-ish/preseason — FTP mirror sync, full-history rebuilds, model retraining with script 01 anchor checks, national bundles (portal, opponents). Uploads the JSON bundle **and the frozen scoring artifacts** (ridge coefficients, maps, ER table) to Blob.
- **Cloud job (Azure timer job):** game evenings, polls the TrackMan API ~4pm–midnight on game days (single nightly run in the offseason); pulls new verified games, runs the incremental build + scoring **using the frozen artifacts downloaded from Blob** (no retraining in the cloud), publishes an updated bundle the same evening.
- Both jobs use max-retry limits, exponential backoff, and orchestration timeouts (org standard), and write a heartbeat that the app surfaces as a stale-data warning.
- Compute vehicle for the cloud job (Container Apps job vs Function) is decided in Phase 2 based on incremental-build memory footprint. Default guess: Container Apps job (full Python pipeline).

## Frontend & UX

- Org data-viz standards: UD Blue primary, one yellow highlight per visual, gray context, direct labels, white backgrounds, minimal gridlines. Oswald headings, Open Sans body.
- **Answer-first layout:** each page leads top-left with the takeaway; filters shrink to a compact header defaulting to current season.
- **Consistent score language:** 100±15 everywhere, consistent color coding, sample-size flags as visible badges, unvalidated scores never rendered.
- **Explanation on demand:** ridge breakdowns, count maps, and methodology one click deep; conclusions by default.
- **Fast pitcher switching:** persistent roster sidebar, state preserved across navigation.
- **"Data through {date}"** on every page; stale-heartbeat warning.
- Desktop-first, iPad-readable; no phone-specific work in v1.

## Auth, security, monitoring

- **Auth:** Entra ID via SWA built-in auth; allowlist/group (Jack, pitching coach, analytics staff). Role model leaves room for a later `pitcher` role (own-data scoping) without restructuring.
- **Data classification:** all TrackMan-derived values are Level II — everything behind auth; Blob non-public, read via managed identity; TrackMan secrets only in the local `.env` and the cloud job's app settings; app repo private and free of licensed raw data.
- **Monitoring:** Application Insights `appi-ud-athletics-stuffplus` in the app's resource group, linked to the shared `log-ud-athletics` workspace; `setAuthenticatedUserContext` after sign-in (football-depth pattern). Custom events: per-pitcher page views, usage-gap resolutions, refresh success/failure. Alert on refresh failure.

## Error handling

- App treats the bundle as possibly stale or partially missing: a failed pitcher file renders an inline error card, never a blank page; manifest version mismatch triggers a soft reload.
- Refresh jobs fail loudly (App Insights alert) rather than silently serving old data.

## Testing

- Unit tests for data-shaping utilities and the client-side ridge scorer, which must reproduce Python scores exactly on a golden fixture (same golden-diff discipline as the API flattener).
- Playwright smoke tests per page against a fixture bundle.
- Publisher script schema-validates the bundle so pipeline changes can't silently break the app contract.

## Open questions / risks

- **Cloud job vehicle** (Container Apps vs Function) — Phase 2 decision.
- **Verified-data latency:** same-evening freshness depends on TrackMan verifying games promptly; if a game verifies late, the bundle updates on the next poll and the app's stamp stays honest.
- **Discovery-endpoint 429 quota** (~hourly) constrains polling cadence; the game-evening poller must reuse discovery results and back off per the org retry standard.
- **Pitcher role** deferred; revisit after Phase 2.
