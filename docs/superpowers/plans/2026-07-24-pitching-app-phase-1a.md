# Pitching App Phase 1a — Platform, Auth, Blob Serving, Staff Board, Local Refresh — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a deployed, Entra-authenticated Static Web App that shows the DEL_BLU pitching staff leaderboard (Pitching+/Stuff+/Location+/AdjRes with sample-size flags and per-pitcher score drivers), fed by a local Python publisher that scores the current season and pushes a JSON bundle to Azure Blob — refreshing without redeploys.

**Architecture:** Two repos. (1) The **frontend** lives in a new private repo `ud-athletics-baseball-pitching` (React 19 + TS + Vite + Tailwind on Azure Static Web Apps); its built-in `/api` function proxies JSON bundle files out of a private Blob container so the browser stays same-origin behind SWA auth. (2) The **publisher** lives in the existing `baseball-stuff-plus` repo (co-located with the 6.9 GB TrackMan data tree and the validated scoring scripts) under `webapp_publisher/`; it runs `component_model/analysis/08_staff_scores.py`, transforms the output into the bundle contract, and uploads to Blob. The bundle (JSON in Blob) is the only contract between the two repos.

**Tech Stack:** React 19, TypeScript (strict), Vite 7, Tailwind CSS, React Router v7, TanStack Query, Recharts, Lucide React, `@microsoft/applicationinsights-web`; Azure Static Web Apps (Standard SKU) built-in Node 20 functions with `@azure/storage-blob`; Python 3.11 publisher with `azure-storage-blob`, `pytest`.

## Global Constraints

- **Data classification: Level II.** Every displayed value derives from licensed TrackMan data. The app is behind Entra auth end-to-end; the Blob container has NO public access; no licensed-data-derived JSON is ever committed to git (frontend repo included). This is a deliberate divergence from the org `refresh-data.yml` pattern (which commits JSON to `public/data/` and redeploys) — do NOT "correct" it back to committing data.
- **Secrets:** TrackMan and Storage credentials live only in local `.env` (publisher) and SWA Application Settings (frontend). `.env` is gitignored before first commit. `.env.example` documents required vars with placeholders. A deny rule blocks staging anything `.env`-shaped — Jack stages `.env.example` himself.
- **Sign convention (verify, don't assume):** internal run values (Target/xT/ridge_pred/location-map `v`) are expected runs from the pitcher's perspective, LOWER = better. The `100±15` display scores in `staff_scores.json` (`Stuff100`, `Loc100`, `AdjRes100`, `Pitch100`) are ALREADY negated to higher-is-better. The frontend treats higher = better; it must not re-flip.
- **Score names on the wire (from `08_staff_scores.py`), all 100±15:** `stuff` (Stuff+), `loc` (Location+, fastball-only), `adjres` (adjusted results), `pitch` (Pitching+ combined). Rate fields: `whiff` (higher better, 0–1), `zone`, `heart`, `mean_height`. `loc_flag` ∈ `"small sample"` (<50 FF) / `"caution"` (50–99) / `""` (100+ FF).
- **TypeScript strictness:** `tsconfig.json` sets `noUnusedLocals`/`noUnusedParameters`; these are ERRORS in `vite build` (CI). Prefix intentionally-unused callback params with `_`. Clean all unused imports before commit.
- **Vite output dir is `dist`** (not `build`) — set `output_location: "dist"` in the deploy workflow or SWA deploys empty.
- **Brand/data-viz:** UD Blue `#00539f` primary; UD Gold `#ffd200` at most ONE highlight per visual; Cool Gray `#bdbdbd` for context; white backgrounds; minimize gridlines; direct labels over legends; key insight top-left. Vanguard/Oswald display, Greycliff/Open Sans body. Color is never the only differentiator (pair with text).
- **Retry/polling safeguards (org standard):** any retry/poll loop needs a max-retry limit that fails loudly, exponential backoff (never fixed sleep), and an orchestration-level timeout.

---

### Task 1: Bootstrap the frontend scaffold

**Files:**
- Create: `ud-athletics-baseball-pitching/package.json`, `vite.config.ts`, `tsconfig.json`, `tsconfig.node.json`, `tailwind.config.ts`, `postcss.config.js`, `index.html`, `.gitignore`, `.env.example`, `src/main.tsx`, `src/App.tsx`, `src/styles/brand.css`, `src/vite-env.d.ts`
- Test: `src/App.test.tsx`, `vitest.config.ts`

**Interfaces:**
- Produces: `App` (default export React component) mounting a React Router `<BrowserRouter>` and a TanStack `QueryClientProvider`; a global `QueryClient` instance exported from `src/lib/queryClient.ts`.

- [ ] **Step 1: Create the new repo directory and scaffold with Vite**

Run in `C:\Users\jackdav\repos`:
```bash
npm create vite@latest ud-athletics-baseball-pitching -- --template react-ts
cd ud-athletics-baseball-pitching
npm install
npm install react-router-dom @tanstack/react-query recharts lucide-react @microsoft/applicationinsights-web
npm install -D tailwindcss@^3 postcss autoprefixer vitest @testing-library/react @testing-library/jest-dom jsdom @vitest/coverage-v8
npx tailwindcss init -p
git init
```

- [ ] **Step 2: Write the failing test**

`src/App.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from './App';

describe('App', () => {
  it('renders the app shell heading', () => {
    render(<App />);
    expect(screen.getByRole('banner')).toHaveTextContent(/UD Pitching/i);
  });
});
```

`vitest.config.ts`:
```ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: { environment: 'jsdom', globals: true, setupFiles: './src/test-setup.ts' },
});
```

`src/test-setup.ts`:
```ts
import '@testing-library/jest-dom';
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `npx vitest run src/App.test.tsx`
Expected: FAIL — `App` renders default Vite content, no `banner` role with "UD Pitching".

- [ ] **Step 4: Write the minimal app shell**

`src/lib/queryClient.ts`:
```ts
import { QueryClient } from '@tanstack/react-query';
export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5 * 60_000, refetchOnWindowFocus: false } },
});
```

`src/App.tsx`:
```tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { queryClient } from './lib/queryClient';

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <header role="banner" className="bg-ud-blue text-white font-display px-6 py-3 text-lg">
          UD Pitching
        </header>
        <main className="p-6">
          <Routes>
            <Route path="/" element={<div>Staff Board coming soon</div>} />
          </Routes>
        </main>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
```

`src/main.tsx`:
```tsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/brand.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode><App /></React.StrictMode>
);
```

- [ ] **Step 5: Write `brand.css` and tailwind config**

`src/styles/brand.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --ud-blue: #00539f; --ud-gold: #ffd200; --ud-navy: #003c71;
  --ud-light-blue: #00a0df; --ud-gray: #bdbdbd; --ud-cream: #eee8c5;
  --ud-success: #16a34a; --ud-warning: #f59e0b; --ud-error: #dc2626;
  --font-display: 'Oswald', sans-serif; --font-body: 'Open Sans', sans-serif;
}
body { font-family: var(--font-body); background: #fff; color: #111; }
```
(Brand `.otf` fonts are added in Task 8 packaging; Oswald/Open Sans fallbacks are acceptable until then.)

`tailwind.config.ts`:
```ts
import type { Config } from 'tailwindcss';
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: { ud: { blue: '#00539f', gold: '#ffd200', navy: '#003c71', 'light-blue': '#00a0df', gray: '#bdbdbd', cream: '#eee8c5' } },
      fontFamily: { display: ['Oswald', 'sans-serif'], body: ['Open Sans', 'sans-serif'] },
    },
  },
  plugins: [],
} satisfies Config;
```

Ensure `tsconfig.json` has `"noUnusedLocals": true, "noUnusedParameters": true` (Vite template default) and `"types": ["vitest/globals", "@testing-library/jest-dom"]`.

- [ ] **Step 6: Run test to verify it passes, then build**

Run: `npx vitest run src/App.test.tsx` → Expected: PASS
Run: `npm run build` → Expected: exits 0, produces `dist/`.

- [ ] **Step 7: Write `.gitignore` and `.env.example`, then commit**

`.gitignore` (append): `.env`, `*.key`, `*.pem`, `secrets/`, `node_modules/`, `dist/`
`.env.example`:
```
# Frontend build-time (Vite) — set in SWA App Settings in production
VITE_APPINSIGHTS_CONNECTION_STRING=
```
```bash
git add -A && git commit -m "Scaffold React/TS/Vite app shell with brand tokens and routing"
```

---

### Task 2: Auth config, App Insights, and layout header with data-freshness slot

**Files:**
- Create: `ud-athletics-baseball-pitching/staticwebapp.config.json`, `src/services/appInsights.ts`, `src/components/layout/Header.tsx`
- Modify: `src/App.tsx` (use `Header`)
- Test: `src/services/appInsights.test.ts`, `src/components/layout/Header.test.tsx`

**Interfaces:**
- Produces: `appInsights` (nullable `ApplicationInsights`), `setAuthenticatedUser(email: string): void` from `src/services/appInsights.ts`; `Header({ dataThrough }: { dataThrough?: string })` rendering the banner + a "Data through {date}" stamp (or "Data pending" when undefined).

- [ ] **Step 1: Write the failing tests**

`src/services/appInsights.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { setAuthenticatedUser } from './appInsights';

describe('appInsights', () => {
  it('does not throw when no connection string is configured', () => {
    expect(() => setAuthenticatedUser('coach@udel.edu')).not.toThrow();
  });
});
```

`src/components/layout/Header.test.tsx`:
```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import Header from './Header';

describe('Header', () => {
  it('shows the data-through date when provided', () => {
    render(<Header dataThrough="2026-03-15" />);
    expect(screen.getByText(/Data through 2026-03-15/)).toBeInTheDocument();
  });
  it('shows pending when no date', () => {
    render(<Header />);
    expect(screen.getByText(/Data pending/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/services/appInsights.test.ts src/components/layout/Header.test.tsx`
Expected: FAIL — modules not found.

- [ ] **Step 3: Implement App Insights service (mirrors `athletic-aid/src/services/appInsights.ts`)**

`src/services/appInsights.ts`:
```ts
import { ApplicationInsights } from '@microsoft/applicationinsights-web';

const connectionString = import.meta.env.VITE_APPINSIGHTS_CONNECTION_STRING as string | undefined;

export const appInsights = connectionString
  ? new ApplicationInsights({ config: { connectionString } })
  : null;

if (appInsights) appInsights.loadAppInsights();

export function setAuthenticatedUser(email: string): void {
  appInsights?.setAuthenticatedUserContext(email.replace(/[^a-zA-Z0-9_-]/g, '_'), email, true);
}
```

- [ ] **Step 4: Implement Header**

`src/components/layout/Header.tsx`:
```tsx
export default function Header({ dataThrough }: { dataThrough?: string }) {
  return (
    <header role="banner" className="bg-ud-blue text-white px-6 py-3 flex items-baseline justify-between">
      <span className="font-display text-lg">UD Pitching</span>
      <span className="text-sm text-ud-cream">
        {dataThrough ? `Data through ${dataThrough}` : 'Data pending'}
      </span>
    </header>
  );
}
```
Wire it into `App.tsx` in place of the inline `<header>` (pass `dataThrough` later from the manifest; for now `<Header />`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/services/appInsights.test.ts src/components/layout/Header.test.tsx` → Expected: PASS

- [ ] **Step 6: Write `staticwebapp.config.json` (Entra auth, /api gated, non-MS providers 404'd)**

`staticwebapp.config.json` (UD tenant `a698667d-8817-4ad9-a7f2-bb287f867e5f`):
```json
{
  "routes": [
    { "route": "/.auth/login/github", "statusCode": 404 },
    { "route": "/.auth/login/twitter", "statusCode": 404 },
    { "route": "/api/*", "allowedRoles": ["authenticated"] },
    { "route": "/*", "allowedRoles": ["authenticated"] }
  ],
  "responseOverrides": { "401": { "statusCode": 302, "redirect": "/.auth/login/aad" } },
  "auth": {
    "identityProviders": {
      "azureActiveDirectory": {
        "registration": {
          "openIdIssuer": "https://login.microsoftonline.com/a698667d-8817-4ad9-a7f2-bb287f867e5f/v2.0",
          "clientIdSettingName": "AAD_CLIENT_ID",
          "clientSecretSettingName": "AAD_CLIENT_SECRET"
        }
      }
    }
  },
  "navigationFallback": { "rewrite": "/index.html", "exclude": ["/assets/*", "/api/*"] },
  "platform": { "apiRuntime": "node:20" }
}
```

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "Add Entra auth config, App Insights service, and freshness-aware header"
```

---

### Task 3: Bundle types and the Staff Board data hook

**Files:**
- Create: `src/lib/types.ts`, `src/hooks/useStaffBoard.ts`, `src/test/fixtures/staff_board.json`, `src/test/fixtures/manifest.json`
- Test: `src/hooks/useStaffBoard.test.tsx`

**Interfaces:**
- Consumes: bundle files served at `/api/bundle/manifest.json` and `/api/bundle/staff_board.json` (Task 4 serves them).
- Produces:
  - `types.ts`: `Manifest { built: string; season: number; dataThrough: string; bundleVersion: string }`; `PitcherRow { id: number; name: string; hand: string; ff: number; stuff: number; loc: number; adjres: number; pitch: number; whiff: number | null; zone: number; heart: number; meanHeight: number; locFlag: '' | 'caution' | 'small sample'; stuffAttr: [string, number][] }`; `StaffBoard { population: number; team: string; pitchers: PitcherRow[] }`.
  - `useStaffBoard(): UseQueryResult<{ manifest: Manifest; board: StaffBoard }>`.

- [ ] **Step 1: Write the failing test**

`src/hooks/useStaffBoard.test.tsx`:
```tsx
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useStaffBoard } from './useStaffBoard';
import board from '../test/fixtures/staff_board.json';
import manifest from '../test/fixtures/manifest.json';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn((url: string) =>
    Promise.resolve({ ok: true, json: () => Promise.resolve(url.includes('manifest') ? manifest : board) } as Response)
  ));
});

function wrap() {
  const qc = new QueryClient();
  return ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={qc}>{children}</QueryClientProvider>;
}

describe('useStaffBoard', () => {
  it('loads and maps the first pitcher row', async () => {
    const { result } = renderHook(() => useStaffBoard(), { wrapper: wrap() });
    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data!.manifest.dataThrough).toBe('2026-03-15');
    expect(result.current.data!.board.pitchers[0].name).toBe('Moyzan, Ben');
    expect(result.current.data!.board.pitchers[0].pitch).toBe(134);
    expect(result.current.data!.board.pitchers[0].locFlag).toBe('');
  });
});
```

- [ ] **Step 2: Create fixtures**

`src/test/fixtures/manifest.json`:
```json
{ "built": "2026-07-24T02:00:00Z", "season": 2026, "dataThrough": "2026-03-15", "bundleVersion": "2026-07-24T02:00:00Z" }
```
`src/test/fixtures/staff_board.json`:
```json
{ "population": 543, "team": "DEL_BLU",
  "pitchers": [
    { "id": 101, "name": "Moyzan, Ben", "hand": "R", "ff": 210, "stuff": 120, "loc": 128, "adjres": 112, "pitch": 134,
      "whiff": 0.23, "zone": 0.52, "heart": 0.24, "meanHeight": 3.1, "locFlag": "",
      "stuffAttr": [["effectivevelo", 34], ["inducedvertbreak", 10], ["horzbreak", 6], ["relheight", -1], ["relside", -3]] },
    { "id": 102, "name": "Marose, Alex", "hand": "L", "ff": 39, "stuff": 108, "loc": 99, "adjres": 101, "pitch": 104,
      "whiff": 0.31, "zone": 0.49, "heart": 0.22, "meanHeight": 3.0, "locFlag": "small sample",
      "stuffAttr": [["effectivevelo", 12], ["horzbreak", 5], ["inducedvertbreak", 3], ["relheight", -2], ["relside", -4]] }
  ] }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `npx vitest run src/hooks/useStaffBoard.test.tsx` → Expected: FAIL — `useStaffBoard` not found.

- [ ] **Step 4: Implement types and hook**

`src/lib/types.ts`:
```ts
export interface Manifest { built: string; season: number; dataThrough: string; bundleVersion: string }
export type LocFlag = '' | 'caution' | 'small sample';
export interface PitcherRow {
  id: number; name: string; hand: string; ff: number;
  stuff: number; loc: number; adjres: number; pitch: number;
  whiff: number | null; zone: number; heart: number; meanHeight: number;
  locFlag: LocFlag; stuffAttr: [string, number][];
}
export interface StaffBoard { population: number; team: string; pitchers: PitcherRow[] }
```

`src/hooks/useStaffBoard.ts`:
```ts
import { useQuery } from '@tanstack/react-query';
import type { Manifest, StaffBoard } from '../lib/types';

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`/api/bundle/${path}`);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

export function useStaffBoard() {
  return useQuery({
    queryKey: ['staffBoard'],
    queryFn: async () => {
      const [manifest, board] = await Promise.all([
        getJson<Manifest>('manifest.json'),
        getJson<StaffBoard>('staff_board.json'),
      ]);
      return { manifest, board };
    },
  });
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run src/hooks/useStaffBoard.test.tsx` → Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Define bundle types and Staff Board data hook with fixtures"
```

---

### Task 4: SWA API function — Blob bundle proxy

**Files:**
- Create: `api/package.json`, `api/host.json`, `api/src/functions/bundle.js`, `api/src/blob.js`
- Test: `api/src/functions/bundle.test.js`
- Modify: `.env.example` (document `STORAGE_CONNECTION_STRING`, `BUNDLE_CONTAINER`)

**Interfaces:**
- Consumes: env `STORAGE_CONNECTION_STRING`, `BUNDLE_CONTAINER` (default `bundles`).
- Produces: HTTP route `GET /api/bundle/{*path}` returning the blob body as `application/json` with `Cache-Control: no-cache`. `fetchBlobText(path, deps)` in `api/src/blob.js` — `deps` injects a `BlobServiceClient` factory for testing. Path must match `^[a-zA-Z0-9._/-]+$` and contain no `..` segment, else 400.

- [ ] **Step 1: Set up the api package**

`api/package.json`:
```json
{
  "name": "pitching-api",
  "version": "1.0.0",
  "type": "module",
  "main": "src/functions/*.js",
  "dependencies": { "@azure/functions": "^4.5.0", "@azure/storage-blob": "^12.24.0" },
  "devDependencies": { "vitest": "^2.0.0" },
  "scripts": { "test": "vitest run" }
}
```
`api/host.json`:
```json
{ "version": "2.0", "extensionBundle": { "id": "Microsoft.Azure.Functions.ExtensionBundle", "version": "[4.*, 5.0.0)" } }
```
Run: `cd api && npm install`

- [ ] **Step 2: Write the failing test**

`api/src/functions/bundle.test.js`:
```js
import { describe, it, expect, vi } from 'vitest';
import { fetchBlobText, isSafePath } from '../blob.js';

describe('isSafePath', () => {
  it('accepts nested bundle paths', () => expect(isSafePath('pitchers/101.json')).toBe(true));
  it('rejects traversal', () => expect(isSafePath('../secrets/.env')).toBe(false));
  it('rejects backslashes', () => expect(isSafePath('a\\b')).toBe(false));
});

describe('fetchBlobText', () => {
  it('downloads the requested blob', async () => {
    const download = vi.fn().mockResolvedValue('{"ok":true}');
    const deps = { getContainerClient: () => ({ getBlobClient: (p) => ({ _p: p, async download() { return { readableStreamBody: null }; } }) }), readAll: download };
    const text = await fetchBlobText('manifest.json', deps);
    expect(text).toBe('{"ok":true}');
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd api && npx vitest run` → Expected: FAIL — `../blob.js` not found.

- [ ] **Step 4: Implement blob helper and function**

`api/src/blob.js`:
```js
import { BlobServiceClient } from '@azure/storage-blob';

export function isSafePath(p) {
  if (!p || p.includes('\\') || p.split('/').includes('..')) return false;
  return /^[a-zA-Z0-9._/-]+$/.test(p);
}

async function streamToText(stream) {
  const chunks = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf-8');
}

function defaultDeps() {
  const svc = BlobServiceClient.fromConnectionString(process.env.STORAGE_CONNECTION_STRING);
  const container = svc.getContainerClient(process.env.BUNDLE_CONTAINER || 'bundles');
  return {
    getContainerClient: () => container,
    readAll: async (blobClient) => streamToText((await blobClient.download()).readableStreamBody),
  };
}

export async function fetchBlobText(path, deps = defaultDeps()) {
  const blobClient = deps.getContainerClient().getBlobClient(path);
  return deps.readAll(blobClient);
}
```

`api/src/functions/bundle.js`:
```js
import { app } from '@azure/functions';
import { fetchBlobText, isSafePath } from '../blob.js';

app.http('bundle', {
  methods: ['GET'],
  authLevel: 'anonymous',
  route: 'bundle/{*path}',
  handler: async (request, context) => {
    const path = request.params.path;
    if (!isSafePath(path)) return { status: 400, jsonBody: { error: 'invalid path' } };
    try {
      const body = await fetchBlobText(path);
      return { status: 200, headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' }, body };
    } catch (err) {
      context.error(`bundle fetch failed for ${path}: ${err.message}`);
      return { status: 404, jsonBody: { error: 'not found' } };
    }
  },
});
```
(Adjust the test's `fetchBlobText` deps mock to return the injected text; the assertion above drives the `readAll` injection point — make `readAll` return the mock value.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd api && npx vitest run` → Expected: PASS

- [ ] **Step 6: Document env and commit**

Append to root `.env.example`:
```
# SWA API function (set in SWA App Settings in production; local: api/local.settings.json)
STORAGE_CONNECTION_STRING=
BUNDLE_CONTAINER=bundles
```
```bash
git add -A && git commit -m "Add Blob bundle proxy API function with path allowlisting"
```

---

### Task 5: Publisher — transform `staff_scores.json` into the bundle

**Files:**
- Create (in `baseball-stuff-plus`): `webapp_publisher/__init__.py`, `webapp_publisher/build_bundle.py`, `webapp_publisher/schema.py`, `webapp_publisher/requirements.txt`
- Test: `webapp_publisher/tests/test_build_bundle.py`, `webapp_publisher/tests/fixtures/staff_scores.json`

**Interfaces:**
- Consumes: a `staff_scores.json` dict as written by `component_model/analysis/08_staff_scores.py` — top level `dict(population, team, staff, grids)`; each `staff` entry has keys `name, hand, ff, adjres, stuff, loc, pitch, whiff, zone, heart, mean_height, loc_flag, stuff_attr`.
- Produces: `build_bundle(staff_scores: dict, *, season: int, data_through: str, built_iso: str) -> dict[str, dict]` returning `{ "manifest.json": {...}, "staff_board.json": {...} }`. `assign_ids(staff)` deterministically maps each pitcher `name` to a stable int id (sorted-name index). Serialization is numpy-safe via `to_native(obj)`.

- [ ] **Step 1: Write the failing test**

`webapp_publisher/tests/test_build_bundle.py`:
```python
import json, pathlib
from webapp_publisher.build_bundle import build_bundle

FIX = pathlib.Path(__file__).parent / "fixtures" / "staff_scores.json"

def test_build_bundle_shapes_manifest_and_board():
    staff_scores = json.loads(FIX.read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="2026-07-24T02:00:00Z")

    manifest = bundle["manifest.json"]
    assert manifest == {"built": "2026-07-24T02:00:00Z", "season": 2026,
                        "dataThrough": "2026-03-15", "bundleVersion": "2026-07-24T02:00:00Z"}

    board = bundle["staff_board.json"]
    assert board["team"] == "DEL_BLU"
    assert board["population"] == 543
    row = board["pitchers"][0]
    assert set(row) == {"id","name","hand","ff","stuff","loc","adjres","pitch",
                        "whiff","zone","heart","meanHeight","locFlag","stuffAttr"}
    assert isinstance(row["id"], int)
    assert row["locFlag"] in ("", "caution", "small sample")
    assert isinstance(row["stuffAttr"], list) and isinstance(row["stuffAttr"][0], list)

def test_ids_are_stable_across_calls():
    staff_scores = json.loads(FIX.read_text())
    b1 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    b2 = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    ids1 = {r["name"]: r["id"] for r in b1["staff_board.json"]["pitchers"]}
    ids2 = {r["name"]: r["id"] for r in b2["staff_board.json"]["pitchers"]}
    assert ids1 == ids2
```

`webapp_publisher/tests/fixtures/staff_scores.json`:
```json
{ "population": 543, "team": "DEL_BLU",
  "staff": [
    { "name": "Moyzan, Ben", "hand": "R", "ff": 210, "adjres": 112.0, "stuff": 120.0, "loc": 128.0, "pitch": 134.0,
      "whiff": 0.23, "zone": 0.52, "heart": 0.24, "mean_height": 3.1, "loc_flag": "",
      "stuff_attr": [["effectivevelo", 34.0], ["inducedvertbreak", 10.0], ["horzbreak", 6.0], ["relheight", -1.0], ["relside", -3.0]] },
    { "name": "Marose, Alex", "hand": "L", "ff": 39, "adjres": 101.0, "stuff": 108.0, "loc": 99.0, "pitch": 104.0,
      "whiff": 0.31, "zone": 0.49, "heart": 0.22, "mean_height": 3.0, "loc_flag": "small sample",
      "stuff_attr": [["effectivevelo", 12.0], ["horzbreak", 5.0], ["inducedvertbreak", 3.0], ["relheight", -2.0], ["relside", -4.0]] }
  ],
  "grids": { "pooled": [] } }
```

- [ ] **Step 2: Run test to verify it fails**

Run (in `baseball-stuff-plus`): `python -m pytest webapp_publisher/tests/test_build_bundle.py -v`
Expected: FAIL — module `webapp_publisher.build_bundle` not found.

- [ ] **Step 3: Implement the builder**

`webapp_publisher/build_bundle.py`:
```python
"""Transform the validated staff_scores.json into the frontend bundle contract.

Input is the dict written by component_model/analysis/08_staff_scores.py.
Scores are already on the 100±15 display scale (higher = better); do not re-flip.
"""
from __future__ import annotations
import math
from typing import Any

try:
    import numpy as np
except ImportError:  # numpy optional for pure-dict inputs
    np = None


def to_native(obj: Any) -> Any:
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [to_native(x) for x in obj.tolist()]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(obj, np.bool_):
            return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def assign_ids(staff: list[dict]) -> dict[str, int]:
    """Stable id per pitcher name: index into the sorted unique-name list, +1."""
    names = sorted({s["name"] for s in staff})
    return {name: i + 1 for i, name in enumerate(names)}


def _row(s: dict, pid: int) -> dict:
    return to_native({
        "id": pid,
        "name": s["name"],
        "hand": s["hand"],
        "ff": s["ff"],
        "stuff": s["stuff"],
        "loc": s["loc"],
        "adjres": s["adjres"],
        "pitch": s["pitch"],
        "whiff": s.get("whiff"),
        "zone": s["zone"],
        "heart": s["heart"],
        "meanHeight": s["mean_height"],
        "locFlag": s["loc_flag"],
        "stuffAttr": [[f, v] for f, v in s["stuff_attr"]],
    })


def build_bundle(staff_scores: dict, *, season: int, data_through: str, built_iso: str) -> dict[str, dict]:
    ids = assign_ids(staff_scores["staff"])
    pitchers = [_row(s, ids[s["name"]]) for s in staff_scores["staff"]]
    pitchers.sort(key=lambda r: r["pitch"], reverse=True)
    return {
        "manifest.json": {
            "built": built_iso, "season": season,
            "dataThrough": data_through, "bundleVersion": built_iso,
        },
        "staff_board.json": {
            "population": to_native(staff_scores["population"]),
            "team": staff_scores["team"],
            "pitchers": pitchers,
        },
    }
```

`webapp_publisher/schema.py`:
```python
"""Lightweight bundle validation — fail loudly before upload."""
REQUIRED_ROW_KEYS = {"id","name","hand","ff","stuff","loc","adjres","pitch",
                     "whiff","zone","heart","meanHeight","locFlag","stuffAttr"}

def validate_bundle(bundle: dict) -> None:
    m = bundle["manifest.json"]
    for k in ("built","season","dataThrough","bundleVersion"):
        if k not in m:
            raise ValueError(f"manifest missing {k}")
    board = bundle["staff_board.json"]
    if not board["pitchers"]:
        raise ValueError("staff_board has no pitchers")
    for r in board["pitchers"]:
        missing = REQUIRED_ROW_KEYS - set(r)
        if missing:
            raise ValueError(f"pitcher row {r.get('name')} missing {missing}")
        if r["locFlag"] not in ("", "caution", "small sample"):
            raise ValueError(f"bad locFlag {r['locFlag']}")
```

`webapp_publisher/requirements.txt`:
```
azure-storage-blob>=12.24
pytest>=8
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest webapp_publisher/tests/test_build_bundle.py -v` → Expected: PASS (2 tests)

- [ ] **Step 5: Add a schema-validation test and commit**

Append to the test file:
```python
from webapp_publisher.schema import validate_bundle
import pytest, json, pathlib

def test_validate_bundle_rejects_bad_flag():
    staff_scores = json.loads((pathlib.Path(__file__).parent/"fixtures"/"staff_scores.json").read_text())
    bundle = build_bundle(staff_scores, season=2026, data_through="2026-03-15", built_iso="x")
    bundle["staff_board.json"]["pitchers"][0]["locFlag"] = "nope"
    with pytest.raises(ValueError):
        validate_bundle(bundle)
```
Run: `python -m pytest webapp_publisher/tests/ -v` → Expected: PASS
```bash
git add webapp_publisher/ && git commit -m "Add web app publisher: staff_scores to bundle transform with validation"
```

---

### Task 6: Publisher — score, upload, and schedule the local refresh

**Files:**
- Create (in `baseball-stuff-plus`): `webapp_publisher/publish.py`, `webapp_publisher/upload.py`, `webapp_publisher/run_refresh.ps1`, `webapp_publisher/.env.example`, `webapp_publisher/README.md`
- Test: `webapp_publisher/tests/test_upload.py`

**Interfaces:**
- Consumes: `build_bundle`, `validate_bundle` (Task 5); env `WEBAPP_STORAGE_CONNECTION_STRING`, `WEBAPP_BUNDLE_CONTAINER`, `STUFFPLUS_DATA`, `STUFFPLUS_WORKDIR`.
- Produces: `upload_bundle(bundle, *, connection_string, container, upload_fn=None) -> list[str]` (returns uploaded blob names; `upload_fn(name, text)` injectable for tests); `publish.py` CLI: `--data`, `--workdir`, `--team DEL_BLU`, `--season`, `--data-through`, `--dry-run` (write bundle to `--workdir/bundle/` instead of uploading).

- [ ] **Step 1: Write the failing test**

`webapp_publisher/tests/test_upload.py`:
```python
from webapp_publisher.upload import upload_bundle

def test_upload_bundle_serializes_each_file_and_reports_names():
    calls = {}
    def fake_upload(name, text):
        calls[name] = text
    bundle = {"manifest.json": {"built": "x"}, "staff_board.json": {"team": "DEL_BLU", "pitchers": [{"a": float('nan')}]}}
    names = upload_bundle(bundle, connection_string="ignored", container="bundles", upload_fn=fake_upload)
    assert set(names) == {"manifest.json", "staff_board.json"}
    # NaN must serialize to null (JSON-valid)
    assert "NaN" not in calls["staff_board.json"]
    assert "null" in calls["staff_board.json"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest webapp_publisher/tests/test_upload.py -v` → Expected: FAIL — module not found.

- [ ] **Step 3: Implement upload**

`webapp_publisher/upload.py`:
```python
import json
from webapp_publisher.build_bundle import to_native


def _default_upload_fn(connection_string, container):
    from azure.storage.blob import BlobServiceClient, ContentSettings
    svc = BlobServiceClient.from_connection_string(connection_string)
    cont = svc.get_container_client(container)
    def upload(name, text):
        cont.upload_blob(
            name=name, data=text.encode("utf-8"), overwrite=True,
            content_settings=ContentSettings(content_type="application/json", cache_control="no-cache"),
        )
    return upload


def upload_bundle(bundle, *, connection_string, container, upload_fn=None):
    upload_fn = upload_fn or _default_upload_fn(connection_string, container)
    uploaded = []
    for name, payload in bundle.items():
        text = json.dumps(to_native(payload), allow_nan=False, separators=(",", ":"))
        upload_fn(name, text)
        uploaded.append(name)
    return uploaded
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest webapp_publisher/tests/test_upload.py -v` → Expected: PASS

- [ ] **Step 5: Implement the publish CLI (orchestrates scoring, build, upload)**

`webapp_publisher/publish.py`:
```python
"""Local refresh job: score current season -> build bundle -> upload to Blob.

Runs the validated scorer (08_staff_scores.py) as a subprocess so its logic is
never restructured, then transforms + uploads. Fails loudly on any step.
"""
import argparse, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
from webapp_publisher.build_bundle import build_bundle
from webapp_publisher.schema import validate_bundle
from webapp_publisher.upload import upload_bundle

REPO = pathlib.Path(__file__).resolve().parents[1]
SCORER = REPO / "component_model" / "analysis" / "08_staff_scores.py"


def run_scorer(data: str, workdir: str, team: str) -> dict:
    workdir_p = pathlib.Path(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCORER), "--data", data, "--workdir", workdir, "--team", team]
    subprocess.run(cmd, check=True)  # raises CalledProcessError -> loud failure
    out = workdir_p / "staff_scores.json"
    if not out.exists():
        raise FileNotFoundError(f"scorer did not produce {out}")
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("STUFFPLUS_DATA"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--team", default="DEL_BLU")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-through", required=True, help="YYYY-MM-DD latest game date in the data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.data or not args.workdir:
        ap.error("--data and --workdir (or STUFFPLUS_DATA/STUFFPLUS_WORKDIR) required")

    staff_scores = run_scorer(args.data, args.workdir, args.team)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle = build_bundle(staff_scores, season=args.season, data_through=args.data_through, built_iso=built)
    validate_bundle(bundle)

    if args.dry_run:
        out = pathlib.Path(args.workdir) / "bundle"
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in bundle.items():
            (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(f"[dry-run] wrote {len(bundle)} files to {out}")
        return 0

    conn = os.environ["WEBAPP_STORAGE_CONNECTION_STRING"]
    container = os.environ.get("WEBAPP_BUNDLE_CONTAINER", "bundles")
    names = upload_bundle(bundle, connection_string=conn, container=container)
    print(f"uploaded {len(names)} files to {container}: {names} (data through {args.data_through})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Write the scheduled-task wrapper with retry/backoff/timeout (org standard)**

`webapp_publisher/run_refresh.ps1`:
```powershell
# Local refresh with bounded retries, exponential backoff, and an overall timeout.
param(
  [int]$MaxRetries = 4,
  [int]$TimeoutMinutes = 30,
  [string]$Season = "2026",
  [string]$DataThrough = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
if (-not $DataThrough) { $DataThrough = (Get-Date).ToString("yyyy-MM-dd") }
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$delay = 5
for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
  if ((Get-Date) -gt $deadline) { Write-Error "Refresh exceeded ${TimeoutMinutes}m timeout"; exit 1 }
  try {
    python -m webapp_publisher.publish --season $Season --data-through $DataThrough
    Write-Host "Refresh succeeded on attempt $attempt"; exit 0
  } catch {
    Write-Warning "Attempt $attempt failed: $_"
    if ($attempt -eq $MaxRetries) { Write-Error "Refresh failed after $MaxRetries attempts"; exit 1 }
    Start-Sleep -Seconds $delay; $delay = $delay * 2
  }
}
```
Register (documented in README, run once by Jack):
```powershell
schtasks /Create /TN "PitchingAppRefresh" /TR "powershell -File C:\Users\jackdav\repos\baseball-stuff-plus\webapp_publisher\run_refresh.ps1" /SC DAILY /ST 23:30
```

- [ ] **Step 7: Write env example + README, run full publisher tests, commit**

`webapp_publisher/.env.example`:
```
WEBAPP_STORAGE_CONNECTION_STRING=
WEBAPP_BUNDLE_CONTAINER=bundles
STUFFPLUS_DATA=C:\Users\jackdav\stuffplus_replication\source_2025_2026.csv
STUFFPLUS_WORKDIR=C:\Users\jackdav\stuffplus_replication\workdir_webapp
```
`webapp_publisher/README.md`: document the data flow, `--dry-run` usage, and that Jack stages `.env.example` (deny rule blocks `.env`-shaped files).
Run: `python -m pytest webapp_publisher/tests/ -v` → Expected: PASS (all)
```bash
git add webapp_publisher/ && git commit -m "Add local refresh job: score, build, upload bundle with bounded retries"
```

---

### Task 7: Staff Board page

**Files:**
- Create: `src/pages/StaffBoard.tsx`, `src/components/ui/ScoreCell.tsx`, `src/components/ui/SampleBadge.tsx`, `src/lib/scoreColor.ts`
- Modify: `src/App.tsx` (route `/` → `StaffBoard`, pass `manifest.dataThrough` to `Header`)
- Test: `src/lib/scoreColor.test.ts`, `src/pages/StaffBoard.test.tsx`

**Interfaces:**
- Consumes: `useStaffBoard` (Task 3), `PitcherRow` (Task 3).
- Produces: `scoreColor(score: number): string` (Tailwind text class by 100±15 band); `SampleBadge({ flag }: { flag: LocFlag })`; `ScoreCell({ value }: { value: number })`; `StaffBoard` page — a sortable leaderboard (default sort `pitch` desc) with an expandable per-pitcher driver row (from `stuffAttr`).

- [ ] **Step 1: Write the failing tests**

`src/lib/scoreColor.test.ts`:
```ts
import { describe, it, expect } from 'vitest';
import { scoreColor } from './scoreColor';

describe('scoreColor', () => {
  it('greens strong scores (>=115)', () => expect(scoreColor(120)).toContain('green'));
  it('blues above-average (105-114)', () => expect(scoreColor(108)).toContain('ud-blue'));
  it('grays average (95-104)', () => expect(scoreColor(100)).toContain('gray'));
  it('reds weak (<95)', () => expect(scoreColor(88)).toContain('red'));
});
```

`src/pages/StaffBoard.test.tsx`:
```tsx
import { render, screen, within } from '@testing-library/react';
import { QueryClientProvider, QueryClient } from '@tanstack/react-query';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import StaffBoard from './StaffBoard';
import board from '../test/fixtures/staff_board.json';
import manifest from '../test/fixtures/manifest.json';

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn((url: string) =>
    Promise.resolve({ ok: true, json: () => Promise.resolve(url.includes('manifest') ? manifest : board) } as Response)));
});
const wrap = (ui: React.ReactNode) => <QueryClientProvider client={new QueryClient()}>{ui}</QueryClientProvider>;

describe('StaffBoard', () => {
  it('renders pitchers sorted by Pitching+ desc with a sample badge on small samples', async () => {
    render(wrap(<StaffBoard />));
    const rows = await screen.findAllByRole('row');
    // header + 2 data rows
    const first = within(rows[1]);
    expect(first.getByText('Moyzan, Ben')).toBeInTheDocument();
    expect(screen.getByText(/small sample/i)).toBeInTheDocument(); // Marose flagged
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/lib/scoreColor.test.ts src/pages/StaffBoard.test.tsx` → Expected: FAIL — modules not found.

- [ ] **Step 3: Implement scoreColor and small UI components**

`src/lib/scoreColor.ts`:
```ts
// 100±15 display scale, higher = better. Pair color with the number (never color-only).
export function scoreColor(score: number): string {
  if (score >= 115) return 'text-ud-success';        // green
  if (score >= 105) return 'text-ud-blue';
  if (score >= 95) return 'text-ud-gray';
  return 'text-ud-error';                             // red
}
```
Ensure `tailwind.config.ts` `colors.ud` includes `success: '#16a34a'` and `error: '#dc2626'` (add them).

`src/components/ui/ScoreCell.tsx`:
```tsx
import { scoreColor } from '../../lib/scoreColor';
export default function ScoreCell({ value }: { value: number }) {
  return <span className={`font-semibold tabular-nums ${scoreColor(value)}`}>{Math.round(value)}</span>;
}
```

`src/components/ui/SampleBadge.tsx`:
```tsx
import type { LocFlag } from '../../lib/types';
export default function SampleBadge({ flag }: { flag: LocFlag }) {
  if (!flag) return null;
  const label = flag === 'small sample' ? 'small sample' : 'caution';
  return (
    <span className="ml-2 rounded px-1.5 py-0.5 text-xs bg-ud-cream text-ud-navy align-middle" title="Fastball sample below reliable threshold">
      {label}
    </span>
  );
}
```

- [ ] **Step 4: Implement the Staff Board page**

`src/pages/StaffBoard.tsx`:
```tsx
import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';
import { useStaffBoard } from '../hooks/useStaffBoard';
import type { PitcherRow } from '../lib/types';
import ScoreCell from '../components/ui/ScoreCell';
import SampleBadge from '../components/ui/SampleBadge';

type SortKey = 'pitch' | 'stuff' | 'loc' | 'adjres';

export default function StaffBoard() {
  const { data, isLoading, isError } = useStaffBoard();
  const [sortKey, setSortKey] = useState<SortKey>('pitch');
  const [openId, setOpenId] = useState<number | null>(null);

  if (isLoading) return <p className="text-ud-gray">Loading staff board…</p>;
  if (isError || !data) return <p className="text-ud-error">Something went wrong loading the staff board. Try refreshing.</p>;

  const rows: PitcherRow[] = [...data.board.pitchers].sort((a, b) => b[sortKey] - a[sortKey]);
  const cols: { key: SortKey; label: string }[] = [
    { key: 'pitch', label: 'Pitching+' }, { key: 'stuff', label: 'Stuff+' },
    { key: 'loc', label: 'Location+' }, { key: 'adjres', label: 'Adj Results' },
  ];

  return (
    <section>
      <h1 className="font-display text-2xl text-ud-navy mb-1">Staff Leaderboard</h1>
      <p className="text-sm text-ud-gray mb-4">{data.board.team} · graded vs {data.board.population} qualified D1 fastballs</p>
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left border-b border-ud-gray">
            <th className="py-2 pr-4">Pitcher</th>
            {cols.map((c) => (
              <th key={c.key} className="py-2 px-3 cursor-pointer select-none" onClick={() => setSortKey(c.key)}
                  aria-sort={sortKey === c.key ? 'descending' : 'none'}>
                {c.label}{sortKey === c.key ? ' ▾' : ''}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <FragmentRow key={r.id} row={r} open={openId === r.id}
              onToggle={() => setOpenId(openId === r.id ? null : r.id)} colCount={cols.length} />
          ))}
        </tbody>
      </table>
    </section>
  );
}

function FragmentRow({ row, open, onToggle, colCount }:
  { row: PitcherRow; open: boolean; onToggle: () => void; colCount: number }) {
  return (
    <>
      <tr className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer" onClick={onToggle}>
        <td className="py-2 pr-4">
          {open ? <ChevronDown className="inline w-4 h-4" /> : <ChevronRight className="inline w-4 h-4" />}
          <span className="ml-1">{row.name}</span>
          <SampleBadge flag={row.locFlag} />
        </td>
        <td className="py-2 px-3"><ScoreCell value={row.pitch} /></td>
        <td className="py-2 px-3"><ScoreCell value={row.stuff} /></td>
        <td className="py-2 px-3"><ScoreCell value={row.loc} /></td>
        <td className="py-2 px-3"><ScoreCell value={row.adjres} /></td>
      </tr>
      {open && (
        <tr className="bg-gray-50">
          <td colSpan={colCount + 1} className="py-2 px-6">
            <span className="text-xs uppercase text-ud-gray mr-3">Stuff+ drivers ({row.hand}HP, {row.ff} FF)</span>
            {row.stuffAttr.map(([feat, pts]) => (
              <span key={feat} className="inline-block mr-4 tabular-nums">
                {feat} <b className={pts >= 0 ? 'text-ud-success' : 'text-ud-error'}>{pts >= 0 ? '+' : ''}{Math.round(pts)}</b>
              </span>
            ))}
          </td>
        </tr>
      )}
    </>
  );
}
```
Wire `App.tsx`: route `/` → `<StaffBoard />`, and read the manifest for the header. Simplest: have `App` render a small wrapper that calls `useStaffBoard` for the date — or pass a static `<Header />` in 1a and defer live date wiring. To keep the freshness stamp live, render `<Header dataThrough={data?.manifest.dataThrough} />` from within a layout that also uses `useStaffBoard` (acceptable — TanStack dedupes the query).

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx vitest run src/lib/scoreColor.test.ts src/pages/StaffBoard.test.tsx` → Expected: PASS
Run: `npm run build` → Expected: exits 0 (watch for `noUnusedLocals` errors — remove any unused imports).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "Add Staff Board leaderboard with score color bands, sample badges, and driver drill-in"
```

---

### Task 8: Deploy workflow, packaging, and go-live checklist

**Files:**
- Create: `.github/workflows/azure-static-web-apps.yml`, `public/favicon.ico` (UD Blue Hen), `public/fonts/` (brand `.otf` weights), `README.md`, `docs/DEPLOY.md`
- Modify: `src/styles/brand.css` (register `@font-face` for shipped fonts), `.env.example`
- Test: manual deploy verification (checklist)

**Interfaces:**
- Consumes: the SWA deployment token GitHub secret `AZURE_STATIC_WEB_APPS_API_TOKEN` (auto-created when Jack links the repo).

- [ ] **Step 1: Write the deploy workflow**

`.github/workflows/azure-static-web-apps.yml`:
```yaml
name: Azure Static Web Apps CI/CD
on:
  push:
    branches: [main, development]
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches: [main]
jobs:
  build_and_deploy:
    if: github.event_name != 'pull_request' || github.event.action != 'closed'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { submodules: true }
      - name: Build and Deploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: upload
          app_location: "/"
          api_location: "api"
          output_location: "dist"
          app_build_command: "npm run build"
  close_pr:
    if: github.event_name == 'pull_request' && github.event.action == 'closed'
    runs-on: ubuntu-latest
    steps:
      - name: Close PR preview
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          action: close
```

- [ ] **Step 2: Register brand fonts (copy weights from standards repo)**

Copy `Oswald`/`Greycliff` weights actually used into `public/fonts/`; add `@font-face` blocks to `brand.css` for `Vanguard CF` and `Greycliff CF` per the react-ts-conventions standard, and update `--font-display`/`--font-body` to prefer them. Keep the weight set minimal.

- [ ] **Step 3: Write `docs/DEPLOY.md` — the Azure provisioning + go-live checklist**

Document (Jack runs these once; they need portal/CLI actions, not code):
```
Azure resources (subscription athl-analytics 6b5acc2f-..., tenant a698667d-...):
- Resource group: rg-ud-athletics-baseball-pitching
- Storage account (private): create; container `bundles` with NO public access. Copy connection string.
- Static Web App (Standard SKU): create, link the GitHub repo (auto-adds AZURE_STATIC_WEB_APPS_API_TOKEN).
- App registration (Entra, single-tenant UD): redirect URI https://<swa>.azurestaticapps.net/.auth/login/aad/callback;
  ENABLE ID tokens (Implicit grant) — skipping this causes a silent sign-in loop; create client secret.
- SWA App Settings: AAD_CLIENT_ID, AAD_CLIENT_SECRET, STORAGE_CONNECTION_STRING, BUNDLE_CONTAINER=bundles,
  VITE_APPINSIGHTS_CONNECTION_STRING.
- App Insights: appi-ud-athletics-baseball-pitching in the app RG, linked to log-ud-athletics workspace.

Go-live checklist:
- [ ] .env not committed; no secrets in git history
- [ ] Publisher dry-run produces a valid bundle (python -m webapp_publisher.publish --dry-run ...)
- [ ] Publisher live run uploads to `bundles`; verify blobs exist
- [ ] SWA deploy green on development; preview URL loads behind login (incognito, @udel.edu only)
- [ ] /api/bundle/manifest.json returns JSON when authed; unauthenticated → redirect to login
- [ ] Header shows correct "Data through" date
- [ ] consumer-coach-baseball agent review of the deployed Staff Board passes before showing the real coach
- [ ] Merge development → main (Jack approves) for production
```

- [ ] **Step 4: Write README and finalize `.env.example`**

`README.md`: what the app is, the two-repo split, local dev (`npm run dev`, and `swa start` with the api), where the data comes from (publisher in baseball-stuff-plus). Note Level II classification and the no-committed-data rule.

- [ ] **Step 5: Verify the full build and test suite, then commit**

Run: `npx vitest run` → Expected: PASS (all)
Run: `cd api && npx vitest run` → Expected: PASS
Run: `npm run build` → Expected: exits 0
```bash
git add -A && git commit -m "Add SWA deploy workflow, brand fonts, and go-live checklist"
```

- [ ] **Step 6: Coach-review gate**

After the first successful deploy to the preview environment, dispatch the `consumer-coach-baseball` agent against the deployed Staff Board URL. Address blocking feedback before showing the real pitching coach. (This gate repeats for every phase.)

---

## Self-Review Notes

- **Spec coverage (Phase 1a slice):** platform/scaffold (T1), Entra auth + App Insights (T2), Blob-backed serving without redeploys (T4), local publisher/refresh with retry safeguards (T5–T6), Staff Board replacing Roster Leaderboard + Pitcher Summary drivers (T7), deploy + monitoring + coach-review gate (T8). Pitcher pages, timelines, pitch-explorer map, Usage Gap Board, Portal Board, Opponent Scouting, and Pitch Design are explicitly OUT of 1a (later plans).
- **Divergence from org standard is intentional and documented:** data served from Blob via `/api` proxy, never committed to `public/data/` (Level II licensed data + refresh-without-redeploy).
- **Type consistency:** wire-format score keys (`stuff/loc/adjres/pitch`, `whiff/zone/heart/meanHeight`, `locFlag`, `stuffAttr`) are identical across the publisher output (T5), the fixtures (T3), the TS types (T3), and the page (T7). Python side uses `mean_height`/`loc_flag`/`stuff_attr` (08's keys) and maps to camelCase in `_row`.
- **Known follow-ups for 1b:** the publisher must add `pitchers/{id}.json` (timelines + pitch-level ridge rows + per-type arsenal) and `location_maps.json`; `08_staff_scores.py` already emits `grids` (pooled + count12) which 1b's pitch-explorer will consume.
