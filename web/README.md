# TimesFM analytical workspace

The Next.js application for the native Windows TimesFM service. Python owns data preparation, inference, evaluation, and numerical results; the browser owns configuration and presentation.

## Run locally

Use Node.js 24 or newer. From this directory:

```powershell
npm ci
npm run dev
```

Open `http://127.0.0.1:3000`. The Next.js rewrite forwards `/api/v1/*` to the native FastAPI service at `http://127.0.0.1:8001`. Follow the repository's native-service instructions to start PostgreSQL, Redis-compatible storage, the API, and workers. The app has no login; it is intended for the local workspace.

For a production build:

```powershell
npm run build
npm start
```

The UI uses locally bundled Geist fonts. It does not need a font CDN. Light, dark, and system themes are available in the header.

The worker device is configured through `TIMESFM_DEVICE=cuda` or `TIMESFM_DEVICE=cpu` when starting the native service. Job specifications preserve numerical settings; they do not select a different worker device per request.

## API contracts

Generated types in `src/lib/generated/api.d.ts` come from FastAPI's OpenAPI document. Jobs, submission specifications, record envelopes, and table pages use these types at the API boundary. Zod adds browser validation for editable forms.

After exporting the schema from the repository root with `uv run --no-sync python -m timesfm_app.openapi .native/openapi.json`, regenerate and check the frontend:

```powershell
npm run generate:api
npm run typecheck
```

TypeScript uses the latest 5.9 release supported by the OpenAPI generator's declared peer dependency. Runtime dependencies use current stable releases and the lockfile fixes the resolved versions.

## Workflows

- **Overview:** real dataset/run counts, active tracking, saved accuracy from the latest evaluation, persistent jobs, recoverable drafts, worker device and memory status.
- **Data:** CSV/Parquet uploads, version selection, bounded previews, source hashes, included demo data.
- **Forecasts:** column mapping, inference settings, grouping/calendar preparation, quality and interpolation previews, job submission, results and exports.
- **Experiments:** rolling backtests, joint/independent comparison, covariate ablation, named configuration comparison, baselines, and anomaly analysis.
- **Scenarios:** named editable future-covariate grids, with saved baseline results and scenario comparisons.
- **Tracking:** associate new dataset versions with issued forecasts, assess actuals, refresh forecasts, opt into automatic refresh, and stop tracking explicitly. Automatic refresh is off by default.
- **Models:** register Hub or local checkpoints and enqueue model checks.

Workspace settings live in the navigation footer and contain retention previews, explicit cleanup controls, and recorded activity.

## State and result guarantees

URLs carry workspace, dataset versions, draft, run, target, dataset, and variant selections across pages. React Hook Form owns editable inputs; TanStack Query owns server state. Drafts save to the server after a short delay, with per-tab recovery in session storage. Saves send the loaded revision; a stale tab must reload or save a separate draft. Background requests never replace edited form values. Changing dataset versions starts another draft and clears column roles.

Submission snapshots the validated specification and sends an idempotency key. The key survives a failed request so retrying can resolve the same job. Later draft changes do not mutate the run. Job polling continues across page visits; workers run independently of the browser.

Result tables request 100 rows per page with server sorting and virtualized visible rows. Charts use direct interval polygons, preserving negative values and breaking bands at missing/crossed bounds and missing forecast steps. They do not sort or repair returned quantiles. All quantiles remain available in result tables and exports. Reduced-resolution history and latest-origin selection are labeled when reported by the API.

The interval-calibration view plots persisted nominal and observed coverage for the selected dataset, target, variant, and horizon step. Scores and coverage counts are never recomputed in the browser.

## Verification

```powershell
npm run typecheck
npm test
npm run build
# With the UI running on port 3000 and Microsoft Edge installed:
npm run test:e2e
```

Browser tests intercept API calls to cover navigation, responsive layouts,
mobile control sizing, table overflow guidance, RTL layout, draft conflicts,
immutable submissions, server pagination, automatic-refresh preferences, and
calibration filters. To also inspect completed forecasts and backtests from the
real native worker:

```powershell
$env:TIMESFM_LIVE_TEST = '1'
npm run test:e2e
```

Live tests require a completed forecast and a backtest with calibration. They read saved results, create a draft from saved settings, and create then remove a temporary tracking record while checking associations and refresh preferences. They do not submit GPU work. Set `PLAYWRIGHT_BASE_URL` to test another locally running UI. Screenshots and failure traces are written to the ignored `test-results/` directory.
