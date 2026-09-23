---
type: Reference
title: Test and validation map
description: Focused Python, workbench, frontend, API-contract, and browser checks used by contributors and CI.
tags: [testing, ci, pytest, vitest, playwright]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-ee3ea3bd39689f7e4f5dc7c6
    resource: repo://.github/workflows/main.yml
  - id: openwiki-source-6c5226111ba453e635259c7c
    resource: repo://.github/workflows/workbench.yml
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-14e56945b7c632a3b335dcec
    resource: repo://web/package.json
  - id: openwiki-source-9953c8a4b5a75bef445cf794
    resource: repo://web/playwright.config.ts
  - id: openwiki-source-52cc4c3bc39b1652ba919753
    resource: repo://web/README.md
  - id: openwiki-source-3d94fba251831ad22978cb80
    resource: repo://web/vitest.config.ts
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Test and validation map

Choose the test boundary that matches the change. Contributor commands live in
[`CONTRIBUTING.md`](../CONTRIBUTING.md#run-checks); CI splits the current Python
package and the React/FastAPI workbench into separate workflows.

## Focused Python checks

| Change area | Focused tests |
|---|---|
| Upload parsing and forecast preparation | `tests/test_explorer.py` |
| Grouping, calendar features, and data quality | `tests/test_data_preparation.py` |
| Historical analysis and scenarios | `tests/test_analysis.py`, `tests/test_analysis_extensions.py` |
| Model checkpoint resolution | `tests/test_checkpoint_selection.py` |
| Workbench endpoints and service boundaries | `tests/test_app_api.py`, `tests/test_app_services.py` |
| Durable jobs and worker lifecycle | `tests/test_app_jobs.py`, `tests/test_app_services.py` |
| Records, artifact safety, and migrations | `tests/test_app_store.py`, `tests/test_app_migration.py` |

For the workbench backend, the repository's focused checks include Ruff, `ty`,
and the app/service/native/migration suites:

```powershell
uv run --no-sync ruff check src/timesfm_app diagnostic_app.py
uv run --no-sync ty check src/timesfm_app
uv run --no-sync pytest -q tests/test_app_api.py tests/test_app_store.py tests/test_app_jobs.py tests/test_app_migration.py tests/test_app_services.py tests/test_app_native.py tests/test_tracking_jobs.py tests/test_diagnostic_client.py
```

For the broader current package, run `uv run pytest -q tests src/timesfm3`.
`CONTRIBUTING.md` also gives package lint, format, type, and build checks. The
historical TimesFM 2.5 implementation is not included in that current-package
test command.

## Frontend and API contract

From the repository root, run frontend unit tests, typecheck, and production
build with:

```powershell
npm --prefix web test
npm --prefix web run typecheck
npm --prefix web run build
```

Frontend unit tests use Vitest and cover `web/tests/unit/**/*.test.ts`. When an
API schema changes, export OpenAPI, regenerate the TypeScript declarations, and
typecheck the result:

```powershell
uv run --no-sync python -m timesfm_app.openapi .native/openapi.json
npm --prefix web run generate:api
npm --prefix web run typecheck
```

CI also checks that generated declarations match the exported schema
([workbench workflow](../.github/workflows/workbench.yml#L18)).

## Browser and live checks

Playwright browser tests run through `npm --prefix web run test:e2e`. They use
Microsoft Edge and a UI at `http://127.0.0.1:3000` by default; set
`PLAYWRIGHT_BASE_URL` to target another local UI. The standard browser suite
stubs API responses to check workspace navigation and editing behavior. For a
live data-backed pass, enable `TIMESFM_LIVE_TEST=1`; those tests require saved
forecast and backtest results and do not submit GPU work
([browser test setup](../web/README.md#verification),
[`playwright.config.ts`](../web/playwright.config.ts#L1)).

Workbench API/service tests inject predictors rather than requiring live model
inference. PostgreSQL concurrency checks are optional and run when
`TIMESFM_TEST_DATABASE_URL` is set. Native process-ownership tests are Windows
only and spawn dummy processes ([contributor guidance](../CONTRIBUTING.md#run-checks),
[workbench CI](../.github/workflows/workbench.yml#L10)).

For validation commands grouped by the full CI matrix, see
[`main.yml`](../.github/workflows/main.yml) and
[`workbench.yml`](../.github/workflows/workbench.yml).
