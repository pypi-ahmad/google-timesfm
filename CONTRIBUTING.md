# Contributing

This fork contains the current TimesFM-3 package, a React/FastAPI workbench, a retained Streamlit explorer,
TimesFM 2.5 compatibility code, archived versions, examples, and research
knowledge. Keep changes within the relevant generation and test boundary.

## Set up the repository

For native Windows workbench development, run `.\dev.ps1 setup` followed by
`.\dev.ps1 dev`. This preserves the installed CUDA Torch build. See the
[native guide](docs/how-to/native-workbench.md). The older explorer setup follows.

```powershell
git clone https://github.com/pypi-ahmad/google-timesfm.git
cd google-timesfm
uv sync --extra torch --extra app --group dev
```

Use the project-root `.venv` managed by `uv`. Do not commit credentials,
checkpoints, datasets, caches, or `.streamlit/secrets.toml`.

## Understand the layout

| Path | Responsibility |
|---|---|
| `src/timesfm3/` | Current TimesFM-3 PyTorch package |
| `src/timesfm_app/` | FastAPI, durable jobs, storage, services, native supervisor |
| `web/` | Next.js product UI and generated API types |
| `migrations/` | PostgreSQL schema migrations |
| `diagnostic_app.py` | Streamlit API-only diagnostic client |
| `streamlit_app.py` | Explorer widgets and session state |
| `src/timesfm3/explorer.py` | App validation, orchestration, and artifacts |
| `src/timesfm3/analysis.py` | Historical analysis and comparisons |
| `src/timesfm3/data_preparation.py` | Bulk grouping, calendar features, and data quality |
| `src/timesfm3/tracking.py` | Forecast assessment and refresh matching |
| `src/timesfm/` | TimesFM 2.5 implementation |
| `tests/` | Package and application tests |
| `timesfm-forecasting/` | Agent skill, compatibility scripts, examples |
| `docs/` | User and developer documentation |
| `knowledge/` | Governed draft research and references |

See the [codebase documentation](docs/codebase/STRUCTURE.md) before a broad
change.

## Make focused changes

- Follow the local two-space Python indentation and 88-character line length.
- Treat the React/FastAPI workbench as the primary product path. Keep Streamlit
  widgets in `streamlit_app.py` as a legacy diagnostic client.
- Keep tabular app policy in `timesfm3.explorer`; place preparation, analysis,
  tracking, model-resolution, and persistence behavior in their focused
  Explorer modules; keep tensor behavior in the forecaster/model layers.
- Preserve archived code unless the task explicitly targets it.
- Add a focused regression test for behavior changes.
- Keep TimesFM-3 and TimesFM 2.5 APIs clearly separated in code and docs.

## Run checks

Workbench checks (inference is injected in unit tests):

```powershell
uv run --no-sync ruff check src/timesfm_app diagnostic_app.py
uv run --no-sync ty check src/timesfm_app
uv run --no-sync pytest -q tests/test_app_api.py tests/test_app_store.py tests/test_app_jobs.py tests/test_app_migration.py tests/test_app_services.py tests/test_app_native.py tests/test_tracking_jobs.py tests/test_diagnostic_client.py
npm --prefix web test
npm --prefix web run build
```

Set `TIMESFM_TEST_DATABASE_URL` in the test process to enable real PostgreSQL
concurrency tests; these skip when no test database is provided. Native process
tests run only on Windows and spawn their own dummy processes. Browser tests use
Playwright; see `web/README.md` for fixture and live-server modes.

After changing API contracts, export and regenerate the TypeScript definitions:

```powershell
uv run --no-sync python -m timesfm_app.openapi .native/openapi.json
npm --prefix web run generate:api
npm --prefix web run typecheck
```

Application and current package tests:

```powershell
uv run pytest -q tests src/timesfm3
```

Focused explorer tests:

```powershell
uv run pytest -q tests/test_explorer.py tests/test_streamlit_app.py
```

Quality checks:

```powershell
uv run ruff check streamlit_app.py src/timesfm3 tests
uv run ruff format --check streamlit_app.py src/timesfm3 tests
uv run ty check streamlit_app.py src/timesfm3
uv build
```

Historical TimesFM 1 and 2 code is available from earlier Git revisions and
releases; it is not part of the current-package test command.

On the CUDA host used for the 2026-09-02 review, the full suite had two
documented TimesFM 2.5 device-placement failures. Do not automatically treat
those as TimesFM-3 explorer failures; consult [REVIEW.md](REVIEW.md) and
reproduce focused tests before changing behavior.

## Test the application

Start the app:

```powershell
.\launch_app.cmd
```

Verify that the demo renders without loading the model. For a real smoke test,
accept the weight restriction, run a short demo forecast, inspect the result,
and download its ZIP. Stop the process when finished.

## Update documentation

1. Update the task-oriented guide and relevant reference page together.
2. Keep internal links relative and use descriptive link text.
3. Mark code fences with a language.
4. Test commands and parse Python examples.
5. Update the coverage matrix in [docs/README.md](docs/README.md), the native
   workbench guide, and the workbench API reference when a workflow changes.
   Update legacy Explorer material only when that retained client changes.
6. Treat current source and tests as authoritative over draft OKF entries.

## Submit a change

Keep commits scoped and describe:

- the user-visible behavior or documentation outcome
- validation commands and results
- hardware-dependent checks not run
- model-version and license implications

Do not publish the package or push changes unless the repository owner has
authorized that external action.
