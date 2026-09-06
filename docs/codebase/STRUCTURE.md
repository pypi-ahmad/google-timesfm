# Codebase Structure

## Top-Level Map

| Path | Purpose | Evidence |
|---|---|---|
| `src/timesfm3/` | Current TimesFM 3 model, Explorer workflows, and forecast interfaces | `src/timesfm3/__init__.py` |
| `src/timesfm_app/` | FastAPI contract, PostgreSQL store, artifacts, workers, and native supervisor | `src/timesfm_app/api.py` |
| `web/` | Next.js workbench, API client, forms, ECharts, and browser tests | `web/README.md` |
| `migrations/` | PostgreSQL schema migrations | `alembic.ini` |
| `src/timesfm/` | TimesFM 2.5 PyTorch/Flax implementation | `README.md` |
| `tests/` | Package and Streamlit tests | `tests/test_explorer.py` |
| `timesfm-forecasting/` | Agent skill, scripts, and examples | `timesfm-forecasting/SKILL.md` |
| `timesfm3-usage/` | Benchmark runners and results | `timesfm3-usage/benchmarks/README.md` |
| `knowledge/` | Governed TimesFM 3 research | `knowledge/index.md` |
| `.github/workflows/` | Build and guarded publish automation | `.github/workflows/main.yml` |

## Entry Points

- Workbench launcher: `launch_workbench.cmd`, then `dev.ps1 launch`.
- Native services: `python -m timesfm_app.native`; FastAPI application factory:
  `timesfm_app.api.create_app`.
- Browser UI: `web/src/app/[page]/page.tsx` served by Next.js on port 3000.

- Legacy Explorer runtime: `streamlit_app.py`, launched by `launch_app.cmd` or
  `streamlit run`.
- Python interfaces: `timesfm3.TimesFM3Forecaster` and
  `timesfm3.TimesFM3Evaluator`, exported by `src/timesfm3/__init__.py`.
- CSV helper: `timesfm-forecasting/scripts/forecast_csv.py`.

## Module Boundaries

| Boundary | Belongs here | Must not be here |
| Workbench UI | Forms, query state, charts, tables | Durable job ownership |
| `timesfm_app` | HTTP, records, artifacts, queues, native processes | Tensor math or browser rendering |
|---|---|---|
| Streamlit page | Widgets, session state, presentation | Model tensor implementation |
| `timesfm3.explorer` | Upload validation, preparation, artifacts | Widget rendering |
| `timesfm3.data_preparation` | Bulk groups, calendar covariates, data quality | Model execution |
| `timesfm3.analysis` | Historical comparisons and diagnostics | Widget rendering |
| `timesfm3.tracking` | Forecast-vintage assessment and refresh matching | Raw upload retention |
| `timesfm3.run_store` | DuckDB schema and derived result persistence | Raw upload retention |
| Forecaster/evaluator | Model loading and inference | CSV/Parquet UI policy |
| Model modules | Neural network operations/configuration | App session state |

## Naming and Organization Rules

- Python files and functions use `snake_case`; types use `PascalCase`.
- Current code is grouped by domain/package, with tests both in `tests/` and
  co-located as `*_test.py` under `src/timesfm3/`.
- Package-internal imports are relative; app/test imports use package names.

## Evidence

- `pyproject.toml`
- `streamlit_app.py`
- `src/timesfm3/__init__.py`
- `README.md`
