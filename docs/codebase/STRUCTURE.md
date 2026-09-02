# Codebase Structure

## Top-Level Map

| Path | Purpose | Evidence |
|---|---|---|
| `src/timesfm3/` | Current TimesFM 3 PyTorch model and forecast interfaces | `src/timesfm3/__init__.py` |
| `src/timesfm/` | TimesFM 2.5 PyTorch/Flax implementation | `README.md` |
| `v1/` | Archived TimesFM 1/2 code | `README.md` |
| `tests/` | Package and Streamlit tests | `tests/test_explorer.py` |
| `timesfm-forecasting/` | Agent skill, scripts, and examples | `timesfm-forecasting/SKILL.md` |
| `timesfm3-usage/` | Benchmark runners and results | `timesfm3-usage/benchmarks/README.md` |
| `knowledge/` | Governed TimesFM 3 research | `knowledge/index.md` |
| `.github/workflows/` | Build and guarded publish automation | `.github/workflows/main.yml` |

## Entry Points

- Explorer runtime: `streamlit_app.py`, launched by `launch_app.cmd` or
  `streamlit run`.
- Python interfaces: `timesfm3.TimesFM3Forecaster` and
  `timesfm3.TimesFM3Evaluator`, exported by `src/timesfm3/__init__.py`.
- CSV helper: `timesfm-forecasting/scripts/forecast_csv.py`.

## Module Boundaries

| Boundary | Belongs here | Must not be here |
|---|---|---|
| Streamlit page | Widgets, session state, presentation | Model tensor implementation |
| `timesfm3.explorer` | Upload validation, preparation, artifacts | Widget rendering |
| Forecaster/evaluator | Model loading and inference | CSV/Parquet UI policy |
| Model modules | Neural network operations/configuration | App session state |
| `v1/` | Archived releases | Current-package test discovery |

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
