# Coding Conventions

## Naming Rules

| Item | Rule | Example | Evidence |
|---|---|---|---|
| Files | `snake_case.py` | `timesfm3_forecaster.py` | `src/timesfm3/` |
| Functions | `snake_case` | `prepare_batch` | `src/timesfm3/explorer.py` |
| Types | `PascalCase` | `ForecastSettings` | `src/timesfm3/explorer.py` |
| Private names | leading underscore | `_time_axis` | `src/timesfm3/explorer.py` |
| Constants | uppercase snake case | `MAX_CONTEXT` | `src/timesfm3/explorer.py` |

## Formatting and Linting

- Ruff is configured for an 88-character line length and two-space indentation
  in `pyproject.toml`.
- Run `uv run ruff check ...` and `uv run ruff format --check ...`.
- `ty` provides targeted static type checks in CI.

## Import and Module Conventions

- Standard-library, third-party, then local imports are separated.
- Package internals use relative imports; public consumers import `timesfm3`.
- `src/timesfm3/__init__.py` explicitly defines public exports through `__all__`.

## Error and Logging Conventions

- App-domain validation raises `ExplorerError`; low-level public shape errors use
  `ValueError`; the UI converts expected failures into `st.error` messages.
- The explorer has no application logging pipeline. Model/download exceptions
  are summarized by type in the UI without echoing tokens or local paths.
- Secrets must stay in environment variables or untracked
  `.streamlit/secrets.toml`.

## Testing Conventions

- Tests use pytest, plain assertions, parametrization, and `unittest.mock`.
- Files are named `test_*.py` or co-located `*_test.py`.
- No enforced coverage threshold is configured.

## Evidence

- `pyproject.toml`
- `src/timesfm3/__init__.py`
- `src/timesfm3/explorer.py`
- `tests/test_explorer.py`
