# Coding Conventions

## Naming Rules

| Item | Rule | Example | Evidence |
|---|---|---|---|
| Files | `snake_case.py` | `timesfm3_forecaster.py` | `src/timesfm3/` |
| Functions | `snake_case` | `prepare_batch` | `src/timesfm3/explorer.py` |
| Types | `PascalCase` | `ForecastSettings` | `src/timesfm3/explorer.py` |
| Private names | leading underscore | `_time_axis` | `src/timesfm3/explorer.py` |
| Constants | uppercase snake case | `MAX_CONTEXT` | `src/timesfm3/explorer.py` |
| TypeScript components | `PascalCase.tsx` | `ForecastChart.tsx` export | `web/src/components/` |
| TypeScript utilities/hooks | `kebab-case.ts` | `use-context.ts` | `web/src/` |

## Formatting and Linting

- Ruff is configured for an 88-character line length and two-space indentation
  in `pyproject.toml`.
- Run `uv run ruff check ...` and `uv run ruff format --check ...`.
- `ty` provides targeted static type checks in CI.
- Prettier formats the React workbench; run `npm --prefix web run typecheck` for
  its static boundary.

## Import and Module Conventions

- Standard-library, third-party, then local imports are separated.
- Package internals use relative imports; public consumers import `timesfm3`.
- `src/timesfm3/__init__.py` explicitly defines public exports through `__all__`.

## Error and Logging Conventions

- App-domain validation raises `ExplorerError`; HTTP exceptions map expected
  failures to actionable API responses; low-level public shape errors use
  `ValueError`.
- The workbench writes structured service logs and Prometheus metrics without
  logging credentials. The legacy Explorer still summarizes expected failures in
  its UI.
- Secrets must stay in environment variables or untracked configuration files.

## Testing Conventions

- Tests use pytest, plain assertions, parametrization, and `unittest.mock`.
- Files are named `test_*.py` or co-located `*_test.py`.
- No enforced coverage threshold is configured.

## Evidence

- `pyproject.toml`
- `src/timesfm3/__init__.py`
- `src/timesfm3/explorer.py`
- `tests/test_explorer.py`
