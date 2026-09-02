# Testing Patterns

## Test Stack and Commands

- Framework: pytest `>=9.1.1`.
- Assertions/mocking: Python assertions, pytest parametrization/fixtures, and
  `unittest.mock`.

```powershell
uv run pytest -q tests src/timesfm3
uv run pytest -q tests/test_explorer.py tests/test_streamlit_app.py
uv run pytest -q src/timesfm3/timesfm3_forecaster_test.py
uv run pytest --cov=timesfm3 tests
```

The coverage command is available through `pytest-cov`, but no threshold is
configured and Windows NumPy/Pandas coverage behavior requires verification.

## Test Layout

- Package/application tests are under `tests/test_*.py`.
- TimesFM 3 model tests are co-located as `src/timesfm3/*_test.py`.
- Historical v1 tests are available from earlier Git revisions and require
  their own dependency environment.
- No global pytest setup file is present.

## Test Scope Matrix

| Scope | Covered? | Target | Notes |
|---|---|---|---|
| Unit | Yes | validation, layers, configs, transforms | deterministic arrays/fakes |
| Integration | Yes | checkpoint save/load, artifact ZIP | local temp files |
| UI startup | Yes | Streamlit page | Streamlit AppTest |
| Browser E2E | No | full uploaded forecast | `[TODO]` no browser suite |
| GPU smoke | Manual | real TimesFM 3 forecast | hardware/checkpoint dependent |

## Mocking and Isolation Strategy

- `FakePredictor` tests explorer orchestration without loading model weights.
- `mock.patch.object` isolates the checkpoint loader.
- Temporary directories isolate save/load tests.
- Flax-specific tests skip before importing Flax when the optional backend is
  absent.

## Coverage and Quality Signals

- Coverage tool: pytest-cov; threshold: `[TODO]` not configured.
- CI builds, runs targeted Ruff/ty, and executes current package tests.
- Real downloads and CUDA behavior are not required by CI.

## Evidence

- `pyproject.toml`
- `tests/test_explorer.py`
- `tests/test_streamlit_app.py`
- `tests/test_model_loading.py`
- `.github/workflows/main.yml`
