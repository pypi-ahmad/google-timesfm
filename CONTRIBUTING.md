# Contributing

This fork contains the current TimesFM-3 package, a local Streamlit explorer,
TimesFM 2.5 compatibility code, archived versions, examples, and research
knowledge. Keep changes within the relevant generation and test boundary.

## Set up the repository

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
| `streamlit_app.py` | Explorer widgets and session state |
| `src/timesfm3/explorer.py` | App validation, orchestration, and artifacts |
| `src/timesfm/` | TimesFM 2.5 implementation |
| `v1/` | Archived TimesFM 1 and 2 code |
| `tests/` | Package and application tests |
| `timesfm-forecasting/` | Agent skill, compatibility scripts, examples |
| `docs/` | User and developer documentation |
| `knowledge/` | Governed draft research and references |

See the [codebase documentation](docs/codebase/STRUCTURE.md) before a broad
change.

## Make focused changes

- Follow the local two-space Python indentation and 88-character line length.
- Keep Streamlit widgets in `streamlit_app.py` and forecast behavior outside it.
- Keep tabular app policy in `timesfm3.explorer` and tensor behavior in the
  forecaster/model layers.
- Preserve archived code unless the task explicitly targets it.
- Add a focused regression test for behavior changes.
- Keep TimesFM-3 and TimesFM 2.5 APIs clearly separated in code and docs.

## Run checks

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
uv run ty check streamlit_app.py src/timesfm3/explorer.py
uv build
```

The archived `v1/` code has separate historical dependencies and should not be
mixed into the current-package test command.

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
5. Update the coverage matrix in [docs/README.md](docs/README.md).
6. Treat current source and tests as authoritative over draft OKF entries.

## Submit a change

Keep commits scoped and describe:

- the user-visible behavior or documentation outcome
- validation commands and results
- hardware-dependent checks not run
- model-version and license implications

Do not publish the package or push changes unless the repository owner has
authorized that external action.
