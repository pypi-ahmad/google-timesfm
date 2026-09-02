# Technology Stack

## Runtime Summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `pyproject.toml` |
| Runtime | Python 3.10 or newer | `pyproject.toml` `requires-python` |
| Package manager | uv | `README.md`, `.github/workflows/main.yml` |
| Build system | setuptools via PEP 517 | `pyproject.toml` |

## Production Frameworks and Dependencies

| Dependency | Version constraint | Role | Evidence |
|---|---|---|---|
| NumPy | `>=1.26.4` | Array and forecasting data | `pyproject.toml` |
| PyTorch | `>=2.0.0` optional | TimesFM inference | `pyproject.toml` |
| Hugging Face Hub | `>=0.28.0` | Checkpoint retrieval | `pyproject.toml` |
| safetensors | `>=0.5.3` | Safe weight loading | `pyproject.toml` |
| Streamlit | `>=1.57` optional | Local explorer UI | `pyproject.toml`, `streamlit_app.py` |
| DuckDB | `>=1.5.5` optional | Upload reading and local run persistence | `pyproject.toml` |
| pandas / PyArrow / Altair | app extras | Tabular I/O and charts | `pyproject.toml` |

## Development Toolchain

| Tool | Purpose | Evidence |
|---|---|---|
| pytest | Tests | `pyproject.toml`, `tests/` |
| Ruff | Lint and format | `pyproject.toml` |
| ty | Static type checking | `pyproject.toml` |
| build | Distribution build | `.github/workflows/main.yml` |

## Key Commands

```powershell
uv sync --extra torch --extra app --group dev
uv run --with build python -m build
uv run pytest -q tests src/timesfm3
uv run ruff check streamlit_app.py src/timesfm3 tests
```

## Environment and Config

- Streamlit port is configured in `.streamlit/config.toml`.
- Hugging Face can use its standard `HF_TOKEN` environment variable; the app
  checks only whether it exists (`src/timesfm3/explorer.py`).
- Default TimesFM 3 weights are non-commercial and non-production, as stated in
  `README.md` and `knowledge/concepts/timesfm3-licensing.md`.
- No container or production process definition is present.

## Evidence

- `pyproject.toml`
- `.streamlit/config.toml`
- `.github/workflows/main.yml`
- `streamlit_app.py`
