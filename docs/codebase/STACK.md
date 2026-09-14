# Technology stack

## Runtime summary

| Area | Value | Evidence |
|---|---|---|
| Primary language | Python | `pyproject.toml` |
| Runtime | Python 3.10 or newer | `pyproject.toml` `requires-python` |
| Package manager | uv | `README.md`, `.github/workflows/main.yml` |
| Build system | setuptools via PEP 517 | `pyproject.toml` |
| Product UI | Next.js, React, TypeScript, Tailwind | `web/package.json` |
| API | FastAPI and Pydantic | `src/timesfm_app/api.py` |
| Durable state | PostgreSQL + SQLAlchemy | `src/timesfm_app/store.py` |
| Job delivery | Memurai + Dramatiq | `src/timesfm_app/jobs.py` |

## Production frameworks and dependencies

| Dependency | Version constraint | Role | Evidence |
|---|---|---|---|
| NumPy | `>=1.26.4` | Array and forecasting data | `pyproject.toml` |
| PyTorch | `>=2.0.0` optional | TimesFM inference | `pyproject.toml` |
| Hugging Face Hub | `>=0.28.0` | Checkpoint retrieval | `pyproject.toml` |
| safetensors | `>=0.5.3` | Safe weight loading | `pyproject.toml` |
| Streamlit | `>=1.57` optional | Local explorer UI | `pyproject.toml`, `streamlit_app.py` |
| DuckDB | `>=1.5.5` optional | Upload reading and local run persistence | `pyproject.toml` |
| FastAPI / Uvicorn | app extra | Local HTTP API and server | `pyproject.toml` |
| SQLAlchemy / psycopg | app extra | PostgreSQL records and migrations | `pyproject.toml` |
| Dramatiq / Redis client | app extra | Durable outbox delivery and workers | `pyproject.toml` |
| pandas / PyArrow / Altair | app extras | Tabular I/O and charts | `pyproject.toml` |

## Development toolchain

| Tool | Purpose | Evidence |
|---|---|---|
| pytest | Tests | `pyproject.toml`, `tests/` |
| Ruff | Lint and format | `pyproject.toml` |
| ty | Static type checking | `pyproject.toml` |
| build | Distribution build | `.github/workflows/main.yml` |

## Key commands

```powershell
.\dev.ps1 setup
.\dev.ps1 doctor
uv run --no-sync pytest -q tests/test_app_api.py tests/test_app_jobs.py
npm --prefix web run typecheck
```

## Environment and config

- The primary workbench UI is on port 3000; FastAPI is on 8001, PostgreSQL on
  55432, and Memurai on 56379.
- Streamlit port configuration applies only to the legacy Explorer.
- Hugging Face can use its standard `HF_TOKEN` environment variable; the app
  checks only whether it exists (`src/timesfm3/explorer.py`).
- Default TimesFM 3 weights are non-commercial and non-production, as stated in
  `README.md` and `knowledge/concepts/timesfm3-licensing.md`.
- The workbench is a local Windows application. It is not configured as a
  multi-user deployment.

## Evidence

- `pyproject.toml`
- `.streamlit/config.toml`
- `.github/workflows/main.yml`
- `streamlit_app.py`
