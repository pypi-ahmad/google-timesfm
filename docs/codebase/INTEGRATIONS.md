# External Integrations

## Integration Inventory

| System | Type | Purpose | Auth | Criticality | Evidence |
|---|---|---|---|---|---|
| Hugging Face Hub | API/cache | Download TimesFM checkpoints | Optional standard token | High | `src/timesfm3/timesfm3_forecaster.py` |
| Local filesystem | files | Uploaded data, checkpoints, ZIP downloads | OS permissions | High | `src/timesfm3/explorer.py` |
| Git | subprocess | Record source revision in manifests | Local checkout | Low | `repository_revision` in `explorer.py` |
| PyPI | package registry | Manual upstream package publication | GitHub secret | Medium | `.github/workflows/manual_publish.yml` |

## Data Stores

| Store | Role | Access layer | Key risk | Evidence |
|---|---|---|---|---|
| Streamlit session memory | Last three runs | `streamlit_app.py` | Lost at process/session end | `streamlit_app.py` |
| Hugging Face cache | Checkpoint reuse | `huggingface_hub` mixin | Disk use/stale revision | `timesfm3_forecaster.py` |

No database, queue, or remote application datastore is configured.

## Secrets and Credentials Handling

- Hugging Face uses its standard environment configuration; the capability
  report exposes presence only, never the token value.
- PyPI publishing reads `PYPI_API_TOKEN` from GitHub Actions secrets and is
  guarded to the upstream repository.
- `.streamlit/secrets.toml` is ignored by Git.
- Rotation policy: `[TODO]` external platform policy is not stored in this repo.

## Reliability and Failure Behavior

- Hugging Face loading delegates retry/cache behavior to the library; this repo
  does not add retry or circuit-breaker logic.
- The only explicit integration timeout is two seconds for the local Git
  revision subprocess.
- Failed checkpoint access is converted into a generic Streamlit error.

## Observability for Integrations

- The UI reports checkpoint cache/auth availability and model-load progress.
- No metrics, tracing, or centralized logs are configured.

## Evidence

- `src/timesfm3/timesfm3_forecaster.py`
- `src/timesfm3/explorer.py`
- `streamlit_app.py`
- `.github/workflows/manual_publish.yml`
