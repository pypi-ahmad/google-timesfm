# Codebase Concerns

## Top Risks

| Severity | Concern | Evidence | Impact | Suggested action |
|---|---|---|---|---|
| High | Default TimesFM 3 weights prohibit production/commercial use | `README.md` | Accidental license breach | Keep acknowledgement and license notice |
| Medium | Model/checkpoint memory can exhaust local GPU/RAM | `timesfm_app.worker`, `services.py` | Failed forecasts | Keep bounded batches and clear OOM guidance |
| Medium | Local PostgreSQL, broker, or artifact state can be unavailable | `timesfm_app` | Jobs or saved artifacts unavailable | Use `dev.ps1 doctor`, backups, and recovery guidance |

## Technical Debt

| Item | Why | Where | Risk | Suggested fix |
|---|---|---|---|---|
| Legacy single-page UI | Retained diagnostic Explorer | `streamlit_app.py` | Divergence from the primary workbench | Keep documentation visibly legacy |
| Two test layouts | Upstream evolution | `tests/`, `src/timesfm3/*_test.py` | Commands can omit tests | Keep the CI command explicit |
| No coverage gate | Not configured | `pyproject.toml` | Regressions may lack tests | Establish threshold after tool stability is proven |

## Security Concerns

| Risk | Category | Evidence | Current mitigation | Gap |
|---|---|---|---|---|
| Untrusted uploads | OWASP A04 | `parse_upload` | Type/raw/decoded-size and value validation | Parquet decoding still depends on PyArrow |
| Local checkpoint files | OWASP A08 | `timesfm3_forecaster.py` | safetensors or `weights_only=True` | Users still choose trusted files/directories |
| Local unauthenticated API | OWASP A01 | `timesfm_app.api` | Loopback binding and origin checks | No auth if deliberately exposed remotely |

## Performance and Scaling Concerns

| Concern | Evidence | Symptom | Scaling risk | Suggested improvement |
|---|---|---|---|---|
| Large checkpoint | resident GPU worker | Slow first run/high VRAM | One owned worker limits throughput | Keep one owned worker and bounded batches |
| Large result artifacts | PostgreSQL metadata plus artifact store | Disk growth | Retention is opt-in | Preview retention before applying it |

## Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe strategy |
|---|---|---|---|
| `src/timesfm3/` | Core inference shapes and device behavior | Recent TimesFM 3 commits | Run co-located model/forecaster tests |
| `streamlit_app.py` / `explorer.py` | New upload-to-model flow | Added in recent explorer commit | Test helpers and AppTest together |

## `[ASK USER]` Questions

1. [ASK USER] Will this Streamlit app remain local-only, or must future work add
   authentication and deployment hardening?
2. [ASK USER] Should a future deployment replace the local DuckDB store with a
   multi-user persistence service?

## Evidence

- `git log --oneline`
- `README.md`
- `streamlit_app.py`
- `src/timesfm3/explorer.py`
