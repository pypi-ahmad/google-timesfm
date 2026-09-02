# Codebase Concerns

## Top Risks

| Severity | Concern | Evidence | Impact | Suggested action |
|---|---|---|---|---|
| High | Default TimesFM 3 weights prohibit production/commercial use | `README.md` | Accidental license breach | Keep acknowledgement and license notice |
| Medium | Model/checkpoint memory can exhaust local GPU/RAM | `streamlit_app.py`, `explorer.py` | Failed forecasts | Keep hard data limits and clear OOM guidance |
| Low | Run history is ephemeral | `streamlit_app.py` | Results lost on restart | Export ZIP; add persistence only if required |

## Technical Debt

| Item | Why | Where | Risk | Suggested fix |
|---|---|---|---|---|
| Single-page UI | Initial local explorer | `streamlit_app.py` | Lower locality as features grow | Split rendering only after another workflow appears |
| Two test layouts | Upstream evolution | `tests/`, `src/timesfm3/*_test.py` | Commands can omit tests | Keep the CI command explicit |
| No coverage gate | Not configured | `pyproject.toml` | Regressions may lack tests | Establish threshold after tool stability is proven |

## Security Concerns

| Risk | Category | Evidence | Current mitigation | Gap |
|---|---|---|---|---|
| Untrusted uploads | OWASP A04 | `parse_upload` | Type/raw/decoded-size and value validation | Parquet decoding still depends on PyArrow |
| Local checkpoint files | OWASP A08 | `timesfm3_forecaster.py` | safetensors or `weights_only=True` | Users still choose trusted files/directories |
| Local unauthenticated UI | OWASP A01 | `streamlit_app.py` | Localhost-oriented launcher | No auth if deliberately exposed remotely |

## Performance and Scaling Concerns

| Concern | Evidence | Symptom | Scaling risk | Suggested improvement |
|---|---|---|---|---|
| Large checkpoint | cached forecaster in `streamlit_app.py` | Slow first run/high VRAM | Concurrent sessions share finite GPU | Keep one cached model and bounded batches |
| In-memory tables/artifacts | `RunArtifact`, session history | RAM increases per run | Large result tables multiply memory | Three-run cap and decoded limits |

## Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe strategy |
|---|---|---|---|
| `src/timesfm3/` | Core inference shapes and device behavior | Recent TimesFM 3 commits | Run co-located model/forecaster tests |
| `streamlit_app.py` / `explorer.py` | New upload-to-model flow | Added in recent explorer commit | Test helpers and AppTest together |

## `[ASK USER]` Questions

1. [ASK USER] Will this Streamlit app remain local-only, or must future work add
   authentication and deployment hardening?
2. [ASK USER] Should run history ever persist beyond the current session, or is
   ZIP export the intended durable record?

## Evidence

- `git log --oneline`
- `README.md`
- `streamlit_app.py`
- `src/timesfm3/explorer.py`
