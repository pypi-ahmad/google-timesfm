# Architecture

## Primary workbench architecture

- Primary style: a local React client over a FastAPI application with durable
  PostgreSQL records, a transactional outbox, Memurai/Dramatiq queues, and
  separate GPU/CPU workers.
- The browser owns configuration and presentation. `timesfm_app` owns transport,
  persistence, process supervision, and job fencing. `timesfm3` owns numerical
  preparation, inference, analysis, and tracking behavior.
- See [the detailed workbench architecture](../explanation/workbench-architecture.md)
  for the data flow and job-state diagram.

## Legacy Streamlit forecast flow

```text
upload/demo -> parse and validate -> map columns -> prepare arrays
            -> cached evaluator -> predict_batch -> metrics/artifact -> UI/ZIP
```

1. `streamlit_app.py` collects files or builds a deterministic demo.
2. `parse_upload` creates an `UploadedDataset` and enforces input limits.
3. `prepare_batch` maps table columns into targets and covariate arrays.
4. `model_loading.resolve_model` resolves a pinned Hub revision or compatible
   local checkpoint, then the app obtains a cached `TimesFM3Evaluator`.
5. `run_forecast` calls the evaluator and records runtime.
6. `make_run_artifact` creates tables, metrics, lineage, and export metadata.

## Primary module responsibilities

| Module | Owns | Must not own |
|---|---|---|
| `web/` | Next.js UI, URL state, forms, tables, charts | Forecast algorithms or durable state |
| `timesfm_app.api` | HTTP contract, origin boundary, SSE | Model construction |
| `timesfm_app.store` | PostgreSQL records, outbox, leases, fencing | Forecast calculations |
| `timesfm_app.jobs` | Outbox delivery and tracking reconciliation | Browser state |
| `timesfm_app.worker` | One owned GPU/CPU attempt and progress | Request validation |
| `timesfm_app.services` | Preparation, execution, artifacts | UI rendering |
| `timesfm3` | TimesFM-3 forecasting and analytical logic | HTTP or persistence policy |

## Legacy module responsibilities

| Module | Owns | Must not own | Evidence |
|---|---|---|---|
| `streamlit_app.py` | UI state and rendering | Forecast algorithms | `streamlit_app.py` |
| `timesfm3.explorer` | App-domain validation and artifacts | Streamlit widgets | `src/timesfm3/explorer.py` |
| `timesfm3.data_preparation` | Bulk grouping, calendar features, readiness | Model execution | `src/timesfm3/data_preparation.py` |
| `timesfm3.analysis` | Leak-free historical comparisons | Widget rendering | `src/timesfm3/analysis.py` |
| `timesfm3.tracking` | Actual matching and immutable assessments | Upload parsing | `src/timesfm3/tracking.py` |
| `TimesFM3Evaluator` | Benchmark-compatible batching | Upload formats | `src/timesfm3/evaluator.py` |
| `TimesFM3Forecaster` | Checkpoint and model inference | UI policy | `src/timesfm3/timesfm3_forecaster.py` |
| `TimesFM3Torch` | Neural model computation | File parsing | `src/timesfm3/model.py` |

## Reused patterns

| Pattern | Where | Purpose |
|---|---|---|
| Frozen dataclasses | `timesfm3.explorer`, forecaster | Stable settings and outputs |
| Adapter | `TimesFM3Evaluator` | Benchmark batching over forecaster |
| Resource cache | `streamlit_app.py` | Reuse one heavyweight checkpoint |
| Dependency seam | predictor protocol/test fake | Test orchestration without loading weights |

## Known architectural risks

- `streamlit_app.py` remains a single declarative page; adding more workflows
  could reduce locality unless rendering stays separate from forecast execution.
- Current and archived packages share the `timesfm` name, so broad test
  discovery can import the wrong generation.
- DuckDB retains 25 untracked runs and 25 analyses; tracked runs are protected.
  Raw uploads remain session-only.

## Evidence

- `src/timesfm_app/`
- `web/`
- `streamlit_app.py` (legacy)
- `src/timesfm3/explorer.py`
- `src/timesfm3/evaluator.py`
- `src/timesfm3/timesfm3_forecaster.py`
