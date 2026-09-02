# Architecture

## Architectural Style

- Primary style: library packages plus a thin local UI adapter.
- Evidence: `streamlit_app.py` delegates parsing, preparation, inference, and
  artifact work to `src/timesfm3/explorer.py`, which calls the evaluator.
- Constraints: variable-length numerical series, optional covariates, large
  checkpoint memory, and a separate license for default TimesFM 3 weights.

## Streamlit Forecast Flow

```text
upload/demo -> parse and validate -> map columns -> prepare arrays
            -> cached evaluator -> predict_batch -> metrics/artifact -> UI/ZIP
```

1. `streamlit_app.py` collects files or builds a deterministic demo.
2. `parse_upload` creates an `UploadedDataset` and enforces input limits.
3. `prepare_batch` maps table columns into targets and covariate arrays.
4. `load_forecaster` obtains a cached `TimesFM3Evaluator` checkpoint instance.
5. `run_forecast` calls the evaluator and records runtime.
6. `make_run_artifact` creates tables, metrics, lineage, and export metadata.

## Module Responsibilities

| Module | Owns | Must not own | Evidence |
|---|---|---|---|
| `streamlit_app.py` | UI state and rendering | Forecast algorithms | `streamlit_app.py` |
| `timesfm3.explorer` | App-domain validation and artifacts | Streamlit widgets | `src/timesfm3/explorer.py` |
| `TimesFM3Evaluator` | Benchmark-compatible batching | Upload formats | `src/timesfm3/evaluator.py` |
| `TimesFM3Forecaster` | Checkpoint and model inference | UI policy | `src/timesfm3/timesfm3_forecaster.py` |
| `TimesFM3Torch` | Neural model computation | File parsing | `src/timesfm3/model.py` |

## Reused Patterns

| Pattern | Where | Purpose |
|---|---|---|
| Frozen dataclasses | `timesfm3.explorer`, forecaster | Stable settings and outputs |
| Adapter | `TimesFM3Evaluator` | Benchmark batching over forecaster |
| Resource cache | `streamlit_app.py` | Reuse one heavyweight checkpoint |
| Dependency seam | predictor protocol/test fake | Test orchestration without loading weights |

## Known Architectural Risks

- `streamlit_app.py` remains a single declarative page; adding more workflows
  could reduce locality unless rendering stays separate from forecast execution.
- Current and archived packages share the `timesfm` name, so broad test
  discovery can import the wrong generation.
- DuckDB retains the newest 25 derived runs; raw uploads remain session-only.

## Evidence

- `streamlit_app.py`
- `src/timesfm3/explorer.py`
- `src/timesfm3/evaluator.py`
- `src/timesfm3/timesfm3_forecaster.py`
