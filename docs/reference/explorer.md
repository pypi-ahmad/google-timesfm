# Explorer Reference

## Runtime

| Item | Value |
|---|---|
| Entry point | `streamlit_app.py` |
| Default URL | `http://localhost:9587` |
| Checkpoint | `google/timesfm-3.0-pytorch` |
| Model cache | One Streamlit resource per process |
| Run history | Newest 25 derived runs in local DuckDB |
| Upload storage | Temporary file deleted after reading; decoded session memory |

## Input limits

| Limit | Value |
|---|---:|
| Raw file size | 50 MiB per file |
| Combined raw uploads | 200 MiB |
| Decoded dataframe | 256 MiB per file |
| Combined decoded dataframes | 512 MiB |
| Context or horizon | 1 to 15,360 steps |
| Batch size | 1 to 64 |
| Variates per forward pass | 32 |

Above 32 combined target and covariate variates, joint mode requires explicit
benchmark-chunking approval. The evaluator uses seed `42`, may subsample
covariates, and chunks targets to fit the model boundary.

## Column roles

| Role | Historical values | Future values | Notes |
|---|---|---|---|
| Timestamp | Optional | Optional | Must parse and be unique |
| Target | Required | Empty for future forecast | At least two context values |
| Past-only covariate | Context span required; cells may be missing | Not required | Same context length as target |
| Past-and-future covariate | Context span required; cells may be missing | Every cell required for full horizon | Context plus horizon |

Historical missing model values are linearly interpolated. Future cells in a
selected past-and-future covariate must be complete.

## Forecast settings

| Setting | Default | Allowed values or range |
|---|---:|---|
| Horizon | `32` in UI | 1 to 15,360 |
| Context length | `512` | 1 to 15,360 |
| Task | Forecast future | Forecast or holdout |
| Series mode | Joint multivariate | Multivariate or independent univariate |
| Return quantiles | On | On/off |
| Symmetric averaging | On | On/off |
| Clamp nonnegative series | On | On/off |
| Sort quantiles | On | On/off |
| External z-normalization | Off | On/off |
| Known-future padding | `none` | `none` or `edge` |
| Batch size | `4` | 1 to 64 |

`edge` padding applies at the model boundary; the explorer still requires
selected known-future covariates for all uploaded future rows.

## Forecast table

Each output row represents one forecast step for one target in one dataset.

| Column | Meaning |
|---|---|
| `dataset` | In-session dataset identifier |
| `target` | Original target column name |
| `step` | One-based forecast step |
| `timestamp` | Uploaded or generated future axis |
| `point` | Median point forecast |
| `actual` | Held-out value; holdout runs only |
| `q0.1` … `q0.9` | Quantile forecasts when enabled |

## Holdout metrics

Metrics are grouped by dataset and target:

- observation count
- mean absolute error (MAE)
- root mean squared error (RMSE)
- symmetric mean absolute percentage error (sMAPE)
- mean pinball loss when quantiles exist
- `q0.1` to `q0.9` empirical coverage when quantiles exist

## ZIP bundle

| File | Included | Contents |
|---|---|---|
| `forecast.csv` | Always | Long-form forecasts and optional actuals/quantiles |
| `metrics.csv` | Holdout with valid observations | Per-target accuracy metrics |
| `run.json` | Always | Reproducibility manifest |

The manifest includes schema version, run ID, UTC creation time, checkpoint,
repository revision, runtime versions, device, settings, mapping, input hashes,
context shapes, lineage, chunking, and license identifier. It does not contain
uploaded data or authentication tokens.

## Error boundary

Expected input and settings failures appear as `ExplorerError` messages. CUDA
out-of-memory failures receive dedicated guidance. Other model or download
errors are summarized by exception type without displaying token values.
