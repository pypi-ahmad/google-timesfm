# Explorer Reference

## Runtime

| Item | Value |
|---|---|
| Entry point | `streamlit_app.py` |
| Default URL | `http://localhost:9587` |
| Default checkpoint | `google/timesfm-3.0-pytorch`; selectable repository/revision or local checkpoint |
| Model cache | One Streamlit resource per process |
| Run history | Latest 25 runs in the result view; tracked vintages remain available in **Track** from local DuckDB |
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
| Bulk forecast output | At most 250,000 rows |

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
| `dataset` | Named source plus canonical group keys for grouped input |
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
| `calibration.csv` | Actuals and interval bounds available | Nominal/observed coverage by target and horizon |
| `run.json` | Always | Reproducibility manifest |

The manifest includes schema version, run ID, UTC creation time, checkpoint,
repository revision, runtime versions, device, settings, mapping, input hashes,
context shapes, lineage, chunking, and license identifier. It does not contain
uploaded data or authentication tokens.

## Error boundary

Expected input and settings failures appear as `ExplorerError` messages. CUDA
out-of-memory failures receive dedicated guidance. Other model or download
errors are summarized by exception type without displaying token values.

## Analysis workflows

Analysis runs use TimesFM 3 and the existing inference settings. They do not
train or modify the model. All variants share one cutoff schedule per dataset.
Context ends strictly before each cutoff; preprocessing cannot see held-out
targets or future past-only covariates. Held-out actuals remain unfilled.

| Setting | Default | Limit or behavior |
|---|---|---|
| Anomaly observations | 50 | 1–1,000; horizon and stride fixed at 1 |
| Backtest windows | 5 | 1–100 |
| Backtest stride | Forecast horizon | Positive number of steps |
| Scenario count | One baseline | Up to three named alternatives |
| Analysis variates | Selected targets plus covariates | At most 32; no chunking or subsampling |
| Prediction rows | Previewed before execution | At most 250,000 across variants and datasets |
| Saved analyses | Newest 25 | Separate from forecast run retention |

For observed extent `n`, horizon `h`, window count `w`, and stride `s`, cutoffs
are `n-h-(w-1)s`, ..., `n-h`. Each context uses at most the configured context
length and needs two finite values per target. Insufficient context or missing
required future covariates rejects the request before model loading.

Anomaly detection requires sorted quantiles. Values strictly below `q0.1` or
above `q0.9` are flagged; equality is inside. Outputs include direction,
actual-minus-median residual, and nonnegative distance beyond the nearest bound.
The q10–q90 band is an **80% nominal prediction interval**.

Joint-versus-independent comparisons exclude covariates in both modes and
require two or more targets. Covariate usefulness keeps joint mode fixed and
runs full, empty, selected-group-removal, and individual-removal variants;
identical variants are deduplicated. Results describe predictive usefulness,
not causal importance.

Scenario edits apply only to mapped known-future covariates over the future
horizon. Each scenario starts from a fresh baseline copy. Differences are
scenario minus baseline; percentage differences are unavailable at zero
baseline. These are conditional predictions, not causal effects.

### Analysis results and persistence

Prediction rows retain dataset, target, variant, forecast origin, horizon step,
timestamp, actual where available, median, and enabled quantiles. Overlapping
windows retain distinct origin/step pairs. Metrics use scored forecast–actual
pairs and include missing counts; summaries do not average precomputed errors.
MAE, RMSE, sMAPE, pinball loss, and interval coverage are reported by target,
window, and forecast distance where applicable. No ranking pools targets with
different units.

The additive DuckDB `analysis_runs` table stores an ID, creation time, kind,
JSON manifest, and Parquet prediction/metric tables. Save and retention pruning
are transactional. Summary listing and loading are read-only; older databases
without this table show no saved analyses until their first save. A failed
forecast fails the analysis rather than publishing incomplete comparisons.

Exports contain `predictions.csv`, `metrics.csv`, and `analysis.json`, plus
workflow-specific anomaly, comparison, or scenario-edit CSVs. The manifest
includes settings, cutoffs, input hashes, variant membership, and authored
scenario overrides. Original uploads and historical context remain in session
memory; derived outputs can include held-out actuals. CSV text uses the same
spreadsheet-formula protection as forecast exports.

## Preparation and calendar features

`group_sources` splits long-format sources before timestamp validation, allowing
invalid groups to remain visible in the quality preview. `prepare_sources` adds
selected calendar features and future rows to session copies. `restore_preparation`
reapplies saved calendar settings during refresh. Uploaded targets are never filled
by these helpers. Model interpolation remains confined to each model context.

Source names and canonical group keys identify series across upload versions;
content hashes identify uploaded versions. Timestamp uniqueness is checked within
each group. Generated features are weekday 0–6, month 1–12, holiday 0/1, and one
0/1 column per inclusive event date range. Holidays use the selected country and
subdivision, including observed holidays. Calendar generation records the package
version, dates, frequency, and feature definitions in preparation metadata.

## Configuration and baseline comparisons

`ExperimentConfiguration(name, settings)` supplies each named variant to
`prepare_analysis(..., configurations=...)`. Settings experiments accept 2–8
configurations and include last-value and seasonal-naive baselines. Baseline
comparisons use the current inference settings. Checkpoint, mapping, mode, horizon,
and batch size stay fixed within a comparison; context and exposed inference
switches may vary. All configurations return quantiles.

Every comparison origin needs the largest requested context and seasonal period
available. Baselines use raw pre-origin observations: last finite value, or the
last seasonal block repeated. Missing seasonal values are not interpolated.
The `scored` flag marks identical finite `(dataset,target,origin,step,timestamp)`
keys across all methods. Metrics and exclusions follow that common mask.
Target-level ranks use MAE; aggregate ranks weight targets equally, without
pooling errors expressed in different units.

Central bands use q0.4–q0.6 (20%), q0.3–q0.7 (40%), q0.2–q0.8 (60%), and
q0.1–q0.9 (80%). Coverage includes observations on either boundary. Tables report
sample counts, mean width, missing actuals, invalid bounds, and crossings. These
are descriptive diagnostics, without interval fitting or a coverage guarantee.

## Model selection and forecast tracking

`ModelSelection` selects a Hub repository/revision or a local folder/file.
`resolve_model` runs only when inference is requested and returns immutable
revision/file fingerprints used in the resource cache key. Folder loading requires
`config.json` and `model.safetensors`. Standalone `.safetensors`, `.pt`, and `.pth`
files use the shipped architecture; PyTorch files use weights-only loading.
All selected checkpoints must match TimesFM-3 parameter names/shapes and q0.1–q0.9.
Offline mode passes local-only resolution and never retries another checkpoint.

Saved forecast vintages are immutable. Additive `tracked_runs`, `tracking_links`,
`run_assessments`, and `run_tables` tables preserve tracking associations,
versioned matched actuals, and derived forecast tables. Legacy rows remain
readable. Identical saves are idempotent; conflicting saves under an existing run
ID fail. Tracked runs are exempt from the 25-untracked-run retention policy.

Tracking associations explicitly map previous dataset IDs to current ones;
legacy `dataset_N` labels never auto-associate. Matching requires timestamps and
uses exact instants, normalizing timezone-aware values to UTC and rejecting
mixed naive/aware conventions. Revised actuals create another assessment.
Refresh restores saved preparation/settings and the pinned model identity,
honors the current offline switch, and creates a linked new forecast. Original
uploads and context history remain outside persistent storage.
