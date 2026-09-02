# How to Use the Streamlit Explorer

## Launch the app

```powershell
uv sync --extra torch --extra app --group dev
.\launch_app.cmd
```

Open <http://localhost:9587>. To launch directly, run:

```powershell
uv run streamlit run streamlit_app.py --server.port=9587
```

## Load data

Use **Demo** to verify the environment, or choose **Upload** for one or more CSV
or Parquet files. The files stay in the local Streamlit process and parsed data
is cached only for the current browser session.

Assign the common columns:

1. Choose a timestamp or use row numbers.
2. Select one or more targets.
3. Optionally select past-only covariates.
4. Optionally select past-and-future covariates.

See [Prepare data](prepare-data.md) for role definitions and future-row rules.

## Choose the task

- **Forecast future:** predict after the final observed target row.
- **Evaluate holdout:** reserve the final observed target rows, forecast them,
  and compute metrics.

Choose **Joint multivariate** to use cross-variate information and covariates.
Choose **Independent univariate** to forecast each target independently; this
mode does not use covariates.

## Configure inference

- **Horizon:** number of future or held-out steps.
- **Context length:** maximum historical steps supplied to the model.
- **Device:** CUDA when available, otherwise CPU.
- **Batch size:** datasets evaluated together per core operation.
- **Return quantiles:** add `q0.1` through `q0.9` predictions.
- **Symmetric averaging:** average forecasts from the series and its sign-flipped
  counterpart.
- **Clamp nonnegative series:** prevent negative forecasts for inferred
  nonnegative inputs.
- **Sort quantiles:** enforce nondecreasing quantile outputs.

Advanced controls and exact defaults are in the
[Explorer reference](../reference/explorer.md).

## Run and compare

Accept the TimesFM-3 weights restriction, then select **Run forecast**. The
status panel separates checkpoint loading from forecasting.

The app retains the newest 25 successful runs in `data/timesfm.duckdb`. Use
**Compare** to overlay up to three point forecasts for the same dataset and
target. Original uploads and input history are not stored.

## Export results

Download the ZIP from **Results**. Store `run.json` with the CSV output because
it records the checkpoint, code revision, input hashes, settings, mappings,
lineage operations, runtime, and license identifier.

Spreadsheet formula prefixes in user-controlled CSV text are neutralized in
the exported CSV. The JSON manifest retains the original mapping names.

## Release model memory

Select **Unload model** in the sidebar to clear the cached model. On CUDA, the
app also asks PyTorch to empty its memory cache.
