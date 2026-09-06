# Tutorial: Complete Your First Forecast

This tutorial launches the local TimesFM-3 Explorer, runs the built-in
multivariate demo, reads its uncertainty chart, and exports a reproducible
forecast. You do not need a data file.

**Time:** about 10 minutes, plus the first checkpoint download.

## Before you begin

Install Python 3.10 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The default
TimesFM-3 weights are restricted to non-commercial, non-production use.

## 1. Install the project

```powershell
git clone https://github.com/pypi-ahmad/google-timesfm.git
cd google-timesfm
uv sync --extra torch --extra app --group dev
```

## 2. Start the Explorer

On Windows, run:

```powershell
.\launch_app.cmd
```

On another platform, run:

```shell
uv run streamlit run streamlit_app.py --server.port=9587
```

Open <http://localhost:9587>. The sidebar shows the runtime, model selection,
and memory controls. The page has four workspaces: **Prepare**, **Forecast**,
**Evaluate**, and **Track**.

## 3. Prepare the demo

Open **Prepare**.

1. Leave **Data source** set to **Demo**.
2. Select **Multivariate + covariates**.
3. Confirm `date` is the timestamp.
4. Confirm `sales` and `demand` are targets.
5. Confirm `temperature` is past-only and `promotion` is past-and-future.

The data-readiness panel reports the context available to the model and any
input issue before a checkpoint is loaded.

## 4. Run the forecast

Open **Forecast**.

1. Accept the model-weights restriction.
2. Keep **Forecast future**, **Joint multivariate**, horizon `32`, and the
   default probabilistic controls.
3. In **Advanced inference**, choose `cuda` when it is available; otherwise
   use `cpu`.
4. Select **Run forecast**.

The first run downloads the default checkpoint. Later forecasts reuse the
cached resolved checkpoint until you select **Clear model from memory** or
change the model selection.

## 5. Read the result

The **Latest result** panel shows a selected dataset and target.

- The historical line is the supplied context.
- The point line is the median forecast.
- Nested shading shows central 20%, 40%, 60%, and 80% nominal prediction
  bands when quantiles are enabled.
- A holdout forecast also shows actual values, accuracy metrics, and observed
  interval coverage.

Prediction bands describe model uncertainty. Check observed coverage on your
own historical data before using them for decisions.

## 6. Export the run

Select **Download result bundle**. The ZIP contains:

- `forecast.csv`: one row per dataset, target, and forecast step
- `metrics.csv`: holdout metrics, when applicable
- `calibration.csv`: interval coverage, when actuals and bounds exist
- `run.json`: settings, mappings, model identity, source hashes, runtime, and
  code revision

The app saves derived forecast outputs in `data/timesfm.duckdb`. It retains the
newest 25 untracked runs; tracked forecast vintages are protected. Original
uploads and historical context arrays are never stored in DuckDB.

You have now run a joint TimesFM-3 forecast and exported its record. Continue
with the [Explorer handbook](../how-to/timesfm3-explorer-handbook.md), or
[prepare your own data](../how-to/prepare-data.md).
