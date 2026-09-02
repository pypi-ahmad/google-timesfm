# Tutorial: Complete Your First Forecast

This tutorial launches the local explorer, forecasts the built-in demo, and
exports the result. You do not need to prepare a data file.

**Time:** about 10 minutes, plus the first checkpoint download.

## Before you begin

Install Python 3.10 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/). The default
TimesFM-3 weights may only be used for non-commercial, non-production work.

## 1. Install the project

```powershell
git clone https://github.com/pypi-ahmad/google-timesfm.git
cd google-timesfm
uv sync --extra torch --extra app --group dev
```

## 2. Start the explorer

On Windows:

```powershell
.\launch_app.cmd
```

On another platform:

```shell
uv run streamlit run streamlit_app.py --server.port=9587
```

Open <http://localhost:9587>. You should see **TimesFM-3 explorer** and a
runtime summary in the sidebar.

## 3. Inspect the demo

In **Data**:

1. Leave **Data source** set to **Demo**.
2. Select **Multivariate + covariates**.
3. Confirm `date` is the timestamp.
4. Confirm `sales` and `demand` are targets.
5. Confirm `temperature` is past-only and `promotion` is past-and-future.

The preview shows the table that will be converted to model arrays.

## 4. Run the forecast

In **Configure**:

1. Accept the model-weights restriction.
2. Leave **Forecast future**, **Joint multivariate**, horizon `32`, and the
   default probabilistic settings selected.
3. Choose `cuda` if it is available; otherwise use `cpu`.
4. Select **Run forecast**.

The first run downloads `google/timesfm-3.0-pytorch` from Hugging Face. Later
runs reuse the cached checkpoint.

## 5. Read the result

In **Results**:

- The gray line is historical context.
- The orange line is the point forecast.
- The shaded region spans `q0.1` to `q0.9` when quantiles are enabled.
- A holdout run also shows actual values and accuracy metrics.

Select a dataset and target to change the plotted series.

## 6. Export the run

Select **Download result bundle**. The ZIP contains:

- `forecast.csv`: one row per dataset, target, and forecast step
- `metrics.csv`: holdout metrics, when applicable
- `run.json`: settings, mappings, source hashes, runtime, and code revision

The app keeps the newest 25 derived runs in `data/timesfm.duckdb`. Original
uploads are not stored. The ZIP remains the portable record for sharing or
archiving a run.

## Checkpoint

You have now loaded the application, exercised a multivariate forecast, read a
prediction interval, and exported a reproducible result bundle.

Next, [prepare your own data](../how-to/prepare-data.md) or learn the complete
[explorer workflow](../how-to/use-streamlit-explorer.md).
