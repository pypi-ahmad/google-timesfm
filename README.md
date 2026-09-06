# TimesFM-3 Workbench

A local React + FastAPI workbench and Python toolkit for Google's TimesFM-3
zero-shot forecasting model with CSV or Parquet data. Persistent jobs, datasets,
drafts, results, and tracking survive browser restarts.

This fork provides a guided upload workflow, univariate and multivariate
forecasts, covariates, holdout evaluation, nested quantile bands, rolling
backtests, scenarios, baseline comparisons, forecast tracking, local or pinned
checkpoints, and portable result bundles.

> [!IMPORTANT]
> The source code is Apache-2.0. The default TimesFM-3 pretrained weights use
> the separate TimesFM Non-Commercial License v1.0 and are restricted to
> non-commercial, non-production use. See
> [Licensing and versions](docs/explanation/licensing-and-versions.md).

## Quick start

### React workbench on Windows 11

Install uv, Node.js 24 or newer, and Git. From this repository:

Double-click **`launch_workbench.cmd`**. It sets up missing components on first
launch, starts the services, waits for readiness, and opens the browser. Running
it again reopens the existing workbench. The terminal stays open with live,
service-labeled logs. Ctrl+C closes the log view; `.\dev.ps1 stop` stops the app.

For manual setup and startup:

```powershell
.\dev.ps1 setup
.\dev.ps1 start
```

Open <http://localhost:3000>. API documentation: <http://127.0.0.1:8001/docs>.
Setup provisions private PostgreSQL and Memurai processes and builds the frontend.
Use `.\dev.ps1 doctor` to inspect service health, `.\dev.ps1 stop` to stop the
workbench, and `.\dev.ps1 dev` for frontend hot reload.

The seven sections cover data, forecasting, experiments, scenarios, tracking,
models, and an overview dashboard. Uploads and derived artifacts are retained;
PostgreSQL stores immutable versions, jobs, results, and revisioned drafts.
Retention is opt-in in Workspace settings.

See [Run the native workbench](docs/how-to/native-workbench.md) for service ports,
CPU selection, recovery, legacy imports, and monitoring.

### Legacy Streamlit Explorer

The following setup and port 9587 apply to the retained Streamlit UI. Its DuckDB
history is separate from the new workbench. For an API-only diagnostic client:

```powershell
uv run --no-sync streamlit run diagnostic_app.py --server.port=9588
```

### Requirements

- Windows 11, macOS, or Linux
- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- Internet access for the first Hugging Face checkpoint download

Clone and install the application:

```powershell
git clone https://github.com/pypi-ahmad/google-timesfm.git
cd google-timesfm
uv sync --extra torch --extra app --group dev
```

On Windows, start the explorer on port `9587`:

```powershell
.\launch_app.cmd
```

Open <http://localhost:9587>. The launcher stops an earlier instance of
this exact application, but refuses to stop an unrelated process using the
port. If that happens, follow the
[alternate-port instructions](docs/troubleshooting.md#the-launcher-says-port-9587-belongs-to-another-process).

For other platforms, or to launch directly:

```powershell
uv run streamlit run streamlit_app.py --server.port=9587
```

The built-in multivariate demo uses targets, past-only covariates, known-future
covariates, quantiles, charts, and ZIP export. It does not require a data file.

DuckDB reads uploaded files and stores the newest 25 derived forecast runs in
`data/timesfm.duckdb`. Original uploads are not stored, and the database is
ignored by Git.

## Use your own data

Upload wide or long-format CSV or Parquet files. Each row is a time step; long
tables can use group identifiers such as `store_id` to create one series per
group.

```csv
date,sales,temperature,promotion
2026-01-01,101,20.1,0
2026-01-02,107,20.4,1
2026-01-03,103,20.2,0
```

In the app, choose a timestamp column if needed, assign targets and optional
covariates, choose forecast or holdout evaluation, accept the model-weights
restriction, and run the forecast. Inspect the charts and metrics, then
download the ZIP result bundle.

Known-future covariates need appended rows for the complete horizon. Leave
target cells empty in those future rows.
[Prepare data](docs/how-to/prepare-data.md) has complete examples.

## Python API

```python
import numpy as np

from timesfm3 import TimesFM3Forecaster

model = TimesFM3Forecaster.from_pretrained(
  "google/timesfm-3.0-pytorch",
  device="cpu",
)
context = np.sin(np.linspace(0, 12, 128)).astype(np.float32)
output = model.predict(
  context=context,
  horizon=24,
  return_quantiles=True,
)

assert output.forecast is not None
assert output.forecast.shape == (24,)
assert output.quantiles is not None
assert output.quantiles.shape == (24, 9)
```

The first run downloads the checkpoint from Hugging Face. Use `device="cuda"`
on a compatible NVIDIA setup. The
[TimesFM-3 API guide](docs/how-to/use-python-api.md) covers batching,
multivariate arrays, evaluator defaults, and covariate shapes.

## Documentation

| Goal | Document |
|---|---|
| Complete a first forecast | [First forecast tutorial](docs/tutorials/first-forecast.md) |
| Use the native workbench | [Native workbench guide](docs/how-to/native-workbench.md) |
| Inspect local API resources | [Workbench API reference](docs/reference/workbench-api.md) |
| Use the legacy diagnostic UI | [Legacy Streamlit Explorer](docs/how-to/use-streamlit-explorer.md) |
| Format CSV or Parquet data | [Prepare data](docs/how-to/prepare-data.md) |
| Forecast from Python | [Use the TimesFM-3 API](docs/how-to/use-python-api.md) |
| Run the legacy CSV helper | [Use the CSV helper](docs/how-to/use-csv-helper.md) |
| Look up legacy Explorer settings | [Legacy Explorer reference](docs/reference/explorer.md) |
| Look up Python interfaces | [Python API reference](docs/reference/python-api.md) |
| Understand the design | [Architecture](docs/explanation/architecture.md) |
| Understand the workbench design | [Workbench architecture](docs/explanation/workbench-architecture.md) |
| Resolve a problem | [Troubleshooting](docs/troubleshooting.md) |
| Work on the repository | [Contributing](CONTRIBUTING.md) |

The [documentation home](docs/README.md) includes the full map, maintainer
codebase notes, and the research knowledge base.

## What is included

- `streamlit_app.py`: local interactive explorer
- `src/timesfm_app/`: FastAPI API, persistence, workers, and local supervisor
- `web/`: Next.js workbench
- `src/timesfm3/`: current TimesFM-3 PyTorch implementation
- `src/timesfm/`: TimesFM 2.5 implementation
- `timesfm-forecasting/`: agent skill, TimesFM 2.5 helper, and examples
- `knowledge/`: draft research and source extracts in OKF format

The TimesFM 2.5 CSV helper and examples are archived compatibility material.
They do not use the TimesFM-3 workbench pipeline. See the
[archived compatibility note](timesfm-forecasting/SKILL.md).

## Development checks

```powershell
uv run pytest -q tests src/timesfm3
uv run ruff check streamlit_app.py src/timesfm3 tests
uv run ty check streamlit_app.py src/timesfm3/explorer.py
uv build
```

[Contributing](CONTRIBUTING.md) lists focused checks and known CUDA-host test
behavior.

## Upstream project

TimesFM was developed by Google Research. This fork is not an officially
supported Google product.

- [Google Research TimesFM repository](https://github.com/google-research/timesfm)
- [TimesFM-3 announcement][timesfm3-announcement]
- [TimesFM-3 checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch)
- [Original TimesFM paper](https://arxiv.org/abs/2310.10688)

## License

Repository source: [Apache License 2.0](LICENSE).

Default TimesFM-3 model materials: separate non-commercial license distributed
with the checkpoint. Review the current checkpoint terms before use.

[timesfm3-announcement]: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
