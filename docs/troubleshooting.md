# Troubleshooting

## The launcher says port 9587 belongs to another process

The Windows launcher only stops a listener whose command line contains both the
absolute path to this app and `--server.port=9587`. It refuses to kill an
unrelated process.

Stop or reconfigure that process, then run:

```powershell
.\launch_app.cmd
```

To use another port temporarily, bypass the launcher:

```powershell
uv run streamlit run streamlit_app.py --server.port=9588
```

## `uv` is not recognized

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), open a
new terminal, and verify:

```powershell
uv --version
```

## A dependency or import is missing

Synchronize the application, PyTorch, and development dependencies:

```powershell
uv sync --extra torch --extra app --group dev
uv run python -c "import streamlit, torch, timesfm3; print('imports ok')"
```

## The first forecast appears stuck

The first run downloads the TimesFM-3 checkpoint from Hugging Face. Check the
terminal for download progress and confirm network access. Later runs use the
Hugging Face cache.

If the checkpoint is private or access rules change, authenticate through the
standard Hugging Face environment configuration. Never add a token to source,
documentation, `.streamlit/config.toml`, or a committed secrets file.

## CUDA ran out of memory

Try these changes in order:

1. Reduce batch size to `1`.
2. Reduce context length.
3. Reduce horizon.
4. Reduce target and covariate counts.
5. Select **Clear model from memory**, then retry.
6. Use CPU if the forecast fits system RAM.

## An upload is rejected as too large

The app enforces compressed and decoded limits. Parquet and compact CSV files
can expand substantially in memory. Split the data into smaller files or remove
unused columns; do not merely recompress the same oversized dataframe.

## A column contains invalid values

Selected model columns must be numeric or missing. The app rejects text,
positive/negative infinity, and values outside the `float32` range. Clean the
source column and upload it again.

## A timestamp is invalid or duplicated

Use one parseable value per row. Mixed timezone-aware timestamps are normalized
to UTC. Unsorted timestamps are accepted and sorted, but duplicate timestamps
are rejected.

## A bulk group has a missing identifier

Group-ID columns are for stable series identity, not model inputs. Select only
columns that are present on every row and are not the timestamp, a target, or a
covariate. The Explorer validates duplicates within each group, so repeated
dates across different stores are valid.

## Known-future covariates are missing

Append one row for every forecast step and populate every selected
past-and-future covariate. Leave targets empty in these rows. See
[Prepare data](how-to/prepare-data.md).

## A local or offline checkpoint cannot load

Offline mode uses only a local file or an already cached Hub snapshot. Check
the selected repository/revision or path, then retry with network access if a
download is allowed. A local folder requires `config.json` and
`model.safetensors`; every selected checkpoint must match TimesFM-3 parameter
names, shapes, and q0.1 through q0.9 quantiles.

The Explorer deliberately does not silently switch to another model. See the
[Explorer handbook](how-to/timesfm3-explorer-handbook.md#9-choose-and-reproduce-a-checkpoint).

## Independent mode rejects covariates

Independent univariate mode does not use cross-series or covariate information.
Remove the covariate roles or switch to joint multivariate mode.

## More than 32 variates are selected

Reduce the selected targets/covariates or enable benchmark chunking in advanced
settings. Chunking uses deterministic seed `42`, can subsample covariates, and
processes targets in groups.

## A full test run fails on a CUDA machine

The review performed on 2026-09-02 records two TimesFM 2.5 CPU/CUDA placement
interactions in the full suite: the force-flip invariance test and the
model-loading forward check. Focused TimesFM-3 and explorer tests were not
implicated in that run. See [REVIEW.md](../REVIEW.md) for its validation counts.

## The app lost previous runs

The app stores the newest 25 untracked derived runs in `data/timesfm.duckdb`. Check that
the file is readable and the project directory is writable. If DuckDB cannot
open or update it, the app warns and continues with browser-session history.
Tracked forecast vintages are protected until tracking is stopped. Original
uploads and input history are not stored, so restored runs show only forecast
outputs and metrics. Download ZIP bundles for portable archives.

## The CSV helper and explorer behave differently

They target different model generations. The explorer uses TimesFM 3; the CSV
helper uses TimesFM 2.5. Use the matching
[version-specific guide](explanation/licensing-and-versions.md).
