# How to Prepare Data

The Explorer accepts UTF-8 CSV, Parquet, and `.pq` files. It decodes uploads in
the local Streamlit process and does not retain the original files in DuckDB.

## Prepare a wide table

Use one row per time step and one numeric series per column.

```csv
date,sales,temperature,promotion
2026-01-01,101,20.1,0
2026-01-02,105,20.4,1
2026-01-03,103,20.2,0
```

In **Prepare**, assign `date` as the timestamp and give every numeric input one
role:

| Role | Use it for | Required forecast values |
|---|---|---|
| Target | A series to predict, such as `sales` | Historical values; blank future rows |
| Past-only covariate | A value only observed after the fact, such as measured temperature | Context only |
| Past-and-future covariate | A value known before prediction, such as a scheduled promotion | Context and every future step |

The timestamp, targets, covariates, and group identifiers are separate roles.
Do not select a target or covariate as a group identifier.

## Supply known-future rows

For a two-step horizon, append two rows with empty targets and complete
past-and-future values:

```csv
date,sales,temperature,promotion
2026-01-01,101,20.1,0
2026-01-02,105,20.4,1
2026-01-03,103,20.2,0
2026-01-04,,,1
2026-01-05,,,0
```

Past-only columns may be blank in appended rows. A selected past-and-future
covariate cannot have a gap in the requested future horizon.

## Prepare long-format bulk data

For many related series, upload a long table and choose stable group-ID columns.

```csv
date,store_id,sales,promotion
2026-01-01,store_a,101,0
2026-01-02,store_a,105,1
2026-01-01,store_b,84,0
2026-01-02,store_b,89,0
```

Choose `store_id` as a group ID and `sales` as a target. The Explorer creates a
separate dataset for each group, then runs them as a batch. A group identifier
must be present and finite for every row. Timestamp uniqueness is checked within
each group, so two stores can share the same date.

Keep the source name and group values stable when you later upload actuals for
forecast tracking. They form the series identity used to associate new data
with issued predictions.

## Generate calendar features

Use **Optional preparation** to generate weekday, month, holiday, and named
event features. The Explorer adds them as numeric past-and-future covariates
for historical and generated future rows.

- Weekday uses Monday `0` through Sunday `6`; month uses `1` through `12`.
- Holidays use the selected country and optional subdivision, including
  observed holiday dates.
- A named event is `1` on every date in its inclusive start/end range and `0`
  otherwise.
- When future timestamps cannot be inferred, provide a frequency before adding
  future rows.

Calendar/event features must have been knowable at each historical forecast
origin. They count toward the 32-input model limit.

## Review readiness before forecasting

The readiness preview reports per-series blocking issues and warnings:

- missing or duplicate group IDs
- invalid, duplicate, irregular, or gapped timestamps
- missing, constant, or insufficient target history
- missing known-future values
- interpolation and leading-trim behavior inside the model context
- requests that need explicit benchmark chunking above 32 inputs

The preview never changes uploaded targets. Internal missing context values can
be linearly interpolated by TimesFM preprocessing; held-out actuals remain
missing and unscored. Fix blocking issues in the source before forecasting.

## Prepare a holdout evaluation

Holdout mode needs no appended rows. Supply observed target values and choose a
horizon smaller than the available history. The Explorer ends context before
the final horizon, forecasts it, and compares the result with the untouched
actual values.

## Use multiple files

Each file becomes one batch item. All files need compatible selected column
names and model shapes because the Explorer applies one mapping and forecast
configuration to the batch. A parsing or validation failure rejects the entire
batch rather than running a partial prefix.

## Limits and validation

- 50 MiB compressed per file and 200 MiB across uploads
- 256 MiB decoded per file and 512 MiB across decoded dataframes
- unique column names after conversion to strings
- timestamps may be unsorted and are sorted; they must parse and be unique per
  series
- selected model values must be finite numbers or missing values representable
  as `float32`
- every target needs at least two context observations

Start with context length `512`. Increase it only when the series needs more
seasonal history and available memory permits it. For the complete operational
flow, see the [Explorer handbook](timesfm3-explorer-handbook.md).
