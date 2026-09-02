# How to Prepare Data

Use a wide CSV or Parquet table: one time step per row and one series per
column. Files are decoded in memory and are not written to a server-side upload
directory.

## Basic forecast data

```csv
date,sales
2026-01-01,101
2026-01-02,105
2026-01-03,103
```

Choose `date` as the timestamp and `sales` as the target. A timestamp is
optional; without one, the app uses integer steps.

## Add covariates

```csv
date,sales,temperature,promotion
2026-01-01,101,20.1,0
2026-01-02,105,20.4,1
2026-01-03,103,20.2,0
```

- **Target:** the series to forecast, such as `sales`.
- **Past-only covariate:** known only for historical rows, such as measured
  temperature.
- **Past-and-future covariate:** known for history and the complete forecast
  horizon, such as a scheduled promotion.

Each numeric column can have only one role. The timestamp cannot also be a
model input.

## Supply known-future rows

For a forecast horizon of two rows, append two rows with empty targets and
complete known-future values:

```csv
date,sales,temperature,promotion
2026-01-01,101,20.1,0
2026-01-02,105,20.4,1
2026-01-03,103,20.2,0
2026-01-04,,,1
2026-01-05,,,0
```

Do not leave gaps in a selected past-and-future covariate over the future
horizon. Past-only columns may be empty in appended rows.

## Prepare holdout evaluation

Holdout mode needs no appended future rows. Supply observed target values and
choose a horizon smaller than the number of observations. The app removes the
last `horizon` observations from the model context, forecasts them, and compares
the result with the held-out values.

## Use multiple files

Each uploaded file becomes one batch item. All files must contain the selected
column names and compatible model shapes. The app applies one shared column
mapping and forecast configuration to the entire batch.

If any file fails parsing or validation, the whole upload batch is rejected;
the app does not run a partial prefix.

## Validation rules

- Supported types: UTF-8 CSV, Parquet, and `.pq`.
- Compressed size: 50 MiB per file and 200 MiB combined.
- Decoded memory: 256 MiB per file and 512 MiB combined.
- Column names must be unique after conversion to strings.
- Timestamps must parse, be unique, and can be unsorted; the app sorts them.
- Each target needs at least two context values.
- Numeric values must be finite or missing and must fit in `float32`.
- Internal missing values are passed through TimesFM's linear interpolation.

## Choose an appropriate context

The app keeps at most the configured context length immediately before the
forecast boundary. A longer context uses more memory. Start with `512`; increase
it only when the series needs longer seasonal history.

For error-specific advice, see [Troubleshooting](../troubleshooting.md).
