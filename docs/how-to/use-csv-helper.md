# How to Use the TimesFM 2.5 CSV Helper

> [!NOTE]
> This compatibility helper uses TimesFM 2.5. It is separate from the
> TimesFM-3 Streamlit explorer and `timesfm3` API.

## Check the environment

```powershell
uv run python timesfm-forecasting/scripts/check_system.py
```

The helper runs this preflight automatically unless `--skip-check` is used.

## Forecast numeric columns

```powershell
uv run python timesfm-forecasting/scripts/forecast_csv.py data.csv `
  --horizon 24 `
  --date-col date `
  --value-cols sales,revenue `
  --output forecasts.csv
```

If `--value-cols` is omitted, all numeric columns except the date column are
used. If `--date-col` is omitted, the helper attempts to find a date-like
column.

## Write JSON

```powershell
uv run python timesfm-forecasting/scripts/forecast_csv.py data.csv `
  --horizon 12 `
  --output forecasts.json
```

The format is inferred from the output extension or selected explicitly with
`--format csv` or `--format json`.

## Output columns

CSV output contains `series`, `step`, `forecast`, `q10`, `q20`, `q50`, `q80`,
and `q90`. When a regular input frequency is inferred, it also includes `date`.

Compatibility aliases such as `lower_90`, `median`, and `upper_90` remain in
the output but are deprecated. Prefer the explicit `qXX` names.

## Missing values

Missing edges are trimmed. Internal missing values remain in place so later
observations do not shift to earlier time steps; TimesFM handles them during
forecast preparation. A column with no finite observations is rejected.

## Available options

```powershell
uv run python timesfm-forecasting/scripts/forecast_csv.py --help
```

Use `--batch-size` to override the preflight recommendation. Avoid
`--skip-check` unless the environment has already been verified.
