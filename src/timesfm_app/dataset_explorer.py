"""Read-only EDA over an immutable uploaded dataset; no model is loaded."""

import numpy as np
import pandas as pd


def profile(frame: pd.DataFrame) -> dict:
  """Compute full-data summaries, bounding the quadratic correlation output."""
  numeric = frame.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan)
  columns = []
  for name in frame.columns:
    series = frame[name]
    missing = int(series.isna().sum())
    columns.append(
      {
        "column": name,
        "dtype": str(series.dtype),
        "non_null": int(series.notna().sum()),
        "missing": missing,
        "missing_percent": 100 * missing / len(frame) if len(frame) else 0,
        "unique": int(series.nunique()),
        "infinite": int(np.isinf(series.dropna()).sum()) if name in numeric else 0,
      }
    )
  correlation_columns = list(numeric.columns[:32])
  correlation = numeric[correlation_columns].corr(min_periods=2)
  row_label = "column"
  while row_label in correlation_columns:
    row_label += "_"
  return {
    "rows": len(frame),
    "columns": columns,
    "numeric_columns": list(numeric.columns),
    "missing_cells": int(frame.isna().sum().sum()),
    "duplicate_rows": int(frame.duplicated().sum()),
    "memory_bytes": int(frame.memory_usage(deep=True).sum()),
    "statistics": (
      numeric.describe().T.rename_axis("column").reset_index().to_dict("records")
      if len(numeric.columns)
      else []
    ),
    "correlation_columns": correlation_columns,
    "correlations": [
      {row_label: name, **correlation.loc[name].to_dict()}
      for name in correlation_columns
    ],
  }


def plot_data(frame: pd.DataFrame, column: str, x: str | None = None) -> dict:
  """Aggregate distributions on all rows and sample paired chart points evenly."""
  series = frame[column]
  numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
    series
  )
  values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
  finite = values.dropna()
  histogram = []
  if numeric and len(finite):
    counts, edges = np.histogram(finite.to_numpy(dtype=float), bins=20)
    histogram = [
      {"lower": edges[i], "upper": edges[i + 1], "count": int(count)}
      for i, count in enumerate(counts)
    ]
  positions = np.linspace(0, len(frame) - 1, min(2000, len(frame)), dtype=int)
  points = (
    [
      {"x": frame[x].iloc[i] if x is not None else int(i), "y": values.iloc[i]}
      for i in positions
    ]
    if numeric
    else []
  )
  categories = series.value_counts(dropna=True).head(20)
  return {
    "numeric": numeric,
    "total": len(frame),
    "sampled": len(frame) > 2000,
    "points": points,
    "histogram": histogram,
    "categories": [
      {"value": str(value), "count": int(count)} for value, count in categories.items()
    ],
    "non_null": int(series.notna().sum()),
  }
