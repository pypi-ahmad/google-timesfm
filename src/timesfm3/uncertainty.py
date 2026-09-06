"""Nominal forecast bands and descriptive, unfitted coverage summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd


def interval_bands(frame: pd.DataFrame) -> pd.DataFrame:
  """Return central quantile bands without modifying model predictions.

  Args:
    frame: Forecast rows containing some or all q0.1 through q0.9 columns.

  Returns:
    One row per available 20%, 40%, 60%, or 80% central band. Crossed or
    non-finite bounds are retained and marked invalid for transparent display.
  """
  quantiles = [f"q{value / 10:.1f}" for value in range(1, 10)]
  present = [column for column in quantiles if column in frame]
  if not present or frame.empty:
    return pd.DataFrame()
  current = frame.loc[frame[present].notna().any(axis=1)]
  identity = [
    column
    for column in (
      "variant",
      "dataset",
      "target",
      "origin",
      "step",
      "timestamp",
      "point",
      "actual",
      "scored",
    )
    if column in current
  ]
  values = current[present].to_numpy(dtype=float)
  crossed = (np.diff(values, axis=1) < 0).any(axis=1)
  bands = []
  for nominal, lower, upper in (
    (20, "q0.4", "q0.6"),
    (40, "q0.3", "q0.7"),
    (60, "q0.2", "q0.8"),
    (80, "q0.1", "q0.9"),
  ):
    if lower not in current or upper not in current:
      continue
    band = current[identity].copy()
    band["nominal_coverage_percent"] = nominal
    band["lower"] = current[lower]
    band["upper"] = current[upper]
    band["crossed"] = crossed
    band["valid_bounds"] = (
      np.isfinite(band["lower"]) & np.isfinite(band["upper"]) & ~crossed
    )
    bands.append(band)
  return pd.concat(bands, ignore_index=True) if bands else pd.DataFrame()


def calibration_table(frame: pd.DataFrame) -> pd.DataFrame:
  """Summarize empirical coverage by target and horizon without fitting bands.

  Args:
    frame: Forecast rows with actual values and quantile predictions.

  Returns:
    Coverage, width, missing-actual, and invalid-bound counts for every
    available central interval. An empty frame means the required values are
    unavailable.
  """
  if "actual" not in frame:
    return pd.DataFrame()
  bands = interval_bands(frame)
  if bands.empty:
    return pd.DataFrame()
  groups = [
    column for column in ("variant", "dataset", "target", "step") if column in bands
  ] + ["nominal_coverage_percent"]
  observed = np.isfinite(bands["actual"])
  bands["observations"] = observed & bands["valid_bounds"]
  bands["missing_actuals"] = ~observed
  bands["invalid_bounds"] = ~bands["valid_bounds"]
  bands["covered"] = bands["observations"] & bands["actual"].between(
    bands["lower"], bands["upper"]
  )
  bands["width"] = (bands["upper"] - bands["lower"]).where(bands["observations"])
  result = (
    bands.groupby(groups, sort=False, dropna=False)
    .agg(
      observations=("observations", "sum"),
      missing_actuals=("missing_actuals", "sum"),
      invalid_bounds=("invalid_bounds", "sum"),
      crossings=("crossed", "sum"),
      covered=("covered", "sum"),
      mean_width=("width", "mean"),
    )
    .reset_index()
  )
  result["observed_coverage_percent"] = (
    100 * result.pop("covered") / result["observations"].replace(0, np.nan)
  )
  return result[
    groups
    + [
      "observations",
      "missing_actuals",
      "invalid_bounds",
      "crossings",
      "observed_coverage_percent",
      "mean_width",
    ]
  ]
