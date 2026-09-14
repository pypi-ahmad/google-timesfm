# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Fixed-origin TimesFM-3 experiments and derived analysis results.

Experiment engine for the analysis workbench: given uploaded datasets and
a column `DatasetMapping` (from `explorer.py`), builds a validated,
leak-free schedule of historical forecast windows (`prepare_analysis`),
runs it through a `BatchPredictor` (`run_analysis`), and derives
comparison/diagnostic tables from the results (anomaly flags, scenario
deltas, configuration rankings, calibration). Supports several
`AnalysisKind`s (anomaly detection, scenario what-ifs, backtests,
joint-vs-independent multivariate comparison, covariate usefulness,
settings/baseline comparisons) that mostly differ in how
`prepare_analysis` builds its `variants` list.

`AnalysisArtifact` intentionally holds only derived results (predictions,
metrics, a JSON-serializable manifest) and never the original uploaded
data or full historical context arrays -- see `analysis_zip`, which is
the only export path and is built from the artifact alone. This module
depends on `explorer.py` for dataset/prediction primitives and
`uncertainty.py` for calibration; see `analysis_ui.py` for how these are
surfaced to a user.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import time
import uuid
import zipfile
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .explorer import (
  CHECKPOINT_ID,
  MAX_VARIATES,
  BatchPredictor,
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  PreparedBatch,
  UploadedDataset,
  _coerce_numeric,
  _csv_safe,
  _time_axis,
  _validate_mapping,
  evaluation_metrics,
  forecast_table,
  prepare_batch,
  repository_revision,
  run_forecast,
  validate_upload_total,
)
from .uncertainty import calibration_table

AnalysisKind = Literal[
  "anomaly",
  "scenario",
  "backtest",
  "joint_independent",
  "covariates",
  "settings",
  "baselines",
]
MAX_PREDICTION_ROWS = 250_000


@dataclasses.dataclass(frozen=True)
class AnalysisSettings:
  """Historical window controls and the experiment to run."""

  kind: AnalysisKind
  windows: int = 5
  stride: int | None = None
  selected_covariates: tuple[str, ...] = ()
  seasonal_period: int = 1


@dataclasses.dataclass(frozen=True)
class ExperimentConfiguration:
  """One named set of inference controls compared on shared holdouts."""

  name: str
  settings: ForecastSettings


@dataclasses.dataclass(frozen=True)
class ScenarioEdit:
  """One authored override at a sorted, zero-based uploaded row."""

  dataset: str
  row: int
  covariate: str
  value: float


@dataclasses.dataclass(frozen=True)
class Scenario:
  """Sparse future edits tied to immutable baseline inputs."""

  name: str
  fingerprint: str
  overrides: tuple[ScenarioEdit, ...] = ()


@dataclasses.dataclass(frozen=True)
class AnalysisArtifact:
  """Derived results; no original uploads or historical context arrays.

  This exclusion is deliberate data minimization: uploaded datasets may
  contain sensitive business data, so anything exported from an analysis
  (see `analysis_zip`) is built only from this artifact's already-derived
  predictions/metrics/manifest, never from the raw uploads.
  """

  analysis_id: str
  created_at: str
  kind: str
  predictions: pd.DataFrame
  metrics: pd.DataFrame
  manifest: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ForecastTask:
  """Small window description sharing a session-only dataset snapshot."""

  variant: str
  dataset: UploadedDataset
  mapping: DatasetMapping
  settings: ForecastSettings
  origin: int
  overrides: tuple[ScenarioEdit, ...] = ()
  method: Literal["model", "last_value", "seasonal_naive"] = "model"
  seasonal_period: int = 1


@dataclasses.dataclass(frozen=True)
class PreparedAnalysis:
  """Validated schedule, ready for model acquisition and sequential execution."""

  tasks: tuple[ForecastTask, ...]
  manifest: dict[str, Any]
  prediction_rows: int

  @property
  def forecast_count(self) -> int:
    return len(self.tasks)


def analysis_fingerprint(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  settings: ForecastSettings,
) -> str:
  """Bind editors to their upload identities, column roles, and horizon.

  Used to detect when a scenario's edits were authored against data,
  mapping, or horizon that has since changed (see `scenario_from_edits`'s
  `fingerprint` and `prepare_analysis`'s check against stale scenarios),
  so edits don't silently get replayed against different underlying data.
  """
  payload = {
    "datasets": [(item.dataset_id, item.sha256) for item in datasets],
    "preparation": {
      item.dataset_id: item.frame.attrs.get("preparation", {}) for item in datasets
    },
    "mapping": dataclasses.asdict(mapping),
    "horizon": settings.horizon,
  }
  return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _snapshots(
  datasets: Sequence[UploadedDataset], mapping: DatasetMapping
) -> tuple[UploadedDataset, ...]:
  """Freezes a validated, timestamp-normalized copy of each dataset.

  The returned tuple is what the rest of an analysis run operates on, so
  later mutation of the caller's original `datasets` (e.g. from a UI
  session) cannot retroactively change an in-flight or completed analysis.
  """
  validate_upload_total(datasets)
  if not datasets:
    raise ExplorerError("Upload at least one dataset.")
  if len({item.dataset_id for item in datasets}) != len(datasets):
    raise ExplorerError("Dataset identifiers must be unique.")
  result = []
  for item in datasets:
    _validate_mapping(item.frame, mapping)
    frame, axis, _ = _time_axis(item.frame, mapping.timestamp)
    frame = frame.copy()
    if mapping.timestamp is not None:
      frame[mapping.timestamp] = axis
    result.append(dataclasses.replace(item, frame=frame))
  return tuple(result)


def _observed_end(dataset: UploadedDataset, mapping: DatasetMapping) -> int:
  """Row index one past the last row with any observed target value.

  Exclusive bound (the "+1"): this is used directly as a forecast
  `origin`/cutoff elsewhere, i.e. history strictly before this index is
  observed and this index itself is the first unobserved (forecastable)
  row.
  """
  targets = _coerce_numeric(dataset.frame, mapping.targets)
  observed = np.flatnonzero(targets.notna().any(axis=1).to_numpy())
  if not len(observed):
    raise ExplorerError(f"{dataset.dataset_id} has no observed target values.")
  return int(observed[-1]) + 1


def scenario_template(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  settings: ForecastSettings,
) -> pd.DataFrame:
  """Build a long-form editor containing only known-future covariate cells."""
  if not mapping.past_future:
    raise ExplorerError("Scenarios require at least one known-future covariate.")
  forecast_settings = dataclasses.replace(
    settings, task="forecast", mode="multivariate"
  )
  rows = []
  for dataset in _snapshots(datasets, mapping):
    origin = _observed_end(dataset, mapping)
    batch = prepare_batch([dataset], mapping, forecast_settings, cutoffs=[origin])
    for step, timestamp in enumerate(batch.series[0].future_time):
      for column in mapping.past_future:
        rows.append(
          {
            "dataset": dataset.dataset_id,
            "row": origin + step,
            "timestamp": timestamp,
            "covariate": column,
            "value": float(dataset.frame.iloc[origin + step][column]),
          }
        )
  return pd.DataFrame(rows)


def scenario_from_edits(
  name: str, baseline: pd.DataFrame, edited: pd.DataFrame, fingerprint: str
) -> Scenario:
  """Validate a native editor result and retain only authored cell changes."""
  name = name.strip()
  if not name or name.casefold() == "baseline":
    raise ExplorerError("Give each scenario a name other than 'baseline'.")
  identifiers = ["dataset", "row", "timestamp", "covariate"]
  if (
    list(edited.columns) != list(baseline.columns)
    or len(edited) != len(baseline)
    or not edited[identifiers]
    .reset_index(drop=True)
    .equals(baseline[identifiers].reset_index(drop=True))
  ):
    raise ExplorerError("Only future covariate values can be edited.")
  values = pd.to_numeric(edited["value"], errors="coerce").to_numpy(dtype=float)
  if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
    raise ExplorerError("Scenario values must be finite numbers within float32 range.")
  changes = []
  for index, row in enumerate(baseline.to_dict(orient="records")):
    if values[index] != float(row["value"]):
      changes.append(
        ScenarioEdit(
          str(row["dataset"]),
          int(row["row"]),
          str(row["covariate"]),
          float(values[index]),
        )
      )
  return Scenario(name, fingerprint, tuple(changes))


def _task_batch(task: ForecastTask) -> PreparedBatch:
  """Slices exactly the rows a task's context+horizon window needs.

  `start`/`stop` bound a leak-free window: history ends at `task.origin`
  (context is `context_length` rows before it, clamped to 0) and the
  horizon extends `settings.horizon` rows past it -- rows outside this
  slice are never loaded, so a task can never see data beyond its own
  origin+horizon. `edit.row` is an absolute row index into the original
  frame, so it's rebased by `- start` to index into this window's slice.
  """
  start = max(0, task.origin - task.settings.context_length)
  stop = task.origin + task.settings.horizon
  frame = task.dataset.frame.iloc[start:stop].copy().reset_index(drop=True)
  for edit in task.overrides:
    frame.loc[edit.row - start, edit.covariate] = edit.value
  dataset = dataclasses.replace(task.dataset, frame=frame)
  batch = prepare_batch(
    [dataset], task.mapping, task.settings, cutoffs=[task.origin - start]
  )
  if task.mapping.timestamp is None:
    series = batch.series[0]
    batch = dataclasses.replace(
      batch,
      series=(
        dataclasses.replace(
          series,
          history_time=tuple(value + start for value in series.history_time),
          future_time=tuple(value + start for value in series.future_time),
        ),
      ),
    )
  return batch


def prepare_analysis(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  forecast_settings: ForecastSettings,
  analysis_settings: AnalysisSettings,
  scenarios: Sequence[Scenario] = (),
  configurations: Sequence[ExperimentConfiguration] = (),
) -> PreparedAnalysis:
  """Build a leak-free, fully validated historical analysis schedule.

  Every task ends context before its cutoff and shares the requested comparison
  schedule. Raises ``ExplorerError`` before model acquisition for invalid
  windows, unavailable covariates, excessive output, or incompatible variants.
  """
  kind = analysis_settings.kind
  if kind not in {
    "anomaly",
    "scenario",
    "backtest",
    "joint_independent",
    "covariates",
    "settings",
    "baselines",
  }:
    raise ExplorerError("Unknown analysis kind.")
  forecast_settings.validate()
  maximum_windows = 1_000 if kind == "anomaly" else 100
  if (
    not isinstance(analysis_settings.windows, int)
    or not 1 <= analysis_settings.windows <= maximum_windows
  ):
    raise ExplorerError(f"Window count must be between 1 and {maximum_windows}.")
  # Anomaly detection is always one-step-ahead: horizon and stride are
  # forced to 1 regardless of the requested forecast horizon/stride, so
  # every historical row gets its own single-step forecast to compare
  # against, rather than the requested (possibly longer) horizon.
  horizon = 1 if kind == "anomaly" else forecast_settings.horizon
  stride = analysis_settings.stride if analysis_settings.stride is not None else horizon
  if not isinstance(stride, int) or stride < 1:
    raise ExplorerError("Window stride must be a positive integer.")
  if kind == "anomaly":
    stride = 1
  settings = dataclasses.replace(
    forecast_settings,
    task="forecast" if kind == "scenario" else "holdout",
    horizon=horizon,
    allow_benchmark_chunking=False,
    return_quantiles=True
    if kind in {"anomaly", "settings", "baselines"}
    else forecast_settings.return_quantiles,
    sort_quantiles=True if kind == "anomaly" else forecast_settings.sort_quantiles,
  )
  effective_mapping = mapping
  if kind == "joint_independent":
    if len(mapping.targets) < 2:
      raise ExplorerError("Joint versus independent comparison requires two targets.")
    # Strip covariates for this comparison: the point is to isolate the
    # effect of joint (multivariate) vs. independent (univariate)
    # modeling of the targets themselves, so covariates are excluded to
    # avoid conflating that with covariate usefulness.
    effective_mapping = dataclasses.replace(mapping, past_only=(), past_future=())
  if (
    len(
      effective_mapping.targets
      + effective_mapping.past_only
      + effective_mapping.past_future
    )
    > MAX_VARIATES
  ):
    raise ExplorerError(
      "Analyses support at most 32 combined variates without chunking."
    )
  snapshots = _snapshots(datasets, effective_mapping)
  fingerprint = analysis_fingerprint(datasets, mapping, forecast_settings)
  variants: list[tuple[str, DatasetMapping, ForecastSettings]] = []
  if kind == "settings":
    if not 2 <= len(configurations) <= 8:
      raise ExplorerError("Provide between two and eight forecast configurations.")
    names = [item.name.strip() for item in configurations]
    if any(
      not name or name.casefold() in {"last_value", "seasonal_naive"} for name in names
    ) or len({name.casefold() for name in names}) != len(names):
      raise ExplorerError(
        "Configuration names must be distinct and cannot use baseline names."
      )
    fixed = ("horizon", "mode", "batch_size", "allow_benchmark_chunking")
    for name, configuration in zip(names, configurations, strict=True):
      current = configuration.settings
      current.validate()
      if any(
        getattr(current, field) != getattr(forecast_settings, field) for field in fixed
      ):
        raise ExplorerError(
          "Configurations must share horizon, mode, batch size, and chunking settings."
        )
      if (
        isinstance(current.context_length, bool)
        or not isinstance(current.context_length, int)
        or current.context_length < 2
      ):
        raise ExplorerError(
          "Configuration context length must be an integer from 2 to 15,360."
        )
      if any(
        not isinstance(getattr(current, field), bool)
        for field in (
          "use_symmetric_averaging",
          "make_positive",
          "sort_quantiles",
          "use_znorm",
        )
      ):
        raise ExplorerError("Configuration inference switches must be true or false.")
      variants.append(
        (
          name,
          mapping,
          dataclasses.replace(
            current,
            task="holdout",
            return_quantiles=True,
            allow_benchmark_chunking=False,
          ),
        )
      )
  elif kind == "joint_independent":
    variants = [
      ("joint", effective_mapping, dataclasses.replace(settings, mode="multivariate")),
      (
        "independent",
        effective_mapping,
        dataclasses.replace(settings, mode="univariate"),
      ),
    ]
  elif kind == "covariates":
    covariates = mapping.past_only + mapping.past_future
    if not covariates:
      raise ExplorerError("Select at least one covariate for usefulness analysis.")
    selected = analysis_settings.selected_covariates or covariates
    if len(set(selected)) != len(selected) or not set(selected).issubset(covariates):
      raise ExplorerError("Select distinct mapped covariates for removal.")
    removals = [("all", ()), ("none", covariates), ("without_selected", selected)]
    removals.extend((f"without:{column}", (column,)) for column in selected)
    seen = set()
    for label, removed in removals:
      current = dataclasses.replace(
        mapping,
        past_only=tuple(c for c in mapping.past_only if c not in removed),
        past_future=tuple(c for c in mapping.past_future if c not in removed),
      )
      key = (current.past_only, current.past_future)
      if key not in seen:
        seen.add(key)
        variants.append(
          (label, current, dataclasses.replace(settings, mode="multivariate"))
        )
  elif kind == "scenario":
    if not mapping.past_future or not 1 <= len(scenarios) <= 3:
      raise ExplorerError(
        "Provide one to three scenarios with known-future covariates."
      )
    names = [scenario.name.strip() for scenario in scenarios]
    if any(not n or n.casefold() == "baseline" for n in names) or len(
      {n.casefold() for n in names}
    ) != len(names):
      raise ExplorerError("Scenario names must be distinct and cannot be 'baseline'.")
    if any(scenario.fingerprint != fingerprint for scenario in scenarios):
      raise ExplorerError(
        "Scenario inputs changed; reset edits to the current baseline."
      )
    settings = dataclasses.replace(settings, mode="multivariate")
    variants = [("baseline", mapping, settings)] + [
      (n, mapping, settings) for n in names
    ]
  else:
    variants = [("baseline", mapping, settings)]

  warmup = 0
  if kind in {"settings", "baselines"}:
    period = analysis_settings.seasonal_period
    if isinstance(period, bool) or not isinstance(period, int) or period < 1:
      raise ExplorerError("Seasonal period must be a positive integer number of rows.")
    # The naive baselines need at least `period` rows of history (to read
    # back one full seasonal cycle) and at least as much as any model
    # variant's own context_length, so every window's history is deep
    # enough for every variant being compared, including the baselines.
    warmup = max(period, *(current.context_length for _, _, current in variants))
    if warmup > 15_360:
      raise ExplorerError("Seasonal period must not exceed 15,360 rows.")
    baseline_settings = dataclasses.replace(
      settings, context_length=warmup, return_quantiles=False
    )
    baseline_mapping = dataclasses.replace(mapping, past_only=(), past_future=())
    variants.extend(
      (name, baseline_mapping, baseline_settings)
      for name in ("last_value", "seasonal_naive")
    )

  window_count = 1 if kind == "scenario" else analysis_settings.windows
  # Upper-bound total output size before running anything (predictor calls
  # are the expensive part) so an oversized request fails fast instead of
  # exhausting memory/time partway through `run_analysis`.
  prediction_rows = (
    len(snapshots) * window_count * len(variants) * len(mapping.targets) * horizon
  )
  if prediction_rows > MAX_PREDICTION_ROWS:
    raise ExplorerError(
      "Analysis exceeds 250,000 prediction rows; reduce windows, targets, or horizon."
    )
  origins = {
    item.dataset_id: _observed_end(item, effective_mapping) for item in snapshots
  }
  edits_by_name = {scenario.name.strip(): scenario.overrides for scenario in scenarios}
  for scenario in scenarios:
    seen_edits = set()
    for edit in scenario.overrides:
      # An edit must: target a dataset in this analysis, land within the
      # forecast horizon starting at that dataset's observed-end origin
      # (edits to history or beyond the horizon aren't meaningful here),
      # target a known-future ("past_future") covariate specifically
      # (not a target or past-only covariate), carry a finite in-range
      # value, and be unique per (dataset, row, covariate).
      key = (edit.dataset, edit.row, edit.covariate)
      if (
        edit.dataset not in origins
        or not isinstance(edit.row, int)
        or not origins[edit.dataset] <= edit.row < origins[edit.dataset] + horizon
        or edit.covariate not in mapping.past_future
        or not np.isfinite(edit.value)
        or abs(edit.value) > np.finfo(np.float32).max
        or key in seen_edits
      ):
        raise ExplorerError(
          "Scenario edits must uniquely identify finite future covariate values."
        )
      seen_edits.add(key)

  tasks = []
  for dataset in snapshots:
    end = origins[dataset.dataset_id]
    # Rolling windows walk backward from the most recent fully-observed
    # origin (`end - horizon`) in steps of `stride`, then are listed in
    # forward (oldest-to-newest) order via `reversed(range(window_count))`
    # so window index 0 is the oldest, not the most recent.
    cutoffs = (
      [end]
      if kind == "scenario"
      else [end - horizon - index * stride for index in reversed(range(window_count))]
    )
    for origin in cutoffs:
      if warmup and origin < warmup:
        raise ExplorerError(
          f"{dataset.dataset_id} needs {warmup} history rows before every comparison window; reduce context, seasonal period, or windows."
        )
      for label, current_mapping, current_settings in variants:
        task = ForecastTask(
          label,
          dataset,
          current_mapping,
          current_settings,
          origin,
          tuple(
            e for e in edits_by_name.get(label, ()) if e.dataset == dataset.dataset_id
          ),
          method=label
          if label in {"last_value", "seasonal_naive"} and warmup
          else "model",
          seasonal_period=analysis_settings.seasonal_period,
        )
        # Validate sequentially without retaining a context copy for every window.
        _task_batch(task)
        tasks.append(task)
  manifest = {
    "schema_version": 2,
    "kind": kind,
    "checkpoint": CHECKPOINT_ID,
    "settings": dataclasses.asdict(settings),
    "analysis_settings": dataclasses.asdict(analysis_settings),
    "mapping": dataclasses.asdict(mapping),
    "fingerprint": fingerprint,
    "datasets": [
      {
        "dataset_id": item.dataset_id,
        "sha256": item.sha256,
        "preparation": item.frame.attrs.get("preparation", {}),
      }
      for item in snapshots
    ],
    "variants": [
      {
        "name": label,
        "mapping": dataclasses.asdict(m),
        "mode": s.mode,
        "settings": dataclasses.asdict(s),
      }
      for label, m, s in variants
    ],
    "windows": [
      {
        "dataset": t.dataset.dataset_id,
        "variant": t.variant,
        "origin": t.origin,
        "horizon": t.settings.horizon,
      }
      for t in tasks
    ],
    "scenarios": [dataclasses.asdict(s) for s in scenarios],
    "known_future_policy": "Only covariates known at each forecast origin may be assigned this role.",
    "license": "timesfm-non-commercial-license-v1.0",
  }
  return PreparedAnalysis(tuple(tasks), manifest, prediction_rows)


def analysis_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
  """Aggregate observed forecast pairs overall, by horizon, and by origin."""
  if "actual" not in predictions:
    return pd.DataFrame()
  rows = []
  base = ["variant", "dataset", "target"]
  for scope, extra in [("overall", []), ("step", ["step"]), ("window", ["origin"])]:
    for _, group in predictions.groupby(base + extra, sort=False):
      record = {column: group[column].iloc[0] for column in base + extra}
      record.update(scope=scope, origin=record.get("origin"), step=record.get("step"))
      scored = group
      if "scored" in group:
        # `scored` (set by matched_predictions) marks rows eligible for a
        # fair cross-variant comparison; rows with an actual but not
        # `scored` are deliberately excluded from metrics here (blanked
        # to NaN) rather than counted as an error, and tracked separately
        # as `excluded` so they're visible without skewing MAE/RMSE.
        scored = group.copy()
        scored.loc[~scored["scored"], "actual"] = np.nan
        record.update(
          excluded=int((group["actual"].notna() & ~group["scored"]).sum()),
          missing_actuals=int(group["actual"].isna().sum()),
        )
      # Drop quantile columns entirely rather than pass all-NaN columns
      # into evaluation_metrics, e.g. for baseline methods that never
      # produce quantiles.
      quantile_columns = [column for column in scored if column.startswith("q0.")]
      if quantile_columns and scored[quantile_columns].isna().all().all():
        scored = scored.drop(columns=quantile_columns)
      values = evaluation_metrics(scored)
      count = int(scored["actual"].notna().sum())
      record.update(observations=count, missing=len(group) - count)
      if values.empty:
        record.update(mae=np.nan, rmse=np.nan, smape_percent=np.nan)
      else:
        record.update(
          values.iloc[0].drop(labels=["dataset", "target", "observations"]).to_dict()
        )
      rows.append(record)
  result = pd.DataFrame(rows)
  result[["origin", "step"]] = result[["origin", "step"]].astype("Int64")
  return result


def _baseline_table(task: ForecastTask, batch: PreparedBatch) -> pd.DataFrame:
  """Forecast raw history without interpolation or future-label access."""
  series = batch.series[0]
  rows = []
  for index, target in enumerate(series.target_names):
    history = series.context[index]
    if task.method == "last_value":
      finite = history[np.isfinite(history)]
      points = np.full(task.settings.horizon, finite[-1] if len(finite) else np.nan)
    else:
      points = np.resize(history[-task.seasonal_period :], task.settings.horizon)
    for step, (timestamp, point) in enumerate(
      zip(series.future_time, points, strict=True)
    ):
      rows.append(
        {
          "dataset": series.dataset_id,
          "target": target,
          "timestamp": timestamp,
          "step": step + 1,
          "point": float(point),
          "actual": float(series.actual[index, step])
          if series.actual is not None
          else np.nan,
        }
      )
  return pd.DataFrame(rows)


def matched_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
  """Mark the same finite forecast pairs as eligible for every method."""
  keys = ["dataset", "target", "origin", "step", "timestamp"]
  if predictions.duplicated(["variant"] + keys).any():
    raise ExplorerError("Comparison predictions contain duplicate forecast keys.")
  result = predictions.copy()
  result["scored"] = np.isfinite(result["actual"]) & np.isfinite(result["point"])
  # A (dataset, target, origin, step, timestamp) key is only "scored" if
  # EVERY variant produced a finite forecast for it (`transform("all")`)
  # AND every variant is actually present for that key
  # (`transform("size").eq(nunique variants)`, guarding against a key
  # that's simply missing from some variant's output rather than merely
  # non-finite) -- otherwise a comparison across variants would be
  # unfair, scoring some variants on an easier/different subset of rows.
  grouped = result.groupby(keys, sort=False, dropna=False)["scored"]
  result["scored"] = grouped.transform("all") & grouped.transform("size").eq(
    result.variant.nunique()
  )
  return result


def configuration_rankings(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Rank MAE within targets and average target ranks with equal weights."""
  if metrics.empty:
    return pd.DataFrame(), pd.DataFrame()
  targets = metrics.loc[metrics.scope.eq("overall")].copy()
  targets["mae_rank"] = targets.groupby(["dataset", "target"])["mae"].rank(
    method="average"
  )
  overall = (
    targets.groupby("variant", sort=False)
    .agg(
      mean_rank=("mae_rank", "mean"),
      scored_targets=("mae_rank", "count"),
    )
    .reset_index()
  )
  overall["excluded_targets"] = (
    targets[["dataset", "target"]].drop_duplicates().shape[0]
    - overall["scored_targets"]
  )
  return targets, overall.sort_values("mean_rank", kind="stable", na_position="last")


def run_analysis(
  predictor: BatchPredictor,
  prepared: PreparedAnalysis,
  progress: Callable[[int, int], None] | None = None,
) -> AnalysisArtifact:
  """Execute a validated schedule and return only complete derived results.

  The predictor is called sequentially so optional progress receives completed
  and total task counts. Any failed task raises instead of publishing a partial
  comparison.
  """
  started = time.perf_counter()
  frames = []
  for index, task in enumerate(prepared.tasks, start=1):
    # Sequential (not batched/parallel) execution is what makes the
    # completed/total progress callback meaningful, and means a task
    # raising immediately aborts the whole run before any partial
    # `AnalysisArtifact` is constructed -- no result is ever published
    # for a schedule that didn't fully complete.
    batch = _task_batch(task)
    if task.method == "model":
      outputs, _ = run_forecast(predictor, batch)
      table = forecast_table(batch, outputs)
      if task.settings.return_quantiles and "q0.1" not in table:
        raise ExplorerError("Requested quantiles were not returned by the predictor.")
    else:
      table = _baseline_table(task, batch)
    table.insert(0, "variant", task.variant)
    table.insert(1, "origin", task.origin)
    frames.append(table)
    if progress is not None:
      progress(index, prepared.forecast_count)
  predictions = pd.concat(frames, ignore_index=True)
  if prepared.manifest["kind"] in {"settings", "baselines"}:
    predictions = matched_predictions(predictions)
  metrics = analysis_metrics(predictions)
  analysis_id = uuid.uuid4().hex[:16]
  created_at = pd.Timestamp.now(tz="UTC").isoformat()
  manifest = dict(prepared.manifest)
  if "scored" in predictions:
    manifest["scoring"] = {
      "policy": "Identical finite forecast keys across every configuration and baseline.",
      "forecast_pairs": len(predictions) // predictions.variant.nunique(),
      "scored_pairs": int(predictions.scored.sum()) // predictions.variant.nunique(),
      "excluded_pairs": int((~predictions.scored).sum())
      // predictions.variant.nunique(),
      "ranking": "Per-target MAE rank; equal-weight mean rank across scored targets.",
    }
  # `model_provenance` is an optional attribute some BatchPredictor
  # implementations carry (duck-typed via getattr rather than an
  # interface method), so predictors without it are still supported.
  model_provenance = getattr(predictor, "model_provenance", None)
  if model_provenance is not None:
    manifest["model_provenance"] = dict(model_provenance)
    selection = model_provenance.get("selection", {})
    if isinstance(selection, dict) and isinstance(selection.get("source"), str):
      manifest["checkpoint"] = selection["source"]
  manifest.update(
    analysis_id=analysis_id,
    created_at=created_at,
    device=str(predictor.device),
    runtime_seconds=time.perf_counter() - started,
    repository_revision=repository_revision(),
  )
  return AnalysisArtifact(
    analysis_id, created_at, manifest["kind"], predictions, metrics, manifest
  )


def anomaly_table(predictions: pd.DataFrame) -> pd.DataFrame:
  """Flag observations strictly outside the q0.1–q0.9 nominal interval.

  Returns direction, median residual, and nonnegative distance beyond the
  closest bound; missing actuals remain explicitly unscored.
  """
  required = {"actual", "point", "q0.1", "q0.9"}
  if not required.issubset(predictions):
    raise ExplorerError(
      "Anomalies require actual observations and q0.1/q0.9 forecasts."
    )
  result = predictions.copy()
  valid = result["actual"].notna()
  lower, upper, actual = result["q0.1"], result["q0.9"], result["actual"]
  if (lower > upper).any():
    raise ExplorerError("Anomaly interval bounds must be sorted.")
  below, above = actual < lower, actual > upper
  # "unscored" (missing actual) is kept distinct from "inside": a missing
  # observation must never be silently treated as a non-anomaly.
  result["direction"] = np.select(
    [~valid, below, above], ["unscored", "below", "above"], default="inside"
  )
  result["residual"] = actual - result["point"]
  # distance is the (nonnegative) amount the actual falls beyond whichever
  # bound it's outside of; 0 when inside the interval.
  result["distance"] = np.maximum(np.maximum(lower - actual, actual - upper), 0)
  # `flagged` uses the nullable "boolean" dtype so unscored rows are
  # pandas NA (missing), not False -- "not flagged" and "cannot be scored"
  # stay distinguishable to consumers.
  result["flagged"] = (below | above).astype("boolean").where(valid, pd.NA)
  return result


def scenario_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
  """Pair each scenario with its immutable baseline prediction."""
  keys = ["dataset", "target", "origin", "step", "timestamp"]
  baseline = predictions.loc[predictions["variant"] == "baseline", keys + ["point"]]
  # validate="many_to_one" fails loudly if the baseline has duplicate
  # keys, catching a data-integrity bug here rather than silently
  # fanning out rows.
  result = predictions.loc[predictions["variant"] != "baseline"].merge(
    baseline.rename(columns={"point": "baseline_point"}),
    on=keys,
    validate="many_to_one",
  )
  result["delta"] = result["point"] - result["baseline_point"]
  # Avoid a divide-by-zero -> inf percent delta when the baseline point is
  # exactly 0; such rows get NaN instead.
  denominator = result["baseline_point"].replace(0, np.nan).abs()
  result["percent_delta"] = 100 * result["delta"] / denominator
  return result


def comparison_table(metrics: pd.DataFrame, kind: str) -> pd.DataFrame:
  """Return matched variant-minus-reference MAE in each target's units."""
  if metrics.empty:
    return pd.DataFrame()
  baseline_name = "joint" if kind == "joint_independent" else "all"
  keys = ["dataset", "target", "scope", "origin", "step"]
  baseline = metrics.loc[
    metrics["variant"] == baseline_name, keys + ["mae", "observations"]
  ]
  result = metrics.loc[metrics["variant"] != baseline_name].merge(
    baseline.rename(
      columns={"mae": "baseline_mae", "observations": "baseline_observations"}
    ),
    on=keys,
    validate="many_to_one",
  )
  # Guard against comparing MAE computed over different numbers of
  # observations per side (e.g. one variant had fewer finite forecasts),
  # which would make the mae_difference not apples-to-apples.
  if not result["observations"].eq(result["baseline_observations"]).all():
    raise ExplorerError("Comparisons require identical scored observations.")
  result["mae_difference"] = result["mae"] - result["baseline_mae"]
  return result


def analysis_zip(artifact: AnalysisArtifact) -> bytes:
  """Export analysis results and provenance without original upload data.

  Builds the zip entirely from `artifact` (predictions/metrics/manifest),
  which by construction never carries raw uploaded rows -- see
  `AnalysisArtifact`.
  """
  tables = {"predictions.csv": artifact.predictions, "metrics.csv": artifact.metrics}
  if artifact.kind == "anomaly":
    tables["anomalies.csv"] = anomaly_table(artifact.predictions)
  elif artifact.kind == "scenario":
    tables["scenario_deltas.csv"] = scenario_deltas(artifact.predictions)
    tables["scenario_edits.csv"] = pd.DataFrame(
      [
        {"scenario": scenario["name"], **edit}
        for scenario in artifact.manifest["scenarios"]
        for edit in scenario["overrides"]
      ],
      columns=["scenario", "dataset", "row", "covariate", "value"],
    )
  elif artifact.kind in {"joint_independent", "covariates"}:
    tables["comparisons.csv"] = comparison_table(artifact.metrics, artifact.kind)
  elif artifact.kind in {"settings", "baselines"}:
    tables["target_rankings.csv"], tables["overall_rankings.csv"] = (
      configuration_rankings(artifact.metrics)
    )
  calibration = calibration_table(artifact.predictions)
  if not calibration.empty:
    tables["calibration.csv"] = calibration
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for name, table in tables.items():
      archive.writestr(name, _csv_safe(table).to_csv(index=False))
    archive.writestr(
      "analysis.json", json.dumps(artifact.manifest, indent=2, default=str)
    )
  return output.getvalue()
