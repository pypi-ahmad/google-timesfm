# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Core data and inference helpers for the TimesFM-3 Streamlit explorer."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import numpy as np
import pandas as pd
import torch

from .evaluator import TimesFM3Evaluator
from .timesfm3_forecaster import ForecastOutput

CHECKPOINT_ID = "google/timesfm-3.0-pytorch"
MAX_CONTEXT = 15_360
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_TOTAL_DECODED_BYTES = 512 * 1024 * 1024
MAX_VARIATES = 32
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class ExplorerError(ValueError):
  """Raised when uploaded data or forecast settings are invalid."""


@dataclasses.dataclass(frozen=True)
class DatasetMapping:
  """Column roles selected by the user."""

  timestamp: str | None
  targets: tuple[str, ...]
  past_only: tuple[str, ...] = ()
  past_future: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ForecastSettings:
  """Stable TimesFM-3 inference controls exposed by the app."""

  horizon: int = 64
  context_length: int = 512
  task: Literal["forecast", "holdout"] = "forecast"
  mode: Literal["multivariate", "univariate"] = "multivariate"
  return_quantiles: bool = True
  use_symmetric_averaging: bool = True
  make_positive: bool = True
  sort_quantiles: bool = True
  use_znorm: bool = False
  padding_mode: Literal["none", "edge"] = "none"
  batch_size: int = 4
  allow_benchmark_chunking: bool = False

  def validate(self) -> None:
    """Validate values again server-side before inference."""
    if not 1 <= self.horizon <= MAX_CONTEXT:
      raise ExplorerError(f"Horizon must be between 1 and {MAX_CONTEXT:,}.")
    if not 1 <= self.context_length <= MAX_CONTEXT:
      raise ExplorerError(f"Context length must be between 1 and {MAX_CONTEXT:,}.")
    if not 1 <= self.batch_size <= 64:
      raise ExplorerError("Batch size must be between 1 and 64.")
    if self.task not in {"forecast", "holdout"}:
      raise ExplorerError("Unknown task mode.")
    if self.mode not in {"multivariate", "univariate"}:
      raise ExplorerError("Unknown forecast mode.")
    if self.padding_mode not in {"none", "edge"}:
      raise ExplorerError("Unknown padding mode.")


@dataclasses.dataclass(frozen=True)
class UploadedDataset:
  """Parsed in-memory upload with a privacy-safe identifier."""

  dataset_id: str
  frame: pd.DataFrame
  sha256: str
  byte_size: int
  memory_bytes: int


@dataclasses.dataclass(frozen=True)
class ChunkingPlan:
  """Deterministic evaluator behavior when inputs exceed 32 variates."""

  total_variates: int
  selected_past_only: tuple[str, ...]
  selected_past_future: tuple[str, ...]
  targets_per_chunk: int
  target_chunks: int


@dataclasses.dataclass(frozen=True)
class PreparedSeries:
  """One file converted to TimesFM arrays."""

  dataset_id: str
  sha256: str
  target_names: tuple[str, ...]
  context: np.ndarray
  past_only: np.ndarray | None
  past_future: np.ndarray | None
  history_time: tuple[Any, ...]
  future_time: tuple[Any, ...]
  actual: np.ndarray | None
  lineage: tuple[dict[str, Any], ...]


@dataclasses.dataclass(frozen=True)
class PreparedBatch:
  """Validated inputs for one evaluator call."""

  series: tuple[PreparedSeries, ...]
  mapping: DatasetMapping
  settings: ForecastSettings
  chunking: ChunkingPlan


@dataclasses.dataclass(frozen=True)
class CapabilityReport:
  """Local runtime capability summary."""

  python: str
  torch: str
  cuda_available: bool
  device: str
  vram_total_gb: float | None
  vram_free_gb: float | None
  checkpoint_cached: bool
  hf_token_present: bool


@dataclasses.dataclass(frozen=True)
class RunArtifact:
  """Serializable result retained in Streamlit session state."""

  run_id: str
  created_at: str
  settings: ForecastSettings
  mapping: DatasetMapping
  history: pd.DataFrame
  forecast: pd.DataFrame
  metrics: pd.DataFrame
  manifest: dict[str, Any]
  runtime_seconds: float
  device: str


class BatchPredictor(Protocol):
  """Narrow evaluator interface used for unit testing."""

  device: Any

  def predict_batch(self, **kwargs: Any) -> Iterable[ForecastOutput]:
    """Return forecast outputs."""


def parse_upload(data: bytes, suffix: str, dataset_id: str) -> UploadedDataset:
  """Parse a CSV or Parquet upload without writing it to disk."""
  if len(data) > MAX_UPLOAD_BYTES:
    raise ExplorerError("Each file must be 50 MB or smaller.")
  normalized_suffix = suffix.lower().lstrip(".")
  try:
    if normalized_suffix == "csv":
      frame = pd.read_csv(io.BytesIO(data), encoding="utf-8-sig")
    elif normalized_suffix in {"parquet", "pq"}:
      from pyarrow import parquet

      metadata = parquet.ParquetFile(io.BytesIO(data)).metadata
      decoded_size = sum(
        metadata.row_group(index).total_byte_size
        for index in range(metadata.num_row_groups)
      )
      if decoded_size > MAX_DECODED_BYTES:
        raise ExplorerError(f"{dataset_id} expands beyond the 256 MB memory limit.")
      frame = pd.read_parquet(io.BytesIO(data))
    else:
      raise ExplorerError("Only CSV and Parquet files are supported.")
  except ExplorerError:
    raise
  except Exception as exc:
    raise ExplorerError(f"Could not parse {dataset_id}: {exc}") from exc
  if frame.empty or not len(frame.columns):
    raise ExplorerError(f"{dataset_id} contains no data.")
  normalized_columns = [str(column) for column in frame.columns]
  if len(normalized_columns) != len(set(normalized_columns)):
    raise ExplorerError(f"{dataset_id} contains duplicate column names.")
  frame.columns = normalized_columns
  memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
  if memory_bytes > MAX_DECODED_BYTES:
    raise ExplorerError(f"{dataset_id} expands beyond the 256 MB memory limit.")
  return UploadedDataset(
    dataset_id=dataset_id,
    frame=frame,
    sha256=hashlib.sha256(data).hexdigest(),
    byte_size=len(data),
    memory_bytes=memory_bytes,
  )


def validate_upload_total(datasets: Sequence[UploadedDataset]) -> None:
  """Bound aggregate upload memory before dataframe processing."""
  if sum(item.byte_size for item in datasets) > MAX_TOTAL_UPLOAD_BYTES:
    raise ExplorerError("Combined uploads must be 200 MB or smaller.")
  if sum(item.memory_bytes for item in datasets) > MAX_TOTAL_DECODED_BYTES:
    raise ExplorerError("Combined decoded uploads must use 512 MB or less.")


def _validate_mapping(frame: pd.DataFrame, mapping: DatasetMapping) -> None:
  if not mapping.targets:
    raise ExplorerError("Select at least one target column.")
  groups = [mapping.targets, mapping.past_only, mapping.past_future]
  assigned = [column for group in groups for column in group]
  if len(assigned) != len(set(assigned)):
    raise ExplorerError("A numeric column can have only one role.")
  if mapping.timestamp is not None and mapping.timestamp in assigned:
    raise ExplorerError("Timestamp column cannot also be a model input.")
  required = set(assigned)
  if mapping.timestamp is not None:
    required.add(mapping.timestamp)
  missing = sorted(required.difference(frame.columns))
  if missing:
    raise ExplorerError(f"Missing mapped columns: {', '.join(missing)}")


def _coerce_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
  numeric = pd.DataFrame(index=frame.index)
  for column in columns:
    converted = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & converted.isna()
    if invalid.any():
      raise ExplorerError(f"Column '{column}' contains non-numeric values.")
    finite = converted.dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all():
      raise ExplorerError(f"Column '{column}' contains infinite values.")
    if np.abs(finite).max(initial=0) > np.finfo(np.float32).max:
      raise ExplorerError(f"Column '{column}' contains values outside float32 range.")
    numeric[column] = converted.astype(float)
  return numeric


def _time_axis(
  frame: pd.DataFrame, timestamp: str | None
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, Any]]]:
  lineage: list[dict[str, Any]] = []
  if timestamp is None:
    axis = pd.Series(np.arange(len(frame)), index=frame.index, name="step")
    return frame.reset_index(drop=True), axis.reset_index(drop=True), lineage

  try:
    parsed = pd.to_datetime(frame[timestamp], errors="coerce", format="mixed")
    if parsed.dtype == object:
      parsed = pd.to_datetime(
        frame[timestamp], errors="coerce", format="mixed", utc=True
      )
  except (TypeError, ValueError, OverflowError) as exc:
    raise ExplorerError(f"Timestamp column '{timestamp}' contains invalid values.") from exc
  if parsed.isna().any():
    raise ExplorerError(f"Timestamp column '{timestamp}' contains invalid values.")
  if parsed.duplicated().any():
    raise ExplorerError(f"Timestamp column '{timestamp}' contains duplicates.")
  order = np.argsort(parsed.to_numpy(), kind="stable")
  if not np.array_equal(order, np.arange(len(frame))):
    lineage.append({"operation": "sort", "column": timestamp})
  sorted_frame = frame.iloc[order].reset_index(drop=True)
  sorted_axis = parsed.iloc[order].reset_index(drop=True)
  return sorted_frame, sorted_axis, lineage


def _future_axis(
  axis: pd.Series,
  history_end: int,
  horizon: int,
  uploaded_end: int,
) -> tuple[Any, ...]:
  uploaded = axis.iloc[history_end : min(uploaded_end, history_end + horizon)]
  if len(uploaded) == horizon:
    return tuple(uploaded.tolist())
  if history_end < 1:
    return tuple(range(1, horizon + 1))
  if pd.api.types.is_datetime64_any_dtype(axis):
    history = pd.DatetimeIndex(axis.iloc[:history_end])
    frequency = (
      pd.infer_freq(history[-min(len(history), 10) :]) if len(history) >= 3 else None
    )
    if frequency:
      generated = pd.date_range(history[-1], periods=horizon + 1, freq=frequency)[1:]
      return tuple(generated.tolist())
    if len(history) >= 2:
      delta = history[-1] - history[-2]
      if delta > pd.Timedelta(0):
        return tuple((history[-1] + delta * step) for step in range(1, horizon + 1))
  start = (
    int(axis.iloc[history_end - 1]) + 1
    if pd.api.types.is_numeric_dtype(axis.dtype)
    else 1
  )
  return tuple(range(start, start + horizon))


def _chunking_plan(mapping: DatasetMapping) -> ChunkingPlan:
  num_targets = len(mapping.targets)
  num_pf = len(mapping.past_future)
  num_po = len(mapping.past_only)
  total = num_targets + num_pf + num_po
  selected_pf = mapping.past_future
  selected_po = mapping.past_only
  if total > MAX_VARIATES:
    rng = np.random.default_rng(42)
    max_pf = min(num_pf, MAX_VARIATES - 1)
    if num_pf > max_pf:
      indices = np.sort(rng.choice(num_pf, max_pf, replace=False))
      selected_pf = tuple(mapping.past_future[index] for index in indices)
    max_po = min(num_po, MAX_VARIATES - 1 - len(selected_pf))
    if num_po > max_po:
      indices = np.sort(rng.choice(num_po, max_po, replace=False))
      selected_po = tuple(mapping.past_only[index] for index in indices)
  targets_per_chunk = MAX_VARIATES - len(selected_pf) - len(selected_po)
  return ChunkingPlan(
    total_variates=total,
    selected_past_only=selected_po,
    selected_past_future=selected_pf,
    targets_per_chunk=max(1, targets_per_chunk),
    target_chunks=max(1, math.ceil(num_targets / max(1, targets_per_chunk))),
  )


def prepare_batch(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  settings: ForecastSettings,
) -> PreparedBatch:
  """Validate uploads and create aligned TimesFM arrays."""
  settings.validate()
  validate_upload_total(datasets)
  if not datasets:
    raise ExplorerError("Upload at least one dataset.")
  chunking = _chunking_plan(mapping)
  if settings.mode == "univariate" and (mapping.past_only or mapping.past_future):
    raise ExplorerError("Independent univariate mode does not use covariates.")
  if (
    settings.mode == "multivariate"
    and chunking.total_variates > MAX_VARIATES
    and not settings.allow_benchmark_chunking
  ):
    raise ExplorerError(
      "More than 32 combined variates requires benchmark chunking approval."
    )

  prepared = tuple(_prepare_series(dataset, mapping, settings) for dataset in datasets)
  return PreparedBatch(
    series=prepared,
    mapping=mapping,
    settings=settings,
    chunking=chunking,
  )


def _prepare_series(
  dataset: UploadedDataset,
  mapping: DatasetMapping,
  settings: ForecastSettings,
) -> PreparedSeries:
  _validate_mapping(dataset.frame, mapping)
  frame, axis, lineage = _time_axis(dataset.frame, mapping.timestamp)
  model_columns = mapping.targets + mapping.past_only + mapping.past_future
  numeric = _coerce_numeric(frame, model_columns)
  target = numeric.loc[:, mapping.targets]
  observed_rows = target.notna().any(axis=1)
  if not observed_rows.any():
    raise ExplorerError(f"{dataset.dataset_id} has no observed target values.")
  history_end = int(np.flatnonzero(observed_rows.to_numpy())[-1]) + 1

  if settings.task == "holdout":
    if history_end <= settings.horizon:
      raise ExplorerError(
        f"{dataset.dataset_id} needs more observed rows than the holdout horizon."
      )
    forecast_start = history_end - settings.horizon
    forecast_end = history_end
    actual = target.iloc[forecast_start:forecast_end].to_numpy(dtype=np.float32).T
  else:
    forecast_start = history_end
    forecast_end = history_end + settings.horizon
    actual = None

  context_start = max(0, forecast_start - settings.context_length)
  context_frame = target.iloc[context_start:forecast_start]
  insufficient = [
    column for column in mapping.targets if context_frame[column].notna().sum() < 2
  ]
  if insufficient:
    raise ExplorerError(
      f"{dataset.dataset_id} needs at least two context values for: "
      + ", ".join(insufficient)
    )
  if context_start > 0:
    lineage.append(
      {
        "operation": "truncate_context",
        "removed_rows": context_start,
        "kept_rows": forecast_start - context_start,
      }
    )
  missing_counts = (
    numeric.loc[context_start : forecast_start - 1, model_columns].isna().sum()
  )
  if missing_counts.any():
    lineage.append(
      {
        "operation": "timesfm_linear_interpolation",
        "missing_by_column": {
          key: int(value) for key, value in missing_counts.items() if value
        },
      }
    )

  po = None
  if mapping.past_only:
    po = (
      numeric.loc[context_start : forecast_start - 1, mapping.past_only]
      .to_numpy(dtype=np.float32)
      .T
    )

  pf = None
  if mapping.past_future:
    if forecast_end > len(numeric):
      raise ExplorerError(
        f"{dataset.dataset_id} needs {settings.horizon} future rows for "
        "past-future covariates."
      )
    future_cov = numeric.loc[context_start : forecast_end - 1, mapping.past_future]
    future_slice = numeric.loc[forecast_start : forecast_end - 1, mapping.past_future]
    if future_slice.isna().any().any():
      raise ExplorerError(
        f"{dataset.dataset_id} has missing known-future covariate values."
      )
    pf = future_cov.to_numpy(dtype=np.float32).T

  future_time = _future_axis(axis, forecast_start, settings.horizon, forecast_end)
  return PreparedSeries(
    dataset_id=dataset.dataset_id,
    sha256=dataset.sha256,
    target_names=mapping.targets,
    context=context_frame.to_numpy(dtype=np.float32).T,
    past_only=po,
    past_future=pf,
    history_time=tuple(axis.iloc[context_start:forecast_start].tolist()),
    future_time=future_time,
    actual=actual,
    lineage=tuple(lineage),
  )


def run_forecast(
  predictor: BatchPredictor, batch: PreparedBatch
) -> tuple[list[ForecastOutput], float]:
  """Run one explicit TimesFM evaluator call."""
  settings = batch.settings
  started = time.perf_counter()
  outputs = list(
    predictor.predict_batch(
      contexts=[item.context for item in batch.series],
      horizon=settings.horizon,
      past_only_covariates=[item.past_only for item in batch.series],
      past_future_covariates=[item.past_future for item in batch.series],
      ts_ids=[item.dataset_id for item in batch.series],
      return_quantiles=settings.return_quantiles,
      use_symmetric_averaging=settings.use_symmetric_averaging,
      make_positive=settings.make_positive,
      sort_quantiles=settings.sort_quantiles,
      use_znorm=settings.use_znorm,
      padding_mode=settings.padding_mode,
      univariate=settings.mode == "univariate",
    )
  )
  return outputs, time.perf_counter() - started


def forecast_table(
  batch: PreparedBatch, outputs: Sequence[ForecastOutput]
) -> pd.DataFrame:
  """Convert evaluator outputs to a long-form dataframe."""
  if len(outputs) != len(batch.series):
    raise ExplorerError("Evaluator returned an unexpected number of outputs.")
  rows: list[dict[str, Any]] = []
  for prepared, output in zip(batch.series, outputs, strict=True):
    if output.forecast is None:
      raise ExplorerError(f"No forecast returned for {prepared.dataset_id}.")
    point = np.asarray(output.forecast)
    if point.ndim == 1:
      point = point[np.newaxis, :]
    quantiles = None if output.quantiles is None else np.asarray(output.quantiles)
    if quantiles is not None and quantiles.ndim == 2:
      quantiles = quantiles[np.newaxis, :, :]
    expected = (len(prepared.target_names), batch.settings.horizon)
    if point.shape != expected:
      raise ExplorerError(
        f"Unexpected forecast shape {point.shape}; expected {expected}."
      )
    if not np.isfinite(point).all():
      raise ExplorerError(f"Forecast for {prepared.dataset_id} contains non-finite values.")
    expected_quantiles = (*expected, len(QUANTILES))
    if quantiles is not None and quantiles.shape != expected_quantiles:
      raise ExplorerError(
        f"Unexpected quantile shape {quantiles.shape}; expected {expected_quantiles}."
      )
    if quantiles is not None and not np.isfinite(quantiles).all():
      raise ExplorerError(
        f"Quantiles for {prepared.dataset_id} contain non-finite values."
      )
    for target_index, target_name in enumerate(prepared.target_names):
      for step in range(batch.settings.horizon):
        row: dict[str, Any] = {
          "dataset": prepared.dataset_id,
          "target": target_name,
          "step": step + 1,
          "timestamp": prepared.future_time[step],
          "point": float(point[target_index, step]),
        }
        if prepared.actual is not None:
          row["actual"] = float(prepared.actual[target_index, step])
        if quantiles is not None:
          for quantile_index, quantile in enumerate(QUANTILES):
            row[f"q{quantile:.1f}"] = float(
              quantiles[target_index, step, quantile_index]
            )
        rows.append(row)
  return pd.DataFrame(rows)


def evaluation_metrics(forecast: pd.DataFrame) -> pd.DataFrame:
  """Compute holdout metrics per dataset and target."""
  if "actual" not in forecast:
    return pd.DataFrame()
  rows: list[dict[str, Any]] = []
  quantile_columns = [f"q{quantile:.1f}" for quantile in QUANTILES]
  for _, group in forecast.groupby(["dataset", "target"], sort=False):
    dataset = group["dataset"].iloc[0]
    target = group["target"].iloc[0]
    valid = group["actual"].notna() & group["point"].notna()
    current = group.loc[valid]
    if current.empty:
      continue
    actual = current["actual"].to_numpy(dtype=float)
    point = current["point"].to_numpy(dtype=float)
    error = actual - point
    denominator = np.abs(actual) + np.abs(point)
    smape_terms = np.divide(
      2 * np.abs(error),
      denominator,
      out=np.zeros_like(error),
      where=denominator != 0,
    )
    row: dict[str, Any] = {
      "dataset": dataset,
      "target": target,
      "observations": len(current),
      "mae": float(np.mean(np.abs(error))),
      "rmse": float(np.sqrt(np.mean(np.square(error)))),
      "smape_percent": float(np.mean(smape_terms) * 100),
    }
    if all(column in current for column in quantile_columns):
      losses = []
      for quantile, column in zip(QUANTILES, quantile_columns, strict=True):
        quantile_error = actual - current[column].to_numpy(dtype=float)
        losses.append(
          np.maximum(quantile * quantile_error, (quantile - 1) * quantile_error)
        )
      row["mean_pinball_loss"] = float(np.mean(np.stack(losses)))
      row["q10_q90_coverage_percent"] = float(
        np.mean(
          (actual >= current["q0.1"].to_numpy(dtype=float))
          & (actual <= current["q0.9"].to_numpy(dtype=float))
        )
        * 100
      )
    rows.append(row)
  return pd.DataFrame(rows)


def history_table(batch: PreparedBatch) -> pd.DataFrame:
  """Convert model context to a chart-ready dataframe."""
  rows: list[dict[str, Any]] = []
  for prepared in batch.series:
    for target_index, target_name in enumerate(prepared.target_names):
      for step, (timestamp, value) in enumerate(
        zip(
          prepared.history_time,
          prepared.context[target_index],
          strict=True,
        ),
        start=1,
      ):
        rows.append(
          {
            "dataset": prepared.dataset_id,
            "target": target_name,
            "step": step,
            "timestamp": timestamp,
            "value": float(value),
          }
        )
  return pd.DataFrame(rows)


def capability_report() -> CapabilityReport:
  """Probe local Torch/CUDA capability without downloading a checkpoint."""
  cuda_available = torch.cuda.is_available()
  device = "cpu"
  total = free = None
  if cuda_available:
    device = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    total = total_bytes / 1024**3
    free = free_bytes / 1024**3
  cache_root = Path(
    os.environ.get(
      "HF_HUB_CACHE",
      Path.home() / ".cache" / "huggingface" / "hub",
    )
  )
  checkpoint_cached = (cache_root / "models--google--timesfm-3.0-pytorch").exists()
  return CapabilityReport(
    python=platform.python_version(),
    torch=torch.__version__,
    cuda_available=cuda_available,
    device=device,
    vram_total_gb=total,
    vram_free_gb=free,
    checkpoint_cached=checkpoint_cached,
    hf_token_present=bool(os.environ.get("HF_TOKEN")),
  )


def make_run_artifact(
  batch: PreparedBatch,
  outputs: Sequence[ForecastOutput],
  runtime_seconds: float,
  device: str,
  repository_revision: str = "unknown",
) -> RunArtifact:
  """Build results and a reproducibility manifest."""
  table = forecast_table(batch, outputs)
  metrics = evaluation_metrics(table)
  history = history_table(batch)
  created_at = pd.Timestamp.now(tz="UTC").isoformat()
  seed = "|".join(item.sha256 for item in batch.series) + created_at
  run_id = hashlib.sha256(seed.encode()).hexdigest()[:10]
  manifest = {
    "schema_version": 1,
    "run_id": run_id,
    "created_at": created_at,
    "checkpoint": CHECKPOINT_ID,
    "repository_revision": repository_revision,
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "device": device,
    "runtime_seconds": runtime_seconds,
    "settings": dataclasses.asdict(batch.settings),
    "mapping": dataclasses.asdict(batch.mapping),
    "datasets": [
      {
        "dataset_id": item.dataset_id,
        "sha256": item.sha256,
        "context_shape": list(item.context.shape),
        "lineage": list(item.lineage),
      }
      for item in batch.series
    ],
    "chunking": dataclasses.asdict(batch.chunking),
    "license": "timesfm-non-commercial-license-v1.0",
  }
  return RunArtifact(
    run_id=run_id,
    created_at=created_at,
    settings=batch.settings,
    mapping=batch.mapping,
    history=history,
    forecast=table,
    metrics=metrics,
    manifest=manifest,
    runtime_seconds=runtime_seconds,
    device=device,
  )


def execute_forecast(
  predictor: BatchPredictor,
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  settings: ForecastSettings,
  repository_revision: str = "unknown",
) -> RunArtifact:
  """Validate, forecast, and package one complete explorer run."""
  batch = prepare_batch(datasets, mapping, settings)
  outputs, runtime_seconds = run_forecast(predictor, batch)
  return make_run_artifact(
    batch,
    outputs,
    runtime_seconds,
    str(predictor.device),
    repository_revision,
  )


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
  """Neutralize spreadsheet formulas in user-controlled CSV cells."""
  safe = frame.copy()
  dangerous = ("=", "+", "-", "@", "\t", "\r")
  for column in safe.select_dtypes(include=["object", "string"]).columns:
    safe[column] = safe[column].map(
      lambda value: (
        "'" + value
        if isinstance(value, str) and value.lstrip(" ").startswith(dangerous)
        else value
      )
    )
  return safe


def artifact_zip(artifact: RunArtifact) -> bytes:
  """Create a portable result bundle entirely in memory."""
  output = io.BytesIO()
  with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("forecast.csv", _csv_safe(artifact.forecast).to_csv(index=False))
    if not artifact.metrics.empty:
      archive.writestr("metrics.csv", _csv_safe(artifact.metrics).to_csv(index=False))
    archive.writestr(
      "run.json",
      json.dumps(artifact.manifest, indent=2, default=str),
    )
  return output.getvalue()


def demo_dataset(kind: Literal["univariate", "multivariate"]) -> pd.DataFrame:
  """Return a deterministic upload-shaped demo dataframe."""
  rng = np.random.default_rng(7)
  context = 160
  horizon = 32
  dates = pd.date_range("2026-01-01", periods=context + horizon, freq="D")
  time_index = np.arange(context + horizon)
  promotion = ((time_index % 14) < 3).astype(float)
  temperature = 20 + 5 * np.sin(2 * np.pi * time_index / 30)
  sales = 100 + 15 * np.sin(2 * np.pi * time_index / 7) + 22 * promotion
  sales += rng.normal(0, 2, context + horizon)
  sales[context:] = np.nan
  if kind == "univariate":
    return pd.DataFrame({"date": dates, "sales": sales})
  demand = 65 + 8 * np.cos(2 * np.pi * time_index / 7) + 10 * promotion
  demand += rng.normal(0, 1.5, context + horizon)
  demand[context:] = np.nan
  return pd.DataFrame(
    {
      "date": dates,
      "sales": sales,
      "demand": demand,
      "temperature": temperature,
      "promotion": promotion,
    }
  )


def load_forecaster(device: str, batch_size: int) -> TimesFM3Evaluator:
  """Load official TimesFM-3 evaluator; caller owns resource caching."""
  return cast(
    TimesFM3Evaluator,
    TimesFM3Evaluator.from_pretrained(
      CHECKPOINT_ID,
      device=device,
      per_core_batch_size=batch_size,
    ),
  )


def repository_revision(root: Path | None = None) -> str:
  """Return current Git revision without raising outside a checkout."""
  try:
    completed = subprocess.run(
      ["git", "rev-parse", "HEAD"],
      cwd=root,
      check=True,
      capture_output=True,
      text=True,
      timeout=2,
    )
  except (OSError, subprocess.SubprocessError):
    return "unknown"
  return completed.stdout.strip() or "unknown"
