# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""DuckDB persistence for derived explorer run artifacts."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .explorer import DatasetMapping, ForecastSettings, RunArtifact

MAX_SAVED_RUNS = 25
_QUANTILES = tuple(f"q{value / 10:.1f}" for value in range(1, 10))
_STORED_QUANTILES = tuple(column.replace(".", "_") for column in _QUANTILES)


class RunStoreError(RuntimeError):
  """Raised when local run persistence is unavailable."""


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
  connection.execute(
    """
    CREATE TABLE IF NOT EXISTS runs (
      run_id VARCHAR PRIMARY KEY,
      created_at VARCHAR NOT NULL,
      settings JSON NOT NULL,
      mapping JSON NOT NULL,
      manifest JSON NOT NULL,
      runtime_seconds DOUBLE NOT NULL,
      device VARCHAR NOT NULL
    );
    CREATE TABLE IF NOT EXISTS forecasts (
      run_id VARCHAR NOT NULL,
      dataset VARCHAR NOT NULL,
      target VARCHAR NOT NULL,
      step BIGINT NOT NULL,
      timestamp VARCHAR NOT NULL,
      timestamp_is_temporal BOOLEAN NOT NULL,
      point DOUBLE NOT NULL,
      actual DOUBLE,
      q0_1 DOUBLE,
      q0_2 DOUBLE,
      q0_3 DOUBLE,
      q0_4 DOUBLE,
      q0_5 DOUBLE,
      q0_6 DOUBLE,
      q0_7 DOUBLE,
      q0_8 DOUBLE,
      q0_9 DOUBLE
    );
    CREATE TABLE IF NOT EXISTS metrics (
      run_id VARCHAR NOT NULL,
      dataset VARCHAR NOT NULL,
      target VARCHAR NOT NULL,
      observations BIGINT NOT NULL,
      mae DOUBLE NOT NULL,
      rmse DOUBLE NOT NULL,
      smape_percent DOUBLE NOT NULL,
      mean_pinball_loss DOUBLE,
      q10_q90_coverage_percent DOUBLE
    );
    """
  )


def _forecast_for_storage(run: RunArtifact) -> pd.DataFrame:
  frame = run.forecast.copy()
  temporal = pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
  frame["timestamp_is_temporal"] = temporal
  frame["timestamp"] = frame["timestamp"].astype(str)
  frame.insert(0, "run_id", run.run_id)
  for column in ("actual", *_QUANTILES):
    if column not in frame:
      frame[column] = None
  return frame.rename(columns=dict(zip(_QUANTILES, _STORED_QUANTILES, strict=True)))[
    [
      "run_id",
      "dataset",
      "target",
      "step",
      "timestamp",
      "timestamp_is_temporal",
      "point",
      "actual",
      *_STORED_QUANTILES,
    ]
  ]


def _metrics_for_storage(run: RunArtifact) -> pd.DataFrame:
  columns = [
    "run_id",
    "dataset",
    "target",
    "observations",
    "mae",
    "rmse",
    "smape_percent",
    "mean_pinball_loss",
    "q10_q90_coverage_percent",
  ]
  if run.metrics.empty:
    return pd.DataFrame(columns=columns)
  frame = run.metrics.copy()
  frame.insert(0, "run_id", run.run_id)
  for column in columns:
    if column not in frame:
      frame[column] = None
  return frame[columns]


def save_run(
  database_path: Path, run: RunArtifact, limit: int = MAX_SAVED_RUNS
) -> None:
  """Save one derived run and prune runs beyond the retention limit."""
  if limit < 1:
    raise ValueError("Run retention limit must be positive.")
  try:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
      _create_schema(connection)
      forecast = _forecast_for_storage(run)
      metrics = _metrics_for_storage(run)
      connection.register("current_forecast", forecast)
      connection.register("current_metrics", metrics)
      connection.execute("BEGIN TRANSACTION")
      connection.execute("DELETE FROM forecasts WHERE run_id = ?", [run.run_id])
      connection.execute("DELETE FROM metrics WHERE run_id = ?", [run.run_id])
      connection.execute("DELETE FROM runs WHERE run_id = ?", [run.run_id])
      connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
          run.run_id,
          run.created_at,
          json.dumps(dataclasses.asdict(run.settings)),
          json.dumps(dataclasses.asdict(run.mapping)),
          json.dumps(run.manifest, default=str),
          run.runtime_seconds,
          run.device,
        ],
      )
      connection.execute("INSERT INTO forecasts SELECT * FROM current_forecast")
      if not metrics.empty:
        connection.execute("INSERT INTO metrics SELECT * FROM current_metrics")
      stale = connection.execute(
        "SELECT run_id FROM runs ORDER BY created_at DESC OFFSET ?", [limit]
      ).fetchall()
      stale_ids = [row[0] for row in stale]
      if stale_ids:
        placeholders = ", ".join("?" for _ in stale_ids)
        connection.execute(
          f"DELETE FROM forecasts WHERE run_id IN ({placeholders})", stale_ids
        )
        connection.execute(
          f"DELETE FROM metrics WHERE run_id IN ({placeholders})", stale_ids
        )
        connection.execute(
          f"DELETE FROM runs WHERE run_id IN ({placeholders})", stale_ids
        )
      connection.execute("COMMIT")
  except Exception as exc:
    raise RunStoreError(f"Could not save local run history: {exc}") from exc


def _load_json(value: Any) -> dict[str, Any]:
  return json.loads(value) if isinstance(value, str) else dict(value)


def _restore_forecast(frame: pd.DataFrame) -> pd.DataFrame:
  if frame.empty:
    return frame
  temporal = frame.pop("timestamp_is_temporal").astype(bool)
  if temporal.all():
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
  else:
    frame["timestamp"] = pd.to_numeric(frame["timestamp"])
  frame = frame.drop(columns="run_id").rename(
    columns=dict(zip(_STORED_QUANTILES, _QUANTILES, strict=True))
  )
  optional = ["actual", *_QUANTILES]
  return frame.drop(columns=[column for column in optional if frame[column].isna().all()])


def _restore_metrics(frame: pd.DataFrame) -> pd.DataFrame:
  if frame.empty:
    return pd.DataFrame()
  frame = frame.drop(columns="run_id")
  optional = ["mean_pinball_loss", "q10_q90_coverage_percent"]
  return frame.drop(columns=[column for column in optional if frame[column].isna().all()])


def load_recent_runs(
  database_path: Path, limit: int = MAX_SAVED_RUNS
) -> list[RunArtifact]:
  """Load newest derived runs, or return an empty list before first save."""
  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      rows = connection.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", [limit]
      ).fetchall()
      runs = []
      for row in reversed(rows):
        run_id, created_at, settings_value, mapping_value, manifest_value, runtime, device = row
        settings = ForecastSettings(**_load_json(settings_value))
        mapping_data = _load_json(mapping_value)
        for field in ("targets", "past_only", "past_future"):
          mapping_data[field] = tuple(mapping_data.get(field, ()))
        forecast = connection.execute(
          "SELECT * FROM forecasts WHERE run_id = ? ORDER BY dataset, target, step",
          [run_id],
        ).df()
        metrics = connection.execute(
          "SELECT * FROM metrics WHERE run_id = ? ORDER BY dataset, target", [run_id]
        ).df()
        runs.append(
          RunArtifact(
            run_id=run_id,
            created_at=pd.Timestamp(created_at).isoformat(),
            settings=settings,
            mapping=DatasetMapping(**mapping_data),
            history=pd.DataFrame(
              columns=["dataset", "target", "step", "timestamp", "value"]
            ),
            forecast=_restore_forecast(forecast),
            metrics=_restore_metrics(metrics),
            manifest=_load_json(manifest_value),
            runtime_seconds=float(runtime),
            device=device,
          )
        )
      return runs
  except Exception as exc:
    raise RunStoreError(f"Could not load local run history: {exc}") from exc
