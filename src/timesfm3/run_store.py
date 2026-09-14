# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""DuckDB persistence for derived explorer run artifacts.

Single-file DuckDB database (path chosen by the caller) storing forecast
"runs" (see `explorer.py:RunArtifact`), their tracking links/assessments
(see `tracking.py`), and derived "analyses" (see `analysis.py`). Forecast
and metrics tables are dual-written on every save: once as flattened SQL
columns (`forecasts`/`metrics`, queryable directly with SQL) and once as
a full-fidelity Parquet BLOB (`run_tables`, used preferentially on load
since it round-trips dtypes/timezones exactly -- see `_table_to_parquet`/
`_table_from_parquet`). Every public function wraps its body in
`except RunStoreError: raise` / `except Exception as exc: raise
RunStoreError(...) from exc`, so callers only ever see `RunStoreError`;
this isn't re-commented at each call site below. `run_id`s are immutable
once saved (`save_run` errors if content differs on a re-save with the
same id); `tracked_runs` membership is what exempts a run from the
retention pruning in `_prune_runs`. See `explorer.py` for how
`RunArtifact`/`ForecastSettings`/`DatasetMapping` are produced.
"""

from __future__ import annotations

import dataclasses
import datetime
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet

from .explorer import DatasetMapping, ForecastSettings, RunArtifact

if TYPE_CHECKING:
  from .analysis import AnalysisArtifact
  from .tracking import TrackingAssessment

MAX_SAVED_RUNS = 25
_QUANTILES = tuple(f"q{value / 10:.1f}" for value in range(1, 10))
_STORED_QUANTILES = tuple(column.replace(".", "_") for column in _QUANTILES)
_OBJECT_TIMESTAMPS = b"timesfm3.object_timestamp_columns.v1"


class RunStoreError(RuntimeError):
  """Raised when local run persistence is unavailable."""


def _is_timestamp(value: Any) -> bool:
  return isinstance(value, (datetime.datetime, np.datetime64, pd.Timestamp))


def _table_to_parquet(frame: pd.DataFrame) -> bytes:
  """Preserve mixed timezone columns that Arrow otherwise coerces to one zone.

  Object-dtype columns holding only timestamp-like values (mixed
  tz-aware/naive, or mixed offsets) would otherwise be silently forced to
  a single timezone by `pa.Table.from_pandas`. Instead, such columns are
  serialized to ISO-8601 strings and the set of affected column names is
  recorded in a versioned custom metadata key (`_OBJECT_TIMESTAMPS`, see
  the ".v1" suffix) so `_table_from_parquet` knows which columns to
  reconstruct as `pd.Timestamp` objects rather than leaving them as
  strings.
  """
  columns = [
    column
    for column in frame
    if pd.api.types.is_object_dtype(frame[column])
    and not frame[column].dropna().empty
    and frame[column].dropna().map(_is_timestamp).all()
  ]
  if not columns:
    return frame.to_parquet(index=False)
  encoded = frame.copy()
  for column in columns:
    encoded[column] = frame[column].map(
      lambda value: None if pd.isna(value) else pd.Timestamp(value).isoformat()
    )
  table = pa.Table.from_pandas(encoded, preserve_index=False)
  metadata = dict(table.schema.metadata or {})
  metadata[_OBJECT_TIMESTAMPS] = json.dumps(columns).encode()
  if frame.attrs:
    metadata[b"PANDAS_ATTRS"] = json.dumps(frame.attrs).encode()
  buffer = io.BytesIO()
  parquet.write_table(table.replace_schema_metadata(metadata), buffer)
  return buffer.getvalue()


def _table_from_parquet(payload: bytes) -> pd.DataFrame:
  """Read both ordinary legacy Parquet and encoded mixed timezone payloads."""
  frame = pd.read_parquet(io.BytesIO(payload))
  metadata = parquet.read_metadata(io.BytesIO(payload)).metadata or {}
  for column in json.loads(metadata.get(_OBJECT_TIMESTAMPS, b"[]")):
    frame[column] = pd.Series(
      [pd.NaT if pd.isna(value) else pd.Timestamp(value) for value in frame[column]],
      index=frame.index,
      dtype=object,
    )
  return frame


def _create_schema(connection: duckdb.DuckDBPyConnection) -> None:
  # No foreign keys are declared between these tables (DuckDB won't
  # enforce cross-table deletes), so `_prune_runs` below deletes matching
  # run_id rows from every dependent table manually, in the same
  # transaction as the write that might trigger pruning.
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
    CREATE TABLE IF NOT EXISTS run_tables (
      run_id VARCHAR PRIMARY KEY,
      forecast BLOB NOT NULL,
      metrics BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tracked_runs (
      run_id VARCHAR PRIMARY KEY,
      tracked_at VARCHAR NOT NULL
    );
    CREATE TABLE IF NOT EXISTS tracking_links (
      run_id VARCHAR PRIMARY KEY,
      previous_run_id VARCHAR NOT NULL,
      associations JSON NOT NULL,
      linked_at VARCHAR NOT NULL
    );
    CREATE TABLE IF NOT EXISTS run_assessments (
      assessment_id VARCHAR PRIMARY KEY,
      run_id VARCHAR NOT NULL,
      created_at VARCHAR NOT NULL,
      fingerprint VARCHAR NOT NULL,
      manifest JSON NOT NULL,
      comparisons BLOB NOT NULL,
      metrics BLOB NOT NULL,
      UNIQUE (run_id, fingerprint)
    );
    """
  )


def _forecast_for_storage(run: RunArtifact) -> pd.DataFrame:
  frame = run.forecast.copy()
  frame["timestamp_is_temporal"] = frame["timestamp"].map(_is_timestamp)
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
  """Save an immutable derived run; retain tracked runs and newest untracked runs."""
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
      existing = connection.execute(
        "SELECT * FROM runs WHERE run_id = ?", [run.run_id]
      ).fetchone()
      if existing is not None:
        # Idempotent re-save of identical content succeeds silently
        # (e.g. a retried caller); re-saving under the same run_id with
        # different content is rejected outright, since run_id is meant
        # to identify one immutable forecast vintage.
        if not _same_run(_run_from_row(connection, existing), run):
          raise RunStoreError("A saved forecast cannot be changed. Create a new run.")
        connection.execute("COMMIT")
        return
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
      # Dual write: flattened SQL columns for ad-hoc querying, plus the
      # exact Parquet blob (below) that `_run_from_row` prefers on read.
      connection.execute("INSERT INTO forecasts SELECT * FROM current_forecast")
      if not metrics.empty:
        connection.execute("INSERT INTO metrics SELECT * FROM current_metrics")
      connection.execute(
        "INSERT INTO run_tables VALUES (?, ?, ?)",
        [
          run.run_id,
          _table_to_parquet(run.forecast),
          _table_to_parquet(run.metrics),
        ],
      )
      # Pruning runs inside the same transaction as the insert keeps the
      # store from ever holding more than `limit` untracked runs
      # mid-transaction, and rolls back together with the insert on
      # failure.
      _prune_runs(connection, limit)
      connection.execute("COMMIT")
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError(f"Could not save local run history: {exc}") from exc


def _table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
  row = connection.execute(
    "SELECT count(*) FROM information_schema.tables "
    "WHERE table_schema = 'main' AND table_name = ?",
    [name],
  ).fetchone()
  return bool(row and row[0])


def _prune_runs(connection: duckdb.DuckDBPyConnection, limit: int) -> None:
  # Keep the `limit` most recent untracked runs (OFFSET skips them);
  # everything older, among untracked runs only, is deleted. Tracked
  # runs (see tracked_runs, e.g. via set_run_tracked/link_runs/
  # save_assessment) are excluded from the candidate set entirely, so
  # they're retained indefinitely regardless of `limit`.
  stale = connection.execute(
    "SELECT run_id FROM runs WHERE run_id NOT IN (SELECT run_id FROM tracked_runs) "
    "ORDER BY CAST(created_at AS TIMESTAMPTZ) DESC, run_id DESC OFFSET ?",
    [limit],
  ).fetchall()
  stale_ids = [row[0] for row in stale]
  if not stale_ids:
    return
  placeholders = ", ".join("?" for _ in stale_ids)
  for table in ("forecasts", "metrics", "run_tables", "run_assessments", "runs"):
    connection.execute(
      f"DELETE FROM {table} WHERE run_id IN ({placeholders})", stale_ids
    )
  connection.execute(
    f"DELETE FROM tracking_links WHERE run_id IN ({placeholders})",
    stale_ids,
  )


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
  # Normalize both frames before comparing: an all-NaN column, column
  # order, and row order can all differ between the freshly-computed
  # `right` frame and a `left` frame reloaded from storage (storage adds
  # or fills optional columns and doesn't guarantee row/column order),
  # without the underlying data actually differing.
  frames = []
  for frame in (left, right):
    frame = frame.dropna(axis="columns", how="all")
    order = [key for key in ("dataset", "target", "step") if key in frame]
    if order:
      frame = frame.sort_values(order)
    frames.append(frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True))
  # assert_frame_equal is used here purely as an equality predicate (the
  # AssertionError is caught, not propagated).
  try:
    pd.testing.assert_frame_equal(
      *frames,
      check_dtype=False,
      check_exact=True,
      check_index_type=False,
      check_column_type=False,
    )
  except AssertionError:
    return False
  return True


def _same_run(left: RunArtifact, right: RunArtifact) -> bool:
  return (
    left.settings == right.settings
    and left.mapping == right.mapping
    and pd.Timestamp(left.created_at) == pd.Timestamp(right.created_at)
    and left.runtime_seconds == right.runtime_seconds
    and left.device == right.device
    and json.dumps(left.manifest, sort_keys=True, default=str)
    == json.dumps(right.manifest, sort_keys=True, default=str)
    and _same_frame(left.forecast, right.forecast)
    and _same_frame(left.metrics, right.metrics)
  )


def _load_json(value: Any) -> dict[str, Any]:
  return json.loads(value) if isinstance(value, str) else dict(value)


def _restore_forecast(frame: pd.DataFrame) -> pd.DataFrame:
  if frame.empty:
    return frame
  temporal = frame.pop("timestamp_is_temporal").astype(bool)
  frame["timestamp"] = pd.Series(
    [
      pd.Timestamp(value) if is_temporal else pd.to_numeric(value)
      for value, is_temporal in zip(frame["timestamp"], temporal, strict=True)
    ],
    index=frame.index,
  )
  frame = frame.drop(columns="run_id").rename(
    columns=dict(zip(_STORED_QUANTILES, _QUANTILES, strict=True))
  )
  optional = ["actual", *_QUANTILES]
  return frame.drop(
    columns=[column for column in optional if frame[column].isna().all()]
  )


def _restore_metrics(frame: pd.DataFrame) -> pd.DataFrame:
  if frame.empty:
    return pd.DataFrame()
  frame = frame.drop(columns="run_id")
  optional = ["mean_pinball_loss", "q10_q90_coverage_percent"]
  return frame.drop(
    columns=[column for column in optional if frame[column].isna().all()]
  )


def _run_from_row(
  connection: duckdb.DuckDBPyConnection, row: tuple[Any, ...]
) -> RunArtifact:
  run_id, created_at, settings, mapping, manifest, runtime, device = row
  mapping_data = _load_json(mapping)
  # JSON has no tuple type, so these round-trip as lists; convert back
  # to tuples to match DatasetMapping's field types (and dataclass
  # equality/hashing semantics, e.g. in _same_run).
  for field in ("targets", "past_only", "past_future"):
    mapping_data[field] = tuple(mapping_data.get(field, ()))
  payload = None
  if _table_exists(connection, "run_tables"):
    payload = connection.execute(
      "SELECT forecast, metrics FROM run_tables WHERE run_id = ?", [run_id]
    ).fetchone()
  if payload is not None:
    # Preferred path: the exact Parquet snapshot written alongside the
    # flattened tables (see save_run). Falls back below only for rows
    # saved before `run_tables` existed (schema evolved without a
    # migration script) or if that table itself is somehow missing.
    forecast = _table_from_parquet(payload[0])
    metrics = _table_from_parquet(payload[1])
  else:
    forecast = _restore_forecast(
      connection.execute(
        "SELECT * FROM forecasts WHERE run_id = ? ORDER BY dataset, target, step",
        [run_id],
      ).df()
    )
    metrics = _restore_metrics(
      connection.execute(
        "SELECT * FROM metrics WHERE run_id = ? ORDER BY dataset, target", [run_id]
      ).df()
    )
  return RunArtifact(
    run_id=run_id,
    created_at=pd.Timestamp(created_at).isoformat(),
    settings=ForecastSettings(**_load_json(settings)),
    mapping=DatasetMapping(**mapping_data),
    history=pd.DataFrame(columns=["dataset", "target", "step", "timestamp", "value"]),
    forecast=forecast,
    metrics=metrics,
    manifest=_load_json(manifest),
    runtime_seconds=float(runtime),
    device=device,
  )


def load_recent_runs(
  database_path: Path, limit: int = MAX_SAVED_RUNS
) -> list[RunArtifact]:
  """Load newest derived runs, or return an empty list before first save."""
  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      rows = connection.execute(
        "SELECT * FROM runs "
        "ORDER BY CAST(created_at AS TIMESTAMPTZ) DESC, run_id DESC LIMIT ?",
        [limit],
      ).fetchall()
      return [_run_from_row(connection, row) for row in reversed(rows)]
  except Exception as exc:
    raise RunStoreError(f"Could not load local run history: {exc}") from exc


def load_tracked_runs(database_path: Path) -> list[RunArtifact]:
  """Load protected forecast vintages without migrating old databases."""
  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      if not _table_exists(connection, "tracked_runs"):
        return []
      rows = connection.execute(
        "SELECT runs.* FROM runs JOIN tracked_runs USING (run_id) "
        "ORDER BY CAST(created_at AS TIMESTAMPTZ), run_id"
      ).fetchall()
      return [_run_from_row(connection, row) for row in rows]
  except Exception as exc:
    raise RunStoreError("Could not load tracked forecasts.") from exc


def list_tracking_runs(database_path: Path) -> list[dict[str, Any]]:
  """List retained forecast summaries without reading prediction payloads."""
  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      if not _table_exists(connection, "runs"):
        return []
      tracked = (
        "run_id IN (SELECT run_id FROM tracked_runs)"
        if _table_exists(connection, "tracked_runs")
        else "FALSE"
      )
      rows = connection.execute(
        f"SELECT run_id, created_at, {tracked} AS tracked FROM runs "
        "ORDER BY CAST(created_at AS TIMESTAMPTZ) DESC, run_id DESC"
      ).fetchall()
      return [
        dict(zip(("run_id", "created_at", "tracked"), row, strict=True)) for row in rows
      ]
  except Exception as exc:
    raise RunStoreError("Could not list saved forecasts.") from exc


def load_run(database_path: Path, run_id: str) -> RunArtifact:
  """Read one saved forecast, including a tracked vintage outside recent history."""
  if not database_path.exists():
    raise RunStoreError("The saved forecast was not found.")
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      row = connection.execute(
        "SELECT * FROM runs WHERE run_id = ?", [run_id]
      ).fetchone()
      if row is None:
        raise RunStoreError("The saved forecast was not found.")
      return _run_from_row(connection, row)
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError("Could not load the saved forecast.") from exc


def set_run_tracked(
  database_path: Path, run_id: str, tracked: bool, limit: int = MAX_SAVED_RUNS
) -> None:
  """Protect a saved forecast from retention, or return it to ordinary retention."""
  if limit < 1:
    raise ValueError("Run retention limit must be positive.")
  if not database_path.exists():
    raise RunStoreError("Save the forecast before tracking it.")
  try:
    with duckdb.connect(str(database_path)) as connection:
      connection.execute("BEGIN TRANSACTION")
      _create_schema(connection)
      _require_run(connection, run_id)
      if tracked:
        connection.execute(
          "INSERT INTO tracked_runs VALUES (?, ?) ON CONFLICT DO NOTHING",
          [run_id, pd.Timestamp.now(tz="UTC").isoformat()],
        )
      else:
        connection.execute("DELETE FROM tracked_runs WHERE run_id = ?", [run_id])
        _prune_runs(connection, limit)
      connection.execute("COMMIT")
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError("Could not update forecast tracking.") from exc


def _require_run(connection: duckdb.DuckDBPyConnection, run_id: str) -> None:
  if (
    connection.execute("SELECT run_id FROM runs WHERE run_id = ?", [run_id]).fetchone()
    is None
  ):
    raise RunStoreError("The saved forecast was not found.")


def link_runs(
  database_path: Path,
  run_id: str,
  previous_run_id: str,
  associations: dict[str, str],
) -> None:
  """Record a refresh relationship and protect both immutable forecast vintages."""
  if run_id == previous_run_id:
    raise RunStoreError("A refreshed forecast must have a new run ID.")
  try:
    with duckdb.connect(str(database_path)) as connection:
      connection.execute("BEGIN TRANSACTION")
      _create_schema(connection)
      _require_run(connection, run_id)
      _require_run(connection, previous_run_id)
      existing = connection.execute(
        "SELECT previous_run_id, associations FROM tracking_links WHERE run_id = ?",
        [run_id],
      ).fetchone()
      if existing and (
        existing[0] != previous_run_id or _load_json(existing[1]) != associations
      ):
        raise RunStoreError("A forecast's previous-run association cannot be changed.")
      now = pd.Timestamp.now(tz="UTC").isoformat()
      connection.execute(
        "INSERT INTO tracking_links VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
        [run_id, previous_run_id, json.dumps(associations), now],
      )
      for identifier in (run_id, previous_run_id):
        connection.execute(
          "INSERT INTO tracked_runs VALUES (?, ?) ON CONFLICT DO NOTHING",
          [identifier, now],
        )
      connection.execute("COMMIT")
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError("Could not link forecast vintages.") from exc


def save_assessment(database_path: Path, assessment: TrackingAssessment) -> None:
  """Append one version of matched actuals without updating the issued forecast."""
  try:
    with duckdb.connect(str(database_path)) as connection:
      connection.execute("BEGIN TRANSACTION")
      _create_schema(connection)
      _require_run(connection, assessment.run_id)
      # Unlike save_run's strict immutability check, a (run_id,
      # fingerprint) conflict here is a silent no-op regardless of
      # whether the conflicting row's content actually matches --
      # `fingerprint` is trusted to already capture content identity, so
      # this is a dedup-by-fingerprint, not a content comparison.
      connection.execute(
        "INSERT INTO run_assessments VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (run_id, fingerprint) DO NOTHING",
        [
          assessment.assessment_id,
          assessment.run_id,
          assessment.created_at,
          assessment.fingerprint,
          json.dumps(assessment.manifest, default=str),
          _table_to_parquet(assessment.comparisons),
          _table_to_parquet(assessment.metrics),
        ],
      )
      # Saving an assessment permanently protects its run from
      # `_prune_runs` (matching link_runs's behavior), since a run with
      # recorded actuals shouldn't silently disappear from history.
      connection.execute(
        "INSERT INTO tracked_runs VALUES (?, ?) ON CONFLICT DO NOTHING",
        [assessment.run_id, assessment.created_at],
      )
      connection.execute("COMMIT")
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError("Could not save the actuals assessment.") from exc


def load_assessments(database_path: Path, run_id: str) -> list[TrackingAssessment]:
  """Load successive actuals versions, oldest first, without a model or uploads."""
  # Deferred import (matching the TYPE_CHECKING-only import above): keeps
  # this lightweight persistence module from paying tracking.py's import
  # cost unless this function is actually called.
  from .tracking import TrackingAssessment

  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      if not _table_exists(connection, "run_assessments"):
        return []
      rows = connection.execute(
        "SELECT assessment_id, run_id, created_at, fingerprint, manifest, "
        "comparisons, metrics FROM run_assessments WHERE run_id = ? "
        "ORDER BY CAST(created_at AS TIMESTAMPTZ), assessment_id",
        [run_id],
      ).fetchall()
      return [
        TrackingAssessment(
          assessment_id=row[0],
          run_id=row[1],
          created_at=row[2],
          fingerprint=row[3],
          manifest=_load_json(row[4]),
          comparisons=_table_from_parquet(row[5]),
          metrics=_table_from_parquet(row[6]),
        )
        for row in rows
      ]
  except Exception as exc:
    raise RunStoreError("Could not load actuals assessments.") from exc


def save_analysis(
  database_path: Path, artifact: AnalysisArtifact, limit: int = 25
) -> None:
  """Save derived analysis tables and prune analysis history transactionally."""
  if limit < 1:
    raise ValueError("Analysis retention limit must be positive.")
  try:
    predictions = _table_to_parquet(artifact.predictions)
    metrics = _table_to_parquet(artifact.metrics)
    manifest = json.dumps(artifact.manifest, default=str)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(database_path)) as connection:
      connection.execute("BEGIN TRANSACTION")
      _create_schema(connection)
      connection.execute(
        """
        CREATE TABLE IF NOT EXISTS analysis_runs (
          analysis_id VARCHAR PRIMARY KEY,
          created_at VARCHAR NOT NULL,
          kind VARCHAR NOT NULL,
          manifest JSON NOT NULL,
          predictions BLOB NOT NULL,
          metrics BLOB NOT NULL
        )
        """
      )
      connection.execute(
        "INSERT OR REPLACE INTO analysis_runs VALUES (?, ?, ?, ?, ?, ?)",
        [
          artifact.analysis_id,
          artifact.created_at,
          artifact.kind,
          manifest,
          predictions,
          metrics,
        ],
      )
      connection.execute(
        """
        DELETE FROM analysis_runs WHERE analysis_id IN (
          SELECT analysis_id FROM analysis_runs
          ORDER BY CAST(created_at AS TIMESTAMPTZ) DESC, analysis_id DESC OFFSET ?
        )
        """,
        [limit],
      )
      connection.execute("COMMIT")
  except Exception as exc:
    raise RunStoreError("Could not save local analysis history.") from exc


def list_analyses(database_path: Path, limit: int = 25) -> list[dict[str, Any]]:
  """List newest analysis summaries without loading their stored tables."""
  if limit < 1:
    raise ValueError("Analysis history limit must be positive.")
  if not database_path.exists():
    return []
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      exists = connection.execute(
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = 'analysis_runs'"
      ).fetchone()
      if not exists or not exists[0]:
        return []
      rows = connection.execute(
        "SELECT analysis_id, created_at, kind FROM analysis_runs "
        "ORDER BY CAST(created_at AS TIMESTAMPTZ) DESC, analysis_id DESC LIMIT ?",
        [limit],
      ).fetchall()
      return [
        dict(zip(("analysis_id", "created_at", "kind"), row, strict=True))
        for row in rows
      ]
  except Exception as exc:
    raise RunStoreError("Could not list local analysis history.") from exc


def load_analysis(database_path: Path, analysis_id: str) -> AnalysisArtifact:
  """Load one saved analysis without changing the database or loading a model."""
  # Deferred import: see load_assessments's comment on the equivalent
  # tracking.py import above.
  from .analysis import AnalysisArtifact

  if not database_path.exists():
    raise RunStoreError("Saved analysis was not found.")
  try:
    with duckdb.connect(str(database_path), read_only=True) as connection:
      row = connection.execute(
        "SELECT analysis_id, created_at, kind, manifest, predictions, metrics "
        "FROM analysis_runs WHERE analysis_id = ?",
        [analysis_id],
      ).fetchone()
      if row is None:
        raise RunStoreError("Saved analysis was not found.")
      return AnalysisArtifact(
        analysis_id=row[0],
        created_at=row[1],
        kind=row[2],
        manifest=_load_json(row[3]),
        predictions=_table_from_parquet(row[4]),
        metrics=_table_from_parquet(row[5]),
      )
  except RunStoreError:
    raise
  except Exception as exc:
    raise RunStoreError("Could not load local analysis history.") from exc
