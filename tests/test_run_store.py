# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for local DuckDB run persistence."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from timesfm3.explorer import DatasetMapping, ForecastSettings, RunArtifact
from timesfm3.run_store import RunStoreError, load_recent_runs, save_run


def artifact(index: int) -> RunArtifact:
  run_id = f"run-{index:02d}"
  created_at = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=index)
  forecast = pd.DataFrame(
    {
      "dataset": ["dataset_1"],
      "target": ["sales"],
      "step": [1],
      "timestamp": [created_at],
      "point": [float(index)],
      "actual": [float(index + 1)],
      "q0.1": [float(index - 1)],
      "q0.9": [float(index + 1)],
    }
  )
  metrics = pd.DataFrame(
    {
      "dataset": ["dataset_1"],
      "target": ["sales"],
      "observations": [1],
      "mae": [1.0],
      "rmse": [1.0],
      "smape_percent": [1.0],
    }
  )
  settings = ForecastSettings(horizon=1, context_length=2)
  mapping = DatasetMapping("date", ("sales",))
  return RunArtifact(
    run_id=run_id,
    created_at=created_at.isoformat(),
    settings=settings,
    mapping=mapping,
    history=pd.DataFrame(),
    forecast=forecast,
    metrics=metrics,
    manifest={
      "run_id": run_id,
      "settings": dataclasses.asdict(settings),
      "mapping": dataclasses.asdict(mapping),
      "datasets": [{"sha256": "abc"}],
    },
    runtime_seconds=0.5,
    device="cpu",
  )


def test_run_round_trip_preserves_derived_results(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  save_run(database, artifact(1))

  loaded = load_recent_runs(database)

  assert len(loaded) == 1
  assert loaded[0].run_id == "run-01"
  assert loaded[0].settings.horizon == 1
  assert loaded[0].mapping.targets == ("sales",)
  assert loaded[0].manifest["datasets"][0]["sha256"] == "abc"
  assert loaded[0].forecast.loc[0, "q0.1"] == 0.0
  assert loaded[0].metrics.loc[0, "mae"] == 1.0
  assert loaded[0].history.empty


def test_run_retention_deletes_oldest_rows(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  for index in range(26):
    save_run(database, artifact(index))

  loaded = load_recent_runs(database)

  assert len(loaded) == 25
  assert loaded[0].run_id == "run-01"
  assert loaded[-1].run_id == "run-25"


def test_store_errors_are_safe_for_in_memory_fallback(tmp_path: Path) -> None:
  database_directory = tmp_path / "not-a-database"
  database_directory.mkdir()
  with pytest.raises(RunStoreError, match="Could not save local run history"):
    save_run(database_directory, artifact(1))
