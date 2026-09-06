"""Derived analysis history round trips without touching uploaded data."""

from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from timesfm3.analysis import AnalysisArtifact
from timesfm3.run_store import (
  RunStoreError,
  list_analyses,
  load_analysis,
  load_recent_runs,
  save_analysis,
)


def artifact(index: int, kind: str = "backtest") -> AnalysisArtifact:
  timestamp = pd.Timestamp("2026-01-01", tz="Asia/Kolkata") + pd.Timedelta(
    minutes=index
  )
  return AnalysisArtifact(
    analysis_id=f"analysis-{index:02}",
    created_at=timestamp.isoformat(),
    kind=kind,
    predictions=pd.DataFrame({"timestamp": [timestamp], "point": [float(index)]}),
    metrics=pd.DataFrame({"target": ["sales"], "mae": [1.0]}),
    manifest={"input_hashes": ["abc"], "scenario_overrides": [{"price": 3}]},
  )


@pytest.mark.parametrize(
  "kind", ["anomaly", "scenario", "backtest", "joint_independent", "covariates"]
)
def test_analysis_round_trip(tmp_path: Path, kind: str) -> None:
  database = tmp_path / "history.duckdb"
  original = artifact(1, kind)
  save_analysis(database, original)
  assert load_recent_runs(database) == []
  loaded = load_analysis(database, original.analysis_id)
  assert loaded.analysis_id == original.analysis_id
  assert loaded.created_at == original.created_at
  assert loaded.kind == kind
  assert loaded.manifest == original.manifest
  pd.testing.assert_frame_equal(loaded.predictions, original.predictions)
  pd.testing.assert_frame_equal(loaded.metrics, original.metrics)
  assert list_analyses(database) == [
    {
      "analysis_id": original.analysis_id,
      "created_at": original.created_at,
      "kind": kind,
    }
  ]


def test_retention_overwrite_and_empty_metrics(tmp_path: Path) -> None:
  database = tmp_path / "history.duckdb"
  for index in range(26):
    save_analysis(database, artifact(index))
  summaries = list_analyses(database)
  assert len(summaries) == 25
  assert summaries[0]["analysis_id"] == "analysis-25"
  assert summaries[-1]["analysis_id"] == "analysis-01"
  updated = replace(artifact(25), metrics=pd.DataFrame(), manifest={"updated": True})
  save_analysis(database, updated)
  assert len(list_analyses(database)) == 25
  loaded = load_analysis(database, updated.analysis_id)
  assert loaded.manifest == {"updated": True}
  assert loaded.metrics.empty
  assert list(loaded.metrics.columns) == []
  with pytest.raises(RunStoreError):
    load_analysis(database, "analysis-00")


def test_old_database_and_missing_database_are_read_only(tmp_path: Path) -> None:
  database = tmp_path / "history.duckdb"
  assert list_analyses(database) == []
  with pytest.raises(RunStoreError):
    load_analysis(database, "missing")
  assert not database.exists()
  with duckdb.connect(str(database)) as connection:
    connection.execute("CREATE TABLE runs (run_id VARCHAR)")
    connection.execute("INSERT INTO runs VALUES ('existing')")
  original = database.read_bytes()
  assert list_analyses(database) == []
  with pytest.raises(RunStoreError):
    load_analysis(database, "missing")
  assert database.read_bytes() == original
  save_analysis(database, artifact(1))
  with duckdb.connect(str(database), read_only=True) as connection:
    assert connection.execute("SELECT * FROM runs").fetchall() == [("existing",)]


def test_invalid_path_and_corruption_have_safe_errors(tmp_path: Path) -> None:
  for database in (tmp_path, tmp_path / "corrupt.duckdb"):
    if database != tmp_path:
      database.write_bytes(b"not a database")
    with pytest.raises(RunStoreError):
      save_analysis(database, artifact(1))
    with pytest.raises(RunStoreError):
      list_analyses(database)
    with pytest.raises(RunStoreError):
      load_analysis(database, "missing")


def test_failed_save_rolls_back_overwrite_and_retention(tmp_path: Path) -> None:
  database = tmp_path / "history.duckdb"
  save_analysis(database, artifact(1))
  with pytest.raises(RunStoreError):
    save_analysis(database, replace(artifact(1), created_at="invalid date"), limit=1)
  assert load_analysis(database, "analysis-01").created_at == artifact(1).created_at


def test_retention_orders_absolute_time_across_offsets(tmp_path: Path) -> None:
  database = tmp_path / "history.duckdb"
  earlier = replace(artifact(1), created_at="2026-01-01T05:30:00+05:30")
  later = replace(artifact(2), created_at="2026-01-01T01:00:00+00:00")
  save_analysis(database, earlier, limit=1)
  save_analysis(database, later, limit=1)
  assert [row["analysis_id"] for row in list_analyses(database)] == [later.analysis_id]
