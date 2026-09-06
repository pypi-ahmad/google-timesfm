"""Legacy import preserves numerical reports without mutating DuckDB."""

import hashlib
import json
import os
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from io import BytesIO

import pandas as pd
import pytest

from timesfm3.analysis import AnalysisArtifact
from timesfm3.explorer import DatasetMapping, ForecastSettings, RunArtifact
from timesfm3.run_store import link_runs, save_analysis, save_assessment, save_run
from timesfm3.tracking import TrackingAssessment
from timesfm_app.artifacts import ArtifactStore
from timesfm_app.migration import import_id, import_legacy
from timesfm_app.store import Store


@pytest.fixture
def store():
  instance = Store.for_testing()
  instance.initialize()
  instance.create_record("workspace", "Local", {}, record_id="local")
  yield instance
  instance.close()


def forecast(identifier="legacy-forecast"):
  return RunArtifact(
    run_id=identifier,
    created_at="2026-01-01T00:00:00+00:00",
    settings=ForecastSettings(horizon=2, context_length=4),
    mapping=DatasetMapping("date", ("sales",)),
    history=pd.DataFrame({"private": [12345]}),
    forecast=pd.DataFrame(
      {
        "dataset": ["east", "west"],
        "target": ["sales", "sales"],
        "step": [1, 1],
        "timestamp": pd.Series(
          [
            pd.Timestamp("2026-01-02", tz="Asia/Kolkata"),
            pd.Timestamp("2026-01-02", tz="America/New_York"),
          ],
          dtype=object,
        ),
        "point": [10.25, 12.75],
        "q0.1": [8.0, 10.0],
        "q0.9": [12.0, 14.0],
      }
    ),
    metrics=pd.DataFrame(),
    manifest={"schema_version": 2},
    runtime_seconds=0.5,
    device="cpu",
  )


def test_legacy_import_twice_preserves_timezones_assessments_links_and_source(
  tmp_path, store
):
  source = tmp_path / "legacy.duckdb"
  original = forecast()
  newer = replace(original, run_id="legacy-refreshed")
  save_run(source, original)
  save_run(source, newer)
  link_runs(source, newer.run_id, original.run_id, {"east": "east", "west": "west"})
  assessment = TrackingAssessment(
    assessment_id="assessment-v1",
    run_id=original.run_id,
    created_at="2026-02-01T00:00:00+00:00",
    fingerprint="actuals-version-hash",
    comparisons=original.forecast.assign(actual=[11.0, 13.0]),
    metrics=pd.DataFrame({"target": ["sales"], "mae": [0.5]}),
    manifest={"run_id": original.run_id, "actual_hashes": ["original-actuals"]},
  )
  save_assessment(source, assessment)
  analysis = AnalysisArtifact(
    analysis_id="backtest-v1",
    created_at="2026-01-01T00:00:00+00:00",
    kind="backtest",
    predictions=original.forecast,
    metrics=pd.DataFrame(),
    manifest={"windows": 2},
  )
  save_analysis(source, analysis)
  source_before = hashlib.sha256(source.read_bytes()).hexdigest()
  artifacts = ArtifactStore(tmp_path / "artifacts")
  first = import_legacy(source, store, artifacts)
  second = import_legacy(source, store, artifacts)
  assert first["created"] == 6
  assert second["created"] == 0
  assert second["skipped"] == 6
  assert hashlib.sha256(source.read_bytes()).hexdigest() == source_before
  identifier = import_id(source, "local", "forecast", original.run_id)
  result = store.get_record(identifier)["payload"]
  pd.testing.assert_frame_equal(
    artifacts.get_frame(result["tables"]["forecast"]["key"]), original.forecast
  )
  assert artifacts.get_frame(result["tables"]["history"]["key"]).empty
  assert result["source_data_available"] is False
  assert result["manifest"]["legacy_id"] == original.run_id
  assert result["manifest"]["history_available"] is False
  assessed = store.get_record(
    import_id(source, "local", "assessment", "assessment-v1")
  )["payload"]
  pd.testing.assert_frame_equal(
    artifacts.get_frame(assessed["tables"]["comparisons"]["key"]),
    assessment.comparisons,
  )
  linked = next(
    item["payload"]
    for item in store.list_records("tracking")
    if "previous_run_id" in item["payload"]
  )
  assert linked["previous_run_id"] == identifier
  assert linked["auto_refresh"] is False
  with zipfile.ZipFile(
    BytesIO(artifacts.get_bytes(result["export"]["key"]))
  ) as archive:
    assert json.loads(archive.read("run.json"))["legacy_id"] == original.run_id
    assert "private" not in " ".join(archive.namelist())


def test_import_enumerates_more_than_default_25_analyses(tmp_path, store):
  source = tmp_path / "legacy.duckdb"
  for index in range(26):
    save_analysis(
      source,
      AnalysisArtifact(
        analysis_id=f"analysis-{index}",
        created_at="2026-01-01T00:00:00+00:00",
        kind="backtest",
        predictions=pd.DataFrame({"timestamp": [index], "point": [1.0]}),
        metrics=pd.DataFrame(),
        manifest={"index": index},
      ),
      limit=100,
    )
  result = import_legacy(source, store, ArtifactStore(tmp_path / "artifacts"))
  assert result["analyses"] == 26
  assert len(store.list_records("run")) == 26


def test_missing_source_is_not_created(tmp_path, store):
  source = tmp_path / "missing.duckdb"
  with pytest.raises(FileNotFoundError):
    import_legacy(source, store, ArtifactStore(tmp_path / "artifacts"))
  assert not source.exists()


def test_import_resumes_after_artifacts_written_before_record_commit(
  tmp_path, store, monkeypatch
):
  source = tmp_path / "legacy.duckdb"
  save_run(source, forecast())
  artifacts = ArtifactStore(tmp_path / "artifacts")
  original = store.create_record

  def unavailable(*args, **kwargs):
    raise ConnectionError("Database disconnected before result publication.")

  monkeypatch.setattr(store, "create_record", unavailable)
  with pytest.raises(ConnectionError):
    import_legacy(source, store, artifacts)
  assert list(artifacts.root.rglob("export.zip"))
  assert store.list_records("run") == []
  monkeypatch.setattr(store, "create_record", original)
  assert import_legacy(source, store, artifacts)["created"] == 1


def test_stable_import_ids_are_workspace_and_entity_scoped(tmp_path):
  source = tmp_path / "legacy.duckdb"
  identifiers = {
    import_id(source, "one", "forecast", "old"),
    import_id(source, "two", "forecast", "old"),
    import_id(source, "one", "assessment", "old"),
  }
  assert len(identifiers) == 3
  assert all(len(identifier) == 32 for identifier in identifiers)


def test_retention_is_opt_in_and_protects_tracked_forecasts(store):
  from timesfm_app.maintenance import apply_retention, retention_preview

  old = store.create_record(
    "run", "Old", {"manifest": {"created_at": "2020-01-01T00:00:00Z"}}
  )
  tracked = store.create_record(
    "run", "Tracked", {"manifest": {"created_at": "2020-01-01T00:00:00Z"}}
  )
  store.create_record("tracking", "Tracking", {"run_id": tracked["id"]})
  assert retention_preview(store)["enabled"] is False
  workspace = store.get_record("local")
  store.update_record(
    "local", {"retention": {"enabled": True, "max_age_days": 30}}, workspace["revision"]
  )
  preview = retention_preview(store)
  assert [item["id"] for item in preview["candidates"]] == [old["id"]]
  assert [item["id"] for item in preview["protected"]] == [tracked["id"]]
  result = apply_retention(store)
  assert result["removed"] == [old["id"]]
  assert store.get_record(tracked["id"])["name"] == "Tracked"


def test_orphan_cleanup_excludes_references_current_attempt_and_fresh_files(
  tmp_path, store
):
  from datetime import datetime, timedelta, timezone

  from timesfm_app.maintenance import cleanup_orphans

  artifacts = ArtifactStore(tmp_path / "artifacts")
  job = store.create_job("local", "forecast", {}, "request")
  store.claim_job(job["id"])
  active_key = f"jobs/{job['id']}/attempt-1/forecast.parquet"
  referenced = artifacts.put_bytes("imports/example/saved", b"saved")
  store.create_record("run", "Saved", {"artifact": referenced})
  for key in (active_key, "jobs/abandoned/attempt-1/orphan", "datasets/fresh"):
    artifacts.put_bytes(key, b"data")
  artifacts.put_bytes("model_snapshots/unmanaged", b"model")
  old = (datetime.now(timezone.utc) - timedelta(hours=25)).timestamp()
  for path in artifacts.root.rglob("*"):
    if path.is_file() and path.name != "fresh":
      os.utime(path, (old, old))
  preview = cleanup_orphans(store, artifacts)
  assert preview["candidates"] == ["jobs/abandoned/attempt-1/orphan"]
  assert preview["removed"] == []
  assert (
    cleanup_orphans(store, artifacts, apply=True)["removed"] == preview["candidates"]
  )
  assert artifacts.get_bytes(active_key) == b"data"
  assert artifacts.get_bytes(referenced["key"]) == b"saved"
  assert artifacts.get_bytes("datasets/fresh") == b"data"


def test_orphan_apply_waits_until_legacy_import_has_published(
  tmp_path, store, monkeypatch
):
  from timesfm_app.maintenance import cleanup_orphans

  source = tmp_path / "legacy.duckdb"
  save_run(source, forecast())
  artifacts = ArtifactStore(tmp_path / "artifacts")
  entered, release = threading.Event(), threading.Event()
  cleanup_started, cleanup_finished = threading.Event(), threading.Event()
  original = artifacts.put_frame

  def paused_put(*args, **kwargs):
    entered.set()
    assert release.wait(5)
    return original(*args, **kwargs)

  def clean():
    cleanup_started.set()
    result = cleanup_orphans(store, artifacts, apply=True)
    cleanup_finished.set()
    return result

  monkeypatch.setattr(artifacts, "put_frame", paused_put)
  with ThreadPoolExecutor(max_workers=2) as pool:
    importing = pool.submit(import_legacy, source, store, artifacts)
    assert entered.wait(3)
    cleaning = pool.submit(clean)
    try:
      assert cleanup_started.wait(2)
      assert not cleanup_finished.wait(0.1)
    finally:
      release.set()
    assert importing.result(timeout=5)["created"] == 1
    assert cleaning.result(timeout=5)["removed"] == []
