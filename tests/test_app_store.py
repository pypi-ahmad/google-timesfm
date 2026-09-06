"""Metadata invariants and immutable, portable artifacts."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from timesfm_app.artifacts import ArtifactStore
from timesfm_app.store import Conflict, NotFound, Store


@pytest.fixture
def store():
  instance = Store.for_testing()
  instance.initialize()
  instance.create_record("workspace", "Local", {}, record_id="local")
  yield instance
  instance.close()


def test_production_store_rejects_sqlite():
  with pytest.raises(ValueError, match="PostgreSQL"):
    Store("sqlite:///:memory:")


def test_records_are_workspace_scoped_and_not_found_is_explicit(store):
  local = store.create_record("dataset", "Sales", {"columns": ["sales"]})
  store.create_record("workspace", "Other", {}, workspace_id="other", record_id="other")
  store.create_record("dataset", "Private", {}, workspace_id="other")
  assert store.list_records("dataset", "local") == [local]
  assert store.get_record(local["id"], "dataset") == local
  with pytest.raises(NotFound):
    store.get_record(local["id"], "run")
  with pytest.raises(NotFound):
    store.get_record("missing")


def test_stale_draft_cannot_overwrite_a_saved_revision(store):
  draft = store.create_record("draft", "Forecast", {"horizon": 4})
  updated = store.update_record(draft["id"], {"horizon": 8}, 1)
  assert updated["revision"] == 2
  with pytest.raises(Conflict, match="changed elsewhere"):
    store.update_record(draft["id"], {"horizon": 16}, 1)
  assert store.get_record(draft["id"])["payload"]["horizon"] == 8


def test_record_rename_shares_the_revision_guard_with_payload(store):
  draft = store.create_record("draft", "Original", {"horizon": 4})
  renamed = store.update_record(draft["id"], {"horizon": 8}, 1, name="Renamed")
  assert renamed["name"] == "Renamed" and renamed["revision"] == 2
  with pytest.raises(Conflict):
    store.update_record(draft["id"], {"horizon": 16}, 1, name="Stale rename")
  assert store.get_record(draft["id"]) == renamed


def test_immutable_versions_and_referenced_records_are_protected(store):
  dataset = store.create_record("dataset", "Sales", {})
  version = store.create_record("version", "Version one", {"dataset_id": dataset["id"]})
  with pytest.raises(Conflict, match="immutable"):
    store.update_record(version["id"], {}, 1)
  with pytest.raises(Conflict, match="depend"):
    store.delete_record(dataset["id"])
  store.delete_record(version["id"])
  store.delete_record(dataset["id"])
  assert store.list_records("dataset") == []


def test_job_reference_blocks_record_deletion(store):
  version = store.create_record("version", "Version one", {})
  store.create_job("local", "forecast", {"version_ids": [version["id"]]}, "request")
  with pytest.raises(Conflict, match="history"):
    store.delete_record(version["id"])


@pytest.mark.parametrize(
  "key",
  [
    "../outside",
    "/absolute",
    "C:/outside",
    "a/../../outside",
    "a\\outside",
    "a//file",
    "a/./file",
    "a/file:stream",
    "NUL.txt",
    "a/CON",
    "file.",
    "file ",
  ],
)
def test_artifact_keys_cannot_escape_or_alias_windows_paths(tmp_path, key):
  artifacts = ArtifactStore(tmp_path)
  with pytest.raises(ValueError):
    artifacts.put_bytes(key, b"data")


def test_artifacts_are_immutable_and_leave_no_partial_files(tmp_path):
  artifacts = ArtifactStore(tmp_path)
  descriptor = artifacts.put_bytes("job/attempt-1/manifest.json", b"complete")
  assert descriptor["size"] == 8
  assert len(descriptor["sha256"]) == 64
  assert artifacts.put_bytes(descriptor["key"], b"complete") == descriptor
  with pytest.raises(Conflict):
    artifacts.put_bytes(descriptor["key"], b"changed")
  assert artifacts.get_bytes(descriptor["key"]) == b"complete"
  assert not list(tmp_path.rglob(".pending-*"))


def test_concurrent_artifact_publication_accepts_only_one_payload(tmp_path):
  artifacts = ArtifactStore(tmp_path)

  def publish(data):
    try:
      artifacts.put_bytes("attempt/result", data)
      return data
    except Conflict:
      return None

  with ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(publish, [b"first", b"second"]))
  assert sum(result is not None for result in results) == 1
  assert artifacts.get_bytes("attempt/result") in results


@pytest.mark.skipif(os.name != "nt", reason="Windows extended path normalization")
def test_artifact_containment_normalizes_windows_extended_paths(tmp_path, monkeypatch):
  artifacts = ArtifactStore(tmp_path)
  original = Path.resolve

  def resolve(path, *args, **kwargs):
    resolved = original(path, *args, **kwargs)
    return Path("\\\\?\\" + str(resolved)) if path.name == "result" else resolved

  monkeypatch.setattr(Path, "resolve", resolve)
  artifacts.put_bytes("attempt/result", b"complete")
  assert artifacts.get_bytes("attempt/result") == b"complete"


def test_parquet_preserves_mixed_temporal_and_numeric_timestamps(tmp_path):
  artifacts = ArtifactStore(tmp_path)
  frame = pd.DataFrame(
    {
      "timestamp": pd.Series(
        [
          pd.Timestamp("2026-01-01", tz="Asia/Kolkata"),
          pd.Timestamp("2026-02-01", tz="America/New_York"),
          pd.Timestamp("2026-03-01"),
          4,
          "label",
          None,
          pd.NaT,
        ],
        dtype=object,
      ),
      "point": [1.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
  )
  artifacts.put_frame("job/attempt-1/forecast.parquet", frame)
  restored = artifacts.get_frame("job/attempt-1/forecast.parquet")
  pd.testing.assert_frame_equal(frame, restored)
  assert str(restored.iloc[0]["timestamp"].tz) == "Asia/Kolkata"


def test_parquet_preserves_homogeneous_timezone_column(tmp_path):
  artifacts = ArtifactStore(tmp_path)
  frame = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=2, tz="UTC")})
  artifacts.put_frame("frame.parquet", frame)
  pd.testing.assert_frame_equal(frame, artifacts.get_frame("frame.parquet"))


def test_alembic_bootstrap_matches_declared_schema(tmp_path):
  from alembic import command
  from alembic.autogenerate import compare_metadata
  from alembic.config import Config
  from alembic.migration import MigrationContext

  from timesfm_app.store import Base

  root = Path(__file__).parents[1]
  config = Config(str(root / "alembic.ini"))
  url = f"sqlite+pysqlite:///{tmp_path / 'migration.sqlite'}"
  config.set_main_option("sqlalchemy.url", url)
  command.upgrade(config, "head")
  migrated = Store.for_testing(url)
  try:
    with migrated.engine.connect() as connection:
      assert (
        compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
      )
    migrated.create_record("workspace", "Local", {}, record_id="local")
    assert migrated.create_record("draft", "After migration", {})["revision"] == 1
  finally:
    migrated.close()


def test_audit_is_atomic_and_does_not_create_deletion_dependencies(store, monkeypatch):
  draft = store.create_record("draft", "Forecast", {"horizon": 8})
  store.update_record(draft["id"], {"horizon": 16}, 1)
  store.delete_record(draft["id"])
  actions = [
    row["payload"]["action"]
    for row in reversed(store.list_records("audit"))
    if row["payload"].get("entity_id") == draft["id"]
  ]
  assert actions == ["record.created", "record.updated", "record.deleted"]
  original = store._audit

  def failed_audit(*args, **kwargs):
    original(*args, **kwargs)
    raise RuntimeError("Audit transaction failed")

  before = store.list_records("audit")
  monkeypatch.setattr(store, "_audit", failed_audit)
  with pytest.raises(RuntimeError):
    store.create_record("dataset", "Must roll back", {})
  assert store.list_records("dataset") == []
  assert store.list_records("audit") == before
  monkeypatch.setattr(store, "_audit", original)
  assert store.deletion_blockers("local") == []
  store.delete_record("local")
  assert store.list_records("audit")
