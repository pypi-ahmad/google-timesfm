"""Parity and process-boundary checks for the durable application shell."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from timesfm3 import ForecastOutput, analysis, explorer, tracking
from timesfm_app import services
from timesfm_app.artifacts import ArtifactStore
from timesfm_app.store import Store


class FakePredictor:
  """Test-only deterministic predictor; the application always loads real weights."""

  device = "cpu"

  def __init__(self):
    self.calls = []

  def predict_batch(self, **kwargs):
    self.calls.append(kwargs)
    results = []
    for identifier, context in zip(kwargs["ts_ids"], kwargs["contexts"], strict=True):
      point = np.repeat(np.atleast_2d(context)[:, -1:], kwargs["horizon"], axis=1)
      quantiles = None
      if kwargs["return_quantiles"]:
        quantiles = np.stack([point + shift for shift in np.linspace(-1, 1, 9)], -1)
      results.append(
        ForecastOutput(ts_id=identifier, forecast=point, quantiles=quantiles)
      )
    return results


@pytest.fixture
def inputs(tmp_path):
  store = Store.for_testing()
  store.initialize()
  store.create_record("workspace", "Local", {}, record_id="local")
  artifacts = ArtifactStore(tmp_path / "artifacts")
  frame = pd.DataFrame(
    {
      "date": pd.date_range("2026-01-01", periods=22),
      "a": np.r_[np.arange(20, dtype=float), np.nan, np.nan],
      "b": np.r_[np.arange(20, dtype=float) * 2, np.nan, np.nan],
      "past": np.arange(22, dtype=float),
      "known": np.arange(22, dtype=float) % 2,
    }
  )
  encoded = frame.to_csv(index=False).encode()
  descriptor = artifacts.put_bytes("uploads/original.csv", encoded)
  version = store.create_record(
    "dataset_version",
    "example",
    {
      "artifact": descriptor,
      "filename": "original.csv",
      "source_name": "example",
      "sha256": descriptor["sha256"],
    },
  )
  spec = {
    "kind": "forecast",
    "dataset_version_ids": [version["id"]],
    "mapping": {
      "timestamp": "date",
      "targets": ["a", "b"],
      "past_only": ["past"],
      "past_future": ["known"],
    },
    "settings": {"horizon": 2, "context_length": 6},
    "analysis": {"windows": 2, "seasonal_period": 2},
  }
  yield store, artifacts, spec
  store.close()


def execute(inputs, **changes):
  store, artifacts, spec = inputs
  return services.execute_spec(
    store,
    artifacts,
    {**spec, **changes},
    lambda *args: None,
    lambda: False,
    FakePredictor(),
  )


@pytest.mark.parametrize(
  "kind",
  [
    "forecast",
    "backtest",
    "joint_independent",
    "covariates",
    "settings",
    "baselines",
    "scenario",
    "anomaly",
  ],
)
def test_service_preserves_core_predictions_and_exports(inputs, kind):
  store, artifacts, spec = inputs
  spec = {**spec, "kind": kind}
  if kind == "settings":
    spec["configurations"] = [
      {"name": "short", "settings": {"context_length": 4}},
      {"name": "long", "settings": {"context_length": 6}},
    ]
  if kind == "scenario":
    spec["scenarios"] = [
      {
        "name": "promotion",
        "overrides": [
          {"dataset": "example", "row": 20, "covariate": "known", "value": 2.0}
        ],
      }
    ]
  datasets, mapping, settings = services.prepare_inputs(store, artifacts, spec)
  reference = FakePredictor()
  if kind == "forecast":
    expected = explorer.execute_forecast(reference, datasets, mapping, settings)
    expected_tables = {
      "forecast": expected.forecast,
      "metrics": expected.metrics,
      "history": expected.history,
    }
    expected_zip = explorer.artifact_zip(expected)
  else:
    scenarios = tuple(
      analysis.Scenario(
        item["name"],
        analysis.analysis_fingerprint(datasets, mapping, settings),
        tuple(analysis.ScenarioEdit(**edit) for edit in item["overrides"]),
      )
      for item in spec.get("scenarios", [])
    )
    configurations = tuple(
      analysis.ExperimentConfiguration(
        item["name"], dataclasses.replace(settings, **item["settings"])
      )
      for item in spec.get("configurations", [])
    )
    prepared = analysis.prepare_analysis(
      datasets,
      mapping,
      settings,
      analysis.AnalysisSettings(kind=kind, **spec["analysis"]),
      scenarios,
      configurations,
    )
    expected = analysis.run_analysis(reference, prepared)
    expected_tables = services._analysis_tables(expected)
    expected_zip = analysis.analysis_zip(expected)
  actual_predictor = FakePredictor()
  actual = services.execute_spec(
    store, artifacts, spec, lambda *args: None, lambda: False, actual_predictor
  )
  assert actual["kind"] == kind
  for name, table in expected_tables.items():
    restored = artifacts.get_frame(actual["tables"][name]["key"])
    if len(table.columns):
      pd.testing.assert_frame_equal(restored, table)
    else:
      assert restored.shape == table.shape
  assert len(actual_predictor.calls) == len(reference.calls)
  for actual_call, expected_call in zip(
    actual_predictor.calls, reference.calls, strict=True
  ):
    for name in expected_call:
      if name in {"contexts", "past_only_covariates", "past_future_covariates"}:
        for left, right in zip(actual_call[name], expected_call[name], strict=True):
          np.testing.assert_equal(left, right)
      else:
        assert actual_call[name] == expected_call[name]
  with (
    zipfile.ZipFile(io.BytesIO(expected_zip)) as before,
    zipfile.ZipFile(io.BytesIO(artifacts.get_bytes(actual["export"]["key"]))) as after,
  ):
    assert set(before.namelist()).issubset(after.namelist())
    assert {"inspection.json", "input_context.csv", "input_events.csv"}.issubset(
      after.namelist()
    )
    for name in before.namelist():
      if name.endswith(".csv"):
        assert before.read(name) == after.read(name)


def test_preview_never_resolves_model_and_is_json_safe(inputs, monkeypatch):
  def forbidden(*args, **kwargs):
    pytest.fail("Preview attempted model acquisition")

  monkeypatch.setattr(services.model_loading, "resolve_model", forbidden)
  result = services.preview_inputs(*inputs)
  assert result["quality"][0]["status"] == "ready"
  assert len(result["scenario_template"]) == 2
  assert result["series"][0]["dataset"] == "example"
  json.dumps(result, allow_nan=False)


def test_disabled_signals_change_preview_and_execution_without_losing_roles(inputs):
  store, artifacts, spec = inputs
  disabled = {**spec, "disabled_covariates": ["known"]}
  preview = services.preview_inputs(store, artifacts, disabled)
  assert preview["mapping"]["past_future"] == ()
  assert preview["scenario_template"] == []
  predictor = FakePredictor()
  result = services.execute_spec(
    store, artifacts, disabled, lambda *args: None, lambda: False, predictor
  )
  assert predictor.calls[0]["past_future_covariates"] == [None]
  assert spec["mapping"]["past_future"] == ["known"]
  assert result["manifest"]["input_windows"][0]["past_future_shape"] is None
  baseline = artifacts.get_frame(result["tables"]["baselines"]["key"])
  assert baseline.loc[baseline.target.eq("a"), "point"].tolist() == [19, 19]
  context = artifacts.get_frame(result["tables"]["input_context"]["key"])
  assert not (context.role.eq("target") & context.phase.eq("future")).any()


def test_input_limits_stop_decoding_before_remaining_versions(inputs, monkeypatch):
  store, artifacts, spec = inputs
  first = store.get_record(spec["dataset_version_ids"][0])["payload"]
  identifiers = list(spec["dataset_version_ids"])
  for name in ("second", "third"):
    identifiers.append(
      store.create_record("dataset_version", name, {**first, "source_name": name})["id"]
    )
  monkeypatch.setattr(
    explorer, "MAX_TOTAL_UPLOAD_BYTES", first["artifact"]["size"] * 1.5
  )
  parse = explorer.parse_upload
  decoded = []

  def recording_parser(*args):
    decoded.append(args[2])
    return parse(*args)

  monkeypatch.setattr(explorer, "parse_upload", recording_parser)
  with pytest.raises(explorer.ExplorerError, match="Combined uploads"):
    services.prepare_inputs(
      store, artifacts, {**spec, "dataset_version_ids": identifiers}
    )
  assert len(decoded) == 2


def test_cancellation_stops_between_analysis_calls_before_artifacts(inputs):
  store, artifacts, spec = inputs
  predictor = FakePredictor()
  with pytest.raises(services.CancellationRequested):
    services.execute_spec(
      store,
      artifacts,
      {**spec, "kind": "backtest"},
      lambda *args: None,
      lambda: len(predictor.calls) >= 1,
      predictor,
    )
  assert len(predictor.calls) == 1
  assert not (artifacts.root / "runs").exists()


def test_preflight_fails_before_model_resolution(inputs, monkeypatch):
  monkeypatch.setattr(
    services.model_loading,
    "resolve_model",
    lambda *args: pytest.fail("Invalid analysis loaded a model"),
  )
  store, artifacts, spec = inputs
  with pytest.raises(explorer.ExplorerError, match="Window"):
    services.execute_spec(
      store,
      artifacts,
      {**spec, "kind": "backtest", "analysis": {"windows": 0}},
      lambda *args: None,
      lambda: False,
    )


def test_tracking_matches_original_core_without_forecast_preparation(inputs):
  store, artifacts, spec = inputs
  issued = execute(inputs)
  record = store.create_record("run", "Issued", {**issued, "spec": spec})
  raw = pd.read_csv(io.BytesIO(artifacts.get_bytes("uploads/original.csv")))
  raw.loc[20:, "a"] = [20, 21]
  raw.loc[20:, "b"] = [40, 42]
  # Tracking only needs actual targets and timestamps, not future covariates.
  raw = raw[["date", "a", "b"]]
  descriptor = artifacts.put_bytes(
    "uploads/actuals.csv", raw.to_csv(index=False).encode()
  )
  version = store.create_record(
    "dataset_version",
    "updated",
    {
      "artifact": descriptor,
      "filename": "actuals.csv",
      "source_name": "updated",
    },
  )
  assessment_spec = {
    **spec,
    "kind": "assessment",
    "parent_run_id": record["id"],
    "dataset_version_ids": [version["id"]],
    "associations": {"example": "updated"},
  }
  result = execute((store, artifacts, assessment_spec))
  run = services._saved_run(store, artifacts, assessment_spec)
  sources, _, _ = services._group_inputs(store, artifacts, assessment_spec)
  expected = tracking.assess_run(run, sources, assessment_spec["associations"])
  pd.testing.assert_frame_equal(
    artifacts.get_frame(result["tables"]["comparisons"]["key"]), expected.comparisons
  )


def test_frozen_local_model_survives_source_mutation(tmp_path):
  source = tmp_path / "source.safetensors"
  source.write_bytes(b"original weights")
  captured = []

  def freeze(value):
    captured.append(value)
    return value

  spec = {
    "model": {"source": str(source), "kind": "local", "offline": True},
    "_model_snapshot_root": str(tmp_path / "snapshots"),
  }
  original = services.resolve_frozen_model(spec, freeze)
  source.write_bytes(b"changed weights")
  restored = services.resolve_frozen_model({**spec, "_resolved_model": captured[0]})
  assert restored == original
  assert Path(restored.path).read_bytes() == b"original weights"
  Path(restored.path).write_bytes(b"corruption")
  with pytest.raises(explorer.ExplorerError, match="integrity"):
    services.resolve_frozen_model({**spec, "_resolved_model": captured[0]})


def test_refresh_restores_saved_calendar_and_forecast_settings(inputs):
  from timesfm3.data_preparation import restore_preparation

  store, artifacts, original = inputs
  spec = {
    **original,
    "mapping": {"timestamp": "date", "targets": ["a", "b"]},
    "preparation": {"calendar": {"weekday": True}, "frequency": "D"},
  }
  issued = execute((store, artifacts, spec))
  parent = store.create_record("run", "Issued", {**issued, "spec": spec})
  raw = pd.read_csv(io.BytesIO(artifacts.get_bytes("uploads/original.csv")))
  raw.loc[20:, "a"] = [20, 21]
  raw.loc[20:, "b"] = [40, 42]
  descriptor = artifacts.put_bytes(
    "uploads/refresh.csv", raw.to_csv(index=False).encode()
  )
  version = store.create_record(
    "dataset_version",
    "Updated",
    {
      "artifact": descriptor,
      "filename": "refresh.csv",
      "source_name": "updated",
    },
  )
  refreshed_spec = {
    **spec,
    "parent_run_id": parent["id"],
    "dataset_version_ids": [version["id"]],
    "associations": {"example": "updated"},
    "settings": {"horizon": 1, "context_length": 2},
  }
  prepared, mapping, settings = services.prepare_inputs(
    store, artifacts, refreshed_spec
  )
  previous = services._saved_run(store, artifacts, refreshed_spec)
  raw_groups, _, _ = services._group_inputs(store, artifacts, refreshed_spec)
  associated = tracking.associated_datasets(
    previous, raw_groups, refreshed_spec["associations"]
  )
  expected, expected_mapping = restore_preparation(
    associated[0],
    previous.mapping,
    previous.manifest["preparation"]["example"],
    horizon=previous.settings.horizon,
  )
  pd.testing.assert_frame_equal(prepared[0].frame, expected.frame)
  assert mapping == expected_mapping
  assert settings == dataclasses.replace(previous.settings, task="forecast")
  result = execute((store, artifacts, refreshed_spec))
  assert result["manifest"]["previous_run_id"] == previous.run_id
  assert result["manifest"]["settings"]["horizon"] == 2


def test_holdout_point_only_parity_preserves_missing_actuals(inputs):
  store, artifacts, spec = inputs
  spec = {
    **spec,
    "settings": {**spec["settings"], "task": "holdout", "return_quantiles": False},
  }
  datasets, mapping, settings = services.prepare_inputs(store, artifacts, spec)
  expected = explorer.execute_forecast(FakePredictor(), datasets, mapping, settings)
  actual = execute((store, artifacts, spec))
  for name in ("forecast", "history", "metrics"):
    pd.testing.assert_frame_equal(
      artifacts.get_frame(actual["tables"][name]["key"]), getattr(expected, name)
    )
  assert "q0.1" not in artifacts.get_frame(actual["tables"]["forecast"]["key"])


def test_cache_switch_releases_before_loading(monkeypatch):
  events = []
  monkeypatch.setattr(services, "_cached_predictor", None)
  monkeypatch.setattr(services, "_cached_identity", None)

  def release():
    events.append("release")

  def load(*args):
    events.append("load")
    return FakePredictor()

  monkeypatch.setattr(services, "release_model", release)
  monkeypatch.setattr(services.model_loading, "load_resolved_model", load)
  resolved = services.model_loading.ResolvedModel(
    services.model_loading.ModelSelection(), "local", "revision", (("weights", "hash"),)
  )
  first = services._acquire_model(resolved, "cpu", 4)
  assert services._acquire_model(resolved, "cpu", 4) is first
  assert services._acquire_model(resolved, "cpu", 2) is not first
  assert events == ["release", "load", "release", "load"]


def test_gpu_lock_excludes_second_process_without_loading_torch(tmp_path):
  from timesfm_app.worker import GPUProcessLock

  path = tmp_path / "gpu.lock"
  # The assert 'torch' not in sys.modules check confirms the lock is a cheap,
  # OS-level exclusion primitive (e.g. a file lock) that can fail fast for a
  # second process without first paying for torch's (and CUDA's) import cost.
  code = """
import sys
from pathlib import Path
from timesfm_app.worker import GPUProcessLock, GPUWorkerBusy
assert 'torch' not in sys.modules
try:
    lock = GPUProcessLock(Path(sys.argv[1]))
except GPUWorkerBusy:
    sys.exit(23)
lock.close()
"""
  options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
  owner = GPUProcessLock(path)
  try:
    blocked = subprocess.run(
      [sys.executable, "-c", code, str(path)], timeout=20, check=False, **options
    )
    assert blocked.returncode == 23
  finally:
    owner.close()
  admitted = subprocess.run(
    [sys.executable, "-c", code, str(path)], timeout=20, check=False, **options
  )
  assert admitted.returncode == 0


@pytest.fixture
def worker_store(tmp_path, monkeypatch):
  from timesfm_app import store as store_module
  from timesfm_app import worker
  from timesfm_app.config import Settings

  store = Store.for_testing(f"sqlite+pysqlite:///{tmp_path / 'worker.sqlite'}")
  store.initialize()
  store.create_record("workspace", "Local", {}, record_id="local")
  config = Settings(artifact_root=tmp_path / "artifacts", device="cpu")
  monkeypatch.setattr(worker, "get_settings", lambda: config)
  monkeypatch.setattr(store_module, "Store", lambda *args, **kwargs: store)
  monkeypatch.setattr(worker, "_own_gpu", lambda: None)
  monkeypatch.setattr(worker, "_synchronize", lambda: None)
  yield store, worker
  store.close()


def test_worker_cancellation_stays_pending_until_gpu_synchronizes(
  worker_store, monkeypatch
):
  store, worker = worker_store
  job = store.create_job("local", "forecast", {"kind": "forecast"}, "cancel-test")
  order = []

  def execute(store, artifacts, spec, progress, check, **kwargs):
    store.cancel_job(job["id"])
    assert store.get_job(job["id"])["status"] == "cancelling"
    check()

  def synchronize():
    assert store.get_job(job["id"])["status"] == "cancelling"
    order.append("synchronized")

  monkeypatch.setattr(services, "execute_spec", execute)
  monkeypatch.setattr(worker, "_synchronize", synchronize)
  worker.run_job(job["id"])
  assert order == ["synchronized"]
  assert store.get_job(job["id"])["status"] == "cancelled"
  assert store.list_records("run") == []


def test_worker_duplicate_delivery_does_not_execute_twice(worker_store, monkeypatch):
  store, worker = worker_store
  job = store.create_job("local", "forecast", {"kind": "forecast"}, "once-test")
  calls = []

  def execute(*args, **kwargs):
    calls.append("executed")
    return {"kind": "forecast", "manifest": {}, "tables": {}, "export": {}}

  monkeypatch.setattr(services, "execute_spec", execute)
  worker.run_job(job["id"])
  worker.run_job(job["id"])
  assert calls == ["executed"]
  assert len(store.list_records("run")) == 1


@pytest.mark.parametrize(
  "exception,retry",
  [
    (TimeoutError("temporary"), True),
    (explorer.ExplorerError("Invalid inputs"), False),
    (RuntimeError("CUDA out of memory"), False),
  ],
)
def test_worker_retry_policy_uses_durable_store(
  worker_store, monkeypatch, exception, retry
):
  store, worker = worker_store
  job = store.create_job("local", "forecast", {"kind": "forecast"}, "retry-test")

  def execute(*args, **kwargs):
    raise exception

  monkeypatch.setattr(services, "execute_spec", execute)
  worker.run_job(job["id"])
  result = store.get_job(job["id"])
  assert result["status"] == ("queued" if retry else "failed")
  assert result["attempt"] == 1
  assert len(store.list_records("run")) == 0


def test_worker_does_not_publish_after_lease_expiry(worker_store, monkeypatch):
  from datetime import datetime, timedelta, timezone

  from timesfm_app.store import Job

  store, worker = worker_store
  job = store.create_job("local", "forecast", {"kind": "forecast"}, "stale-test")

  def execute(*args, **kwargs):
    with store.session() as session:
      session.get(Job, job["id"]).lease_until = datetime.now(timezone.utc) - timedelta(
        seconds=1
      )
    return {"kind": "forecast", "manifest": {}, "tables": {}, "export": {}}

  monkeypatch.setattr(services, "execute_spec", execute)
  worker.run_job(job["id"])
  assert not store.list_records("run")
  assert store.get_job(job["id"])["status"] == "running"


def test_worker_middleware_has_no_asynchronous_injection_or_broker_retries():
  from timesfm_app.worker import broker

  names = {type(item).__name__ for item in broker.middleware}
  assert names.isdisjoint({"TimeLimit", "Retries", "ShutdownNotifications"})


def test_poisoned_cuda_context_retires_process(monkeypatch):
  from types import SimpleNamespace

  from timesfm_app import worker

  def broken():
    raise RuntimeError("CUDA context failure")

  def exit_process(code):
    raise SystemExit(code)

  monkeypatch.setitem(
    sys.modules,
    "torch",
    SimpleNamespace(
      cuda=SimpleNamespace(is_initialized=lambda: True, synchronize=broken)
    ),
  )
  monkeypatch.setattr(worker.os, "_exit", exit_process)
  with pytest.raises(SystemExit) as failure:
    worker._synchronize()
  assert failure.value.code == 70
