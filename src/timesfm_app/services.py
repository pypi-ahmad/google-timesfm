"""Durable application inputs and artifacts around the unchanged TimesFM core."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from timesfm3 import analysis, data_preparation, explorer, model_loading, tracking
from timesfm3.uncertainty import calibration_table


class CancellationRequested(Exception):
  """Execution stopped at a boundary outside a native model call."""


def _checkpoint(cancel_check: Callable[[], Any]) -> None:
  if cancel_check():
    raise CancellationRequested("Cancellation requested.")


def _mapping(value: dict[str, Any]) -> explorer.DatasetMapping:
  return explorer.DatasetMapping(
    timestamp=value.get("timestamp"),
    targets=tuple(value.get("targets", ())),
    past_only=tuple(value.get("past_only", ())),
    past_future=tuple(value.get("past_future", ())),
  )


def _settings(value: dict[str, Any]) -> explorer.ForecastSettings:
  settings = explorer.ForecastSettings(**value)
  settings.validate()
  return settings


def ingest_metadata(data: bytes, filename: str, source_name: str) -> dict:
  """Validate uploads with the same parser used by execution, without a model."""
  dataset = explorer.parse_upload(data, Path(filename).suffix, source_name)
  return {
    "columns": list(dataset.frame.columns),
    "rows": len(dataset.frame),
    "memory_bytes": dataset.memory_bytes,
  }


def _record(store: Any, identifier: str, kind: str, workspace: str) -> dict:
  record = store.get_record(identifier, kind=kind)
  if record is None or record["workspace_id"] != workspace:
    raise explorer.ExplorerError(f"The selected {kind} is unavailable.")
  return record


def _group_inputs(store: Any, artifacts: Any, spec: dict) -> tuple:
  mapping = _mapping(spec.get("mapping", {}))
  settings = _settings(spec.get("settings", {}))
  workspace = spec.get("workspace_id", spec.get("workspace", "local"))
  datasets, source_names = [], {}
  identifiers = spec.get("dataset_version_ids", [])
  if not identifiers or len(set(identifiers)) != len(identifiers):
    raise explorer.ExplorerError("Select distinct dataset versions.")
  for identifier in identifiers:
    payload = _record(store, identifier, "dataset_version", workspace)["payload"]
    descriptor = payload["artifact"]
    data = artifacts.get_bytes(descriptor["key"])
    expected = payload.get("sha256", descriptor.get("sha256"))
    if not expected or hashlib.sha256(data).hexdigest() != expected:
      raise explorer.ExplorerError("A stored dataset failed its integrity check.")
    dataset = explorer.parse_upload(data, Path(payload["filename"]).suffix, identifier)
    datasets.append(dataset)
    explorer.validate_upload_total(datasets)
    source_names[identifier] = payload["source_name"]
  groups = data_preparation.group_sources(
    datasets,
    mapping,
    source_names=source_names,
    group_columns=tuple(spec.get("preparation", {}).get("group_columns", ())),
  )
  return groups, mapping, settings


def _prepare_groups(groups: list, mapping: Any, settings: Any, preparation: dict):
  calendar = dict(preparation.get("calendar", {}))
  calendar["events"] = tuple(
    data_preparation.CalendarEvent(**event) for event in calendar.get("events", ())
  )
  return data_preparation.prepare_sources(
    groups,
    mapping,
    source_names={item.dataset_id: item.dataset_id for item in groups},
    calendar=data_preparation.CalendarSettings(**calendar),
    horizon=settings.horizon,
    frequency=preparation.get("frequency"),
  )


def prepare_inputs(store: Any, artifacts: Any, spec: dict) -> tuple:
  """Read pinned source bytes and apply the existing grouping/calendar policy."""
  if spec.get("kind") == "forecast" and spec.get("parent_run_id"):
    previous = _saved_run(store, artifacts, spec)
    saved_spec = {
      **spec,
      "mapping": dataclasses.asdict(previous.mapping),
      "settings": dataclasses.asdict(previous.settings),
    }
    groups, _, _ = _group_inputs(store, artifacts, saved_spec)
    selected = tracking.associated_datasets(
      previous, groups, spec.get("associations", {})
    )
    settings = dataclasses.replace(previous.settings, task="forecast")
    datasets = []
    mapping = previous.mapping
    for item in selected:
      old_id = item.frame.attrs["tracking_previous_dataset"]
      metadata = previous.manifest.get("preparation", {}).get(old_id, {})
      prepared, mapping = data_preparation.restore_preparation(
        item, previous.mapping, metadata, horizon=settings.horizon
      )
      datasets.append(prepared)
    return datasets, mapping, settings
  groups, mapping, settings = _group_inputs(store, artifacts, spec)
  preparation = spec.get("preparation", {})
  excluded = set(preparation.get("excluded_groups", ()))
  selected = [item for item in groups if item.dataset_id not in excluded]
  if not selected:
    raise explorer.ExplorerError("Select at least one dataset group.")
  datasets, mapping = _prepare_groups(selected, mapping, settings, preparation)
  return datasets, mapping, settings


def frame_rows(frame: pd.DataFrame) -> list[dict]:
  """Serialize missing values and timestamps using pandas' JSON conversion."""
  encoded = frame.to_json(orient="records", date_format="iso")
  assert encoded is not None
  return json.loads(encoded)


def preview_inputs(store: Any, artifacts: Any, spec: dict) -> dict:
  """Inspect every group without resolving or acquiring a model."""
  groups, mapping, settings = _group_inputs(store, artifacts, spec)
  preparation = spec.get("preparation", {})
  excluded = set(preparation.get("excluded_groups", ()))
  quality, series, templates, imputations = [], [], [], []
  effective_mapping = mapping
  for group in groups:
    try:
      prepared, effective_mapping = _prepare_groups(
        [group], mapping, settings, preparation
      )
      report = data_preparation.quality_report(
        prepared, effective_mapping, settings, preparation.get("frequency")
      )
      series.append(
        {
          "dataset": group.dataset_id,
          "rows": len(prepared[0].frame),
          "columns": list(prepared[0].frame.columns),
          "excluded": group.dataset_id in excluded,
          "preview": frame_rows(prepared[0].frame.head(100)),
          "preparation": prepared[0].frame.attrs.get("preparation", {}),
        }
      )
      if group.dataset_id not in excluded:
        try:
          missing = data_preparation.imputation_preview(
            prepared[0], effective_mapping, settings
          )
          missing.insert(0, "dataset", group.dataset_id)
          imputations.extend(frame_rows(missing))
        except explorer.ExplorerError:
          pass  # The quality report already explains blocked/chunked inputs.
        if effective_mapping.past_future:
          try:
            templates.extend(
              frame_rows(
                analysis.scenario_template(prepared, effective_mapping, settings)
              )
            )
          except explorer.ExplorerError:
            pass  # Future cells may be unavailable for a historical-only input.
    except explorer.ExplorerError as exc:
      report = data_preparation.quality_report([group], mapping, settings)
      report["status"] = "blocked"
      report["details"] = str(exc)
      series.append(
        {
          "dataset": group.dataset_id,
          "rows": len(group.frame),
          "columns": list(group.frame.columns),
          "excluded": group.dataset_id in excluded,
          "preview": frame_rows(group.frame.head(100)),
        }
      )
    report["excluded"] = group.dataset_id in excluded
    quality.extend(frame_rows(report))
  return {
    "quality": quality,
    "series": series,
    "columns": sorted({column for item in series for column in item["columns"]}),
    "mapping": dataclasses.asdict(effective_mapping),
    "settings": dataclasses.asdict(settings),
    "scenario_template": templates,
    "imputation": imputations,
  }


def _file_digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_frozen_model(spec: dict, freeze_model: Callable | None = None):
  """Copy resolved weights to a verified snapshot before recording ownership."""
  frozen = spec.get("_resolved_model")
  if frozen is None:
    resolved = model_loading.resolve_model(
      model_loading.ModelSelection(**spec.get("model", {}))
    )
    root = spec.get("_model_snapshot_root")
    if root:
      identity = hashlib.sha256(json.dumps(resolved.fingerprints).encode()).hexdigest()
      destination = Path(root) / identity
      destination.mkdir(parents=True, exist_ok=True)
      source = Path(resolved.path)
      for name, expected in resolved.fingerprints:
        target = destination / name
        if not target.is_file() or _file_digest(target) != expected:
          descriptor, temporary = tempfile.mkstemp(dir=destination, suffix=".partial")
          os.close(descriptor)
          try:
            shutil.copyfile(source / name if source.is_dir() else source, temporary)
            if _file_digest(Path(temporary)) != expected:
              raise explorer.ExplorerError("Checkpoint changed while being frozen.")
            os.replace(temporary, target)
          finally:
            Path(temporary).unlink(missing_ok=True)
      path = destination if source.is_dir() else destination / source.name
      resolved = dataclasses.replace(resolved, path=str(path))
    frozen = dataclasses.asdict(resolved)
    if freeze_model is not None:
      frozen = freeze_model(frozen)
      if frozen is None:
        raise CancellationRequested("Model ownership expired before loading.")
  resolved = model_loading.ResolvedModel(
    selection=model_loading.ModelSelection(**frozen["selection"]),
    path=frozen["path"],
    revision=frozen["revision"],
    fingerprints=tuple(tuple(item) for item in frozen["fingerprints"]),
  )
  path = Path(resolved.path)
  for name, expected in resolved.fingerprints:
    asset = path / name if path.is_dir() else path
    if not asset.is_file() or _file_digest(asset) != expected:
      raise explorer.ExplorerError("The frozen checkpoint failed its integrity check.")
  return resolved


_cached_predictor: Any = None
_cached_identity: tuple | None = None


def release_model() -> None:
  """Drain the current device before releasing the sole resident evaluator."""
  global _cached_predictor, _cached_identity
  import torch

  if torch.cuda.is_initialized():
    torch.cuda.synchronize()
  _cached_predictor = None
  _cached_identity = None
  gc.collect()
  if torch.cuda.is_initialized():
    torch.cuda.empty_cache()


def _acquire_model(resolved: Any, device: str, batch_size: int):
  global _cached_predictor, _cached_identity
  identity = (resolved.path, resolved.fingerprints, device, batch_size)
  if identity != _cached_identity:
    release_model()
    _cached_predictor = model_loading.load_resolved_model(resolved, device, batch_size)
    _cached_identity = identity
  _cached_predictor.model_provenance = resolved.provenance()
  return _cached_predictor


class _CheckedPredictor:
  """Preserve evaluator calls and check cancellation outside native execution."""

  def __init__(self, predictor: Any, cancel_check: Callable):
    self.predictor = predictor
    self.cancel_check = cancel_check
    self.device = predictor.device
    self.model_provenance = getattr(predictor, "model_provenance", None)

  def predict_batch(self, **kwargs: Any):
    _checkpoint(self.cancel_check)
    result = list(self.predictor.predict_batch(**kwargs))
    _checkpoint(self.cancel_check)
    return result


def _saved_run(store: Any, artifacts: Any, spec: dict) -> explorer.RunArtifact:
  record = _record(
    store,
    spec["parent_run_id"],
    "run",
    spec.get("workspace_id", spec.get("workspace", "local")),
  )
  payload = record["payload"]
  manifest = payload["manifest"]
  if payload["kind"] != "forecast":
    raise explorer.ExplorerError("Tracking requires a saved forecast run.")
  tables = payload["tables"]
  return explorer.RunArtifact(
    run_id=manifest["run_id"],
    created_at=manifest["created_at"],
    settings=_settings(manifest["settings"]),
    mapping=_mapping(manifest["mapping"]),
    history=artifacts.get_frame(tables["history"]["key"]),
    forecast=artifacts.get_frame(tables["forecast"]["key"]),
    metrics=artifacts.get_frame(tables["metrics"]["key"]),
    manifest=manifest,
    runtime_seconds=manifest["runtime_seconds"],
    device=manifest["device"],
  )


def _analysis_tables(result: analysis.AnalysisArtifact) -> dict:
  tables = {
    "predictions": result.predictions,
    "metrics": result.metrics,
    "calibration": calibration_table(result.predictions),
  }
  if result.kind == "anomaly":
    tables["anomalies"] = analysis.anomaly_table(result.predictions)
  elif result.kind == "scenario":
    tables["scenario_deltas"] = analysis.scenario_deltas(result.predictions)
    tables["scenario_edits"] = pd.DataFrame(
      [
        {"scenario": scenario["name"], **edit}
        for scenario in result.manifest["scenarios"]
        for edit in scenario["overrides"]
      ],
      columns=["scenario", "dataset", "row", "covariate", "value"],
    )
  elif result.kind in {"joint_independent", "covariates"}:
    tables["comparisons"] = analysis.comparison_table(result.metrics, result.kind)
  elif result.kind in {"settings", "baselines"}:
    tables["target_rankings"], tables["overall_rankings"] = (
      analysis.configuration_rankings(result.metrics)
    )
  return tables


def execute_spec(
  store: Any,
  artifacts: Any,
  spec: dict,
  progress: Callable,
  cancel_check: Callable,
  predictor: Any = None,
  *,
  freeze_model: Callable | None = None,
) -> dict:
  """Execute one frozen request and stage complete artifacts before publication."""
  _checkpoint(cancel_check)
  kind = spec["kind"]
  progress("preparing", 0, None)
  if kind == "model_check":
    resolved = resolve_frozen_model(spec, freeze_model)
    _checkpoint(cancel_check)
    # A CPU check verifies parameter names/shapes through the same strict loader.
    checked = model_loading.load_resolved_model(resolved, "cpu", 1)
    manifest = {"model_provenance": checked.model_provenance, "compatible": True}
    del checked
    gc.collect()
    tables = {}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
      archive.writestr("model.json", json.dumps(manifest, indent=2))
    exported = output.getvalue()
  elif kind == "assessment":
    run = _saved_run(store, artifacts, spec)
    assessment_spec = {
      **spec,
      "mapping": dataclasses.asdict(run.mapping),
      "settings": dataclasses.asdict(run.settings),
    }
    # Matching needs original observations, not forecast-ready future covariates.
    datasets, _, _ = _group_inputs(store, artifacts, assessment_spec)
    _checkpoint(cancel_check)
    result = tracking.assess_run(run, datasets, spec["associations"])
    manifest = {
      **result.manifest,
      "assessment_id": result.assessment_id,
      "created_at": result.created_at,
      "fingerprint": result.fingerprint,
      "parent_run_id": spec["parent_run_id"],
    }
    tables = {
      "comparisons": result.comparisons,
      "metrics": result.metrics,
      "calibration": calibration_table(result.comparisons),
    }
    exported = tracking.assessment_zip(result)
  else:
    datasets, mapping, settings = prepare_inputs(store, artifacts, spec)
    previous = None
    if kind == "forecast" and spec.get("parent_run_id"):
      previous = _saved_run(store, artifacts, spec)
      provenance = previous.manifest.get("model_provenance")
      if provenance:
        spec = {
          **spec,
          "_resolved_model": {
            "selection": provenance["selection"],
            "path": provenance["local_path"],
            "revision": provenance.get("resolved_revision"),
            "fingerprints": list(provenance["files"].items()),
          },
        }
    _checkpoint(cancel_check)
    prepared = None
    if kind == "forecast":
      explorer.prepare_batch(datasets, mapping, settings)
    else:
      controls = spec.get("analysis", {})
      options = analysis.AnalysisSettings(
        kind=kind,
        windows=controls.get("windows", 5),
        stride=controls.get("stride"),
        selected_covariates=tuple(controls.get("selected_covariates", ())),
        seasonal_period=controls.get("seasonal_period", 1),
      )
      fingerprint = analysis.analysis_fingerprint(datasets, mapping, settings)
      scenarios = tuple(
        analysis.Scenario(
          item["name"],
          fingerprint,
          tuple(analysis.ScenarioEdit(**edit) for edit in item.get("overrides", ())),
        )
        for item in spec.get("scenarios", ())
      )
      configurations = tuple(
        analysis.ExperimentConfiguration(
          item["name"], _settings({**dataclasses.asdict(settings), **item["settings"]})
        )
        for item in spec.get("configurations", ())
      )
      prepared = analysis.prepare_analysis(
        datasets, mapping, settings, options, scenarios, configurations
      )
    if predictor is None:
      progress("loading_model", 0, None)
      resolved = resolve_frozen_model(spec, freeze_model)
      _checkpoint(cancel_check)
      predictor = _acquire_model(
        resolved, spec.get("_device", "cuda"), settings.batch_size
      )
    checked = _CheckedPredictor(predictor, cancel_check)
    progress("forecasting", 0, prepared.forecast_count if prepared else 1)
    if kind == "forecast":
      result = explorer.execute_forecast(
        checked, datasets, mapping, settings, explorer.repository_revision()
      )
      manifest = result.manifest
      if previous is not None:
        manifest["previous_run_id"] = previous.run_id
        manifest["parent_run_id"] = spec["parent_run_id"]
      tables = {
        "forecast": result.forecast,
        "history": result.history,
        "metrics": result.metrics,
        "calibration": calibration_table(result.forecast),
      }
      exported = explorer.artifact_zip(result)
      progress("forecasting", 1, 1)
    else:

      def analysis_progress(completed: int, total: int) -> None:
        _checkpoint(cancel_check)
        progress("forecasting", completed, total)

      assert prepared is not None
      result = analysis.run_analysis(checked, prepared, progress=analysis_progress)
      manifest = result.manifest
      tables = _analysis_tables(result)
      exported = analysis.analysis_zip(result)
  _checkpoint(cancel_check)
  prefix = spec.get("_artifact_prefix", f"runs/{uuid.uuid4().hex}")
  progress("saving", 0, len(tables) + 1)
  descriptors = {}
  for index, (name, frame) in enumerate(tables.items(), start=1):
    _checkpoint(cancel_check)
    descriptors[name] = artifacts.put_frame(f"{prefix}/{name}.parquet", frame)
    progress("saving", index, len(tables) + 1)
  _checkpoint(cancel_check)
  export = artifacts.put_bytes(f"{prefix}/export.zip", exported)
  _checkpoint(cancel_check)
  return {"kind": kind, "manifest": manifest, "tables": descriptors, "export": export}
