"""Read-only DuckDB import into immutable application records and artifacts."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from timesfm3 import analysis, explorer, run_store, tracking
from timesfm3.uncertainty import calibration_table

from .artifacts import ArtifactStore, S3ArtifactStore, make_artifact_store
from .store import Conflict, NotFound, Store


def import_id(source: Path, workspace: str, kind: str, legacy_id: str) -> str:
  identity = [str(source.resolve()).casefold(), workspace, kind, legacy_id]
  return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:32]


def _stable_zip(data: bytes) -> bytes:
  """Remove ZIP wall-clock metadata so interrupted imports can safely resume."""
  output = BytesIO()
  with (
    zipfile.ZipFile(BytesIO(data)) as original,
    zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive,
  ):
    for name in sorted(original.namelist()):
      info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
      info.compress_type = zipfile.ZIP_DEFLATED
      archive.writestr(info, original.read(name))
  return output.getvalue()


def import_legacy(
  source: str | Path,
  store: Store,
  artifacts: ArtifactStore | S3ArtifactStore,
  workspace_id: str = "local",
) -> dict[str, Any]:
  with store.artifact_operation_lock():
    return _import_legacy(source, store, artifacts, workspace_id)


def _import_legacy(
  source: str | Path,
  store: Store,
  artifacts: ArtifactStore | S3ArtifactStore,
  workspace_id: str,
) -> dict[str, Any]:
  """Import every saved vintage/report; never open the source for writing."""
  source = Path(source).resolve()
  if not source.is_file():
    raise FileNotFoundError("The legacy DuckDB file does not exist.")
  store.get_record(workspace_id, "workspace")
  with duckdb.connect(str(source), read_only=True) as connection:
    tables = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    run_ids = (
      [
        row[0]
        for row in connection.execute(
          "SELECT run_id FROM runs ORDER BY run_id"
        ).fetchall()
      ]
      if "runs" in tables
      else []
    )
    analysis_ids = (
      [
        row[0]
        for row in connection.execute(
          "SELECT analysis_id FROM analysis_runs ORDER BY analysis_id"
        ).fetchall()
      ]
      if "analysis_runs" in tables
      else []
    )
    tracked = (
      dict(connection.execute("SELECT run_id, tracked_at FROM tracked_runs").fetchall())
      if "tracked_runs" in tables
      else {}
    )
    links = (
      connection.execute(
        "SELECT run_id, previous_run_id, associations, linked_at FROM tracking_links"
      ).fetchall()
      if "tracking_links" in tables
      else []
    )
  counts = {
    "created": 0,
    "skipped": 0,
    "forecasts": 0,
    "analyses": 0,
    "assessments": 0,
    "tracking": 0,
  }
  source_key = hashlib.sha256(str(source).casefold().encode()).hexdigest()
  mapped_runs = {
    legacy: import_id(source, workspace_id, "forecast", legacy) for legacy in run_ids
  }

  def existing(identifier: str) -> bool:
    try:
      store.get_record(identifier)
    except NotFound:
      return False
    counts["skipped"] += 1
    return True

  def save(identifier: str, kind: str, name: str, payload: dict) -> None:
    try:
      store.create_record(kind, name, payload, workspace_id, identifier)
    except Conflict:
      # A concurrent importer may have published the same immutable record.
      if store.get_record(identifier)["payload"] != payload:
        raise
      counts["skipped"] += 1
    else:
      counts["created"] += 1

  def persist(
    identifier: str,
    kind: str,
    legacy_id: str,
    manifest: dict,
    frames: dict[str, pd.DataFrame],
    exported: bytes,
    spec: dict | None = None,
  ) -> None:
    prefix = f"imports/{source_key}/{workspace_id}/{identifier}"
    descriptors = {
      name: artifacts.put_frame(f"{prefix}/{name}.parquet", frame)
      for name, frame in frames.items()
    }
    payload = {
      "kind": kind,
      "manifest": manifest,
      "spec": spec or {},
      "tables": descriptors,
      "export": artifacts.put_bytes(f"{prefix}/export.zip", _stable_zip(exported)),
      "legacy": True,
      "source_data_available": False,
    }
    save(identifier, "run", f"Imported {kind}: {legacy_id}", payload)

  def manifest_for(original: dict, legacy_id: str, created_at: str) -> dict:
    return {
      **original,
      "legacy_id": legacy_id,
      "legacy_source": source_key,
      "created_at": created_at,
      "source_data_available": False,
      "history_available": False,
    }

  for legacy_id in run_ids:
    identifier = mapped_runs[legacy_id]
    counts["forecasts"] += 1
    if not existing(identifier):
      result = run_store.load_run(source, legacy_id)
      manifest = manifest_for(result.manifest, legacy_id, result.created_at)
      manifest.update(
        run_id=legacy_id,
        settings=dataclasses.asdict(result.settings),
        mapping=dataclasses.asdict(result.mapping),
        device=result.device,
        runtime_seconds=result.runtime_seconds,
      )
      result = dataclasses.replace(result, manifest=manifest)
      persist(
        identifier,
        "forecast",
        legacy_id,
        manifest,
        {
          "forecast": result.forecast,
          "history": result.history,
          "metrics": result.metrics,
          "calibration": calibration_table(result.forecast),
        },
        explorer.artifact_zip(result),
      )
    for assessment in run_store.load_assessments(source, legacy_id):
      counts["assessments"] += 1
      assessment_id = import_id(
        source, workspace_id, "assessment", assessment.assessment_id
      )
      if existing(assessment_id):
        continue
      manifest = manifest_for(
        assessment.manifest, assessment.assessment_id, assessment.created_at
      )
      manifest.update(
        assessment_id=assessment.assessment_id,
        fingerprint=assessment.fingerprint,
        run_id=legacy_id,
        parent_run_id=identifier,
      )
      assessment = dataclasses.replace(assessment, manifest=manifest)
      persist(
        assessment_id,
        "assessment",
        assessment.assessment_id,
        manifest,
        {
          "comparisons": assessment.comparisons,
          "metrics": assessment.metrics,
          "calibration": calibration_table(assessment.comparisons),
        },
        tracking.assessment_zip(assessment),
        {"parent_run_id": identifier},
      )

  for legacy_id in analysis_ids:
    counts["analyses"] += 1
    identifier = import_id(source, workspace_id, "analysis", legacy_id)
    if existing(identifier):
      continue
    result = run_store.load_analysis(source, legacy_id)
    manifest = manifest_for(result.manifest, legacy_id, result.created_at)
    manifest["analysis_id"] = legacy_id
    result = dataclasses.replace(result, manifest=manifest)
    # Use the same derived report tables and exports as the new worker.
    from .services import _analysis_tables

    persist(
      identifier,
      result.kind,
      legacy_id,
      manifest,
      _analysis_tables(result),
      analysis.analysis_zip(result),
    )

  link_map = {row[0]: row for row in links}
  for current, previous, _, linked_at in links:
    tracked.setdefault(current, linked_at)
    tracked.setdefault(previous, linked_at)
  for legacy_id, tracked_at in tracked.items():
    counts["tracking"] += 1
    identifier = import_id(source, workspace_id, "tracking", legacy_id)
    if existing(identifier):
      continue
    payload = {
      "run_id": mapped_runs[legacy_id],
      "legacy": True,
      "legacy_id": legacy_id,
      "legacy_source": source_key,
      "tracked_at": tracked_at,
      "auto_refresh": False,
    }
    if legacy_id in link_map:
      _, previous, associations, linked_at = link_map[legacy_id]
      payload.update(
        previous_run_id=mapped_runs[previous],
        linked_at=linked_at,
        associations=json.loads(associations)
        if isinstance(associations, str)
        else associations,
      )
    save(identifier, "tracking", f"Imported tracking: {legacy_id}", payload)
  return {"source": str(source), "workspace_id": workspace_id, **counts}


def main() -> None:
  """Import compatible legacy DuckDB records without modifying their source."""
  from .config import get_settings

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("database", type=Path)
  parser.add_argument("--workspace", default="local")
  args = parser.parse_args()
  settings = get_settings()
  store = Store(settings.database_url)
  try:
    print(
      json.dumps(
        import_legacy(
          args.database, store, make_artifact_store(settings), args.workspace
        ),
        indent=2,
      )
    )
  finally:
    store.close()


if __name__ == "__main__":
  main()
