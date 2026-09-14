"""Reconcile tracked forecasts against new immutable logical dataset versions.

Called from jobs.py's dispatch loop on every poll. Safe to call repeatedly:
the job idempotency key folds in a digest of the proposed submission, so
store.create_job (see store.py) de-duplicates re-submissions of the same
version combination rather than this module needing to track what it already
enqueued across polls.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import Any

from .schemas import ActualsSubmission, RunSpec
from .store import Conflict, NotFound, QuotaExceeded, Store

logger = logging.getLogger(__name__)


def _associations(
  parent: dict, replacements: list[tuple[dict, dict]]
) -> dict[str, str]:
  """Use the same canonical source/group identities as group_sources."""
  manifest = parent["payload"].get("manifest", {})
  names: dict[str, str] = {}
  for old, new in replacements:
    old_name = old["payload"]["source_name"].strip()
    new_name = new["payload"]["source_name"].strip()
    if old_name in names or not new_name:
      raise ValueError("Source identities are ambiguous.")
    names[old_name] = new_name
  associations = {}
  for dataset in manifest.get("datasets", []):
    old_id = dataset["dataset_id"]
    metadata = manifest.get("preparation", {}).get(old_id, {})
    source_name = metadata.get("source_name", old_id)
    if source_name not in names:
      raise ValueError("The saved series has no logical source association.")
    new_id = names[source_name]
    group_key = metadata.get("group_key", {})
    if group_key:
      new_id += " / " + json.dumps(
        group_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
      )
    associations[old_id] = new_id
  if not associations or len(set(associations.values())) != len(associations):
    raise ValueError("The saved series identities cannot be mapped uniquely.")
  return associations


def _submission(parent: dict, versions: list[dict]) -> ActualsSubmission | None:
  by_id = {version["id"]: version for version in versions}
  latest: dict[str, dict] = {}
  for version in sorted(
    versions, key=lambda row: (row["created_at"], row["id"]), reverse=True
  ):
    logical_id = version["payload"].get("dataset_id")
    if logical_id:
      latest.setdefault(logical_id, version)
  pinned = parent["payload"].get("spec", {}).get("dataset_version_ids", [])
  if not pinned:
    return None  # Legacy results without pinned source records need manual association.
  replacements = []
  for identifier in pinned:
    old = by_id.get(identifier)
    if old is None:
      raise NotFound("A pinned source version is unavailable in this workspace.")
    candidate = latest[old["payload"]["dataset_id"]]
    replacements.append((old, candidate))
  if all(old["id"] == new["id"] for old, new in replacements):
    return None
  return ActualsSubmission(
    dataset_version_ids=[new["id"] for _, new in replacements],
    associations=_associations(parent, replacements),
  )


def _spec(parent: dict, body: ActualsSubmission, kind: str) -> dict[str, Any]:
  specification = {
    key: value
    for key, value in parent["payload"].get("spec", {}).items()
    if key in RunSpec.model_fields
  }
  specification.update(
    kind=kind,
    dataset_version_ids=body.dataset_version_ids,
    parent_run_id=parent["id"],
    associations=body.associations,
  )
  if kind == "forecast":
    from timesfm3.model_loading import selection_from_provenance

    provenance = parent["payload"].get("manifest", {}).get("model_provenance", {})
    if (
      not provenance.get("local_path")
      or not provenance.get("files")
      or not provenance.get("selection")
    ):
      raise ValueError("Automatic refresh requires verified parent model provenance.")
    selection = selection_from_provenance(provenance)
    if selection.kind == "hub" and not selection.revision:
      raise ValueError("Automatic refresh requires a pinned Hub revision.")
    specification["model"] = dataclasses.asdict(selection)
    specification["settings"] = {
      **specification.get("settings", {}),
      "task": "forecast",
    }
  return RunSpec.model_validate(specification).model_dump(mode="json")


def reconcile_tracking(store: Store) -> dict[str, int]:
  """Queue each new version combination once; quota pressure defers later polls."""
  counts = {"enqueued": 0, "unchanged": 0, "skipped": 0, "deferred": 0}
  cached_versions: dict[str, list[dict]] = {}
  known_jobs: dict[str, set[str]] = {}
  for tracked in store.list_records("tracking", None):
    workspace = tracked["workspace_id"]
    try:
      parent = store.get_record(tracked["payload"]["run_id"], "run")
      if (
        parent["workspace_id"] != workspace
        or parent["payload"].get("kind") != "forecast"
      ):
        raise ValueError("Tracking requires a forecast in the same workspace.")
      if workspace not in cached_versions:
        cached_versions[workspace] = store.list_records("dataset_version", workspace)
        known_jobs[workspace] = {job["id"] for job in store.list_jobs(workspace)}
      body = _submission(parent, cached_versions[workspace])
      if body is None:
        counts["unchanged"] += 1
        continue
      digest = hashlib.sha256(
        json.dumps(body.model_dump(), sort_keys=True).encode()
      ).hexdigest()
      kinds = ["assessment"]
      if tracked["payload"].get("auto_refresh") is True:
        kinds.append("forecast")
      for kind in kinds:
        try:
          spec = _spec(parent, body, kind)
          prefix = "refresh" if kind == "forecast" else "assessment"
          job = store.create_job(
            workspace, kind, spec, f"{prefix}:{tracked['id']}:{digest}"
          )
        except QuotaExceeded:
          counts["deferred"] += 1
        except (Conflict, NotFound, ValueError):
          counts["skipped"] += 1
          logger.info(
            "Tracking submission requires attention (%s, %s).", tracked["id"], kind
          )
        else:
          if job["id"] not in known_jobs[workspace]:
            counts["enqueued"] += 1
            known_jobs[workspace].add(job["id"])
    except (KeyError, TypeError, NotFound, ValueError):
      counts["skipped"] += 1
      logger.info("Tracking source association is unavailable (%s).", tracked["id"])
  return counts
