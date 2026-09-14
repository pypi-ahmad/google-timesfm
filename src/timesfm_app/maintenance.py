"""Opt-in retention and conservative cleanup of abandoned artifact files.

Two independent policies: age/count-based run retention (retention_preview/
apply_retention, driven by a per-workspace policy record) and orphaned-file
sweeping (cleanup_orphans), which deletes artifact keys no live record or
job references. Both are safe to run repeatedly and concurrently with normal
traffic; see store.py's delete_record/artifact_operation_lock for how races
against in-flight writes are closed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select

from .artifacts import ArtifactStore, S3ArtifactStore, make_artifact_store
from .store import Conflict, Job, NotFound, Record, Store


def _now() -> datetime:
  return datetime.now(timezone.utc)


def _timestamp(value: str) -> datetime:
  result = datetime.fromisoformat(value)
  return result if result.tzinfo is not None else result.replace(tzinfo=timezone.utc)


def retention_preview(
  store: Store, workspace_id: str = "local", *, now: datetime | None = None
) -> dict[str, Any]:
  workspace = store.get_record(workspace_id, "workspace")
  policy = workspace["payload"].get("retention", {})
  if not policy.get("enabled", False):
    return {"enabled": False, "candidates": [], "protected": []}
  max_age = policy.get("max_age_days")
  max_runs = policy.get("max_runs")
  for value in (max_age, max_runs):
    if value is not None and (
      not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
      raise ValueError("Retention age and count must be positive integers.")
  now = now or _now()
  runs = sorted(
    store.list_records("run", workspace_id),
    key=lambda row: _timestamp(
      row["payload"].get("manifest", {}).get("created_at", row["created_at"])
    ),
    reverse=True,
  )
  candidates, protected = [], []
  for index, run in enumerate(runs):
    created_at = run["payload"].get("manifest", {}).get("created_at", run["created_at"])
    reasons = []
    if max_age is not None and _timestamp(created_at) < now - timedelta(days=max_age):
      reasons.append("age")
    if max_runs is not None and index >= max_runs:
      reasons.append("count")
    if not reasons:
      continue
    blockers = store.deletion_blockers(run["id"])
    entry = {
      "id": run["id"],
      "name": run["name"],
      "created_at": created_at,
      "reasons": reasons,
      "blockers": blockers,
    }
    (protected if blockers else candidates).append(entry)
  return {"enabled": True, "candidates": candidates, "protected": protected}


def apply_retention(store: Store, workspace_id: str = "local") -> dict[str, Any]:
  """Recheck every dependency transactionally; files stay for orphan grace cleanup."""
  preview = retention_preview(store, workspace_id)
  removed, retained = [], []
  for candidate in preview["candidates"]:
    try:
      store.delete_record(candidate["id"])
    except (Conflict, NotFound):
      retained.append(candidate["id"])
    else:
      removed.append(candidate["id"])
  return {**preview, "removed": removed, "retained": retained}


def _artifact_keys(value: Any) -> set[str]:
  if isinstance(value, dict):
    own = (
      {value["key"]}
      if isinstance(value.get("key"), str) and "sha256" in value
      else set()
    )
    return own.union(*(_artifact_keys(item) for item in value.values()))
  if isinstance(value, list):
    return set().union(*(_artifact_keys(item) for item in value))
  return set()


def _protected_artifacts(store: Store) -> tuple[set[str], tuple[str, ...]]:
  with store.session() as session:
    keys: set[str] = set()
    prefixes = []
    for record in session.scalars(select(Record)):
      keys.update(_artifact_keys(record.payload))
    for job in session.scalars(select(Job)):
      keys.update(_artifact_keys(job.spec))
      if job.status in {"running", "cancelling"}:
        prefixes.append(f"jobs/{job.id}/attempt-{job.attempt}/")
  return keys, tuple(prefixes)


def _objects(artifacts: ArtifactStore | S3ArtifactStore):
  prefixes = ("jobs/", "imports/", "datasets/")
  if isinstance(artifacts, ArtifactStore):
    for prefix in prefixes:
      folder = artifacts.root / prefix
      if not folder.exists() or folder.is_symlink():
        continue
      for path in folder.rglob("*"):
        if not path.is_file() or path.is_symlink():
          continue
        key = path.relative_to(artifacts.root).as_posix()
        # Reject any resolved escape before listing it as a deletion candidate.
        artifacts._path(key)
        yield key, datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
  else:
    paginator = artifacts.client.get_paginator("list_objects_v2")
    for prefix in prefixes:
      for page in paginator.paginate(Bucket=artifacts.bucket, Prefix=prefix):
        for item in page.get("Contents", []):
          yield item["Key"], item["LastModified"]


def cleanup_orphans(
  store: Store,
  artifacts: ArtifactStore | S3ArtifactStore,
  *,
  apply: bool = False,
  now: datetime | None = None,
) -> dict[str, Any]:
  if apply:
    # This lock is also taken by migration.py's legacy import, so an orphan
    # sweep cannot run concurrently with an import writing new artifact keys
    # under the same prefixes. Preview mode skips the lock since it only reads.
    with store.artifact_operation_lock():
      return _cleanup_orphans(store, artifacts, apply=True, now=now)
  return _cleanup_orphans(store, artifacts, apply=False, now=now)


def _cleanup_orphans(
  store: Store,
  artifacts: ArtifactStore | S3ArtifactStore,
  *,
  apply: bool,
  now: datetime | None,
) -> dict[str, Any]:
  """Only abandoned files older than 24 hours under managed artifact prefixes."""
  cutoff = (now or _now()) - timedelta(hours=24)
  keys, prefixes = _protected_artifacts(store)
  candidates, removed = [], []
  for key, modified in _objects(artifacts):
    if modified >= cutoff or key in keys or key.startswith(prefixes):
      continue
    candidates.append(key)
    if apply:
      # Recheck immediately before deletion; publication may have completed since preview.
      current_keys, current_prefixes = _protected_artifacts(store)
      if key in current_keys or key.startswith(current_prefixes):
        continue
      artifacts.delete(key)
      removed.append(key)
  return {"candidates": candidates, "removed": removed, "applied": apply}


def main() -> None:
  """Preview or apply eligible artifact and result-retention cleanup."""
  from .config import get_settings

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--workspace", default="local")
  parser.add_argument(
    "--apply", action="store_true", help="Apply the previewed policy."
  )
  parser.add_argument(
    "--orphans", action="store_true", help="Include orphan artifacts."
  )
  parser.add_argument(
    "--keep-days", type=int, help="Set an enabled workspace age policy."
  )
  parser.add_argument(
    "--keep-runs", type=int, help="Set an enabled workspace count policy."
  )
  args = parser.parse_args()
  for value in (args.keep_days, args.keep_runs):
    if value is not None and value < 1:
      parser.error("Retention limits must be positive.")
  settings = get_settings()
  store = Store(settings.database_url)
  try:
    if args.keep_days is not None or args.keep_runs is not None:
      workspace = store.get_record(args.workspace, "workspace")
      policy = {**workspace["payload"].get("retention", {}), "enabled": True}
      if args.keep_days is not None:
        policy["max_age_days"] = args.keep_days
      if args.keep_runs is not None:
        policy["max_runs"] = args.keep_runs
      store.update_record(
        workspace["id"],
        {**workspace["payload"], "retention": policy},
        workspace["revision"],
      )
    result = (apply_retention if args.apply else retention_preview)(
      store, args.workspace
    )
    if args.orphans:
      result["orphans"] = cleanup_orphans(
        store, make_artifact_store(settings), apply=args.apply
      )
    print(json.dumps(result, indent=2))
  finally:
    store.close()


if __name__ == "__main__":
  main()
