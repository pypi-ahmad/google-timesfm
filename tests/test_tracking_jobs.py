"""Automatic tracking uses immutable version identity and explicit refresh consent."""

import hashlib
import json

import pytest

from timesfm_app.jobs import dispatch_once
from timesfm_app.schemas import ActualsSubmission, RunSpec
from timesfm_app.store import Store
from timesfm_app.tracking_jobs import reconcile_tracking


@pytest.fixture
def store():
  repository = Store.for_testing()
  repository.initialize()
  repository.create_record("workspace", "Local", {}, record_id="local")
  yield repository
  repository.close()


def source_version(store, dataset, suffix, name="Demand"):
  return store.create_record(
    "dataset_version",
    suffix,
    {
      "dataset_id": dataset["id"],
      "source_name": name,
      "sha256": suffix,
    },
    workspace_id=dataset["workspace_id"],
  )


def tracked_forecast(store, *, auto_refresh=False, grouped=False):
  dataset = store.create_record("dataset", "Demand", {})
  version = source_version(store, dataset, "original")
  group_key = {"store": "west"} if grouped else {}
  identity = 'Demand / {"store":"west"}' if grouped else "Demand"
  spec = RunSpec.model_validate(
    {
      "dataset_version_ids": [version["id"]],
      "mapping": {"timestamp": "date", "targets": ["sales"]},
      "settings": {"horizon": 2, "context_length": 8, "task": "holdout"},
      "preparation": {
        "group_columns": ["store"] if grouped else [],
        "frequency": "D",
        "calendar": {"weekday": True},
      },
      "model": {"source": "example/model", "revision": "main"},
    }
  ).model_dump(mode="json")
  run = store.create_record(
    "run",
    "Issued",
    {
      "kind": "forecast",
      "spec": {**spec, "_resolved_model": {"private": "execution-only"}},
      "manifest": {
        "datasets": [{"dataset_id": identity}],
        "preparation": {identity: {"source_name": "Demand", "group_key": group_key}},
        "model_provenance": {
          "selection": {
            "source": "example/model",
            "kind": "hub",
            "revision": "main",
            "offline": True,
          },
          "resolved_revision": "pinned-commit",
          "local_path": "snapshots/immutable",
          "files": {"model.safetensors": "sha256"},
        },
      },
    },
  )
  tracked = store.create_record(
    "tracking",
    "Track demand",
    {
      "run_id": run["id"],
      "auto_refresh": auto_refresh,
    },
  )
  return dataset, version, run, tracked


def test_unchanged_versions_never_queue_work_and_duplicate_polls_are_idempotent(store):
  dataset, _, parent, tracked = tracked_forecast(store)
  assert reconcile_tracking(store)["enqueued"] == 0
  assert store.list_jobs() == []
  latest = source_version(store, dataset, "updated")
  assert reconcile_tracking(store)["enqueued"] == 1
  assert reconcile_tracking(store)["enqueued"] == 0
  assert len(store.list_jobs()) == 1
  job = store.list_jobs()[0]
  assert job["kind"] == "assessment"
  assert job["spec"]["dataset_version_ids"] == [latest["id"]]
  assert job["spec"]["parent_run_id"] == parent["id"]
  assert job["spec"]["associations"] == {"Demand": "Demand"}
  body = ActualsSubmission(
    dataset_version_ids=[latest["id"]], associations={"Demand": "Demand"}
  )
  digest = hashlib.sha256(
    json.dumps(body.model_dump(), sort_keys=True).encode()
  ).hexdigest()
  assert job["idempotency_key"] == f"assessment:{tracked['id']}:{digest}"


def test_refresh_is_opt_in_and_retains_parent_pin_and_preparation(store):
  dataset, _, parent, _ = tracked_forecast(store, auto_refresh=True, grouped=True)
  latest = source_version(store, dataset, "updated", name="Renamed demand")
  assert reconcile_tracking(store)["enqueued"] == 2
  refresh = next(job for job in store.list_jobs() if job["kind"] == "forecast")
  assert refresh["spec"]["model"]["revision"] == "pinned-commit"
  assert refresh["spec"]["preparation"] == parent["payload"]["spec"]["preparation"]
  assert refresh["spec"]["settings"]["task"] == "forecast"
  assert refresh["spec"]["settings"]["horizon"] == 2
  assert refresh["spec"]["dataset_version_ids"] == [latest["id"]]
  assert refresh["spec"]["associations"] == {
    'Demand / {"store":"west"}': 'Renamed demand / {"store":"west"}',
  }
  assert "_resolved_model" not in refresh["spec"]
  assert reconcile_tracking(store)["enqueued"] == 0


def test_unrelated_logical_datasets_and_workspaces_do_not_trigger_tracking(store):
  dataset, _, _, _ = tracked_forecast(store)
  other = store.create_record("dataset", "Demand", {})
  source_version(store, other, "unrelated")
  store.create_record("workspace", "Other", {}, record_id="other")
  foreign = store.create_record("dataset", "Demand", {}, workspace_id="other")
  source_version(store, foreign, "foreign-newest")
  assert reconcile_tracking(store)["enqueued"] == 0
  latest = source_version(store, dataset, "correct")
  assert reconcile_tracking(store)["enqueued"] == 1
  assert store.list_jobs()[0]["spec"]["dataset_version_ids"] == [latest["id"]]
  assert store.list_jobs("other") == []


def test_dispatcher_sends_automatic_assessment_from_the_same_outbox_poll(store):
  dataset, _, _, _ = tracked_forecast(store)
  source_version(store, dataset, "updated")
  messages = []
  assert dispatch_once(store, messages.append) == 1
  assert messages == [store.list_jobs()[0]["id"]]
  assert dispatch_once(store, messages.append) == 0


def test_refresh_without_verified_provenance_is_skipped_but_assessment_runs(store):
  dataset, _, parent, tracked = tracked_forecast(store, auto_refresh=True)
  # Make a separate legacy-style immutable run; never edit an issued record.
  legacy = store.create_record(
    "run",
    "Unverified",
    {
      **parent["payload"],
      "manifest": {
        key: value
        for key, value in parent["payload"]["manifest"].items()
        if key != "model_provenance"
      },
    },
  )
  store.delete_record(tracked["id"])
  store.create_record(
    "tracking", "Unverified", {"run_id": legacy["id"], "auto_refresh": True}
  )
  source_version(store, dataset, "updated")
  result = reconcile_tracking(store)
  assert result["enqueued"] == 1 and result["skipped"] == 1
  assert [job["kind"] for job in store.list_jobs()] == ["assessment"]


def test_quota_defers_refresh_without_losing_or_duplicating_assessment(store):
  store.queued_job_limit = 1
  dataset, _, _, _ = tracked_forecast(store, auto_refresh=True)
  source_version(store, dataset, "updated")
  result = reconcile_tracking(store)
  assert result["enqueued"] == 1 and result["deferred"] == 1
  assessment = store.list_jobs()[0]
  store.cancel_job(assessment["id"])
  assert reconcile_tracking(store)["enqueued"] == 1
  assert len(store.list_jobs()) == 2
  assert reconcile_tracking(store)["enqueued"] == 0


def test_legacy_run_without_source_versions_requires_manual_association(store):
  legacy = store.create_record(
    "run", "Imported", {"kind": "forecast", "spec": {}, "manifest": {}}
  )
  store.create_record(
    "tracking", "Imported", {"run_id": legacy["id"], "auto_refresh": True}
  )
  assert reconcile_tracking(store)["enqueued"] == 0
  assert store.list_jobs() == []
