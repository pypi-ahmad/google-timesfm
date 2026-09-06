"""Crash recovery, duplicate delivery, cancellation, and attempt fences."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from timesfm_app.jobs import dispatch_once
from timesfm_app.store import Conflict, Job, QuotaExceeded, Store


@pytest.fixture
def store():
  instance = Store.for_testing()
  instance.initialize()
  instance.create_record("workspace", "Local", {}, record_id="local")
  yield instance
  instance.close()


def create(store, key="request", spec=None):
  return store.create_job("local", "forecast", spec or {"horizon": 8}, key)


def expire(store, job_id):
  with store.session() as session:
    session.get(Job, job_id).lease_until = datetime.now(timezone.utc) - timedelta(
      seconds=1
    )


def test_idempotent_submission_has_one_job_and_one_outbox_entry(store):
  first = create(store)
  assert create(store)["id"] == first["id"]
  assert len(store.outbox_pending()) == 1
  with pytest.raises(Conflict, match="different request"):
    create(store, spec={"horizon": 16})


def test_idempotency_hash_is_independent_of_object_key_order(store):
  first = create(store, spec={"horizon": 8, "context": 16})
  second = create(store, spec={"context": 16, "horizon": 8})
  assert first["id"] == second["id"]


def test_duplicate_delivery_cannot_run_or_publish_twice(store):
  job = create(store)
  attempt = store.claim_job(job["id"])
  assert attempt["attempt"] == 1
  assert store.claim_job(job["id"]) is None
  result = store.publish_job(job["id"], 1, {"manifest": {"sha256": "verified"}})
  assert result["payload"]["spec"] == job["spec"]
  assert result["payload"]["job_id"] == job["id"]
  assert result["payload"]["kind"] == "forecast"
  assert store.publish_job(job["id"], 1, {}) is None
  assert store.claim_job(job["id"]) is None
  assert len(store.list_records("run")) == 1


def test_cancel_before_publish_waits_for_worker_quiescence(store):
  job = create(store)
  store.claim_job(job["id"])
  assert store.cancel_job(job["id"])["status"] == "cancelling"
  assert not store.heartbeat(job["id"], 1)
  assert store.publish_job(job["id"], 1, {}) is None
  assert store.get_job(job["id"])["result_id"] is None
  assert store.confirm_cancel(job["id"], 1)
  assert store.get_job(job["id"])["status"] == "cancelled"
  assert store.list_records("run") == []
  with pytest.raises(Conflict):
    store.retry_job(job["id"])


def test_queued_cancel_never_claims_and_success_cannot_be_cancelled(store):
  queued = create(store)
  assert store.cancel_job(queued["id"])["status"] == "cancelled"
  assert store.claim_job(queued["id"]) is None
  completed = create(store, "second")
  store.claim_job(completed["id"])
  store.publish_job(completed["id"], 1, {})
  assert store.cancel_job(completed["id"])["status"] == "succeeded"


def test_expired_and_superseded_attempts_cannot_publish_or_heartbeat(store):
  job = create(store)
  store.claim_job(job["id"])
  expire(store, job["id"])
  assert store.publish_job(job["id"], 1, {}) is None
  assert not store.heartbeat(job["id"], 1)
  assert store.reconcile_expired() == 1
  assert store.claim_job(job["id"])["attempt"] == 2
  assert not store.progress(job["id"], 1, "old progress")
  assert not store.fail_job(job["id"], 1, {"message": "old error"})
  assert store.publish_job(job["id"], 1, {}) is None
  assert store.publish_job(job["id"], 2, {}) is not None


def test_retryable_failure_stops_after_two_automatic_retries(store):
  job = create(store)
  for attempt in range(1, 4):
    assert store.claim_job(job["id"])["attempt"] == attempt
    assert store.fail_job(job["id"], attempt, {"code": "temporary"}, retryable=True)
  assert store.get_job(job["id"])["status"] == "failed"
  assert store.claim_job(job["id"]) is None
  assert store.retry_job(job["id"])["status"] == "queued"
  assert store.claim_job(job["id"])["attempt"] == 4


def test_resolved_model_is_pinned_once_and_reused_on_retry(store):
  job = create(store)
  store.claim_job(job["id"])
  original = {"path": "snapshot/first", "fingerprints": [["weights", "hash"]]}
  assert store.freeze_job_model(job["id"], 1, original) == original
  assert store.freeze_job_model(job["id"], 1, {"path": "changed"}) == original
  store.fail_job(job["id"], 1, {"code": "temporary"}, retryable=True)
  retried = store.claim_job(job["id"])
  assert retried["spec"]["_resolved_model"] == original
  assert store.freeze_job_model(job["id"], 1, original) is None
  assert create(store)["id"] == job["id"]


def test_expired_cancellation_requires_quiescence_and_events_resume(store):
  job = create(store)
  store.claim_job(job["id"])
  store.progress(job["id"], 1, "forecast", completed=1, total=2)
  initial_events = store.events(job["id"])
  last_seq = initial_events[-1]["seq"]
  store.cancel_job(job["id"])
  expire(store, job["id"])
  assert store.reconcile_expired() == 1
  later_events = store.events(job["id"], after=last_seq)
  assert [event["status"] for event in later_events] == ["cancelling", "cancelling"]
  assert store.get_job(job["id"])["stage"] == "recovery_required"
  assert store.reconcile_expired() == 0
  assert store.confirm_cancel(job["id"], 1)
  assert all(event["seq"] > last_seq for event in later_events)


def test_quota_counts_active_jobs_and_idempotent_replay_is_exempt():
  store = Store.for_testing(queued_job_limit=1)
  store.initialize()
  store.create_record("workspace", "Local", {}, record_id="local")
  try:
    first = create(store)
    assert create(store)["id"] == first["id"]
    with pytest.raises(QuotaExceeded):
      create(store, "second")
    store.cancel_job(first["id"])
    assert create(store, "second")["status"] == "queued"
  finally:
    store.close()


def test_dispatch_failure_preserves_outbox_until_acknowledged(store):
  job = create(store)

  def offline(_job_id):
    raise ConnectionError("broker unavailable")

  assert dispatch_once(store, offline) == 0
  assert len(store.outbox_pending()) == 1
  messages = []
  assert dispatch_once(store, messages.append) == 1
  assert messages == [job["id"]]
  assert store.outbox_pending() == []


def test_crash_after_send_redelivers_but_claim_is_idempotent(store, monkeypatch):
  job = create(store)
  messages = []
  original = store.mark_dispatched

  def crash(_entry_id):
    raise RuntimeError("dispatcher crashed after broker acknowledgment")

  monkeypatch.setattr(store, "mark_dispatched", crash)
  with pytest.raises(RuntimeError):
    dispatch_once(store, messages.append)
  monkeypatch.setattr(store, "mark_dispatched", original)
  assert dispatch_once(store, messages.append) == 1
  assert messages == [job["id"], job["id"]]
  assert store.claim_job(messages[0]) is not None
  assert store.claim_job(messages[1]) is None


def test_unclaimed_delivery_is_repaired_once_after_grace_period(store):
  job = create(store)
  messages = []
  assert dispatch_once(store, messages.append) == 1
  assert store.reconcile_queued() == 0
  with store.session() as session:
    session.get(Job, job["id"]).updated_at = (
      datetime.now(timezone.utc) - timedelta(seconds=31)
    ).isoformat()
  assert store.reconcile_queued() == 1
  assert store.reconcile_queued() == 0
  assert dispatch_once(store, messages.append) == 1
  assert messages == [job["id"], job["id"]]
  assert store.get_job(job["id"])["attempt"] == 0


def test_supervisor_cancellation_checks_ownership_before_stopping_process(store):
  job = create(store)
  store.claim_job(job["id"])
  store.cancel_job(job["id"])
  stopped = []
  callback = lambda: stopped.append(True) or True
  assert not store.confirm_cancel_after_exit(job["id"], 2, callback)
  assert stopped == []
  assert not store.confirm_cancel_after_exit(job["id"], 1, lambda: False)
  assert store.get_job(job["id"])["status"] == "cancelling"
  assert store.confirm_cancel_after_exit(job["id"], 1, callback)
  assert stopped == [True]
  assert not store.confirm_cancel_after_exit(job["id"], 1, callback)
  assert stopped == [True]


def test_missing_workspace_and_deleted_input_cannot_be_submitted(store):
  from timesfm_app.store import NotFound

  with pytest.raises(NotFound, match="Workspace"):
    store.create_job("missing", "forecast", {}, "request")
  with pytest.raises(NotFound, match="referenced"):
    create(store, spec={"dataset_version_ids": ["deleted"]})
  with pytest.raises(NotFound):
    store.create_record("draft", "Orphan", {}, workspace_id="missing")


def test_nonempty_workspace_is_protected_and_removed_results_keep_a_tombstone(store):
  job = create(store)
  store.claim_job(job["id"])
  run = store.publish_job(job["id"], 1, {})
  with pytest.raises(Conflict):
    store.delete_record("local")
  store.delete_record(run["id"])
  assert store.get_job(job["id"])["status"] == "succeeded"
  assert store.get_job(job["id"])["result_id"] is None
  assert store.events(job["id"])[-1]["payload"]["removed_result_id"] == run["id"]
  audit = [
    row["payload"]
    for row in store.list_records("audit")
    if row["payload"]["entity_id"] == job["id"]
  ]
  assert {entry["action"] for entry in audit} == {
    "job.submitted",
    "job.published",
    "job.result_removed",
  }
  assert (
    next(entry for entry in audit if entry["action"] == "job.result_removed")[
      "removed_result_id"
    ]
    == run["id"]
  )


@pytest.fixture
def postgres_store():
  """Use only an explicitly selected test database; isolate data by workspace."""
  url = os.environ.get("TIMESFM_TEST_DATABASE_URL")
  if not url:
    pytest.skip("Set TIMESFM_TEST_DATABASE_URL for PostgreSQL concurrency checks.")
  instance = Store(url, queued_job_limit=1)
  instance.initialize()
  workspace = uuid4().hex
  instance.create_record("workspace", "Concurrency test", {}, workspace, workspace)
  try:
    yield instance, workspace
  finally:
    from sqlalchemy import delete, select

    from timesfm_app.store import JobEvent, Outbox, Record

    with instance.session() as session:
      job_ids = list(
        session.scalars(select(Job.id).where(Job.workspace_id == workspace))
      )
      session.execute(delete(JobEvent).where(JobEvent.job_id.in_(job_ids)))
      session.execute(
        delete(Outbox).where(Outbox.payload["job_id"].as_string().in_(job_ids))
      )
      session.execute(delete(Job).where(Job.workspace_id == workspace))
      session.execute(delete(Record).where(Record.workspace_id == workspace))
    instance.close()


def test_postgres_concurrent_idempotency_quota_and_claim(postgres_store):
  instance, workspace = postgres_store

  def enqueue(_index):
    return instance.create_job(workspace, "forecast", {"horizon": 8}, "same")

  with ThreadPoolExecutor(max_workers=4) as pool:
    created = list(pool.map(enqueue, range(4)))
  assert len({job["id"] for job in created}) == 1
  with pytest.raises(QuotaExceeded):
    instance.create_job(workspace, "forecast", {}, "another")
  with ThreadPoolExecutor(max_workers=4) as pool:
    claims = list(pool.map(instance.claim_job, [created[0]["id"]] * 4))
  assert sum(claim is not None for claim in claims) == 1
  instance.cancel_job(created[0]["id"])
  assert instance.publish_job(created[0]["id"], 1, {}) is None
  assert instance.confirm_cancel(created[0]["id"], 1)


def test_postgres_artifact_operation_lock_serializes_connections(postgres_store):
  instance, _ = postgres_store
  started, acquired = threading.Event(), threading.Event()

  def other_connection():
    started.set()
    with instance.artifact_operation_lock():
      acquired.set()

  with ThreadPoolExecutor(max_workers=1) as pool:
    with instance.artifact_operation_lock():
      future = pool.submit(other_connection)
      assert started.wait(2)
      assert not acquired.wait(0.1)
    assert acquired.wait(3)
    future.result(timeout=3)


def test_postgres_result_deletion_keeps_audit_without_foreign_key_violation(
  postgres_store,
):
  instance, workspace = postgres_store
  job = instance.create_job(workspace, "forecast", {}, "published")
  instance.claim_job(job["id"])
  run = instance.publish_job(job["id"], 1, {"manifest": {"source": "test"}})
  instance.delete_record(run["id"])
  assert instance.get_job(job["id"])["result_id"] is None
  assert any(
    row["payload"].get("removed_result_id") == run["id"]
    for row in instance.list_records("audit", workspace)
  )


def test_postgres_submit_and_input_deletion_cannot_create_dangling_job(postgres_store):
  from timesfm_app.store import NotFound

  instance, workspace = postgres_store
  version = instance.create_record("dataset_version", "Input", {}, workspace)
  barrier = threading.Barrier(2)

  def submit():
    barrier.wait(timeout=3)
    try:
      instance.create_job(
        workspace, "forecast", {"dataset_version_ids": [version["id"]]}, "race"
      )
      return "submitted"
    except NotFound:
      return "rejected"

  def remove():
    barrier.wait(timeout=3)
    try:
      instance.delete_record(version["id"])
      return "deleted"
    except Conflict:
      return "protected"

  with ThreadPoolExecutor(max_workers=2) as pool:
    submitted = pool.submit(submit)
    removed = pool.submit(remove)
    assert (submitted.result(timeout=5), removed.result(timeout=5)) in {
      ("submitted", "protected"),
      ("rejected", "deleted"),
    }


def test_job_audit_records_only_committed_state_changes(store, monkeypatch):
  first = create(store)
  create(store)
  store.cancel_job(first["id"])
  entries = [
    row["payload"]
    for row in store.list_records("audit")
    if row["payload"]["entity_id"] == first["id"]
  ]
  assert sorted(entry["action"] for entry in entries) == [
    "job.cancelled",
    "job.submitted",
  ]
  before_jobs, before_outbox = store.list_jobs(), store.outbox_pending()

  def unavailable(*_args, **_kwargs):
    raise RuntimeError("Audit insertion failed")

  monkeypatch.setattr(store, "_audit", unavailable)
  with pytest.raises(RuntimeError):
    create(store, "must-roll-back")
  assert store.list_jobs() == before_jobs
  assert store.outbox_pending() == before_outbox
