"""Native ownership regressions using only disposable processes created here."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from timesfm_app import native
from timesfm_app.store import Store


@pytest.fixture
def spawned():
  children = []

  def start(source=None):
    process = subprocess.Popen(
      [
        sys.executable,
        "-u",
        "-c",
        source or "import time; print('ready', flush=True); time.sleep(120)",
      ],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      creationflags=native.HIDDEN,
    )
    children.append(process)
    line = process.stdout.readline().strip()
    assert line, process.stderr.read()
    return process, native.identity(process), line

  yield start
  for process in children:
    if process.poll() is None:
      process.kill()
    process.wait(timeout=10)
    process.stdout.close()
    process.stderr.close()


def worker_record(identity, *, session="owned", queue="gpu", job="job-1", attempt=1):
  return {
    "payload": {
      "pid": identity["pid"],
      "creation_filetime": identity["creation_filetime"],
      "process_created": identity["created"],
      "supervisor_session": session,
      "queue": queue,
      "current_job_id": job,
      "current_attempt": attempt,
    }
  }


def records(*items):
  return SimpleNamespace(
    list_records=lambda *args: list(items),
    confirm_cancel_after_exit=lambda *args: False,
  )


def test_pid_birth_checks_are_exact_and_preserve_unmatched_process(spawned):
  process, identity, _ = spawned()
  assert native.existing(identity).pid == process.pid
  legacy_mismatch = {"pid": process.pid, "created": identity["created"] - 0.005}
  assert native.existing(legacy_mismatch) is None
  native.terminate_owned(legacy_mismatch)
  assert process.poll() is None
  if identity["creation_filetime"] is not None:
    mismatch = {
      **identity,
      "creation_filetime": str(int(identity["creation_filetime"]) + 1),
    }
    assert native.existing(mismatch) is None
    native.terminate_owned(mismatch)
    assert process.poll() is None


def test_termination_stops_respawning_parent_before_its_child(spawned):
  source = """
import subprocess, sys, time
def start():
    return subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
child = start()
print(child.pid, flush=True)
while True:
    if child.poll() is not None:
        child = start()
    time.sleep(0.01)
"""
  parent, identity, child_pid = spawned(source)
  child = psutil.Process(int(child_pid))
  child_identity = native.identity(child)
  try:
    native.terminate_owned(identity)
    parent.wait(timeout=10)
    assert native.existing(identity) is None
    assert native.existing(child_identity) is None
  finally:
    native.terminate_owned(identity)
    native.terminate_owned(child_identity)


def test_worker_record_requires_session_and_birth_identity(spawned):
  process, identity, _ = spawned()
  valid = worker_record(identity)
  wrong_session = worker_record(identity, session="someone-else")
  wrong_birth = worker_record(
    {**identity, "creation_filetime": "invalid", "created": -1}
  )
  found = native.owned_worker_records(
    records(valid, wrong_session, wrong_birth), "owned"
  )
  assert len(found) == 1 and found[0][1]["pid"] == process.pid
  assert not native.owned_worker_records(records(valid), None)


@pytest.mark.parametrize(
  "change",
  [
    {"job": "next-job"},
    {"attempt": 2},
    {"queue": "cpu"},
    {"session": "someone-else"},
  ],
)
def test_cancellation_never_terminates_an_unmatched_worker(spawned, change):
  process, identity, _ = spawned()
  store = records(worker_record(identity, **change))
  state = {"session": "owned", "gpu": identity}
  assert not native.stop_cancelled_job(store, state, "job-1", 1, "gpu")
  assert process.poll() is None


def test_dead_worker_can_confirm_cancel_without_killing_replacement(spawned):
  old, old_identity, _ = spawned()
  replacement, replacement_identity, _ = spawned()
  store = records(worker_record(old_identity))
  native.terminate_owned(old_identity)
  old.wait(timeout=10)
  state = {"session": "owned", "gpu": replacement_identity, "descendants": {"gpu": {}}}
  assert native.stop_cancelled_job(store, state, "job-1", 1, "gpu")
  assert replacement.poll() is None


def test_restart_retires_recorded_orphans_but_preserves_other_sessions(spawned):
  supervisor, supervisor_identity, _ = spawned()
  service, service_identity, _ = spawned()
  orphan, orphan_identity, _ = spawned()
  other, other_identity, _ = spawned()
  native.terminate_owned(supervisor_identity)
  supervisor.wait(timeout=10)
  store = records(
    worker_record(orphan_identity, job=None),
    worker_record(other_identity, session="other", job=None),
  )
  state = {
    "supervisor": supervisor_identity,
    "session": "owned",
    "api": service_identity,
    "descendants": {"gpu": {str(orphan.pid): orphan_identity}},
  }
  native.cleanup_previous(store, state)
  service.wait(timeout=10)
  orphan.wait(timeout=10)
  assert native.existing(service_identity) is None
  assert native.existing(orphan_identity) is None
  assert other.poll() is None


def test_restart_refuses_to_replace_a_live_supervisor(spawned):
  supervisor, identity, _ = spawned()
  with pytest.raises(RuntimeError, match="still running"):
    native.cleanup_previous(records(), {"supervisor": identity, "session": "owned"})
  assert supervisor.poll() is None


def test_cancel_callback_not_invoked_after_job_already_stopped(tmp_path):
  store = Store.for_testing(f"sqlite+pysqlite:///{tmp_path / 'native.sqlite'}")
  store.initialize()
  store.create_record("workspace", "Local", {}, record_id="local")
  try:
    job = store.create_job("local", "forecast", {"kind": "forecast"}, "cancel-race")
    attempt = store.claim_job(job["id"])["attempt"]
    store.cancel_job(job["id"])
    store.confirm_cancel(job["id"], attempt)
    assert not store.confirm_cancel_after_exit(
      job["id"],
      attempt,
      lambda: pytest.fail("A completed cancellation killed the next job"),
    )
  finally:
    store.close()


def test_failed_state_replacement_preserves_previous_ownership(tmp_path, monkeypatch):
  target = tmp_path / "processes.json"
  monkeypatch.setattr(native, "STATE", target)
  native.save_state({"session": "previous"})

  def failed_replace(*args):
    raise OSError("simulated interruption")

  monkeypatch.setattr(native.os, "replace", failed_replace)
  with pytest.raises(OSError):
    native.save_state({"session": "new"})
  assert json.loads(target.read_text()) == {"session": "previous"}
  assert list(tmp_path.iterdir()) == [target]


def test_birth_liveness_query_precedes_filetime_access(monkeypatch):
  def gone():
    raise psutil.NoSuchProcess(123)

  monkeypatch.setattr(
    native.psutil, "Process", lambda pid: SimpleNamespace(pid=pid, create_time=gone)
  )
  monkeypatch.setattr(
    native,
    "creation_filetime",
    lambda pid: pytest.fail("Accessed stale process handle"),
  )
  assert native.existing({"pid": 123, "created": 1.0, "creation_filetime": "1"}) is None


def test_reused_pid_rejected_before_inaccessible_filetime(monkeypatch):
  monkeypatch.setattr(
    native.psutil,
    "Process",
    lambda pid: SimpleNamespace(pid=pid, create_time=lambda: 2.0),
  )
  monkeypatch.setattr(
    native,
    "creation_filetime",
    lambda pid: pytest.fail("Accessed reused process handle"),
  )
  assert native.existing({"pid": 123, "created": 1.0, "creation_filetime": "1"}) is None


@pytest.mark.skipif(native.os.name != "nt", reason="Exact Windows FILETIME boundary")
def test_live_worker_birth_without_derived_psutil_float(spawned):
  process, identity, _ = spawned()
  payload = worker_record(identity)["payload"]
  payload.pop("process_created")
  restored = native.worker_identity(payload)
  assert restored["created"] is None
  assert native.existing(restored).pid == process.pid


def test_unverified_process_does_not_confirm_cancellation(monkeypatch):
  payload = {
    "pid": 123,
    "creation_filetime": "134331638753656510",
    "process_created": 1.0,
    "queue": "gpu",
    "supervisor_session": "owned",
    "current_job_id": "job-1",
    "current_attempt": 1,
  }

  def denied(*args):
    raise psutil.AccessDenied(123)

  monkeypatch.setattr(native, "existing", denied)
  assert not native.confirmed_gone({"pid": 123})
  assert not native.stop_cancelled_job(
    records({"payload": payload}), {"session": "owned"}, "job-1", 1, "gpu"
  )


def test_cleanup_continues_other_descendants_without_claiming_unverified_exit(
  monkeypatch,
):
  visited = []

  def terminate(identity):
    visited.append(identity["pid"])
    if identity["pid"] == 1:
      raise psutil.AccessDenied(1)

  monkeypatch.setattr(native, "terminate_owned", terminate)
  assert not native.stop_group(
    records(), {"descendants": {"web": {"1": {"pid": 1}, "2": {"pid": 2}}}}, "web"
  )
  assert visited == [1, 2]


@pytest.mark.parametrize("frontend_ready", [True, False])
def test_doctor_requires_api_and_frontend_readiness(
  monkeypatch, capsys, frontend_ready
):
  import redis

  from timesfm_app import config, store

  monkeypatch.setattr(
    store, "Store", lambda *args: SimpleNamespace(list_records=lambda *args: [])
  )
  monkeypatch.setattr(
    redis.Redis, "from_url", lambda *args, **kwargs: SimpleNamespace(ping=lambda: True)
  )
  monkeypatch.setattr(
    config,
    "get_settings",
    lambda: SimpleNamespace(
      database_url="unused", redis_url="unused", api_port=8001, frontend_port=3000
    ),
  )
  urls = []

  def check_http(url, **kwargs):
    urls.append(url)
    if not frontend_ready and url.endswith(":3000/"):
      raise TimeoutError("private details must not appear")

  monkeypatch.setattr(native, "http_ready", check_http)
  assert native.doctor() is frontend_ready
  result = json.loads(capsys.readouterr().out)
  assert urls == ["http://127.0.0.1:8001/api/v1/health", "http://127.0.0.1:3000/"]
  assert result["api"] == "ready"
  assert result["frontend"] == (
    "ready" if frontend_ready else "unavailable (TimeoutError)"
  )
