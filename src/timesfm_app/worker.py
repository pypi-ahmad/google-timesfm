"""Spawn-safe Dramatiq actors with fenced jobs and one resident GPU process.

Launch with ``dramatiq timesfm_app.worker --use-spawn -p 1 -t 1 -Q gpu``.
The separate CPU worker consumes ``cpu``; importing this module never imports
torch or initializes CUDA. Application leases, not broker retries, own recovery.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import AgeLimit, Callbacks, Middleware, Pipelines
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from .config import get_settings

logger = logging.getLogger(__name__)
_jobs = Counter(
  "timesfm_worker_jobs_total", "Completed job attempts", ["kind", "outcome", "device"]
)
_runtime = Histogram(
  "timesfm_worker_job_seconds",
  "Job attempt execution time",
  ["kind", "device"],
  buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600),
)
_state = Gauge("timesfm_worker_state", "Current worker state", ["state", "device"])
_heartbeat = Gauge(
  "timesfm_worker_heartbeat_timestamp_seconds",
  "Last successful status heartbeat",
  ["device"],
)
_vram_free = Gauge(
  "timesfm_worker_vram_free_bytes", "Available device memory", ["device"]
)
_vram_total = Gauge(
  "timesfm_worker_vram_total_bytes", "Total device memory", ["device"]
)
broker = RedisBroker(
  url=get_settings().redis_url,
  middleware=[AgeLimit(), Callbacks(), Pipelines()],
)
dramatiq.set_broker(broker)


class _JSONFormatter(logging.Formatter):
  def format(self, record: logging.LogRecord) -> str:
    payload = {
      "timestamp": dt.datetime.fromtimestamp(
        record.created, dt.timezone.utc
      ).isoformat(),
      "level": record.levelname,
      "logger": record.name,
      "message": record.getMessage(),
      "pid": os.getpid(),
    }
    for name in ("job_id", "attempt", "kind", "outcome"):
      if hasattr(record, name):
        payload[name] = getattr(record, name)
    if record.exc_info and record.exc_info[0] is not None:
      payload["error_type"] = record.exc_info[0].__name__
    return json.dumps(payload)


class GPUWorkerBusy(Exception):
  """A live worker already owns this machine's GPU execution slot."""


class OwnershipLost(Exception):
  """The current attempt may no longer publish or start another model call."""


class GPUProcessLock:
  """An OS-owned handle released on process death, including after PC sleep."""

  def __init__(self, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    self.handle: Any = None
    if os.name == "nt":
      from ctypes import wintypes

      kernel = ctypes.WinDLL("kernel32", use_last_error=True)
      kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
      ]
      kernel.CreateFileW.restype = wintypes.HANDLE
      kernel.CloseHandle.argtypes = [wintypes.HANDLE]
      kernel.CloseHandle.restype = wintypes.BOOL
      # No FILE_SHARE_* flags: another process cannot open the same handle.
      handle = kernel.CreateFileW(
        str(path.resolve()), 0xC0000000, 0, None, 4, 0x80, None
      )
      if handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        if error in {32, 33}:
          raise GPUWorkerBusy("Another GPU worker is still alive.")
        raise ctypes.WinError(error)
      self.kernel = kernel
      self.handle = handle
    else:
      import fcntl

      handle = path.open("a+b")
      try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
      except BlockingIOError as exc:
        handle.close()
        raise GPUWorkerBusy("Another GPU worker is still alive.") from exc
      self.handle = handle

  def close(self) -> None:
    """For tests/shutdown only; production retains the handle until process exit."""
    if self.handle is not None:
      if os.name == "nt":
        self.kernel.CloseHandle(self.handle)
      else:
        self.handle.close()
      self.handle = None


def _creation_filetime() -> str | None:
  """Exact Windows process birth value for PID-reuse-safe supervisor checks."""
  if os.name != "nt":
    return None
  from ctypes import wintypes

  kernel = ctypes.WinDLL("kernel32", use_last_error=True)
  kernel.GetCurrentProcess.restype = wintypes.HANDLE
  kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [
    ctypes.POINTER(wintypes.FILETIME)
  ] * 4
  kernel.GetProcessTimes.restype = wintypes.BOOL
  times = [wintypes.FILETIME() for _ in range(4)]
  if not kernel.GetProcessTimes(
    kernel.GetCurrentProcess(), *(ctypes.byref(t) for t in times)
  ):
    raise ctypes.WinError(ctypes.get_last_error())
  return str((times[0].dwHighDateTime << 32) | times[0].dwLowDateTime)


_gpu_lock: GPUProcessLock | None = None
_process_identity = uuid.uuid4().hex
_process_started = dt.datetime.now(dt.timezone.utc).isoformat()
_process_creation = _creation_filetime()
_process_queue: str | None = None
_process_state = "idle"
_active_job: dict[str, Any] = {}
_status_lock = threading.Lock()


def _own_gpu() -> None:
  global _gpu_lock
  if _gpu_lock is None:
    root = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir())) / "TimesFM"
    _gpu_lock = GPUProcessLock(root / "gpu-worker.lock")


def _synchronize() -> None:
  torch = sys.modules.get("torch")
  if torch is not None and torch.cuda.is_initialized():
    try:
      torch.cuda.synchronize()
    except RuntimeError:
      # A poisoned CUDA context cannot safely serve another attempt. Process exit
      # releases the OS handle; the supervisor and durable lease recover the job.
      logger.critical("CUDA synchronization failed; retiring the worker process")
      os._exit(70)


def _worker_status(store: Any, device: str, status: str) -> None:
  with _status_lock:
    _worker_status_locked(store, device, status)


def _worker_status_locked(store: Any, device: str, status: str) -> None:
  """Publish capabilities from the worker that owns the device."""
  from .store import NotFound

  identifier = f"worker-{os.getpid()}"
  payload: dict[str, Any] = {
    "pid": os.getpid(),
    "process_identity": _process_identity,
    "supervisor_session": os.environ.get("TIMESFM_SUPERVISOR_SESSION"),
    "started_at": _process_started,
    "creation_filetime": _process_creation,
    "queue": _process_queue,
    "current_job_id": _active_job.get("id"),
    "current_attempt": _active_job.get("attempt"),
    "device": device,
    "status": status,
    "heartbeat_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "cached_model": None,
    "vram_free_gb": None,
    "vram_total_gb": None,
  }
  services = sys.modules.get("timesfm_app.services")
  predictor = getattr(services, "_cached_predictor", None)
  if predictor is not None:
    payload["cached_model"] = getattr(predictor, "model_provenance", None)
  torch = sys.modules.get("torch")
  if device == "cuda" and torch is not None and torch.cuda.is_initialized():
    free, total = torch.cuda.mem_get_info()
    payload.update(vram_free_gb=free / 2**30, vram_total_gb=total / 2**30)
  try:
    old = store.get_record(identifier, kind="worker")
  except NotFound:
    store.create_record("worker", identifier, payload, record_id=identifier)
  else:
    store.update_record(identifier, payload, expected_revision=old["revision"])
  _heartbeat.labels(device).set(time.time())
  for label in ("idle", "running", "cancelling"):
    _state.labels(label, device).set(int(status == label))
  if payload["vram_free_gb"] is not None:
    _vram_free.labels(device).set(payload["vram_free_gb"] * 2**30)
    _vram_total.labels(device).set(payload["vram_total_gb"] * 2**30)


def _retryable(exc: Exception) -> bool:
  """Retry transient transport failures, never invalid data or memory exhaustion."""
  causes = []
  current: BaseException | None = exc
  while current is not None and current not in causes:
    causes.append(current)
    current = current.__cause__ or current.__context__
  if any(
    "outofmemory" in type(item).__name__.lower() or "out of memory" in str(item).lower()
    for item in causes
  ):
    return False
  return any(
    isinstance(item, (ConnectionError, TimeoutError, OSError)) for item in causes
  )


def run_job(job_id: str, *, cpu: bool = False) -> None:
  """Claim once, heartbeat independently, and fence every native-call boundary."""
  global _process_state, _active_job
  from .artifacts import make_artifact_store
  from .store import Store

  config = get_settings()
  store = Store(
    config.database_url,
    queued_job_limit=config.queued_job_limit,
    lease_seconds=config.lease_seconds,
  )
  if not cpu:
    while True:
      try:
        _own_gpu()
        break
      except GPUWorkerBusy:
        # Retain the unacknowledged message until the OS owner exits. A lease
        # expiry never overrides physical ownership or opens a second context.
        if store.get_job(job_id)["status"] != "queued":
          store.close()
          return
        threading.Event().wait(1)
  job = store.claim_job(job_id)
  if job is None:
    store.close()
    return
  attempt = job["attempt"]
  _active_job = {"id": job_id, "attempt": attempt}
  spec = job["spec"]
  if cpu != (spec["kind"] in {"assessment", "model_check"}):
    store.fail_job(
      job_id,
      attempt,
      {"code": "wrong_queue", "message": "Job was sent to the wrong worker queue."},
      retryable=False,
    )
    store.close()
    _active_job = {}
    return
  # Import the numerical modules only inside the spawned actor process.
  from opentelemetry import trace

  from .services import CancellationRequested, execute_spec

  span = trace.get_tracer(__name__).start_span(
    "timesfm.execute",
    attributes={
      "job.id": job_id,
      "job.attempt": attempt,
      "job.kind": spec["kind"],
    },
  )

  stop = threading.Event()
  lost = threading.Event()
  device = "cpu" if cpu else config.device
  started = time.perf_counter()
  outcome = "failed"
  logger.info(
    "Job attempt started",
    extra={"job_id": job_id, "attempt": attempt, "kind": spec["kind"]},
  )

  def heartbeat() -> None:
    while not stop.wait(config.heartbeat_seconds):
      try:
        if not store.heartbeat(job_id, attempt):
          lost.set()
          return
      except Exception:  # noqa: BLE001 - ownership must fail closed on any heartbeat failure.
        lost.set()  # Database uncertainty fails closed at the next boundary.
        logger.error("Worker heartbeat failed; execution ownership is uncertain")
        return

  def checkpoint() -> None:
    # The authoritative ownership check: renews the lease (store.heartbeat)
    # and fails closed if it can't, meaning a later attempt/claim may already
    # own this job row. Every checkpoint() call is a point where execution
    # must be safe to abandon without side effects on shared state.
    state = store.get_job(job_id)
    if state and state["attempt"] == attempt and state.get("cancel_requested"):
      raise CancellationRequested("Cancellation requested.")
    if lost.is_set() or not store.heartbeat(job_id, attempt):
      raise OwnershipLost("Job ownership expired or changed.")

  def progress(stage: str, completed: int | None, total: int | None) -> None:
    checkpoint()
    span.add_event(
      "progress", {"stage": stage, "completed": completed or 0, "total": total or 0}
    )
    if not store.progress(job_id, attempt, stage, completed, total):
      checkpoint()
      raise OwnershipLost("Progress ownership expired.")

  thread = threading.Thread(target=heartbeat, name="job-heartbeat", daemon=True)
  try:
    checkpoint()
    _process_state = "running"
    _worker_status(store, device, _process_state)
    thread.start()
    execution_spec = {
      **spec,
      "workspace_id": job["workspace_id"],
      "_device": device,
      "_artifact_prefix": f"jobs/{job_id}/attempt-{attempt}",
      "_model_snapshot_root": str(config.artifact_root / "model_snapshots"),
    }
    result = execute_spec(
      store,
      make_artifact_store(config),
      execution_spec,
      progress,
      checkpoint,
      freeze_model=lambda model: store.freeze_job_model(job_id, attempt, model),
    )
    checkpoint()
    _synchronize()
    # Artifacts for this attempt are already written by the time we get here
    # (execute_spec staged them before returning); publish_job only records
    # the result pointer. If ownership was lost in that gap, the written
    # files are simply orphaned bytes at an attempt-scoped key that
    # maintenance.py's orphan sweep can later reclaim - never partially
    # visible application state.
    if store.publish_job(job_id, attempt, result) is None:
      checkpoint()
      raise OwnershipLost("Completed output lost publication ownership.")
    span.set_attribute("job.status", "succeeded")
    outcome = "succeeded"
  except CancellationRequested:
    # Keep 'cancelling' visible until no native kernel is still using the device.
    _process_state = "cancelling"
    _synchronize()
    store.confirm_cancel(job_id, attempt)
    span.set_attribute("job.status", "cancelled")
    outcome = "cancelled"
  except OwnershipLost:
    _synchronize()
    logger.info("Stopped stale job %s attempt %s", job_id, attempt)
    span.set_attribute("job.status", "ownership_lost")
    outcome = "ownership_lost"
  except Exception as exc:  # noqa: BLE001 - actor boundary persists all execution failures.
    _synchronize()
    from timesfm3.explorer import ExplorerError

    message = (
      str(exc)
      if isinstance(exc, ExplorerError)
      else f"Execution failed ({type(exc).__name__})."
    )
    store.fail_job(
      job_id,
      attempt,
      {"code": type(exc).__name__, "message": message},
      retryable=_retryable(exc),
    )
    span.set_attribute("job.status", "failed")
    span.set_attribute("error.type", type(exc).__name__)
    logger.error("Job %s attempt %s failed (%s)", job_id, attempt, type(exc).__name__)
  finally:
    stop.set()
    if thread.is_alive():
      thread.join(timeout=config.heartbeat_seconds + 1)
    _process_state = "idle"
    _active_job = {}
    try:
      _worker_status(store, device, _process_state)
    except Exception:  # noqa: BLE001 - cleanup must still close the database pool.
      logger.warning("Could not publish idle worker status")
    store.close()
    _jobs.labels(spec["kind"], outcome, device).inc()
    _runtime.labels(spec["kind"], device).observe(time.perf_counter() - started)
    logger.info(
      "Job attempt completed",
      extra={
        "job_id": job_id,
        "attempt": attempt,
        "kind": spec["kind"],
        "outcome": outcome,
      },
    )
    span.end()


class _WorkerLifecycle(Middleware):
  """Acquire the device before consuming messages and report idle heartbeats."""

  def before_worker_boot(self, broker, worker) -> None:
    global _process_queue
    for handler in logging.getLogger().handlers:
      handler.setFormatter(_JSONFormatter())
    queues = worker.consumer_whitelist
    if queues not in ({"gpu"}, {"cpu"}):
      raise SystemExit(
        "Select exactly one worker queue with --queues gpu or --queues cpu."
      )
    if worker.worker_threads != 1:
      raise SystemExit("TimesFM workers require --threads 1.")
    self.device = "cpu" if queues == {"cpu"} else get_settings().device
    _process_queue = "cpu" if queues == {"cpu"} else "gpu"
    self.metrics_port = 9202 if queues == {"cpu"} else 9201
    if queues == {"gpu"}:
      try:
        _own_gpu()
      except GPUWorkerBusy as exc:
        raise SystemExit(str(exc)) from exc
    self.stop = threading.Event()
    if get_settings().otlp_endpoint:
      from opentelemetry import trace
      from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
      from opentelemetry.sdk.resources import Resource
      from opentelemetry.sdk.trace import TracerProvider
      from opentelemetry.sdk.trace.export import BatchSpanProcessor

      provider = TracerProvider(
        resource=Resource.create({"service.name": "timesfm-worker"})
      )
      provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=get_settings().otlp_endpoint))
      )
      trace.set_tracer_provider(provider)

  def after_worker_boot(self, broker, worker) -> None:
    self.metrics_server = None
    if getattr(get_settings(), "worker_metrics_enabled", True):
      self.metrics_server, _ = start_http_server(self.metrics_port, addr="127.0.0.1")

    def report() -> None:
      from .store import Store

      config = get_settings()
      store = Store(config.database_url, lease_seconds=config.lease_seconds)
      try:
        while not self.stop.is_set():
          try:
            _worker_status(store, self.device, _process_state)
          except Exception:  # noqa: BLE001 - telemetry failure cannot stop a worker.
            logger.warning("Worker status unavailable; reporting will resume")
          self.stop.wait(config.heartbeat_seconds)
      finally:
        store.close()

    self.thread = threading.Thread(target=report, name="worker-status", daemon=True)
    self.thread.start()

  def before_worker_shutdown(self, broker, worker) -> None:
    self.stop.set()
    self.thread.join(timeout=5)
    if self.metrics_server is not None:
      self.metrics_server.shutdown()
      self.metrics_server.server_close()


broker.add_middleware(_WorkerLifecycle())


@dramatiq.actor(queue_name="gpu", broker=broker)
def submit_job(job_id: str) -> None:
  run_job(job_id)


@dramatiq.actor(queue_name="cpu", broker=broker)
def submit_cpu_job(job_id: str) -> None:
  run_job(job_id, cpu=True)
