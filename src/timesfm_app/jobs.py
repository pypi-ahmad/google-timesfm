"""Deliver the PostgreSQL outbox to Dramatiq; duplicate delivery is safe."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from .store import Store
from .tracking_jobs import reconcile_tracking

logger = logging.getLogger(__name__)


def dispatch_once(store: Store, send_job: Callable[[str], object] | None = None) -> int:
  """Mark only acknowledged sends; a crash after send can safely deliver twice."""
  if send_job is None:
    from .worker import submit_cpu_job, submit_job

    def send_job(job_id: str) -> object:
      actor = (
        submit_cpu_job
        if store.get_job(job_id)["kind"] in {"assessment", "model_check"}
        else submit_job
      )
      return actor.send(job_id)

  store.reconcile_expired()
  store.reconcile_queued()
  reconcile_tracking(store)
  dispatched = 0
  for entry in store.outbox_pending():
    try:
      send_job(entry["payload"]["job_id"])
    except Exception:  # noqa: BLE001 - broker errors are retried from durable state.
      # Connection exceptions can include credentials. Keep the durable entry pending.
      logger.warning("Job dispatch failed; the outbox entry remains pending.")
      continue
    store.mark_dispatched(entry["id"])
    dispatched += 1
  return dispatched


def dispatch_loop(store: Store, stop: threading.Event, interval: float = 1.0) -> None:
  while not stop.is_set():
    try:
      dispatch_once(store)
    except Exception:  # noqa: BLE001 - keep the dispatcher alive without logging secrets.
      logger.warning("Outbox polling failed; polling will resume.")
    stop.wait(interval)


def main() -> None:
  from .config import get_settings

  settings = get_settings()
  store = Store(settings.database_url, queued_job_limit=settings.queued_job_limit)
  stop = threading.Event()
  try:
    dispatch_loop(store, stop)
  except KeyboardInterrupt:
    stop.set()
  finally:
    store.close()


if __name__ == "__main__":
  main()
