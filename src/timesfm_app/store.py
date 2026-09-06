"""Transactional metadata and fenced job state; PostgreSQL is authoritative."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
  JSON,
  BigInteger,
  Boolean,
  DateTime,
  ForeignKey,
  Index,
  Integer,
  String,
  UniqueConstraint,
  create_engine,
  func,
  select,
  text,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

_TEST_OPERATION_LOCK = threading.RLock()
_ARTIFACT_OPERATION_KEY = int.from_bytes(
  hashlib.sha256(b"timesfm_app:legacy_import_and_orphan_cleanup").digest()[:8],
  "big",
  signed=True,
)


class NotFound(LookupError):
  """The requested record or job does not exist."""


class Conflict(ValueError):
  """The operation conflicts with committed state."""


class QuotaExceeded(Conflict):
  """The workspace already has its permitted number of active jobs."""


class Base(DeclarativeBase):
  pass


def _now() -> datetime:
  return datetime.now(timezone.utc)


def _iso() -> str:
  return _now().isoformat()


def _id() -> str:
  return uuid4().hex


class Record(Base):
  __tablename__ = "records"
  id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
  workspace_id: Mapped[str] = mapped_column(String(100), index=True)
  kind: Mapped[str] = mapped_column(String(40), index=True)
  name: Mapped[str] = mapped_column(String(500))
  revision: Mapped[int] = mapped_column(Integer, default=1)
  payload: Mapped[dict[str, Any]] = mapped_column(JSON)
  created_at: Mapped[str] = mapped_column(String(40), default=_iso)


class Job(Base):
  __tablename__ = "jobs"
  __table_args__ = (
    UniqueConstraint("workspace_id", "idempotency_key", name="uq_job_idempotency"),
    Index("ix_jobs_status_lease", "status", "lease_until"),
  )
  id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
  workspace_id: Mapped[str] = mapped_column(String(100), index=True)
  kind: Mapped[str] = mapped_column(String(40))
  status: Mapped[str] = mapped_column(String(20), default="queued")
  stage: Mapped[str] = mapped_column(String(100), default="queued")
  spec: Mapped[dict[str, Any]] = mapped_column(JSON)
  idempotency_key: Mapped[str] = mapped_column(String(200))
  request_hash: Mapped[str] = mapped_column(String(64))
  attempt: Mapped[int] = mapped_column(Integer, default=0)
  lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
  result_id: Mapped[str | None] = mapped_column(ForeignKey("records.id"))
  error: Mapped[dict[str, Any] | None] = mapped_column(JSON)
  created_at: Mapped[str] = mapped_column(String(40), default=_iso)
  updated_at: Mapped[str] = mapped_column(String(40), default=_iso)


class JobEvent(Base):
  __tablename__ = "job_events"
  seq: Mapped[int] = mapped_column(
    BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
  )
  job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
  type: Mapped[str] = mapped_column(String(40))
  status: Mapped[str] = mapped_column(String(20))
  stage: Mapped[str] = mapped_column(String(100))
  payload: Mapped[dict[str, Any]] = mapped_column(JSON)
  created_at: Mapped[str] = mapped_column(String(40), default=_iso)


class Outbox(Base):
  __tablename__ = "outbox"
  id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
  payload: Mapped[dict[str, Any]] = mapped_column(JSON)
  available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
  dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
  __table_args__ = (Index("ix_outbox_pending", "dispatched_at", "available_at"),)


def _dict(row: Any) -> dict[str, Any]:
  result = {column.name: getattr(row, column.name) for column in row.__table__.columns}
  return {
    key: value.isoformat() if isinstance(value, datetime) else value
    for key, value in result.items()
  }


def _contains(value: Any, record_id: str) -> bool:
  if isinstance(value, dict):
    return any(_contains(item, record_id) for item in value.values())
  if isinstance(value, list):
    return any(_contains(item, record_id) for item in value)
  return isinstance(value, str) and value == record_id


def _json_copy(value: dict[str, Any]) -> dict[str, Any]:
  return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _unexpired(job: Job) -> bool:
  if job.lease_until is None:
    return False
  lease = job.lease_until
  if lease.tzinfo is None:  # SQLite test driver does not retain timezone information.
    lease = lease.replace(tzinfo=timezone.utc)
  return lease > _now()


class Store:
  """Short synchronous transactions, with an explicit SQLite-only test factory."""

  def __init__(
    self, database_url: str, *, queued_job_limit: int = 10, lease_seconds: int = 60
  ) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
      raise ValueError("The application requires PostgreSQL; use for_testing in tests.")
    if url.drivername == "postgresql":
      url = url.set(drivername="postgresql+psycopg")
    self._configure(url, queued_job_limit, lease_seconds)

  @classmethod
  def for_testing(
    cls,
    database_url: str = "sqlite+pysqlite:///:memory:",
    *,
    queued_job_limit: int = 10,
    lease_seconds: int = 60,
  ) -> Store:
    """Create an isolated SQLite store. Never used by application configuration."""
    if make_url(database_url).get_backend_name() != "sqlite":
      raise ValueError("The test factory only accepts SQLite URLs.")
    store = cls.__new__(cls)
    store._configure(make_url(database_url), queued_job_limit, lease_seconds)
    return store

  def _configure(self, url: Any, quota: int, lease_seconds: int) -> None:
    if quota < 1 or lease_seconds < 1:
      raise ValueError("Job quota and lease duration must be positive.")
    options: dict[str, Any] = {"pool_pre_ping": True, "hide_parameters": True}
    if url.get_backend_name() == "sqlite":
      options["connect_args"] = {"check_same_thread": False}
      if url.database in (None, "", ":memory:"):
        options["poolclass"] = StaticPool
    self.engine = create_engine(url, **options)
    self.sessions = sessionmaker(self.engine, expire_on_commit=False)
    self.queued_job_limit = quota
    self.lease_seconds = lease_seconds

  def initialize(self) -> None:
    """Create missing tables for development/tests; deployments use Alembic."""
    Base.metadata.create_all(self.engine)

  def close(self) -> None:
    self.engine.dispose()

  @contextmanager
  def session(self) -> Iterator[Session]:
    with self.sessions.begin() as session:
      yield session

  def _workspace_lock(self, session: Session, workspace_id: str) -> None:
    # Serialize quota and dependency decisions even when a workspace has no rows yet.
    if self.engine.dialect.name == "postgresql":
      key = int.from_bytes(
        hashlib.sha256(workspace_id.encode()).digest()[:8], "big", signed=True
      )
      session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})

  @contextmanager
  def artifact_operation_lock(self) -> Iterator[None]:
    """Serialize legacy import and orphan deletion across application processes."""
    if self.engine.dialect.name == "sqlite":
      with _TEST_OPERATION_LOCK:
        yield
      return
    with self.engine.connect().execution_options(
      isolation_level="AUTOCOMMIT"
    ) as connection:
      connection.execute(
        text("SELECT pg_advisory_lock(:key)"), {"key": _ARTIFACT_OPERATION_KEY}
      )
      try:
        yield
      finally:
        try:
          connection.execute(
            text("SELECT pg_advisory_unlock(:key)"), {"key": _ARTIFACT_OPERATION_KEY}
          )
        except SQLAlchemyError:
          connection.invalidate()  # Do not pool a session that may retain the lock.
          raise

  def _audit(
    self,
    session: Session,
    workspace_id: str,
    action: str,
    entity_id: str,
    entity_kind: str,
    **details: Any,
  ) -> None:
    session.add(
      Record(
        workspace_id=workspace_id,
        kind="audit",
        name=action,
        payload={
          "action": action,
          "entity_id": entity_id,
          "entity_kind": entity_kind,
          **details,
        },
      )
    )

  def _require_workspace(self, session: Session, workspace_id: str) -> None:
    row = session.get(Record, workspace_id)
    if row is None or row.kind != "workspace":
      raise NotFound("Workspace not found.")

  def _validate_references(
    self, session: Session, payload: Any, workspace_id: str
  ) -> None:
    fields = {
      "dataset_id",
      "dataset_version_id",
      "dataset_version_ids",
      "run_id",
      "parent_run_id",
      "previous_run_id",
      "model_id",
      "draft_id",
      "context_id",
    }
    if not isinstance(payload, dict):
      return
    for key, value in payload.items():
      if key in {"manifest", "_resolved_model"}:
        continue  # Historical provenance is not a live application reference.
      if key in fields:
        for reference in value if isinstance(value, list) else [value]:
          if reference is None:
            continue
          row = session.get(Record, reference) if isinstance(reference, str) else None
          if row is None or row.workspace_id != workspace_id:
            raise NotFound("A referenced record is missing from this workspace.")
      elif isinstance(value, dict):
        self._validate_references(session, value, workspace_id)

  def _record(self, session: Session, record_id: str, lock: bool = False) -> Record:
    query = select(Record).where(Record.id == record_id)
    row = session.scalar(query.with_for_update() if lock else query)
    if row is None:
      raise NotFound("Record not found.")
    return row

  def _job(self, session: Session, job_id: str, lock: bool = False) -> Job:
    query = select(Job).where(Job.id == job_id)
    row = session.scalar(query.with_for_update() if lock else query)
    if row is None:
      raise NotFound("Job not found.")
    return row

  def list_records(
    self, kind: str, workspace_id: str | None = "local"
  ) -> list[dict[str, Any]]:
    with self.session() as session:
      query = select(Record).where(Record.kind == kind)
      if workspace_id is not None:
        query = query.where(Record.workspace_id == workspace_id)
      rows = session.scalars(query.order_by(Record.created_at.desc(), Record.id))
      return [_dict(row) for row in rows]

  def get_record(self, record_id: str, kind: str | None = None) -> dict[str, Any]:
    with self.session() as session:
      row = self._record(session, record_id)
      if kind is not None and row.kind != kind:
        raise NotFound("Record not found.")
      return _dict(row)

  def create_record(
    self,
    kind: str,
    name: str,
    payload: dict[str, Any],
    workspace_id: str = "local",
    record_id: str | None = None,
  ) -> dict[str, Any]:
    payload = _json_copy(payload)
    record_id = record_id or _id()
    if kind == "workspace":
      workspace_id = record_id
    with self.session() as session:
      self._workspace_lock(session, workspace_id)
      if kind != "workspace":
        self._require_workspace(session, workspace_id)
        self._validate_references(session, payload, workspace_id)
      if record_id is not None and session.get(Record, record_id) is not None:
        existing = self._record(session, record_id)
        if (
          kind == "workspace" and existing.kind == kind and existing.payload == payload
        ):
          return _dict(existing)
        raise Conflict("A record with this ID already exists.")
      row = Record(
        id=record_id or _id(),
        workspace_id=workspace_id,
        kind=kind,
        name=name,
        payload=payload,
      )
      session.add(row)
      session.flush()
      if kind not in {"audit", "worker"}:
        self._audit(
          session,
          workspace_id,
          "record.created",
          row.id,
          row.kind,
          revision=row.revision,
        )
      return _dict(row)

  def update_record(
    self,
    record_id: str,
    payload: dict[str, Any],
    expected_revision: int,
    *,
    name: str | None = None,
  ) -> dict[str, Any]:
    payload = _json_copy(payload)
    with self.session() as session:
      row = self._record(session, record_id)
      self._workspace_lock(session, row.workspace_id)
      session.expire(row)
      row = self._record(session, record_id, lock=True)
      if row.kind in {"version", "dataset_version", "run", "audit"}:
        raise Conflict("This record is immutable; create a new version.")
      if row.revision != expected_revision:
        raise Conflict("This draft changed elsewhere; reload or save a copy.")
      self._validate_references(session, payload, row.workspace_id)
      row.payload = payload
      if name is not None:
        row.name = name
      row.revision += 1
      session.flush()
      if row.kind not in {"audit", "worker"}:
        self._audit(
          session,
          row.workspace_id,
          "record.updated",
          row.id,
          row.kind,
          revision=row.revision,
        )
      return _dict(row)

  def _deletion_blockers(self, session: Session, row: Record) -> list[str]:
    records = session.scalars(
      select(Record).where(
        Record.workspace_id == row.workspace_id,
        Record.id != row.id,
        Record.kind != "audit",
      )
    )
    blockers = [
      f"record:{other.id}"
      for other in records
      if row.kind == "workspace" or _contains(other.payload, row.id)
    ]
    jobs = session.scalars(select(Job).where(Job.workspace_id == row.workspace_id))
    blockers.extend(
      f"job:{job.id}"
      for job in jobs
      if row.kind == "workspace"
      or _contains(job.spec, row.id)
      or (
        job.result_id == row.id
        and job.status not in {"succeeded", "failed", "cancelled"}
      )
    )
    return blockers

  def deletion_blockers(self, record_id: str) -> list[str]:
    with self.session() as session:
      return self._deletion_blockers(session, self._record(session, record_id))

  def delete_record(self, record_id: str) -> None:
    with self.session() as session:
      row = self._record(session, record_id)
      self._workspace_lock(session, row.workspace_id)
      row = self._record(session, record_id, lock=True)
      if self._deletion_blockers(session, row):
        raise Conflict("Other records or job history depend on this record.")
      for job in session.scalars(
        select(Job).where(Job.result_id == record_id).with_for_update()
      ):
        job.result_id = None
        self._event(session, job, "result_removed", {"removed_result_id": record_id})
      if row.kind not in {"audit", "worker"}:
        self._audit(
          session,
          row.workspace_id,
          "record.deleted",
          row.id,
          row.kind,
          revision=row.revision,
        )
      session.delete(row)

  def _event(
    self,
    session: Session,
    job: Job,
    event_type: str,
    payload: dict[str, Any] | None = None,
  ) -> None:
    job.updated_at = _iso()
    session.add(
      JobEvent(
        job_id=job.id,
        type=event_type,
        status=job.status,
        stage=job.stage,
        payload={"attempt": job.attempt, **(payload or {})},
      )
    )

    actions = {
      "queued": "job.submitted" if job.attempt == 0 else "job.retried",
      "retrying": "job.retrying",
      "cancelling": "job.cancel_requested",
      "cancelled": "job.cancelled",
      "succeeded": "job.published",
      "failed": "job.failed",
      "result_removed": "job.result_removed",
    }
    if event_type in actions:
      details = {
        key: value
        for key, value in (payload or {}).items()
        if key in {"result_id", "removed_result_id"}
      }
      self._audit(
        session,
        job.workspace_id,
        actions[event_type],
        job.id,
        "job",
        attempt=job.attempt,
        status=job.status,
        **details,
      )

  def _enqueue(self, session: Session, job: Job, delay: int = 0) -> None:
    session.add(
      Outbox(payload={"job_id": job.id}, available_at=_now() + timedelta(seconds=delay))
    )

  def _check_quota(self, session: Session, workspace_id: str) -> None:
    count = session.scalar(
      select(func.count())
      .select_from(Job)
      .where(
        Job.workspace_id == workspace_id,
        Job.status.in_(("queued", "running", "cancelling")),
      )
    )
    if count is not None and count >= self.queued_job_limit:
      raise QuotaExceeded("Workspace job limit reached; wait or cancel a queued job.")

  def create_job(
    self, workspace_id: str, kind: str, spec: dict[str, Any], idempotency_key: str
  ) -> dict[str, Any]:
    if not idempotency_key or len(idempotency_key) > 200:
      raise ValueError("An idempotency key of 1 to 200 characters is required.")
    spec = _json_copy(spec)
    digest = hashlib.sha256(
      json.dumps(
        {"kind": kind, "spec": spec},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
      ).encode()
    ).hexdigest()
    with self.session() as session:
      self._workspace_lock(session, workspace_id)
      self._require_workspace(session, workspace_id)
      existing = session.scalar(
        select(Job).where(
          Job.workspace_id == workspace_id, Job.idempotency_key == idempotency_key
        )
      )
      if existing is not None:
        if existing.request_hash != digest:
          raise Conflict("This idempotency key belongs to a different request.")
        return _dict(existing)
      self._check_quota(session, workspace_id)
      self._validate_references(session, spec, workspace_id)
      job = Job(
        workspace_id=workspace_id,
        kind=kind,
        spec=spec,
        idempotency_key=idempotency_key,
        request_hash=digest,
      )
      session.add(job)
      session.flush()
      self._event(session, job, "queued")
      self._enqueue(session, job)
      return _dict(job)

  def get_job(self, job_id: str) -> dict[str, Any]:
    with self.session() as session:
      return _dict(self._job(session, job_id))

  def list_jobs(self, workspace_id: str = "local") -> list[dict[str, Any]]:
    with self.session() as session:
      return [
        _dict(job)
        for job in session.scalars(
          select(Job)
          .where(Job.workspace_id == workspace_id)
          .order_by(Job.created_at.desc(), Job.id)
        )
      ]

  def claim_job(self, job_id: str) -> dict[str, Any] | None:
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if job.status != "queued" or job.cancel_requested:
        return None
      job.status, job.stage = "running", "starting"
      job.attempt += 1
      job.lease_until = _now() + timedelta(seconds=self.lease_seconds)
      job.error = None
      self._event(session, job, "claimed")
      return _dict(job)

  def _current(self, job: Job, attempt: int) -> bool:
    return (
      job.status == "running"
      and job.attempt == attempt
      and not job.cancel_requested
      and _unexpired(job)
    )

  def heartbeat(self, job_id: str, attempt: int) -> bool:
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if not self._current(job, attempt):
        return False
      job.lease_until = _now() + timedelta(seconds=self.lease_seconds)
      job.updated_at = _iso()
      return True

  def progress(
    self,
    job_id: str,
    attempt: int,
    stage: str,
    completed: int | None = None,
    total: int | None = None,
  ) -> bool:
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if not self._current(job, attempt):
        return False
      job.stage = stage
      self._event(session, job, "progress", {"completed": completed, "total": total})
      return True

  def freeze_job_model(
    self, job_id: str, attempt: int, model: dict[str, Any]
  ) -> dict[str, Any] | None:
    """Pin the resolved checkpoint once; every later attempt reuses that snapshot."""
    model = _json_copy(model)
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if not self._current(job, attempt):
        return None
      existing = job.spec.get("_resolved_model")
      if existing is not None:
        return existing
      job.spec = {**job.spec, "_resolved_model": model}
      self._event(session, job, "model_resolved")
      return model

  def publish_job(
    self, job_id: str, attempt: int, result_payload: dict[str, Any]
  ) -> dict[str, Any] | None:
    payload = _json_copy(result_payload)
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if not self._current(job, attempt):
        return None
      run = Record(
        workspace_id=job.workspace_id,
        kind="run",
        name=str(payload.get("name", f"{job.kind} run")),
        payload={
          **payload,
          "spec": job.spec,
          "kind": job.kind,
          "job_id": job.id,
          "attempt": job.attempt,
        },
      )
      session.add(run)
      session.flush()
      job.result_id = run.id
      job.status, job.stage, job.lease_until = "succeeded", "complete", None
      self._event(session, job, "succeeded", {"result_id": run.id})
      return _dict(run)

  def _fail(
    self, session: Session, job: Job, error: dict[str, Any], retryable: bool
  ) -> None:
    job.error = error
    job.lease_until = None
    if retryable and job.attempt < 3:
      job.status, job.stage = "queued", "retrying"
      self._enqueue(session, job, delay=2**job.attempt)
      self._event(session, job, "retrying", {"error": error})
    else:
      job.status, job.stage = "failed", "failed"
      self._event(session, job, "failed", {"error": error})

  def fail_job(
    self, job_id: str, attempt: int, error: dict[str, Any], retryable: bool = False
  ) -> bool:
    error = _json_copy(error)
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if not self._current(job, attempt):
        return False
      self._fail(session, job, error, retryable)
      return True

  def cancel_job(self, job_id: str) -> dict[str, Any]:
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if job.status in {"queued", "running"}:
        job.cancel_requested = True
        job.status = "cancelled" if job.status == "queued" else "cancelling"
        job.stage = job.status
        if job.status == "cancelled":
          job.lease_until = None
        self._event(session, job, job.status)
      return _dict(job)

  def confirm_cancel(self, job_id: str, attempt: int) -> bool:
    return self.confirm_cancel_after_exit(job_id, attempt, lambda: True)

  def confirm_cancel_after_exit(
    self, job_id: str, attempt: int, stop_callback: Callable[[], bool]
  ) -> bool:
    """Hold ownership while a supervisor proves its worker has exited."""
    with self.session() as session:
      job = self._job(session, job_id, lock=True)
      if job.attempt != attempt or job.status != "cancelling":
        return False
      if not stop_callback():
        return False
      job.status, job.stage, job.lease_until = "cancelled", "cancelled", None
      self._event(session, job, "cancelled")
      return True

  def retry_job(self, job_id: str) -> dict[str, Any]:
    with self.session() as session:
      job = self._job(session, job_id)
      self._workspace_lock(session, job.workspace_id)
      session.expire(job)
      job = self._job(session, job_id, lock=True)
      if job.status != "failed":
        raise Conflict(
          "Only failed jobs can be retried; submit a new cancelled request."
        )
      self._check_quota(session, job.workspace_id)
      job.status, job.stage, job.error = "queued", "queued", None
      self._enqueue(session, job)
      self._event(session, job, "queued")
      return _dict(job)

  def events(self, job_id: str, after: int = 0) -> list[dict[str, Any]]:
    with self.session() as session:
      self._job(session, job_id)
      return [
        _dict(row)
        for row in session.scalars(
          select(JobEvent)
          .where(JobEvent.job_id == job_id, JobEvent.seq > after)
          .order_by(JobEvent.seq)
        )
      ]

  def outbox_pending(self, limit: int = 100) -> list[dict[str, Any]]:
    with self.session() as session:
      return [
        _dict(row)
        for row in session.scalars(
          select(Outbox)
          .where(Outbox.dispatched_at.is_(None), Outbox.available_at <= _now())
          .order_by(Outbox.available_at, Outbox.id)
          .limit(limit)
        )
      ]

  def mark_dispatched(self, outbox_id: str) -> None:
    with self.session() as session:
      row = session.get(Outbox, outbox_id)
      if row is None:
        raise NotFound("Outbox entry not found.")
      if row.dispatched_at is None:
        row.dispatched_at = _now()

  def reconcile_expired(self) -> int:
    with self.session() as session:
      jobs = session.scalars(
        select(Job)
        .where(
          Job.status.in_(("running", "cancelling")),
          Job.lease_until <= _now(),
          Job.stage != "recovery_required",
        )
        .with_for_update(skip_locked=True)
      )
      count = 0
      for job in jobs:
        if job.status == "cancelling":
          job.stage = "recovery_required"
          self._event(session, job, "recovery_required")
        else:
          self._fail(
            session,
            job,
            {
              "code": "lease_expired",
              "message": "Worker lease expired before completion.",
            },
            True,
          )
        count += 1
      return count

  def reconcile_queued(self, grace_seconds: int = 30) -> int:
    """Redeliver jobs abandoned before claim, including a broker restart or busy GPU."""
    cutoff = (_now() - timedelta(seconds=grace_seconds)).isoformat()
    with self.session() as session:
      jobs = session.scalars(
        select(Job)
        .where(Job.status == "queued", Job.updated_at < cutoff)
        .with_for_update(skip_locked=True)
      )
      count = 0
      for job in jobs:
        pending = session.scalar(
          select(Outbox.id)
          .where(
            Outbox.payload["job_id"].as_string() == job.id,
            Outbox.dispatched_at.is_(None),
          )
          .limit(1)
        )
        if pending is None:
          self._enqueue(session, job)
          self._event(session, job, "redispatched")
          count += 1
      return count
