"""Versioned local HTTP API. Inference runs exclusively in durable workers.

FastAPI app factory: validates and stages requests (schemas.py) into
store.py (durable records/jobs) and artifacts.py (immutable files), but never
runs a model itself — job execution happens out-of-process in worker.py,
delivered via jobs.py's outbox dispatcher. Read schemas.py first for the
request/response contracts, then store.py for what persisting a request
actually does.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
  FastAPI,
  File,
  Form,
  Header,
  HTTPException,
  Query,
  Request,
  UploadFile,
)
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import Settings, get_settings
from .schemas import (
  ActualsSubmission,
  JobResponse,
  JobSubmission,
  ModelSelection,
  RecordCreate,
  RecordPatch,
  RecordResponse,
  RetentionPolicy,
  RunSpec,
  TablePage,
)

LOGGER = logging.getLogger("timesfm.api")
REQUESTS = Counter("timesfm_api_requests_total", "HTTP responses", ["method", "status"])
MAX_UPLOAD = 50 * 1024 * 1024


def clean(value: Any) -> Any:
  """JSON transport never emits NaN/Infinity or changes missing labels to zero."""
  import numpy as np
  import pandas as pd

  if isinstance(value, dict):
    return {str(key): clean(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [clean(item) for item in value]
  if value is pd.NA or value is pd.NaT:
    return None
  if isinstance(value, (pd.Timestamp, datetime)):
    return value.isoformat()
  if isinstance(value, np.generic):
    return clean(value.item())
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def frame_page(frame, offset=0, limit=100):
  return {
    "columns": list(frame.columns),
    "rows": clean(frame.iloc[offset : offset + limit].to_dict("records")),
    "total": len(frame),
  }


def create_app(store=None, artifacts=None, settings: Settings | None = None) -> FastAPI:
  """Create the local workbench API with injected seams for tests."""
  from .artifacts import make_artifact_store
  from .store import Conflict, NotFound, QuotaExceeded, Store

  config = settings or get_settings()
  repository = store or Store(
    config.database_url,
    queued_job_limit=config.queued_job_limit,
    lease_seconds=config.lease_seconds,
  )
  storage = artifacts or make_artifact_store(config)

  @asynccontextmanager
  async def lifespan(application):
    try:
      await asyncio.to_thread(repository.initialize)
      try:
        repository.get_record("local", "workspace")
      except NotFound:
        repository.create_record(
          "workspace", "Local workspace", {}, workspace_id="local", record_id="local"
        )
    except SQLAlchemyError as exc:
      LOGGER.error("database_unavailable", extra={"error_type": type(exc).__name__})
    yield

  app = FastAPI(title="TimesFM-3 Workbench", version="1.0.0", lifespan=lifespan)
  app.state.store = repository
  app.state.artifacts = storage
  app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver"],
  )
  origins = [
    f"http://localhost:{config.frontend_port}",
    f"http://127.0.0.1:{config.frontend_port}",
  ]
  app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Idempotency-Key", "If-Match", "Last-Event-ID"],
  )

  @app.middleware("http")
  async def local_boundary(request, call_next):
    # Browser requests from arbitrary websites must not control an unauthenticated
    # local application. Native API clients without Origin remain supported.
    origin = request.headers.get("origin")
    if (
      request.method not in {"GET", "HEAD", "OPTIONS"}
      and origin
      and origin not in origins
    ):
      return JSONResponse(
        {"detail": "This origin cannot control the local workbench."}, status_code=403
      )
    started = time.monotonic()
    response = await call_next(request)
    REQUESTS.labels(request.method, str(response.status_code)).inc()
    LOGGER.info(
      json.dumps(
        {
          "event": "request",
          "method": request.method,
          "path": request.url.path,
          "status": response.status_code,
          "duration_ms": round((time.monotonic() - started) * 1000),
        }
      )
    )
    return response

  @app.exception_handler(NotFound)
  async def not_found(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=404)

  @app.exception_handler(Conflict)
  async def conflict(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)

  @app.exception_handler(QuotaExceeded)
  async def quota(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=429)

  @app.exception_handler(SQLAlchemyError)
  async def database_error(request, exc):
    # Log only the exception type, not str(exc): SQLAlchemy driver errors can
    # include the connection URL/credentials. The client response is likewise
    # a fixed generic message, never exc detail.
    LOGGER.error("database_error", extra={"error_type": type(exc).__name__})
    return JSONResponse(
      {
        "detail": "PostgreSQL is unavailable. Run the native diagnostic command and retry."
      },
      status_code=503,
    )

  @app.exception_handler(ValueError)
  async def validation_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=422)

  @app.exception_handler(FileNotFoundError)
  async def missing_artifact(request, exc):
    return JSONResponse(
      {
        "detail": "The stored artifact is unavailable. Restore it from backup or create a new run."
      },
      status_code=410,
    )

  def workspace(workspace_id):
    return repository.get_record(workspace_id, "workspace")

  def validate_spec(spec, workspace_id):
    # Referential checks pydantic (schemas.RunSpec) cannot do on its own: the
    # spec's ids must actually exist and belong to this workspace. Runs after
    # RunSpec.model_validate in enqueue(), which already enforced field-level
    # shape/range constraints.
    workspace(workspace_id)
    for version_id in spec["dataset_version_ids"]:
      version = repository.get_record(version_id, "dataset_version")
      if version["workspace_id"] != workspace_id:
        raise HTTPException(422, "Every dataset version must belong to this workspace.")
    if spec.get("parent_run_id"):
      parent = repository.get_record(spec["parent_run_id"], "run")
      if parent["workspace_id"] != workspace_id:
        raise HTTPException(422, "The parent run belongs to another workspace.")
    if spec["kind"] != "model_check" and not spec["dataset_version_ids"]:
      raise HTTPException(422, "Select at least one dataset version.")
    if (
      spec["kind"] not in {"model_check", "assessment"}
      and not spec["mapping"]["targets"]
    ):
      raise HTTPException(422, "Select at least one target.")

  def enqueue(kind, spec, workspace_id, key):
    """Validate a submission and atomically create its durable job."""
    spec = RunSpec.model_validate({**spec, "kind": kind}).model_dump(mode="json")
    validate_spec(spec, workspace_id)
    return repository.create_job(workspace_id, kind, spec, key)

  @app.get("/api/v1/health")
  def health():
    database = "ready"
    try:
      repository.list_records("workspace", "local")
    except SQLAlchemyError:
      database = "unavailable"
    return {
      "status": "ok" if database == "ready" else "degraded",
      "database": database,
      "storage": config.storage_backend,
      "authentication": "local",
      "version": "1.0.0",
    }

  @app.get("/metrics", include_in_schema=False)
  def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

  @app.get("/api/v1/workspaces", response_model=list[RecordResponse])
  def workspaces():
    # Workspace rows themselves are globally enumerated; their content remains
    # explicitly scoped everywhere else.
    return repository.list_records("workspace", None)

  @app.post("/api/v1/workspaces", status_code=201)
  def new_workspace(body: RecordCreate):
    identity = uuid.uuid4().hex
    return repository.create_record(
      "workspace", body.name, {}, workspace_id=identity, record_id=identity
    )

  @app.get("/api/v1/workspaces/{workspace_id}/retention")
  def retention(workspace_id: str):
    return RetentionPolicy.model_validate(
      workspace(workspace_id)["payload"].get("retention", {})
    )

  @app.patch("/api/v1/workspaces/{workspace_id}/retention")
  def set_retention(workspace_id: str, policy: RetentionPolicy):
    current = workspace(workspace_id)
    repository.update_record(
      workspace_id,
      {**current["payload"], "retention": policy.model_dump()},
      current["revision"],
    )
    return policy

  @app.get("/api/v1/workspaces/{workspace_id}/retention/preview")
  def preview_retention(workspace_id: str):
    from .maintenance import retention_preview

    return retention_preview(repository, workspace_id)

  @app.post("/api/v1/workspaces/{workspace_id}/retention/apply")
  def clean_retention(workspace_id: str):
    from .maintenance import apply_retention

    return apply_retention(repository, workspace_id)

  @app.get("/api/v1/datasets", response_model=list[RecordResponse])
  def datasets(workspace_id: str = "local"):
    workspace(workspace_id)
    return repository.list_records("dataset", workspace_id)

  def ingest(data, filename, name, workspace_id, dataset_id=None):
    from .services import ingest_metadata

    workspace(workspace_id)
    # The client-supplied filename is untrusted: strip any directory
    # component (including a Windows-style path smuggled past PurePosixPath)
    # so it can only ever contribute a bare name to the artifact key built
    # below, never a path segment.
    filename = Path(filename.replace("\\", "/")).name
    metadata = ingest_metadata(data, filename, name)
    if dataset_id:
      dataset = repository.get_record(dataset_id, "dataset")
      if dataset["workspace_id"] != workspace_id:
        raise HTTPException(422, "The dataset belongs to another workspace.")
    else:
      dataset = repository.create_record("dataset", name, {}, workspace_id=workspace_id)
    # Content-based dedup: re-uploading identical bytes for this dataset
    # returns the existing version instead of creating a duplicate artifact
    # and record, making repeated uploads of the same file idempotent.
    digest = hashlib.sha256(data).hexdigest()
    for version in repository.list_records("dataset_version", workspace_id):
      if (
        version["payload"].get("dataset_id") == dataset["id"]
        and version["payload"].get("sha256") == digest
      ):
        return version
    version_id = uuid.uuid4().hex
    descriptor = storage.put_bytes(
      f"datasets/{dataset['id']}/{version_id}/source{Path(filename).suffix.lower()}",
      data,
    )
    payload = {
      **metadata,
      "dataset_id": dataset["id"],
      "filename": filename,
      "source_name": dataset["name"],
      "sha256": digest,
      "artifact": descriptor,
    }
    record = repository.create_record(
      "dataset_version",
      f"{dataset['name']} · {version_id[:8]}",
      payload,
      workspace_id=workspace_id,
      record_id=version_id,
    )
    repository.update_record(
      dataset["id"],
      {
        **dataset["payload"],
        "latest_version_id": version_id,
        "columns": payload["columns"],
        "rows": payload["rows"],
      },
      dataset["revision"],
    )
    return record

  @app.post("/api/v1/datasets", status_code=201)
  async def upload_dataset(
    file: Annotated[UploadFile, File()],
    name: str = Form("Dataset"),
    workspace_id: str = Form("local"),
    dataset_id: str | None = Form(None),
  ):
    # Read one byte past the cap: this bounds memory for an oversized upload
    # (no unbounded buffering before the size check) while still requiring
    # only a single read call.
    data = await file.read(MAX_UPLOAD + 1)
    await file.close()
    if len(data) > MAX_UPLOAD:
      raise HTTPException(413, "Files must be at most 50 MiB.")
    return await asyncio.to_thread(
      ingest,
      data,
      file.filename or "data.csv",
      name.strip() or "Dataset",
      workspace_id,
      dataset_id,
    )

  @app.post("/api/v1/datasets/demo", status_code=201)
  def demo(workspace_id: str = "local"):
    from timesfm3.explorer import demo_dataset

    frame = demo_dataset("multivariate")
    return ingest(
      frame.to_csv(index=False).encode(), "demo.csv", "Demand demo", workspace_id
    )

  @app.get(
    "/api/v1/datasets/{dataset_id}/versions", response_model=list[RecordResponse]
  )
  def dataset_versions(dataset_id: str):
    dataset = repository.get_record(dataset_id, "dataset")
    return [
      record
      for record in repository.list_records("dataset_version", dataset["workspace_id"])
      if record["payload"].get("dataset_id") == dataset_id
    ]

  def dataset_frame(version_id: str):
    from timesfm3.explorer import parse_upload

    version = repository.get_record(version_id, "dataset_version")["payload"]
    return parse_upload(
      storage.get_bytes(version["artifact"]["key"]),
      Path(version["filename"]).suffix,
      version["source_name"],
    ).frame

  @app.get("/api/v1/datasets/versions/{version_id}/profile")
  def dataset_profile(version_id: str):
    from .dataset_explorer import profile

    return clean(profile(dataset_frame(version_id)))

  @app.get("/api/v1/datasets/versions/{version_id}/plot")
  def dataset_plot(version_id: str, column: str, x: str | None = None):
    from .dataset_explorer import plot_data

    frame = dataset_frame(version_id)
    if column not in frame.columns or (x is not None and x not in frame.columns):
      raise HTTPException(422, "Choose an existing dataset column.")
    return clean(plot_data(frame, column, x))

  @app.get("/api/v1/datasets/versions/{version_id}/preview")
  def dataset_preview(
    version_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
  ):
    return frame_page(dataset_frame(version_id), offset, limit)

  @app.post("/api/v1/preview")
  def preview(spec: RunSpec, workspace_id: str = "local"):
    from .services import preview_inputs

    workspace(workspace_id)
    return clean(
      preview_inputs(
        repository,
        storage,
        {**spec.model_dump(mode="json"), "workspace_id": workspace_id},
      )
    )

  @app.get("/api/v1/drafts", response_model=list[RecordResponse])
  def drafts(workspace_id: str = "local"):
    return repository.list_records("draft", workspace_id)

  @app.post("/api/v1/drafts", status_code=201)
  def new_draft(body: RecordCreate):
    workspace(body.workspace_id)
    return repository.create_record(
      "draft", body.name, body.payload, workspace_id=body.workspace_id
    )

  @app.get("/api/v1/drafts/{draft_id}", response_model=RecordResponse)
  def get_draft(draft_id: str):
    return repository.get_record(draft_id, "draft")

  @app.patch("/api/v1/drafts/{draft_id}")
  def save_draft(draft_id: str, body: RecordPatch, if_match: str | None = Header(None)):
    repository.get_record(draft_id, "draft")
    if if_match and if_match.strip('"') != str(body.revision):
      raise HTTPException(
        409, "Draft revision does not match. Reload or save a separate draft."
      )
    return repository.update_record(
      draft_id, body.payload, body.revision, name=body.name
    )

  @app.get("/api/v1/jobs", response_model=list[JobResponse])
  def jobs(workspace_id: str = "local"):
    return repository.list_jobs(workspace_id)

  @app.post("/api/v1/jobs", status_code=202, response_model=JobResponse)
  def submit(
    body: JobSubmission,
    idempotency_key: str = Header(..., min_length=1, max_length=200),
  ):
    return enqueue(
      body.kind, body.spec.model_dump(mode="json"), body.workspace_id, idempotency_key
    )

  @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
  def job(job_id: str):
    return repository.get_job(job_id)

  @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse)
  def cancel(job_id: str):
    return repository.cancel_job(job_id)

  @app.post("/api/v1/jobs/{job_id}/retry", status_code=202, response_model=JobResponse)
  def retry(job_id: str):
    return repository.retry_job(job_id)

  @app.get("/api/v1/jobs/{job_id}/events")
  async def events(
    job_id: str,
    request: Request,
    after: int = Query(0, ge=0),
    last_event_id: str | None = Header(None),
  ):
    initial = await asyncio.to_thread(repository.get_job, job_id)
    try:
      cursor = max(after, int(last_event_id or 0))
    except ValueError:
      raise HTTPException(422, "Last-Event-ID must be an integer.") from None

    async def stream():
      nonlocal cursor
      yield f"event: snapshot\ndata: {json.dumps(jsonable_encoder(initial))}\n\n"
      while not await request.is_disconnected():
        batch = await asyncio.to_thread(repository.events, job_id, cursor)
        for event in batch:
          cursor = event["seq"]
          yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(jsonable_encoder(event))}\n\n"
        current = await asyncio.to_thread(repository.get_job, job_id)
        if current["status"] in {"succeeded", "failed", "cancelled"}:
          yield f"event: snapshot\ndata: {json.dumps(jsonable_encoder(current))}\n\n"
          return
        yield ": heartbeat\n\n"
        await asyncio.sleep(1)

    return StreamingResponse(
      stream(),
      media_type="text/event-stream",
      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

  @app.get("/api/v1/runs", response_model=list[RecordResponse])
  def runs(workspace_id: str = "local"):
    return repository.list_records("run", workspace_id)

  @app.get("/api/v1/runs/{run_id}", response_model=RecordResponse)
  def run(run_id: str):
    return repository.get_record(run_id, "run")

  def load_table(run_id, table):
    result = repository.get_record(run_id, "run")["payload"]
    descriptor = result.get("tables", {}).get(table)
    if not descriptor:
      raise HTTPException(404, f"This run has no {table} table.")
    return storage.get_frame(descriptor["key"])

  def filter_frame(frame, dataset=None, target=None, variant=None):
    for column, value in [
      ("dataset", dataset),
      ("target", target),
      ("variant", variant),
    ]:
      if value is not None and column in frame:
        frame = frame.loc[frame[column].astype(str).eq(value)]
    return frame

  @app.get("/api/v1/runs/{run_id}/tables/{table}", response_model=TablePage)
  def table(
    run_id: str,
    table: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    dataset: str | None = None,
    target: str | None = None,
    variant: str | None = None,
    step: int | None = Query(None, ge=1),
    sort_by: str | None = None,
    descending: bool = False,
  ):
    frame = filter_frame(load_table(run_id, table), dataset, target, variant)
    if step is not None:
      if "step" not in frame:
        raise HTTPException(422, "This table has no forecast-distance column.")
      frame = frame.loc[frame["step"].eq(step)]
    if sort_by:
      if sort_by not in frame:
        raise HTTPException(422, "Unknown sort column.")
      frame = frame.sort_values(sort_by, ascending=not descending, kind="stable")
    return frame_page(frame, offset, limit)

  @app.get("/api/v1/runs/{run_id}/chart")
  def chart(
    run_id: str,
    dataset: str | None = None,
    target: str | None = None,
    variant: str | None = None,
    reference: str | None = None,
  ):
    result = repository.get_record(run_id, "run")["payload"]
    tables = result.get("tables", {})
    name = (
      "forecast"
      if "forecast" in tables
      else "predictions"
      if "predictions" in tables
      else "comparisons"
    )
    frame = load_table(run_id, name)
    datasets = (
      clean(frame["dataset"].drop_duplicates().tolist()) if "dataset" in frame else []
    )
    dataset = dataset or (str(datasets[0]) if datasets else None)
    frame = filter_frame(frame, dataset)
    targets = (
      clean(frame["target"].drop_duplicates().tolist()) if "target" in frame else []
    )
    target = target or (str(targets[0]) if targets else None)
    frame = filter_frame(frame, target=target)
    variants = (
      clean(frame["variant"].drop_duplicates().tolist()) if "variant" in frame else []
    )
    variant = variant or (str(variants[0]) if variants else None)
    all_variants = frame
    frame = filter_frame(frame, variant=variant)
    # Origins must not be silently joined into one line on rolling backtests.
    if "origin" in frame and frame["origin"].nunique() > 1:
      frame = frame.loc[frame.origin.eq(frame.origin.max())]
    origin = clean(frame.origin.iloc[0]) if "origin" in frame and len(frame) else None
    references = [str(value) for value in variants if value != variant]
    baseline = (
      filter_frame(load_table(run_id, "baselines"), dataset, target)
      if "baselines" in tables
      else None
    )
    if baseline is not None:
      references.extend(
        str(value) for value in baseline.variant.unique() if value not in references
      )
    reference_rows = []
    if reference is not None:
      if reference not in references:
        raise HTTPException(422, "Choose an available reference variant.")
      other = filter_frame(
        baseline
        if baseline is not None and reference in baseline.variant.values
        else all_variants,
        variant=reference,
      )
      if origin is not None and "origin" in other:
        other = other.loc[other.origin.eq(origin)]
      keys = [
        key
        for key in ("dataset", "target", "origin", "step", "timestamp")
        if key in frame and key in other
      ]
      if frame.duplicated(keys).any() or other.duplicated(keys).any():
        raise HTTPException(422, "Comparison contains duplicate forecast keys.")
      reference_rows = clean(
        frame[keys]
        .merge(other, on=keys, how="left", validate="one_to_one")
        .to_dict("records")
      )
    history = []
    sampled = False
    if "history" in tables and "input_context" not in tables:
      historical = filter_frame(load_table(run_id, "history"), dataset, target)
      if len(historical) > 5000:
        # Display-only min/max decimation retains spikes. Full data remains in tables.
        import numpy as np

        groups = np.array_split(np.arange(len(historical)), 2000)
        indices = {0, len(historical) - 1}
        for group in groups:
          values = historical.iloc[group]["value"]
          finite = values.dropna()
          if len(finite):
            indices.update(
              [
                int(group[np.nanargmin(values.to_numpy())]),
                int(group[np.nanargmax(values.to_numpy())]),
              ]
            )
        historical = historical.iloc[sorted(indices)]
        sampled = True
      history = clean(historical.to_dict("records"))
    elif "input_context" in tables:
      historical = filter_frame(
        load_table(run_id, "input_context"), dataset, variant=variant
      )
      historical = historical.loc[
        historical.signal.eq(target) & historical.phase.eq("history")
      ]
      history = clean(historical.to_dict("records"))
      sampled = any(
        w.get("sampled")
        for w in result.get("manifest", {}).get("input_windows", [])
        if w["dataset"] == dataset and (not variant or w["variant"] == variant)
      )
    return {
      "history": history,
      "forecast": clean(frame.to_dict("records")),
      "datasets": datasets,
      "targets": targets,
      "variants": variants,
      "dataset": dataset,
      "target": target,
      "history_sampled": sampled,
      "origin_policy": "latest",
      "variant": variant,
      "origin": origin,
      "references": references,
      "reference": reference,
      "reference_rows": reference_rows,
    }

  @app.get("/api/v1/runs/{run_id}/summary")
  def run_summary(
    run_id: str,
    dataset: str | None = None,
    target: str | None = None,
    variant: str | None = None,
    reference: str | None = None,
  ):
    import pandas as pd

    from .run_inspection import summarize

    result = repository.get_record(run_id, "run")["payload"]
    tables = result.get("tables", {})
    name = next(
      (n for n in ("forecast", "predictions", "comparisons") if n in tables), None
    )
    if name is None:
      return {"metrics": None, "delta": None}
    frame = load_table(run_id, name)
    metrics = load_table(run_id, "metrics") if "metrics" in tables else pd.DataFrame()
    summary = summarize(frame, metrics, dataset, target, variant, reference)
    if result.get("kind") == "assessment":
      summary["scope"] = "observed_actuals"
    if result.get("kind") == "covariates" and "comparisons" in tables:
      summary["sensitivity"] = filter_frame(
        load_table(run_id, "comparisons"), dataset, target
      ).to_dict("records")
    return clean(summary)

  @app.get("/api/v1/runs/{run_id}/input-context")
  def run_input_context(
    run_id: str, dataset: str | None = None, variant: str | None = None
  ):
    result = repository.get_record(run_id, "run")["payload"]
    tables = result.get("tables", {})
    windows = [
      w
      for w in result.get("manifest", {}).get("input_windows", [])
      if (not dataset or w["dataset"] == dataset)
      and (not variant or w["variant"] == variant)
    ]
    return clean(
      {
        "windows": windows,
        "rows": filter_frame(
          load_table(run_id, "input_context"), dataset, variant=variant
        ).to_dict("records")
        if "input_context" in tables
        else [],
        "events": filter_frame(
          load_table(run_id, "input_events"), dataset, variant=variant
        ).to_dict("records")
        if "input_events" in tables
        else [],
      }
    )

  @app.get("/api/v1/runs/{run_id}/replay")
  def run_replay(run_id: str):
    from .run_inspection import replay_script

    record = repository.get_record(run_id, "run")
    if not record["payload"].get("spec"):
      raise HTTPException(422, "This run has no saved specification to replay.")
    return Response(
      replay_script(record),
      media_type="text/x-python",
      headers={"Content-Disposition": 'attachment; filename="replay_run.py"'},
    )

  @app.get("/api/v1/runs/{run_id}/export")
  def export(run_id: str):
    result = repository.get_record(run_id, "run")["payload"]
    if not result.get("export"):
      raise HTTPException(404, "This run has no export.")
    content = storage.get_bytes(result["export"]["key"])
    return Response(
      content,
      media_type="application/zip",
      headers={"Content-Disposition": f'attachment; filename="timesfm3-{run_id}.zip"'},
    )

  @app.get("/api/v1/tracking", response_model=list[RecordResponse])
  def tracking(workspace_id: str = "local"):
    return repository.list_records("tracking", workspace_id)

  @app.post("/api/v1/tracking", status_code=201)
  def track(body: RecordCreate):
    parent = repository.get_record(str(body.payload.get("run_id", "")), "run")
    if (
      parent["workspace_id"] != body.workspace_id
      or parent["payload"]["kind"] != "forecast"
    ):
      raise HTTPException(422, "Track a forecast in this workspace.")
    return repository.create_record(
      "tracking",
      body.name,
      {**body.payload, "auto_refresh": bool(body.payload.get("auto_refresh", False))},
      workspace_id=body.workspace_id,
    )

  @app.post("/api/v1/tracking/{tracking_id}/assess", status_code=202)
  def assess(tracking_id: str, body: ActualsSubmission):
    tracked = repository.get_record(tracking_id, "tracking")
    parent = repository.get_record(tracked["payload"]["run_id"], "run")
    parent_spec = {
      key: value
      for key, value in parent["payload"].get("spec", {}).items()
      if key in RunSpec.model_fields
    }
    spec = {
      **parent_spec,
      "dataset_version_ids": body.dataset_version_ids,
      "parent_run_id": parent["id"],
      "associations": body.associations,
    }
    key = (
      "assessment:"
      + tracking_id
      + ":"
      + hashlib.sha256(
        json.dumps(body.model_dump(), sort_keys=True).encode()
      ).hexdigest()
    )
    return enqueue("assessment", spec, tracked["workspace_id"], key)

  @app.delete("/api/v1/tracking/{tracking_id}", status_code=204)
  def stop_tracking(tracking_id: str):
    repository.get_record(tracking_id, "tracking")
    repository.delete_record(tracking_id)
    return Response(status_code=204)

  @app.patch("/api/v1/tracking/{tracking_id}")
  def update_tracking(tracking_id: str, body: RecordPatch):
    current = repository.get_record(tracking_id, "tracking")
    if body.payload.get("run_id") != current["payload"].get("run_id"):
      raise HTTPException(
        422, "Create another tracking record to change the issued forecast."
      )
    return repository.update_record(
      tracking_id, body.payload, body.revision, name=body.name
    )

  @app.post("/api/v1/tracking/{tracking_id}/refresh", status_code=202)
  def refresh(tracking_id: str, body: ActualsSubmission):
    tracked = repository.get_record(tracking_id, "tracking")
    parent = repository.get_record(tracked["payload"]["run_id"], "run")
    spec = {
      key: value
      for key, value in parent["payload"].get("spec", {}).items()
      if key in RunSpec.model_fields
    }
    if not spec or not parent["payload"].get("manifest", {}).get("model_provenance"):
      raise HTTPException(
        422,
        "This legacy run needs source data and a verified checkpoint association before refresh.",
      )
    import dataclasses

    from timesfm3.model_loading import selection_from_provenance

    spec["model"] = dataclasses.asdict(
      selection_from_provenance(parent["payload"]["manifest"]["model_provenance"])
    )
    spec.update(
      dataset_version_ids=body.dataset_version_ids,
      parent_run_id=parent["id"],
      associations=body.associations,
    )
    spec["settings"] = {**spec["settings"], "task": "forecast"}
    key = (
      "refresh:"
      + tracking_id
      + ":"
      + hashlib.sha256(
        json.dumps(body.model_dump(), sort_keys=True).encode()
      ).hexdigest()
    )
    return enqueue("forecast", spec, tracked["workspace_id"], key)

  @app.get("/api/v1/models", response_model=list[RecordResponse])
  def models(workspace_id: str = "local"):
    return repository.list_records("model", workspace_id)

  @app.post("/api/v1/models", status_code=201)
  def add_model(body: RecordCreate):
    workspace(body.workspace_id)
    selection = ModelSelection.model_validate(body.payload)
    return repository.create_record(
      "model",
      body.name,
      selection.model_dump(mode="json"),
      workspace_id=body.workspace_id,
    )

  @app.post("/api/v1/models/{model_id}/check", status_code=202)
  def check_model(model_id: str):
    model = repository.get_record(model_id, "model")
    return enqueue(
      "model_check",
      {"model": model["payload"]},
      model["workspace_id"],
      f"model:{model_id}:{uuid.uuid4().hex}",
    )

  @app.get("/api/v1/workers", response_model=list[RecordResponse])
  def workers():
    records = repository.list_records("worker", None)
    now = datetime.now(timezone.utc)
    for record in records:
      payload = record["payload"]
      try:
        heartbeat = datetime.fromisoformat(payload["heartbeat_at"])
        available = (now - heartbeat).total_seconds() <= config.lease_seconds
      except (KeyError, ValueError, TypeError):
        available = False
      payload["available"] = available
      if not available:
        payload.update(status="offline", vram_free_gb=None)
    return sorted(records, key=lambda record: not record["payload"]["available"])

  @app.get("/api/v1/activity", response_model=list[RecordResponse])
  def activity(workspace_id: str = "local"):
    return repository.list_records("audit", workspace_id)

  if config.otlp_endpoint:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(
      BatchSpanProcessor(OTLPSpanExporter(endpoint=config.otlp_endpoint))
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
  return app


def main():
  import uvicorn

  logging.basicConfig(level=logging.INFO, format="%(message)s")
  uvicorn.run(
    create_app(),
    host="127.0.0.1",
    port=get_settings().api_port,
  )


if __name__ == "__main__":
  main()
