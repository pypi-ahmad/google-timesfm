# Workbench API reference

The local FastAPI service exposes interactive OpenAPI documentation at
<http://127.0.0.1:8001/docs>. This page describes the stable resource groups
used by the React workbench. It does not replace the generated OpenAPI schema.

## Service boundary

The API listens on loopback and has no user authentication. A workspace groups
records; it is not a security boundary. Do not expose the service publicly as
configured.

## Resources

| Resource | Main operations | Purpose |
| --- | --- | --- |
| Health and metrics | `GET /api/v1/health`, `GET /metrics` | Inspect local service state and Prometheus metrics. |
| Workspaces | List, create, retention preview/apply | Organize local records and manage opt-in retention. |
| Datasets | List, upload, demo, versions, preview | Store immutable source versions and inspect prepared data. |
| Drafts | List, create, get, patch | Persist revisioned forecast configurations. |
| Jobs | List, submit, get, cancel, retry, events | Run durable asynchronous work. |
| Runs | List, get, tables, chart, export | Read immutable completed results. |
| Tracking | List, create, assess, refresh, update, delete | Compare issued forecasts with later actuals. |
| Models and workers | List/register/check; list workers | Register checkpoints and inspect worker availability. |

## Submit and monitor a job

`POST /api/v1/jobs` requires an `Idempotency-Key` header and returns `202` with
a job record. Reuse the same key only for the same request: it returns the
original job instead of creating a duplicate. A changed request with the same
key returns `409`.

Jobs move through `queued`, `running`, terminal success/failure/cancellation,
and retry states. `GET /api/v1/jobs/{job_id}/events` is a server-sent event
stream: it emits a snapshot, persisted progress events, heartbeats, and a final
snapshot. Runs are only published after a worker owns the current attempt.

## Draft and result consistency

Draft updates include the loaded revision. A newer save from another tab causes
a conflict instead of overwriting the draft. Submission freezes the validated
specification and resolved model details. Later draft edits cannot alter a
published run.

## Generate frontend types

Export the OpenAPI document without loading a model, then regenerate the
browser boundary types:

```powershell
uv run --no-sync python -m timesfm_app.openapi .native/openapi.json
npm --prefix web run generate:api
npm --prefix web run typecheck
```

See [workbench architecture](../explanation/workbench-architecture.md) for
storage, queues, and attempt fencing.
