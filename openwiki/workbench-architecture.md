---
type: Architecture
title: Workbench architecture and boundaries
description: Ownership and request flow across the React client, FastAPI, durable storage, workers, and TimesFM-3 services.
tags: [workbench, architecture, fastapi, nextjs, workers]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-752929e2bab49a28d23f829a
    resource: repo://docs/codebase/ARCHITECTURE.md
  - id: openwiki-source-0b9ef32c51e8c19e2a8a6317
    resource: repo://docs/explanation/workbench-architecture.md
  - id: openwiki-source-d8aba1f61e02eca0e3f27ede
    resource: repo://docs/how-to/native-workbench.md
  - id: openwiki-source-677429aa2b6ed69a047636a3
    resource: repo://src/timesfm_app/api.py
  - id: openwiki-source-d860d6dedca5461557e32b9c
    resource: repo://src/timesfm_app/services.py
  - id: openwiki-source-01de8aeb3ad57fa80ad16688
    resource: repo://tests/test_app_api.py
  - id: openwiki-source-e7eca8f7f239774980d634af
    resource: repo://web/next.config.ts
  - id: openwiki-source-e7f75bc9b8df37775fc82ec6
    resource: repo://web/src/lib/api.ts
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Workbench architecture and boundaries

The primary product is a local Next.js/React client backed by FastAPI. The
browser owns configuration and presentation; `timesfm_app` owns the HTTP
contract, durable records, artifact access, job dispatch, and process
supervision; `timesfm3` owns data preparation, forecasting, analysis, and
tracking calculations. The legacy Streamlit client is separate
([module map](../docs/codebase/ARCHITECTURE.md#primary-workbench-architecture)).

```mermaid
flowchart LR
    Browser[React workbench] -->|/api/v1| API[FastAPI]
    API -->|records and outbox| DB[(PostgreSQL)]
    API --> Artifacts[Local or S3 artifacts]
    DB --> Dispatcher[Outbox dispatcher]
    Dispatcher --> Broker[Memurai / Dramatiq]
    Broker --> GPU[GPU worker]
    Broker --> CPU[CPU worker]
    GPU --> Services[timesfm_app services]
    CPU --> Services
    Services --> Core[timesfm3 preparation and model logic]
    Services --> Artifacts
    Services --> DB
    DB --> Events[Persistent job events / SSE]
    Events --> Browser
```

## Request and result path

The web client calls relative `/api/v1` paths through the Next.js API proxy
([client wrapper](../web/src/lib/api.ts#L1)). The FastAPI app validates local
host and browser-origin boundaries. It has no login: it accepts only configured
localhost origins for mutating browser requests, while native API clients with
no `Origin` remain supported ([API boundary](../src/timesfm_app/api.py#L116)).

Dataset upload validates the incoming table, stores source bytes as an artifact,
and creates an immutable dataset-version record with the content hash and
artifact descriptor. Re-uploading identical bytes to the same dataset returns
the existing version ([upload path](../src/timesfm_app/api.py#L297),
[duplicate identity test](../tests/test_app_api.py#L46)).

Forecast and analysis submission validates the request against workspace-owned
versions and creates a durable idempotent job. The worker calls
`timesfm_app.services.execute_spec`, which prepares inputs and performs
preflight before model resolution, then invokes the `timesfm3` functions. It
writes output artifacts before the current attempt publishes a result record
([submission](../src/timesfm_app/api.py#L204),
[`execute_spec`](../src/timesfm_app/services.py#L417)). The browser follows job
state through persisted events and an SSE endpoint. See [durable jobs](durable-jobs.md)
for retry, lease, and cancellation details.

## Workspace and local trust boundary

Workspace IDs scope records and references; they are not user identities or
security boundaries. The app is unauthenticated and configured for loopback use.
The API checks that dataset versions and parent runs belong to the requested
workspace, but this does not provide multi-user isolation
([workspace validation](../src/timesfm_app/api.py#L201),
[cross-workspace test](../tests/test_app_api.py#L103)). Do not expose the
configured service publicly as if it were an authenticated multi-user app.

The API and workers keep metadata in PostgreSQL and tabular/binary objects in a
separate local or S3 artifact store. The browser never owns inference or a saved
result. For operations and health checks, see [local operations](operations.md);
for data-to-model execution, see [forecasting pipeline](forecasting-pipeline.md).
