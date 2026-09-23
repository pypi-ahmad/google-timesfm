---
type: Runbook
title: Local operations and service lifecycle
description: Setup, start, health-check, stop, and recover the loopback-only Windows workbench.
tags: [operations, windows, services, monitoring]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-0bc10ac18ae117fcc6342194
    resource: repo://dev.ps1
  - id: openwiki-source-d8aba1f61e02eca0e3f27ede
    resource: repo://docs/how-to/native-workbench.md
  - id: openwiki-source-4be3050cbc27c9c38810df5f
    resource: repo://monitoring/prometheus.yml
  - id: openwiki-source-2ebd2db3d9b62c2e8041c033
    resource: repo://src/timesfm_app/config.py
  - id: openwiki-source-798655e74ad7113aacd1a88a
    resource: repo://src/timesfm_app/native.py
  - id: openwiki-source-32d78e8d1b4f768a0ccccc9e
    resource: repo://src/timesfm_app/worker.py
  - id: openwiki-source-3ff2c4d7c5276f24e0cd5216
    resource: repo://tests/test_app_native.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Local operations and service lifecycle

The React workbench is supervised locally. The native setup and supervisor are
Windows-specific; on Windows 11, use the repository's PowerShell entrypoint.
The supervisor owns the API, dispatcher, GPU worker, CPU worker, and Next.js
frontend processes, plus the local PostgreSQL and Memurai dependencies
([launcher](../dev.ps1#L1),
[`supervise`](../src/timesfm_app/native.py#L455)).

## Setup and daily commands

Run setup after installing `uv`, Node.js, and Git, or double-click
`launch_workbench.cmd` for first-time setup followed by startup and readiness
checks. The launcher waits up to 90 seconds for `doctor`, opens the frontend,
then streams labeled service logs. Closing that log view leaves services running.

```powershell
.\dev.ps1 setup
.\dev.ps1 start
.\dev.ps1 doctor
.\dev.ps1 stop
```

Use `.\dev.ps1 dev` for frontend hot reload. Stop the workbench before switching
between normal and development mode. Setup uses `uv` to preserve an existing
CUDA-capable PyTorch installation, provisions local native dependencies, and
installs/builds the web frontend ([commands](../dev.ps1#L107)).

## Local topology and configuration

The defaults are loopback-only:

| Component | Address | Role |
|---|---|---|
| Next.js frontend | `127.0.0.1:3000` | Browser UI |
| FastAPI | `127.0.0.1:8001` | Workbench API and `/docs` |
| PostgreSQL | `127.0.0.1:55432` | Durable application records and outbox |
| Memurai | `127.0.0.1:56379` | Redis-compatible job delivery |
| GPU worker metrics | `127.0.0.1:9201` | GPU queue metrics |
| CPU worker metrics | `127.0.0.1:9202` | CPU queue metrics |

The API and frontend ports, database and queue URLs, artifact root, worker
device, heartbeat/lease timing, and optional OTLP endpoint are configured through
`TIMESFM_`-prefixed environment variables. Defaults and supported fields are in
[`config.py`](../src/timesfm_app/config.py#L17); settings are cached within each
process. The supervisor starts one GPU-queue process and one CPU-queue process;
assessment and model-check jobs use the CPU queue
([queue routing](../src/timesfm_app/native.py#L476),
[worker metrics](../src/timesfm_app/worker.py#L447)).

To explicitly use CPU for forecast inference, set the worker device before
starting the processes:

```powershell
$env:TIMESFM_DEVICE = 'cpu'
.\dev.ps1 start
```

Device choice is explicit; an out-of-memory error does not automatically switch
an inference job to CPU. See the [native workbench guide](../docs/how-to/native-workbench.md#install-and-start)
for hardware and license notes.

## Health, logs, and safe recovery

`.\dev.ps1 doctor` checks database access, queue connectivity, API health, and
frontend readiness, and reports native PostgreSQL/Memurai files. It exits
successfully only when the database, queue, API, and frontend are ready
([doctor](../src/timesfm_app/native.py#L595),
[health test](../tests/test_app_native.py#L296)). Logs are written under
`.native/logs/`.

The supervisor records process identities and stops only processes it can
verify as owned by its run. Cancellation also verifies the worker's job,
attempt, queue, session, and process identity before termination
([ownership tests](../tests/test_app_native.py#L111)). If a worker or service
fails, inspect `doctor` output and the matching log first; run `.\dev.ps1 stop`
and then `.\dev.ps1 start` if a clean restart is appropriate. Stop leaves
PostgreSQL available for saved data. `.native/` and `data/product/` contain
persistent state and are not ordinary cleanup targets
([stop behavior](../src/timesfm_app/native.py#L648),
[native workbench guide](../docs/how-to/native-workbench.md#recovery-and-diagnostics)).

The repository's Prometheus scrape configuration targets the API and both
worker metric endpoints ([scrape targets](../monitoring/prometheus.yml)). For
job lease, retry, and cancellation behavior, see [durable jobs](durable-jobs.md).
