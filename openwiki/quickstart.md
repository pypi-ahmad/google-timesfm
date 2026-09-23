---
type: Guide
title: Repository quickstart
description: Start the main local workbench and find the current TimesFM-3, application, and legacy entrypoints.
tags: [quickstart, windows, workbench, timesfm-3]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-0bc10ac18ae117fcc6342194
    resource: repo://dev.ps1
  - id: openwiki-source-0a892dc1dd687b5788bf1976
    resource: repo://launch_workbench.cmd
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-64044ca87b3b58d06e5c7040
    resource: repo://src/timesfm3/__init__.py
  - id: openwiki-source-cd6ad1156d060f171c4fdf54
    resource: repo://src/timesfm3/timesfm3_forecaster.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Repository quickstart

The primary product is the local React + FastAPI workbench. On Windows 11,
install `uv`, Node.js 24 or newer, and Git, then double-click
`launch_workbench.cmd`. It performs first-run setup when needed, starts the
services, waits for readiness, and opens the browser
([README quickstart](../README.md#quick-start)).

For manual control from the repository root:

```powershell
.\dev.ps1 setup
.\dev.ps1 start
.\dev.ps1 doctor
```

Open <http://localhost:3000>. The API's interactive docs are at
<http://127.0.0.1:8001/docs>. Stop with `.\dev.ps1 stop`; use
`.\dev.ps1 dev` for frontend hot reload. See [local operations](operations.md)
for ports, worker/device settings, logs, and recovery.

## Where to start in the code

- `src/timesfm3/` is the current TimesFM-3 PyTorch package. Its public
  `TimesFM3Forecaster` is re-exported by `timesfm3`.
- `src/timesfm_app/` contains the FastAPI application, persistence, services,
  workers, and native supervisor; `web/` contains the React client.
- `streamlit_app.py` is the retained legacy explorer, separate from the primary
  workbench. See [legacy explorer](legacy-explorer.md) before using its setup or
  DuckDB history.
- `timesfm-forecasting/SKILL.md` is the repository's first-party forecasting
  agent skill ([repo instructions](../AGENTS.md#timesfm--agent-entry-point)).

For Python inference, the public entrypoint is:

```python
from timesfm3 import TimesFM3Forecaster

model = TimesFM3Forecaster.from_pretrained(device="cpu")
```

The first checkpoint load may need internet access. The default pretrained
weights have a separate usage license; check
[licensing and versions](../docs/explanation/licensing-and-versions.md) before
use. For validation commands, see [testing](testing.md).
