---
type: Guide
title: Legacy Streamlit explorer and DuckDB flow
description: How the retained local Streamlit interface uses the shared TimesFM-3 core and keeps its own session and DuckDB history.
tags: [legacy, streamlit, duckdb, timesfm-3]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-47362d42244342e6d0b1beed
    resource: repo://launch_app.cmd
  - id: openwiki-source-51dd5f97d993733fe7bd93f9
    resource: repo://src/timesfm3/run_store.py
  - id: openwiki-source-964922c41d4be25ac5f159ec
    resource: repo://streamlit_app.py
  - id: openwiki-source-10168fd9970eb88a5388836e
    resource: repo://tests/test_run_store.py
  - id: openwiki-source-6ad70607402c6dcfd96b9274
    resource: repo://tests/test_streamlit_app.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Legacy Streamlit explorer and DuckDB flow

`streamlit_app.py` is the retained local Streamlit dashboard for preparing data,
running forecasts, evaluating results, and tracking forecast actuals. Unlike the
primary React workbench, it loads the model in the Streamlit process and runs
inference directly; it does not use the FastAPI job backend
([module boundary](../streamlit_app.py#L4)). On Windows, `launch_app.cmd` starts
it on port 9587. The repository README identifies the [React workbench as the
primary app](../README.md#quick-start) and labels this UI legacy.

## Typical flow

The tabs separate preparation, forecast settings, evaluation, and tracking. The
Prepare tab accepts uploaded files or a demo, maps timestamp/target/covariate
roles, and can apply grouping and calendar preparation. The Forecast tab chooses
future forecasting or holdout evaluation, series mode, horizon, context length,
device, and inference options. Running a forecast requires acknowledging the
default checkpoint's usage restriction in the UI
([tab and form behavior](../streamlit_app.py#L592),
[app test](../tests/test_streamlit_app.py#L39)).

The shared core is in `timesfm3.explorer`, `data_preparation`, `analysis`, and
`tracking`; Streamlit modules render those APIs rather than owning their
algorithms. `analysis_ui` prepares before model acquisition and submits analysis
only through the explicit form. `data_preparation_ui` previews grouping and
calendar changes. `tracking_ui` lets a user associate updated uploads with a
saved forecast, assess actuals, and optionally create a refreshed forecast
vintage. See [forecast pipeline](forecasting-pipeline.md) and
[analysis and tracking](analysis-and-tracking.md) for the shared behavior.

## Session state and model cache

Uploaded tables are parsed and cached only for the current Streamlit session.
The model uses `st.cache_resource` with one resident configuration; switching
checkpoint, device, or batch size clears the cached model before acquiring a new
one ([session cache](../streamlit_app.py#L89),
[upload cache](../streamlit_app.py#L142)). Persisted forecasts can be loaded
without their original uploads; starting the interface with its demo does not
load a model ([startup test](../tests/test_streamlit_app.py#L39),
[saved-run test](../tests/test_streamlit_app.py#L62)).

## Local history

Forecasts and analyses use `data/timesfm.duckdb`, next to `streamlit_app.py`.
`run_store` persists derived run tables and manifests transactionally. Saved
forecast vintages are immutable; identical saves are idempotent and conflicting
content under an existing run ID is rejected. The default run retention keeps
the newest 25 untracked forecasts while protecting tracked runs
([`save_run`](../src/timesfm3/run_store.py#L227),
[retention](../src/timesfm3/run_store.py#L300),
[store tests](../tests/test_run_store.py#L64)). Tracking assessments append
versions without changing the issued forecast, and saved analyses have their
own bounded history ([assessment storage](../src/timesfm3/run_store.py#L602),
[analysis storage](../src/timesfm3/run_store.py#L676)).

DuckDB history is best-effort at startup: if it cannot be read, the app warns
and continues with session-only history ([fallback](../streamlit_app.py#L129),
[fallback test](../tests/test_streamlit_app.py#L85)). The newer React workbench
uses PostgreSQL records, separate artifact storage, and queued workers; see
[workbench architecture](workbench-architecture.md) and
[persistence](persistence.md).
