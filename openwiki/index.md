---
okf_version: "0.2"
---

# Files

- [Analysis, scenarios, and forecast tracking](analysis-and-tracking.md) - How TimesFM-3 historical analyses, future-covariate scenarios, uncertainty summaries, and actuals assessments work.
- [Durable job dispatch, retries, and cancellation](durable-jobs.md) - How workbench submissions move through the database outbox, Dramatiq workers, attempt leases, and fenced result publication.
- [Dataset preparation and forecast pipeline](forecasting-pipeline.md) - How workbench dataset versions become grouped, prepared TimesFM inputs, previews, forecasts, and result tables.
- [Legacy Streamlit explorer and DuckDB flow](legacy-explorer.md) - How the retained local Streamlit interface uses the shared TimesFM-3 core and keeps its own session and DuckDB history.
- [TimesFM-3 model loading and inference](model-runtime.md) - Current PyTorch forecaster API, evaluator adapter, checkpoint resolution, and provenance behavior.
- [Local operations and service lifecycle](operations.md) - Setup, start, health-check, stop, and recover the loopback-only Windows workbench.
- [Records, artifacts, migrations, and retention](persistence.md) - Persistence boundaries for workbench records and artifacts, legacy DuckDB history, imports, and cleanup.
- [Repository quickstart](quickstart.md) - Start the main local workbench and find the current TimesFM-3, application, and legacy entrypoints.
- [Test and validation map](testing.md) - Focused Python, workbench, frontend, API-contract, and browser checks used by contributors and CI.
- [Workbench architecture and boundaries](workbench-architecture.md) - Ownership and request flow across the React client, FastAPI, durable storage, workers, and TimesFM-3 services.
