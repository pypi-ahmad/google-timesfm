---
type: Guide
title: Dataset preparation and forecast pipeline
description: How workbench dataset versions become grouped, prepared TimesFM inputs, previews, forecasts, and result tables.
tags: [timesfm-3, forecasting, data-preparation, covariates]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-d860d6dedca5461557e32b9c
    resource: repo://src/timesfm_app/services.py
  - id: openwiki-source-0efb7eee04c662fa35f1b8e4
    resource: repo://src/timesfm3/data_preparation.py
  - id: openwiki-source-7ea7129cb9c1462ea0d88732
    resource: repo://src/timesfm3/explorer.py
  - id: openwiki-source-54157c753ff292f759f3f781
    resource: repo://tests/test_app_services.py
  - id: openwiki-source-c7991ae09f75f45bfc125796
    resource: repo://tests/test_data_preparation.py
  - id: openwiki-source-e0cc848cdeec929cbc02a85b
    resource: repo://tests/test_explorer.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Dataset preparation and forecast pipeline

The primary workbench sends pinned dataset-version IDs and forecast settings to
`timesfm_app.services`. That service reads and verifies the stored source bytes,
parses the table, applies grouping and calendar preparation, validates model
inputs, then calls the shared `timesfm3` forecasting layer. The API and UI
workflow is described in [workbench architecture](workbench-architecture.md);
this page focuses on the data and numerical path.

## Ingest and preparation

The shared parser accepts CSV and Parquet, rejects unsupported or empty inputs,
checks upload and decoded-memory limits, and computes a content hash
([`parse_upload`](../src/timesfm3/explorer.py#L191)). In the workbench, the
service verifies each saved artifact against its recorded SHA-256 before
parsing. It then checks dataset versions and column roles, splits selected
grouping columns into stable series identities, applies calendar settings, and
tracks the preparation metadata ([service path](../src/timesfm_app/services.py#L75),
[`group_sources`](../src/timesfm3/data_preparation.py#L284),
[`prepare_sources`](../src/timesfm3/data_preparation.py#L360)).

Calendar preparation may append timestamp rows to cover the requested horizon
and generate future calendar covariates; it does not fill target values. A
frequency must be resolvable when future timestamp rows are needed
([calendar extension](../src/timesfm3/data_preparation.py#L185),
[regression test](../tests/test_data_preparation.py#L120)). The quality report
surfaces readiness, missing context, duplicate/invalid timestamps, and cadence
warnings without modifying or interpolating the data
([`quality_report`](../src/timesfm3/data_preparation.py#L490),
[quality test](../tests/test_data_preparation.py#L322)).

The workbench preview uses the same grouping and preparation policy as execution
but does not resolve a model. It returns per-group quality, sample rows, input
windows where available, optional imputation previews, and a known-future
covariate scenario template ([preview implementation](../src/timesfm_app/services.py#L176),
[preview test](../tests/test_app_services.py#L201)). Invalid groups can be
reported as blocked separately; selected groups can be excluded in the run
specification.

## Building model inputs

`prepare_batch` validates the column mapping and forecast settings, verifies
aggregate upload limits, and enforces the 32-variate limit unless benchmark
chunking is explicitly enabled ([batch preparation](../src/timesfm3/explorer.py#L409)).
Each series is converted to a numeric target context, optional past-only
covariates, optional past-and-future covariates, and an aligned future time axis.
Holdout mode saves the withheld target values for scoring; forecast mode has no
actuals for its future horizon. Missing context values are retained for the
forecaster's interpolation step, while known-future covariates must be present
through the full horizon ([series construction](../src/timesfm3/explorer.py#L449),
[covariate and holdout tests](../tests/test_explorer.py#L137)).

Grouped dataset identity and calendar metadata are preserved with each prepared
series. Preparation tests cover order-independent group identity, group-role
validation, calendar extension, and avoiding changes to uploaded targets
([grouping tests](../tests/test_data_preparation.py#L44),
[calendar tests](../tests/test_data_preparation.py#L120)).

## Inference and outputs

The explorer makes one explicit `predict_batch` call for the prepared series and
forwards the horizon, covariates, and inference settings. Returned point
forecasts and quantiles are shape- and finiteness-checked before they become a
long-form table keyed by dataset, target, step, and timestamp
([execution and table conversion](../src/timesfm3/explorer.py#L581)). Holdout
metrics use rows with both actual and point values; forecast tables without
actuals do not produce holdout scores ([metrics](../src/timesfm3/explorer.py#L663)).

In workbench execution, preparation and analysis preflight happen before model
resolution. The resulting forecast tables include forecast, history, metrics,
and calibration summaries; the service also records input-window context for
inspection and builds a portable export ([execution service](../src/timesfm_app/services.py#L475),
[result tables](../src/timesfm_app/services.py#L525)). A service-level test
checks that invalid analysis input fails before model resolution
([preflight regression](../tests/test_app_services.py#L274)).

The legacy Streamlit explorer also uses `timesfm3.explorer`, but has a separate
UI and DuckDB run history; see [legacy explorer](legacy-explorer.md). Historical
analysis schedules, scenarios, and tracking are covered in
[analysis and tracking](analysis-and-tracking.md). The forecaster and checkpoint
selection are covered in [model runtime](model-runtime.md).
