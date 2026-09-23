---
type: Guide
title: Analysis, scenarios, and forecast tracking
description: How TimesFM-3 historical analyses, future-covariate scenarios, uncertainty summaries, and actuals assessments work.
tags: [timesfm-3, analysis, scenarios, tracking, evaluation]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-c329abfab207b03a81378ab6
    resource: repo://src/timesfm_app/tracking_jobs.py
  - id: openwiki-source-1ea8c321f2016bda54d3ff97
    resource: repo://src/timesfm3/analysis.py
  - id: openwiki-source-98d12744281ee45d2c770127
    resource: repo://src/timesfm3/tracking.py
  - id: openwiki-source-15d7b7645e4dd2d2e1460f87
    resource: repo://src/timesfm3/uncertainty.py
  - id: openwiki-source-d3e2efd64f38c23c3d984bfc
    resource: repo://tests/test_analysis_extensions.py
  - id: openwiki-source-ab968e17ecb75be6235ea32e
    resource: repo://tests/test_analysis.py
  - id: openwiki-source-7a90cb30ccd421e3c32b0be2
    resource: repo://tests/test_tracking_jobs.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Analysis, scenarios, and forecast tracking

The analysis layer prepares a bounded schedule first, then executes it into a
complete result artifact. Supported analysis kinds include anomaly checks,
future-covariate scenarios, rolling backtests, joint-versus-independent forecasts,
covariate and settings comparisons, and naive baselines. See
[`analysis.py`](../src/timesfm3/analysis.py#L317) for preflight and
[`analysis_ui.py`](../src/timesfm3/analysis_ui.py#L236) for the retained
Streamlit workflow.

## Historical evaluation

`prepare_analysis` validates the analysis kind, window count, horizon/stride,
column roles, and resource limits before model execution. Each task is built for
one forecast origin. The history stops at that origin; held-out targets and
future values of past-only covariates are not included in its context. Tests
mutate held-out values and verify the generated context and predictions do not
change ([cutoff regression](../tests/test_analysis.py#L99),
[`_task_batch`](../src/timesfm3/analysis.py#L283)).

Backtests retain overlapping forecast pairs when windows overlap and report
metrics overall, by forecast step, and by origin. Missing actuals stay unscored
without shifting the forecast origin ([metrics](../src/timesfm3/analysis.py#L630),
[rolling test](../tests/test_analysis.py#L80)). For settings and baseline
comparisons, a key is scored only when every variant supplies a finite forecast
for the same dataset, target, origin, step, and timestamp. Comparisons reject
unequal observation counts; rankings average per-target MAE ranks with equal
target weight ([matching and ranking](../src/timesfm3/analysis.py#L703),
[comparison tests](../tests/test_analysis_extensions.py#L232)).

Anomaly analysis is one step ahead and requests sorted quantiles. An observation
is flagged only when it falls strictly outside the q0.1–q0.9 interval; missing
actuals are marked unscored rather than treated as non-anomalies
([implementation](../src/timesfm3/analysis.py#L816),
[boundary test](../tests/test_analysis.py#L139)).

## Future-covariate scenarios

A scenario template contains known-future covariate cells for the forecast
horizon. The editor validation preserves row identity and accepts only numeric
value changes; scenario execution pairs each edited variant with an unchanged
baseline. The comparison measures how the model's conditional predictions
change under those supplied covariates. It does not establish that changing a
covariate would cause the corresponding real-world outcome
([template and validation](../src/timesfm3/analysis.py#L221),
[future-only regression](../tests/test_analysis.py#L215)).

`scenario_deltas` reports absolute differences from the baseline and leaves
percentage change undefined when the baseline prediction is zero
([delta calculation](../src/timesfm3/analysis.py#L849),
[zero-baseline test](../tests/test_analysis.py#L258)).

## Quantile bands and calibration

The uncertainty helpers reshape existing quantile columns into central 20%,
40%, 60%, and 80% bands and summarize empirical coverage, width, missing actuals,
and invalid/crossed bounds. They are descriptive: they do not fit or adjust
forecast intervals. Crossed or non-finite bounds remain visible but are excluded
from valid coverage observations ([helpers](../src/timesfm3/uncertainty.py#L1),
[regression test](../tests/test_analysis_extensions.py#L269)).

## Tracking issued forecasts

Tracking compares an immutable issued forecast with newly uploaded actuals. The
caller must explicitly map each saved dataset identity to a distinct current
upload. Matching then requires the saved timestamp column, target names, exact
timestamps, and a consistent timezone convention; only observed actuals at
issued forecast timestamps are scored. The assessment stores derived comparisons
and metrics without replacing the issued forecast
([association and assessment](../src/timesfm3/tracking.py#L84),
[matching rules](../src/timesfm3/tracking.py#L118)).

The workbench reconciler looks for newer versions of the source datasets pinned
to each tracked run. An unchanged combination queues nothing; changed inputs
create an idempotent assessment job. A new forecast vintage is additionally
queued only when automatic refresh is enabled, and that refresh requires
verified model provenance and a pinned Hub revision where applicable
([reconciliation](../src/timesfm_app/tracking_jobs.py#L55),
[refresh policy](../src/timesfm_app/tracking_jobs.py#L82),
[idempotency and opt-in tests](../tests/test_tracking_jobs.py#L88)).

See [the forecast pipeline](forecasting-pipeline.md) for dataset preparation and
[persistence](persistence.md) for saved assessments and run history.
