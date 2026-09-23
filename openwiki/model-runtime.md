---
type: Reference
title: TimesFM-3 model loading and inference
description: Current PyTorch forecaster API, evaluator adapter, checkpoint resolution, and provenance behavior.
tags: [timesfm-3, pytorch, inference, checkpoints]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-64044ca87b3b58d06e5c7040
    resource: repo://src/timesfm3/__init__.py
  - id: openwiki-source-0ae45088b98222af0d7a06d4
    resource: repo://src/timesfm3/evaluator.py
  - id: openwiki-source-9f5981be4ae310f79c8c2746
    resource: repo://src/timesfm3/model_loading.py
  - id: openwiki-source-25b8c4882ba757037c16268a
    resource: repo://src/timesfm3/timesfm3_forecaster_test.py
  - id: openwiki-source-cd6ad1156d060f171c4fdf54
    resource: repo://src/timesfm3/timesfm3_forecaster.py
  - id: openwiki-source-2d2579edb1cb938296abcef8
    resource: repo://tests/test_checkpoint_selection.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# TimesFM-3 model loading and inference

The current model implementation is `src/timesfm3`, a PyTorch TimesFM-3
forecaster. This page describes that package only; the repository also retains
older TimesFM 2.5 code for compatibility, which is not the active TimesFM-3
runtime ([repository scope](../AGENTS.md)).

## Python API

The public `TimesFM3Forecaster` accepts a single series through `predict` or a
list of series through `predict_batch`. Inputs may be one-dimensional for a
single target or shaped `(variates, time)` for multivariate targets. Past-only
covariates cover context rows; past-future covariates cover context plus the
requested horizon. The returned `ForecastOutput.forecast` is selected from the
configured median-quantile channel; full quantiles are optional
([API](../src/timesfm3/timesfm3_forecaster.py#L120),
[`predict` and `predict_batch`](../src/timesfm3/timesfm3_forecaster.py#L478),
[shape tests](../src/timesfm3/timesfm3_forecaster_test.py#L180)).

```python
import numpy as np
from timesfm3 import TimesFM3Forecaster

model = TimesFM3Forecaster.from_pretrained(
    "google/timesfm-3.0-pytorch", device="cpu"
)
result = model.predict(
    context=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    horizon=4,
    return_quantiles=True,
)
```

The basic `TimesFM3Forecaster` defaults `return_quantiles` to false. The
`TimesFM3Evaluator` adapter used by the repository's forecasting pipeline
provides benchmark defaults, including quantiles, sorted quantiles, symmetric
averaging, and non-negativity handling; it also supports high-dimensional
benchmark chunking and independent univariate mode
([adapter defaults](../src/timesfm3/evaluator.py#L42)). Use the interface that
matches the caller rather than assuming the two classes share identical
defaults.

The forecaster validates batch list lengths, context/covariate ranks, and the
time lengths of both covariate types. `past_future_covariates` must provide
exactly context plus horizon rows. Inference pads/truncates to model patch
boundaries internally and returns only the requested horizon
([input validation](../src/timesfm3/timesfm3_forecaster.py#L510),
[regressions](../src/timesfm3/timesfm3_forecaster_test.py#L91)).

## Selecting and resolving checkpoints

`ModelSelection` makes the source explicit: a Hugging Face repository or a
local path, with optional revision and offline mode. Hub resolution downloads
only `config.json` and `model.safetensors`; local folders require those files,
while standalone `.safetensors`, `.pt`, and `.pth` files are accepted. Resolved
files receive SHA-256 fingerprints that become part of provenance and model
identity ([selection and resolver](../src/timesfm3/model_loading.py#L27),
[`resolve_model`](../src/timesfm3/model_loading.py#L70)).

Loading uses the resolved local path and disables further network access. Folder
weights are checked for exact parameter names and shapes, and incompatibility
raises a user-facing error instead of silently falling back to another model.
The loaded evaluator carries the selection, resolved revision, local path, and
file fingerprints as provenance ([loader](../src/timesfm3/model_loading.py#L127),
[checkpoint tests](../tests/test_checkpoint_selection.py#L24)).

The local workbench freezes a resolved model snapshot for retries; see
[durable jobs](durable-jobs.md). For runtime/device setup, see
[local operations](operations.md). For the data arrays passed to inference, see
[forecasting pipeline](forecasting-pipeline.md).
