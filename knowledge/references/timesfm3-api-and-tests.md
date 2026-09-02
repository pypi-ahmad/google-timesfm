---
type: Reference
title: TimesFM-3 forecaster API and tests
description: Public Python API and behavior evidence from the PyTorch forecaster and its tests.
resource: https://github.com/google-research/timesfm/blob/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster.py
tags: [timesfm-3, api, pytorch, source]
status: draft
generated: { by: mcp-fetch/1.0, at: 2026-09-02T07:53:54Z }
sources:
  - id: forecaster
    resource: https://raw.githubusercontent.com/google-research/timesfm/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster.py
    title: TimesFM-3 forecaster
  - id: tests
    resource: https://github.com/google-research/timesfm/tree/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3
    title: TimesFM-3 implementation and tests
---

# Extracted evidence

`ModelConfig` defaults to the `google/timesfm-3.0-pytorch` checkpoint, 32-step
input patches, 64-step output patches, and quantiles 0.1 through 0.9.[^forecaster]

The public forecaster surface includes pretrained loading plus single and batch
prediction. Forecast outputs hold the median point forecast and optional full
quantile tensor; inputs support targets with optional past-only and
past-future covariates.[^forecaster]

The v3 source directory includes unit tests for the forecaster, model,
transformer, primitives, and CPM/RevIN refinement.[^tests]

[^forecaster]: TimesFM-3 forecaster
[^tests]: TimesFM-3 implementation and tests
