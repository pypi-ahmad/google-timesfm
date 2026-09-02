---
type: API Reference
title: TimesFM-3 Python API and input shapes
description: Public PyTorch forecaster surface, forecast outputs, and covariate inputs.
tags: [timesfm-3, api, pytorch, covariates]
status: draft
generated: { by: okf-skill/0.2, at: 2026-09-02T07:53:54Z }
sources:
  - id: api
    resource: /references/timesfm3-api-and-tests.md
    title: TimesFM-3 forecaster API and tests
  - id: readme
    resource: /references/upstream-repository.md
    title: TimesFM upstream repository baseline
---

# API

Use the public `ModelConfig` and pretrained forecaster APIs for single or batch
inference. Point forecasts are median quantiles; full quantiles are optional.[^api]

# Inputs

Contexts may be one-dimensional univariate arrays or `(variates, time)` arrays.
Past-only covariates cover observed context; past-future covariates also cover
the requested horizon.[^api][^readme]

[^api]: TimesFM-3 forecaster API and tests
[^readme]: TimesFM upstream repository baseline
