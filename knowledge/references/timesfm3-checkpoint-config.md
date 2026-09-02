---
type: Reference
title: TimesFM-3.0 checkpoint configuration
description: Machine-readable architecture and inference configuration for the official PyTorch checkpoint.
resource: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/config.json
tags: [timesfm-3, checkpoint, configuration, source]
status: draft
generated: { by: mcp-fetch/1.0, at: 2026-09-02T07:53:54Z }
sources:
  - id: config
    resource: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/config.json
    title: Official checkpoint config.json
---

# Extracted evidence

The configuration specifies input patches of 32, output patches of 64, nine
quantiles from 0.1 to 0.9, 20 layers, 1,280 model dimensions, 16 heads, and a
maximum of 32 variates.[^config]

It enables causal attention, variate attention, iterative CPM RevIN refinement,
linear detrending, stitching, and scaled-dot-product attention.[^config]

[^config]: Official checkpoint config.json
