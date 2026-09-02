---
type: Architecture
title: TimesFM-3 architecture and inference
description: Patched multivariate transformer design and single-pass forecasting behavior.
tags: [timesfm-3, architecture, transformer, multivariate]
status: draft
generated: { by: okf-skill/0.2, at: 2026-09-02T07:53:54Z }
sources:
  - id: blog
    resource: /references/google-timesfm-3-blog.md
    title: Google Research TimesFM-3 launch post
  - id: config
    resource: /references/timesfm3-checkpoint-config.md
    title: TimesFM-3.0 checkpoint configuration
  - id: lineage
    resource: /references/timesfm-original-paper.md
    title: Original TimesFM paper
---

# Architecture

TimesFM-3 patches input time series, alternates causal temporal and full variate
attention, and uses masked future patches to forecast the requested horizon in
a single forward pass.[^blog]

The official checkpoint config specifies 32-step input patches, 64-step output
patches, 20 layers, 1,280 dimensions, 16 heads, and at most 32 variates.[^config]

The original TimesFM paper establishes patched-decoder lineage but is not a
TimesFM-3-specific paper.[^lineage]

[^blog]: Google Research TimesFM-3 launch post
[^config]: TimesFM-3.0 checkpoint configuration
[^lineage]: Original TimesFM paper
