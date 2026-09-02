---
type: Risk Reference
title: TimesFM-3 limitations and open questions
description: Evidence boundaries, compatibility constraints, and questions not resolved by the collected sources.
tags: [timesfm-3, limitations, risks]
status: draft
generated: { by: okf-skill/0.2, at: 2026-09-02T07:53:54Z }
sources:
  - id: config
    resource: /references/timesfm3-checkpoint-config.md
    title: TimesFM-3.0 checkpoint configuration
  - id: cpm
    resource: /references/tirex-cpm-paper.md
    title: TiRex contiguous patch masking paper
  - id: report
    resource: /reports/timesfm-3-deep-research.md
    title: TimesFM-3 primary-source deep research
---

# Constraints

The public checkpoint configuration caps variates at 32. Validate shapes and
covariate coverage before treating it as a drop-in forecasting replacement.[^config]

# Open questions

The CPM paper is TiRex work, not a TimesFM-3 paper. The collected materials do
not establish a dedicated TimesFM-3 technical paper.[^cpm][^report]

Benchmark positions, integration availability, and deployment permissions must
be rechecked at use time.[^report]

[^config]: TimesFM-3.0 checkpoint configuration
[^cpm]: TiRex contiguous patch masking paper
[^report]: TimesFM-3 primary-source deep research
