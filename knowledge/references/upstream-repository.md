---
type: Reference
title: TimesFM upstream repository baseline
description: README, package metadata, and agent-entry-point evidence at the inspected master revision.
resource: https://github.com/google-research/timesfm/tree/45e0a3bc7fc4acef17b7ba7910488be2159bae5f
tags: [timesfm, source, repository]
status: draft
generated: { by: mcp-fetch/1.0, at: 2026-09-02T07:53:54Z }
sources:
  - id: repository
    resource: https://github.com/google-research/timesfm/tree/45e0a3bc7fc4acef17b7ba7910488be2159bae5f
    title: TimesFM master revision
  - id: readme
    resource: https://raw.githubusercontent.com/google-research/timesfm/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/README.md
    title: TimesFM README
  - id: pyproject
    resource: https://raw.githubusercontent.com/google-research/timesfm/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/pyproject.toml
    title: TimesFM package metadata
---

# Extracted evidence

At this revision the package version is `3.0.1`, requires Python 3.10 or newer,
and declares NumPy, Hugging Face Hub, and SafeTensors as base dependencies.[^pyproject]

The README identifies TimesFM 3.0 as the latest model version, describes native
multivariate forecasting and past-only/past-future covariates, and warns that
the default 3.0 weights are non-commercial and non-production.[^readme]

The repository agent entry point names `timesfm-forecasting/SKILL.md`; its
development-path statement references `src/timesfm/`, while v3 implementation
currently lives in `src/timesfm3/`. Treat that as a source inconsistency.[^repository]

[^repository]: TimesFM master revision
[^readme]: TimesFM README
[^pyproject]: TimesFM package metadata
