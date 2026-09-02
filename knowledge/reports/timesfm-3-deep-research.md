---
type: Research Report
title: "TimesFM-3: Primary-Source Deep Research"
description: A cited technical and operational assessment of the TimesFM-3 checkpoint, its current repository implementation, evaluation claims, and use constraints.
tags:
  - timesfm
  - time-series-forecasting
  - foundation-models
  - multivariate-forecasting
status: draft
generated:
  by: okf-skill/0.2
  at: 2026-09-02T07:51:26Z
sources:
  - id: repo-master
    resource: https://github.com/google-research/timesfm/tree/45e0a3bc7fc4acef17b7ba7910488be2159bae5f
    title: TimesFM repository at inspected master revision
  - id: repo-readme
    resource: https://raw.githubusercontent.com/google-research/timesfm/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/README.md
    title: TimesFM README at inspected master revision
  - id: repo-api
    resource: https://github.com/google-research/timesfm/blob/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster.py
    title: TimesFM-3 PyTorch forecaster implementation
  - id: repo-tests
    resource: https://github.com/google-research/timesfm/blob/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster_test.py
    title: TimesFM-3 forecaster tests
  - id: github-tag
    resource: https://api.github.com/repos/google-research/timesfm/git/ref/tags/v3.0.0
    title: GitHub reference for TimesFM v3.0.0
  - id: google-blog
    resource: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
    title: Google Research blog — TimesFM-3
  - id: hf-model-card
    resource: https://huggingface.co/google/timesfm-3.0-pytorch
    title: Official TimesFM 3.0 PyTorch model card
  - id: hf-config
    resource: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/config.json
    title: Official TimesFM 3.0 checkpoint configuration
  - id: hf-license
    resource: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/LICENSE
    title: TimesFM Non-Commercial License v1.0
  - id: timesfm-paper
    resource: https://arxiv.org/abs/2310.10688
    title: A decoder-only foundation model for time-series forecasting
  - id: fev
    resource: https://github.com/autogluon/fev
    title: AutoGluon Forecast Evaluation library
  - id: gift-eval
    resource: https://github.com/SalesforceAIResearch/gift-eval
    title: GIFT-Eval benchmark repository
  - id: bigquery
    resource: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast
    title: BigQuery ML AI.FORECAST documentation
---

# TimesFM-3: Primary-Source Deep Research

## Executive Summary

TimesFM-3 is Google Research's current open TimesFM checkpoint family member: a 330M-parameter, zero-shot model for univariate and multivariate forecasting with past-only and known-future covariates. Google describes pretraining on real and synthetic data exceeding one trillion time points and positions the model as a single-forward-pass, probabilistic forecaster.[^google-blog]

This checkout is the upstream `master` revision `45e0a3bc7fc4acef17b7ba7910488be2159bae5f`, whose package metadata is `timesfm` 3.0.1. That is distinct from the upstream `v3.0.0` tag, which resolves to commit `331c6d33cb1ac2611de3056d0ac7164aab6301eb`; conclusions about current code should therefore be tied to master, not silently attributed to the release tag.[^repo-master][^github-tag]

The source package is Apache-2.0, but the default TimesFM-3 pretrained checkpoint is distributed under the TimesFM Non-Commercial License v1.0. The repository expressly says that those weights are restricted to non-commercial, non-production use, so Apache licensing of repository code is not authority to deploy the default weights commercially.[^repo-readme][^hf-license]

## Key Findings

1. TimesFM-3 extends the prior patched decoder-only TimesFM line with native multivariate inputs and alternating temporal/variate attention; the model card lists 20 layers, 1,280 model dimensions, 16 heads, 32-step input patches, 64-step output patches, and nine quantiles.[^timesfm-paper][^google-blog][^hf-model-card][^hf-config]

2. The public Python implementation exposes `TimesFM3Forecaster.from_pretrained`, `predict`, and `predict_batch`; it accepts NumPy contexts, optional past-only and past-future covariates, returns median point forecasts by default, and can return the full quantile tensor.[^repo-api]

3. Google claims top rank among pretrained foundation models on GIFT-Eval, FEV-Bench, and TIME under both point and probabilistic metrics. These are vendor claims, not independently reproduced results in this report; the checked FEV and GIFT repositories confirm that those are benchmark frameworks, while their live leaderboards and exact TimesFM-3 rows were not independently captured here.[^google-blog][^repo-readme][^fev][^gift-eval]

4. The sole cited TimesFM paper is the original 2023/2024 decoder-only paper. The TimesFM-3 model card cites that same paper, and the checked primary sources do not establish a dedicated TimesFM-3 technical paper.[^timesfm-paper][^hf-model-card]

## Detailed Analysis

### Lineage and release state

The original TimesFM work describes a patched-decoder attention model pretrained for zero-shot forecasting across differing histories, prediction lengths, and temporal granularities. The repository labels 2.5 as archived under `src/timesfm`, 1.0/2.0 under `v1`, and 3.0 as its latest model version.[^timesfm-paper][^repo-readme]

The README contains a release-state inconsistency: its top resource list still says a new TimesFM 3.0 blog post is “coming soon,” while Google Research published the supplied TimesFM-3 blog post on August 31, 2026. Treat the blog as current product communication and the README wording as stale editorial text, not evidence that no announcement exists.[^repo-readme][^google-blog]

### Architecture and inference

Google says TimesFM-3 groups contiguous observations into 32-step patches and applies per-series normalization. Target and past-covariate tokens contain one patch; known-future covariate tokens use a lookahead construction containing the current and future patches.[^google-blog]

The transformer is described as a two-dimensional grid with causal temporal attention within each series and full variate attention across series at a time step. These mechanisms alternate, allowing information across time and across coevolving series while preserving the stated temporal-causality constraint.[^google-blog]

For decoding, Google describes Contiguous Patch Masking: target and past-covariate horizon tokens are masked, known-future covariates remain visible, and all horizon patches are predicted in one forward pass. The stated output is nine quantiles, from 0.1 through 0.9, at each predicted step; the official checkpoint configuration independently lists exactly those nine quantiles.[^google-blog][^hf-config]

The current implementation rounds a requested horizon up to the 64-step output-patch boundary before decoding, then slices outputs back to the requested horizon. It likewise truncates or left-pads context to a patch boundary, caps global context at 15,360 steps, linearly interpolates NaNs, and supports optional z-normalization, symmetric averaging, non-negativity clipping, and quantile sorting.[^repo-api][^repo-tests]

### API and practical usage

`ModelConfig` defaults to `google/timesfm-3.0-pytorch`; `TimesFM3Forecaster` chooses CUDA when available and otherwise CPU, and `from_pretrained` can use a Hugging Face repository, local directory, or supported local checkpoint file. The `predict` convenience method delegates to `predict_batch` and returns a `ForecastOutput` containing an optional series ID, median forecast, and optionally quantiles.[^repo-api]

For univariate input, the implementation returns `(horizon,)` point forecasts and, when requested, `(horizon, 9)` quantiles. For a multivariate context shaped `(target_variates, context_length)`, it returns `(target_variates, horizon)` point forecasts and `(target_variates, horizon, 9)` quantiles; repository tests cover both forms and covariate handling.[^repo-api][^repo-tests]

Past-only covariates are aligned to context. Past-future covariates must cover context plus horizon, which matches the intended known-future-signal use case. Input series within one batch must have equal target-variate counts; callers should validate alignment, future-covariate availability, and task suitability before treating forecasts as operational decisions.[^repo-api]

Google Cloud documents a separately supported BigQuery ML `AI.FORECAST` service using a built-in TimesFM model, avoiding user-managed training and model management. This report does not equate that managed service with local use of the TimesFM-3 Hugging Face checkpoint, whose model card says it is not an officially supported Google product.[^bigquery][^hf-model-card]

### Data and evaluation

Google reports TimesFM-3 pretraining on a real-world and synthetic corpus of more than one trillion time points. The model card identifies GiftEvalPretrain excluding data overlapping FEV-Bench, Wikimedia Pageviews through November 2023, Google Trends top queries through end-2022, and synthetic/augmented data; it does not supply a complete dataset inventory or a reproducible, immutable training-data manifest.[^google-blog][^hf-model-card]

The repository includes runners/notebooks for FEV-Bench (100 tasks), GIFT-Eval, and the TIME benchmark (98 tasks). Its benchmark descriptions are useful reproduction starting points, but they do not by themselves independently validate leaderboard rank, environment, data revision, metric aggregation, or comparison-model settings.[^repo-master]

#### Separately labeled benchmark checks

**Google/TimesFM claim.** The TimesFM-3 blog says the model is top-ranked among pretrained foundation models on GIFT-Eval, FEV-Bench, and TIME for both point and probabilistic measures, and says multivariate mode improves results when cross-series information and covariates are available.[^google-blog]

**Independent framework check — FEV.** AutoGluon's FEV repository describes itself as a lightweight, reproducible forecast-evaluation library built around comparable tasks and datasets. This confirms an independently maintained evaluation framework exists; it does not independently confirm TimesFM-3's current ranking.[^fev]

**Independent framework check — GIFT-Eval.** Salesforce's GIFT-Eval repository describes a benchmark spanning seven domains, univariate/multivariate data, short/long horizons, and probabilistic forecasting. This confirms the scope of the framework; it does not independently reproduce or validate a TimesFM-3 leaderboard entry.[^gift-eval]

**TIME check.** The TimesFM repository claims TIME coverage and labels it as 50 domain datasets/98 tasks, but the referenced standalone GitHub URL returned 404 during collection. No independent TIME leaderboard or TimesFM-3 row is asserted here.[^repo-readme]

## Contrarian Views and Risks

Zero-shot benchmark performance does not guarantee accuracy for a particular business process, data-generating mechanism, target definition, intervention, or decision cost. The need to supply correctly aligned past-future covariates means leakage prevention is a caller responsibility, and the implementation’s interpolation/padding behavior can materially affect sparse or irregular data.[^google-blog][^repo-api]

“Top-ranked” benchmark language is time-sensitive and depends on benchmark version, task protocol, metric, participating models, and whether the stated comparison set is “pre-trained foundation models.” The public sources checked here provide claims and reproduction assets, not an independent rerun of the rankings.[^google-blog][^repo-master][^fev][^gift-eval]

The licensing split is a deployment risk: the repository's Apache-2.0 code license and the checkpoint's non-commercial license are separate grants. The checkpoint license also defines commercial use broadly, including revenue-generating activities, production/end-user interactions, and commercial training, fine-tuning, or distillation; obtain appropriate rights before use beyond allowed research/evaluation.[^repo-readme][^hf-license]

## Open Questions

- What precise TIME benchmark repository/leaderboard is authoritative for the claims, after the referenced GitHub path returned 404 during collection?[^repo-readme]
- Which exact benchmark versions, dates, hardware, model revisions, and aggregation rules produced the displayed ranks? The inspected claims do not provide a complete reproducibility ledger.[^google-blog][^repo-master]
- Will Google publish a TimesFM-3-specific technical paper or a complete pretraining-data and contamination-accounting record? The present model card points to the original TimesFM paper rather than a dedicated TimesFM-3 paper.[^hf-model-card][^timesfm-paper]
- What commercial licensing path applies to the TimesFM-3 checkpoint or a managed service for a given intended deployment? This requires confirmation from Google and qualified legal review, not inference from this report.[^hf-license][^bigquery]

## Sources

[^repo-master]: Inspected local checkout at upstream master `45e0a3bc7fc4acef17b7ba7910488be2159bae5f` (package metadata and benchmark assets), mirrored by [repository revision][repo-master].
[^repo-readme]: [TimesFM README at inspected master][repo-readme].
[^repo-api]: [TimesFM-3 PyTorch forecaster][repo-api].
[^repo-tests]: [TimesFM-3 forecaster tests][repo-tests].
[^github-tag]: [GitHub tag ref for `v3.0.0`][github-tag].
[^google-blog]: [Google Research: TimesFM-3][google-blog], published August 31, 2026.
[^hf-model-card]: [Official Hugging Face model card][hf-model-card].
[^hf-config]: [Official checkpoint `config.json`][hf-config].
[^hf-license]: [TimesFM Non-Commercial License v1.0][hf-license].
[^timesfm-paper]: [Das et al., *A decoder-only foundation model for time-series forecasting*][timesfm-paper], arXiv:2310.10688, revised April 17, 2024.
[^fev]: [AutoGluon FEV repository][fev].
[^gift-eval]: [Salesforce GIFT-Eval repository][gift-eval].
[^bigquery]: [BigQuery ML `AI.FORECAST` documentation][bigquery].

[repo-master]: https://github.com/google-research/timesfm/tree/45e0a3bc7fc4acef17b7ba7910488be2159bae5f
[repo-readme]: https://raw.githubusercontent.com/google-research/timesfm/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/README.md
[repo-api]: https://github.com/google-research/timesfm/blob/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster.py
[repo-tests]: https://github.com/google-research/timesfm/blob/45e0a3bc7fc4acef17b7ba7910488be2159bae5f/src/timesfm3/timesfm3_forecaster_test.py
[github-tag]: https://api.github.com/repos/google-research/timesfm/git/ref/tags/v3.0.0
[google-blog]: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
[hf-model-card]: https://huggingface.co/google/timesfm-3.0-pytorch
[hf-config]: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/config.json
[hf-license]: https://huggingface.co/google/timesfm-3.0-pytorch/raw/main/LICENSE
[timesfm-paper]: https://arxiv.org/abs/2310.10688
[fev]: https://github.com/autogluon/fev
[gift-eval]: https://github.com/SalesforceAIResearch/gift-eval
[bigquery]: https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast

## Rerun Inputs

```yaml
workflow: firecrawl-deep-research + firecrawl-knowledge-base
topic: Google TimesFM-3
depth: thorough
output: knowledge/reports/timesfm-3-deep-research.md
primary_sources:
  - https://github.com/google-research/timesfm
  - https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
  - https://huggingface.co/google/timesfm-3.0-pytorch
collection:
  firecrawl: unavailable (HTTP 402 insufficient credits)
  fallback: MCP Fetch for static/raw sources; Crawl4AI 0.9.3 for the Google Research blog
baseline:
  master: 45e0a3bc7fc4acef17b7ba7910488be2159bae5f
  v3_0_0: 331c6d33cb1ac2611de3056d0ac7164aab6301eb
retrieved_at: 2026-09-02T07:51:26Z
```
