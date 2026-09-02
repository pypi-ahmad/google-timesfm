# Licensing and Model Versions

Code and model materials do not all share one license. Treat the boundary as a
deployment constraint, not a documentation footnote.

## License boundary

| Material | License or status | Practical effect |
|---|---|---|
| Repository source | Apache License 2.0 | May be used under the repository license |
| TimesFM weights up to 2.5 | Described upstream as Apache-2.0 | Check the selected checkpoint terms |
| Default TimesFM-3 weights | TimesFM Non-Commercial License v1.0 | No commercial or production use |
| This fork | Not an official Google product | No Google product support commitment |

Always review the license distributed with the exact checkpoint revision. The
application acknowledgement is a reminder; it is not legal advice or a grant
of rights.

## Version map

| Version | Repository location | Primary interface in this fork |
|---|---|---|
| TimesFM 3 | `src/timesfm3/` | `TimesFM3Forecaster`, `TimesFM3Evaluator`, explorer |
| TimesFM 2.5 | `src/timesfm/` | `TimesFM_2p5_200M_torch`, CSV helper |
| TimesFM 1 and 2 | Earlier Git revisions/releases | Historical implementation |

The Streamlit application always loads `google/timesfm-3.0-pytorch`. The
`timesfm-forecasting/scripts/forecast_csv.py` helper loads TimesFM 2.5 and uses
the older `ForecastConfig` API. Examples cannot be copied between those APIs
without adaptation.

## Appropriate use

The explorer is designed for local research, evaluation, and learning. It has
no authentication, persistence layer, multi-tenant isolation, production
monitoring, or service-level guarantees. Do not expose it as a public service
without a separate security and deployment design, and do not deploy the
default TimesFM-3 weights for production use.

## Research evidence

The repository's [knowledge index](../../knowledge/index.md) includes the
TimesFM-3 research report, model-card extract, weight-license extract, benchmark
sources, and limitations. These entries are currently draft and unverified;
confirm consequential claims against their cited primary sources.
