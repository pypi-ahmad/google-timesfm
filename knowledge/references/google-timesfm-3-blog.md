---
type: Reference
title: Google Research TimesFM-3 launch post
description: Cleaned source extract collected for the TimesFM-3 research bundle.
resource: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
tags: [timesfm, timesfm-3, source]
status: draft
generated:
  by: crawl4ai/0.9.3
  at: 2026-09-02T07:53:30.600818Z
sources:
  - id: canonical-source
    resource: https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/
    title: Google Research TimesFM-3 launch post
---

# Source extract

Canonical source: <https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/>.

## Extracted content

[Skip to main content](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/#page-content)
[ Google Research ](https://research.google/)
Search
![TimesFM-3 architecture diagram illustrating time series patching, transformer layers, and multivariate forecast output.](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM31_Architecture.png)
# TimesFM-3: A zero-shot foundation model for multivariate forecasting
August 31, 2026
Ayush Jain and Rajat Sen, Research Scientists, Google Research
We introduce TimesFM-3, a state-of-the-art time series foundation model that enables highly accurate multivariate time series forecasting in a single forward pass, significantly outperforming other forecasting models across major benchmarks.
## Quick links
  * [ GitHub ](https://github.com/google-research/timesfm)
  * [ HuggingFace ](https://huggingface.co/google/timesfm-3.0-pytorch)
  * Share
    * [ ](https://twitter.com/intent/tweet?text=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ "Share on Twitter")
    * [ ](https://www.facebook.com/sharer/sharer.php?u=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ "Share on Facebook")
    * [ ](https://www.linkedin.com/shareArticle?url=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/&mini=true "Share on LinkedIn")
    *     * Copy link
× 


Since the debut of [TimesFM](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/) in 2024, we’ve seen the adoption of time-series foundation models for real-world time-series forecasting tasks across multiple domains, such as [retail, finance, observability, manufacturing, healthcare and natural sciences](https://icml-structured-fm-workshop.github.io/).
Up until [TimesFM-2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch) (released in September 2025), our models were strictly limited to univariate forecasting: forecasting using only the history of a single time series. Yet, most real-world forecasting problems are inherently multivariate: where multiple time series and auxiliary external features jointly impact the future forecast of a time series. Consider forecasting ice cream sales for a retail chain. Past sales alone rarely tell the full story. A good forecast should also draw on sales of related products (e.g., ice cream cones, syrups), historical foot traffic, and known future events like weather forecasts, promotions, and holidays.
Today we introduce [TimesFM-3](https://huggingface.co/google/timesfm-3.0-pytorch), the next generation of our time-series foundation model that is natively pre-trained for multivariate forecasting. TimesFM-3 has 330 million parameters and is pre-trained on a real-world and synthetic time-series corpus comprising more than 1 trillion time points. Building on the efficiency and zero-shot generalization of its predecessors, TimesFM-3 adds robust support for complex multivariate scenarios in a zero-shot manner. It can jointly predict multiple coevolving time series, capturing dependencies that improve overall accuracy without requiring task-specific fine-tuning. The model natively supports:
  * _Multiple targets:_ Forecast multiple related time series simultaneously (e.g., jointly forecasting different brands of ice cream). The model supports both point and quantile forecasts for all targets.
  * _Past covariates:_ Incorporate features that are only known historically (e.g., past foot traffic).
  * _Past-future (dynamic) covariates:_ Leverage known future events to guide the forecast (e.g., planned promotional campaigns or weather forecasts).


## Under the hood: Architecture & inference
TimesFM-3 builds on the proven decoder-only [transformer architecture](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/) of its predecessors. As in previous versions, we process time series efficiently by grouping contiguous data points into patches of 32 time steps. We then apply normalization per time-series similar to [that of TimesFM-2.5](https://github.com/google-research/timesfm/blob/master/src/timesfm/flax/util.py#L43) in order to account for time series with vastly different scales.
### Multivariate token construction
For target and past-covariate series, a token is constructed directly from a single patch. However, for past-future covariates, TimesFM-3 employs a clever "lookahead" strategy: each token concatenates the current patch with future patches, allowing the model to peek at upcoming known signals.
### Alternating attention architecture
Once the patches are tokenized, they pass through an input residual block and enter the main transformer stack, which operates as a 2D grid:
  1. _Causal temporal attention:_ Tokens attend horizontally across time. To prevent data leakage, this attention is strictly causal — a token can only look at past tokens within its _own_ specific time series.
  2. _Full variate attention:_ Tokens attend vertically across series. At any given time step, a token can look at all other time series in the dataset, allowing the model to learn complex cross-series correlations (e.g., how a promotion in one series affects sales in another).


These two attention mechanisms alternate for several layers, seamlessly blending temporal patterns with cross-series relationships.
![TimesFM-3 architecture diagram illustrating time series patching, transformer layers, and multivariate forecast output.](https://storage.googleapis.com/gweb-research2023-media/images/TimesFM31_Architecture.width-1250.png)
_TimesFM-3 architecture._
### Non-autoregressive decode: Forecasting in a single pass
Previous versions of TimesFM generated forecasts one patch at a time, introducing latency, compounding error accumulation, and computational cost. TimesFM-3 uses the strategy of [Contiguous Patch Masking](https://arxiv.org/abs/2505.23719) to generate the entire forecasting horizon in a single forward pass. The model appends masked placeholder tokens for the future horizon alongside the observed context. Target and past-covariate series are masked in the horizon (since their future values are unknown), while past–future covariates remain visible, providing the model with known future signals like holidays or scheduled events. Through the alternating attention layers, the model fills in all masked horizon patches simultaneously, with no iterative loop required. The model predicts 9 quantiles (from the 10th to the 90th percentile) for each target time series at every horizon step, providing a full probabilistic view of the forecast uncertainty.
## Illustrative example for multivariate forecasting
Let’s revisit the ice cream sales example. Imagine you are working on next month’s promotion schedule and want to forecast the sales to anticipate. A standard univariate model (the red line, below) looks at the historical sales and projects a weekly pattern forward — but it has no idea about planned promotions on specific days. TimesFM-3's multivariate mode (the blue line, below) takes a different approach: by passing in the planned promotion schedule as a past-future covariate, the model learns the relationship between promotions and sales lift from the historical context, then applies that knowledge to future days with planned promotions. The result is a forecast that anticipates a ~20% sales bump on each promotion day. In the chart below, the amber blocks in the promotion covariates highlight which days have promotions — and the blue forecast visibly responds to each one, while the red forecast does not. Over the full month, this adds up to a more accurate forecast for projected revenue.
![Time series line chart displaying sales forecast with promotion covariate spikes highlighted across the prediction window.](https://storage.googleapis.com/gweb-research2023-media/images/TimesFM3_PromotionsGraph.width-1250.png)
_Planning promotions: TimesFM-3's multivariate forecast uses a promotion covariate to anticipate sales lift on planned promotion days in the future._
## Evaluation and benchmarks
We evaluated TimesFM-3 on three comprehensive public forecasting benchmarks: [Gift-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), [FEV-Bench](https://huggingface.co/spaces/autogluon/fev-bench), and [Time](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard). On all three benchmarks, TimesFM-3 is the top-ranked model in terms of both point and probabilistic forecasting metrics among all pre-trained foundation models. The plots below show average rank across tasks for both point forecast accuracy and probabilistic forecast quality (lower is better) for the three benchmarks. We compare against recent foundation models including multivariate-capable models, such as Chronos-2 and the Toto 2.0 family, as well as our previous model TimesFM-2.5.
Each plot includes two entries for TimesFM-3. The "univariate mode" point shows performance when the model is evaluated without any covariate or cross-series information, treating each target series independently, just like a traditional univariate model. Even in this univariate mode, TimesFM-3 already matches or outperforms other competing models. When we switch to the full multivariate mode, TimesFM-3 takes another leap, achieving the best average rank in both point and probabilistic forecasting across the board.
![Benchmark evaluation chart comparing TimesFM-3 against leading forecasting models on the GIFT-Eval benchmark.](https://storage.googleapis.com/gweb-research2023-media/images/TimesFM33_Gift-Eval.width-1250.png)
![Benchmark chart showing TimesFM-3 forecasting accuracy across multivariate datasets on FEV-Bench.](https://storage.googleapis.com/gweb-research2023-media/images/TimesFM34_Fev-Bench.width-1250.png)
![Inference time and efficiency comparison chart highlighting the speed and latency of TimesFM-3 across forecasting horizons.](https://storage.googleapis.com/gweb-research2023-media/images/TimesFM35_Time.width-1250.png)
_Performance on Gift-Eval (_**_top_** _), Fev-Bench (_**_middle_** _), and Time (_**_bottom_** _): TimesFM-3 in univariate mode already outperforms other replicable time-series foundation models in both point and probabilistic forecasting metrics. Multivariate mode further improves performance by leveraging cross-series information and covariates when available._
## Conclusion
We introduce TimesFM-3, the latest generation of our TimesFM family of zero-shot time series foundation models, that obtains state-of-the-art multivariate and univariate forecasting performance on multiple public benchmarks. TimesFM-3 is now available on [GitHub](https://github.com/google-research/timesfm) and [Hugging Face](https://huggingface.co/google/timesfm-3.0-pytorch), with its BigQuery integration landing in the coming weeks. In the meantime, you can try TimesFM-2.5 immediately on your univariate tasks to familiarize yourself with the [AI.FORECAST](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast) command in BigQuery - no ML expertise required.
## Acknowledgements
_This project is joint work with Yichen Zhou, Petros Mol, Abhimanyu Das and Samet Oymak._
  * Labels:
  * [Data Management](https://research.google/blog/label/data-management)
  * [Machine Intelligence](https://research.google/blog/label/machine-intelligence)
  * [Product](https://research.google/blog/label/product)


## Quick links
  * [ GitHub ](https://github.com/google-research/timesfm)
  * [ HuggingFace ](https://huggingface.co/google/timesfm-3.0-pytorch)
  * Share
    * [ ](https://twitter.com/intent/tweet?text=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ "Share on Twitter")
    * [ ](https://www.facebook.com/sharer/sharer.php?u=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/ "Share on Facebook")
    * [ ](https://www.linkedin.com/shareArticle?url=https%3A//research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/&mini=true "Share on LinkedIn")
    *     * Copy link
× 


## Other posts of interest
  * [ ![Four satellite maps showing concentrated gas emission plumes over California, Turkmenistan, Shanxi, and Delhi.](https://storage.googleapis.com/gweb-research2023-media/original_images/MAPL-EMIT-overview-hero.png) September 1, 2026 Mapping global methane emissions from space with deep learning 
    * Climate & Sustainability ·
    * Earth AI ·
    * Machine Intelligence  ](https://research.google/blog/mapping-global-methane-emissions-from-space-with-deep-learning/)
  * [ ![Flowchart detailing the four stages of the Planetary Prediction Engine from data selection to final report generation.](https://storage.googleapis.com/gweb-research2023-media/original_images/PlanetaryPredictionEngine_Cover.png) August 27, 2026 Planetary prediction engine: Automating global models via Earth AI 
    * Earth AI ·
    * Generative AI ·
    * Machine Intelligence  ](https://research.google/blog/planetary-prediction-engine-automating-global-models-via-earth-ai/)
  * [ ![Overview diagram of the GlucoFM framework detailing data preprocessing, JEPA-style pre-training architecture, and clinical forecasting applications.](https://storage.googleapis.com/gweb-research2023-media/original_images/GlucoFM1_Overview.png) August 26, 2026 GlucoFM: Foundation model for continuous glucose monitoring 
    * Health & Bioscience ·
    * Machine Intelligence  ](https://research.google/blog/glucofm-foundation-model-for-continuous-glucose-monitoring/)


× ❮ ❯
![TimesFM34_Fev-Bench](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM34_Fev-Bench.png)
Benchmark chart showing TimesFM-3 forecasting accuracy across multivariate datasets on FEV-Bench. 
![TimesFM31_Architecture](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM31_Architecture.png)
TimesFM-3 architecture diagram illustrating time series patching, transformer layers, and multivariate forecast output. 
![TimesFM3_PromotionsGraph](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM3_PromotionsGraph.png)
Time series line chart displaying sales forecast with promotion covariate spikes highlighted across the prediction window. 
![TimesFM35_Time](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM35_Time.png)
Inference time and efficiency comparison chart highlighting the speed and latency of TimesFM-3 across forecasting horizons. 
![TimesFM33_Gift-Eval](https://storage.googleapis.com/gweb-research2023-media/original_images/TimesFM33_Gift-Eval.png)
Benchmark evaluation chart comparing TimesFM-3 against leading forecasting models on the GIFT-Eval benchmark. 
×
