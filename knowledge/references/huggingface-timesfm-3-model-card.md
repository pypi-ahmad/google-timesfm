---
type: Reference
title: Official TimesFM 3.0 PyTorch model card
description: Cleaned source extract collected for the TimesFM-3 research bundle.
resource: https://huggingface.co/google/timesfm-3.0-pytorch
tags: [timesfm, timesfm-3, source]
status: draft
generated:
  by: crawl4ai/0.9.3
  at: 2026-09-02T07:53:31.480121Z
sources:
  - id: canonical-source
    resource: https://huggingface.co/google/timesfm-3.0-pytorch
    title: Official TimesFM 3.0 PyTorch model card
---

# Source extract

Canonical source: <https://huggingface.co/google/timesfm-3.0-pytorch>.

## Extracted content

[![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) Hugging Face](https://huggingface.co/)
# 
[ ![](https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/WtA3YYitedOr9n02eHfJe.png) ](https://huggingface.co/google)
[google](https://huggingface.co/google)
/
[timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch)
like 240
Follow
![](https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/WtA3YYitedOr9n02eHfJe.png) Google 65.9k
[ Time Series Forecasting ](https://huggingface.co/models?pipeline_tag=time-series-forecasting)[ TimesFM ](https://huggingface.co/models?library=timesfm)[ Safetensors ](https://huggingface.co/models?library=safetensors)[ PyTorch ](https://huggingface.co/models?library=pytorch)[ time-series ](https://huggingface.co/models?other=time-series)[ forecasting ](https://huggingface.co/models?other=forecasting)[ pretrained ](https://huggingface.co/models?other=pretrained)[ google ](https://huggingface.co/models?other=google)
arxiv: 2310.10688
License: timesfm-non-commercial-license-v1.0
[ Model card ](https://huggingface.co/google/timesfm-3.0-pytorch)[ Files Files and versions xet ](https://huggingface.co/google/timesfm-3.0-pytorch/tree/main)[ Community 2 ](https://huggingface.co/google/timesfm-3.0-pytorch/discussions)
Deploy
Copy to bucket new
Use this model
#  [ ](https://huggingface.co/google/timesfm-3.0-pytorch#timesfm-30-pytorch) TimesFM 3.0 (PyTorch) 
TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting.
This repository contains the official PyTorch weights and configurations for **TimesFM 3.0**.
##  [ ](https://huggingface.co/google/timesfm-3.0-pytorch#license) License 
This model is released under the **[TimesFM Non-Commercial License v1.0](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)**.
##  [ ](https://huggingface.co/google/timesfm-3.0-pytorch#model-details) Model Details 
  * **Architecture** : Stacked Mixing Transformer with Variate Attention and CPM Iterative RevIN.
  * **Context Patch Length** : 32
  * **Forecast Horizon Patch Length** : 64
  * **Layers** : 20 transformer layers (model dim: 1280, heads: 16)
  * **Quantiles** : (median at index 4)


##  [ ](https://huggingface.co/google/timesfm-3.0-pytorch#data) Data 
timesfm-3.0 is pretrained using
  * GiftEvalPretrain excluding the datasets that overlap with fev-bench
  * Wikipedia Pageviews, cutoff Nov 2023 (see paper for details).
  * Google Trends top queries, cutoff EoY 2022 (see paper for details).
  * Synthetic and augmented data.


##  [ ](https://huggingface.co/google/timesfm-3.0-pytorch#citation) Citation 
@article{das2023decoder, title={A decoder-only foundation model for time-series forecasting}, author={Das, Abhimanyu and Kong, Weihao and Sen, Rajat and Zhou, Yichen}, journal={arXiv preprint arXiv:2310.10688}, year={2023} } 

Downloads last month
    -
Safetensors[](https://huggingface.co/docs/safetensors)
Model size
0.3B params
Tensor type
F32 
·
Files info
Inference Providers [NEW](https://huggingface.co/docs/inference-providers)
Time Series Forecasting
This model isn't deployed by any Inference Provider. [🙋 Ask for provider support](https://huggingface.co/spaces/huggingface/InferenceSupport/discussions/new?title=google/timesfm-3.0-pytorch&description=React%20to%20this%20comment%20with%20an%20emoji%20to%20vote%20for%20%5Bgoogle%2Ftimesfm-3.0-pytorch%5D\(%2Fgoogle%2Ftimesfm-3.0-pytorch\)%20to%20be%20supported%20by%20Inference%20Providers.%0A%0A\(optional\)%20Which%20providers%20are%20you%20interested%20in%3F%20\(Novita%2C%20Hyperbolic%2C%20Together%E2%80%A6\)%0A)
##  Spaces using google/timesfm-3.0-pytorch 8
##  Collection including google/timesfm-3.0-pytorch
#### [TimesFM Release Collection TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model developed by Google Research for time-series forecasting. • 8 items • Updated 5 days ago • 71 ](https://huggingface.co/collections/google/timesfm-release)
##  Paper for google/timesfm-3.0-pytorch
#### [A decoder-only foundation model for time-series forecasting Paper • 2310.10688 • Published Oct 14, 2023 • 39 ](https://huggingface.co/papers/2310.10688)
