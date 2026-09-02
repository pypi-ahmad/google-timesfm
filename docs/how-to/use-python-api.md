# How to Use the TimesFM-3 Python API

## Install

From this checkout:

```powershell
uv sync --extra torch --group dev
```

The examples use the default checkpoint, whose weights are restricted to
non-commercial, non-production use.

## Forecast one series

```python
import numpy as np

from timesfm3 import TimesFM3Forecaster

model = TimesFM3Forecaster.from_pretrained(device="cpu")
context = np.sin(np.linspace(0, 20, 256)).astype(np.float32)

output = model.predict(
  context=context,
  horizon=24,
  return_quantiles=True,
  sort_quantiles=True,
)

assert output.forecast is not None
assert output.forecast.shape == (24,)
assert output.quantiles is not None
assert output.quantiles.shape == (24, 9)
```

For multiple targets, pass a two-dimensional context shaped
`(target_variates, context_length)`.

## Forecast a batch

```python
contexts = [
  np.random.default_rng(1).normal(size=128).astype(np.float32),
  np.random.default_rng(2).normal(size=96).astype(np.float32),
]

outputs = list(
  model.predict_batch(
    contexts=contexts,
    horizon=12,
    ts_ids=["store_a", "store_b"],
    return_quantiles=True,
  )
)
```

Batch entries may have different context lengths, but they must have the same
number of target and covariate variates.

## Add covariates

```python
rng = np.random.default_rng(7)
context_length = 128
horizon = 24

targets = rng.normal(size=(2, context_length)).astype(np.float32)
past_only = rng.normal(size=(1, context_length)).astype(np.float32)
known_future = rng.normal(
  size=(1, context_length + horizon)
).astype(np.float32)

output = model.predict(
  context=targets,
  horizon=horizon,
  past_only_covariates=past_only,
  past_future_covariates=known_future,
  return_quantiles=True,
)
```

Past-only covariates must match the context time length. Past-and-future
covariates must span `context_length + horizon`.

## Use evaluator defaults

`TimesFM3Evaluator` extends the forecaster with benchmark-oriented defaults,
optional independent-univariate mode, and deterministic chunking above 32
combined variates.

```python
from timesfm3 import TimesFM3Evaluator

evaluator = TimesFM3Evaluator.from_pretrained(device="cpu")
outputs = list(
  evaluator.predict_batch(
    contexts=[targets],
    horizon=horizon,
    return_quantiles=True,
  )
)
```

Use `TimesFM3Forecaster` when you want explicit inference controls. Use
`TimesFM3Evaluator` when you want the repository's evaluation defaults and
chunking behavior.

## Select a device and checkpoint revision

```python
model = TimesFM3Forecaster.from_pretrained(
  "google/timesfm-3.0-pytorch",
  device="cuda",
  revision="main",
  cache_dir="D:/model-cache",
)
```

Use `local_files_only=True` when the checkpoint is already cached and network
access must be disabled. A local `.safetensors`, `.pth`, or `.pt` path is also
accepted; only load checkpoint files you trust.

See the [Python API reference](../reference/python-api.md) for complete shapes,
defaults, and errors.
