# TimesFM-3 Python API Reference

The public `timesfm3` package exports `TimesFM3Forecaster`,
`TimesFM3Evaluator`, `TimesFM3Torch`, `ModelConfig`, `ForecastOutput`, and model
configuration types. Most applications should start with a forecaster or
evaluator loaded from a checkpoint.

## `TimesFM3Forecaster.from_pretrained`

```python
TimesFM3Forecaster.from_pretrained(
  pretrained_model_name_or_path="google/timesfm-3.0-pytorch",
  device=None,
  **kwargs,
)
```

Creates an inference wrapper from a Hugging Face repository, local checkpoint
directory, `.safetensors` file, or `.pth`/`.pt` file.

Useful keyword arguments include `per_core_batch_size`, `cache_dir`,
`force_download`, `token`, `revision`, and `local_files_only`. If `device` is
omitted, CUDA is selected when available; otherwise CPU is used.

## `TimesFM3Forecaster.predict`

```python
model.predict(
  context,
  horizon,
  past_only_covariates=None,
  past_future_covariates=None,
  ts_id=None,
  return_quantiles=False,
  use_symmetric_averaging=False,
  make_positive=False,
  sort_quantiles=True,
  use_znorm=False,
  padding_mode="none",
)
```

Runs one forecast and returns `ForecastOutput`.

## `TimesFM3Forecaster.predict_batch`

Accepts lists of contexts and optional companion lists. Every companion list
must have one entry per context.

### Input shapes

| Input | One series | Multivariate series |
|---|---|---|
| `context` | `(context_length,)` | `(targets, context_length)` |
| `past_only_covariates` | `(context_length,)` | `(covariates, context_length)` |
| `past_future_covariates` | `(context_length + horizon,)` | `(covariates, context_length + horizon)` |

Across a batch, context lengths may differ. Target counts and each present
covariate type's variate count must be consistent.

The forecaster rejects a nonpositive horizon, an empty batch, empty or
three-dimensional arrays, companion-length mismatches, incompatible time
lengths, and inconsistent variate counts.

### Missing values

Non-finite values are converted to missing values before per-variate linear
interpolation. Leading fully missing steps are trimmed. An entirely missing
context is converted to zeros by the low-level forecaster; the Streamlit
explorer applies stricter validation and rejects targets without observations.

## `TimesFM3Evaluator.predict_batch`

The evaluator adds:

- quantiles, symmetric averaging, positivity inference, and quantile sorting on
  by default
- `padding_mode="none"` and `use_znorm=False`
- `univariate=True` to forecast input channels independently
- deterministic covariate subsampling and target chunking above 32 variates

The explorer requires users to approve benchmark chunking before invoking this
behavior on high-dimensional joint forecasts.

## `ForecastOutput`

| Field | Type | Shape |
|---|---|---|
| `ts_id` | `str \| None` | Scalar identifier |
| `forecast` | `np.ndarray \| None` | `(horizon,)` or `(targets, horizon)` |
| `quantiles` | `np.ndarray \| None` | `(horizon, 9)` or `(targets, horizon, 9)` |

Quantile indices correspond to `0.1` through `0.9`; the point forecast is the
median quantile.

## `ModelConfig`

`ModelConfig` is the public alias for the forecaster configuration dataclass.
Checkpoint-derived architecture settings are synchronized after loading.

| Field | Default | Purpose |
|---|---|---|
| `checkpoint_path` | `google/timesfm-3.0-pytorch` | Hub ID or local path |
| `per_core_batch_size` | `4` | Inference batch size |
| `input_patch_length` | `32` | Input patch size |
| `output_patch_length` | `64` | Output patch size |
| `quantiles` | `0.1` to `0.9` | Requested quantiles |
| `device` | Auto | PyTorch device |
| `cache_dir` | `None` | Hugging Face cache override |
| `revision` | `None` | Hub revision |
| `local_files_only` | `False` | Disable network retrieval |

Advanced architecture fields should normally come from the checkpoint rather
than application overrides.

## Advanced exported types

| Type | Purpose | Intended audience |
|---|---|---|
| `TimesFM3Torch` | Low-level PyTorch model with `forward` and autoregressive `decode` methods | Model developers |
| `ResidualBlockConfig` | Immutable residual-block dimensions, activation, normalization, bias, and dropout settings | Architecture work |
| `TransformerConfig` | Immutable per-transformer dimensions, attention, masking, rotary-position, and execution settings | Architecture work |
| `StackedTransformersConfig` | Transformer layer count, shared transformer configuration, and rematerialization setting | Architecture work |
| `_ModelConfig` | Original internal name behind the public `ModelConfig` alias | Compatibility only |

Application code should prefer `ModelConfig`, `TimesFM3Forecaster`, and
`TimesFM3Evaluator`. Constructing `TimesFM3Torch` or overriding checkpoint
architecture settings requires compatible dimensions and weights; mismatches
raise model-construction or state-dictionary errors.
