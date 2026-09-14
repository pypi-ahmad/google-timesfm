# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TimesFM 2.5 (200M) model: Flax/JAX module, decode loop, and forecast API.

JAX/pmap counterpart to `timesfm_2p5_torch.py`; kept numerically aligned
with it (same patch/quantile/flip-invariance semantics, see comments
there for the underlying math). The main structural differences here are
JAX-specific: multi-device batching via `pmap` (axis `t` = device count,
merged with the per-core batch axis `b` as `(tb)` before/after each device
call), `donate_argnums` to let XLA reuse input buffers as output buffers,
and `nnx.scan`/`nnx.vmap` to apply/construct the `x` stacked transformer
layers without a Python loop. `TimesFM_2p5_200M_flax_module` is the raw
model; `TimesFM_2p5_200M_flax` wraps it with checkpoint loading (Orbax /
HF Hub) and the forecast-config flags from `../configs.py`.
"""

import dataclasses
import functools
import gc
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict

import einshape
from flax import nnx
import huggingface_hub
import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import orbax.checkpoint as ocp

from .. import configs
from ..flax import dense, transformer, util
from . import timesfm_2p5_base

jax_einshape = einshape.jax_einshape
scan = util.scan_along_axis
revin = util.revin

Float = jaxtyping.Float
Bool = jaxtyping.Bool
Array = jaxtyping.Array


def try_gc():
  """Best-effort GC when device memory usage crosses 75%.

  `memory_stats()` returns None on backends that don't report it (e.g.
  some CPU/TPU configurations), in which case this silently no-ops rather
  than raising -- device memory pressure just goes unmonitored there.
  """
  for d in jax.local_devices():
    stats = d.memory_stats()
    if stats is None:
      return
    if stats["bytes_in_use"] / stats["bytes_limit"] > 0.75:
      gc.collect()
      break


@nnx.vmap(in_axes=(None, 0), out_axes=0)
def _create_stacked_transformers(
  config: configs.StackedTransformersConfig, key: jax.Array
):
  """Builds all `num_layers` Transformer layers at once via vmap.

  `key` is a batch of per-layer PRNG keys (see the caller's
  `jax.random.split`); vmapping the constructor gives each layer an
  independent init key while producing one stacked pytree of parameters
  (consumed by `_apply_stacked_transformers`'s `nnx.scan`) instead of a
  Python list of `num_layers` separate modules.
  """
  return transformer.Transformer(config.transformer, rngs=nnx.Rngs(key))


def _scan_along_axis(f, init, xs, axis: int, **kwargs):
  """Scans along an axis."""
  moved_xs = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, axis, 0), xs)
  carry, moved_ys = jax.lax.scan(f, init, moved_xs, **kwargs)
  return (
    carry,
    jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, axis), moved_ys),
  )


@nnx.scan(in_axes=(0, nnx.Carry, None, 0), out_axes=(nnx.Carry, 0))
def _apply_stacked_transformers(
  model: transformer.Transformer,
  x: Float[Array, "b n d"],
  m: Float[Array, "b n"],
  decode_cache: util.DecodeCache | None = None,
) -> Float[Array, "b n d"]:
  """Applies the stacked layers sequentially via `nnx.scan`.

  `model` carries a leading layer axis (0) that's scanned over; `x` (the
  embeddings) is the carry threaded through every layer in order, `m`
  (the patch mask) is broadcast unchanged to every layer (`in_axes=None`),
  and `decode_cache`'s leading axis is per-layer, so each layer reads/
  writes its own cache slot.
  """
  return model(x, m, decode_cache=decode_cache)


class TimesFM_2p5_200M_flax_module(nnx.Module):  # pylint: disable=invalid-name
  """TimesFM 2.5 with 200M parameters."""

  config = timesfm_2p5_base.TimesFM_2p5_200M_Definition()
  decode_index: int = 5
  compiled_decode: Callable[..., Any] | None = None
  backend: str = ""
  context: int = 0
  horizon: int = 0
  per_core_batch_size: int = 0

  def __init__(self):
    super().__init__()
    self.backend = jax.devices()[0].platform
    self.num_devices = len(jax.devices(self.backend))

    # Names constants.
    self.p = self.config.input_patch_len  # 32
    self.o = self.config.output_patch_len  # 128
    self.os = self.config.output_quantile_len  # 1024
    self.m = self.o // self.p  # 4
    self.x = self.config.stacked_transformers.num_layers  # 20
    self.h = self.config.stacked_transformers.transformer.num_heads  # 16
    self.md = self.config.stacked_transformers.transformer.model_dims  # 1280
    self.hd = self.md // self.h  # 80
    self.q = len(self.config.quantiles) + 1  # 10
    self.aridx = self.config.decode_index  # 5

    # Layers.
    self.tokenizer = dense.ResidualBlock(self.config.tokenizer)
    self.stacked_xf = _create_stacked_transformers(
      self.config.stacked_transformers,
      jax.random.split(jax.random.key(42), self.x),
    )
    self.output_projection_point = dense.ResidualBlock(
      self.config.output_projection_point
    )
    self.output_projection_quantiles = dense.ResidualBlock(
      self.config.output_projection_quantiles
    )

  def __call__(
    self,
    inputs: Float[Array, "b n p"],
    masks: Bool[Array, "b n p"],
    decode_cache: util.DecodeCache | None = None,
  ):
    # Concatenate values with the (float-cast) mask as extra channels, so
    # the tokenizer's input_dims (64 = 2 * input_patch_len) sees both the
    # patch values and which of them are padding.
    tokenizer_inputs = jnp.concatenate([inputs, masks.astype(inputs.dtype)], axis=-1)
    input_embeddings = self.tokenizer(tokenizer_inputs)
    if decode_cache is None:
      decode_cache = [None] * self.x
    # Reduce the per-timestep mask to one bit per patch using only the
    # last (most recent) position: padding is a contiguous left prefix,
    # so a patch is only fully-padding if even its last position is
    # masked; a part-real, part-padding patch is treated as attendable.
    output_embeddings, decode_cache = _apply_stacked_transformers(
      self.stacked_xf, input_embeddings, masks[..., -1], decode_cache
    )
    output_ts = self.output_projection_point(output_embeddings)
    output_quantile_spread = self.output_projection_quantiles(output_embeddings)
    return (
      input_embeddings,
      output_embeddings,
      output_ts,
      output_quantile_spread,
    ), decode_cache

  @nnx.jit(static_argnames=("horizon",))
  def decode(self, horizon: int, inputs, masks):
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // self.o
    num_input_patches = context // self.p
    # Preallocate the KV cache for every patch that will ever be written:
    # the prefill (context) patches plus every patch generated during
    # autoregressive decoding (each of num_decode_steps steps advances the
    # cache by self.m patches).
    decode_cache_size = num_input_patches + num_decode_steps * self.m

    # Prefill
    patched_inputs = jax_einshape("b(np)->bnp", inputs, b=batch_size, p=self.p)
    patched_masks = jax_einshape("b(np)->bnp", masks, b=batch_size, p=self.p)
    # Compute mean/std causally, patch by patch (via `scan` over axis=1):
    # each patch is normalized using stats accumulated from itself and
    # every earlier patch only, never from later (future) patches, so no
    # future information leaks into a patch's own normalization.
    (last_n, last_mu, last_sigma), (_, context_mu, context_sigma) = scan(
      lambda carry, xs: util.update_running_stats(*carry, *xs),
      init=(zero := jnp.zeros(shape=(batch_size)), zero, zero),
      xs=(patched_inputs, patched_masks),
      axis=1,
    )
    decode_cache = util.DecodeCache(
      next_index=jnp.zeros(shape=(self.x, batch_size), dtype=jnp.int32),
      num_masked=jnp.zeros(shape=(self.x, batch_size), dtype=jnp.int32),
      key=jnp.zeros(shape=(self.x, batch_size, decode_cache_size, self.h, self.hd)),
      value=jnp.zeros(shape=(self.x, batch_size, decode_cache_size, self.h, self.hd)),
    )
    normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = jnp.where(patched_masks, 0.0, normed_inputs)
    (_, _, normed_outputs, normed_quantile_spread), decode_cache = self(
      normed_inputs, patched_masks, decode_cache
    )
    renormed_outputs = jax_einshape(
      "bn(oq)->bnoq",
      revin(normed_outputs, context_mu, context_sigma, reverse=True),
      o=self.o,
      q=self.q,
    )
    renormed_quantile_spread = jax_einshape(
      "bn(oq)->bnoq",
      revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
      o=self.os,
      q=self.q,
    )[:, -1, ...]

    # Autogressive decode
    @nnx.scan(in_axes=(None, nnx.Carry, 0), out_axes=(nnx.Carry, 1))
    def _ar_decode(module, carry, unused_iter):
      """One autoregressive decode step, looped `num_decode_steps` times via `nnx.scan`.

      `unused_iter` only exists to give `nnx.scan` a length to iterate
      over (its value is never read); all step state lives in `carry`.
      """
      last_renormed_output, (last_n, last_mu, last_sigma), decode_cache = carry
      new_patched_input = jax_einshape(
        "b(mp)->bmp", last_renormed_output, m=module.m, p=module.p
      )
      new_mask = jnp.zeros_like(new_patched_input, dtype=jnp.bool)
      carry_stats, (_, new_mu, new_sigma) = scan(
        lambda carry, xs: util.update_running_stats(*carry, *xs),
        init=(last_n, last_mu, last_sigma),
        xs=(new_patched_input, new_mask),
        axis=1,
      )
      new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
      (_, _, new_normed_output, _), decode_cache = module(
        new_normed_input, new_mask, decode_cache
      )
      # new_renormed_output holds self.m per-patch forecasts (one per
      # newly fed-in patch); keep only the last one -- the forecast that
      # follows the most recently generated patch -- as this step's
      # output. The einshape slice `[..., -1, :, :]` does that selection.
      new_renormed_output = jax_einshape(
        "bm(oq)->bmoq",
        revin(new_normed_output, new_mu, new_sigma, reverse=True),
        o=module.o,
        q=module.q,
      )[..., -1, :, :]

      return (
        (
          # module.decode_index (5) selects which of the q=10 output
          # channels (1 point channel + 9 quantiles [0.1..0.9]) is fed
          # back in as the next input patch -- index 5 is the median.
          new_renormed_output[..., module.decode_index],
          carry_stats,
          decode_cache,
        ),
        new_renormed_output,
      )

    if num_decode_steps > 0:
      _, ar_renormed_outputs = _ar_decode(
        self,
        (
          renormed_outputs[..., -1, :, self.decode_index],
          (last_n, last_mu, last_sigma),
          decode_cache,
        ),
        jnp.arange(num_decode_steps),
      )
    else:
      ar_renormed_outputs = None

    return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs

  def compile(
    self,
    context: int,
    horizon: int,
    per_core_batch_size: int = 1,
  ):
    if context % self.p != 0:
      logging.info(
        "When compiling, context needs to be multiple of the patch size %d."
        " Modifying context to %d.",
        self.p,
        context := math.ceil(context / self.p) * self.p,
      )
    if horizon % self.o != 0:
      logging.info(
        "When compiling, horizon needs to be multiple of the output patch"
        " size %d. Modifying horizon to %d.",
        self.o,
        horizon := math.ceil(horizon / self.o) * self.o,
      )

    self.context = context
    self.horizon = horizon
    self.per_core_batch_size = per_core_batch_size

    @nnx.pmap(
      # `model` and `horizon` are broadcast to every device (in_axes=None,
      # and horizon is additionally static since it controls Python-level
      # control flow / trace shape in `decode`); `inputs`/`masks` are
      # sharded across the leading (device) axis.
      in_axes=(None, None, 0, 0),
      out_axes=(0, 0, 0),
      devices=jax.devices(self.backend),
      axis_size=self.num_devices,
      static_broadcasted_argnums=(1,),
      axis_name="global_batch",
    )
    def compiled_decode_kernel(model, horizon, inputs, masks):
      return model.decode(horizon, inputs, masks)

    self.compiled_decode = functools.partial(compiled_decode_kernel, self)


def _flip_quantile_fn(x):
  # Leave channel 0 alone; reverse the ordering of the remaining 9
  # quantile channels, because negating a value that was the q-th
  # quantile of the flipped series makes it the (1-q)-th quantile of the
  # original series, so channel order must flip to stay sorted low-to-high.
  return jnp.concatenate([x[..., :1], jnp.flip(x[..., 1:], axis=-1)], axis=-1)


@functools.partial(
  jax.jit,
  donate_argnums=(0, 1, 2),
)
def _force_flip_invariance_fn(
  flipped_pf_outputs,
  flipped_quantile_spreads,
  flipped_ar_outputs,
):
  """Forces flip invariance.

  Reshapes and quantile-flips the results of decoding on -inputs so they
  can be averaged with the non-flipped results in `_after_model_decode`
  to extend TimesFM's a>=0 flip-invariance guarantee to a<0 (see
  `ForecastConfig.force_flip_invariance` in ../configs.py). `donate_argnums`
  tells XLA these input buffers may be reused/overwritten for the outputs
  since callers don't need the pre-flip values afterward.
  """
  flipped_pf_outputs = _flip_quantile_fn(flipped_pf_outputs)
  # "tb...->(tb)..." folds the pmap device axis (t) back into the batch
  # axis (b), undoing the per-device split applied before model.decode.
  flipped_pf_outputs = jax_einshape("tb...->(tb)...", flipped_pf_outputs)
  flipped_quantile_spreads = _flip_quantile_fn(flipped_quantile_spreads)
  flipped_quantile_spreads = jax_einshape("tb...->(tb)...", flipped_quantile_spreads)
  to_concat = [flipped_pf_outputs[:, -1, ...]]
  if flipped_ar_outputs is not None:
    flipped_ar_outputs = _flip_quantile_fn(flipped_ar_outputs)
    flipped_ar_outputs = jax_einshape("tbno...->(tb)(no)...", flipped_ar_outputs)
    to_concat.append(flipped_ar_outputs)
  flipped_full_forecast = jnp.concatenate(to_concat, axis=1)

  return flipped_quantile_spreads, flipped_pf_outputs, flipped_full_forecast


@functools.partial(
  jax.jit,
  static_argnames=("max_horizon",),
  donate_argnums=(0,),
)
def _use_continuous_quantile_head_fn(full_forecast, quantile_spreads, max_horizon):
  """Uses continuous quantile head.

  Replaces the (coarser, per-patch) quantile channels with the
  continuous-quantile-head's spread re-centered onto this run's own
  median (channel 5): quantile_spreads' own median is subtracted off and
  full_forecast's median added back, so only the *shape* of the
  continuous head's quantile spread is used, not its absolute level.
  Channels 0 and 5 pass through unchanged.
  """
  to_stack = [full_forecast[..., :max_horizon, 0]]
  for quantile_index in [1, 2, 3, 4]:
    to_stack.append(
      quantile_spreads[:, :max_horizon, quantile_index]
      - quantile_spreads[:, :max_horizon, 5]
      + full_forecast[:, :max_horizon, 5]
    )
  to_stack.append(full_forecast[..., :max_horizon, 5])
  for quantile_index in [6, 7, 8, 9]:
    to_stack.append(
      quantile_spreads[:, :max_horizon, quantile_index]
      - quantile_spreads[:, :max_horizon, 5]
      + full_forecast[:, :max_horizon, 5]
    )
  return jnp.stack(to_stack, axis=-1)


@functools.partial(jax.jit, donate_argnums=(0,))
def _fix_quantile_crossing_fn(full_forecast):
  """Fixes quantile crossing.

  Enforces monotonically non-decreasing quantiles (channel 1 = 0.1 ...
  channel 9 = 0.9) by clamping outward from the median (channel 5) via
  a `reverse` scan for the lower half and a forward scan for the upper
  half, so each channel's clamp only depends on its already-fixed
  neighbor closer to the median.
  """
  lower_quantiles = _scan_along_axis(
    lambda carry, x: (w := jnp.minimum(carry, x), w),
    init=full_forecast[..., 5],
    xs=full_forecast[..., 1:5],
    axis=-1,
    reverse=True,
  )[1]
  upper_quantiles = _scan_along_axis(
    lambda carry, x: (w := jnp.maximum(carry, x), w),
    init=full_forecast[..., 5],
    xs=full_forecast[..., 6:10],
    axis=-1,
    reverse=False,
  )[1]
  return jnp.concatenate(
    [
      full_forecast[..., :1],
      lower_quantiles,
      full_forecast[..., 5:6],
      upper_quantiles,
    ],
    axis=-1,
  )


@functools.partial(jax.jit, static_argnames=("fc",), donate_argnums=(1, 2))
def _before_model_decode(fc, inputs, masks):
  """All Jax steps before model decode call.

  `donate_argnums=(1, 2)` (inputs, masks) lets XLA reuse those buffers,
  since callers pass freshly-converted arrays that aren't needed
  afterward in their pre-split shape.
  """
  if fc.infer_is_positive:
    is_positive = jnp.all(inputs >= 0, axis=-1, keepdims=True)
  else:
    is_positive = None

  if fc.normalize_inputs:
    mu = jnp.mean(inputs, axis=-1, keepdims=True)
    sigma = jnp.std(inputs, axis=-1, keepdims=True)
    inputs = revin(inputs, mu, sigma, reverse=False)
  else:
    mu, sigma = None, None

  # "(tb)...->tb..." splits the flat batch axis into (device, per-core
  # batch) axes ahead of the pmap'd `model.compiled_decode` call.
  inputs = jax_einshape("(tb)...->tb...", inputs, b=fc.per_core_batch_size)
  masks = jax_einshape("(tb)...->tb...", masks, b=fc.per_core_batch_size)

  return inputs, masks, is_positive, mu, sigma


@functools.partial(
  jax.jit,
  static_argnames=(
    "fc",
    "p",
  ),
  donate_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9),
)
def _after_model_decode(
  fc,
  pf_outputs,
  quantile_spreads,
  ar_outputs,
  flipped_pf_outputs,
  flipped_quantile_spreads,
  flipped_ar_outputs,
  is_positive,
  mu,
  sigma,
  p,
):
  """All Jax steps after model decode call.

  `donate_argnums=(1..9)` lets XLA reuse the pmap outputs' buffers for
  this function's outputs, since the caller only needs `full_forecast`
  afterward, not any of these intermediates in their original layout.
  """
  # t: num_devices, b: per_core_batch_size
  pf_outputs = jax_einshape("tb...->(tb)...", pf_outputs)
  quantile_spreads = jax_einshape("tb...->(tb)...", quantile_spreads)
  to_concat = [pf_outputs[:, -1, ...]]
  if ar_outputs is not None:
    ar_outputs = jax_einshape("tbno...->(tb)(no)...", ar_outputs)
    to_concat.append(ar_outputs)
  full_forecast = jnp.concatenate(to_concat, axis=1)

  if fc.force_flip_invariance:
    (
      flipped_quantile_spreads,
      flipped_pf_outputs,
      flipped_full_forecast,
    ) = _force_flip_invariance_fn(
      flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs
    )
    quantile_spreads = (quantile_spreads - flipped_quantile_spreads) / 2
    pf_outputs = (pf_outputs - flipped_pf_outputs) / 2
    full_forecast = (full_forecast - flipped_full_forecast) / 2

  if fc.use_continuous_quantile_head:
    full_forecast = _use_continuous_quantile_head_fn(
      full_forecast, quantile_spreads, fc.max_horizon
    )

  if fc.return_backcast:
    # In-sample "backcast": for every context patch except the last, take
    # only its first `p` (input_patch_len) predicted points -- the part
    # of that patch's prediction window overlapping the next patch's
    # actual start -- and prepend to the real forecast. Consumed by
    # forecast_with_covariates in timesfm_2p5_base.py.
    full_backcast = jax_einshape("...npq->...(np)q", pf_outputs[:, :-1, :p, :])
    full_forecast = jnp.concatenate([full_backcast, full_forecast], axis=1)

  if fc.fix_quantile_crossing:
    full_forecast = _fix_quantile_crossing_fn(full_forecast)

  if fc.normalize_inputs:
    full_forecast = revin(full_forecast, mu, sigma, reverse=True)

  if is_positive is not None:
    full_forecast = jnp.where(
      is_positive[..., None],
      jnp.maximum(full_forecast, jnp.zeros_like(full_forecast)),
      full_forecast,
    )

  return full_forecast


class TimesFM_2p5_200M_flax(timesfm_2p5_base.TimesFM_2p5):
  """Flax implementation of TimesFM 2.5 with 200M parameters."""

  # Class-level default is only a type/documentation placeholder -- every
  # instance immediately overwrites `self.model` with its own module in
  # __init__ below (each with freshly-initialized/untrained parameters
  # until load_checkpoint or from_pretrained is called).
  model: nnx.Module = TimesFM_2p5_200M_flax_module()

  def __init__(self, **kwargs):
    self.model = TimesFM_2p5_200M_flax_module()

  def load_checkpoint(self, path: str):
    """Loads a TimesFM model from a checkpoint."""
    if os.path.isdir(path):
      model_file_path = path
    else:
      model_file_path = os.path.dirname(path)

    checkpointer = ocp.StandardCheckpointer()
    graph, state = nnx.split(self.model)
    state = checkpointer.restore(model_file_path, state)
    self.model = nnx.merge(graph, state)

  @classmethod
  def from_pretrained(
    cls,
    model_id: str = "google/timesfm-2.5-200m-flax",
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
    proxies: Dict | None = None,
    resume_download: bool | None = None,
    local_files_only: bool | None = None,
    token: str | None = None,
    **model_kwargs,
  ):
    """Loads a Flax TimesFM model."""

    # Create an instance of the model wrapper class.
    instance = cls(**model_kwargs)

    # Determine the path to the model weights.
    model_file_path = ""
    if os.path.isdir(model_id):
      logging.info("Loading checkpoint from local directory: %s", model_id)
      model_file_path = model_id
    else:
      logging.info("Downloading checkpoint from Hugging Face repo %s", model_id)
      model_file_path = huggingface_hub.snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        force_download=force_download,
        proxies=proxies,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
      )
      logging.info("Loading checkpoint from: %s", model_file_path)

    instance.load_checkpoint(model_file_path)
    return instance

  def compile(
    self,
    forecast_config: configs.ForecastConfig,
    dryrun: bool = True,
    **kwargs
  ):
    # Acrobym used during validation.
    print("Compiling model...")

    fc = forecast_config
    if fc.max_context % self.model.p != 0:
      logging.info(
        "When compiling, max context needs to be multiple of the patch size"
        " %d. Using max context = %d instead.",
        self.model.p,
        new_context := math.ceil(fc.max_context / self.model.p) * self.model.p,
      )
      fc = dataclasses.replace(fc, max_context=new_context)
    if fc.max_horizon % self.model.o != 0:
      logging.info(
        "When compiling, max horizon needs to be multiple of the output patch"
        " size %d. Using max horizon = %d instead.",
        self.model.o,
        new_horizon := math.ceil(fc.max_horizon / self.model.o) * self.model.o,
      )
      fc = dataclasses.replace(fc, max_horizon=new_horizon)
    if fc.max_context + fc.max_horizon > self.model.config.context_limit:
      raise ValueError(
        "Context + horizon must be less than the context limit."
        f" {fc.max_context} + {fc.max_horizon} >"
        f" {self.model.config.context_limit}."
      )
    if fc.use_continuous_quantile_head and (fc.max_horizon > self.model.os):
      raise ValueError(
        f"Continuous quantile head is not supported for horizons > {self.model.os}."
      )

    self.forecast_config = fc
    self.model.compile(
      context=self.forecast_config.max_context,
      horizon=self.forecast_config.max_horizon,
      per_core_batch_size=fc.per_core_batch_size,
    )
    self.per_core_batch_size = self.forecast_config.per_core_batch_size
    self.num_devices = self.model.num_devices
    self.global_batch_size = (
      self.forecast_config.per_core_batch_size * self.model.num_devices
    )

    def compiled_decode_kernel(fc, horizon, inputs, masks):
      inputs = jnp.array(inputs, dtype=jnp.float32)
      masks = jnp.array(masks, dtype=jnp.bool)
      if horizon > fc.max_horizon:
        raise ValueError(
          f"Horizon must be less than the max horizon. {horizon} > {fc.max_horizon}."
        )
      to_trim = fc.max_horizon - horizon

      inputs, masks, is_positive, mu, sigma = _before_model_decode(fc, inputs, masks)

      pf_outputs, quantile_spreads, ar_outputs = self.model.compiled_decode(
        fc.max_horizon, inputs, masks
      )
      if fc.force_flip_invariance:
        # Second full decode pass on the negated input; combined with
        # `flipped_*` reversal/averaging in `_after_model_decode` this
        # extends the a>=0-only flip invariance to a<0 (see
        # ForecastConfig.force_flip_invariance in ../configs.py). This is
        # ~2x the decode cost when enabled.
        flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs = (
          self.model.compiled_decode(fc.max_horizon, -inputs, masks)
        )
      else:
        flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs = (
          None,
          None,
          None,
        )

      full_forecast = _after_model_decode(
        fc,
        pf_outputs,
        quantile_spreads,
        ar_outputs,
        flipped_pf_outputs,
        flipped_quantile_spreads,
        flipped_ar_outputs,
        is_positive,
        mu,
        sigma,
        self.model.p,
      )
      # Copy off-device to a plain numpy array, then explicitly drop the
      # JAX array reference and opportunistically GC before returning, so
      # device memory used by this call doesn't linger for the caller.
      full_forecast_np = np.array(full_forecast)
      del full_forecast
      try_gc()
      if to_trim > 0:
        full_forecast_np = full_forecast_np[..., :-to_trim, :]
      return full_forecast_np[..., 5], full_forecast_np

    self.compiled_decode = functools.partial(
      compiled_decode_kernel, self.forecast_config
    )

    if dryrun:
      # Run once on dummy all-zero data to force JAX/pmap tracing and
      # compilation to happen now (eagerly, during compile()) rather than
      # lazily on the first real forecast call.
      _ = self.compiled_decode(
        self.forecast_config.max_horizon,
        jnp.zeros(
          (self.global_batch_size, self.forecast_config.max_context), dtype=jnp.float32
        ),
        jnp.zeros(
          (self.global_batch_size, self.forecast_config.max_context), dtype=jnp.bool
        ),
      )
    print("Compiling done.")
