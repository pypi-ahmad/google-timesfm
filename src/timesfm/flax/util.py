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

"""Flax utility functions for TimesFM layers.

Holds the `DecodeCache` structure consumed by `transformer.py`'s attention
during autoregressive decoding, plus small numeric helpers (running-stat
merge, scan-over-axis, reversible instance normalization) used by the
model's input normalization path. See `../timesfm_2p5/timesfm_2p5_flax.py`
for where these are called end-to-end.
"""

import dataclasses
import functools
import jax
import jax.numpy as jnp
import jaxtyping

Float = jaxtyping.Float
Array = jaxtyping.Array
Bool = jaxtyping.Bool
Integer = jaxtyping.Integer

_TOLERANCE = 1e-6


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=False)
class DecodeCache:
  """Cache for decoding.

  `key`/`value` are pre-allocated to the full cache length `n` (not the
  number of tokens written so far); `next_index` is the per-batch write
  cursor into that buffer, advanced via `lax.dynamic_update_slice` in
  transformer.py. `num_masked` tracks how many leading positions in the
  original (unpadded) sequence were padding, so rotary position ids can
  stay aligned with real (non-padded) tokens across decode steps.
  """

  next_index: Integer[Array, "b"]
  num_masked: Integer[Array, "b"]
  key: Float[Array, "b n h d"]
  value: Float[Array, "b n h d"]


@jax.jit
def update_running_stats(
  n: Float[Array, "b"],
  mu: Float[Array, "b"],
  sigma: Float[Array, "b"],
  x: Float[Array, "b p"],
  mask: Bool[Array, "b p"],
) -> tuple[
  tuple[Float[Array, "b"], Float[Array, "b"], Float[Array, "b"]],
  tuple[Float[Array, "b"], Float[Array, "b"], Float[Array, "b"]],
]:
  """Updates the running stats.

  Merges the (n, mu, sigma) running mean/std computed over prior chunks
  with the stats of a new chunk `x`, using the parallel-variance
  combination formula (pooled variance across two partitions), and returns
  the merged stats twice (as `(carry, y)`) for use with `lax.scan`. `mask`
  marks positions to exclude (e.g. padding); a chunk contributing zero
  valid (unmasked) elements leaves the running stats unchanged.
  """
  is_legit = jnp.logical_not(mask)
  inc_n = jnp.sum(is_legit.astype(jnp.float32), axis=-1, keepdims=False)
  inc_mu = jnp.where(
    inc_n == 0, 0.0, jnp.mean(x, axis=-1, keepdims=False, where=is_legit)
  )
  inc_sigma = jnp.where(
    inc_n == 0, 0.0, jnp.std(x, axis=-1, keepdims=False, where=is_legit)
  )
  new_n = n + inc_n
  new_mu = jnp.where(new_n == 0, 0.0, (n * mu + inc_mu * inc_n) / new_n)
  new_sigma = jnp.sqrt(
    jnp.where(
      new_n == 0,
      0.0,
      (
        n * sigma * sigma
        + inc_n * inc_sigma * inc_sigma
        + n * (mu - new_mu) * (mu - new_mu)
        + inc_n * (inc_mu - new_mu) * (inc_mu - new_mu)
      )
      / new_n,
    )
  )
  return (w := (new_n, new_mu, new_sigma), w)


def scan_along_axis(f, init, xs, axis: int, **kwargs):
  """Scans along an axis."""
  moved_xs = jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, axis, 0), xs)
  carry, moved_ys = jax.lax.scan(f, init, moved_xs, **kwargs)
  return (
    carry,
    jax.tree_util.tree_map(lambda x: jnp.moveaxis(x, 0, axis), moved_ys),
  )


@functools.partial(jax.jit, static_argnames=("reverse",))
def revin(
  x: Float[Array, "b ..."],
  mu: Float[Array, "b ..."],
  sigma: Float[Array, "b ..."],
  reverse: bool = False,
):
  """Reversible per-instance normalization.

  `mu`/`sigma` are broadcast onto `x` by inferring how many trailing axes
  `x` has beyond them (1 or 2, e.g. a time axis and optionally a channel
  axis); other rank differences are not handled and will broadcast-fail.
  `reverse=False` normalizes (forward pass); `reverse=True` denormalizes
  outputs/quantiles back to the original scale, so callers must pass the
  same `mu`/`sigma` on both calls.
  """
  if len(mu.shape) == len(x.shape) - 1:
    mu = mu[..., None]
    sigma = sigma[..., None]
  elif len(mu.shape) == len(x.shape) - 2:
    mu = mu[..., None, None]
    sigma = sigma[..., None, None]
  if reverse:
    return x * sigma + mu
  else:
    # Guard against near-zero sigma (e.g. a constant input series) to avoid
    # dividing by ~0; treat it as sigma=1 instead of NaN/inf.
    return (x - mu) / jnp.where(sigma < _TOLERANCE, 1.0, sigma)
