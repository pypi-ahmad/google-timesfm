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

"""PyTorch utility functions for TimesFM layers.

Torch counterpart to `../flax/util.py`. Holds `DecodeCache` (consumed by
`transformer.py` for autoregressive decoding) and numeric helpers for the
model's per-instance normalization path (`update_running_stats`, `revin`).
"""

import dataclasses
import torch

_TOLERANCE = 1e-6


@dataclasses.dataclass(frozen=False)
class DecodeCache:
  """Cache for decoding.

  `key`/`value` are pre-allocated to the full cache length (not the number
  of tokens written so far); `next_index` is the write cursor advanced via
  slice assignment in transformer.py, and `num_masked` tracks how many
  leading positions were padding so rotary position ids stay aligned with
  real tokens across decode steps.
  """

  next_index: torch.Tensor
  num_masked: torch.Tensor
  key: torch.Tensor
  value: torch.Tensor


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
  """Updates the running stats.

  Merges prior running (n, mu, sigma) with the stats of a new chunk `x`
  using the parallel-variance (pooled variance) combination formula.
  `mask` marks positions to exclude (e.g. padding); wherever a count is
  zero, the corresponding `_safe` denominator is substituted with 1 to
  avoid a division by zero and the result is then forced back to 0 via
  `torch.where`, so it never affects the merged output.
  """
  is_legit = torch.logical_not(mask)
  inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

  inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
  inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
  inc_mu = inc_mu_numerator / inc_n_safe
  inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

  inc_var_numerator = torch.sum(
      ((x - inc_mu.unsqueeze(-1)) ** 2) * is_legit, dim=-1
  )
  inc_var = inc_var_numerator / inc_n_safe
  inc_var = torch.where(inc_n == 0, 0.0, inc_var)
  inc_sigma = torch.sqrt(inc_var)

  new_n = n + inc_n
  new_n_safe = torch.where(new_n == 0, 1.0, new_n)

  new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
  new_mu = torch.where(new_n == 0, 0.0, new_mu)

  term1 = n * sigma.pow(2)
  term2 = inc_n * inc_sigma.pow(2)
  term3 = n * (mu - new_mu).pow(2)
  term4 = inc_n * (inc_mu - new_mu).pow(2)

  new_var = (term1 + term2 + term3 + term4) / new_n_safe
  new_var = torch.where(new_n == 0, 0.0, new_var)
  # Clamp before sqrt: the pooled-variance formula can produce a tiny
  # negative value from floating-point cancellation even when the true
  # variance is ~0.
  new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

  return (w := (new_n, new_mu, new_sigma), w)


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
  """Reversible instance normalization.

  `mu`/`sigma` are broadcast onto `x` by inferring how many trailing axes
  `x` has beyond them (1 or 2); other rank differences are not handled.
  `reverse=False` normalizes; `reverse=True` denormalizes back to the
  original scale, so callers must reuse the same `mu`/`sigma` for both.
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
    # dividing by ~0.
    return (x - mu) / torch.where(sigma < _TOLERANCE, 1.0, sigma)
