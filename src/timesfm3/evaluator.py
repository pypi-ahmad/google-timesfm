# Copyright 2026 Google LLC
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

"""Evaluator subclass extending TimesFM3Forecaster for benchmark evaluation.

Overrides `predict_batch` to add two behaviors on top of
`timesfm3_forecaster.py:TimesFM3Forecaster`: benchmark-standard defaults,
and automatic chunking of high-dimensional multivariate inputs across
multiple forward passes (since one forward pass is limited to
`_MAX_VARIATES_PER_FORWARD` total variates -- targets plus covariates --
by the model; see `model.py`). See `../timesfm3/uncertainty.py` for
coverage/calibration helpers commonly used downstream of this evaluator's
output.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np

from .timesfm3_forecaster import (
  ForecastOutput,
  TimesFM3Forecaster,
)

_MAX_VARIATES_PER_FORWARD = 32


class TimesFM3Evaluator(TimesFM3Forecaster):
  """Evaluator subclass extending TimesFM3Forecaster for benchmark evaluation.

  Specializes inference specifically for zero-shot benchmark evaluation:
  - Sets benchmark evaluation defaults (return_quantiles=True, use_symmetric_averaging=True,
    make_positive=True, sort_quantiles=True).
  - Enforces padding_mode="none" and use_znorm=False.
  - Implements automatic variate chunking (<= _MAX_VARIATES_PER_FORWARD variates per forward pass) 
    for high-dimensional multivariate inputs.
  - Supports univariate mode (univariate=True) unrolling channels into independent series.
  - Delegates batch forecasting, cross-variate attention, quantile sorting, symmetric
    averaging, and non-negativity clamping directly to TimesFM3Forecaster.
  """

  def predict_batch(
    self,
    contexts: list[np.ndarray],
    horizon: int,
    past_only_covariates: list[np.ndarray | None] | None = None,
    past_future_covariates: list[np.ndarray | None] | None = None,
    ts_ids: list[str] | None = None,
    return_quantiles: bool = True,
    use_symmetric_averaging: bool = True,
    make_positive: bool = True,
    sort_quantiles: bool = True,
    use_znorm: bool = False,
    padding_mode: str = "none",
    univariate: bool = False,
  ) -> Iterator[ForecastOutput]:
    """Runs inference on a batch of time series with official benchmark defaults & chunking."""
    # ts_ids is validated for length by the superclass; num_original_ts is
    # derived here (before delegating) since the univariate/chunking paths
    # below need it to reconstruct per-input-series outputs.
    num_original_ts = len(contexts)
    original_ts_ids = (
      list(ts_ids) if ts_ids is not None else [None] * num_original_ts
    )

    if not contexts:
      return

    if univariate:
      # Unroll each (possibly multivariate) input into independent
      # single-variate series, forecast them all in one flat batch (with
      # no cross-variate attention), then regroup by original series
      # below. Covariates are intentionally dropped for this path (passed
      # as None to super().predict_batch) since they're inherently
      # cross-variate/multivariate constructs.
      flat_contexts = []
      variate_counts = []
      was_1d_list = []

      for ctx in contexts:
        ctx_arr = np.asarray(ctx)
        was_1d = ctx_arr.ndim == 1
        was_1d_list.append(was_1d)

        ctx_2d = np.atleast_2d(ctx_arr)
        num_vars = ctx_2d.shape[0]
        variate_counts.append(num_vars)
        for v in range(num_vars):
          flat_contexts.append(ctx_2d[v, :])

      flat_outs = list(
        super().predict_batch(
          contexts=flat_contexts,
          horizon=horizon,
          past_only_covariates=None,
          past_future_covariates=None,
          return_quantiles=return_quantiles,
          use_symmetric_averaging=use_symmetric_averaging,
          make_positive=make_positive,
          sort_quantiles=sort_quantiles,
          use_znorm=use_znorm,
          padding_mode=padding_mode,
        )
      )

      curr_idx = 0
      for i in range(num_original_ts):
        n_vars = variate_counts[i]
        series_outs = flat_outs[curr_idx : curr_idx + n_vars]
        curr_idx += n_vars

        f_list = [out.forecast for out in series_outs if out.forecast is not None]
        q_list = [out.quantiles for out in series_outs if out.quantiles is not None]

        if was_1d_list[i]:
          combined_f = f_list[0] if f_list else None
          combined_q = q_list[0] if q_list else None
        else:
          combined_f = np.stack(f_list, axis=0) if f_list else None
          combined_q = np.stack(q_list, axis=0) if q_list else None

        yield ForecastOutput(
          ts_id=original_ts_ids[i],
          forecast=combined_f,
          quantiles=combined_q,
        )
      return

    contexts_2d = [np.atleast_2d(ctx) for ctx in contexts]
    po_2d = (
      [np.atleast_2d(cov) if cov is not None else None for cov in past_only_covariates]
      if past_only_covariates is not None
      else [None] * num_original_ts
    )
    pf_2d = (
      [np.atleast_2d(cov) if cov is not None else None for cov in past_future_covariates]
      if past_future_covariates is not None
      else [None] * num_original_ts
    )

    num_targets_in = contexts_2d[0].shape[0] if contexts_2d else 1
    num_pf = max((cov.shape[0] for cov in pf_2d if cov is not None), default=0)
    num_po = max((cov.shape[0] for cov in po_2d if cov is not None), default=0)
    total_variates = num_targets_in + num_pf + num_po

    if total_variates > _MAX_VARIATES_PER_FORWARD:
      # Fixed seed: subsampling choices are reproducible across repeated
      # evaluation runs on the same inputs, at the cost of always
      # dropping the "same" covariates when a given input recurs.
      rng = np.random.default_rng(42)
      # Step 1a: Subsample future covariates to at most 31 slots.
      # Reserve at least 1 slot (_MAX_VARIATES_PER_FORWARD - 1) for
      # targets even if covariates alone would otherwise fill the budget.
      max_pf = min(num_pf, _MAX_VARIATES_PER_FORWARD - 1)
      if num_pf > max_pf:
        pf_idx = np.sort(rng.choice(num_pf, max_pf, replace=False))
        pf_2d = [
          cov[pf_idx] if cov is not None else None
          for cov in pf_2d
        ]
        num_pf = max_pf
      # Step 1b: Subsample past-only covariates to at most (31 - pf) slots.
      max_po = min(num_po, _MAX_VARIATES_PER_FORWARD - 1 - num_pf)
      if num_po > max_po:
        po_idx = np.sort(rng.choice(num_po, max_po, replace=False))
        po_2d = [
          cov[po_idx] if cov is not None else None
          for cov in po_2d
        ]
        num_po = max_po
      # Step 2: Compute how many target slots fit per forward pass.
      targets_per_chunk = _MAX_VARIATES_PER_FORWARD - num_pf - num_po
      assert (
        targets_per_chunk >= 1
      ), f"Not enough variate slots for targets: pf={num_pf}, po={num_po}"
      num_chunks = math.ceil(num_targets_in / targets_per_chunk)

      # Step 3: Process each target chunk, concatenate along variate axis.
      chunk_results = []
      for c in range(num_chunks):
        v_start = c * targets_per_chunk
        v_end = min((c + 1) * targets_per_chunk, num_targets_in)
        actual_chunk_size = v_end - v_start
        chunk_inputs = [inp[v_start:v_end, :] for inp in contexts_2d]
        if actual_chunk_size < targets_per_chunk:
          # Last chunk may have fewer than targets_per_chunk real target
          # variates; pad the variate axis up to the fixed chunk size by
          # tiling the chunk's own target variates (wrapping around as
          # needed) rather than zeros, so the forward pass still sees
          # "real" data on every variate slot. The padded outputs are
          # trimmed back off below via `[:actual_chunk_size, ...]`.
          pad_needed = targets_per_chunk - actual_chunk_size
          padded_inputs = []
          for inp, c_inp in zip(contexts_2d, chunk_inputs):
            reps = math.ceil(pad_needed / inp.shape[0])
            repeated = np.tile(inp, (reps, 1))[:pad_needed, :]
            padded_inputs.append(np.concatenate([c_inp, repeated], axis=0))
          chunk_inputs = padded_inputs
        chunk_outs = list(
          super().predict_batch(
            contexts=chunk_inputs,
            horizon=horizon,
            past_only_covariates=po_2d,
            past_future_covariates=pf_2d,
            ts_ids=ts_ids,
            return_quantiles=return_quantiles,
            use_symmetric_averaging=use_symmetric_averaging,
            make_positive=make_positive,
            sort_quantiles=sort_quantiles,
            use_znorm=use_znorm,
            padding_mode=padding_mode,
          )
        )
        trimmed_outs = []
        for out in chunk_outs:
          f_2d = (
            out.forecast[:actual_chunk_size, ...]
            if out.forecast is not None
            else None
          )
          q_3d = (
            out.quantiles[:actual_chunk_size, ...]
            if out.quantiles is not None
            else None
          )
          trimmed_outs.append(
            ForecastOutput(
              ts_id=out.ts_id,
              forecast=f_2d,
              quantiles=q_3d,
            )
          )
        chunk_results.append(trimmed_outs)

      was_1d_input = len(contexts) > 0 and np.ndim(contexts[0]) == 1

      # Concatenate each original series' per-chunk (already-trimmed)
      # target-variate outputs back along the variate axis, in the same
      # v_start:v_end order the chunks were split in above.
      for i in range(num_original_ts):
        combined_f = (
          np.concatenate(
            [chunk_results[c][i].forecast for c in range(num_chunks)], axis=0
          )
          if chunk_results[0][i].forecast is not None
          else None
        )
        combined_q = (
          np.concatenate(
            [chunk_results[c][i].quantiles for c in range(num_chunks)], axis=0
          )
          if chunk_results[0][i].quantiles is not None
          else None
        )
        if was_1d_input and combined_f is not None:
          combined_f = combined_f[0]
        if was_1d_input and combined_q is not None:
          combined_q = combined_q[0]

        yield ForecastOutput(
          ts_id=original_ts_ids[i], forecast=combined_f, quantiles=combined_q
        )
      return

    yield from super().predict_batch(
      contexts=contexts,
      horizon=horizon,
      past_only_covariates=past_only_covariates,
      past_future_covariates=past_future_covariates,
      ts_ids=ts_ids,
      return_quantiles=return_quantiles,
      use_symmetric_averaging=use_symmetric_averaging,
      make_positive=make_positive,
      sort_quantiles=sort_quantiles,
      use_znorm=use_znorm,
      padding_mode=padding_mode,
    )
