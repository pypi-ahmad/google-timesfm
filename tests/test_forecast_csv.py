# Copyright 2026 Ahmad Mujtaba
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

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = (
  Path(__file__).parents[1] / "timesfm-forecasting" / "scripts" / "forecast_csv.py"
)
SPEC = importlib.util.spec_from_file_location("forecast_csv", SCRIPT_PATH)
assert SPEC and SPEC.loader
forecast_csv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(forecast_csv)


class RecordingModel:
  def __init__(self) -> None:
    self.inputs: list[np.ndarray] = []

  def forecast(
    self, *, horizon: int, inputs: list[np.ndarray]
  ) -> tuple[np.ndarray, np.ndarray]:
    self.inputs = inputs
    point = np.arange(horizon, dtype=np.float32)[None, :]
    quantiles = np.stack(
      [point + quantile_index for quantile_index in range(10)], axis=-1
    )
    return point, quantiles


def test_forecast_trims_only_missing_edges_and_preserves_internal_gaps() -> None:
  model = RecordingModel()
  frame = pd.DataFrame({"value": [np.nan, np.inf, 1.0, np.nan, -np.inf, 4.0, np.nan]})

  forecast_csv.forecast_series(model, frame, ["value"], horizon=2)

  assert len(model.inputs) == 1
  np.testing.assert_equal(model.inputs[0], np.array([1, np.nan, np.nan, 4]))
  assert model.inputs[0].dtype == np.float32


@pytest.mark.parametrize(
  "values",
  ([np.nan, None], [np.inf, -np.inf], [1e100, -1e100]),
)
def test_forecast_rejects_series_without_finite_float32_values(values: list) -> None:
  with pytest.raises(ValueError, match="Series 'value' contains no finite values"):
    forecast_csv.forecast_series(
      RecordingModel(), pd.DataFrame({"value": values}), ["value"], horizon=2
    )


def test_forecast_exposes_explicit_quantiles_and_legacy_aliases() -> None:
  results = forecast_csv.forecast_series(
    RecordingModel(), pd.DataFrame({"value": [1.0, 2.0]}), ["value"], horizon=2
  )["value"]

  for explicit, legacy in (
    ("q10", "lower_90"),
    ("q20", "lower_80"),
    ("q50", "median"),
    ("q80", "upper_80"),
    ("q90", "upper_90"),
  ):
    assert results[explicit] == results[legacy]


def test_csv_output_contains_explicit_and_legacy_quantile_columns(
  tmp_path: Path,
) -> None:
  frame = pd.DataFrame({"value": [1.0, 2.0]})
  results = forecast_csv.forecast_series(RecordingModel(), frame, ["value"], horizon=2)
  output = tmp_path / "forecast.csv"

  forecast_csv.write_csv_output(results, str(output), frame, None, horizon=2)

  columns = pd.read_csv(output).columns.tolist()
  assert columns == [
    "series",
    "step",
    "forecast",
    "q10",
    "q20",
    "q50",
    "q80",
    "q90",
    "lower_90",
    "lower_80",
    "median",
    "upper_80",
    "upper_90",
  ]


def test_irregular_dates_fall_back_to_step_numbers(tmp_path: Path) -> None:
  frame = pd.DataFrame(
    {
      "date": pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-08"]),
      "value": [1.0, 2.0, 3.0],
    }
  )
  results = forecast_csv.forecast_series(RecordingModel(), frame, ["value"], horizon=2)
  output = tmp_path / "forecast.csv"

  forecast_csv.write_csv_output(results, str(output), frame, "date", horizon=2)

  written = pd.read_csv(output)
  assert "date" not in written.columns
  assert written["step"].tolist() == [1, 2]
