# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Behavioral checks for leakage-free TimesFM-3 analyses."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import zipfile
from typing import Any

import numpy as np
import pandas as pd
import pytest

from timesfm3 import ForecastOutput, analysis, explorer
from timesfm3.timesfm3_forecaster import linear_interpolation


def dataset(frame: pd.DataFrame | None = None) -> explorer.UploadedDataset:
  if frame is None:
    frame = pd.DataFrame(
      {
        "date": pd.date_range("2026-01-01", periods=16),
        "a": np.arange(16, dtype=float),
        "b": np.arange(16, dtype=float) * 2,
        "past": np.arange(16, dtype=float) * 3,
        "known": np.arange(16, dtype=float) % 2,
      }
    )
  encoded = frame.to_csv(index=False).encode()
  return explorer.UploadedDataset(
    "example",
    frame,
    hashlib.sha256(encoded).hexdigest(),
    len(encoded),
    int(frame.memory_usage(deep=True).sum()),
  )


MAPPING = explorer.DatasetMapping("date", ("a", "b"), ("past",), ("known",))
SETTINGS = explorer.ForecastSettings(horizon=2, context_length=6)


class RecordingPredictor:
  """Forecast last observed value and retain model inputs for leakage checks."""

  device = "cpu"

  def __init__(self) -> None:
    self.calls: list[dict[str, Any]] = []

  def predict_batch(self, **kwargs: Any) -> list[ForecastOutput]:
    self.calls.append(kwargs)
    results = []
    for identity, context in zip(kwargs["ts_ids"], kwargs["contexts"], strict=True):
      point = np.repeat(np.atleast_2d(context)[:, -1:], kwargs["horizon"], axis=1)
      quantiles = None
      if kwargs["return_quantiles"]:
        quantiles = np.stack([point + offset for offset in np.linspace(-1, 1, 9)], -1)
      results.append(
        ForecastOutput(ts_id=identity, forecast=point, quantiles=quantiles)
      )
    return results


def execute(kind: analysis.AnalysisKind, *, windows: int = 2, **kwargs: Any):
  predictor = RecordingPredictor()
  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind=kind, windows=windows, **kwargs),
  )
  return predictor, prepared, analysis.run_analysis(predictor, prepared)


def test_rolling_metrics_use_forecast_pairs_and_preserve_overlap() -> None:
  _, prepared, result = execute("backtest", stride=1)
  assert prepared.prediction_rows == 8
  assert len(result.predictions) == 8
  assert result.predictions.origin.nunique() == 2
  assert result.predictions.timestamp.nunique() == 3
  overall = result.metrics.query("scope == 'overall'").set_index("target")
  assert overall.loc["a", "mae"] == pytest.approx(1.5)
  assert overall.loc["a", "rmse"] == pytest.approx(np.sqrt(2.5))
  assert overall.loc["b", "mae"] == pytest.approx(3)
  assert overall.loc["a", "mean_pinball_loss"] == pytest.approx(7 / 12)
  assert overall.loc["a", "q10_q90_coverage_percent"] == 50
  assert overall.loc["b", "q10_q90_coverage_percent"] == 0
  steps = result.metrics.query("scope == 'step' and target == 'a'").set_index("step")
  assert steps.loc[1, "mae"] == 1
  assert steps.loc[2, "mae"] == 2
  assert set(result.metrics.scope) == {"overall", "step", "window"}


def test_cutoff_context_excludes_heldout_targets_and_past_only_future() -> None:
  original = dataset()
  changed_frame = original.frame.copy()
  changed_frame.loc[14:, ["a", "b", "past"]] = 99999.0
  predictions = []
  calls = []
  for current in [original, dataset(changed_frame)]:
    predictor = RecordingPredictor()
    prepared = analysis.prepare_analysis(
      [current],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings(kind="backtest", windows=1),
    )
    predictions.append(analysis.run_analysis(predictor, prepared).predictions.point)
    calls.append(predictor.calls[0])
  pd.testing.assert_series_equal(*predictions)
  for name in ["contexts", "past_only_covariates", "past_future_covariates"]:
    for left, right in zip(calls[0][name], calls[1][name], strict=True):
      np.testing.assert_array_equal(left, right)
  assert calls[0]["contexts"][0][0, -1] == 13


def test_missing_labels_remain_unscored_without_moving_origin() -> None:
  current = dataset()
  current.frame.loc[14, "a"] = np.nan
  predictor = RecordingPredictor()
  prepared = analysis.prepare_analysis(
    [current],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="backtest", windows=1),
  )
  result = analysis.run_analysis(predictor, prepared)
  a = result.predictions.query("target == 'a'")
  assert a.actual.isna().tolist() == [True, False]
  assert predictor.calls[0]["contexts"][0][0, -1] == 13
  assert result.metrics.query("scope == 'overall' and target == 'a'").iloc[0].mae == 2


def test_anomaly_forces_one_step_quantiles_and_sorted_intervals() -> None:
  predictor = RecordingPredictor()
  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    dataclasses.replace(SETTINGS, return_quantiles=False, sort_quantiles=False),
    analysis.AnalysisSettings(kind="anomaly", windows=3),
  )
  result = analysis.run_analysis(predictor, prepared)
  assert len(result.predictions) == 6
  for call in predictor.calls:
    assert call["horizon"] == 1
    assert call["return_quantiles"] is True
    assert call["sort_quantiles"] is True


def test_anomaly_boundaries_missing_actuals_and_distances() -> None:
  frame = pd.DataFrame(
    {
      "actual": [-2.0, -1.0, 1.0, 3.0, np.nan],
      "point": [0.0] * 5,
      "q0.1": [-1.0] * 5,
      "q0.9": [1.0] * 5,
    }
  )
  result = analysis.anomaly_table(frame)
  assert result.direction.tolist() == ["below", "inside", "inside", "above", "unscored"]
  assert result.flagged.iloc[:4].tolist() == [True, False, False, True]
  assert pd.isna(result.flagged.iloc[4])
  np.testing.assert_allclose(result.distance.iloc[:4], [1, 0, 0, 2])
  np.testing.assert_allclose(result.residual.iloc[:4], [-2, -1, 1, 3])


def test_joint_comparison_excludes_covariates_and_shares_context() -> None:
  predictor, _, result = execute("joint_independent")
  assert set(result.predictions.variant) == {"joint", "independent"}
  assert {call["univariate"] for call in predictor.calls} == {False, True}
  by_mode = {}
  for call in predictor.calls:
    assert all(value is None for value in call["past_only_covariates"])
    assert all(value is None for value in call["past_future_covariates"])
    by_mode.setdefault(call["univariate"], []).extend(call["contexts"])
  for left, right in zip(by_mode[False], by_mode[True], strict=True):
    np.testing.assert_array_equal(left, right)
  joint = result.predictions.query("variant == 'joint'").reset_index(drop=True)
  independent = result.predictions.query("variant == 'independent'").reset_index(
    drop=True
  )
  pd.testing.assert_frame_equal(
    joint.drop(columns="variant"), independent.drop(columns="variant")
  )


def test_covariate_removal_deduplicates_variants() -> None:
  predictor, _, result = execute(
    "covariates", windows=1, selected_covariates=("past", "known")
  )
  assert result.predictions.variant.nunique() == 4
  assert len(predictor.calls) == 4
  memberships = []
  for call in predictor.calls:
    assert not call["univariate"]
    memberships.append(
      (
        call["past_only_covariates"][0] is not None,
        call["past_future_covariates"][0] is not None,
      )
    )
  assert set(memberships) == {
    (True, True),
    (False, False),
    (True, False),
    (False, True),
  }


def test_scenario_edits_are_future_only_and_baseline_is_immutable() -> None:
  frame = dataset().frame.copy()
  frame.loc[14:, ["a", "b"]] = np.nan
  uploads = [dataset(frame)]
  baseline = analysis.scenario_template(uploads, MAPPING, SETTINGS)
  original = baseline.copy(deep=True)
  edited = baseline.copy(deep=True)
  edited.loc[0, "value"] = 123.0
  fingerprint = analysis.analysis_fingerprint(uploads, MAPPING, SETTINGS)
  scenario = analysis.scenario_from_edits("Promotion", baseline, edited, fingerprint)
  prepared = analysis.prepare_analysis(
    uploads,
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="scenario"),
    (scenario,),
  )
  predictor = RecordingPredictor()
  result = analysis.run_analysis(predictor, prepared)
  pd.testing.assert_frame_equal(baseline, original)
  assert set(result.predictions.variant) == {"baseline", "Promotion"}
  assert len(predictor.calls) == 2
  np.testing.assert_array_equal(
    predictor.calls[0]["contexts"][0], predictor.calls[1]["contexts"][0]
  )
  assert any(
    np.any(call["past_future_covariates"][0] == 123) for call in predictor.calls
  )
  assert not np.any(predictor.calls[0]["past_future_covariates"][0] == 123)
  invalid = edited.copy()
  invalid.loc[0, "row"] = 0
  with pytest.raises(explorer.ExplorerError):
    analysis.scenario_from_edits("Bad", baseline, invalid, fingerprint)
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      uploads,
      MAPPING,
      dataclasses.replace(SETTINGS, horizon=1),
      analysis.AnalysisSettings(kind="scenario"),
      (scenario,),
    )


def test_scenario_zero_baseline_has_no_percentage_delta() -> None:
  frame = pd.DataFrame(
    {
      "dataset": ["d", "d"],
      "target": ["a", "a"],
      "origin": [5, 5],
      "step": [1, 1],
      "timestamp": [5, 5],
      "variant": ["baseline", "change"],
      "point": [0.0, 4.0],
    }
  )
  result = analysis.scenario_deltas(frame)
  row = result.query("variant == 'change'").iloc[0]
  assert row.delta == 4
  assert pd.isna(row.percent_delta)


def test_preflight_rejects_excess_variates_even_with_chunking_enabled() -> None:
  # allow_benchmark_chunking raises the per-request variate limit for plain
  # forecasting, but analysis workflows issue one forecast call per window
  # and must still refuse an unreasonably wide dataset outright rather than
  # silently chunking it.
  frame = pd.DataFrame({f"x{i}": np.arange(10, dtype=float) for i in range(33)})
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      [dataset(frame)],
      explorer.DatasetMapping(None, tuple(frame.columns)),
      dataclasses.replace(SETTINGS, allow_benchmark_chunking=True),
      analysis.AnalysisSettings(kind="backtest", windows=1),
    )


def test_preflight_rejects_prediction_row_budget() -> None:
  frame = pd.DataFrame({f"x{i}": np.arange(1000, dtype=float) for i in range(32)})
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      [dataset(frame)],
      explorer.DatasetMapping(None, tuple(frame.columns)),
      dataclasses.replace(SETTINGS, horizon=100),
      analysis.AnalysisSettings(kind="backtest", windows=100, stride=1),
    )


def test_point_only_backtest_and_zip_export() -> None:
  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    dataclasses.replace(SETTINGS, return_quantiles=False),
    analysis.AnalysisSettings(kind="backtest", windows=1),
  )
  result = analysis.run_analysis(RecordingPredictor(), prepared)
  assert "q0.1" not in result.predictions
  assert not result.metrics.empty
  with zipfile.ZipFile(io.BytesIO(analysis.analysis_zip(result))) as archive:
    assert {"predictions.csv", "metrics.csv", "analysis.json"} <= set(
      archive.namelist()
    )


def test_model_failure_aborts_analysis() -> None:
  class FailingPredictor(RecordingPredictor):
    def predict_batch(self, **kwargs: Any) -> list[ForecastOutput]:
      if self.calls:
        raise RuntimeError("deliberate model failure")
      return super().predict_batch(**kwargs)

  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="backtest", windows=2),
  )
  with pytest.raises(
    (RuntimeError, explorer.ExplorerError), match="deliberate model failure"
  ):
    analysis.run_analysis(FailingPredictor(), prepared)


def test_explicit_cutoff_keeps_missing_final_label_in_holdout() -> None:
  current = dataset()
  current.frame.loc[15, ["a", "b"]] = np.nan
  batch = explorer.prepare_batch(
    [current], MAPPING, dataclasses.replace(SETTINGS, task="holdout"), cutoffs=[14]
  )
  assert batch.series[0].context[0, -1] == 13
  assert batch.series[0].actual is not None
  assert np.isnan(batch.series[0].actual[:, -1]).all()
  assert batch.series[0].future_time[-1] == current.frame.date.iloc[15]


def test_interpolation_never_uses_a_value_after_cutoff() -> None:
  current = dataset()
  current.frame.loc[13, "a"] = np.nan
  current.frame.loc[14, "a"] = 1_000_000
  batch = explorer.prepare_batch(
    [current], MAPPING, dataclasses.replace(SETTINGS, task="holdout"), cutoffs=[14]
  )
  assert np.isnan(batch.series[0].context[0, -1])
  assert linear_interpolation(batch.series[0].context)[0, -1] == 12
  assert batch.series[0].actual is not None
  assert batch.series[0].actual[0, 0] == 1_000_000


@pytest.mark.parametrize("kind", ["scenario", "covariates"])
def test_covariate_workflows_reject_missing_column_roles(
  kind: analysis.AnalysisKind,
) -> None:
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      [dataset()],
      explorer.DatasetMapping("date", ("a",)),
      SETTINGS,
      analysis.AnalysisSettings(kind=kind),
    )


def test_joint_comparison_requires_multiple_targets() -> None:
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      [dataset()],
      explorer.DatasetMapping("date", ("a",)),
      SETTINGS,
      analysis.AnalysisSettings(kind="joint_independent", windows=1),
    )


def test_incomplete_known_future_covariates_fail_before_execution() -> None:
  current = dataset()
  current.frame.loc[15, "known"] = np.nan
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      [current],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings(kind="backtest", windows=1),
    )


@pytest.mark.parametrize(
  "edit",
  [
    analysis.ScenarioEdit("unknown", 14, "known", 3),
    analysis.ScenarioEdit("example", 13, "known", 3),
    analysis.ScenarioEdit("example", 16, "known", 3),
    analysis.ScenarioEdit("example", 14, "past", 3),
    analysis.ScenarioEdit("example", 14, "known", float("inf")),
  ],
)
def test_direct_scenario_overrides_cannot_bypass_editor_validation(
  edit: analysis.ScenarioEdit,
) -> None:
  frame = dataset().frame.copy()
  frame.loc[14:, ["a", "b"]] = np.nan
  uploads = [dataset(frame)]
  fingerprint = analysis.analysis_fingerprint(uploads, MAPPING, SETTINGS)
  with pytest.raises(explorer.ExplorerError):
    analysis.prepare_analysis(
      uploads,
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings(kind="scenario"),
      (analysis.Scenario("Invalid", fingerprint, (edit,)),),
    )


def test_missing_requested_quantiles_aborts_instead_of_silent_point_only() -> None:
  class PointOnlyPredictor(RecordingPredictor):
    def predict_batch(self, **kwargs: Any) -> list[ForecastOutput]:
      return super().predict_batch(**{**kwargs, "return_quantiles": False})

  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="anomaly", windows=1),
  )
  with pytest.raises(explorer.ExplorerError, match="quantiles"):
    analysis.run_analysis(PointOnlyPredictor(), prepared)


def test_comparison_sign_and_unmatched_observation_counts() -> None:
  _, _, result = execute("joint_independent", windows=1)
  changed = result.metrics.copy()
  independent = changed.variant == "independent"
  changed.loc[independent, "mae"] += 2
  comparison = analysis.comparison_table(changed, "joint_independent")
  assert comparison.mae_difference.eq(2).all()
  changed.loc[independent, "observations"] += 1
  with pytest.raises(explorer.ExplorerError, match="identical scored observations"):
    analysis.comparison_table(changed, "joint_independent")


def test_all_missing_window_keeps_counts_and_nan_score() -> None:
  frame = pd.DataFrame(
    {
      "variant": ["baseline"] * 2,
      "dataset": ["d"] * 2,
      "target": ["a"] * 2,
      "origin": [4] * 2,
      "step": [1, 2],
      "actual": [np.nan, np.nan],
      "point": [2.0, 2.0],
    }
  )
  result = analysis.analysis_metrics(frame)
  assert result.observations.eq(0).all()
  assert result.mae.isna().all()
  overall = result.query("scope == 'overall'").iloc[0]
  assert overall.missing == 2


def test_progress_reports_completed_forecasts_only() -> None:
  prepared = analysis.prepare_analysis(
    [dataset()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="backtest", windows=2),
  )
  progress = []
  analysis.run_analysis(
    RecordingPredictor(), prepared, lambda done, total: progress.append((done, total))
  )
  assert progress == [(1, 2), (2, 2)]


def test_cropped_context_preserves_absolute_integer_forecast_axis() -> None:
  uploads = [dataset(pd.DataFrame({"a": np.arange(20, dtype=float)}))]
  prepared = analysis.prepare_analysis(
    uploads,
    explorer.DatasetMapping(None, ("a",)),
    SETTINGS,
    analysis.AnalysisSettings(kind="backtest", windows=2, stride=1),
  )
  predictor = RecordingPredictor()
  result = analysis.run_analysis(predictor, prepared)
  assert result.predictions.origin.tolist() == [17, 17, 18, 18]
  assert result.predictions.timestamp.tolist() == [17, 18, 18, 19]
  np.testing.assert_array_equal(predictor.calls[0]["contexts"][0], [np.arange(11, 17)])
  np.testing.assert_array_equal(predictor.calls[1]["contexts"][0], [np.arange(12, 18)])


def test_different_dataset_lengths_share_cutoffs_between_variants() -> None:
  # "short" has 12 rows and "long" has 16; cutoffs are chosen relative to
  # each dataset's own end (not a shared absolute row index), so origins
  # differ per dataset while joint/independent variants still line up
  # against the same origins within each dataset.
  short = dataclasses.replace(
    dataset(dataset().frame.iloc[:12].copy()), dataset_id="short"
  )
  long = dataclasses.replace(dataset(), dataset_id="long")
  prepared = analysis.prepare_analysis(
    [short, long],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings(kind="joint_independent", windows=2, stride=1),
  )
  result = analysis.run_analysis(RecordingPredictor(), prepared)
  assert len(result.predictions) == 32
  expected_origins = {"short": [9, 10], "long": [13, 14]}
  for identifier, origins in expected_origins.items():
    current = result.predictions.loc[result.predictions.dataset == identifier]
    for variant in ["joint", "independent"]:
      assert (
        current.loc[current.variant == variant, "origin"].unique().tolist() == origins
      )
    joint = (
      current.loc[current.variant == "joint"]
      .drop(columns="variant")
      .reset_index(drop=True)
    )
    independent = (
      current.loc[current.variant == "independent"]
      .drop(columns="variant")
      .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(joint, independent)


def test_scenario_blank_future_targets_keep_absolute_edit_rows_after_crop() -> None:
  # Scenario edits are recorded against absolute row positions in the
  # uploaded frame. When the context fed to the model is later cropped
  # (rows before the forecast origin dropped), the edit at row 0 of the
  # template must still land on the correct absolute row/timestamp.
  frame = pd.DataFrame(
    {"a": np.arange(20, dtype=float), "known": np.arange(20, dtype=float)}
  )
  frame.loc[18:, "a"] = np.nan
  uploads = [dataset(frame)]
  mapping = explorer.DatasetMapping(None, ("a",), past_future=("known",))
  baseline = analysis.scenario_template(uploads, mapping, SETTINGS)
  assert baseline.row.tolist() == [18, 19]
  assert baseline.timestamp.tolist() == [18, 19]
  edited = baseline.copy()
  edited.loc[0, "value"] = 777
  scenario = analysis.scenario_from_edits(
    "Change",
    baseline,
    edited,
    analysis.analysis_fingerprint(uploads, mapping, SETTINGS),
  )
  prepared = analysis.prepare_analysis(
    uploads,
    mapping,
    SETTINGS,
    analysis.AnalysisSettings(kind="scenario"),
    (scenario,),
  )
  predictor = RecordingPredictor()
  result = analysis.run_analysis(predictor, prepared)
  assert result.predictions.origin.tolist() == [18, 18, 18, 18]
  assert result.predictions.timestamp.tolist() == [18, 19, 18, 19]
  assert "actual" not in result.predictions
  np.testing.assert_array_equal(predictor.calls[1]["contexts"][0], [np.arange(12, 18)])
  np.testing.assert_array_equal(
    predictor.calls[1]["past_future_covariates"][0], [[12, 13, 14, 15, 16, 17, 777, 19]]
  )
  np.testing.assert_array_equal(
    predictor.calls[0]["past_future_covariates"][0], [np.arange(12, 20)]
  )
  assert frame.loc[18, "known"] == 18
