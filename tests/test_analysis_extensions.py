"""Paired configuration comparisons, naive references, and interval summaries."""

import dataclasses
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from timesfm3 import analysis
from timesfm3.explorer import (
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  UploadedDataset,
)
from timesfm3.timesfm3_forecaster import ForecastOutput
from timesfm3.uncertainty import calibration_table, interval_bands


def uploaded(values: np.ndarray | None = None) -> UploadedDataset:
  values = np.arange(24, dtype=float) if values is None else values
  frame = pd.DataFrame(
    {"date": pd.date_range("2026-01-01", periods=len(values)), "value": values}
  )
  frame.attrs["preparation"] = {"calendar": "weekday"}
  return UploadedDataset(
    "series", frame, "hash", 100, int(frame.memory_usage(deep=True).sum())
  )


MAPPING = DatasetMapping("date", ("value",))
SETTINGS = ForecastSettings(context_length=6, horizon=3)


class Predictor:
  device = "cpu"

  def __init__(self) -> None:
    self.model_provenance = {
      "resolved_revision": "abc123",
      "selection": {"source": "D:/models/local"},
    }
    self.calls: list[dict[str, Any]] = []

  def predict_batch(self, **kwargs: Any) -> list[ForecastOutput]:
    self.calls.append(kwargs)
    results = []
    for identity, context in zip(kwargs["ts_ids"], kwargs["contexts"], strict=True):
      last = np.array([row[np.isfinite(row)][-1] for row in context])
      point = np.repeat(last[:, None], kwargs["horizon"], axis=1)
      quantiles = np.stack([point + offset for offset in np.arange(-4, 5)], axis=-1)
      results.append(
        ForecastOutput(ts_id=identity, forecast=point, quantiles=quantiles)
      )
    return results


def configurations() -> tuple[analysis.ExperimentConfiguration, ...]:
  return (
    analysis.ExperimentConfiguration("Long", SETTINGS),
    analysis.ExperimentConfiguration(
      "Short",
      dataclasses.replace(
        SETTINGS, context_length=3, use_znorm=True, return_quantiles=False
      ),
    ),
  )


def test_scenario_fingerprint_changes_when_calendar_definitions_change() -> None:
  original = uploaded()
  changed = dataclasses.replace(original, frame=original.frame.copy())
  original.frame.attrs["preparation"] = {"calendar": {"holiday_country": "US"}}
  changed.frame.attrs["preparation"] = {"calendar": {"holiday_country": "GB"}}
  assert original.sha256 == changed.sha256
  assert analysis.analysis_fingerprint(
    [original], MAPPING, SETTINGS
  ) != analysis.analysis_fingerprint([changed], MAPPING, SETTINGS)


def test_settings_share_origins_but_keep_context_lengths_and_provenance() -> None:
  prepared = analysis.prepare_analysis(
    [uploaded()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings("settings", windows=2, stride=2, seasonal_period=3),
    configurations=configurations(),
  )
  predictor = Predictor()
  result = analysis.run_analysis(predictor, prepared)
  naive_metrics = result.metrics.loc[
    result.metrics.variant.isin(["last_value", "seasonal_naive"])
  ]
  assert naive_metrics["q10_q90_coverage_percent"].isna().all()
  assert naive_metrics["mean_pinball_loss"].isna().all()
  assert len(predictor.calls) == 4
  assert [call["contexts"][0].shape[-1] for call in predictor.calls] == [6, 3, 6, 3]
  assert all(call["return_quantiles"] for call in predictor.calls)
  assert predictor.calls[1]["use_znorm"] is True
  assert prepared.prediction_rows == 24
  assert (
    result.predictions.groupby("variant").origin.unique().map(list).tolist()
    == [[19, 21]] * 4
  )
  assert result.predictions.scored.all()
  assert result.manifest["variants"][1]["settings"]["context_length"] == 3
  assert result.manifest["datasets"][0]["preparation"] == {"calendar": "weekday"}
  assert result.manifest["model_provenance"] == predictor.model_provenance
  assert result.manifest["checkpoint"] == "D:/models/local"
  assert result.manifest["schema_version"] == 2


@pytest.mark.parametrize(
  "configured", [configurations()[:1], configurations() * 5, configurations() * 2]
)
def test_configuration_count_and_names_fail_preflight(configured) -> None:
  with pytest.raises(ExplorerError):
    analysis.prepare_analysis(
      [uploaded()],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings("settings"),
      configurations=configured,
    )


def test_eight_configurations_include_baselines_in_the_row_budget(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  configured = tuple(
    analysis.ExperimentConfiguration(f"Setting {index}", SETTINGS) for index in range(8)
  )
  prepared = analysis.prepare_analysis(
    [uploaded()],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings("settings", windows=1),
    configurations=configured,
  )
  assert prepared.forecast_count == 10
  assert prepared.prediction_rows == 30
  monkeypatch.setattr(analysis, "MAX_PREDICTION_ROWS", 29)
  with pytest.raises(ExplorerError, match="prediction rows"):
    analysis.prepare_analysis(
      [uploaded()],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings("settings", windows=1),
      configurations=configured,
    )


@pytest.mark.parametrize(
  "changes", [{"horizon": 2}, {"context_length": 1}, {"use_znorm": "yes"}]
)
def test_configuration_controls_are_validated(changes) -> None:
  configured = (
    configurations()[0],
    analysis.ExperimentConfiguration(
      "Changed", dataclasses.replace(SETTINGS, **changes)
    ),
  )
  with pytest.raises(ExplorerError):
    analysis.prepare_analysis(
      [uploaded()],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings("settings", windows=1),
      configurations=configured,
    )


def test_comparisons_require_full_warmup_for_every_window() -> None:
  with pytest.raises(ExplorerError, match="history rows"):
    analysis.prepare_analysis(
      [uploaded()],
      MAPPING,
      SETTINGS,
      analysis.AnalysisSettings("baselines", windows=2, seasonal_period=20),
    )


def test_explicit_season_counts_rows_on_an_irregular_time_axis() -> None:
  data = uploaded()
  data.frame.loc[22:, "date"] += pd.Timedelta(days=2)
  prepared = analysis.prepare_analysis(
    [data],
    MAPPING,
    SETTINGS,
    analysis.AnalysisSettings("baselines", windows=1, seasonal_period=3),
  )
  result = analysis.run_analysis(Predictor(), prepared)
  seasonal = result.predictions.loc[result.predictions.variant.eq("seasonal_naive")]
  assert seasonal.point.tolist() == [18, 19, 20]


def test_raw_baselines_keep_phase_missing_values_and_future_isolation() -> None:
  values = np.arange(15, dtype=float)
  values[8:10] = np.nan
  values[10:] = 1000
  settings = dataclasses.replace(SETTINGS, horizon=5)
  prepared = analysis.prepare_analysis(
    [uploaded(values)],
    MAPPING,
    settings,
    analysis.AnalysisSettings("baselines", windows=1, seasonal_period=3),
  )
  predictor = Predictor()
  result = analysis.run_analysis(predictor, prepared)
  seasonal = result.predictions.loc[result.predictions.variant.eq("seasonal_naive")]
  np.testing.assert_allclose(
    seasonal.point, [7, np.nan, np.nan, 7, np.nan], equal_nan=True
  )
  assert (
    result.predictions.loc[result.predictions.variant.eq("last_value"), "point"]
    .eq(7)
    .all()
  )
  assert len(predictor.calls) == 1
  assert result.predictions.groupby("variant").scored.sum().eq(2).all()
  overall = result.metrics.loc[result.metrics.scope.eq("overall")]
  assert overall.observations.eq(2).all()
  assert overall.excluded.eq(3).all()
  naive = overall.loc[overall.variant.isin(["last_value", "seasonal_naive"])]
  assert naive.q10_q90_coverage_percent.isna().all()
  assert result.manifest["scoring"]["excluded_pairs"] == 3


def test_equal_counts_with_different_keys_do_not_count_as_matched() -> None:
  frame = pd.DataFrame(
    {
      "variant": ["a", "a", "b", "b"],
      "dataset": ["d"] * 4,
      "target": ["t"] * 4,
      "origin": [4] * 4,
      "step": [1, 2, 1, 3],
      "timestamp": [5, 6, 5, 7],
      "point": [1.0] * 4,
      "actual": [1.0] * 4,
    }
  )
  assert analysis.matched_predictions(frame).scored.tolist() == [
    True,
    False,
    True,
    False,
  ]


def test_mean_rank_gives_targets_equal_weight_despite_different_units() -> None:
  metrics = pd.DataFrame(
    {
      "variant": ["a", "b", "a", "b"],
      "dataset": ["d"] * 4,
      "target": ["small", "small", "large", "large"],
      "scope": ["overall"] * 4,
      "mae": [1.0, 2.0, 2000.0, 1000.0],
    }
  )
  targets, aggregate = analysis.configuration_rankings(metrics)
  assert targets.mae_rank.tolist() == [1, 2, 2, 1]
  assert aggregate.mean_rank.eq(1.5).all()
  assert aggregate.scored_targets.eq(2).all()


def test_nested_bands_and_calibration_preserve_quantiles_and_exclude_crossings() -> (
  None
):
  frame = pd.DataFrame(
    {
      "variant": ["model"] * 3,
      "dataset": ["d"] * 3,
      "target": ["t"] * 3,
      "step": [1] * 3,
      "actual": [5.0, np.nan, 5.0],
    }
  )
  for index in range(1, 10):
    frame[f"q0.{index}"] = float(index)
  frame.loc[2, "q0.4"] = 8.0
  before = frame.copy(deep=True)
  bands = interval_bands(frame)
  assert len(bands) == 12
  assert bands.nominal_coverage_percent.unique().tolist() == [20, 40, 60, 80]
  assert bands.crossed.sum() == 4
  coverage = calibration_table(frame)
  assert coverage.observations.eq(1).all()
  assert coverage.missing_actuals.eq(1).all()
  assert coverage.crossings.eq(1).all()
  assert coverage.observed_coverage_percent.eq(100).all()
  assert coverage.mean_width.tolist() == [2, 4, 6, 8]
  pd.testing.assert_frame_equal(frame, before)
  assert calibration_table(frame.drop(columns="actual")).empty


@pytest.mark.parametrize("label", ["Forecast configurations", "Naive baselines"])
def test_new_workflows_submit_once_and_render_saved_comparisons(label: str) -> None:
  script = """
from pathlib import Path
import numpy as np
import pandas as pd
from timesfm3.analysis_ui import render_analysis
from timesfm3.explorer import DatasetMapping, ForecastSettings, parse_upload
frame = pd.DataFrame({'date': pd.date_range('2026-01-01', periods=40), 'value': np.arange(40, dtype=float)})
data = parse_upload(frame.to_csv(index=False).encode(), '.csv', 'series')
render_analysis([data], DatasetMapping('date', ('value',)), ForecastSettings(horizon=3, context_length=6),
                'cpu', True, lambda device, batch: object(), Path('unused.duckdb'))
"""
  with (
    mock.patch("timesfm3.analysis_ui.list_analyses", return_value=[]),
    mock.patch("timesfm3.analysis_ui.save_analysis") as save,
    mock.patch(
      "timesfm3.analysis_ui.run_analysis",
      side_effect=lambda predictor, prepared, progress: analysis.run_analysis(
        Predictor(), prepared, progress
      ),
    ) as run,
  ):
    app = AppTest.from_string(script, default_timeout=20).run()
    app.selectbox(key="analysis_workflow").set_value(label).run()
    assert not app.exception
    run.assert_not_called()
    next(
      button for button in app.button if button.label == "Run analysis"
    ).click().run()
    assert not app.exception
    assert not app.error
    run.assert_called_once()
    save.assert_called_once()
    artifact = app.session_state["analysis_latest"]
    assert artifact.predictions.variant.nunique() == (
      4 if label == "Forecast configurations" else 3
    )
    app.run()
    run.assert_called_once()
