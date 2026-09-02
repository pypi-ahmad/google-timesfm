# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Tests for TimesFM-3 explorer business logic."""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from timesfm3 import ForecastOutput, explorer


def upload(
  frame: pd.DataFrame, dataset_id: str = "dataset_1"
) -> explorer.UploadedDataset:
  """Create an upload through the real CSV parser."""
  return explorer.parse_upload(frame.to_csv(index=False).encode(), "csv", dataset_id)


def simple_frame(rows: int = 12, future: int = 3) -> pd.DataFrame:
  """Create upload-shaped target and known-future data."""
  target = np.arange(rows, dtype=float)
  target[rows - future :] = np.nan
  return pd.DataFrame(
    {
      "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
      "target": target,
      "past": np.arange(rows, dtype=float) * 2,
      "known": np.arange(rows, dtype=float) % 2,
    }
  )


def mapping() -> explorer.DatasetMapping:
  return explorer.DatasetMapping("date", ("target",), ("past",), ("known",))


def settings(**changes: object) -> explorer.ForecastSettings:
  base = explorer.ForecastSettings(horizon=3, context_length=6)
  return dataclasses.replace(base, **changes)


class FakePredictor:
  """Records evaluator arguments and returns deterministic forecasts."""

  device = torch.device("cpu")

  def __init__(self) -> None:
    self.arguments: dict[str, object] = {}

  def predict_batch(self, **kwargs: Any):
    self.arguments = kwargs
    contexts = kwargs["contexts"]
    horizon = int(kwargs["horizon"])
    return_quantiles = bool(kwargs["return_quantiles"])
    outputs = []
    for index, context in enumerate(contexts):
      targets = np.atleast_2d(context).shape[0]
      point = np.full((targets, horizon), index + 2.0)
      quantiles = None
      if return_quantiles:
        quantiles = np.stack(
          [point + offset for offset in np.linspace(-1, 1, 9)], axis=-1
        )
      outputs.append(
        ForecastOutput(
          ts_id=f"dataset_{index + 1}", forecast=point, quantiles=quantiles
        )
      )
    return outputs


def test_parse_csv_and_parquet() -> None:
  frame = simple_frame()
  csv_upload = upload(frame)
  assert csv_upload.dataset_id == "dataset_1"
  assert len(csv_upload.sha256) == 64
  assert csv_upload.memory_bytes > 0

  buffer = io.BytesIO()
  frame.to_parquet(buffer, index=False)
  parquet = explorer.parse_upload(buffer.getvalue(), ".parquet", "dataset_2")
  assert list(parquet.frame.columns) == list(frame.columns)


def test_parse_rejects_decoded_memory_limit(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(explorer, "MAX_DECODED_BYTES", 1)
  with pytest.raises(explorer.ExplorerError, match="memory limit"):
    upload(simple_frame())


@pytest.mark.parametrize("value", [np.inf, -np.inf, float(np.finfo(np.float32).max) * 2])
def test_prepare_rejects_nonfinite_or_float32_overflow(value: float) -> None:
  frame = simple_frame()
  frame.loc[0, "target"] = value
  with pytest.raises(explorer.ExplorerError):
    explorer.prepare_batch([upload(frame)], mapping(), settings())


@pytest.mark.parametrize("suffix", ["xlsx", "json", "txt"])
def test_parse_rejects_unsupported_type(suffix: str) -> None:
  with pytest.raises(explorer.ExplorerError, match="Only CSV and Parquet"):
    explorer.parse_upload(b"data", suffix, "bad")


def test_parse_rejects_empty_invalid_and_large(monkeypatch: pytest.MonkeyPatch) -> None:
  with pytest.raises(explorer.ExplorerError, match="no data"):
    explorer.parse_upload(b"a\n", "csv", "empty")
  with pytest.raises(explorer.ExplorerError, match="Could not parse"):
    explorer.parse_upload(b"\xff", "csv", "invalid")
  monkeypatch.setattr(explorer, "MAX_UPLOAD_BYTES", 2)
  with pytest.raises(explorer.ExplorerError, match="50 MB"):
    explorer.parse_upload(b"abc", "csv", "large")


def test_prepare_future_with_covariates() -> None:
  batch = explorer.prepare_batch([upload(simple_frame())], mapping(), settings())
  item = batch.series[0]
  assert item.context.shape == (1, 6)
  assert item.past_only is not None and item.past_only.shape == (1, 6)
  assert item.past_future is not None and item.past_future.shape == (1, 9)
  assert len(item.future_time) == 3
  assert batch.chunking.total_variates == 3


def test_prepare_holdout_sorts_and_tracks_missing_values() -> None:
  frame = simple_frame(future=0).iloc[::-1].reset_index(drop=True)
  frame.loc[5, "target"] = np.nan
  batch = explorer.prepare_batch([upload(frame)], mapping(), settings(task="holdout"))
  item = batch.series[0]
  assert item.actual is not None and item.actual.shape == (1, 3)
  operations = [record["operation"] for record in item.lineage]
  assert "sort" in operations
  assert "truncate_context" in operations
  assert "timesfm_linear_interpolation" in operations


def test_prepare_generates_integer_future_axis_without_timestamp() -> None:
  frame = pd.DataFrame({"target": np.arange(8, dtype=float)})
  batch = explorer.prepare_batch(
    [upload(frame)],
    explorer.DatasetMapping(None, ("target",)),
    settings(horizon=2, context_length=5),
  )
  assert batch.series[0].future_time == (8, 9)


@pytest.mark.parametrize(
  ("frame_change", "expected"),
  [
    (lambda frame: frame.assign(target=np.nan), "no observed target"),
    (lambda frame: frame.assign(target="bad"), "non-numeric"),
    (lambda frame: frame.assign(date="bad"), "invalid values"),
    (
      lambda frame: frame.assign(date=pd.Timestamp("2026-01-01")),
      "duplicates",
    ),
  ],
)
def test_prepare_rejects_bad_data(frame_change, expected: str) -> None:
  with pytest.raises(explorer.ExplorerError, match=expected):
    explorer.prepare_batch(
      [upload(frame_change(simple_frame()))], mapping(), settings()
    )


def test_prepare_rejects_short_context_and_missing_future_covariate() -> None:
  frame = simple_frame()
  frame.loc[:7, "target"] = np.nan
  with pytest.raises(explorer.ExplorerError, match="at least two context values"):
    explorer.prepare_batch([upload(frame)], mapping(), settings())

  frame = simple_frame()
  frame.loc[10, "known"] = np.nan
  with pytest.raises(explorer.ExplorerError, match="missing known-future"):
    explorer.prepare_batch([upload(frame)], mapping(), settings())


def test_prepare_rejects_covariates_in_univariate_mode() -> None:
  with pytest.raises(explorer.ExplorerError, match="does not use covariates"):
    explorer.prepare_batch(
      [upload(simple_frame())], mapping(), settings(mode="univariate")
    )


def test_chunking_requires_approval_and_is_deterministic() -> None:
  columns = {f"target_{index}": np.arange(8) for index in range(33)}
  frame = pd.DataFrame(columns)
  wide_mapping = explorer.DatasetMapping(None, tuple(columns))
  with pytest.raises(explorer.ExplorerError, match="benchmark chunking"):
    explorer.prepare_batch(
      [upload(frame)], wide_mapping, settings(horizon=2, context_length=6)
    )
  batch = explorer.prepare_batch(
    [upload(frame)],
    wide_mapping,
    settings(horizon=2, context_length=6, allow_benchmark_chunking=True),
  )
  assert batch.chunking.targets_per_chunk == 32
  assert batch.chunking.target_chunks == 2


def test_mapping_and_total_upload_validation(monkeypatch: pytest.MonkeyPatch) -> None:
  dataset = upload(simple_frame())
  duplicate = explorer.DatasetMapping(None, ("target",), ("target",))
  with pytest.raises(explorer.ExplorerError, match="only one role"):
    explorer.prepare_batch([dataset], duplicate, settings())
  missing = explorer.DatasetMapping(None, ("missing",))
  with pytest.raises(explorer.ExplorerError, match="Missing mapped columns"):
    explorer.prepare_batch([dataset], missing, settings())
  monkeypatch.setattr(explorer, "MAX_TOTAL_UPLOAD_BYTES", 1)
  with pytest.raises(explorer.ExplorerError, match="Combined uploads"):
    explorer.prepare_batch([dataset], mapping(), settings())


@pytest.mark.parametrize(
  "changes",
  [
    {"horizon": 0},
    {"context_length": 0},
    {"batch_size": 0},
    {"task": "bad"},
    {"mode": "bad"},
    {"padding_mode": "bad"},
  ],
)
def test_settings_validate_server_side(changes: dict[str, object]) -> None:
  with pytest.raises(explorer.ExplorerError):
    settings(**changes).validate()


def test_run_forecast_forwards_every_stable_flag() -> None:
  current_settings = settings(
    task="holdout",
    mode="multivariate",
    return_quantiles=True,
    use_symmetric_averaging=False,
    make_positive=False,
    sort_quantiles=False,
    use_znorm=True,
    padding_mode="edge",
  )
  batch = explorer.prepare_batch(
    [upload(simple_frame(future=0))], mapping(), current_settings
  )
  predictor = FakePredictor()
  outputs, runtime = explorer.run_forecast(predictor, batch)
  assert len(outputs) == 1
  assert runtime >= 0
  assert predictor.arguments["use_symmetric_averaging"] is False
  assert predictor.arguments["make_positive"] is False
  assert predictor.arguments["sort_quantiles"] is False
  assert predictor.arguments["use_znorm"] is True
  assert predictor.arguments["padding_mode"] == "edge"
  assert predictor.arguments["univariate"] is False


def test_artifact_metrics_and_zip_are_reproducible() -> None:
  batch = explorer.prepare_batch(
    [upload(simple_frame(future=0))], mapping(), settings(task="holdout")
  )
  predictor = FakePredictor()
  outputs, runtime = explorer.run_forecast(predictor, batch)
  artifact = explorer.make_run_artifact(
    batch, outputs, runtime, "cpu", repository_revision="abc123"
  )
  assert set(artifact.forecast).issuperset(
    {"dataset", "target", "point", "actual", "q0.1", "q0.9"}
  )
  assert set(artifact.metrics).issuperset(
    {"mae", "rmse", "smape_percent", "mean_pinball_loss", "q10_q90_coverage_percent"}
  )
  assert artifact.manifest["repository_revision"] == "abc123"
  assert "HF_TOKEN" not in json.dumps(artifact.manifest)

  with zipfile.ZipFile(io.BytesIO(explorer.artifact_zip(artifact))) as archive:
    assert set(archive.namelist()) == {"forecast.csv", "metrics.csv", "run.json"}
    manifest = json.loads(archive.read("run.json"))
    assert manifest["run_id"] == artifact.run_id


def test_forecast_table_rejects_bad_outputs() -> None:
  batch = explorer.prepare_batch([upload(simple_frame())], mapping(), settings())
  with pytest.raises(explorer.ExplorerError, match="number of outputs"):
    explorer.forecast_table(batch, [])
  with pytest.raises(explorer.ExplorerError, match="No forecast"):
    explorer.forecast_table(batch, [ForecastOutput()])
  with pytest.raises(explorer.ExplorerError, match="Unexpected forecast shape"):
    explorer.forecast_table(batch, [ForecastOutput(forecast=np.ones((2, 2)))])
  with pytest.raises(explorer.ExplorerError, match="non-finite"):
    explorer.forecast_table(
      batch, [ForecastOutput(forecast=np.full((1, 3), np.inf))]
    )


def test_two_timestamp_history_uses_last_interval() -> None:
  axis = pd.Series(pd.date_range("2026-01-01", periods=2, freq="2D"))
  future = explorer._future_axis(axis, history_end=2, horizon=2, uploaded_end=2)
  assert future == (pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07"))


def test_execute_forecast_is_complete_pipeline() -> None:
  artifact = explorer.execute_forecast(
    FakePredictor(),
    [upload(simple_frame())],
    mapping(),
    settings(),
    repository_revision="revision",
  )
  assert artifact.device == "cpu"
  assert artifact.manifest["repository_revision"] == "revision"


def test_csv_export_neutralizes_formula_cells() -> None:
  current_mapping = explorer.DatasetMapping("date", ("=SUM(A1:A2)",))
  frame = simple_frame().rename(columns={"target": "=SUM(A1:A2)"})
  artifact = explorer.execute_forecast(
    FakePredictor(), [upload(frame)], current_mapping, settings()
  )
  with zipfile.ZipFile(io.BytesIO(explorer.artifact_zip(artifact))) as archive:
    exported = pd.read_csv(io.BytesIO(archive.read("forecast.csv")))
  assert exported.loc[0, "target"].startswith("'=")


def test_point_only_artifact_has_no_holdout_metrics_file() -> None:
  current_settings = settings(return_quantiles=False)
  batch = explorer.prepare_batch([upload(simple_frame())], mapping(), current_settings)
  outputs, runtime = explorer.run_forecast(FakePredictor(), batch)
  artifact = explorer.make_run_artifact(batch, outputs, runtime, "cpu")
  assert artifact.metrics.empty
  with zipfile.ZipFile(io.BytesIO(explorer.artifact_zip(artifact))) as archive:
    assert set(archive.namelist()) == {"forecast.csv", "run.json"}


def test_demo_capabilities_revision_and_loader(monkeypatch: pytest.MonkeyPatch) -> None:
  assert list(explorer.demo_dataset("univariate")) == ["date", "sales"]
  assert list(explorer.demo_dataset("multivariate")) == [
    "date",
    "sales",
    "demand",
    "temperature",
    "promotion",
  ]
  report = explorer.capability_report()
  assert report.python
  assert report.torch
  assert explorer.repository_revision(Path.cwd()) != "unknown"
  assert explorer.repository_revision(Path("Z:/does-not-exist")) == "unknown"

  sentinel = object()
  with mock.patch.object(
    explorer.TimesFM3Evaluator, "from_pretrained", return_value=sentinel
  ) as loader:
    assert explorer.load_forecaster("cpu", 2) is sentinel
    loader.assert_called_once_with(
      explorer.CHECKPOINT_ID, device="cpu", per_core_batch_size=2
    )
