# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Headless smoke tests for the Streamlit entry point."""

from pathlib import Path
from unittest import mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from timesfm3.explorer import DatasetMapping, ForecastSettings, RunArtifact
from timesfm3.run_store import RunStoreError


def persisted_artifact() -> RunArtifact:
  return RunArtifact(
    run_id="saved-run",
    created_at="2026-01-01T00:00:00+00:00",
    settings=ForecastSettings(horizon=1, context_length=2),
    mapping=DatasetMapping("date", ("sales",)),
    history=pd.DataFrame(columns=["dataset", "target", "step", "timestamp", "value"]),
    forecast=pd.DataFrame(
      {
        "dataset": ["dataset_1"],
        "target": ["sales"],
        "step": [1],
        "timestamp": [pd.Timestamp("2026-01-02")],
        "point": [1.0],
      }
    ),
    metrics=pd.DataFrame(),
    manifest={"run_id": "saved-run"},
    runtime_seconds=0.1,
    device="cpu",
  )


def test_app_starts_with_demo_without_loading_model() -> None:
  app = Path(__file__).parents[1] / "streamlit_app.py"
  test_app = AppTest.from_file(str(app), default_timeout=20).run()
  assert not test_app.exception
  assert test_app.title[0].value == "TimesFM-3 explorer"
  assert test_app.dataframe
  assert test_app.session_state["runs"] is not None
  tab_labels = [tab.label for tab in test_app.tabs]
  assert [
    label
    for label in tab_labels
    if label in {"Prepare", "Forecast", "Evaluate", "Track"}
  ] == ["Prepare", "Forecast", "Evaluate", "Track"]
  assert any(button.label == "Save settings for analysis" for button in test_app.button)
  submit = next(button for button in test_app.button if button.label == "Run forecast")
  assert submit.disabled
  next(
    item for item in test_app.checkbox if item.label.startswith("I understand")
  ).set_value(True).run()
  submit = next(button for button in test_app.button if button.label == "Run forecast")
  assert not submit.disabled


def test_app_loads_persisted_run_without_input_history() -> None:
  app = Path(__file__).parents[1] / "streamlit_app.py"
  with mock.patch(
    "timesfm3.run_store.load_recent_runs", return_value=[persisted_artifact()]
  ):
    test_app = AppTest.from_file(str(app), default_timeout=20).run()
  assert not test_app.exception
  assert test_app.session_state["runs"][0].run_id == "saved-run"


def test_demo_switch_has_one_source_of_truth_for_role_widgets(caplog) -> None:
  app = Path(__file__).parents[1] / "streamlit_app.py"
  test_app = AppTest.from_file(str(app), default_timeout=20).run()

  test_app.segmented_control(key="demo_kind").set_value("Univariate").run()
  test_app.segmented_control(key="demo_kind").set_value(
    "Multivariate + covariates"
  ).run()

  assert not test_app.exception
  assert "created with a default value" not in caplog.text


def test_app_warns_when_persistence_is_unavailable() -> None:
  app = Path(__file__).parents[1] / "streamlit_app.py"
  with mock.patch(
    "timesfm3.run_store.load_recent_runs",
    side_effect=RunStoreError("Could not load local run history"),
  ):
    test_app = AppTest.from_file(str(app), default_timeout=20).run()
  assert not test_app.exception
  assert "browser session" in test_app.warning[0].value
