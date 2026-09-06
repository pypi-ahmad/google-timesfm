"""Analysis forms submit once and render persisted outputs without inference."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from timesfm3.explorer import ExplorerError

SCRIPT = """
from pathlib import Path
from timesfm3.analysis_ui import render_analysis
from timesfm3.explorer import DatasetMapping, ForecastSettings, demo_dataset, parse_upload
data = parse_upload(demo_dataset("multivariate").to_csv(index=False).encode(), ".csv", "demo")
mapping = DatasetMapping("date", ("sales", "demand"), ("temperature",), ("promotion",))
render_analysis([data], mapping, ForecastSettings(horizon=2, context_length=16),
                "cpu", True, lambda device, batch: object(), Path("unused.duckdb"))
"""


def artifact(kind: str = "backtest") -> SimpleNamespace:
  return SimpleNamespace(
    analysis_id="test-analysis",
    created_at="2026-01-01T00:00:00+00:00",
    kind=kind,
    predictions=pd.DataFrame(
      {
        "dataset": ["demo"],
        "target": ["sales"],
        "variant": ["baseline"],
        "origin": [10],
        "step": [1],
        "timestamp": [10],
        "point": [3.0],
        "actual": [5.0],
        "q0.1": [2.0],
        "q0.9": [4.0],
      }
    ),
    metrics=pd.DataFrame(),
    manifest={},
  )


@pytest.mark.parametrize(
  "label,kind",
  [
    ("Rolling backtesting", "backtest"),
    ("Anomaly detection", "anomaly"),
    ("What-if scenarios", "scenario"),
    ("Joint versus independent", "joint_independent"),
    ("Covariate usefulness", "covariates"),
  ],
)
def test_workflows_submit_once(label: str, kind: str) -> None:
  template = pd.DataFrame(
    {
      "dataset": ["demo"],
      "row": [10],
      "timestamp": [10],
      "covariate": ["promotion"],
      "value": [0.0],
    }
  )
  result = artifact(kind)
  flagged = result.predictions.assign(
    flagged=True, direction="above", residual=2.0, distance=1.0
  )
  with (
    mock.patch("timesfm3.analysis_ui.list_analyses", return_value=[]),
    mock.patch("timesfm3.analysis_ui.save_analysis") as save,
    mock.patch(
      "timesfm3.analysis_ui.prepare_analysis",
      return_value=SimpleNamespace(forecast_count=1, prediction_rows=1),
    ),
    mock.patch("timesfm3.analysis_ui.run_analysis", return_value=result) as run,
    mock.patch("timesfm3.analysis_ui.analysis_zip", return_value=b"zip"),
    mock.patch("timesfm3.analysis_ui.scenario_template", return_value=template),
    mock.patch("timesfm3.analysis_ui.scenario_from_edits", return_value=object()),
    mock.patch("timesfm3.analysis_ui.scenario_deltas", return_value=result.predictions),
    mock.patch("timesfm3.analysis_ui.anomaly_table", return_value=flagged),
  ):
    app = AppTest.from_string(SCRIPT, default_timeout=20).run()
    assert not app.exception
    app.selectbox(key="analysis_workflow").set_value(label).run()
    assert not app.exception
    run.assert_not_called()
    next(
      button for button in app.button if button.label == "Run analysis"
    ).click().run()
    assert not app.exception
    run.assert_called_once()
    save.assert_called_once()
    app.run()
    assert not app.exception
    run.assert_called_once()
    app.selectbox(key="analysis_result_test-analysis_target").set_value("sales").run()
    assert not app.exception
    run.assert_called_once()


def test_apply_settings_does_not_forecast() -> None:
  with (
    mock.patch("timesfm3.explorer.load_forecaster") as load,
    mock.patch("timesfm3.explorer.execute_forecast") as run,
    mock.patch("timesfm3.run_store.load_recent_runs", return_value=[]),
    mock.patch("timesfm3.analysis_ui.list_analyses", return_value=[]),
  ):
    app = AppTest.from_file(
      str(Path(__file__).parents[1] / "streamlit_app.py"), default_timeout=20
    ).run()
    next(item for item in app.number_input if item.label == "Horizon").set_value(8)
    next(
      item for item in app.button if item.label == "Save settings for analysis"
    ).click().run()
    assert not app.exception
    assert next(item for item in app.number_input if item.label == "Horizon").value == 8
    load.assert_not_called()
    run.assert_not_called()


def test_empty_covariate_selection_is_rejected() -> None:
  with (
    mock.patch("timesfm3.analysis_ui.list_analyses", return_value=[]),
    mock.patch("timesfm3.analysis_ui.prepare_analysis") as prepare,
    mock.patch("timesfm3.analysis_ui.run_analysis") as run,
  ):
    app = AppTest.from_string(SCRIPT, default_timeout=20).run()
    app.selectbox(key="analysis_workflow").set_value("Covariate usefulness").run()
    app.multiselect[0].set_value([])
    next(item for item in app.button if item.label == "Run analysis").click().run()
    assert not app.exception
    assert app.error[0].value == "Select at least one covariate to assess."
    prepare.assert_not_called()
    run.assert_not_called()


def test_invalid_analysis_never_acquires_model() -> None:
  with (
    mock.patch("timesfm3.analysis_ui.list_analyses", return_value=[]),
    mock.patch(
      "timesfm3.analysis_ui.prepare_analysis",
      side_effect=ExplorerError("Insufficient context"),
    ),
    mock.patch("timesfm3.analysis_ui.run_analysis") as run,
  ):
    script = SCRIPT.replace(
      "lambda device, batch: object()",
      "lambda device, batch: (_ for _ in ()).throw(AssertionError('model loaded'))",
    )
    app = AppTest.from_string(script, default_timeout=20).run()
    next(
      button for button in app.button if button.label == "Run analysis"
    ).click().run()
    assert not app.exception
    assert app.error[0].value == "Insufficient context"
    run.assert_not_called()


def test_saved_analysis_without_uploaded_data() -> None:
  script = """
from pathlib import Path
from timesfm3.analysis_ui import render_analysis
render_analysis([], None, None, "cpu", False, lambda *args: None, Path("unused.duckdb"))
"""
  result = artifact()
  with (
    mock.patch(
      "timesfm3.analysis_ui.list_analyses",
      return_value=[
        {
          "analysis_id": result.analysis_id,
          "created_at": result.created_at,
          "kind": result.kind,
        }
      ],
    ),
    mock.patch("timesfm3.analysis_ui.load_analysis", return_value=result),
    mock.patch("timesfm3.analysis_ui.analysis_zip", return_value=b"zip"),
    mock.patch("timesfm3.analysis_ui.run_analysis") as run,
  ):
    app = AppTest.from_string(script, default_timeout=20).run()
    next(
      button for button in app.button if button.label == "Load saved analysis"
    ).click().run()
    assert not app.exception
    assert app.session_state["analysis_latest"].analysis_id == "test-analysis"
    run.assert_not_called()
