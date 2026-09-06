"""Issued forecasts stay immutable while observations and tracking evolve."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from timesfm3.analysis import AnalysisArtifact
from timesfm3.explorer import (
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  RunArtifact,
  UploadedDataset,
)
from timesfm3.run_store import (
  RunStoreError,
  link_runs,
  list_tracking_runs,
  load_analysis,
  load_assessments,
  load_recent_runs,
  load_run,
  load_tracked_runs,
  save_analysis,
  save_assessment,
  save_run,
  set_run_tracked,
)
from timesfm3.tracking import assess_run, assessment_zip, associated_datasets


def artifact(index: int = 0, dataset: str = "shop:west") -> RunArtifact:
  return RunArtifact(
    run_id=f"run-{index:02}",
    created_at=(
      pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=index)
    ).isoformat(),
    settings=ForecastSettings(horizon=2, context_length=4),
    mapping=DatasetMapping("date", ("sales",)),
    history=pd.DataFrame({"private_context": [123456]}),
    forecast=pd.DataFrame(
      {
        "dataset": [dataset, dataset],
        "target": ["sales", "sales"],
        "step": [1, 2],
        "timestamp": pd.date_range("2026-01-02", periods=2),
        "point": [10.0, 12.0],
        "q0.1": [8.0, 10.0],
        "q0.9": [12.0, 14.0],
        "lower_40": [9.0, 11.0],
        "upper_40": [11.0, 13.0],
      }
    ),
    metrics=pd.DataFrame(),
    manifest={"schema_version": 2},
    runtime_seconds=0.5,
    device="cpu",
  )


def upload(
  values: tuple[float | None, float | None] = (11.0, 13.0), dataset: str = "shop:west"
) -> UploadedDataset:
  frame = pd.DataFrame(
    {
      "date": pd.date_range("2026-01-02", periods=2),
      "sales": list(values),
      "unused_private_column": [999, 888],
    }
  )
  data = frame.to_csv(index=False).encode()
  return UploadedDataset(
    dataset,
    frame,
    hashlib.sha256(data).hexdigest(),
    len(data),
    int(frame.memory_usage().sum()),
  )


def test_forecast_idempotent_save_and_conflicting_overwrite(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  original = artifact()
  save_run(database, original)
  save_run(database, original)
  restored = load_run(database, original.run_id)
  save_run(database, restored)
  pd.testing.assert_frame_equal(restored.forecast, original.forecast)
  assert restored.history.empty
  changed = original.forecast.assign(point=[99.0, 100.0])
  with pytest.raises(RunStoreError, match="cannot be changed"):
    save_run(database, dataclasses.replace(original, forecast=changed))
  with pytest.raises(RunStoreError, match="cannot be changed"):
    save_run(database, dataclasses.replace(original, manifest={"changed": True}))
  pd.testing.assert_frame_equal(
    load_run(database, original.run_id).forecast, original.forecast
  )
  assert len(load_recent_runs(database)) == 1


@pytest.mark.parametrize("zones", [(None, "Asia/Kolkata"), ("UTC", "America/New_York")])
def test_mixed_timestamp_payloads_preserve_each_dataset(tmp_path: Path, zones) -> None:
  frames = []
  datasets = []
  for dataset, zone in zip(("shop:west", "shop:east"), zones, strict=True):
    current = upload(dataset=dataset)
    forecast = artifact(dataset=dataset).forecast
    if zone is not None:
      forecast = forecast.assign(timestamp=forecast.timestamp.dt.tz_localize(zone))
      current = dataclasses.replace(
        current,
        frame=current.frame.assign(date=current.frame.date.dt.tz_localize(zone)),
      )
    frames.append(forecast)
    datasets.append(current)
  original = dataclasses.replace(
    artifact(), forecast=pd.concat(frames, ignore_index=True)
  )
  database = tmp_path / "runs.duckdb"
  save_run(database, original)
  restored = load_run(database, original.run_id)
  assert [value.isoformat() for value in restored.forecast.timestamp] == [
    value.isoformat() for value in original.forecast.timestamp
  ]
  assert [value.tzinfo is not None for value in restored.forecast.timestamp] == [
    value.tzinfo is not None for value in original.forecast.timestamp
  ]
  assert list(restored.forecast) == list(original.forecast)
  assert restored.forecast.timestamp.dtype == original.forecast.timestamp.dtype
  save_run(database, original)
  associations = {item.dataset_id: item.dataset_id for item in datasets}
  assessment = assess_run(restored, datasets, associations)
  assert len(assessment.comparisons) == 4
  save_assessment(database, assessment)
  persisted_assessment = load_assessments(database, original.run_id)[0]
  assert [
    value.isoformat() for value in persisted_assessment.comparisons.timestamp
  ] == [value.isoformat() for value in assessment.comparisons.timestamp]
  predictions = original.forecast.copy()
  predictions["origin"] = predictions.timestamp.map(
    lambda value: value - pd.Timedelta(days=1)
  )
  analysis = AnalysisArtifact(
    "mixed-analysis", original.created_at, "backtest", predictions, pd.DataFrame(), {}
  )
  save_analysis(database, analysis)
  restored_analysis = load_analysis(database, analysis.analysis_id)
  for column in ("timestamp", "origin"):
    assert [value.isoformat() for value in restored_analysis.predictions[column]] == [
      value.isoformat() for value in predictions[column]
    ]
  assert list(restored_analysis.predictions) == list(predictions)
  with duckdb.connect(str(database)) as connection:
    flags = connection.execute("SELECT timestamp_is_temporal FROM forecasts").fetchall()
    assert all(flag[0] for flag in flags)
    connection.execute("DELETE FROM run_tables WHERE run_id = ?", [original.run_id])
  fallback = load_run(database, original.run_id)
  for dataset in associations:
    expected = original.forecast.loc[original.forecast.dataset == dataset, "timestamp"]
    actual = fallback.forecast.loc[fallback.forecast.dataset == dataset, "timestamp"]
    assert [value.isoformat() for value in actual] == [
      value.isoformat() for value in expected
    ]


def test_tracked_retention_and_untracking_cleans_assessments(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  first = artifact()
  save_run(database, first)
  assessment = assess_run(first, [upload()], {"shop:west": "shop:west"})
  save_assessment(database, assessment)
  for index in range(1, 27):
    save_run(database, artifact(index))
  summaries = list_tracking_runs(database)
  assert len(summaries) == 26
  assert [run.run_id for run in load_tracked_runs(database)] == [first.run_id]
  assert len(load_recent_runs(database)) == 25
  assert load_run(database, first.run_id).run_id == first.run_id
  assert len(load_assessments(database, first.run_id)) == 1
  set_run_tracked(database, first.run_id, False)
  assert len(list_tracking_runs(database)) == 25
  assert load_assessments(database, first.run_id) == []
  with pytest.raises(RunStoreError, match="not found"):
    load_run(database, first.run_id)


def test_legacy_database_reads_do_not_migrate(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  original = artifact(dataset="dataset_1")
  save_run(database, original)
  with duckdb.connect(str(database)) as connection:
    for table in ("run_tables", "tracked_runs", "tracking_links", "run_assessments"):
      connection.execute(f"DROP TABLE {table}")
  before = database.read_bytes()
  restored = load_run(database, original.run_id)
  assert list_tracking_runs(database)[0]["tracked"] is False
  assert load_assessments(database, original.run_id) == []
  assert load_tracked_runs(database) == []
  assert database.read_bytes() == before
  save_run(database, restored)
  set_run_tracked(database, original.run_id, True)
  assert load_tracked_runs(database)[0].run_id == original.run_id
  with pytest.raises(ExplorerError, match="Associate"):
    assess_run(restored, [upload()], {})
  assessment = assess_run(restored, [upload()], {"dataset_1": "shop:west"})
  assert len(assessment.comparisons) == 2


def test_actual_revisions_are_append_only_and_exports_identify_version(
  tmp_path: Path,
) -> None:
  database = tmp_path / "runs.duckdb"
  original = artifact()
  save_run(database, original)
  first = assess_run(original, [upload()], {"shop:west": "shop:west"})
  revised = assess_run(original, [upload((13.0, 15.0))], {"shop:west": "shop:west"})
  save_assessment(database, first)
  save_assessment(database, first)
  save_assessment(database, revised)
  assessments = load_assessments(database, original.run_id)
  assert len(assessments) == 2
  assert assessments[0].metrics.mae.tolist() == [1.0]
  assert assessments[-1].metrics.mae.tolist() == [3.0]
  pd.testing.assert_frame_equal(
    load_run(database, original.run_id).forecast, original.forecast
  )
  with zipfile.ZipFile(io.BytesIO(assessment_zip(first))) as archive:
    manifest = json.loads(archive.read("assessment.json"))
    assert manifest["assessment_id"] == first.assessment_id
    assert manifest["upload_hashes"] == {"shop:west": upload().sha256}
    assert b"unused_private_column" not in archive.read("comparisons.csv")
    assert "calibration.csv" in archive.namelist()
  with duckdb.connect(str(database), read_only=True) as connection:
    payload = connection.execute(
      "SELECT comparisons FROM run_assessments LIMIT 1"
    ).fetchone()[0]
    assert "unused_private_column" not in pd.read_parquet(io.BytesIO(payload))


def test_exact_identity_timestamp_matching_and_missing_actuals() -> None:
  unrelated = upload((1000.0, 1000.0), "shop:east")
  original = artifact()
  partial = upload((11.0, None))
  assessment = assess_run(original, [unrelated, partial], {"shop:west": "shop:west"})
  assert assessment.comparisons.actual.tolist() == [11.0]
  assert assessment.comparisons.timestamp.dt.tz is None
  assert assessment.metrics.observations.tolist() == [1]
  associated = associated_datasets(
    original, [unrelated, partial], {"shop:west": "shop:west"}
  )[0]
  assert associated.dataset_id == partial.dataset_id
  assert associated.frame.attrs["tracking_previous_dataset"] == "shop:west"
  assert "tracking_previous_dataset" not in partial.frame.attrs
  shifted = dataclasses.replace(
    partial, frame=partial.frame.assign(date=pd.date_range("2027-01-01", periods=2))
  )
  with pytest.raises(ExplorerError, match="no observed"):
    assess_run(original, [shifted], {"shop:west": "shop:west"})


def test_assessment_preserves_actuals_recorded_at_issue() -> None:
  original = artifact()
  original = dataclasses.replace(
    original, forecast=original.forecast.assign(actual=[9.0, 11.0])
  )
  assessment = assess_run(original, [upload()], {"shop:west": "shop:west"})
  assert assessment.comparisons.actual.tolist() == [11.0, 13.0]
  assert assessment.comparisons.actual_at_issue.tolist() == [9.0, 11.0]
  assert original.forecast.actual.tolist() == [9.0, 11.0]


def test_tracking_rejects_ambiguous_and_positional_timestamps() -> None:
  original = artifact()
  current = upload()
  duplicated = dataclasses.replace(
    current, frame=current.frame.assign(date=["2026-01-02", "2026-01-02"])
  )
  with pytest.raises(ExplorerError, match="duplicate timestamps"):
    assess_run(original, [duplicated], {"shop:west": "shop:west"})
  positional = dataclasses.replace(original, mapping=DatasetMapping(None, ("sales",)))
  with pytest.raises(ExplorerError, match="timestamp column"):
    assess_run(positional, [current], {"shop:west": "shop:west"})
  numeric = dataclasses.replace(current, frame=current.frame.assign(date=[0, 1]))
  with pytest.raises(ExplorerError, match="row positions"):
    assess_run(original, [numeric], {"shop:west": "shop:west"})
  zoned = dataclasses.replace(
    current, frame=current.frame.assign(date=current.frame.date.dt.tz_localize("UTC"))
  )
  with pytest.raises(ExplorerError, match="timezone convention"):
    assess_run(original, [zoned], {"shop:west": "shop:west"})


def test_aware_timestamps_match_the_same_instant() -> None:
  original = artifact()
  original = dataclasses.replace(
    original,
    forecast=original.forecast.assign(
      timestamp=original.forecast.timestamp.dt.tz_localize("UTC")
    ),
  )
  current = upload()
  current = dataclasses.replace(
    current,
    frame=current.frame.assign(
      date=current.frame.date.dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
    ),
  )
  assessment = assess_run(original, [current], {"shop:west": "shop:west"})
  assert len(assessment.comparisons) == 2


def test_refresh_links_keep_both_vintages(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  first, second = artifact(0), artifact(1)
  save_run(database, first)
  save_run(database, second)
  link_runs(database, second.run_id, first.run_id, {"shop:west": "shop:west"})
  link_runs(database, second.run_id, first.run_id, {"shop:west": "shop:west"})
  assert len(load_tracked_runs(database)) == 2
  with pytest.raises(RunStoreError, match="cannot be changed"):
    link_runs(database, second.run_id, first.run_id, {"shop:west": "shop:east"})
  pd.testing.assert_frame_equal(
    load_run(database, first.run_id).forecast, first.forecast
  )


def _tracking_app(database: str) -> None:
  from pathlib import Path

  from timesfm3.tracking_ui import render_tracking

  render_tracking([], Path(database))


def test_tracking_ui_reads_saved_assessments_without_uploads(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  original = artifact()
  save_run(database, original)
  save_assessment(
    database, assess_run(original, [upload()], {"shop:west": "shop:west"})
  )
  app = AppTest.from_function(_tracking_app, args=(str(database),)).run()
  assert not app.exception
  assert any(select.label == "Actuals version" for select in app.selectbox)
  assert app.dataframe


def test_tracking_ui_confirms_before_removing_retention(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  original = artifact()
  save_run(database, original)
  set_run_tracked(database, original.run_id, True)
  app = AppTest.from_function(_tracking_app, args=(str(database),)).run()

  next(button for button in app.button if button.label == "Stop tracking").click().run()
  assert not app.exception
  assert len(load_tracked_runs(database)) == 1
  assert any(button.label == "Keep tracking" for button in app.button)

  app.button(key=f"tracking_confirm_{original.run_id}").click().run()
  assert not app.exception
  assert load_tracked_runs(database) == []


def _tracking_upload_app(database: str) -> None:
  import dataclasses
  import hashlib
  from pathlib import Path

  import pandas as pd
  import streamlit as st

  from timesfm3.explorer import UploadedDataset
  from timesfm3.tracking_ui import render_tracking

  frame = pd.DataFrame(
    {"date": pd.date_range("2026-01-02", periods=2), "sales": [11.0, 13.0]}
  )
  data = frame.to_csv(index=False).encode()
  current = UploadedDataset(
    "shop:west", frame, hashlib.sha256(data).hexdigest(), len(data), 128
  )

  def refresh(previous, datasets):
    st.session_state.refresh_calls = st.session_state.get("refresh_calls", 0) + 1
    assert datasets[0].frame.attrs["tracking_previous_dataset"] in {
      "shop:west",
      "dataset_1",
    }
    return dataclasses.replace(
      previous, run_id="refreshed-run", created_at="2026-01-04T00:00:00+00:00"
    )

  render_tracking([current], Path(database), refresh_run=refresh, acknowledged=True)


def test_tracking_ui_submits_actuals_without_forecasting(tmp_path: Path) -> None:
  database = tmp_path / "runs.duckdb"
  save_run(database, artifact())
  app = AppTest.from_function(_tracking_upload_app, args=(str(database),)).run()
  assert not app.exception
  assert load_assessments(database, artifact().run_id) == []
  next(
    button for button in app.button if button.label == "Evaluate against actuals"
  ).click().run()
  assert not app.exception
  assert len(load_assessments(database, artifact().run_id)) == 1
  assert len(list_tracking_runs(database)) == 1
  assert app.session_state.filtered_state.get("refresh_calls", 0) == 0
  app.run()
  assert len(load_assessments(database, artifact().run_id)) == 1


def test_tracking_ui_refreshes_once_and_legacy_needs_association(
  tmp_path: Path,
) -> None:
  database = tmp_path / "runs.duckdb"
  save_run(database, artifact())
  app = AppTest.from_function(_tracking_upload_app, args=(str(database),)).run()
  next(
    button for button in app.button if button.label == "Refresh forecast"
  ).click().run()
  assert not app.exception
  assert len(load_tracked_runs(database)) == 2
  assert app.session_state.filtered_state["refresh_calls"] == 1
  app.run()
  assert app.session_state.filtered_state["refresh_calls"] == 1
  legacy_database = tmp_path / "legacy.duckdb"
  save_run(legacy_database, artifact(dataset="dataset_1"))
  legacy_app = AppTest.from_function(
    _tracking_upload_app, args=(str(legacy_database),)
  ).run()
  next(
    button for button in legacy_app.button if button.label == "Refresh forecast"
  ).click().run()
  assert not legacy_app.exception
  assert legacy_app.error
  assert legacy_app.session_state.filtered_state.get("refresh_calls", 0) == 0
