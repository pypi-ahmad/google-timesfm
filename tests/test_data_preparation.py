# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Preparation boundaries: grouped identity, calendar dates, and unchanged observations."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from timesfm3 import explorer
from timesfm3.data_preparation import (
  CalendarEvent,
  CalendarSettings,
  group_sources,
  imputation_preview,
  prepare_sources,
  quality_report,
  restore_preparation,
)

MAPPING = explorer.DatasetMapping("date", ("value",))
SETTINGS = explorer.ForecastSettings(horizon=2, context_length=5)


def upload(frame: pd.DataFrame, identifier: str = "file_1") -> explorer.UploadedDataset:
  return explorer.parse_upload(frame.to_csv(index=False).encode(), "csv", identifier)


def daily(rows: int = 5) -> pd.DataFrame:
  return pd.DataFrame(
    {
      "date": pd.date_range("2026-01-01", periods=rows),
      "value": np.arange(rows, dtype=float),
    }
  )


def test_group_identity_survives_row_order_and_upload_order() -> None:
  frame = pd.concat(
    [daily().assign(store="B"), daily().assign(store="A")], ignore_index=True
  )
  first = upload(frame)
  second = upload(daily(), "file_2")
  grouped, mapping = prepare_sources(
    [first], MAPPING, source_names={"file_1": "sales"}, group_columns=("store",)
  )
  shuffled, _ = prepare_sources(
    [upload(frame.iloc[::-1])],
    MAPPING,
    source_names={"file_1": "sales"},
    group_columns=("store",),
  )
  assert {item.dataset_id for item in grouped} == {item.dataset_id for item in shuffled}
  assert len(grouped) == 2
  assert mapping == MAPPING
  for item in grouped:
    assert item.frame["store"].nunique() == 1
    assert item.frame["value"].tolist() == list(range(5))
  left, _ = prepare_sources(
    [upload(daily()), second],
    MAPPING,
    source_names={"file_1": "sales", "file_2": "inventory"},
  )
  right, _ = prepare_sources(
    [second, upload(daily())],
    MAPPING,
    source_names={"file_1": "sales", "file_2": "inventory"},
  )
  assert {item.dataset_id for item in left} == {item.dataset_id for item in right}


def test_grouping_preserves_multivariate_targets_and_source_byte_budget(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  frame = pd.concat(
    [daily().assign(store=store, demand=np.arange(5) + 20) for store in ("A", "B", "C")]
  )
  dataset = upload(frame)
  monkeypatch.setattr(explorer, "MAX_TOTAL_UPLOAD_BYTES", dataset.byte_size + 1)
  mapping = dataclasses.replace(MAPPING, targets=("value", "demand"))
  groups, prepared_mapping = prepare_sources(
    [dataset], mapping, source_names={}, group_columns=("store",)
  )
  assert sum(item.byte_size for item in groups) == dataset.byte_size
  batch = explorer.prepare_batch(groups, prepared_mapping, SETTINGS)
  assert len(batch.series) == 3
  assert all(item.context.shape == (2, 5) for item in batch.series)
  assert batch.chunking.total_variates == 2


@pytest.mark.parametrize("group_columns", [("value",), ("date",), ("store", "store")])
def test_rejects_ambiguous_group_roles(group_columns: tuple[str, ...]) -> None:
  with pytest.raises(explorer.ExplorerError, match="Group columns"):
    prepare_sources(
      [upload(daily().assign(store="A"))],
      MAPPING,
      source_names={},
      group_columns=group_columns,
    )


def test_missing_group_keys_and_duplicate_source_names_are_errors() -> None:
  frame = daily().assign(store=["A", "A", None, "B", "B"])
  with pytest.raises(explorer.ExplorerError, match="missing group identifiers"):
    prepare_sources([upload(frame)], MAPPING, source_names={}, group_columns=("store",))
  with pytest.raises(explorer.ExplorerError, match="unique, non-empty"):
    prepare_sources(
      [upload(daily()), upload(daily(), "second")],
      MAPPING,
      source_names={"file_1": "sales", "second": "sales"},
    )


def test_calendar_extends_numeric_covariates_without_filling_targets() -> None:
  source = upload(daily())
  original = source.frame.copy(deep=True)
  calendar = CalendarSettings(
    weekday=True,
    month=True,
    events=(CalendarEvent("launch", "2026-01-06", "2026-01-07"),),
  )
  datasets, mapping = prepare_sources(
    [source], MAPPING, source_names={}, calendar=calendar, horizon=2
  )
  frame = datasets[0].frame
  assert len(frame) == 7
  assert frame["value"].iloc[-2:].isna().all()
  assert frame["calendar_weekday"].tolist() == [3, 4, 5, 6, 0, 1, 2]
  assert frame["calendar_month"].tolist() == [1] * 7
  assert frame["calendar_event:launch"].tolist() == [0, 0, 0, 0, 0, 1, 1]
  batch = explorer.prepare_batch(datasets, mapping, SETTINGS)
  assert batch.series[0].past_future.shape == (3, 7)
  assert batch.chunking.total_variates == 4
  pd.testing.assert_frame_equal(source.frame, original)
  assert json.loads(json.dumps(frame.attrs["preparation"]))["appended_rows"] == 2


def test_holidays_use_country_subdivision_and_displayed_local_dates() -> None:
  # Built directly as an UploadedDataset rather than via upload()/parse_upload
  # so the Asia/Kolkata tz-aware "date" column survives untouched: a CSV
  # round trip changes how the timezone offset is (re)represented, which
  # made this test's holiday-day assertions flaky/timezone-dependent before.
  frame = pd.DataFrame(
    {
      "date": pd.date_range("2026-01-01", periods=3, tz="Asia/Kolkata"),
      "value": [1, 2, 3],
    }
  )
  source = explorer.UploadedDataset(
    "file_1", frame, "hash", 100, int(frame.memory_usage(deep=True).sum())
  )
  datasets, _ = prepare_sources(
    [source],
    MAPPING,
    source_names={},
    calendar=CalendarSettings(holiday_country="US", holiday_subdivision="CA"),
    horizon=1,
  )
  assert datasets[0].frame["calendar_holiday"].tolist() == [1, 0, 0, 0]
  assert datasets[0].frame["date"].iloc[0].day == 1
  metadata = datasets[0].frame.attrs["preparation"]
  assert metadata["holidays_version"]
  assert "known before every forecast origin" in metadata["known_future_policy"]
  assert "holidays_version" not in metadata["calendar"]


@pytest.mark.parametrize(
  "calendar",
  [
    CalendarSettings(events=(CalendarEvent("sale", "2026-01-02", "2026-01-01"),)),
    CalendarSettings(events=(CalendarEvent("sale", "bad date"),)),
    CalendarSettings(events=(CalendarEvent("", "2026-01-01"),)),
    CalendarSettings(
      events=(CalendarEvent("sale", "2026-01-01"), CalendarEvent("sale", "2026-01-02"))
    ),
    CalendarSettings(holiday_subdivision="CA"),
    CalendarSettings(holiday_country="NO_SUCH_COUNTRY"),
  ],
)
def test_rejects_invalid_calendar_settings(calendar: CalendarSettings) -> None:
  with pytest.raises(explorer.ExplorerError):
    prepare_sources(
      [upload(daily())], MAPPING, source_names={}, calendar=calendar, horizon=2
    )


def test_calendar_rejects_name_conflicts_and_absent_timestamp() -> None:
  with pytest.raises(explorer.ExplorerError, match="already exist"):
    prepare_sources(
      [upload(daily().assign(calendar_weekday=99))],
      MAPPING,
      source_names={},
      calendar=CalendarSettings(weekday=True),
    )
  with pytest.raises(explorer.ExplorerError, match="timestamp column"):
    prepare_sources(
      [upload(daily())],
      dataclasses.replace(MAPPING, timestamp=None),
      source_names={},
      calendar=CalendarSettings(month=True),
    )


def test_ambiguous_calendar_extension_requires_frequency_without_repairing_history() -> (
  None
):
  source = upload(daily().drop(index=2))
  with pytest.raises(explorer.ExplorerError, match="select a frequency"):
    prepare_sources(
      [source],
      MAPPING,
      source_names={},
      calendar=CalendarSettings(weekday=True),
      horizon=2,
    )
  datasets, mapping = prepare_sources(
    [source],
    MAPPING,
    source_names={},
    calendar=CalendarSettings(weekday=True),
    horizon=2,
    frequency="D",
  )
  dates = datasets[0].frame["date"]
  assert pd.Timestamp("2026-01-03") not in set(dates)
  assert list(dates.tail(2)) == list(pd.date_range("2026-01-06", periods=2))
  report = quality_report(datasets, mapping, SETTINGS, frequency="D")
  assert report.iloc[0]["status"] == "warning"
  assert "gaps" in report.iloc[0]["details"]


def test_irregular_history_without_calendar_is_warning_and_supplied_future_dates_work() -> (
  None
):
  source = upload(daily().drop(index=2))
  datasets, mapping = prepare_sources([source], MAPPING, source_names={})
  assert quality_report(datasets, mapping, SETTINGS).iloc[0]["status"] == "warning"
  frame = daily().drop(index=2)
  frame.loc[3:, "value"] = np.nan
  supplied, _ = prepare_sources(
    [upload(frame)],
    MAPPING,
    source_names={},
    calendar=CalendarSettings(weekday=True),
    horizon=2,
  )
  assert len(supplied[0].frame) == len(frame)


def test_explicit_frequency_extends_two_observation_history_and_rejects_nonpositive() -> (
  None
):
  datasets, _ = prepare_sources(
    [upload(daily(2))], MAPPING, source_names={}, frequency="D", horizon=2
  )
  assert list(datasets[0].frame["date"].tail(2)) == list(
    pd.date_range("2026-01-03", periods=2)
  )
  with pytest.raises(explorer.ExplorerError, match="positive frequency"):
    prepare_sources([upload(daily())], MAPPING, source_names={}, frequency="0D")


def test_calendar_does_not_fill_uploaded_future_covariates() -> None:
  source = upload(daily().assign(price=np.arange(5)))
  mapping = dataclasses.replace(MAPPING, past_future=("price",))
  datasets, prepared_mapping = prepare_sources(
    [source],
    mapping,
    source_names={},
    calendar=CalendarSettings(weekday=True),
    horizon=2,
  )
  report = quality_report(datasets, prepared_mapping, SETTINGS)
  assert report.iloc[0]["status"] == "blocked"
  assert "missing known-future" in report.iloc[0]["details"]
  assert datasets[0].frame["price"].tail(2).isna().all()


def test_restoring_saved_calendar_is_idempotent_and_keeps_user_future_rows() -> None:
  original = daily(7)
  original.loc[5:, "value"] = np.nan
  datasets, mapping = prepare_sources(
    [upload(original)],
    MAPPING,
    source_names={"file_1": "sales"},
    calendar=CalendarSettings(month=True),
    horizon=3,
  )
  metadata = datasets[0].frame.attrs["preparation"]
  assert metadata["appended_rows"] == 1
  restored, restored_mapping = restore_preparation(
    datasets[0], mapping, metadata, horizon=1
  )
  assert len(restored.frame) == 7
  assert restored.dataset_id == "sales"
  assert restored_mapping.past_future == ("calendar_month",)
  repeated, repeated_mapping = restore_preparation(
    restored, restored_mapping, metadata, horizon=1
  )
  pd.testing.assert_frame_equal(restored.frame, repeated.frame)
  assert repeated_mapping == restored_mapping
  updated, _ = prepare_sources(
    [upload(daily(10))],
    MAPPING,
    source_names={"file_1": "sales"},
    calendar=CalendarSettings(weekday=True),
    horizon=1,
  )
  replay, replay_mapping = restore_preparation(updated[0], mapping, metadata, horizon=2)
  assert len(replay.frame) == 12
  assert "calendar_weekday" not in replay.frame
  assert replay_mapping.past_future == ("calendar_month",)
  assert replay.frame["value"].iloc[:10].notna().all()


def test_quality_reports_duplicate_timestamps_and_missing_context_without_modification() -> (
  None
):
  duplicates = daily()
  duplicates.loc[2, "date"] = duplicates.loc[1, "date"]
  duplicate_report = quality_report([upload(duplicates)], MAPPING, SETTINGS)
  assert duplicate_report.iloc[0]["duplicate_timestamps"] == 2
  assert duplicate_report.iloc[0]["status"] == "blocked"
  frame = daily()
  frame.loc[2, "value"] = np.nan
  source = upload(frame)
  report = quality_report([source], MAPPING, SETTINGS)
  assert report.iloc[0]["missing_context_values"] == 1
  assert report.iloc[0]["status"] == "warning"
  assert pd.isna(source.frame.loc[2, "value"])


def test_imputation_preview_cannot_use_holdout_target_values() -> None:
  frame = daily(7)
  frame.loc[4, "value"] = np.nan
  frame.loc[5:, "value"] = [1000, 2000]
  source = upload(frame)
  settings = dataclasses.replace(SETTINGS, task="holdout")
  preview = imputation_preview(source, MAPPING, settings)
  assert preview["model_value"].tolist() == [3.0]
  assert preview["timestamp"].tolist() == [pd.Timestamp("2026-01-05")]
  assert pd.isna(source.frame.loc[4, "value"])


def test_generated_features_obey_model_and_decoded_memory_limits(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  frame = daily()
  for number in range(31):
    frame[f"target_{number}"] = np.arange(5)
  mapping = dataclasses.replace(
    MAPPING, targets=tuple(column for column in frame if column != "date")
  )
  datasets, prepared_mapping = prepare_sources(
    [upload(frame)],
    mapping,
    source_names={},
    calendar=CalendarSettings(weekday=True),
    horizon=2,
  )
  assert (
    quality_report(datasets, prepared_mapping, SETTINGS).iloc[0]["status"] == "blocked"
  )
  chunking = dataclasses.replace(SETTINGS, allow_benchmark_chunking=True)
  assert (
    quality_report(datasets, prepared_mapping, chunking).iloc[0]["model_inputs"] == 33
  )
  monkeypatch.setattr(explorer, "MAX_DECODED_BYTES", 1)
  with pytest.raises(explorer.ExplorerError, match="memory limit"):
    prepare_sources([datasets[0]], mapping, source_names={}, horizon=2)


def test_preparation_ui_supports_grouping_and_calendar_without_loading_model() -> None:
  app = AppTest.from_string(
    """
import pandas as pd
import streamlit as st
from timesfm3.explorer import DatasetMapping, parse_upload
from timesfm3.data_preparation_ui import render_preparation
frame = pd.DataFrame({"date": list(pd.date_range("2026-01-01", periods=3)) * 2, "store": ["A"] * 3 + ["B"] * 3, "value": [1, 2, 3, 4, 5, 6]})
source = parse_upload(frame.to_csv(index=False).encode(), "csv", "sales")
datasets, mapping = render_preparation([source], DatasetMapping("date", ("value",)), 2)
st.session_state.prepared_ids = [item.dataset_id for item in datasets]
st.session_state.prepared_covariates = mapping.past_future
""",
    default_timeout=20,
  ).run()
  app.multiselect(key="preparation_group_columns").set_value(["store"]).run()
  assert not app.exception
  assert len(app.session_state["prepared_ids"]) == 2
  app.toggle(key="preparation_calendar_enabled").set_value(True).run()
  app.checkbox(key="preparation_weekday").set_value(True).run()
  assert not app.exception
  assert app.session_state["prepared_covariates"] == ("calendar_weekday",)


def test_preparation_ui_discards_model_roles_from_stale_group_state() -> None:
  app = AppTest.from_string(
    """
import streamlit as st
from timesfm3.data_preparation_ui import render_preparation
from timesfm3.explorer import DatasetMapping, demo_dataset, parse_upload
frame = demo_dataset("multivariate")
source = parse_upload(frame.to_csv(index=False).encode(), "csv", "demo")
st.session_state.preparation_group_columns = ["demand", "promotion", "temperature"]
datasets, _ = render_preparation(
    [source],
    DatasetMapping("date", ("sales", "demand"), ("temperature",), ("promotion",)),
    32,
)
st.session_state.prepared_ids = [item.dataset_id for item in datasets]
""",
    default_timeout=20,
  ).run()

  assert not app.exception
  assert not app.error
  assert app.multiselect(key="preparation_group_columns").value == []
  assert app.session_state["prepared_ids"] == ["demo"]


def test_missing_identifier_group_remains_visible_and_can_be_excluded() -> None:
  source = upload(pd.concat([daily().assign(store="A"), daily().assign(store=None)]))
  groups = group_sources([source], MAPPING, source_names={}, group_columns=("store",))
  report = quality_report(groups, MAPPING, SETTINGS)
  assert set(report["status"]) == {"ready", "blocked"}
  valid_ids = set(report.loc[report["status"] != "blocked", "dataset"])
  prepared, _ = prepare_sources(
    [group for group in groups if group.dataset_id in valid_ids],
    MAPPING,
    source_names={},
  )
  assert len(prepared) == 1
  assert prepared[0].frame.attrs["preparation"]["group_key"] == {"store": "A"}


def test_preparation_ui_can_exclude_invalid_groups_before_calendar_generation() -> None:
  app = AppTest.from_string(
    """
import pandas as pd
import streamlit as st
from timesfm3.explorer import DatasetMapping, parse_upload
from timesfm3.data_preparation_ui import render_preparation
frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-01", "2026-01-03"], "store": [1] * 3 + [2] * 3, "value": [1, 2, 3, 4, 5, 6]})
source = parse_upload(frame.to_csv(index=False).encode(), "csv", "sales")
datasets, mapping = render_preparation([source], DatasetMapping("date", ("value",)), 2)
st.session_state.prepared_ids = [item.dataset_id for item in datasets]
st.session_state.target_names = mapping.targets
""",
    default_timeout=20,
  ).run()
  app.multiselect(key="preparation_group_columns").set_value(["store"]).run()
  quality = next(
    frame.value for frame in app.dataframe if "duplicate_timestamps" in frame.value
  )
  assert set(quality["status"]) == {"warning", "blocked"}
  assert app.session_state["target_names"] == ("value",)
  valid_id = quality.loc[quality["status"] != "blocked", "dataset"].iloc[0]
  app.toggle(key="preparation_all_groups").set_value(False).run()
  app.multiselect(key="preparation_selected_groups").set_value([valid_id]).run()
  app.toggle(key="preparation_calendar_enabled").set_value(True).run()
  app.checkbox(key="preparation_month").set_value(True).run()
  assert not app.exception
  assert not app.error
  assert app.session_state["prepared_ids"] == [valid_id]


def test_quality_warns_when_requested_context_exceeds_available_history() -> None:
  settings = dataclasses.replace(SETTINGS, context_length=8)
  report = quality_report([upload(daily())], MAPPING, settings).iloc[0]
  assert report["context_rows"] == 5
  assert report["status"] == "warning"
  assert "Only 5 context rows are available; 8 were requested" in report["details"]


def test_joint_preview_trims_all_target_missing_prefix_and_aligned_covariates() -> None:
  frame = daily(8)
  frame["value"] = [np.nan, np.nan, 1, np.nan, 3, 4, np.nan, np.nan]
  frame["other"] = [np.nan, np.nan, np.nan, 5, 6, 7, np.nan, np.nan]
  frame["past"] = [1000, 2000, np.nan, 8, np.nan, 12, np.nan, np.nan]
  frame["known"] = [1000, 2000, np.nan, 8, np.nan, 12, 14, 16]
  source = upload(frame)
  mapping = explorer.DatasetMapping("date", ("value", "other"), ("past",), ("known",))
  settings = dataclasses.replace(SETTINGS, context_length=6)
  preview = imputation_preview(source, mapping, settings)
  assert set(preview["timestamp"]) == set(
    pd.to_datetime(["2026-01-03", "2026-01-04", "2026-01-05"])
  )
  values = {
    (row.column, row.timestamp): row.model_value for row in preview.itertuples()
  }
  assert values[("past", pd.Timestamp("2026-01-03"))] == 8
  assert values[("known", pd.Timestamp("2026-01-03"))] == 8
  assert values[("other", pd.Timestamp("2026-01-03"))] == 5
  assert values[("value", pd.Timestamp("2026-01-04"))] == 2
  report = quality_report([source], mapping, settings).iloc[0]
  assert report["context_rows"] == 6
  assert report["leading_trimmed_target_values"] == 4
  assert json.loads(report["leading_trim_by_target"]) == {"value": 2, "other": 2}
  assert report["missing_context_values"] == len(preview) == 6
  assert "drops 2 leading time rows" in report["details"]
  assert source.frame["value"].iloc[:2].isna().all()


def test_independent_preview_trims_each_target_prefix_separately() -> None:
  frame = daily(6)
  frame["value"] = [np.nan, np.nan, 1, np.nan, 3, 4]
  frame["other"] = [np.nan, np.nan, np.nan, 5, 6, 7]
  source = upload(frame)
  mapping = dataclasses.replace(MAPPING, targets=("value", "other"))
  settings = dataclasses.replace(SETTINGS, context_length=6, mode="univariate")
  preview = imputation_preview(source, mapping, settings)
  assert preview["column"].tolist() == ["value"]
  assert preview["timestamp"].tolist() == [pd.Timestamp("2026-01-04")]
  assert preview["model_value"].tolist() == [2]
  report = quality_report([source], mapping, settings).iloc[0]
  assert report["leading_trimmed_target_values"] == 5
  assert json.loads(report["leading_trim_by_target"]) == {"value": 2, "other": 3}
  assert report["missing_context_values"] == 1
  assert (
    "Independent forecasts drop leading missing rows per target" in report["details"]
  )


def test_preview_does_not_claim_exact_values_for_benchmark_chunks() -> None:
  frame = daily()
  for index in range(32):
    frame[f"target_{index}"] = np.arange(5)
  source = upload(frame)
  mapping = dataclasses.replace(
    MAPPING, targets=tuple(column for column in frame if column != "date")
  )
  settings = dataclasses.replace(SETTINGS, allow_benchmark_chunking=True)
  with pytest.raises(
    explorer.ExplorerError, match="unavailable for benchmark chunking"
  ):
    imputation_preview(source, mapping, settings)
  report = quality_report([source], mapping, settings).iloc[0]
  assert report["leading_trimmed_target_values"] is None
  assert report["missing_context_values"] is None
