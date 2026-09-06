# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Session-only grouping, calendar features, and forecast data diagnostics."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from typing import Any, cast

import numpy as np
import pandas as pd

from . import explorer
from .explorer import (
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  PreparedSeries,
  UploadedDataset,
)
from .timesfm3_forecaster import linear_interpolation


@dataclasses.dataclass(frozen=True)
class CalendarEvent:
  """A named, inclusive range of local calendar dates."""

  name: str
  start: dt.date | str
  end: dt.date | str | None = None


@dataclasses.dataclass(frozen=True)
class CalendarSettings:
  """Numeric features whose dates are known before forecasting."""

  weekday: bool = False
  month: bool = False
  holiday_country: str | None = None
  holiday_subdivision: str | None = None
  events: tuple[CalendarEvent, ...] = ()


def _event_dates(event: CalendarEvent) -> tuple[dt.date, dt.date]:
  try:
    start = pd.Timestamp(event.start)
    end = pd.Timestamp(event.end if event.end is not None else event.start)
    if pd.isna(start) or pd.isna(end):
      raise ValueError("Missing date")
  except (TypeError, ValueError, OverflowError) as exc:
    raise ExplorerError(
      f"Event '{event.name}' needs valid start and end dates."
    ) from exc
  if end.date() < start.date():
    raise ExplorerError(f"Event '{event.name}' ends before it starts.")
  return cast(dt.date, start.date()), cast(dt.date, end.date())


def _calendar_columns(calendar: CalendarSettings) -> tuple[str, ...]:
  names = []
  if calendar.weekday:
    names.append("calendar_weekday")
  if calendar.month:
    names.append("calendar_month")
  if calendar.holiday_country:
    names.append("calendar_holiday")
  elif calendar.holiday_subdivision:
    raise ExplorerError("Select a holiday country before a subdivision.")
  for event in calendar.events:
    if not event.name.strip():
      raise ExplorerError("Every custom event needs a name.")
    _event_dates(event)
    names.append(f"calendar_event:{event.name.strip()}")
  if len(names) != len(set(names)):
    raise ExplorerError("Custom event names must be unique.")
  return tuple(names)


def _calendar_metadata(calendar: CalendarSettings) -> dict[str, Any]:
  result = dataclasses.asdict(calendar)
  result["events"] = [
    {"name": event.name.strip(), "start": start.isoformat(), "end": end.isoformat()}
    for event in calendar.events
    for start, end in [_event_dates(event)]
  ]
  return result


def _group_value(value: Any) -> Any:
  if pd.isna(value):
    return None
  if isinstance(value, np.generic):
    value = value.item()
  if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
    return {"date": value.isoformat()}
  if isinstance(value, float):
    if not np.isfinite(value):
      raise ExplorerError("Group identifiers must be finite, non-missing values.")
    return int(value) if value.is_integer() else value
  if isinstance(value, (str, int, bool)):
    return value
  raise ExplorerError("Group identifiers must be text, numbers, or dates.")


def _frequency(axis: pd.DatetimeIndex, requested: str | None) -> str | None:
  if requested:
    try:
      offset = pd.tseries.frequencies.to_offset(requested)
      if offset.n <= 0:
        raise ValueError("Non-positive frequency")
      return offset.freqstr
    except (TypeError, ValueError) as exc:
      raise ExplorerError(
        "Use a positive frequency such as D, B, h, W-MON, or MS."
      ) from exc
  if len(axis) >= 3:
    return pd.infer_freq(axis)
  return None


def _calendar_values(
  axis: pd.DatetimeIndex, calendar: CalendarSettings
) -> dict[str, np.ndarray]:
  """Use the dates displayed on the parsed axis, retaining its timezone."""
  values: dict[str, np.ndarray] = {}
  dates = [cast(pd.Timestamp, value).date() for value in axis]
  if calendar.weekday:
    values["calendar_weekday"] = np.asarray(
      [day.weekday() for day in dates], dtype=np.int8
    )
  if calendar.month:
    values["calendar_month"] = np.asarray([day.month for day in dates], dtype=np.int8)
  if calendar.holiday_country:
    try:
      import holidays

      holiday_dates = holidays.country_holidays(
        calendar.holiday_country,
        subdiv=calendar.holiday_subdivision,
        years=sorted({day.year for day in dates}),
      )
      values["calendar_holiday"] = np.asarray(
        [day in holiday_dates for day in dates], dtype=np.int8
      )
    except ImportError as exc:
      raise ExplorerError(
        "Holiday features require the app's holidays package."
      ) from exc
    except (KeyError, ValueError, NotImplementedError) as exc:
      raise ExplorerError(
        "Select a supported holiday country and subdivision."
      ) from exc
  for event in calendar.events:
    start, end = _event_dates(event)
    values[f"calendar_event:{event.name.strip()}"] = np.asarray(
      [start <= day <= end for day in dates], dtype=np.int8
    )
  return values


def _prepare_dataset(
  dataset: UploadedDataset,
  mapping: DatasetMapping,
  calendar: CalendarSettings,
  horizon: int,
  frequency: str | None,
  metadata: dict[str, Any],
) -> UploadedDataset:
  if any(value is None for value in metadata.get("group_key", {}).values()):
    raise ExplorerError("This group has missing group identifiers.")
  explorer._validate_mapping(dataset.frame, mapping)
  generated = _calendar_columns(calendar)
  conflicts = set(generated).intersection(dataset.frame.columns)
  if conflicts:
    raise ExplorerError(
      "Generated calendar columns already exist: " + ", ".join(sorted(conflicts))
    )
  if generated and mapping.timestamp is None:
    raise ExplorerError("Calendar features require a timestamp column.")
  frame, axis, lineage = explorer._time_axis(dataset.frame, mapping.timestamp)
  frame = frame.copy()
  appended = 0
  resolved_frequency = None
  if mapping.timestamp is not None:
    frame[mapping.timestamp] = axis
    dates = pd.DatetimeIndex(axis)
    resolved_frequency = _frequency(dates, frequency)
    if generated or frequency:
      numeric = explorer._coerce_numeric(frame, mapping.targets)
      observed = np.flatnonzero(numeric.notna().any(axis=1).to_numpy())
      if not len(observed):
        raise ExplorerError(f"{dataset.dataset_id} has no observed target values.")
      appended = max(0, int(observed[-1]) + 1 + horizon - len(frame))
      if appended:
        if resolved_frequency is None:
          raise ExplorerError(
            f"{dataset.dataset_id}: select a frequency to generate future calendar dates."
          )
        try:
          future = pd.date_range(
            dates[-1], periods=appended + 1, freq=resolved_frequency
          )
          # Anchored frequencies may start after the final observation.
          future = future[future > dates[-1]][:appended]
        except (ValueError, OverflowError, pd.errors.OutOfBoundsDatetime) as exc:
          raise ExplorerError(
            "The selected frequency cannot extend these timestamps."
          ) from exc
        estimated = (len(frame) + appended) * (
          max(1, dataset.memory_bytes // max(1, len(frame))) + len(generated) * 8
        )
        if estimated > explorer.MAX_DECODED_BYTES:
          raise ExplorerError(
            f"{dataset.dataset_id} expands beyond the 256 MB memory limit."
          )
        original_length = len(frame)
        frame = frame.reindex(range(original_length + appended))
        frame.loc[original_length:, mapping.timestamp] = future
        for column in metadata.get("group_key", {}):
          frame.loc[original_length:, column] = dataset.frame[column].iloc[0]
        dates = pd.DatetimeIndex(frame[mapping.timestamp])
      for column, values in _calendar_values(dates, calendar).items():
        frame[column] = values
  metadata = {
    **metadata,
    "frequency": resolved_frequency,
    "calendar": _calendar_metadata(calendar),
    "holidays_version": version("holidays") if calendar.holiday_country else None,
    "known_future_policy": "Calendar dates and custom event ranges are assumed known before every forecast origin.",
    "generated_columns": list(generated),
    "appended_rows": appended,
    "sorted": bool(lineage) or metadata.get("sorted", False),
  }
  frame.attrs["preparation"] = metadata
  memory_bytes = int(frame.memory_usage(index=True, deep=True).sum())
  if memory_bytes > explorer.MAX_DECODED_BYTES:
    raise ExplorerError(f"{dataset.dataset_id} expands beyond the 256 MB memory limit.")
  return dataclasses.replace(dataset, frame=frame, memory_bytes=memory_bytes)


def group_sources(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  *,
  source_names: Mapping[str, str],
  group_columns: tuple[str, ...] = (),
) -> list[UploadedDataset]:
  """Split sources before timestamp validation so bad groups remain inspectable.

  Source names and canonical group keys determine identity, independent of upload
  order and content hashes. Split groups account for their source's bytes once.
  Only in-memory copies and serializable preparation metadata are produced.
  """
  explorer.validate_upload_total(datasets)
  if len(group_columns) != len(set(group_columns)):
    raise ExplorerError("Group columns must be unique.")
  assigned = set(mapping.targets + mapping.past_only + mapping.past_future)
  if assigned.intersection(group_columns) or mapping.timestamp in group_columns:
    raise ExplorerError("Group columns cannot also be timestamps or model inputs.")
  names = [
    source_names.get(item.dataset_id, item.dataset_id).strip() for item in datasets
  ]
  if any(not name for name in names) or len(set(names)) != len(names):
    raise ExplorerError("Give each uploaded source a unique, non-empty name.")
  result: list[UploadedDataset] = []
  for dataset, name in zip(datasets, names, strict=True):
    missing = set(group_columns).difference(dataset.frame.columns)
    if missing:
      raise ExplorerError(
        f"{name} is missing group columns: {', '.join(sorted(missing))}"
      )
    columns = tuple(sorted(group_columns))
    groups = (
      dataset.frame.groupby(list(columns), sort=False, observed=True, dropna=False)
      if columns
      else [((), dataset.frame)]
    )
    for index, (key, frame) in enumerate(groups):
      key = key if isinstance(key, tuple) else (key,)
      group_key = {
        column: _group_value(value) for column, value in zip(columns, key, strict=True)
      }
      identifier = name
      if columns:
        identifier += " / " + json.dumps(
          group_key, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
      grouped_frame = frame.reset_index(drop=True)
      grouped_frame.attrs["preparation"] = (
        dict(dataset.frame.attrs["preparation"])
        if not columns and dataset.frame.attrs.get("preparation")
        else {"source_name": name, "group_key": group_key}
      )
      grouped = dataclasses.replace(
        dataset,
        dataset_id=identifier,
        frame=grouped_frame,
        byte_size=dataset.byte_size if index == 0 else 0,
        memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
      )
      result.append(grouped)
  if len({item.dataset_id for item in result}) != len(result):
    raise ExplorerError("Source names and group keys must identify distinct datasets.")
  explorer.validate_upload_total(result)
  return result


def prepare_sources(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  *,
  source_names: Mapping[str, str],
  group_columns: tuple[str, ...] = (),
  calendar: CalendarSettings | None = None,
  horizon: int = 1,
  frequency: str | None = None,
) -> tuple[list[UploadedDataset], DatasetMapping]:
  """Prepare selected groups and calendar values without filling uploaded targets."""
  if not 1 <= horizon <= explorer.MAX_CONTEXT:
    raise ExplorerError(f"Horizon must be between 1 and {explorer.MAX_CONTEXT:,}.")
  calendar = calendar or CalendarSettings()
  generated = _calendar_columns(calendar)
  groups = group_sources(
    datasets, mapping, source_names=source_names, group_columns=group_columns
  )
  prepared = []
  for dataset in groups:
    try:
      prepared.append(
        _prepare_dataset(
          dataset,
          mapping,
          calendar,
          horizon,
          frequency,
          dataset.frame.attrs["preparation"],
        )
      )
    except ExplorerError as exc:
      raise ExplorerError(f"{dataset.dataset_id}: {exc}") from exc
  explorer.validate_upload_total(prepared)
  return prepared, dataclasses.replace(
    mapping, past_future=mapping.past_future + generated
  )


def restore_preparation(
  dataset: UploadedDataset,
  mapping: DatasetMapping,
  metadata: Mapping[str, Any],
  *,
  horizon: int,
) -> tuple[UploadedDataset, DatasetMapping]:
  """Apply saved preparation to an updated group without retaining its raw upload."""
  if not 1 <= horizon <= explorer.MAX_CONTEXT:
    raise ExplorerError(f"Horizon must be between 1 and {explorer.MAX_CONTEXT:,}.")
  current = dataset.frame.attrs.get("preparation", {})
  old_columns = tuple(current.get("generated_columns", ()))
  appended = int(current.get("appended_rows", 0))
  frame = dataset.frame.iloc[
    : len(dataset.frame) - appended if appended else None
  ].copy()
  frame = frame.drop(columns=list(old_columns), errors="ignore")
  frame.attrs = {}
  saved_columns = tuple(metadata.get("generated_columns", ()))
  base_mapping = dataclasses.replace(
    mapping,
    past_future=tuple(
      column
      for column in mapping.past_future
      if column not in old_columns + saved_columns
    ),
  )
  settings = dict(metadata.get("calendar", {}))
  settings["events"] = tuple(
    CalendarEvent(**event) for event in settings.get("events", ())
  )
  calendar = CalendarSettings(**settings)
  prepared = _prepare_dataset(
    dataclasses.replace(
      dataset,
      frame=frame,
      memory_bytes=int(frame.memory_usage(index=True, deep=True).sum()),
    ),
    base_mapping,
    calendar,
    horizon,
    metadata.get("frequency"),
    {
      "source_name": metadata.get("source_name", dataset.dataset_id),
      "group_key": dict(metadata.get("group_key", {})),
    },
  )
  return prepared, dataclasses.replace(
    base_mapping, past_future=base_mapping.past_future + _calendar_columns(calendar)
  )


def _model_columns(
  prepared: PreparedSeries, mapping: DatasetMapping, mode: str
) -> list[tuple[str, np.ndarray, int]]:
  """Mirror leading-NaN trimming before the forecaster interpolates valid inputs."""
  if mode == "univariate":
    offsets = [int(np.argmax(~np.isnan(values))) for values in prepared.context]
  else:
    first_valid = int(np.argmax(~np.isnan(prepared.context).all(axis=0)))
    offsets = [first_valid] * len(mapping.targets)
  columns = [
    (name, values[offset:], offset)
    for name, values, offset in zip(
      mapping.targets, prepared.context, offsets, strict=True
    )
  ]
  for names, arrays in (
    (mapping.past_only, prepared.past_only),
    (mapping.past_future, prepared.past_future),
  ):
    if arrays is not None:
      columns.extend(
        (name, values[offsets[0] :], offsets[0])
        for name, values in zip(names, arrays, strict=True)
      )
  return columns


def quality_report(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  settings: ForecastSettings,
  frequency: str | None = None,
) -> pd.DataFrame:
  """Report readiness and warnings per group; never alter or interpolate data."""
  rows = []
  for dataset in datasets:
    row: dict[str, Any] = {
      "dataset": dataset.dataset_id,
      "status": "ready",
      "rows": len(dataset.frame),
      "observed_rows": 0,
      "context_rows": 0,
      "leading_trimmed_target_values": 0,
      "leading_trim_by_target": "{}",
      "start": None,
      "end": None,
      "frequency": "row number" if mapping.timestamp is None else None,
      "duplicate_timestamps": 0,
      "invalid_timestamps": 0,
      "missing_context_values": 0,
      "constant_targets": "",
      "model_inputs": len(mapping.targets + mapping.past_only + mapping.past_future),
      "details": "",
    }
    warnings: list[str] = []
    try:
      if any(
        value is None
        for value in dataset.frame.attrs.get("preparation", {})
        .get("group_key", {})
        .values()
      ):
        raise ExplorerError("This group has missing group identifiers.")
      explorer._validate_mapping(dataset.frame, mapping)
      if mapping.timestamp:
        parsed = pd.to_datetime(
          dataset.frame[mapping.timestamp], errors="coerce", format="mixed", utc=True
        )
        row["duplicate_timestamps"] = int(parsed.dropna().duplicated(keep=False).sum())
        row["invalid_timestamps"] = int(parsed.isna().sum())
      frame, axis, lineage = explorer._time_axis(dataset.frame, mapping.timestamp)
      if lineage or dataset.frame.attrs.get("preparation", {}).get("sorted"):
        warnings.append("Rows are sorted by timestamp for forecasting.")
      row["start"], row["end"] = str(axis.iloc[0]), str(axis.iloc[-1])
      if mapping.timestamp:
        dates = pd.DatetimeIndex(axis)
        chosen = frequency or dataset.frame.attrs.get("preparation", {}).get(
          "frequency"
        )
        inferred = _frequency(dates, chosen)
        row["frequency"] = inferred or "unknown"
        if inferred is None:
          warnings.append(
            "Cadence is ambiguous or has gaps; select a frequency when generating future dates."
          )
        elif not pd.date_range(dates[0], periods=len(dates), freq=inferred).equals(
          dates
        ):
          warnings.append(
            f"Timestamps have gaps or do not follow {inferred}; no rows are inserted."
          )
      numeric = explorer._coerce_numeric(
        frame, mapping.targets + mapping.past_only + mapping.past_future
      )
      row["observed_rows"] = int(
        numeric.loc[:, mapping.targets].notna().any(axis=1).sum()
      )
      prepared = explorer.prepare_batch([dataset], mapping, settings).series[0]
      context_rows = prepared.context.shape[-1]
      row["context_rows"] = context_rows
      if context_rows < settings.context_length:
        warnings.append(
          f"Only {context_rows} context rows are available; {settings.context_length} were requested."
        )
      chunked = (
        settings.mode == "multivariate" and row["model_inputs"] > explorer.MAX_VARIATES
      )
      if chunked:
        row["leading_trimmed_target_values"] = None
        row["leading_trim_by_target"] = "Determined per benchmark chunk"
        row["missing_context_values"] = None
        warnings.append(
          "Leading trimming and interpolation counts depend on benchmark target chunks and selected covariates."
        )
      else:
        columns = _model_columns(prepared, mapping, settings.mode)
        trimmed = {
          name: offset for name, _, offset in columns[: len(mapping.targets)] if offset
        }
        row["leading_trimmed_target_values"] = sum(trimmed.values())
        row["leading_trim_by_target"] = json.dumps(trimmed, sort_keys=True)
        if trimmed:
          if settings.mode == "univariate":
            warnings.append(
              "Independent forecasts drop leading missing rows per target: "
              + ", ".join(f"{name}: {offset}" for name, offset in trimmed.items())
              + "."
            )
          else:
            warnings.append(
              f"The model drops {next(iter(trimmed.values()))} leading time rows where all targets are missing, including aligned covariate rows."
            )
        missing = sum(
          int(np.isnan(values[: context_rows - offset]).sum())
          for _, values, offset in columns
        )
        row["missing_context_values"] = missing
        if missing:
          warnings.append(
            f"The model interpolates {missing} retained missing context values; uploaded data stays unchanged."
          )
      constant = [
        name
        for name, values in zip(mapping.targets, prepared.context, strict=True)
        if len(np.unique(values[np.isfinite(values)])) == 1
      ]
      row["constant_targets"] = ", ".join(constant)
      if constant:
        warnings.append("Constant targets: " + ", ".join(constant) + ".")
      if prepared.actual is not None and np.isnan(prepared.actual).any():
        warnings.append("Missing holdout actuals remain unscored.")
      if row["model_inputs"] > explorer.MAX_VARIATES:
        warnings.append(
          "Benchmark chunking can subsample covariates above 32 model inputs."
        )
      if warnings:
        row["status"] = "warning"
    except (ExplorerError, ValueError, OverflowError) as exc:
      row["status"] = "blocked"
      warnings.append(str(exc))
    row["details"] = " ".join(warnings)
    rows.append(row)
  return pd.DataFrame(rows)


def imputation_preview(
  dataset: UploadedDataset,
  mapping: DatasetMapping,
  settings: ForecastSettings,
) -> pd.DataFrame:
  """Show missing context cells and model interpolation, excluding holdout targets."""
  prepared = explorer.prepare_batch([dataset], mapping, settings).series[0]
  if (
    settings.mode == "multivariate"
    and len(mapping.targets + mapping.past_only + mapping.past_future)
    > explorer.MAX_VARIATES
  ):
    raise ExplorerError(
      "Exact interpolation preview is unavailable for benchmark chunking; trimming can differ between target chunks."
    )
  rows = []
  for column, original, offset in _model_columns(prepared, mapping, settings.mode):
    filled = linear_interpolation(original)
    history_time = prepared.history_time[offset:]
    for index in np.flatnonzero(np.isnan(original[: len(history_time)])):
      rows.append(
        {
          "timestamp": history_time[index],
          "column": column,
          "uploaded_value": np.nan,
          "model_value": float(filled[index]),
        }
      )
  return pd.DataFrame(
    rows, columns=["timestamp", "column", "uploaded_value", "model_value"]
  )
