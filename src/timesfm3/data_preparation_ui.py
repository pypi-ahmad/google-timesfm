# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Native Streamlit controls for session-only upload preparation.

UI layer over `data_preparation.py`: renders grouping, frequency,
calendar/holiday/event, and group-selection controls, then calls
`group_sources`/`prepare_sources` and previews the result. All state is
Streamlit `session_state` (per this app session) plus the returned
`UploadedDataset`/`DatasetMapping` values; nothing here is written to
disk. `render_preparation` is the entry point -- read it first, then
`data_preparation.py` for what each call actually does. Errors raised by
the data layer (`ExplorerError`) are caught here and shown inline via
`st.error` rather than propagating, so one bad prepare step degrades to
an empty result instead of crashing the page.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import streamlit as st

from .data_preparation import (
  CalendarEvent,
  CalendarSettings,
  group_sources,
  prepare_sources,
  quality_report,
)
from .explorer import (
  MAX_VARIATES,
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  UploadedDataset,
)


def _calendar_controls(has_timestamp: bool) -> CalendarSettings:
  with st.expander("Calendar and events", icon=":material/calendar_month:"):
    enabled = st.toggle(
      "Add known-future calendar features",
      disabled=not has_timestamp,
      key="preparation_calendar_enabled",
    )
    if not has_timestamp:
      st.caption("Select a timestamp column to use calendar features.")
    if not enabled or not has_timestamp:
      return CalendarSettings()
    weekday = st.checkbox("Weekday (Monday = 0, Sunday = 6)", key="preparation_weekday")
    month = st.checkbox("Month (January = 1, December = 12)", key="preparation_month")
    country = None
    subdivision = None
    if st.checkbox("Public holiday flag", key="preparation_holidays"):
      try:
        import holidays
      except ImportError as exc:
        raise ExplorerError(
          "Holiday features require the app's holidays package."
        ) from exc
      countries = holidays.list_supported_countries(include_aliases=False)
      country = st.selectbox(
        "Holiday country (ISO code)",
        sorted(countries),
        index=None,
        placeholder="Select a country",
        key="preparation_country",
      )
      if country is None:
        raise ExplorerError("Select a country for public holidays.")
      subdivision = st.selectbox(
        "Subdivision",
        [None, *countries[country]],
        format_func=lambda value: "National holidays" if value is None else value,
        key="preparation_subdivision",
      )
    event_rows = st.data_editor(
      pd.DataFrame(columns=["name", "start", "end"]),
      num_rows="dynamic",
      hide_index=True,
      key="preparation_events",
    )
    st.caption(
      "Custom events: enter a name and YYYY-MM-DD start date; leave end blank for "
      "one day. Date ranges include both endpoints. Calendar flags use the "
      "displayed timestamp dates; mixed timezone offsets normalize to UTC."
    )
    events = []
    for row in event_rows.to_dict("records"):
      if all(pd.isna(value) or str(value).strip() == "" for value in row.values()):
        continue
      end = row.get("end")
      start = row.get("start")
      events.append(
        CalendarEvent(
          name="" if pd.isna(row.get("name")) else str(row["name"]),
          start="" if pd.isna(start) else str(start),
          end=None if pd.isna(end) or str(end).strip() == "" else end,
        )
      )
    return CalendarSettings(weekday, month, country, subdivision, tuple(events))


def render_preparation(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping,
  horizon: int,
  *,
  source_names: Mapping[str, str] | None = None,
) -> tuple[list[UploadedDataset], DatasetMapping]:
  """Render grouping/calendar controls; the caller reports quality using real settings."""
  if not datasets:
    return [], mapping
  source_names = source_names or {}
  try:
    names = {}
    with st.expander("Source names", icon=":material/edit:"):
      st.caption(
        "Reuse these names when uploading updated observations to match earlier forecasts."
      )
      for item in datasets:
        default = source_names.get(item.dataset_id, item.dataset_id)
        # `default` is embedded in the widget key (not just dataset_id):
        # if the caller's source_names default changes across reruns,
        # Streamlit sees a new key and resets the input to that new
        # default rather than preserving a stale edited value tied to an
        # old default.
        names[item.dataset_id] = st.text_input(
          f"Name for {default}",
          value=default,
          key=f"preparation_source:{item.dataset_id}:{default}",
        )
    common = set(datasets[0].frame.columns)
    for item in datasets[1:]:
      common.intersection_update(item.frame.columns)
    assigned = set(mapping.targets + mapping.past_only + mapping.past_future)
    options = sorted(common - {mapping.timestamp} - assigned)
    # Prune any previously-selected group columns that are no longer
    # valid options (e.g. the user changed the target/mapping columns
    # since the last run) before the multiselect widget below is created
    # with this session_state value, so Streamlit doesn't raise on a
    # stale selection outside its current option list.
    group_key = "preparation_group_columns"
    if group_key in st.session_state:
      st.session_state[group_key] = [
        value for value in st.session_state[group_key] if value in options
      ]
    group_columns = st.multiselect(
      "Group columns (optional)",
      options,
      key=group_key,
      help="For stacked series, choose identifiers such as store or product. Each group is forecast separately; its target columns remain multivariate.",
    )
    frequency = (
      st.text_input(
        "Frequency (optional)",
        placeholder="Examples: D, B, h, W-MON, MS",
        key="preparation_frequency_input",
        disabled=mapping.timestamp is None,
        help="Use a pandas frequency when cadence cannot be inferred. Missing observations are never inserted or filled.",
      ).strip()
      or None
    )
    if mapping.timestamp is None:
      frequency = None
    st.session_state.preparation_frequency = frequency
    grouped = group_sources(
      datasets,
      mapping,
      source_names=names,
      group_columns=tuple(group_columns),
    )
    # Preview only: uses default ForecastSettings (mode, etc.) since the
    # real forecast settings aren't chosen yet at this point in the UI
    # flow -- the caller re-runs quality_report with actual settings
    # later (see this function's docstring).
    initial_quality = quality_report(
      grouped, mapping, ForecastSettings(horizon=horizon), frequency
    )
    initial_blocked = initial_quality.status.eq("blocked").any()
    with st.expander("Upload checks", expanded=initial_blocked):
      st.caption("Forecast settings determine final readiness.")
      st.dataframe(
        initial_quality,
        hide_index=True,
        key="preparation_group_quality",
      )
    identifiers = [item.dataset_id for item in grouped]
    if len(grouped) > 1 and not st.toggle(
      "Use all groups", value=True, key="preparation_all_groups"
    ):
      # Same stale-selection pruning as group_key above, keyed to the
      # current set of group identifiers.
      selection_key = "preparation_selected_groups"
      if selection_key in st.session_state:
        st.session_state[selection_key] = [
          value for value in st.session_state[selection_key] if value in identifiers
        ]
      selected = st.multiselect("Groups to use", identifiers, key=selection_key)
      grouped = [item for item in grouped if item.dataset_id in selected]
    if not grouped:
      st.info("Select at least one group to forecast.")
      return [], mapping
    calendar = _calendar_controls(mapping.timestamp is not None)
    prepared, prepared_mapping = prepare_sources(
      grouped,
      mapping,
      source_names={},
      calendar=calendar,
      horizon=horizon,
      frequency=frequency,
    )
    variates = len(
      prepared_mapping.targets
      + prepared_mapping.past_only
      + prepared_mapping.past_future
    )
    st.caption(
      f"{len(prepared):,} group(s) · {variates} / {MAX_VARIATES} model inputs per group"
    )
    if variates > MAX_VARIATES:
      st.warning(
        "More than 32 inputs requires benchmark chunking in Configure; some covariates may be omitted."
      )
    with st.expander("Prepared data preview", icon=":material/table_chart:"):
      selected_id = st.selectbox(
        "Preview group",
        [item.dataset_id for item in prepared],
        key="preparation_preview_group",
      )
      preview = next(item for item in prepared if item.dataset_id == selected_id)
      st.dataframe(
        preview.frame.head(200), hide_index=True, key="preparation_data_preview"
      )
      generated = preview.frame.attrs["preparation"]["generated_columns"]
      if generated:
        st.caption("Generated calendar values at the end of the prepared timeline")
        st.dataframe(
          preview.frame.loc[:, [prepared_mapping.timestamp, *generated]].tail(
            min(horizon, 12)
          ),
          hide_index=True,
          key="preparation_calendar_preview",
        )
    return prepared, prepared_mapping
  except ExplorerError as exc:
    st.error(str(exc), icon=":material/error:")
    return [], mapping
