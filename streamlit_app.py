# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Interactive local explorer for TimesFM-3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import altair as alt
import pandas as pd
import streamlit as st
import torch

from timesfm3.explorer import (
  CHECKPOINT_ID,
  MAX_CONTEXT,
  MAX_DECODED_BYTES,
  MAX_TOTAL_UPLOAD_BYTES,
  MAX_VARIATES,
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  RunArtifact,
  UploadedDataset,
  artifact_zip,
  capability_report,
  demo_dataset,
  execute_forecast,
  load_forecaster,
  parse_upload,
  repository_revision,
  validate_upload_total,
)
from timesfm3.run_store import MAX_SAVED_RUNS, RunStoreError, load_recent_runs, save_run

DATABASE_PATH = Path(__file__).parent / "data" / "timesfm.duckdb"

st.set_page_config(
  page_title="TimesFM-3 explorer",
  page_icon=":material/query_stats:",
  layout="wide",
)


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_forecaster(device: str, batch_size: int):
  """Keep one heavyweight model instance in process memory."""
  return load_forecaster(device, batch_size)


def _initialize_state() -> None:
  if "runs" not in st.session_state:
    try:
      st.session_state.runs = load_recent_runs(DATABASE_PATH)
    except RunStoreError as exc:
      st.session_state.runs = []
      st.session_state.persistence_warning = str(exc)
  st.session_state.setdefault("upload_cache", {})
  st.session_state.setdefault("persistence_warning", None)


def _parse_in_session(data: bytes, suffix: str, dataset_id: str) -> UploadedDataset:
  """Cache decoded uploads only for the current browser session."""
  key = (hashlib.sha256(data).hexdigest(), suffix.lower(), dataset_id)
  cached = st.session_state.upload_cache.get(key)
  if cached is None:
    cached = parse_upload(data, suffix, dataset_id)
  st.session_state.next_upload_cache[key] = cached
  return cached


def _numeric_candidates(frame: pd.DataFrame, timestamp: str | None) -> list[str]:
  candidates = []
  for column in frame.columns:
    if column == timestamp:
      continue
    converted = pd.to_numeric(frame[column], errors="coerce")
    if converted.notna().any():
      candidates.append(str(column))
  return candidates


def _timestamp_default(columns: list[str]) -> str | None:
  for column in columns:
    if any(token in column.lower() for token in ("date", "time", "timestamp")):
      return column
  return None


def _append_run(run: RunArtifact) -> None:
  runs: list[RunArtifact] = st.session_state.runs
  st.session_state.runs = [*runs, run][-MAX_SAVED_RUNS:]
  try:
    save_run(DATABASE_PATH, run)
    st.session_state.persistence_warning = None
  except RunStoreError as exc:
    st.session_state.persistence_warning = str(exc)


def _series_chart(run: RunArtifact, dataset: str, target: str) -> Any:
  history = run.history.query("dataset == @dataset and target == @target").copy()
  forecast = run.forecast.query("dataset == @dataset and target == @target").copy()
  temporal_source = history if len(history) else forecast
  temporal = bool(len(temporal_source)) and isinstance(
    temporal_source.iloc[0]["timestamp"], pd.Timestamp
  )
  x_type = "temporal" if temporal else "quantitative"
  x_encoding = alt.X("timestamp", type=x_type, title="Time")
  forecast_chart: Any = alt.Chart(forecast)
  point_line = forecast_chart.mark_line(color="#e85d04", strokeWidth=3).encode(
    x=x_encoding,
    y=alt.Y("point:Q", title=target),
    tooltip=["timestamp", "point"],
  )
  layers: list[Any] = []
  if len(history):
    history_chart: Any = alt.Chart(history)
    layers.append(
      history_chart.mark_line(color="#6b7280").encode(
        x=x_encoding,
        y=alt.Y("value:Q", title=target),
        tooltip=["timestamp", "value"],
      )
    )
  if "q0.1" in forecast and "q0.9" in forecast:
    band = forecast_chart.mark_area(color="#f48c06", opacity=0.18).encode(
      x=x_encoding, y="q0.1:Q", y2="q0.9:Q"
    )
    layers.append(band)
  layers.append(point_line)
  if "actual" in forecast:
    actual_line = forecast_chart.mark_line(color="#0077b6", strokeDash=[5, 3]).encode(
      x=x_encoding, y="actual:Q", tooltip=["timestamp", "actual"]
    )
    layers.append(actual_line)
  return alt.layer(*layers).properties(height=420).interactive()


def _render_run(run: RunArtifact) -> None:
  datasets = list(run.forecast["dataset"].drop_duplicates())
  dataset = st.selectbox("Dataset", datasets, key=f"result_dataset_{run.run_id}")
  targets = list(
    run.forecast.loc[run.forecast["dataset"] == dataset, "target"].drop_duplicates()
  )
  target = st.selectbox("Target", targets, key=f"result_target_{run.run_id}")
  with st.container(horizontal=True):
    st.metric("Run", run.run_id, border=True)
    st.metric("Runtime", f"{run.runtime_seconds:.2f} s", border=True)
    st.metric("Device", run.device, border=True)
    st.metric("Rows", f"{len(run.forecast):,}", border=True)
  with st.container(border=True):
    st.altair_chart(_series_chart(run, dataset, target))
  if not run.metrics.empty:
    with st.container(border=True):
      st.subheader("Holdout metrics", icon=":material/analytics:")
      st.dataframe(run.metrics, hide_index=True, key=f"metrics_{run.run_id}")
  with st.container(border=True):
    st.subheader("Forecast data", icon=":material/table_chart:")
    st.dataframe(run.forecast, hide_index=True, key=f"forecast_{run.run_id}")
  st.download_button(
    "Download result bundle",
    data=artifact_zip(run),
    file_name=f"timesfm3-{run.run_id}.zip",
    mime="application/zip",
    icon=":material/download:",
    key=f"download_{run.run_id}",
  )


_initialize_state()
capabilities = capability_report()

st.title("TimesFM-3 explorer", icon=":material/query_stats:")
st.caption(
  "Local zero-shot univariate and multivariate forecasting with covariates and "
  "probabilistic outputs."
)
if st.session_state.persistence_warning:
  st.warning(
    f"{st.session_state.persistence_warning} Runs will remain in this browser session.",
    icon=":material/database_off:",
  )

with st.sidebar:
  st.subheader("Runtime", icon=":material/memory:")
  st.badge(
    "CUDA ready" if capabilities.cuda_available else "CPU only",
    color="green" if capabilities.cuda_available else "orange",
  )
  st.caption(f"Python {capabilities.python} · Torch {capabilities.torch}")
  st.caption(capabilities.device)
  if capabilities.vram_free_gb is not None:
    st.caption(
      f"VRAM {capabilities.vram_free_gb:.1f} GB free / "
      f"{capabilities.vram_total_gb:.1f} GB total"
    )
  st.caption(
    "Checkpoint cached"
    if capabilities.checkpoint_cached
    else "Checkpoint downloads on first run"
  )
  st.caption(
    "Hugging Face authentication available"
    if capabilities.hf_token_present
    else "Public Hugging Face access"
  )
  if st.button("Unload model", icon=":material/delete:"):
    cached_forecaster.clear()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    st.toast("Model cache cleared")

data_tab, configure_tab, results_tab, compare_tab, about_tab = st.tabs(
  ["Data", "Configure", "Results", "Compare", "About"]
)

datasets: list[UploadedDataset] = []
mapping: DatasetMapping | None = None

with data_tab:
  st.subheader("Choose data", icon=":material/upload_file:")
  source = st.segmented_control(
    "Data source",
    ["Upload", "Demo"],
    default="Demo",
    key="data_source",
  )
  try:
    st.session_state.next_upload_cache = {}
    if source == "Upload":
      files = st.file_uploader(
        "Upload wide CSV or Parquet files",
        type=["csv", "parquet", "pq"],
        accept_multiple_files=True,
        max_upload_size=50,
        key="uploads",
        help="Each file is one batch item. Mapped columns must match across files.",
      )
      if sum(uploaded.size for uploaded in files or []) > MAX_TOTAL_UPLOAD_BYTES:
        raise ExplorerError("Combined uploads must be 200 MB or smaller.")
      for index, uploaded in enumerate(files or [], start=1):
        suffix = Path(uploaded.name).suffix
        datasets.append(
          _parse_in_session(uploaded.getvalue(), suffix, f"dataset_{index}")
        )
    else:
      demo_kind = st.segmented_control(
        "Demo",
        ["Univariate", "Multivariate + covariates"],
        default="Multivariate + covariates",
        key="demo_kind",
      )
      demo = demo_dataset("univariate" if demo_kind == "Univariate" else "multivariate")
      demo_bytes = demo.to_csv(index=False).encode()
      datasets.append(_parse_in_session(demo_bytes, "csv", "demo"))
    validate_upload_total(datasets)
    st.session_state.upload_cache = st.session_state.next_upload_cache
  except ExplorerError as exc:
    datasets = []
    st.session_state.upload_cache = {}
    st.error(str(exc), icon=":material/error:")

  if datasets:
    common_columns = set(map(str, datasets[0].frame.columns))
    for item in datasets[1:]:
      common_columns.intersection_update(map(str, item.frame.columns))
    columns = [
      str(column)
      for column in datasets[0].frame.columns
      if str(column) in common_columns
    ]
    default_timestamp = _timestamp_default(columns)
    timestamp_options: list[str | None] = [None, *columns]
    timestamp = st.selectbox(
      "Timestamp column",
      timestamp_options,
      index=timestamp_options.index(default_timestamp),
      format_func=lambda value: "Use row number" if value is None else value,
      key="timestamp_column",
    )
    numeric = _numeric_candidates(datasets[0].frame, timestamp)
    demo_multivariate = (
      source == "Demo" and st.session_state.demo_kind == "Multivariate + covariates"
    )
    default_targets = ["sales", "demand"] if demo_multivariate else numeric[:1]
    targets = tuple(
      st.multiselect(
        "Target columns",
        numeric,
        default=[column for column in default_targets if column in numeric],
        key="target_columns",
      )
    )
    remaining = [column for column in numeric if column not in targets]
    default_po = ["temperature"] if demo_multivariate else []
    past_only = tuple(
      st.multiselect(
        "Past-only covariates",
        remaining,
        default=[column for column in default_po if column in remaining],
        key="past_only_columns",
      )
    )
    remaining = [column for column in remaining if column not in past_only]
    default_pf = ["promotion"] if demo_multivariate else []
    past_future = tuple(
      st.multiselect(
        "Past-and-future covariates",
        remaining,
        default=[column for column in default_pf if column in remaining],
        key="past_future_columns",
      )
    )
    mapping = DatasetMapping(timestamp, targets, past_only, past_future)
    st.caption(
      f"{len(datasets)} dataset(s) · {len(datasets[0].frame):,} rows in first dataset · "
      f"{len(targets) + len(past_only) + len(past_future)} model variates"
    )
    st.dataframe(datasets[0].frame.head(200), hide_index=True, key="data_preview")
  else:
    st.info("Upload data or select a demo to configure a forecast.")

with configure_tab:
  st.subheader("Configure forecast", icon=":material/tune:")
  if not datasets or mapping is None:
    st.info("Choose data first.")
  else:
    acknowledged = st.checkbox(
      "I understand the default TimesFM-3 weights are restricted to "
      "non-commercial, non-production use.",
      key="license_acknowledged",
    )
    with st.form("forecast_settings"):
      task = st.segmented_control(
        "Task",
        ["Forecast future", "Evaluate holdout"],
        default="Forecast future",
      )
      mode = st.segmented_control(
        "Series mode",
        ["Joint multivariate", "Independent univariate"],
        default="Joint multivariate",
      )
      with st.container(horizontal=True):
        horizon = st.number_input(
          "Horizon", min_value=1, max_value=MAX_CONTEXT, value=32
        )
        context_length = st.number_input(
          "Context length", min_value=1, max_value=MAX_CONTEXT, value=512
        )
        device_options = ["cuda", "cpu"] if capabilities.cuda_available else ["cpu"]
        device = st.selectbox("Device", device_options)
        batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=4)

      st.markdown("**Probabilistic inference**")
      with st.container(horizontal=True):
        return_quantiles = st.checkbox("Return quantiles", value=True)
        symmetric = st.checkbox("Symmetric averaging", value=True)
        positive = st.checkbox("Clamp nonnegative series", value=True)
        sort_quantiles = st.checkbox("Sort quantiles", value=True)

      with st.expander("Advanced options", icon=":material/settings:"):
        use_znorm = st.checkbox("External z-normalization", value=False)
        padding_mode = st.selectbox("Known-future padding", ["none", "edge"])
        variate_count = len(mapping.targets + mapping.past_only + mapping.past_future)
        allow_chunking = st.checkbox(
          "Allow benchmark chunking above 32 variates",
          value=False,
          disabled=variate_count <= MAX_VARIATES or mode == "Independent univariate",
          help="Evaluator uses fixed seed 42 and may subsample covariates.",
        )
        if mode == "Independent univariate" and (
          mapping.past_only or mapping.past_future
        ):
          st.warning(
            "Independent mode ignores covariates; remove their roles before running."
          )
        if horizon > 1024:
          st.warning("Large horizons can exhaust GPU memory. Start with batch size 1.")

      submitted = st.form_submit_button(
        "Run forecast",
        type="primary",
        icon=":material/play_arrow:",
        disabled=not acknowledged,
      )

    if submitted:
      settings = ForecastSettings(
        horizon=int(horizon),
        context_length=int(context_length),
        task="holdout" if task == "Evaluate holdout" else "forecast",
        mode="univariate" if mode == "Independent univariate" else "multivariate",
        return_quantiles=return_quantiles,
        use_symmetric_averaging=symmetric,
        make_positive=positive,
        sort_quantiles=sort_quantiles,
        use_znorm=use_znorm,
        padding_mode=cast(Literal["none", "edge"], padding_mode),
        batch_size=int(batch_size),
        allow_benchmark_chunking=allow_chunking,
      )
      try:
        with st.status("Running TimesFM-3", expanded=True) as status:
          st.write("Loading checkpoint")
          predictor = cached_forecaster(device, settings.batch_size)
          st.write("Forecasting")
          artifact = execute_forecast(
            predictor,
            datasets,
            mapping,
            settings,
            repository_revision(Path(__file__).parent),
          )
          _append_run(artifact)
          status.update(label="Forecast complete", state="complete", expanded=False)
        st.success(f"Run {artifact.run_id} complete.", icon=":material/check_circle:")
      except ExplorerError as exc:
        st.error(str(exc), icon=":material/error:")
      except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error(
          "CUDA ran out of memory. Reduce context, horizon, variates, or batch size; "
          "CPU retry is manual.",
          icon=":material/memory:",
        )
      except Exception as exc:  # noqa: BLE001 - model/download boundary must stay in UI.
        st.error(
          f"Forecast failed ({type(exc).__name__}). Check checkpoint access and "
          "runtime configuration.",
          icon=":material/error:",
        )

with results_tab:
  st.subheader("Latest result", icon=":material/monitoring:")
  runs: list[RunArtifact] = st.session_state.runs
  if runs:
    _render_run(runs[-1])
  else:
    st.info("Run a forecast to see results.")

with compare_tab:
  st.subheader("Compare runs", icon=":material/compare_arrows:")
  runs = st.session_state.runs
  if len(runs) < 2:
    st.info("Complete at least two runs. Up to 25 are saved locally.")
  else:
    labels = {run.run_id: run for run in runs}
    selected = st.multiselect(
      "Runs", list(labels), default=list(labels), max_selections=3, key="compare_runs"
    )
    summaries = []
    comparison_frames = []
    for run_id in selected:
      run = labels[run_id]
      summaries.append(
        {
          "run": run_id,
          "task": run.settings.task,
          "mode": run.settings.mode,
          "horizon": run.settings.horizon,
          "context": run.settings.context_length,
          "runtime_seconds": run.runtime_seconds,
        }
      )
      current = run.forecast.copy()
      current["run"] = run_id
      comparison_frames.append(current)
    if summaries:
      st.dataframe(pd.DataFrame(summaries), hide_index=True, key="compare_summary")
    if comparison_frames:
      combined = pd.concat(comparison_frames, ignore_index=True)
      dataset = st.selectbox("Comparison dataset", combined["dataset"].unique())
      target = st.selectbox(
        "Comparison target",
        combined.loc[combined["dataset"] == dataset, "target"].unique(),
      )
      selected_data = combined.query("dataset == @dataset and target == @target")
      temporal = isinstance(selected_data.iloc[0]["timestamp"], pd.Timestamp)
      comparison_chart: Any = alt.Chart(selected_data)
      chart = (
        comparison_chart.mark_line(strokeWidth=2)
        .encode(
          x=alt.X("timestamp", type="temporal" if temporal else "quantitative"),
          y=alt.Y("point:Q", title=target),
          color=alt.Color("run:N", title="Run"),
          tooltip=["run", "timestamp", "point"],
        )
        .properties(height=420)
        .interactive()
      )
      st.altair_chart(chart)

with about_tab:
  st.subheader("Capabilities and limits", icon=":material/info:")
  st.table(
    {
      "Checkpoint": CHECKPOINT_ID,
      "Inputs": "Univariate, multivariate, past-only and known-future covariates",
      "Outputs": "Median point forecast and q0.1–q0.9 quantiles",
      "Maximum context": f"{MAX_CONTEXT:,} steps",
      "Model variates": f"{MAX_VARIATES} per forward pass; evaluator can chunk targets",
      "Data handling": (
        f"Uploads stay in browser-session memory; {MAX_DECODED_BYTES // 1024**2} MB "
        "decoded limit per file"
      ),
      "Run history": "Newest 25 derived runs in data/timesfm.duckdb",
    },
    border="horizontal",
    width="content",
  )
  st.warning(
    "Repository code is Apache-2.0. Default TimesFM-3 weights use the separate "
    "TimesFM Non-Commercial License v1.0 and are not permitted for commercial "
    "or production use. Forecasts require human validation.",
    icon=":material/gavel:",
  )
  st.markdown(
    "[TimesFM repository](https://github.com/google-research/timesfm) · "
    "[TimesFM-3 announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · "
    "[Checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch)"
  )
