# Copyright 2026 Ahmad Mujtaba
# Licensed under the Apache License, Version 2.0 (the "License");

"""Interactive local explorer for TimesFM-3."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import altair as alt
import pandas as pd
import streamlit as st
import torch

from timesfm3.analysis_ui import render_analysis
from timesfm3.data_preparation import (
  imputation_preview,
  quality_report,
  restore_preparation,
)
from timesfm3.data_preparation_ui import render_preparation
from timesfm3.explorer import (
  CHECKPOINT_ID,
  MAX_CONTEXT,
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
  prepare_batch,
  repository_revision,
  validate_upload_total,
)
from timesfm3.model_loading import (
  ModelSelection,
  ResolvedModel,
  load_resolved_model,
  resolve_model,
  selection_from_provenance,
)
from timesfm3.run_store import MAX_SAVED_RUNS, RunStoreError, load_recent_runs, save_run
from timesfm3.tracking_ui import render_tracking
from timesfm3.uncertainty import calibration_table, interval_bands

DATABASE_PATH = Path(__file__).parent / "data" / "timesfm.duckdb"

st.set_page_config(
  page_title="TimesFM-3 explorer",
  page_icon=":material/query_stats:",
  layout="wide",
)


@st.cache_resource(max_entries=1, show_spinner=False)
def cached_forecaster(
  device: str, batch_size: int, resolved: ResolvedModel | None = None
):
  """Keep one heavyweight model instance in process memory."""
  if resolved is None:
    return load_forecaster(device, batch_size)
  return load_resolved_model(resolved, device, batch_size)


def _acquire_forecaster(
  device: str,
  batch_size: int,
  selection: ModelSelection,
  expected_files: dict[str, str] | None = None,
):
  resolved = resolve_model(selection)
  if expected_files is not None and expected_files != dict(resolved.fingerprints):
    raise ExplorerError(
      "The previous local checkpoint changed. Restore its original files before refreshing."
    )
  cache_identity = (resolved, device, batch_size)
  if st.session_state.get("active_model") != cache_identity:
    cached_forecaster.clear()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  predictor = cached_forecaster(device, batch_size, resolved)
  st.session_state.active_model = cache_identity
  return predictor


def _selected_forecaster(device: str, batch_size: int):
  return _acquire_forecaster(device, batch_size, model_selection)


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
  bands = interval_bands(forecast)
  if not bands.empty:
    for coverage in [80, 60, 40, 20]:
      current = bands.loc[
        bands.nominal_coverage_percent.eq(coverage) & bands.valid_bounds
      ]
      band_chart: Any = alt.Chart(current)
      layers.append(
        band_chart.mark_area(color="#f48c06", opacity=0.16).encode(
          x=x_encoding,
          y="lower:Q",
          y2="upper:Q",
          tooltip=["timestamp", "nominal_coverage_percent", "lower", "upper"],
        )
      )
  layers.append(point_line)
  if "actual" in forecast:
    actual_line = forecast_chart.mark_line(color="#0077b6", strokeDash=[5, 3]).encode(
      x=x_encoding, y="actual:Q", tooltip=["timestamp", "actual"]
    )
    layers.append(actual_line)
  return alt.layer(*layers).properties(height=420).interactive()


def _render_run(run: RunArtifact) -> None:
  summary = (
    run.forecast.groupby(["dataset", "target"], sort=False)
    .agg(forecast_rows=("point", "size"), mean_forecast=("point", "mean"))
    .reset_index()
  )
  if not run.metrics.empty:
    summary = summary.merge(run.metrics, on=["dataset", "target"], how="left")
  datasets = list(run.forecast["dataset"].drop_duplicates())
  series_view, batch_view = st.tabs(["Selected series", "Batch summary"])
  with series_view:
    dataset_column, target_column = st.columns(2)
    with dataset_column:
      dataset = st.selectbox("Dataset", datasets, key=f"result_dataset_{run.run_id}")
    targets = list(
      run.forecast.loc[run.forecast["dataset"] == dataset, "target"].drop_duplicates()
    )
    with target_column:
      target = st.selectbox("Target", targets, key=f"result_target_{run.run_id}")
    with st.container(horizontal=True):
      st.metric("Run", run.run_id, border=True)
      st.metric("Runtime", f"{run.runtime_seconds:.2f} s", border=True)
      st.metric("Device", run.device, border=True)
      st.metric("Rows", f"{len(run.forecast):,}", border=True)
    selected_forecast = run.forecast.loc[
      run.forecast.dataset.eq(dataset) & run.forecast.target.eq(target)
    ]
    with st.container(border=True):
      st.altair_chart(_series_chart(run, dataset, target))
      st.caption("Central 20%, 40%, 60%, and 80% prediction intervals.")
      bands = interval_bands(selected_forecast)
      if not bands.empty and (~bands.valid_bounds).any():
        st.warning("Invalid interval bounds are omitted from the chart.")
    coverage = calibration_table(selected_forecast)
    if not coverage.empty:
      with st.expander("Interval calibration"):
        st.dataframe(
          coverage.loc[coverage.dataset.eq(dataset) & coverage.target.eq(target)],
          hide_index=True,
        )
    with st.expander("Forecast data", icon=":material/table_chart:"):
      st.dataframe(
        selected_forecast,
        hide_index=True,
        key=f"forecast_{run.run_id}_{dataset}_{target}",
      )
  with batch_view:
    st.dataframe(summary, hide_index=True, key=f"batch_summary_{run.run_id}")
    if not run.metrics.empty:
      with st.expander("Holdout metrics", icon=":material/analytics:"):
        st.dataframe(run.metrics, hide_index=True, key=f"metrics_{run.run_id}")
  st.download_button(
    "Download result bundle",
    data=artifact_zip(run),
    file_name=f"timesfm3-{run.run_id}.zip",
    mime="application/zip",
    icon=":material/download:",
    key=f"download_{run.run_id}",
  )


def _render_comparison(runs: Sequence[RunArtifact]) -> None:
  if len(runs) < 2:
    st.info("Complete at least two runs to compare them.")
    return
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
  if not comparison_frames:
    return
  combined = pd.concat(comparison_frames, ignore_index=True)
  selector_columns = st.columns(2)
  with selector_columns[0]:
    dataset = st.selectbox("Comparison dataset", combined["dataset"].unique())
  with selector_columns[1]:
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


_initialize_state()
capabilities = capability_report()

st.title("TimesFM-3 explorer", icon=":material/query_stats:")
st.caption(
  "Forecast time series locally with multivariate inputs, covariates, and uncertainty."
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
  st.caption(capabilities.device)
  with st.expander("Runtime details"):
    st.caption(f"Python {capabilities.python} · Torch {capabilities.torch}")
    if capabilities.vram_free_gb is not None:
      st.caption(
        f"VRAM {capabilities.vram_free_gb:.1f} GB free / "
        f"{capabilities.vram_total_gb:.1f} GB total"
      )
  with st.expander("Model checkpoint", icon=":material/deployed_code:"):
    checkpoint_kind = st.selectbox(
      "Checkpoint source", ["Hugging Face", "Local checkpoint"]
    )
    if checkpoint_kind == "Hugging Face":
      checkpoint_source = st.text_input("Model repository", value=CHECKPOINT_ID)
      checkpoint_revision = st.text_input(
        "Model revision",
        help="Commit SHA, tag, or branch; the resolved commit is recorded.",
      )
    else:
      checkpoint_source = st.text_input("Local model folder or checkpoint file")
      checkpoint_revision = ""
    checkpoint_offline = st.checkbox("Offline loading", value=False)
    st.caption("Access and compatibility are checked when inference starts.")
  model_selection = ModelSelection(
    checkpoint_source,
    "hub" if checkpoint_kind == "Hugging Face" else "local",
    checkpoint_revision.strip() or None,
    checkpoint_offline,
  )
  st.caption(
    model_selection.source
    + (" · authenticated" if capabilities.hf_token_present else " · public access")
  )
  if st.button("Clear model from memory", icon=":material/memory:"):
    cached_forecaster.clear()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    st.toast("Model cache cleared")
  with st.expander("About and limits", icon=":material/info:"):
    st.markdown(
      f"""
- **Inputs:** univariate, multivariate, and covariates
- **Outputs:** median forecast and q0.1–q0.9 quantiles
- **Maximum context:** {MAX_CONTEXT:,} steps
- **Model inputs:** {MAX_VARIATES} per forward pass
- **Run history:** 25 untracked runs; tracked runs are retained
"""
    )
    st.warning(
      "Default TimesFM-3 weights allow non-commercial, non-production use only. "
      "Forecasts require human validation.",
      icon=":material/gavel:",
    )
    st.markdown(
      "[Repository](https://github.com/google-research/timesfm) · "
      "[TimesFM-3 announcement](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/) · "
      "[Checkpoint](https://huggingface.co/google/timesfm-3.0-pytorch)"
    )

(
  prepare_tab,
  forecast_tab,
  evaluate_tab,
  track_tab,
) = st.tabs(["Prepare", "Forecast", "Evaluate", "Track"])

datasets: list[UploadedDataset] = []
source_names: dict[str, str] = {}
mapping: DatasetMapping | None = None
settings: ForecastSettings | None = None
device = "cpu"
acknowledged = False

with prepare_tab:
  st.subheader("Prepare data", icon=":material/upload_file:")
  st.caption("Choose a source, assign series roles, and review forecast readiness.")
  st.markdown("#### 1. Data source")
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
        "Upload CSV or Parquet files",
        type=["csv", "parquet", "pq"],
        accept_multiple_files=True,
        max_upload_size=50,
        key="uploads",
        help="Each file is one batch item. Mapped columns must match across files.",
      )
      if sum(uploaded.size for uploaded in files or []) > MAX_TOTAL_UPLOAD_BYTES:
        raise ExplorerError("Combined uploads must be 200 MB or smaller.")
      for index, uploaded in enumerate(files or [], start=1):
        source_names[f"dataset_{index}"] = Path(uploaded.name).stem
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
    st.markdown("#### 2. Series mapping")
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
    for role in ("target_columns", "past_only_columns", "past_future_columns"):
      if role in st.session_state:
        old = st.session_state[role]
        retained = [column for column in old if column in numeric]
        if retained != old:
          st.session_state[role] = retained or (
            default_targets if role == "target_columns" else []
          )
    if "target_columns" not in st.session_state:
      st.session_state.target_columns = [
        column for column in default_targets if column in numeric
      ]
    targets = tuple(
      st.multiselect(
        "Target columns",
        numeric,
        key="target_columns",
      )
    )
    remaining = [column for column in numeric if column not in targets]
    default_po = ["temperature"] if demo_multivariate else []
    if "past_only_columns" not in st.session_state:
      st.session_state.past_only_columns = [
        column for column in default_po if column in remaining
      ]
    past_only = tuple(
      st.multiselect(
        "Past-only covariates",
        remaining,
        key="past_only_columns",
      )
    )
    remaining = [column for column in remaining if column not in past_only]
    default_pf = ["promotion"] if demo_multivariate else []
    if "past_future_columns" not in st.session_state:
      st.session_state.past_future_columns = [
        column for column in default_pf if column in remaining
      ]
    past_future = tuple(
      st.multiselect(
        "Past-and-future covariates",
        remaining,
        key="past_future_columns",
      )
    )
    mapping = DatasetMapping(timestamp, targets, past_only, past_future)
    st.markdown("#### Optional preparation")
    datasets, mapping = render_preparation(
      datasets,
      mapping,
      int(st.session_state.get("forecast_horizon", 32)),
      source_names=source_names,
    )
    if datasets:
      st.caption(
        f"{len(datasets)} dataset(s) · "
        f"{len(targets) + len(past_only) + len(past_future)} model inputs"
      )
  else:
    st.info("Upload data or select a demo to prepare a forecast.")

with forecast_tab:
  st.subheader("Build forecast", icon=":material/tune:")
  st.caption(
    "Choose the forecast shape, then run TimesFM-3 or save the settings for analysis."
  )
  if not datasets or mapping is None:
    st.info("Prepare data before configuring a forecast.")
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
          "Horizon",
          min_value=1,
          max_value=MAX_CONTEXT,
          value=32,
          key="forecast_horizon",
        )
        context_length = st.number_input(
          "Context length", min_value=1, max_value=MAX_CONTEXT, value=512
        )
      device_options = ["cuda", "cpu"] if capabilities.cuda_available else ["cpu"]
      with st.expander("Advanced inference", icon=":material/settings:"):
        device_column, batch_column = st.columns(2)
        with device_column:
          device = st.selectbox("Device", device_options)
        with batch_column:
          batch_size = st.number_input("Batch size", min_value=1, max_value=64, value=4)
        st.markdown("**Probabilistic outputs**")
        return_quantiles = st.checkbox("Return quantiles", value=True)
        symmetric = st.checkbox("Symmetric averaging", value=True)
        positive = st.checkbox("Clamp nonnegative series", value=True)
        sort_quantiles = st.checkbox("Sort quantiles", value=True)
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
      st.form_submit_button("Save settings for analysis")

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
    with prepare_tab:
      st.markdown("#### 3. Data readiness")
      readiness = quality_report(
        datasets,
        mapping,
        settings,
        frequency=st.session_state.get("preparation_frequency") or None,
      )
      blocked = readiness.status.eq("blocked").any()
      warned = readiness.status.eq("warning").any()
      st.badge(
        "Blocked" if blocked else "Review warnings" if warned else "Ready to forecast",
        color="red" if blocked else "orange" if warned else "green",
      )
      with st.expander("Detailed readiness checks", expanded=blocked):
        st.dataframe(readiness, hide_index=True)
      with st.expander("Interpolated context values"):
        preview_id = st.session_state.get("preparation_preview_group")
        preview_dataset = next(
          (item for item in datasets if item.dataset_id == preview_id), datasets[0]
        )
        try:
          interpolated = imputation_preview(preview_dataset, mapping, settings)
          if interpolated.empty:
            st.caption("No missing model context values in this group.")
          else:
            st.caption(
              "Preview uses only the selected model context. Uploaded values and held-out actuals remain unchanged."
            )
            st.dataframe(interpolated.head(5000), hide_index=True)
            if len(interpolated) > 5000:
              st.caption("Showing the first 5,000 affected context cells.")
        except ExplorerError as exc:
          st.caption(str(exc))
    if submitted:
      try:
        with st.status("Running TimesFM-3", expanded=True) as status:
          if len(datasets) * len(mapping.targets) * settings.horizon > 250_000:
            raise ExplorerError(
              "Batch exceeds 250,000 output rows; select fewer series or a shorter horizon."
            )
          prepare_batch(datasets, mapping, settings)
          st.write("Loading checkpoint")
          predictor = _selected_forecaster(device, settings.batch_size)
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
    st.divider()
    st.subheader("Latest result", icon=":material/monitoring:")
    runs: list[RunArtifact] = st.session_state.runs
    if runs:
      _render_run(runs[-1])
    else:
      st.info("Run a forecast to see its results here.")

with evaluate_tab:
  st.subheader("Evaluate forecasts", icon=":material/analytics:")
  st.caption("Compare saved runs or test forecasting choices on historical data.")
  comparison_view, analysis_view = st.tabs(["Compare runs", "Run analysis"])
  with comparison_view:
    _render_comparison(st.session_state.runs)
  with analysis_view:
    render_analysis(
      datasets,
      mapping,
      settings,
      device,
      acknowledged,
      _selected_forecaster,
      DATABASE_PATH,
    )


def _refresh_run(
  previous: RunArtifact, current: Sequence[UploadedDataset]
) -> RunArtifact:
  refresh_settings = dataclasses.replace(previous.settings, task="forecast")
  restored = []
  restored_mapping = previous.mapping
  preparation = previous.manifest.get("preparation", {})
  for item in current:
    # The tracking UI stores the source association on the selected session frame.
    old_id = item.frame.attrs.get("tracking_previous_dataset", item.dataset_id)
    metadata = preparation.get(old_id, {})
    restored_item, restored_mapping = restore_preparation(
      item,
      previous.mapping,
      metadata,
      horizon=refresh_settings.horizon,
    )
    restored.append(restored_item)
  prepare_batch(restored, restored_mapping, refresh_settings)
  provenance = previous.manifest.get("model_provenance", {})
  selection = dataclasses.replace(
    selection_from_provenance(provenance),
    offline=model_selection.offline,
  )
  predictor = _acquire_forecaster(
    device,
    refresh_settings.batch_size,
    selection,
    expected_files=provenance.get("files") if selection.kind == "local" else None,
  )
  refreshed = execute_forecast(
    predictor,
    restored,
    restored_mapping,
    refresh_settings,
    repository_revision(Path(__file__).parent),
  )
  refreshed.manifest["previous_run_id"] = previous.run_id
  _append_run(refreshed)
  return refreshed


with track_tab:
  render_tracking(
    datasets, DATABASE_PATH, refresh_run=_refresh_run, acknowledged=acknowledged
  )
