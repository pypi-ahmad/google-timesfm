"""Explicit-submit analysis workflows for the local Streamlit explorer."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st
import torch

from timesfm3.analysis import (
  AnalysisArtifact,
  AnalysisSettings,
  ExperimentConfiguration,
  Scenario,
  analysis_fingerprint,
  analysis_zip,
  anomaly_table,
  comparison_table,
  configuration_rankings,
  prepare_analysis,
  run_analysis,
  scenario_deltas,
  scenario_from_edits,
  scenario_template,
)
from timesfm3.explorer import (
  MAX_CONTEXT,
  DatasetMapping,
  ExplorerError,
  ForecastSettings,
  UploadedDataset,
)
from timesfm3.run_store import (
  RunStoreError,
  list_analyses,
  load_analysis,
  save_analysis,
)
from timesfm3.uncertainty import calibration_table, interval_bands

WORKFLOWS = {
  "Rolling backtesting": "backtest",
  "Anomaly detection": "anomaly",
  "What-if scenarios": "scenario",
  "Joint versus independent": "joint_independent",
  "Covariate usefulness": "covariates",
  "Forecast configurations": "settings",
  "Naive baselines": "baselines",
}
WORKFLOW_DESCRIPTIONS = {
  "Rolling backtesting": "Measure accuracy across several historical cutoffs.",
  "Anomaly detection": "Flag observations outside forecasts made from earlier data.",
  "What-if scenarios": "Compare conditional forecasts after editing future covariates.",
  "Joint versus independent": "Test whether multivariate forecasting improves accuracy.",
  "Covariate usefulness": "Measure held-out accuracy with and without selected covariates.",
  "Forecast configurations": "Rank context lengths and inference settings by backtest accuracy.",
  "Naive baselines": "Compare TimesFM-3 with last-value and seasonal-naive forecasts.",
}


def _result(artifact: AnalysisArtifact) -> None:
  """Filter stored results without initiating model work."""
  st.subheader(f"Analysis {artifact.analysis_id}")
  predictions = artifact.predictions
  if predictions.empty:
    st.info("This analysis has no prediction rows.")
    return
  prefix = f"analysis_result_{artifact.analysis_id}"
  dataset = st.selectbox("Analysis dataset", predictions.dataset.unique(), key=prefix)
  selected = predictions.loc[predictions.dataset == dataset]
  target = st.selectbox(
    "Analysis target", selected.target.unique(), key=prefix + "_target"
  )
  selected = selected.loc[selected.target == target]
  if artifact.kind != "anomaly":
    origin = st.selectbox(
      "Forecast origin", selected.origin.unique(), key=prefix + "_origin"
    )
    selected = selected.loc[selected.origin == origin]
  display = selected
  if len(display) > 5000:
    display = display.iloc[:: (len(display) + 4999) // 5000]
    st.caption(
      "Chart sampled to at most 5,000 rows. Tables and exports contain all rows."
    )
  temporal = isinstance(display.iloc[0].timestamp, pd.Timestamp)
  chart: Any = alt.Chart(display)
  x = alt.X("timestamp", type="temporal" if temporal else "quantitative", title="Time")
  layers = [
    chart.mark_line().encode(
      x=x, y="point:Q", color="variant:N", tooltip=["timestamp", "variant", "point"]
    )
  ]
  bands = interval_bands(display)
  if not bands.empty:
    band_variants = list(bands.variant.unique())
    if len(band_variants) > 1:
      band_variant = st.selectbox(
        "Interval bands for",
        band_variants,
        key=prefix + "_band_variant",
      )
      bands = bands.loc[bands.variant.eq(band_variant)]
    for nominal in (20, 40, 60, 80):
      band = bands.loc[bands.nominal_coverage_percent.eq(nominal) & bands.valid_bounds]
      if not band.empty:
        band_chart: Any = alt.Chart(band)
        layers.insert(
          0,
          band_chart.mark_area(opacity=0.12).encode(
            x=x,
            y="lower:Q",
            y2="upper:Q",
            color="variant:N",
            tooltip=[
              "timestamp",
              "variant",
              "nominal_coverage_percent",
              "lower",
              "upper",
            ],
          ),
        )
    st.caption(
      "20%, 40%, 60% and 80% nominal prediction bands; observed coverage may differ."
    )
    if bands.crossed.any():
      st.warning(
        "Crossing quantiles are excluded from interval charts and coverage scores."
      )
  if "actual" in display:
    layers.append(
      chart.mark_line(color="#333333", strokeDash=[4, 3]).encode(
        x=x, y="actual:Q", detail="variant:N"
      )
    )
  if artifact.kind == "anomaly":
    anomalies = anomaly_table(selected)
    flagged = anomalies.loc[anomalies.flagged.fillna(False)]
    flagged_chart: Any = alt.Chart(flagged)
    layers.append(
      flagged_chart.mark_point(color="#c62828", size=90).encode(
        x=x, y="actual:Q", tooltip=["timestamp", "actual", "distance", "direction"]
      )
    )
    st.metric("Flagged observations", len(flagged))
    st.dataframe(anomalies, hide_index=True)
  st.altair_chart(alt.layer(*layers).properties(height=360).interactive())
  if artifact.kind == "scenario":
    st.caption("Conditional predictions, not causal effects.")
    st.dataframe(scenario_deltas(selected), hide_index=True)
  if not artifact.metrics.empty:
    scope = st.selectbox(
      "Metric detail", ["overall", "step", "window"], key=prefix + "_scope"
    )
    metrics = artifact.metrics
    visible = metrics.loc[
      (metrics.dataset == dataset)
      & (metrics.target == target)
      & (metrics.scope == scope)
    ]
    st.dataframe(visible, hide_index=True)
    if artifact.kind in {"joint_independent", "covariates"}:
      st.caption(
        "Positive MAE difference favors joint forecasting."
        if artifact.kind == "joint_independent"
        else "Positive MAE difference favors including covariates. Predictive usefulness is not causal importance."
      )
      st.dataframe(comparison_table(visible, artifact.kind), hide_index=True)
    if artifact.kind in {"settings", "baselines"}:
      target_ranks, overall_ranks = configuration_rankings(artifact.metrics)
      st.subheader("Configuration rankings")
      st.caption(
        "Lower ranks are better. Each scored target has equal weight in the mean rank; ties share a rank."
      )
      st.dataframe(overall_ranks, hide_index=True)
      with st.expander("Evaluated configurations"):
        st.dataframe(
          pd.DataFrame(
            [
              {"name": item["name"], **item.get("settings", {})}
              for item in artifact.manifest.get("variants", [])
            ]
          ),
          hide_index=True,
        )
      st.dataframe(
        target_ranks.loc[
          target_ranks.dataset.eq(dataset) & target_ranks.target.eq(target)
        ],
        hide_index=True,
      )
      scoring = artifact.manifest.get("scoring", {})
      if scoring:
        st.caption(
          f"Shared scored pairs: {scoring['scored_pairs']:,}. Excluded pairs: {scoring['excluded_pairs']:,}."
        )
  coverage = calibration_table(
    predictions.loc[predictions.dataset.eq(dataset) & predictions.target.eq(target)]
  )
  if not coverage.empty:
    with st.expander("Interval coverage by forecast step"):
      st.caption(
        "Descriptive coverage across historical windows. No intervals are fitted or adjusted; overlapping windows are dependent observations."
      )
      st.dataframe(coverage, hide_index=True)
  with st.expander("Prediction data"):
    st.dataframe(selected, hide_index=True)
  st.download_button(
    "Download analysis bundle",
    analysis_zip(artifact),
    file_name=f"timesfm3-analysis-{artifact.analysis_id}.zip",
    mime="application/zip",
    key=prefix + "_download",
  )


def render_analysis(
  datasets: Sequence[UploadedDataset],
  mapping: DatasetMapping | None,
  settings: ForecastSettings | None,
  device: str,
  acknowledged: bool,
  forecaster: Callable[..., Any],
  database: Path,
) -> None:
  """Prepare before model acquisition and retain one completed analysis in session."""
  if datasets and mapping is not None and settings is not None:
    st.caption(
      "Saved Forecast settings · "
      f"{settings.mode} · horizon {settings.horizon} · "
      f"context {settings.context_length} · {device}"
    )
    label = st.selectbox("Analysis workflow", list(WORKFLOWS), key="analysis_workflow")
    st.caption(WORKFLOW_DESCRIPTIONS[label])
    kind = WORKFLOWS[label]
    scenarios: list[Scenario] = []
    configurations: list[ExperimentConfiguration] = []
    template: pd.DataFrame | None = None
    fingerprint = analysis_fingerprint(datasets, mapping, settings)
    if st.session_state.get("analysis_editor_fingerprint") != fingerprint:
      for key in list(st.session_state):
        if isinstance(key, str) and key.startswith(
          ("scenario_editor_", "configuration_editor_")
        ):
          del st.session_state[key]
      st.session_state.analysis_editor_fingerprint = fingerprint
    if kind == "scenario":
      st.caption(
        "Edit future covariate values only. Conditional predictions, not causal effects."
      )
      try:
        template = scenario_template(datasets, mapping, settings)
      except ExplorerError as exc:
        st.error(str(exc))
      count = int(
        st.number_input(
          "Scenarios", min_value=1, max_value=3, value=1, key="scenario_count"
        )
      )
      if st.button("Reset scenario edits"):
        for key in list(st.session_state):
          if isinstance(key, str) and key.startswith("scenario_editor_"):
            del st.session_state[key]
        st.rerun()
    else:
      count = 0
    configuration_count = 0
    if kind == "settings":
      configuration_count = int(
        st.number_input(
          "Configurations", min_value=2, max_value=8, value=2, key="configuration_count"
        )
      )
    with st.form("analysis_form"):
      windows, stride = 5, settings.horizon
      seasonal_period = 1
      edited_configurations = None
      covariates: tuple[str, ...] = ()
      editors = []
      if kind == "anomaly":
        windows = int(
          st.number_input(
            "Observations to check", min_value=1, max_value=1000, value=50
          )
        )
        stride = 1
        st.caption(
          "One-step forecasts use only earlier observations. Quantiles and sorting are required."
        )
      elif kind != "scenario":
        windows = int(
          st.number_input("Historical windows", min_value=1, max_value=100, value=5)
        )
        stride = int(
          st.number_input("Window stride", min_value=1, value=settings.horizon)
        )
      if kind == "joint_independent":
        st.caption(
          "Both modes exclude covariates and use identical historical cutoffs."
        )
      if kind in {"settings", "baselines"}:
        suggested_period = 1
        if mapping.timestamp is not None:
          times = pd.to_datetime(
            datasets[0].frame[mapping.timestamp], errors="coerce", utc=True
          ).sort_values()
          frequency = (
            pd.infer_freq(times)
            if len(times) >= 3 and times.notna().all() and not times.duplicated().any()
            else None
          )
          if frequency:
            frequency = frequency.upper()
            suggested_period = (
              24
              if frequency == "H"
              else 7
              if frequency == "D"
              else 52
              if frequency.startswith("W")
              else 12
              if frequency.startswith(("MS", "ME"))
              else 4
              if frequency.startswith(("QS", "QE"))
              else 1
            )
        seasonal_period = int(
          st.number_input(
            "Seasonal period (rows)",
            min_value=1,
            max_value=MAX_CONTEXT,
            value=suggested_period,
          )
        )
        st.caption(
          "Confirm the number of rows in one season. On irregular data this counts observations, not calendar periods. Baselines use raw earlier observations; missing seasonal values stay unscored."
        )
        st.caption(
          "Every window must contain the full largest context and seasonal period. All methods are scored on the same observations."
        )
      if kind == "settings":
        fields = (
          "context_length",
          "use_symmetric_averaging",
          "make_positive",
          "sort_quantiles",
          "use_znorm",
          "padding_mode",
        )
        rows = [
          dict(
            name=f"Configuration {index + 1}",
            **{field: getattr(settings, field) for field in fields},
          )
          for index in range(configuration_count)
        ]
        rows[1]["context_length"] = max(2, settings.context_length // 2)
        edited_configurations = st.data_editor(
          pd.DataFrame(rows),
          hide_index=True,
          num_rows="fixed",
          key="configuration_editor_settings",
          column_config={
            "name": st.column_config.TextColumn("Name", required=True),
            "context_length": st.column_config.NumberColumn(
              "Context rows", min_value=2, max_value=MAX_CONTEXT, step=1, required=True
            ),
            "use_symmetric_averaging": st.column_config.CheckboxColumn(
              "Symmetric averaging", required=True
            ),
            "make_positive": st.column_config.CheckboxColumn(
              "Positive forecasts", required=True
            ),
            "sort_quantiles": st.column_config.CheckboxColumn(
              "Sort quantiles", required=True
            ),
            "use_znorm": st.column_config.CheckboxColumn(
              "Normalize context", required=True
            ),
            "padding_mode": st.column_config.SelectboxColumn(
              "Padding", options=["none", "edge"], required=True
            ),
          },
        )
      if kind == "covariates":
        options = list(mapping.past_only + mapping.past_future)
        covariates = tuple(
          st.multiselect("Covariates to assess", options, default=options)
        )
      if template is not None:
        for index in range(count):
          name = st.text_input(
            "Scenario name", value=f"Scenario {index + 1}", key=f"scenario_name_{index}"
          )
          edited = st.data_editor(
            template,
            hide_index=True,
            disabled=[column for column in template.columns if column != "value"],
            num_rows="fixed",
            key=f"scenario_editor_{index}",
          )
          editors.append((name, edited))
      submitted = st.form_submit_button(
        "Run analysis", disabled=not acknowledged, type="primary"
      )
    if submitted:
      try:
        if edited_configurations is not None:
          if edited_configurations.isna().any().any():
            raise ExplorerError("Fill every configuration value before running.")
          for row in edited_configurations.to_dict(orient="records"):
            configurations.append(
              ExperimentConfiguration(
                str(row.pop("name")), dataclasses.replace(settings, **row)
              )
            )
        if kind == "covariates" and not covariates:
          raise ExplorerError("Select at least one covariate to assess.")
        if kind == "scenario":
          if template is None:
            raise ExplorerError(
              "Provide valid known-future covariates before running scenarios."
            )
          scenarios = [
            scenario_from_edits(name, template, edited, fingerprint)
            for name, edited in editors
          ]
        analysis_settings = AnalysisSettings(
          kind=cast(Any, kind),
          windows=windows,
          stride=stride,
          selected_covariates=covariates,
          seasonal_period=seasonal_period,
        )
        prepared = prepare_analysis(
          datasets, mapping, settings, analysis_settings, scenarios, configurations
        )
        st.write(
          f"{prepared.forecast_count:,} forecasts · {prepared.prediction_rows:,} prediction rows"
        )
        with st.status("Running analysis", expanded=True) as status:
          progress = st.progress(0.0)

          def update_progress(done: int, total: int) -> None:
            progress.progress(done / total if total else 1.0)

          predictor = forecaster(device, settings.batch_size)
          artifact = run_analysis(
            predictor,
            prepared,
            progress=update_progress,
          )
          st.session_state.analysis_latest = artifact
          try:
            save_analysis(database, artifact)
            st.session_state.analysis_save_warning = None
          except RunStoreError as exc:
            st.session_state.analysis_save_warning = str(exc)
          status.update(label="Analysis complete", state="complete", expanded=False)
      except ExplorerError as exc:
        st.error(str(exc))
      except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error(
          "CUDA ran out of memory. Reduce context, horizon, or batch size; CPU retry is manual."
        )
      except Exception as exc:  # noqa: BLE001 - model boundary stays in UI.
        st.error(
          f"Analysis failed ({type(exc).__name__}). Check checkpoint access and runtime configuration."
        )
  else:
    st.info(
      "Prepare data and save Forecast settings to start an analysis. "
      "Saved results remain available below."
    )
  if st.session_state.get("analysis_save_warning"):
    st.warning(
      f"{st.session_state.analysis_save_warning} Results remain in this browser session."
    )
  current = st.session_state.get("analysis_latest")
  if current is not None:
    _result(current)
  with st.expander("Saved analyses"):
    try:
      saved = list_analyses(database)
      if saved:
        labels = {
          item[
            "analysis_id"
          ]: f"{item['created_at']} · {item['kind']} · {item['analysis_id']}"
          for item in saved
        }
        selected_id = st.selectbox(
          "Saved analysis", list(labels), format_func=labels.__getitem__
        )
        if st.button("Load saved analysis"):
          st.session_state.analysis_latest = load_analysis(database, selected_id)
          st.rerun()
      else:
        st.caption("No saved analyses. The newest 25 are retained locally.")
    except RunStoreError as exc:
      st.warning(str(exc))
