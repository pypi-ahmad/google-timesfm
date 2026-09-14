"""Explicit upload assessment and manual refresh of saved forecast vintages.

Streamlit UI over `tracking.py` (assessment/scoring) and `run_store.py`
(persisted runs/assessments, retention). `render_tracking` is the entry
point: pick a saved run, associate its old dataset identities with newly
uploaded datasets, then either score them against actuals (`assess_run`)
or fully rerun the forecast via the caller-supplied `refresh_run`
callback (kept injectable here so this module doesn't itself depend on
loading/running a model). See `run_store.py` for what "tracked" and
retention/eviction of untracked runs mean.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import streamlit as st

from .explorer import ExplorerError, RunArtifact, UploadedDataset, artifact_zip
from .run_store import (
  RunStoreError,
  link_runs,
  list_tracking_runs,
  load_assessments,
  load_run,
  save_assessment,
  save_run,
  set_run_tracked,
)
from .tracking import assess_run, assessment_zip, associated_datasets
from .uncertainty import calibration_table


def _assessment_result(database: Path, run: RunArtifact) -> None:
  assessments = load_assessments(database, run.run_id)
  if not assessments:
    st.info("No updated actuals have been assessed for this forecast.")
    return
  choice = st.selectbox(
    "Actuals version",
    ["Latest assessment", "First assessment", "Choose assessment"],
    key=f"tracking_version_{run.run_id}",
  )
  # Assumes load_assessments returns them in chronological order (oldest
  # first), per run_store.py, so [0]/[-1] are the first/latest.
  selected = assessments[-1] if choice == "Latest assessment" else assessments[0]
  if choice == "Choose assessment":
    by_id = {item.assessment_id: item for item in assessments}
    identifier = st.selectbox(
      "Recorded assessment",
      list(by_id),
      format_func=lambda key: by_id[key].created_at,
      key=f"tracking_assessment_{run.run_id}",
    )
    selected = by_id[identifier]
  st.caption(
    f"Recorded {selected.created_at}. "
    f"Matched {len(selected.comparisons):,} of {len(run.forecast):,} issued predictions. "
    "Missing actuals are excluded. The issued forecast is unchanged."
  )
  st.dataframe(selected.metrics, hide_index=True)
  calibration = calibration_table(selected.comparisons)
  if not calibration.empty:
    st.caption("Observed interval coverage by target and horizon; counts may be small.")
    st.dataframe(calibration, hide_index=True)
  frame = selected.comparisons
  dataset = st.selectbox(
    "Tracked dataset", frame.dataset.unique(), key=f"tracking_dataset_{run.run_id}"
  )
  frame = frame.loc[frame.dataset == dataset]
  target = st.selectbox(
    "Tracked target", frame.target.unique(), key=f"tracking_target_{run.run_id}"
  )
  frame = frame.loc[frame.target == target]
  st.line_chart(frame.set_index("timestamp")[["point", "actual"]])
  with st.expander("Matched predictions and actuals"):
    st.dataframe(frame, hide_index=True)
  st.download_button(
    "Download this assessment",
    assessment_zip(selected),
    file_name=f"timesfm3-{selected.assessment_id}.zip",
    mime="application/zip",
    key=f"tracking_assessment_download_{run.run_id}",
  )


@st.dialog("Stop tracking this forecast?")
def _confirm_stop_tracking(database_path: Path, run_id: str) -> None:
  # "25" mirrors run_store.py's retention policy for untracked runs;
  # unclear from this file alone whether that count is a named constant
  # there that this string could drift out of sync with -- see
  # run_store.py.
  st.warning(
    "This removes retention protection. If the forecast is outside the newest "
    "25 untracked runs, it and its saved assessments may be removed immediately."
  )
  with st.container(horizontal=True):
    st.button("Keep tracking", key=f"tracking_cancel_{run_id}")
    st.button(
      "Stop tracking",
      type="primary",
      key=f"tracking_confirm_{run_id}",
      on_click=set_run_tracked,
      args=(database_path, run_id, False),
    )


def render_tracking(
  datasets: Sequence[UploadedDataset],
  database_path: Path,
  *,
  refresh_run: Callable[[RunArtifact, Sequence[UploadedDataset]], RunArtifact]
  | None = None,
  acknowledged: bool = False,
) -> None:
  """Assess only on submit; acquire a model only through an explicit refresh callback."""
  st.subheader("Track forecasts", icon=":material/history:")
  if notice := st.session_state.pop("tracking_refresh_notice", None):
    st.success(notice)
  st.caption(
    "Compare saved predictions with new actuals or refresh a forecast from saved settings."
  )
  try:
    summaries = list_tracking_runs(database_path)
    if not summaries:
      st.info("Save a forecast to begin tracking its predictions.")
      return
    by_id = {row["run_id"]: row for row in summaries}
    run_id = st.selectbox(
      "Saved forecast to track",
      list(by_id),
      format_func=lambda key: (
        f"{key} · {by_id[key]['created_at']}"
        + (" · tracked" if by_id[key]["tracked"] else "")
      ),
      key="tracking_saved_run",
    )
    run = load_run(database_path, run_id)
    tracked = by_id[run_id]["tracked"]
    with st.container(horizontal=True):
      tracking_label = "Stop tracking" if tracked else "Keep this forecast tracked"
      if st.button(tracking_label, key=f"tracking_toggle_{run_id}"):
        if tracked:
          _confirm_stop_tracking(database_path, run_id)
        else:
          set_run_tracked(database_path, run_id, True)
          st.rerun()
      st.download_button(
        "Download issued forecast",
        artifact_zip(run),
        file_name=f"timesfm3-run-{run_id}.zip",
        mime="application/zip",
        key=f"tracking_forecast_download_{run_id}",
      )
    previous_run_id = run.manifest.get("previous_run_id")
    if previous_run_id:
      expired = " (no longer retained)" if previous_run_id not in by_id else ""
      st.caption(f"Previous forecast: {previous_run_id}{expired}.")
    if datasets:
      current_ids = [dataset.dataset_id for dataset in datasets]
      old_ids = sorted(run.forecast.dataset.unique())
      # "dataset_<N>" identifies a run saved before dataset identity was
      # based on (source name, group key) instead of upload position
      # (see data_preparation.py); such positional labels can't be
      # trusted to match the same series across uploads, so they're
      # flagged for the user to associate manually rather than
      # auto-matched by name below.
      legacy = any(re.fullmatch(r"dataset_\d+", str(key)) for key in old_ids)
      if legacy:
        st.info(
          "This saved run uses legacy dataset labels. Associate each dataset explicitly."
        )
      rows = [
        {
          "Saved dataset": key,
          "Updated dataset": key
          if key in current_ids and not re.fullmatch(r"dataset_\d+", str(key))
          else None,
        }
        for key in old_ids
      ]
      # Hash the current dataset identities + content into the form/widget
      # keys below, so the association editor's state resets whenever the
      # uploaded set changes (new/removed/re-uploaded datasets) instead of
      # Streamlit reusing stale edited associations from a previous upload.
      upload_key = hashlib.sha256(
        "|".join(f"{item.dataset_id}:{item.sha256}" for item in datasets).encode()
      ).hexdigest()[:12]
      with st.form(f"tracking_upload_{run_id}_{upload_key}"):
        st.caption("Match dataset identities. Leave a row blank to skip it.")
        edited = st.data_editor(
          pd.DataFrame(rows),
          hide_index=True,
          disabled=["Saved dataset"],
          column_config={
            "Updated dataset": st.column_config.SelectboxColumn(
              "Updated dataset", options=current_ids
            ),
          },
          key=f"tracking_associations_{run_id}_{upload_key}",
        )
        assess = st.form_submit_button("Evaluate against actuals")
        refresh = st.form_submit_button(
          "Refresh forecast",
          disabled=refresh_run is None or not acknowledged,
          help="Uses this forecast's saved mapping, settings, and model revision.",
        )
      if assess or refresh:
        associations = {
          str(row["Saved dataset"]): str(row["Updated dataset"])
          for _, row in edited.iterrows()
          if pd.notna(row["Updated dataset"])
        }
        selected_datasets = associated_datasets(run, datasets, associations)
        if assess:
          assessment = assess_run(run, datasets, associations)
          save_assessment(database_path, assessment)
          st.success("Actuals evaluation saved. This forecast is now tracked.")
        elif refresh and refresh_run is not None:
          set_run_tracked(database_path, run_id, True)
          with st.spinner("Rerunning the saved forecast settings…"):
            updated = refresh_run(run, selected_datasets)
            save_run(database_path, updated)
            link_runs(database_path, updated.run_id, run_id, associations)
          st.session_state.tracking_refresh_notice = (
            f"New forecast {updated.run_id} saved and linked to {run_id}."
          )
          st.rerun()
    else:
      st.info("Upload updated data in Prepare to assess or rerun this forecast.")
    _assessment_result(database_path, run)
  except (ExplorerError, RunStoreError) as exc:
    # These carry user-facing messages already (data/validation errors),
    # so shown verbatim.
    st.error(str(exc))
  except RuntimeError:
    # A bare RuntimeError here is assumed to come from refresh_run's
    # model load/inference (e.g. OOM, missing weights) rather than from
    # this module's own logic; its message is deliberately not shown
    # (may be a raw framework error) in favor of a generic, actionable
    # hint.
    st.error(
      "Could not refresh the forecast. Check model availability and memory, then try again."
    )
