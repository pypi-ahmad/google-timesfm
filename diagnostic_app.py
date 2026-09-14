"""Temporary Streamlit diagnostic client for the local workbench API.

Thin HTTP client over the FastAPI job backend (``src/timesfm_app/api.py``);
this module holds no business logic of its own — it only builds requests,
renders responses, and manages a little UI-only session state (pending
idempotency keys, selected job/run IDs). Dataset parsing, job execution, and
run persistence all happen server-side.

Run with ``uv run streamlit run diagnostic_app.py`` after starting the native
workbench (``.\\dev.ps1 start`` or ``launch_workbench.cmd``); the API must
already be listening or every request in this app fails with a connection
error. The API base URL is ``http://127.0.0.1:<TIMESFM_API_PORT>/api/v1``,
where ``TIMESFM_API_PORT`` defaults to 8001 (see
``src/timesfm_app/config.py``) if the env var is unset.

This is a debugging aid alongside the native web workbench (``web/``), not
the primary UI, and is unrelated to ``streamlit_app.py`` (the standalone
in-process explorer) — the two should not run against the GPU at the same
time; see the warning banner below.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Any

import httpx
import streamlit as st

from timesfm_app.schemas import RunSpec

API_ROOT = f"http://127.0.0.1:{int(os.environ.get('TIMESFM_API_PORT', '8001'))}/api/v1"


def request_api(method: str, path: str, **kwargs: Any) -> httpx.Response | None:
  """Report recoverable API failures without losing editor or selection state.

  Returns ``None`` on any request failure instead of raising, so callers can
  render an error and keep going rather than crash the whole rerun.
  """
  try:
    response = httpx.request(
      method,
      f"{API_ROOT}{path}",
      timeout=httpx.Timeout(30, connect=3),
      follow_redirects=False,  # the API is same-origin/local; a redirect would indicate misconfiguration, not a normal response
      trust_env=False,  # ignore HTTP(S)_PROXY etc. so localhost calls are never routed through a system proxy
      **kwargs,
    )
    response.raise_for_status()
    return response
  except httpx.TimeoutException:
    st.error(
      "The local API timed out. Refresh to reconnect. A submitted job may already "
      "exist; resubmitting unchanged JSON reuses the same request key."
    )
  except httpx.HTTPStatusError as exc:
    try:
      detail = exc.response.json().get("detail", "Request rejected.")
    except (ValueError, AttributeError):
      detail = "Request rejected. Check the API log."
    st.error(f"API {exc.response.status_code}: {detail}")
  except httpx.RequestError:
    st.error(
      "The local API is unavailable. Start the workbench with .\\dev.ps1 start, then refresh."
    )
  return None


def read_records(path: str, workspace: str) -> list[dict]:
  response = request_api("GET", path, params={"workspace_id": workspace})
  return response.json() if response is not None else []


def select_record(label: str, records: list[dict], key: str) -> dict | None:
  """Keep selection attached to a stable ID when ordering or status changes."""
  choices = {record["id"]: record for record in records}
  if not choices:
    return None
  if st.session_state.get(key) not in choices:
    st.session_state[key] = next(iter(choices))
  identifier = st.selectbox(
    label,
    list(choices),
    key=key,
    format_func=lambda value: (
      f"{choices[value].get('name', choices[value].get('kind', 'Job'))} · {value[:12]}"
    ),
  )
  return choices.get(identifier)


st.set_page_config(page_title="TimesFM API diagnostics", layout="wide")
st.title("TimesFM API diagnostics")
st.info(
  "This client uses the local API. Only the GPU worker runs inference. "
  "Use this diagnostic client alongside the workbench; the original Streamlit "
  "explorer can load a separate GPU model and should remain stopped."
)
st.caption(API_ROOT)
st.session_state.setdefault(
  "spec_json", json.dumps(RunSpec().model_dump(mode="json"), indent=2)
)
st.session_state.setdefault("pending_submission", None)
st.session_state.setdefault("download", None)
workspace = st.text_input("Workspace ID", value="local", key="workspace")
st.button("Refresh", key="refresh")

with st.expander("Upload a dataset"):
  with st.form("upload"):
    source_name = st.text_input("Source name", value="Diagnostic dataset")
    uploaded = st.file_uploader("CSV or Parquet", type=["csv", "parquet", "pq"])
    upload_clicked = st.form_submit_button("Upload", key="upload_submit")
  if upload_clicked:
    if uploaded is None:
      st.warning("Choose a CSV or Parquet file first.")
    else:
      response = request_api(
        "POST",
        "/datasets",
        data={"name": source_name, "workspace_id": workspace},
        files={
          "file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")
        },
      )
      if response is not None:
        version = response.json()
        st.success(f"Uploaded version {version['id']}.")
        st.json(version)

with st.expander("Submit a job", expanded=True):
  st.caption("Set dataset_version_ids and column roles in the frozen RunSpec JSON.")
  with st.form("submit"):
    spec_json = st.text_area("RunSpec JSON", key="spec_json", height=300)
    submitted = st.form_submit_button("Submit job", key="submit_job")
  if submitted:
    try:
      specification = RunSpec.model_validate_json(spec_json).model_dump(mode="json")
    except ValueError as exc:
      st.error(f"Invalid RunSpec JSON: {exc}")
    else:
      body = {
        "workspace_id": workspace,
        "kind": specification["kind"],
        "spec": specification,
      }
      # The Idempotency-Key is stable per distinct request body: resubmitting
      # unchanged JSON (e.g. after a timeout, per the error message in
      # request_api) reuses the same key so the API can dedupe the retry
      # instead of creating a second job. A body edit changes the fingerprint
      # and mints a fresh key.
      fingerprint = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode()
      ).hexdigest()
      pending = st.session_state.pending_submission
      if pending is None or pending["fingerprint"] != fingerprint:
        pending = {"fingerprint": fingerprint, "key": uuid.uuid4().hex}
        st.session_state.pending_submission = pending
      response = request_api(
        "POST", "/jobs", json=body, headers={"Idempotency-Key": pending["key"]}
      )
      if response is not None:
        st.session_state.selected_job_id = response.json()["id"]
        st.session_state.pending_submission = None
        st.success(f"Submitted job {response.json()['id']}.")

st.subheader("Jobs")
jobs = read_records("/jobs", workspace)
if jobs:
  st.dataframe(
    [
      {
        key: job.get(key)
        for key in ("id", "kind", "status", "stage", "attempt", "created_at")
      }
      for job in jobs
    ],
    hide_index=True,
  )
  job = select_record("Selected job", jobs, "selected_job_id")
  if job is not None:
    st.json(job)
    with st.container(horizontal=True):
      if st.button(
        "Cancel job",
        key="cancel_job",
        disabled=job["status"] not in {"queued", "running"},
      ):
        response = request_api("POST", f"/jobs/{job['id']}/cancel")
        if response is not None:
          st.rerun()
      if st.button("Retry job", key="retry_job", disabled=job["status"] != "failed"):
        response = request_api("POST", f"/jobs/{job['id']}/retry")
        if response is not None:
          st.rerun()
else:
  st.caption("No jobs to display.")

st.subheader("Saved runs")
runs = read_records("/runs", workspace)
selected_run = select_record("Selected run", runs, "selected_run_id")
if selected_run is None:
  st.caption("No saved runs to display.")
else:
  payload = selected_run["payload"]
  st.json(payload.get("manifest", {}), expanded=False)
  names = list(payload.get("tables", {}))
  if names:
    if st.session_state.get("table_name") not in names:
      st.session_state.table_name = names[0]
    table_name = st.selectbox("Result table", names, key="table_name")
    offset = st.number_input(
      "Row offset",
      min_value=0,
      step=100,
      key=f"offset_{selected_run['id']}_{table_name}",
    )
    response = request_api(
      "GET",
      f"/runs/{selected_run['id']}/tables/{table_name}",
      params={"offset": int(offset), "limit": 100},
    )
    if response is not None:
      page = response.json()
      st.caption(f"{len(page['rows'])} rows shown · {page['total']} total rows")
      st.dataframe(page["rows"], hide_index=True)
  if payload.get("export") and st.button("Load original ZIP", key="load_export"):
    # Fetching the export is a separate, explicit step (rather than fetching
    # it whenever a run is selected) because export bytes can be large and
    # are only needed if the user actually wants to download them.
    response = request_api("GET", f"/runs/{selected_run['id']}/export")
    if response is not None:
      st.session_state.download = {
        "run_id": selected_run["id"],
        "bytes": response.content,
      }
  saved_download = st.session_state.download
  if saved_download is not None and saved_download["run_id"] == selected_run["id"]:
    st.download_button(
      "Download original ZIP",
      saved_download["bytes"],
      file_name=f"timesfm3-{selected_run['id']}.zip",
      mime="application/zip",
      on_click="ignore",  # skip the rerun a download click would otherwise trigger; bytes are already cached in session_state
      key="download_export",
    )
