"""The temporary diagnostic UI uses HTTP and never acquires a model."""

import ast
from pathlib import Path

import httpx
from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "diagnostic_app.py"


def response(method, url, **kwargs):
  path = httpx.URL(url).path
  request = httpx.Request(method, url)
  if path.endswith("/jobs"):
    result = [{"id": "job-1", "kind": "forecast", "status": "running", "attempt": 1}]
  elif path.endswith("/runs"):
    result = []
  else:
    result = {}
  return httpx.Response(200, json=result, request=request)


def test_diagnostic_imports_only_http_ui_and_contracts():
  tree = ast.parse(APP.read_text(encoding="utf-8"))
  imports = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      imports.extend(alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
      imports.append(node.module or "")
  assert "httpx" in imports
  assert "timesfm_app.schemas" in imports
  assert not any(
    name.startswith(("torch", "timesfm3", "timesfm_app.services", "timesfm_app.worker"))
    for name in imports
  )


def test_diagnostic_lists_jobs_without_gpu_and_keeps_selection(monkeypatch):
  jobs = [
    {"id": "job-1", "kind": "forecast", "status": "running", "attempt": 1},
    {"id": "job-2", "kind": "backtest", "status": "queued", "attempt": 0},
  ]

  def mocked(method, url, **kwargs):
    if url.endswith("/jobs"):
      return httpx.Response(200, json=jobs, request=httpx.Request(method, url))
    return response(method, url, **kwargs)

  monkeypatch.setattr(httpx, "request", mocked)
  app = AppTest.from_file(str(APP), default_timeout=10).run()
  assert not app.exception
  app.selectbox(key="selected_job_id").select("job-2").run()
  jobs.reverse()
  app.button(key="refresh").click().run()
  assert app.selectbox(key="selected_job_id").value == "job-2"
  assert not app.exception


def test_timed_out_submission_reuses_idempotency_key(monkeypatch):
  keys = []

  def mocked(method, url, **kwargs):
    if method == "POST" and url.endswith("/jobs"):
      keys.append(kwargs["headers"]["Idempotency-Key"])
      assert kwargs["trust_env"] is False
      if len(keys) == 1:
        raise httpx.ReadTimeout("timeout")
      return httpx.Response(
        202, json={"id": "job-1"}, request=httpx.Request(method, url)
      )
    return response(method, url, **kwargs)

  monkeypatch.setattr(httpx, "request", mocked)
  app = AppTest.from_file(str(APP), default_timeout=10).run()
  app.button(key="submit_job").click().run()
  assert app.error and not app.exception
  app.button(key="submit_job").click().run()
  assert keys[0] == keys[1]
  assert app.session_state.pending_submission is None
  assert app.success and not app.exception


def test_cancel_and_retry_use_api(monkeypatch):
  calls = []
  state = "running"

  def mocked(method, url, **kwargs):
    nonlocal state
    if method == "POST":
      calls.append(httpx.URL(url).path)
      state = "cancelling" if url.endswith("cancel") else "queued"
      return httpx.Response(200, json={}, request=httpx.Request(method, url))
    if url.endswith("/jobs"):
      return httpx.Response(
        200,
        json=[{"id": "job-1", "kind": "forecast", "status": state}],
        request=httpx.Request(method, url),
      )
    return response(method, url, **kwargs)

  monkeypatch.setattr(httpx, "request", mocked)
  app = AppTest.from_file(str(APP), default_timeout=10).run()
  app.button(key="cancel_job").click().run()
  assert calls == ["/api/v1/jobs/job-1/cancel"]
  state = "failed"
  app.run()
  app.button(key="retry_job").click().run()
  assert calls[-1] == "/api/v1/jobs/job-1/retry"
  assert not app.exception


def test_invalid_json_and_api_outage_are_recoverable(monkeypatch):
  def unavailable(*args, **kwargs):
    raise httpx.ConnectError("connection refused")

  monkeypatch.setattr(httpx, "request", unavailable)
  app = AppTest.from_file(str(APP), default_timeout=10).run()
  assert app.error and not app.exception
  app.text_area(key="spec_json").set_value("{invalid").run()
  app.button(key="submit_job").click().run()
  assert any("Invalid RunSpec" in error.value for error in app.error)
  assert app.text_area(key="spec_json").value == "{invalid"
  assert not app.exception


def test_saved_table_paging_and_export_bytes(monkeypatch):
  offsets = []
  export = b"original ZIP bytes"

  def mocked(method, url, **kwargs):
    if url.endswith("/runs"):
      data = [
        {
          "id": "run-1",
          "name": "Saved",
          "payload": {
            "manifest": {},
            "tables": {"forecast": {"key": "unused"}},
            "export": {"key": "unused"},
          },
        }
      ]
    elif "/tables/" in url:
      offsets.append(kwargs["params"]["offset"])
      data = {"columns": ["point"], "rows": [{"point": 3.5}], "total": 300}
    elif url.endswith("/export"):
      return httpx.Response(200, content=export, request=httpx.Request(method, url))
    else:
      return response(method, url, **kwargs)
    return httpx.Response(200, json=data, request=httpx.Request(method, url))

  monkeypatch.setattr(httpx, "request", mocked)
  app = AppTest.from_file(str(APP), default_timeout=10).run()
  app.number_input(key="offset_run-1_forecast").set_value(100).run()
  assert offsets[-1] == 100
  app.button(key="load_export").click().run()
  assert app.session_state.download["bytes"] == export
  assert not app.exception
