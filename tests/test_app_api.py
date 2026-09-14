"""HTTP transport, persistent draft conflicts, and workspace isolation."""

import io
import zipfile

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from timesfm_app.api import create_app
from timesfm_app.artifacts import ArtifactStore
from timesfm_app.config import Settings
from timesfm_app.store import Store


@pytest.fixture
def client(tmp_path):
  store = Store.for_testing()
  storage = ArtifactStore(tmp_path / "artifacts")
  with TestClient(create_app(store, storage, Settings())) as client:
    yield client
  store.close()


def upload(client, workspace="local"):
  data = (
    b"date,sales,promo\n2026-01-01,2,0\n2026-01-02,3,1\n2026-01-03,4,0\n2026-01-04,,1\n"
  )
  response = client.post(
    "/api/v1/datasets",
    data={"name": "Shop", "workspace_id": workspace},
    files={"file": ("sales.csv", data)},
  )
  assert response.status_code == 201, response.text
  return response.json()


def spec(version):
  return {
    "dataset_version_ids": [version["id"]],
    "mapping": {"timestamp": "date", "targets": ["sales"], "past_future": ["promo"]},
    "settings": {"horizon": 1, "context_length": 3},
  }


def test_upload_version_reopen_preview_and_duplicate_identity(client):
  version = upload(client)
  assert version["payload"]["columns"] == ["date", "sales", "promo"]
  response = client.get(f"/api/v1/datasets/versions/{version['id']}/preview")
  assert response.status_code == 200
  assert response.json()["rows"][-1]["sales"] is None
  duplicate = client.post(
    "/api/v1/datasets",
    data={"name": "Shop", "dataset_id": version["payload"]["dataset_id"]},
    files={
      "file": (
        "sales.csv",
        b"date,sales,promo\n2026-01-01,2,0\n2026-01-02,3,1\n2026-01-03,4,0\n2026-01-04,,1\n",
      )
    },
  )
  assert duplicate.json()["id"] == version["id"]


def test_draft_stale_tab_does_not_overwrite(client):
  created = client.post(
    "/api/v1/drafts", json={"name": "Forecast", "payload": {"spec": {"horizon": 2}}}
  ).json()
  uri = f"/api/v1/drafts/{created['id']}"
  update = {
    "revision": 1,
    "payload": {"spec": {"horizon": 3}},
    "name": "Updated forecast",
  }
  assert client.patch(uri, json=update, headers={"If-Match": '"1"'}).status_code == 200
  update["payload"]["spec"]["horizon"] = 99
  assert client.patch(uri, json=update).status_code == 409
  assert client.get(uri).json()["payload"]["spec"]["horizon"] == 3
  assert client.get(uri).json()["name"] == "Updated forecast"


def test_submission_idempotency_and_cross_workspace_guard(client):
  version = upload(client)
  body = {"kind": "forecast", "spec": spec(version)}
  first = client.post("/api/v1/jobs", json=body, headers={"Idempotency-Key": "click1"})
  assert first.status_code == 202, first.text
  second = client.post("/api/v1/jobs", json=body, headers={"Idempotency-Key": "click1"})
  assert second.json()["id"] == first.json()["id"]
  body["spec"]["settings"]["horizon"] = 2
  assert (
    client.post(
      "/api/v1/jobs", json=body, headers={"Idempotency-Key": "click1"}
    ).status_code
    == 409
  )
  other = client.post("/api/v1/workspaces", json={"name": "Other"}).json()
  body["workspace_id"] = other["id"]
  assert (
    client.post(
      "/api/v1/jobs", json=body, headers={"Idempotency-Key": "click2"}
    ).status_code
    == 422
  )


def test_group_roles_and_foreign_origin_are_rejected(client):
  version = upload(client)
  body = spec(version)
  body["preparation"] = {"group_columns": ["sales"]}
  assert client.post("/api/v1/preview", json=body).status_code == 422
  assert (
    client.post(
      "/api/v1/datasets/demo", headers={"Origin": "https://unrelated.example"}
    ).status_code
    == 403
  )
  assert (
    client.get("/api/v1/health", headers={"Host": "unrelated.example"}).status_code
    == 400
  )


def test_result_pagination_and_chart_negative_values(client):
  store, artifacts = client.app.state.store, client.app.state.artifacts
  forecast = pd.DataFrame(
    {
      "dataset": ["a"] * 4,
      "target": ["sales"] * 4,
      "step": [1, 2, 3, 4],
      "timestamp": [1, 2, 3, 4],
      "point": [-3, -2, -1, 0],
      "q0.1": [-5, -4, -3, -2],
      "q0.9": [-1, 0, 1, 2],
    }
  )
  descriptor = artifacts.put_frame("test/forecast.parquet", forecast)
  run = store.create_record(
    "run", "Result", {"kind": "forecast", "tables": {"forecast": descriptor}}
  )
  uri = f"/api/v1/runs/{run['id']}"
  page = client.get(
    uri + "/tables/forecast?offset=1&limit=2&sort_by=point&descending=true"
  ).json()
  assert page["total"] == 4
  assert [row["point"] for row in page["rows"]] == [-1, -2]
  chart = client.get(uri + "/chart").json()
  assert chart["forecast"][0]["q0.1"] == -5
  selected_step = client.get(uri + "/tables/forecast?step=3").json()
  assert selected_step["total"] == 1
  assert selected_step["rows"][0]["point"] == -1
  assert client.get(uri + "/tables/forecast?limit=1001").status_code == 422


def test_preview_and_cancel_terminal_event_stream(client):
  version = upload(client)
  response = client.post("/api/v1/preview", json=spec(version))
  assert response.status_code == 200, response.text
  assert "quality" in response.json()
  response = client.post(
    "/api/v1/jobs", json={"spec": spec(version)}, headers={"Idempotency-Key": "cancel"}
  )
  job = response.json()
  assert client.post(f"/api/v1/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
  events = client.get(f"/api/v1/jobs/{job['id']}/events")
  assert "event: snapshot" in events.text
  assert '"status": "cancelled"' in events.text


def test_real_result_export_is_downloaded_without_reencoding(client):
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w") as archive:
    archive.writestr("forecast.csv", "point\n1\n")
  artifact = client.app.state.artifacts.put_bytes("test/export.zip", buffer.getvalue())
  run = client.app.state.store.create_record("run", "Export", {"export": artifact})
  response = client.get(f"/api/v1/runs/{run['id']}/export")
  assert response.content == buffer.getvalue()
  assert response.headers["content-type"] == "application/zip"


def test_retention_defaults_and_explicit_tracking_stop(client):
  policy = client.get("/api/v1/workspaces/local/retention").json()
  assert policy["enabled"] is False
  assert (
    client.patch(
      "/api/v1/workspaces/local/retention", json={"enabled": True, "max_runs": 0}
    ).status_code
    == 422
  )
  saved = client.patch(
    "/api/v1/workspaces/local/retention", json={"enabled": True, "max_runs": 10}
  )
  assert saved.status_code == 200
  assert client.get("/api/v1/workspaces/local/retention").json()["max_runs"] == 10
  assert client.get("/api/v1/workspaces/local/retention/preview").status_code == 200
  run = client.app.state.store.create_record(
    "run", "Issued forecast", {"kind": "forecast"}
  )
  tracked = client.post(
    "/api/v1/tracking", json={"name": "Watch", "payload": {"run_id": run["id"]}}
  )
  assert tracked.status_code == 201
  tracking_id = tracked.json()["id"]
  assert client.delete("/api/v1/tracking/" + tracking_id).status_code == 204
  assert client.get("/api/v1/runs/" + run["id"]).status_code == 200
  assert client.get("/api/v1/tracking").json() == []
  assert client.get("/api/v1/activity").status_code == 200


def test_missing_artifact_is_recoverable_and_does_not_expose_path(client):
  run = client.app.state.store.create_record(
    "run", "Missing export", {"export": {"key": "missing/export.zip"}}
  )
  response = client.get("/api/v1/runs/" + run["id"] + "/export")
  assert response.status_code == 410
  assert "missing/export.zip" not in response.text


def test_expired_worker_is_not_reported_as_available(client):
  # Staleness is derived from heartbeat_at at read time, not persisted: the
  # stored record still says "idle" (last assertion), but the API response
  # reports "offline"/unavailable because the heartbeat is far too old to
  # be trusted.
  store = client.app.state.store
  worker = store.create_record(
    "worker",
    "Old worker",
    {
      "status": "idle",
      "heartbeat_at": "2020-01-01T00:00:00+00:00",
      "vram_free_gb": 8,
    },
  )
  payload = client.get("/api/v1/workers").json()[0]["payload"]
  assert payload["status"] == "offline"
  assert payload["available"] is False
  assert payload["vram_free_gb"] is None
  assert store.get_record(worker["id"])["payload"]["status"] == "idle"
