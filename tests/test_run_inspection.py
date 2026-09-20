"""Saved-run inspection has no inference side effects or fabricated measurements."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from timesfm_app.api import clean
from timesfm_app.run_inspection import (
  last_value_baseline,
  replay_script,
  snapshot_batch,
  summarize,
)
from timesfm_app.schemas import JobSubmission, RunSpec


def test_snapshot_preserves_short_events_and_never_includes_future_labels():
  values = np.zeros(3002)
  values[1] = 1
  series = SimpleNamespace(
    dataset_id="shop",
    target_names=["sales"],
    context=np.arange(3000, dtype=float)[None, :],
    past_only=None,
    past_future=values[None, :],
    history_time=list(range(3000)),
    future_time=[3000, 3001],
    lineage=[],
  )
  batch = SimpleNamespace(
    settings=SimpleNamespace(mode="multivariate"),
    chunking=SimpleNamespace(
      selected_past_only=(), selected_past_future=("calendar_event:sale",)
    ),
    series=[series],
    mapping=SimpleNamespace(past_only=[], past_future=["calendar_event:sale"]),
  )
  rows, windows, events = snapshot_batch(batch, "scenario", 3000)
  assert windows[0]["sampled"]
  assert windows[0]["context_shape"] == [1, 3000]
  assert events[0]["start"] == events[0]["end"] == 1
  assert not any(row["role"] == "target" and row["phase"] == "future" for row in rows)
  assert len([row for row in rows if row["role"] == "target"]) == 2000
  assert last_value_baseline(batch).point.tolist() == [2999, 2999]
  json.dumps(
    clean({"rows": rows, "windows": windows, "events": events}), allow_nan=False
  )


def test_summary_matches_keys_instead_of_positions_and_handles_zero_baseline():
  frame = pd.DataFrame(
    [
      {
        "dataset": "shop",
        "target": "sales",
        "variant": v,
        "origin": 10,
        "step": s,
        "timestamp": s,
        "point": p,
      }
      for v, s, p in [
        ("base", 2, 0),
        ("scenario", 1, 3),
        ("base", 1, 0),
        ("scenario", 2, 5),
        ("scenario", 3, 9),
      ]
    ]
  )
  result = summarize(frame, pd.DataFrame(), "shop", "sales", "scenario", "base")
  assert result["metrics"] is None
  assert result["delta"]["total"] == 8
  assert result["delta"]["percent"] is None
  assert not result["delta"]["complete"]
  assert result["delta"]["matched_steps"] == 2


def test_saved_scores_use_overall_scope_and_scored_period():
  frame = pd.DataFrame(
    {
      "dataset": ["a"] * 3,
      "target": ["y"] * 3,
      "variant": ["model"] * 3,
      "origin": [1, 2, 3],
      "timestamp": [1, 2, 3],
      "point": [2.0, 5.0, 9.0],
      "actual": [1.0, 3.0, 1.0],
      "scored": [True, True, False],
    }
  )
  scores = pd.DataFrame(
    [
      {
        "dataset": "a",
        "target": "y",
        "variant": "model",
        "scope": scope,
        "mae": value,
        "observations": 2,
      }
      for scope, value in [("window", 1), ("overall", 1.5)]
    ]
  )
  result = summarize(frame, scores, "a", "y", "model")
  assert result["metrics"]["mae"] == 1.5
  assert result["period_end"] == 2
  assert result["scope"] == "all_evaluated_windows"


def test_replay_is_valid_public_submission_and_does_not_leak_internal_fields():
  record = {
    "workspace_id": "local",
    "payload": {
      "kind": "forecast",
      "spec": {
        "dataset_version_ids": ["version"],
        "mapping": {"targets": ["sales"]},
        "_device": "secret-internal",
        "_resolved_model": {"path": "private"},
      },
      "manifest": {"model_provenance": {"resolved_revision": "fixed-sha"}},
    },
  }
  code = replay_script(record)
  assert "secret-internal" not in code and "private" not in code
  captured = []

  class Response:
    def __enter__(self):
      return self

    def __exit__(self, *args):
      pass

    def read(self):
      return b'{"id":"new-job"}'

  def send(request):
    captured.append(json.loads(request.data))
    return Response()

  with patch("urllib.request.urlopen", send):
    exec(compile(code, "replay_run.py", "exec"), {})  # noqa: S102 -- generated code under test; HTTP is mocked
  submission = JobSubmission.model_validate(captured[0])
  assert submission.spec.model.revision == "fixed-sha"


def test_disabled_signal_validation_and_scenario_conflict():
  base = {"mapping": {"targets": ["sales"], "past_future": ["promo"]}}
  RunSpec.model_validate({**base, "disabled_covariates": ["promo"]})
  for disabled in (["missing"], ["promo", "promo"]):
    with pytest.raises(ValidationError):
      RunSpec.model_validate({**base, "disabled_covariates": disabled})
  with pytest.raises(ValidationError, match="Remove scenario overrides"):
    RunSpec.model_validate(
      {
        **base,
        "disabled_covariates": ["promo"],
        "scenarios": [
          {
            "name": "sale",
            "overrides": [
              {"dataset": "shop", "row": 5, "covariate": "promo", "value": 1}
            ],
          }
        ],
      }
    )
