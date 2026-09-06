"""Root-app refresh honors the pinned model and current offline policy."""

from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.test_streamlit_app import persisted_artifact
from timesfm3.explorer import ExplorerError
from timesfm3.model_loading import ModelSelection, ResolvedModel


def _tracking_submit(previous):
  def render(datasets, database, *, refresh_run, acknowledged):
    if st.button("Test saved refresh"):
      try:
        refresh_run(previous, datasets)
      except ExplorerError as exc:
        st.error(str(exc))

  return render


def test_refresh_uses_current_offline_flag_and_saved_revision():
  previous = persisted_artifact()
  previous.manifest["model_provenance"] = {
    "selection": {"source": "example/timesfm3", "kind": "hub", "offline": False},
    "resolved_revision": "a" * 40,
  }
  with (
    mock.patch(
      "timesfm3.tracking_ui.render_tracking", side_effect=_tracking_submit(previous)
    ),
    mock.patch("timesfm3.run_store.load_recent_runs", return_value=[]),
    mock.patch(
      "timesfm3.model_loading.resolve_model",
      side_effect=ExplorerError("Resolver reached"),
    ) as resolve,
    mock.patch("timesfm3.model_loading.load_resolved_model") as load,
  ):
    app = AppTest.from_file(
      str(Path(__file__).parents[1] / "streamlit_app.py"), default_timeout=20
    ).run()
    next(item for item in app.checkbox if item.label == "Offline loading").set_value(
      True
    ).run()
    resolve.assert_not_called()
    next(
      item for item in app.button if item.label == "Test saved refresh"
    ).click().run()
    assert not app.exception
    resolve.assert_called_once()
    selection = resolve.call_args.args[0]
    assert selection.offline is True
    assert selection.source == "example/timesfm3"
    assert selection.revision == "a" * 40
    load.assert_not_called()


def test_changed_local_checkpoint_is_rejected_before_model_loading():
  previous = persisted_artifact()
  selection = ModelSelection("saved.pth", "local")
  previous.manifest["model_provenance"] = {
    "selection": {"source": selection.source, "kind": "local"},
    "files": {"saved.pth": "original"},
  }
  resolved = ResolvedModel(selection, "saved.pth", None, (("saved.pth", "changed"),))
  with (
    mock.patch(
      "timesfm3.tracking_ui.render_tracking", side_effect=_tracking_submit(previous)
    ),
    mock.patch("timesfm3.run_store.load_recent_runs", return_value=[]),
    mock.patch("timesfm3.model_loading.resolve_model", return_value=resolved),
    mock.patch("timesfm3.model_loading.load_resolved_model") as load,
  ):
    app = AppTest.from_file(
      str(Path(__file__).parents[1] / "streamlit_app.py"), default_timeout=20
    ).run()
    next(
      item for item in app.button if item.label == "Test saved refresh"
    ).click().run()
    assert not app.exception
    assert any("checkpoint changed" in item.value for item in app.error)
    load.assert_not_called()
