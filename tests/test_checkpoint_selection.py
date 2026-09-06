"""Explicit checkpoint selection, offline resolution, and provenance."""

import json
from types import SimpleNamespace
from unittest import mock

import pytest

from timesfm3.explorer import QUANTILES, ExplorerError
from timesfm3.model_loading import (
  ModelSelection,
  load_resolved_model,
  resolve_model,
  selection_from_provenance,
)


def folder(tmp_path):
  (tmp_path / "config.json").write_text(json.dumps({"quantiles": list(QUANTILES)}))
  (tmp_path / "model.safetensors").write_bytes(b"test weights")
  return tmp_path


def test_hub_revision_is_resolved_and_reused(tmp_path):
  path = folder(tmp_path)
  with mock.patch(
    "timesfm3.model_loading.snapshot_download", return_value=str(path)
  ) as download:
    resolved = resolve_model(ModelSelection(revision="tag", offline=True))
  assert download.call_args.kwargs["revision"] == "tag"
  assert download.call_args.kwargs["local_files_only"] is True
  assert resolved.revision == path.name
  assert selection_from_provenance(resolved.provenance()).revision == path.name


@pytest.mark.parametrize("extension", [".safetensors", ".pt", ".pth"])
def test_local_files_are_fingerprinted_and_never_downloaded(tmp_path, extension):
  file = tmp_path / ("checkpoint" + extension)
  file.write_bytes(b"first")
  with mock.patch("timesfm3.model_loading.snapshot_download") as download:
    original = resolve_model(ModelSelection(str(file), "local", offline=True))
    file.write_bytes(b"second")
    updated = resolve_model(ModelSelection(str(file), "local", offline=True))
  download.assert_not_called()
  assert original != updated


def test_missing_offline_assets_do_not_fall_back():
  with (
    mock.patch(
      "timesfm3.model_loading.snapshot_download", side_effect=FileNotFoundError
    ) as download,
    pytest.raises(ExplorerError, match="local files/cache"),
  ):
    resolve_model(ModelSelection(offline=True))
  assert download.call_count == 1
  assert download.call_args.kwargs["local_files_only"] is True


def test_folder_rejects_incompatible_quantiles(tmp_path):
  folder(tmp_path)
  (tmp_path / "config.json").write_text('{"quantiles": [0.5]}')
  with pytest.raises(ExplorerError, match="q0.1"):
    resolve_model(ModelSelection(str(tmp_path), "local"))


def test_folder_requires_the_format_supported_by_local_hub_loader(tmp_path):
  (tmp_path / "config.json").write_text("{}")
  (tmp_path / "pytorch_model.bin").write_bytes(b"unsupported folder weights")
  with pytest.raises(ExplorerError, match="model.safetensors"):
    resolve_model(ModelSelection(str(tmp_path), "local"))


def test_resolved_loader_is_local_and_records_provenance(tmp_path):
  resolved = resolve_model(ModelSelection(str(folder(tmp_path)), "local"))
  predictor = SimpleNamespace(config=SimpleNamespace(quantiles=list(QUANTILES)))
  with (
    mock.patch(
      "timesfm3.evaluator.TimesFM3Evaluator.from_pretrained", return_value=predictor
    ) as loader,
    mock.patch("timesfm3.model_loading._validate_folder_weights"),
  ):
    result = load_resolved_model(resolved, "cpu", 2)
  assert loader.call_args.kwargs["local_files_only"] is True
  assert result.model_provenance == resolved.provenance()


def test_bad_weights_raise_without_default_checkpoint_retry(tmp_path):
  resolved = resolve_model(ModelSelection(str(folder(tmp_path)), "local"))
  with (
    mock.patch(
      "timesfm3.evaluator.TimesFM3Evaluator.from_pretrained", side_effect=RuntimeError
    ) as loader,
    pytest.raises(ExplorerError, match="compatible"),
  ):
    load_resolved_model(resolved, "cpu", 1)
  assert loader.call_count == 1


def test_folder_weights_must_match_all_parameters(tmp_path):
  import torch
  from safetensors.torch import save_file

  from timesfm3.model_loading import _validate_folder_weights

  model = torch.nn.Linear(2, 1)
  save_file(model.state_dict(), tmp_path / "model.safetensors")
  _validate_folder_weights(SimpleNamespace(model=model), tmp_path)
  save_file({"weight": model.weight}, tmp_path / "model.safetensors")
  with pytest.raises(ExplorerError, match="parameter names"):
    _validate_folder_weights(SimpleNamespace(model=model), tmp_path)
