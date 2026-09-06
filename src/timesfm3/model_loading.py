"""Resolve reproducible TimesFM-3 checkpoints without implicit fallbacks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from huggingface_hub import snapshot_download

from .explorer import CHECKPOINT_ID, QUANTILES, ExplorerError


@dataclasses.dataclass(frozen=True)
class ModelSelection:
  """Explicit model source; resolving is an execution-time operation."""

  source: str = CHECKPOINT_ID
  kind: Literal["hub", "local"] = "hub"
  revision: str | None = None
  offline: bool = False


@dataclasses.dataclass(frozen=True)
class ResolvedModel:
  """Content identity suitable for a heavyweight resource cache key."""

  selection: ModelSelection
  path: str
  revision: str | None
  fingerprints: tuple[tuple[str, str], ...]

  def provenance(self) -> dict[str, Any]:
    return {
      "selection": dataclasses.asdict(self.selection),
      "resolved_revision": self.revision,
      "local_path": self.path,
      "files": dict(self.fingerprints),
    }


def _digest(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_model(selection: ModelSelection) -> ResolvedModel:
  """Resolve and fingerprint selected TimesFM-3 assets.

  Offline selections consult only local files and cached Hub snapshots. Raises
  ``ExplorerError`` when required assets are unavailable or incompatible.
  """
  source = selection.source.strip()
  if not source or selection.kind not in {"hub", "local"}:
    raise ExplorerError("Select a Hugging Face repository or a local checkpoint.")
  revision = None
  try:
    if selection.kind == "hub":
      path = Path(
        snapshot_download(
          source,
          revision=selection.revision or None,
          local_files_only=selection.offline,
          allow_patterns=["config.json", "model.safetensors"],
        )
      )
      revision = path.name
    else:
      path = Path(source).expanduser().resolve(strict=True)
    if path.is_dir():
      config_path = path / "config.json"
      if not config_path.is_file():
        raise ExplorerError("A local model folder must contain config.json.")
      config = json.loads(config_path.read_text(encoding="utf-8"))
      if config.get("quantiles", list(QUANTILES)) != list(QUANTILES):
        raise ExplorerError("Checkpoint must expose TimesFM-3 q0.1 through q0.9.")
      weights = path / "model.safetensors"
      if not weights.is_file():
        raise ExplorerError("Checkpoint folders must contain model.safetensors.")
      files = [config_path, weights]
    else:
      if path.suffix.lower() not in {".safetensors", ".pt", ".pth"}:
        raise ExplorerError("Select a model folder, .safetensors, .pt, or .pth file.")
      files = [path]
    fingerprints = tuple((file.name, _digest(file)) for file in files)
  except ExplorerError:
    raise
  except Exception as exc:
    message = (
      "Checkpoint is unavailable in local files/cache."
      if selection.offline
      else "Could not resolve the selected checkpoint."
    )
    raise ExplorerError(message) from exc
  return ResolvedModel(selection, str(path), revision, fingerprints)


def load_resolved_model(resolved: ResolvedModel, device: str, batch_size: int):
  """Load one resolved checkpoint through the built-in TimesFM-3 evaluator.

  The returned evaluator carries immutable provenance for run manifests. Raises
  ``ExplorerError`` if the checkpoint cannot be loaded with compatible weights.
  """
  from .evaluator import TimesFM3Evaluator

  try:
    predictor: Any = TimesFM3Evaluator.from_pretrained(
      resolved.path,
      device=device,
      per_core_batch_size=batch_size,
      local_files_only=True,
    )
    if list(predictor.config.quantiles) != list(QUANTILES):
      raise ExplorerError("Checkpoint must expose TimesFM-3 q0.1 through q0.9.")
    _validate_folder_weights(predictor, Path(resolved.path))
  except ExplorerError:
    raise
  except Exception as exc:
    raise ExplorerError(
      "Checkpoint cannot be loaded as a compatible TimesFM-3 model."
    ) from exc
  predictor.model_provenance = resolved.provenance()
  return predictor


def _validate_folder_weights(predictor: Any, path: Path) -> None:
  """Hub mixin loading is non-strict by default; refuse missing/random parameters."""
  if not path.is_dir():
    return  # Standalone files already use strict load_state_dict in the core loader.
  expected = {
    name: tuple(value.shape) for name, value in predictor.model.state_dict().items()
  }
  from safetensors import safe_open

  with safe_open(path / "model.safetensors", framework="pt", device="cpu") as weights:
    names = weights.keys()
    actual = {name: tuple(weights.get_slice(name).get_shape()) for name in names}
  if actual != expected:
    raise ExplorerError("Checkpoint parameter names or shapes do not match TimesFM-3.")


def selection_from_provenance(provenance: dict[str, Any]) -> ModelSelection:
  """Restore a model selection and pin a Hub refresh to its issued revision."""
  selection = ModelSelection(**provenance.get("selection", {}))
  if selection.kind == "hub" and provenance.get("resolved_revision"):
    selection = dataclasses.replace(selection, revision=provenance["resolved_revision"])
  return selection
