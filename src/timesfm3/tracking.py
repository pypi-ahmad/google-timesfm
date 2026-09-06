"""Match updated observations to immutable issued forecast vintages."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import numbers
import zipfile
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .explorer import (
  ExplorerError,
  RunArtifact,
  UploadedDataset,
  _csv_safe,
  evaluation_metrics,
)
from .uncertainty import calibration_table


@dataclasses.dataclass(frozen=True)
class TrackingAssessment:
  """Derived matches and scores from one explicitly submitted upload version."""

  assessment_id: str
  run_id: str
  created_at: str
  fingerprint: str
  comparisons: pd.DataFrame
  metrics: pd.DataFrame
  manifest: dict[str, Any]


def _timestamps(values: pd.Series) -> tuple[pd.DatetimeIndex, bool]:
  if pd.api.types.is_numeric_dtype(values) or any(
    isinstance(value, numbers.Number) for value in values
  ):
    raise ExplorerError(
      "Tracking requires timestamps; row positions cannot be matched."
    )
  try:
    parsed = [pd.Timestamp(value) for value in values]
    if any(pd.isna(value) for value in parsed):
      raise ValueError("Missing timestamp")
    awareness = {value.tzinfo is not None for value in parsed}
    if len(awareness) > 1:
      raise ValueError("Mixed timezone conventions")
    aware = next(iter(awareness), False)
    normalized = pd.DatetimeIndex(pd.to_datetime(parsed, utc=aware))
  except (TypeError, ValueError, OverflowError) as exc:
    raise ExplorerError(
      "Tracking requires valid, consistently zoned timestamps."
    ) from exc
  return normalized, aware


def associated_datasets(
  run: RunArtifact,
  datasets: Sequence[UploadedDataset],
  associations: dict[str, str],
) -> list[UploadedDataset]:
  """Validate explicit old-to-current identities and select updated uploads.

  Associations are deliberately explicit because upload order is not a stable
  series identity. Raises ``ExplorerError`` for missing, unknown, or reused
  dataset identities.
  """
  previous = set(run.forecast["dataset"].astype(str))
  current = {dataset.dataset_id: dataset for dataset in datasets}
  if len(current) != len(datasets):
    raise ExplorerError("Uploaded dataset identities must be unique.")
  if not associations:
    raise ExplorerError("Associate at least one saved dataset with an updated dataset.")
  if not set(associations).issubset(previous):
    raise ExplorerError(
      "An association names a dataset absent from the saved forecast."
    )
  if not set(associations.values()).issubset(current):
    raise ExplorerError("An associated updated dataset is no longer uploaded.")
  if len(set(associations.values())) != len(associations):
    raise ExplorerError("Each saved dataset must match a distinct updated dataset.")
  selected = []
  for key in sorted(associations):
    dataset = current[associations[key]]
    frame = dataset.frame.copy(deep=False)
    frame.attrs = {**dataset.frame.attrs, "tracking_previous_dataset": key}
    selected.append(dataclasses.replace(dataset, frame=frame))
  return selected


def assess_run(
  run: RunArtifact,
  datasets: Sequence[UploadedDataset],
  associations: dict[str, str],
) -> TrackingAssessment:
  """Match immutable issued predictions to updated observations and score them.

  Exact timestamps, target names, and timezone conventions are required. The
  returned assessment preserves the issued forecast and records only derived
  comparisons and metrics.
  """
  associated_datasets(run, datasets, associations)
  timestamp_column = run.mapping.timestamp
  if timestamp_column is None:
    raise ExplorerError("Tracking requires a timestamp column in the saved forecast.")
  current = {dataset.dataset_id: dataset for dataset in datasets}
  matched_frames = []
  hashes = {}
  for old_id, new_id in sorted(associations.items()):
    uploaded = current[new_id]
    hashes[new_id] = uploaded.sha256
    if timestamp_column not in uploaded.frame:
      raise ExplorerError(f"Updated dataset '{new_id}' needs '{timestamp_column}'.")
    actual_times, actual_aware = _timestamps(uploaded.frame[timestamp_column])
    if actual_times.duplicated().any():
      raise ExplorerError(f"Updated dataset '{new_id}' has duplicate timestamps.")
    issued = run.forecast.loc[run.forecast.dataset == old_id].copy()
    issued_times, issued_aware = _timestamps(issued["timestamp"])
    if actual_aware != issued_aware:
      raise ExplorerError(
        "Saved and updated timestamps must use the same timezone convention."
      )
    issued["timestamp"] = issued_times
    if issued.duplicated(["target", "timestamp"]).any():
      raise ExplorerError("The saved forecast has ambiguous target timestamps.")
    if "actual" in issued:
      issued = issued.rename(columns={"actual": "actual_at_issue"})
    for target in issued.target.unique():
      if target not in uploaded.frame:
        raise ExplorerError(f"Updated dataset '{new_id}' needs target '{target}'.")
      values = pd.to_numeric(uploaded.frame[target], errors="coerce")
      invalid = uploaded.frame[target].notna() & values.isna()
      if invalid.any() or not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
        raise ExplorerError(f"Updated target '{target}' contains invalid observations.")
      actuals = pd.DataFrame({"timestamp": actual_times, "actual": values.to_numpy()})
      matched = issued.loc[issued.target == target].merge(
        actuals.dropna(subset=["actual"]),
        on="timestamp",
        how="inner",
        validate="one_to_one",
      )
      matched["updated_dataset"] = new_id
      matched["error"] = matched["point"] - matched["actual"]
      matched["absolute_error"] = matched["error"].abs()
      matched_frames.append(matched)
  comparisons = pd.concat(matched_frames, ignore_index=True)
  if comparisons.empty:
    raise ExplorerError(
      "The updated upload has no observed values at saved forecast timestamps."
    )
  comparisons = comparisons.sort_values(["dataset", "target", "step"]).reset_index(
    drop=True
  )
  manifest = {
    "schema_version": 1,
    "run_id": run.run_id,
    "associations": dict(sorted(associations.items())),
    "upload_hashes": hashes,
    "matched_observations": len(comparisons),
    "issued_observations": len(run.forecast),
    "timestamp_matching": "exact; timezone-aware values normalized to UTC",
  }
  fingerprint = hashlib.sha256(
    json.dumps(manifest, sort_keys=True).encode()
  ).hexdigest()
  return TrackingAssessment(
    assessment_id=f"assessment-{fingerprint[:20]}",
    run_id=run.run_id,
    created_at=pd.Timestamp.now(tz="UTC").isoformat(),
    fingerprint=fingerprint,
    comparisons=comparisons,
    metrics=evaluation_metrics(comparisons),
    manifest=manifest,
  )


def assessment_zip(assessment: TrackingAssessment) -> bytes:
  """Export one matched-actuals version, metrics, coverage, and provenance."""
  buffer = io.BytesIO()
  with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    archive.writestr(
      "assessment.json",
      json.dumps(
        {
          **assessment.manifest,
          "assessment_id": assessment.assessment_id,
          "created_at": assessment.created_at,
          "fingerprint": assessment.fingerprint,
        },
        indent=2,
      ),
    )
    archive.writestr(
      "comparisons.csv", _csv_safe(assessment.comparisons).to_csv(index=False)
    )
    archive.writestr("metrics.csv", _csv_safe(assessment.metrics).to_csv(index=False))
    archive.writestr(
      "calibration.csv",
      _csv_safe(calibration_table(assessment.comparisons)).to_csv(index=False),
    )
  return buffer.getvalue()
