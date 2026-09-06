"""Immutable artifact files and an optional S3 adapter."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .store import Conflict

_OBJECT_COLUMNS = b"timesfm_app.temporal_object_columns.v1"
_WINDOWS_RESERVED = {
  "CON",
  "PRN",
  "AUX",
  "NUL",
  *(f"COM{i}" for i in range(1, 10)),
  *(f"LPT{i}" for i in range(1, 10)),
}


def _key(key: str) -> str:
  if not key or "\\" in key or ":" in key or "\x00" in key:
    raise ValueError("Artifact keys must be relative POSIX paths.")
  parts = key.split("/")
  if any(
    part in {"", ".", ".."}
    or part.endswith((".", " "))
    or part.split(".")[0].upper() in _WINDOWS_RESERVED
    or any(ord(char) < 32 or char in '<>"|?*' for char in part)
    for part in parts
  ):
    raise ValueError("Artifact key contains an unsafe path component.")
  if PurePosixPath(key).is_absolute():
    raise ValueError("Artifact keys must be relative paths.")
  return key


def _descriptor(key: str, data: bytes) -> dict[str, Any]:
  return {"key": key, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _resolved(path: Path) -> Path:
  resolved = str(path.resolve())
  # Windows realpath can retain the extended prefix when another writer creates
  # a missing parent concurrently. Normalize both sides of the containment check.
  if os.name == "nt":
    if resolved.startswith("\\\\?\\UNC\\"):
      resolved = "\\\\" + resolved[8:]
    elif resolved.startswith("\\\\?\\"):
      resolved = resolved[4:]
  return Path(resolved)


def _temporal(value: Any) -> bool:
  return isinstance(value, (pd.Timestamp, datetime, date, np.datetime64))


def _encode_cell(value: Any) -> str:
  if value is pd.NaT:
    item = ["nat", None]
  elif value is pd.NA:
    item = ["na", None]
  elif isinstance(value, date) and not isinstance(value, datetime):
    item = ["date", value.isoformat()]
  elif _temporal(value):
    stamp = pd.Timestamp(value)
    item = ["timestamp", stamp.isoformat(), str(stamp.tz) if stamp.tz else None]
  else:
    if isinstance(value, np.generic):
      value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
      item = ["float", str(value)]
    else:
      item = ["value", value]
  return json.dumps(item, allow_nan=False)


def _decode_cell(value: str) -> Any:
  item = json.loads(value)
  if item[0] == "timestamp":
    stamp = pd.Timestamp(item[1])
    return stamp.tz_convert(item[2]) if item[2] is not None else stamp
  if item[0] == "date":
    return date.fromisoformat(item[1])
  if item[0] == "nat":
    return pd.NaT
  if item[0] == "na":
    return pd.NA
  if item[0] == "float":
    return float(item[1])
  return item[1]


def _frame_bytes(frame: pd.DataFrame) -> bytes:
  columns = [
    column
    for column in frame
    if pd.api.types.is_object_dtype(frame[column])
    and frame[column].map(_temporal).any()
  ]
  serializable = frame.copy()
  for column in columns:
    serializable[column] = frame[column].map(_encode_cell)
  table = pa.Table.from_pandas(serializable, preserve_index=False)
  metadata = dict(table.schema.metadata or {})
  metadata[_OBJECT_COLUMNS] = json.dumps(columns).encode()
  table = table.replace_schema_metadata(metadata)
  buffer = BytesIO()
  pq.write_table(table, buffer)
  return buffer.getvalue()


def _read_frame(data: bytes) -> pd.DataFrame:
  table = pq.read_table(BytesIO(data))
  frame = table.to_pandas()
  metadata = table.schema.metadata or {}
  for column in json.loads(metadata.get(_OBJECT_COLUMNS, b"[]")):
    frame[column] = pd.Series(
      [_decode_cell(value) for value in frame[column]], dtype=object
    )
  return frame


class ArtifactStore:
  """Publish complete immutable files beneath one resolved local directory."""

  def __init__(self, root: str | Path) -> None:
    self.root = _resolved(Path(root))
    self.root.mkdir(parents=True, exist_ok=True)

  def _path(self, key: str) -> Path:
    path = self.root.joinpath(*_key(key).split("/"))
    if not _resolved(path).is_relative_to(self.root):
      raise ValueError("Artifact path escapes the configured directory.")
    return path

  def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
    path = self._path(key)
    descriptor = _descriptor(key, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    self._path(key)  # Check resolved parents after creating directories.
    if path.exists():
      if path.read_bytes() != data:
        raise Conflict("Artifact keys are immutable; use a new attempt path.")
      return descriptor
    temporary: Path | None = None
    try:
      with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".pending-", delete=False
      ) as file:
        temporary = Path(file.name)
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
      try:
        if os.name == "nt":
          os.rename(temporary, path)  # Windows rename refuses an existing destination.
        else:
          os.link(temporary, path)  # Atomic no-clobber publication on POSIX.
      except FileExistsError:
        if path.read_bytes() != data:
          raise Conflict(
            "Artifact keys are immutable; use a new attempt path."
          ) from None
      return descriptor
    finally:
      if temporary is not None:
        temporary.unlink(missing_ok=True)

  def get_bytes(self, key: str) -> bytes:
    return self._path(key).read_bytes()

  def delete(self, key: str) -> None:
    self._path(key).unlink(missing_ok=True)

  def put_frame(self, key: str, frame: pd.DataFrame) -> dict[str, Any]:
    return self.put_bytes(key, _frame_bytes(frame))

  def get_frame(self, key: str) -> pd.DataFrame:
    return _read_frame(self.get_bytes(key))


class S3ArtifactStore:
  """S3 conditional puts retain the same immutable-key contract as local files."""

  def __init__(self, bucket: str, endpoint_url: str | None = None) -> None:
    import boto3

    if not bucket:
      raise ValueError("An S3 bucket is required.")
    self.bucket = bucket
    self.client = boto3.client("s3", endpoint_url=endpoint_url)

  def put_bytes(self, key: str, data: bytes) -> dict[str, Any]:
    from botocore.exceptions import ClientError

    key = _key(key)
    descriptor = _descriptor(key, data)
    try:
      self.client.put_object(
        Bucket=self.bucket,
        Key=key,
        Body=data,
        IfNoneMatch="*",
        Metadata={"sha256": descriptor["sha256"]},
        ChecksumSHA256=base64.b64encode(hashlib.sha256(data).digest()).decode(),
      )
    except ClientError as exc:
      if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
        raise
      if self.get_bytes(key) != data:
        raise Conflict("Artifact keys are immutable; use a new attempt path.") from None
    return descriptor

  def get_bytes(self, key: str) -> bytes:
    response = self.client.get_object(Bucket=self.bucket, Key=_key(key))
    with response["Body"] as body:
      data = body.read()
    expected = response.get("Metadata", {}).get("sha256")
    if expected and hashlib.sha256(data).hexdigest() != expected:
      raise OSError("Artifact checksum does not match its stored checksum.")
    return data

  def delete(self, key: str) -> None:
    self.client.delete_object(Bucket=self.bucket, Key=_key(key))

  def put_frame(self, key: str, frame: pd.DataFrame) -> dict[str, Any]:
    return self.put_bytes(key, _frame_bytes(frame))

  def get_frame(self, key: str) -> pd.DataFrame:
    return _read_frame(self.get_bytes(key))


def make_artifact_store(settings: Any) -> ArtifactStore | S3ArtifactStore:
  if settings.storage_backend == "local":
    return ArtifactStore(settings.artifact_root)
  if settings.storage_backend == "s3":
    return S3ArtifactStore(settings.s3_bucket, settings.s3_endpoint_url)
  raise ValueError("Unsupported artifact storage backend.")
