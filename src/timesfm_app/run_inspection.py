"""Saved-input displays and summaries, independent of model loading."""

import json

import numpy as np
import pandas as pd


def snapshot_batch(batch, variant="", origin=None):
  """Capture pre-interpolation inputs; never expose held-out targets as inputs."""
  rows, windows, events = [], [], []
  used_signals = (
    set(batch.chunking.selected_past_only + batch.chunking.selected_past_future)
    if batch.settings.mode != "univariate"
    else set()
  )
  for series in batch.series:
    history = len(series.history_time)
    horizon = len(series.future_time)
    common = {"dataset": series.dataset_id, "variant": variant, "origin": origin}
    signals = [
      (name, "target", series.context[i]) for i, name in enumerate(series.target_names)
    ]
    for role, names, array in (
      ("past_only", batch.mapping.past_only, series.past_only),
      ("known_future", batch.mapping.past_future, series.past_future),
    ):
      if array is not None:
        signals.extend((name, role, array[i]) for i, name in enumerate(names))
    axis = list(series.history_time) + list(series.future_time)
    positions = list(np.linspace(0, history - 1, min(history, 2000), dtype=int)) + list(
      range(history, history + horizon)
    )
    windows.append(
      {
        **common,
        "context_length": history,
        "horizon": horizon,
        "context_shape": list(series.context.shape),
        "past_only_shape": list(series.past_only.shape)
        if series.past_only is not None
        else None,
        "past_future_shape": list(series.past_future.shape)
        if series.past_future is not None
        else None,
        "sampled": history > 2000,
        "lineage": list(series.lineage),
        "signals": [
          {
            "signal": name,
            "role": role,
            "missing": int(np.isnan(values).sum()),
            "used": role == "target" or name in used_signals,
          }
          for name, role, values in signals
        ],
      }
    )
    for name, role, values in signals:
      if name.startswith("calendar_event:") or name == "calendar_holiday":
        active = np.flatnonzero(np.isfinite(values) & (values != 0))
        for block in np.split(active, np.where(np.diff(active) != 1)[0] + 1):
          if len(block):
            events.append(
              {
                **common,
                "signal": name,
                "start": axis[int(block[0])],
                "end": axis[int(block[-1])],
              }
            )
      for position in positions:
        if position >= len(values):
          continue
        rows.append(
          {
            **common,
            "signal": name,
            "role": role,
            "position": int(position),
            "phase": "history" if position < history else "future",
            "timestamp": axis[position],
            "value": float(values[position]),
          }
        )
  return rows, windows, events


def last_value_baseline(batch):
  rows = []
  for series in batch.series:
    for i, target in enumerate(series.target_names):
      finite = series.context[i][np.isfinite(series.context[i])]
      value = float(finite[-1]) if len(finite) else np.nan
      rows.extend(
        {
          "dataset": series.dataset_id,
          "target": target,
          "variant": "last_value",
          "step": step + 1,
          "timestamp": timestamp,
          "point": value,
        }
        for step, timestamp in enumerate(series.future_time)
      )
  return pd.DataFrame(rows)


def summarize(predictions, metrics, dataset, target, variant, reference=None):
  """Use saved overall scores and complete forecast pairs, not display samples."""

  def selected(frame, chosen=variant):
    for key, value in (("dataset", dataset), ("target", target), ("variant", chosen)):
      if value is not None and key in frame:
        frame = frame.loc[frame[key].astype(str).eq(value)]
    return frame

  scores = selected(metrics)
  if "scope" in scores:
    scores = scores.loc[scores.scope.eq("overall")]
  forecast = selected(predictions)
  actuals = forecast
  if "actual" in actuals:
    actuals = actuals.loc[np.isfinite(actuals.actual) & np.isfinite(actuals.point)]
    if "scored" in actuals:
      actuals = actuals.loc[actuals.scored]
  else:
    actuals = actuals.iloc[:0]
  result = {
    "metrics": scores.iloc[0].to_dict() if len(scores) == 1 else None,
    "period_start": actuals.timestamp.min()
    if len(actuals) and "timestamp" in actuals
    else None,
    "period_end": actuals.timestamp.max()
    if len(actuals) and "timestamp" in actuals
    else None,
    "scope": "all_evaluated_windows" if "origin" in forecast else "holdout",
    "delta": None,
  }
  if reference and "variant" in predictions and reference != variant:
    base = selected(predictions, reference)
    keys = [
      key
      for key in ("dataset", "target", "origin", "step", "timestamp")
      if key in predictions
    ]
    if not forecast.duplicated(keys).any() and not base.duplicated(keys).any():
      pairs = forecast.merge(
        base[keys + ["point"]],
        on=keys,
        suffixes=("", "_baseline"),
        validate="one_to_one",
      )
      pairs = pairs.loc[np.isfinite(pairs.point) & np.isfinite(pairs.point_baseline)]
      if len(pairs):
        difference = pairs.point - pairs.point_baseline
        total = float(pairs.point_baseline.sum())
        result["delta"] = {
          "total": float(difference.sum()),
          "percent": float(difference.sum() / abs(total) * 100) if total else None,
          "largest_increase": float(difference.max()),
          "largest_decrease": float(difference.min()),
          "matched_steps": len(pairs),
          "complete": len(pairs) == len(forecast) == len(base),
          "reference": reference,
        }
  return result


def replay_script(record):
  from .schemas import RunSpec

  source = record["payload"].get("spec", {})
  public = {key: value for key, value in source.items() if key in RunSpec.model_fields}
  public["kind"] = record["payload"]["kind"]
  provenance = record["payload"].get("manifest", {}).get("model_provenance", {})
  if (
    provenance.get("resolved_revision")
    and public.get("model", {}).get("kind", "hub") == "hub"
  ):
    public["model"] = {
      **public.get("model", {}),
      "revision": provenance["resolved_revision"],
    }
  body = {
    "workspace_id": record["workspace_id"],
    "kind": record["payload"]["kind"],
    "spec": RunSpec.model_validate(public).model_dump(mode="json"),
  }
  return "\n".join(
    [
      "# Creates a NEW job. Requires this workbench, original dataset versions and checkpoint.",
      "import json",
      "import uuid",
      "from urllib.request import Request, urlopen",
      "",
      f"body = json.loads({json.dumps(body)!r})",
      "request = Request('http://127.0.0.1:8001/api/v1/jobs',",
      "    data=json.dumps(body).encode(), method='POST',",
      "    headers={'Content-Type': 'application/json', 'Idempotency-Key': str(uuid.uuid4())})",
      "with urlopen(request) as response:",
      "    print(json.load(response)['id'])",
      "",
    ]
  )
