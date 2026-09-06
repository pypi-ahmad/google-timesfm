---
name: timesfm-forecasting
description: Archived TimesFM 2.5 compatibility material retained with this repository.
license: Apache-2.0
metadata:
  status: archived
  replacement: TimesFM-3 workbench and Python API documentation
---

# Archived TimesFM 2.5 compatibility material

This directory is retained for historical scripts, examples, and compatibility
references. It is not the recommended forecasting path for this repository.

Use the TimesFM-3 interfaces instead:

- [Run the native React workbench](../docs/how-to/native-workbench.md).
- [Complete a first workbench forecast](../docs/tutorials/first-forecast.md).
- [Use the TimesFM-3 Python API](../docs/how-to/use-python-api.md).
- [Read the TimesFM-3 API reference](../docs/reference/python-api.md).

The scripts and examples below continue to document the historical TimesFM 2.5
API only. Do not combine their model classes, quantile conventions, or XReg
interfaces with the TimesFM-3 workbench.

## Retained material

| Path | Historical purpose |
| --- | --- |
| `scripts/check_system.py` | TimesFM 2.5 machine preflight |
| `scripts/forecast_csv.py` | TimesFM 2.5 CSV helper |
| `examples/` | Historical 2.5 forecasting demonstrations |
| `references/` | Historical 2.5 input and API notes |

The default TimesFM-3 checkpoint has its own non-commercial,
non-production license. See
[Licensing and model versions](../docs/explanation/licensing-and-versions.md).
