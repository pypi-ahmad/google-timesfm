# TimesFM-3 Documentation

Choose the shortest path for what you need to do.

The React workbench is the new native Windows UI. Existing Explorer tutorials
describe the retained Streamlit application.

## Start here

- New React app: [run the native workbench](how-to/native-workbench.md).
- Workbench internals: [React + FastAPI architecture](explanation/workbench-architecture.md).
- Migration checks: [workbench validation](reference/workbench-validation.md).
- Learning the model: [read the complete TimesFM-3 guide](timesfm3-guide.md).
- New to the app: [complete your first forecast](tutorials/first-forecast.md).
- Run a first native forecast: [follow the workbench tutorial](tutorials/first-forecast.md).
- Bringing your own file: [prepare CSV or Parquet data](how-to/prepare-data.md).
- Integrating Python: [use the TimesFM-3 API](how-to/use-python-api.md).
- Something failed: [open the troubleshooting guide](troubleshooting.md).

## Tutorial

- [First forecast](tutorials/first-forecast.md): launch the native workbench,
  run the demo, read the chart, and export a bundle.

## How-to guides

- [Run the native workbench](how-to/native-workbench.md)
- [Prepare data](how-to/prepare-data.md)
- [Use the TimesFM-3 Python API](how-to/use-python-api.md)
- [Use the legacy Streamlit Explorer](how-to/use-streamlit-explorer.md)
- [Use the legacy Explorer handbook](how-to/timesfm3-explorer-handbook.md)
- [Use the archived TimesFM 2.5 CSV helper](how-to/use-csv-helper.md)

## Reference

- [Workbench API reference](reference/workbench-api.md)
- [Legacy Explorer reference](reference/explorer.md)
- [Python API reference](reference/python-api.md)

## Explanation

- [Learn Google TimesFM-3](timesfm3-guide.md)
- [React + FastAPI workbench architecture](explanation/workbench-architecture.md)
- [Legacy Explorer architecture](explanation/architecture.md)
- [Licensing and model versions](explanation/licensing-and-versions.md)

## Operations and development

- [Troubleshooting](troubleshooting.md)
- [Contributing](../CONTRIBUTING.md)
- [Code review findings](../REVIEW.md)
- [Codebase architecture](codebase/ARCHITECTURE.md)
- [Codebase structure](codebase/STRUCTURE.md)
- [Technology stack](codebase/STACK.md)
- [Testing patterns](codebase/TESTING.md)
- [Coding conventions](codebase/CONVENTIONS.md)
- [External integrations](codebase/INTEGRATIONS.md)
- [Known concerns](codebase/CONCERNS.md)

## Research knowledge

The [OKF knowledge index](../knowledge/index.md) contains research reports,
curated concepts, and primary-source extracts. Those entries are currently
draft and unverified. For runtime behavior, current source and tests take
precedence.

## Documentation coverage

| Capability | Tutorial | How-to | Reference or explanation |
|---|---:|---:|---:|
| Install and launch | Yes | Yes | Troubleshooting |
| Demo and file upload | Yes | Yes | Workbench API reference |
| Targets and covariates | Yes | Yes | Workbench and Python API reference |
| Forecast and holdout modes | Yes | Yes | Workbench API reference |
| Bulk series, readiness, and calendar features | No | Yes | Workbench API reference |
| Backtesting, anomalies, scenarios, and usefulness comparisons | No | Yes | Workbench API reference |
| Settings, baselines, and interval calibration | No | Yes | Workbench API reference |
| Tracking, refresh, and checkpoint selection | No | Yes | Workbench API reference |
| Quantiles and exports | Yes | Yes | Workbench API reference |
| Python integration | No | Yes | Python API reference |
| Archived CSV helper | No | Yes | Version explanation |
| Architecture and contribution | No | No | Explanation and codebase maps |
| Licensing and limitations | Yes | Yes | Licensing explanation |
| TimesFM-3 concepts and internals | Yes | No | Complete TimesFM-3 guide |
