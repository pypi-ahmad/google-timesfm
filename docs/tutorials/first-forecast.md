# Tutorial: Your first TimesFM-3 workbench forecast

This tutorial takes you from a local checkout to a saved TimesFM-3 forecast in
the React workbench. It uses the included demand demo, so you do not need a
file of your own.

**Time:** about ten minutes, plus the first checkpoint download.

> [!IMPORTANT]
> The default TimesFM-3 weights have a separate non-commercial,
> non-production license. A polished local workbench does not change those
> terms.

## Before you begin

Use Windows 11 with Git, Node.js 24 or newer, and
[uv](https://docs.astral.sh/uv/getting-started/installation/). Clone this
repository, then double-click `launch_workbench.cmd`. It prepares missing
components on its first run, opens the workbench, and keeps service logs in the
launcher terminal.

Open <http://localhost:3000> when the launcher prints `Workbench ready`.

## 1. Create demo data

1. Open **Data** from the navigation.
2. Select **Load demo data**.
3. Select the new dataset version when the page returns to the library.

Dataset versions are immutable. A later upload to the same logical dataset
creates another version instead of changing an earlier forecast's source data.

## 2. Configure the forecast

1. Open **Forecasts**.
2. Select the demo version in **Dataset versions**.
3. Set `date` as the timestamp and `demand` as the target.
4. Keep the default multivariate mode, horizon, context, and quantile settings.
5. Select **Preview data quality** and resolve any reported issue.
6. Select **Run forecast**.

The workbench saves this configuration as a revisioned draft. You can navigate
away while the job waits for a worker or runs.

## 3. Read the result

When the job completes, the centre panel shows observed history, the forecast,
and nested 20%, 40%, 60%, and 80% intervals. Narrower bands sit inside wider
ones. A band is a model quantile interval, not a guarantee of future coverage.

Use the result controls to choose a target, dataset, or variant. The result
tables retain every returned quantile, while the chart keeps gaps and crossed
intervals visible rather than repairing them.

## 4. Save and export

The completed job publishes an immutable saved run. Select **Export run bundle**
from run details to download its tables, configuration, and provenance. Select
**Copy run settings** to begin another draft with the same settings.

## Next steps

- Bring your own data with [Prepare data](../how-to/prepare-data.md).
- Compare historical performance in [Run a native workbench](../how-to/native-workbench.md).
- Inspect API resources in the [workbench API reference](../reference/workbench-api.md).
- Use the [legacy Streamlit Explorer guide](../how-to/use-streamlit-explorer.md)
  only when you need the retained diagnostic interface.
