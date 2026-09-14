# Legacy: Use the Streamlit Explorer

> [!NOTE]
> The React + FastAPI workbench is the primary local interface. Use this guide
> only for the retained Streamlit diagnostic client. Start with
> [the native workbench guide](native-workbench.md) for new work.

## Launch the app

```powershell
uv sync --extra torch --extra app --group dev
.\launch_app.cmd
```

Open <http://localhost:9587>. To launch directly, run:

```powershell
uv run streamlit run streamlit_app.py --server.port=9587
```

For one end-to-end path that connects preparation, evaluation, tracking, and
reproducibility, start with the [Explorer handbook](timesfm3-explorer-handbook.md).

## Load data

Open **Prepare**. Use **Demo** to verify the environment, or choose **Upload** for one or more CSV
or Parquet files. The files stay in the local Streamlit process and parsed data
is cached only for the current browser session.

Assign the common columns:

1. Choose a timestamp or use row numbers.
2. Select one or more targets.
3. Optionally select past-only covariates.
4. Optionally select past-and-future covariates.

See [Prepare data](prepare-data.md) for role definitions and future-row rules.

## Choose the task

- **Forecast future:** predict after the final observed target row.
- **Evaluate holdout:** reserve the final observed target rows, forecast them,
  and compute metrics.

Choose **Joint multivariate** to use cross-variate information and covariates.
Choose **Independent univariate** to forecast each target independently; this
mode does not use covariates.

## Configure inference

- **Horizon:** number of future or held-out steps.
- **Context length:** maximum historical steps supplied to the model.
- **Device:** CUDA when available, otherwise CPU.
- **Batch size:** datasets evaluated together per core operation.
- **Return quantiles:** add `q0.1` through `q0.9` predictions.
- **Symmetric averaging:** average forecasts from the series and its sign-flipped
  counterpart.
- **Clamp nonnegative series:** prevent negative forecasts for inferred
  nonnegative inputs.
- **Sort quantiles:** enforce nondecreasing quantile outputs.

Advanced controls and exact defaults are in the
[Explorer reference](../reference/explorer.md).

## Run and compare

Accept the TimesFM-3 weights restriction, then select **Run forecast**. The
status panel separates checkpoint loading from forecasting.

The app retains the newest 25 untracked runs in `data/timesfm.duckdb`; tracked
forecasts remain until removed from tracking. Use
**Evaluate** > **Compare runs** to overlay up to three point forecasts for the
same dataset and target. Original uploads and input history are not stored.

## Export results

Download the ZIP from **Forecast** > **Latest result**. Store `run.json` with the CSV output because
it records the checkpoint, code revision, input hashes, settings, mappings,
lineage operations, runtime, and license identifier.

Spreadsheet formula prefixes in user-controlled CSV text are neutralized in
the exported CSV. The JSON manifest retains the original mapping names.

## Analyze historical accuracy and future scenarios

Open **Evaluate** > **Run analysis** after choosing data, column roles, and
inference settings.
Use **Save settings for analysis** in Forecast to submit setting changes without
running a forecast.
Select a workflow and submit its form to run it. Editing controls, filtering
results, and downloading results do not initiate another model run.

1. **Rolling backtesting:** forecast across recent historical cutoffs. Start
   with five windows and a stride equal to your horizon. Compare errors by
   target, forecast distance, and window.
2. **Anomaly detection:** forecast each of the latest 50 observations one step
   ahead using only earlier observations. Inspect flagged points and their
   distance outside the q10–q90 interval. Missing actuals are unscored.
3. **Joint versus independent:** compare both modes on identical historical
   windows, without covariates in either mode. Select at least two targets.
   Positive independent-minus-joint MAE means joint forecasting performed better.
4. **Covariate usefulness:** compare all covariates with none, a selected group
   removed, and each selected covariate removed separately. Positive
   removed-minus-full MAE means the covariates improved measured accuracy.
5. **What-if scenarios:** edit future known-future covariate values for up to
   three named scenarios. Historical values, timestamps, and targets stay fixed.
   Compare each scenario with the unchanged baseline. Changing input data,
   mapping, or horizon resets incompatible edits.

Historical analyses interpolate model context only before each cutoff. They
never fill missing held-out actuals. Declare a covariate known-future only when
its values would have been available when making the forecast. The anomaly
interval is **80% nominal**, not a calibrated anomaly probability. Scenario
forecasts and covariate usefulness measure conditional predictions, not causal
effects.

Analyses require at most 32 combined target and covariate variates; benchmark
chunking and covariate subsampling are not used. A request cannot exceed
250,000 prediction rows. Submission validates and displays the planned forecast
and row counts before loading the model. See the [analysis reference](../reference/explorer.md#analysis-workflows)
for adjustable limits.

The newest 25 analyses are saved separately from the newest 25 forecast runs.
Select a saved analysis to view its results without loading the model. ZIP
downloads include predictions, metrics, a reproducibility manifest, and
workflow-specific output. Stored results include derived actuals and authored
scenario overrides, but not original uploads or historical context arrays.
Re-running a saved analysis requires the matching uploads. If database saving
fails, completed results remain available in the current session.

## Release model memory

Select **Clear model from memory** in the sidebar to clear the cached model. On CUDA, the
app also asks PyTorch to empty its memory cache.

## Prepare calendar features and bulk series

Use long format for rows such as `date, store_id, sales`. Select the group-ID
columns and a reusable source name. Each group is forecast separately; multiple
target columns inside a group can still use joint multivariate forecasting.
Keep source names and group keys consistent when uploading new observations.

The preparation preview lists gaps, duplicate timestamps within groups, sampling
issues, missing values, and usable history. Select the groups to forecast and
resolve blocking issues. Missing context cells show the values model preprocessing
will interpolate; uploaded data and held-out actuals are not repaired silently.

Enable weekday, month, country/subdivision holidays, or named date-range events
to add numeric known-future covariates. Weekdays use Monday=0 through Sunday=6;
months use 1–12. Holidays and events are evaluated on each timestamp's calendar
date. Enter a frequency when future dates cannot be inferred reliably. Preview
future rows before running; uploaded non-calendar covariates still need future
values. Generated columns count toward the 32-input limit. Historical event
backtests assume the event schedule was known at the forecast origin.

## Compare settings, baselines, and interval coverage

Settings experiments compare two to eight named configurations on identical
historical cutoffs. Start with context length, then vary the inference controls.
Every cutoff must have the longest requested history available. Target-level
rankings use MAE; the aggregate gives each target equal weight through mean ranks.
These are measured backtest results, not a guarantee about future performance.

Baseline comparisons include last-value and seasonal-naive forecasts. Supply
the season length in observations: for example, seven for weekly seasonality
in daily data. Missing seasonal source values remain unscored. Comparisons use
the same scored cases for every method and report exclusions.

Forecast charts show central 20%, 40%, 60%, and 80% nominal bands. Calibration
tables compare their nominal coverage with observed coverage by target and
forecast distance, alongside widths and sample counts. Overlapping backtest
windows count separate forecast-origin/distance pairs. These diagnostics do not
fit or alter intervals; crossed or unavailable bounds are excluded and reported.

## Refresh and track forecasts

In **Track**, add a saved forecast, then upload updated observations and associate
the current sources with its series. Legacy anonymous IDs require an explicit
association. Matching uses series identity, target, and timestamp, never row order.

Assess earlier forecasts against the uploaded actuals, or explicitly refresh
using the previous settings. Each refresh creates a new forecast vintage.
Corrected actuals create another assessment, leaving issued predictions unchanged.
Tracked runs are exempt from ordinary retention. Only derived forecasts, matched
actuals, and metadata are stored; source uploads remain in session memory.

## Choose a checkpoint or work offline

The sidebar accepts a Hugging Face repository and optional revision, a local
model folder, or a compatible `.safetensors`, `.pt`, or `.pth` state dictionary.
Folders require `config.json` and `model.safetensors`; standalone files must match the
shipped TimesFM-3 architecture. PyTorch files use weights-only loading.

Enable offline loading to use local files and cached Hub snapshots exclusively.
Missing or incompatible assets produce an error without switching models.
Every run records the resolved Hub revision or local file fingerprints. Refresh
reuses the saved revision and rejects changed local checkpoint files.
