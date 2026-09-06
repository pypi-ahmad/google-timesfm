# TimesFM-3 Explorer Handbook

Use this handbook when you want one practical path from raw time series to a
repeatable forecasting decision. It links to the detailed guides and reference
instead of duplicating every setting.

The Explorer is a local evaluation tool. The default TimesFM-3 weights are
restricted to non-commercial, non-production use. Read
[licensing and model versions](../explanation/licensing-and-versions.md) before
using a different checkpoint.

## 1. Start with a forecastable question

Write down the target, its observation frequency, the forecast horizon, and the
decision that the forecast informs. A forecast step has the same cadence as one
row in your data; 14 can mean fourteen days, weeks, or hours.

Choose the simplest target set first. Add related targets or covariates only
when you can evaluate whether they improve held-out accuracy.

## 2. Prepare the data

Open **Prepare** and select a demo or upload CSV/Parquet files.

- Use a wide table when every series shares one timestamp axis.
- Use a long table such as **date, store_id, sales** when one file contains many
  series. Select stable group-ID columns to split it into forecastable groups.
- Map every numeric model column to exactly one role: target, past-only
  covariate, or past-and-future covariate.
- Keep targets and covariates out of group IDs. A missing group ID blocks the
  affected group instead of becoming a model input.
- In **Optional preparation**, open **Calendar and events** to create weekday,
  month, holiday, or named event covariates and append future rows where needed.

The readiness panel is the first checkpoint. Resolve blocked rows before
forecasting. Warnings identify gaps, irregular sampling, missing context,
constant targets, interpolation behavior, and high-dimensional requests.

[Prepare data](prepare-data.md) contains CSV examples and exact validation
rules.

## 3. Build a baseline forecast

Open **Forecast** and choose:

- **Forecast future** to predict after the final observed target.
- **Evaluate holdout** to reserve the last horizon and score it.
- **Joint multivariate** to use relationships between targets and covariates.
- **Independent univariate** to forecast every target from its own history.
  This mode does not use covariates.

Start with horizon 32, context length 512, and the default probabilistic
controls. Use CPU for portability or CUDA when a compatible GPU is available.
Accept the weights restriction, then select **Run forecast**.

The Explorer resolves the selected checkpoint only when inference begins. The
default is the TimesFM-3 checkpoint from Hugging Face; the sidebar also accepts
a pinned Hub revision, a local folder, or a compatible state-dictionary file.

## 4. Read forecasts and uncertainty

The **Latest result** view contains a chart and tables for one selected dataset
and target. The point line is the median forecast. Quantiles form central 20%,
40%, 60%, and 80% nominal prediction bands.

A wider band means the model produced more dispersed quantiles. It does not
guarantee coverage. Use holdouts or rolling backtests to compare nominal and
observed coverage before treating bands as operational thresholds.

Download the result bundle when you need a portable record. Its manifest records
model identity, settings, mappings, input hashes, lineage, runtime, and source
revision. It never includes the original upload.

## 5. Test historical performance

Open **Evaluate** and choose **Run analysis**. Save forecast settings first if
you changed them without running a forecast.

All historical workflows end model context strictly before the evaluation
cutoff. Held-out target values and future past-only covariates are not used to
create the forecast.

| Workflow | Use it to answer | Key interpretation |
|---|---|---|
| Rolling backtesting | How does accuracy vary across time and forecast distance? | Compare by target and step, not only an overall average. |
| Forecast configurations | Which context/inference configuration works best? | Rank two to eight configurations on shared historical cutoffs. |
| Naive baselines | Does TimesFM improve on simple rules? | Compare against last-value and seasonal-naive forecasts on the same scored pairs. |
| Joint versus independent | Does multivariate forecasting help? | A positive independent-minus-joint MAE favors joint forecasting. |
| Covariate usefulness | Do selected covariates improve prediction? | A positive removed-minus-full MAE favors including covariates; this is predictive, not causal. |
| Anomaly detection | Which observations departed from a pre-observation forecast interval? | A point outside q10–q90 is flagged with direction and distance beyond the bound. |

Settings, baseline, joint/independent, and covariate comparisons use identical
historical schedules. Do not pool errors across targets with different units.

## 6. Explore conditional scenarios

Choose **What-if scenarios** only after mapping at least one past-and-future
covariate and supplying all future values. Edit promotion, price, event, or
other known-future values for up to three scenarios.

Each scenario starts with a fresh copy of the same baseline inputs. The result
is scenario minus baseline. It answers how this forecasting model changes when
the supplied future covariate changes; it does not establish a causal effect.

## 7. Calibrate interval expectations

Holdout, rolling analysis, and tracked assessments expose interval coverage by
target and forecast step. Compare observed coverage with the 20%, 40%, 60%, and
80% nominal bands.

Treat small samples, overlapping historical windows, invalid bounds, and missing
actuals carefully. The Explorer reports them and does not fit or alter model
intervals.

## 8. Track issued forecasts

Open **Track** after a forecast has been saved.

1. Choose the saved forecast.
2. Select **Keep this forecast tracked** to protect it from normal retention.
3. Upload new observations in **Prepare**.
4. Map each saved dataset to one distinct updated source; leave unmatched saved
   datasets blank to skip them.
5. Select **Evaluate against actuals** to save immutable matched actuals and
   error metrics, or **Refresh forecast** to issue a linked new vintage.

Tracking matches exact timestamps, series identity, and target names. Corrected
actuals create another assessment; they never rewrite issued predictions.
Refresh reuses the saved mapping, preparation, settings, and model revision.
Changed local checkpoint files are rejected to preserve the historical record.

Tracked forecasts are exempt from the newest-25 untracked-run retention rule.
Stopping tracking removes that protection and can prune an old run immediately.

## 9. Choose and reproduce a checkpoint

Use the sidebar to select a Hub repository and optional revision, a local model
folder, or a compatible **.safetensors**, **.pt**, or **.pth** file.

- A local folder needs **config.json** and **model.safetensors**.
- Every checkpoint must have matching TimesFM-3 parameters and q0.1 through
  q0.9 quantiles.
- Offline mode resolves only local files and cached Hub snapshots.
- Every run records the resolved revision or file fingerprint.

Use a pinned revision or local fingerprint whenever results must be reproducible
across time.

## 10. Keep the working set manageable

The Explorer keeps decoded uploads only in browser-session memory. DuckDB stores
derived forecasts, analysis outputs, tracking links, and assessments locally in
**data/timesfm.duckdb**.

Use **Clear model from memory** after a large GPU run when you need to free the
cached checkpoint. Reduce context length, horizon, batch size, or variate count
when memory is constrained. Requests above 32 joint inputs require explicit
benchmark chunking approval and may subsample covariates.

For troubleshooting and exact limits, use the
[Explorer reference](../reference/explorer.md) and
[troubleshooting guide](../troubleshooting.md).
