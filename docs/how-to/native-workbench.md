# Run the native Windows workbench

The workbench runs on your Windows account with no sign-in. Workspaces organize
local data; they are not security boundaries between users. Services bind to
loopback. The older Streamlit explorer remains available as a legacy diagnostic
client.

## Install and start

Double-click `launch_workbench.cmd` in the repository root for one-step startup.
It runs setup when required files are missing, starts or reuses the workbench,
and opens your browser after service health checks pass. The launcher terminal
stays open and streams new service logs with labels such as `[api]`, `[gpu]`, and
`[web]`, including startup output. First setup needs
internet access. If startup fails, the launcher keeps the error visible.
Closing its window does not stop the background services; use `.\dev.ps1 stop`.

Prerequisites: Windows 11, Git, Node.js 24+, and uv. Setup downloads pinned
PostgreSQL and Memurai Developer binaries under `.native/`, creates a private
database cluster, installs dependencies, and builds Next.js. It does not install
Windows services or change user or machine environment variables. Memurai is a
separately licensed Developer edition; review its terms for your use.

```powershell
.\dev.ps1 setup
.\dev.ps1 start
.\dev.ps1 doctor
```

Open <http://localhost:3000>. The first forecast needs a checkpoint download,
unless Models points to a compatible local checkpoint or an already cached
revision with offline loading enabled. The default weights retain their separate
non-commercial, non-production license; application engineering quality does not
change that license.

Setup preserves an existing CUDA Torch installation. Do not replace that setup
command with an ordinary dependency sync that could select a CPU Torch wheel.
For explicit CPU inference, set a process variable before starting:

```powershell
$env:TIMESFM_DEVICE = 'cpu'
.\dev.ps1 start
```

Device selection is a worker setting. There is no automatic CPU fallback after
an out-of-memory error. Reduce the workload or restart with an explicit device
choice. For frontend hot reload, use `.\dev.ps1 dev`. Stop before switching modes.

| Process | Default address | Purpose |
|---|---|---|
| Next.js | localhost:3000 | Browser UI and API proxy |
| FastAPI | 127.0.0.1:8001 | API; interactive contract at `/docs` |
| PostgreSQL | 127.0.0.1:55432 | Persistent records and job outbox |
| Memurai | 127.0.0.1:56379 | Dramatiq delivery |
| GPU worker metrics | 127.0.0.1:9201 | Inference worker counters |
| CPU worker metrics | 127.0.0.1:9202 | Assessment/checkpoint-check counters |

PostgreSQL authenticates the current Windows account through SSPI without a
password in a project file. Stop leaves the database available. `.native/` and
`data/product/` contain persistent state; do not delete them for an ordinary job
failure.

## Run a forecast

1. In Data, upload CSV/Parquet or create the demand demo. Upload replacements to
   the same logical dataset to create immutable versions.
2. In Forecasts, select versions, assign timestamp/targets/covariates, and choose
   context and horizon. Group identifiers cannot also be targets or covariates.
3. Preview preparation and quality. Resolve invalid groups or explicitly exclude
   them. Calendar features can include future rows.
4. Submit the job. You can navigate away and reopen it later.
5. Inspect nested 20/40/60/80% bands, select a target, inspect result tables, and
   export the result bundle. Copy a saved run's settings to edit a new draft.

Drafts use revision checks. If another tab saved a newer revision, reload it or
save a separate draft. Retried HTTP submissions keep their idempotency key so
one click does not create duplicate jobs.

## Experiments, scenarios, and tracking

Experiments include rolling backtesting, naive baselines, joint-versus-independent
comparison, covariate ablation, settings comparisons, and leakage-free anomaly
analysis. Accuracy tables retain target and forecast-distance detail. Rolling
charts show the latest origin; tables contain every evaluated origin.

Scenarios edit known-future covariates and compare conditional predictions with a
baseline. They do not establish causal effects. Tracking associates an issued
forecast with incoming actuals, retains assessment versions, and can create a new
forecast vintage. Refresh is opt-in. Upload actuals as another version of the same
logical dataset; changed series identities require explicit association.

The dispatcher detects the latest available version combination and queues an
assessment once per combination. Opt-in automatic refresh adds a forecast job.
Several uploads between polls can be coalesced into the latest combination;
use manual assessment when an intermediate version needs separate evaluation.

Workspace settings contains activity history and opt-in retention. Preview the
affected runs before applying deletion. Referenced records are protected. Removing
a result keeps its terminal job history and records that the result was removed.
Stopping tracking does not delete the original forecast.

Retention removes eligible result records. Artifact cleanup is separate and
protects referenced files plus files younger than 24 hours. Preview abandoned
artifacts with `uv run --no-sync python -m timesfm_app.maintenance --orphans`;
add `--apply` only after reviewing the candidate list. That command also applies
the workspace's enabled result-retention policy.

## Recovery and diagnostics

Logs live under `.native/logs/`. Cancellation first requests a cooperative stop.
If inference cannot reach a cancellation point within the grace period, the
supervisor retires its owned worker before confirming cancellation. Attempt fences
prevent a late result from publishing after a retry. Transient failures can retry
within the attempt limit; validation and GPU memory failures require correction.
The database outbox survives broker outages.

```powershell
.\dev.ps1 doctor
uv run --no-sync streamlit run diagnostic_app.py --server.port=9588
```

The diagnostic client uses FastAPI and does not load a model in Streamlit.
`streamlit_app.py` still uses its original local execution model.

Import DuckDB history without modifying the source database:

```powershell
uv run --no-sync python -m timesfm_app.migration data/timesfm.duckdb
```

The import is idempotent. Legacy runs can lack original uploads or a verified
checkpoint association; they need those inputs before refreshing. Back up the
PostgreSQL cluster and artifact store together.

## Storage and monitoring

Local artifacts under `data/product/artifacts` are the default. For S3, configure
`TIMESFM_STORAGE_BACKEND=s3`, `TIMESFM_S3_BUCKET`, and optionally
`TIMESFM_S3_ENDPOINT_URL` in the launching process. The adapter uses boto3's normal
credential chain. Keep credentials out of browser configuration and repository
files. Changing backends does not move existing local artifacts automatically.

The API serves Prometheus metrics at `/metrics`; workers use the ports above.
`monitoring/prometheus.yml` is a scrape configuration. The supervisor does not
install or launch Prometheus or an OpenTelemetry Collector. Run those separately
if needed, using `monitoring/otel-collector.yml` as the collector configuration,
and set `TIMESFM_OTLP_ENDPOINT` to the collector's HTTP trace endpoint before
starting the workbench. Structured logs work without a collector.

Export the API contract without loading the model or contacting services:

```powershell
uv run --no-sync python -m timesfm_app.openapi .native/openapi.json
```
