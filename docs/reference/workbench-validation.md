# Workbench migration validation

Validated locally on Windows 11 with Python 3.13.15, CUDA Torch 2.14.0+cu132,
an RTX 4060 laptop GPU, private PostgreSQL, and Memurai. The existing TimesFM-3
numerical modules remain unchanged by this migration.

## Automated checks

- The retained Explorer, preparation, analysis, checkpoint-selection, and tracking
  regression suite passed: 167 tests.
- New HTTP/service/storage/job/import/diagnostic/native-process tests passed.
  The 19 native-process tests create their own disposable processes and check exact
  process identity, orphan cleanup, cancellation, and replacement behavior.
- The persistence/tracking subset passed 61 tests with native PostgreSQL enabled,
  including four database concurrency tests otherwise skipped without a database.
- Ruff, ty, the uv lock check, and the Python wheel build passed.
- The optimized Next.js build, TypeScript, 11 frontend unit tests, and 16 browser
  tests passed. Browser checks cover draft conflicts, immutable submissions,
  navigation, narrow layouts, pagination, scenarios, result selection, tracking
  mapping, refresh preferences, persisted accuracy, and calibration filtering.

See [CONTRIBUTING](../../CONTRIBUTING.md#run-checks) and
[web/README](../../web/README.md#verification) for repeatable commands.

## Real model checks

Using the included synthetic demand data and a cached TimesFM-3 checkpoint, HTTP
requests passed through the database outbox, Memurai, and native GPU worker to
produce persisted results for:

- Forecasting with known-future covariates.
- Rolling backtests and last-value/seasonal-naive comparisons.
- Joint-versus-independent comparison and covariate usefulness analysis.
- Context-setting comparisons, future promotion scenarios, and anomaly analysis.
- Automatic assessment after a new dataset version, followed by an opted-in
  forecast refresh; the original forecast record remained unchanged.
- Cancellation during forecasting without a published partial result.

A separate CPU job verified checkpoint compatibility. API and both worker
Prometheus endpoints responded successfully. Saved results remained available
after restarting the native application.

## Limits of this validation

The S3 adapter was not exercised against a remote bucket. Prometheus and
OpenTelemetry configuration is provided; external collector deployment was not
tested. This is local application validation, not a multi-user or load-testing
claim. The separate legacy TimesFM 2.5 tests are outside this migration's
acceptance suite. GPU success on the demonstration data does not establish
forecast accuracy for another dataset.
