---
type: Guide
title: Records, artifacts, migrations, and retention
description: Persistence boundaries for workbench records and artifacts, legacy DuckDB history, imports, and cleanup.
tags: [persistence, postgres, artifacts, duckdb, migration]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-ca976da183a76903427058bc
    resource: repo://src/timesfm_app/artifacts.py
  - id: openwiki-source-8c1049cc7a39057b6abb0540
    resource: repo://src/timesfm_app/maintenance.py
  - id: openwiki-source-fa70842e08bf501d0aa143cd
    resource: repo://src/timesfm_app/migration.py
  - id: openwiki-source-4c504c9233c31027d06d8250
    resource: repo://src/timesfm_app/store.py
  - id: openwiki-source-51dd5f97d993733fe7bd93f9
    resource: repo://src/timesfm3/run_store.py
  - id: openwiki-source-4f630bcb5d7794a1659c12c9
    resource: repo://tests/test_app_migration.py
  - id: openwiki-source-a66f1d3eadaf3a9e8c2feb57
    resource: repo://tests/test_app_store.py
  - id: openwiki-source-10168fd9970eb88a5388836e
    resource: repo://tests/test_run_store.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Records, artifacts, migrations, and retention

There are two distinct persistence paths:

- The primary workbench stores workspace-scoped records, drafts, jobs, events,
  and result manifests in PostgreSQL. Dataset bytes and derived result tables
  live in the artifact store.
- The retained Streamlit explorer stores derived forecast and analysis history
  in `data/timesfm.duckdb`.

The [workbench architecture](workbench-architecture.md) and [legacy explorer](legacy-explorer.md)
pages explain which application uses each path.

## Workbench records

`timesfm_app.store.Store` requires PostgreSQL for application configuration;
its explicit SQLite factory is for isolated tests only. Records belong to a
workspace, and payload references are checked against records in that same
workspace. Draft updates use an expected revision, while version and run records
are immutable ([store models and setup](../src/timesfm_app/store.py#L70),
[`Store`](../src/timesfm_app/store.py#L161),
[reference/update rules](../src/timesfm_app/store.py#L274)).

Deletion is dependency-aware: referenced records and records still needed by
job history are protected. Removing a result record clears its job's result
pointer and records a result-removed event, preserving terminal job history
([deletion checks](../src/timesfm_app/store.py#L418),
[store tests](../tests/test_app_store.py#L58)).

## Artifacts

Artifact descriptors carry a key, SHA-256, and byte size. Keys must be safe
relative POSIX paths. The local store publishes complete files without
overwriting an existing key with different bytes; the S3 adapter uses a
conditional put and verifies stored checksums when reading
([key validation](../src/timesfm_app/artifacts.py#L49),
[local store](../src/timesfm_app/artifacts.py#L152),
[S3 store](../src/timesfm_app/artifacts.py#L211),
[immutability tests](../tests/test_app_store.py#L94)).

## Retention and orphan cleanup

Workbench result retention is disabled unless a workspace policy enables it.
Preview identifies age/count candidates and protects runs that still have
dependencies; applying the policy rechecks dependencies before deleting records
([retention preview](../src/timesfm_app/maintenance.py#L33),
[opt-in test](../tests/test_app_migration.py#L200)). Artifact deletion is a
separate operation. The orphan sweep considers managed artifact prefixes only,
skips files newer than 24 hours, referenced keys, and active-attempt prefixes;
apply mode rechecks references immediately before removal and serializes with
legacy imports ([orphan cleanup](../src/timesfm_app/maintenance.py#L104),
[cleanup test](../tests/test_app_migration.py#L223)).

Preview cleanup before applying it. This command reports candidate files; add
`--apply` only after reviewing the output. It also applies any enabled workspace
result-retention policy.

```powershell
uv run --no-sync python -m timesfm_app.maintenance --orphans
```

See [local operations](operations.md) for the persistent state directories that
should not be removed during routine recovery.

## Importing legacy DuckDB history

`timesfm_app.migration` opens the source DuckDB database read-only and writes
derived records and artifacts into the workbench. Import IDs are deterministic
from the resolved source, workspace, entity kind, and legacy ID, so repeating an
import skips existing records. The source file is not modified. Imported
forecasts and analyses preserve their derived data and reports, but original
uploads are unavailable and automatic forecast refresh is disabled for imported
tracking entries ([import flow](../src/timesfm_app/migration.py#L32),
[run and assessment import](../src/timesfm_app/migration.py#L51),
[migration tests](../tests/test_app_migration.py#L71)).

Import writes artifacts before the referencing record is committed. If it is
interrupted in that gap, rerunning the import completes it; orphan cleanup is
coordinated with the import lock ([resume test](../tests/test_app_migration.py#L164),
[`import_legacy`](../src/timesfm_app/migration.py#L51)).

## Legacy DuckDB history

The Streamlit `run_store` transactionally saves immutable forecast vintages and
retains the newest 25 untracked runs by default; tracked runs are excluded from
pruning. Analyses use a separate history with a default limit of 25. These
limits are distinct from the workbench's opt-in workspace retention policy
([forecast store](../src/timesfm3/run_store.py#L227),
[analysis store](../src/timesfm3/run_store.py#L676),
[retention tests](../tests/test_run_store.py#L80)).
