---
type: Guide
title: Durable job dispatch, retries, and cancellation
description: How workbench submissions move through the database outbox, Dramatiq workers, attempt leases, and fenced result publication.
tags: [workbench, jobs, workers, reliability]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:48.709Z
sources:
  - id: openwiki-source-677429aa2b6ed69a047636a3
    resource: repo://src/timesfm_app/api.py
  - id: openwiki-source-14c5a154f256eaf4beadc34c
    resource: repo://src/timesfm_app/jobs.py
  - id: openwiki-source-4c504c9233c31027d06d8250
    resource: repo://src/timesfm_app/store.py
  - id: openwiki-source-32d78e8d1b4f768a0ccccc9e
    resource: repo://src/timesfm_app/worker.py
  - id: openwiki-source-8a75387badf2255379aee732
    resource: repo://tests/test_app_jobs.py
generated: { by: "codex", at: "2026-09-23T14:20:48.709Z" }
---

# Durable job dispatch, retries, and cancellation

The workbench stores job state in the application database and uses Dramatiq as
the delivery mechanism. Delivery may repeat: the database claim, attempt number,
and lease determine which worker currently owns execution and may publish a
result. The main components are the [job API](../src/timesfm_app/api.py#L471),
[`Store`](../src/timesfm_app/store.py#L532),
[`jobs.py`](../src/timesfm_app/jobs.py#L21), and
[`worker.py`](../src/timesfm_app/worker.py#L254).

## Submission and dispatch

Clients submit a job with an idempotency key. The store hashes the canonical
request, checks workspace quota and references, then records the queued job,
event, and outbox entry in the same database transaction. Replaying the same
key and request returns the existing job; using that key for different content
is rejected ([creation](../src/timesfm_app/store.py#L532),
[idempotency tests](../tests/test_app_jobs.py#L35)).

The dispatcher polls pending outbox entries, sends the job ID to the appropriate
worker actor, and marks an entry dispatched only after the send is acknowledged.
If the broker send fails, the entry remains pending. If the dispatcher crashes
after send but before marking it, delivery can happen again; workers use the
database claim to make that duplicate harmless. Reconciliation also re-enqueues
jobs left queued without a pending outbox message after a grace period
([dispatcher](../src/timesfm_app/jobs.py#L21),
[reconciliation](../src/timesfm_app/store.py#L762),
[failure and redelivery tests](../tests/test_app_jobs.py#L174)).

## Attempt ownership and results

A worker claims only a queued, non-cancelled job. Claiming increments its
attempt number and starts a lease. Heartbeats renew that lease; progress and
publication are accepted only while the attempt is current, running, uncancelled,
and unexpired. The attempt number is a fence: an expired or superseded worker
cannot publish late output over the current attempt
([claim and fence](../src/timesfm_app/store.py#L592),
[publication](../src/timesfm_app/store.py#L653),
[fencing regression](../tests/test_app_jobs.py#L91)).

The worker writes attempt-scoped artifacts before publishing the database result
pointer. If ownership is lost in between, the result is not made visible and the
leftover artifact can be handled by orphan cleanup
([worker publication boundary](../src/timesfm_app/worker.py#L363)).

Transient transport failures can be retried automatically, with a maximum of
three attempts in the initial sequence; input validation and out-of-memory
failures are not considered retryable. A failed job can also be explicitly
retried through the API. The resolved model is frozen on the job so automatic
retries reuse the same resolved checkpoint ([retry policy](../src/timesfm_app/worker.py#L237),
[store retry and model pin](../src/timesfm_app/store.py#L637),
[retry tests](../tests/test_app_jobs.py#L110)).

## Cancellation and events

Cancelling a queued job prevents it from being claimed. Cancelling a running job
first changes it to `cancelling`; it becomes `cancelled` only after execution has
reached a safe boundary and the worker/device is quiescent. If a cancellation
lease expires, reconciliation requests supervisor recovery rather than assuming
the process has stopped ([state transitions](../src/timesfm_app/store.py#L704),
[expired cancellation](../src/timesfm_app/store.py#L782),
[quiescence tests](../tests/test_app_jobs.py#L67)).

Job state changes and progress are recorded as sequenced events. The API exposes
an event stream with a resumable sequence cursor, allowing clients to continue
after reconnecting ([event endpoint](../src/timesfm_app/api.py#L495),
[`Store.events`](../src/timesfm_app/store.py#L750)).

For process ownership and recovery operations, see [local operations](operations.md).
For database records and artifact lifetimes, see [persistence](persistence.md).
