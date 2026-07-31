# Events and Telemetry Contract

Status: normative implemented contract

Authority: server/runtime typed event owners and telemetry fan-out. Renderers
project these facts but do not own event meaning.

## Producer and consumers

Runtime, server, session, generation, listener, and shutdown owners publish one
ordered event stream. Consumers are the daemon raw console, local protocol
subscribers, status/metrics accumulation, `runtime watch`, `runtime trace`, and
the interactive console.

No consumer scrapes another renderer's text.

## Event identity

Each event carries a schema, global sequence, wall and monotonic timestamps,
severity, kind, process/runtime identities, and applicable model, artifact,
variant, session, request, turn, phase, counter, timing, rate, and result
fields. Content is excluded unless an explicit trace-content policy admits it.

Events observe state after the owning publication boundary. An event cannot
make an uncommitted state visible or promote capability.

## Lifecycle coverage

Typed events cover at least:

- startup, artifact/binding admission, materialization, residency, and ready;
- listener preparation/readiness/failure;
- client/session attach and lifecycle;
- request admission, queueing, tokenization, prefix reuse, and prefill;
- first token, decode progress, fragment publication, commit, and stop;
- cancellation, refusal, failure, and partial progress;
- memory/resource counters and bounded profile stages;
- shutdown admission, drain, model close, and completion.

## Fan-out and overflow

One bounded fan-out distributes the event to raw JSONL, protocol subscribers,
metrics, and human renderers. Low-priority progress may be coalesced or dropped
under pressure, and overflow remains an explicit fact. Lifecycle and terminal
events may not disappear silently.

Subscription failure or a slow client cannot block the model worker
indefinitely. Disconnect releases subscriber resources without closing the
runtime model.

## Projections

`yvexd --console raw` and `yvex runtime trace --json` emit canonical JSONL for
the admitted trace schema. `runtime status` is a bounded snapshot rather than
an event replay. Human `runtime watch` renders the compact semantic stage
stream. Human `runtime trace` adds sequence, severity, turn, phase, timing, and
rate to the same semantic facts. Neither renderer exposes generic positional
counter names. Native prefill progress sent to the REPL is another projection
of the sealed event, not a synthetic client event.

## Privacy and content

Default telemetry excludes prompt text, response text, logits, hidden values,
tensor payloads, and KV contents. Explicit content tracing is opt-in, locally
scoped, and unsuitable for source control. Sensitive values and private paths
are not part of ordinary status or machine discovery.

## Timing and counters

CPU timings use a monotonic clock. CUDA device work uses device-complete timing
when kernel duration is claimed; asynchronous enqueue time is not kernel time.
Normal serving does not add synchronizations merely for telemetry.

Unavailable timing or placement fields are marked unavailable. Counters name
actual observed movement, allocation, launch, synchronization, and publication
facts rather than inferring them from tensor sizes.

## Side effects and failure

Event publication may update bounded metrics and subscriber queues or write an
explicitly selected console/trace sink. It does not mutate model state.
Serialization, sink, overflow, and subscriber failures remain distinguishable
and cannot fabricate a successful runtime operation.

## Compatibility and non-claims

Raw event and status schemas are versioned independently of human rendering.
Telemetry is not evaluation, benchmark authority, conversation history,
release evidence by itself, or a public monitoring service.
