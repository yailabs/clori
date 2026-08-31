# Events and Telemetry Contract

Status: normative implemented contract

Authority: server/runtime typed event owners and telemetry fan-out. Renderers
project these facts but do not own event meaning.

## Producer and consumers

Runtime, server, session, generation, listener, and shutdown owners publish one
ordered event stream. Consumers are the server raw console, local protocol
subscribers, status/metrics accumulation, `host logs`, and the interactive
console.

No consumer scrapes another renderer's text.

Operational events, execution metrics, audit evidence, and profiler output are
separate classes. An operational event contains only lifecycle facts known at
publication; it does not fill future counters with candidate capacity.
Execution accounting may aggregate work without becoming audit evidence, and
full probes never enter normal watch merely because trace verbosity increased.

## Event identity

Each event carries a schema, global sequence, wall and monotonic timestamps,
severity, kind, process/runtime identities, and applicable model, engine
generation, artifact, specialization, session, request, turn, phase, counter,
timing, rate, and result fields. Content is excluded unless an explicit
trace-content policy admits it.
The sealed semantic identity binds sequence, kind, correlations, counters,
durations, result facts, and model identities. Process ID and observed wall or
monotonic clock values remain diagnostic fields and do not enter that identity.

Events observe state after the owning publication boundary. An event cannot
make an uncommitted state visible or promote capability.

## Lifecycle coverage

Typed events cover at least:

- host startup/readiness, engine load, package/binding admission,
  specialization, materialization, residency, draining, unload, and failure;
- listener preparation/readiness/failure;
- client/session attach and lifecycle;
- request admission, queueing, tokenization, prefix reuse, and prefill;
- draft start/completion, target verification start/completion, accepted prefix,
  candidate rejection, and speculative-cycle commit;
- first token, decode progress, fragment publication, commit, and stop;
- media request, conditioning, latent iteration, video/audio decode,
  publication, completion, cancellation, and failure;
- cancellation, refusal, failure, and partial progress;
- memory/resource counters and bounded profile stages;
- host shutdown admission, engine drain/close, and completion.

## Fan-out and overflow

One bounded fan-out distributes the event to raw JSONL, protocol subscribers,
metrics, and human renderers. Low-priority progress may be coalesced or dropped
under pressure, and overflow remains an explicit fact. Lifecycle and terminal
events may not disappear silently.

Subscription failure or a slow client cannot block scheduler workers
indefinitely. Disconnect releases subscriber resources without closing an
engine or the host.

## Projections

`yvex serve` renders the compact human projection in the owning foreground
terminal by default. `yvex host logs` attaches the same projection from
another terminal, while `host logs --verbose` additionally renders individual
speculative cycles. `yvex serve --logs json` and `yvex host logs --json`
emit canonical JSONL for the admitted trace schema. `host status` is a
bounded host snapshot and `engine list` is the engine-inventory snapshot;
neither is an event replay. Human projections render retained history plus live
events in stable semantic categories. They retain operator-significant host,
engine, session, contended queue, prefill, first-token, aggregate speculative
economics, completion, cancellation, and failure events while suppressing
connection churn, uncontended queue admission, fragments, intermediate
draft/verification steps, and profile rows. They render bytes in human units,
speculative acceptance as
accepted/proposed, and stop codes as their named contract values. The `--json`
projection retains the full subscribed event sequence with sequence, severity,
turn, phase, timing, and rate. Neither projection
exposes generic positional counter names. Native prefill
progress sent to the REPL is another projection of the sealed event, not a
synthetic client event.

Media progress is likewise server-authored. The interactive client may project
bounded completed/total iteration facts but does not fabricate percentages or
assistant prose. Component-open events distinguish full hash, verified reopen,
fallback hash, receipt state, bytes actually hashed, file extent, and elapsed
time. These authentication facts do not imply materialization or residency.

Speculative events carry availability-bearing named generation mode, cycle,
candidate extent, selected-verification, accepted, rejected, stop-discarded,
correction/bonus, promoted, replay, verification, confidence, timing, and
policy-identity facts. Legacy generic event counters remain part of the
versioned base event record but do not encode DSpark meaning. The human log
groups each request and its cycle summary; JSONL preserves the individual
events. Neither projection publishes draft token text.

## Privacy and content

Default telemetry excludes prompt text, response text, logits, hidden values,
tensor payloads, and KV contents. Explicit content tracing is opt-in, locally
scoped, and unsuitable for source control. Sensitive values and private paths
are not part of ordinary status or machine discovery.

## Timing and counters

CPU timings use a monotonic clock. CUDA device work uses device-complete timing
when kernel duration is claimed; asynchronous enqueue time is not kernel time.
Normal serving does not add synchronizations merely for telemetry.

Draft and verification durations are reported separately from committed
generation rate. Proposed tokens never contribute to generated-token or
completion-usage counters; accepted-prefix and correction/bonus accounting is
explicit.

Unavailable timing or placement fields are marked unavailable. Counters name
actual target/draft forwards and rows, verifications, accepted/promoted/replayed
rows, state copies and candidate bytes, output-head rows, logits movement,
full-array scans/digests, row/expert pairs and unique experts, launches, graph
launches, waits, synchronizations, allocations, shape hits/misses/rebuilds,
expert-worklist counts, pair/bucket counts, maximum bucket width, narrow rows,
bounded tails and Tensor Core eligible/executed rows, multi-source physical
batch counts, real rows and source counts, and the multi-source expert-bucket
population histogram. These are publication facts rather than values inferred
from tensor sizes. A measured
zero synchronization duration is never interpreted as zero synchronization
cost.

## Side effects and failure

Event publication may update bounded metrics and subscriber queues or write an
explicitly selected console/trace sink. It does not mutate model state.
Serialization, sink, overflow, and subscriber failures remain distinguishable
and cannot fabricate a successful runtime operation.

## Compatibility and non-claims

Raw event and status schemas are versioned independently of human rendering.
Telemetry is not evaluation, benchmark authority, conversation history,
release evidence by itself, or a public monitoring service.
