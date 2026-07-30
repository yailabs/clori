# Client And Terminal Architecture

This document owns the incompatible v2 client, terminal, and output doctrine.
The former flat command catalog and selectable `normal|table|audit` layouts are
retired; they are not compatibility surfaces.

## Binary boundaries

| Binary | Owner | Engine linkage | Terminal authority |
| --- | --- | --- | --- |
| `yvexd` | runtime host and server adapters | yes | optional raw JSONL console, loopback HTTP/SSE, and fatal stderr |
| `yvex` | unified public command | runtime lane: no; offline lane: yes | conversation, compact status, watch, trace, and explicit offline evidence |

Runtime and model code never writes product output. It publishes typed facts or
typed events. Client renderers own layout; only client I/O owners and daemon
entrypoint write terminal streams.

## Product grammar

```text
yvex
yvex chat [--session NAME] [--max-new-tokens N]
yvex run [options] TEXT

yvex runtime start|stop|status|watch|trace
yvex session new|list|show|attach|detach|reset|close
yvex model list|use|show
yvex artifact show|verify|metadata|tensors|materialize|emit ...
yvex graph ...
yvex quant preset|plan|emit|summarize|explain|policy|imatrix ...
yvex tokenizer show|encode|decode|prompt ...
yvex source manifest|native ...
yvex tensor map|collection ...
yvex evidence target|model|moe|backend|cuda ...
yvex help
yvex version
```

The runtime-facing lane has no dependency edge to engine execution. Direct
graph, materialization, tokenizer, metadata, tensor-map, and target-report
operations enter a separately guarded finite offline lane in the same ELF. No
deprecated, hidden, or forwarding aliases exist.

`model use NAME --artifact FILE --runtime-binding FILE` atomically records one
private XDG selection for a later `runtime start`; it never opens the model and
cannot hot-switch a running daemon.

`yvex` and `yvex chat` require a TTY. `yvex run` is the noninteractive one-shot
form. A missing daemon produces one concise refusal plus the exact runtime-start
hint. Unknown and duplicate options follow the product parser's typed refusal
policy.

The REPL owns bounded in-memory history, UTF-8 code-point deletion, bracketed
multiline paste, resize redraw, and two-stage SIGINT/EOF behavior. History is
not persisted and never becomes telemetry content.

## Offline engineering grammar

Direct engineering capability is grouped by semantic owner:

```text
yvex graph ...
yvex artifact show|verify|metadata|tensors|materialize|emit ...
yvex quant preset|plan|emit|summarize|explain|policy|imatrix ...
yvex tokenizer show|encode|decode|prompt ...
yvex source manifest|native ...
yvex tensor map|collection ...
yvex runtime input|context ...
yvex evidence target|model|moe|backend|cuda ...
```

This is a nested offline lane, not a renamed flat registry or a second process.
Its handlers may link engine owners, while runtime-client handlers continue to
cross the local protocol.

## Canonical projections

### Conversation

The REPL prints a short banner, `you>` prompt, streamed committed assistant
bytes, and one concise turn summary. It never prints raw events, logits,
identities, tensor facts, or capability walls.

### Compact status

Human status names the runtime state, selected logical model and physical
variant, backend, context, sessions, queue, process memory, and resident memory.
It includes at most one blocker and one actionable hint.

### Operational stream

`yvex runtime watch` projects the typed event sequence into chronological,
rate-limited lines. It shows authoritative startup, request, queue, suffix
prefill, TTFT, decode rate, final position, stop, cancellation, failure, and
shutdown facts. It excludes content and per-token rows by default.

### Raw stream

`yvexd --console raw` and `yvex runtime trace` serialize the same event
sequence as JSONL. Raw means complete event records, not tensor, hidden, logits,
KV, or memory dumps. Text content is excluded unless `--trace-content` is
explicitly enabled at the host.

### Machine status

`yvex runtime status --json` is a stable protocol-derived object. Human prose
never shares its stdout. Developer owners may expose their own explicitly
versioned JSON or evidence-file schemas.

### Errors

Product errors contain one class/reason and, where known, one remediation hint.
Errors go to stderr. Exit `2` is parser/usage refusal; runtime and protocol
refusals are nonzero without turning diagnostic evidence into normal output.

### Application protocol

The OpenAI adapter inside `yvexd` is not a third terminal renderer. It returns
the documented compatibility JSON or SSE schema over loopback HTTP. Its
response objects project typed provider and YVEX protocol facts; they never
scrape `yvex` or daemon-console text. The exact profile lives in
[`openai-compatibility.md`](openai-compatibility.md).

## Typed event fan-out

One event has a schema, global sequence, UTC and monotonic timestamps, severity,
kind, process identity, model/artifact/variant identities when present, session,
request, turn, phase, counters, timing/rate, and canonical identity.

```text
authoritative runtime owner
    -> typed event
       -> raw JSONL renderer
       -> operational renderer
       -> metrics accumulator
       -> generation/client projection
```

Renderers do not infer state by scraping another renderer. The bounded event
queue records overflow; low-priority progress may be coalesced or dropped,
while lifecycle and terminal events remain explicit.

## Streams

- Product result: stdout.
- Product error: stderr.
- Daemon raw console: stdout only when selected.
- Daemon fatal process diagnostic: stderr.

All views respect `NO_COLOR`, TTY detection, explicit byte lengths, and Unicode
boundaries supplied by the tokenizer decoder.

## Non-claims

The local client architecture and bounded loopback adapter do not establish
public HTTP serving, authentication, TLS, remote security, full OpenAI API or
Anthropic compatibility, continuous batching, model quality, benchmark
authority, or release qualification.
