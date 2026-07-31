# Command and Operation Architecture

Status: current implemented architecture

This document owns the implemented command, operation, and output-projection
architecture. The former flat command catalog and selectable
`normal|table|audit` layouts are retired; they are not compatibility surfaces.
The final console presentation is deliberately excluded and belongs to the
active [`V010.OPERATOR.REPL.CONSOLE.0`](../milestones/runtime-console-repl.md)
boundary.

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

yvex runtime start|stop|status|model|memory|watch|trace
yvex session new|list|show|attach|detach|reset|close|cancel
yvex model list|show|selected|select
yvex compile ...
yvex artifact show|verify|materialize
yvex help [PATH...]
yvex help --advanced
yvex help --json
yvex completion bash|zsh|fish
yvex version
```

The runtime-facing lane has no dependency edge to engine execution. Direct
component execution, materialization, tokenizer conformance, metadata,
tensor-map, and target-report operations enter a separately guarded finite
offline lane in the same ELF. Retired top-level namespaces refuse with a
migration hint and never execute hidden aliases.

`model select NAME --artifact FILE --runtime-binding FILE --target TARGET
--backend BACKEND --context TOKENS` atomically records one complete private XDG
selection for a later `runtime start`; it never opens the model and cannot
hot-switch a running daemon. `model list` and `model show` inspect registry
entries, `model selected` reads the inert selection, and `runtime model` reads
the model actually open in `yvexd`.

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
yvex compile source|map|quant|emit ...
yvex artifact show|verify|materialize
yvex inspect source|artifact|tensor|tokenizer|target|model|context ...
yvex inspect backend|qtype|quant|attention|moe ...
yvex execute input|tokenizer|artifact|attention|moe|transformer|model ...
yvex profile attention ...
yvex system paths|accounts|cuda
```

This is a visibility-aware finite offline lane, not a renamed developer
namespace or a second process. Its handlers may link engine owners, while
runtime-client handlers continue to cross the local protocol. Default help
stays compact; `yvex help --advanced` exposes supported advanced and
engineering leaves.

## Canonical operation authority

`config/operator/registry.json` is the strict
`yvex.operator.registry.v1` source for operation IDs, paths, visibility,
arguments, flags, requirements, projections, and stable adapter IDs. A bounded
build-time generator validates the source and emits immutable static C
descriptors plus a content identity under `build/generated/operator/`. The
product parses no mutable registry file at runtime.

The compiled descriptors drive path resolution, syntax admission, human help,
`yvex.command.discovery.v1`, shell completion, and the slash catalog consumed
by the transitional REPL. Generated data contains no callbacks, domain logic,
protocol serialization, allocation, or mutable runtime state. Domain owners
retain semantic validation and defaults.

## Canonical projections

### Conversation

The current REPL is a transitional projection over the canonical slash and
protocol schemas. Its present role labels and compact turn summary are not the
intended interface. The successor console will use one `yvex>` prompt, direct
committed model output, semantic progress, and typed final metrics without
changing operation ownership. Conversation output never includes raw events,
logits, tensor facts, or capability walls.

### Compact status

Human status names the runtime state, selected logical model and physical
variant, backend, context, sessions, queue, process memory, and resident memory.
It includes at most one blocker and one actionable hint.

### Operational stream

`yvex runtime watch` subscribes to the typed event sequence. Its current
renderer is transitional: it prints the event kind and generic numeric slots
rather than a complete semantic interpretation. Those slot labels are not an
operator contract. The successor console owns the compact semantic mapping for
startup, queue, prefill, first token, decode, commit, stop, cancellation,
failure, and shutdown. Content remains excluded by default.

### Raw stream

`yvexd --console raw` and `yvex runtime trace --json` serialize the same event
sequence as JSONL. `yvex runtime trace` currently uses the transitional human
projection; the mature semantic trace renderer belongs to the runtime-console
milestone. Raw means complete event records, not tensor, hidden, logits, KV, or
memory dumps. Text content is excluded unless `--trace-content` is explicitly
enabled at the host.

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
[`openai-compatibility.md`](../openai-compatibility.md).

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
boundaries supplied by the execute tokenizer decoder.

## Non-claims

The local client architecture and bounded loopback adapter do not establish
public HTTP serving, authentication, TLS, remote security, full OpenAI API or
Anthropic compatibility, continuous batching, model quality, benchmark
authority, or release qualification.
