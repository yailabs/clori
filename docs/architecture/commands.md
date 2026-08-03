# Command and Operation Architecture

Status: current implemented architecture

This document owns the implemented command, operation, and output-projection
architecture. The former flat command catalog and selectable
`normal|table|audit` layouts are retired; they are not compatibility surfaces.
The same typed operation and protocol facts drive the interactive console,
semantic watch, human trace, and machine JSONL projections.

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

The local model registry owns complete startup profiles: artifact, runtime
binding, target, backend, and context. `model list` marks which entries have a
complete readable profile, and `model select NAME` atomically copies one into
the private XDG selection for a later `runtime start`. Selection never opens
the model and cannot hot-switch a running daemon. `model show` inspects a
registry entry, `model selected` reads the inert selection, and `runtime model`
reads the model actually open in `yvexd`.

### Hosted startup semantics

There is no independent hosted `load` operation. In the normal product path,
`yvex runtime start` reads the selected profile and executes the sibling
`yvexd` in the foreground. The daemon authenticates those identities, creates
one immutable runtime model, establishes host/device residency, then publishes
readiness. The client prints and flushes the selected profile, target, backend,
generation mode, and context before executing the daemon, so a long admission
never begins without identifying the model being opened. Direct daemon options
remain an advanced administration boundary, not the normal model-selection
workflow.
`yvex chat` and `yvex run` are protocol clients of that resident model; they do
not link into runtime execution or open weights locally. The complete operator
sequence and memory interpretation live in the
[local runtime runbook](../operator-runbook.md).

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
by the REPL. Generated data contains no callbacks, domain logic,
protocol serialization, allocation, or mutable runtime state. Domain owners
retain semantic validation and defaults.

## Canonical projections

### Conversation

The REPL is a linear client attached to the already resident daemon. A compact
vertical attachment block separates the live target, physical variant, runtime,
session, context, memory, and OpenAI-listener facts instead of relying on
terminal wrapping. The complete slash catalog then projects one registry-owned
command and summary per line before the stable `yvex>` prompt. It streams
committed model text without role labels through a bounded incremental UTF-8
and Markdown renderer; raw output preserves exact canonical bytes. Prefill progress comes from sealed
server events; one inline terminal result renders prompt/reuse/prefill,
generation, TTFT, context, stop, and session facts from the typed protocol
result. When DSpark is active, the same result line projects proposed, accepted,
rejected, and target-verification counts. Candidate token text is never
rendered. Conversation output never includes raw events, logits, tensor facts,
or capability walls.

The registry also owns `/think`, `/think-max`, and `/nothink`. They are
admitted only when the live model advertises the source-authored explicit
reasoning channel. The policy is session-bound, changes prompt identity, and
cannot be switched over existing committed context. Reasoning text is dim on a
TTY and separate from final text; delimiter tags and inferred hidden reasoning
never enter the projection.

The line editor owns bounded in-memory history, registry-derived slash
completion, UTF-8 deletion, bracketed multiline paste, resize redraw, active
screen clearing, and terminal restoration. Ctrl-L clears the visible terminal
and redraws the current prompt and input without changing session state. Ctrl-D
at an idle prompt exits, discarding any unfinished line. Ctrl-C cancels an
active turn; at an idle prompt it clears the line, and a second consecutive
Ctrl-C exits.

### Compact status

Human status names the runtime state, selected logical model and physical
variant, backend, context, sessions, queue, process memory, and resident memory.
It includes at most one blocker and one actionable hint.

### Operational stream

`yvex runtime watch` first renders the current bounded runtime snapshot, then
subscribes to retained operational history and live events. Fixed semantic
categories make startup, readiness, sessions, requests, prefill, DSpark, and
generation visually distinguishable. The stream retains queue events only
when depth exceeds one and otherwise renders operator-significant tokenization,
prefill, first-token, speculative commit, completion, cancellation, failure,
and shutdown facts. Human units, accepted/proposed ratios, and named stop
reasons replace opaque counters. Watch omits connection churn, fragments,
intermediate draft/verification steps, profile rows, trace sequence, severity,
turn, and phase detail. Content remains excluded by default.

### Raw stream

`yvexd --console raw` and `yvex runtime trace --json` serialize the same event
sequence as JSONL. `yvex runtime trace` is the detailed human projection: it
adds sequence, severity, turn, phase, timing, rate, and the same semantic
counter names used by watch. Raw means complete event records, not tensor,
hidden, logits, KV, or memory dumps. Text content is excluded unless
`--trace-content` is explicitly enabled at the host.

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

Human views use one terminal-style owner: cyan for prompt and active work,
green for readiness and completion, orange for warning or cancellation, red
for refusal, and dim text for secondary facts. All views respect `NO_COLOR`,
TTY detection, explicit byte lengths, and Unicode boundaries supplied by the
execute tokenizer decoder. Redirected and machine-readable output contains no
ANSI controls.

## Non-claims

The local client architecture and bounded loopback adapter do not establish
public HTTP serving, authentication, TLS, remote security, full OpenAI API or
Anthropic compatibility, continuous batching, model quality, benchmark
authority, or release qualification.
