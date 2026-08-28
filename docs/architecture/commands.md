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
| `yvex` | unified public command and explicit foreground server mode | client lane: no; server/offline lanes: yes | server lifecycle, conversation, compact status, logs, and explicit offline evidence |

Runtime and model code never writes product output. It publishes typed facts or
typed events. CLI renderers own layout; only CLI I/O owners, including the
explicit server entrypoint, write terminal streams.

## Product grammar

```text
yvex
yvex chat [--session NAME] [--max-new-tokens N]
yvex run [options] TEXT

yvex server [server options]
yvex server load|unload MODEL
yvex server stop|status|models|memory|log [--json]
yvex session new|list|show|attach|detach|reset|close|cancel
yvex model search [QUERY]
yvex model inspect OWNER/REPOSITORY
yvex model acquire ...
yvex model list|show
yvex compile ...
yvex artifact show|verify|materialize
yvex help [PATH...]
yvex help --advanced
yvex help --json
yvex completion bash|zsh|fish
yvex version
```

Runtime-client controls have no dependency edge to engine execution. The
`server` entrypoint is a separately guarded engine-linked foreground host lane
in the same ELF; it is not a client wrapper and does not execute a sibling
binary. Direct component execution, materialization, tokenizer conformance,
metadata, tensor-map, and target-report operations enter the finite offline
lane. Retired top-level namespaces refuse with a migration hint and never
execute hidden aliases.

Remote search and inspection consume provider-neutral typed model and
representation records. Their table, audit, interactive drill-down, and JSON
projections share one domain API; provider output is never parsed from a human
table. Search proves remote availability only. Exact revision acquisition,
source verification, package preparation, registry membership, and live engine
state remain different lifecycle stages. `model list` may project the public
engine inventory when the host is reachable, but it never opens an engine.

The local model registry owns complete typed startup profiles. Text runtimes
carry one artifact, runtime binding, target, backend, generation mode, and
startup context; composite runtimes carry an installed component root, target,
backend, and capability mode. `model list` marks which entries have a complete
readable profile and `model show` inspects one entry. The foreground TTY console
lists complete aliases with `profiles`; its `load MODEL|N` resolves an exact
alias, a numbered row, or an unambiguous runtime target. Multiple profiles for
one target require an exact alias or number; the CLI does not invent a preferred
model policy. The protocol command `server load MODEL` continues to name one
exact profile. No persisted selection is required or consulted. `server models`
reads the engine generations actually known to the host.

### Hosted startup semantics

`yvex server` starts the persistent host without opening a package. On a human
TTY the same foreground process exposes a small operator console for profile
selection, load, inventory, status, unload, and host stop; external protocol
clients remain available concurrently. If the configured socket already owns a
healthy compatible YVEX host, the command attaches that console to the existing
host instead of creating another listener or engine manager. In an attached
console, `exit` detaches while `stop` explicitly shuts down the shared host. A
lifecycle request resolves and
authenticates the named profile, creates one immutable engine generation,
establishes its admitted residency, and publishes engine readiness. Unload
drains and closes that generation without stopping the host. Client requests
bind to an exact alias and engine generation rather than a mutable
process-global model selection.
`yvex chat` and `yvex run` are protocol clients of that resident model; they do
not link into runtime execution or open weights locally. The complete operator
sequence and memory interpretation live in the
[local runtime runbook](../operator-runbook.md).

`yvex` and `yvex chat` require a TTY. `yvex run` is the noninteractive one-shot
form. A missing host produces one concise refusal plus the exact `yvex server`
hint; a missing engine points to `yvex server load MODEL`. Unknown and
duplicate options follow the product parser's typed refusal policy.

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

The REPL is a linear client attached to the already resident server. A compact
vertical attachment block separates the live target, physical variant, runtime,
session, context, memory, and OpenAI-listener facts instead of relying on
terminal wrapping. The complete slash catalog then projects one registry-owned
command and summary per line before a prompt labelled with the attached
engine's typed model alias, for example `deepseek4-v4-flash-dspark>`. The alias
remains stable for that engine generation and gains an explicit
`[disconnected]` marker when transport is lost. It streams
committed model text without role labels through a bounded incremental UTF-8
and Markdown renderer; raw output preserves exact canonical bytes. Prefill
progress comes from sealed server events; one inline terminal result renders
prompt/reuse/prefill, generation, TTFT, context, stop, and session facts from
the typed protocol result. A reasoning turn adds one bounded line with
reasoning/final token counts and rates, TTFR, TTFF, and total completion rate.
When DSpark is active, the result line also projects proposed, accepted,
rejected, and target-verification counts. Candidate token text is never
rendered. Conversation output never includes raw events, logits, tensor facts,
or capability walls.

The registry also owns `/think`, `/think-max`, and `/nothink`. They are
admitted only when the live model advertises the source-authored explicit
reasoning channel. The policy is request-bound and retained as the session's
next-turn selection. A change that rewrites the encoded prefix causes a checked
physical-state rebase and full prefill from authoritative semantic history; it
does not silently reuse incompatible state or require transcript loss.
Reasoning text is dim on a TTY and separate from final text; delimiter tags and
inferred hidden reasoning never enter the projection. A terminal-bound
`yvex run --reasoning none|high|max` uses the same typed presentation and
flushes the final block before its completion summary. When stdout is
redirected, `run` instead emits only canonical channel bytes there and sends
its typed completion summary to stderr, preserving byte-faithful pipelines.

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

`yvex server log` first renders the current bounded runtime snapshot, then
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

`yvex server log --json` serializes the event sequence as canonical JSONL.
Without `--json`, the same public operation renders the compact operational
view. Raw means complete event records, not tensor, hidden, logits, KV, or
memory dumps. Text content is excluded unless `--trace-content` is explicitly
enabled when starting the server.

### Machine status

`yvex server status --json` is a stable protocol-derived object. Human prose
never shares its stdout. Developer owners may expose their own explicitly
versioned JSON or evidence-file schemas.

### Errors

Product errors contain one class/reason and, where known, one remediation hint.
Errors go to stderr. Exit `2` is parser/usage refusal; runtime and protocol
refusals are nonzero without turning diagnostic evidence into normal output.

### Application protocol

The OpenAI adapter inside the foreground `yvex server` process is not a third
terminal renderer. It returns
the documented compatibility JSON or SSE schema over loopback HTTP. Its
response objects project typed provider and YVEX protocol facts; they never
scrape CLI or server-console text. The exact profile lives in
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
