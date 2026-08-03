# Local Protocol v5

Status: normative private protocol contract

Schema/version: `YVEX_CLIENT_PROTOCOL_VERSION = 5`.

Authority: `include/yvex/server.h` and `src/server/protocol.c`. This document
explains the wire and lifecycle contract; code remains authoritative for exact
field layout and bounds.

## Producer and consumer

`yvexd` is the server and runtime authority. Runtime-client adapters in `yvex`
and the in-process OpenAI adapter are clients. The protocol is carried over one
private UID-owned Unix-domain socket and is not a public network API.

## Framing and negotiation

Every connection negotiates version 5 and exchanges bounded typed frames.
Lengths, enums, strings, arrays, message/tool fields, and correlations are
validated before dispatch. Oversized, truncated, duplicate, unknown, or
malformed fields refuse without entering the model worker.

Version 4 is refused explicitly. There is no private pre-v0.1 compatibility
decoder. Unknown operations and response kinds fail closed.

## Operations

Protocol v5 carries runtime start-state/status/stop, live model and memory
facts, selected target-only or DSpark generation mode, session lifecycle,
generation turns and cancellation, speculative lifecycle events, event
subscriptions, and composed console status. Offline compile, artifact,
inspect, execute, profile, and system operations do not cross this protocol.

The removed model/artifact facade operation values are absent. Artifact
inspection is an offline-engine operation; live model inspection comes from
the runtime owner.

## Request lifecycle

```text
decode and admit frame
-> correlate client/session/request/turn
-> enqueue one typed server operation
-> publish accepted/progress/fragments/terminal result
-> close or retain connection by operation contract
```

Connections do not own runtime sessions. Detach or disconnect leaves admitted
session state under the server owner. Cancellation targets the exact active
request/turn and does not infer identity from connection lifetime.

## Streaming

Generation may return accepted, progress, committed fragments, and one terminal
result. Fragment bytes carry explicit lengths and a typed stream channel:
final text, explicit model-emitted reasoning when supported, tool call, tool
result, or control/event.

Native generation connections receive identity-sealed tokenizer and prefill
events from the same telemetry authority as watch and trace. Prefill-start,
per-chunk progress, and completion facts let the console update one truthful
line without timing the asynchronous request locally. Provider/OpenAI requests
retain their provider stream contract and do not receive these native console
messages.

Ordinary current DeepSeek text uses the final-text channel. An unavailable
explicit-reasoning channel remains unavailable; clients may not infer hidden
reasoning or classify prose by style.

DSpark proposal tokens are not stream fragments. Drafting, verification, and
accepted-prefix events carry typed counters, but final text is emitted only
after the target-authored prefix and matching session state commit. A client
never retracts a candidate because no candidate is published.

## Status and console facts

`runtime.status` returns the bounded runtime snapshot. `console.status` returns
a server-composed snapshot containing, where authoritative:

- readiness and runtime configuration;
- live model, artifact, binding, physical variant, backend, and context;
- selected generation mode and speculative policy identity when admitted;
- client attachment and selected session;
- session position, turn count, context and KV use;
- active phase, progress, cancellation, and last terminal facts.

Selected startup configuration and the live runtime model are distinct. Values
the server cannot own are explicitly unavailable.

## Turn result

The terminal generation result carries exact prompt, reuse and new-prefill
token counts; prefill time/rate; TTFT; generated-token count; generation/decode
time/rate; inter-token facts where available; final position; stop or
cancellation class; usage; state/turn identity; and publication timing where
owned. A speculative result additionally carries draft cycles and forwards,
proposed and selected-verification tokens, target verifications, accepted,
rejected and discarded drafts, correction/bonus tokens, maximum and mean
accepted prefix, confidence facts, separate draft/verification/commit timing,
effective committed rate, and policy identity. Exact seconds are never
reconstructed from rounded rates.

Existing generated-token and completion-usage fields retain their meaning:
only target-verified committed tokens count. Proposal counts remain separate.
The runtime's decode-step count likewise denotes committed sequence positions;
draft-forward and target-verification counts report execution calls separately.

## Side effects

The protocol may create/reset/close sessions, enqueue/cancel generation,
commit runtime state through the worker, publish events, or initiate bounded
daemon shutdown. Parsing and status operations do not open artifacts or execute
model work locally in the client.

## Failure and cleanup

Failures preserve typed application, input, runtime, capability, resource,
timeout, cancellation, transport, and internal classes. A malformed client,
write failure, or disconnect closes only its connection and correlated work as
required. The daemon and other sessions survive.

Cancellation during draft or verification discards uncommitted candidate
state. If an accepted prefix committed before cancellation, the terminal
message reports that exact partial progress. Rejected or disconnected work
cannot make proposal state visible to a later request.

Frame buffers, subscriber state, and connection resources are bounded and
released on every exit path. No terminal success precedes model completion.

## Compatibility policy

This protocol is private, local, UID-scoped, and pre-v0.1. Schema changes bump
the version when field meaning or operation semantics become incompatible.
HTTP/OpenAI compatibility has its own version and does not expose this wire
format.

## Non-claims

Protocol v5 is not a public remote API, authentication protocol, TLS transport,
stable cross-version SDK promise, distributed serving protocol, or model
quality contract.
