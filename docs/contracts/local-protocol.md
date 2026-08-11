# Local Protocol v10

Status: normative private protocol contract

Schema/version: `YVEX_LOCAL_PROTOCOL_VERSION = 10`.

Authority: `include/yvex/server.h` and `src/server/protocol.c`. This document
explains the wire and lifecycle contract; code remains authoritative for exact
field layout and bounds.

## Producer and consumer

The foreground `yvex server` mode is the server and runtime authority.
Runtime-client adapters in the same executable and the in-process OpenAI
adapter are clients. The protocol is carried over one private UID-owned
Unix-domain socket and is not a public network API.

## Framing and negotiation

Every connection negotiates version 10 and exchanges bounded typed frames.
Lengths, enums, strings, arrays, message/tool fields, and correlations are
validated before dispatch. Oversized, truncated, duplicate, unknown, or
malformed fields refuse without entering the server scheduler.

Every earlier version, including v9, is refused explicitly. There is no private
pre-v0.1 compatibility decoder. Unknown operations and response kinds fail
closed.

## Operations

Protocol v10 carries server status/stop, live model and memory
facts, selected target-only or DSpark generation mode, session lifecycle,
bounded copy-on-write session fork, generation turns and cancellation,
speculative lifecycle events, event subscriptions, and composed console status.
Offline compile, artifact,
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
result, control/event, or error. Output kind and stream channel must agree;
diagnostic text cannot reclassify a fragment.

Native generation connections receive identity-sealed tokenizer and prefill
events from the same telemetry authority as `server log`. Prefill-start,
per-chunk progress, and completion facts let the console update one truthful
line without timing the asynchronous request locally. Provider/OpenAI requests
retain their provider stream contract and do not receive these native console
messages.

The admitted DeepSeek DSpark tokenizer contract classifies source-authored
explicit reasoning separately from final text when the request selects
`enabled` or `maximum`. The opening `<think>` token is part of the admitted
generation prompt. Only the corresponding source-authored `</think>` token
transitions the output stream to final text. Delimiter bytes are consumed by
that owner and never enter either projection. An enabled stream that terminates
without the transition fails as an incomplete grammar. Disabled reasoning is
final-text only. Clients may not infer hidden reasoning, inspect arbitrary
prose, or search output for delimiter-looking strings.

DSpark proposal tokens are not stream fragments. Drafting, verification, and
accepted-prefix events carry typed counters, but final text is emitted only
after the target-authored prefix and matching session state commit. A client
never retracts a candidate because no candidate is published.

## Status and console facts

`server.status` returns the bounded hosted-runtime snapshot. It includes the
identity-bound startup capacity plan, its required and unreserved bytes, the
admitted concurrent-sequence count, and separate facts for independent-session
scheduling and physical continuous batching. The latter remains false until a
compatible-row generation scheduler is admitted; multiple server workers do
not manufacture that claim. `console.status` returns
a server-composed snapshot containing, where authoritative:

- readiness and runtime configuration;
- live model, artifact, binding, physical variant, backend, and context;
- selected generation mode and speculative policy identity when admitted;
- client attachment and selected session;
- session position, turn count, context and KV use;
- active phase, progress, cancellation, and last terminal facts.

The named registry profile and the live runtime model are distinct. Values the
server cannot own are explicitly unavailable.

## Turn result

The terminal generation result carries exact prompt, reuse and new-prefill
token counts; prefill time/rate; TTFT; generated-token count; generation/decode
time/rate; reasoning and final token counts and rates; time to first reasoning
and final token; total completion time/rate; inter-token facts where available;
final position; stop or cancellation class; usage; state/turn identity; and
publication timing where owned. A speculative result additionally carries draft cycles and forwards,
proposed and selected-verification tokens, target verifications, accepted,
rejected and discarded drafts, correction/bonus tokens, maximum and mean
accepted prefix, confidence facts, separate draft/verification/commit timing,
effective committed rate, and policy identity. Exact seconds are never
reconstructed from rounded rates.

Existing generated-token and completion-usage fields retain their meaning:
only target-verified committed tokens count. Proposal counts remain separate.
The runtime's decode-step count likewise denotes committed sequence positions;
draft-forward and target-verification counts report execution calls separately.

A terminal failure after committed output carries
`yvex.client.partial-turn.v1`: availability, committed-progress and
reset-required flags; initial/final position; committed-token and published-byte
counts; target, draft, RNG, ledger, decoder, history and transcript generations
where authoritative; state identities; failure class; and stop reason. This
snapshot is distinct from cancellation and from a failure with no committed
progress. A partial session refuses an ordinary turn until reset.

## Side effects

The protocol may create/reset/close sessions, fork one idle committed session
into an independently mutable child under an explicit shared-byte budget,
enqueue/cancel generation,
commit runtime state through the keyed scheduler, save one immutable model-state
checkpoint, restore it at the exact current semantic-session position, publish
events, or initiate bounded server shutdown. State checkpoint messages carry
the file digest, byte extent, scope count, committed position, and bound model,
binding, and artifact identities. Parsing and status operations do not open
artifacts or execute model work locally in the client.

## Failure and cleanup

Failures preserve typed application, input, runtime, capability, resource,
timeout, cancellation, transport, and internal classes. A malformed client,
write failure, or disconnect closes only its connection and correlated work as
required. The server and other sessions survive.

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

Protocol v10 is not a public remote API, authentication protocol, TLS transport,
stable cross-version SDK promise, distributed serving protocol, or model
quality contract. Versioned checkpoints preserve the admitted model and
semantic-session state across restart; the in-memory fork does not create a
durable shared-prefix namespace, CUDA-shared state cache, or cross-process
copy-on-write authority.
