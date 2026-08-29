# Local Protocol v16

Status: normative private protocol contract

Schema/version: `YVEX_LOCAL_PROTOCOL_VERSION = 16`.

Authority: `include/yvex/server.h` and `src/server/protocol.c`. This document
explains the wire and lifecycle contract; code remains authoritative for exact
field layout and bounds.

## Producer and consumer

The foreground `yvex server` mode is the server and runtime authority.
Runtime-client adapters in the same executable and the in-process OpenAI
adapter are clients. The protocol is carried over one private UID-owned
Unix-domain socket and is not a public network API.

## Framing and negotiation

Every connection negotiates version 16 and exchanges bounded typed frames.
Lengths, enums, strings, arrays, message/tool fields, and correlations are
validated before dispatch. Oversized, truncated, duplicate, unknown, or
malformed fields refuse without entering the server scheduler.

Every earlier version, including v15, is refused explicitly. There is no private
pre-v0.1 compatibility decoder. Unknown operations and response kinds fail
closed.

## Operations

Protocol v16 carries host status/stop, engine load/list/unload, model and memory
facts for each engine generation, text or media engine kind, target-only or
speculative text execution strategy,
session lifecycle, bounded copy-on-write session fork, generation turns and
cancellation, speculative lifecycle events, event subscriptions, and composed
console status. Offline compile, artifact, inspect, execute, profile, and system
operations do not cross this protocol.

Every engine-scoped request names a model alias and, after resolution, the exact
process-local engine generation. Alias equality never permits a stale session or
request to continue on a replacement generation. A load or unload operation is
host administration; it is not inferred from a generation request.

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

Native media turns carry server-authored request, conditioning, latent,
decoder, publication, completion, cancellation, and failure progress. These
are control facts, never assistant text. A successful media turn carries one
typed result containing the publication path, resolved geometry, duration,
frame and audio facts, seed, model-evaluation count, engine generation, task,
condition count, byte extent, and the hosted preset, trajectory, RNG, plan,
execution, file, and publication identities. The protocol retains that
identity-bearing result and refuses a media success that omits it. Creative
prompt bytes are model input; the protocol does not parse them into execution
policy.

A media turn may select the explicit preview or released FL2VA trajectory and
may carry paired width/height, duration in milliseconds, and seed facts. An
omitted trajectory selects the released product policy at the media runtime
owner. Width without height, a zero material value, an unknown trajectory, or
media execution facts on a text engine fail before generation. The resolved
request is identity-bearing; no client or server silently reduces geometry,
duration, evaluation count, or seed.

A media request may carry up to two typed image conditions with distinct
`first` and `last` roles. Condition kind, role, admitted media identity, path
extent, and exact path bytes are versioned wire facts. Duplicate roles,
unsupported kinds, malformed extents, or a condition attached to a non-media
request fail before scheduler admission. Conditions are request-owned and do
not alter engine identity or persist in a later turn.

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

`server.status` returns the bounded host snapshot: host readiness, listener
state, queue/worker capacity, engine counts, process memory, and aggregate
lifecycle counters. A healthy host may have zero loaded engines. `server.models`
returns one typed summary per known engine slot, including alias, generation,
state, target, backend, engine kind, execution strategy, capacity, memory classes, package/runtime and
specialization identities, session/work counts, and executable readiness.

Text and composite media engines use the same summary while exposing only facts
their engine owner can authenticate. Physical continuous-batching readiness is
per engine and remains false until the engine scheduler and executable path have
proved it; multiple host workers do not manufacture that claim. `console.status`
returns a server-composed snapshot containing, where authoritative:

- host readiness and the selected alias and engine generation;
- live model, artifact, binding, specialization, backend, and context;
- selected execution strategy and speculative policy identity when admitted;
- client attachment and selected session;
- session position, turn count, context and KV use;
- active phase, progress, cancellation, and last terminal facts.

The local registry profile, loaded engine generation, and session binding are
distinct. Values the server cannot own are explicitly unavailable.

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

The protocol may load an admitted registry profile into a new engine generation,
drain and unload that exact generation while leaving the host alive,
create/reset/close generation-bound sessions, fork one idle committed session
into an independently mutable child under an explicit shared-byte budget,
enqueue/cancel generation,
commit runtime state through the engine scheduler, save one immutable model-state
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

Protocol v16 is not a public remote API, authentication protocol, TLS transport,
stable cross-version SDK promise, distributed serving protocol, or model
quality contract. Versioned checkpoints preserve the admitted model and
semantic-session state across restart; the in-memory fork does not create a
durable shared-prefix namespace, CUDA-shared state cache, or cross-process
copy-on-write authority.
