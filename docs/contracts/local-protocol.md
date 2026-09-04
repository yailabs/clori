# Local Protocol v20

Status: normative private protocol contract

Schema/version: `YVEX_LOCAL_PROTOCOL_VERSION = 20`.

Authority: `include/yvex/server.h` and `src/server/protocol.c`. This document
explains the wire and lifecycle contract; code remains authoritative for exact
field layout and bounds.

## Producer and consumer

The foreground `yvex serve` mode is the server and runtime authority.
Runtime-client adapters in the same executable and the in-process OpenAI
adapter are clients. The protocol is carried over one private UID-owned
Unix-domain socket and is not a public network API.

## Framing and negotiation

Every connection negotiates version 20 and exchanges bounded typed frames.
Lengths, enums, strings, arrays, message/tool fields, and correlations are
validated before dispatch. Oversized, truncated, duplicate, unknown, or
malformed fields refuse without entering the server scheduler.

Every earlier version, including v19, is refused explicitly. There is no private
pre-v0.1 compatibility decoder. Unknown operations and response kinds fail
closed.

## Operations

Protocol v20 carries host status/stop, engine load/list/unload, demand-active
model lease acquire/release, model and memory
facts for each engine generation, text or media engine kind, target-only or
speculative text execution strategy,
session lifecycle, bounded copy-on-write session fork, ordered typed-content
generation turns and
cancellation, speculative lifecycle events, event subscriptions, and composed
console status. Offline compile, artifact, inspect, execute, profile, and system
operations do not cross this protocol.

Every engine-scoped request names a model alias and, after resolution, the exact
process-local engine generation. Alias equality never permits a stale session or
request to continue on a replacement generation. A load or unload operation is
host administration. An explicit ensure-active operation may invoke that same
lifecycle owner and returns an exact lease; it is never inferred from a
generation request or from content.

The removed model/artifact facade operation values are absent. Artifact
inspection is an offline-engine operation; live model inspection comes from
the runtime owner.

## Ordered content and provenance

A native generation turn may carry one ordered collection of at most 32 typed
parts. Each part identifies its schema, kind (`text`, `image`, `audio`, `video`,
`file`, or `tensor`), storage form, byte extent, media type, optional shape or
duration, content digest, and optional `derived_from_content_identity`. The
collection identity binds order and metadata as well as the individual content
identities. Original media and a derived representation such as a transcript
remain distinct parts linked by that provenance identity.

Inline wire payload admits bounded raw bytes without a base64 representation.
The reference CLI uses an absolute local-file reference for non-text content
over the UID-owned Unix socket: it seals the regular file's exact byte extent
and SHA-256, and the server reopens it without following a final symlink and
verifies both facts before scheduler admission. Large media therefore need not
be copied into the protocol frame or expanded through JSON. The reference is
local and process-lifetime transport state, not a portable artifact identity or
a remote capability. A specialization/content mismatch fails before numerical
execution; no media is silently converted to text.

Legacy prompt bytes and provider requests remain mutually exclusive with typed
content. Existing text-only callers retain their byte-exact path.

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
events from the same telemetry authority as `host logs`. Prefill-start,
per-chunk progress, and completion facts let the console update one truthful
line without timing the asynchronous request locally. Provider/OpenAI requests
retain their provider stream contract and do not receive these native console
messages.

Engine preparation and load publish the same server-authored progress contract.
Artifact verification reports real bytes and residency reports real tensors
when their owners know a denominator. Binding, opening, admission,
materialization, sealing, backend open, and workspace preparation otherwise
report phase and elapsed activity without fabricating a percentage. Replaceable
progress is coalesced; lifecycle completion and failure remain retained.

The shared event stream also carries low-frequency rolling decode progress with
committed count, sequence position, elapsed/rate, reasoning count and optional
cumulative speculative economics. Normal cadence is bounded to one second or
64 newly committed tokens, whichever first makes progress observable. Token
fragments remain available only at token/full trace levels and are never the
operational progress authority.

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

The admitted tokenizer contract classifies source-authored explicit reasoning
separately from final text. Protocol v20 permits an omitted policy to remain
`source-default` until the exact loaded model resolves it; concrete `disabled`,
`low`, `enabled`, and `maximum` choices remain request facts. Provider request
v4 independently carries source-default/drop/preserve reasoning-history policy.
The family conversation descriptor owns the exact generation prefix and full
reasoning-to-answer delimiter, including structural whitespace. Delimiter bytes
are consumed by that owner and never enter either projection. A reasoning
stream that terminates without the required transition fails as incomplete;
disabled reasoning is final-text only. Clients may not infer hidden reasoning,
inspect arbitrary prose, or search output for delimiter-looking strings.

DSpark proposal tokens are not stream fragments. Drafting, verification, and
accepted-prefix events carry typed counters, but final text is emitted only
after the target-authored prefix and matching session state commit. A client
never retracts a candidate because no candidate is published.

## Status and console facts

`server.status` returns the bounded host snapshot: host readiness, listener
state, queue/worker capacity, engine counts, process memory, and aggregate
lifecycle counters. A healthy host may have zero loaded engines. `server.models`
returns one typed summary per known engine slot, including alias, generation,
state, target, backend, engine kind, execution strategy, capacity, memory
classes, package/runtime and specialization identities, attached-client and
model-lease counts, directional input/output capabilities, session/work counts,
and executable readiness. `model active` filters this same typed authority to
loaded/draining/unloading generations; it never scrapes a renderer or guesses
physical residency.

The offline model/profile catalog publishes the same capability schema on each
launchable deployment. This lets an orchestrator discover a READY model before
requesting ensure-active; activation does not create or upgrade capability.

Text and composite media engines use the same summary while exposing only facts
their engine owner can authenticate. Capacity schema v1 separates session
capacity, scheduler-visible runnable work, admitted physical sequence width,
cooperative scheduling, compatible-operation batching, and dynamic continuous
batching. The latter remains false until ready sequences can enter and leave
physical decode batches; multiple host workers or compatible row coalescing do
not manufacture that claim. `console.status`
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
publication timing where owned. Version 17 additionally carries the initial
position, whether the client explicitly requested an output bound, that exact
request when present, and the runtime-resolved maximum completion envelope.
An omitted native or provider limit remains zero on input and is resolved by
the server from the engine bound and remaining context; it is not replaced by a
CLI default. A speculative result additionally carries draft cycles and forwards,
proposed and selected-verification tokens, target verifications, accepted,
rejected and discarded drafts, correction/bonus tokens, maximum and mean
accepted prefix, confidence facts, separate draft/verification/commit timing,
effective committed rate, and policy identity. Exact seconds are never
reconstructed from rounded rates.

Protocol v20 retains measurement schema v1. Each record identifies
its phase scope, host/device clock, top-level/nested/enclosing/overlapping
composition, work unit, and availability. A cumulative rate uses the complete
declared work/duration denominator; rolling decode uses its own recent work and
duration, with a bounded 32-token window for current text generation. Missing
or overlapping phase facts remain explicit rather than being forced into an
invalid wall-time sum.

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
ensure a named READY deployment has an active engine through the existing
model loader, acquire/release an exact model lease,
create/reset/close generation-bound sessions, fork one idle committed session
into an independently mutable child under an explicit shared-byte budget,
enqueue/cancel generation,
commit runtime state through the engine scheduler, save one immutable model-state
checkpoint, restore it at the exact current semantic-session position, publish
events, or initiate bounded server shutdown. State checkpoint messages carry
the file digest, byte extent, scope count, committed position, and bound model,
binding, and artifact identities. Parsing and status operations do not open
artifacts or execute model work locally in the client.

A model lease protects only one exact engine generation. It neither selects nor
rebinds a conversational session. Unload refuses while any session or model
lease still owns the generation; no implicit eviction policy is introduced.

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

## Resource truth

Resource schema v1 distinguishes immutable artifact and mapped bytes, prepared
model state, explicit host/device allocation, device-addressable mappings,
typed attention/recurrent/convolution/candidate session state, physical session
state, activation arenas, reusable workspace, transients, process RSS, current
and peak facts, and logical movement. Availability and placement are explicit.
On unified-memory systems, device addressability does not become a claim about
physical page residency; an unknown physical working set remains unmeasured.
Overlapping model spans, typed session subsets, and peak classes must not be
summed into a synthetic total.

## Non-claims

Protocol v20 is not a public remote API, authentication protocol, TLS transport,
stable cross-version SDK promise, distributed serving protocol, or model
quality contract. Versioned checkpoints preserve the admitted model and
semantic-session state across restart; the in-memory fork does not create a
durable shared-prefix namespace, CUDA-shared state cache, or cross-process
copy-on-write authority.
