# Hosted Runtime Contract

Status: normative implemented contract

Authority: hosted model, session, execution, publication, failure, and cleanup
semantics. C interfaces are documented separately in [YVEX C API](c-api.md).

## Parties and scope

Producer: foreground `yvex server`, common runtime, graph/backend, tokenizer,
generation, and server owners.

Consumers: runtime-client operations and the interactive console in `yvex`,
the in-process OpenAI adapter, and focused runtime tests.

The contract begins with one admitted complete artifact, exact runtime binding,
Physical Execution IR and compiled execution profile and ends with typed
state/results, committed text, events, or refusal.
It does not define compilation, command grammar, HTTP syntax, evaluation,
benchmark policy, or release state.

## Runtime model lifecycle

One hosted process owns one immutable runtime model:

```text
closed -> opening -> authenticated -> materialized -> ready -> stopping -> closed
```

Opening validates artifact snapshot and identity, runtime binding identity and
schema, model descriptor, physical variant, Physical Execution IR,
qtype/backend prerequisites, compiled profile, and resource budgets before
readiness. The runtime imports compiler facts; it does not reconstruct
transformation or writer plans.

Startup performs bounded binding admission before artifact open. The retained
system reserve is the greater of 8 GiB and one eighth of the effective memory
capacity, where the effective capacity is constrained by the caller's host
budget and the process's complete cgroup-v2 `memory.max`/`memory.high`
hierarchy. When the binding's admitted resident payload plus that reserve
exceeds the configured or currently available extent, opening refuses with the
exact capacity component and byte extents. The same live check runs again after
artifact authentication and immediately before residency allocation, so memory
consumed during a long hash cannot turn a previously valid observation into an
OOM allocation. No materialization arena or model residency is created after
either refusal.

CUDA free memory is a separate constraint unless backend facts prove that the
selected managed placement and the host use one physical memory domain. That
proof requires unified addressing, managed access, and an exact match between
effective system and device capacity. In the shared-domain case Linux `MemAvailable`,
already bounded by the cgroup hierarchy, owns reclaimable-capacity admission;
`cuMemGetInfo` is not allowed to turn reclaimable page cache into a false
refusal. A shared CUDA session inherits the same immutable domain facts from
its model-owned context.

The model owns model-lifetime artifact/binding handles, encoded weights,
backend resources, tokenizer plan, output-head residency, immutable execution
descriptors, target and draft plans, and shared caches. A DSpark plan shares the
target model, tokenizer, output head, backend context, and immutable residency;
it is not a second runtime model. Readiness is published only after every
requirement of the selected generation mode and the worker is usable.

Failure during opening publishes no ready model and releases every acquired
resource transactionally. Shutdown closes the model once after request/session
admission stops and the worker has resolved bounded work.

## Session lifecycle

Each server session owns one mutable runtime execution session plus token
ledger, transcript, tokenizer decoder, sampling state, turn state, and
persistent sequence state. Sessions sharing one model share no mutable state.

```text
absent -> created -> attached/detached -> active turn -> retained -> reset/closed
```

A client connection is not a session. Disconnect and detach do not reset or
close retained state. Reset clears committed sequence/text/sampling state while
retaining compatible allocation; close releases the session. Repeated close
and cleanup are bounded and safe.

## Prompt and prefix admission

The server renders and tokenizes the complete conversational prompt. Reuse is
admitted only when the committed token ledger is an exact prefix of the new
token sequence. The runtime prefills only the suffix. Position, reusable token
count, and new-prefill token count are authoritative server/runtime facts.

An incompatible prefix, context overflow, malformed UTF-8, unsupported prompt
template, or invalid session state refuses before mutation. Reset is explicit;
the runtime never silently discards committed state to make a turn fit.

## Transformer execution

Prefill and decode are explicit phases over one admitted Transformer owner.
The execution consumes exact token input, immutable model resources, reusable
workspace, and a candidate persistent-state transaction. It executes selected
embedding, the ordered layer stack, family-correct attention/state, FFN or MoE,
residual composition, final collapse, and final normalization.

The normalized hidden result is not logits or text. Output-head projection
consumes it once and publishes a complete finite vocabulary row. Sampling
consumes the complete admitted row. A selected ordinary token is fed through
decode exactly once before it is appended and detokenized. EOS or tokenizer
stop tokens terminate according to the tokenizer contract without fabricated
state advancement.

CPU and CUDA consume the same typed logical contract. Numerical device values
carry explicit typed views; scalar, row, audit-digest, and forensic-full host
materialization are distinct requests. Production CUDA greedy selection
returns a bounded token/status result without full-vocabulary host transfer.
The exact host stochastic path remains an explicit portable-reference class.
Unsupported CUDA qtypes, operations, modes, workspace, or resources refuse;
no explicit CUDA request falls back to CPU.

## Generation modes and verification

One model may admit `target-only` and `dspark` modes. The selected mode is a
runtime-profile fact and appears in live status and turn results. Target-only
is the ordinary target reference path. DSpark is admitted only when artifact,
binding, draft plan, target-verification plan, qtypes, workspace, and backend
capabilities all agree. An explicit DSpark request never falls back silently.

A DSpark cycle produces a bounded candidate block and confidence facts without
mutating committed target truth. The complete target evaluates the ordered
candidate prefix. Greedy verification accepts exact target-token matches.
Stochastic verification uses the admitted target-distribution-preserving
accept/reject and residual-sampling rule; drafter confidence and decoded text
are never correctness authorities.

`decode_step_count` retains its sequence meaning: it counts target-verified
positions committed to model state. It is not a target-forward counter. Draft
forwards and target block verifications have separate counters because one
verification may commit several positions.

## Persistent state transaction

Persistent sequence state is session-owned. The state provider and backend
residency coordinate the same candidate generation:

```text
validate input and capacity
-> begin candidate
-> execute and stage output/state
-> validate finite/numeric/device status
-> check cancellation
-> publish output and candidate state
-> commit logical state
```

Failure aborts the candidate. Earlier successful chunks or tokens remain
committed, and the result identifies the exact completed prefix and first
incomplete unit. State position and generation come from the committed owner,
not from a renderer or decode-local counter.

Speculation uses one bounded prefix-addressable candidate transaction. Ordered
candidate deltas retain the target checkpoint after every verified position;
SWA uses a ring projection while compressed, indexer and rolling state retain
exact per-position boundaries. The transaction covers target attention/KV
state, DSpark candidate state, position, token ledger, decoder, generated text,
target and draft RNG state, stop class, and turn identities. It promotes the
checkpoint for exactly the accepted target-authored prefix plus any separately
executed correction or bonus token defined by the algorithm. Accepted target
rows are not replayed, rejected suffixes are discarded, and rollback never
means decrementing counters after publication.

An idle, complete paged provider may capture its committed attention state as
an immutable in-memory prefix. Capture is admitted against an explicit byte
budget and seals the state-layout, capacity-plan, content and backing
identities before publication. A compatible empty provider may attach that
backing without charging it as private residency. Shared pages remain
immutable; the first write to an attached tail faults a private copy, so a
child extension cannot mutate its source or another child. Incompatible
geometry, identity, capacity or destination state refuses before attachment.
Reset, invalidation and close release references through the prefix owner.

This in-memory prefix contract is not durable persistence. The server session
lifecycle combines it with a deep semantic clone of the token ledger, RNG,
decoder, transcript and conversation state, then publishes one child only after
both physical and semantic extents agree. Persisted state continues to use the
separate versioned checkpoint/store contract.

## Execution and evidence admission

Every request consumes one immutable compiled execution profile binding the
logical model, physical variant, Physical Execution IR, artifact,
materialization, runtime binding, kernel bundle, hardware, context, mode,
workload, evidence profile, and execution class. Profile v2 also records typed
attention, MoE, and sampling resolutions. An executable resolution is either
`exact` or `compatible-degraded`; resource pressure, unsupported capability,
and trust failure cannot be sealed as an executable profile. The aggregate
resolution is derived from those three operation facts rather than supplied as
a second authority. Backend owners report capability facts but never select a
resolution. `production` admits bounded
status and accounting only; `audit` may add selected probes and device digests;
`forensic` may explicitly materialize full intermediates. Trace verbosity does
not select evidence depth, and production identity does not require complete
hidden, logits, probability, state, or event hashing.

Forensic CUDA attention comparison may select the canonical-order numerical
adapter rather than the production Q8-activation/F32 implementation. The
adapter exists only to compare every intermediate against the independent
reference contract; it cannot be selected by a production execution profile.

Descriptor-bearing execution also consumes independently sealed hardware and
workload profiles plus one capacity plan. Context limits, pooled state,
candidate and prefix reserves, logical batching, physical row geometry,
workspace and system reserve remain separately identified. Persistent-state
page tokens are a per-state-class planning result, not a global runtime flag.
The model-authored maximum is sealed upstream in Semantic Model IR and must
match the target/draft maxima in the compiled context envelope. Requested
context is owned by the workload profile; hardware-fit admission remains the
generic capacity planner's responsibility. No family projection owns selected
capacity or a machine-memory constant.

The identity-bearing capacity plan uses stable physical and configured-budget
facts. Before generation allocates session state or workspace, a separate
transient preflight compares the plan's non-weight requirement with current
system/cgroup availability and, for CUDA, current free device memory. Transient
free bytes therefore prevent overcommit without changing page geometry,
capacity identity, or persistent-state compatibility between requests and
restarts.

Target prefill/decode, draft width five, verification widths two through six,
correction, and reset select an execution shape before mutation. The key binds
scope, phase, operation, width, context band, visibility, capacity, workspace,
attention/state/kernel identities, and evidence profile. The runtime may admit
a compatible eager/reference shape before execution; otherwise refusal names
the exact capacity component, configured and required values, position, width,
scope and shape/workspace/state identities. It never changes the active shape
after output begins.

## Generation and text publication

One generation turn composes tokenizer, prompt rendering, prefix admission,
prefill, logits, sampling, decode, stop classification, incremental decoding,
and server publication. It does not create alternative implementations of
those domains.

Fragments are published only after:

1. model execution has committed the corresponding state;
2. the incremental decoder has accepted the token bytes; and
3. the internal text ledger has committed the same fragment.

Sink failure or disconnect stops further publication and preserves exact
model-committed partial state. The typed partial result records committed
position/tokens/text, participating state generations and identities, failure
class, and recovery requirement. A partial session is visibly distinct,
refuses ordinary continuation, and requires explicit reset unless a future
versioned recovery operation is admitted. A partial turn is never labeled
complete.

Draft candidates are not fragments. They never enter native streaming, HTTP
or SSE output, transcript, completion usage, or ordinary generated-token
counts. Those surfaces count only target-verified committed tokens.

## Cancellation

Cancellation is server-owned and correlated to the exact session/request/turn.
It remains observable during tokenization, prefill, Transformer/MoE execution,
decode, output-head projection, sampling, and publication at the bounded safe
points provided by those owners.

A cancellation during drafting or verification discards uncommitted candidate
state. A cancellation after atomic accepted-prefix commit reports that exact
committed prefix. Every cancelled request publishes a typed cancellation
class, completed token and position facts, and no false terminal success. It
does not close the server, model, or unrelated sessions, poison immutable
caches, or prevent reset and a subsequent request.

## Resource and concurrency rules

The server owns one bounded queue and one scheduler mutation authority.
Listener threads admit, frame, and project requests but never mutate model
state directly. A bounded worker set may execute distinct session keys
concurrently; requests sharing a session key remain strictly serialized. This
is independent-session scheduling, not compatible-row continuous batching.
The startup capacity plan accounts the admitted worker count before readiness
and the protocol exposes both capabilities separately.

Prepared warm execution reuses immutable weights, output-head residency,
workspace, persistent-state allocation, and stable execution resources within
their admitted capacities. A request outside capacity refuses rather than
silently resizing a stable or captured execution.

Host mapping, anonymous host residency, CUDA-addressable storage,
accelerator-resident storage, KV, workspace, and staging remain separately
accounted. Unified physical memory does not erase placement or movement facts.

## Inputs and outputs

Inputs include exact model/session/request identities, prompt or provider
messages, generation policy, token/context limits, cancellation correlation,
and output sink. Semantic defaults come from typed generation/sampling owners.

Outputs include streamed channel fragments, complete turn result, usage,
prompt/reuse/prefill/generation facts, TTFT, stop/cancellation class, final
position, state/turn identity, and typed errors. Speculative results also name
the execution mode and policy and report draft cycles/forwards, proposed and
selected tokens, target verifications, accepted and rejected drafts,
correction/bonus tokens, accepted-prefix statistics, and separate draft,
verification, and commit durations. Unavailable data is marked unavailable;
zero is retained for real zero values.

## Side effects

Admitted side effects are session creation/reset/close, persistent-state and
token-ledger commit, bounded model/session memory use, event publication, local
socket/loopback output, and optional explicitly configured traces. The runtime
does not modify source snapshots or artifacts.

## Failure and refusal

Failure classes remain owned and distinguishable: input, source/artifact drift,
binding incompatibility, capability, resource, backend, numerical, state,
cancellation, timeout, transport, sink, and cleanup. Error text is a projection
and never becomes the classifier.

No output or state is published before its producer completes. Cleanup failure
is reported without pretending the owner was released. Malformed or hostile
requests cannot terminate the server or corrupt another session.

Malformed draft geometry, feature taps, noise token, Markov rank, missing
companions, unsupported draft qtypes, candidate workspace overflow,
verification-state mismatch, RNG mismatch, and rejected-state reuse are typed
refusals or failures. None may become a successful target-only turn.

## Compatibility

The runtime behavior is consumed through private local protocol v10 and the
bounded OpenAI compatibility profile v2. Public C ABI and internal ABI follow
their header/version contracts. Pre-v0.1 private protocol versions may be
refused rather than decoded compatibly.

## Explicit non-claims

This contract does not establish public/remote serving, authentication, TLS,
continuous batching, multi-model hosting, distributed execution,
restart-persistent sessions, complete accelerator residency, complete
device-side stochastic sampling/tokenization, load-aware confidence scheduling, DSpark acceleration,
model evaluation, release benchmark performance, or release qualification.
