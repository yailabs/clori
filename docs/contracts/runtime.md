# Hosted Runtime Contract

Status: normative implemented contract

Authority: hosted model, session, execution, publication, failure, and cleanup
semantics. C interfaces are documented separately in [YVEX C API](c-api.md).

## Parties and scope

Producer: `yvexd`, common runtime, graph/backend, tokenizer, generation, and
server owners.

Consumers: runtime-client operations in `yvex`, the in-process OpenAI adapter,
focused runtime tests, and future console renderers.

The contract begins with one admitted complete artifact and exact runtime
binding and ends with typed state/results, committed text, events, or refusal.
It does not define compilation, command grammar, HTTP syntax, evaluation,
benchmark policy, or release state.

## Runtime model lifecycle

One hosted process owns one immutable runtime model:

```text
closed -> opening -> authenticated -> materialized -> ready -> stopping -> closed
```

Opening validates artifact snapshot and identity, runtime binding identity and
schema, model descriptor, physical variant, qtype/backend prerequisites, and
resource budgets before readiness. The runtime imports compiler facts; it does
not reconstruct transformation or writer plans.

The model owns model-lifetime artifact/binding handles, encoded weights,
backend resources, tokenizer plan, output-head residency, immutable execution
descriptors, and shared caches. Readiness is published only after required
resources and the worker are usable.

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

CPU and CUDA consume the same typed logical contract. Unsupported CUDA qtypes,
operations, modes, workspace, or resources refuse; no explicit CUDA request
falls back to CPU.

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
model-committed partial state. A partial turn is never labeled complete.

## Cancellation

Cancellation is server-owned and correlated to the exact session/request/turn.
It remains observable during tokenization, prefill, Transformer/MoE execution,
decode, output-head projection, sampling, and publication at the bounded safe
points provided by those owners.

A cancelled request publishes a typed cancellation class, completed token and
position facts, and no false terminal success. It does not close the daemon,
model, or unrelated sessions, poison immutable caches, or prevent a subsequent
request.

## Resource and concurrency rules

The daemon owns one bounded queue and one model worker. Listener threads admit,
frame, and project requests but never mutate model state directly. One active
generation request is executed at a time unless a later independently admitted
scheduler changes that contract.

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
position, state/turn identity, and typed errors. Unavailable data is marked
unavailable; zero is retained for real zero values.

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
requests cannot terminate the daemon or corrupt another session.

## Compatibility

The runtime behavior is consumed through private local protocol v4 and the
bounded OpenAI compatibility profile v1. Public C ABI and internal ABI follow
their header/version contracts. Pre-v0.1 private protocol versions may be
refused rather than decoded compatibly.

## Explicit non-claims

This contract does not establish public/remote serving, authentication, TLS,
continuous batching, multi-model hosting, distributed execution,
restart-persistent sessions, complete accelerator residency, device-side
sampling/tokenization, model evaluation, release benchmark performance, or
release qualification.
