# Hosted Runtime Contract

Status: normative implemented contract

Authority: host/engine lifecycle, execution sessions, scheduling, transactional
publication, resource admission, failure, and cleanup. C interfaces are
documented separately in [YVEX C API](c-api.md).

## Parties and scope

The foreground `yvex serve`, common runtime, graph/backend, tokenizer,
generation, and server owners produce this contract. Native clients, the
interactive console, the loopback OpenAI adapter, and focused tests consume it.

The contract begins with a healthy host and one locally known model profile. It
may open an authenticated package as an engine generation and ends each
operation with typed state/results, committed output, or refusal. It does not
define source acquisition, compilation, command syntax, HTTP field syntax,
evaluation, benchmark policy, or release state.

## Host lifecycle

The server host lifecycle is independent from every model:

```text
configured -> starting -> ready -> stopping -> stopped
                         \-> failed
```

A ready host owns its private Unix listener, optional loopback OpenAI listener,
telemetry, engine manager, and bounded external request capacity. It is valid
and useful with zero loaded engines. Host stop refuses new connections, drains
bounded client work, closes every engine, releases listeners, and then
terminates the process.

Engine count has three separate bounds. The implementation safety maximum is a
memory/indexing guard, `--max-engines` selects the configured host-slot limit,
and live package/resource admission decides how many engines can actually fit.
Host status reports the configured maximum and current population; neither fact
changes model or package meaning.

A client connection is not a session. A model engine is not the server.
Disconnecting a client, closing a session, or unloading one engine does not
stop the host.

## Engine lifecycle

Each engine slot follows:

```text
unloaded -> loading -> loaded -> draining -> unloading -> unloaded
                \-> failed               \-> failed
```

A load request names one complete local registry profile. The host reserves the
alias, assigns a new nonzero process-local generation, and opens the profile
outside the engine-manager lock. A text engine authenticates one artifact and
runtime binding. A composite engine authenticates every required component and
publishes one logical engine identity. No engine is routable while loading or
failed.

Engine open validates package identity, runtime binding, model descriptor,
Physical Execution IR, specialization compatibility, backend capability, and
resource budgets before publishing `loaded`. The runtime imports compiler
facts; it never reconstructs transformation or writer plans or resolves a
concrete family implementation from a target string.

A first artifact admission hashes every byte. A later open may use a rebuildable
verified-reopen lease only when expected identity and the complete filesystem
snapshot agree. Invalid, stale, or missing lease data falls back to complete
authentication. The lease is neither artifact authority nor portable evidence.
Composite components apply the same rule independently.

Unload requires the exact loaded alias and, when supplied, generation. It marks
the engine draining before waiting: new work and new sessions refuse, active
work receives cancellation, and the manager waits for its lease count to reach
zero. Cleanup then releases sessions, scheduler, backend resources, residency,
package handles, and component resources. The host and unrelated engines remain
ready. Reloading the alias produces a distinct generation.

An opening or cleanup failure never publishes a ready engine. A failed slot
remains observable with its generation and state but owns no executable
resources.

## Package and specialization

The engine authenticates package facts from runtime binding v15 and PEIR v5.
Package facts include model/operator identities, terminal tensor roles and
encoded ranges, canonical qtype/layout, compiled plans, tokenizer/conversation
policy, and numerical obligations.

The engine then seals a backend/device specialization. It binds package
decisions to admitted implementation classes, activation representations, real
width envelopes, equivalent fallback class, and hardware crossovers. The
specialization identity changes when those deployment-significant facts change;
the artifact and package identities do not.

A backend may select another tile, warp, grid, stream, or graph implementation
only when it is equivalent under the selected specialization. An exact
accumulation/reduction contract is not an equivalent-choice detail. No explicit
CUDA request falls back to CPU, and no explicit exact request silently changes
implementation class.

Current v14 runtime bindings remain readable only through their explicit
authenticated compatibility importer. Legacy records that describe canonical
package storage are normalized to PEIR v5; a v14 derived-layout requirement that
cannot be represented truthfully refuses. V14 bytes are never decoded as v15.

## Resource admission and accounting

Before large allocation, engine open compares required bytes with configured
host/device budgets, the process cgroup hierarchy, live system availability,
and typed backend facts. The retained system reserve is the greater of 8 GiB
and one eighth of effective capacity. A second live check after artifact
authentication prevents a long hash from turning an earlier observation into
an unsafe allocation.

Residency schema v7 owns one explicit storage backing and one backend execution
resource. Artifact-mapped placement borrows the immutable authenticated mapping;
CUDA execution requires an admitted registration and device address. Copied
host, locked-host, or CUDA-managed placement owns the copied prepared bytes.
The engine reports mapped package, prepared, resident host, and resident device
bytes separately. Unified physical memory does not collapse these classes.

Sequence state, reusable workspace, and transient allocations are session or
execution resources, not model-package meaning. Live free bytes may admit or
refuse a resource action but never enter semantic/package identity.

Each engine generation owns a bounded resource catalog. Canonical mappings,
component resources, prepared tensors/groups/layouts, backend handles,
executable caches, sequence state, workspace, and temporaries are distinct
entry kinds. Every entry declares package provenance, specialization and
admission provenance where applicable, numerical class, byte classes,
dependency, lifetime, readiness, borrows, release behavior, and eviction
eligibility. Stale generations refuse. Referenced entries and entries with live
dependents cannot be released.

Evicting a prepared entry releases only its owned bytes and invalidates its
process-local resource handle. The admitted package mapping and package
identity remain unchanged, so an equivalent entry may be prepared again from
the same package and specialization facts. The contract admits this lifecycle;
it does not claim that a retained optimized DeepSeek selective layout or an
automatic eviction policy currently exists.

## Session lifecycle and routing

Every server session binds to one alias and exact engine generation. Alias
equality never migrates a session to a reloaded engine.

```text
absent -> created -> ready/detached -> running -> retained/partial
                                        \-> reset -> ready
                                        \-> closing -> closed
```

A session owns one mutable runtime execution session plus token ledger,
conversation transcript, incremental decoder, RNG/sampling state, turn state,
and persistent sequence state. Sessions sharing an engine share immutable
resources but no mutable state or workspace. Detach and disconnect retain
session state. Reset clears committed semantic state while retaining compatible
allocation. Close releases it.

An engine-scoped operation may omit a model only when exactly one loaded engine
is unambiguous. Multiple loaded engines require explicit model selection.
Unknown, unloaded, loading, draining, failed, or stale generations refuse
before session mutation. OpenAI `model` and native `--model` resolve through
the same engine manager.

## Prompt, execution, and publication

Conversation rendering and tokenization produce the complete prompt. Prefix
reuse is admitted only when the session's committed token ledger is an exact
prefix. The runtime prefills only the suffix. Incompatible prefix, context
overflow, malformed UTF-8, unsupported prompt policy, or invalid session state
refuses before mutation.

Prefill, decode, draft, verification, and correction are phases over common
engine/operator primitives. A normalized hidden row is not logits, sampling is
not decode, and a selected token is fed through stateful decode exactly once.

Generation uses a three-part lifecycle:

```text
begin prompt/prefill transaction
-> advance one target step or one speculative cycle per work budget unit
-> finish, validate, publish terminal result, release turn state
```

Fragments publish only after model state, token ledger, incremental decoder,
and internal text ledger agree. A sink failure after committed output returns a
typed partial-turn snapshot. It never relabels a partial result as complete or
rewinds already committed state.

## Transactional sequence state

State representations remain domain-specific, but their lifecycle is generic.
The transaction coordinator owns a bounded participant collection. Each
participant may stage, prepare commit, publish commit, abort, reset, and close
its own representation.

```text
validate -> begin candidate -> execute/stage -> validate/cancel check
         -> prepare every participant -> publish every participant
         -> commit visible output
```

Failure before publication aborts every participant and preserves prior
committed state. A prepare failure cannot leave one participant visible.
Attention/KV providers, backend state residency, target/draft state, token
ledger, decoder, RNG, and publication metadata remain separate participants.

DSpark candidates are private. The complete target selects one ordered
checkpoint; the transaction promotes exactly the accepted target-authored
prefix and discards the rejected suffix. Accepted target rows are not replayed.
Greedy DSpark must commit the same target token sequence as target-only from the
same initial state.

An idle committed provider may expose a bounded immutable in-memory prefix.
Attach checks layout, capacity, identity, and empty destination state before
copy-on-write sharing. This is not a durable cross-process prefix cache.

## Scheduling and executable batches

The host routes external operations and owns per-session serialization. Each
engine owns one compatible-work scheduler. Distinct sessions may run
concurrently within admitted worker and sequence capacity; the same session key
never mutates concurrently.

Active generation contexts submit real ready rows to the engine scheduler. The
scheduler compares typed compatibility keys, performs a bounded rendezvous, and
forms one physical batch only from matching engine generation, phase, operation,
backend, scope, execution class, geometry, and implementation envelope.
Cancellation removes or refuses stale work before publication.

The execution batch retains source and row provenance. The expert worklist
retains deterministic expert-major routed populations. CUDA receives those
objects and may execute bounded tails, but it cannot regroup semantics or
manufacture width.

The engine scheduler retains multiple independent runnable turns and advances
them cooperatively at transaction-safe execution quanta. This logical runnable
capacity is distinct from both the number of resident sessions and the physical
sequence width of one backend operation. Compatible active operations may
rendezvous into one physical batch, but ready sequences do not dynamically join
and leave decode batches. This contract therefore does not claim continuous
batching; `continuous_batching_ready` remains false.

## Execution accounting

Execution measurement schema v1 binds every available duration or rate to an
explicit scope, host/device clock, composition, work unit, and denominator.
Top-level, nested, enclosing, and overlapping measurements are not silently
added together. Unavailable attribution stays unavailable.

For text generation, cumulative decode rate is all committed decode work divided
by its complete measured decode wall. Rolling decode rate uses only the most
recent committed intervals, currently bounded to 32 tokens, and carries its own
work and duration. Prefill, first decode, later decode, model forward, attention,
output, logits publication, sampling, state commit, synchronization,
detokenization, and client publication are distinct when their owners can time
them. Media uses the same scopes with typed evaluation, frame, sample, byte, or
operation units rather than pretending every workload is token execution.

## Cancellation and draining

Cancellation is correlated to one engine generation, session, request, and
turn. It remains observable at the bounded safe points provided by tokenizer,
prefill, Transformer/MoE, verification, logits, sampling, media execution, and
publication owners.

Cancellation before commit discards the candidate. Cancellation after an atomic
accepted-prefix commit reports that prefix. Engine draining stops new leases,
requests cancellation of active work, and waits for bounded completion before
resource release. It does not invalidate another engine or stop the host.

## Failure and recovery

Every internal runtime failure has an origin and a recovery action. Origins are
external request, integrity, capability, resource, backend, sequence, engine,
and internal invariant. Recovery actions are request refusal, transaction
abort, retry of an already-admitted equivalent, prepare/evict and retry,
sequence invalidation, engine drain, engine-open refusal, and internal invariant
failure.

The two axes are not collapsed into diagnostic prose. Artifact/integrity failure
always fails closed. `auto` may select another admitted numerically equivalent
strategy when state is still private and policy permits it. An explicit exact
request refuses when unavailable. Cleanup failure is reported without
pretending the owner was released.

## Resource truth

Resource summary schema v1 separates artifact/mapped/prepared model spans,
explicit host/device allocations, device-addressable bytes, typed session state,
activation arenas, reusable workspace, transients, process RSS, current/peak,
and logical movement. Artifact and mapped spans may overlap; typed state classes
are subsets of physical session state; peak classes may overlap in time. They
are not an additive total.

Placement and availability qualify every number. On unified-memory hardware, a
mapped artifact may be CUDA-addressable while physical page residency remains
unmeasured. A zero explicit CUDA allocation therefore never means a zero GPU
working set. The server aggregates only facts owned by each engine/component and
does not duplicate immutable model bytes per session.

## Evidence

Production carries only the state, counters, compact lineage, and event seams
needed for correct operation. Audit and forensic profiles may request selected
digests or full intermediates. Benchmark and roofline owners build rich records
from these seams; trace verbosity never changes the numerical path.

Cold package identities and transaction identities remain complete. Transient
engine-generation handles can be joined back to package, binding,
specialization, session, operation, and source-row lineage by retained evidence.

## Inputs, outputs, and side effects

Inputs include exact model/session/request identities, provider messages or
text, generation and sampling policy, token/context bounds, cancellation
correlation, and output sink.

Outputs include typed host/engine/session summaries, committed channel
fragments, terminal turn or media results, usage, prompt/reuse/prefill and
generation facts, stop/cancellation class, final position, state/turn identity,
or typed refusal. Unavailable facts remain unavailable; real zero remains zero.

Admitted side effects are engine load/drain/unload, session
create/reset/fork/close, state and token-ledger commit, bounded resource use,
event publication, local socket/loopback output, and explicit traces. Runtime
does not modify source snapshots or artifacts.

## Compatibility and non-claims

Hosted behavior crosses private local protocol v19 and the bounded OpenAI
compatibility profile v2. Pre-v0.1 private protocol versions may refuse rather
than decode compatibly. Public and internal C ABI follow their typed header and
schema contracts.

This contract does not establish public/remote serving, authentication, TLS,
global ready-sequence continuous batching, independently evictable selective
prepared layouts, distributed execution, restart-persistent engine instances,
complete device-side stochastic sampling/tokenization, load-aware confidence
scheduling, DSpark acceleration, model evaluation, release benchmark
performance, or release qualification.
