# Reference Architecture for Verified Transformer Inference

Date: 2026-07-23
Status: implementation-agnostic research architecture
Authority: implementation-independent architectural decomposition, invariants,
and conformance

## Abstract

An open-weight model is not executable merely because its serialized parameters
can be read into memory. Reproducible inference requires a chain of
identity-preserving transformations from model semantics to physical tensors,
admitted artifacts, executable operations, persistent state, device resources,
and published outputs. This paper specifies a reference architecture for that
chain.

The architecture separates six interacting planes: model, execution, state,
resource, device, and evidence. It distinguishes logical models from physical
variants and artifacts; semantic graphs from executable and launch graphs;
immutable model resources from mutable sessions; persistent state from
workspace; and numerical output from execution evidence. Typed boundaries,
transactional state transitions, fail-closed capability admission, and
identity-scoped evidence are treated as architectural requirements rather than
implementation details.

The principal subject is local or accelerated inference for open-weight,
decoder-oriented sequence models. The decomposition also admits encoders,
encoder-decoder models, recurrent or linear attention, mixture-of-experts
networks, and state-space sequence mixers when their semantics are represented
by explicit contracts. This document defines a conformance model. It neither
prescribes one software implementation nor establishes that a particular
system implements any capability.

YVEX's current projection of this architecture is documented separately under
[Implemented Architecture](../architecture/system.md). That projection cannot
alter the implementation-independent conformance model defined here.

## 1. Research Question and Scope

The central question is:

> Which architectural boundaries are necessary to transform an identified
> logical model into repeatable, stateful, hardware-executed inference without
> allowing semantic, physical, lifecycle, or evidentiary responsibilities to
> collapse into one another?

The reference architecture addresses the path:

```text
logical model
  -> tensor semantics
  -> transformation plan
  -> physical variant
  -> serialized artifact
  -> admitted runtime binding
  -> executable graph
  -> runtime model and session
  -> persistent state and workspace
  -> backend operations
  -> kernel or launch graph
  -> validated output and state commit
```

The architecture covers model interpretation, artifact production and
admission, runtime preparation, CPU or accelerator execution, autoregressive
state, failure atomicity, observability, and capability evidence. It does not
specify a training system, a model-serving scheduler, a distributed topology,
a user interface, or a particular container format. These may consume or
extend the architecture but cannot replace its core contracts.

The formulation begins with the Transformer [1], but does not equate the
architecture with conventional softmax attention. IO-aware attention [2],
paged state management [3], sparse expert routing [4], and selective
state-space models [5] demonstrate that sequence execution, resource placement,
and persistent state vary independently of the external block interface.

## 2. Method and Epistemic Boundary

This document is constructed from three classes of evidence:

1. primary model and systems research;
2. format and device specifications;
3. pinned implementations used as decomposition and comparison references.

Research sources motivate abstractions; specifications define external
contracts; implementations expose operational pressure. None transfers its API,
process model, feature set, numerical correctness, or performance to a
conforming implementation. A local capability claim requires local code,
positive and negative tests, executable consumption, and identity-bound
evidence.

Normative terms are used as follows:

- **must** identifies a requirement for conformance;
- **should** identifies a default whose violation requires an explicit,
  testable rationale;
- **may** identifies an optional mechanism that does not alter the required
  boundary.

## 3. Terminology

Table 1 defines the objects used throughout the paper.

| Term | Definition |
| --- | --- |
| Logical model | Backend-neutral topology, tensor roles, operators, state semantics, and numerical policy that define model behavior. |
| Source snapshot | Identified upstream parameters and sidecars from which a physical model may be derived. |
| Transformation plan | Immutable, artifact-neutral operations that map source contributions to logical output tensors. |
| Physical variant | A realization of a logical model with selected dtypes, qtypes, layouts, alignments, decomposition, and placement constraints. |
| Artifact | A concrete serialization of one physical variant. |
| Admission | Validation that an object satisfies the complete contract required by its next consumer. |
| Runtime binding | Immutable, content-addressed bridge from admitted artifact and compiled facts to runtime-consumable tensor and execution descriptors. |
| Runtime model | Immutable, shareable process object for one admitted binding and its model-lifetime resources. |
| Runtime session | Mutable execution object for one request, sequence, or explicitly shared state domain. |
| Persistent state | Semantically observable state that survives an execution unit, such as KV, recurrent state, position, or routing history. |
| Workspace | Temporary memory whose contents have no semantic meaning after the owning execution unit completes. |
| Semantic graph | Backend-neutral operations and dependencies derived from model meaning. |
| Executable graph | Lowered operations, physical bindings, memory plan, backend assignments, and execution variants. |
| Launch graph | Device-level kernels, transfers, barriers, events, and their dependencies. |
| Capability | An admitted operation over a bounded input, artifact, backend, mode, and resource domain. |
| Evidence | Identity-bound facts proving what executed and what result or refusal occurred. |

The word *model* must be qualified when ambiguity is possible. A source
repository, logical model, physical variant, artifact, and runtime model are
related but non-interchangeable objects.

## 4. Architectural Thesis

Model semantics and hardware execution are separated by a sequence of typed
lowerings. A device does not execute a "Transformer block" as a semantic
primitive. It executes operations whose addresses, shapes, strides, numerical
parameters, dependencies, and resource lifetimes have already been resolved.

Figure 1 shows the vertical decomposition.

```text
Model semantics
    |
    v
Tensor roles and artifact-neutral transformations
    |
    v
Physical tensors, layouts, and complete artifact
    |
    v
Runtime binding and executable graph
    |
    v
Memory, state, and backend admission
    |
    v
Device operations and launch
    |
    v
Validated output, committed state, and evidence
```

**Figure 1 — Semantic-to-physical lowering.**

Each transition has an input contract, an output identity, a failure
vocabulary, and a downstream consumer. No lower layer may reconstruct
authoritative facts owned by a higher layer. Conversely, no higher layer may
claim that an unexecuted physical boundary is available merely because its
semantic description exists.

## 5. Representation and Identity Hierarchy

### 5.1 Logical Model

The logical model defines architecture class, layer topology, residual-stream
geometry, normalization, positional policy, sequence mixer, feed-forward or
expert structure, tokenizer/output relationship, and tensor roles. It is
independent of storage qtype and execution device.

Logical-model construction must be deterministic over its identified source
facts. Family names, filenames, and lexical tensor-name matches are
insufficient identities.

### 5.2 Transformation Plan

The transformation plan connects source tensors to logical tensors through
typed operations such as selection, concatenation, permutation, aggregation,
scaling, or conversion. It defines semantic derivation before choosing a
container or backend.

Plan construction must not require payload execution. Execution of the plan
must consume exact source ranges and must not rediscover roles, axes, or
companion tensors from untyped names.

### 5.3 Physical Variant

A physical variant selects:

- storage dtype or qtype;
- row and block geometry;
- logical-to-physical shape mapping;
- alignment and padding;
- tensor aggregation or partition;
- placement and residency constraints;
- numerical accumulation and conversion policies.

Two physical variants may implement one logical model. Their logical identity
may match while their derivation, artifact, residency, and executable
identities differ.

### 5.4 Artifact

An artifact serializes the physical variant as metadata, a tensor directory,
offsets, encoded payloads, and compatibility facts. A format specification such
as GGUF [6] defines container semantics, not model support.

Artifact admission must validate the complete byte and metadata contract needed
by the next boundary. Parsing a header, mapping a file, or recognizing a qtype
is a lower evidence stage than admitting a complete executable artifact.

### 5.5 Runtime Binding

The runtime binding prevents the warm runtime from reconstructing compilation.
It must bind, directly or by immutable reference:

- logical-model, derivation, physical-variant, and artifact identities;
- typed tensor roles and physical tensor locations;
- executable descriptors and numerical policies;
- capability prerequisites and compatibility constraints;
- binding schema version and content identity.

A runtime binding is not an opportunistic cache. It is an admitted artifact
with an independent identity and lifecycle.

### 5.6 Identity Propagation

The minimum identity chain is:

```text
source snapshot identity
    -> logical model identity
    -> transformation identity
    -> physical variant identity
    -> artifact identity
    -> runtime binding identity
    -> executable identity
    -> launch identity
    -> state transition identity
    -> output identity
    -> evidence identity
```

**Figure 2 — Identity derivation chain.**

An upstream semantic change invalidates all dependent identities. A memory-plan
change may invalidate executable and launch identities without changing the
logical model. A new device capture may change only the launch identity. An
implementation must not hash pointer values, padding, timestamps, or local
paths into a semantic identity.

## 6. Plane Decomposition

The architecture comprises five operational planes and one transverse evidence
plane.

```text
+-----------------------------------------------------------+
| MODEL PLANE                                               |
| architecture, roles, topology, policies, composition      |
+-----------------------------------------------------------+
| EXECUTION PLANE                                           |
| lowering, graphs, runtime objects, dispatch, publication  |
+-----------------------------------------------------------+
| STATE PLANE                                               |
| KV, recurrent state, position, transaction, rollback      |
+-----------------------------------------------------------+
| RESOURCE PLANE                                            |
| artifact, residency, arena, workspace, transfer, budget   |
+-----------------------------------------------------------+
| DEVICE PLANE                                              |
| backend, context, stream, kernel, launch graph, hardware  |
+-----------------------------------------------------------+
| EVIDENCE PLANE                                            |
| identities, failures, numerical facts, timing, conformance|
+-----------------------------------------------------------+
```

**Figure 3 — Architectural planes.**

Table 2 assigns decisions to their owning plane.

| Plane | Owns | Must not own |
| --- | --- | --- |
| Model | Model meaning, tensor roles, block grammar, sequence-mixer and output policy | Allocation, transfer, launch, session mutation |
| Execution | Lowering, dependency order, runtime dispatch, execution modes, publication boundary | Source authenticity, backend feature invention |
| State | Persistent layout, bounds, views, candidate deltas, commit, reset, rollback | Weight residency, temporary workspace |
| Resource | Byte lifetime, mapping, residency, arenas, staging, budgets | Model topology, state semantics |
| Device | Capability admission, device allocation, transfer, launch, synchronization, cleanup | Family interpretation, artifact reconstruction |
| Evidence | Typed observation tied to all relevant identities | Capability decisions or semantic ownership |

The evidence plane is transverse because every operational plane produces
facts. It remains observational: reports, metrics, traces, and renderers cannot
promote a capability.

## 7. Tensor Roles and Typed Bindings

A tensor must be identified by architectural role rather than by an incidental
string. A tensor-role contract includes:

- component and layer or expert coordinates;
- transformation direction;
- logical shape and physical geometry;
- storage qtype and compute dtype;
- layout, stride, and alignment;
- identity and source derivation;
- family-specific policy where irreducible.

The binding sequence is:

```text
typed tensor role
    -> physical tensor descriptor
    -> admitted host view or resident handle
    -> operator operand
```

This contract must prevent substitution across roles, layers, experts, logical
and physical shapes, storage and accumulation types, or persistent and
temporary buffers. Runtime lookup by scattered string conventions is
non-conforming.

## 8. Model Execution Semantics

### 8.1 Decoder Block

A decoder block has a stable external residual width even when its internal
operators expand, compress, partition, or route the representation.

```text
prior residual
    -> pre-mixer normalization
    -> sequence mixer
    -> output transform
    -> residual transition
    -> pre-FFN normalization
    -> dense FFN or MoE
    -> residual transition
    -> next residual
```

The execution contract must identify the prior residual, each candidate
update, the combination rule, and the publication point. In-place storage is
permitted only when it preserves the same transactional semantics.

### 8.2 Sequence-Mixer Contract

The common model cannot assume one attention formulation. A typed
sequence-mixer descriptor must define:

- input and output hidden geometry;
- persistent-state shape and interpretation;
- positional and masking semantics;
- prefill transition;
- decode transition;
- candidate-delta and commit unit;
- numerical and backend requirements.

Conventional softmax attention, recurrent or linear attention, and state-space
mixers may implement this contract with different state and execution
strategies. Classification does not imply that an implementation supports
every member of the class.

### 8.3 Attention

An attention operation comprises current-input transformation, query
construction, admitted history access, scoring or selection, masking,
normalization, value aggregation, output transformation, and a candidate
history update. IO-aware implementations may fuse or tile these stages [2],
but fusion must not erase their semantic contract.

The history representation may be conventional keys and values, compressed
state, local windows, sparse indexes, rolling summaries, or a hybrid. The state
owner, not the attention kernel, defines its persistent interpretation.

### 8.4 Dense Feed-Forward Network

A dense feed-forward sublayer performs a position-local expansion, activation
or gate, contraction, and residual update. It may use fused physical
operators, but it does not own sequence history.

### 8.5 Mixture of Experts

A sparse expert sublayer adds routing and conditional parameter access [4]:

```text
hidden state
    -> router scores
    -> admitted top-k selection
    -> dispatch plan
    -> expert execution
    -> weighted aggregation
    -> candidate residual update
```

Routing semantics, expert topology, placement, dispatch, execution,
aggregation, and cleanup are separate contracts. A common runtime must not
branch on model-family strings to recover these facts.

## 9. Graph Hierarchy

The term *graph* denotes three distinct representations.

### 9.1 Semantic Graph

The semantic graph expresses model operations, typed values, state effects, and
dependencies. It is backend-neutral and does not require device addresses or
launch geometry.

### 9.2 Executable Graph

The executable graph resolves semantic operations into physical operators,
prefill or decode variants, tensor bindings, state views, memory assignments,
backend choices, and execution modes. It is specific to a physical variant and
execution profile.

### 9.3 Launch Graph

The launch graph contains kernels, copies, memory operations, events, barriers,
and device dependencies. CUDA Graph definition, instantiation, and repeated
launch are distinct lifecycle stages [7].

The relationship is:

```text
semantic graph
    --lowering--> executable graph
        --backend materialization--> launch graph
            --instantiation--> launch executable
                --submission--> device execution
```

**Figure 4 — Graph hierarchy.**

Graph identity must remain distinct at every stage. A device graph is not the
source of model semantics, and an eager sequence is not evidence of graph
capture. Explicit execution modes must fail if their prerequisites are absent;
they must not silently degrade to another mode.

## 10. Runtime Object Model

### 10.1 Immutable Runtime Model

The runtime model represents one admitted model opened for repeated execution.
It owns or retains:

- one admitted runtime binding;
- authenticated artifact and tensor directory views;
- family-neutral runtime descriptors;
- typed family adapter;
- shareable resident weights and backend resources;
- semantic and executable identities;
- capability facts for the exact resources it holds.

Its lifecycle is:

```text
open -> authenticate -> bind -> prepare residency -> share -> close
```

The runtime model does not own request position, KV, sampling state, or
request-local workspace.

### 10.2 Mutable Runtime Session

The runtime session owns the changing state of one request or explicitly
defined sequence domain:

- current position and persistent sequence state;
- workspace and input/output staging;
- cancellation and failure state;
- execution counters and graph instances;
- random seed and sampling state where applicable.

Multiple sessions may share one runtime model. They must not share mutable
state unless the shared-state protocol is explicit, synchronized, and part of
the model semantics.

### 10.3 Lifetime Ordering

The minimum destruction order is the reverse of dependency:

```text
in-flight execution
    -> session graph instances
    -> session workspace and persistent state
    -> shared resident resources
    -> backend context
    -> artifact and binding views
```

Destruction must either wait for, cancel, or reject concurrent use according to
a typed lifecycle contract.

### 10.4 Hosted Runtime And Conversation Sessions

A local runtime host may retain one immutable runtime model across many client
connections and requests. It owns listener, queue, worker, session registry,
telemetry and graceful shutdown. Transport threads submit typed work; one
initial worker serializes graph-mutating execution without implying continuous
batching.

A conversation session is longer-lived than a request and distinct from both a
client connection and the underlying execution session. It owns the exact
message transcript, committed token ledger, continuation policy, RNG state and
turn records that correspond to its KV. Client detach preserves it. Reset must
clear every mutable representation through one authority.

Multi-turn reuse is admitted only when the newly rendered and encoded prompt
has the committed token ledger as an exact prefix. The host prefills only the
remaining suffix. String similarity or silent replay cannot establish KV reuse.

An application-compatibility listener should remain a source-separated adapter
rather than another model host. It translates a bounded external schema into
provider-neutral typed requests consumed by the same session and execution
authority. It owns neither model resources nor persistent sequence state, and
it may execute application tools only under a separately admitted action-loop
contract.

Operator syntax is a projection of semantic operations, not an execution
authority. A command registry may own spelling, arguments, visibility, and
requirements, but it cannot own domain defaults, arithmetic, resources,
serialization, or lifecycle. Client and finite offline projections remain
mechanically distinct even when one executable presents both.

## 11. Transactional State

Persistent state is any information required to continue inference without
recomputing all prior input. It may contain KV, compressed or recurrent
history, local windows, page tables, indexes, positions, prefix state, or
family-specific summaries.

Let `S_t` be the admitted state before an execution unit. An operator evaluates:

```text
(Y_t, Delta_t, E_t) = Execute(X_t, View(S_t), R)
```

where `X_t` is current input, `R` is immutable runtime state, `Y_t` is candidate
output, `Delta_t` is a candidate state change, and `E_t` is execution evidence.
Only after output, numerical status, bounds, cancellation, and device
completion are validated may the state owner apply:

```text
S_(t+1) = Commit(S_t, Delta_t)
```

On any failure:

```text
published output = none
visible persistent state = S_t
```

The commit unit—token, chunk, layer, complete decode step, or speculative
branch—must be explicit. It determines rollback, cancellation, concurrency,
and checkpoint semantics. Workspace mutation is not a state commit.

## 12. Memory and Resource Architecture

Memory is partitioned by ownership and lifetime:

| Class | Typical contents | Lifetime |
| --- | --- | --- |
| Model memory | Weights, immutable metadata, resident operator tables | Runtime model |
| Session memory | KV, recurrent state, position, sampling state | Runtime session |
| Workspace | Activations, reductions, temporary routing buffers | Execution unit or reusable session arena |
| Staging | Host/device input and output transfer buffers | Transfer or bounded session lifetime |
| Graph resources | Captured nodes, executable instances, stable pointer tables | Graph registry or session |
| Published output | Validated logits, hidden states, tokens, evidence copies | Caller-defined |

### 12.1 Residency

Residency states where bytes remain between executions: storage, pageable host
memory, pinned host memory, coherent memory, device memory, or a bounded cache.
It must identify content, location, generation, budget, and invalidation
conditions.

Weight residency is established only when a warm execution neither reopens the
artifact nor repeats the weight transfer. Mapping is not residency, and
residency is not execution.

### 12.2 Stable Address

Captured graphs, persistent kernels, pointer tables, and launch caches may
depend on stable addresses [7]. State and workspace arenas should therefore be
planned for the lifetime required by the execution mode. Reallocation must
invalidate dependent launch objects before their next use.

### 12.3 Budget

Resource admission must account separately for:

- resident weights;
- persistent-state capacity and growth;
- peak workspace;
- transfer staging;
- graph and module overhead;
- expert cache or streamed parameters;
- output publication;
- safety margin.

Budget refusal is an architectural result, not an unclassified kernel failure.
Checked arithmetic must precede allocation and address computation.

## 13. Backend and Kernel Architecture

### 13.1 Backend Boundary

A backend adapts executable operations to a device. It owns device discovery,
context and module lifecycle, memory, transfer, streams, launch,
synchronization, failure projection, cleanup, and profiling hooks.

Backend capability must be admitted over the tuple:

```text
(operation, physical variant, shape, numerical contract,
 execution mode, resource plan, device features)
```

A visible device or loaded module is not sufficient. Required kernel symbols,
qtypes, layouts, geometry, memory, and execution mode must be admitted
atomically.

### 13.2 Kernel Boundary

A kernel receives device addresses, dimensions, strides, qtypes, numerical
parameters, status storage, and launch geometry. It must not receive authority
to infer artifact identity, session lifecycle, model family, or tensor role.

Each numerical kernel defines:

- thread, warp, block, or tile mapping;
- access and alignment requirements;
- accumulation precision and rounding;
- reduction order and determinism;
- non-finite behavior;
- status publication and synchronization;
- supported shape and qtype domain.

Direct encoded computation may unpack, scale, multiply, and accumulate
quantized weights without materializing a complete floating-point copy. Its
admission depends on row geometry, decoder exactness, accumulation policy, and
independent numerical comparison.

### 13.3 Prefill and Decode Variants

Prefill commonly exposes matrix-matrix parallelism over several tokens; decode
often exposes matrix-vector or narrow-matrix work and greater bandwidth
pressure. The two phases may share semantics while requiring different kernels,
workspace, graph geometry, and state access.

Kernel fusion and launch graphs are orthogonal. Fusion combines semantic
operators into one kernel to reduce intermediate traffic or launch count.
Graph execution reduces host orchestration across one or more kernels.

## 14. Execution Lifecycle

### 14.1 Cold Path

The cold path may perform:

```text
artifact open and authentication
    -> runtime-binding admission
    -> descriptor and tensor resolution
    -> backend and module admission
    -> residency preparation
    -> workspace and state-arena construction
    -> graph capture or construction
    -> capability qualification
```

Each completed stage publishes an object only after its full invariant holds.
Partial construction must be cleaned up without making the next stage visible.

### 14.2 Warm Path

The warm path should contain only:

```text
input and state-view update
    -> phase-aware dispatch
    -> kernel launch or graph replay
    -> completion and numerical validation
    -> output publication and state commit
```

Source verification, complete artifact hashing, tensor upload, compilation, and
memory planning must not recur unless their identity or resource generation was
invalidated.

### 14.3 Execution Modes

| Mode | Contract |
| --- | --- |
| Eager | Operations are submitted individually; dependencies and failures remain explicit. |
| Piecewise graph | Stable regions are instantiated independently and composed by the runtime. |
| Full graph | One larger stable execution unit is instantiated under stricter shape, address, and lifetime requirements. |
| Auto | A policy selects one admitted mode and records that choice in evidence. |

An explicit mode request must either execute that mode or return a typed
refusal. Auto selection must not obscure which mode ran.

## 15. End-to-End Inference Phases

### 15.1 Prefill

Prefill transforms a known token sequence into persistent state and a final
hidden boundary:

```text
token identifiers
    -> embeddings
    -> one or more activation chunks
    -> complete block stack
    -> state transaction
    -> final hidden state
```

Chunked and monolithic prefill must be semantically equivalent within an
explicit numerical tolerance. State publication occurs only at the selected
commit unit.

### 15.2 Decode

Decode consumes the current token or activation and admitted persistent state:

```text
current token
    -> embedding
    -> complete block stack
    -> final hidden state
    -> candidate persistent-state delta
```

Attention decode is one component of model decode. Position and persistent
state advance only after the complete step succeeds.

### 15.3 Logits and Sampling

Final normalization and vocabulary projection produce one logit per vocabulary
entry. Sampling applies an explicit policy and session-owned random state:

```text
final hidden
    -> final normalization
    -> vocabulary projection
    -> logits
    -> filters and sampling policy
    -> next token
```

Deterministic sampling requires an admitted seed, policy, numerical behavior,
and ordering contract.

### 15.4 Generation

Autoregressive generation composes:

```text
decode -> logits -> sample -> append -> stop decision
```

Tokenizer and detokenizer boundaries connect tokens to external text.
Generation exists only when all phases execute through the same admitted model,
artifact, runtime, state, and backend path. A component kernel, fixture loop, or
diagnostic output cannot substitute for this composition.

Hosted generation adds request queueing, exact reusable-prefix admission,
committed-fragment streaming, partial-turn state and disconnect/cancellation
semantics around the same generation owner. It does not create another model
loop.

## 16. Failure and Publication Model

Every boundary must fail closed. Failure classes should distinguish:

- invalid input or shape;
- incompatible or incomplete artifact;
- unsupported capability or execution mode;
- resource budget or checked-arithmetic failure;
- allocation, mapping, or transfer failure;
- launch or synchronization failure;
- numerical-status failure;
- cancellation or invalidation;
- cleanup failure;
- identity or snapshot drift.

The first causal failure remains primary. Cleanup failures may be recorded as
secondary evidence but must not overwrite the cause. No failed operation may
publish partial output, advance persistent state, or leave a reusable object
marked ready when its invariant is no longer true.

## 17. Concurrency

The runtime model may be shared read-only across sessions. Session state must
remain isolated unless a typed sharing mechanism defines synchronization,
reference ownership, copy-on-write or lease behavior, invalidation, and
destruction ordering.

A concurrency contract must specify:

- runtime-model thread safety;
- stream or queue ownership;
- session arena and graph-instance ownership;
- persistent-state mutation serialization;
- cancellation visibility;
- in-flight destruction behavior;
- sharing rules for resident weights, prefixes, or expert caches.
- transport-to-worker handoff and bounded queue behavior;
- session attach/detach and active-turn serialization;
- subscriber backpressure and shutdown drain.

Sharing immutable weights is generally safe and desirable. Implicitly sharing
workspace, KV, random state, or candidate deltas is non-conforming.

## 18. Evidence and Evaluation

Evidence must identify what ran, not merely what output was observed. At
minimum, execution evidence binds:

- logical model, physical variant, artifact, and runtime binding;
- executable and launch identities;
- prior state and candidate delta;
- backend, device, driver, kernel bundle, and mode;
- input and output identities;
- typed failure or completion status;
- timing interval and synchronization boundary when measured.

Equal CPU and GPU output bytes do not imply equal execution evidence.
Numerical conformance, component performance, full-model performance, model
quality, and serving performance are separate evaluations.

Benchmarks must distinguish cold preparation, residency, capture, warm
submission, device completion, state commit, output publication, and
end-to-end latency. A component benchmark cannot promote model or serving
capability.

## 19. Capability Lattice

Capabilities form a dependency lattice rather than a single Boolean:

```text
artifact parse
    -> artifact admission
    -> runtime open
    -> weight residency
    -> primitive execution
    -> sequence-mixer execution
    -> persistent state
    -> full prefill
    -> FFN or MoE
    -> complete block stack
    -> model decode
    -> logits
    -> sampling
    -> tokenizer-backed generation
    -> model evaluation
    -> full-model benchmark
    -> serving
```

Some branches are independent: a backend may admit one primitive without
admitting another, and an eager mode may exist without a graph mode. Promotion
requires the complete prerequisite set, production interface, positive test,
refusal test, cleanup proof, and real downstream consumer for the exact
capability scope.

## 20. Multi-Family Extensibility

The common runtime owns lifecycle, binding admission, sessions, transactional
state, resource management, backend dispatch, graph execution, and evidence.
A family adapter owns only irreducible facts:

- architecture and block composition;
- tensor-role lowering;
- sequence-mixer and persistent-state semantics;
- positional, FFN, MoE, tokenizer, and output policies;
- family-specific numerical and capability requirements.

A new family should extend a common contract only when a real consumer exposes
a missing invariant. Speculative abstractions, duplicated qtype or failure
registries, family-specific common runtimes, and branches on target strings
indicate boundary failure.

## 21. Architectural Invariants

Table 3 defines the conformance invariants.

| Identifier | Invariant |
| --- | --- |
| INV-SEM | Model-family semantics are resolved before runtime and backend execution. |
| INV-ID | Logical, physical, artifact, binding, executable, launch, state, output, and evidence identities remain distinct. |
| INV-LIFE | Model resources, session state, workspace, graph resources, and outputs have explicit non-overlapping lifetimes. |
| INV-STATE | Operators produce candidate deltas; only the state owner commits them after complete validation. |
| INV-FAIL | Failure publishes neither partial output nor persistent-state mutation. |
| INV-CAP | Capability is admitted for an exact operation, variant, shape, mode, resource plan, and device domain. |
| INV-WARM | Warm execution does not reconstruct compilation, reread weights, or repeat residency without invalidation. |
| INV-EVID | Evidence is identity-bound and cannot itself become the authority that grants capability. |
| INV-FAMILY | Common mechanisms do not recover family policy from names or duplicate family implementations. |
| INV-EXT | A lower capability never proves a higher composed capability. |

## 22. Conformance Criteria

An implementation conforms to this reference architecture only if it can
demonstrate all of the following for each claimed capability:

1. a typed input, output, failure, and lifetime contract;
2. the owning plane and authoritative identity;
3. deterministic or explicitly nondeterministic transformation semantics;
4. checked resource admission and complete cleanup;
5. transactional state and output publication where mutation occurs;
6. fail-closed backend and execution-mode admission;
7. a positive executable test and a corresponding refusal test;
8. an independent numerical reference for numerical execution;
9. identity-bound evidence from a real downstream consumer;
10. explicit non-claims for every unclosed higher boundary.

Conformance is capability-scoped. Partial conformance is meaningful only when
the exact boundary and absent higher capabilities are reported.

## 23. Limitations and Open Research Questions

This architecture does not resolve several policy choices:

- distributed tensor, pipeline, expert, or context parallelism;
- continuous batching and serving admission control;
- speculative decoding and branch-merge state transactions;
- cross-session prefix sharing and confidentiality;
- heterogeneous CPU/GPU/storage placement;
- dynamic expert streaming and eviction;
- online adaptation or mutable weights;
- multimodal preprocessing and encoder state;
- fault recovery across process or device loss.

These extensions should preserve the identity, lifetime, state, and capability
separations defined here. They require new typed contracts rather than
exceptions to existing ownership.

## 24. Conclusion

A verified inference engine is a composition of semantic, physical,
stateful, and evidentiary systems. The model defines what must be computed.
Tensor roles and transformations connect that meaning to data. Physical
variants and artifacts make representation explicit. Runtime bindings prevent
warm-path reconstruction. Runtime models and sessions separate immutable
resources from mutable requests. The state plane preserves sequence
continuity transactionally. The resource plane controls byte location and
lifetime. Backends admit exact device operations, and kernels perform bounded
numerical work.

Inference becomes an architectural capability only when these layers form one
identity-preserving, fail-closed, executable chain. Maturity is therefore
measured not by the presence of weights, kernels, graphs, or output in
isolation, but by the precision with which each boundary can state what it
owns, what it consumed, what it changed, what evidence it produced, and which
higher capability remains unproved.

---

## References

[1] A. Vaswani et al., ["Attention Is All You
Need,"](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)
*Advances in Neural Information Processing Systems 30*, 2017.

[2] T. Dao et al., ["FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness,"](https://arxiv.org/abs/2205.14135) arXiv:2205.14135, 2022.

[3] W. Kwon et al., ["Efficient Memory Management for Large Language Model
Serving with PagedAttention,"](https://arxiv.org/abs/2309.06180)
arXiv:2309.06180, 2023.

[4] W. Fedus, B. Zoph, and N. Shazeer, ["Switch Transformers: Scaling to
Trillion Parameter Models with Simple and Efficient
Sparsity,"](https://arxiv.org/abs/2101.03961) arXiv:2101.03961, 2021.

[5] A. Gu and T. Dao, ["Mamba: Linear-Time Sequence Modeling with Selective
State Spaces,"](https://arxiv.org/abs/2312.00752) arXiv:2312.00752, 2023.

[6] ggml contributors, ["GGUF
Specification,"](https://github.com/ggml-org/ggml/blob/af97976c7810cdabb1863172f31c432dab767de7/docs/gguf.md)
ggml commit `af97976c7810cdabb1863172f31c432dab767de7`.

[7] NVIDIA, ["CUDA Programming
Guide,"](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) and
["CUDA Driver API: Graph
Management,"](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html).

[8] vLLM Project, ["Architecture
Overview,"](https://docs.vllm.ai/en/latest/design/arch_overview/) and pinned
DeepSeek-V4 integration at commit
`8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`.

[9] SGLang Project, DeepSeek-V4 integration at commit
[`96a04cb13f9c3ed86028e090784a9eb059cf5318`](https://github.com/sgl-project/sglang/blob/96a04cb13f9c3ed86028e090784a9eb059cf5318/python/sglang/srt/models/deepseek_v4.py).

[10] llama.cpp Project, DeepSeek-V4 mapping and loader at commit
[`e920c523e3b8a0163fe498af5bf90df35ff51d25`](https://github.com/ggml-org/llama.cpp/tree/e920c523e3b8a0163fe498af5bf90df35ff51d25).

[11] MLC LLM Project, library and weight packaging contract at commit
[`a2bcc5c86678b72a86b7aadc29b643a5ce63c747`](https://github.com/mlc-ai/mlc-llm/tree/a2bcc5c86678b72a86b7aadc29b643a5ce63c747).

[12] IREE Project, [deployment-aware lowering
architecture](https://iree.dev/) at commit
[`0893eac771d532b7110f1f7581d3f4cd0b9172bf`](https://github.com/iree-org/iree/tree/0893eac771d532b7110f1f7581d3f4cd0b9172bf).

[13] NVIDIA, [TensorRT-LLM Architecture
Overview](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html) at
commit
[`02cedf6e4e421ac48d271452bf3836cb57caf297`](https://github.com/NVIDIA/TensorRT-LLM/tree/02cedf6e4e421ac48d271452bf3836cb57caf297).

[14] PyTorch, [ExecuTorch Architecture and
Components](https://docs.pytorch.org/executorch/stable/getting-started-architecture)
at commit
[`ae754e9ed8b650e78b921906b2ba8af65ea408ab`](https://github.com/pytorch/executorch/tree/ae754e9ed8b650e78b921906b2ba8af65ea408ab).

[15] S. Sanfilippo, [DwarfStar
engine](https://github.com/antirez/ds4/tree/80ebbc396aee40eedc1d829222f3362d10fa4c6c)
at commit `80ebbc396aee40eedc1d829222f3362d10fa4c6c`.

[16] DeepSeek-AI, ["DeepSeek-V4: Towards Highly Efficient Million-Token
Context Intelligence,"](https://arxiv.org/abs/2606.19348v1)
arXiv:2606.19348v1, 2026.

[17] NVIDIA, [CUTLASS](https://github.com/NVIDIA/cutlass/tree/1e7394829291360bdcf07036cbe5411631d2d33b)
at commit `1e7394829291360bdcf07036cbe5411631d2d33b`.
