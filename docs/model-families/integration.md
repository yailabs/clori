# Model-Family Integration Contract

Status: normative family-integration architecture

This document owns the common contract by which a model family enters YVEX.
Family-specific facts live in separate records. Current release scope and gate
state remain in [`ROADMAP.md`](../../ROADMAP.md).

## Family and target

A **model family** is a repeated architecture pattern: topology, tensor naming
and roles, tokenizer behavior, sequence-mixer and state semantics, FFN or MoE
composition, and output policy.

A **model target** is one concrete source instance in that family. It binds a
repository or local source, revision, configuration, tokenizer, parameter
class, license posture, and intended artifact class. Backend or machine names
do not belong in target identity.

Family classification is not family support. A target becomes executable only
after exact roles are admitted in an artifact, materialized, lowered, executed,
and consumed by the complete text path.

## Promotion path

```text
source snapshot
  -> source tensor inventory
  -> family architecture signature
  -> validated tensor collections
  -> canonical role map
  -> transformation and physical policy
  -> complete artifact contract
  -> Physical Execution IR, materialization and runtime binding
  -> runtime descriptor and family adapter
  -> backend and graph admission
  -> prefill and persistent state
  -> decode, logits, sampling, tokenizer
  -> generation
  -> evaluation
  -> benchmark
  -> release qualification
```

No stage inherits a later claim from a name, report, fixture, external engine,
or structurally valid container.

## Architecture signature

Before role mapping, a target must provide an identity-bound architecture
signature containing as applicable:

- family and target identity;
- source, configuration, tokenizer, and license facts;
- decoder and dense/sparse/hybrid class;
- layer, hidden, intermediate, and vocabulary geometry;
- sequence-mixer and position policy;
- head counts, dimensions, context policy, and persistent-state semantics;
- normalization and residual policy;
- dense FFN or MoE topology;
- expert count, active experts, shared experts, and routing policy;
- output-head and tokenizer relationship;
- auxiliary proposal topology, target feature dependencies, verification, and
  accepted-prefix rules when speculative execution is present;
- source dtype/qtype constraints;
- backend requirements and explicit blockers.

The signature is derived from exact source facts. A family name or lexical
tensor pattern is insufficient.

## Source inventory and canonical roles

The source tensor inventory records native names, shapes, dtypes, shards,
layer/expert coordinates, tied-weight facts, and byte ranges. Candidate tensor
collections are promoted only after family rules validate their membership and
geometry.

The common role space includes embedding, attention norms and projections,
position metadata, feed-forward norms and projections, final norm, output head,
and tokenizer facts. Sparse families add router, correction, routed-expert,
shared-expert, indexing, weighting, and accumulation roles.

A canonical role map preserves source derivation, scope, shape, dtype, layer,
expert, shard, axis, tied-weight, and companion-tensor semantics. Runtime code
does not recover these facts from strings.

## Family adapter

One typed family adapter projects irreducible family facts into common
compilation and execution contracts. A family may have up to three admitted
source projections when each owns a genuinely different dependency boundary:

- `src/model/families/<family>.c` interprets source, coverage, canonical roles,
  logical lowering and family metadata;
- `src/graph/families/<family>.c` composes only the irreducible semantic and
  executable graph recipe;
- `src/backend/<backend>/families/<family>.*` implements only a genuinely fused
  family lowering that generic backend operations cannot express.

The repeated family basename is intentional namespace continuity, not
duplicate ownership. The directory and `config/source_owners.tsv` identify the
level. A family-specific runtime source or directory is forbidden: the common
runtime consumes typed adapters and execution plans.

Across those projections the family may own:

- architecture class and layer schedule;
- native-to-canonical role lowering;
- sequence-mixer, position, and state policy;
- dense or MoE composition and routing semantics;
- output-head and tokenizer constraints;
- accepted physical combinations and required backend operations;
- family-specific graph recipes and numerical policy selection.

It does not own artifact I/O, generic allocation, session lifecycle,
persistent-state storage, workspace, backend memory, common numerical
primitives, telemetry, or claim promotion. YVEX has one common runtime, not a
runtime per family.

For a speculative family, the adapter additionally projects irreducible draft
geometry, feature taps, proposal composition, and family-specific acceptance
policy. The common runtime owns proposal, full-target verification,
accepted-prefix transaction, cancellation, committed accounting, and
publication. It never recovers draft semantics from family names or tensor
strings.

## Artifact prerequisites

A family artifact contract identifies every required global, per-layer, and
per-expert role; metadata and tokenizer fields; qtype/layout constraints;
shape and count expectations; output-head policy; integrity rules; and
derivation identities.

A tensor proof artifact proves only a bounded property. A complete artifact
contains every required role. A supported artifact additionally passes the
complete runtime, generation, evaluation, benchmark, and release gates.

## Runtime prerequisites

The runtime descriptor and binding must project the exact admitted artifact
and Physical Execution IR into family-neutral tensor locations, execution
descriptors, state geometry, workspace requirements, and capability
prerequisites. The compiled execution profile binds those facts to hardware,
context, workload, evidence depth and execution class. The family adapter may
select an irreducible schedule but may not create another immutable model,
session hierarchy, KV owner, worker, or telemetry authority.

When one target has a draft assistant, a single immutable runtime model owns
both plans and their shared resources. A binding that advertises speculative
execution must admit every draft and verification requirement; a target-only
binding cannot be promoted by a runtime flag.

Runtime support requires the same path used by the product:

```text
tokenizer -> prefill -> persistent state -> decode -> logits
          -> sampling -> stop -> detokenization -> text
```

Direct component execution, attention-local prefill/decode phases, or
teacher-forced token execution do not by themselves establish generation.

Speculative generation adds a bounded transaction:

```text
committed target state
  -> proposal workspace
  -> complete-target candidate verification
  -> exact accepted prefix
  -> atomic target/token/decoder/RNG commit
  -> committed text publication
```

Proposal tokens are neither generated output nor persistent target state.
Target verification returns a prefix-addressable candidate transaction;
promotion transfers the exact accepted checkpoint without replaying accepted
target rows. Confidence is scheduling input, not correctness authority. A
rejected suffix is discarded before publication, and usage counts only
target-verified committed tokens.

## Backend prerequisites

Backends consume typed operations and physical operands. They own capability,
allocation, transfer, launch, synchronization, numerical execution, and
cleanup. They do not classify the family or reconstruct topology.

Family pressure may expose requirements for qtypes, tensor layouts, memory,
expert placement, attention state, or launch geometry. Pressure is planning
input, not backend support. An explicit backend request either runs the
admitted operation or refuses.

## Dense and sparse composition

A dense decoder uses a fixed FFN path for every token. A sparse/MoE decoder
adds routing, top-k selection, selected expert dispatch, shared-expert policy,
weighted accumulation, and conditional parameter access.

Sparse records distinguish total, active, shared, routed, resident, moved, and
executed parameters. Total parameter size is not token compute; active
parameters are not residency; residency is not observed movement.

## Persistent-state contract

The family defines state meaning, geometry, position rules, prefill writes,
decode reads, continuity, and invalidation. The common state provider owns
allocation, committed/candidate views, begin/stage/commit/abort, reset, and
cleanup. Persistent state and workspace never share semantic ownership.

## Evidence stages

Family records use the lowest true stage:

```text
not studied
-> source profiled
-> tensor inventoried
-> family classified
-> role mapped
-> artifact contracted
-> complete artifact admitted
-> materialized and bound
-> component execution proven
-> prefill/state/decode proven
-> complete generation proven
-> evaluated
-> benchmarked
-> release qualified
```

Each transition requires an owning implementation, positive and refusal tests,
an executable consumer, lifecycle/cleanup evidence, and exact identities.

## Failure classification

Missing configuration or tokenizer blocks source interpretation. Ambiguous
roles block mapping. Missing qtype/layout support blocks artifact or backend
admission. Missing placement blocks materialization. Missing graph consumers
block execution. Missing real state writes/reads block decode. Missing output
head, sampling, tokenizer stop, or detokenization blocks the complete text
path. Evaluation and benchmark remain unavailable until generation uses the
same hosted path exposed to the operator.

## Current records

- [DeepSeek-V4-Flash-DSpark](deepseek-v4-flash.md) is the sole complete
  source-to-text vertical and the only family with target-verified speculative
  generation; optimization, evaluation, benchmark, and release remain open.
- [Qwen](qwen.md) has source/header and candidate-role evidence only.
- [Gemma](gemma.md) has source/header and candidate-role evidence only.
- [MiniMax-H3 FL2VA](minimax-h3.md) is a source-profiled research target with
  reconciled tensor headers and an integration decision; it is not an admitted
  executable family.

No second complete family is currently admitted.
