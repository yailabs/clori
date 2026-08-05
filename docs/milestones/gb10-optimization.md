# Model-Derived GB10 Execution

Status: normative implementation-boundary contract

Milestone: `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0`

Status source: [`ROADMAP.md`](../../ROADMAP.md)

This boundary specializes the accepted DeepSeek-V4-Flash-DSpark vertical for
GB10 without creating a second runtime or making the common engine depend on
DeepSeek constants. Live state and dependency order belong only to
[`ROADMAP.md`](../../ROADMAP.md).

## Contract evolution

Contract versions change only when an admitted fact cannot be represented by
the current persistence, wire, layout, or binary boundary. The frozen
[contract matrix](../audits/gb10-optimization-691814/contracts.tsv) records the
reader, writer, migration, and test consequences for every considered change.

Runtime binding v8 is the first admitted persisted change. It authenticates the
source-derived model-execution descriptor that v7 cannot carry. V7 remains
readable as the retained reference profile; v8 is emitted beside it. The local
protocol remains v6, runtime events remain schema v3, Physical Execution IR and
compiled profiles remain schema v1, generation remains ABI v4, and the public C
API remains unchanged. A future protocol or public API revision requires a
concrete product consumer and its own matrix row.

An implementation-discovered identity change is admitted beside that frozen
entry audit:

| Contract | Current | Admitted | New fact | Incompatibility | Old behavior | Rule | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel-bundle identity | v2 | v3 | ordered set of independently compiled modules | semantic identity derivation only; no wire or persisted layout | v2 hashes one image and cannot identify the module set | full rebuild admits all manifest-owned modules atomically; model artifacts do not migrate | PTX/native admission, missing-symbol rollback, checked unload retry, identity mutation |

## Planning authority

Verified checkpoint and family facts seal one immutable model-execution
descriptor. Hardware probes seal a separate hardware profile; operator intent
seals a workload profile. Capacity and execution shapes are compiled from
those identities rather than from model-name branches or common-runtime
literals.

Capacity distinguishes model maximum, execution maximum, per-session and
per-request context, pooled state, candidate reserve, prefix budget, logical
batch, attention microbatch, MoE rows, output rows, workspace, scheduler and
system reserve. Page geometry is selected independently for each state class.
Its candidates derive from representation blocks, alignment, kernel tiles,
page-table cost, fragmentation, copy-on-write and promotion granularity.

Model residency is preflighted after bounded binding admission and before
artifact open or materialization. The admitted resident payload must fit both
the caller's host budget and currently available system memory while preserving
at least 8 GiB. Refusal reports configured or available bytes against required
bytes before the model candidate mutates artifact or residency state.

## Causal optimization

The phase roofline ledger is the priority authority. For prefill layer, decode
layer, verification sweep, draft sweep, output head, promotion and batched
decode it binds active weight, state, activation, temporary and transfer bytes;
launches, synchronizations, occupancy, duration, work and committed tokens.
Measured memory lower bounds and remaining headroom decide optimization order.
Attention, MoE and output-head dependencies do not prescribe that order.

The v1 ledger accepts partial measurements with explicit fact masks. It keeps
all causal phase slots, exposes missing phases and facts, and marks rankings
provisional until active-byte and movement evidence can establish a roofline.
The additive internal change does not bump a persisted or wire contract and
keeps the original zero-mask v1 writer representation readable.

The production CUDA generation path now records partial phase evidence without
depending on trace verbosity and exposes it through the finite offline
generation operator. Duration, work and committed-token accounting is live.
Target-only prefill/decode aggregate compulsory memory through one checked fact
type. Device-native embedding, attention, MoE and final projection contribute
exact active weights, touched state, external activations and actual temporary
workspace; measured and missing operation counts prevent a partial aggregate
from acquiring the complete memory mask. Transactional overflow preserves the
prior aggregate. These phases also report exact launch facts.
Output projection reports encoded weights, compulsory input/output activation
spans, actually allocated temporary spans, exact zero persistent-state bytes,
movement, launches and synchronization. Those facts complete its memory lower
bound without substituting allocation capacity or workspace peaks. Target-only
transformer phases additionally report exact
H2D/D2H/D2D movement and synchronization, including selected feature and
bounded status transfers. Output projection now consumes the same canonical
memory representation rather than maintaining a parallel report truth. DSpark
draft/verification transactionally merge the same transformer and batched
output-head memory facts with exact launch, H2D/D2H/D2D and synchronization
counters. Host feature projection, Markov, sampling and acceptance remain
outside the device-byte lower bound; their elapsed cost remains inside the
phase duration. Occupancy remains unavailable, so optimization ordering is
still provisional rather than a promoted kernel-priority decision.

Compatible width-N CUDA output rows now execute through one encoded-head
operation. Host producers use one bounded row-batch upload; device producers
must expose one contiguous identity-compatible view. The encoded head is
counted once, Q8 activation preparation and projection launch once for the
complete width, and the full output block is downloaded once for the retained
host sampling consumer. Aggregate physical facts belong to the ordered logits
execution rather than being multiplied into each row. Mixed, non-contiguous or
invalid source directories remain on the explicit row-local reference path,
which preserves complete-row failure semantics.

The same width-N owner can now retain the complete output block on CUDA and
publish one identity-bound row view per vocabulary row. Production device
results require a compiled profile, device-native execution class and
contiguous CUDA source directory; they cannot fall back to row-local or host
output. Their aggregate movement excludes the vocabulary download, each row
binds the correct target or draft state generation, and the device sampling
owner consumes the views before the reusable output buffer is overwritten.
The host result class and its full-array evidence remain the explicit
reference path. This uses the existing internal logits/device-view schemas and
changes no persisted, wire or public C contract.

Greedy DSpark target verification now consumes that resident result class.
One speculation-owned verification context borrows the existing model,
session, output-head residency and compiled profile; it projects all target
rows together, performs one width-N CUDA argmax launch and synchronization
through the sampling owner, and transfers only bounded aggregate
selection/status facts. Physical accounting is attached once to the ordered
batch rather than multiplied across logical rows. The accepted-prefix decision remains in
the target-authoritative speculation transaction, so cancellation and failed
verification cannot publish selected IDs or state. CPU, audit/forensic and
stochastic DSpark retain the complete-distribution oracle. The drafter feature,
Markov and confidence path remains host materialized and is still optimization
debt. No persisted, wire, public C or profile schema needed a version change.

CUDA attention graph replay now separates mutable state preparation from the
captured kernel topology. The graph-stream preamble refreshes the current state
bank before capture and every warm replay; promotion generation is therefore
not a graph-key fact while allocation, workspace, capacity and execution shape
remain compatibility facts. Live piecewise/full execution proves repeated
state commits against the independent attention oracle without recapture.

Target-only CUDA generation now cuts over from the eager reference to full
attention graphs when the admitted binding and live Driver expose the complete
graph capability. The existing compiled-profile schema already distinguishes
eager-reference ownership, so no contract version changes. CPU and DSpark stay
on the explicit eager reference until the target/draft arena is graph-stable.
Shape reconfiguration within one graph mode and profile identity retains
compatible cached executables; a mode or identity change still invalidates
them.

The CUDA kernel bundle now admits an ordered set of independently compiled
manifest-owned modules. General kernels and the MoE kernel family share one
toolchain-only qtype primitive interface, while module loading, required-symbol
resolution, rollback and checked unload remain atomic at the aggregate bundle
boundary. This earns kernel-bundle identity v3; it changes neither a persisted
model contract nor public API and creates the compilation boundary needed for
real width-N MoE without exceeding translation-unit policy.

Production CUDA MoE now consumes the existing width-N runtime contract through
one backend capability table. Workspace size derives from every admitted layer
qtype and the compiled row capacity. Each layer routes all rows, builds a
deterministic expert-major pair order, executes resident routed/shared packs,
and publishes only selected experts, weights, unique count, and bounded status.
The independent token-local CPU/CUDA implementation remains reachable only as
the portable audit/reference oracle. The internal source ABI rebuilds
atomically; no persisted, wire, public C, execution-profile, or state-layout
schema changes.

Target-only production stochastic sampling now keeps the complete vocabulary
row on CUDA. Runtime stages exactly one PCG transition, the backend applies the
sealed temperature/top-k/min-p/typical-p/top-p order and categorical draw, and
runtime publishes the RNG state only after bounded device facts survive
cancellation and validation. The correctness-first kernel downloads 100 bytes
of token, survivor, probability and status facts rather than the vocabulary.
Audit/forensic profiles and DSpark retain the host distribution oracle. The
existing compiled-profile reference flag and internal device-view contracts
already represent this cutover, so no persisted, wire, public C, execution-
profile or state-layout version changes.

Source-selected target features now collapse their mHC residual streams on
CUDA and publish each reduced row into both bounded host evidence and a
transaction-owned token-major device directory. Production feature projection
consumes that directory without re-upload, then executes its encoded width-N
matrix and RMSNorm on CUDA with exact activation, temporary, launch,
synchronization and movement facts. Its normalized device rows are then bound
to the draft core after current workspace and state generations are prepared;
the producer-owned digest removes the consumer's duplicate full-row host scan.
The CPU implementation and one bounded host materialization remain the
numerical oracle and full-evidence path. Feature evidence and draft/Markov work
are not yet fully device-resident, so this does not claim the complete DSpark
cutover and requires no persisted, wire, public C or profile-schema change.

Speculative prefill now contributes the merged target, projection and draft-core
physical facts to the phase roofline ledger. Checked addition is transactional,
and missing compulsory-memory operations remain unavailable rather than becoming
numeric zero or a complete lower bound.

The CUDA transformer final operation now has one optional device output for
the BF16 pre-normalized row. Production drafting reuses final-layer attention
storage after its last consumer, then explicitly materializes that bounded row
without downloading the expanded residual streams or recomputing the final
stage on the CPU. Aliased normalized and pre-normalized outputs fail before a
kernel launch. Full evidence retains expanded-row materialization and the CPU
final-stage oracle. This extends only the existing internal backend ABI and
earns no persisted, wire, profile or public C version change.

Accepted-prefix promotion now contributes its exact state-residency H2D,
synchronization and zero kernel/D2H/D2D facts. These counters are deltas around
the serialized session mutation, not estimates from configured capacity.
The uploaded destination span is the exact compulsory device-state byte count;
promotion has no active weights, device activations or temporary device span.
It therefore earns the complete memory mask without treating host-side
candidate projection as device traffic. Repeated occupancy evidence is a
work-unit-weighted mean with transactional overflow refusal.

Physical-variant research uses a funnel: role-level numerical probes,
representative-layer encoding, kernel microbenchmarks, bounded logit and DSpark
acceptance checks, then complete emission only for surviving candidates. A
calibrated candidate is not materialized merely because an uncalibrated recipe
exists.

Published target and stretch values remain engineering ambitions. A competitive
hard gate is admitted only from a same-checkpoint comparison or a measured
roofline plus active-byte lower bound. This distinction cannot lower an
existing target or manufacture closure.

## Serving authority

Continuous serving has one scheduler authority and one serialized mutation
domain. Thread count is not part of that semantic contract. Independently
admitted execution and I/O workers may be used when ownership, cancellation,
publication, fairness and cleanup remain exact. The public server API is not
versioned speculatively; its current entrypoints must converge on the admitted
scheduler before a concrete external consumer may justify a new ABI.

## Non-claims

The currently admitted code establishes model-derived geometry, v7/v8 binding
coexistence, typed capacity/page planning, pre-materialization memory refusal,
the availability-aware production phase ledger, and identity-bound native
SM121 CUBIN admission. It also establishes identity-bound width-N CUDA logits
publication and greedy DSpark target selection without full-vocabulary host
materialization.
It does not yet establish Tensor Core execution, specialized attention,
GB10-competitive grouped MoE or zero per-layer MoE synchronization, full-model
live qualification of target-only device stochastic sampling or greedy DSpark
verification, device-resident draft/Markov or stochastic DSpark
acceptance/correction, host-free target-feature evidence, paged state allocation,
prefix persistence, continuous batching,
competitive throughput, evaluation, benchmark qualification, release
qualification, or Hugging Face publication.
