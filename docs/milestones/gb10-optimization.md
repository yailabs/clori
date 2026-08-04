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
the phase-ledger contract, and identity-bound native SM121 CUBIN admission.
It does not yet establish Tensor Core execution, specialized attention,
width-N production MoE, device
stochastic sampling, device-resident DSpark, paged state allocation, prefix
persistence, continuous batching, competitive throughput, evaluation,
benchmark qualification, release qualification, or Hugging Face publication.
