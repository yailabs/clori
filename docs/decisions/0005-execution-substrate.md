# 0005 — Identity-bound execution substrate

Date: 2026-08-03
Status: accepted

## Context

The first complete DeepSeek-V4-Flash-DSpark vertical proved source-to-text
correctness, but its ordinary path retained mechanisms originally introduced
for reference evidence: whole-array host digests, token-local width handling,
one mutable attention-capacity view, and replay of target work already executed
during speculative verification. Those mechanisms made correctness observable,
but also made the host the implicit numerical authority and coupled execution
identity to the cost of proving every intermediate.

The filesystem also exposes three same-named DeepSeek implementation files.
Without an explicit ownership rule they can look like competing family
runtimes, even though each projects one different layer of the dependency DAG.

## Decision

YVEX separates semantic identity from evidence depth and physical execution.
An identity-bound Physical Execution IR projects admitted terminal tensors into
consumer, layout, placement, activation, backend, shape and fallback decisions.
A compiled execution profile then binds the logical model, physical variant,
Physical Execution IR, artifact, materialization, runtime binding, kernel
bundle, hardware, context, generation mode, workload, evidence profile and
reference-adapter class. Paths, pointers, timestamps and process identities do
not enter either identity.

Evidence is admitted as one of `production`, `audit` or `forensic`.
Production retains bounded status, token, acceptance, stop, failure and
accounting facts but does not require full hidden, logits, probability or state
materialization merely to hash them. Audit admits bounded probes and device
digests. Forensic admits explicitly requested full materialization and retained
reference comparison.

Persistent attention state is prefix-addressable. Multi-row target verification
retains ordered candidate deltas and rolling checkpoints. Acceptance promotes
the exact verified prefix; a rejected suffix is discarded. The only additional
target execution admitted is one correction or bonus token required by the
speculative algorithm. Accepted target rows are never replayed in the product
path. Sliding-window history uses a ring-backed contiguous view rather than a
full-window move.

CUDA admission uses a registry keyed by execution profile, attention plan,
state layout, kernel bundle, target/draft scope, phase, width, context band,
candidate visibility, capacities, workspace generation and evidence profile.
A compatible eager/reference shape may be constructed and admitted before
numerical mutation; incompatible capacity refuses before streaming and names
the failed component, configured and required values, position, width and
identities.

Device numerical values cross owners through typed views. Full host
materialization is explicit as scalar, row, audit-digest or forensic-full work.
The CUDA production greedy path selects the token on the device and transfers
bounded result facts. Exact host stochastic sampling remains an explicitly
named portable-reference adapter until a later device implementation preserves
the complete target distribution.

Product levels are not source directories. A model family may project its
irreducible facts at exactly three current boundaries:

- `src/model/families/deepseek_v4.c` owns source interpretation, family facts,
  coverage and logical lowering;
- `src/graph/families/deepseek_v4.c` owns the irreducible execution recipe;
- `src/backend/cuda/families/deepseek_v4.c` owns fused CUDA lowering that cannot
  be expressed by generic backend operations.

The common runtime is family-neutral. A concrete family hierarchy beneath the
runtime namespace is forbidden, as are concrete-family imports in
materialization and generic runtime owners.
Machine-readable ownership and architecture guards enforce this distinction.

Protocol version 6 carries the incompatible partial-turn and explicit-reasoning
facts. Reasoning is classified only by the tokenizer's source-authored
DeepSeek prompt/output contract. It is never inferred from prose or exposed as
hidden chain of thought. Terminal Markdown and operational rendering are
projections of canonical bytes and typed events; raw and redirected output stay
byte-preserving and free of terminal controls.

## Consequences

- The GB10 optimization work can change packing, placement, kernels, graph
  capture and scheduling without reopening identity, state publication,
  capacity ownership or output-channel semantics.
- Portable token-local MoE, row-local output projection, eager attention and
  host stochastic sampling remain visible reference classes rather than the
  common ABI.
- A partial session carries exact committed-state facts and requires explicit
  reset before another ordinary turn.
- The canonical artifact remains the byte authority. A future derived
  execution asset must bind both artifact and Physical Execution IR identities
  and can never be trusted by path alone.
- Protocol 5 is refused rather than decoded as protocol 6.
- This decision establishes no performance, evaluation, benchmark or release
  claim.

## Alternatives considered

Increasing one attention-capacity constant was rejected because target, draft,
verification and correction have different shapes and context-depth demands.
Replaying accepted target rows was rejected because verification already
produces authoritative candidate state. Keeping full-array hashes in normal
execution was rejected because evidence depth is not semantic identity.
Implicit host pointers were rejected because they make materialization an
accidental consumer contract. A family-specific runtime directory was rejected
because it would duplicate model, session, state and telemetry authority.
Renaming the three valid family projections solely to hide their common family
identity was rejected; directory namespace and the ownership manifest already
state their distinct roles.
