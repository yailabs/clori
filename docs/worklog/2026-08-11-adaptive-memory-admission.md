# Adaptive System-Reserve Admission

| Field | Value |
| --- | --- |
| Date | 2026-08-11 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `b6486478fd4c0ee8509a038741069bd5261007c8` |
| Checkpoint | `d46d9cffc56738fe497bae38afafaf185ddee019` |
| Subsystem | runtime model admission and generation capacity |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 operator incident |
| Evidence | software tests; operator incident characterization; source and contract inspection |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

Complete-model startup checked the encoded payload against a caller budget and
live memory while preserving a fixed 8 GiB minimum reserve. The check occurred
before artifact open, but a long artifact-authentication interval separated
that observation from resident-arena allocation. Generation independently
inserted the same fixed reserve into its workload profile.

An operator-observed full-model attempt ran beneath a 96 GiB process limit and
reached roughly 91.3 GiB resident memory before system-wide OOM pressure killed
unrelated services. These figures are retained as incident characterization;
the raw machine receipt is not tracked in Git.

## Problem

The admission contract conflated a fixed minimum reserve with the reserve
required by the effective machine or process capacity. It also trusted one
live-memory observation across artifact authentication. A request could
therefore pass the initial check and reach expensive residency mutation after
available memory had fallen.

## Causal analysis

The fixed 8 GiB reserve did not scale with the admitted capacity. The runtime
read cgroup availability but did not retain the corresponding effective
capacity needed to derive a proportional reserve. There was also a time-of-
check/time-of-use gap between preflight and allocation. The checkpoint diff and
its injected memory tests establish both defects; the incident establishes the
consequence, not a general performance bound.

## Decision

Keep admission in the generic runtime owner and derive its system reserve from
explicit resource facts. Preserve the greater of 8 GiB and one eighth of the
effective system, cgroup, and configured-host capacity. Revalidate live memory
after artifact authentication and immediately before residency allocation.

Rejecting the model before expensive mutation was chosen over relying on the
Linux OOM killer or weakening the reserve to fit the current artifact. General
topology placement and SSD-backed residency remain separate downstream owners.

## Implementation

Checkpoint `d46d9cffc56738fe497bae38afafaf185ddee019` replaced the availability-
only helper with a capacity-and-availability query across system and cgroup-v2
limits. One checked reserve function now feeds model-open admission and the
generation workload/capacity plan. Model open performs the same typed preflight
twice: before artifact mutation and after authentication, before the resident
arena opens.

The focused runtime-binding fixture injects deterministic total and available
memory, checks the proportional 16 GiB reserve for a 128 GiB capacity, verifies
pre-artifact host/system/cgroup refusal, and forces memory to shrink after hash
completion to prove residency never begins after the second refusal.

## After

Runtime startup refuses when payload plus the resource-derived reserve cannot
fit the configured budget or current effective availability. A decrease during
artifact authentication is caught before residency mutation. The stable
capacity plan retains the derived reserve without hashing transient free-memory
values into durable state identity.

This establishes OOM-safe admission policy for the current complete-model
loader. It does not establish that every admissible configuration will be fast,
that all transient allocations are fully modeled, or that the complete model
fits every DGX Spark workload.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Reserve rule at 128 GiB effective capacity | fixed 8 GiB minimum | 16 GiB | directly comparable contract fact |
| Live checks | one before artifact open | two; second before residency | directly comparable |
| Unsafe attempt | OOM at ~91.3 / 96 GiB | refusal ~0.10 s / ~7 MiB RSS | characterization only |

## Evidence

- Baseline `b6486478fd4c0ee8509a038741069bd5261007c8` and checkpoint
  `d46d9cffc56738fe497bae38afafaf185ddee019` delimit the repair.
- The checkpoint adds deterministic proportional-reserve, host-budget,
  system/cgroup, and post-hash TOCTOU refusal assertions to
  `tests/unit/runtime_binding.c`.
- Runtime and architecture contracts at the checkpoint state the two-stage
  pre-mutation refusal and the resource-derived reserve.
- The original operator incident and the approximately 0.10 s / 7 MiB refusal
  are characterization evidence reported during the development session. The
  absence of a retained raw receipt prevents promotion to a public performance
  claim.

## Remaining limitations

- Evidence gap: the raw incident and refusal measurement receipts are not
  retained in Git.
- The repair does not implement heterogeneous topology planning, multi-device
  placement, SSD weight streaming, or tiered residency.
- The complete loader still has model-loading and transient-memory costs that
  require separate measured optimization.
- This is runtime qualification evidence, not a model benchmark, evaluation,
  or release claim.

## Why it matters

Model admission now protects a resource-derived system reserve at the last
safe point before residency mutation, so an unsafe configuration is a typed
refusal rather than a machine-wide OOM experiment.

## Communication projections

### Short update

YVEX model admission now derives its system reserve from effective system,
cgroup, and configured-host capacity, then rechecks live memory after artifact
authentication before residency allocation. The repair turns the observed
OOM-prone startup path into a pre-mutation typed refusal; general topology and
SSD residency remain separate work.

### Longer post seed

1. A large model can fit a static payload check and still exhaust the machine
   during the transient loading window.
2. The root cause was not merely “too little RAM”: admission used a fixed
   reserve and trusted a stale pre-hash availability observation.
3. YVEX now derives the reserve from the effective resource envelope and
   repeats admission at the last point before resident allocation.
4. Deterministic tests force memory to shrink during hashing and prove that
   residency does not begin after refusal.
5. This is an admission repair, not yet a topology planner or loading-speed
   optimization.

### Article seed

**Possible title:** Refusing Before Residency: Making Large-Model Admission
Resource-Aware

**Central thesis:** Safe local inference requires modeling the transient
admission boundary, not only steady-state model bytes.

- Reconstruct the operator-visible failure without treating it as a benchmark.
- Explain fixed reserve versus effective-capacity-derived reserve.
- Show the artifact-hash TOCTOU window.
- Walk through pre-mutation refusal and deterministic fault injection.
- Separate this repair from future topology and SSD residency.

Strongest evidence: the checkpoint's two admission sites and the test that
reduces available memory after hashing but before residency.

### Visual candidates

- Before/after admission timeline around artifact authentication.
- Memory-envelope diagram separating payload, reserve, and transient overlap.
- Table mapping injected system/cgroup conditions to typed refusal fields.

### Quoteable technical facts

- “Model admission is revalidated after artifact authentication and before
  residency allocation.”
- “At 128 GiB effective capacity, the admitted reserve rule yields 16 GiB.”
- “Transient free memory is not part of durable capacity-plan identity.”
