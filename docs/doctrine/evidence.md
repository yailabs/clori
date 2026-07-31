# Evidence and Claim Discipline

Status: normative doctrine

This document owns the hierarchy by which YVEX evidence may support a claim.
[`ROADMAP.md`](../../ROADMAP.md) owns which current gates have passed.

## Evidence classes

| Class | Proves | Does not prove |
| --- | --- | --- |
| Software tests | Typed implementation behavior, refusal, lifecycle, and cleanup for their exercised scope | Numerical agreement, model behavior, or performance |
| Numerical conformance | Agreement with an independent exactness or tolerance contract | Repeated runtime lifecycle or user-visible model quality |
| Runtime qualification | Repeated identity, resource, transaction, cancellation, invalidation, and cleanup invariants | Model quality or throughput |
| Component benchmark | Performance of one already-correct identity-bound component | Full-model or release performance |
| Model behavior evaluation | Repeatable behavior over the complete tokenizer-to-text path | Quality against an external scorer unless one is defined |
| Model quality evaluation | Scored behavior against a declared task, dataset, and scorer | Runtime reliability or release readiness by itself |
| Agent runtime evaluation | A separately admitted action loop, tools, authority, state, budget, and termination | Ordinary text generation or tool-shaped output alone |
| Full-model benchmark | Latency, throughput, memory, and reliability for an exact full-model workload | Every context, policy, machine, or concurrency level |
| Release qualification | The complete release gate set for one declared target | Capability outside that release scope |

## Promotion rules

Evidence promotes only the exact boundary whose prerequisites, production
owner, positive and refusal tests, executable consumer, cleanup, and identities
are present. In particular:

- a software test is not numerical conformance;
- numerical conformance is not runtime qualification;
- runtime qualification is not model evaluation;
- component timing is not a full-model benchmark;
- HTTP syntax compatibility is not model quality;
- a complete artifact is not a supported artifact;
- executable generation is not release qualification.

Later evidence cannot repair a missing earlier contract. A fast result does not
admit incorrect arithmetic, and a correct primitive does not admit a missing
composition. Refusal behavior is part of every capability.

## Identity and scope

Evidence records every identity needed to reproduce or interpret it: source,
logical model, transformation, physical variant, artifact, binding, executable,
state, backend, machine, workload, and policy as applicable. It names whether
the scope is a fixture, tensor subset, component, complete text path,
evaluation, benchmark, or release.

Reports, traces, charts, and documentation are projections. They may preserve
facts but never become capability authorities. Raw benchmark/profile data,
model text, runtime state, and machine-local paths remain external operator
assets unless a contract explicitly admits a small deterministic fixture.

## Claim wording

Use the lowest true stage. State the target and boundary, then state the next
unproved stage. Prefer “the complete DeepSeek text path executes on the
admitted mixed GB10 path; evaluation remains open” over an unqualified
“DeepSeek is supported.”

Unsupported or unavailable facts remain explicit. Zero is not a substitute for
unavailable, and documentation reorganization cannot change a readiness fact.
