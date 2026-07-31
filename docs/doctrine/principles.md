# YVEX Principles

Status: normative doctrine

This document owns the stable architectural thesis of YVEX. It is independent
of a command grammar, model family, backend, repository layout, or current
milestone.

## Purpose

Open weights are not an executable model by themselves. Reproducible inference
requires an identity-preserving chain from an identified source snapshot,
through model semantics and physical representation, to admitted resources,
stateful execution, and published output.

YVEX exists to make every boundary in that chain explicit. A consumer should
be able to determine which exact object was consumed, which transformation was
performed, which state changed, which resource owner was responsible, and what
evidence supports the resulting claim.

## Identity-bound derivation

Each semantic transition produces a distinct identity. Source snapshots,
logical models, transformation plans, physical variants, artifacts, runtime
bindings, runtime models, sessions, executions, state transitions, outputs,
and evidence are related but not interchangeable.

An upstream semantic change invalidates dependent identities. Process-local
facts such as pointers, padding, paths, process IDs, and timestamps never enter
a semantic identity.

## Logical and physical separation

The logical model defines topology, tensor roles, numerical policy, sequence
state, and composition. A physical variant chooses qtypes, dtypes, layouts,
alignment, partitioning, and placement constraints. Multiple physical variants
may implement one logical model without becoming the same artifact or runtime
binding.

Physical policy is resolved once by its owner. Artifact writers,
materialization, and runtime execution consume that decision; they do not
independently rediscover or override it.

## Compilation, admission, and execution

Compilation derives explicit physical model facts from verified input.
Admission proves that a complete object satisfies the contract of its next
consumer. Execution consumes admitted objects and may not reconstruct compiler
truth from names, paths, or incidental layout.

These boundaries are intentionally separate. Parsing is not admission;
admission is not materialization; materialization is not execution; execution
is not evaluation or release qualification.

## Fail-closed capability

A capability is bounded by exact input, identity, backend, mode, and resource
requirements. Missing or incompatible prerequisites produce a typed refusal.
An explicit accelerator request never becomes host execution, a graph request
never aliases eager execution, and a stale binding never opens against a
different artifact.

## Transactional state

Persistent state is published only after the corresponding execution succeeds
and validates. Candidate state, numerical output, cancellation, and failure are
resolved before commit. A failed or cancelled unit preserves the exact earlier
committed prefix and reports partial progress truthfully.

## Explicit lifecycle ownership

Every resource has one owner and a bounded lifetime. Immutable model resources
may be shared; mutable session state is isolated. Persistent state is distinct
from workspace. A client connection is distinct from a runtime session, and a
runtime session is distinct from the process-lifetime runtime model.

## Scoped evidence and claims

Evidence is bound to the identities and scope it actually observed. Reports
and renderers project evidence but cannot create capability. A lower evidence
class never promotes a higher one. The canonical hierarchy and promotion rules
are defined in [Evidence and Claim Discipline](evidence.md).

## Architectural consequence

YVEX prefers fewer, stronger semantic owners. Family code supplies irreducible
model facts and scheduling; common owners retain lifecycle, state, resource,
backend, and evidence mechanisms. A new abstraction is admitted by a concrete
consumer and executable contract, not by hypothetical reuse.
