# 0006 — Shared multi-family development

Date: 2026-08-24
Status: accepted

## Context

DeepSeek and MiniMax are architectural consumers of the same YVEX compiler,
artifact, runtime, server and backend substrate. They are not isolated model
products whose common mechanisms can evolve safely without reciprocal
qualification.

Mandatory separate family branches forced a generic repair to travel from one
family branch through `main` and then into another family branch before both
consumers could observe it. That cycle added integration latency and deferred
cross-family validation. Isolated worktrees also cannot expose uncommitted
changes to one another, so branch separation provided no substitute for
explicit source and index discipline.

## Decision

YVEX admits shared multi-family development branches and shared physical
worktrees. A branch is neither an agent identity nor a model-family identity.
Agent ownership remains the semantic delivery boundary: each agent edits,
tests, stages and commits only its owned behavior even when source state is
shared.

The working tree and Git index are mutable shared state. Contributors reread a
file and its current diff before editing, inspect the result afterward, and use
the index only as a short-lived focused commit transaction. Concurrent `HEAD`
advancement is normal. A conflict exists only where deliveries require
incompatible semantics; independent work may continue outside that overlap.

Generic changes may land directly on the active shared development branch.
They qualify every affected currently supported family before promotion.
Separate feature or family branches remain optional when isolation provides a
real integration benefit. Published histories are merged rather than rebased.
`main` remains the stable integration authority and is not a normal development
branch.

Repository-wide agent schedulers, databases, daemons and global source locks
remain forbidden. Coordination is limited to actual exclusive resources such
as one mutable build tree, GPU, live-model slot, fixed port or benchmark
directory. Long-running QA authenticates its source at both start and finish;
evidence over a mutated source snapshot is invalid even when its individual
tests passed.

## Consequences

- Common changes become visible immediately to all families on a shared
  branch.
- Supported families act as reciprocal architecture tests for generic owners.
- Cross-family qualification occurs before stable integration rather than
  after a chain of branch synchronization waves.
- Fewer mechanical integrations are needed to transport common substrate.
- Stale file reads and long-lived staged state become correctness hazards.
- Source-stable QA evidence and strict worktree/index discipline are mandatory.
- True semantic overlap requires explicit coordination and cannot be resolved
  by filename ownership or mechanical Git strategy.

This decision changes development topology only. It does not change product
capability, family support, runtime identity, performance, evaluation or
release state.

## Alternatives considered

One mandatory branch per family was rejected because it delays shared-substrate
qualification. One branch per agent was rejected because conversation or
worker identity is not source ownership. A global repository lock and a global
agent scheduler were rejected because they serialize independent work and
create a second coordination authority. Routing every generic commit through
`main` was rejected because `main` is stable integration, not an inter-agent
message bus. Treating isolated worktrees as shared uncommitted visibility was
rejected because Git does not provide that property.
