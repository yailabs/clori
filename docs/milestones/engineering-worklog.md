# Repository-Native Engineering Worklog

Status: accepted implementation-boundary contract

Milestone: `V010.DEVELOPMENT.ENGINEERING.WORKLOG.0`

Status source: [`ROADMAP.md`](../../ROADMAP.md)

This boundary turns selected material engineering changes into evidence-backed
semantic records without making documentation a capability authority or a
publication system.

## Accepted boundary

- `AGENTS.md` owns the concise trigger obligation.
- [The repository skill](../../.agents/skills/engineering-worklog/SKILL.md) is
  the sole procedural authority and is discoverable at repository scope.
- `build/worklog/` contains ignored drafts and intermediate projections.
- [`docs/worklog/`](../worklog/2026-08-11-adaptive-memory-admission.md) contains
  only intentionally retained records.
- Records distinguish checkpoint, repair, performance, and closure triggers;
  trivial changes require no record.
- Every record carries before, problem, causal analysis, decision,
  implementation, after, evidence, remaining limitations, and technical
  consequence.
- Optional communication projections introduce no new facts and trigger no
  publication.
- Publication state is explicit: `private-draft`, `reviewed`, or `public-safe`.

## Non-goals

This boundary adds no product command, runtime dependency, automatic social
publication, benchmark promotion, generated changelog, or mandatory diary of
every commit. It does not replace pull-request evidence, roadmap control,
decisions, audits, evaluation, benchmark, or release qualification.

## Acceptance

The skill passes the canonical skill validator, repository documentation
guards admit the worklog class and schema, and one retroactive DeepSeek
admission repair proves the semantic quality bar using an immutable baseline
and checkpoint. Existing product source and build membership remain unchanged.
