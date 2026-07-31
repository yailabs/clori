# 0003 — Documentation information architecture

Date: 2026-07-31
Status: accepted

## Context

YVEX documentation grew as implementation boundaries closed. Strong material
existed, but flat documents mixed stable doctrine, implementation-independent
architecture, current implementation, model-family facts, normative contracts,
operator procedure, project state, release gates, and historical audit detail.
Several subjects had competing definitions and future changes were routinely
asked to update broad sets of files.

The canonical command architecture and public roadmap provide stable inputs
for separating those information classes without changing product behavior or
capability claims.

## Decision

Organize documentation by authority and lifecycle: doctrine, reference,
implemented architecture, family records, contracts, operations, development,
milestones, decisions, audits, migrations, and releases. Keep root surfaces
limited to public/repository entry, project control, contribution, change
history, and legal material.

Use `config/documentation_owners.tsv` as the exact inventory of tracked
Markdown classes and canonical subjects. It is validation metadata, not a
runtime or content source. Use `config/frozen_documents.tsv` to preserve
accepted point-in-time audits byte-for-byte.

`docs/development/documentation-policy.md` owns creation, split, projection,
README, roadmap, changelog, ADR, audit, migration, and validation rules. One
canonical glossary and one evidence doctrine remove repeated terminology and
claim hierarchies.

Frozen audits retain their historical language. Git history remains the
authority for retired detailed chronology, so no shadow project ledger or
historical documentation monolith is created.

## Consequences

- Every living document has a declared class, authority mode, audience, and
  subject.
- Stable concepts can change independently from volatile project state.
- Family integration and individual family evidence no longer compete in one
  monolith.
- Contracts, architecture explanation, and operator procedures have different
  owners.
- Moving a document requires a migration record rather than a forwarding copy.
- Documentation validation becomes an explicit repository guard.
- Editorial restructuring cannot promote implementation or release readiness.

## Alternatives considered

Keeping the flat layout with a larger index was rejected because it would not
remove mixed authority. Creating one file per subsystem was rejected because
source structure is not an information architecture. Copying old documents
into category directories was rejected because it would preserve duplicate
truth. A generated documentation site was rejected because the repository
needs reviewable Markdown authority and no runtime documentation dependency.
