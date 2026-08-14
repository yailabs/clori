# Documentation Policy

Status: normative documentation governance

This document owns how YVEX documentation is classified, created, updated,
linked, frozen, migrated, and validated. It does not own product capability or
project state.

## Authority rules

Code and tests own implemented behavior. `ROADMAP.md` owns live macro state,
gates, non-claims, and Active Next. `AGENTS.md` owns repository engineering
rules. ADRs own durable decisions and rationale. Every other document has one
bounded subject and may only summarize facts owned elsewhere.

[`config/documentation_owners.tsv`](../../config/documentation_owners.tsv) is
the machine-readable inventory of tracked Markdown documents. It records each
document's class, authority mode, audience, lifecycle, and canonical subject.
The manifest is governance metadata, not product behavior.

## Document classes

| Class | Responsibility | Lifecycle |
| --- | --- | --- |
| entry | Public or repository entry and navigation | durable, compact |
| doctrine | Stable thesis, terminology, and claim rules | changes only with doctrine decision |
| reference | Implementation-independent architecture or external baseline | stable technical revision |
| architecture | Explanation of currently implemented YVEX structure | changes with implemented boundaries |
| family | Common integration contract or one family's admitted facts | changes with family evidence |
| contract | Normative interfaces, lifecycle, input/output, side effects, refusal, compatibility | versioned with owner |
| operations | Executable operator or validation procedure | changes with admitted commands/behavior |
| development | Contributor policy, ownership, and engineering reference | changes with repository practice |
| project-control | Live macro state | `ROADMAP.md` only |
| milestone | One implementation-boundary contract | retained; never live status authority |
| decision | Durable choice, context, consequences, alternatives | immutable after acceptance except status/supersession correction |
| audit | Point-in-time evidence | frozen after acceptance |
| migration | Deterministic old-to-new mapping | retained historical bridge |
| release | Public change history, stable release doctrine, or version record | versioned/public |
| legal | License and notice obligations | legal lifecycle |
| contribution | Public contribution workflow | changes with workflow |
| test-support | Narrow fixture or test-format explanation | changes with test owner |
| worklog | Selected evidence-backed semantic engineering event | retained; publication state advances only by review |

## Canonical owner and projection

One document owns each major subject. Another document may include a short
summary and a link, but it must not copy mutable tables, defaults, capability
state, or normative wording. When the summary and owner could drift, keep only
the link and local consequence.

A documentation update names:

1. the admitted implementation or doctrine fact that changed;
2. the canonical owning document;
3. any projections whose summary or navigation must change;
4. the validation proving no competing authority was created.

Generic instructions to “update all docs” are invalid. README is changed only
when the public entry, public capability, minimal workflow, current release
status, limits, or navigation changes.

## Creation and split criteria

Create or split a document only when it gains a distinct authority, audience,
lifecycle, compatibility contract, or volatility class. Directory symmetry,
chronology, file length alone, and prospective future content are not reasons.

Merge or remove documents that compete for the same subject, mix project state
with stable architecture, preserve obsolete command grammar, or have no
consumer. Detailed retired chronology belongs in Git history rather than a new
living archive.

## Status vocabulary

Use:

- `normative` for a contract or doctrine owner;
- `current` for implemented architecture, family facts, or procedure;
- `unreleased` for a version record not yet qualified;
- `accepted` or `superseded` for ADRs and milestone contracts;
- `frozen` for point-in-time audits;
- `historical` only when a retained document is explicitly not current truth.

Documentation status never changes implementation readiness. Avoid vague
labels such as “complete support” or “production ready.”

## Naming and links

Documentation paths use concise lowercase concept names, with hyphens for
multiword prose documents and snake case only for machine-owned formats.
Prefer subject nouns and avoid milestone IDs outside `docs/milestones/`.

Internal links are relative, resolve to tracked files or headings, and use the
canonical path. Do not add forwarding documents solely to preserve an old
pre-v0.1 path. A migration record provides historical lookup.

Root documents remain limited to public/repository entry, project control,
contribution, changelog, license, and notices. A root exception requires a
clear public entry purpose.

## Root surfaces

- `README.md` is the durable public entry, not a technical manual.
- `ROADMAP.md` is the only live macro control surface and contains exactly one
  Active Next.
- `CHANGELOG.md` records externally meaningful public changes, not every
  commit, test, milestone, or refactor.
- `CONTRIBUTING.md` explains contribution workflow.
- `AGENTS.md` is the engineering contract, not a product manual or glossary.

## Decisions, audits, milestones, and worklogs

An ADR records a durable choice and its rationale. Accepted ADR content is not
silently rewritten; a new ADR supersedes it.

An accepted audit is frozen. Its original terms, links, counts, and findings
remain point-in-time evidence even when current paths change. Navigation and a
migration record explain the current owner. Frozen hashes are checked through
[`config/frozen_documents.tsv`](../../config/frozen_documents.tsv).

A milestone contract states one implementation boundary and acceptance. Its
state appears only in `ROADMAP.md`; a milestone document must not contain
Active Next.

A worklog records the semantic before/problem/cause/decision/after boundary of
one selected engineering event. It may project communication material, but it
does not own implementation, project state, benchmark, evaluation, or release
claims. Its technical record remains stable after acceptance; publishability
may advance only through explicit sensitivity and evidence review. Intermediate
worklog material remains outside tracked documentation.

## Deprecation and migration

When a living document moves, splits, merges, or is deleted, add one bounded
migration record mapping the old path to the resulting canonical owner or Git
history. Do not leave duplicate compatibility copies. Deprecated terminology
is recorded in the [glossary](../doctrine/glossary.md); active owners use the
replacement.

## Changelog policy

Record an entry only for public capability, command/protocol/API incompatibility,
executable topology, admitted family boundary, artifact/schema version,
operationally meaningful fix, security change, or release. Internal wave names,
test additions, generated identities, and documentation moves are omitted
unless they materially change public navigation or contract.

## Validation

Documentation changes run:

```sh
git diff --check
python3 tests/documentation_architecture.py
sh tests/test_project_control.sh
sh tests/test_docs_surface.sh
make check-docs
```

The architecture validator checks manifest coverage, authority uniqueness,
links and anchors, required entry surfaces, frozen hashes, removed paths,
deprecated command vocabulary, console/current-future wording, family-stage
language, changelog shape, and project-control exclusivity.

Documentation-only work does not run the model, restart the daemon, or claim
numerical validation. Build/test dependency changes additionally satisfy the
repository's repeat-check rule.
