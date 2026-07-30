# 0001 — Public project control

Date: 2026-07-30
Status: accepted

## Context

The former `PROJECT.md` mixed current priorities, 696 recovered and canonical
IDs, track accounting, capability snapshots, historical migrations, release
gates, and contributor instructions in one 2,701-line file. Maintaining every
historical row alongside each active transition obscured the current boundary
and made public contribution workflow depend on an internal recovery ledger.

The accepted product topology and frozen operator-surface audit provide enough
truth to extract the work that remains. Stable historical IDs still require
traceability, but not perpetual duplication in the working tree.

## Decision

Use four distinct project-control surfaces:

1. `ROADMAP.md` is the sole live macro authority for active/blocked milestones,
   dependency order, release-gate state, non-claims, and `Active Next`.
2. GitHub issues own bounded implementation problems and acceptance criteria.
3. Pull requests own delivery diffs and validation/progression evidence.
4. Decision records own durable architectural and doctrine choices.

`CONTRIBUTING.md` defines how those surfaces interact. Technical contracts,
runbooks, and audits retain their narrower authorities.

The former complete ledger is removed from the working tree. Its exact final
form remains available at commit
`447257dca7b122bafbddb86073d55eaa7be9513f`. Frozen audits retain their baseline
language and do not become live authority.

Stable IDs are never reassigned. Live open IDs stay in `ROADMAP.md`; completed
or superseded historical IDs are recovered from Git only when needed. A new
successor records structural changes instead of silently rewriting an old ID.

## Consequences

- A contributor can determine current priority by reading one concise file.
- Historical traceability remains exact without a parallel archive document.
- Project transitions change fewer duplicated rows and are easier to review.
- Repository guards must reject a restored `PROJECT.md`, multiple `Active
  Next` entries, or a mismatch between the active milestone and Active Next.
- Issues and boards may organize detail but cannot override the roadmap.
- Documentation-only project control cannot promote implementation capability.

## Alternatives considered

Keeping the full ledger as an appendix was rejected because it would preserve
the same duplicate maintenance burden. Moving the ledger unchanged under
`docs/` was rejected because it would create a shadow authority. Making GitHub
issues the sole authority was rejected because repository state would then be
unavailable from a checkout and could drift from reviewed code.
