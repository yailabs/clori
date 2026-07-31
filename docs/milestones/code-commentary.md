# Natural Technical Commentary

Milestone ID: `V010.REPO.CODE.COMMENTARY.0`

Status source: [`ROADMAP.md`](../../ROADMAP.md). This contract does not own
live milestone state.

## Mission

Replace mandatory template-shaped source comments with selective technical
commentary that explains intent, rationale, ownership, lifetime, invariants,
trade-offs, and non-obvious behavior. Obvious code remains quiet; public and
cross-subsystem interfaces retain the contracts their types cannot express.

## Required boundaries

- `config/source_owners.tsv` remains the sole source-ownership authority;
- module comments describe shared decisions rather than repeat manifest facts;
- private helpers are commented only when their rationale is not evident;
- state, concurrency, transaction, resource, failure, algorithm, and hardware
  boundaries preserve useful engineering explanation;
- the canonical structural scanner rejects obsolete labels, repeated
  boilerplate, stale topology names, commented-out code, and ownerless
  maintenance markers without scoring prose mechanically;
- every tracked first-party C, CUDA, header, Python, shell, and build source is
  reconciled by the commentary guard;
- production lexical tokens outside comments and insignificant whitespace are
  unchanged.

## Exclusions

This boundary changes no algorithm, control flow, ABI, protocol, command,
output, synchronization, data layout, build product, runtime capability, or
readiness fact. Generated outputs, vendored code, licenses, exact test vectors,
and frozen audits remain outside commentary migration.

## Acceptance

The boundary closes when the repository engineering contract and its canonical
guard enforce natural selective commentary, all governed first-party sources
have been reviewed, obsolete templates are absent, interface contracts remain
adequate, production token equivalence is exact, focused and repository-wide
checks pass, and the frozen baseline audit proves the migration without
becoming another policy authority.
