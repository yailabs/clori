# Documentation Policy

Status: current development policy

YVEX documentation explains implemented code, public contracts, operator
procedures, or current project state. It never establishes capability by
assertion.

## Current authorities

- `README.md` is the bounded public entry, with room for an operational and
  architectural overview (at most 500 physical lines).
- `ROADMAP.md` is the only live macro project-control surface.
- `docs/architecture/` explains implemented ownership and execution.
- `docs/contracts/` owns externally relevant and cross-subsystem contracts.
- `docs/model-families/` owns current family integration and admitted family
  facts.
- `docs/operator-runbook.md` owns the normal operator journey.
- `docs/development/` owns build, QA, source ownership, and hardware
  qualification.
- `docs/decisions/` retains current durable architectural rationale.
- `docs/releases/` owns public release gates and version records.

Code, generated schemas, and tests remain authoritative for behavior. A
document links to those owners and does not duplicate mutable tables or
defaults that can be generated or inspected directly.

## Admission

Add a document only for a distinct audience, lifecycle, compatibility
contract, or durable subject. File size, chronology, a completed wave, and
directory symmetry are not reasons.

Delete retired plans, audits, migration matrices, and routine worklogs from the
working tree when they are no longer necessary to understand or operate the
current system. Git history is their archive. Do not create an `archive/`
directory or a manual inventory/hash registry for documentation.

Selected evidence records may remain when they are the compact canonical
explanation of a current empirical barrier. They are historical evidence, not
project-control, benchmark, or release authority.

## Editing

A documentation change names:

1. the implemented or policy fact that changed;
2. its canonical current owner;
3. any entry/navigation projection that must change;
4. the guard proving links and current vocabulary remain valid.

README changes are limited to the public purpose, working workflow, implemented
ownership, measured current evidence, validation entry points, material limits,
and navigation. Detailed procedures and contracts remain linked to their owners;
the expanded entry is not a second runbook or project-control ledger. Historical documents are
not rewritten to make old states look current; unnecessary historical
documents are removed and remain recoverable from Git.

Use concise subject names, relative links, and one source for mutable facts.
Current documentation must not require repository archaeology to correct it.

## Changelog

Record public capabilities, commands/protocol/API incompatibilities, package or
schema versions, operationally meaningful fixes, security changes, and release
facts. Omit internal wave names, generated identities, routine tests,
documentation moves, and refactor chronology unless they change the public
contract.

## Validation

Documentation changes run:

```sh
git diff --check
python3 tests/documentation_architecture.py
sh tests/test_project_control.sh
sh tests/test_docs_surface.sh
```

The documentation guard checks relative links, required current entry
surfaces, absence of retired governance directories/registries, one Active
Next, current protocol/command vocabulary, and public claim hygiene. It does
not freeze prose or force historical records to remain in the working tree.
