# Architecture Decision Records

Decision records preserve durable choices that affect YVEX architecture,
ownership, public ABI, protocols, executable topology, release doctrine, or
project-control doctrine.

They explain why a choice was made. They do not become implementation,
capability, milestone, or release authority. Current macro state remains in
[`ROADMAP.md`](../../ROADMAP.md); current technical truth remains in code and
its owning contracts.

## Format

Use a numbered Markdown file with:

- title;
- date and status;
- context;
- decision;
- consequences;
- alternatives considered; and
- supersession link when applicable.

Accepted statuses are `proposed`, `accepted`, and `superseded`. Do not edit an
accepted decision to reverse its conclusion; add a successor and mark the old
record superseded.

## Index

| Record | Status | Decision |
| --- | --- | --- |
| [0001](0001-public-project-control.md) | accepted | Separate compact live project control from historical ledger evidence and issue/PR delivery workflow. |
| [0002](0002-command-operation-registry.md) | accepted | Generate immutable command descriptors from one strict, versioned operation registry while preserving typed execution lanes. |
| [0003](0003-documentation-architecture.md) | accepted | Separate documentation by authority and lifecycle, with exact ownership and frozen-evidence validation. |
| [0004](0004-target-verified-speculation.md) | accepted | Keep target and DSpark drafting in one runtime model and publish only an atomically committed target-verified prefix. |
| [0005](0005-execution-substrate.md) | accepted | Separate execution identity from evidence depth, promote verified candidate state without replay, and admit CUDA work through typed profiles and shapes. |
| [0006](0006-shared-multifamily-development.md) | accepted | Admit shared multi-family branches and worktrees while preserving semantic ownership, source-stable evidence, and stable main integration. |
