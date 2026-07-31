# Documentation Architecture Findings

Status: frozen baseline findings

## Authority conflicts

1. The public README acted as entry page, operator manual, doctrine,
   implemented architecture, family capability record, build guide, and
   documentation index.
2. `docs/contract.md` defined several independent normative boundaries whose
   versions and consumers change separately.
3. `docs/model-families.md` was both the common family contract and the current
   evidence record for DeepSeek, Qwen, Gemma, GLM, and hypothetical candidates.
4. The implementation-independent reference paper contained YVEX process
   diagrams and milestone-to-source traceability.
5. Stable release-gate meanings and volatile version readiness shared one
   document.
6. `system-target.md` duplicated project-like gap tracking inside a source and
   module ownership document.
7. Artifact terminology and claim-promotion language were repeated across
   README, artifact policy, family, runtime, reference, release, and contributor
   documents without one glossary/evidence authority.
8. Both operator milestone contracts contained live-state transition
   assignments, including `Active Next`, that belonged only in `ROADMAP.md`.

## Missing classes

The baseline had no public changelog, documentation-governance policy,
machine-readable document inventory, frozen-document checksum owner, canonical
glossary, standalone evidence doctrine, implemented-architecture layer, or
family-specific Qwen/Gemma records.

## Strong owners retained

`ROADMAP.md`, `AGENTS.md`, `CONTRIBUTING.md`, both ADRs, milestone contracts,
the command migration, the operator runbook, the OpenAI profile, and the
existing frozen operator audit already had coherent authorities. They were
kept, with narrow link/navigation changes and removal of milestone-owned live
state where required.

## Historical disposition

The topology audit is frozen at its final content and moved into the audit
class without editing. The operator-surface audit remains byte-identical at its
original path. Obsolete flat paths receive a migration mapping, not forwarding
documents. Detailed retired chronology remains in Git history.

## Claim review

No unsupported positive claim was required to explain the system. The
DeepSeek vertical remains generation-capable but unevaluated and unqualified;
Qwen and Gemma remain source/header evidence only; performance remains partial;
public security and release readiness remain false.

## Documentation-owned resolutions

All identified authority, terminology, navigation, current-command, stale-link,
document-class, changelog, governance, and validation gaps are resolved by the
migration. Remaining gaps belong to product milestones and are listed in
`gaps.tsv` rather than concealed by prose.
