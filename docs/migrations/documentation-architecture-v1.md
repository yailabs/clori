# Documentation Architecture v1 Migration

Baseline: `51a5c087eafe857d71df1566ce90c2f87a2fcfc1`

This record maps every documentation surface in the pre-refoundation Git tree
to its current owner or historical disposition. It is a migration aid, not a
content or project-state authority. Removed paths have no forwarding copies.

| Baseline path | Disposition | Current owner or recovery |
| --- | --- | --- |
| `.github/pull_request_template.md` | kept | `.github/pull_request_template.md` |
| `AGENTS.md` | kept; governance rule added | `AGENTS.md` |
| `CONTRIBUTING.md` | kept; authority map updated | `CONTRIBUTING.md` |
| `LICENSE` | kept | `LICENSE` |
| `MODEL_ARTIFACTS.md` | moved and edited | `docs/contracts/artifacts.md` |
| `NOTICE.md` | kept | `NOTICE.md` |
| `README.md` | rewritten as compact public entry | `README.md` |
| `ROADMAP.md` | kept and transitioned | `ROADMAP.md` |
| `docs/api.md` | moved and link-normalized | `docs/contracts/c-api.md` |
| `docs/audits/operator-surface-ec7dcc/README.md` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/commands.tsv` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/dispositions.tsv` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/findings.md` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/flags.tsv` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/operations.tsv` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/project-control-input.md` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/registry-input.md` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/surfaces.tsv` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/target-taxonomy.md` | frozen unchanged | same path |
| `docs/audits/operator-surface-ec7dcc/workflows.md` | frozen unchanged | same path |
| `docs/cli-output-architecture.md` | moved and narrowed | `docs/architecture/commands.md` |
| `docs/contract.md` | split by normative lifecycle | `docs/contracts/runtime.md`, `docs/contracts/local-protocol.md`, `docs/contracts/events-telemetry.md` |
| `docs/decisions/0001-public-project-control.md` | kept | same path |
| `docs/decisions/0002-command-operation-registry.md` | kept | same path |
| `docs/decisions/README.md` | kept; decision 0003 indexed | same path |
| `docs/diagrams/autoregressive_execution.mmd` | kept | same path |
| `docs/diagrams/autoregressive_execution.svg` | kept | same path |
| `docs/diagrams/physical_compilation.mmd` | kept | same path |
| `docs/diagrams/physical_compilation.svg` | kept | same path |
| `docs/diagrams/runtime_host_sessions.mmd` | kept | same path |
| `docs/diagrams/runtime_host_sessions.svg` | kept | same path |
| `docs/diagrams/system_overview.mmd` | kept | same path |
| `docs/diagrams/system_overview.svg` | kept | same path |
| `docs/logo.svg` | kept | same path |
| `docs/migrations/command-architecture-v1.md` | kept | same path |
| `docs/milestones/command-architecture.md` | kept; live-state transition removed | same path |
| `docs/milestones/runtime-console-repl.md` | kept; live-state transition removed | same path |
| `docs/model-families.md` | split by authority and family | `docs/model-families/integration.md`, `docs/model-families/deepseek-v4-flash.md`, `docs/model-families/qwen.md`, `docs/model-families/gemma.md` |
| `docs/openai-compatibility.md` | kept at registry-bound path; authority clarified | same path |
| `docs/operator-runbook.md` | kept at registry-bound path; navigation clarified | same path |
| `docs/reference-architecture.md` | split and moved | `docs/reference/verified-inference.md`, `docs/development/reference-baseline.md` |
| `docs/runbooks/README.md` | merged and deleted | `docs/README.md` |
| `docs/runbooks/common.md` | moved and corrected | `docs/operations/validation.md` |
| `docs/runbooks/deepseek.md` | moved and narrowed to procedure | `docs/operations/deepseek.md` |
| `docs/system-target.md` | moved; live gap table removed | `docs/development/source-ownership.md` |
| `docs/topology-closure-audit.md` | moved without content change and frozen | `docs/audits/topology-closure.md` |
| `docs/v010-release-doctrine.md` | split by volatility | `docs/releases/doctrine.md`, `docs/releases/v0.1.md` |
| `tests/vectors/README.md` | kept | same path |

## New canonical owners

The migration adds `docs/README.md`, doctrine principles/glossary/evidence,
implemented system/compilation/runtime architecture, family-specific records,
separate runtime/protocol/events contracts, documentation policy, release
doctrine/readiness, a public changelog, and exact document/frozen ownership
manifests.

## Historical recovery

The complete pre-refoundation contents remain available at the baseline commit.
The topology audit and operator-surface audit remain in-tree frozen evidence.
Detailed retired project chronology remains at the commit named by ADR 0001;
no `PROJECT.md`, history monolith, or duplicate living owner is introduced.
