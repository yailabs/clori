# YVEX Documentation

This index is the navigation and authority map for YVEX documentation. It does
not own product capability or project state. Code and tests own implemented
behavior; [`ROADMAP.md`](../ROADMAP.md) owns the live project sequence.

## Start here

- [README](../README.md) — public project definition, minimal quick start, and
  current limits.
- [Operator runbook](operator-runbook.md) — start, inspect, use, and stop the
  resident runtime.
- [Canonical glossary](doctrine/glossary.md) — precise terms used throughout
  the project.
- [Implemented architecture](architecture/system.md) — what the current YVEX
  system is and where its boundaries lie.
- [Contributing](../CONTRIBUTING.md) — contribution, testing, and review
  workflow.

## Authority map

| Information | Canonical owner |
| --- | --- |
| Public definition and entry | [`README.md`](../README.md) |
| Macro state, gates, non-claims, Active Next | [`ROADMAP.md`](../ROADMAP.md) |
| Engineering contract | [`AGENTS.md`](../AGENTS.md) |
| Architectural thesis | [YVEX principles](doctrine/principles.md) |
| Canonical terminology | [Glossary](doctrine/glossary.md) |
| Claims and evidence | [Evidence discipline](doctrine/evidence.md) |
| Implementation-independent architecture | [Verified inference reference](reference/verified-inference.md) |
| Implemented process and subsystem topology | [Implemented system](architecture/system.md) |
| Compilation and physical artifacts | [Compilation architecture](architecture/compilation.md) |
| Runtime, execution, state, and resources | [Runtime architecture](architecture/runtime.md) |
| GB10 execution objectives and measurement key | [GB10 target table](development/gb10-targets.md) |
| Active GB10 implementation boundary | [Model-derived GB10 execution](milestones/gb10-optimization.md) |
| Command and operation projections | [Command architecture](architecture/commands.md) |
| Family integration | [Family integration contract](model-families/integration.md) |
| DeepSeek-V4-Flash-DSpark facts | [DeepSeek record](model-families/deepseek-v4-flash.md) |
| Qwen facts | [Qwen record](model-families/qwen.md) |
| Gemma facts | [Gemma record](model-families/gemma.md) |
| MiniMax-H3 FL2VA research facts | [MiniMax-H3 record](model-families/minimax-h3.md) |
| Artifact admission | [Artifact contract](contracts/artifacts.md) |
| Hosted runtime behavior | [Runtime contract](contracts/runtime.md) |
| Private local protocol | [Local protocol v8](contracts/local-protocol.md) |
| OpenAI-compatible HTTP | [Compatibility profile](openai-compatibility.md) |
| Installed and internal C interfaces | [C API](contracts/c-api.md) |
| Events and telemetry | [Events contract](contracts/events-telemetry.md) |
| Normal runtime operation | [Operator runbook](operator-runbook.md) |
| DeepSeek-specific operation | [DeepSeek operation](operations/deepseek.md) |
| Build and validation | [Validation](operations/validation.md) |
| Documentation governance | [Documentation policy](development/documentation-policy.md) |
| Source and module ownership | [Source ownership](development/source-ownership.md) |
| External reference traceability | [Reference-engineering baseline](development/reference-baseline.md) |
| Release-gate meanings | [Release doctrine](releases/doctrine.md) |
| v0.1 target and readiness contract | [v0.1 readiness](releases/v0.1.md) |
| Public change history | [`CHANGELOG.md`](../CHANGELOG.md) |

## Evidence and change records

- [Decisions](decisions/README.md) own durable choices and rationale.
- [Audits](audits/documentation-architecture-51a5c/README.md) preserve
  point-in-time evidence; the
  [code-commentary migration](audits/code-commentary-7c90ce1/README.md),
  [execution-substrate disposition](audits/execution-substrate-dd91fb/README.md),
  [GB10 optimization baseline](audits/gb10-optimization-691814/README.md),
  [MiniMax-H3 FL2VA intake](audits/minimax-h3-fl2va-b8b09e3/README.md),
  [repository-compression disposition](audits/repository-compression-7226f7/README.md),
  operator-surface, and topology audits remain frozen beside the documentation
  baseline.
- [Migrations](migrations/documentation-architecture-v1.md) map superseded
  paths and terms to current owners; the
  [DSpark source migration](migrations/deepseek-dspark-source.md) records the
  sole-target replacement and refusal boundary.
- [Milestone contracts](milestones/documentation-architecture.md) define
  bounded implementation acceptance without owning live state; the
  [commentary contract](milestones/code-commentary.md) records the selective
  source-commentary boundary; the
  [DSpark rebase contract](milestones/deepseek-dspark-rebase.md) owns the
  source-to-speculative-text implementation boundary; and the
  [execution-substrate contract](milestones/product-architecture.md) records
  the identity, state-promotion, shape and operational-surface boundary; and
  the [GB10 optimization contract](milestones/gb10-optimization.md) owns the
  active model-derived execution boundary.

## Document classes

The repository separates stable doctrine, implementation-independent
reference architecture, implemented architecture, family records, normative
contracts, operator procedures, development policy, project control,
decisions, frozen audits, migrations, releases, and milestone contracts.
Their update rules are defined by the
[documentation policy](development/documentation-policy.md) and checked
against [`config/documentation_owners.tsv`](../config/documentation_owners.tsv).

Decision records explain why durable choices were made. Frozen audits preserve
point-in-time evidence. Migration records map superseded paths and vocabulary.
Milestone contracts describe one implementation boundary but never own live
status. Detailed retired chronology remains recoverable from Git history.
