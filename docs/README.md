# YVEX Documentation

Code, schemas, and tests own implemented behavior. This page routes readers
to current information owners; it does not duplicate their status.

## Entry and project method

| Question | Owner |
| --- | --- |
| What is YVEX and how do I start? | [README](../README.md) |
| Where is the project and what comes next? | [ROADMAP](../ROADMAP.md) |
| How do I contribute? | [CONTRIBUTING](../CONTRIBUTING.md) |
| What rules must an engineering agent obey? | [AGENTS](../AGENTS.md) |
| How does evidence drive agent-assisted progression? | [Agentic engineering](development/agentic-engineering.md) |
| Where do I get help or report vulnerabilities? | [Support](../SUPPORT.md), [security](../SECURITY.md) |
| What terms and attribution apply? | [License](../LICENSE), [notices](../NOTICE.md) |

## Architecture and contracts

| Subject | Owner |
| --- | --- |
| System boundaries and lifetimes | [System](architecture/system.md) |
| Source-to-package lowering and binding | [Compilation](architecture/compilation.md) |
| Engine execution, sessions, state, and resources | [Runtime architecture](architecture/runtime.md) |
| Command grammar and generated projections | [Commands](architecture/commands.md) |
| External terminal-editor ownership and pin | [REPLAI decision](decisions/0007-external-terminal-editor.md) |
| Family integration and accepted boundaries | [Integration](model-families/integration.md) |
| Family-specific facts and evidence barriers | [DeepSeek](model-families/deepseek-v4-flash.md), [MiniMax](model-families/minimax-h3.md), [Mamba2](model-families/mamba2.md) |
| Managed model storage and location | [Model storage](contracts/model-storage.md) |
| Artifact admission and lifecycle | [Artifact contract](contracts/artifacts.md) |
| Runtime behavior and failure semantics | [Runtime contract](contracts/runtime.md) |
| Native wire contract | [Local protocol](contracts/local-protocol.md) |
| Installed and cross-subsystem records | [C API](contracts/c-api.md) |
| Event and measurement semantics | [Events and telemetry](contracts/events-telemetry.md) |
| Bounded compatibility HTTP | [OpenAI compatibility](openai-compatibility.md) |

Architecture explains organization; contracts specify interfaces and invariants.
Family records do not replace either with another runtime description.

## Operation, development, and release

| Subject | Owner |
| --- | --- |
| Startup, acquisition, sessions, media, diagnosis, shutdown | [Operator runbook](operator-runbook.md) |
| Source membership and build projection | [Source ownership](development/source-ownership.md) |
| QA lanes, prerequisites, and evidence | [QA](development/qa.md) |
| Hardware measurement targets and empirical barriers | [GB10 targets](development/gb10-targets.md) |
| Independent reference provenance | [Reference baseline](development/reference-baseline.md) |
| Durable decision rationale | [ADRs](decisions/README.md) |
| Release gate semantics | [Release doctrine](releases/doctrine.md) |
| Version-specific release scope | [v0.1](releases/v0.1.md) |
| Public changes | [CHANGELOG](../CHANGELOG.md) |

Routine worklogs and retired implementation plans are recoverable from Git.
Documentation admission and history policy belong to the engineering method,
not a separate documentation registry.
