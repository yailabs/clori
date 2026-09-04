# YVEX Documentation

Code and tests own implemented behavior. [ROADMAP](../ROADMAP.md) owns the live
project sequence. This page is only the compact route to current architecture,
contracts, operation, and evidence.

## Start here

- [Public README](../README.md) — what YVEX does, current evidence, quick start,
  and limits.
- [Operator runbook](operator-runbook.md) — discover or register a package,
  start the host, load an engine, run work, inspect it, and unload it.
- [Implemented system](architecture/system.md) — package, specialization,
  engine, state, scheduler, backend, and evidence boundaries.
- [Contributing](../CONTRIBUTING.md) — build, tests, review, and safe shared
  development.

## Architecture and contracts

| Subject | Current owner |
| --- | --- |
| Compilation and immutable package | [Compilation architecture](architecture/compilation.md) |
| Engine/runtime/state/resources | [Runtime architecture](architecture/runtime.md) |
| Command and operation projections | [Command architecture](architecture/commands.md) |
| Model-family integration | [Family integration](model-families/integration.md) |
| DeepSeek vertical | [DeepSeek-V4-Flash](model-families/deepseek-v4-flash.md) |
| MiniMax composite vertical | [MiniMax-H3](model-families/minimax-h3.md) |
| Artifact admission | [Artifact contract](contracts/artifacts.md) |
| Runtime behavior and failure | [Runtime contract](contracts/runtime.md) |
| Local transport | [Local protocol v19](contracts/local-protocol.md) |
| Installed and internal C ABI | [C API](contracts/c-api.md) |
| Events and observability | [Events and telemetry](contracts/events-telemetry.md) |
| OpenAI-compatible HTTP | [Compatibility profile](openai-compatibility.md) |

## Development and release

| Subject | Current owner |
| --- | --- |
| Repository engineering rules | [AGENTS](../AGENTS.md) |
| Source/build ownership | [Source ownership](development/source-ownership.md) |
| QA lanes and evidence reports | [QA](development/qa.md) |
| GB10 measurement targets | [GB10 targets](development/gb10-targets.md) |
| External reference provenance | [Reference baseline](development/reference-baseline.md) |
| Documentation admission | [Documentation policy](development/documentation-policy.md) |
| Durable current decisions | [Architecture decisions](decisions/README.md) |
| Release gate meanings | [Release doctrine](releases/doctrine.md) |
| v0.1 target | [v0.1 record](releases/v0.1.md) |
| Public changes | [CHANGELOG](../CHANGELOG.md) |

One selected record retains the empirical boundary immediately preceding the
current work:

- [DeepSeek GB10 representation barrier](worklog/2026-08-28-deepseek-gb10-matrix-tile-execution.md).

They are evidence history, not live architecture or project control. Detailed
retired audits, migrations, milestone plans, and routine worklogs remain
available through Git history instead of the current documentation tree.
