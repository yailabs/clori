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
| Classical REPL mapping and the REPLAI-backed chat lifecycle | [Interactive terminal path](architecture/system.md#interactive-terminal-path) |
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
| Models, provider authentication, local files, preparation and eviction | [Model lifecycle guide](model-lifecycle.md) |
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

## Canonical technical figures

| Figure / engineering question | Information owner |
| --- | --- |
| 1. Where are the system and process boundaries? | [System](architecture/system.md#system-boundary) |
| 2. How does source become executable reality? | [Compilation](architecture/compilation.md#pipeline) |
| 3. Who owns state, resources and execution lifetimes? | [Runtime](architecture/runtime.md#authority-map) |
| 4. Where do family semantics end and common mechanisms begin? | [Family integration](model-families/integration.md#promotion-path) |
| 5. What belongs to REPLAI versus YVEX? | [Interactive client](architecture/system.md#interactive-terminal-path) |
| 6. How do storage, working set and residency differ? | [Model storage](contracts/model-storage.md) |
| 7. Which evidence supports which claim? | [Engineering method](development/agentic-engineering.md#classify-evidence) |

Each owner links an editable JSON figure and embeds its generated SVG. The
bounded fixed-layout format gives panels, lifetimes and orthogonal routes exact
publication geometry; Mermaid auto-layout and the previously independent SVG
sketches are retired together. [The renderer](../tools/render_diagrams.py) uses
Python standard library only, without a browser, font download or build service:

```sh
python3 tools/render_diagrams.py
python3 tools/render_diagrams.py --check
```

JSON owns labels, coordinates, semantic classes, owner and source-authority
pointers; the renderer owns typography, geometry and edge styles. Edit JSON,
then regenerate, never maintain SVG independently. `--check` writes nothing and
rejects stale output. The ordinary documentation guard also tests the renderer,
pairs, references and orphan absence. Authority pointers aid review; they do not
automatically prove the figure's claims against code.

The monochrome palette uses the canonical mark's `#161616`, `#525252` and light
grays on an explicit white canvas. Lettered node classes remain meaningful in
grayscale; containment expresses scope, not a dependency arrow. Solid filled
arrows carry data/results, dashed open arrows control, dotted arrows observation,
dash-dot lines identity binding, and solid open arrows lifecycle/promotion.
Concepts use sans-serif text; exact identifiers can use monospace. SVGs contain
accessible titles/descriptions and no external resources or embedded HTML.
Use these same vectors for GitHub and print; temporary PNG/PDF previews are not
canonical assets. Review fonts, routes and scale after every material edit.
