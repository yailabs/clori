# Operator Surface Audit — ec7dcc

This directory is a frozen audit of the operator and automation surface at
baseline ec7dccede90c1a1efa87b4c2519c25b30d5e1733
(refactor(product): consolidate command and runtime processes).

It records what the baseline exposes. It is not a command registry, is not read
by production code, and must not be edited to describe later behavior. Later
changes should create a new identity-bound audit or update the eventual
canonical operation authority.

## Method

The audit combined source-complete static extraction with bounded dynamic
discovery:

1. verified HEAD, main, origin/main, and remote main at the exact baseline;
2. confirmed a clean main worktree;
3. created a detached worktree at
   /tmp/yvex-operator-surface-audit-worktree;
4. built libyvex, yvex, and yvexd there without deploying them;
5. inspected ELF dependencies, symbols, archive members, entrypoints, and
   embedded help;
6. read the complete CLI, daemon, server, provider, protocol, OpenAI, tests,
   package, runbook, architecture, and project-control owners named by the
   delivery;
7. extracted parser branches, route descriptors, literal options, protocol
   enums, HTTP method/path dispatch, Make targets, shell scripts, and consumed
   environment/configuration inputs;
8. invoked only help/version surfaces from the detached build;
9. queried the resident daemon only through read-only status/process/listener
   inspection; and
10. reconciled every extracted category against the TSV inventories.

No generation request was issued. The resident daemon was not stopped,
restarted, signalled, or replaced.

During discovery, one initial build command was accidentally executed in the
main checkout rather than the detached worktree. It rebuilt root products but
did not deploy, signal, or restart the resident daemon. The resulting yvexd and
libyvex hashes matched the accepted products and the running daemon executable.
All authoritative discovery builds were then repeated in the detached
worktree. This procedural deviation changed no tracked file and no running
process.

## Baseline context

| Fact | Value |
|---|---|
| Audit time | 2026-07-30 |
| Git baseline | ec7dccede90c1a1efa87b4c2519c25b30d5e1733 |
| Host architecture | aarch64 |
| Compiler | GCC 13.3.0 (Ubuntu toolchain) |
| Production C/CUDA/header paths | 210 |
| Product executables | yvex, yvexd |
| Implementation library | libyvex |
| Local protocol | version 3 |
| Package profile | bin/yvex, bin/yvexd, metadata and notices |
| Resident process topology | one yvexd with Unix and loopback HTTP listeners |

The detached build produced exactly one yvex main and one yvexd main. It did
not produce yvex-dev or yvex-openai.

| Detached product | SHA-256 | Bytes |
|---|---|---:|
| yvex | 12fd8fb1730c5d838cc08e8a9520dbf17d28fa3c8bfc905c70af1a9f85b2fa3a | 4,041,960 |
| yvexd | 535edca046a9ecb88d7a331afb286f7d9205bcec850775d10bdc6b3dd553cf74 | 3,207,608 |
| libyvex.a | 1b87041ce6547d05853a6b29ba9b382305e055c2a363fa228160626fb6c31df3 | 4,811,976 |

libyvex contained 119 source-relative archive members. yvex had 1,321 defined
global symbols; yvexd had 1,028; the archive had 1,076. Their dynamic
dependencies were limited to the system loader, libc, and libm at this build
profile.

## Inventories

| Inventory | Rows | Unit |
|---|---:|---|
| commands.tsv | 70 | route-level executable or REPL command |
| flags.tsv | 426 | command/flag pairs |
| operations.tsv | 99 | transport-independent or explicit projection operations |
| surfaces.tsv | 312 | binaries, protocols, targets, scripts, catalogs, environment/config inputs |
| dispositions.tsv | 382 | one disposition for every command and surface row |
| Offline dispatch rows | 39 | exact src/cli/main.c routes |
| Runtime-client command rows | 20 | implicit/product/nested actions |
| REPL slash branches | 10 | exact parser branches |
| Daemon command rows | 1 | yvexd process grammar |
| Local protocol operations | 17 | exact enum values |
| HTTP endpoints | 5 | exact method/path pairs |
| Make targets | 121 | concrete Makefile targets |
| Shell scripts | 52 | repository test/operator scripts |
| Environment/build inputs | 100 | explicit getenv, script environment, and ?= build inputs |
| Orphan CLI catalogs | 26 | unconsumed .def files under src/cli/catalog |
| External child-CLI projections | 2 | Hugging Face and GitHub argv adapters |

commands.tsv intentionally records each of the 39 hardcoded offline route rows
once. Nested action sets are preserved in the positional-arguments field and
expanded where they represent distinct operations, notably the 29 graph
leaves. This keeps route-table reconciliation exact without pretending a
parent dispatcher is itself a numerical capability.

flags.tsv records flags at the parser route that currently owns them. Every
long option parsed by the current C/H CLI and daemon owners is represented;
five additional literals used only to construct external provider argv are
classified as child-CLI projections rather than falsely presented as yvex
flags. A broad
dispatcher such as graph or evidence models therefore has action-dependent
rows. That action dependence is itself an audited defect and an input to the
future registry; it is not silently normalized here.

## Inspected authorities

The inspection covered, completely, the sources required by the delivery:

- AGENTS.md, PROJECT.md, README.md, MODEL_ARTIFACTS.md, Makefile, package and
  source-owner manifests;
- src/cli/main.c, src/cli/private.h, and every source/private header under the
  CLI commands, input, I/O, render, and model-artifact trees;
- all 26 historical .def files under src/cli/catalog and their production
  reference graph (which is empty at this baseline);
- src/daemon/yvexd.c;
- every server, server/OpenAI, and provider source/private header;
- installed and internal headers defining client operations, server messages,
  telemetry, provider requests, artifacts, models, and profiles;
- all CLI, client, REPL, OpenAI, package, architecture, layout, documentation,
  project-ledger, integration, and live scripts;
- operator-runbook, CLI-output, API, contract, OpenAI compatibility, reference
  architecture, system target, and topology closure documents; and
- Make target recipes, service invocations, package instructions, and the
  unresolved command/operator obligations in PROJECT.md.

Searches included argv/argc branches, strcmp/strncmp, option literals, usage
text, getenv, XDG and YVEX inputs, YVEX_CLIENT_OP values, endpoint dispatch,
Make rules, shell case branches, and command invocations in tests/docs. Help
output was supporting evidence only, never the sole discovery source.

## Completeness result

All required reconciliation gates close at zero unmatched:

| Gate | Discovered | Inventoried | Unmatched |
|---|---:|---:|---:|
| Offline routes | 39 | 39 | 0 |
| Runtime-client routes/actions | 20 | 20 | 0 |
| REPL slash branches | 10 | 10 | 0 |
| yvexd options including help/version | 16 | 16 | 0 |
| Parsed CLI/daemon long-option names | 246 | 246 | 0 |
| External child-CLI-only option names | 5 | 5 | 0 |
| Local protocol operations | 17 | 17 | 0 |
| OpenAI endpoints | 5 | 5 | 0 |
| Concrete Make targets | 121 | 121 | 0 |
| Shell scripts | 52 | 52 | 0 |
| Consumed environment/build inputs | 100 | 100 | 0 |
| Historical CLI catalog files | 26 | 26 | 0 |

Every command has a documentation status and a disposition. Every inventory ID
is unique, every TSV row has the declared column count, and rows are sorted
deterministically. No machine-specific artifact, binding, service environment,
prompt, response, trace, or generated evidence path is committed.

## Classification summary

The 70 command rows classify as:

| Visibility | Count |
|---|---:|
| product-default | 34 |
| product-advanced | 21 |
| engineering | 15 |
| automation command | 0 |
| API-only command | 0 |
| test-only command | 0 |
| direct executable removal candidate | 0 |

API-only integration is represented by five HTTP operation rows. Automation
and test-only surfaces are represented in surfaces.tsv rather than promoted to
commands. No current executable route is deleted by this audit; evidence is a
namespace dissolution candidate, three protocol facades are removal/redefinition
candidates, and one inert flag is a removal/definition candidate.

## Top findings

- The unified yvex topology is real: runtime-client routes remain protocol-only
  and offline routes enter a separately guarded in-process engine lane.
- Thirty-nine offline routes are hardcoded independently from product dispatch,
  top-level help, slash dispatch, and slash help.
- model list and model show call the same selected-configuration renderer;
  neither lists models nor reports the live daemon model.
- protocol operations MODEL_SHOW, ARTIFACT_SHOW, and ARTIFACT_VERIFY all return
  the same runtime status message and are provisional false facades.
- /status sends SESSION_SHOW for the current session, while yvex runtime status
  sends RUNTIME_STATUS. Their names imply a shared operation that does not
  exist.
- runtime trace accepts --follow but follows with or without it.
- runtime watch exposes generic a/b event payloads rather than the semantic
  operational view claimed by documentation.
- current human runtime status omits authoritative physical-variant and context
  facts that the JSON response contains and the CLI-output document promises.
- the evidence namespace mixes claim/report language with account login,
  downloads, path mutation, model selection, CUDA probing, and bounded
  materialization.
- graph spans execution, state, capture, profiling, qualification, benchmarking,
  and direct generation under one implementation term.
- sampling and runtime defaults are repeated across client, daemon, provider,
  and generation owners without one syntax/default descriptor.
- there is no machine-readable command discovery authority.
- twenty-six historical option/field/boundary .def catalogs have no production
  include or consumer and therefore cannot serve as current authority.

findings.md contains the severity assessment and source evidence. No P0
authority/correctness violation was found. P1 findings must be resolved or
explicitly accepted by the later command milestone before declaring the final
public grammar.

## Overlap accounting

The audit analyzes 18 named overlap groups.

- 4 exact duplicate surface groups;
- 5 semantic-overlap groups requiring one declared authority;
- 39 offline adapter projections, of which 23 reconstruct or rename the lower
  adapter spelling;
- 15 user-facing transport/interaction projections (10 slash commands and 5
  HTTP endpoints), backed by 17 local protocol operations.

These categories are not summed: a projection can belong to an overlap group
without being a duplicate implementation. The audit deliberately preserves
separate syntax and renderers when they project one authoritative operation.

## Target recommendation

Keep the accepted process topology and orient future human discovery around:

- default chat and one-shot run;
- runtime, session, and selected/live model administration;
- compile and artifact workflows;
- inspect, profile, and system support planes;
- engineering execution reachable but absent from default help; and
- eval and bench only when their actual capability milestones exist.

Conceptual planes remain Compile, Run, Execute, and Integrate. They are not all
required to become literal top-level namespaces. target-taxonomy.md records the
proposed spelling and migration decisions.

The next implementation should use one machine-readable source schema that
generates checked static C descriptors. The compiled descriptors should drive
dispatch metadata, parsing, help, slash adapters, completion, discovery, tests,
documentation checks, and the future TUI catalog. Domain APIs remain the
semantic authorities; the registry owns syntax and projection metadata only.

## Backend and renderer gaps

The runtime already owns authoritative readiness, model identities, resident
memory, session state, position, generation progress, TTFT, stop, cancellation,
and typed raw events. Most console pressure is therefore a renderer/projection
gap.

Protocol gaps remain for selected-versus-running model semantics, explicit
runtime configuration, current client attachment, detailed KV/context
projection, thinking policy, and a typed model-emitted reasoning channel.
YVEX must not invent hidden reasoning or expose chain of thought. Only explicit
model output admitted by tokenizer/family policy can become a separate channel.

## Exclusions

This audit did not:

- modify C, CUDA, headers, Make behavior, parsers, dispatch, protocol, OpenAI,
  REPL, package, services, or runtime behavior; the only test edits advance the
  two project-control/documentation guards to the accepted successor state;
- inspect or run external application repositories;
- issue model generation;
- restart or deploy the daemon;
- establish numerical, evaluation, benchmark, or release evidence;
- implement the proposed taxonomy or registry; or
- reopen the accepted yvex/yvexd process topology.

## Files

- surfaces.tsv — binaries, protocol, package, Make, scripts, environment/config.
- commands.tsv — route-level CLI, daemon, and slash grammar.
- flags.tsv — parser-owned flag pairs and normalization findings.
- operations.tsv — semantic operations and transport projections.
- dispositions.tsv — current-to-proposed decisions.
- findings.md — severity and overlap analysis.
- workflows.md — fourteen current workflows and future shortest truthful forms.
- target-taxonomy.md — proposed command discovery hierarchy.
- registry-input.md — canonical operation-registry design input.
- project-control-input.md — open ledger extraction for the successor.
