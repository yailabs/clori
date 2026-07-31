# Canonical Operation and Command Architecture

Milestone: `V010.OPERATOR.COMMAND.ARCHITECTURE.0`

Track: `TRACK.OPERATOR`

State: blocked

Depends on: `V010.PROJECT.CONTROL.PUBLIC.0`

Successor: `V010.OPERATOR.REPL.CONSOLE.0`

This file is a future implementation contract. It does not establish command,
protocol, registry, or console capability.

## Mission

Replace the transitional, independently maintained YVEX command projections
with one canonical operation and command architecture.

The accepted input is the frozen post-cutover audit at
`docs/audits/operator-surface-ec7dcc/`. That audit records:

- 70 route-level commands;
- 426 command/flag pairs;
- 99 operations;
- 39 offline adapter routes;
- an independent runtime-client dispatcher;
- manually rendered top-level help;
- manually parsed slash commands and slash help;
- three provisional local-protocol facades; and
- no machine-readable discovery authority.

The accepted executable topology remains one public `yvex` command process and
one persistent `yvexd` runtime process. This milestone cannot restore a
developer executable, a separate OpenAI gateway, or a second runtime owner.

## Required Architecture

Create one versioned machine-readable operation source and checked generated
static C descriptors.

The source owns command-projection facts only:

- operation ID;
- command namespace and action;
- aliases and deprecation state;
- arguments and flags;
- typed defaults and validation references;
- visibility;
- side effects and lifecycle requirements;
- daemon, model, artifact, backend, and TTY requirements;
- protocol and interaction projections; and
- test and documentation owners.

The generated descriptors must drive or structurally validate:

- CLI dispatch;
- namespace and leaf help;
- machine-readable discovery;
- shell completion;
- slash-command discovery and adapters;
- the future TUI catalog;
- documentation checks; and
- positive, refusal, and lane-separation dispatch tests.

The registry does not own domain arithmetic, runtime objects, model policy,
artifact trust, protocol serialization, renderer implementation, or mutable
process state. Semantic defaults remain in their canonical domain owner; the
registry references them instead of copying them.

## Command-Semantic Repairs

The implementation must resolve the audit findings rather than encode them as
permanent compatibility behavior.

Required repairs:

1. distinguish selected model configuration from the live runtime model;
2. give model list and model show truthful, distinct semantics;
3. remove or truthfully implement:
   - `YVEX_CLIENT_OP_MODEL_SHOW`;
   - `YVEX_CLIENT_OP_ARTIFACT_SHOW`;
   - `YVEX_CLIENT_OP_ARTIFACT_VERIFY`;
4. dissolve the `evidence` namespace into semantic owners;
5. decompose the broad `graph` namespace by operation intent and visibility;
6. move quantization into the compilation plane;
7. separate hosted runtime administration from offline input/context proof;
8. choose one explicit artifact-verification grammar;
9. remove `runtime trace --follow` or give it real bounded semantics;
10. normalize repeated flags and defaults through canonical owners; and
11. remove or replace the 26 orphan historical CLI catalogs.

The accepted taxonomy is determined from the audit and implementation proof.
Compile, Run, Execute, and Integrate remain architectural planes; they are not
automatically literal top-level commands.

## Runtime and Console Facts

This milestone may add or repair typed backend and protocol facts required by
the subsequent console:

- selected model;
- live model;
- current client attachment;
- runtime configuration;
- context capacity and use;
- KV use;
- generation phase;
- prefill progress;
- decode progress;
- final stop reason;
- complete prefill, TTFT, decode, and publication timings; and
- metadata for an explicitly model-emitted reasoning channel.

Every fact must have one semantic owner and one typed transport. Renderer needs
cannot manufacture runtime truth. No protocol field may imply hidden model
reasoning.

## Projection and Lane Invariants

Runtime-facing `yvex` operations remain local-protocol clients. They cannot
open artifacts, initialize CUDA, or enter the offline engine lane.

Finite offline operations may consume admitted `libyvex` APIs but cannot host a
persistent model or create a second session/KV authority.

Slash commands, completion, help, HTTP relationships, and a future TUI are
projections of canonical operations. They cannot implement independent
argument parsing, defaults, state mutation, or failure classification.

## Out of Scope

This milestone does not:

- redesign terminal presentation;
- add decorative color;
- build the final REPL;
- retain a second slash parser;
- expose hidden reasoning;
- optimize CUDA execution;
- perform model evaluation or benchmarking; or
- change the accepted `yvex` plus `yvexd` process topology.

## Acceptance

The milestone closes only when:

- one versioned operation source exists;
- generated descriptors are deterministic and freshness-checked;
- every admitted command, flag, default, alias, projection, and side effect has
  one descriptor and one semantic owner;
- help, discovery, completion, slash catalog, documentation checks, and
  dispatch tests consume the same authority;
- runtime-client and offline-engine lanes remain mechanically separated;
- all required semantic repairs are implemented and tested;
- console-required facts are typed or explicitly reported unavailable;
- no compatibility executable or `dev` namespace returns;
- the REPL presentation remains unchanged except where required to consume a
  corrected operation contract; and
- refs agree and the worktree is clean.

## Downstream consumer

The mature runtime-console boundary consumes the registry, protocol facts, and
slash projections established here. Only [`ROADMAP.md`](../../ROADMAP.md)
activates that consumer or records milestone state.
