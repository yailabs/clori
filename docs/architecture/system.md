# Implemented YVEX System

Status: current implemented architecture

This document owns the implemented process topology and subsystem boundaries.
It explains the current system; it does not own project state, command syntax,
or release claims.

## System boundary

YVEX is one native compiler and inference system with two product executables:

```text
yvex   finite public command process
yvexd  long-lived local runtime process
```

`libyvex` supplies reusable compilation, artifact, runtime, graph, backend,
tokenizer, and generation owners. It is not a process or a separate public
command surface.

![Verified sources become an identity-bound artifact served by one long-lived runtime host to isolated sessions, CPU or CUDA execution, and local clients.](../diagrams/system_overview.svg)

The editable source is
[`system_overview.mmd`](../diagrams/system_overview.mmd). The diagram is a
curated explanation, not runtime evidence.

## Product processes

`yvex` has two mechanically separated lanes:

- the runtime-client lane uses the private local protocol for chat, one-shot
  generation, runtime administration, sessions, live model inspection,
  cancellation, watch, and trace;
- the finite offline-engine lane calls admitted library owners for compilation,
  artifact operations, inspection, direct component execution, profiling, and
  system facts.

The runtime-client lane cannot open an artifact, initialize CUDA, execute a
Transformer, or host a model. The offline lane closes all resources before the
process exits and never owns persistent sessions.

`yvexd` owns one process-lifetime runtime model, one bounded worker and queue,
one server-session registry, one private Unix listener, one loopback
OpenAI-compatible listener, and one telemetry authority. HTTP and native
clients enter the same worker, model, session, KV, and cancellation owners.

## Subsystem direction

The implemented dependency direction is:

```text
core and public ABI
  -> source, GGUF, artifact, model, tokenizer
  -> compilation and graph planning
  -> materialization and runtime
  -> backend execution
  -> generation
  -> evaluation and benchmark

domain facts -> typed results/events -> client render -> client I/O
```

Source owners establish provenance, retained tensor inventories, bounded
payload access, and trust. Compilation owners derive artifact-neutral
transformations and physical policy. GGUF and artifact owners serialize,
parse, authenticate, and admit complete artifacts. Runtime owners import the
binding, prepare immutable model resources, and isolate mutable execution
sessions. Graph and backend owners execute already-admitted operations; they do
not reconstruct model topology.

## Authority boundaries

| Boundary | Current owner |
| --- | --- |
| Source provenance, inventory, payload trust | `src/source/` |
| Family facts and irreducible lowering | `src/model/families/` |
| Artifact-neutral transformation and physical policy | `src/model/compilation/`, model compilation owners |
| GGUF container, qtypes, writer, layout | `src/gguf/` |
| Artifact snapshot, integrity, admission, materialization | `src/artifact/` |
| Semantic/executable graph and state protocols | `src/graph/` |
| Runtime binding, immutable model, sessions, residency, workspace | `src/runtime/` |
| Device capability, memory, kernels, launch graphs | `src/backend/` |
| Autoregressive composition | `src/runtime/generation.c` and typed generation owners |
| Hosted model, sessions, protocol, telemetry | `src/server/` |
| OpenAI-compatible projection | `src/server/openai/` and `src/provider/` |
| Command metadata and projections | `config/operator/registry.json`, generated descriptors, `src/cli/` |

The exact source-file ownership manifest is
[`config/source_owners.tsv`](../../config/source_owners.tsv). Contributor-facing
layout rules are in [Source and Module Ownership](../development/source-ownership.md).

## Application surfaces

The private local protocol is version 4. It carries typed requests, streamed
fragments, status, session results, progress, terminal results, and refusals.
The in-process OpenAI adapter translates the bounded compatibility profile to
the same protocol/session semantics. Neither transport enters graph, tokenizer,
sampling, or model APIs directly.

The command registry is validated at build time and compiled into `yvex` as
immutable descriptors. It owns operation IDs and projection metadata, not
domain behavior. Human help, JSON discovery, completion, dispatch syntax, and
REPL slash schemas consume that one authority.

## Current implementation scope

The admitted DeepSeek-V4-Flash vertical reaches streamed text through one
complete artifact and binding on CPU and the admitted mixed GB10 path. The
current CUDA path keeps model execution on CUDA while tokenizer work, sampling,
protocol handling, and orchestration remain host-owned. This is not a claim of
complete device residency, model evaluation, release benchmark performance, or
release qualification. Current gates remain in [`ROADMAP.md`](../../ROADMAP.md).
