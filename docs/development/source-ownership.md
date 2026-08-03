# Source and Module Ownership

Status: current filesystem and module ownership contract
Authority: source/module topology and placement; current project state belongs
only to [`ROADMAP.md`](../../ROADMAP.md).

This document records the current filesystem architecture for the
execution-core and control-plane boundary. It is an implementation contract,
not a capability claim.

## Current Actual Tree

Current core areas:

```text
config/operator/         canonical versioned command/operation registry source
tools/                   bounded build-time registry generator and validators
build/generated/operator generated immutable descriptor data, never tracked or installed
src/cli/io/client.c      runtime-client lane, REPL, protocol projections
src/daemon/              long-lived runtime-host entrypoint
src/server/              protocol, worker, sessions, telemetry, host lifecycle and server adapters
src/provider/            transport-neutral application request/result semantics
src/cli/                 unified runtime-client and offline command lanes, render and IO
src/source/              source manifests, provenance, inventory, payload trust/streaming
src/model/target/        generic target catalogs, gates and qtype reports
src/model/families/      family architecture, coverage and lowering recipes
src/model/artifacts/     model registry, reference, gate and write ownership
src/model/               core model tables and artifact-neutral compilation
src/gguf/                GGUF parser plus target ABI/writer/roundtrip owners
src/artifact/            artifact IO, identity, integrity, descriptor gates
src/graph/               graph core, plans, attention protocol/numeric owners and family recipes
src/backend/             backend abstraction, compute admission and platform implementations
src/runtime/             common immutable model, binding, execution sessions, state and benchmark
```

## Ownership flow

```text
registry JSON -> strict build validation -> immutable compiled descriptors -> yvex dispatch/help
product argv -> protocol v5 -> yvexd worker/session -> typed events/results -> client render
application -> OpenAI profile -> provider contract -> local protocol -> same yvexd worker/session
engineering argv -> nested owner route -> report/domain -> engineering render -> cli/io

file writer -> explicit local files only
source facts -> architecture IR -> coverage -> contribution map -> transformation IR
payload session -> bounded chunks -> transformation execution -> quantization
GGUF ABI -> artifact descriptor -> materialization -> runtime descriptor -> runtime binding
runtime binding -> immutable runtime model -> resident weights -> execution session
execution session -> attention prefill/decode phases -> state delta
persistent KV -> model prefill -> transformer -> logits -> sampling -> generation
```

No domain file owns command grammar or operator byte output. No renderer owns
domain algorithms. No writer owns command output.

## Owner Rules

- The operator-registry source owns operation IDs and projection metadata only.
- Its build-time generator owns strict validation and deterministic immutable C
  data emission; generated data owns no behavior.
- CLI input owns registry-driven syntax admission and typed domain argument
  adaptation only.
- CLI command adapters dispatch one admitted lane only.
- CLI surfaces resolve command families from compiled descriptors only.
- CLI renderers format typed facts.
- CLI IO writes operator bytes.
- Explicit writer modules write local files only.
- Core shard-index foundation owns allocation-free canonical key admission and
  deterministic lookup over caller-owned entries. Source payload consumes it;
  future artifact shard owners may reuse it without inheriting source policy or
  implying an artifact payload reader.
- Source JSON owns bounded structured parsing primitives, without source policy.
- Source provenance owns pinned repository/revision and manifest facts.
- Source family adapters own raw configuration and tokenizer sidecar facts.
- Source inventory owns indexed or explicitly header-derived shard inventory
  and the single canonical safetensors header pass. Its retained immutable
  snapshot carries deterministic tensor identity and lookup facts to consumers.
- Source payload owns snapshot-bound shard and tensor-range indexes, payload
  trust identity, bounded page/chunk plans, secure read-only handle admission,
  exact positioned reads, resource budgets, and transactional consumer
  delivery. It consumes the retained snapshot and never reparses headers.
- Model-target coverage owns IR-derived source requirements and exact snapshot
  reconciliation; it does not own source IO, GGUF naming, transforms, or
  payload access.
- The DeepSeek mapping owner composes IR and complete coverage into indexed
  source contributions and projects them into typed transforms, canonical GGUF
  names, logical GGML shapes, metadata prerequisites, and deterministic
  identity. This is a concrete GGUF lowering, not the artifact-neutral
  transformation owner. GGUF name/layout primitives remain their format
  owners; neither side reads payloads or emits physical file bytes.
- The DeepSeek payload handoff owner binds every canonical mapping contribution
  to the common source payload index and plan. It proves coverage and pressure
  facts but performs no source transform, quantization, or artifact emission.
- The compilation owner consumes architecture, coverage, source-contribution,
  and payload-range facts to construct immutable transformation plans and
  derivation identities. Physical-variant planning may consume typed
  format, quantization, residency, backend, evaluation, and benchmark
  requirements. Neither path performs source IO, quantization, writing,
  allocation, kernel execution, evaluation, or benchmark measurement.
- Transformation-plan construction is metadata-only. Transformation execution
  consumes exact payload chunks later, and quantization may not rediscover
  source names, roles, aggregation axes, or scaling companions.
- Source verification coordinates those owners and decides blockers; it does
  not parse JSON, rescan headers, render, serialize, or independently read
  tensor payloads.
- Source writers atomically publish verifier-approved manifests and explicit
  derived inventory outside official model source trees. Payload trust is
  published only after complete digest verification or explicitly local
  sealing; partial trust state is never published.
- Model target owns target catalogs, maps, gates, and qtype reports.
- Model architecture owns immutable normalized topology built from successful
  strict source verification. It does not reopen source files, classify tensor
  names, map roles, or infer runtime support.
- Model artifacts own registry, references, gates, typed artifact reports, and
  explicit file writers.
- GGUF owns container ABI, metadata ABI, tensor_info ABI, canonical qtype
  identity and row-aware storage geometry, ranges, reader, writer, roundtrip,
  emitted names, emitted layout, descriptor, and typed format facts.
- Artifact owns YVEX artifact descriptors, materialization boundary, roundtrip
  gates, identity, integrity, and artifact reports.
- Model owns runtime-descriptor projection. Runtime owns descriptor import and
  consumption, binding import, immutable model lifecycle, mutable sessions,
  residency, state, and bounded benchmark execution; model families supply
  typed family facts.
- Graph owns bind plans and graph execution boundary.
- Backend owns exact tensor, primitive, bundle, failure, cleanup, and qtype
  support/refusal facts.

## Forbidden Placements

- Domain command grammar.
- Domain operator byte output.
- Renderer file IO or backend execution.
- CLI files in `CORE_SRCS`.
- Writer modules writing command output.
- GGUF parsing language that implies materialization.
- Materialization language that implies backend execution.
- Backend language that implies graph or generation support.
- Payload, quantizer, or writer logic that becomes the transformation-semantics
  authority.
- Compilation coordination that reimplements source IO, quantization, writing,
  residency, backend execution, evaluation, or benchmark measurement.

## GGUF ownership map

| Owner | Boundary |
| --- | --- |
| `include/yvex/qtype.h` | public qtype identity, admission, and typed storage result |
| `include/yvex/artifact.h` | read-only file handle, optional explicit mapping, and exact positioned reads |
| `include/yvex/gguf.h` | public reader budgets, typed parse/layout results, immutable view, failures, byte/IO metrics, and accessors |
| `src/gguf/core.c` | file-backed GGUF v3 decoding, canonical container/metadata admission, and owned metadata/tensor view |
| `src/gguf/qtype.c` | pinned qtype registry and row-aware tensor storage |
| `src/gguf/layout_integrity.c` | bounded range arithmetic, canonical ordered layout, padding/span/tail/drift admission, and typed structural facts |
| `src/gguf/reader.c` | reader policy, resource defaults, and typed failure ABI |
| `src/gguf/writer.c` | transactional GGUF v3 writer planning and emission |
| `src/artifact/roundtrip_gate.c` | writer-reader equivalence boundary |
| `src/model/target/tensor_naming.c` | emitted GGUF tensor names and layout projection |
| `src/gguf/descriptor.c` | GGUF descriptor facts |

## Artifact and materialization ownership map

| Owner | Boundary |
| --- | --- |
| `src/artifact/descriptor.c` | YVEX artifact descriptor facts |
| `src/artifact/materialize.c` | admitted artifact materialization, range binding, and lifecycle |
| `src/artifact/roundtrip_gate.c` | emitted artifact roundtrip gate |

## Runtime ownership map

| Owner | Boundary |
| --- | --- |
| `src/runtime/descriptor.c` | runtime-descriptor ABI, deterministic import, validation, lookup, and typed result projection |
| `src/runtime/core.c` | immutable runtime-model and mutable execution-session lifecycle |
| `src/runtime/binding.c` | transactional, content-addressed runtime-binding serialization and admission |
| `src/runtime/residency.c` | read-only resident attention-weight packs and generation-bound invalidation |
| `src/runtime/graph.c` | execution descriptors, phase/mode dispatch, reusable workspace, and transactional publication |
| `src/runtime/benchmark.c` | identity-bound runtime timing, baseline, CSV, and deterministic SVG serialization |
| `src/server/core.c` | one-model host, private listener, bounded queue, worker and shutdown |
| `src/server/session.c` | exact conversation sessions, KV continuation, turns and partial state |
| `src/server/protocol.c` | bounded versioned local framing and thin protocol client |
| `src/server/telemetry.c` | one typed event sequence, subscribers and metrics accumulation |

## Model architecture ownership map

| Owner | Boundary |
| --- | --- |
| `src/model/families/deepseek_v4.c` | immutable architecture, exact source coverage, family Transformation IR construction, GGUF lowering, runtime-descriptor facts, and payload handoff for the admitted identity |
| `include/yvex/internal/families/deepseek_v4.h` | private typed DeepSeek recipe ABI shared by production consumers |
| `src/source/inventory.c` | retained immutable source tensor snapshot, deterministic identity, indexed lookup, one-header-pass and zero-payload-read accounting |
| `src/model/target/tensor_collection.c` | release-target collection projection from canonical coverage; Qwen/Gemma evidence remains separate |
| `src/model/target/missing_role.c` | release-target missing-role projection from canonical coverage |
| `src/model/target/mapping_gate.c` | operational projection of the canonical mapping plan and payload-streaming handoff |
| `src/model/target/model_class_profile.c` | strict source-verification coordination and report ownership for the canonical release target; Qwen/Gemma lexical evidence remains separate |
| `src/cli/render/model_target.c` | presentation of typed IR facts without architecture decisions |

## Graph and backend ownership map

| Owner | Boundary |
| --- | --- |
| `src/graph/plan.c` | runtime descriptor roles, immutable graph plan and backend admission facts |
| `src/graph/attention.c` | generic attention protocol, identity validation and transactional state boundary |
| `src/graph/numeric.c` | reusable attention numerical operations without family policy |
| `src/graph/state.c` | immutable prior-state views, candidate deltas, and transactional attention-state lifecycle |
| `src/graph/families/deepseek_v4.c` | DeepSeek schedule, recurrence and CPU/CUDA operation composition |
| `src/backend/core.c` | backend lifecycle, tensor binding and canonical qtype compute projection |
| `src/backend/report.c` | typed device, context, bundle, exact-variant, and memory reports |
| `src/backend/cuda/capability.c` | atomic generated-bundle admission, exact CUDA capability, launch/sync demotion, and cleanup failure |
| `src/backend/cuda/graph.c` | CUDA launch-graph registry, capture, instantiate, replay, update, invalidation, and release |
| `src/backend/cuda/ops.c` | validated host launch binding for admitted exact variants |
| `src/backend/cuda/kernels.cu` | canonical bounded device kernels; generated bundle remains build output |
| `src/backend/cuda/qtype.c` | CUDA qtype capability/refusal facts |

## Client and engineering ownership map

| Layer | Owner |
| --- | --- |
| Canonical operation and command metadata | `config/operator/registry.json` |
| Registry validation and deterministic C projection | `tools/generate_operator_registry.py` |
| Generated immutable descriptor data | `build/generated/operator/*` |
| Product entry, REPL and compact render | `src/cli/io/client.c` |
| Runtime-host entry | `src/daemon/yvexd.c` |
| Local protocol and host | `src/server/*` |
| Provider-neutral application contract | `include/yvex/provider.h`, `src/provider/core.c` |
| OpenAI compatibility adapter | `src/server/openai/*` |
| Unified command entry and nested dispatch | `src/cli/main.c` |
| Offline input/commands/render | `src/cli/input`, `src/cli/commands`, `src/cli/render` |
| Runtime-client and terminal IO | `src/cli/io/*` |

## GGUF Structural Reader Boundary

The artifact handle keeps a read-only file descriptor and maps the full file
only when a payload consumer explicitly requests `map`. The GGUF reader uses
exact positioned reads for the variable-size header, metadata, and tensor
directory, then owns copied length-aware values and names until close. Reader
metrics distinguish structural and payload bytes; structural open always
reports zero payload bytes. The separate canonical layout owner borrows the
opened artifact and parsed view, enforces power-of-two alignment and exact
directory-order padded continuation, validates zero padding and the complete
file span, and detects snapshot drift. It reads padding only and reports zero
tensor payload bytes. Complete-artifact admission remains separate.

## GGUF Qtype ABI Boundary

The qtype owner admits the pinned GGUF on-disk range, accounts removed and
outside-baseline identities, and derives bytes from the complete shape with
`ne[0]` as row width. Dtype, range, integrity, conversion, and memory-plan
owners project these facts instead of copying geometry. This boundary does not
provide reference dequantization, quantization, emission, backend arithmetic,
artifact completion, or runtime support. Current milestone state belongs only
to [`ROADMAP.md`](../../ROADMAP.md).

## Forbidden Claims

This target does not claim:

- public or remote serving, authentication, TLS, or compatibility APIs
- CUDA sampling, tokenizer execution, or fused logits/sampling execution
- evaluation or release-path full-model benchmark results
- a selected release artifact or release readiness

Attention-local prefill/decode phases operate on activation tensors and an
explicit state view. Runtime-local benchmark/profile output and deterministic
charts measure cold/warm latency plus identity-bound residency, transfer,
allocation, and graph-launch facts for that boundary only. Raw measurements,
generated charts, and machine-specific evidence remain external operator
assets.
