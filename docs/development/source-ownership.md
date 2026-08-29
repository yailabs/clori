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
config/source_owners.tsv canonical production ownership and source membership
config/operator/         canonical versioned command/operation registry source
tools/                   bounded build-time projection generators and validators
build/generated/sources.mk deterministic source/build projection, never tracked
build/generated/operator generated immutable descriptor data, never tracked or installed
src/cli/io/client.c      runtime-client lane, REPL, protocol projections
src/server/              persistent host, engine manager, protocol, sessions, telemetry and adapters
src/provider/            transport-neutral application request/result semantics
src/cli/                 unified runtime-client and offline command lanes, render and IO
src/source/              source manifests, provenance, inventory, payload trust/streaming
src/model/target/        generic target catalogs, gates and qtype reports
src/model/families/      family architecture, coverage and lowering recipes
src/model/artifacts/     model registry serialization, references and gates
src/model/catalog.c      distinct local source/package catalog records
src/model/remote.c       provider-neutral remote model and representation records
src/model/materialization.c backend-owned weight materialization lifecycle
src/model/               core model tables and artifact-neutral compilation
src/gguf/                GGUF parser plus target ABI/writer/roundtrip owners
src/artifact/            artifact IO, identity, integrity, descriptor gates
src/graph/               semantic/physical execution plans, state, attention and family recipes
src/backend/             backend abstraction, compute admission and platform implementations
src/runtime/             family-neutral immutable model, sessions, execution and benchmark
```

## Ownership flow

```text
registry JSON -> strict build validation -> immutable compiled descriptors -> yvex dispatch/help
product argv -> protocol v16 -> host routing -> exact engine/session -> typed results -> client render
application -> OpenAI profile -> provider contract -> local protocol -> same server worker/session
engineering argv -> nested owner route -> report/domain -> engineering render -> cli/io

file writer -> explicit local files only
source facts -> architecture IR -> coverage -> contribution map -> transformation IR
payload session -> bounded chunks -> transformation execution -> quantization
GGUF ABI -> artifact descriptor -> package Physical Execution IR -> materialization -> binding v15
binding -> deployment specialization -> immutable engine generation -> execution session
ready sequence -> engine scheduler -> execution batch -> backend -> transactional state delta
persistent KV -> model prefill -> transformer -> logits -> sampling -> generation
```

No domain file owns command grammar or operator byte output. No renderer owns
domain algorithms. No writer owns command output.

## Owner Rules

- `config/source_owners.tsv` is the sole handwritten production membership
  authority. Its deterministic generator validates exact `src/` and `include/`
  parity and projects product/toolchain classes under `BUILD_DIR` for Make.
- Aggregate file, line, header, owner, and symbol counts are review metrics, not
  architectural ceilings. Admission is decided at each ABI, lifecycle,
  algorithm, backend, family, or entrypoint boundary; machine guards enforce
  owner partitions, consumer sets, family budgets, dependency direction, and
  per-file and per-function limits.
- The root Makefile composes products and supported validation entrypoints from
  that projection. It does not repeat individual production paths or use
  source wildcards.
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
- The tokenizer compilation owner converts those source-authored facts into one
  pointer-free identity-bound policy; runtime and server consume it without
  importing or enumerating family implementations.
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
- Artifact owns YVEX artifact descriptors, bounded artifact-range
  materialization, roundtrip gates, identity, integrity, and artifact reports.
- Model materialization owns backend weight-table construction and lifecycle;
  it does not change semantic or package identity.
- Graph physical execution owns stable terminal consumer, package layout,
  canonical qtype, encoded range, and persisted numerical-obligation facts.
- Model owns runtime-descriptor projection. Runtime imports binding/package
  truth, seals deployment specialization, owns immutable engine generations,
  model resources, mutable sessions, ready-work scheduling, residency, state,
  and bounded benchmark execution; model families supply typed semantic facts.
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
| `src/gguf/core.c` | file-backed reader lifecycle, policy defaults, typed failure ABI, decoding, metadata admission, and owned container view |
| `src/gguf/writer.c` | transactional GGUF v3 writer planning and emission |
| `src/artifact/roundtrip_gate.c` | writer-reader equivalence boundary |
| `src/graph/families/deepseek_v4.c` | DeepSeek tensor roles, emitted GGUF names, and family lowering policy |
| `src/gguf/descriptor.c` | GGUF descriptor facts |

## Artifact and materialization ownership map

| Owner | Boundary |
| --- | --- |
| `src/artifact/descriptor.c` | YVEX artifact descriptor facts |
| `src/artifact/materialize.c` | admitted artifact-range planning, binding, and session lifecycle |
| `src/artifact/roundtrip_gate.c` | emitted artifact roundtrip gate |
| `src/model/materialization.c` | backend-owned weight-table materialization and cleanup |

## Runtime ownership map

| Owner | Boundary |
| --- | --- |
| `src/runtime/descriptor.c` | runtime-descriptor ABI, deterministic import, validation, lookup, and typed result projection |
| `src/runtime/core.c` | immutable model-engine generations and mutable execution-session lifecycle |
| `src/runtime/binding.c` | transactional, content-addressed runtime-binding serialization and admission |
| `src/runtime/specialization.c` | deployment implementation class, hardware capability, admitted execution envelope, and specialization identity |
| `src/runtime/residency.c` | typed package mappings, prepared/resident engine resources, accounting and model-lifetime sharing |
| `src/runtime/resource.c` | generation-bound resource catalog, dependencies, borrows, readiness, accounting, eviction and release |
| `src/runtime/scheduler.c` | one engine-owned ready-work and compatible-operation scheduling authority |
| `src/runtime/state_residency.c` | session persistent-state banks, CUDA paging, publication, rollback, reset and invalidation |
| `src/runtime/graph.c` | execution descriptors, phase/mode dispatch, reusable workspace, and transactional publication |
| `src/runtime/benchmark.c` | identity-bound runtime timing, baseline, CSV, and deterministic SVG serialization |
| `src/graph/execution.c` | Physical Execution IR v5 package/storage facts, deterministic identity and validation |
| `src/graph/device_view.c` | compact checked device-value view admission without model-policy reconstruction |
| `src/graph/worklist.c` | deterministic execution-batch sealing, expert-major worklist construction, validation, identity, and observation aggregation |
| `src/graph/candidate.c` | prefix-addressable attention candidate deltas and exact accepted-prefix projection |
| `src/graph/state.c` | committed/candidate persistent-state transactions, prefix promotion, rollback and state identity |
| `src/graph/state_pages.c` | stable virtual state spans, per-class host-page commitment, pool accounting and release lifecycle |
| `src/graph/state_recipe.c` | immutable family-neutral persistent-state recipe projection |
| `src/server/core.c` | zero-engine-capable persistent host, private listener, control dispatch and shutdown |
| `src/server/engine.c` | load, generation-bound routing, draining, unload, resource admission and engine inventory |
| `src/server/session.c` | exact conversation sessions, KV continuation, turns and partial state |
| `src/server/protocol.c` | bounded versioned local framing and thin protocol client |
| `src/server/telemetry.c` | one typed event sequence, subscribers and metrics accumulation |

## Model architecture ownership map

The three DeepSeek family files deliberately share one basename because they
project the same family at three different DAG levels. Their directories and
machine-readable owner rows disambiguate the responsibility. A common runtime
family directory would duplicate model/session/state authority and is
forbidden.

| Owner | Boundary |
| --- | --- |
| `src/model/families/deepseek_v4.c` | immutable architecture, exact source coverage, family Transformation IR construction, GGUF lowering, runtime-descriptor facts, and payload handoff for the admitted identity |
| `include/yvex/internal/families/deepseek_v4.h` | private typed DeepSeek recipe ABI shared by production consumers |
| `src/source/inventory.c` | retained immutable source tensor snapshot, deterministic identity, indexed lookup, one-header-pass and zero-payload-read accounting |
| `src/model/target/tensor_collection.c` | release-target collection projection from canonical coverage; Qwen/Gemma evidence remains separate |
| `src/model/target/missing_role.c` | release-target missing-role projection from canonical coverage |
| `src/model/target/mapping_gate.c` | operational projection of the canonical mapping plan and payload-streaming handoff |
| `src/model/target/report.c` | strict source-verification coordination and typed report projection for the canonical release target; Qwen/Gemma lexical evidence remains separate |
| `src/cli/render/model_target.c` | presentation of typed IR facts without architecture decisions |

## Graph and backend ownership map

| Owner | Boundary |
| --- | --- |
| `src/graph/plan.c` | runtime descriptor roles, immutable graph plan and backend admission facts |
| `src/graph/attention.c` | generic attention protocol, identity validation and transactional state boundary |
| `src/graph/numeric.c` | reusable attention and bounded tensor numerical operations without family policy |
| `src/graph/state.c` | immutable prior-state views, candidate deltas, and transactional attention-state lifecycle |
| `src/graph/families/deepseek_v4.c` | DeepSeek schedule, recurrence and CPU/CUDA operation composition |
| `src/backend/cuda/attention.c` | generic CUDA execution of a fully compiled encoded-attention job; no model topology or family authority |
| `src/backend/core.c` | backend lifecycle, tensor binding and canonical qtype compute projection |
| `src/backend/core.c` | backend lifecycle, tensor binding, canonical qtype compute projection, and typed device/context/bundle/memory reports |
| `src/backend/cuda/capability.c` | atomic generated-bundle admission, exact CUDA capability, launch/sync demotion, and cleanup failure |
| `src/backend/cuda/graph.c` | CUDA launch-graph registry, capture, instantiate, replay, update, invalidation, and release |
| `src/backend/cuda/ops.c` | validated host launch binding for admitted exact variants |
| `src/backend/cuda/kernels.cu` | canonical general device kernels; generated modules remain build output |
| `src/backend/cuda/transformer.c` | CUDA transformer primitive validation, launch, and numerical reporting |
| `src/backend/cuda/transformer_kernels.cu` | independently compiled CUDA transformer kernels under the same owner |
| `src/backend/cuda/moe_kernels.cu` | independently compiled routed/shared MoE kernel family |
| `src/backend/cuda/kernel_primitives.h` | toolchain-only qtype/device primitives shared by both CUDA kernel families |
| `src/backend/cuda/qtype.c` | CUDA qtype capability/refusal facts |

## Client and engineering ownership map

| Layer | Owner |
| --- | --- |
| Canonical operation and command metadata | `config/operator/registry.json` |
| Registry validation and deterministic C projection | `tools/generate_operator_registry.py` |
| Generated immutable descriptor data | `build/generated/operator/*` |
| Product entry, REPL and compact render | `src/cli/io/client.c` |
| Foreground server entry | `src/cli/io/server.c` |
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
- device-resident DSpark acceptance/correction, fused output-head sampling, or
  full-model qualification of target-only CUDA stochastic sampling
- evaluation or release-path full-model benchmark results
- a selected release artifact or release readiness

Attention-local prefill/decode phases operate on activation tensors and an
explicit state view. Runtime-local benchmark/profile output and deterministic
charts measure cold/warm latency plus identity-bound residency, transfer,
allocation, and graph-launch facts for that boundary only. Raw measurements,
generated charts, and machine-specific evidence remain external operator
assets.
