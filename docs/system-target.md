# YVEX System Target

Date: 2026-07-22
Status: filesystem and module ownership contract
Authority: filesystem and module topology; current project state belongs only
to `PROJECT.md`

This document records the target filesystem architecture for the execution-core
and control-plane boundary. It is an implementation contract, not a capability
claim.

## Current Actual Tree

Current core areas:

```text
src/cli/                 CLI dispatch, input, surfaces, render, IO
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

## Target Tree Summary

```text
input -> command -> surface router -> report/domain -> render -> cli/io

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

## Gap Table

| Area | Current state | Target state | Next row |
| --- | --- | --- | --- |
| GGUF artifact ABI | scalable file-backed v3 reader, typed refusal, target-scale budgets, immutable view, and zero payload reads | preserve the structural reader as input to canonical layout and later artifact admission | see `PROJECT.md` |
| GGUF qtype storage | pinned IDs 0-39, exact row geometry, and typed refusal are canonical | preserve storage independently from decoder, quantizer, emitter, and compute support | see `PROJECT.md` |
| GGUF layout integrity | canonical directory-order spans, power-of-two alignment, zero padding, truncation/tail refusal, and snapshot drift checks | preserve global layout as input to complete-model artifact admission | see `PROJECT.md` |
| CUDA truth | context, bundle, functions, and exact variants are typed; no-bundle builds refuse kernels and generated-bundle variants have GB10 reference proof | preserve fail-closed admission while later architecture/runtime rows define and prove the DeepSeek operation set | see `PROJECT.md` |
| Architecture IR | immutable typed DeepSeek-V4-Flash model, 43 main-layer, and MTP topology consumed by release-target profiling | preserve the source-to-IR boundary and feed complete tensor-role derivation without source reparsing | see `PROJECT.md` |
| Tensor coverage | the IR-derived 69,187-slot DeepSeek requirement set reconciles exactly against one retained verified header snapshot | preserve one-to-one source coverage as the admission input to GGUF mapping | see `PROJECT.md` |
| GGUF names/layout | immutable 69,187-contribution to 1,360-descriptor concrete DeepSeek GGUF lowering plan | preserve the format projection while moving artifact-neutral transformation semantics to the compilation owner | see `PROJECT.md` |
| Source payload | verified snapshot-bound shard/range indexes, payload trust, and bounded transactional streaming | preserve as artifact-neutral byte input to transformation execution without decoding or reinterpretation | see `PROJECT.md` |
| Transformation plan | sealed artifact-neutral IR binds all 69,187 source values to 1,360 terminal tensors and exact payload identity | preserve immutable transformation truth as the input to physical-profile decisions and execution | see `PROJECT.md` |
| Quantization | selected profile encodes all 1,360 terminals with canonical codecs, reference decoding, numeric evidence, and direct selected-qtype CPU/CUDA compute | preserve the profile and execution identities independently from artifact serialization | see `PROJECT.md` |
| GGUF writer | deterministic v3 plan and transactional file writer complete | preserve checked layout and atomic publication for admitted profiles | see `PROJECT.md` |
| Artifact emission | source-faithful and selected complete DeepSeek artifacts emitted outside the repository | preserve exact physical identities and complete metadata/tokenizer evidence | see `PROJECT.md` |
| GGUF roundtrip | both complete artifacts pass native full-byte and pinned official-reader admission; the selected artifact has deterministic second-serialization proof | preserve reader/writer equivalence and corruption refusal | see `PROJECT.md` |
| Materialization | family-neutral materialization and a full bounded selected-artifact walk are complete; the runtime owns an attention-only resident pack, not full-model device residency | preserve exact physical bindings while later owners add KV, MoE and output-head residency | see `PROJECT.md` |
| Runtime descriptor | immutable DeepSeek descriptor binds all 1,360 admitted tensors and topology facts and is consumed by the common attention runtime and persistent-state recipes | feed full-model prefill and complete transformer composition without rebuilding compiler truth | see `PROJECT.md` |
| Graph/backend | complete DeepSeek SWA/CSA/HCA attention is admitted on CPU eager and GB10 CUDA eager/piecewise/full modes over all 43 layers and 634 bindings; later executions consume committed session state | model prefill, MoE, and backend-bound complete transformer path | see `PROJECT.md` |
| Common runtime | content-addressed binding, immutable family-neutral model, session-owned persistent state/workspace, resident attention weights, phase-aware dispatch, and CPU/CUDA execution | preserve warm reuse and typed family adapters while full-model prefill becomes the next consumer | see `PROJECT.md` |

## Owner Rules

- CLI input owns `argc/argv` parsing only.
- CLI command adapters dispatch only.
- CLI surfaces route command families only.
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
  derivation identities. Future physical-variant planning may consume typed
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

## GGUF Target Map

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

## Artifact And Materialization Target Map

| Owner | Boundary |
| --- | --- |
| `src/artifact/descriptor.c` | YVEX artifact descriptor facts |
| `src/artifact/materialize.c` | admitted artifact materialization, range binding, and lifecycle |
| `src/artifact/roundtrip_gate.c` | emitted artifact roundtrip gate |

## Runtime Target Map

| Owner | Boundary |
| --- | --- |
| `src/runtime/descriptor.c` | runtime-descriptor ABI, deterministic import, validation, lookup, and typed result projection |
| `src/runtime/core.c` | immutable runtime-model and mutable execution-session lifecycle |
| `src/runtime/binding.c` | transactional, content-addressed runtime-binding serialization and admission |
| `src/runtime/residency.c` | read-only resident attention-weight packs and generation-bound invalidation |
| `src/runtime/graph.c` | execution descriptors, phase/mode dispatch, reusable workspace, and transactional publication |
| `src/runtime/benchmark.c` | identity-bound runtime timing, baseline, CSV, and deterministic SVG serialization |

## Model Architecture Target Map

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

## Graph And Backend Target Map

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

## CLI Target Map

| Layer | Owner |
| --- | --- |
| Entry | `src/cli/main.c` |
| Input | `src/cli/input/<surface>.c` |
| Command | `src/cli/commands/<surface>.c` |
| Family workflow | `src/cli/model_artifacts/<surface>.c` where required |
| Render | `src/cli/render/<surface>.c` for typed domain projections |
| Operator IO | `src/cli/io/*` |

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
tensor payload bytes. Complete-model artifact admission remains separate.

## GGUF Qtype ABI Boundary

The qtype owner admits the pinned GGUF on-disk range, accounts removed and
outside-baseline identities, and derives bytes from the complete shape with
`ne[0]` as row width. Dtype, range, integrity, conversion, and memory-plan
owners project these facts instead of copying geometry. This boundary does not
provide reference dequantization, quantization, emission, backend arithmetic,
artifact completion, or runtime support. Current milestone state belongs only
to `PROJECT.md`.

## Forbidden Claims

This target does not claim:

- persistent family-correct KV
- tokenizer-backed prompt prefill
- MoE or complete transformer execution
- model decode, logits, sampling, or generation
- evaluation or full-model benchmark results
- supported-model or release readiness

Attention-local prefill/decode phases operate on activation tensors and an
explicit state view. Runtime-local benchmark/profile output and deterministic
SVG charts measure cold/warm latency plus identity-bound residency, transfer,
allocation and graph-launch facts for that boundary only. The curated chart
set may be retained under `assets/benchmarks/attention/`; raw evidence remains
external.
