# YVEX API

This document maps the installed C ABI and the non-installed contracts used by
the YVEX operator binary. It describes ownership and lifetime; it does not turn
an internal runtime boundary into a public compatibility promise.

Runtime behavior is governed by [the runtime contract](contract.md). Project
state and decommission obligations in `PROJECT.md` remain authoritative.

## Header Tiers

External consumers may include the convenience umbrella:

```c
#include <yvex/api.h>
```

Production YVEX code includes the exact domain header it consumes. The umbrella
contains these twelve installed domain headers:

| Header | Stable domain |
| --- | --- |
| `<yvex/core.h>` | status, bounded errors, paths, logging, version identity |
| `<yvex/source.h>` | source provenance, accounts, manifests, native tensor inventory |
| `<yvex/gguf.h>` | bounded GGUF v3 parsing, metadata, tensor directory, layout facts |
| `<yvex/artifact.h>` | immutable file snapshots, identity, integrity and admission |
| `<yvex/model.h>` | dtypes, tensor roles, descriptors, materialized-weight views |
| `<yvex/qtype.h>` | canonical GGUF qtype identity and storage geometry |
| `<yvex/quant.h>` | quantization policy, job and calibration manifests |
| `<yvex/graph.h>` | generic graph, planning and memory-plan contracts |
| `<yvex/backend.h>` | backend admission, device tensors and primitive dispatch |
| `<yvex/tokenizer.h>` | tokenizer views, tokenization and prompt rendering |
| `<yvex/registry.h>` | local model registry and typed reference resolution |
| `<yvex/server.h>` | bounded HTTP parsing and server lifecycle |

Headers below `include/yvex/internal/` are non-installed cross-subsystem ABI.
They are available to repository production owners and focused tests only;
`<yvex/api.h>` never includes them. No source-local header is part of either
surface.

The common-runtime cutover intentionally retires the former installed
`runtime.h`, `generation.h`, and `metrics.h` diagnostic contracts. Those
headers exposed a bounded proof engine, flat F32 KV, fixture logits/sampling,
and report-only metrics; they were not a model-backed runtime ABI. Retaining
them would preserve a second lifecycle beside the sealed runtime model and
session. This is an incompatible pre-release ABI cutover, not a compatibility
alias: future KV, generation, and observability surfaces must be admitted by
their owning milestones over the common runtime.

All public headers are independently includable in C and C++. Public option
structures borrow pointer fields for the duration of a call. An opaque object
returned through an output pointer is caller-owned until its matching close or
release function runs.

## Status, Failure, And Publication

Public functions return `YVEX_OK` on success and a typed status otherwise.
Functions accepting `yvex_error *` write bounded copied context; the message
does not borrow parser, backend or stack storage.

Failure remains attached to its owning boundary. Parse refusal, artifact drift,
materialization failure, backend refusal, execution failure and cleanup failure
are not interchangeable. A renderer may project the code and context, but it
does not classify capability from error text.

Transactional owners publish only complete state. Artifact and manifest
writers use no-replace atomic publication. Graph and runtime execution produce
candidate output and state first, then commit them together. Cancellation or
failure leaves the previous committed state unchanged.

## Artifact And GGUF Ownership

`yvex_artifact_open` retains one read-only file handle and immutable snapshot
until `yvex_artifact_close`. With mapping disabled, callers use bounded
positioned reads; an optional read-only mapping is valid only for the handle
lifetime.

`yvex_gguf_open_ex` borrows an artifact during construction and then owns its
decoded metadata, names, arrays, tensor directory and indexes. Structural parse
does not read tensor payload bytes. `yvex_gguf_layout_validate` checks canonical
directory order, qtype-sized ranges, alignment, zero padding, total span and
snapshot stability without promoting the file to a complete artifact.

Qtype geometry comes only from `<yvex/qtype.h>`. Storage admission uses
`dims[0]` as row width and checks block divisibility and every multiplication.
Storage geometry, decoding, encoding, CPU compute, CUDA compute and runtime
support remain separate facts.

Complete-artifact admission under `<yvex/artifact.h>` binds physical structure,
required metadata, tokenizer evidence, tensor inventory and exact file
identity. The admitted file remains external operator data. A complete model
artifact is still not a supported generation artifact.

## Model, Materialization, And Backend

`<yvex/model.h>` exposes canonical dtype, tensor-role and model-descriptor
facts. Materialized-weight objects describe bounded backend-owned storage; their
presence does not imply complete model residency.

`<yvex/backend.h>` exposes backend discovery, capability facts, device tensor
lifecycle and admitted primitives. Backend code consumes typed operations and
does not infer family topology. `yvex_backend_close_checked` nulls its owner
only after complete discharge and retains it when cleanup must be retried;
`yvex_backend_close` remains the best-effort compatibility projection for
callers without a failure channel.

The generated CUDA bundle, Driver API module/function resolution and CUDA Graph
objects are repository-internal backend contracts. They are not installed C
ABI and they do not imply a model-generation path.

## Common Internal Runtime

The common runtime model and execution session are deliberately non-installed
contracts consumed through `<yvex/internal/runtime.h>` by the operator binary.
The internal runtime is family-neutral. Its main objects are:

| Object | Ownership |
| --- | --- |
| `yvex_runtime_binding` | immutable content-addressed bridge from an admitted artifact to runtime identities and executable requirements |
| `yvex_runtime_family_adapter` | typed family projection; DeepSeek is the first admitted adapter, not a separate runtime |
| `yvex_runtime_model` | immutable verified artifact handle, binding, imported descriptor/plan and read-only resident weights |
| `yvex_runtime_execution_session` | mutable backend context, reusable workspace, attention-local state, cancellation and CUDA Graph registry |
| execution descriptor | canonical pointer-free identity over phase, mode, scope, geometry, residency, workspace, state and device facts |

The binding is generated transactionally outside the repository, named by its
content identity and independently reopened. Runtime open validates it against
the exact admitted artifact. Runtime execution does not read source headers or
payloads and does not rebuild Transformation IR, quantization plans or GGUF
writer plans.

One runtime model performs one complete artifact hash and one GGUF directory
admission. Warm operations reuse the same verified handle, immutable descriptor,
attention graph and resident weight pack. Before and after execution, snapshot
drift invalidates the model, sessions, residency, workspace, graph executables
and candidate state.

Sessions own mutable state. Prepared steady-state execution performs no host or
device allocation, weight read, upload, workspace resize or graph capture. The
runtime refuses requests outside the prepared capacities instead of resizing a
captured execution implicitly.

### Internal DeepSeek Attention Operator Boundary

`yvex_graph_attention_operator_execute` is the non-installed typed adapter used
by `yvex graph attention ...`. It consumes a runtime binding, common runtime
model/session, admitted external artifact and canonical probe activation. It
never calls Make, a test executable, another process or the test-only oracle.

The operator distinguishes:

- attention `prefill`: a multi-token activation chunk with an immutable prior
  attention-state view;
- attention `decode`: one activation token with an immutable prior state view;
- mixed and speculative phases: represented but refused;
- attention core, attention envelope and complete release-attention-set scopes.

These phase names do not mean tokenizer-backed prompt prefill or model decode.
Persistent KV, embedding, MoE, transformer composition, logits, sampling and
generation remain outside this API.

CPU admits eager execution. CUDA admits eager, piecewise CUDA Graph and full
CUDA Graph execution plus an `auto` dispatcher. Explicit mode requests either
run that mode or refuse; only `auto` may select another admitted mode and must
report why. CUDA execution does not fall back to CPU numerical work.

The runtime returns four different identities:

| Field | Hashes |
| --- | --- |
| `tensor_output_digest` | canonical output tensor geometry and bytes |
| `state_delta_digest` | canonical candidate attention-state delta |
| `execution_evidence_digest` | backend/mode-specific stages, graph facts and counters |
| `execution_identity` | complete request/result compatibility contract |

CPU and CUDA expose separate exact output and state-delta digests. Equal bytes
produce a common digest; when exact bytes differ, the common field is
unavailable even if the versioned numerical comparison passes. The comparison
reports output/state value counts, finite and non-finite counts, first failing
stage and coordinate, maximum absolute/relative error, RMSE, and separate
byte-equality facts. Its state lane covers raw KV, compressed/indexer emissions
and positions, and both rolling-state components. Evidence digests remain
backend and execution-mode specific.

## Runtime Binding And Operator Actions

The main CLI provides the production consumer for the internal ABI:

```text
yvex graph attention prepare
yvex graph attention describe
yvex graph attention capabilities
yvex graph attention plan
yvex graph attention execute
yvex graph attention compare
yvex graph attention state inspect|validate|exercise
yvex graph attention residency inspect
yvex graph attention capture|replay
yvex graph attention cuda-graph list|inspect|warmup|update|invalidate|release
yvex graph attention trace|profile|benchmark|qualify
yvex graph attention benchmark compare
```

`prepare` is the compiler-side producer for an external runtime binding.
Execution actions require the binding and do not regenerate it. `plan` seals a
request descriptor without numerical dispatch. State and graph-registry actions
operate on real process-local session state rather than report-only labels.
Registry inspection reports captured kernel, copy and memset nodes plus capture,
instantiation, update and replay timings. It is not a persistent cross-process
graph cache.

The canonical probe preserves real model width, heads, bindings, qtypes,
position policy and attention history geometry. It is deterministic activation
input, not prompt text.

## Qualification, Benchmark, And Chart Contract

`qualify` executes the production runtime path and reports software-contract,
numerical-conformance, runtime-qualification and component-benchmark status as
separate facts. It also reports that model behavior evaluation, model quality
evaluation, agent runtime/evaluation and release qualification are unavailable.
The command does not invoke source-tree tests or link the test-only oracle.

`profile` and `benchmark` also use the production runtime path. Benchmark
records identify their scope as `attention_component`, require correctness and
runtime preconditions, and keep correctness, structural-runtime and performance
status independent. Samples separate cold preparation, first execution,
publication and cleanup from steady-state execution. Distributions report
minimum, mean, dispersion, median, p50, p90, p95, p99 and maximum together with
allocation, transfer, launch, residency and workspace counters.

`--write-baseline --baseline FILE` writes a versioned identity-bound external
baseline. A later compatible run or `benchmark compare --baseline OLD
--current NEW` may compare two records. The schema-five key binds the build,
artifact, materialization, logical model, runtime binding, numeric policy,
runtime and execution descriptors, semantic and executable graphs, residency,
workspace, state layout, kernel bundle, machine, device, phase, attention
class, scope, mode, geometry, capture bucket, trace policy and iteration count.
Compatibility comparison retains both commits as provenance while excluding
the commit alone from the workload-equivalence key. An incompatible comparison
refuses and identifies the first mismatched field.

`benchmark compare --max-regression-bps N` enables an explicit caller-owned
ceiling for latency, inverse throughput, memory, transfers, allocation counts,
and launch counts. No threshold is implied when the option is absent:
performance remains `measured`. The policy identity and comparison identity
bind the selected ceiling and result. A compatible comparison exits nonzero
when any measured dimension exceeds the ceiling; structural runtime failures
remain separate and cannot be converted into performance allowances.

`--chart PATH.svg` writes a deterministic SVG containing cold preparation,
warm latency distributions, resident/workspace bytes, resident H2D bytes, and
kernel/graph launch, capture, replay, and node counters, optionally against a
compatible baseline. The schema-five baseline seals those structural counters,
timings and complete reproducibility identity. Schemas one through four require
regeneration.
Baseline and SVG publication are independently atomic and
no-replace; an SVG failure never withdraws an already admitted baseline. JSON,
CSV, baseline and SVG outputs are external operator assets. They are never
tracked and they are not full-model benchmark or release evidence.

## Capability And Claim Boundary

The common runtime publishes granular facts for semantics, core/envelope,
CPU/CUDA phase and mode, residency, workspace, state delta, trace, profile and
benchmark readiness. Compatibility booleans may be derived from that lattice;
they are not independent capability authorities.

The current runtime supports production DeepSeek attention over admitted
weights. It does not provide persistent KV, tokenizer-backed prompt prefill,
MoE, a complete transformer, model decode, logits, sampling, text generation,
evaluation, a full-model benchmark or release readiness.

## Extension Rules

New installed declarations require a stable external lifecycle and tests.
Internal implementation convenience is not a public ABI reason. A future model
family registers typed facts and sequence-mixer lowering against the common
runtime; it does not receive its own model/session implementation.

Every API extension must define:

1. owner and header tier;
2. borrowed and owned inputs;
3. success publication and failure rollback;
4. identity and invalidation dependencies;
5. cleanup behavior;
6. focused positive, refusal and lifecycle tests;
7. the exact capability boundary it does and does not promote.
