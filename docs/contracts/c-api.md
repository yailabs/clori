# YVEX C API

Status: installed and internal interface reference

Authority: installed headers under `include/yvex/` and non-installed headers
under `include/yvex/internal/`. This document projects their ownership,
lifetime, inputs, outputs, side effects, failure, and compatibility policy;
the headers remain exact ABI authority.

This document maps the installed C ABI and the non-installed contracts used by
the engine-linked foreground server and finite engineering operations. It describes
ownership and lifetime; it does not turn an internal runtime boundary into a public
compatibility promise.

Runtime behavior is governed by [the hosted runtime contract](runtime.md).
Project state remains authoritative only in [`ROADMAP.md`](../../ROADMAP.md).

## Header Tiers

External consumers may include the convenience umbrella:

```c
#include <yvex/api.h>
```

Production YVEX code includes the exact domain header it consumes. The umbrella
contains these fifteen installed domain headers:

| Header | Stable domain |
| --- | --- |
| `<yvex/core.h>` | status, bounded errors, paths, logging, version identity |
| `<yvex/source.h>` | source provenance, accounts, manifests, native tensor inventory |
| `<yvex/gguf.h>` | bounded GGUF v3 parsing, metadata, tensor directory, layout facts |
| `<yvex/artifact.h>` | immutable file snapshots, identity, integrity and admission |
| `<yvex/model.h>` | artifact-neutral dtypes, tensor roles and model descriptors |
| `<yvex/catalog.h>` | remote-provider, acquired-source and local-package catalog records |
| `<yvex/materialization.h>` | backend-owned materialized-weight lifecycle and views |
| `<yvex/qtype.h>` | canonical GGUF qtype identity and storage geometry |
| `<yvex/quant.h>` | quantization policy, job and calibration manifests |
| `<yvex/graph.h>` | generic graph, planning and memory-plan contracts |
| `<yvex/backend.h>` | backend admission, device tensors and primitive dispatch |
| `<yvex/provider.h>` | transport-neutral application request and result semantics |
| `<yvex/tokenizer.h>` | tokenizer views, tokenization and prompt rendering |
| `<yvex/registry.h>` | local model registry and typed reference resolution |
| `<yvex/server.h>` | local protocol, runtime host, sessions, telemetry and thin client lifecycle |

Headers below `include/yvex/internal/` are non-installed cross-subsystem ABI.
They are available to repository production owners and focused tests only;
`<yvex/api.h>` never includes them. No source-local header is part of either
surface.

The common-runtime cutover intentionally retired the former installed
`runtime.h`, `generation.h`, and `metrics.h` diagnostic contracts. Those
headers exposed a bounded proof engine, flat F32 KV, fixture logits/sampling,
and report-only metrics; they were not a model-backed runtime ABI. Retaining
them would preserve a second lifecycle beside the sealed runtime model and
session. Current KV, generation, and observability contracts are admitted
through explicit internal owners and the installed server/protocol boundary
rather than compatibility headers.

All public headers are independently includable in C and C++. Public option
structures borrow pointer fields for the duration of a call. An opaque object
returned through an output pointer is caller-owned until its matching close or
release function runs.

## Installed ABI Versioning

A versioned installed record is identified by its C type and schema value.
One such pair names one field layout and semantic contract. A binary layout or
incompatible semantic change advances that record's schema; it does not advance
the private local protocol unless the wire contract also changes. Internal
records rebuilt with the binary do not acquire public schema versions merely
because their implementation changes.

`yvex_server_options` schema v4 is the current host-options contract. It adds
`maximum_engines` between worker capacity and listener policy, preserving the
separate implementation ceiling, configured engine capacity, and actual
resource admission. The public v3 constant remains as historical source
identity, but `yvex_server_create` refuses v3 before reading any field absent
from its legacy layout. Repository callers use
`YVEX_SERVER_OPTIONS_SCHEMA_CURRENT`; there is no binary v3 reinterpretation.

`yvex_tokenizer_plan_summary` schema v4 identifies prompt composition with
family-neutral `conversation` and `verbatim` values. Pretokenizer behavior is
selected independently from the admitted tokenizer metadata. Schema v3 remains
the historical family-named semantic identity; newly sealed plans publish v4.

The installed catalog split introduced unversioned pre-v0.1 source/API
migrations rather than assigning an existing schema identity to new layouts:
remote records, local source records, local package records, and live engine
observations now have distinct owners. Graph and materialization declarations
moved to their installed domain headers without changing their layouts. The
public ABI guard binds every explicitly versioned installed record to a
comment-insensitive declaration signature and compiler-checked 64-bit C/C++
layout facts. Changing one requires an explicit schema and migration decision.

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
identity. The admitted file remains external operator data. A complete
artifact is still not a supported artifact.

## Model Registry And Startup Profiles

`<yvex/registry.h>` owns the local model catalog and typed reference
resolution. Registry schema `yvex.models.local.v6` binds a catalog entry to one
typed startup profile. Every profile records engine kind independently from
execution strategy. `single-artifact` text profiles carry the absolute
artifact, exact runtime-binding path, runtime target, backend, semantic
`target-only` or `speculative` strategy, and positive context capacity.
`composite` media profiles carry an installed component root, target and
backend with a `not-applicable` execution strategy; they do not manufacture a
singular artifact or runtime binding. Older v1 through v5 catalogs remain
readable. The importer maps the former `target-only`, `dspark`, and `media`
mode values onto the two current axes; current writers never emit the mixed
legacy mode.

The installed in-process `yvex_model_registry_entry` contract is explicitly
versioned at schema v1. Callers set `schema_version` to
`YVEX_MODEL_REGISTRY_ENTRY_SCHEMA_CURRENT`; mutation and startup validation
reject any other value before reading the remaining fields. The former
unversioned layout is not a binary compatibility surface. Model-library
projections expose the separately versioned
`yvex_model_runtime_profile_fact` v1 record.

`yvex_model_registry_startup_validate` checks the facts required by the profile
kind and the corresponding local file or installation accessibility. It does
not authenticate identities, materialize weights, initialize a backend, or
establish runtime support. After the persistent host is running,
the engine-load operation performs full singular or composite admission and
publishes one new engine generation. The TTY CLI selects a profile
interactively; API and noninteractive clients always supply its exact identity.

A composite media target follows the same readiness rule. Before publishing
`READY`, the server opens the tokenizer and every component artifact, reconciles
their typed admissions, and retains those immutable views under one runtime-model
identity. Payload materialization and CUDA residency remain separate facts: a
component set larger than the device's memory envelope is staged at execution
phase boundaries rather than being reported as simultaneously resident.

The default catalog is user-local data at
`~/.local/share/yvex/models.local.json`; an explicit `YVEX_DATA_DIR` changes
that owner for controlled deployments. Catalog entry, invocation-selected
profile, and live runtime model are three distinct facts.

## Model, Materialization, And Backend

`<yvex/model.h>` exposes canonical dtype, tensor-role and model-descriptor
facts. `<yvex/materialization.h>` separately owns materialized-weight objects
that describe bounded backend-owned storage; their presence does not imply
complete runtime-model residency.

`<yvex/catalog.h>` keeps provider records, acquired local sources, and admitted
local packages as separate record types. A CLI or future GUI may join those
records with live engine observations, but no catalog record acquires server or
engine authority.

`<yvex/backend.h>` exposes backend discovery, capability facts, device tensor
lifecycle and admitted primitives. Backend code consumes typed operations and
does not infer family topology. `yvex_backend_close_checked` nulls its owner
only after complete discharge and retains it when cleanup must be retried;
`yvex_backend_close` remains the best-effort compatibility projection for
callers without a failure channel.

The concrete backend object and dispatch table are source-local backend ABI.
Graph and runtime owners hold an opaque `yvex_backend` and use typed operations
for allocation, transfer, capability queries, residency mappings and workspace
access. In particular, session-state publication advances the backend-visible
residency generation through a checked operation; callers cannot mutate backend
dispatch, placement or generation fields directly.

The generated CUDA bundle, Driver API module/function resolution and CUDA Graph
objects are repository-internal backend contracts. They are not installed C
ABI and they do not imply a model-generation path.

Kernel-bundle identity v3 binds the selected image class, target architecture,
ordered module count, and exact bytes of every manifest-owned CUDA module. An
`sm_121` build embeds both portable PTX and native CUBIN for each independently
compiled kernel family; capability 12.1 selects the complete CUBIN set, while
other devices retain the complete explicit PTX set. Admission loads every
module and resolves every required function before publishing any capability.
The existing non-installed CUDA summary carries the selected image class,
architecture and aggregate identity. It does not add installed API.

## Common Internal Runtime

The model engine and execution session are deliberately non-installed
contracts consumed through `<yvex/internal/runtime.h>` by the operator binary.
The internal runtime is family-neutral. Its main objects are:

| Object | Ownership |
| --- | --- |
| `yvex_runtime_binding` | immutable content-addressed package bridge from an admitted artifact to compiled model/operator truth |
| `yvex_family_compiler_adapter` | compilation-only family projection that seals policy into the binding and never enters runtime model state |
| `yvex_model_engine` | one opened package generation with mappings, imported plans, backend specializations, engine resources, scheduler, and attached sessions |
| engine specialization | process-local mapping from PEIR package decisions to implementation classes admitted by one backend/device |
| `yvex_runtime_execution_session` | mutable backend context, reusable workspace, committed target state, bounded speculative candidate state, cancellation and CUDA Graph registry |
| execution profile | generation-bound selection of one engine specialization, workload, kernel bundle, mode, evidence class, and typed operation resolutions |

Model-execution descriptor schema v1 is a non-installed fieldwise projection
of source/family context, attention, MoE, output, DSpark and state facts.
Runtime binding v15 persists and authenticates it together with the canonical
operator graph identity, Physical Execution IR v5 package decisions, compiled
model plan, and pointer-free tokenizer/conversation policy. The explicit v14
reader authenticates legacy bytes and normalizes only canonical package records
to PEIR v5; an unsupported legacy derived-layout requirement refuses. Bindings
v7 through v13 remain explicit rebuild boundaries.
Hardware-profile,
workload-profile, capacity-plan and phase-roofline schemas begin at v1 as
internal contracts. Server options schema v4 owns host/listener policy
independently from `yvex_server_engine_options`; engine schema v2 separates
engine kind from text execution strategy while retaining alias, package,
backend, capacity, memory, and generation facts. Engine schema v1 is refused
before the added fields are read.
The source-authored conversation boundary admits provider request/wire schema
v3, tokenizer plan v3, tokenizer provider result v2, and local protocol v17.
Runtime event schema v5 and generation plan/result schema v5 remain current.
Generation plan ABI v5 adds the workload-profile identity
required to bind phase evidence to the compiled workload. Generation result
schema v5 adds the identity-bearing committed-token extent of a
source-output-channel boundary; the target-only continuation extent is derived
from the final committed extent. Runtime event schema v5 projects those facts,
engine lifecycle, rolling committed progress and aggregate telemetry pressure
without serializing the C result layout.

Phase-roofline v1 accepts both its original complete record and an additive
availability mask. A zero mask retains the original all-facts meaning; new
writers mark measured facts explicitly. The ledger retains all seven phase
slots while reporting measured, missing and rooflined masks, so absence is
never projected as a zero measurement. This does not earn a schema bump: the
contract is non-persisted, non-wire, rebuilt with every binary, and has no old
binary reader. Duration and work remain mandatory for every admitted record.
The graph execution owner also accumulates repeated phase deltas with checked
arithmetic and refuses any availability change within one phase; generation
does not keep a second accumulation policy.

The internal generation result carries an optional phase ledger. The result
layout is rebuilt with every product binary, is not serialized as C object
memory, and remains excluded from semantic generation identity. Generation
plan ABI v5 binds the ledger's workload-profile identity instead of comparing
it with the distinct per-request profiling identity. Result validation checks
that like-for-like identity and the availability masks when a ledger is
present; old internal results with no ledger retain their zero-initialized
meaning. The phase-ledger change did not alter the wire or event schema.

CUDA producers currently expose exact active-weight and launch facts for
target-only prefill and decode together with their exact H2D/D2H/D2D movement
and synchronization facts. Output projection exposes exact active-weight,
activation, temporary, launch, movement and synchronization facts, including
its bounded CUDA status transfer without treating that transfer as full-
vocabulary D2H. Compatible width-N rows contribute one aggregate output-head
execution instead of multiplying that work into each logical row. Draft and
verification sweeps merge that aggregate once with their transformer facts.
Their transformer active weight, state, activation, temporary and occupancy
stay explicitly unavailable; a partial record is not a complete roofline.

For production evidence, CUDA attention publishes core and envelope activations
through the existing caller-owned device output and does not materialize a
duplicate host row. The publication remains semantically identity-bound;
audit and forensic evidence continue to carry numerical host rows and their
checksums. Persistent attention state still follows the existing host-visible
candidate and residency transaction, so this change neither claims nor changes
the state-provider ABI.

The non-installed CUDA graph execution ABI accepts explicit timing,
session-stream and deferred-completion flags. Production attention pieces
borrow the session stream and return before a graph-local wait; their existing
layer publication barrier owns completion. Audit timing uses an isolated stream
and completes immediately. Launch-graph v3 and graph-executable v2 identities
bind that stream policy. No installed declaration, persisted model contract,
protocol or compiled-profile schema changed.

Accepted-prefix selection returns a v1 in-process physical-facts record. It
derives CUDA H2D and synchronization deltas from the session-owned state
residency counters and states the zero D2H, D2D and kernel facts explicitly.
This changed one internal source signature rebuilt with the product; it did not
change the state-provider ABI, generation ABI, protocol or persisted state.
Repeated phase occupancy, once supplied, is a checked work-unit-weighted mean;
failed accumulation leaves the prior measurement unchanged.

The binding is generated transactionally outside the repository, named by its
content identity and independently reopened. Runtime open validates it against
the exact admitted artifact. Runtime execution does not read source headers or
payloads and does not rebuild Transformation IR, quantization plans or GGUF
writer plans.

A descriptor-bearing binding is emitted as v8 beside existing v7 data. Old v7
writers cannot represent model geometry; v7 readers reject v8 as expected.
The rebuilt reader admits both versions and refuses malformed, truncated or
identity-mismatched descriptor payloads.

One first runtime-model admission performs a complete artifact hash and one
GGUF directory admission. A later open of the exact local filesystem snapshot
may consume a rebuildable verified-reopen lease and records zero payload bytes
hashed for that open; an absent, malformed, or stale lease falls back to the
complete hash. Residency schema v7 selects an admitted immutable backing.
Descriptor tensors retain exact artifact offsets. When the compiled physical
plan requires no derived asset and CUDA can register the immutable artifact
mapping, that mapping is the production weight backing. Readiness consumes the
returned device address without requiring a whole-artifact prefetch. Plans
requiring derived layouts select managed CUDA residency and complete their
prefetch before readiness. Neither alternative may be substituted silently.
Warm operations reuse the same verified handle,
immutable descriptor, attention graph and weight backing. Before and after
execution, snapshot drift invalidates the model, sessions, residency, workspace,
graph executables and candidate state.

Sessions own mutable state. `yvex_runtime_session_prepare_persistent_state`
seals the provider layout and CPU/CUDA residency for an exact capacity;
`yvex_runtime_session_reset_persistent_state` clears committed content while
retaining compatible allocation. The graph-state ABI provides checked
begin/stage/commit/abort, committed/candidate views, summaries, invalidation,
and release. These declarations are internal C contracts, not installed ABI.

Prepared steady-state execution performs no host or device allocation, weight
read, upload, workspace resize or graph capture. The runtime refuses requests
outside the prepared capacities instead of resizing a captured execution
implicitly.

### Internal Activation-Prefill Boundary

`include/yvex/internal/runtime_prefill.h` owns the non-installed production
contract for activation-driven attention prefill. Schema v1 binds the logical
model, runtime numeric, runtime descriptor, attention plan, operation scope,
token range, all 43 ordered layer identities, exact widths and strides,
canonical little-endian F32 payload ranges, payload digest, and input identity.
It serializes fields explicitly and never hashes native structures, pointers,
paths, padding, or timestamps.

`yvex_runtime_activation_input_open_memory` and
`yvex_runtime_activation_input_open_file` project the same immutable facts.
The file adapter retains a read-only regular-file handle and bounded mapping,
rejects symlinks, duplicate/missing/reordered layers, invalid dimensions,
overlap, truncation, trailing bytes, digest mismatch, non-finite payload,
resource overflow, and file drift. Close is idempotent.

`yvex_runtime_activation_prefill_execute` admits position and capacity before
mutation, divides the activation range into deterministic chunks, and invokes
the shared production attention executor for every layer. Each successful
chunk commits one complete persistent-state generation and advances position
once. Failure or cancellation aborts the failing chunk while preserving the
exact earlier committed prefix. CPU and CUDA eager consume the same activation
contract and session-owned provider; CUDA never falls back to CPU.

The result publishes activation-input identity, chunk/layer/class counts,
attention-output digest, persistent-state digest, committed prefix, position
and generation transitions, and execution identity. It is not a complete
transformer hidden state. Prompt text, tokenization, embedding, FFN/MoE,
cross-layer transformer composition, model decode, and generation remain
outside this API.

### Internal MoE Execution Boundary

`include/yvex/internal/moe.h` owns the non-installed MoE plan, typed input,
generic graph/backend execution packets, inspect context, and operator result.
The plan imports immutable runtime descriptor and materialization facts for all
layers; family policy enters through the explicit compiler-facing graph adapter
and is sealed into the pointer-free plan before runtime model-open. Generic MoE
compilation does not discover concrete families through a global registry.

`yvex_moe_input_open_memory` and `yvex_moe_input_open_file` admit the same
schema-v1 identity chain. The bounded file form stores explicit little-endian
header and layer records followed by finite F32 activations and numeric U32
token IDs. Token IDs are routing input only. The adapter rejects unsafe files,
stale identities, malformed layer order or geometry, invalid ranges, payload
digest mismatch, non-finite values, drift, and resource overflow.

`yvex_runtime_moe_context_open` seals reusable session resources and returns
the exact CUDA workspace requirement derived from the admitted layer geometry
and maximum row width. `yvex_runtime_moe_execute_layer` is the retained
token-local CPU/CUDA oracle; it accepts one expanded hidden activation and
returns distinct router, routed, shared, combined, and deferred mHC post facts.
`yvex_runtime_moe_rows` is the ordered width-N execute/complete contract.
Full-stack production CUDA routes the complete row set, constructs one
deterministic expert-major pair order, executes grouped routed and shared paths,
and defers publication across its layers. It transfers only bounded status and
unique-expert facts, validates them once at stack completion, and derives exact
active bytes without materializing selected routes. Portable, audit, forensic
and standalone block execution retain a completion-safe immediate oracle.
`yvex_runtime_moe_execute` executes an ordered all-layer input and publishes
only after the complete request succeeds. Reset and close preserve session
isolation and never advance persistent KV or sequence position.

The CPU and CUDA backends consume `yvex_moe_layer_job`. The retained oracle
executes exact selected routed-expert subviews plus the separate shared expert;
the production CUDA row operation consumes resident complete expert packs
through a private capability table. CUDA performs all numerical stages on
device and has no CPU fallback. `yvex_moe_operator_result` is a copied result
surface; it is not capability authority and is not a transformer or generation
result.

### Internal Transformer Execution Boundary

`include/yvex/internal/transformer.h` owns the non-installed transformer plan,
canonical numeric token input, reusable execution context, single-block
completion, full-stack coordination, and copied operator result. Schema-v1
plans bind exact runtime, attention, MoE, embedding, mHC, and final-norm facts;
schema-v1 inputs bind canonical U32 token IDs and their model/runtime/plan
identities. Memory and bounded-file admission share one validation contract.

`yvex_runtime_transformer_execute_block` consumes the attention publication
already staged inside an active request transaction, executes the admitted MoE
layer and deferred FFN mHC post, and returns field-wise routing/output/execution
identities. The full-stack coordinator alone selects deferred device-native MoE;
a standalone block remains complete before it returns.
`yvex_runtime_transformer_execute` owns chunk planning, selected-row embedding,
43 ordered blocks, final mHC collapse, final RMSNorm, atomic state commit, and
normalized-hidden publication. Its request carries an explicit
prefill or decode phase; one-token geometry alone never selects decode
semantics. The repeated-decode owner reuses this exact component.

CPU and CUDA share the typed contracts. The GB10 CUDA path retains inter-layer
activations on device and has no CPU numerical fallback. The API publishes
normalized hidden state only; tokenizer text, output-head projection, logits,
sampling, and generation remain outside it.

### Internal Repeated Decode Boundary

`include/yvex/internal/decode.h` owns the non-installed teacher-forced decode
coordinator. It borrows one already-open transformer context and its exact
execution session; it does not reopen the artifact or binding, rebuild plans,
or allocate a second KV owner. Inputs are bounded one-token views of the
existing schema-v1 transformer token input.

`yvex_runtime_decode_step` validates the authoritative committed position,
executes the production transformer in explicit decode phase, and publishes
one `[1,4096]` normalized hidden row only after that token's KV commit.
`yvex_runtime_decode_execute` repeats the same operation over ordered external
token IDs. Each successful step is independently durable; a later refusal
returns a non-success status with the completed step directory, final committed
prefix, generation, and first incomplete ordinal intact.

Step and aggregate identities serialize typed fields individually, including
the phase-bearing transformer identity, token, position, generation, routing,
hidden, persistent state, and structural counters. They exclude pointers,
padding, native object layout, timing, and local paths. The coordinator owns no
token-choice, logits, sampling, tokenizer, or generation policy.

### Internal Vocabulary-Logits Boundary

`include/yvex/internal/logits.h` owns the non-installed, family-neutral output-
head plan, authenticated normalized-hidden source, reusable logits context,
single-row projection, ordered repeated projection, typed results, and operator
adapter. The context borrows one runtime model/session and transformer plan. It
shares immutable output-head residency but owns mutable host/device logits
workspace and concurrency exclusion.

`yvex_runtime_logits_source_from_transformer` and
`yvex_runtime_logits_source_from_decode` seal only producer-authenticated final-
prefill or decode hidden rows. `yvex_runtime_logits_project` computes every
vocabulary coordinate directly from the resident encoded output head and
publishes the caller-owned row only after complete success.
`yvex_runtime_logits_execute` preserves earlier complete rows on a later
failure and records the exact first incomplete row. On CUDA it groups a
compatible width-N directory when host rows form one bounded batch or device
rows form one contiguous identity-compatible view. The group performs one
activation preparation, one encoded-head execution and one ordered output
transfer; its result owns the aggregate physical facts. Mixed, non-contiguous
or invalid sources take the explicit row-local path, preserving the same
complete-row publication and failure contract.

The logits API publishes raw F32 values and field-wise plan, source, residency,
backend, row, and aggregate identities. It neither repeats final norm nor owns
persistent state, sampling, tokenizer, or generation policy.

### Internal Real-Logits Sampling Boundary

`include/yvex/internal/sampling.h` owns the non-installed family-neutral
sampling policy, complete-logits source, reusable fixed-workspace context,
single-row selection, ordered repeated selection, typed results, and operator
adapter. The context copies the immutable output-head plan identity and owns
only candidate/probability workspace, private versioned RNG state, counters,
and concurrency exclusion. It borrows complete caller-owned logits and has no
model, session, artifact, KV, transformer, decode, or tokenizer ownership.

`yvex_runtime_sampling_source_from_logits` revalidates a completed logits-row
identity, full vocabulary extent, canonical raw digest, finite values, source
phase and position, hidden digest, and output-head plan. Greedy selection scans
the complete row and resolves exact ties to the lowest token ID without RNG.
Stochastic selection uses the schema-v1 API with filter-order v2: compensated
normalization removes exact zero mass before entropy-bearing filters, then one
PCG-XSH-RR 64/32 transition commits only after complete token/evidence
publication. Result validation authenticates every authoritative evidence
field. Atomic close admission drains active use before workspace release.
`yvex_runtime_sampling_execute` preserves completed earlier rows and the exact
committed RNG state when a later row refuses or is cancelled.

Sampling result identities bind the source, policy, ordered survivor IDs,
selected token, canonical probability, and applicable RNG states field by
field. The API does not mutate logits or persistent state and does not append,
decode, tokenize, stop, detokenize, or generate.

### Internal DeepSeek Attention Operator Boundary

`yvex_graph_attention_operator_execute` is the non-installed typed adapter used
by the offline `yvex bench attention ...` lane. Inspection and profiling of
the same owner use the `yvex inspect attention ...` and
`yvex bench attention ...` projections. The adapter consumes a runtime
binding, common runtime model/session, admitted external artifact, and either a
canonical diagnostic probe or admitted tensor-file activation input. It never
calls Make, a test executable, another process or the test-only oracle.

The operator distinguishes:

- attention `prefill`: a multi-token activation chunk with an immutable prior
  attention-state view;
- attention `decode`: one activation token with an immutable prior state view;
- mixed and speculative phases: represented but refused;
- attention core, attention envelope and complete release-attention-set scopes.

These phase names do not mean tokenizer-backed prompt prefill or model decode.
The same operator may consume and publish the session-owned persistent
attention-state provider. Embedding, full-model prefill, MoE, transformer
composition, logits, sampling and generation remain outside this API.

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

## Artifact-Bound Tokenizer Runtime

`<yvex/tokenizer.h>` exposes the exact admitted tokenizer plan, explicit-length
UTF-8 encoding, bounded source-authored conversation rendering, batch and incremental
ByteLevel decoding, special/EOS classification, and a generation-local token
append directory. Immutable vocabulary/merge/added-token indexes follow model
lifetime; incremental decoder and append contexts are isolated mutable owners.
Every owned result publishes only after complete success and carries field-wise
identities. The compiler-facing `<yvex/internal/tokenizer.h>` owns the canonical
pointer-free policy and codec; runtime instantiation never searches a family
registry. These operations do not read weights, mutate KV, append sampled tokens
to decode, or compose generation.

## Compiled Operator Registry Boundary

The operator registry is a build-time, non-installed source contract rather
than public C ABI. Strict `yvex.operator.registry.v1` JSON is validated and
projected into immutable generated C descriptors compiled into `yvex`. The
descriptors contain stable operation IDs, typed adapter enums, syntax,
visibility, requirements, and transport projections. They contain no domain
callbacks, arithmetic, resources, or protocol codecs.

The generated registry header is consumed only by product command owners; it
is not included by `<yvex/api.h>` or packaged as a mutable runtime dependency.
Domain APIs retain semantic validation and lifecycle. Runtime-client adapter
objects remain protocol-only, while finite offline adapters may consume the
non-installed engine interfaces already documented here.

## Application Provider And Local Protocol v17

`<yvex/provider.h>` is the installed transport-neutral application request and
result ABI. Provider schema v3 additionally represents an omitted completion
limit as adaptive while binding separate assistant reasoning content,
reasoning policy, source-authored drop behavior, field-presence facts, and a
bounded ordered tool-call set in addition to the v1 request facts. Provider wire
v3 carries those fields and the adaptive limit. V1 remains readable and writable only with disabled
reasoning, at most one assistant tool call, and its original field semantics.
Clone and wire-decode publish only a complete owned request graph. The provider
owner neither parses HTTP nor renders model-family prompt syntax.

`<yvex/server.h>` protocol v17 carries the sealed provider request through the
private Unix socket. Provider output messages distinguish assistant text,
explicit reasoning, function calls, usage, terminal completion, and failure.
Typed events bind the provider adapter, provider-request identity, and external
correlation ID while excluding prompt and output content.

Protocol v17 carries host status/stop, engine load/list/unload, exact
alias/generation routing, separate engine kind and semantic execution strategy,
speculative lifecycle events,
accepted-prefix facts, exact proposal/verification/commit accounting, turn
timing and cancellation classes, an exact partial-turn schema, source-authored
reasoning policy, typed reasoning/final/tool/error channels, and separate
reasoning/final count, rate and first-token timing facts. It also carries the
explicit source session, child name, and shared-prefix byte bound required for
one transactional copy-on-write session fork. It
retains the typed `console.status` and the removal of former
model/artifact facades introduced by the preceding protocol. Facts that are
not authoritative, including selected client configuration, active
micro-phase, or KV byte use when unavailable, have explicit availability bits
and are never fabricated. Version 8 introduced immutable committed
model-state checkpoint save/restore operations with an explicit file bound and
typed digest/identity evidence. Version 9 adds the startup capacity-plan
identity, required and unreserved bytes, admitted concurrent sequences, and
separate independent-session-scheduling and continuous-batching readiness.
Provider v3's adaptive limit and generation-bound routing are not executable by
an older peer, so every non-v17 frame refuses during the handshake;
there is no private pre-v0.1 compatibility decoder.

Version 12 added the typed terminal media result. Version 13 separates host
lifetime from engine lifetime and adds typed engine summaries plus exact
generation routing. A successful media turn binds
its absolute publication path, geometry, frame/audio facts, seed, byte extent,
hosted-preset identity, execution identity, file identity, and publication
identity without reclassifying runtime control text as model output. Media
progress remains server-authored telemetry. Text-generation request and result
semantics are unchanged.

Version 14 added typed first/last image conditions to hosted media requests.
Version 15 removes the mixed public generation-mode axis while retaining those
conditions. Engine records identify text versus media independently from
target-only versus speculative text execution. DSpark remains a DeepSeek
implementation of the semantic speculative strategy below the server boundary.
The incompatible engine and event layouts advance to schemas v2 and v4; v14
frames and legacy record schemas fail closed rather than being reinterpreted.
Version 16 added typed image-conditioning and media execution selection without
changing text routing. Version 17 adds engine load/unload lifecycle kinds,
runtime-event schema v5, and explicit versus server-resolved completion-envelope
facts. The event identity binds those new kinds and v16 frames fail closed.

Protocol error messages carry `yvex_client_failure_class`, so adapters map
queue capacity, timeout, incompatible state and unsupported input without
inspecting diagnostic text. `yvex_client_timeout_set()` bounds application
adapter send/receive waiting; zero restores unbounded post-handshake waiting
for native watch and trace consumers.

The source-separated OpenAI adapter inside the foreground server consumes only the provider
contract, protocol client, and bounded HTTP/JSON/SSE owners. It opens no second
artifact or model, owns no KV, and cannot call Transformer, generation, or CUDA
owners directly. The exact HTTP profile is documented in
[`openai-compatibility.md`](../openai-compatibility.md).

## Physical Execution And Candidate-State ABI

`<yvex/internal/execution.h>` owns Physical Execution IR schema v5 package
decisions, hardware/workload/capacity records, physical evidence records, and
device-value views. PEIR binds canonical terminal role, qtype, row geometry,
encoded range, consumer, stable layout, and sharing; it contains no selected
backend, activation, kernel family, request width, or live resource fact.

Runtime specialization maps each PEIR decision to a typed implementation
record for one backend/device. It owns activation representation, admitted real
widths, equivalent fallback class, and hardware crossover. The non-persisted
execution profile binds that specialization to one engine generation and
workload. The removed execution-shape registry is not a compatibility surface;
transient compatibility keys carry only the facts needed to form real batches.

`<yvex/internal/execution_batch.h>` owns the typed execution-batch and expert-
worklist contracts. The specialization contains only deployment-stable
implementation facts; one runtime instance binds actual sources, rows,
provenance, expert buckets,
offsets, populations and route weights. A bucket contains rows for exactly one
compatible expert. CUDA may select an equivalent microkernel and execute a
bounded tail, but it cannot regroup routes, merge sessions, or fabricate width.
The copied observation is pointer-free developer evidence, not execution authority.

`<yvex/internal/candidate.h>` owns prefix projection from an attention
candidate delta. It can reconstruct any admitted verified prefix without
rerunning accepted target rows. The logical state provider and backend
residency still publish the same generation atomically; candidate projection is
not a second persistent-state authority.

## Internal Generation And Hosted Turn Boundary

`include/yvex/internal/generation.h` owns the family-neutral generation plan,
prompt admission, exact suffix prefill, target-only and speculative execution,
accepted-prefix transaction, stop reasons, incremental committed-text
publication, partial progress, and result validation. Its implementation is
split among the admitted runtime generation, session, speculation, context,
and result owners rather than exposing another ABI. It borrows one admitted
model engine and one execution session; it does not reopen artifacts or
duplicate tokenizer, Transformer, logits, sampling, or KV semantics.

The speculative boundary consumes a family-projected draft plan and generic
proposal/verification contracts. Candidate tokens and draft RNG state remain
private until full-target verification determines the accepted result. Model,
token-ledger, incremental-decoder, text, and RNG participants prepare and
publish one accepted prefix together; rejection and cancellation abort every
uncommitted participant. Existing generated-token and completion-usage counts
remain committed-target counts.

`<yvex/server.h>` exposes the local protocol, persistent host, engine manager,
engine generations, server session, typed event, metrics snapshot, and thin
protocol-client lifecycles. `yvex serve` starts with zero engines;
`yvex_server_engine_load`, `yvex_server_engine_unload`, and
`yvex_server_engine_snapshot` own the in-process lifecycle. Server sessions
retain independent execution state, exact token ledgers, transcripts, and turn
records across client detach and are bound to one exact generation. The next
turn reuses state only after exact token-prefix admission and prefills only the
new suffix.

Streaming sends a fragment after model, decoder, and internal text commit.
Client delivery failure preserves committed state and reports a partial turn.
The runtime-client object lane in `yvex` links the protocol/client surface only
and cannot call the internal generation or runtime-model APIs directly. Offline
routes in the same ELF have separately guarded engine dependencies.

## Runtime Binding And Operator Actions

The offline command lane provides the direct production consumer for the internal ABI:

```text
yvex bench attention prepare
yvex inspect attention describe
yvex inspect attention capabilities
yvex inspect attention plan
yvex bench attention execute
yvex bench attention compare
yvex inspect attention state
yvex bench attention state validate|exercise
yvex inspect attention residency
yvex bench attention capture|replay
yvex bench attention graph list|inspect|warmup|update|invalidate|release
yvex bench attention trace|profile|component|qualify
yvex bench attention benchmark compare
yvex bench moe
yvex bench transformer execute
```

`prepare` is the compiler-side producer for an external runtime binding.
Execution actions require the binding and do not regenerate it. `plan` seals a
request descriptor without numerical dispatch. State actions allocate and
exercise the real process-local persistent provider through multiple production
executions, including causal read-after-write and clear/reuse. Graph-registry
actions operate on the same session rather than report-only labels.
Registry inspection reports captured kernel, copy and memset nodes plus capture,
instantiation, update and replay timings. It is not a persistent cross-process
graph cache.

The canonical probe preserves real model width, heads, bindings, qtypes,
position policy and attention history geometry. It is deterministic diagnostic
input, not prompt text. Production activation prefill instead selects
`--input tensor-file --input-file FILE`, validates the schema-v1 bundle, and
reports `activation_prefill_ready` separately from
`full_model_prefill_ready`.

`bench moe` requires explicit artifact and runtime-binding paths,
`--backend cpu|cuda`, `--input tensor-file`, `--input-file FILE`, `--scope
full`, and `--progress off`. It calls the production runtime MoE API directly;
it does not run a fixture, test executable, Make target, or second process.

`bench transformer execute` requires explicit artifact, runtime binding, and a
schema-v1 `--input token-ids --input-file FILE`. It accepts `--backend
cpu|cuda`, `--phase prefill`, positive chunk/context capacities, and
`--progress off`. It calls the production transformer API directly and reports
normalized hidden and persistent-state facts without invoking tokenizer,
logits, or generation owners.

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

`--chart PATH.svg` is available on `profile`, `benchmark`, and `benchmark
compare`; it is not a global attention option. It writes a deterministic SVG
containing cold preparation, warm latency distributions, resident/workspace
bytes, resident H2D bytes, and kernel/graph launch, capture, replay, and node
counters, optionally against a compatible baseline. The schema-five baseline
seals those structural counters, timings and complete reproducibility identity.
Schemas one through four require regeneration.
Baseline and SVG publication are independently atomic and no-replace; an SVG
failure never withdraws an already admitted baseline. JSON, CSV, baseline, and
generated SVG outputs are external operator assets and are not tracked. They
are not full-model benchmark or release evidence.

## Capability And Claim Boundary

The common runtime publishes granular facts for semantics, core/envelope,
CPU/CUDA phase and mode, residency, workspace, state delta, trace, profile and
benchmark readiness. Compatibility booleans may be derived from that lattice;
they are not independent capability authorities.

The current runtime supports the complete hosted DeepSeek prompt-to-text path on
CPU and the admitted mixed GB10 CUDA path. It composes exact tokenizer encoding,
prompt-suffix prefill, persistent state, MoE, the complete Transformer, raw
vocabulary logits, common-host sampling, sampled-token decode feedback, typed
stop, incremental detokenization, and committed streaming through one server
model and isolated server sessions. It does not establish public serving,
model evaluation, a release-path full-model benchmark, or release readiness.

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
