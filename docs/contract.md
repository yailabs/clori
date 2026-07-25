# YVEX Runtime Contract

This document defines observable runtime behavior: ownership, admission,
publication, refusal, cleanup and capability claims. The installed C surface is
mapped in [docs/api.md](api.md).
Project state and dependency order are defined only by `PROJECT.md`.

The rule is simple: each runtime result identifies the immutable facts it
consumed, the mutable owner it changed, the exact execution boundary it crossed
and the point where a refusal stopped progress.

## Scope

YVEX builds two root executables:

```text
./yvex   operator CLI
./yvexd  bounded status/server shell
```

The admitted attention path is:

```text
complete external artifact
  -> external content-addressed runtime binding
  -> immutable common runtime model
  -> mutable execution session
  -> resident attention weights + reusable workspace
  -> session-owned persistent attention state
  -> phase- and mode-aware attention execution
  -> output tensor + atomically committed state delta + typed evidence
```

DeepSeek-V4-Flash is the first family adapter. It is not the owner of a second
runtime. The admitted persistent state is an attention-session boundary;
tokenizer-backed prompt prefill, MoE, complete transformer execution, model
decode, logits, sampling and generation remain outside this contract.

## Command Contract

The command catalog is discoverable through:

```sh
./yvex commands
./yvex help
./yvex help graph attention
```

Model selection is explicit. A command may resolve a typed registry alias or an
existing path where its input contract allows either, but it never chooses an
arbitrary model. Refusal returns non-zero even when a structured report is
rendered successfully.

The `graph attention` hierarchy is the operator surface for the common runtime:

```text
prepare, describe, capabilities, plan, execute, compare
state inspect, state validate, state exercise
residency inspect
capture, replay
cuda-graph list, inspect, warmup, update, invalidate, release
trace, profile, benchmark
```

The CLI parses typed input, invokes production runtime APIs and renders copied
results. CUDA Graph lifecycle actions operate on a real registry within the
command's process-lifetime session; they do not claim persistent cross-process
state. State actions allocate, inspect, validate, exercise, clear, and reuse the
production session-owned persistent provider; they do not use a CLI cache. The
CLI does not implement attention math, call Make, run a test program, spawn
another YVEX process or link the test-only oracle.

## Filesystem Contract

Build output, external runtime bindings, model artifacts, benchmark baselines,
and JSON/CSV reports are operator assets. They remain outside source control.
The canonical chart publication target may retain six validated deterministic
SVG snapshots under `docs/assets/benchmarks/attention/`; these are bounded
documentation assets derived from external evidence, not benchmark records.

Complete artifacts are opened read-only. Canonical filesystem owners reject
symlink substitution, unsafe paths and accidental replacement. Transactional
publication uses a unique temporary file in the destination filesystem,
complete writes, file sync, atomic no-replace rename and parent-directory sync.
Failure removes only temporary resources created by that operation.

Tracked GGUF files are reserved for bounded fixtures. No complete artifact,
source model payload, runtime binding, resident snapshot, benchmark baseline,
or raw benchmark report belongs in git. Curated chart snapshots follow the
bounded documentation exception above.

### Native GGUF Reader Contract

The GGUF v3 reader uses bounded positioned reads and owns decoded metadata,
names, arrays, tensor directory and indexes. Structural parse stops at the
aligned data boundary and reads no tensor payload bytes.

The layout validator requires power-of-two alignment, canonical directory
order, exact padded continuation, zero padding, valid aggregate spans, canonical
tail policy and stable file snapshot. Parse or layout acceptance alone is not
complete-artifact admission.

### Verified Source Payload Contract

The verified payload session is a compiler-side construction boundary. It is
created only from one exact source-verification result and its immutable tensor
inventory. It owns bounded shard handles and tensor ranges, detects drift and
publishes transactional range delivery.

Source payload streaming remains build-time access and does not enter runtime
model open or warm execution. Runtime code reads no safetensors header, source
index or source payload byte.

### Model Compilation Contract

Compilation owns architecture facts, complete source coverage, artifact-neutral
Transformation IR, physical lowering, quantization planning and GGUF writer
planning. Each sealed identity excludes pointers, allocation order, paths,
timestamps and execution timing.

The runtime consumes compilation output only through an admitted artifact and a
versioned runtime binding. It does not reconstruct model roles from tensor names
or rebuild Transformation IR, quantization plans or writer plans.

These are implementation facts, not a runtime progress ladder. Current
milestone state, dependency order and the active boundary remain defined only by
`PROJECT.md`.

## Artifact Admission Contract

Complete-artifact admission binds one immutable file snapshot to exact
container structure, metadata, tokenizer material, tensor inventory, qtypes,
shapes, offsets, padding, payload ranges and full-file SHA-256.

The artifact handle remains open for the lifetime of a runtime model. Its
snapshot binds device, inode, regular-file type, size, mtime, ctime and digest
obtained from that handle. Runtime validates drift before and after execution.
A detected replacement or mutation invalidates the model and every dependent
session, resident pack, workspace, CUDA Graph executable and candidate state.

## Runtime Binding Contract

A runtime binding is an immutable content-addressed artifact, not a cache. It
binds at least:

- schema and family-adapter identity;
- artifact, materialization and logical-model identities;
- runtime-numeric and runtime-descriptor identities;
- attention plan, semantic graph and executable graph identities;
- required tensor bindings, qtype and backend requirements;
- physical compatibility and invalidation facts.

The compiler-side `yvex graph attention prepare` action generates and publishes
the binding transactionally outside the repository. Runtime open independently
parses the record and verifies every imported descriptor and plan against the
exact artifact. A missing binding refuses with the preparation command; runtime
execution never silently regenerates it.

## Common Runtime Model Contract

The common runtime contains no family-name branch. A typed family adapter
projects already admitted family facts and graph composition. Unsupported
families or sequence-mixer semantics refuse through the adapter registry.

The immutable runtime model owns:

- the verified open artifact handle and one complete SHA-256 pass;
- one parsed runtime binding;
- one imported materialization view and runtime descriptor;
- one semantic and one executable attention graph;
- one read-only resident encoded attention-weight pack;
- capability facts derived from those exact owners.

Within one model lifetime, warm execution performs zero complete artifact hash,
zero GGUF parse, zero source/compiler reconstruction and zero weight artifact
read. Model seal produces one identity over canonical values, never native
object bytes.

The execution session owns mutable backend context, reusable workspace,
attention-local state, cancellation, counters and CUDA Graph registry. Different
sessions may share immutable model and resident weights; mutable workspace and
state are never shared implicitly. Concurrent use of one busy session refuses.

## Residency And Workspace Contract

The resident pack resolves every required attention binding once and preserves
encoded qtype bytes. CPU sessions read stable resident bytes. CUDA sessions
upload required weights once to stable device addresses and reuse them across
eager and graph execution.

Prepared steady state performs:

```text
zero host allocations
zero CUDA allocations or frees
zero weight artifact reads
zero weight uploads
zero workspace resizing
zero CUDA Graph capture
```

Host and device capacities are explicit. A request exceeding its prepared
workspace, state or capture bucket refuses before dispatch; it is not truncated
or resized under a captured graph.

## Runtime Phase And State Contract

Attention phases are precise:

- `prefill` means a multi-token activation chunk with an immutable prior
  attention-state view;
- `decode` means one activation token with an immutable prior state view;
- `mixed` and `verify` are represented but currently refused.

These names do not mean prompt prefill or autoregressive model decode. Input is
an explicit activation tensor or the canonical deterministic attention probe at
real model geometry.

Each execution session owns one sealed persistent-state layout derived from the
runtime descriptor and exact family recipes. DeepSeek projects distinct SWA,
CSA, and HCA local, compressed, indexer, and rolling components across all 43
main layers. CPU state remains in stable reusable host storage; admitted CUDA
sessions own two stable device banks and resolve committed spans directly
without CPU numerical fallback.

State has an immutable committed view and a transactional candidate generation.
One append begins at the committed position, stages every required layer
publication, validates completion, and commits all layer banks plus sequence
position exactly once. Failure, cancellation, or abort publishes neither a
partial state nor an advanced position. Clear retains compatible allocation,
resets content and position, and invalidates dependent graph executions.
Capacity, non-contiguous append, stale generation, invalid coordinates, and
artifact/runtime incompatibility refuse before mutation.

For equal initial state and input activation sequence, one N-token attention
chunk and N ordered one-token attention decode operations must agree on every
output, compressed/indexer transition, local tail and final state delta.

## Graph Execution Contract

The runtime identifies three levels independently:

| Level | Contents |
| --- | --- |
| semantic graph | family-correct equations, policies, shapes, dtypes and state transitions |
| executable graph | lowered operations, dependencies, buffers, residency, workspace and backend variants |
| CUDA launch graph | concrete Driver API nodes, stable addresses, dependencies and instantiated executable |

An attention-plan identity is not a CUDA Graph identity. Artifact, binding,
numeric, descriptor, graph, residency, workspace, state-layout or device drift
invalidates only the dependent levels and prevents stale replay.

### DeepSeek Attention Execution Contract

DeepSeek attention consumes the runtime descriptor, immutable attention plan,
resident encoded weights, activation/position input and explicit state view.
Preflight validates all 43 main-layer descriptors and 634 attention-core
bindings, qtypes, shapes, head geometry, state, scratch and backend variants
before mutation.

The core executes 2 SWA, 21 CSA and 20 HCA layers. Rolling compression, index
scoring, deterministic top-k, masks, stable softmax, value reduction and output
projection are numerical inputs to the committed result. HCA uses exact
ratio-128 grouping. The attention envelope owns only immediate attention-side
normalization/residual/mHC transforms; it does not execute deferred FFN/MoE
work or claim a transformer block.

The independent full-equation oracle is test-only and linked separately. The
production binary never calls it. CPU/CUDA equality is compared against the
numeric contract and is not accepted merely because two production paths agree.

### CUDA Execution Mode Contract

CPU supports eager mode. CUDA supports:

- `eager`: direct production kernel launches over resident weights;
- `piecewise`: multiple instantiated CUDA Graphs matching real executable
  subgraphs and explicit typed boundaries;
- `full`: one instantiated CUDA Graph for the selected stable execution unit;
- `auto`: deterministic selection of full, then piecewise, then eager among
  admitted modes.

An explicit request runs the requested mode or refuses. It never downgrades.
Only `auto` may select another mode and reports the reason. Piecewise and full
use the existing generated production kernel bundle through the CUDA Driver
API. No fallback PTX or CPU numerical completion is permitted.

Capture buckets bind token and history capacities to stable workspace
addresses. Padding, where admitted, is masked and excluded from state
publication and output identity. A request never enters a smaller bucket.

### DeepSeek Attention Operator Contract

The main binary exposes the runtime through `yvex graph attention ...`.
Representative execution is:

```sh
./yvex graph attention execute --target deepseek4-v4-flash \
  --runtime-binding /path/to/binding.yvex-runtime-binding \
  --backend cpu --phase prefill --mode eager \
  --operation-scope envelope --tokens 4 --probe canonical --output json

./yvex graph attention execute --target deepseek4-v4-flash \
  --runtime-binding /path/to/binding.yvex-runtime-binding \
  --backend cuda --phase decode --mode full \
  --operation-scope release-attention-set --probe canonical --output json
```

Quick scope executes representative SWA/CSA/HCA layers. Full scope executes all
43 layers and 634 core bindings. Both retain exact geometry and admitted
weights. Neither accepts prompt text.

Planning seals an execution descriptor without numerical dispatch. State,
residency, capture/replay, registry, trace, profile and benchmark actions call
the same production owners and return typed refusal when their exact prerequisite
is absent.

## Digest And Evidence Contract

Runtime output keeps four concepts distinct:

```text
tensor_output_digest       canonical output geometry and bytes
state_delta_digest         canonical candidate state content and geometry
execution_evidence_digest  backend/mode/graph stages and counters
execution_identity         full request/result compatibility identity
```

CPU and CUDA always retain their own exact tensor and state digests. A common
tensor or state digest is published only when the corresponding canonical bytes
are identical; it is unavailable rather than forged when numerical admission
passes but exact bytes differ. Numerical comparison is a separate versioned
contract over output values and the complete state delta: raw KV, emitted
compressed/indexer values and positions, rolling geometry, KV, and scores.
Signed zero may therefore pass numerical admission while remaining visibly
non-bitwise. Evidence remains path-specific. Identity serialization hashes
canonical fields, never pointers, native padding, local paths or timing.

Trace levels are `none`, `summary`, `stages` and `full`. Evidence mode observes
the same production execution; it does not substitute a second algorithm.
Machine-readable stdout contains no progress lines.

## Quality, Qualification, Benchmark, And Evaluation Contract

YVEX keeps software testing, numerical conformance, runtime qualification,
component benchmarking, model behavior evaluation, model quality evaluation,
agent runtime evaluation, and release qualification as separate evidence
classes. `graph attention qualify` exercises the installed production runtime;
it does not rebuild the repository or run source-tree sanitizers. Its typed
result reports software-contract admission, numerical-conformance admission,
runtime structural qualification, component-benchmark availability, and the
unavailable higher stages independently.

Runtime qualification is pass/fail and requires the sealed runtime model,
admitted binding, valid artifact, complete attention residency/workspace,
reusable execution session, valid graph registry, cancellation safety,
transactional output/state publication, deterministic identities, and cleanup.
It also enforces one cold artifact hash, zero warm hashes, zero runtime
source/compiler planning, zero warm weight reads/uploads, zero warm host/device
allocations or frees, no stale graph replay, no backend fallback, and no
explicit-mode downgrade. Performance never changes this structural verdict.

`config/attention_quality.tsv` is the machine-readable admission matrix for
core, envelope, and release-attention-set scopes across SWA/CSA/HCA,
attention-prefill/chunk and attention-decode-step phases, CPU/CUDA modes, and
trace levels. Each rule states support or a typed reason and binds an evidence
identity. Blank cells do not imply support.

### Attention Component Benchmark

Attention `profile` and `benchmark` measure the common runtime only. They
separate artifact authentication, model seal, residency, workspace and graph
preparation from first and steady-state execution. Warm samples report minimum,
p50, p90, p95, p99, maximum, mean and dispersion plus allocations, transfers,
launches and memory peaks. The result declares
`benchmark_scope=attention_component`, correctness/runtime preconditions, and
separate correctness, structural-runtime, and performance statuses.

A benchmark record is versioned and identity-bound to its build commit, artifact,
runtime binding, logical model, runtime numeric policy, runtime and semantic/
executable/execution descriptors, residency, workspace, state layout, kernel
bundle, machine, CPU/GPU, memory, device, driver, CUDA build, class, layer,
phase, mode, scope, tokens, history, trace, bucket and iteration count.
`graph attention benchmark compare` reopens two records independently and
refuses at the first incompatible identity field. Workload compatibility
deliberately excludes the commit so a regression lane can compare two builds
while reporting both commit identities.

Comparison has no implicit performance threshold. The optional
`--max-regression-bps N` policy applies one caller-owned basis-point ceiling to
host/device latency and inverse throughput, peak host/device memory, H2D/D2H
traffic, steady-state allocation counters, and kernel/graph launch counts. The
policy and comparison outcome have distinct canonical identities. Without a
policy, performance status is `measured`; with a policy, a breached ceiling
returns a nonzero process status. Warm allocations, uploads, and other runtime
qualification violations refuse before performance policy is evaluated.

`--chart PATH.svg` is accepted by `profile`, `benchmark`, and `benchmark
compare`; execution and inspection actions do not manufacture charts. It
produces a deterministic SVG of cold preparation, warm latency,
resident/workspace bytes, resident H2D bytes, and kernel/graph launch, capture,
replay, and node counters, optionally paired with a compatible baseline. Schema
five seals complete reproducibility identity, timing boundaries, resource
counters, and build provenance; schemas one through four refuse and require
regeneration instead of being silently reinterpreted. The chart identity covers
its exact bytes. Baseline and chart are independent no-replace publications: a
valid baseline remains authoritative if optional chart publication later refuses.
JSON, CSV, baseline and SVG files are local operator evidence; they are not
full-model benchmark results and are not tracked.

The attention oracle and canonical tensor probe are numerical-conformance
inputs, not question-answering or model-quality evaluation. Model behavior,
model quality, full DeepSeek generation benchmark, agent runtime/evaluation,
and release qualification remain unavailable. No generic `qa_passed` or
`qa_ready` fact is authoritative.

## Backend Contract

Backend capability is typed and variant-specific. A ready context is not a
ready operation. CUDA bundle, device compute capability, function variant,
workspace, residency and graph API admission all complete before dispatch.

Allocation, transfer, launch, synchronization and cleanup failures remain
distinct. A CUDA request never switches to CPU. A graph request never aliases
eager execution. Failure publishes neither output nor state delta and releases
only resources owned by the failed transaction.

## Output Contract

Normal output is compact. Table and audit output project the same typed result
at increasing detail. JSON and CSV stdout remain parseable and contain no ANSI
or progress noise; optional human progress uses stderr.

Unavailable values use a typed unavailable representation when zero could be a
valid measurement. Renderer success does not convert domain refusal into exit
status zero.

## Server Contract

`yvexd` remains a bounded status/server shell. Health, metrics and model-listing
surfaces do not imply runtime generation. OpenAI/Anthropic compatibility,
streaming, tool calls and provider sessions remain unavailable until the same
runtime path owns complete generation.

## Validation Contract

Runtime changes require positive, refusal, cancellation, rollback, cleanup,
identity and concurrency tests. Numeric execution requires an independent
reference and backend comparison. Lifecycle changes run ASan/LeakSanitizer and
UBSan; CUDA changes run no-`nvcc` refusal and real-device validation.

Minimum repository validation remains:

```sh
git diff --check
make
make smoke
make test-core
make check
make check
make check-docs
make check-guardrails
make test-cuda-no-nvcc
make check-cuda
```

Real weights, complete GGUF artifacts, runtime bindings, benchmark baselines,
and raw benchmark reports must remain untracked. Only the curated SVG
documentation snapshots admitted by the filesystem contract may be tracked.

## Claim Promotion Contract

A capability becomes true only when its owner, prerequisites, production API,
operator command, tests, failure behavior, cleanup and identity-bound evidence
all exist. A complete artifact is not runtime. Attention prefill/decode is not
model prefill/decode. Attention residency is not full-model residency. An
attention benchmark is not a full-model benchmark.

The current common runtime admits attention semantics, attention core/envelope,
CPU eager phases, CUDA eager/piecewise/full phases, resident attention weights,
reusable workspace, session-owned persistent DeepSeek attention state, and
runtime-local operator evidence. Mixed/speculative attention, tokenizer-backed
full-model prefill, MoE, transformer, model decode, logits, sampling,
generation, evaluation, full-model benchmark and release remain unsupported.
