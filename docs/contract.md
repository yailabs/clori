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
runtime. The admitted persistent state is session-owned and consumed by the
numeric token-ID transformer backbone. Tokenizer-backed prompt prefill,
token append and generation remain outside this contract.

The admitted token-local MoE path is:

```text
typed per-layer expanded activation + numeric token ID
  -> immutable family-projected MoE plan
  -> hash or learned router + deterministic top-k
  -> selected routed expert subviews + shared expert
  -> CPU or CUDA encoded execution
  -> combined MoE output + deferred transformer-owned post state
```

The admitted transformer path is:

```text
canonical numeric token IDs
  -> selected encoded embedding rows + initial mHC residual state
  -> 43 ordered attention/MoE blocks + one request-level state transaction
  -> final mHC collapse + final RMSNorm
  -> normalized hidden state + atomically committed persistent state
```

The admitted repeated-decode path is:

```text
committed nonzero prefix + externally supplied numeric token ID
  -> explicit decode-phase transformer execution over prior KV
  -> one complete 43-block transaction
  -> one committed state advance + one normalized hidden row
  -> next externally supplied token or typed partial completion
```

Decode is teacher-forced. It does not select tokens or form an autoregressive
generation loop.

The admitted logits path is:

```text
transformer-authenticated normalized hidden row
  -> exact separate output-head plan and model-lifetime residency
  -> direct encoded CPU or GB10 CUDA projection over all vocabulary rows
  -> one complete F32 raw-logits row + typed identities
```

Logits projection does not repeat final norm, change persistent state, publish
probabilities, or select a token.

The admitted sampling path is:

```text
complete identity-bound F32 logits row
  -> canonical greedy or temperature-scaled stochastic policy
  -> top-k, min-p, locally typical, and top-p filtering
  -> transactional versioned RNG draw where stochastic
  -> one selected canonical vocabulary token ID
```

Sampling is a common host-runtime operation. It neither appends the selected
token nor changes KV, position, generation, logits values, tokenizer state, or
model resources.

## Command Contract

The command catalog is discoverable through:

```sh
./yvex commands
./yvex help
./yvex help graph attention
./yvex help graph transformer
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

`graph moe execute` is the operator surface for the production token-local MoE
boundary. It consumes a schema-v1 tensor file and the admitted artifact/runtime
binding directly. Numeric token IDs participate only in hash routing; they are
not tokenizer output or tokenizer support.

`graph transformer execute` is the operator surface for the complete numeric
token-ID backbone. It admits a schema-v1 token file, executes the production
CPU or CUDA transformer API, and publishes normalized hidden state plus the
committed state transition. It does not tokenize text, project logits, or run
an autoregressive loop.

`graph transformer decode` reuses that schema and opens one model, session, and
transformer context. It commits the first explicit token span as prefill, then
executes each remaining externally supplied ID as one decode-phase transaction.
Successful steps remain committed when a later step fails or is cancelled; the
result publishes the exact completed count and first incomplete ordinal.

`graph transformer logits` opens the same production model/session/transformer
plane, projects the final prefill hidden row and every completed teacher-forced
decode hidden row, and reports bounded complete-row identities. It neither
dumps the raw vocabulary tensor nor chooses a token.

`graph transformer sample` runs that logits workflow once, admits every value
and identity in each complete row, and applies one explicit greedy or seeded
stochastic policy. Selected token IDs are evidence only: they are not appended
or fed back into decode.

The CLI parses typed input, invokes production runtime APIs and renders copied
results. CUDA Graph lifecycle actions operate on a real registry within the
command's process-lifetime session; they do not claim persistent cross-process
state. State actions allocate, inspect, validate, exercise, clear, and reuse the
production session-owned persistent provider; they do not use a CLI cache. The
CLI does not implement attention math, call Make, run a test program, spawn
another YVEX process or link the test-only oracle.

## Filesystem Contract

Build output, external runtime bindings, model artifacts, benchmark baselines,
JSON/CSV reports, and generated charts are operator assets. They remain outside
source control.

Complete artifacts are opened read-only. Canonical filesystem owners reject
symlink substitution, unsafe paths and accidental replacement. Transactional
publication uses a unique temporary file in the destination filesystem,
complete writes, file sync, atomic no-replace rename and parent-directory sync.
Failure removes only temporary resources created by that operation.

Tracked GGUF files are reserved for bounded fixtures. No complete artifact,
source model payload, runtime binding, resident snapshot, benchmark baseline,
raw benchmark report, or generated chart belongs in git.

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

## Production Activation-Prefill Contract

The runtime admits activation prefill through one versioned pointer-free input.
Schema v1 binds the logical model, runtime numeric policy, runtime descriptor,
attention plan, operation scope, token start/count, all 43 ordered layer
identities, exact widths/strides and payload ranges, payload digest, and input
identity. The canonical tensor-file encoding is little-endian, has an explicit
header and record directory, carries only finite F32 payload values, and
permits neither native structure bytes nor path-derived identity.

The memory and file adapters use the same admission logic. A tensor file must
be a regular non-symlink file and must match its stable open-handle snapshot
before and after execution. Missing, duplicate, reordered, or unknown layers;
identity, width, stride, range, digest, token, or scope mismatches; overlap,
truncation, trailing bytes, non-finite values, drift, and budget overflow
refuse before state mutation.

One production prefill request starts at the session's exact committed
position. The coordinator checks model context, persistent-state, activation,
workspace, host/device, and chunk capacities before execution. It runs the
existing production attention executor for every ordered layer and commits the
provider and backend state once per complete chunk. A failing or cancelled
chunk publishes no partial layer state and advances neither position nor
generation; earlier committed chunks remain authoritative and are reported as
the committed prefix.

CPU eager and admitted GB10 CUDA eager consume the same activation bundle and
persistent-state contract. They do not copy attention equations into a prefill
executor and CUDA does not fall back to CPU. Canonical probes remain available
for attention diagnostics but do not establish activation-prefill capability.

This boundary publishes attention/envelope output facts, output and state
digests, chunk/layer/class counts, committed prefix, position/generation
transitions, and identities. It does not publish a complete transformer hidden
state. Prompt text, tokenization, embedding, cross-layer hidden-state
propagation, FFN/MoE, complete transformer prefill, model decode, logits,
sampling, and generation remain unsupported.

## Production MoE Contract

The immutable MoE plan is derived from the admitted runtime descriptor,
attention plan, materialization, and registered family adapter. Each of the 43
main layers binds its router class, score/top-k policy, routed/shared geometry,
activation and scaling policy, mHC FFN ordering, exact tensor roles, qtypes,
and stable identity. Runtime execution reopens no source inventory and infers
neither family policy from target strings nor roles from tensor names.

Schema-v1 MoE input binds the logical model, numeric policy, runtime descriptor,
MoE plan, token range, all ordered layer identities, exact widths/strides,
finite little-endian F32 activation ranges, numeric token IDs, payload digests,
and input identity. File admission retains a regular non-symlink handle and
rejects malformed geometry, ordering, overlap, truncation, trailing data,
digest mismatch, stale identity, drift, and resource overflow.

The first three admitted DeepSeek layers select six routed experts from the
exact token-ID hash-table row. The remaining 40 execute the real BF16 router
projection, sqrt-softplus scoring, correction bias, deterministic noaux top-k,
unbiased probability normalization, and routed scaling. Selected routed
gate/up/down subviews are addressed through the materialization expert API;
the runtime neither decodes nor uploads the complete 256-expert collection.
Each selected Q2_K SwiGLU path and the distinct Q8_0 shared expert execute
through the requested backend, then combine under the family plan.

CPU and GB10 CUDA consume the same typed plan and input. A CUDA request performs
the numerical work on device and refuses unavailable qtypes, kernels, device
capabilities, transfers, or synchronization without CPU fallback. Runtime
contexts own reusable host scratch and stable device workspace. Cancellation,
read/upload/kernel failure, or cleanup failure publishes no partial result and
does not mutate persistent attention state or sequence position.

The result distinguishes router selection, routed aggregate, shared-expert
output, combined MoE output, and deferred transformer-owned post state. It
publishes qtype/access/transfer counters and separate routing, routed, shared,
combined, input, plan, and execution identities. It is not a complete
transformer hidden state, full-model prefill/decode, or generation output.

## Production Transformer Contract

The immutable schema-v1 transformer plan binds the runtime, attention, MoE,
embedding, residual/mHC, final-head, and final-norm identities for the exact 43
layer order. The family adapter supplies initial residual, block ordering,
deferred FFN post, and final collapse policy. Runtime code neither reconstructs
these facts from tensor names nor branches on a target string.

Schema-v1 transformer input binds canonical little-endian U32 token IDs to the
logical model, numeric policy, runtime descriptor, and transformer plan. Memory
and bounded-file forms share one identity. Admission rejects unsafe files,
stale identity, discontinuous position, invalid vocabulary ordinals, malformed
extent, trailing bytes, digest mismatch, drift, and resource overflow. Numeric
token input does not establish tokenizer support.

`yvex_runtime_transformer_execute_block` completes one ordered block from the
attention publication staged inside the active request transaction. It invokes
the admitted single-layer MoE path and exact deferred mHC post; it does not
commit KV independently. `yvex_runtime_transformer_execute` coordinates
embedding, all 43 blocks, final mHC collapse, final RMSNorm, and one atomic
persistent-state commit per chunk. Failure publishes neither the failing
chunk's state nor its hidden output, while earlier committed chunks remain
authoritative.

CPU and GB10 CUDA consume the same plan and token input. CUDA performs selected
embedding decode, attention, MoE, residual composition, final collapse, and
normalization on device without a CPU numerical fallback or inter-layer
activation roundtrip. The published `[token_count, hidden_width]` tensor is a
normalized transformer hidden state. Repeated decode and logits consume this
boundary; output-head projection never re-executes final norm. Sampling,
tokenizer execution, and generation remain separate capabilities.

## Production Repeated Decode Contract

The decode coordinator borrows one admitted transformer context and its paired
execution session. It consumes bounded one-token views from the existing
schema-v1 transformer input, requires a nonzero committed prefix and exact
next position, and records decode as an explicit canonical phase. It does not
reopen the artifact or runtime binding, rebuild plans, or create another KV
owner.

Each successful step executes selected embedding, 43 attention/MoE blocks,
final mHC, and final RMSNorm, then commits one persistent-state advance and
publishes one `[1,4096]` normalized hidden row. Position and generation come
from the session state, never from a decode-owned counter. Caller output
capacity and the complete requested context extent are validated before any
step mutates state.

Repeated execution is step-atomic. A failing or cancelled step publishes no KV
or hidden row; earlier successful steps remain committed and the typed result
identifies the exact completed count and first incomplete ordinal. Step and
aggregate identities serialize token, position, generation, routing, hidden,
state, phase-bearing transformer identity, and structural counters field by
field. Numeric IDs remain externally supplied, so this boundary establishes
neither tokenizer behavior nor autoregressive token choice.

## Production Vocabulary-Logits Contract

The schema-v1 logits plan binds one transformer plan to the exact separate,
unbiased `YVEX_TENSOR_ROLE_OUTPUT_HEAD` tensor. For the admitted DeepSeek
vertical the binding is BF16 with logical shape `[129280,4096]` and
1,059,061,760 encoded bytes. The immutable runtime model shares one checked
host view and one CUDA-resident span across sessions; mutable logits buffers
remain context-local.

A logits source is admitted only through the producing transformer or decode
result. Its model, binding, plan, execution, phase, position, hidden width, and
canonical hidden digest must agree. CPU and CUDA compute each vocabulary row
directly from the encoded head and publish one complete finite F32 row only
after every coordinate succeeds. Repeated projection preserves earlier rows,
publishes no failing row, and reports the exact first incomplete ordinal.

Plan, source, residency, backend evidence, row, and aggregate identities are
serialized field by field. Logits execution does not advance session position
or generation. It publishes raw values and bounded diagnostics only; softmax,
token selection, penalties, EOS/stop policy, and generation remain separate
owners.

## Production Real-Logits Sampling Contract

The schema-v1 family-neutral sampler consumes an immutable complete logits row
whose raw digest, logits-row identity, output-head plan, source phase, source
position, hidden digest, and backend evidence all validate. It rehashes every
canonical F32 value before use, requires the complete vocabulary extent, and
never changes caller-owned logits.

Greedy scans the complete row and selects the lowest token ID among equal
finite maxima without touching RNG state. Stochastic sampling requires an
explicit seed and applies the versioned order: temperature, stable full-row
softmax, top-k, min-p, locally typical, top-p, then one categorical draw over
survivors ordered by token ID. Each filtering stage renormalizes. The private
PCG-XSH-RR 64/32 state advances exactly once only after a stochastic result is
fully validated and published; refusal or cancellation leaves it unchanged.

One context owns fixed candidate, probability, and sorting workspace plus its
private RNG and busy lifecycle. Warm calls allocate no workspace. Repeated
sampling preserves successful earlier rows and reports the first incomplete
row; a failing row publishes neither token nor RNG transition. Separate
contexts isolate mutable state. Sampling remains a common host operation even
when CUDA produced the logits and does not establish CUDA sampling, token
append, tokenizer execution, stop policy, or autoregressive generation.

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

./yvex graph attention execute --target deepseek4-v4-flash \
  --runtime-binding /path/to/binding.yvex-runtime-binding \
  --backend cuda --phase prefill --mode eager --scope full \
  --operation-scope core --input tensor-file \
  --input-file /path/to/input.yvex-activations \
  --chunk-tokens 2 --context-capacity 4096 --output json
```

Quick scope executes representative SWA/CSA/HCA layers. Full scope executes all
43 layers and 634 core bindings. Both retain exact geometry and admitted
weights. Tensor-file prefill requires full scope and executes every ordered
layer record. Neither input class accepts prompt text.

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
raw benchmark reports, and generated charts must remain untracked.

## Claim Promotion Contract

A capability becomes true only when its owner, prerequisites, production API,
operator command, tests, failure behavior, cleanup and identity-bound evidence
all exist. A complete artifact is not runtime. Attention prefill/decode is not
model prefill/decode. Attention residency is not full-model residency. An
attention benchmark is not a full-model benchmark.

The current common runtime admits attention semantics, attention core/envelope,
CPU eager phases, CUDA eager/piecewise/full phases, resident attention weights,
reusable workspace, session-owned persistent DeepSeek attention state, and
runtime-local operator evidence. Token-local MoE and the numeric-token complete
transformer backbone are admitted on CPU and GB10 CUDA. Teacher-forced repeated
decode reuses the same token schema, transformer context, and persistent state.
Transformer-normalized prefill/decode hidden rows project through the complete
resident output head to raw vocabulary logits on CPU and GB10 CUDA. The common
host sampler admits every value in those rows and performs deterministic greedy
or explicitly seeded canonical stochastic token selection. Mixed/speculative
attention, prompt/tokenizer execution, token append, CUDA sampling, generation,
evaluation, full-model benchmark and release remain unsupported.
