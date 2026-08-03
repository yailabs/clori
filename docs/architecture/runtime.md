# Runtime and Execution Architecture

Status: current implemented architecture

This document explains the implemented runtime lifecycle, execution pipeline,
persistent state, and resource model. Normative behavior belongs to the
[Runtime Contract](../contracts/runtime.md), [Local Protocol](../contracts/local-protocol.md),
and [Events Contract](../contracts/events-telemetry.md).

## Hosted lifecycle

One `yvexd` process owns one immutable runtime model from startup until
shutdown. Startup authenticates the complete artifact and runtime binding,
imports descriptors and Physical Execution IR, seals the compiled execution
profile, builds model-lifetime host and accelerator resources, prepares
tokenizer/output-head facts, starts the bounded worker, and publishes readiness
before admitting clients.

![One runtime host owns immutable model resources and isolated conversation/execution sessions.](../diagrams/runtime_host_sessions.svg)

The editable source is
[`runtime_host_sessions.mmd`](../diagrams/runtime_host_sessions.mmd).

Shutdown stops new HTTP and local-protocol admission, resolves bounded active
work, closes server sessions, stops the worker, closes the runtime model once,
and releases listeners. A client disconnect is not a model or session close.

## Runtime objects

The immutable runtime model retains:

- the authenticated artifact and binding;
- the Physical Execution IR and compiled execution profile;
- imported family-neutral descriptors and the typed family adapter;
- read-only encoded weights and model-lifetime backend resources;
- the tokenizer plan and output-head residency;
- target and DSpark draft/verification plans, their shared resources, and
  executable capability and identity facts.

Each server session owns one mutable execution session, committed token ledger,
transcript, incremental decoder, sampling/RNG state, turn state, and persistent
target sequence state. A DSpark session also owns bounded proposal and target
verification candidate state. Sessions share immutable weights but never
mutable KV or workspace.

## Execution path

```text
rendered prompt
  -> exact tokenizer IDs
  -> committed-prefix comparison
  -> suffix prefill
  -> complete Transformer
  -> final normalized hidden state
  -> full-vocabulary logits
  -> sampling
  -> sampled-token decode feedback
  -> stop classification
  -> incremental detokenization
  -> committed streamed text
```

![Autoregressive execution commits model state before publishing text and feeds each sampled token back through decode exactly once.](../diagrams/autoregressive_execution.svg)

The editable source is
[`autoregressive_execution.mmd`](../diagrams/autoregressive_execution.mmd).

Prefill and decode consume the same Transformer and persistent-state boundary.
Decode is a phase, not a second model implementation. Generation composes
tokenizer, prefill, logits, sampling, decode, stop, and detokenization owners;
it does not replace their semantics.

## Target-only and DSpark generation

The same runtime model admits two explicit modes. `target-only` executes one
ordinary target step at a time and remains the semantic reference. `dspark`
captures the source-required target features, constructs a bounded candidate
block with the checkpoint drafter, verifies it with the complete target, and
commits the target-authored accepted result.

```text
committed target state
  -> DSpark proposal and confidence facts
  -> complete-target block verification
  -> accepted prefix plus correction or bonus token
  -> atomic model/token/decoder/RNG commit
  -> committed streaming
```

A proposal does not advance target position, persistent state, transcript,
usage, or output. Target verification retains a prefix-addressable candidate
transaction with ordered state checkpoints. Acceptance promotes the exact
verified checkpoint, rejected suffix state is discarded, and accepted target
rows are not replayed. The only additional target execution is one correction
or bonus token required by the speculative algorithm. Greedy DSpark output
matches target-only output from the same initial state; admitted stochastic
execution preserves the target distribution through exact accept/reject and
residual sampling rather than token-ID equality.

The target plan captures ordered normalized feature taps after layers 40, 41,
and 42 during ordinary execution. The drafter does not rerun the target trunk
to reconstruct them. Draft Markov state is token-conditioned workspace, while
committed sequence truth remains target-owned.

## Transformer composition

The DeepSeek adapter supplies the irreducible 43-layer schedule, attention
classes, residual composition, and MoE policy. Generic runtime and graph owners
retain allocation, state transactions, numerical primitives, dispatch,
publication, and cleanup.

The current mixed GB10 execution keeps inter-layer activations on CUDA for the
model backbone and output head. Typed device views carry hidden, expanded
residual, feature, logits, probability, candidate, accepted-prefix and
workspace values with explicit owner, identity, generation, extent, lifetime,
synchronization and materialization policy. Production greedy selection uses a
device argmax and transfers only the selected token and bounded status.
Stochastic sampling, token-local MoE, row-local output projection, eager
attention and host feature projection remain explicitly named portable
reference adapters. Unsupported CUDA operations fail closed; no requested
CUDA execution silently falls back to CPU.

## Persistent state

Persistent state includes committed position and family-correct attention/KV
representations. It is session-owned and capacity-bounded. Workspace is
separate reusable temporary storage.

An ordinary execution unit follows:

```text
admit -> begin candidate -> execute -> validate -> check cancellation
      -> publish output/state -> commit
```

Failure aborts the candidate and retains the prior committed state. Multi-turn
reuse occurs only when the newly rendered token sequence has the committed
ledger as an exact prefix. The runtime prefills only the suffix; an
incompatible prefix refuses or requires explicit reset.

A speculative cycle extends the same discipline to several positions. Target
attention state uses ordered candidate deltas and checkpoints; SWA uses a
ring-backed contiguous projection instead of moving the complete window.
Compressed, indexer, rolling and mHC state, token ledger, incremental decoder,
target and draft sampling state, and RNG publication participate in one
accepted-prefix transaction. Cancellation before commit discards the whole
candidate. Cancellation after commit reports that exact committed prefix; it
never rewinds published state by decrementing counters.

## Execution profiles and shape admission

One compiled execution profile binds logical model, physical variant, Physical
Execution IR, artifact, materialization, runtime binding, kernel bundle,
hardware, context, generation mode, workload, evidence profile and execution
class. `production`, `audit`, and `forensic` evidence are independent from
trace verbosity: production cannot require complete hidden, state, logits, or
probability host scans merely to derive evidence.

The portable CUDA production profile keeps its admitted Q8 activation codec
and parallel F32 reductions. Full forensic attention comparison instead selects
a canonical-order F64 accumulator and disables activation compression for that
operation, so the independent stage oracle measures semantic execution rather
than the production approximation. This slower numerical adapter is unreachable
from production and is not a performance path.

Before target prefill/decode, draft, verification, correction, or reset, the
runtime selects an identity-bound execution shape. The shape distinguishes
target/draft scope, phase, operation, width, context band, candidate
visibility, local/compressed/indexer/rolling/candidate capacities, workspace
generation, attention/state/kernel identities, and evidence profile. A missing
compatible shape is admitted as a new eager/reference shape before numerical
mutation or refuses with the exact component, configured and required
capacity, position, width, scope and identities. Target and draft work never
overwrite one shared mutable capacity record.

## Memory and residency

The process distinguishes file mapping, anonymous host residency, locked or
CUDA-addressable host storage, accelerator-resident storage, session KV,
workspace, and transient staging. Physical unified memory does not collapse
these placement and accounting classes.

The current complete encoded payload is copied into a process-lifetime host
arena. A bounded accelerator placement contains the admitted CUDA-resident
resources. Status reports host and accelerator bytes separately. A placement
counter is a memory fact, not by itself a causal performance diagnosis.

Mutable session resources remain isolated while immutable model caches may be
shared. Allocation, transfer, synchronization, execution, and cleanup failures
retain separate typed classes.

## Worker, queue, and concurrency

One bounded model worker serializes admitted generation work from native and
HTTP clients. Socket/listener threads parse and project requests but do not
mutate model state directly. Continuous batching, multiple hosted models, and
distributed serving are not implemented.

## Publication and observability

Token fragments are published only after model state, decoder state, and
internal text state agree. A failure after committed output produces a typed
partial-turn snapshot containing exact position, committed-token and published
byte counts, state generations and identities, failure class, and reset
requirement. A partial session visibly refuses another ordinary turn until
reset. Cancellation, disconnect, failure, and maximum-token stop remain
distinct states.

One typed event authority carries lifecycle, queue, tokenizer, prefill, first
token, decode, draft, verification, accepted-prefix commit, stop,
cancellation, failure, memory, and listener facts.
Human status, semantic watch, detailed trace, raw JSONL, and metrics are
projections of that authority. Watch groups startup and each request into
coherent units while trace retains detailed events. The REPL incrementally
renders UTF-8 and Markdown across arbitrary protocol fragments without
changing canonical bytes; raw and redirected output remains byte-exact and
free of terminal controls. Explicit reasoning is classified only through the
tokenizer's source-authored channel contract and never by prose inspection.

## Current limits

The DSpark bootstrap path is a correctness baseline and may be slower than
target-only execution. Warm DeepSeek performance remains explicit optimization
debt. The architecture does not claim load-aware confidence scheduling, native
FP4 Tensor Core execution, continuous batching, multi-model hosting,
restart-persistent sessions, public security, model evaluation, a release
benchmark, or release qualification.
