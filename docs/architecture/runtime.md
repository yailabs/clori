# Runtime and Execution Architecture

Status: current implemented architecture

This document explains the implemented runtime lifecycle, execution pipeline,
persistent state, and resource model. Normative behavior belongs to the
[Runtime Contract](../contracts/runtime.md), [Local Protocol](../contracts/local-protocol.md),
and [Events Contract](../contracts/events-telemetry.md).

## Hosted lifecycle

One `yvexd` process owns one immutable runtime model from startup until
shutdown. Startup authenticates the complete artifact and runtime binding,
imports descriptors, builds model-lifetime host and accelerator resources,
prepares tokenizer/output-head facts, starts the bounded worker, and publishes
readiness before admitting clients.

![One runtime host owns immutable model resources and isolated conversation/execution sessions.](../diagrams/runtime_host_sessions.svg)

The editable source is
[`runtime_host_sessions.mmd`](../diagrams/runtime_host_sessions.mmd).

Shutdown stops new HTTP and local-protocol admission, resolves bounded active
work, closes server sessions, stops the worker, closes the runtime model once,
and releases listeners. A client disconnect is not a model or session close.

## Runtime objects

The immutable runtime model retains:

- the authenticated artifact and binding;
- imported family-neutral descriptors and the typed family adapter;
- read-only encoded weights and model-lifetime backend resources;
- the tokenizer plan and output-head residency;
- executable capability and identity facts.

Each server session owns one mutable execution session, committed token ledger,
transcript, incremental decoder, sampling/RNG state, turn state, and persistent
sequence state. Sessions share immutable weights but never mutable KV or
workspace.

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

## Transformer composition

The DeepSeek adapter supplies the irreducible 43-layer schedule, attention
classes, residual composition, and MoE policy. Generic runtime and graph owners
retain allocation, state transactions, numerical primitives, dispatch,
publication, and cleanup.

The current mixed GB10 execution keeps inter-layer activations on CUDA for the
model backbone and output head. Sampling and tokenizer work remain on the host.
Unsupported CUDA operations fail closed; no requested CUDA execution silently
falls back to CPU.

## Persistent state

Persistent state includes committed position and family-correct attention/KV
representations. It is session-owned and capacity-bounded. Workspace is
separate reusable temporary storage.

Every execution unit follows:

```text
admit -> begin candidate -> execute -> validate -> check cancellation
      -> publish output/state -> commit
```

Failure aborts the candidate and retains the prior committed state. Multi-turn
reuse occurs only when the newly rendered token sequence has the committed
ledger as an exact prefix. The runtime prefills only the suffix; an
incompatible prefix refuses or requires explicit reset.

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
internal text state agree. A cancelled or disconnected request reports exact
partial progress and does not fabricate completion.

One typed event authority carries lifecycle, queue, tokenizer, prefill, first
token, decode, commit, stop, cancellation, failure, memory, and listener facts.
Human status, watch, trace, raw JSONL, and metrics are projections of that
authority. The active console milestone owns improved human rendering, not a
new state source.

## Current limits

Warm DeepSeek performance remains below the admission target and is explicit
optimization debt. The architecture does not claim continuous batching,
multi-model hosting, restart-persistent sessions, public security, model
evaluation, a release benchmark, or release qualification.
