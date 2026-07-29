# YVEX

YVEX is a native C/CUDA model compiler and local inference runtime for
identity-bound, verified open-weight execution. It turns a verified source
snapshot into an explicit physical model variant, admits the resulting artifact,
and executes it through a long-lived runtime host with isolated sessions.

Today, the complete product path is available for DeepSeek-V4-Flash on CPU and
the admitted NVIDIA GB10 CUDA path. The model stays open in `yvexd`; native
clients use the private local protocol and applications may use the bounded
`yvex.openai.compat.v1` HTTP/SSE gateway. Evaluation, release benchmarking, and
release qualification remain open gates.

Start here:

- [Run YVEX](#run-yvex) for the first local session.
- [Product topology](#product-topology) for the process and binary boundaries.
- [What YVEX guarantees](#what-yvex-guarantees) for the execution contract.
- [Documentation](#documentation) for operator, architecture, and API detail.
- [`PROJECT.md`](PROJECT.md) for the sole live capability and roadmap authority.

## Run YVEX

### Prerequisites

You need:

- a built YVEX checkout;
- one complete YVEX-produced GGUF outside the repository;
- the runtime binding produced for that exact artifact;
- enough host and device memory for the selected physical variant;
- a supported CUDA environment for the GB10 path, or the admitted CPU path.

Build the product binaries:

```sh
make -j4 all
```

For the examples below, point two shell variables at external operator assets:

```sh
export YVEX_MODEL_ARTIFACT=/absolute/model.gguf
export YVEX_RUNTIME_BINDING=/absolute/model.yvex-runtime-binding
```

The normal workflow uses three terminals around one daemon. It does not load
three copies of the model.

### Terminal 1 — runtime and raw events

Start the long-lived host and select the structured runtime-event stream:

```sh
./yvexd --model "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096 --console raw --trace-level stages
```

Wait for the `runtime.ready` JSONL event. The daemon authenticates the artifact
and binding, copies the complete encoded model payload into its process-lifetime
host RAM arena, builds the accelerator residency once, then accepts local client
connections. Raw events exclude prompt and response text by default.

### Terminal 2 — engine watch

Subscribe to the same event authority through the compact operational view:

```sh
./yvex runtime watch
```

This view shows startup, queueing, prompt tokens, exact prefix reuse, prefill,
time to first token, decode rate, memory, stop reasons, and shutdown. It does
not print logits, tensors, KV contents, or conversation text.

### Terminal 3 — REPL

Open or attach to a named multi-turn session:

```sh
./yvex chat --session main
```

The session belongs to the daemon, not the terminal. Detaching and reconnecting
preserves its exact committed token ledger and KV state. A later turn reuses KV
only when the newly rendered prompt has the committed ledger as an exact token
prefix.

The REPL supports `/new`, `/sessions`, `/attach`, `/status`, `/reset`,
`/cancel`, `/detach`, `/close`, `/help`, and `/quit`. `Ctrl-C` first cancels an
active turn; a later interrupt or end-of-file exits according to the terminal
state.

### One-shot run

Use the existing daemon for one streamed response:

```sh
./yvex run "Explain attention in one sentence."
```

By default, `run` creates an ephemeral session, closes that session when the
turn ends, and leaves the daemon and model alive. An explicit `--session NAME`
uses an admitted existing session instead.

### OpenAI-compatible applications

Start the engine-free loopback gateway after `runtime.ready`:

```sh
./yvex-openai --host 127.0.0.1 --port 8001
```

Point an OpenAI-compatible client at `http://127.0.0.1:8001/v1` with any local
non-secret API-key placeholder. The admitted profile covers model discovery,
Chat Completions, Responses, SSE, bounded function calls, usage, stop strings,
and JSON-object validation. YVEX never executes application tools. See the
[compatibility profile](docs/openai-compatibility.md) for exact fields and
refusals.

### Status and shutdown

Inspect the authoritative runtime snapshot:

```sh
./yvex runtime status
./yvex runtime status --json
```

Stop the host through the local protocol:

```sh
./yvex runtime stop
```

Shutdown refuses new work, drains or cancels active work under typed rules,
closes sessions, closes the model once, and removes the private socket.

For model selection through private XDG configuration instead of explicit start
arguments, see the [operator runbook](docs/operator-runbook.md).

## Product topology

![YVEX system overview: verified sources become an identity-bound artifact served by one long-lived runtime host to isolated sessions, CPU or CUDA execution, and local clients.](docs/diagrams/system_overview.svg)

*One compiler path produces admitted artifacts; one daemon owns model execution;
multiple clients project the same session and telemetry authorities. See the
[editable diagram source](docs/diagrams/system_overview.mmd) and the
[detailed architecture](docs/reference-architecture.md).*

The product separates process and linkage responsibilities:

| Component | Responsibility | Engine linkage |
| --- | --- | --- |
| `libyvex` | Compilation, artifact admission, runtime, backend, tokenizer, and generation domain owners | engine |
| `yvexd` | One long-lived model host, bounded worker queue, session registry, local protocol, telemetry, and graceful shutdown | yes |
| `yvex` | REPL, one-shot client, runtime/session administration, status, watch, and trace | no |
| `yvex-openai` | Loopback HTTP/JSON/SSE adapter over the private YVEX protocol; opens no model or artifact | no |
| `yvex-dev` | Optional direct compiler, artifact, graph, tokenizer, and evidence tooling | yes |

`yvex` cannot open a model, materialize weights, execute a Transformer, or run
generation in-process. It communicates over a private, versioned Unix-domain
socket. Direct engineering operations remain separated in `yvex-dev` and may
be omitted from a product package.

The runtime layers remain distinct:

```text
verified source
  -> logical model and Transformation IR
  -> policy-selected physical variant
  -> complete GGUF and runtime binding
  -> yvexd immutable runtime model
  -> server-owned execution sessions and KV
  -> CPU/CUDA model execution
  -> committed streamed text
```

One typed event sequence feeds the daemon JSONL stream, engine watch, metrics,
and client progress. Renderers do not infer state by parsing another renderer's
text.

## What YVEX guarantees

### Identity-bound derivation

Source snapshots, logical models, transformation plans, physical variants,
artifacts, runtime bindings, sessions, executions, and evidence retain separate
canonical identities. Semantic identities exclude pointers, native padding,
local paths, process IDs, and wall-clock values.

### Logical and physical separation

The model definition does not force one qtype profile. A sealed policy resolves
each terminal tensor into one admissible physical representation. The writer,
materializer, and runtime consume that resolved plan or artifact truth; they do
not independently choose quantization.

### Fail-closed execution

Unsupported qtypes, stale bindings, invalid identities, unsafe paths, capacity
violations, unavailable CUDA operations, malformed protocol frames, and
incompatible continuation prefixes refuse explicitly. A CUDA request never
silently becomes CPU execution.

### Transactional state

Prompt suffixes and generated tokens update persistent model state only after
their owning execution boundary succeeds. Failed or cancelled work preserves
the exact earlier committed prefix. Streamed bytes are sent only after model,
decoder, and internal text commits.

### Explicit lifecycle ownership

The daemon owns one immutable runtime model. Each server session owns mutable
KV, token ledger, transcript, decoder, RNG policy, and turn state. A client
connection is neither of those lifetimes, so disconnect does not imply model
close or session reset.

### Scoped evidence

Software tests, numerical conformance, runtime qualification, component
measurements, model evaluation, full-model benchmarks, and release
qualification are different evidence classes. A lower class never promotes a
higher capability.

## Current vertical

DeepSeek-V4-Flash is the sole complete model-to-text vertical in the current
v0.1 line.

Implemented facts include:

- exact verification of the pinned source snapshot and tokenizer material;
- complete source coverage and an immutable artifact-neutral Transformation IR;
- policy-driven physical compilation over all 1,360 terminal tensors;
- complete YVEX-produced source-faithful, Q8_0/Q2_K, and mixed
  IQ2_XXS/Q2_K GGUF variants outside the repository;
- variant-adaptive materialization and identity-bound runtime bindings;
- the 43-layer Transformer, persistent DeepSeek state, MoE, logits over the
  complete 129,280-token vocabulary, sampling, and exact tokenizer execution;
- complete prompt-to-text generation on CPU and the admitted mixed GB10 path;
- a long-lived local host with streaming one-shot and exact multi-turn sessions;
- a bounded OpenAI-compatible application gateway over the same hosted model.

The CUDA product path executes the model backbone and output head on CUDA while
sampling, tokenizer work, protocol handling, and orchestration remain on the
host. YVEX does not call that GPU-resident end-to-end generation.

No physical variant is selected as the release profile merely because it is
smaller or executable. Evaluation and the full-model benchmark own that later
decision. Exact artifacts, sizes, identities, and current gates are recorded in
[`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md) and [`PROJECT.md`](PROJECT.md).

## Build

Build the engine, daemon, product client, application gateway, and developer
tools:

```sh
make -j4 all
```

Run the repository checks twice when validating a topology or generated-
dependency change:

```sh
make -j2 check
make -j2 check
```

Validate the admitted CUDA build separately:

```sh
make check-cuda
```

Build a product package manifest and staged package:

```sh
make package
```

Model weights, complete GGUF files, runtime bindings, logs, traces, generated
text, benchmark output, and build products remain untracked. The full validation
and hygiene workflow is in the [common runbook](docs/runbooks/common.md).

## Documentation

### Operate YVEX

- [Operator runbook](docs/operator-runbook.md): explicit first startup, the
  three-terminal workflow, sessions, status, shutdown, and recovery.
- [DeepSeek runbook](docs/runbooks/deepseek.md): exact model-specific product
  and developer workflows.

### Understand YVEX

- [Reference architecture](docs/reference-architecture.md): complete planes,
  identities, lifecycles, execution, state, and evidence model.
- [Model families](docs/model-families.md): family integration and DeepSeek
  semantics.
- [Model artifacts](MODEL_ARTIFACTS.md): artifact terminology, admission, and
  support boundary.

### Integrate YVEX

- [C API](docs/api.md): installed and internal interfaces and lifetimes.
- [Runtime contract](docs/contract.md): admission, publication, failure,
  cleanup, protocol, and server behavior.
- [OpenAI compatibility](docs/openai-compatibility.md): exact local endpoints,
  request fields, streaming events, SDK profile, and explicit refusals.

### Develop YVEX

- [Repository rules](AGENTS.md): source ownership, boundaries, tests, and claim
  discipline.
- [System target](docs/system-target.md): filesystem and module ownership.
- [Validation runbook](docs/runbooks/common.md): build, checks, CUDA, and
  artifact hygiene.
- [Client and terminal architecture](docs/cli-output-architecture.md): product,
  developer, raw, operational, and conversation surfaces.

### Track YVEX

- [Project state](PROJECT.md): the sole live milestone, capability, gate, and
  Active Next authority.
- [Release doctrine](docs/v010-release-doctrine.md): v0.1 gate meanings and
  closure rules.

## Current limits

YVEX does not currently claim:

- a public or remote production server;
- the full OpenAI API, Anthropic compatibility, authentication, TLS, or remote
  security;
- multi-model hosting, hot model reload, continuous batching, or distributed
  serving;
- session persistence across daemon restart;
- GPU-resident sampling, tokenizer execution, or complete device-side
  orchestration;
- model behavior or model quality evaluation;
- a release-grade full-model benchmark;
- release qualification;
- MTP or speculative execution;
- a second complete model-family vertical.

The local protocol is private and UID-scoped. Operational timings are useful
runtime facts, but they are not evaluation or release benchmark evidence.

## License

YVEX is licensed under [`LICENSE`](LICENSE). Third-party notices are recorded
in [`NOTICE.md`](NOTICE.md).
