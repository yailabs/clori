<p align="center">
  <img src="docs/yvex-primary-lockup.svg" alt="YVEX logo" width="320">
</p>

YVEX is a native C/CUDA model compiler and local inference system for
identity-bound, verified open-weight execution. It turns authenticated source
facts into immutable model packages, specializes those packages for an admitted
machine, and serves isolated sessions through one persistent local host.

DeepSeek-V4-Flash-DSpark is the v0.1 text target on CPU and the admitted mixed
NVIDIA GB10 path. Qwen's admitted text specialization and MiniMax-H3 FL2VA's
composite media execution exercise the same host, session, resource, and backend
owners. Mamba2 currently reaches source admission and recurrent-state component
evidence only: it is not a loadable model. Performance optimization, evaluation,
release benchmarking, and release qualification remain open.

Start with the [quick start](#quick-start) to operate an admitted model, the
[architecture overview](#architecture-and-lifetimes) to understand the ownership
boundaries, or [development and validation](#development-and-validation) to
contribute. The [operator runbook](docs/operator-runbook.md) contains the full
procedures; [`ROADMAP.md`](ROADMAP.md) owns live project and release state.

## Why YVEX

Open weights are not executable merely because their bytes can be read. YVEX
makes each authority and lifetime explicit:

```text
verified source -> semantic model -> operator schedule -> physical package
  -> artifact + runtime binding
  -> deployment specialization -> model-engine generation
  -> transactional session state -> executable batch -> backend
  -> committed text or media result
```

Package meaning does not change because a different equivalent CUDA tile wins.
Deployment-significant numerical choices remain identity-bound. Dynamic rows,
resource availability, and sequence state remain dynamic. Unsupported qtypes,
stale bindings, stale engine generations, missing resources, unsafe paths, and
unavailable exact implementations fail closed.

## What is actually implemented

YVEX distinguishes a discovered checkpoint, a verified source, an admitted
artifact, a READY deployment, and an executing model. These are different
evidence stages, not interchangeable labels for support.

| Model boundary | Current evidence | Important limit |
| --- | --- | --- |
| DeepSeek-V4-Flash-DSpark | Complete source-to-text vertical; target-only and target-verified speculative execution through the persistent host | The admitted physical representation and backend matter; operational generation is not release-grade quality or performance qualification |
| Qwen3.8-27B | Admitted text path with full attention, recurrent/convolution state, retained sessions, and hosted generation | Text-only capability; the presence of other source components does not qualify image, audio, or video input |
| MiniMax-H3 FL2VA | Admitted composite media path with conditioning, iterative state, and typed media output on the common runtime | Component evidence and complete trajectory evidence remain distinct; see the exact [family record](docs/model-families/minimax-h3.md) |
| Mamba-Codestral-7B-v0.1 / Mamba2 | Pinned product acquisition, exact BF16 tensor roles, transactional recurrent-state geometry, and portable CPU selective-SSD component numerics | **PARTIAL:** no complete artifact or decoder, no READY deployment, no model load, and no hosted generation or chat |

For Mamba2, tokenizer/special-token and normalization authority remain
unresolved, and the compiled decoder still needs a truthful SSM-only topology.
`model prepare` refuses promotion rather than creating an executable descriptor
from source recognition. The [Mamba2 record](docs/model-families/mamba2.md)
states the pinned source, numerical evidence, and remaining boundary precisely.

Availability is determined by the current deployment and machine, not this
table alone. Use `model show` for lineage and blockers, and `model active` for
actual engines. A successful download does not imply that preparation or
execution is implemented for that checkpoint.

## Quick start

### 1. Build

The native build uses a C11 compiler, GNU Make, Python 3 for generated
registries, and the platform development libraries used by the Makefile
(including zlib, threads, dynamic loading, and math). CUDA execution additionally
requires a compatible NVIDIA driver and CUDA toolkit. The provider acquisition
surface uses the installed provider tooling; Hugging Face access uses `hf`.
Model weights are acquired separately and are never included in the repository.

```sh
make info
make -j4 all
./yvex help
```

The Makefile owns build configuration and reports the detected CUDA path.
Building without `nvcc` does not qualify CUDA execution; an exact unavailable
backend request is refused. For CUDA diagnostics, use `make cuda-info` and the
product's `./yvex inspect cuda` projection. Hardware-specific admission and
measurement requirements are documented under [GB10 targets](docs/development/gb10-targets.md).

Running `./yvex` prints the compact product command map and exits. Interactive
generation is always explicit through `./yvex chat`; it is a native client of
an already-running host and never opens weights or initializes CUDA itself.

### 2. Find, pull, and prepare a model

```sh
./yvex model search "MODEL"
./yvex model pull hf://OWNER/REPOSITORY --format safetensors --dry-run --verbose
./yvex model pull hf://OWNER/REPOSITORY@IMMUTABLE_REVISION --format safetensors
./yvex model list
./yvex model show MODEL
./yvex model prepare MODEL
```

`OWNER/REPOSITORY`, `IMMUTABLE_REVISION`, and `MODEL` are placeholders: use the
repository returned by search, the revision resolved by the dry run, and the
logical model name reported by the local catalog. These examples do not imply
that an arbitrary Hugging Face model has a YVEX compiler implementation.

The product workflow is `search -> pull -> prepare -> load -> chat`. Search is
discovery only. Pull resolves an immutable provider revision and acquires or
records one selected representation; it never loads the runtime. If a
repository offers several representation classes, a terminal shows a linear
selector, while automation supplies `--format` and, when necessary,
`--variant`. Local files and directories use the same command:

```sh
./yvex model pull /mnt/models/MODEL --managed
./yvex model pull /mnt/external/MODEL --reference
./yvex model pull hf://OWNER/REPOSITORY --format gguf --prepare
```

`--managed` makes a verified copy in YVEX-owned storage; `--reference` records
an exact external dependency without duplicating its bytes. `model list` is one
logical-model catalog, not a registry dump. It exposes origin, selected format,
quantization or precision, size, product state, execution backend, variant
count, and location. `model show` expands the exact source, artifact, profile,
and loaded-generation lineage. Unsupported family compilation or provider
transport fails explicitly; a source record is never relabelled READY merely
because its bytes exist.

When a repository contains both consolidated and sharded Safetensors payloads,
acquisition selects one coherent representation; it does not acquire every
large file merely because the extension matches. Source verification separately
checks the selected index, tensor inventory, and authoritative payload hashes.
The dry run resolves metadata and planned files without changing acquisition
state or downloading weights.

Inspect provider access without exposing credentials:

```sh
./yvex source accounts providers --output table
./yvex source accounts status --output table
./yvex source accounts whoami huggingface --output table
```

Public repositories do not require changing credentials merely to run this
workflow. Gated or private access must be established through the existing
provider account owner. Never place tokens in tracked configuration or logs.

Long transfers retain their own operation lifecycle:

```sh
./yvex model status MODEL
./yvex model stop MODEL
./yvex model pull hf://OWNER/REPOSITORY@IMMUTABLE_REVISION --format safetensors --resume
```

Resume preserves completed selected files; partial byte-range reuse depends on
the provider implementation. A transfer completing does not load a model.

### 3. Start the persistent host

If `./yvex host status` already reports a ready host, use it. Otherwise run
this in the first terminal:

```sh
./yvex serve
```

The host publishes its private socket and optional loopback OpenAI listener with
zero loaded engines. It remains alive across model load and unload, and its
foreground terminal shows the one-time boot report followed by host logs. It
never reads stdin as an administrative REPL. Invoking `./yvex serve` again
refuses the duplicate host and exits without attaching as a client.

### 4. Load a model

From another terminal, select a launchable model and representation:

```sh
./yvex model load
./yvex model list --wide
./yvex host status
```

With no argument, `model load` displays an ordinary line-oriented selector; it
never asks the user to copy a profile alias or export a shell variable. A
script uses `model load MODEL` and adds `--variant VARIANT` only when several
valid representations exist. Internally YVEX still resolves the exact artifact
and deployment profile and creates one process-local engine generation.
Advanced operators can inspect that generation with `engine list --json`.
`./yvex model active` shows only the current active engine set, including exact
generations, clients, leases, directional capabilities, and resource facts.

Host readiness and model readiness are separate. A healthy zero-engine host is
normal. READY means the deployment is currently compatible with its admitted
binding and implementation; LOADED means a process-local engine generation
exists. Restarting the host does not restore old engine or session handles.

### 5. Use the engine

```sh
./yvex chat
```

The client session remains bound to that exact engine generation. Type the
prompt in the linear REPL. If exactly one text model is loaded it is selected
automatically. With several loaded text models, a terminal shows a model
selector; automation uses `./yvex chat --model MODEL`. Programmatic inference
uses an admitted provider/protocol surface rather than a second one-shot CLI
generation command. The prompt shows the product model name, while the startup
summary retains the exact variant/backend facts. Typed reasoning is rendered as
muted secondary text and the final channel begins at an explicit `answer`
boundary; a bounded Markdown projection and 112-column prose measure change
presentation only. Omitting `--max-new-tokens` leaves completion-envelope
selection to the host, while an explicit value remains a hard user bound and
the terminal summary names the resulting stop reason.
Use `/attach PATH` repeatedly to stage local media for the next ordered
multipart turn; `/attachments` inspects the stage and `/attachments-clear`
drops it. Attachments never create a session or select another model.

### 6. Inspect and release resources

Unload the model without stopping the host, or stop the host separately:

```sh
./yvex model active --json
./yvex host memory
./yvex host logs
./yvex model unload MODEL
./yvex host stop
```

Unload refuses to destroy a generation still required by live sessions or model
leases. Finish or release those consumers through their owning lifecycle first.
Disconnecting a telemetry subscriber is not an engine unload, and leaving one
chat client is not a command to stop the host.

Discover the current command surface without parsing documentation:

```sh
./yvex help
./yvex help --advanced
./yvex help --json
```

The [operator runbook](docs/operator-runbook.md) owns complete lifecycle,
session, observation, memory, and recovery procedures.

## Architecture and lifetimes

The compiler terminates source-specific interpretation before numerical
execution. Runtime consumes an authenticated binding and prepared deployment;
it does not reopen a provider inventory or choose topology from a model name.

| Authority | Owns | Does not imply |
| --- | --- | --- |
| Source snapshot | Provider revision, provenance, file and tensor inventories, ranges, and trust | An executable artifact |
| Family interpretation | Architecture facts, canonical roles, topology/state meaning, numerical obligations | Its own runtime, allocator, or session manager |
| Compilation | Semantic lowering, operator schedules, physical package legality, and immutable bindings | Live device availability |
| Artifact | Package identity, admission, mapping, and lifecycle | A currently compatible deployment |
| Deployment specialization | Admitted backend and hardware-significant implementation choices | A loaded engine |
| Engine generation | Executable resources, prepared model state, and stale-reference boundaries | Ownership of a conversation's mutable state |
| Session | Mutable sequence/component state and transactional continuity | A second copy of immutable model weights |
| Scheduler and backend | Ready-work progress, real execution populations, submission, buffers, and equivalent kernels | Global continuous batching or a family-specific execution service |

Sequence state is not synonymous with KV. Attention, recurrent state,
convolution history, draft state, and other typed components share lifecycle
coordination without sharing storage geometry. Transactions stage candidate
state and publish it at the owning commit boundary; failure, cancellation,
reset, and cleanup must preserve isolation and committed truth.

Physical implementation choices remain separate from model semantics. A
quantized package is a derived representation of a source model, not a new
source identity. Backends execute admitted operations and select equivalent
hardware details; they do not reconstruct family topology from tensor names.

See [compilation](docs/architecture/compilation.md),
[runtime](docs/architecture/runtime.md), and the
[family-integration contract](docs/model-families/integration.md) for the
normative boundaries and promotion obligations.

## Multiple models and typed conversation content

The selected conversational model, active engines, sessions, and model leases
are independent concepts. Several different engines can coexist when resource
admission permits. An explicit provider ensure-active request can reuse an
existing generation or invoke the existing loader for a READY deployment and
return a lease. It does not replace a primary session's selected model.

Leases name an exact engine generation and block unload while held. There is no
parallel auxiliary loader, family-name placement rule, or opaque automatic LRU.
If resources do not permit activation, admission refuses the request rather
than silently evicting another live consumer. `model active` and its JSON form
derive from the same typed runtime authority; an orchestrator never needs to
scrape the human table.

One turn can contain ordered text, image, audio, video, file, and tensor parts.
The representation carries content identity, typed metadata, and optional
derived-from provenance. Original audio and a transcript derived from it remain
distinct parts. Supporting the representation does not claim that every model
can consume each modality: input and output capabilities are qualified
independently for the admitted specialization.

The reference chat client stages local files for one upcoming turn:

```text
/attach /absolute/path/image-one.png
/attach /absolute/path/image-two.png
/attachments
Describe these inputs.
```

This illustrates staging, not qualified image understanding by a text-only
model. Unsupported combinations are rejected before numerical execution. Later
turns can stage different files in the same session; `/attachments-clear`
clears the stage without resetting the conversation. Native local transfer
uses bounded binary content or authenticated local-file references, not a
requirement to expand large media as base64 JSON. File references are private
local transport, not permission to expose arbitrary server paths remotely.

YVEX owns typed execution, capability publication, admission, engine/resource
truth, and execution explicitly requested by a provider client. YAI or another
application owns capture/presentation, conversation policy, primary/auxiliary
roles, and the decision to request a capability. Microphone capture, webcam
controls, application routing, and agent orchestration are outside this CLI.
Transport plumbing alone does not qualify an audio, vision, or retrieval model.

## Reading execution and memory truth

Use current snapshots for current state and events for chronology:

```sh
./yvex host status --json
./yvex model active --json
./yvex host memory
./yvex host logs --follow
```

`host logs --json` exposes the canonical event records. Progress comes from the
operation owner. Verification bytes or inspected tensors can have real
completed/total denominators; a phase without a known total reports activity
and elapsed time, not an invented percentage. Replaceable progress is bounded
and coalesced under pressure, with explicit loss accounting and lifecycle/
terminal retention priority.

Generation rates name their scope. Cumulative decode (`avg`) is committed
decode work divided by its complete decode wall. Rolling decode (`r32`) uses
recent committed work and its own measured interval, bounded to 32 token
intervals. A declining local tail can therefore be visible while the cumulative
average remains higher. Request-wide counters are not token-local costs, and
overlapping host/device spans are not summed twice. Unmeasured time remains
unavailable or explicitly unattributed.

Capacity also has separate meanings: logical runnable concurrency, physical
execution width, and host/session resource capacity. Cooperative progress by
multiple requests is implemented; ready sequences dynamically joining and
leaving decode batches is not. YVEX does not claim global continuous batching.

Memory reports distinguish immutable mapping, prepared model allocations,
explicit backend allocations, device-addressability, typed session state,
activation arenas, workspace, transients, and process RSS, with current/peak and
availability facts. These classes are not all additive. On unified-memory
hardware, zero explicit device allocation does not mean zero GPU working set:
mapped weights may be device-addressable while physical page residency remains
unknown or not measured.

Normal operational observability and detailed profiling have different costs.
Controlled comparison holds execution/output identity and workload fixed while
recording the observability condition separately. Detailed attribution is not
an ordinary benchmark mode, and terminal throughput is not a release benchmark.
The [events and telemetry contract](docs/contracts/events-telemetry.md) owns
the exact measurement and retention semantics.

## Product boundary

| Component | Responsibility |
| --- | --- |
| `yvex` | Persistent host, native/OpenAI clients, REPL, model lifecycle controls, and bounded offline compilation/inspection/execution operations |
| `libyvex` | Reusable source, compilation, package, engine, runtime, graph, backend, tokenizer, generation, and media implementation |

Runtime-facing native clients cross private local protocol v20. The `serve`
entrypoint owns host lifetime in the same executable; engine commands own
loaded-generation lifecycle through that protocol, and client handlers do not
open weights or initialize CUDA. Finite offline operations close their engine
resources before exit and never create another persistent host.

One compiled operation registry drives command paths, syntax, help, JSON
discovery, completion, and REPL slash schemas without becoming a capability
authority. The console renders server-authored committed channels and typed
metrics; it never promotes draft candidates, control events, or inferred
reasoning into model output.

Live milestone and release-gate state remains only in
[`ROADMAP.md`](ROADMAP.md).

## Development and validation

Implementation lives under `src/`, installed headers under `include/yvex/`,
and tests under `tests/`. `config/source_owners.tsv` is the single production
membership authority. Operator grammar and QA identities likewise have
canonical registries; generated build projections are not edited by hand.

Read [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before editing.
Inspect branch, HEAD, index, and the complete local diff in a shared worktree.
Preserve unrelated changes and published history; use focused semantic commits
and ordinary merges instead of rewriting a development epoch.

Resolve validation from the changed owners, using the actual comparison commit
in place of `BASE`:

```sh
python3 tools/qa.py doctor
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
python3 tools/qa.py report latest
git diff --check
```

For quick iteration, `python3 tools/qa.py run fast` runs the bounded lane, but
does not replace required numerical, transactional, sanitizer, architecture,
ABI/protocol, CUDA, or live qualification. Build topology changes also require
two consecutive builds without cleaning. Missing mandatory assets or hardware
are BLOCKED, never a successful test.

Structured evidence and per-test logs remain under ignored `build/qa/` paths.
The source snapshot must remain unchanged while a run executes. GPU and mutable
build-tree resources have their own exclusive locks; do not run competing
qualification against them. Keep model weights, artifacts, downloaded
dependencies, credentials, raw traces, and local registries outside Git.

Software contracts, independent numerical conformance, runtime qualification,
component measurements, model evaluation, and release qualification have
different evidence ranks. Passing one does not automatically close the next.
See the [QA contract](docs/development/qa.md) for exact lane selection and
report semantics.

## Documentation

- [Documentation map](docs/README.md) — canonical owners and navigation.
- [Implemented system](docs/architecture/system.md),
  [compilation](docs/architecture/compilation.md), and
  [runtime](docs/architecture/runtime.md) — current architecture.
- [Family integration](docs/model-families/integration.md),
  [DeepSeek](docs/model-families/deepseek-v4-flash.md),
  [MiniMax](docs/model-families/minimax-h3.md), and
  [Mamba2](docs/model-families/mamba2.md) — family boundaries, evidence, and
  explicit non-executable stages.
- [Runtime](docs/contracts/runtime.md),
  [artifact](docs/contracts/artifacts.md),
  [local protocol](docs/contracts/local-protocol.md), and
  [OpenAI compatibility](docs/openai-compatibility.md) — normative contracts.
- [Engineering contract](AGENTS.md),
  [QA evidence](docs/development/qa.md), and
  [source ownership](docs/development/source-ownership.md) — executable
  repository invariants.
- [Contributing](CONTRIBUTING.md),
  [roadmap](ROADMAP.md), and [release target](docs/releases/v0.1.md).

## Current limits

YVEX does not currently claim:

- public or remote serving, authentication, TLS, or remote security;
- complete OpenAI API or another provider compatibility surface;
- arbitrary checkpoint execution merely because source acquisition succeeds;
- complete Mamba2 decoder/artifact, READY deployment, or hosted generation;
- image/audio/video understanding by a text-only specialization;
- global ready-sequence continuous batching or distributed serving;
- a retained optimized selective DeepSeek weight layout or automatic
  prepared-resource eviction policy;
- restart-persistent engine instances or automatic resource-driven engine swap;
- complete accelerator residency, device-side stochastic sampling, or tokenizer execution;
- optimized DSpark execution or production load-aware confidence scheduling;
- model behavior/quality evaluation or a release-grade full-model benchmark;
- release qualification.

Several fitting engines can coexist, but resource admission may permit only one
huge model at a time on GB10. Operational timings are runtime facts, not
evaluation or release benchmark evidence. A complete artifact is not a
supported artifact until every later gate closes.

## License

YVEX is licensed under [`LICENSE`](LICENSE). Third-party notices are in
[`NOTICE.md`](NOTICE.md). Public changes are recorded in
[`CHANGELOG.md`](CHANGELOG.md).
