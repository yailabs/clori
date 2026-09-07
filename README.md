<p align="center">
  <img src="docs/yvex-primary-lockup.svg" alt="YVEX logo" width="320">
</p>

YVEX is a native C/CUDA model compiler and local inference system for
identity-bound, verified open-weight execution. It compiles authenticated
source facts into immutable packages, specializes them for an admitted
machine, and serves isolated sessions through one persistent local host.

## Why YVEX

A readable checkpoint is not necessarily executable. YVEX separates source
provenance, package meaning, deployment compatibility, engine resources, and
transactional session state. Unsupported implementations, stale identities,
and insufficient resources fail closed instead of silently changing the
requested execution.

The compiler and runtime are developed together: family interpretation states
what a model means, compilation seals legal physical work, and the runtime
executes that work through shared engine, session and backend owners. A new
architecture should extend those owners where necessary, not acquire its own
loader, scheduler or session manager.

This makes YVEX both an operator-facing local inference product and a systems
engineering substrate. Its unit of support is an evidenced source,
representation and deployment—not a model name or a successfully parsed file.

The [system architecture](docs/architecture/system.md) explains these owners.
Code and tests establish capability; documentation describes its limits.

![YVEX system context: source compilation and local clients meet at a persistent host with generation-bound engines, isolated sessions and CPU/CUDA execution.](docs/diagrams/system_overview.svg)

*Figure 1 — System context. Clients and the integrated compatibility adapter
share one host; observation does not own execution. This topology does not
imply general remote serving or equal qualification of every model family.*

## Available execution

| Model boundary | Demonstrated path | Limit |
| --- | --- | --- |
| [DeepSeek V4 Flash](docs/model-families/deepseek-v4-flash.md) | Source-to-text, target-only and target-verified speculative generation | Operational execution is not release quality or performance qualification |
| [Qwen3.8-27B text](docs/model-families/integration.md#current-family-boundaries) | Hybrid attention/recurrent execution and retained hosted text sessions | Text-only; unexecuted source components do not confer vision capability |
| [MiniMax-H3 FL2VA](docs/model-families/minimax-h3.md) | Composite conditioning, iterative execution, and typed synchronized-media publication | Component numerics and a completed trajectory do not prove useful full-model output |
| [Mamba2 / Mamba-Codestral](docs/model-families/mamba2.md) | Pinned acquisition, tensor roles, recurrent-state and CPU component evidence | Partial: no complete artifact, READY deployment, load, or hosted generation |

Use `model show` for actual deployment lineage and blockers, and `model active`
for current engines. This table is not a promise that every checkpoint in a
named family can be prepared on every machine.

## Download tested model artifacts

The first YaiLabs releases are available on Hugging Face:

| Release | Available representations |
| --- | --- |
| [DeepSeek V4 Flash GGUF](https://huggingface.co/yailabs/DeepSeek-V4-Flash-GGUF) | Two mixed representations: IQ2_XXS / Q2_K with Q8_0, or with MXFP4 |
| [Qwen3.8-27B Text GGUF](https://huggingface.co/yailabs/Qwen3.8-27B-Text-GGUF) | BF16 text representation; it does not include all upstream capabilities |

The exact released bytes were exercised through YVEX load → chat → unload on
one recorded NVIDIA DGX Spark configuration. Each model card provides the
checksums, immutable upstream revision, build lineage, license and bounded
validation evidence. This qualification does not imply universal hardware
compatibility or a model-quality benchmark.

See the [model lifecycle guide](docs/model-lifecycle.md#published-yailabs-representations)
for pinned download commands, manual-file adoption and local storage behavior.
Choose one DeepSeek representation; pulling it does not require the other file
or the upstream Safetensors.

## Quick start

### 1. Build

The build requires a C11 compiler, GNU Make, Python 3, pkg-config, current
stable Rust/Cargo, and the platform development libraries selected by the
Makefile, including zlib and threads.
CUDA additionally requires a compatible NVIDIA driver and toolkit. Provider
acquisition uses installed provider tooling; Hugging Face access uses `hf`.
Weights are acquired separately and never included in Git.

Chat statically consumes [pinned REPLAI](docs/decisions/0007-external-terminal-editor.md);
its verified producer is built automatically without a sibling checkout.

```sh
make info
make -j4 all
./yvex help
```

Without `nvcc`, a build does not qualify CUDA execution. See
[contribution setup](CONTRIBUTING.md#set-up-a-development-checkout) for QA
prerequisites and the [runbook](docs/operator-runbook.md) for diagnostics.

### 2. Acquire and prepare

```sh
./yvex model search "MODEL"
./yvex model pull hf://OWNER/REPOSITORY --format safetensors --dry-run --verbose
./yvex model pull hf://OWNER/REPOSITORY@IMMUTABLE_REVISION --format safetensors
./yvex model list
./yvex model show MODEL
./yvex model prepare MODEL
```

Replace the placeholders with the repository found by search, the immutable
revision resolved by the dry run, and the logical model reported by the
catalog. A dry run resolves the selected representation without downloading
weights or changing acquisition state. Pulling a source does not establish a
launchable deployment; prepare refuses unsupported compilation.

Provider accounts, local managed/reference sources, variants, cancellation,
and resume belong to the [operator runbook](docs/operator-runbook.md).

### 3. Serve and chat

If `./yvex host status` reports an existing ready host, use it. Otherwise start
the foreground host in one terminal:

```sh
./yvex serve
```

In another terminal:

```sh
./yvex model load MODEL
./yvex model active
./yvex chat --model MODEL
```

The host starts with zero engines and survives model load/unload. Chat is a
client of that host, not another model runtime. Bare `./yvex` prints help.

Inside chat, `/attach PATH` stages a local object for the next turn; multiple
attachments and later multipart turns retain the same session identity.
Attachments do not grant the selected model new input capabilities.
Unsupported combinations are rejected before numerical execution.

After closing dependent sessions and releasing leases:

```sh
./yvex model unload MODEL
```

### 4. Inspect the running system

```sh
./yvex host status
./yvex host memory
./yvex model list --wide
./yvex model show MODEL
./yvex model active --json
```

The catalog describes known material and deployment readiness. The active
view describes loaded engine generations, including idle engines; it is not
another list of downloaded checkpoints. A host can be healthy with no model
loaded, and unloading a model does not stop the host.

The runbook owns the complete [startup procedure](docs/operator-runbook.md#first-verified-startup),
session recovery, media execution, inspection, and shutdown.

REPLAI owns terminal mechanics; YVEX owns input semantics and result rendering.
The [interactive boundary figure](docs/architecture/system.md#interactive-terminal-path)
shows editing, dispatch, execution and the return to the next prompt.

## Architecture and lifetimes

Source, artifact, deployment, engine and session are different objects with
different lifetimes. This distinction is what lets one persistent host serve
different model architectures without making mutable state global.

| Boundary | What it establishes | What remains separate |
| --- | --- | --- |
| Source and family interpretation | Immutable provenance, tensor roles, topology and state meaning | Executable support must still be proved |
| Compilation and artifact | Legal physical representation, authenticated package and runtime binding | Current hardware compatibility and live resources |
| Deployment | Admitted implementation choices for the current backend/device | An engine need not be loaded |
| Engine generation | Executable model resources and stale-reference protection | Mutable conversation state belongs to sessions |
| Session and transaction | Independent component state, candidate updates and committed continuity | Shared immutable weights are not duplicated per conversation |
| Scheduler and backend | Selected work, buffers, submission and equivalent physical operations | They do not decide model meaning or application intent |

The runtime consumes admitted bindings. Source-specific tensor interpretation
terminates at compilation; backends execute admitted operations rather than
selecting a model architecture from filenames. Incomplete lowering remains an
explicit refusal, as it currently does for the pure-SSM Mamba2 decoder.

Sequence state is not synonymous with KV. Attention state, recurrent state,
convolution history and speculative state have different geometries. Common
transaction machinery coordinates their lifetime and commit/abort boundaries
without pretending that their update rules are identical.

A quantized artifact is a derived physical representation of its source, not
a different source model. A logical relationship between two source variants
does not erase their exact revisions, artifact identities or deployment
selectors. Working-set membership is likewise independent of storage location
and engine residency.

The deeper owners are [compilation](docs/architecture/compilation.md),
[runtime](docs/architecture/runtime.md),
[family integration](docs/model-families/integration.md), and the
[storage contract](docs/contracts/model-storage.md).

## Multiple models and typed turns

The selected conversational model need not be the only loaded model. An
external client can explicitly ensure that a READY model is active and hold a
lease on its engine generation. The same lifecycle owner admits that engine;
there is no auxiliary loader. Live leases and sessions prevent premature
unload, and resource pressure does not authorize silent eviction.

Each turn can carry ordered text, image, audio, video, file or tensor parts.
Content identities and derivation provenance distinguish an original object
from a representation derived from it, such as an audio transcript. Input and
output capabilities belong to the admitted execution specialization: transport
support for audio or images is not a claim that a text model understands them.

The reference client stages attachments for the next turn:

```text
/attach /absolute/path/first.png
/attach /absolute/path/second.png
/attachments
Describe these inputs.
```

This illustrates multipart submission, not qualified vision in the text-only
models above. Later turns can stage different files without replacing the
session. `/attachments-clear` discards the staged parts, not the conversation.
Native local media transport does not require base64-expanded JSON; local
references remain subject to admission and are not arbitrary remote file access.

## Reading execution and memory truth

YVEX exposes typed execution events and resource facts through human and
machine projections. The labels distinguish scopes: `decode-avg` describes
cumulative decode progress, a rolling rate describes a bounded recent window,
and a request total is not a per-token measurement. A declining local decode
rate can therefore remain visible while the cumulative average is still high.

Memory observations distinguish mapped model bytes, explicit backend
allocations, prepared resources, workspace, typed session state, transients
and process RSS. These are not automatically additive totals. In particular,
on unified-memory hardware a device-addressable mapping is not a measurement
of physical GPU page residency, and zero explicit device allocation does not
mean zero GPU-used memory.

Normal operational observation and detailed profiling have different costs.
Characterization must retain the artifact/binding, workload, backend,
observability configuration and warm/cold identity needed to repeat it. A
component timing, an engineering target and a whole-model benchmark are
different evidence classes.

See [execution and resource contracts](docs/contracts/runtime.md) and
[event and telemetry semantics](docs/contracts/events-telemetry.md) for precise
scopes, overlap and unavailable measurements.

## Product boundary

YVEX owns typed execution, published model capabilities, engine availability,
resource admission, and transactional results. Multiple engines may coexist
when resources allow; explicit auxiliary leases do not replace a primary
session's selected model.

Application policy, user media capture, provider routing, cognitive roles,
and tool execution belong to consumers. The reference CLI is a linear
provider client, not an application harness. HTTP clients use the bounded
[OpenAI compatibility profile](docs/openai-compatibility.md), not an implied
implementation of every upstream API.

REPLAI provides editing mechanics. A private terminal adapter isolates current
POSIX interrupt capture, terminal observation and temporary output-state
restoration from YVEX command and cancellation semantics. This interface
separation does not claim a qualified Windows or macOS product build; the
pinned dependency and local transport have their own platform constraints.

## Development and validation

Engineering starts by reconciling the repository and understanding current
owners. Changes earn promotion through negative admission, numerical,
lifecycle and product evidence appropriate to the boundary being changed.
Green software tests alone do not establish whole-model quality or release
readiness.

The QA catalog maps changed owners to their required checks:

```sh
git diff --check
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
```

Replace `BASE` with the commit preceding the delivery. GPU/live lanes are
separate from routine checks and require exact admitted assets and exclusive
hardware access. Weights, generated packages, raw profiles and local registries
stay outside Git. The [QA contract](docs/development/qa.md) owns validation;
the [engineering method](docs/development/agentic-engineering.md) explains how
evidence selects the next boundary. [ROADMAP](ROADMAP.md) owns current direction.

## Current limits

- Family support is evidence- and representation-specific. Mamba2 preparation
  still refuses READY; MiniMax's full-scale numerical/behavioral boundary is
  open.
- Cooperative runnable concurrency is not continuous batching. Physical
  execution width and engine/resource capacity are separate facts.
- UMA device-addressability does not prove physical page residency. Explicit
  device allocation is not the total GPU working set.
- Model behavior evaluation, release benchmarking, and further family release
  qualification remain open. The downloadable releases above have bounded
  artifact-specific lifecycle evidence; characterization numbers are not
  quality or performance claims.
- CPU and admitted CUDA paths exist; other backends are not implied.

## Documentation

- [Documentation map](docs/README.md): current owners and reading routes.
- [ROADMAP](ROADMAP.md): accepted macro state and next project boundary.
- [Family integration](docs/model-families/integration.md): evidence ladder
  from source interpretation to executable model.
- [Contributing](CONTRIBUTING.md), [AGENTS](AGENTS.md), and
  [engineering method](docs/development/agentic-engineering.md): contribution
  workflow, mandatory agent rules, and evidence-driven progression.
- [Support](SUPPORT.md), [security](SECURITY.md), [license](LICENSE), and
  [notices](NOTICE.md): assistance, trust boundary, and attribution.
