<p align="center">
  <img src="docs/yvex-primary-lockup.svg" alt="YVEX logo" width="320">
</p>

YVEX is a native C/CUDA model compiler and local inference system for
identity-bound, verified open-weight execution. It turns authenticated source
facts into immutable model packages, specializes those packages for an admitted
machine, and serves isolated sessions through one persistent local host.

DeepSeek-V4-Flash-DSpark is the v0.1 text target on CPU and the admitted mixed
NVIDIA GB10 path. MiniMax-H3 FL2VA is also an admitted composite media vertical
on the same generic host/runtime/backend substrate. Performance optimization,
evaluation, release benchmarking, and release qualification remain open.

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

## Quick start

### 1. Build

```sh
make -j4 all
```

Running `./yvex` prints the compact product command map and exits. Interactive
generation is always explicit through `./yvex chat`; it is a native client of
an already-running host and never opens weights or initializes CUDA itself.

### 2. Find, pull, and prepare a model

```sh
./yvex model search "MODEL"
./yvex model pull hf://OWNER/REPOSITORY
./yvex model list
./yvex model show MODEL
./yvex model prepare MODEL
```

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

Unload the model without stopping the host, or stop the host separately:

```sh
./yvex model unload MODEL
./yvex host stop
```

Discover the current command surface without parsing documentation:

```sh
./yvex help
./yvex help --advanced
./yvex help --json
```

The [operator runbook](docs/operator-runbook.md) owns complete lifecycle,
session, observation, memory, and recovery procedures.

## Product boundary

| Component | Responsibility |
| --- | --- |
| `yvex` | Persistent host, native/OpenAI clients, REPL, model lifecycle controls, and bounded offline compilation/inspection/execution operations |
| `libyvex` | Reusable source, compilation, package, engine, runtime, graph, backend, tokenizer, generation, and media implementation |

Runtime-facing native clients cross private local protocol v17. The `serve`
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

## Documentation

- [Documentation map](docs/README.md) — canonical owners and navigation.
- [Implemented system](docs/architecture/system.md),
  [compilation](docs/architecture/compilation.md), and
  [runtime](docs/architecture/runtime.md) — current architecture.
- [Family integration](docs/model-families/integration.md),
  [DeepSeek](docs/model-families/deepseek-v4-flash.md), and
  [MiniMax](docs/model-families/minimax-h3.md) — family boundaries and evidence.
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
