<p align="center">
  <img src="docs/logo.svg" alt="YVEX logo" width="132">
</p>

# YVEX

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

### 2. Find or prepare a model profile

```sh
./yvex model list
./yvex model show deepseek4-v4-flash-dspark-runtime-iq2xxs
```

Use an alias printed by `model list` whose startup profile is complete. If no
suitable profile exists, discover a remote representation before acquiring an
exact revision:

```sh
./yvex model search "MODEL"
./yvex model inspect OWNER/REPOSITORY --revision REVISION
./yvex model acquire --repo OWNER/REPOSITORY --family FAMILY \
  --name LOCAL_NAME --revision EXACT_REVISION --include 'PAYLOAD_PATTERN'
```

Acquisition creates local source truth, not a ready engine package. Follow the
runbook's
[discovery and preparation procedure](docs/operator-runbook.md#discover-acquire-and-prepare-a-model)
or, for an already prepared external package, its
[registration procedure](docs/operator-runbook.md#registering-an-existing-model).

### 3. Start the persistent host

If `./yvex server status` already reports a ready host, use it. Otherwise run
this in the first terminal:

```sh
./yvex server
```

The host publishes its private socket and optional loopback OpenAI listener with
zero loaded engines. It remains alive across model load and unload.

### 4. Load and inspect an engine

In a second terminal:

```sh
./yvex server load deepseek4-v4-flash-dspark-runtime-iq2xxs
./yvex server status
./yvex server models
./yvex server memory
```

Loading authenticates the named registry profile and creates one process-local
engine generation. Large packages can take several minutes. `server models`
shows the exact alias, generation, lifecycle, backend, capability, and memory
facts owned by the live host.

### 5. Use the engine

```sh
./yvex chat --model deepseek4-v4-flash-dspark-runtime-iq2xxs --session main
./yvex run --model deepseek4-v4-flash-dspark-runtime-iq2xxs \
  "Explain attention in one sentence."
```

The named session remains bound to that exact engine generation. If exactly one
engine is loaded, `--model` may be omitted. With several loaded engines,
selection must be explicit.

Unload the engine without stopping the host, or stop the host separately:

```sh
./yvex server unload deepseek4-v4-flash-dspark-runtime-iq2xxs
./yvex server stop
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

Runtime-facing clients cross private local protocol v13. The server entrypoint
owns host and engine lifecycle in the same executable; client handlers do not
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
- [Principles](docs/doctrine/principles.md),
  [glossary](docs/doctrine/glossary.md), and
  [evidence discipline](docs/doctrine/evidence.md) — stable doctrine.
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
