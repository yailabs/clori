<p align="center">
  <img src="docs/logo.svg" alt="YVEX logo" width="132">
</p>

# YVEX

YVEX is a native C/CUDA model compiler and local inference runtime for
identity-bound, verified open-weight execution. It derives explicit physical
variants from verified source snapshots, admits complete artifacts, and runs
them through one long-lived host with isolated sessions.

The current complete vertical is DeepSeek-V4-Flash on CPU and the admitted
mixed NVIDIA GB10 CUDA path. `yvexd` keeps one model open; `yvex` provides the
public command surface; local applications may use the bounded
`yvex.openai.compat.v1` HTTP/SSE profile in the same daemon. Evaluation,
release benchmarking, and release qualification remain open.

## Why YVEX

Open weights are not executable merely because their bytes can be read. YVEX
makes the entire derivation and execution chain explicit:

```text
verified source snapshot
  -> logical model and transformation plan
  -> policy-selected physical variant
  -> complete artifact and runtime binding
  -> immutable runtime model
  -> isolated session state
  -> admitted CPU/CUDA execution
  -> committed streamed text
```

Each boundary has a distinct identity, owner, lifecycle, failure contract, and
evidence scope. Unsupported qtypes, stale bindings, missing resources, unsafe
paths, capacity violations, and unavailable CUDA operations fail closed.
Persistent state commits transactionally, and a failed or cancelled request
preserves exact earlier progress.

![YVEX system overview: verified sources become an identity-bound artifact served by one long-lived runtime host to isolated sessions and local clients.](docs/diagrams/system_overview.svg)

## Quick start

Build the two product executables:

```sh
make -j4 all
```

Provide one admitted complete GGUF artifact and its exact runtime binding as
external operator assets:

```sh
export YVEX_MODEL_ARTIFACT=/absolute/model.gguf
export YVEX_RUNTIME_BINDING=/absolute/model.yvex-runtime-binding
```

Start the resident host and wait for `runtime.ready`:

```sh
./yvexd --model "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096 --console raw --trace-level stages --openai on --openai-port 8001
```

In another terminal, open a retained session:

```sh
./yvex chat --session main
```

Or run one ephemeral streamed turn against the same resident model:

```sh
./yvex run "Explain attention in one sentence."
```

Inspect the runtime and discover commands without parsing prose:

```sh
./yvex runtime status
./yvex help
./yvex help --advanced
./yvex help --json
```

The [operator runbook](docs/operator-runbook.md) owns complete startup,
three-terminal observation, sessions, shutdown, configuration, and recovery.

## Product boundary

| Component | Responsibility |
| --- | --- |
| `yvex` | Public REPL, one-shot and administrative protocol client, plus finite offline compile, artifact, inspect, execute, profile, and system operations |
| `yvexd` | One long-lived model, worker, queue, session/KV registry, private protocol, loopback OpenAI adapter, and telemetry authority |
| `libyvex` | Reusable compilation, artifact, runtime, graph, backend, tokenizer, and generation implementation |

Runtime-facing `yvex` operations always cross private local protocol v4. The
finite offline lane may link engine owners but never hosts a persistent model.
One compiled operation registry drives command paths, syntax, help, JSON
discovery, completion, and slash schemas without becoming a domain-policy
owner.

The current REPL is functional but transitional. A mature daemon-backed
console with semantic progress, complete typed metrics, watch, and human trace
is the next project boundary; current state is recorded only in
[`ROADMAP.md`](ROADMAP.md).

## Documentation

- [Documentation map](docs/README.md) — canonical owners and navigation.
- [YVEX principles](docs/doctrine/principles.md) and
  [glossary](docs/doctrine/glossary.md) — stable thesis and terminology.
- [Verified inference reference](docs/reference/verified-inference.md) —
  implementation-independent architecture.
- [Implemented system](docs/architecture/system.md),
  [compilation](docs/architecture/compilation.md), and
  [runtime](docs/architecture/runtime.md) — current YVEX architecture.
- [DeepSeek-V4-Flash record](docs/model-families/deepseek-v4-flash.md) — exact
  family facts and present evidence boundary.
- [Runtime](docs/contracts/runtime.md),
  [artifact](docs/contracts/artifacts.md),
  [local protocol](docs/contracts/local-protocol.md), and
  [OpenAI compatibility](docs/openai-compatibility.md) — normative contracts.
- [Contributing](CONTRIBUTING.md) and
  [documentation policy](docs/development/documentation-policy.md) —
  repository workflow and information governance.
- [Roadmap](ROADMAP.md), [changelog](CHANGELOG.md), and
  [v0.1 readiness](docs/releases/v0.1.md) — current sequence, public changes,
  and release scope.

## Current limits

YVEX does not currently claim:

- public or remote serving, authentication, TLS, or remote security;
- complete OpenAI API or another provider compatibility surface;
- multi-model hosting, hot reload, continuous batching, or distributed serving;
- session persistence across daemon restart;
- complete accelerator residency, device-side sampling, or tokenizer execution;
- MTP or speculative execution;
- model behavior or quality evaluation;
- a release-grade full-model benchmark;
- a second complete model-family vertical;
- release qualification.

Operational timings are runtime facts, not evaluation or release benchmark
evidence. A complete artifact is not a supported artifact until every later
gate closes.

## License

YVEX is licensed under [`LICENSE`](LICENSE). Third-party notices are in
[`NOTICE.md`](NOTICE.md). Public changes are recorded in
[`CHANGELOG.md`](CHANGELOG.md).
