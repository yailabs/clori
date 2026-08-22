<p align="center">
  <img src="docs/logo.svg" alt="YVEX logo" width="132">
</p>

# YVEX

YVEX is a native C/CUDA model compiler and local inference runtime for
identity-bound, verified open-weight execution. It derives explicit physical
variants from verified source snapshots, admits complete artifacts, and runs
them through one long-lived host with isolated sessions.

The current complete vertical is DeepSeek-V4-Flash-DSpark on CPU and the
admitted mixed NVIDIA GB10 CUDA path. One foreground `yvex server` process owns
target-only and target-verified speculative generation; `yvex` also provides
the client and offline command surfaces. Local applications may use the bounded
`yvex.openai.compat.v2` HTTP/SSE profile in the same server. Performance optimization, evaluation,
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
  -> target execution and optional verified DSpark proposal
  -> committed streamed text
```

Each boundary has a distinct identity, owner, lifecycle, failure contract, and
evidence scope. Unsupported qtypes, stale bindings, missing resources, unsafe
paths, capacity violations, and unavailable CUDA operations fail closed.
Persistent state commits transactionally, and a failed or cancelled request
preserves exact earlier progress.

![YVEX system overview: verified sources become an identity-bound artifact served by one long-lived runtime host to isolated sessions and local clients.](docs/diagrams/system_overview.svg)

## Quick start

### 1. Build YVEX

```sh
make -j4 all
```

### 2. Find a registered model

The model registry owns the artifact path, runtime binding, target, backend,
and startup context. Normal startup does not require environment variables or
model paths. List the local entries and inspect one whose `STARTUP` column is
`yes`:

```sh
./yvex model list
./yvex model show deepseek4-v4-flash-dspark-runtime-iq2xxs
```

The alias above is an example; use an alias printed by `model list`. If no
startup-ready model is listed, follow the runbook's one-time
[model registration](docs/operator-runbook.md#registering-an-existing-model)
procedure. The profile name is supplied explicitly when the server starts.

### 3. Run the model server

If `./yvex server status` already reports `ready`, do not start a second host;
continue with step 4. Otherwise, run this in the first terminal:

```sh
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs
```

There is no separate model-load command. `server MODEL` directly enters the
foreground server lifecycle, authenticates the named profile's artifact and
binding, copies the encoded weights into its process-lifetime host arena,
builds runtime residency, and keeps that runtime model open. Before admission
starts it prints the profile, target, backend, mode, requested context,
artifact, binding, endpoint, and stop instruction. Use `--ctx N` for an
explicit startup capacity override and `--parallel N` for capacity-admitted
independent-session scheduling. Compatible sessions can share the current typed
MoE and output-head execution-batch boundaries; attention and sampling remain
session-local.
Prefill chunking is adaptive unless a positive `--prefill-chunk N` override is
supplied. Leave that terminal running. A large model
can take several minutes before the server becomes ready.

### 4. Verify the resident server

In a second terminal:

```sh
./yvex server status
./yvex server model
./yvex server memory
```

These commands report readiness, the model actually open in the server, and its
current host/device memory accounting. They do not load another model.

### 5. Start chat

```sh
./yvex chat --session main
```

The named session retains conversation and KV state while sharing the one
resident runtime model. Alternatively, run one ephemeral streamed turn:

```sh
./yvex run "Explain attention in one sentence."
```

Discover commands without parsing prose:

```sh
./yvex help
./yvex help --advanced
./yvex help --json
```

The [operator runbook](docs/operator-runbook.md) owns complete startup,
three-terminal observation, sessions, shutdown, configuration, and recovery.

## Product boundary

| Component | Responsibility |
| --- | --- |
| `yvex` | Explicit foreground model server, public REPL, one-shot and administrative protocol client, plus finite offline compile, artifact, inspect, execute, profile, and system operations |
| `libyvex` | Reusable compilation, artifact, runtime, graph, backend, tokenizer, and generation implementation |

Runtime-facing `yvex` operations always cross private local protocol v10. The
finite offline lane may link engine owners but never hosts a persistent model.
One compiled operation registry drives command paths, syntax, help, JSON
discovery, completion, and slash schemas without becoming a domain-policy
owner.

The server-backed console uses one `yvex>` prompt, one compact vertical
attachment view, an immediately visible registry-derived slash catalog,
in-place server-authored prefill progress, one compact turn summary, semantic
TTY color, a categorized operational watch, and a detailed human trace.
Ctrl-L clears and redraws the active prompt without changing session state;
Ctrl-D exits cleanly even when discarding an unfinished line. For the admitted
DSpark prompt contract, `/think`, `/think-max`, and `/nothink` select only the
model's explicitly emitted reasoning channel; YVEX never infers or exposes
hidden reasoning. Noninteractive callers use
`yvex run --reasoning none|high|max`; stdout remains byte-faithful while
metrics are written separately.
Live milestone and release-gate state remains only in
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
- [DeepSeek-V4-Flash-DSpark record](docs/model-families/deepseek-v4-flash.md) — exact
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
- session persistence across server restart;
- complete accelerator residency, device-side sampling, or tokenizer execution;
- optimized DSpark execution or production load-aware confidence scheduling;
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
