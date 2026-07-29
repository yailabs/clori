# YVEX

YVEX is a native C/CUDA model compiler and inference runtime for
identity-bound, verified open-weight execution. It keeps source trust, logical
model semantics, the immutable Transformation IR, physical variants, artifacts,
materialization, runtime state, backend execution, generation, and evidence as
separate typed boundaries.

[`PROJECT.md`](PROJECT.md) is the sole authority for current capability and
release state. [`AGENTS.md`](AGENTS.md) defines repository and implementation
rules.

## Product topology

The local product has four explicit parts:

```text
libyvex engine
    -> yvexd: one long-lived local runtime host
    -> yvex: thin product client over the local protocol
    -> yvex-dev: optional compiler, graph, artifact, and evidence tooling
```

`yvexd` authenticates one GGUF and runtime binding, opens one immutable runtime
model, retains its materialization and residency, and serializes model-mutating
work through one bounded worker queue. Server-owned sessions retain independent
KV, exact committed token ledgers, message transcripts, sampling state, and
turn evidence across client detach and reconnect.

`yvex` never opens an artifact or invokes Transformer, logits, sampling, or
generation APIs directly. It connects to the private Unix-domain socket and
renders protocol results. `yvex-dev` may link the engine and owns deep direct
engineering operations.

The default socket is:

```text
$XDG_RUNTIME_DIR/yvex/yvexd.sock
```

When `XDG_RUNTIME_DIR` is absent, YVEX uses a private UID-owned directory below
`/tmp`. The directory and socket refuse unsafe ownership, permissions, and
symlink state.

## Runtime flow

The admitted DeepSeek vertical composes:

```text
prompt text or typed messages
    -> artifact-bound tokenizer and exact prompt policy
    -> exact reusable token prefix
    -> prefill only the new suffix
    -> persistent session KV
    -> normalized hidden state
    -> complete vocabulary logits
    -> greedy or seeded stochastic sampling
    -> exact sampled-token decode feedback
    -> incremental committed UTF-8 fragments
```

The daemon keeps the model open across requests. A client connection is not a
session; disconnecting does not close the model or erase session state. A
partial turn remains explicitly partial and must be reset or otherwise resolved
before an ordinary next user turn.

## Product client

Select one model alias for later starts. This records inert private XDG
configuration; the daemon remains responsible for artifact admission:

```sh
./yvex model use deepseek \
  --artifact /absolute/model.gguf \
  --runtime-binding /absolute/model.yvex-runtime-binding \
  --backend cuda \
  --context 4096
```

Start the runtime in the foreground:

```sh
./yvex runtime start \
  --model /absolute/model.gguf \
  --runtime-binding /absolute/model.yvex-runtime-binding \
  --backend cuda \
  --context 4096
```

After `model use`, plain `./yvex runtime start` uses that selection. Explicit
start options remain available and are independently validated by `yvexd`.

In separate terminals:

```sh
./yvex runtime watch
./yvex chat --session main
```

The daemon can expose the same authoritative typed event sequence as JSONL:

```sh
./yvexd \
  --model /absolute/model.gguf \
  --runtime-binding /absolute/model.yvex-runtime-binding \
  --backend cuda \
  --console raw \
  --trace-level stages
```

Default telemetry excludes prompt and generated content. Content requires the
daemon's explicit `--trace-content` opt-in.

The complete product grammar is compact and hierarchical:

```text
yvex
yvex chat
yvex run
yvex runtime start|stop|status|watch|trace
yvex session new|list|show|attach|detach|reset|close
yvex model list|use|show
yvex artifact show|verify
yvex quant preset|plan|emit|explain
yvex help
yvex version
```

`yvex run [options] TEXT` creates an ephemeral session by default, streams one
response, closes that session, and leaves the daemon and model alive. Supplying
`--session NAME` reuses an exact existing session when its continuation policy
admits the new prompt.

The old flat public command registry and its aliases are intentionally absent.
Direct graph, tensor, tokenizer, quantization, and artifact diagnostics live
under `yvex-dev`:

```text
yvex-dev graph ...
yvex-dev artifact ...
yvex-dev quant ...
yvex-dev tokenizer ...
yvex-dev source ...
yvex-dev tensor ...
yvex-dev runtime ...
yvex-dev evidence ...
```

## Three terminal views

One typed runtime-event authority feeds three simultaneous projections:

1. `yvexd --console raw` emits stable JSONL records for the daemon process.
2. `yvex runtime watch` renders compact load, queue, prefill, TTFT, decode,
   stop, memory, and shutdown activity.
3. `yvex chat` renders only conversation text and concise turn metrics.

None of these views reconstructs runtime truth by parsing another view's prose.
Operational timing is runtime evidence, not a release benchmark.

## Build and package

```sh
make -j4 all
make package
```

The build produces:

- `build/lib/libyvex.a`, the engine library;
- `./yvexd`, the engine-linked runtime host;
- `./yvex`, the thin protocol client;
- `./yvex-dev`, the optional engine-linked developer client.

The product package contains `yvex`, `yvexd`, notices, package identity, and
protocol facts. It never contains model weights, GGUF files, runtime bindings,
tests, logs, traces, or generated output.
`share/yvex/build.tsv` binds the source commit, protocol/backend facts, and
SHA-256 identities of packaged binaries and the engine library.

## Validation

```sh
make test-protocol test-runtime-host test-runtime-sessions
make test-runtime-turns test-runtime-telemetry test-runtime-streaming
make test-client test-repl test-cli-cutover test-packaging
make test-runtime-sanitizers
make check
```

CUDA qualification additionally uses `make check-cuda`; a no-`nvcc` build must
retain the host client/protocol and refuse unavailable CUDA execution before
dispatch.

## Capability boundary

DeepSeek-V4-Flash is the first complete local hosted vertical. YVEX does not
yet claim public or remote serving, authentication, TLS, OpenAI or Anthropic
compatibility, multi-model hosting, continuous batching, distributed serving,
persistent sessions across daemon restart, model quality, full-model benchmark
qualification, release qualification, MTP, or speculative execution.

## Documentation

- [`docs/contract.md`](docs/contract.md): lifecycle, protocol, publication,
  failure, and ownership contracts.
- [`docs/api.md`](docs/api.md): public and internal C APIs.
- [`docs/operator-runbook.md`](docs/operator-runbook.md): installed product
  operation and recovery.
- [`docs/runbooks/deepseek.md`](docs/runbooks/deepseek.md): exact DeepSeek host
  and developer procedures.
- [`docs/cli-output-architecture.md`](docs/cli-output-architecture.md): product,
  developer, raw-event, and terminal layout doctrine.
- [`MODEL_ARTIFACTS.md`](MODEL_ARTIFACTS.md): artifact terminology and admission.

## License

YVEX is licensed under [`LICENSE`](LICENSE). Third-party notices are recorded
in [`NOTICE.md`](NOTICE.md).
