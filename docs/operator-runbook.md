# YVEX Operator Runbook — Local Runtime

This runbook owns first startup and routine operation of the installed local
runtime host and clients. It begins with explicit artifact and binding inputs;
configured model defaults are an optional convenience after that path works.
It is not a capability ledger: consult [`PROJECT.md`](../PROJECT.md) for current
gates.

## Prerequisites

Builds provide four distinct products:

- `yvexd` hosts one process-resident runtime model;
- `yvex` connects as the product client;
- `yvex-openai` adapts bounded loopback HTTP/JSON/SSE requests to the private
  local protocol without linking the inference engine;
- `yvex-dev` retains direct engineering and conformance operations.

The first startup requires one admitted complete GGUF and its exact runtime
binding. Keep both outside the repository and assign their absolute paths:

```sh
export YVEX_MODEL_ARTIFACT=/absolute/model.gguf
export YVEX_RUNTIME_BINDING=/absolute/model.yvex-runtime-binding
test -r "$YVEX_MODEL_ARTIFACT" && test -r "$YVEX_RUNTIME_BINDING"
```

Use `--backend cpu` when the admitted build or device does not support CUDA.
Backend selection never falls back silently.

## First verified startup

Start the host directly in the first terminal:

```sh
./yvexd --model "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096 --console raw --trace-level stages
```

Foreground operation is intentional. Keep this terminal open and wait for the
`runtime.ready` JSONL event before connecting a client. The daemon authenticates
the artifact and binding, builds immutable residency once, publishes its private
local socket, and reports one model-open lifecycle.

The host admits the mapped artifact, then copies every encoded model tensor
into one process-lifetime anonymous RAM arena before publishing `runtime.ready`.
`resident_host_bytes` is the authoritative payload-residency count; the mapped
file size and the smaller CUDA/unified accelerator prefix remain separate
metrics. The current mixed IQ2_XXS/Q2_K artifact therefore needs about 87.7 GiB of host
RAM for its 94,142,453,320-byte tensor payload, plus runtime state and backend
workspace. A cold start can take several minutes because authentication and
the complete RAM transfer finish before the socket becomes ready.

## Three-terminal operation

All three terminals attach to the same daemon and model. They do not create
three model copies.

Terminal 1 owns the host and raw typed events:

```sh
./yvexd --model "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096 --console raw --trace-level stages
```

Terminal 2 renders the operational engine view:

```sh
./yvex runtime watch
```

Terminal 3 owns the interactive conversation:

```sh
./yvex chat --session main
```

Start Terminal 1 first. Terminals 2 and 3 may attach in either order after
`runtime.ready`. Raw and operational views derive from the same typed event
sequence. Default telemetry excludes prompt and answer content.

## One-shot requests

An ephemeral one-shot session streams one answer and closes while leaving the
daemon and model alive:

```sh
./yvex run "Explain attention in one sentence."
```

Reuse an existing named session only when conversational continuation is
intended:

```sh
./yvex run --session main "Continue more briefly."
```

## OpenAI-compatible application provider

Keep `yvexd` running, then start the application gateway in another terminal:

```sh
./yvex-openai --host 127.0.0.1 --port 8001
```

Configure compatible applications with `base_url=http://127.0.0.1:8001/v1`
and a local non-secret API-key placeholder. One-line readiness and request
checks are:

```sh
curl -fsS http://127.0.0.1:8001/health
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek4-v4-flash","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

The model identifier must match `GET /v1/models`; it is not a quantization
preset name. The gateway is loopback-only, opens no model, owns no KV, executes
no tools, and may be restarted without closing `yvexd`. Exact Chat Completions,
Responses, SSE, function-call, stop, JSON-object, error, and unsupported-field
semantics are in [`openai-compatibility.md`](openai-compatibility.md).

## Session lifecycle

Named sessions retain their own transcript, committed token ledger, sampling
state, and persistent KV while sharing immutable model resources:

```sh
./yvex session new main
./yvex session list
./yvex session show main
./yvex session attach main
./yvex session detach main
./yvex session reset main
./yvex session close main
```

Client disconnect and detach do not close the model. A partial or cancelled
turn can retain model-committed state and is never silently marked complete.
Reset clears the session KV, tokens, transcript, decoder, and RNG policy without
closing the host.

## Status, metrics, and trace

Use compact status for normal operation:

```sh
./yvex runtime status
./yvex runtime status --json
```

Subscribe to compact engine activity or the typed trace independently of the
daemon console:

```sh
./yvex runtime watch
./yvex runtime trace
```

Raw daemon JSONL is selected at host startup with `--console raw`. Increase
`--trace-level` from `summary` to `stages`, `tokens`, or `full` only when the
additional volume is required. Text content remains excluded unless the host is
started with the explicit `--trace-content` opt-in.

## Graceful shutdown

Request shutdown through the local protocol:

```sh
./yvex runtime stop
```

The host refuses new work, drains or cancels queued and active requests under
their typed state, closes sessions, closes the model exactly once, emits the
terminal shutdown event, and removes its socket and singleton lock.

## Optional configured defaults

After the explicit startup path succeeds, a private XDG configuration can store
an inert model selection for shorter future starts:

```sh
./yvex model use deepseek --artifact "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096
./yvex model show
./yvex runtime start
```

`model use` does not admit an artifact, open a model, or change a running host.
`yvexd` still authenticates the selected artifact and binding on every process
start. Applying another selection requires a daemon restart.

## Local paths

- `$XDG_RUNTIME_DIR/yvex/yvexd.sock` is the private mode-0600 local protocol
  endpoint; its directory and singleton lock are private to the owning UID.
- `$XDG_CONFIG_HOME/yvex/model.conf` stores the optional selected model alias
  and inert startup options at mode 0600.
- `$XDG_STATE_HOME/yvex/` is reserved for explicit opt-in history, log, and
  trace sinks. The current client does not persist prompts, answers, tokens, or
  KV.

When XDG variables are absent, the client uses the documented HOME-based
configuration fallback and the protocol owner uses its private runtime
fallback.

## Recovery

- Missing socket: repeat the explicit first startup and wait for
  `runtime.ready`.
- Stale or unsafe socket: verify UID, mode, runtime-directory ownership, and
  singleton-lock ownership; never delete another user's socket.
- Binding or artifact mismatch: select the binding for that exact artifact
  identity; never bypass admission.
- Partial session: inspect it, then explicitly reset or close it before an
  ordinary new turn.
- Unsupported CUDA: start an admitted CPU host or repair CUDA admission; no
  CUDA request falls back silently.
- Queue refusal: wait for current work or reduce client concurrency; do not
  launch another daemon against the same socket.
- Gateway `503 runtime_unavailable`: start `yvexd`, wait for `runtime.ready`,
  and verify that `yvex-openai` uses the same private socket.
- Gateway `422 unsupported_parameter`: remove the named unsupported field;
  fields are never ignored silently.

DeepSeek-specific operation is documented in
[`runbooks/deepseek.md`](runbooks/deepseek.md). Direct graph, tokenizer,
artifact, and physical-compilation diagnostics belong to `yvex-dev`, not the
product startup path.
