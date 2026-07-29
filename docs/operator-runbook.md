# YVEX Operator Runbook — Local Runtime

This runbook owns installed local host and client operation. It is not a
capability ledger; consult [`PROJECT.md`](../PROJECT.md) for current gates.

## Local paths

- `$XDG_RUNTIME_DIR/yvex/yvexd.sock` is the private mode-0600 local protocol
  endpoint; its directory and singleton lock are private to the owning UID.
- `$XDG_CONFIG_HOME/yvex/model.conf` stores the selected model alias and inert
  start options at mode 0600.
- `$XDG_STATE_HOME/yvex/` is reserved for explicit opt-in history/log/trace
  sinks. The current client does not persist prompts, answers, tokens, or KV.

When the XDG variables are absent, the client uses the documented HOME-based
configuration fallback and the protocol owner uses its private runtime fallback.

## Start

Select a reusable local default when desired:

```sh
./yvex model use deepseek \
  --artifact /absolute/model.gguf \
  --runtime-binding /absolute/model.yvex-runtime-binding \
  --backend cuda \
  --context 4096
./yvex model show
./yvex runtime start
```

The selection file is private XDG configuration. It contains inert paths and
options; `yvexd` still authenticates the exact artifact and binding on every
process start. Applying another selection requires a daemon restart.

An explicit foreground start remains available:

```sh
./yvex runtime start \
  --model /absolute/model.gguf \
  --runtime-binding /absolute/model.yvex-runtime-binding \
  --backend cuda \
  --context 4096
```

Foreground operation is the default. The daemon authenticates the artifact and
binding, builds immutable residency once, publishes its private local socket,
then reports `READY`. It does not load the complete GGUF into anonymous RAM;
artifact mappings and admitted resident packs have separate counters.

## Three-terminal operation

Run these commands in three separate terminals. They connect to one daemon and
one resident runtime model; they do not open three model copies.

Terminal 1 owns the daemon and the complete structured event stream:

```sh
./yvexd \
  --model "$ARTIFACT" \
  --runtime-binding "$RUNTIME_BINDING" \
  --backend cuda \
  --context 4096 \
  --console raw \
  --trace-level tokens
```

Terminal 2 projects the same event authority as a compact engine view:

```sh
./yvex runtime watch
```

Terminal 3 is the interactive conversation client:

```sh
./yvex chat --session main
```

Start Terminal 1 first and wait for `runtime.ready`; Terminals 2 and 3 may then
attach in either order. Default telemetry excludes prompt and answer content.

## Observe

```sh
./yvex runtime status
./yvex runtime status --json
./yvex runtime watch
./yvex runtime trace
```

Raw daemon JSONL is selected at host startup with `--console raw`. Default
telemetry never contains prompt or answer content.

## Use

```sh
./yvex chat --session main
./yvex run "Explain attention in one sentence."
./yvex run --session main "Continue more briefly."
```

The first form is interactive. The second uses an ephemeral session and leaves
the daemon alive. An explicit named session retains exact KV and transcript
state across detach and reconnect.

## Sessions

```sh
./yvex session new main
./yvex session list
./yvex session show main
./yvex session attach main
./yvex session detach main
./yvex session reset main
./yvex session close main
```

A partial or cancelled turn can retain model-committed state. It is never
silently marked complete. Reset clears KV, tokens, transcript, decoder, and RNG
policy through the session authority without closing the model.

## Stop

```sh
./yvex runtime stop
```

Shutdown refuses new work, cancels/drains queued and active requests according
to their typed state, closes sessions, closes the model exactly once, emits the
terminal shutdown event, and removes the socket and singleton lock.

## Recovery

- Missing socket: start the runtime with an explicit model and binding.
- Stale or unsafe socket: verify UID, mode, runtime directory ownership, and
  that no daemon instance owns the lock; never delete another user's socket.
- Binding/artifact mismatch: generate or select the binding for that exact
  artifact identity; do not bypass admission.
- Partial session: inspect it, then explicitly reset or close it before a new
  ordinary turn.
- Unsupported CUDA: use an admitted CUDA build/device or explicitly start a CPU
  host; no CUDA request falls back silently.

Deep direct diagnostics and physical compilation are documented in
[`runbooks/deepseek.md`](runbooks/deepseek.md) and run through `yvex-dev`.
