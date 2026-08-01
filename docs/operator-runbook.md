# YVEX Operator Runbook — Local Runtime

This runbook owns first startup and routine operation of the installed local
runtime host and clients. Normal operation is registry-first: users list a
complete local model profile, select it by name, start the runtime, and enter
chat without exporting paths or repeating daemon flags. Its commands follow
the canonical operation registry. The REPL attaches to the resident daemon and
uses the same typed session, progress, and result facts as noninteractive
clients.
It is not a capability ledger: consult [`ROADMAP.md`](../ROADMAP.md) for current
gates.

## Prerequisites

Builds provide two executable products:

- `yvexd` hosts one process-resident runtime model, the private Unix listener,
  and the bounded loopback OpenAI-compatible listener;
- `yvex` owns native runtime clients and finite offline engineering operations.

Starting a host requires one registry entry containing an admitted complete
GGUF, its exact runtime binding, target, backend, and context capacity. Inspect
the available entries first:

```sh
./yvex model list
```

Only a row with `STARTUP` equal to `yes` can be selected for hosted execution.
If the table is empty or every row says `no`, complete the one-time
[registration procedure](#registering-an-existing-model). Backend selection is
part of that profile and never falls back silently.

## First verified startup

First check whether this user already owns a ready host:

```sh
./yvex runtime status
```

If it reports `ready`, do not start another daemon; proceed to chat or runtime
inspection. If it refuses because no host is present, inspect and select one
startup-ready registry entry. The alias below is illustrative; use one printed
by `model list`:

```sh
./yvex model list
./yvex model show deepseek4-v4-flash-runtime-iq2xxs
./yvex model select deepseek4-v4-flash-runtime-iq2xxs
./yvex model selected
```

Then start the host in the first terminal:

```sh
./yvex runtime start
```

`runtime start` replaces its finite client process with the sibling `yvexd`
binary. Foreground operation is intentional: keep this terminal open. Exactly
one daemon owns the model, sessions, KV, worker, local socket, OpenAI listener,
and telemetry.

Startup authenticates the selected artifact and binding, creates the immutable
runtime model, builds residency once, and only then publishes the local socket.
Large models can spend several minutes in this phase. From a second terminal,
`yvex runtime status` refuses while the socket is absent and reports `ready`
only after admission completes. A refusal or startup failure leaves no
partially ready listener.

## What “load the model” means

There is no separate hosted model-load command. The relevant commands have
different responsibilities:

- `yvex model list` reads the local model registry and marks complete readable
  startup profiles;
- `yvex model select NAME` copies one profile into the inert configuration for
  a future launch; it does not open the artifact or change a running daemon;
- `yvex runtime start` opens and authenticates the selected artifact and binding,
  materializes runtime-owned resources, copies the encoded payload into the
  daemon's host arena, and keeps the resulting runtime model open;
- `yvex runtime model` reports the identities actually open in the daemon;
- `yvex runtime memory` reports current process, mapped, host-resident, and
  device-resident memory facts;
- `yvex chat` and `yvex run` use the already resident model through the local
  protocol and never create another model copy.

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

Terminal 1 owns the foreground host lifecycle:

```sh
./yvex runtime start
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

If observation is not needed, two terminals are sufficient: keep `runtime
start` in the first and run `runtime status`, then `chat`, in the second.

## Interactive console

Chat opens one concise attachment view and the stable prompt:

```text
YVEX 0.1.0 · protocol 4
deepseek-v4-flash · CUDA · variant 0123456789ab
runtime ready · attached to resident runtime
session main · position 0 · turns 0 · context 0/4096 · KV unavailable

yvex>
```

The exact identities come from the running daemon; the example values are not
admission evidence. Model output is streamed directly without repeated role
labels. During a turn, the console updates one daemon-authored prefill line in
place. The terminal result then reports prefill and generation separately,
including TTFT, context, stop reason, and session.

Slash commands are discovered from the canonical registry. `/help` lists the
admitted set; `/status`, `/runtime`, `/model`, `/memory`, and `/context` inspect
state; `/session`, `/sessions`, `/new`, `/attach`, `/detach`, `/reset`, and
`/close` manage the session; `/cancel` cancels active generation; and `/quit`
exits locally. Tab completes an unambiguous slash command. Commands for an
unsupported explicit reasoning channel are absent rather than simulated.

Ctrl-D exits from the prompt and discards an unfinished line. Ctrl-C during a
turn requests server-owned cancellation and returns to the prompt; a second
Ctrl-C requests exit. With no active turn, the first Ctrl-C clears the line and
a second consecutive Ctrl-C exits. EOF, cancellation, resize, and failure all
restore bracketed-paste and terminal modes before returning control to the
shell. Cancellation leaves the generation context partial and requires
`/reset` before new generation; the reset keeps the resident model open while
discarding the incomplete session state.

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

The same `yvexd` process owns the application listener. After
`runtime.ready`, verify it without starting another process:

```sh
curl -fsS http://127.0.0.1:8001/health
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
preset name. The adapter is loopback-only, opens no second model, owns no KV,
and executes no tools. Exact Chat Completions,
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
./yvex runtime model
./yvex runtime memory
```

`runtime model` proves which artifact, binding, physical variant, target, and
backend are actually open. `runtime memory` separates mapped artifact bytes,
the process-lifetime host arena, and device-resident allocations.

Subscribe to compact engine activity or the typed trace independently of the
daemon console:

```sh
./yvex runtime watch
./yvex runtime trace
./yvex runtime trace --json
```

`watch` requests the compact stage stream, while `trace` requests the detailed
event stream. Watch names the operational event and its semantic counters.
Human trace additionally shows sequence, severity, turn, phase, timing, and
rate. `trace --json` emits the canonical complete JSONL event record. Prompts
and answers remain absent from all three by default.

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

## Registering an existing model

The ordinary selector consumes a complete local registry profile. When an
artifact and binding already exist but no profile was recorded, import them
once with the advanced registry operation and absolute paths:

```sh
./yvex model registry add \
  --alias deepseek4-v4-flash-runtime-iq2xxs \
  --family deepseek4 \
  --model v4-flash \
  --scope runtime \
  --class iq2xxs \
  --path /srv/yvex/models/deepseek-v4-flash.gguf \
  --runtime-binding /srv/yvex/models/deepseek-v4-flash.yvex-runtime-binding \
  --target deepseek4-v4-flash \
  --backend cuda \
  --context 4096 \
  --support-level runtime-profile-configured
```

This operation reads the GGUF, records its identity and metadata, checks that
the startup profile is structurally complete, and stores it in the user-local
registry. It does not establish runtime admission; `yvexd` authenticates
the artifact and binding again when it opens the model. Normal subsequent use
contains no paths or environment variables:

```sh
./yvex model list
./yvex model select deepseek4-v4-flash-runtime-iq2xxs
./yvex model selected
./yvex runtime start
```

`model selected` reads only the inert private selection; `model list` reads
real registry entries; and `runtime model` reads the identities actually open
in `yvexd`. These states may differ without being conflated. Applying another
selection requires a daemon restart; hot model switching and multi-model
hosting are not current capabilities.

## Local paths

- `$XDG_RUNTIME_DIR/yvex/yvexd.sock` is the private mode-0600 local protocol
  endpoint; its directory and singleton lock are private to the owning UID.
- `$XDG_CONFIG_HOME/yvex/model.conf` stores the optional selected model alias
  and inert startup options at mode 0600.
- `~/.local/share/yvex/models.local.json` stores local model registry entries,
  including complete startup profiles. `YVEX_DATA_DIR` may override the parent
  for controlled deployments. The file is local configuration, not tracked
  repository data.
- `$XDG_STATE_HOME/yvex/` is reserved for explicit opt-in history, log, and
  trace sinks. The current client does not persist prompts, answers, tokens, or
  KV.

When XDG variables are absent, the client uses the documented HOME-based
configuration fallback and the protocol owner uses its private runtime
fallback.

## Recovery

- Missing socket: confirm `model selected`, run `yvex runtime start`, and wait
  for `runtime status` to report `ready`.
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
- OpenAI `503 runtime_unavailable`: start the host with `yvex runtime start`,
  wait for `runtime.ready`, and confirm `openai_ready` in
  `yvex runtime status --json`.
- OpenAI `422 unsupported_parameter`: remove the named unsupported field;
  fields are never ignored silently.

DeepSeek-specific operation is documented in
[`operations/deepseek.md`](operations/deepseek.md). Direct component execution,
tokenizer conformance, artifact inspection, and physical-compilation
diagnostics use the advanced `inspect`, `execute`, `profile`, and `system`
surfaces in the finite offline lane. Discover them with
`yvex help --advanced`; they are not part of the normal hosted startup path.
