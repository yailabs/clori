# YVEX Operator Runbook — Local Runtime

This runbook owns first startup and routine operation of the installed local
server and clients. Normal operation is registry-first: users list a complete
local model profile, name it when starting the foreground server, and enter
chat without exporting paths or repeating internal paths. Its commands follow
the canonical operation registry. The REPL attaches to the resident server and
uses the same typed session, progress, and result facts as noninteractive
clients.
It is not a capability ledger: consult [`ROADMAP.md`](../ROADMAP.md) for current
gates.

## Prerequisites

Builds provide one executable product. `yvex server MODEL` hosts one
process-resident runtime model, the private Unix listener, and the bounded
loopback OpenAI-compatible listener in the foreground. Other `yvex` modes own
native clients and finite offline engineering operations.

Starting a host requires one registry entry containing an admitted complete
GGUF, its exact runtime binding, target, backend, and context capacity. Inspect
the available entries first:

```sh
./yvex model list
```

Only a row with `STARTUP` equal to `yes` can be named for hosted execution.
Use `model show NAME` to confirm its backend and `target-only` or `dspark`
generation mode before startup. If the table is empty or every row says
`no`, complete the one-time
[registration procedure](#registering-an-existing-model). Backend selection is
part of that profile and never falls back silently.

Before admitting a GB10 performance result, inspect the compiled CUDA image and
run the bounded bandwidth fixture:

```sh
./yvex system cuda
./yvex system cuda bandwidth
```

The second command performs five timed samples over one 32 MiB working set and
reports CUDA streaming traffic, asynchronous D2D copy and coherent host access
separately. Its evidence identity binds every elapsed sample and the admitted
kernel bundle. The result is a machine-state observation; it is not the 273 GB/s
hardware specification and is not a benchmark claim when another workload is
active.

## Verifying source payloads for compilation

Source acquisition and source compilation are separate operations. After an
exact snapshot and its v1 acquisition manifest exist outside the repository,
publish the payload-trusted v3 manifest before constructing Transformation IR:

```sh
./yvex compile source verify \
  --source /srv/yvex/sources/DeepSeek-V4-Flash-DSpark \
  --models-root /srv/yvex \
  --source-manifest /srv/yvex/manifests/deepseek-v4-flash-dspark-source.json
```

This finite offline command first rechecks revision, sidecars, index and every
Safetensors header. It then reads every admitted shard, compares its SHA-256
with the pinned provider metadata and transactionally publishes the aggregate
payload identity. It refuses when authoritative shard digests are unavailable;
the product command does not turn a local-only seal into upstream evidence. A
current upstream-verified v3 manifest reopens without rereading 167 GB of
payload. The manifest is mutable provenance and should remain outside the
source snapshot whose bytes it identifies.

This command neither maps tensors nor emits an artifact. The subsequent
compilation stages consume the published payload identity and retained source
inventory through their typed owners.

## Probing a physical-variant candidate

Before paying the storage and admission cost of a complete candidate GGUF, run
one decision from its sealed physical plan against the real source payload:

```sh
./yvex compile quant probe \
  --target deepseek4-v4-flash-dspark \
  --source /srv/yvex/sources/DeepSeek-V4-Flash-DSpark \
  --models-root /srv/yvex \
  --source-manifest /srv/yvex/manifests/deepseek-v4-flash-dspark-source.json \
  --policy /srv/yvex/plans/candidate-policy.json \
  --imatrix-manifest /srv/yvex/calibration/deepseek-v4-flash.imatrix \
  --backend cuda \
  --plan /srv/yvex/plans/candidate.plan \
  --tensor blk.21.ffn_down_exps.weight
```

The command executes exactly the selected terminal from the identity-bound
plan and reports its encoded bytes and reconstruction metrics. It does not
publish an artifact, alter the registry, or claim whole-model quality. Use it
as the role-level funnel before promoting a surviving policy to complete
artifact emission.

## First verified startup

First check whether this user already owns a ready server:

```sh
./yvex server status
```

If it reports `ready`, do not start another server; proceed to chat or runtime
inspection. If it refuses because no server is present, inspect one
startup-ready registry entry. The alias below is illustrative; use one printed
by `model list`:

```sh
./yvex model list
./yvex model show deepseek4-v4-flash-dspark-runtime-iq2xxs
```

Then start the host in the first terminal:

```sh
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs
```

`server MODEL` directly enters the server entrypoint in the `yvex` process.
Foreground operation is intentional: keep this terminal open. Exactly one
server owns the model, sessions, KV, worker, local socket, OpenAI listener, and
telemetry.

Before the potentially long admission begins, the command prints the complete
selected startup identity and states that the host remains in the foreground:

```text
YVEX server · foreground
  profile deepseek4-v4-flash-dspark-runtime-iq2xxs
  target deepseek4-v4-flash-dspark · backend cuda · mode dspark · requested ctx 4096
  local endpoint .../yvexd.sock · OpenAI 127.0.0.1:8001
  stop with Ctrl-C here or `yvex server stop` from another terminal
```

This is the model the new server is about to open. It is not a projection of a
previously running process and it is flushed before admission begins.

Startup authenticates the selected artifact and binding, creates the immutable
runtime model, builds residency once, and only then publishes the local socket.
Large models can spend several minutes in this phase. The foreground server
prints an elapsed-time heartbeat every ten seconds until admission completes or
fails; it does not claim a percentage that the admission pipeline cannot prove.
From a second terminal, `yvex server status` refuses while the socket is absent
and reports `ready` only after admission completes. A refusal or startup failure
leaves no partially ready listener.

## What “load the model” means

There is no separate hosted model-load command. The relevant commands have
different responsibilities:

- `yvex model list` reads the local model registry and marks complete readable
  startup profiles;
- `yvex server NAME` opens and authenticates the named profile's artifact and
  binding, materializes runtime-owned resources, copies the encoded payload
  into the server's host arena, and keeps the resulting runtime model open;
- `yvex server model` reports the identities actually open in the server;
- `yvex server memory` reports current process, mapped, host-resident, and
  device-resident memory facts;
- `yvex chat` and `yvex run` use the already resident model through the local
  protocol and never create another model copy.

The host admits the mapped artifact, then copies every encoded model tensor
into one process-lifetime anonymous RAM arena before publishing `runtime.ready`.
`resident_host_bytes` is the authoritative payload-residency count; the mapped
file size and the smaller CUDA/unified accelerator prefix remain separate
metrics. The current DSpark bootstrap artifact therefore needs about 100.84 GiB of host
RAM for its 108,274,154,488-byte tensor payload, plus runtime state and backend
workspace. A cold start can take several minutes because authentication and
the complete RAM transfer finish before the socket becomes ready.

## Three-terminal operation

All three terminals attach to the same server and model. They do not create
three model copies.

Terminal 1 owns the foreground host lifecycle:

```sh
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs
```

Terminal 2 renders the operational engine view:

```sh
./yvex server log
```

Terminal 3 owns the interactive conversation:

```sh
./yvex chat --session main
```

Start Terminal 1 first. Terminals 2 and 3 may attach in either order after
`runtime.ready`. Raw and operational views derive from the same typed event
sequence. Default telemetry excludes prompt and answer content.

If observation is not needed, two terminals are sufficient: keep `runtime
server in the first and run `server status`, then `chat`, in the second.

## Interactive console

Chat opens one concise attachment view and the stable prompt:

```text
YVEX 0.1.0 · protocol 8

  model      deepseek4-v4-flash-dspark
  variant    abcdef012345
  runtime    ● ready · attached to resident runtime · CUDA · DSpark
  session    main · position 0 · turns 0
  context    0/4096
  memory     100.84 GiB process · 91.31 GiB artifact mapped · 0.02 GiB device
  OpenAI     ● ready · 127.0.0.1:8001

commands
  /help        Discover canonical commands and operations.
  /context     Show authoritative context and KV use for the attached session.
  /status      Return one composed runtime and attached-session snapshot.
  ...
  /quit        Exit the interactive client.

  Ctrl-C       cancel an active turn or clear input; press again to exit
  Ctrl-D       exit and discard an unfinished line
  Ctrl-L       clear and redraw input

yvex>
```

The exact identities come from the running server; the example values are not
admission evidence. Model output is streamed directly without repeated role
labels. During a turn, the console updates one server-authored prefill line in
place. The terminal result then reports prefill, generation, TTFT, speculation,
context, stop reason, and session on one compact line. Candidate token text is
never displayed.

On a TTY, cyan marks the prompt and active work, green marks readiness and
completion, orange marks cancellation or warning, red marks refusal, and dim
text carries secondary facts. Redirected and machine-readable output never
contains terminal controls. Set `NO_COLOR=1` to disable color explicitly; if
that variable is already exported in the shell, unset it to see the semantic
colors.

Slash commands are discovered from the canonical registry and their complete
current catalog is visible at startup. `/help` adds one-line descriptions;
`/status`, `/runtime`, `/model`, `/memory`, and `/context` inspect state;
`/session`, `/sessions`, `/new`, `/attach`, `/detach`, `/reset`, and `/close`
manage the session; `/cancel` cancels active generation; and `/quit` exits
locally. `/exit` is a registry-owned alias for `/quit`; bare `exit` remains
ordinary model input. Tab completes an unambiguous slash command. Commands for an unsupported
explicit reasoning channel refuse rather than simulate support. The current
DSpark profile admits `/think`, `/think-max`, and `/nothink`; they select its
source-authored model-emitted channel and never expose hidden reasoning. A
policy change that alters the encoded prefix safely rebuilds only physical
sequence state and re-prefills the authoritative semantic history; reset is
not required merely to change reasoning mode.

Ctrl-D exits from the prompt and discards an unfinished line. Ctrl-C during a
turn requests server-owned cancellation and returns to the prompt; a second
Ctrl-C requests exit. With no active turn, the first Ctrl-C clears the line and
a second consecutive Ctrl-C exits. EOF, cancellation, resize, and failure all
restore bracketed-paste and terminal modes before returning control to the
shell. Cancellation or failure requires `/reset` only when the server reports
committed partial progress. The console names that state and refuses a new turn
instead of silently appending to it. Reset creates a fresh execution session
while keeping the process-resident model open.

Left/Right, Home/End, Delete and Backspace edit the current UTF-8 line without
submitting it; Up/Down navigate local history. If the server connection closes,
an active progress row is terminated cleanly and the prompt changes to
`yvex [disconnected]>`. Local help and exit remain available. The next remote
operation attempts one foreground reconnect; when that fails, the unsent line
is preserved for another attempt rather than discarded.

Ctrl-L clears the visible terminal while the REPL prompt is active, then redraws
the prompt and any input already typed. It does not detach, reset, cancel, or
otherwise mutate the server-owned session.

## One-shot requests

An ephemeral one-shot session streams one answer and closes while leaving the
server and model alive:

```sh
./yvex run "Explain attention in one sentence."
./yvex run --reasoning high "Work this out, then answer briefly."
./yvex run --reasoning max "Prove the result and state the conclusion."
```

`--reasoning none|high|max` selects the same source-authored policy as
`/nothink`, `/think`, and `/think-max`. Non-interactive stdout is the exact
concatenation of canonical reasoning and final channel payload bytes; status,
summary, and separate reasoning/final timing metrics go to stderr. Neither
stream receives ANSI controls when redirected.

Reuse an existing named session only when conversational continuation is
intended:

```sh
./yvex run --session main "Continue more briefly."
```

## OpenAI-compatible application provider

The same `yvex server` process owns the application listener. After
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
curl -fsS http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek4-v4-flash-dspark","messages":[{"role":"user","content":"Hello"}],"stream":false}'
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
./yvex session state save main /var/lib/yvex/main-state.yvex
./yvex session state restore main /var/lib/yvex/main-state.yvex 1073741824
./yvex session close main
```

Client disconnect and detach do not close the model. A partial or cancelled
turn can retain model-committed state and is never silently marked complete.
Protocol v8 reports the exact committed position, token/text counts, state
generations, failure class, and reset requirement. Reset clears the session KV,
tokens, transcript, decoder, and RNG policy without closing the host.

State checkpoints are immutable and restore only when their model, binding,
artifact, scope and committed position match the live session. The restore byte
bound is mandatory. This operation currently protects model state inside one
live semantic session; it is not yet a cross-restart conversation restore.

## Status, metrics, and logs

Use compact status for normal operation:

```sh
./yvex server status
./yvex server status --json
./yvex server model
./yvex server memory
```

`server model` proves which artifact, binding, physical variant, target, and
backend are actually open. `server memory` separates mapped artifact bytes,
the process-lifetime host arena, and device-resident allocations.

Follow typed server activity independently of the foreground console:

```sh
./yvex server log
./yvex server log --json
```

The human `server log` starts with one stable startup/runtime block, then projects each request
as one coherent unit with stable time, request, session, phase, duration and
result fields. It groups prefill and DSpark cycles, shows queue pressure only
when contended, uses human byte units and named stop reasons, and replaces any
active progress line with one stable completion or failure summary. It
suppresses ordinary connection churn, token fragments and profiler detail.
`server log --json` emits the canonical complete JSONL event record, including
the typed detail omitted by the compact human view. Prompts and answers remain
absent from both projections by default.

Raw server-console JSONL is selected at startup with `--console raw`. Increase
`--trace-level` from `summary` to `stages`, `tokens`, or `full` only when the
additional volume is required. Text content remains excluded unless the host is
started with the explicit `--trace-content` opt-in.

## Graceful shutdown

Request shutdown through the local protocol:

```sh
./yvex server stop
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
  --alias deepseek4-v4-flash-dspark-runtime-iq2xxs \
  --family deepseek4 \
  --model v4-flash-dspark \
  --scope runtime \
  --class iq2xxs \
  --path /srv/yvex/models/deepseek-v4-flash-dspark-bootstrap-q2-v1.gguf \
  --runtime-binding /srv/yvex/models/deepseek-v4-flash-dspark.yvex-runtime-binding \
  --target deepseek4-v4-flash-dspark \
  --backend cuda \
  --generation-mode dspark \
  --ctx 4096 \
  --support-level selected-tensor-materialized
```

This operation reads the GGUF, records its identity and metadata, checks that
the startup profile is structurally complete, and stores it in the user-local
registry. It does not establish runtime admission; `yvex server` authenticates
the artifact and binding again when it opens the model. Normal subsequent use
contains no paths or environment variables:

`--support-level` records only the artifact inspection/materialization stage.
The binding, target, backend, mode, and context fields separately own startup
profile readiness; the registry profile and the model live in the server remain
separate facts.

```sh
./yvex model list
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs
```

`model list` reads registry entries; the positional `MODEL` argument selects
one entry for this invocation; and `server model` reads the identities actually
open in the resident server. Starting another model requires stopping the
current server first; hot model switching and multi-model hosting are not
current capabilities.

Generation mode is part of the startup profile. `dspark` requires a binding
that contains target, draft, and target-verification plans; `target-only` is
the explicit reference/debug mode. A DSpark startup refusal is not permission
to fall back silently. Select a compatible profile or repair the
artifact/binding.

## Local paths

- `$XDG_RUNTIME_DIR/yvex/yvexd.sock` is the private mode-0600 local protocol
  endpoint; its directory and singleton lock are private to the owning UID.
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

- Missing socket: run `yvex server MODEL` and wait for `server status` to
  report `ready`.
- Stale or unsafe socket: verify UID, mode, runtime-directory ownership, and
  singleton-lock ownership; never delete another user's socket.
- Binding or artifact mismatch: select the binding for that exact artifact
  identity; never bypass admission.
- Partial session: inspect it, then explicitly reset or close it before an
  ordinary new turn.
- Unsupported CUDA: start an admitted CPU host or repair CUDA admission; no
  CUDA request falls back silently.
- Queue refusal: wait for current work or reduce client concurrency; do not
  launch another server against the same socket.
- OpenAI `503 runtime_unavailable`: start the host with `yvex server MODEL`,
  wait for `runtime.ready`, and confirm `openai_ready` in
  `yvex server status --json`.
- OpenAI `422 unsupported_parameter`: remove the named unsupported field;
  fields are never ignored silently.

DeepSeek-specific operation is documented in
[`operations/deepseek.md`](operations/deepseek.md). Direct component execution,
tokenizer conformance, artifact inspection, and physical-compilation
diagnostics use the advanced `inspect`, `execute`, `profile`, and `system`
surfaces in the finite offline lane. Discover them with
`yvex help --advanced`; they are not part of the normal hosted startup path.

The admitted MiniMax-H3 Audio VAE component is reachable through that lane:

```sh
./yvex execute component audio-vae \
  --target minimax-h3-fl2va \
  --artifact /srv/yvex/artifacts/minimax-h3/audio_vae.gguf \
  --backend cpu \
  --input-file /srv/yvex/evidence/audio-latent.f32 \
  --batch 1 \
  --latent-steps 1 \
  --out /srv/yvex/evidence/audio-samples.f32
```

The input is contiguous F32 `[batch,32,latent_steps]`; the output is contiguous
mono F32 `[batch,800*latent_steps]` at the source-declared 32 kHz rate. The
command authenticates the exact component artifact, bounds workspace, executes
the native CPU decoder, and publishes the output without replacing an existing
file. Raw component samples are numerical evidence, not synchronized media or
an admitted MiniMax generation path.

The MiniMax-H3 Visual VAE CPU component is reachable through the same lane:

```sh
./yvex execute component video-vae \
  --target minimax-h3-fl2va \
  --artifact /srv/yvex/artifacts/minimax-h3/video_vae.gguf \
  --backend cpu \
  --input-file /srv/yvex/evidence/video-latent.f32 \
  --batch 1 \
  --latent-frames 1 \
  --latent-height 1 \
  --latent-width 1 \
  --out /srv/yvex/evidence/rgb-frames.f32
```

The input is contiguous F32 `[1,24,T,H,W]`; the output is contiguous F32
`[1,3,T*4,H*16,W*16]`. The three latent dimensions on the command must match
the input file. The command authenticates the complete Visual VAE artifact,
bounds workspace, executes all 36 native CPU decoder blocks with exact partial
3D RoPE, and publishes the output without replacing an existing file. Only
bounded small geometry has live qualification; this does not admit full-scale
or tiled execution. Raw RGB frames are numerical evidence, not a playable video
or synchronized media path.
