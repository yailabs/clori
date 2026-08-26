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

Starting a host requires one complete registry startup profile. A text runtime
binds one admitted GGUF to its exact runtime binding, target, backend, and
context capacity. A composite runtime instead binds an installed component root
to its target, backend, and capability mode without inventing a singular
artifact or text-runtime binding. Inspect the local lifecycle catalog first:

```sh
./yvex model list
```

The table keeps acquired source, package readiness, and observed engine state
separate. Only a `package-ready` package can cross the engine boundary. Use
`model show NAME` for the package's full registry and startup facts. If the
table has no ready package, complete the appropriate acquisition and preparation
path or the one-time
[registration procedure](#registering-an-existing-model). Backend selection is
part of that profile and never falls back silently.

## Discover, acquire, and prepare a model

The normal model lifecycle is explicit:

```text
remote model -> representation -> exact revision -> acquired source
             -> verified source or inspected GGUF -> YVEX package
             -> local catalog -> engine handoff
```

Search Hugging Face metadata without downloading payloads:

```sh
./yvex model search "MiniMax H3"
./yvex model search "Qwen" --author Qwen --page 1 --limit 20
```

The first view groups results by provider-derived family and reports remote
representation classes, gated state, YVEX support stage, and local presence.
Search may show unsupported models. Remote availability never implies source
ingest, package readiness, or engine capability. `--json` returns the typed
`yvex.remote-model-catalog.v1` record used by noninteractive consumers;
`--interactive` offers a terminal drill-down while calling the same domain API.

Inspect one repository before selecting a representation:

```sh
./yvex model inspect MiniMaxAI/MiniMax-H3
./yvex model inspect MiniMaxAI/MiniMax-H3 --revision REVISION --audit
```

Inspection resolves the exact provider revision and lists its safetensors and
GGUF files. Safetensors precision comes from provider metadata when present.
Remote GGUF qtypes inferred from filenames remain explicitly provisional until
the acquired container passes YVEX GGUF inspection. A provider-reported base
model is retained as lineage; repositories without that evidence remain
separate.

Acquire only after choosing the repository, representation, and immutable
revision:

```sh
./yvex model acquire --repo OWNER/NAME --family FAMILY --name LOCAL_NAME \
  --revision EXACT_REVISION --include '*.safetensors' --models-root /srv/yvex

./yvex model acquire --repo OWNER/GGUF_REPOSITORY --family FAMILY \
  --name LOCAL_NAME --revision EXACT_REVISION --include 'selected-file.gguf' \
  --models-root /srv/yvex
```

The first path creates an acquired source. Source verification, semantic
compilation, physical policy, and transformation remain distinct preparation
stages. The second path retains the GGUF qtype as acquired physical truth;
compatible GGUF may be admitted directly after structural, family, role, qtype,
layout, and package checks, while incompatible containers require an explicit
repack. YVEX does not requantize a compatible external GGUF merely to relabel it.

Use the local catalog to find the next legal state:

```sh
./yvex model list
./yvex model list --json
./yvex model show PACKAGE_NAME --audit
```

`model list` combines acquisition receipts and the package registry without
opening payloads or engines. A moving provider reference is shown as
`moving-reference`; it is not silently promoted to immutable source evidence.
Package identities and startup facts remain available through `model show`.
Engine state is `not-observed` until the committed host API provides a live
projection; package readiness never implies residency or serving activity.

Hugging Face credentials remain owned by `yvex system accounts` and the
official provider CLI. Discovery arguments, tables, JSON, receipts, and logs do
not contain token bytes. Public discovery can proceed without authentication;
gated or authentication-required refusals remain explicit. Provider metadata is
discovery evidence only. Acquisition binds the selected exact revision through
the source provenance contract rather than allowing a later moving `main` to
change a prepared package.

Before admitting a GB10 performance result, inspect the compiled CUDA image and
run the bounded bandwidth fixture:

```sh
./yvex system cuda
./yvex system cuda bandwidth
```

The first command reports whether the admitted bundle is native, its exact architecture and its
content identity. The default build selects native code only when local hardware detection is
unambiguous and supported by `nvcc`; otherwise it remains an explicitly reported portable-PTX build.

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
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs --parallel 2
```

`server MODEL` directly enters the server entrypoint in the `yvex` process.
Foreground operation is intentional: keep this terminal open. Exactly one
server owns the model, sessions, KV, keyed scheduler, local socket, OpenAI listener, and
telemetry.

Before the potentially long admission begins, the command prints the complete
selected startup identity and states that the host remains in the foreground:

```text
YVEX server · foreground
  profile deepseek4-v4-flash-dspark-runtime-iq2xxs
  target deepseek4-v4-flash-dspark · backend=cuda · mode=dspark · requested ctx=4096 · parallel=2
  artifact /absolute/path/to/model.gguf
  binding /absolute/path/to/model.yvex-runtime-binding
  local endpoint .../yvexd.sock · OpenAI 127.0.0.1:8001
  stop with Ctrl-C here or `yvex server stop` from another terminal
```

This is the model the new server is about to open. It is not a projection of a
previously running process and it is flushed before admission begins.

Startup authenticates the selected artifact and binding, creates the immutable
runtime model, builds residency once, and only then publishes the local socket.
`--parallel N` admits capacity for N independently scheduled sessions before
readiness. When the binding admits compatible execution width, ready independent
sessions may contribute real rows to the same typed MoE or output-head execution
batch; requests for one named session remain serialized. Attention and sampling
remain session-local, so this is not transformer-wide continuous batching.

When `--prefill-chunk` is omitted, the server resolves an adaptive chunk before
capacity admission. Width-one operation retains the established interactive
chunk, while concurrent serving starts from the real admitted scheduling width
with a bounded amortization floor. `yvex server status` reports the resolved
`prefill` value. An explicit positive `--prefill-chunk N` remains authoritative;
zero is invalid, and an unsafe override is refused before model residency.
DSpark keeps its separately admitted bounded lookahead workspace; prompt chunk
selection does not truncate a real speculative verification batch.
Large models can spend substantial time in this phase. On a TTY the foreground
server maintains one in-place elapsed-time admission line until admission completes
or fails; redirected output retains periodic ten-second receipts. Neither projection
claims a percentage that the admission pipeline cannot prove.
From a second terminal, `yvex server status` refuses while the socket is absent
and reports `ready` only after admission completes. A refusal or startup failure
leaves no partially ready listener.

## What “load the model” means

There is no separate hosted model-load command. The relevant commands have
different responsibilities:

- `yvex model list` reads the local model registry and marks complete readable
  startup profiles;
- `yvex server NAME` opens and authenticates the named profile's artifact and
  binding, constructs its compiler-selected residency, and keeps the resulting
  runtime model open;
- `yvex server model` reports the identities actually open in the server;
- `yvex server memory` reports current process, mapped, host-resident, and
  device-resident memory facts;
- `yvex chat` and `yvex run` use the already resident model through the local
  protocol and never create another model copy.

The host keeps the immutable artifact mapping as canonical backing. A compiled
artifact-backed placement registers that mapping once for CUDA addressability;
compiler-required derived layouts instead own their separately accounted managed
storage. `server memory` reports mapped, non-artifact host, device, RSS and capacity
facts separately. Authentication and selected residency complete before the socket
becomes ready.

## Direct MiniMax-H3 media host

MiniMax-H3 uses the same persistent server and local chat protocol, but its four
large component artifacts are staged at request phase boundaries rather than
kept resident simultaneously. Its installed composite startup profile owns the
component location, CUDA backend, and media mode. Normal operation is therefore
the same registry-first command used by other hosted models:

```sh
./yvex model list
./yvex server minimax-h3-fl2va-runtime-media
```

The default publication directory is `$YVEX_DATA_DIR/media`, or
`$HOME/.local/share/yvex/media` when that override is absent. YVEX creates and
admits it as an owned absolute non-symlink directory. `--output-root` and
`--media-artifact-root` remain explicit engineering overrides; they are not
normal startup requirements. Startup opens the tokenizer and all four
identity-bound component artifacts. The server retains their admitted immutable
views under one runtime-model identity, but does not preload the component
payloads or create a CUDA context. A completed media request stages each
already-admitted component through the native YVEX runtime. The composite
registry profile is a local deployment contract; family semantics still select
and validate the exact four-component topology.

Component byte authentication uses the common verified-reopen authority. A
cold open fully verifies a component and publishes its snapshot-bound receipt;
an unchanged warm open authenticates that component without rereading its full
payload. Missing, malformed, stale, or unusable receipt evidence falls back to
full verification and is repaired only after the bytes match. The four
components are independent: one fallback does not invalidate three valid warm
reopens. This mechanism does not materialize weights or imply host/CUDA
residency.

From another terminal, start the ordinary client:

```sh
./yvex chat --session video
```

Submit one creative prompt. The prompt immediately starts native generation;
there is no parameter questionnaire, keyword parser, assistant model, or
conversation planner. Prompt bytes are opaque execution input and reach the
existing tokenizer/conditioning path unchanged by operator policy.

Ordinary hosted execution currently selects the identity-bearing YVEX policy
`interactive-preview-v1`: profile `preview`, 192x192, 124 frames, two sigma-grid
points (one Transformer evaluation), AVI publication, and seed 42. These are
explicit YVEX interactive test policy, not MiniMax or upstream defaults.
Changing the policy changes its identity. Advanced offline engineering commands
retain explicit parameter surfaces; creative words such as `HD`, `seed`,
`draft`, `MOV`, numbers, or durations do not alter the hosted preset.

Chat projects server-authored conditioning, latent, decoder, publication,
completion, cancellation, and failure events as control state rather than
model-authored prose. On success it renders the typed publication path and
media identities. Ctrl-C uses the existing request cancellation contract and
must leave no partial published file.

All preview profiles fit the current 8,192-row Omni execution bound even for a
256-token prompt. The 5,757-row `preview-384` Transformer envelope passed all
50 blocks against an independent manual BF16 oracle under a 100 GiB hard limit
with zero swap. Full media playback at that tier is still pending; the earlier
192x192 AVI playback and synchronization are verified, but sampled frames
remain visually unrecognizable. Preview names describe bounded geometry, not
model quality. The source-sized path would require 37,726 packed rows at the
same prompt and duration and therefore refuses before model materialization.
The smoke profile retains the earlier repeatable 32x32 evidence. The historical
pre-direct-execution server/chat acceptance used `smoke`, five seconds, two
sigma-grid points, AVI, and seed 42; it returned a 1,048,544-byte seekable file
after 560.36 seconds.
Independent GStreamer playback recovered 124 frames and 165,333 stereo samples
per channel with a 10,416 ns duration delta. Peak server RSS was 62.57 GiB
inside the 88 GiB hard limit, with no residual component residency after the
turn.

## Normal two-terminal operation

Both terminals use the same server and model. The client does not create another
model copy.

Terminal 1 owns the foreground host lifecycle:

```sh
./yvex server deepseek4-v4-flash-dspark-runtime-iq2xxs
```

The foreground terminal transitions from admission into the compact operational
event stream. A second terminal owns the interactive conversation:

```sh
./yvex chat --session main
```

Start Terminal 1 first and attach after `runtime.ready`. An optional observer can
run `./yvex server log`; add `--verbose` for individual DSpark cycles or `--json`
for canonical JSONL. All views derive from the same typed event sequence. Default
telemetry excludes prompt and answer content.

## Interactive console

Chat opens one concise attachment view and the stable prompt:

```text
YVEX 0.1.0 · protocol 12

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

Fork an idle committed session only with an explicit upper bound for shared
prefix backing:

```sh
./yvex session fork main experiment 1073741824
./yvex run --session experiment "Continue independently."
```

The child starts at the exact committed position and receives independent RNG,
token ledger, decoder, transcript and conversation state. Shared state pages
remain immutable and become private on write. The command refuses an active or
partial source, a duplicate child name, incompatible state geometry, or a
prefix exceeding the supplied byte bound.

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
Protocol v12 reports the exact committed position, token/text counts, state
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
./yvex server log --verbose
./yvex server log --json
```

The foreground human console and `server log` project each request as one coherent
unit with stable time, request, session, phase, duration and result fields. They
group prefill and DSpark cycles, show queue pressure only
when contended, uses human byte units and named stop reasons, and replaces any
active progress line with one stable completion or failure summary. It
suppresses ordinary connection churn, token fragments and profiler detail.
`server log --verbose` exposes each DSpark cycle. `server log --json` emits the
canonical complete JSONL event record, including typed detail omitted by the compact
human view. Prompts and answers remain absent from every projection by default.

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
  --backend cuda \
  --input-file /srv/yvex/evidence/audio-latent.f32 \
  --batch 1 \
  --latent-steps 1 \
  --max-device-bytes 2147483648 \
  --out /srv/yvex/evidence/audio-samples.f32
```

The input is contiguous F32 `[batch,32,latent_steps]`; the output is contiguous
mono F32 `[batch,800*latent_steps]` at the source-declared 32 kHz rate. The
command authenticates the exact component artifact, bounds host workspace and
CUDA residency, executes the native decoder, and publishes the output without
replacing an existing file. Use `--backend cpu` and omit
`--max-device-bytes` for the CPU path. Raw component samples are numerical
evidence, not synchronized media or an admitted MiniMax generation path.

The MiniMax-H3 Visual VAE CPU and CUDA component paths are reachable through
the same lane:

```sh
./yvex execute component video-vae \
  --target minimax-h3-fl2va \
  --artifact /srv/yvex/artifacts/minimax-h3/video_vae.gguf \
  --backend cuda \
  --input-file /srv/yvex/evidence/video-latent.f32 \
  --batch 1 \
  --latent-frames 1 \
  --latent-height 1 \
  --latent-width 1 \
  --max-device-bytes 17179869184 \
  --out /srv/yvex/evidence/rgb-frames.f32
```

The input is contiguous F32 `[1,24,T,H,W]`; the output is contiguous F32
`[1,3,T*4,H*16,W*16]`. The three latent dimensions on the command must match
the input file. The command authenticates the complete Visual VAE artifact,
bounds workspace and CUDA residency, executes all 36 native decoder blocks with
exact partial 3D RoPE, and publishes the output without replacing an existing
file. Use `--backend cpu` and omit `--max-device-bytes` for the CPU path. Only
bounded small geometry has live qualification; this does not admit full-scale
or tiled execution. Raw RGB frames are numerical evidence, not a playable video
or synchronized media path.

Decoded video and audio can be synchronized and atomically published through
the native finite offline lane:

```sh
./yvex execute media publish \
  --video-file /srv/yvex/evidence/rgb-frames.f32 \
  --frames 124 --width 32 --height 32 \
  --fps-numerator 24 --fps-denominator 1 \
  --audio-file /srv/yvex/evidence/audio-stereo.f32 \
  --audio-channels 2 --audio-samples 165600 --sample-rate 32000 \
  --max-host-bytes 1073741824 --max-output-bytes 4294967296 \
  --out /srv/yvex/evidence/minimax-h3.avi --output audit
```

The input files are contiguous planar F32 `[3,frames,height,width]` RGB and
`[channels,samples]` PCM. The native AVI writer stores uncompressed BGR24 video
and PCM S16LE audio, trims surplus PCM to the exact rational video duration,
validates the finished RIFF structure, and publishes without replacing an
existing destination. Its identities exclude the local paths. This command is
playable decoded-media publication, not MiniMax prompt-to-video generation; the
model phases remain separate until the end-to-end composition is admitted.
