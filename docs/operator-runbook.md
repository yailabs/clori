# YVEX Operator Runbook — Local Runtime

This runbook owns first startup and routine operation of the installed local
runtime and clients. The normal product workflow is:

```text
search -> pull -> prepare -> load -> chat
                    push -> outward distribution
```

Users select logical models; they do not copy deployment profile aliases into
shell variables. Exact source revisions, artifacts, profiles, engine
generations, and sessions remain authoritative beneath this porcelain and are
available through advanced commands. `yvex chat` is the sole linear console
and remains a native client of the resident host. This is not a capability
ledger; consult [`ROADMAP.md`](../ROADMAP.md) for current gates.

## Prerequisites

Builds provide one executable product. `yvex serve` owns the private Unix
listener and bounded loopback OpenAI-compatible listener in the foreground.
`yvex model load` resolves a launchable logical model to an exact deployment
and asks that host to create an engine generation. Several generations may
coexist within the admitted host bound. Other `yvex` modes own native clients
and finite offline engineering operations.

Loading a model requires one complete registry startup profile. A text runtime
binds one admitted GGUF to its exact runtime binding, target, backend, and
context capacity. A composite runtime instead binds an installed component root
to its target, backend, and capability mode without inventing a singular
artifact or text-runtime binding. Inspect the product catalog first:

```sh
./yvex model list --wide
./yvex model show MODEL
```

The table contains one row per proven logical model and exposes its selected
format, quantization or precision, representation size, state, backend,
location, and number of alternatives. `model show` expands the exact lineage.
Only READY models can cross the engine boundary. If none is ready, complete the
pull and preparation path or the one-time advanced
[registration procedure](#registering-an-existing-model). Backend selection is
part of the resolved deployment and never falls back silently.

Provider credentials remain owned by the installed provider CLI; YVEX records
only redacted observations and never persists a raw token. Discover the exact
implemented account operations and inspect current state with:

```sh
./yvex help source accounts
./yvex source accounts providers --output table
./yvex source accounts status --output table
./yvex source accounts whoami huggingface --output table
```

Authentication and removal remain explicit:

```sh
./yvex source accounts login huggingface
./yvex source accounts logout huggingface
```

The Hugging Face integration delegates credentials to the installed `hf`
client; the GitHub integration delegates them to `gh`. `--json` provides the
complete redacted machine projection. Account commands never print a token.

## Discover, acquire, and prepare a model

Search remote or local catalogs without downloading payloads:

```sh
./yvex model search "MiniMax H3"
./yvex model search "Qwen" --author Qwen --page 1 --limit 20
./yvex model search "local-name" --provider local
```

Remote availability and YVEX support are separate columns. Search never pulls,
prepares, or loads anything. Acquire one representation with a deterministic
locator:

```sh
./yvex model pull hf://OWNER/REPOSITORY
./yvex model pull hf://OWNER/REPOSITORY@IMMUTABLE_REVISION --format safetensors
./yvex model pull hf://OWNER/GGUF_REPOSITORY --format gguf --variant VARIANT
./yvex model pull hf://OWNER/REPOSITORY --format safetensors --dry-run
```

If the locator omits `@REVISION`, YVEX resolves the current provider reference
to an immutable identity before acquisition and reports the repository,
revision, chosen representation, and bytes. Multiple representation classes or
variants produce a line-oriented TTY selector. Non-TTY callers must pass
`--format` and, when still ambiguous, `--variant`; no provider file glob is
required in the normal workflow. Credentials come from the existing provider
account owner and tokens are never printed.

`--dry-run` is the safe metadata-only acquisition check: it resolves the
immutable provider revision and representation inventory, but downloads no
payload and creates no YVEX source, catalog, receipt, or transfer-log state.
The human default is a bounded repository/revision/representation summary;
add `--verbose` to inspect the provider's complete planned file inventory.

Local files and directories use the same distribution verb:

```sh
./yvex model pull /mnt/models/MODEL --managed
./yvex model pull file:///mnt/models/model.gguf --managed
./yvex model pull /mnt/external/MODEL --reference
```

`--managed` creates and verifies a durable YVEX-owned copy. `--reference`
records a verified external dependency without copying tens of gigabytes; if
the path disappears the model becomes BLOCKED. In a TTY, omitting both asks
which storage policy to use. Automation must choose explicitly. Unsupported
locators such as `ssh://` fail as unavailable transports and are never passed
to `scp` or a shell.

Prepare the acquired model through the high-level owner:

```sh
./yvex model prepare MODEL
./yvex model prepare MODEL --quant CANONICAL_PRESET
./yvex model pull hf://OWNER/REPOSITORY --format safetensors --prepare
```

Preparation delegates verification, family recognition, mapping, quantization,
materialization, package validation, and deployment creation to their canonical
owners. It succeeds only when that complete family/representation binding is
implemented. A canonical quantization preset is validated exactly; missing
full-package emission fails rather than silently using another qtype. An
already admitted GGUF is verified and bound without forced requantization.
`model pull --prepare` combines distribution and preparation but never loads
the host. Authenticated streamed preparation is not currently admitted, so
`--stream` refuses and does not claim that bytes avoided the machine.
Machine callers run `model pull --json` and `model prepare --json` as separate
operations; combining `--prepare` and `--json` is rejected so stdout remains
one valid JSON document with one operation schema.

Long acquisitions expose the existing status, stop, and resume lifecycle:

```sh
./yvex model status MODEL
./yvex model stop MODEL
./yvex model pull hf://OWNER/REPOSITORY --resume
```

Once READY, start the host and load the logical model:

```sh
./yvex serve
./yvex model load
./yvex model list --wide
./yvex host status
```

The TTY chooser shows model names and physical facts, never profile aliases.
Automation uses `model load MODEL` and supplies `--variant` only if several
launchable representations remain. The protocol request still carries one
exact profile and creates one exact generation. Detailed source, artifact,
profile, and engine commands remain available through `help --advanced` for
compiler work and qualification.

Before admitting a GB10 performance result, inspect the compiled CUDA image and
run the bounded bandwidth fixture:

```sh
./yvex inspect cuda
./yvex inspect cuda bandwidth
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
./yvex source verify \
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
./yvex host status
```

If it reports `ready`, do not start another server; proceed to chat or runtime
inspection. If it refuses because no server is present, inspect one READY
logical model:

```sh
./yvex model list --wide
./yvex model show MODEL
```

Then start the host in the first terminal:

```sh
./yvex serve
```

Foreground operation is intentional: keep this terminal open. The host owns the
local socket, OpenAI listener, telemetry, and bounded engine manager. Its
terminal renders operational events but never reads chat or lifecycle commands.
Select and load the intended model from another terminal:

```sh
./yvex model load
./yvex model list --wide
./yvex host status
```

The chooser first presents launchable logical models and, only when needed,
their physically distinct variants. It shows format, precision, size, backend,
and mode but never asks for a profile alias. Scripts use `model load MODEL` and
add `--variant VARIANT` when the catalog reports several choices. The same
terminal can then enter chat with `yvex chat`, while other protocol clients and
OpenAI consumers remain independent.

The host publishes its socket before any engine exists. `model load` resolves
the selected registry profile, authenticates its package and binding, seals the
deployment specialization, builds admitted engine resources, then publishes
one loaded engine generation. Product state becomes LOADED; advanced `engine
list` exposes exact loading, loaded, draining, unloading, and failure facts as
applicable. `host status` continues to report the independent host lifecycle.

Large engines can spend substantial time in load. Typed events and bounded
status expose real completed stages without inventing a percentage. A failed
load releases its partial engine resources and leaves the host, socket, OpenAI
listener, other engines, and telemetry alive. Package context, parallel
capacity, prefill chunk, generation mode, and backend come from the admitted
startup profile. When compatible execution width exists, independent active
workers may rendezvous at the engine scheduler for real MoE or output-head
rows; same-session mutation remains serialized. This is compatible-operation
batching, not global ready-sequence continuous batching.

## What “load the model” means

The relevant commands have different responsibilities:

- `yvex model list` derives one logical catalog from exact source, artifact,
  deployment, and resident-engine facts;
- `yvex serve` starts the model-neutral persistent host;
- `yvex model load MODEL` resolves one launchable representation and exact
  profile, then publishes one engine generation;
- `yvex model unload MODEL` retires that generation only after dependent
  sessions and leases are released, without stopping the host;
- `yvex model active` projects loaded generations, exact activity, clients,
  sessions, leases, directional capabilities, and resource placement;
- advanced `yvex engine list|show|load|unload` retains exact profile and
  generation control for engineering and qualification;
- `yvex host memory` reports current process, mapped, host-resident, and
  device-resident memory facts;
- `yvex chat` uses the already resident model through the local protocol and
  never creates another model copy.

The host keeps the immutable artifact mapping as canonical backing. A compiled
artifact-backed placement registers that mapping once for CUDA addressability;
compiler-required derived layouts instead own their separately accounted managed
storage. `host memory` reports mapped, prepared, resident, sequence-state,
workspace, device, RSS, and capacity facts separately. Authentication and
selected resources complete before the engine becomes `loaded`; host readiness
does not depend on one engine.

## Direct MiniMax-H3 media host

MiniMax-H3 uses the same persistent server and local chat protocol, but its four
large component artifacts are staged at request phase boundaries rather than
kept resident simultaneously. Its installed composite startup profile owns the
component location, CUDA backend, and media mode. Normal operation is therefore
the same registry-first command used by other hosted models:

```sh
./yvex model show minimax-h3-fl2va
./yvex serve
./yvex model load minimax-h3-fl2va
```

The default publication directory is `$YVEX_DATA_DIR/media`, or
`$HOME/.local/share/yvex/media` when that override is absent. YVEX creates and
admits it as an owned absolute non-symlink directory. `--output-root` and
`--media-artifact-root` remain explicit engineering overrides; they are not
normal startup requirements. Startup opens the tokenizer and all four
identity-bound component artifacts. The server retains their admitted immutable
views under one engine-generation identity, but does not preload the component
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
./yvex chat --model minimax-h3-fl2va --session video
```

Submit one creative prompt. The prompt immediately starts native generation;
there is no parameter questionnaire, keyword parser, assistant model, or
conversation planner. Prompt bytes are opaque execution input and reach the
existing tokenizer/conditioning path unchanged by operator policy.

Ordinary hosted execution selects the identity-bearing released FL2VA policy:
50 sigma points, 49 paired evaluations, terminal zero, a 1344x768 default
canvas, 124 frames, and seed 42. Select the released canvas, duration, and
deterministic seed when entering the native client, then type the creative
prompt at its prompt:

```sh
./yvex chat --model minimax-h3-fl2va \
  --trajectory released --width 768 --height 768 \
  --duration 5 --seed 42
```

Dimensions are paired, multiples of 32, within the released area and aspect
envelope, and include square, wide 1344x768, and portrait 768x1344 canvases.
Duration is aligned upward to the released `17n+5` frame rule and must remain
inside the 124-to-345-frame contract. The terminal result reports the resolved
duration, frame count, evaluation count, seed, trajectory, RNG, plan, engine,
and publication identities. `--trajectory preview` retains the separate
`interactive-preview-v1` YVEX test policy at 192x192, 124 frames, one
evaluation, and seed 42. Creative words such as `HD`, `seed`, `draft`, `MOV`,
numbers, or durations never alter either typed policy.

Chat projects server-authored conditioning, latent, decoder, publication,
completion, cancellation, and failure events as control state rather than
model-authored prose. On success it renders the typed publication path and
media identities. Ctrl-C uses the existing request cancellation contract and
must leave no partial published file.

The released maximum wide dual-anchor plan contains 106,238 packed rows. The
generic CUDA joint Transformer admits up to 131,072 rows and uses a 64-query
chunked exact attention path whose maximum released workspace is
15,230,279,684 bytes. The family admits 16 GiB of workspace and 64 GiB of peak
device resources, while retaining explicit resource refusal rather than
silently reducing canvas or trajectory. Preview names still describe bounded
geometry, not model quality. The smoke profile retains the earlier repeatable
32x32 evidence. The historical
pre-direct-execution server/chat acceptance used `smoke`, five seconds, two
sigma-grid points, AVI, and seed 42; it returned a 1,048,544-byte seekable file
after 560.36 seconds.
Independent GStreamer playback recovered 124 frames and 165,333 stereo samples
per channel with a 10,416 ns duration delta. Peak server RSS was 62.57 GiB
inside the 88 GiB hard limit, with no residual component residency after the
turn.

## Foreground host and client terminal

The [startup procedure](#first-verified-startup) assigns host lifetime to the
foreground terminal and administration/chat to clients. The host renders its
boot report and operational stream, never a command prompt. Duplicate
`yvex serve` refuses rather than attaching or taking over listeners.

After a model is loaded, `yvex chat --session main` and `yvex host logs`
use the same typed runtime authority. Logs support `--verbose` and `--json`;
prompts and answers are excluded by default. Stop the shared host only when
its users have agreed to shutdown.

## Interactive console

`./yvex chat` is the interactive entrypoint. Its attachment view derives model,
engine, session, context, and resource facts from the host; example memory
numbers or a screenshot are not admission evidence. `/help` gives the current
registry-authored command catalog.

The [external REPLAI editor](decisions/0007-external-terminal-editor.md) owns
input editing, history navigation, paste framing, and redraw. YVEX owns
commands, attachment conversion, session/engine binding, and typed generation
output and cancellation. The editor closes before generation starts; retained
presentation helpers are not a second native editor.

The prompt label is a product-catalog projection, while the
session remains bound to the exact engine generation. On transport loss the
same prompt adds `[disconnected]`; it never silently switches models. Model
output uses typed `reasoning` and `answer` sections when the source emits both;
disabled reasoning shows only `answer`. The terminal renderer supports bounded
headings, lists, emphasis, inline/fenced code and quotes, wraps prose to at most
112 columns, and never changes canonical response bytes. During a turn, the
console updates one server-authored prefill line in place. The terminal result
then reports prefill, generation, TTFT, speculation, initial/final context,
adaptive or explicit output envelope, truthful stop reason, and session on one
compact line. Candidate token text is never displayed.

On a TTY, cyan marks the prompt and active work, green marks readiness and
completion, orange marks cancellation or warning, red marks refusal, and dim
text carries secondary facts. Redirected and machine-readable output never
contains terminal controls. Set `NO_COLOR=1` to disable color explicitly; if
that variable is already exported in the shell, unset it to see the semantic
colors.

Slash commands are discovered from the canonical registry and their complete
current catalog is visible at startup. `/help` adds one-line descriptions;
`/status`, `/runtime`, `/model`, `/memory`, and `/context` inspect state;
`/session`, `/sessions`, `/new`, `/use`, `/detach`, `/reset`, and `/close`
manage the session; `/attach PATH` stages one local media object for the next
turn, `/attachments` lists the bounded ordered stage, and
`/attachments-clear` discards it. Repeated attachments belong to the same next
turn; after accepted submission the stage clears while the exact session stays
attached. `/cancel` cancels active generation; and `/quit` exits locally.
`/exit` is a registry-owned alias for `/quit`; bare `exit` remains
ordinary model input. Tab completes an unambiguous slash command. Commands for an unsupported
explicit reasoning channel refuse rather than simulate support. The current
DSpark profile admits `/think`, `/think-max`, and `/nothink`; a family with an
authenticated low-effort policy additionally admits `/think-low`. They select the
source-authored model-emitted channel and never expose hidden reasoning. A
policy change that alters the encoded prefix safely rebuilds only physical
sequence state and re-prefills the authoritative semantic history; reset is
not required merely to change reasoning mode.

Ctrl-D deletes at the cursor on nonempty input and exits on empty input.
Ctrl-C during a turn requests server-owned cancellation and returns to the
prompt; a second Ctrl-C requests exit. With no active turn, the first Ctrl-C clears the line and
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

While a generation owns the stream, chat suppresses terminal echo and does not
accept a draft for the next turn. Arrow/editing/UTF-8 bytes entered during that
interval are discarded before the line editor returns; they cannot enter the
assistant transcript or the following prompt. Ctrl-C remains an admitted
server-side cancellation signal. Terminal attributes are restored on every
success, refusal, disconnect and cancellation path.

## Interactive and programmatic requests

Human generation remains inside `yvex chat`. `/nothink`, `/think-low`, `/think`,
and `/think-max` select an admitted source-authored reasoning policy for the attached
session and that policy remains active until changed. Reuse an existing named
session by starting chat with `--session NAME`. Omitting
`--max-new-tokens` sends no client cap: the host resolves an adaptive envelope
from the loaded engine and remaining context. Supplying
`--max-new-tokens N` is an explicit upper bound. Neither mode is infinite;
EOS, source-authored stops, cancellation, context exhaustion and the resolved
output envelope remain distinct terminal reasons.

Programmatic inference uses the admitted private protocol or the loopback
OpenAI compatibility API described below; it is not projected as a second
one-shot CLI command. Fork an idle committed session with an explicit upper
bound for shared prefix backing:

```sh
./yvex session fork main experiment 1073741824
./yvex chat --session experiment
```

The child starts at the exact committed position and receives independent RNG,
token ledger, decoder, transcript and conversation state. Shared state pages
remain immutable and become private on write. The command refuses an active or
partial source, a duplicate child name, incompatible state geometry, or a
prefix exceeding the supplied byte bound.

## OpenAI-compatible application provider

The same `yvex serve` process owns the application listener. After
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
state, and typed persistent sequence/component state while sharing immutable
model resources. KV, recurrent, and convolution classes depend on the admitted
model; a session is not synonymous with a KV cache:

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

Client disconnect and detach do not close the engine. A partial or cancelled
turn can retain model-committed state and is never silently marked complete.
Protocol v20 reports the exact engine generation, committed position,
token/text counts, state generations, failure class, and reset requirement.
Reset clears sequence/component state, tokens, transcript, decoder, and RNG policy without
closing the engine or host.

State checkpoints are immutable and restore only when their model, binding,
artifact, engine generation, scope, and committed position match the live
session. The restore byte bound is mandatory. This operation currently protects
model state inside one live semantic session on the same engine generation; it
is not yet a cross-restart conversation restore.

## Status, metrics, and logs

Use compact status for normal operation:

```sh
./yvex host status
./yvex host status --json
./yvex model list --wide
./yvex host memory
```

`model list` marks resident logical models LOADED without turning generations
into peer models. Advanced `engine list` reports exact generation,
specialization, backend, residency, and readiness facts. `host memory`
separates artifact/mapped/prepared model spans, device-addressable and explicit
device allocation, typed session state, arena/workspace/transient current and
peak, process RSS, placement, and whether physical residency was measured. On
UMA, addressable mapped weights are not reported as zero GPU use merely because
the explicit device allocator owns zero bytes; unknown page placement remains
`not measured`. The displayed classes can overlap and must not all be summed.

Follow typed server activity independently of the foreground host stream:

```sh
./yvex host logs
./yvex host logs --follow
./yvex host logs --verbose
./yvex host logs --json
```

The foreground server stream and `host logs` project each request as one
coherent compact unit. Normal rows use `REQ`, `PF`, `1ST`, `DEC`, `DONE`,
`CXL`, and `FAIL`; live generation uses `t`/`p` for generated tokens/position,
`avg` for cumulative decode rate, `r32` for the bounded recent rate, `sp` for
speculative acceptance, and `rss`/`dev`/`ws`/`st`/`run`/`q` for live resource
and queue facts. Long identifiers are shortened only in this human projection.
They group speculative cycles, show queue pressure only when contended, and
finish with one stable terminal row. Ordinary connection churn, duplicate load
lifecycle detail, token fragments, and profiler rows are suppressed. `host
status` remains a current snapshot; `host logs` contains
chronology only and never prepends status sections. Without `--follow`, the
command returns after a bounded recent retained event tail.
`host logs --follow` remains attached for live events. `host logs --verbose`
exposes each typed speculative cycle. `host logs --json` emits the
canonical complete JSONL event record, including typed detail omitted by the compact
human view. Prompts and answers remain absent from every projection by default.

Large engine loads use server-authored `LOAD` phases. `verify` reports actual
bytes and `res` actual tensors when a denominator exists; phases such as `bind`,
`open`, `admit`, `mat`, `seal`, `be`, and `ws` show activity and elapsed time
without inventing percentages.

Raw server-event JSONL is selected at startup with `--logs json`. Increase
`--trace-level` from `summary` to `stages`, `tokens`, or `full` only when the
additional volume is required. Text content remains excluded unless the host is
started with the explicit `--trace-content` opt-in.

## Graceful shutdown

Release one engine while retaining the host and its other engines:

```sh
./yvex model unload MODEL
./yvex host status
```

Unload refuses while a live session or model lease still requires the engine.
Close dependent sessions and release leases explicitly; detach alone is not
session closure. Once admitted, retirement drains work and releases engine
resources while leaving the host ready. Host shutdown is a separate operation:

```sh
./yvex host stop
```

The host refuses new work, drains or cancels queued and active requests under
their typed state, closes sessions, closes the model exactly once, emits the
terminal shutdown event, and removes its socket and singleton lock.

## Registering an existing model

The ordinary selector consumes a complete local registry profile. When an
artifact and binding already exist but no profile was recorded, import them
once with the advanced registry operation and absolute paths:

```sh
./yvex profile create \
  --alias my-deepseek-dspark-profile \
  --family deepseek4 \
  --model v4-flash-dspark \
  --scope runtime \
  --class iq2xxs \
  --path /srv/yvex/models/deepseek-v4-flash-dspark-bootstrap-q2-v1.gguf \
  --runtime-binding /srv/yvex/models/deepseek-v4-flash-dspark.yvex-runtime-binding \
  --target deepseek4-v4-flash-dspark \
  --backend cuda \
  --execution-strategy speculative \
  --ctx 4096 \
  --support-level selected-tensor-materialized
```

This operation reads the GGUF, records its identity and metadata, checks that
the startup profile is structurally complete, and stores it in the user-local
registry. It does not establish runtime admission; `yvex model load`
authenticates the artifact and binding again when it opens the engine. Normal
subsequent use contains no paths or environment variables:

`--support-level` records only the artifact inspection/materialization stage.
The binding, target, backend, mode, and context fields separately own startup
profile readiness; the registry profile and the model live in the server remain
separate facts.

```sh
./yvex model show v4-flash-dspark
./yvex serve
./yvex model load v4-flash-dspark
```

Run `model load` from another terminal. If several launchable representations
exist, the TTY selector shows their physical facts; automation supplies
`--variant`. The foreground host continues to own only server lifetime and its
event stream. Advanced `profile list` reads exact deployment entries and
`engine list` reads exact resident generations. Loading and unloading does not
require restarting the host.

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
  sequence state.

When XDG variables are absent, the client uses the documented HOME-based
configuration fallback and the protocol owner uses its private runtime
fallback.

## Recovery

- Missing socket: run `yvex serve` and wait for `host status` to report the
  host ready, then use `yvex model load`.
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
- OpenAI `503 runtime_unavailable`: start the host with `yvex serve`, load the
  selected model with `yvex model load MODEL`, and confirm product and host
  readiness through `yvex model list --json` and `yvex host status --json`.
- OpenAI `422 unsupported_parameter`: remove the named unsupported field;
  fields are never ignored silently.

DeepSeek-specific semantics and source-boundary events are described in the
[family record](model-families/deepseek-v4-flash.md). Direct component execution,
tokenizer conformance, artifact inspection, and physical-compilation
diagnostics use the advanced `inspect`, `artifact`, `compile`, and `bench`
surfaces in the finite offline lane. Discover them with
`yvex help --advanced`; they are not part of the normal hosted startup path.

The admitted MiniMax-H3 Audio VAE component is reachable through that lane:

```sh
./yvex bench component audio-vae \
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
./yvex bench component video-vae \
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
./yvex bench media publish \
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
