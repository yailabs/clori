# Operating DeepSeek-V4-Flash-DSpark

DeepSeek-V4-Flash-DSpark is the complete hosted text vertical for the v0.1
target. MiniMax is the admitted composite media counterexample on the same
host/runtime/backend substrate.
Family identities and architecture facts live in the
[DeepSeek technical record](../model-families/deepseek-v4-flash.md); current
gates live in [`ROADMAP.md`](../../ROADMAP.md).

## Product path

List the local startup profiles, inspect the admitted DeepSeek entry, start the
persistent host, then load the package as one engine generation:

```sh
./yvex profile list
./yvex serve
```

The foreground host has no administrative prompt. From another terminal,
inspect exact profiles and load the selected DeepSeek deployment:

```sh
./yvex profile list
./yvex profile show PROFILE
./yvex engine load PROFILE
```

Use a profile whose execution strategy is `dspark` for speculative execution.
No model path or environment variable is required during normal operation.
`engine load` authenticates the selected artifact and binding, seals the
deployment specialization, and creates one immutable engine generation. The
host itself starts with zero engines and stays alive across load and unload.
Importing an existing artifact into the registry is a one-time advanced
operation documented in the
[operator runbook](../operator-runbook.md#registering-an-existing-model), not
artifact admission.

After the engine reports `loaded`, verify and use that exact generation from
another terminal:

```sh
./yvex host status
./yvex engine list
./yvex host memory
./yvex chat --model PROFILE --session main
```

An optional third terminal may run `yvex host logs`. Status, logs, and chat
are clients of the same host; none reloads weights or opens a second engine.

The engine retains the canonical encoded package mapping plus its tokenizer,
attention, output-head, target, draft, verification, and admitted prepared
resources. The resource report distinguishes mapped package bytes from
prepared, resident, sequence-state, and workspace bytes. Each named session is
bound to the exact engine generation and owns independent DeepSeek persistent
state, bounded speculative candidate workspace, and prompt/token continuation.

On turn two, the host renders and encodes the complete expected conversation,
proves that the committed token ledger is its exact prefix, and prefills only
the new suffix. An incompatible prefix refuses; reset is explicit.

## Generation path

Human generation uses the linear `yvex chat` client. Programmatic inference
uses the native typed protocol or the loopback OpenAI profile; there is no
second one-shot CLI generation operation. CUDA greedy sampling selects the
token on device without copying the complete vocabulary row. Stochastic
sampling remains an explicit common-host reference adapter. Streamed fragments
are sent only after sampled-token decode commit and incremental detokenization
commit.

## Target-only and DSpark modes

Generation mode belongs to the named startup profile. `engine list --json`
reports the mode of each loaded generation. There is no per-turn fallback
switch.

`target-only` retains ordinary one-token target generation as the semantic
reference. `dspark` uses the checkpoint drafter to propose bounded blocks and
the complete target to verify them. Only the accepted target-authored result is
committed and streamed. Draft proposals do not appear in chat, native
streaming, SSE, transcript, or completion usage.

The final turn result reports the execution mode and compact speculation
counts. Use `host logs --json` for cycle-level draft,
verification, accepted-prefix, rejection, timing, and policy facts. A requested
DSpark profile refuses startup if its artifact, binding, backend, or workspace
requirements are incomplete; it never runs target-only silently.

For explicit reasoning, the source-authored reasoning terminator ends the
speculative shape. The committed final channel continues through ordinary
target decode. The `source-boundary` fact emitted by `host logs --json` reports
the boundary extent in `a`, the target-only continuation in `b`, and replayed
accepted target rows in `c`. The last value must remain zero. This is an
identity-bound DSpark sub-policy, not a fallback.

## Explicit reasoning

The current DSpark source profile admits explicit model-emitted reasoning. In
the interactive console use:

```text
/think      enable the source-authored reasoning policy
/think-max  enable its maximum policy
/nothink    disable it
```

The selected policy persists for the attached session until changed. If a
change no longer extends the committed token prefix, YVEX
rebuilds only physical sequence state and re-prefills the authoritative
semantic history. Reasoning is streamed through its own typed channel and
shown in dim text on a TTY; final answer text returns to the normal foreground.
Programmatic clients select the corresponding typed policy through their
admitted protocol field. YVEX does not infer reasoning from prose or expose
hidden chain of thought.

## Application-provider path

The same host exposes the bounded loopback profile without loading another
DeepSeek model. After `runtime.ready`, verify it:

```sh
curl -fsS http://127.0.0.1:8001/health
```

Discover the exact admitted model identifier with one command:

```sh
curl -fsS http://127.0.0.1:8001/v1/models
```

OpenAI Python and JavaScript SDKs use
`base_url=http://127.0.0.1:8001/v1`. Chat Completions and Responses translate
typed messages through the DeepSeek tokenizer/prompt owner; the adapter never
constructs DeepSeek control-token syntax. Function tools return typed calls for
the application to execute. `reasoning_effort` accepts `none`, `high`, or
`max`; explicit output is exposed separately as `reasoning_content`. See the
[bounded compatibility profile](../openai-compatibility.md).

## Offline engineering path

The engine-linked offline lane of `yvex` retains direct proof surfaces:

```sh
./yvex bench attention prepare --help
./yvex bench transformer execute --help
./yvex bench transformer decode --help
./yvex bench transformer logits --help
./yvex bench transformer sample --help
./yvex bench transformer generate --help

./yvex inspect tokenizer --help
./yvex inspect tokenizer encode --help
./yvex compile quant preset --help
./yvex compile quant plan --help
./yvex artifact materialize --help
```

These are finite engineering and conformance operations, not hosted-generation
aliases. They may open the engine directly but never create a persistent model
authority.

`bench transformer generate --audit` prints the identity-bound CUDA phase
ledger after a real finite request. `--json` exposes the same seven stable phase
slots and their measured/missing fact masks. Missing active bytes, occupancy or
batched-decode work are reported as unavailable and must not be read as zero.

For a bounded direct comparison over one admitted artifact and binding, the
engineering generator accepts `--generation-mode target-only|dspark`. This is
not the ordinary chat path and does not alter the loaded engine profile.

## Runtime evidence

For a hosted two-turn proof record:

- artifact, package, binding, specialization, engine-generation, session, and
  turn identities;
- model/artifact open and residency build counts;
- first and second prompt token counts;
- exact reusable prefix and second-turn suffix counts;
- sampled-token/decode-input equality;
- generation mode, draft cycles, proposals, target verifications, accepted and
  rejected drafts, maximum accepted prefix, and speculative policy identity;
- final positions and persistent-state digests;
- TTFT, prefill/decode timing and memory snapshots;
- cancellation, detach/reconnect, reset, and shutdown results.

Logs, traces, prompts, model output, artifacts, bindings, and model files remain
untracked external operator assets.

## Non-claims

Hosted generation and local compatibility evidence are not model-quality
evaluation, full-model benchmark evidence, release qualification, full OpenAI
service equivalence, public serving, remote security, continuous batching,
load-aware confidence scheduling, DSpark acceleration, or speculative support
for another family.
