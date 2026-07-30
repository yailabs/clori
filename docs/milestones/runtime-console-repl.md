# Mature Runtime Console and Interactive REPL

Milestone: `V010.OPERATOR.REPL.CONSOLE.0`

Track: `TRACK.OPERATOR`

State: blocked

Depends on: `V010.OPERATOR.COMMAND.ARCHITECTURE.0`

Successor: `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0`

This file is a future implementation contract. It does not establish console,
reasoning-channel, terminal, or performance capability.

## Mission

Transform the current functional daemon-backed chat client into a mature,
linear technical console without creating a TUI wall or a second operation
authority.

The REPL consumes the canonical operation registry, protocol facts, parsers,
defaults, and discovery supplied by
`V010.OPERATOR.COMMAND.ARCHITECTURE.0`. It cannot reimplement them.

## Reference Inspection

At milestone entry, pin and record the exact inspected commit of
`antirez/ds4`. Inspect its CLI loop, help, line editing, progress rendering,
explicit thinking rendering, interruption, and timing behavior.

Also inspect mature interaction patterns from `redis-cli`, the SQLite shell,
PostgreSQL `psql`, and GDB or LLDB. Extract principles only:

- prompt stability;
- TTY versus batch separation;
- history and completion;
- contextual help;
- cancellation and resize;
- multiline input;
- output hierarchy and semantic color;
- pager/large-output policy; and
- structured automation mode.

DwarfStar is a minimum maturity reference, not source, grammar, or layout to
copy.

## YVEX Console Doctrine

The YVEX REPL is attached to a model already resident inside `yvexd`.

It must communicate `attached to resident runtime`. It must not display:

- fake model-loading progress;
- duplicate protocol acknowledgements;
- raw session attachment rows;
- `assistant>` or `you>` role prefixes;
- generic `a=` or `b=` event fields; or
- hidden-reasoning claims.

The stable input prompt is:

```text
yvex>
```

Model output is streamed directly without a repeated role label.

An indicative layout is:

```text
YVEX 0.1-dev
DeepSeek-V4-Flash · mixed IQ2/Q2 · CUDA/GB10
runtime ready · session main · context 0/4096

yvex> Explain RoPE briefly.

processing 11 input tokens · 11/11 · 100%

RoPE rotates query and key features by position-dependent angles, encoding
relative token distance directly into their dot products.

prefill      11 tokens · 9.42 s · 1.17 tok/s
generation   24 tokens · 34.16 s · 0.70 tok/s
TTFT         11.53 s
context      35 / 4096 · reused 0
stop         EOS

yvex>
```

The exact typography may evolve. The semantic hierarchy may not.

## Semantic Style

Color is an optional semantic projection:

- accent/cyan: prompt, active progress, admitted interactive commands;
- normal terminal foreground: final model answer;
- dim grey: secondary identities, metrics, and explicitly emitted reasoning;
- green: ready, complete, admitted success;
- yellow: warning, degradation, near-capacity state;
- red: refusal, error, failed operation.

One terminal-style owner controls styling. Color must respect `NO_COLOR`, turn
off for non-TTY output, remain unnecessary for comprehension, never enter JSON
or other machine output, and remain readable on light and dark themes.

## Startup and Attachment

Startup performs one composed status operation and renders one concise block:

- YVEX and protocol versions;
- model and family;
- physical variant;
- backend and available machine/backend identity;
- runtime readiness;
- session and attachment;
- position and turn count;
- context use; and
- KV use when authoritative.

It must not print a session table row, a standalone attachment acknowledgement,
and a second banner for the same facts. Detailed identities remain available
through `/status`, `/model`, and `/runtime`.

## Progress

The daemon and protocol remain authoritative for progress.

Prefill may render one in-place line:

```text
processing 41 input tokens · 29/41 · 70.7%
```

On completion it becomes a stable row or is cleanly replaced before model
output. Decode may show a bounded status such as:

```text
generating · 12 tokens · 0.70 tok/s
```

Progress cannot corrupt streamed output and must remain truthful under slow
decode, cancellation, EOS, maximum-token stop, runtime failure, resize, and
redirection.

## Explicit Thinking and Reasoning

Grey rendering of explicitly emitted thinking is an interaction reference.
YVEX must classify it through typed family, tokenizer, prompt, protocol, and
output-channel contracts rather than terminal substring replacement.

Potential commands are `/think`, `/think-max`, and `/nothink`, but they appear
only when the preceding milestone supplies a typed thinking policy, context
requirements, prompt-template semantics, and explicit output classification.

No command reveals hidden chain of thought. No renderer infers reasoning from
prose. If the active family/profile does not admit an explicit reasoning
channel, these operations remain unavailable or refuse as unsupported. Tags
such as `<think>` never appear in the final user projection.

## Slash Catalog

The final slash catalog is projected from the canonical operation registry.
Expected categories are:

```text
discovery:       /help
composed state:  /status /runtime /model /memory /context
session:         /session /sessions /new NAME /attach NAME /detach /reset /close
generation:      /cancel /ctx N /think /think-max /nothink
local:           /read FILE /quit
```

An entry appears only when its operation is implemented and admitted for the
current runtime and family. `/quit` and line editing may remain REPL-local.
Every other entry is an adapter over a canonical operation.

## Status, Watch, and Trace

The milestone owns terminal renderers for:

```text
yvex runtime status
yvex runtime watch
yvex runtime trace
yvex runtime trace --json
```

The contracts are distinct:

- status: bounded authoritative snapshot;
- watch: compact semantic operational stream;
- trace: detailed human technical stream;
- trace `--json`: canonical JSONL event stream.

Watch renders event meaning: queueing, tokenization, prefix reuse, prefill,
first token, decode, commit, stop, and cancellation. Trace may show deeper typed
facts, but never tensors, logits, KV contents, prompt text, or response text by
default.

## Final Turn Metrics

One typed protocol result supplies, where available:

- prompt, reused, and newly prefetched token counts;
- prefill seconds and throughput;
- TTFT;
- generated tokens and generation seconds;
- decode throughput and inter-token latency summary;
- final position and context capacity/use;
- stop or cancellation class; and
- session identity.

Compact, verbose, status, and JSON projections consume the same facts.
Generation duration is not reconstructed from rounded throughput.

## Terminal Behavior

Acceptance covers at least:

- UTF-8, emoji, and multibyte token fragments;
- bracketed and multiline paste;
- history and completion;
- resize, narrow, and wide terminals;
- `NO_COLOR` and non-TTY output;
- stdout/stderr redirection;
- absent or terminating daemon;
- incomplete final newline and long output;
- Ctrl-C during prefill and decode;
- a second Ctrl-C, EOF, detach/reconnect, and session close; and
- unsupported commands and invalid arguments.

The first Ctrl-C cancels an active turn and returns to the prompt. A subsequent
Ctrl-C without active work clears the line or exits according to one documented
bounded policy. Terminal state must always be restored.

## Acceptance

The milestone closes only when:

- startup and attachment output is singular and truthful;
- `yvex>` is stable and model output has no role prefix;
- progress is semantic and protocol-authoritative;
- explicit reasoning is distinct only when genuinely emitted and admitted;
- slash commands derive from the canonical registry;
- `/status` has one explicit composed meaning;
- selected and live model facts remain separate;
- prefill and generation metrics remain distinct;
- watch is semantic and trace has human and JSON projections;
- cancellation, resize, paste, disconnect, and redirection preserve terminal
  state;
- machine output contains no color or terminal controls;
- no second parser, operation owner, or session authority is introduced; and
- no numerical-performance claim is made.

## Transition

On accepted closure:

```text
V010.OPERATOR.REPL.CONSOLE.0 = complete
V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 = active
Active Next = V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0
```
