# Interactive Console Quality Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-18 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `repair/repl-console-quality` |
| Baseline | `dca38fcf81155fd3c87966b0bfeb7d349e715976` |
| Checkpoint | `091603a587586492bce2f9eea88ca154fbb8279f` |
| Subsystem | operator interactive console |
| Model family | generic; observed during DeepSeek-V4-Flash qualification |
| Hardware | Linux pseudo-terminal fixture; DGX Spark operator observation |
| Evidence | canonical changed-file QA; PTY integration; operator-registry validation |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The interactive client accepted a line only at its tail. Arrow, Home, End and
Delete sequences were not interpreted as editing operations, so correcting a
word in place was not possible. A server disconnect during prefill could leave
the in-place progress row unterminated, append the transport failure on the same
visual line, and return to a prompt that still appeared attached. Exiting that
prompt with end-of-file attempted a remote detach and could print an unrelated
`unknown session` failure.

The startup snapshot also collapsed process RSS and mapped artifact bytes into
one ambiguous host-memory label. `/quit` was canonical, but the familiar
`/exit` spelling was not admitted by the operator registry.

## Problem

The console's input model and connection model were both too narrow for normal
interactive use. Terminal escape sequences were consumed without cursor-aware
editing, while a closed local socket was treated as one failed request rather
than a persistent disconnected client state. The result was typing friction and
misleading recovery behavior around the same backend failure operators were
trying to diagnose.

## Causal analysis

The line reader stored only an append position and performed blocking escape
lookahead. Its redraw logic therefore could not preserve a cursor inside UTF-8
input. Separately, generation reported the protocol error but did not return the
connection-loss fact to the REPL lifecycle. The prompt and detach path continued
to assume a live server. The progress renderer cleared no active row before the
error text was written.

The startup memory line used available typed metrics but labelled their sum as
host residency, obscuring the distinction between process RSS and immutable
artifact mappings.

## Decision

Keep the repair in the generic operator client. Make cursor position and
connection state explicit, retain slash aliases in the canonical operator
registry, and preserve the existing foreground lifecycle: a disconnected
client does not invent a daemon or silently reconnect in the background.

The next remote operation may attempt one foreground reconnect. If that attempt
fails, the unsent line remains available for another attempt. Local help and
exit remain usable without a server.

## Implementation

Checkpoint `091603a587586492bce2f9eea88ca154fbb8279f` adds cursor-aware UTF-8
insertion, deletion and redraw; Left/Right, Home/End, Delete, Backspace,
Ctrl-A/Ctrl-E and existing history behavior now share one line buffer. Escape
lookahead uses bounded polling so incomplete terminal sequences do not impose a
visible input stall.

Generation transport failure now terminates the active progress row, marks the
REPL disconnected and renders `yvex [disconnected]>`. End-of-file exits locally
without a stale detach. A failed foreground reconnect preserves the draft.
`/exit` is a generated registry alias for `/quit`; bare `exit` remains model
input. The startup view reports process RSS, mapped artifact bytes and device
residency as separate facts.

## After

Operators can revise a line before submission, exit with either canonical
`/quit` or admitted `/exit`, and distinguish a disconnected client from an
attached session. Socket loss no longer joins the transport error to an active
`0%` progress row or produces a secondary unknown-session error on exit.

The repair changes operator behavior only. It does not repair a server-side
model crash, add background reconnection, or establish a new product daemon.

## Evidence

- Canonical QA evidence
  `build/qa/evidence/09645b29643e5cf37423e93b3b029365ab42c242a537ad31001d17d1bd893d6f.json`
  resolved the branch change to 95 tests and completed with 95 PASS, zero FAIL,
  SKIP, BLOCKED or ERROR.
- The PTY integration fixture submits a line assembled through Home, End, Left
  and Delete and proves the corrected text reaches the server.
- The same fixture terminates the server at 0% prefill, proves a clean
  disconnected prompt, rejects `0%yvex:` concatenation and observes no
  `unknown session` on end-of-file.
- Operator-registry validation proves `/exit` is a unique generated alias and
  preserves deterministic registry identity.
- `git diff --check` passed at the checkpoint.

## Remaining limitations

- Reconnection is foreground and demand-driven; the client does not monitor or
  restart a stopped server.
- The local editor is intentionally bounded and is not a replacement for a
  complete Readline-compatible library.
- The repair makes server failure legible but does not establish the cause or
  correctness of any backend failure.
- This is software and operator integration evidence, not a model performance,
  evaluation or release claim.

## Why it matters

The interactive console now remains usable and truthful while the resident
runtime is healthy and when it fails, so operators can distinguish input/UI
friction from backend execution defects.

## Communication projections

### Short update

YVEX's local REPL now supports cursor-aware UTF-8 editing and presents socket
loss as an explicit disconnected state. Progress output closes cleanly, drafts
survive failed foreground reconnects, and `/exit` is a registry-owned alias
rather than an untracked parser special case.

### Quoteable technical facts

- "A closed local socket now moves the REPL into an explicit disconnected
  state."
- "The operator registry, not an ad hoc parser branch, owns the `/exit` alias."
- "Startup separates process RSS, mapped artifact bytes and device residency."
