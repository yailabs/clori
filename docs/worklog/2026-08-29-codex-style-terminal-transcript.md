# Unified Terminal Transcript and Composer

| Field | Value |
| --- | --- |
| Date | 2026-08-29 |
| Type | checkpoint |
| Milestone | operator TUI/UX repair |
| Branch | `models1` |
| Baseline | `be38e60d40a84308716c008a2985d417d63f78d0` |
| Checkpoint | `4464ef5c45df5305c6306de867e13abd9bcdbf70` |
| Subsystem | terminal UI, typed operator registry projection, operator runbook |
| Model family | generic |
| Hardware | not applicable |
| Evidence | software tests, PTY integration, registry and structural guards |
| Comparability | not applicable |
| Publishability | reviewed |

## Before

The bare `yvex` terminal divided model operation across persistent Home, Models, Sessions, and
Runtime screens. Tab cycled between those screens, slash commands lived in a separate palette,
and the composer was secondary to navigation. That interaction model made ordinary conversation
depend on application-page state and diverged from the requested Codex-style terminal workflow.

## Decision

The terminal now owns one transcript and one multiline composer. Slash discovery is projected
from the typed operator registry inside the composer. Model/profile selection, sessions, remote
search, and shortcut help are temporary overlays; Tab completes commands or moves between fields
inside a selector and never changes application screens.

Enter sends or confirms, Ctrl-J and kitty Shift-Enter insert a newline, PageUp and PageDown scroll
the transcript, and Esc closes an overlay or requests cancellation. `Ctrl-O` and `/model` open the
model selector, while `/runtime` refreshes typed host state inline.

## Implementation

The obsolete surface enum and tab-cycling state were removed from the TUI owner. Rendering was
collapsed into transcript, composer, inline typed engine/turn/memory facts, and transient overlays.
The command popup consumes generated operator-registry descriptors, with build dependencies
recorded for that generated interface.

The composer remains editable when no engine is available. A send in that state preserves the
draft and opens model selection. During an active generation, at most eight later messages are
queued against the exact session, engine alias, and engine generation. Runtime identity drift
restores queued text for operator review instead of sending it to a different engine generation.

No model-family, runtime-private, MiniMax, backend, protocol, or artifact owner changed. The TUI
continues to render typed protocol and registry facts rather than parsing human output or reading
backend-private state.

## Evidence

Focused evidence passed for `unit.tui`, `integration.repl`, and
`structural.operator-registry`. Two consecutive warning-clean `make -j2 all` runs completed under
the build-tree lock. Source ownership, architecture boundaries, `git diff --check`, and tracked
weight hygiene checks passed.

The final canonical changed-file run recorded 98 PASS and 5 FAIL in
`build/qa/evidence/c7db9acb75aac8ba278c7a90ba05dac48e161687931ab6289ce8fb797ce5cc74.json`.
Every TUI, PTY, operator-registry, documentation, ownership, and architecture obligation passed.
The five branch-existing failures are outside this delivery: one stale protocol expectation in
`integration.cli`, one 201-line function in `src/cli/io/client.c`, and source-size violations for
that file plus `src/model/families/minimax_h3.c` and `src/server/protocol.c`. They were not changed
to preserve concurrent ownership and the explicit MiniMax exclusion.

The documentation-only follow-up recorded 11 PASS and the same four source-structure FAIL rows in
`build/qa/evidence/02805e0f9b33cc569882a5d0338d5cb5c8cc0fe0ae6a3044114c234724274ea4.json`;
all documentation-specific guards passed.

## After

Bare `yvex` presents a stable conversation-first terminal without persistent tabs. Operator
actions remain reachable through the composer and temporary selectors, disconnected work keeps
its draft, queued work cannot cross runtime identity silently, and live execution facts remain
typed inline projections.

This checkpoint establishes the requested TUI interaction contract. It does not claim that the
five unrelated branch-wide QA failures are repaired, nor does it make a model-support,
performance, evaluation, or release claim.

```text
progression_decision: proceed
downstream_safe: true for the delivered TUI interaction contract
downstream_consumer: normal bare yvex operator path
gate blockers: none in the TUI owner
boundary incompleteness: none in the delivered TUI scope
evidence gaps: five unrelated existing branch-wide QA failures remain outside this delivery
deferred depth: none in the delivered TUI scope
optimization debt: none measured
generalization debt: none; the interaction contract is family-neutral
external blockers: none
required repairs: the owners of client protocol fixtures and source-size limits retain their failures
higher-capability non-claims: no model support, benchmark, evaluation or release qualification
```
