# Foreground Host Lifecycle Console Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-28 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` adjunct operator repair |
| Branch | `models1` |
| Baseline | `cd807a844c4578cca72cde80679db0993e377ebc` |
| Checkpoint | `b436cb4a4c0c4a1fade4b2b5fd95132fe0466038` |
| Subsystem | foreground server CLI, local model-profile projection, operator documentation |
| Model family | generic host; DeepSeek exercised |
| Hardware | not-applicable to the console contract; NVIDIA GB10 used for one diagnostic load |
| Evidence | software tests, structural guards, runtime lifecycle qualification, operator diagnostic |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

`yvex server` already started a persistent zero-engine host and exposed native and loopback OpenAI
listeners, but its foreground TTY was only a startup banner plus the human event stream. Loading
the first engine required a second process running `yvex server load MODEL`. Living documentation
also repeated one old DeepSeek alias while the local registry contained several later profiles
for the same runtime target.

The reported `runtime binding open failed` was not a failure of persistent-host topology. The
operator had selected the first 100.85 GiB historical DeepSeek profile. The local registry held
eight structurally startup-ready DeepSeek rows, including the current 88.52 GiB profile with
artifact identity rooted at `b9825a070028a66af28cdb25614f7a86c6ad1ec396eed6ae961039db1507ce0e`
and runtime-binding identity
`31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.

## Problem

The foreground product surface made a healthy persistent host awkward to operate and obscured the
exact profile choice. A user had to open another terminal before loading any model, while the
ordinary table truncated long aliases enough to make old and current DeepSeek rows difficult to
distinguish. Fixing this in server core or by launching a hidden child daemon would have duplicated
authority. Automatically treating the last registered row as current would instead have made the
CLI a new model-selection policy owner.

## Causal analysis

The existing foreground process already owned the model-loader callback, host lifecycle, event
projection, and typed engine APIs. It therefore had every authority needed for a bounded operator
console without changing server, protocol, runtime, or public ABI contracts. The model registry
could truthfully provide complete aliases and structural startup readiness, but it did not own a
preferred-profile fact. Runtime admission still had to occur through the existing engine-load
path.

A real local-registry smoke confirmed the distinction. The console listed nine profiles, including
eight DeepSeek rows and one MiniMax row. Selecting DeepSeek row 8 opened the authenticated current
binding, published engine generation 1, unloaded it, and left the zero-engine host ready. The old
row remains historical local state and still fails admission when selected explicitly.

## Decision

The human TTY form of `yvex server` now owns a small foreground lifecycle console in the same
process. It lists full profile aliases and accepts an exact alias or displayed number. A runtime
target is accepted only when it identifies exactly one structurally ready profile; multiple rows
refuse as ambiguous and require explicit selection. This preserves registry and model-selection
authority rather than encoding a last-row or filename preference in CLI code.

The prompt is `yvex[host] >` with zero loaded engines, the exact runtime target with one loaded
engine, and `yvex[multi-engine] >` with several loaded engines. Non-TTY operation, human-console
disablement, and raw-console operation preserve the prior external-client workflow. Native and
OpenAI clients remain available while the foreground console is active.

The console labels registry rows as structurally complete rather than loadable. A failed runtime
admission explains that distinction and directs the operator to choose another exact profile or
repair the selected record; it does not silently substitute another artifact or binding identity.

Rejected alternatives were a hidden child server, shelling out to the public client, selecting the
last profile for a target, or reinterpreting structural startup validation as runtime admission.

## Implementation

The foreground CLI gained bounded commands for `profiles`, `load`, `models`, `unload`, `status`,
`help`, and `stop`. They call the existing typed registry and server APIs directly. Registry access
inside the process is serialized because its borrowed entry projection is not reentrant. A poll
loop lets external host shutdown terminate an idle console without making stdin block server
lifecycle completion.

The existing engine loader remains the only admission owner. The console does not parse runtime
errors, fabricate readiness, or introduce a server/core specialization. Living command,
DeepSeek-operation, OpenAI, runbook, README, changelog, and documentation-surface projections were
updated to describe same-terminal lifecycle management and exact profile selection. The README
guard now accepts either the canonical Markdown title or the centered YVEX lockup with accessible
alt text, preserving the separately published logo decision.

## After

One terminal can start the host, list exact profiles, load one model, inspect engine and host state,
unload the engine, and stop the host. After a successful DeepSeek load the prompt changes from the
host label to `deepseek4-v4-flash-dspark >`; after unload it returns to `yvex[host] >`. External
clients continue to share that same host and engine generation.

The real profile-8 diagnostic observed 92,822,224 KiB process RSS at peak inspection, of which
92,610,664 KiB was file-backed and only 168,360 KiB anonymous. The process reported zero swapped
bytes and 186 MiB of CUDA allocation at that point. System-available memory remained about
117 GiB, system swap use did not increase, and unload returned available memory to about 118 GiB.
These are lifecycle/memory characterizations of one diagnostic run, not a throughput benchmark.

## Evidence

The exact clean checkpoint `b436cb4a4c0c4a1fade4b2b5fd95132fe0466038` passed:

- warning-clean C/CUDA product build;
- `tests/cli/server.sh`, including PTY lifecycle, ambiguous-target refusal, numeric selection,
  fail-closed binding admission, host survival, and socket cleanup;
- `tests/client_cutover.sh`;
- documentation architecture and documentation surface guards;
- source ownership, architecture-boundary, repository-layout, and source-layout guards;
- `git diff --check`.

A second diagnostic used the real local registry on protocol v14 from the concurrently modified
shared worktree. Profile 8 loaded, published generation 1, unloaded, left the host ready, then
stopped with its socket removed. Because unrelated MiniMax source was uncommitted and mutable, that
run is diagnostic only and is not retained as exact-tree branch qualification.

The final branch-wide QA required by the parent DeepSeek performance wave remains an evidence gap
until the concurrent MiniMax delivery reaches a stable checkpoint.

## Remaining limitations

- The console does not invent a preferred profile. Duplicate runtime targets require a full alias
  or displayed number until a canonical model-domain preference exists.
- Historical registry rows are not deleted or silently repaired. Selecting an obsolete binding
  still fails closed.
- This is a host-lifecycle console, not a second chat REPL or a replacement for native/OpenAI
  clients.
- The repair did not run generation or make a throughput, quality, release, or evaluation claim.
- Final combined source-stable QA remains pending for the shared branch.

## Why it matters

YVEX now exposes its persistent-host architecture as an operable product rather than forcing the
operator to reconstruct it across terminals. The change improves first use without duplicating
engine authority or weakening exact profile admission.
