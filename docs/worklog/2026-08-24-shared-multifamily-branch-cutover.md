# Shared Multifamily Branch Cutover

| Field | Value |
| --- | --- |
| Date | 2026-08-24 |
| Type | closure |
| Milestone | not-applicable |
| Branch | `models1` |
| Baseline | `2df3b84cc840dfca8b38f6fc387a833169b5598e` |
| Checkpoint | `f14dc2ad17e22984d1c7f70cfd5fa4f68ccaefa3` |
| Subsystem | repository development topology, QA orchestration, and hosted runtime qualification |
| Model family | DeepSeek-V4-Flash-DSpark and MiniMax-H3 |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 / 128 GiB unified memory |
| Evidence | Git ancestry; structural, integration, runtime, CUDA, sanitizer, and bounded live qualification |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

`main` was the common stable base at `2df3b84cc840dfca8b38f6fc387a833169b5598e`.
DeepSeek had advanced three commits to `7c729f4f366487670af874b3bb73ea4c9bf947ad`; MiniMax
had independently advanced four commits to
`6534c315d9d1b35d70e32735f25a5deb05d17458`. Both feature tips descended from the common base,
but neither contained the other.

The repository already admitted concurrent agents and compatible same-file work. Its doctrine
still required family work to remain on family branches and generic repairs to travel through
`main` before another family consumed them. Long QA runs authenticated their initial source but
did not fail closed if `HEAD` or the dirty source delta changed before completion.

## Problem

DeepSeek and MiniMax are active consumers of the same compiler, runtime, backend, server, protocol,
registry and artifact owners. Mandatory family-to-main-to-family routing delayed reciprocal
qualification of generic changes. Moving both histories into one worktree without stronger source
stability would create a different risk: a green report could describe a source snapshot that no
longer existed at run completion.

The cutover therefore needed to preserve both published histories, define semantic rather than
branch-based ownership, and make long evidence invalid when the shared working tree moves.

## Causal analysis

The branch histories had one textual conflict in `config/documentation_owners.tsv`; both sides had
added retained worklogs. Keeping every independently owned row produced one valid combined
documentation authority. Git reported no conflict in `tests/client_cutover.sh`, but its isolated
empty registry still used schema v4 while the MiniMax side had made schema v5 current. Updating
only that fixture preserved the current production registry contract.

Combined live qualification later exposed a separate stale test assertion. The DeepSeek server
opened the exact artifact and v14 binding, completed 26 bounded requests, recorded cancellation,
and shut down cleanly. The live runner nevertheless expected an older human trace renderer after
the CLI had made `server log` a compact `REQUEST`/`COMPLETE` projection and `server log --json` a
schema-three JSON stream. Two repeated failures reproduced the mismatch; retained output proved
the expected current events were present. No runtime repair was required.

## Decision

Admit shared multifamily branches and shared physical worktrees. A branch is neither an agent nor
a model family. Agents own semantic deliveries and reread current source, worktree diff, `HEAD`
and index state at their mutation and commit boundaries. `main` remains integration-only;
separate family branches remain optional provenance or isolation tools.

Keep resource locking specific to the mutable build tree, ports, CUDA device, live-model slot and
benchmark directory. Do not introduce a repository scheduler, global source lock or agent
database.

Extend evidence v1 additively with start and finish snapshots of `HEAD`, clean/dirty state and the
tracked-plus-untracked source-delta identity. Any changed field makes the evidence invalid and
both execution and report rendering return non-zero. Older v1 reports remain readable without
being granted a source-stability claim retroactively.

## Implementation

Merge commit `3a739b2e1d7b580239e1760382a609b610fe6fa1` preserves the complete MiniMax
and DeepSeek histories. ADR 0006 and the contributor/QA projections define the shared-development
contract without making `models1` a permanent architectural name.

Checkpoint `7b9302c05b3144b2d0010f88f16a31f3c150bd62` adds source snapshots, mutation
classification, invalid evidence rendering and deterministic property coverage. The test seam
uses a temporary Git repository to prove that an untracked source input changes the delta
identity; it does not rely on timing or sleep races.

Checkpoint `f14dc2ad17e22984d1c7f70cfd5fa4f68ccaefa3` aligns the DeepSeek live client
assertions with the current operator and JSON projections. The registry now truthfully declares
both external inputs required by that runner: the model artifact and runtime binding.

## After

The local `models1` history contains both family checkpoints as ancestors and one combined tree.
Common owners compile and qualify with both family surfaces present. `AGENTS.md`,
`CONTRIBUTING.md`, ADR 0006 and the QA architecture now agree that topology does not determine
semantic ownership or evidence obligations.

A stable QA run records identical clean start and finish snapshots. A moving source tree leaves
individual results available for diagnosis but cannot produce admissible green evidence. The
current DeepSeek hosted path accepts protocol v12 and the existing runtime binding v14 without a
family-specific media assumption.

## Evidence

- Canonical combined changed-file evidence
  `07a8fcdcac6dd110e27c0eca515dffc2a3e8e01045edd34c33c60668405b274e` records
  102 `PASS`, zero `FAIL`, zero `ERROR`, zero `SKIP`, and eight external-asset `BLOCKED` rows.
  Its source-stability record has identical clean start and finish identities at
  `f14dc2ad17e22984d1c7f70cfd5fa4f68ccaefa3`.
- The passing combined rows include CUDA native and no-NVCC refusal; CLI, OpenAI, REPL and tiny
  vertical integration; both sanitizer owners; architecture, ownership, layout, documentation,
  project control and topology; artifact, registry, protocol, server, runtime binding, runtime
  media, DeepSeek and MiniMax unit owners.
- Bounded DeepSeek hosted evidence
  `4dc0bf533c9a7d1662f5a3c867dcf525057f116ae7fc5e63e0270fc56a0e29cc` records one
  `PASS` on artifact identity
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53` and runtime
  binding v14 identity
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- MiniMax's CUDA joint-transformer execution source is unchanged by the DeepSeek candidate-ring
  alias repair. The accepted exact Omni/latent numerical authority remains
  `9d87e4556d92373a54188ced8bb8cca4b48a213eb8fee292eb493c9b68003e29`; the combined
  tree requalified its current common registry, artifact, runtime-media, protocol, server and
  family unit owners without launching a redundant multi-hour media generation.
- Deterministic QA properties cover unchanged snapshots, changed `HEAD`, changed dirty deltas,
  untracked inputs, visible invalid rendering, non-zero invalid status, legacy v1 reading and
  canonical `build-tree` assignment.
- Two consecutive product builds without cleaning succeeded. `git diff --check` passed. No model
  safetensors, binary payload or complete artifact is tracked; tracked GGUF files remain bounded
  parser fixtures.

The eight blocked rows are seven externally configured family live fixtures plus
`performance.runtime`, whose benchmark directory is absent. They are not cutover-gate evidence
gaps: the affected family execution is covered by current bounded or unchanged-source evidence,
and this delivery makes no performance claim. The rows remain explicit prerequisites for a future
fresh all-assets or performance qualification.

## Remaining limitations

- This closure does not promote `models1` into `main`, delete the historical family branches or
  retire their worktrees automatically.
- The repository-wide semantic cleanup, dead-code sweep, terminology rewrite and historical
  documentation pruning have not started.
- The cutover makes shared source mutation observable; it does not serialize independent source
  development or eliminate the need to coordinate true semantic overlap.
- The current DeepSeek performance milestone remains open. No performance or model-quality claim
  is created here.
- A fresh all-assets MiniMax live campaign and runtime benchmark were not required for this
  topology cutover; their existing evidence remains separately owned.

## Why it matters

Both model families can now exercise one evolving YVEX substrate immediately, while long-running
qualification fails closed if another writer changes the source it claims to validate.

```text
progression_decision: proceed
downstream_safe: true
downstream_consumer: separately reviewed total semantic cleanup on the shared multifamily tree
gate blockers: none
boundary incompleteness: none for branch consolidation, shared-development doctrine, or source-stable QA
evidence gaps: none for branch consolidation, shared-development doctrine, source-stable QA, or
  bounded common-runtime regression
deferred depth: historical branch/worktree retirement and total semantic cleanup
optimization debt: DeepSeek GB10 performance and MiniMax model execution performance remain separately owned
generalization debt: none for shared-branch doctrine; both active family consumers are present
external blockers: none for publishing models1
required repairs: none before judge review
higher-capability non-claims: no main promotion, repository cleanup, performance, quality, evaluation, or release claim
```
