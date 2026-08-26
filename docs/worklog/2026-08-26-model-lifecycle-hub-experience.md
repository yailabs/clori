# Model lifecycle hub experience checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-26 |
| Type | checkpoint |
| Milestone | YVEX.MODEL.LIFECYCLE.HUB.EXPERIENCE.0 |
| Branch | models1 |
| Baseline | 1865659744672f5c5e841527caaf467758dfbedc |
| Checkpoint | 6960b26f6d457a6c55302e092dadb9defc964c63 |
| Subsystem | model.catalog, accounts.provider, cli.models |
| Model family | multifamily |
| Hardware | not-applicable |
| Evidence | software tests; bounded live provider metadata smoke |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

YVEX could acquire known provider repositories, preserve resumable provider process state,
inspect local artifacts, prepare packages, and register hosted profiles. Those operations were
separate command lanes. The model namespace had no provider-neutral remote discovery record,
no representation inventory, and no local projection that kept acquired sources, admitted
packages, and live engines visibly distinct. Operators needed repository history to know which
command was legal next.

The shared `models1` worktree already contained an independent, concurrent refoundation of the
persistent server and engine-generation boundary. That work was uncommitted and intentionally
outside this checkpoint.

## Problem

Remote repository existence, source representation, local acquisition, package readiness, and
engine state were being presented through unrelated surfaces. A repository name or local file
could therefore be mistaken for YVEX support, and a registered package could be mistaken for a
resident engine. Remote safetensors and GGUF files also lacked one typed comparison surface:
filename-derived GGUF qtypes and provider safetensors metadata could not be distinguished from
verified local container facts.

## Causal analysis

The acquisition subsystem already owned credentials, provider child lifecycle, progress,
interruption, resume, cleanup, safe paths, and source receipts. Replacing it would have duplicated
working lifecycle machinery. The missing boundary was metadata discovery before acquisition and
composition after acquisition: provider output needed normalization into domain records, while
local receipts and package entries needed a read-only catalog that did not open payloads or
engines.

The host API did not expose a stable engine inventory when the lifecycle work began. The lifecycle
therefore retained an isolated handoff until the Main Session committed protocol v13 engine
generations and their read-only inventory. Consuming that committed boundary later in the same
delivery avoided both a speculative server ABI and a second engine owner.

## Decision

`model.catalog.remote` owns provider-neutral remote model, exact revision, representation, file,
lineage, and support-stage records. Hugging Face is the first adapter and is invoked through the
existing bounded provider process authority. Discovery never receives credential bytes as command
arguments and never downloads model payloads.

`model.catalog.local` composes acquisition receipts and package registry entries. When the
persistent host is reachable, the CLI projects the committed engine inventory onto package
aliases; otherwise it reports `not-observed`. The catalog does not infer serving activity from
package readiness and never opens an engine.

Remote GGUF qtypes derived from filenames remain provisional. Acquired GGUF must pass YVEX
container, family, role, qtype, layout, and package admission. Safetensors precision uses provider
metadata when available and otherwise remains unknown. Provider-proven base-model metadata is
retained; visual repository-name similarity does not create lineage.

## Implementation

Public typed records now distinguish remote models, representations, exact remote files, acquired
sources, packages, support stages, and local engine observation. The Hugging Face adapter consumes
machine JSON from the official provider CLI, bounds output and pagination, validates repository and
file paths, resolves immutable revisions, and maps provider failures without persisting raw output.

The CLI adds table-first `yvex model search` and `yvex model inspect` projections plus typed JSON
and audit views. Interactive search is terminal-only and drills into the same inspection API;
scripts never need to drive prompts. `yvex model list` now combines source-acquisition receipts and
the model registry without opening payloads. It consumes protocol v13 `ENGINE_LIST` through a
CLI-owned resolver, so the model domain and renderer remain independent of the server ABI.
Existing granular acquisition, preparation, inspection, and registry commands remain intact.

Acquisition receipts now count GGUF files separately from safetensors. The local catalog directs
an immutable safetensors source toward compilation/package preparation and an immutable GGUF source
toward compatibility inspection followed by direct admission or explicit repack.

Two concurrent shared-file edits were isolated by hunk. `config/source_owners.tsv` retained the
other delivery's server-engine owner while adding the two catalog owners. The operator registry
retained the other delivery's host/load/unload edits while adding only model discovery, inspection,
and local-catalog records. After the host cutover committed, its generated command migration and
registry refusal tests were aligned with `server load`, `server models`, and `/models`; no server or
runtime implementation was modified by this delivery.

## After

The same typed lifecycle can now be projected to CLI, JSON consumers, or a later graphical model
interface:

```text
remote model -> representation -> exact revision -> acquired source
             -> verified source or inspected GGUF -> YVEX package
             -> local catalog -> committed engine handoff
```

Search reports external availability without claiming YVEX support. Support stage does not claim
local presence. Local presence does not claim package readiness. Package readiness does not claim
engine residency. Existing safetensors and GGUF acquisition paths remain distinct and converge only
at the admitted package boundary.

## Evidence

- `tests/cli/models.sh`: PASS on repeated exact-tree runs. It covers provider search,
  pagination, empty results, unknown and recognized families, gated/authentication failures,
  provider failure, malformed output, exact revision refusal, safetensors and GGUF
  representations, multiple GGUF qtypes, sharded safetensors, unknown sidecars, local catalog,
  acquisition, resume/cancellation owners, and token redaction.
- Canonical `integration.cli`: PASS at `6960b26f6d457a6c55302e092dadb9defc964c63`, evidence
  `cb09d87a0f60523ee55f87f5a7614757ac5085802c20d3feb5732a80001d293c`.
- `tests/test_source_ownership.sh`: PASS for the two new catalog owners.
- Canonical operator registry generation and its schema/refusal/audit/discovery tests: PASS after
  the committed host command cutover was reflected in its consumer test.
- Two consecutive exact-tree `make yvex` builds completed warning-clean. Documentation
  architecture, QA registry, source ownership, and project-control guards also pass for the
  lifecycle-owned delta.
- Bounded live Hugging Face metadata smoke: `model search "MiniMax H3"` returned provider records
  without payload acquisition; exact MiniMax inspection resolved revision
  `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08` and enumerated source representations and files.
  Network timing is characterization only, not a deterministic YVEX benchmark.
- No real model payload, operator-owned server, GPU workload, or performance qualification was
  started by this delivery. The focused CLI test used only its isolated model-neutral host.

The combined branch still has Main-Session structural gates outside this delivery: its committed
engine owner raises the repository aggregate by one file/translation unit/semantic owner beyond
the lifecycle-only measured policy, and `generation_turn` plus OpenAI `handle_generation` exceed
the 200-line function limit. The active Main-Session runtime transaction rewrite also invalidated
one combined build while headers and consumers were temporarily inconsistent. Those facts do not
invalidate the exact lifecycle evidence, but they prevent a combined-tree downstream-safe claim
until the Main Session closes them.

Concurrency accounting at this checkpoint:

```text
concurrent_head_advances_observed: 3
main_session_commits_observed_during_wave: 3
same_file_parallel_edits: 2
same_file_nonconflicting_edits_preserved: 2
real_semantic_conflicts: 0
deferred_overlaps: 0
source_mutated_qa_runs: 3
lost_foreign_changes: 0
```

The engine-state overlap was initially deferred, then consumed after the Main Session committed
its API. Three source snapshots or builds were superseded while the shared host/runtime work
advanced; focused lifecycle evidence was rerun against the final exact checkpoint. Rereading the
full diff before edits and commits prevented any Main-Session hunk from entering lifecycle commits.

## Remaining limitations

- No high-level `model use` orchestration was added. Acquisition and package preparation stay
  explicit, inspectable operations.
- Remote GGUF qtype hints are not local compatibility proof. No external GGUF was promoted without
  actual admission.
- No payload download, compilation, quantization, package publication, or engine load was exercised
  against a large live model for this checkpoint.
- Engine load/unload remains owned by the committed host API; the lifecycle only observes it and
  documents the handoff.
- Main-Session aggregate layout and long-function repairs must close before the combined branch can
  be called structurally clean.

## Why it matters

YVEX now has one lifecycle vocabulary from provider discovery to the engine boundary, so a human
table and a future GUI can expose the same facts without parsing command output or promoting one
stage into another.
