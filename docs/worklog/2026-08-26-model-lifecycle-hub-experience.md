# Model lifecycle hub experience checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-26 |
| Type | checkpoint |
| Milestone | YVEX.MODEL.LIFECYCLE.HUB.EXPERIENCE.0 |
| Branch | models1 |
| Baseline | 1865659744672f5c5e841527caaf467758dfbedc |
| Checkpoint | 913785507235deb1f5fb46128eba38dce6f0055e |
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
The acquisition help exposes immutable revision selection and repeatable include/exclude patterns,
so a representation selector reported by remote inspection maps directly to an explicit payload
acquisition rather than an implicit whole-repository download.

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
- Canonical `integration.cli`: PASS at `913785507235deb1f5fb46128eba38dce6f0055e`, evidence
  `54ed1467ecc20012bc386a6c256766a4fe98bea94a13e1cb48a821ca2cd3b7d6`.
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

## Glow-up continuation

Checkpoint `543237e390da3cd6248bb15a594a629addb0111a` repairs the product model over
the same lifecycle boundary. Provider order is retained only as one ranking
input. The YVEX catalog now scores normalized exact-name matches, qualified
canonical source repositories, artifact kind, family evidence, support stage,
exact local lifecycle state, metadata completeness, and provider rank. Equal
scores fall back to provider rank and repository identity, so ordering is
deterministic. The qualified-source catalog supplies generic canonical-publisher
evidence; there is no query-specific MiniMax ranking branch.

Remote records now distinguish full models, conversions, adapters, components,
deltas, derivatives, and unknown artifacts. Kind evidence retains whether the
classification came from the qualified source catalog, provider metadata or
lineage, or a provisional repository-name hint. A related LoRA may retain
`minimax-h3` family affinity without acquiring the full-model identity
`MiniMaxAI/MiniMax-H3`. Repositories and their safetensors or GGUF
representations remain separate records. Filename-derived GGUF qtypes render as
provisional and stay structurally marked in JSON until local GGUF admission.

Historical `yvex.source-acquisition.v1` manifests and composite-package
`repository.json` provenance are now indexed read-only. Exact provider,
repository, and immutable revision identities reconcile old source and package
state without relying on directory names. The live provider default advanced
after the original checkpoint: on 2026-08-26 `MiniMaxAI/MiniMax-H3` resolved to
`42ed227ee7df40d41602854ae760620d6eb651fe`, while the qualified local source
and package remain at
`b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08`. Normal inspection therefore
reports the local assets as another revision rather than claiming either
`LOCAL=no` or an exact installation. Explicit inspection of `b8b09e34...`
reconciles both the acquired source and package.

The primary search table now projects repository, kind, family affinity,
parameters, representation classes, local lifecycle, and product-facing YVEX
status. Internal stages remain in audit/JSON. Inspection resolves a default or
explicit reference to an immutable revision, separates repository-not-found
from revision-not-found, and presents model, revision, local lifecycle, and
representation sections. Remote JSON and operator result schemas advance to
`yvex.remote-model-catalog.v2`; local JSON is
`yvex.local-model-catalog.v2`. Engine state is still observed exclusively
through the Main-owned engine inventory. Absence of a reachable host remains
`not-observed`; no process inference or duplicate engine cache was introduced.

Focused deterministic provider and CLI coverage passes for adversarial provider
ordering, the full remote-kind taxonomy, exact and other-revision local
reconciliation, default and explicit revision resolution, revision-specific
refusal, typed JSON, compact tables, provisional GGUF qtypes, zero discovery
payload downloads, and token redaction. Live metadata-only smoke placed the
canonical MiniMax full model first, reconciled the historical source/package,
and inspected `unsloth/MiniMax-H3-GGUF` without acquisition; all GGUF qtypes
remained filename evidence. No model payload, engine, server, GPU workload, or
benchmark was started by the continuation.

The continuation observed four committed Main-session HEAD advances before
`543237e` and preserved all of them. It then encountered a Main-owned
execution-batching delta while uncommitted; one combined build was correctly
classified `SOURCE MUTATED / EVIDENCE INVALID` when that delta changed an ABI
mid-build. Lifecycle objects compiled independently and the focused suite
passed, but final combined QA remains deferred until that shared Main boundary
is committed and source-stable. No Main-owned runtime, graph, or unit-test hunk
was staged in the lifecycle commit.

Main subsequently published `84fb6a075599ba28bfaee957809e1060db50f3a9`
and left only this worklog modified. On that stable source, two consecutive
`BUILD_DIR=build/model-hub` builds, the complete isolated model CLI/provider
suite, operator-registry generation/refusal, source ownership, documentation,
project control, and `git diff --check` passed. The exact focused source
snapshot remained `84fb6a0` with delta identity
`0a0c34880d1ef83febca205200052b80f7badb1a7c9c73b5c8b6303b07dd3446`
through those checks.

Whole-tree layout and architecture remained Main-owned blockers at that
snapshot: eight aggregate layout limits, the pre-existing CLI/client include
cycle, public and non-public global-count ceilings, and the single-consumer
`yvex_server_telemetry_identities` declaration were still outside policy. A
canonical 110-row QA run then overlapped a new uncommitted Main execution and
runtime reduction spanning 13 source files. QA correctly marked evidence
`69a91fe21957dd5d39fea24d6c9df8240b0759ac8bb2b038617cc1cbdfc57641`
as `SOURCE MUTATED / EVIDENCE INVALID`; its cascading build failures are not
Model Hub evidence. The eight live/performance rows were independently
`BLOCKED` by unconfigured external assets. The continuation therefore closes
with focused lifecycle qualification PASS, whole-tree final qualification
blocked by shared Main WIP, five committed Main HEAD advances observed, five
source-mutated attempts retained, and zero lost foreign changes.

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
