# MiniMax Main Recanonicalization

| Field | Value |
| --- | --- |
| Date | 2026-08-14 |
| Type | repair |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `ee9b9fce93076ab995aee740253a3fd834ff87a4` |
| Checkpoint | `fe070edbcc14b2c7cf254769ff474a9e16548e9b` |
| Subsystem | MiniMax graph, generic component execution, tokenizer, and CUDA backend |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10; GPU regression deferred due active workload |
| Evidence | software tests; operator tokenizer acceptance; build and architecture guards |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The MiniMax branch carried its accepted source, compilation, component-artifact,
Audio VAE, Visual VAE, conditioning, latent, and bounded media work on a generic
substrate older than current `main`. Some execution details still lived in a
MiniMax CUDA family owner. The branch also relied on a direct prompt-tokenizer
policy that the newer compiled tokenizer contract could not express.

The current main line had consolidated compilation, runtime binding, server
operation, generic text execution, and repository worklog ownership. Merging
the histories textually was therefore insufficient: old MiniMax mechanisms
could compile while bypassing the new generic authorities.

## Problem

The initial merge exposed four semantic mismatches. MiniMax text conditioning
duplicated the new generic CUDA text encoder. The Omni Transformer remained a
family-specific backend projection forbidden by the current architecture.
Audio alias-decoder CPU execution sat in graph ownership even though it consumes
materialized runtime state. Finally, the compiled tokenizer assumed a
conversation template, while FL2VA requires exact `verbatim-no-special-v1`
prompt tokenization.

Treating these as textual conflicts would either restore superseded owners or
silently change accepted MiniMax semantics. The old root `yvexd` build product
also survived locally after main retired that executable, causing a truthful
surface guard to fail even though it was untracked.

## Causal analysis

The branches had evolved independently across owner boundaries, not merely
across function bodies. MiniMax implemented concrete execution before main had
the generic text, component-residency, compiler, and integrated-server
boundaries now required by repository policy. The tokenizer mismatch came from
modeling every prompt policy as a conversation rather than retaining direct
source-authored prompt behavior as a distinct typed contract.

The first Omni migration attempt placed its recipe in the broad Transformer
header. Structural analysis caught the resulting subsystem include cycle. That
failure demonstrated that the recipe needed a narrow graph/backend ABI instead
of borrowing compiler or model dependencies.

## Decision

Keep exact MiniMax geometry, weight names, recipe identity, and phase
composition in the graph family. Move only reusable execution mechanics to
generic owners. Represent the current joint-modality CUDA implementation with
an explicit recipe and fail closed on unsupported geometry rather than hiding
family constants in a generic backend.

Retain direct prompt tokenization as a first-class compiled tokenizer policy;
do not synthesize a conversation or default missing chat semantics. Keep the
server topology from main and remove only the obsolete untracked local daemon
binary. Do not disturb the active DeepSeek GPU workload to obtain redundant
live evidence.

## Implementation

Merge checkpoint `fe070ed` absorbed current main without rebasing either
published history. MiniMax text embedding and encoder execution now use the
generic backend text operations. The former MiniMax CUDA family source was
replaced by the generic `joint_transformer` backend owner and its narrow
internal recipe ABI; the MiniMax graph supplies the exact FL2VA geometry and
identity domain.

Generic runtime component execution now owns the CPU alias decoder, while the
MiniMax graph owns only its weight-name templates, recipe, and result
projection. The family catalog selects exact tokenizer policy. The tokenizer
compiler and executor admit both conversation and direct policies, preserving
the byte format and identity behavior of existing conversation policies while
binding MiniMax to `verbatim-no-special-v1`.

Focused live-runner sources were updated to consume the generic backend ABI and
public resident-allocation boundary. The obsolete untracked `./yvexd` build
product was removed after verifying that current main no longer owns or builds
it.

## After

`origin/main` is an ancestor of the MiniMax branch. MiniMax terminates at its
model and graph family projections; no MiniMax backend family source remains.
The branch builds through the current compiler/runtime/server substrate, exact
MiniMax prompt tokenization is operator reachable, and the narrow joint
Transformer recipe is identity-bound and geometry-checked.

This checkpoint preserves the already accepted MiniMax implementation boundary
on the new substrate. It does not newly establish full-model CUDA execution or
a final prompt-to-playable synchronized media workflow.

## Evidence

- `make generate-source-manifest` and `make check-source-manifest` admitted 645
  generated source-manifest lines.
- `make -j2 build/tests/test yvex` completed twice without cleaning.
- `./build/tests/test` passed the complete non-CUDA unit suite.
- Focused `minimax_h3`, `graph`, `runtime_media`, and `runtime_tokenizer` unit
  sections passed.
- `tests/cli/minimax_tokenizer.sh` passed against the exact admitted MiniMax
  text artifact and checked deterministic tokens and refusal behavior.
- The MiniMax Omni and Transformer live runners compiled against the generic
  joint-Transformer ABI.
- `make -j2 check-docs` and `make -j2 check-guardrails` passed twice without
  cleaning; ownership, layout, natural commentary, dependency, and
  architecture checks passed.
- `git diff --check` passed and tracked weight scans found only intentional
  small GGUF refusal/format fixtures, not model artifacts.

## Remaining limitations

- Evidence gap: a fresh MiniMax CUDA and media regression was not run because
  an accepted DeepSeek workload actively occupied the shared GPU. No process
  was stopped and no competing CUDA context was created.
- The checkpoint does not prove final chat-driven prompt negotiation, server
  media request routing, full Omni iteration at target scale, or synchronized
  media publication through the persistent operator workflow.
- The generic joint Transformer currently admits the one exact geometry
  required by this concrete consumer. Further geometry is not claimed.
- End-to-end MiniMax completion remains open and belongs to the resumed
  `R010.MINIMAX.H3.FL2VA.END_TO_END.0` delivery.

## Why it matters

MiniMax can now continue on the same compiler, runtime, server, tokenizer, and
development substrate as current main without preserving a parallel backend
architecture or weakening its exact FL2VA source semantics.
