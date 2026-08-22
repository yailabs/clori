# MiniMax Main Recanonicalization 2

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.MAIN.RECANONICALIZATION.1` |
| Branch | `feature/minimax-h3` |
| Baseline | `99c971c4bb4529d5df8ab38748ff1c08df42df35` |
| Checkpoint | `9c6d1a79151092eed89a57e679983a898bd75732` |
| Subsystem | common compiler, graph, runtime, CUDA, server, operator, and QA substrate |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | canonical QA, live component admission, sanitizers, structural guards |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The published MiniMax branch retained the complete native FL2VA vertical and the repaired
persistent media-model admission boundary. Its server opened the tokenizer and four immutable
component views before readiness, sealed one composite identity, and kept materialization and CUDA
residency phase-scoped. It did not claim that the approximately 144 GB source population was
simultaneously resident on GB10.

The common substrate had meanwhile advanced independently. Main was 120 commits ahead of the
shared merge base while MiniMax was 64 commits ahead. Main carried newer compiler/family
consolidation, Physical Execution IR v4, runtime binding v14, local protocol v11, compatible
session batching, prefix/state ownership, CUDA residency and qtype repairs, and expanded server
lifecycle and QA authorities. MiniMax still expressed some valid behavior through earlier generic
owner shapes.

## Problem

A textual merge alone could retain two incompatible interpretations of the same common contracts.
Git reported conflicts in server, runtime, CUDA, manifests, command registration, and tests, but
some higher-risk ownership mismatches occurred on non-conflicting lines. Preserving an old generic
MiniMax mechanism would have recreated parallel compiler, residency, scheduler, or server
authority; choosing main wholesale would have discarded valid FL2VA semantics and its readiness
repair.

The branch therefore needed one architecture recanonicalization: current main remains the common
authority, while only source-derived MiniMax facts and FL2VA composition remain family-owned.

## Causal analysis

The largest conflicts came from independent evolution around the same runtime boundaries:

- Main replaced the old server worker lifecycle with generic session scheduling, compatible
  batching, state storage, and protocol v11. MiniMax had added a media registry and conversational
  request lifecycle to the earlier host.
- Main expanded generic component, residency, sampling, and latent owners. MiniMax had placed a
  reusable component-session lifecycle in residency and deterministic normal latent creation in
  sampling because those were the available owners when the vertical was built.
- Main centralized CUDA kernel-function identity and transformer interfaces. MiniMax still carried
  local declarations and a local identity helper.
- Both histories modified the operator registry, ownership manifests, server CLI, and acceptance
  tests without textual conflicts covering every semantic dependency.

These were chronology-induced ownership mismatches, not evidence that MiniMax required a separate
runtime. The existing common owners could represent the retained behavior after moving each fact
to its current canonical boundary.

## Decision

Consume main as the authority for compiler representations, Physical Execution IR, runtime
binding, sessions, state and prefix lifecycle, batching, backend mechanics, CUDA residency,
server scheduling, protocol, operator registration, QA, and project control.

Retain MiniMax authority for pinned FL2VA source facts, component identities, tokenizer behavior,
Qwen conditioning, Omni geometry and iteration, Visual and Audio VAE semantics, media profiles,
and family-specific numerical contracts.

Preserve the semantic result of checkpoint
`2277236d032eac9e0188806b1c18b88428d0d2a5`: MiniMax media readiness still requires the tokenizer
and four component admissions under one deterministic composite identity. Mapped views remain
distinct from materialized payload and device residency. The implementation may consume current
generic owners, but readiness must not regress to path-only configuration or impossible eager
full-payload residency.

Do not implement prepared conditioning or DiT caches, decoder caches, SSD prefetch, two-slot
Transformer streaming, or another performance frontier during this integration.

## Implementation

Merge checkpoint `9c6d1a79151092eed89a57e679983a898bd75732` joins both published histories without rebasing or
rewriting either one.

The reconciled server uses main's generic scheduler, session lifecycle, state store, protocol v11,
and response publication rules. Media requests enter that scheduler through the retained generic
media registry and are excluded from incompatible batching. Startup opens the persistent composite
media model before readiness; stop and finish cancel and close media state exactly once alongside
the common scheduler and session owners.

The reusable component-session lifecycle moved from generic residency into the existing generic
component owner. Deterministic normal latent initialization moved from sampling into the existing
latent owner; the shared PCG generator remains a bounded internal sampling ABI because both latent
initialization and token sampling consume it. These moves retain the numerical identity and
refusal contracts while removing chronology-shaped ownership.

CUDA graph execution now consumes main's kernel-function identity and canonical transformer ABI.
The merge did not add a MiniMax backend family, make CUDA infer FL2VA topology, or apply DeepSeek
expert-worklist and MoE policy to MiniMax.

Manifest and registry conflicts were resolved at their handwritten authorities and regenerated.
The combined operator registry retains main's parallel-session flag and MiniMax's media output
root; its deterministic identity is
`dc03dc51762b27b5d885b1c0853f6a358a0949a2c5ceee2fd612c518f6a570a1`.

## After

Current main is an ancestor of the reconciled MiniMax checkpoint. One tree now contains current
common compiler, graph, runtime, backend, server, operator, and QA owners together with the
accepted MiniMax FL2VA vertical.

Physical Execution IR remains v4, runtime binding remains v14, and local protocol remains v11.
The integration did not introduce a new identity-bearing fact requiring another version. The
MiniMax composite media model continues to report one model open, four artifact opens, one sealed
admission, and zero payload materialization or device-residency claim before readiness through the
current telemetry projection.

The structural hard caps remain unchanged: 2,000 lines per production translation unit, 600 per
header, and 200 per function. Ownership and generated source membership reconcile 284 production
files, 210 translation units, 74 headers, and 254 semantic owners without a new family source.

## Evidence

Canonical branch-delta QA evidence
`1bc24b01a62e84c5ce13e186be35f59ccd10b94d5217d1b19f3df4dbae9ae42b` resolved 110 tests across
CI, CUDA, fast, numeric, performance, runtime, sanitizer, and structural lanes. It records 102
`PASS`, eight `BLOCKED`, and zero `FAIL`, `SKIP`, or `ERROR`.

The passing set includes MiniMax and DeepSeek unit contracts, native CUDA, no-`nvcc` refusal,
runtime and quantization sanitizers, CLI, OpenAI, REPL, tiny vertical integration, source
ownership, layout, architecture, documentation, operator registration, and project control.
Focused live admission over the current external artifacts additionally passed:

| Scope | Evidence identity | Result |
| --- | --- | --- |
| MiniMax Audio VAE artifact | `9e24318094c8bf1d0b57e82eccfe2897d1a75ef6e89a46dc0e991d25cd4b4627` | PASS |
| MiniMax Visual VAE artifact | `ab0de7529958fadd2f1d73b1dd0cf8208c0ceffe0d573ba028ced38299b05d00` | PASS |
| MiniMax tokenizer artifact | `265d6bc6e20ad9c7a24af0655489dd03d8fd69140c8633cab619d1051f2772bd` | PASS |

The configured Omni bundle available locally predates the source-precision repair recorded at
`9b2b5970648c6ae99c81a3d2b066ad508f3982d2`. It is not a valid post-repair oracle and therefore
does not promote the canonical Omni or latent rows. Those rows remain `BLOCKED`, as do live
DeepSeek generation and performance characterization without their configured external artifacts
and baselines. No blocked row was converted into a pass or replaced by a synthetic claim.

Two consecutive compositional builds completed without cleaning. `git diff --check`, tracked
weight scans, ownership and generated-manifest checks, operator-registry generation, and the
affected documentation and architecture guards passed.

## Remaining limitations

- The independent post-repair Omni and latent oracle bundle remains unavailable to the canonical
  QA registry, so those live rows are an evidence gap rather than a regression claim.
- Live DeepSeek generation and performance characterization remain blocked by absent configured
  artifact, binding, and benchmark assets. Existing shared-owner software, CUDA, numerical, and
  sanitizer regressions pass; no new performance claim is made.
- Prepared-DiT and conditioning caches, Visual VAE decoder caching, two-slot Transformer streaming,
  streamed Qwen prefetch, and SSD asynchronous prefetch remain deliberately unimplemented.
- This checkpoint does not promote MiniMax to main, establish release support, improve generation
  latency, or replace the accepted source-scale visual qualification with a new media run.

## Why it matters

The repository again has one common executable substrate for both active verticals. MiniMax keeps
its source-derived FL2VA truth and truthful composite-model readiness without reviving obsolete
generic owners or importing DeepSeek-specific physical policy. The remaining uncertainty is
explicitly evidence-bound and can be judged separately from architecture integration.

## Communication projections

### Short update

MiniMax-H3 now consumes current main's compiler, Physical IR v4, runtime binding v14, CUDA,
session scheduler, and protocol v11 while retaining its native FL2VA semantics and phase-scoped
media-model admission. Canonical QA has no failures; post-repair Omni/latent oracle assets remain
an explicit evidence gap.

### Quoteable technical facts

- "One common YVEX substrate now represents both current verticals without a family-specific
  runtime."
- "MiniMax readiness still admits four immutable component views without claiming 144 GB of
  simultaneous device residency."
- "The integration records 102 canonical passes and preserves eight external-asset blockers as
  blockers."
