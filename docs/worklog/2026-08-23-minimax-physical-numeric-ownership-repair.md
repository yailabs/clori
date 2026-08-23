# MiniMax Physical Numeric Ownership Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-23 |
| Type | repair |
| Milestone | `R010.MINIMAX.PHYSICAL.NUMERIC.OWNERSHIP.REPAIR.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `711bd161ac3dd8093557896770245b594673df76` |
| Checkpoint | `8be966557fd03c9a08aca53c94d2cb3f94af13c3` |
| Subsystem | graph physical lowering and generic CUDA F32 linear execution |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | exact Omni and latent oracles, DeepSeek v14 compatibility, sanitizers, canonical QA |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The accepted numerical checkpoint already made the graph-bound 466-row, 50-block MiniMax Omni
execution and its two-step latent consumer byte-exact against the independent source oracle. The
video output projection used cuBLASLt algorithm 10, tile 32x32, split-K 10, and in-place reduction;
the audio projection used algorithm 20, tile 128x32, split-K 3, compute-type reduction, and the
8x5 stage configuration.

Those execution-significant facts were embedded in `yvex_transformer_joint_recipe` as
`video_output_numeric` and `audio_output_numeric`. The semantic joint-Transformer header therefore
depended on the backend header, the family recipe carried CUDA algorithm policy, and the physical
choice was not represented by the canonical compiler owner. This was numerically correct but
architecturally inverted.

Physical Execution IR was version 4 and the serialized runtime binding was version 14. The MiniMax
media component does not consume that serialized runtime-binding path, while the authenticated
DeepSeek artifact does. Extending either schema solely to transport a MiniMax component fact would
have created a false dependency and needlessly invalidated accepted DeepSeek evidence.

## Problem

The prior repair proved that reduction configuration changes model output bits. Tile, split-K,
reduction, algorithm, stage configuration, workspace, backend, and target hardware are therefore
physical identity facts. Leaving them in semantic recipe state made model semantics own backend
mechanics; leaving their reconstruction to CUDA would make the backend a second planner.

The required invariant is now explicit: the family recipe describes the mathematical operation;
the graph compiler selects and seals one physical implementation; component execution carries the
immutable plan; CUDA validates and executes it without family knowledge or heuristic substitution.

## Causal analysis

The numerical defect had already been repaired, but its implementation placed the remedy in the
semantic recipe because that was the shortest path from the MiniMax adapter to CUDA. That path
made a backend-owned struct part of family semantics and left no canonical compiler identity for
the exact reduction tree. Since changing split-K or reduction changes output bits, the missing
physical identity was not documentation debt: it was an executable ownership defect.

The MiniMax component boundary does not read a serialized runtime binding. The minimal causal
repair is therefore an immutable compiled operation plan carried through the existing component
request, not a Physical Execution IR or binding wire migration. This preserves the exact algorithm
while removing both the semantic dependency on CUDA policy and the opportunity for runtime or
backend reconstruction.

## Decision

Use a canonical compiled operation physical plan adjacent to Physical Execution IR rather than
extend the per-terminal Physical Execution IR v4 schema. The new generic
`yvex_transformer_linear_physical_plan` is produced by graph physical lowering, carries semantic
operation and physical identities separately, and is copied by value into each admitted joint
Transformer request.

The MiniMax graph adapter selects two generic SM121 profiles before component execution. It does
not place tile, split-K, reduction, or workspace fields in the semantic recipe. Runtime component
code carries the already-sealed values and does not derive them. CUDA admits only the exact
supported profile and hardware combination, then instantiates that configuration explicitly.

Physical Execution IR remains v4 and runtime binding remains v14. The new plan has an internal
pointer-free schema v1, but it is not added to the runtime-binding wire format because the MiniMax
component boundary is not a serialized runtime-binding consumer. Old v14 bytes are neither
reinterpreted nor migrated.

## Implementation

The semantic joint recipe moved from internal schema V2 to V3 and no longer includes the backend
header or `yvex_backend_linear_numeric_policy`. Source-derived QKV layout, SwiGLU ordering,
geometry, dtypes, and output widths remain semantic facts.

The graph Transformer owner now compiles and seals a generic F32 bias-linear physical record. Its
operation identity covers semantic domain, operation role, input width, and output width. Its
physical identity additionally covers implementation family, reduction, stages, backend,
algorithm, tile geometry, split-K, compute capability, workspace, and deterministic/exact flags.
Resealing detects stale identities.

The MiniMax graph adapter compiles the accepted decisions:

| Output | Operation identity | Physical identity | Algorithm | Tile | Split-K | Reduction | Stages |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Video 96 | `ac89181af1a8eb6fab4f718edb32ee673f35a430cb9899240b4c702ac6683649` | `36060b7776c4fe2b3bbd53b88c3bf9e064d458769ba4123c345810301022ede0` | 10 | 32x32 | 10 | in-place | default |
| Audio 32 | `64eb032840698822a63ea1e6a76bcc6b616d52186afaaf45bb022be1b240654a` | `665632fa75ff3173c2813bccc0f90fb53fa0eb24e99fd950e1e3426dd370344b` | 20 | 128x32 | 3 | compute-type | 8x5 |

Both require CUDA, compute capability 12.1, F32 bias-linear execution, one MiB of workspace, and
the exact deterministic numeric contract. The component path compiles the pair once before latent
iteration and carries the plans through every production evaluation.

Generic CUDA validates the sealed identity, operator geometry, implementation, workspace, and
exact SM121 capability before marking output mutable or allocating work. It initializes, sets, and
checks the selected cuBLASLt algorithm explicitly. Unsupported profiles, stale identities,
mismatched dimensions, wrong output roles, insufficient compatibility, or an absent F32 plan fail
closed. F32 execution cannot silently fall back to a new heuristic. No MiniMax switch exists in
generic runtime or CUDA.

Focused tests prove compiler emission, semantic/physical identity separation, tile, split-K, and
reduction identity mutation, stale-plan refusal, unsupported-profile refusal, wrong-output binding
refusal, no-plan heuristic refusal, exact workspace cleanup, and generic CUDA execution.
Architecture guards prevent backend policy from re-entering the semantic recipe and prevent
MiniMax selection from entering generic CUDA or runtime.

No public ABI, production file, semantic owner, artifact, protocol, Physical Execution IR wire
schema, or runtime-binding wire schema changed. The repository remains at 284 production files,
210 translation units, 74 headers, and 254 semantic owners. The added generic internal compiler
ABI moves library globals from 1,503 to 1,506 and non-public globals from 1,103 to 1,106.

## After

The exact 466-row, 50-block Omni request remains source-conformant:

- video maximum absolute, relative L2, and scaled maximum are `0`; cosine is `1`;
- audio maximum absolute, relative L2, and scaled maximum are `0`; cosine is `1`;
- video SHA-256 is `ceb890960d96bcb75e83361c99953300e222a0a6bf654572faac2cb755789a94`;
- audio SHA-256 is `1841ddc1c4e5e8292f4020128f463a0fb5e4b5f1d1ca7edead3f76db969192c0`;
- execution identity is `978f13eb74661762560d2e575f0067a13060d548c1334a63f84cf59da25c2a91`.

Two consecutive executions produced the same output identities. The two-step latent trajectory is
also exact and repeatable: video
`b0388b0896923f8f0cb764b4ca5e6a3e6f3512da9ac9ddc74143e58599113ab1`, audio
`ca9b5616ab217b8253effc153db7fc7b36739b9ef7528afaead2b2f1d3e07306`, and Transformer-chain
identity `ad3a20bda3e832ebed9cfa7655f94b22725804603a7e90921ad861f062244128`.

The authenticated DeepSeek runtime-binding v14 identity
`31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b` remains admitted and
completed target-only and verified DSpark generation. The repair does not require re-materializing
an artifact or migrating accepted binding bytes.

## Evidence

- Canonical wave-delta QA evidence
  `b7126364a67df62c68a5590fe975e988adb62582963605a154a84ec41933cf53` records 109 `PASS`, one
  `BLOCKED`, and zero `FAIL`, `ERROR`, or `SKIP` across CUDA native/no-NVCC, live models,
  numeric, sanitizers, architecture, runtime, and unit obligations.
- Exact Omni evidence `3bb7b5b77942e4ed0002cced402a516a72529ac47e6727f5f766a2ea0f948f30`
  and repeat `01dabed8cc5159c7adfe55d3369a48a32e54e9cc51f34dadd5435fa5b04c29b1`
  retain zero error and byte-identical outputs.
- Exact latent evidence `c6c57b3f9a3e327553c0522007ba395f46d86fb02cf83e42fa0e4f74a0d3f89a`
  and repeat `dbc79dc4f0e894a6351a2697cb746bdece60342e8b4b80a7400d3a4d1778d88b`
  retain the accepted two-step identities.
- Independent DeepSeek compatibility evidence
  `d7b97590d0d80f546c32563f3d3079c0e3229b69c294ac1af509ee06a1cfe0e5` passed, and the same
  lane passed inside the canonical report.
- Focused unit and CUDA policy evidence
  `298bbc5eed511db03b37f196974729f77bda3cdbba306420449d5c32690467f7` and
  `73e860552dba6416d1a39147dcf347a75f1b7be5cef8da542cc75f65b131321d` passed.
- Quant sanitizer evidence
  `aef65b503b2f4d975021fa74a334d94b7c03119dbbde92655883100c0aff3236` and runtime sanitizer
  evidence `9271a676599966257b9001574e188fbb5e821c30a6c2072bfdf4be403b4cb2ec` passed independently;
  both also passed in canonical QA.
- Source layout, ownership, architecture, manifest parity, tracked-weight scans, and
  `git diff --check` passed after the implementation checkpoint.

The QA report's sole `BLOCKED` row is `performance.runtime`, because
`YVEX_RUNTIME_BENCHMARK_DIR` is not configured. It is a performance-only external asset gap and
does not weaken this correctness-preserving ownership repair.

## Rejected alternatives

- Extending Physical Execution IR v4 or runtime binding v14 was rejected because the MiniMax
  component path does not serialize this plan and the change would create a fake wire dependency.
- Retaining the backend policy in the family semantic recipe was rejected because it confuses
  model meaning with hardware execution.
- Moving tile and reduction selection into runtime was rejected because runtime is a carrier and
  compatibility validator, not a planner.
- Reconstructing policy in CUDA from family, output width, qtype, or row geometry was rejected
  because it creates a second physical authority.
- Silent cuBLASLt heuristic fallback was rejected because the prior repair proved that reduction
  selection changes output bits.
- Widening tolerances or regenerating the oracle was rejected because this wave changes ownership,
  not accepted numerics.

## Remaining limitations

- No performance result or speedup is claimed. The existing runtime benchmark asset remains
  unavailable to canonical QA.
- Hugging Face acquisition, model-prepare UX, caching, prepared DiT, decoder caching, SSD/two-slot
  streaming, new quantization, and performance optimization remain deferred.
- Media-server startup code was not changed; server socket readiness therefore did not require a
  new lifecycle implementation. Live component admission and cleanup remained qualified.
- Physical Execution IR v4 and runtime binding v14 remain intentionally unchanged. A future
  serialized consumer of compiled operation plans must introduce an honest versioned wire
  boundary rather than reinterpret this internal schema.

## Why it matters

MiniMax exact output no longer depends on a CUDA policy hidden in model semantics. The accepted
reduction trees are compiler-sealed, identity-bearing, fail-closed physical facts consumed by one
generic backend path. This restores the semantic/compiler/runtime/backend ownership chain before
the MiniMax vertical is considered for promotion into `main`.

```text
progression_decision: proceed
downstream_safe: true
downstream_consumer: judge-reviewed promotion of feature/minimax-h3 into main
gate blockers: none
boundary incompleteness: none
evidence gaps: performance.runtime only; performance is outside this correctness gate
deferred depth: acquisition, caching, streaming, quantization, and optimization retain later owners
optimization debt: unchanged; no performance claim
generalization debt: none for the ownership boundary; DeepSeek v14 supplies cross-family regression
external blockers: YVEX_RUNTIME_BENCHMARK_DIR absent for the non-required performance lane
required repairs: none before correctness promotion
higher-capability non-claims: no performance, release, acquisition, caching, or streaming closure
```
