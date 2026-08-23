# MiniMax Omni First-Divergence Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-23 |
| Type | repair |
| Milestone | `R010.MINIMAX.OMNI.NUMERICAL.LOCALIZATION.REPAIR.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `6fb63e0dfb2e43ee76bfa8ad3ec47907a4e4f16d` |
| Checkpoint | `b97175d7b2b1d0fad78766478c6b7d7d545a045b` |
| Subsystem | generic CUDA joint Transformer and MiniMax numeric-policy selection |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | independent source oracle, live Omni and latent execution, DeepSeek regression, sanitizers, canonical QA |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The recanonicalized MiniMax branch could execute the complete 466-row, 50-block Omni stack, but
canonical evidence `94379bcaff49fb06b91d2e8b91769cd6d93a77c69675b4773eaef714c9524225`
failed its independent source oracle. Video relative L2 was `0.263416433` with cosine
`0.966404438`; audio relative L2 was `0.641847052` with cosine `0.827596457`.

The historical evidence formed a useful contrast: three packed rows through 50 blocks passed,
and 466 rows through one block passed, while 466 rows through 50 blocks failed. After the prior
source-faithful RMS publication repair, block-one relative L2 was approximately `0.00176`, then
grew to approximately `0.00398` at block 10, `0.0105` at block 30, `0.0977` at block 40, and
`0.300` at block 50. Identical YVEX runs also began to differ around block 10. The path was
executable but neither deterministic nor source-conformant.

The authoritative post-repair reference is the untracked manual PyTorch source oracle rooted at
`latent-source-attention-466-50b-2step-s42-source-v3`. Its script SHA-256 is
`1b25df252223ea3f61e34873370f959cc80a0641eb38f4a79bed4a58cc83e821`, its manifest SHA-256 is
`c9be40d2f802f590404370497b678f3aed15f5c3ce4eb571f9b0ba179f60ecaf`, and it binds MiniMax
source revision `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08` to Transformer index SHA-256
`fb457a26ffa6294660e249b0ddd03a337f2e5393f770b5c34c8b8f90a29a7efb`.

## Problem

The request-shaped stack crossed several source-observable numeric and lifecycle boundaries that
the earlier small fixtures did not isolate. The production executor mixed streams while reusing
workspace, used generic GEMM/softmax choices where the source CUDA publication order mattered,
constructed timestep features through a different rounding path, and left final output projection
reduction policy to the locally installed cuBLASLt heuristic.

The last point made the exact model result depend on library-version heuristic choice. The
independent oracle was generated with CUDA 12 source semantics, while CUDA 13 selected different
split-K reductions for the same output heads. An execution identity that omitted that choice could
not truthfully describe the produced values.

## Localization

The bounded block and stage observers compared borrowed F32 values without changing publication
or execution policy. Before the stream repair, the first repeat-to-repeat difference appeared at
approximately block 10 and amplified through the remaining stack. Binding cuBLAS and device copies
to the operation launch stream, resetting reusable GQA workspace at its lifecycle boundary, and
reapplying the deterministic zero-workspace cuBLAS contract removed that nondeterminism. Repeated
post-repair executions are byte-identical.

The independent comparison already differed at block one, so localization proceeded inside the
earliest block rather than treating block 50 as the cause. It found source-contract mismatches in
the refiner residual publication, attention accumulation/publication, RMS grouping, and timestep
embedding. After those were repaired, the final block stages were narrowed further:

| Boundary | Before terminal repair | After |
| --- | ---: | ---: |
| Final time activation | exact | exact |
| Final AdaLN | 2 of 10,752 values differed; max absolute `7.45e-09` | exact |
| Final norm | exact | exact |
| Final hidden | 1 of 2,505,216 values differed | exact |
| Video output head | relative L2 `1.4819324e-06` | exact |
| Audio output head | relative L2 `1.23016665e-06` | exact |

The terminal first divergent semantic operation was therefore the F32 output projection with bias.
A standalone cuBLASLt probe using the same weights, inputs, shapes, and CUDA 13 installation proved
that the source-selected configurations reproduce the oracle byte for byte: video uses tile 32,
split-K 10 with in-place reduction; audio uses tile 128, split-K 3 with compute-type reduction and
the admitted stage configuration.

## Causal analysis

The defects were independent and cumulative:

- cuBLAS work and D2D publication did not share one explicit stream owner, so reusable buffers
  could be consumed before the intended producer completed;
- refiner projection and residual publication did not reproduce the source BF16 boundaries;
- attention narrowed Q/K too early and did not retain the source F32 score, warp-softmax, and
  probability-times-value accumulation order;
- RMS reduction grouping and timestep feature construction differed from source CUDA execution;
- final AdaLN relied on a heuristic BF16 bias epilogue rather than its exact admitted operation;
- the video and audio output projections allowed cuBLASLt to choose different reduction trees by
  library version, even though that choice changes output bits.

The output-head reduction tree is an irreducible source numeric fact. MiniMax selects that fact in
its graph recipe; the generic backend validates and executes it. CUDA does not infer family
topology, and the family does not duplicate the reusable linear operation.

## Decision

Make every proven source-observable choice explicit at its canonical owner. Keep the stage observer
bounded, internal, inactive by default, and family-neutral. Add a typed linear numeric policy to the
joint recipe and its execution identity, then make cuBLASLt initialization, configuration, checking,
workspace admission, and failure behavior generic.

Reject tolerance widening, compensating scheduler changes, a MiniMax-specific backend, global
device synchronization, and reliance on whichever heuristic one CUDA release happens to return.
The independent CUDA 13 PyTorch run was retained only as a diagnostic contrast because changing
all reduction heuristics changes many values; it did not replace the pinned source-faithful oracle.

## Implementation

The generic CUDA joint Transformer now:

- binds every cuBLAS operation and related asynchronous copy to the admitted launch stream;
- resets reusable workspace before each GQA allocation boundary;
- performs F32 Q/K score GEMM and scaling, source-shaped warp softmax, and F32 probability-times-
  value accumulation before the defined BF16 publication;
- matches source CUDA RMS reduction grouping and device-side timestep embedding;
- exposes BF16 publication for refiner residual addition and exact cuBLASLt bias projection;
- accepts an identity-bearing F32 output-linear numeric policy and fails closed on unsupported
  tile, split-K, reduction, algorithm, stage, or workspace facts.

The MiniMax joint recipe moved to schema V2 and selects the exact video and audio output policies.
Unit coverage verifies policy identity mutation, refusal of unsupported policy, real-shape sparse
F32 projection values, cleanup, reusable workspace reset, BF16 residual publication, source GQA
facts, and repeat execution. The live runner gained bounded block/stage checkpoints and a true
two-step latent request against external source vectors.

No public API, production file, semantic owner, family source, Physical Execution IR version,
runtime-binding version, or protocol version changed. Production files remain 284, comprising 210
translation units and 74 headers across 254 semantic owners. Five internal cross-translation-unit
symbols were added for the reusable backend operations; library globals moved from 1,498 to 1,503
and non-public globals from 1,098 to 1,103.

## After

The graph-bound 466-row, 50-block Omni request now matches its independent step-zero oracle exactly:

- video maximum absolute, relative L2, and scaled maximum are `0`; cosine is `1`;
- audio maximum absolute, relative L2, and scaled maximum are `0`; cosine is `1`;
- output SHA-256 values are
  `ceb890960d96bcb75e83361c99953300e222a0a6bf654572faac2cb755789a94` and
  `1841ddc1c4e5e8292f4020128f463a0fb5e4b5f1d1ca7edead3f76db969192c0`;
- 50 blocks complete with 3,595 kernel launches, 271,739,396 peak operation bytes, and no OOM.

The full two-step latent trajectory with seed 42 also matches exactly. Its video and audio SHA-256
values are `b0388b0896923f8f0cb764b4ca5e6a3e6f3512da9ac9ddc74143e58599113ab1` and
`ca9b5616ab217b8253effc153db7fc7b36739b9ef7528afaead2b2f1d3e07306`. Two consecutive focused
runs produced those same identities. The accepted plan, layout, latent, and Transformer-chain
identities are respectively `a6ec3aa32c33397b200020de4c93a13388b7bbe240bf92ec1f0044568767bdf9`,
`4e23ba2d45b1f7e70740992c3788c26931bf5792129af5bbe86abf84f5a622b4`,
`74f8f887bc15124b18fabe79dcbb3a4314fad41e7365a69b65754b2be7a1d4e0`, and
`b222ddb4103121aacc996b352e06e852e03089ed39d0d9fb7b08af50a78c924d`.

The small untracked Omni projection is derived only from authenticated source-v3 step-zero inputs
and velocity outputs. Its semantic projection identity is
`448b009fff8c1ff53c3201437b60362195b790959f4fed7df69bd65c8a8e6ff0`; its local manifest file
SHA-256 is `34a87c902608ebf7959989051229eea9d3672ce4ee68ecf181573f0b1b73b8b6`. It does not use YVEX
output as an oracle and does not alter the original bundle.

## Evidence

- Canonical branch-delta QA run
  `9d87e4556d92373a54188ced8bb8cca4b48a213eb8fee292eb493c9b68003e29` records 109 `PASS`, one
  `BLOCKED`, and zero `FAIL`, `SKIP`, or `ERROR`. It binds source delta
  `db8d66510117d7720d6db34250026f8e1ed251556a33e8c5b56946f4be105fcd` and build identity
  `026eb5aaf4970cf117f5752494c9aa2dc48d17700fa62d1089cf09d8f38ce1ef`.
- `live.minimax.omni-transformer-artifact` passed exact 466-row, 50-block comparison with residency
  identity `d9b1c2225f651d97020d74521347fdc3e2702b916db5837651f6f3f10d909ba6` and execution identity
  `20070775fd4c89ec3b5bcf9d69cc8e97295f913a248ec9d2291d4452b430986e`.
- `live.minimax.latent` passed exact 466-row, 50-block, two-step comparison with 7,190 launches and
  272,147,972 peak device bytes.
- `live.minimax.text`, `live.minimax.tokenizer`, Audio VAE, Visual VAE, CUDA numeric, no-NVCC,
  architecture, ownership, layout, repository, and project-control obligations passed.
- Bounded DeepSeek generation passed on the combined tree using authenticated artifact identity
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53` and runtime-binding identity
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Quant ASan/LeakSanitizer and UBSan passed in 387.7 seconds; runtime ASan/LeakSanitizer and UBSan
  passed in 196.1 seconds.
- Two consecutive builds and two consecutive documentation/guardrail checks passed without
  cleaning. Source-manifest parity and tracked-weight scans passed.

## Remaining limitations

- `performance.runtime` remains `BLOCKED` because `YVEX_RUNTIME_BENCHMARK_DIR` is not configured.
  It is performance-only and does not block this correctness repair or combined-tree promotion.
- This repair does not claim a new performance result, source-scale media-quality rerun, release
  qualification, or new product capability.
- Hugging Face platform acquisition, prepared-DiT or conditioning caches, decoder caching, SSD
  prefetch, two-slot Transformer streaming, new quantization, and other performance frontiers remain
  deferred and were not implemented.

## Why it matters

The active MiniMax vertical no longer depends on hidden stream races or library-version GEMM
heuristics for its result. The first divergent operations were localized, their source contracts
became identity-bearing execution facts, and both the complete Omni stack and recurrent latent
consumer now pass independent exact evidence without weakening the oracle or its tolerances.
