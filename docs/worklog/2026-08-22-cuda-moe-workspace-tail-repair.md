# CUDA MoE Workspace and Tail Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `e9bd0866f50fab270afd63497067f046d88980a7` |
| Checkpoint | `09243d523949a1f9c39b701eb5a68a5e378eacfb` |
| Subsystem | CUDA qtype primitives and MoE workspace admission |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | exact CUDA qtype oracle; complete 43-layer MoE live; target-only and DSpark live; sanitizer; structural QA |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

One CUDA context and reusable arena served both the canonical row-batch MoE path and the
token-local numerical path. The workspace callback admitted only the row-batch allocation
geometry. A token-local execution could therefore discover that its additional gate and up
intermediates did not fit after execution had already entered the backend.

The short-row Q8 warp primitive also assigned hardware tail groups block indices beyond the
semantic block count. Those groups participated in the later warp-wide shuffle but first read
weight and activation storage using the out-of-range block index. Adjacent valid allocation made
this defect input-layout dependent rather than reliably fail-closed.

## Problem

The shared CUDA owner had two inconsistent extent contracts. Preflight did not cover every
execution path using the arena, and hardware-width lane groups were allowed to become semantic
memory indices. Either defect could surface as a live MoE refusal, incorrect read, or process
failure even though focused full-width fixtures remained green.

## Causal analysis

The token-local path allocates three intermediate ranges sized to the larger of routed-pair and
shared-expert intermediate populations. The row-batch contract already dominates its status,
routing metadata, output and auxiliary storage, but previously represented only one routed and
one shared intermediate. The missing bound was the two additional maximum-width ranges.

For short Q8 rows, not entering the group reduction from inactive lane groups was not a valid
repair: the group helper contains warp collectives and divergent participation can hang the
warp. Every lane must enter the collective. The safe hardware tail uses bounded block zero as a
valid read source, completes the collective, then discards the result before the final reduction.

Several performance hypotheses were evaluated while isolating this repair and were rejected:

- compiler-selected bucket tiling measured approximately 8.759 token/s against the retained
  9.584 token/s width-one median;
- retained Q8 activation staging measured approximately 9.69 token/s but produced no causal
  complete-profile MoE reduction;
- a four-warp grouped MoE geometry measured an approximately 8.93 token/s median.

These observations are approximately comparable or characterization-only. None is retained and
none supports a performance claim.

## Decision

Keep one shared workspace authority and make its admitted extent dominate both canonical
row-batch and token-local execution before device mutation. Keep all hardware lanes in the Q8
collective while preventing hardware tail groups from addressing semantic tail storage.

Do not retain a speculative performance path merely because it was exercised during diagnosis.
The current exact width-one regime remains authoritative.

## Implementation

The canonical MoE workspace calculation now derives the maximum token-local intermediate and
admits the two additional gate/up ranges alongside the row-batch ranges. Checked multiplication
and the existing aligned workspace accumulator preserve overflow refusal and arena geometry.

The Q8 warp-dot primitive maps an inactive group to bounded block zero for the collective and
sets its contribution to zero afterwards. Reduction order for real groups is unchanged.

Both changes remain inside the existing CUDA semantic owners. Physical Execution IR v4, runtime
binding v14, artifact identity, execution-batch authority and expert-worklist authority are
unchanged. The source change is line-count neutral and introduces no new production owner.

## After

Preflight now covers both MoE consumers of the shared CUDA arena, and short-row hardware tail
groups cannot read adjacent semantic storage. The complete 43-layer live MoE path executes all
43 layers, 258 routed experts and 43 shared experts with stable output and routing digests.

The accepted code remains the exact DP4A-class width-one execution regime. This repair establishes
correctness and lifecycle safety; it does not reduce the retained device-time floor.

## Quantitative delta

| Fact | Before | After | Evidence class |
| --- | ---: | ---: | --- |
| Workspace paths covered by preflight | row-batch only | row-batch and token-local | implementation plus CUDA/live execution |
| Short Q8 tail addressing | hardware block index | bounded semantic block, discarded tail result | exact qtype oracle |
| CUDA qtype max absolute/relative difference | exact accepted path | 0 / 0 for all admitted qtypes | measured focused oracle |
| Complete MoE layers | capacity refusal possible | 43/43 complete | measured live qualification |
| Routed/shared expert executions | not completed under the failing boundary | 258 / 43 | measured live qualification |
| Source line delta | not-applicable | 0 | repository layout evidence |
| Accepted performance claim | none | none | characterization only |

## Evidence

- Code checkpoint `09243d523949a1f9c39b701eb5a68a5e378eacfb` contains the bounded Q8 tail
  and shared workspace repair.
- Canonical source QA evidence
  `c0eed502e933b54e0d3b19038f9f470c503112cc96bf7dd5a35ff350b448c97c`
  recorded 96 PASS across native CUDA, no-NVCC refusal, complete target-only and DSpark live
  generation, numeric owners, performance characterization, ASan/LeakSanitizer and UBSan. Its
  three structural failures were only the intermediate line-budget form of the same repair.
- Canonical structural rerun evidence
  `72c75e370749848bf2d4c1b0c8327daede066aa64bccedc1be4ff2d4c57910b9`
  recorded 14 PASS and zero FAIL, SKIP, BLOCKED or ERROR after the line-neutral cutover.
- `make -j2 test-cuda` passed the native SM121 build and CUDA graph, materialization, operation,
  parity, qtype, sampling and tensor owners. F32, F16, BF16, I32, Q8_0, Q2_K, IQ2_XXS and MXFP4
  qtype comparisons each reported zero maximum absolute and relative difference.
- The complete DeepSeek MoE live target executed 43 layers, three hash-router layers, 40 learned
  router layers, 258 routed experts, 43 shared experts and 774 expert subviews. It retained output
  digest `5c0285ad92e2cbba7a30cb2507f30ea7095ed3efcf3775c25862dab0685ab8f1`
  and routing digest `6312cc6f35b667b69dbea2a7ebff1867d977c229169f1d64632dcd5adf522922`.
- The live artifact remained
  `b9825a070028a66af28cdb25614f7a86c6ad1ec396eed6ae961039db1507ce0e`; runtime
  binding v14 remained
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Raw benchmark and profiler artifacts remain untracked operator evidence.

## Remaining limitations

- This repair does not demonstrate the 20--24 token/s single-user target and does not change the
  retained approximately 9.584 token/s baseline.
- Production Tensor Core coverage remains zero for the accepted width-one path.
- Q8 activation staging, bucket tiling and four-warp grouped execution remain rejected hypotheses,
  not dormant fallback paths.
- The existing qtype plus routed-MoE lower bound remains approximately 42.04 ms/useful token before
  other attention, state and publication work. A compiler-sealed matrix/tile physical regime or
  another real-width source remains the next causal GB10 owner.
- This is not GB10 milestone closure, release qualification, or a public benchmark.

## Why it matters

The repair makes memory admission and warp participation describe the real execution contract:
every consumer fits before mutation, and hardware width can no longer become an out-of-range
semantic read. Performance work can now continue from a stable exact baseline instead of being
confounded by capacity refusals or layout-dependent tail behavior.
