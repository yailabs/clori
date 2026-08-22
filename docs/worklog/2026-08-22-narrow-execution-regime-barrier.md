# Narrow Execution Regime Barrier

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `a3d8ad0bcf7535b3ac9f1a6d7165eef727f0a9d8` |
| Checkpoint | `a3d8ad0bcf7535b3ac9f1a6d7165eef727f0a9d8` |
| Subsystem | compiler-admitted CUDA qtype and grouped MoE execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | complete-request profiler attribution; live behavior; rejected performance experiments; external-reference comparison |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The accepted Physical Execution IR v4 / runtime binding v14 path produced a 9.584 token/s median
over the retained ten-sample width-one target-only fixture. The matching complete-request profile
contained 50,992 CUDA launches and 2,240.066 ms of device time. It produced the exact bounded
15-token output `1 2 3 4 5 6 7 8`.

Recent exact microkernel and state-lifetime repairs had removed material local taxes, but production
Tensor Core coverage remained zero. Generic qtype matvec and grouped MoE were the two largest
remaining execution classes. The next question was whether either could still yield a material
gain through an equivalent narrow-row backend implementation, or whether the compiled physical
regime itself had reached its useful floor.

## Problem

The complete profile named large symbols but did not establish which semantic operations and
execution geometries owned them. Optimizing either symbol generically would have repeated a class
of already falsified experiments: operation roles, qtypes and row populations have different
economics even when they share one backend entrypoint.

The 20--24 token/s single-session objective permits roughly 42--50 ms of complete wall time per
generated token. Before another kernel change, YVEX needed to determine how much of that budget was
already consumed by exact row-dot work alone.

## Causal analysis

Every `yvex_qtype_matvec` call in the retained profile was attributed through its stable
stage/geometry sequence. No qtype device time remained unattributed:

| Semantic operation | Device time | Share of qtype time | Share of CUDA time |
| --- | ---: | ---: | ---: |
| attention output-B | 138.403 ms | 32.91% | 6.18% |
| output head | 102.308 ms | 24.33% | 4.57% |
| router/gate | 48.490 ms | 11.53% | 2.16% |
| attention q-a | 37.154 ms | 8.83% | 1.66% |
| attention mHC | 25.906 ms | 6.16% | 1.16% |
| MoE mHC | 25.314 ms | 6.02% | 1.13% |
| attention index weights | 22.864 ms | 5.44% | 1.02% |
| attention KV projection | 20.135 ms | 4.79% | 0.90% |
| **qtype total** | **420.574 ms** | **100%** | **18.78%** |

Grouped MoE remained a separate owner:

| Grouped MoE operation | Device time |
| --- | ---: |
| routed up | 330.791 ms |
| routed down | 195.421 ms |
| shared up | 88.686 ms |
| shared down | 41.612 ms |
| **grouped MoE total** | **656.510 ms** |

The decode portion assigned 311.567 ms of grouped MoE up/down work across 15 useful output tokens,
or 20.771 ms/token. Decode qtype operations including the output head accounted for approximately
319.11 ms, or 21.27 ms/token. These two non-overlapping classes therefore consumed approximately
42.04 ms/useful token before attention auxiliaries, rolling state, launch/dispatch, publication and
other model work. This is a derived lower bound, not a complete decode-only device-time measure.

The row geometry explains the floor. Width-one routed experts read roughly 1.116 GB/token for
gate/up weights and 0.710 GB/token for down weights, delivering derived effective bandwidth near
113--119 GB/s. The output head reads approximately 1.059 GB per call and reaches about 166 GB/s.
These operations are doing material encoded-weight work; eliminating host gaps or launch barriers
cannot make their combined cost disappear.

Three new hypotheses were tested and rejected without retaining source changes:

1. A direct grouped-up producer generated the exact Q8 intermediate without publishing the F32
   matrix first. One block then serialized 32 output waves and the bounded live run fell to 6.478
   token/s. The run is characterization only, but the large regression disproved that topology.
2. The existing BF16 output-head matrix path was admitted at width one. Five warm exact-output
   samples measured a median of 8.823 token/s, approximately 7.94% below the retained 9.584
   token/s baseline. Packing, synchronization and GEMM setup were not amortized at real width one.
3. Grouped MoE blocks were widened from eight to sixteen warps. Five warm exact-output samples
   measured a 9.320 token/s median, approximately 2.75% below the retained baseline. Reduced block
   concurrency outweighed the additional activation reuse.

The current DS4 MMQ implementation at commit
`84cc882352757baf628a1776badf7cc54d584e28` was inspected as a DeepSeek-specific reference. Its
expert-major assignment, gathered Q8 activation lifetime and matrix-oriented quantized execution
were classified as:

- **ADOPT:** build execution from real expert-major assignments and retain a compatible quantized
  activation across the expert operation;
- **ADAPT:** express a matrix/tile expert regime through YVEX compiler planning, Physical IR,
  identity-bound binding and canonical worklists, with runtime populations supplied above CUDA;
- **REJECT:** CUDA-local semantic regrouping, model-specific runtime policy, fake row width or an
  unsealed alternate physical planner.

## Decision

Stop narrow-row microkernel tuning at this characterization barrier. The next admissible owner is
a compiler-sealed matrix/tile quantized execution regime that can perform more useful row-dot work
per physical GPU unit. Whether that regime requires an identity-bound derived expert layout remains
an evidence question; artifact re-materialization is not assumed in advance.

The regime must preserve the existing worklist authority. Compiler and Physical IR own compatible
grouping, layout, execution width, workspace lifetime and publication. Runtime instantiates actual
populations. CUDA may select and execute only an admitted equivalent tile implementation.

## Implementation

No production implementation was retained at this checkpoint. Each experimental change was built,
executed against the complete model, rejected on causal evidence and removed. Source HEAD therefore
remained `a3d8ad0bcf7535b3ac9f1a6d7165eef727f0a9d8`, with Physical Execution IR v4, binding v14 and
the accepted exact width-one path unchanged.

The retained result is a complete semantic attribution, a falsified set of narrow-regime
hypotheses and a bounded architectural decision for the next GB10 implementation pass. No dormant
fast path or second backend policy authority was left in the repository.

## After

YVEX now has zero-unattributed qtype evidence and a causal lower bound for the existing row-dot
operation mix. Grouped MoE and qtype together accounted for 1,077.084 ms, or 48.08% of complete
profiled CUDA time. Their decode work alone consumed approximately 42.04 ms/useful token.

The accepted runtime remains at the prior 9.584 token/s median and exact output because this
checkpoint intentionally changes no production behavior. It also establishes that another
equivalent width-one CUDA shape selector is not a credible route to the 20--24 token/s objective:
the next experiment must change compiled physical execution granularity.

## Quantitative delta

| Fact | Accepted regime | Experimental result | Decision | Evidence class |
| --- | ---: | ---: | --- | --- |
| Target-only warm median | 9.584 token/s | unchanged | retain accepted path | prior directly comparable checkpoint |
| Complete-request CUDA time | 2,240.066 ms | unchanged | retain accepted path | measured profile |
| Qtype attribution | 420.574 ms | zero unattributed | causal map retained | measured/derived |
| Grouped MoE attribution | 656.510 ms | four semantic owners | causal map retained | measured |
| Decode qtype + routed MoE floor | not previously combined | approximately 42.04 ms/useful token | structural barrier | derived |
| Direct Q8 grouped-up producer | 9.584 token/s retained median | 6.478 token/s single run | reject | characterization only |
| BF16 output head at width one | 9.584 token/s retained median | 8.823 token/s median | reject | approximately comparable, 5 warm samples |
| Sixteen-warp grouped MoE | 9.584 token/s retained median | 9.320 token/s median | reject | approximately comparable, 5 warm samples |
| Physical IR / binding | v4 / v14 | v4 / v14 | unchanged | identity evidence |
| Production Tensor Core coverage | 0% | 0% | unchanged | selected production profile |

## Evidence

- Documentation-checkpoint QA evidence
  `300dd06669165a3458b107591205725dcfe7becb28984fce8d61b882aabc5c71`
  resolved the 14-test documentation-only plan with 14 PASS and zero FAIL, SKIP, BLOCKED or
  ERROR. It covered documentation ownership, project control, architecture, layout, topology,
  source ownership and no-NVCC refusal.
- The retained complete-request Nsight Systems SQLite export has digest
  `2282a9bb49f7774ec643f3abcd01093649684201c255340278b602d7a100fa0e`.
  Its deterministic stage/geometry map has digest
  `90f069906b48a4e4a46df6adb0be3b4700d98bf7da79b0c16f11bc8379f14d6a`.
- Canonical source-checkpoint QA evidence
  `865d8265a0f83c374124b61fe15eb71e122e6739764fc11a42bc3c8d11f80854`
  resolved 101 tests with 101 PASS and zero FAIL, SKIP, BLOCKED or ERROR. It includes native CUDA,
  no-NVCC refusal, complete target-only and DSpark generation, sanitizers, numeric owners,
  transactional behavior and structural guards.
- Every retained or rejected live sample used the same artifact variant
  `b9825a070028a66af28cdb25614f7a86c6ad1ec396eed6ae961039db1507ce0e`,
  runtime binding
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`,
  context 4096 and bounded target-only output. The direct-producer response digest is
  `8d6cf37f5591d0294c5b58aa3a77f881e44c75cb2c344124b39c6e3a0d1e4293`.
- The rejected experimental source was removed after each run. Repository source and selected
  runtime policy therefore remain at the previously qualified checkpoint.
- The standalone legacy MoE live target currently reports a reusable-workspace capacity refusal
  before expert execution. The complete production server fixture succeeds, so this is retained as
  an evidence gap rather than attributed to a reverted experiment.
- Raw profiles, response JSON, generated benchmark material, complete artifacts and bindings remain
  untracked external operator evidence.

## Remaining limitations

- This checkpoint implements no matrix/tile expert primitive, new derived layout, production
  Tensor Core cutover or wider single-user execution source.
- The 42.04 ms/token value is a derived qtype-plus-routed-MoE lower bound, not the complete current
  decode-only device floor. It cannot be promoted to a throughput benchmark.
- The current row-major encoded layout has not yet been proven sufficient or insufficient for the
  next matrix-oriented regime. A derived layout must pass compiler identity, numeric and
  complete-model evidence if it becomes necessary.
- The standalone MoE live workspace refusal requires reconciliation before it can serve as
  qualification evidence for the next expert-execution change.
- Production Tensor Core coverage remains zero. The 20--24 token/s single-session goal, deep
  context, persistence, complete continuous batching, packaging, model-quality evaluation and
  release qualification remain open. GB10 remains active.

## Why it matters

The checkpoint replaces symbol-level intuition with a quantified architectural boundary: the two
largest exact row-dot classes already consume the complete target latency budget before the rest of
the model runs, so the next useful acceleration must widen or reorganize compiled physical work.

## Communication projections

### Short update

YVEX attributed every remaining qtype matvec and found that decode qtype plus routed MoE already
consume approximately 42.04 ms per useful token. Three narrower CUDA alternatives regressed, so
the accepted path remains unchanged and the next owner is a compiler-sealed matrix/tile regime.

### Longer post seed

1. Every qtype invocation was mapped to its semantic operation and geometry.
2. Grouped MoE and qtype together own 48.08% of complete profiled CUDA time.
3. Their decode work alone uses approximately the complete 20--24 token/s latency budget.
4. Direct Q8 production, width-one BF16 GEMM and wider narrow-row blocks all regressed.
5. The next experiment must change Physical IR execution granularity, not add another CUDA shape
   heuristic.

### Article seed

**Title:** Knowing When a Faster Kernel Regime Has Reached Its Floor

**Thesis:** Complete semantic attribution and rejected end-to-end experiments can prove that the
next performance owner is compiled execution granularity rather than another local microkernel.

Suggested sections:

1. From generic qtype symbol to semantic operation ledger.
2. The 42.04 ms/token row-dot lower bound.
3. Why three plausible narrow-regime alternatives regressed.
4. DS4's matrix-execution invariant through YVEX ownership boundaries.
5. The compiler-sealed matrix/tile regime required next.

Strongest evidence: qtype and routed MoE alone consume approximately 42.04 ms/useful token before
the remaining transformer execution.

### Visual candidates

- qtype semantic-operation share table;
- complete CUDA time split between qtype, grouped MoE and other execution;
- 42--50 ms target budget versus the 42.04 ms row-dot lower bound;
- rejected narrow-regime experiments and the next authority boundary.

### Quoteable technical facts

- "Every qtype matvec in the retained profile is attributed to a semantic operation."
- "Qtype and routed MoE alone consume approximately 42.04 ms per useful token."
- "Three complete-model experiments were rejected rather than retained as dormant fast paths."
