# Grouped MoE Activation Reuse

| Field | Value |
| --- | --- |
| Date | 2026-08-15 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `abd0eb91c905a61dc5fca907643e98c5822e881c` |
| Checkpoint | `4a66ef2c0eb480d8ed79f883a90cf0a2df06589c` |
| Subsystem | generic CUDA grouped MoE execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The exact grouped MoE up and down kernels assigned eight output rows to the
eight warps in each CUDA block. Every warp independently read the same F32
input activation while decoding a different MXFP4 weight row. Across the fixed
complete-model profile, grouped up consumed 177.703 ms and grouped down consumed
106.536 ms in 430 launches each.

The launch topology already grouped routed expert work and kept routing on the
device. The remaining cost was therefore not evidence for another host launch
loop or token-local expert execution.

## Problem

The grouped operation reused expert scheduling but not its pair-local
activation. Eight warps in one block repeatedly fetched identical activation
elements while weight rows, output rows, qtype decoding, numerical order, and
the operation count remained independent.

Grouped up and down together accounted for 31.717% of the baseline profiled
device time. Rewriting the complete MoE path before separating activation,
weight-decode, routing, launch, and synchronization costs would have risked
changing already-correct mechanics without targeting the measured cause.

## Causal analysis

Shape attribution showed that the largest gains were available on routed
expert shapes: grouped up at grid 7,680 and grouped down at grid 15,360. Small
shared-expert shapes were already near their floor. The launch counts, host and
device transfer counts, synchronization counts, routing representation, and
active weight bytes did not change between profiles.

The prior kernels logically requested about 2,818,572,288 activation bytes
across the measured grouped operations. Pair-local staging reduces that request
to about 352,321,536 bytes, an estimated 87.5% reduction in repeated activation
loads. This is a layout-derived traffic estimate, not a measured DRAM counter.
Hardware occupancy and DRAM-throughput counters were unavailable because the
machine denied Nsight Compute access with `ERR_NVGPUCTRPERM`.

Current DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` exploits a
broader DeepSeek-specific D2R regime: expert-major worklists, shared activation
tiles, Q8 activation, fused gate/up and route, and Tensor Core MMA. YVEX already
adopts expert-major device grouping. It adapts the shared-activation invariant
to the admitted exact F32/MXFP4 operation. Copying DS4's Q8 intermediate and
MMA regime is rejected for this checkpoint because Physical Execution IR v3
and binding v13 retain F32 after whole-model Q8 output divergence, and the
current sparse row population does not admit the existing Tensor Core
crossover.

## Decision

Keep the compiler-selected F32/MXFP4 physical contract and the existing grouped
MoE operation. Make each non-Tensor-Core CUDA block belong to one ordered
expert-row pair, cooperatively stage that pair's exact F32 activation once in
dynamic shared memory, and let its output-row warps reuse the staged values.

Q8 and admitted Tensor Core alternatives retain their existing launch geometry
and receive no dynamic shared allocation. No model-family or common-runtime
special case is introduced.

## Implementation

Checkpoint `4a66ef2c0eb480d8ed79f883a90cf0a2df06589c` changes the generic grouped-up
and grouped-down row kernels and their backend launch geometry. Row blocks now
remain pair-local, including partial groups when an output width is not
divisible by eight. Every thread participates in staging and synchronization
before inactive tail warps return.

The host boundary validates the pair-local grid and dynamic shared extent with
checked arithmetic. Routed up stages 16 KiB for the 4,096-value activation;
routed down stages 8 KiB for the 2,048-value intermediate. The canonical weight
decoder, FMA order, routing order, output publication, Tensor Core selection,
and failure contract are unchanged.

## After

Grouped-up device time fell to 117.329 ms and grouped-down device time fell to
67.829 ms. Their combined time fell by 99.081 ms while retaining exactly 430
launches for each operation. Total profiled kernel time fell from 896.246 ms to
747.926 ms.

The fixed complete-model request retained the same committed text and digest.
Under the directly comparable Nsight Systems lane, prefill rose from 6.68 to
7.69 token/s and generation rose from 6.97 to 7.83 token/s. Separate short
unprofiled windows were disturbed by concurrent machine workloads and are kept
as characterization rather than promoted end-to-end evidence.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Grouped-up device time | 177.703 ms | 117.329 ms | directly comparable; 33.974% lower |
| Grouped-down device time | 106.536 ms | 67.829 ms | directly comparable; 36.333% lower |
| Combined grouped MoE time | 284.239 ms | 185.158 ms | directly comparable; 34.858% lower |
| Grouped-up launches | 430 | 430 | directly comparable; unchanged |
| Grouped-down launches | 430 | 430 | directly comparable; unchanged |
| Total kernel launches | 13,924 | 13,924 | directly comparable; unchanged |
| Total kernel device time | 896.246 ms | 747.926 ms | directly comparable; 16.549% lower |
| Logical activation-read request | 2,818,572,288 bytes | 352,321,536 bytes | derived estimate; 87.5% lower |
| Profiled prefill | 6.68 token/s | 7.69 token/s | directly comparable profiler lane; 15.120% higher |
| Profiled generation | 6.97 token/s | 7.83 token/s | directly comparable profiler lane; 12.339% higher |

## Evidence

- The baseline and checkpoint Nsight report digests are
  `ae9ef8bf1e53e9018913fec517de341d607d9353b480bb4ee5f90918c06419c1`
  and `9ace4cd6ddb40bbba7ddfdd6e5aee9b49f1556c00ee0b788281fdbce9ce04766`.
- Their exported SQLite digests are
  `5f0f7a2e0b534cc735df0c1a2747f8efb02bd33b3ea1ca7b7d843312ec59d843`
  and `30b21a0171641629fc6d63cce125b47ce63aa0254376f93dfd5c4de5589c38e8`.
- The selected artifact identity is
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
- Every measured completion retained output text `你好！有什么可以帮助` and digest
  `0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.
- H2D remained 1,621 copies and 198,692,428 bytes; D2H remained 670 copies and
  61,938,608 bytes; D2D remained 4,796 copies and 1,630,240,644 bytes.
- Event synchronization remained 10 calls, stream wait remained 269 calls, and
  context synchronization remained 306 calls.
- `make -j2 check-cuda`, no-NVCC fail-closed validation, exact qtype and CUDA
  MoE tests, the complete ASan/LeakSanitizer and UBSan runtime lane, ownership,
  layout, architecture, source-manifest, and diff guards pass.
- Raw profiler reports, benchmark receipts, and the complete model remain
  external identity-bound evidence and are not tracked in Git.

## Remaining limitations

- This checkpoint does not close the GB10 decode, prefill, context, serving, or
  packaging gates.
- The result removes repeated exact activation loads; it does not reduce MoE
  launch count, routing work, active weight bytes, or synchronization.
- `yvex_qtype_matvec` remains the largest measured kernel family at 291.891 ms,
  or about 39.03% of checkpoint device time, and still requires operation-level
  attribution before another generic change.
- Grouped up and down retain MXFP4 scalar decode/FMA paths. This checkpoint does
  not claim Tensor Core execution for those row kernels.
- Hardware occupancy, achieved DRAM bandwidth, and Tensor Core utilization
  remain an evidence gap until GPU performance-counter access is admitted.
- Deep-context qualification, durable restore, continuous batching, favorable
  DSpark performance, canonical packaging, and deployment closure remain open.
- Topology planning, SSD streaming, multiple devices, and model-generation hot
  reload remain downstream architecture work.

## Why it matters

The grouped MoE owner now reuses the data its execution topology already proves
to be shared, cutting the two expert kernels by about one third without changing
the compiler-selected numerical representation or adding a DeepSeek runtime
branch.

## Communication projections

### Short update

YVEX decomposed grouped MoE before rewriting it: routing, launches, transfers,
and synchronization were stable, while eight output-row warps repeatedly read
the same F32 activation. Pair-local shared staging cut grouped-up time by 34.0%
and grouped-down by 36.3%, preserving the exact F32/MXFP4 result and all launch
counts.

### Longer post seed

1. Grouped MoE was already device-routed, so its aggregate profile percentage
   did not justify another scheduling rewrite.
2. Shape attribution separated repeated activation loads from weight decode,
   routing, launch, synchronization, and transfer costs.
3. DS4 supplied the shared-tile invariant, while YVEX rejected its unadmitted
   Q8/MMA physical regime for the current binding.
4. One pair-local block now stages the exact activation once for eight output
   rows without changing FMA order or output bits.
5. The next frontier returns to operation-level attribution of the remaining
   generic qtype family.

### Article seed

**Possible title:** Grouping Work Is Not the Same as Reusing Its Data

**Thesis:** A grouped CUDA operation can remain memory-inefficient when its
warps independently fetch an operand that the grouping contract proves is
shared.

Suggested sections:

1. Why aggregate MoE time was not enough to select a rewrite.
2. Separating activation reads from routing, launches, and weight decode.
3. Adapting DS4's shared-tile invariant without adopting Q8 numerics.
4. Pair-local blocks, partial output groups, and synchronization safety.
5. Reading a 34.9% component reduction without overstating GB10 closure.

Strongest evidence: unchanged launch/transfer/synchronization counts, the
derived 87.5% activation-load reduction, exact output digest, and the directly
comparable grouped-up/down timings.

### Visual candidates

- Eight warp-local activation reads versus one pair-local shared staging pass.
- Grouped-up/down before-and-after component timing.
- DS4 ADOPT/ADAPT/REJECT table for the D2R physical regime.

### Quoteable technical facts

- “Grouped-up device time fell by 34.0% without changing its 430 launches.”
- “Grouped-down device time fell by 36.3% while retaining exact F32/MXFP4 output.”
- “The optimization changes activation reuse, not routing, transfer, or synchronization.”
