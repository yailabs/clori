# Grouped Attention Output Decode

| Field | Value |
| --- | --- |
| Date | 2026-08-14 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `c75c54a046ab85435b0dccc7a817395aafae945e` |
| Checkpoint | `23404bf44c6f66182bd3b90ffda8f8f7ce6ae7cd` |
| Subsystem | generic CUDA qtype and attention projection |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

Every DeepSeek output-A group launched the generic encoded qtype matvec
independently during single-row decode. Eight model-derived groups therefore
required eight launches even though they shared one operation boundary and
used contiguous group-major weights.

On the complete selected artifact, a fixed four-token target-only Nsight
capture reported 4,555 generic qtype matvec instances and 443.851 ms inside
that kernel family. Five repeated EOS completions reached a median 7.05 token/s.

## Problem

The phase roofline showed launch depth and qtype projection time as a material
decode cost. The output-A grouping was already explicit in the compiled
operation, but backend dispatch decomposed it into host-side per-group launches.

## Causal analysis

Current DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` executes grouped
DeepSeek output projection with one grid and a shared encoded-weight operation.
That established the grouping invariant worth adopting. Its activation
representation was not directly admissible in YVEX: the current Physical
Execution IR and binding v13 select F32 activation semantics for this path.

A candidate that quantized the grouped activation to Q8 was stable in focused
CUDA tests but changed complete-model output. The exact F32 candidate retained
the baseline output and reduced both launch count and profiled qtype time. This
isolates launch decomposition as the causal owner without conflating the result
with an unadmitted numerical-policy change.

## Decision

Adopt the DS4 grouping invariant, adapt it to YVEX's compiler-selected F32
activation representation, and reject Q8 activation under the current binding.

The backend owns one native grouped-decode kernel because its group-major launch
geometry and resource contract differ from the ordinary matvec kernel. The
kernel reuses the canonical qtype row-dot implementation and does not infer a
model family, group count, qtype policy, or activation policy.

## Implementation

Checkpoint `23404bf44c6f66182bd3b90ffda8f8f7ce6ae7cd` adds the grouped-decode
symbol to the native/portable CUDA bundle and capability admission. For an
admitted single-row, non-forensic grouped projection, backend dispatch launches
one grid spanning all groups. Each row selects its group's distinct F32
activation slice and performs the existing encoded qtype warp dot, including
the exceptional finite-row recovery and BF16 output policy.

Multi-row, Tensor Core, and forensic paths retain their prior admitted
semantics. No artifact, qtype, Physical Execution IR, runtime binding, wire,
state-layout, protocol, or public API contract changes.

## After

The fixed profiled request replaced 1,376 ordinary matvec launches with 172
grouped launches: eight old launches became one for each of 172 output-A
operations. Total qtype-family launches fell by 1,204 and combined qtype time
fell from 443.851 ms to 398.568 ms.

Five directly comparable EOS completions improved from a median 7.05 to 7.72
token/s while preserving text, stop, usage, and output identity. Twenty
additional short sessions retained one identical output digest without the
resident-memory growth observed in a rejected broader kernel experiment.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Output-A launches per operation | 8 | 1 | directly comparable |
| Generic qtype matvec instances | 4,555 | 3,179 | directly comparable Nsight capture |
| Grouped-decode instances | 0 | 172 | directly comparable Nsight capture |
| Total qtype-family instances | 4,555 | 3,351 | directly comparable; 1,204 fewer |
| Combined qtype-family device time | 443.851 ms | 398.568 ms | directly comparable; 10.20% lower |
| Five-run EOS generation median | 7.05 token/s | 7.72 token/s | directly comparable; 9.50% higher |
| Five-run EOS range | 6.44–7.10 token/s | 7.66–7.80 token/s | directly comparable |
| Five-run EOS coefficient of variation | 3.61% | 0.62% | directly comparable |
| Twenty-session short output stability | not run in the five-run lane | 20/20 identical | characterization only |

## Evidence

- The before and after Nsight report digests are
  `170cd24a1e7f26971ab3f34f358ec0a7c7a395849cbea8b3be26d847a0970061`
  and `60e19224c0baa0a046a1b5bbb1a3866edfe53ddc8c46130eee01bd03f0ee9866`.
- The selected complete artifact identity is
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
- Every short completion retained output digest
  `0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.
- Every EOS completion retained output digest
  `d2f910a57d9fe2b2bd32fdd1979ae86f5e9e41becebe9efff868b13f51c5a409`.
- The native SM121 kernel CUBIN digest is
  `875d06924f21e0a9565649c8fb42a18afce50e02a3a525301aa2cce1c6a4f295`;
  `cuobjdump` exposes real SASS for `yvex_qtype_grouped_decode`.
- The focused CUDA oracle compares every grouped MXFP4 output row with the
  independent CPU qtype dot, requires one launch, and checks scoped completion
  and full resource cleanup.
- `make -j2 check-cuda` passes twice, including the 43-layer, 6,772,096-value
  attention oracle and byte-identical live repeat.
- Native SM121 admission, no-NVCC fail-closed behavior, runtime and quant
  ASan/LeakSanitizer, runtime and quant UBSan, ownership, layout, architecture,
  documentation, and project-control lanes pass.
- Raw profiler reports and full-model receipts remain external identity-bound
  evidence and are not tracked in Git.

## Remaining limitations

- This optimization does not close the GB10 24 token/s target-only decode gate;
  the directly comparable median remains 7.72 token/s.
- It closes one output-A launch owner, not specialized attention computation,
  the complete SM121 execution stack, or all per-layer synchronization debt.
- Q8 activation remains available as a tested primitive but is not admitted for
  this whole-stack physical binding.
- Deep-context qualification, durable state restore, continuous batching,
  favorable DSpark performance, canonical packaging, and deployment closure
  remain open GB10 gates.
- Topology planning, SSD streaming, multiple devices, and model-generation hot
  reload remain downstream architecture work.

## Why it matters

The runtime now preserves exact compiler-selected numerics while eliminating a
host-side model-derived launch loop from a measured hot path. This improves
complete-model decode without adding a DeepSeek branch to generic runtime code.

## Communication projections

### Short update

YVEX now projects all eight DeepSeek output-A groups in one native CUDA launch
while retaining the binding's exact F32 activation contract. A directly
comparable full-model profile removed 1,204 qtype launches, reduced qtype device
time by 10.20%, and improved five-run EOS generation median from 7.05 to 7.72
token/s with identical output.

### Visual candidates

- Eight per-group launches versus one group-major CUDA grid.
- Nsight qtype-family launch and device-time before/after table.
- Compiler-selected F32 activation versus the rejected unadmitted Q8 candidate.

### Quoteable technical facts

- “Eight DeepSeek output-A launches now collapse into one exact grouped CUDA
  operation.”
- “The directly comparable profile removed 1,204 qtype launches and reduced
  qtype-family device time by 10.20%.”
- “The five-run EOS median improved from 7.05 to 7.72 token/s with identical
  output identity.”
