# MXFP4 Q8 Activation Reuse

| Field | Value |
| --- | --- |
| Date | 2026-08-15 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `9c729ada5827ca4589f718801bf4adb4f9c415e5` |
| Checkpoint | `406873c63d735472760261b96552a51fd92a238f` |
| Subsystem | generic CUDA qtype attention projection |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

Complete operation attribution assigned all 2,835 generic qtype matvec launches
and 291.891 ms of device time to semantic operations. There was no unattributed
qtype time. Attention q_b and output-B were the two largest remaining qtype
classes, each accounting for about 65 ms under the fixed target-only request.

The q_b projection used MXFP4 weights with a 1,024-element activation and one
or five input rows. Every output-row warp independently read the same admitted
Q8 activation from global memory. The arithmetic was already correct, but its
physical execution discarded reuse available within one row batch.

## Problem

The generic matvec symbol combined materially different geometries. Optimizing
all of them together risked changing occupancy or numerical policy for output-B,
the output head, attention state projections, and other consumers that did not
share q_b's physical economics.

The aggregate qtype percentage therefore did not justify a generic fast path.
The selected repair first had to prove that one exact semantic/physical class
owned a reusable cost and that isolating its resources improved the complete
profile.

## Causal analysis

The q_b class accounted for 215 launches and 65.537 ms: 172 target-decode
launches took 30.793 ms and 43 five-row prefill launches took 34.744 ms. The
MXFP4 row dot itself was not approximate; repeated global reads of one already
quantized activation were the avoidable work.

Current DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` reuses a
quantized activation across row work and specializes execution by matrix
geometry. YVEX adopted that reuse invariant, adapted it to its compiler-admitted
F32-to-Q8 activation and exact MXFP4 row dot, and rejected DS4's physical
representation as a drop-in artifact policy.

Two broader alternatives were measured and rejected. Applying the same row
kernel to output-B increased its single-row cost by about 10% and its five-row
cost by about 28%. Folding shared storage into `yvex_qtype_matvec` allocated the
resource to every qtype geometry and raised total profiled device time from
747.926 ms to 882.415 ms. Those results establish that this is a distinct
physical regime, not a generic matvec default.

## Decision

Add one backend-owned exact MXFP4 narrow-row kernel for admitted K=1024,
one-to-eight-row batches with at least 4,096 output rows. One block stages the
Q8 activation once and assigns eight independent output rows to eight warps.

The canonical `q8_warp_dot` remains the arithmetic owner. Physical IR/binding
qtype and activation policy remain unchanged. The ordinary qtype matvec remains
the fallback and independent CUDA comparison path. Output-B and other shapes do
not enter this regime.

## Implementation

Checkpoint `406873c63d735472760261b96552a51fd92a238f` adds the native
SM121 kernel to the admitted CUDA capability bundle and selects it in the
generic attention operation from explicit qtype, row width, row count, and
input-row geometry.

The kernel uses dynamic shared memory only for the selected invocation, so its
resource cost cannot reduce occupancy for unrelated qtype operations. Checked
arithmetic bounds the grid and shared extent. No family name, model geometry,
artifact layout, Physical Execution IR, runtime binding, protocol, or public API
changed.

The focused test constructs a real 4,096-by-1,024 MXFP4 matrix with five input
rows. It requires the production selector, compares every result with an
independent CPU codec oracle, requires raw bit identity with the ordinary CUDA
row-dot path, and verifies cleanup.

## After

The q_b class fell from 65.537 ms to 61.215 ms, a 6.595% reduction with the same
215 launches. Its target-decode component fell 6.184% and its five-row prefill
component fell 6.959%.

Across the fixed request, combined ordinary and specialized qtype execution
fell from 291.891 ms to 282.449 ms. Total profiled CUDA device time fell from
747.926 ms to 724.837 ms while the total launch count remained 13,924. The
generated text and output digest were unchanged.

The single profiled generation rate was effectively flat, so this checkpoint
does not claim an end-to-end decode-throughput improvement. The result is a
component and complete-device-time improvement under the profiling fixture.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| q_b launches | 215 | 215 | directly comparable; unchanged |
| q_b device time | 65.537 ms | 61.215 ms | directly comparable; 6.595% lower |
| q_b target-decode time | 30.793 ms | 28.889 ms | directly comparable; 6.184% lower |
| q_b five-row prefill time | 34.744 ms | 32.326 ms | directly comparable; 6.959% lower |
| Combined qtype launches | 2,835 | 2,835 | directly comparable; unchanged |
| Combined qtype device time | 291.891 ms | 282.449 ms | directly comparable; 3.235% lower |
| Total kernel launches | 13,924 | 13,924 | directly comparable; unchanged |
| Total CUDA device time | 747.926 ms | 724.837 ms | directly comparable; 3.087% lower |
| Profiled prefill | 7.69 token/s | 8.00 token/s | directly comparable; single run |
| Profiled generation | 7.83 token/s | 7.80 token/s | directly comparable; single run |

## Evidence

- Baseline and checkpoint Nsight report digests are
  `9ace4cd6ddb40bbba7ddfdd6e5aee9b49f1556c00ee0b788281fdbce9ce04766`
  and `e4c877bd2bb53f41135c31a766046827c3daeb00f777de6ac1cac39354a63a95`.
- Their exported SQLite digests are
  `30b21a0171649fc6d63cce125b47ce63aa0254376f93dfd5c4de5589c38e8` and
  `b42a47ac1bf1281f6c75299d34bd0fb799b0bde18c9518504d1e511156048817`.
- Artifact identity
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
  and binding identity
  `3dfdd0d2a9578a755495673740397d043e079171be4ffb8f2c064831d201d250`
  are unchanged.
- Both runs produced `你好！有什么可以帮助` with digest
  `0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.
- Native kernel CUBIN digest
  `41cfb96afe5ec74610ae31517d30c3195ea25d09c52532e54ad28bda0f79a6ea`
  exposes SM121 SASS for `yvex_mxfp4_q8_rows`.
- The focused CUDA test passes with native-SM121 admission. The complete
  `make -j2 check-cuda` lane passes, including the 43-layer attention oracle and
  byte-identical repeat. No-NVCC fail-closed, quant and runtime
  ASan/LeakSanitizer, quant and runtime UBSan, source ownership, repository
  layout, and architecture guards pass.
- Raw profiles and the complete model remain external identity-bound evidence;
  they are not tracked in Git.

## Remaining limitations

- The measured short generation result was flat. This checkpoint does not
  close the GB10 decode-throughput gate.
- Output-A grouped rows remain the largest named CUDA kernel in the selected
  profile. Output-B remains a separate roughly 66 ms qtype owner whose broader
  shared-row experiment regressed and was rejected.
- The selected kernel is exact DP4A-era qtype execution, not a Tensor Core
  claim. Competitive SM121 execution still requires further causal work.
- Full deep-context qualification, durable restore, continuous batching,
  favorable DSpark performance, canonical packaging, and final deployment
  remain open GB10 gates.
- Topology planning, SSD streaming, multiple devices, and model-generation hot
  reload remain downstream architecture work.

## Why it matters

YVEX now reuses q_b's quantized activation without imposing that resource cost
on unrelated matvec shapes. The rejected generic implementation demonstrates
why semantic attribution must lead to a physical execution regime rather than a
blanket kernel modification.

## Communication projections

### Short update

YVEX attributed every qtype matvec before optimizing it, then isolated the
K=1024 MXFP4 q_b regime. Reusing one exact Q8 activation across eight output-row
warps cut q_b device time by 6.595% and total profiled CUDA time by 3.087%, with
the same launches and identical generated output.

### Longer post seed

1. A 39% generic qtype bucket was decomposed to zero unattributed device time.
2. q_b and output-B had similar aggregate cost but different geometry.
3. A broad output-B fast path regressed, so only K=1024 survived the funnel.
4. One block now stages the admitted Q8 activation for eight exact MXFP4 row dots.
5. Putting shared storage in the generic kernel regressed unrelated shapes,
   proving the resource belongs to a distinct physical regime.
6. Component and CUDA device time improved, while end-to-end decode remained
   flat and is not promoted.

### Article seed

**Possible title:** When a Generic CUDA Hotspot Needs a Physical Regime

**Thesis:** Complete operation attribution and rejected broad experiments are
what distinguish a reusable optimization from a resource regression hidden by
one generic symbol.

Suggested sections:

1. Decomposing `qtype_matvec` to semantic operation and execution regime.
2. Separating q_b frequency from per-call cost.
3. DS4's reuse invariant versus YVEX's admitted physical representation.
4. Why output-B and the generic shared-memory path were rejected.
5. Exact activation reuse and its component/device-time evidence.
6. Why flat decode throughput remains an explicit non-claim.

Strongest evidence: zero unattributed qtype time, the 6.595% q_b reduction, the
3.087% total CUDA-time reduction, and bit-identical output.

### Visual candidates

- An operation/regime table for all attributed qtype matvec calls.
- One Q8 activation read per output warp versus one shared activation per eight
  output-row warps.
- A decision funnel showing the rejected output-B and generic-kernel variants.
- Component time, total CUDA time, and end-to-end token rate in one non-promotion
  table.

### Quoteable technical facts

- “Every profiled qtype matvec was attributed before the selected path changed.”
- “Exact q_b device time fell by 6.595% with no launch-count change.”
- “A generic shared-memory implementation regressed total device time, so the
  resource was isolated to one physical geometry.”
- “Generated output remained bit-identical while profiled CUDA time fell by
  3.087%.”
