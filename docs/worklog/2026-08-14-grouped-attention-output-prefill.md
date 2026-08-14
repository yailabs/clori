# Grouped Attention Output Prefill

| Field | Value |
| --- | --- |
| Date | 2026-08-14 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `9008d467ae056f433e1ada04bdab59ce68f06a1c` |
| Checkpoint | `21cf7ed33d46cb4879c4b78f36978fad9d8c8279` |
| Subsystem | generic CUDA qtype and attention projection |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The preceding checkpoint grouped the eight output-A projections for single-row
decode, but multi-row output-A prefill still called the generic encoded qtype
matvec once per group. The fixed complete-model request therefore issued 344
output-A prefill launches across 43 layers and spent 69.705 ms in that operation.

The generic qtype symbol accounted for 47.214% of profiled device time, but the
symbol name did not establish its economic owner. Routed and shared expert math
already used distinct grouped MoE kernels, so treating the entire qtype symbol
as one optimization target would have obscured materially different operations.

## Problem

The profiler exposed a large generic qtype bucket without enough phase ownership
to select the next kernel honestly. After attribution, output-A was the largest
qtype operation family at 16.453% of total device time: 6.994% in grouped decode
and 9.459% in prefill. Prefill retained a host-side group launch loop even though
the compiled operation already described one group-major projection over all
input rows.

## Causal analysis

Attribution assigned every generic qtype launch to an admitted operation. In
addition to output-A, output-B accounted for 8.708% of total device time,
attention q_b for 8.681%, CSA main rolling projection for 3.740%, output head for
3.645%, and the remaining named qtype operations for 5.987%. Grouped MoE up and
down execution accounted separately for 15.591% and 8.852%; they were not hidden
inside `yvex_qtype_matvec`.

Current DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` treats multi-row
DeepSeek output-A as one grouped semantic operation. That supports adopting the
operation boundary, not copying DS4's complete implementation. DS4's Q8
activation and F16-derived layout remain incompatible with the current YVEX
binding: the prior Q8 candidate changed whole-model output and no equivalent F16
derived layout is admitted by Physical Execution IR v3 and binding v13.

## Decision

Adopt one grouped multi-row output-A operation, adapt it to YVEX's exact
F32/MXFP4 physical contract, and reject the unadmitted Q8/F16 physical
alternative.

The generic CUDA backend owns a row-batch kernel spanning both groups and input
rows. It reuses the ordinary matvec's token tiling and canonical encoded row dot.
The compiler-selected qtype, activation representation, output representation,
and forensic alternative remain authoritative.

## Implementation

Checkpoint `21cf7ed33d46cb4879c4b78f36978fad9d8c8279` generalizes the exact
grouped decode kernel into a grouped row-batch kernel. Token-major activation
strides and group-major weight rows are explicit launch operands, so the same
backend operation handles decode and bounded prefill without inferring model
topology.

Capability admission requires the new native symbol. Non-forensic grouped
projection uses it when the ordinary qtype geometry does not require the
separate block-row regime; existing Tensor Core, forensic, and block-row paths
retain their prior selection and refusal contracts. No artifact, Physical
Execution IR, runtime binding, protocol, state layout, or public API version
changes.

## After

At the real eight-group, five-input-row scheduling geometry, one grouped launch
is bit-identical to the prior eight-launch loop and remains within the
independent CPU qtype tolerance. The complete-model profile reduced output-A
prefill from 344 launches to 43 and from 69.705 ms to 55.061 ms.

Across the fixed request, the combined qtype family lost 301 launches and 17.415
ms. Total profiled kernel time fell by 17.910 ms. Five short measured runs after
one warmup improved median prefill throughput, but their generation median did
not improve; this checkpoint therefore makes no decode-throughput claim.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Output-A prefill launches | 344 | 43 | directly comparable; 301 fewer |
| Output-A prefill device time | 69.705 ms | 55.061 ms | directly comparable; 21.008% lower |
| Total qtype-family launches | 3,351 | 3,050 | directly comparable; 301 fewer |
| Total qtype-family device time | 398.568 ms | 381.153 ms | directly comparable; 4.369% lower |
| Total kernel launches | 14,225 | 13,924 | directly comparable; 301 fewer |
| Total kernel device time | 735.259 ms | 717.348 ms | directly comparable; 2.436% lower |
| Five-run short prefill median | 7.90 token/s | 8.04 token/s | directly comparable; 1.77% higher |
| Five-run short prefill range | 7.84–7.91 token/s | 7.99–8.05 token/s | directly comparable |
| Five-run short generation median | 8.35 token/s | 8.11 token/s | directly comparable; 2.87% lower |
| Five-run short generation range | 8.32–8.44 token/s | 8.02–8.13 token/s | directly comparable |

## Evidence

- The baseline and checkpoint Nsight report digests are
  `60e19224c0baa0a046a1b5bbb1a3866edfe53ddc8c46130eee01bd03f0ee9866`
  and `2b8cf61ae925067eb7489cf615015e8b4b5acfd4753664e7f395338c4425d2e1`.
- The selected complete artifact identity is
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
- Every measured completion retained output digest
  `0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.
- The native kernel CUBIN digest is
  `45d16d1d76b505481d7a7ecdde7bba460fdf381405d470b6b38e1f8b2dc3c7b9`;
  `cuobjdump` exposes real SM121 SASS for `yvex_qtype_grouped_rows`.
- The focused CUDA oracle uses eight groups, five input rows, and MXFP4. It
  requires one grouped launch, compares every result with an independent CPU
  qtype dot, requires raw bit identity with the ordinary per-group CUDA loop,
  and verifies scoped completion and resource cleanup.
- `make -j2 check-cuda` passes twice, including the 43-layer, 6,772,096-value
  attention oracle and byte-identical repeat. Native SM121 admission,
  no-NVCC fail-closed behavior, runtime and quant ASan/LeakSanitizer, runtime and
  quant UBSan, the full repository check, ownership, layout, architecture,
  documentation, and project-control lanes pass.
- Raw profiler reports, benchmark receipts, and the complete model remain
  external identity-bound evidence and are not tracked in Git.

## Remaining limitations

- The complete-model target-only decode hard gate remains open. The measured
  short generation median regressed in this lane, so the component improvement
  is not promoted to an end-to-end decode improvement.
- Output-B, q_b, CSA projection, output head, MoE, and other qtype owners retain
  their distinct measured costs and require separate causal decisions.
- The current physical binding continues to require exact F32 activation for
  this operation; Q8 is a tested primitive, not an admitted substitution.
- Cold model admission remains materially slower than the DS4 reference and
  retains significant residency/transient-memory debt. This kernel checkpoint
  does not change model loading.
- The complete competitive SM121 stack, deep-context qualification, durable
  restore, continuous batching, favorable DSpark performance, canonical
  packaging, and deployment closure remain open GB10 gates.
- Topology planning, SSD streaming, multiple devices, and model-generation hot
  reload remain downstream architecture work.

## Why it matters

YVEX now removes the remaining per-group launch loop from the exact output-A
prefill path while keeping physical numeric policy compiler-owned. The phase
attribution also prevents unrelated MoE and dense projection costs from being
mistaken for one generic matvec problem.

## Communication projections

### Short update

YVEX attributed the generic qtype hotspot before changing it: MoE already used
separate kernels, while output-A was the largest qtype family. Extending its
exact grouped CUDA operation to prefill reduced that phase from 344 launches to
43 and cut component device time by 21.008%, with bit-identical CUDA output.

### Longer post seed

1. A generic profiler symbol accounted for 47.214% of device time but combined
   several economically different operations.
2. Attribution separated MoE kernels from attention and output projection, then
   identified output-A prefill as the largest remaining launch-decomposed case.
3. DS4 supported grouping the semantic operation; YVEX retained its own
   compiler-selected F32/MXFP4 physical contract.
4. One row-batch grid replaced eight per-group launches without changing any
   output bits.
5. The component result is real, but end-to-end decode did not improve, so the
   next owner still requires new causal evidence.

### Article seed

**Possible title:** A Generic Kernel Name Is Not a Performance Owner

**Thesis:** Phase attribution must separate semantic operations before a CUDA
hotspot can be optimized without changing the wrong contract.

Suggested sections:

1. Why `qtype_matvec` obscured attention, output, and state economics.
2. Separating grouped MoE from generic qtype attribution.
3. Using DS4 to adopt an invariant without copying its physical policy.
4. Preserving F32/MXFP4 bit identity in a grouped row-batch grid.
5. Reading component wins alongside a non-improving end-to-end median.

Strongest evidence: the operation-attribution table, the 344-to-43 launch
change, the 21.008% component-time reduction, and raw CUDA bit identity.

### Visual candidates

- A phase ownership chart splitting generic qtype time from distinct MoE
  kernels.
- Eight per-group prefill launches versus one group/input-row CUDA grid.
- A component-versus-end-to-end table showing why the result is not promoted
  to a decode-throughput claim.

### Quoteable technical facts

- “Output-A prefill fell from 344 launches to 43 while retaining bit-identical
  CUDA output.”
- “The directly comparable output-A prefill component time fell by 21.008%.”
- “The measured short generation median did not improve, so this checkpoint
  makes no decode-throughput claim.”
