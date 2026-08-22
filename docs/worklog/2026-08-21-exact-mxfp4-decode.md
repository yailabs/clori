# Exact MXFP4 Decode on the Width-One CUDA Path

| Field | Value |
| --- | --- |
| Date | 2026-08-21 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `0638db6d8f337c7d12a14852f907a63a1dfadd87` |
| Checkpoint | `0dc1ff981791b90db25ef88b51ccda2896be0838` |
| Subsystem | CUDA qtype kernel primitives |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; CUDA qualification; runtime qualification; sanitizer; performance characterization |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The selected width-one production path decoded MXFP4 E2M1 values through a
per-element scalar switch. Generic qtype matrix-vector execution also selected
the qtype inside the inner value decoder, and the exact MXFP4/Q8 DP4A path
constructed four signed E2M1 bytes through four independent branch chains.

On the retained 15-token target-only fixture, warm generation characterized at
an 8.326 token/s median. The directly comparable Nsight Systems baseline
measured 125.372 ms wall time and 103.339 ms of CUDA kernel time per useful
token, with 2,413 launches per token. The selected profile attributed 8.300 ms
per token to the specialized MXFP4/Q8 rows and 12.841 ms per token to grouped
qtype rows.

Physical Execution IR v4 and runtime binding v14 were already authoritative.
They admitted the exact physical representations and operations; the backend
was selecting equivalent implementations inside those operations.

## Problem

The complete width-one path remained decode-bound inside exact quantized
arithmetic. The E2M1 representation has only sixteen values, but the production
kernel repeatedly paid control-flow and generic dispatch cost to recover those
values. This was removable backend work rather than useful model arithmetic.

The repair could not introduce Q8/F16 approximation, manufacture Tensor Core
width, change artifact interpretation, or allow CUDA to reconstruct a new
physical plan.

## Causal analysis

Operation attribution identified two distinct MXFP4 consumers with material
device cost: the specialized MXFP4/Q8 row dot and generic/grouped MXFP4
matrix-vector work. The original scalar decoder contained a switch per weight,
while the packed DP4A decoder expanded four nibbles through a loop containing
the same magnitude classification. The cost therefore scaled with encoded
weight population rather than launch count.

The accepted profile supports that diagnosis. After replacing both decode
forms and selecting MXFP4 once per generic row, the MXFP4/Q8 owner fell from
8.300 to 6.775 ms per token and grouped qtype rows fell from 12.841 to 6.718 ms
per token. Launch count remained exactly 2,413 per token, while complete kernel
time fell by 9.10%. The improvement came from less work inside the existing
operations, not from launch or synchronization compression.

Several alternative hypotheses were measured and rejected:

- fusing the mandatory RMS normalization into the residual/MHC kernel removed
  86 launches per token but left kernel time at 94.321 ms and did not improve
  warm throughput;
- warp-shuffling one packed MXFP4 byte across paired nibble lanes preserved the
  oracle but reduced warm throughput;
- a direct scalar FP32 bit-construction variant and a packed IQ2 magnitude
  variant produced no material end-to-end gain;
- unrolling the scalar MXFP4 FMA loop by two preserved numerical order but
  regressed warm throughput.

These results reinforce the earlier finding that launch reduction alone is not
the current width-one owner.

## Decision

Keep Physical Execution IR v4 and binding v14 unchanged. Implement the repair
as equivalent exact backend microkernel selection inside already compiled
operations:

- derive the scalar E2M1 magnitude arithmetically and apply the sign bit without
  a switch;
- select the MXFP4 row regime once before the generic warp-dot inner loop;
- decode four packed E2M1 nibbles into exact signed bytes with lane-independent
  packed integer operations before DP4A.

The external-reference classification is:

- **ADOPT:** combine quantized decode with the arithmetic that immediately
  consumes it and avoid repeated representation dispatch;
- **ADAPT:** retain YVEX's exact qtype, Physical IR, binding and numeric
  authorities while using an equivalent backend microkernel;
- **REJECT:** import a model-specific execution architecture or derived layout,
  approximate the activation/weight representation, or fabricate execution
  width to force Tensor Core admission.

## Implementation

The CUDA kernel-primitives owner now maps E2M1 magnitude codes to the exact
`0, 1, 2, 3, 4, 6, 8, 12` set with integer arithmetic. The sign bit is applied
to the final FP32 representation, preserving positive and negative zero.

Generic warp-dot execution has a dedicated MXFP4 branch outside the innermost
qtype dispatch. It still uses the same row layout, scale, FP32 accumulation
order, finite-value refusal and publication contract.

The MXFP4/Q8 DP4A helper decodes four low or high nibbles in parallel byte
lanes. Bounded packed additions cannot overflow between bytes, and two's
complement sign application produces the same four signed values consumed by
the existing DP4A instruction sequence.

No model, graph, runtime, artifact, binding, public ABI or family owner changed.
The source remained at its prior 598-line structural budget.

## After

The final target-only live control characterized at an 8.982 token/s median
over five warm samples, with a range of 8.965--9.058 token/s. Every sample
returned the same 15-token bounded output as the baseline. The directly
comparable profile measured 115.376 ms wall time and 93.938 ms of CUDA kernel
time per useful token.

The final DSpark control characterized at a 9.910 token/s median over five warm
samples, with a range of 9.902--9.931 token/s, and preserved the same bounded
output. This is a current characterization, not a directly comparable DSpark
before/after claim.

The change crosses the first structural-acceleration progression threshold of
less than 95 ms of GPU kernel time per token. Production Tensor Core coverage
remains zero, and launch topology is unchanged.

## Quantitative delta

| Fact | Before | After | Delta | Evidence class |
| --- | --- | --- | --- | --- |
| Target-only warm median | 8.326 token/s | 8.982 token/s | +7.87% | directly comparable live fixture |
| CUDA kernel time per token | 103.339 ms | 93.938 ms | -9.10% | directly comparable Nsight Systems profile |
| Wall time per token | 125.372 ms | 115.376 ms | -7.97% | directly comparable Nsight Systems profile |
| Specialized MXFP4/Q8 rows | 8.300 ms/token | 6.775 ms/token | -18.37% | directly comparable kernel attribution |
| Grouped qtype rows | 12.841 ms/token | 6.718 ms/token | -47.68% | directly comparable kernel attribution |
| Launches per useful token | 2,413 | 2,413 | unchanged | directly comparable kernel attribution |
| DSpark warm median | not promoted as a delta | 9.910 token/s | characterization only | final live control |
| Production Tensor Core coverage | 0% | 0% | unchanged | selected production profile |

## Evidence

- Canonical changed-file performance evidence
  `build/qa/evidence/df464a8aa56974aaaf45f7a26ececdbb6e008ed750a7d6f48b46da746600152e.json`
  completed `performance.runtime` with PASS and zero FAIL, SKIP, BLOCKED or
  ERROR. It retained identity-bound eager, piecewise and full attention
  benchmark pairs outside the repository.
- Sanitizer evidence
  `build/qa/evidence/93638657d7d42ba8178cfe2f56572f1b80c15252ad0d44c131b0d87038627d3b.json`
  completed `sanitizer.quant` and `sanitizer.runtime` with 2 PASS and zero FAIL,
  SKIP, BLOCKED or ERROR.
- Final structural evidence
  `build/qa/evidence/fb217ebf043b84797db724da1502db1cdf1a08cde4abc20c6ee9e6ec02bdfefa.json`
  completed 14 tests with 14 PASS. It includes no-NVCC fail-closed behavior,
  architecture, source ownership, repository layout, documentation and
  project-control guards.
- Final focused CUDA qtype evidence
  `build/qa/evidence/6a05c0f9641c2e01e549bd0e71ee50f5a537ca5978a8e3d6fcde9a5f43a82706.json`
  completed PASS. Native SM121, CUDA information, DeepSeek attention and runtime
  MoE focused controls also passed during the checkpoint funnel.
- Exhaustive host-side validation compared all sixteen scalar E2M1 codes and
  all `16^4` packed code combinations for both nibble selections against the
  previous exact decoder.
- Warm target-only and DSpark controls used the same selected artifact,
  binding, GB10 hardware, context, prompt, temperature and 15-token bound.
  Every response preserved exact text and stop reason.
- The complete artifact identity remained
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
  The runtime binding identity remained
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Raw Nsight reports and benchmark records remain untracked external operator
  assets. No complete model artifact, binding or generated performance record
  entered Git.

## Remaining limitations

- This is an exact microkernel improvement inside the existing physical
  operation mix. It does not create the wider compiler-sealed execution regime
  requested by the structural-acceleration wave.
- The remaining selected profile still attributes 31.839 ms per token to
  generic qtype matrix-vector work, 13.391 ms to grouped MoE up, 8.609 ms to
  residual/MHC preparation and 7.376 ms to grouped MoE down.
- At 93.938 ms of measured GPU kernel time per token, even eliminating every
  measured non-kernel gap implies only an approximately 10.65 token/s ceiling.
  The current width-one physical operation mix therefore does not demonstrate
  a path to the 20--24 token/s target.
- Production Tensor Core coverage remains zero. Real compatible width, an
  admitted derived physical representation, or another compiler-owned physical
  regime is still required before Tensor Core execution can be reconsidered.
- Physical IR v4, binding v14, continuous batching, deep context, persistence,
  evaluation, benchmark and release gates remain independently open or
  unchanged. No GB10 closure is claimed.

## Why it matters

YVEX removed exact quantized decode work that consumed nearly one tenth of the
width-one device floor, without changing numerical representation, execution
authority or launch topology; the resulting profile also makes clear that the
next material gain must change physical execution rather than continue scalar
decoder tuning.

## Communication projections

### Short update

YVEX replaced branch-heavy MXFP4 E2M1 decoding with exact packed arithmetic and
moved qtype selection outside the inner warp-dot loop. On the same GB10 fixture,
GPU kernel time fell from 103.339 to 93.938 ms per token and target-only warm
generation improved from 8.326 to 8.982 token/s, with identical output and no
change in launch count.

### Longer post seed

1. Operation attribution showed that exact MXFP4 decode remained a material
   cost inside both specialized DP4A and generic qtype paths.
2. The repair kept the representation exact: all scalar codes and packed nibble
   combinations were compared with the previous decoder.
3. Device time fell while launch count remained unchanged, isolating the gain
   to useful work inside the existing physical operations.
4. The new 93.938 ms/token floor crosses one progression threshold but still
   cannot support the 20--24 token/s objective without a wider physical regime.

### Article seed

**Title:** Removing Exact MXFP4 Decode Tax Without Weakening Numerical Policy

**Thesis:** Quantized inference can remove control-flow overhead while keeping
representation, accumulation and compiler authority unchanged, but the
resulting roofline must still determine when microkernel work is exhausted.

Suggested sections:

1. Attributing MXFP4 cost across generic and specialized operations.
2. Exact scalar and packed E2M1 reconstruction.
3. Numeric and complete-model qualification.
4. Why unchanged launches strengthen the causal result.
5. The remaining 93.938 ms/token architectural floor.

Strongest evidence: a 9.10% reduction in complete CUDA kernel time per useful
token with exact output and unchanged launch topology.

### Visual candidates

- before/after kernel-time attribution for the two MXFP4 owners;
- E2M1 nibble-to-magnitude mapping and four-byte packed decode diagram;
- device-time roofline showing the 93.938 ms/token floor against the 42--50 ms
  target region;
- accepted and rejected hypothesis table.

### Quoteable technical facts

- "Exact MXFP4 decode reduced complete CUDA kernel time from 103.339 to 93.938
  ms per useful token on the retained GB10 fixture."
- "The performance gain occurred with 2,413 launches per token both before and
  after, so it did not come from launch-count compression."
- "Physical Execution IR v4 and binding v14 remained unchanged; the repair was
  an equivalent exact backend microkernel selection."
