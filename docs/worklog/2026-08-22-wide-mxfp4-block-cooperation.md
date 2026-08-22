# Cooperative MXFP4 Block Decode for Wide Qtype Rows

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `3d3b3cd417b66ea778c20c7ded626e4cc0191422` |
| Checkpoint | `1cfba23e22afc6d67cadccecf1e6a3b6358a4970` |
| Subsystem | exact CUDA qtype kernel primitives |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; CUDA qualification; live runtime qualification; performance |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The accepted width-one CUDA path already used exact branch-free MXFP4 decode and retained
Physical Execution IR v4 plus runtime binding v14 as its physical authorities. Its latest
retained characterization measured 8.982 token/s and 93.938 ms of CUDA kernel time per useful
token. Production Tensor Core coverage remained zero.

For qtype rows wider than sixteen 256-value blocks, `q8_warp_dot` assigned one complete block to
each warp lane. The selected K=8192 MXFP4 geometry therefore made each of 32 lanes serially decode
and accumulate an entire block before the warp reduction. A fresh ten-sample warm control on the
fixed 15-token target-only fixture measured a median of 8.990 token/s.

## Problem

The current physical plan exposed an exact matrix-vector operation, but its equivalent CUDA
implementation did not cooperate within each wide MXFP4 block. That left avoidable serial decode
and DP4A work inside two material K=8192 qtype geometries.

The repair could not change representation, accumulation order, launches, grouping, publication,
artifact interpretation, or compiler authority. A general qtype rule was also not justified:
different qtypes and block populations had already demonstrated different execution economics.

## Causal analysis

The fresh control Nsight Systems profile contained 49,839 launches and 2,537.106 ms of CUDA
kernel time for the complete request. Within `yvex_qtype_matvec`, the two affected K=8192 MXFP4
geometries were:

| Grid X | Calls | Control device time |
| ---: | ---: | ---: |
| 512 | 774 | 147.010 ms |
| 4096 | 129 | 95.149 ms |

The exact dot product already had a bounded cooperative decoder for smaller block populations.
Extending that invariant specifically to 32 real MXFP4 blocks allowed eight four-lane groups to
process eight blocks per round. Four rounds cover all 32 real blocks. A shuffle then returns each
block result to its original lane before the existing warp reduction, preserving reduction order.

The current DS4 source informed the arithmetic-adjacent packed-decode hypothesis. The comparison
was classified as:

- **ADOPT:** keep quantized decode adjacent to the dot product and expose cooperation within real
  encoded work;
- **ADAPT:** retain YVEX's exact MXFP4 representation, block layout and final reduction order while
  assigning four CUDA lanes to one real block;
- **REJECT:** replace the accepted decoder with the DS4 byte-permutation form when its focused
  oracle passed but complete-request evidence did not retain a gain.

Two broader alternatives were rejected. Applying the four-lane rule to every qtype with at most
32 blocks reduced the warm median from 9.253 to 9.192 token/s and increased profiled request kernel
time from 2,456.040 to 2,463.509 ms. The accepted rule therefore remains MXFP4- and width-specific.
Changing numerical representation or manufacturing row width was not considered admissible.

## Decision

Admit one equivalent backend microkernel regime for exact MXFP4 operations containing exactly 32
real blocks. Keep all smaller admitted cooperative regimes and the ordinary one-lane-per-block
fallback unchanged.

This is backend microkernel selection, not physical execution planning: the compiled operation,
row population, qtype, representation, workspace, publication contract and launch topology remain
unchanged. Physical Execution IR v4 and binding v14 therefore do not change.

## Implementation

Checkpoint `1cfba23e22afc6d67cadccecf1e6a3b6358a4970` extends the canonical warp-dot primitive with
four-lane groups for the exact 32-block MXFP4 case. Each round consumes real encoded weights and
the corresponding admitted Q8 activation segment. No zero semantic rows, fake Tensor Core width,
shared-memory staging, backend-global cache, family switch or approximate intermediate is added.

The shuffle destination reconstructs the original block-to-lane assignment before the unchanged
warp reduction. Existing focused numerical tests therefore exercise the new production selector
against the independent qtype oracle rather than accepting a new accumulation contract.

## After

Under the directly comparable ten-sample warm fixture, target-only generation improved from an
8.990 token/s median to 9.253 token/s, a 2.919% increase. The control range was 8.790--9.094
token/s; the checkpoint range was 8.991--9.351 token/s. Every sample produced the same 15-token
bounded output and `length` stop reason.

The complete-request profile retained 49,839 launches while kernel time fell from 2,537.106 to
2,456.040 ms, a 3.195% reduction. The affected grid-512 qtype class fell 39.94% and the grid-4096
class fell 46.87%. Because the profile includes prefill as well as generation, this record does
not convert that full-request delta into a new decode-only GPU-ms/token floor.

The final DSpark control characterized at a 10.138 token/s median over ten warm samples, with a
9.648--10.211 token/s range and the same bounded output. It is characterization only because a
fresh DSpark before/after pair was not retained for this checkpoint.

## Quantitative delta

| Fact | Before | After | Delta | Evidence class |
| --- | ---: | ---: | ---: | --- |
| Target-only warm median | 8.990 token/s | 9.253 token/s | +2.919% | directly comparable, 10 samples |
| Target-only warm range | 8.790--9.094 | 8.991--9.351 token/s | shifted higher | directly comparable |
| Complete-request CUDA kernel time | 2,537.106 ms | 2,456.040 ms | -3.195% | directly comparable profile |
| Complete-request kernel launches | 49,839 | 49,839 | unchanged | directly comparable profile |
| K=8192 qtype, grid 512 | 147.010 ms / 774 calls | 88.300 ms / 774 calls | -39.94% | directly comparable profile |
| K=8192 qtype, grid 4096 | 95.149 ms / 129 calls | 50.550 ms / 129 calls | -46.87% | directly comparable profile |
| Generic four-lane extension | 9.253 token/s | 9.192 token/s | -0.66% | rejected comparable experiment |
| DSpark warm median | not promoted as a delta | 10.138 token/s | characterization only | 10 samples |
| Physical IR / binding | v4 / v14 | v4 / v14 | unchanged | identity evidence |
| Production Tensor Core coverage | 0% | 0% | unchanged | selected production profile |

## Evidence

- Canonical QA evidence
  `ef8ae173ec2ae15813bd49f78591b1c03ecb0f138e77f04bbef33b45f3bf5e6e`
  resolved 101 tests and completed with 101 PASS, zero FAIL, SKIP, BLOCKED or ERROR. It includes
  native SM121 CUDA, no-NVCC refusal, CLI/OpenAI/REPL, tiny vertical, complete target-only and
  DSpark generation, numeric GGUF lanes, identity-bound performance, quant and runtime
  ASan/LeakSanitizer plus UBSan, structural guards and resolved unit owners.
- Focused final qtype evidence
  `4e4c7b21cd6d78d1276e1070bac548f006dd27920fe0fcd53ad5c581254e8ec5`
  passed the independent CUDA qtype oracle after the structural-budget repair.
- The fresh control and checkpoint Nsight reports have digests
  `d9de813b119807da7e35462f8395531d2a68387b4930ab2c93e83bfa6ff30ab3` and
  `a428b22bd57b90afe9a6ae769a6f80684e8cdb8c824e7b161192bda38ac579c3`.
  Their SQLite exports have digests
  `d236262ba3b463493d0b43e3a916b598c36123b42afdf52ec581311f76e482c9` and
  `0492c6ec0fae46049c7cb038d85918725f32b9e4aed8bacaba27cfa3050ab41e`.
- Warm target-only comparison used the same artifact, runtime binding, hardware, context, prompt,
  generation mode, 15-token bound, temperature and output. The comparison used one warmup followed
  by ten retained samples for each side.
- The complete artifact identity remained
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
  Runtime binding identity remained
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- The current DS4 reference at commit `84cc882352757baf628a1776badf7cc54d584e28` informed the
  packed-decode experiment. No external terminology or model-specific execution owner entered
  production code.
- Raw profiles, response JSON, generated benchmark tables/charts, complete artifacts and runtime
  bindings remain untracked external operator evidence.

## Remaining limitations

- This is an exact backend microkernel improvement inside the existing physical operation mix. It
  does not create a wider compiler-sealed execution batch, change publication granularity or
  establish a new Tensor Core regime.
- The full-request profile establishes a device-time reduction but not a new decode-only kernel
  floor. The retained 93.938 ms/token characterization remains the latest accepted width-one
  roofline until a directly comparable decode-only profile supersedes it.
- Generic qtype execution, grouped MoE up/down and residual/MHC work remain material device owners.
  This checkpoint does not establish that another narrow qtype change can reach the 42--50
  ms/token target region.
- Production Tensor Core coverage remains zero. The worklist, real-width and sparse-population
  conclusions from the preceding checkpoint remain unchanged.
- The 20--24 token/s single-session objective, deep-context qualification, persistence, complete
  continuous batching, packaging, model-quality evaluation and release qualification remain open.
  No GB10 closure or public benchmark claim is made.

## Why it matters

YVEX removed a measurable serial decode tax from real K=8192 MXFP4 operations without changing
physical authority, numerical policy or launch topology, while the rejected generic extension
kept the optimization scoped to the geometry that actually benefited.

## Communication projections

### Short update

YVEX made four CUDA lanes cooperate on each real K=8192 MXFP4 block while restoring the original
reduction order. On the same GB10 fixture, warm target-only generation improved from 8.990 to
9.253 token/s and complete-request kernel time fell 3.195%, with identical output and launch count.

### Longer post seed

1. The accepted exact MXFP4 decoder still left one wide encoded block serialized per warp lane.
2. Four-lane cooperation reduced the two affected K=8192 qtype classes by roughly 40--47%.
3. Shuffling each block result back to its original lane preserved reduction order and output.
4. Extending the rule generically regressed, so only the measured MXFP4 geometry was retained.
5. The resulting 2.919% warm end-to-end gain remains a component checkpoint, not GB10 closure.

### Article seed

**Title:** Preserving Reduction Order While Parallelizing Exact MXFP4 Blocks

**Thesis:** A backend microkernel can expose cooperation inside an already compiled exact
operation, but only operation-specific complete-model evidence should determine where that regime
is selected.

Suggested sections:

1. The one-lane-per-block serial decode tax.
2. Four-lane real-block cooperation and round coverage.
3. Restoring the original reduction order with warp shuffle.
4. CUDA oracle, full-model repeat and directly comparable A/B evidence.
5. Why the all-qtype generalization and byte-permutation alternative were rejected.

Strongest evidence: unchanged output and 49,839 launches accompanied a 3.195% reduction in
complete-request CUDA kernel time and a 2.919% warm target-only throughput improvement.

### Visual candidates

- one-lane-per-block versus four-lane cooperative MXFP4 decode diagram;
- K=8192 grid-512/grid-4096 before/after device-time table;
- control and checkpoint warm throughput distribution;
- accepted MXFP4 regime versus rejected generic extension decision table.

### Quoteable technical facts

- "Four-lane cooperation reduced the two affected K=8192 MXFP4 qtype classes by 39.94% and
  46.87% without changing their launch counts."
- "Warm target-only generation improved from 8.990 to 9.253 token/s with the same bounded output."
- "Physical Execution IR v4 and runtime binding v14 remained unchanged because the repair selected
  an equivalent exact microkernel inside the existing operation."
