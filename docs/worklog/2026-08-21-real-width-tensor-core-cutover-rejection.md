# Real-Width Tensor Core Expert Cutover Rejection

| Field | Value |
| --- | --- |
| Date | 2026-08-21 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `abd522cdfc317092d648fe0336f841ffc615e08e` |
| Checkpoint | `0b45c2b21e50bb8b9b3fb0dd9a4bbff7f4b27079` |
| Subsystem | CUDA MoE physical execution and compiler-admitted expert worklists |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; CUDA qualification; runtime qualification; performance characterization |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

Physical Execution IR v4 and runtime binding v14 already sealed typed execution batches and
expert-major worklists. Runtime supplied real expert IDs, offsets, populations, source rows,
route weights and provenance above CUDA. DSpark had produced worklist widths up to six, but the
selected production profile admitted no Tensor Core expert regime and reported zero eligible or
executed Tensor Core routed pairs.

The retained exact grouped DP4A path was therefore the production authority. An earlier sparse
Tensor Core experiment had been rejected because one real row plus zero-filled instruction rows
was not valid execution width. The open question was whether the new canonical worklist could
make an exact Tensor Core tile both correct and economically useful when it contained only real
same-expert rows.

## Problem

The existing Tensor Core expert implementation did not consume the canonical bucket metadata.
It selected one routed pair and filled the remaining instruction columns with zeros. Reusing it
would have bypassed the new physical-width authority and repeated the already falsified geometry.

The repair also could not be accepted from a focused kernel speedup alone. The current DSpark
population is sparse, so a locally faster wide tile could still add more launch and dispatch tax
than it removes across the complete model.

## Causal analysis

The corrected width-four fixture supplied four real rows per expert bucket. All 24 routed pairs
were eligible and executed through two Tensor Core launches, while the narrow path processed zero
pairs. The same fixture through grouped DP4A processed the same rows and output. Component
profiling measured grouped expert up plus down at 198.592 us for the Tensor Core regime versus
330.944 us for DP4A, a 39.99% reduction.

Complete-model qualification produced the opposite result. The observed bucket histogram was:

| Bucket population | Bucket count |
| ---: | ---: |
| 1 | 2,071 |
| 2 | 360 |
| 3 | 149 |
| 4 | 57 |
| 5 | 52 |
| 6 | 11 |

A compiler experiment admitting populations of four or more made 554 of 3,792 routed pairs
eligible, about 14.61%. Under the same artifact, binding policy, prompt, generation mode and
hardware, warm throughput fell from a 4.88 token/s median on DP4A to 4.49 token/s, about 7.99%
lower. Output text and DSpark accounting remained identical.

A second compiler experiment raised the crossover to eight. Across 709 worklists, 14,562 routed
pairs and 9,856 buckets, the maximum population remained six. No pair became eligible, Tensor Core
coverage remained zero and warm throughput returned to approximately 4.89 token/s. The threshold
was safe but inert.

The evidence isolates the cause: the exact wide kernel is locally faster, but the current sparse
same-expert distribution does not amortize the additional wide/narrow launch topology. Increasing
an admission threshold cannot create execution width.

## Decision

Keep the corrected real-width Tensor Core implementation as a qualified backend capability, but
do not admit it in the current DeepSeek production plan. The family compiler retains
`tensor_core_minimum = 0` and no Tensor Core expert kernel family. Regenerating the binding
therefore returns the canonical identity
`31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.

The external-reference classification remains:

- **ADOPT:** expert-major tiles contain real routed pairs for one expert; unused lanes are only a
  bounded hardware tail.
- **ADAPT:** YVEX obtains compatibility, offsets, populations, identity and publication semantics
  from its compiler-sealed worklist rather than a model-specific CUDA scheduler.
- **REJECT:** fake width, CUDA-local compatibility inference, and a production cutover justified
  only by component timing.

The next admissible experiment must change the width source or remove the extra launch topology;
it must not lower another threshold over the same population.

## Implementation

The Tensor Core up/down kernels now consume the canonical expert IDs, bucket offsets, bucket
populations and worklist observation. Each MMA column maps to a real routed row. The kernels
validate expert isolation, pair bounds, qtype geometry and maximum tile population before
publication. Bounded unused columns can occur only after the real bucket population.

The grouped DP4A kernels consume the same bucket facts and skip only pairs assigned to an admitted
wide bucket, preventing duplicate execution. The exact Tensor Core accumulation reproduces the
accepted block reduction order, including the Q2_K minimum term and MXFP4/IQ2 scale order. The
backend accepts the hybrid regime only for the compiler-admitted IQ2_XXS gate/up plus Q2_K down
contract and fails closed when a derived or Tensor Core layout lacks a real-width worklist.

Focused CUDA tests now prove a real width-four Tensor Core execution, all routed-pair accounting,
exact width-two DP4A fallback, distinct expert rows, route-weight preservation and refusal/bounds
behavior. The obsolete direct single-row derived execution path was removed.

## After

YVEX has an exact Tensor Core expert implementation that can consume real compiler/runtime width;
fake-width execution is no longer available through that owner. The selected DeepSeek production
profile does not admit the implementation because the complete-model crossover is not proven.

Physical Execution IR remains v4, runtime binding remains v14, artifact identity remains
`d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`, and the canonical binding
identity remains unchanged. Production Tensor Core routed-pair coverage remains zero and the
exact DP4A path remains authoritative. This checkpoint makes no end-to-end speedup claim.

## Quantitative delta

| Fact | DP4A / before | Tensor Core experiment / after | Status |
| --- | ---: | ---: | --- |
| Width-four grouped up + down | 330.944 us | 198.592 us | 39.99% lower; directly comparable component |
| Width-four real routed pairs | 24 narrow | 24 eligible and executed | exact component qualification |
| Threshold-four warm median | 4.88 token/s | 4.49 token/s | 7.99% regression; rejected |
| Threshold-four eligible coverage | 0 | 554 / 3,792 pairs | 14.61%; insufficient economics |
| Threshold-eight warm median | 4.88 token/s control | approximately 4.89 token/s | no Tensor Core execution |
| Threshold-eight live population | not applicable | maximum bucket 6 | 0 / 14,562 eligible/executed |
| Production Tensor Core coverage | 0% | 0% | unchanged |
| Physical IR / binding | v4 / v14 | v4 / v14 | unchanged |

The component comparison and threshold-four whole-model comparison are directly comparable within
their respective fixed fixtures. The threshold-eight throughput is a characterization confirming
that an unreachable policy returns to the narrow regime; it is not promoted as a speedup.

## Evidence

- Canonical QA evidence
  `f01217074fa7f363e472eb4b6caad95f3c6ce7aeeac5d0f950f9f23006066317`
  passed all 101 resolved tests with zero FAIL, SKIP, BLOCKED or ERROR. It includes native CUDA and
  no-NVCC refusal, CLI/OpenAI/REPL, tiny vertical, complete DeepSeek target-only and DSpark live
  generation, numerical GGUF lanes, performance evidence, ASan/LeakSanitizer, UBSan, structural
  guards and every resolved unit owner.
- Focused CUDA evidence exercised the width-four real-row regime with 24/24 eligible and executed
  pairs, two Tensor Core launches, zero narrow pairs and the accepted output oracle. Width two
  exercised 12 narrow pairs with zero Tensor Core launches and exact output.
- The retained Nsight Systems report for the exact component comparison has digest
  `e25b630f0a35499b5d2d6fd3f02c3062e6334153ea4f6b593e63a3a375106357`; its SQLite export digest is
  `d0ffad07eff60dcc4b01eadb3df66a0627d4e8513aab0a30de062fc4054f83ca`.
- Threshold-four complete-model output was exactly the same as the DP4A control. DSpark retained
  35 proposed, 2 accepted, 27 rejected and 7 verified tokens under the fixed fixture.
- The threshold-four experimental binding identity was
  `4945b83643fdbfaecf715c582ff4cc96db04638159fc8e2fbded7a2f569eed0e`; the threshold-eight identity
  was `b6b97eedbcf0c8f584c6db6994a618ddf8d0980900ab5c16365cb14c4846d6a9`. Both remain external
  experiment assets and neither replaced the canonical registry.
- The current DS4 reference at commit `84cc882352757baf628a1776badf7cc54d584e28` informed the
  real-row expert-major invariant. No external architecture or terminology entered production
  ownership.
- Raw profiles, generated bindings, complete artifacts, benchmark JSON/CSV/SVG and server traces
  remain untracked external evidence.

## Remaining limitations

- Current DSpark same-expert populations top out at six and are dominated by populations one and
  two. They do not provide an economical production Tensor Core crossover.
- The retained production profile still has zero Tensor Core routed-pair coverage. A wider real
  producer, launch-integrated persistent expert regime, or another compiler-owned physical regime
  is required before another cutover attempt.
- No lower width-one device floor was demonstrated. The prior 93.938 ms/token characterization
  remains the latest accepted floor and still cannot support the 20--24 token/s objective.
- This checkpoint does not implement broader compatible batching across the transformer, deep
  context, persistence, SSD streaming, multi-device topology, model-quality evaluation, release
  qualification or a public benchmark.
- GB10 remains active. The next performance frontier requires judge/review rather than automatic
  continuation from this checkpoint.

## Why it matters

The checkpoint separates exact Tensor Core capability from production admission: YVEX can now
execute only real expert width, and it also has complete-model evidence that the present sparse
population is not yet worth the cutover.

## Communication projections

### Short update

YVEX made its SM121 expert tile consume only real compiler-sealed worklist rows. The isolated
width-four MoE operation became 39.99% faster, but the complete model regressed 7.99% because only
14.61% of routed pairs crossed the threshold. The cutover was rejected and production remains on
the exact DP4A path.

### Longer post seed

1. A prior Tensor Core path manufactured width with zero rows and was rejected.
2. Typed worklists made real same-expert width available above CUDA.
3. The corrected tile produced a large component gain with exact output.
4. Sparse complete-model populations could not amortize the extra launch topology.
5. Compiler admission was withdrawn while the corrected capability and evidence were retained.

### Article seed

**Title:** When a 40% Faster Tensor Core Kernel Makes the Model Slower

**Thesis:** Physical width must be both semantically real and economically dense; component
speedups do not authorize production cutovers without complete-model evidence.

Suggested sections:

1. Fake instruction width versus real expert width.
2. Compiler-sealed worklists as the compatibility authority.
3. Exact Tensor Core reduction and bounded tails.
4. Component win, sparse population and end-to-end regression.
5. Why production admission remained fail-closed.

Strongest evidence: 39.99% lower focused MoE time accompanied by a 7.99% complete-model throughput
regression under the sparse threshold-four population.

### Visual candidates

- expert bucket histogram with threshold-four eligible coverage;
- DP4A versus real-width Tensor Core component and complete-model deltas;
- compiler worklist to backend tile ownership diagram;
- accepted capability versus rejected production-cutover decision table.

### Quoteable technical facts

- "A 39.99% faster real-width MoE tile regressed complete-model throughput by 7.99%, so YVEX kept
  Tensor Core production admission disabled."
