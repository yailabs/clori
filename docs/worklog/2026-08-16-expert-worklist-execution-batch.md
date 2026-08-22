# DeepSeek GB10 Expert Worklist Execution Batch

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.EXPERT.WORKLIST.EXECUTION.BATCH.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `435637c27c03646d5b9d6badb59e9f928022c907` |
| Checkpoint | `dbd56909bf3ddc5676d5fc0d923ae3e06ad77150` |
| Subsystem | physical execution planning, expert worklists, runtime MoE, CUDA MoE |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling; architecture characterization |
| Comparability | approximately comparable |
| Publishability | reviewed |

## Before

Physical Execution IR v3 and runtime binding v13 selected MoE kernel classes,
but did not describe the real compatible row population available to an expert
operation. Runtime passed raw routed rows into CUDA, and CUDA reconstructed an
expert-major order before executing them. Total operation rows and kernel shape
were therefore easy to mistake for useful same-expert execution width.

The retained target-only baseline used artifact identity
`d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
and binding identity
`3dfdd0d2a9578a755495673740397d043e079171be4ffb8f2c064831d201d250`.
It measured 7.805 token/s median warm decode and 102.723 ms of device kernels
per warm token. The selected complete-model path executed 0% of its observed
work through the competitive Tensor Core MoE kernels.

The directly preceding sparse-width experiment had reduced isolated m8 Tensor
Core MoE kernel durations, but complete-model qualification rejected it. One
real routed row plus zero-filled instruction rows did not constitute valid
execution width and produced unstable or incomplete runs. Target-only decode
naturally exposed one row, while DSpark verification and compatible future
sessions were the identified legitimate width sources.

## Problem

YVEX lacked one typed authority capable of saying which routed pairs were
physically compatible, where their width came from, and which compiled regimes
could consume that width. Backend-local grouping could order pairs for a
kernel, but could not prove session, candidate, phase, physical-variant,
workspace, or publication compatibility.

Without that boundary, re-enabling Tensor Core execution would either infer
physical policy inside CUDA or manufacture width through padding. Building a
DSpark-only batching structure would also create a second architecture when
continuous multi-session execution later supplies the same kind of compatible
rows.

## Causal analysis

The retained worklogs established four constraints:

- deterministic route/weight ordering and grouped MoE activation reuse were
  already sound;
- removing 255 stream synchronizations did not improve token latency;
- the selected workload remained device-work bound at roughly 102.7 ms of
  kernels per token;
- the rejected m8 path proved that instruction width and real semantic width
  are different facts.

The missing owner was above CUDA. Stable facts such as legal width masks,
representation requirements, workspace class, publication semantics, and
kernel capability belong to Physical Execution IR and the binding. Actual
rows, expert populations, source sessions or candidates, bucket offsets, and
bounded tails belong to one runtime batch instance.

Current DSpark qualification then measured the real population rather than
assuming it. Across 221 observed worklists it executed 3,792 routed pairs in
2,700 expert buckets. Bucket populations ranged from one to six:

| Bucket population | Buckets |
| ---: | ---: |
| 1 | 2,071 |
| 2 | 360 |
| 3 | 149 |
| 4 | 57 |
| 5 | 52 |
| 6 | 11 |

The same run observed worklist widths 1, 5, and 6. Provenance identified 86
speculative-verification worklists, 129 prefill worklists, and six other
compiler-compatible worklists. This proves that real multi-row width exists,
but its same-expert distribution remains sparse. No current compiled policy
admits that distribution to the Tensor Core expert kernel, so eligible and
executed Tensor Core routed-pair counts correctly remained zero.

## Decision

Introduce one family-neutral execution-batch and expert-worklist contract.
Compiler-sealed Physical Execution IR owns legal grouping regimes and explicit
execution-width policy. Runtime seals identities and dynamic row provenance,
then constructs deterministic expert-major buckets. CUDA consumes the validated
order and populations, chooses only an equivalent admitted microkernel, and
handles bounded tails. It no longer infers semantic compatibility from total
rows or qtype.

The external-reference decisions remained:

- **ADOPT:** expert-major buckets must contain real routed pairs for the same
  expert, with only bounded tail lanes left unused;
- **ADAPT:** carry that invariant through YVEX identities, Physical Execution
  IR, runtime transactions, and typed provenance rather than a model-specific
  executor;
- **REJECT:** fake width, CUDA-local compatibility inference, and a separate
  speculative-only worklist architecture.

Physical Execution IR advances to version 4 and runtime binding to version 14
because the new execution-width and worklist policy are identity-bearing
physical semantics. Version 13 bindings fail explicitly rather than acquiring
new meaning silently.

## Implementation

The internal execution-batch contract now distinguishes logical rows,
compatible rows, bucket population, admitted physical width, actual runtime
width, and bounded tail. Typed provenance distinguishes single-row,
speculative-verification, multi-session, prefill, and other
compiler-compatible sources. Batch identities bind the runtime model, binding,
physical variant, execution profile, operation, session, state generation, and
candidate ownership.

The generic graph worklist owner validates sources and policy, orders routed
pairs deterministically by expert, emits contiguous offsets and populations,
preserves source and destination row indices and route weights, and produces a
pointer-free observation. It proves that every pair belongs to exactly one
bucket and that bucket populations sum to the pair count. Its focused tests use
different rows and experts so wrong-expert execution cannot pass accidentally.

Runtime MoE now builds the canonical worklist for both narrow and multi-row
execution. DSpark carries verification provenance and candidate positions into
that same contract. Prefill and target-only decode use it without a parallel
authority. The CUDA production builder consumes device-resident routes to emit
the admitted order, offsets, populations, row maps, and observation without a
host regrouping round trip.

The exact grouped DP4A path consumes the canonical worklist at width one and at
the currently admitted wider populations. The rejected fake-width Tensor Core
selection was removed from production. Tensor Core component kernels remain a
reference capability, but production cannot select them until Physical
Execution IR admits a profitable real population.

Developer telemetry projects worklist counts, pair and bucket counts, maximum
population, provenance, narrow rows, tail rows, and Tensor Core eligible and
executed rows from the canonical observation. It does not create a second
capability authority.

## After

YVEX now obtains real execution width from runtime semantics and represents it
above CUDA. DSpark verification supplies real width up to six rows in the
qualified request, and the worklist proves which of those routed rows share an
expert. Target-only decode retains a valid width-one path and exact output.

Physical Execution IR v4 and binding v14 own the admissible worklist regime.
The backend receives already validated expert buckets and cannot manufacture
width by construction. The same batch schema accepts multiple session sources;
a focused two-source proof validates independent source identities and rejects
ambiguous source-row reuse. No continuous scheduler is claimed.

Tensor Core production coverage remains 0%. The measured population is real
but too sparse for the currently qualified Tensor Core economics, so the wave
does not convert a structural abstraction into a false acceleration claim.
Target-only performance did not regress, but this checkpoint does not establish
the intended 20--24 token/s class.

## Quantitative delta

| Fact | Before | After | Interpretation |
| --- | ---: | ---: | --- |
| Physical Execution IR | v3 | v4 | identity-bearing worklist-width policy |
| Runtime binding | v13 | v14 | stale v13 refuses explicitly |
| Canonical expert worklist | absent | 221 in qualified DSpark request | production path |
| Routed pairs observed | backend-local order | 3,792 typed pairs | exact association retained |
| Expert buckets observed | not projected | 2,700 | contiguous offsets and populations |
| Maximum real bucket population | not authoritative | 6 | measured, not fabricated |
| Tensor Core eligible/executed pairs | 0 / 0 | 0 / 0 | no false promotion |
| DP4A routed pairs | not projected | 3,792 | exact admitted fallback |
| Target-only worklist width | implicit row count | 1 | no regression of narrow path |
| Target-only warm decode | 7.805 token/s | 8.085 token/s median | approximately comparable; not attributed as acceleration |
| Warm device-kernel interval | 102.723 ms/token | 95.892 ms/token median | approximately comparable; 6.65% lower |
| Warm wall token interval | 126.582 ms/token | 128.954 ms/token median | approximately comparable; no latency improvement |
| Categorized request launches | 13,883 | 13,883 | unchanged topology |
| DSpark warm decode | no retained comparable lane | 7.865 token/s median | characterization only |
| Tensor Core production share | 0% | 0% | structural gate remains closed |

The before and after complete-model runs use the same retained artifact and
workload on the same GB10, but binding identity advances from v13 to v14 and the
profiling samples differ. Those performance deltas are therefore approximately
comparable and are not promoted as a causal speedup.

## Evidence

- The retained artifact identity is
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
  Its version-14 characterization binding identity is
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Eight target-only warm runs measured 8.085 token/s median with a
  6.88--8.11 token/s range and identical generated text. Eight DSpark warm runs
  measured 7.865 token/s median with a 7.81--7.92 token/s range and identical
  generated text.
- The current Nsight Systems characterization measured 13,883 kernels and
  713.681 ms of categorized device time. Twelve warm kernel segments measured
  95.892 ms/token median with a 94.164--97.532 ms range.
- The bounded full-model DSpark qualification preserved greedy equivalence,
  acceptance-corpus behavior, speculation cancellation, reasoning-mode
  equivalence, candidate-state transactions, and exact generated output. It
  observed widths 1, 5, and 6 and a maximum same-expert bucket population of
  six without executing a Tensor Core expert pair.
- The target-only control observed 215 worklists, 1,290 routed pairs, 1,290
  buckets, width one throughout, and exact output while exercising reset,
  mutation, cancellation, capacity-stop, and partial-result contracts.
- Focused worklist tests cover deterministic ordering, exact offsets and
  populations, source/destination mapping, route weights, tails, invalid width,
  overflow, stale identity, multi-source compatibility, and ambiguous-source
  refusal.
- `make -j2 check-cuda` passes CUDA qtype and MoE oracles, native SM121
  admission, CLI CUDA smoke, the complete 43-layer attention fixture,
  byte-identical repeat, faults, cancellation, and cleanup.
- Runtime and quant ASan/LeakSanitizer and UBSan lanes pass. The no-NVCC
  fail-closed lane, tiny executable vertical, source ownership, repository
  layout, architecture boundaries, source policy, documentation architecture,
  project control, and diff checks pass.
- Grouped-up and grouped-down worklist construction costs are not separately
  timed in the retained trace. This is an evidence gap, not a zero-cost claim.
  The bounded DSpark run reports 8.002 ms in the aggregate MoE runtime ledger.
- Raw profiles, complete-model logs, bindings, and model artifacts remain
  external identity-bound operator evidence and are not tracked.

## Remaining limitations

- Real same-expert populations reached only six in the qualified DSpark request
  and were dominated by population-one buckets. Current evidence does not
  justify a production Tensor Core cutover.
- The worklist admits future multi-session rows structurally, but only a
  bounded two-source contract proof exists. Continuous batching, fairness,
  cancellation isolation at scheduler scale, and aggregate throughput remain
  unqualified.
- Target-only one-row latency remains governed by the narrow physical regime.
  Multi-session width can improve aggregate throughput but cannot by itself
  establish the 20--24 token/s single-user target.
- The worklist build cost is not yet separated from route/gather/scatter cost
  in complete-model profiling.
- No new derived expert layout or physical artifact was created. The canonical
  artifact remains unchanged.
- Hardware-counter evidence for occupancy, achieved bandwidth, and Tensor Core
  instruction share remains unavailable while non-administrator counters are
  disabled.
- Deep-context qualification, durable prefix/session restore, continuous
  batching, canonical packaging, final deployment, public evaluation, and
  release qualification remain open GB10 gates.
- This checkpoint is not a public benchmark, model-quality evaluation, or
  complete GB10 closure.

## Why it matters

Execution width is now a compiler- and runtime-owned fact rather than a CUDA
guess. YVEX can consume real verification or future session width through one
expert worklist while making fake Tensor Core width impossible, so the next
throughput experiment can test genuine batching economics without redesigning
the MoE boundary again.

## Communication projections

### Short update

YVEX now carries real DeepSeek expert width through a typed Physical IR and
runtime worklist instead of inferring it inside CUDA. A qualified DSpark run
observed 3,792 routed pairs and same-expert populations up to six, but the
distribution remained too sparse for an honest Tensor Core cutover, so
production coverage correctly stayed at zero.

### Longer post seed

1. A smaller Tensor Core tile failed because one real row plus zero lanes was
   instruction width, not model work.
2. Physical Execution IR v4 now seals legal width and representation policy;
   runtime v14 supplies real rows and provenance.
3. One deterministic expert-major worklist serves width-one DP4A, DSpark
   verification, and future compatible sessions.
4. Full-model evidence found real width up to six but sparse same-expert
   populations, so no Tensor Core claim was promoted.
5. The next question is whether continuous compatible sessions create enough
   useful width for aggregate throughput while preserving per-session latency.

### Article seed

**Possible title:** Tensor Core Width Is a Semantic Fact, Not a Block Size

**Thesis:** Sparse MoE acceleration becomes correct only when compilation and
runtime can prove which real routed rows are physically compatible before the
backend selects a tile.

Suggested sections:

1. The false-width m8 experiment and why its component oracle was insufficient.
2. Stable versus dynamic worklist facts across Physical IR and runtime.
3. Deterministic expert-major buckets and identity-bound provenance.
4. DSpark as a real single-session width producer.
5. Why measured width six still did not justify Tensor Core production.
6. Reusing the same contract for future continuous batching.

Strongest evidence is the 3,792-pair population distribution, the zero
Tensor-Core promotion, exact complete-model behavior, and the unchanged
width-one target path.

### Visual candidates

- A stage diagram from compiled width policy through runtime rows, expert
  buckets, and backend microkernel selection.
- A histogram of the measured expert bucket populations from one qualified
  DSpark request.
- A comparison of fake instruction width and real same-expert width.
- A table separating single-session latency from future multi-session
  aggregate throughput.

### Quoteable technical facts

- Physical Execution IR v4 makes admissible expert width explicit; CUDA no
  longer infers semantic compatibility from total rows.
- One qualified DSpark request produced 3,792 routed pairs in 2,700 exact
  expert buckets, with a maximum real population of six.
- Tensor Core production coverage remained zero because the measured real
  population did not satisfy the admitted execution regime.
- Target-only decode retained the same exact width-one worklist path.
