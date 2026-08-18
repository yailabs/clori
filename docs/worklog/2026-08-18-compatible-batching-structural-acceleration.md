# DeepSeek GB10 Compatible Batching Structural Acceleration

| Field | Value |
| --- | --- |
| Date | 2026-08-18 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.COMPATIBLE.BATCHING.STRUCTURAL.ACCELERATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `ceae429ae497a9132bfbc257c06b951efff3b1f5` |
| Checkpoint | `013962e88453ef5657bd5cb500a333ed869a656b` |
| Subsystem | runtime compatible batching, server scheduling, expert worklists, CUDA MoE, attention state |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | runtime qualification; numerical conformance; live-model qualification; performance characterization |
| Comparability | approximately comparable |
| Publishability | reviewed |

## Before

Physical Execution IR v4 and runtime binding v14 already described legal
execution widths and carried real DSpark rows into one typed execution batch and
expert worklist. The selected production path nevertheless executed independent
server requests serially. A future session was schema-compatible with the
worklist, but no runtime owner could prove that several live sessions were ready
for the same operation and submit their rows as one physical unit.

The retained single-session frontier was approximately 8.1 token/s for
target-only decode, 7.9 token/s for DSpark, and 95.9 ms of device kernels per
warm token. DSpark exposed real verification width up to six, but the
same-expert population remained sparse and the production Tensor Core routed-pair
coverage was zero.

The server could host more than one request, but queueing requests behind the
same model worker was not continuous physical batching. It could improve
concurrency accounting without increasing the useful row width seen by qtype,
attention, MoE, or the output head.

## Problem

YVEX needed one family-neutral owner that could answer three questions before a
backend launch:

- which ready rows are compatible with the same compiled physical operation;
- where each row came from and which session/candidate owns its publication;
- whether waiting for another row is possible and bounded, rather than an
  unconditional latency tax.

The backend could not be allowed to infer compatibility from matching tensor
shapes. The server could not create a separate batching representation. A
single-session request also had to remain immediately executable when no
compatible peer existed.

During full-model qualification a second execution tax became measurable. The
attention candidate-state path allocated or reallocated retained publication
storage for every token and layer. An initial warm probe observed 294 host
allocation events. This lifecycle churn was not model arithmetic and obscured
the attention component measurement.

## Causal analysis

The first multi-session implementation waited for a fixed interval even when no
compatible producer could arrive. Under two otherwise compatible requests the
timeout count reached 184 and coalescing accumulated roughly 0.969 s. The
requests eventually executed together, but the wait policy consumed much of
the benefit.

The repair made the wait conditional on a known compatible cohort. The batcher
counts registered producers, queued compatible tickets, expected group size,
and the remaining possible producers before it waits. If a cohort is impossible
or already complete, execution proceeds immediately. In the comparable
two-request probe, timeouts fell from 184 to 4 and coalescing time fell to about
0.224 s. Per-request completion moved from approximately 3.656 s to 3.394 s,
about 7.2% lower.

The runtime then proved the physical-width limit directly. Two aligned sessions
produced multi-source expert worklists with maximum real width two. Adding three
or four sessions increased concurrent throughput, but the maximum compatible
same-step width remained two because sessions reached layer/operation
rendezvous at different times. Waiting at every layer to force a wider cohort
regressed wall time and was rejected.

An eight-session attempt did not bypass admission. It was refused before unsafe
mutation because the projected requirement was 127,991,515,000 bytes while
123,086,217,216 bytes were available. No 8- or 16-session performance claim is
made.

The candidate allocation trace identified publication-owned arrays as the warm
host churn. Retaining capacity only for arrays present in the first publication
was insufficient: periodic compressed/indexer state caused 84 later
reallocations. Preflighting those future rolling emissions from the sealed
head-dimension and checkpoint geometry removed the remaining warm allocations.

## Decision

Introduce one runtime-compatible batch coordinator that consumes the existing
typed compatibility key and canonical expert worklist. The coordinator is a
generic runtime resource lifecycle, not a server queue and not a backend
scheduler. Server request workers register as producers and submit operation
tickets; runtime validates identities and geometry, forms a bounded FIFO cohort,
and invokes the already compiled physical operation once for the compatible
rows.

Keep the existing width-one path as the fallback through the same owner. A lone
request does not wait for a hypothetical peer. Cancellation before dispatch,
failure publication, stale identity, width overflow, and shutdown have explicit
typed outcomes.

Preserve Physical Execution IR v4 and binding v14. This change instantiates a
dynamic population already admitted by those versions; it does not add a new
compiled physical regime or silently change binding semantics.

The retained external-reference decisions were:

- **ADOPT:** useful hardware width must come from real compatible rows and
  expert-major populations, never fabricated padding;
- **ADAPT:** express scheduling compatibility through YVEX execution identities,
  session transactions, typed worklists, and runtime publication contracts;
- **REJECT:** a DeepSeek-specific global scheduler, backend-local session
  merging, fixed batch-wait latency, and per-layer waiting that increases wall
  time without increasing useful width.

For candidate state, retain publication storage inside the candidate-delta
owner. Replacement is transactional: all required capacity is preflighted
before visible fields change, same-geometry publications reuse storage, and
close releases every retained slot. No backend-global cache was introduced.

## Implementation

The new runtime batching owner maintains a bounded queue, worker lifecycle,
producer registration, typed compatibility matching, cancellation, completion,
and saturated observability counters. Compatibility includes model generation,
binding/profile and physical identities, operation, phase, layer, representation,
geometry, execution class, publication contract, backend, and admitted width.

The server scheduler and OpenAI session owner coordinate ready requests through
that runtime boundary. They preserve per-session state, RNG, output, stop,
cancellation, and failure status. A multi-source execution copies inputs and
outputs across session streams with explicit CUDA event ordering; an early
illegal-address failure exposed missing cross-stream lifetime ordering and was
repaired before live qualification.

Runtime transformer and MoE execution carry the actual batch width into the
canonical expert worklist. The CUDA MoE owner consumes the sealed offsets,
populations, source rows, destination rows, and route weights. It does not
regroup routes or infer semantic width. The exact DP4A path remains selected for
the observed populations because no current population crosses an admitted
Tensor Core crossover.

Developer telemetry now projects physical batches, multi-source batches, rows,
source count, worklists, routed pairs, maximum same-expert population,
population histogram, DP4A/Tensor Core rows, coalescing, rendezvous, and
compatibility mismatches from the runtime authority. Normal chat output remains
unchanged.

The candidate-delta owner now retains ten typed storage classes covering raw,
compressed, indexer, token, and rolling checkpoint state. Same-geometry updates
use alias-safe moves into retained storage. Focused tests prove zero allocation
events on replacement and on the first periodic rolling emission.

The attention benchmark now measures device time only when requested, reports
the immutable encoded artifact backing as resident physical storage, and
validates positive identity-bound residency consistently across eager,
piecewise, and full modes. The canonical QA registry now requires the selected
DeepSeek artifact explicitly instead of allowing the bootstrap default to fail
late.

## After

Two independent server sessions can contribute real rows to one canonical MoE
physical execution. The width is represented and validated above CUDA, and the
backend cannot manufacture it. Width one, DSpark width, and multi-session width
consume the same execution-batch and worklist authorities.

The bounded two-session fixture measured real physical width two and preserved
identical output across both sessions. A controlled warm characterization
measured about 11.45 aggregate token/s at two sessions versus about 8.44 token/s
for the width-one control, a 35.7% aggregate increase. Four active sessions
reached about 12.57 aggregate token/s, but still formed only width-two physical
cohorts. This is serving-throughput evidence, not a single-user decode claim.

Production Tensor Core coverage remains zero. The wave created real width but
did not produce enough same-expert density to justify the existing Tensor Core
tiles. Target-only single-session execution remains on the narrow exact regime;
no 20--24 token/s single-user result is claimed.

The warm full-model attention component now performs zero host allocations,
zero device allocations/frees, zero weight uploads, and zero artifact reads
during its measured region. The final retained eager characterization measured
103.157 ms wall p50 and 92.468 ms device p50 across 20 samples with 1,213 kernel
launches. Piecewise and full graph modes preserved the same tensor and state
digests but did not beat eager wall time in this retained fixture.

## Quantitative delta

| Fact | Before | After | Evidence classification |
| --- | ---: | ---: | --- |
| Physical Execution IR | v4 | v4 | unchanged authority |
| Runtime binding | v14 | v14 | unchanged identity semantics |
| Real multi-session physical width | absent | 2 maximum | measured |
| Two-session aggregate decode | ~8.44 token/s width-one control | ~11.45 token/s | directly comparable characterization; +35.7% |
| Two-session completion | ~3.656 s/request | ~3.394 s/request | directly comparable; ~7.2% lower |
| Coalescing timeouts | 184 | 4 | measured after possible-cohort repair |
| Coalescing time | ~0.969 s | ~0.224 s | measured |
| Four-session aggregate decode | not qualified | ~12.57 token/s | characterization; physical width still 2 |
| Eight-session admission | not attempted | refused before mutation | required 127,991,515,000 B; available 123,086,217,216 B |
| Tensor Core routed-pair coverage | 0% | 0% | measured; no forced cutover |
| Warm attention host allocations | 294 events | 0 | measured allocation epoch |
| Warm attention device alloc/free | not canonical | 0 / 0 | measured |
| Eager attention wall p50 | allocation-affected characterization | 103.157 ms | 20-sample characterization |
| Eager attention device p50 | allocation-affected characterization | 92.468 ms | 20-sample characterization |
| Eager attention launches | 1,339 in early repair probe | 1,213 | changed measured scope; not a causal launch claim |
| Whole-model output | accepted reference | identical | live target-only, stochastic repeat, and DSpark qualification |

The two-session measurements use the same artifact, binding, server process,
prompt, output length, and hardware. The attention component numbers use the
same final source and identities across its three modes, but they are not a
whole-generation token/s measurement and are not compared numerically with the
server decode fixture.

## Evidence

- The retained artifact identity is
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
  The runtime binding identity is
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Canonical QA evidence
  `d04197267e6fcf419f30e5160a981a3c37d31cf41a68c0fadc1e1a9951545a5a`
  passed 101 required tests with zero failures, skips, blocked tests, or errors.
  It includes CUDA native and no-NVCC, CLI/OpenAI/REPL, tiny vertical, numeric
  GGUF, live DeepSeek generation, performance, runtime and quant sanitizers,
  structural guards, and all resolved unit owners.
- Independent live-model evidence
  `33f570f3f808bc5a57b703bb18f421dd9b2b7215c2faed71315e6162183f536d`
  passed target-only CPU/CUDA composition, deterministic stochastic repeats,
  DSpark greedy/stochastic acceptance, speculation cancellation, and both
  operator-reachable generation commands.
- Independent retained performance evidence
  `784e98e615b9f92806f3b358f8d35aa3e40540f2bf86db4ba6594a3731ee0e55`
  validated six baseline/comparison artifacts across eager, piecewise, and full
  modes. All modes reported zero warm host/device allocation, identical tensor
  digest `0db8798f91a3...`, and identical state digest `718835f7ae61...`.
- Structural evidence
  `9ceed957a615033ebb1f77e4d9497423fa158768aa1267b54a724d402c511097`
  passed all 14 architecture, ownership, layout, documentation, project-control,
  topology, and no-NVCC guards.
- Two consecutive CUDA builds passed without cleaning. `git diff --check`
  passed. No model weights, benchmark records, profiles, or generated charts
  were added to the repository.
- Raw server traces, retained benchmark JSON/CSV/SVG, complete artifacts, and
  runtime bindings remain external identity-bound operator evidence.

## Remaining limitations

- Multi-session width is production-reachable only at the current compatible
  MoE worklist boundary. Attention projections, qtype projections, output-A/B,
  and the output head still execute per session. This is not full-transformer
  physical batching.
- The maximum measured physical cohort was two even with three or four active
  sessions. Per-layer waiting increased latency and did not create width, so it
  is not retained.
- Tensor Core eligible and executed routed pairs remain zero. No production
  Tensor Core acceleration or non-zero Tensor Core share is claimed.
- Four-session aggregate throughput improved only modestly beyond two sessions.
  The current scheduler/execution boundary does not yet scale linearly with
  active sessions.
- Eight-session execution was memory-refused; 8- and 16-session latency,
  fairness, memory peak, and throughput are unqualified.
- Target-only single-session latency did not receive a demonstrated causal
  speedup from multi-session batching. The 20--24 token/s single-user target
  remains open.
- Worklist construction, route/bucket formation, and gather/scatter cost are
  observable but do not yet have a retained complete-request separated timing
  ledger.
- Hardware counters remain unavailable while non-administrator GPU profiling
  is disabled. Occupancy, DRAM throughput, and Tensor Core utilization are not
  inferred.
- This checkpoint is not full continuous batching closure, a public benchmark,
  model-quality evaluation, release qualification, or GB10 closure.

## Why it matters

YVEX now proves that independent sessions can create real physical width without
duplicating the execution architecture or letting CUDA guess compatibility. The
result also defines the current limit honestly: useful width reaches two at the
MoE boundary, aggregate throughput improves, but Tensor Core density and
single-user latency remain unsolved. The next frontier can therefore target the
width-destroying projection/attention boundary or a different single-row
physical regime from measured evidence rather than scheduler optimism.

## Communication projections

### Short update

YVEX now combines compatible rows from independent DeepSeek sessions into one
typed MoE execution batch. Two-session aggregate decode increased about 35.7%
in the controlled characterization, while output stayed identical. Real width
reached two, not eight, and Tensor Core coverage remained zero, so this is a
serving-throughput checkpoint rather than a single-user 20--24 token/s claim.

### Longer post seed

1. A typed expert worklist existed, but the server still serialized requests.
2. Runtime compatibility had to include identities, operation, layer, geometry,
   publication, and cancellation rather than tensor shape alone.
3. Fixed batch waits created 184 timeouts; waiting only for a possible cohort
   reduced them to four.
4. Two sessions produced real width two and improved aggregate throughput.
5. More active sessions did not automatically produce wider same-step expert
   populations, and an unsafe width-eight run was refused before mutation.
6. Tensor Core coverage remained zero, preserving the distinction between real
   width and profitable hardware width.

### Article seed

**Possible title:** Real Batch Width Is a Runtime Fact, Not a CUDA Heuristic

**Central thesis:** Continuous serving becomes a physical acceleration only
when compatible session rows survive through the compiled execution boundary;
queue depth and nominal batch size are insufficient evidence.

Suggested sections:

1. From typed worklists to live session producers.
2. The compatibility key and publication isolation.
3. Why fixed coalescing waits regressed latency.
4. Real width two, nominal concurrency four.
5. Why zero Tensor Core coverage is the correct result.
6. The remaining projection and single-row device floors.

Strongest evidence: 101/101 canonical QA, 184-to-4 timeout reduction,
two-session aggregate throughput delta, exact output preservation, zero fake
Tensor Core rows, and fail-closed eight-session memory admission.

### Visual candidates

- compatibility flow from ready sessions to one runtime batch and expert
  worklist;
- nominal active sessions versus measured physical width;
- two-session before/after coalescing timeline;
- Tensor Core eligibility funnel showing why real width two still selects DP4A;
- candidate-delta storage lifetime before and after retained capacity.

### Quoteable technical facts

- “Compatible multi-session width is now represented above CUDA and consumed by
  the canonical expert worklist.”
- “The comparable two-session repair reduced coalescing timeouts from 184 to 4
  while preserving identical output.”
- “Four active sessions still produced a maximum physical width of two; nominal
  concurrency is not hardware width.”
- “The eight-session workload was refused before mutation when projected memory
  exceeded the available capacity.”
- “Warm attention candidate-state storage now reuses capacity with zero measured
  host or device allocations in the retained region.”
