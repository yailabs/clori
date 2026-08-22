# DeepSeek GB10 Execution-Topology Roofline

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `801f2c6333b6431594d3377e77f66634b09176fd` |
| Checkpoint | `801f2c6333b6431594d3377e77f66634b09176fd` |
| Subsystem | CUDA execution topology and physical execution planning |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | runtime qualification; component profiling; architectural characterization |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The accepted CUDA path had already reduced attention output and grouped MoE
cost, ordered routed experts by weight, and added an exact MXFP4/Q8 activation
reuse regime. Under the fixed target-only request it still issued 13,924 CUDA
kernels and spent 724.837 ms in device kernels across the complete profile.

The selected artifact identity was
`d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
and the binding identity was
`3dfdd0d2a9578a755495673740397d043e079171be4ffb8f2c064831d201d250`.
The workload used profile
`deepseek4-v4-flash-dspark-runtime-iq2xxs-q2k-mxfp4-b9825a07-v13`,
target-only generation, context 4,096, the prompt `Hi`, and four output tokens.
It produced `你好！有什么可以帮助` with digest
`0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.

Eight current warm, non-profiled repetitions measured a median 7.83 token/s
prefill and 7.805 token/s generation. The generation range was 7.54--7.89
token/s. GB10 therefore remained materially below its 20--24 token/s decode
objective despite the accepted local kernel improvements.

## Problem

The remaining gap could no longer be assigned safely to one named kernel. The
same complete request combined quantized arithmetic, state operations, grouped
MoE, thousands of auxiliary kernels, transfers, graph launches, host dispatch,
and synchronization. Optimizing launch count or a generic qtype symbol without
separating those taxes risked improving a profiler statistic while leaving the
user-visible token interval unchanged.

The specific question was whether the current physical operation mix still had
enough removable dispatch and synchronization tax to approach the GB10 target,
or whether the dominant floor had moved into the device work itself and
required a wider compiled execution regime.

## Causal analysis

The complete request contained 13,883 categorized kernel launches and 715.244
ms of categorized device work. The full report contained 13,924 launches and
724.837 ms, including work outside the bounded request interval.

| Kernel class | Launches | Device time | Share of categorized request time |
| --- | ---: | ---: | ---: |
| Generic qtype matvec | 2,515 | 213.050 ms | 29.787% |
| Attention and state auxiliary work | 5,641 | 132.692 ms | 18.552% |
| Grouped MoE up | 430 | 114.084 ms | 15.950% |
| Grouped attention output-A | 215 | 106.800 ms | 14.932% |
| Specialized MXFP4/Q8 rows | 320 | 69.399 ms | 9.703% |
| Grouped MoE down | 430 | 65.264 ms | 9.125% |
| Other bounded kernels | 2,507 | 7.377 ms | 1.031% |
| Activation conversion | 1,610 | 3.376 ms | 0.472% |
| Routing | 215 | 3.202 ms | 0.448% |

The request also recorded 4,755 D2D copies carrying 254,508,932 bytes, 1,364
H2D copies carrying 173,682,212 bytes, 670 D2H copies carrying 61,938,608
bytes, and 6,560 memsets covering 324,126,712 bytes. Their measured device
durations totalled 18.039 ms. These transfers are real execution work, but the
aggregate request interval spans prefill and decode and therefore does not
justify dividing that number into a per-token claim.

The host runtime recorded 430 CUDA Graph launches, 6,655 direct kernel launch
calls, 7,227 graph-node parameter updates, 172 graph uploads, and 172 graph
instantiations. It also recorded 269 stream synchronizations carrying 435.778
ms of API duration. That duration overlaps queued GPU execution and is not an
additive wall-clock tax.

Warm decode segmentation made the bound explicit. The baseline median token
interval was 126.582 ms, of which 102.723 ms was occupied by device kernels and
25.857 ms was the measured inter-kernel or host gap. Removing the complete
measured non-kernel gap would therefore yield only about 9.735 token/s. Even
that impossible best case remains below the 20--24 token/s objective.

### Execution-tax ledger

| Tax | Evidence class | Observed fact | Interpretation |
| --- | --- | --- | --- |
| Useful arithmetic and quant decode | measured together | dominant named kernel classes above | hardware counters do not yet separate MMA, dot product, and decode |
| Transfer and memset | measured | 18.039 ms device time; 814,256,464 bytes in the complete request | real cost, not safely attributable per decode token from this fixture |
| Direct and graph launch topology | measured | 6,655 direct calls; 430 graph launches; 13,924 kernels | highly fragmented, but launch count alone is not causal evidence |
| Explicit stream completion | measured | 269 stream synchronizations before the experiment | API duration overlaps device work |
| Non-kernel decode gap | derived from trace | 25.857 ms median | complete removal raises the current mix only to about 9.735 token/s |
| Occupancy, achieved bandwidth, Tensor Core utilization | unknown | counters unavailable with `RmProfilingAdminOnly=1` | requires an operator-authorized counter configuration or privileged profiler |
| Redundant traffic inside fused kernels | unknown | no hardware-counter attribution | cannot be promoted from source inspection alone |

The admitted native CUBIN contains SM121 `IMMA.16816.S8.S8` instructions in
Tensor Core kernels for qtype rows and grouped MoE. The selected complete-model
profile, however, launched none of those named Tensor Core kernels. The observed
fraction of this request executed by an admitted Tensor Core kernel was
therefore 0%. The narrow one-row decode and five-row prefill shapes do not meet
the current 16-row Tensor Core admission geometry.

## DS4 comparison

Current DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` uses larger
DeepSeek-specific CUDA Graph islands, keeps routed MoE quantization and expert
work within a graph region, and performs stream completion at a command
boundary rather than treating every graph cache entry as a separate public
completion point.

YVEX **adapted** the macro-execution invariant as a testable hypothesis: stable
work should be completed at the largest boundary permitted by its dependencies
and transaction contract. YVEX **rejected** direct adoption of DS4's
family-specific graph architecture because generic runtime and backend owners
may not reconstruct DeepSeek topology. It also rejected retaining the tested
shared-stream completion change because the measured token interval did not
improve.

Any future widening, fusion, publication change, workspace lifetime, or
multi-operation graph regime must be selected by Physical Execution IR and the
runtime binding. CUDA may choose an equivalent microkernel for an already
compiled operation; it may not infer a second physical plan from qtype and
runtime dimensions.

## Decision

Test the highest-confidence removable topology tax before changing arithmetic:
replace per-graph cached `in_flight` completion markers with one shared-stream
completion that retires every graph known to be ordered before the barrier.
This preserved the existing stream order and graph-cache ownership, changed no
model semantics, and required no approximation.

The experiment reduced request stream synchronizations from 269 to 14, a
94.8% reduction. It did not reduce launch count, device work, or warm decode
time. The profiled generation rate moved only from 7.80 to 7.81 token/s, while
the three-sample median warm token interval increased from 126.582 to 127.993
ms. Total profiled device time increased from 724.837 to 763.659 ms; this noisy
increase is not attributed to the synchronization change, but it rules out an
improvement claim.

The experiment was therefore rejected and removed. No production-code change
survives this checkpoint. The result disproves the hypothesis that cached graph
completion barriers are the causal route to competitive decode in the current
single-request physical regime.

## Implementation

The uncommitted experiment changed only the generic CUDA graph-cache completion
path and added a focused replay check proving that two graphs sharing one stream
could be retired by one barrier without stale `in_flight` state. The focused
CUDA graph test passed.

After complete-model measurement, every experimental source and test change was
reverted through a reviewed patch. The production tree and binary were rebuilt
from checkpoint `801f2c6333b6431594d3377e77f66634b09176fd`. This durable
record is the only repository change: it preserves the falsified hypothesis so
future work does not repeat it.

## After

The executable behavior and identities are unchanged. The execution frontier
is now narrower and better evidenced:

- stream synchronization count is not the dominant removable tax for the
  current single-request path;
- launch count remains structurally high, but a reduction is admissible only
  when it also reduces device work or the measured token interval;
- the existing physical operation mix has a roughly 9.735 token/s upper bound
  if only its measured non-kernel gap disappears;
- competitive 20--24 token/s decode requires reducing roughly 53--61 ms from
  the current median 102.723 ms of device-kernel time per token;
- the selected physical variant does not exercise the Tensor Core kernels that
  are present in the native SM121 bundle;
- the next structural candidate must widen or change the compiler-admitted
  qtype/MoE physical regime rather than add backend-local shape policy.

This is an execution-topology characterization checkpoint, not a component
speedup or end-to-end performance claim.

## Quantitative delta

| Fact | Accepted baseline | Shared-stream experiment | Comparison |
| --- | ---: | ---: | --- |
| Stream synchronizations per request | 269 | 14 | directly comparable; 94.8% lower |
| Total kernel launches | 13,924 | 13,924 | directly comparable; unchanged |
| Profiled CUDA device time | 724.837 ms | 763.659 ms | directly comparable; no improvement |
| Profiled prefill | 8.00 token/s | 7.59 token/s | directly comparable; single run |
| Profiled generation | 7.80 token/s | 7.81 token/s | directly comparable; single run; flat |
| Median warm token interval | 126.582 ms | 127.993 ms | directly comparable three-sample trace; no improvement |
| Median warm device-kernel interval | 102.723 ms | 106.658 ms | directly comparable three-sample trace; no improvement |
| Exact generated-output digest | `0943e8c8...accb55a` | `0943e8c8...accb55a` | identical |

The final unmodified checkpoint was also characterized across eight current
warm non-profiled requests: median generation was 7.805 token/s with a
7.54--7.89 token/s range. This lane is characterization of the retained source,
not a before/after comparison with the rejected experiment.

## Evidence

- The retained Nsight Systems report digest is
  `e4c877bd2bb53f41135c31a766046827c3daeb00f777de6ac1cac39354a63a95`;
  its SQLite export digest is
  `b42a47ac1bf1281f6c75299d34bd0fb799b0bde18c9518504d1e511156048817`.
- The validated experimental report digest is
  `74b0b6f6528eef0c19ed0b1677b5e64089af79db1d6b979aca1922f2ffa78724`;
  its SQLite export digest is
  `1829c24339d86fc0405dc8ee30678ce52624957bdf2087ae6f16a8b1828f853a`.
- Both complete-model profiles produced the same text and output digest. An
  earlier run performed during unrelated GPU-memory contention was invalid and
  is excluded from the comparison.
- `make -j2 check-cuda` passes, including the complete CUDA oracle and
  byte-identical 43-layer attention repeat. Native SM121 SASS admission,
  no-NVCC fail-closed, source ownership, repository layout, architecture
  boundaries, and the production tiny executable vertical pass.
- Runtime and quant ASan/LeakSanitizer and UBSan lanes pass.
- Raw profiles, exported databases, and the complete model remain external
  identity-bound evidence and are not tracked in Git.

## Remaining limitations

- Hardware-counter evidence for occupancy, achieved bandwidth, and instruction
  mix remains unavailable because non-administrator GPU profiling is disabled
  on the machine. This checkpoint does not infer those values.
- The 20--24 token/s GB10 decode objective remains open. Current warm target-only
  execution is approximately 7.8 token/s.
- The next physical regime is not implemented. It must be selected after review
  and must be compiler/binding-owned when it changes width, grouping, derived
  layout, workspace, or publication semantics.
- DSpark economics, real deep-context qualification, durable prefix/session
  restore, continuous batching, canonical packaging, and final deployment
  remain open GB10 gates.
- This checkpoint is not a public model benchmark or release qualification.

## Why it matters

YVEX removed a plausible 95% synchronization-count tax and proved that it was
not the throughput owner. The result prevents launch and synchronization
metrics from substituting for causal token-time evidence and establishes that
competitive GB10 decode needs a wider compiler-admitted device regime, not more
polish on the current narrow operation topology.

## Communication projections

### Short update

YVEX tested whether its 13,924-launch DeepSeek request was primarily blocked by
graph completion barriers. Stream synchronizations fell from 269 to 14, but
warm decode did not improve, so the patch was rejected. The current kernel mix
would reach only about 9.7 token/s even if its measured non-kernel gap vanished;
the next step must change compiler-admitted device work, not just host topology.

### Longer post seed

1. Previous local kernel work reduced qtype and MoE device cost without moving
   decode to the competitive range.
2. A complete request was partitioned by kernel, transfer, launch, graph, and
   synchronization facts.
3. The largest plausible host-topology hypothesis reduced stream barriers by
   94.8% while preserving output.
4. Token time stayed flat, proving those API waits mostly represented queued
   device work rather than additive delay.
5. Warm trace segmentation bounds the existing kernel mix at about 9.7 token/s
   if all measured host gap disappears.
6. The next physical regime must widen useful SM121 work through compiler and
   binding authority.

### Article seed

**Possible title:** When 95% Fewer CUDA Synchronizations Produce No Faster Tokens

**Thesis:** Launch and synchronization counts become useful only when trace
segmentation proves that they own wall time; a rejected optimization can expose
the architectural floor more clearly than another local speedup.

Suggested sections:

1. The accumulated qtype, attention, and MoE optimization frontier.
2. Building an evidence-classed execution-tax ledger.
3. Separating CUDA API wait duration from additive wall-clock tax.
4. The shared-stream completion experiment and exact-output proof.
5. Why a 94.8% counter reduction was rejected.
6. The 9.7 token/s bound and the need for wider compiled physical work.

Strongest evidence: 269-to-14 stream synchronizations, unchanged 13,924 kernel
launches, unchanged output digest, flat warm decode, and the measured
102.723 ms kernel interval per token.

### Visual candidates

- A stacked execution-tax ledger with evidence class on every row.
- The before/after synchronization topology beside unchanged token time.
- Warm token interval split into device-kernel and non-kernel gap.
- Current 9.7 token/s bounded ceiling versus the 20--24 token/s GB10 objective.
- Physical IR authority versus backend microkernel selection.

### Quoteable technical facts

- “Reducing request stream synchronizations from 269 to 14 did not make the
  measured DeepSeek token interval faster.”
- “The current physical kernel mix reaches only about 9.7 token/s if its entire
  measured non-kernel gap is removed.”
- “The selected complete-model profile launched none of the Tensor Core kernels
  present in the native SM121 bundle.”
- “The synchronization patch was rejected because exact correctness alone did
  not establish an end-to-end performance gain.”
