# DeepSeek GB10 Sparse Tensor Core Width Rejection

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.STRUCTURAL.ACCELERATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `f848436c05374b819f2bb8f382387d9d76f5f140` |
| Checkpoint | `71cd9252633ca070335cb059c3ddf035c50e1aa7` |
| Subsystem | CUDA MoE physical execution and compiler-admitted execution width |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; component profiling; architecture characterization |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The accepted Physical Execution IR v3 and runtime binding v13 selected the
portable grouped CUDA MoE kernel for the production DeepSeek profile. The
compiler-sealed family policy admitted the existing SM121 Tensor Core expert
kernel only when an operation population reached 1,024 rows. The fixed
complete-model baseline used artifact identity
`d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
and binding identity
`3dfdd0d2a9578a755495673740397d043e079171be4ffb8f2c064831d201d250`.

The retained target-only characterization measured a median 7.805 token/s,
126.582 ms per warm token, and 102.723 ms of device kernels per token. Removing
the complete measured non-kernel gap would raise that operation mix only to
about 9.735 token/s. The selected production path launched none of the
competitive Tensor Core kernels already present in the native SM121 bundle.

The existing Tensor Core MoE implementation used an exact m16n16k16 integer
MMA tile. A compiler-generated experimental binding with identity
`482b317db88d8b288b801178dfb58fc88c69f3b615b85abb51cb842986c9c833`
lowered the large-row threshold to 24 so the complete-model fixture could
exercise that implementation. The experimental binding remained an external
operator asset and never replaced the canonical profile.

## Problem

The m16 implementation performed one real activation row while filling the
remaining Tensor Core rows with zeros. It established exact arithmetic and
native SM121 reachability, but it did not establish competitive execution
economics. The experimental binding launched 86 Tensor Core MoE kernels in one
short complete-model DSpark request, yet its repeated warm execution was slower
than the retained grouped DP4A regime.

The structural question was whether changing the tile to m8n8k16 could turn
the same narrow physical operation into an efficient Tensor Core unit, or
whether real multiple-row work had to exist before Tensor Core admission.

## Causal analysis

A candidate m8n8k16 implementation reduced the isolated Tensor Core MoE kernel
durations substantially:

| Component | m16n16k16 | m8n8k16 | Observed change |
| --- | ---: | ---: | ---: |
| Grouped expert up | 119.168 us | 74.432 us | 37.5% lower |
| Grouped expert down | 89.952 us | 40.736 us | 54.7% lower |

Those component measurements used the same focused CUDA fixture, one launch per
kernel, the same encoded weights, and the same hardware. They prove only the
local tile cost. They do not establish complete-model performance.

The first fixture was structurally inadequate: every encoded expert row was
identical, so selecting the wrong expert row could still produce the expected
output. The fixture was strengthened to alternate the sign of adjacent expert
rows and to assert that their encoded bytes differ. The m8 candidate still
matched the portable grouped kernel within the existing absolute tolerance,
and CUDA racecheck reported no race. Those results showed that the bounded
kernel invocation was deterministic and numerically admitted for its test
shape; they did not prove the complete model's execution topology.

Complete-model same-session reset qualification falsified the candidate. The
m8 regime produced three output identities across 12 nominally identical
requests: nine expected outputs, two prematurely different outputs, and one
different continuation. The retained DP4A profile produced the expected output
in 12 of 12 requests. Rebuilding the existing m16 Tensor Core implementation
with the same experimental binding also produced the expected output in 12 of
12 requests.

A second m8 experiment removed shared-memory weight staging and reconstructed
qtype words directly from immutable encoded backing. Its focused numerical
oracle passed, but complete-model qualification completed only the first five
requests. The sixth request reported deferred CUDA MoE device status 1, the
seventh was cancelled before attention publication, and the resident server
then stopped. Seven of 12 requests did not complete. Shared-memory staging was
therefore not the root cause.

The current DS4 source at commit
`84cc882352757baf628a1776badf7cc54d584e28` supplied the decisive comparison.
Its exact m8n8k16 MoE tile groups as many as eight real routed pairs belonging
to one expert. Every MMA row maps to a real activation row when present, and
the expert-major worklist carries pair counts and offsets into the kernel.
YVEX's candidate instead put one real activation into the tile and supplied
zero rows for the rest. Reducing the instruction tile did not create physical
execution width.

YVEX already produces a deterministic expert-major pair order and counts unique
experts on the device. It does not yet seal expert bucket offsets, per-expert
tile populations, or an admitted same-expert batch regime in Physical
Execution IR. The current runtime selects the large-row kernel from total
operation population. That fact is not equivalent to the number of compatible
rows available for one expert tile.

For target-only one-row decode, six distinct routed experts normally expose no
same-expert row width. Legitimate width must come from multiple verification
rows or physically compatible independent sessions. The server currently
reports independent session scheduling but not compatible-row continuous
batching. A one-row CUDA microkernel cannot infer or manufacture that width.

## Decision

Reject both m8 candidates and retain the published production MoE regime. Do
not lower the canonical large-row threshold and do not add another CUDA-local
shape heuristic.

The external comparison produced three explicit decisions:

- **ADOPT:** expert-major grouping must feed Tensor Core tiles with real routed
  pairs, and unused rows may only represent a bounded tail.
- **ADAPT:** express expert bucket offsets, tile population, execution width,
  workspace, and publication rules through YVEX Physical Execution IR and the
  runtime binding, then let the backend execute that admitted worklist.
- **REJECT:** copying a DeepSeek-specific executor or treating an m8 instruction
  shape as proof of useful width. The generic runtime must not reconstruct
  family topology.

The next legitimate structural owner is a typed expert worklist and compatible
execution-batch regime. It must derive width from DSpark verification rows or
real multi-session batching. This changes grouping, workspace lifetime, and
publication semantics, so it belongs above backend microkernel dispatch.

## Implementation

No experimental production fast path survives this checkpoint. Both m8
implementations were removed, the original `src/backend/cuda/moe.c` and
`src/backend/cuda/tensorcore.cu` owners were restored, and the native CUBIN and
`yvex` executable were rebuilt from the accepted source.

The durable code change strengthens `tests/unit/cuda/info.c`. Its mixed-qtype
MoE fixture now encodes adjacent expert rows with different values, verifies
that their encoded bytes differ, and retains the existing portable-versus-
native numerical tolerance. This closes the false-positive test geometry that
the experiment exposed without weakening the oracle.

Raw Nsight reports, SQLite exports, complete-model outputs, experimental
bindings, and server logs remain external identity-bound evidence. They are
not repository authorities.

## After

Production execution, artifact identity, binding identity, Physical Execution
IR version, launch topology, memory use, and measured throughput are unchanged.
The canonical large-row threshold remains 1,024, so the selected complete-model
profile continues to use the exact grouped DP4A MoE path.

The new admitted truth is narrower:

- a smaller Tensor Core instruction tile can improve an isolated narrow
  kernel without creating a correct complete-model physical regime;
- the current m16 Tensor Core path is repeat-stable but economically negative
  for sparse one-row work;
- real Tensor Core acceleration requires multiple compatible expert-row pairs,
  not zero-filled instruction width;
- total operation population is insufficient as the compiler's large-row
  admission fact;
- the next physical regime requires a compiler-sealed expert worklist and
  runtime execution batch before another backend kernel is legitimate.

This is a rejected structural hypothesis and an evidence-backed boundary, not
a performance improvement. The 20--24 token/s GB10 objective remains open.

## Quantitative delta

| Fact | Retained production | Experimental observation | Status |
| --- | ---: | ---: | --- |
| Warm target-only decode | 7.805 token/s median | no accepted comparable improvement | unchanged |
| Device-kernel interval | 102.723 ms/token median | no accepted comparable improvement | unchanged |
| Tensor Core production share | 0% of selected workload | 86 launches under experimental binding | not promoted |
| Grouped up component | 119.168 us m16 | 74.432 us m8 | local improvement; rejected end to end |
| Grouped down component | 89.952 us m16 | 40.736 us m8 | local improvement; rejected end to end |
| Retained DP4A repeat | 12/12 expected outputs | not-applicable | stable control |
| Existing m16 TC repeat | not production-selected | 12/12 expected outputs | stable but slower |
| First m8 TC repeat | not production-selected | 3 output identities in 12 requests | rejected |
| Direct-word m8 repeat | not production-selected | 7 of 12 requests did not complete | rejected |
| Production memory delta | 0 bytes | experimental-only assets external | unchanged |

Component timings are directly comparable within the focused Nsight fixture.
Full-model Tensor Core observations are characterization only because the
experimental binding and execution regime differ from the retained production
profile.

## Evidence

- The m8 focused Nsight report digest is
  `a6a918db580015507b67414409ec79c9d0d97fe78b767b81bd010951d168f27d`;
  its SQLite export digest is
  `cecdfc6f55ab488b434d0cce7af8bde0ed773f44186066a30dcdb74d9e262651`.
- The m16 focused Nsight report digest is
  `2a3d17e9969c0b3612d113d1a2c2bf0fc3f6303ad58554188569b0268d2cc191`;
  its SQLite export digest is
  `5693c5a72271089c1099b00e79259e2991533ee505ee55518bdaa2659233ef86`.
- The expected generated-output digest is
  `0943e8c85cdf24b4a50d21681604e1189cf4530e476c79db4f0973729accb55a`.
  The first m8 repeat also produced
  `e1c84ef0ac2ba915ab9c1158b3e4eb494fbd1b136555106dbfef755c32e677b2`
  and
  `a47dccc346f167470cae7caddf1ab6d0bb03f4645d9b088664812fa399f0d694`.
- The strengthened native SM121 MoE oracle passes. `make -j2 check-cuda`
  passes the complete CUDA unit lane, CLI CUDA smoke, full 43-layer attention
  oracle, byte-identical repeat, numerical comparisons, refusal cases,
  cancellation, and cleanup.
- The rebuilt native kernel bundle identity is
  `2880cfcb98347b035fd32a553cabf5e55c85082272538f4e0d1ee1d39d188a67`.
- The existing execution-topology worklog supplies the retained baseline,
  full request attribution, sanitizer status, no-NVCC evidence, structural
  guards, and tiny executable vertical evidence. No production owner changed
  at this checkpoint, so those accepted boundaries were not rerun solely for
  documentary output.

## Remaining limitations

- A typed expert worklist with per-expert offsets and admissible tile widths is
  not yet represented in Physical Execution IR or runtime binding.
- DSpark verification does not yet carry its multi-row width through every
  compatible attention and MoE physical operation.
- Compatible-row continuous batching remains absent; independent session
  scheduling alone does not create GPU batch width.
- The canonical physical artifact has no newly admitted Tensor-Core-oriented
  derived expert layout. This checkpoint does not authorize one implicitly.
- The selected production workload still executes 0% of its observed work
  through the competitive Tensor Core kernels already present in the bundle.
- The retained measured device floor remains 102.723 ms/token, above all staged
  structural targets and the approximate 42--50 ms/token target region.
- Hardware-counter evidence remains unavailable while non-administrator GPU
  profiling is disabled. No occupancy or bandwidth claim is inferred.
- Deep-context qualification, durable prefix/session restore, continuous
  batching, canonical packaging, and final deployment remain open GB10 gates.
- This checkpoint is not a public benchmark, model-quality evaluation, release
  qualification, or GB10 closure.

## Why it matters

The experiment separates Tensor Core instruction reachability from useful
Tensor Core execution width. YVEX now has direct evidence that another narrow
kernel cannot close GB10: the compiler and runtime must expose real compatible
expert-row batches before the backend can execute an efficient exact tile.

## Communication projections

### Short update

YVEX reduced isolated DeepSeek MoE Tensor Core kernels by 37.5% and 54.7% with
an m8 tile, then rejected the change because identical complete-model requests
produced three outputs. The missing ingredient was not a smaller instruction:
it was real expert-major batch width. The next regime must be sealed by the
Physical Execution IR rather than inferred inside CUDA.

### Longer post seed

1. The retained DeepSeek path is bounded near 9.7 token/s if only its measured
   host gap disappears, so device economics must change.
2. A smaller exact Tensor Core tile materially improved isolated MoE kernels.
3. A stronger oracle exposed that the original fixture used identical expert
   rows and could hide row-selection defects.
4. Full-model reset qualification produced three outputs from one nominally
   identical m8 regime, while DP4A and m16 controls were stable.
5. Removing shared-memory staging did not repair the failure.
6. The useful external invariant is expert-major grouping of real routed pairs,
   not the external engine's vocabulary or architecture.
7. YVEX must represent that width in its compiler-sealed physical plan before
   another Tensor Core kernel can be admitted.

### Article seed

**Possible title:** A Faster Tensor Core Kernel That YVEX Correctly Rejected

**Thesis:** Component speed and exact focused numerics cannot establish an
execution regime when the production dependency graph does not supply the
physical width the instruction assumes.

Suggested sections:

1. The 102.7 ms/token device floor.
2. Why m16 zero-filled width was exact but uneconomic.
3. The m8 component win and the degenerate expert-row fixture.
4. End-to-end non-determinism and the shared-memory falsification.
5. Real expert-major width as the reusable invariant.
6. Moving worklist and batch authority into Physical Execution IR.

Strongest evidence: 37.5% and 54.7% component reductions, three output
identities in 12 complete-model requests, stable 12-of-12 DP4A and m16
controls, and zero accepted production behavior change.

### Visual candidates

- Component speedup beside the rejected end-to-end correctness funnel.
- One real row plus zero padding versus eight real same-expert pairs.
- Current pair order versus the missing compiler-sealed expert worklist.
- Physical IR authority above runtime batching and backend Tensor Core tiles.
- The unchanged 102.723 ms/token device floor versus the 42--50 ms target
  region.

### Quoteable technical facts

- “A 54.7% faster isolated MoE kernel was rejected because the complete model
  did not preserve repeat output.”
- “A smaller Tensor Core instruction did not create real execution width.”
- “The retained DP4A and existing m16 controls each produced the expected
  output in 12 of 12 same-session reset requests.”
- “The next admissible fast path needs compiler-sealed expert worklists, not a
  backend-local shape heuristic.”
