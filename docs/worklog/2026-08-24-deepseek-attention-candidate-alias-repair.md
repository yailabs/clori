# DeepSeek Attention Candidate Alias Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-24 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `2df3b84cc840dfca8b38f6fc387a833169b5598e` |
| Checkpoint | `5d1771b434905b16ba48ae5e6807e7ef3439f308` |
| Subsystem | CUDA attention residency and persistent sequence state |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | CUDA chunk-equivalence live; attention oracle; target-verification equality; sanitizer; structural QA |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

CUDA attention distinguished its transient local workspace capacity from the smaller committed
local-ring publication capacity. State publication used the committed-ring capacity, but the
residency-alias decision still admitted an input whenever it fit the wider transient workspace.

The prior bounded chunk-equivalence fixture stopped before the boundary where those capacities
diverge. A 127-token execution therefore passed while a 128-token execution could alias candidate
input with another range in the reusable attention workspace.

## Problem

At target layer 2, token position 124 and candidate width 4, the committed local ring admitted 127
rows while transient execution required 128. The residency path treated the 128-row candidate as
persistent input because it compared against transient capacity. The resulting local raw-KV alias
overlapped compressed-state workspace by 1,032 bytes.

This was a lifecycle and ownership defect, not an attention equation or qtype defect. Chunked and
repeated one-token execution could consume different bytes even though both executions used the
same model, binding, tokens and committed starting state.

## Causal analysis

The storage/publication separation was introduced deliberately: the backend may need a wider
temporary envelope while the persistent ring publishes only its admitted capacity. The alias
predicate, however, retained the earlier single-capacity assumption. At the first request that
filled the transient row beyond the committed ring, a device-resident source pointer escaped the
ownership boundary that would otherwise allocate separately owned candidate storage.

Bounded stage observation localized the first difference to the local raw-KV input before later
attention and MoE work could amplify it. Replacing the alias predicate with the committed-ring
capacity removed every observed stage divergence. The same 128-token fixture then produced exact
normalized-hidden and persistent-state equality between chunk-4 and repeated chunk-1 execution.

Focused attention comparison remained within its existing independent contract across all 43
layers: the reference and CPU values were exact, and CUDA maximum absolute difference was
0.00390625 with RMSE 0.0000056856851103341334. Direct MoE qualification also remained green.

The historical complete Transformer CPU/CUDA comparison is a separate evidence gap. It was red at
the baseline and older published DeepSeek checkpoints, and the CPU path does not reproduce all
source BF16 publication boundaries. This repair neither weakens that check nor claims it as an
independent source oracle.

## Decision

Use the committed local-ring capacity as the sole admission boundary for persistent local-state
aliasing. Transient candidate capacity remains the workspace owner, but it cannot grant persistent
ownership to bytes that have not crossed publication.

Retain one attention implementation and one state-residency lifecycle. Do not add synchronization,
copy every persistent input, or special-case DeepSeek in generic runtime code. Extend the live
Transformer fixture to exercise the exact 128-token boundary without feature observation, because
observer downloads can introduce synchronization that hides an asynchronous ownership defect.

The following alternatives were rejected:

- retaining the transient-capacity comparison, because it permits the proven overlap;
- adding global device synchronization, because it does not repair buffer ownership;
- treating the 32-token observer-assisted pass as closure, because it did not reach the boundary
  and observation could mask asynchronous behavior;
- changing qtype activation guards, because an A/B candidate did not change complete Transformer
  output and was removed.

## Implementation

The CUDA attention run state now names the committed bound `local_ring_capacity`. Allocation may
resolve local input to persistent residency only when initial committed rows plus current candidate
rows fit that bound. State publication consumes the same field, leaving one canonical fact for the
ring lifecycle.

The live DeepSeek Transformer runner now supports a bounded CUDA-only comparison with independently
owned execution sessions, configurable token/context capacity, and no per-layer feature capture.
The canonical live target executes 128 deterministic tokens as chunk width 4 and repeated width 1
before its pre-existing complete CPU/CUDA characterization. It compares normalized hidden values
and persistent-state digests exactly.

Physical Execution IR v4, runtime binding v14, artifact identity, qtype policy, expert worklists and
backend kernel selection are unchanged.

## After

The final 128-token production fixture reports:

```text
cuda_chunk_equivalence=pass tokens=128 layers=43
```

Chunk-4 and repeated chunk-1 execution now publish identical normalized hidden state and identical
persistent state. The same target-only and DSpark greedy 256-token fixture produced an identical
committed token sequence with digest
`76eaa3a4b8baa05d28098fd030882714175e2f4a18c3fb78e0a353a53d44bd08`.

The repair changes no performance regime. The accepted production path remains the compiler-sealed
DP4A-class narrow regime with Physical Execution IR v4 and runtime binding v14.

## Quantitative delta

| Fact | Before | After | Evidence class |
| --- | ---: | ---: | --- |
| Largest passing diagnostic context | 127 tokens | 128 tokens | directly comparable CUDA live |
| Local ring capacity at failing boundary | 127 rows | 127 rows | runtime state fact |
| Transient candidate capacity | 128 rows | 128 rows | CUDA workspace fact |
| Incorrect overlap | 1,032 bytes | 0 bytes | bounded stage localization |
| Chunk-4 versus chunk-1 output | divergent | exact | CUDA live |
| Chunk-4 versus chunk-1 persistent state | divergent | exact | CUDA live |
| Retained performance claim | none | none | not-applicable |

## Evidence

- Repair checkpoint `5d1771b434905b16ba48ae5e6807e7ef3439f308` contains the ownership fix and
  its bounded production regression.
- The exact 128-token CUDA-only production runner passed all 43 layers without feature observation,
  OOM or state mismatch.
- Focused attention evidence
  `806e03d97e4d688ea54782a49e281f91d977533907d9719e31d48f66d301f4b1`
  covered all 43 attention layers, 12 fault cases and checked cleanup. Its final runtime-oracle row
  was blocked by an unrelated bootstrap-artifact identity mismatch, not attention math.
- Canonical structural evidence
  `8acdab3d5198a60e797392686e613bf8fa353c564d3b3d7ef238bf994874d406`
  recorded 14 PASS and zero FAIL, SKIP, BLOCKED or ERROR, including no-NVCC refusal, architecture,
  ownership, layout, operator registry and project control.
- Canonical sanitizer evidence
  `87fd47e07756af9d49f6d25ddd51bf555589e349d923c4622e4696bf5ae67368`
  recorded ASan/LeakSanitizer and UBSan PASS for both registered sanitizer owners.
- `git diff --check`, the focused Transformer build, repository layout and source-layout guards
  passed on the retained diff.
- The admitted artifact remains
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`;
  runtime binding v14 remains
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.

## Remaining limitations

- This repair does not establish the 20--24 token/s target and makes no performance claim.
- A fresh uncontended current-tree performance baseline and decode-only causal profile remain
  required before selecting the next matrix/tile physical regime.
- The existing full Transformer CPU/CUDA comparison remains an evidence gap. It must be reconciled
  against an independent source-faithful full-stack oracle rather than weakened to accept drift.
- Production Tensor Core MoE coverage remains zero in the accepted narrow regime. The prior
  real-width Tensor Core threshold remains rejected because measured bucket density did not make it
  economically useful end to end.
- The macro GB10 optimization milestone remains active; this is a repair checkpoint, not closure,
  evaluation, release qualification or a public benchmark.

## Why it matters

Persistent state can now be aliased only after it fits the exact publication contract. Performance
work resumes from deterministic 128-token execution rather than from a context-dependent workspace
overlap that could corrupt every later layer.
