# Transaction-Local Rolling-State Mutation

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `78d3c5a443817df3672854bc364bf1ae87296981` |
| Checkpoint | `2c575880706f162b15ea37a7ffb747c06cefe804` |
| Subsystem | CUDA encoded-attention rolling-state execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; CUDA qualification; live runtime qualification; sanitizer; performance |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The accepted Physical Execution IR v4 / runtime binding v14 path allocated one main and index
rolling-state version for every input token plus the initial state. Each rolling operation bound
distinct `before` and `after` buffers, and `yvex_attention_rolling_state` copied the complete KV
and score state before publishing the current token's changed slot.

That versioning is required when the caller requests retained prefix checkpoints. It was also
performed in target-only execution where the uploaded state already lived in a private candidate
arena, rollback discarded that arena, and no consumer retained historical per-token versions.

The fixed target-only profile contained 50,992 kernel launches and 2,328.954 ms of CUDA kernel
time. Rolling-state update accounted for 94.916 ms across 2,170 calls. A ten-sample warm control
measured a 9.382 token/s median with a 9.332--9.534 token/s range.

## Problem

The backend treated internal candidate mutation as though every token boundary were an externally
retained checkpoint. That implementation-created boundary copied the complete rolling state even
though ordinary target-only rollback and failure semantics required only the transaction-private
candidate and its final publication.

The cost was not launch overhead: the update already used one launch per rolling operation. The
removable tax was the complete KV and score copy performed inside each launch.

## Causal analysis

Runtime ownership establishes two different physical lifetimes. With retained prefix checkpoints,
each token version may be consumed later and must remain distinct. Without them, the uploaded state
is private to the current transaction; no intermediate version is observable, and failure discards
the arena before logical publication. Main and index rolling state can therefore alias their
`before` and `after` buffers while preserving the existing rollback boundary.

The accepted profile kept exactly 2,170 rolling launches and 50,992 total launches, but reduced
rolling-state kernel time from 94.916 to 9.540 ms. The 85.376 ms component reduction explains
96.05% of the 88.888 ms reduction in complete-request CUDA kernel time. This unchanged launch
topology isolates eliminated state traffic as the causal owner.

Several alternative component hypotheses were rejected before this owner was selected. A fused
mHC rounding/square change reduced complete device time by only about 0.49%; a warp-local mHC
variant regressed it. Paired IQ2 gate/up activation loads and a two-lane K=7168 qtype regime also
regressed complete device time. Those candidates were reverted rather than retained as dormant
fast paths.

The current DS4 source at commit `84cc882352757baf628a1776badf7cc54d584e28` was inspected as a
DeepSeek-specific reference. Its M2 compressor prototype directly writes the changed rolling-state
slot rather than copying the complete state. The comparison was classified as:

- **ADOPT:** a per-token compressor update need only write the changed state slot when no retained
  historical version is required;
- **ADAPT:** YVEX admits in-place mutation only inside its transaction-private no-checkpoint arena
  and keeps distinct versions for prefix-checkpoint execution;
- **REJECT:** import the family-specific fused compressor architecture or apply global in-place
  mutation that would weaken YVEX checkpoint, rollback or publication semantics.

## Decision

Make rolling-state version lifetime conditional on the already admitted
`retain_prefix_checkpoints` fact. Preserve the existing versioned allocation and bindings when the
fact is true. Otherwise allocate one state extent, alias the rolling operation's input and output,
and let the equivalent CUDA microkernel skip only the redundant whole-state copy.

This is backend workspace and microkernel behavior inside the same compiled physical operation. It
does not change grouping, numerical order, public state, execution width, representation or
publication. Physical Execution IR v4 and runtime binding v14 remain authoritative and unchanged.

## Implementation

Checkpoint `2c575880706f162b15ea37a7ffb747c06cefe804` makes main and index state extent selection
conditional on checkpoint retention. One local binder maps both rolling classes to either
ordinal/ordinal-plus-one versions or the single transaction-private version.

The existing rolling kernel detects aliased before/after storage and omits the complete KV and score
copy. It still writes the current slot, performs the same exact emission and overlap shift, and
uses the original checkpointed path unchanged when the pointers are distinct. No new ABI, file,
physical planner, family switch or fallback was introduced.

## After

The directly comparable profile reduced complete-request CUDA kernel time from 2,328.954 to
2,240.066 ms, a 3.817% reduction. Rolling-state time fell from 94.916 to 9.540 ms, an 89.95%
reduction, with the same call and total launch counts.

The directly comparable ten-sample target-only median improved from 9.382 to 9.584 token/s, a
2.145% increase. The checkpoint range was 9.512--9.652 token/s. Every control and checkpoint sample
produced the same 15-token `1 2 3 4 5 6 7 8` output under the same artifact, binding, context,
prompt and target-only generation policy.

One bounded DSpark run produced the same output at 10.051 token/s. It is characterization only
because a fresh retained DSpark before/after sample set was not created for this checkpoint.

## Quantitative delta

| Fact | Before | After | Delta | Evidence class |
| --- | ---: | ---: | ---: | --- |
| Target-only warm median | 9.382 token/s | 9.584 token/s | +2.145% | directly comparable, 10 samples |
| Target-only warm range | 9.332--9.534 token/s | 9.512--9.652 token/s | shifted higher | directly comparable |
| Complete-request CUDA kernel time | 2,328.954 ms | 2,240.066 ms | -3.817% | directly comparable profile |
| Rolling-state kernel time | 94.916 ms | 9.540 ms | -89.95% | directly comparable profile |
| Rolling-state launches | 2,170 | 2,170 | unchanged | directly comparable profile |
| Complete-request launches | 50,992 | 50,992 | unchanged | directly comparable profile |
| DSpark bounded run | not promoted as a delta | 10.051 token/s | characterization only | one live run |
| Physical IR / binding | v4 / v14 | v4 / v14 | unchanged | identity evidence |
| Production Tensor Core coverage | 0% | 0% | unchanged | selected production profile |

## Evidence

- Canonical QA evidence
  `865d8265a0f83c374124b61fe15eb71e122e6739764fc11a42bc3c8d11f80854`
  resolved the 101-test changed-source plan with 101 PASS and zero FAIL, SKIP, BLOCKED or ERROR.
  It included native CUDA, no-NVCC refusal, REPL and protocol integration, complete live DeepSeek
  target-only and DSpark generation, numeric GGUF owners, performance, quant/runtime sanitizers,
  structural guards and all resolved unit owners. Its source delta identity is
  `3c6271406ce150c8119c2df8b70b56ccc6b63c9fe79ce1784a765eb98a9a073c`.
- The retained focused `make check-cuda` execution exited successfully. It covered the native
  SM121 bundle, qtype parity, CUDA CLI smoke, complete 43-layer CPU/CUDA attention comparison,
  byte-identical repeat, 12 device fault cases, cleanup, cancellation and missing-symbol refusal.
  The 43-layer RMSE remained
  `5.6856851103341334e-06`.
- The control Nsight Systems report digest is
  `bdb0fa6326d9305df8af5aeca309e13ae18b51a311f2d54b78c39b6411cdb6c4`;
  the checkpoint report digest is
  `e343254f00e54c9cdac47aade56278b1bbea900cf5df58c47c07a731166e9390`.
  Their exported SQLite digests are respectively
  `502396249bde9f93500b091e7ac69733f7736dc378a3d046bf14a7491465ccf3` and
  `2282a9bb49f7774ec643f3abcd01093649684201c255340278b602d7a100fa0e`.
- The matching response JSON digests are
  `c5d8709fbea970bfbdfd0d3f730537b6d8dba702dcfdef4aafe4bbab9ac9842d` and
  `4659649fbf9104bd840f1cd64d0694f938a280ae864f70b15434699b1698c946`.
  Both record the exact same output and token counts.
- The selected artifact variant identity remained
  `b9825a070028a66af28cdb25614f7a86c6ad1ec396eed6ae961039db1507ce0e`.
  Runtime binding identity remained
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`;
  the native kernel bundle identity remained
  `fb9627c803e22364a914f761169f096b72dbc689c693fc7e56280add55259deb`.
- Warm target-only A/B used the same hardware, artifact, binding, context 4096, 20-token prompt,
  target-only mode, 15-token output and one warmup plus ten measured samples per side.
- Raw profiles, response JSON, benchmark baselines/charts, the complete artifact and runtime binding
  remain untracked external operator evidence.

## Remaining limitations

- Checkpointed execution intentionally retains distinct state versions and still pays the copy
  required by that contract. This checkpoint does not optimize that path or weaken its semantics.
- The repair changes no launch topology, execution width or Tensor Core admission. Production
  Tensor Core coverage remains zero.
- The profile covers the complete bounded request, not a new decode-only GPU-ms/token floor. It
  does not demonstrate the 42--50 ms/token region required for 20--24 token/s single-user decode.
- Generic qtype matvec and grouped MoE up remain the two largest measured device owners at about
  420.6 and 419.5 ms in the checkpoint profile. Their near tie requires fresh causal attribution;
  this record does not select the next patch by symbol name alone.
- Continuous batching throughput, single-user latency, deep-context qualification, persistence,
  packaging, model-quality evaluation and release qualification remain distinct open gates.
- The complete backend/live QA passed, but this performance checkpoint does not claim that every
  historical interactive-console rendering or `/think` UX defect has been closed.

## Why it matters

YVEX removed a full-state copy from every ordinary rolling update by aligning physical state
lifetime with the existing transaction contract, reducing real device work without adding a
hidden planner or weakening prefix-checkpoint correctness.

## Communication projections

### Short update

YVEX now mutates non-checkpoint rolling attention state inside its private candidate transaction
instead of copying the complete KV and score state for every token. Rolling-state device time fell
89.95%, complete-request CUDA time fell 3.817%, and warm target-only throughput improved 2.145%
with identical output and unchanged Physical IR v4 / binding v14.

### Longer post seed

1. Every rolling update copied complete KV and score state into a new per-token version.
2. Prefix checkpoints need that history; ordinary target-only transactions do not.
3. The physical lifetime now follows the already admitted checkpoint-retention fact.
4. Aliased candidate buffers skip only the redundant copy while preserving exact slot updates and
   rollback.
5. Launch counts stayed fixed, isolating removed state traffic as the causal improvement.

### Article seed

**Title:** State Lifetime Is an Execution Policy, Not a Copy Loop

**Thesis:** Transaction ownership can expose safe in-place GPU execution without weakening
checkpoint semantics or moving policy into a backend heuristic.

Suggested sections:

1. The hidden cost of per-token rolling-state versioning.
2. Candidate visibility, rollback and retained checkpoint semantics.
3. One typed operation with two admitted storage lifetimes.
4. Why unchanged launch counts strengthen the causal result.
5. Whole-model evidence and the remaining device-time floor.

Strongest evidence: rolling-state time fell from 94.916 to 9.540 ms while the same 2,170 kernel
calls and exact output were preserved.

### Visual candidates

- checkpointed version chain versus one transaction-private in-place state;
- rolling-state and total CUDA time before/after table;
- unchanged launch count alongside reduced device time;
- post-checkpoint device-owner ranking.

### Quoteable technical facts

- "Non-checkpoint rolling state now mutates in its private candidate arena; retained checkpoints
  keep distinct versions."
- "Rolling-state device time fell 89.95% with no launch-count change."
- "Physical Execution IR v4 and runtime binding v14 remain unchanged."
