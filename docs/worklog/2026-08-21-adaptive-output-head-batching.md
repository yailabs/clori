# Adaptive Prefill and Compatible Output-Head Batching

| Field | Value |
| --- | --- |
| Date | 2026-08-21 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `41c9ba604a2e610d15e9781155c7bdc77276e6ae` |
| Checkpoint | `5c362b50aca813d820806bd80325febaacc42979` |
| Subsystem | server admission, runtime compatible batching, logits, speculative generation |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; runtime qualification; CUDA qualification; sanitizer; performance characterization |
| Comparability | approximately comparable |
| Publishability | reviewed |

## Before

The server used a fixed 64-token prefill chunk whether it admitted one session
or several concurrent sessions. At parallel width four, that fixed workspace
multiplied across the active set even though the runtime already owned a real
compatible execution width. Operators could override the chunk, but omission
did not select a resource-aware serving default.

Physical Execution IR v4 and runtime binding v14 already sealed compatible
execution width. The runtime carried that width through the canonical MoE
worklist, but each compatible session still projected its output head in a
separate physical operation. Sampling was correctly session-local, so the row
width was discarded immediately before the last large projection.

DSpark also used the prompt prefill-chunk setting as the upper bound for its
target-feature projection. A small serving prefill chunk could therefore
truncate an independently admitted speculative verification extent.

## Problem

The server needed an automatic prefill policy that preserved width-one
interactive behavior without multiplying the same large workspace across a
concurrent active set. The resolved decision also had to remain inspectable and
an explicit positive override had to remain authoritative.

The runtime needed to consume real compatible session rows in one output-head
operation without merging session state, publication, sampling or RNG. CUDA
could not infer compatibility from matching shapes, and the change could not
create another batching authority beside the existing typed coordinator.

Finally, prompt chunking and speculative lookahead were separate resource
owners and could not continue sharing one limit.

## Causal analysis

The retained four-session profile showed that compatible width survived MoE
but was destroyed at the output-head boundary. Sixty useful generated rows
produced sixty session-local sampling operations. After batching, the same
sixty rows required fifteen four-row output-head operations while sampling
remained sixty independent operations. The output-head operation therefore had
real reusable width; the sampler did not.

The accepted trace measured 93.961 ms across the fifteen four-row output-head
operations, or 6.264 ms per physical batch and 1.566 ms per real row. A retained
width-one attribution measured approximately 2.901 ms per row. These profiles
are approximately comparable rather than a release benchmark, but they support
the causal result that the wider physical output-head operation reduced its
per-row cost.

An attempted multi-session attention projection used the same real row width
and passed its focused numerical oracle, but regressed the live fixture by
5.24%. It was removed. This rejected result shows that compatible width is not
by itself sufficient reason to widen every operator.

The DSpark failure was an ownership error rather than a kernel limit: the
feature workspace was admitted from its own bounded speculative row capacity,
but execution checked the unrelated prompt prefill chunk.

## Decision

Resolve an omitted server prefill chunk before capacity admission. Width-one
operation retains the established interactive chunk of 64. Concurrent serving
starts from the admitted scheduling width with a floor of four, capped by the
admitted context. A positive CLI override remains exact; zero is refused as an
invalid explicit override. Server status publishes the resolved value.

Route compatible output-head requests through the canonical runtime batch
coordinator. The compatibility key seals model generation, binding, physical
variant, execution profile, operation, phase, representation and admitted
width. Runtime supplies actual session rows. The logits owner executes one
already-admitted multi-row output-head operation and returns each result to its
original session. Sampling remains session-local.

Physical Execution IR stays at v4 and the runtime binding stays at v14. They
already admit the operation and maximum width; this checkpoint instantiates a
dynamic population without changing grouping, representation or publication
semantics. Backend selection remains equivalent microkernel execution inside a
compiled operation, not a hidden physical planner.

The retained external-reference classification is:

- **ADOPT:** use real compatible rows to issue a wider physical projection;
- **ADAPT:** seal compatibility and publication through YVEX identities,
  sessions and the canonical coordinator;
- **REJECT:** fake row width, family-specific runtime scheduling, approximate
  output representations and generic widening without live evidence.

## Implementation

The server owns adaptive prefill resolution and records the admitted value in
its typed summary. CLI status exposes the value in human and JSON projections,
and explicit `--prefill-chunk 0` is rejected. Unit, CLI and tiny-vertical tests
cover the automatic values, explicit override and inspection surface.

The runtime generation owner submits device-resident output-head rows to the
existing compatible batcher. A bounded one-millisecond coalescing window allows
an already-ready cohort to rendezvous without introducing a second scheduler.
Tickets are deterministically ordered by session identity. The logits owner
validates one model and output-head plan, distinct sessions and device-resident
sources; gathers hidden rows device-to-device; executes one encoded multi-row
projection; and scatters logits into session-owned publications. The leader
owns physical accounting while every row retains its own completion contract.

Target-only logits workspace admits the compatible width, while sampling
continues to admit one row. DSpark keeps its existing bounded speculative logits
extent and now validates target-feature rows against that workspace rather than
the prompt chunk.

The live OpenAI qualification now hosts four sessions with the real artifact
and binding, submits four concurrent completions, checks continuous compatible
batching and verifies the immutable artifact-backed residency contract.

## After

An ordinary width-one server resolves prefill to 64. Parallel widths two and
four resolve to four; width eight resolves to eight. The choice is visible in
server status and remains subordinate to adaptive admission. An explicit
positive override is unchanged.

Four compatible target-only sessions now execute one four-row output-head
operation per generation step where they rendezvous. No session state, logits
publication, sampler, RNG or stop policy is shared. The same canonical batching
authority still owns width one, DSpark width and multi-session width.

The approximately comparable four-session target-only fixture improved from a
13.126 token/s median to 13.793 token/s, a 5.08% aggregate gain. The component
output-head operation count fell from the derived width-one topology of sixty
operations to fifteen measured four-row operations for sixty useful rows.

Single-session controls remained healthy: target-only characterized at 8.342
token/s median and DSpark at 8.962 token/s median. Four-session DSpark
characterized at 12.181 aggregate token/s and remained slower than the
four-session target-only fixture; this checkpoint does not claim competitive
DSpark serving.

## Quantitative delta

| Fact | Before | After | Evidence class |
| --- | --- | --- | --- |
| Four-session target-only aggregate median | 13.126 token/s | 13.793 token/s | approximately comparable; same artifact, binding, hardware and 4 x 15-token fixture |
| Output-head physical operations for 60 rows | 60 derived width-one operations | 15 measured four-row operations | topology plus profiler evidence |
| Output-head time per real row | approximately 2.901 ms | 1.566 ms | approximately comparable component profiles |
| Target-only width-one median | not promoted as a before/after delta | 8.342 token/s | characterization only |
| DSpark width-one median | not promoted as a before/after delta | 8.962 token/s | characterization only |
| DSpark four-session aggregate median | not promoted as a before/after delta | 12.181 token/s | characterization only |
| Production MoE Tensor Core routed-pair coverage | 0% | 0% | unchanged binding-v14 DP4A regime |

## Evidence

- Canonical changed-file QA evidence
  `build/qa/evidence/d86669656cf3b37ee2c030d63114b0b83419e16c6b49679f6f6399a77222bc39.json`
  resolved 99 required tests and completed with 99 PASS and zero FAIL, SKIP,
  BLOCKED or ERROR. It includes structural, numeric, runtime, sanitizer,
  no-NVCC and complete DeepSeek live generation evidence. The report records
  build identity
  `026eb5aaf4970cf117f5752494c9aa2dc48d17700fa62d1089cf09d8f38ce1ef`
  for the checkpoint source delta.
- Four-session OpenAI live evidence
  `build/qa/evidence/5cf2d27fa4d37ef0cc5ffde11cd7aa59b443363a2f22a0db68cade323b6fe140.json`
  completed PASS with the complete external artifact, CUDA, concurrent
  sessions, bounded cleanup and artifact-backed residency checks.
- The focused CUDA logits oracle passed three distinct rows against the
  reference with maximum absolute error `0.000217437744140625` and RMSE
  `0.000021473`.
- Nsight Systems report `yvex-p4-output-head` retained externally measured the
  fifteen multi-row output-head operations and the component timings above.
  Raw profiler records remain untracked operator assets.
- Warm target-only and DSpark samples used the same selected
  DeepSeek-V4-Flash candidate, binding identity
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`,
  GB10 hardware and bounded generation fixture. The complete artifact identity
  remained
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
- Two consecutive builds, `git diff --check`, ownership, layout, architecture,
  documentation and project-control guards passed. No complete model artifact
  or raw performance record entered Git.

## Remaining limitations

- Compatible batching still does not make attention or sampling multi-session
  physical operations. The rejected attention experiment remains evidence that
  those owners require a different causal geometry.
- The complete trace still attributes 27.8% of device time to generic
  `qtype_matvec`; grouped MoE and other qtype owners remain materially larger
  than the now 0.8% output-head trace share.
- Production MoE Tensor Core routed-pair coverage remains zero. This checkpoint
  creates no fake width and does not alter the exact DP4A regime admitted by
  binding v14.
- Parallel width eight was refused by the existing OOM-safe admission boundary
  because the projected requirement exceeded available memory. No width-eight
  throughput claim is made.
- Four-session DSpark remained slower than target-only. Speculative policy,
  verification economics and the single-user 20--24 token/s objective remain
  open GB10 work.
- Deep context, durable session persistence, complete continuous transformer
  batching, evaluation, benchmark and release qualification are not closed.

## Why it matters

YVEX now preserves real compatible width through one more large production
operation and improves four-session serving throughput without inventing rows,
weakening numerical policy or merging logical sessions.

## Communication projections

### Short update

YVEX now batches compatible DeepSeek output-head rows through the same typed
runtime coordinator already used for expert work. Sixty session rows became
fifteen real four-row projections, lowering the measured component cost per row
and improving the comparable four-session target-only median from 13.126 to
13.793 token/s while keeping sampling session-local.

### Longer post seed

1. Real multi-session width already survived through the expert worklist but
   disappeared before the output head.
2. The runtime extended its identity-sealed compatibility contract rather than
   teaching CUDA to merge sessions.
3. One four-row projection now publishes four independent logits rows, followed
   by four independent samplers.
4. The wider operation reduced output-head cost per row, but profiling moved the
   frontier back to qtype and MoE rather than closing single-user performance.

### Article seed

**Title:** Preserving Real Batch Width Through an Identity-Bound Output Head

**Thesis:** Continuous execution width becomes useful only when it survives
physical operator boundaries without erasing session ownership.

Suggested sections:

1. Where compatible width disappeared in the original execution graph.
2. Identity-sealed batching above CUDA.
3. Gather, grouped projection, scatter and session-local sampling.
4. Adaptive prefill admission under a 128 GiB system constraint.
5. Component gain versus remaining end-to-end owners.

Strongest evidence: sixty rows mapped to fifteen four-row output-head
operations, and the four-session target-only median improved by 5.08% under an
approximately comparable fixture.

### Visual candidates

- Before/after execution path showing four independent output-head operations
  versus one grouped projection plus four session-local samplers.
- Four-session throughput and output-head per-row timing table.
- Execution frontier chart retaining qtype and MoE as the remaining owners.

### Quoteable technical facts

- "Sixty real session rows executed as fifteen four-row output-head operations."
- "Compatible batching ends before sampling, so RNG and stop policy remain
  session-local."
- "The four-session target-only median improved from 13.126 to 13.793 token/s
  under an approximately comparable fixture."
