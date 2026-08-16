# MiniMax Chunked BF16 Attention Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `f055d5de69e2bda1e36aa91cf541a445ae530392` |
| Checkpoint | `aa4e547a5309a39749de5312acaab36f930c523a` |
| Subsystem | generic CUDA dense Transformer attention and MiniMax Omni conformance |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; numerical conformance; bounded runtime and media characterization |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

Generic CUDA grouped-query attention used the exact online four-query kernel for every admitted
geometry. It avoided a quadratic score allocation, but every attention result remained a scalar
CUDA reduction. The MiniMax path was bounded to 2,048 packed rows, and the prepared 466-row
fifty-block oracle had not yet been executed while another model occupied most unified memory.

The accepted 192x192, 124-frame media path was playable and synchronized but visually
unrecognizable. Its prior 49-evaluation run took about 49 minutes. That result established native
execution and publication only; it did not establish prompt fidelity or model quality.

## Problem

Source-qualified MiniMax geometry needs tens of thousands of packed rows. Exact full attention
therefore needs a path whose matrix work can use GB10 tensor cores without materializing the
complete quadratic score matrix. Increasing the row admission cap while the scalar kernel was
the only implementation would have lacked both a credible execution mechanism and full-stack
request-shaped conformance evidence.

The prepared oracle also exposed an evidence-contract mismatch: the live runner applied the
strict one-block one-percent envelope to a composed fifty-block stack, despite the previously
established complete-stack aggregate contract being four percent with cosine at least 0.9995.

## Causal analysis

MiniMax Omni attention has equal query and KV head counts. That geometry allows every head's Q/K
and probability/value matrices to be evaluated as strided batched BF16 GEMMs. A fixed query chunk
keeps score and probability workspace linear in total token count while preserving exact full
attention. Stable softmax remains explicit and bounded, and the existing online GQA path remains
necessary for grouped-head, small-row, and captured execution.

The poor media result is not explained by attention capacity alone. The complete 466-row stack
now agrees with the independent BF16 oracle inside the established aggregate envelope, while a
new one-evaluation 192x192 file is still colored mosaic noise. Geometry, conditioning, iterative
error, and whole-pipeline oracle parity therefore remain separate correctness questions.

## Decision

Keep the mechanism in the generic CUDA Transformer owner. For eager equal-head attention at 128
or more tokens, pack token-major F32 activations to head-major BF16, process 64 query rows at a
time with strided batched GEMM, normalize each score row stably, and unpack the F32 result. Keep
the existing online path as the fail-closed fallback for unsupported geometry and graph capture.

Retain the strict one-block oracle contract and select the already established aggregate
complete-stack contract only when the live runner executes more than one block. Do not raise the
production packed-row cap or claim useful-resolution performance in this checkpoint.

## Implementation

The CUDA backend now resolves the optional strided-batched cuBLAS entry point and admits three
kernel functions for BF16 packing, stable softmax, and output layout restoration. The attention
owner allocates one bounded workspace, performs Q/K and probability/value GEMMs per 64-row chunk,
checks device numeric status, publishes typed tensor-core and temporary-byte facts, and releases
all temporary allocations before return.

A focused 129-token test crosses the activation threshold and three query chunks. It checks
causal and non-causal results against an independent scalar mixed-precision oracle and verifies
execution facts and cleanup. The MiniMax live runner continues to enforce the stricter one-block
contract and applies the established composed-stack envelope only to multi-block execution.

## After

One 466-row MiniMax block completed against the pinned independent BF16 oracle with relative L2
`0.00362431205`, cosine `0.999993434053`, and scaled maximum absolute error `0.00806451589`.

All fifty blocks then completed over the same 466-row request-shaped input with relative L2
`0.0159593791`, cosine `0.9998726482`, and scaled maximum absolute error `0.0233644862`. The run
admitted `64,557,184,000` resident weight bytes, launched 3,100 kernels, reached peak RSS
`63,447,936` KiB, and completed without an OOM.

A bounded one-evaluation 192x192 media run subsequently published a playable 5.166666666-second
AVI with 124 RGB frames at 24 fps and stereo PCM at 32 kHz. Its file SHA-256 is
`29c659e51e027021d468bbb5782abba770bc8435e12850e9821133f5ce68e663`. Visual inspection still
showed structured colored noise, so this is negative quality evidence rather than a quality
advance.

## Evidence

- Two consecutive focused CUDA executions passed the 129-token causal and non-causal scalar
  oracle checks.
- The pinned fifty-block oracle SHA-256 is
  `4a814c5ebde22aafe958c1ac290e7265b7ff8f4260ecb1377ef3462a5c6500ec`; the chunked YVEX output
  SHA-256 is `81d5f2cc904f5229018314c82b9d3790de3174047477a54a86a02efb6e478205`.
- The complete independent oracle took 3 minutes 32 seconds and reached 2,610,864 KiB RSS after
  its shard mappings were made phase-bounded. This is oracle characterization, not YVEX timing.
- The 466-row fifty-block YVEX wall time was about 10 minutes and includes complete artifact
  verification and materialization; no comparable component-only timing was captured.
- Two cgroup runs with zero permitted swap refused safely at CUDA context creation after the
  large host residency was staged. A `64G` memory-high, `100G` memory-max, `12G` swap-max envelope
  allowed the bounded media run to complete with peak RSS `65,616,904` KiB and no process swap
  reported. No process was OOM-killed and failed attempts published no output.
- The no-`nvcc` fail-closed lane, source ownership, C structure, architecture, documentation,
  tracked-artifact, and repository guardrails passed.

## Quantitative delta

| Mechanism or evidence fact | Before | After |
| --- | ---: | ---: |
| equal-head attention implementation at 466 rows | scalar online kernel | chunked BF16 batched GEMM |
| bounded query rows per score tile | not applicable | 64 |
| request-shaped fifty-block oracle coverage | prepared, not executed | 50 of 50 blocks executed |
| 466-row complete-stack relative L2 | unavailable | 0.0159593791 |
| 466-row complete-stack cosine | unavailable | 0.9998726482 |
| production packed-row cap | 2,048 | 2,048 |

These facts are conformance and mechanism characterization. They are not a directly comparable
latency or throughput benchmark.

## Remaining limitations

- The production row cap remains 2,048; source-default 768-short-edge geometry is not admitted.
- The tensor-core path currently applies only to equal query/KV head counts. Qwen's grouped-head
  attention and graph-captured execution retain the existing exact online path.
- The complete-stack oracle permits accumulated BF16 variation under the aggregate contract; it
  does not prove step-by-step latent parity across 49 evaluations.
- The 192x192 result is not recognizable. Prompt fidelity, useful resolution, model quality,
  practical speed, 768p generation, evaluation, benchmark, and release support remain non-claims.
- The exact whole-pipeline divergence must be localized across prompt conditioning, initial
  noise, per-step Transformer output, scheduler updates, and final decoder input before another
  expensive quality run is promoted.

## Why it matters

YVEX now has an exact full-attention path that uses GB10 tensor cores with token-linear temporary
workspace, plus complete request-shaped fifty-block conformance evidence. Capacity work can
proceed without treating an untested row-cap increase or an unrecognizable AVI as model quality.
