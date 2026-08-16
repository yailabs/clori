# MiniMax Reusable Source-Scale Attention Workspace

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | performance |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `7b449c9642d25b37bca7cd81667102e85f0fcca8` |
| Checkpoint | `0540ab2c36f9443e3289f484a6427d0be01d5cb5` |
| Subsystem | CUDA grouped-query attention and runtime component residency |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; numerical conformance; runtime qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

Chunked BF16 grouped-query attention allocated and released its packed Q/K/V, output, score,
probability, and status buffers for every operation. A complete fifty-block Transformer
evaluation therefore repeated the raw CUDA allocation lifecycle fifty times, and a 49-evaluation
media trajectory could repeat it 2,450 times. The reported device-byte peak covered persistent
joint-block activations but omitted the simultaneous attention scratch.

## Problem

At 21,741 packed rows, the source-square attention scratch is 3,428,468,740 bytes. Repeatedly
allocating that region adds avoidable unified-memory lifecycle pressure to the dominant model
phase, while omitting it from the published peak understates admission requirements. The
optimization had to preserve the exact BF16 execution path and independent numerical contract.

## Causal analysis

Every MiniMax Transformer block uses identical admitted attention geometry. The scratch lifetime
ends at each attention operation, so one arena owned by the component session can be reset and
reused across blocks and latent evaluations without sharing mutable state between sessions. The
required size is mechanically determined by token count, head geometry, query tile, and the exact
temporary dtypes.

This checkpoint characterizes memory lifecycle rather than a directly comparable speedup. The
available live runs include full component admission and filesystem effects, so they do not
support a latency delta claim.

## Decision

Let the generic CUDA Transformer owner compute the exact workspace extent and let the existing
runtime component session own one same-backend arena until close. Reset the arena at every GQA
boundary, fail closed if its geometry changes, and include peak temporary bytes in joint-block
device reporting. Retain the source-faithful BF16 Transformer; the rejected selective-Q8 candidate
does not enter runtime execution.

## Implementation

The CUDA GQA owner now exposes a checked internal workspace-size calculation for its chunked BLAS
path and reports zero for the allocation-free online path. Runtime component residency can attach
one exact I8 arena, reuse it for consecutive operations, and detach and release it before backend
close. MiniMax requests that generic arena from its admitted token and head geometry before
binding Transformer weights. Joint-block facts add the maximum simultaneous scratch to persistent
activation bytes.

Focused CUDA tests execute causal and non-causal attention consecutively on the same exact-size
arena. The live MiniMax request fixture now enters the production component session even for a
single source-square block.

## After

A 5,757-row fifty-block evaluation reused one 907,855,876-byte arena and retained the accepted
oracle result. A 21,741-row source-square block reused one 3,428,468,740-byte arena and retained
the accepted source-square oracle result. Both new output pairs are byte-identical to the previous
q256 implementation outputs.

Corrected peak device facts are 3,430,740,740 bytes at 5,757 rows and 12,952,601,348 bytes at
21,741 rows. The source-square live process completed under bounded memory with swap disabled and
released all CUDA residency.

## Evidence

- CUDA unit tests cover exact workspace sizing, online-path zero sizing, consecutive causal and
  non-causal reuse, numerical reference comparison, detach, and release.
- The 5,757-row fifty-block manual BF16 oracle comparison retained video relative L2
  `0.00478590711` and cosine `0.999988563893`, plus audio relative L2 `0.0136364694` and cosine
  `0.999907029737`.
- The 21,741-row one-block manual BF16 oracle comparison retained video relative L2
  `0.0060228159` and cosine `0.999981866929`, plus audio relative L2 `0.00375755049` and cosine
  `0.99999295103`.
- Source-square output SHA-256 values remained
  `59e1135447ca15e66d716e4295b1ae805de950b12fa420a50c6f48635a79f37f` for video and
  `b67f44b1cb0e66b87119283c47f4ae4e08af7044aa900e8f512b1ac7df7b3e08` for audio.
- Source ownership, layout, architecture, no-`nvcc`, ASan/LeakSanitizer, UBSan, and repository
  guardrails passed.
- No model weight or runtime evidence file entered Git.

## Quantitative delta

| Geometry | Persistent joint bytes | Reused GQA scratch | Corrected peak device bytes |
| --- | ---: | ---: | ---: |
| 5,757 rows | 2,522,884,864 | 907,855,876 | 3,430,740,740 |
| 21,741 rows | 9,524,132,608 | 3,428,468,740 | 12,952,601,348 |

These are exact geometry-derived and runtime-reported memory facts. They are not a latency or
throughput comparison.

## Remaining limitations

- A complete 21,741-row, 49-evaluation media trajectory has not yet been qualified.
- This checkpoint does not prove recognizable prompt-conditioned video or practical generation
  speed.
- Exact attention remains quadratic and is still the dominant source-scale compute cost.
- Visual VAE source-scale output quality and the complete media lifecycle require their own live
  qualification.
- The rejected selective-Q8 physical candidate remains outside runtime admission.

## Why it matters

Source-scale MiniMax attention now has one explicit, reusable, correctly accounted memory
lifecycle instead of thousands of opaque raw allocation transitions, without weakening numerical
conformance or session ownership.
