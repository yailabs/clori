# MiniMax Source-Square Attention Tile Performance

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | performance |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `7e0af31a9e5e26b69eb6f2f7ec419112f7946e47` |
| Checkpoint | `68668dea9107ee74374fffc4017eae5f3049a7a5` |
| Subsystem | generic CUDA dense Transformer attention |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; numerical conformance; bounded component timing |
| Comparability | approximately comparable |
| Publishability | reviewed |

## Before

The equal-head BF16 attention path processed 64 query rows per score tile. A source-square
MiniMax request packs 21,312 video rows, 414 audio rows, and the prompt-conditioning rows into
one exact full-attention sequence. On the 21,741-row conformance fixture, one Omni block used
1,140 reported kernel launches and completed in 1 minute 46.48 seconds.

The full 768x768 request had already stayed inside its bounded cgroup without swap or an OOM,
but its dense attention advanced too slowly for a practical qualification run. That interrupted
run published no partial media because output publication remained atomic.

## Problem

The 64-row tile divided every large equal-head attention operation into hundreds of small
batched matrix products. At source-square geometry, launch frequency and underfilled GEMMs
dominated enough of each Omni block that a 49-evaluation, 50-block qualification projected to
several hours even though the workload fit in memory.

## Causal analysis

The MiniMax Omni Transformer has equal query and KV head counts, so it consumes the generic
strided-batched BF16 path. Increasing only the query tile leaves attention semantics, score
normalization, causal policy, source tensors, and logical output unchanged. It trades bounded
temporary score/probability workspace for fewer and larger GEMMs.

Sequential 64-row and 256-row executions used the same artifact, fixture, runner, deterministic
CUDA workspace setting, GB10, and cgroup limits. Their file-system input totals differed, so the
wall-time delta is approximately rather than directly comparable. Kernel-launch counts and
byte-identical outputs independently establish the mechanism and semantic preservation.

## Decision

Use a 256-row query tile for eager equal-head BF16 attention. Keep the existing online path for
grouped-head attention, small sequences, and captured execution. Do not enlarge the tile again
without another bounded memory and numerical comparison; the next larger tile would consume
more unified-memory headroom during the already memory-intensive source-square request.

## Implementation

The generic CUDA Transformer owner now caps one BF16 score tile at 256 query rows. Its focused
test crosses the new boundary with 257 tokens, verifies two-chunk causal and non-causal parity,
and checks the reduced launch facts. No family ABI, runtime policy, model geometry, or public API
changed.

## After

The 21,741-row one-block run completed in 1 minute 25.94 seconds with 375 reported kernel
launches. Video and audio outputs were byte-identical to the 64-row run. The 256-row run retained
the accepted one-block conformance: video relative L2 `0.0060228159` with cosine
`0.999981866929`, and audio relative L2 `0.00375755049` with cosine `0.99999295103`.

Both runs completed with zero swap and about 5.34 GB peak host RSS reported by the process. The
runner reported the same `9,524,132,608` device bytes for both executions. These values
characterize the bounded component test; they are not full-pipeline peak-residency measurements.

## Evidence

- The focused CUDA qtype test passed twice after rebuilding the 256-row implementation.
- The complete CUDA unit suite passed, including graph, tensor, operations, parity, qtype,
  sampling, and CUDA materialization.
- The no-`nvcc` build refused CUDA execution through its admitted fail-closed lane.
- Source ownership, repository layout, architecture, and generated-source guardrails passed.
- The video SHA-256 for both tile sizes was
  `59e1135447ca15e66d716e4295b1ae805de950b12fa420a50c6f48635a79f37f`.
- The audio SHA-256 for both tile sizes was
  `b67f44b1cb0e66b87119283c47f4ae4e08af7044aa900e8f512b1ac7df7b3e08`.

## Quantitative delta

| 21,741-row one-block fact | 64-row tile | 256-row tile | Delta |
| --- | ---: | ---: | ---: |
| wall time | 106.48 s | 85.94 s | 19.3% lower |
| reported kernel launches | 1,140 | 375 | 67.1% lower |
| output bytes | reference | byte-identical | unchanged |
| process swap | 0 | 0 | unchanged |

## Remaining limitations

- The timing includes artifact verification, materialization, host I/O, and one complete block;
  it is not an isolated attention-kernel benchmark.
- The path remains quadratic full attention. A larger tile reduces launch overhead but does not
  change asymptotic source-square compute or memory movement.
- This evidence does not establish useful video quality, full-run latency, 768p practicality,
  model evaluation, or release readiness.
- Qwen grouped-head attention and captured execution continue to use their existing paths and
  receive no performance claim from this change.

## Why it matters

Source-square MiniMax execution now performs the same admitted dense attention with far fewer
launches and byte-identical component output while retaining bounded memory behavior.
