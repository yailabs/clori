# MiniMax Exact Attention Tiling Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `3398ab0c6f427b1b97cc129b09cbc50a2a145a88` |
| Checkpoint | `5a229408c67a913549969b6f8b36cddb0068d9a0` |
| Subsystem | generic CUDA grouped attention and MiniMax Omni conformance |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; numerical conformance; bounded runtime qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

Generic CUDA grouped attention assigned one block to each query/head pair. Every query traversed
all visible keys twice: once to find the softmax maximum and again to compute probabilities and
the value accumulation. Neighboring queries independently reloaded the same K/V rows. The
implementation avoided a quadratic score allocation, but its repeated traversal was a concrete
barrier to increasing the MiniMax packed-row envelope.

The MiniMax live block runner could compare one selected block over 466 packed rows or all fifty
blocks over three rows. It could not select a complete multi-block stack at request-shaped row
count for the next independent conformance pass.

## Problem

The accepted 192x192, 124-frame media execution used 1,766 packed rows and required roughly 49
minutes for 49 model evaluations. Its output was playable but not visually recognizable. The
source-qualified 768-short-edge geometry requires tens of thousands of packed rows, so merely
raising the 2,048-row admission limit would multiply full-attention work without an executable
capacity argument.

This checkpoint needed to remove the known redundant attention work while retaining exact full
attention and independent numerical evidence. It did not need to guess sparse attention or
promote an unmeasured speed claim.

## Causal analysis

The previous kernel recomputed every Q/K dot product during its second pass and fetched the same
K/V row independently for adjacent queries. Neither repetition was required by the model
contract. A numerically stable online softmax can update its maximum, denominator, and weighted
value accumulator in one traversal. Four warp-owned queries can share each K/V row without
changing masks, head grouping, scale, or output ownership.

The current poor media result is not thereby explained. Conditioning, one-block Omni, the
three-row fifty-block stack, and both VAEs already have independent bounded evidence. Useful
geometry and request-shaped fifty-block conformance remain separate open evidence boundaries.

## Decision

Keep the mechanism in the generic CUDA Transformer owner. Use one warp per query and one
four-query block, cache each query in registers, share one K/V row across the block, and apply an
online stable softmax. Preserve causal and non-causal semantics and continue to avoid a score
matrix.

Extend the existing MiniMax Omni live runner to select one through fifty exact source blocks.
Do not add a family-specific attention backend, sparse-attention guess, larger row claim, or
parallel test owner.

## Implementation

The GQA launch now admits the ceiling of token rows into four-query tiles. The CUDA kernel loads
each shared K/V row once per tile, performs one Q/K traversal, and publishes only after the online
normalizer is complete. Tail warps remain barrier participants, which preserves safe shared-row
reuse when the token count is not divisible by four.

The focused CUDA test crosses a tile boundary with five tokens and a 128-element head. It checks
both causal and non-causal outputs against an independent two-pass scalar softmax. The MiniMax
live runner now derives exact tensor names for a selected 1..50 block stack, seals the selected
weight identity, retains the existing refusal probes, and preserves checked cleanup.

## After

One request-shaped 466-row MiniMax block completed against the pinned BF16 oracle with relative
L2 `0.003273011`, cosine `0.999995331`, and scaled maximum absolute error `0.008064516`. These
remain inside the pre-existing one-block contract. The prior and tiled YVEX outputs differ by
relative L2 `0.000388482`; both have the same 256 maximum absolute error against an oracle whose
maximum magnitude makes that a scaled error of `0.008064516`.

The selected run admitted 1,291,143,680 resident weight bytes, used 205,342,208 device bytes,
launched 35 kernels, completed with peak RSS 1,692,832 KiB, and used no swap. Its execution
identity is `716d9d93575e9dfba18b3960fa546cb47b089783c3e7dca2b9b7b1340b345ea1`.

The SM121 native compilation reports 40 registers, one barrier, and zero spill loads or stores
for the tiled GQA kernel. This is compiler characterization, not an end-to-end performance
result.

## Evidence

- The randomized five-token CUDA test passed for causal and non-causal attention against the
  stable scalar oracle.
- The 466-row one-block live run passed the independent BF16 oracle and completed with zero swap
  and exit status zero.
- The pinned oracle file SHA-256 is
  `58deffd914912995773dd21fbebd58c7b50ec9ae08dd14662a771fe63de959aa`; the tiled YVEX output
  SHA-256 is `eceb0d69be0e557dd90d3e714e54e1728894695ce0a2e6ad0ac5f172ea01e543`.
- All 500 required block-weight roles for the 50-block live fixture reconcile to the 13
  Transformer shards.
- PTX and SM121 native compilation, the no-`nvcc` fail-closed lane, source ownership, structure
  layout, architecture, documentation, and tracked-artifact guards passed.
- The 79.76-second live-run wall time includes complete artifact verification and external I/O.
  No directly comparable before timing was captured, so it is not a performance delta.

## Quantitative delta

| Mechanism or conformance fact | Before | After |
| --- | ---: | ---: |
| Q/K traversals per query | 2 | 1 |
| neighboring queries sharing each K/V load | 1 | 4 |
| 466-row relative L2 versus BF16 oracle | 0.003272757 | 0.003273011 |
| 466-row cosine versus BF16 oracle | 0.999995332 | 0.999995331 |
| SM121 spill loads/stores | not recorded for this checkpoint | 0 / 0 |

The first two rows are algorithmic work counts. They do not establish a wall-time multiplier.

## Remaining limitations

- The production packed-row maximum remains 2,048. Native 768-short-edge MiniMax geometry is
  not admitted by this checkpoint.
- The 466-row fifty-block oracle is prepared but not executed. It requires approximately 66 GB
  of Transformer residency and was deliberately not run while an unrelated DeepSeek server held
  about 98 GB RSS.
- No comparable attention microbenchmark or repeated media timing has yet measured the wall-time
  effect of tiling.
- The 192x192 media result remains visually unrecognizable. This checkpoint does not establish
  prompt fidelity, useful resolution, HD output, model quality, practical speed, evaluation, or
  release support.
- A scalable exact attention implementation and an evidence-bound row-cap increase remain
  necessary before source-qualified geometry can execute on one GB10.

## Why it matters

YVEX removed a known redundant full-attention traversal without weakening MiniMax numerical
evidence, and the next capacity change now has a complete request-shaped block-stack oracle path
instead of relying on three-row conformance.
