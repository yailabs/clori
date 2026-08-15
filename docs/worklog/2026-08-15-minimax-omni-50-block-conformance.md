# MiniMax Omni 50-Block Conformance Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-15 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `a272022f0c3d95e342bd7526a751e76b1abf31a0` |
| Checkpoint | `861cff654d889ed221ba8189b022705bee859625` |
| Subsystem | MiniMax graph recipe and generic CUDA joint Transformer conformance |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | numerical conformance; artifact admission; runtime residency and cleanup evidence |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

Request-shaped one-block conformance covered 466 packed rows after the AdaLN rounding repair,
but complete-stack evidence remained limited to an older three-row fixture. The artifact-mode
live runner could not requalify all fifty blocks in the current branch because it passed no
joint Transformer recipe to the component session.

The native two-evaluation media path was executable and memory-bounded, but its 32x32 output
still had no recognizable prompt structure. Without current full-stack conformance, spending
more evaluations on visual qualification would not distinguish an insufficient schedule from a
Transformer composition defect.

## Problem

The full-artifact live runner admitted and materialized the Transformer, then production refused
the request with `a resident component and bounded Transformer request are required`. This was a
truthful refusal: the request's recipe pointer was absent. The selected-weight proof path already
bound the canonical recipe, while the runtime-component path did not.

## Causal analysis

The runner's artifact path called the graph component executor with a zero-initialized request.
The graph owner requires exact pointer identity with its admitted Omni recipe before binding any
weights. Supplying the graph-owned recipe is therefore required test setup, not a relaxation of
production validation.

Once that edge was restored, the same 66,280,430,144-byte Transformer artifact executed all
fifty blocks and compared successfully with the pinned independent PyTorch BF16 oracle. No
production change was required to obtain the result.

## Decision

Have the existing artifact live runner obtain the recipe from the MiniMax graph registration and
fail explicitly if it is unavailable. Preserve the production pointer-identity refusal and use
the existing runtime component session; do not add a test-only production bypass or duplicate
the recipe in the runner.

## Implementation

The artifact execution lane now binds `graph->omni_recipe()` before component admission and
reports the same unsupported state as the selected-weight lane if registration is incomplete.
The runner otherwise retains the existing complete-artifact admission, staged host residency,
CUDA execution, oracle comparison, and checked cleanup path.

## After

One three-row envelope containing one video, one text-conditioning, and one audio row now passes
the independent full-stack oracle through all fifty Omni blocks. The run admitted
66,280,430,144 resident bytes, used 2,132,224 device bytes, launched 1,772 kernels, and completed
without swap or OOM.

The execution identity was
`bef1d060ab150cbfb23e0262bc8139e29b67d88a8210ea07673b8d3d41f6b77c`; the residency identity
was `8fabd88629b671abf1ca8dd3d659d7ec9f173996039082d7354ff73371ef7308`.

## Evidence

- Video relative L2 was 0.008826793 with cosine 0.999964158 and scaled maximum absolute error
  0.012824830.
- Audio relative L2 was 0.018843032 with cosine 0.999823405 and scaled maximum absolute error
  0.021691594.
- The run completed in 184.74 seconds with peak RSS 65,251,116 KiB, no swap, and exit status
  zero under an 88 GiB cgroup memory ceiling.
- Source-membership, structure ownership, layout, and architecture guards passed.
- `git diff --check` passed and tracked artifact scans found no new model payloads.

## Quantitative delta

| Full-stack fact | Before | After |
| --- | --- | ---: |
| Current artifact-mode 50-block execution | refused before CUDA because recipe was absent | accepted |
| Video relative L2 versus independent oracle | unavailable | 0.008826793 |
| Audio relative L2 versus independent oracle | unavailable | 0.018843032 |
| Completed Omni blocks | 0 | 50 |

## Remaining limitations

- The 50-block fixture has three packed rows. Full-stack conformance over the 466-row smoke
  geometry remains an evidence gap; the same geometry has one-block conformance.
- The independent fixture is synthetic and does not capture a full-pipeline intermediate.
- A two-evaluation 32x32 media run remains visually unrecognizable. More schedule evaluations
  and a qualified geometry still require real execution and inspection.
- No prompt fidelity, useful-scene, HD, performance, model-quality, evaluation, or release claim
  follows from this numerical checkpoint.

## Why it matters

All fifty native Omni blocks now have current independent conformance evidence, so the next
visual-quality experiment can focus on schedule depth and geometry instead of an untested full
Transformer stack.
