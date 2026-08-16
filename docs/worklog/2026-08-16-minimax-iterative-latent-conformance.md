# MiniMax Iterative Latent Conformance Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `d30b5239ccb9e4e9c554e5b2081a1f07f675102c` |
| Checkpoint | `0642a816ae48d42cc3fb073a8d69b8be0ac2bdf0` |
| Subsystem | MiniMax latent iteration and resident Omni Transformer |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | independent numerical conformance; bounded runtime qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The native MiniMax path had independent conformance for complete fifty-block Transformer
evaluations and separate software tests for the paired scheduler. A 49-evaluation 192x192 media
run nevertheless produced synchronized playable colored mosaic rather than a recognizable scene.
The evidence did not distinguish a correct single evaluation followed by an incorrect recurrent
trajectory from an envelope that is simply too far below the released model geometry.

## Problem

Single-evaluation agreement cannot establish that video and audio states remain in the correct
order, use their separate sigma schedules, or feed the next model evaluation without an in-place
update defect. Running another costly media qualification before checking that boundary would
have repeated an unresolved experiment rather than isolating the cause.

The test also needed the exact production initialization stream. Substituting a PyTorch generator
would have changed the starting state and prevented a direct final-latent comparison.

## Causal analysis

The pinned upstream scheduler and the YVEX generic latent owner use the same shifted linear sigma
grid, `t = 1 - sigma`, denoised estimate, and ratio update independently for video and audio.
YVEX initializes the concatenated video-then-audio state through PCG XSH RR 64/32 and Box-Muller.
Reproducing that versioned stream in the external oracle makes every input to both evaluations
identical without making PyTorch part of production.

The recurrent path is therefore testable as one composed boundary: seed initialization, paired
schedule, complete resident Transformer evaluation, state update, second evaluation, and final
publication.

## Decision

Add a dynamic iterative mode to the existing MiniMax Transformer live owner. Compare two complete
fifty-block evaluations over the exact 466-row request shape with a manual PyTorch BF16 oracle,
using seed 42 and the production PCG stream. Keep all arrays and raw evidence outside Git.

Do not relax numerical tolerance, change production scheduling, claim 49-step parity, or treat the
result as media-quality evidence.

## Implementation

The existing live runner now builds the production FL2VA plan and packed layout for explicit
geometry, opens the admitted Transformer artifact through the current component session, executes
the production latent callback for a bounded number of iterations, and compares every final video
and audio value with the external oracle. It records the plan, layout, latent, residency, and
Transformer-chain identities and preserves normal component cleanup.

The external oracle remains untracked. It reproduces the versioned PCG normal stream, constructs
the two shifted sigma grids, executes all fifty source blocks twice, advances each modality with
its own schedule, and transactionally publishes the fixture files and manifest.

## After

The two-evaluation run covered 37 video rows, 414 audio rows, and 15 text rows: 466 packed rows in
total. Video relative L2 was `0.004744679`, cosine was `0.999988780`, and scaled maximum absolute
error was `0.009163232`. Audio relative L2 was `0.004959823`, cosine was `0.999987701`, and scaled
maximum absolute error was `0.008284728`.

YVEX completed 6,313 kernels with 205,342,208 peak device bytes. The run took 3 minutes 13 seconds,
reached 65,318,864 KiB maximum RSS, used zero swap, and exited zero under a 100 GiB hard memory
limit with swap disabled. The independent oracle took 13 minutes 18 seconds, reached 1,810,564
KiB RSS, used about 1.7 GiB GPU memory, used zero swap, and exited zero under a 24 GiB hard limit.

The accepted identities are:

| Evidence | Identity |
| --- | --- |
| Plan | `a6ec3aa32c33397b200020de4c93a13388b7bbe240bf92ec1f0044568767bdf9` |
| Packed layout | `4e23ba2d45b1f7e70740992c3788c26931bf5792129af5bbe86abf84f5a622b4` |
| Latent execution | `e36dd478d890c6d9c6a5979bfb287afb58385596f1a9a1a4027e886bb6cceb36` |
| Transformer chain | `534d9e9e17cf1d5a674877f82250fabd854ce2e4fef0a9e363d3b64e9e42fbf5` |

## Evidence

- The external manifest SHA-256 is
  `c8973779cefd586c658084a7e881c6c26563601371e453c02c54b09e6f9d54ec`.
- External video and audio oracle SHA-256 values are
  `02b37eaff8de0209e861c3cee10d7c083e24378d3c85cf23c1dbad8514d3ffc1` and
  `76f3e971c6254cdd400c1000e77e676928a915f5c72fc1ae77cddd2fcde43cb4`.
- YVEX video and audio output SHA-256 values are
  `116d454f50d5a3d1f1cbcc7b89f51dbb9a2beb6a1c726c0580f0394f0771a770` and
  `83eca0123e6ce07fdf7c83ad9a74b2203e7b2bfb2f14b77769772605e5ca9271`.
- The fixture manifest revalidated every recorded file size and SHA-256 before native execution.
- The live runner compiled without warnings; both executions completed with zero process swap and
  no failed output publication.

## Remaining limitations

- Two evaluations do not prove the complete 49-evaluation trajectory. A later bounded pass owns
  longer-chain evidence if useful-resolution generation still diverges.
- The conformance geometry is 32x32. It proves recurrence and scheduling, not model behavior at
  the released 768-pixel short edge.
- The existing 192x192 media remains visually unrecognizable. This checkpoint establishes no
  prompt fidelity, useful video, model-quality, generation-speed, evaluation, or release claim.
- Released square geometry requires about 21,741 packed rows for the same duration; 1344x768 needs
  about 37,725. Both remain above the admitted 4,096-row capacity and have quadratic dense-attention
  cost despite token-linear temporary storage.

## Why it matters

The first recurrent execution boundary now agrees with an independent source-weight oracle, so
the colored mosaic is no longer reasonably attributable to a one-step scheduler/order defect.
Work can move toward useful-resolution capacity with a materially narrower correctness risk and
without spending hours on another uninstrumented low-resolution media run.
