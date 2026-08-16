# MiniMax 256 Preview Envelope Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `e623536d664978643799ecb109a6539a8e3255f6` |
| Checkpoint | `32dbc0b0bb2e40cceabefb6882351c3d4c7ce9c6` |
| Subsystem | MiniMax media profile, joint Transformer admission, and Visual VAE conformance |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; independent numerical conformance; bounded runtime qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The native MiniMax media path admitted at most 2,048 packed Omni rows. Its largest exposed
profile was 192x192 for 124 frames, whose request uses 1,766 packed rows with the accepted prompt.
That profile produced a synchronized playable AVI, but sampled frames remained colored mosaic
noise rather than a recognizable scene.

One request-shaped Transformer block had independent conformance at 466 rows, and all fifty
blocks had independent conformance only at smaller geometry. The Visual VAE had bounded decoder
evidence but not a complete 37-latent-frame to 124-frame temporal comparison through the live
production ABI.

## Problem

A 256x256, 124-frame request uses 2,368 video rows, 414 audio rows, and 15 conditioning rows:
2,797 packed rows in total. Merely raising the admission limit would have exposed an unqualified
full Transformer stack. Conversely, retaining the fixed limit prevented a bounded geometry
increase even though the exact full-attention kernel allocates no quadratic score matrix.

The server also recognized hard-coded profile geometry rather than deriving selection from the
registered typed profiles. That would have made every new admitted size require another parsing
special case.

## Causal analysis

The 2,048 maximum was a capacity guard repeated consistently across the internal family ABI,
family recipe, and CUDA joint-Transformer owner; it was not a source-declared MiniMax geometry.
The next exact profile remains within a 4,096-row ceiling even when a request reaches the
256-token conditioning limit. Its memory pressure is dominated by staged source-dtype weights,
not a resident attention-score matrix.

Profile selection already receives the typed profile registry. Matching the canonical
`WIDTHxHEIGHT` spelling generated from each registered entry preserves generic server ownership
and avoids a MiniMax-specific conversation parser.

## Decision

Raise the joint-Transformer bound to 4,096 only together with independent one-block and full
50-block evidence at 2,797 rows. Add a distinct `preview-256` profile rather than changing the
meaning of the existing `preview` alias. Keep source, HD, FHD, 2K, and 4K requests refused.

Extend the existing live owners for dynamic request fixtures and complete Visual VAE temporal
comparison. Do not add a family-specific backend, a second media execution path, or a quality
claim.

## Implementation

The family ABI, graph recipe, and generic CUDA joint-Transformer owner now share the 4,096-row
maximum. The MiniMax live Transformer runner accepts bounded external fixtures with dynamic
video, audio, conditioning, position, index, tag, block, and timestep geometry, while retaining
exact artifact admission and output comparison.

The MiniMax media descriptor registers `preview-256` at 256x256 and 124 frames. The common server
selects an exact registered profile by name or canonical geometry and retains the partial typed
request across turns. Focused tests cover the new geometry and verify that unsupported quality
tiers still fail closed.

The Visual VAE runner now compares the complete reconstructed frame sequence through the current
generic component ABI, including temporal order. No tensor payload, oracle array, or generated
media file is tracked.

## After

The 2,797-row one-block run passed the independent manual BF16 oracle with video relative L2
`0.006350138`, cosine `0.999979839`, and scaled maximum error `0.0126354`; audio relative L2 was
`0.004082210`, cosine `0.999991683`, and scaled maximum error `0.00885813`.

The full 50-block run passed with video relative L2 `0.004822398`, cosine `0.999988385`, and
scaled maximum error `0.0206144`; audio relative L2 was `0.015343347`, cosine `0.999882448`, and
scaled maximum error `0.0290099`. It admitted 66,280,430,144 resident weight bytes, reported
1,226,357,504 device bytes and 8,582 kernel launches, reached 65,505,928 KiB maximum RSS, and used
zero swap. The cgroup enforced a 76 GiB soft threshold, 100 GiB hard limit, and no swap.

The full Visual VAE temporal comparison reconstructed 124 ordered 32x32 RGB frames from 37 latent
frames with maximum absolute error `0.000067681`, relative L2 `0.000046348`, and no temporal swap.
The common server dialogue and family-profile tests admit explicit `preview-256` selection while
leaving the 192x192 `preview` alias unchanged.

## Evidence

- The independent 50-block fixture manifest SHA-256 is
  `dd51d186dc790be67417528e7ea9b7dd73c0172836ea689e69be7a68f3702c18`.
- Its video and audio oracle SHA-256 identities are
  `9f9d542f8530c1ac8d9978bd10dcc962031eb4710a4db80b86da34dce5e59247` and
  `0cc8c1ccb6e4074bcdd0aa7e6c62db12fef3fb8ecd90494eb9a192e4d2714078`.
- The corresponding YVEX video and audio outputs have SHA-256
  `e26a510097056216097559936e034ef8e0132edd6f0c40d3d6567f6d6508c7c7` and
  `e58da65c594564fbf3138e10827e7dd27c53ccf3fbe5b00a9a185dc8f922538c`.
- Focused `minimax_h3`, `runtime_media`, and `server` unit sections passed.
- Documentation architecture, project control, documentation surface, source manifest,
  ownership, repository layout, architecture, and tracked-artifact guards passed.
- The upstream-oracle generation and native YVEX run both exited zero with zero swap.

## Remaining limitations

- A complete `preview-256` prompt-to-AVI run has not yet been qualified. The profile has exact
  Transformer and component admission evidence, not end-to-end media or visual-quality evidence.
- The 192x192 result remains visually unrecognizable. This checkpoint does not prove prompt
  fidelity, model quality, useful video, HD output, or practical generation speed.
- Source-short-edge geometry requires approximately 37,726 packed rows and remains refused. The
  current dense full-attention implementation and 4,096-row guard do not admit it.
- The exact scheduler trajectory and conditioning stack remain active correctness suspects for
  the mosaic output even though their bounded component contracts have passed prior tests.

## Why it matters

YVEX can now represent and numerically execute a materially larger real MiniMax request without
weakening memory or capability admission. The next 256x256 media run tests the unresolved visual
behavior through the same native chat/runtime path; it no longer depends on an unverified
Transformer row increase.
