# MiniMax Source-Scale Correctness Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-18 |
| Type | repair |
| Milestone | `R012.MINIMAX.H3.SOURCE_SCALE.CORRECTNESS.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `516738bd8a0fd4498aa2d92f87f5e622d008492f` |
| Checkpoint | `9b2b5970648c6ae99c81a3d2b066ad508f3982d2` |
| Subsystem | MiniMax tensor-role lowering, generic CUDA transformer execution, and source-scale media qualification |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | source-derived tensor contracts; focused CPU/CUDA conformance; sanitizers; full native 768p media execution; independent container decode; human visual qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

YVEX could complete the full native 768x768 MiniMax-H3 trajectory, decode both media domains,
and atomically publish a playable synchronized AVI without swapping. The resulting frames were
nevertheless a regular colored mosaic with an observed 16-pixel spatial period. The failure
survived the transition from smaller preview geometry to the released square geometry, so it
could not truthfully be classified as a low-resolution model-quality limitation.

The retained failure specimen has SHA-256
`cf54a70f6854c33d775ef4a624f60e6ecafb54f0302d4277b68943f31bfe48f0`. It remains external
evidence and is not tracked or overwritten.

## Problem

A structurally valid container was hiding corrupted model math upstream of publication. Small
fixtures did not exercise the raw checkpoint layouts and precision boundaries strongly enough
to reveal whether the packed trajectory, transformer, Visual VAE, or writer first diverged.
The source-square path also needed bounded workspace accounting: admitting an underestimated
attention allocation would trade one visual defect for an OOM risk.

## Causal analysis

The first proven divergent boundary was the Omni-Transformer interpretation of raw source
tensors and its attention precision contract. Raw checkpoint QKV tensors store Q, K, and V
interleaved per head rather than as three flat contiguous projections. Raw `mlp.fc1` stores the
SwiGLU gate before the up projection. The existing path interpreted both orders incorrectly.
Attention also rounded V and the probability-times-value result too early relative to the
source execution, and final AdaLN introduced an extra pre-bias narrowing.

These errors corrupted the repeated latent trajectory before the Visual VAE. Correcting them
produced a coherent, recognizable scene through the unchanged native VAE and media publication
path. Independent AVI decode preserved the exact dimensions, frame count, frame ordering,
audio format, and end-of-stream behavior. This acquits the tile compositor and writer for the
qualified request; it does not claim bit identity with an external framework.

## Decision

Keep checkpoint-specific ordering in the MiniMax graph recipe and implement reusable precision
and workspace mechanics in the generic CUDA transformer owner. Represent QKV and SwiGLU source
orders as typed recipe facts instead of inferring them from names or special-casing 768 pixels.
Preserve source-precision attention until the admitted output narrowing boundary, and account
for the allocator's actual alignment rather than a packed-byte approximation.

A 224-row attention query chunk was evaluated and rejected. It reduced the theoretical GEMM
count but made the 192x192 result severely chromatic and deformed, ran 8.15 seconds slower than
the 64-row control, and therefore did not satisfy correctness or performance admission. The
canonical query chunk remains 64 rows.

## Implementation

The MiniMax recipe now declares per-head QKV interleaving and gate-then-up SwiGLU ordering for
the raw source roles. Main and token-refiner projections use the same checked split contract.
Final AdaLN performs one narrowing after bias, and the final output head remains F32.

The generic CUDA GQA path now retains Q and K as BF16 while keeping V, probabilities, and the
probability-times-value accumulation in F32 until the caller's admitted output boundary. Its
workspace planner includes every 256-byte attachment alignment and reports the actual attached
high-water mark. A source-square 21,741-row, 56-head, 128-wide plan is checked at exactly
2,493,431,812 bytes, below the 4 GiB request budget.

Focused observers retain bounded block checkpoints without turning normal execution into an
unbounded tensor dump. Unit coverage rejects ordering, dtype, and unaligned-workspace
regressions at their semantic owners.

## After

The repaired native path generated 124 coherent 768x768 frames of a clearly recognizable red
fox walking through a snowy pine forest, accompanied by non-silent stereo audio. Five sampled
points across the timeline preserve the subject, natural color regions, forest background, and
motion without the previous checkerboard, spatial wrapping, patch permutation, channel
scrambling, or reconstruction seams.

The complete 49-evaluation, 50-block execution used the four admitted MiniMax artifacts and
published through the ordinary YVEX media transaction. It exited successfully with no OOM and
zero swap. The resulting file has SHA-256
`85d90a217b00fa785c68ca969700ecaeda0503127b64b59eef960b2273851759` and remains external
identity-bound evidence.

## Evidence

| Scope | Result | Evidence |
| --- | --- | --- |
| Exact 768x768 native generation | PASS | execution `2a1175a7a9e1a36e1f3e81347ff4f5aecb2b9e6a04b355cab60a725b33743488` |
| Media publication | PASS | publication `4b64f76287d30270786e7b947574f2f6238b3fa97801bc58f5e4d3a275110038` |
| Independent AVI decode to EOS | PASS | 768x768, 124 frames, 24 fps |
| Audio inspection | PASS | PCM S16LE stereo, 32 kHz, 165,333 samples per channel, non-silent |
| Human timeline inspection | PASS | recognizable stable fox and forest; no regular mosaic |
| CUDA native conformance | PASS | focused canonical QA |
| CUDA no-`nvcc` refusal | PASS | focused canonical QA |
| ASan/LeakSanitizer and UBSan | PASS | quantization and runtime sanitizer lanes |
| Structural ownership/layout/architecture | PASS | focused canonical QA |

The post-merge canonical changed-plan run has evidence identity
`8449a9f3273ccdac9e62cdf5218297d4cb24198fd05889b25211b2af7bf80d81`: 101 tests passed,
eight were blocked by absent configured external assets, and none failed, skipped, or errored.
Blocked rows include the independent MiniMax text, latent, transformer, VAE, and performance
asset contracts. They remain `BLOCKED`; the explicit source-square production run is separate
live evidence and does not rewrite those registry results as passes.

The source-scale writer produced 220,082,144 bytes. Audio duration was 5.16665625 seconds and
video duration was 5.166666666667 seconds, a 10.4167-microsecond skew. An independent decoder
read every frame and both audio channels through end of stream.

## Quantitative delta

| Fact | Before | After |
| --- | --- | --- |
| Source-scale visual result | regular 16-pixel mosaic | recognizable coherent fox and snowy forest |
| Geometry and trajectory | 768x768, 124 frames, 49 evaluations, 50 blocks | unchanged |
| Wall time | 2:30:28 on the prior semantics | 4:03:39 on the source-precision repair |
| CUDA kernel launches | 942,556 | 3,078,123 |
| Peak device bytes | 12,960,345,604 | 12,035,250,948 |
| Peak reported reusable workspace | 1,485,742,080 | 1,485,742,080 |
| Maximum resident set | 67,176,200 KiB | 67,160,084 KiB |
| Swap / OOM | 0 / 0 | 0 / 0 |

The timings are characterization only because the numerical contracts differ. The repaired
path is materially slower and is not an admitted performance improvement.

## Remaining limitations

- Four hours for about 5.17 seconds of media is unacceptable product latency. The 3,078,123
  kernel launches and repeated full attention are explicit PASS 4 optimization debt, not a
  reason to weaken the repaired numerical contract.
- The canonical live oracle rows still require their configured external fixtures. Their QA
  state remains `BLOCKED`, even though the exact production request completed successfully.
- This checkpoint does not establish bit-identical parity with PyTorch, Diffusers, or the
  upstream implementation; those remain independent oracle surfaces only.
- It does not qualify alternate aspect ratios, containers beyond the admitted AVI path,
  Ref2VA, Context-IR, Regenerate-2K, model-quality benchmarks, or release support.

## Why it matters

The repair turns a successful-but-corrupted GPU run into a truthful source-scale media result
while preserving native execution and bounded memory. It also keeps the remaining problem in
the correct category: generation latency is now measurable optimization debt rather than a
visual correctness ambiguity.

## Communication projections

### Short update

YVEX now completes the native MiniMax-H3 768x768 trajectory on one GB10 and publishes a
recognizable synchronized fox video without swap or OOM. The raw checkpoint ordering and
attention precision contract were the visual-corruption boundary; generation still takes four
hours and is the next optimization target.

### Quoteable technical facts

- “The same 768x768, 124-frame request changed from a regular 16-pixel mosaic to a coherent
  recognizable scene after repairing source tensor order and transformer precision.”
- “The corrected run completed 49 evaluations across 50 Omni blocks with zero swap and zero
  OOM.”
- “Correctness is admitted; four-hour generation latency is not.”
