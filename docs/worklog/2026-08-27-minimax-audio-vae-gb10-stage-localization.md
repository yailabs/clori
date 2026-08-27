# MiniMax Audio VAE GB10 Stage Localization

| Field | Value |
| --- | --- |
| Date | 2026-08-27 |
| Type | performance |
| Milestone | `MINIMAX.H3.AUDIO.VAE.GB10.STAGE.LOCALIZATION.0` |
| Branch | `models1` |
| Baseline | `e15c7ddbd2f9b5a914523f2b8dbe3c8963236d8f` |
| Checkpoint | `c684a28da361bd698cb683576090ee031f43f9b2` |
| Subsystem | MiniMax-H3 Audio VAE live evidence over generic component and CUDA execution |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | independent bounded oracle, request-shaped repeat, Nsight Systems CUDA activity |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

One current hosted `interactive-preview-v1` generation attributed 141.877 seconds, 638 exposed
launches, and 36.61% of complete request wall time to the Audio VAE phase. That interval included
the component callback as a whole. It did not distinguish byte authentication, component
preparation, CUDA attachment, decoder kernels, synchronization, output transfer, or cleanup.

The dedicated Audio VAE live owner already admitted the exact artifact and checked a one-step CPU
decode against an independent reference at maximum absolute error `1e-5`. It did not accept a CUDA
backend or request-shaped latent geometry, so another complete media request would have been the
only way to repeat the 142-second observation.

## Problem

The phase duration alone could support two incompatible performance theories. The decoder might be
genuinely compute-bound, in which case kernel and operation selection own the next boundary. Or the
same interval might mostly measure artifact hashing, page migration, component preparation, or host
synchronization, in which case changing decoder math would optimize the wrong owner.

No generic runtime, server, scheduler, protocol, residency, component, or CUDA instrumentation was
authorized for this localization. Evidence therefore had to use the existing MiniMax component API
and external CUDA activity capture.

## Decision

Extend only `tests/live/minimax_h3_audio.c` with two diagnostic modes. One converts an independently
retained packed Audio latent into the exact decoder layout using the current family normalization
facts. The other calls the production Audio VAE CUDA component callback with explicit batch and
latent-step geometry and records the typed execution result. The existing CPU command and its
independent oracle remain unchanged.

Use the retained two-step Audio latent rather than regenerate Qwen and Omni state. Its packed
identity is `ca9b5616ab217b8253effc153db7fc7b36739b9ef7528afaead2b2f1d3e07306`.
The source-faithful unpacked decoder fixture is 52,992 bytes with identity
`f51a0eb06f333ed6bb6ea2478401d3b3f65abd1e5c24d4a9c435c37069414c44`.
This is exact retained numerical evidence, but it is not the latent from the earlier eclipse run.

Profile one complete request-shaped decoder with Nsight Systems 2025.3.2, then repeat it without a
profiler. Map kernels to stages from the admitted deterministic recipe and launch order instead of
adding stage callbacks to generic execution.

## Implementation

The MiniMax live owner gained bounded dynamic fixture I/O, source-faithful packed-latent unpacking,
explicit request geometry, and direct CUDA component execution. It records callback wall time and
the existing typed execution facts. Invalid extents, malformed fixture sizes, unavailable CUDA,
and insufficient output contracts fail without publishing output. Production source and generic
interfaces are unchanged.

## Current source geometry

The admitted Audio VAE consumes 32 latent channels and emits one PCM channel per batch item. The
media pipeline uses batch two for stereo. The request-shaped input is `[2, 32, 207]`; the decoder
expands each channel by 800 and emits `[2, 1, 165600]`, or 331,200 F32 values. Publication of the
124-frame, 24 FPS media request retains 165,333 samples per channel, so it trims 267 raw decoder
samples per channel to the media duration.

The decoder first projects 32 channels to 2,048, applies a width-1,024 pre-convolution, and then
executes seven source-derived upsampling stages with rates `[5, 5, 2, 2, 2, 2, 2]`. Each stage owns
three residual blocks, each block owns three two-convolution layers, and each activation uses the
source alias-free up/down filter pair. The terminal F32 convolution projects eight channels to one.

| Stage | Input shape | Output shape | Rate | Upsample kernel |
| ---: | --- | --- | ---: | ---: |
| 0 | `[2,1024,207]` | `[2,512,1035]` | 5 | 9 |
| 1 | `[2,512,1035]` | `[2,256,5175]` | 5 | 9 |
| 2 | `[2,256,5175]` | `[2,128,10350]` | 2 | 4 |
| 3 | `[2,128,10350]` | `[2,64,20700]` | 2 | 4 |
| 4 | `[2,64,20700]` | `[2,32,41400]` | 2 | 4 |
| 5 | `[2,32,41400]` | `[2,16,82800]` | 2 | 4 |
| 6 | `[2,16,82800]` | `[2,8,165600]` | 2 | 4 |

## Measurement snapshot

The profiler executable SHA-256 is
`f5be6fd456c47be26583aa43fc5297763e6d46d40dd2417af3564e1b3935e661`, with ELF build ID
`12aebb1fcae07feeed90d5b192f110e33310a53f`. It was built from baseline `e15c7dd...` plus the
single live-test source delta
`1278cec861dc16f1a98bc3da02a54e4ad34455d89cbc7e5cdce25d1f2d4a72db`; that content is the test
checkpoint recorded above. The source and delta were unchanged before and after both measured
decodes.

The exact Audio VAE artifact identity is
`52a10c9f6f6e3b9b81569a95329f503fcb3cbddb224d12bf7851b4929d02e1c1`; the component identity is
`be921beb8581b44624aaad452f30f77f1e204159ae8fe11da455d5208dc4e62b`. The artifact contains 1,087
tensors, 605,306,340 payload bytes, and 605,401,984 file bytes.

The Nsight report identity is
`ceba3b70d2c4d4fbd436349530a69727819e6629feae03653cbd1e6bd7b610ce`; its SQLite export identity is
`ee34b714bdb680b0c006fc5b4e67173d6e69e4082d3a3c368907e6c0d9dc030e`. Both are external
operator evidence and are not tracked in Git.

## After

The profiled component callback completed in 141.104438 seconds. Decoder kernels occupied
138.811345 seconds, or 98.375% of callback wall. Seven transposed Conv1D upsample kernels alone
occupied 138.143235 seconds, or 97.901% of callback wall and 99.559% of all Conv1D kernel time.

| Decoder boundary | GPU kernel time | Callback wall | Kernels | D2D copies |
| --- | ---: | ---: | ---: | ---: |
| Input projection | 0.000142 s | 0.000% | 1 | 0 |
| Pre-decoder | 0.017957 s | 0.013% | 2 | 0 |
| Stage 0 | 2.723980 s | 1.930% | 86 | 4 |
| Stage 1 | 15.973955 s | 11.321% | 86 | 4 |
| Stage 2 | 23.948198 s | 16.972% | 86 | 4 |
| Stage 3 | 24.023285 s | 17.025% | 86 | 4 |
| Stage 4 | 24.054370 s | 17.047% | 86 | 4 |
| Stage 5 | 24.037421 s | 17.035% | 86 | 4 |
| Stage 6 | 24.031704 s | 17.031% | 86 | 4 |
| Final projection and clamp | 0.000333 s | 0.000% | 5 | 0 |

The five longest concrete operations were the stage 4, stage 6, stage 5, stage 3, and stage 2 F32
transposed Conv1D upsample kernels at 24.020815, 24.017977, 24.017164, 23.962933, and 23.834013
seconds. Each launched 10,350 blocks of 256 threads. Stage 1 used the same grid and took 15.751729
seconds; stage 0 used 4,140 blocks and took 2.538604 seconds.

CUDA memory activity does not explain the phase. The decoder uploaded 52,992 input bytes in 2.720
microseconds and downloaded 1,324,800 output bytes in 27.264 microseconds. Its 28 D2D copies moved
271,319,040 bytes in 2.039 milliseconds. Preparation performed 41 BF16-round kernels in 9.088
milliseconds and 41 D2D copies totaling 1,375,731,712 bytes in 11.614 milliseconds. All captured
CUDA memcpy and memset device time totaled approximately 27.2 milliseconds.

`cuCtxSynchronize` accumulated 138.832012 seconds across 973 calls because the current generic
decoder synchronizes after individual operations. That duration overlaps the GPU work and is host
waiting for the long kernels; it is not an additional 138 seconds of overhead. Kernel-launch API
time was 2.555 milliseconds. The remaining approximately 2.293 seconds between decoder-kernel sum
and callback wall includes component admission, context/session setup, allocation, module loading,
other device activity, host gaps, and cleanup. This capture cannot split every one of those host
owners, but their aggregate is too small to own the 142-second observation.

## Causal analysis

The generic CUDA transposed Conv1D kernel assigns one thread to each output value. For every output
position it loops over every input channel, every input position, and every kernel position, then
tests whether the projected input position equals that output position. It therefore scans the
complete temporal input for each temporal output instead of solving the small set of contributing
positions.

At request geometry, stage 0 executes about 2.022 trillion iterations of that inner search, stage 1
executes about 12.637 trillion, and each of stages 2 through 6 executes about 14.041 trillion. The
constant-looking late-stage times follow directly from the source expansion schedule: temporal
length doubles while channel width halves, leaving the current exhaustive search population almost
constant. The measured stage plateau is therefore causal evidence for the generic transposed
convolution mechanism, not merely correlation with a large Audio VAE phase.

The MiniMax recipe is source-derived and the same generic kernel executes its admitted operation.
No redundant family operation, wrong rate, wrong channel width, extra attention block, or alternate
decoder was observed. The dominant owner is generic CUDA convolution, not MiniMax composition,
artifact I/O, residency, or media publication.

## Evidence

The unchanged CPU oracle produced maximum absolute error `3.87430191e-7`, below `1e-5`, with output
identity `fb1408a27e08d53863803bc14c373f0554660fe65496743ddd3a500b470a2210`. The CUDA one-step decode
produced maximum absolute error `3.7252903e-7`, also below `1e-5`.

Two request-shaped CUDA executions took 141.104438 and 141.257596 seconds. Both sealed execution
identity `565795cb2adfcb444c291b54c30410e7843d2a8f303cd0260d28ef5e6166f895` and produced byte-identical
PCM identity `87d4cb46904afce2c7381823894812d63b42096c259bdff126f262da826cb23d`.
The typed result reported 638 logical launches, 52,992 H2D bytes, 1,324,800 D2H bytes, and
84,787,200 peak activation/device bytes. Nsight resolves the 638 logical operations into 610
decoder kernels and 28 D2D tensor copies; its additional 41 kernels belong to preparation.

An isolated no-NVCC build compiled the extended live owner and refused CUDA execution with
`kernel-bundle-absent` without publishing output. Repository layout, ownership, architecture,
natural-code, tracked-weight, and diff checks passed.

## Remaining limitations

External CUDA activity separates decoder compute from device movement, but the callback has no
test-owned markers around host artifact admission, component-session open, and cleanup. Their
combined remainder is bounded to approximately 2.293 seconds; individual values remain not
currently observable without changing a generic owner. The profile uses an exact retained
two-step latent with the same request geometry, not the earlier eclipse latent. This is one GB10
characterization, not a release benchmark or speedup baseline.

## Next boundary

The evidence-derived candidate is a generic exact transposed Conv1D implementation that enumerates
only contributing input/kernel coordinates while preserving the accepted F32 accumulation and
publication order. It belongs to the common CUDA convolution owner and requires Main coordination.
It must not be implemented as a MiniMax-specific kernel. The independent one-step oracle and
request-shaped byte-repeat fixture are the minimum numerical guards for that later optimization.

No optimization, precision change, layout change, fusion, algorithm selection, workspace policy,
runtime lifecycle change, server change, scheduler change, protocol change, or new performance
claim was made in this localization.

## Why it matters

The 142-second Audio VAE interval is now explained by measured compute rather than attributed to a
broad media phase. Seven generic transposed-convolution kernels own almost all of it; artifact and
memory movement are secondary by orders of magnitude. The next performance work can therefore be
reviewed at the correct shared owner with exact numerical evidence instead of changing MiniMax
composition or runtime lifecycle speculatively.

```text
progression_decision: proceed
downstream_safe: true for a separately authorized, Main-coordinated generic CUDA optimization boundary
downstream_consumer: review and design of exact generic transposed Conv1D execution
gate blockers: none for localization
boundary incompleteness: none for localization; no optimization was authorized
evidence gaps: host-side component admission and cleanup are only aggregate-bounded, not individually timed
deferred depth: implementation and qualification of the generic convolution optimization
optimization debt: output-centric transposed Conv1D performs exhaustive temporal scans
generalization debt: a future generic repair requires non-MiniMax convolution fixtures as well as Audio VAE evidence
external blockers: none
required repairs: none in this measurement boundary
higher-capability non-claims: no speedup, release benchmark, visual-quality, caching, streaming, quantization, or release claim
```
