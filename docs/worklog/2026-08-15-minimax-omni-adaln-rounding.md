# MiniMax Omni AdaLN Rounding Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-15 |
| Type | repair |
| Milestone | `not-applicable` |
| Branch | `feature/minimax-h3` |
| Baseline | `aa8019905c08d228b1a54b808de9a9cd51f1c425` |
| Checkpoint | `efa885a7daaeecd1322e6f36e6cc526b1c2040b4` |
| Subsystem | generic CUDA joint Transformer backend and MiniMax live conformance |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; numerical conformance; sanitizer evidence; CUDA refusal and cleanup evidence |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The native MiniMax path had already produced synchronized playable media through the persistent
server and chat surface. Its accepted live qualification used a 32x32, 124-frame request with
two scheduler grid points, which means one Transformer evaluation. Visual inspection found
colored block structure rather than a recognizable scene, so that execution proof established
wiring and publication but not numerical quality.

The independent Omni block oracle covered three packed rows and one timestep. That fixture
proved the basic block composition but did not exercise the 466-row text, audio, and video
envelope used by the accepted smoke request or distinct audio/video timestep modulation.

## Problem

The CUDA joint-Transformer owner rounded the AdaLN projection result to BF16 before adding its
bias, then rounded again in the broadcast bias kernel. The source `Linear` contract adds the
bias to the projection accumulator and rounds the combined result once. This extra rounding was
repeated for every block and could accumulate coherently across the fifty-block stack and every
denoising evaluation.

The existing live runner could not express request-shaped positions, per-row AdaLN table
indices, or more than one timestep, so it could not expose this error at the smoke packed-row
scale.

## Causal analysis

The pinned upstream block applies the BF16 AdaLN linear operation to an activated timestep
embedding with its bias as part of the linear result. YVEX kept the cuBLAS projection in F32,
explicitly rounded that result, then used a second kernel to add the broadcast bias and round
again. The first round had no source counterpart.

On the same 466-row synthetic input and random timestep fixture, removing the premature round
reduced relative L2 error from 0.009383294 to 0.007100209 and improved cosine similarity from
0.999955981 to 0.999974805. This comparison used the same source weights, upstream oracle,
positions, row mapping, and hardware; only the YVEX rounding order changed.

The random timestep fixture produced out-of-distribution modulation magnitudes, so a second
fixture derived its two timestep embeddings from the released MiniMax weights and the video and
audio schedules. That request-shaped fixture is the admitted conformance evidence; the random
fixture is retained only as a directly comparable causal measurement.

## Decision

Keep the existing generic encoded projection and broadcast-bias owners, but preserve the F32
projection accumulator until the bias kernel performs the single BF16 publication round. A new
fused kernel was rejected because the existing composition already represents the exact source
order once the premature round is removed.

Extend the existing MiniMax Omni live runner with bounded dynamic rows, positions, AdaLN
indices, and timestep count rather than adding a second test owner. Retain aggregate L2 and
cosine requirements and allow a one-percent maximum-element envelope for legal BF16 reduction
ordering across millions of values; the maximum allowance does not stand alone.

## Implementation

The joint Transformer no longer rounds the AdaLN matrix product before the broadcast bias
operation. The bias kernel remains the single BF16 publication point, removing one kernel from
each block without changing ownership or ABI.

The live runner now supports both its original three-row fixture and an explicit bounded mode.
The new mode reads exact F32 hidden states, timestep embeddings and positions plus unsigned
AdaLN indices, accepts at most 2,048 rows and 64 timestep entries, and preserves the existing
zero-block, out-of-range-index, undersized-output, transactional cleanup, and oracle refusals.

## After

One exact MiniMax block now passes an independent upstream BF16 oracle over 466 packed rows:
15 text rows, 414 audio rows, and 37 video rows. Two real timestep embeddings select distinct
video/text and audio modulation rows. The run admitted 1,291,143,680 resident weight bytes,
used 205,342,208 device bytes, launched 35 kernels, and completed without swap or OOM.

The accepted output had relative L2 error 0.003272757, cosine similarity 0.999995332, and
maximum scaled absolute error 0.008064516. Its execution identity was
`739b795745e37b3af884370c05968c41acf072d30f4aeb99710359798e103d71`; the block residency
identity was `32b94bd45f27f89ddc14b119c4229a432fed22b5d7ca873c12797c57c640dc20`.

This admits request-shaped one-block numerical conformance. It does not establish full-stack or
multi-step parity and does not change the prior video-quality claim.

## Evidence

- The oracle used source revision
  `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08` and pinned reference revision
  `f53d552036a0d1bd5570782a39cd40cfabf112bc`.
- The deterministic fixture manifest SHA-256 was
  `10b6e7adb5f886076a9841d0fb61b83a65a5d75b88a971d8a9e3f4be84f0e3d5`; the upstream oracle
  SHA-256 was `58deffd914912995773dd21fbebd58c7b50ec9ae08dd14662a771fe63de959aa`.
- The accepted YVEX output SHA-256 was
  `0bed9c507128ec691f529f3f70ce2de7c4f03ca63b44faf36d5fc59298da1e90`.
- The representative live run completed in 77.63 seconds with peak RSS 1,672,812 KiB, no
  swap, and exit status zero.
- `make -j2 test-cuda test-cuda-no-nvcc` passed the CUDA unit, parity, qtype, cleanup, and
  no-`nvcc` fail-closed lanes.
- `make -j2 test-runtime-sanitizers` passed ASan/LeakSanitizer and UBSan runtime aggregates.
- `make -j2 check-guardrails check-docs` passed source membership, structure ownership,
  layout, architecture, documentation, and project-control checks.
- `git diff --check` passed. Tracked weight scans found no model payloads; the repository still
  contains only its established small GGUF structural fixtures.

## Quantitative delta

| Directly comparable fact | Before | After |
| --- | ---: | ---: |
| 466-row random-fixture relative L2 | 0.009383294 | 0.007100209 |
| 466-row random-fixture cosine similarity | 0.999955981 | 0.999974805 |
| 466-row random-fixture RMSE | 4,207.692750 | 3,183.902760 |
| AdaLN publication rounds per block | projection round plus bias round | bias round only |

## Remaining limitations

- Conformance covers one of fifty blocks. A complete fifty-block request-shaped oracle has not
  been run because simultaneous upstream and YVEX full-weight residency would materially raise
  the GB10 OOM risk.
- Hidden-state input is deterministic synthetic BF16 data. The timestep embedding, weights,
  row classes, positions, and geometry are source-derived, but the fixture is not a captured
  full-pipeline intermediate.
- The prior native media path has not yet been regenerated after this repair. End-to-end
  regression and visual inspection remain an evidence gap before any quality conclusion.
- The generic full-attention implementation remains bounded to 2,048 packed rows. The source
  768-short-edge geometry is outside this admitted envelope and requires a later memory and
  execution pass.
- No useful-scene, prompt-fidelity, HD, performance, evaluation, or release claim follows from
  one-block numerical conformance.

## Why it matters

The first request-shaped Transformer oracle removed a source-order rounding defect that repeats
through the entire denoising core, while keeping the fix inside the existing generic backend
composition and within the GB10 memory envelope.
