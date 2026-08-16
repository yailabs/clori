# MiniMax Component CUDA Preopen Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | repair |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `af8bb0fe1598a44e5d0f9bba0070b94d5c3ebc09` |
| Checkpoint | `aa6a2e1d5a35f9b4ed02081e36219d03425654f7` |
| Subsystem | generic component runtime residency lifecycle |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; runtime qualification; independent container inspection |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The first complete 384x384 MiniMax media qualification admitted the exact source artifacts,
finished Transformer residency, and then failed while opening the CUDA context. The component
session faulted and locked the roughly 66 GB Transformer arena before asking the CUDA driver to
create its process context. Under the 100 GiB hard memory limit this left no independent context
headroom and `cuCtxCreate` returned `CUDA_ERROR_OUT_OF_MEMORY`.

The process failed through the scoped memory boundary without a global host OOM, tensor payload
publication, or residual model process. Existing full-model runtime open already established the
CUDA context before materialization and residency, but the staged component path did not share
that lifecycle order.

## Problem

Component execution placed a fixed CUDA allocation after the largest host-residency transition.
The resulting failure depended on transient unified-memory pressure rather than the admitted
host/device budgets, so a request that passed preflight could still fail before its first kernel.
Repeating the run with a larger unbounded memory envelope would have hidden the lifecycle defect
and weakened OOM containment.

## Causal analysis

The failure occurred at CUDA context creation after the complete Transformer residency was
committed and locked. No graph or decoder kernel had executed. The same request succeeded when
the context was created immediately after materialization commit and before the resident arena
was faulted, while retaining the exact artifact, geometry, schedules, budgets, and execution
code.

The existing residency attachment ABI already accepts a prepared CUDA backend, transfers its
ownership to the residency, and creates a private shared session backend. The missing behavior
was therefore ordering, not a new allocation mechanism or a MiniMax-specific runtime policy.

## Decision

Open the component CUDA backend before component residency preparation, then pass that prepared
backend through the existing generic attachment path. Preserve CPU ordering and all existing
cleanup ownership. Keep the 100 GiB hard cgroup limit and zero-swap policy as part of live
qualification.

Do not add a family-specific allocator, relax the memory limit, retain the complete Transformer
after its phase, or infer useful visual quality from process completion.

## Implementation

`yvex_runtime_component_session_open` now creates the selected CUDA context after the admitted
materialization transaction commits and before the complete component payload is faulted into
the locked residency arena. `yvex_runtime_residency_cuda_session_attach` consumes the prepared
backend through its existing ownership contract. The established component-session cleanup path
closes it on every later failure.

The repository structural budget was advanced by exactly the measured production-line delta;
no source owner, file, ABI, backend mechanism, family budget, or executable surface changed.

## After

The same 384x384, 124-frame, seed-42 request completed all 49 model evaluations, 50 Transformer
blocks per evaluation, both VAE decoders, and atomic AVI publication under the same 100 GiB hard
limit with swap disabled. It launched 886,360 kernels, reported 2,528,579,072 peak device bytes,
and reached 65,792,152 KiB maximum RSS. The cgroup peak was 96,421,195,776 bytes and its swap peak
was zero.

The resulting 55,521,248-byte file contains 124 uncompressed 384x384 RGB frames at 24 fps and
165,333 stereo PCM samples per channel at 32 kHz. GStreamer independently discovered a seekable
AVI with a 5.166666666-second video duration. The file identity is
`1f2928f59abfdeace614617bc7645f0b1661a5343a14ec854d289d59c21045f4` and the native execution
identity is `8bd8b6555b15e88c62aa9b9a820edc666ec6b8a6f8391c6a0373dbfc4e5f7b65`.

Sampled frames remain a colored mosaic without a recognizable eclipse. This repair establishes
bounded lifecycle completion at 384x384; it is negative visual-quality evidence, not prompt
fidelity or model-quality evidence.

## Evidence

- `git diff --check`, focused `materialization_runtime` and `runtime_media` tests, documentation
  checks, and repository architecture/ownership/layout guardrails passed.
- The failing and succeeding commands used the same source artifacts, prompt, 384x384 geometry,
  124 frames, 49 evaluations, 50 blocks, seed 42, 80 GiB host admission, 4 GiB device admission,
  4 GiB workspace admission, and a 100 GiB process hard limit with swap disabled.
- The initial attempt failed at `yvex_backend_open_cuda_impl` with CUDA error 2 before execution.
  The repaired attempt exited zero and released all component and CUDA residency.
- The publication identity is
  `2738fd637e988c083a567fd2436911f1c18d60a400f78d05822e1d4e44486a37`; the video and audio
  execution identities are `be7720c90f9ef37a9c34eed013b466d9f3b66a7b18d3a659648ec667e379fa6f`
  and `95c63374349e7614afeb28019f3be40f06f08aec2b49ad345df834bbd9424c0b`.

## Quantitative delta

| Fact | Before | After |
| --- | ---: | ---: |
| CUDA context order | after locked component residency | before locked component residency |
| Identical 384x384 request | refused at context creation | completed and published |
| Model evaluations | 0 | 49 |
| Cgroup workload swap | 0 bytes | 0 bytes |
| Published media bytes | 0 | 55,521,248 |

## Remaining limitations

- The complete 49-evaluation latent trajectory has not yet been compared with an independent
  source-weight oracle; that is the active correctness evidence gap for the mosaic result.
- Combined useful-geometry Visual VAE spatial and temporal conformance remains narrower than the
  384x384 media request.
- The released checkpoint targets a 768-pixel short edge. The admitted 384 profile is a bounded
  diagnostic envelope, not HD qualification.
- No prompt fidelity, recognizable-scene, model quality, practical generation speed, benchmark,
  evaluation, hosted serving, or release claim follows from this repair.

## Why it matters

Large staged components now reserve CUDA context headroom before their dominant locked-host
transition, turning an avoidable late OOM into a bounded successful lifecycle without creating a
family-specific memory path.
