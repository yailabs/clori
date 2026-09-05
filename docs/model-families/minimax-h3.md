# MiniMax-H3 FL2VA Family Record

Status: admitted composite execution with bounded numerical conformance;
full-scale numerical/behavioral qualification remains open.

This record owns current family-specific source, topology, physical policy,
and evidence barriers. The common [integration contract](integration.md) owns
promotion rules; [ROADMAP](../../ROADMAP.md) owns macro progression.
Implementation chronology is recoverable from Git, not an active family branch.

The product accepts text-only, first-frame, last-frame, and first-plus-last-frame
FL2VA requests. These use the common engine/session transaction, staged component
execution, and synchronized-media publication. A complete 49-evaluation
trajectory has executed, but its observed mosaic output is negative behavioral
evidence, not useful full-model generation qualification.

## Target and source identity

| Fact | Value |
| --- | --- |
| Repository | `MiniMaxAI/MiniMax-H3` |
| Immutable revision | `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08` |
| Admitted subtree | `FL2VA/` only |
| Source-tree identity | `91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e` |
| Acquisition identity | `7eec5b07bbb6427611553b16670f9dc31969ae8ba602a79c0c7a2693a5fa168a` |
| Source snapshot | `897ceaff08708f431132c6643bc8f1041ace8c0444a3ea248bbf727fc7da9943` |
| Component manifest | `715f2359aaff048ccca8207976421af5f9f76b08b6f24986b3cc186d2822bc0e` |
| Phase DAG | `c02c17739fda5ef12698f486f201d638e9b5afdd2cc97c27b24768f8e51ccfa5` |
| Architecture | `47a03bbac2b5346771f70ae39155920f9b1c6e6cec17f2639dd0cbedfa90b517` |
| Tensor-role map | `61e7a2cfc29e6dd3da966878f5388f1472a406d7e33ba34ef65f44b61f08f013` |
| Transformation IR | `bd941103d754df8c1eb02ff9b90db4ba86b7e389691f2d0c4027343eccbc0b0b` |
| Aggregate derivation | `cc2886a388a475c2df246558dfb41c8d66e549afd4c4e34d19e2bbd3b70a3ff5` |

Acquisition admits 83 files: 29 weighted shards containing 144,016,376,436
file bytes and 54 metadata/license files containing 34,827,744 bytes. Complete
verification covers 144,051,204,180 bytes, 3,240 tensors, 69,235,580,593
elements, and 144,016,000,740 declared tensor-payload bytes.

`Ref2VA/`, root hosted-system components, `transformer_ref/`, 2K/context/
regeneration scripts, and non-FL2VA weights are excluded. Similarly named
root and FL2VA components are not interchangeable. Paths, transfer URLs,
timestamps, and development branches do not enter semantic identity.

Intake evidence is external under
`<models-root>/representations/minimax-h3/<FULL_REVISION>/`, evidence identity
`1e7db0167eafc1e43ecaad37897198cd54f838398907b686211886dbc288b662`.
Reproduction may use another location; the pinned files determine identity.

## License observations

The pinned source declares license `other`. Its MiniMax H3 Community License
Agreement, dated August 2, 2026, has SHA-256
`59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`.
The recorded Applicable Territory excludes the European Union, United Kingdom,
Republic of Korea, and United States. The Qwen3-VL encoder is separately
identified as Apache 2.0.

The source license FAQ has SHA-256
`26900a2868636f886d241efe94002ad11d858b3c3eb005e58c0c2ad60f0de7ae`.
These are pinned-source observations, not legal advice or a current eligibility
determination. Research, transformation, redistribution, hosted use, and
commercial eligibility remain operator authorization questions.

## Components and artifacts

| Weighted component | Shards | Tensors | Elements | Payload bytes | Source dtype |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-VL text/vision encoder | 14 | 1,058 | 33,357,390,064 | 66,714,780,128 | BF16 |
| Omni-Transformer | 13 | 535 | 33,122,992,912 | 66,280,430,144 | BF16 plus 13 F32 tensors |
| Visual VAE | 1 | 560 | 2,603,871,032 | 10,415,484,128 | F32 |
| Audio VAE | 1 | 1,087 | 151,326,585 | 605,306,340 | F32 |

Processor, tokenizer, controller, and pipeline metadata have no separate
weight allocation. Tokenizer copies under processor/tokenizer/text encoder
are byte-identical; this is duplicated metadata, not duplicated model memory.

The admitted source-faithful component GGUF evidence uses these whole-file
identities:

| Component | SHA-256 |
| --- | --- |
| Text/vision encoder | `61407a737bf019cef8f0d786394986d419af957bd23a120f6dcc070128abb7ff` |
| Omni-Transformer | `aa1c84ac801a50f8806b591fb419e60513f0e6bd312b5e3abc5352194a31b992` |
| Visual VAE | `29bb1df65227fa05444c4002e18d61934d70d872d8472c4757e93971f9e474cd` |
| Audio VAE | `52a10c9f6f6e3b9b81569a95329f503fcb3cbddb224d12bf7851b4929d02e1c1` |

Each binding authenticates component, source snapshot, physical plan,
aggregate payload, and whole-file identity before materialization. These are
evidence anchors, not aliases for every future deployment. Inspect the current
catalog/profile for its exact binding. One component is not the whole model.

## Architecture and conditioning

The logical phase DAG is prompt/media preparation, text/vision conditioning,
keyframe encoding where requested, joint latent trajectory, visual decoding,
audio decoding, and transactional publication. Family recipes supply topology
and numerical policy; common runtime owners allocate, schedule, commit, cancel,
and release resources.

### Text and vision

The text stack has 64 layers, hidden width 5,120, FFN width 25,600, 64 query
heads, eight KV heads, and head dimension 128. Conditioning consumes
unnormalized `hidden_states[50]`: layers 0–49, without final normalization or
LM head. Equal-shaped embedding and LM-head tensors are untied.

The vision stack has 27 layers, width 1,152, FFN width 4,304, 16 heads,
2×16×16 patches, and spatial merge two. Deep-stack features from layers 8,
16, and 24 enter after language layers 0, 1, and 2.

The artifact-bound Qwen2 BPE policy has 151,643 base tokens, 151,387 merges,
26 added tokens, and 14 specials. Valid UTF-8 is NFC-normalized with Unicode
15 semantics. Prompt text is verbatim: no invented BOS or chat template.
EOS is 151645 and padding 151643. Image framing belongs to the source processor.

First/only images use the released stretch path; last images use cover crop.
Lanczos resizing, smart-size selection, bicubic patch extraction, grid and
visual token framing precede the vision stack. Original image conditions and
their derived latents remain distinct.

### Keyframe state

The causal Visual VAE encoder computes a posterior, clamps its source-defined
range, samples with F16 rounding, and normalizes the latent. Condition noise
uses the released 0.999 policy and consumes RNG before target initialization.
Immutable condition rows are not updated with mutable target rows.

### Joint trajectory

Omni has 50 blocks, hidden width 5,376, FFN width 14,336, 56 heads, and head
dimension 128. Two refiner layers precede joint execution. Text width is 5,120;
video/audio latent channels are 24/32. Video patching 1×2×2 produces width 96.

T2VA packs `[text | audio | video]`; conditioned execution packs
`[text-and-visual | condition-video | audio | video]`. Attention is exact and
unmasked, not causal. Modality tags are video=0, text=1, audio=2. MMRoPE uses
source FP64 coordinate construction followed by F32 values.

Video temporal coordinates use the source 5/3 schedule and 32-reference
spatial grid. Audio uses 40 latent steps/s with channel-major flattening.
A 124-frame video has 37 latent frames; the admitted temporal geometry follows
17n+5 output frames to 5n+2 latent frames. Do not substitute a text KV model for
this iterative state.

The released policy `released-fl2va-v1` uses 50 sigma points, 49 evaluations,
terminal sigma zero, video/audio shifts 12/3, timestep `1-sigma`, and the
source-ordered F32 rectified-flow update. YVEX PCG XSH-RR plus Box–Muller is
deterministic under its own identity; the same seed does not claim PyTorch RNG
parity. AdaLN timestep rounding and normalization remain numerically sensitive;
an unproved cache or rearrangement is not interchangeable.

## Physical and output policy

The source owner defines 24 fps, 32 kHz audio, 124–345 output frames,
2–64 inference points, and a default 1344×768 canvas. Dimensions are multiples
of 32, area ranges from 768² to 768×1344, and aspect ratio from 1/4 to 4.
The family packed-row bound is 106,238.

Bounded preview profiles deliberately differ from the released trajectory:
192²/124 frames/two sigma points means one evaluation. A preview completion
must not be rendered as 49-evaluation source-default qualification.
The [model owner](../../src/model/families/minimax_h3.c) and
[typed family geometry](../../include/yvex/internal/families/minimax_h3.h)
are the current policy authority.

All weights total about 144 GB, exceeding the 128 GB GB10 machine. One
composite engine stages components at phase boundaries; mapping all source
files is not simultaneous execution residency. Largest source component
payload is about 66.7 GB. Current/peak prepared resources, arena, workspace,
state, explicit allocations, and process RSS come from common accounting.
Do not add overlapping allocator/RSS/mapped totals or infer UMA page placement.

### Visual and audio decoding

Visual decoding uses 36 noncausal blocks, width 2,048, 32 heads of dimension
64, partial 3D RoPE width 48, spatial expansion 16, and temporal expansion four.
Source-defined 256 tiles, overlaps, and top-then-left blending reconstruct
frames before final clamping and output normalization. A 124-frame 768²
output uses seven temporal windows and 16 spatial tiles: 112 decoder calls,
not a single-tile full-model proof.

Audio decoding maps 32 latent channels through width 2,048 and the BigVGAN
stack: seven upsampling rates with product 800, 21 AMP blocks, SnakeBeta, and
stored resampling filters. CPU and CUDA component implementations emit 32 kHz
PCM.

### Publication

The native terminal product is AVI 1.0 with BGR24 video and PCM16LE stereo.
Publication is atomic and no-replace. Excess audio is cropped to the exact
rational video duration; insufficient audio, non-finite samples, cancellation,
or failure publishes no terminal file. GStreamer supplies an independent
container/decode check.

Operator commands, preview selection, attachments, and output paths belong to
the [media procedure](../operator-runbook.md#direct-minimax-h3-media-host).
The family does not own another host, session manager, event plane, or CLI.

## Numerical and lifecycle evidence

These retained results are scoped comparisons, not a claim of whole-model
quality. Raw oracle output remains external. The pinned upstream component
code and independent PyTorch/Transformers calculations supply numerical
references, never a hidden production dependency. The live comparison owners
are [audio](../../tests/live/minimax_h3_audio.c),
[visual](../../tests/live/minimax_h3_video.c),
[text](../../tests/live/minimax_h3_text.c), and
[joint Transformer](../../tests/live/minimax_h3_transformer.c).

The detailed fixture/input/output hashes and intermediate experiments remain
recoverable in this record at documentation checkpoint
`3f4a1c182d35e5a0e163adb81008ae7a366efcc6`. That is a history pointer, not a
claim that all listed experiments ran on that tree. New qualification must
record its own source, oracle, artifact, binding, and run identities.

| Boundary | Retained observation | Scope |
| --- | --- | --- |
| Audio VAE, one latent step | CPU max abs 3.8743e-7; CUDA 3.72529e-7; tolerance 1e-5 | Component |
| Visual VAE, 1×24×1×1×1 | CPU max abs 5.0664e-6; CUDA 8.34465e-7; tolerance 1e-5 | One small decode; reference CPU/CUDA divergence on larger spatial fixtures can reach 3.2687e-4 |
| Text conditioning, nine tokens/50 layers | Relative L2 0.00425595374, cosine 0.999999444619, RMSE 0.304108353 | Largest absolute difference 64 occurs on an approximately 15,000-valued element; relative agreement must not hide it |
| Processor | Token IDs, tags and grid exact; patch max abs 5.91389835e-8 | Released image preparation |
| Vision merger / deep stack | Relative L2 0.0431341 / 0.026254; cosine 0.999091 / 0.999658 | BF16 multimodal component |
| Selected multimodal conditioning | Relative L2 0.00921249; cosine 0.999958 | Selected 50-layer output |
| Visual encoder | Posterior max abs 6.66379929e-5; sampled latent 0.00115990639 | Bounded keyframe fixture |
| Conditioned Omni first/last/both | Video relative L2 9.3183e-5 / 3.5189e-5 / 6.6433e-5; audio 5.7267e-6 / 4.6593e-6 / 3.3702e-5 | Condition-aware component |
| Two evaluations, 466 packed rows, all 50 blocks | Video relative L2 0.004744679, cosine 0.999988780; audio 0.004959823, cosine 0.999987701 | Bounded repeated trajectory |
| Source-scale 768², one block | Video relative L2 0.0060228159; audio 0.00375755049 | One block, not 49-evaluation conformance |

Four bounded product requests (text, first, last, both) ran in one engine,
each producing 124 192² frames at 24 fps with 32 kHz stereo in a 14,381,024-byte
AVI. Component resources returned to zero after each turn. This proves
bounded hosted composition and cleanup, not useful-resolution quality.
Cancellation and publication remain common transaction obligations.

### Full-scale negative boundary

A historical 768², 124-frame, 49-evaluation run executed all 50 blocks per
evaluation and published a 220,082,144-byte AVI. Execution identity:

`3089bfbf486c78cfa360c7421ad91031bb4770cfdef110de0821af25184e77c6`

AVI SHA-256:

`cf54a70f6854c33d775ef4a624f60e6ecafb54f0302d4277b68943f31bfe48f0`

That run took 2h30m28s and recorded 942,556 kernels, peak explicit device
allocation 12,960,345,604 bytes, workspace 1,485,742,080 bytes, peak RSS
67,176,200 KiB, and zero swap. These historical counters are not current
performance or additive memory totals.

Seven sampled frames showed mosaic-like output rather than a coherent scene.
The completed lifecycle therefore does not close full-model numerical or
behavioral correctness. A later faster evaluation or passing component oracle
cannot erase this boundary; it needs an identity-bound whole-model comparison.

### Rejected physical representation

An experimental 48,217,105,664-byte Q8 Omni artifact converted 200 matrices
and retained 335 source tensors. On a 21,741-row block it failed the declared
video limits: relative L2 0.0385167 > 0.02, cosine 0.99927958 < 0.9998, and
scaled maximum 0.0701 > 0.02. It is not a registered runtime representation.
Its existence does not authorize quantized full-model promotion.

## Explicit non-claims

This record does not establish Ref2VA, the complete upstream hosted system,
useful full-resolution output, whole-trajectory numerical conformance,
quality parity, release benchmarking, or release qualification. It does not
claim that one-evaluation previews or one-component artifacts represent the
entire deployment.

No graph/allocator optimization changes these evidence ranks. Future work
must inspect the current source and reproduce the remaining boundary before
selecting a physical repair.
