# MiniMax-H3 FL2VA Research Record

Status: source-profiled research target; not an admitted executable family

This record owns current YVEX facts for MiniMax-H3 Base FL2VA. It does not add
a model target, artifact format, runtime adapter, backend operation, hosted
path, or release obligation. The frozen quantitative evidence is the
[FL2VA intake audit](../audits/minimax-h3-fl2va-b8b09e3/README.md); the common
promotion rules remain in the [family integration contract](integration.md).

## Target and source identity

| Fact | Value |
| --- | --- |
| Repository | `MiniMaxAI/MiniMax-H3` |
| Immutable revision | `b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08` |
| Admitted subtree | `FL2VA/` only |
| Initial task | `t2va`, text-to-audio-video |
| Source-tree identity | `91972f8e4e6562562456c339b43eed1fba5f7b9d7fb13987f495b416a5109b5e` |
| External evidence identity | `1e7db0167eafc1e43ecaad37897198cd54f838398907b686211886dbc288b662` |
| Model-index SHA-256 | `d1113e0f123c69f79cd0de35ca1771606ebc3ec924270d257b771f96f584aa6b` |
| Evidence stage | source profiled and tensor headers reconciled |

The external evidence directory used during the intake was
`/home/dgmothx/lab/models/intake/minimax-h3/<FULL_REVISION>/`. That path is an
operator location, not semantic identity. Reproduction accepts any output root
and derives the same identities from repository, revision, subtree, selected
metadata, and safetensors headers.

The source tree contains 280 files, of which 81 are beneath `FL2VA/`. The
intake admits those 81 files and the repository license documents. It excludes
`Ref2VA/`, root hosted-system components, `transformer_ref/`, 2K/context/
regeneration scripts, assets, and all non-FL2VA weights. It does not assume
that similarly named root and FL2VA components are interchangeable.

## License facts

The repository declares the Hugging Face license identifier `other`. The
tracked source document is titled “MiniMax H3 Community License Agreement,” is
dated August 2, 2026, and has SHA-256
`59b99642b95ea21630e311198ddbfffbfe05aadba0c2f5d884cbdf4efcc90f44`.
Its defined Applicable Territory is worldwide excluding the European Union,
United Kingdom, Republic of Korea, and United States. It states conditions for
use, modification, derivative works, distribution, hosted services,
commercial products, acceptable use, notices, and termination. It separately
notes that the Qwen3-VL encoder is licensed under Apache 2.0.

The repository also includes `docs/QA-about-License.md`, SHA-256
`26900a2868636f886d241efe94002ad11d858b3c3eb005e58c0c2ad60f0de7ae`.
It discusses the territorial restriction, separate authorization for
organizations in restricted regions, and possible future changes. These are
source facts, not legal advice. YVEX has not determined eligibility for local
research use, transformed artifacts, redistribution, derivative checkpoints,
hosted use, or future commercial deployment. Each remains a legal and
operator authorization question outside this engineering intake.

## Component inventory

The exact file lists, largest tensors, inputs, outputs, lifetimes, reference
owners, and blockers are in the audit’s [`components.tsv`](../audits/minimax-h3-fl2va-b8b09e3/components.tsv).

| Component | Declared class | Shards | Tensors | Elements | Tensor payload bytes | Source dtype | Initial `t2va` role |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Processor | `Qwen3VLProcessor` | 0 | 0 | 0 | 0 | metadata | prompt and optional media preparation |
| Tokenizer | `Qwen2TokenizerFast` | 0 | 0 | 0 | 0 | metadata | exact prompt tokenization and special-token framing |
| Text/vision encoder | `MiniMaxH3Qwen3VLHFEncoder`; `Qwen3VLForConditionalGeneration` | 14 | 1,058 | 33,357,390,064 | 66,714,780,128 | BF16 | conditioning representation |
| Omni-Transformer | `MiniMaxH3DiTModel` | 13 | 535 | 33,122,992,912 | 66,280,430,144 | BF16 plus 13 F32 tensors | repeated joint latent update |
| Visual VAE | `MiniMaxH3VideoVAE`; `AutoencoderKLLegacy` | 1 | 560 | 2,603,871,032 | 10,415,484,128 | F32 | video latent decode |
| Audio VAE | `MiniMaxH3AudioVAE` | 1 | 1,087 | 151,326,585 | 605,306,340 | F32 | audio latent decode |
| Latent controller | scheduler entry is `null` | 0 | 0 | 0 | 0 | metadata only | initialization, timesteps, and candidate update |
| Pipeline metadata | `MiniMaxH3Pipeline` | 0 | 0 | 0 | 0 | metadata only | component graph and `t2va`/`fl2va` task boundary |

The four weighted components contain 3,240 tensors, 69,235,580,593 parameter
elements, and 144,016,000,740 declared payload bytes. BF16 accounts for 1,580
tensors and 132,926,321,632 bytes; F32 accounts for 1,660 tensors and
11,089,679,108 bytes. File bytes include safetensors headers and metadata and
must not be substituted for runtime residency.

The processor, tokenizer, and text-encoder copies of `tokenizer.json` are
byte-identical, as are their `tokenizer_config.json` copies. That is metadata
duplication, not shared weight ownership. The text encoder explicitly declares
`tie_word_embeddings=false`; its embedding and `lm_head` have equal shapes but
are distinct source tensors.

## Architecture signature

### Source-declared facts

The FL2VA model index declares one Qwen3-VL processor, Qwen2 fast tokenizer,
Qwen3-VL encoder wrapper, MiniMax H3 DiT, visual VAE, and audio VAE. It declares
tasks `t2va` and `fl2va`, no standalone scheduler component, and sigma shift
scales 12.0 for video and 3.0 for audio.

The tokenizer declares a 262,144-token maximum, no added BOS, `<|im_end|>` as
EOS, `<|endoftext|>` as pad, and image/video framing tokens. The processor
declares spatial patch size 16, temporal patch size 2, merge size 2, and
separate image/video pixel bounds.

The Qwen3-VL text configuration declares:

- 64 layers, width 5,120, FFN width 25,600;
- 64 query heads, 8 KV heads, head dimension 128;
- SiLU, RMSNorm epsilon `1e-6`, no attention bias or dropout;
- RoPE theta 5,000,000 and interleaved MM-RoPE sections `[24, 20, 20]`;
- vocabulary 151,936 and BF16 source dtype.

The vision configuration declares 27 layers, width 1,152, FFN width 4,304,
16 heads, patch geometry `2 x 16 x 16`, spatial merge 2, output width 5,120,
and deep-stack outputs at layers 8, 16, and 24.

The Omni-Transformer declares 50 layers, width 5,376, FFN width 14,336,
56 heads of dimension 128, Q/K normalization epsilon `1e-5`, two token-refiner
layers, text width 5,120, 24 video latent channels, 32 audio latent channels,
video patch `[1, 2, 2]`, and an inverse-frequency tensor length of 16. Timestep
features are width 256, projected through 5,376 to 2,688. Each block’s AdaLN
projection has 96,768 outputs; the final AdaLN has 10,752 outputs.

The visual VAE declares 24 latent channels, RGB input/output, spatial factor
16, temporal factor 4, a causal encoder and noncausal decoder, 3D convolution,
and a 36-layer, 32-head RMS-normalized ViT decoder. Its standalone wrapper
declares tile size 256, minimum overlap 64, clip length 17, and token drop 3.

The audio VAE declares 32 latent channels, stereo output, and 32,000 Hz sample
rate. Encoder rates `[2, 4, 4, 5, 5]` and decoder rates
`[5, 5, 2, 2, 2, 2, 2]` both multiply to 800.

### Mechanically derived facts

The Omni attention QKV width is 21,504, exactly three times
`56 * 128`; its output projection consumes 7,168 values. The header therefore
supports a full 56-head Q/K/V representation, not grouped KV. This is an
architecture classification, not proof of the exact attention kernel or mask.

The video patch projection consumes 96 values, equal to
`24 * 1 * 2 * 2`, and the final video head emits 96. The audio patch and final
audio projections consume and emit 32 values. The per-block AdaLN width is
`18 * 5,376`; its 50 weight matrices and biases occupy 26,020,915,200 bytes.
If their inputs are timestep-only, an identity-bound precompute could replace
those resident matrices with 9,676,800 BF16 bytes per iteration. That cache
property still requires numerical confirmation.

For dimensions divisible by the declared ratios, visual latent geometry is
approximately `[B, 24, ceil(F/4), ceil(H/16), ceil(W/16)]`; patch tokens are
`ceil(F/4) * ceil(H/32) * ceil(W/32)`. Audio latent cadence is mechanically
40 steps per second from `32000 / 800`. Boundary padding can change exact
ceilings and is not yet admitted.

### Inferences awaiting numerical confirmation

Tensor widths and the common DiT owner suggest that conditioning, patched
video latents, and audio latents enter one joint full-attention sequence. Names
and widths suggest gated SiLU FFNs, timestep-driven AdaLN shift/scale/gates,
and separate final projections back to audio and video latent layouts. These
are strong lowering hypotheses, not executable contracts.

The source class and sigma shifts imply an iterative diffusion or flow-style
latent process. The admitted metadata does not establish the noise
distribution, seed algorithm, solver, timestep sequence, update equation,
guidance rule, or iteration count. No YVEX implementation may invent them.

### Unknowns at this evidence boundary

- exact Omni MM-RoPE axis allocation, replication, and interleave policy;
- exact attention masks and conditioning-token placement;
- latent initialization and candidate-update equation;
- solver, timestep schedule, step count, and cancellation safe points;
- output frame count, frame rate, width, height, and duration defaults;
- exact video temporal padding/reconstruction at clip boundaries;
- exact audio/video duration alignment and timestamp policy;
- output codec, container, color/range convention, and atomic publication ABI;
- whether every component can be unloaded immediately at the inferred phase
  boundary without retaining implementation-owned buffers;
- numerical acceptability of quantization, F32 narrowing, modulation caching,
  or other derived execution assets.

## Component and execution graph

```text
prompt + optional conditioning media
  -> Qwen3-VL processor + Qwen2 tokenizer
  -> Qwen3-VL text/vision context encoder
  -> immutable 5120-wide conditioning representation
  -> video/audio latent initialization
  -> patch/projection + two conditioning token-refiner blocks
  -> repeated 50-block Omni-Transformer candidate update
  -> committed video latent -----> visual VAE -----> RGB frames
  -> committed audio latent -----> audio VAE ------> stereo 32 kHz samples
  -> synchronized media packaging
  -> atomic output publication
```

This is a latent-iteration DAG. It does not use autoregressive prefill/decode,
KV-cache, logits, token sampling, or accepted-token terminology for the DiT
loop.

| Edge | Shape/dtype evidence | Owner and lifetime | Placement/serialization | Failure rule |
| --- | --- | --- | --- | --- |
| request -> processor | UTF-8 text; optional RGB/media tensors; source metadata only | request-owned, discard after encoding | host preprocessing; no serialization | invalid media or template publishes no request |
| processor -> Qwen3-VL | token IDs plus patch/grid descriptors | encoder input, immutable for one context phase | staged host/device | token/grid refusal releases encoder stage |
| Qwen3-VL -> conditioning | width 5,120; runtime dtype unconfirmed | immutable request cache, retained through DiT | device-preferred; no file format selected | failed encoding publishes no conditioning identity |
| controller -> latent initialization | video 24-channel and audio 32-channel; exact extents/dtype unknown | mutable request candidates | device-preferred; seed and state must be identity-bound | cancellation rolls RNG and both latent candidates back |
| latents -> Omni input | video patch width 96, audio width 32, projected to 5,376 | mutable candidate plus immutable condition | same execution device; staged component residency | shape/mask/MM-RoPE refusal starts no iteration |
| iteration `i` -> `i+1` | same video/audio latent geometry; update equation unknown | one candidate transaction per step | no serialization; optional derived modulation cache | audio and video advance together or neither advances |
| committed video -> visual VAE | 24-channel latent; source F32 weights | discard latent after successful frame decode | VAE may be late staged; tiled workspace | partial frames remain unpublished |
| committed audio -> audio VAE | 32-channel latent to two 32 kHz channels | discard latent after successful waveform decode | VAE may be late staged | partial samples remain unpublished |
| decoders -> packager | RGB frames plus stereo samples; timing unknown | request output transaction | host/device transfer and media serialization required | mismatched duration or serializer failure publishes nothing |

## Operation-gap result

The complete 46-row map is in
[`operation-gap.tsv`](../audits/minimax-h3-fl2va-b8b09e3/operation-gap.tsv).
It finds 2 available generic mechanisms, 14 bounded extensions, 22 new generic
primitives, 3 backend-fusion candidates, 4 family-specific compositions, and
1 unknown schedule boundary.

Existing backend allocation/copy/synchronization and artifact identity are
usable. Dense math, embedding, RMSNorm, SiLU MLP, RoPE, tokenizer execution,
residency, inter-component transfer, cancellation, and atomic file
publication need bounded type, shape, or lifecycle extensions.

The material missing boundaries are batched memory-efficient full attention,
arbitrary masks, MM-RoPE, patchify/unpatchify, elementwise residual/modulation,
Conv1D/2D/3D, temporal and spatial resampling, alias-free audio operations,
AdaLN and derived-cache lifecycle, tensor-normal RNG, typed audio/video data,
media serialization, synchronized output transactions, and composite phase
residency. Visual/audio VAE execution and joint iterative updates are family
compositions over those generic mechanisms. A similarly named decoder
primitive does not establish any of these semantics.

## Single-GB10 memory feasibility

The hardware budget is treated as 128,000,000,000 bytes; using 128 GiB would
not change scenario A. Source payload is stated separately from runtime
residency. Estimates reserve 8 GB for the allocator and 12.8 GB as safety
margin. They are capacity analysis, not observed runtime measurements.

| Scenario | Weight basis | Estimated peak | Classification | Controlling condition |
| --- | ---: | ---: | --- | --- |
| A — simultaneous original dtype | 144.016 GB | at least 167.8 GB before a realistic large attention/VAE workspace | infeasible | weights alone exceed 128 GB and 128 GiB |
| B — staged original dtype | max phase 66.715 GB | 90.8–111.1 GB with tiled/online attention; an eager full matrix is far larger | conditionally-feasible | exact component release, memory-efficient attention, tiled VAE, admitted request geometry |
| C — staged plus selective transforms | 40.260 GB transformer after confirmed AdaLN precompute; hypothetical 35.025 GB 8-bit encoder and 5.208 GB BF16 visual VAE | 59–84 GB | conditionally-feasible | independent conformance for every transform and cache identity |

The scenario-B lower/upper envelope includes 2.5–12 GB activation memory,
0.5–8 GB tiled attention workspace, up to 32 GB VAE workspace in its separate
phase, 0.01–1 GB latent candidates, 0.05–2 GB conditioning, 0.1–1 GB output
buffers, reserve, and margin. These ranges are assumptions because the source
does not declare request geometry or backend workspace.

For scale pressure only, assume 144 frames at 1,360 x 768 and six seconds of
audio. The declared ratios give 37,152 video tokens and 240 audio tokens,
before conditioning. At an illustrative total of 37,400 tokens, two BF16
score/probability matrices for 56 heads require about 313.3 GB:
`4 * 56 * 37400^2`. A memory-efficient attention implementation is therefore
a feasibility prerequisite, not an optimization.

Likely compute blockers are the joint sequence length, quadratic full
attention across 50 blocks, unknown iteration count, the 36-block visual VAE
decoder, convolutional resampling, and repeated movement of staged components.
The evidence contains no released sparse-attention contract, no sequence-
parallel contract, and no multi-device plan. Nothing here establishes
acceptable generation speed, 768p practicality, or any 2K path.

## Integration decision

| Alternative | Identity/artifact and reuse | Runtime/residency/output | Disposition |
| --- | --- | --- | --- |
| A. Existing target plus several artifacts and one program | preserves individual artifacts but the existing target/Physical Execution IR is decoder- and single-model-shaped; phase identity would be implicit | cannot represent component release, latent request state, or synchronized media output without hidden conventions | reject |
| B. Composite model target with component manifest | binds source derivation, exact component artifacts, roles, phase DAG, residency plan, and output contract while allowing encoder/VAE reuse | one common runtime owns request/session and transitions; graph owns the typed program; backend admits each operation | **conditionally-adopt** |
| C. Monolithic GGUF | one byte identity but destroys natural component lifetimes, selective reuse, and staged residency; no evidence supports one GGUF schema for media/codecs | forces 144 GB source payload into one artificial lifecycle and couples unrelated failures | reject |
| D. Generic execution package above artifacts | could generalize to image/audio/video and encoder-decoder models | useful only after the concrete composite target proves which package invariants are common; premature now | defer |
| E. No integration until every numeric primitive exists | avoids early runtime work but also blocks safe source/IR admission that does not consume numeric execution | missing numeric prerequisites have explicit later consumers and need not block identity/IR work | reject as the immediate strategy |

Alternative B is conditional on a typed component manifest and phase-DAG
contract being proven by the next source/IR slice. It is not a composite
artifact: each physical artifact retains its own identity, admission,
materialization, and failure. The composite target binds them without
promoting any one artifact to runtime readiness.

The decision preserves one common YVEX runtime, no family runtime hierarchy,
backend ownership of allocation/transfer/execution/cleanup, graph ownership of
executable semantics, artifact identity separate from runtime readiness, and
no promotion from metadata. It deliberately does not implement a component
manifest, execution package, Physical Execution IR extension, residency
transition, media session, artifact format, or backend operation in this wave.

## Selected first executable slice

Scores are 0–5 with 5 favorable. Full scoring is frozen in the audit metrics.

| Candidate | Score / 50 | Disposition |
| --- | ---: | --- |
| Audio VAE standalone decode/roundtrip | 34 | small weights but many missing audio primitives and expensive numerical surface |
| Visual VAE reduced fixture | 27 | useful output but 10.4 GB F32 weights and Conv3D/ViT/tiling breadth |
| One Omni-Transformer block | 35 | high generic value, but exact mask/MM-RoPE/AdaLN semantics must be admitted first |
| Full source-to-architecture/Transformation IR without numeric execution | 40 | selected; closes identity, role, component, and phase ownership before numeric abstraction |

The selected next wave is `R010.MINIMAX.H3.FL2VA.IR.0`.

Its required after-state is one production-admitted, artifact-neutral
MiniMax-H3 FL2VA source-to-Transformation-IR path. It must:

- admit the exact repository/revision/subtree through the canonical source
  snapshot and payload-trust owners rather than consuming this research tool;
- add at most one authorized `minimax_h3` model-family production owner in
  the canonical family directory, with manifest/build registration and no
  runtime family source;
- extend the existing Transformation IR only where the concrete component,
  scope, role, and phase facts cannot be represented, without adding numerical
  execution operations to Transformation IR;
- identity-bind all eight logical components, 29 shards, 3,240 tensor roles,
  duplicate tokenizer metadata, source dtypes/shapes/ranges, component edges,
  phase lifetimes, and higher-stage unknowns;
- expose the result through an existing capability-oriented `yvex` inspection
  command, with no startup-ready registry entry;
- stop before artifact emission, materialization, Physical Execution IR,
  runtime sessions, backend execution, or media output.

Dependencies are the canonical source inventory/payload APIs, model-family
registration, Transformation IR, documentation ownership, and the exact
external intake identities. Acceptance requires deterministic IR identity,
complete role coverage, positive tests, duplicate/missing/stale/refusal tests,
source revision and subtree mismatch refusal, resource-budget refusal, two
repeat builds/checks, operator reachability, and proof that no payload byte is
executed.

Numerical conformance, scheduler semantics, component artifact formats,
runtime residency, Omni execution, VAE execution, media serialization,
generation quality, and performance remain explicit non-claims of that slice.

## Progression and non-claims

`progression_decision: proceed`

`downstream_safe: true`

The downstream consumer is `R010.MINIMAX.H3.FL2VA.IR.0`. It consumes source,
role, component, and identity facts only; it does not consume the unknown
solver or numerical paths. There are no intake gate blockers or boundary
incompletenesses. Deferred depth consists of the exact scheduler and MM-RoPE
numeric contract, component execution, artifact formats, residency, media
transaction, evaluation, and benchmark, each owned by a later executable
consumer. No external blocker prevents the selected IR slice. License review
remains an external authorization prerequisite for any use that requires an
eligibility conclusion.

This intake does not prove:

- MiniMax-H3 execution in YVEX or artifact emission;
- complete composite-artifact or execution-package support;
- simultaneous or staged runtime residency;
- audio or video generation, synchronized media output, or hosted serving;
- Diffusers, SGLang, or vLLM parity;
- model quality, generation speed, 768p practicality, or 2K generation;
- Ref2VA, H3-Context-IR, H3-Regenerate-2K, or the complete hosted system;
- commercial or redistribution eligibility;
- evaluation, benchmark, release support, or a second complete family vertical.
