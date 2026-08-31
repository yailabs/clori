# MiniMax-H3 Canonical FL2VA Keyframes

| Field | Value |
| --- | --- |
| Date | 2026-08-28 |
| Type | closure |
| Milestone | `H3.CANONICAL.1` |
| Branch | `models1` |
| Baseline | `e97ddb705a811ca0e23c8fba5b0e92032718299e` |
| Checkpoint | `c7826d4173db1aeb222066f73a8e5a9f7d198ce8` |
| Subsystem | tokenizer, image input, MiniMax family, CUDA vision/VAE composition, media runtime, protocol, CLI |
| Model family | MiniMax-H3 Base FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | independent upstream oracles, four product executions, focused unit/structural/ABI tests |
| Comparability | not-applicable |
| Publishability | reviewed |

Command note: this record uses the current explicit `yvex chat` client
spelling; bare `yvex` now prints the product command map.

## Before

The admitted MiniMax-H3 FL2VA engine executed only text-to-audio-video. It rejected non-ASCII
prompt bytes, never ran the Qwen3-VL vision tower, recognized Visual VAE encoder weights without
executing them, and packed only `[text | target-audio | target-video]`. The product request could
not carry a first or last image role. Consequently the released first-frame, last-frame, and
first-plus-last-frame FL2VA modes were absent even though their component weights were already in
the authenticated package.

The bounded T2VA path, 50-block Omni execution, both decoders, staged residency, composite reopen,
and transactional publication were retained authorities. This delivery extended those owners; it
did not create another runtime, engine alias, Transformer, scheduler, or media generator.

The implementation began from executable source `e740e6dd6f8c5b9c286e480070a05a98c01664a9`.
The shared branch advanced to the baseline above through one README-logo-only commit while work was
in progress; no executable owner or retained result changed.

## Problem

The package admitted a checkpoint whose released task contract included keyframe-conditioned
FL2VA, but the product could express and execute only T2VA. Treating model weights as sufficient
would have hidden missing processor, vision, encoder, condition-RNG, layout and request semantics.
The ASCII refusal also contradicted the source tokenizer's normal Unicode input contract.

## Causal analysis

The gap was not one missing switch. The text-only request had no typed condition carrier, the Qwen
component executor did not consume its vision tensors, the Visual VAE implementation exposed only
decode, and the packed trajectory had no immutable condition partition. Each absent stage was an
independent source semantic that had to join the same authenticated component and engine lifecycle.

The source authority is `MiniMaxAI/MiniMax-H3` revision
`b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08`, subtree `FL2VA/`.

### Source semantics

The released FL2VA processor distinguishes zero, one, and two image conditions. A sole or first
image is resized to the admitted canvas; a subsequent last image uses cover-crop semantics. Qwen
smart-size admission, bicubic vision resizing, block-major patch packing, framing tokens and
`grid_thw` form one multimodal processor record. The 27-layer vision tower publishes merger output
and deep-stack states at vision layers 8, 16, and 24. Those states enter the language stack after
language layers 0, 1, and 2; MiniMax retains the selected unnormalized `hidden_states[50]` at width
5,120.

Each condition also traverses the released Visual VAE encoder. The posterior is clamped, sampled
with an explicit request-bound draw stream, narrowed through the source F16 boundary, and
normalized. Condition noise is drawn before target audio/video initialization and constructs the
`.999` visual condition state. First and last roles own distinct temporal anchors. Condition rows
remain immutable through the trajectory while only target audio/video rows participate in the
paired commit.

## Decision

Represent first and last images as typed request conditions and preserve prompt opacity. Reuse one
FL2VA engine generation for all four modes. Put image container decoding, NFC and reusable numeric
operations in generic owners; retain processor rules, keyframe roles, posterior/RNG order and
condition layout in the admitted MiniMax family projections. Require independent evidence at the
processor, Qwen, VAE encoder and first Omni evaluation boundaries before product execution.

## Implementation

The generic tokenizer owner now validates UTF-8 and performs deterministic Unicode 15 NFC
normalization. Canonically composed and decomposed forms therefore reach the byte-level BPE as the
same normalized bytes; malformed UTF-8 fails closed.

The generic image owner decodes strict PNG input into an identity-bearing RGB record and implements
the admitted Lanczos resize/crop and bicubic projection primitives. Container decoding remains
separate from MiniMax preprocessing policy.

Protocol v14 and the typed media request carry up to one `first` and one `last` image condition.
Duplicate roles, unsupported kinds, unreadable images and incompatible condition geometry fail
before publication. The native `yvex chat` client exposes `--first-image` and
`--last-image`; prompt contents remain opaque and select no execution policy.

Generic CUDA owners gained the reusable Qwen3-VL vision operations, Conv2D/Conv3D support needed by
the encoder, group normalization and GELU. The third admitted MiniMax family projection composes
the released processor, vision tower and Visual VAE encoder over those mechanisms. MiniMax-specific
roles, anchor times, layouts and posterior ordering remain family facts. Zero-condition T2VA skips
vision and encoding and retains its previous packed identity.

The condition-aware layout is:

```text
[text including visual tokens | condition-video | target-audio | target-video]
```

The joint Transformer validates tag and stream ownership independently, preserves exact row
partitions, and updates targets without making condition rows mutable.

## After

One product engine now accepts zero, first, last, or first-plus-last image conditions. Conditioned
requests execute image decode, the source processor, all 27 Qwen vision layers, deep-stack language
integration, Visual VAE encoding, immutable keyframe construction and condition-aware Omni before
the existing decoders and transactional publisher. T2VA skips the added stages and retains its
existing path. Valid multilingual input is normalized to NFC; malformed UTF-8 fails closed.

## Evidence

A real RGB fixture and Unicode prompt matched the upstream processor exactly for token IDs, token
classes, tags, positions and `grid_thw`. Patch preprocessing differed by maximum
`5.91389835e-8`. Vision merger relative L2 was `0.0431341` with cosine `0.999091`; deep-stack
relative L2 was `0.026254` with cosine `0.999658`; selected 50-layer conditioning relative L2 was
`0.00921249` with cosine `0.999958`.

The independent Visual VAE encoder fixture reported maximum posterior-moment error
`6.66379929e-5` and maximum sampled-normalized-latent error `0.00115990639`. First, last and
dual-anchor first-evaluation oracles all passed. Video relative L2 was respectively
`9.31833116e-5`, `3.51886138e-5`, and `6.64326257e-5`; audio relative L2 was
`5.72668571e-6`, `4.65925275e-6`, and `3.37019462e-5`. Existing T2VA, Visual VAE decoder and Audio
VAE evidence remained inside their accepted contracts.

### Product acceptance

One persistent engine generation completed four normal product requests under the unchanged
identity-bearing `interactive-preview-v1` YVEX policy. The prompts contained composed Unicode,
Chinese or Japanese text. Conditioned requests used two distinct real PNG inputs.

| Mode | Conditions | Wall time | AVI SHA-256 |
| --- | --- | ---: | --- |
| T2VA | none | 177.751307 s | `0aa30690c82d1eabe7c45d901170418640db094c846538838896bbf772279944` |
| First | first image | 190.494415 s | `46ef6c667d50c96ae15e3cab19e14a63b6ef34dd55b53f55ee87331e0de5b48c` |
| Last | last image | 185.529224 s | `805a7dacfe32e8b82e8aa954c77164c8fa507c141c042c1581d975982aac1b9d` |
| First + last | two distinct images | 187.360994 s | `e5b14902e8e99b2def45afe1543dc0f95f90e35cf9a3c48a634563c413e97e3a` |

Every output is 14,381,024 bytes. GStreamer independently discovered 124 RGB frames at 192x192
and 24 fps, stereo 32 kHz non-silent audio, and end of stream. No partial publication remained.
After all four requests the engine reported zero component host/device residency; normal stop
closed one model with zero active sessions.

The first diagnostic run exposed a generic listener-ownership fault: a second host contender that
failed the lock could unlink the active owner's socket during cleanup. `server.core` now unlinks a
socket only when that server owns the lock. A focused two-host regression proved the active socket
survives contender cleanup. The final persistent host remained reachable across all four accepted
requests and stopped cleanly.

## Remaining limitations

This boundary qualifies released FL2VA task selection and keyframe-conditioning semantics at the
bounded preview policy. It does not establish source-default/full-trajectory inference, practical
768p execution, useful visual quality, PyTorch seed equality, compressed-codec output, Ref2VA,
H3-Context-IR, H3-Regenerate-2K, release readiness or a performance benchmark. The historical
source-square tiled output remains negative behavior evidence and has not been promoted by these
bounded executions.

## Why it matters

The authenticated FL2VA package now represents the released checkpoint's four task forms rather
than only its text-only slice. A user can supply no image, a first image, a last image, or both
through typed product controls while the same engine generation executes the real multimodal
processor, vision tower, Visual VAE encoder and condition-aware Omni trajectory.

```text
progression_decision: proceed
downstream_safe: true for H3.CANONICAL.2 trajectory, canvas and duration work
downstream_consumer: full released production trajectory and practical geometry envelope
gate blockers: none inside H3.CANONICAL.1
boundary incompleteness: none
evidence gaps: no visual-quality evaluation or full-scale run was claimed by this boundary
deferred depth: released trajectory/defaults, practical 768p, duration and Ref2VA have later owners
optimization debt: generation latency remains measured but is not a correctness blocker
generalization debt: no second media family consumes typed first/last conditions yet
external blockers: license eligibility and closed hosted-system components remain external
required repairs: none
higher-capability non-claims: quality, release, Ref2VA, Context-IR and Regenerate-2K remain unproved
```
