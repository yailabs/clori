# DeepSeek V4 Flash / DSpark Technical Record

Status: current family and target record

This record owns current DeepSeek V4 Flash family facts and the exact DSpark
target admitted by YVEX. It does not own macro gate state, operator procedure,
or release promotion.

## Identity and source

| Fact | Value |
| --- | --- |
| Family | DeepSeek V4 |
| Canonical target | `deepseek4-v4-flash-dspark` |
| Source repository | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Pinned revision | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| Conversation encoder | `encoding/encoding_dsv4.py` |
| Conversation encoder SHA-256 | `bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0` |
| Snapshot shape | 48 Safetensors shards; 72,317 indexed tensors |
| Target topology | 43-layer hybrid SWA/CSA/HCA decoder with mHC and MoE |
| Draft topology | five-position DSpark block conditioned by three target feature taps |
| Model maximum context | 1,048,576 positions |
| Hidden width / output vocabulary | 4,096 / 129,280 |
| Routed experts / selected per row | 256 / 6 |

The external acquisition record binds the exact revision, sidecars, tokenizer,
index, every shard header and payload identity, and the ordered aggregate
snapshot identity. YVEX source intake independently produces and consumes its
canonical `yvex.source_manifest.v3` payload manifest before compilation. The
operator record is acquisition evidence, not a substitute input schema; a
directory name, mutable branch, or local modification time is never source
identity.

The superseded `deepseek-ai/DeepSeek-V4-Flash` snapshot and
`deepseek4-v4-flash` target are not current aliases. Living product surfaces
refuse the old target spelling with one migration hint. Historical records and
Git retain its provenance.

## Target architecture

Layers 0 and 1 use sliding-window attention. Layers 2 through 42 alternate 21
compressed sparse-attention layers at ratio 4 with 20 heavy-compression layers
at ratio 128. SWA uses base RoPE without YaRN; compressed classes use the
versioned YaRN extension. HCA retains incomplete groups as raw local history
and composes raw and compressed representations without cross-representation
deduplication.

mHC represents four 4096-wide streams as one 16,384-wide residual state. Its
transitions carry 24 by 16,384 mixing geometry, 20 Sinkhorn iterations,
pre/post attention transforms, deferred feed-forward transforms, and final
collapse before RMS normalization.

Every target layer has one shared expert and 256 routed experts with top-6
selection. The first three layers use token-ID hash routing. The remaining 40
use a learned BF16 router, sqrt-softplus scores, correction bias,
deterministic no-aux top-k, and normalized route weights. Generic MoE,
runtime, and backend owners execute the family-selected schedule.

## DSpark architecture

DSpark proposes a block of five positions. The target execution plan captures
the four mHC streams after target layers 40, 41, and 42, averages each tap,
and projects the normalized features into three ordered draft stages. Draft
queries use noise token `128799` and mutually visible block positions.

The three draft Transformer/MoE stages are distinct from the public
configuration's `num_nextn_predict_layers = 1`. The bundled inference
configuration declares `n_mtp_layers = 3`, and the exact tensor geometry
establishes those three executable stages. Stage zero owns the main feature
projection and normalization. The final stage owns output normalization, the
rank-256 Markov projections, confidence projection, and the final hidden
combination. Target embedding and vocabulary output resources are shared by
identity rather than copied into a second model.

The Markov component is a low-rank token-conditioned vocabulary bias, not a
second persistent recurrent state. Confidence values are scheduling facts;
they never replace full-target verification or authorize publication.

## Tokenizer and output

The target has a 128,000-entry base tokenizer plus 1,283 added-token records
and an untied 129,280-entry output head. The tokenizer owner admits exact
prompt rendering, UTF-8 encode/decode, special/EOS classification,
incremental detokenization, and committed-token append semantics.

The source declares a one-million-token context contract. Hosted context
capacity is selected and admitted independently by the runtime; source
capacity is not an automatic runtime configuration or performance claim.

These facts are sealed once by the family projection into model-execution
descriptor schema v1. The descriptor also carries RoPE/YaRN, attention-class,
compression, indexer, mHC, FFN, output, proposal, feature-source, Markov,
confidence, special-token and persistent-state geometry. Common runtime,
server, protocol, CLI and generic backend code consume that descriptor rather
than duplicate these values.

The pinned conversation encoder owns three admitted modes. Chat/non-think emits
the assistant marker followed by `</think>`; think-high emits the marker
followed by `<think>`; think-max additionally prepends the encoder's exact
maximum-effort instruction. These source facts live in the model-family
conversation descriptor. Common runtime, protocol and adapters consume typed
policies and channels and contain no DeepSeek token literals.

In thinking mode, output before the exact source-authored `</think>` token is
explicit reasoning and output after it is final content. The tokenizer owner
consumes the delimiter and refuses an unfinished reasoning grammar. It does not
search disabled-mode prose or classify delimiter-looking text in ordinary
final content. DSML tool blocks are parsed only through the source grammar and
remain a third typed result. This is model-emitted text, not access to hidden
internal chain of thought.

Ordinary multi-turn prompts apply the encoder's `drop_thinking` behavior:
reasoning from assistant turns before the latest user turn is omitted. A
tool-enabled prompt disables that drop so reasoning, calls and ordered tool
results retain the continuity required by the official format.

## Coverage, transformation, and artifact

Exact source coverage reconciles 72,317 source tensors, 3,130 more than the
superseded snapshot inventory. The sealed Transformation IR has 1,409 terminal
descriptors: 1,328 target-trunk descriptors and 81 DSpark descriptors. Every
source tensor is required and mapped, explicitly shared, or rejected; scale
companions, expert coordinates, target taps, draft stages, and global shared
resources remain distinct.

The current bootstrap physical policy is
`deepseek-v4-flash-dspark-bootstrap-q2-v1`. It preserves the admitted mixed
IQ2_XXS/Q2_K decisions for the same target roles, not on an assumption that the
replacement checkpoint has identical payload bytes. The retained DS4
importance matrix remains bound to its predecessor source identity and is used
only as a bootstrap prior; it is not represented as DSpark calibration. New
draft norms, controls, feature and Markov projections, confidence tensors, and
other sensitive small roles use conservative exact, BF16, or Q8_0 storage.
Draft expert decisions are role-specific and cannot inherit an aggressive
low-precision default from their auxiliary scope.

The complete GGUF records source, logical model, transformation, physical
variant, target and draft role inventories, DSpark configuration, and exact
artifact identity. One runtime binding requires target execution, draft
execution, complete target verification, persistent target state, and bounded
draft workspace. Container validity or target-only opening alone does not
establish DSpark support.

## Hosted execution

One immutable runtime model owns both `target-only` and `dspark` execution
plans. One server session owns committed target state, token ledger,
transcript, incremental decoder and sampling state, plus bounded draft and
verification candidate state. No second process, model opening, tokenizer,
session registry, CUDA context, or output head is created for drafting.

In DSpark mode, proposals do not advance position, KV, transcript, usage, or
text. The complete target verifies the ordered candidate block and retains an
exact checkpoint after each target-authored row. Greedy mode
requires exact target-token equality; admitted stochastic mode uses
target-distribution-preserving accept/reject and residual sampling. The runtime
promotes only the accepted target-authored checkpoint without replaying
accepted rows, discards the rejected suffix, and publishes text after model,
token, decoder, and RNG state agree.

Target-only remains the explicit semantic reference and debug mode. An
explicit DSpark request fails closed when any draft tensor, plan, qtype,
workspace, backend capability, or policy requirement is absent. It never
silently falls back to target-only.

## Current capability

DeepSeek-V4-Flash-DSpark is the sole complete YVEX source-to-streamed-text vertical.
The hosted native, interactive, and bounded OpenAI-compatible paths
consume one target-verified runtime authority. Target-only and DSpark modes,
multi-turn reuse, cancellation, reset, and committed-only streaming are
implemented under private local protocol v9. The admitted tokenizer/prompt
profile also supports explicit reasoning high, maximum, and disabled policies
through separate reasoning, final, tool, and error channels.

## Explicit non-claims

This record does not claim:

- an optimized GB10 physical variant or a DSpark speedup;
- native MXFP4/NVFP4 Tensor Core execution;
- production load-aware confidence scheduling or continuous batching;
- complete accelerator residency or complete device-side stochastic
  sampling/tokenization;
- multi-device or distributed serving;
- model behavior or quality evaluation or quality parity;
- a public full-model benchmark;
- speculative support for another family;
- release qualification.

Current gate state is owned only by [`ROADMAP.md`](../../ROADMAP.md).
