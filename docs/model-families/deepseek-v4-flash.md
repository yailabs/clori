# DeepSeek-V4-Flash Technical Record

Status: current family record

This record owns DeepSeek-V4-Flash family and target facts. It does not own
macro gate state, operator procedures, or release promotion.

## Identity and source

| Fact | Value |
| --- | --- |
| Family | DeepSeek V4 |
| Canonical target | `deepseek4-v4-flash` |
| Source repository | `deepseek-ai/DeepSeek-V4-Flash` |
| Pinned revision | `60d8d70770c6776ff598c94bb586a859a38244f1` |
| Runtime class | hybrid SWA/CSA/HCA decoder with mHC and MoE |
| Main layers | 43 |
| Auxiliary topology | one distinct MTP descriptor, not executed by the current product path |

Strict source verification retains configuration, tokenizer sidecars, 46
safetensors headers, exact payload ranges, and snapshot identity. The family
owner consumes those typed facts and does not reopen source configuration or
reclassify tensor names.

## Architecture

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

## Mixture of experts

Every main layer has one shared expert and 256 routed experts with top-6
selection. The first three layers use token-ID hash routing. The remaining 40
use a learned BF16 router, sqrt-softplus scores, correction bias,
deterministic no-aux top-k, and normalized route weights.

The runtime executes only selected routed expert subviews plus the shared
expert. The family defines routing and composition; common MoE, runtime, and
backend owners retain execution, memory, transaction, and cleanup mechanisms.

## Tokenizer and output

The target has a 128,000-entry base tokenizer plus 1,283 added-token records
and an untied 129,280-entry output head. The current tokenizer owner admits
exact prompt rendering, UTF-8 encode/decode, special/EOS classification,
incremental detokenization, and committed token append semantics.

The source declares a one-million-token context contract. Current hosted
context capacity is selected and admitted independently by the runtime; source
capacity is not an automatic runtime configuration or performance claim.

## Mapping and artifacts

Exact coverage reconciles 69,187 source requirements. The logical GGUF map
produces 1,360 terminal descriptors: 1,328 trunk descriptors follow the pinned
llama.cpp mapping at `e920c523e3b8a0163fe498af5bf90df35ff51d25`, while 32
MTP descriptors use `yvex.mtp.v1` because the pinned external converter omits
MTP.

The mapping preserves FP8 E4M3 weights and UE8M0 companions, aggregates 256
ordered FP4 expert pairs into the required MXFP4 storage geometry, and records
checked I64-to-I32 hash-table conversion.

Complete source-faithful, Q8_0/Q2_K, and mixed IQ2_XXS/Q2_K physical variants
have been emitted and admitted outside Git. Each has a distinct physical
variant, artifact, materialization, and runtime-binding identity. None is the
release variant merely because it exists or executes.

## Runtime and backend coverage

The current common runtime opens one exact artifact and binding, builds
process-lifetime host residency, and creates isolated server-owned sessions.
The production path includes:

- exact prompt tokenization and prefix comparison;
- suffix prefill and family-correct persistent state;
- 43-layer attention, mHC, and MoE composition;
- final norm and full 129,280-coordinate output-head logits;
- greedy and admitted stochastic sampling;
- sampled-token decode feedback;
- typed EOS, stop, cancellation, and partial progress;
- incremental detokenization and committed streaming;
- retained exact multi-turn sessions;
- native and bounded OpenAI-compatible application projections.

CPU and the admitted mixed GB10 CUDA path consume the same logical owners. The
GB10 path executes the backbone and output head on CUDA without CPU numerical
fallback. Tokenizer work, sampling, protocol handling, and orchestration remain
host-owned. Host-addressable or unified placement is not called complete device
residency.

## Current capability

DeepSeek-V4-Flash is the sole complete YVEX source-to-streamed-text vertical.
This establishes model-backed generation through the admitted local product
path. Startup optimization and bounded warm profiling are accepted, while warm
decode optimization remains open.

## Explicit non-claims

This record does not claim:

- MTP or speculative execution;
- complete accelerator residency or device-side sampling/tokenization;
- every context, sampling policy, concurrency level, or physical variant has
  the same performance;
- public/remote security, authentication, TLS, multi-model hosting,
  continuous batching, or distributed serving;
- model behavior or quality evaluation;
- a release-path full-model benchmark;
- release qualification.

Current gate state is owned only by [`ROADMAP.md`](../../ROADMAP.md).
