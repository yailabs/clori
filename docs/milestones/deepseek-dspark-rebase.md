# DeepSeek V4 Flash DSpark Rebase

Status: accepted implementation-boundary contract

Milestone: `V010.REBASE.DEEPSEEK.DSPARK.0`

This contract replaces the earlier DeepSeek-V4-Flash checkpoint with the exact
DeepSeek-V4-Flash-DSpark source and admits speculative generation through the
same hosted runtime authority. Live state and dependency order remain owned by
[`ROADMAP.md`](../../ROADMAP.md).

## Source boundary

The sole admitted source is `deepseek-ai/DeepSeek-V4-Flash-DSpark` at revision
`62af8fffb2f7030cac4de2f0169f5b8d1101b646`. An external acquisition record
must prove the complete 48-shard snapshot, tokenizer and configuration
sidecars, reconciled Safetensors headers, and ordered source identity before a
complete artifact can be emitted. Source payloads and acquisition evidence
remain outside Git.

The current target ID is `deepseek4-v4-flash-dspark`. The retired
`deepseek4-v4-flash` spelling is a refusal with a migration hint, not an alias.
Historical evidence retains the former identity without keeping a second
current target.

## Required implementation

The DeepSeek family owner must project the complete 43-layer target and the
source-authored DSpark drafter, including:

- a five-position draft block with noise token `128799`;
- ordered target feature taps after layers 40, 41, and 42;
- three draft Transformer/MoE stages;
- rank-256 Markov conditioning;
- draft vocabulary logits and confidence logits;
- complete target verification and accepted-prefix semantics.

Every source tensor is mapped, explicitly shared, or refused. The sealed
Transformation IR and complete GGUF retain distinct target and draft scopes,
source companions, sharing, physical policy, and consumer requirements. The
bootstrap physical policy inherits role-level target decisions without
asserting payload equality or relabeling predecessor calibration, and uses
conservative precision for new draft control, projection, Markov, confidence,
and expert tensors.

One immutable runtime model owns target and draft plans. One session owns the
committed target state, token ledger, decoder and RNG state, plus bounded draft
workspace and candidate transactions. Candidate tokens are proposals only.
The complete target verifies them, and the runtime atomically commits and
publishes only the accepted target-authored prefix. Rejected suffixes never
enter KV, transcript, usage, or streamed output.

The same model admits explicit `target-only` and `dspark` modes. Requested
DSpark execution fails closed when its artifact, binding, backend, policy, or
workspace requirements are absent; it never silently falls back to
target-only.

## Product and protocol boundary

The server remains the sole model, worker, session, KV, cancellation, and
telemetry authority. Native and OpenAI-compatible streams publish only
committed text, and OpenAI completion usage excludes proposals. Protocol v5
carries the selected generation mode, speculative lifecycle events, and exact
proposal, verification, acceptance, rejection, timing, and commit facts.

The normal console remains compact. Detailed cycle evidence belongs to human
trace and JSONL trace rather than one line per candidate token.

## Acceptance evidence

Closure requires:

- zero-gap source, architecture, role, Transformation IR, quantization, GGUF,
  artifact, materialization, and runtime-binding admission;
- target-only hosted generation through native, REPL, and OpenAI projections;
- real DSpark proposals from checkpoint tensors and complete target
  verification;
- exact greedy equality with target-only generation from the same initial
  state;
- target-distribution-preserving admitted stochastic semantics;
- at least one measured multi-token accepted prefix on a predeclared bounded
  corpus;
- transactional partial acceptance, cancellation, reset, and multi-turn reuse;
- no unverified native or HTTP publication;
- full deterministic, sanitizer, no-`nvcc`, CUDA, package, documentation, and
  repository validation;
- one controlled switchover to the new external artifact and binding.

The superseded heavy artifact and binding may be retired only after the new
hosted path passes. Compact historical identity records remain.

## Non-claims

This boundary does not establish an optimized GB10 physical variant, native
MXFP4/NVFP4 execution, production load-aware confidence scheduling,
continuous batching, multi-device serving, model-quality parity, a public
benchmark, release qualification, speculative support for another family, or
the final product architecture.
