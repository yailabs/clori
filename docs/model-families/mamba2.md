# Mamba2 source and recurrent-state boundary

Status: partial; source admission and CPU component evidence, not a launchable model.

The reference is `mistralai/Mamba-Codestral-7B-v0.1`, acquired through
`yvex model pull` at immutable Hugging Face revision
`4f086c08c1e0f07bdc50ca25125dbbf7475d21da`. The selected
`safetensors-source` representation contains three numbered shards and nine
sidecars: 14,574,191,162 acquired bytes, including 14,570,807,296 tensor bytes.
The alternative consolidated payload is not part of this acquisition.

The provider inventory now separates a standalone Safetensors candidate from
a complete numbered shard population in the same directory. Selection excludes
the other payload, including with broad include patterns. This is provisional
representation evidence, not proof that two representations are numerically
equivalent. Source admission still authenticates the index, provider hashes,
headers and exact tensor population. Incomplete or duplicate shard populations
fail closed. `model pull --resume --clear-stale-locks` exposes the existing
bounded stale-lock recovery; dry runs never delete locks. Resume retains
completed files; reuse of incomplete byte ranges depends on the provider CLI.

## Implemented owners

[The family interpreter](../../src/model/families/mamba2.c) authenticates the
configuration/params/tokenizer/generation metadata and derives a pure Mamba2
signature. It validates 579 BF16 tensors in both directions: three global
roles and nine per layer, across 64 layers. Required shapes, dtypes, layers
and source ranges are checked; unexpected attention tensors cannot be renamed
into an SSM role. Mamba1, hybrid declarations and inconsistent group/head
geometry are refused.

The acquired geometry is hidden 4096, intermediate 8192, vocabulary 32768,
128 heads of width 64, state dimension 128, eight groups and convolution
kernel four. These are source-derived facts, not runtime name switches.

[Semantic SSD geometry](../../include/yvex/internal/semantic_decoder.h)
describes projected gate/x/B/C/dt partitions, scalar decay, time-step bounds
and explicit gated-normalization policy. The
[portable CPU lowering](../../src/graph/state_space.c) performs an F32 serial
scan over caller-owned candidate state. Single-token execution consumes only
the previous state; it does not replay the prefix. This is not a chunk-parallel
SSD kernel or a CUDA implementation.

[Common sequence state](../../src/runtime/sequence_state.c) now consumes a
sealed F32 layout and transition identity instead of requiring a gated-delta
plan. It retains allocation, two-bank transactions, isolation, reset, cleanup
and resource summaries for both existing delta consumers and SSD fixtures.
No separate SSM session, allocator, runtime, resource ledger or KV cache exists.
Per layer, the reference geometry requires 40,960 convolution and 1,048,576
recurrent F32 values. Across 64 layers this derives 10 MiB convolution plus
256 MiB recurrence per committed bank, with an equal candidate bank. These
are derived full-model geometry, **not measured hosted-model allocations**.
CPU workspace is separately caller-owned and requires convolution-width F32
values; it is not persistent state.

## Evidence and promotion barrier

The normal unit catalog contains source-contract adversaries and scan versus
retained-step state/outputs, transactional abort after partial mutation,
session isolation, reset, cleanup and current/candidate byte checks. The
external [reference helper](../../tests/reference/selective_ssd.py) can compare
both final state classes and mixer output with Transformers `Mamba2Mixer`.
It only reads an already acquired local snapshot; Transformers is never a
production execution dependency. Tests with first-layer real BF16 tensors
expanded to F32 pass at absolute tolerance `3e-6` plus relative `3e-5`.
This does not qualify a complete block, LM-head logits, text, or all-layer
numerical behavior.

The acquired metadata contains unresolved execution-authority disagreements:

- `config.json` declares BOS/EOS/PAD 0/0/0; `generation_config.json` declares
  0/2/1; tokenizer vocabulary uses UNK/BOS/EOS 0/1/2 and declares no pad token.
  The two SentencePiece assets are byte-identical. No chat template is invented.
- `config.json` declares normalization before gating. The examined Transformers
  implementation uses gate-before-normalization with a global RMS reduction;
  the Mistral/mamba-ssm recipe uses grouped gate-before-normalization. The
  component oracle establishes the Transformers variant only. Source inspection
  preserves the conflicting declarations; it does not silently choose a
  whole-model numerical policy.

[The source promotion gate](../../src/graph/families/mamba2.c) therefore publishes
no executable descriptor. `yvex model prepare mamba-codestral-7b-v0.1` reports
exact role coverage and refuses READY. There is no complete artifact, physical
decoder plan, deployment profile, loaded Mamba engine or hosted chat evidence.
The remaining generic integration must admit an SSM-only decoder without a
mandatory dense FFN or nonzero rotary/KV workspace; the current decoder owners
still impose those requirements. It also needs complete artifact lowering,
tokenizer/output-policy adjudication, and all-layer/logit/session qualification.
An attention-shaped placeholder would misrepresent this boundary.

Protocol v20 and installed public ABIs are unchanged. The existing decoder
serialization remains unchanged: common state layouts are derived from its
authenticated transition geometry on import. A02-A09, YAI roles, performance
optimization and release benchmarks receive no support claim from this work.
