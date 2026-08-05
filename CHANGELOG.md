# Changelog

All externally meaningful YVEX changes are recorded here. The project is not
yet released; entries remain under **Unreleased** until release qualification
and a version tag are accepted.

This changelog follows the spirit of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
without treating every internal milestone, test, or refactor as a public
change. Git history preserves implementation chronology.

## Unreleased

### Added

- DeepSeek-V4-Flash-DSpark as the sole current DeepSeek source target, with
  target-only reference generation and target-verified speculative generation
  in the same resident runtime model and session authority.
- Complete daemon-backed DeepSeek-V4-Flash source-to-streamed-text execution
  through native and bounded OpenAI-compatible local surfaces.
- Exact server-owned multi-turn sessions with committed-prefix reuse,
  cancellation, partial-progress truth, and one persistent model lifecycle.
- Registry-driven command discovery, advanced help, JSON discovery, and Bash,
  Zsh, and Fish completion.
- A daemon-backed `yvex>` console with composed attachment state, live prefill
  progress, direct streamed output, typed turn metrics, registry-derived slash
  completion, semantic watch/human trace, and clean Ctrl-C/Ctrl-D handling.
- Source-authored explicit-reasoning controls and a typed reasoning stream that
  remains separate from final text and never infers hidden chain of thought.
- A source-derived model-execution descriptor and identity-bound hardware,
  workload, per-state capacity, page-geometry and phase-roofline contracts for
  GB10 execution planning.
- An operator-reachable, identity-bound CUDA bandwidth fixture that records raw
  streaming, D2D and coherent-host samples instead of treating peak hardware
  bandwidth as measured evidence.
- An identity-bound phase roofline ledger on real CUDA generation. Audit and
  JSON output distinguish measured facts from unavailable ones. Target-only
  prefill/decode and output projection publish exact active-weight and launch
  counts. Target-only transformer phases now publish exact H2D/D2H/D2D
  movement and synchronization, including feature and status transfers;
  output projection publishes its exact H2D/D2H movement and synchronization.
  DSpark draft/verification now publish exact launch, H2D/D2H/D2D movement and
  synchronization facts across their transformer and output-head work.
  Accepted-prefix promotion publishes exact state-residency H2D,
  synchronization and zero kernel/D2H/D2D facts; repeated occupancy samples
  use a checked work-weighted mean instead of additive accumulation. CUDA
  output-head rows now publish complete compulsory weight, activation,
  temporary and zero-state byte facts, giving that phase a real memory lower
  bound without estimating traffic from allocation capacity. Compatible
  width-N CUDA rows now share Q8 activation preparation and one encoded-head
  execution, while the ordered logits result owns one aggregate physical-facts
  record and incompatible inputs retain an explicit row-local fallback.
  Device-native output batches now publish contiguous identity-bound logits
  row views for downstream CUDA selection without allocating or downloading a
  complete host vocabulary buffer.
- Production CUDA MoE now routes a complete compatible row batch, orders its
  row/expert pairs by expert, executes resident routed and shared packs through
  one width-N backend transaction, and derives workspace from admitted layer
  qtypes and row capacity. The token-local CPU/CUDA implementation remains the
  explicit portable audit/reference oracle.
- Target-only production stochastic sampling now filters and selects directly
  from resident CUDA logits. The host publishes the deterministic PCG advance
  only after cancellation-safe validation, while audit/forensic and stochastic
  DSpark retain the complete-distribution host reference.
- Production greedy DSpark verification now projects its complete target row
  batch to CUDA-resident logits, selects every row through one width-N argmax
  launch and one synchronization, and transfers only bounded aggregate facts. The
  greedy draft path now keeps the shared-head base rows resident, fuses each
  encoded Markov projection with its base row, and transfers only the selected
  token and bounded status. Greedy CUDA now gathers the encoded Markov embedding
  row directly from admitted residency and uploads only its bounded row ID. Its
  normalized and pre-normalized drafter rows now remain device resident through
  output-head and confidence projection; confidence consumes the resident hidden
  and Markov views directly and transfers one scalar result. The portable host
  distribution and confidence paths remain the CPU, audit/forensic and stochastic
  oracles.
- Production CUDA target-feature capture now averages source-selected mHC
  residual streams into bounded host evidence and a transaction-owned
  token-major device directory. Production feature projection consumes that
  directory without re-upload, then batches the rows through the resident
  encoded matrix and CUDA RMSNorm with exact physical accounting. The normalized
  rows feed the draft core through an identity-bound device view, and the
  producer-owned feature digest removes the consumer's duplicate full-row scan.
  The CPU projector and bounded host materialization remain the audit/reference
  oracle; host-free feature evidence, Markov embedding decode and confidence
  execution remain explicit debt. Speculative prefill now contributes its merged
  target, projection and draft-core physical facts to the phase roofline ledger.
- The production CUDA transformer final stage now preserves an optional
  pre-normalized BF16 row in device storage before applying output RMSNorm.
  DSpark drafting materializes only that bounded hidden row instead of the
  expanded residual streams and no longer recomputes the final stage on the
  host; full evidence retains the independent CPU oracle.
- Compulsory memory accounting now has one transactional internal fact owner.
  CUDA embedding, attention, MoE and final projection contribute measured or
  explicitly missing operations through transformer and decode aggregation;
  target generation publishes a complete phase memory mask only when every
  contributing operation is measured. Output-head reporting uses the same
  representation instead of retaining duplicate mutable fields.

### Changed

- CUDA kernel admission now binds and atomically owns multiple independently
  compiled manifest-owned PTX/native modules under kernel-bundle identity v3;
  the routed/shared MoE kernel family no longer shares one monolithic CUDA
  translation unit with general device kernels.
- CUDA attention graphs now refresh mutable state-bank inputs before capture
  and replay, allowing allocation-stable graphs to survive committed-state
  promotion without restoring stale state or recapturing each turn.
- Production attention graph replay no longer resolves a CUDA timing event in
  every layer. Mandatory completion synchronization remains fail-closed;
  audit and forensic evidence retain explicit per-layer device timing.
- Target-only CUDA generation now selects admitted full attention graphs through
  the existing compiled-profile contract; CPU and DSpark retain explicit eager
  reference execution, and compatible shape changes preserve cached graph
  executables.
- Consolidated the product topology to the public `yvex` command and the
  long-lived `yvexd` host; the OpenAI-compatible listener now runs inside the
  daemon.
- Replaced implementation-era top-level command buckets with the canonical
  `compile`, `artifact`, `inspect`, `execute`, `profile`, and `system`
  projections.
- Advanced the private local protocol to version 4, separating selected model
  configuration from the live runtime model and removing false artifact/model
  facade operations.
- Advanced the private local protocol to version 5 for typed speculative-cycle,
  accepted-prefix, and committed-only usage facts; version 4 is refused.
- Advanced the private local protocol to version 6 for exact partial-turn and
  explicit-reasoning facts; version 5 is refused.
- Made hosted startup registry-first: `model list` reports complete startup
  profiles, `model select NAME` resolves one profile without path flags, and
  `runtime start` opens the selected model without environment variables.
- Made the human terminal surface compact and semantic: startup announces the
  selected model before admission; REPL attachment facts and commands use a
  stable vertical hierarchy while turn metrics remain compact; TTY color
  respects `NO_COLOR`; Ctrl-L clears and redraws active input; and categorized
  operational watch separates signal from connection churn and detailed
  trace/profile output.
- Reorganized documentation by authority and lifecycle, with canonical
  terminology, family records, contracts, operator procedures, frozen audits,
  and validated migration paths.
- Separated physical execution decisions and evidence depth from artifact
  identity, promoted verified speculative state without accepted-token replay,
  and made CUDA capacity admission execution-shape specific.
- Separated artifact capability from runtime-profile readiness in the local
  model registry, eliminating false metadata drift without changing artifact
  or binding identity.
- Advanced runtime bindings to v8 when they carry sealed model geometry while
  retaining v7 reference bindings, and made startup refuse insufficient model
  residency memory before opening the complete artifact.
- Made explicit SM121 builds admit an identity-bound native CUBIN on GB10 while
  retaining portable PTX as a separately identified fallback class.

### Removed

- Retired the separate `yvex-dev` and `yvex-openai` product executables.
- Removed the old top-level `evidence`, `graph`, `quant`, `source`, `tensor`,
  and `tokenizer` command namespaces; migration hints do not execute hidden
  aliases.

### Security

- The hosted protocol and OpenAI-compatible endpoint remain local-only and
  fail closed. Authentication, TLS, CORS, and remote exposure are not part of
  the current compatibility profile.
