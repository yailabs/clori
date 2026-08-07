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
- Exact source-authored chat/non-think, think-high and think-max conversation
  encoding, including tool continuity and drop-thinking multi-turn behavior.
  Typed reasoning, final, tool and error streams remain separate; REPL and raw
  output preserve their respective terminal and byte contracts, and the OpenAI
  v2 projection exposes explicit model output as `reasoning_content` without
  inferring or exposing hidden chain of thought.
- A source-derived model-execution descriptor and identity-bound hardware,
  workload, per-state capacity, page-geometry and phase-roofline contracts for
  GB10 execution planning.
- Stable virtual host-state spans whose physical pages are committed from the
  admitted per-class capacity plan, with bounded candidate/committed preflight,
  exact reset release, and refusal before an over-budget layer becomes visible.
  CUDA state residency and deep-context product qualification remain open.
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
  qtypes and row capacity. Expert scores are evaluated cooperatively, while
  source tie-breaking and weight accumulation remain ordered. Expert-major
  pair counts and stable pair emission are likewise parallel rather than owned
  by one device lane. Selected routes and weights remain device-local;
  each layer enqueues only bounded status and unique-expert facts, and one
  transformer-stack completion validates them and reconstructs exact active
  bytes. A proved final-stage session-stream barrier avoids a redundant wait.
  Immediate and token-local CPU/CUDA execution remain the explicit portable
  audit/reference oracles. Resident-weight oracle execution no longer reserves
  an unused expert staging range, and full forensic evidence disables Q8
  activation compression so live tests distinguish its tighter numerical
  contract from the admitted production approximation.
- CUDA mHC envelope gates, combination rows and Sinkhorn row/column passes now
  execute across their independent stream lanes. Ordered reductions and FP64
  source transforms remain intact, while BF16 residual-square accumulation no
  longer consumes FP64 issue bandwidth; the full 43-layer CPU/CUDA oracle stays
  bit-exact.
- CUDA DeepSeek RMSNorm now reduces independent BF16 residual squares across
  the block instead of serializing FP64 accumulation through one lane. The
  inverse, encoded scale and BF16 publication contracts remain unchanged, and
  the full attention oracle remains bit-exact.
- CUDA DeepSeek attention now uses one stable online-softmax sweep over visible
  local and compressed history. Each source-ordered dot product is consumed
  once, with prior value lanes renormalized when the running maximum changes;
  the complete CPU/CUDA oracle remains inside its admitted numerical contract.
- Normal production CUDA attention now stages completed non-prefix sequence
  state directly into a session-owned candidate bank through ordered D2D
  copies. Commit flips the committed bank, while abort and later reuse clone
  the exact committed state. The retained host oracle no longer causes a
  duplicate state upload; prefix-addressable speculation and audit/forensic
  paths keep their explicit host materialization. Logical state identity is
  token- and position-derived, independent of whether the same prefix arrived
  through target-only execution, verification or accepted-prefix promotion.
- Short CUDA qtype rows now form geometry-selected two-, four- or eight-lane
  groups across Q8_0, Q2_K, IQ2_XXS and MXFP4 activation dots. Only integer
  terms are redistributed; every encoded block is reconstructed before the
  original floating-point reduction order, preserving exact output while
  filling lanes that the row geometry would otherwise leave idle.
- IQ2_XXS CUDA sign reconstruction now derives the encoded parity bit with the
  hardware population-count primitive instead of a serial bit-shift loop,
  preserving exact qtype, attention and grouped-MoE results.
- Wide F32 projections with at most 32 output rows now assign one CUDA block to
  each row/input pair instead of leaving a single warp to traverse 16K source
  values. Encoded qtypes retain their warp-owned geometry, while the complete
  DeepSeek attention oracle continues to admit the optimized reduction.
- Multi-row CUDA qtype projections now group compatible input rows around the
  same encoded matrix row and tile wider batches explicitly. Output-head and
  verification launches therefore preserve row locality without changing the
  existing warp arithmetic or padding logical work.
- Target-only production stochastic sampling now filters and selects directly
  from resident CUDA logits. The host publishes the deterministic PCG advance
  only after cancellation-safe validation. Production stochastic DSpark keeps
  adjusted draft and target-verification rows resident, performs p/q acceptance
  and residual correction on CUDA, then publishes RNG and state only after the
  bounded facts reseal through the canonical acceptance identity. CPU and
  audit/forensic profiles retain the complete-distribution host reference.
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
  residual streams directly into a transaction-owned token-major device
  directory and transfers only bounded status. Feature projection consumes that
  directory without re-upload, executes the resident encoded matrix and RMSNorm
  without downloading the normalized rows, and feeds the draft core through an
  identity-bound device view. Its unused host feature workspaces are no longer
  allocated. Prefix-specific semantic identities bind device-only candidate and
  promoted rows without a full-array host scan. CPU and audit/forensic profiles
  retain explicit host feature oracles. Speculative prefill
  contributes its merged target, projection and draft-core physical facts to the
  phase roofline ledger.
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
- Physical-variant research can execute one exact tensor decision from a sealed
  quantization plan and report bounded reconstruction evidence without emitting
  a complete artifact. Source-native MXFP4 weights now use the shared Q8_K CUDA
  activation path in dense, attention and grouped-MoE execution, preserving the
  portable reference while candidate policies are filtered by role.

### Changed

- Generation plan ABI v5 now binds the compiled workload-profile identity, so
  CUDA phase-roofline evidence validates against its actual workload instead
  of being rejected against the distinct per-request profiling identity.
- Production CUDA attention now keeps completed activation rows in the
  caller-owned device output; only audit and forensic profiles materialize the
  duplicate numerical host rows.
- CUDA kernel admission now binds and atomically owns multiple independently
  compiled manifest-owned PTX/native modules under kernel-bundle identity v3;
  the routed/shared MoE kernel family no longer shares one monolithic CUDA
  translation unit with general device kernels.
- CUDA attention graphs now refresh mutable state-bank inputs before capture
  and replay, allowing allocation-stable graphs to survive committed-state
  promotion without restoring stale state or recapturing each turn.
- Production attention graph pieces now borrow the session execution stream
  and defer to one fail-closed layer publication barrier instead of
  synchronizing every captured piece. Audit and forensic timing retains
  isolated immediate graph completion.
- CUDA backends now own one non-blocking execution stream per session. Eager
  attention and width-N MoE completion synchronize that stream instead of the
  complete CUDA context, while legacy Driver configurations retain an
  explicitly accounted context-wide fallback.
- Greedy and stochastic CUDA selection now enqueue bounded result downloads
  behind their kernels and complete only the owning session stream. The phase
  ledger accounts stream and device-wide synchronization classes through one
  checked aggregate without changing persisted, wire, or public C contracts.
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
- Advanced the private local protocol to version 7 for typed reasoning/final/
  tool/error channels and separate reasoning/final timing and token metrics;
  every non-v7 peer is refused.
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
