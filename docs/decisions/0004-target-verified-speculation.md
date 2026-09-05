# 0004 — Target-verified speculative generation

Date: 2026-08-02
Status: accepted

Protocol versions below record the original decision, not today's wire
contract; see the [current protocol](../contracts/local-protocol.md).
The accepted target-verification invariant remains current.

## Context

The exact DeepSeek-V4-Flash-DSpark checkpoint adds a source-authored drafter to
the 43-layer target. The drafter consumes ordered target feature taps, proposes
five-position blocks, adds rank-256 Markov conditioning, and produces confidence
facts. Those proposals are useful only when the complete target verifies them
without weakening target sampling semantics, session state, usage, or streamed
publication.

YVEX already has one daemon-owned immutable runtime model and isolated mutable
sessions. Treating the drafter as a second model process, session registry, KV
owner, or provider path would duplicate authority and make target/draft state
publication ambiguous. Treating a candidate as several independent one-token
commits would make rejected-suffix rollback depend on undoing visible state.

## Decision

One admitted artifact and runtime binding contain the target, drafter, shared
resources, and complete-target verification requirements. One immutable
`yvex_runtime_model` owns explicit target-only and speculative execution plans;
the DeepSeek family adapter supplies DSpark as the admitted speculative
implementation.
One server session owns committed target truth and bounded proposal and
verification candidate state. No draft process, model opening, tokenizer,
output head, session registry, or backend context is created.

The DeepSeek family adapter owns target feature taps, draft-stage geometry,
noise-token use, Markov composition, and checkpoint-specific acceptance facts.
The common runtime owns proposal, complete-target verification, stochastic
target-distribution preservation, accepted-prefix transaction, cancellation,
accounting, and publication. Common owners never infer DSpark structure from a
family name or tensor string.

A proposal is workspace, not output. It cannot advance committed target state,
token ledger, transcript, decoder, usage, or text. The complete target verifies
the ordered block. Greedy verification requires exact target-token equality;
admitted stochastic verification uses accept/reject and residual sampling from
the target and draft distributions.

All persistent participants prepare one bounded accepted prefix before any of
them publishes it. The commit covers target sequence state, token ledger,
incremental decoder, generated text, target/draft RNG state, position, stop
classification, and turn identity. Rejected suffix state is discarded.
Cancellation before commit leaves the preceding state unchanged; cancellation
after commit reports the exact committed prefix.

Private local protocol v5 carries mode, proposal, verification, acceptance,
rejection, timing, policy, and committed-only usage facts. Native and HTTP/SSE
surfaces receive only target-verified committed fragments. The OpenAI profile
version remains unchanged because its external JSON/SSE contract does not
change.

The functional bootstrap uses fixed bounded verification and represents
confidence exactly. Production load-aware scheduling, batched verification
optimization, native FP4 execution, and performance claims remain later
owners.

## Consequences

- Target-only generation remains an explicit semantic oracle and debug mode.
- A DSpark request fails closed when any artifact, binding, draft, verifier,
  qtype, workspace, backend, or policy requirement is unavailable.
- Proposal counts never inflate generated-token or OpenAI completion usage.
- The runtime can later lower block verification more efficiently without
  hiding candidate extent or changing commit semantics.
- Reset rebuilds clean target and draft session state while retaining the one
  resident immutable model.
- Correct speculative execution does not by itself establish acceleration,
  model-quality parity, benchmark evidence, or release qualification.

## Alternatives considered

A separate draft-model process was rejected because it duplicates model and
session authority, resource lifecycle, and failure coordination. Direct
drafter-to-stream publication was rejected because candidate retraction would
violate committed-output semantics. Token-ID equality for stochastic sampling
was rejected because it changes the target distribution. Silent fallback to
target-only was rejected because it would misreport the selected execution
mode. Counter-based rollback after per-token publication was rejected because
it cannot restore all target, decoder, RNG, and transcript state exactly.
