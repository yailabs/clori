# Model-Derived GB10 Execution

Status: normative implementation-boundary contract

Milestone: `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0`

Status source: [`ROADMAP.md`](../../ROADMAP.md)

This boundary specializes the accepted DeepSeek-V4-Flash-DSpark vertical for
GB10 without creating a second runtime or making the common engine depend on
DeepSeek constants. Live state and dependency order belong only to
[`ROADMAP.md`](../../ROADMAP.md).

## Contract evolution

Contract versions change only when an admitted fact cannot be represented by
the current persistence, wire, layout, or binary boundary. The frozen
[contract matrix](../audits/gb10-optimization-691814/contracts.tsv) records the
reader, writer, migration, and test consequences for every considered change.

Runtime binding v8 is the first admitted persisted change. It authenticates the
source-derived model-execution descriptor that v7 cannot carry. V7 remains
readable as the retained reference profile; v8 is emitted beside it. The
source-authored conversation product path earns local protocol v7, provider
request/wire v2, tokenizer plan v3, tokenizer provider result v2 and OpenAI
compatibility profile v2. The installed server-construction API and public
declaration count remain unchanged. Runtime events remain schema v3, Physical
Execution IR and compiled profiles remain schema v1, and generation remains
ABI v4. Every admitted change has a concrete fact and compatibility rule in the
contract matrix; the size of this milestone alone earns no version bump.

An implementation-discovered identity change is admitted beside that frozen
entry audit:

| Contract | Current | Admitted | New fact | Incompatibility | Old behavior | Rule | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kernel-bundle identity | v2 | v3 | ordered set of independently compiled modules | semantic identity derivation only; no wire or persisted layout | v2 hashes one image and cannot identify the module set | full rebuild admits all manifest-owned modules atomically; model artifacts do not migrate | PTX/native admission, missing-symbol rollback, checked unload retry, identity mutation |
| Launch-graph identity | v2 | v3 | owned versus borrowed launch-stream policy | semantic identity derivation only; no wire or persisted layout | v2 cannot distinguish an isolated graph stream from the session stream | rebuilt graph registries recapture under the selected stream policy; no model, binding or profile migration | identity mutation, isolated/shared registry separation, capture and warm replay |
| Graph-executable identity | v1 | v2 | owned versus borrowed launch-stream policy | semantic identity derivation only; no wire or persisted layout | v1 authenticates device and executable topology but omits completion ownership | rebuilt executables are admitted beneath launch-graph v3; old in-process objects are never mixed | exact identity, shared-stream capture/replay, checked teardown |
| CUDA graph execution internal ABI | v1 | v1 | explicit stream and deferred-completion flags plus returned completion facts | in-process function and result layout only; no independently deployed reader or writer | old callers pass a Boolean timing policy and always wait on a graph-owned stream | complete rebuild; isolated immediate execution remains the audit path; no persistence, wire, public API or profile migration | invalid flag combinations, zero local waits, scoped completion, eager numerical parity |
| Physical-facts internal ABI | v1 | v1 | stream and device-wide synchronization deltas enter one checked aggregate | in-process function signature only; the stored fact remains one aggregate | old objects are never mixed with rebuilt objects; old callers supplied one combined count | complete rebuild; no persistence, wire, layout, or identity migration | class-sum overflow rollback, runtime execution, sampling and speculation accounting |
| Width-N MoE internal ABI | v1 | v1 | deferred stack completion plus bounded per-layer status, unique-expert and active-byte factors | in-process result and operation-table layout only; no independently deployed reader or writer | old readers require completed rows and route arrays; old writers synchronize and materialize them per layer | complete rebuild; immediate execution remains the audit oracle; no persistence, wire, public API or profile migration | immediate/deferred output and active-byte parity, untouched route sentinels, bounded D2H, proved-barrier accounting, sync fault refusal |
| Sampling-transaction internal ABI | v1 | v1 | the existing selection call accepts an optional staged RNG transaction; its test-only result validator is retired | in-process signature and private transaction layout only; result layout is unchanged | old callers select directly or obtain uniform values and select outside the sampling owner | complete rebuild; direct callers pass no transaction; outer transactions retain prepare/publish/abort authority; no persistence, wire, public API or profile migration | CPU/CUDA abort and exact retry, commit-only sample and RNG accounting, stale-base refusal, bounded device result |
| DSpark target-anchor internal ABI | v1 | v1 | one typed sampling source and its staged selection replace mandatory complete-probability materialization | in-process helper and result-field naming only; no persisted, wire, state-layout or public incompatibility | old rebuilt callers materialize one complete probability row before target selection | complete rebuild; CPU and evidence-bearing paths use the host sampling oracle, production CUDA consumes resident logits; explicit full distributions remain forensic operations | runtime sampling/speculation/generation, CUDA bounded selection, full-array production guard, transaction abort and commit |
| Stochastic-speculation internal ABI | v1 | v1 | resident draft/target row directories and bounded p/q acceptance facts | in-process operation table and helper signatures only; no persisted, wire or state-layout incompatibility | old objects are never mixed with rebuilt objects; the retained oracle expects host probability rows | complete rebuild; production CUDA uses the device operation, while CPU and audit/forensic profiles retain the oracle; no artifact, binding, protocol, event, profile or public API migration | every accepted prefix, residual and bonus agreement, workspace and malformed-token refusal, canonical identity reseal |

The later source-authored conversation gate earned these product-boundary
changes. Reader and writer behavior remain separate because compatibility in
one direction does not imply compatibility in the other.

| Contract | Current | Admitted | New unrepresentable fact | Incompatibility | Old reader | Old writer | Migration or side-by-side rule | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conversation descriptor | absent | v1 | source prompt syntax, reasoning/tool grammar and drop-thinking policy | new internal semantic identity | no reader | no writer | seal the family descriptor from the pinned source and expose typed facts | source digest, prompt hashes, mutation, grammar |
| Tokenizer plan | v2 | v3 | reasoning token IDs, source identity and conversation capabilities | in-process layout and semantic identity | rebuilt consumers do not mix v2 objects | cannot publish conversation facts | complete rebuild; family-owned retained-binding projection | seal, missing/mutated capability refusal |
| Tokenizer provider result | v1 | v2 | separate reasoning, final and tool output | in-process result layout | rebuilt consumers do not mix v1 objects | cannot publish typed channels | complete rebuild; no persistence | exact grammar, arbitrary-text non-inference, incomplete refusal |
| Provider request/wire | v1 | v2 | assistant reasoning, policy/drop facts, presence facts and ordered multiple tool calls | installed C layout, request identity and serialized bytes | old binary rejects or cannot represent v2; current reader admits both | v1 emits only disabled reasoning and one call | retain v1 within exact limits; new facts require v2 | v1/v2 clone, seal, roundtrip, truncation, malformed refusal |
| Local protocol | v6 | v7 | typed error channel and separate reasoning/final metrics | private Unix framing and payload | every non-v7 peer fails handshake | v6 cannot carry the new terminal facts | atomic daemon/client cutover; no compatibility decoder | message/status roundtrip and non-v7 refusal |
| OpenAI compatibility | v1 | v2 | reasoning policy, `reasoning_content`, reasoning SSE and multiple calls | documented HTTP JSON profile | v1 clients can consume shared additive fields | cannot express the new contract | v2 documents the YVEX-specific projections | Chat/Responses, SSE, tools, usage, cancellation |
| Public server construction | v1 | v1 | none; no concrete scheduler-construction consumer exists | none | unchanged | unchanged | current entrypoints converge on new internals | declaration count and ABI guard |

## Planning authority

Verified checkpoint and family facts seal one immutable model-execution
descriptor. Hardware probes seal a separate hardware profile; operator intent
seals a workload profile. Capacity and execution shapes are compiled from
those identities rather than from model-name branches or common-runtime
literals.

Capacity distinguishes model maximum, execution maximum, per-session and
per-request context, pooled state, candidate reserve, prefix budget, logical
batch, attention microbatch, MoE rows, output rows, workspace, scheduler and
system reserve. Page geometry is selected independently for each state class.
Its candidates derive from representation blocks, alignment, kernel tiles,
page-table cost, fragmentation, copy-on-write and promotion granularity.

Model residency is preflighted after bounded binding admission and before
artifact open or materialization. The admitted resident payload must fit both
the caller's host budget and currently available system memory while preserving
at least 8 GiB. Refusal reports configured or available bytes against required
bytes before the model candidate mutates artifact or residency state.

## Causal optimization

The phase roofline ledger is the priority authority. For prefill layer, decode
layer, verification sweep, draft sweep, output head, promotion and batched
decode it binds active weight, state, activation, temporary and transfer bytes;
launches, synchronizations, occupancy, duration, work and committed tokens.
Measured memory lower bounds and remaining headroom decide optimization order.
Attention, MoE and output-head dependencies do not prescribe that order.

The v1 ledger accepts partial measurements with explicit fact masks. It keeps
all causal phase slots, exposes missing phases and facts, and marks rankings
provisional until active-byte and movement evidence can establish a roofline.
The additive internal change does not bump a persisted or wire contract and
keeps the original zero-mask v1 writer representation readable.

The production CUDA generation path now records partial phase evidence without
depending on trace verbosity and exposes it through the finite offline
generation operator. Duration, work and committed-token accounting is live.
Target-only prefill/decode aggregate compulsory memory through one checked fact
type. Device-native embedding, attention, MoE and final projection contribute
exact active weights, touched state, external activations and actual temporary
workspace; measured and missing operation counts prevent a partial aggregate
from acquiring the complete memory mask. Transactional overflow preserves the
prior aggregate. These phases also report exact launch facts.
Output projection reports encoded weights, compulsory input/output activation
spans, actually allocated temporary spans, exact zero persistent-state bytes,
movement, launches and synchronization. Those facts complete its memory lower
bound without substituting allocation capacity or workspace peaks. Target-only
transformer phases additionally report exact
H2D/D2H/D2D movement and synchronization, including selected feature and
bounded status transfers. Output projection now consumes the same canonical
memory representation rather than maintaining a parallel report truth. DSpark
draft/verification transactionally merge the same transformer and batched
output-head memory facts with exact launch, H2D/D2H/D2D and synchronization
counters. Stochastic feature, sampling and acceptance references remain
outside the device-byte lower bound; their elapsed cost remains inside the
phase duration. Occupancy remains unavailable, so optimization ordering is
still provisional rather than a promoted kernel-priority decision.

Compatible width-N CUDA output rows now execute through one encoded-head
operation. Host producers use one bounded row-batch upload; device producers
must expose one contiguous identity-compatible view. The encoded head is
counted once, Q8 activation preparation and projection launch once for the
complete width, and the full output block is downloaded once for the retained
host sampling consumer. Aggregate physical facts belong to the ordered logits
execution rather than being multiplied into each row. Mixed, non-contiguous or
invalid source directories remain on the explicit row-local reference path,
which preserves complete-row failure semantics.

The same width-N owner can now retain the complete output block on CUDA and
publish one identity-bound row view per vocabulary row. Production device
results require a compiled profile, device-native execution class and
contiguous CUDA source directory; they cannot fall back to row-local or host
output. Their aggregate movement excludes the vocabulary download, each row
binds the correct target or draft state generation, and the device sampling
owner consumes the views before the reusable output buffer is overwritten.
The host result class and its full-array evidence remain the explicit
reference path. This uses the existing internal logits/device-view schemas and
changes no persisted, wire or public C contract.

Greedy DSpark target verification now consumes that resident result class.
One speculation-owned verification context borrows the existing model,
session, output-head residency and compiled profile; it projects all target
rows together, performs one width-N CUDA argmax launch and synchronization
through the sampling owner, and transfers only bounded aggregate
selection/status facts. Physical accounting is attached once to the ordered
batch rather than multiplied across logical rows. The accepted-prefix decision
remains in the target-authoritative speculation transaction, so cancellation
and failed verification cannot publish selected IDs or state. CPU,
audit/forensic and host-reference DSpark retain the complete-distribution oracle.
No persisted, wire, public C or profile schema needed a version change.

The source-authored mHC envelope keeps its FP64 sigmoid, Sinkhorn and
normalization contract, but independent stream rows no longer execute behind
one device lane. Stream-local pre/post gates, combination rows and alternating
Sinkhorn row/column normalizations run in parallel while each row or column
retains the original ordered reduction. BF16 residual squares accumulate in
F32 before the final FP64 inverse; the complete 43-layer CUDA oracle remains
bit-exact against the CPU/reference projection. No family geometry enters the
common kernel and no model, state, wire or profile identity changes.

Weighted DeepSeek RMSNorm applies the same admitted BF16 boundary: lanes
accumulate independent residual squares in F32, reduce them cooperatively and
retain the FP64 inverse, encoded weight application and BF16 publication. The
complete 43-layer attention oracle remains bit-exact. On the candidate artifact
and fixed four-token DSpark request, an identity-bound Nsight Systems capture
reduced 2,487 weighted-normalization instances from 175,153,088 ns to
25,373,280 ns; five warm resident runs retained the exact token and usage
fixture at a median 2.95367613 final token/s. The external capture digest is
`3719695af07073ea69f9506bdaa36c48cefa0f15148e3c587ad660880956621d`.
This component result is causal optimization evidence, not a promoted model
performance claim, and changes no persisted, wire, public or profile contract.

DeepSeek attention reduction now evaluates each visible history row once. A
stable online softmax retains source-order dot reductions and rescales the
accumulated denominator and value row whenever the running maximum changes;
the earlier two-sweep implementation remains represented by the independent
CPU/reference oracle. Across 6,772,096 compared values, the full 43-layer CUDA
lane remains inside its admitted contract (`max_abs=0.00390625`,
`rmse=5.685668969537683e-06`) and deterministic across repeated execution. On
the candidate artifact and fixed four-token DSpark request, an identity-bound
Nsight Systems capture reduced 699 attention-reduction instances from
212,617,664 ns to 118,771,328 ns while preserving exact generated text, token
usage and stop facts. Five warm resident runs measured a median 3.09469247
final token/s. The external capture digest is
`0abbf09993c4a1b7ed15181f06969816fa03180c7df72cf731d9d4d5b3b05fd2`.
This is component evidence rather than a promoted model-performance claim and
does not change a persisted, wire, public or profile contract.

CUDA attention graph replay now separates mutable state preparation from the
captured kernel topology. The graph-stream preamble refreshes the current state
bank before capture and every warm replay; promotion generation is therefore
not a graph-key fact while allocation, workspace, capacity and execution shape
remain compatibility facts. Live piecewise/full execution proves repeated
state commits against the independent attention oracle without recapture.

Target-only CUDA generation now cuts over from the eager reference to full
attention graphs when the admitted binding and live Driver expose the complete
graph capability. The existing compiled-profile schema already distinguishes
eager-reference ownership, so no contract version changes. CPU and DSpark stay
on the explicit eager reference until the target/draft arena is graph-stable.
Shape reconfiguration within one graph mode and profile identity retains
compatible cached executables; a mode or identity change still invalidates
them.

The CUDA kernel bundle now admits an ordered set of independently compiled
manifest-owned modules. General kernels and the MoE kernel family share one
toolchain-only qtype primitive interface, while module loading, required-symbol
resolution, rollback and checked unload remain atomic at the aggregate bundle
boundary. This earns kernel-bundle identity v3; it changes neither a persisted
model contract nor public API and creates the compilation boundary needed for
real width-N MoE without exceeding translation-unit policy.

Production CUDA MoE now consumes the existing width-N runtime contract through
one backend capability table. Workspace size derives from every admitted layer
qtype and the compiled row capacity. Each layer routes all rows, builds a
deterministic expert-major pair order, executes resident routed/shared packs,
and leaves its selected experts and weights on the device. Independent expert
scores are evaluated cooperatively; one ordered lane retains the source
tie-breaking and double-precision weight accumulation contract. Pair counting
and stable expert-major emission use one lane per expert, while the resulting
order is identical to the serial oracle. The owner enqueues only one bounded
status word and one unique-expert count per layer. One completion after the full
transformer stack validates all layer statuses, recovers exact active weight
bytes from sealed base/per-expert factors, and publishes the aggregate facts. A
later final-stage read or synchronization on the same session stream satisfies
that completion without another barrier; otherwise one stream wait closes the
stack. The independent immediate and token-local CPU/CUDA paths remain the
audit/reference oracles. Resident weights no longer reserve an unused
per-expert staging range, so that oracle fits the same shape-preflighted
workspace as the width-N path. Full forensic evidence disables Q8 activation
compression for its row dots; live admission therefore checks the production
approximation and the tighter forensic CPU/CUDA comparison separately. Qtype
access expectations are derived from the admitted MoE plan rather than a
duplicated physical-variant table. The internal source ABI rebuilds atomically;
no persisted, wire, public C, execution-profile, or state-layout schema changes.

The shared CUDA qtype primitive now admits short-row shapes that form
geometry-selected two-, four- or eight-lane groups for Q8_0, Q2_K, IQ2_XXS and
MXFP4 activation dots. Group-local work redistributes only exact integer terms;
each leader reconstructs the canonical block float before returning values to
the unchanged warp reduction lanes. The same primitive therefore fills idle
lanes across dense and grouped-MoE consumers without a model-name branch,
derived-layout change or numerical-contract bump.

IQ2_XXS sign-grid reconstruction now uses the CUDA population-count primitive
to recover the source parity bit rather than shifting the seven-bit code in a
serial loop. Every affected row-dot and full attention comparison remains
exact. On the candidate artifact and fixed four-token DSpark request, an
identity-bound Nsight Systems capture reduced the 12,075 generic qtype matvec
instances from 1,282,725,728 ns to 1,243,993,280 ns. The external capture
digest is
`e89e01fd362403a135c375cce60eb3eb6234dfbc4a69a42568e8fcb36e3d4632`.
The measured component reduction is not a promoted model-performance claim and
changes no qtype, layout, persisted, wire, public or profile contract.

The model-derived F32 mHC projections expose 24 rows over 16,384 source values.
For this narrow-output geometry, each row/input pair now owns one 256-lane
reduction block; encoded qtypes and split-input reference execution retain the
canonical warp-owned topology. The complete 43-layer CUDA attention oracle
compared 6,772,096 values inside the admitted contract
(`max_abs=0.00390625`, `rmse=5.685668969537683e-06`) and remained byte-stable
across repeated runs. On the fixed four-token request, comparable native-SM121
captures reduced the affected projection from 344,227.7 ns to 32,808.4 ns per
instance and generic qtype matvec time from 1,287,263,456 ns to 962,302,860.8 ns
per request. Five warm resident runs retained exact text, stop and 9/4/13 usage
facts at a median wall time of 2.746745 s (min 2.743537 s, max and nearest-rank
p95 2.947920 s, coefficient of variation 0.0290). The before/after external
capture digests are
`47de758b3d3645ea83663b469ca79bae1cdf1d672272361f84fa0f8266af1a93`
and `97a01cb0fe2f082c83a3a19a7da2606f21e345acfe7a960680507f974aef95e9`.
This is causal component evidence, not a promoted model-performance claim, and
changes only the complete-rebuild kernel ABI: no qtype, persisted, wire,
public, execution-profile or state-layout contract changes.

Multi-row qtype projections now derive their launch topology from both matrix
rows and input rows. Up to eight compatible input rows share one encoded-row
block, while wider batches tile that same mapping without padding the logical
row count. This keeps output-head and verification rows contiguous in the
kernel launch, preserves the existing per-warp arithmetic and tail refusal,
and changes no qtype, numerical, persisted, wire or public contract.

Each CUDA backend/session now owns one non-blocking ordinary-execution stream.
Production attention graph pieces borrow that stream, preserve their order and
defer completion to the existing layer publication barrier; piecewise execution
therefore performs one scoped wait per layer instead of one wait per captured
piece plus publication. Audit timing retains an isolated graph stream and
immediate completion because resolving its timing event is explicit evidence
work. Width-N MoE defers completion until the transformer stack ends and reuses
a proved same-stream barrier when one already exists. A Driver without a
complete stream lifecycle retains the portable context-wide fallback and
reports that wider barrier. Creation, completion and checked cleanup are
fail-closed and independently fault-tested. The final per-layer attention
publication wait remains explicit optimization debt.

Target-only production stochastic sampling now keeps the complete vocabulary
row on CUDA. Runtime stages exactly one PCG transition, the backend applies the
sealed temperature/top-k/min-p/typical-p/top-p order and categorical draw, and
runtime publishes the RNG state only after bounded device facts survive
cancellation and validation. The correctness-first kernel downloads 100 bytes
of token, survivor, probability and status facts rather than the vocabulary.
Greedy and stochastic bounded downloads are enqueued behind selection on the
session execution stream and wait on that stream once; the incomplete-Driver
fallback remains an explicitly counted device-wide barrier. Physical-ledger
addition accepts the two synchronization classes separately and owns their
checked aggregate, so overflow cannot partially publish phase accounting.
Audit/forensic profiles retain the host sampling oracle. Explicit complete
distribution materialization remains available only to an admitted forensic
consumer; it is no longer mandatory for a DSpark target anchor. The existing
compiled-profile reference flag and internal device-view contracts already
represent this cutover, so no persisted, wire, public C, execution-profile or
state-layout version changes.

The same sampling owner can execute a stochastic selection against an open RNG
transaction without advancing the context authority. Host and CUDA paths
derive result identities from the transaction's staged before/after states;
abort leaves the context unchanged, exact retry selects the same token, and
caller-owned prepare/publish advances one draw only at the admitted caller
commit boundary.

Production stochastic DSpark now keeps every adjusted draft row and width-N
target-verification row on CUDA. The device applies the sealed sampling filter
to p and q, evaluates each explicit target-owned acceptance draw, selects the
residual correction or target bonus, and returns only committed token IDs and
bounded acceptance facts. Runtime reseals those facts through the canonical
acceptance identity before state or RNG publication. Workspace capacity is
derived from vocabulary and model-authored proposal width before execution;
undersized capacity and malformed candidates refuse before a launch. CPU and
audit/forensic execution retain the complete-distribution oracle. This changes
only an internal operation table and helper signature; binding v8, protocol v7,
events v3, compiled profile v1, generation ABI v4 and the server-construction
API remain unchanged by this optimization.

Production CUDA DSpark now selects each target-authored anchor directly from
the resident output-head row. Greedy selection returns one bounded token/value
record; stochastic selection remains inside the speculation-owned RNG
transaction until target/draft state and publication commit together. CPU and
evidence-bearing profiles select through the same sampling owner over the host
row, rather than reconstructing a second selection algorithm in speculation.
The normal production profile refuses host-authored selection facts and records
zero full-array host-scan bytes. This is an internal complete-rebuild ABI change
only; binding v8, protocol v7, events v3, generation ABI v4 and the
server-construction API remain unchanged by this optimization.

Source-selected target features now collapse their mHC residual streams on
CUDA. Production CUDA execution publishes reduced rows only into the
transaction-owned token-major device directory and transfers bounded status;
evidence-bearing paths retain the compact host output. Feature
projection consumes resident input without re-upload, executes its encoded
width-N matrix and RMSNorm on CUDA, and no longer downloads normalized rows in
the production path. The draft core consumes those rows through its device view,
while semantic identities bind the exact producer, execution, tensor and
promoted-prefix facts without scanning device values on the host. Corresponding
unused host feature workspaces are omitted. The CPU, audit/forensic and
host implementations remain explicit numerical oracles. This requires no
persisted, wire, public C or profile-schema change.

Speculative prefill now contributes the merged target, projection and draft-core
physical facts to the phase roofline ledger. Checked addition is transactional,
and missing compulsory-memory operations remain unavailable rather than becoming
numeric zero or a complete lower bound.

The CUDA transformer final operation now has one optional device output for
the BF16 pre-normalized row. Production drafting reuses final-layer attention
storage after its last consumer, then explicitly materializes that bounded row
without downloading the expanded residual streams or recomputing the final
stage on the CPU. Aliased normalized and pre-normalized outputs fail before a
kernel launch. Full evidence retains expanded-row materialization and the CPU
final-stage oracle. This extends only the existing internal backend ABI and
earns no persisted, wire, profile or public C version change.

Accepted-prefix promotion now contributes its exact state-residency H2D,
synchronization and zero kernel/D2H/D2D facts. These counters are deltas around
the serialized session mutation, not estimates from configured capacity.
The uploaded destination span is the exact compulsory device-state byte count;
promotion has no active weights, device activations or temporary device span.
It therefore earns the complete memory mask without treating host-side
candidate projection as device traffic. Repeated occupancy evidence is a
work-unit-weighted mean with transactional overflow refusal.

Physical-variant research uses a funnel: role-level numerical probes,
representative-layer encoding, kernel microbenchmarks, bounded logit and DSpark
acceptance checks, then complete emission only for surviving candidates. A
calibrated candidate is not materialized merely because an uncalibrated recipe
exists.

Published target and stretch values remain engineering ambitions. A competitive
hard gate is admitted only from a same-checkpoint comparison or a measured
roofline plus active-byte lower bound. This distinction cannot lower an
existing target or manufacture closure.

## Serving authority

Continuous serving has one scheduler authority and one serialized mutation
domain. Thread count is not part of that semantic contract. Independently
admitted execution and I/O workers may be used when ownership, cancellation,
publication, fairness and cleanup remain exact. The public server API is not
versioned speculatively; its current entrypoints must converge on the admitted
scheduler before a concrete external consumer may justify a new ABI.

## Source-authored conversation protocol

The DeepSeek model projection owns one immutable conversation descriptor bound
to the pinned `encoding/encoding_dsv4.py` bytes (27,908 bytes, SHA-256
`bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0`).
It supplies prompt syntax, special-token IDs, exact thinking delimiters,
effort-control text, tool grammar and the source `drop_thinking` rule. Common
tokenizer, runtime, server and adapter owners consume typed facts; none
hardcodes DeepSeek delimiters or special-token IDs.

The admitted modes are chat/non-think, think-high and think-max. The operator
surface exposes them as `/nothink`, `/think`, `/think-max` and as the equivalent
non-interactive `--reasoning` policy. Only the exact model-emitted
`<think>...</think>` grammar is classified. An incomplete or malformed grammar
fails closed, arbitrary prose is never reclassified, and no hidden model state
is exposed.

Protocol v7 carries distinct reasoning, final, tool and error channels plus
reasoning/final token counts, rates, first-token times and total completion
time. The REPL renders explicit reasoning incrementally and distinctly; raw
execution writes canonical channel payload bytes without terminal decoration.
OpenAI compatibility profile v2 projects explicit model output through
`reasoning_content` and never merges it into final content. Ordinary multi-turn
prompt construction may drop prior reasoning exactly as the source encoder
does, while tool-enabled continuity retains the source-required reasoning and
tool sequence. These are explicit model outputs, not inferred chain of thought.

## Non-claims

The currently admitted code establishes model-derived geometry, v7/v8 binding
coexistence, typed capacity/page planning, pre-materialization memory refusal,
the availability-aware production phase ledger, and identity-bound native
SM121 CUBIN admission. It also establishes identity-bound width-N CUDA logits
publication and greedy/stochastic DSpark target-anchor selection without
full-vocabulary host materialization.
It does not yet establish Tensor Core execution, specialized attention,
GB10-competitive grouped MoE, zero per-layer attention synchronization,
full-model live qualification of target-only or DSpark device sampling,
full-model live qualification of all reasoning modes, paged state allocation,
prefix persistence, continuous batching, competitive throughput, evaluation,
benchmark qualification, release qualification, or Hugging Face publication.
