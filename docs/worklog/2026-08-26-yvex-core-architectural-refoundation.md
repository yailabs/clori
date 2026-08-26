# YVEX core architectural refoundation

| Field | Value |
| --- | --- |
| Date | 2026-08-26 |
| Type | closure |
| Milestone | YVEX.CORE.ARCHITECTURAL.REFOUNDATION.0 |
| Branch | models1 |
| Baseline | 77036fb58d418185c2a928fc71838bb16046ccf8 |
| Checkpoint | e536897b3e19a14e432026b04e5c362c3df7d48b |
| Subsystem | execution, runtime, server, sequence state, backend, family integration |
| Model family | DeepSeek V4 Flash and MiniMax H3 |
| Hardware | NVIDIA GB10, compute capability 12.1, 128 GiB unified memory |
| Evidence | software tests; numerical conformance; runtime qualification; controlled performance |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

YVEX already had strong immutable source, artifact, numerical, and transactional-state
contracts. The accepted DeepSeek binding authenticated a complete compiled execution and the
MiniMax output projection used an exact compiler-selected numerical policy. Candidate and
committed sequence state were separate, speculative target verification was authoritative, and
the CUDA backend refused unsupported exact execution rather than silently changing semantics.

The middle of the system nevertheless mixed facts with different lifetimes. Physical Execution
IR v4 and runtime binding v14 carried both package/storage truth and deployment execution policy.
Runtime descriptors, workload profiles, execution shapes, compatibility keys, batches, and
backend jobs repeated identity and implementation facts. The foreground server was still
conceptually opened around one model, while request scheduling and compatible batching lived in
separate partial schedulers. MiniMax exact output-linear configurations appeared as generic graph
cases and CUDA recognition logic. Runtime evidence records also crossed ordinary production ABIs.

The product boundary was protocol v12. Model and server lifetimes were coupled in the canonical
workflow, so changing the loaded model meant changing the foreground process. Resource telemetry
also treated mapped package bytes, prepared bytes, device residency, sequence state, and
workspace less distinctly than their actual ownership required.

## Problem

An engineer tracing one token or media request had to reconcile too many overlapping authorities:

- permanent model and package facts;
- hardware/deployment implementation policy;
- one engine generation's resources;
- dynamic sequence population and phase;
- backend-local equivalent implementation detail;
- evidence-only lineage.

That overlap made a hardware specialization change look like a model-package change, made
runtime binding growth the default response to new execution regimes, repeated full identities in
transient hot-path objects, and left no clean owner for selective prepared representations.
Server topology also encoded a resource limitation as a product ontology: one foreground server
was effectively one model process.

## Causal analysis

The package and deployment concepts were not intrinsically invalid. The problem was that PEIR v4
and binding v14 had become the only durable place to put both. Historical additions were each
locally defensible, but together they caused compiled profile, shape, compatibility, batch, and
backend structures to restate one another.

The server had two different scheduling responsibilities. Its scheduler serialized external
requests and sessions, while runtime batching formed compatible physical rows. Neither alone was
an engine-level inference-progress authority. The server could therefore own multiple worker
threads without truthfully owning continuous sequence scheduling.

DeepSeek and MiniMax provided complementary evidence. DeepSeek showed that additional derived
CUDA weight copies can worsen unified-memory economics, so a prepared representation must be an
optional resource rather than a replacement model arena. MiniMax showed that an engine cannot be
defined only as an autoregressive text model: one logical model can own several independently
materialized components and an exact media numerical contract.

The accepted exact MiniMax output-linear choices were numerically significant, but their graph
and CUDA encoding did not make them model semantics. Conversely, warp, grid, and equivalent
microkernel decisions did not deserve permanent package or binding churn. This distinction
required one deployment specialization authority, not another copy of PEIR.

## Decision

YVEX now separates the authorities as follows:

```text
verified source facts
        -> semantic model and declarative operator schedule
        -> PEIR v5 package/storage decisions
        -> immutable package and binding

deployment boundary
        -> engine specialization v1
        -> exact model-engine generation and typed resources
        -> transactional sequence state
        -> engine scheduler
        -> executable batch and expert worklist
        -> generic executor and backend implementation

evidence observes these boundaries without becoming their transport protocol
```

PEIR v5 retains canonical operator/tensor identity, role and scope, source-derived geometry,
canonical qtype, encoded byte range and alignment, stable package layout and sharing, and the
physical decision identity. Deployment backend, hardware capability, selected implementation
class, workload capability, and exact algorithm choices live in one immutable engine
specialization created when an engine opens.

An engine specialization can bind a numerically significant exact algorithm. Within that class,
CUDA may choose equivalent tile, warp, grid, stream, graph, or microkernel details without
changing package meaning. Unsupported exact combinations still fail closed.

The current writer is runtime binding v15. Existing v14 bytes are parsed through an explicit
v14 wire representation, their v4 physical identities are authenticated, and compatible package
facts are normalized to PEIR v5. Old bytes are never reinterpreted as v15. A v14 requirement that
cannot be represented without reviving retired deployment policy is refused.

The server is a persistent host. It can be healthy with zero engines, owns a bounded engine
manager, and routes requests and sessions to an exact alias plus generation. Load, draining,
unload, reload, and host shutdown are distinct operations. Resource admission may limit which
large engines coexist, but that does not limit host topology.

One scheduler now belongs to each engine. The server still owns external admission, routing,
fairness, cancellation, and same-session serialization. The engine scheduler owns ready prefill,
decode, draft, verification, correction, and publication work. Canonical execution batches and
expert worklists remain separate real-population contracts.

## Implementation

Physical lowering and binding import were narrowed around PEIR v5. Runtime specialization now
creates typed implementation records and compact decision handles once per engine. Transient
execution batches carry engine generation and compact lineage rather than re-copying the full
runtime, binding, physical variant, profile, and operation identity chain. The transient execution
shape registry was removed, and workload profiles became engine-specialization inputs.

The runtime model-engine owner now binds one admitted package to one specialization and one
process-local generation. Its typed resource graph distinguishes canonical mappings, prepared
views, resident host/device resources, sequence state, reusable workspace, and temporary
allocations. Resource summaries report current ownership and clear on unload; mapped package
bytes are not mislabeled as device allocation on unified-memory hardware.

Protocol v13 adds engine load, unload, and inventory operations. `yvex server` starts with zero
engines; `server load`, `server models`, `run --model`, and `server unload` cross the same typed
protocol. OpenAI `model` routing and native sessions resolve an exact engine generation. Reloading
the same alias produces a new generation, and stale sessions cannot bind to it.

The old server scheduler translation unit was removed. Request queuing is now named and scoped as
external request serialization, while runtime scheduling owns bounded sequence work quanta and
compatible row formation. Transaction coordination uses a bounded participant collection, so
attention state, draft state, RNG, token ledger, decoder state, and metadata can share one
begin/prepare/publish/abort lifecycle without sharing one data representation.

MiniMax exact output-linear requirements moved out of magic generic graph cases. The package
retains the exact numerical obligation; engine specialization resolves video and audio output
operations to the accepted implementation class; generic CUDA consumes the typed decision. The
CUDA layout translation unit and its parallel nominal policy chain were removed.

The normal engine ABI no longer carries full operator benchmark and roofline result objects.
Production publishes lightweight events and counters, while independent reference, benchmark,
and forensic owners build richer evidence. Failure handling now composes failure origin with a
recovery action, allowing auto mode to respecialize only among already-admitted equivalent
strategies while preserving explicit exact refusal and integrity failure.

Component execution, decoder dispatch, weight views, convolution mechanisms, and resource
lifecycle moved to generic runtime/backend owners. Family files retain source-derived geometry,
tensor roles, schedules, state meaning, composition, and numerical obligations. Static inspection
finds no DeepSeek or MiniMax execution switch in generic runtime, server, or CUDA owners.

The final real MiniMax unload exercise exposed two cleanup residues: an open media session was
removed from its registry without decrementing host active-session telemetry, and the unloaded
engine summary retained its previous session count. The media registry now closes every remaining
session under its lifecycle lock, and engine close clears current session population only after
the registry is gone. Unit, tiny vertical, and real composite lifecycle evidence cover both.

## After

The immutable package can be reopened independently of which admitted backend currently executes
it. Engine open chooses and authenticates deployment specialization without changing model or
artifact identity. Equivalent backend implementation details can evolve without rebuilding the
package, while exact numerically significant deployment choices remain specialization-identity
bearing.

The host can start and remain ready with zero models, load an engine, bind sessions to its exact
generation, serve work, drain and unload it, and remain ready for another engine. Tiny production
fixtures prove two simultaneous fitting engines and unambiguous routing. Real GB10 exercises prove
the current DeepSeek package load/use/unload lifecycle and the MiniMax composite
load/session/unload lifecycle. Large DeepSeek and MiniMax engines were intentionally serialized;
simultaneous residency was not claimed.

DeepSeek still executes the retained artifact-mapped narrow DP4A regime. No rejected selective
cache or duplicated derived MoE layout was restored. The new resource/specialization owners make
a later profitable prepared representation possible without making it permanent model truth.
MiniMax remains one logical composite engine with component-specific resources and exact video,
audio, and latent numerical authority.

The tree contains one engine-level scheduling authority and one external request-queue authority.
It supports real compatible-operation batching but continues to report
`continuous_batching_ready=false`; worker concurrency is not promoted into a continuous-batching
claim.

## Quantitative delta

All throughput rows below compare the same authenticated DeepSeek artifact and binding, greedy
workload class, GB10, and retained method. Short lanes use ten retained samples; 256-token lanes
use three. The refoundation did not retain a performance optimization.

| Lane | Before | After | Delta | Classification |
| --- | ---: | ---: | ---: | --- |
| target-only short | 9.83 tok/s | 9.61 tok/s | -2.2% | directly comparable |
| target-only 256-token | 7.62 tok/s | 7.49 tok/s | -1.7% | directly comparable |
| DSpark short | 10.68 tok/s | 10.575 tok/s | -1.0% | directly comparable |
| DSpark 256-token | 9.72 tok/s | 9.72 tok/s | 0.0% | directly comparable |

No repeated median decode regression exceeded the 3% investigation threshold. A separate
1000-token reasoning observation measured target-only at 2.22 tok/s and DSpark at 2.09 tok/s;
swap activity makes those values characterization only. The identical committed output digest
was `088567...`; this observation does not establish positive long-form DSpark economics.

The real DeepSeek lifecycle probe loaded 95,050,210,304 mapped package bytes with zero prepared
or device-copy bytes, served one greedy token, unloaded in 0.41 seconds, and returned current
mapped, prepared, host-resident, and device-resident bytes to zero. The one-token rate is not a
benchmark claim.

## Evidence

- DeepSeek artifact identity:
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
- Existing DeepSeek runtime binding v14 identity:
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
  Binding import and live generation accept it through the explicit v14 compatibility path.
- DeepSeek target-only, stochastic, DSpark, greedy-equality, reasoning, cancellation, partial
  publication, mutation, and lifecycle lanes passed on CPU/CUDA. Compatible scheduling produced
  real width-two execution, four-row rendezvous, 175 physical/multi-source batches, and 172
  expert worklists without replaying accepted target rows.
- The real persistent host started with zero engines, loaded the exact DeepSeek artifact as
  generation 1, served a bounded greedy request, unloaded it, remained protocol-ready, reported
  one open and one close, released all current model resources, then shut down separately.
- Tiny production vertical identities remained
  `a946a8447534b15556e9b6d38c57cfee2bb3f5f05ffa5932ac9c672b7641341d` for the artifact and
  `5ecd3e97e2fbe6a8b5a37af95df11e47fdb3507a59a1026cc4bf2c27da39804b` for the binding. It
  proves zero-engine startup, load/use/unload, generation-safe reload, two fitting engines,
  explicit routing, stale-generation refusal, cleanup, and separate host stop.
- MiniMax Omni 466-row by 50-block execution remained exact: video and audio relative L2 `0`,
  cosine `1`, and scaled maximum `0`. Output identities remained
  `ceb890960d96bcb75e83361c99953300e222a0a6bf654572faac2cb755789a94` and
  `1841ddc1c4e5e8292f4020128f463a0fb5e4b5f1d1ca7edead3f76db969192c0`.
- MiniMax two-step latent identities remained
  `b0388b0896923f8f0cb764b4ca5e6a3e6f3512da9ac9ddc74143e58599113ab1` and
  `ca9b5616ab217b8253effc153db7fc7b36739b9ef7528afaead2b2f1d3e07306`.
- The real MiniMax composite host loaded generation 1, bound a media session to model identity
  `c9c8f8684058ba3d46df39ae64a8de55e74e0c6cbae9b6eb3ea29f1024dc369b` and specialization
  `fe268f00359897026c2b6a3917c40eb2839e45da3826d6eabe01ad711210c691`, unloaded with zero
  current sessions/resources, remained ready, and stopped separately. Exact media execution is
  established by the numerical lanes above rather than inferred from load alone.
- `unit.server` and `integration.tiny-vertical` passed after the media cleanup repair. CUDA native,
  no-NVCC refusal, ASan/LeakSanitizer, and UBSan lanes passed on the combined implementation.
- A full branch-delta diagnostic resolved 110 obligations and produced 107 PASS, two MiniMax text
  fixture selection failures, and one external performance BLOCKED row. The text failures were
  configuration errors: the v2 component lacked current package metadata. The current v3 text
  component passed exact token-1 embedding error `0` against the independent v2 numerical oracle.
  Final closure qualification uses that corrected asset selection.

Concurrent development remained source-safe. Fourteen Model Lifecycle / Hugging Face commits were
accepted as independent branch advancement and are documented in
`2026-08-26-model-lifecycle-hub-experience.md`; they are not claimed as this delivery's core work.
Two shared files received same-file, non-overlapping edits and preserved both deliveries. No real
semantic conflict occurred and no foreign hunk was lost. The main delivery invalidated at least
two long-running checks after source movement and reran retained evidence; the lifecycle delivery
independently recorded five source-mutated attempts across its own run. These counts are kept
separate because some intervals may overlap.

## Remaining limitations

- Global ready-sequence continuous batching is not claimed. Current production owns bounded
  engine scheduling and real compatible-operation batching only.
- The host data model supports several engines, but the GB10 resource manager may admit only one
  huge current family at a time. DeepSeek and MiniMax were not simultaneously resident.
- Typed prepared resources are now possible, but no selective DeepSeek cache/layout is retained.
  Previous output-head and derived MoE cache candidates remain rejected on complete-model
  economics and memory/swap pressure.
- DeepSeek remains below the 20 tok/s minimum performance floor. The active performance milestone
  and matrix/tile representation frontier remain open; this refoundation makes that work
  architecturally cheaper but does not claim to solve it.
- Long-reasoning DSpark remains economically negative in the retained characterization. Explicit
  DSpark is not silently converted to target-only.
- MiniMax in-flight component materialization is still not generally interruptible. Load/unload,
  terminal cancellation, cleanup, and exact execution are qualified; random low-level read
  cancellation was not added.
- `performance.runtime` still requires a legitimate external identity-bound benchmark directory.
  An arbitrary directory was not supplied merely to remove a BLOCKED row.
- Binding v14 compatibility is deliberate and bounded. It is not a promise to retain every
  pre-v0.1 internal schema indefinitely.

## Why it matters

YVEX can now evolve package format, deployment specialization, engine lifecycle, scheduling,
resource policy, backend implementation, and evidence at their actual lifetimes. DeepSeek and
MiniMax remain correctness tests of one substrate, while a server can change loaded engines
without changing process identity or weakening provenance and transactional state.
