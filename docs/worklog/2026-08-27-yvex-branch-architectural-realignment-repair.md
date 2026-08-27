# YVEX branch architectural realignment repair

| Field | Value |
| --- | --- |
| Date | 2026-08-27 |
| Type | closure |
| Milestone | YVEX.BRANCH.ARCHITECTURAL.REALIGNMENT.REPAIR.0 |
| Branch | models1 |
| Baseline | 5de22c3ec5edf889f76c0952244c3b1cf55cbf42 |
| Checkpoint | 9b2b83209e271bb9ce6fff1e811b5566a4433241 |
| Subsystem | deployment, runtime, server, model catalog, source catalog, graph, evidence |
| Model family | DeepSeek V4 Flash and MiniMax H3 |
| Hardware | NVIDIA GB10, compute capability 12.1, 128 GiB unified memory |
| Evidence | software tests; numerical conformance; runtime qualification; controlled performance; bounded live provider metadata |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The preceding core refoundation had already separated immutable package truth from deployment
specialization, introduced PEIR v5 and runtime binding v15 with explicit v14 import, established
process-local engine generations, and made the foreground server capable of remaining healthy
with zero loaded engines. It also retained exact transactional sequence state, exact MiniMax
numerics, and one generic runtime/backend substrate for DeepSeek and MiniMax.

That checkpoint exposed a second layer of overlap. The engine scheduler coalesced compatible
physical operations, but the outer generation loop still owned most inference-progress choices.
Engine resource accounting had correct categories without one complete catalog for independently
prepared and releasable resources. Rich execution ancestry and engineering result types still
crossed ordinary execution ABIs. Model, remote-provider, local-package, live-engine, and UI
projection facts remained too close in the model-domain surface. Deployment planning, device
views, package execution facts, and roofline evidence also still shared one broad execution
interface.

The concurrent Model Lifecycle delivery had added provider-neutral discovery and acquisition, but
its successful product projection made those lifetime overlaps easier to see. The branch needed
one converged architecture before another family or performance regime could safely build on it.

## Problem

One ordinary token, one MiniMax component operation, and one catalog row each crossed descriptors
that repeated facts owned elsewhere. The practical risks differed by path:

- a request worker could remain the hidden inference-progress scheduler while an engine scheduler
  existed only as a physical-operation rendezvous;
- a prepared representation could be accounted for without being independently owned, evicted,
  and recreated from its package and specialization provenance;
- production dispatch could reconstruct full lineage that an authenticated engine generation
  already owned;
- benchmark and forensic structures could become ordinary runtime transport;
- an origin/action failure record could coexist with a historical refusal ontology;
- a catalog domain record could join provider, package, engine, and presentation facts before the
  actual projection boundary;
- provider implementation injection and canonical model revisions could acquire duplicate owners;
- a compile-time engine slot constant could look like model topology rather than deployment
  capacity.

These were authority problems, not naming problems. Adding a second descriptor beside each old
one would have made the branch harder to understand while preserving the same ambiguity.

## Causal analysis

Source tracing confirmed the scheduler concern. Ready sequence work was advanced by the
generation owner and reached the runtime scheduler only after a compatible physical operation had
already formed. Introducing ready-work leases established the correct engine authority, but the
first integration also exposed a false equivalence: the mere presence of the progress scheduler
was treated as permission to route every single-session MoE operation through compatible physical
batching. Controlled DeepSeek throughput fell sharply. Profiles showed scheduler rendezvous on a
population that had no real cross-sequence width. Separating progress scheduling from explicit
compatible-operation scheduling restored the direct single-session path while retaining real
width-two batching in its admitted test lane.

The resource limitation was also confirmed. Residency categories existed, but selected prepared
views did not yet have one engine-owned lifecycle with explicit dependencies, numerical contract,
generation validity, byte accounting, and release behavior. A typed catalog supplied this
ownership without recreating either the rejected DeepSeek selective caches or the rejected full
derived MoE layouts.

The execution and evidence audit found that package PEIR, deployment policy, device materialized
views, and rich observation facts had different lifetimes despite sharing interfaces. Moving
deployment planning and device/evidence views to their real owners narrowed transport. Compact
engine generation, implementation handle, operation handle, dynamic geometry, and transaction
lineage remain in production; complete ancestry is recovered from the engine only when evidence
requires it.

The model/catalog hypotheses were confirmed. Remote provider truth, acquired source, admitted
local package, live engine observation, and catalog presentation could be composed, but were not
the same record lifetime. Public search options also exposed a Hugging Face subprocess override,
and remote ranking repeated qualified repository/revision facts already owned by the source
target catalog. Those facts now compose through typed queries and a catalog projection.

Two compatibility concerns were falsified. Runtime binding v15 did not require a new wire version:
its persisted facts remained package facts after the deployment split, and the authenticated v14
normalization path did not need retired policy to remain operational. MiniMax did not require a
second runtime or scheduler; component resources and exact numerical obligations fit the same
engine/resource/specialization model.

Final qualification exposed three independent defects rather than architectural exceptions. An
empty catalog called `qsort` with no backing array, which is undefined despite a zero element
count. A failed or cancelled media turn could emit progress, then return an error without a
terminal protocol response because generic error publication was suppressed after the first
response. Finally, the default MiniMax Omni test extent requested two timesteps from a one-entry
oracle manifest. Each defect was repaired at its owner; no tolerance, state contract, or evidence
gate was weakened.

## Decision

The branch now uses these non-overlapping authorities:

```text
source catalog and provider adapter
        -> remote/source/package domain records
        -> catalog projection for CLI or future UI

semantic model and operator schedule
        -> PEIR v5 package/storage truth
        -> immutable package plus runtime binding v15

deployment plan
        -> engine specialization and implementation handles
        -> exact engine generation
        -> typed engine resource catalog
        -> transactional sequence/component state
        -> ready-work scheduler
        -> executable batch and expert worklist
        -> generic backend implementation

lightweight observations
        -> evidence, benchmark, and forensic assembly
```

The engine scheduler owns which ready work advances. Compatible-operation batching remains a
separate physical mechanism enabled only when the execution context admits a real compatible
population. Executable batches still describe actual selected rows, and expert worklists still
order actual routed pairs. No semantic width is manufactured.

The canonical package mapping remains sufficient to reopen the model. Prepared resources are
optional engine-owned descendants with explicit provenance and release semantics. Removing one
does not alter package identity, and this milestone adds no speculative production cache merely
to demonstrate the mechanism.

Rich evidence remains reconstructible but does not define ordinary execution transport. Failure
is represented by one origin plus recovery action and scoped state consequence. Exact requests,
integrity failures, and committed publication boundaries still fail closed; automatic recovery
can choose only an already admitted numerically equivalent strategy.

## Implementation

Runtime scheduling now admits bounded work kinds for prefill, decode, draft, target verification,
correction, and publication. A lease binds the sequence and state generation so stale queued work
cannot mutate an advanced session. The scheduler's progress authority and the compatible physical
batcher are separately configured; default single-session generation does not infer batching from
the existence of an engine scheduler.

The engine resource catalog owns package mappings, prepared representations, components, backend
handles and caches, sequence state, workspace, and temporaries. Entries retain owner generation,
package and specialization provenance, byte class, readiness, dependencies, numerical/admission
contract, references, and release eligibility. Unit coverage proves independent prepare, admit,
depend, evict, recreate, and cleanup behavior using bounded resources.

Deployment planning moved to `deployment` ownership. Device execution views and rich graph
observation moved to dedicated internal contracts. Runtime execution and generation headers were
narrowed; rich benchmark, profile, and forensic records are assembled by evidence owners from
lightweight production observations. Full package ancestry is not recopied or rehashed merely to
dispatch an operation inside an already authenticated engine generation.

Failure handling converged on the origin/recovery model, while qtype and GGUF refusal identifiers
that still encode real numerical/container diagnostics remain domain-local. The server media owner
now publishes exactly one terminal typed error after failed or cancelled media progress, preserving
the failure class, phase, cancellation class, stop reason, session state, and elapsed time.

The public model surface was divided into durable catalog and materialization domains. Provider
records no longer own local package or live engine state; a projection explicitly joins remote,
local, and observed authorities. The Hugging Face command override moved out of the public model
ABI. Canonical repository and immutable-revision defaults now come from the source target catalog,
including acquisition, inspection, and ranking consumers.

Engine capacity now distinguishes a bounded implementation safety maximum, a configured maximum,
and actual resource admission. The default remains eight configured slots, but neither the model
ontology nor the machine's current ability to admit large engines is inferred from that default.

Current architecture, runtime, C API, source-ownership, operator, migration, README, and changelog
projections were reconciled only where the implemented authority changed. Historical worklogs were
not rewritten. The deterministic tiny lifecycle now uses the production provider, acquisition,
verification, package, catalog, server, generation, observation, unload, and host-survival path.

## After

One engine generation owns deployment specialization, resources, stale-reference protection,
sequence/component state, and inference-progress admission. The server owns external request
admission, routing, fairness, cancellation ingress, connection lifecycle, and same-session
serialization; it is not the hidden model executor. A host may be ready with zero engines, may
own multiple fitting engines, and may restrict real residency through resource admission rather
than topology.

Package truth contains no deployment-transient tile, warp, capacity, evidence, or live-resource
policy. Numerically significant implementation classes remain specialization identity; equivalent
backend-local execution details can change without package or binding churn. PEIR remains v5,
runtime binding remains v15, and compatible v14 assets reopen only through explicit authenticated
normalization.

Remote provider, local source/package, and live engine facts are independently valid. The catalog
projection joins them for operators without making presentation a capability authority. The
bounded live Hugging Face metadata smoke returned the canonical MiniMax repository first, resolved
the immutable default revision, and reported the retained local source/package as a related older
revision without downloading payload bytes.

DeepSeek and MiniMax continue to use the same generic server, engine, scheduler, resource,
transaction, and backend substrate. Generic server, runtime, and CUDA execution contain no
family-name policy branch. Continuous batching remains explicitly not ready; the result is one
correct scheduling authority and proven compatible batching, not an inflated product claim.

The final structural snapshot contains 306 production files: 223 translation units and 83 headers
(16 public, 53 internal, 14 source-local), across 271 semantic owners. It reports 1,562 library
globals, zero duplicate definitions, zero foreign globals, zero include cycles, zero functions
over 200 lines, zero hard-width violations, and zero architecture, layout, naturalness, or
ownership-policy violations. The largest translation unit is 2,000 lines and the largest function
is 200 lines, both at the admitted boundary rather than beyond it.

## Quantitative delta

DeepSeek throughput compares the same admitted artifact and binding, GB10, greedy workload class,
and retained sampling method. Short lanes retain ten measured samples; 256-token lanes retain
three. The controlled measurements were taken at `8d53d52fbd52f0b8d473b7e67f69f73eaf340946`;
the two later production changes affect only server media terminal publication and the MiniMax
test fixture default, so the DeepSeek execution source is unchanged at the final checkpoint.

| Lane | Refoundation checkpoint | Realignment | Range | Delta | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| target-only short | 9.61 tok/s | 9.545 tok/s | 9.510-9.630 | -0.68% | directly comparable |
| target-only 256-token | 7.49 tok/s | 7.420 tok/s | 7.410-7.430 | -0.93% | directly comparable |
| DSpark short | 10.575 tok/s | 10.550 tok/s | 10.460-10.620 | -0.24% | directly comparable |
| DSpark 256-token | 9.72 tok/s | 9.670 tok/s | 9.650-9.690 | -0.51% | directly comparable |

No retained median crossed the 3% investigation threshold. The realignment therefore establishes
no material performance regression and makes no optimization claim. The final exact-tree
component benchmark gate retained three benchmark modes and six charts in the external
identity-bound benchmark directory; raw benchmark and profile artifacts remain untracked.

## Evidence

- Canonical combined QA run identity:
  `25187ca40bd1e5983308b7eb1c20d2d31ac02e556aeb2c5e1acdca0c0d85a13e`.
  It ran from clean commit `9b2b83209e271bb9ce6fff1e811b5566a4433241` with clean source-delta
  identity `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
  Start and finish snapshots were identical; source stability was valid.
- The combined change plan resolved nine obligation groups into 111 tests: 111 PASS, zero FAIL,
  zero ERROR, zero SKIP, and zero BLOCKED. Build identity was
  `026eb5aaf4970cf117f5752494c9aa2dc48d17700fa62d1089cf09d8f38ce1ef`; registry identity was
  `90855570b3063145d2e04b2cf02384be32100fa0795d2dc653ec0ec660575185`.
- CUDA native, no-NVCC fail-closed, ASan/LeakSanitizer, UBSan, project control, documentation,
  repository layout, source ownership, dependency architecture, operator registry, product
  topology, naturalness, and topology closure all passed.
- DeepSeek live qualification passed target-only deterministic and stochastic generation, DSpark,
  greedy target equality, reasoning-mode equality, cancellation, partial progress, mutation
  refusal, committed/candidate state, and cleanup. The explicit compatible lane formed 175 real
  two-source physical batches and 172 expert worklists; default single-session performance profiles
  formed zero physical batches.
- The real DeepSeek host loaded the exact current package, served bounded generation, exposed
  resource state, unloaded to zero current model resources, remained ready, and stopped separately.
  Mapped package bytes remained distinct from prepared and device-copy bytes.
- MiniMax 466-row by 50-block Omni execution remained exact for video and audio: relative L2 `0`,
  cosine `1`, and scaled maximum `0`. Residency identity was
  `d9b1c2225f651d97020d74521347fdc3e2702b916db5837651f6f3f10d909ba6`; execution identity was
  `0b12647d0f0123da129952d7f73fe776174989911536c482d7c7e934130739b8`.
- MiniMax two-step latent execution remained exact for both domains. Plan, layout, latent, and
  transformer-chain identities were respectively
  `a6ec3aa32c33397b200020de4c93a13388b7bbe240bf92ec1f0044568767bdf9`,
  `4e23ba2d45b1f7e70740992c3788c26931bf5792129af5bbe86abf84f5a622b4`,
  `74f8f887bc15124b18fabe79dcbb3a4314fad41e7365a69b65754b2be7a1d4e0`, and
  `69e7cd108956d74f514ab8220d534932dd95ed06460b5c38b0d042513f9797a3`.
- The real MiniMax composite lifecycle loaded one engine, bound a media session to its exact
  generation, entered production conditioning, handled cancellation with one terminal protocol
  error, published no partial media file, unloaded with zero current sessions/resources, left the
  host ready, and stopped separately. The retained probe added no swap-out and peaked at about
  62.6 GiB resident memory. Exact media numerics are established by the independent oracle lanes,
  not inferred from lifecycle reachability.
- The production-shaped model-neutral vertical passed provider search, remote inspection, immutable
  revision selection, acquisition, verification, package publication, local catalog projection,
  persistent-host load, bounded generation, live observation, unload, and host survival. The live
  Hugging Face smoke resolved `MiniMaxAI/MiniMax-H3` at
  `42ed227ee7df40d41602854ae760620d6eb651fe` without payload download.

One combined QA attempt was invalidated when concurrent published source advanced and was not used
as qualification evidence. A later performance attempt was stopped when a separate profiler began
using the exclusive GPU. Two other diagnostic attempts correctly exposed, rather than hid, the
one-entry Omni fixture mismatch and a non-empty unrelated benchmark directory. The final retained
run used an uncontended GPU, the corrected fixture contract, a fresh identity-bound benchmark
directory, and one stable source snapshot. No foreign source change was lost and no real semantic
conflict required mechanical resolution.

## Remaining limitations

- DeepSeek's retained artifact-mapped narrow DP4A regime remains below the 20 tok/s first
  performance floor. Matrix/tile execution and economically positive long-form DSpark remain later
  performance owners; this wave neither restored rejected caches nor tuned kernels.
- `continuous_batching_ready` remains false. The engine scheduler now owns progress and compatible
  batching is real, but production multi-sequence continuous progress is not claimed.
- Simultaneous residency of the full DeepSeek and MiniMax packages was not attempted. Tiny fitting
  engines prove host topology; resource admission may serialize the two huge real packages.
- Arbitrary low-level MiniMax component reads are not yet interruptible at every internal read
  boundary. Cancellation is transaction-safe and terminally published, but bounded cancellation
  latency remains a product capability frontier.
- The end-to-end remote lifecycle uses a deterministic provider payload through production code.
  Live Hugging Face evidence is metadata-only; no massive remote model was downloaded merely to
  restate already admitted package and engine behavior.
- No third family is claimed. Static adversarial tracing for a dense decoder, sparse/MoE decoder,
  and composite media model found registration and semantic recipe work, not a required family
  branch in generic server, scheduler, catalog, or CUDA execution.

These are performance or product capability frontiers, not known overlapping authority in the
realigned architecture. This wave leaves zero known unresolved architectural debt within its
declared boundary.

## Why it matters

Future deployment regimes can now add a prepared representation, scheduling policy, or equivalent
backend implementation at the lifetime that actually owns it, while remote discovery, immutable
package truth, exact engine generations, transactional state, and operator evidence remain
separate and independently understandable.
