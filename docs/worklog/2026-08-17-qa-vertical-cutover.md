# Canonical QA Vertical Cutover

| Field | Value |
| --- | --- |
| Date | 2026-08-17 |
| Type | closure |
| Milestone | `V010.DEVELOPMENT.QA.TEST.ARCHITECTURE.REFOUNDATION.0` |
| Branch | `main` with downstream DeepSeek and MiniMax cutovers |
| Baseline | `93a484be49cc5c0a15139f9c68f54c2ce571869b` |
| Checkpoint | `537c86622ff264a6888ebd32c5e5a8d2cc8be7e0` |
| Subsystem | development QA, evidence orchestration, and vertical qualification metadata |
| Model family | branch-neutral foundation; DeepSeek and MiniMax consumers |
| Hardware | NVIDIA GB10, compute capability 12.1, Linux aarch64 |
| Evidence | registry parity; CI; CUDA; no-NVCC; sanitizers; numeric; runtime; structural; blocked live prerequisites |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The main-derived QA foundation had established one registry, generated native
test registration, explicit result states, and change-aware obligation
resolution. DeepSeek and MiniMax still carried tests added after their common
main ancestor, so the repository-wide architecture was not closed until both
verticals consumed the same authority.

The first DeepSeek downstream execution also exposed a concurrency defect in
the initial resource model: independent QA workers could invoke Make against
one mutable build tree. A no-NVCC build could therefore overlap a native CUDA
build and produce a false CUDA failure even though the isolated CUDA lane was
correct.

MiniMax additionally retained CLI assertions from superseded component and
process contracts. Those assertions expected parser-owned CUDA component
budgets and a hidden `runtime start` daemon transition after the corresponding
production authorities had moved to typed component bindings and direct
foreground `server` execution.

## Problem

A branch-neutral registry on `main` was insufficient while current vertical
tests remained outside it. Leaving the feature branches on handwritten runner
registration would recreate parallel QA systems immediately. Allowing multiple
Make-backed tests to mutate one build root would also make parallel execution
nondeterministic.

The MiniMax stale tests were not harmless historical wording. They encoded
obsolete product and ownership contracts, including the command transition the
canonical server architecture explicitly removed.

## Causal analysis

The downstream audit established three separate causes:

- DeepSeek added an expert-worklist unit owner and an SM121 SASS validation
  target after the main baseline; neither existed in the original registry.
- MiniMax added media, latent-runtime, live tokenizer, and bounded omni
  transformer owners after the baseline; those tests needed family-specific
  fixture metadata but not a separate framework.
- The orchestrator initially treated resource declarations as optional test
  metadata. Native runners and Make commands all share mutable generated and
  object products, so build-tree exclusivity must be mechanically derived from
  runner kind rather than remembered by each registry row.

The initial concurrent CUDA failure disappeared when the native lane ran in
isolation. After build-tree ownership became generated policy, the complete
DeepSeek and MiniMax changed plans passed their CUDA and no-NVCC obligations in
the same execution. That sequence identifies shared build mutation as the
orchestration cause rather than a CUDA correctness defect.

## Decision

Keep `config/qa/registry.json` and `config/qa/obligations.json` as the only QA
policy authorities. Represent family differences as test identities,
requirements, fixtures, and obligation metadata; do not add family branches to
the orchestrator.

Derive the exclusive `build-tree` resource for every native C runner and every
Make-backed command during registry normalization. Validate that rule with a
registry property test so a future row cannot bypass it by omission.

Merge the resulting `main` normally into each feature branch. Resolve vertical
tests against current production contracts, retain no compatibility assertions
for the removed hidden daemon grammar, and never merge one vertical into the
other.

## Implementation

The main integration history seals the QA foundation and its resource repair
through merge commits `e1f9d77c67017e9f62a8378e2ae86ba7800a0127`,
`97f2d53ba8eb26f828619b3fb094e6a3dcbe11a3`, and
`537c86622ff264a6888ebd32c5e5a8d2cc8be7e0`.

DeepSeek merged that history, registered its expert-worklist and native SM121
evidence, and published checkpoint
`80f2b92b91aa74af65d845ac77b3b7c82532c30d`.

MiniMax merged the same main history, registered its media and latent unit
owners plus distinct audio, text, tokenizer, video, latent, and omni live
fixtures, and published checkpoint
`5f7eca0db6fc06a6a1df67a2cdf20b0e37c186ae`. Its CLI tests now validate the
typed component-binding refusals and the canonical foreground server grammar
that production actually implements.

The dedicated `refactor/qa-test-foundation` branch and worktree were retired
after their complete history was proven reachable from `main`. Large model
artifacts, benchmark roots, logs, and structured QA reports remain untracked.

## After

All three active lines consume one QA architecture:

- `main` owns 127 canonical test identities across 13 lanes;
- DeepSeek owns 128 identities, adding its current expert-worklist boundary;
- MiniMax owns 135 identities, adding its current unit and live vertical
  boundaries;
- native test registration, Make membership projection, prerequisites,
  resource serialization, and result state all derive from the registry;
- both feature heads contain current `main` without containing the other
  feature branch.

Required live or performance prerequisites that are absent produce `BLOCKED`
before execution. They do not load a large artifact, become a skip, or promote
a model/release claim.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Canonical registry on active branches | main only | main 127; DeepSeek 128; MiniMax 135 identities | directly comparable architecture fact |
| Execution lanes | implicit and branch-specific | 13 canonical lanes on all three branches | directly comparable architecture fact |
| Main hosted-CI plan | no canonical shared plan | 95 / 95 PASS | characterization |
| DeepSeek changed plan | ad hoc vertical checklist | 99 PASS; 2 BLOCKED; 0 FAIL; 0 ERROR | characterization |
| MiniMax changed plan | ad hoc vertical checklist | 101 PASS; 8 BLOCKED; 0 FAIL; 0 ERROR | characterization |
| Build-tree resource ownership | row-by-row optional metadata | mechanically derived for native and Make runners | directly comparable ownership fact |
| Vertical QA frameworks | divergent registration surfaces | one orchestrator and registry schema | directly comparable architecture fact |

## Evidence

| Scope | Result | Evidence fact |
| --- | --- | --- |
| Main CI | PASS | run `57fc9758b506745cd99fee8af73dceef3b21b8dc64d1d639769e571614854ea9`; 95 / 95 |
| DeepSeek changed plan | PASS with explicit blockers | run `f621e3313ac8e1198eb6a3a3bd9dce3316dd188b14609cab0cbd2a81aad6f0a1`; 99 PASS, 2 BLOCKED |
| MiniMax changed plan | PASS with explicit blockers | run `cb7d0940b3175d3b618f9dc7f17b96767b75732e09fc5e952e49341e58266a34`; 101 PASS, 8 BLOCKED |
| CUDA and no-NVCC | PASS | both vertical plans exercised native CUDA and fail-closed no-NVCC lanes |
| Sanitizers | PASS | quant and runtime ASan/LeakSanitizer/UBSan aggregates on both vertical plans |
| Numeric/runtime/structural | PASS | affected reference, lifecycle, tiny vertical, CLI/protocol, ownership, layout, and architecture checks |
| DeepSeek live/performance | BLOCKED | selected artifact, runtime binding, and benchmark root were not configured |
| MiniMax live | BLOCKED | required external audio, text, video, transformer, and oracle assets were not configured |
| Static diagnostics | SKIP when selected | optional Clang executable was absent; no mandatory lane was promoted |
| Repository payload scan | PASS | no tracked safetensors, bin, or dat payloads; only bounded GGUF fixtures remain tracked |

The blocked rows are truthful evidence gaps for live model and performance
claims. They are not failures of the branch-neutral QA cutover and were reached
without opening a complete model artifact.

## Remaining limitations

- DeepSeek live generation and performance qualification still require its
  exact artifact, binding, benchmark root, and workload evidence.
- MiniMax live audio, text, tokenizer, video, latent, and omni qualification
  still require their identity-bound external fixtures.
- The optional Clang static lane remains unavailable on this host.
- Coverage remains a diagnostic and has no arbitrary repository-wide gate.
- The release lane composes obligations but is not green and does not establish
  release readiness.
- This delivery changes development QA and test contracts. It claims no model
  quality improvement, runtime performance gain, continuous batching, or GB10
  closure.

## Why it matters

Main, DeepSeek, and MiniMax now answer “what evidence is required?” through the
same executable authority. Vertical semantics remain distinct, while missing
assets, shared resources, and higher-stage non-claims are represented
mechanically instead of being reconstructed from wave prose.

## Communication projections

### Short update

YVEX completed its QA cutover across main, DeepSeek, and MiniMax. The branches
now share one registry, change-aware planner, generated native runner, and
structured result model; their current changed plans passed 99 and 101 checks
respectively, while unavailable live assets remained explicit `BLOCKED`
evidence rather than silent skips.

### Article seed

**Possible title:** One Evidence Architecture for Divergent Model Verticals

**Central thesis:** A generic test framework becomes real only when independently
advancing vertical branches can express different fixtures and claims without
forking orchestration policy.

- Why integrating a registry into main did not complete the cutover.
- Turning build mutation into a typed exclusive resource.
- Migrating DeepSeek and MiniMax without cross-merging product histories.
- Repairing stale tests that encoded removed product architecture.
- Why blocked live prerequisites are stronger evidence than permissive skips.

### Visual candidates

- Branch flow from main QA authority to independent DeepSeek and MiniMax
  registries.
- Resource diagram showing parallel plan resolution with serialized build-tree
  and CUDA ownership.
- Result table separating hermetic PASS from asset-bound BLOCKED evidence.

### Quoteable technical facts

- “Main, DeepSeek, and MiniMax now consume the same 13-lane QA architecture.”
- “Build-tree exclusivity is derived mechanically for native and Make-backed
  tests; individual registry rows cannot omit it.”
- “Missing live artifacts are `BLOCKED` before execution, not treated as a
  successful skip.”
