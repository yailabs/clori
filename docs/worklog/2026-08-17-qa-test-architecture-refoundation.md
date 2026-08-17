# Canonical QA Evidence Architecture

| Field | Value |
| --- | --- |
| Date | 2026-08-17 |
| Type | closure |
| Milestone | `V010.DEVELOPMENT.QA.TEST.ARCHITECTURE.REFOUNDATION.0` |
| Branch | `refactor/qa-test-foundation` |
| Baseline | `93a484be49cc5c0a15139f9c68f54c2ce571869b` |
| Checkpoint | `b1e1689202335458762c6fb82ca0e587390c5993` |
| Subsystem | development QA, test registration, and evidence orchestration |
| Model family | branch-neutral |
| Hardware | hosted CPU-compatible foundation with optional local CUDA/live resources |
| Evidence | registry parity; hermetic lanes; CUDA; sanitizers; coverage characterization |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

YVEX had substantial unit, integration, structural, CUDA, sanitizer, and live
coverage, but test identity and invocation were distributed across manual C
prototypes, manual runner call lists, Make recipes, shell entrypoints, and
wave-specific instructions. The Makefile simultaneously acted as build owner,
test catalog, lane selector, and evidence planner.

A contributor could run useful targets, but the repository could not derive a
canonical answer to which evidence a change required or distinguish a missing
required live asset from a successful qualification. C test additions required
synchronized edits to several handwritten registration surfaces.

## Problem

The test corpus was stronger than its orchestration contract. Validation scope
depended too heavily on contributor memory, duplicated registration could drift,
and skips, unavailable tools, and missing mandatory resources lacked one shared
result model. Local workflows and a future CI adapter had no single evidence
authority to consume.

This was a QA architecture defect rather than a lack of individual tests.

## Causal analysis

The repository had accumulated evidence owners incrementally around semantic
deliveries. Each delivery added the target it needed, while no owner sealed:

- stable test identities and evidence classes;
- lane membership and prerequisites;
- change-to-obligation policy;
- resource exclusivity and cleanup scope;
- structured result and blocked-evidence semantics;
- generated C runner registration.

Directory names and Make targets therefore carried implicit policy that could
not be validated mechanically. Similar names were not assumed to prove
duplication; the retained inventory classifies every legacy target for migration
without deleting tests on naming evidence alone.

## Decision

Make `config/qa/registry.json` the canonical test/evidence catalog and
`config/qa/obligations.json` the canonical change-to-evidence policy. Generate
the secondary C declarations, runner tables, build memberships, and inventory
projections below the selected build directory.

Keep Make as the build/dependency authority while moving test selection,
prerequisite resolution, resource acquisition, status semantics, and evidence
reports into a development-only standard-library Python orchestrator. Preserve
the small native YVEX C harness instead of importing an external test framework.

Use the formal states `PASS`, `FAIL`, `SKIP`, `BLOCKED`, and `ERROR`. Required
CUDA or live evidence that cannot run is `BLOCKED`; it never becomes a green
release result through an implicit skip.

## Implementation

Checkpoint `b1e1689202335458762c6fb82ca0e587390c5993` introduces:

- a registry of 127 stable test identities across 13 execution lanes;
- generated C runner declarations and tables, replacing handwritten parallel
  registration in the unit, CUDA, quant, and artifact runners;
- a change-aware obligation resolver that maps semantic ownership to required
  lanes and explains every selection;
- `tools/qa.py` for validation, listing, diagnosis, planning, execution,
  resource locking, and versioned structured reports;
- worktree- and host-scoped resource locks with ownership checks;
- explicit live-model, CUDA, sanitizer, performance, coverage, static, and
  property seams;
- a hosted CI adapter that invokes the same canonical `ci` lane used locally;
- canonical engineering documentation and contributor/PR guidance;
- engineering-worklog integration that summarizes QA evidence rather than
  inventing a second test database.

Generated projections remain untracked build products. Large live artifacts,
profiler output, coverage data, logs, and structured run receipts remain
identity-bound external or build-local evidence.

## After

The repository can now answer both:

```text
which tests exist and what do they prove?
```

and:

```text
given this semantic change, which evidence is mandatory and why?
```

The current change-aware plan resolves 97 tests across the `ci`, `fast`,
`numeric`, `runtime`, `sanitizer`, and `structural` lanes with no unknown path.
Unknown future ownership fails conservatively into broader evidence rather than
silently reducing coverage.

Local and CI execution now share the same registry, lanes, status model, and
structured evidence schema. A release lane exists as composition policy but
cannot become green while mandatory live evidence is unavailable.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Canonical test/evidence registry | none | 127 stable test identities | directly comparable architecture fact |
| Canonical execution lanes | implicit target families | 13 registry-defined lanes | directly comparable architecture fact |
| Manual C registration authorities | prototypes, runner calls, Make membership | one registry plus generated projections | directly comparable ownership fact |
| Change-aware evidence planning | wave/contributor checklist | 97-test explained plan for this checkpoint | characterization |
| Structured result states | target-specific | `PASS` / `FAIL` / `SKIP` / `BLOCKED` / `ERROR` | directly comparable contract fact |
| Hermetic/live split | implicit | 108 hermetic, 18 live, one diagnostic/performance remainder | registry characterization |

## Evidence

| Lane or check | Result | Evidence fact |
| --- | --- | --- |
| Registry/projection properties | PASS | deterministic generation and mutation refusals |
| CI | PASS | 95 / 95 registered hermetic checks |
| Structural | PASS | 14 / 14 |
| Numeric/reference | PASS | 15 / 15 |
| Runtime | PASS | 21 / 21 |
| CUDA native aggregate | PASS | native CUDA registered lane |
| Sanitizer | PASS | quant and runtime ASan/LeakSanitizer/UBSan boundaries |
| Coverage | PASS | characterization: 40,342 / 81,580 lines; 2,671 / 4,057 functions |
| Static diagnostics | SKIP | optional Clang tool unavailable on the validation host |
| Property seam | PASS | registry mutation and drift properties |
| Live | BLOCKED | 18 / 18 require explicitly configured external assets |
| Performance | BLOCKED | required binding and benchmark evidence root unavailable |
| Repeated build/check | PASS | consecutive composition runs remained clean |

The live and performance results are evidence that mandatory prerequisites are
represented honestly. They are not failures of the hermetic QA foundation and
do not establish model qualification.

## Remaining limitations

- The branch-neutral foundation still requires downstream recanonicalization of
  DeepSeek- and MiniMax-only tests after integration into `main`.
- Full DeepSeek, MiniMax, SM121, and live-model qualification require their
  external artifacts and appropriate hardware.
- Static-analysis execution remains optional until a supported analyzer is
  installed in the execution environment.
- Coverage is diagnostic characterization; no arbitrary repository-wide
  percentage gate was introduced.
- The release lane is a truthful composition boundary, not evidence that YVEX
  is release-ready.
- This change modifies development infrastructure and test registration. It
  claims no runtime, model-quality, or performance improvement.

## Why it matters

YVEX validation obligations are now repository truth rather than an ad hoc
checklist. Missing required evidence remains visible, while contributors and CI
consume the same deterministic plan and the same test identities.

## Communication projections

### Short update

YVEX now has one canonical QA registry and an explainable change-to-evidence
resolver. The initial foundation registers 127 checks across 13 lanes, generates
native runner registration, and distinguishes mandatory blocked evidence from
valid skips. CI and local development consume the same authority; live model
qualification remains explicitly external and unsatisfied where assets are
absent.

### Article seed

**Possible title:** Turning a Large Native Test Corpus into an Evidence System

**Central thesis:** More tests do not automatically yield reliable project
claims; test identity, obligations, resources, and blocked evidence need one
machine-readable authority.

- Why manual registration drift survives even in a well-tested C/CUDA project.
- Separating build ownership from QA selection and result semantics.
- Mapping semantic source ownership to explainable evidence obligations.
- Why `BLOCKED` and `SKIP` cannot be synonyms for live/GPU qualification.
- Keeping hosted CI and operator-driven large-model evidence on one policy.

### Visual candidates

- Authority flow: source ownership and QA registry to plan, execution, and
  structured evidence.
- Before/after registration diagram for native C runners.
- Result-state decision table for optional versus mandatory prerequisites.

### Quoteable technical facts

- “The initial canonical registry contains 127 stable test identities across 13
  execution lanes.”
- “A missing mandatory live artifact is `BLOCKED`; it cannot silently satisfy a
  release lane.”
- “Local development and hosted CI resolve tests from the same QA authority.”
