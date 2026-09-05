# AGENTS.md

## Repository purpose

YVEX is a native C/CUDA compiler and runtime for identity-bound verified
open-weight inference. Code and tests own capability. Documentation explains
implemented truth and never promotes a fixture, report, or name into support.

Work in this order unless a delivery explicitly owns a policy boundary:

1. code;
2. tests;
3. project control and living documentation.

Model weights, generated packages, runtime dumps, raw profiles, local
registries, credentials, downloaded dependencies, and build products stay out
of Git.

## Shared development

A branch is an integration line, not an agent or model-family identity. In a
shared worktree, HEAD, the tree, and the index may move between operations.

Before editing or committing:

- read the current branch, HEAD, status, staged diff, complete unstaged diff,
  and the exact file/region being changed;
- identify the delivery's semantic owner and affected tests;
- preserve unrelated work, including compatible changes in the same file;
- stage only owned paths or hunks and review the staged diff.

Never use destructive cleanup, whole-tree stashes, rebase of published
history, force push, or mechanical ours/theirs resolution to remove another
delivery. A real conflict is incompatible behavior in the same ABI, function,
record, lines, lifecycle, or invariant; stop only that overlap. Integrate
published histories with merge. Resource locks serialize only the actual
exclusive resource, such as the GPU or a daemon port.

## Source layout and ownership

Implementation lives under `src/`, installed headers under
`include/yvex/`, and tests under `tests/`. Paths are the namespace:
lowercase snake_case names, no `yvex_` source prefixes, no repeated directory
tokens, no root C sources/private headers, no flattened object identities, and
no mechanically paired private files.

`config/source_owners.tsv` is the single production membership and ownership
authority. Every C, CUDA, and header file under `src/` and `include/` appears
exactly once. A new production file must own an ABI, lifecycle, reusable
multi-consumer algorithm, backend/platform boundary, generated boundary,
admitted family recipe, or executable entrypoint. Otherwise extend the
existing owner or keep the helper static.

Headers have three tiers:

- `include/yvex/*.h`: installed public ABI;
- `include/yvex/internal/*.h`: cross-subsystem non-installed ABI;
- `src/<subsystem>/private.h`: source-local ABI used by multiple translation
  units or a required backend boundary.

Public headers are self-contained in C and C++. Production does not include
`yvex/api.h`, internal headers do not include source-private headers, and
source names every dependency explicitly. A non-public global requires an
internal ABI and multiple production consumers; implementation helpers are
`static`.

Source files are limited to 2,000 physical lines, headers to 600, functions to
200, and code to the warning/style policy in `config/c_policy.json`. Fix
warnings at their owner. Do not hide size in macros, generated blobs, or
compressed one-line code.

## Architecture

The dependency direction is:

```text
core/public ABI
  -> source, GGUF, artifact, model, tokenizer
  -> compilation and graph planning
  -> materialization and runtime
  -> backend execution
  -> generation
  -> evaluation and benchmark

domain facts -> typed report -> renderer -> CLI I/O
```

Production never includes tests. Core/model/graph/runtime/generation never
depend on CLI. Generic owners never include family implementations. Planning
does not depend on backend implementation or payload bytes. Backends execute
admitted operations and never reconstruct model topology. Cross-subsystem
include cycles and duplicate global symbols are forbidden.

The main authority split is:

- source owns provenance, inventories, ranges, trust, and delivery;
- compilation owns semantic lowering, physical package legality, and immutable
  bindings;
- artifact owns package identity, admission, mapping, and lifecycle;
- deployment specialization owns hardware-significant admitted implementation
  choices;
- an engine generation owns executable resources and stale-reference
  boundaries;
- sessions own mutable sequence/component state;
- the engine scheduler owns ready execution progress;
- execution batches and expert worklists describe real selected populations;
- backends choose equivalent hardware implementation details;
- evidence observes execution through typed counters/events and does not own it.

Runtime consumes an admitted content-addressed binding. It does not reopen
source inventories, rebuild compiler plans, or switch on family names.

### Family boundary

A family owns source/config interpretation, tensor roles, semantic validation,
declarative schedules, architecture-specific operations, topology/state
meaning, and numerical obligations. Generic owners implement reusable
mechanisms. A family normally has one model source, one optional graph recipe,
one optional fused backend source, and one optional header.

Architecture-specific operations are valid when their semantics, reference,
and backend implementation are explicit. Do not hide a family mechanism under
a generic name or push generic session, sampling, allocation, protocol, or
publication behavior into a family.

### Backend boundary

Upstream declares legal work, numerical class, real populations, dependencies,
and optional prepared representation. Deployment selects an admitted
implementation class. A backend owns buffers, command submission,
synchronization, launch geometry, equivalent kernels, and device profiling.

CUDA types, streams, events, graph handles, warp details, and SM-specific
choices stay below the backend boundary. Capability does not imply
profitability. An unavailable optional acceleration uses a known-correct
admitted fallback; integrity failure, missing mandatory semantics, or an
explicit unsupported exact request fails closed.

### State and identity

Each runtime session owns independent mutable state. Transactional participants
stage candidate state and publish atomically through begin, prepare, commit, or
abort. Semantically different KV, rolling, draft, media, RNG, token-ledger, and
decoder representations share lifecycle coordination, not storage geometry.

Persist or hash a fact only when another lifetime must reopen, authenticate,
cache, invalidate, or inspect it independently. Never hash C object memory,
pointers, padding, local paths, or timestamps. Inside one authenticated engine
generation, prefer compact handles with recoverable cold lineage.

## Public ABI and product surface

One public schema/version identity maps to one layout and semantic contract.
Audit every changed installed record under `include/yvex/`; reject stale
layouts before reading fields absent from them. Wire protocol versions change
only for wire-contract changes. Public headers expose durable concepts, not
provider subprocesses, test injection, CUDA internals, scheduler structures,
or benchmark machinery.

`yvex` is the single product executable. The persistent server may exist with
zero engines; load/unload creates and retires exact engine generations without
restarting host transport. Requests route by model and engine generation.
Public strategy names express semantics such as target-only or speculative,
not an implementation name such as DSpark.

CLI commands consume typed public/application APIs. `src/cli/main.c` only
dispatches; input adapters parse; renderers format typed facts; only CLI I/O and
the server entrypoint write operator output. No UI parses human output,
fabricates telemetry, or reads backend-private state.

## Evidence

Keep evidence ranks distinct:

- software tests prove implementation contracts;
- numerical conformance compares production with an independent reference;
- runtime qualification proves lifecycle, transaction, cancellation, and
  cleanup;
- component benchmarks measure an admitted component;
- model behavior/quality evaluation requires the tokenizer-to-text path and a
  declared scorer;
- release qualification combines all required gates.

Optimized YVEX matching older optimized YVEX is cross-implementation evidence,
not independent authority. A tensor proof is not a complete package;
materialization is not execution; component timing is not a model benchmark.
Missing mandatory evidence is `BLOCKED` or `SKIP`, never a pass.

Performance evidence records exact source/tree, package/binding,
representation, backend/device/runtime, workload, sampler, warm/cold state,
sample count, memory, and source stability. Report measured and derived facts
separately. A performance candidate is judged by throughput, latency, memory,
preparation cost, and numerical effect together.

## Validation

Use the canonical test catalog in `config/qa/registry.json` and change mapping
in `config/qa/obligations.json`:

```sh
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
```

The fast lane is for routine iteration; GB10/live qualification is separate.
Do not run expensive model work for documentation or cosmetic UI changes. Use
`/tmp/yvex-gpu.lock` for exclusive hardware qualification. Resource
contention is `BLOCKED_BY_RESOURCE`, not a functional failure. Evidence from
a source snapshot that moved during execution is invalid and must be rerun.

Always run:

```sh
git diff --check
focused tests for changed owners
affected ownership/layout/architecture/documentation guards
git ls-files '*.safetensors' '*.bin' '*.dat'
git ls-files '*.gguf'
```

Runtime/backend/CUDA/state changes also require the applicable no-NVCC,
numerical, CPU/CUDA, sanitizer, cancellation, rollback, cleanup, and
concurrency lanes. Build topology changes require two consecutive builds or
checks without cleaning.

## Project control and closure

The [engineering method](docs/development/agentic-engineering.md) explains
delivery design and independent review; this file owns mandatory agent rules.

`ROADMAP.md` is the only live macro project-control authority. Issues own
bounded work; PRs own delivery diffs and evidence; ADRs own current durable
decisions. Git history owns retired audits, migrations, milestone plans, and
detailed implementation chronology.

Commits are focused semantic boundaries and use conventional prefixes such as
`feat`, `fix`, `refactor`, `perf`, `test`, `docs`, or `chore`.
Do not commit rejected experiments or generated/runtime assets.

A closure reports exact source/tree, commits, meaningful ownership changes,
tests/evidence, blocked gates, compatibility decisions, remaining non-claims,
and:

```text
progression_decision: proceed | repair_same_boundary | complete_evidence | blocked_external
downstream_safe: true | false
```

`proceed` requires an implemented consumer-safe boundary. A green build alone
does not close missing ownership, rollback, operator reachability, numerical
authority, or required evidence.
