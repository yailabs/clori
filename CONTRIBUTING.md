# Contributing to YVEX

YVEX accepts focused changes that make the native compiler and runtime more
executable, tested, or internally coherent. Read [`AGENTS.md`](AGENTS.md) for
the repository contract and [`ROADMAP.md`](ROADMAP.md) for current project
state before proposing implementation work.

## Before opening work

Search existing issues and the active roadmap boundary first. A good issue has:

- one concrete problem and one semantic owner;
- current implementation evidence rather than an intended design alone;
- an explicit downstream consumer;
- an executable after-state and refusal behavior;
- focused positive, failure, cleanup, and conformance tests as applicable;
- capability and release non-claims; and
- a progression classification when it changes a milestone boundary.

Use a decision record when a proposal changes architecture, ownership,
protocol, public ABI, executable topology, project doctrine, or a durable
tradeoff. Small implementation choices belong in code contracts and the pull
request, not in a decision record.

Do not open placeholder work for hypothetical family needs. Common owners
change when a concrete supported consumer exposes a missing invariant.

## Concurrent development

Concurrent agents and branches are supported; a branch is not an agent
identity or a model-family identity. Shared development branches and shared
physical worktrees are supported. Divide work by delivery and semantic scope,
preserve unrelated changes, reread mutable source before editing, and stage
only owned paths or hunks. Same-file changes are valid when their semantics are
distinct; report real overlaps instead of overwriting another delivery. The
canonical [concurrent-agent policy](AGENTS.md#concurrent-agent-development) defines
same-branch collaboration, conflict handling, and resource-specific
coordination.

Branch topology does not determine source ownership or QA scope. A coherent
generic change may land directly on an active shared development branch, but
must qualify every affected supported family. Separate family branches remain
available when useful. Published histories use merge, not rebase, and are
never force-pushed. `main` remains the accepted integration branch and is not
used for normal development.

## Development order

Unless a delivery explicitly owns documentation or project doctrine, work in
this order:

1. implementation;
2. focused tests and failure proof;
3. project control and documentation.

Documentation records implemented truth. It cannot establish runtime,
evaluation, benchmark, or release capability.

## Ownership and layout

The directory is the namespace. Before adding or moving production code:

1. identify the existing semantic owner and all real consumers;
2. prove that a new translation unit, header, subsystem, or family file meets
   the admission rules in `AGENTS.md`;
3. obtain explicit authorization where required;
4. update `config/source_owners.tsv` in the same patch;
5. regenerate the deterministic build projection with
   `make generate-source-manifest`;
6. preserve source-relative object and archive identities; and
7. run ownership, repository-layout, source-layout, and architecture guards.

The owner manifest is the sole handwritten production membership list. The
root Makefile consumes its generated projection; do not add a parallel source
list or a wildcard that silently admits unowned files. Validate the projection
with `make check-source-manifest`.

Prefer extending a strong existing owner over introducing a phase-shaped file,
forwarding header, compatibility shell, or duplicate registry.

Command and operation projection metadata has one source:
`config/operator/registry.json`. When changing a command path, argument, flag,
visibility, adapter, protocol projection, or slash projection, update that
source and its owned tests; do not edit `build/generated/operator/registry.h`
or `registry.c`. Validate deterministic generated products and the frozen audit
mapping with:

```sh
make generate-operator-registry
make check-operator-registry
make test-operator-registry
```

The generated descriptors are compiled into `yvex`; the JSON source is not an
installed runtime dependency. Domain defaults and semantic admission remain
with their typed owners rather than being copied into the registry.

Runtime-facing `yvex` commands remain protocol-only. Offline commands may call
admitted engine APIs but must terminate and never become a second daemon.
`yvexd` remains the only persistent model, session, KV, worker, and telemetry
authority.

## Code and contracts

- Use native C/CUDA and the existing typed failure style.
- Keep public, internal, and source-private headers in their declared tiers.
- Make allocation, lifecycle, identity, I/O, state, and failure contracts
  explicit next to non-trivial functions.
- Use checked arithmetic and bounded parsing.
- Never hash pointers, padding, process-local paths, timestamps, or native C
  object memory into semantic identities.
- Fix warnings at their semantic owner; do not add broad suppressions.
- Format production code with the repository `.clang-format` policy.
- Do not create a separate fast path that bypasses admission or correctness.

Model/family facts select policies and schedules. Generic owners retain common
mechanisms. Backend code executes admitted operations and does not rediscover
model topology.

## Tests

Every behavior needs a positive test and every refusal needs a failure test.
Numeric work also needs an independent reference or tolerance contract, edge
cases, cleanup, and backend comparison where applicable.

Test identities and evidence obligations are repository truth. Resolve them
before handoff rather than constructing a wave-specific checklist:

```sh
python3 tools/qa.py doctor
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
python3 tools/qa.py report latest
```

The [QA architecture](docs/development/qa.md) defines lane contents,
requirements, result states, resource ownership, live assets, and structured
reports. Add tests through `config/qa/registry.json`; do not manually add a
prototype, runner call, and Make membership as separate authorities.

Always run:

```sh
git diff --check
git ls-files '*.safetensors' '*.bin' '*.dat'
git ls-files '*.gguf'
```

Then run focused tests for the changed owner plus affected documentation,
ownership, layout, and architecture guards. Changes to executable behavior also
require the applicable build, CLI acceptance, integration, refusal, and
cleanup checks. Runtime/backend/CUDA/state changes require the relevant
no-`nvcc`, sanitizer, cancellation, rollback, and live CPU/CUDA evidence.

Build-topology or generated-dependency changes require two consecutive
successful checks without cleaning.

Do not add a test to production objects. Keep complete artifacts, runtime
bindings, traces, benchmark/profile records, generated charts, downloaded
dependencies, and local registries outside Git.

## Commit and pull request

Use one focused Conventional Commit:

```text
type(scope): imperative description
```

Valid types are `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `build`,
`ci`, and `chore`. Use the narrow semantic owner as scope. Milestone IDs are
not commit subjects.

A pull request must state:

- problem and owned boundary;
- implementation and intentional non-changes;
- production command/API reachability, or the exact non-applicability reason;
- QA plan identity, required lanes, structured result summary, and any blocked
  mandatory evidence;
- resource, cleanup, cancellation, and failure evidence when relevant;
- capability/non-claim impact;
- progression decision and downstream consumer when a milestone changes; and
- any decision record added or superseded.

Do not mix unrelated cleanup with a capability change. Do not rewrite existing
history solely to normalize old commit subjects.

## Engineering worklogs

After a material checkpoint, repair, architectural cutover, comparable
performance change, or milestone closure, invoke the repository skill with:

```text
$engineering-worklog
```

The skill is the procedural authority for semantic records, evidence handling,
publication review, and optional communication projections. Drafts remain under
the ignored `build/worklog/` tree; only intentionally selected records enter
[`docs/worklog/`](docs/worklog/2026-08-11-adaptive-memory-admission.md). Worklogs
do not replace pull-request evidence, project control, evaluation, benchmarks,
or release qualification, and they do not publish anything automatically.

## Capability and evidence language

Use the lowest truthful evidence class:

- software test;
- numerical conformance;
- runtime qualification;
- component benchmark;
- model behavior evaluation;
- model quality evaluation;
- agent runtime evaluation; or
- release qualification.

A fixture is not a complete text path. A complete artifact is not a supported
artifact. Component timing is not a model benchmark. OpenAI-compatible JSON is
not model evaluation. One successful request is not release qualification.

Before advancing a milestone, classify every unresolved item as a gate blocker,
boundary incompleteness, evidence gap, deferred depth, optimization debt,
generalization debt, or external blocker according to `AGENTS.md`. Only an
accepted update to `ROADMAP.md` changes the active macro boundary.

## Security and private data

Do not commit credentials, prompts, generated responses, machine-specific
service environments, private filesystem paths, raw telemetry, or model
payloads. Default logs and errors must not expose prompt/output content,
filesystem layout, credentials, or internal memory.

The current server is local and loopback-only. Do not expose it remotely or
claim authentication/TLS without a separately admitted security boundary.

## Where information belongs

| Need | Authority |
| --- | --- |
| current milestones, dependency order, gates, non-claims | `ROADMAP.md` |
| repository ownership and contribution invariants | `AGENTS.md` |
| documentation classes and update policy | `docs/development/documentation-policy.md` |
| canonical terminology | `docs/doctrine/glossary.md` |
| implementation-independent architecture | `docs/reference/verified-inference.md` |
| implemented system architecture | `docs/architecture/` |
| runtime behavior and failure | `docs/contracts/runtime.md` |
| installed/internal API lifetimes | `docs/contracts/c-api.md` |
| release-gate meanings | `docs/releases/doctrine.md` |
| exact operator workflow | `docs/operator-runbook.md` and `docs/operations/` |
| durable architectural choice | `docs/decisions/` |
| bounded implementation work | GitHub issue |
| implementation review and validation evidence | pull request |

Before editing documentation, identify the admitted implementation or doctrine
fact, update its canonical owner, and change only projections whose navigation
or bounded summary is affected. Do not request or perform a generic “update all
docs” pass. README changes only when the public entry, public capability,
minimal workflow, release status, limits, or documentation map changes.
