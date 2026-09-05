# Contributing to YVEX

YVEX accepts focused changes that make the native compiler and runtime more
executable, tested, or internally coherent. Read [`AGENTS.md`](AGENTS.md) for
the repository contract and [`ROADMAP.md`](ROADMAP.md) for current project
state before proposing implementation work.

For usage help, start with [SUPPORT.md](SUPPORT.md). Report vulnerabilities
through [SECURITY.md](SECURITY.md), rather than a public issue or pull request.

## Set up a development checkout

Use the [build instructions](README.md#1-build), then run
`python3 tools/qa.py doctor` to inspect the prerequisites for the relevant QA
lane. Repository guards also require `ripgrep`. The official SDK integration
test needs `uv`, `curl`, Node.js, and npm on `PATH`; its pinned SDK packages
are downloaded into external caches or temporary directories. CUDA and live
model requirements belong to their separate lanes.

Use a development branch or fork and submit a pull request against `main`.
Documentation fixes and reproducible bug reports are welcome; you do not need
to know the internal owner to report a user-visible failure.

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
canonical [shared-development policy](AGENTS.md#shared-development) defines
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
3. update `config/source_owners.tsv` in the same patch;
4. regenerate the deterministic build projection with
   `make generate-source-manifest`;
5. preserve source-relative object and archive identities; and
6. run ownership, repository-layout, source-layout, and architecture guards.

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
or `registry.c`. Validate deterministic generated products and current product
projections with:

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
`yvex serve` owns the persistent model, session, KV, worker, and telemetry
lifecycle within the single `yvex` executable.

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

A pull request must state, with detail proportional to the change:

- problem and owned boundary;
- implementation and affected contracts;
- production command/API reachability, or the exact non-applicability reason;
- QA plan identity, required lanes, structured result summary, and any blocked
  mandatory evidence;
- resource, cleanup, cancellation, and failure evidence when relevant;
- capability/non-claim impact;
- progression decision and downstream consumer when a milestone changes; and
- any decision record added or superseded.

Do not mix unrelated cleanup with a capability change. Do not rewrite existing
history solely to normalize old commit subjects.

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

Before advancing a milestone, report unresolved blockers and evidence gaps,
the progression decision, and whether downstream work can safely proceed as
required by `AGENTS.md`. Only an accepted update to `ROADMAP.md` changes the
active macro boundary.

## Security and private data

Do not commit credentials, prompts, generated responses, machine-specific
service environments, private filesystem paths, raw telemetry, or model
payloads. Default logs and errors must not expose prompt/output content,
filesystem layout, credentials, or internal memory.

The [security policy](SECURITY.md) owns vulnerability reporting and the local
trust boundary, including the unauthenticated loopback HTTP listener.

## License and attribution

Contributions to YVEX's original code and documentation are made under the
repository's [MIT license](LICENSE). Submit only material you have the right
to contribute. Preserve third-party copyright and license notices and identify
any imported code or data, its source, and its terms in the pull request and
[NOTICE.md](NOTICE.md) where applicable.

Model weights, tokenizer assets, datasets, and external tools retain their own
terms. Their availability or successful execution does not place them under
YVEX's license. Packaged distributions must retain `LICENSE` and `NOTICE.md`.

## Where information belongs

| Need | Authority |
| --- | --- |
| current milestones, dependency order, gates, non-claims | `ROADMAP.md` |
| repository ownership and contribution invariants | `AGENTS.md` |
| vulnerability reporting and local trust boundary | `SECURITY.md` |
| usage help and public bug reporting | `SUPPORT.md` |
| software license and third-party attribution | `LICENSE` and `NOTICE.md` |
| documentation classes and update policy | `docs/development/documentation-policy.md` |
| repository terminology and invariants | `AGENTS.md` and owning C interfaces |
| implemented system architecture | `docs/architecture/` |
| runtime behavior and failure | `docs/contracts/runtime.md` |
| installed/internal API lifetimes | `docs/contracts/c-api.md` |
| release-gate meanings | `docs/releases/doctrine.md` |
| exact operator workflow | `docs/operator-runbook.md` |
| durable architectural choice | `docs/decisions/` |
| bounded implementation work | GitHub issue |
| implementation review and validation evidence | pull request |

Before editing documentation, identify the admitted implementation or policy
fact, update its canonical owner, and change only projections whose navigation
or bounded summary is affected. Do not request or perform a generic “update all
docs” pass. README changes only when the public entry, public capability,
minimal workflow, release status, limits, or documentation map changes.
