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
5. preserve source-relative object and archive identities; and
6. run ownership, repository-layout, source-layout, and architecture guards.

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

Do not add a test to production objects. Keep full model artifacts, runtime
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
- tests and commands actually run;
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

A fixture is not a complete model. A complete artifact is not a supported
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
| technical architecture | `docs/reference-architecture.md` |
| runtime behavior and failure | `docs/contract.md` |
| installed/internal API lifetimes | `docs/api.md` |
| release-gate meanings | `docs/v010-release-doctrine.md` |
| exact operator workflow | `docs/operator-runbook.md` and runbooks |
| durable architectural choice | `docs/decisions/` |
| bounded implementation work | GitHub issue |
| implementation review and validation evidence | pull request |
