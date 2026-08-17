# Quality Assurance Architecture

Status: current repository QA and evidence contract

This document owns the development-time QA architecture. It does not own
product capability, release state, model quality, or the semantic tests
themselves. Test owners provide behavior and oracles; the QA layer makes their
identities, obligations, prerequisites, execution, and results canonical.

## Authorities

[`config/qa/registry.json`](../../config/qa/registry.json) is the sole
handwritten catalog of QA identities. Each entry declares its semantic owner,
evidence classes, lanes, runner, requirements, resources, repeat policy,
fixture, and admitted claims. `tools/generate_qa_registry.py` validates it and
generates C declarations, runner tables, Make membership, an inventory, and a
registry identity below the selected `BUILD_DIR`. Generated projections are
never tracked or edited.

[`config/qa/obligations.json`](../../config/qa/obligations.json) maps changed
owners to mandatory evidence. It reuses `config/source_owners.tsv`; it is not a
second source-ownership system. [`tools/qa.py`](../../tools/qa.py) validates and
resolves those authorities, owns resource arbitration and timeouts, executes
tests, and writes versioned reports under ignored `build/qa/` paths.

The Makefile remains the build authority and provides bounded execution
adapters. It is not the QA catalog or change-impact planner. GitHub Actions is
an environment adapter that invokes the same orchestrator and registry used
locally.

## Evidence taxonomy

QA identities may carry more than one evidence class:

- `unit`: one bounded component;
- `component` or `integration`: multiple internal owners composed together;
- `structural`: repository layout, ownership, generated authority, and
  architecture boundaries;
- `numeric` or `reference`: an exact, canonical-conversion, tolerance-qualified,
  or independent reference comparison;
- `runtime` or `transactional`: lifecycle, reset, cancellation, rollback,
  generation, persistence, and cleanup;
- `cuda`: real device execution, capability, parity, faults, and cleanup;
- `sanitizer`: ASan, LeakSanitizer, UBSan, or another explicitly admitted tool;
- `live`: a real identity-bound artifact, binding, backend, and workload;
- `performance`: repeated identity-bound characterization or comparable delta;
- `release`: the composed evidence required by one declared release;
- `coverage`, `static`, and `property`: engineering diagnostics and bounded
  property seams.

Directory placement does not determine evidence class. Model quality and agent
evaluation remain separate higher evidence owners.

## Lanes

The registry defines `fast`, `structural`, `numeric`, `runtime`, `cuda`,
`sanitizer`, `live`, `perf`, `release`, and `ci` gate lanes, plus optional
`static`, `coverage`, and `fuzz` diagnostics. A lane resolves to canonical test
IDs; it is not a copied Make recipe.

`ci` contains only hermetic evidence suitable for ordinary hosted Linux. It
does not pretend to qualify GB10, SM121, or large external model artifacts.
`release` is allowed to resolve mandatory live requirements and therefore
become `BLOCKED` when the required hardware or assets are absent.

## Result states

Every resolved test ends in exactly one state:

- `PASS`: its admitted assertion completed successfully;
- `FAIL`: the test ran and its contract failed;
- `SKIP`: an optional unsupported prerequisite was absent;
- `BLOCKED`: mandatory evidence could not run because a prerequisite was
  absent;
- `ERROR`: orchestration, build, timeout, or execution infrastructure failed.

`SKIP`, `BLOCKED`, and `ERROR` never count as `PASS`. A gate with mandatory
`BLOCKED` evidence is not green and cannot support a downstream-safe claim.

## Normal workflow

From the repository root:

```sh
python3 tools/qa.py doctor
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
python3 tools/qa.py report latest
```

`plan --changed` reports changed paths, matched change classes, mandatory
lanes/tests, and the reason for each obligation. Unknown ownership expands to
a conservative CPU composition plan. It never silently narrows evidence.

Direct use remains available for investigation:

```sh
python3 tools/qa.py list --lane runtime
python3 tools/qa.py explain unit.runtime_binding
python3 tools/qa.py run fast
python3 tools/qa.py run unit.runtime_binding
```

Stable Make conveniences are `make qa`, `make qa-fast`, `make qa-structural`,
`make qa-cuda`, `make qa-ci`, and `make qa-doctor`. Product CLI grammar does
not expose QA.

## Registry additions

To add a test:

1. implement it with its semantic owner and a discriminating fixture;
2. add one stable entry to `config/qa/registry.json`;
3. declare evidence, requirements, resources, timeout, repeat, and claim;
4. add a change obligation only when existing owner patterns do not express
   the consumer;
5. run `make check-qa-registry` and the resolved plan.

C unit and CUDA functions are discovered and checked against the registry.
Tests do not add prototypes to `tests/test.h`, calls to `tests/test.c`, and a
Make membership list independently. The generated projection provides all
three runner surfaces deterministically. Command scripts must be registered or
explicitly classified as non-executable support.

## Fixtures and numerical authority

A fixture declares its owner and semantic purpose in the registry. Checked-in
fixtures are small, deterministic, and adversarial enough to distinguish the
paths they claim to test. Generated fixtures have a deterministic generator.
Large models and bindings remain external identity-bound operator assets.

Accelerated or quantized tests state whether their authority is bit-exact,
equivalent after canonical conversion, tolerance-qualified, or behavioral at
complete-model composition. Tolerances belong to the numeric owner. A focused
kernel oracle does not admit a production fast path when complete-model
composition changes output.

## Resources and cleanup

Registry resources describe genuinely exclusive assets such as a mutable build
tree, fixed port, CUDA device, large live model slot, or benchmark directory.
The orchestrator locks only the declared resource. Host-scoped GPU, model,
port, and benchmark locks coordinate across worktrees; fixture and build locks
remain local to one worktree. Each run owns its report and log root. A test may
delete only paths it created and validated. Completed lanes must not leave
server, profiler, or accelerator processes behind.

The registry normalizer assigns `build-tree` to every generated native runner
and Make-backed command. They cannot execute concurrently against the same
worktree build products merely because a handwritten resource list omitted that
mechanically implied ownership. Tests without that dependency remain eligible
for parallel execution.

Missing mandatory live assets are `BLOCKED`. Configure their environment
variables explicitly; paths and identities remain outside Git. `qa doctor`
reports prerequisites before an expensive lane starts.

## Performance evidence

Performance evidence records source and dirty identity, build identity,
hardware/backend facts, artifact and binding identities, workload, profiler
mode, warmup, samples, distribution, and output identity where the owning test
provides them. Comparisons are classified as `characterization`, `comparable`,
or `regression`. A missing compatible baseline yields characterization, not an
automatic performance pass. Hardware-scoped baselines and raw profiles remain
external unless separately admitted for durable retention.

## Structured reports

Each run writes `yvex.qa.evidence.v1` JSON below `build/qa/evidence/`. The report
contains the registry/run/source/build identities, invocation, host facts,
resolved plan, test IDs, result states, duration, prerequisite reasons,
fixture/evidence metadata, and summary counts. Logs are per-test files
referenced by the report, not embedded unbounded output.

Engineering worklogs summarize applicable QA evidence by lane and state. They
do not copy the registry or become a second test database.

## CI, sanitizers, static analysis, coverage, and property seams

Hosted CI invokes the canonical `ci` lane. Local GPU/live runners use the same
registry and result schema. Sanitizer compatibility is declared per test;
unsupported segments remain explicit rather than disabling a sanitizer
silently. Static analysis and coverage are optional diagnostics when their
tools are available. Coverage locates evidence gaps and has no arbitrary
repository-wide percentage gate.

Bounded property tests target parsers, schemas, registries, state machines, and
other pure validation seams. A retained failure includes a reproducible input
or seed. Random reruns are not a flakiness policy.

## Branch-neutral vertical qualification

DeepSeek and MiniMax register different family fixtures and live requirements
through this same architecture. The orchestrator understands metadata,
resources, and evidence classes; it contains no family execution policy.
Feature branches merge the current `main` QA substrate and extend the registry
for their own additional tests. They never merge into one another to transport
QA infrastructure.
