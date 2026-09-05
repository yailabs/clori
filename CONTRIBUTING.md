# Contributing to YVEX

Read [AGENTS](AGENTS.md) for mandatory repository rules,
[ROADMAP](ROADMAP.md) for current macro state, and the
[engineering method](docs/development/agentic-engineering.md) for how
evidence-driven deliveries are designed and reviewed.

For usage help see [SUPPORT](SUPPORT.md). Report vulnerabilities privately
through [SECURITY](SECURITY.md), not a public issue or pull request.

## Set up a development checkout

Use the [build instructions](README.md#1-build), then inspect the prerequisites
for the relevant test lane:

```sh
python3 tools/qa.py doctor
```

Repository guards require ripgrep. The official SDK integration test also
needs `uv`, `curl`, Node.js, and npm; pinned dependencies remain in external
caches or temporary directories. CUDA and real-model requirements belong to
their separate lanes.

External contributors normally use a fork or development branch and a pull
request to `main`. In an existing shared development checkout, preserve the
selected integration line and follow its established ownership; do not create
a branch merely to give an agent a separate identity.

## Before opening work

Search issues and the roadmap boundary. Describe one concrete problem, its
current evidence and consumer, the desired after-state, refusal behavior,
and relevant non-goals. Include a reproducible failure when possible.

Use an ADR for a durable architecture, ownership, ABI, protocol, or policy
decision. Small implementation choices and delivery chronology belong in the
diff and pull request. Do not open speculative infrastructure work solely for
a hypothetical future family.

The [shared-development rules](AGENTS.md#shared-development) govern concurrent
changes, index ownership, and published history. Same-file work is not
automatically a conflict; incompatible changes to the same invariant are.

## Development order

Unless the delivery explicitly owns documentation or policy:

1. reconcile the checkout and inspect current owners;
2. implement at the correct ownership boundary;
3. qualify positive and adversarial behavior;
4. update affected contracts and project state only to the evidence obtained.

For new production membership, use the
[source-ownership workflow](docs/development/source-ownership.md). For command
changes, update the canonical operation registry described by
[command architecture](docs/architecture/commands.md), not generated help
or independent flag tables.

Documentation changes must identify where each retained fact belongs.
The [documentation lifecycle](docs/development/agentic-engineering.md#documentation-lifecycle)
covers admission, compression, historical records, and changelog scope.

## Tests

Use the registered change mapping, with `BASE` set to the reconciled source
boundary:

```sh
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
git diff --check
```

[QA](docs/development/qa.md) owns lane selection, prerequisites, locks,
evidence output, and invalidation. A missing mandatory lane is blocked or
skipped, never passed. Documentation and cosmetic work do not need expensive
model execution. Runtime changes require their mapped lifecycle, numerical,
sanitizer, and backend evidence.

For documentation topology, also run the focused surface checks:

```sh
python3 tests/documentation_architecture.py
sh tests/test_project_control.sh
sh tests/test_docs_surface.sh
```

## Commit and pull request

Make focused semantic commits with conventional prefixes. Stage only owned
paths or hunks and review the staged diff. Never commit weights, runtime
assets, credentials, dependencies, rejected experiments, or generated builds.

The pull request should state:

- the property and meaningful ownership change;
- exact source identity, tests, evidence pointers, and failure/cleanup proof;
- compatibility effects and remaining non-claims;
- any changed public contract, operator procedure, or macro state;
- a progression decision justified by evidence.

Review the actual implementation and refusal paths, not only the author's
summary. Independent repository verification and the distinction between
published facts and local runtime evidence are explained in the
[engineering method](docs/development/agentic-engineering.md#verify-the-delivery).

Follow [AGENTS closure rules](AGENTS.md#project-control-and-closure) for source
stability and downstream safety. License and attribution obligations remain
in [LICENSE](LICENSE) and [NOTICE](NOTICE.md).
