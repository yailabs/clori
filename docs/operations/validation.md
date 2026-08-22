# Build and Validation

Status: current repository validation and hygiene procedure

## Purpose

This runbook covers repository-wide build, test, claim, topology, and artifact
hygiene. It deliberately excludes selected model execution, diagnostic runtime
command atlases, and future command syntax.

The canonical lane, registry, requirement, result-state, and structured-report
contract is the [QA architecture](../development/qa.md). This runbook retains
operator-oriented entrypoints and does not duplicate lane membership.

Validation proves only the behavior exercised by each test. Existing hosted
DeepSeek generation does not itself establish model evaluation, the release-
path full-model benchmark, a selected release artifact, or release
qualification. Bounded attention benchmarks do not satisfy those higher gates.

## Fast Build Check

Requires:
  repository root and the normal C toolchain.

Writes:
  ignored files under `build/` and generated root binaries.

Safe to rerun:
  yes.

Stop after:
  build and smoke tests pass.

Boundary:
  build and existing CLI regression only.

```sh
python3 tools/qa.py doctor
python3 tools/qa.py run fast
```

## Focused Documentation Check

Requires:
  repository root.

Writes:
  nothing.

Safe to rerun:
  yes.

Stop after:
  project control, documentation ownership, exact target, unsupported
  boundaries, and canonical terminology guards pass.

Boundary:
  documentation/claim consistency only.

```sh
make check-docs
sh tests/test_docs_surface.sh
python3 tests/documentation_architecture.py
```

## Full Repository Validation

Requires:
  repository root and build dependencies.

Writes:
  ignored build/test output.

Safe to rerun:
  yes.

Stop after:
  every required non-CUDA validation command passes.

Boundary:
  implementation and architecture regression for existing behavior; no
  capability promotion.

```sh
git diff --check
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
python3 tools/qa.py report latest
```

## CUDA Validation

Requires:
  a CUDA-capable host with the repository CUDA dependencies.

Writes:
  ignored CUDA build/test output.

Safe to rerun:
  yes.

Stop after:
  CUDA build, unit, smoke, refusal, and reference checks pass.

Boundary:
  generated-bundle CUDA validation for the currently admitted operations; it
  does not establish model evaluation, full-model benchmark, or release status.

```sh
python3 tools/qa.py run cuda
make test-cuda-native-sm121
```

The native target builds in `build/sm121`, verifies the Tensor Core CUBIN
contains the admitted Q8 row entrypoint and `IMMA.16816.S8.S8` SASS, opens the
native bundle on the real device, and refuses success if the backend selected
PTX instead. Its qtype lane also requires an accounted Tensor Core launch and
numerical agreement with the independent codec oracle.

## Artifact Guardrail

Requires:
  repository root.

Writes:
  nothing.

Safe to rerun:
  yes.

Stop after:
  no model payloads are tracked and every tracked GGUF is a tiny test fixture.

Boundary:
  repository hygiene only.

```sh
git ls-files '*.safetensors' '*.bin' '*.dat'
git ls-files '*.gguf'
```

Expected result:

- the first command prints nothing;
- the second command lists only tiny files under `tests/fixtures/gguf/`.

## Claim Scan

Requires:
  repository root.

Writes:
  nothing.

Safe to rerun:
  yes.

Stop after:
  permanent naturalness and documentation guards pass, and changed canonical
  documents contain no unsupported positive claim.

Boundary:
  claim drift detection only; negative boundary statements remain valid.

```sh
sh tests/test_code_natural.sh
sh tests/test_docs_surface.sh
git diff -- README.md ROADMAP.md CHANGELOG.md CONTRIBUTING.md AGENTS.md docs config/documentation_owners.tsv
```

## Operator-Local Cleanup

Never add source weights, emitted artifacts, runtime bindings, local
registries, benchmark baselines, JSON/CSV reports, generated charts, logs, pid
files, caches, partial downloads, or generated backend outputs to Git.

Before committing:

```sh
git status --short
git diff --check
```

Inspect every staged path explicitly when the worktree already contains user
changes. Do not use an all-files stage operation in a mixed worktree.

## Current Product Boundary

[`ROADMAP.md`](../../ROADMAP.md) alone owns current project state and sequencing.
The [release doctrine](../releases/doctrine.md) defines gate semantics, while
[`deepseek.md`](deepseek.md)
defines the current operator boundary.

This common runbook contains no model run because model-specific hosted
operation belongs to `deepseek.md`; project support state remains in
`../../ROADMAP.md`.
