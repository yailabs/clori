# Common Operator Runbook

Date: 2026-07-29
Status: repository validation and hygiene

## Purpose

This runbook covers repository-wide build, test, claim, topology, and artifact
hygiene. It deliberately excludes selected model execution, diagnostic runtime
command atlases, and future command syntax.

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
make
make smoke
```

## Focused Documentation Check

Requires:
  repository root.

Writes:
  nothing.

Safe to rerun:
  yes.

Stop after:
  the project ledger, documentation ownership, exact target, unsupported
  boundaries, and canonical terminology guards pass.

Boundary:
  documentation/claim consistency only.

```sh
make check-docs
sh tests/test_docs_surface.sh
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
make
make smoke
make check
make check-docs
sh tests/test_docs_surface.sh
sh tests/test_surface.sh
sh tests/test_source_layout.sh
sh tests/test_code_natural.sh
sh tests/test_topology_closure_audit.sh
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
make check-cuda
```

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
git diff -- docs MODEL_ARTIFACTS.md AGENTS.md
```

## Operator-Local Cleanup

Never add source weights, emitted artifacts, runtime bindings, registries,
benchmark baselines, JSON/CSV reports, ad hoc SVG charts, logs, pid files,
caches, partial downloads, or generated backend outputs to git. The only chart
exception is the validated deterministic attention set under
`../assets/benchmarks/attention/`, regenerated through its canonical Make
target from external raw evidence.

Before committing:

```sh
git status --short
git diff --check
```

Inspect every staged path explicitly when the worktree already contains user
changes. Do not use an all-files stage operation in a mixed worktree.

## Current Product Boundary

`../../ROADMAP.md` alone owns current project state and sequencing.
`../v010-release-doctrine.md` defines gate semantics, while `deepseek.md`
defines the current operator boundary.

This common runbook contains no model run because model-specific hosted
operation belongs to `deepseek.md`; project support state remains in
`../../ROADMAP.md`.
