# Code Commentary Baseline and Migration Audit

Status: frozen point-in-time evidence

Baseline: `7c90ce16bbd59f2ba844432ccc47832e3e27fa7a`

Date: 2026-07-31

## Purpose

This audit records the repository-wide migration from mandatory labeled
function contracts to natural, selective systems commentary. It proves the
population reviewed, the policy and scanner boundary, comment reconciliation,
line-sensitive review, and production token equivalence.

The normative policy is in [`AGENTS.md`](../../../AGENTS.md) and the canonical
guard is `tests/c_structure.py`. This file is evidence, not a second commentary
standard or live project-control surface.

The original delivery named an older project-control commit. The operator
explicitly admitted execution from the then-current published `main`; this
audit is therefore bound to the exact baseline above, after command and
documentation architecture closure.

## Population and method

The governed population came from the commentary roots and suffixes in
`config/c_policy.json`, reconciled against tracked files at the baseline and
working tree. C-family comments were extracted by the repository C lexer;
Python comments by the standard tokenizer; shell and Make comments by a
conservative line scanner. Shebangs are not commentary.

| Population | Count |
| --- | ---: |
| Reviewed first-party files | 374 |
| C files | 257 |
| CUDA files | 1 |
| Headers | 54 |
| Python files | 9 |
| Shell files | 52 |
| Build files | 1 |
| Production C-family files | 211 |
| Interface headers | 50 |

Generated build output, external dependencies, frozen audits, licenses,
attribution, and binary fixture bytes were excluded explicitly. The tracked
GGUF fixture generator was reviewed as first-party maintenance code; its
generated binary vectors were not rewritten or regenerated.

For reconciliation, comments were normalized without changing prose, matched
in file order, and classified as retained, rewritten, added/expanded, or
deleted. A rewrite means an existing comment position now contains different
technical prose; it does not imply a code change.

## Comment reconciliation

| Commentary population | Before | After | Retained | Rewritten | Added or expanded | Deleted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All governed files | 5,449 | 1,536 | 423 | 1,113 | 0 | 3,913 |
| C, CUDA, and headers | 5,241 | 1,341 | 243 | 1,098 | 0 | 3,900 |
| Production C-family | 4,630 | 1,168 | 195 | 973 | 0 | 3,462 |
| Test C-family | 611 | 173 | 48 | 125 | 0 | 438 |

The old `Owner`, `Owns`, `Does not own`, `Invariants`, `Boundary`, `Purpose`,
`Inputs`, `Effects`, and `Failure` labels occurred 19,464 times in baseline
commentary and zero times after migration. Product strings that intentionally
render words such as “boundary” or “failure” are not comments and remain
unchanged.

Fifty-four reviewed files now contain no comments. They are short or
self-explanatory adapters, tests, guards, and tools for which a module or helper
comment added no information:

```text
src/cli/commands/accounts.c
src/cli/commands/artifact.c
src/cli/commands/gguf.c
src/cli/commands/paths.c
tests/c_structure.py
tests/cli/accounts.sh
tests/cli/artifact_corruption.sh
tests/cli/artifact_identity.sh
tests/cli/artifact_integrity.sh
tests/cli/artifact_integrity_regression.sh
tests/cli/artifact_metadata.sh
tests/cli/integrity_report.sh
tests/cli/materialize.sh
tests/cli/materialize_gate.sh
tests/cli/model_gate.sh
tests/cli/models.sh
tests/cli.sh
tests/documentation_architecture.py
tests/integration/openai_sdk.py
tests/live/physical_variant.sh
tests/reference/tokenizer.py
tests/support/validate_runtime_benchmark.py
tests/test_code_natural.sh
tests/test_cuda_failclosed.sh
tests/test_docs_surface.sh
tests/test_gguf_qtype_abi.sh
tests/test_operator_registry.py
tests/test_project_control.sh
tests/test_source_ownership.sh
tests/test_surface.sh
tests/test_topology_closure_audit.sh
tests/unit/artifact_integrity.c
tests/unit/artifact_naming.c
tests/unit/conversion_payload.c
tests/unit/conversion_plan.c
tests/unit/cuda/materialize.c
tests/unit/deepseek_adapter.c
tests/unit/gguf_emit.c
tests/unit/gguf_template.c
tests/unit/imatrix.c
tests/unit/materialize_cpu.c
tests/unit/materialize_gate.c
tests/unit/model_gate.c
tests/unit/model_ref.c
tests/unit/model_registry.c
tests/unit/native_weights.c
tests/unit/qtype_support.c
tests/unit/quant_job.c
tests/unit/quant_policy.c
tests/unit/qwen_adapter.c
tests/unit/safetensors_header.c
tests/unit/weight_mapping.c
tests/unit/weights.c
tools/generate_operator_registry.py
```

## Interface contract coverage

All 50 production interface headers retain useful module-level commentary:

| Tier | Headers | Module commentary |
| --- | ---: | ---: |
| Installed public ABI | 14 | 14 |
| Cross-subsystem internal ABI | 26 | 26 |
| Source-local/platform ABI | 10 | 10 |

The declaration set is unchanged: 385 public function declarations and 837
internal or source-local function declarations. Review retained or rewrote
contracts where ownership transfer, borrowing, lifetime, transactional
visibility, failure preservation, buffer extent, cancellation, or blocking was
not established by types. It removed comments from obvious private helpers
instead of treating comment count as interface quality.

## Representative decisions

- Runtime generation now explains why sampled-token visibility follows the
  matching decode commit and why completed partial progress is not rolled back.
- Runtime graph commentary records the shared candidate-generation boundary
  across logical state and device residency.
- CUDA launch-graph commentary explains exclusive capture and preservation of
  the previous executable graph during a failed update.
- Public ABI headers explain borrowing, lifetime, admission, and capability
  separation rather than repeat header names.
- Straightforward CLI adapters no longer carry headers that merely say they
  parse arguments, call an API, and return an exit code.
- Focused tests retain rationale for independent or transaction-sensitive
  fixtures while routine setup and assertion helpers remain quiet.
- Shell and Python tool headers were reduced to the operational fact that is
  not evident from their command bodies.

No reviewed comment exposed an unresolved code invariant that required a
behavior change. Claim and capability non-promotions were narrowed or retained
only where they protect a real owner boundary.

## Line-sensitive and lexical evidence

Production sources contain no `__LINE__` use and no `#line` directive. Tests
contain two `__LINE__` uses in `tests/test.h`; both only report assertion source
locations and do not derive semantic identity or behavior. No `#line` directive
exists in tests.

For every one of the 211 production C, CUDA, and header files, the scanner
removed comments and insignificant whitespace and hashed the remaining byte
stream. All 211 per-file digests match the baseline. Their reconciled aggregate
identity remains:

```text
645c477e137064a4e4027e9b9ba0a6b1ebc8d077b504a43f27b114e78fbd501b
```

Code lines remain 142,584 and executable lines remain 89,569. Function count
remains 3,985. The change reduces production comment-only lines from 18,202 to
4,474; surrounding whitespace accounts for the remaining physical-line change.

## Guard after-state

The canonical scanner no longer requires a labeled file header or adjacent
comment on every function. It now enforces only reliable policy boundaries:

- complete governed-file enumeration;
- useful module commentary on every interface-header tier;
- absence of obsolete labels and known synthetic boilerplate;
- absence of exact long duplicate commentary and stale executable names;
- absence of commented-out code and ownerless maintenance markers;
- unchanged ownership, layout, dependency, symbol, output, and claim guards.

Generated and third-party exclusions are declared in policy rather than
silently skipped. Human review remains the authority for whether natural prose
contains useful rationale.

## Preserved state and gaps

No ABI, protocol, command, output string, algorithm, control flow, data layout,
synchronization, memory ownership, build product, runtime capability,
readiness, evaluation, benchmark, or release claim changed. The existing
partial GB10 performance state and every higher-capability non-claim remain
unchanged.

Unresolved commentary gaps: none.
