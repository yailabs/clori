# YVEX public ABI versioning repair

| Field | Value |
| --- | --- |
| Date | 2026-08-27 |
| Type | repair |
| Milestone | YVEX.PUBLIC.ABI.VERSIONING.REPAIR.0 |
| Branch | models1 |
| Baseline | e50a77d8576b04ccfd3943ec6a21935774fdf75e |
| Checkpoint | 0152eb2753212055e8b7e13eed64f3656fecb037 |
| Subsystem | installed C ABI, server, QA |
| Model family | not applicable |
| Hardware | NVIDIA GB10, compute capability 12.1, 128 GiB unified memory |
| Evidence | software tests; runtime qualification; numerical conformance; source-stable combined QA |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The branch architectural realignment introduced configured engine capacity in the installed
`yvex_server_options` record. Before that change, schema v3 contained
`request_queue_capacity`, `worker_count`, and the remaining server configuration, but no
`maximum_engines`. The realigned public header added `maximum_engines` while continuing to name
the resulting layout `YVEX_SERVER_OPTIONS_SCHEMA_V3`.

The installed ABI therefore assigned one schema identity to two layouts. Functional server tests
passed because every in-tree caller rebuilt against the new header and used the same v3 constant.
They did not exercise an independently compiled stale caller or enforce that a changed public
record must receive an explicit version decision.

## Problem

An installed schema number must identify one layout and semantic contract. Treating the expanded
record as v3 allowed old bytes to be interpreted as though they contained `maximum_engines` and
made safe pre-field validation impossible. This was a public ABI defect found during post-closure
judge review, not a protocol-v13 defect and not a change to engine-capacity behavior.

The repair also required a complete audit of the installed ABI delta from
`5de22c3ec5edf889f76c0952244c3b1cf55cbf42`. Fixing only the reported record without examining
other moved, added, or changed contracts would not establish that the same versioning error was
absent elsewhere.

## Causal analysis

The previous QA surface favored behavioral compatibility within one source tree. It compiled the
current writers and readers together, so a schema constant left unchanged across a layout change
remained self-consistent. No compiler-owned ledger tied the declared schema identity to stable
layout facts.

The installed-header comparison found one missing version advance: `yvex_server_options`. Other
public changes fell into different categories:

- graph and materialization contracts moved to cohesive installed headers without changing their
  declarations;
- catalog authorities were new or explicit pre-v0.1 unversioned source-API migrations and did not
  reuse a schema identity;
- server summary and console status layouts remained unchanged and gained explicit v1 constants;
- the engine-capacity constants changed as part of the server-options v4 contract;
- protocol v13 and the public server callback/function signatures did not change.

The audit also enumerated every installed record whose first member is `schema_version`. There are
23 such records across provider, quantization, server, and tokenizer APIs. None of the remaining
records reused one version identity for different current layouts.

## Decision

The current server-options layout is schema v4. The historical v3 constant remains public so the
migration is explicit, but the current implementation does not emulate the absent v3 field. A v3
request is rejected before any field after `schema_version` is read. This is the smallest safe
pre-v0.1 compatibility policy: identify the old contract, never reinterpret its bytes, and avoid a
second legacy constructor ABI without a retained consumer.

Protocol v13 remains unchanged because the repaired object is an in-process C options record, not
a wire message. The configured maximum remains distinct from both the implementation safety
ceiling and actual resource admission, and its default remains eight.

Versioned installed records now have a compiler-checked ABI guard. The guard discovers the
current installed records, normalizes declaration tokens so comments and formatting are ignored,
and checks the declared schema plus `sizeof` and `offsetof` facts in C11 and C++17. For server
options it additionally checks every field offset and type and the v3/v4 constant relationship.
The ledger states its LP64 target assumptions instead of presenting the result as a portable ABI
hash.

## Implementation

- Added `YVEX_SERVER_OPTIONS_SCHEMA_V4` and made
  `YVEX_SERVER_OPTIONS_SCHEMA_CURRENT` select v4.
- Preserved the public v3 constant as a migration identity and made server creation explicitly
  reject it before accessing fields absent from the historical layout.
- Updated every production, CLI, integration, fixture, and unit-test writer to use the current
  schema.
- Gave the unchanged server-summary and console-status records explicit v1 schema constants.
- Added `tests/test_public_abi.py` and the `structural.public-abi` QA obligation for installed
  public headers.
- Scoped the changed-file obligation to the explicit installed-header set so an internal header
  cannot acquire public ABI authority through a recursive glob.
- Documented the v3-to-v4 source/API migration and versioning rule in the C API contract and
  changelog.

No model execution, numerical algorithm, prepared-resource policy, protocol message, or backend
path changed.

## After

`YVEX_SERVER_OPTIONS_SCHEMA_V3` names only the historical layout, while the layout containing
`maximum_engines` is unambiguously v4. Current callers use v4. A stale schema fails explicitly and
cannot be treated as the current record. Every current installed versioned record is included in
one deterministic C/C++ layout check, and changing one without updating its declared contract and
ledger fails the structural QA owner.

The complete public-ABI delta has an explicit classification. No other missing public version
advance was found. The architectural realignment remains intact; this repair closes its one known
public ABI identity defect.

## Why it matters

Public callers compile record offsets into their own binaries. A schema name is useful only when
it lets a callee classify those bytes before touching fields that may not exist. Giving two
layouts the same name converts an explicit compatibility boundary into accidental same-tree
behavior. Schema v4 and the compiler-owned layout guard restore that boundary without burdening
normal server execution or inventing a protocol migration.

## Evidence

Focused evidence passed for independent public-header inclusion in C and C++, the ABI ledger,
stale-schema refusal, server and protocol units, persistent-host lifecycle, the tiny production
vertical, source ownership, architecture boundaries, repository layout, documentation contracts,
project control, and `git diff --check`.

The retained exact-tree qualification used implementation checkpoint
`0152eb2753212055e8b7e13eed64f3656fecb037`, tree
`18f58d60141b51ca5cee0b8d3a7bdae6b5996f6b`, clean source-delta identity
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`, build identity
`026eb5aaf4970cf117f5752494c9aa2dc48d17700fa62d1089cf09d8f38ce1ef`, and QA registry identity
`22e8cf3326bc5d75022875e8e01ebd50dd6e557e0f5eaf9c4eda734961e62276`.

Canonical run
`651cd6974263e03dfe925ca9d3359cdca428115a2c0b731c5867701bd54d28c4` completed with 112 PASS,
0 FAIL, 0 ERROR, 0 SKIP, and 0 BLOCKED. HEAD, source state, and source-delta identity were unchanged
between run start and completion. It included CUDA and no-NVCC qualification, ASan/LeakSanitizer,
UBSan, DeepSeek generation, MiniMax exact numerical lanes, persistent host and protocol paths,
performance non-regression evidence, and all structural/unit owners resolved from the baseline.

An earlier source-stable diagnostic run reported one CUDA fault-matrix failure while only about
9.1 GB of device-visible unified memory was free. After releasing cached pages belonging to the
configured model/evidence assets, the focused CUDA fault matrix passed and the retained full run
passed. The diagnostic resource-pressure result was not promoted to qualification evidence.

## Remaining limitations

- Binary compatibility execution for the historical v3 record is deliberately not implemented;
  callers rebuild for v4, and stale v3 input receives a safe explicit refusal.
- The ABI ledger is intentionally tied to the admitted LP64 architecture contract. It is not a
  claim that one raw struct layout is portable across arbitrary C ABIs.
- This repair makes no throughput or model-quality claim. Existing DeepSeek performance remains
  characterization, and MiniMax numerical behavior is unchanged.
