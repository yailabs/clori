# Project-Control Input

This file extracts command/operator obligations from PROJECT.md at baseline
ec7dccede90c1a1efa87b4c2519c25b30d5e1733. It is input to
V010.PROJECT.CONTROL.PUBLIC.0, not a replacement ledger.

The extraction does not add milestone rows, change states, or reinterpret
capability. PROJECT.md remains sole authority until its admitted successor
changes that doctrine.

## Critical-path truth at audit baseline

| ID | State | Audit interpretation |
|---|---|---|
| V010.RUNTIME.DEEPSEEK.PERFORMANCE.0 | partial | startup/profile evidence accepted; warm decode below admission |
| V010.PRODUCT.SURFACE.REALIGNMENT.0 | complete | yvex plus yvexd topology accepted |
| V010.OPERATOR.SURFACE.AUDIT.0 | active | this frozen inventory |
| V010.PROJECT.CONTROL.PUBLIC.0 | blocked | consumes this audit |
| V010.OPERATOR.COMMAND.CONSOLE.0 | blocked | consumes public control and this audit |
| V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 | blocked | later measured runtime continuation |
| V010.EVAL.DEEPSEEK.0 | blocked | follows optimization |
| V010.BENCH.DEEPSEEK.0 | not-measured | follows evaluation |
| V010.RELEASE.0 | blocked | follows all release gates |

## Obligation extraction

The classifications below are about project-control treatment:

- retain: still-open obligation needs a visible macro owner;
- absorb: a valid requirement moves into a current macro owner;
- already complete: implementation exists and only historical evidence remains;
- discard: stale topology/syntax claim must not survive public refoundation.

### Retain — 12

| Legacy ID/owner | Current state | Evidence | Unresolved requirement | Proposed macro issue |
|---|---|---|---|---|
| V010.PROJECT.CONTROL.PUBLIC.0 | blocked | monolithic PROJECT.md | public roadmap, contribution workflow, issue/PR and decision ownership | Public project control |
| V010.OPERATOR.COMMAND.CONSOLE.0 | blocked | 70 command rows; no registry | canonical operations, flags, help, REPL/TUI projections | Command architecture and console |
| V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 | blocked | measured 0.432 tok/s decode | measured attention/MoE/launch/sync optimization | GB10 runtime optimization |
| V010.EVAL.DEEPSEEK.0 | blocked | complete generation exists | behavior and quality evaluation | DeepSeek evaluation |
| V010.BENCH.DEEPSEEK.0 | not-measured | component benchmark only | complete-model benchmark | DeepSeek benchmark |
| V010.RELEASE.0 | blocked | release gates false | release qualification and package/tag evidence | v0.1 release |
| V010.TRACE.3 | planned | raw typed events exist | bounded tensor-role trace only if an admitted consumer needs it | Observability depth |
| V010.LOGITS.10 | planned | full logits exist | logprob diagnostics with typed schema | Evaluation/inspection support |
| V010.KV.15 | planned | contiguous session KV exists | paged-KV plan after measured pressure | Runtime memory depth |
| V010.KV.17 | planned | host-resident state supported | host spill experiment when measured | Runtime memory depth |
| V010.KV.18 | planned | no SSD spill | SSD spill experiment when measured | Runtime memory depth |
| V010.KV.19 | planned | unquantized admitted KV | KV quantization policy when correctness/evidence exists | Runtime memory depth |

These depth rows must not become public commands merely because they exist in
the old ledger. A command appears only with real operator applicability.

### Absorb — 6

| Legacy ID/owner | Current state | Implementation evidence | Unresolved requirement | Absorbing macro issue |
|---|---|---|---|---|
| V010.OPERATOR.RUNTIME.CONSOLE.0 | superseded | event/status/REPL foundations exist | DwarfStar-style console and semantic observability | V010.OPERATOR.COMMAND.CONSOLE.0 |
| V010.CLI.DEEPSEEK.GENERATE.0 | superseded | yvex run/chat over yvexd | no separate direct one-shot milestone | client/runtime history; command-console syntax |
| CLI.ARCH.AUDIT.0 | complete evidence | earlier porcelain/plumbing doctrine | current two-binary grammar disposition | this audit then command-console |
| V010.GEN.16 operator proof | complete | graph transformer generate in offline lane | classify as engineering, not normal product run | command-console |
| V010.KV.14 trace/inspect | complete evidence | session/status/event facts | project authoritative console projection | command-console |
| V010.RUNTIME.1 profile/benchmark surface | complete component | direct graph profile/benchmark leaves | separate engineering profile from future full-model bench | command-console plus benchmark |

### Already complete — 8

| Legacy ID | State | Current implementation evidence | Public-control treatment |
|---|---|---|---|
| V010.RUNTIME.CLIENT.REFOUNDATION.0 | complete | long-lived host, sessions, v3 protocol, REPL | concise architecture history |
| V010.DOCS.README.PRODUCT.0 | complete | product-first README/visual workflow | keep current public entrypoint |
| V010.SERVE.OPENAI.COMPAT.0 | complete | integrated bounded OpenAI profile | keep compatibility scope/non-claims |
| V010.PRODUCT.SURFACE.REALIGNMENT.0 | complete | only yvex/yvexd products | make topology invariant prominent |
| V010.RUNTIME.DEEPSEEK.GENERATION.0 | complete | tokenizer-to-streamed-text | keep capability; no quality claim |
| V010.RUNTIME.DEEPSEEK.TOKENIZER.0 | complete | exact artifact-bound tokenizer/prompt | keep domain owner |
| V010.RUNTIME.SAMPLING.0 | complete | greedy/stochastic typed policy | keep domain owner/default authority |
| V010.COMPILATION.PHYSICAL.VARIANT.1 | complete | policy-driven variants/bindings | keep compile/artifact workflow prerequisite |

### Discard as obsolete — 5

| Historical obligation/claim | Why obsolete | Replacement |
|---|---|---|
| separate yvex-dev product surface | executable retired and routes absorbed | guarded offline lane in yvex |
| separate yvex-openai process/service | listener integrated into yvexd | server/OpenAI adapter owner |
| flat top-level command registry as public doctrine | incompatible cutover complete | audited nested surface then registry |
| CLI-only product claim | yvexd is a persistent local runtime host | two-executable topology |
| status-only yvexd/gateway-shell claim | real model/session/generation/OpenAI serving exists | runtime/server truth |

Historical IDs remain in history where required; obsolete product claims do
not remain active obligations.

## Command and console obligations

The successor project map must retain these exact unresolved facts without
copying the old command catalogue:

1. current grammar is transitional, not canonical;
2. 39 offline routes, product dispatch, slash dispatch, and both help surfaces
   are separate authorities;
3. selected model and live runtime model are distinct;
4. normal hosted paths remain protocol-only;
5. finite engineering paths may link libyvex but never become a daemon;
6. the three false protocol facade IDs require explicit versioned disposition;
7. runtime watch needs semantic rendering over typed events;
8. trace needs human and canonical JSON projections;
9. full turn metrics need prefill, TTFT, decode, final position, and stop;
10. thinking/reasoning can show only explicit model-emitted output;
11. eval and bench command roles remain absent until capability owners close;
12. no dev namespace or compatibility executable may return.

## Open surface macro-issues

### Public project control

Must own:

- concise ROADMAP.md or equivalent macro-roadmap;
- CONTRIBUTING.md;
- issue and PR workflow/templates;
- decision-record ownership;
- stable release/claim doctrine navigation;
- extraction of open work from the legacy ledger;
- one active/next authority during transition; and
- preservation of historical stable IDs without maintaining a public wall.

It changes project-control documentation only and must not promote capability.

### Command architecture and console

Consumes commands.tsv, flags.tsv, operations.tsv, findings.md,
target-taxonomy.md, and registry-input.md.

Acceptance pressure:

- one schema/descriptor authority;
- route/client-lane separation;
- visibility-aware help and machine discovery;
- normalized flags/default references;
- slash adapters;
- selected/live model distinction;
- semantic watch, human trace, raw JSON trace;
- complete authoritative turn/memory/session facts; and
- no hidden reasoning fabrication.

### GB10 runtime optimization

Remains separate. The audit does not change the measured partial performance
state or select expert cache, managed memory, fusion, grouped MoE, graph
capture, or prefetch.

### Evaluation and benchmark

Remain blocked. Component timing, runtime profiling, OpenAI transport, and
operator command availability do not establish behavior evaluation, quality
evaluation, or complete-model benchmark evidence.

## Counts

| Classification | Count |
|---|---:|
| retained open obligations | 12 |
| absorbed obligations | 6 |
| already complete boundaries | 8 |
| obsolete claims/surfaces discarded | 5 |

## Successor safety

V010.PROJECT.CONTROL.PUBLIC.0 can proceed from this audit because it changes
project-control/documentation authority, not executable behavior. It must carry
P1-P3 command findings forward to V010.OPERATOR.COMMAND.CONSOLE.0 rather than
silently treating current spellings as accepted final grammar.

Performance remains partial and unchanged. Evaluation, benchmark, and release
remain blocked.
