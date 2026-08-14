# YVEX Roadmap

Date: 2026-08-10
Status: living public project control
Release target: v0.1.0

This file is the sole live authority for YVEX macro milestones, dependency
order, release-gate state, explicit non-claims, and `Active Next`. It is kept
short enough to review as a whole.

Implementation contracts remain with their owning code and technical
documents. GitHub issues own bounded implementation work; pull requests own
delivery evidence; decision records own durable architectural choices. None of
those surfaces may silently change the order or claims recorded here.

## Product target

YVEX v0.1.0 targets identity-bound DeepSeek-V4-Flash-DSpark text generation on
an NVIDIA DGX Spark / GB10 CUDA system from a complete GGUF artifact produced
by YVEX. The accepted product topology is:

```text
yvex  one public executable with finite client/offline modes and an explicit
      foreground server mode
```

`yvex server MODEL [--ctx N]` directly enters the foreground server lifecycle
and owns the model, worker, queue, sessions, KV, Unix-domain protocol, loopback
OpenAI-compatible listener, and runtime telemetry. Runtime-facing client
operations cross the local protocol. Finite compiler and engineering operations
may link engine owners but never become another hosted runtime.

The current implementation can compile admitted physical variants, open one
complete model, generate streamed text, retain exact multi-turn sessions, and
serve the bounded `yvex.openai.compat.v2` profile. This establishes executable
product capability, not model quality, release benchmark, or release
qualification.

## Control model

One macro milestone is `active`, and the same ID appears exactly once as
`Active Next`. A milestone ID is stable after publication. A successor or
`superseded` marker records structural change; IDs are never silently reused.

States used here are:

| State | Meaning |
| --- | --- |
| `active` | the single milestone currently being implemented |
| `blocked` | retained work whose declared predecessor is incomplete |
| `partial` | useful accepted work landed, but the stated gate remains open |
| `not-measured` | required benchmark evidence does not yet exist |
| `complete` | the stated boundary is implemented and validated |
| `superseded` | retained naming/history marker replaced before completion |

Technical evidence uses the lowest truthful rank defined in
[`AGENTS.md`](AGENTS.md). Documentation, a command, a fixture, one successful
run, or a self-authored report cannot promote a capability by itself.

## Current sequence

| Order | Milestone | State | Owned after-state | Depends on |
| ---: | --- | --- | --- | --- |
| 1 | `V010.PROJECT.CONTROL.PUBLIC.0` | `complete` | Public roadmap, contribution workflow, issue/PR templates, decision records, compact open-work extraction, and project-control guards replace the retired monolithic ledger. | `V010.OPERATOR.SURFACE.AUDIT.0` |
| 2 | `V010.OPERATOR.COMMAND.ARCHITECTURE.0` | `complete` | One versioned operation authority drives command paths, flags, defaults, validation, help, discovery, protocol projections, slash catalog, and command tests. | `V010.PROJECT.CONTROL.PUBLIC.0` |
| 3 | `V010.DOCS.INFORMATION.ARCHITECTURE.0` | `complete` | One governed information architecture separates doctrine, reference and implemented architecture, family facts, contracts, operations, development policy, project control, decisions, audits, migrations, releases, and history. | `V010.OPERATOR.COMMAND.ARCHITECTURE.0` |
| 4 | `V010.REPO.CODE.COMMENTARY.0` | `complete` | Natural selective technical commentary replaces mandatory labeled prose across every governed first-party source while production lexical tokens remain unchanged. | `V010.DOCS.INFORMATION.ARCHITECTURE.0` |
| 5 | `V010.OPERATOR.REPL.CONSOLE.0` | `complete` | A mature server-backed linear console consumes the canonical operation authority and renders truthful status, progress, metrics, the unified server log, and cancellation; explicit reasoning remains conditional on an admitted typed channel. | `V010.REPO.CODE.COMMENTARY.0` |
| 6 | `V010.REBASE.DEEPSEEK.DSPARK.0` | `complete` | The sole DeepSeek vertical is rebound to the exact DSpark source and reaches target-verified speculative text through the hosted product path. | `V010.OPERATOR.REPL.CONSOLE.0` |
| 7 | `V010.PRODUCT.ARCHITECTURE.REFOUNDATION.0` | `complete` | Identity-bound execution profiles, Physical Execution IR, prefix-addressable candidate state, shape-safe CUDA admission, exact partial turns, typed device views, and operational projections form one verified execution substrate. | `V010.REBASE.DEEPSEEK.DSPARK.0` |
| 8 | `V010.REPO.ARCHITECTURE.COMPRESSION.0` | `complete` | One source-membership authority and fewer ceremonial owners, internal ABIs, symbols, build declarations, targets, and duplicate registry facts preserve the accepted execution behavior in a directly navigable repository. | `V010.PRODUCT.ARCHITECTURE.REFOUNDATION.0` |
| 9 | `V010.CORE.COMPILATION.FAMILY.CONSOLIDATION.0` | `complete` | Family semantics terminate in compiler-owned Semantic Model IR and canonical Operator Graph IR; generic passes seal one Physical Execution IR and runtime binding that model-open authenticates and instantiates without reconstructing family topology. | `V010.REPO.ARCHITECTURE.COMPRESSION.0` |
| 10 | `V010.DEVELOPMENT.ENGINEERING.WORKLOG.0` | `complete` | One repository skill turns selected material checkpoints, repairs, comparable performance changes, and closures into evidence-backed semantic records without automatic publication or product coupling. | `V010.DOCS.INFORMATION.ARCHITECTURE.0` |
| 11 | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | `active` | Measured warm-runtime work addresses the proven attention, MoE, launch, synchronization, movement, and batching owners without creating another execution path. | `V010.CORE.COMPILATION.FAMILY.CONSOLIDATION.0` |
| 12 | `V010.EVAL.DEEPSEEK.0` | `blocked` | Repeatable behavior, quality, tokenizer, regression, long-context, and refusal evaluation runs over the accepted hosted path. | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| 13 | `V010.BENCH.DEEPSEEK.0` | `not-measured` | Reproducible full-model latency, throughput, memory, reliability, and workload evidence is bound to exact source, artifact, runtime, and machine identities. | `V010.EVAL.DEEPSEEK.0` |
| 14 | `V010.RELEASE.0` | `blocked` | All v0.1.0 software, conformance, runtime, evaluation, benchmark, packaging, operator, claim, and tag gates close together. | `V010.BENCH.DEEPSEEK.0` |

Active Next: V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0

Detailed accepted and successor contracts for the current sequence are:

- [Canonical Operation and Command Architecture](docs/milestones/command-architecture.md)
- [Documentation and Information Architecture](docs/milestones/documentation-architecture.md)
- [Natural Technical Commentary](docs/milestones/code-commentary.md)
- [Mature Runtime Console and Interactive REPL](docs/milestones/runtime-console-repl.md)
- [DeepSeek V4 Flash DSpark Rebase](docs/milestones/deepseek-dspark-rebase.md)
- [Verified Execution-Substrate Refoundation](docs/milestones/product-architecture.md)
- [Repository Architecture Compression](docs/milestones/repository-compression.md)
- [Core Compiler and Family Consolidation](docs/milestones/core-compilation-consolidation.md)
- [Repository-Native Engineering Worklog](docs/milestones/engineering-worklog.md)
- [Model-Derived GB10 Execution](docs/milestones/gb10-optimization.md)

The pre-implementation combined plans remain traceable without staying on the
active path:

| Milestone | State | Successor |
| --- | --- | --- |
| `V010.OPERATOR.RUNTIME.CONSOLE.0` | `superseded` | `V010.OPERATOR.REPL.CONSOLE.0` |
| `V010.OPERATOR.COMMAND.CONSOLE.0` | `superseded` | `V010.OPERATOR.COMMAND.ARCHITECTURE.0` then `V010.OPERATOR.REPL.CONSOLE.0` |

## Accepted foundations

These are current implementation facts consumed by the open sequence. This is
not a replacement historical ledger.

| Boundary | Current truth |
| --- | --- |
| Source and compilation | The pinned 48-shard DSpark source, tokenizer, 72,317-tensor coverage, 1,409-terminal Transformation IR, bootstrap physical policy, qtypes, and deterministic GGUF emission have typed owners and identity-bound outputs. |
| Compiler and family boundary | Compiler-facing family adapters seal source-authored facts into Semantic Model IR and a canonical Operator Graph IR. Generic passes seal Physical Execution IR and one content-addressed binding; runtime model-open authenticates and instantiates that compiled truth without consulting a concrete family implementation. |
| Artifact and admission | One complete DSpark bootstrap artifact contains the target and drafter; structural, payload, roundtrip, materialization, and one binding containing target/draft/verification plans are admitted outside Git. |
| Runtime and generation | One authenticated model opens in the foreground `yvex server` process; server-owned sessions retain exact target KV/token truth and bounded candidate state; target-only and target-verified DSpark text run on CPU and the admitted mixed CUDA/host path. |
| Application serving | Local protocol v10 and YVEX OpenAI Compatibility Profile v2 share one runtime, scheduler, session registry, and telemetry authority; bounded session fork composes immutable prefix sharing with independent semantic state. |
| Product surface | `yvex` is the sole product executable. Its explicit foreground `server MODEL` mode hosts the model; `chat` and `run` remain protocol clients, while finite compiler and engineering lanes remain bounded offline operations. |
| Command architecture | `yvex.operator.registry.v1` generates immutable descriptors compiled into `yvex`; canonical operation IDs drive the truthful command taxonomy, lane-safe dispatch, help, JSON discovery, completion, and slash-command schemas. |
| Documentation architecture | Canonical doctrine, terminology, reference and implemented architecture, family records, contracts, operations, development policy, audits, migrations, and release surfaces have separate governed owners. |
| Code commentary | Every governed first-party source follows the natural selective-commentary doctrine; the structural guard rejects obsolete templates and boilerplate while production lexical tokens remain unchanged. |
| Engineering worklog | One repository-scoped Codex skill captures selected material semantic changes and evidence in retained records; ignored drafts and optional communication projections create no product or publication authority. |
| Runtime console | The server-backed `yvex>` console renders one composed attachment view, protocol-authored prefill progress, direct model output, typed final metrics, registry-derived slash discovery and completion, the canonical `server log` observability stream, server cancellation, and bounded terminal restoration including Ctrl-D. |
| Operator audit | The frozen post-cutover audit inventories 70 route-level commands, 426 command/flag pairs, 99 semantic operations, 10 slash commands, 14 protocol operations, 5 HTTP endpoints, and every Make/script/environment surface with zero unmatched categories. |
| Performance | `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` remains `partial`: startup and bounded profiling are accepted; warm decode remains below admission. |

The frozen audit is under
[`docs/audits/operator-surface-ec7dcc/`](docs/audits/operator-surface-ec7dcc/README.md).
Its tables describe baseline `ec7dccede90c1a1efa87b4c2519c25b30d5e1733` and
are implementation input, not live command or project authority.

## Open work

### GB10 runtime optimization

The earlier performance delivery reduced cold startup and added internal
profiling, but did not optimize warm decode. The continuation starts from
measured owners rather than a preselected mechanism. Current accepted
observations include approximately 0.794 prompt tokens/s, 0.432 decode
tokens/s, attention at 43.88% of measured warm execution, MoE at 19.29%, about
4,511 kernel launches and 63 synchronizations per token. These are diagnostic
baseline facts, not a release benchmark.

The active implementation now derives execution geometry from one sealed model
descriptor and admits binding v13 as the complete compiler authority; older
bindings are refused because they cannot represent the compiled sparse/large-row
MoE alternatives and crossover carried by Physical Execution IR v3.
Hardware, workload, capacity and state-page facts remain separate. Admission
refuses insufficient model-residency memory before artifact open. Host graph
state commits through stable per-class virtual pages; a phase roofline ledger
and identity-bound native `sm_121` CUBIN coexist with portable PTX. On
Driver-VMM hardware, CUDA session
state now reserves stable logical banks and commits only the physical granules
reached by visible or pre-admitted candidate spans; the non-VMM path remains an
explicit full-bank fallback. The admitted Q8_0/Q8_K native path now has a
mandatory SM121 SASS proof for `IMMA.16816.S8.S8` plus native numerical and
launch-accounting coverage. Physical Execution IR now admits a measured
sparse/large-row MoE crossover. Exact grouped attention output projection now
collapses the model-derived output-A group launches for both decode and bounded
prefill without changing the compiler-selected F32 activation representation.
Specialized Tensor Core coverage beyond those paths, the remaining attention
stack, real deep-context qualification and the optimized serving after-state
remain open.

The continuation selects expert placement, cache, grouped execution, fusion,
prefetch, graph capture and kernel order only from measured phase economics.
Correctness, identity, transactional KV, cancellation, and fail-closed CUDA
behavior remain mandatory.

### Evaluation, benchmark, and release

Evaluation measures model behavior and quality over the accepted runtime path.
Benchmarking then records reproducible full-model performance. Release
qualification consumes both plus the complete software, operator, package,
claim, and artifact evidence defined in the
[release doctrine](docs/releases/doctrine.md).

### Deferred depth

These retained needs are not on the current critical path and do not create
public commands by existing in the roadmap:

| Stable ID | Later owner | Admission trigger |
| --- | --- | --- |
| `V010.TRACE.3` | observability depth | an admitted consumer requires bounded tensor-role trace |
| `V010.LOGITS.10` | evaluation/inspection support | a typed log-probability consumer and schema exist |
| `V010.KV.15` | runtime memory depth | measured pressure justifies paged KV |
| `V010.KV.17` | runtime memory depth | measured pressure justifies host spill |
| `V010.KV.18` | runtime memory depth | measured pressure justifies SSD spill |
| `V010.KV.19` | runtime memory depth | a correctness and evidence contract admits KV quantization |

## Release gates

Gate meanings and closure evidence are normative in
[`docs/releases/doctrine.md`](docs/releases/doctrine.md). Current state
is summarized here only to control progression.

| Gate | State |
| --- | --- |
| verified source, architecture, mapping, transformation, quantization, artifact, integrity, materialization, runtime descriptor | complete for the admitted DeepSeek vertical and named physical variants |
| CUDA transformer-to-text generation | complete for the admitted mixed CUDA/host execution contract |
| long-lived runtime, sessions, streaming, and bounded local OpenAI compatibility | complete |
| public command architecture | complete |
| mature runtime console | complete |
| DSpark source rebase and verified speculative generation | complete |
| warm GB10 performance admission | blocked; earlier work remains partial |
| model behavior and quality evaluation | blocked |
| full-model release benchmark | not measured |
| release qualification | blocked |

Machine-readable readiness facts retained for claim guards:

```text
canonical_operation_registry_ready=1
generated_command_descriptors_ready=1
protocol_v8_ready=1
mature_repl_console_ready=1
deepseek_dspark_source_ready=1
target_only_generation_ready=1
dspark_verified_generation_ready=1
server_log_renderer_ready=1
explicit_reasoning_renderer_ready=1
model_behavior_evaluation_ready=0
model_quality_evaluation_ready=0
full_model_release_benchmark_ready=0
release_qualification_ready=0
continuous_batching_ready=0
multi_model_server_ready=0
remote_server_ready=0
authentication_ready=0
tls_ready=0
```

## Public workflow

Contributors start with [`CONTRIBUTING.md`](CONTRIBUTING.md). The workflow is:

1. locate the active macro boundary here;
2. open or select one bounded issue with an owner, consumer, acceptance, tests,
   and non-claims;
3. record architectural decisions under
   [`docs/decisions/`](docs/decisions/README.md) when ownership or doctrine
   changes;
4. implement code, then tests, then project control and documentation;
5. attach exact validation and progression classification to the pull request;
6. update this file atomically only when accepted evidence changes macro state.

Issue labels and boards may organize work, but they do not override this file.
An issue closes implementation scope; a pull request closes delivery evidence;
only an accepted project-control change advances `Active Next`.

## Historical traceability

The retired 2,701-line `PROJECT.md` ledger is preserved in Git history at
commit `447257dca7b122bafbddb86073d55eaa7be9513f`. It contains 696 canonical
historical IDs and the recovery-era track accounting. It is historical
evidence, not a file that contributors must maintain in parallel.

Stable IDs from that ledger remain stable. Completed, superseded, or deferred
rows are recovered with Git when needed; they are not copied into another
public wall. The frozen operator audit preserves the open-obligation extraction
used for this cutover.

## Current non-claims

YVEX does not currently claim:

- a full-screen TUI or an explicit reasoning channel for a runtime profile that
  does not advertise one;
- warm decode optimization or 5, 10, or 20 tokens/s admission;
- DSpark acceleration, optimized block verification, or production load-aware
  confidence scheduling;
- model behavior or model quality evaluation;
- a release-grade full-model benchmark;
- release qualification;
- a public or remote production server, authentication, TLS, or remote
  security;
- the full OpenAI API, Anthropic compatibility, hosted tools, multimodal
  OpenAI input, or model-server tool execution;
- continuous batching, multi-model serving, distributed serving, or sessions
  persisted across server restart;
- a second complete model-family vertical.
