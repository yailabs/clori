# YVEX Roadmap

Date: 2026-07-31
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

YVEX v0.1.0 targets identity-bound DeepSeek-V4-Flash text generation on an
NVIDIA DGX Spark / GB10 CUDA system from a complete GGUF artifact produced by
YVEX. The accepted product topology is:

```text
yvex   one finite public command process
yvexd  one long-lived runtime process
```

`yvexd` owns the model, worker, queue, sessions, KV, Unix-domain protocol,
loopback OpenAI-compatible listener, and runtime telemetry. Runtime-facing
`yvex` operations cross the local protocol. Finite compiler and engineering
operations may link engine owners but never become another hosted runtime.

The current implementation can compile admitted physical variants, open one
complete model, generate streamed text, retain exact multi-turn sessions, and
serve the bounded `yvex.openai.compat.v1` profile. This establishes executable
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
| 3 | `V010.OPERATOR.REPL.CONSOLE.0` | `active` | A mature daemon-backed linear console consumes the canonical operation authority and renders truthful status, progress, metrics, watch, trace, cancellation, and explicit model-emitted reasoning channels. | `V010.OPERATOR.COMMAND.ARCHITECTURE.0` |
| 4 | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | `blocked` | Measured warm-runtime work addresses the proven attention, MoE, launch, synchronization, movement, and batching owners without creating another execution path. | `V010.OPERATOR.REPL.CONSOLE.0` |
| 5 | `V010.EVAL.DEEPSEEK.0` | `blocked` | Repeatable behavior, quality, tokenizer, regression, long-context, and refusal evaluation runs over the accepted hosted path. | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| 6 | `V010.BENCH.DEEPSEEK.0` | `not-measured` | Reproducible full-model latency, throughput, memory, reliability, and workload evidence is bound to exact source, artifact, runtime, and machine identities. | `V010.EVAL.DEEPSEEK.0` |
| 7 | `V010.RELEASE.0` | `blocked` | All v0.1.0 software, conformance, runtime, evaluation, benchmark, packaging, operator, claim, and tag gates close together. | `V010.BENCH.DEEPSEEK.0` |

Active Next: V010.OPERATOR.REPL.CONSOLE.0

Detailed accepted contracts for the two immediate operator milestones are:

- [Canonical Operation and Command Architecture](docs/milestones/command-architecture.md)
- [Mature Runtime Console and Interactive REPL](docs/milestones/runtime-console-repl.md)

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
| Source and compilation | The pinned DeepSeek source, tokenizer material, complete tensor coverage, Transformation IR, physical policy, qtypes, and deterministic GGUF emission have typed owners and identity-bound outputs. |
| Artifact and admission | Complete source-faithful, Q8_0/Q2_K, and mixed IQ2_XXS/Q2_K artifacts exist outside Git; structural, payload, roundtrip, materialization, and binding admission are implemented. |
| Runtime and generation | One authenticated model opens in `yvexd`; server-owned sessions retain exact KV and token state; the complete tokenizer-to-streamed-text path runs on CPU and the admitted mixed CUDA/host path. |
| Application serving | Local protocol v4 and YVEX OpenAI Compatibility Profile v1 share one runtime, worker, queue, session registry, and telemetry authority. |
| Product surface | `yvex` and `yvexd` are the only product executables. The former developer and OpenAI gateway executables are retired. |
| Command architecture | `yvex.operator.registry.v1` generates immutable descriptors compiled into `yvex`; canonical operation IDs drive the truthful command taxonomy, lane-safe dispatch, help, JSON discovery, completion, and slash-command schemas. |
| Operator audit | The frozen post-cutover audit inventories 70 route-level commands, 426 command/flag pairs, 99 semantic operations, 10 slash commands, 14 protocol operations, 5 HTTP endpoints, and every Make/script/environment surface with zero unmatched categories. |
| Performance | `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` remains `partial`: startup and bounded profiling are accepted; warm decode remains below admission. |

The frozen audit is under
[`docs/audits/operator-surface-ec7dcc/`](docs/audits/operator-surface-ec7dcc/README.md).
Its tables describe baseline `ec7dccede90c1a1efa87b4c2519c25b30d5e1733` and
are implementation input, not live command or project authority.

## Open work

### Runtime console and REPL

The REPL is a client attached to an already resident daemon. Its successor must
provide one concise attachment view, a stable `yvex>` prompt, semantic prefill
and decode progress, complete final metrics, session/context/KV facts,
visibility-aware slash commands, semantic watch, human trace, canonical JSONL
trace, and bounded terminal behavior. It must never simulate model loading or
expose hidden chain of thought.

### GB10 runtime optimization

The earlier performance delivery reduced cold startup and added internal
profiling, but did not optimize warm decode. The continuation starts from
measured owners rather than a preselected mechanism. Current accepted
observations include approximately 0.794 prompt tokens/s, 0.432 decode
tokens/s, attention at 43.88% of measured warm execution, MoE at 19.29%, about
4,511 kernel launches and 63 synchronizations per token. These are diagnostic
baseline facts, not a release benchmark.

The continuation may select expert placement, cache, grouped execution,
fusion, prefetch, or graph capture only after the owning profile supports that
choice. Correctness, identity, transactional KV, cancellation, and fail-closed
CUDA behavior remain mandatory.

### Evaluation, benchmark, and release

Evaluation measures model behavior and quality over the accepted runtime path.
Benchmarking then records reproducible full-model performance. Release
qualification consumes both plus the complete software, operator, package,
claim, and artifact evidence defined in the
[v0.1.0 release doctrine](docs/v010-release-doctrine.md).

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
[`docs/v010-release-doctrine.md`](docs/v010-release-doctrine.md). Current state
is summarized here only to control progression.

| Gate | State |
| --- | --- |
| verified source, architecture, mapping, transformation, quantization, artifact, integrity, materialization, runtime descriptor | complete for the admitted DeepSeek vertical and named physical variants |
| CUDA transformer-to-text generation | complete for the admitted mixed CUDA/host execution contract |
| long-lived runtime, sessions, streaming, and bounded local OpenAI compatibility | complete |
| public command architecture | complete |
| mature runtime console | active |
| warm GB10 performance admission | blocked; earlier work remains partial |
| model behavior and quality evaluation | blocked |
| full-model release benchmark | not measured |
| release qualification | blocked |

Machine-readable readiness facts retained for claim guards:

```text
canonical_operation_registry_ready=1
generated_command_descriptors_ready=1
protocol_v4_ready=1
mature_repl_console_ready=0
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

- a mature REPL or TUI;
- warm decode optimization or 5, 10, or 20 tokens/s admission;
- model behavior or model quality evaluation;
- a release-grade full-model benchmark;
- release qualification;
- a public or remote production server, authentication, TLS, or remote
  security;
- the full OpenAI API, Anthropic compatibility, hosted tools, multimodal
  OpenAI input, or model-server tool execution;
- continuous batching, multi-model serving, distributed serving, or sessions
  persisted across daemon restart;
- a second complete model-family vertical.
