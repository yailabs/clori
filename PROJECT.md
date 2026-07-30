# YVEX Project Control

Date: 2026-07-30
Status: living engineering control
Authority: product target, architecture tracks, complete wave ledger, milestone
state, dependencies, evidence rank, family scope, release gates, and Active Next
Recovery baseline: pre-refoundation `docs/spine.md` at commit `10ad6c3`
Current proof stage: the accepted product topology exposes one public `yvex` command process and one long-lived `yvexd` runtime process. Runtime-facing commands cross local protocol v3, offline engineering commands share the finite `yvex` process without becoming a hosted model authority, and the bounded OpenAI-compatible listener runs inside `yvexd`. The identity-bound post-cutover surface audit is complete with zero unmatched executable, flag, slash, protocol, HTTP, Make, script, and environment/configuration inputs. Public project-control refoundation is now active before the canonical command architecture, the separately owned mature runtime REPL, measured optimization, and release-path evaluation work

## 1. Authority And Update Contract

`PROJECT.md` is the single project-control authority for YVEX. It is the map
used by maintainers and agents to answer, without consulting Git history:

- what YVEX is building;
- which target closes the current release;
- which model-family and common engineering work already exists;
- which stable architecture track owns a change;
- which conclusive milestone is active;
- which wave IDs are complete, planned, blocked, reopened, deferred, partial,
  superseded, or not measured;
- which report, fixture, selected proof, CLI, topology, and diagnostic rows
  remain useful;
- which milestone consumes each subordinate row;
- which dependency and release gate a milestone closes.

Git history is recovery evidence, not a substitute for current project state.
No row may silently disappear, change owner, change ID, or be merged. Record an
explicit migration or successor when project structure changes.

No second authority, compatibility copy, redirect, archive, or shadow ledger
may exist. Technical contracts, API references, runbooks, and point-in-time
audits may live elsewhere, but none owns the track registry, wave state, Active
Next, family matrix, critical path, or release truth.

### 1.1 Atomic Update Rule

At milestone start and closure, update in one patch:

1. milestone state and exact before/after capability;
2. owning track summary and calculated counts;
3. dependencies and newly unblocked successor;
4. evidence added, completed, absorbed, reopened, or superseded;
5. affected model-family scopes;
6. release gates and non-claims;
7. the single Active Next.

A new row has exactly one stable ID and one owning track. Only a `milestone`
row may become Active Next.

## 2. Rank, State, And Proof Semantics

Rank states what a row can prove. State says what happened to that exact row.
They are independent: a complete report remains evidence, not a milestone.

| Rank | Meaning | Independently schedulable | May close a product gate |
| --- | --- | --- | --- |
| `milestone` | Conclusive architectural or functional after-state with one owner, a real downstream consumer, hard acceptance, tests, and a dependency transition | yes, when Active Next | only its owned gate |
| `capability` | Durable implemented behavior below a product gate | no | no by itself |
| `evidence` | Report, fixture, selected proof, diagnostic state, reference comparison, trace, audit, checksum, or bounded fact | no | no |
| `subtask` | Retained requirement assigned to a conclusive milestone | no | no |
| `migration` | Superseded project structure or naming marker with an explicit successor | no | no |
| `future` | Deferred post-v0.1 work requiring a later scope decision | no | no |

| State | Exact meaning |
| --- | --- |
| `complete` | The row outcome is implemented and validated at its declared rank and proof stage |
| `active` | The single current first-class milestone |
| `partial` | Useful work landed, but the stated milestone outcome did not close |
| `blocked` | A milestone has unmet predecessors |
| `planned` | A retained requirement or non-active supporting milestone |
| `reopened` | Earlier evidence was narrower than the claimed boundary |
| `not-measured` | Benchmark evidence does not exist |
| `deferred` | Explicitly outside the active release |
| `superseded` | Replaced and retained only for traceability |

A row is not milestone-complete merely because it adds a report, CLI command,
structured output, fixture, selected tensor, wrapper, source move, plan, or
documentation. Those are evidence or subtasks unless the same row implements
and tests a conclusive downstream-consumed after-state.

## 3. Product, Release, And Engineering Scope

### 3.1 v0.1.0 Product Outcome

YVEX generates real text with DeepSeek-V4-Flash on the DGX Spark CUDA backend
from a complete GGUF artifact produced by YVEX.

Canonical source:

```text
$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash
```

Future canonical full target:

```text
deepseek4-v4-flash
```

The existing aliases remain bounded proof surfaces until an owning milestone
absorbs or removes them:

```text
deepseek4-v4-flash-selected-embed
deepseek4-v4-flash-selected-embed-rmsnorm
```

They are not supported targets, complete model artifacts, or release-path
stages. The selected v0.1.0 physical profile is
`deepseek-v4-flash-q8_0-q2_k-v1`: exact scalar roles remain F32, BF16, or I32,
ordinary quantizable matrices use Q8_0, and routed-expert tensors use Q2_K.
Its encoded payload is 102,396,843,592 bytes. This is measured quantization
evidence, not a DGX Spark residency or runtime-fit claim.

DeepSeek-V4-Flash is the only model whose complete source-to-text chain closes
v0.1.0. This release decision does not make DeepSeek the only engineering
scope.

### 3.2 Multi-Family Engineering Scope

Source intake, tensor mapping, qtypes, GGUF, integrity, materialization,
residency, descriptor projection, backend operation, operator, evaluation, and
release owners remain reusable and family-aware.

Qwen, Gemma, and dense/common work already implemented remains active project
state at its truthful rank. It supplies real regression evidence and prevents
DeepSeek-specific behavior from leaking into common owners. It does not claim
Qwen or Gemma runtime generation. GLM remains planned and does not require a
canonical runbook until implementation starts.

Family-specific architecture, tokenizer, attention, position, KV, MoE, and
execution behavior enter through typed profiles and explicit milestones. Common
owners must not branch on ad hoc target strings when a typed family boundary is
required.

### 3.3 Real Generation Contract

```text
prompt text
-> exact tokenizer
-> full prefill
-> family-correct attention, position handling and MoE
-> real KV writes
-> repeated decode that reads prior KV
-> transformer-owned final mHC collapse and final norm
-> output-head projection
-> vocabulary logits
-> sampling
-> token append and stop policy
-> detokenized text
```

The complete chain must execute over the complete YVEX-produced GGUF for the
exact target. CLI acceptance, selected tensors, synthetic logits, reports,
printed fixture tokens, and bounded diagnostic loops do not satisfy it.

### 3.4 Artifact Terminology

| Term | Canonical meaning |
| --- | --- |
| Tensor proof artifact | One tensor or a bounded subset used to prove only a named parser, layout, materialization, primitive, or lifecycle property |
| Complete model artifact | Every tensor and metadata item required to execute one exact model target |
| Supported model artifact | A complete artifact that passes integrity, materialization, runtime, generation, evaluation, benchmark, and release gates |

The unqualified term "model artifact" never means a selected-tensor proof file.

### 3.5 Model Compilation Boundaries

YVEX treats model compilation and model execution as related but non-equivalent
systems. A logical model is identified by its verified source and family-correct
semantics, not by GGUF or another physical container. GGUF remains the required
v0.1.0 release container and one concrete physical lowering of that logical
model.

| Boundary | Owned meaning |
| --- | --- |
| Source snapshot | Exact repository, revision, configuration, tokenizer, index, shard, source-tensor, and metadata identity. |
| Payload capability | Trusted or locally sealed bytes exposed through immutable ranges, bounded reads, resource limits, cancellation, and transactional delivery. |
| Logical model | Architecture semantics and canonical tensor roles independent of container, qtype, placement, and backend. |
| Transformation plan | Immutable artifact-neutral operations deriving physical tensor values from exact logical source contributions without reading payload bytes. |
| Physical model variant | One identified result of lowering a transformation plan under explicit artifact, precision, hardware, memory, quality, and workload constraints. |
| Artifact serialization | A concrete external representation of one physical variant; GGUF is the v0.1.0 representation. |
| Materialization | An admitted artifact's immutable tensor bindings and bounded access/session lifecycle; it is neither serialization nor execution. |
| Runtime binding | A content-addressed immutable association among the physical variant, admitted artifact, materialization facts, runtime descriptor, attention plan, residency requirements, and backend requirements; mutable execution state remains session-owned. |
| Execution evidence | Correctness, quality, memory, IO, and performance measurements bound to an exact source, plan, variant, machine, and workload. |

The planning plane is metadata-only:

```text
verified source facts
-> architecture IR
-> tensor coverage
-> source contribution map
-> transformation IR
-> physical lowering plan
```

The byte-execution plane consumes the already verified source-byte capability:

```text
verified payload session
-> exact bounded source chunks
-> transformation execution
-> quantization/conversion
-> artifact writer
```

Planning does not read model payload bytes. Payload readers do not interpret
family roles or transformation semantics. Quantization consumes canonical
transformation truth and may not rediscover source names, roles, aggregation
axes, or scaling companions.

Future variant planning is a constrained multi-objective problem. For source
model `S`, hardware profile `H`, workload profile `W`, and candidate plan `p`,
admission requires at least:

```text
coverage(p) = 1
payload_binding(S, p) = 1
capability(H, p) = 1
memory(p, H) <= M_H
quality_drift(p, W) <= epsilon
```

An admitted planner exposes a Pareto set; it does not claim one universal
optimum. Predicted estimates remain distinct from measurements owned by
`TRACK.EVAL` and `TRACK.BENCH`. Evidence may eventually bind artifact build
time and size, payload and transformation bytes, load time, time to first
token, prefill/decode throughput, storage-tier peaks, KV, scratch, SSD traffic
and stalls, quality drift, and energy only when an owning measurement exists.

The memory model distinguishes source storage, artifact storage, SSD staging,
future inference-time SSD tensor streaming, host memory, unified memory, device
memory, persistent KV, and temporary scratch. Completed source payload
streaming is build-time source access; it is not inference-time SSD expert
streaming.

### 3.6 Engineering Construction Method

YVEX is constructed through human-directed, repository-grounded, independently
verified state transitions. The method separates architectural authority,
reasoning assistance, repository implementation, and acceptance so that no
conversation, prompt, agent, or report can promote itself into project truth.

#### Authority And Roles

The human project authority owns the product outcome, scope, priority,
trade-offs, architectural judgment, interpretation of evidence, and final
acceptance or rejection. Reasoning LLMs and repository coding agents assist
that authority; neither replaces it.

| Role | Owned contribution | Explicit non-ownership |
| --- | --- | --- |
| Human project authority | Chooses product direction, scopes work, resolves trade-offs, judges architecture, interprets evidence, and accepts or rejects results. | Does not delegate project authority merely by requesting analysis or implementation. |
| Reasoning LLM | Compresses large repository contexts, studies primary sources and mature implementations, compares alternatives, exposes hidden assumptions, separates facts from inferences and proposals, and compiles intent into ownership and acceptance criteria. | Does not own repository truth, capability state, or capability promotion. |
| Repository coding agent | Reads the actual owners and contracts, implements one bounded delta, chooses coherent internal APIs after inspection, tests success/refusal/failure/cleanup, and produces a candidate patch, commit, and closure assertion. | Does not redefine the product outcome or certify its own candidate as accepted truth. |
| Independent verifier | Reconstructs the accepted baseline, reads the complete remote diff, checks ownership and consumers, executes proportionate validation, audits claims and project-state changes, and returns pass or repair. | Does not infer acceptance from the implementation agent's report or activity volume. |
| Repository | Stores the accepted implementation, contracts, tests, guards, evidence boundaries, and project decisions used by later work. | Does not treat chat history, prompts, or unmerged candidates as durable state. |

Conversations are exploratory working memory. The repository is durable project
memory. Any conclusion that changes later implementation must become an owned
contract, type boundary, test, guard, acceptance criterion, reference decision,
or explicit project-control update. Chat transcripts, delivery prompts, and
closure reports may explain a candidate; they never substitute for the current
repository.

#### Delivery And Candidate Implementation

A delivery is the intermediate representation between architectural intent and
a repository patch. A technical delivery combines:

```text
persistent repository contracts
+ one row-specific delta
+ mandatory ownership
+ implementation freedom inside that ownership
+ hard acceptance criteria
+ focused and complete validation
```

The delivery defines the required after-state and forbidden shortcuts. It does
not invent a final internal API before the coding agent has inspected current
types, owners, consumers, failure conventions, and tests. Implementation freedom
never overrides the mandatory boundary or acceptance criteria.

The coding agent's closure report is an assertion requiring independent
verification. The verifier checks the remote commit and baseline, complete
diff, owner placement, real consumers, duplicate state, focused tests, full
validation where applicable, refusal/failure/cleanup behavior, claim limits,
and any proposed project transition. An implementation candidate may contain
an atomic ledger transition, but no successor implementation proceeds until
that closure is accepted. Rejection preserves the previously accepted state
and produces a repair delta or reopens the same outcome before progression.

Progress is the sequence of accepted state transitions and behavior consumed
by a real downstream owner. Token use, session duration, diff size, file count,
report length, generated-code volume, relocation, renaming, and diagnostic
output alone are not completion evidence.

#### Vertical-First Generalization

YVEX develops common architecture under pressure from exact working verticals:

```text
exact vertical
-> implementation pressure
-> observed invariant
-> common owner
-> second-family pressure
-> harden or split
-> preserved working verticals
```

DeepSeek-V4-Flash is the current release vertical and supplies concrete
attention, mHC, MoE, low-precision, payload, and memory constraints. It is not
permission to put DeepSeek names or semantics into common source, qtype, GGUF,
artifact, residency, backend, or runtime owners. Qwen, Gemma, and later exact
models act as falsifiers of common assumptions even when they are not release
targets.

A mechanism may be common from its first consumer when its contract is
intrinsically format-, storage-, arithmetic-, lifecycle-, or backend-general.
Model semantics remain family-specific until repeated evidence establishes a
stable invariant. When another family contradicts an abstraction, strengthen
the common contract or split ownership at the real semantic boundary. Never
add a target-name conditional to conceal the contradiction, and never weaken a
working vertical to preserve a false abstraction.

#### References, Audit, And Learning Feedback

Primary model sources and papers define model semantics. Specifications define
formats and ABIs. Mature repositories provide comparative implementation
evidence. Hardware documentation defines physical constraints. Only YVEX tests
and identity-bound measurements determine YVEX capability. Reusing an external
design transfers none of its API, process topology, support matrix, benchmark,
or claim.

Implementation is also an architectural experiment. A false assumption,
missing constraint, misplaced owner, invalid abstraction, or inadequate test
revealed by a wave must return to persistent contracts before later work
depends on it. The source-payload/Transformation-IR separation is one such
feedback transition: implementation evidence showed that byte delivery and
transformation semantics require distinct owners. Knowledge that remains only
in a conversation or closure report is not part of the architecture.

## 4. Current Hard Truth

| Boundary | Current truth |
| --- | --- |
| Project control | The 631-ID historical ledger is recovered and ranked here, then organized across 26 canonical tracks; project recovery, compilation-architecture rebasing, both public README cutovers, repository semantic compression, C canonicalization, and role-based product-surface realignment are complete. The physical-variant compiler, runtime/client refoundation, bounded application-provider milestone, partial full-model performance delivery, remaining surface-refoundation sequence, measured optimization continuation, and reopened real Bet-tennis evidence bring the canonical set to 694 IDs. `config/source_owners.tsv` registers every production source/header exactly once; explicit header tiers, source-relative object/archive identities, semantic contracts, and repository layout, ownership, dependency, complexity, and ABI gates are hard failures. |
| Source | The canonical DeepSeek target is verified against `deepseek-ai/DeepSeek-V4-Flash` at commit `60d8d70770c6776ff598c94bb586a859a38244f1`. The pinned upstream index, structured configs, tokenizer assets, 46/46 safetensors headers, 69,187 unique tensor records, and verifier-owned manifest agree. Manifest v3 binds every shard to its authoritative Hugging Face Git LFS SHA-256 and atomically publishes aggregate payload identity `e22b3678d131d334f154a93214bdddfafc172c9869f4c52db28fea198eaa9165` only after all 159,617,149,040 shard-file bytes pass digest and drift checks. |
| Family profiles | The exact DeepSeek-V4-Flash source projects to one immutable typed IR with 43 main layers, one MTP layer, explicit SWA/CSA/HCA, mHC, MoE, position/KV, output, tokenizer, and source constraints. Runtime numeric schema v2 binds BF16/F32 compute boundaries, activation fake-quantization, UE8M0 scaling, deterministic top-k, and the full Hadamard authority at commit `e7706faf8d1c3b9f241e36860640ad1dac644ede`. SWA uses base RoPE without YaRN; compressed CSA/HCA classes use the versioned YaRN policy. Qwen, Gemma, and dense/MoE evidence remains at its prior rank. |
| Tensor coverage | One immutable IR-derived requirement set reconciles exactly against all 69,187 tensors in the verified DeepSeek snapshot: every entry is assigned once, with zero missing, duplicate, ambiguous, unsupported, or unexpected tensors and zero payload reads. Qwen/Gemma evidence remains at its prior rank. |
| Mapping | The sealed artifact-neutral Transformation IR now projects to 1,360 immutable GGUF lowering descriptors: 1,328 pinned-standard trunk descriptors and 32 explicit YVEX MTP extension descriptors. The GGUF adapter adds names, qtypes, metadata, and emitted layout without reconstructing transformation semantics; all 69,187 contributions and mapping identity `1aecbbe25b04de0d` remain exact. |
| Compilation | Schema-v1 immutable Transformation IR owns 69,187 source values, 1,360 typed terminal operations, 69,187 edges, deterministic topology, checked shapes/dtypes/axes, canonical identity `cc774dffb6aa3a8e9f507b1dd454fbf7f5c68187138736f9a330ee9eaec07067`, and an exhaustive payload-range binding. Policy-v2 presets or custom rules resolve every terminal with deterministic priority/conflict semantics into one pointer-free physical-variant plan while reading zero payload bytes. Exact sizing precedes execution; automatic precision/target-size optimization, placement selection, feedback, and Pareto search remain deferred. |
| GGUF foundations | The canonical row-aware qtype storage ABI, scalable native GGUF v3 structural reader ABI, and global directory-order layout admission are closed. `V010.GGUF.ARTIFACT.ABI.0` remains reopened as bounded historical evidence. |
| Qtype | Pinned GGUF identities, removed/outside-baseline refusal, exact scalar/block geometry, and shape-aware storage accounting are canonical. One numeric capability registry owns deterministic codec, calibration, and CPU/CUDA compute truth. F32, F16, BF16, I32, Q8_0, Q2_K, IQ2_XXS, and MXFP4 codecs and direct encoded row-dot proofs are complete. IQ2_XXS uses exact 256-value/66-byte geometry, imatrix-weighted encoding, an independent decoder, and bit-identical CPU/GB10 CUDA row compute; unsupported identities refuse explicitly. |
| Artifact | One immutable GGUF v3 writer plan consumes canonical lowering, quantization, tokenizer, and provenance owners. It emitted three complete DeepSeek artifacts outside the repository: source-faithful is 177,680,573,600 bytes with identity `f16e800c0d7383ee76cb2e2fa8bdd674bab29c017cba64eaba85c39016e257ca`; Q8_0/Q2_K is 102,408,545,440 bytes with identity `01b2bed4f070d0a3fdb02e546764b3a49cb69886eebe17b4877d20294725682c`; the mixed IQ2_XXS/Q2_K candidate is 94,154,155,392 bytes with identity `bdc6b442cf03f86f9d9a5cfe2a2f46a6ed7e3268b3ee6f3124725c999ca75a53`. Each retains exact tokenizer material and all 1,360 tensors. The baseline and candidate artifacts are consumed by variant-adaptive materialization; neither is supported-model or release-profile evidence. |
| Integrity | Canonical layout admission, full-file SHA-256 identity, exact payload-range verification, native writer-reader equivalence, pinned official ggml reader acceptance, deterministic serialization, and complete-artifact support admission are closed for the admitted DeepSeek artifacts. The physical catalog selects only immutable extent and exact digest evidence; operator admission rehashes every byte before binding. Tensor proofs and external or structurally incomplete GGUF files remain outside the complete-artifact path. |
| Materialization | The admitted Q8_0/Q2_K baseline and mixed IQ2_XXS/Q2_K candidate use the same family-neutral materialization owner over all 1,360 tensors. Candidate materialization identity `04efd6362926cc009d9096b16f583993425dca91cd644ee369a5446c823ef263` derives 94,142,453,320 payload bytes and mixed routed-expert subviews from actual directory qtypes without preset-name branches. Cold admission remains bounded and file-backed; the runtime then copies every encoded tensor into one checked model-lifetime host arena and releases consumed artifact cache pages. |
| Runtime descriptor | The common runtime descriptor and DeepSeek specialization project the admitted artifact, materialization plan, canonical qtype facts, all 1,360 tensor bindings, topology, MoE/expert geometry, output/vocabulary geometry, tokenizer availability, and runtime numeric authority into one immutable graph-input descriptor. Logical-model identity `ec22b4bf78811265d1881071919593991f33ab883303f3df16d64c0689a63950` and runtime-numeric identity `33182fd6b75e9263861d5a873550e6d0c5d5010267fb315ec687c693c9572dcd` are shared semantic facts; the candidate derives runtime-descriptor identity `e1b25e80d6a0ae97cbde61b1aeec261a170bd2d8874f8bbc85f496db46d973d0` from its distinct physical bindings. |
| Runtime | Content-addressed schema-V6 runtime bindings open each admitted physical artifact into one family-neutral process-resident model. Before `READY`, `yvexd` copies all 1,360 encoded tensor payloads into one anonymous host arena and requires the complete arena to be locked in physical RAM; the mixed candidate proves `resident_host_bytes=94,142,453,320` and `VmLck=91,935,992 KiB`. The sole long-lived `yvexd` process owns that model for its complete lifetime, one bounded model worker, independent server sessions, the private Unix listener, and the loopback OpenAI listener. Protocol v3 carries provider messages, tools, usage, finish classes, correlations, and integrated-listener status while the OpenAI adapter remains source-separated and reaches the existing session/worker authority through the local protocol. The partial performance delivery reduced the accepted cold-start candidate to 236.97 seconds, corrected residency accounting, and measured a 32-token warm request at 0.432 decode tokens/s with bounded stage, movement, launch, and synchronization facts; these are optimization evidence, not benchmark results or full performance admission. Physical-variant identities remain descriptor facts rather than host, adapter, or client policy. |
| CUDA | Production C contains no fallback PTX. Context, Driver API memory operations, generated-bundle admission, resolved functions, exact variants, runtime-session CUDA Graph lifecycles, and persistent-state residency are distinct typed facts. A no-`nvcc` build refuses every kernel before dispatch. The GB10 eager path executes the backbone and direct encoded BF16 output-head projection over every vocabulary row without CPU numerical fallback. |
| Transformer | Numeric token IDs execute through selected encoded embedding rows, exact four-stream mHC initialization, 43 canonical attention/MoE blocks, deferred FFN mHC posts, final mHC collapse, and final RMSNorm on CPU and GB10 CUDA. One global state transaction commits all attention publications and advances position only after finite normalized hidden output is ready. The generation owner consumes this numeric boundary unchanged for prompt prefill and sampled-token decode. |
| Operator | `yvex` is the sole public command executable. Its runtime-client lane owns `chat`, `run`, `runtime`, `session`, and hosted-model administration and can reach the engine only through the local protocol; its offline lane owns the absorbed nested artifact, graph, quantization, tokenizer, source, tensor, and evidence operations and may link `libyvex` for finite engineering work. Route-level dependency and dispatch guards prevent daemon-backed commands from opening artifacts, initializing CUDA, or becoming a second hosted model authority. The frozen `ec7dcc` audit reconciles 70 route-level commands, 426 command/flag pairs, 99 operations, 121 Make targets, 52 scripts, 100 environment/build inputs, 26 orphan catalogs, and two external child-CLI projections with zero unmatched categories. The canonical command-architecture successor owns registry, parsing, protocol facts, discovery, and semantic repair; the separately ordered REPL successor consumes that authority and owns only the mature terminal experience. |
| Text path | Exact artifact-bound text/message encoding composes with real prefill, full logits, sampling, unchanged sampled-token decode feedback, typed terminal/context stops, and incremental detokenization. `yvex run` streams one daemon-backed turn; `yvex chat` provides retained multi-turn sessions whose second turn reuses exact prior KV and prefills only the new prompt suffix. Model quality remains unclaimed. |
| Evaluation | Model behavior and model quality evaluation require the complete tokenizer-to-text and application-provider path. Evaluation is blocked behind the active public project-control refoundation, canonical command architecture, mature REPL console, and measured GB10 optimization continuation; no behavior or quality result exists yet, and attention numerical conformance is not model evaluation. |
| Benchmark | The runtime owns schema-v5 `attention_component` cold/warm, eager/piecewise/full measurements, identity-bound baseline comparison, JSON/CSV facts, and deterministic external SVG charts. Correctness, structural-runtime, and performance status remain independent. Release-path generation and full-model benchmark evidence remain not measured. |
| Agent | No agent runtime, tool execution loop, agent evaluation owner, milestone, or operator command exists. Development agents, JSON output, external harnesses, and `yvexd` do not establish agent capability. |
| Release | Release qualification remains blocked and is not promoted by component qualification or benchmark evidence. |

The admitted runtime capability lattice is explicit. Compatibility umbrella
flags are derived from these facts and are not independent authorities:

```text
attention_semantics_ready=1
attention_core_ready=1
attention_envelope_ready=1
cpu_prefill_eager_ready=1
cpu_decode_eager_ready=1
cuda_prefill_eager_ready=1
cuda_decode_eager_ready=1
cuda_prefill_piecewise_graph_ready=1
cuda_decode_piecewise_graph_ready=1
cuda_prefill_full_graph_ready=1
cuda_decode_full_graph_ready=1
attention_weight_residency_ready=1
attention_workspace_ready=1
attention_state_delta_ready=1
attention_operator_ready=1
attention_trace_ready=1
attention_profile_ready=1
attention_benchmark_ready=1
model_behavior_evaluation_ready=0
model_quality_evaluation_ready=0
agent_runtime_ready=0
agent_tool_execution_ready=0
agent_evaluation_ready=0
release_qualification_ready=0
mixed_attention_ready=0
speculative_attention_ready=0
persistent_kv_ready=1
activation_prefill_ready=1
prefill_persistent_state_ready=1
moe_plan_ready=1
moe_router_ready=1
moe_routed_expert_ready=1
moe_shared_expert_ready=1
moe_block_ready=1
embedding_ready=1
transformer_plan_ready=1
transformer_block_ready=1
transformer_stack_ready=1
transformer_final_head_ready=1
transformer_final_norm_ready=1
transformer_hidden_state_ready=1
moe_prefill_composed=1
moe_decode_composed=1
full_model_prefill_ready=1
transformer_ready=1
single_token_transformer_component_ready=1
decode_step_ready=1
decode_repeat_ready=1
decode_hidden_state_ready=1
decode_partial_progress_ready=1
model_decode_ready=1
output_head_binding_ready=1
output_head_residency_ready=1
output_head_projection_ready=1
logits_cpu_ready=1
logits_cuda_ready=1
logits_prefill_ready=1
logits_decode_ready=1
logits_full_vocabulary_ready=1
logits_hidden_contract_ready=1
logits_partial_progress_ready=1
logits_ready=1
sampling_source_contract_ready=1
sampling_policy_ready=1
sampling_greedy_ready=1
sampling_temperature_ready=1
sampling_top_k_ready=1
sampling_top_p_ready=1
sampling_min_p_ready=1
sampling_typical_ready=1
sampling_stochastic_ready=1
sampling_seed_reproducibility_ready=1
sampling_real_logits_ready=1
sampling_partial_progress_ready=1
sampling_ready=1
token_append_ready=1
tokenizer_plan_ready=1
tokenizer_artifact_bound_ready=1
tokenizer_vocab_ready=1
tokenizer_merges_ready=1
tokenizer_added_tokens_ready=1
tokenizer_special_tokens_ready=1
tokenizer_prompt_template_ready=1
tokenizer_encode_ready=1
tokenizer_decode_ready=1
tokenizer_incremental_decode_ready=1
tokenizer_token_append_boundary_ready=1
tokenizer_runtime_ready=1
eos_policy_ready=1
stop_policy_ready=1
detokenization_ready=1
generation_plan_ready=1
generation_prompt_ready=1
generation_prefill_ready=1
generation_first_token_ready=1
sampled_token_feedback_ready=1
generation_decode_loop_ready=1
generation_logits_loop_ready=1
generation_sampling_loop_ready=1
generation_token_append_ready=1
generation_eos_stop_ready=1
generation_context_stop_ready=1
generation_incremental_text_ready=1
generation_partial_progress_ready=1
generation_cpu_ready=1
generation_cuda_model_path_ready=1
generation_loop_ready=1
generation_ready=1
quant_policy_v2_ready=1
quant_policy_cli_overlay_ready=0
quant_preset_ready=1
current_q8_q2_profile_ready=1
full_model_policy_resolution_ready=1
physical_variant_plan_ready=1
exact_size_prediction_ready=1
imatrix_ingestion_ready=1
imatrix_coverage_ready=1
iq2_xxs_encoder_ready=1
iq2_xxs_reference_decoder_ready=1
iq2_xxs_cpu_compute_ready=1
iq2_xxs_cuda_compute_ready=1
mixed_routed_expert_qtype_ready=1
policy_driven_full_model_quant_ready=1
physical_variant_artifact_ready=1
variant_materialization_ready=1
variant_runtime_binding_ready=1
variant_generation_smoke_ready=1
automatic_precision_optimizer_ready=0
automatic_target_size_solver_ready=0
pareto_variant_selection_ready=0
evaluation_feedback_optimizer_ready=0
product_binary_topology_ready=1
engine_daemon_split_ready=1
unified_public_yvex_surface_ready=1
role_binary_topology_ready=1
yvex_product_binary_ready=1
yvexd_runtime_binary_ready=1
engineering_commands_in_yvex_ready=1
separate_developer_binary_ready=0
runtime_client_lane_separation_ready=1
offline_command_lane_ready=1
operator_surface_audit_ready=1
canonical_operation_registry_ready=0
canonical_command_taxonomy_ready=0
local_runtime_host_ready=1
one_runtime_process_ready=1
one_runtime_authority_ready=1
runtime_model_process_persistence_ready=1
runtime_model_reuse_across_requests_ready=1
full_model_host_residency_ready=1
full_model_host_ram_lock_ready=1
runtime_worker_ready=1
bounded_request_queue_ready=1
local_protocol_ready=1
protocol_versioning_ready=1
session_registry_ready=1
session_attach_detach_ready=1
session_reset_ready=1
multi_turn_session_ready=1
multi_turn_kv_reuse_ready=1
exact_prompt_suffix_ready=1
turn_boundary_ready=1
partial_turn_ready=1
streamed_generation_ready=1
typed_runtime_event_ready=1
raw_runtime_terminal_ready=1
engine_watch_terminal_ready=1
runtime_metrics_ready=1
repl_terminal_ready=1
one_shot_client_ready=1
cli_layout_v2_ready=1
old_cli_removed=1
packaging_topology_ready=1
cli_generate_ready=1
repl_ready=1
interactive_chat_ready=1
local_server_generation_ready=1
server_generation_ready=0
public_server_generation_ready=0
remote_server_ready=0
authentication_ready=0
tls_ready=0
provider_compatibility_boundary_ready=1
provider_neutral_request_ready=1
provider_neutral_tool_contract_ready=1
local_protocol_v3_ready=1
openai_compatibility_ready=1
openai_profile_v1_ready=1
openai_in_daemon_ready=1
separate_openai_gateway_ready=0
integrated_openai_health_ready=1
integrated_openai_streaming_ready=1
integrated_openai_cancellation_ready=1
openai_models_ready=1
openai_chat_completions_ready=1
openai_chat_streaming_ready=1
openai_responses_ready=1
openai_responses_streaming_ready=1
openai_usage_mapping_ready=1
openai_error_mapping_ready=1
openai_stop_strings_ready=1
openai_json_object_ready=1
openai_function_tools_ready=1
openai_tool_result_continuation_ready=1
openai_agent_function_loop_ready=1
openai_python_sdk_ready=1
openai_javascript_sdk_ready=1
bet_tennis_openai_provider_ready=1
real_bet_tennis_deepseek_ready=0
full_model_profile_ready=0
startup_profile_ready=0
prefill_profile_ready=0
decode_profile_ready=0
movement_profile_ready=0
synchronization_profile_ready=0
moe_profile_ready=0
optimized_startup_ready=0
expert_placement_ready=0
expert_cache_ready=0
device_router_ready=0
grouped_moe_prefill_ready=0
multi_turn_suffix_chunk_ready=0
optimized_decode_ready=0
normal_serving_reduced_evidence_io_ready=0
audit_execution_ready=0
performance_regression_gate_ready=0
decode_cuda_graph_ready=0
full_decode_graph_ready=0
cuda_sampling_ready=0
complete_device_residency_ready=0
openai_full_api_ready=0
openai_multimodal_ready=0
openai_embeddings_ready=0
openai_hosted_tools_ready=0
openai_mcp_tools_ready=0
openai_realtime_ready=0
openai_batch_ready=0
anthropic_compatibility_ready=0
multi_model_server_ready=0
continuous_batching_ready=0
distributed_serving_ready=0
persistent_session_restart_ready=0
```

## 5. Active Work And Critical Path

```text
V010.DOCS.REFOUNDATION.0: complete (documentation/claim only)
V010.PROJECT.RECOVERY.0: partial
V010.PROJECT.RECOVERY.1: complete
V010.DOCS.ARCHITECTURE.0: complete (documentation architecture/ownership only)
V010.REBASE.DEEPSEEK.0: complete (exact source identity/config/index/header verification)
V010.GGUF.QTYPE.ABI.1: complete (canonical common GGUF qtype storage ABI)
V010.GGUF.ARTIFACT.ABI.1: complete (scalable common native GGUF structural reader ABI)
V010.GGUF.LAYOUT.INTEGRITY.1: complete (canonical common global GGUF layout admission)
V010.CUDA.FAILCLOSED.0: complete (bounded fail-closed CUDA primitive proof)
V010.MODEL.ARCH.IR.0: complete (typed DeepSeek-V4-Flash architecture specification)
V010.TENSOR.COVERAGE.DEEPSEEK.0: complete (exact source tensor coverage)
V010.MAP.GGUF.DEEPSEEK.0: complete (canonical concrete GGUF lowering plan)
V010.SOURCE.PAYLOAD.STREAM.0: complete (trusted bounded source payload handoff)
V010.PROJECT.COMPILATION.0: complete (compilation ownership and planning contract only)
V010.DOCS.README.COMPILATION.0: complete (public compilation/execution identity only)
V010.MODEL.TRANSFORM.IR.0: complete (artifact-neutral immutable plan, GGUF projection, and quantizer binding)
V010.QUANT.2: complete (selected physical profile, numeric codecs, bounded execution, and CPU/CUDA qtype truth)
V010.GGUF.WRITER.1: complete (immutable writer plan, transactional file sink, exact physical identity, and atomic publication)
V010.ARTIFACT.EMIT.DEEPSEEK.0: complete (complete source-faithful and selected DeepSeek GGUF emission)
V010.GGUF.ROUNDTRIP.1: complete (native full-byte, pinned official-reader, and deterministic serialization equivalence)
V010.ARTIFACT.SUPPORT.CUTOVER.0: complete (canonical complete-artifact admission separated from tensor proofs)
V010.ARTIFACT.MATERIALIZE.0: complete (family-neutral admitted-artifact materialization plan/session)
V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0: complete (selected DeepSeek full-payload bounded access proof)
V010.RUNTIME.DESCRIPTOR.GGUF.0: complete (common runtime descriptor from admitted materialization)
V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0: complete (DeepSeek descriptor binding all 1,360 tensors)
V010.REPO.SEMANTIC.COMPRESSION.0: complete (repository-wide semantic ownership, compact paths, dependency boundaries, and permanent gates)
V010.REPO.C.CANONICALIZATION.0: complete (canonical C interfaces, ownership, contracts, archive identity, and complexity gates)
V010.GRAPH.DEEPSEEK.ATTENTION.0: complete (independent reference, complete SWA/CSA/HCA CPU execution, and direct GB10 CUDA parity)
V010.CLI.GRAPH.0: complete (production attention is reachable through the main YVEX binary with honest probe semantics)
V010.RUNTIME.1: complete (common sealed runtime model/session, runtime binding, resident attention execution, phase/mode dispatch, and bounded benchmark/profile and external-chart capability)
V010.RUNTIME.DEEPSEEK.KV.0: complete (session-owned family-correct CPU/CUDA state, atomic append/read, clear/reuse, and production causality)
V010.RUNTIME.DEEPSEEK.PREFILL.0: complete (identity-bound activation chunks execute every attention layer and atomically populate persistent state)
V010.RUNTIME.DEEPSEEK.MOE.0: complete (hash/learned routing, selected routed/shared experts, and token-local combination)
V010.GRAPH.DEEPSEEK.TRANSFORMER.0: complete (numeric tokens execute embedding, 43 attention/MoE blocks, final mHC/norm, and one atomic KV publication)
V010.RUNTIME.DEEPSEEK.DECODE.0: complete (teacher-forced one-token steps reuse the warm backbone, consume prior KV, and preserve partial progress)
V010.RUNTIME.DEEPSEEK.LOGITS.0: complete (project final-prefill and decode normalized hidden rows through the exact complete output head)
V010.RUNTIME.SAMPLING.0: complete (post-closure repair revalidated canonical real-logits selection, lifecycle, evidence, and sanitizer coverage)
V010.RUNTIME.DEEPSEEK.TOKENIZER.0: complete (exact artifact-bound encoding, prompt, special/EOS, append-directory, and incremental detokenization semantics)
V010.RUNTIME.DEEPSEEK.GENERATION.0: complete (exact prompt prefill, sampled-token decode feedback, typed stop, partial progress, and incremental text)
V010.COMPILATION.PHYSICAL.VARIANT.1: complete (policy-v2 full-model resolution, IQ2_XXS/imatrix execution, complete candidate artifact, and variant-adaptive runtime binding)
V010.ARTIFACT.MATERIALIZE.1: complete (artifact-qtype-driven materialization and expert subviews without preset-name policy)
V010.CLI.DEEPSEEK.GENERATE.0: superseded (fresh-only CLI plan replaced before implementation by the runtime/client refoundation)
V010.RUNTIME.CLIENT.REFOUNDATION.0: complete (long-lived local runtime host, exact sessions, protocol, telemetry, client, REPL, and incompatible CLI cutover)
V010.DOCS.README.PRODUCT.0: complete (public product README, three-terminal quickstart, and reviewed architecture visuals)
V010.SERVE.OPENAI.COMPAT.0: complete (provider-neutral application contract, protocol-v3 adapter path, bounded OpenAI profile, and external consumer proof)
V010.SERVE.OPENAI.BETTENNIS.REAL.1: reopened (the unchanged provider has not yet traversed the adapter and real resident DeepSeek in one proof)
V010.RUNTIME.DEEPSEEK.PERFORMANCE.0: partial (startup and bounded profiling close; warm decode remains below admission)
V010.PRODUCT.SURFACE.REALIGNMENT.0: complete (one public yvex surface, one yvexd runtime process, absorbed engineering commands, and integrated OpenAI listener)
V010.OPERATOR.SURFACE.AUDIT.0: complete (frozen identity-bound command, flag, operation, automation, configuration, taxonomy, registry, and project-control input with zero unmatched categories)
V010.PROJECT.CONTROL.PUBLIC.0: active (refound public project control from the accepted post-cutover topology and frozen audit)
V010.OPERATOR.RUNTIME.CONSOLE.0: superseded (the earlier isolated console plan is replaced by the ordered command-architecture and mature-REPL contracts)
V010.OPERATOR.COMMAND.CONSOLE.0: superseded (the combined command and console plan is split before implementation)
V010.OPERATOR.COMMAND.ARCHITECTURE.0: blocked (canonical operations, commands, flags, protocol facts, and discovery wait for public project control)
V010.OPERATOR.REPL.CONSOLE.0: blocked (the mature daemon-backed terminal consumes the canonical command architecture)
V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0: blocked (measured warm-runtime continuation waits for the mature REPL boundary)
V010.EVAL.DEEPSEEK.0: blocked (PASS 5 waits for the measured GB10 optimization continuation)
Active Next: V010.PROJECT.CONTROL.PUBLIC.0
```

Repository compression and C canonicalization preserved every admitted source,
payload, GGUF mapping, quantization profile, artifact, and materialization
identity. The canonicalization baseline contained 306 owned C/CUDA/header
files, 164 translation units, and 142 headers. Its accepted pre-runtime closure
contained 198 production files, 159 translation units, 39 headers, 174 test
files, 84 C/CUDA/header test files, and 196 semantic owners.

Runtime, persistent-state, activation-prefill, MoE, transformer composition, and decode leave 182
production C/CUDA/header files: 140 C translation units, one CUDA translation unit, and 41
headers. The test tree contains 161 files, including 85 C/CUDA/header test
files. The headers comprise 13 installed public, 22 explicit internal, and six
source-local contracts. Mechanical same-stem pairs, production umbrella
imports, ambiguous bare internal includes, basename violations, and
project-prefix violations remain zero.

Physical production lines changed from 131,559 to 137,562, a net increase of
6,003 lines; code lines changed from 115,138 to 114,774, and executable lines
from 77,367 to 69,862. All 3,250 production functions have adjacent contracts;
2,607 non-trivial functions carry the complete semantic contract. Functions
above 200 lines are zero, the largest function is 200 lines, the largest
translation unit is 2,000 lines, and the hard line-width violation count is
zero.

The static archive changed from 116 members with only 95 unique identities to
103 source-relative members with 103 unique identities. Library-global symbols
fell from 969 to 912: 325 public and 587 non-public. Semantic owners fell from
230 at the canonicalization baseline and 196 at the pre-runtime closure to 170;
164 owners have one file and six have two. DeepSeek uses exactly its admitted
family budget of three sources and one header. The current tree has zero
one-consumer private headers, duplicate definitions, include cycles, forbidden
dependency edges, unregistered files, family-budget violations, or
layout/ownership exceptions. Manifest, Makefile source, source-relative object,
and archive-member parity are exact. The canonical ownership, layout,
architecture, and code-naturalness gates enforce these facts without promoting
full-model prefill, runtime generation, or any higher inference capability.

Strict source proof consumes the pinned upstream index with Git OID
`84692cbe7af556a01e2e5353341100079c387aee`, validates the exact model,
tokenizer, and generation config structure, and reads 69,187 unique tensor
records from 46/46 safetensors headers in one canonical pass. The verified
local footprint is 159,629,046,930 bytes after admission of the index and its
provider metadata. The verifier now admits the 46 authoritative Git LFS
SHA-256 facts, streams all 159,617,149,040 shard-file bytes through the common
payload reader, and atomically publishes manifest v3 only after every digest,
file identity, and aggregate identity succeeds. A local-only seal remains a
distinct `local_payload_snapshot_sealed` trust class and is never promoted to
upstream verification.

The global layout proof matches pinned ggml
`af97976c7810cdabb1863172f31c432dab767de7`: alignment is power-of-two,
directory and physical order agree, every relative offset equals the previous
padded end, required padding is zero, and the exact padded file span is
snapshot-safe. The 69,187-tensor scale proof is linear and reads zero tensor
payload bytes. Layout admission remains a container property, not complete
model-artifact support.

CUDA capability admission now consumes only the generated
`kernels.cu` bundle. A baseline build without `nvcc` keeps context and
proven Driver API memory/transfer facts while refusing all kernel variants
before dispatch. On NVIDIA GB10, F32/F16 embedding, F32 RMSNorm with F32/F16
weights, F32 RoPE/matmul, bounded dense/routed F32 MLP, and bounded
causal/non-causal F32 attention resolve atomically and pass independent
reference comparisons. Module/function rollback, launch/sync failure, output
state, temporary allocation, and checked cleanup fail closed. This completes
CUDA capability truthfulness, not the v0.1.0 DeepSeek backend gate.

The canonical architecture IR consumes one successful exact-source result and
does not reopen config, tokenizer, index, or shard headers. It owns 43 explicit
main-layer descriptors plus one MTP descriptor: 2 SWA, 21 CSA, and 20 HCA main
layers; 3 token-ID hash routers and 40 hidden-state learned routers; four mHC
residual streams over a 16,384-wide expanded state; an untied 129,280-entry
output head; and the exact tokenizer and BF16/FP4/FP8 source constraints. The
live proof scanned 46 headers once, observed 69,187 tensor records, and read
zero tensor payload bytes. Paper `arXiv:2606.19348v1`, SGLang
`96a04cb13f9c3ed86028e090784a9eb059cf5318`, and vLLM
`8df14cfc8c8a09b4e57f082e59593a3abce4ffb3` are the pinned interpretation
baselines. This closes architecture semantics only; tensor-role coverage is
owned separately.

The coverage owner derives 69,187 typed requirements from that IR and matches
them one-to-one against the retained upstream-indexed snapshot without
reopening any shard. The closed inventory contains 6 global, 572 attention,
164 compressor, 147 indexer, 88 normalization, 264 mHC, 88 router, 67,584
routed-expert, 264 shared-expert, and 10 auxiliary tensors. Indexed lookup and
deterministic iteration make the pass linear at target scale; the live proof
performs one header scan and reads zero payload bytes. Coverage does not choose
GGUF names, transformations, emitted layouts, or payload policy.

The canonical DeepSeek GGUF map now consumes the sealed Transformation IR
without reopening source metadata or independently constructing transformation
truth. It projects all 69,187 source values into 1,360 logical outputs: 1,328
standard trunk descriptors and 32 namespaced MTP extension descriptors. The
adapter adds GGUF names, forced source-format qtypes, metadata, emitted layout,
and deterministic source/emitted/role indexes. Descriptor-by-descriptor tests
preserve mapping identity `1aecbbe25b04de0d`; payload conversion, writer bytes,
and runtime behavior remain outside the mapping owner.

The payload handoff consumes the same retained snapshot and map. It builds
immutable O(1)-expected shard and tensor indexes, checked absolute ranges, and
physical-order page/chunk plans without reparsing headers. A verified session
uses secure relative paths, `openat`/`O_NOFOLLOW`, regular-file and replacement
checks, bounded pinned LRU handles, one bounded reusable buffer, exact
`pread`-style loops, cancellation, and transactional begin/chunk/commit or
abort delivery. The live trust-and-deliver pass resolved all 69,187 mapping
contributions and 1,360 descriptors, delivered 159,609,485,896 logical bytes,
and observed zero short reads, digest mismatches, or identity drift. This is a
source payload capability for the future transformation executor; it performs
no conversion, quantization, GGUF emission, artifact admission,
materialization, or runtime execution.

The compilation owner now seals an artifact-neutral schema-v1 DAG before any
physical lowering. The complete DeepSeek plan contains 69,187 canonical source
values, 1,360 terminal values and nodes, 69,187 edges, maximum fan-in 512, and
maximum depth one. Its closed operations comprise 850 identity transfers, 375
scale-paired decodes, three checked integer casts, and 132 expert-axis
aggregations. Canonical SHA-256 encoding excludes allocation order, pointers,
local paths, GGUF names, qtypes, and artifact offsets. The quantizer binding
resolves all 69,187 source values to matching manifest-admitted payload ranges
while retaining source snapshot, required payload, and Transformation IR
identities as distinct facts. The live plan performs one retained header pass
and zero payload reads. `V010.QUANT.2` now consumes this immutable truth rather
than rediscovering source roles, companions, aggregation axes, or logical
shapes.

The quantization owner consumes that sealed DAG and exact payload binding
through profile `deepseek-v4-flash-q8_0-q2_k-v1`. The current plan-only
derivation has identity
`431696b26898bbc98ef0a8de2e8c8992f2771d631aaa53a446c089b7d4fca1fb`;
the immutable artifact already admitted by the repository retains its executed
physical-profile identity
`04be09e124fd997ae3b785d0d3018f9d571cb6b96df5488d0ab21de3345bce25`.
All 1,360 terminal decisions biject the GGUF lowering descriptors: 417 F32,
433 BF16, three I32, 375 Q8_0, and 132 Q2_K tensors. The complete read-only
execution consumed all 69,187 trusted ranges and 159,609,485,896 payload bytes,
committed 102,396,843,592 encoded bytes to a digest/discard sink, and produced
execution identity
`b81f3c5d670737bf20c938e635a1bffdbb0d60f885f994225a02225bb7ba51db`.
Q8_0 maximum absolute error/RMSE were 0.0146484375/0.00016908988226012133;
Q2_K values were 0.507720947265625/0.0076422877123441847. Derived block bounds,
role-grouped metrics, independent reference decoding, and direct CPU/CUDA
encoded-row compute passed with zero numeric-bound violations, short reads,
payload drift, sink failures, aborted terminals, or incomplete terminals. The
profile explicitly requires no calibration. Policy schema v2 now also ingests
the pinned 450,892,654-byte DeepSeek imatrix with identity
`0de71592bdd450acec67d81f71942736af96bdff268840fac5ea05f7aba3c455`;
coverage, source, dataset, producer, geometry, and numeric use are validated
before source payload reads. `V010.QUANT.3` is therefore complete for imported
calibration evidence; calibration collection inside YVEX remains deferred.

The sealed mixed IQ2_XXS/Q2_K policy identity
`e482aac9f97b1fad3011288ff24a836b3d3bae81693161a6be91698856c0d93f`
resolves all 1,360 terminals without ambiguity into physical-variant identity
`f0ba4b56e57842e4ca82fa4fc60d0a79c32061b5715b541d896a52ea76289ba4`.
It selects 417 F32 tensors totaling 144,672,072 bytes, 433 BF16 tensors
totaling 2,830,518,528 bytes, three I32 tensors totaling 9,308,160 bytes, 378
Q8_0 tensors totaling 13,244,563,456 bytes, 43 Q2_K tensors totaling
30,299,652,096 bytes, and 86 IQ2_XXS tensors totaling 47,613,739,008 bytes.
The exact payload is 94,142,453,320 bytes; the predicted and emitted GGUF size
is 94,154,155,392 bytes. Main, MTP, and global scope contributions are
84,997,375,048, 7,026,684,396, and 2,118,393,876 bytes respectively. Planning
reads zero source payload bytes and refuses missing defaults, equal-priority
conflicts, exact-role overrides, unsupported codecs/backends, and incomplete
imatrix coverage.

The policy-v2 projection of `deepseek-v4-flash-q8_0-q2_k-v1` preserves all
1,360 physical decisions and all 102,396,843,592 tensor payload bytes. A new
policy-v2 emission is 160 bytes larger than the already admitted baseline file
because it carries the policy and explicit `none` imatrix provenance entries;
the existing 102,408,545,440-byte artifact is unchanged and remains admitted.

The artifact owner now consumes the lowering and quantization owners through
immutable GGUF v3 writer plans. The published source-faithful plan identity
`3bcd8e5bd58fc05684121a05ee87c13c14dcf09f379ca37f06e0361591d06eb0`
produced 177,668,871,752 tensor payload bytes in a 177,680,573,600-byte file;
the published selected plan identity
`4b47814e06c43b3426efcaab72b836596c42358a7c59ea5619ddd70c0eefe9fd`
produced 102,396,843,592 tensor payload bytes in a 102,408,545,440-byte file.
Plan-only re-derivation over the current runtime-numeric-aware transformation
identity yields `9dc39cf987270562859238ba423a3f17b1d3c11c1f2e049516866a853c600a84`
and `db0344991f10237119b32152a8c4ccb6bffc31f968464820a1a74e13590c7b30`,
respectively, with zero payload reads and no artifact publication.
The source-faithful and Q8_0/Q2_K artifacts contain 47 lowering metadata entries plus 21 exact
artifact/tokenizer/provenance entries, 129,280 tokenizer tokens, 127,741
merges, and all 1,360 tensors. Native YVEX full-byte roundtrip, the pinned
official ggml reader at
`af97976c7810cdabb1863172f31c432dab767de7`, and a second complete selected
serialization accepted identical structure and bytes. Artifact identities
`f16e800c0d7383ee76cb2e2fa8bdd674bab29c017cba64eaba85c39016e257ca`
and `01b2bed4f070d0a3fdb02e546764b3a49cb69886eebe17b4877d20294725682c`
bind the exact physical files independently from their quantization execution
identities. The complete-artifact gate admits those immutable snapshots for
materialization; it does not claim residency, runtime execution, transformer
execution, or generation.

The mixed IQ2_XXS/Q2_K writer plan identity
`8aa949e9a3e3b77211176a3a6dfed99ea0e63f210c7d80a3aedfb9b57527360d`
consumes the same lowering and tokenizer scope plus policy/imatrix provenance.
Its complete 70-metadata-entry artifact has identity
`bdc6b442cf03f86f9d9a5cfe2a2f46a6ed7e3268b3ee6f3124725c999ca75a53`,
payload-byte identity
`def97e67996ce6eb5902c33e8446c03561f2e8ec6d582439311bad5b5df80037`,
and quant execution identity
`b1446e5caf1d4768488c7daaf42784c6050e56d1422e759a34971520905206c4`.
Native full-byte validation hashed all 94,154,155,392 bytes and the pinned
official reader accepted all 1,360 tensors and 94,142,453,320 raw tensor bytes.
A second complete emission reproduced the same policy, physical-variant,
writer-plan, quant-execution, payload-byte, artifact, materialization, and
schema-v6 runtime-binding identities, and independently repeated both native
full-byte and pinned-official roundtrip acceptance.
The candidate is admitted materialization/runtime evidence, not the selected
release profile; evaluation and benchmark own that future choice.

The candidate runtime binding identity is
`686f670720f92a2baf7ff779ae57c8ed110668c626b05d6a793d058f4e709497` and remains
schema V6. Its materialization identity is
`04efd6362926cc009d9096b16f583993425dca91cd644ee369a5446c823ef263`.
The bounded real-generation smoke executed one prompt token and one sampled
token on both CPU and GB10 CUDA. Both selected and decoded token `42498`,
advanced the same session position from one to two, ended at persistent
generation 46, and produced text digest
`1ec5808c1256f22626a2f91181c29a058615a2ab9b28bce4bbd38216c4da0047`
and KV digest
`b17f7252c3de3a29b3db764f8d3c9e2e61c2d37176998fa020f5f97a4265cf20`.
The CPU and CUDA execution identities differ by backend evidence, as required;
the token, decode input, committed state, and visible bytes agree. This is a
correctness smoke, not model-quality or performance evidence.

The admitted attention executor consumes the immutable DeepSeek runtime
descriptor and materialized encoded weights through attention plan identity
`4fbcc44b8b12a0174673b2b4cc26f7674280ff6580e738316b6a4c394c9cba5b`.
An independent test-only full-equation reference and the production CPU and
generated-bundle CUDA paths execute all 43 main layers and 634 bindings: two
SWA, 21 CSA, and 20 HCA descriptors. Rolling compression, selected compressed
history, deterministic top-k, causal/class masks, stable softmax, reduction,
and output projection are numerical dependencies rather than report facts.
SWA applies base RoPE without YaRN. CSA and HCA apply the compressed-class YaRN
policy; HCA emits complete non-overlapping groups at the exact ratio 128,
retains an incomplete tail as raw local history, and lets raw and compressed
representations participate without cross-representation deduplication.

The complete live comparison covered 6,772,096 values and observed zero
maximum absolute error, zero maximum relative error, and zero finite-only RMSE
for reference/CPU, reference/CUDA, and CPU/CUDA comparisons on GB10 compute
capability 12.1. CPU/CUDA comparison admission uses versioned comparison schema
v2: every finite pair must satisfy
`abs(a - b) <= 5e-4 + 5e-4 * max(abs(a), abs(b))`, non-finite values refuse,
and RMSE is accumulated only over finite pairs. Raw object-byte comparison
also observed bitwise equality for this evidence run, but bitwise equality is
not required and no causal explanation is inferred from that observation.
The completed `V010.GRAPH.DEEPSEEK.ATTENTION.0` checkpoint recorded historical
attention execution evidence identity
`f4afb4a97ef13c7f7d5805509dbffdbbf15bdc10a933e7b0a0871733238833e5`;
independent-oracle trace and output identities are
`96a0ff53a4bfb82c3d1db1f3dce2694e1d1500eb5c305dbd5470978bf5e16435`
and `c6952cf145632719458cbca26ef26c4f03cc5543f22f21072e223c3a14338128`.
They remain provenance for that closed milestone and are not current runtime
model/session benchmark identities.
The earlier `fd635d6a820cd5bda00e88bd1fc5eff649ac990f9667120c49b4c3bf811ba858`
was bounded, unversioned checkpoint evidence; it is superseded and is not
repurposed as complete execution evidence. The owner now reports
`attention_execution_supported=1`, `attention_cuda_execution_ready=1`, and
`runtime_generation_ready=1`. Persistent session state, activation-driven
attention prefill, token-local MoE, the complete transformer backbone, and
teacher-forced repeated decode, complete logits, common host sampling, exact
tokenization, and autoregressive composition are admitted by their retained
milestones; the final top-level CLI, evaluation, full-model benchmarking, and
release remain separate unsupported or blocked boundaries.
Here `prefill` names tokenizer-backed full-model prompt prefill, not the admitted
attention-local activation-chunk phase.

Correcting the false checkpoint assumption that applied YaRN to SWA, and
binding explicit BF16/F32 compute semantics in runtime-numeric schema v2,
changed semantic derivation identities without changing physical artifact
bytes. Logical model
`034b6a0fe8969f878e8a455d4829219495dd5e5614ff4288712e3b4b02b5ccc5` became
`ec22b4bf78811265d1881071919593991f33ab883303f3df16d64c0689a63950`;
Transformation IR
`be3afd02188f187228b2bc1fccbe90731ed51cddca04280e7c904fb564d82235` became
`cc774dffb6aa3a8e9f507b1dd454fbf7f5c68187138736f9a330ee9eaec07067`;
runtime numeric
`50bac88b4b7cc5ef1ccc0acbca0e74e4c0a47df0eba982f92cea8be9c8882a4f` became
`33182fd6b75e9263861d5a873550e6d0c5d5010267fb315ec687c693c9572dcd`;
runtime descriptor
`1214401ff15faecece0d83ad0a49f1aa5e17c5aede94e2ac6683ae3a42bd6118` became
`61e25532554505b1536e8ec11198b680ca812e0cdc8f2966a6f6652103a95574`;
and attention plan
`013ff42a92125471ea40f3831358c77435afc0f33c730e24f85a4e2897307351` became
`154761f606f2c5cbf2464f1446bc98374cf8071029bb898ffaab72291135db25`.
The runtime envelope admission then versioned that plan from
`154761f606f2c5cbf2464f1446bc98374cf8071029bb898ffaab72291135db25`
to `a562cf9949c01d25cebe5c0669664a11045f4426e523334c1eccdcc94650b5b4`
by binding the immediate mHC ingress, input normalization, residual egress,
and envelope output geometry. Runtime attention-plan schema v3 then produced
`45061c65a16e2bf0d773f620f0ca42fa1ddbac71a5e11fb8cf5028886e564a18`
by additionally binding the exact required envelope-binding count used by
runtime residency. Runtime-binding schema v5 then corrected the stale
descriptor value of eight experts per token to the admitted family value of
six and bound the immutable MoE plan. The runtime descriptor advanced from
`61e25532554505b1536e8ec11198b680ca812e0cdc8f2966a6f6652103a95574` to
`68b6b6baf90576c3b03499340a9cd2bbbc488d529cd5b8492be75ef4ebb8a123`,
and the attention plan advanced to
`4fbcc44b8b12a0174673b2b4cc26f7674280ff6580e738316b6a4c394c9cba5b`.
Artifact and materialization bytes remain unchanged.
Derived quant profile
`d28393a69f51e9909602f2a68eaaf380705805845dfd43579bc5f81ba1be22a4` became
`431696b26898bbc98ef0a8de2e8c8992f2771d631aaa53a446c089b7d4fca1fb`.
Source-faithful/selected writer plans
`3b5d331029a9cda537d00789e7e7fa1d8fb5b1c7729ca19004c0443734cbad74`
and `0a2dd84be2c98e72b8ce039442e514a528a55bb4d5f9319e458816bac35e906e`
became `9dc39cf987270562859238ba423a3f17b1d3c11c1f2e049516866a853c600a84`
and `db0344991f10237119b32152a8c4ccb6bffc31f968464820a1a74e13590c7b30`.
Mapping identity `1aecbbe25b04de0d`, selected artifact identity
`01b2bed4f070d0a3fdb02e546764b3a49cb69886eebe17b4877d20294725682c`,
and materialization identity
`77dba687ceedc417c447265e33deb7a6e34d0ccaec76b1b23843203b7595d0c8`
remain unchanged.

The operator cutover proves that semantic derivation drift does not require new
physical bytes through three independent facts. First, the zero-read canonical
payload recipe for all 1,360 tensors has identity
`6c6289c096b5502eba98498bf498c80d9ca9c13ab06f5dcb62075e372274e97b`;
it covers source ranges, transformation operations, physical decisions,
lowering geometry, ordering, and encoded sizes, but is not a digest of emitted
payload bytes. Second, a complete quantization pass and an independent read of
the admitted artifact produced the same aggregate payload-byte identity
`249277b42eb1aa231bddcb33b33ae3d805f3aa5991eaa99ae091f2ea9b928eb0`.
Third, operator admission rehashes the exact 102,408,545,440-byte GGUF and
revalidates its immutable file snapshot against artifact identity
`01b2bed4f070d0a3fdb02e546764b3a49cb69886eebe17b4877d20294725682c`
before graph dispatch. The zero-read comparison separately proves names,
ranks, logical and physical dimensions, qtypes, encoded sizes, directory order,
offsets, and payload ranges; it does not claim to have read or digested payload
bytes. Together these gates report `physical_payload_compatible=1`,
`artifact_rebuild_required=0`, and `materialization_rebuild_required=0`. The
writer now owns copied provenance
strings instead of retaining stack-borrowed metadata values; its corrected
deterministic current selected-plan identity is
`db0344991f10237119b32152a8c4ccb6bffc31f968464820a1a74e13590c7b30`.
That bug fix changes planning evidence, not the unchanged serialized artifact.

The main executable exposes the production boundary through
`yvex graph attention`. The `prepare` action resolves the admitted external
artifact, constructs the current runtime descriptor and attention plan, and
publishes a content-addressed runtime binding. Execution actions independently
reopen the supplied binding, seal the immutable model and reusable session, and
invoke the production API with a full-width deterministic attention probe.
Quick scope executes one SWA, CSA, and HCA representative including real top-k
and ratio-128 state; full scope executes all 43 main layers and 634 bindings.
CPU, generated-bundle CUDA, JSON, audit, and CPU/CUDA comparison modes are
operator reachable. The independent full-equation oracle remains test-only and
is excluded from both `libyvex.a` and `yvex`. Operator CPU/CUDA admission uses
the versioned combined absolute-plus-relative schema-v2 bound, refuses
non-finite pairs, and computes RMSE over finite pairs only. Raw object bytes
matched in the admitted run, but that bitwise observation is neither required
nor presented as proof of why the numerical results agree; independent
mutation and history/top-k/ratio sensitivity tests prevent a tautological
comparison.

The current content-addressed runtime-binding identity is
`23ca6e2516a767013b17bea225aab497dfed84216769a2160ebbd717ece812db`.
It binds semantic graph identity
`07d05dcb0601da72c6a434ae1258b14cfdab499671ae82ff2269fd008ea24c18`
to executable graph identity
`6ea96dcc683f3e20654297a25a282a4e0c12331884e2e3338e7634267777b8d8`.
Runtime-model, residency, workspace, execution-descriptor, launch-graph,
graph-exec, and benchmark identities remain request/session evidence and are
reported by their exact operator executions rather than promoted to model
identity.

`V010.RUNTIME.1` also performs the intentional incompatible cutover from the
former public diagnostic engine/session, flat F32 KV, fixture generation, and
generic run-report ABI. Those owners were bounded proof scaffolds tied to the
superseded runtime spine, not implementations of the retained KV, generation,
or benchmark milestones. Keeping them would create duplicate lifecycle and
capability authorities. The future milestones therefore consume the common
runtime state-provider and execution contracts directly; no compatibility
header or forwarding implementation remains.

Recovered `V010.BACKEND.0`, `.6`, `.7`, `.8`, and `.9` are promoted from
planned subtasks to completed bounded capabilities because the exact matrix,
refusal, no-bundle fallback, scratch ownership, and failure/cleanup contracts
now have consumed implementation and tests. Metal `.10` and ROCm `.11` are
reclassified as deferred future scope. `V010.BACKEND.12` closes the complete
DeepSeek backbone operation set; output-head and sampling close through their
downstream milestones while later release operations retain separate owners.

```text
V010.DOCS.README.COMPILATION.0
-> V010.MODEL.TRANSFORM.IR.0
-> V010.QUANT.2
-> V010.GGUF.WRITER.1
-> V010.ARTIFACT.EMIT.DEEPSEEK.0
-> V010.GGUF.ROUNDTRIP.1
-> V010.ARTIFACT.SUPPORT.CUTOVER.0
-> V010.ARTIFACT.MATERIALIZE.0
-> V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0
-> V010.RUNTIME.DESCRIPTOR.GGUF.0
-> V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0
-> V010.REPO.SEMANTIC.COMPRESSION.0
-> V010.REPO.C.CANONICALIZATION.0
-> V010.GRAPH.DEEPSEEK.ATTENTION.0
-> V010.CLI.GRAPH.0
-> V010.RUNTIME.1
-> V010.RUNTIME.DEEPSEEK.KV.0
-> V010.RUNTIME.DEEPSEEK.PREFILL.0
-> V010.RUNTIME.DEEPSEEK.MOE.0
-> V010.GRAPH.DEEPSEEK.TRANSFORMER.0
-> V010.RUNTIME.DEEPSEEK.DECODE.0
-> V010.RUNTIME.DEEPSEEK.LOGITS.0
-> V010.RUNTIME.SAMPLING.0
-> V010.RUNTIME.DEEPSEEK.TOKENIZER.0
-> V010.RUNTIME.DEEPSEEK.GENERATION.0
-> V010.COMPILATION.PHYSICAL.VARIANT.1
-> V010.RUNTIME.CLIENT.REFOUNDATION.0
-> V010.DOCS.README.PRODUCT.0
-> V010.SERVE.OPENAI.COMPAT.0
-> V010.RUNTIME.DEEPSEEK.PERFORMANCE.0
-> V010.PRODUCT.SURFACE.REALIGNMENT.0
-> V010.OPERATOR.SURFACE.AUDIT.0
-> V010.PROJECT.CONTROL.PUBLIC.0
-> V010.OPERATOR.COMMAND.ARCHITECTURE.0
-> V010.OPERATOR.REPL.CONSOLE.0
-> V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0
-> V010.EVAL.DEEPSEEK.0
-> V010.BENCH.DEEPSEEK.0
-> V010.RELEASE.0
```

Common capability milestones may use Qwen, Gemma, dense fixtures, and other
admitted inputs. They must close their typed common boundary and must not claim
another supported model.

## 6. Family Capability Matrix

| Family/scope | Source/profile truth | Tensor/map truth | Artifact/materialization truth | Runtime truth | Project role |
| --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash | Exact source metadata/header verification, upstream payload trust, bounded streaming handoff, and typed architecture IR complete | Exact 69,187-entry coverage, artifact-neutral Transformation IR, 1,360-descriptor GGUF lowering, and selected quantization plan complete | Source-faithful and selected complete GGUF artifacts emitted, roundtrip-verified, admitted, fully walked through bounded materialization access, and projected into a DeepSeek runtime descriptor | One long-lived local host executes exact multi-turn prompt-to-text generation through the complete CPU/GB10 CUDA model path, host sampling over 129,280 logits, sampled-token decode feedback, artifact-bound incremental detokenization, exact KV reuse, and streamed committed fragments | sole v0.1.0 release target |
| Qwen | Source target/profile, header inventory, naming-map, and role-coverage work exists | Implemented at bounded source/header/report stages | No complete supported artifact or full materialization claim | generation unsupported | active multi-family/common architecture evidence |
| Gemma | Source target/profile and header tensor-collection work exists | Dense/common mapping can be reused; exact complete family gate not claimed | No complete supported artifact or full materialization claim | generation unsupported | active dense/common architecture evidence |
| Dense/common | Common naming, collections, proof artifacts, validators, and primitive evidence exist | Partial reusable capability | Family-neutral admitted-artifact materialization, runtime binding, descriptor projection, sealed runtime model/session, and execution lifecycle are implemented; DeepSeek is the first admitted vertical adapter | no supported full-model runtime or second-family execution adapter | common engine architecture and regression surface |
| GLM | No canonical implemented target contract | planned only | unsupported | unsupported | parked pending an implementation milestone |
| Other families/backends | uncommitted | uncommitted | unsupported | unsupported | require an explicit future scope decision |

## 7. Track Registry And Dashboard

The recovery baseline contained 629 table entries across 24 tracks. Two source
rows held paired IDs; those are expanded into 631 unique recovered IDs. The
canonical architecture now contains 26 tracks. Sixty-one new milestone/future/
evidence IDs and two explicit migration markers produce 694 unique canonical IDs.

### 7.1 Global Counts

These declarations are checked against the canonical ledger. State counts are
calculated from rows rather than protected by a summary hash.

| Metric | Count |
| --- | ---: |
| Recovered IDs | 631 |
| Explicit new IDs | 65 |
| Canonical IDs | 696 |
| First-class milestones | 57 |
| State: complete | 448 |
| State: active | 1 |
| State: partial | 2 |
| State: blocked | 5 |
| State: planned | 198 |
| State: reopened | 3 |
| State: deferred | 20 |
| State: superseded | 18 |
| State: not-measured | 1 |

| Track | Owns | Current truth | Conclusive handoff |
| --- | --- | --- | --- |
| `TRACK.SCOPE` | Project control, release scope, family/release distinction, claim boundary, version policy, repository semantic ownership, C interface/build boundaries, and documentation-control transitions. | role-based product-surface realignment and the post-cutover surface audit are complete; public project-control refoundation is active | One truthful public project map, one explicit release contract, one product-first entry point, and one enforced source-owner topology. |
| `TRACK.SOURCE` | Source identity, revision, provider intake, manifests, shards, sidecars, header inventory, payload trust, and bounded payload access. | exact metadata/header verification and trusted bounded payload access are complete | Verified source facts and readable payload ranges. |
| `TRACK.MAP` | Family source names, canonical runtime roles, GGUF names, physical-lowering projections, layouts, ambiguity refusal, and complete mapping coverage. | canonical DeepSeek GGUF lowering evidence is complete | Every required source tensor maps exactly once to a runtime role and emitted layout. |
| `TRACK.COMPILATION` | Artifact-neutral transformation IR, derivation identity/DAGs, physical variant identity, constraint profiles, requirement composition, variant selection, and evaluation/benchmark feedback intake. | one sealed policy now resolves all terminal decisions and drives exact sizing, byte execution, writer emission, materialization, and runtime binding; automatic optimization and Pareto search remain deferred | One immutable physical-variant plan carries every user-selected admissible representation into the exact downstream artifact. |
| `TRACK.QUANT` | Dtype/qtype ABI, storage geometry, role policy, conversion, quantization, reference dequantization, compute truth, and refusal. | canonical numeric registry, Q8_0/Q2_K and IQ2_XXS codecs, imatrix use, bounded transformation execution, reference decoding, numeric bounds, and direct CPU/CUDA compute are complete | Every emitted role has truthful storage, calibration, and compute behavior. |
| `TRACK.ARTIFACT` | GGUF container ABI, native writer, conversion coordination, complete emission, identity, registration, and writer-reader roundtrip. | scalable native writer, complete DeepSeek emission, exact physical identities, native/pinned-official roundtrip, and deterministic serialization are complete | A complete YVEX-produced GGUF reopens identically through YVEX. |
| `TRACK.INTEGRITY` | Container, metadata, tensor directory, offsets, alignment, qtype-sized ranges, corruption, drift, and artifact admission. | canonical global layout and complete-artifact support admission are complete; tensor proofs remain explicitly separate | A complete artifact passes every pre-payload and pre-runtime integrity gate. |
| `TRACK.MODEL` | Family architecture profiles, typed architecture IR, layer topology, attention/position/KV/MoE rules, and runtime descriptor projection. | canonical DeepSeek family facts, runtime-numeric schema v2, common/DeepSeek runtime descriptor projection, and complete attention consumption are admitted | A family-correct typed model specification and executable runtime descriptor. |
| `TRACK.TENSOR` | Canonical tensor collections, role requirements, global/layer/attention/MoE/norm/output/tokenizer coverage, and missing-role truth. | exact DeepSeek source coverage is complete; multi-family evidence remains at its prior rank | No unresolved required tensor collection or runtime role. |
| `TRACK.RESIDENCY` | Payload streaming, materialization, placement, memory planning, CUDA residency, movement, ownership, cleanup, and release. | variant-adaptive bounded materialization feeds one complete model-lifetime encoded host arena; every dense, routed/shared expert, embedding, attention, envelope, norm, and output-head tensor is identity-bound and locked in physical RAM, while the direct CUDA prefix is uploaded once | Any admitted tensor map can materialize from its actual qtype inventory; runtime owners acquire exact immutable and mutable resources without redefining family semantics. |
| `TRACK.RUNTIME` | Runtime binding consumption, immutable runtime-model sealing, execution-session lifecycle, workload descriptors, phase/mode dispatch, reusable contexts, capability truth, invalidation, timing, profiling, and common state-provider boundaries. | correctness-complete hosting, optimized startup, corrected residency accounting, and bounded warm-path profiling are admitted; decode remains below the serving floor and its measured continuation is blocked | One admitted binding opens once into a reusable family-neutral runtime model and supplies a measured, transactionally equivalent full-model execution path to the operator and later evaluation owners. |
| `TRACK.BACKEND` | Hardware/build profiles, CPU/CUDA capability, qtype operations, reference parity, scratch, fallback, synchronization, refusal, and cleanup. | CUDA is fail-closed; selected-qtype backbone operations and direct encoded complete-vocabulary output-head projection have GB10 proof with no CPU numerical fallback | Every required DGX Spark logits-boundary operation is real, reference-compared, and fail-closed. |
| `TRACK.GRAPH` | Primitive contracts, graph construction/planning, attention, position handling, MoE routing/experts, residuals, layers, scratch, and transformer execution. | Complete DeepSeek attention, token-local MoE, embedding, 43-block composition, final mHC collapse, and final RMSNorm are admitted through CPU and GB10 CUDA paths and consumed unchanged by repeated decode | Normalized hidden states from repeated steps hand off to output-head projection. |
| `TRACK.PREFILL` | Identity-bound activation input, deterministic chunking, all-layer attention execution, state ownership, position progression, persistent-state write integration, and cleanup. | numeric token chunks execute embedding and all 43 DeepSeek attention/MoE blocks on CPU and GB10 CUDA and commit persistent state atomically per chunk; prompt text remains unsupported | Repeated decode consumes the same transformer, position, cancellation, and state contracts. |
| `TRACK.KV` | Family-correct KV geometry, allocation, indexing, append/read, capacity, reuse, lifecycle, and cleanup. | session-owned persistent DeepSeek attention state is admitted on CPU and GB10 CUDA with atomic append/read, exact sequence position, clear/reuse, and causal production consumption | Prefill and decode consume the same owned persistent model state. |
| `TRACK.DECODE` | One-step and repeated model-backed decode over descriptor, positions, KV, transformer state, cancellation, and cleanup. | explicit teacher-forced one-token steps reuse one warm transformer/session, consume every prior commit, publish normalized hidden rows, and preserve exact partial progress on CPU/GB10 CUDA | Output-head projection consumes one transformer-normalized decode hidden row. |
| `TRACK.LOGITS` | Normalized hidden-state admission, output-head placement/projection, vocabulary logits, numeric checks, and buffer lifecycle. | final-prefill and decode hidden rows project through the resident separate BF16 output head to complete CPU/CUDA vocabulary logits without re-executing final norm | Real vocabulary logits hand one complete F32 row to sampling. |
| `TRACK.SAMPLING` | Deterministic and stochastic token selection over real vocabulary logits, seeding, reproducibility, validation, and refusal. | repaired full-vocabulary greedy and filter-order-v2 stochastic selection use compensated normalization, positive-mass entropy, atomic close/drain, authenticated evidence, and transactional PCG progress | Selected token IDs and their identities hand off to tokenizer/generation composition without implying append or decode. |
| `TRACK.TOKENIZER` | Exact tokenizer loading, prompt encoding, templates, special/EOS/stop policy, append boundary, detokenization, and failure behavior. | the selected GGUF reconstructs one immutable exact ByteLevel-BPE plan with 129,280 tokens, 127,741 merges, added/special policy, bounded DeepSeek prompt rendering, and batch/incremental decode | Prompt text and sampled IDs traverse the exact tokenizer contract without mutating model state. |
| `TRACK.GENERATION` | Tokenizer/prefill/KV/decode/logits/sampling composition, append, stop, cancellation, partial output, trace, cleanup, and autoregression. | one reusable common turn owner executes exact prompt suffix prefill, first-token sampling, sampled-token decode feedback, typed stop, partial progress, committed-fragment streaming, and incremental text on CPU and mixed CUDA/host paths | Server sessions invoke the same generation owner for fresh and retained turns without rebuilding its loop or lower owners. |
| `TRACK.OPERATOR` | Public and offline CLI grammar, typed protocol input, dispatch, rendering, refusal, topology guards, and operator acceptance. | one `yvex` entrypoint exposes a protocol-only runtime lane and an offline engineering lane; the complete audit freezes the current surface and its P1-P3 debt without changing behavior | The canonical command architecture first establishes one operation/projection authority; the mature REPL then consumes it without duplicating parsing or runtime state. |
| `TRACK.SERVE` | Long-lived local runtime hosting, session/KV authority, streaming, typed telemetry, local and application protocols, product clients, and process topology. | one `yvexd` process owns the process-resident model, exact sessions, Unix protocol listener, and loopback OpenAI listener; protocol v3 and OpenAI Compatibility Profile v1 are complete, while unchanged Bet-tennis-to-real-DeepSeek composition evidence is reopened and public/remote serving remains deferred | Native and OpenAI-compatible clients consume one session/worker/model authority without reopening models or bypassing session semantics. |
| `TRACK.EVAL` | Complete-model behavior, quality, regression, tokenizer, context, and refusal evaluation after real generation exists. | blocked behind active public project control, canonical command architecture, mature REPL console, and measured GB10 optimization; operational profiles are not model evaluation | Repeatable evaluation passes over the optimized hosted application-consumable release path. |
| `TRACK.BENCH` | Reproducible component and complete-model performance measurements with machine, identity, workload, timing, throughput, memory, regression, and visualization metadata. | schema-v5 attention-component measurements and deterministic SVG evidence are implemented; full-model generation benchmark evidence is not measured | Accepted DGX Spark benchmark evidence over the release path. |
| `TRACK.RELEASE` | Final software, conformance, runtime, evaluation, benchmark, operator, packaging, claim, version, release-record, and tag qualification. | blocked; component qualification does not close the release gate | Every release gate passes in one traceable transcript. |
| `TRACK.POST010` | Explicitly deferred portability, serving hardening, speculative execution, extra-family runtime promotion, and later product scope. | deferred | No work enters the active path without a new version-scope decision. |

### 7.2 Per-Track Counts

First-class count order is `complete / active / partial / planned / blocked /
not-measured`. "Complete support" records bounded capability/evidence, not track
closure.

| Track | Recovered IDs | Canonical IDs | First-class C/A/Pa/P/B/NM | Complete support | Open support | Superseded/deferred |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `TRACK.SCOPE` | 25 | 36 | 9/1/1/0/0/0 | 9 | 9 | 7 |
| `TRACK.SOURCE` | 24 | 26 | 2/0/0/0/0/0 | 23 | 1 | 0 |
| `TRACK.MAP` | 12 | 13 | 1/0/0/0/0/0 | 11 | 1 | 0 |
| `TRACK.COMPILATION` | 0 | 13 | 2/0/0/0/0/0 | 0 | 0 | 11 |
| `TRACK.QUANT` | 5 | 6 | 2/0/0/0/0/0 | 3 | 1 | 0 |
| `TRACK.ARTIFACT` | 11 | 16 | 5/0/0/0/0/0 | 3 | 6 | 2 |
| `TRACK.INTEGRITY` | 14 | 15 | 1/0/0/0/0/0 | 10 | 4 | 0 |
| `TRACK.MODEL` | 21 | 23 | 3/0/0/0/0/0 | 9 | 11 | 0 |
| `TRACK.TENSOR` | 27 | 28 | 1/0/0/0/0/0 | 18 | 9 | 0 |
| `TRACK.RESIDENCY` | 42 | 44 | 2/0/0/0/0/0 | 33 | 9 | 0 |
| `TRACK.RUNTIME` | 18 | 20 | 1/0/1/0/1/0 | 17 | 0 | 0 |
| `TRACK.BACKEND` | 29 | 30 | 1/0/0/0/0/0 | 12 | 15 | 2 |
| `TRACK.GRAPH` | 71 | 75 | 3/0/0/0/0/0 | 67 | 4 | 1 |
| `TRACK.PREFILL` | 27 | 28 | 1/0/0/0/0/0 | 24 | 3 | 0 |
| `TRACK.KV` | 21 | 22 | 1/0/0/0/0/0 | 16 | 5 | 0 |
| `TRACK.DECODE` | 15 | 16 | 1/0/0/0/0/0 | 14 | 1 | 0 |
| `TRACK.LOGITS` | 17 | 19 | 1/0/0/0/0/0 | 14 | 3 | 1 |
| `TRACK.SAMPLING` | 15 | 16 | 1/0/0/0/0/0 | 15 | 0 | 0 |
| `TRACK.TOKENIZER` | 13 | 14 | 1/0/0/0/0/0 | 13 | 0 | 0 |
| `TRACK.GENERATION` | 35 | 36 | 1/0/0/0/0/0 | 34 | 1 | 0 |
| `TRACK.OPERATOR` | 81 | 87 | 2/0/0/1/2/0 | 37 | 40 | 2 |
| `TRACK.SERVE` | 12 | 15 | 2/0/0/0/0/0 | 11 | 2 | 0 |
| `TRACK.EVAL` | 15 | 16 | 0/0/0/0/1/0 | 0 | 15 | 0 |
| `TRACK.BENCH` | 16 | 17 | 0/0/0/0/0/1 | 7 | 9 | 0 |
| `TRACK.RELEASE` | 42 | 42 | 0/0/0/0/1/0 | 4 | 37 | 0 |
| `TRACK.POST010` | 23 | 23 | 0/0/0/0/0/0 | 0 | 14 | 9 |

### 7.3 Stable Track Names

The attempted compact recovery introduced replacement lanes that hid stable
ownership. The complete ledger restores the original names:

| Attempted name | Canonical disposition |
| --- | --- |
| `TRACK.ARCHITECTURE` | Use `TRACK.MODEL`; it owns family architecture IR and runtime descriptor projection. |
| `TRACK.EXECUTION` | Use `TRACK.GRAPH`; it owns graph, attention, MoE, layers, and transformer composition. |
| `TRACK.MODELS` | Removed as a future bucket. Family is a scope dimension across source/model/tensor/map/artifact/residency/runtime tracks. |
| `TRACK.PROJECT` | Project-control work remains under `TRACK.SCOPE`; `PROJECT.md` is the authority, not a parallel track. |
| `TRACK.CLAIMS` | Claim boundary remains under `TRACK.SCOPE` and release auditing under `TRACK.RELEASE`. |
| `TRACK.TOPOLOGY` | Existing topology rows remain under `TRACK.OPERATOR` until `TOPOLOGY.CELL.CLOSURE.0` proves a justified split. |

`TRACK.COMPILATION` is an intentional post-recovery addition. It owns immutable
plans and identities between logical mapping and physical lowering; it is not a
replacement name for `TRACK.MODEL`, `TRACK.MAP`, `TRACK.QUANT`, or
`TRACK.ARTIFACT`.

`TRACK.RUNTIME` is an explicit ownership extraction from `TRACK.GENERATION`.
It owns one common model/session execution lifecycle; family adapters provide
typed semantics, while generation retains autoregressive composition. It does
not create a DeepSeek-specific runtime spine or duplicate any canonical ID.

## 8. First-Class Milestone Roadmap

These rows alone are independently schedulable. Detailed delivery prompts are
written after inspection of owner code, consumers, tests, and current contracts.

| Order | Milestone | Track | Scope | State | Conclusive after-state | Depends on |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `V010.DOCS.REFOUNDATION.0` | `TRACK.SCOPE` | project | `complete` | Established the exact DeepSeek v0.1.0 claim boundary and non-claims; its over-deletion is corrected by project recovery. | - |
| 2 | `V010.PROJECT.RECOVERY.0` | `TRACK.SCOPE` | project | `partial` | Recovered product doctrine and a short milestone map but did not restore the per-track wave ledger or multi-family engineering scope. | V010.DOCS.REFOUNDATION.0 |
| 3 | `V010.PROJECT.RECOVERY.1` | `TRACK.SCOPE` | project | `complete` | Installed PROJECT.md as the sole authority, recovered every old wave at a truthful rank, restored counts, and preserved Qwen/Gemma/common work. | V010.PROJECT.RECOVERY.0 |
| 4 | `V010.DOCS.ARCHITECTURE.0` | `TRACK.SCOPE` | project | `complete` | Established non-overlapping documentation owners, absorbed repair obligations, removed duplicate project state, and installed dynamic ledger guards. | V010.PROJECT.RECOVERY.1 |
| 5 | `V010.REBASE.DEEPSEEK.0` | `TRACK.SOURCE` | DeepSeek | `complete` | Verify the exact DeepSeek-V4-Flash source identity, revision, config, tokenizer, shard inventory, footprint, and architecture facts. | V010.DOCS.ARCHITECTURE.0 |
| 6 | `V010.GGUF.QTYPE.ABI.1` | `TRACK.QUANT` | common | `complete` | Closed the pinned GGUF on-disk identity set, exact scalar/block geometry, overflow-safe row-aware byte calculation, typed refusal, and canonical consumer projection. | V010.REBASE.DEEPSEEK.0 |
| 7 | `V010.GGUF.ARTIFACT.ABI.1` | `TRACK.ARTIFACT` | common | `complete` | Closed the file-backed GGUF v3 container, metadata, tensor-directory, qtype/range, resource-budget, immutable-lifetime, typed-refusal, and zero-payload-read ABI at target scale. | V010.GGUF.QTYPE.ABI.1 |
| 8 | `V010.GGUF.LAYOUT.INTEGRITY.1` | `TRACK.INTEGRITY` | common | `complete` | Closed power-of-two alignment, directory-order offsets, qtype-sized raw spans, exact padded continuation, zero padding, aggregate span, truncation, tail, and snapshot-drift admission. | V010.GGUF.ARTIFACT.ABI.1 |
| 9 | `V010.CUDA.FAILCLOSED.0` | `TRACK.BACKEND` | DGX Spark | `complete` | Removed production fallback PTX; separated context, memory, generated bundle, function, and exact variant facts; proved admitted bounded variants and fail-closed rollback/cleanup on GB10. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| 10 | `V010.MODEL.ARCH.IR.0` | `TRACK.MODEL` | DeepSeek | `complete` | Create the execution-complete typed DeepSeek-V4-Flash architecture IR covering attention, positions, KV, mHC, MoE, norms, output, and tokenizer. | V010.CUDA.FAILCLOSED.0 |
| 11 | `V010.TENSOR.COVERAGE.DEEPSEEK.0` | `TRACK.TENSOR` | DeepSeek | `complete` | Derived and reconciled all 69,187 DeepSeek source requirements exactly from the architecture IR against one verified immutable snapshot, with typed refusal and no payload reads. | V010.MODEL.ARCH.IR.0 |
| 12 | `V010.MAP.GGUF.DEEPSEEK.0` | `TRACK.MAP` | DeepSeek | `complete` | Map every required DeepSeek source tensor to one role, GGUF name, transform, shape, expert index, and emitted layout. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| 13 | `V010.SOURCE.PAYLOAD.STREAM.0` | `TRACK.SOURCE` | common | `complete` | Bind verified source snapshots to explicit payload trust, indexed checked ranges, bounded exact reads, and transactional consumer delivery. | V010.MAP.GGUF.DEEPSEEK.0 |
| 14 | `V010.PROJECT.COMPILATION.0` | `TRACK.SCOPE` | project | `complete` | Established model-compilation terminology, ownership, planning/byte-execution planes, future constraints, and the transformation-before-quantization dependency without changing code capability. | V010.SOURCE.PAYLOAD.STREAM.0 |
| 15 | `V010.DOCS.README.COMPILATION.0` | `TRACK.SCOPE` | project | `complete` | Rebuilt the public README around the owned compilation/execution architecture, logical/physical identity, planning/byte-execution split, constraint system, and one truthful implementation boundary. | V010.PROJECT.COMPILATION.0 |
| 16 | `V010.MODEL.TRANSFORM.IR.0` | `TRACK.COMPILATION` | DeepSeek + common plan | `complete` | Sealed one immutable artifact-neutral transformation plan for every required DeepSeek logical output over exact source contributions and payload-range identity, without payload IO during construction. | V010.DOCS.README.COMPILATION.0 |
| 17 | `V010.QUANT.2` | `TRACK.QUANT` | common + DeepSeek roles | `complete` | Implemented the selected physical profile, canonical codecs/capabilities, bounded numeric executor, transactional sink, reference metrics, and direct CPU/CUDA qtype compute without artifact emission. | V010.MODEL.TRANSFORM.IR.0 |
| 18 | `V010.GGUF.WRITER.1` | `TRACK.ARTIFACT` | common | `complete` | Emit concrete GGUF bytes from canonical metadata and tensor descriptors with checked offsets, alignment, atomic publication, failure cleanup, and deterministic order. | V010.QUANT.2 |
| 19 | `V010.ARTIFACT.EMIT.DEEPSEEK.0` | `TRACK.ARTIFACT` | DeepSeek | `complete` | Produce complete reference and selected-release-qtype DeepSeek-V4-Flash GGUF artifacts from verified sources without promoting runtime support. | V010.GGUF.WRITER.1 |
| 20 | `V010.GGUF.ROUNDTRIP.1` | `TRACK.ARTIFACT` | common + DeepSeek artifact | `complete` | Prove writer-reader equivalence through YVEX and an official GGUF reader for metadata, tensor inventory, layout, payload facts, determinism, and cleanup. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| 21 | `V010.ARTIFACT.SUPPORT.CUTOVER.0` | `TRACK.ARTIFACT` | common | `complete` | Admit only complete artifacts into model-support gates and retain bounded files explicitly as tensor proof artifacts. | V010.GGUF.ROUNDTRIP.1 |
| 22 | `V010.ARTIFACT.MATERIALIZE.0` | `TRACK.RESIDENCY` | common | `complete` | Materialized tensors from admitted tensor maps through a family-neutral plan/session with layout/qtype checks, owned bindings, bounded file-backed/staged access, failure cleanup, and release. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| 23 | `V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0` | `TRACK.RESIDENCY` | DeepSeek / DGX Spark | `complete` | Walked every required selected DeepSeek tensor byte through an explicit bounded placement/access plan: 1,360 tensors, 102,396,843,592 payload bytes, 33,792 expert subviews, and 16 MiB peak executor-owned staging. | V010.ARTIFACT.MATERIALIZE.0 |
| 24 | `V010.RUNTIME.DESCRIPTOR.GGUF.0` | `TRACK.MODEL` | common | `complete` | Projected admitted artifact and committed materialization facts into one canonical family-neutral runtime descriptor. | V010.ARTIFACT.MATERIALIZE.0 |
| 25 | `V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0` | `TRACK.MODEL` | DeepSeek | `complete` | Specialized the canonical descriptor into one execution-complete DeepSeek graph-input descriptor with all 1,360 terminal tensors bound and zero missing, duplicate, or unexpected bindings. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 + V010.RUNTIME.DESCRIPTOR.GGUF.0 |
| 26 | `V010.REPO.SEMANTIC.COMPRESSION.0` | `TRACK.SCOPE` | project | `complete` | Reconstructed repository-wide semantic ownership, compacted paths and symbols, enforced family budgets and dependency boundaries, and preserved all admitted identities without promoting inference. | V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0 |
| 27 | `V010.REPO.C.CANONICALIZATION.0` | `TRACK.SCOPE` | project | `complete` | Canonicalized public/internal/private C interfaces, explicit includes, source-relative archive identities, symbol visibility, semantic contracts, large owners, and hard ABI/complexity gates without promoting inference. | V010.REPO.SEMANTIC.COMPRESSION.0 |
| 28 | `V010.GRAPH.DEEPSEEK.ATTENTION.0` | `TRACK.GRAPH` | DeepSeek | `complete` | Complete SWA/CSA/HCA execution consumes the admitted descriptor and real encoded weights through independent reference, production CPU, and device-complete GB10 CUDA paths. | V010.REPO.C.CANONICALIZATION.0 |
| 29 | `V010.CLI.GRAPH.0` | `TRACK.OPERATOR` | DeepSeek + common operator | `complete` | Exposed complete production DeepSeek attention through the main YVEX binary with CPU, CUDA, quick/full canonical probes, typed structured refusal, and explicit non-generation semantics. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| 30 | `V010.RUNTIME.1` | `TRACK.RUNTIME` | common | `complete` | Sealed one family-neutral runtime model/session plane over an immutable runtime binding, resident attention resources, phase-aware execution descriptors, CPU eager and GB10 CUDA eager/graph modes, invalidation, and bounded benchmark/profile and external-chart capability. | V010.CLI.GRAPH.0 |
| 31 | `V010.RUNTIME.DEEPSEEK.KV.0` | `TRACK.KV` | DeepSeek | `complete` | Own session-local persistent DeepSeek state across all 43 attention layers, with exact SWA/CSA/HCA geometry, checked CPU/GB10 CUDA residency, atomic append/read, sequence advancement, capacity, clear/reuse, invalidation, and deterministic release. | V010.RUNTIME.1 |
| 32 | `V010.RUNTIME.DEEPSEEK.PREFILL.0` | `TRACK.PREFILL` | DeepSeek | `complete` | Admit identity-bound 43-layer activation chunks and execute attention prefill on CPU or GB10 CUDA with atomic per-chunk persistent-state publication, exact committed-prefix progression, cancellation rollback, and tensor-file operator reachability. | V010.RUNTIME.DEEPSEEK.KV.0 |
| 33 | `V010.RUNTIME.DEEPSEEK.MOE.0` | `TRACK.GRAPH` | DeepSeek | `complete` | Execute exact hash/learned routing, deterministic top-k, selected routed experts, shared experts, and token-local combination over admitted weights on CPU/CUDA. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| 34 | `V010.GRAPH.DEEPSEEK.TRANSFORMER.0` | `TRACK.GRAPH` | DeepSeek | `complete` | Execute numeric token IDs through selected embedding rows, 43 ordered attention/MoE blocks, deferred mHC residuals, final mHC collapse, final RMSNorm, and one atomic persistent-state transaction on CPU/GB10 CUDA. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| 35 | `V010.RUNTIME.DEEPSEEK.DECODE.0` | `TRACK.DECODE` | DeepSeek | `complete` | Run repeated model-backed decode steps that consume prior KV, advance positions once per token, preserve typed partial progress, and produce real normalized hidden state. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| 36 | `V010.RUNTIME.DEEPSEEK.LOGITS.0` | `TRACK.LOGITS` | DeepSeek | `complete` | Consume transformer-owned normalized hidden state and apply the complete output-head projection to produce vocabulary logits with numeric proof. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| 37 | `V010.RUNTIME.SAMPLING.0` | `TRACK.SAMPLING` | common | `complete` | Select token IDs from complete real vocabulary logits through repaired canonical greedy and filter-order-v2 stochastic policies with transactional RNG, lifecycle, and evidence publication. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| 38 | `V010.RUNTIME.DEEPSEEK.TOKENIZER.0` | `TRACK.TOKENIZER` | DeepSeek | `complete` | Reconstruct the exact tokenizer from admitted GGUF metadata, encode text and bounded prompts, classify specials/EOS, and decode token IDs through batch and incremental lifecycles. | V010.RUNTIME.SAMPLING.0 |
| 39 | `V010.RUNTIME.DEEPSEEK.GENERATION.0` | `TRACK.GENERATION` | DeepSeek | `complete` | Compose tokenizer, prefill, KV, decode, logits, sampling, append, stop, cancellation, partial-output, and cleanup for multiple real tokens. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| 40 | `V010.COMPILATION.PHYSICAL.VARIANT.1` | `TRACK.COMPILATION` | common compiler + DeepSeek first variant | `complete` | Resolve one sealed policy over all 1,360 terminals, execute calibrated mixed qtypes, emit and admit a complete variant, and carry its identity through variant-adaptive materialization and generation. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| 41 | `V010.CLI.DEEPSEEK.GENERATE.0` | `TRACK.OPERATOR` | DeepSeek | `superseded` | The fresh-only top-level CLI plan was replaced before implementation by the process, session, protocol, client, and total CLI refoundation. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| 42 | `V010.RUNTIME.CLIENT.REFOUNDATION.0` | `TRACK.SERVE` | common host + DeepSeek first session vertical | `complete` | Replaced the one-shot process and flat CLI with one long-lived model host, exact multi-turn KV sessions, local streaming protocol, typed telemetry, product client, REPL, a then-separated developer surface, and explicit package topology. | V010.COMPILATION.PHYSICAL.VARIANT.1 |
| 43 | `V010.DOCS.README.PRODUCT.0` | `TRACK.SCOPE` | project | `complete` | Rebuilt the public README around the accepted long-lived host, exact sessions, three terminal views, compact product workflow, binary topology, reviewed SVG architecture, and current non-claims without promoting executable capability. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| 44 | `V010.SERVE.OPENAI.COMPAT.0` | `TRACK.SERVE` | common provider + DeepSeek first tool vertical | `complete` | Added one provider-neutral application contract and a source-separated loopback OpenAI-compatible adapter over the local protocol, with typed messages, tools, structured output, SSE, SDK, and unchanged external-consumer proof. | V010.DOCS.README.PRODUCT.0 |
| 45 | `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` | `TRACK.RUNTIME` | DeepSeek / GB10 | `partial` | Reduced admitted cold startup from roughly fourteen minutes to under four, corrected residency accounting, formalized the warm baseline, and added bounded internal profiling; warm decode remains below admission. | V010.SERVE.OPENAI.COMPAT.0 |
| 46 | `V010.PRODUCT.SURFACE.REALIGNMENT.0` | `TRACK.SCOPE` | product | `complete` | Reduced the product to one public `yvex` command executable and one long-lived `yvexd` process, absorbed engineering routes into a guarded offline lane, integrated the OpenAI listener into the daemon, and removed the retired binaries, service, and package entries without changing runtime mathematics. | V010.RUNTIME.DEEPSEEK.PERFORMANCE.0 |
| 47 | `V010.OPERATOR.SURFACE.AUDIT.0` | `TRACK.OPERATOR` | product + engineering operations | `complete` | Froze the complete post-cutover executable, command, flag, operation, protocol, HTTP, Make, script, environment/configuration, workflow, taxonomy, registry-input, and project-control inventory with zero unmatched categories and no behavior change. | V010.PRODUCT.SURFACE.REALIGNMENT.0 |
| 48 | `V010.PROJECT.CONTROL.PUBLIC.0` | `TRACK.SCOPE` | project | `active` | Refound public project control around the audited product surfaces, explicit successors, gates, non-claims, and one canonical critical path. | V010.OPERATOR.SURFACE.AUDIT.0 |
| 49 | `V010.OPERATOR.RUNTIME.CONSOLE.0` | `TRACK.OPERATOR` | common console + DeepSeek first vertical | `superseded` | The earlier standalone console plan is replaced by the ordered command-architecture and mature-REPL contracts. | V010.OPERATOR.REPL.CONSOLE.0 |
| 50 | `V010.OPERATOR.COMMAND.CONSOLE.0` | `TRACK.OPERATOR` | product commands + runtime console | `superseded` | The combined plan is split before implementation so command/protocol authority closes before terminal presentation begins. | V010.OPERATOR.COMMAND.ARCHITECTURE.0; V010.OPERATOR.REPL.CONSOLE.0 |
| 51 | `V010.OPERATOR.COMMAND.ARCHITECTURE.0` | `TRACK.OPERATOR` | product + engineering operations | `blocked` | Establish one versioned operation source and checked generated descriptors for commands, flags, defaults, validation, help, discovery, projections, and semantic repairs. | V010.PROJECT.CONTROL.PUBLIC.0 |
| 52 | `V010.OPERATOR.REPL.CONSOLE.0` | `TRACK.OPERATOR` | daemon-backed terminal + DeepSeek first vertical | `blocked` | Build the mature linear runtime console, semantic progress, explicit admitted reasoning projection, metrics, watch, trace, and terminal lifecycle entirely over the canonical operation architecture. | V010.OPERATOR.COMMAND.ARCHITECTURE.0 |
| 53 | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | `TRACK.RUNTIME` | DeepSeek / GB10 | `blocked` | Resume issue-shaped warm-runtime optimization from measured attention, MoE, movement, and launch pressure; select no mechanism before its dedicated evidence gate. | V010.OPERATOR.REPL.CONSOLE.0 |
| 54 | `V010.EVAL.DEEPSEEK.0` | `TRACK.EVAL` | DeepSeek | `blocked` | Run repeatable correctness, tokenizer, regression, long-context, refusal, and release-path generation evaluations. | V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 |
| 55 | `V010.BENCH.DEEPSEEK.0` | `TRACK.BENCH` | DeepSeek / DGX Spark | `not-measured` | Record reproducible artifact, qtype, prompt/context, prefill, decode, generation, timing, throughput, and memory evidence. | V010.EVAL.DEEPSEEK.0 |
| 56 | `V010.RELEASE.0` | `TRACK.RELEASE` | DeepSeek v0.1.0 | `blocked` | Close every source, architecture, mapping, artifact, materialization, backend, transformer, generation, evaluation, benchmark, validation, claim, operator, packaging, and version gate. | V010.BENCH.DEEPSEEK.0 |
| 57 | `TOPOLOGY.CELL.CLOSURE.0` | `TRACK.OPERATOR` | common | `planned` | Close residual mixed ownership only where concrete behavior still crosses domain/report/input/command/render/write boundaries. | V010.DOCS.ARCHITECTURE.0; supporting, not a product-stage promotion |

## 9. Complete Track/Wave Ledger

This is the canonical expansion of every track. Every ID appears exactly once
in this section. First-class milestones are separated from recovered subordinate
rows. "Consumer" names the conclusive milestone or future decision that owns
the row's use.

### 9.1 TRACK.SCOPE

**Owner:** Project control, release scope, family/release distinction, claim boundary, version policy, repository semantic ownership, C interface/build boundaries, architecture-control transitions, and public-document transitions.

**Current truth:** repository-wide semantic compression, C canonicalization, the product-first README/architecture cutover, role-based product-surface realignment, and the frozen post-cutover operator audit are complete; public project-control refoundation is active

**Ledger summary:** 25 recovered IDs; 11 first-class milestones; 7 complete support rows; 12 open support rows; 6 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.DOCS.REFOUNDATION.0` | project | `complete` | Established the exact DeepSeek v0.1.0 claim boundary and non-claims; its over-deletion is corrected by project recovery. | - | current |
| `V010.PROJECT.RECOVERY.0` | project | `partial` | Recovered product doctrine and a short milestone map but did not restore the per-track wave ledger or multi-family engineering scope. | V010.DOCS.REFOUNDATION.0 | current |
| `V010.PROJECT.RECOVERY.1` | project | `complete` | Installed PROJECT.md as the sole authority, recovered every old wave at a truthful rank, restored counts, and preserved Qwen/Gemma/common work. | V010.PROJECT.RECOVERY.0 | current |
| `V010.DOCS.ARCHITECTURE.0` | project | `complete` | Established non-overlapping documentation owners, absorbed repair obligations, removed duplicate project state, and installed dynamic ledger guards. | V010.PROJECT.RECOVERY.1 | current |
| `V010.PROJECT.COMPILATION.0` | project | `complete` | Established payload/transformation separation, model-compilation ownership, planning constraints, future obligations, and the Transformation IR dependency without changing executable capability. | V010.SOURCE.PAYLOAD.STREAM.0 | current |
| `V010.DOCS.README.COMPILATION.0` | project | `complete` | Rebuilt README.md around the owned compilation/execution architecture while preserving project-state ownership, exact implemented truth, and unsupported boundaries. | V010.PROJECT.COMPILATION.0 | current |
| `V010.REPO.SEMANTIC.COMPRESSION.0` | project | `complete` | Reconstructed every owned source boundary around compact semantic owners, enforced the dependency DAG and family budget, preserved executable identities, and installed permanent repository-wide gates. | V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0 | current |
| `V010.REPO.C.CANONICALIZATION.0` | project | `complete` | Canonicalized the three C header tiers, explicit include and archive identities, symbol visibility, semantic contracts, large owners, and hard ABI/complexity gates while preserving all capability refusals. | V010.REPO.SEMANTIC.COMPRESSION.0 | current |
| `V010.DOCS.README.PRODUCT.0` | project | `complete` | Rebuilt README.md as the concise public product entry point, validated the accepted three-terminal/one-shot workflow, added reviewed editable product and system architecture visuals, routed depth to owned documents, and changed no executable capability. `cli_applicability=not_applicable`: documentation-only cutover. | V010.RUNTIME.CLIENT.REFOUNDATION.0 | current |
| `V010.PRODUCT.SURFACE.REALIGNMENT.0` | product | `complete` | Provide one public `yvex` command executable, one long-lived `yvexd` process, route-separated runtime and offline lanes, an in-daemon OpenAI listener, and a two-binary package/service topology without promoting runtime capability. | V010.RUNTIME.DEEPSEEK.PERFORMANCE.0 | current |
| `V010.PROJECT.CONTROL.PUBLIC.0` | project | `active` | Refound public project control around the audited product surfaces, successors, gates, non-claims, and one canonical critical path. | V010.OPERATOR.SURFACE.AUDIT.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `SPINE.RETARGET.MULTIFAMILY.0` | `migration` | multi-family | `superseded` | Former multi-family v0.1 release lock. Superseded only as a release contract; its Qwen/Gemma engineering work remains active. | V010.PROJECT.RECOVERY.1 |
| `SPINE.TRACK.CANON.0` | `migration` | project/release | `superseded` | Replace the oversized active spine with the compact track-first map. Superseded by the complete ranked PROJECT.md ledger. | V010.PROJECT.RECOVERY.1 |
| `SPINE.ACTIVE.REWRITE.1` | `migration` | project/release | `superseded` | Superseded active-spine rewrite attempt kept only as a naming marker. Superseded by the complete ranked PROJECT.md ledger. | V010.PROJECT.RECOVERY.1 |
| `SPINE.ROW.CATALOG.0` | `migration` | project/release | `superseded` | Restore explicit active row labels without restoring historical ledger content. Superseded by the complete ranked PROJECT.md ledger. | V010.PROJECT.RECOVERY.1 |
| `SPINE.ROW.CATALOG.1` | `migration` | project/release | `superseded` | Promote the row-label catalog into a trackmap with status and description columns. Superseded by the complete ranked PROJECT.md ledger. | V010.PROJECT.RECOVERY.1 |
| `SPINE.CAPABILITY.MAP.0` | `migration` | project/release | `superseded` | Replace the current snapshot with the v0.1.0 pipeline capability map. Superseded by the complete ranked PROJECT.md ledger. | V010.PROJECT.RECOVERY.1 |
| `V010.SCOPE.0` | `capability` | project/release | `complete` | v0.1.0 release doctrine. | V010.RELEASE.0 |
| `V010.SCOPE.1` | `capability` | project/release | `complete` | v0.1.0 minimum gates. | V010.RELEASE.0 |
| `V010.SCOPE.2` | `subtask` | project/release | `planned` | v0.1.0 non-goals. | V010.RELEASE.0 |
| `V010.SCOPE.3` | `subtask` | project/release | `planned` | v0.1.0 included track map. | V010.RELEASE.0 |
| `V010.SCOPE.4` | `subtask` | project/release | `planned` | v0.1.0 excluded and postponed track map. | V010.RELEASE.0 |
| `V010.SCOPE.5` | `subtask` | project/release | `planned` | v0.1.0 target selection policy. | V010.RELEASE.0 |
| `V010.SCOPE.6` | `subtask` | project/release | `planned` | v0.1.0 release-readiness vocabulary. | V010.RELEASE.0 |
| `V010.SCOPE.7` | `subtask` | project/release | `planned` | v0.1.0 claim boundary map. | V010.RELEASE.0 |
| `V010.TARGET.0` | `capability` | project/release | `complete` | Canonical target registry now separates target ID, repository, revision, source leaf, and inventory authority. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.1` | `evidence` | project/release | `complete` | selected-runtime-slice target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.2` | `evidence` | project/release | `complete` | full-runtime-candidate target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.3` | `evidence` | dense/common | `complete` | dense candidate target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.4` | `subtask` | MoE/common | `planned` | MoE candidate target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.5` | `migration` | DeepSeek | `superseded` | DeepSeek pressure-target framing was absorbed by exact canonical target selection and verification. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.6` | `subtask` | GLM | `planned` | GLM source-only pressure target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.7` | `evidence` | Qwen / Metal | `complete` | Qwen/Metal pressure target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.8` | `subtask` | project/release | `planned` | external reference target report. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.9` | `capability` | project/release | `complete` | v0.1.0 target decision record. | V010.REBASE.DEEPSEEK.0 |
| `V010.TARGET.10` | `capability` | project/release | `complete` | Target selection remains distinct from artifact, runtime, and generation support with typed refusal. | V010.REBASE.DEEPSEEK.0 |


### 9.2 TRACK.SOURCE

**Owner:** Source identity, revision, provider intake, manifests, shards, sidecars, header inventory, payload trust, and bounded payload access.

**Current truth:** exact DeepSeek identity, pinned index, structured sidecars, 46/46 headers, authoritative shard digests, aggregate payload identity, and bounded transactional reads are verified

**Ledger summary:** 24 recovered IDs; 2 first-class milestones; 23 complete support rows; 1 open support row; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.REBASE.DEEPSEEK.0` | DeepSeek | `complete` | Verify the exact DeepSeek-V4-Flash source identity, revision, config, tokenizer, shard inventory, footprint, and architecture facts. | V010.DOCS.ARCHITECTURE.0 | current |
| `V010.SOURCE.PAYLOAD.STREAM.0` | common | `complete` | Binds verified snapshots to explicit upstream-verified or local-sealed payload identity, immutable indexes, checked plans, exact bounded reads, transactional delivery, budgets, concurrency, drift refusal, and deterministic cleanup. | V010.MAP.GGUF.DEEPSEEK.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.SOURCE.0` | `capability` | common | `complete` | Verifier-owned manifests retain v2 metadata-only truth and add v3 per-shard digest authority, payload trust, aggregate identity, parser refusal, and atomic publication. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.1` | `evidence` | common | `complete` | source family/profile fields. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.2` | `capability` | common | `complete` | source artifact class fields. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.3` | `evidence` | common | `complete` | source shard count and footprint report. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.SOURCE.4` | `capability` | common | `complete` | source provenance fields. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.5` | `evidence` | common | `complete` | native safetensors inventory. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.SOURCE.6` | `evidence` | common | `complete` | source tensor metadata inventory. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.7` | `capability` | common | `complete` | source manifest/provenance hardening. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.8` | `subtask` | GLM | `planned` | GLM source pressure report. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.9` | `evidence` | Qwen | `complete` | Qwen source pressure report. | V010.REBASE.DEEPSEEK.0 |
| `V010.SOURCE.10` | `evidence` | common | `complete` | Exact metadata/index/header acceptance remains distinct from payload trust; the payload milestone consumes rather than collapses both states. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.SOURCE.7A` | `capability` | common | `complete` | Add native source tensor download under the models namespace. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `MODELS.DOWNLOAD.0` | `capability` | common | `complete` | Add native source tensor download under the models namespace. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.SOURCE.7B` | `capability` | common | `complete` | Add local provider account preflight for Hugging Face and GitHub. | V010.REBASE.DEEPSEEK.0 |
| `ACCOUNTS.PROVIDER.0` | `capability` | common | `complete` | Add local provider account preflight for Hugging Face and GitHub. | V010.REBASE.DEEPSEEK.0 |
| `MODELS.DOWNLOAD.LIVE.0` | `capability` | common | `complete` | Expose live/plain/log/off source download progress modes. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `MODELS.DOWNLOAD.SIGNAL.0` | `capability` | common | `complete` | Preserve partial source state across interrupted downloads. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `MODELS.DOWNLOAD.CONTROL.0` | `evidence` | common | `complete` | Add download status, stop, resume, and explicit cleanup controls. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `MODELS.SOURCE.IDENTITY.0` | `capability` | common | `complete` | Make downloaded source targets visible to downstream source commands. | V010.REBASE.DEEPSEEK.0 |
| `MODELS.SOURCE.MAP.HANDOFF.0` | `capability` | common | `complete` | Hand downloaded Qwen/Gemma targets into existing map surfaces. | V010.REBASE.DEEPSEEK.0 |
| `MODELS.SOURCE.ROLEMAP.COVERAGE.0` | `evidence` | common | `complete` | Report dynamic downloaded target role coverage from header evidence. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `OWI.TARGETS.QWEN.0` | `evidence` | Qwen | `complete` | Expose the Qwen source target profile. | V010.REBASE.DEEPSEEK.0 |
| `OWI.TARGETS.GEMMA.0` | `evidence` | Gemma | `complete` | Expose the Gemma source target profile. | V010.REBASE.DEEPSEEK.0 |
| `MODEL.TARGET.IDENTITY.0` | `capability` | common | `complete` | Use backend-neutral source target IDs for Qwen and Gemma. | V010.REBASE.DEEPSEEK.0 |


### 9.3 TRACK.MAP

**Owner:** Family source names, canonical runtime roles, GGUF names, physical-lowering projections, layouts, ambiguity refusal, and complete mapping coverage.

**Current truth:** exact DeepSeek source-to-role-to-GGUF lowering and artifact-neutral transformation projection are complete; quantization consumes them without rediscovery, while physical emission remains blocked

**Ledger summary:** 12 recovered IDs; 1 first-class milestone; 11 complete support rows; 1 open support row; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.MAP.GGUF.DEEPSEEK.0` | DeepSeek | `complete` | Map every required DeepSeek source tensor to one role, GGUF name, transform, shape, expert index, and emitted layout. | V010.TENSOR.COVERAGE.DEEPSEEK.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.MAP.0` | `capability` | common | `complete` | Canonical typed mapping schema is consumed by the immutable DeepSeek GGUF lowering plan. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.1` | `capability` | dense/common | `complete` | dense tensor naming map. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.2` | `capability` | MoE/common | `complete` | Routed/shared expert roles, aggregation axes, source contributions, and emitted names are represented by the concrete DeepSeek GGUF lowering plan. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.3` | `capability` | DeepSeek | `complete` | Every verified DeepSeek source tensor maps exactly once into the concrete GGUF lowering plan. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.4` | `subtask` | GLM | `planned` | GLM tensor naming map. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.5` | `capability` | Qwen | `complete` | Qwen tensor naming map. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.6` | `capability` | common | `complete` | output-head tensor mapping. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.7` | `capability` | common | `complete` | tokenizer metadata mapping. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.8` | `evidence` | common | `complete` | missing-role blocker report. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.9` | `evidence` | common | `complete` | Legacy bounded tensor-mapping report gate; it does not close complete DeepSeek mapping. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.GGUF.NAMES.0` | `capability` | common | `complete` | Typed runtime roles project to pinned-standard or explicit extension GGUF names without lexical fallback. | V010.MAP.GGUF.DEEPSEEK.0 |
| `V010.MAP.GGUF.LAYOUT.0` | `capability` | common | `complete` | Logical shapes, qtypes, transforms, source contributions, and aggregation axes are absorbed by the canonical plan; byte layout remains writer-owned. | V010.MAP.GGUF.DEEPSEEK.0 |


### 9.4 TRACK.COMPILATION

**Owner:** Artifact-neutral transformation IR, immutable derivation identity and DAGs, physical model variant identity, compilation constraints, downstream requirement composition, admitted variant selection, and evaluation/benchmark feedback intake.

**Current truth:** the immutable artifact-neutral Transformation IR now feeds one policy-v2 resolver, exact size model, sealed 1,360-decision physical-variant plan, bounded mixed-qtype executor, and native writer. The verified Q8_0/Q2_K preset and one calibrated IQ2_XXS/Q2_K candidate traverse the same owner; automatic precision/target-size optimization, Pareto selection, and placement optimization remain unimplemented

**Ledger summary:** 0 recovered IDs; 2 complete first-class milestones; 0 complete support rows; 0 open support rows; 11 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.MODEL.TRANSFORM.IR.0` | DeepSeek + common plan | `complete` | Represent every required DeepSeek logical output as immutable artifact-neutral typed operations over exact source contributions, bind every input to canonical payload-range identity without plan-time IO, and make physical GGUF lowering and quantization consume that truth. | V010.DOCS.README.COMPILATION.0 | current |
| `V010.COMPILATION.PHYSICAL.VARIANT.1` | common compiler + DeepSeek first variant | `complete` | Resolve sealed presets or custom policy-v2 rules over every terminal, bind calibration, predict exact bytes, execute and emit one complete variant, and authenticate variant-adaptive materialization/runtime execution. | V010.RUNTIME.DEEPSEEK.GENERATION.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `POST010.COMPILATION.DAG.0` | `future` | post-v0.1 | `deferred` | Content-addressed transformation DAG identity and dependency closure. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.REUSE.0` | `future` | post-v0.1 | `deferred` | Incremental reuse of unchanged derivations with identity-safe invalidation. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.VARIANTS.0` | `migration` | post-v0.1 | `superseded` | Explicit physical variants derived from one exact logical source were pulled forward into the policy-driven compiler boundary; automatic multi-variant search remains separately deferred. | V010.COMPILATION.PHYSICAL.VARIANT.1 |
| `POST010.COMPILATION.HARDWARE.PROFILE.0` | `future` | post-v0.1 | `deferred` | Typed hardware constraint profiles without inherited backend support claims. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.WORKLOAD.PROFILE.0` | `future` | post-v0.1 | `deferred` | Typed prompt, context, concurrency, latency, and throughput workload profiles. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.PRECISION.0` | `migration` | post-v0.1 | `superseded` | Explicit role-, collection-, layer-, expert-, operation-, and physical-class precision rules were pulled forward; automatic quality-constrained optimization remains deferred. | V010.COMPILATION.PHYSICAL.VARIANT.1 |
| `POST010.COMPILATION.PLACEMENT.0` | `future` | post-v0.1 | `deferred` | Physical variant placement constraints across SSD, host, unified, and device memory. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.FEEDBACK.0` | `future` | post-v0.1 | `deferred` | Identity-bound evaluation and benchmark feedback ingestion without automatic support promotion. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.PARETO.0` | `future` | post-v0.1 | `deferred` | Pareto-set selection across admitted memory, quality, IO, latency, and throughput evidence. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.RUNTIME.BINDING.0` | `future` | post-v0.1 | `deferred` | Admission identity connecting a physical variant to artifact, residency, backend, descriptor, and runtime state. | post-v0.1 compilation scope decision |
| `POST010.COMPILATION.EXECUTION.STATE.0` | `future` | post-v0.1 | `deferred` | Optional persistent execution-state identity with explicit compatibility and invalidation rules. | post-v0.1 compilation scope decision |


### 9.5 TRACK.QUANT

**Owner:** Dtype/qtype ABI, storage geometry, role policy, conversion, quantization, reference dequantization, compute truth, and refusal.

**Current truth:** the canonical numeric registry, source and output codecs, Q8_0/Q2_K baseline, deterministic IQ2_XXS codec/reference/direct CPU-CUDA compute, exact imatrix ingestion and coverage, bounded Transformation-IR executor, transactional sinks, and role/qtype numeric metrics are complete; artifact emission remains writer-owned

**Ledger summary:** 5 recovered IDs; 2 first-class milestones; 3 complete support rows; 1 open support row; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.GGUF.QTYPE.ABI.1` | common | `complete` | Closed pinned IDs 0-39, removed and outside-baseline identity, exact scalar/block geometry, row-aware shape accounting, overflow/refusal facts, and canonical consumer projection. | V010.REBASE.DEEPSEEK.0 | current |
| `V010.QUANT.2` | common + DeepSeek roles | `complete` | Implemented the role/qtype profile, codecs, bounded execution, reference-dequantization, numeric-bound, transactional-output, and CPU/CUDA compute/refusal matrix without rediscovering source mapping or emitting an artifact. | V010.MODEL.TRANSFORM.IR.0 | recovered/promoted |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.QUANT.0` | `evidence` | common | `complete` | qtype policy report. | V010.QUANT.2 |
| `V010.QUANT.1` | `evidence` | common | `complete` | multi-family dtype/qtype support by runtime role. | V010.QUANT.2 |
| `V010.GGUF.QTYPE.ABI.0` | `evidence` | common | `reopened` | GGUF qtype byte geometry and refusal ABI. Fixture-bounded evidence only; reopened because it does not close the complete ABI. | V010.GGUF.QTYPE.ABI.1 |
| `V010.QUANT.3` | `capability` | common | `complete` | Pinned imatrix ingestion binds source, dataset, producer, file, coverage, and numeric values; every calibrated policy decision refuses uncovered or mismatched evidence before payload reads. | V010.COMPILATION.PHYSICAL.VARIANT.1 |


### 9.6 TRACK.ARTIFACT

**Owner:** GGUF container ABI, native writer, conversion coordination, complete emission, identity, registration, and writer-reader roundtrip.

**Current truth:** scalable native writer, complete DeepSeek reference and selected emission, exact physical identity, native/pinned-official roundtrip, deterministic serialization, and complete-artifact admission are complete

**Ledger summary:** 11 recovered IDs; 5 first-class milestones; 3 complete support rows; 6 open support rows; 2 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.GGUF.ARTIFACT.ABI.1` | common | `complete` | Closed file-backed GGUF v3 structural parsing with target-scale budgets, typed failures, indexed duplicate detection, immutable owned views, canonical qtype/range projection, and zero payload reads. | V010.GGUF.QTYPE.ABI.1 | current |
| `V010.GGUF.WRITER.1` | common | `complete` | Emit concrete GGUF bytes from canonical metadata and tensor descriptors with checked offsets, alignment, atomic publication, failure cleanup, and deterministic order. | V010.QUANT.2 | current |
| `V010.ARTIFACT.EMIT.DEEPSEEK.0` | DeepSeek | `complete` | Produce complete reference and selected-release-qtype DeepSeek-V4-Flash GGUF artifacts from verified sources without promoting runtime support. | V010.GGUF.WRITER.1 | current |
| `V010.GGUF.ROUNDTRIP.1` | common + DeepSeek artifact | `complete` | Prove writer-reader equivalence through YVEX and an official GGUF reader for metadata, tensor inventory, layout, payload facts, determinism, and cleanup. | V010.ARTIFACT.EMIT.DEEPSEEK.0 | current |
| `V010.ARTIFACT.SUPPORT.CUTOVER.0` | common | `complete` | Admit only complete artifacts into model-support gates and retain bounded files explicitly as tensor proof artifacts. | V010.GGUF.ROUNDTRIP.1 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `MODELS.ARTIFACTS.LIST.0` | `evidence` | common | `complete` | List/status local GGUF artifact presence without emitting new artifacts. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| `V010.GGUF.ARTIFACT.ABI.0` | `evidence` | common | `reopened` | GGUF container, metadata, tensor_info, and absolute range ABI boundary. Fixture-bounded evidence only; reopened because it does not close the complete ABI. | V010.GGUF.ARTIFACT.ABI.1 |
| `V010.GGUF.WRITER.0` | `migration` | common | `superseded` | Initial writer row replaced by V010.GGUF.WRITER.1 after ABI refoundation. | V010.GGUF.WRITER.1 |
| `V010.GGUF.ROUNDTRIP.0` | `migration` | common | `superseded` | Initial roundtrip row replaced by V010.GGUF.ROUNDTRIP.1 after ABI refoundation. | V010.GGUF.ROUNDTRIP.1 |
| `V010.ARTIFACT.EMIT.0` | `capability` | common | `complete` | Controlled bounded artifact emission; not complete model emission. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.1` | `evidence` | common | `complete` | Selected tensor proof artifact emission; not complete model emission. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.2` | `subtask` | common | `planned` | generation-capable quantized artifact emission. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.3` | `subtask` | common | `planned` | split artifact plan. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.4` | `subtask` | common | `planned` | artifact parse roundtrip. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.5` | `subtask` | common | `planned` | artifact registration. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |
| `V010.ARTIFACT.EMIT.6` | `subtask` | common | `planned` | v0.1.0 artifact production gate. | V010.ARTIFACT.EMIT.DEEPSEEK.0 |


### 9.7 TRACK.INTEGRITY

**Owner:** Container, metadata, tensor directory, offsets, alignment, qtype-sized ranges, corruption, drift, and artifact admission.

**Current truth:** canonical common global layout and complete v0.1.0 artifact support admission are complete; runtime support remains outside this track

**Ledger summary:** 14 recovered IDs; 1 first-class milestones; 10 complete support rows; 4 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.GGUF.LAYOUT.INTEGRITY.1` | common | `complete` | Closed power-of-two alignment, directory-order offsets, qtype-sized raw spans, exact padded continuation, zero padding, aggregate span, truncation, tail, and snapshot-drift admission. | V010.GGUF.ARTIFACT.ABI.1 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.INTEGRITY.0` | `capability` | common | `complete` | artifact identity manifest. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.1` | `capability` | common | `complete` | size/digest gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.2` | `capability` | common | `complete` | metadata parse gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.3` | `capability` | common | `complete` | tensor directory gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.4` | `capability` | common | `complete` | tensor byte-range gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.5` | `capability` | common | `complete` | shape/rank/dtype gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.6` | `capability` | common | `complete` | element count/overflow gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.7` | `subtask` | common | `planned` | qtype support gate. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| `V010.INTEGRITY.8` | `capability` | common | `complete` | registry drift gate. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.9` | `capability` | common | `complete` | corruption fixture regression. | V010.GGUF.LAYOUT.INTEGRITY.1 |
| `V010.INTEGRITY.10` | `capability` | common | `complete` | materialization preflight gate. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| `V010.INTEGRITY.11` | `subtask` | common | `planned` | graph integrity gate. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| `V010.INTEGRITY.12` | `subtask` | common | `planned` | runtime integrity gate. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |
| `V010.INTEGRITY.13` | `subtask` | common | `planned` | Accept the complete v0.1.0 artifact only after emitted identity, role coverage, payload integrity, and support admission close. | V010.ARTIFACT.SUPPORT.CUTOVER.0 |


### 9.8 TRACK.MODEL

**Owner:** Family architecture profiles, typed architecture IR, layer topology, attention/position/KV/MoE rules, and runtime descriptor projection.

**Current truth:** the immutable execution-complete DeepSeek architecture IR and descriptor drive admitted attention, persistent state, MoE, embedding, transformer ordering, final mHC, final norm, output-head, and tokenizer availability through typed family policies; repeated decode/logits and the artifact-bound tokenizer consume those facts without introducing family policy into common coordinators, while generation composition remains unsupported

**Ledger summary:** 21 recovered IDs; 3 first-class milestones; 9 complete support rows; 11 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.MODEL.ARCH.IR.0` | DeepSeek | `complete` | Created the immutable execution-complete typed DeepSeek-V4-Flash architecture IR covering 43 main layers, one MTP layer, attention, positions, KV requirements, mHC, MoE, norms, output, tokenizer, and source constraints. | V010.CUDA.FAILCLOSED.0 | current |
| `V010.RUNTIME.DESCRIPTOR.GGUF.0` | common | `complete` | Projected admitted artifact and committed materialization facts into one canonical family-neutral runtime descriptor. | V010.ARTIFACT.MATERIALIZE.0 | recovered/promoted |
| `V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0` | DeepSeek | `complete` | Specialized the canonical descriptor into one execution-complete DeepSeek graph-input descriptor with all 1,360 terminal tensors bound and zero missing, duplicate, or unexpected bindings. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 + V010.RUNTIME.DESCRIPTOR.GGUF.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `MODEL.CLASS.QWEN.0` | `evidence` | Qwen | `complete` | Profile Qwen model class from header and sidecar metadata. | V010.MODEL.ARCH.IR.0 |
| `MODEL.CLASS.GEMMA.0` | `evidence` | Gemma | `complete` | Profile Gemma model class from header and sidecar metadata. | V010.MODEL.ARCH.IR.0 |
| `MOE.CLASS.0` | `evidence` | MoE/common | `complete` | Report MoE class facts and runtime blockers. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.0` | `subtask` | common | `planned` | model-class schema finalization. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.1` | `subtask` | common | `planned` | target class detector. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.2` | `subtask` | dense/common | `planned` | dense model-class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.3` | `evidence` | MoE/common | `complete` | MoE model-class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.4` | `subtask` | common | `planned` | source-only class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.5` | `subtask` | common | `planned` | selected-slice class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.6` | `evidence` | DeepSeek | `complete` | DeepSeek class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.7` | `subtask` | GLM | `planned` | GLM class/source-only report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.8` | `evidence` | Qwen | `complete` | Qwen class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.9` | `evidence` | common | `complete` | context class integration. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.10` | `evidence` | common | `complete` | attention class integration. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.11` | `evidence` | common | `complete` | KV class integration. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.12` | `subtask` | common | `planned` | output-head class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.13` | `subtask` | common | `planned` | tokenizer class report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.14` | `subtask` | common | `planned` | runtime requirement report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.15` | `subtask` | common | `planned` | dynamic path selection report. | V010.MODEL.ARCH.IR.0 |
| `V010.CLASS.16` | `subtask` | common | `planned` | v0.1.0 class acceptance gate. | V010.MODEL.ARCH.IR.0 |


### 9.9 TRACK.TENSOR

**Owner:** Canonical tensor collections, role requirements, global/layer/attention/MoE/norm/output/tokenizer coverage, and missing-role truth.

**Current truth:** exact DeepSeek source coverage is complete; multi-family header evidence remains available at its prior rank

**Ledger summary:** 27 recovered IDs; 1 first-class milestones; 18 complete support rows; 9 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.TENSOR.COVERAGE.DEEPSEEK.0` | DeepSeek | `complete` | Derived and reconciled all 69,187 source requirements exactly from the architecture IR against one verified immutable snapshot, with typed refusal and no payload reads. | V010.MODEL.ARCH.IR.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `TENSOR.COLLECTION.QWEN.0` | `evidence` | Qwen | `complete` | Inventory Qwen tensor collections from safetensors headers only. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `TENSOR.COLLECTION.GEMMA.0` | `evidence` | Gemma | `complete` | Inventory Gemma tensor collections from safetensors headers only. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `TENSOR.MOE.0` | `evidence` | MoE/common | `complete` | Report MoE tensor collection coverage and missing runtime pieces. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.0` | `subtask` | common | `planned` | tensor collection schema. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.1` | `evidence` | common | `complete` | embedding collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.2` | `subtask` | common | `planned` | attention norm collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.3` | `subtask` | common | `planned` | post-attention norm collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.4` | `subtask` | common | `planned` | final norm collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.5` | `evidence` | common | `complete` | Q projection collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.6` | `evidence` | common | `complete` | K projection collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.7` | `evidence` | common | `complete` | V projection collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.8` | `evidence` | common | `complete` | O projection collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.9` | `subtask` | common | `planned` | RoPE/position metadata collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.10` | `subtask` | common | `planned` | attention mask/rule collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.11` | `evidence` | common | `complete` | KV runtime-state collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.12` | `evidence` | dense/common | `complete` | dense MLP gate/up/down collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.13` | `subtask` | dense/common | `planned` | dense activation collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.14` | `evidence` | MoE/common | `complete` | MoE router collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.15` | `evidence` | MoE/common | `complete` | MoE expert gate/up/down collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.16` | `evidence` | MoE/common | `complete` | MoE shared expert collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.17` | `evidence` | MoE/common | `complete` | MoE dispatch metadata collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.18` | `evidence` | common | `complete` | output-head collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.19` | `evidence` | common | `complete` | tokenizer metadata collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.20` | `subtask` | common | `planned` | runtime input/output token collection. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.21` | `evidence` | common | `complete` | required tensor coverage report. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.22` | `evidence` | common | `complete` | missing tensor blocker report. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |
| `V010.TENSOR.23` | `subtask` | common | `planned` | v0.1.0 tensor collection gate. | V010.TENSOR.COVERAGE.DEEPSEEK.0 |


### 9.10 TRACK.RESIDENCY

**Owner:** Payload streaming, materialization, placement, memory planning, CUDA residency, movement, ownership, cleanup, and release.

**Current truth:** common source-side streaming and family-neutral materialization derive qtype geometry, binding ranges, expert subviews, and backend capability from each admitted artifact and physical-variant identity. Runtime preparation copies all encoded tensors into one model-lifetime anonymous arena, authenticates every extent, releases consumed artifact cache pages, and requires the complete arena to be locked in physical host RAM before readiness. CUDA retains one direct accelerator prefix while dense, routed/shared expert, embedding, attention, envelope, norm, and output-head bytes remain available from the same immutable host owner

**Ledger summary:** 42 recovered IDs; 2 first-class milestones; 33 complete support rows; 9 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.ARTIFACT.MATERIALIZE.0` | common | `complete` | Materialized tensors from admitted tensor maps through a family-neutral plan/session with layout/qtype checks, owned bindings, bounded file-backed/staged access, failure cleanup, and release. | V010.ARTIFACT.SUPPORT.CUTOVER.0 | recovered/promoted |
| `V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0` | DeepSeek / DGX Spark | `complete` | Walked every required selected DeepSeek tensor byte through an explicit bounded placement/access plan: 1,360 tensors, 102,396,843,592 payload bytes, 33,792 expert subviews, and 16 MiB peak executor-owned staging. | V010.ARTIFACT.MATERIALIZE.0 | current |

#### Recovered And Subordinate Rows

At payload closure, `V010.STORAGE.0` through `.16` and `.18` through `.20`
are explicitly reclassified from planned subtasks to completed capability or
evidence rows because their outcomes are now implemented, consumed, and
tested. Their IDs, `TRACK.RESIDENCY` ownership, and
`V010.SOURCE.PAYLOAD.STREAM.0` consumer remain stable; they are not new
milestones. `V010.STORAGE.17` remains planned because no concrete GLM source
target or payload snapshot exists.

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.ARTIFACT.MATERIALIZE.1` | `capability` | common | `complete` | Materialization consumes actual artifact qtype geometry and physical-variant identity, including mixed IQ2_XXS/Q2_K routed-expert subviews, without fixed-profile or preset-name branches. | V010.COMPILATION.PHYSICAL.VARIANT.1 |
| `V010.STORAGE.0` | `evidence` | common | `complete` | Exact source-stream doctrine is enforced by immutable catalogs, explicit trust, bounded buffers and handles, transactional delivery, and no tensor cache. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.1` | `capability` | common | `complete` | Canonical verified source roots admit only normalized relative shard names through secure root-relative read-only opens; buffers and handles remain session-owned. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.2` | `capability` | common | `complete` | Immutable source shard and tensor indexes build once in linear work and provide deterministic indexed lookup without repeated shard scans. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.3` | `capability` | common | `complete` | A family-neutral immutable shard-index primitive provides the reusable artifact-index foundation; no artifact payload reader or complete artifact is claimed. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.4` | `capability` | common | `complete` | Immutable tensor descriptors bind source and payload identities to checked shard-relative and absolute byte ranges, dtype, rank, and shape. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.5` | `capability` | common | `complete` | Checked range plans expose exact first/last page coverage without widening the requested payload range. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.6` | `capability` | common | `complete` | Single- and multi-range planners produce bounded deterministic chunks grouped by shard and physical order while preserving tensor identity. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.7` | `evidence` | common | `complete` | Cold-advisory probe facts distinguish requested, logical, and physical bytes, handle misses, chunks, elapsed time, and page-cache control limits. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.8` | `evidence` | common | `complete` | Warm-read fixture probes measure the same reader and report actual handle/buffer reuse without treating timing as a release benchmark. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.9` | `evidence` | common | `complete` | Repeated-read diagnostics prove identical bytes and bounded accounting for repeated indexed range delivery. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.10` | `evidence` | common | `complete` | Staged multi-range proof exercises physical-order planning and exact transactional delivery without aggregation or conversion. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.11` | `capability` | common | `complete` | The host policy forbids tensor materialization caches and enforces one bounded reusable chunk buffer under total in-flight byte limits. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.12` | `capability` | common | `complete` | A deterministic LRU shard-handle cache pins active handles, evicts only unpinned entries, and revalidates identity on reopen. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.13` | `evidence` | common | `complete` | Partial-read, early-EOF, and truncation injection abort with typed short-read facts and never commit partial delivery. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.14` | `evidence` | common | `complete` | Missing and unindexed shard fixtures refuse construction or lookup without leaked handles or output success. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.15` | `evidence` | common | `complete` | Known-answer, incremental, fixture mismatch, and live authoritative SHA-256 proofs enforce digest failure and rollback. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.16` | `evidence` | common | `complete` | Partial construction, callback abort, cancellation, busy close, repeated release, and injected cleanup failure have defined counters and cleanup behavior. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.17` | `subtask` | GLM | `planned` | GLM source storage-pressure requirement remains parked until a concrete target and snapshot exist. | future GLM source-scope decision |
| `V010.STORAGE.18` | `evidence` | MoE/common | `complete` | The live map-to-range proof resolves all 67,584 routed-expert contributions plus shared experts under bounded physical-order plans. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.19` | `evidence` | common | `complete` | The live proof resolves and streams the complete output-head contribution as an indexed bounded range. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.STORAGE.20` | `evidence` | common | `complete` | The v0.1.0 source-storage gate admits all 46 shards, 69,187 tensors and contributions, and 1,360 descriptors with upstream payload trust and zero drift/read failures. | V010.SOURCE.PAYLOAD.STREAM.0 |
| `V010.RESIDENCY.0` | `evidence` | common | `complete` | Typed residency facts distinguish the complete locked host arena, direct CUDA prefix, session state, workspaces, access counters, and lifecycle state. | V010.SERVE.OPENAI.COMPAT.0 |
| `V010.RESIDENCY.1` | `capability` | common | `complete` | The sealed runtime model derives one immutable identity-bound resident tensor plan from the admitted runtime binding before execution. | V010.RUNTIME.1 |
| `V010.RESIDENCY.2` | `capability` | common | `complete` | CPU sessions reuse one stable host attention-weight pack and perform zero artifact weight reads during warm execution. | V010.RUNTIME.1 |
| `V010.RESIDENCY.3` | `capability` | CUDA / DGX Spark | `complete` | GB10 sessions upload the complete attention pack once to stable device addresses and perform zero warm weight uploads or allocations. | V010.RUNTIME.1 |
| `V010.RESIDENCY.4` | `subtask` | common | `planned` | managed-memory report. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.5` | `subtask` | common | `planned` | host-staged residency plan. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.6` | `subtask` | common | `planned` | SSD-staged residency plan. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.7` | `subtask` | common | `planned` | SSD-streamed residency plan. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.8` | `subtask` | common | `planned` | hybrid residency plan. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.9` | `subtask` | common | `planned` | distributed future-only report. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.10` | `capability` | common | `complete` | Only selected encoded embedding rows are staged; CUDA decoding and hidden initialization remain device-owned. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.RESIDENCY.11` | `capability` | common | `complete` | The runtime seals all 806 required attention-core and envelope bindings, totaling 5,766,703,652 encoded bytes, into one reusable resident pack. | V010.RUNTIME.1 |
| `V010.RESIDENCY.12` | `capability` | common | `complete` | Session-owned persistent DeepSeek attention state uses stable reusable CPU storage and two GB10 CUDA device banks with checked layout identity, zero warm resizing, and graph invalidation on generation change. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.RESIDENCY.13` | `capability` | dense/common | `complete` | Every admitted dense MLP tensor is indexed in the complete immutable host arena and served without warm artifact reads. | V010.SERVE.OPENAI.COMPAT.0 |
| `V010.RESIDENCY.14` | `capability` | MoE/common | `complete` | Every routed and shared expert tensor is indexed in the complete immutable host arena; mixed qtype expert subviews resolve from stable locked bytes without warm artifact reads. | V010.SERVE.OPENAI.COMPAT.0 |
| `V010.RESIDENCY.15` | `capability` | common | `complete` | The exact 1,059,061,760-byte encoded output head has stable shared model-lifetime CPU/CUDA residency and zero warm reads, uploads, or allocations. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.RESIDENCY.16` | `subtask` | common | `planned` | tokenizer/runtime metadata residency. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |
| `V010.RESIDENCY.17` | `evidence` | common | `complete` | Cold preparation, warm reuse, stable-address verification, and artifact-drift invalidation prove the complete attention residency transition. | V010.RUNTIME.1 |
| `V010.RESIDENCY.18` | `evidence` | common | `complete` | Allocation, upload, cancellation, invalidation, repeated release, and cleanup failures preserve typed state and never publish partial residency. | V010.RUNTIME.1 |
| `V010.RESIDENCY.19` | `subtask` | common | `planned` | v0.1.0 residency gate. | V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0 |


### 9.11 TRACK.RUNTIME

**Owner:** Runtime binding consumption, immutable runtime-model sealing, mutable execution-session lifecycle, execution descriptors, phase/mode dispatch, reusable contexts, capability truth, invalidation, timing, identity-bound profiling, full-model execution optimization, and common state-provider boundaries.

**Current truth:** one family-neutral sealed runtime model/session plane consumes a content-addressed binding without rebuilding compilation truth; the bounded PASS 4 delivery reduced startup, corrected residency accounting, and added internal warm-path profiling, while decode remains below admission and its measured continuation is blocked behind the product-surface sequence

**Ledger summary:** 18 recovered IDs; 3 first-class milestones; 17 complete support rows; 0 open support rows; 0 superseded/deferred rows.

`V010.RUNTIME.0` through `V010.RUNTIME.17` move here explicitly from
`TRACK.GENERATION`; their IDs remain stable. `V010.RUNTIME.1` is promoted from
a planned subtask to the completed common runtime milestone.
`V010.RUNTIME.3`, `.6`, `.7`, and `.8` are reclassified from planned subtasks
to completed evidence because their state-mutation, phase-failure, preflight,
and graph-failure contracts now have executable consumers. All other retained
ranks and states remain unchanged. No row is duplicated and no family-specific
runtime spine is introduced.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.1` | common | `complete` | Sealed one immutable family-neutral runtime model and process-lifetime execution-session plane over a content-addressed binding, resident attention resources, phase-aware execution descriptors, CPU eager and GB10 CUDA eager/graph modes, invalidation, timing, and bounded benchmark/profile and external-chart capability. | V010.CLI.GRAPH.0 | recovered/promoted |
| `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` | DeepSeek / GB10 | `partial` | Reduced cold startup to the accepted sub-four-minute candidate, corrected residency accounting, formalized warm generation, and added bounded movement/stage/synchronization profiling; warm decode remains below admission. | V010.SERVE.OPENAI.COMPAT.0 | current |
| `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | DeepSeek / GB10 | `blocked` | Resume issue-shaped optimization from measured attention, MoE, movement, and launch pressure without preselecting placement, cache, fusion, graph, or prefetch mechanisms. | V010.OPERATOR.REPL.CONSOLE.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.0` | `evidence` | common | `complete` | runtime lifecycle report. | V010.RUNTIME.1 |
| `V010.RUNTIME.2` | `evidence` | common | `complete` | runtime state creation. | V010.RUNTIME.1 |
| `V010.RUNTIME.3` | `evidence` | common | `complete` | runtime state mutation rules. | V010.RUNTIME.1 |
| `V010.RUNTIME.4` | `evidence` | common | `complete` | runtime cleanup idempotence. | V010.RUNTIME.1 |
| `V010.RUNTIME.5` | `evidence` | common | `complete` | partial output preservation. | V010.RUNTIME.1 |
| `V010.RUNTIME.6` | `evidence` | common | `complete` | phase failure vocabulary. | V010.RUNTIME.1 |
| `V010.RUNTIME.7` | `evidence` | common | `complete` | preflight failure behavior. | V010.RUNTIME.1 |
| `V010.RUNTIME.8` | `evidence` | common | `complete` | graph failure behavior. | V010.RUNTIME.1 |
| `V010.RUNTIME.9` | `capability` | common | `complete` | Failing transformer chunks publish neither partial KV nor hidden output and preserve prior committed prefixes. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.RUNTIME.10` | `evidence` | common | `complete` | Persistent state cancellation, staged-layer failure, capacity refusal, abort, commit failure, invalidation, and cleanup preserve the prior committed generation and position. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.RUNTIME.11` | `evidence` | common | `complete` | Failing or cancelled decode steps publish neither KV nor hidden rows; prior successful steps remain committed and the first incomplete ordinal remains exact. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.RUNTIME.12` | `capability` | common | `complete` | Logits failures publish no partial vocabulary row, preserve prior repeated rows, leave persistent position unchanged, and retain the primary causal error through cleanup. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.RUNTIME.13` | `evidence` | common | `complete` | Sampling failures and cancellation publish no token or candidate identity, advance no RNG state, retain prior repeated results, and never mutate model/session state. | V010.RUNTIME.SAMPLING.0 |
| `V010.RUNTIME.14` | `capability` | common | `complete` | Sampled, submitted, model-committed, detokenized, and text-published stages preserve exact append failure and partial-progress truth. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.RUNTIME.15` | `evidence` | common | `complete` | cancellation safe points. | V010.RUNTIME.1 |
| `V010.RUNTIME.16` | `capability` | common | `complete` | The internal operator maps SIGINT/SIGTERM observation into generation cancellation without owning lower transaction semantics. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.RUNTIME.17` | `capability` | common | `complete` | Atomic close ownership, active-use drain, concurrent-entry refusal, cancellation, retry-safe cleanup, and sanitizer aggregation close the generation lifecycle gate. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |


### 9.12 TRACK.BACKEND

**Owner:** Hardware/build profiles, CPU/CUDA capability, qtype operations, reference parity, scratch, fallback, synchronization, refusal, and cleanup.

**Current truth:** CUDA capability admission is fail-closed; attention retains admitted eager and CUDA Graph modes, the complete GB10 eager transformer path executes with device-resident inter-layer activations, and the logits path projects the complete vocabulary directly from the resident encoded output head without CPU numerical fallback

**Ledger summary:** 29 recovered IDs; 1 first-class milestones; 12 complete support rows; 15 open support rows; 2 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.CUDA.FAILCLOSED.0` | DGX Spark | `complete` | Remove advertised no-op fallback support and make every claimed CUDA operation real, reference-compared, or explicitly unsupported. | V010.GGUF.LAYOUT.INTEGRITY.1 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `CUDA.KERNEL.0` | `capability` | CUDA / DGX Spark | `complete` | Harden bounded CUDA primitive kernels without claiming CUDA runtime generation. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.0` | `capability` | common | `complete` | Exact typed context, memory, generated-bundle, function, dtype/variant, and failure capability matrix. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.1` | `evidence` | common | `complete` | CPU baseline capability report. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.2` | `evidence` | CUDA / DGX Spark | `complete` | CUDA capability report. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.3` | `capability` | CUDA / DGX Spark | `complete` | CUDA allocation proof. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.4` | `capability` | CUDA / DGX Spark | `complete` | CUDA transfer proof. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.5` | `capability` | CUDA / DGX Spark | `complete` | CUDA op parity subset. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.6` | `capability` | common | `complete` | Exact unsupported/failed variants refuse before dispatch with typed reasons. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.7` | `capability` | common | `complete` | A build without the generated kernel bundle retains only proven Driver API memory operations and refuses kernels. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.8` | `capability` | common | `complete` | Current bounded primitive scratch is explicit caller-owned tensor storage or checked temporary CUDA allocation. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.9` | `capability` | common | `complete` | Module/function rollback, launch/sync demotion, output state, checked release, and truthful allocation accounting are tested. | V010.CUDA.FAILCLOSED.0 |
| `V010.BACKEND.10` | `future` | Metal | `deferred` | Metal feasibility and implementation require a post-v0.1 scope decision. | post-v0.1 scope decision |
| `V010.BACKEND.11` | `future` | ROCm | `deferred` | ROCm feasibility and implementation require a post-v0.1 scope decision. | post-v0.1 scope decision |
| `V010.BACKEND.12` | `capability` | DeepSeek / DGX Spark | `complete` | The GB10 eager backbone operation set executes embedding, attention, MoE, mHC residual composition, final collapse, and final norm without CPU numerical fallback. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.HARDWARE.0` | `subtask` | common | `planned` | local workstation profile. | V010.RELEASE.0 |
| `V010.HARDWARE.1` | `subtask` | common | `planned` | Spark/GB10 profile. | V010.RELEASE.0 |
| `V010.HARDWARE.2` | `subtask` | common | `planned` | Mac/Apple Silicon profile. | V010.RELEASE.0 |
| `V010.HARDWARE.3` | `subtask` | common | `planned` | Strix Halo/ROCm future profile. | V010.RELEASE.0 |
| `V010.HARDWARE.4` | `subtask` | common | `planned` | memory budget report. | V010.RELEASE.0 |
| `V010.HARDWARE.5` | `subtask` | common | `planned` | storage bandwidth pressure report. | V010.RELEASE.0 |
| `V010.HARDWARE.6` | `subtask` | common | `planned` | reproducibility metadata profile. | V010.RELEASE.0 |
| `V010.BUILD.0` | `subtask` | common | `planned` | build profile matrix. | V010.RELEASE.0 |
| `V010.BUILD.1` | `subtask` | common | `planned` | CPU debug build. | V010.RELEASE.0 |
| `V010.BUILD.2` | `subtask` | common | `planned` | CPU release build. | V010.RELEASE.0 |
| `V010.BUILD.3` | `subtask` | CUDA / DGX Spark | `planned` | CUDA debug build. | V010.RELEASE.0 |
| `V010.BUILD.4` | `subtask` | CUDA / DGX Spark | `planned` | CUDA release build. | V010.RELEASE.0 |
| `V010.BUILD.5` | `subtask` | common | `planned` | sanitizer build. | V010.RELEASE.0 |
| `V010.BUILD.6` | `subtask` | common | `planned` | release artifact hygiene. | V010.RELEASE.0 |
| `V010.BUILD.7` | `subtask` | common | `planned` | v0.1.0 build gate. | V010.RELEASE.0 |


### 9.13 TRACK.GRAPH

**Owner:** Primitive contracts, graph construction/planning, attention, position handling, MoE routing/experts, residuals, layers, scratch, and transformer execution.

**Current truth:** the complete DeepSeek backbone is admitted from numeric token IDs to normalized hidden state through selected embedding rows, four-stream mHC state, 43 ordered attention/MoE blocks, deferred FFN posts, final mHC collapse, and final RMSNorm. CPU and GB10 CUDA repeated decode consume the same immutable plan and one transaction per token. The generation owner now consumes this unchanged graph boundary for exact prompt prefill and sampled-token feedback without moving graph mathematics into runtime composition

**Ledger summary:** 71 recovered IDs; 3 first-class milestones; 3 complete milestones; 67 complete support rows; 4 open support rows; 1 superseded/deferred row.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.GRAPH.DEEPSEEK.ATTENTION.0` | DeepSeek | `complete` | Complete SWA/CSA/HCA execution with real history/top-k participation, exact ratio-128 HCA behavior, independent full-equation reference, and direct GB10 CUDA parity. | V010.REPO.C.CANONICALIZATION.0 | current |
| `V010.RUNTIME.DEEPSEEK.MOE.0` | DeepSeek | `complete` | Execute exact hash/learned routing, deterministic top-k, selected routed/shared experts, and token-local combination over admitted weights on CPU/CUDA. | V010.RUNTIME.DEEPSEEK.PREFILL.0 | current |
| `V010.GRAPH.DEEPSEEK.TRANSFORMER.0` | DeepSeek | `complete` | Execute numeric token IDs through selected embedding rows, 43 ordered attention/MoE blocks, deferred mHC residuals, final mHC collapse, final RMSNorm, and one atomic persistent-state transaction on CPU/GB10 CUDA. | V010.RUNTIME.DEEPSEEK.MOE.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.GRAPH.PRIM.0` | `subtask` | common | `planned` | primitive inventory report. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.PRIM.1` | `capability` | common | `complete` | Versioned RoPE/YaRN position primitives execute through the runtime-numeric authority on CPU and CUDA. | V010.RUNTIME.1 |
| `V010.GRAPH.PRIM.2` | `capability` | common | `complete` | Generic attention primitives implement typed history, top-k, masks, softmax, reduction, state deltas, and transactional publication. | V010.RUNTIME.1 |
| `V010.GRAPH.PRIM.3` | `capability` | common | `complete` | Encoded F32, BF16, and Q8_0 projection primitives consume admitted resident bindings on CPU and CUDA. | V010.RUNTIME.1 |
| `V010.GRAPH.PRIM.4` | `capability` | common | `complete` | Generic typed routed/shared SwiGLU primitives execute exact selected expert work on CPU/CUDA. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.PRIM.5` | `capability` | MoE/common | `complete` | Canonical materialization expert subviews address only selected gate/up/down ranges. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.PRIM.6` | `capability` | common | `complete` | Stable-softmax, mask, scale, and reduction policies are versioned, reference-compared, and backend-admitted. | V010.RUNTIME.1 |
| `V010.GRAPH.PRIM.7` | `capability` | common | `complete` | Versioned SiLU/SwiGLU activation policy and limit execute under the family-selected MoE contract. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.PRIM.8` | `capability` | common | `complete` | The generic envelope executes the immediate typed residual/add and mHC composition owned by attention. | V010.RUNTIME.1 |
| `V010.GRAPH.PRIM.9` | `capability` | common | `complete` | Attention/FFN RMSNorm and transformer final RMSNorm execute through one numeric policy. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.PRIM.10` | `evidence` | common | `complete` | Embedding, mHC, normalization, attention, and MoE composition pass focused and live CPU/CUDA regression. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.0` | `evidence` | common | `complete` | The immutable transformer plan binds every ordered component and final-stage requirement. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.1` | `capability` | common | `complete` | Numeric token input gathers exact selected encoded embedding rows without full-matrix decode/upload. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.2` | `capability` | common | `complete` | The attention envelope applies the admitted input-normalization boundary before Q/K/V projection. | V010.RUNTIME.1 |
| `V010.GRAPH.3` | `capability` | common | `complete` | Q projection executes from resident encoded weights under the immutable descriptor and runtime-numeric contract. | V010.RUNTIME.1 |
| `V010.GRAPH.4` | `capability` | common | `complete` | K projection executes from resident encoded weights and produces the typed attention-state candidate. | V010.RUNTIME.1 |
| `V010.GRAPH.5` | `capability` | common | `complete` | V projection executes from resident encoded weights and participates in local and compressed history reduction. | V010.RUNTIME.1 |
| `V010.GRAPH.6` | `capability` | common | `complete` | Versioned RoPE/YaRN position application executes at the exact family-selected semantic boundary. | V010.RUNTIME.1 |
| `V010.GRAPH.7` | `capability` | common | `complete` | SWA, CSA, and HCA score paths consume real local, compressed, indexed, and selected history. | V010.RUNTIME.1 |
| `V010.GRAPH.8` | `capability` | common | `complete` | Causal, sliding-window, sparse-candidate, padding, and validity masks are applied before publication. | V010.RUNTIME.1 |
| `V010.GRAPH.9` | `capability` | common | `complete` | Stable softmax executes with explicit finite-value and numeric-bound handling on CPU and CUDA. | V010.RUNTIME.1 |
| `V010.GRAPH.10` | `capability` | common | `complete` | Value reduction consumes selected attention probabilities and complete value vectors in every admitted mode. | V010.RUNTIME.1 |
| `V010.GRAPH.11` | `capability` | common | `complete` | The attention-owned O projection consumes resident encoded weights and emits the separate core result. | V010.RUNTIME.1 |
| `V010.GRAPH.12` | `capability` | common | `complete` | The immediate residual/mHC post-transform publishes the separate envelope result without executing FFN-owned work. | V010.RUNTIME.1 |
| `V010.GRAPH.13` | `capability` | common | `complete` | Attention envelope output enters the exact FFN mHC/RMSNorm boundary before MoE. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.14` | `subtask` | dense/common | `planned` | dense MLP gate/up/down path. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.15` | `subtask` | dense/common | `planned` | dense MLP residual path. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.16` | `capability` | common | `complete` | Transformer-owned final mHC collapse and final RMSNorm execute over the final expanded state. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.17` | `capability` | common | `complete` | Transformer publishes normalized `[token_count, hidden_width]` state for decode/logits consumers. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.18` | `capability` | common | `complete` | Sealed execution descriptors bind reusable host/device workspace and attention scratch with zero warm resizing or allocation. | V010.RUNTIME.1 |
| `V010.GRAPH.19` | `evidence` | common | `complete` | Allocation, dispatch, cancellation, publication, rollback, synchronization, and cleanup faults preserve prior committed graph state. | V010.RUNTIME.1 |
| `V010.GRAPH.20` | `capability` | common | `complete` | One production block composes attention, MoE, and deferred FFN mHC post without copying component mathematics. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.21` | `capability` | common | `complete` | All 43 blocks execute in canonical order with cross-layer residual propagation and one KV transaction. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.22` | `evidence` | common | `complete` | Selected-slice graph proof; not full attention or transformer execution. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.23` | `evidence` | common | `complete` | The admitted complete artifact executes one token through the full CPU/GB10 CUDA backbone with equal final digest. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.GRAPH.24` | `evidence` | common | `complete` | The transformer graph gate closes exact ordering, component coverage, atomic state, CLI reachability, and higher-stage refusals. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.ATTN.0` | `evidence` | common | `complete` | Typed requirements cover every admitted SWA, CSA, and HCA descriptor, binding, numeric policy, state boundary, and backend variant. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.1` | `capability` | common | `complete` | Q bindings are shape/qtype validated and consumed by real encoded projection execution. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.2` | `capability` | common | `complete` | K bindings are shape/qtype validated and consumed by real projection and state-delta execution. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.3` | `capability` | common | `complete` | V bindings are shape/qtype validated and consumed by real local and selected-history reduction. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.4` | `capability` | common | `complete` | O bindings are shape/qtype validated and consumed by the complete attention output projection. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.5` | `capability` | common | `complete` | Typed head, grouped-head, latent, and compressor geometry is validated before mutation for all 43 layers. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.6` | `capability` | common | `complete` | Position application follows the family-selected RoPE/YaRN rule across prefill-chunk and decode phases. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.7` | `capability` | common | `complete` | Causal, window, candidate, padding, and validity masks participate in the real attention result. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.8` | `capability` | common | `complete` | Attention scratch is bounded, session-owned, reusable, identity-bound, and rollback-safe. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.9` | `capability` | common | `complete` | Complete SWA/CSA/HCA core execution and its immediate envelope run through production CPU and GB10 CUDA paths. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.10` | `subtask` | common | `planned` | GQA/MQA/MLA family rule if required. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.11` | `evidence` | common | `complete` | Independent full-equation oracle, mutation, boundary, and CPU/CUDA parity lanes prove the admitted attention semantics. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.12` | `capability` | common | `complete` | Typed attention failure, cancellation, atomic state publication, rollback, and idempotent cleanup preserve prior committed state. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.ATTN.13` | `evidence` | common | `complete` | The v0.1 attention gate executes all 43 layers and 634 core bindings while retaining explicit KV, transformer, and generation refusals. | V010.GRAPH.DEEPSEEK.ATTENTION.0 |
| `V010.MOE.0` | `capability` | MoE/common | `complete` | One immutable typed MoE plan owns the executable token-local requirement set. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.1` | `evidence` | MoE/common | `complete` | Exact routed/shared expert counts and geometry derive from admitted family facts. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.2` | `capability` | MoE/common | `complete` | Six selected routed experts per token execute without reading the full collection. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.3` | `capability` | MoE/common | `complete` | One distinct shared expert per layer executes outside routed subview addressing. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.4` | `capability` | MoE/common | `complete` | Hash tables and learned router weights/biases bind exact admitted tensor roles. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.5` | `capability` | MoE/common | `complete` | Learned router logits and transformed scores publish through typed execution evidence. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.6` | `capability` | MoE/common | `complete` | `sqrtsoftplus`, correction bias, `noaux_tc`, normalization, and routed scaling are exact. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.7` | `capability` | MoE/common | `complete` | Generic deterministic top-k defines stable ties and refuses malformed scores. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.8` | `capability` | MoE/common | `complete` | Runtime resolves only exact gate/up/down physical subviews for selected experts. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.9` | `capability` | MoE/common | `complete` | Typed CPU/CUDA jobs dispatch selected routed and shared expert work. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.10` | `evidence` | MoE/common | `complete` | Hash and learned dispatch pass independent and live selected-artifact proof. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.11` | `capability` | MoE/common | `complete` | Encoded Q2_K routed and Q8_0 shared SwiGLU compute passes CPU/CUDA reference comparison. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.12` | `capability` | MoE/common | `complete` | Weighted routed accumulation uses exact normalized/scaled selection weights. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.13` | `capability` | MoE/common | `complete` | Shared output is combined distinctly with the routed aggregate. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.14` | `capability` | MoE/common | `complete` | Transformer consumes MoE post/combination facts in the exact deferred FFN mHC residual update. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.MOE.15` | `capability` | MoE/common | `complete` | Cancellation/failure publishes no partial output and cleanup remains deterministic. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.16` | `evidence` | MoE/common | `complete` | Real selected-expert subviews execute over the admitted complete artifact. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.17` | `capability` | MoE/common | `complete` | The token-local MoE block publishes router, routed, shared, combined, and deferred-post facts. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.MOE.18` | `capability` | MoE/common | `complete` | Numeric-token full-model prefill composes every layer's routed/shared MoE output. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.MOE.19` | `capability` | MoE/common | `complete` | Every repeated decode step composes all 43 admitted routed/shared MoE blocks through the existing transformer path. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.MOE.20` | `evidence` | MoE/common | `complete` | All 43 layers execute 3 hash and 40 learned routers on CPU/GB10 CUDA through the main CLI. | V010.RUNTIME.DEEPSEEK.MOE.0 |
| `V010.RUNTIME.DEEPSEEK.ATTENTION.KV.0` | `migration` | DeepSeek | `superseded` | Attempted recovery milestone split into real owner milestones for attention, KV, and prefill; no implementation was lost. | V010.GRAPH.DEEPSEEK.ATTENTION.0 + V010.RUNTIME.DEEPSEEK.KV.0 + V010.RUNTIME.DEEPSEEK.PREFILL.0 |


### 9.14 TRACK.PREFILL

**Owner:** Identity-bound activation input, deterministic chunking, all-layer attention execution, state ownership, position progression, persistent-state write integration, and cleanup.

**Current truth:** the retained activation-prefill component remains admitted, and the transformer now adds numeric token embedding plus MoE composition to produce complete normalized hidden chunks. Both paths share deterministic chunking, session position, cancellation, and atomic persistent-state publication; prompt text remains unsupported

**Ledger summary:** 27 recovered IDs; 1 first-class milestone; 24 complete support rows; 3 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.PREFILL.0` | DeepSeek | `complete` | Admit exact activation chunks for all 43 attention layers, execute them on CPU or GB10 CUDA, and atomically publish persistent state and committed-prefix progression without claiming prompt or full-model prefill. | V010.RUNTIME.DEEPSEEK.KV.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.CONTEXT.0` | `capability` | common | `complete` | active activation-context and chunk-capacity policy. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.1` | `evidence` | common | `complete` | model max context report. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.2` | `evidence` | common | `complete` | requested context report. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.3` | `evidence` | common | `complete` | chunk size policy. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.4` | `evidence` | common | `complete` | chunk planner. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.5` | `evidence` | common | `complete` | prefill position policy. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.6` | `evidence` | common | `complete` | decode position policy. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.7` | `evidence` | common | `complete` | overflow refusal behavior. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.8` | `subtask` | common | `planned` | context stop behavior. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.9` | `capability` | common | `complete` | activation context, position, chunk, and committed-prefix trace. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.CONTEXT.10` | `capability` | common | `complete` | activation-prefill context gate. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.0` | `evidence` | common | `complete` | prefill requirement report. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.1` | `evidence` | common | `complete` | token input to prefill planner. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.2` | `capability` | common | `complete` | Numeric-token transformer prefill executes selected encoded embedding rows. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.PREFILL.3` | `capability` | common | `complete` | typed activation entry for the first and every ordered attention layer. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.4` | `capability` | common | `complete` | production activation-driven attention prefill. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.5` | `capability` | common | `complete` | atomic persistent-state write during activation prefill. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.6` | `subtask` | dense/common | `planned` | dense MLP prefill. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.7` | `capability` | MoE/common | `complete` | Full-model numeric-token prefill executes routed/shared MoE for every transformer block. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 |
| `V010.PREFILL.8` | `capability` | common | `complete` | repeated execution across all 43 ordered attention-layer activations. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.9` | `capability` | common | `complete` | deterministic bounded activation chunk execution. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.10` | `subtask` | common | `planned` | staged/SSD prefill plan. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.11` | `capability` | common | `complete` | session-owned prefill state and exact committed-prefix progression. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.12` | `capability` | common | `complete` | per-chunk abort, cancellation, rollback, and cleanup. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.13` | `capability` | common | `complete` | typed activation-prefill execution and state trace. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.14` | `capability` | common | `complete` | causal, chunk-equivalence, clear/reuse, and CPU/CUDA regression. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |
| `V010.PREFILL.15` | `capability` | common | `complete` | activation-attention prefill component gate. | V010.RUNTIME.DEEPSEEK.PREFILL.0 |


### 9.15 TRACK.KV

**Owner:** Family-correct KV geometry, allocation, indexing, append/read, capacity, reuse, lifecycle, and cleanup.

**Current truth:** persistent family-correct DeepSeek attention state is admitted across 2 SWA, 21 CSA, and 20 HCA layers, including raw local, compressed, indexer, and rolling components; isolated sessions allocate exact CPU/GB10 CUDA storage and atomically append, read, advance, clear, invalidate, and release state consumed by later production attention execution

**Ledger summary:** 21 recovered IDs; 1 first-class milestone; 16 complete support rows; 5 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.KV.0` | DeepSeek | `complete` | Own exact session-local DeepSeek state for every attention class and layer through checked CPU/CUDA residency, all-layer transactional publication, causal read-after-write, capacity/position continuity, clear/reuse, invalidation, and release. | V010.RUNTIME.1 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.KV.0` | `evidence` | common | `complete` | KV requirement report. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.1` | `evidence` | common | `complete` | KV shape policy. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.2` | `capability` | common | `complete` | Exact F32 attention-state representation is derived from typed family recipes and bound into the layout identity; no KV quantization claim is made. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.3` | `evidence` | common | `complete` | KV capacity estimator. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.4` | `capability` | common | `complete` | Each session allocates one exact bounded host layout and retains stable storage across warm execution and clear/reuse. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.5` | `capability` | CUDA / DGX Spark | `complete` | Each GB10 session allocates two stable device banks once, performs checked D2D reads and staged publication, and never falls back to CPU numerical work. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.6` | `capability` | common | `complete` | Real attention key, compressed, indexer, and rolling publications stage into the candidate generation. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.7` | `capability` | common | `complete` | Real attention value and corresponding rolling publications stage into the same model-state transaction. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.8` | `capability` | common | `complete` | Later production attention resolves and consumes the exact committed state view on CPU or its resident device span on CUDA. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.9` | `capability` | common | `complete` | One checked provider mapping owns layer, component, logical position, physical slot, width, valid range, and bank selection. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.10` | `capability` | common | `complete` | Sequence position and mutable generation advance exactly once after complete all-layer commit. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.11` | `capability` | common | `complete` | Overflow, non-contiguous append, stale generation, invalid component, and out-of-range read refuse before mutation. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.12` | `capability` | common | `complete` | Clear preserves compatible allocation, resets content and position, invalidates graph dependents, and matches a fresh-session first execution. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.13` | `evidence` | common | `complete` | Failure, cancellation, abort, invalidation, repeated release, and partial CUDA allocation cleanup preserve committed state or return typed failure. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.14` | `evidence` | common | `complete` | KV trace/inspect. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.15` | `subtask` | common | `planned` | paged KV plan. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.16` | `subtask` | common | `planned` | paged KV skeleton. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.17` | `subtask` | common | `planned` | host spill experiment. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.18` | `subtask` | common | `planned` | SSD spill experiment. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.19` | `subtask` | common | `planned` | KV quantization policy. | V010.RUNTIME.DEEPSEEK.KV.0 |
| `V010.KV.20` | `evidence` | common | `complete` | The v0.1.0 persistent-state gate passes full 43-layer CPU/GB10 CUDA append/read causality, transactional lifecycle, CLI reachability, and refusal evidence. | V010.RUNTIME.DEEPSEEK.KV.0 |


### 9.16 TRACK.DECODE

**Owner:** One-step and repeated model-backed decode over descriptor, positions, KV, transformer state, cancellation, and cleanup.

**Current truth:** one explicit decode coordinator reuses the admitted transformer token schema and one warm transformer/session context. Each externally supplied token executes all 43 blocks over authoritative prior KV, commits once, publishes one normalized hidden row, and advances position/generation once; cancellation or failure preserves exact prior-step progress

**Ledger summary:** 15 recovered IDs; 1 first-class milestone; 14 complete support rows; 1 open support row; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.DECODE.0` | DeepSeek | `complete` | Run repeated model-backed decode steps that consume prior KV, advance position and generation once per successful token, preserve typed partial progress, and produce real normalized hidden state. | V010.GRAPH.DEEPSEEK.TRANSFORMER.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.DECODE.0` | `evidence` | common | `complete` | decode requirement report. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.1` | `evidence` | common | `complete` | decode state ownership. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.2` | `evidence` | common | `complete` | decode position input. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.3` | `capability` | common | `complete` | Each step validates and consumes the authoritative committed persistent-state prefix. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.4` | `capability` | common | `complete` | Explicit one-token decode executes all 43 admitted DeepSeek attention blocks. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.5` | `subtask` | dense/common | `planned` | decode dense MLP path. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.6` | `capability` | MoE/common | `complete` | Each step composes 3 hash and 40 learned routers, 258 routed selections, and 43 shared experts through the existing MoE owner. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.7` | `capability` | common | `complete` | Every successful step publishes one caller-owned normalized hidden row of shape `[1,4096]`. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.8` | `capability` | common | `complete` | One production decode-step API validates position, executes the complete backbone, and commits state exactly once. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.9` | `capability` | common | `complete` | One repeated coordinator executes ordered teacher-forced tokens over a reused transformer/session context. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.10` | `evidence` | common | `complete` | Cancellation between steps preserves every earlier committed row and identifies the first incomplete ordinal. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.11` | `evidence` | common | `complete` | Failing-step output/state remains unpublished and dependent-first cleanup preserves the primary error. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.12` | `evidence` | common | `complete` | Field-wise step and aggregate identities bind token, phase, position, generation, routing, hidden, state, and structural facts. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.13` | `evidence` | common | `complete` | Sequence equivalence, prefix/token causality, CPU/CUDA parity, and warm-context reuse pass selected-artifact regression. | V010.RUNTIME.DEEPSEEK.DECODE.0 |
| `V010.DECODE.14` | `evidence` | common | `complete` | The v0.1.0 teacher-forced decode gate passes production API, CLI, lifecycle, CPU, and GB10 CUDA evidence. | V010.RUNTIME.DEEPSEEK.DECODE.0 |


### 9.17 TRACK.LOGITS

**Owner:** Normalized hidden-state admission, output-head placement/projection, vocabulary logits, numeric checks, and buffer lifecycle.

**Current truth:** one immutable family-neutral plan admits the separate unbiased DeepSeek `output.weight` binding, model-lifetime CPU/CUDA residency, and direct encoded BF16 projection. Transformer-authenticated final-prefill and decode hidden rows produce complete ordered 129,280-value F32 logits rows without re-executing final norm or mutating persistent state; partial execution publishes only its completed row prefix

**Ledger summary:** 17 recovered IDs; 1 first-class milestones; 14 complete support rows; 3 open support rows; 1 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.LOGITS.0` | DeepSeek | `complete` | Consume transformer-owned normalized hidden state and apply the complete output-head projection to produce vocabulary logits with numeric proof. | V010.RUNTIME.DEEPSEEK.DECODE.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.LOGITS.0` | `evidence` | common | `complete` | logits requirement report. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.1` | `evidence` | common | `complete` | Transformer-owned normalized final hidden-state input contract. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.2` | `capability` | common | `complete` | Admit transformer-normalized prefill/decode hidden input without re-executing final norm. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.3` | `evidence` | common | `complete` | output-head tensor mapping. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.4` | `capability` | common | `complete` | Exact model-lifetime output-head host/device residency with stable warm reuse. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.5` | `capability` | common | `complete` | Direct encoded CPU/CUDA projection across every vocabulary row. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.6` | `evidence` | common | `complete` | logits buffer allocation. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.7` | `evidence` | common | `complete` | Complete F32 logits rows publish finite counts and exact numeric ranges. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.8` | `evidence` | common | `complete` | logits checksum report. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.9` | `evidence` | common | `complete` | top-k diagnostics. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.10` | `subtask` | common | `planned` | logprob diagnostics. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.11` | `subtask` | common | `planned` | sharded output-head plan. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.12` | `subtask` | common | `planned` | staged/SSD output-head plan. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.13` | `capability` | common | `complete` | Transactional row publication, cancellation, failure, cleanup, and partial progress. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.14` | `evidence` | common | `complete` | Field-wise plan, source, residency, row, backend, and aggregate evidence identities. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.15` | `evidence` | common | `complete` | Complete-vocabulary independent reference and CPU/CUDA numerical conformance. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.LOGITS.16` | `capability` | common | `complete` | The v0.1.0 logits gate passes for final-prefill and repeated decode sources. | V010.RUNTIME.DEEPSEEK.LOGITS.0 |
| `V010.RUNTIME.DEEPSEEK.LOGITS.SAMPLING.0` | `migration` | DeepSeek/common | `superseded` | Attempted recovery milestone split so output-head/logits and sampling retain distinct owners. | V010.RUNTIME.DEEPSEEK.LOGITS.0 + V010.RUNTIME.SAMPLING.0 |


### 9.18 TRACK.SAMPLING

**Owner:** Deterministic and stochastic token selection over real vocabulary logits, seeding, reproducibility, validation, and refusal.

**Current truth:** complete real vocabulary logits feed one family-neutral fixed-workspace host sampler with canonical greedy and filter-order-v2 stochastic selection. Independent post-closure review reopened this same milestone; compensated normalization, pre-entropy zero-mass compaction, atomic close/drain, complete evidence authentication, valid-prefix logits admission, and sanitizer aggregate coverage reclosed all six findings before tokenizer activation

**Ledger summary:** 15 recovered IDs; 1 first-class milestones; 15 complete support rows; 0 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.SAMPLING.0` | common | `complete` | Select token IDs from complete real vocabulary logits through numerically robust canonical greedy and explicitly seeded stochastic policies with transactional RNG, lifecycle, and evidence publication. | V010.RUNTIME.DEEPSEEK.LOGITS.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.SAMPLE.0` | `evidence` | common | `complete` | sampling requirement report. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.1` | `capability` | common | `complete` | Greedy scans the complete logits row and resolves exact maximum ties to the lowest token ID. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.2` | `evidence` | common | `complete` | Selected-token and execution identity v2 bind every authoritative result field, source, policy, probability, token ID, and RNG transition. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.3` | `evidence` | common | `complete` | Candidate-set evidence binds the raw-logits digest, policy, filter order, and final ordered survivors. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.4` | `evidence` | common | `complete` | Finite positive temperature and Neumaier-compensated double-precision normalization admit full-vocabulary uniform and extreme distributions. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.5` | `capability` | common | `complete` | Deterministic probability-descending/token-ascending top-k filters and renormalizes. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.6` | `capability` | common | `complete` | Deterministic nucleus top-p retains the threshold-crossing candidate and renormalizes. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.7` | `capability` | common | `complete` | Inclusive relative-to-maximum min-p filtering preserves at least one candidate. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.8` | `capability` | common | `complete` | Locally typical filtering consumes only positive mass after canonical underflow compaction, includes the threshold crossing, and renormalizes. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.9` | `capability` | common | `complete` | Explicitly seeded PCG-XSH-RR 64/32 advances once per published stochastic sample. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.10` | `evidence` | common | `complete` | Fixed seed, policy, logits, and call order reproduce the same selected-token sequence. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.11` | `evidence` | common | `complete` | Atomic close admission drains active use; failure, cancellation, concurrency refusal, and cleanup preserve logits and uncommitted RNG state. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.12` | `evidence` | common | `complete` | Bounded source, candidate, selected-token, RNG, per-row, and aggregate identity v2 authenticates every authoritative sampling field. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.13` | `evidence` | common | `complete` | Independent reference, CPU/CUDA-origin rows, complete mutation sensitivity, lifecycle stress, and ASan/UBSan aggregate execution close regression evidence. | V010.RUNTIME.SAMPLING.0 |
| `V010.SAMPLE.14` | `capability` | common | `complete` | The repaired v0.1.0 gate consumes all 129,280 real logits, including uniform and underflow cases, without appending or decoding selections. | V010.RUNTIME.SAMPLING.0 |


### 9.19 TRACK.TOKENIZER

**Owner:** Exact tokenizer loading, prompt encoding, templates, special/EOS/stop policy, append boundary, detokenization, and failure behavior.

**Current truth:** the selected admitted GGUF reconstructs one immutable runtime-bound ByteLevel-BPE plan with exact 129,280-token vocabulary, 127,741 ranked merges, 1,283 added tokens, byte-level pre-tokenizer/decoder, special/EOS/PAD facts, and an authenticated absence of tokenizer-config BOS/EOS insertion. Model-lifetime lookup indexes serve explicit-length UTF-8 encoding and bounded DeepSeek message rendering. Batch and incremental decode, split-UTF-8 buffering, EOS/stop classification, and a generation-local token append directory publish field-wise identities without mutating model/session/KV state

**Ledger summary:** 13 recovered IDs; 1 first-class milestone; 13 complete support rows; 0 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.TOKENIZER.0` | DeepSeek | `complete` | Reconstruct the exact tokenizer from admitted GGUF metadata, encode text and bounded prompts, classify specials/EOS, and decode token IDs through batch and incremental lifecycles. | V010.RUNTIME.SAMPLING.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.TOKENIZER.0` | `evidence` | common | `complete` | Tokenizer requirements are reconciled against the exact selected artifact and pinned reference. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.1` | `capability` | common | `complete` | GGUF tokenizer JSON/config metadata seals the immutable typed plan and bounded lookup indexes once per model. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.2` | `evidence` | common | `complete` | Explicit-length canonical U32 token-ID input remains the transformer handoff. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.3` | `capability` | common | `complete` | Encoding publishes one complete ordered token sequence and field-wise input/token/encoding identities or nothing. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.4` | `evidence` | common | `complete` | Exact added/special token IDs, matching policy, visibility, BOS/EOS/PAD, and absent UNK facts are typed. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.5` | `capability` | common | `complete` | EOS ID 1 is classified without applying a generation-loop decision. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.6` | `capability` | common | `complete` | The immutable stop-token set and suppression facts are exposed without owning stop-loop composition. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.7` | `capability` | DeepSeek | `complete` | The authenticated bounded DeepSeek chat/thinking prompt policy renders ordered system/user/assistant/tool-result messages and generation prompts. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.8` | `capability` | common | `complete` | Exact batch and incremental ByteLevel decode preserve incomplete UTF-8 and suppress typed control tokens transactionally. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.9` | `evidence` | common | `complete` | Capacity, UTF-8, token-ID, cancellation, close/entry concurrency, failure rollback, and cleanup refusals are covered. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.10` | `evidence` | common | `complete` | Plan, input, prompt, encoding, decoding, append-state, and runtime-binding identities authenticate published facts. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.11` | `evidence` | DeepSeek | `complete` | Pinned tokenizers-0.20.3 parity, Unicode/added-token cases, sampled-ID decoding, deterministic reuse, mutation/refusal, and sanitizer regression are admitted. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |
| `V010.TOKENIZER.12` | `capability` | DeepSeek | `complete` | The v0.1 tokenizer gate consumes only admitted GGUF material and leaves model state and generation capability unchanged. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 |


### 9.20 TRACK.GENERATION

**Owner:** Tokenizer/prefill/KV/decode/logits/sampling composition, append, stop, cancellation, partial output, trace, cleanup, and autoregression.

**Current truth:** the family-neutral generation owner executes exact text/message encoding, bounded prompt prefill, first-token logits/sampling, unchanged sampled-token decode feedback, typed terminal/context stops, incremental text, partial progress, and atomic close/drain over one model/session plane. CPU and mixed CUDA-model/host-tokenizer-sampling paths are admitted; the polished top-level CLI remains separate

**Ledger summary:** 35 recovered IDs; 1 first-class milestone; 34 complete support rows; 1 open support row; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.DEEPSEEK.GENERATION.0` | DeepSeek | `complete` | Exact prompt input produces multiple real sampled tokens, each ordinary token commits the same ID through decode and publishes incremental text with typed stop and partial progress. | V010.RUNTIME.DEEPSEEK.TOKENIZER.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.GEN.0` | `evidence` | common | `complete` | generation requirement report. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.1` | `evidence` | common | `complete` | generation state ownership. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.2` | `evidence` | common | `complete` | generation option parser. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.3` | `evidence` | common | `complete` | prefill to decode composition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.4` | `evidence` | common | `complete` | decode to logits composition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.5` | `evidence` | common | `complete` | logits to sample composition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.6` | `evidence` | common | `complete` | sample to append composition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.7` | `evidence` | common | `complete` | token append. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.8` | `capability` | common | `complete` | Context capacity is checked before the next sample and stops without a false state transition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.9` | `capability` | common | `complete` | EOS sampling is recorded and suppressed without decode or KV advancement. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.10` | `capability` | common | `complete` | Tokenizer and request-admitted stop IDs terminate before decode with exact terminal evidence. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.11` | `evidence` | common | `complete` | max-new-tokens policy. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.12` | `evidence` | common | `complete` | generation checksum. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.13` | `evidence` | common | `complete` | generation trace. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.14` | `evidence` | common | `complete` | failure/cancel safe points. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.15` | `evidence` | common | `complete` | cleanup/release. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.16` | `capability` | common | `complete` | `yvex graph transformer generate` retains the finite direct production-runtime proof in the guarded offline command lane. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.17` | `evidence` | common | `complete` | Bounded diagnostic generation smoke; not model-backed autoregressive generation. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.18` | `evidence` | common | `complete` | Focused, operator, CPU/CUDA live, manual-composition, mutation, lifecycle, sanitizer, and lower-owner regressions cover generation. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.GEN.19` | `capability` | DeepSeek | `complete` | Exact prompt text produces bounded real autoregressive text with no teacher-forced post-prompt token. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.0` | `capability` | common | `complete` | Typed trace policy distinguishes none, summary, stages, and full evidence without changing the production execution path. | V010.RUNTIME.1 |
| `V010.TRACE.1` | `evidence` | common | `complete` | token trace. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.2` | `evidence` | common | `complete` | Semantic, executable, launch-graph, graph-exec, stage, and mode identities are exposed through bounded runtime graph evidence. | V010.RUNTIME.1 |
| `V010.TRACE.3` | `subtask` | common | `planned` | tensor role trace. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.4` | `evidence` | common | `complete` | Runtime evidence reports resident bindings, encoded bytes, host/device ownership, upload/reuse counters, and invalidation generation. | V010.RUNTIME.1 |
| `V010.TRACE.5` | `evidence` | common | `complete` | Prompt identity, token count, chunk count, and final-prefill boundary are published. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.6` | `evidence` | common | `complete` | Each committed token publishes position, persistent generation, and state digest. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.7` | `evidence` | common | `complete` | Each ordinary token binds sampled ID, identical decode input, decode execution identity, and position transition. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.8` | `evidence` | common | `complete` | Every sample binds its source complete-logits identity and aggregate projection count. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.9` | `evidence` | common | `complete` | Sampling result and RNG transition identities bind every selected token. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.10` | `evidence` | common | `complete` | generation trace. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.11` | `evidence` | common | `complete` | cleanup/failure trace. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.12` | `evidence` | common | `complete` | Generation evidence is bounded and exposes no raw logits, hidden, KV, probability, or tensor dumps. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |
| `V010.TRACE.13` | `evidence` | common | `complete` | Normal, audit, JSON, and CSV projections serialize the same typed bounded runtime evidence without exposing test-oracle state. | V010.RUNTIME.1 |
| `V010.TRACE.14` | `capability` | common | `complete` | Field-wise plan, token-step, generated-token/text, stop, state, RNG, and execution identities close the generation trace gate. | V010.RUNTIME.DEEPSEEK.GENERATION.0 |


### 9.21 TRACK.OPERATOR

**Owner:** Public and offline CLI grammar, typed input, dispatch, rendering, refusal, control-plane integration, topology guards, and operator acceptance.

**Current truth:** `yvex` is the sole public command entrypoint. Its runtime lane remains protocol-only while its guarded offline lane owns the absorbed engineering operations; the separate developer executable is gone. The complete identity-bound audit freezes every current route, flag, operation, automation/configuration input, overlap, finding, workflow, and proposed disposition. The combined command-console plan is superseded before implementation: canonical operations and projections close first, then the mature daemon-backed REPL consumes them

**Ledger summary:** 81 recovered IDs; 8 first-class milestones; 37 complete support rows; 40 open support rows; 2 superseded/deferred support rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.CLI.GRAPH.0` | DeepSeek + common operator | `complete` | Expose production DeepSeek attention through the main YVEX binary with CPU, CUDA, quick/full canonical probes, typed structured refusal, and explicit non-generation semantics. | V010.GRAPH.DEEPSEEK.ATTENTION.0 | recovered/promoted |
| `V010.CLI.DEEPSEEK.GENERATE.0` | DeepSeek | `superseded` | Fresh-only top-level generation UX was superseded before implementation by the long-lived runtime/client product boundary. | V010.RUNTIME.CLIENT.REFOUNDATION.0 | current/superseded |
| `V010.OPERATOR.SURFACE.AUDIT.0` | product + engineering operations | `complete` | Froze the authoritative two-executable command, flag, operation, automation/configuration, taxonomy, registry-input, and migration inventory with zero behavior change and zero unmatched completeness gates. | V010.PRODUCT.SURFACE.REALIGNMENT.0 | current |
| `V010.OPERATOR.RUNTIME.CONSOLE.0` | common console + DeepSeek first vertical | `superseded` | The earlier standalone console plan is replaced by the ordered command-architecture and mature-REPL contracts. | V010.OPERATOR.REPL.CONSOLE.0 | current/superseded |
| `V010.OPERATOR.COMMAND.CONSOLE.0` | product commands + runtime console | `superseded` | The combined plan is split before implementation so operation/protocol authority closes before terminal presentation. | V010.OPERATOR.COMMAND.ARCHITECTURE.0; V010.OPERATOR.REPL.CONSOLE.0 | current/superseded |
| `V010.OPERATOR.COMMAND.ARCHITECTURE.0` | product + engineering operations | `blocked` | Establish one versioned operation source and checked generated descriptors for commands, flags, defaults, validation, help, discovery, projections, and semantic repairs. | V010.PROJECT.CONTROL.PUBLIC.0 | current |
| `V010.OPERATOR.REPL.CONSOLE.0` | daemon-backed terminal + DeepSeek first vertical | `blocked` | Build the mature linear runtime console, semantic progress, explicitly admitted reasoning projection, final metrics, watch, trace, and terminal lifecycle entirely over the canonical operation architecture. | V010.OPERATOR.COMMAND.ARCHITECTURE.0 | current |
| `TOPOLOGY.CELL.CLOSURE.0` | common | `planned` | Close residual mixed ownership only where concrete behavior still crosses domain/report/input/command/render/write boundaries. | V010.DOCS.ARCHITECTURE.0; supporting, not a product-stage promotion | recovered/promoted |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `SPINE.OUTPUT.UX.CONTRACT.0` | `evidence` | common | `complete` | Define CLI output UX contract and diagnostic demotion plan. | V010.PROJECT.RECOVERY.1 |
| `CLI.ARCH.AUDIT.0` | `evidence` | common | `complete` | Inventory print/output pressure and porcelain/plumbing doctrine. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `SPINE.CLI.REBASE.1` | `capability` | common | `complete` | Rebase Operator CLI track after V010.CLI.26 grammar work. | V010.PROJECT.RECOVERY.1 |
| `TOPOLOGY.FS.0` | `capability` | common | `complete` | Move C implementation under src modules and quarantine model-target CLI command surface. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.SOURCE.CONTRACT.0` | `capability` | common | `complete` | Add source file and function contract guardrails for module ownership. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CLI.PRINT.ALL.0` | `capability` | common | `complete` | Move production operator printing, help, usage, renderers, and CLI metadata out of domain modules into src/cli. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.DOMAIN.RESTORE.0` | `capability` | common | `complete` | Restore domain implementation files after invalid CLI command-file displacement. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.SOURCE.0` | `capability` | common | `complete` | Extract source into domain/report/input/command/render/write cell. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.GENERATION.0` | `capability` | common | `complete` | Extract generate into domain/report/input/command/render/trace cell. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.KV.0` | `capability` | common | `complete` | Extract KV into domain/report/input/command/render cell. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.SAMPLING.0` | `capability` | common | `complete` | Extract sampling into domain/report/input/command/render cell. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.GRAPH.0` | `capability` | common | `complete` | Extract graph into domain/report/input/command/render cell and separate graph facts from operator output. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_TARGET.0` | `capability` | common | `complete` | Extract model-target into catalog/report/input/command/render cell and remove target facts from CLI command adapter. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_TARGET.1` | `capability` | common | `complete` | Decompose model-target report monolith into owned target catalog, decision, candidate, map, qtype, and sidecar modules. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_TARGET.2` | `capability` | common | `complete` | Dissolve model-target internal compatibility backend into specialized ownership modules. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_TARGET.3` | `capability` | common | `complete` | Remove model-target runner, CLI-shaped report request, and text-buffer report API. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.CLOSURE.1` | `capability` | common | `complete` | Audit remaining topology residue after source, generation, KV, sampling, graph, and model-target cell extraction. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_ARTIFACTS.0` | `capability` | common | `complete` | Extract model-artifacts monolith into registry/ref/gate/report/input/command/render/write ownership. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_ARTIFACTS.1` | `capability` | common | `complete` | Split transitional model-artifacts CLI command surface into thin adapter and CLI-only surface ownership. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_ARTIFACTS.2` | `capability` | common | `complete` | Decompose transitional model-artifacts CLI surface into command-family surface owners and remove the 14k-line surface monolith. | TOPOLOGY.CELL.CLOSURE.0 |
| `TOPOLOGY.CELL.MODEL_ARTIFACTS.3` | `capability` | common | `complete` | Move model-artifacts surface output and parser residue into input/render owners. | TOPOLOGY.CELL.CLOSURE.0 |
| `SPINE.REBASE.GGUF.0` | `capability` | common | `complete` | Normalize active spine around GGUF/artifact ABI, writer, roundtrip, materialization, and runtime descriptor rows. | V010.PROJECT.RECOVERY.1 |
| `SPINE.SYSTEM.TARGET.0` | `capability` | common | `complete` | Codify target filesystem map and install GGUF/artifact/runtime descriptor/graph/backend owner files. | V010.PROJECT.RECOVERY.1 |
| `V010.PATHS.0` | `evidence` | common | `complete` | operator root layout report. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.1` | `evidence` | common | `complete` | source/artifact/reference/report/cache separation. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.2` | `capability` | common | `complete` | registry path layout. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.3` | `evidence` | common | `complete` | report output layout. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.4` | `subtask` | common | `planned` | runtime cache layout. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.5` | `evidence` | common | `complete` | artifact hygiene report. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.6` | `capability` | common | `complete` | path override precedence. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.7` | `capability` | common | `complete` | missing path/refusal behavior. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.PATHS.8` | `capability` | common | `complete` | v0.1.0 path acceptance gate. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.0` | `subtask` | common | `planned` | command inventory refresh. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.1` | `subtask` | common | `planned` | help layout refresh. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.2` | `subtask` | common | `planned` | normal path first policy. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.3` | `subtask` | common | `planned` | advanced diagnostic flags policy. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.12` | `subtask` | common | `planned` | refusal wording audit. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.13` | `subtask` | common | `planned` | structured output mode. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.15` | `subtask` | common | `planned` | command proof transcript. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.16` | `subtask` | common | `planned` | v0.1.0 CLI gate. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.17` | `capability` | common | `complete` | normal output contract and layout baseline. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.18` | `evidence` | common | `complete` | diagnostic output demotion. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.19` | `evidence` | common | `complete` | compact report/table output. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.20` | `subtask` | common | `planned` | raw/plumbing JSON foundation. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.21` | `subtask` | common | `planned` | metric output surface. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.22` | `subtask` | common | `planned` | audit output surface. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.23` | `subtask` | common | `planned` | quiet/no-color/non-TTY terminal policy. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.24` | `capability` | common | `complete` | hardcoded print reduction pass. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.25` | `capability` | common | `complete` | renderer ownership foundation. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.26` | `capability` | common | `complete` | base CLI grammar and command catalog. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.27` | `subtask` | common | `planned` | base status and refusal grammar. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.28` | `subtask` | common | `planned` | error/log/diagnostic surface split. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.MODELS.0` | `subtask` | common | `planned` | models namespace grammar. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.MODELS.1` | `subtask` | common | `planned` | models list/current/status porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.MODELS.2` | `subtask` | common | `planned` | models prepare/check porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.MODELS.3` | `subtask` | common | `planned` | models download/control porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.MODELS.4` | `subtask` | common | `planned` | models artifacts porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.TARGET.0` | `subtask` | common | `planned` | model-target namespace grammar. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.TARGET.1` | `subtask` | common | `planned` | model-target inspect/list porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.TARGET.2` | `subtask` | common | `planned` | model-target tensor-map/missing-roles/gate porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.TARGET.3` | `subtask` | common | `planned` | model-target quant-policy porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.SOURCE.0` | `subtask` | common | `planned` | source-manifest/native-weights porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.ACCOUNTS.0` | `subtask` | common | `planned` | accounts/provider porcelain. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.PATHS.0` | `subtask` | common | `planned` | paths porcelain finalization. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.RUNTIME.0` | `subtask` | common | `planned` | runtime diagnostic command grammar. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.CLI.GENERATE.0` | `migration` | common | `superseded` | Direct diagnostic generation porcelain was replaced by daemon-backed `yvex run`; direct proof remains developer/test-only. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.CLI.CHAT.0` | `migration` | common | `superseded` | The former fresh-only chat plan was replaced by daemon-backed retained-session `yvex chat`. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.DOCTOR.0` | `subtask` | common | `planned` | doctor command scope. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.1` | `subtask` | common | `planned` | environment checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.2` | `subtask` | common | `planned` | build/backend checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.3` | `subtask` | CUDA / DGX Spark | `planned` | CUDA checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.4` | `subtask` | common | `planned` | artifact checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.5` | `subtask` | common | `planned` | registry checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.6` | `subtask` | common | `planned` | model target checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.7` | `subtask` | common | `planned` | graph checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.8` | `subtask` | common | `planned` | runtime checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.9` | `subtask` | common | `planned` | generation readiness checks. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.10` | `subtask` | common | `planned` | common failure cookbook. | V010.CLI.DEEPSEEK.GENERATE.0 |
| `V010.DOCTOR.11` | `subtask` | common | `planned` | v0.1.0 doctor gate. | V010.CLI.DEEPSEEK.GENERATE.0 |


### 9.22 TRACK.SERVE

**Owner:** Long-lived local runtime hosting, model-worker coordination, exact session/KV authority, streaming, typed telemetry, local and application protocols, product-client process topology, and runtime administration.

**Current truth:** one local-only `yvexd` process owns a process-resident model, bounded worker queue, exact server sessions, multi-turn KV reuse, committed-fragment streaming, typed telemetry, metrics, raw/engine terminals, the private Unix listener, the loopback OpenAI listener, and graceful shutdown. Protocol v3 carries bounded provider messages, tools, response formats, stop strings, usage, finish classes, correlation, and integrated-listener status. The OpenAI adapter remains source-separated and uses the existing protocol/session/worker authority. Chat Completions, Responses, SSE, function-tool continuation, official SDK clients, and unchanged Bet-tennis provider syntax against the adapter test host are complete; the combined unchanged Bet-tennis -> adapter -> real resident DeepSeek proof is reopened and consumed by the measured optimization continuation, while public and remote serving remain deferred

**Ledger summary:** 12 recovered IDs; 2 complete first-class milestones; 11 complete support rows; 2 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RUNTIME.CLIENT.REFOUNDATION.0` | common host + DeepSeek first session vertical | `complete` | Open one runtime model for the daemon lifetime, own exact multi-turn KV sessions and turn streaming through one local protocol and typed telemetry authority, establish separable client/engineering lanes, and remove the old public CLI completely. | V010.COMPILATION.PHYSICAL.VARIANT.1 | current |
| `V010.SERVE.OPENAI.COMPAT.0` | common provider + DeepSeek first tool vertical | `complete` | Carries bounded typed application messages, tools, structured output, stop, usage, and streaming through the local protocol and a source-separated loopback OpenAI-compatible adapter, with official SDK and unchanged external-consumer acceptance. | V010.DOCS.README.PRODUCT.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.SERVE.0` | `evidence` | local runtime | `complete` | One ownership map separates daemon host, model worker, protocol, server session, runtime-client lane, offline engineering lane, OpenAI adapter, and typed telemetry authority. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.1` | `capability` | local runtime | `complete` | Daemon state derives from one authoritative runtime model lifetime with exact open/close counters. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.2` | `capability` | local runtime | `complete` | The selected logical model, artifact, physical variant, binding, backend, and context facts are exposed without client-side model admission. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.3` | `capability` | local runtime | `complete` | Protocol-v3 local status, readiness, session, request, provider, stream, event, cancellation, listener, and shutdown messages are bounded and typed. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.4` | `capability` | local runtime | `complete` | Server-owned turns reuse exact session KV and prefill only the canonical new prompt suffix. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.5` | `capability` | local runtime | `complete` | Only internally committed and detokenized fragments stream through the local protocol. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.6` | `capability` | local runtime | `complete` | Request, disconnect, SIGINT, queue, session, and graceful-shutdown cancellation preserve typed partial progress. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.7` | `capability` | application provider | `complete` | One provider-neutral typed request/result owner carries ordered messages, sampling, response formats, tools, finish classes, usage, cancellation, and field-wise identity without transport syntax. | V010.SERVE.OPENAI.COMPAT.0 |
| `V010.SERVE.8` | `capability` | application provider | `complete` | YVEX OpenAI Compatibility Profile v1 maps bounded Models, Chat Completions, Responses, SSE, stop, JSON-object, function tools, usage, and errors through the source-separated adapter without direct engine dependency. | V010.SERVE.OPENAI.COMPAT.0 |
| `V010.SERVE.9` | `subtask` | post-v0.1 | `planned` | Anthropic compatibility after generation. | future scope decision |
| `V010.SERVE.10` | `capability` | local runtime | `complete` | One bounded typed event authority feeds raw JSONL, compact engine views, metrics, client projections, privacy defaults, and coalesced backpressure. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.11` | `evidence` | local runtime | `complete` | Product binaries prove one hosted model, exact multi-turn sessions, REPL/one-shot streaming, route-level runtime-client separation, packaging, and total incompatible CLI cutover on GB10. | V010.RUNTIME.CLIENT.REFOUNDATION.0 |
| `V010.SERVE.OPENAI.BETTENNIS.REAL.1` | `evidence` | Bet-tennis + real DeepSeek | `reopened` | Prove the unchanged Bet-tennis provider through the integrated OpenAI listener, protocol v3, the shared `yvexd` session/worker authority, and the real resident DeepSeek model for health, model discovery, chat, structured JSON, tools, SSE, cancellation, and daemon survival; the prior synthetic-host proof is retained but narrower. | V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 |


### 9.23 TRACK.EVAL

**Owner:** Complete-model behavior, quality, regression, tokenizer, context, refusal, and capability evaluation after real generation exists.

**Current truth:** blocked until product-surface realignment, command/binary audit, public project-control refoundation, canonical command architecture, mature REPL console, and the measured GB10 optimization continuation close; software testing, numerical conformance, runtime qualification, operational metrics, and component profiling do not satisfy model behavior or quality evaluation

**Ledger summary:** 15 recovered IDs; 1 first-class milestones; 0 complete support rows; 15 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.EVAL.DEEPSEEK.0` | DeepSeek | `blocked` | Run repeatable correctness, tokenizer, regression, long-context, refusal, and release-path generation evaluations. | V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.EVAL.0` | `subtask` | common | `planned` | eval harness structure. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.1` | `subtask` | common | `planned` | fixture graph eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.2` | `subtask` | common | `planned` | primitive eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.3` | `subtask` | common | `planned` | selected partial graph eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.4` | `subtask` | common | `planned` | full-runtime-candidate graph eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.5` | `subtask` | common | `planned` | prefill eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.6` | `subtask` | common | `planned` | KV eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.7` | `subtask` | common | `planned` | decode eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.8` | `subtask` | common | `planned` | logits eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.9` | `subtask` | common | `planned` | sampling eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.10` | `subtask` | common | `planned` | generation smoke eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.11` | `subtask` | common | `planned` | tokenizer/stop eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.12` | `subtask` | common | `planned` | failure-path eval. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.13` | `subtask` | common | `planned` | capability eval plan. | V010.EVAL.DEEPSEEK.0 |
| `V010.EVAL.14` | `subtask` | common | `planned` | v0.1.0 eval gate. | V010.EVAL.DEEPSEEK.0 |


### 9.24 TRACK.BENCH

**Owner:** Reproducible component and complete-model performance measurements with machine, artifact, qtype, workload, timing, throughput, memory, regression, baseline, and visualization metadata.

**Current truth:** the common runtime implements schema-v5 `attention_component` cold/warm measurement, p50/p90/p95/p99 distributions, complete reproducibility identity, correctness/runtime preconditions, identity-bound file comparison, caller-owned regression thresholds, JSON/CSV projection, and deterministic external SVG charts; no repository-tracked result or release-path/full-model generation benchmark evidence is claimed

**Ledger summary:** 16 recovered IDs; 1 first-class milestone; 7 complete support rows; 9 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.BENCH.DEEPSEEK.0` | DeepSeek / DGX Spark | `not-measured` | Record reproducible artifact, qtype, prompt/context, prefill, decode, generation, timing, throughput, and memory evidence. | V010.EVAL.DEEPSEEK.0 | current |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.BENCH.0` | `capability` | common | `complete` | Schema-v5 attention-component benchmark records identity-bound cold/warm phases, first execution, latency distributions, counters, baselines, file comparison, and external SVG evidence. | V010.RUNTIME.1 |
| `V010.BENCH.1` | `subtask` | common | `planned` | machine profile record. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.2` | `evidence` | common | `complete` | Every bounded runtime benchmark binds artifact, materialization, model, runtime, graph, execution, residency, workspace, state-layout, kernel-bundle, machine, device, mode, class, geometry, trace, and scope identities. | V010.RUNTIME.1 |
| `V010.BENCH.3` | `subtask` | common | `planned` | qtype/context/backend record. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.4` | `evidence` | common | `complete` | Warmup/run counts, deterministic non-timing fields, p50/p90/p95/p99, dispersion, correctness/runtime/performance separation, compatible-baseline comparison, explicit identity-bound regression thresholds, and chart reproduction are explicit. | V010.RUNTIME.1 |
| `V010.BENCH.5` | `subtask` | common | `planned` | prefill benchmark. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.6` | `subtask` | common | `planned` | decode benchmark. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.7` | `subtask` | common | `planned` | generation benchmark. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.8` | `subtask` | common | `planned` | memory pressure benchmark. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.9` | `subtask` | common | `planned` | server benchmark. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.10` | `subtask` | DeepSeek | `planned` | DeepSeek benchmark only after DeepSeek generation. | V010.BENCH.DEEPSEEK.0 |
| `V010.BENCH.11` | `subtask` | common | `planned` | v0.1.0 benchmark decision gate. | V010.BENCH.DEEPSEEK.0 |
| `V010.PROFILE.0` | `evidence` | common | `complete` | Bounded attention-runtime profiles separate cold model preparation from eager, piecewise-graph, and full-graph warm execution. | V010.RUNTIME.1 |
| `V010.PROFILE.1` | `evidence` | common | `complete` | Profiles report resident, workspace, state, staging, peak host, and peak device memory with steady-state allocation counters. | V010.RUNTIME.1 |
| `V010.PROFILE.2` | `evidence` | common | `complete` | Profiles distinguish artifact hashing, artifact reads, warm weight reads, H2D, D2H, and resident encoded bytes. | V010.RUNTIME.1 |
| `V010.PROFILE.3` | `evidence` | common | `complete` | Backend profiles identify device/driver/build, requested and selected modes, kernel/graph launches, captures, replays, and fallback refusals. | V010.RUNTIME.1 |


### 9.25 TRACK.RELEASE

**Owner:** Validation, artifact guardrail, claim audit, operator transcript, packaging, versioning, release record, and tag.

**Current truth:** blocked

**Ledger summary:** 42 recovered IDs; 1 first-class milestones; 4 complete support rows; 37 open support rows; 0 superseded/deferred rows.

#### First-Class Milestones

| Milestone | Scope | State | Conclusive after-state | Depends on | Origin |
| --- | --- | --- | --- | --- | --- |
| `V010.RELEASE.0` | DeepSeek v0.1.0 | `blocked` | Close every source, architecture, mapping, artifact, materialization, backend, transformer, generation, evaluation, benchmark, validation, claim, operator, packaging, and version gate. | V010.BENCH.DEEPSEEK.0 | recovered/promoted |

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.VERSION.0` | `subtask` | project/release | `planned` | version string policy. | V010.RELEASE.0 |
| `V010.VERSION.1` | `subtask` | project/release | `planned` | v0.1.0 version bump. | V010.RELEASE.0 |
| `V010.PACKAGE.0` | `subtask` | project/release | `planned` | binary packaging policy. | V010.RELEASE.0 |
| `V010.PACKAGE.1` | `subtask` | project/release | `planned` | release build artifact policy. | V010.RELEASE.0 |
| `V010.PACKAGE.2` | `subtask` | project/release | `planned` | no model artifact packaging rule. | V010.RELEASE.0 |
| `V010.RELEASE.1` | `subtask` | project/release | `planned` | target lock. | V010.RELEASE.0 |
| `V010.RELEASE.2` | `subtask` | project/release | `planned` | command proof transcript. | V010.RELEASE.0 |
| `V010.RELEASE.3` | `subtask` | project/release | `planned` | failure proof transcript. | V010.RELEASE.0 |
| `V010.RELEASE.4` | `subtask` | project/release | `planned` | artifact guardrail transcript. | V010.RELEASE.0 |
| `V010.RELEASE.5` | `subtask` | project/release | `planned` | claim audit. | V010.RELEASE.0 |
| `V010.RELEASE.6` | `subtask` | project/release | `planned` | docs audit. | V010.RELEASE.0 |
| `V010.RELEASE.7` | `subtask` | project/release | `planned` | changelog/release notes. | V010.RELEASE.0 |
| `V010.RELEASE.8` | `subtask` | project/release | `planned` | tag readiness report. | V010.RELEASE.0 |
| `V010.RELEASE.9` | `subtask` | project/release | `planned` | v0.1.0 tag. | V010.RELEASE.0 |
| `V010.CI.0` | `subtask` | project/release | `planned` | CI/test matrix refresh. | V010.RELEASE.0 |
| `V010.CI.1` | `subtask` | project/release | `planned` | make check gate. | V010.RELEASE.0 |
| `V010.CI.2` | `subtask` | project/release | `planned` | make smoke gate. | V010.RELEASE.0 |
| `V010.CI.3` | `subtask` | CUDA / DGX Spark | `planned` | make check-cuda gate where available. | V010.RELEASE.0 |
| `V010.CI.4` | `capability` | project/release | `complete` | docs surface gate. | V010.RELEASE.0 |
| `V010.CI.5` | `capability` | project/release | `complete` | source layout gate. | V010.RELEASE.0 |
| `V010.CI.6` | `capability` | project/release | `complete` | code natural gate. | V010.RELEASE.0 |
| `V010.CI.7` | `capability` | project/release | `complete` | artifact guardrail. | V010.RELEASE.0 |
| `V010.CI.8` | `subtask` | project/release | `planned` | forbidden claim scan. | V010.RELEASE.0 |
| `V010.CI.9` | `subtask` | project/release | `planned` | public docs leak scan. | V010.RELEASE.0 |
| `V010.CI.10` | `subtask` | project/release | `planned` | command proof transcript gate. | V010.RELEASE.0 |
| `V010.CI.11` | `subtask` | project/release | `planned` | failure-path transcript gate. | V010.RELEASE.0 |
| `V010.CI.12` | `subtask` | project/release | `planned` | v0.1.0 CI acceptance gate. | V010.RELEASE.0 |
| `V010.DOCS.INTERNAL.0` | `subtask` | project/release | `planned` | internal v0.1.0 spine summary. | V010.RELEASE.0 |
| `V010.DOCS.RUNBOOK.0` | `subtask` | project/release | `planned` | operator v0.1.0 runbook. | V010.RELEASE.0 |
| `V010.DOCS.RUNBOOK.1` | `subtask` | project/release | `planned` | model-specific runbooks. | V010.RELEASE.0 |
| `V010.DOCS.API.0` | `subtask` | project/release | `planned` | API docs for implemented surface. | V010.RELEASE.0 |
| `V010.DOCS.CONTRACT.0` | `subtask` | project/release | `planned` | behavior contract update. | V010.RELEASE.0 |
| `V010.DOCS.README.0` | `subtask` | project/release | `planned` | README runtime thesis update. | V010.RELEASE.0 |
| `V010.DOCS.DIAGRAM.0` | `subtask` | project/release | `planned` | artifact-to-runtime diagram. | V010.RELEASE.0 |
| `V010.DOCS.DIAGRAM.1` | `subtask` | project/release | `planned` | runtime ladder diagram. | V010.RELEASE.0 |
| `V010.DOCS.DIAGRAM.2` | `subtask` | project/release | `planned` | evidence/benchmark diagram. | V010.RELEASE.0 |
| `V010.DOCS.DIAGRAM.3` | `subtask` | MoE/common | `planned` | dense vs MoE path diagram. | V010.RELEASE.0 |
| `V010.DOCS.DIAGRAM.4` | `subtask` | project/release | `planned` | storage/residency diagram. | V010.RELEASE.0 |
| `V010.DOCS.PUBLIC.0` | `subtask` | project/release | `planned` | public claim audit. | V010.RELEASE.0 |
| `V010.DOCS.PUBLIC.1` | `subtask` | project/release | `planned` | internal ID leak audit. | V010.RELEASE.0 |
| `V010.DOCS.PUBLIC.2` | `subtask` | project/release | `planned` | v0.1.0 docs acceptance gate. | V010.RELEASE.0 |


### 9.26 TRACK.POST010

**Owner:** Explicitly deferred portability, serving hardening, speculative execution, extra-family runtime promotion, and later product scope.

**Current truth:** deferred

**Ledger summary:** 23 recovered IDs; 0 first-class milestones; 0 complete support rows; 14 open support rows; 9 superseded/deferred rows.

#### First-Class Milestones

No first-class milestone is committed in the active v0.1.0 path. Retained rows remain subordinate or future scope.

#### Recovered And Subordinate Rows

| Wave | Rank | Scope | State | Exact retained outcome or requirement | Consumer or enclosing milestone |
| --- | --- | --- | --- | --- | --- |
| `V010.SPEC.0` | `subtask` | post-v0.1 | `planned` | speculative reference registry. | future scope decision |
| `V010.SPEC.1` | `subtask` | post-v0.1 | `planned` | DSpark reference record. | future scope decision |
| `V010.SPEC.2` | `subtask` | post-v0.1 | `planned` | DFlash/HyperDFlash reference record. | future scope decision |
| `V010.SPEC.3` | `subtask` | post-v0.1 | `planned` | draft source report. | future scope decision |
| `V010.SPEC.4` | `subtask` | post-v0.1 | `planned` | token verification semantics. | future scope decision |
| `V010.SPEC.5` | `subtask` | post-v0.1 | `planned` | accepted-prefix accounting. | future scope decision |
| `V010.SPEC.6` | `subtask` | post-v0.1 | `planned` | rejected-token behavior. | future scope decision |
| `V010.SPEC.7` | `subtask` | post-v0.1 | `planned` | KV rollback/reuse policy. | future scope decision |
| `V010.SPEC.8` | `subtask` | post-v0.1 | `planned` | dense speculative verification. | future scope decision |
| `V010.SPEC.9` | `subtask` | post-v0.1 | `planned` | MoE routing-aware verification report. | future scope decision |
| `V010.SPEC.10` | `subtask` | post-v0.1 | `planned` | MoE expert-budget verification. | future scope decision |
| `V010.SPEC.11` | `subtask` | post-v0.1 | `planned` | verification-cost utility report. | future scope decision |
| `V010.SPEC.12` | `subtask` | post-v0.1 | `planned` | DeepSeek speculative path. | future scope decision |
| `V010.SPEC.13` | `subtask` | post-v0.1 | `planned` | speculative benchmark. | future scope decision |
| `POST010.GLM.RUNTIME.0` | `future` | post-v0.1 | `deferred` | GLM runtime promotion path after v0.1.0. | future scope decision |
| `POST010.QWEN.METAL.0` | `future` | post-v0.1 | `deferred` | Qwen Metal runtime path after baseline release. | future scope decision |
| `POST010.ROCM.0` | `future` | post-v0.1 | `deferred` | ROCm/Strix Halo backend path after v0.1.0. | future scope decision |
| `POST010.STORAGE.GEN.0` | `future` | post-v0.1 | `deferred` | SSD-streamed generation exploration after baseline generation. | future scope decision |
| `POST010.SERVE.PUBLIC.0` | `future` | post-v0.1 | `deferred` | production serving surface hardening after v0.1.0. | future scope decision |
| `POST010.SPEC.0` | `future` | post-v0.1 | `deferred` | speculative acceleration program after baseline generation. | future scope decision |
| `POST010.BENCH.PUBLIC.0` | `future` | post-v0.1 | `deferred` | public benchmark table expansion after measured release-path generation evidence. | future scope decision |
| `POST010.EVAL.CAPABILITY.0` | `future` | post-v0.1 | `deferred` | broader capability eval suite after v0.1.0. | future scope decision |
| `POST010.DOCS.PUBLIC.0` | `future` | post-v0.1 | `deferred` | public evidence expansion after release-safe claims. | future scope decision |

## 10. Evidence Lanes

Current diagnostic, report, fixture, selected, CLI, topology, and audit work
stays visible because it contains real implementation and prevents repeated
work. Historical rows remain visible only as decommission provenance; they do
not imply that the retired code or ABI is still executable. Evidence rank
limits the claim either class may support.

| Evidence lane | Valid use | Non-claim |
| --- | --- | --- |
| Source/model reports | Inspect facts, exercise typed boundaries, test refusal, and seed complete architecture/map work | reports alone do not verify source payloads, mapping, artifacts, or runtime support |
| Source payload fixtures/live proof | Exercise digest authority, manifest publication, indexed ranges, exact bounded reads, budgets, cancellation, drift, transactional delivery, and complete DeepSeek mapping handoff | not conversion, quantization, artifact emission, materialization, runtime residency, or generation |
| Transformation IR fixtures/live plan | Exercise immutable DAG construction, operation/shape/dtype refusal, canonical identity, large fan-in, allocation rollback, exhaustive payload-range binding, and GGUF lowering equivalence | not byte transformation, precision selection, quantization, artifact emission, materialization, runtime execution, or generation |
| Physical-variant planning and quantization | Exercise policy-v2 conjunction/priority/conflict rules, total zero-read terminal resolution, exact size prediction, identity-bound imatrix coverage, canonical codecs/refusals, trusted bounded reads, transactional output, role/qtype numeric metrics, allocation/sink faults, and direct CPU/CUDA qtype compute | not automatic precision/size/Pareto optimization, model quality, release-profile selection, or arbitrary-family quantization support |
| Complete GGUF emission/roundtrip | Exercise immutable writer planning, exact tokenizer metadata, transactional terminal delivery, atomic publication, physical SHA-256 identity, native full-byte verification, pinned official-reader parsing, deterministic reserialization, and complete-artifact admission | not materialization, residency, runtime binding, transformer execution, generation, evaluation, benchmark, or a supported model artifact |
| GGUF fixtures/tensor proofs | Parser, geometry, range, writer-fragment, materialization, and primitive regression | not a complete or supported model artifact |
| Primitive/selected graph proofs | One operation, transfer, cleanup, tolerance, or bounded composition | not full attention, transformer execution, or generation |
| Common runtime attention, activation-prefill, and persistent-state plane | Exercise immutable runtime-binding admission, one process-lifetime model/session, resident attention weights, versioned ordered activation input, exact session-owned DeepSeek state, all-layer per-chunk atomic append/read, clear/reuse, CPU/CUDA residency, causal history consumption, whole-chunk/subchunk equivalence, tensor-file operator execution, phase/mode dispatch, eager/CUDA Graph execution, invalidation, cold/warm timing, baseline comparison, and deterministic external charts | does not by itself prove prompt/token embedding, token-local MoE, complete transformer execution, model decode, generation, evaluation, or full-model benchmark behavior |
| Token-local DeepSeek MoE | Exercise immutable MoE-plan admission, typed activation/token-ID input, hash and learned routing, deterministic top-k, selected routed-expert subviews, distinct shared experts, CPU/CUDA encoded qtype compute, exact combination, cancellation, cleanup, and direct operator execution | not attention/MoE composition, complete transformer execution, model prefill/decode, generation, evaluation, or full-model benchmark proof |
| Numeric-token transformer and repeated decode | Exercise selected embedding rows, 43 ordered attention/MoE blocks, final mHC/norm, explicit prefill/decode phase identity, one commit per teacher-forced step, causal prior-KV consumption, ordered hidden rows, typed partial progress, CPU/CUDA parity, and direct operator execution | not tokenizer-backed prompt input, token selection, sampling, autoregressive generation, evaluation, or full-model benchmark proof |
| Complete output-head logits | Exercise transformer-authenticated final-prefill and decode hidden admission, exact separate-head binding, shared model-lifetime residency, direct encoded full-vocabulary CPU/CUDA projection, independent numerical conformance, transactional row publication, warm reuse, and direct operator execution | not probabilities, token selection, sampling, tokenizer behavior, autoregressive generation, evaluation, or full-model benchmark proof |
| Real-logits sampling | Exercise completed-prefix admission, compensated softmax normalization, pre-entropy zero compaction, canonical filters, seeded transactional PCG, authenticated evidence, atomic close/drain, sanitizer coverage, independent reference, warm reuse, and direct operator execution | not token append, tokenizer semantics, EOS/stop policy, CUDA sampling, autoregressive feedback, generation, evaluation, or benchmark proof |
| Historical diagnostic runtime/generation state | Decommissioned lifecycle, refusal, cancellation, flat F32 KV, fixture logits/sampling, bounded token-loop, and cleanup evidence removed by the `V010.RUNTIME.1` cutover | historical provenance only; not current executable evidence and not model-backed prefill, KV, decode, logits, sampling, or generation |
| Operator/topology evidence | Discoverability, ownership enforcement, refusal propagation, and transcript regression | cannot create a lower domain capability |
| Internal fixtures | Deterministic vectors, corrupt files, synthetic tokens/logits, and allocation failures | not model quality, benchmark, or release proof |

Evidence may be removed only when the behavior itself is intentionally removed
and the owning milestone records the decision. Demotion from first-class status
is not authorization to delete code, tests, or family work.

### 10.1 Decommission Obligations

These are project-owned cutover obligations for retained and already
decommissioned residue. Availability is row-specific and is never presumed.
An undecommissioned surface remains at its current evidence rank only until
its consuming milestone removes, replaces, retains, or absorbs it and passes
the stated acceptance boundary. `V010.RUNTIME.1` is the completed cutover
consumer for the former diagnostic engine/session, flat F32 KV, fixture
generation, and generic run-report surface; those superseded owners and their
installed headers were intentionally removed rather than left available.

| Surface | Code-grounded locations | Required disposition | Consuming milestone and acceptance boundary |
| --- | --- | --- | --- |
| Selected embedding and segment commands | Remaining compatibility parsing/rendering is confined to `src/cli/input/graph.c`, `src/cli/render/graph.c`, `src/runtime/graph.c`, and `tests/cli/attention_graph.sh`; standalone partial/segment scripts are removed. | Absorb valid backend/reference comparisons into internal proofs; remove any surviving selected product aliases. | `V010.GRAPH.DEEPSEEK.TRANSFORMER.0`: full transformer tests own public graph proof and selected command discovery is gone. |
| Persistent KV | Diagnostic prefill/report/CLI and flat interchangeable K/V precursors are removed. The session-owned family-correct provider, CPU/CUDA residency, atomic model-state transaction, production read-after-write, clear/reuse, and operator exercise are admitted. Activation prefill, the complete transformer, and repeated decode consume this boundary directly and add no parallel cache. | Preserve this owner as the sole persistent state boundary; token-local MoE does not mutate it and repeated decode reuses the same session position, cancellation, and state contracts. | KV, activation prefill, MoE, transformer, and decode milestones are complete; output-head logits consume decode hidden rows without owning KV. |
| Diagnostic decode | The former summary-only command, production owner, and CLI tests are removed. | Preserve the replacement teacher-forced decode coordinator over the common runtime, transformer, and persistent KV owners; do not restore a fixture path. | `V010.RUNTIME.DEEPSEEK.DECODE.0`: complete; `yvex graph transformer decode` consumes the executable descriptor and attention-backed KV. |
| Fixture logits and sampling | Former production fixture owners and public CLI surfaces are removed. Real logits consume transformer-normalized hidden state and the exact output head through the runtime owner. | Preserve real logits; introduce sampling only over the complete admitted F32 vocabulary row. | `V010.RUNTIME.DEEPSEEK.LOGITS.0`: complete; `V010.RUNTIME.SAMPLING.0`: sampled IDs derive from full-vocabulary model logits. |
| Bounded diagnostic generation | Former diagnostic production/CLI owners and token-printing tests are removed; no fixture generation path was restored. | Preserve only the accepted tokenizer/prefill/KV/decode/logits/sample/append/stop composition owner. | `V010.RUNTIME.DEEPSEEK.GENERATION.0`: complete; multiple real sampled tokens are decoded and detokenized. |
| Selected-artifact support levels | `include/yvex/model.h`, `src/model/artifacts/gate.c`, registry/report owners, prepare renderers, model gate/registry tests | Complete-artifact admission now consumes one canonical typed result; retained bounded subsets remain tensor proof artifacts and cannot enter that path. | `V010.ARTIFACT.SUPPORT.CUTOVER.0`: complete; runtime support still requires materialization and every higher gate. |
| Report-only fullmodel surfaces | `src/cli/model_artifacts/fullmodel.c` and `tests/cli/fullmodel.sh` | Replace useful descriptor/materialization facts with typed full-target APIs; remove reports that only restate missing runtime behavior. | `V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0`: one typed executable descriptor boundary remains and obsolete report commands are gone. |
| Former flat CLI and split developer executable | Retained domain adapters dispatch through nested offline routes in the sole `yvex` executable; hosted runtime routes remain confined to the protocol-client lane. | Preserve hard refusal for old top-level commands, route-level runtime-client separation, and one finite offline proof lane; do not restore a developer executable or public `dev` namespace. | `V010.PRODUCT.SURFACE.REALIGNMENT.0`: complete role-based binary/process cutover. |

## 11. Release Gates

| Gate | Required executable evidence | Current state | Owning milestone |
| --- | --- | --- | --- |
| Project control | One authority, complete ranked ledger, unique IDs, calculated counts, one Active Next, no shadow spine | complete | `V010.PROJECT.RECOVERY.1` |
| Documentation architecture | Standard taxonomy, durable-content ownership, no duplicate roadmap, reference/paper map | complete | `V010.DOCS.ARCHITECTURE.0` |
| Public product documentation | Concise executable three-terminal and one-shot workflow, accepted binary/process topology, reviewed editable SVG architecture, valid navigation, and current non-claims | complete; documentation-only and capability-neutral | `V010.DOCS.README.PRODUCT.0` |
| Source | Exact local identity, revision, config, tokenizer, shard inventory, headers, and payload trust | complete | `V010.REBASE.DEEPSEEK.0` and `V010.SOURCE.PAYLOAD.STREAM.0` |
| Architecture | Typed execution-complete DeepSeek specification | complete | `V010.MODEL.ARCH.IR.0` |
| Tensor coverage | Complete required-role set with no unresolved role | complete | `V010.TENSOR.COVERAGE.DEEPSEEK.0` |
| Mapping | Complete source-role-GGUF map with transforms/layouts and ambiguity refusal | complete | `V010.MAP.GGUF.DEEPSEEK.0` |
| Transformation | Artifact-neutral plan binds every logical output to exact source contributions and payload-range identity before physical lowering | complete | `V010.MODEL.TRANSFORM.IR.0` |
| Qtype/quantization | Sealed policy-v2 resolution, baseline and candidate presets, canonical codecs/refusals, identity-bound imatrix coverage, independent decoding, numeric bounds, trusted bounded execution, and direct CPU/CUDA selected-qtype compute | complete for explicit policy-driven variants; automatic precision/target-size solving remains unsupported | `V010.QUANT.2`, `V010.QUANT.3`, and `V010.COMPILATION.PHYSICAL.VARIANT.1` |
| Artifact | Complete YVEX-produced GGUF from the exact physical-variant plan with native and pinned official-reader equivalence | complete for the Q8_0/Q2_K baseline and the IQ2_XXS/Q2_K candidate; release-profile selection remains evaluation/benchmark-owned | `V010.GGUF.WRITER.1` through `V010.GGUF.ROUNDTRIP.1` and `V010.COMPILATION.PHYSICAL.VARIANT.1` |
| Integrity/admission | Canonical global layout, complete physical identity, payload integrity, role coverage, and complete-artifact admission | complete | `V010.GGUF.LAYOUT.INTEGRITY.1` and `V010.ARTIFACT.SUPPORT.CUTOVER.0` |
| Materialization | Family-neutral materializer derives ranges, qtype geometry, expert subviews, and backend capability from each admitted variant | complete for baseline and mixed IQ2_XXS/Q2_K candidate cold access; runtime readiness additionally requires every encoded tensor in one authenticated, locked, model-lifetime host arena | `V010.ARTIFACT.MATERIALIZE.0`, `V010.ARTIFACT.MATERIALIZE.1`, and runtime milestones |
| Descriptor | Canonical common descriptor and execution-complete DeepSeek graph-input specialization | complete and consumed by attention, MoE, and the full transformer backbone | descriptor, attention, MoE, and transformer milestones |
| Runtime | Content-addressed binding, one process-resident authenticated model, server-owned mutable sessions, persistent state, exact prefix continuation, bounded worker execution, local protocol, typed telemetry, and reusable execution without compiler reconstruction | correctness-complete through hosted exact text encoding, prompt-suffix prefill, sampled-token feedback, repeated decode, complete logits, common-host sampling, stop, partial progress, streaming, detach/reconnect, reset, and graceful shutdown; startup and bounded warm-path profiling are admitted | runtime, KV, prefill, MoE, transformer, decode, logits, sampling, tokenizer, generation, runtime/client refoundation, and `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` |
| Full-model performance | Identity-bound cold/warm profiles, preserved golden behavior, measured movement/synchronization, reusable expert placement, device routing, grouped multi-token MoE prefill, suffix chunking, stable decode, and quantitative GB10 admission | partial: startup is under four minutes and profile attribution is available, but warm decode is 0.432 tok/s and fails the 5/10/20 tok/s gates; continuation is explicit | `V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` and `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| CUDA | Capability admission is fail-closed; selected embedding, complete DeepSeek attention, routed/shared MoE, residual mHC, final collapse/norm, and direct encoded output-head projection have GB10 proof with no CPU numerical fallback | complete eager backbone and full-vocabulary logits ready; transformer/logits CUDA Graph and optimized/fused output-head execution remain unsupported | attention/KV/prefill/MoE/transformer/logits milestones |
| Transformer | Numeric tokens execute selected embedding, 43 canonical attention/MoE blocks, deferred residuals, final mHC collapse, final RMSNorm, and atomic persistent-state publication | complete CPU/GB10 CUDA backbone and normalized hidden output, consumed by repeated decode and logits without duplicating final norm | transformer, decode, and logits milestones |
| Text generation | Exact tokenizer, output head, logits, sampling, repeated decode, stop, detokenization, session continuation, and committed-fragment streaming | complete bounded CPU and mixed CUDA-model/host-sampling-tokenizer generation runtime with daemon-backed one-shot and retained multi-turn clients; model quality is not claimed | logits/sampling/tokenizer/generation/runtime-client milestones |
| Operator | One truthful public command entrypoint invokes the hosted release path through a protocol-only runtime lane and finite engineering operations through a guarded offline lane | complete for the two-executable role topology and frozen surface audit; command/protocol authority and mature terminal presentation remain separate ordered successors | `V010.RUNTIME.CLIENT.REFOUNDATION.0`, `V010.PRODUCT.SURFACE.REALIGNMENT.0`, and `V010.OPERATOR.SURFACE.AUDIT.0` |
| Application provider | Provider-neutral typed requests traverse protocol v3 through the source-separated in-daemon OpenAI adapter with bounded JSON/SSE, tools, official SDKs, and unchanged application-consumer proof | implementation complete for YVEX OpenAI Compatibility Profile v1 with one `yvexd` model/session/worker authority; the unchanged Bet-tennis-to-real-DeepSeek composition evidence is reopened, while full OpenAI API and remote serving are not claimed | `V010.SERVE.OPENAI.COMPAT.0`, `V010.PRODUCT.SURFACE.REALIGNMENT.0`, and `V010.SERVE.OPENAI.BETTENNIS.REAL.1` |
| Evaluation | Repeatable release-path quality, regression, context, and refusal cases | blocked behind active public project control, canonical command architecture, mature REPL console, and measured GB10 optimization; no behavior or quality result exists yet | `V010.EVAL.DEEPSEEK.0` |
| Benchmark | Reproducible release-path prompt/context, full-model prefill/decode/generation throughput, latency, and memory measurements | not measured; bounded attention runtime benchmarks are available but do not satisfy this gate | `V010.BENCH.DEEPSEEK.0` |
| Release | Full validation, artifact guardrail, claim audit, transcript, packaging, version, and tag | blocked | `V010.RELEASE.0` |

A gate changes state only through its owning milestone and executable proof.
Documentation can correct claims and dependency control; it cannot promote an
artifact, runtime, generation, evaluation, benchmark, or release capability.

## 12. Reference Architecture Ownership

`docs/reference-architecture.md` is the sole owner of the
implementation-agnostic inference architecture, its terminology, planes,
identity and lifecycle decomposition, conformance invariants, and the external
paper/specification/implementation map. Its traceability appendix connects
primary vLLM, SGLang, GGUF/ggml/llama.cpp, MLC LLM, IREE/MLIR,
TensorRT-LLM/CUTLASS, ExecuTorch, DwarfStar, and DeepSeek sources to concrete
YVEX owners and milestones. The reference architecture informs design and
independent comparison; it does not confer API compatibility, runtime topology,
backend support, model support, compilation correctness, or benchmark
evidence, and it does not own current project state.

## 13. Explicit Non-Claims

YVEX does not currently claim:

- a supported DeepSeek-V4-Flash target;
- a supported DeepSeek-V4-Flash model artifact; admitted variants pass hosted generation but have not passed evaluation, benchmark, and release gates;
- automatic precision optimization, target-size solving, calibration-data collection, placement search, evaluation-feedback optimization, or Pareto variant selection;
- a release-profile or quality claim for the mixed IQ2_XXS/Q2_K candidate before evaluation and benchmark evidence;
- arbitrary-family policy admission beyond the first complete DeepSeek physical-variant vertical;
- an artifact materialization cache or inference-time SSD expert streaming;
- complete GPU model residency, paged weight residency, inference-time SSD streaming, or automatic placement; the complete encoded model is locked in host RAM while one admitted accelerator prefix is resident on CUDA;
- paged, spilled, or quantized KV;
- CUDA sampling; upstream logits may come from CUDA, while the canonical sampler is common-host;
- a public or remote production server, authentication, TLS, full OpenAI API equivalence, Anthropic compatibility, multimodal OpenAI input, hosted tools, multi-model hosting, continuous batching, distributed serving, or sessions persisted across daemon restart; only YVEX OpenAI Compatibility Profile v1 is complete;
- GPU-resident sampling or tokenizer execution; the CUDA generation path uses CUDA model/logits execution with common-host orchestration, sampling, and tokenization;
- model behavior, model quality, question-answering, judge-model, or release-path evaluation readiness;
- full-model/generation benchmark evidence or release qualification; bounded attention-component qualification, measurements, and charts do not close those gates;
- an agent runtime, tool registry, tool execution, generalized action/observation loop, agent evaluation, agent CLI, agent track, or agent milestone; the provider function-call contract leaves tool execution to the application and does not establish those capabilities;
- Qwen, Gemma, GLM, or another model as a supported generation target;
- Metal, ROCm, remote/distributed serving, or speculative support.

Qwen/Gemma profiles, mappings, inventories, proof artifacts, primitive
comparisons, CLI grammar, reports, fixtures, and external runner output remain
real project evidence at their named stage. They are neither deleted nor
promoted into unsupported claims.

## 14. Version Sequence

| Version | Contract |
| --- | --- |
| v0.1.0 | DeepSeek-V4-Flash text generation from a complete YVEX-produced GGUF on DGX Spark CUDA |
| v0.1.x | Correctness, reliability, extra qtypes, streaming, performance, and memory hardening of the same path |
| v0.2.0 | Qwen dense full generation over the same family-aware common engine, subject to explicit complete gates |
| v0.3.0 | Gemma full generation, subject to explicit complete gates |
| v0.4.0 | Multi-family/multi-qtype matrix and backend portability after explicit scope decisions |

The version sequence records intended dependency order, not current support.

## 15. Documentation Ownership And Cutover

| Document | Sole responsibility |
| --- | --- |
| `README.md` | Public project identity, first successful product workflow, concise product topology/current vertical, build entry point, documentation navigation, and public non-claims |
| `PROJECT.md` | Product/release target, engineering scope, track registry, complete wave ledger, milestone state, dependencies, family matrix, gates, and Active Next |
| `AGENTS.md` | Persistent repository ownership, implementation, testing, validation, claim, and project-update invariants |
| `docs/v010-release-doctrine.md` | Release meaning and gate-closure semantics |
| `docs/system-target.md` | Filesystem and module ownership |
| `docs/model-families.md` | Normative family-integration architecture and implemented family profiles, not progress state |
| `docs/contract.md` | Runtime, CLI, lifecycle, failure, and ownership contracts for implemented surfaces |
| `docs/api.md` | Public C API facts and lifetime boundaries |
| `MODEL_ARTIFACTS.md` | Artifact terminology, admission, integrity, writer/reader, materialization, and support contract |
| `docs/operator-runbook.md` | Current implemented operator workflows, refusals, and recovery |
| `docs/runbooks/deepseek.md` | Exact current DeepSeek operator boundary |
| `docs/topology-closure-audit.md` | Point-in-time topology evidence, not project state |
| `docs/audits/operator-surface-ec7dcc/` | Frozen identity-bound operator-surface evidence and successor design input, never live command or project authority |
| `docs/milestones/command-architecture.md` | Future canonical operation, command, flag, projection, and discovery implementation contract; no current capability claim |
| `docs/milestones/runtime-console-repl.md` | Future mature daemon-backed terminal and REPL contract consuming canonical operations; no current capability claim |
| `docs/reference-architecture.md` | Implementation-agnostic inference architecture, conformance model, external sources, and YVEX owner/milestone traceability |
| `docs/diagrams/*.mmd` and `docs/diagrams/*.svg` | Editable and canonical vector projections of README product topology and detailed system architecture; orientation only, never runtime evidence |

## 16. Agent Start Checklist

Before proposing or implementing a delivery:

1. Read `AGENTS.md`, `PROJECT.md`, the owning contract, implementation,
   direct consumers, and focused tests.
2. Confirm the single Active Next and its predecessors.
3. Confirm the row is rank `milestone`; never schedule a subordinate row as a
   standalone delivery.
4. Inspect existing types, registries, APIs, fixtures, and guards before
   proposing files or symbols.
5. State exact before/after capability, owner, consumer, family scope, failure
   behavior, proof stage, and higher-stage non-claims.
6. Reuse common infrastructure; do not hard-code DeepSeek behavior into
   family-neutral artifact, materialization, descriptor, or backend owners.
7. Preserve Qwen, Gemma, dense/common, fixture, diagnostic, and topology work at
   its truthful rank.
8. Run focused tests and complete repository validation.
9. Classify `cli_applicability` for every milestone. Executable capability is
   incomplete until the main `yvex` binary invokes its production API directly;
   non-executable work records an owned non-applicability reason.
10. Update this map atomically with implementation result and next dependency.

The complete current ledger remains in this file. Future refinement may improve
rank, descriptions, or milestone grouping, but it must preserve every ID or
record an explicit migration.
