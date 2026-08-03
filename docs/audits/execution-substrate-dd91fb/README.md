# Verified Execution-Substrate Baseline and Disposition Audit

Status: frozen closure audit

Baseline: `dd91fb611918b9bbefe623318a434821b718151e`
(`feat(cli): support Ctrl-L in the REPL`)

Machine: `spark-7c3d`, NVIDIA DGX Spark / GB10, compute capability 12.1,
20-core Arm CPU, 128 GB coherent unified memory.

This audit records the host-authoritative mechanisms inspected before the
execution-substrate refoundation and their accepted dispositions. It is
point-in-time evidence, not architecture or project-state authority.

## Bound assets

| Asset | Identity / fact |
| --- | --- |
| Source | `deepseek-ai/DeepSeek-V4-Flash-DSpark` at `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| Source aggregate | `35e04611d8fd85a55aa394864a8a2adb5e0e3336fb40d871566be4f30d105903`; 74 files; 48 shards; 166,898,666,055 bytes |
| Target | `deepseek4-v4-flash-dspark` |
| Artifact | `deepseek-v4-flash-dspark-bootstrap-q2-v1.gguf`; 108,285,860,832 bytes |
| Runtime binding | filename identity `01f447828a734b7d664c4289e9080f08e3928b634825d713e8027c7255c1e489` |
| Selected profile | `deepseek4-v4-flash-dspark-runtime-iq2xxs`; CUDA; DSpark; context 4096 |
| Baseline protocol | 5 |
| Baseline daemon | PID 3839453 recorded before implementation; one model process |

The source acquisition and full runtime assets remain outside Git. Paths are
omitted here because machine-local paths are not portable identity.

## Reference execution fixture

The retained six-prompt-token, twelve-output-token observation reported:

| Phase / fact | Observation |
| --- | ---: |
| Prefill | 8.160 s; 0.74 token/s |
| Generation | 42.36 s; 0.28 token/s |
| TTFT | 21.54 s |
| DSpark | 15 proposed; 8 accepted; 3 rejected; 3 verifications; one 5/5 cycle |
| Draft cycles | 1.225 s; 1.235 s; 1.254 s |
| Verification cycles | 7.165 s; 7.829 s; 7.958 s |
| Commit projections | 4.914 s; 8.955 s; 1.631 s |
| Movement | H2D 85,544,288; D2H 36,775,028; D2D 16,908,288 bytes |
| Execution | 18,371 launches; 258 downloads; 258 device synchronizations; 4,644 expert subviews |

These values describe one retained causal fixture. They are not a benchmark or
performance admission.

## Disposition inventory

| Mechanism | Baseline class | Disposition | Replacement owner / closure proof |
| --- | --- | --- | --- |
| accepted-prefix target replay | production correctness | declassify-to-reference | prefix-addressable target state promotes verified rows; product replay counter is zero |
| whole-bank candidate copy | production transaction | replace | ordered candidate deltas plus synchronized banks |
| SWA full-window move | production state | replace | ring-backed doubled storage with contiguous projected view |
| complete state-value hashing | production evidence | move | initial/forensic content check; production advances identity from typed publication identity |
| embedding CPU shadow | production evidence | declassify-to-audit | production evidence profile omits shadow comparison |
| expanded-hidden host shadow | production evidence | declassify-to-audit | device view plus explicit audit/forensic materialization |
| target feature-tap D2H | portable execution | defer-to-GB10-wave | typed device feature view; portable adapter remains explicit |
| full-vocabulary D2H for greedy | production selection | replace | CUDA device argmax returns bounded token/status |
| full logits scan and digest | production evidence | move | availability-bearing audit/forensic evidence |
| host greedy sampling | production selection | replace | device-native greedy selection |
| host stochastic sampling | exact reference | declassify-to-reference | explicit stochastic-reference execution class |
| DSpark feature projection | family execution | preserve | source-authored semantics behind typed feature-view contract |
| DSpark Markov path | family execution | preserve | source-authored rank-256 composition |
| DSpark confidence path | family execution | preserve | scheduling facts remain non-authoritative |
| token-local MoE | portable execution | defer-to-GB10-wave | width-N common contract; token-local adapter named portable reference |
| per-token expert subview construction | portable execution | defer-to-GB10-wave | width-N row/expert and unique-expert accounting |
| per-layer status download | production evidence | move | bounded status facts; full status belongs to audit/forensic |
| per-layer device synchronization | portable execution | defer-to-GB10-wave | explicit synchronization metrics and later backend owner |
| eager production attention | portable execution | preserve | explicit eager portable-reference class |
| attention-only launch graph | backend execution | strengthen | shape/profile identity includes kernel bundle and operation scope |
| generic qtype CUDA matvec | portable execution | defer-to-GB10-wave | Physical Execution IR names kernel-family requirement |
| physical variant | product level | strengthen | distinct from Physical Execution IR and compiled execution profile |
| role-agnostic materialization placement | production planning | replace | typed role, consumer, sharing, phase and placement decisions |
| model residency | runtime lifecycle | strengthen | explicit host canonical, CUDA-addressable and device classes |
| concrete-family materialization import | forbidden dependency | retire | generic materializer consumes terminal typed facts; guard rejects family imports |
| runtime execution descriptor | runtime identity | split | Physical Execution IR plus compiled execution profile |
| single worker and bounded queue | serving policy | preserve | workload profile names current serialization; batching remains later work |
| full event hashing on numerical path | production evidence | move | operational events carry bounded typed facts; deeper evidence is explicit |
| raw watch rendering | operator projection | split | semantic watch, detailed human trace and independent JSONL |
| direct fragment write | terminal projection | replace | bounded incremental UTF-8/Markdown renderer plus byte-exact raw mode |
| reasoning substring classification | unsupported behavior | retire | tokenizer-owned source delimiters and typed explicit-reasoning channel |
| partial-session ambiguity | server lifecycle | replace | protocol-v6 partial-turn facts and reset-required continuation refusal |
| one mutable attention capacity | production admission | replace | identity-bound execution-shape registry and exact refusal payload |

Disposition totals: preserve 5; strengthen 3; move 4; split 2;
declassify-to-reference 2; declassify-to-audit 2; replace 8; retire 2;
defer-to-GB10-wave 5. The 33 inventoried mechanisms reconcile exactly and
no accepted-irreducible item remains.

## Capacity-failure fixture

The baseline prompt requesting a simple CUDA kernel failed after streaming an
open code fence with `CUDA attention history exceeds capture capacity`. The
failure lacked component, configured/required capacity, scope, phase, width,
position and shape identity, and left the session visibly ambiguous.

The replacement admits a target/draft/phase/width/context shape before
numerical mutation. A refusal names all available capacity and identity facts.
The terminal renderer ends only its visual code projection, renders the typed
failure separately and marks committed output partial without adding bytes to
the canonical transcript. Continued ordinary generation requires reset.

## Family-source reconciliation

The three same-basename DeepSeek sources are not duplicate runtimes. Model,
graph and fused CUDA lowering are separate family projections and have distinct
machine-readable owners. The empty concrete-family runtime directory was
removed. Architecture validation rejects a concrete-family runtime source, a
fourth DeepSeek family projection, a generic materializer family import or a
runtime call into source/compiler registration.

## Evidence boundary

Deterministic tests cover execution-profile identity, shape admission and exact
capacity refusal, device-view extent, prefix promotion for every admitted
five-token prefix, correction extension, ring state, protocol-v6 roundtrip and
v5 refusal, partial continuation refusal, arbitrary terminal fragment splits,
raw output, `NO_COLOR`, and source-authored explicit reasoning policy.

Full model, CUDA, sanitizer, package and controlled deployment results are
reported by the closure change and remain reproducible operator evidence; raw
logs and measurements are not tracked in this frozen audit.

## Preserved non-claims

This audit does not establish optimized GB10 kernels or layout, native FP4
Tensor Core execution, DSpark acceleration, a competitive rate, continuous
batching, evaluation, a release benchmark, another complete family or release
qualification.
