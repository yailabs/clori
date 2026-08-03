# External Reference-Engineering Baseline

Status: maintained development reference

This document owns the pinned external sources used to pressure or check YVEX
architecture and implementation. It does not import external capability,
project state, or release claims. The implementation-independent architecture
is documented separately in the
[verified-inference reference](../reference/verified-inference.md).

This record is intentionally separate from the implementation-agnostic
architecture. It maps primary papers, specifications, and implementation
references to YVEX owners. It does not own project state, milestone state,
dependency order, capability claims, or Active Next; those belong only to
[`ROADMAP.md`](../../ROADMAP.md).

References identify proven decomposition, exact model behavior, container
rules, numerical contracts, and independent comparison points. YVEX still owns
its C API, process model, memory lifetime, failure behavior, tests, and support
claims.

## Ownership rules

- Prefer the original paper, format specification, official documentation, or
  direct implementation source over summaries and blog posts.
- Record the exact upstream revision when a milestone turns a reference into
  executable behavior or a golden comparison.
- Translate concepts through the owning YVEX type or module; do not reproduce a
  Python class hierarchy as C file structure.
- External support, output, artifacts, and benchmarks do not prove YVEX
  support.
- Conflicting sources are blockers until the exact local model source and a
  validated implementation contract resolve them.

## Reference-to-owner map

| Primary reference | Concept studied | YVEX owner | Consuming milestone | YVEX does not inherit or claim |
| --- | --- | --- | --- | --- |
| [vLLM architecture](https://docs.vllm.ai/en/latest/design/arch_overview/) and pinned DeepSeek-V4 source at [`8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`](https://github.com/vllm-project/vllm/blob/8df14cfc8c8a09b4e57f082e59593a3abce4ffb3/vllm/model_executor/models/deepseek_v4.py) | Model/runner lifetime separation, input preparation, model integration, attention/KV handoff, and backend selection | `src/model/families/`, `src/runtime/`, `src/graph/` | `V010.MODEL.ARCH.IR.0`, both runtime-descriptor milestones, `V010.RUNTIME.1`, `V010.GRAPH.DEEPSEEK.ATTENTION.0`, `V010.RUNTIME.DEEPSEEK.KV.0`, `V010.RUNTIME.DEEPSEEK.GENERATION.0` | Python APIs, worker processes, scheduling policy, distributed topology, serving interfaces, cache layout, or model support |
| SGLang runtime and DeepSeek-V4 model source pinned at [`96a04cb13f9c3ed86028e090784a9eb059cf5318`](https://github.com/sgl-project/sglang/blob/96a04cb13f9c3ed86028e090784a9eb059cf5318/python/sglang/srt/models/deepseek_v4.py) | Model loader/runner decomposition, DeepSeek layer construction, attention variants, KV use, MoE routing/expert execution, and prefill/decode composition | `src/model/families/`, `src/runtime/`, `src/graph/`, `src/backend/cuda/` | `V010.MODEL.ARCH.IR.0`, `V010.RUNTIME.1`, `V010.GRAPH.DEEPSEEK.ATTENTION.0`, `V010.RUNTIME.DEEPSEEK.KV.0`, `V010.RUNTIME.DEEPSEEK.PREFILL.0`, `V010.RUNTIME.DEEPSEEK.MOE.0`, `V010.GRAPH.DEEPSEEK.TRANSFORMER.0` | SGLang APIs, Python runtime, scheduler, distributed execution, kernel availability, or automatic compatibility |
| Pinned ggml `af97976c7810cdabb1863172f31c432dab767de7`: [GGUF specification](https://github.com/ggml-org/ggml/blob/af97976c7810cdabb1863172f31c432dab767de7/docs/gguf.md), [official GGUF reader/writer](https://github.com/ggml-org/ggml/blob/af97976c7810cdabb1863172f31c432dab767de7/src/gguf.cpp), [`ggml_type` and row API](https://github.com/ggml-org/ggml/blob/af97976c7810cdabb1863172f31c432dab767de7/include/ggml.h), and [type traits](https://github.com/ggml-org/ggml/blob/af97976c7810cdabb1863172f31c432dab767de7/src/ggml.c) | Container metadata and tensor directory ABI, power-of-two alignment, ordered padded layout, and qtype geometry | `src/gguf/`, `src/artifact/` | `V010.GGUF.QTYPE.ABI.1`, `V010.GGUF.ARTIFACT.ABI.1`, `V010.GGUF.LAYOUT.INTEGRITY.1`, `V010.QUANT.2`, `V010.GGUF.WRITER.1`, `V010.GGUF.ROUNDTRIP.1` | qtypes outside the pinned on-disk baseline, conversion or compute support from geometry, or artifact support from parse/layout acceptance alone |
| Pinned llama.cpp DeepSeek-V4 mapping at [`e920c523e3b8a0163fe498af5bf90df35ff51d25`](https://github.com/ggml-org/llama.cpp/blob/e920c523e3b8a0163fe498af5bf90df35ff51d25/conversion/deepseek.py), its [loader](https://github.com/ggml-org/llama.cpp/blob/e920c523e3b8a0163fe498af5bf90df35ff51d25/src/models/deepseek4.cpp), [architecture names](https://github.com/ggml-org/llama.cpp/blob/e920c523e3b8a0163fe498af5bf90df35ff51d25/src/llama-arch.cpp), and [GGUF constants](https://github.com/ggml-org/llama.cpp/blob/e920c523e3b8a0163fe498af5bf90df35ff51d25/gguf-py/gguf/constants.py) | DeepSeek trunk roles, canonical names, logical shapes, FP8 companion handling, 256-expert MXFP4 aggregation, and I64 hash-table conversion | `src/model/families/deepseek_v4.c`, `src/model/target/tensor_naming.c` | `V010.MAP.GGUF.DEEPSEEK.0` | payload conversion correctness, writer output, artifact support, runtime support, or upstream compatibility for MTP; the pinned converter explicitly discards `mtp.*` |
| MLC LLM pinned at [`a2bcc5c86678b72a86b7aadc29b643a5ce63c747`](https://github.com/mlc-ai/mlc-llm/tree/a2bcc5c86678b72a86b7aadc29b643a5ce63c747) and its [library/weight packaging contract](https://github.com/mlc-ai/mlc-llm/blob/a2bcc5c86678b72a86b7aadc29b643a5ce63c747/docs/compilation/package_libraries_and_weights.rst) | Separation of model/weight conversion, target-specific compiled libraries, packaged physical variants, runtime memory estimates, and compilation-cache identity | `TRACK.COMPILATION`, later physical-lowering and artifact owners | `V010.MODEL.TRANSFORM.IR.0`, `POST010.COMPILATION.DAG.0`, `POST010.COMPILATION.REUSE.0`, `POST010.COMPILATION.VARIANTS.0` | TVM/Relax IR, Python tooling, MLC artifact formats, cache keys, target support, optimization results, or runtime compatibility |
| IREE pinned at [`0893eac771d532b7110f1f7581d3f4cd0b9172bf`](https://github.com/iree-org/iree/tree/0893eac771d532b7110f1f7581d3f4cd0b9172bf) and its [MLIR lowering architecture](https://iree.dev/) | Successive typed lowering, explicit deployment constraints, separation of compiler IR from runtime objects, and target-specific executable production | `TRACK.COMPILATION`, later graph/runtime descriptor owners | `V010.MODEL.TRANSFORM.IR.0`, `POST010.COMPILATION.HARDWARE.PROFILE.0`, `POST010.COMPILATION.RUNTIME.BINDING.0` | MLIR dialects, IREE compiler/runtime APIs, executable formats, scheduling decisions, device support, or performance claims |
| TensorRT-LLM pinned at [`02cedf6e4e421ac48d271452bf3836cb57caf297`](https://github.com/NVIDIA/TensorRT-LLM/tree/02cedf6e4e421ac48d271452bf3836cb57caf297), its [checkpoint-to-engine workflow](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html), and [benchmark-guided engine build](https://nvidia.github.io/TensorRT-LLM/performance/perf-benchmarking.html) | Distinct source checkpoint, engine build, engine load/evaluation identities; workload-derived build constraints; precision, parallelism, KV, latency, and throughput profiles | `TRACK.COMPILATION`, `TRACK.EVAL`, `TRACK.BENCH`, later artifact/runtime owners | `POST010.COMPILATION.HARDWARE.PROFILE.0`, `POST010.COMPILATION.WORKLOAD.PROFILE.0`, `POST010.COMPILATION.FEEDBACK.0`, `POST010.COMPILATION.PARETO.0` | TensorRT checkpoints/engines, Python builders, plugins, heuristic choices, benchmark results, or automatic YVEX variant admission |
| ExecuTorch pinned at [`ae754e9ed8b650e78b921906b2ba8af65ea408ab`](https://github.com/pytorch/executorch/tree/ae754e9ed8b650e78b921906b2ba8af65ea408ab) and its [export/compile/execute architecture](https://docs.pytorch.org/executorch/stable/getting-started-architecture) | Separation of source export, graph transformations, target lowering, emitted program, runtime preparation, memory planning, and execution evidence | `TRACK.COMPILATION`, later artifact, residency, and runtime descriptor owners | `V010.MODEL.TRANSFORM.IR.0`, `POST010.COMPILATION.PLACEMENT.0`, `POST010.COMPILATION.RUNTIME.BINDING.0`, `POST010.COMPILATION.EXECUTION.STATE.0` | PyTorch/EXIR dialects, delegates, FlatBuffer/PTE formats, C++ runtime, operator support, or target capability |
| DwarfStar pinned at [`80ebbc396aee40eedc1d829222f3362d10fa4c6c`](https://github.com/antirez/ds4/tree/80ebbc396aee40eedc1d829222f3362d10fa4c6c), including its [CLI loop](https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_cli.c), [local help](https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_help.c), [engine-specific model contract](https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/README.md), [SSD expert streaming](https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/ds4_ssd.c), and [quality validation](https://github.com/antirez/ds4/blob/80ebbc396aee40eedc1d829222f3362d10fa4c6c/gguf-tools/quality-testing/README.md) | Stable interactive prompt, local discovery, progress and interruption behavior, explicit-output presentation, engine-specific physical artifacts, routed-expert cache budgets, SSD/host memory tradeoffs, official-logit and continuation comparison, and variant-specific quality checks | `src/cli/io/`, future compilation placement/feedback obligations, `TRACK.EVAL`, `TRACK.BENCH` | `V010.OPERATOR.REPL.CONSOLE.0`, `POST010.COMPILATION.PRECISION.0`, `POST010.COMPILATION.PLACEMENT.0`, `POST010.COMPILATION.FEEDBACK.0`, `POST010.STORAGE.GEN.0` | DwarfStar command grammar or process topology, hidden reasoning access, GGUF layouts, DeepSeek support, Metal/ROCm/CUDA capability, SSD-streaming implementation, published quality, or measured performance |
| NVIDIA [TensorRT-LLM architecture](https://nvidia.github.io/TensorRT-LLM/architecture/overview.html) at [`02cedf6e4e421ac48d271452bf3836cb57caf297`](https://github.com/NVIDIA/TensorRT-LLM/tree/02cedf6e4e421ac48d271452bf3836cb57caf297) and CUTLASS at [`1e7394829291360bdcf07036cbe5411631d2d33b`](https://github.com/NVIDIA/cutlass/tree/1e7394829291360bdcf07036cbe5411631d2d33b) | CUDA execution boundaries, KV/resource ownership, model-engine separation, Blackwell matrix kernels, low-precision layouts, MoE execution, synchronization, and profiling | `src/backend/cuda/`, `src/graph/plan.c`, `src/graph/memory_plan.c` | `V010.CUDA.FAILCLOSED.0`, `V010.GRAPH.DEEPSEEK.ATTENTION.0`, `V010.RUNTIME.DEEPSEEK.MOE.0`, `V010.GRAPH.DEEPSEEK.TRANSFORMER.0`, `V010.BENCH.DEEPSEEK.0` | TensorRT engine format, plugins, executor/process model, multi-GPU scope, fused-kernel availability, performance, or backend support |
| NVIDIA [Driver API module management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html) and [execution control](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html) | Distinct context, module load, function resolution, launch, synchronization, rollback, and cleanup states | `src/backend/cuda/backend.c`, `src/backend/cuda/capability.c`, `src/backend/cuda/ops.c` | `V010.CUDA.FAILCLOSED.0` and later exact runtime-operation proofs | Context availability as kernel support, successful launch return as numerical proof, generated artifacts in source control, or CUDA model support |
| NVIDIA [Driver API graph management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html) | Graph construction/capture, dependency nodes, instantiation, update, launch, stable-address requirements, invalidation and destruction | `src/backend/cuda/graph.c`, `src/runtime/` | `V010.RUNTIME.1` attention eager/piecewise/full execution modes | CUDA Graph labels as capture proof, eager aliases as graph execution, automatic performance claims, CPU fallback, or serialized graph artifacts |
| DeepSeek [V4 technical report v1](https://arxiv.org/abs/2606.19348v1), official model snapshot at [`60d8d70770c6776ff598c94bb586a859a38244f1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1), its exact config/tokenizer sidecars, and [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Exact architecture identifier, hybrid attention, compressed KV, position rules, mHC, MoE topology, tokenizer/encoding contract, source dtype/quantization facts, and reference kernel behavior | `src/source/`, `src/model/families/`, `src/tokenizer/`, `src/graph/`, `src/backend/cuda/` | `V010.REBASE.DEEPSEEK.0`, `V010.MODEL.ARCH.IR.0`, `V010.TENSOR.COVERAGE.DEEPSEEK.0`, `V010.MAP.GGUF.DEEPSEEK.0`, `V010.RUNTIME.DEEPSEEK.TOKENIZER.0`, attention/MoE milestones | Facts absent from the local source, assumptions inferred from the family name, external runtime output, external GGUF support, or DeepSeek support in YVEX |
| Official [DeepSeek-V4-Flash-DSpark snapshot](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/tree/62af8fffb2f7030cac4de2f0169f5b8d1101b646), [DSpark paper v1](https://arxiv.org/abs/2607.05147v1), bundled inference code at the same source revision, DeepSpec at [`005e03b81cec38b7da6399833d609ee89a2587f2`](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2), and the vLLM DSpark implementation at [`d9dac2b3d4cdd57ff2f7c5311bc8aaf3eef3feec`](https://github.com/vllm-project/vllm/tree/d9dac2b3d4cdd57ff2f7c5311bc8aaf3eef3feec/vllm/v1/worker/gpu/spec_decode/dspark) | Five-position semi-autoregressive drafting, ordered target feature taps, three draft stages, noise queries, rank-256 Markov conditioning, confidence logits, full-target verification, stochastic residual sampling, and accepted-prefix commit | `src/source/`, `src/model/families/`, `src/graph/families/`, `src/runtime/`, `src/server/` | `V010.REBASE.DEEPSEEK.DSPARK.0` | External process topology, Python ownership, external runtime support, published speedups, load-aware scheduling, quality parity, or YVEX support before local live acceptance |

## Validation boundary

### Pinned DeepSeek-V4 architecture baseline

`V010.MODEL.ARCH.IR.0` interprets the exact model snapshot at
`60d8d70770c6776ff598c94bb586a859a38244f1` against DeepSeek-V4 paper
`arXiv:2606.19348v1`, SGLang commit
`96a04cb13f9c3ed86028e090784a9eb059cf5318`, and vLLM commit
`8df14cfc8c8a09b4e57f082e59593a3abce4ffb3`. The source snapshot is the
identity/config authority; the paper and runtimes resolve architecture
semantics and independent topology expectations. The canonical YVEX result is
the immutable owner under `src/model/families/`, not an inherited Python class
hierarchy or external support claim.

The current source rebase consumes the DSpark snapshot at
`62af8fffb2f7030cac4de2f0169f5b8d1101b646`. Its configuration and tensor
inventory are authoritative for payload identity and geometry. DSpark paper
`arXiv:2607.05147v1`, the bundled inference code, DeepSpec commit
`005e03b81cec38b7da6399833d609ee89a2587f2`, and vLLM DSpark commit
`d9dac2b3d4cdd57ff2f7c5311bc8aaf3eef3feec` resolve proposal, verification,
acceptance, and commit semantics. The official vLLM recipe uses greedy draft
sampling with standard target rejection sampling; the distinct optional block
verification method is not DSpark's default semantic contract. In particular,
the public configuration's
single next-token prediction declaration is not treated as the draft depth:
the bundled inference configuration and tensor geometry establish three draft
stages conditioned by target features from layers 40, 41, and 42.

`V010.GRAPH.DEEPSEEK.ATTENTION.0` resolves the executable attention boundary
against those same pinned sources and runtime-numeric authority commit
`e7706faf8d1c3b9f241e36860640ad1dac644ede`. The local source and pinned
runtime implementations establish base RoPE without YaRN for SWA, the
compressed-class YaRN policy for CSA/HCA, and HCA admission of a complete
non-overlapping group when its end position becomes available at the exact
ratio 128. The paper remains semantic context; it does not override the exact
executable source transition or transfer external support into YVEX.

`V010.RUNTIME.1` consumes those semantics through a typed DeepSeek adapter and
one family-neutral runtime model/session. vLLM and SGLang inform the separation
between immutable model resources, mutable request state and phase-aware
execution; YVEX does not inherit their worker topology, scheduler, Python API or
KV layout. CUDA Graph execution follows the Driver API lifecycle directly and
captures YVEX production kernels from the existing generated bundle. The
semantic graph, lowered executable graph, CUDA launch graph and instantiated
graph executable retain separate identities.

### Pinned GGUF qtype storage baseline

`V010.GGUF.QTYPE.ABI.1` uses ggml commit
`af97976c7810cdabb1863172f31c432dab767de7`. At that revision the GGUF
specification defines the on-disk type range through ID 39. The implementation
also defines `NVFP4`, `Q1_0`, and `Q2_0` as IDs 40 through 42. YVEX records
those identities and their upstream geometry but refuses them as outside the
pinned on-disk baseline.

`include/yvex/qtype.h` and `src/gguf/qtype.c` own the admitted identity and
geometry. Storage uses `ne[0]` as row width, requires exact block division, and
multiplies row bytes by the checked product of the remaining dimensions. Dtype,
parser, range, integrity, conversion, and memory-plan owners consume that
registry. Storage admission remains independent from reference dequantization,
quantization, emission, and backend compute.

Each consuming milestone must record which upstream revision and local source
facts it used, the YVEX owner that implements the result, the independent test
or comparison, and any unresolved divergence. A reference becomes evidence
only through that milestone's executable tests. Until then it is design input.
