# DeepSeek Selective Residency Barrier

| Field | Value |
| --- | --- |
| Date | 2026-08-24 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `aa08bf0fa9cfcaa94c2f1ce3d46d22303669288c` |
| Checkpoint | `aa08bf0fa9cfcaa94c2f1ce3d46d22303669288c` |
| Subsystem | compiler physical lowering, runtime residency, and CUDA quantized execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 / 128 GiB unified memory |
| Evidence | controlled live characterization; decode profile; CUDA oracle; memory and swap observation |
| Comparability | approximately comparable |
| Publishability | reviewed |

## Before

The repaired DeepSeek tree retained Physical Execution IR v4, runtime binding v14 and the exact
artifact identity `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
The canonical GGUF remained an immutable artifact-backed mapping, while CUDA consumed its encoded
weights directly through unified addressing. No separate SSD streaming or device-weight copy was
active.

A fresh uncontended baseline on the current tree measured a 9.83 token/s target-only short median
over ten retained samples and 7.62 token/s over the three retained 256-token target-only samples.
DSpark measured 10.68 token/s on the matching short fixture and 9.72 token/s on the matching
256-token fixture. The greedy committed output was identical between target-only and DSpark.

The current decode characterization still attributed the largest device owners to routed MoE and
quantized row execution. The captured request contained 47,477 kernels and approximately 2,022 ms
of kernel intervals. Named contributions included approximately 390 ms routed-up, 210 ms
routed-down, 292 ms generic qtype, 191 ms MXFP4-to-Q8 work, 189 ms grouped qtype, 153 ms mHC,
145 ms attention and 98 ms output-head work.

## Problem

DS4 demonstrates mature matrix-oriented execution through Q8 activation reuse and layout-aware
MMQ kernels. That observation did not establish that copying YVEX weights into CUDA memory would
improve the current GB10 path. YVEX needed to separate three questions:

1. whether caching canonical encoded bytes changes execution economics;
2. whether one small high-reuse tensor is worth an explicit cache;
3. whether a compiler-sealed derived expert layout fits the 128 GiB resource contract.

The comparison had to preserve YVEX ownership: the compiler seals stable layout facts, runtime
instantiates admitted storage, and CUDA consumes the decision without recognizing DeepSeek.

## Causal analysis

A 5.40 GiB selective raw CUDA cache preserved exact output but did not change the row-dot
algorithm. Against a 9.74 token/s ten-sample baseline it measured 9.285 token/s, a 4.7% regression.
A reverse-order five-sample comparison measured 9.48 token/s cached versus 9.76 token/s baseline,
a 2.9% regression. The direction was therefore stable across ordering.

An output-head-only cache used approximately 0.99 GiB and preserved exact output. Its ten-sample
median was 9.675 token/s against 9.74 token/s, a 0.7% regression. The mapped source already
provided an adequate CUDA-addressable range; an extra resident copy added lifecycle and memory
cost without changing the physical projection.

A compiler-sealed derived MoE pack was then evaluated. The existing Physical IR v4 decision and
binding v14 identity could already represent the derived backend layout without a schema bump.
The focused CUDA oracle proved exact derived-layout decoding and fail-closed rejection. However,
packing all 129 routed gate/up/down tensors required approximately 72.56 GiB in addition to the
canonical artifact mapping. Model admission reached roughly 114 GiB RSS and increased swap-out by
approximately 0.8 GiB before producing a token, so the candidate was removed.

The minimum material subset was then reduced to the 43 routed-down Q2_K tensors, the approximately
210 ms/request profiled owner. Its derived pack required about 30 GiB. During real model admission
the process reached approximately 120 GiB RSS; `pswpout` increased by 96,852 pages, approximately
378 MiB, before the first token. The run was stopped at the resource gate and the generated binding
and all candidate source were removed.

The large RSS values include reclaimable file-backed GGUF pages and do not mean that YVEX copied
the full artifact into anonymous RAM. The swap delta is nevertheless decisive: the added managed
pack placed anonymous residency pressure on the unified-memory machine even while Linux still
reported substantial reclaimable memory.

## Decision

Retain the artifact-mapped canonical encoded path and reject all three cache configurations. A
cache is not admitted merely because it moves bytes closer to CUDA. It must change a material
physical operation, fit without swap and improve the complete model under a directly comparable
fixture.

Do not add a runtime heuristic that chooses cached tensors from qtype, dimensions, model name or
current free memory. The next credible owner is an identity-bound physical artifact/layout that
replaces, rather than duplicates, canonical execution bytes, or a bounded residency-window design.
The latter is SSD/payload streaming work and remains explicitly deferred.

The DS4 study is therefore classified as:

- **ADOPT:** matrix-oriented quantized execution and persistent activation reuse;
- **ADAPT:** derived layout provenance through YVEX compiler, Physical IR and binding identity;
- **REJECT:** unconditional device copies, CUDA-local family policy and a second runtime planner.

## Implementation

No cache implementation was retained. Each candidate was built, tested and exercised against the
complete model, then removed after failing its performance or resource gate. The accepted product
checkpoint remains `aa08bf0fa9cfcaa94c2f1ce3d46d22303669288c`; Physical Execution IR remains v4,
runtime binding remains v14, and the current production path remains artifact-mapped narrow DP4A.

Canonical QA exposed one test-isolation defect: the client-cutover test created a private home but
used the operator's real home until halfway through the script. A newer external model registry
could therefore alter a CLI grammar assertion. The test now exports its owned home and starts with
an empty valid registry before invoking the client. This changes test isolation only, not product
execution.

The durable repository changes are this evidence checkpoint and the hermetic client-cutover test
repair. Raw outputs, profiles, generated bindings and benchmark registries remain untracked
external operator evidence.

## After

YVEX has an explicit selective-cache policy boundary: cache admission requires a compiler-sealed
physical decision, exact numeric evidence, zero swap and a material complete-model gain. Neither
raw caching nor the tested derived MoE duplication satisfies that contract on the GB10.

Target-only remains at the fresh 9.83 token/s short characterization and 7.62 token/s long
characterization. DSpark remains positive on the controlled 256-token no-think fixture but not on
the 1,000-token reasoning characterization: target-only measured 2.22 token/s while DSpark measured
2.08 token/s with 241 accepted of 530 proposed tokens. The 20 token/s gate and preferred 24 token/s
gate remain failed.

## Quantitative delta

| Regime | Retained samples | Median token/s | Matched control | Result |
| --- | ---: | ---: | ---: | --- |
| Target-only short accepted path | 10 | 9.83 | not applicable | retained baseline |
| DSpark short accepted path | 10 | 10.68 | 9.83 target-only | characterization gain |
| Target-only 256-token | 3 | 7.62 | not applicable | retained baseline |
| DSpark 256-token | 3 | 9.72 | 7.62 target-only | positive economics |
| Target-only reasoning, 1,000 token | 1 | 2.22 | not applicable | characterization only |
| DSpark reasoning, 1,000 token | 1 | 2.08 | 2.22 target-only | negative economics |
| 5.40 GiB raw selective cache | 10 | 9.285 | 9.74 | reject, -4.7% |
| 5.40 GiB raw cache reverse order | 5 | 9.48 | 9.76 | reject, -2.9% |
| 0.99 GiB output-head cache | 10 | 9.675 | 9.74 | reject, -0.7% |
| 72.56 GiB derived routed MoE | no token | not available | zero-swap gate | reject |
| approximately 30 GiB derived routed-down | no token | not available | zero-swap gate | reject |

## Evidence

- External performance evidence is rooted at the identity-bound
  `2026-08-24-physical-regime-closure-aa08bf0` operator directory. The current decode profile
  SQLite digest is `9dbf56f8b292c5dc988005484b1684787e9712f82cd968eb5a3688f1c1aced7f`.
- Structural QA evidence
  `1d81d4f2e0365ec8ea1e3d9189817e7ac1071e848585ca35775f6aa310f51e4f`
  recorded 14 PASS and zero FAIL, SKIP, BLOCKED or ERROR after rejected source was removed.
- Canonical branch-delta QA evidence
  `493626b816fc6a406dc3625a50d079fa1810466033ecbf8b4ccd4e8261a51a01`
  recorded 101 PASS, one environment-sensitive CLI failure and two missing-asset blocks. The CLI
  failure exposed the test-home isolation defect repaired by this checkpoint.
- Focused post-repair CLI evidence
  `8fe1ef71776a47c4f6edcda27c0cca485af24967f60bcf594b8e0190ed810031`
  recorded one PASS and no unresolved result.
- Live DeepSeek evidence
  `95efe1329038f62cb96e1f13a7aea24bea09fdae17e0076c812ed0b38ea59846`
  recorded one PASS for CPU/CUDA target-only, stochastic repeatability, DSpark and operator
  execution after the artifact and v14 binding were configured explicitly.
- Runtime performance evidence
  `5e894ac21d10b188e8ab0db3455d7bc7d30404d3e48259d2da95e406d37725af`
  recorded one PASS for the external schema-five eager, piecewise and full attention-component
  evidence. Their CUDA device p50 values were 65.603, 66.204 and 66.314 ms respectively.
- Sanitizer evidence
  `87fd47e07756af9d49f6d25ddd51bf555589e349d923c4622e4696bf5ae67368`
  recorded ASan/LeakSanitizer and UBSan PASS for both registered sanitizer owners.
- The current CUDA suite passed, including exact Q2_K, IQ2_XXS and MXFP4 qtype oracles. The
  temporary derived Q2_K oracle also passed before its rejected implementation was removed.
- The retained 128-token CUDA chunk-equivalence fixture passes all 43 layers exactly after the
  candidate-alias repair at checkpoint `5d1771b434905b16ba48ae5e6807e7ef3439f308`.
- The short and 256-token target-only/DSpark fixtures retained identical greedy committed output.
- No candidate binding, source path, server, profiler or GPU process survived rejection.

## Remaining limitations

- `target_20_tok_s` and `preferred_24_tok_s` remain failed. This checkpoint is not milestone
  closure, release qualification or a public benchmark.
- Production Tensor Core coverage remains zero in the accepted narrow regime.
- DSpark is economically positive on the controlled no-think fixture but negative on the retained
  long reasoning characterization. It is not universally accelerated.
- The current model continues to use artifact-mapped encoded weights. Explicit device memory in
  operator telemetry does not include file-backed CUDA-addressable pages and must not be read as
  total physical model residency.
- An in-place derived physical artifact or bounded residency-window design requires a separately
  reviewed architecture. SSD/payload streaming remains owned by its deferred milestone.
- The full independent source-semantic oracle layer requested by the macro optimization milestone
  is not completed by this performance checkpoint.

## Why it matters

The checkpoint prevents a mature-looking cache feature from becoming permanent architecture when
it either slows the complete model or violates the GB10 memory contract. Future acceleration can
start from the measured need to replace physical bytes or window them, not from another duplicate
CUDA cache.
