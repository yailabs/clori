# MoE Route-Weight Numerical Order

| Field | Value |
| --- | --- |
| Date | 2026-08-14 |
| Type | repair |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `01831d79b011ffb6d03c27ab2c4e858a5a673f5d` |
| Checkpoint | `8d4d2926c4d0169e59326c8faa97ce881eb9e272` |
| Subsystem | generic MoE semantics and CUDA execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | software tests; numerical conformance; runtime qualification |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

YVEX rounded each routed expert's SwiGLU activation to BF16, executed the down
projection, rounded that result to BF16, and only then multiplied by the router
weight. The CPU path, portable CUDA path, derived-layout path, and native SM121
Tensor Core path shared this numerical order.

The complete DeepSeek artifact exhibited a nondeterministic routed-down failure:
one of five repeated greedy runs reported device status 1, while other runs
diverged in committed text despite identical artifact, binding, prompt, and
sampling configuration.

## Problem

The implementation order did not match the pinned source model's expert
equation. A large SwiGLU intermediate could become non-finite when published as
BF16 before the router weight had reduced its magnitude. Moving the weight
after the down projection also changed the model's rounding contract.

## Causal analysis

The pinned DeepSeek source multiplies the router weight into the SwiGLU result
before converting that activation to the expert input dtype and applying the
down projection. Current DS4 commit
`84cc882352757baf628a1776badf7cc54d584e28` independently uses the same order in
its ordinary and D2R MoE kernels.

Focused kernel instrumentation localized the intermittent status to routed
down execution. Bypassing CUDA Graph launch did not remove it. Restoring the
source-authored numerical order removed the failure and the output divergence
across repeated complete-model runs. The evidence supports numerical ordering
as the cause; it does not establish a throughput improvement.

## Decision

Adopt the source and DS4 invariant while retaining the generic YVEX MoE owner.
The route weight is an explicit expert-execution operand and scales SwiGLU
before BF16 publication and the down projection. Shared experts use weight
`1.0` through the same contract.

Copying DS4 implementation code or introducing a DeepSeek branch in common
runtime was rejected. Applying a compensating clamp after the down projection
was also rejected because it would hide the overflow while preserving the
wrong model equation.

## Implementation

Checkpoint `8d4d2926c4d0169e59326c8faa97ce881eb9e272` changes the generic CPU expert
ABI and every admitted CUDA MoE shape: token-local, grouped, width-N,
derived-layout, and native Tensor Core. Up/SwiGLU kernels now consume routed
weights; down kernels consume already weighted BF16 activations and no longer
apply a second weight.

The independent CPU oracle now evaluates a non-unit router weight before its
BF16 intermediate. CUDA tests exercise the revised kernel signatures and the
SM121 bundle. Repository line budgets were reconciled to the exact admitted
tree after several accepted post-budget commits had left the clean baseline
outside its own structural guard.

## After

CPU, portable CUDA, derived-layout CUDA, and SM121 Tensor Core execution share
the source-authored route-weight order. Five repeated 96-token full-model runs
and twenty additional 32-token stress runs completed without routed-down
status failures or byte divergence.

The repaired execution remains bound to Physical Execution IR v3 and runtime
binding v13. It does not add a family-specific runtime path or change artifact,
binding, session transaction, RNG, or accepted-token replay semantics.

## Quantitative delta

| Fact | Before | After | Comparison |
| --- | --- | --- | --- |
| Five-run complete-model stability | 1 device failure and committed-output divergence | 5/5 byte-identical completions | directly comparable |
| Additional repaired stress | not run | 20/20 byte-identical completions | characterization only |
| Warm generation median | about 6.87 token/s | about 6.84 token/s | approximately comparable; no material regression claim |
| Route-weight order | after BF16 down output | before BF16 intermediate and down projection | directly comparable contract fact |

## Evidence

- The pinned source snapshot revision is
  `62af8fffb2f7030cac4de2f0169f5b8d1101b646`; its expert implementation applies
  route weight before the down projection.
- DS4 commit `84cc882352757baf628a1776badf7cc54d584e28` applies router weight to its
  pair-major SwiGLU intermediate before down execution. YVEX adopted the
  invariant, not the implementation.
- `runtime_moe` and the independent CPU oracle pass with a non-unit route
  weight.
- `cuda-info`, `make -j2 check-cuda`, and `test-cuda-native-sm121` pass,
  including native CUBIN admission and all admitted qtype equivalence checks.
- ASan/LeakSanitizer and UBSan pass for the complete runtime sanitizer lane.
- `make test-cuda-no-nvcc`, `make -j2 test`, and `make -j2 check` pass.
- The five 96-token repaired runs share output digest
  `9af1707c5f5c4a100bd7ee487499fdc7960de85cdf0c710b6d0476697bde83da`.
- The twenty 32-token stress runs share output digest
  `5f7d2338e831be8e96b081cbc3b63831de8a6279b1445f20b14baca6b5dc5a96`.
- Raw full-model receipts remain external identity-bound evidence and are not
  tracked in Git.

## Remaining limitations

- This repair does not close the GB10 performance targets; MoE remains the
  dominant measured decode phase.
- The approximately comparable throughput observations are not a promoted
  benchmark result.
- Deep-context qualification, durable state restore, continuous batching, and
  canonical distribution packaging remain separate open GB10 gates.
- This repair does not implement topology planning, SSD streaming, multiple
  devices, or model-generation hot reload.

## Why it matters

All admitted MoE execution paths now preserve the model's numerical equation,
turning an intermittent complete-model CUDA failure into deterministic output
without adding a family-specific runtime exception.

## Communication projections

### Short update

YVEX now applies MoE router weights to SwiGLU activations before BF16
publication and the down projection, matching the pinned DeepSeek source across
CPU, portable CUDA, derived layouts, and native SM121 Tensor Core execution.
The repaired path completed 25 repeated full-model runs without the prior
routed-down failure or output divergence; this is a correctness repair, not a
throughput claim.

### Visual candidates

- Before/after expert equation showing the BF16 rounding boundary.
- Five-run baseline versus repaired stability table.
- CPU, portable CUDA, derived-layout, and Tensor Core path-equivalence diagram.

### Quoteable technical facts

- “MoE router weights now scale SwiGLU before BF16 publication and the down
  projection.”
- “The repaired DeepSeek path completed 25 repeated full-model runs without
  routed-down status failure or byte divergence.”
- “The repair changes numerical order without creating a DeepSeek-specific
  common-runtime path.”
