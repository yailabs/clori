# GB10 Execution Target Table

Status: current optimization input

Schema: `yvex.gb10.execution-targets.v1`

This document owns the engineering target table consumed by the GB10
optimization boundary. It records targets, not achieved performance or release
claims. Raw measurements remain external identity-bound assets. Current macro
state remains in [`ROADMAP.md`](../../ROADMAP.md).

## Hardware envelope

The measured host is one NVIDIA DGX Spark with a GB10 GPU at compute capability
12.1, 48 SMs, a 20-core Arm CPU and 128 GB coherent unified memory. NVIDIA's
[DGX Spark hardware specification](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
states 273 GB/s peak memory bandwidth, 6,144 CUDA cores, two copy engines and
up to 1 PFLOP FP4 with sparsity. Those are physical ceilings, not YVEX-achieved
rates. The accepted artifact is a 108.29 GB mixed IQ2_XXS/Q2_K/BF16/Q8_0
bootstrap variant, so neither the FP4 compute ceiling nor artifact size alone
predicts token throughput.

The currently measured short-context reference fixture produced 6 prompt and
12 committed tokens: prefill 8.160 s (0.74 token/s), generation 42.36 s (0.28
token/s), TTFT 21.54 s, 18,371 kernel launches, 258 downloads, 258 device
synchronizations, H2D 85,544,288 bytes, D2H 36,775,028 bytes and D2D 16,908,288
bytes. Three DSpark verifications proposed 15 and accepted 8 draft tokens,
including one 5/5 cycle. This fixture is causal baseline evidence, not a model
benchmark.

## Target dimensions

Rates are committed output token/s. `unmeasured` means the next owner must
establish a reproducible baseline before comparing. A ratio is relative to the
same identity-bound target-only run. Competitive thresholds are deliberately
unadmitted until one reproducible same-machine, same-checkpoint, comparable-
precision external result exists; a headline from another model or physical
variant is not substituted.

An exact competitive hard gate is admitted after reference reproduction from
either a same-checkpoint comparable lane or a measured phase roofline with an
active-byte lower bound. This separates the physical closure threshold from
engineering ambition. Existing YVEX and stretch targets are not lowered when
comparability is absent.

An active-byte lower bound counts each compulsory device span once for the
measured phase. It is neither allocated capacity nor a claim about cache reuse
or profiler-observed DRAM transactions. A byte class remains unavailable until
its producing owner can publish that exact lower bound.

| Workload | Context / width | Current measured | Hard functional minimum | Competitive threshold | YVEX engineering target | Stretch target | Physical bound / governing evidence | Confidence |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| target-only decode | short, width 1 | 0.432 token/s retained warm baseline | correct sequence; >0 | unadmitted | 20 token/s | 25 token/s | active bytes, 273 GB/s, launch and synchronization depth | low |
| target-only prefill | short, chunk 6–64 | 0.74 token/s fixture | correct prefix; >0 | unadmitted | 20 token/s | 30 token/s | attention class, chunk width, active bytes, 48 SMs | low |
| target-only decode | 12K context, width 1 | unmeasured | exact continuation; >0 | unadmitted | 15 token/s | 20 token/s | context-band state traffic and attention mix | low |
| target-only decode | 64K context, width 1 | unmeasured | exact continuation; >0 | unadmitted | 8 token/s | 12 token/s | compressed/indexer history and capacity-safe state access | low |
| target-only decode | near admitted capacity, width 1 | unmeasured | no overflow; exact stop | unadmitted | 0.80 × short rate | 0.90 × short rate | shape registry and context-bound state traffic | low |
| DSpark favorable code | width 5 verification | 0.28 token/s fixture; 8/15 accepted | real proposal, verification and multi-token promotion | unadmitted | 30 token/s | 40 token/s | accepted rows per target sweep, draft and verify cost | low |
| DSpark ordinary dialogue | width 5 verification | unmeasured | target-equivalent committed sequence | unadmitted | 25 token/s | 35 token/s | measured acceptance distribution and verification width | low |
| DSpark low acceptance | width 5 verification | unmeasured | no silent fallback; exact residual sampling | unadmitted | at least 0.90 × target-only | at least 1.00 × target-only | bounded wasted draft/verification work | low |
| concurrent target decode | batch 2–4 | unsupported scheduler | isolated exact sessions | unadmitted | 35 aggregate token/s | 60 aggregate token/s | row batching, queue policy and 48-SM occupancy | low |
| long-context prefill | 12K–64K, admitted chunks | unmeasured | exact state and bounded memory | unadmitted | 50 token/s | 80 token/s | chunk width, attention class, 273 GB/s and launch count | low |

The numerical YVEX and stretch columns are optimization objectives. They are
not admissions, forecasts, competitive claims or release gates. The next wave
must either replace each low-confidence objective with measured, comparable
evidence or retain it as an unmet target. It may not lower correctness,
precision identity, context, workload or evidence depth to manufacture a pass.

## Required measurement key

Every result compared with this table binds:

- source revision, artifact, runtime binding, Physical Execution IR, compiled
  execution profile and kernel bundle;
- target-only or DSpark mode, evidence profile, sampling class and accepted
  prefix distribution;
- prompt/corpus identity, context band, input/output lengths, width, batch and
  concurrency;
- daemon lifecycle, memory placement, driver/toolkit, clocks, power/thermal
  state and machine identity;
- wall/device/host time, movement, state copies, full-array scans, launches,
  waits, synchronizations, memory high-water mark and committed-token counts.

An optimization passes only its named metric and correctness gates. Component
timing cannot promote a full-model rate, and this table cannot promote model
quality or release qualification.

## Next-owner obligations

The GB10 optimization owner must select GB10-specific expert layouts and
kernels from causal evidence. The correctness-first width-N MoE path already
defers bounded status/unique-expert publication across the transformer stack,
validates it once, reconstructs exact active bytes and reuses a proved final
session-stream barrier. It no longer materializes selected routes or weights on
the production host path. The portable Driver fallback reports any context-wide
synchronization explicitly; immediate and token-local paths remain the
audit/reference oracles. Target-only production now
selects stochastic tokens from resident CUDA logits with bounded result
transfer; audit/forensic and stochastic DSpark still own explicit host sampling
references. Greedy and stochastic CUDA selection enqueue those bounded facts
on the session stream and complete only that stream; any legacy context-wide
fallback remains separately visible. Production greedy DSpark verification now
retains its width-N
target logits on CUDA, selects the complete row directory with one argmax
launch and synchronization, and transfers only bounded aggregate facts.
Eager attention, reference layouts and host-materialized stochastic adapters
may be replaced only through the existing typed execution profile. Production
greedy CUDA target-feature capture reduces mHC streams directly into a
transaction-owned token-major device directory and transfers only bounded
status. Feature projection consumes that directory without an intervening
upload or normalized-row download, executes the resident encoded width-N
projection and batched RMSNorm, then binds those rows directly into the draft
core. Device-only candidate and promoted-prefix identities derive from exact
producer, binding, tensor and prefix facts rather than a host array scan. CPU,
audit/forensic and stochastic DSpark retain their explicit host feature oracles.
CUDA final projection also preserves the
pre-normalized BF16 drafter row before RMSNorm, so production no longer
downloads expanded residual streams or recomputes that final stage on the
host. Compatible width-N CUDA output rows already share activation preparation
and one encoded-head execution; incompatible and reference directories retain
an explicit row-local fallback. Batched device selection
now consumes ordered resident logits views with no vocabulary D2H; greedy
DSpark target verification consumes the same result class. Greedy drafting also
keeps the shared-head base rows resident, fuses the encoded Markov projection
with each base row and selects each proposal on CUDA without vocabulary D2H or a
host vocabulary scan. Production greedy CUDA gathers each encoded Markov row
from admitted residency and transfers only its row identifier. The drafter's
normalized and pre-normalized rows remain resident through output-head and
confidence projection, while confidence consumes split hidden/Markov device
views and transfers one scalar. Host Markov decode and confidence remain
reachable only through the CPU, audit/forensic and stochastic references;
stochastic distribution and acceptance still require their bounded device
cutover.
The wave must keep prefix promotion, shape
admission, partial-turn semantics, protocol channels and one model/session
authority unchanged.

Kernel order follows the identity-bound phase roofline ledger. Attention, MoE
and output-head work retain their technical dependencies, but profiling of
prefill, decode, verification, drafting and concurrent serving selects the
next causal bottleneck. Physical-variant exploration first uses role probes,
representative layers, kernel microbenchmarks and bounded logit/acceptance
checks; complete artifacts are emitted only for surviving candidates.

Production attention graph pieces borrow the session execution stream and
defer completion to the existing layer publication barrier. Piecewise mode now
uses one scoped wait per layer instead of one graph-local wait per piece plus
publication. Per-layer device timing belongs to audit and forensic evidence;
that path retains isolated immediate graphs. Production phase accounting keeps
exact launch, movement and completion-synchronization facts. Removing the
remaining layer publication wait requires a later transaction owner that can
defer state and status visibility across the transformer stack.
