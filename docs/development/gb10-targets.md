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
rates. Obtain the exact artifact, mapping, preparation, and current/peak
resource facts from the admitted deployment; an old artifact size or advertised
FP4 ceiling cannot predict current token throughput.

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
| target-only decode | short, width 1 | baseline required | correct sequence; >0 | unadmitted | 20 token/s | 24 token/s first preferred checkpoint | active bytes, 273 GB/s, launch and synchronization depth | medium |
| target-only decode | 256 output tokens, width 1 | baseline required | exact continuation; >0 | unadmitted | 20 token/s | 24 token/s first preferred checkpoint | routed MoE, qtype row execution and remaining transformer work | medium |
| target-only prefill | bounded current fixtures | baseline required | correct prefix; >0 | unadmitted | 20 token/s | 30 token/s | attention class, chunk width, active bytes, 48 SMs | low |
| target-only decode | 12K context, width 1 | unmeasured | exact continuation; >0 | unadmitted | 15 token/s | 20 token/s | context-band state traffic and attention mix | low |
| target-only decode | 64K context, width 1 | unmeasured | exact continuation; >0 | unadmitted | 8 token/s | 12 token/s | compressed/indexer history and capacity-safe state access | low |
| target-only decode | near admitted capacity, width 1 | unmeasured | no overflow; exact stop | unadmitted | 0.80 × short rate | 0.90 × short rate | shape registry and context-bound state traffic | low |
| DSpark short | width 1--5 verification | baseline required | target-equivalent committed sequence | unadmitted | at least target-only | material positive speedup | accepted rows per target sweep, draft and verify cost | medium |
| DSpark no-think | 256 output tokens, width 1--5 verification | baseline required | target-equivalent committed sequence | unadmitted | at least target-only | material positive speedup | measured acceptance distribution and verification width | medium |
| DSpark reasoning | 1,000 output tokens, width 1--5 verification | baseline required | target-equivalent committed sequence | unadmitted | at least target-only | material positive speedup | acceptance and proposal/verification cost | low |
| DSpark low acceptance | width 5 verification | unmeasured | no silent fallback; exact residual sampling | unadmitted | at least 0.90 × target-only | at least 1.00 × target-only | bounded wasted draft/verification work | low |
| physical batched target decode | batch 2–4 | unmeasured; requires admitted physical width | isolated exact sessions | unadmitted | 35 aggregate token/s | 60 aggregate token/s | row batching, queue policy and 48-SM occupancy | low |
| long-context prefill | 12K–64K, admitted chunks | unmeasured | exact state and bounded memory | unadmitted | 50 token/s | 80 token/s | chunk width, attention class, 273 GB/s and launch count | low |

Logical runnable concurrency is separate from the physical batch rows above.
Cooperative session scheduling alone does not qualify a batched-throughput result.

The numerical YVEX and stretch columns are optimization objectives. The
20--24 token/s class is an initial minimum engineering floor: 20 token/s is the
first gate and 24 token/s the first preferred checkpoint, not an optimization
destination. These values are not admissions, forecasts, competitive claims or
release gates. A delivery must replace low-confidence objectives with measured,
comparable evidence or retain them as unmet targets. It may not lower
correctness, precision identity, context, workload or evidence depth to
manufacture a pass.

## Required measurement key

Every result compared with this table binds:

- source revision, artifact, runtime binding, Physical Execution IR, compiled
  execution profile and kernel bundle;
- target-only or DSpark mode, evidence profile, sampling class and accepted
  prefix distribution;
- prompt/corpus identity, context band, input/output lengths, width, batch and
  concurrency;
- host lifecycle, memory placement, driver/toolkit, clocks, power/thermal
  state and machine identity;
- wall/device/host time, movement, state copies, full-array scans, launches,
  waits, synchronizations, memory high-water mark and committed-token counts.

An optimization passes only its named metric and correctness gates. Component
timing cannot promote a full-model rate, and this table cannot promote model
quality or release qualification.

## Retained empirical barrier

A bounded historical investigation at commit
`08edb119681882fdcb73b010acf55defacfc3f59` used a 95,050,210,304-byte artifact,
SHA-256 `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`,
binding `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`,
and specialization
`d6c205636f1c4a333faff8bb6fcae47e8146293154af2d75a09961353100cbe6`.
Its raw evidence pointers and experiment details remain recoverable from the
former DeepSeek representation worklog at documentation checkpoint
`3f4a1c182d35e5a0e163adb81008ae7a366efcc6`; that record was written after the
execution checkpoint above.

The unresolved lesson is representation economics, not a current hardware
ceiling: a faster width-four tile applied to only 14.61% of eligible work and
did not improve complete execution (reported 7.99% regression). Real selected
populations reached at most six, so width eight was not available. A complete
CUDA-owned package copy did not improve target-only/speculative rates; a
managed copy caused swap pressure and invalidated direct comparison.
No candidate from that investigation was retained.

These are historical characterization findings. The worklog's final QA was
invalidated by source movement; neither its old qtype coverage nor its timings
qualify the current tree. The next performance owner must repeat a controlled
baseline, measure physical populations and compulsory bytes, and evaluate
throughput, memory, preparation cost, and numerical effect together.

Current sampling, session-stream synchronization, resource ownership, and
measurement architecture belong to [runtime architecture](../architecture/runtime.md)
and the [runtime contract](../contracts/runtime.md), not this target table.
