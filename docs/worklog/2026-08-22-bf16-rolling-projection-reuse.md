# Shared BF16 Activation Loads for Rolling Projections

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | performance |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `feature/deepseek-v4-flash` |
| Baseline | `dd69d71b1b6d6fa804bdcca1a0dd73659fa39866` |
| Checkpoint | `b564de5b1f6c4af04baeaed1069b69af265497e6` |
| Subsystem | exact CUDA encoded-attention execution |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | numerical conformance; CUDA qualification; live runtime qualification; performance |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The accepted encoded-attention path used Physical Execution IR v4 and runtime binding v14. For
each main or index rolling-state update, it submitted two independent qtype matvec operations:
the KV projection and the gate/score projection. Both admitted weights were BF16, had the same
shape and consumed the same F32 activation, but each kernel loaded that activation separately.

A fresh ten-sample width-one control on the fixed 15-token target-only fixture measured a median
of 9.242 token/s. The corresponding Nsight Systems profile contained 49,839 kernel launches and
2,352.651 ms of CUDA kernel time. `yvex_qtype_matvec` accounted for 663.044 ms across 9,005 calls.

## Problem

The two projections were already one typed rolling operation at the backend boundary, but their
equivalent microkernel execution did not exploit their shared activation. This created both a
launch tax and redundant device reads inside a frequently repeated attention stage.

Merely putting two independent tasks in one launch was not sufficient. An initial paired-launch
experiment reduced the launch count to 47,855 but retained approximately the same complete-request
device time as the preceding profile. Its apparently higher warm throughput was therefore not
accepted as causal evidence.

## Causal analysis

The two projection rows are independent dot products over one activation. A warp can retain each
activation value after one load, update two independent F32 accumulators in the original element
order, and perform the same shuffle reduction separately for each accumulator. This removes one
activation load stream without changing either dot-product order.

The accepted profile replaced 3,968 generic qtype matvec calls with 1,984 paired calls. Across two
checkpoint profiles, remaining `yvex_qtype_matvec` time averaged 400.795 ms and the paired kernel
averaged 158.413 ms. Their combined 559.208 ms was 103.836 ms below the control qtype total. The
complete profile improved by a larger 153.086 ms on average; that additional difference is an
observed whole-request effect rather than being assigned wholly to the pair kernel.

The current DS4 source at commit `84cc882352757baf628a1776badf7cc54d584e28` was inspected for
its paired compressor execution invariant. The comparison was classified as:

- **ADOPT:** reuse one activation across compatible compressor projections;
- **ADAPT:** keep YVEX's typed rolling operation, exact BF16 weights, independent F32 accumulation
  and existing publication contract;
- **REJECT:** copy DS4's family-specific execution architecture or treat launch coalescing alone as
  evidence of a useful physical improvement.

## Decision

Admit one exact backend microkernel for the two BF16 projections already owned by a single rolling
attention operation. Use it only when both admitted weights are BF16; retain the ordinary qtype
matvec path for other physical representations.

This is equivalent microkernel selection. It does not change grouping, execution width, workspace,
state lifetime, publication, representation or model semantics. Physical Execution IR v4 and
runtime binding v14 therefore remain authoritative and unchanged.

## Implementation

Checkpoint `b564de5b1f6c4af04baeaed1069b69af265497e6` binds an SM121-native
`yvex_attention_bf16_pair` kernel into the admitted encoded-attention bundle. One warp owns one
row from each matrix, loads each activation value once, and updates two independent accumulators.
Both outputs are published into the same pre-existing rolling buffers.

The host selector consumes the already validated rolling geometry and chooses the pair only for
the exact BF16 case. CUDA does not reconstruct family topology or physical planning. The focused
oracle uses deliberately different weight matrices and compares each paired output byte-for-byte
against the ordinary production BF16 matvec, including a nine-row tail case.

## After

The directly comparable ten-sample target-only median improved from 9.242 to 9.437 token/s, a
2.112% increase. The control range was 9.162--9.318 token/s; the checkpoint range was
9.358--9.540 token/s. Every sample produced the same 15-token output and `length` stop reason.

Two checkpoint profiles measured 2,206.592 and 2,192.538 ms of CUDA kernel time, for a
2,199.565 ms mean. Compared with the 2,352.651 ms control, this is a 6.507% reduction. Kernel
launches fell from 49,839 to 47,855, a 3.981% reduction. The paired kernel executed 1,984 times in
each profile and preserved both projection results exactly.

The final ten-sample DSpark characterization measured a 10.598 token/s median with a
9.842--10.668 token/s range and the same bounded output. It is characterization only because a
fresh DSpark before/after pair was not retained for this checkpoint.

## Quantitative delta

| Fact | Before | After | Delta | Evidence class |
| --- | ---: | ---: | ---: | --- |
| Target-only warm median | 9.242 token/s | 9.437 token/s | +2.112% | directly comparable, 10 samples |
| Target-only warm range | 9.162--9.318 | 9.358--9.540 token/s | shifted higher | directly comparable |
| Complete-request CUDA kernel time | 2,352.651 ms | 2,199.565 ms mean | -6.507% | directly comparable profile |
| Complete-request kernel launches | 49,839 | 47,855 | -3.981% | directly comparable profile |
| Generic qtype matvec | 663.044 ms / 9,005 calls | 400.795 ms / 5,037 calls | -39.552% symbol time | directly comparable profile |
| Paired BF16 projection | not present | 158.413 ms / 1,984 calls | new exact owner | directly comparable profile |
| Qtype plus paired owner | 663.044 ms | 559.208 ms | -15.661% | directly comparable profile |
| DSpark warm median | not promoted as a delta | 10.598 token/s | characterization only | 10 samples |
| Physical IR / binding | v4 / v14 | v4 / v14 | unchanged | identity evidence |
| Production Tensor Core coverage | 0% | 0% | unchanged | selected production profile |

## Evidence

- Canonical QA evidence
  `dd316af6327c95b332541dae0846fd48953cece4d31aa530ab2d2940d5544ed3`
  resolved 101 obligations and completed with 100 PASS, one FAIL and zero SKIP, BLOCKED or ERROR.
  CUDA-native, no-NVCC, complete live DeepSeek generation, both sanitizer lanes, every structural
  gate and all resolved unit owners passed. The sole failure was the performance runner refusing a
  non-empty external evidence directory before execution.
- Canonical performance rerun evidence
  `4bdbcba5a3e40fb1d32da62cb4dae808fc48510bac82132d2fc9cfe2ea6c1026`
  passed `performance.runtime` on a new empty identity-bound evidence root. It retained eager,
  piecewise and full baselines and their directly comparable repetitions.
- Focused `make check-cuda` passed the production BF16 pair oracle, CUDA CLI smoke, complete
  43-layer CPU/CUDA attention comparison, byte-identical repeat, 12 fault cases, cleanup,
  cancellation and missing-symbol refusal. The 43-layer RMSE remained
  `5.6856851103341334e-06`.
- Structural QA evidence
  `2facdb18e651460d1e99a048e9b4845a57274872f5918d925d773e81e8598703`
  passed all 14 structural tests after the canonical C inventory ratchet was updated.
- The control profile digest was
  `bb9adb2156f588804630b955cd3bb2e72d7fce0056d36495645fba82288f442c`.
  The two checkpoint profile digests were
  `99c6b79992a2e0ec5fc80f6c78df06ebeb4521171a69b53033040b7117bc60ca` and
  `746c6284902c8fb7b5a60a63ff10812338009fb50c1523ea93514e826a6f52c9`.
- Warm target-only A/B used the same artifact, runtime binding, hardware, context, prompt,
  generation mode, 15-token bound, temperature and output. Each side used ten retained samples.
- Complete artifact identity remained
  `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`.
  Runtime binding identity remained
  `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`.
- Raw profiles, response JSON, generated benchmark tables/charts, the complete artifact and the
  runtime binding remain untracked external operator evidence.

## Remaining limitations

- The pair kernel is an exact width-one microkernel improvement inside the existing physical
  operation mix. It does not create a wider execution batch, change publication granularity or
  establish Tensor Core execution.
- The complete-request profile includes prefill and generation; it does not establish a new
  decode-only GPU-ms/token floor. The retained single-session structural target remains open.
- Generic qtype matvec and grouped MoE up/down remain the largest measured device owners after the
  repair. This checkpoint does not prove which one should own the next GB10 patch without a fresh
  post-checkpoint attribution.
- Production Tensor Core coverage remains zero. Continuous batching throughput and single-user
  latency remain distinct problems.
- The 20--24 token/s single-session objective, deep-context qualification, persistence, complete
  batching qualification, packaging, model-quality evaluation and release qualification remain
  open. No GB10 closure or public benchmark claim is made.

## Why it matters

YVEX converted two frequently repeated BF16 projections from separate activation load streams
into one exact cooperative operation, reducing both device work and launch fragmentation without
moving physical policy into CUDA or changing model output.

## Communication projections

### Short update

YVEX now evaluates two compatible rolling BF16 projections in one warp while loading their shared
activation once. On the same GB10 fixture, complete-request CUDA kernel time fell 6.507% and warm
target-only throughput improved 2.112%, with byte-identical projection outputs and unchanged
Physical IR v4 / binding v14.

### Longer post seed

1. Two rolling-state projections consumed the same activation through separate qtype matvecs.
2. A launch-only pairing experiment reduced launches but did not reduce the device floor.
3. The accepted kernel shares each activation load while preserving two independent accumulation
   and reduction orders.
4. Generic qtype calls fell by 3,968 and one paired call replaced each compatible projection pair.
5. Full CUDA, live generation and sanitizer evidence preserved behavior; the result remains a
   component checkpoint, not GB10 closure.

### Article seed

**Title:** When Kernel Fusion Is Not Enough: Reusing the Activation Load

**Thesis:** Collapsing launches is only useful when the new physical unit eliminates real device
work while preserving the compiled operation's numerical and publication contracts.

Suggested sections:

1. The duplicated rolling-projection activation stream.
2. Why launch-only pairing failed the profiler test.
3. Two independent accumulators over one loaded activation.
4. Physical IR authority versus equivalent backend microkernel selection.
5. Whole-model A/B, numerical oracle and remaining device owners.

Strongest evidence: 1,984 paired calls replaced 3,968 generic matvec calls; complete-request
kernel time fell from 2,352.651 to a 2,199.565 ms mean while the paired outputs remained
byte-identical to ordinary BF16 matvec.

### Visual candidates

- separate activation streams versus one-load/two-accumulator warp diagram;
- launch-only rejected experiment versus shared-load accepted experiment;
- control/candidate kernel-time and launch-count table;
- qtype symbol time plus paired-owner decomposition.

### Quoteable technical facts

- "One paired BF16 call now replaces two rolling qtype matvec calls without changing either
  projection's accumulation order."
- "Complete-request CUDA kernel time fell 6.507% while launches fell 3.981%."
- "Physical Execution IR v4 and runtime binding v14 remain unchanged."
