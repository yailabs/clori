# Generic CUDA Transposed Conv1D Direct Enumeration

| Field | Value |
| --- | --- |
| Date | 2026-08-28 |
| Type | performance |
| Milestone | `YVEX.GENERIC.CUDA.TRANSPOSED.CONV.INTEGRATION.0` |
| Branch | `models1` |
| Baseline | `f1aa6ea16d980eea98015baba4beea3bf9734082` |
| Checkpoint | `7dc810a9ac549ec16787f5f8743e960f23ca5abb` |
| Subsystem | generic CUDA convolution execution and kernel-bundle admission |
| Model family | MiniMax-H3 FL2VA consumer; DeepSeek common-owner regression |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | independent numeric references, Audio VAE oracle and repeat, Nsight Systems, canonical QA |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The generic CUDA owner exposed one `yvex_conv1d_f32` kernel for ordinary and transposed Conv1D.
One argument selected the algorithm inside that kernel. The ordinary path enumerated the bounded
input neighborhood for each output position. The transposed path instead assigned one thread to
each output value, then scanned every input position and every kernel coordinate before rejecting
all coordinates that did not project to that output.

The retained request-shaped MiniMax Audio VAE characterization used the exact artifact and latent
fixture retained below. Its two dedicated callback runs took `141.104438` and `141.257596` seconds.
Nsight attributed `138.143235` seconds to the seven source-derived transposed Conv1D operations.
The relevant production decoder owners and live runner did not change between that characterization
checkpoint and this delivery's baseline.

## Problem

The transposed branch executed work proportional to every output position multiplied by every
input position and kernel coordinate, although only a narrow interval of input positions can
contribute to one output. At the seven Audio VAE geometries it enumerated approximately
84.862 trillion inner iterations for approximately 7.018 billion useful multiply-accumulates.
That asymptotic mismatch made seven otherwise ordinary decoder operations consume more than two
minutes on GB10.

Keeping both algorithms behind one divergent kernel also made their separate algorithm and
register profiles invisible to kernel-bundle admission and later tuning.

## Causal analysis

For output position `o`, padding `p`, input position `i`, stride `s`, kernel coordinate `k`, and
dilation `d`, a contribution exists exactly when:

```text
o + p = i * s + k * d
```

Positive `d` makes `k` unique for fixed `o` and `i`. The contributing input interval is bounded by
`o + p - (K - 1) * d <= i * s <= o + p`. Clipping that interval to the admitted input extent and
checking divisibility by `d` removes only rejected coordinates. Iterating input channels ascending
and retained input positions ascending preserves the relative order of every actual F32 addition.

The preceding isolated experiment proved this property mechanically on generic stride, dilation,
padding, bias, scale, and boundary cases. It also produced byte-identical baseline and candidate
outputs for each of the seven exact production geometries. This integration then preserved the
complete historical request-shaped PCM digest through the production component path.

## Decision

Keep ordinary and transposed convolution as separate CUDA algorithms:

```text
ordinary Conv1D   -> yvex_conv1d_f32
transposed Conv1D -> yvex_conv1d_transposed_f32
```

The compiler/runtime geometry already carries the transposed fact. The host dispatch therefore
selects one separately admitted function without a family switch or kernel-internal algorithm
branch. The transposed kernel enumerates only possible contributors while retaining the accepted
accumulation and publication order.

Rejected alternatives were MiniMax-specific CUDA, a compensating numeric change, a widened
tolerance, and a combined mega-kernel with an internal transposed branch. None was required by the
mathematical contract or ownership model.

## Implementation

The generic CUDA convolution owner gained the separate direct-contributor kernel and selects it
from typed convolution geometry. Kernel-bundle admission now requires both function symbols and
publishes neither capability if either one is unavailable. Ordinary Conv1D retains its previous
arithmetic body and no longer receives an unused algorithm selector.

A new generic CUDA numeric test uses an independent exhaustive F32 reference. Two ordinary and
nine transposed cases cover bias and scale presence, stride, dilation, padding, output padding,
multiple channels, sparse contributors, and boundaries. Each case runs twice and requires byte
identity. The bundle fault test separately removes both convolution symbols and requires atomic
admission rollback.

No runtime, server, scheduler, protocol, residency, family recipe, tensor layout, precision,
weight, or artifact contract changed.

## Quantitative delta

The request-shaped workload is `[2,32,207]` input and `[2,1,165600]` output. The previous and new
Nsight captures use the same artifact, fixture, decoder schedule, GB10, production component API,
and CUDA synchronization policy. Times below are GPU kernel durations for the seven production
transposed operations.

| Stage | Geometry `[B,Cin,Tin] -> [B,Cout,Tout]` | Before | After | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 0 | `[2,1024,207] -> [2,512,1035]` | 2.538604 s | 16.429792 ms | 154.512x |
| 1 | `[2,512,1035] -> [2,256,5175]` | 15.751729 s | 16.903264 ms | 931.875x |
| 2 | `[2,256,5175] -> [2,128,10350]` | 23.834013 s | 7.952128 ms | 2,997.187x |
| 3 | `[2,128,10350] -> [2,64,20700]` | 23.962933 s | 4.036128 ms | 5,937.109x |
| 4 | `[2,64,20700] -> [2,32,41400]` | 24.020815 s | 2.217056 ms | 10,834.555x |
| 5 | `[2,32,41400] -> [2,16,82800]` | 24.017164 s | 1.174528 ms | 20,448.354x |
| 6 | `[2,16,82800] -> [2,8,165600]` | 24.017977 s | 0.641184 ms | 37,458.790x |
| **Total** | seven production shapes | **138.143235 s** | **49.354080 ms** | **2,799.024x** |

Two unprofiled production callbacks completed in `2.787840` and `2.901029` seconds. Their median
is `2.8444345` seconds versus the directly comparable prior median of `141.181017` seconds: a
`49.634x` callback speedup and `97.985%` wall-time reduction. The profiled callback completed in
`2.770580` seconds.

The after-state profile attributes `625.613120` milliseconds to 129 ordinary Conv1D kernels and
`49.354080` milliseconds to the seven transposed kernels. Ordinary Conv1D is now 84.6% of captured
decoder kernel time. That is evidence for a possible later boundary, not authorization or proof of
another optimization.

## After

All seven production transposed operations execute the direct algorithm through the common CUDA
backend. Their aggregate production GPU time is within 1.622 milliseconds of the preceding
isolated candidate measurement. The full request-shaped output is byte-identical across both new
runs and to the historical output, and the component execution identity is unchanged.

The independent one-step Audio VAE oracle remains inside its registered tolerance. Kernel-bundle
admission is fail-closed for the new symbol. A no-NVCC build still produces the host product and
refuses CUDA execution rather than substituting another path.

This is a generic algorithmic performance repair demonstrated by MiniMax. It does not claim a new
media quality level, full-generation benchmark, release readiness, caching, streaming, fusion, or
quantization capability.

## Evidence

The stable executable source commit is `7dc810a9ac549ec16787f5f8743e960f23ca5abb`; its source delta
is empty. The Audio VAE runner SHA-256 is
`80a9a3f852f1a0aaee09242085a23b9ea9d94cfbb0940a0b33d030fb766df5bb`, and the SM121 convolution
cubin SHA-256 is `eb117215133060b53cde497eb331e3d38e162853433597ae75d5a28c71a1159b`.

The exact Audio VAE artifact identity is
`52a10c9f6f6e3b9b81569a95329f503fcb3cbddb224d12bf7851b4929d02e1c1`. The request-shaped latent
fixture identity is `f51a0eb06f333ed6bb6ea2478401d3b3f65abd1e5c24d4a9c435c37069414c44`.
Both new outputs and the retained historical output have SHA-256
`87d4cb46904afce2c7381823894812d63b42096c259bdff126f262da826cb23d`; all seal execution identity
`565795cb2adfcb444c291b54c30410e7843d2a8f303cd0260d28ef5e6166f895`.

The independent `[1,32,1]` oracle produced output identity
`95b82eae8ea5fce6ac8b3b6f69a83ea4d8e70e300bb1bdfcd29e9edc983004c2` and maximum absolute error
`3.7252903e-7`, below the unchanged `1e-5` threshold. Compute Sanitizer memcheck reported zero
memory errors for the generic convolution matrix. Its first default run reported the loader's
intentional missing-symbol probes as CUDA API diagnostics; repeating with API-error reporting
disabled retained all memory checks and closed at zero errors.

Nsight Systems 2025.3.2 report identity is
`dfd8d9b8bd6972e7b035fd83cff606e0dcfdfb428a51fc5c516b0614ae6e14a8`. The kernel-summary and
per-launch trace identities are `7c73692d8c4090664caca17583977b4660f742d4e3499118b2766ee1e74566cd`
and `db3cf90bd15dc0ad4896a8750c487cdca36b09d366c0a56e79636405735de12e`. Raw reports, decoded PCM,
logs, and benchmark baselines remain external operator evidence and are not tracked.

Canonical changed-file QA required 106 tests across CUDA, fast, numeric, performance, runtime,
sanitizer, and structural lanes. It completed with `106 PASS`, zero blocked, failed, skipped, or
errored rows. Run identity is
`edb7e0bedac95b34d8d8cf8cfb680565a9900d0fc556c17cbcab256ba4c6fccd`; build identity is
`026eb5aaf4970cf117f5752494c9aa2dc48d17700fa62d1089cf09d8f38ce1ef`. Source stability was true.
The run includes native CUDA/reference execution, no-NVCC refusal, bounded DeepSeek live
generation, fresh identity-bound performance evidence, ASan/LeakSanitizer, UBSan, and all resolved
structural and unit obligations.

## Remaining limitations

The timing is one directly comparable GB10 component workload, not a release benchmark or a claim
about full media generation. The retained 387.566-second interactive preview predates this repair;
no new end-to-end video was generated merely to restate the measured component delta.

The ordinary Conv1D kernels now dominate decoder GPU time, but their algorithm has not been
localized against a new candidate. Context creation, allocation, module loading, other operations,
and cleanup remain part of the approximately 2.1 seconds outside captured decoder kernel time.
Neither observation permits automatic follow-on optimization.

## Why it matters

YVEX removed more than 138 seconds from the exact request-shaped Audio VAE callback by deleting
mathematically impossible work at the common CUDA owner, without changing any accepted output bit.

```text
progression_decision: proceed
downstream_safe: true for current generic CUDA and MiniMax Audio VAE consumers
downstream_consumer: separately authorized evidence-derived performance work
gate blockers: none
boundary incompleteness: none
evidence gaps: no new full media-generation timing or quality observation was required
deferred depth: ordinary Conv1D and remaining host lifecycle costs require separate localization
optimization debt: ordinary Conv1D is now the dominant captured Audio VAE kernel class
generalization debt: no known second family currently consumes transposed Conv1D at production scale
external blockers: none
required repairs: none
higher-capability non-claims: no media quality, release, caching, streaming, fusion, or quantization claim
```
