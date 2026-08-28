# DeepSeek GB10 Matrix/Tile Representation Barrier

| Field | Value |
| --- | --- |
| Date | 2026-08-28 |
| Type | checkpoint |
| Milestone | `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` |
| Branch | `models1` |
| Baseline | `f1aa6ea16d980eea98015baba4beea3bf9734082` |
| Checkpoint | `08edb119681882fdcb73b010acf55defacfc3f59` |
| Subsystem | DeepSeek physical execution, generic CUDA qtype/MoE execution, and engine resource economics |
| Model family | DeepSeek-V4-Flash-DSpark |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 / 128 GiB unified memory |
| Evidence | controlled live characterization; causal CUDA profile; qtype numerical probes; complete-model placement experiments |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The retained production regime opened the authenticated 95,050,210,304-byte GGUF as canonical
mapped package bytes and executed IQ2_XXS, Q2_K and MXFP4 weights through the narrow DP4A-class
CUDA path. Artifact identity
`d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`, runtime-binding
v14 identity `31cfc96974c972568efc6f99593b35998c876b0801b3f44253ef05988f78d61b`, runtime-model
identity `cfc84385c7db8c6cf01a3c940707818e11dace23b8222a4d0bf102690f3e12a8`, and specialization
identity `d6c205636f1c4a333faff8bb6fcae47e8146293154af2d75a09961353100cbe6` were retained.

The directly comparable realignment characterization measured approximately 9.545 token/s
target-only short, 7.420 token/s target-only 256, 10.550 token/s DSpark short and 9.670 token/s
DSpark 256. No selective derived cache, full derived MoE layout or production Tensor Core regime
was admitted. Earlier output-head, routed-down and complete routed-MoE copies had already failed
complete-model economics.

## Problem

The first 20 token/s target-only floor requires reducing a roughly 100 ms useful-token interval to
at most 50 ms. Another narrow-row shape selector cannot supply that reduction. This checkpoint had
to determine whether a compiler-sealed matrix/tile regime could obtain material single-session
width from real expert populations or from a byte-neutral prepared layout, and whether moving the
complete encoded package into CUDA-owned storage materially changed GB10 execution economics.

The investigation had to preserve the package/deployment boundary, real expert-major worklists,
numerical contracts, resource accounting and engine load/unload lifecycle. It could not duplicate
rows, infer DeepSeek inside CUDA, or retain a dormant candidate after rejection.

## Causal analysis

A fresh exact-tree target-only profile measured 1,488.368 ms across 14 steady decode advances,
106.312 ms wall time per token. CUDA kernels accounted for 1,197.708 ms, or 85.551 ms per token.
The decode roofline record attributed 39,699,284,304 active device bytes and 902,059,320 transfer
bytes across 15 work units, 27.435 GB/s measured effective throughput, 35,235 layer kernels and 45
synchronizations. Occupancy was not measured; the evidence explicitly reports the occupancy fact
missing rather than estimating it.

The measured decode interval attributed approximately 27.81 ms/token to attention qtype work,
19.54 ms/token to routed gate/up/down work, 4.89 ms/token to shared experts, 8.61 ms/token to the
mHC pre-residual operation and 6.38 ms/token to the output head. The complete Nsight capture
ranked grouped routed up at 392.597 ms, qtype matvec at 292.323 ms, grouped routed down at
211.237 ms, MXFP4 Q8 rows at 190.881 ms, grouped qtype at 190.501 ms, mHC pre at 155.261 ms,
attention BF16 pairs at 146.890 ms and the output head at 96.202 ms.

The routed gate/up operation was already one fused generic CUDA operation that reuses one
quantized activation and consumes the canonical expert-major worklist. There was no missing
gate/up launch fusion to recover. Target-only single-session decode presented a true population
width of one for the relevant matrix admission boundary; fabricating width would have changed the
semantic population. Production Tensor Core launches remained zero.

The prior real-width Tensor Core experiment already supplied the relevant crossover control. At
width four its exact focused MoE tile was 39.99% faster, but only 554 of 3,792 routed pairs,
14.61%, were eligible and complete-model throughput regressed 7.99%. Raising the threshold to
eight admitted no work because the observed maximum bucket population was six. The current
profile therefore did not justify repeating a lower crossover over the same sparse population.

Two bounded representation probes then isolated layout value without paying complete-model cost.
An IQ2_XXS aligned representation improved a single entry from 184.6 to 196.8 GB/s, 6.6%, while
paired gate/up reached 211.8 GB/s, 7.6% above two aligned entries. A fused variant reached
210.4 GB/s and all variants passed the existing quantized numerical contract. A byte-neutral Q2_K
alignment improved 447.0 to 471.9 GB/s, 5.6%, and was bit-exact. Projecting those component deltas
onto the measured mix yielded only low-single-digit end-to-end headroom, approximately 1.8 ms per
token, far short of the roughly 56 ms required for 20 token/s.

The complete package was then tested under two placement policies. A full managed allocation used
approximately 95 GB of resident CUDA-managed storage, loaded in 2:27.06 and increased `pswpout` by
1,410,233 pages and `pswpin` by 2,190,487 pages during load, approximately 5.38 GiB out and
8.36 GiB in. Its five short target samples had median 7.98 token/s. The arithmetic delta from the
9.62 mapped control is -17.0%, but the managed probe used a shorter six-token prompt rather than
the retained 16-token prompt and became bimodal as pages warmed. It is therefore characterization,
not a directly comparable throughput result. The swap, load cost and unstable execution economics
are independently sufficient to reject it.

A byte-identical `cuMemAlloc` copy retained the canonical mapping and added 95,050,210,304 prepared
device bytes. Target-only load took 3:07.97; DSpark load took 3:05.24. The target short median was
9.03 token/s, 6.1% below the 9.62 mapped baseline. DSpark short measured 10.48 token/s, 0.8% below
10.56. Available memory remained about 28.8--29.7 GB and `pswpout` did not increase, so this was
not a swap-driven false negative: direct full-device placement itself did not improve the actual
generation path. Both tested engine loads rebuilt the complete prepared copy; no warm prepared
reuse was established. Unload took about one second, released all prepared resources and left the
host alive.

## Decision

Retain the canonical artifact-mapped narrow DP4A regime and reject every candidate from this
checkpoint. Do not retain the aligned primitive, a full managed placement, a full device copy, a
new Tensor Core threshold, or a prepared resource merely because the refounded resource catalog
can represent one.

Classify the result as **D: representation barrier proven**. The remaining material gain requires
a different canonical physical representation or precision/quantization regime that makes SM121
matrix execution economical at the real single-session population. That representation must
replace or materially compress canonical bytes rather than duplicate another 72--95 GB beside
them. This checkpoint does not authorize such a package-format or numerical-contract change.

## Implementation

No performance source, selector, resource kind, binding field or dormant CUDA fast path was
retained. Candidate implementations were introduced only long enough to prove numerical behavior,
resource lifecycle and complete-model economics, then removed from the production tree. PEIR v5,
runtime binding v15 with explicit v14 import, engine specialization, expert worklists, executable
batches and the typed engine resource catalog remain unchanged.

Raw profiles, SQLite databases, generated benchmark files, candidate binaries and model outputs
remain untracked identity-bound operator evidence. The retained execution checkpoint is the clean
tree `096f5bab0e4fe8b32b2620ffee322560cfb08c27` at commit
`08edb119681882fdcb73b010acf55defacfc3f59`.

## After

The current production execution regime remains authenticated artifact-mapped IQ2_XXS/Q2_K/MXFP4
weights with narrow DP4A-class CUDA execution and zero production Tensor Core coverage. No extra
prepared representation is loaded, no ordinary inference path relies on swap and package identity
is unchanged.

The fresh short target-only median is 9.620 token/s over ten warm samples; target-only 256 is
7.570 token/s over three samples. DSpark short is 10.560 token/s over ten samples and DSpark 256 is
10.050 token/s over three samples. These are controlled characterizations, not release benchmarks.
The short and 256 target-only/DSpark outputs are byte-identical within their matched fixtures.

## Quantitative delta

| Regime | Samples | Median token/s | Matched control | Result |
| --- | ---: | ---: | ---: | --- |
| Retained target-only short | 10 | 9.620 | realignment 9.545 | characterization |
| Retained target-only 256 | 3 | 7.570 | realignment 7.420 | characterization |
| Retained DSpark short | 10 | 10.560 | realignment 10.550 | characterization |
| Retained DSpark 256 | 3 | 10.050 | realignment 9.670 | characterization |
| IQ2_XXS aligned primitive | 2,000 iterations | 196.8 GB/s | 184.6 GB/s | reject; insufficient model effect |
| IQ2_XXS paired gate/up | 2,000 iterations | 211.8 GB/s | 196.8 GB/s per aligned entry | reject; insufficient model effect |
| Q2_K aligned primitive | bounded probe | 471.9 GB/s | 447.0 GB/s | reject; insufficient model effect |
| Full managed package | 5 | 7.98 | 9.62 mapped | reject; non-comparable prompt, bimodal result, swap pressure |
| Full device copy, target | 5 | 9.03 | 9.62 mapped | reject; -6.1% |
| Full device copy, DSpark | 5 | 10.48 | 10.56 mapped | reject; -0.8% |

## Evidence

The stable baseline and candidate evidence is rooted in the external operator directory
`2026-08-28-08edb11-matrix-tile-baseline`. Profile identities include:

- offline generation/roofline JSON
  `e5fc16eccc2f7285a5726a95161ed2ebff2669a51f25e38f1d272f3e59b83755`;
- kernel-shape inventory
  `e6e4a21c2d57de56518f19aff8aad2d16ae334cc71f745f6198a4546784f4f09`;
- Nsight summary
  `fb2bf3ecb2d2f45d21d202c3eb1314aaea9aa90c6fa9fb0a6f34ec696b9ecf92`;
- tensor inventory
  `f76333706197006d9d4c74bb676009ccfc89a13303e52e26ead50b99e5b340ef`;
- IQ2_XXS aligned probe
  `261df8a33a061cb43116a5fcc5de7b3cc8a5ff746dcd88d1afee41b2e89f39b6`;
- Q2_K aligned probe
  `50be8302f7d46b8638c474747849a281452ba078066db080349cd96d6d940f59`.

All ten short target-only outputs and all five device-copy target outputs have SHA-256
`63d9219a600685084840a92bd9487c5292ad17504da934d7584af5a4b55f6d97`. All three retained
256-token target-only and DSpark outputs have SHA-256
`097e9e9643b117df4029271c21b0ab99c57eb3867892e2a43d832841498ef367`.

The retained target-only and DSpark lanes used the explicit greedy argmax strategy. YVEX models
greedy selection separately from stochastic temperature sampling: its admitted policy keeps the
neutral `temperature=1.0`, `top_p=1.0`, `top_k=0`, no seed and no stochastic draw. These runs did
not approximate greedy behavior by setting temperature to zero.

Canonical changed-file QA requires 106 tests across CUDA, fast, numeric, performance, runtime,
sanitizer and structural lanes. The first exact-tree run recorded 104 PASS and two correctly
BLOCKED live rows because the asset variables were not configured. A second run configured the
real artifact, binding and an empty external benchmark directory; live DeepSeek generation passed
in 564.859 seconds, but a concurrent tokenizer/image delivery changed the shared source delta
during the run. Source stability changed from clean
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` to dirty
`2949a85d09b6be639807bafd64d02d8e39c091d7c60fc297bf5b9ba90ea2a25c`, so run
`39833de2fd9f864846bdc22f3fd31c1e7f00d01e635586dd4b83d7178bcfb4a0` is explicitly invalid
and supports no retained QA claim. Final exact-tree QA remains an evidence gap until the concurrent
delivery stabilizes and the combined tree is rerun.

## Remaining limitations

- `target_20_tok_s` and `preferred_24_tok_s` remain failed. This checkpoint is not optimization
  milestone closure, model evaluation, release qualification or a public benchmark.
- Production Tensor Core coverage remains zero. The existing exact Tensor Core implementation is
  not economical at the true width-one single-session population.
- The evidence does not prove a hardware ceiling. It proves that the current canonical
  IQ2_XXS/Q2_K/MXFP4 representation and tested duplicate placements cannot economically reach the
  first floor.
- A hardware-native package representation, different precision regime or quantization contract
  is a separate compiler/package/numerical boundary and requires explicit authorization.
- Occupancy was not measured in the retained profile. No occupancy claim is made.
- Final source-stable combined QA is pending after concurrent source stabilization.

## Why it matters

The checkpoint prevents a visually attractive matrix/cache architecture from becoming production
debt when its bounded gain cannot close the end-to-end gap and its complete-model form either
regresses or consumes another model-sized allocation. Future work can start at the proven
representation boundary instead of repeating narrow-row or duplicate-cache experiments.

```text
progression_decision: complete_evidence pending a source-stable combined QA rerun
downstream_safe: false until that exact-tree qualification completes
downstream_consumer: separately authorized package/precision representation work
gate blockers: none in the retained execution regime
boundary incompleteness: none in the retained artifact-mapped production path
evidence gaps: final combined QA was invalidated by concurrent source movement
deferred depth: hardware-native package/precision representation requires a separate boundary
optimization debt: 20 token/s floor and 24 token/s preferred floor remain unmet
generalization debt: not applicable; no new production mechanism was retained
external blockers: concurrent source delivery must stabilize before exact-tree QA
required repairs: rerun canonical 106-test plan on the final stable combined tree
higher-capability non-claims: no new Tensor Core regime, package format, model benchmark, evaluation or release claim
```
