# MiniMax Selective Q8 Transformer Candidate

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | performance |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `257010947a47c5c6d451efb1fe1c5f1836d80be1` |
| Checkpoint | `712293a565b963bda926506e98cb457f7662534a` |
| Subsystem | component compilation and quantization |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; artifact validation; negative numerical conformance |
| Comparability | directly comparable |
| Publishability | reviewed |

## Before

The admitted MiniMax Transformer component retained every source tensor in BF16 or F32. The
source-square 768x768 request fit inside a bounded no-swap cgroup, but the complete run was
stopped before publication because exact dense attention remained too slow to justify several
hours of qualification. The interruption released all MiniMax residency and published no
partial media.

## Problem

The roughly 66 GB source-faithful Transformer dominates staged residency and requires reading a
large artifact before every isolated qualification. A selective 8-bit physical variant could
reduce that boundary, but only if it retained the independent numerical contract. File-size
reduction alone could not admit a runtime path.

## Causal analysis

The main-block attention and MLP matrices are BF16 rank-two weights with row widths compatible
with Q8_0 blocks. Norms, modulation, embeddings, final heads, positional data, token refiners,
and non-BF16 tensors need not be approximated to test this hypothesis. Keeping those tensors in
their source physical type isolates the error introduced by the 200 selected linear weights.

The resulting component is a different physical derivation of the same exact source snapshot.
It therefore needs its own profile and identities, while the runtime must continue to use the
source-faithful artifact until conformance promotes the candidate.

## Decision

Retain the deterministic selective-Q8 compiler and artifact-emission path as an experimental
physical candidate. Do not register the emitted file as a runtime Transformer artifact and do
not run a full 768x768 trajectory with it: the first real source-square block exceeded the
existing video error contract.

## Implementation

The generic identity physical-plan builder now accepts one synchronous bounded decision
override. The MiniMax component adapter exposes the exact
`minimax-h3-transformer-q8_0-v1` profile only for the Transformer, and the family projection
marks only main-block attention and MLP roles as quantizable. The existing `yvex compile quant`
surface accepts the explicit profile and backend while retaining refusal for unknown profiles,
wrong components, invalid sources, and unsupported backends.

No runtime catalog, backend admission, CUDA operation, public C API, or server behavior changed.

## After

Native emission produced a 48,217,105,664-byte GGUF with all 535 Transformer tensors: 200 Q8_0,
322 BF16, and 13 F32. Native roundtrip and the pinned official GGUF reader accepted its
structure. The candidate artifact identity is
`517f0f0aa097452a85577491de708859dbed1225c58ea106ffa58897345a12e0`; its transformation
identity is `5d5a23636b2f074da432254efda4908d05528cdc017deca7a71670893fd8b784`.

One 21,741-row, one-block GB10 comparison then refused the candidate. Video relative L2 was
`0.0385166827` against the admitted `0.02` maximum, cosine was `0.99927958` against the
`0.9998` minimum, and scaled maximum absolute error was about `0.0701` against the `0.02`
maximum. The process completed in 92.70 seconds with 4,981,816 KiB maximum RSS, zero swap, and
no OOM under 76 GiB memory-high and 88 GiB memory-max limits.

## Evidence

- Focused source-payload tests prove that an admitted quantizable identity terminal can receive
  one deterministic physical override while exact terminals remain protected.
- MiniMax family tests bind the candidate profile to only the Transformer attention and MLP
  semantic-role mask.
- CLI tests cover invalid source and wrong-component refusal for the profile.
- Source ownership, repository layout, architecture, and whitespace guards passed.
- The artifact has profile identity
  `c81d6f86ec3939869a999c1ed35e5b581c27ec3ecf4c04895a74747383a7cada` and payload-plan
  identity `9a0ef43e487b7f476d5979a78ac1a6fb623ba8dcf7afae2ba8ce0601dc84227d`.
- The emitted artifact remains an untracked external operator asset. No model weight entered
  Git.

## Remaining limitations

- Q8_0 for all selected attention and MLP matrices is numerically inadmissible for the current
  runtime contract; the failure is not a model-quality evaluation.
- Audio comparison was not promoted after the video lane had already failed the joint block.
- No complete Q8 latent trajectory or media generation was attempted.
- A future candidate needs narrower role selection, another physical format, or stronger
  reference evidence before any runtime admission.
- Source-square full attention and source-faithful artifact I/O remain performance debt.

## Why it matters

YVEX can now derive and validate a compact MiniMax Transformer candidate without weakening
source identity or exact-role ownership. More importantly, independent numerical evidence stops
that smaller artifact from silently becoming the production executor and producing faster but
less trustworthy media.
