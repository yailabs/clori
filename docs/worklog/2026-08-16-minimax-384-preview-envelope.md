# MiniMax 384 Preview Envelope Checkpoint

| Field | Value |
| --- | --- |
| Date | 2026-08-16 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `a88b741ca17fd53e41de9f29cf72f81424dbc104` |
| Checkpoint | `6c48c52482611070e9880c5edafc69cb7a0d8152` |
| Subsystem | MiniMax media profile and CUDA joint Transformer admission |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; independent numerical conformance; bounded runtime qualification |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

YVEX admitted 192x192 and 256x256 five-second MiniMax previews under a 4,096 packed-row cap.
The 256 envelope had complete fifty-block conformance, but no useful-quality result. Independent
two-evaluation latent evidence subsequently removed the observed early scheduler trajectory as
the most immediate explanation for colored mosaic output.

The released checkpoint instead targets a 768-pixel short edge. A square five-second request uses
about 21,741 packed rows, beyond the current single-GB10 admission and practical dense-attention
envelope.

## Problem

The next visual experiment needed materially more spatial rows without pretending source geometry
was already practical. A 384x384, 124-frame request uses 5,328 video rows, 414 audio rows, and 15
oracle text rows: 5,757 packed rows. With the maximum 256-token prompt it uses 5,998 rows.

Raising a repeated cap without a complete-stack oracle would have exposed a profile whose largest
numeric owner had not executed at that geometry. Advertising the geometry as HD or quality would
also have exceeded the evidence.

## Causal analysis

The exact equal-head CUDA attention path processes 64 query rows at a time and does not retain a
quadratic score matrix. Its temporary storage therefore remains bounded at 5,757 rows; source-dtype
Transformer weights still dominate residency. The 4,096 limit was a prior evidence boundary, not
a source semantic constant.

The common server already selects canonical `WIDTHxHEIGHT` spellings from typed family profiles.
No new parser, runtime, or family-specific server mechanism is required.

## Decision

Raise the MiniMax joint-Transformer admission ceiling to 8,192 only after complete 5,757-row,
fifty-block conformance. Add a distinct `preview-384` profile at 384x384 and 124 frames. Keep
source, draft, HD, FHD, 2K, and 4K refused and preserve the existing preview meanings.

Do not claim recognizable media, model quality, source-scale practicality, or generation speed.

## Implementation

The internal MiniMax family ABI, graph recipe, existing CUDA joint-Transformer owner, and live
fixture bound now agree on 8,192 packed rows. The family media descriptor registers
`preview-384`, raises only the typed maximum canvas to 384x384, and leaves all resource budgets and
component identities unchanged. The common server refusal reports the exact four admitted
profiles.

Focused dialogue tests select 384x384 through the generic registered-profile mechanism, retain
missing fields, close their session cleanly, and keep unsupported tiers fail-closed. The family
host projection exposes the four deterministic profile entries.

## After

The manual BF16 oracle completed all fifty blocks over 5,757 rows in 6 minutes 15 seconds with
1,816,928 KiB peak RSS, about 3.4 GiB device use, and zero swap under a 32 GiB hard limit. The
oracle used exact full attention with 64-query chunks to bound temporary storage without changing
row semantics.

YVEX completed the same request in 3 minutes 25 seconds, admitted 66,280,430,144 resident weight
bytes, reported 2,522,884,864 device bytes and 15,482 kernel launches, reached 65,760,132 KiB peak
RSS, and used zero swap under a 100 GiB hard limit. Video relative L2 was `0.004785907`, cosine
`0.999988564`, and scaled maximum absolute error `0.023739371`. Audio relative L2 was
`0.013636469`, cosine `0.999907030`, and scaled maximum error `0.030424278`.

## Evidence

- The oracle manifest SHA-256 is
  `3196ebe01165ecb46edf5dea569495a7e391832ee423e3d919bc6f1b7f550e00`.
- Video and audio oracle SHA-256 values are
  `058d1fb6b246282b7cd2c9ad9d6601a838277b5a136846e30153878802c2bde5` and
  `c316d4b7a360895fc710978862f0552cd63dfba75cd062246fe94ff0c99cee37`.
- YVEX video and audio SHA-256 values are
  `a310a3ac4c038947b57e08a021ae69bf23b5048698fbda0aa1bbbcc15d252778` and
  `c11cb403861df7ea669da28b05b4c006bb500355b2addeabdfc90fe58afd50ba`.
- The residency identity is
  `8fabd88629b671abf1ca8dd3d659d7ec9f173996039082d7354ff73371ef7308`; the execution identity is
  `cfba384ee8dc1fb619fde3e750191d89b3aadca40dda13a5d235488beb603e2b`.
- Focused `minimax_h3`, `runtime_media`, and `server` unit sections passed.
- Source manifest, ownership, C structure, repository layout, architecture, and tracked-artifact
  guards passed.

## Remaining limitations

- No complete prompt-to-AVI execution has yet qualified `preview-384`; the profile has exact
  Transformer and dialogue evidence only.
- The existing low-resolution media remains visually unrecognizable. This checkpoint establishes
  no prompt fidelity, model quality, useful output, or speed claim.
- The released 768x768 square envelope is roughly 21,741 packed rows, and 1344x768 is roughly
  37,725. Both remain refused and would multiply dense full-attention work materially.
- Longer recurrent conformance beyond two evaluations and useful-resolution VAE qualification
  remain separate evidence boundaries if the 384 media result still diverges.

## Why it matters

YVEX now has a numerically qualified intermediate geometry large enough to produce new visual
evidence without weakening the 128 GiB memory boundary or pretending that native 768p is already
practical. The next real media run can test spatial envelope as a cause instead of repeating the
same 192-pixel experiment.
