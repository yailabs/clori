# MiniMax Source-Square Media Qualification

| Field | Value |
| --- | --- |
| Date | 2026-08-17 |
| Type | checkpoint |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `c5635f1ba8749c4c2a79c45ad38789268a514a39` |
| Checkpoint | `fbcd4f62b5088a0056ae151d73b17d1e9258b504` |
| Subsystem | MiniMax latent execution, staged residency, and media qualification |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | source-square numerical conformance; live CUDA runtime; playable synchronized media; canonical QA |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The native MiniMax vertical could already publish a bounded synchronized AVI, but its 384x384
visual result was a regular colored mosaic. The source-square attention boundary had separately
proved one 21,741-row Omni block against the independent reference contract, including exact
reusable-workspace lifecycle and byte-identical output relative to the preceding admitted path.

That evidence did not establish whether a complete 49-evaluation, 50-block trajectory at 768x768
would fit the GB10 memory envelope, finish without swapping, publish valid media, or repair the
visual behavior seen at smaller geometry.

## Problem

Memory feasibility, transactional media correctness, and useful prompt-conditioned output are
different claims. A smaller run could not determine whether the mosaic came from the reduced
spatial envelope. Conversely, completing a source-square run would not by itself prove visual
semantics unless the decoded frames became recognizable and coherent.

The qualification therefore needed to run the exact admitted source-square path while retaining
enough identity and media evidence to distinguish infrastructure success from model-behavior
failure.

## Causal analysis

The complete 768x768 execution finished without swap or allocation failure, decoded both media
domains, and published a structurally valid synchronized container. This rules out the earlier
memory-capacity and incomplete-publication hypotheses for the admitted square profile.

Seven samples spanning the beginning, middle, and end remained a regular colored tile pattern.
Increasing the geometry from 384x384 to 768x768 therefore did not remove the observed defect. The
current evidence localizes the unresolved boundary to the numerical interpretation between the
iterative latent result and visual reconstruction, including the Visual VAE or their interface.
It does not identify which of those owners is wrong, so assigning the cause more narrowly would
be a guess.

## Decision

Retain the completed source-to-container path as real product and runtime evidence, but do not
promote it to useful prompt-conditioned video or model-quality evidence. Classify recognizable
source-square output as a gate blocker that requires same-boundary numerical localization.

Do not spend another two-and-a-half-hour full trajectory merely to reproduce the mosaic. The next
execution must first obtain an independent upstream latent or Visual VAE boundary comparison that
can discriminate between the remaining owners.

## Implementation

The previously admitted reusable attention workspace carried the complete source-square request
through 49 model evaluations and all 50 Omni blocks per evaluation. The component session staged
and released its residency, the visual and audio decoders produced their streams, and the media
transaction atomically published one AVI.

The resulting file was independently inspected and decoded with GStreamer. A byte-identical copy
was also delivered to the Exon operator video directory over the direct local network. No model
payload or runtime evidence was added to Git.

The MiniMax family record now separates this successful execution and publication evidence from
the unresolved visual correctness claim. It changes progression for useful source-square output
to `repair_same_boundary` while preserving the already accepted bounded playable-file vertical.

## After

| Fact | Observed result |
| --- | --- |
| Request geometry | 768x768, 124 RGB frames, 24 fps |
| Model execution | 49 evaluations; 50 Omni blocks per evaluation |
| CUDA execution | 942,556 kernel launches |
| Wall time | 2 hours 30 minutes 28 seconds |
| Peak device bytes | 12,960,345,604 |
| Peak reusable workspace bytes | 1,485,742,080 |
| Maximum resident set | 67,176,200 KiB |
| Swap | 0 |
| Audio | 165,333 stereo samples per channel, signed 16-bit PCM, 32 kHz |
| Video duration | 5.166666666667 seconds |
| Audio duration | 5.16665625 seconds |
| Stream-duration delta | 10,416.667 ns |
| Published bytes | 220,082,144 |
| File identity | `cf54a70f6854c33d775ef4a624f60e6ecafb54f0302d4277b68943f31bfe48f0` |
| Execution identity | `3089bfbf486c78cfa360c7421ad91031bb4770cfdef110de0821af25184e77c6` |
| Conditioning identity | `e94b983ddf292138fce22a62b91b938f19ac4c4c59b3fa76c1725e1e12dbb726` |
| Latent identity | `4d040e01a2b6cd38708d135ef74ecef30d7fb9ffc673220b6156de57ce4804b0` |
| Container inspection and decode | PASS |
| Recognizable prompt-conditioned scene | FAIL: regular colored mosaic |

The 768x768 file is playable and tightly synchronized. It is not recognizable as the requested
eclipse scene and does not establish correct source-square Visual VAE reconstruction.

## Evidence

Canonical changed-plan QA resolved 109 tests across the required CI, CUDA, fast, numeric,
performance, runtime, sanitizer, and structural lanes. Evidence identity
`eb714855512cc0f1e6bf9f8e7c35a338b440b228d57310c293f1505c54acdd64` records 101 `PASS`,
eight `BLOCKED`, zero `FAIL`, zero `SKIP`, and zero `ERROR` results.

The blocked rows retain missing external asset or baseline prerequisites for DeepSeek live
generation, MiniMax text and transformer oracle fixtures, and performance qualification. They did
not execute or become successful substitutes. Three available MiniMax assets were then exercised
through the same canonical registry:

| Scope | Result | Evidence identity |
| --- | --- | --- |
| MiniMax Audio VAE live artifact | PASS | `2e21fee55d09db93e46f3cac1ea6a264e5007d97547e7566bf4a3e3211ac534c` |
| MiniMax Visual VAE live artifact | PASS | `61fb37505fc54865ef77869baedcc84de52236bb1c7dbabe52d46789062534f3` |
| MiniMax tokenizer live artifact | PASS | `35ad5948fd9e8e5beb1d4331645d26ab404bb60ccb1f51b1ae1b9a26f0c5b242` |

The canonical MiniMax latent and Omni live rows remain `BLOCKED` because their independent oracle
bundle is not configured under the registry contract. Separate current-wave live evidence retains
the accepted 21,741-row one-block conformance and complete source-square media execution; it does
not masquerade as the missing registered oracle evidence.

## Remaining limitations

- The regular 768x768 mosaic blocks progression to useful prompt-conditioned video.
- The exact defect owner among latent interpretation, Visual VAE reconstruction, and their
  interface requires an independent intermediate comparison.
- Canonical MiniMax text, latent, and Omni live rows still require their exact external reference
  assets; their absence remains `BLOCKED` rather than `PASS`.
- The two-and-a-half-hour trajectory is performance characterization, not acceptable generation
  speed or a benchmark baseline.
- This checkpoint does not prove model quality, practical 768p generation, compressed media,
  hosted serving, Ref2VA, Context-IR, Regenerate-2K, or release support.

## Why it matters

The source-square run cleanly separates an infrastructure success from a numerical product defect.
YVEX can execute the admitted workload within the GB10 memory envelope, release its resources,
synchronize audio and video, and publish atomically. The same evidence prevents those successes
from hiding the fact that the requested scene is still unusable, and gives the next repair a
specific numerical boundary instead of another blind full-model run.

## Communication projections

### Short update

YVEX completed the full MiniMax-H3 768x768 source-square trajectory on one GB10 without swapping
and published a playable, synchronized AVI. The frames remain a regular colored mosaic, so useful
prompt-conditioned video is a same-boundary correctness repair rather than a completed quality
claim.

### Article seed

**Possible title:** When a Successful GPU Run Is Still a Failed Video

**Central thesis:** Runtime, media, and model-behavior evidence must remain separate even when one
identity-bound execution produces all three.

- How reusable attention workspace made the source-square run memory-feasible.
- Why playable and synchronized does not mean visually correct.
- Using negative model behavior to narrow the next numerical oracle boundary.
- Why the next test should compare intermediates instead of repeating the entire trajectory.

### Visual candidates

- The identity-bound path from conditioning through latent execution, both VAEs, and atomic AVI
  publication.
- A claim ladder separating memory feasibility, container validity, synchronization, visual
  reconstruction, and model quality.

### Quoteable technical facts

- “The full 768x768 trajectory completed 942,556 CUDA kernel launches without swap.”
- “Audio and video durations differ by 10.4 microseconds in the published AVI.”
- “A playable synchronized file is real runtime evidence, but a regular mosaic is not useful
  prompt-conditioned video.”
