# MiniMax Chat Media Closure

| Field | Value |
| --- | --- |
| Date | 2026-08-14 |
| Type | closure |
| Milestone | `R011.MINIMAX.H3.RECANONICALIZE.CHAT.MEDIA.RUNTIME.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `aae16ce5bcf5dff64ec65cf5580883dea5b24b26` |
| Checkpoint | `521b75a5936521e9547729cf5dad462cc5749582` |
| Subsystem | model family, graph family, generic media runtime, server media, and operator CLI |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; sanitizer evidence; runtime qualification; operator acceptance; independent media inspection |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The recanonicalized MiniMax branch could execute the accepted native FL2VA path through the
bounded offline `yvex execute media generate` surface. That path admitted the exact component
artifacts, ran Qwen3-VL conditioning, fifty Omni Transformer blocks, both VAE decoders, and
published synchronized media. It did not make that capability available through the normal
persistent `yvex server MODEL` and `yvex chat` product flow.

The server already owned typed conversational media state and the runtime already owned the
native component executor. Starting a server, however, still assumed one registry entry with a
sealed runtime binding. The composite FL2VA target instead needs four independently admitted
component artifacts and family-authored geometry while keeping allocation lazy enough for the
GB10 memory envelope.

## Problem

An operator had to reconstruct a long engineering command containing artifact paths, geometry,
memory limits, scheduler points, and output settings. Natural-language chat could not select the
MiniMax media capability, retain an incomplete request, ask for supported choices, or return the
published file path. Wiring the offline command directly into chat would have duplicated
admission and execution semantics and coupled generic server code to MiniMax implementation
details.

The live acceptance run also exposed one parsing defect: in a reply containing `32x32` before
`5 secondi`, the duration parser could associate the unit with the first number and interpret the
resolution as duration.

## Causal analysis

The missing boundary was not another generator. It was a typed projection from family facts and
graph execution facts into the generic runtime media host profile. Model ownership already knew
component identities, qualified tiers, latent geometry, media rates, and resource limits. Graph
ownership already knew the exact executable callbacks and packed-row limit. Those facts had no
narrow family-neutral ABI through which the server could build one lazy composite profile.

The duration defect came from searching for a unit anywhere after a number instead of requiring
the unit to be adjacent to that candidate number. The square resolution syntax therefore became
an accidental duration prefix.

## Decision

Introduce one narrow internal media-target contract consumed by compiler, graph, runtime, and
server owners. The model family supplies irreducible FL2VA target facts; the graph family supplies
the already admitted native execution recipe; the generic runtime validates and seals their
combination. The server selects this capability through the existing family adapter and invokes
the existing media executor. No MiniMax-specific server, daemon, chat path, or second generator
is introduced.

Keep component allocation lazy until the conversational request is complete. Keep the offline
media command as the lower-level engineering lane. Treat the registry entry as the explicit
family selector while the typed profile admits all four component artifacts from the configured
immutable root. Require an adjacent duration unit and retain all unsupported choices as typed
refusals rather than inferred defaults.

## Implementation

The component-variant adapter now carries a media target profile and media execution recipe.
MiniMax model ownership projects the exact target, source identity, four relative component
artifacts, qualified `smoke` and `preview` tiers, frame/audio rates, VAE geometry, scheduler
bounds, and host/device/workspace budgets. MiniMax graph ownership projects the existing native
text, latent, Transformer, Visual VAE, and Audio VAE callbacks without moving family semantics
into generic runtime code.

The generic runtime media owner validates profile schema, path containment, required callbacks,
tier geometry, packed-row bounds, component paths, and output ownership before publishing a
host profile. The standard server command recognizes the admitted family capability, requires
CUDA and an explicit artifact/output root, remains unallocated during parameter negotiation,
and configures the existing server media owner. Chat returns the atomically published AVI path.

Focused server and CLI tests cover profile construction, deterministic facts, missing and
traversing paths, unsupported families, CPU/OpenAI refusal, lazy startup, conversational request
completion, and the resolution-versus-duration regression. The generic REPL fixture was also
repaired separately to assert the current protocol version rather than the retired version.

## After

An operator can start the persistent MiniMax media server with `yvex server`, enter `yvex chat`,
ask naturally for a video, answer the supported tier, duration, solver-point, container, and seed
questions, and receive the path to a playable synchronized file. The chat and offline paths share
the same native runtime executor and component artifacts. Negotiation does not create a CUDA
context or retain model weights; the accepted live request released staged component residency
after completion.

This closes the first operator-reachable native prompt-to-media product flow for the admitted
MiniMax-H3 FL2VA weights. It establishes execution and publication, not visual quality,
high-resolution practicality, model evaluation, or performance readiness.

## Evidence

- The real product flow started `yvex server minimax-h3-fl2va-runtime-media`, entered
  `yvex chat --session eclipse-r011-live`, submitted an eclipse request, completed the prompted
  choices, and returned one atomically published AVI path.
- The output SHA-256 was
  `f384f998957d00e42606b1bf12e5863211107871973f7fc68a4d0d0eda9f4365`; its size was
  1,048,544 bytes.
- Independent GStreamer discovery and playback admitted a seekable 5.166666666-second AVI with
  uncompressed RGB video at 32x32 and 24 frames per second plus stereo S16 PCM audio at 32 kHz.
- Demuxed media contained 124 video frames and 165,333 audio samples per channel. The measured
  stream-duration delta was 10,416 nanoseconds.
- The live run completed in 560.36 seconds. Peak resident memory was 62.57 GiB, below the
  76 GiB high and 88 GiB hard process limits; no OOM occurred. Post-request host/device residency
  returned to 0.00 GiB and process RSS to approximately 0.50 GiB.
- `make test-runtime-asan`, `make test-runtime-ubsan`, and the no-`nvcc` fail-closed lane passed.
  The sanitizer lanes included the server, protocol, runtime, OpenAI adapter, REPL, and tiny
  artifact-to-generation vertical.
- Focused server unit and CLI tests, tiny vertical, REPL PTY, source-manifest validation, and two
  consecutive documentation and repository guardrail passes succeeded. Tracked-artifact scans
  found no model weights; only the established small GGUF format/refusal fixtures remain.
- Visual inspection found colored block structure rather than a recognizable eclipse. That is
  negative quality evidence and prevents a model-quality claim.

## Quantitative delta

| Fact | Before | After |
| --- | ---: | ---: |
| Natural chat requests reaching native MiniMax media execution | 0 | 1 live accepted flow |
| Playable synchronized files returned through chat | 0 | 1 live AVI |
| Live accepted geometry | not applicable | 32x32, 124 frames, 24 fps |
| Live A/V duration delta | not applicable | 10,416 ns |
| Peak resident memory for the accepted run | not measured on this product path | 62.57 GiB |

## Remaining limitations

- The live product qualification covers only the `smoke` 32x32 tier, AVI, 5 seconds, two solver
  points, and seed 42. The admitted `preview` profile was not requalified through chat in this
  closure. HD, FHD, 4K, 768p, other containers, and longer outputs are not claimed.
- The published frame is not a recognizable eclipse. No MiniMax model-quality, prompt fidelity,
  upstream parity, or useful-generation claim follows from this execution proof.
- A 560.36-second smoke generation is characterization, not an acceptable-speed claim. Kernel,
  attention, residency, quantization, and high-resolution optimization remain later passes.
- The artifact selector and component roots are operator configuration. A future generic model
  installation flow may make that setup less manual without changing the admitted runtime path.
- The closure does not establish Ref2VA, Context-IR, Regenerate-2K, commercial eligibility, or
  release support.

## Why it matters

MiniMax media generation is now reached through the same persistent YVEX server and chat product
surface as other model capabilities while preserving native execution, typed admission, bounded
memory, and truthful capability limits.
