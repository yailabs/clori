# MiniMax Direct Media and Verified Reopen Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-24 |
| Type | repair |
| Milestone | `R014.MINIMAX.H3.DIRECT.MEDIA.EXECUTION.REOPEN.REPAIR.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `2af2857f5380807d5b27a42a4b1f31bfe295f413` |
| Checkpoint | `151f752139b7bf440e042c6beb797c9da0f6317d` |
| Subsystem | artifact admission, composite runtime model, hosted media server, local protocol, and operator client |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | unit, integration, sanitizer, canonical QA, real cold/warm admission, and cancelled live execution |
| Comparability | characterization only |
| Publishability | reviewed |

## Before

The first-class hosted MiniMax server opened one tokenizer and four immutable GGUF component
views before READY, but every start authenticated the component bytes by rereading the complete
payload population. The single-artifact runtime already used the generic identity-bound
verified-reopen lease, while the composite runtime always called full artifact-identity
verification. The installed component population totaled 144,029,232,896 bytes.

The media session also implemented a deterministic parameter questionnaire in `server.media`.
It scanned creative text for words and numbers that looked like quality, duration, inference-step,
format, and seed controls, then returned fixed Italian questions as if they were model output. A
turn could therefore complete without calling the native H3 engine. Creative prompt bytes and
execution policy had no truthful boundary.

## Problem

Two product claims were false or incomplete. Warm server startup behaved like a cold trust scan
even when all immutable snapshots had already been authenticated. Chat attachment looked
conversational, but the apparent assistant was a C keyword parser and formatter, not MiniMax-H3
inference. The repair had to reuse the existing generic artifact trust mechanism and make one
creative turn directly invoke the accepted native media generator without adding an agent,
secondary language model, or larger natural-language parser.

## Causal analysis

Component catalog admission combined structural reconciliation and byte verification in one
unconditional operation. The runtime media owner could supply the correct family catalog but had
no generic way to request verified reopen or receive per-component authentication evidence. Cache
policy consequently ended at the single-artifact `runtime.model` path.

The hosted media server separately treated missing execution parameters as dialogue state. Its
`text_has_term`, profile, duration, step, format, seed, parse, and question helpers interpreted
arbitrary prompt bytes as control data. That behavior was deterministic but semantically
misrepresented: MiniMax-H3 Base is a media generator, and the Qwen conditioning component is not
an autoregressive chat assistant.

## Decision

Split generic artifact catalog admission from byte authentication. Family code continues to
select the exact component catalog; the artifact owner authenticates bytes through one options
record carrying the generic reopen-cache root and one evidence record carrying the chosen path.
Missing, stale, or malformed receipts fall back to complete verification. Successful verification
publishes or repairs the canonical receipt; an artifact mismatch still fails closed.

Keep cache policy on media-model open, never on the generation request. The common YVEX paths owner
supplies the same cache root used by the single-artifact runtime. Neither MiniMax family semantics
nor the backend sees a cache path.

Remove parameter dialogue entirely. Each submitted creative turn is opaque model input and uses
one identity-bearing hosted execution policy named `interactive-preview-v1`: profile `preview`,
192x192, 124 frames, two sigma-grid points, AVI, and seed 42. This is an explicit YVEX interactive
test policy, not a source-declared MiniMax default.

## Implementation

`artifact.roundtrip` now exposes structural catalog admission separately from
`yvex_artifact_admission_authenticate`. Authentication records file bytes, bytes actually hashed,
full-hash versus verified-reopen versus fallback-full-hash mode, receipt state, publication or
repair, elapsed time, and cache warning state. The common filesystem owner gained narrow atomic
replacement publication so a valid full verification can repair one malformed receipt without
deleting unrelated cache content.

The composite runtime model consumes those generic options for each weighted component and seals
the same admitted model identity regardless of cold or warm authentication path. Its open summary
retains per-component evidence and aggregate byte counts. Legacy component and single-artifact
callers continue to request full verification when no cache policy is supplied.

The media server no longer owns parameter-dialogue state or prompt keyword selectors. It copies
the submitted prompt as the complete creative request and invokes
`yvex_runtime_media_model_generate` immediately with the typed preset. Protocol v12 adds a typed
media result instead of overloading assistant text with output path and publication facts. Runtime
control events identify request, preset, conditioning, latent iteration, both VAEs, publication,
completion, cancellation, and failure. The foreground and `server log` projections consume the
same events, including bounded per-component cold/warm receipts.

Tests use control-like creative phrases such as `HD stars`, `high desert`, `draft horses`, a
falling `seed`, and numbers embedded in scene prose. None changes the execution preset, and the
prompt copied into the native request remains byte-for-byte equal to the submitted prompt.

## After

The normal server command remains:

```sh
./yvex server minimax-h3-fl2va-runtime-media
```

With no existing MiniMax component receipts, the real server authenticated all four installed
components by full hash and reached READY in 122.801 seconds. A second unchanged start reached
READY in 0.376 seconds. Each component reported `verified-reopen`, and aggregate full-hash bytes
fell from 144,029,232,896 to zero. This is startup characterization from one cold/warm pair, not a
release benchmark or percentage-speedup claim.

`yvex chat --session video` accepted one 195-byte Italian eclipse prompt without technical
parameters. The server emitted `request.started`, the 192x192/124-frame hosted preset, and
`conditioning-start`; the native conditioning path completed 63 tokens with a 1,701-value result.
No parameter question or fabricated assistant text appeared.

The operator requested cancellation and then normal server stop. Cancellation became terminal
after the in-flight text-component materialization returned: `generation.cancelled` was emitted
after 274.363 seconds, the session returned a typed cancelled result, phase memory was released,
the runtime emitted both shutdown events, and no new media file was published. The long delay is
retained as a real responsiveness limitation rather than described as immediate cancellation.

## Quantitative delta

| Component | Cold mode | Cold bytes hashed | Cold time | Warm mode | Warm bytes hashed | Warm time |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Text encoder | full-hash | 66,727,837,152 | 56.819 s | verified-reopen | 0 | 0.001149 s |
| Transformer | full-hash | 66,280,465,664 | 56.932 s | verified-reopen | 0 | 0.000101 s |
| Visual VAE | full-hash | 10,415,528,096 | 8.198 s | verified-reopen | 0 | 0.000099 s |
| Audio VAE | full-hash | 605,401,984 | 0.499 s | verified-reopen | 0 | 0.000120 s |
| Composite total | full-hash | 144,029,232,896 | 122.801 s | 4/4 verified-reopen | 0 | 0.376 s |

The cold and warm runs used the same executable checkpoint, installed component paths, filesystem
snapshots, server command, cache owner, and machine. The figures characterize authentication and
startup only; they do not measure model generation performance.

## Evidence

- Canonical changed-file QA evidence
  `14b48cc5a13871cf32c76c4fa939c93b06694e1c5630044468149f9ade78d0c3`
  records 101 `PASS`, seven asset-unconfigured live rows `BLOCKED`, and zero `FAIL`, `SKIP`, or
  `ERROR`. Both quant and runtime sanitizer lanes passed.
- The seven blocked rows are `live.deepseek.generation`, `live.minimax.audio`,
  `live.minimax.latent`, `live.minimax.omni-transformer-artifact`, `live.minimax.text`,
  `live.minimax.tokenizer`, and `live.minimax.video`. The runner did not have their external asset
  variables configured. No row was weakened, skipped, or relabeled.
- R014's changed behavior was exercised independently of those historical model-conformance rows:
  a real four-component cold open, a real four-component warm open, normal status/model/memory/log
  control, one direct prompt reaching conditioning, terminal cancellation, cleanup, and absence of
  partial publication all ran against the installed MiniMax profile.
- Cold receipts were published independently for artifact identities
  `61407a737bf019cef8f0d786394986d419af957bd23a120f6dcc070128abb7ff`,
  `aa1c84ac801a50f8806b591fb419e60513f0e6bd312b5e3abc5352194a31b992`,
  `29bb1df65227fa05444c4002e18d61934d70d872d8472c4757e93971f9e474cd`, and
  `52a10c9f6f6e3b9b81569a95329f503fcb3cbddb224d12bf7851b4929d02e1c1`.
  Their next open was four independent hits; a per-component stale receipt does not invalidate the
  other three.
- Controlled fixtures proved missing receipt, valid unchanged receipt, stale snapshot, malformed
  receipt repair, malformed receipt plus invalid bytes refusal, one-component selective fallback,
  cache-root absence, cache I/O warning without false admission, and cross-artifact receipt refusal.
- The existing DeepSeek registry entries remain schema-v5 startup-ready with runtime binding v14.
  Generic single-artifact cold fallback and warm verified reopen pass in runtime binding tests; the
  real registered artifact identity `d27b87a9e7c7959c442b0231621588d274f22a8aa916cb05750508cd39ff6f53`
  retained its prior generic receipt. No DeepSeek family or branch state changed.
- Focused protocol, server, artifact, runtime-media, runtime-binding, MiniMax, REPL, layout,
  ownership, architecture, documentation, project-control, no-NVCC, and tracked-weight checks
  passed. Two compositional builds and `git diff --check` passed.

## Remaining limitations

- Cancellation is terminal and cleans up, but the current materialization owner does not poll the
  request cancellation flag while reading and preparing the text component. The acceptance
  cancellation therefore waited 274.363 seconds and caused temporary swap activity before the
  component returned and rollback released memory. No OOM occurred. Interruptible component
  materialization is a later runtime lifecycle/performance owner, not part of this direct-routing
  and verified-reopen repair.
- Verified reopen authenticates unchanged local bytes; it does not materialize payloads, create
  CUDA residency, prepare DiT state, cache a decoder, stream weights, or accelerate model kernels.
- The preset is deliberately one bounded interactive policy. User-selectable typed presets and a
  future explicitly owned context/planning layer are not implied.
- The cancelled execution proves reachability and lifecycle, not a fresh video-quality result.
  Previously accepted Omni, latent, audio, video, and source-scale numerical evidence remains the
  separate authority for those claims.
- Protocol v12 is a local private product cutover. Older v11 clients fail closed rather than
  interpreting the new typed media result.

## Why it matters

Warm composite hosting now reuses the same generic identity-bound trust mechanism as a
single-artifact runtime, while creative prompt bytes reach the native MiniMax engine without being
silently reinterpreted as execution controls. Startup speed no longer depends on rereading 144 GB
of unchanged payload, and the product no longer presents deterministic C control prose as model
conversation.

```text
progression_decision: proceed
downstream_safe: true
downstream_consumer: judge review of direct hosted MiniMax execution and later separately owned performance work
gate blockers: none for direct prompt routing, composite verified reopen, or terminal cancellation
boundary incompleteness: none for R014
evidence gaps: canonical external live rows were not configured in this run; unchanged numerical owners retain prior accepted evidence
deferred depth: user-selectable typed presets and any future context/planning owner
optimization debt: cancellation cannot interrupt in-flight component materialization; model execution remains slow
generalization debt: composite verified reopen has one real four-component family consumer and generic fixture coverage
external blockers: none
required repairs: none before judge review of R014; interruptible materialization belongs to a later authorized boundary
higher-capability non-claims: no video-quality, generation-performance, release, agent, secondary-LLM, caching, or streaming claim
```
