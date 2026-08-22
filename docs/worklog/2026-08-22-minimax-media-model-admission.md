# MiniMax Media Model Admission Repair

| Field | Value |
| --- | --- |
| Date | 2026-08-22 |
| Type | repair |
| Milestone | `R010.MINIMAX.H3.FL2VA.END_TO_END.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `1f6eb90f11813604b9b82574bd85c064f869bdfb` |
| Checkpoint | `2277236d032eac9e0188806b1c18b88428d0d2a5` |
| Subsystem | generic media runtime and persistent server lifecycle |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 with 128 GiB unified memory |
| Evidence | software tests; runtime lifecycle qualification; sanitizer; independent source inspection |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The persistent MiniMax media server admitted a conversational profile and its
component locations, then published `READY` without opening the tokenizer or
any of the four component artifacts. Telemetry truthfully reported zero model
and artifact opens, but that also meant a missing or corrupt component could
remain undiscovered until a completed chat request began generation.

Every request reopened the text encoder, Transformer, Visual VAE, and Audio VAE
views. Payload residency was already staged by phase, which kept the roughly
144 GB source population from being made simultaneously resident on the GB10.
The chat parser also searched for output-format and profile substrings inside
ordinary words, so Italian scene text such as `movimento` or `salta` could be
misclassified as an unsupported format or quality request.

## Problem

The server readiness boundary did not mean that one immutable composite media
model had been admitted. The operator could see an apparently idle process and
an unchanged terminal while no component had yet been checked, then pay both
model admission and execution setup only after submitting a complete request.
Loading every payload eagerly at startup was not a valid repair because the
four admitted artifacts contain about 144 GB of tensor payload and exceed the
machine's 128 GiB unified-memory envelope before activations and workspaces.

The substring parser was a separate product-path correctness defect: creative
prompt words were allowed to masquerade as control terms.

## Causal analysis

The generic DeepSeek server path opens and admits its runtime model before
publishing readiness. The media-specific server path had intentionally kept
all component allocation lazy, but it also omitted the lower-cost immutable
model-open boundary: regular-file snapshots, GGUF metadata, tokenizer state,
tensor tables, component admissions, and one composite identity.

Independent inspection of `antirez/h3.c` commit
`8974cc055ea9c02fcd14cc27dfda3e1027c05153` supports the same distinction.
Its `h3_load_dir()` inventories the FL2VA components and probes the device
without loading all tensor payloads. Generation loads and releases Qwen, DiT,
and decoder phases separately. Its interactive mode additionally offers
identity-keyed conditioning, prepared-DiT, and video-decoder caches; its
low-memory DiT mode alternates two BF16 layer slots with asynchronous SSD
prefetch. The inspected `h3.c` file has SHA-256
`38e1fb2a1ec668dc849fc8f1395f681094267505ad7ea8e3b3b082a70184259f`.

This source evidence does not make that external runtime a YVEX executor and
does not prove that its Metal-specific caching or SSD economics transfer to
CUDA/GB10 unchanged.

## Decision

Give the common media runtime one persistent, identity-bound composite-model
owner. Server startup must open the tokenizer and all four component GGUF
views, perform exact component admission, seal their ordered admissions into
one runtime-model identity, and only then publish readiness.

Retain immutable artifact and tensor views for the process lifetime, while
keeping payload materialization, CUDA residency, workspace, and mutable latent
state phase-scoped. Do not report mapped artifact bytes as resident bytes and
do not retain the Transformer, Qwen encoder, and decoders simultaneously merely
to make startup look eager.

Keep the conversational parser bounded, but require whole control terms rather
than arbitrary substrings. Defer prepared-component caching and two-slot
payload streaming until their GB10 lifecycle, correctness, and workload
economics are independently admitted.

## Implementation

Checkpoint `2277236d032eac9e0188806b1c18b88428d0d2a5` adds an opaque generic media
model lifecycle. Model open validates the immutable request contract, opens
the tokenizer, retains four component views, requires complete admission
identities, accounts their source bytes without claiming residency, and seals
one deterministic composite identity. Generation consumes that opened owner;
the offline API preserves its existing behavior through a bounded ephemeral
open/generate/close wrapper.

The persistent server now performs this model open before creating the ready
media endpoint. Telemetry records one logical model, four opened artifacts,
and one admitted binding while leaving materialization, residency, device
bytes, and uploads at zero. Shutdown closes the persistent model exactly once.

Conversational term matching now requires non-alphanumeric boundaries. MOV is
still refused as an unsupported container, while `movimento`, `salta`, and
`testa` remain prompt text.

## After

`READY` now means that the exact MiniMax media model's tokenizer and all four
weighted component artifacts have been opened and admitted under one runtime
identity. Missing component files fail server startup rather than the first
generation request. The retained mappings are immutable file-backed views;
they do not establish payload residency or create a CUDA context.

At request time, conditioning, Transformer, Visual VAE, and Audio VAE
materialization remain ordered and phase-scoped. The change therefore improves
startup truth and avoids repeated metadata reconstruction without weakening
the existing GB10 no-simultaneous-residency invariant.

## Evidence

- Canonical changed-file QA evidence
  `37f18b0ee0241460efb30cd68e6ccefc8c85277f36b9978d6e915f410239b164`
  resolved six required lanes and 100 tests. It completed with 100 PASS and
  zero FAIL, SKIP, BLOCKED, or ERROR.
- The focused runtime-media test opens one model, performs two deterministic
  generations through the retained views, proves a second independent open
  produces the same model identity, and refuses a request whose sealed
  contract changes.
- The focused server test proves model-open count 1, artifact-open count 4,
  binding-open count 1, and zero materialization, residency-build, and resident
  device bytes before readiness. Shutdown records one model close.
- CLI acceptance proves an empty component root fails startup with model
  admission failure instead of publishing a usable media socket.
- ASan/LeakSanitizer, UBSan, architecture, ownership, layout, documentation,
  operator-registry, runtime, and server obligations passed in the canonical
  QA run.
- Two consecutive compositional builds completed without cleaning, and no new
  tracked model weights or generated artifacts were added.
- The independent `h3.c` inspection established inventory-before-generation,
  separate model phases, optional identity-keyed interactive caches, and the
  two-slot SSD-streaming design. No external code or runtime dependency entered
  the production path.

## Remaining limitations

- The server process that was already running during this repair still uses
  its previously loaded executable. The new readiness behavior applies after
  a normal operator-controlled restart; this delivery did not stop or replace
  that process.
- Component payloads are still materialized per generation phase. YVEX does
  not yet retain a prepared DiT or Visual VAE decoder across requests.
- YVEX does not yet implement antirez's two-slot Transformer SSD stream or its
  streamed Qwen prefetch ring. Their memory/speed trade-offs require a separate
  CUDA/GB10 design and comparable evidence.
- No startup-speed, first-generation latency, warm-generation speed, model
  quality, release readiness, or simultaneous full-residency claim follows
  from this repair.
- This checkpoint must be reconciled with the current canonical `main` before
  further MiniMax implementation proceeds.

## Why it matters

The persistent product surface now distinguishes an admitted composite model
from phase-scoped payload residency: operators receive a truthful readiness
boundary without forcing an impossible 144 GB eager-residency policy onto a
128 GiB machine.

## Communication projections

### Short update

YVEX now admits MiniMax-H3's tokenizer and four component artifacts before the
media server reports ready, while retaining phase-scoped CUDA residency on
GB10. The lifecycle matches the useful separation found in the independent
`h3.c` implementation without importing its Metal runtime or claiming that all
144 GB are resident.

### Quoteable technical facts

- "MiniMax media readiness now requires one admitted composite-model identity."
- "Mapped artifact bytes and resident CUDA bytes remain separate facts."
- "The 144 GB component population remains phase-staged on a 128 GiB GB10."
