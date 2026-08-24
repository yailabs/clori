# MiniMax First-Class Hosted Runtime

| Field | Value |
| --- | --- |
| Date | 2026-08-24 |
| Type | repair |
| Milestone | `R013.MINIMAX.H3.FIRST_CLASS.HOSTED.RUNTIME.0` |
| Branch | `feature/minimax-h3` |
| Baseline | `2df3b84cc840dfca8b38f6fc387a833169b5598e` |
| Checkpoint | `b1cb37080718b937f61af5fbd2e2542110b654ba` |
| Subsystem | local model registry, persistent server, local protocol, and operator client |
| Model family | MiniMax-H3 FL2VA |
| Hardware | NVIDIA DGX Spark / GB10 / SM121 |
| Evidence | real composite admission, local control plane, chat negotiation, canonical QA |
| Comparability | not-applicable |
| Publishability | reviewed |

## Before

The local registry described startup exclusively as one artifact, one serialized runtime binding,
one target, one backend, one generation mode, and one positive text context. The installed
`minimax-h3-fl2va-runtime-media` entry therefore used Audio VAE as its catalog anchor but had no
complete startup profile. `yvex model list` rendered `BACKEND=-`, `CONTEXT=0`, and `STARTUP=no`.

The server CLI contained a media escape path: if a family adapter exposed media execution, it
bypassed normal startup-profile validation and accepted operator-supplied component and output
roots. A user had to reconstruct deployment details and run the server with
`--media-artifact-root`, `--output-root`, and frequently redundant backend or mode flags.

The admitted composite media owner correctly opened the tokenizer and four component GGUF views
before readiness, while keeping payload materialization and CUDA residency phase-scoped. However,
the server summary labeled the media-profile identity as a serialized runtime-binding identity and
the source identity as a singular artifact identity. Local protocol v11 also required a text
capacity plan for every READY runtime, so status, model, and memory clients rejected the otherwise
valid READY media summary as an unexpected response.

## Problem

YVEX had a qualified native MiniMax engine and a persistent media endpoint, but not one coherent
hosted-product contract. Registry startup described only text models, the CLI bypassed that
contract for media, and protocol validation universalized text-only capacity and identity facts.
The result was a model that could be started only with internal deployment knowledge and could not
be inspected through the normal control plane after readiness.

The repair could not be a MiniMax condition around protocol validation. Hosted models need a
generic way to distinguish singular-artifact and composite deployments while preserving exact
identity meanings and all existing DeepSeek runtime-binding v14 behavior.

## Causal analysis

Three owners expressed incompatible assumptions:

1. Registry schema v4 considered a startup profile complete only when artifact, binding, target,
   backend, mode, and positive context were all present.
2. The media server bypassed this validator and resolved component topology from the family adapter
   only after the operator supplied a component root.
3. Protocol v11 treated `runtime_ready` as synonymous with text-generation capacity readiness and
   rejected a media summary without required/unreserved capacity bytes and a capacity-plan
   identity.

The media server then populated the required-looking identity fields with unrelated facts. That
made the wire frame structurally valid only by making its semantic contents false. Removing the
capacity requirement globally would have weakened the text runtime. Fabricating one-byte capacity,
one fake runtime binding, or one singular artifact would have preserved the same defect under new
values.

## Decision

Version the local registry to schema v5 and make startup profiles explicitly typed:

- `single-artifact` retains artifact, serialized runtime binding, target, backend, text generation
  mode, and positive context;
- `composite` retains an installed component root, target, backend, and capability mode, with no
  serialized runtime binding and no text context.

Older v1 through v4 registries remain readable. An absent profile kind in an older complete entry
retains the single-artifact interpretation. Composite validation admits an absolute readable,
searchable, non-symlink installation directory and refuses a fake binding, non-media mode, non-CUDA
backend, or positive text context.

Keep local protocol v11 because its existing optional identity and capacity fields can represent
both contracts without reinterpretation. Validation becomes capability-aware: text readiness still
requires singular artifact, runtime binding, physical variant, runtime model, and capacity-plan
identities; media readiness requires the real composite runtime-model and media-profile identities
and requires all text-only identity/capacity fields to remain absent.

Resolve the default publication root through the common YVEX path owner. Normal startup uses
`$YVEX_DATA_DIR/media`, falling back to `$HOME/.local/share/yvex/media`, while the media publication
owner performs the final absolute-directory, ownership, and non-symlink admission. Explicit roots
remain engineering overrides rather than normal product requirements.

## Implementation

The public registry view and its internal owned representation now carry `runtime_profile` and
`runtime_installation`. The schema v5 reader/writer persists both fields transactionally. The model
CLI admits the two typed startup shapes and projects `MODE` separately from `CONTEXT`; a media
profile renders context as unavailable instead of inventing an LLM capacity.

Server profile resolution no longer consults a family adapter to bypass registry validation. It
validates every selected model, recognizes media only from a complete composite profile, and
resolves the installation root, CUDA backend, media mode, and default output root before invoking
the existing family media adapter. The accepted MiniMax tokenizer, Qwen conditioning, Omni,
Visual VAE, Audio VAE, staged residency, and AVI publication implementation did not change.

Media server options carry zero context, prefill chunk, and maximum-new-token values. Generic
server admission accepts that shape only for media and retains the positive text requirements for
target-only and DSpark. The READY media summary publishes:

- composite runtime-model identity
  `c9c8f8684058ba3d46df39ae64a8de55e74e0c6cbae9b6eb3ea29f1024dc369b`;
- media-profile identity
  `7799e9f4b039027bf8e61c2ab763c64cb7a083e11abeda085f11773ae5a1bef0`;
- no serialized runtime-binding identity;
- no singular artifact identity;
- no text capacity-plan identity or byte counts.

The status, model, memory, and chat renderers project media facts explicitly. The foreground banner
names a composite four-component model and its resolved component and output roots instead of
presenting Audio VAE as the hosted model or printing a fake text context.

The existing local MiniMax entry was republished as a composite profile against the installed
identity-bound component root. No unrelated DeepSeek entry changed; all eight existing DeepSeek
profiles remained startup-ready with their serialized binding and 4,096-token context semantics.

## After

`./yvex model list` reports MiniMax as `minimax-h3`, `cuda`, `media`, context unavailable, and
`STARTUP=yes`. The only required startup command is:

```sh
./yvex server minimax-h3-fl2va-runtime-media
```

The real command authenticated and admitted the installed component set in 191 seconds. It reached
READY with about 68 MiB current RSS and 160 MiB peak RSS, while host-resident, device-resident, and
mapped payload counters remained zero. No CUDA context or simultaneous 144 GB payload residency
was claimed at startup.

Across the real local protocol:

- `yvex server status` reported READY, CUDA, media, context unavailable, and one model open;
- `yvex server model` reported the composite runtime-model and media-profile identities with
  artifact and binding unavailable;
- `yvex server memory` separated host, device, mapped, current RSS, and peak RSS and marked text
  capacity unavailable;
- `yvex server log` observed the canonical event stream;
- `yvex chat --session video` attached to the resident media runtime;
- one Italian Porsche-in-rain request returned the next parameter-negotiation question without
  starting media generation;
- `yvex server stop` emitted runtime shutdown start/complete and the foreground process exited
  successfully.

The resolved default output root was `/home/dgmothx/.local/share/yvex/media`. It was an owned
absolute directory with mode 0775 and was not a symlink. No video generation was required or
started for this hosted-product acceptance.

## Evidence

- Canonical changed-file QA evidence
  `f56258900a32733b7c849b7352fb8776835212775c08a11bc6008373fe333d95` records 101 `PASS` and
  zero `FAIL`, `SKIP`, `BLOCKED`, or `ERROR` across integration, protocol, CLI, numeric,
  sanitizer, structural, no-NVCC, server, runtime, registry, and DeepSeek common owners.
- Focused fast evidence `7ecef7cd6dc6ac54cc0621f6a03b5e475833434e1f1590c372a6c190079b5bcc`
  records 80 `PASS`; focused structural/no-NVCC evidence
  `0a468009a10f1b41fb3b1f220fe03397915a668729dd56b042110d0218d085f9` records 14 `PASS`.
- Focused runtime sanitizer evidence
  `33fbf52d8475f96b45ef91f6917b40e583e3ae46db3f19928da652d30076da47` passed. Quant and runtime
  sanitizers also passed in the canonical report.
- One canonical attempt recorded 100 PASS and one CUDA illegal-address failure while an unrelated
  DeepSeek process occupied about 73% of unified memory. After that external workload ended,
  `unit.runtime_binding` passed twice under evidence
  `1015e64db8c4403f8099d9bbde384c5c0679e2e175e8966dd6055850248d96bd` and
  `4a6d5335fe48a4c37e2ba273b92f90c87468cfc1b6e55184f0e975dc7078ae16`, then passed inside the
  clean 101-test canonical run. The unrelated process was neither stopped nor modified.
- Registry, protocol, server, CLI, model-list, output-root override, incomplete-profile refusal,
  fake-identity refusal, DeepSeek fixture, and REPL tests passed. Two consecutive compositional
  builds completed without cleaning.
- Operator registry generation/check, documentation, project control, architecture, source
  ownership, layout, topology, `git diff --check`, and tracked-weight scans passed.

## Rejected alternatives

- A media-only bypass in generic registry or protocol code was rejected because it would preserve
  the incorrect universal hosted-model abstraction.
- A fake serialized runtime binding, singular artifact identity, or one-byte capacity plan was
  rejected because field names retain their declared semantics.
- A protocol-version bump was rejected because v11 already represents optional identity facts and
  can validate them by admitted capability without wire reinterpretation.
- Hardcoding the installed MiniMax path in source or family documentation was rejected because
  installation location is local deployment state.
- Inferring the component root from the Audio VAE filename or parent directories was rejected
  because family topology and local placement have separate owners.
- Systemd/cgroup wrappers and explicit component paths were rejected as normal startup
  requirements. They remain outside the hosted-model contract.
- A source-scale generation was rejected because it would not test this server/registry/protocol
  repair and the numerical media engine was unchanged.

## Remaining limitations

- Startup authenticates about 144 GB of file-backed component data and took 191 seconds on this
  machine. This repair makes readiness truthful; it does not claim startup performance
  optimization.
- Payload materialization and CUDA residency remain phase-scoped during generation. No prepared
  DiT, decoder cache, SSD prefetch, two-slot streaming, quantization, or performance mechanism was
  added.
- The chat turn proves attachment and typed negotiation, not a new creative-context compiler or a
  fresh source-scale video-quality result.
- AVI remains the admitted publication container. No MP4 or alternative encoder support follows
  from this repair.

## Why it matters

MiniMax now participates in the same normal YVEX hosted-product lifecycle as a text model without
pretending that both products have identical physical composition or readiness evidence. The
registry owns deployment shape, the family owns irreducible component semantics, the server owns
one persistent lifecycle, and protocol fields retain their real identities. Operators can start,
inspect, chat with, and stop the admitted composite model without reconstructing internal paths or
receiving a control-plane response that the client cannot validate.

```text
progression_decision: proceed
downstream_safe: true
downstream_consumer: subsequent judge-reviewed MiniMax product or performance boundary
gate blockers: none
boundary incompleteness: none for first-class hosted startup and negotiation
evidence gaps: none for this hosted-product repair
deferred depth: creative-context redesign, source-scale generation, alternate containers
optimization debt: 191-second composite authentication; later measured startup owner
generalization debt: composite profile has one admitted media-family consumer; generic shape is tested independently
external blockers: none
required repairs: none before normal MiniMax hosted use
higher-capability non-claims: no fresh media-quality, performance, release, caching, or streaming claim
```
