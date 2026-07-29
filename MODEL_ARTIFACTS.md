# YVEX Model Artifacts

Date: 2026-07-29
Status: artifact policy

Complete and supported model artifacts are external operator assets. They are
never committed to this repository. The repository may contain only tiny GGUF
fixtures under `tests/`.

## Terminology

| Term | Meaning |
| --- | --- |
| Tensor proof artifact | A file containing one tensor or a bounded tensor subset, used only to prove a named parser, range, materialization, primitive, or reference property. |
| Complete model artifact | A file containing every tensor and metadata item required to execute one exact model. |
| Supported model artifact | A complete model artifact that passes integrity, materialization, runtime, generation, evaluation, benchmark, and release gates. |

The unqualified term "model artifact" does not refer to a selected-tensor proof
file. A structurally valid GGUF is not necessarily complete or supported.

Tensor byte accounting consumes the canonical row-aware GGUF qtype owner.
Block-quantized shapes require `ne[0]` to divide exactly by the block width;
flattened element-count alignment and partial-row padding are not admitted.
Known storage geometry does not imply a decoder, quantizer, emitter, compute
kernel, complete artifact, or supported artifact.

The logical model is independent of this artifact contract. An immutable
transformation plan derives a physical model variant from exact logical source
contributions under explicit format, precision, hardware, memory, quality, and
workload constraints. An artifact serializes one such variant; GGUF is the
v0.1.0 release lowering, not the identity of the logical model.

## v0.1.0 Target

Exact source:

```text
$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash
```

Canonical full target:

```text
deepseek4-v4-flash
```

Required artifact outcome:

```text
a complete GGUF for DeepSeek-V4-Flash produced by YVEX
```

Three complete DeepSeek-V4-Flash model artifacts currently exist outside the
repository: the source-faithful profile, the Q8_0/Q2_K baseline, and the
DS4-like IQ2_XXS/Q2_K candidate. All three passed their owned writer/reader
admission. The baseline has complete hosted prompt-to-text, retained-session,
CPU, and admitted GB10 CUDA evidence. The DS4-like candidate passed
variant-adaptive materialization, binding, mixed-qtype CPU/CUDA execution, and
generation smoke.

These are complete artifacts, but none is yet a supported model artifact.
Evaluation, the release-path full-model benchmark, release-profile selection,
and release qualification remain incomplete as recorded in `PROJECT.md`. The
DS4-like candidate is not promoted to the release profile by size or execution
smoke alone.

## Source Payload Handoff

The canonical source payload session is construction input, not an artifact.
It consumes one exact verifier result and its retained header snapshot, binds
each source tensor to a securely admitted shard and checked absolute byte
range, and delivers deterministic bounded chunks through a transactional sink.
It neither reparses source headers nor retains whole tensors.

For the pinned DeepSeek source, authoritative Hugging Face Git LFS SHA-256
values establish `upstream_payload_verified` only after every complete shard
digest matches. If an equivalent provider has no authoritative payload digest,
the distinct `local_payload_snapshot_sealed` class may identify the local byte
snapshot but may not claim upstream verification. The ordered aggregate
payload identity and per-shard trust facts are published atomically in the
canonical source manifest outside the repository.

This handoff makes exact source bytes available to a later transformation
executor. It does not define transformation semantics, perform datatype
conversion or expert aggregation, select qtypes, quantize, encode GGUF, emit a
complete artifact, materialize tensors, or execute a runtime.

## Native Structural Reader

The canonical native reader parses GGUF v3 container, metadata, and tensor
directory records through bounded positioned reads. Its immutable parsed view
owns decoded strings, arrays, tensor facts, and lookup indexes. Structural
inspection neither maps nor copies the complete artifact and reads zero tensor
payload bytes.

Parser budgets bound hostile counts and declared strings or arrays while
admitting the known DeepSeek-scale directory. Duplicate identifiers, malformed
records, refused qtypes, arithmetic failures, and short reads produce typed
non-success results. Typed human, machine, and integrity projections consume
that operational result; building a report cannot convert a rejection into an
accepted artifact.

The canonical global layout validator consumes the immutable reader view and
same opened artifact snapshot. It requires nonzero power-of-two alignment, the
first tensor at relative offset zero, exact directory-order padded
continuation, zero directory/tensor padding, an exact aggregate data span, and
no trailing bytes. It reads only padding intervals and reports zero tensor
payload bytes read. Typed layout acceptance is a container property, not a
complete-model or supported-artifact claim.

## Admission Contract

A complete model artifact must record or prove:

- exact source and target identity;
- architecture and tokenizer metadata;
- every required global, layer, attention, position, KV, MoE, expert, norm, and
  output-head tensor role;
- exact qtype, shape, layout, offset, alignment, range, and byte accounting;
- deterministic writer output and writer-reader equivalence;
- artifact identity and corruption refusal;
- materialization and runtime descriptor compatibility.

A supported model artifact additionally requires the runtime, generation,
evaluation, benchmark, and release gates in `docs/v010-release-doctrine.md`.

## Existing Proof Files

Legacy selected DeepSeek files and aliases may still exist outside the
repository while their owning decommission rows are pending. They are tensor
proof artifacts only. Their digest, range, materialization, or primitive
evidence must be named by the specific proof and must not be promoted to model
support.

The canonical decommission obligations and consuming milestones are recorded in
`PROJECT.md`. This policy does not retain an artifact card or historical
validation catalogue for those files.

## Repository Guardrail

```sh
git ls-files '*.safetensors' '*.bin' '*.dat'
git ls-files '*.gguf'
```

Expected result:

- no tracked safetensors, bin, or dat model payloads;
- tracked GGUF files are tiny test fixtures only;
- no local registries, reports, source downloads, emitted artifacts, logs, or
  generated backend outputs are tracked.

## Non-Claims

This policy records the completed quantization, GGUF emission, roundtrip,
variant-adaptive materialization, runtime, and hosted-generation evidence named
above. It does not claim that any complete artifact is a supported model
artifact, nor does it claim a selected release profile, model behavior or
quality evaluation, a release-path full-model benchmark, or release readiness.
