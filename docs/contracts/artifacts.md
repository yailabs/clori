# Artifact and Admission Contract

Status: normative implemented contract

Authority: artifact terminology, complete-artifact admission, repository
payload policy, and the boundary between artifact, materialization, runtime,
and release support. Producers are compilation/GGUF owners; consumers are
artifact admission, materialization, runtime binding, and operator inspection.

Complete and supported artifacts are external operator assets. They are
never committed to this repository. The repository may contain only tiny GGUF
fixtures under `tests/`.

## Terminology

| Term | Meaning |
| --- | --- |
| Tensor proof artifact | A file containing one tensor or a bounded tensor subset, used only to prove a named parser, range, materialization, primitive, or reference property. |
| Complete artifact | A file containing every tensor and metadata item required to execute one exact logical model. |
| Supported artifact | A complete artifact that passes integrity, materialization, runtime, generation, evaluation, benchmark, and release-qualification gates for its declared scope. |

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

Current DeepSeek physical variants and their admitted evidence are recorded in
the [family technical record](../model-families/deepseek-v4-flash.md).
Version-specific release requirements belong to the
[v0.1 readiness contract](../releases/v0.1.md). Neither record changes this
general admission contract.

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
complete-artifact or supported-artifact claim.

## Admission Contract

A complete artifact must record or prove:

- exact source and target identity;
- architecture and tokenizer metadata;
- every required global, layer, attention, position, KV, MoE, expert, norm, and
  output-head tensor role;
- exact qtype, shape, layout, offset, alignment, range, and byte accounting;
- deterministic writer output and writer-reader equivalence;
- artifact identity and corruption refusal;
- materialization and runtime descriptor compatibility.

A supported artifact additionally requires the runtime, generation,
evaluation, benchmark, and release gates in
[Release Doctrine](../releases/doctrine.md).

## Existing Proof Files

Legacy selected DeepSeek files and aliases may still exist outside the
repository while their owning decommission rows are pending. They are tensor
proof artifacts only. Their digest, range, materialization, or primitive
evidence must be named by the specific proof and must not be promoted to model
support.

This contract does not retain an artifact card or historical validation
catalogue for those files. Their retired chronology is recoverable from Git.

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

This contract does not claim that any complete artifact is a supported
artifact, nor does it claim a selected release profile, model behavior or
quality evaluation, a release-path full-model benchmark, or release readiness.
