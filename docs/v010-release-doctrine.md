# v0.1.0 Release Doctrine

Date: 2026-07-22
Status: normative release contract

## Authority

`PROJECT.md` owns the v0.1.0 product target and active execution sequence. This
document owns the meaning and closure rules of the release gates. The two
documents must change together when release scope changes.

Documentation records implementation truth. It cannot create capability.

## Release Identity

v0.1.0 is the first release in which YVEX generates real text with
DeepSeek-V4-Flash on the DGX Spark CUDA backend from a complete GGUF artifact
produced by YVEX.

Exact source target:

```text
$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash
```

Canonical full target:

```text
deepseek4-v4-flash
```

The target is not yet a supported release artifact because evaluation,
full-model benchmark and release qualification remain open. v0.1.0 does not
include another model family or backend.

## Generation Contract

Release generation is the complete chain:

```text
prompt text
-> real tokenizer
-> full prefill
-> family-correct attention and MoE
-> real KV writes
-> output head and vocabulary logits
-> sampling
-> decode that reads the KV state
-> multiple autoregressive tokens
-> detokenized text
```

The chain must use the same complete YVEX-produced GGUF, runtime descriptor,
CUDA execution path, and state lifecycle that the operator command uses.

The following do not close generation:

- CLI parsing or help output;
- reports or readiness fields;
- fixture token printing;
- synthetic logits or bounded sampling;
- selected tensor or segment execution;
- diagnostic state advancement;
- external GGUF files or external runners.

## Artifact Terminology

- A **tensor proof artifact** contains one tensor or a bounded tensor subset.
  It proves only the explicitly named lower-level property.
- A **complete model artifact** contains every tensor and metadata item required
  to execute the exact model.
- A **supported model artifact** is a complete model artifact that passes
  integrity, materialization, runtime, generation, evaluation, benchmark, and
  release gates.

An unqualified model artifact is never a selected-tensor proof file.
Structural GGUF validity alone does not make a complete or supported model
artifact.

## Closure Rule

v0.1.0 closes only when every gate below passes for the exact source, target,
artifact identity, backend, and operator path.

| Gate | Minimum closure evidence | Cannot close from |
| --- | --- | --- |
| Source | Verified local identity, revision, config, tokenizer, shard inventory, and payload-readable tensor index | path presence, download receipt, header list alone |
| Architecture | Typed attention, position, KV, MoE, norm, output-head, tokenizer, and validation facts | model name, report text, lexical tensor patterns |
| Mapping | Complete unique role map and GGUF name/layout map with required shape constraints | partial inventory, missing-role report, selected subset |
| Transformation | Immutable artifact-neutral plan binds every logical output to exact source contributions, typed operations, output geometry, and payload-range identity | GGUF names alone, quantizer rediscovery, payload reads during planning |
| Qtype and quantization | Exact storage geometry, role policy, arithmetic/refusal truth, reference quantization, and error bounds | qtype names, policy report, storage sizing alone |
| Artifact | Complete YVEX-produced GGUF, identity, global range integrity, deterministic writer, and writer-reader equivalence | tensor proof artifact, external GGUF, file discovery |
| Materialization | All required tensors admitted, placed, transferred, and cleaned up under the DGX Spark residency plan | allocation, estimate, or selected tensor transfer |
| Descriptor | Execution-complete descriptor derived from admitted artifact facts | descriptor report with missing roles |
| CUDA | Fail-closed capability, host binding, implemented kernels, reference parity, failure paths, and cleanup | device probe, module load, no-op entry point |
| Transformer | Full prefill, family-correct attention/position/KV, MoE routing/expert execution, complete layers, final norm | primitive or segment proof |
| Text path | Real tokenizer, output-head vocabulary logits, sampling, KV-backed decode, append/stop, and detokenization | synthetic logits, token IDs printed by fixtures |
| Generation | Multiple autoregressive tokens through the complete operator path with deterministic failure and cleanup evidence | bounded control loop or trace output |
| Evaluation | Repeatable quality, correctness, and regression cases over the release path | primitive comparison or planned harness |
| Benchmark | Reproducible measured record with machine, artifact, qtype, prompt, run count, timing, and memory metadata | estimate, one unrecorded run, target table |
| Release | Clean full validation, artifact guardrail, claim audit, operator transcript, and versioned release record | documentation, passing subset, intent |

Gate closure is monotonic only while its evidence remains valid. A reopened
dependency reopens every downstream gate that relies on it.

## Gate State Ownership

`PROJECT.md` is the sole owner of current gate state, milestone state,
dependencies, and Active Next. This doctrine defines what evidence closes each
gate but does not reproduce the live sequence.

Bounded fixture evidence remains valid only for its named scope. If project
control reopens a dependency, every downstream gate that relies on it reopens;
documentation work alone cannot promote artifact, materialization, descriptor,
transformer, generation, evaluation, benchmark, or release capability.

## Evidence Rules

- Use the lowest true evidence stage from `AGENTS.md`.
- Implementation, tests, command proof, cleanup, and documentation must agree.
- Refusal behavior is part of every gate.
- External execution is comparison evidence only.
- A later-stage result cannot repair missing earlier-stage ownership or facts.
- Release-benchmark evidence is accepted only from the same supported release
  path; lower-stage attention measurements do not close the release benchmark.
- Release evidence must be reproducible from a clean checkout plus the named
  operator-local source and artifact.

## Explicit Non-Claims

YVEX does not currently claim:

- DeepSeek-V4-Flash supported-model or release-qualified status;
- that complete GGUF emission, materialization or admitted attention execution
  alone closes the supported-artifact contract;
- public or remote serving, authentication, TLS, compatibility APIs,
  multi-model hosting, continuous batching, or session persistence over restart;
- evaluation readiness, a full-model release benchmark or release readiness;
- v0.1.0 support for Qwen, Gemma, GLM, Metal, ROCm, or another target.

## Change Control

Changing the exact source, full target, DGX Spark CUDA backend, generation
definition, or required gates requires an explicit scope row plus coordinated
changes to `PROJECT.md`, this doctrine, focused guards, target facts, and
operator acceptance. A report, command, candidate table, or public claim cannot
change release scope implicitly.
