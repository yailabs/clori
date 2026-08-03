# Canonical Glossary

Status: normative terminology

This glossary owns YVEX terminology. Other documents may give local detail but
must not redefine these terms. Qualify *model*, *support*, *complete*,
*verified*, *native*, *runtime*, *artifact*, *binding*, *target*, *variant*,
*graph*, and *execution* whenever more than one definition could apply.

## Objects and boundaries

| Term | Canonical meaning |
| --- | --- |
| Source snapshot | Identified upstream parameter bytes and sidecars retained as one immutable input to derivation. |
| Model family | A repeated architecture pattern with defined topology, tensor roles, tokenizer behavior, state semantics, and composition rules. |
| Model target | One concrete source instance within a family, including revision, configuration, tokenizer, and intended artifact class. |
| Logical model | Backend-neutral topology, tensor roles, operations, state semantics, and numerical policy. |
| Transformation plan | Immutable artifact-neutral operations mapping exact source contributions to logical output tensors. |
| Physical variant | One realization of a logical model with selected storage, layout, alignment, decomposition, and placement constraints. |
| Artifact | A concrete serialization of one physical variant. Use a narrower term when scope matters. |
| Tensor proof artifact | An artifact containing one tensor or bounded subset and proving only its named lower boundary. |
| Complete artifact | An artifact containing every tensor and metadata item required to execute the exact logical model. |
| Supported artifact | A complete artifact that has passed admission, runtime, generation, evaluation, benchmark, and release-qualification gates for its declared scope. |
| Admission | Validation that an object satisfies the complete contract required by its next consumer. |
| Materialization | Construction of checked runtime-consumable storage and bindings from an admitted artifact. It is not execution. |
| Runtime binding | Immutable content-addressed bridge from one admitted artifact and compiled facts to runtime descriptors and tensor locations. |
| Runtime model | Immutable process object for one admitted binding and model-lifetime resources. |
| Runtime session | Mutable execution object with isolated sequence state, position, workspace, cancellation, and backend resources. |
| Persistent state | Semantically observable state surviving an execution unit, including KV, recurrent state, position, or routing history. |
| Workspace | Temporary memory without semantic meaning after its owning execution unit finishes. |
| Draft plan | Immutable auxiliary execution plan that proposes tokens from admitted model features and weights. It is not a second runtime model or correctness authority. |
| Candidate block | Ordered bounded proposal awaiting complete-target verification; its tokens are not committed output or persistent target state. |
| Target verification | Complete target-model evaluation that determines the target-authored result for an ordered candidate block. Drafter confidence is not verification. |
| Accepted prefix | Exact leading portion of one verified candidate result admitted for atomic target/session publication. Rejected suffix state is discarded. |
| Speculative generation | Generation that uses an auxiliary proposal followed by complete-target verification while preserving target semantics and publishing only committed target-authored output. |
| Semantic graph | Backend-neutral operations, typed values, state effects, and dependencies derived from model meaning. |
| Executable graph | Lowered operations, physical bindings, memory plan, backend assignments, and execution variants. |
| Launch graph | Device kernels, transfers, barriers, events, and dependencies submitted as one device-level execution structure. |
| Capability | An admitted operation over a bounded input, identity, backend, mode, and resource domain. |
| Evidence | Identity-bound facts showing what executed, refused, or was measured. Evidence observes capability; it does not create it. |
| Evaluation | Repeatable assessment of model behavior or quality over the admitted text path. |
| Benchmark | Reproducible measurement over an already-correct identity-bound workload. Qualify component versus full-model scope. |
| Release qualification | Combined closure of all implementation, conformance, runtime, operator, evaluation, benchmark, packaging, claim, and release gates. |

## Common qualifications

- *Native source* means the source format and tensor inventory before YVEX
  transformation. *Native execution* means YVEX-owned execution without an
  external inference engine. The two uses must not be conflated.
- *Verified source snapshot* names a passed source identity and payload-trust
  contract. *Verified execution* names admitted execution bound to the exact
  derivation and runtime identities. Neither means release qualified.
- *Runtime* may describe the subsystem, an immutable runtime model, a mutable
  runtime session, or one execution. Name the object when ownership matters.
- *Support* always names a scope and evidence stage. Source inspection,
  complete artifact production, runtime execution, and release support are
  different claims.

## Deprecated and transitional vocabulary

| Transitional term | Canonical replacement |
| --- | --- |
| model file | artifact, complete artifact, or tensor proof artifact |
| loaded model | runtime model, with artifact and binding identities when relevant |
| runtime artifact | artifact or runtime binding, whichever object is meant |
| graph | semantic graph, executable graph, or launch graph |
| model support | the exact support stage and target, such as source-profiled or generation-capable |
| full model | complete artifact, complete Transformer execution, or complete text path, whichever boundary is meant |
| evidence command bucket | the semantic `inspect`, `execute`, `profile`, or `system` operation |
| graph command bucket | the operation-specific `inspect`, `execute`, or `profile` projection |
| developer binary | engineering operation in the finite offline lane of `yvex` |
| OpenAI gateway | in-process OpenAI-compatible adapter owned by `yvexd` |

Historical audits and migrations preserve their original vocabulary. Living
documents use the canonical replacement unless they are explicitly explaining
the transition.
