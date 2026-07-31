# Release Doctrine

Status: normative release contract

This document owns the stable meaning and closure rules of YVEX release gates.
[`ROADMAP.md`](../../ROADMAP.md) owns current gate state, dependency order, and
release target. Version-specific facts live in their release record.

Documentation records implementation truth. It cannot create capability.

## Release identity

A release binds a declared source snapshot, model target, logical model,
physical variant, artifact, runtime binding, backend profile, operator path,
evaluation suite, benchmark workload, package, and source commit. Changing one
of those identities requires re-admission of every dependent gate.

## Gate meanings

| Gate | Minimum closure evidence | Cannot close from |
| --- | --- | --- |
| Source | Verified revision, configuration, tokenizer, shard inventory, payload trust, and retained source identity | path presence, download receipt, header names alone |
| Architecture | Typed topology, tensor roles, state, tokenizer/output, and family constraints | family name or external runtime support |
| Transformation | Immutable complete plan binding exact source contributions to logical outputs | emitted names or payload-time rediscovery |
| Physical policy | Complete dtype/qtype/layout decisions with canonical geometry and refusal | preset name or storage estimate |
| Artifact | Complete deterministic artifact, exact identity, integrity, and writer-reader equivalence | tensor proof, external file, structural parse alone |
| Materialization and binding | Every required tensor admitted and bound to exact runtime facts with cleanup | allocation estimate or selected tensor transfer |
| Backend and Transformer | Fail-closed complete Transformer execution with state, reference/conformance, failure, and cleanup | device probe or isolated primitive |
| Text path and generation | Real tokenizer, prefill/state, logits, sampling, sampled-token decode, stop, and detokenization through the product path | fixture logits, token IDs, selected layer, trace output |
| Evaluation | Repeatable behavior and quality cases over the release path | conformance, manual prompt, or planned harness |
| Benchmark | Reproducible full-model workload with identities, run counts, latency, throughput, memory, and reliability | estimate, component timing, or one unrecorded run |
| Package and operator | Clean package identity, canonical commands, refusal, lifecycle, and installation evidence | source-tree test binary or documentation example |
| Release qualification | Every required gate, claim audit, version record, tag, and reproducibility check closes together | a lower gate or editorial assertion |

## Closure rules

Each gate requires its production owner, API, operator reachability where
applicable, positive and refusal tests, lifecycle/cleanup evidence, and exact
identity. A downstream result cannot repair a missing upstream contract.

Gate closure remains valid only while its evidence and dependencies remain
valid. Source, physical policy, artifact, runtime, backend, workload, or package
drift reopens dependent gates.

No release gate may be waived by changing terminology. A complete artifact is
not a supported artifact; executable generation is not evaluation; a
component benchmark is not a full-model release benchmark.

## Failure and security

Release qualification includes fail-closed behavior, malformed-input refusal,
cancellation, partial-progress truth, rollback, cleanup, resource limits, and
the security boundary actually claimed by the product. Unimplemented public
authentication, TLS, or remote hardening remain non-claims rather than implied
properties of a local service.

## Evidence and publication

Release evidence is reproducible from a clean checkout plus named
operator-local sources and artifacts. Large artifacts, raw traces, generated
text, evaluation datasets/results, and benchmark records remain external and
identity-bound. The release record summarizes accepted evidence without
becoming its substitute.

## Change control

Changing release scope or required gates requires a reviewed roadmap update,
version-specific release-record update, affected technical contracts, tests,
and claim audit. Decisions with durable architectural consequences receive an
ADR. Internal commits and milestones are not copied into the public changelog
unless they change an externally meaningful contract.
