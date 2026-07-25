# YVEX Operator Runbook

Date: 2026-07-22
Status: runbook index and repository operation boundary

## Purpose

This document routes operators to current executable procedures. It is not a
command catalogue, delivery ledger, capability dashboard, or substitute for
`./yvex help`.

The v0.1.0 product target is DeepSeek-V4-Flash from:

```text
$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash
```

The canonical full target is `deepseek4-v4-flash`. Its selected GGUF and
common attention runtime are admitted; persistent KV, transformer composition,
and model generation remain unsupported.

## Runbook Index

| Runbook | Current purpose | Capability boundary |
| --- | --- | --- |
| `runbooks/deepseek.md` | Exact source trust, admitted artifact, runtime-binding, attention execution, and benchmark-chart procedure | attention activation probes only; no prompt or generation procedure |
| `runbooks/common.md` | Build, validation, documentation guards, artifact hygiene, and operator-local cleanup | validation does not create runtime capability |

Model-family architecture is defined in `model-families.md`. Release gates are
defined in `v010-release-doctrine.md`. Current project state, dependencies, and
Active Next are defined only in `../PROJECT.md`.

## Current Entry

Use the DeepSeek runbook for source verification and real artifact-backed
attention execution. Use the common runbook for repository validation. Do not
misclassify the attention activation probe as prompt prefill, model decode, or
generation.

The attention `qualify` surface separates software acceptance, numerical
conformance and runtime reliability. The attention `benchmark` surface measures
one admitted component configuration. Neither surface is model behavior or
quality evaluation, an agent-runtime evaluation, a full-model benchmark, or
release qualification.

Consult `../PROJECT.md` before selecting work. This runbook does not mirror the
current milestone. The current operator path executes admitted attention;
generation requests must still refuse explicitly.

## Operator-Local State

The following remain outside git:

- model sources, emitted GGUF files, and runtime bindings;
- local registries and artifact identities;
- benchmark baselines, JSON/CSV reports, SVG charts, logs, pid files, caches,
  and partial downloads;
- generated backend outputs and build products.

Repository guardrails are listed in `runbooks/common.md` and
`MODEL_ARTIFACTS.md` at the repository root.

Attention component benchmark comparisons accept an optional
`--max-regression-bps N` caller policy. Without it, compatible records report
measured deltas. With it, latency, inverse throughput, memory, transfer,
allocation, and launch regressions share the explicit basis-point ceiling and
produce a nonzero status when breached. Runtime qualification failures remain
independent and cannot be waived by performance policy.
