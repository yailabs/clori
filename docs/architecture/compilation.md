# Compilation and Artifact Architecture

Status: current implemented architecture

This document owns the explanation of how YVEX turns an identified source
snapshot into an admitted physical artifact and runtime binding. Normative
artifact requirements are in the [Artifact and Admission Contract](../contracts/artifacts.md).

## Pipeline

```text
verified source snapshot
  -> logical model and canonical tensor roles
  -> immutable transformation plan
  -> policy-selected physical variant
  -> deterministic transformation execution
  -> complete GGUF artifact
  -> artifact admission and materialization
  -> content-addressed runtime binding
```

![Physical model compilation resolves each terminal tensor through sealed policy and capability authorities before artifact construction.](../diagrams/physical_compilation.svg)

The editable source is
[`physical_compilation.mmd`](../diagrams/physical_compilation.mmd).

## Source intake and trust

Source intake records repository/revision, configuration, tokenizer sidecars,
shards, tensor names, shapes, dtypes, and byte ranges. Structural inventory and
payload trust are separate. A source can be inventoried without having every
payload digest authenticated.

The retained source snapshot is immutable and indexed. Downstream owners
consume typed facts and exact bounded ranges; they do not rescan source headers
or infer semantics from filenames.

## Logical projection and transformation

Family owners interpret exact source facts into model topology and canonical
roles. The transformation plan then binds every terminal output tensor to its
ordered source contributions and typed operations. Plan construction is
artifact-neutral and payload-free.

Transformation execution reads only the ranges named by the sealed plan. It
may select, concatenate, permute, aggregate, scale, convert, or quantize as
declared. It may not rediscover axes, expert ordering, companion scales, or
qtype policy from source names.

## Physical policy

A physical variant resolves storage dtype/qtype, row geometry, layout,
alignment, aggregation, and placement constraints for every terminal tensor.
The policy is part of the variant identity. Quantization codecs and qtype
geometry are canonical owners, while selection of a qtype for one tensor is a
physical-policy decision.

Writer and runtime owners consume the resolved variant. They do not pick a
different representation for convenience.

## Artifact emission and admission

The GGUF writer plans exact metadata, tensor directory order, alignment,
padding, ranges, and payload bytes before transactional publication. Native
reader and global-layout owners independently validate the emitted container.

Complete-artifact admission additionally binds model metadata, tokenizer
facts, every required tensor role, qtype support, source/derivation/variant
identities, and exact file identity. Structural GGUF validity is necessary but
not sufficient.

Materialization produces checked backend-addressable tensor views and resource
ownership. It does not execute a graph or establish support for a model.

## Runtime binding

The runtime binding is a separate content-addressed object. It bridges the
admitted artifact to runtime descriptors, physical tensor locations,
executable requirements, numerical identities, and compatibility constraints.
The warm runtime reopens and authenticates it rather than rebuilding compiler
plans.

Artifact drift, binding drift, unsupported qtypes, missing roles, resource
overflow, or incompatible runtime requirements refuse before model execution.

## Current DeepSeek compilation

For the admitted DeepSeek target, exact source coverage accounts for 69,187
requirements and lowers them to 1,360 terminal tensors. Source-faithful,
Q8_0/Q2_K, and mixed IQ2_XXS/Q2_K complete artifacts have been produced outside
Git and admitted through their own identities. These facts establish the
compilation and complete-artifact boundaries; they do not select a release
variant or establish evaluation, benchmark, or release qualification.

## Repository boundary

Model weights, source payloads, complete artifacts, runtime bindings,
registries, transformation outputs, and raw evidence stay outside the
repository. Tiny test fixtures are admitted only by their focused test owner.
