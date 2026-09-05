# Source and Module Ownership

[AGENTS](../../AGENTS.md#source-layout-and-ownership) owns mandatory admission,
dependency, visibility, and size rules. This document explains how to maintain
the source/build ownership projection. It is not another per-file registry or
capability table.

## Membership authority

[`config/source_owners.tsv`](../../config/source_owners.tsv) is the sole
handwritten production membership authority. Every C, CUDA, and header file
under `src/` and `include/` appears exactly once. Inspect that manifest for
current paths rather than copying a mutable file catalog into documentation.

| Field | Meaning |
| --- | --- |
| `path` | Exact source-relative identity |
| `subsystem` | Owning subsystem |
| `semantic_owner` | Behavior/contract owner within that subsystem |
| `scope` | Generic or family-specific scope |
| `visibility` | Public or private exposure |
| `boundary` | ABI, implementation, or other admitted boundary |
| `primary_consumers` | Real consumer set |
| `partition` | Interface/implementation partition |
| `exception_id` | Explicit governed exception or `none` |

The deterministic generator validates filesystem parity and produces
`build/generated/sources.mk` under the selected build directory. The root
Makefile consumes this projection; it neither repeats production path lists
nor admits arbitrary files through wildcards. Source-relative object/archive
identities preserve namespace and prevent basename collisions.

## Changing an owner

Before adding or moving a file, identify its ABI, lifecycle, reusable
multi-consumer algorithm, backend boundary, generated boundary, family recipe,
or executable entrypoint. If there is no distinct owner, extend the existing
file or keep the helper static. A phase name or size split alone is not an
ownership argument.

Update the manifest in the same semantic change, then run:

```sh
make generate-source-manifest
make check-source-manifest
python3 tools/qa.py plan --changed BASE
python3 tools/qa.py run --changed BASE
```

Review affected dependency, visibility, layout, and family-consumer guards.
Generated products remain outside Git. For command metadata use the separate
[operation-registry workflow](../architecture/commands.md): that registry
owns grammar and projections, not production membership.

## Navigating the implementation

| Level | Current subject |
| --- | --- |
| `src/source/` | Provenance, acquisition, immutable inventories, trust, bounded payload delivery |
| `src/model/`, `src/gguf/`, `src/artifact/` | Semantic interpretation, physical package construction, format/admission and mapping |
| `src/graph/` | Typed plans, generic operators and persistent-state mechanisms |
| `src/runtime/` | Bound model execution, session transactions, generation and measurement |
| `src/backend/` | Admitted physical operations, memory, submission and synchronization |
| `src/server/`, `src/provider/` | Persistent host, routing, typed application requests and results |
| `src/cli/` | Input adaptation, typed rendering and operator I/O |

[System architecture](../architecture/system.md) owns the dependency map,
[compilation](../architecture/compilation.md) owns source-to-binding
transitions, and [runtime](../architecture/runtime.md) owns execution
lifetimes. Family files at model, graph, and backend levels are distinct
projections described by [family integration](../model-families/integration.md),
not parallel runtimes.

## Format boundaries

The GGUF structural reader uses positioned reads for metadata and the tensor
directory. It owns copied length-aware values until close. The artifact handle
maps payload only on an explicit consumer request. Structural-reader metrics
therefore report zero payload bytes.

The canonical layout owner borrows the opened artifact and parsed view,
validates alignment, directory-order padded continuation, zero padding, the
complete span, and snapshot stability. Reading padding is not tensor execution
or complete-artifact admission.

The qtype owner derives bytes from the complete shape with `ne[0]` as row
width and classifies removed/outside-baseline identities. Dtype, integrity,
conversion, and planning consumers use those facts without duplicating
geometry. Structural/qtype support alone does not imply numerical execution.
The [artifact contract](../contracts/artifacts.md) owns admission semantics;
the [C API](../contracts/c-api.md) owns record and lifetime obligations.
