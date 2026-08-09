# Repository Architecture Compression Audit

Status: frozen at accepted closure

Baseline: `7226f74fe8b2c063f2485daa5c9f36be0be0387d`

This audit binds the repository-wide owner and dependency inventory used by
the architecture-compression delivery. It is point-in-time evidence, not a
living source-layout authority. Current ownership remains in
[`config/source_owners.tsv`](../../../config/source_owners.tsv), current policy
in [`AGENTS.md`](../../../AGENTS.md), and live project state in
[`ROADMAP.md`](../../../ROADMAP.md).

## Population and method

The inventory covers every baseline `.c`, `.cu`, and `.h` beneath `src/` and
`include/`, every test/build classification consumed by Make, all resolved
first-party include edges, library-defined and undefined `yvex_` symbols, and
the root Makefile target and source declarations. Counts use the same
dependency-free lexer and policy as `tests/c_structure.py`; object and archive
facts use GNU `nm` and `ar`; Make facts use the parsed root file and `make -pn`.
Tool versions, commands, inclusion rules, and before/after values are retained
in [`metrics.json`](metrics.json).

[`owners.tsv`](owners.tsv) records one baseline row and final disposition for
every governed production file. Syntax-derived columns are measurements, not
claims that syntax establishes a semantic owner. Resource, policy, ABI, and
failure columns therefore project the reviewed machine-readable boundary
rather than guessing intent from function names.

[`edges.tsv`](edges.tsv) retains the resolved include, link-consumer, and build
authority edges needed to review the dispositions. It does not duplicate raw
compiler dependency files or dump every local symbol.

## Disposition rules

A physical boundary was retained when it owns an invariant, lifecycle,
resource, policy, ABI/schema, failure/recovery contract, generated product,
distinct consumers, selectable implementation, reference oracle, trust
boundary, or independent toolchain. A file was merged only when it separated
one such owner into a forwarding, formatting, default-policy, or one-consumer
implementation fragment. Functions were staticized only after object-symbol
consumers proved external linkage unnecessary.

The three DeepSeek projections remain deliberately separate:

- `src/model/families/deepseek_v4.c` owns family/source interpretation and
  logical lowering;
- `src/graph/families/deepseek_v4.c` owns irreducible graph composition; and
- The then-family-owned fused CUDA lowering has since moved behind the generic
  encoded-attention job; the current owner is `src/backend/cuda/attention.c`.

No runtime family-implementation hierarchy exists. Source trust/model
interpretation, Transformation IR/Physical Execution IR, GGUF/artifact
admission, materialization/runtime, graph/backend, immutable model/session,
candidate/committed state, generation/sampling, protocol/rendering, and
reference/production evidence remain distinct.

## Significant decisions

- `config/source_owners.tsv` is now the sole handwritten production
  source-membership authority. `tools/generate_source_manifest.py` validates
  exact parity and emits a deterministic, ignored Make projection.
- Ten ceremonial translation units and one redundant internal header were
  merged into their existing semantic owners. The public C API, protocol,
  artifact, binding, physical variant, tokenizer, state layout, execution
  profile, and numerical path did not change.
- The old CLI model-artifact surface forwarders were removed; canonical
  command symbols now terminate directly at the owning adapters.
- Historical Make aliases with no independent lifecycle or consumer were
  retired. Canonical build, package, validation, sanitizer, CUDA, and live
  entrypoints remain.
- Registry `support_level` now admits artifact capability only. Runtime
  binding/profile readiness is rendered and validated separately, repairing
  the selected DSpark entry without changing its artifact or binding.

## Final result

The final counts and deltas are authoritative in `metrics.json`. The
repository has fewer production files, translation units, internal headers,
semantic owners, non-public global symbols, forwarding functions, handwritten
source declarations, root Makefile lines, and historical targets. Public
headers and public declarations did not increase. Generated projections remain
non-authoritative and reproducible.

The remaining duplicate-function candidates are reviewed small local codecs,
identity builders, vtable adapters, public default wrappers, or independent
reference/backend contracts. Centralizing them would add a broader ABI or
conflate owners, so they are retained. Hardware-layout, token-local MoE,
row-local output projection, eager attention, host stochastic sampling,
host-materialized DSpark features, generic qtype CUDA matvec, launch,
synchronization, and placement depth remain explicitly owned by
`V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0`.

Unresolved repository-architecture blockers: none.
