# Core Compiler and Family Consolidation

Status: accepted implementation-boundary contract

Milestone: `V010.CORE.COMPILATION.FAMILY.CONSOLIDATION.0`

Status source: [`ROADMAP.md`](../../ROADMAP.md)

This contract owns the compiler/family boundary established between repository
compression and continued GB10 optimization. Live state and dependency order
belong only to [`ROADMAP.md`](../../ROADMAP.md).

## Required boundary

Compiler-facing family adapters interpret source-authored facts and irreducible
composition only while producing compiler-owned representations. Semantic Model
IR owns immutable model capability, geometry, canonical roles, state semantics,
and family attributes. Canonical Operator Graph IR owns backend-neutral operands,
operations, state edges, ordering, shapes, and numerical contracts. Generic
validation and lowering then seal Physical Execution IR, the admitted execution
envelope, and one content-addressed runtime binding.

Runtime model-open authenticates and instantiates those compiled products. It
does not consult a concrete family registry, call family graph construction, or
reconstruct model topology. Backends execute typed admitted operations and
capability resolutions instead of inferring family policy. DeepSeek and MiniMax
retain their source facts and irreducible semantics while reusable attention,
component, artifact, quantization, residency, and execution transactions remain
generic owners.

The model-authored maximum context is a Semantic Model IR capability. Requested
and selected `ctx` values are workload facts admitted inside the compiled
context envelope; they are not family constants. Hardware-fit admission remains
a downstream capacity and topology responsibility.

The sole public executable exposes the long-lived host truthfully as
`yvex server MODEL [--ctx N]`. It enters the foreground server lifecycle
directly. `server status`, `server model`, `server memory`, `server log`, and
`server stop` are the canonical control and observation operations; internal
runtime terminology remains unchanged where it names execution objects.

## Acceptance evidence

A deterministic CPU-only tiny vertical generates an untracked artifact and
binding through production compiler APIs, launches the real foreground server,
checks the selected context through protocol status, generates exact text
through the real client path, observes a typed completion event, stops cleanly,
reproduces both identities, and refuses a corrupted artifact. It introduces no
production test family and requires no external model.

Architecture guards require family semantics to terminate at compilation,
runtime model-open to consume sealed binding truth, generic owners to avoid
concrete family implementations, and capability fallback to remain explicit.
Combined build, ownership, layout, documentation, CPU/no-`nvcc`, CLI, sanitizer,
tiny-vertical, and repeat-run validation preserve the admitted verticals.

## Non-claims

This boundary does not close GB10 Tensor Core performance, deep-context
qualification, prefix persistence, continuous batching, evaluation, benchmark,
release, hot model reload, topology planning, tiered or SSD residency,
multi-device execution, or public remote serving. Those capabilities retain
their existing owners and evidence gates.
