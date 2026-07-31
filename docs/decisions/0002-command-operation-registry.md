# 0002 — Canonical command and operation registry

Date: 2026-07-31
Status: accepted

## Context

After the two-binary product cutover, `yvex` still projected one semantic
surface through several independent authorities: a handwritten offline route
table, a separate runtime-client dispatcher, manually maintained help, a
separate slash-command parser, and 26 unconsumed historical `.def` catalogs.
The frozen operator audit found 70 route-level commands, 426 command/flag
pairs, and 99 operations that had to be reconciled without making an installed
data file part of executable behavior.

The command architecture also has two deliberately different execution lanes.
Runtime-client operations must remain protocol-only. Finite offline-engine
operations may consume admitted `libyvex` owners, but must close their resources
and must never become another hosted runtime.

## Decision

Use [`config/operator/registry.json`](../../config/operator/registry.json) as
the sole versioned source for command and operation projection metadata. Its
schema identity is `yvex.operator.registry.v1`. The source is strict UTF-8 JSON,
deterministically ordered, machine-path independent, and validated before C
compilation.

[`tools/generate_operator_registry.py`](../../tools/generate_operator_registry.py)
is a Python 3 standard-library-only build tool. It validates operation IDs,
paths, aliases, arguments, flags, defaults, owners, adapters, renderers,
protocol projections, visibility, requirements, lane relationships, and frozen
audit coverage. It emits deterministic immutable products under
`build/generated/operator/`:

- `registry.h`, containing descriptor and typed adapter-enum declarations;
- `registry.c`, containing only immutable descriptor data; and
- `registry.sha256`, containing the identity of normalized registry input.

The generated C is compiled into `yvex`. It contains no callbacks, domain
logic, protocol serialization, runtime objects, allocation, or mutable state.
Process startup never reads the JSON source or an installed registry file.
This keeps reviewed metadata and executable behavior bound to the same build
while preventing local data mutation from changing dispatch.

Registry adapter IDs resolve through separate typed enum families for the
runtime-client, offline-engine, daemon-entrypoint, and REPL-local lanes. The
entrypoint performs registry-driven path resolution and syntax admission, then
selects exactly one lane. Existing domain adapters retain semantic validation;
registry metadata does not become a capability or policy owner. Runtime-client
objects remain guarded against engine dependencies, and every finite offline
dispatch closes its optional cleanup lease before exit.

The compiled registry identity is exposed by `yvex version`, command discovery,
and product package metadata. The mutable source is not installed as a runtime
dependency. Package identity binds source commit, protocol version, backend
profile, registry identity, product binary hashes, and library hash.

Schema changes require an explicit version change when they alter the
machine-readable contract. Operation IDs remain stable; removal or structural
replacement names an explicit successor. Pre-v0.1 command grammar changes are
recorded in the deterministic migration matrix, while removed top-level paths
produce refusal hints and never execute compatibility aliases.

## Consequences

- Human help, `yvex.command.discovery.v1`, shell completion, CLI dispatch,
  slash projections, documentation checks, and audit reconciliation consume
  one compiled metadata authority.
- The final command taxonomy can change without preserving implementation-era
  namespaces as hidden aliases.
- A stale generated product or unmatched audit row fails before product
  acceptance.
- Python is a build-time tool only; installed YVEX remains native C/CUDA.
- Generated descriptors are reviewable build products, not source owners or
  domain APIs.
- The subsequent REPL milestone can consume slash and operation metadata
  without creating another semantic parser.

## Alternatives considered

Parsing JSON at runtime was rejected because an installed mutable file would
become an unreviewed behavior authority and add a runtime parser dependency.
Maintaining static C descriptors by hand was rejected because it would preserve
the drift between dispatch, help, completion, and slash commands. Generating
callback implementations was rejected because generated code would acquire
domain behavior and obscure lane ownership. Keeping the historical `.def`
catalogs as a compatibility layer was rejected because they had no production
consumer and would form a shadow registry.
