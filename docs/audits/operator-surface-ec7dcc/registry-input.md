# Canonical Operation Registry Design Input

This document defines the information required by the later command/console
milestone. It does not implement a registry and production code must not read
the frozen audit TSVs.

## Recommendation

Use a versioned machine-readable source schema committed under a canonical
operator/config owner. Generate checked static C descriptors from that schema
at build time.

The generated C is compiled into yvex. A machine-readable discovery document
and shell-completion inputs may be generated from the same source. Runtime
dispatch must not depend on a mutable installed data file.

Why this shape:

- static C descriptors preserve native bounded parsing and make route
  reachability auditable at link time;
- one source schema avoids repeating hundreds of C initializers and supports
  deterministic documentation/completion output;
- generation can reject duplicate paths, aliases, operation IDs, flags,
  conflicting defaults, and missing owners before compilation;
- the yvex binary works without an installed data directory; and
- domain APIs stay typed C authorities rather than being redefined in data.

A hand-written static C table alone would reproduce the current drift at larger
scale. Runtime-loaded installed data would make product behavior depend on
mutable files and complicate package identity. Generated C plus generated
discovery is the smallest evidence-supported choice.

## Registry record

Each operation descriptor needs:

| Field | Meaning |
|---|---|
| schema_version | registry ABI/version |
| operation_id | stable semantic operation, independent of spelling |
| namespace | current command grouping |
| action | leaf action |
| aliases | explicit syntax aliases with lifecycle/deprecation |
| plane | Compile, Run, Execute, Integrate, or support plane |
| visibility | product-default, product-advanced, engineering, automation, API-only, test-only |
| summary | bounded human purpose |
| arguments | ordered typed positional descriptors |
| flags | ordered/repeatable typed option descriptors |
| defaults | references to one canonical default owner |
| validation | syntax admission and domain validator reference |
| input_schema | typed request/report schema identity |
| result_schema | typed result schema identity |
| side_effects | read/write/process/session/network/device classes |
| tty_policy | required, optional, forbidden |
| daemon_requirement | absent, optional, required |
| local_capability | CPU/CUDA/tool/filesystem requirements |
| model_requirement | none, selected config, artifact, runtime model |
| artifact_requirement | none, descriptor, admitted artifact, binding |
| protocol_operation | optional local protocol operation |
| api_owner | typed C owner invoked after parsing |
| projections | CLI, slash, completion, HTTP, future TUI |
| deprecation | current, alias, deprecated, removed |
| test_owner | positive/refusal/dispatch acceptance owner |
| documentation_owner | help/runbook/API owner |

An argument descriptor needs name, type, multiplicity, requiredness, range,
enum values, completion provider, sensitive-display policy, and validator.

A flag descriptor needs canonical spelling, aliases, value type, multiplicity,
requiredness, default reference, range/enum, conflicts, dependencies,
environment/config equivalence, protocol field, output interaction, and
deprecation.

## Ownership boundaries

The registry owns:

- command path and aliases;
- syntax metadata;
- visibility and discovery order;
- parser construction;
- syntax-level conflicts;
- routing to a typed adapter;
- projection availability;
- help/completion/documentation validation metadata; and
- stable operation IDs.

The registry does not own:

- artifact or source trust;
- qtype or model policy;
- numerical operations;
- runtime/session/KV objects;
- tokenizer or generation semantics;
- sampling admission;
- renderer implementation;
- protocol serialization;
- process-global mutable state;
- test fixtures; or
- capability/readiness claims.

Defaults with semantic meaning remain in their domain owner. The registry
references them through a typed default provider or a generated constant whose
source is that owner. It must not copy 128, 256, 4096, CUDA, target names, or
sampling thresholds into several syntax descriptors.

## Generated products

The same accepted descriptors should generate or validate:

1. top-level yvex help;
2. namespace and leaf help;
3. argv parsing;
4. flag parsing and duplicate/conflict refusal;
5. dispatch tables;
6. REPL /help;
7. slash-to-operation adapters;
8. shell completion;
9. machine-readable discovery;
10. future TUI command palette/catalog;
11. documentation command/link checks;
12. positive/refusal dispatch matrix;
13. route/client-lane dependency guards; and
14. package surface checks.

Renderer code remains separately owned. The descriptor chooses a result schema
and renderer ID; it does not contain formatting callbacks that become semantic
authorities.

## Projection model

A stable operation can have several projections:

~~~text
operation session.reset
  CLI      yvex session reset NAME
  REPL     /reset
  protocol YVEX_CLIENT_OP_SESSION_RESET
  TUI      reset-session action
~~~

Each projection has its own syntax and interaction policy, but all call one
typed operation adapter and one domain owner.

HTTP projections remain defined in the OpenAI compatibility profile. The
registry may record their relationship for discovery/tests; it must not
generate OpenAI parsing from CLI flags or expose OpenAI object names to runtime
owners.

## Lane enforcement

Every descriptor declares one lane:

- runtime-client: only client/protocol/config/render dependencies;
- offline-engine: finite in-process domain/report API;
- daemon-entrypoint: yvexd process configuration;
- REPL-local: interaction without domain semantics;
- API-only/test-only: unavailable to ordinary CLI dispatch.

Build guards should generate two dispatch tables or two statically typed
adapter families rather than a generic void-pointer callback. This preserves
the post-cutover invariant that chat/run/status/session never enter engine
handlers even though the yvex ELF also links offline engine objects.

## Validation gates

The generator/validator must refuse:

- duplicate operation IDs;
- duplicate canonical command paths;
- aliases colliding with canonical paths;
- the same flag with conflicting types/defaults in one path;
- missing parser or API owner;
- public routes without help and documentation status;
- product routes without positive and refusal tests;
- runtime-client routes targeting offline adapters;
- engine routes presented as daemon-backed;
- an unsupported JSON/CSV output claim;
- default values without a semantic owner;
- projections naming nonexistent protocol operations;
- removed yvex-dev/yvex-openai executables;
- placeholder eval/bench operations;
- unversioned machine-readable schema changes; and
- orphan descriptors not reachable from any admitted projection.

## Migration sequence

1. Freeze this baseline audit and approve target taxonomy.
2. Add schema and generator/validator without changing existing syntax.
3. Encode all 70 current command rows and 426 route/flag pairs.
4. Generate dispatch/help in shadow validation and compare to current
   positive/refusal behavior.
5. Switch product and offline dispatch to generated descriptors.
6. Convert slash commands into registry projections.
7. Resolve P1 semantics one operation at a time with explicit breaking
   decisions.
8. Apply target taxonomy and visibility.
9. Add semantic runtime watch/trace/console renderers.
10. Remove obsolete adapters only after no descriptor/test/documentation
    consumer remains.

No compatibility executable, dev namespace, or second runtime process is part
of migration.

## Required registry tests

- schema parse/version;
- deterministic generation;
- source-to-generated freshness;
- command and alias uniqueness;
- flag type/default/conflict admission;
- every current route represented;
- every descriptor dispatches one expected adapter;
- client-lane symbol/dependency guard;
- offline cleanup lease behavior;
- help/discovery completeness;
- slash projection equivalence;
- shell completion smoke;
- machine discovery schema;
- documentation command validation;
- removed binary names absent; and
- repeat build/object/archive parity.

## Open questions for implementation

The command milestone must decide:

- exact schema syntax (for example strict TSV, JSON, or a small declarative
  format);
- whether completion/discovery is emitted at build or served from compiled
  descriptors;
- final aliases and deprecation window;
- final advanced-help switch;
- whether direct engineering execution uses execute as a literal namespace;
- whether profile owns trace/benchmark-like graph leaves;
- selected-model versus registry command spelling; and
- protocol removal/versioning for the three false facade operations.

These choices cannot change product topology or domain ownership.
