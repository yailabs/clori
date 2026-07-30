# Target Command Taxonomy

This is design input, not implemented grammar.

## Principles

1. yvex remains the sole public command executable.
2. yvexd remains the sole persistent runtime process.
3. Default help optimizes for running a model and producing a verified artifact.
4. Advanced help exposes supported inspection and configuration.
5. Engineering execution remains reachable but cannot dominate default
   discovery.
6. Syntax, transport, renderer, and domain operation are separate fields.
7. Verification boundaries stay explicit even when configuration removes
   repeated paths.
8. eval and bench do not appear until their capability owners exist.
9. There is no dev namespace and no compatibility executable.
10. HTTP remains an integration protocol, not CLI syntax.

## Current named top-level surface

The baseline has fourteen named top-level entries plus the implicit chat entry:

~~~text
yvex
chat
run
runtime
session
model
artifact
graph
quant
tokenizer
source
tensor
evidence
help
version
~~~

runtime contains both hosted administration and offline input/context reports.
evidence is an audience/claim bucket. graph is a broad implementation term.
Quant, source, tensor, tokenizer, and evidence expose advanced detail at the
same discovery level as chat and run.

## Recommended discovery hierarchy

~~~text
yvex
  chat
  run

  runtime
    start
    stop
    status
    model
    memory
    watch
    trace
    cancel

  session
    new
    list
    show
    attach
    detach
    reset
    close

  model
    selected
    select
    list
    show

  compile
    source
    plan
    quant
    emit

  artifact
    show
    verify
    prepare

  inspect
    source
    artifact
    model
    tensor
    tokenizer
    context
    backend
    target
    moe
    qtype

  execute
    input
    tokenizer
    attention
    moe
    transformer

  profile
    runtime
    attention
    graph

  system
    status
    paths
    accounts
    cuda
    package

  help
  version

  eval      future; absent until implemented
  bench     future; absent until implemented
~~~

This is a taxonomy proposal, not a requirement that every conceptual plane
become a literal namespace. In particular, Compile, Run, Execute, and Integrate
are architecture planes. Integrate remains primarily protocol/API
documentation; it does not need a yvex integrate command.

## Default help

Default help should show only:

~~~text
yvex
yvex chat
yvex run
yvex runtime start|stop|status|watch
yvex session ...
yvex model selected|select
yvex compile ...
yvex artifact show|verify|prepare
yvex help [PATH]
yvex version
~~~

One line should route expert users to:

~~~text
yvex help --advanced
yvex help execute
yvex help inspect
yvex help profile
yvex help system
~~~

The exact advanced switch is not selected by this audit. The registry must
support visibility-aware discovery without hiding executable behavior.

## Current-family dispositions

### source

source manifest create/report belongs to Compile because it establishes
provenance and source admission. Header-only inventory belongs to Inspect.

Proposed:

~~~text
compile source admit|verify
inspect source
~~~

### tensor

Tensor mapping is compilation/lowering. Tensor directory and collection
coverage are inspection.

Proposed:

~~~text
compile tensor-map
inspect tensor
inspect tensor collection
~~~

tensor should not remain a top-level default namespace.

### tokenizer

Tokenizer show is inspection. encode/decode/prompt are direct execution useful
for experts and conformance. Normal chat/run consume tokenizer semantics
without exposing them.

Proposed:

~~~text
inspect tokenizer
execute tokenizer encode|decode|prompt
~~~

### quant

Quantization belongs under compile. Capability/report forms belong under
inspect. imatrix and job are engineering manifest boundaries.

Proposed:

~~~text
compile quant preset|policy|calibration|plan|emit|convert
inspect quant plan|decision
inspect qtype
~~~

### materialization

Artifact materialization is preparation of an admitted artifact for a backend.
Runtime startup consumes it but does not own its identity.

Proposed default:

~~~text
artifact prepare
~~~

The materialization/model gates remain engineering execute operations. Runtime
process-lifetime materialization remains internal to yvexd startup.

### graph

graph is not a product concept with one user intent. Its current leaves split:

| Current leaf | Future plane |
|---|---|
| describe/capabilities/plan | inspect |
| execute/state exercise | execute |
| residency inspect | inspect or profile |
| capture/replay/cuda-graph | engineering execute/profile |
| trace/profile | profile |
| benchmark/qualify | future benchmark/qualification owners |
| transformer generate | engineering execute proof; not normal run |

No graph top-level namespace is recommended for default discovery.

### runtime input/context

These are offline engineering routes and do not administer yvexd.

Proposed:

~~~text
execute input
inspect context
~~~

Hosted runtime remains reserved for process, model, session/KV, generation,
memory, events, and configuration.

### evidence

evidence should dissolve. Evidence rank is a project/claim property, not a
domain namespace.

| Current route | Proposed owner |
|---|---|
| evidence target | inspect target |
| evidence model | inspect model; artifact prepare; execute materialization |
| evidence moe | inspect moe |
| evidence backend | inspect backend or system status |
| evidence cuda | system cuda |
| evidence accounts | system accounts |
| evidence paths | system paths |
| evidence models | model registry/acquire/select plus artifact inspection |

### model

Three distinct facts must be separated:

- configured selection for the next start;
- registry/catalog entries available locally;
- identities of the model currently open in yvexd.

Proposed:

~~~text
yvex model selected
yvex model select NAME
yvex model list
yvex model show NAME
yvex runtime model
~~~

No command may project selected configuration as live state.

## Automation discovery

Human help and machine discovery should consume the same registry but render
different products.

Recommended machine form:

~~~text
yvex help --json
~~~

or a dedicated, registry-defined discovery operation. The final spelling is
left to the command milestone. Its schema must include stable operation IDs,
arguments, flags, defaults, side effects, requirements, and projections.

Automation should bind operation IDs and versioned schemas, not scrape human
tables. JSON support remains per typed result schema; a generic output switch
does not manufacture a machine contract.

## REPL

Slash commands become adapters:

| Slash | Operation |
|---|---|
| /sessions | session.list |
| /new | session.new |
| /attach | session.attach |
| /detach | session.detach |
| /reset | session.reset |
| /close | session.close |
| /cancel | generation.cancel |
| /quit | REPL-local exit |
| /help | command discovery |
| /status | runtime.status |

The adapter may tailor interaction and rendering, but cannot duplicate
validation or session semantics.

## Future roles

eval and bench are reserved concepts because PROJECT.md has explicit future
owners. They are not added to current help, parsing, package contents, or
readiness flags until the corresponding capabilities are implemented.

The potential future yvex-eval and yvex-bench role binaries remain a roadmap
decision. This audit neither creates nor requires them.
