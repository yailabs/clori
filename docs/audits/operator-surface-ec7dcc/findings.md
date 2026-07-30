# Findings

This file classifies baseline defects. It does not repair them. Line references
identify the baseline owner; later edits may move them.

## Severity summary

| Severity | Count | Progression effect |
|---|---:|---|
| P0 — authority/correctness violation | 0 | none found |
| P1 — public semantic defect | 8 | input to command-console; three provisional protocol operations require explicit disposition |
| P2 — architecture/maintenance debt | 13 | registry and taxonomy work |
| P3 — presentation/documentation debt | 5 | renderer/help work |

No runtime-facing yvex handler was found opening an artifact, initializing CUDA,
or bypassing yvexd. No second model, session, KV, worker, queue, or telemetry
authority was found. The accepted two-executable topology remains sound.

## P1 — public semantic defects

### P1-01 — model list does not list models

Evidence: src/cli/io/client.c dispatches both model list and model show to
model_config_show(). The function reads exactly one private selected-model
configuration.

Consequence: scripts and users cannot distinguish enumeration from inspection.

Disposition: remove the false list spelling or bind it to a real registry/list
operation. Rename the existing fact to model selected.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-02 — selected model and live model are conflated

Evidence: model_config_show() reads $XDG_CONFIG_HOME/yvex/model.conf without
contacting yvexd. runtime status contains the live model, artifact, binding, and
variant identities.

Consequence: model show can be interpreted as the running model even when the
selection is stale or the daemon runs different identities.

Disposition: separate selected configuration from live runtime identity in
operation IDs, help, and rendering.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-03 — three local-protocol operations are false facades

Evidence: src/server/core.c routes YVEX_CLIENT_OP_MODEL_SHOW,
YVEX_CLIENT_OP_ARTIFACT_SHOW, and YVEX_CLIENT_OP_ARTIFACT_VERIFY through the
same status_message() path as YVEX_CLIENT_OP_RUNTIME_STATUS.

Consequence: their enum names promise model/artifact semantics that the wire
result does not implement. Current product CLI no longer uses them, but future
clients could.

Disposition: remove them in an honestly versioned protocol successor or
implement separately admitted typed semantics. Do not retain aliases returning
runtime status.

Owner: protocol part of V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-04 — runtime trace --follow is inert

Evidence: src/cli/io/client.c accepts --follow, then invokes runtime_events(1)
exactly as it does without the flag. Both calls block and follow events.

Consequence: the public grammar advertises a mode switch with no semantic
effect, producing unstable automation expectations.

Disposition: remove the flag or implement a bounded snapshot/non-follow mode
with a real operation contract.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-05 — human runtime status contradicts documented fields

Evidence: render_status() receives physical-variant identity and the complete
server summary; its JSON projection includes the variant. The human renderer
omits variant identity and context capacity, while
docs/cli-output-architecture.md says compact status includes variant and
context.

Consequence: the default operator view hides identity/capacity facts promised
by its documentation.

Disposition: drive human and JSON projections from one typed status
descriptor; retain different layouts, not different truth.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-06 — runtime watch is not the claimed semantic view

Evidence: runtime_events(0) prints event name followed by generic a= and b=
fields. The CLI-output architecture describes an interpreted operational
engine view.

Consequence: operators must know event-internal field positions and cannot
reliably distinguish prompt, reuse, prefill, TTFT, decode, memory, or stop facts.

Disposition: add a semantic typed renderer over the existing event authority.
Do not change event truth or create a second metric system.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-07 — help cannot describe the reachable grammar

Evidence: print_help() is a manual summary independent of the 39-row offline
table. It omits admitted actions such as artifact template/model-gate,
quant convert/job/qtype, evidence paths/models, and runtime input/context.
Namespace --help for chat, run, runtime, session, and model is refused; run
--help is parsed as an unknown run option.

Consequence: reachable product behavior is not discoverable from the product
binary, and documentation/tests can drift without a structural mismatch.

Disposition: generate/validate help from the canonical operation registry with
default versus advanced visibility.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

### P1-08 — artifact verify has two unowned grammars

Evidence: src/cli/main.c silently rewrites a bare first argument into
integrity check --model. Dynamic help documents check/report forms, not the
adapter shorthand.

Consequence: the same public path has a hidden compatibility-like grammar whose
validation and documentation do not share an authority.

Disposition: choose one explicit public form; if shorthand remains, declare it
as a registry alias with tests and a deprecation state.

Owner: V010.OPERATOR.COMMAND.CONSOLE.0.

## P2 — architecture and maintenance debt

| ID | Finding | Evidence | Required owner |
|---|---|---|---|
| P2-01 | Thirty-nine offline routes are a separate hardcoded registry. | src/cli/main.c routes[] | command-console |
| P2-02 | Product dispatch is an independent strcmp tree. | yvex_client_dispatch() | command-console |
| P2-03 | Top-level help is independent from both dispatchers. | print_help() | command-console |
| P2-04 | Slash dispatch and slash help are independently hardcoded. | REPL branches and repl_help() | command-console |
| P2-05 | 426 exposed route/flag pairs have no common descriptor; broad routes encode action applicability in local branches. | flags.tsv | command-console |
| P2-06 | Neutral sampling defaults are repeated in request_init(), turn_options_init(), provider translation, daemon limits, and generation owners. | client/provider/server sources | command-console plus sampling API owner |
| P2-07 | Target/backend/context/output defaults are repeated between model selection and daemon startup. | client.c and yvexd.c | command-console/runtime config |
| P2-08 | Hosted runtime administration shares runtime with offline input/context reports. | current top-level grammar | taxonomy |
| P2-09 | evidence is an audience/claim bucket containing reports, provider login, paths, downloads, selection, probing, and bounded execution. | 8 evidence routes | taxonomy |
| P2-10 | graph combines execution, state, capture, profiling, qualification, benchmark, and direct generation. | graph help and parser | taxonomy |
| P2-11 | normal/table/audit/JSON/CSV and include-* controls are distributed across report owners with no surface policy descriptor. | flag inventory | registry/render policy |
| P2-12 | 121 Make targets and 52 scripts are not mapped to typed operations or CI/project-control ownership from one authority. | surfaces.tsv | public project control |
| P2-13 | Twenty-six .def option/field/boundary catalogs under src/cli/catalog have no production include or consumer. | source-reference search and surfaces.tsv | command-console removal/migration |

## P3 — presentation and documentation debt

| ID | Finding | Consequence | Later action |
|---|---|---|---|
| P3-01 | Final turn summary omits authoritative prefill seconds/rate, final position, and stop details. | Operators cannot separate prefill from decode. | console renderer |
| P3-02 | runtime trace has only canonical raw JSONL; there is no human technical trace projection. | Human diagnosis requires external parsing. | trace renderer |
| P3-03 | /health still returns a gateway field after the gateway process was retired. | Stale topology terminology. | OpenAI renderer/schema version review |
| P3-04 | Advanced/engineering commands are mixed into one flat help summary. | Normal discovery is noisy while many valid actions remain hidden. | visibility-aware help |
| P3-05 | Namespace and action errors usually return status 2 without local usage. | Users must guess or return to top-level help. | registry-generated targeted help |

## Overlap and duplicate analysis

The audit uses four distinct terms:

- exact duplicate: the same syntax-level intent reaches the same implementation
  without a meaningful projection distinction;
- semantic duplicate: different implementations or names claim the same user
  operation and need one authority;
- adapter projection: yvex reconstructs argv for an admitted lower command
  adapter;
- transport projection: REPL or HTTP syntax maps to an existing semantic
  operation and is correctly allowed to render differently.

Seventeen overlap groups were reviewed.

### Four exact duplicate groups

1. yvex model list and yvex model show both call model_config_show().
2. make client and make cli build the same yvex product.
3. make daemon and make server build the same yvexd product.
4. test-client, test-cli, and test-cli-cutover converge on the same cutover
   acceptance recipe.

Build/test aliases may remain if project control declares one canonical CI
name. The model duplicate is a public defect.

### Four semantic-overlap groups

1. selected model configuration versus live runtime model identity;
2. evidence models registry/current/use versus product model list/show/use;
3. artifact show/verify versus the three provisional protocol facade IDs;
4. graph profile/benchmark/qualify versus dedicated profile/benchmark planes.

These are not assumed equivalent. Each needs one operation authority and
explicit projection mapping.

### Adapter projections

All 39 offline routes are adapter projections. Twenty-three reconstruct a
lower spelling or injected action rather than passing a same-named action
directly. Examples include:

| Public route | Lower adapter |
|---|---|
| artifact show | inspect |
| artifact verify | integrity; optional check --model injection |
| artifact template | gguf-template |
| artifact emit | gguf-emit |
| quant preset/plan/emit/summarize/explain | quant plus injected action |
| quant policy | quant-policy |
| quant job | quant-job |
| quant qtype | qtype-support |
| tokenizer encode/decode | tokenize/detokenize |
| source manifest/native | source-manifest/native-weights |
| tensor map/collection | tensor-map/tensor-collection |
| evidence target/model/cuda | model-target/fullmodel/cuda-info |

A projection is valid only when public syntax, lower semantics, errors, and
help are intentionally bound. Several are naming debt, not compatibility
violations.

### Transport and interaction projections

There are 15 user-facing projection rows: 10 slash commands and 5 HTTP
endpoints. The local protocol has 17 operation enum values.

Correct projections include runtime status through both `yvex runtime status`
and `/status`, session new/list/attach/detach/reset/close, and generation
cancellation. Chat Completions and Responses legitimately map HTTP syntax to
provider-neutral requests and the same server generation/session authority.

The three false facade enum values are problem projections. They must not be
counted as separate domain capabilities.

## Misleading or provisional surfaces

Eight surface-level items require explicit disposition:

1. yvex model list;
2. yvex model show;
3. yvex runtime watch human rendering contract;
4. yvex runtime trace --follow grammar;
5. YVEX_CLIENT_OP_MODEL_SHOW;
6. YVEX_CLIENT_OP_ARTIFACT_SHOW;
7. YVEX_CLIENT_OP_ARTIFACT_VERIFY;
8. /health gateway terminology.

The artifact verify shorthand and incomplete help are additional grammar
defects, but are not counted again as standalone surfaces.

## Progression assessment

No P0 blocks the project-control refoundation. The successor can consume this
frozen inventory without changing commands. P1 through P3 findings remain
owned by V010.OPERATOR.COMMAND.CONSOLE.0, except project/CI ownership extracted
for V010.PROJECT.CONTROL.PUBLIC.0.

The successor must not accidentally bless current command names as final.
