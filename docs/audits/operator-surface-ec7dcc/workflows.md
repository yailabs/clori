# Current Workflows and Composition Pressure

Commands below describe the audited baseline. Placeholders are intentional;
machine-specific paths and identities are not frozen into this audit.

The future forms are recommendations, not implemented grammar.

## A. Inspect a verified source

Current:

~~~sh
yvex source manifest report --family deepseek --release v0.1.0 --source "$SOURCE" --models-root "$MODELS" --target deepseek4-v4-flash --strict --audit
yvex source native --source "$SOURCE" --limit 20
~~~

Repeated input: source path, model root, target/family.

Manual handoff: source identity and manifest path are copied into later
compilation commands.

Future shortest truthful form:

~~~text
yvex compile source verify SOURCE_OR_ALIAS
yvex inspect source SOURCE_OR_ALIAS
~~~

The source owner must still expose exact identity, retained inventory, and
payload-trust boundaries independently.

## B. Produce or examine compilation inputs

Current:

~~~sh
yvex source manifest create --hf-repo OWNER/REPO --revision REV --local-path "$SOURCE" --status verified --out "$SOURCE_MANIFEST"
yvex tensor map --arch deepseek --native-source "$SOURCE" --template "$TEMPLATE"
yvex artifact template validate --template "$TEMPLATE"
~~~

Pressure: repository/family/architecture/template facts are separately repeated
and output paths are manually propagated.

Future:

~~~text
yvex compile source admit SOURCE --manifest MANIFEST
yvex inspect tensor-map SOURCE
~~~

Do not hide source verification or tensor coverage behind artifact emission.

## C. Plan quantization

Current:

~~~sh
yvex quant preset show PRESET
yvex quant plan --target TARGET --source "$SOURCE" --models-root "$MODELS" --source-manifest "$SOURCE_MANIFEST" --preset PRESET --out-plan "$PLAN"
~~~

Pressure: target, source, models root, source manifest, and policy selection are
repeated on emit. Plan identity is manually carried as a path.

Future:

~~~text
yvex compile quant plan SOURCE_OR_ALIAS --policy POLICY --out PLAN
~~~

Preset and policy remain inspectable. Physical planning remains distinct from
quantized-byte execution and artifact emission.

## D. Emit an artifact

Current:

~~~sh
yvex quant emit --target TARGET --source "$SOURCE" --models-root "$MODELS" --source-manifest "$SOURCE_MANIFEST" --preset PRESET --plan "$PLAN" --out "$ARTIFACT"
~~~

Pressure: emit regenerates/admit-checks planning facts but the shell still
repeats every source/policy path.

Future:

~~~text
yvex compile artifact emit PLAN --out ARTIFACT
~~~

The plan must remain identity-bound to source, policy, imatrix, and terminal
decisions; shortening syntax cannot weaken re-admission.

## E. Verify an artifact

Current explicit grammar:

~~~sh
yvex artifact verify check --model "$ARTIFACT" --expect-sha256 "$SHA256"
~~~

Current hidden shorthand:

~~~sh
yvex artifact verify "$ARTIFACT"
~~~

The entrypoint rewrites the shorthand to integrity check --model. Help
documents only check/report.

Future:

~~~text
yvex artifact verify ARTIFACT [--expect-sha256 SHA256]
yvex inspect artifact integrity ARTIFACT
~~~

One operation must own validation; a report can be a renderer projection.

## F. Materialize an artifact

Current:

~~~sh
yvex artifact materialize --model "$ARTIFACT" --backend cuda --require-all
~~~

For the resident runtime, materialization is normally implicit in admitted
yvexd startup from artifact plus runtime binding.

Future:

~~~text
yvex artifact prepare ARTIFACT --backend cuda
~~~

The offline finite proof and daemon-owned process-lifetime materialization must
remain different lifecycles even if their admission API is shared.

## G. Select a model and runtime binding

Current:

~~~sh
yvex model use NAME --artifact "$ARTIFACT" --runtime-binding "$BINDING" --target TARGET --backend cuda --context 4096
yvex model show
~~~

model list is currently identical to model show. Selection writes an inert
private XDG file; it does not switch or inspect the running daemon.

Future:

~~~text
yvex model select NAME --artifact ARTIFACT --binding BINDING
yvex model selected
yvex runtime model
~~~

The selected configuration and live runtime identities must be explicit.

## H. Start the resident runtime

Current selected-config path:

~~~sh
yvex runtime start
~~~

Current explicit path:

~~~sh
yvex runtime start --model "$ARTIFACT" --runtime-binding "$BINDING" --backend cuda --context 4096 --console raw --trace-level stages
~~~

runtime start execs yvexd in the foreground. Service deployment repeats the
same daemon flags outside the client.

Future:

~~~text
yvex runtime start
~~~

Persistent policy belongs in an admitted runtime configuration. Direct yvexd
flags remain an advanced process/service boundary.

## I. Inspect runtime readiness

Current human:

~~~sh
yvex runtime status
~~~

Current machine form:

~~~sh
yvex runtime status --json
~~~

The JSON includes more authoritative facts than the human view. The future
operation remains runtime.status with human, JSON, REPL, and TUI projections
derived from the same schema.

## J. Run one generation

Current:

~~~sh
yvex run --strategy greedy --max-new-tokens 32 "PROMPT"
~~~

A session name is optional. Without one, the client creates and closes an
ephemeral run-PID session while yvexd and its model remain resident.

Future: keep the same short product shape. Sampling policy/defaults should be
described once by the registry and admitted by the sampling owner.

## K. Run retained multi-turn chat

Current:

~~~sh
yvex session new main
yvex chat --session main --max-new-tokens 128
~~~

Implicit yvex enters chat with the default session. Slash commands then
duplicate external session operations through independent parser branches.

Future:

~~~text
yvex
yvex chat --session main
~~~

Slash commands become interaction adapters over the same operation descriptors;
they do not get separate session implementations.

## L. Access the OpenAI-compatible endpoint

The listener is already inside yvexd; there is no gateway process.

Current health:

~~~sh
curl -sS http://127.0.0.1:8001/health
~~~

Current Chat Completions request:

~~~sh
curl -N -sS http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MODEL","messages":[{"role":"user","content":"PROMPT"}],"stream":true,"max_tokens":32}'
~~~

Future command taxonomy does not absorb HTTP syntax. HTTP remains an
application projection over provider-neutral requests and local protocol
operations.

## M. Observe runtime events

Current operational projection:

~~~sh
yvex runtime watch
~~~

Current canonical raw stream:

~~~sh
yvex runtime trace
~~~

Current trace --follow is behaviorally identical to trace. watch prints generic
a/b payloads. Future projections should be:

~~~text
yvex runtime watch
yvex runtime trace
yvex runtime trace --json
~~~

The typed event stream remains the sole telemetry authority.

## N. Run direct numerical/component proof

Current examples:

~~~sh
yvex graph attention execute --target TARGET --backend cuda [owner-specific options]
yvex graph moe execute --target TARGET --artifact "$ARTIFACT" --runtime-binding "$BINDING" --backend cuda --input tensor-file --input-file INPUT --scope full
yvex graph transformer generate --target TARGET --artifact "$ARTIFACT" --runtime-binding "$BINDING" --backend cuda --user "PROMPT" --max-new-tokens 4
~~~

These finite engineering operations can initialize engine/CUDA owners directly.
They are not hosted generation and do not create another persistent model
authority.

Future:

~~~text
yvex execute attention ...
yvex execute moe ...
yvex execute transformer ...
yvex profile runtime ...
~~~

Exact names remain subject to the command milestone. Default product help
should not present component proof as the normal model workflow.

## Cross-workflow pressure

| Repeated fact | Current repetition | Required authority |
|---|---|---|
| artifact path/alias | compile, verify, materialize, model select, graph | artifact/model reference schema |
| runtime binding | model select, daemon, direct graph | runtime binding reference |
| target/family | source, plan, graph, daemon | admitted target/model descriptor |
| backend | materialize, graph, daemon, request reports | capability plus runtime config |
| context | model selection, daemon, context reports | runtime config/request policy |
| sampling defaults | client, provider, server/generation | sampling descriptor/domain admission |
| output mode | many report parsers | renderer projection descriptor |
| registry/model root | source, model, evidence routes | system path configuration |
| trace level/content | daemon, events, direct graph | telemetry policy |

Composition must remove repeated shell transcription without hiding identity or
verification boundaries.
