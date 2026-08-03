# Command Architecture v1 Migration

This deterministic migration matrix reconciles the frozen operator audit with
`yvex.operator.registry.v1`. It is documentation, not runtime command authority.

- Frozen audit baseline: `ec7dccede90c1a1efa87b4c2519c25b30d5e1733`
- Registry identity: `dd6d40087c94088258ea995bd3a5edc2beb8ea93e76d25e22926ff4ac9e01888`
- Compatibility policy: pre-v0.1 breaking grammar cutover; removed paths never execute aliases.

| Old path | Old operation | Final operation | Final projection | Visibility | Compatibility | Rationale |
|---|---|---|---|---|---|---|
| `yvex artifact emit` | `artifact.emit.controlled` | `artifact.emit.controlled` | `yvex compile emit artifact` | engineering | breaking-cutover | consolidate: adapter projects gguf-emit |
| `yvex artifact materialize` | `artifact.materialize` | `artifact.materialize` | `yvex artifact materialize` | product-default | retained | keep: truthful |
| `yvex artifact materialize-gate` | `artifact.materialize.gate` | `artifact.materialize.gate` | `yvex execute artifact materialize-gate` | engineering | breaking-cutover | move: truthful engineering surface |
| `yvex artifact metadata` | `artifact.metadata` | `artifact.metadata` | `yvex inspect artifact metadata` | product-advanced | breaking-cutover | move: truthful |
| `yvex artifact model-gate` | `artifact.model.gate` | `artifact.model.gate` | `yvex execute artifact model-gate check` | engineering | breaking-cutover | move: truthful engineering surface |
| `yvex artifact show` | `artifact.inspect` | `artifact.inspect` | `yvex artifact show` | product-default | retained | keep: truthful adapter projection inspect |
| `yvex artifact template` | `artifact.template` | `artifact.template` | `yvex compile emit template` | engineering | breaking-cutover | move: adapter projects gguf-template |
| `yvex artifact tensors` | `artifact.tensors` | `artifact.tensors` | `yvex inspect artifact tensors` | product-advanced | breaking-cutover | move: truthful |
| `yvex artifact verify` | `artifact.verify` | `artifact.verify` | `yvex artifact verify` | product-default | retained | normalize: hidden argv reconstruction for bare FILE |
| `yvex evidence accounts` | `system.accounts` | `system.accounts` | `yvex system accounts` | product-advanced | breaking-cutover | move: side-effecting system operation under evidence |
| `yvex evidence backend` | `system.backend` | `system.backend` | `yvex inspect backend` | product-advanced | breaking-cutover | move: claim namespace obscures system operation |
| `yvex evidence cuda` | `system.cuda` | `system.cuda` | `yvex system cuda` | product-advanced | breaking-cutover | move: claim namespace obscures system operation |
| `yvex evidence model` | `evidence.model` | `inspect.model.full.report`, `inspect.model.full.materialization_plan`, `inspect.model.full.descriptor`, `inspect.model.full.family_runtime`, `execute.materialization.full_model` | `yvex inspect model full report`; `yvex inspect model full materialization-plan`; `yvex inspect model full descriptor`; `yvex inspect model full family-runtime`; `yvex execute model materialize` | engineering | breaking-cutover | split: mixes reports and bounded execution |
| `yvex evidence models` | `model.registry` | `artifact.check`, `artifact.prepare`, `artifact.registry.list`, `artifact.registry.status`, `evidence.target`, `model.acquire`, `model.acquire.cleanup`, `model.acquire.resume`, `model.acquire.status`, `model.acquire.stop`, `model.list`, `model.registry.add`, `model.registry.remove`, `model.registry.scan`, `model.registry.verify`, `model.show` | `yvex inspect artifact check`; `yvex compile artifact prepare`; `yvex inspect artifact registry`; `yvex inspect artifact status`; `yvex inspect target`; `yvex model acquire`; `yvex model acquisition cleanup`; `yvex model acquisition resume`; `yvex model acquisition status`; `yvex model acquisition stop`; `yvex model list`; `yvex model registry add`; `yvex model registry remove`; `yvex model registry scan`; `yvex model registry verify`; `yvex model show` | engineering, product-advanced, product-default | breaking-cutover | split: large side-effecting catalog under evidence; overlaps product model |
| `yvex evidence moe` | `evidence.moe` | `evidence.moe` | `yvex inspect moe` | engineering | breaking-cutover | move: report-only but named by claim audience |
| `yvex evidence paths` | `system.paths` | `system.paths` | `yvex system paths` | product-advanced | breaking-cutover | move: configuration operation under evidence |
| `yvex evidence target` | `evidence.target` | `evidence.target` | `yvex inspect target` | engineering | breaking-cutover | dissolve: audience/claim namespace, not stable domain |
| `yvex graph` | `execute.graph` | `execute.graph.attention.capture`, `execute.graph.attention.capabilities`, `execute.graph.attention.compare`, `execute.graph.attention.describe`, `execute.graph.attention.execute`, `execute.graph.attention.plan`, `execute.graph.attention.prepare`, `execute.graph.attention.replay`, `execute.graph.attention.residency.inspect`, `execute.graph.attention.state.exercise`, `execute.graph.attention.state.inspect`, `execute.graph.attention.state.validate`, `execute.graph.moe.execute`, `execute.graph.transformer.decode`, `execute.graph.transformer.execute`, `execute.graph.transformer.generate`, `execute.graph.transformer.logits`, `execute.graph.transformer.sample` | `yvex execute attention capture`; `yvex inspect attention capabilities`; `yvex execute attention compare`; `yvex inspect attention describe`; `yvex execute attention run`; `yvex inspect attention plan`; `yvex execute attention prepare`; `yvex execute attention replay`; `yvex inspect attention residency`; `yvex execute attention state exercise`; `yvex inspect attention state`; `yvex execute attention state validate`; `yvex execute moe`; `yvex execute transformer decode`; `yvex execute transformer run`; `yvex execute transformer generate`; `yvex execute transformer logits`; `yvex execute transformer sample` | engineering | breaking-cutover | split-by-operation: broad implementation namespace |
| `yvex quant convert` | `quant.convert` | `quant.convert` | `yvex compile quant convert` | engineering | breaking-cutover | move: truthful |
| `yvex quant emit` | `quant.emit` | `quant.emit` | `yvex compile quant emit` | product-default | breaking-cutover | move: truthful |
| `yvex quant explain` | `quant.explain` | `quant.explain` | `yvex inspect quant decision` | product-advanced | breaking-cutover | move: truthful |
| `yvex quant --help` | `command.discovery.quant` | `command.discovery` | `yvex help` | product-default | breaking-cutover | replace-registry-projection: manual command help |
| `yvex quant imatrix` | `quant.imatrix` | `quant.imatrix` | `yvex compile quant imatrix` | engineering | breaking-cutover | move: truthful |
| `yvex quant job` | `quant.job` | `quant.job` | `yvex compile quant job` | engineering | breaking-cutover | move: adapter projects quant-job |
| `yvex quant plan` | `quant.plan` | `quant.plan` | `yvex compile quant plan` | product-default | breaking-cutover | move: truthful |
| `yvex quant policy` | `quant.policy` | `quant.policy` | `yvex compile quant policy` | product-advanced | breaking-cutover | move: adapter projects quant-policy |
| `yvex quant preset` | `quant.preset` | `quant.preset` | `yvex compile quant preset` | product-default | breaking-cutover | move: truthful |
| `yvex quant qtype` | `quant.qtype.support` | `quant.qtype.support` | `yvex inspect qtype` | product-advanced | breaking-cutover | move: adapter projects qtype-support |
| `yvex quant summarize` | `quant.summarize` | `quant.summarize` | `yvex inspect quant summary` | product-advanced | breaking-cutover | move: truthful |
| `yvex runtime context` | `context.report` | `context.report` | `yvex inspect context` | engineering | breaking-cutover | move: misplaced under hosted runtime namespace |
| `yvex runtime input` | `input.prepare` | `input.prepare` | `yvex execute input` | engineering | breaking-cutover | move: misplaced under hosted runtime namespace |
| `yvex source manifest` | `source.manifest` | `source.manifest` | `yvex compile source manifest` | product-default | breaking-cutover | move: truthful but report and mutation share route |
| `yvex source native` | `source.native.inspect` | `source.native.inspect` | `yvex inspect source` | product-advanced | breaking-cutover | move: adapter projects native-weights |
| `yvex tensor collection` | `tensor.collection.report` | `tensor.collection.report` | `yvex inspect tensor collection` | engineering | breaking-cutover | move: adapter projects tensor-collection |
| `yvex tensor map` | `tensor.map` | `tensor.map` | `yvex compile map` | product-default | breaking-cutover | move: adapter projects tensor-map |
| `yvex tokenizer decode` | `tokenizer.decode` | `tokenizer.decode` | `yvex execute tokenizer decode` | engineering | breaking-cutover | move: truthful |
| `yvex tokenizer encode` | `tokenizer.encode` | `tokenizer.encode` | `yvex execute tokenizer encode` | engineering | breaking-cutover | move: truthful |
| `yvex tokenizer prompt` | `tokenizer.prompt` | `tokenizer.prompt` | `yvex execute tokenizer prompt` | engineering | breaking-cutover | move: truthful |
| `yvex tokenizer show` | `tokenizer.show` | `tokenizer.show` | `yvex inspect tokenizer` | product-advanced | breaking-cutover | move: truthful |
| `yvex chat` | `generation.chat` | `generation.chat` | `yvex chat` | product-default | retained | keep: truthful; namespace help absent |
| `yvex` | `generation.chat` | `generation.chat` | `yvex chat` | product-default | breaking-cutover | keep: truthful alias |
| `yvex help` | `command.discovery` | `command.discovery` | `yvex help` | product-default | retained | replace-registry-projection: incomplete and manually duplicated |
| `yvex model list` | `model.list` | `model.list` | `yvex model list` | product-default | retained | rename-or-remove: misleading: identical to model show and does not list models |
| `yvex model show` | `model.show` | `model.show` | `yvex model show` | product-default | retained | rename: misleading: selected configuration is not live runtime identity |
| `yvex model use` | `model.use` | `model.select` | `yvex model select` | product-default | breaking-cutover | normalize: truthful but DeepSeek/CUDA/4096 defaults are client-local policy |
| `yvex run` | `generation.turn` | `generation.turn` | `yvex run` | product-default | retained | keep: truthful; detailed help absent |
| `yvex runtime start` | `runtime.start` | `runtime.start` | `yvex runtime start` | product-default | retained | keep: truthful foreground exec; selected config fallback |
| `yvex runtime status` | `runtime.status` | `runtime.status` | `yvex runtime status` | product-default | retained | keep-and-expand-renderer: human view omits authoritative variant/context facts |
| `yvex runtime stop` | `runtime.stop` | `runtime.stop` | `yvex runtime stop` | product-default | retained | keep: truthful |
| `yvex runtime trace` | `runtime.trace` | `runtime.trace` | `yvex runtime trace` | product-default | retained | normalize-flag: --follow is inert because trace always follows |
| `yvex runtime watch` | `runtime.watch` | `runtime.watch` | `yvex runtime watch` | product-default | retained | replace-renderer: claims operational view but renders generic a/b fields |
| `yvex session attach` | `session.attach` | `session.attach` | `yvex session attach` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session close` | `session.close` | `session.close` | `yvex session close` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session detach` | `session.detach` | `session.detach` | `yvex session detach` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session list` | `session.list` | `session.list` | `yvex session list` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session new` | `session.new` | `session.new` | `yvex session new` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session reset` | `session.reset` | `session.reset` | `yvex session reset` | product-default | retained | keep: truthful; namespace help absent |
| `yvex session show` | `session.show` | `session.show` | `yvex session show` | product-default | retained | keep: truthful; namespace help absent |
| `yvex version` | `system.version` | `system.version` | `yvex version` | product-default | retained | keep: truthful |
| `yvexd` | `runtime.host` | `runtime.host` | `yvexd` | product-advanced | retained | retain-configure: truthful but service policy repeated as flags |
| `/attach` | `session.attach` | `session.attach` | `/attach` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/cancel` | `generation.cancel` | `generation.cancel` | `/cancel` | product-default | retained | retain interaction: hardcoded separately from external dispatch |
| `/close` | `session.close` | `session.close` | `/close` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/detach` | `session.detach` | `session.detach` | `/detach` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/help` | `command.discovery` | `command.discovery` | `/help` | product-default | retained | replace with registry projection: hardcoded separately from external dispatch |
| `/new` | `session.new` | `session.new` | `/new` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/quit` | `repl.quit` | `repl.quit` | `/quit` | product-default | retained | retain interaction: hardcoded separately from external dispatch |
| `/reset` | `session.reset` | `session.reset` | `/reset` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/sessions` | `session.list` | `session.list` | `/sessions` | product-default | retained | retain adapter: hardcoded separately from external dispatch |
| `/status` | `runtime.status` | `console.status` | `/status` | product-default | retained | registry-adapter: truthful transport projection; hardcoded separately from external dispatch |

## Explicit flag removals

- `cli.offline.artifact.verify --model` — remove the hidden integrity-check grammar rewrite.
- `cli.offline.evidence.models --force-sidecars` — remove a parsed but unconsumed option.
- `cli.offline.evidence.models --no-use` — separate acquisition from selected startup state.
- `cli.yvex.model.use --artifact` — resolve startup facts from the selected registry profile.
- `cli.yvex.model.use --backend` — resolve startup facts from the selected registry profile.
- `cli.yvex.model.use --context` — resolve startup facts from the selected registry profile.
- `cli.yvex.model.use --runtime-binding` — resolve startup facts from the selected registry profile.
- `cli.yvex.model.use --target` — resolve startup facts from the selected registry profile.
- `cli.yvex.runtime.trace --follow` — trace is already a continuous subscription.

The retired top-level namespaces `evidence`, `graph`, `quant`, `source`,
`tensor`, and `tokenizer` are refusal-only migration hints, not aliases.
