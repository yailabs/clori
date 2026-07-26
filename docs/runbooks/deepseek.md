# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`; target:
`deepseek4-v4-flash` on DGX Spark CUDA at snapshot `60d8d70770c6776ff598c94bb586a859a38244f1`.

## Current Boundary

YVEX admits the selected complete GGUF and executes its attention bindings
through the common CPU/CUDA runtime. Sessions own exact persistent state for
all 43 layers. Versioned activation bundles commit that state atomically per
chunk. A distinct token-local MoE input executes hash/learned routing, selected
routed experts, shared experts, and combination across all 43 layers. Prompt/
token embedding, full-model prefill, transformer composition, and text
generation remain unsupported.

There is no supported DeepSeek generation command to run yet.

Prepare the immutable runtime binding, then execute the production attention
path. Raw generated evidence remains outside the repository:

```sh
MODELS_ROOT="$HOME/lab/models"
ARTIFACT="$MODELS_ROOT/gguf/deepseek/deepseek-v4-flash-q8_0-q2_k-v1.gguf"
EVIDENCE="$(mktemp -d /tmp/yvex-runtime-evidence.XXXXXX)"; mkdir "$EVIDENCE/bindings"
./yvex graph attention prepare \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding-dir "$EVIDENCE/bindings" --output json \
  >"$EVIDENCE/prepare.json"
BINDING="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_binding_path"])' \
  "$EVIDENCE/prepare.json")"
./yvex graph attention execute \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase decode --mode full \
  --operation-scope release-attention-set --probe canonical --scope full --output json
./yvex graph attention qualify \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase decode --mode full \
  --operation-scope release-attention-set --probe canonical --scope full --output json
```

An upstream production consumer or the focused live target may create an
untracked schema-v1 activation bundle. Execute one with:

```sh
ACTIVATIONS="/absolute/path/to/input.yvex-activations"
./yvex graph attention execute \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase prefill --mode eager \
  --operation-scope core --input tensor-file --input-file "$ACTIVATIONS" \
  --chunk-tokens 2 --context-capacity 4096 --scope full \
  --progress off --output json
```

The file binds exact runtime identities, token/layer geometry, payload ranges,
and digest. It reports attention/state facts, never a model or prompt output.

An upstream production consumer or the focused live target may also create an
untracked schema-v1 MoE input. Execute it with:

```sh
MOE_INPUT="/absolute/path/to/input.yvex-moe-input"
./yvex graph moe execute \
  --target deepseek4-v4-flash --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda \
  --input tensor-file --input-file "$MOE_INPUT" \
  --scope full --progress off --output json
```

The file contains one exact expanded activation per main layer plus numeric
token IDs required by the first three hash-router layers. Those IDs do not
establish tokenizer support. The result is token-local MoE output and deferred
transformer-owned state; it neither mutates KV nor produces model output.

The installed namespace also provides `./yvex graph attention qualify` and
`./yvex graph attention benchmark compare`. The latter accepts
`--max-regression-bps 500` as an explicit operator policy. Attention component
evidence is not model evaluation, agent evaluation or full-model benchmark.

## Source Verification

```sh
./yvex source-manifest report \
  --release v0.1.0 --family deepseek --target deepseek4-v4-flash --models-root "$HOME/lab/models" --audit --strict
make test-source-payload-live-plan
make test-source-payload-live
```

## Canonical Control

Current milestone state, dependencies, gates, and Active Next live only in `PROJECT.md`.
See `../v010-release-doctrine.md`, `../system-target.md`, and `../../MODEL_ARTIFACTS.md`.
