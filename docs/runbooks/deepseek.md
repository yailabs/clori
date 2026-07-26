# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`; target:
`deepseek4-v4-flash` on DGX Spark CUDA at snapshot `60d8d70770c6776ff598c94bb586a859a38244f1`.

## Current Boundary

YVEX admits the selected complete GGUF and executes its attention bindings
through the common CPU/CUDA runtime. Sessions own exact persistent state for
all 43 layers. Versioned activation bundles commit that state atomically per
chunk. A distinct token-local MoE input executes hash/learned routing, selected
routed experts, shared experts, and combination across all 43 layers. A
schema-v1 numeric token input executes selected embedding rows, the complete 43
block backbone, final mHC collapse, and final RMSNorm. Prompt text, tokenizer
execution, repeated decode, logits, and text generation remain unsupported.

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

The component attention-activation and MoE tensor-file commands remain
available for focused operator diagnosis; `docs/api.md` owns their schemas and
`./yvex help` owns their complete flag catalog.

An upstream production consumer or the focused live target may create an
untracked schema-v1 transformer token input. Execute it with:

```sh
TOKENS="/absolute/path/to/input.yvex-transformer-input"
./yvex graph transformer execute \
  --target deepseek4-v4-flash --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase prefill \
  --input token-ids --input-file "$TOKENS" \
  --chunk-tokens 2 --context-capacity 4096 \
  --progress off --output json
```

The file binds canonical U32 token IDs to the exact logical model, runtime
numeric, descriptor, and transformer-plan identities. The command executes the
production backbone and commits all 43 attention publications once per chunk.
Its normalized hidden result is not tokenizer output, logits, or generation.

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
