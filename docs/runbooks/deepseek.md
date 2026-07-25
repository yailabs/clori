# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`; target:
`deepseek4-v4-flash` on DGX Spark CUDA at snapshot `60d8d70770c6776ff598c94bb586a859a38244f1`.

## Current Boundary

YVEX admits the selected complete GGUF and executes its attention bindings
through the common CPU/CUDA runtime. Sessions own exact persistent state for
all 43 layers. Versioned activation bundles commit that state atomically per
chunk. Prompt/token embedding, full-model prefill, transformer/MoE execution, and
text generation remain unsupported. There is no supported DeepSeek generation command to run yet.

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

The canonical activation probe is not prompt text. This lane writes an
identity-bound baseline, JSON/CSV reports, and deterministic SVGs:

```sh
MODE=full
./yvex graph attention benchmark --target deepseek4-v4-flash \
  --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --phase decode --mode "$MODE" --scope full --probe canonical \
  --operation-scope release-attention-set --warmup 3 --repeat 20 \
  --progress off --baseline "$EVIDENCE/$MODE.yvex-benchmark" --write-baseline \
  --chart "$EVIDENCE/$MODE.svg" --output json >"$EVIDENCE/$MODE.json"
./yvex graph attention benchmark compare \
  --baseline "$EVIDENCE/$MODE.yvex-benchmark" \
  --current "$EVIDENCE/$MODE-current.yvex-benchmark" \
  --max-regression-bps 500 --chart "$EVIDENCE/$MODE-file-comparison.svg" \
  --output json
```

Repeat with `MODE=eager` and `MODE=piecewise`. These files are component
evidence, not model evaluation, agent evaluation or full-model benchmark. Omitting
`--max-regression-bps` reports deltas without a pass threshold.

Regenerate and publish the chart set from one clean external evidence directory:

```sh
EVIDENCE="$(mktemp -d /tmp/yvex-runtime-benchmark.XXXXXX)"
make update-runtime-benchmark-charts \
  YVEX_RUNTIME_BENCHMARK_DIR="$EVIDENCE" YVEX_RUNTIME_BINDING="$BINDING"
```

The target measures all CUDA modes twice and updates only validated
`../assets/benchmarks/attention/*.svg`.

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
