# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`.
Canonical v0.1.0 release target: `deepseek4-v4-flash` on DGX Spark CUDA.
The verified snapshot pins commit `60d8d70770c6776ff598c94bb586a859a38244f1`,
46 shards and 69,187 tensor records with independent payload trust.

## Current Boundary

YVEX has produced and admitted the selected complete DeepSeek-V4-Flash GGUF,
materializes its attention bindings, and executes the complete attention core
and immediate envelope through the common CPU/CUDA runtime. It does not yet
own persistent KV, execute the transformer/MoE stack, or generate text.
There is no supported DeepSeek generation command to run yet.

Prepare the immutable runtime binding, then execute the production attention
path. All generated evidence remains outside the repository:

```sh
MODELS_ROOT="$HOME/lab/models"
ARTIFACT="$MODELS_ROOT/gguf/deepseek/deepseek-v4-flash-q8_0-q2_k-v1.gguf"
EVIDENCE="$(mktemp -d /tmp/yvex-runtime-evidence.XXXXXX)"
mkdir "$EVIDENCE/bindings"

./yvex graph attention prepare \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding-dir "$EVIDENCE/bindings" --output json \
  >"$EVIDENCE/prepare.json"
BINDING="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["runtime_binding_path"])' \
  "$EVIDENCE/prepare.json")"

./yvex graph attention execute \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase decode --mode full \
  --operation-scope release-attention-set --probe canonical --scope full \
  --output json

./yvex graph attention qualify \
  --target deepseek4-v4-flash --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" \
  --runtime-binding "$BINDING" --backend cuda --phase decode --mode full \
  --operation-scope release-attention-set --probe canonical --scope full \
  --output json
```

The input is a canonical activation probe, not prompt text. This exact lane
writes an identity-bound baseline, JSON/CSV reports and deterministic SVGs:

```sh
MODE=full
./yvex graph attention benchmark --target deepseek4-v4-flash \
  --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --phase decode --mode "$MODE" --scope full \
  --operation-scope release-attention-set --probe canonical --warmup 3 --repeat 20 \
  --progress off --baseline "$EVIDENCE/$MODE.yvex-benchmark" --write-baseline \
  --chart "$EVIDENCE/$MODE.svg" --output json >"$EVIDENCE/$MODE.json"
./yvex graph attention benchmark --target deepseek4-v4-flash \
  --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --phase decode --mode "$MODE" --scope full \
  --operation-scope release-attention-set --probe canonical --warmup 3 --repeat 20 \
  --progress off --baseline "$EVIDENCE/$MODE-current.yvex-benchmark" --write-baseline \
  --chart "$EVIDENCE/$MODE-comparison.svg" --output csv \
  >"$EVIDENCE/$MODE-comparison.csv"
./yvex graph attention benchmark compare \
  --baseline "$EVIDENCE/$MODE.yvex-benchmark" \
  --current "$EVIDENCE/$MODE-current.yvex-benchmark" \
  --max-regression-bps 500 --output json
```

Repeat with `MODE=eager` and `MODE=piecewise` for all three CUDA modes. The
qualification separates software acceptance, numerical conformance and runtime
reliability. The files measure an attention component; they are runtime-local
evidence, not model evaluation, agent evaluation or full-model benchmark
results. The example permits a caller-selected five-percent ceiling; omitting
`--max-regression-bps` reports measured deltas without manufacturing a pass
threshold.

## Source Verification

```sh
./yvex source-manifest report \
  --release v0.1.0 --family deepseek --target deepseek4-v4-flash \
  --models-root "$HOME/lab/models" --audit --strict
make test-source-payload-live-plan
make test-source-payload-live
```

The plan reads no payload; the live target verifies all 46 shard digests and
about 159.6 GB. This proves source trust, not artifact or runtime support.

## Canonical Control

Current milestone state, dependencies, gates, and Active Next live only in
`PROJECT.md`. See `../v010-release-doctrine.md`, `../system-target.md`, and
`../../MODEL_ARTIFACTS.md` for their owned contracts.
