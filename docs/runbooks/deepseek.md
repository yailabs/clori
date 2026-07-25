# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`; canonical
v0.1.0 target: `deepseek4-v4-flash` on DGX Spark CUDA. Snapshot
`60d8d70770c6776ff598c94bb586a859a38244f1` binds 46 shards and 69,187 records.

## Current Boundary

YVEX admits the selected complete DeepSeek-V4-Flash GGUF, materializes its
attention bindings, and executes the complete attention core and immediate
envelope through the common CPU/CUDA runtime. Persistent KV, transformer/MoE
execution, text generation, and a DeepSeek generation command remain unsupported.
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
./yvex graph attention benchmark --target deepseek4-v4-flash \
  --models-root "$MODELS_ROOT" --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --phase decode --mode "$MODE" --scope full --probe canonical \
  --operation-scope release-attention-set --warmup 3 --repeat 20 --progress off \
  --baseline "$EVIDENCE/$MODE-current.yvex-benchmark" --write-baseline \
  --chart "$EVIDENCE/$MODE-comparison.svg" --output csv >"$EVIDENCE/$MODE-comparison.csv"
./yvex graph attention benchmark compare \
  --baseline "$EVIDENCE/$MODE.yvex-benchmark" \
  --current "$EVIDENCE/$MODE-current.yvex-benchmark" \
  --max-regression-bps 500 --chart "$EVIDENCE/$MODE-file-comparison.svg" \
  --output json
```

Repeat with `MODE=eager` and `MODE=piecewise`. Qualification separates software
acceptance, numerical conformance, and runtime reliability. These files are
attention-component evidence, not model evaluation, agent evaluation or full-model benchmark.
Omitting `--max-regression-bps` reports deltas without a pass threshold.

Regenerate and publish the complete curated chart set from one clean external
evidence directory:

```sh
EVIDENCE="$(mktemp -d /tmp/yvex-runtime-benchmark.XXXXXX)"
make update-runtime-benchmark-charts YVEX_RUNTIME_BENCHMARK_DIR="$EVIDENCE" \
  YVEX_RUNTIME_BINDING="$BINDING"
```

The target measures all three CUDA modes twice, validates the external evidence,
then atomically updates only `../assets/benchmarks/attention/*.svg`.

## Source Verification

```sh
./yvex source-manifest report \
  --release v0.1.0 --family deepseek --target deepseek4-v4-flash \
  --models-root "$HOME/lab/models" --audit --strict
make test-source-payload-live-plan
make test-source-payload-live
```

The plan reads no payload; the live target verifies all 46 shard digests and about 159.6 GB.

## Canonical Control

Current milestone state, dependencies, gates, and Active Next live only in
`PROJECT.md`. See `../v010-release-doctrine.md`, `../system-target.md`, and
`../../MODEL_ARTIFACTS.md` for their owned contracts.
