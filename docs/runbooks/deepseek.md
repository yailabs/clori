# DeepSeek Operator Runbook

## Current Target

Canonical source: `$HOME/lab/models/hf/deepseek/DeepSeek-V4-Flash`; target:
`deepseek4-v4-flash` on DGX Spark CUDA at snapshot `60d8d70770c6776ff598c94bb586a859a38244f1`.

## Current Boundary

YVEX admits the selected complete GGUF through the common CPU/CUDA runtime.
Sessions own exact 43-layer persistent state. Activation bundles, token-local
MoE, and schema-v1 numeric token input execute the complete embedding,
attention/MoE, final mHC, and final RMSNorm backbone. The same schema drives
teacher-forced decode over one warm context. Final-prefill and decode hidden
rows project through the separate BF16 output head to complete raw logits. The
common host sampler selects tokens from every complete row. Prompt text,
tokenizer execution, token append, EOS/stop behavior, and generation remain
unsupported.

There is no supported DeepSeek generation command to run yet. Sampling output
is evidence and is not appended to the model sequence.
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

Component tensor-file commands remain available for focused diagnosis;
`docs/api.md` owns their schemas and `./yvex help` owns the flag catalog. Set
`TOKENS` to an untracked schema-v1 canonical U32 token input.

Split one numeric stream into prefix and teacher-forced decode without reopening the model:

```sh
./yvex graph transformer decode \
  --target deepseek4-v4-flash --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --input token-ids --input-file "$TOKENS" --prefill-tokens 1 \
  --prefill-chunk-tokens 1 --context-capacity 8 \
  --progress off --output json
```

Every remaining ID commits one 43-block step and hidden row; no token is chosen.
Project final-prefill and decode rows through the complete head without reopening:

```sh
./yvex graph transformer logits \
  --target deepseek4-v4-flash --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --input token-ids --input-file "$TOKENS" \
  --prefill-tokens 1 --prefill-chunk-tokens 1 --context-capacity 8 \
  --progress off --output json
```

For one prefix and two decode tokens, three bounded records each cover all
129,280 F32 logits. Output contains digests/ranges, not the tensor or a choice.

Select bounded evidence tokens from those same real rows without feeding them
back into decode:

```sh
./yvex graph transformer sample \
  --target deepseek4-v4-flash --artifact "$ARTIFACT" --runtime-binding "$BINDING" \
  --backend cuda --input token-ids --input-file "$TOKENS" \
  --prefill-tokens 1 --prefill-chunk-tokens 1 --context-capacity 8 \
  --strategy stochastic --temperature 0.8 --top-k 50 --top-p 0.95 \
  --min-p 0.05 --typical-p 1.0 --seed 42 \
  --progress off --output json
```

Use `--strategy greedy` without stochastic parameters for deterministic
maximum-logit selection. The sampler is a common host operation even when CUDA
produced the logits; it changes neither the token stream nor persistent state.

The installed namespace also provides `./yvex graph attention qualify` and
`./yvex graph attention benchmark compare`. The latter accepts
`--max-regression-bps 500` as an explicit operator policy. Attention component
evidence is not model evaluation, agent evaluation or full-model benchmark.

## Canonical Control

Current milestone state, dependencies, gates, and Active Next live only in `PROJECT.md`.
See `../v010-release-doctrine.md`, `../system-target.md`, and `../../MODEL_ARTIFACTS.md`.
