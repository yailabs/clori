#!/usr/bin/env sh
set -eu

. tests/support/cleanup.sh

YVEX_BIN="${YVEX_BIN:-./yvex}"
ROOT=${YVEX_TEST_OUT_DIR:-build/tests/models-cli}
REG="$ROOT/models.local.json"
CATALOG_ROOT="$ROOT/catalog-empty"
GGUF="$ROOT/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"

matches() {
  file=$1
  pattern=$2
  grep -E -- "$pattern" "$file" >/dev/null
}

make_missing_role_source() {
  dir=$1
  variant=${2:-complete}
  yvex_test_cleanup "$dir"
  mkdir -p "$dir"
  if [ "$variant" != "missing-metadata" ]; then
    cat > "$dir/config.json" <<'JSON'
{
  "model_type": "qwen",
  "vocab_size": 16,
  "hidden_size": 8,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "unk_token_id": 3
}
JSON
    cat > "$dir/tokenizer_config.json" <<'JSON'
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "unk_token_id": 3
}
JSON
    cat > "$dir/special_tokens_map.json" <<'JSON'
{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "additional_special_tokens": []
}
JSON
    cat > "$dir/generation_config.json" <<'JSON'
{
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}
JSON
    printf '{"version":"1.0","model":{"type":"BPE"}}\n' > "$dir/tokenizer.json"
  fi
  python3 - "$dir/model.safetensors" "$variant" <<'PY'
import json
import struct
import sys

variant = sys.argv[2]
items = [
    ("model.embed_tokens.weight", [16, 8]),
    ("model.layers.0.self_attn.q_proj.weight", [8, 8]),
    ("model.layers.0.self_attn.k_proj.weight", [8, 8]),
    ("model.layers.0.self_attn.v_proj.weight", [8, 8]),
    ("model.layers.0.self_attn.o_proj.weight", [8, 8]),
    ("model.layers.0.mlp.gate_proj.weight", [16, 8]),
    ("model.layers.0.mlp.up_proj.weight", [16, 8]),
    ("model.layers.0.mlp.down_proj.weight", [8, 16]),
    ("model.layers.0.input_layernorm.weight", [8]),
    ("model.layers.0.post_attention_layernorm.weight", [8]),
    ("model.norm.weight", [8]),
    ("lm_head.weight", [16, 8]),
]
if variant == "missing-attention-k":
    items = [item for item in items if item[0] != "model.layers.0.self_attn.k_proj.weight"]
if variant == "missing-output-head":
    items = [item for item in items if item[0] != "lm_head.weight"]
if variant == "ambiguous-output-head":
    items.append(("output.weight", [16, 8]))
offset = 0
header = {}
for name, shape in items:
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * 4
    header[name] = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + nbytes],
    }
    offset += nbytes
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY
}

expect_rc() {
  expected=$1
  shift
  set +e
  "$@"
  rc=$?
  set -e
  test "$rc" -eq "$expected"
}

assert_output_contract_pass() {
  file=$1
  grep 'status: pass' "$file"
  grep 'runtime_claim: unsupported' "$file"
  grep 'generation: unsupported-full-model' "$file"
  grep 'benchmark_status: not-measured' "$file"
  grep 'release_ready: false' "$file"
  grep 'boundary: output-contract check only; no runtime/generation claim' "$file"
  ! grep 'status: fail-' "$file"
  ! grep 'generation_ready: tr''ue' "$file"
  ! grep 'release_ready: tr''ue' "$file"
  ! grep 'benchmark_status: mea''sured' "$file"
}

write_fake_transformer_safetensors() {
  out=$1
  variant=${2:-complete}
  dtype=${3:-F32}
  mkdir -p "$(dirname "$out")"
  python3 - "$out" "$variant" "$dtype" <<'PY'
import json
import struct
import sys

variant = sys.argv[2]
dtype = sys.argv[3]
element_bytes = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
}.get(dtype, 4)
tensor_bytes = element_bytes * 4
names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]
if variant == "qwen-incomplete":
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.unmapped.qwen_probe.weight",
    ]
elif variant == "qwen-coverage":
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.self_attn.k_norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.layers.0.mlp.experts.down_proj",
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
        "model.language_model.layers.0.optional_probe.weight",
    ]
elif variant == "gemma-incomplete-no-head":
    names = [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.norm.weight",
        "model.unmapped.gemma_probe.weight",
    ]
elif variant == "gemma-language-head":
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.self_attn.k_norm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.pre_feedforward_layernorm.weight",
        "model.language_model.layers.0.post_feedforward_layernorm.weight",
        "model.language_model.layers.0.layer_scalar",
        "model.language_model.norm.weight",
        "model.language_model.lm_head.weight",
    ]
elif variant == "gemma-language-tied":
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.self_attn.q_norm.weight",
        "model.language_model.layers.0.self_attn.k_norm.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.pre_feedforward_layernorm.weight",
        "model.language_model.layers.0.post_feedforward_layernorm.weight",
        "model.language_model.layers.0.layer_scalar",
        "model.language_model.norm.weight",
    ]
elif variant == "gemma-language-no-head":
    names = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.o_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.post_attention_layernorm.weight",
        "model.language_model.layers.0.pre_feedforward_layernorm.weight",
        "model.language_model.layers.0.post_feedforward_layernorm.weight",
        "model.language_model.layers.0.layer_scalar",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.unmapped_probe.weight",
    ]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": dtype,
        "shape": [2, 2],
        "data_offsets": [offset, offset + tensor_bytes],
    }
    offset += tensor_bytes
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY
}

write_fake_tokenizer_sidecars() {
  dir=$1
  family=$2
  mkdir -p "$dir"
  cat > "$dir/config.json" <<JSON
{
  "model_type": "$family",
  "vocab_size": 2,
  "hidden_size": 2,
  "bos_token_id": 1,
  "eos_token_id": 1,
  "pad_token_id": 0,
  "unk_token_id": 0,
  "tie_word_embeddings": false
}
JSON
  cat > "$dir/tokenizer_config.json" <<'JSON'
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "bos_token_id": 1,
  "eos_token_id": 1,
  "pad_token_id": 0,
  "unk_token_id": 0,
  "chat_template": "{{ bos_token }}{{ messages }}"
}
JSON
  cat > "$dir/special_tokens_map.json" <<'JSON'
{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]
}
JSON
  cat > "$dir/generation_config.json" <<'JSON'
{
  "bos_token_id": 1,
  "eos_token_id": 1,
  "pad_token_id": 0
}
JSON
  cat > "$dir/tokenizer.json" <<'JSON'
{
  "version": "1.0",
  "model": {"type": "BPE"},
  "vocab_size": 2,
  "added_tokens": [{"id": 1, "content": "</s>"}]
}
JSON
  if [ "$family" = "qwen" ]; then
    printf '{"<pad>":0,"</s>":1}\n' > "$dir/vocab.json"
    printf '#version: 0.2\n< p\n' > "$dir/merges.txt"
    printf '{{ bos_token }}{{ messages }}\n' > "$dir/chat_template.jinja"
  else
    rm -f "$dir/vocab.json" "$dir/merges.txt" "$dir/chat_template.jinja"
  fi
}

yvex_test_cleanup "$ROOT"
mkdir -p "$ROOT"
case "$ROOT" in
  /*)
    XDG_CONFIG_HOME="$ROOT/xdg"
    XDG_DATA_HOME="$ROOT/xdg-data"
    XDG_RUNTIME_DIR="$ROOT/runtime"
    YVEX_DATA_DIR="$ROOT/data"
    ;;
  *)
    XDG_CONFIG_HOME="$PWD/$ROOT/xdg"
    XDG_DATA_HOME="$PWD/$ROOT/xdg-data"
    XDG_RUNTIME_DIR="$PWD/$ROOT/runtime"
    YVEX_DATA_DIR="$PWD/$ROOT/data"
    ;;
esac
export XDG_CONFIG_HOME XDG_DATA_HOME XDG_RUNTIME_DIR YVEX_DATA_DIR
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_RUNTIME_DIR" "$YVEX_DATA_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p "$CATALOG_ROOT"

FAKE_HF="$PWD/tests/fixtures/bin/fake-hf"
FAKE_HF_LOG="$ROOT/fake-hf-discovery.log"
export YVEX_FAKE_HF_LOG="$FAKE_HF_LOG"

RECON_ROOT="$ROOT/catalog-reconcile"
RECON_REV=b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08
RECON_SOURCE="$RECON_ROOT/source/hf/MiniMaxAI/MiniMax-H3/$RECON_REV"
mkdir -p "$RECON_SOURCE"
cat > "$RECON_SOURCE/yvex-source-acquisition.json" <<JSON
{"schema":"yvex.source-acquisition.v1","repository":"MiniMaxAI/MiniMax-H3",\
"revision":"$RECON_REV","acquisition_complete":true,"source_bytes":144016000740}
JSON
mkdir -p "$RECON_ROOT/registry/minimax-h3"
cat > "$RECON_ROOT/registry/minimax-h3/MiniMax-H3.download.json" <<JSON
{"schema":"yvex.model_download.registry.v1","target_id":"MiniMax-H3",
"family":"minimax-h3","provider":"huggingface","repo_id":"MiniMaxAI/MiniMax-H3",
"revision":"$RECON_REV","local_source_dir":"$RECON_SOURCE",
"status":"model-download-pass","total_regular_file_bytes":144016000740,
"safetensors_count":1,"upstream_identity_verified":true,"payload_hash_verified":true}
JSON
"$YVEX_BIN" source list --models-root "$RECON_ROOT" --registry "$REG" --json \
  > "$ROOT/reconciled-source.json"
python3 - "$ROOT/reconciled-source.json" <<'PY'
import json
import sys

sources = json.load(open(sys.argv[1], encoding="utf-8"))["sources"]
source = next(item for item in sources if item["name"] == "MiniMax-H3")
assert source["representation"] == "safetensors-source"
assert source["acquisition_state"] == "source-acquired"
assert source["verification_state"] == "payload-verification-recorded"
assert source["size_bytes"] == 144016000740
PY

COLUMNS=240 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" model search "MiniMax H3" --limit 2 \
  --models-root "$RECON_ROOT" \
  > "$ROOT/search.out"
grep 'REMOTE MODELS · "MiniMax H3"' "$ROOT/search.out"
grep 'MODEL.*PROVIDER.*REPOSITORY.*ARCH.*FORMATS.*SIZE.*STATE.*YVEX.*LOCATION' "$ROOT/search.out"
grep 'MiniMax-H3.*Hugging Face.*MiniMaxAI/MiniMax-H3.*safetensors.*source.*supported.*hf://MiniMaxAI/MiniMax-H3@' "$ROOT/search.out"
grep 'MiniMax-H3-GGUF.*Hugging Face.*unsloth/MiniMax-H3-GGUF.*gguf.*remote.*inspect.*hf://unsloth/MiniMax-H3-GGUF@' "$ROOT/search.out"
! grep 'package-preparation\|source-ingest\|physical-inspection' "$ROOT/search.out"

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" model search "MiniMax H3" --all --json \
  --models-root "$RECON_ROOT" > "$ROOT/search-all.json"
python3 - "$ROOT/search-all.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
assert catalog["schema"] == "yvex.model-catalog-projection.v1"
assert catalog["authorities"] == ["remote-provider", "local-catalog"]
models = catalog["models"]
assert models[0]["repository"] == "MiniMaxAI/MiniMax-H3"
assert models[0]["kind"] == "full model"
assert models[0]["model_identity"] == "MiniMaxAI/MiniMax-H3"
assert models[0]["family_affinity"] == "minimax-h3"
assert models[0]["local_source"] is True
by_repo = {model["repository"]: model for model in models}
assert by_repo["fal/MiniMax-H3-Realism-People-LoRA"]["kind"] == "adapter / LoRA"
assert by_repo["fal/MiniMax-H3-Realism-People-LoRA"]["model_identity"] == ""
assert by_repo["LBH-123-AI/Minimax_h3_latent_Upscaler"]["kind"] == "component"
assert by_repo["unsloth/MiniMax-H3-GGUF"]["kind"] == "conversion"
assert by_repo["community/MiniMax-H3-finetune"]["kind"] == "derivative / fork"
assert by_repo["community/unknown-model"]["kind"] == "unknown"
PY

YVEX_FAKE_HF_RESOLVED_SHA=42ed227ee7df40d41602854ae760620d6eb651fe \
  YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" model search "MiniMax H3" --limit 1 --json \
  --models-root "$RECON_ROOT" > "$ROOT/search-other-revision.json"
python3 - "$ROOT/search-other-revision.json" "$RECON_REV" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["models"][0]
assert model["repository"] == "MiniMaxAI/MiniMax-H3"
assert model["kind"] == "full model"
assert model["canonical"] is True
assert model["local_source"] is False
assert model["local_related_revision"] is True
assert model["local_source_revision"] == sys.argv[2]
assert model["product_status"] == "acquirable"
PY

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" model search "MiniMax H3" --page 2 --limit 2 \
  --models-root "$RECON_ROOT" --json \
  > "$ROOT/search-page.json"
python3 - "$ROOT/search-page.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
assert catalog["schema"] == "yvex.model-catalog-projection.v1"
assert [model["repository"] for model in catalog["models"]] == [
    "community/MiniMax-H3-finetune",
    "fal/MiniMax-H3-Realism-People-LoRA",
]
PY

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source inspect MiniMaxAI/MiniMax-H3 \
  --models-root "$RECON_ROOT" --audit \
  > "$ROOT/remote-inspect.out"
grep 'revision_reference: default' "$ROOT/remote-inspect.out"
grep 'resolved_revision: b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08' \
  "$ROOT/remote-inspect.out"
grep 'family: minimax-h3' "$ROOT/remote-inspect.out"
grep 'kind: full model' "$ROOT/remote-inspect.out"
grep 'local_source: true' "$ROOT/remote-inspect.out"
grep 'base_model: MiniMaxAI/MiniMax-H3-Base' "$ROOT/remote-inspect.out"
grep 'identity=safetensors-source format=safetensors precision=BF16+F16' \
  "$ROOT/remote-inspect.out"
grep 'identity=gguf-Q4_K_M format=gguf precision=Q4_K_M evidence=filename-hint' \
  "$ROOT/remote-inspect.out"
grep 'selector=model-Q4_K_M.gguf' "$ROOT/remote-inspect.out"
grep 'identity=gguf-IQ2_XXS format=gguf precision=IQ2_XXS evidence=filename-hint' \
  "$ROOT/remote-inspect.out"
grep 'file\[0\]: path=config.json kind=configuration representation=configuration' \
  "$ROOT/remote-inspect.out"
grep 'path=tokenizer.json kind=tokenizer representation=tokenizer' "$ROOT/remote-inspect.out"
grep 'path=README.md kind=sidecar representation=sidecar' "$ROOT/remote-inspect.out"

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source inspect MiniMaxAI/MiniMax-H3 \
  --models-root "$RECON_ROOT" --json \
  > "$ROOT/remote-inspect.json"
python3 - "$ROOT/remote-inspect.json" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["models"][0]
assert model["requested_revision"] == "default"
assert model["resolved_revision"] == "b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
assert model["local_source"] is True
assert {file["kind"] for file in model["files"]} >= {
    "configuration",
    "tokenizer",
    "safetensors",
    "gguf",
    "sidecar",
}
assert next(file for file in model["files"] if file["path"] == "model-Q4_K_M.gguf")[
    "representation"
] == "gguf-Q4_K_M"
PY

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source inspect unsloth/MiniMax-H3-GGUF \
  --output table > "$ROOT/inspect-gguf.out"
grep 'kind        conversion' "$ROOT/inspect-gguf.out"
grep 'Q4_K_M (filename)' "$ROOT/inspect-gguf.out"
grep 'acquire-and-inspect-required' "$ROOT/inspect-gguf.out"

YVEX_FAKE_HF_DISCOVERY_MODE=model-not-found YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 1 "$YVEX_BIN" source inspect missing/model \
  > "$ROOT/inspect-model-missing.out" 2> "$ROOT/inspect-model-missing.err"
grep 'remote model was not found' "$ROOT/inspect-model-missing.err"
YVEX_FAKE_HF_DISCOVERY_MODE=revision-not-found YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 1 "$YVEX_BIN" source inspect MiniMaxAI/MiniMax-H3 --revision DOES_NOT_EXIST \
  > "$ROOT/inspect-revision-missing.out" 2> "$ROOT/inspect-revision-missing.err"
grep 'remote revision or reference was not found' "$ROOT/inspect-revision-missing.err"

! grep -A2 '^  download$' "$FAKE_HF_LOG"

expect_rc 2 "$YVEX_BIN" model search MiniMax --interactive \
    > "$ROOT/search-interactive.out" 2> "$ROOT/search-interactive.err"
grep 'model search: unknown flag: --interactive' "$ROOT/search-interactive.err"

YVEX_FAKE_HF_RESOLVED_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  YVEX_HF_CLI="$FAKE_HF" expect_rc 1 "$YVEX_BIN" source inspect MiniMaxAI/MiniMax-H3 \
  --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 \
  > "$ROOT/inspect-revision-mismatch.out" 2> "$ROOT/inspect-revision-mismatch.err"
grep 'provider resolved revision does not match the requested identity' \
  "$ROOT/inspect-revision-mismatch.err"

YVEX_FAKE_HF_RESOLVED_SHA=62af8fffb2f7030cac4de2f0169f5b8d1101b646 \
  YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source inspect \
  deepseek-ai/DeepSeek-V4-Flash-DSpark --revision \
  62af8fffb2f7030cac4de2f0169f5b8d1101b646 --audit > "$ROOT/inspect-deepseek.out"
grep 'family: deepseek' "$ROOT/inspect-deepseek.out"
grep 'support_stage: package-preparation' "$ROOT/inspect-deepseek.out"

YVEX_FAKE_HF_DISCOVERY_MODE=unsafe-file YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 4 "$YVEX_BIN" source inspect MiniMaxAI/MiniMax-H3 \
  > "$ROOT/inspect-unsafe-file.out" 2> "$ROOT/inspect-unsafe-file.err"
grep 'provider file listing is malformed or oversized' "$ROOT/inspect-unsafe-file.err"

YVEX_FAKE_HF_DISCOVERY_MODE=empty YVEX_HF_CLI="$FAKE_HF" \
  "$YVEX_BIN" model search none > "$ROOT/search-empty.out"
grep '^REMOTE MODELS · "none"' "$ROOT/search-empty.out"

YVEX_FAKE_HF_DISCOVERY_MODE=malformed YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 4 "$YVEX_BIN" model search malformed > "$ROOT/search-malformed.out" \
  2> "$ROOT/search-malformed.err"
grep 'provider search did not return a JSON array' "$ROOT/search-malformed.err"

YVEX_FAKE_HF_DISCOVERY_AUTH=1 YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 1 "$YVEX_BIN" model search private > "$ROOT/search-auth.out" \
  2> "$ROOT/search-auth.err"
grep 'authentication is required' "$ROOT/search-auth.err"

YVEX_FAKE_HF_FAIL=1 YVEX_HF_CLI="$FAKE_HF" \
  expect_rc 1 "$YVEX_BIN" model search failed > "$ROOT/search-fail.out" \
  2> "$ROOT/search-fail.err"
grep 'provider operation failed' "$ROOT/search-fail.err"

HF_TOKEN='discovery-secret-must-not-leak' YVEX_HF_CLI="$FAKE_HF" \
  "$YVEX_BIN" model search redaction --limit 1 --json > "$ROOT/search-redaction.json"
! grep 'discovery-secret-must-not-leak' "$FAKE_HF_LOG"
! grep 'discovery-secret-must-not-leak' "$ROOT/search-redaction.json"

"$YVEX_BIN" compile artifact emit \
  --out "$GGUF" \
  --model-name model-registry-test \
  --arch llama \
  --overwrite >/dev/null

BINDING="$PWD/$ROOT/runtime.binding"
ARTIFACT="$PWD/$GGUF"
printf 'binding fixture\n' > "$BINDING"

"$YVEX_BIN" profile scan --root "$ROOT" --registry "$REG" > "$ROOT/scan.out"
grep 'candidate: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/scan.out"
grep 'status: models-scan' "$ROOT/scan.out"

"$YVEX_BIN" profile create --path "$ARTIFACT" --registry "$REG" \
  --support-level selected-tensor-materialized \
  --runtime-binding "$BINDING" --target deepseek4-v4-flash-dspark \
  --backend cpu --execution-strategy speculative --ctx 4096 > "$ROOT/add.out"
grep 'alias: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/add.out"
grep 'status: models-added' "$ROOT/add.out"
test -f "$REG"

"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" > "$ROOT/list.out"
grep 'deepseek4-v4-flash-dspark-selected-embed' "$ROOT/list.out"
grep 'DEPLOYMENT PROFILES' "$ROOT/list.out"
grep 'cpu/text/speculative' "$ROOT/list.out"
grep 'context=4096' "$ROOT/list.out"
grep 'runnable' "$ROOT/list.out"

"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" --output table \
  > "$ROOT/list-table.out"
grep 'deepseek4-v4-flash-dspark-selected-embed' "$ROOT/list-table.out"
grep 'runnable' "$ROOT/list-table.out"

"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" --json \
  > "$ROOT/list-audit.out"
python3 - "$ROOT/list-audit.out" <<'PY'
import json
import sys

profiles = json.load(open(sys.argv[1], encoding="utf-8"))["profiles"]
profile, = profiles
assert profile["identity"] == "deepseek4-v4-flash-dspark-selected-embed"
assert profile["launchable"] is False
assert profile["blocker"].startswith("malformed-binding:")
assert profile["backend"] == "cpu"
assert profile["execution_strategy"] == "speculative"
PY

"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" --output nope \
  > "$ROOT/list-bad-output.out" 2> "$ROOT/list-bad-output.err" && exit 1 || true
grep 'model plumbing --output requires table|audit|json' "$ROOT/list-bad-output.err"

"$YVEX_BIN" profile show deepseek4-v4-flash-dspark-selected-embed --registry "$REG" > "$ROOT/inspect.out"
grep 'model: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/inspect.out"
grep 'family: deepseek4 class=embed' "$ROOT/inspect.out"
grep 'artifact: support=selected-tensor-materialized execution=not-established-by-inspection' "$ROOT/inspect.out"
grep 'runtime profile: unavailable (malformed-binding:' \
  "$ROOT/inspect.out"
grep 'status: models-inspect' "$ROOT/inspect.out"
test "$(wc -l < "$ROOT/inspect.out")" -le 8

"$YVEX_BIN" profile show deepseek4-v4-flash-dspark-selected-embed --registry "$REG" --audit > "$ROOT/inspect-audit.out"
grep 'alias: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/inspect-audit.out"
grep 'artifact_support_level: selected-tensor-materialized' "$ROOT/inspect-audit.out"
grep 'artifact_execution_ready: false' "$ROOT/inspect-audit.out"
grep 'startup_profile_status: unavailable' "$ROOT/inspect-audit.out"
grep 'deployment_compatibility: malformed-binding' "$ROOT/inspect-audit.out"
grep 'gguf:' "$ROOT/inspect-audit.out"
grep 'tensor_count: 1' "$ROOT/inspect-audit.out"
grep 'status: models-inspect' "$ROOT/inspect-audit.out"

COMPOSITE_ROOT=$(realpath "$ROOT")
cat > "$COMPOSITE_ROOT/repository.json" <<JSON
{"repository":"MiniMaxAI/MiniMax-H3","requested_revision":"$RECON_REV",\
"resolved_revision":"$RECON_REV"}
JSON
"$YVEX_BIN" profile create --path "$ARTIFACT" --registry "$REG" \
  --alias minimax-h3-fl2va-runtime-media --family minimax-h3 \
  --startup-profile composite --installation-root "$COMPOSITE_ROOT" \
  --target minimax-h3-fl2va --backend cuda \
  > "$ROOT/add-composite.out"
"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" \
  > "$ROOT/list-composite.out"
grep 'minimax-h3-fl2va-runtime-media' "$ROOT/list-composite.out"
grep 'minimax-h3-fl2va · 0/1 runnable' "$ROOT/list-composite.out"
grep 'cuda/media/not-applicable' "$ROOT/list-composite.out"
grep 'required composite deployment component is unavailable' \
  "$ROOT/list-composite.out"

# Presence alone is not a composite deployment binding. Four readable GGUF
# files with the wrong family/component identities remain non-launchable.
mkdir -p "$COMPOSITE_ROOT/physical-v3" "$COMPOSITE_ROOT/physical-v4" \
  "$COMPOSITE_ROOT/physical"
cp "$ARTIFACT" "$COMPOSITE_ROOT/physical-v3/text_encoder.gguf"
cp "$ARTIFACT" "$COMPOSITE_ROOT/physical-v4/transformer.gguf"
cp "$ARTIFACT" "$COMPOSITE_ROOT/physical/video_vae.gguf"
cp "$ARTIFACT" "$COMPOSITE_ROOT/physical/audio_vae.gguf"
"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" \
  > "$ROOT/list-composite-mismatch.out"
grep 'minimax-h3-fl2va · 0/1 runnable' "$ROOT/list-composite-mismatch.out"
grep 'component does not match the current family execution contract' \
  "$ROOT/list-composite-mismatch.out"

"$YVEX_BIN" model list --models-root "$CATALOG_ROOT" --registry "$REG" \
  > "$ROOT/library-friendly.out"
grep '^MODELS$' "$ROOT/library-friendly.out"
grep 'v4-flash.*BLOCKED.*not current' "$ROOT/library-friendly.out"
grep 'minimax-h3-fl2va.*BLOCKED.*not current' "$ROOT/library-friendly.out"
! grep 'provider:huggingface' "$ROOT/library-friendly.out"
"$YVEX_BIN" source list --models-root "$RECON_ROOT" --registry "$REG" \
  > "$ROOT/sources-friendly.out"
grep '^SOURCES$' "$ROOT/sources-friendly.out"
grep 'model binding:' "$ROOT/sources-friendly.out"
"$YVEX_BIN" artifact list --models-root "$CATALOG_ROOT" --registry "$REG" \
  > "$ROOT/artifacts-friendly.out"
grep '^ARTIFACTS$' "$ROOT/artifacts-friendly.out"
grep 'runnable profiles' "$ROOT/artifacts-friendly.out"
! grep 'ready=' "$ROOT/artifacts-friendly.out"
"$YVEX_BIN" model list --models-root "$CATALOG_ROOT" --registry "$REG" --json \
  > "$ROOT/library-friendly.json"
"$YVEX_BIN" artifact list --models-root "$CATALOG_ROOT" --registry "$REG" --json \
  > "$ROOT/artifacts-friendly.json"
"$YVEX_BIN" profile list --models-root "$CATALOG_ROOT" --registry "$REG" --json \
  > "$ROOT/profiles-friendly.json"
python3 - "$ROOT/library-friendly.json" "$ROOT/artifacts-friendly.json" \
  "$ROOT/profiles-friendly.json" <<'PY'
import json
import sys

models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
artifacts = json.load(open(sys.argv[2], encoding="utf-8"))["artifacts"]
profiles = json.load(open(sys.argv[3], encoding="utf-8"))["profiles"]
assert {model["name"] for model in models} == {
    "DeepSeek-V4-Flash",
    "minimax-h3-fl2va",
}
assert all(artifact["profile_count"] == 1 for artifact in artifacts)
assert all(artifact["launchable_profile_count"] == 0 for artifact in artifacts)
assert all(profile["deployment_class"] for profile in profiles)
assert all("runtime_binding" in profile for profile in profiles)
PY
"$YVEX_BIN" profile show minimax-h3-fl2va-runtime-media --registry "$REG" --audit \
  > "$ROOT/show-composite.out"
grep 'runtime_profile: composite' "$ROOT/show-composite.out"
grep "runtime_installation: $COMPOSITE_ROOT" "$ROOT/show-composite.out"
grep 'runtime_binding: $' "$ROOT/show-composite.out"
grep 'runtime_context: 0' "$ROOT/show-composite.out"
grep 'startup_profile_status: unavailable' "$ROOT/show-composite.out"
grep 'deployment_compatibility: artifact-mismatch' "$ROOT/show-composite.out"
YVEX_MODELS_REGISTRY="$REG" YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source inspect \
  MiniMaxAI/MiniMax-H3 --models-root "$RECON_ROOT" --json \
  > "$ROOT/reconciled-package.json"
python3 - "$ROOT/reconciled-package.json" <<'PY'
import json
import sys

model = json.load(open(sys.argv[1], encoding="utf-8"))["models"][0]
assert model["local_source"] is True
assert model["local_package"] is True
assert model["product_status"] == "supported"
PY
set +e
"$YVEX_BIN" profile create --path "$ARTIFACT" --registry "$REG" \
  --alias minimax-h3-fl2va-runtime-incomplete --family minimax-h3 \
  --startup-profile composite --target minimax-h3-fl2va --backend cuda \
  > "$ROOT/add-composite-bad.out" \
  2> "$ROOT/add-composite-bad.err"
composite_bad_status=$?
set -e
test "$composite_bad_status" -eq 2
grep 'composite startup profile requires --installation-root' \
  "$ROOT/add-composite-bad.err"

"$YVEX_BIN" profile show deepseek4-v4-flash-dspark-selected-embed --registry "$REG" --output nope > "$ROOT/inspect-bad-output.out" 2> "$ROOT/inspect-bad-output.err" && exit 1 || true
grep 'unsupported output mode: nope' "$ROOT/inspect-bad-output.err"

"$YVEX_BIN" profile create --path "$ARTIFACT" --registry "$REG" \
  --alias deepseek4-v4-flash-dspark-runtime-incomplete > "$ROOT/add-incomplete.out"
SERVER_RUNTIME="$ROOT/server-runtime"
yvex_test_cleanup "$SERVER_RUNTIME"
mkdir -m 700 "$SERVER_RUNTIME"
(
  server_pid=
  finish_model_host()
  {
    status=$?
    trap - EXIT HUP INT TERM
    if test -n "$server_pid" && kill -0 "$server_pid" 2>/dev/null; then
      XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" host stop >/dev/null 2>&1 || true
      wait "$server_pid" 2>/dev/null || true
    fi
    exit "$status"
  }
  trap finish_model_host EXIT HUP INT TERM

  YVEX_MODELS_REGISTRY="$REG" XDG_RUNTIME_DIR="$SERVER_RUNTIME" \
    "$YVEX_BIN" serve --logs off --openai off \
    > "$ROOT/lifecycle-host.out" 2> "$ROOT/lifecycle-host.err" &
  server_pid=$!
  ready=0
  attempt=0
  while test "$attempt" -lt 100; do
    if XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" host status --json \
        > "$ROOT/lifecycle-status.json" 2> "$ROOT/lifecycle-status.err"; then
      ready=1
      break
    fi
    kill -0 "$server_pid" 2>/dev/null || break
    attempt=$((attempt + 1))
    sleep 0.02
  done
  test "$ready" -eq 1

  XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" profile list \
    --models-root "$CATALOG_ROOT" --registry "$REG" --json \
    > "$ROOT/list-host-observed.json"
  python3 - "$ROOT/list-host-observed.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
profiles = {profile["identity"]: profile for profile in catalog["profiles"]}
assert "deepseek4-v4-flash-dspark-selected-embed" in profiles
assert "minimax-h3-fl2va-runtime-media" in profiles
assert all("engine_state" not in profile for profile in profiles.values())
PY

  expect_rc 1 env XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" engine load \
    deepseek4-v4-flash-dspark-runtime-incomplete \
    > "$ROOT/use-incomplete.out" 2> "$ROOT/use-incomplete.err"
  grep 'model has no complete startup profile' "$ROOT/use-incomplete.err"
  "$YVEX_BIN" profile remove deepseek4-v4-flash-dspark-runtime-incomplete \
    --registry "$REG" > "$ROOT/remove-incomplete.out"

  "$YVEX_BIN" profile remove deepseek4-v4-flash-dspark-selected-embed \
    --registry "$REG" > "$ROOT/remove.out"
  grep 'removed: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/remove.out"
  grep 'status: models-removed' "$ROOT/remove.out"
  "$YVEX_BIN" profile remove minimax-h3-fl2va-runtime-media --registry "$REG" \
    > "$ROOT/remove-composite.out"

  expect_rc 1 env XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" engine load missing \
    > "$ROOT/use-missing.out" 2> "$ROOT/use-missing.err"
  grep 'profile is not registered: missing' "$ROOT/use-missing.err"

  XDG_RUNTIME_DIR="$SERVER_RUNTIME" "$YVEX_BIN" host stop \
    > "$ROOT/lifecycle-stop.out" 2> "$ROOT/lifecycle-stop.err"
  wait "$server_pid"
  server_pid=
)

"$YVEX_BIN" help --advanced > "$ROOT/help.out"
grep 'yvex source acquire' "$ROOT/help.out"
grep 'yvex source status' "$ROOT/help.out"
grep 'yvex compile' "$ROOT/help.out"
grep 'yvex artifact status' "$ROOT/help.out"
"$YVEX_BIN" source acquire --help > "$ROOT/help-acquire.out"
grep -- '--auth' "$ROOT/help-acquire.out"
grep -- '--progress' "$ROOT/help-acquire.out"
grep -- '--revision' "$ROOT/help-acquire.out"
grep -- '--include' "$ROOT/help-acquire.out"
"$YVEX_BIN" compile --help > "$ROOT/help-prepare.out"
grep -- '--audit' "$ROOT/help-prepare.out"

FAKE_GH="$PWD/tests/fixtures/bin/fake-gh"
DOWNLOAD_ROOT="$ROOT/download"
export YVEX_CONFIG_DIR="$ROOT/accounts-config"

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --dry-run --models-root "$DOWNLOAD_ROOT" --auth never --audit > "$ROOT/download-dry-run.out"
grep 'status: model-download-dry-run' "$ROOT/download-dry-run.out"
grep 'family: gemma' "$ROOT/download-dry-run.out"
grep 'stage: account-provider skipped' "$ROOT/download-dry-run.out"
grep 'payload_loaded: false' "$ROOT/download-dry-run.out"
grep 'gguf_created: false' "$ROOT/download-dry-run.out"
grep 'generation: unsupported' "$ROOT/download-dry-run.out"
! grep 'tick: elapsed=' "$ROOT/download-dry-run.out"
test ! -e "$DOWNLOAD_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
test ! -e "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.download.receipt"
test ! -e "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.download.active.json"
test ! -e "$DOWNLOAD_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
test ! -e "$DOWNLOAD_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$DOWNLOAD_ROOT" --auth auto --audit > "$ROOT/download-gemma.out"
grep 'status: model-download-pass' "$ROOT/download-gemma.out"
grep 'provider: huggingface' "$ROOT/download-gemma.out"
grep 'stage: account-provider pass' "$ROOT/download-gemma.out"
grep 'stage: provider-cli pass' "$ROOT/download-gemma.out"
grep 'stage: source-manifest pass' "$ROOT/download-gemma.out"
grep 'stage: native-inventory pass' "$ROOT/download-gemma.out"
grep 'stage: progress-stream pass' "$ROOT/download-gemma.out"
grep 'source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08' "$ROOT/download-gemma.out"
grep 'gguf_created: false' "$ROOT/download-gemma.out"
grep 'payload_loaded: false' "$ROOT/download-gemma.out"
grep 'generation: unsupported' "$ROOT/download-gemma.out"
grep 'benchmark_status: not-measured' "$ROOT/download-gemma.out"
test -f "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.source-manifest.json"
grep '"status": "in-progress"' "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.source-manifest.json"
test -f "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.native-inventory.json"
test -f "$DOWNLOAD_ROOT/evidence/build/gemma/gemma-4-12b-it.download-report.json"
test -f "$DOWNLOAD_ROOT/registry/gemma/gemma-4-12b-it.download.json"
! find "$DOWNLOAD_ROOT/gguf" -type f -name '*.gguf' 2>/dev/null | grep .

"$YVEX_BIN" source list --models-root "$DOWNLOAD_ROOT" --registry "$REG" --json \
  > "$ROOT/local-catalog.json"
python3 - "$ROOT/local-catalog.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
source = next(item for item in catalog["sources"] if item["name"] == "gemma-4-12b-it")
assert source["representation"] == "safetensors-source"
assert source["acquisition_state"] == "source-acquired"
assert source["verification_state"] == "revision-verified"
assert source["blocker"] == ""
PY

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire \
  --repo community/minimax-h3-gguf --family minimax-h3 --name minimax-h3-community-gguf \
  --revision 2222222222222222222222222222222222222222 \
  --include model-Q4_K_M.gguf --models-root "$DOWNLOAD_ROOT" --auth auto \
  --no-native-inventory --audit > "$ROOT/download-gguf.out"
grep 'status: model-download-pass' "$ROOT/download-gguf.out"
grep 'safetensors_count: 0' "$ROOT/download-gguf.out"
grep 'gguf_count: 1' "$ROOT/download-gguf.out"
grep 'boundary: acquired GGUF, structural inspection and package admission required' \
  "$ROOT/download-gguf.out"
"$YVEX_BIN" source list --models-root "$DOWNLOAD_ROOT" --registry "$REG" --json \
  > "$ROOT/local-catalog-gguf.json"
python3 - "$ROOT/local-catalog-gguf.json" <<'PY'
import json
import sys

catalog = json.load(open(sys.argv[1], encoding="utf-8"))
source = next(
    item for item in catalog["sources"] if item["name"] == "minimax-h3-community-gguf"
)
assert source["representation"] == "gguf"
assert source["acquisition_state"] == "source-acquired"
assert source["verification_state"] == "revision-verified"
assert source["blocker"] == ""
PY

LIVE_ROOT="$ROOT/download-live"
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_STEP_DELAY=1 YVEX_FAKE_HF_STEPS=3 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$LIVE_ROOT" --auth required --progress plain --tick-seconds 1 --audit > "$ROOT/download-live.out" 2>&1 &
LIVE_PID=$!
sleep 1
grep 'model-download: start target=gemma-4-12b-it' "$ROOT/download-live.out"
grep 'stage: download running' "$ROOT/download-live.out"
wait "$LIVE_PID"
grep 'tick: elapsed=' "$ROOT/download-live.out"
grep 'files=' "$ROOT/download-live.out"
grep 'bytes=' "$ROOT/download-live.out"
grep 'fake-hf: resolving repo' "$ROOT/download-live.out"
grep 'fake-hf: downloading shard' "$ROOT/download-live.out"
grep 'progress_mode: plain' "$ROOT/download-live.out"
grep 'tick_seconds: 1' "$ROOT/download-live.out"
grep 'stdout_streamed: true' "$ROOT/download-live.out"
grep 'stderr_streamed: true' "$ROOT/download-live.out"
grep 'provider_exit_code: 0' "$ROOT/download-live.out"
test -f "$LIVE_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
test -f "$LIVE_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"
grep 'fake-hf: resolving repo' "$LIVE_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
grep 'fake-hf: stderr resolving repo' "$LIVE_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"

LIVE_FAIL_ROOT="$ROOT/download-live-fail"
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_FAIL_AT_STEP=2 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$LIVE_FAIL_ROOT" --progress plain --tick-seconds 1 --audit > "$ROOT/download-live-fail.out" 2>&1 && exit 1 || true
grep 'status: model-download-fail' "$ROOT/download-live-fail.out"
grep 'provider_exit_code: 43' "$ROOT/download-live-fail.out"
grep 'stdout_log:' "$ROOT/download-live-fail.out"
grep 'stderr_log:' "$ROOT/download-live-fail.out"
grep 'top_blocker: provider-download-failed' "$ROOT/download-live-fail.out"
test -f "$LIVE_FAIL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
test -f "$LIVE_FAIL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"
grep 'fake-hf: downloading shard 1' "$LIVE_FAIL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
grep 'fake-hf: failing at step 2' "$LIVE_FAIL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"

SIGNAL_ROOT="$ROOT/download-signal"
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_STEP_DELAY=5 YVEX_FAKE_HF_STEPS=8 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$SIGNAL_ROOT" --progress plain --tick-seconds 1 --audit > "$ROOT/download-signal.out" 2>&1 &
SIGNAL_PID=$!
sleep 1
kill -INT "$SIGNAL_PID"
set +e
wait "$SIGNAL_PID"
SIGNAL_RC=$?
set -e
test "$SIGNAL_RC" -ne 0
grep 'status: model-download-interrupted' "$ROOT/download-signal.out"
grep 'stage: download interrupted' "$ROOT/download-signal.out"
grep 'signal: SIGINT' "$ROOT/download-signal.out"
grep 'child_signal_forwarded: true' "$ROOT/download-signal.out"
grep 'child_exit_status: interrupted' "$ROOT/download-signal.out"
grep 'orphan_check_performed: true' "$ROOT/download-signal.out"
grep 'orphan_check_status: pass' "$ROOT/download-signal.out"
grep 'partial_source_preserved: true' "$ROOT/download-signal.out"
grep 'lock_cleanup: not-attempted' "$ROOT/download-signal.out"
grep 'lock_files_deleted: false' "$ROOT/download-signal.out"
grep 'provider_pid:' "$ROOT/download-signal.out"
grep 'provider_process_group:' "$ROOT/download-signal.out"
PROVIDER_PID=$(awk '/provider_pid:/ { print $2; exit }' "$ROOT/download-signal.out")
PROVIDER_PGID=$(awk '/provider_process_group:/ { print $2; exit }' "$ROOT/download-signal.out")
test -n "$PROVIDER_PID"
test -n "$PROVIDER_PGID"
! kill -0 "$PROVIDER_PID" 2>/dev/null
! ps -o pid= -g "$PROVIDER_PGID" | grep .
test -f "$SIGNAL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
test -f "$SIGNAL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"
test -f "$SIGNAL_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/config.json"
grep 'fake-hf: resolving repo' "$SIGNAL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stdout.log"
grep 'fake-hf: stderr resolving repo' "$SIGNAL_ROOT/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"

CONTROL_ROOT="$ROOT/download-control"
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_STEP_DELAY=5 YVEX_FAKE_HF_STEPS=8 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$CONTROL_ROOT" --auth required --progress log --tick-seconds 1 --audit > "$ROOT/download-control-run.out" 2>&1 &
CONTROL_PID=$!
i=0
while [ ! -f "$CONTROL_ROOT/evidence/build/gemma/gemma-4-12b-it.download.active.json" ] && [ "$i" -lt 10 ]; do
  sleep 1
  i=$((i + 1))
done
test -f "$CONTROL_ROOT/evidence/build/gemma/gemma-4-12b-it.download.active.json"

"$YVEX_BIN" source status gemma-4-12b-it --models-root "$CONTROL_ROOT" --audit > "$ROOT/download-control-status-running.out"
grep 'status: model-download-status' "$ROOT/download-control-status-running.out"
grep 'receipt_status: active' "$ROOT/download-control-status-running.out"
grep 'provider_process_alive: true' "$ROOT/download-control-status-running.out"
grep 'stop_available: true' "$ROOT/download-control-status-running.out"
grep 'resume_available: false' "$ROOT/download-control-status-running.out"

"$YVEX_BIN" source stop gemma-4-12b-it --models-root "$CONTROL_ROOT" --audit > "$ROOT/download-control-stop.out"
grep 'status: model-download-stopped' "$ROOT/download-control-stop.out"
grep 'child_signal_forwarded: true' "$ROOT/download-control-stop.out"
grep 'cleanup: preserved-partial-source' "$ROOT/download-control-stop.out"
set +e
wait "$CONTROL_PID"
CONTROL_RC=$?
set -e
test "$CONTROL_RC" -ne 0
test -f "$CONTROL_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/config.json"

"$YVEX_BIN" source status gemma-4-12b-it --models-root "$CONTROL_ROOT" --audit > "$ROOT/download-control-status-stopped.out"
grep 'provider_process_alive: false' "$ROOT/download-control-status-stopped.out"
grep 'last_receipt_status: stopped' "$ROOT/download-control-status-stopped.out"
grep 'resume_available: true' "$ROOT/download-control-status-stopped.out"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source resume gemma-4-12b-it --models-root "$CONTROL_ROOT" --auth required --progress log --tick-seconds 1 --audit > "$ROOT/download-control-resume.out" 2>&1
grep 'status: model-download-resume-pass' "$ROOT/download-control-resume.out"
grep 'stage: download pass' "$ROOT/download-control-resume.out"
test -f "$CONTROL_ROOT/evidence/build/gemma/gemma-4-12b-it.download.last.json"
"$YVEX_BIN" source status gemma-4-12b-it --models-root "$CONTROL_ROOT" --audit > "$ROOT/download-control-status-resumed.out"
grep 'last_receipt_status: pass' "$ROOT/download-control-status-resumed.out"

STALE_ROOT="$ROOT/download-control-stale"
mkdir -p "$STALE_ROOT/evidence/build/gemma" "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
cat > "$STALE_ROOT/evidence/build/gemma/gemma-4-12b-it.download.active.json" <<EOF
{
  "schema": "yvex.model_download.active.v1",
  "target_id": "gemma-4-12b-it",
  "family": "gemma",
  "provider": "huggingface",
  "repo_id": "google/gemma-4-12B-it",
  "revision": "b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08",
  "local_source_dir": "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08",
  "provider_pid": 99999999,
  "provider_pgid": 99999999,
  "status": "running"
}
EOF
"$YVEX_BIN" source status gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$STALE_ROOT" --audit > "$ROOT/download-control-stale-status.out"
grep 'receipt_status: stale-active-receipt' "$ROOT/download-control-stale-status.out"

mkdir -p "$STALE_ROOT/cache/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/huggingface/download"
ln -s "$PWD/$STALE_ROOT/cache/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache"
printf 'lock\n' > "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache/huggingface/download/model.safetensors.lock"
YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source resume gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$STALE_ROOT" --auth required --audit > "$ROOT/download-control-lock-blocked.out" 2>&1 && exit 1 || true
grep 'status: model-download-resume-blocked' "$ROOT/download-control-lock-blocked.out"
grep 'top_blocker: stale-lock-candidates' "$ROOT/download-control-lock-blocked.out"

"$YVEX_BIN" source cleanup gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$STALE_ROOT" --stale-locks --dry-run --audit > "$ROOT/download-control-cleanup-dry-run.out"
grep 'status: model-download-cleanup-dry-run' "$ROOT/download-control-cleanup-dry-run.out"
grep 'stale_locks: 1' "$ROOT/download-control-cleanup-dry-run.out"
test -f "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache/huggingface/download/model.safetensors.lock"

"$YVEX_BIN" source cleanup gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$STALE_ROOT" --stale-locks --yes --audit > "$ROOT/download-control-cleanup.out"
grep 'status: model-download-cleanup' "$ROOT/download-control-cleanup.out"
grep 'deleted: 1' "$ROOT/download-control-cleanup.out"
test ! -f "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache/huggingface/download/model.safetensors.lock"

printf 'lock\n' > "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache/huggingface/download/model.safetensors.lock"
YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source resume gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$STALE_ROOT" --auth required --clear-stale-locks --audit > "$ROOT/download-control-lock-clear-resume.out" 2>&1
grep 'status: model-download-resume-pass' "$ROOT/download-control-lock-clear-resume.out"
test ! -f "$STALE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/.cache/huggingface/download/model.safetensors.lock"

SAFE_ROOT="$ROOT/download-control-safe"
SAFE_SRC="$SAFE_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
mkdir -p "$SAFE_SRC"
python3 - "$SAFE_SRC/model-ok.safetensors" "$SAFE_SRC/model-truncated.safetensors" <<'PY'
import json
import struct
import sys

def write(path, payload_len, actual_len):
    header = {
        "__metadata__": {"format": "pt"},
        "embed.weight": {
            "dtype": "F16",
            "shape": [1, max(1, payload_len // 2)],
            "data_offsets": [0, payload_len],
        },
    }
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(bytes(actual_len))

write(sys.argv[1], 16, 16)
write(sys.argv[2], 32, 8)
PY
"$YVEX_BIN" source status gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$SAFE_ROOT" --audit > "$ROOT/download-control-safe-truncated.out"
grep 'safetensors_header_checked: true' "$ROOT/download-control-safe-truncated.out"
grep 'safetensors_size_status: truncated' "$ROOT/download-control-safe-truncated.out"
rm -f "$SAFE_SRC/model-truncated.safetensors"
"$YVEX_BIN" source status gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$SAFE_ROOT" --audit > "$ROOT/download-control-safe-ok.out"
grep 'safetensors_size_status: ok' "$ROOT/download-control-safe-ok.out"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-e2b --models-root "$ROOT/download-off" --no-progress --audit > "$ROOT/download-off.out" 2>&1
! grep 'model-download: start' "$ROOT/download-off.out"
! grep 'tick: elapsed=' "$ROOT/download-off.out"
! grep 'fake-hf: resolving repo' "$ROOT/download-off.out"
grep 'status: model-download-pass' "$ROOT/download-off.out"

LOG_PROGRESS_ROOT="$ROOT/download-log-progress"
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_STEP_DELAY=1 YVEX_FAKE_HF_STEPS=3 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-e2b-it --models-root "$LOG_PROGRESS_ROOT" --progress log --tick-seconds 1 --audit > "$ROOT/download-log-progress.out" 2>&1
grep 'tick: elapsed=' "$ROOT/download-log-progress.out"
! grep 'fake-hf: resolving repo' "$ROOT/download-log-progress.out"
grep 'progress_mode: log' "$ROOT/download-log-progress.out"
test -f "$LOG_PROGRESS_ROOT/evidence/build/acquisition/gemma-4-e2b-it.download.stdout.log"
grep 'fake-hf: resolving repo' "$LOG_PROGRESS_ROOT/evidence/build/acquisition/gemma-4-e2b-it.download.stdout.log"

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-e2b --models-root "$DOWNLOAD_ROOT/noauth" --auth never --audit > "$ROOT/download-auth-never.out"
grep 'stage: account-provider skipped' "$ROOT/download-auth-never.out"
grep 'status: model-download-pass' "$ROOT/download-auth-never.out"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire qwen3-8b --models-root "$DOWNLOAD_ROOT" --auth auto --audit > "$ROOT/download-qwen.out"
grep 'family: qwen' "$ROOT/download-qwen.out"
grep 'source/hf/Qwen/Qwen3-8B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08' "$ROOT/download-qwen.out"
grep 'status: model-download-pass' "$ROOT/download-qwen.out"

DYNAMIC_ROOT="$ROOT/download-dynamic-targets"
mkdir -p "$DYNAMIC_ROOT/evidence/fixtures/deepseek" "$DYNAMIC_ROOT/evidence/fixtures/qwen" "$DYNAMIC_ROOT/evidence/fixtures/gemma"
printf 'selected deepseek fixture\n' > "$DYNAMIC_ROOT/evidence/fixtures/deepseek/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
printf 'selected deepseek rmsnorm fixture\n' > "$DYNAMIC_ROOT/evidence/fixtures/deepseek/deepseek4-v4-flash-dspark-selected-embed-rmsnorm-F16-noimatrix-yvex-v1.gguf"
printf 'selected qwen fixture\n' > "$DYNAMIC_ROOT/evidence/fixtures/qwen/qwen3-8b-selected-embed-F16-noimatrix-yvex-v1.gguf"
YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire --repo Qwen/Qwen3.6-35B-A3B --family qwen --name qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --auth auto --progress off --audit > "$ROOT/download-dynamic-qwen.out"
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/download-dynamic-qwen.out"
grep 'repo_id: Qwen/Qwen3.6-35B-A3B' "$ROOT/download-dynamic-qwen.out"
test -f "$DYNAMIC_ROOT/registry/qwen/qwen3-6-35b-a3b.download.json"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.download-report.json"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.source-manifest.json"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.native-inventory.json"
"$YVEX_BIN" source status qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/download-dynamic-qwen-status.out"
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/download-dynamic-qwen-status.out"
grep 'family: qwen' "$ROOT/download-dynamic-qwen-status.out"
grep 'repo_id: Qwen/Qwen3.6-35B-A3B' "$ROOT/download-dynamic-qwen-status.out"
grep 'safetensors_count: 2' "$ROOT/download-dynamic-qwen-status.out"
grep 'safetensors_size_status: ok' "$ROOT/download-dynamic-qwen-status.out"
grep 'status: model-download-status' "$ROOT/download-dynamic-qwen-status.out"
"$YVEX_BIN" source cleanup qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --stale-locks --dry-run --audit > "$ROOT/download-dynamic-qwen-cleanup.out"
grep 'model-download-cleanup: target=qwen3-6-35b-a3b' "$ROOT/download-dynamic-qwen-cleanup.out"
grep 'status: model-download-cleanup-dry-run' "$ROOT/download-dynamic-qwen-cleanup.out"
rm -f "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/"*.safetensors
write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" qwen-coverage BF16
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_status: naming-map-candidate' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_family: qwen' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_target_id: qwen3-6-35b-a3b' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.embed_tokens.weight -> model.embedding.token.weight' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.self_attn.q_proj.weight -> model.layers.0.attention.q_proj.weight' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.linear_attn.A_log -> model.layers.0.qwen_linear_attn.A_log' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.mlp.gate.weight -> model.layers.0.moe.router.weight' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.mlp.experts.gate_up_proj -> model.layers.0.moe.experts.all.gate_up_proj.weight' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.mlp.shared_expert.down_proj.weight -> model.layers.0.moe.shared_expert.down_proj.weight' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_qwen_linear_attn_count: [1-9]' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_moe_router_count: [1-9]' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_moe_expert_count: [1-9]' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_moe_shared_count: [1-9]' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'tensor_map_required_role_coverage_status: required-groups-present' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-dynamic-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-dynamic-qwen-audit.out"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tensor-map.json"
grep '"row": "MODELS.SOURCE.MAP.HANDOFF.0"' "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tensor-map.json"
grep '"required_role_coverage_status": "required-groups-present"' "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tensor-map.json"
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/tensor-map-dynamic-qwen-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-dynamic-qwen-table.out"
grep -F 'FAMILY  TARGET                STATUS                      TOTAL   EMBED    ATTN     MLP    NORM    HEAD     MOE   UNKNOWN   LAYERS  NEXT' "$ROOT/tensor-map-dynamic-qwen-table.out"
grep -F 'qwen    qwen3-6-35b-a3b       naming-map-candidate' "$ROOT/tensor-map-dynamic-qwen-table.out"
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role output-head --audit > "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'output_head_map_family: qwen' "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'output_head_map_target_id: qwen3-6-35b-a3b' "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'output_head_native_name: lm_head.weight' "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-dynamic-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-dynamic-qwen-audit.out"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.output-head-map.json"
grep '"row": "MODELS.SOURCE.MAP.HANDOFF.0"' "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.output-head-map.json"
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role output-head --output table > "$ROOT/output-head-dynamic-qwen-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-dynamic-qwen-table.out"
grep -F 'FAMILY  TARGET                STATUS                           HEAD  FINAL_NORM  EMBED  TIE_POLICY                          SHAPE_RELATION            NEXT' "$ROOT/output-head-dynamic-qwen-table.out"
grep -F 'qwen    qwen3-6-35b-a3b       output-head-profiled             yes' "$ROOT/output-head-dynamic-qwen-table.out"
write_fake_tokenizer_sidecars "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" qwen
"$YVEX_BIN" inspect target tokenizer-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" > "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'tokenizer-map: qwen3-6-35b-a3b' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'family: qwen' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'status: present-report-only' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'tokenizer: present' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'vocab: present' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'merges: present' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'chat_template: present' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'specials: present' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'runtime: unsupported' "$ROOT/tokenizer-map-dynamic-qwen.out"
grep 'next: V010.QUANT.1' "$ROOT/tokenizer-map-dynamic-qwen.out"
"$YVEX_BIN" inspect target tokenizer-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/tokenizer-map-dynamic-qwen-table.out"
grep 'TOKENIZER METADATA MAP' "$ROOT/tokenizer-map-dynamic-qwen-table.out"
grep -F 'TARGET                FAMILY  STATUS               TOKENIZER  VOCAB                         MERGES                  CHAT_TEMPLATE  SPECIALS  NEXT' "$ROOT/tokenizer-map-dynamic-qwen-table.out"
matches "$ROOT/tokenizer-map-dynamic-qwen-table.out" '^qwen3-6-35b-a3b[[:space:]]{2,}qwen[[:space:]]{2,}present-report-only[[:space:]]{2,}yes[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}V010\.QUANT\.1$'
"$YVEX_BIN" inspect target tokenizer-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'tokenizer_map_target_id: qwen3-6-35b-a3b' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'evidence_basis: sidecar-json-only' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'vocab_status: present' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'merges_status: present' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'tokenizer_backend_type: BPE' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'added_tokens_count: 1' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'special_tokens_status: present' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'stop_token_candidate.0.id: 1' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'chat_template_hash_status: not-computed' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'prompt_template_status: present-report-only' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'detokenization_status: not-implemented' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'gguf_tokenizer_contract_status: planned' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
grep 'next_required_rows: V010.QUANT.1' "$ROOT/tokenizer-map-dynamic-qwen-audit.out"
"$YVEX_BIN" inspect target tokenizer-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output json > "$ROOT/tokenizer-map-dynamic-qwen-json.out"
grep '"status":"present-report-only"' "$ROOT/tokenizer-map-dynamic-qwen-json.out"
grep '"target_id":"qwen3-6-35b-a3b"' "$ROOT/tokenizer-map-dynamic-qwen-json.out"
grep '"next":"V010.QUANT.1"' "$ROOT/tokenizer-map-dynamic-qwen-json.out"
test -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tokenizer-map.json"
grep '"schema_version": "yvex.source.tokenizer_map.v1"' "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tokenizer-map.json"
grep '"tokenizer_map_status": "present-report-only"' "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tokenizer-map.json"
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role tokenizer --audit > "$ROOT/tokenizer-map-dynamic-qwen-compat-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/tokenizer-map-dynamic-qwen-compat-audit.out"
"$YVEX_BIN" compile source manifest report --family qwen --release v0.1.0 --source "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/source-dynamic-qwen-audit.out"
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/source-dynamic-qwen-audit.out"
grep 'model: Qwen3.6-35B-A3B' "$ROOT/source-dynamic-qwen-audit.out"
! grep 'target_id: qwen3-8b' "$ROOT/source-dynamic-qwen-audit.out"
grep 'source_manifest_path: .*qwen3-6-35b-a3b.source-manifest.json' "$ROOT/source-dynamic-qwen-audit.out"
grep 'native_inventory_path: .*qwen3-6-35b-a3b.native-inventory.json' "$ROOT/source-dynamic-qwen-audit.out"
grep 'source_manifest_status: present' "$ROOT/source-dynamic-qwen-audit.out"
grep 'native_inventory_report_status: available-report-only' "$ROOT/source-dynamic-qwen-audit.out"
grep 'tensor_map_status: available-report-only' "$ROOT/source-dynamic-qwen-audit.out"
grep 'tensor_role_map_status: available-report-only' "$ROOT/source-dynamic-qwen-audit.out"
grep 'output_head_map_status: available-report-only' "$ROOT/source-dynamic-qwen-audit.out"
grep 'tokenizer_map_path: .*qwen3-6-35b-a3b.tokenizer-map.json' "$ROOT/source-dynamic-qwen-audit.out"
grep 'tokenizer_map_status: available-report-only' "$ROOT/source-dynamic-qwen-audit.out"
grep 'next_required_rows: V010.QUANT.1' "$ROOT/source-dynamic-qwen-audit.out"
! grep 'missing-qwen-tensor-role-map' "$ROOT/source-dynamic-qwen-audit.out"
! grep 'missing-qwen-tensor-map' "$ROOT/source-dynamic-qwen-audit.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" > "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'missing-roles: qwen3-6-35b-a3b' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'top_blocker: quant-policy-or-artifact-emitter' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'next: V010.QUANT.1' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'qwen-linear-attn.*present' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'moe-router.*present' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'moe-experts.*present' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'shared-expert.*present' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'unknown-tensors.*unclassified-header-name' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
grep 'tokenizer.*present-report-only' "$ROOT/missing-roles-dynamic-qwen-coverage.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'tensor_map_status: present-report-only' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'role_group.qwen_linear_attn.status: present' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'role_group.moe_router.status: present' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'role_group.moe_experts.status: present' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'role_group.shared_expert.status: present' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'role_group.unknown_tensors.status: unclassified-header-name' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'top_blocker: quant-policy-or-artifact-emitter' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
grep 'next: V010.QUANT.1' "$ROOT/missing-roles-dynamic-qwen-coverage-audit.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output json > "$ROOT/missing-roles-dynamic-qwen-coverage-json.out"
grep '"top_blocker":"quant-policy-or-artifact-emitter"' "$ROOT/missing-roles-dynamic-qwen-coverage-json.out"
grep '"qwen_linear_attn":"present"' "$ROOT/missing-roles-dynamic-qwen-coverage-json.out"
grep '"shared_expert":"present"' "$ROOT/missing-roles-dynamic-qwen-coverage-json.out"
grep '"tokenizer":"present-report-only"' "$ROOT/missing-roles-dynamic-qwen-coverage-json.out"
"$YVEX_BIN" compile qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --dry-run --audit > "$ROOT/prepare-dynamic-qwen-coverage.out" 2>&1 && exit 1 || true
grep 'tensor_map_status: present-report-only' "$ROOT/prepare-dynamic-qwen-coverage.out"
grep 'output_head_map_status: present-report-only' "$ROOT/prepare-dynamic-qwen-coverage.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/prepare-dynamic-qwen-coverage.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/prepare-dynamic-qwen-coverage.out"
grep 'reason: family quantization plan unimplemented' "$ROOT/prepare-dynamic-qwen-coverage.out"
grep 'next: not-scheduled' "$ROOT/prepare-dynamic-qwen-coverage.out"
"$YVEX_BIN" inspect target quant-policy qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role-support > "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'qtype-role-support: qwen3-6-35b-a3b' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'family: qwen' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'status: blocked' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'source_dtype: BF16' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'preferred_artifact_qtype: unresolved' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'supported_roles: [1-9][0-9]*' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'blocked_roles: [1-9][0-9]*' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'next: not-scheduled' "$ROOT/qtype-role-support-dynamic-qwen.out"
grep 'boundary: qtype role report only; no quantization/GGUF/runtime/generation' "$ROOT/qtype-role-support-dynamic-qwen.out"
! grep 'runtime_claim:' "$ROOT/qtype-role-support-dynamic-qwen.out"
"$YVEX_BIN" inspect target quant-policy qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role-support --output table > "$ROOT/qtype-role-support-dynamic-qwen-table.out"
grep 'QTYPE ROLE SUPPORT' "$ROOT/qtype-role-support-dynamic-qwen-table.out"
grep 'ROLE[[:space:]][[:space:]]*SRC_DTYPE[[:space:]][[:space:]]*ARTIFACT_QTYPE[[:space:]][[:space:]]*STORAGE[[:space:]][[:space:]]*COMPUTE[[:space:]][[:space:]]*CALIBRATION[[:space:]][[:space:]]*STATUS' "$ROOT/qtype-role-support-dynamic-qwen-table.out"
grep 'qwen_linear_attn_A_log[[:space:]][[:space:]]*BF16[[:space:]][[:space:]]*unresolved[[:space:]][[:space:]]*header-storage-profiled[[:space:]][[:space:]]*cpu-cuda-available[[:space:]][[:space:]]*deferred[[:space:]][[:space:]]*present' "$ROOT/qtype-role-support-dynamic-qwen-table.out"
grep 'moe_expert_gate_up[[:space:]][[:space:]]*BF16' "$ROOT/qtype-role-support-dynamic-qwen-table.out"
"$YVEX_BIN" inspect target quant-policy qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role-support --audit > "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'report: qtype-role-support' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'source_dtype: BF16' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.role_name: qwen_linear_attn_A_log' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.role_name: tokenizer_metadata' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.source_dtype: BF16' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.compute_support_status: cpu-cuda-available' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.artifact_emission_allowed: false' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'role\.[0-9][0-9]*\.artifact_emission_blocker: family-quantization-plan-unimplemented' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'payload_bytes_read: false' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'quantization_performed: false' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'gguf_emitted: false' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/qtype-role-support-dynamic-qwen-audit.out"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role-support --output json > "$ROOT/qtype-role-support-json.out" 2> "$ROOT/qtype-role-support-json.err"
grep 'JSON output is unsupported' "$ROOT/qtype-role-support-json.err"
rm -f "$DYNAMIC_ROOT/evidence/build/qwen/qwen3-6-35b-a3b.tokenizer-map.json"
write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" qwen-incomplete
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-qwen-incomplete-audit.out"
grep 'tensor_map_status: naming-map-incomplete' "$ROOT/tensor-map-dynamic-qwen-incomplete-audit.out"
grep 'tensor_map_target_id: qwen3-6-35b-a3b' "$ROOT/tensor-map-dynamic-qwen-incomplete-audit.out"
grep 'unmapped_unknown_count: [1-9]' "$ROOT/tensor-map-dynamic-qwen-incomplete-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --role output-head --audit > "$ROOT/output-head-dynamic-qwen-incomplete-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/output-head-dynamic-qwen-incomplete-audit.out"
grep 'output_head_map_target_id: qwen3-6-35b-a3b' "$ROOT/output-head-dynamic-qwen-incomplete-audit.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" > "$ROOT/missing-roles-dynamic-qwen.out"
grep 'missing-roles: qwen3-6-35b-a3b' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'family: qwen' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'status: blocked' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'top_blocker: incomplete-tensor-map' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'qwen-linear-attn.*missing' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'output-head.*present' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'tokenizer.*missing' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'artifact.*missing' "$ROOT/missing-roles-dynamic-qwen.out"
grep 'boundary: missing-role report only; no GGUF/runtime/generation' "$ROOT/missing-roles-dynamic-qwen.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/missing-roles-dynamic-qwen-table.out"
grep 'qwen3-6-35b-a3b.*qwen.*blocked.*incomplete-tensor-map' "$ROOT/missing-roles-dynamic-qwen-table.out"
grep 'qwen3-6-35b-a3b.*missing.*missing.*V010.MAP.8' "$ROOT/missing-roles-dynamic-qwen-table.out"
grep 'qwen3-6-35b-a3b.*[[:space:]][1-9][0-9]*[[:space:]]*missing[[:space:]]*missing' "$ROOT/missing-roles-dynamic-qwen-table.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'family: qwen' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'source_status: present' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'tensor_map_status: incomplete-report-only' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'tensor_map_path: .*qwen3-6-35b-a3b.tensor-map.json' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'tensor_map_unmapped_unknown_count: [1-9]' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'output_head_map_status: present-report-only' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'tokenizer_map_status: missing' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'artifact_status: missing' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'expected_artifact_path: .*qwen3-6-35b-a3b.gguf' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'artifact_emission_status: not-performed' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'artifact_identity_status: missing' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'prepare_blocker_count:' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'top_blocker: incomplete-tensor-map' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'next: V010.MAP.8' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'runtime_execution: not-performed' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'generation: unsupported' "$ROOT/missing-roles-dynamic-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/missing-roles-dynamic-qwen-audit.out"
"$YVEX_BIN" inspect target missing-roles qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --output json > "$ROOT/missing-roles-dynamic-qwen-json.out"
grep '"target_id":"qwen3-6-35b-a3b"' "$ROOT/missing-roles-dynamic-qwen-json.out"
grep '"top_blocker":"incomplete-tensor-map"' "$ROOT/missing-roles-dynamic-qwen-json.out"
"$YVEX_BIN" compile qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --dry-run --audit > "$ROOT/prepare-dynamic-qwen.out" 2>&1 && exit 1 || true
grep 'target_id: qwen3-6-35b-a3b' "$ROOT/prepare-dynamic-qwen.out"
grep 'family: qwen' "$ROOT/prepare-dynamic-qwen.out"
grep 'source_status: present' "$ROOT/prepare-dynamic-qwen.out"
grep 'model_class_status: present' "$ROOT/prepare-dynamic-qwen.out"
grep 'tensor_map_status: incomplete-report-only' "$ROOT/prepare-dynamic-qwen.out"
grep 'output_head_map_status: present-report-only' "$ROOT/prepare-dynamic-qwen.out"
grep 'tokenizer_map_status: missing' "$ROOT/prepare-dynamic-qwen.out"
grep 'artifact_status: missing' "$ROOT/prepare-dynamic-qwen.out"
grep 'expected_artifact_path: .*qwen3-6-35b-a3b.gguf' "$ROOT/prepare-dynamic-qwen.out"
grep 'artifact_plan_status: planned-full-gguf' "$ROOT/prepare-dynamic-qwen.out"
grep 'artifact_emission_status: not-performed' "$ROOT/prepare-dynamic-qwen.out"
grep 'artifact_identity_status: missing' "$ROOT/prepare-dynamic-qwen.out"
grep 'prepare_blocker_count:' "$ROOT/prepare-dynamic-qwen.out"
grep 'top_blocker: incomplete-tensor-map' "$ROOT/prepare-dynamic-qwen.out"
grep 'reason: incomplete tensor map / tokenizer metadata mapping / artifact path missing' "$ROOT/prepare-dynamic-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/prepare-dynamic-qwen.out"
grep 'status: model-prepare-unsupported' "$ROOT/prepare-dynamic-qwen.out"
! grep 'status: model-prepare-unknown-target' "$ROOT/prepare-dynamic-qwen.out"
! grep 'reason: missing compile map / model class / artifact path' "$ROOT/prepare-dynamic-qwen.out"
"$YVEX_BIN" compile qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --dry-run > "$ROOT/prepare-dynamic-qwen-normal.out" 2>&1 && exit 1 || true
grep 'models prepare: qwen3-6-35b-a3b \[blocked\]' "$ROOT/prepare-dynamic-qwen-normal.out"
grep 'family: qwen  source: present  artifact: missing' "$ROOT/prepare-dynamic-qwen-normal.out"
grep 'plan: full-gguf planned  emission: not-performed' "$ROOT/prepare-dynamic-qwen-normal.out"
grep 'top_blocker: incomplete-tensor-map' "$ROOT/prepare-dynamic-qwen-normal.out"
grep 'next: V010.MAP.8' "$ROOT/prepare-dynamic-qwen-normal.out"
grep 'boundary: prepare dry-run only; no artifact emission/runtime/generation' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'source_manifest_path:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'native_inventory_path:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'tensor_map_path:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'output_head_map_path:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'expected_artifact_path:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'artifact_identity_status:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'prepare_blocker_count:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'runtime_execution:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'generation:' "$ROOT/prepare-dynamic-qwen-normal.out"
! grep 'reason:' "$ROOT/prepare-dynamic-qwen-normal.out"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire --repo google/Gemma-4-31B-it --family gemma --name gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --auth auto --progress off --audit > "$ROOT/download-dynamic-gemma.out"
grep 'target_id: gemma-4-31b-it' "$ROOT/download-dynamic-gemma.out"
grep 'repo_id: google/Gemma-4-31B-it' "$ROOT/download-dynamic-gemma.out"
test -f "$DYNAMIC_ROOT/registry/gemma/gemma-4-31b-it.download.json"
test -f "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.source-manifest.json"
"$YVEX_BIN" source status gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/download-dynamic-gemma-status.out"
grep 'target_id: gemma-4-31b-it' "$ROOT/download-dynamic-gemma-status.out"
grep 'family: gemma' "$ROOT/download-dynamic-gemma-status.out"
grep 'repo_id: google/Gemma-4-31B-it' "$ROOT/download-dynamic-gemma-status.out"
grep 'safetensors_size_status: ok' "$ROOT/download-dynamic-gemma-status.out"
rm -f "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/"*.safetensors
write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" gemma-language-head BF16
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map_status: naming-map-profiled' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map_family: gemma' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map_target_id: gemma-4-31b-it' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.embed_tokens.weight -> model.embedding.token.weight' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.self_attn.q_proj.weight -> model.layers.0.attention.q_proj.weight' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'tensor_map.entry.[0-9][0-9]*.mapping: model.language_model.layers.0.layer_scalar -> model.layers.0.layer_scalar' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-dynamic-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-dynamic-gemma-audit.out"
test -f "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.tensor-map.json"
grep '"row": "MODELS.SOURCE.MAP.HANDOFF.0"' "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.tensor-map.json"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/tensor-map-dynamic-gemma-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-dynamic-gemma-table.out"
grep -F 'FAMILY  TARGET                STATUS                      TOTAL   EMBED    ATTN     MLP    NORM    HEAD     MOE   UNKNOWN   LAYERS  NEXT' "$ROOT/tensor-map-dynamic-gemma-table.out"
grep -F 'gemma   gemma-4-31b-it        naming-map-profiled' "$ROOT/tensor-map-dynamic-gemma-table.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role output-head --audit > "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'output_head_map_family: gemma' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'output_head_map_target_id: gemma-4-31b-it' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'output_head_native_name: model.language_model.lm_head.weight' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'tie_policy_status: separate-output-head-candidate' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-dynamic-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-dynamic-gemma-audit.out"
test -f "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.output-head-map.json"
grep '"row": "MODELS.SOURCE.MAP.HANDOFF.0"' "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.output-head-map.json"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role output-head --output table > "$ROOT/output-head-dynamic-gemma-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-dynamic-gemma-table.out"
grep -F 'FAMILY  TARGET                STATUS                           HEAD  FINAL_NORM  EMBED  TIE_POLICY                          SHAPE_RELATION            NEXT' "$ROOT/output-head-dynamic-gemma-table.out"
grep -F 'gemma   gemma-4-31b-it        output-head-profiled             yes' "$ROOT/output-head-dynamic-gemma-table.out"
write_fake_tokenizer_sidecars "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" gemma
"$YVEX_BIN" inspect target tokenizer-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" > "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'tokenizer-map: gemma-4-31b-it' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'family: gemma' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'status: present-report-only' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'tokenizer: present' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'vocab: embedded-or-tokenizer-json' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'merges: not-required-or-absent' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'chat_template: present' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'specials: present' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'runtime: unsupported' "$ROOT/tokenizer-map-dynamic-gemma.out"
grep 'next: V010.QUANT.1' "$ROOT/tokenizer-map-dynamic-gemma.out"
"$YVEX_BIN" inspect target tokenizer-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/tokenizer-map-dynamic-gemma-table.out"
grep 'TOKENIZER METADATA MAP' "$ROOT/tokenizer-map-dynamic-gemma-table.out"
matches "$ROOT/tokenizer-map-dynamic-gemma-table.out" '^gemma-4-31b-it[[:space:]]{2,}gemma[[:space:]]{2,}present-report-only[[:space:]]{2,}yes[[:space:]]{2,}embedded-or-tokenizer-json[[:space:]]{2,}not-required-or-absent[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}V010\.QUANT\.1$'
"$YVEX_BIN" inspect target tokenizer-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'tokenizer_map_target_id: gemma-4-31b-it' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'vocab_status: embedded-or-tokenizer-json' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'merges_status: not-required-or-absent' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'special_tokens_status: present' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
grep 'next_required_rows: V010.QUANT.1' "$ROOT/tokenizer-map-dynamic-gemma-audit.out"
"$YVEX_BIN" inspect target tokenizer-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --output json > "$ROOT/tokenizer-map-dynamic-gemma-json.out"
grep '"target_id":"gemma-4-31b-it"' "$ROOT/tokenizer-map-dynamic-gemma-json.out"
grep '"vocab_status":"embedded-or-tokenizer-json"' "$ROOT/tokenizer-map-dynamic-gemma-json.out"
grep '"next":"V010.QUANT.1"' "$ROOT/tokenizer-map-dynamic-gemma-json.out"
test -f "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.tokenizer-map.json"
grep '"tokenizer_map_status": "present-report-only"' "$DYNAMIC_ROOT/evidence/build/gemma/gemma-4-31b-it.tokenizer-map.json"
"$YVEX_BIN" compile source manifest report --family gemma --release v0.1.0 --source "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/source-dynamic-gemma-audit.out"
grep 'target_id: gemma-4-31b-it' "$ROOT/source-dynamic-gemma-audit.out"
grep 'model: Gemma-4-31B-it' "$ROOT/source-dynamic-gemma-audit.out"
! grep 'target_id: gemma-4-12b-it' "$ROOT/source-dynamic-gemma-audit.out"
grep 'source_manifest_path: .*gemma-4-31b-it.source-manifest.json' "$ROOT/source-dynamic-gemma-audit.out"
grep 'native_inventory_path: .*gemma-4-31b-it.native-inventory.json' "$ROOT/source-dynamic-gemma-audit.out"
grep 'tensor_map_status: available-report-only' "$ROOT/source-dynamic-gemma-audit.out"
grep 'tensor_role_map_status: available-report-only' "$ROOT/source-dynamic-gemma-audit.out"
grep 'output_head_map_status: available-report-only' "$ROOT/source-dynamic-gemma-audit.out"
grep 'tokenizer_map_status: available-report-only' "$ROOT/source-dynamic-gemma-audit.out"
grep 'next_required_rows: V010.QUANT.1' "$ROOT/source-dynamic-gemma-audit.out"
! grep 'missing-gemma-tensor-role-map' "$ROOT/source-dynamic-gemma-audit.out"
! grep 'missing-gemma-tensor-map' "$ROOT/source-dynamic-gemma-audit.out"
"$YVEX_BIN" compile gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --dry-run --audit > "$ROOT/prepare-dynamic-gemma.out" 2>&1 && exit 1 || true
grep 'target_id: gemma-4-31b-it' "$ROOT/prepare-dynamic-gemma.out"
grep 'family: gemma' "$ROOT/prepare-dynamic-gemma.out"
grep 'source_status: present' "$ROOT/prepare-dynamic-gemma.out"
grep 'model_class_status: present' "$ROOT/prepare-dynamic-gemma.out"
grep 'tensor_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma.out"
grep 'output_head_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma.out"
grep 'artifact_status: missing' "$ROOT/prepare-dynamic-gemma.out"
grep 'expected_artifact_path: .*gemma-4-31b-it.gguf' "$ROOT/prepare-dynamic-gemma.out"
grep 'artifact_plan_status: planned-full-gguf' "$ROOT/prepare-dynamic-gemma.out"
grep 'artifact_emission_status: not-performed' "$ROOT/prepare-dynamic-gemma.out"
grep 'artifact_identity_status: missing' "$ROOT/prepare-dynamic-gemma.out"
grep 'prepare_blocker_count:' "$ROOT/prepare-dynamic-gemma.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/prepare-dynamic-gemma.out"
grep 'reason: family quantization plan unimplemented' "$ROOT/prepare-dynamic-gemma.out"
grep 'next: not-scheduled' "$ROOT/prepare-dynamic-gemma.out"
grep 'status: model-prepare-unsupported' "$ROOT/prepare-dynamic-gemma.out"
! grep 'status: model-prepare-unknown-target' "$ROOT/prepare-dynamic-gemma.out"
! grep 'reason: missing compile map / model class / artifact path' "$ROOT/prepare-dynamic-gemma.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role-support > "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'qtype-role-support: gemma-4-31b-it' "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'family: gemma' "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'status: blocked' "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'source_dtype: BF16' "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/qtype-role-support-dynamic-gemma.out"
grep 'next: not-scheduled' "$ROOT/qtype-role-support-dynamic-gemma.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role-support --output table > "$ROOT/qtype-role-support-dynamic-gemma-table.out"
grep 'attention_q_norm[[:space:]][[:space:]]*BF16' "$ROOT/qtype-role-support-dynamic-gemma-table.out"
grep 'output_head_tied_embedding[[:space:]][[:space:]]*BF16' "$ROOT/qtype-role-support-dynamic-gemma-table.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role-support --audit > "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'role\.[0-9][0-9]*\.role_name: pre_feedforward_layernorm' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'role\.[0-9][0-9]*\.role_name: layer_scalar' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'role\.[0-9][0-9]*\.role_name: tokenizer_metadata' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'role\.[0-9][0-9]*\.compute_support_status: cpu-cuda-available' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/qtype-role-support-dynamic-gemma-audit.out"
write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" qwen-coverage BF16
write_fake_tokenizer_sidecars "$DYNAMIC_ROOT/source/hf/Qwen/Qwen3.6-35B-A3B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" qwen
"$YVEX_BIN" inspect target tensor-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-qwen-restored-audit.out"
"$YVEX_BIN" inspect target tokenizer-map qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" > "$ROOT/tokenizer-map-dynamic-qwen-restored.out"
"$YVEX_BIN" inspect target quant-policy --gate v0.1.0 --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/qtype-role-support-gate-table.out"
grep 'FAMILY[[:space:]][[:space:]]*TARGET[[:space:]][[:space:]]*STATUS[[:space:]][[:space:]]*ROLES[[:space:]][[:space:]]*BLOCKED[[:space:]][[:space:]]*TOP_BLOCKER[[:space:]][[:space:]]*NEXT' "$ROOT/qtype-role-support-gate-table.out"
grep 'deepseek[[:space:]][[:space:]]*deepseek4-v4-flash-dspark[[:space:]][[:space:]]*blocked[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*artifact-materialization-unimplemented[[:space:]][[:space:]]*V010.ARTIFACT.MATERIALIZE.0' "$ROOT/qtype-role-support-gate-table.out"
grep 'qwen[[:space:]][[:space:]]*qwen3-6-35b-a3b[[:space:]][[:space:]]*blocked[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*family-quantization-plan-unimplemented[[:space:]][[:space:]]*not-scheduled' "$ROOT/qtype-role-support-gate-table.out"
grep 'gemma[[:space:]][[:space:]]*gemma-4-31b-it[[:space:]][[:space:]]*blocked[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*[1-9][0-9]*[[:space:]][[:space:]]*family-quantization-plan-unimplemented[[:space:]][[:space:]]*not-scheduled' "$ROOT/qtype-role-support-gate-table.out"
"$YVEX_BIN" inspect target quant-policy --gate v0.1.0 --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/qtype-role-support-gate-audit.out"
grep 'report: qtype-role-support-gate' "$ROOT/qtype-role-support-gate-audit.out"
grep 'family.0.top_blocker: artifact-materialization-unimplemented' "$ROOT/qtype-role-support-gate-audit.out"
grep 'family.1.top_blocker: family-quantization-plan-unimplemented' "$ROOT/qtype-role-support-gate-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/qtype-role-support-gate-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/qtype-role-support-gate-audit.out"

write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" gemma-language-tied
printf '{"tie_word_embeddings":true,"vocab_size":2,"bos_token_id":1,"eos_token_id":1,"pad_token_id":0,"unk_token_id":0}\n' > "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/config.json"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-gemma-tied-audit.out"
grep 'tensor_map_status: naming-map-profiled' "$ROOT/tensor-map-dynamic-gemma-tied-audit.out"
grep 'tensor_map_required_role_coverage_status: required-groups-present' "$ROOT/tensor-map-dynamic-gemma-tied-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role output-head --audit > "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'output_head_map_status: tied-output-head-report-only' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'output_head_native_name: model.language_model.embed_tokens.weight' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'output_head_canonical_role: model.output_head.tied_embedding' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'output_head_mapping_status: tied-to-token-embedding-candidate' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'tie_policy_status: tied-output-head-candidate' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
grep 'config_tie_word_embeddings_status: true' "$ROOT/output-head-dynamic-gemma-tied-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role output-head --output table > "$ROOT/output-head-dynamic-gemma-tied-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-dynamic-gemma-tied-table.out"
grep -F 'FAMILY  TARGET                STATUS                           HEAD  FINAL_NORM  EMBED  TIE_POLICY                          SHAPE_RELATION            NEXT' "$ROOT/output-head-dynamic-gemma-tied-table.out"
grep -F 'gemma   gemma-4-31b-it        tied-output-head-report-only     yes' "$ROOT/output-head-dynamic-gemma-tied-table.out"
"$YVEX_BIN" inspect target missing-roles gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'tensor_map_status: present-report-only' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'output_head_map_status: present-report-only' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'role_group.output_head.status: present' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'role_group.tied_head_policy.status: tied-output-head-candidate' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'top_blocker: quant-policy-or-artifact-emitter' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
grep 'next: V010.QUANT.1' "$ROOT/missing-roles-dynamic-gemma-tied-audit.out"
"$YVEX_BIN" compile gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --dry-run --audit > "$ROOT/prepare-dynamic-gemma-tied.out" 2>&1 && exit 1 || true
grep 'tensor_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma-tied.out"
grep 'output_head_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma-tied.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma-tied.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/prepare-dynamic-gemma-tied.out"

write_fake_transformer_safetensors "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/model.safetensors" gemma-language-no-head
printf '{"tie_word_embeddings":false}\n' > "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/config.json"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/tensor-map-dynamic-gemma-incomplete-audit.out"
grep 'tensor_map_status: naming-map-candidate' "$ROOT/tensor-map-dynamic-gemma-incomplete-audit.out"
grep 'tensor_map_target_id: gemma-4-31b-it' "$ROOT/tensor-map-dynamic-gemma-incomplete-audit.out"
grep 'unmapped_unknown_count: [1-9]' "$ROOT/tensor-map-dynamic-gemma-incomplete-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --role output-head --audit > "$ROOT/output-head-dynamic-gemma-missing-audit.out"
grep 'output_head_map_status: output-head-missing' "$ROOT/output-head-dynamic-gemma-missing-audit.out"
grep 'output_head_map_target_id: gemma-4-31b-it' "$ROOT/output-head-dynamic-gemma-missing-audit.out"
grep 'output_head_missing_status: missing' "$ROOT/output-head-dynamic-gemma-missing-audit.out"
grep 'tie_policy_status: not-proven' "$ROOT/output-head-dynamic-gemma-missing-audit.out"
grep 'config_tie_word_embeddings_status: false' "$ROOT/output-head-dynamic-gemma-missing-audit.out"
"$YVEX_BIN" inspect target missing-roles gemma-4-31b-it --models-root "$DYNAMIC_ROOT" > "$ROOT/missing-roles-dynamic-gemma.out"
grep 'missing-roles: gemma-4-31b-it' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'family: gemma' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'status: blocked' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'top_blocker: missing-output-head-map' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'next: V010.MAP.8' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'output-head.*missing' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'tied-head-policy.*not-proven' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'tokenizer.*present-report-only' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'unknown-tensors.*unclassified-header-name' "$ROOT/missing-roles-dynamic-gemma.out"
grep 'artifact.*missing' "$ROOT/missing-roles-dynamic-gemma.out"
"$YVEX_BIN" inspect target missing-roles gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --output table > "$ROOT/missing-roles-dynamic-gemma-table.out"
grep 'gemma-4-31b-it.*gemma.*blocked.*missing-output-head-map' "$ROOT/missing-roles-dynamic-gemma-table.out"
grep 'gemma-4-31b-it.*present-report-only.*missing.*V010.MAP.8' "$ROOT/missing-roles-dynamic-gemma-table.out"
grep 'gemma-4-31b-it.*[[:space:]][1-9][0-9]*[[:space:]]*present-report-only[[:space:]]*missing' "$ROOT/missing-roles-dynamic-gemma-table.out"
"$YVEX_BIN" inspect target missing-roles gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'target_id: gemma-4-31b-it' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'family: gemma' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'source_status: present' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'tensor_map_status: present-report-only' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'output_head_map_status: missing-in-report' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'artifact_status: missing' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'expected_artifact_path: .*gemma-4-31b-it.gguf' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'top_blocker: missing-output-head-map' "$ROOT/missing-roles-dynamic-gemma-audit.out"
grep 'next: V010.MAP.8' "$ROOT/missing-roles-dynamic-gemma-audit.out"
"$YVEX_BIN" compile source manifest report --family gemma --release v0.1.0 --source "$DYNAMIC_ROOT/source/hf/google/Gemma-4-31B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08" --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/source-dynamic-gemma-incomplete-map.out"
grep 'tensor_map_status: available-report-only' "$ROOT/source-dynamic-gemma-incomplete-map.out"
grep 'tensor_role_map_status: available-report-only' "$ROOT/source-dynamic-gemma-incomplete-map.out"
grep 'output_head_map_status: missing-in-report' "$ROOT/source-dynamic-gemma-incomplete-map.out"
! grep 'missing-gemma-tensor-role-map' "$ROOT/source-dynamic-gemma-incomplete-map.out"
! grep 'missing-gemma-tensor-map' "$ROOT/source-dynamic-gemma-incomplete-map.out"
"$YVEX_BIN" compile gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --dry-run --audit > "$ROOT/prepare-dynamic-gemma-incomplete-map.out" 2>&1 && exit 1 || true
grep 'tensor_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
grep 'output_head_map_status: missing-in-report' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
grep 'top_blocker: missing-output-head-map' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
grep 'reason: output head mapping missing / artifact path missing' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
grep 'status: model-prepare-unsupported' "$ROOT/prepare-dynamic-gemma-incomplete-map.out"
"$YVEX_BIN" compile gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --dry-run > "$ROOT/prepare-dynamic-gemma-normal.out" 2>&1 && exit 1 || true
grep 'models prepare: gemma-4-31b-it \[blocked\]' "$ROOT/prepare-dynamic-gemma-normal.out"
grep 'family: gemma  source: present  artifact: missing' "$ROOT/prepare-dynamic-gemma-normal.out"
grep 'plan: full-gguf planned  emission: not-performed' "$ROOT/prepare-dynamic-gemma-normal.out"
grep 'top_blocker: missing-output-head-map' "$ROOT/prepare-dynamic-gemma-normal.out"
grep 'next: V010.MAP.8' "$ROOT/prepare-dynamic-gemma-normal.out"
grep 'boundary: prepare dry-run only; no artifact emission/runtime/generation' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'source_manifest_path:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'native_inventory_path:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'tensor_map_path:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'output_head_map_path:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'expected_artifact_path:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'artifact_identity_status:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'prepare_blocker_count:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'runtime_execution:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'generation:' "$ROOT/prepare-dynamic-gemma-normal.out"
! grep 'reason:' "$ROOT/prepare-dynamic-gemma-normal.out"

"$YVEX_BIN" inspect artifact registry --models-root "$DYNAMIC_ROOT" > "$ROOT/artifacts-list.out"
grep 'deepseek4-v4-flash-dspark-selected-embed.*deepseek.*yvex-selected-gguf.*present.*ready' "$ROOT/artifacts-list.out"
grep 'deepseek4-v4-flash-dspark-selected-embed-rmsnorm.*deepseek.*yvex-selected-gguf.*present.*ready' "$ROOT/artifacts-list.out"
grep 'qwen3-8b-selected-embed.*qwen.*yvex-selected-gguf.*present.*ready' "$ROOT/artifacts-list.out"
grep 'qwen3-6-35b-a3b.*qwen.*planned-full-gguf.*missing.*blocked' "$ROOT/artifacts-list.out"
grep 'gemma-4-31b-it.*gemma.*planned-full-gguf.*missing.*blocked' "$ROOT/artifacts-list.out"
grep 'status: artifacts-list' "$ROOT/artifacts-list.out"
! grep 'runtime_ready' "$ROOT/artifacts-list.out"
"$YVEX_BIN" inspect artifact registry --models-root "$DYNAMIC_ROOT" --family qwen --output table > "$ROOT/artifacts-list-qwen-table.out"
grep 'qwen3-8b-selected-embed.*qwen.*present' "$ROOT/artifacts-list-qwen-table.out"
grep 'qwen3-6-35b-a3b.*qwen.*missing.*blocked' "$ROOT/artifacts-list-qwen-table.out"
! grep 'gemma-4-31b-it' "$ROOT/artifacts-list-qwen-table.out"
"$YVEX_BIN" inspect artifact registry --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/artifacts-list-audit.out"
grep 'artifact\.[0-9][0-9]*\.expected_artifact_path: .*qwen3-6-35b-a3b.gguf' "$ROOT/artifacts-list-audit.out"
grep 'artifact\.[0-9][0-9]*\.expected_artifact_path: .*gemma-4-31b-it.gguf' "$ROOT/artifacts-list-audit.out"
grep 'artifact\.[0-9][0-9]*\.tensor_map_path: .*gemma-4-31b-it.tensor-map.json' "$ROOT/artifacts-list-audit.out"
grep 'source_payload_loaded: false' "$ROOT/artifacts-list-audit.out"
grep 'hash_performed: false' "$ROOT/artifacts-list-audit.out"
"$YVEX_BIN" inspect artifact registry --models-root "$DYNAMIC_ROOT" --output json > "$ROOT/artifacts-list-json.out"
grep '"status":"artifacts-list"' "$ROOT/artifacts-list-json.out"
grep '"target_id":"qwen3-6-35b-a3b"' "$ROOT/artifacts-list-json.out"
"$YVEX_BIN" inspect artifact status qwen3-6-35b-a3b --models-root "$DYNAMIC_ROOT" > "$ROOT/artifacts-status-qwen.out"
grep 'artifact: qwen3-6-35b-a3b' "$ROOT/artifacts-status-qwen.out"
grep 'family: qwen' "$ROOT/artifacts-status-qwen.out"
grep 'class: planned-full-gguf' "$ROOT/artifacts-status-qwen.out"
grep 'source: present' "$ROOT/artifacts-status-qwen.out"
grep 'artifact_status: missing' "$ROOT/artifacts-status-qwen.out"
grep 'expected: .*qwen3-6-35b-a3b.gguf' "$ROOT/artifacts-status-qwen.out"
grep 'prepare: blocked' "$ROOT/artifacts-status-qwen.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/artifacts-status-qwen.out"
grep 'next: not-scheduled' "$ROOT/artifacts-status-qwen.out"
grep 'boundary: artifact discovery only; no runtime/generation' "$ROOT/artifacts-status-qwen.out"
"$YVEX_BIN" inspect artifact status gemma-4-31b-it --models-root "$DYNAMIC_ROOT" --audit > "$ROOT/artifacts-status-gemma-audit.out"
grep 'artifact: gemma-4-31b-it' "$ROOT/artifacts-status-gemma-audit.out"
grep 'family: gemma' "$ROOT/artifacts-status-gemma-audit.out"
grep 'class: planned-full-gguf' "$ROOT/artifacts-status-gemma-audit.out"
grep 'artifact_status: missing' "$ROOT/artifacts-status-gemma-audit.out"
grep 'expected: .*gemma-4-31b-it.gguf' "$ROOT/artifacts-status-gemma-audit.out"
grep 'prepare: blocked' "$ROOT/artifacts-status-gemma-audit.out"
grep 'top_blocker: missing-output-head-map' "$ROOT/artifacts-status-gemma-audit.out"
grep 'next: V010.MAP.8' "$ROOT/artifacts-status-gemma-audit.out"
grep 'tensor_map_status: present-report-only' "$ROOT/artifacts-status-gemma-audit.out"
grep 'output_head_map_status: missing-in-report' "$ROOT/artifacts-status-gemma-audit.out"
grep 'source_payload_loaded: false' "$ROOT/artifacts-status-gemma-audit.out"
grep 'hash_performed: false' "$ROOT/artifacts-status-gemma-audit.out"
grep 'status: artifacts-status' "$ROOT/artifacts-status-gemma-audit.out"

QWEN32_STATUS_ROOT="$ROOT/download-qwen32-status"
"$YVEX_BIN" source status qwen3-32b --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$QWEN32_STATUS_ROOT" --audit > "$ROOT/download-qwen32-status.out"
grep 'target_id: qwen3-32b' "$ROOT/download-qwen32-status.out"
grep 'family: qwen' "$ROOT/download-qwen32-status.out"
grep 'repo_id: Qwen/Qwen3-32B' "$ROOT/download-qwen32-status.out"
grep 'source/hf/Qwen/Qwen3-32B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08' "$ROOT/download-qwen32-status.out"
grep 'status: model-download-status' "$ROOT/download-qwen32-status.out"

QWEN32_CLEANUP_ROOT="$ROOT/download-qwen32-cleanup"
QWEN32_CLEANUP_SRC="$QWEN32_CLEANUP_ROOT/source/hf/Qwen/Qwen3-32B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08"
mkdir -p "$QWEN32_CLEANUP_SRC/.cache/huggingface/download"
printf 'partial\n' > "$QWEN32_CLEANUP_SRC/.cache/huggingface/download/model.safetensors.incomplete"
for path in \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.download.receipt" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.download.active.json" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.download.last.json" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.download-report.json" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.source-manifest.json" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.native-inventory.json" \
  "$QWEN32_CLEANUP_ROOT/registry/qwen/qwen3-32b.download.json" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/acquisition/qwen3-32b.download.stdout.log" \
  "$QWEN32_CLEANUP_ROOT/evidence/build/acquisition/qwen3-32b.download.stderr.log"
do
  mkdir -p "$(dirname "$path")"
  printf 'sidecar\n' > "$path"
done
"$YVEX_BIN" source cleanup qwen3-32b --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$QWEN32_CLEANUP_ROOT" --failed-partials --dry-run --audit > "$ROOT/download-qwen32-cleanup-dry-run.out"
grep 'cleanup_failed_partials: true' "$ROOT/download-qwen32-cleanup-dry-run.out"
grep 'cleanup_sidecars: true' "$ROOT/download-qwen32-cleanup-dry-run.out"
grep 'cleanup_logs: true' "$ROOT/download-qwen32-cleanup-dry-run.out"
grep 'status: model-download-cleanup-dry-run' "$ROOT/download-qwen32-cleanup-dry-run.out"
test -d "$QWEN32_CLEANUP_SRC"
test -f "$QWEN32_CLEANUP_ROOT/evidence/build/acquisition/qwen3-32b.download.stderr.log"
"$YVEX_BIN" source cleanup qwen3-32b --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$QWEN32_CLEANUP_ROOT" --failed-partials --yes --audit > "$ROOT/download-qwen32-cleanup.out"
grep 'cleanup_failed_partials: true' "$ROOT/download-qwen32-cleanup.out"
grep 'deleted_source_entries: ' "$ROOT/download-qwen32-cleanup.out"
grep 'deleted_sidecars: 7' "$ROOT/download-qwen32-cleanup.out"
grep 'deleted_logs: 2' "$ROOT/download-qwen32-cleanup.out"
grep 'status: model-download-cleanup' "$ROOT/download-qwen32-cleanup.out"
test ! -e "$QWEN32_CLEANUP_SRC"
test ! -e "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.download-report.json"
test ! -e "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.source-manifest.json"
test ! -e "$QWEN32_CLEANUP_ROOT/evidence/build/qwen/qwen3-32b.native-inventory.json"
test ! -e "$QWEN32_CLEANUP_ROOT/registry/qwen/qwen3-32b.download.json"
test ! -e "$QWEN32_CLEANUP_ROOT/evidence/build/acquisition/qwen3-32b.download.stdout.log"
test ! -e "$QWEN32_CLEANUP_ROOT/evidence/build/acquisition/qwen3-32b.download.stderr.log"

YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$DOWNLOAD_ROOT/required" --auth required --audit > "$ROOT/download-auth-required.out" 2> "$ROOT/download-auth-required.err" && exit 1 || true
grep 'stage: account-provider blocked' "$ROOT/download-auth-required.out"
grep 'top_blocker: provider-login-required' "$ROOT/download-auth-required.out"

YVEX_HF_CLI=/missing/hf "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$DOWNLOAD_ROOT/missing" --auth auto --audit > "$ROOT/download-missing-hf.out" 2> "$ROOT/download-missing-hf.err" && exit 1 || true
grep 'status: model-download-blocked' "$ROOT/download-missing-hf.out"
grep 'top_blocker: missing-huggingface-cli' "$ROOT/download-missing-hf.out"
grep 'stage: account-provider blocked' "$ROOT/download-missing-hf.out"

YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_FAIL=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --revision b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08 --models-root "$DOWNLOAD_ROOT/fail" --auth auto --audit > "$ROOT/download-fail.out" 2> "$ROOT/download-fail.err" && exit 1 || true
grep 'status: model-download-fail' "$ROOT/download-fail.out"
test -f "$DOWNLOAD_ROOT/fail/evidence/build/acquisition/gemma-4-12b-it.download.stderr.log"

YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire --repo test-org/test-model --family gemma --name test-model --models-root "$DOWNLOAD_ROOT/direct" --auth auto --audit > "$ROOT/download-direct.out"
grep 'repo_id: test-org/test-model' "$ROOT/download-direct.out"
grep 'source/hf/test-org/test-model/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08' "$ROOT/download-direct.out"
grep 'status: model-download-pass' "$ROOT/download-direct.out"

YVEX_FAKE_GH_AUTH=1 YVEX_GH_CLI="$FAKE_GH" "$YVEX_BIN" source acquire --provider github --repo test-org/test-model --release v1 --asset '*.gguf' --models-root "$DOWNLOAD_ROOT/github" --auth auto --audit > "$ROOT/download-github.out"
grep 'provider: github' "$ROOT/download-github.out"
grep 'stage: account-provider pass' "$ROOT/download-github.out"
grep 'stage: download pass' "$ROOT/download-github.out"
grep 'github/test-org/test-model/v1' "$ROOT/download-github.out"
grep 'gguf_created: false' "$ROOT/download-github.out"
grep 'generation: unsupported' "$ROOT/download-github.out"
test -f "$DOWNLOAD_ROOT/github/source/github/test-org/test-model/v1/fake-model.gguf"

"$YVEX_BIN" source acquire --models-root "$DOWNLOAD_ROOT/parser" > "$ROOT/download-missing-target.out" 2> "$ROOT/download-missing-target.err" && exit 1 || true
grep 'requires TARGET or --repo' "$ROOT/download-missing-target.err"
"$YVEX_BIN" source acquire --repo test-org/test-model --models-root "$DOWNLOAD_ROOT/parser" > "$ROOT/download-repo-no-family.out" 2> "$ROOT/download-repo-no-family.err" && exit 1 || true
grep 'requires a safe lower-case --family key' "$ROOT/download-repo-no-family.err"
"$YVEX_BIN" source acquire --repo test-org/test-model --family 'llama/unsafe' --models-root "$DOWNLOAD_ROOT/parser" > "$ROOT/download-bad-family.out" 2> "$ROOT/download-bad-family.err" && exit 1 || true
grep 'requires a safe lower-case --family key' "$ROOT/download-bad-family.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --models-root "" > "$ROOT/download-empty-root.out" 2> "$ROOT/download-empty-root.err" && exit 1 || true
grep 'models download --models-root value is empty or invalid' "$ROOT/download-empty-root.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --max-workers 0 > "$ROOT/download-bad-workers.out" 2> "$ROOT/download-bad-workers.err" && exit 1 || true
grep 'requires a positive integer' "$ROOT/download-bad-workers.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --auth maybe > "$ROOT/download-bad-auth.out" 2> "$ROOT/download-bad-auth.err" && exit 1 || true
grep 'auth requires auto|required|never' "$ROOT/download-bad-auth.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --progress nope > "$ROOT/download-bad-progress.out" 2> "$ROOT/download-bad-progress.err" && exit 1 || true
grep 'requires auto|live|plain|log|off' "$ROOT/download-bad-progress.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --tick-seconds 0 > "$ROOT/download-bad-tick.out" 2> "$ROOT/download-bad-tick.err" && exit 1 || true
grep 'tick-seconds requires a positive integer' "$ROOT/download-bad-tick.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --source s3 > "$ROOT/download-bad-source.out" 2> "$ROOT/download-bad-source.err" && exit 1 || true
grep 'supports hf only' "$ROOT/download-bad-source.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --auth nope > "$ROOT/download-bad-auth.out" 2> "$ROOT/download-bad-auth.err" && exit 1 || true
grep 'requires auto|required|never' "$ROOT/download-bad-auth.err"
"$YVEX_BIN" source acquire --provider github --repo test-org/test-model > "$ROOT/download-github-no-asset.out" 2> "$ROOT/download-github-no-asset.err" && exit 1 || true
grep 'requires --asset GLOB' "$ROOT/download-github-no-asset.err"
"$YVEX_BIN" source acquire gemma-4-12b-it --surprise > "$ROOT/download-unknown-flag.out" 2> "$ROOT/download-unknown-flag.err" && exit 1 || true
grep 'unknown flag: --surprise' "$ROOT/download-unknown-flag.err"
"$YVEX_BIN" source acquire gemma-4-12b-it extra > "$ROOT/download-extra-positional.out" 2> "$ROOT/download-extra-positional.err" && exit 1 || true
grep 'extra positional argument' "$ROOT/download-extra-positional.err"

HF_TOKEN=secret-value YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-e2b --models-root "$DOWNLOAD_ROOT/token" --auth auto --audit > "$ROOT/download-token.out"
grep 'auth_state: env-token-present' "$ROOT/download-token.out"
grep 'token_value_redacted: true' "$ROOT/download-token.out"
! grep -R 'secret-value' "$ROOT/download-token.out" "$DOWNLOAD_ROOT/token" "$ROOT"
! git ls-files '*.safetensors' '*.bin' '*.dat' | grep .

PREP="$ROOT/prepare"
PREP_SOURCE="$PREP/source/hf/deepseek-ai/DeepSeek-V4-Flash-DSpark/62af8fffb2f7030cac4de2f0169f5b8d1101b646"
PREP_REG="$PREP/registry/models.local.json"
PREP_GGUF="$PREP/evidence/fixtures/deepseek/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
mkdir -p "$PREP_SOURCE"

python3 - "$PREP_SOURCE/model-00001.safetensors" <<'PY'
import json
import struct
import sys

path = sys.argv[1]
header = {
    "__metadata__": {"format": "pt"},
    "embed.weight": {"dtype": "F16", "shape": [8, 4], "data_offsets": [0, 64]},
}
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(path, "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(bytes(range(64)))
PY

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --dry-run --models-root "$PREP" --registry "$PREP_REG" > "$ROOT/prepare-dry-run.out"
grep 'status: model-prepare-dry-run' "$ROOT/prepare-dry-run.out"
grep 'stage: convert-emit planned' "$ROOT/prepare-dry-run.out"
grep 'generation: unsupported' "$ROOT/prepare-dry-run.out"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --models-root "$ROOT/missing-prepare" --registry "$ROOT/missing-prepare/registry/models.local.json" > "$ROOT/prepare-missing.out" 2> "$ROOT/prepare-missing.err" && exit 1 || true
grep 'stage: source-path fail' "$ROOT/prepare-missing.out"
grep 'status: model-prepare-fail' "$ROOT/prepare-missing.out"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed-rmsnorm --dry-run > "$ROOT/prepare-segment-unsupported.out" 2> "$ROOT/prepare-segment-unsupported.err" && exit 1 || true
grep 'status: model-prepare-unsupported' "$ROOT/prepare-segment-unsupported.out"
grep 'segment prepare is planned' "$ROOT/prepare-segment-unsupported.out"

"$YVEX_BIN" compile glm-5.2-official-safetensors --dry-run > "$ROOT/prepare-glm-unsupported.out" 2> "$ROOT/prepare-glm-unsupported.err" && exit 1 || true
grep 'status: model-prepare-unsupported' "$ROOT/prepare-glm-unsupported.out"
grep 'YVEX-produced GGUF emission for this target is planned' "$ROOT/prepare-glm-unsupported.out"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --models-root "$PREP" --registry "$PREP_REG" --overwrite --no-register > "$ROOT/prepare-no-register.out"
grep 'stage: source-manifest pass' "$ROOT/prepare-no-register.out"
grep 'stage: convert-emit pass' "$ROOT/prepare-no-register.out"
grep 'stage: registry-add skipped' "$ROOT/prepare-no-register.out"
grep 'status: model-prepare' "$ROOT/prepare-no-register.out"
test -f "$PREP_GGUF"
test ! -f "$PREP_REG"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed \
  --models-root "$PREP" --registry "$PREP_REG" --overwrite --no-use \
  > "$ROOT/prepare-no-use.out" 2> "$ROOT/prepare-no-use.err" && exit 1 || true
grep 'unknown flag: --no-use' "$ROOT/prepare-no-use.err"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --models-root "$PREP" --registry "$PREP_REG" --overwrite > "$ROOT/prepare-register-use.out"
grep 'stage: registry-remove-existing not-found' "$ROOT/prepare-register-use.out"
grep 'stage: registry-add pass' "$ROOT/prepare-register-use.out"
grep 'stage: registry-verify pass' "$ROOT/prepare-register-use.out"
grep 'status: model-prepare' "$ROOT/prepare-register-use.out"

"$YVEX_BIN" profile verify deepseek4-v4-flash-dspark-selected-embed --registry "$PREP_REG" > "$ROOT/prepare-verify.out"
grep 'status: models-identity-pass' "$ROOT/prepare-verify.out"
grep 'verify: pass alias=deepseek4-v4-flash-dspark-selected-embed' "$ROOT/prepare-verify.out"

"$YVEX_BIN" profile verify deepseek4-v4-flash-dspark-selected-embed --registry "$PREP_REG" --audit > "$ROOT/prepare-verify-audit.out"
grep 'current_sha256:' "$ROOT/prepare-verify-audit.out"
grep 'digest_status: pass' "$ROOT/prepare-verify-audit.out"
grep 'status: models-identity-pass' "$ROOT/prepare-verify-audit.out"

"$YVEX_BIN" profile verify deepseek4-v4-flash-dspark-selected-embed --registry "$PREP_REG" --output nope > "$ROOT/verify-bad-output.out" 2> "$ROOT/verify-bad-output.err" && exit 1 || true
grep 'unsupported output mode: nope' "$ROOT/verify-bad-output.err"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --models-root "$PREP" --registry "$PREP_REG" > "$ROOT/prepare-overwrite-refused.out" 2> "$ROOT/prepare-overwrite-refused.err" && exit 1 || true
grep 'stage: convert-emit refused' "$ROOT/prepare-overwrite-refused.out"
grep 'status: model-prepare-refused' "$ROOT/prepare-overwrite-refused.out"

"$YVEX_BIN" compile deepseek4-v4-flash-dspark-selected-embed --out "$PREP_GGUF" --out-dir "$PREP/evidence/fixtures/deepseek" > "$ROOT/prepare-invalid.out" 2> "$ROOT/prepare-invalid.err" && exit 1 || true
grep 'conflicts with --out-dir' "$ROOT/prepare-invalid.err"

CHECK="build/tests/model-check"
CHECK_REG="$CHECK/registry/models.local.json"
CHECK_GGUF="$CHECK/models/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
CHECK_ROOT="$CHECK/root"
CHECK_ROOT_GGUF="$CHECK_ROOT/evidence/fixtures/deepseek/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf"
yvex_test_cleanup "$CHECK"
mkdir -p "$CHECK/models" "$CHECK/registry" "$CHECK_ROOT/evidence/fixtures/deepseek"

"$YVEX_BIN" compile artifact emit --out "$CHECK_GGUF" --model-name model-check-test --arch llama --overwrite >/dev/null
"$YVEX_BIN" compile artifact emit --out "$CHECK_ROOT_GGUF" --model-name model-check-target-root-test --arch llama --overwrite >/dev/null
"$YVEX_BIN" profile create --path "$CHECK_GGUF" --alias deepseek4-v4-flash-dspark-selected-embed --support-level selected-tensor-materialized --registry "$CHECK_REG" > "$ROOT/check-add.out"
grep 'status: models-added' "$ROOT/check-add.out"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --level quick --registry "$CHECK_REG" > "$ROOT/check-quick-normal.out"
grep 'model-check: pass target=deepseek4-v4-flash-dspark-selected-embed level=quick' "$ROOT/check-quick-normal.out"
grep 'boundary: selected-slice check only, generation unsupported' "$ROOT/check-quick-normal.out"
test "$(wc -l < "$ROOT/check-quick-normal.out")" -le 8

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --level quick --registry "$CHECK_REG" --audit > "$ROOT/check-quick.out"
grep 'status: model-check' "$ROOT/check-quick.out"
grep 'target_id: deepseek4-v4-flash-dspark-selected-embed' "$ROOT/check-quick.out"
grep 'backend: cpu' "$ROOT/check-quick.out"
grep 'level: quick' "$ROOT/check-quick.out"
grep 'stage: inspect pass' "$ROOT/check-quick.out"
grep 'stage: tensors pass' "$ROOT/check-quick.out"
grep 'stage: metadata pass' "$ROOT/check-quick.out"
grep 'stage: registry-identity pass' "$ROOT/check-quick.out"
grep 'stage: integrity-check pass' "$ROOT/check-quick.out"
grep 'stage: materialize skipped' "$ROOT/check-quick.out"
grep 'stage: graph-partial skipped' "$ROOT/check-quick.out"
grep 'execution_ready: false' "$ROOT/check-quick.out"
grep 'generation: unsupported' "$ROOT/check-quick.out"
grep 'status: model-check-pass' "$ROOT/check-quick.out"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --level quick --models-root "$CHECK_ROOT" --audit > "$ROOT/check-models-root.out"
grep 'model_input_kind: target' "$ROOT/check-models-root.out"
grep 'build/tests/model-check/root/evidence/fixtures/deepseek/deepseek4-v4-flash-dspark-selected-embed-F16-noimatrix-yvex-v1.gguf' "$ROOT/check-models-root.out"
grep 'stage: registry-identity unregistered' "$ROOT/check-models-root.out"
grep 'stage: integrity-check pass' "$ROOT/check-models-root.out"
grep 'status: model-check-pass' "$ROOT/check-models-root.out"

"$YVEX_BIN" artifact status glm-5.2-official-safetensors --dry-run > "$ROOT/check-invalid-dry-run.out" 2> "$ROOT/check-invalid-dry-run.err" && exit 1 || true
grep 'unknown flag: --dry-run' "$ROOT/check-invalid-dry-run.err"

"$YVEX_BIN" artifact status glm-5.2-official-safetensors --level quick --audit > "$ROOT/check-glm-unsupported.out" 2> "$ROOT/check-glm-unsupported.err" && exit 1 || true
grep 'status: model-check-unsupported' "$ROOT/check-glm-unsupported.out"
grep 'source-only target cannot be checked as a YVEX-produced runtime artifact yet' "$ROOT/check-glm-unsupported.out"
grep 'generation: unsupported' "$ROOT/check-glm-unsupported.out"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed-rmsnorm --level quick --audit > "$ROOT/check-segment-unsupported.out" 2> "$ROOT/check-segment-unsupported.err" && exit 1 || true
grep 'status: model-check-unsupported' "$ROOT/check-segment-unsupported.out"
grep 'segment check is planned' "$ROOT/check-segment-unsupported.out"
grep 'generation: unsupported' "$ROOT/check-segment-unsupported.out"

"$YVEX_BIN" artifact status > "$ROOT/check-invalid-missing-target.out" 2> "$ROOT/check-invalid-missing-target.err" && exit 1 || true
grep 'requires TARGET' "$ROOT/check-invalid-missing-target.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --backend > "$ROOT/check-invalid-backend-missing.out" 2> "$ROOT/check-invalid-backend-missing.err" && exit 1 || true
grep 'requires a value' "$ROOT/check-invalid-backend-missing.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --backend missing > "$ROOT/check-invalid-backend.out" 2> "$ROOT/check-invalid-backend.err" && exit 1 || true
grep 'unknown backend kind' "$ROOT/check-invalid-backend.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --level > "$ROOT/check-invalid-level-missing.out" 2> "$ROOT/check-invalid-level-missing.err" && exit 1 || true
grep 'requires a value' "$ROOT/check-invalid-level-missing.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --level impossible > "$ROOT/check-invalid-level.out" 2> "$ROOT/check-invalid-level.err" && exit 1 || true
grep 'unknown models check level' "$ROOT/check-invalid-level.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --registry "" > "$ROOT/check-invalid-registry.out" 2> "$ROOT/check-invalid-registry.err" && exit 1 || true
grep 'empty or invalid' "$ROOT/check-invalid-registry.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --report-dir "" > "$ROOT/check-invalid-report-dir.out" 2> "$ROOT/check-invalid-report-dir.err" && exit 1 || true
grep 'empty or invalid' "$ROOT/check-invalid-report-dir.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --unknown-flag > "$ROOT/check-invalid-unknown.out" 2> "$ROOT/check-invalid-unknown.err" && exit 1 || true
grep 'unknown flag: --unknown-flag' "$ROOT/check-invalid-unknown.err"

"$YVEX_BIN" artifact status deepseek4-v4-flash-dspark-selected-embed --output nope > "$ROOT/check-invalid-output.out" 2> "$ROOT/check-invalid-output.err" && exit 1 || true
grep 'unsupported output mode: nope' "$ROOT/check-invalid-output.err"

"$YVEX_BIN" inspect target --help > "$ROOT/model-target-help.out"
grep 'operation: evidence.target' "$ROOT/model-target-help.out"
grep 'plane: Inspect' "$ROOT/model-target-help.out"
grep 'lane: offline-engine' "$ROOT/model-target-help.out"
grep -- '--candidate' "$ROOT/model-target-help.out"
grep -- '--output' "$ROOT/model-target-help.out"
grep -- '--strict' "$ROOT/model-target-help.out"

CLASS_MISSING_ROOT="$ROOT/qwen-class-missing-root"
expect_rc 5 "$YVEX_BIN" inspect target class-profile deepseek4-v4-flash-dspark \
  --models-root "$CLASS_MISSING_ROOT" > "$ROOT/model-class-deepseek-blocked.out"
grep 'model-class: deepseek' "$ROOT/model-class-deepseek-blocked.out"
grep 'status: architecture-ir-blocked' "$ROOT/model-class-deepseek-blocked.out"
grep 'reason: missing-source-path' "$ROOT/model-class-deepseek-blocked.out"
grep 'runtime/generation unsupported' "$ROOT/model-class-deepseek-blocked.out"

expect_rc 5 "$YVEX_BIN" inspect target class-profile deepseek4-v4-flash-dspark \
  --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/model-class-deepseek-blocked-table.out"
grep 'TARGET  SOURCE  IR  REASON' "$ROOT/model-class-deepseek-blocked-table.out"
grep 'deepseek4-v4-flash-dspark  blocked  not-built  missing-source-path' "$ROOT/model-class-deepseek-blocked-table.out"

expect_rc 5 "$YVEX_BIN" inspect target class-profile deepseek4-v4-flash-dspark \
  --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/model-class-deepseek-blocked-audit.out"
grep 'architecture_ir_status: blocked' "$ROOT/model-class-deepseek-blocked-audit.out"
grep 'source_verification_status: blocked' "$ROOT/model-class-deepseek-blocked-audit.out"
grep 'runtime_execution: unsupported' "$ROOT/model-class-deepseek-blocked-audit.out"
grep 'generation: unsupported' "$ROOT/model-class-deepseek-blocked-audit.out"

expect_rc 5 "$YVEX_BIN" inspect target class-profile deepseek4-v4-flash-dspark \
  --models-root "$CLASS_MISSING_ROOT" --output json > "$ROOT/model-class-deepseek-blocked.json"
jq -e '.status == "architecture-ir-blocked" and .target_id == "deepseek4-v4-flash-dspark" and .reason == "missing-source-path" and .runtime == "unsupported" and .generation == "unsupported"' \
  "$ROOT/model-class-deepseek-blocked.json" >/dev/null

"$YVEX_BIN" inspect target class-profile qwen3-8b --models-root "$CLASS_MISSING_ROOT" > "$ROOT/model-class-qwen-missing.out"
grep 'model-class: qwen' "$ROOT/model-class-qwen-missing.out"
grep 'target: qwen3-8b' "$ROOT/model-class-qwen-missing.out"
grep 'status: source-missing' "$ROOT/model-class-qwen-missing.out"
grep 'class: qwen-source-model-class-profile' "$ROOT/model-class-qwen-missing.out"
grep 'evidence: header-metadata-only' "$ROOT/model-class-qwen-missing.out"
grep 'patterns: tensors=0 attn=0 mlp=0 norm=0 head=0 moe=0' "$ROOT/model-class-qwen-missing.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/model-class-qwen-missing.out"
grep 'next: V010.MAP.8' "$ROOT/model-class-qwen-missing.out"
grep 'no tensor role mapping/runtime/generation' "$ROOT/model-class-qwen-missing.out"

"$YVEX_BIN" inspect target class-profile qwen3-8b --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/model-class-qwen-missing-audit.out"
grep 'model_class_profile_status: source-missing' "$ROOT/model-class-qwen-missing-audit.out"
grep 'model_class_source_metadata_status: missing' "$ROOT/model-class-qwen-missing-audit.out"
grep 'model_class_tensor_count: 0' "$ROOT/model-class-qwen-missing-audit.out"
grep 'model_class_pattern_status: lexical-only' "$ROOT/model-class-qwen-missing-audit.out"
grep 'model_class_role_mapping_status: not-implemented' "$ROOT/model-class-qwen-missing-audit.out"
grep 'backend_selection: deferred' "$ROOT/model-class-qwen-missing-audit.out"
grep 'backend_pressure: metal-planned' "$ROOT/model-class-qwen-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/model-class-qwen-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/model-class-qwen-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/model-class-qwen-missing-audit.out"
grep 'release_ready: false' "$ROOT/model-class-qwen-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/model-class-qwen-missing-audit.out"

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tensor-collection-qwen-missing.out"
grep 'tensor-collection: qwen' "$ROOT/tensor-collection-qwen-missing.out"
grep 'target: qwen3-8b' "$ROOT/tensor-collection-qwen-missing.out"
grep 'status: source-missing' "$ROOT/tensor-collection-qwen-missing.out"
grep 'stage: header-collection-inventory' "$ROOT/tensor-collection-qwen-missing.out"
grep 'evidence: header-metadata-only' "$ROOT/tensor-collection-qwen-missing.out"
grep 'collections: embedding=0 attention_qkvo=0 mlp_gud=0 norm=0 head=0 moe=0' "$ROOT/tensor-collection-qwen-missing.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/tensor-collection-qwen-missing.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-collection-qwen-missing.out"
grep 'boundary: tensor collection inventory only; no role mapping/runtime/generation' "$ROOT/tensor-collection-qwen-missing.out"

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/tensor-collection-qwen-missing-table.out"
grep 'TENSOR COLLECTION INVENTORY' "$ROOT/tensor-collection-qwen-missing-table.out"
matches "$ROOT/tensor-collection-qwen-missing-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}source-missing[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_status: source-missing' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_family: qwen' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_target_id: qwen3-8b' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_stage: header-collection-inventory' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_evidence_basis: header-metadata-only' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_source_status: missing' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_tensor_count: 0' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_embedding_tensor_count: 0' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_attention_complete_qkvo_layer_count: 0' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_mlp_complete_gud_layer_count: 0' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'tensor_collection_role_mapping_status: not-implemented' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'release_ready: false' "$ROOT/tensor-collection-qwen-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-collection-qwen-missing-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tensor-map-qwen-missing.out"
grep 'tensor-map: qwen3-8b \[blocked\]' "$ROOT/tensor-map-qwen-missing.out"
grep 'family: qwen  stage: header-naming-map  evidence: header-only' "$ROOT/tensor-map-qwen-missing.out"
grep 'roles: total=0 embedding=0 attention=0 mlp=0 norm=0 head=0 moe=0 unknown=0' "$ROOT/tensor-map-qwen-missing.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/tensor-map-qwen-missing.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-map-qwen-missing.out"
grep 'boundary: report-only; use --audit for tensor entries' "$ROOT/tensor-map-qwen-missing.out"
! grep 'tensor_map.entry.' "$ROOT/tensor-map-qwen-missing.out"
! grep 'runtime_claim:' "$ROOT/tensor-map-qwen-missing.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/tensor-map-qwen-missing-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-qwen-missing-table.out"
matches "$ROOT/tensor-map-qwen-missing-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}source-missing[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map qwen3-8b --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_status: source-missing' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_family: qwen' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_target_id: qwen3-8b' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_stage: header-naming-map' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_evidence_basis: header-metadata-only' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_source_status: missing' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_mapped_total_count: 0' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_unmapped_unknown_count: 0' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_runtime_role_coverage_status: report-only' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_artifact_contract_status: not-implemented' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_runtime_descriptor_status: not-implemented' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'tensor_map_graph_consumer_status: not-implemented' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'release_ready: false' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/tensor-map-qwen-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-map-qwen-missing-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tensor-map-gemma-missing.out"
grep 'tensor-map: gemma-4-12b-it \[blocked\]' "$ROOT/tensor-map-gemma-missing.out"
grep 'family: gemma  stage: header-naming-map  evidence: header-only' "$ROOT/tensor-map-gemma-missing.out"
grep 'roles: total=0 embedding=0 attention=0 mlp=0 norm=0 head=0 moe=0 unknown=0' "$ROOT/tensor-map-gemma-missing.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/tensor-map-gemma-missing.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-map-gemma-missing.out"
grep 'boundary: report-only; use --audit for tensor entries' "$ROOT/tensor-map-gemma-missing.out"
! grep 'tensor_map.entry.' "$ROOT/tensor-map-gemma-missing.out"
! grep 'runtime_claim:' "$ROOT/tensor-map-gemma-missing.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/tensor-map-gemma-missing-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-gemma-missing-table.out"
matches "$ROOT/tensor-map-gemma-missing-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}source-missing[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_status: source-missing' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_family: gemma' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_target_id: gemma-4-12b-it' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_stage: header-naming-map' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_evidence_basis: header-metadata-only' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_source_status: missing' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_mapped_total_count: 0' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_unmapped_unknown_count: 0' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_runtime_role_coverage_status: report-only' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_artifact_contract_status: not-implemented' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_runtime_descriptor_status: not-implemented' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'tensor_map_graph_consumer_status: not-implemented' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'release_ready: false' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/tensor-map-gemma-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-map-gemma-missing-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --models-root "$CLASS_MISSING_ROOT" > "$ROOT/output-head-qwen-missing.out"
grep 'output-head-map: qwen3-8b \[blocked\]' "$ROOT/output-head-qwen-missing.out"
grep 'family: qwen  evidence: header-only' "$ROOT/output-head-qwen-missing.out"
grep 'head: missing  final_norm: missing  embedding: missing  tie: unknown' "$ROOT/output-head-qwen-missing.out"
grep 'shape: unknown' "$ROOT/output-head-qwen-missing.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/output-head-qwen-missing.out"
grep 'next: V010.MAP.8' "$ROOT/output-head-qwen-missing.out"
grep 'boundary: mapping only; no logits/runtime/generation' "$ROOT/output-head-qwen-missing.out"
! grep 'output_head_map_' "$ROOT/output-head-qwen-missing.out"
! grep 'runtime_claim:' "$ROOT/output-head-qwen-missing.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --models-root "$CLASS_MISSING_ROOT" > "$ROOT/output-head-gemma-missing.out"
grep 'output-head-map: gemma-4-12b-it \[blocked\]' "$ROOT/output-head-gemma-missing.out"
grep 'family: gemma  evidence: header-only' "$ROOT/output-head-gemma-missing.out"
grep 'head: missing  final_norm: missing  embedding: missing  tie: unknown' "$ROOT/output-head-gemma-missing.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/output-head-gemma-missing.out"
grep 'next: V010.MAP.8' "$ROOT/output-head-gemma-missing.out"
! grep 'output_head_map_' "$ROOT/output-head-gemma-missing.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/output-head-gemma-missing-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-gemma-missing-table.out"
matches "$ROOT/output-head-gemma-missing-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}source-missing[[:space:]]{2,}no[[:space:]]{2,}no[[:space:]]{2,}no[[:space:]]{2,}unknown[[:space:]]{2,}unknown[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_status: source-missing' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_family: gemma' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_target_id: gemma-4-12b-it' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_stage: header-output-head-map' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_evidence_basis: header-metadata-only' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_map_source_status: missing' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_candidate_count: 0' "$ROOT/output-head-gemma-missing-audit.out"
grep 'output_head_missing_status: missing' "$ROOT/output-head-gemma-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-gemma-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-gemma-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/output-head-gemma-missing-audit.out"
grep 'release_ready: false' "$ROOT/output-head-gemma-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/output-head-gemma-missing-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tokenizer-map-qwen-missing.out"
grep 'tokenizer-map: qwen3-8b' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'family: qwen' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'status: source-missing' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'tokenizer: missing' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'vocab: missing' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'merges: missing' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'chat_template: unknown' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'specials: missing' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'runtime: unsupported' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'next: V010.MAP.7' "$ROOT/tokenizer-map-qwen-missing.out"
grep 'boundary: tokenizer metadata mapping only; no tokenization/detokenization/runtime/generation' "$ROOT/tokenizer-map-qwen-missing.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/tokenizer-map-qwen-missing-table.out"
grep 'TOKENIZER METADATA MAP' "$ROOT/tokenizer-map-qwen-missing-table.out"
matches "$ROOT/tokenizer-map-qwen-missing-table.out" '^qwen3-8b[[:space:]]{2,}qwen[[:space:]]{2,}source-missing[[:space:]]{2,}no[[:space:]]{2,}missing[[:space:]]{2,}missing[[:space:]]{2,}unknown[[:space:]]{2,}missing[[:space:]]{2,}V010\.MAP\.7$'

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_status: source-missing' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_family: qwen' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_target_id: qwen3-8b' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_stage: metadata-tokenizer-map' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_evidence_basis: sidecar-json-only' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_map_source_status: missing' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'tokenization_status: not-implemented' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'detokenization_status: not-implemented' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'release_ready: false' "$ROOT/tokenizer-map-qwen-missing-audit.out"
grep 'next_required_rows: V010.MAP.7' "$ROOT/tokenizer-map-qwen-missing-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role tokenizer --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tokenizer-map-gemma-missing.out"
grep 'tokenizer-map: gemma-4-12b-it' "$ROOT/tokenizer-map-gemma-missing.out"
grep 'family: gemma' "$ROOT/tokenizer-map-gemma-missing.out"
grep 'status: source-missing' "$ROOT/tokenizer-map-gemma-missing.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/tokenizer-map-gemma-missing.out"
grep 'next: V010.MAP.7' "$ROOT/tokenizer-map-gemma-missing.out"

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" > "$ROOT/model-class-gemma-missing.out"
grep 'model-class: gemma' "$ROOT/model-class-gemma-missing.out"
grep 'target: gemma-4-12b-it' "$ROOT/model-class-gemma-missing.out"
grep 'status: source-missing' "$ROOT/model-class-gemma-missing.out"
grep 'class: gemma-source-model-class-profile' "$ROOT/model-class-gemma-missing.out"
grep 'evidence: header-metadata-only' "$ROOT/model-class-gemma-missing.out"
grep 'patterns: tensors=0 attn=0 mlp=0 norm=0 head=0 moe=0' "$ROOT/model-class-gemma-missing.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/model-class-gemma-missing.out"
grep 'next: V010.MAP.8' "$ROOT/model-class-gemma-missing.out"
grep 'no tensor role mapping/runtime/generation' "$ROOT/model-class-gemma-missing.out"

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/model-class-gemma-missing-table.out"
grep 'MODEL CLASS PROFILE' "$ROOT/model-class-gemma-missing-table.out"
matches "$ROOT/model-class-gemma-missing-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}source-missing[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_profile_status: source-missing' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_family: gemma' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_target_id: gemma-4-12b-it' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_name: gemma-source-model-class-profile' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_runtime_shape: dense-causal-decoder-candidate-pending-config' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_source_metadata_status: missing' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_tensor_count: 0' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_pattern_status: lexical-only' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_role_mapping_status: not-implemented' "$ROOT/model-class-gemma-missing-audit.out"
grep 'model_class_runtime_status: unsupported' "$ROOT/model-class-gemma-missing-audit.out"
grep 'backend_selection: deferred' "$ROOT/model-class-gemma-missing-audit.out"
grep 'backend_pressure: cpu-cuda-baseline-planned' "$ROOT/model-class-gemma-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/model-class-gemma-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/model-class-gemma-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/model-class-gemma-missing-audit.out"
grep 'release_ready: false' "$ROOT/model-class-gemma-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/model-class-gemma-missing-audit.out"

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" > "$ROOT/tensor-collection-gemma-missing.out"
grep 'tensor-collection: gemma' "$ROOT/tensor-collection-gemma-missing.out"
grep 'target: gemma-4-12b-it' "$ROOT/tensor-collection-gemma-missing.out"
grep 'status: source-missing' "$ROOT/tensor-collection-gemma-missing.out"
grep 'stage: header-collection-inventory' "$ROOT/tensor-collection-gemma-missing.out"
grep 'evidence: header-metadata-only' "$ROOT/tensor-collection-gemma-missing.out"
grep 'collections: embedding=0 attention_qkvo=0 mlp_gud=0 norm=0 head=0 moe=0' "$ROOT/tensor-collection-gemma-missing.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/tensor-collection-gemma-missing.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-collection-gemma-missing.out"
grep 'boundary: tensor collection inventory only; no role mapping/runtime/generation' "$ROOT/tensor-collection-gemma-missing.out"

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --output table > "$ROOT/tensor-collection-gemma-missing-table.out"
grep 'TENSOR COLLECTION INVENTORY' "$ROOT/tensor-collection-gemma-missing-table.out"
matches "$ROOT/tensor-collection-gemma-missing-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}source-missing[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --models-root "$CLASS_MISSING_ROOT" --audit > "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_status: source-missing' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_family: gemma' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_target_id: gemma-4-12b-it' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_stage: header-collection-inventory' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_evidence_basis: header-metadata-only' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_source_status: missing' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_tensor_count: 0' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_embedding_tensor_count: 0' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_attention_complete_qkvo_layer_count: 0' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_mlp_complete_gud_layer_count: 0' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'tensor_collection_role_mapping_status: not-implemented' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'release_ready: false' "$ROOT/tensor-collection-gemma-missing-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-collection-gemma-missing-audit.out"

QWEN_CLASS_SOURCE="${TMPDIR:-/tmp}/yvex-qwen-class-profile-test-$$"
yvex_test_cleanup "$QWEN_CLASS_SOURCE"
mkdir -p "$QWEN_CLASS_SOURCE"
printf '{}\n' > "$QWEN_CLASS_SOURCE/config.json"
printf '{}\n' > "$QWEN_CLASS_SOURCE/tokenizer.json"
python3 - "$QWEN_CLASS_SOURCE/model-00001-of-00001.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "lm_head.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target class-profile qwen3-8b --source "$QWEN_CLASS_SOURCE" > "$ROOT/model-class-qwen.out"
grep 'status: metadata-profiled' "$ROOT/model-class-qwen.out"
grep 'patterns: tensors=10 attn=4 mlp=3 norm=2 head=1 moe=0' "$ROOT/model-class-qwen.out"
grep 'top_blocker: missing-qwen-tensor-role-map' "$ROOT/model-class-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/model-class-qwen.out"

"$YVEX_BIN" inspect target class-profile qwen3-8b --source "$QWEN_CLASS_SOURCE" --output table > "$ROOT/model-class-qwen-table.out"
grep 'MODEL CLASS PROFILE' "$ROOT/model-class-qwen-table.out"
matches "$ROOT/model-class-qwen-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}metadata-profiled[[:space:]]{2,}10[[:space:]]{2,}4[[:space:]]{2,}3[[:space:]]{2,}2[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target class-profile qwen3-8b --source "$QWEN_CLASS_SOURCE" --audit > "$ROOT/model-class-qwen-audit.out"
grep 'model_class_profile_status: metadata-profiled' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_config_status: present' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_tokenizer_status: present' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_source_metadata_status: header-only' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_tensor_count: 10' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_embedding_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_attention_q_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_attention_k_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_attention_v_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_attention_o_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_mlp_gate_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_mlp_up_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_mlp_down_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_norm_pattern_count: 2' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_output_head_pattern_count: 1' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_moe_router_pattern_count: 0' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_moe_expert_pattern_count: 0' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_other_pattern_count: 0' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_pattern_status: lexical-only' "$ROOT/model-class-qwen-audit.out"
grep 'model_class_role_mapping_status: not-implemented' "$ROOT/model-class-qwen-audit.out"
grep 'backend_selection: deferred' "$ROOT/model-class-qwen-audit.out"
grep 'backend_pressure: metal-planned' "$ROOT/model-class-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/model-class-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/model-class-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/model-class-qwen-audit.out"
grep 'release_ready: false' "$ROOT/model-class-qwen-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/model-class-qwen-audit.out"

QWEN_CLASS_MODELS_ROOT="$ROOT/qwen-class-models-root"
YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire qwen3-8b --models-root "$QWEN_CLASS_MODELS_ROOT" --auth auto --progress off > "$ROOT/qwen-class-acquire.out"
cp -R "$QWEN_CLASS_SOURCE/." "$QWEN_CLASS_MODELS_ROOT/source/hf/Qwen/Qwen3-8B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/"
"$YVEX_BIN" inspect target class-profile qwen3-8b --models-root "$QWEN_CLASS_MODELS_ROOT" --audit > "$ROOT/model-class-qwen-models-root-audit.out"
grep 'model_class_profile_status: metadata-profiled' "$ROOT/model-class-qwen-models-root-audit.out"
matches "$ROOT/model-class-qwen-models-root-audit.out" 'source_path: .*/qwen-class-models-root/source/hf/Qwen/Qwen3-8B/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08$'
grep 'model_class_source_metadata_status: header-only' "$ROOT/model-class-qwen-models-root-audit.out"

QWEN_COLLECTION_SOURCE="${TMPDIR:-/tmp}/yvex-qwen-tensor-collection-test-$$"
yvex_test_cleanup "$QWEN_COLLECTION_SOURCE"
mkdir -p "$QWEN_COLLECTION_SOURCE"
printf '{}\n' > "$QWEN_COLLECTION_SOURCE/config.json"
printf '{}\n' > "$QWEN_COLLECTION_SOURCE/tokenizer.json"
python3 - "$QWEN_COLLECTION_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --source "$QWEN_COLLECTION_SOURCE" > "$ROOT/tensor-collection-qwen.out"
grep 'tensor-collection: qwen' "$ROOT/tensor-collection-qwen.out"
grep 'target: qwen3-8b' "$ROOT/tensor-collection-qwen.out"
grep 'status: collection-profiled' "$ROOT/tensor-collection-qwen.out"
grep 'stage: header-collection-inventory' "$ROOT/tensor-collection-qwen.out"
grep 'evidence: header-metadata-only' "$ROOT/tensor-collection-qwen.out"
grep 'collections: embedding=1 attention_qkvo=1 mlp_gud=1 norm=3 head=1 moe=0' "$ROOT/tensor-collection-qwen.out"
grep 'layers_observed: 1' "$ROOT/tensor-collection-qwen.out"
grep 'top_blocker: missing-qwen-tensor-role-map' "$ROOT/tensor-collection-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-collection-qwen.out"
grep 'boundary: tensor collection inventory only; no role mapping/runtime/generation' "$ROOT/tensor-collection-qwen.out"

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --output table > "$ROOT/tensor-collection-qwen-table.out"
grep 'TENSOR COLLECTION INVENTORY' "$ROOT/tensor-collection-qwen-table.out"
matches "$ROOT/tensor-collection-qwen-table.out" '^FAMILY[[:space:]]{2,}TARGET[[:space:]]{2,}STATUS[[:space:]]{2,}EMBED[[:space:]]{2,}ATTN_QKVO[[:space:]]{2,}MLP_GUD[[:space:]]{2,}NORM[[:space:]]{2,}HEAD[[:space:]]{2,}MOE[[:space:]]{2,}LAYERS[[:space:]]{2,}NEXT$'
matches "$ROOT/tensor-collection-qwen-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}collection-profiled[[:space:]]{2,}1[[:space:]]{2,}1[[:space:]]{2,}1[[:space:]]{2,}3[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}1[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target tensor-collection qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --audit > "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_status: collection-profiled' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_family: qwen' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_target_id: qwen3-8b' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_stage: header-collection-inventory' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_evidence_basis: header-metadata-only' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_source_status: present' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_manifest_status: not-checked' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_config_status: present' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_tokenizer_status: present' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_tensor_count: 12' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_layer_count_observed: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_embedding_status: candidate' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_embedding_tensor_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_status: candidate' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_q_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_k_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_v_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_o_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_attention_complete_qkvo_layer_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_mlp_status: candidate' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_mlp_gate_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_mlp_up_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_mlp_down_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_mlp_complete_gud_layer_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_norm_status: candidate' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_norm_tensor_count: 3' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_output_head_status: candidate' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_output_head_tensor_count: 1' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_moe_status: not-observed' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_moe_router_count: 0' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_moe_expert_count: 0' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_tokenizer_collection_status: sidecar-observed' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_kv_runtime_state_status: runtime-state-required-not-implemented' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_validation_status: lexical-and-header-only' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_role_mapping_status: not-implemented' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_runtime_descriptor_status: not-implemented' "$ROOT/tensor-collection-qwen-audit.out"
grep 'tensor_collection_graph_consumer_status: not-implemented' "$ROOT/tensor-collection-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-collection-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-collection-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-collection-qwen-audit.out"
grep 'release_ready: false' "$ROOT/tensor-collection-qwen-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-collection-qwen-audit.out"
! grep 'generation_ready: tr''ue' "$ROOT/tensor-collection-qwen-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" > "$ROOT/tensor-map-qwen.out"
grep 'tensor-map: qwen3-8b \[reported\]' "$ROOT/tensor-map-qwen.out"
grep 'family: qwen  stage: header-naming-map  evidence: header-only' "$ROOT/tensor-map-qwen.out"
grep 'roles: total=12 embedding=1 attention=4 mlp=3 norm=3 head=1 moe=0 unknown=0' "$ROOT/tensor-map-qwen.out"
grep 'layers: 1' "$ROOT/tensor-map-qwen.out"
grep 'top_blocker: missing-qwen-runtime-role-validation' "$ROOT/tensor-map-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-map-qwen.out"
grep 'boundary: report-only; use --audit for tensor entries' "$ROOT/tensor-map-qwen.out"
! grep 'tensor_map.entry.' "$ROOT/tensor-map-qwen.out"
! grep 'runtime_claim:' "$ROOT/tensor-map-qwen.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --output table > "$ROOT/tensor-map-qwen-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-qwen-table.out"
matches "$ROOT/tensor-map-qwen-table.out" '^FAMILY[[:space:]]{2,}TARGET[[:space:]]{2,}STATUS[[:space:]]{2,}TOTAL[[:space:]]{2,}EMBED[[:space:]]{2,}ATTN[[:space:]]{2,}MLP[[:space:]]{2,}NORM[[:space:]]{2,}HEAD[[:space:]]{2,}MOE[[:space:]]{2,}UNKNOWN[[:space:]]{2,}LAYERS[[:space:]]{2,}NEXT$'
matches "$ROOT/tensor-map-qwen-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}naming-map-profiled[[:space:]]{2,}12[[:space:]]{2,}1[[:space:]]{2,}4[[:space:]]{2,}3[[:space:]]{2,}3[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}1[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --audit > "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_status: naming-map-profiled' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_family: qwen' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_target_id: qwen3-8b' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_stage: header-naming-map' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_evidence_basis: header-metadata-only' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_source_status: present' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_config_status: present' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_tokenizer_status: present' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_tensor_count: 12' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_mapped_total_count: 12' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_unmapped_unknown_count: 0' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_ambiguous_count: 0' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_layer_count_observed: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_embedding_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_attention_count: 4' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_attention_q_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_attention_k_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_attention_v_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_attention_o_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_mlp_count: 3' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_mlp_gate_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_mlp_up_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_mlp_down_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_norm_count: 3' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_output_head_count: 1' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_moe_router_count: 0' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_moe_expert_count: 0' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_validation_status: lexical-and-header-only' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_canonical_role_status: mapped-candidates' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_runtime_role_coverage_status: report-only' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_artifact_contract_status: not-implemented' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_runtime_descriptor_status: not-implemented' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map_graph_consumer_status: not-implemented' "$ROOT/tensor-map-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-map-qwen-audit.out"
grep 'release_ready: false' "$ROOT/tensor-map-qwen-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-map-qwen-audit.out"
grep 'tensor_map.entry.' "$ROOT/tensor-map-qwen-audit.out"
grep 'model.embed_tokens.weight -> model.embedding.token.weight' "$ROOT/tensor-map-qwen-audit.out"
grep 'model.layers.0.self_attn.q_proj.weight -> model.layers.0.attention.q_proj.weight' "$ROOT/tensor-map-qwen-audit.out"
grep 'model.layers.0.input_layernorm.weight -> model.layers.0.attention.norm.weight' "$ROOT/tensor-map-qwen-audit.out"
grep 'lm_head.weight -> model.output_head.weight' "$ROOT/tensor-map-qwen-audit.out"
! grep 'generation_ready: tr''ue' "$ROOT/tensor-map-qwen-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --check-output-contract normal > "$ROOT/output-contract-qwen-tensor-map-normal.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --check-output-contract table > "$ROOT/output-contract-qwen-tensor-map-table.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_COLLECTION_SOURCE" --check-output-contract audit > "$ROOT/output-contract-qwen-tensor-map-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" > "$ROOT/output-head-qwen.out"
grep 'output-head-map: qwen3-8b \[reported\]' "$ROOT/output-head-qwen.out"
grep 'family: qwen  evidence: header-only' "$ROOT/output-head-qwen.out"
grep 'head: model.output_head.weight  final_norm: model.final_norm.weight  embedding: model.embedding.token.weight  tie: separate-output-head-candidate' "$ROOT/output-head-qwen.out"
grep 'shape: compatible-same-shape' "$ROOT/output-head-qwen.out"
grep 'top_blocker: missing-output-head-runtime-consumer' "$ROOT/output-head-qwen.out"
grep 'next: V010.MAP.8' "$ROOT/output-head-qwen.out"
grep 'boundary: mapping only; no logits/runtime/generation' "$ROOT/output-head-qwen.out"
! grep 'output_head_map_' "$ROOT/output-head-qwen.out"
! grep 'native_output_head:' "$ROOT/output-head-qwen.out"
! grep 'runtime_claim:' "$ROOT/output-head-qwen.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" --output table > "$ROOT/output-head-qwen-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-qwen-table.out"
matches "$ROOT/output-head-qwen-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}output-head-profiled[[:space:]]{2,}yes[[:space:]]{2,}yes[[:space:]]{2,}yes[[:space:]]{2,}separate-output-head-candidate[[:space:]]{2,}compatible-same-shape[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" --audit > "$ROOT/output-head-qwen-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_map_family: qwen' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_map_target_id: qwen3-8b' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_map_stage: header-output-head-map' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_map_evidence_basis: header-metadata-only' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_native_name: lm_head.weight' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_canonical_role: model.output_head.weight' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_mapping_status: mapped-candidate' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_candidate_count: 1' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_ambiguous_count: 0' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_missing_status: present' "$ROOT/output-head-qwen-audit.out"
grep 'embedding_canonical_role: model.embedding.token.weight' "$ROOT/output-head-qwen-audit.out"
grep 'final_norm_canonical_role: model.final_norm.weight' "$ROOT/output-head-qwen-audit.out"
grep 'tie_policy_status: separate-output-head-candidate' "$ROOT/output-head-qwen-audit.out"
grep 'config_tie_word_embeddings_status: missing' "$ROOT/output-head-qwen-audit.out"
grep 'shape_relation_status: compatible-same-shape' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_runtime_consumer_status: target-runtime-owned' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_logits_status: target-capability-dependent' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_artifact_contract_status: artifact-owner' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_runtime_descriptor_status: runtime-owner' "$ROOT/output-head-qwen-audit.out"
grep 'output_head_graph_consumer_status: runtime-logits-owner' "$ROOT/output-head-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/output-head-qwen-audit.out"
grep 'release_ready: false' "$ROOT/output-head-qwen-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/output-head-qwen-audit.out"
grep 'output_head.entry.output.native_name: lm_head.weight' "$ROOT/output-head-qwen-audit.out"
grep 'output_head.entry.output.canonical_role: model.output_head.weight' "$ROOT/output-head-qwen-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" --check-output-contract normal > "$ROOT/output-contract-qwen-output-head-normal.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" --check-output-contract table > "$ROOT/output-contract-qwen-output-head-table.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_COLLECTION_SOURCE" --check-output-contract audit > "$ROOT/output-contract-qwen-output-head-audit.out"

TOKENIZER_COMPLETE_SOURCE="${TMPDIR:-/tmp}/yvex-tokenizer-map-complete-test-$$"
yvex_test_cleanup "$TOKENIZER_COMPLETE_SOURCE"
mkdir -p "$TOKENIZER_COMPLETE_SOURCE"
cat > "$TOKENIZER_COMPLETE_SOURCE/config.json" <<'JSON'
{
  "model_type": "qwen",
  "vocab_size": 16,
  "hidden_size": 8,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "unk_token_id": 3,
  "tie_word_embeddings": false
}
JSON
cat > "$TOKENIZER_COMPLETE_SOURCE/tokenizer_config.json" <<'JSON'
{
  "tokenizer_class": "PreTrainedTokenizerFast",
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0,
  "unk_token_id": 3,
  "chat_template": "{{ bos_token }}{{ messages }}"
}
JSON
cat > "$TOKENIZER_COMPLETE_SOURCE/special_tokens_map.json" <<'JSON'
{
  "bos_token": "<s>",
  "eos_token": "</s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "additional_special_tokens": ["<extra_0>", "<extra_1>"]
}
JSON
cat > "$TOKENIZER_COMPLETE_SOURCE/generation_config.json" <<'JSON'
{
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 0
}
JSON
cat > "$TOKENIZER_COMPLETE_SOURCE/tokenizer.json" <<'JSON'
{
  "version": "1.0",
  "model": {"type": "BPE"},
  "added_tokens": []
}
JSON
python3 - "$TOKENIZER_COMPLETE_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

items = [
    ("model.embed_tokens.weight", [16, 8]),
    ("model.norm.weight", [8]),
    ("lm_head.weight", [16, 8]),
]
offset = 0
header = {}
for name, shape in items:
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * 4
    header[name] = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + nbytes],
    }
    offset += nbytes
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tokenizer-map qwen3-8b --source "$TOKENIZER_COMPLETE_SOURCE" > "$ROOT/tokenizer-map-qwen.out"
grep 'tokenizer-map: qwen3-8b' "$ROOT/tokenizer-map-qwen.out"
grep 'family: qwen' "$ROOT/tokenizer-map-qwen.out"
grep 'status: present-report-only' "$ROOT/tokenizer-map-qwen.out"
grep 'tokenizer: present' "$ROOT/tokenizer-map-qwen.out"
grep 'vocab: embedded-or-tokenizer-json' "$ROOT/tokenizer-map-qwen.out"
grep 'chat_template: present' "$ROOT/tokenizer-map-qwen.out"
grep 'specials: present' "$ROOT/tokenizer-map-qwen.out"
grep 'runtime: unsupported' "$ROOT/tokenizer-map-qwen.out"
grep 'top_blocker: quant-policy-or-artifact-emitter' "$ROOT/tokenizer-map-qwen.out"
grep 'next: V010.QUANT.1' "$ROOT/tokenizer-map-qwen.out"
grep 'boundary: tokenizer metadata mapping only; no tokenization/detokenization/runtime/generation' "$ROOT/tokenizer-map-qwen.out"

"$YVEX_BIN" inspect target tokenizer-map qwen3-8b --source "$TOKENIZER_COMPLETE_SOURCE" --output table > "$ROOT/tokenizer-map-qwen-table.out"
grep 'TOKENIZER METADATA MAP' "$ROOT/tokenizer-map-qwen-table.out"
matches "$ROOT/tokenizer-map-qwen-table.out" '^TARGET[[:space:]]{2,}FAMILY[[:space:]]{2,}STATUS[[:space:]]{2,}TOKENIZER[[:space:]]{2,}VOCAB[[:space:]]{2,}MERGES[[:space:]]{2,}CHAT_TEMPLATE[[:space:]]{2,}SPECIALS[[:space:]]{2,}NEXT$'
matches "$ROOT/tokenizer-map-qwen-table.out" '^qwen3-8b[[:space:]]{2,}qwen[[:space:]]{2,}present-report-only[[:space:]]{2,}yes[[:space:]]{2,}embedded-or-tokenizer-json[[:space:]]{2,}missing[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}V010\.QUANT\.1$'

"$YVEX_BIN" inspect target tokenizer-map qwen3-8b --source "$TOKENIZER_COMPLETE_SOURCE" --audit > "$ROOT/tokenizer-map-qwen-audit.out"
grep 'schema_version: yvex.source.tokenizer_map.v1' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_map_family: qwen' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_map_target_id: qwen3-8b' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_map_stage: metadata-tokenizer-map' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_map_evidence_basis: sidecar-json-only' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_json_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_config_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'special_tokens_map_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'generation_config_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'config_json_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_class: PreTrainedTokenizerFast' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'model_type: qwen' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'vocab_size_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'vocab_size: 16' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'config_vocab_size: 16' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'output_head_vocab_dim_candidate: 16' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'output_head_vocab_relation_status: vocab-size-matches-output-head' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'bos_token_id_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'bos_token_id: 1' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'eos_token_id_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'eos_token_id: 2' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'pad_token_id_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'pad_token_id: 0' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'unk_token_id_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'unk_token_id: 3' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'additional_special_tokens_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'additional_special_tokens_count: 2' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'chat_template_status: present' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'chat_template_present: true' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'chat_template_hash_status: not-computed' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'gguf_tokenizer_contract_status: planned' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'tokenization_status: not-implemented' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'detokenization_status: not-implemented' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'eos_stop_policy_status: not-implemented' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'release_ready: false' "$ROOT/tokenizer-map-qwen-audit.out"
grep 'next_required_rows: V010.QUANT.1' "$ROOT/tokenizer-map-qwen-audit.out"

"$YVEX_BIN" inspect target tokenizer-map gemma-4-12b-it --source "$TOKENIZER_COMPLETE_SOURCE" > "$ROOT/tokenizer-map-gemma.out"
grep 'tokenizer-map: gemma-4-12b-it' "$ROOT/tokenizer-map-gemma.out"
grep 'family: gemma' "$ROOT/tokenizer-map-gemma.out"
grep 'status: present-report-only' "$ROOT/tokenizer-map-gemma.out"
grep 'next: V010.QUANT.1' "$ROOT/tokenizer-map-gemma.out"

"$YVEX_BIN" inspect target tokenizer-map gemma-4-12b-it --source "$TOKENIZER_COMPLETE_SOURCE" --output table > "$ROOT/tokenizer-map-gemma-table.out"
grep 'TOKENIZER METADATA MAP' "$ROOT/tokenizer-map-gemma-table.out"
matches "$ROOT/tokenizer-map-gemma-table.out" '^gemma-4-12b-it[[:space:]]{2,}gemma[[:space:]]{2,}present-report-only[[:space:]]{2,}yes[[:space:]]{2,}embedded-or-tokenizer-json[[:space:]]{2,}not-required-or-absent[[:space:]]{2,}present[[:space:]]{2,}present[[:space:]]{2,}V010\.QUANT\.1$'

"$YVEX_BIN" inspect target tokenizer-map gemma-4-12b-it --source "$TOKENIZER_COMPLETE_SOURCE" --audit > "$ROOT/tokenizer-map-gemma-audit.out"
grep 'tokenizer_map_status: present-report-only' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'tokenizer_map_family: gemma' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'tokenizer_map_target_id: gemma-4-12b-it' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'output_head_vocab_relation_status: vocab-size-matches-output-head' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-gemma-audit.out"

TOKENIZER_MISSING_SOURCE="${TMPDIR:-/tmp}/yvex-tokenizer-map-missing-test-$$"
yvex_test_cleanup "$TOKENIZER_MISSING_SOURCE"
mkdir -p "$TOKENIZER_MISSING_SOURCE"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --source "$TOKENIZER_MISSING_SOURCE" --audit > "$ROOT/tokenizer-map-metadata-missing-audit.out"
grep 'tokenizer_map_status: metadata-missing' "$ROOT/tokenizer-map-metadata-missing-audit.out"
grep 'top_blocker: missing-tokenizer-sidecars' "$ROOT/tokenizer-map-metadata-missing-audit.out"
grep 'next_required_rows: V010.MAP.7' "$ROOT/tokenizer-map-metadata-missing-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-metadata-missing-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-metadata-missing-audit.out"
yvex_test_cleanup "$TOKENIZER_MISSING_SOURCE"

TOKENIZER_INCOMPLETE_SOURCE="${TMPDIR:-/tmp}/yvex-tokenizer-map-incomplete-test-$$"
yvex_test_cleanup "$TOKENIZER_INCOMPLETE_SOURCE"
mkdir -p "$TOKENIZER_INCOMPLETE_SOURCE"
printf '{"vocab_size":16}\n' > "$TOKENIZER_INCOMPLETE_SOURCE/config.json"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --source "$TOKENIZER_INCOMPLETE_SOURCE" --audit > "$ROOT/tokenizer-map-incomplete-audit.out"
grep 'tokenizer_map_status: tokenizer-metadata-incomplete' "$ROOT/tokenizer-map-incomplete-audit.out"
grep 'vocab_size: 16' "$ROOT/tokenizer-map-incomplete-audit.out"
grep 'tokenizer_json_status: missing' "$ROOT/tokenizer-map-incomplete-audit.out"
grep 'tokenizer_runtime_status: not-implemented' "$ROOT/tokenizer-map-incomplete-audit.out"
grep 'next_required_rows: V010.MAP.7' "$ROOT/tokenizer-map-incomplete-audit.out"
yvex_test_cleanup "$TOKENIZER_INCOMPLETE_SOURCE"

TOKENIZER_MISMATCH_SOURCE="${TMPDIR:-/tmp}/yvex-tokenizer-map-mismatch-test-$$"
yvex_test_cleanup "$TOKENIZER_MISMATCH_SOURCE"
cp -R "$TOKENIZER_COMPLETE_SOURCE" "$TOKENIZER_MISMATCH_SOURCE"
perl -0pi -e 's/"vocab_size": 16/"vocab_size": 17/' "$TOKENIZER_MISMATCH_SOURCE/config.json"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --source "$TOKENIZER_MISMATCH_SOURCE" --audit > "$ROOT/tokenizer-map-mismatch-audit.out"
grep 'tokenizer_map_status: tokenizer-metadata-ambiguous' "$ROOT/tokenizer-map-mismatch-audit.out"
grep 'vocab_size: 17' "$ROOT/tokenizer-map-mismatch-audit.out"
grep 'output_head_vocab_relation_status: vocab-size-mismatch-output-head' "$ROOT/tokenizer-map-mismatch-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-mismatch-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-mismatch-audit.out"
yvex_test_cleanup "$TOKENIZER_MISMATCH_SOURCE"

TOKENIZER_MALFORMED_SOURCE="${TMPDIR:-/tmp}/yvex-tokenizer-map-malformed-test-$$"
yvex_test_cleanup "$TOKENIZER_MALFORMED_SOURCE"
cp -R "$TOKENIZER_COMPLETE_SOURCE" "$TOKENIZER_MALFORMED_SOURCE"
printf '{"tokenizer_class":' > "$TOKENIZER_MALFORMED_SOURCE/tokenizer_config.json"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --source "$TOKENIZER_MALFORMED_SOURCE" --audit > "$ROOT/tokenizer-map-malformed-audit.out"
grep 'tokenizer_map_status: tokenizer-metadata-malformed' "$ROOT/tokenizer-map-malformed-audit.out"
grep 'tokenizer_config_status: malformed' "$ROOT/tokenizer-map-malformed-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tokenizer-map-malformed-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tokenizer-map-malformed-audit.out"
yvex_test_cleanup "$TOKENIZER_MALFORMED_SOURCE"
yvex_test_cleanup "$TOKENIZER_COMPLETE_SOURCE"

MISSING_ROLE_COMPLETE_SOURCE="${TMPDIR:-/tmp}/yvex-missing-role-complete-test-$$"
make_missing_role_source "$MISSING_ROLE_COMPLETE_SOURCE" complete

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" > "$ROOT/missing-role-qwen.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen.out"
grep 'family: qwen  evidence: header+sidecar-only' "$ROOT/missing-role-qwen.out"
grep 'source_roles: 12/12 present, 0 missing, 0 ambiguous' "$ROOT/missing-role-qwen.out"
grep 'metadata_roles: 4/4 present, 0 missing, 0 ambiguous' "$ROOT/missing-role-qwen.out"
grep 'top_blocker: missing-artifact-contract' "$ROOT/missing-role-qwen.out"
grep 'next: V010.MAP.9' "$ROOT/missing-role-qwen.out"
grep 'boundary: report-only; use --audit for role details' "$ROOT/missing-role-qwen.out"
! grep 'missing_role.entry.' "$ROOT/missing-role-qwen.out"
! grep 'downstream_blockers:' "$ROOT/missing-role-qwen.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --output table > "$ROOT/missing-role-qwen-table.out"
grep 'MISSING ROLE BLOCKER REPORT' "$ROOT/missing-role-qwen-table.out"
matches "$ROOT/missing-role-qwen-table.out" '^FAMILY[[:space:]]{2,}TARGET[[:space:]]{2,}STATUS[[:space:]]{2,}OBS_SRC[[:space:]]{2,}MISS_SRC[[:space:]]{2,}AMBIG_SRC[[:space:]]{2,}OBS_META[[:space:]]{2,}MISS_META[[:space:]]{2,}TOP_BLOCKER[[:space:]]{2,}NEXT$'
matches "$ROOT/missing-role-qwen-table.out" '^qwen[[:space:]]{2,}qwen3-8b[[:space:]]{2,}missing-role-report-blocked[[:space:]]{2,}12[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}4[[:space:]]{2,}0[[:space:]]{2,}missing-artifact-contract[[:space:]]{2,}V010\.MAP\.9$'

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --audit > "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_report_status: missing-role-report-blocked' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_report_family: qwen' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_report_target_id: qwen3-8b' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_report_stage: missing-role-blocker-report' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_report_evidence_basis: header-and-sidecar-metadata-only' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_source_role_required_count: 12' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_source_role_observed_count: 12' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_source_role_missing_count: 0' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_metadata_required_count: 4' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_metadata_observed_count: 4' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_metadata_missing_count: 0' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_embedding_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_attention_norm_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_attention_q_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_attention_k_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_attention_v_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_attention_o_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_mlp_norm_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_mlp_gate_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_mlp_up_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_mlp_down_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_final_norm_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_output_head_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_tokenizer_metadata_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_config_metadata_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_generation_metadata_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_special_tokens_status: present' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_artifact_contract_status: missing' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_runtime_descriptor_status: missing' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_graph_consumer_status: missing' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_logits_runtime_status: missing' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_tokenizer_runtime_status: missing' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_top_blocker: missing-artifact-contract' "$ROOT/missing-role-qwen-audit.out"
grep 'missing_role_next_required_row: V010.MAP.9' "$ROOT/missing-role-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/missing-role-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/missing-role-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/missing-role-qwen-audit.out"
grep 'release_ready: false' "$ROOT/missing-role-qwen-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-qwen-missing-roles-normal.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract table > "$ROOT/output-contract-qwen-missing-roles-table.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-qwen-missing-roles-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" > "$ROOT/missing-role-gemma.out"
grep 'missing-roles: gemma-4-12b-it \[blocked\]' "$ROOT/missing-role-gemma.out"
grep 'family: gemma  evidence: header+sidecar-only' "$ROOT/missing-role-gemma.out"
grep 'source_roles: 12/12 present, 0 missing, 0 ambiguous' "$ROOT/missing-role-gemma.out"
grep 'metadata_roles: 4/4 present, 0 missing, 0 ambiguous' "$ROOT/missing-role-gemma.out"
grep 'top_blocker: missing-artifact-contract' "$ROOT/missing-role-gemma.out"
grep 'next: V010.MAP.9' "$ROOT/missing-role-gemma.out"
! grep 'missing_role.entry.' "$ROOT/missing-role-gemma.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --output table > "$ROOT/missing-role-gemma-table.out"
matches "$ROOT/missing-role-gemma-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}missing-role-report-blocked[[:space:]]{2,}12[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}4[[:space:]]{2,}0[[:space:]]{2,}missing-artifact-contract[[:space:]]{2,}V010\.MAP\.9$'
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --audit > "$ROOT/missing-role-gemma-audit.out"
grep 'missing_role_report_status: missing-role-report-blocked' "$ROOT/missing-role-gemma-audit.out"
grep 'missing_role_report_family: gemma' "$ROOT/missing-role-gemma-audit.out"
grep 'missing_role_source_role_observed_count: 12' "$ROOT/missing-role-gemma-audit.out"
grep 'missing_role_metadata_observed_count: 4' "$ROOT/missing-role-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/missing-role-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/missing-role-gemma-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-gemma-missing-roles-normal.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract table > "$ROOT/output-contract-gemma-missing-roles-table.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-gemma-missing-roles-audit.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" > "$ROOT/tensor-mapping-gate-qwen.out"
grep 'tensor-mapping-gate: qwen3-8b \[reported\]' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'gate: v0.1.0  family: qwen' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'roles: source 12/12, metadata 4/4, missing 0, ambiguous 0' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'result: pass' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'top_blocker: missing-qtype-policy-report' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'next: V010.QUANT.0' "$ROOT/tensor-mapping-gate-qwen.out"
grep 'boundary: report-only; no artifact/runtime/generation' "$ROOT/tensor-mapping-gate-qwen.out"
! grep 'tensor_naming_map:' "$ROOT/tensor-mapping-gate-qwen.out"
! grep 'runtime_claim:' "$ROOT/tensor-mapping-gate-qwen.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --output table > "$ROOT/tensor-mapping-gate-qwen-table.out"
grep 'TENSOR MAPPING GATE' "$ROOT/tensor-mapping-gate-qwen-table.out"
matches "$ROOT/tensor-mapping-gate-qwen-table.out" '^TARGET[[:space:]]{2,}FAMILY[[:space:]]{2,}GATE[[:space:]]{2,}SOURCE_ROLES[[:space:]]{2,}META_ROLES[[:space:]]{2,}MISSING[[:space:]]{2,}AMBIG[[:space:]]{2,}TOP_BLOCKER[[:space:]]{2,}STATUS[[:space:]]{2,}NEXT$'
matches "$ROOT/tensor-mapping-gate-qwen-table.out" '^qwen3-8b[[:space:]]{2,}qwen[[:space:]]{2,}v0\.1\.0[[:space:]]{2,}12/12[[:space:]]{2,}4/4[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}missing-qtype-policy-report[[:space:]]{2,}passed-for-artifact-planning[[:space:]]{2,}V010\.QUANT\.0$'
! grep 'runtime_claim:' "$ROOT/tensor-mapping-gate-qwen-table.out"
! grep 'release_ready:' "$ROOT/tensor-mapping-gate-qwen-table.out"

"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --audit > "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'tensor_mapping_gate_status: passed-for-artifact-planning' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'tensor_mapping_gate_result: pass' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'tensor_mapping_gate_target_id: qwen3-8b' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'tensor_naming_map_status: naming-map-profiled' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'tokenizer_metadata_map_status: present-report-only' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'missing_role_report_status: missing-role-report-blocked' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'expected_source_role_count: 12' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'observed_source_role_count: 12' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'expected_metadata_role_count: 4' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'observed_metadata_role_count: 4' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'missing_roles: none' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'ambiguous_roles: none' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'downstream_blockers: artifact_contract=missing qtype_policy=missing runtime_descriptor=missing graph_consumer=missing backend_residency=missing logits_runtime=missing tokenizer_runtime=missing generation_runtime=missing eval_benchmark=missing' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'next_required_rows: V010.QUANT.0' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'payload_bytes_read: false' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'artifact_emitted: false' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'runtime_descriptor_constructed: false' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'graph_consumer_fed: false' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-mapping-gate-qwen-audit.out"
grep 'release_ready: false' "$ROOT/tensor-mapping-gate-qwen-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-qwen-gate-normal.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract table > "$ROOT/output-contract-qwen-gate-table.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-qwen-gate-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" > "$ROOT/tensor-mapping-gate-gemma.out"
grep 'tensor-mapping-gate: gemma-4-12b-it \[reported\]' "$ROOT/tensor-mapping-gate-gemma.out"
grep 'gate: v0.1.0  family: gemma' "$ROOT/tensor-mapping-gate-gemma.out"
grep 'roles: source 12/12, metadata 4/4, missing 0, ambiguous 0' "$ROOT/tensor-mapping-gate-gemma.out"
grep 'next: V010.QUANT.0' "$ROOT/tensor-mapping-gate-gemma.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --output table > "$ROOT/tensor-mapping-gate-gemma-table.out"
matches "$ROOT/tensor-mapping-gate-gemma-table.out" '^gemma-4-12b-it[[:space:]]{2,}gemma[[:space:]]{2,}v0\.1\.0[[:space:]]{2,}12/12[[:space:]]{2,}4/4[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}missing-qtype-policy-report[[:space:]]{2,}passed-for-artifact-planning[[:space:]]{2,}V010\.QUANT\.0$'
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --audit > "$ROOT/tensor-mapping-gate-gemma-audit.out"
grep 'tensor_mapping_gate_status: passed-for-artifact-planning' "$ROOT/tensor-mapping-gate-gemma-audit.out"
grep 'tensor_mapping_gate_family: gemma' "$ROOT/tensor-mapping-gate-gemma-audit.out"
grep 'next_required_rows: V010.QUANT.0' "$ROOT/tensor-mapping-gate-gemma-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-gemma-gate-normal.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract table > "$ROOT/output-contract-gemma-gate-table.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --source "$MISSING_ROLE_COMPLETE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-gemma-gate-audit.out"

MISSING_ROLE_NO_K_SOURCE="${TMPDIR:-/tmp}/yvex-missing-role-no-k-test-$$"
make_missing_role_source "$MISSING_ROLE_NO_K_SOURCE" missing-attention-k
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_K_SOURCE" > "$ROOT/missing-role-qwen-no-k.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen-no-k.out"
grep 'source_roles: 11/12 present, 1 missing, 0 ambiguous' "$ROOT/missing-role-qwen-no-k.out"
grep 'missing_source: attention_k' "$ROOT/missing-role-qwen-no-k.out"
grep 'top_blocker: missing-source-role-attention-k' "$ROOT/missing-role-qwen-no-k.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_K_SOURCE" --audit > "$ROOT/missing-role-qwen-no-k-audit.out"
grep 'missing_role_attention_k_status: missing' "$ROOT/missing-role-qwen-no-k-audit.out"
grep 'missing_role_top_blocker: missing-source-role-attention-k' "$ROOT/missing-role-qwen-no-k-audit.out"
grep 'missing_role.entry.0.role: attention_k' "$ROOT/missing-role-qwen-no-k-audit.out"
grep 'missing_role.entry.0.blocker_class: source-role-missing' "$ROOT/missing-role-qwen-no-k-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_NO_K_SOURCE" > "$ROOT/tensor-mapping-gate-qwen-no-k.out"
grep 'tensor-mapping-gate: qwen3-8b \[blocked\]' "$ROOT/tensor-mapping-gate-qwen-no-k.out"
grep 'roles: source 11/12, metadata 4/4, missing 1, ambiguous 0' "$ROOT/tensor-mapping-gate-qwen-no-k.out"
grep 'missing: attention_k' "$ROOT/tensor-mapping-gate-qwen-no-k.out"
grep 'top_blocker: missing-source-role-attention-k' "$ROOT/tensor-mapping-gate-qwen-no-k.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-qwen-no-k.out"

MISSING_ROLE_NO_HEAD_SOURCE="${TMPDIR:-/tmp}/yvex-missing-role-no-head-test-$$"
make_missing_role_source "$MISSING_ROLE_NO_HEAD_SOURCE" missing-output-head
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_HEAD_SOURCE" > "$ROOT/missing-role-qwen-no-head.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen-no-head.out"
grep 'missing_source: output_head' "$ROOT/missing-role-qwen-no-head.out"
grep 'top_blocker: missing-source-role-output-head' "$ROOT/missing-role-qwen-no-head.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_HEAD_SOURCE" --audit > "$ROOT/missing-role-qwen-no-head-audit.out"
grep 'missing_role_output_head_status: missing' "$ROOT/missing-role-qwen-no-head-audit.out"
grep 'missing_role_top_blocker: missing-source-role-output-head' "$ROOT/missing-role-qwen-no-head-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_NO_HEAD_SOURCE" > "$ROOT/tensor-mapping-gate-qwen-no-head.out"
grep 'tensor-mapping-gate: qwen3-8b \[blocked\]' "$ROOT/tensor-mapping-gate-qwen-no-head.out"
grep 'missing: output_head' "$ROOT/tensor-mapping-gate-qwen-no-head.out"
grep 'top_blocker: missing-output-head-tensor' "$ROOT/tensor-mapping-gate-qwen-no-head.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-qwen-no-head.out"

MISSING_ROLE_NO_METADATA_SOURCE="${TMPDIR:-/tmp}/yvex-missing-role-no-metadata-test-$$"
make_missing_role_source "$MISSING_ROLE_NO_METADATA_SOURCE" missing-metadata
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_METADATA_SOURCE" > "$ROOT/missing-role-qwen-no-metadata.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen-no-metadata.out"
grep 'source_roles: 12/12 present, 0 missing, 0 ambiguous' "$ROOT/missing-role-qwen-no-metadata.out"
grep 'metadata_roles: 0/4 present, 4 missing, 0 ambiguous' "$ROOT/missing-role-qwen-no-metadata.out"
grep 'missing_metadata: tokenizer_metadata,config_metadata,generation_metadata,special_tokens' "$ROOT/missing-role-qwen-no-metadata.out"
grep 'top_blocker: missing-tokenizer-metadata' "$ROOT/missing-role-qwen-no-metadata.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_NO_METADATA_SOURCE" --audit > "$ROOT/missing-role-qwen-no-metadata-audit.out"
grep 'missing_role_tokenizer_metadata_status: missing' "$ROOT/missing-role-qwen-no-metadata-audit.out"
grep 'missing_role_config_metadata_status: missing' "$ROOT/missing-role-qwen-no-metadata-audit.out"
grep 'missing_role_generation_metadata_status: missing' "$ROOT/missing-role-qwen-no-metadata-audit.out"
grep 'missing_role_special_tokens_status: missing' "$ROOT/missing-role-qwen-no-metadata-audit.out"
grep 'missing_role_top_blocker: missing-tokenizer-metadata' "$ROOT/missing-role-qwen-no-metadata-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_NO_METADATA_SOURCE" > "$ROOT/tensor-mapping-gate-qwen-no-metadata.out"
grep 'tensor-mapping-gate: qwen3-8b \[blocked\]' "$ROOT/tensor-mapping-gate-qwen-no-metadata.out"
grep 'missing: tokenizer_metadata,config_metadata,generation_metadata,special_tokens' "$ROOT/tensor-mapping-gate-qwen-no-metadata.out"
grep 'top_blocker: missing-tokenizer-sidecars' "$ROOT/tensor-mapping-gate-qwen-no-metadata.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-qwen-no-metadata.out"

MISSING_ROLE_AMBIG_SOURCE="${TMPDIR:-/tmp}/yvex-missing-role-ambig-test-$$"
make_missing_role_source "$MISSING_ROLE_AMBIG_SOURCE" ambiguous-output-head
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_AMBIG_SOURCE" > "$ROOT/missing-role-qwen-ambiguous.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen-ambiguous.out"
grep 'source_roles: 11/12 present, 0 missing, 1 ambiguous' "$ROOT/missing-role-qwen-ambiguous.out"
grep 'top_blocker: ambiguous-source-role-output-head' "$ROOT/missing-role-qwen-ambiguous.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source "$MISSING_ROLE_AMBIG_SOURCE" --audit > "$ROOT/missing-role-qwen-ambiguous-audit.out"
grep 'missing_role_output_head_status: ambiguous' "$ROOT/missing-role-qwen-ambiguous-audit.out"
grep 'missing_role_top_blocker: ambiguous-source-role-output-head' "$ROOT/missing-role-qwen-ambiguous-audit.out"
grep 'missing_role.entry.0.role: output_head' "$ROOT/missing-role-qwen-ambiguous-audit.out"
grep 'missing_role.entry.0.blocker_class: source-role-ambiguous' "$ROOT/missing-role-qwen-ambiguous-audit.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --source "$MISSING_ROLE_AMBIG_SOURCE" > "$ROOT/tensor-mapping-gate-qwen-ambiguous.out"
grep 'tensor-mapping-gate: qwen3-8b \[blocked\]' "$ROOT/tensor-mapping-gate-qwen-ambiguous.out"
grep 'ambiguous: output_head' "$ROOT/tensor-mapping-gate-qwen-ambiguous.out"
grep 'top_blocker: ambiguous-output-head-tensor' "$ROOT/tensor-mapping-gate-qwen-ambiguous.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-qwen-ambiguous.out"

MISSING_ROLE_MISSING_ROOT="$ROOT/missing-role-missing-root"
yvex_test_cleanup "$MISSING_ROLE_MISSING_ROOT"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --models-root "$MISSING_ROLE_MISSING_ROOT" > "$ROOT/missing-role-qwen-missing-source.out"
grep 'missing-roles: qwen3-8b \[blocked\]' "$ROOT/missing-role-qwen-missing-source.out"
grep 'source_roles: 0/12 present, 12 missing, 0 ambiguous' "$ROOT/missing-role-qwen-missing-source.out"
grep 'metadata_roles: 0/4 present, 4 missing, 0 ambiguous' "$ROOT/missing-role-qwen-missing-source.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/missing-role-qwen-missing-source.out"
grep 'next: V010.MAP.9' "$ROOT/missing-role-qwen-missing-source.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --models-root "$MISSING_ROLE_MISSING_ROOT" > "$ROOT/tensor-mapping-gate-qwen-missing-source.out"
grep 'tensor-mapping-gate: qwen3-8b \[blocked\]' "$ROOT/tensor-mapping-gate-qwen-missing-source.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/tensor-mapping-gate-qwen-missing-source.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-qwen-missing-source.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role missing-roles --models-root "$MISSING_ROLE_MISSING_ROOT" > "$ROOT/missing-role-gemma-missing-source.out"
grep 'missing-roles: gemma-4-12b-it \[blocked\]' "$ROOT/missing-role-gemma-missing-source.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/missing-role-gemma-missing-source.out"
grep 'next: V010.MAP.9' "$ROOT/missing-role-gemma-missing-source.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --gate v0.1.0 --models-root "$MISSING_ROLE_MISSING_ROOT" > "$ROOT/tensor-mapping-gate-gemma-missing-source.out"
grep 'tensor-mapping-gate: gemma-4-12b-it \[blocked\]' "$ROOT/tensor-mapping-gate-gemma-missing-source.out"
grep 'top_blocker: missing-gemma-source-path' "$ROOT/tensor-mapping-gate-gemma-missing-source.out"
grep 'next: V010.MAP.9' "$ROOT/tensor-mapping-gate-gemma-missing-source.out"

MODEL_TARGET_QTYPE_SOURCE="$MISSING_ROLE_COMPLETE_SOURCE"

"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" > "$ROOT/qtype-policy-qwen.out"
grep 'qtype-policy: qwen3-8b \[reported\]' "$ROOT/qtype-policy-qwen.out"
grep 'family: qwen  mapping_gate: passed-for-artifact-planning' "$ROOT/qtype-policy-qwen.out"
grep 'source_dtype: F32=12 F16=0 BF16=0 other=0' "$ROOT/qtype-policy-qwen.out"
grep 'policy: artifact-planning-storage-policy' "$ROOT/qtype-policy-qwen.out"
grep 'preferred: F16' "$ROOT/qtype-policy-qwen.out"
grep 'candidates: F16,BF16,F32,Q8_0,Q2_K,IQ2_XXS' "$ROOT/qtype-policy-qwen.out"
grep 'refused: Q4_K' "$ROOT/qtype-policy-qwen.out"
grep 'top_blocker: family-quantization-plan-unimplemented' "$ROOT/qtype-policy-qwen.out"
grep 'next: not-scheduled' "$ROOT/qtype-policy-qwen.out"
grep 'boundary: report-only; no quantization/artifact/runtime' "$ROOT/qtype-policy-qwen.out"
! grep 'qtype_policy_status:' "$ROOT/qtype-policy-qwen.out"
! grep 'calibration_status:' "$ROOT/qtype-policy-qwen.out"
! grep 'runtime_claim:' "$ROOT/qtype-policy-qwen.out"

"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" --output table > "$ROOT/qtype-policy-qwen-table.out"
grep 'QTYPE POLICY' "$ROOT/qtype-policy-qwen-table.out"
matches "$ROOT/qtype-policy-qwen-table.out" '^TARGET[[:space:]]{2,}FAMILY[[:space:]]{2,}SOURCE_DTYPE[[:space:]]{2,}POLICY[[:space:]]{2,}PREFERRED[[:space:]]{2,}CANDIDATES[[:space:]]{2,}REFUSED[[:space:]]{2,}STATUS[[:space:]]{2,}NEXT$'
matches "$ROOT/qtype-policy-qwen-table.out" '^qwen3-8b[[:space:]]{2,}qwen[[:space:]]{2,}F32=12 F16=0 BF16=0 other=0[[:space:]]{2,}artifact-planning-storage-policy[[:space:]]{2,}F16[[:space:]]{2,}F16,BF16,F32,Q8_0,Q2_K,IQ2_XXS[[:space:]]{2,}Q4_K[[:space:]]{2,}policy-reported[[:space:]]{2,}not-scheduled$'
! grep 'CALIBRATION' "$ROOT/qtype-policy-qwen-table.out"
! grep 'calibration_status:' "$ROOT/qtype-policy-qwen-table.out"
! grep 'runtime_claim:' "$ROOT/qtype-policy-qwen-table.out"

"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" --audit > "$ROOT/qtype-policy-qwen-audit.out"
grep 'source_dtype_profile_status: profiled' "$ROOT/qtype-policy-qwen-audit.out"
grep 'source_dtype_counts: F32=12,F16=0,BF16=0' "$ROOT/qtype-policy-qwen-audit.out"
grep 'source_tensor_count: 12' "$ROOT/qtype-policy-qwen-audit.out"
grep 'mapping_gate_status: passed-for-artifact-planning' "$ROOT/qtype-policy-qwen-audit.out"
grep 'tensor_map_status: naming-map-profiled' "$ROOT/qtype-policy-qwen-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/qtype-policy-qwen-audit.out"
grep 'tokenizer_metadata_map_status: present-report-only' "$ROOT/qtype-policy-qwen-audit.out"
grep 'missing_role_report_status: missing-role-report-blocked' "$ROOT/qtype-policy-qwen-audit.out"
grep 'qtype_policy_basis: header-only-source-metadata+canonical-numeric-registry' "$ROOT/qtype-policy-qwen-audit.out"
grep 'qtype_policy_status: reported' "$ROOT/qtype-policy-qwen-audit.out"
grep 'numeric_capability.Q8_0: encoder=available decoder=available cpu=available cuda=available calibration=none' "$ROOT/qtype-policy-qwen-audit.out"
grep 'numeric_capability.Q2_K: encoder=available decoder=available cpu=available cuda=available calibration=optional' "$ROOT/qtype-policy-qwen-audit.out"
grep 'refusal_reasons: Q4_K:encoder-unavailable IQ2_XXS:calibration-required' "$ROOT/qtype-policy-qwen-audit.out"
grep 'artifact_identity_status: missing' "$ROOT/qtype-policy-qwen-audit.out"
grep 'runtime_descriptor_status: missing' "$ROOT/qtype-policy-qwen-audit.out"
grep 'graph_consumer_status: missing' "$ROOT/qtype-policy-qwen-audit.out"
grep 'backend_residency_status: missing' "$ROOT/qtype-policy-qwen-audit.out"
grep 'downstream_blockers: family_quantization_plan=missing artifact_emit=missing artifact_identity=missing runtime_descriptor=missing graph_consumer=missing backend_residency=missing generation_runtime=missing eval_benchmark=missing' "$ROOT/qtype-policy-qwen-audit.out"
grep 'next_required_rows: not-scheduled' "$ROOT/qtype-policy-qwen-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/qtype-policy-qwen-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/qtype-policy-qwen-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/qtype-policy-qwen-audit.out"
grep 'release_ready: false' "$ROOT/qtype-policy-qwen-audit.out"
"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-qwen-qtype-normal.out"
"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract table > "$ROOT/output-contract-qwen-qtype-table.out"
"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-qwen-qtype-audit.out"

"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" > "$ROOT/qtype-policy-gemma.out"
grep 'qtype-policy: gemma-4-12b-it \[reported\]' "$ROOT/qtype-policy-gemma.out"
grep 'family: gemma  mapping_gate: passed-for-artifact-planning' "$ROOT/qtype-policy-gemma.out"
grep 'source_dtype: F32=12 F16=0 BF16=0 other=0' "$ROOT/qtype-policy-gemma.out"
grep 'preferred: F16' "$ROOT/qtype-policy-gemma.out"
grep 'next: not-scheduled' "$ROOT/qtype-policy-gemma.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" --output table > "$ROOT/qtype-policy-gemma-table.out"
matches "$ROOT/qtype-policy-gemma-table.out" '^gemma-4-12b-it[[:space:]]{2,}gemma[[:space:]]{2,}F32=12 F16=0 BF16=0 other=0[[:space:]]{2,}artifact-planning-storage-policy[[:space:]]{2,}F16[[:space:]]{2,}F16,BF16,F32,Q8_0,Q2_K,IQ2_XXS[[:space:]]{2,}Q4_K[[:space:]]{2,}policy-reported[[:space:]]{2,}not-scheduled$'
"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" --audit > "$ROOT/qtype-policy-gemma-audit.out"
grep 'mapping_gate_status: passed-for-artifact-planning' "$ROOT/qtype-policy-gemma-audit.out"
grep 'qtype_policy_status: reported' "$ROOT/qtype-policy-gemma-audit.out"
grep 'next_required_rows: not-scheduled' "$ROOT/qtype-policy-gemma-audit.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract normal > "$ROOT/output-contract-gemma-qtype-normal.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract table > "$ROOT/output-contract-gemma-qtype-table.out"
"$YVEX_BIN" inspect target quant-policy gemma-4-12b-it --source "$MODEL_TARGET_QTYPE_SOURCE" --check-output-contract audit > "$ROOT/output-contract-gemma-qtype-audit.out"

"$YVEX_BIN" inspect target quant-policy qwen3-8b --models-root "$MISSING_ROLE_MISSING_ROOT" > "$ROOT/qtype-policy-qwen-missing-source.out"
grep 'qtype-policy: qwen3-8b \[blocked\]' "$ROOT/qtype-policy-qwen-missing-source.out"
grep 'family: qwen  mapping_gate: blocked-missing-source' "$ROOT/qtype-policy-qwen-missing-source.out"
grep 'top_blocker: missing-qwen-source-path' "$ROOT/qtype-policy-qwen-missing-source.out"
grep 'next: V010.MAP.9' "$ROOT/qtype-policy-qwen-missing-source.out"

QTYPE_MISSING_DTYPE_SOURCE="${TMPDIR:-/tmp}/yvex-qtype-policy-missing-dtype-test-$$"
yvex_test_cleanup "$QTYPE_MISSING_DTYPE_SOURCE"
cp -R "$MODEL_TARGET_QTYPE_SOURCE" "$QTYPE_MISSING_DTYPE_SOURCE"
rm -f "$QTYPE_MISSING_DTYPE_SOURCE"/*.safetensors
"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$QTYPE_MISSING_DTYPE_SOURCE" > "$ROOT/qtype-policy-qwen-missing-dtype.out"
grep 'qtype-policy: qwen3-8b \[blocked\]' "$ROOT/qtype-policy-qwen-missing-dtype.out"
grep 'source_dtype: F32=0 F16=0 BF16=0 other=0' "$ROOT/qtype-policy-qwen-missing-dtype.out"
grep 'top_blocker: missing-source-dtype-profile' "$ROOT/qtype-policy-qwen-missing-dtype.out"
grep 'next: V010.MAP.9' "$ROOT/qtype-policy-qwen-missing-dtype.out"
yvex_test_cleanup "$QTYPE_MISSING_DTYPE_SOURCE"

"$YVEX_BIN" inspect target quant-policy qwen3-8b --source "$MISSING_ROLE_NO_K_SOURCE" > "$ROOT/qtype-policy-qwen-blocked-gate.out"
grep 'qtype-policy: qwen3-8b \[blocked\]' "$ROOT/qtype-policy-qwen-blocked-gate.out"
grep 'family: qwen  mapping_gate: blocked-missing-runtime-roles' "$ROOT/qtype-policy-qwen-blocked-gate.out"
grep 'top_blocker: missing-source-role-attention-k' "$ROOT/qtype-policy-qwen-blocked-gate.out"
grep 'next: V010.MAP.9' "$ROOT/qtype-policy-qwen-blocked-gate.out"

expect_rc 2 "$YVEX_BIN" inspect target quant-policy nope > "$ROOT/qtype-policy-unknown-target.out" 2> "$ROOT/qtype-policy-unknown-target.err"
grep 'qtype-policy: nope \[unsupported\]' "$ROOT/qtype-policy-unknown-target.out"
"$YVEX_BIN" inspect target quant-policy deepseek4-v4-flash-dspark-selected-embed > "$ROOT/qtype-policy-unsupported-class.out"
grep 'qtype-policy: deepseek4-v4-flash-dspark-selected-embed \[blocked\]' "$ROOT/qtype-policy-unsupported-class.out"
grep 'top_blocker: unsupported-target-class' "$ROOT/qtype-policy-unsupported-class.out"
"$YVEX_BIN" inspect target quant-policy deepseek4-v4-flash-dspark-selected-embed-rmsnorm --role-support > "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'qtype-role-support: deepseek4-v4-flash-dspark-selected-embed-rmsnorm' "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'family: deepseek' "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'status: blocked' "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'source_dtype: selected-slice' "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'top_blocker: complete-artifact-admission-required' "$ROOT/qtype-role-support-deepseek-selected.out"
grep 'next: V010.ARTIFACT.MATERIALIZE.0' "$ROOT/qtype-role-support-deepseek-selected.out"
"$YVEX_BIN" inspect target quant-policy deepseek4-v4-flash-dspark-selected-embed-rmsnorm --role-support --audit > "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'selected_slice_evidence_only: true' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'full_family_artifact_status: missing' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'role\.[0-9][0-9]*\.role_status: selected-slice-evidence-only' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'role\.[0-9][0-9]*\.artifact_emission_blocker: complete-artifact-admission-required' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/qtype-role-support-deepseek-selected-audit.out"
"$YVEX_BIN" inspect target quant-policy glm-5.2-official-safetensors > "$ROOT/qtype-policy-unsupported-family.out"
grep 'qtype-policy: glm-5.2-official-safetensors \[unsupported\]' "$ROOT/qtype-policy-unsupported-family.out"
grep 'top_blocker: unsupported-family' "$ROOT/qtype-policy-unsupported-family.out"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy > "$ROOT/qtype-policy-missing-target.out" 2> "$ROOT/qtype-policy-missing-target.err"
grep 'requires TARGET' "$ROOT/qtype-policy-missing-target.err"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-8b --output nope > "$ROOT/qtype-policy-bad-output.out" 2> "$ROOT/qtype-policy-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/qtype-policy-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy nope --check-output-contract normal > "$ROOT/qtype-policy-contract-unknown-target.out"
grep 'status: unsupported-target' "$ROOT/qtype-policy-contract-unknown-target.out"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-8b --check-output-contract nope > "$ROOT/qtype-policy-contract-bad-mode.out"
grep 'status: unsupported-mode' "$ROOT/qtype-policy-contract-bad-mode.out"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-8b --check-output-contract \
  > "$ROOT/qtype-policy-contract-missing-mode.out" \
  2> "$ROOT/qtype-policy-contract-missing-mode.err"
grep -- '--check-output-contract requires a value' \
  "$ROOT/qtype-policy-contract-missing-mode.err"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-8b --source > "$ROOT/qtype-policy-missing-source-arg.out" 2> "$ROOT/qtype-policy-missing-source-arg.err"
grep -- '--source requires a value' "$ROOT/qtype-policy-missing-source-arg.err"
expect_rc 2 "$YVEX_BIN" inspect target quant-policy qwen3-8b --models-root > "$ROOT/qtype-policy-missing-models-root.out" 2> "$ROOT/qtype-policy-missing-models-root.err"
grep -- '--models-root requires a value' "$ROOT/qtype-policy-missing-models-root.err"

yvex_test_cleanup "$MISSING_ROLE_COMPLETE_SOURCE" "$MISSING_ROLE_NO_K_SOURCE" \
  "$MISSING_ROLE_NO_HEAD_SOURCE" "$MISSING_ROLE_NO_METADATA_SOURCE" \
  "$MISSING_ROLE_AMBIG_SOURCE"

QWEN_OUTPUT_HEAD_MISSING_SOURCE="${TMPDIR:-/tmp}/yvex-qwen-output-head-missing-test-$$"
yvex_test_cleanup "$QWEN_OUTPUT_HEAD_MISSING_SOURCE"
mkdir -p "$QWEN_OUTPUT_HEAD_MISSING_SOURCE"
python3 - "$QWEN_OUTPUT_HEAD_MISSING_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

items = [
    ("model.embed_tokens.weight", [16, 8]),
    ("model.norm.weight", [8]),
]
offset = 0
header = {}
for name, shape in items:
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * 4
    header[name] = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + nbytes],
    }
    offset += nbytes
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_OUTPUT_HEAD_MISSING_SOURCE" --audit > "$ROOT/output-head-qwen-missing-head-audit.out"
grep 'output_head_map_status: output-head-missing' "$ROOT/output-head-qwen-missing-head-audit.out"
grep 'output_head_missing_status: missing' "$ROOT/output-head-qwen-missing-head-audit.out"
grep 'top_blocker: missing-output-head-tensor' "$ROOT/output-head-qwen-missing-head-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-qwen-missing-head-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-qwen-missing-head-audit.out"
yvex_test_cleanup "$QWEN_OUTPUT_HEAD_MISSING_SOURCE"

QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE="${TMPDIR:-/tmp}/yvex-qwen-output-head-ambiguous-test-$$"
yvex_test_cleanup "$QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE"
mkdir -p "$QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE"
python3 - "$QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

items = [
    ("model.embed_tokens.weight", [16, 8]),
    ("model.norm.weight", [8]),
    ("lm_head.weight", [16, 8]),
    ("output.weight", [16, 8]),
]
offset = 0
header = {}
for name, shape in items:
    size = 1
    for dim in shape:
        size *= dim
    nbytes = size * 4
    header[name] = {
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + nbytes],
    }
    offset += nbytes
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source "$QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE" --audit > "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'output_head_map_status: output-head-ambiguous' "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'output_head_candidate_count: 2' "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'output_head_ambiguous_count: 1' "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'output_head_mapping_status: ambiguous' "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-qwen-ambiguous-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-qwen-ambiguous-audit.out"
yvex_test_cleanup "$QWEN_OUTPUT_HEAD_AMBIGUOUS_SOURCE"

QWEN_TENSOR_MAP_UNKNOWN_SOURCE="${TMPDIR:-/tmp}/yvex-qwen-tensor-map-unknown-test-$$"
yvex_test_cleanup "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE"
mkdir -p "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE"
printf '{}\n' > "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE/config.json"
printf '{}\n' > "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE/tokenizer.json"
python3 - "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.0.weird_unknown.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE" > "$ROOT/tensor-map-qwen-unknown.out"
grep 'tensor-map: qwen3-8b \[naming-map-candidate\]' "$ROOT/tensor-map-qwen-unknown.out"
grep 'roles: total=12 embedding=1 attention=4 mlp=3 norm=3 head=1 moe=0 unknown=1' "$ROOT/tensor-map-qwen-unknown.out"
"$YVEX_BIN" inspect target tensor-map qwen3-8b --source "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE" --audit > "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'tensor_map_status: naming-map-candidate' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'tensor_map_tensor_count: 13' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'tensor_map_mapped_total_count: 12' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'tensor_map_unmapped_unknown_count: 1' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'tensor_map.entry.' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'model.layers.0.weird_unknown.weight' "$ROOT/tensor-map-qwen-unknown-audit.out"
grep 'mapping_status: unmapped-unknown' "$ROOT/tensor-map-qwen-unknown-audit.out"
yvex_test_cleanup "$QWEN_TENSOR_MAP_UNKNOWN_SOURCE"
yvex_test_cleanup "$QWEN_COLLECTION_SOURCE"

BAD_RUNTIME_CLAIM='runtime_claim: support''ed'
BAD_GENERATION_READY='generation_ready: tr''ue'
BAD_BENCHMARK_MEASURED='benchmark_status: measur''ed'
BAD_RELEASE_READY='release_ready: tr''ue'
! grep "$BAD_RUNTIME_CLAIM" "$ROOT/model-class-qwen-audit.out"
! grep "$BAD_GENERATION_READY" "$ROOT/model-class-qwen-audit.out"
! grep "$BAD_BENCHMARK_MEASURED" "$ROOT/model-class-qwen-audit.out"
! grep "$BAD_RELEASE_READY" "$ROOT/model-class-qwen-audit.out"
yvex_test_cleanup "$QWEN_CLASS_SOURCE"

GEMMA_CLASS_SOURCE="${TMPDIR:-/tmp}/yvex-gemma-class-profile-test-$$"
yvex_test_cleanup "$GEMMA_CLASS_SOURCE"
mkdir -p "$GEMMA_CLASS_SOURCE"
printf '{}\n' > "$GEMMA_CLASS_SOURCE/config.json"
printf '{}\n' > "$GEMMA_CLASS_SOURCE/tokenizer.json"
python3 - "$GEMMA_CLASS_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" > "$ROOT/model-class-gemma.out"
grep 'status: metadata-profiled' "$ROOT/model-class-gemma.out"
grep 'class: gemma-source-model-class-profile' "$ROOT/model-class-gemma.out"
grep 'patterns: tensors=12 attn=4 mlp=3 norm=3 head=1 moe=0' "$ROOT/model-class-gemma.out"
grep 'top_blocker: missing-gemma-tensor-role-map' "$ROOT/model-class-gemma.out"
grep 'next: V010.MAP.8' "$ROOT/model-class-gemma.out"

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --output table > "$ROOT/model-class-gemma-table.out"
grep 'MODEL CLASS PROFILE' "$ROOT/model-class-gemma-table.out"
matches "$ROOT/model-class-gemma-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}metadata-profiled[[:space:]]{2,}12[[:space:]]{2,}4[[:space:]]{2,}3[[:space:]]{2,}3[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --audit > "$ROOT/model-class-gemma-audit.out"
grep 'model_class_profile_status: metadata-profiled' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_family: gemma' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_target_id: gemma-4-12b-it' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_name: gemma-source-model-class-profile' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_runtime_shape: dense-causal-decoder-candidate-pending-config' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_evidence_basis: header-metadata-only' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_config_status: present' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_tokenizer_status: present' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_source_metadata_status: header-only' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_tensor_count: 12' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_embedding_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_attention_q_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_attention_k_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_attention_v_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_attention_o_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_mlp_gate_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_mlp_up_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_mlp_down_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_norm_pattern_count: 3' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_output_head_pattern_count: 1' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_moe_router_pattern_count: 0' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_moe_expert_pattern_count: 0' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_other_pattern_count: 0' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_pattern_status: lexical-only' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_role_mapping_status: not-implemented' "$ROOT/model-class-gemma-audit.out"
grep 'model_class_runtime_status: unsupported' "$ROOT/model-class-gemma-audit.out"
grep 'backend_selection: deferred' "$ROOT/model-class-gemma-audit.out"
grep 'backend_pressure: cpu-cuda-baseline-planned' "$ROOT/model-class-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/model-class-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/model-class-gemma-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/model-class-gemma-audit.out"
grep 'release_ready: false' "$ROOT/model-class-gemma-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/model-class-gemma-audit.out"

GEMMA_CLASS_MODELS_ROOT="$ROOT/gemma-class-models-root"
YVEX_FAKE_HF_AUTH=1 YVEX_HF_CLI="$FAKE_HF" "$YVEX_BIN" source acquire gemma-4-12b-it --models-root "$GEMMA_CLASS_MODELS_ROOT" --auth auto --progress off > "$ROOT/gemma-class-acquire.out"
cp -R "$GEMMA_CLASS_SOURCE/." "$GEMMA_CLASS_MODELS_ROOT/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08/"
"$YVEX_BIN" inspect target class-profile gemma-4-12b-it --models-root "$GEMMA_CLASS_MODELS_ROOT" --audit > "$ROOT/model-class-gemma-models-root-audit.out"
grep 'model_class_profile_status: metadata-profiled' "$ROOT/model-class-gemma-models-root-audit.out"
matches "$ROOT/model-class-gemma-models-root-audit.out" 'source_path: .*/gemma-class-models-root/source/hf/google/gemma-4-12B-it/b8b09e34f8d2b9d1b7a51982ccb26ae2b8b9ef08$'
grep 'model_class_source_metadata_status: header-only' "$ROOT/model-class-gemma-models-root-audit.out"

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" > "$ROOT/tensor-collection-gemma.out"
grep 'tensor-collection: gemma' "$ROOT/tensor-collection-gemma.out"
grep 'target: gemma-4-12b-it' "$ROOT/tensor-collection-gemma.out"
grep 'status: collection-profiled' "$ROOT/tensor-collection-gemma.out"
grep 'stage: header-collection-inventory' "$ROOT/tensor-collection-gemma.out"
grep 'evidence: header-metadata-only' "$ROOT/tensor-collection-gemma.out"
grep 'collections: embedding=1 attention_qkvo=1 mlp_gud=1 norm=3 head=1 moe=0' "$ROOT/tensor-collection-gemma.out"
grep 'layers_observed: 1' "$ROOT/tensor-collection-gemma.out"
grep 'top_blocker: missing-gemma-tensor-role-map' "$ROOT/tensor-collection-gemma.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-collection-gemma.out"
grep 'boundary: tensor collection inventory only; no role mapping/runtime/generation' "$ROOT/tensor-collection-gemma.out"

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --output table > "$ROOT/tensor-collection-gemma-table.out"
grep 'TENSOR COLLECTION INVENTORY' "$ROOT/tensor-collection-gemma-table.out"
matches "$ROOT/tensor-collection-gemma-table.out" '^FAMILY[[:space:]]{2,}TARGET[[:space:]]{2,}STATUS[[:space:]]{2,}EMBED[[:space:]]{2,}ATTN_QKVO[[:space:]]{2,}MLP_GUD[[:space:]]{2,}NORM[[:space:]]{2,}HEAD[[:space:]]{2,}MOE[[:space:]]{2,}LAYERS[[:space:]]{2,}NEXT$'
matches "$ROOT/tensor-collection-gemma-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}collection-profiled[[:space:]]{2,}1[[:space:]]{2,}1[[:space:]]{2,}1[[:space:]]{2,}3[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}1[[:space:]]{2,}V010\.MAP\.8$'

"$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --audit > "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_status: collection-profiled' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_family: gemma' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_target_id: gemma-4-12b-it' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_stage: header-collection-inventory' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_evidence_basis: header-metadata-only' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_source_status: present' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_manifest_status: not-checked' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_config_status: present' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_tokenizer_status: present' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_tensor_count: 12' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_layer_count_observed: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_embedding_status: candidate' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_embedding_tensor_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_status: candidate' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_q_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_k_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_v_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_o_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_attention_complete_qkvo_layer_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_mlp_status: candidate' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_mlp_gate_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_mlp_up_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_mlp_down_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_mlp_complete_gud_layer_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_norm_status: candidate' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_norm_tensor_count: 3' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_output_head_status: candidate' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_output_head_tensor_count: 1' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_moe_status: not-observed' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_moe_router_count: 0' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_moe_expert_count: 0' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_tokenizer_collection_status: sidecar-observed' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_kv_runtime_state_status: runtime-state-required-not-implemented' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_validation_status: lexical-and-header-only' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_role_mapping_status: not-implemented' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_runtime_descriptor_status: not-implemented' "$ROOT/tensor-collection-gemma-audit.out"
grep 'tensor_collection_graph_consumer_status: not-implemented' "$ROOT/tensor-collection-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-collection-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-collection-gemma-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-collection-gemma-audit.out"
grep 'release_ready: false' "$ROOT/tensor-collection-gemma-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-collection-gemma-audit.out"
! grep 'generation_ready: tr''ue' "$ROOT/tensor-collection-gemma-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" > "$ROOT/tensor-map-gemma.out"
grep 'tensor-map: gemma-4-12b-it \[reported\]' "$ROOT/tensor-map-gemma.out"
grep 'family: gemma  stage: header-naming-map  evidence: header-only' "$ROOT/tensor-map-gemma.out"
grep 'roles: total=12 embedding=1 attention=4 mlp=3 norm=3 head=1 moe=0 unknown=0' "$ROOT/tensor-map-gemma.out"
grep 'layers: 1' "$ROOT/tensor-map-gemma.out"
grep 'top_blocker: missing-dense-runtime-role-validation' "$ROOT/tensor-map-gemma.out"
grep 'next: V010.MAP.8' "$ROOT/tensor-map-gemma.out"
grep 'boundary: report-only; use --audit for tensor entries' "$ROOT/tensor-map-gemma.out"
! grep 'tensor_map.entry.' "$ROOT/tensor-map-gemma.out"
! grep 'runtime_claim:' "$ROOT/tensor-map-gemma.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --output table > "$ROOT/tensor-map-gemma-table.out"
grep 'TENSOR NAMING MAP' "$ROOT/tensor-map-gemma-table.out"
matches "$ROOT/tensor-map-gemma-table.out" '^FAMILY[[:space:]]{2,}TARGET[[:space:]]{2,}STATUS[[:space:]]{2,}TOTAL[[:space:]]{2,}EMBED[[:space:]]{2,}ATTN[[:space:]]{2,}MLP[[:space:]]{2,}NORM[[:space:]]{2,}HEAD[[:space:]]{2,}MOE[[:space:]]{2,}UNKNOWN[[:space:]]{2,}LAYERS[[:space:]]{2,}NEXT$'
matches "$ROOT/tensor-map-gemma-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}naming-map-profiled[[:space:]]{2,}12[[:space:]]{2,}1[[:space:]]{2,}4[[:space:]]{2,}3[[:space:]]{2,}3[[:space:]]{2,}1[[:space:]]{2,}0[[:space:]]{2,}0[[:space:]]{2,}1[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --audit > "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_status: naming-map-profiled' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_family: gemma' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_target_id: gemma-4-12b-it' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_stage: header-naming-map' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_evidence_basis: header-metadata-only' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_source_status: present' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_config_status: present' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_tokenizer_status: present' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_tensor_count: 12' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_mapped_total_count: 12' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_unmapped_unknown_count: 0' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_ambiguous_count: 0' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_layer_count_observed: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_embedding_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_attention_count: 4' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_attention_q_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_attention_k_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_attention_v_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_attention_o_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_mlp_count: 3' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_mlp_gate_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_mlp_up_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_mlp_down_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_norm_count: 3' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_output_head_count: 1' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_moe_router_count: 0' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_moe_expert_count: 0' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_validation_status: lexical-and-header-only' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_canonical_role_status: mapped-candidates' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_runtime_role_coverage_status: report-only' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_artifact_contract_status: not-implemented' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_runtime_descriptor_status: not-implemented' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map_graph_consumer_status: not-implemented' "$ROOT/tensor-map-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-gemma-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/tensor-map-gemma-audit.out"
grep 'release_ready: false' "$ROOT/tensor-map-gemma-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/tensor-map-gemma-audit.out"
grep 'tensor_map.entry.' "$ROOT/tensor-map-gemma-audit.out"
grep 'model.embed_tokens.weight -> model.embedding.token.weight' "$ROOT/tensor-map-gemma-audit.out"
grep 'model.layers.0.self_attn.q_proj.weight -> model.layers.0.attention.q_proj.weight' "$ROOT/tensor-map-gemma-audit.out"
grep 'model.layers.0.input_layernorm.weight -> model.layers.0.attention.norm.weight' "$ROOT/tensor-map-gemma-audit.out"
grep 'lm_head.weight -> model.output_head.weight' "$ROOT/tensor-map-gemma-audit.out"
! grep 'generation_ready: tr''ue' "$ROOT/tensor-map-gemma-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --check-output-contract normal > "$ROOT/output-contract-gemma-tensor-map-normal.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --check-output-contract table > "$ROOT/output-contract-gemma-tensor-map-table.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_CLASS_SOURCE" --check-output-contract audit > "$ROOT/output-contract-gemma-tensor-map-audit.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" > "$ROOT/output-head-gemma.out"
grep 'output-head-map: gemma-4-12b-it \[reported\]' "$ROOT/output-head-gemma.out"
grep 'family: gemma  evidence: header-only' "$ROOT/output-head-gemma.out"
grep 'head: model.output_head.weight  final_norm: model.final_norm.weight  embedding: model.embedding.token.weight  tie: separate-output-head-candidate' "$ROOT/output-head-gemma.out"
grep 'shape: compatible-same-shape' "$ROOT/output-head-gemma.out"
grep 'top_blocker: missing-output-head-runtime-consumer' "$ROOT/output-head-gemma.out"
grep 'next: V010.MAP.8' "$ROOT/output-head-gemma.out"
grep 'boundary: mapping only; no logits/runtime/generation' "$ROOT/output-head-gemma.out"
! grep 'output_head_map_' "$ROOT/output-head-gemma.out"
! grep 'native_output_head:' "$ROOT/output-head-gemma.out"
! grep 'runtime_claim:' "$ROOT/output-head-gemma.out"

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" --output table > "$ROOT/output-head-gemma-table.out"
grep 'OUTPUT HEAD TENSOR MAP' "$ROOT/output-head-gemma-table.out"
matches "$ROOT/output-head-gemma-table.out" '^gemma[[:space:]]{2,}gemma-4-12b-it[[:space:]]{2,}output-head-profiled[[:space:]]{2,}yes[[:space:]]{2,}yes[[:space:]]{2,}yes[[:space:]]{2,}separate-output-head-candidate[[:space:]]{2,}compatible-same-shape[[:space:]]{2,}V010.MAP.8$'

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" --audit > "$ROOT/output-head-gemma-audit.out"
grep 'output_head_map_status: output-head-profiled' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_map_family: gemma' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_map_target_id: gemma-4-12b-it' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_native_name: lm_head.weight' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_canonical_role: model.output_head.weight' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_mapping_status: mapped-candidate' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_candidate_count: 1' "$ROOT/output-head-gemma-audit.out"
grep 'embedding_canonical_role: model.embedding.token.weight' "$ROOT/output-head-gemma-audit.out"
grep 'final_norm_canonical_role: model.final_norm.weight' "$ROOT/output-head-gemma-audit.out"
grep 'tie_policy_status: separate-output-head-candidate' "$ROOT/output-head-gemma-audit.out"
grep 'shape_relation_status: compatible-same-shape' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_runtime_consumer_status: target-runtime-owned' "$ROOT/output-head-gemma-audit.out"
grep 'output_head_logits_status: target-capability-dependent' "$ROOT/output-head-gemma-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/output-head-gemma-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/output-head-gemma-audit.out"
grep 'benchmark_status: not-measured' "$ROOT/output-head-gemma-audit.out"
grep 'release_ready: false' "$ROOT/output-head-gemma-audit.out"
grep 'next_required_rows: V010.MAP.8' "$ROOT/output-head-gemma-audit.out"
grep 'output_head.entry.output.native_name: lm_head.weight' "$ROOT/output-head-gemma-audit.out"
grep 'output_head.entry.output.canonical_role: model.output_head.weight' "$ROOT/output-head-gemma-audit.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" --check-output-contract normal > "$ROOT/output-contract-gemma-output-head-normal.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" --check-output-contract table > "$ROOT/output-contract-gemma-output-head-table.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role output-head --source "$GEMMA_CLASS_SOURCE" --check-output-contract audit > "$ROOT/output-contract-gemma-output-head-audit.out"

GEMMA_TENSOR_MAP_UNKNOWN_SOURCE="${TMPDIR:-/tmp}/yvex-gemma-tensor-map-unknown-test-$$"
yvex_test_cleanup "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE"
mkdir -p "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE"
printf '{}\n' > "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE/config.json"
printf '{}\n' > "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE/tokenizer.json"
python3 - "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.norm.weight",
    "lm_head.weight",
    "model.layers.0.weird_unknown.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE" > "$ROOT/tensor-map-gemma-unknown.out"
grep 'tensor-map: gemma-4-12b-it \[naming-map-candidate\]' "$ROOT/tensor-map-gemma-unknown.out"
grep 'roles: total=12 embedding=1 attention=4 mlp=3 norm=3 head=1 moe=0 unknown=1' "$ROOT/tensor-map-gemma-unknown.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE" --audit > "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'tensor_map_status: naming-map-candidate' "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'tensor_map_tensor_count: 13' "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'tensor_map_mapped_total_count: 12' "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'tensor_map_unmapped_unknown_count: 1' "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'model.layers.0.weird_unknown.weight' "$ROOT/tensor-map-gemma-unknown-audit.out"
grep 'mapping_status: unmapped-unknown' "$ROOT/tensor-map-gemma-unknown-audit.out"
yvex_test_cleanup "$GEMMA_TENSOR_MAP_UNKNOWN_SOURCE"

GEMMA_TENSOR_MAP_NORM_SOURCE="${TMPDIR:-/tmp}/yvex-gemma-tensor-map-norm-test-$$"
yvex_test_cleanup "$GEMMA_TENSOR_MAP_NORM_SOURCE"
mkdir -p "$GEMMA_TENSOR_MAP_NORM_SOURCE"
python3 - "$GEMMA_TENSOR_MAP_NORM_SOURCE/model.safetensors" <<'PY'
import json
import struct
import sys

names = [
    "model.layers.0.pre_feedforward_layernorm.weight",
    "model.layers.0.post_feedforward_layernorm.weight",
]
offset = 0
header = {}
for name in names:
    header[name] = {
        "dtype": "F32",
        "shape": [2, 2],
        "data_offsets": [offset, offset + 16],
    }
    offset += 16
blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(b"x" * offset)
PY

"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_TENSOR_MAP_NORM_SOURCE" > "$ROOT/tensor-map-gemma-norm.out"
grep 'tensor-map: gemma-4-12b-it \[blocked\]' "$ROOT/tensor-map-gemma-norm.out"
grep 'roles: total=2 embedding=0 attention=0 mlp=0 norm=2 head=0 moe=0 unknown=0' "$ROOT/tensor-map-gemma-norm.out"
"$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source "$GEMMA_TENSOR_MAP_NORM_SOURCE" --audit > "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'tensor_map_norm_count: 2' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'model.layers.0.pre_feedforward_layernorm.weight -> model.layers.0.mlp.norm.weight' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'model.layers.0.post_feedforward_layernorm.weight -> model.layers.0.mlp.norm.weight' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'tensor_map_runtime_role_coverage_status: report-only' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'runtime_claim: unsupported' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'generation: unsupported-full-model' "$ROOT/tensor-map-gemma-norm-audit.out"
grep 'release_ready: false' "$ROOT/tensor-map-gemma-norm-audit.out"
yvex_test_cleanup "$GEMMA_TENSOR_MAP_NORM_SOURCE"

! grep "$BAD_RUNTIME_CLAIM" "$ROOT/model-class-gemma-audit.out"
! grep "$BAD_GENERATION_READY" "$ROOT/model-class-gemma-audit.out"
! grep "$BAD_BENCHMARK_MEASURED" "$ROOT/model-class-gemma-audit.out"
! grep "$BAD_RELEASE_READY" "$ROOT/model-class-gemma-audit.out"
yvex_test_cleanup "$GEMMA_CLASS_SOURCE"

expect_rc 2 "$YVEX_BIN" inspect target class-profile > "$ROOT/model-class-missing-target.out" 2> "$ROOT/model-class-missing-target.err"
grep 'requires TARGET' "$ROOT/model-class-missing-target.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile nope > "$ROOT/model-class-bad-target.out" 2> "$ROOT/model-class-bad-target.err"
grep 'unsupported target: nope' "$ROOT/model-class-bad-target.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile deepseek4-v4-flash > "$ROOT/model-class-retired-deepseek.out" 2> "$ROOT/model-class-retired-deepseek.err"
grep 'unsupported target: deepseek4-v4-flash; use deepseek4-v4-flash-dspark' "$ROOT/model-class-retired-deepseek.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile qwen-metal-portability > "$ROOT/model-class-old-target.out" 2> "$ROOT/model-class-old-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/model-class-old-target.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile gemma-dense-portability > "$ROOT/model-class-old-gemma-target.out" 2> "$ROOT/model-class-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/model-class-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile qwen3-8b --output nope > "$ROOT/model-class-bad-output.out" 2> "$ROOT/model-class-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/model-class-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile qwen3-8b --source > "$ROOT/model-class-missing-source.out" 2> "$ROOT/model-class-missing-source.err"
grep -- '--source requires a value' "$ROOT/model-class-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile gemma-4-12b-it --output nope > "$ROOT/model-class-gemma-bad-output.out" 2> "$ROOT/model-class-gemma-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/model-class-gemma-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target class-profile gemma-4-12b-it --source > "$ROOT/model-class-gemma-missing-source.out" 2> "$ROOT/model-class-gemma-missing-source.err"
grep -- '--source requires a value' "$ROOT/model-class-gemma-missing-source.err"

expect_rc 2 "$YVEX_BIN" inspect target tensor-collection > "$ROOT/tensor-collection-missing-target.out" 2> "$ROOT/tensor-collection-missing-target.err"
grep 'requires TARGET' "$ROOT/tensor-collection-missing-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection nope > "$ROOT/tensor-collection-bad-target.out" 2> "$ROOT/tensor-collection-bad-target.err"
grep 'unsupported target: nope' "$ROOT/tensor-collection-bad-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection qwen-metal-portability > "$ROOT/tensor-collection-old-target.out" 2> "$ROOT/tensor-collection-old-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/tensor-collection-old-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection qwen3-8b --output nope > "$ROOT/tensor-collection-bad-output.out" 2> "$ROOT/tensor-collection-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tensor-collection-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection qwen3-8b --source > "$ROOT/tensor-collection-missing-source.out" 2> "$ROOT/tensor-collection-missing-source.err"
grep -- '--source requires a value' "$ROOT/tensor-collection-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection qwen3-8b --models-root > "$ROOT/tensor-collection-missing-models-root.out" 2> "$ROOT/tensor-collection-missing-models-root.err"
grep -- '--models-root requires a value' "$ROOT/tensor-collection-missing-models-root.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection gemma-dense-portability > "$ROOT/tensor-collection-old-gemma-target.out" 2> "$ROOT/tensor-collection-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/tensor-collection-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --output nope > "$ROOT/tensor-collection-gemma-bad-output.out" 2> "$ROOT/tensor-collection-gemma-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tensor-collection-gemma-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-collection gemma-4-12b-it --source > "$ROOT/tensor-collection-gemma-missing-source.out" 2> "$ROOT/tensor-collection-gemma-missing-source.err"
grep -- '--source requires a value' "$ROOT/tensor-collection-gemma-missing-source.err"

expect_rc 2 "$YVEX_BIN" inspect target tensor-map > "$ROOT/tensor-map-missing-target.out" 2> "$ROOT/tensor-map-missing-target.err"
grep 'requires TARGET' "$ROOT/tensor-map-missing-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map nope > "$ROOT/tensor-map-bad-target.out" 2> "$ROOT/tensor-map-bad-target.err"
grep 'unsupported target: nope' "$ROOT/tensor-map-bad-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen-metal-portability > "$ROOT/tensor-map-old-target.out" 2> "$ROOT/tensor-map-old-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/tensor-map-old-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --output nope > "$ROOT/tensor-map-bad-output.out" 2> "$ROOT/tensor-map-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tensor-map-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map nope --check-output-contract normal > "$ROOT/tensor-map-contract-unknown-target.out"
grep 'status: unsupported-target' "$ROOT/tensor-map-contract-unknown-target.out"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --check-output-contract nope > "$ROOT/tensor-map-contract-bad-mode.out"
grep 'status: unsupported-mode' "$ROOT/tensor-map-contract-bad-mode.out"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --check-output-contract \
  > "$ROOT/tensor-map-contract-missing-mode.out" \
  2> "$ROOT/tensor-map-contract-missing-mode.err"
grep -- '--check-output-contract requires a value' \
  "$ROOT/tensor-map-contract-missing-mode.err"
for output_contract_report in "$ROOT"/output-contract-*.out; do
  test -f "$output_contract_report"
  assert_output_contract_pass "$output_contract_report"
done
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --source > "$ROOT/tensor-map-missing-source.out" 2> "$ROOT/tensor-map-missing-source.err"
grep -- '--source requires a value' "$ROOT/tensor-map-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --models-root > "$ROOT/tensor-map-missing-models-root.out" 2> "$ROOT/tensor-map-missing-models-root.err"
grep -- '--models-root requires a value' "$ROOT/tensor-map-missing-models-root.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --output nope > "$ROOT/tensor-map-gemma-bad-output.out" 2> "$ROOT/tensor-map-gemma-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tensor-map-gemma-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --source > "$ROOT/tensor-map-gemma-missing-source.out" 2> "$ROOT/tensor-map-gemma-missing-source.err"
grep -- '--source requires a value' "$ROOT/tensor-map-gemma-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-dense-portability > "$ROOT/tensor-map-old-gemma-target.out" 2> "$ROOT/tensor-map-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/tensor-map-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role > "$ROOT/output-head-missing-role.out" 2> "$ROOT/output-head-missing-role.err"
grep -- '--role requires a value' "$ROOT/output-head-missing-role.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role nope > "$ROOT/output-head-qwen-bad-role.out" 2> "$ROOT/output-head-qwen-bad-role.err"
grep 'unsupported role: nope' "$ROOT/output-head-qwen-bad-role.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-4-12b-it --role nope > "$ROOT/output-head-gemma-bad-role.out" 2> "$ROOT/output-head-gemma-bad-role.err"
grep 'unsupported role: nope' "$ROOT/output-head-gemma-bad-role.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen-metal-portability --role output-head > "$ROOT/output-head-old-qwen-target.out" 2> "$ROOT/output-head-old-qwen-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/output-head-old-qwen-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-dense-portability --role output-head > "$ROOT/output-head-old-gemma-target.out" 2> "$ROOT/output-head-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/output-head-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --output nope > "$ROOT/output-head-bad-output.out" 2> "$ROOT/output-head-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/output-head-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role output-head --source > "$ROOT/output-head-missing-source.out" 2> "$ROOT/output-head-missing-source.err"
grep -- '--source requires a value' "$ROOT/output-head-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen-metal-portability --role tokenizer > "$ROOT/tokenizer-old-qwen-target.out" 2> "$ROOT/tokenizer-old-qwen-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/tokenizer-old-qwen-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-dense-portability --role tokenizer > "$ROOT/tokenizer-old-gemma-target.out" 2> "$ROOT/tokenizer-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/tokenizer-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --output nope > "$ROOT/tokenizer-bad-output.out" 2> "$ROOT/tokenizer-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tokenizer-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role tokenizer --source > "$ROOT/tokenizer-missing-source.out" 2> "$ROOT/tokenizer-missing-source.err"
grep -- '--source requires a value' "$ROOT/tokenizer-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --output nope > "$ROOT/missing-role-bad-output.out" 2> "$ROOT/missing-role-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/missing-role-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --source > "$ROOT/missing-role-missing-source.out" 2> "$ROOT/missing-role-missing-source.err"
grep -- '--source requires a value' "$ROOT/missing-role-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen-metal-portability --role missing-roles > "$ROOT/missing-role-old-qwen-target.out" 2> "$ROOT/missing-role-old-qwen-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/missing-role-old-qwen-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map gemma-dense-portability --role missing-roles > "$ROOT/missing-role-old-gemma-target.out" 2> "$ROOT/missing-role-old-gemma-target.err"
grep 'unsupported target: gemma-dense-portability' "$ROOT/missing-role-old-gemma-target.err"
expect_rc 2 "$YVEX_BIN" inspect target missing-roles > "$ROOT/missing-roles-direct-missing-target.out" 2> "$ROOT/missing-roles-direct-missing-target.err"
grep 'requires TARGET' "$ROOT/missing-roles-direct-missing-target.err"
expect_rc 2 "$YVEX_BIN" inspect target missing-roles qwen3-8b --output nope > "$ROOT/missing-roles-direct-bad-output.out" 2> "$ROOT/missing-roles-direct-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/missing-roles-direct-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target missing-roles qwen3-8b --models-root > "$ROOT/missing-roles-direct-missing-root.out" 2> "$ROOT/missing-roles-direct-missing-root.err"
grep -- '--models-root requires a value' "$ROOT/missing-roles-direct-missing-root.err"
expect_rc 2 "$YVEX_BIN" inspect target missing-roles qwen3-8b --source > "$ROOT/missing-roles-direct-missing-source.out" 2> "$ROOT/missing-roles-direct-missing-source.err"
grep -- '--source requires a value' "$ROOT/missing-roles-direct-missing-source.err"
expect_rc 2 "$YVEX_BIN" inspect target missing-roles qwen-metal-portability > "$ROOT/missing-roles-direct-old-qwen-target.out" 2> "$ROOT/missing-roles-direct-old-qwen-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/missing-roles-direct-old-qwen-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --gate > "$ROOT/tensor-mapping-gate-missing-value.out" 2> "$ROOT/tensor-mapping-gate-missing-value.err"
grep -- '--gate requires a value' "$ROOT/tensor-mapping-gate-missing-value.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v9.9.9 > "$ROOT/tensor-mapping-gate-bad-release.out" 2> "$ROOT/tensor-mapping-gate-bad-release.err"
grep 'unsupported release: v9.9.9' "$ROOT/tensor-mapping-gate-bad-release.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --gate v0.1.0 --output nope > "$ROOT/tensor-mapping-gate-bad-output.out" 2> "$ROOT/tensor-mapping-gate-bad-output.err"
grep 'unsupported output mode: nope' "$ROOT/tensor-mapping-gate-bad-output.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen-metal-portability --gate v0.1.0 > "$ROOT/tensor-mapping-gate-old-qwen-target.out" 2> "$ROOT/tensor-mapping-gate-old-qwen-target.err"
grep 'unsupported target: qwen-metal-portability' "$ROOT/tensor-mapping-gate-old-qwen-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map nope --gate v0.1.0 > "$ROOT/tensor-mapping-gate-unknown-target.out" 2> "$ROOT/tensor-mapping-gate-unknown-target.err"
grep 'unsupported target: nope' "$ROOT/tensor-mapping-gate-unknown-target.err"
expect_rc 2 "$YVEX_BIN" inspect target tensor-map qwen3-8b --role missing-roles --gate v0.1.0 > "$ROOT/tensor-mapping-gate-role-conflict.out" 2> "$ROOT/tensor-mapping-gate-role-conflict.err"
grep 'gate cannot be combined with --role' "$ROOT/tensor-mapping-gate-role-conflict.err"

"$YVEX_BIN" inspect target decision --help > "$ROOT/model-target-decision-help.out"
grep 'usage: yvex inspect target \[arguments ...\] \[options\]' \
  "$ROOT/model-target-decision-help.out"
grep -- '--release' "$ROOT/model-target-decision-help.out"

"$YVEX_BIN" inspect target decision --release v0.1.0 > "$ROOT/model-target-decision-normal.out"
grep 'report: target-decision' "$ROOT/model-target-decision-normal.out"
grep 'status: target-selected-mapping-specified' "$ROOT/model-target-decision-normal.out"
grep 'selected: deepseek4-v4-flash-dspark' "$ROOT/model-target-decision-normal.out"
grep 'top_blocker: source payload trust' "$ROOT/model-target-decision-normal.out"
grep 'next: V010.SOURCE.PAYLOAD.STREAM.0' "$ROOT/model-target-decision-normal.out"
grep 'boundary: release target selected; artifact/runtime/generation unsupported; benchmark not measured' "$ROOT/model-target-decision-normal.out"
! grep 'deepseek4-v4-flash-dspark-selected' "$ROOT/model-target-decision-normal.out"

"$YVEX_BIN" inspect target decision --release v0.1.0 --output table > "$ROOT/model-target-decision-table.out"
matches "$ROOT/model-target-decision-table.out" '^REPORT[[:space:]]{2,}STATUS[[:space:]]{2,}SELECTED[[:space:]]{2,}ELIGIBLE[[:space:]]{2,}NEXT$'
matches "$ROOT/model-target-decision-table.out" '^target-decision[[:space:]]{2,}selected-mapping-specified[[:space:]]{2,}deepseek4-v4-flash-dspark[[:space:]]{2,}0[[:space:]]{2,}V010\.SOURCE\.PAYLOAD\.STREAM\.0$'

"$YVEX_BIN" inspect target decision --release v0.1.0 --output json > "$ROOT/model-target-decision-json.out"
jq -e '.selected_target_id == "deepseek4-v4-flash-dspark" and .upstream_repository == "deepseek-ai/DeepSeek-V4-Flash-DSpark" and .source_verification == "complete" and .architecture_ir == "complete" and .tensor_coverage == "complete" and .gguf_mapping == "complete" and .artifact_status == "not-produced" and .runtime == "unsupported" and .generation == "unsupported" and .next == "V010.SOURCE.PAYLOAD.STREAM.0"' "$ROOT/model-target-decision-json.out" >/dev/null

"$YVEX_BIN" inspect target decision --release v0.1.0 --output nope > "$ROOT/model-target-decision-bad-output.out" 2> "$ROOT/model-target-decision-bad-output.err" && exit 1 || true
grep 'model-target decision: unsupported output mode: nope' "$ROOT/model-target-decision-bad-output.err"

"$YVEX_BIN" inspect target decision --release v0.1.0 --audit --include-candidates --include-pressure-targets --include-blockers --include-critical-path --include-next > "$ROOT/model-target-decision.out"
grep 'target_decision: v0.1.0' "$ROOT/model-target-decision.out"
grep 'status: target-selected-mapping-specified' "$ROOT/model-target-decision.out"
grep 'decision_state: selected' "$ROOT/model-target-decision.out"
grep 'selected_target_id: deepseek4-v4-flash-dspark' "$ROOT/model-target-decision.out"
grep 'upstream_repository: deepseek-ai/DeepSeek-V4-Flash-DSpark' "$ROOT/model-target-decision.out"
grep 'source_verification_status: complete' "$ROOT/model-target-decision.out"
grep 'architecture_ir_status: complete' "$ROOT/model-target-decision.out"
grep 'tensor_coverage_status: complete' "$ROOT/model-target-decision.out"
grep 'gguf_mapping_status: complete' "$ROOT/model-target-decision.out"
grep 'full_runtime_candidate_status: unsupported' "$ROOT/model-target-decision.out"
grep 'selected_runtime_slice_eligible: false' "$ROOT/model-target-decision.out"
grep 'source_only_eligible: false' "$ROOT/model-target-decision.out"
grep 'external_reference_eligible: false' "$ROOT/model-target-decision.out"
grep 'runtime_claim: unsupported' "$ROOT/model-target-decision.out"
grep 'generation: unsupported-full-model' "$ROOT/model-target-decision.out"
grep 'benchmark_status: not-measured' "$ROOT/model-target-decision.out"
grep 'release_ready: false' "$ROOT/model-target-decision.out"
grep 'candidate.0.id: deepseek4-v4-flash-dspark' "$ROOT/model-target-decision.out"
grep 'candidate.0.class: release-source-target' "$ROOT/model-target-decision.out"
grep 'candidate.0.status: selected-mapping-specified' "$ROOT/model-target-decision.out"
grep 'qwen_engineering_scope: preserved-non-release' "$ROOT/model-target-decision.out"
grep 'gemma_engineering_scope: preserved-non-release' "$ROOT/model-target-decision.out"
grep 'selected_slice_scope: bounded-evidence-only' "$ROOT/model-target-decision.out"
grep 'next_required_rows: V010.SOURCE.PAYLOAD.STREAM.0' "$ROOT/model-target-decision.out"
! grep 'candidate.*deepseek4-v4-flash-dspark-selected' "$ROOT/model-target-decision.out"

"$YVEX_BIN" inspect target decision --release v0.1.0 --audit --candidate deepseek4-v4-flash-dspark --include-blockers --include-next > "$ROOT/model-target-decision-deepseek.out"
grep 'candidate_count: 1' "$ROOT/model-target-decision-deepseek.out"
grep 'candidate.0.id: deepseek4-v4-flash-dspark' "$ROOT/model-target-decision-deepseek.out"
grep 'candidate.0.status: selected-mapping-specified' "$ROOT/model-target-decision-deepseek.out"
grep 'generation: unsupported-full-model' "$ROOT/model-target-decision-deepseek.out"

expect_rc 2 "$YVEX_BIN" inspect target decision --release v0.1.0 --audit --candidate deepseek4-v4-flash-dspark-selected-embed-rmsnorm --include-blockers --include-next > "$ROOT/model-target-decision-rmsnorm.out" 2> "$ROOT/model-target-decision-rmsnorm.err"
grep 'status: missing-candidate' "$ROOT/model-target-decision-rmsnorm.out"

expect_rc 2 "$YVEX_BIN" inspect target decision --release v0.1.0 --audit --candidate glm-5.2-official-safetensors --include-blockers --include-next > "$ROOT/model-target-decision-glm.out" 2> "$ROOT/model-target-decision-glm.err"
grep 'status: missing-candidate' "$ROOT/model-target-decision-glm.out"

"$YVEX_BIN" inspect target decision --release v0.1.0 --candidate missing-target --include-blockers > "$ROOT/model-target-decision-missing.out" 2> "$ROOT/model-target-decision-missing.err" && exit 1 || true
grep 'status: missing-candidate' "$ROOT/model-target-decision-missing.out"
grep 'candidate_requested: missing-target' "$ROOT/model-target-decision-missing.out"
grep 'runtime_claim: unsupported' "$ROOT/model-target-decision-missing.out"

"$YVEX_BIN" inspect target decision --release v9.9.9 > "$ROOT/model-target-decision-unsupported-release.out" 2> "$ROOT/model-target-decision-unsupported-release.err" && exit 1 || true
grep 'target_decision: v9.9.9' "$ROOT/model-target-decision-unsupported-release.out"
grep 'status: unsupported-release' "$ROOT/model-target-decision-unsupported-release.out"
grep 'runtime_claim: unsupported' "$ROOT/model-target-decision-unsupported-release.out"
grep 'generation: unsupported-full-model' "$ROOT/model-target-decision-unsupported-release.out"
grep 'benchmark_status: not-measured' "$ROOT/model-target-decision-unsupported-release.out"
grep 'release_ready: false' "$ROOT/model-target-decision-unsupported-release.out"
