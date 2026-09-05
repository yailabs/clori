#!/bin/sh
# Qualifies the porcelain model workflow without downloading production weights.
set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
ROOT=${YVEX_TEST_OUT_DIR:-build/tests/cli-model-workflow}
case "$ROOT" in
  /*) ;;
  *) ROOT="$PWD/$ROOT" ;;
esac
MODELS_ROOT="$ROOT/models"
REGISTRY="$ROOT/models.local.json"
RUNTIME="$ROOT/runtime"
FAKE_HF="$PWD/tests/fixtures/bin/fake-hf"

finish()
{
    status=$?
    trap - EXIT HUP INT TERM
    yvex_test_cleanup_preserving_status "$status" "$ROOT"
}
trap finish EXIT HUP INT TERM

yvex_test_cleanup "$ROOT"
mkdir -p "$MODELS_ROOT" "$RUNTIME" "$ROOT/input" "$ROOT/export"
chmod 0700 "$RUNTIME"
printf '{"schema":"yvex.models.local.v6","models":[]}\n' >"$REGISTRY"
export XDG_CONFIG_HOME="$ROOT/config"
export XDG_DATA_HOME="$ROOT/data"
export XDG_RUNTIME_DIR="$RUNTIME"
export YVEX_DATA_DIR="$ROOT/yvex-data"
export YVEX_MODELS_REGISTRY="$REGISTRY"
export YVEX_HF_CLI="$FAKE_HF"
export YVEX_FAKE_HF_LOG="$ROOT/fake-hf.log"
mkdir -p "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$YVEX_DATA_DIR"

"$YVEX_BIN" help model list >"$ROOT/help-list.out"
"$YVEX_BIN" help model show >"$ROOT/help-show.out"
"$YVEX_BIN" help model search >"$ROOT/help-search.out"
"$YVEX_BIN" help model pull >"$ROOT/help-pull.out"
grep -F -- '--wide' "$ROOT/help-list.out" >/dev/null
! grep -E -- '--audit|--output' "$ROOT/help-list.out" >/dev/null
! grep -E -- '--all|--wide|--audit|--output' "$ROOT/help-show.out" >/dev/null
! grep -E -- '--interactive|--cli|--audit|--output' "$ROOT/help-search.out" >/dev/null
grep -F -- '--provider local|hf|huggingface' "$ROOT/help-search.out" >/dev/null
grep -F -- '--page N' "$ROOT/help-search.out" >/dev/null
grep -F -- 'range 1..20' "$ROOT/help-search.out" >/dev/null
grep -F -- '--all' "$ROOT/help-search.out" | grep -F -- \
    'conflicts --page, --limit' >/dev/null
grep -F -- '--format safetensors|gguf' "$ROOT/help-pull.out" >/dev/null
grep -F -- '--models-root PATH' "$ROOT/help-pull.out" >/dev/null
grep -F -- '--include TEXT' "$ROOT/help-pull.out" | grep -F -- \
    'repeatable' >/dev/null
grep -F -- '--quant NAME' "$ROOT/help-pull.out" | grep -F -- \
    'requires --prepare' >/dev/null
grep -F -- '--reference' "$ROOT/help-pull.out" | grep -F -- \
    'conflicts --managed, --resume' >/dev/null
grep -F -- '--verbose' "$ROOT/help-pull.out" | grep -F -- \
    'conflicts --json' >/dev/null

expect_rc()
{
    expected=$1
    shift
    set +e
    "$@"
    actual=$?
    set -e
    test "$actual" -eq "$expected"
}

contains()
{
    grep -F -- "$2" "$1" >/dev/null
}

# Registry metadata drives both discovery and early validation.  These
# invalid combinations must fail before provider, filesystem, or runtime work.
expect_rc 2 "$YVEX_BIN" model search tiny --all --page 2 \
    >"$ROOT/search-conflict.out" 2>"$ROOT/search-conflict.err"
contains "$ROOT/search-conflict.err" '--all conflicts with --page'
expect_rc 2 "$YVEX_BIN" model pull ssh://example/model --stream \
    >"$ROOT/stream-dependency.out" 2>"$ROOT/stream-dependency.err"
contains "$ROOT/stream-dependency.err" '--stream requires --prepare'
expect_rc 2 "$YVEX_BIN" model pull ssh://example/model --format unknown \
    >"$ROOT/format-enum.out" 2>"$ROOT/format-enum.err"
contains "$ROOT/format-enum.err" 'invalid value for --format: unknown'

# Local paths and file URIs use one deterministic pull grammar.  Automation
# must choose managed storage or an explicit external reference.
printf 'tiny external gguf\n' >"$ROOT/input/tiny-external.gguf"
expect_rc 2 "$YVEX_BIN" model pull "$ROOT/input/tiny-external.gguf" \
    --managed --prepare --json --models-root "$MODELS_ROOT" \
    >"$ROOT/pull-prepare-json.out" 2>"$ROOT/pull-prepare-json.err"
contains "$ROOT/pull-prepare-json.err" '--prepare conflicts with --json'

"$YVEX_BIN" model pull "$ROOT/input/tiny-external.gguf" --managed \
    --prepare --dry-run --name dry-run-local --family qwen \
    --models-root "$MODELS_ROOT" >"$ROOT/pull-prepare-dry-run.out"
contains "$ROOT/pull-prepare-dry-run.out" \
    'prepare     planned after acquisition; no source or build state changed'
test ! -e "$MODELS_ROOT/local/qwen/dry-run-local"

expect_rc 2 "$YVEX_BIN" model pull "$ROOT/input/tiny-external.gguf" \
    --models-root "$MODELS_ROOT" >"$ROOT/local-mode.out" 2>"$ROOT/local-mode.err"
contains "$ROOT/local-mode.err" 'requires --managed or --reference'

"$YVEX_BIN" model pull "file://$ROOT/input/tiny-external.gguf" \
    --reference --name tiny-external --family qwen --models-root "$MODELS_ROOT" \
    --json >"$ROOT/pull-external.json"
python3 - "$ROOT/pull-external.json" "$ROOT/input/tiny-external.gguf" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.pull.v1"
assert item["model"] == "tiny-external"
assert item["storage"] == "external"
assert item["format"] == "gguf"
assert item["location"] == sys.argv[2]
assert len(item["digest"]) == 64
PY

"$YVEX_BIN" model push tiny-external "$ROOT/export/tiny.gguf" \
    --representation source --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/push-local.json"
cmp "$ROOT/input/tiny-external.gguf" "$ROOT/export/tiny.gguf"
python3 - "$ROOT/push-local.json" "$ROOT/export/tiny.gguf" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.push.v1"
assert item["model"] == "tiny-external"
assert item["destination"] == sys.argv[2]
assert item["bytes"] > 0
assert len(item["representation_identity"]) == 64
PY

printf 'tiny managed gguf\n' >"$ROOT/input/tiny-managed.gguf"
"$YVEX_BIN" model pull "$ROOT/input/tiny-managed.gguf" --managed \
    --name tiny-managed --family qwen --models-root "$MODELS_ROOT" \
    --json >"$ROOT/pull-managed.json"
managed_location=$(python3 - "$ROOT/pull-managed.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["storage"] == "managed"
assert item["model"] == "tiny-managed"
assert item["format"] == "gguf"
from pathlib import Path
location = Path(item["location"])
assert location.name == "model.gguf"
assert len(location.parent.name) == 64
assert all(c in "0123456789abcdef" for c in location.parent.name)
assert location.parent.parent.name == "local"
assert location.parent.parent.parent.name == "source"
print(item["location"])
PY
)
test -f "$managed_location"
rm "$ROOT/input/tiny-managed.gguf"
printf 'tiny managed alternate gguf\n' >"$ROOT/input/tiny-managed-alt.gguf"
"$YVEX_BIN" model pull "$ROOT/input/tiny-managed-alt.gguf" --managed \
    --name tiny-managed --family qwen --models-root "$MODELS_ROOT" \
    --json >"$ROOT/pull-managed-alt.json"

# The catalog is one logical row per model and exposes origin, storage,
# representation, state, exact location, and machine identities.
"$YVEX_BIN" model list --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/models.json"
python3 - "$ROOT/models.json" "$managed_location" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
assert len(models) == 2
by_name = {item["selector"]: item for item in models}
external = by_name["tiny-external"]
managed = by_name["tiny-managed"]
assert external["state"] == "EXTERNAL"
assert external["origin"] == "local"
assert external["format"] == "gguf"
assert external["execution"] == "not prepared"
assert external["sources"][0]["storage"] == "external"
assert managed["state"] == "VERIFIED"
assert managed["format"] == "multiple"
assert managed["quant_precision"] == "select variant"
assert managed["location"] == "2 local source representations"
assert sys.argv[2] in {item["path"] for item in managed["sources"]}
assert all(item["verification"] == "payload-verified"
           for item in managed["sources"])
assert len(managed["sources"]) == 2
assert managed["representation_count"] == 2
PY

COLUMNS=240 NO_COLOR=1 "$YVEX_BIN" model list --wide \
    --models-root "$MODELS_ROOT" --registry "$REGISTRY" >"$ROOT/models-wide.out"
contains "$ROOT/models-wide.out" 'MODEL'
contains "$ROOT/models-wide.out" 'FAMILY'
contains "$ROOT/models-wide.out" 'ORIGIN'
contains "$ROOT/models-wide.out" 'FORMAT'
contains "$ROOT/models-wide.out" 'QUANT/PRECISION'
contains "$ROOT/models-wide.out" 'SIZE'
contains "$ROOT/models-wide.out" 'STATE'
contains "$ROOT/models-wide.out" 'EXEC'
contains "$ROOT/models-wide.out" 'VARIANTS'
contains "$ROOT/models-wide.out" 'LOCATION'
contains "$ROOT/models-wide.out" 'tiny-external'
contains "$ROOT/models-wide.out" 'EXTERNAL'
contains "$ROOT/models-wide.out" 'tiny-managed'
contains "$ROOT/models-wide.out" '2'
contains "$ROOT/models-wide.out" "$ROOT/input/tiny-external.gguf"
! LC_ALL=C grep "$(printf '\033')" "$ROOT/models-wide.out" >/dev/null

COLUMNS=70 NO_COLOR=1 "$YVEX_BIN" model list --all \
    --models-root "$MODELS_ROOT" --registry "$REGISTRY" >"$ROOT/models-all-narrow.out"
contains "$ROOT/models-all-narrow.out" "$ROOT/input/tiny-external.gguf"

COLUMNS=70 NO_COLOR=1 "$YVEX_BIN" model list \
    --models-root "$MODELS_ROOT" --registry "$REGISTRY" >"$ROOT/models-narrow.out"
contains "$ROOT/models-narrow.out" '2 representations'

# Styling is a TTY-only projection, and NO_COLOR removes every escape byte
# without changing the model facts being rendered.
pty_command="$YVEX_BIN model list --wide --models-root $MODELS_ROOT --registry $REGISTRY"
env -u NO_COLOR COLUMNS=180 TERM=xterm-256color \
    script -q -e -c "$pty_command" "$ROOT/models-color.typescript" </dev/null >/dev/null
LC_ALL=C grep "$(printf '\033')" "$ROOT/models-color.typescript" >/dev/null
NO_COLOR=1 COLUMNS=180 TERM=xterm-256color \
    script -q -e -c "$pty_command" "$ROOT/models-no-color.typescript" </dev/null >/dev/null
! LC_ALL=C grep "$(printf '\033')" "$ROOT/models-no-color.typescript" >/dev/null

"$YVEX_BIN" model show tiny-managed --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" >"$ROOT/model-show.out"
for section in MODEL 'ORIGIN / SOURCE' REPRESENTATIONS RUNTIME; do
    contains "$ROOT/model-show.out" "$section"
done
contains "$ROOT/model-show.out" "$managed_location"
contains "$ROOT/model-show.out" 'payload-verified'
contains "$ROOT/model-show.out" 'not launchable'
test "$(grep -c ' · managed · ' "$ROOT/model-show.out")" -eq 2

"$YVEX_BIN" model list --all --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" >"$ROOT/local-models-all.out"
test "$(grep -c '^tiny-managed' "$ROOT/local-models-all.out")" -eq 1
test "$(grep -c 'source.*gguf' "$ROOT/local-models-all.out")" -ge 2

"$YVEX_BIN" model search tiny --provider local --models-root "$MODELS_ROOT" \
    >"$ROOT/search-local.out"
contains "$ROOT/search-local.out" 'LOCAL MODELS'
contains "$ROOT/search-local.out" 'tiny-external'
contains "$ROOT/search-local.out" 'tiny-managed'

# Removing an external dependency changes the derived product state rather
# than silently treating the reference as present.
rm "$ROOT/input/tiny-external.gguf"
"$YVEX_BIN" model list --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/models-missing.json"
python3 - "$ROOT/models-missing.json" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
item = next(model for model in models if model["selector"] == "tiny-external")
assert item["state"] == "BLOCKED"
assert item["sources"][0]["state"] == "source-missing"
PY

# Hugging Face references resolve to immutable revisions without downloading
# payloads.  Ambiguous representation selection is refused outside a TTY.
expect_rc 2 "$YVEX_BIN" model pull hf://MiniMaxAI/MiniMax-H3 --reference \
    --models-root "$MODELS_ROOT" >"$ROOT/hf-ambiguous.out" 2>"$ROOT/hf-ambiguous.err"
contains "$ROOT/hf-ambiguous.err" 'multiple representations are available'
"$YVEX_BIN" model pull hf://MiniMaxAI/MiniMax-H3 --reference \
    --format safetensors --name remote-h3 --family minimax-h3 \
    --models-root "$MODELS_ROOT" --json >"$ROOT/hf-reference.json"
python3 - "$ROOT/hf-reference.json" <<'PY'
import json, re, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["state"] == "REMOTE"
assert item["format"] == "safetensors"
assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item["revision"])
assert item["origin"].endswith("@" + item["revision"])
PY
! grep -A2 '^  download$' "$YVEX_FAKE_HF_LOG" >/dev/null

# Porcelain derives a stable name and authenticated family metadata.  An
# unknown family is still a valid exact remote source record, not a failed
# acquisition, and remote facts are inspectable through model show.
YVEX_FAKE_HF_DISCOVERY_MODE=tiny \
    "$YVEX_BIN" model show hf://community/unknown-model \
    --models-root "$MODELS_ROOT" >"$ROOT/remote-show.out"
contains "$ROOT/remote-show.out" 'repository  community/unknown-model'
contains "$ROOT/remote-show.out" 'family      unknown'
contains "$ROOT/remote-show.out" 'REPRESENTATIONS'
contains "$ROOT/remote-show.out" 'use --json for exact paths'
! grep -F -- 'use --audit' "$ROOT/remote-show.out" >/dev/null
YVEX_FAKE_HF_DISCOVERY_MODE=tiny \
    "$YVEX_BIN" model pull hf://community/unknown-model --reference \
    --format gguf --models-root "$MODELS_ROOT" --json \
    >"$ROOT/unknown-reference.json"
python3 - "$ROOT/unknown-reference.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["model"] == "unknown-model"
assert item["state"] == "REMOTE"
assert item["format"] == "gguf"
PY
"$YVEX_BIN" model list --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/models-with-unknown.json"
python3 - "$ROOT/models-with-unknown.json" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
item = next(model for model in models if model["selector"] == "unknown-model")
assert item["family"] == "unknown"
assert item["state"] == "REMOTE"
assert item["execution"] == "unbound source"
PY

# Standalone and numbered payloads are separate choices. Selection excludes the
# unselected payload even when include globs are broad; malformed populations fail.
YVEX_FAKE_HF_DISCOVERY_MODE=alternative-safetensors \
    expect_rc 2 "$YVEX_BIN" model pull hf://community/alternative-model \
    --format safetensors --dry-run --models-root "$MODELS_ROOT" \
    >"$ROOT/alternative-ambiguous.out" 2>"$ROOT/alternative-ambiguous.err"
contains "$ROOT/alternative-ambiguous.err" 'multiple representations are available'
YVEX_FAKE_HF_LOG="$ROOT/alternative-hf.log" \
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_DISCOVERY_MODE=alternative-safetensors \
    "$YVEX_BIN" model pull hf://community/alternative-model \
    --format safetensors --variant safetensors-source --dry-run --models-root "$MODELS_ROOT" \
    >"$ROOT/alternative-selected.out"
contains "$ROOT/alternative-selected.out" 'representation safetensors-source'
python3 - "$ROOT/alternative-hf.log" <<'PY'
import pathlib, sys
calls = pathlib.Path(sys.argv[1]).read_text().split("fake-hf argv:\n")
download = next([line.strip() for line in call.splitlines()] for call in calls
                if call.startswith("  download\n"))
assert "--dry-run" in download
excluded = [download[i + 1] for i, arg in enumerate(download[:-1]) if arg == "--exclude"]
assert "consolidated.safetensors" in excluded
assert "model-00001-of-00002.safetensors" not in excluded
assert "model-00002-of-00002.safetensors" not in excluded
PY
standalone_variant="safetensors-file-$(printf '%s' consolidated.safetensors | sha256sum | cut -d ' ' -f 1)"
YVEX_FAKE_HF_LOG="$ROOT/standalone-hf.log" \
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_DISCOVERY_MODE=alternative-safetensors \
    "$YVEX_BIN" model pull hf://community/alternative-model \
    --format safetensors --variant "$standalone_variant" --dry-run --models-root "$MODELS_ROOT" \
    >"$ROOT/standalone-selected.out"
python3 - "$ROOT/standalone-hf.log" <<'PY'
import pathlib, sys
calls = pathlib.Path(sys.argv[1]).read_text().split("fake-hf argv:\n")
download = next([line.strip() for line in call.splitlines()] for call in calls
                if call.startswith("  download\n"))
excluded = [download[i + 1] for i, arg in enumerate(download[:-1]) if arg == "--exclude"]
assert "consolidated.safetensors" not in excluded
assert {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"} <= set(excluded)
PY
test ! -d "$MODELS_ROOT/hf/minimax/alternative-model"
for population in incomplete-shards duplicate-shards; do
    YVEX_FAKE_HF_DISCOVERY_MODE="$population" \
        expect_rc 4 "$YVEX_BIN" model pull hf://community/alternative-model \
        --format safetensors --dry-run --models-root "$MODELS_ROOT" \
        >"$ROOT/$population.out" 2>"$ROOT/$population.err"
    contains "$ROOT/$population.err" 'provider'
done

# A real bounded provider pull uses the existing acquisition owner, pins the
# immutable revision, records local content identity without promoting it to
# unavailable provider-object hash proof, and is visible through
# the model-level lifecycle commands.  The fake provider creates only tiny
# structurally valid shards; no production model payload is downloaded.
YVEX_FAKE_HF_AUTH=1 YVEX_FAKE_HF_LOCAL_CACHE=1 \
    "$YVEX_BIN" model pull hf://MiniMaxAI/MiniMax-H3 \
    --format safetensors --name pulled-h3 --family minimax-h3 \
    --models-root "$MODELS_ROOT" --json >"$ROOT/hf-pull.json"
python3 - "$ROOT/hf-pull.json" <<'PY'
import json, re, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.pull.v1"
assert item["status"] == "model-download-pass"
assert item["model"] == "pulled-h3"
assert re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", item["revision"])
assert item["safetensors_files"] == 2
assert item["files"] >= 2 and item["bytes"] > 0
assert item["format"] == "safetensors"
assert item["precision"] == "F16"
assert len(item["local_content_digest"]) == 64
assert item["upstream_identity_verified"] is True
assert item["payload_hash_verified"] is False
PY
pulled_revision=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$ROOT/hf-pull.json")
pulled_source="$MODELS_ROOT/source/hf/MiniMaxAI/MiniMax-H3/$pulled_revision"
test -L "$pulled_source/.cache"
test "$(readlink "$pulled_source/.cache")" = "$MODELS_ROOT/cache/hf/MiniMaxAI/MiniMax-H3/$pulled_revision"
test -f "$pulled_source/.cache/huggingface/download/config.json.metadata"
"$YVEX_BIN" model status pulled-h3 --models-root "$MODELS_ROOT" \
    --json >"$ROOT/hf-status.json"
python3 - "$ROOT/hf-status.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.acquisition.status.v1"
assert item["model"] == "pulled-h3"
assert item["active"] is False
assert item["files"] >= 2 and item["bytes"] > 0
PY
"$YVEX_BIN" model list --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/models-after-hf-pull.json"
python3 - "$ROOT/models-after-hf-pull.json" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
item = next(model for model in models if model["selector"] == "pulled-h3")
assert item["state"] == "SOURCE"
assert item["format"] == "safetensors"
assert item["quant_precision"] == "FP16"
assert item["origin"] == "Hugging Face"
assert item["sources"][0]["storage"] == "managed"
assert item["sources"][0]["verification"] == "revision-verified"
assert len(item["sources"][0]["digest"]) == 64
PY

expect_rc 3 "$YVEX_BIN" model pull ssh://example/model --reference \
    --models-root "$MODELS_ROOT" >"$ROOT/ssh.out" 2>"$ROOT/ssh.err"
contains "$ROOT/ssh.err" 'ssh transport unavailable'
expect_rc 3 "$YVEX_BIN" model pull hf://MiniMaxAI/MiniMax-H3 --prepare --stream \
    --models-root "$MODELS_ROOT" >"$ROOT/stream.out" 2>"$ROOT/stream.err"
contains "$ROOT/stream.err" 'streamed preparation is unavailable'
expect_rc 3 "$YVEX_BIN" model push tiny-managed hf://user/repository \
    --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    >"$ROOT/push-hf.out" 2>"$ROOT/push-hf.err"
contains "$ROOT/push-hf.err" 'provider write unavailable'

# Arbitrary local GGUF records are preserved without requantization and remain
# blocked until an exact family/runtime binding exists.
expect_rc 3 "$YVEX_BIN" model prepare tiny-managed --quant source-faithful \
    --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    >"$ROOT/prepare-quant.out" 2>"$ROOT/prepare-quant.err"
contains "$ROOT/prepare-quant.err" 'preserved without requantization'
expect_rc 3 "$YVEX_BIN" model prepare tiny-managed --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" --json >"$ROOT/prepare-blocked.json"
python3 - "$ROOT/prepare-blocked.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.prepare.v1"
assert item["model"] == "tiny-managed"
assert item["state"] == "BLOCKED" and not item["changed"]
assert "preserved without requantization" in item["blocker"]
PY

# One exact immutable DeepSeek source resolves to the family-owned compiler
# and deployment defaults.  Dry-run proves the complete orchestration without
# quantizing production-sized weights in the fast CLI lane.
YVEX_FAKE_HF_AUTH=1 \
YVEX_FAKE_HF_RESOLVED_SHA=62af8fffb2f7030cac4de2f0169f5b8d1101b646 \
    "$YVEX_BIN" model pull hf://deepseek-ai/DeepSeek-V4-Flash-DSpark \
    --format safetensors --name dry-run-dspark --family deepseek \
    --models-root "$MODELS_ROOT" --dry-run >"$ROOT/deepseek-pull-dry-run.out"
contains "$ROOT/deepseek-pull-dry-run.out" 'status: model-download-dry-run'
contains "$ROOT/deepseek-pull-dry-run.out" 'model-download: plan target=dry-run-dspark'
contains "$ROOT/deepseek-pull-dry-run.out" 'stage: download planned (dry-run)'
! grep 'fake-hf: dry-run' "$ROOT/deepseek-pull-dry-run.out" >/dev/null
! grep 'model-download: start' "$ROOT/deepseek-pull-dry-run.out" >/dev/null
! grep 'stage: download running' "$ROOT/deepseek-pull-dry-run.out" >/dev/null
! grep 'tick: elapsed=' "$ROOT/deepseek-pull-dry-run.out" >/dev/null
test ! -e "$MODELS_ROOT/source/hf/deepseek-ai/DeepSeek-V4-Flash-DSpark/62af8fffb2f7030cac4de2f0169f5b8d1101b646"
test ! -e "$MODELS_ROOT/evidence/build/deepseek/dry-run-dspark.download.receipt"
test ! -e "$MODELS_ROOT/evidence/build/deepseek/dry-run-dspark.download.active.json"
test ! -e "$MODELS_ROOT/evidence/build/acquisition/dry-run-dspark.download.stdout.log"
test ! -e "$MODELS_ROOT/evidence/build/acquisition/dry-run-dspark.download.stderr.log"
YVEX_FAKE_HF_AUTH=1 \
YVEX_FAKE_HF_RESOLVED_SHA=62af8fffb2f7030cac4de2f0169f5b8d1101b646 \
    "$YVEX_BIN" model pull hf://deepseek-ai/DeepSeek-V4-Flash-DSpark \
    --format safetensors --name dry-run-dspark --family deepseek \
    --models-root "$MODELS_ROOT" --dry-run --verbose \
    >"$ROOT/deepseek-pull-dry-run-verbose.out"
contains "$ROOT/deepseek-pull-dry-run-verbose.out" 'fake-hf: dry-run'
YVEX_FAKE_HF_AUTH=1 \
YVEX_FAKE_HF_RESOLVED_SHA=62af8fffb2f7030cac4de2f0169f5b8d1101b646 \
    "$YVEX_BIN" model pull hf://deepseek-ai/DeepSeek-V4-Flash-DSpark \
    --format safetensors --name deepseek-dspark --family deepseek \
    --models-root "$MODELS_ROOT" --json >"$ROOT/deepseek-pull.json"
"$YVEX_BIN" model prepare deepseek-dspark --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" --dry-run --json >"$ROOT/deepseek-prepare-plan.json"
python3 - "$ROOT/deepseek-prepare-plan.json" "$MODELS_ROOT" <<'PY'
import json, os, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.prepare.v1"
assert item["model"] == "DeepSeek-V4-Flash"
assert item["state"] == "PLANNED" and not item["changed"]
assert item["revision"] == "62af8fffb2f7030cac4de2f0169f5b8d1101b646"
assert item["target"] == "deepseek4-v4-flash-dspark"
assert item["quant"] == "deepseek-v4-flash-dspark-q8_0-q2_k-v1"
assert item["backend"] == "cuda"
assert item["artifact"] == (sys.argv[2] + "/representations/deepseek4-v4-flash-dspark/"
                            "<physical-variant-identity>/model.gguf")
assert not os.path.exists(item["artifact"])
PY

# Historical variants remain inspectable, but arbitrary files named as runtime
# bindings cannot create launchable porcelain candidates.
printf 'runtime binding fixture\n' >"$ROOT/runtime.binding"
for qtype in F16 F32; do
    lower=$(printf '%s' "$qtype" | tr '[:upper:]' '[:lower:]')
    artifact="$ROOT/workflow-demo-$lower.gguf"
    "$YVEX_BIN" compile artifact emit --out "$artifact" --model-name workflow-demo \
        --target-name workflow-target --target-qtype "$qtype" --overwrite >/dev/null
    "$YVEX_BIN" profile create --path "$artifact" --registry "$REGISTRY" \
        --alias "deepseek4-workflow-demo-selected-$lower" \
        --family deepseek4 --model workflow-demo \
        --target workflow-target --qprofile "$lower" --backend cpu \
        --runtime-binding "$ROOT/runtime.binding" --execution-strategy target-only \
        --ctx 1024 >/dev/null
done
printf 'newer runtime binding fixture\n' >"$ROOT/runtime.binding.next"
"$YVEX_BIN" profile create --path "$ROOT/workflow-demo-f32.gguf" \
    --registry "$REGISTRY" \
    --alias deepseek4-workflow-demo-selected-f32-current \
    --family deepseek4 --model workflow-demo --target workflow-target \
    --qprofile f32 --backend cpu --runtime-binding "$ROOT/runtime.binding.next" \
    --execution-strategy target-only --ctx 1024 >/dev/null

"$YVEX_BIN" model list --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    --json >"$ROOT/models-variants.json"
python3 - "$ROOT/models-variants.json" <<'PY'
import json, sys
models = json.load(open(sys.argv[1], encoding="utf-8"))["models"]
matches = [item for item in models if item["selector"] == "workflow-demo"]
assert len(matches) == 1
assert len(matches[0]["representations"]) == 2
assert len(matches[0]["profiles"]) == 3
assert matches[0]["state"] == "BLOCKED"
assert matches[0]["format"] == "multiple"
assert matches[0]["quant_precision"] == "select variant"
assert matches[0]["selected_profile"] is None
assert all(not profile["launchable"] for profile in matches[0]["profiles"])
assert matches[0]["representation_count"] == 2
PY
"$YVEX_BIN" model list --all --models-root "$MODELS_ROOT" --registry "$REGISTRY" \
    >"$ROOT/models-all.out"
test "$(grep -c '^workflow-demo' "$ROOT/models-all.out")" -eq 1
test "$(grep -Ec 'alternate.*gguf' "$ROOT/models-all.out")" -eq 2
test "$(grep -Ec 'BLOCKED +not current' "$ROOT/models-all.out")" -eq 2
"$YVEX_BIN" model show workflow-demo --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" >"$ROOT/workflow-show.out"
contains "$ROOT/workflow-show.out" 'State           BLOCKED'
contains "$ROOT/workflow-show.out" 'Execution       not current'
contains "$ROOT/workflow-show.out" 'FP16'
contains "$ROOT/workflow-show.out" 'FP32'
contains "$ROOT/workflow-show.out" 'not launchable; run `yvex model prepare MODEL`'
! grep -F 'DEPLOYS' "$ROOT/workflow-show.out" >/dev/null

expect_rc 3 "$YVEX_BIN" model prepare workflow-demo --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" >"$ROOT/prepare-blocked-history.out" \
    2>"$ROOT/prepare-blocked-history.err"
contains "$ROOT/prepare-blocked-history.err" \
    'model has no exact acquired source-to-ready compiler binding'
expect_rc 3 "$YVEX_BIN" model prepare workflow-demo --models-root "$MODELS_ROOT" \
    --registry "$REGISTRY" --json >"$ROOT/prepare-blocked-history.json"
python3 - "$ROOT/prepare-blocked-history.json" <<'PY'
import json, sys
item = json.load(open(sys.argv[1], encoding="utf-8"))
assert item["schema"] == "yvex.model.prepare.v1"
assert item["model"] == "workflow-demo"
assert item["state"] == "BLOCKED" and not item["changed"]
assert item["blocker"] == \
    "model has no exact acquired source-to-ready compiler binding"
PY
expect_rc 1 "$YVEX_BIN" model load workflow-demo \
    >"$ROOT/load-selected.out" 2>"$ROOT/load-selected.err"
contains "$ROOT/load-selected.err" 'model is not launchable: workflow-demo'
expect_rc 1 "$YVEX_BIN" model load workflow-demo \
    --variant deepseek4-workflow-demo-selected-f32-current \
    >"$ROOT/load-profile-alias.out" 2>"$ROOT/load-profile-alias.err"
contains "$ROOT/load-profile-alias.err" 'model is not launchable: workflow-demo'
set +e
printf '1\nq\n' | NO_COLOR=1 TERM=xterm-256color script -q -e -c \
    "$YVEX_BIN model load" \
    "$ROOT/load-variant-selector.typescript" >/dev/null 2>&1
selector_status=$?
set -e
test "$selector_status" -eq 1
contains "$ROOT/load-variant-selector.typescript" \
    'no launchable models are known locally'
! grep -F 'Select representation and deployment' \
    "$ROOT/load-variant-selector.typescript" >/dev/null
expect_rc 1 "$YVEX_BIN" model load workflow-demo --variant f16 \
    >"$ROOT/load-no-host.out" 2>"$ROOT/load-no-host.err"
contains "$ROOT/load-no-host.err" 'model is not launchable: workflow-demo'

printf 'model workflow porcelain: ok\n'
