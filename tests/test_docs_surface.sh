#!/usr/bin/env sh
set -eu

fail() {
  printf 'docs surface: %s\n' "$1" >&2
  exit 1
}

require_file() {
  test -f "$1" || fail "missing file: $1"
}

require_text() {
  grep -nF -- "$2" "$1" >/dev/null || fail "$1 missing required text: $2"
}

reject_text() {
  if grep -nF -- "$2" "$1" >/dev/null; then
    fail "$1 retains forbidden text: $2"
  fi
}

for file in \
  README.md AGENTS.md PROJECT.md MODEL_ARTIFACTS.md NOTICE.md \
  docs/api.md docs/contract.md docs/model-families.md \
  docs/operator-runbook.md docs/cli-output-architecture.md \
  docs/reference-architecture.md docs/v010-release-doctrine.md \
  docs/topology-closure-audit.md docs/system-target.md \
  docs/diagrams/system_overview.mmd docs/diagrams/system_overview.svg \
  docs/diagrams/physical_compilation.mmd docs/diagrams/physical_compilation.svg \
  docs/diagrams/runtime_host_sessions.mmd docs/diagrams/runtime_host_sessions.svg \
  docs/diagrams/autoregressive_execution.mmd docs/diagrams/autoregressive_execution.svg \
  docs/runbooks/README.md docs/runbooks/deepseek.md docs/runbooks/common.md
do
  require_file "$file"
done

test ! -e docs/spine.md || fail 'obsolete project-control path exists'
test ! -e docs/runbooks/glm.md || fail 'unsupported GLM runbook remains'
test "$(find docs/runbooks -maxdepth 1 -type f -name '*.md' | wc -l | tr -d ' ')" -eq 3 ||
  fail 'unexpected runbook count'

for text in \
  '`PROJECT.md` is the sole project-control authority.' \
  'is exactly one active milestone and exactly one Active Next.' \
  '### Progression admissibility' \
  '### Six-pass vertical iteration' \
  '### Quality and evaluation taxonomy' \
  'New commits use Conventional Commits:'
do
  require_text AGENTS.md "$text"
done

require_text README.md '# YVEX'
require_text README.md '## Run YVEX'
require_text README.md '### Terminal 1 — runtime and raw events'
require_text README.md '### Terminal 2 — engine watch'
require_text README.md '### Terminal 3 — REPL'
require_text README.md '### One-shot run'
require_text README.md '## Product topology'
require_text README.md '## What YVEX guarantees'
require_text README.md '## Current vertical'
require_text README.md '## Build'
require_text README.md '## Documentation'
require_text README.md '### Operate YVEX'
require_text README.md '## Current limits'
require_text README.md 'docs/diagrams/system_overview.svg'
require_text README.md 'docs/diagrams/system_overview.mmd'
require_text README.md 'libyvex'
require_text README.md '`yvexd`'
require_text README.md '`yvex`'
require_text README.md '`yvex-dev`'
require_text README.md './yvexd'
require_text README.md './yvex runtime watch'
require_text README.md './yvex chat --session main'
require_text README.md './yvex run "Explain attention in one sentence."'
require_text README.md './yvex runtime status --json'
require_text README.md './yvex runtime stop'
require_text README.md '[`PROJECT.md`](PROJECT.md)'
require_text README.md 'a public or remote production server'
reject_text README.md 'Active Next:'
reject_text README.md 'YVEX is release-ready'
reject_text README.md 'YVEX is CLI-only'
reject_text README.md 'yvexd bounded status shell'
reject_text README.md 'top-level generation CLI pending'
reject_text README.md '--output normal'
reject_text README.md '--output table'
reject_text README.md '--output audit'
reject_text README.md '--include-'

readme_lines=$(wc -l < README.md | tr -d ' ')
test "$readme_lines" -le 450 || fail "README exceeds 450-line hard cap: $readme_lines"
test "$readme_lines" -ge 200 || fail "README is too short to own the product entry point: $readme_lines"

for old in \
  './yvex graph ' './yvex materialize ' './yvex quant-policy ' \
  './yvex tokenizer ' './yvex metadata ' './yvex tensor-map ' \
  './yvex model-target ' './yvex fullmodel '
do
  if grep -nF -- "$old" README.md docs/*.md docs/runbooks/*.md >/dev/null; then
    fail "public documentation retains old command: $old"
  fi
done

require_text docs/cli-output-architecture.md '# Client And Terminal Architecture'
require_text docs/cli-output-architecture.md '## Binary boundaries'
require_text docs/cli-output-architecture.md '## Product grammar'
require_text docs/cli-output-architecture.md '## Developer grammar'
require_text docs/cli-output-architecture.md '## Typed event fan-out'
require_text docs/cli-output-architecture.md 'The former flat command catalog and selectable `normal|table|audit` layouts are'

require_text docs/contract.md '## Server Contract'
require_text docs/contract.md 'one process-resident runtime model'
require_text docs/contract.md 'A client connection is not a session.'
require_text docs/contract.md 'publishes bytes only after model commit'
require_text docs/contract.md 'Public'
require_text docs/contract.md 'HTTP, authentication, TLS'
require_text docs/api.md '`<yvex/server.h>` | local protocol, runtime host, sessions, telemetry'

require_text docs/operator-runbook.md './yvex runtime start'
require_text docs/operator-runbook.md './yvex session reset main'
require_text docs/operator-runbook.md 'It does not load the complete GGUF into anonymous RAM'
require_text docs/operator-runbook.md '## Prerequisites'
require_text docs/operator-runbook.md '## First verified startup'
require_text docs/operator-runbook.md '## Three-terminal operation'
require_text docs/operator-runbook.md '## Optional configured defaults'
require_text docs/operator-runbook.md './yvexd --model "$YVEX_MODEL_ARTIFACT" --runtime-binding "$YVEX_RUNTIME_BINDING" --backend cuda --context 4096 --console raw --trace-level stages'
reject_text README.md '### Use YVEX'

first_operator_command=$(grep -nE '^\./yvex(d|-dev)? ' docs/operator-runbook.md | head -n 1 || true)
case "$first_operator_command" in
  *:./yvexd\ --model\ *) ;;
  *) fail 'operator runbook does not begin product operation with explicit yvexd startup' ;;
esac

defaults_line=$(grep -nF '## Optional configured defaults' docs/operator-runbook.md | cut -d: -f1)
model_use_line=$(grep -nF './yvex model use ' docs/operator-runbook.md | cut -d: -f1)
test "$model_use_line" -gt "$defaults_line" ||
  fail 'operator runbook promotes model use before optional configured defaults'

if grep -nE '\\[[:space:]]*$' docs/operator-runbook.md; then
  fail 'operator runbook contains a multiline shell continuation'
fi

require_text docs/runbooks/deepseek.md './yvex-dev graph transformer generate --help'
require_text docs/runbooks/deepseek.md 'On turn two'
require_text docs/reference-architecture.md '### 10.4 Hosted Runtime And Conversation Sessions'
require_text docs/reference-architecture.md 'exact prefix'
require_text docs/reference-architecture.md 'diagrams/physical_compilation.svg'
require_text docs/reference-architecture.md 'diagrams/physical_compilation.mmd'
require_text docs/reference-architecture.md 'diagrams/runtime_host_sessions.svg'
require_text docs/reference-architecture.md 'diagrams/runtime_host_sessions.mmd'
require_text docs/reference-architecture.md 'diagrams/autoregressive_execution.svg'
require_text docs/reference-architecture.md 'diagrams/autoregressive_execution.mmd'

for svg in \
  docs/diagrams/system_overview.svg \
  docs/diagrams/physical_compilation.svg \
  docs/diagrams/runtime_host_sessions.svg \
  docs/diagrams/autoregressive_execution.svg
do
  require_text "$svg" '<svg '
  require_text "$svg" '<title '
  require_text "$svg" '<desc '
  require_text "$svg" 'role="img"'
  require_text "$svg" '@media (prefers-color-scheme: dark)'
  require_text "$svg" '</svg>'
done

require_text docs/diagrams/system_overview.mmd 'B. Run — yvexd'
require_text docs/diagrams/system_overview.mmd 'Session registry'
require_text docs/diagrams/system_overview.mmd 'yvex-dev'
require_text docs/diagrams/physical_compilation.mmd 'Physical-variant plan'
require_text docs/diagrams/physical_compilation.mmd 'Imatrix evidence'
require_text docs/diagrams/runtime_host_sessions.mmd 'Persistent execution session'
require_text docs/diagrams/runtime_host_sessions.mmd 'Typed event authority'
require_text docs/diagrams/autoregressive_execution.mmd 'Prompt prefill'
require_text docs/diagrams/autoregressive_execution.mmd 'Terminal record'
require_text docs/diagrams/autoregressive_execution.mmd 'Model + KV commit'

test -z "$(find docs/diagrams -maxdepth 1 -type f \
  \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) -print -quit)" ||
  fail 'architecture diagrams retain a raster text asset'

require_text PROJECT.md 'V010.RUNTIME.CLIENT.REFOUNDATION.0'
require_text PROJECT.md 'V010.DOCS.README.PRODUCT.0'
require_text PROJECT.md 'V010.CLI.DEEPSEEK.GENERATE.0: superseded'
require_text PROJECT.md 'Active Next: V010.EVAL.DEEPSEEK.0'
require_text PROJECT.md 'model_behavior_evaluation_ready=0'
require_text PROJECT.md 'release_qualification_ready=0'

require_text MODEL_ARTIFACTS.md 'Tensor proof artifact'
require_text MODEL_ARTIFACTS.md 'Complete model artifact'
require_text MODEL_ARTIFACTS.md 'Supported model artifact'

if grep -nE '\b(qa_ready|qa_passed)[[:space:]]*=[[:space:]]*1\b' \
  src/*.c src/*/*.c include/yvex/*.h include/yvex/internal/*.h PROJECT.md README.md \
  2>/dev/null; then
  fail 'generic QA fact attempts to promote capability'
fi

if grep -niE \
  'production-ready|blazing fast|state of the art|enterprise-grade|seamless|cutting-edge|revolutionary' \
  README.md; then
  fail 'README contains forbidden marketing language'
fi

if grep -RniF -- 'J. C. Prado Angelo' README.md docs MODEL_ARTIFACTS.md; then
  fail 'public documentation contains injected or foreign text'
fi

if grep -nE 'V010\.|POST010\.' README.md; then
  fail 'README exposes internal project-control IDs'
fi

if grep -nE '(/home/|/Users/|\$HOME/)' README.md; then
  fail 'README exposes a local filesystem path'
fi

for target in $(grep -oE '\]\([^)]+\)' README.md |
  sed 's/^](//; s/)$//' | grep -Ev '^(https?://|mailto:|#)' || true)
do
  target=${target%%#*}
  test -e "$target" || fail "README local link does not resolve: $target"
done

if test -x ./yvex; then
  client_help=$(./yvex --help)
  printf '%s\n' "$client_help" | grep -F 'yvex run [options] TEXT' >/dev/null ||
    fail 'built yvex help lacks one-shot client'
  printf '%s\n' "$client_help" | grep -F 'yvex runtime start|stop|status|watch|trace' >/dev/null ||
    fail 'built yvex help lacks runtime administration'
  printf '%s\n' "$client_help" | grep -F 'yvex session new|list|show|attach|detach|reset|close' >/dev/null ||
    fail 'built yvex help lacks session administration'
fi

if test -x ./yvexd; then
  daemon_help=$(./yvexd --help)
  printf '%s\n' "$daemon_help" | grep -F '[--console off|raw]' >/dev/null ||
    fail 'built yvexd help lacks the raw terminal contract'
  printf '%s\n' "$daemon_help" | grep -F '[--trace-level summary|stages|tokens|full]' >/dev/null ||
    fail 'built yvexd help lacks trace levels'
fi

if test -x ./yvex-dev; then
  developer_help=$(./yvex-dev --help)
  printf '%s\n' "$developer_help" | grep -F 'yvex-dev graph ...' >/dev/null ||
    fail 'built yvex-dev help lacks separated graph tooling'
  printf '%s\n' "$developer_help" | grep -F 'yvex-dev quant preset|plan|emit|summarize|explain|policy|imatrix' >/dev/null ||
    fail 'built yvex-dev help lacks separated quant tooling'
fi

sh tests/test_project_ledger.sh
