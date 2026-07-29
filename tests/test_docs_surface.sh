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
require_text README.md '## Product topology'
require_text README.md '## Runtime flow'
require_text README.md '## Product client'
require_text README.md '## Three terminal views'
require_text README.md '## Build and package'
require_text README.md '## Capability boundary'
require_text README.md '-> yvexd: one long-lived local runtime host'
require_text README.md '-> yvex: thin product client over the local protocol'
require_text README.md '-> yvex-dev: optional compiler, graph, artifact, and evidence tooling'
require_text README.md './yvex runtime start'
require_text README.md './yvex runtime watch'
require_text README.md './yvex chat --session main'
require_text README.md 'The old flat public command registry and its aliases are intentionally absent.'
require_text README.md 'public or remote serving'
reject_text README.md 'Active Next:'
reject_text README.md 'YVEX is release-ready'

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
require_text docs/runbooks/deepseek.md './yvex-dev graph transformer generate --help'
require_text docs/runbooks/deepseek.md 'On turn two'
require_text docs/reference-architecture.md '### 10.4 Hosted Runtime And Conversation Sessions'
require_text docs/reference-architecture.md 'exact prefix'

require_text PROJECT.md 'V010.RUNTIME.CLIENT.REFOUNDATION.0'
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

sh tests/test_project_ledger.sh
