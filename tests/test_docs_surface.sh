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
  README.md CHANGELOG.md AGENTS.md ROADMAP.md CONTRIBUTING.md NOTICE.md \
  docs/README.md docs/doctrine/principles.md docs/doctrine/glossary.md \
  docs/doctrine/evidence.md docs/reference/verified-inference.md \
  docs/architecture/system.md docs/architecture/compilation.md \
  docs/architecture/runtime.md docs/architecture/commands.md \
  docs/model-families/integration.md \
  docs/model-families/deepseek-v4-flash.md docs/model-families/qwen.md \
  docs/model-families/gemma.md docs/contracts/artifacts.md \
  docs/contracts/runtime.md docs/contracts/local-protocol.md \
  docs/contracts/events-telemetry.md docs/contracts/c-api.md \
  docs/openai-compatibility.md docs/operator-runbook.md \
  docs/operations/deepseek.md docs/operations/validation.md \
  docs/development/documentation-policy.md \
  docs/development/source-ownership.md \
  docs/releases/doctrine.md docs/releases/v0.1.md \
  docs/decisions/0003-documentation-architecture.md \
  docs/migrations/documentation-architecture-v1.md \
  docs/audits/code-commentary-7c90ce1/README.md \
  docs/audits/documentation-architecture-51a5c/README.md \
  docs/milestones/code-commentary.md \
  docs/milestones/documentation-architecture.md \
  config/documentation_owners.tsv config/frozen_documents.tsv
do
  require_file "$file"
done

for retired in \
  PROJECT.md MODEL_ARTIFACTS.md docs/api.md docs/cli-output-architecture.md \
  docs/contract.md docs/model-families.md docs/reference-architecture.md \
  docs/runbooks/README.md docs/runbooks/common.md docs/runbooks/deepseek.md \
  docs/system-target.md docs/topology-closure-audit.md \
  docs/v010-release-doctrine.md docs/spine.md
do
  test ! -e "$retired" || fail "retired documentation path exists: $retired"
done

require_text README.md '# YVEX'
require_text README.md '## Why YVEX'
require_text README.md '## Quick start'
require_text README.md '## Product boundary'
require_text README.md '## Documentation'
require_text README.md '## Current limits'
require_text README.md '## License'
require_text README.md './yvex model list'
require_text README.md './yvex model select deepseek4-v4-flash-runtime-iq2xxs'
require_text README.md './yvex model selected'
require_text README.md './yvex runtime start'
require_text README.md 'There is no separate model-load command.'
require_text README.md './yvex runtime model'
require_text README.md './yvex runtime memory'
require_text README.md './yvex chat --session main'
require_text README.md './yvex run "Explain attention in one sentence."'
require_text README.md './yvex help --json'
require_text README.md '[`ROADMAP.md`](ROADMAP.md)'
reject_text README.md 'Active Next:'
reject_text README.md '`yvex-dev`'
reject_text README.md '`yvex-openai`'
reject_text README.md 'Terminal 1 —'
reject_text README.md 'export YVEX_MODEL_ARTIFACT'
reject_text README.md './yvex runtime start \'

readme_lines=$(wc -l < README.md | tr -d ' ')
test "$readme_lines" -ge 80 && test "$readme_lines" -le 220 ||
  fail "README compact-entry bounds failed: $readme_lines lines"

require_text CHANGELOG.md '## Unreleased'
require_text CHANGELOG.md '### Added'
require_text CHANGELOG.md '### Changed'
require_text CHANGELOG.md '### Removed'
reject_text CHANGELOG.md 'V010.'

require_text docs/doctrine/principles.md '# YVEX Principles'
require_text docs/doctrine/principles.md '## Identity-bound derivation'
require_text docs/doctrine/evidence.md '## Promotion rules'
require_text docs/doctrine/glossary.md '| Complete artifact |'
require_text docs/doctrine/glossary.md '| Supported artifact |'
require_text docs/doctrine/glossary.md '| Semantic graph |'
require_text docs/doctrine/glossary.md '| Executable graph |'
require_text docs/doctrine/glossary.md '| Launch graph |'

require_text docs/reference/verified-inference.md '# Reference Architecture for Verified Transformer Inference'
require_text docs/architecture/system.md '# Implemented YVEX System'
require_text docs/architecture/compilation.md '## Runtime binding'
require_text docs/architecture/runtime.md '## Persistent state'
require_text docs/architecture/commands.md 'yvex.operator.registry.v1'
require_text docs/architecture/commands.md 'There is no independent hosted `load` operation.'

require_text docs/model-families/integration.md '# Model-Family Integration Contract'
require_text docs/model-families/deepseek-v4-flash.md 'sole complete YVEX source-to-streamed-text vertical'
require_text docs/model-families/qwen.md 'unsupported runtime family'
require_text docs/model-families/gemma.md 'unsupported runtime family'

require_text docs/contracts/artifacts.md '# Artifact and Admission Contract'
require_text docs/contracts/runtime.md 'A client connection is not a session.'
require_text docs/contracts/runtime.md 'no explicit CUDA request'
require_text docs/contracts/local-protocol.md 'YVEX_CLIENT_PROTOCOL_VERSION = 4'
require_text docs/contracts/events-telemetry.md 'No consumer scrapes another renderer'
require_text docs/contracts/c-api.md '## Compiled Operator Registry Boundary'
require_text docs/contracts/c-api.md 'yvex.models.local.v3'
require_text docs/openai-compatibility.md 'yvex.openai.compat.v1'
require_text docs/openai-compatibility.md 'YVEX never executes application tools.'

require_text docs/operator-runbook.md '## First verified startup'
require_text docs/operator-runbook.md '## What “load the model” means'
require_text docs/operator-runbook.md '## Three-terminal operation'
require_text docs/operator-runbook.md '## Registering an existing model'
require_text docs/operator-runbook.md './yvex model list'
require_text docs/operator-runbook.md './yvex model select deepseek4-v4-flash-runtime-iq2xxs'
require_text docs/operator-runbook.md './yvex runtime start'
require_text docs/operator-runbook.md './yvex runtime model'
require_text docs/operator-runbook.md './yvex runtime memory'
require_text docs/operator-runbook.md './yvex chat --session main'
require_text docs/operator-runbook.md './yvex runtime watch'
require_text docs/operator-runbook.md './yvex runtime trace'
reject_text docs/operator-runbook.md 'export YVEX_MODEL_ARTIFACT'
require_text docs/operations/deepseek.md './yvex execute transformer generate --help'

require_text docs/development/documentation-policy.md '## Authority rules'
require_text docs/development/documentation-policy.md '## Changelog policy'
require_text docs/development/documentation-policy.md '## Validation'
require_text docs/releases/doctrine.md '## Gate meanings'
require_text docs/releases/v0.1.md 'Status: unreleased target record'

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
done

test -z "$(find docs/diagrams -maxdepth 1 -type f \
  \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) -print -quit)" ||
  fail 'architecture diagrams contain a raster text asset'

if grep -niE \
  'production-ready|blazing fast|state of the art|enterprise-grade|seamless|cutting-edge|revolutionary' \
  README.md; then
  fail 'README contains unsupported marketing language'
fi

if grep -nE 'V010\.|POST010\.|(/home/|/Users/|\$HOME/)' README.md; then
  fail 'README contains project-control or machine-local detail'
fi

if test -x ./yvex; then
  client_help=$(./yvex --help)
  for command in 'yvex run' 'yvex runtime status' 'yvex session cancel' \
    'yvex compile quant plan'
  do
    printf '%s\n' "$client_help" | grep -F "$command" >/dev/null ||
      fail "built yvex help lacks canonical command: $command"
  done
  if printf '%s\n' "$client_help" | grep -F 'yvex graph' >/dev/null; then
    fail 'built yvex help exposes retired graph namespace'
  fi
fi

if test -x ./yvexd; then
  daemon_help=$(./yvexd --help)
  printf '%s\n' "$daemon_help" | grep -F '[--console off|raw]' >/dev/null ||
    fail 'built yvexd help lacks raw console policy'
  printf '%s\n' "$daemon_help" | grep -F '[--openai on|off]' >/dev/null ||
    fail 'built yvexd help lacks integrated OpenAI listener policy'
fi

test ! -e ./yvex-dev || fail 'retired yvex-dev executable remains'
test ! -e ./yvex-openai || fail 'retired yvex-openai executable remains'

python3 tests/documentation_architecture.py
