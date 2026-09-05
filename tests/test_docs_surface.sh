#!/usr/bin/env sh
set -eu

fail() {
  printf 'docs surface: %s\n' "$1" >&2
  exit 1
}

require_text() {
  grep -nF -- "$2" "$1" >/dev/null || fail "$1 missing required text: $2"
}

reject_text() {
  if grep -nF -- "$2" "$1" >/dev/null; then
    fail "$1 retains forbidden text: $2"
  fi
}

# Inventory, links, protocol identity, assets and retired paths have one owner.
python3 tests/documentation_architecture.py
require_text README.md '## Why YVEX'
require_text README.md '## Quick start'
require_text README.md '## Product boundary'
require_text README.md '## Documentation'
require_text README.md '## Current limits'
require_text README.md './yvex model list'
require_text README.md './yvex serve'
require_text README.md './yvex model load'
require_text README.md './yvex chat'
reject_text README.md 'yvex run'
reject_text README.md 'yvex server'
reject_text README.md 'Active Next:'
reject_text README.md 'export YVEX_MODEL_ARTIFACT'

readme_lines=$(wc -l < README.md | tr -d ' ')
test "$readme_lines" -le 500 || fail "README exceeds bounded public entry surface: $readme_lines"

require_text docs/architecture/system.md '# Implemented YVEX System'
require_text docs/architecture/compilation.md '## Runtime binding'
require_text docs/architecture/runtime.md '## Sessions and transactional state'
require_text docs/architecture/commands.md 'yvex.operator.registry.v1'
require_text docs/model-families/integration.md '# Model-Family Integration Contract'
require_text docs/contracts/artifacts.md '# Artifact and Admission Contract'
require_text docs/contracts/runtime.md 'A client connection is not a session.'
require_text docs/contracts/runtime.md 'no explicit exact request silently changes'
require_text docs/contracts/events-telemetry.md 'No consumer scrapes another renderer'
require_text docs/contracts/c-api.md '## Compiled Operator Registry Boundary'
require_text docs/openai-compatibility.md 'YVEX never executes application tools.'
require_text docs/operator-runbook.md '## First verified startup'
test ! -e ./yvexd || fail 'retired hidden server executable remains'
if test -x ./yvex; then
  help=$(./yvex)
  for command in 'chat' 'serve' 'host' 'model' 'inspect' 'help' 'version'
  do
    printf '%s\n' "$help" | grep -F "$command" >/dev/null ||
      fail "built yvex help lacks canonical command: $command"
  done
  for plumbing in 'engine' 'session' 'source' 'artifact' 'profile' 'compile' 'bench'
  do
    printf '%s\n' "$help" | grep -F "  $plumbing " >/dev/null &&
      fail "built yvex help exposes advanced root: $plumbing"
  done
  for retired in 'yvex run' 'yvex server'; do
    printf '%s\n' "$help" | grep -F "$retired" >/dev/null &&
      fail "built yvex help exposes retired command: $retired"
  done
  advanced=$(./yvex help --advanced)
  for command in 'yvex bench attention execute' 'yvex engine load [PROFILE]' \
                 'yvex source list' 'yvex artifact list' 'yvex profile list'
  do
    printf '%s\n' "$advanced" | grep -F "$command" >/dev/null ||
      fail "advanced help lacks canonical command: $command"
  done
  model_help=$(./yvex help model)
  for command in 'yvex model search' 'yvex model pull' 'yvex model prepare' \
                 'yvex model load' 'yvex model unload' 'yvex model push'
  do
    printf '%s\n' "$model_help" | grep -F "$command" >/dev/null ||
      fail "model help lacks porcelain command: $command"
  done
  session_help=$(./yvex help session)
  printf '%s\n' "$session_help" | grep -F 'yvex session cancel' >/dev/null ||
    fail 'session help lacks canonical cancel command'
  compile_help=$(./yvex help compile)
  printf '%s\n' "$compile_help" | grep -F 'yvex compile quant plan' >/dev/null ||
    fail 'compile help lacks canonical quant plan'
  printf '%s\n' "$help" | grep -F 'yvex graph' >/dev/null &&
    fail 'built yvex help exposes retired graph namespace'
fi

serve_help=$(./yvex serve --help)
printf '%s\n' "$serve_help" | grep -F -- 'operation: host.serve' >/dev/null ||
  fail 'serve help lacks foreground host operation'
printf '%s\n' "$serve_help" | grep -F -- 'engine load' >/dev/null &&
  fail 'serve help embeds engine administration'
engine_help=$(./yvex help engine)
printf '%s\n' "$engine_help" | grep -F -- 'yvex engine load [PROFILE]' >/dev/null ||
  fail 'engine help lacks explicit profile load'
printf '%s\n' "$engine_help" | grep -F -- 'yvex engine unload ENGINE' >/dev/null ||
  fail 'engine help lacks independent unload'
