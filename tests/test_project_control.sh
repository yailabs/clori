#!/usr/bin/env sh
set -eu

fail() {
  printf 'project control: %s\n' "$1" >&2
  exit 1
}

require_file() {
  test -f "$1" || fail "missing file: $1"
}

require_text() {
  grep -nF -- "$2" "$1" >/dev/null || fail "$1 missing required text: $2"
}

roadmap=ROADMAP.md

require_file "$roadmap"
require_file CONTRIBUTING.md
require_file docs/decisions/README.md
require_file docs/decisions/0001-public-project-control.md
require_file docs/decisions/0003-documentation-architecture.md
require_file docs/development/documentation-policy.md
require_file docs/milestones/code-commentary.md
require_file docs/milestones/command-architecture.md
require_file docs/milestones/documentation-architecture.md
require_file docs/milestones/runtime-console-repl.md
require_file .github/ISSUE_TEMPLATE/bug_report.yml
require_file .github/ISSUE_TEMPLATE/engineering_change.yml
require_file .github/ISSUE_TEMPLATE/config.yml
require_file .github/pull_request_template.md

test ! -e PROJECT.md || fail 'retired PROJECT.md exists'
test ! -e docs/spine.md || fail 'retired project spine exists'

require_text "$roadmap" 'Status: living public project control'
require_text "$roadmap" 'This file is the sole live authority'
require_text "$roadmap" '| `V010.PROJECT.CONTROL.PUBLIC.0` | `complete` |'
require_text "$roadmap" '| `V010.OPERATOR.COMMAND.ARCHITECTURE.0` | `complete` |'
require_text "$roadmap" '| `V010.DOCS.INFORMATION.ARCHITECTURE.0` | `complete` |'
require_text "$roadmap" '| `V010.REPO.CODE.COMMENTARY.0` | `complete` |'
require_text "$roadmap" '| `V010.OPERATOR.REPL.CONSOLE.0` | `complete` |'
require_text "$roadmap" '| `V010.RUNTIME.DEEPSEEK.GB10.OPTIMIZATION.0` | `active` |'
require_text "$roadmap" '| `V010.EVAL.DEEPSEEK.0` | `blocked` |'
require_text "$roadmap" '| `V010.BENCH.DEEPSEEK.0` | `not-measured` |'
require_text "$roadmap" '| `V010.RELEASE.0` | `blocked` |'
require_text "$roadmap" '| `V010.OPERATOR.COMMAND.CONSOLE.0` | `superseded` |'
require_text "$roadmap" 'model_behavior_evaluation_ready=0'
require_text "$roadmap" 'mature_repl_console_ready=1'
require_text "$roadmap" 'full_model_release_benchmark_ready=0'
require_text "$roadmap" 'release_qualification_ready=0'
require_text "$roadmap" 'V010.RUNTIME.DEEPSEEK.PERFORMANCE.0` remains `partial`'
require_text "$roadmap" '447257dca7b122bafbddb86073d55eaa7be9513f'

tmp_base="${TMPDIR:-/tmp}/yvex-project-control.$$"
rows="$tmp_base.rows"
ids="$tmp_base.ids"
trap 'rm -f "$rows" "$ids"' EXIT HUP INT TERM

awk -F '|' '
function trim(value) {
  gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
  return value
}
function uncode(value) {
  value = trim(value)
  gsub(/^`|`$/, "", value)
  return value
}
/^## Current sequence$/ { in_sequence = 1; next }
/^Active Next: / { in_sequence = 0 }
in_sequence && /^\| [0-9]+ \| `V010\./ {
  print uncode($3) "\t" uncode($4)
}
' "$roadmap" > "$rows"

row_count=$(wc -l < "$rows" | tr -d ' ')
test "$row_count" -eq 9 || fail "expected 9 current milestones, found $row_count"

cut -f 1 "$rows" | LC_ALL=C sort > "$ids"
unique_count=$(uniq "$ids" | wc -l | tr -d ' ')
test "$unique_count" -eq "$row_count" || fail 'current milestone IDs are not unique'

active_count=$(awk -F '\t' '$2 == "active" { count++ } END { print count + 0 }' "$rows")
test "$active_count" -eq 1 || fail "expected one active milestone, found $active_count"
active_id=$(awk -F '\t' '$2 == "active" { print $1 }' "$rows")

active_lines=$(grep -c '^Active Next: ' "$roadmap")
test "$active_lines" -eq 1 || fail "expected one Active Next, found $active_lines"
active_next=$(sed -n 's/^Active Next: \([^[:space:]]*\)$/\1/p' "$roadmap")
test "$active_next" = "$active_id" ||
  fail "Active Next does not match active milestone: $active_next/$active_id"

all_active_files=$(find . -path './.git' -prune -o -path './build' -prune -o \
  -type f -name '*.md' -exec grep -l '^Active Next: ' {} + | LC_ALL=C sort)
test "$all_active_files" = './ROADMAP.md' ||
  fail "Active Next exists outside ROADMAP.md: $all_active_files"

roadmap_lines=$(wc -l < "$roadmap" | tr -d ' ')
test "$roadmap_lines" -le 350 || fail "ROADMAP.md exceeds 350 lines: $roadmap_lines"

require_text CONTRIBUTING.md '## Before opening work'
require_text CONTRIBUTING.md '## Development order'
require_text CONTRIBUTING.md '## Tests'
require_text CONTRIBUTING.md '## Commit and pull request'
require_text CONTRIBUTING.md '## Capability and evidence language'
require_text CONTRIBUTING.md 'ROADMAP.md'

require_text docs/decisions/README.md 'Current macro state remains in'
require_text docs/decisions/0001-public-project-control.md '## Decision'
require_text docs/decisions/0001-public-project-control.md 'The former complete ledger is removed'
require_text docs/decisions/0003-documentation-architecture.md '## Decision'
require_text docs/development/documentation-policy.md '`ROADMAP.md` is the only live macro control surface'
require_text docs/milestones/code-commentary.md 'Status source:'
require_text docs/milestones/documentation-architecture.md 'Status source:'

issue_count=$(find .github/ISSUE_TEMPLATE -maxdepth 1 -type f -name '*.yml' |
  wc -l | tr -d ' ')
test "$issue_count" -eq 3 || fail "unexpected issue-template count: $issue_count"
require_text .github/ISSUE_TEMPLATE/config.yml 'blank_issues_enabled: false'
require_text .github/pull_request_template.md '## Claims and progression'

for document in ROADMAP.md CONTRIBUTING.md docs/decisions/README.md \
  docs/decisions/0001-public-project-control.md \
  docs/decisions/0003-documentation-architecture.md \
  docs/development/documentation-policy.md \
  docs/milestones/documentation-architecture.md
do
  base=$(dirname "$document")
  for target in $(grep -oE '\]\([^)]+\)' "$document" |
    sed 's/^](//; s/)$//' | grep -Ev '^(https?://|mailto:|#)' || true)
  do
    target=${target%%#*}
    test -e "$base/$target" ||
      fail "$document local link does not resolve: $target"
  done
done

living_docs=$(awk -F '\t' 'NR > 1 && $5 == "living" && $1 != "ROADMAP.md" { print $1 }' \
  config/documentation_owners.tsv)
if printf '%s\n' "$living_docs" | xargs rg -n 'PROJECT\.md'; then
  fail 'live documentation still points at retired PROJECT.md'
fi

printf 'project control: ok (milestones=%s active=%s roadmap_lines=%s)\n' \
  "$row_count" "$active_id" "$roadmap_lines"
