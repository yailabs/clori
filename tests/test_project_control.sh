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
required_files="
$roadmap
CONTRIBUTING.md
docs/decisions/README.md
docs/decisions/0001-public-project-control.md
docs/development/agentic-engineering.md
.github/ISSUE_TEMPLATE/bug_report.yml
.github/ISSUE_TEMPLATE/engineering_change.yml
.github/ISSUE_TEMPLATE/config.yml
.github/pull_request_template.md
"
for file in $required_files
do
  require_file "$file"
done

test ! -e PROJECT.md || fail 'retired PROJECT.md exists'
test ! -d docs/milestones || fail 'retired milestone plans remain in the current tree'

require_text "$roadmap" 'Status: living public project control'
require_text "$roadmap" 'This file is the sole live authority'
require_text "$roadmap" 'Active Next:'
require_text "$roadmap" 'model_behavior_evaluation_ready=0'
require_text "$roadmap" 'full_model_release_benchmark_ready=0'
require_text "$roadmap" 'release_qualification_ready=0'

tmp_base="${TMPDIR:-/tmp}/yvex-project-control.$$"
rows="$tmp_base.rows"
trap 'rm -f "$rows"' EXIT HUP INT TERM

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
in_sequence && /^\| [0-9]+ \| `[A-Z0-9.]+`/ {
  print uncode($3) "\t" uncode($4)
}
' "$roadmap" > "$rows"

row_count=$(wc -l < "$rows" | tr -d ' ')
test "$row_count" -gt 0 || fail 'current sequence is empty'
unique_count=$(cut -f 1 "$rows" | LC_ALL=C sort -u | wc -l | tr -d ' ')
test "$unique_count" -eq "$row_count" || fail 'current milestone IDs are not unique'
active_count=$(awk -F '\t' '$2 == "active" { count++ } END { print count + 0 }' "$rows")
test "$active_count" -eq 1 || fail "expected one active milestone, found $active_count"
active_id=$(awk -F '\t' '$2 == "active" { print $1 }' "$rows")
active_next=$(sed -n 's/^Active Next: \([^[:space:]]*\)$/\1/p' "$roadmap")
test "$active_next" = "$active_id" ||
  fail "Active Next does not match active milestone: $active_next/$active_id"

all_active_files=$(git ls-files --cached --others --exclude-standard -- '*.md' | while IFS= read -r file; do
  test -f "$file" || continue
  grep -l '^Active Next: ' "$file" || :
done | LC_ALL=C sort)
test "$all_active_files" = 'ROADMAP.md' ||
  fail "Active Next exists outside ROADMAP.md: $all_active_files"

roadmap_lines=$(wc -l < "$roadmap" | tr -d ' ')
test "$roadmap_lines" -le 350 || fail "ROADMAP.md exceeds 350 lines: $roadmap_lines"

require_text CONTRIBUTING.md '## Before opening work'
require_text CONTRIBUTING.md '## Development order'
require_text CONTRIBUTING.md '## Tests'
require_text CONTRIBUTING.md '## Commit and pull request'
require_text CONTRIBUTING.md 'ROADMAP.md'
require_text docs/decisions/README.md 'Current macro state remains in'
require_text docs/decisions/0001-public-project-control.md '## Decision'
require_text docs/development/agentic-engineering.md 'The only live macro project-control surface'

issue_count=$(find .github/ISSUE_TEMPLATE -maxdepth 1 -type f -name '*.yml' | wc -l | tr -d ' ')
test "$issue_count" -eq 3 || fail "unexpected issue-template count: $issue_count"
require_text .github/ISSUE_TEMPLATE/config.yml 'blank_issues_enabled: false'
require_text .github/pull_request_template.md '## Claims and progression'

printf 'project control: ok (milestones=%s active=%s roadmap_lines=%s)\n' "$row_count" "$active_id" "$roadmap_lines"
