#!/usr/bin/env sh
set -eu

project=PROJECT.md
tmp_base="${TMPDIR:-/tmp}/yvex-project-ledger.$$"
rows="${tmp_base}.rows"
all_ids="${tmp_base}.all-ids"
new_ids="${tmp_base}.new-ids"
recovered_ids="${tmp_base}.recovered-ids"
tracks="${tmp_base}.tracks"
expected_tracks="${tmp_base}.expected-tracks"
declared_global="${tmp_base}.declared-global"
actual_global="${tmp_base}.actual-global"
declared_dashboard="${tmp_base}.declared-dashboard"
actual_dashboard="${tmp_base}.actual-dashboard"
trap 'rm -f "${tmp_base}."*' EXIT

fail() {
  printf 'project ledger: %s\n' "$1" >&2
  exit 1
}

test -f "$project" || fail "missing $project"
test ! -e docs/spine.md || fail "shadow project authority exists: docs/spine.md"
test -z "$(find docs -maxdepth 1 -type d -name repair -print -quit)" ||
  fail "temporary project-control directory still exists"

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
/^## 9\. Complete Track\/Wave Ledger$/ {
  in_ledger = 1
  next
}
/^## 10\. Evidence Lanes$/ {
  in_ledger = 0
  next
}
!in_ledger { next }
/^### 9\.[0-9]+ TRACK\./ {
  track = $0
  sub(/^### 9\.[0-9]+ /, "", track)
  kind = ""
  next
}
/^#### First-Class Milestones$/ {
  kind = "milestone"
  next
}
/^#### Recovered And Subordinate Rows$/ {
  kind = "support"
  next
}
/^\| `[^`]+` \|/ {
  id = uncode($2)
  if (kind == "milestone") {
    rank = "milestone"
    state = uncode($4)
  } else if (kind == "support") {
    rank = uncode($3)
    state = uncode($5)
    consumer = trim($7)
    if (consumer == "") {
      printf "project ledger: missing consumer for %s\n", id > "/dev/stderr"
      exit 1
    }
  } else {
    printf "project ledger: row outside a ledger table: %s\n", id > "/dev/stderr"
    exit 1
  }
  print track "\t" id "\t" rank "\t" state
}
' "$project" > "$rows"

row_count=$(wc -l < "$rows" | tr -d ' ')
test "$row_count" -eq 681 || fail "expected 681 canonical IDs, found $row_count"

cut -f 2 "$rows" | LC_ALL=C sort > "$all_ids"
unique_count=$(uniq "$all_ids" | wc -l | tr -d ' ')
test "$unique_count" -eq "$row_count" ||
  fail "canonical IDs are not unique: $unique_count/$row_count"

duplicate=$(uniq -d "$all_ids" | head -n 1 || true)
test -z "$duplicate" || fail "duplicate canonical ID: $duplicate"

id_hash=$(sha256sum "$all_ids" | awk '{ print $1 }')
expected_id_hash=d11967228789599029f1c428c0e8f8a2b94b58ce65d84c61850a827101739af7
test "$id_hash" = "$expected_id_hash" ||
  fail "canonical ID set changed without an explicit migration: $id_hash"

cat > "$new_ids" <<'EOF'
V010.DOCS.REFOUNDATION.0
V010.PROJECT.RECOVERY.0
V010.PROJECT.RECOVERY.1
V010.DOCS.ARCHITECTURE.0
V010.PROJECT.COMPILATION.0
V010.DOCS.README.COMPILATION.0
V010.REPO.SEMANTIC.COMPRESSION.0
V010.REPO.C.CANONICALIZATION.0
V010.REBASE.DEEPSEEK.0
V010.SOURCE.PAYLOAD.STREAM.0
V010.MAP.GGUF.DEEPSEEK.0
V010.MODEL.TRANSFORM.IR.0
V010.GGUF.QTYPE.ABI.1
V010.GGUF.ARTIFACT.ABI.1
V010.GGUF.WRITER.1
V010.ARTIFACT.EMIT.DEEPSEEK.0
V010.GGUF.ROUNDTRIP.1
V010.ARTIFACT.SUPPORT.CUTOVER.0
V010.GGUF.LAYOUT.INTEGRITY.1
V010.MODEL.ARCH.IR.0
V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0
V010.TENSOR.COVERAGE.DEEPSEEK.0
V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0
V010.CUDA.FAILCLOSED.0
V010.GRAPH.DEEPSEEK.ATTENTION.0
V010.RUNTIME.DEEPSEEK.MOE.0
V010.GRAPH.DEEPSEEK.TRANSFORMER.0
V010.RUNTIME.DEEPSEEK.PREFILL.0
V010.RUNTIME.DEEPSEEK.KV.0
V010.RUNTIME.DEEPSEEK.DECODE.0
V010.RUNTIME.DEEPSEEK.LOGITS.0
V010.RUNTIME.SAMPLING.0
V010.RUNTIME.DEEPSEEK.TOKENIZER.0
V010.RUNTIME.DEEPSEEK.GENERATION.0
V010.CLI.DEEPSEEK.GENERATE.0
V010.EVAL.DEEPSEEK.0
V010.BENCH.DEEPSEEK.0
V010.RUNTIME.DEEPSEEK.ATTENTION.KV.0
V010.RUNTIME.DEEPSEEK.LOGITS.SAMPLING.0
POST010.COMPILATION.DAG.0
POST010.COMPILATION.REUSE.0
POST010.COMPILATION.VARIANTS.0
POST010.COMPILATION.HARDWARE.PROFILE.0
POST010.COMPILATION.WORKLOAD.PROFILE.0
POST010.COMPILATION.PRECISION.0
POST010.COMPILATION.PLACEMENT.0
POST010.COMPILATION.FEEDBACK.0
POST010.COMPILATION.PARETO.0
POST010.COMPILATION.RUNTIME.BINDING.0
POST010.COMPILATION.EXECUTION.STATE.0
EOF

LC_ALL=C sort -u "$new_ids" -o "$new_ids"
new_count=$(wc -l < "$new_ids" | tr -d ' ')
test "$new_count" -eq 50 || fail "expected 50 explicit new IDs, found $new_count"

missing_new=$(comm -23 "$new_ids" "$all_ids" | head -n 1 || true)
test -z "$missing_new" || fail "explicit new ID is absent: $missing_new"

comm -23 "$all_ids" "$new_ids" > "$recovered_ids"
recovered_count=$(wc -l < "$recovered_ids" | tr -d ' ')
test "$recovered_count" -eq 631 ||
  fail "expected 631 recovered IDs, found $recovered_count"

recovered_hash=$(sha256sum "$recovered_ids" | awk '{ print $1 }')
expected_recovered_hash=bac12219a65a8f9aa73f694160c418016bad55aece941d366b00f52ceaa2b9c4
test "$recovered_hash" = "$expected_recovered_hash" ||
  fail "recovered ID baseline changed without an explicit migration: $recovered_hash"

awk -F '\t' '
BEGIN { ok = 1 }
$3 !~ /^(milestone|capability|evidence|subtask|migration|future)$/ {
  printf "project ledger: invalid rank for %s: %s\n", $2, $3 > "/dev/stderr"
  ok = 0
}
$4 !~ /^(complete|active|partial|blocked|planned|reopened|not-measured|deferred|superseded)$/ {
  printf "project ledger: invalid state for %s: %s\n", $2, $4 > "/dev/stderr"
  ok = 0
}
$3 == "milestone" && $4 !~ /^(complete|active|partial|blocked|planned|not-measured)$/ { ok = 0 }
$3 == "capability" && $4 != "complete" { ok = 0 }
$3 == "evidence" && $4 !~ /^(complete|reopened)$/ { ok = 0 }
$3 == "subtask" && $4 != "planned" { ok = 0 }
$3 == "migration" && $4 != "superseded" { ok = 0 }
$3 == "future" && $4 != "deferred" { ok = 0 }
$4 == "active" && $3 != "milestone" { ok = 0 }
END { exit ok ? 0 : 1 }
' "$rows" || fail "rank/state contract violation"

milestone_count=$(awk -F '\t' '$3 == "milestone" { count++ } END { print count + 0 }' "$rows")
active_count=$(awk -F '\t' '$3 == "milestone" && $4 == "active" { count++ } END { print count + 0 }' "$rows")
test "$active_count" -eq 1 || fail "expected one active milestone, found $active_count"
active_id=$(awk -F '\t' '$3 == "milestone" && $4 == "active" { print $2 }' "$rows")

active_lines=$(grep -c '^Active Next: ' "$project")
test "$active_lines" -eq 1 || fail "expected one machine-readable Active Next, found $active_lines"
project_active=$(sed -n 's/^Active Next: \([^[:space:]]*\)$/\1/p' "$project")
test "$project_active" = "$active_id" ||
  fail "Active Next does not match active milestone: $project_active/$active_id"

awk -F '|' '
function trim(value) {
  gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
  return value
}
/^### 7\.1 Global Counts$/ { in_counts = 1; next }
/^### 7\.2 Per-Track Counts$/ { in_counts = 0 }
in_counts && /^\| (Recovered IDs|Explicit new IDs|Canonical IDs|First-class milestones|State: [a-z-]+) \|/ {
  print trim($2) "\t" trim($3)
}
' "$project" | LC_ALL=C sort > "$declared_global"

{
  printf 'Recovered IDs\t%s\n' "$recovered_count"
  printf 'Explicit new IDs\t%s\n' "$new_count"
  printf 'Canonical IDs\t%s\n' "$row_count"
  printf 'First-class milestones\t%s\n' "$milestone_count"
  for state in complete active partial blocked planned reopened deferred superseded not-measured; do
    count=$(awk -F '\t' -v wanted="$state" '$4 == wanted { count++ } END { print count + 0 }' "$rows")
    printf 'State: %s\t%s\n' "$state" "$count"
  done
} | LC_ALL=C sort > "$actual_global"

cmp -s "$declared_global" "$actual_global" || {
  diff -u "$declared_global" "$actual_global" >&2 || true
  fail "declared global counts do not match the ledger"
}

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
/^### 7\.2 Per-Track Counts$/ { in_dashboard = 1; next }
/^### 7\.3 Stable Track Names$/ { in_dashboard = 0 }
in_dashboard && /^\| `TRACK\./ {
  print uncode($2) "\t" trim($3) "\t" trim($4) "\t" trim($5) "\t" trim($6) "\t" trim($7) "\t" trim($8)
}
' "$project" | LC_ALL=C sort > "$declared_dashboard"

awk -F '\t' '
FNR == NR { explicit_new[$1] = 1; next }
{
  track = $1
  id = $2
  rank = $3
  state = $4
  canonical[track]++
  if (!(id in explicit_new)) recovered[track]++
  if (rank == "milestone") milestone[track, state]++
  else if (state == "complete") complete_support[track]++
  else if (state == "planned" || state == "reopened") open_support[track]++
  else if (state == "superseded" || state == "deferred") retired[track]++
}
END {
  for (track in canonical) {
    vector = (milestone[track, "complete"] + 0) "/" (milestone[track, "active"] + 0) "/" (milestone[track, "partial"] + 0) "/" (milestone[track, "planned"] + 0) "/" (milestone[track, "blocked"] + 0) "/" (milestone[track, "not-measured"] + 0)
    print track "\t" recovered[track] + 0 "\t" canonical[track] + 0 "\t" vector "\t" complete_support[track] + 0 "\t" open_support[track] + 0 "\t" retired[track] + 0
  }
}
' "$new_ids" "$rows" | LC_ALL=C sort > "$actual_dashboard"

cmp -s "$declared_dashboard" "$actual_dashboard" || {
  diff -u "$declared_dashboard" "$actual_dashboard" >&2 || true
  fail "per-track dashboard counts do not match the ledger"
}

grep '^### 9\.[0-9][0-9]* TRACK\.' "$project" |
  sed 's/^### 9\.[0-9][0-9]* //' > "$tracks"

cat > "$expected_tracks" <<'EOF'
TRACK.SCOPE
TRACK.SOURCE
TRACK.MAP
TRACK.COMPILATION
TRACK.QUANT
TRACK.ARTIFACT
TRACK.INTEGRITY
TRACK.MODEL
TRACK.TENSOR
TRACK.RESIDENCY
TRACK.RUNTIME
TRACK.BACKEND
TRACK.GRAPH
TRACK.PREFILL
TRACK.KV
TRACK.DECODE
TRACK.LOGITS
TRACK.SAMPLING
TRACK.TOKENIZER
TRACK.GENERATION
TRACK.OPERATOR
TRACK.SERVE
TRACK.EVAL
TRACK.BENCH
TRACK.RELEASE
TRACK.POST010
EOF

track_count=$(wc -l < "$tracks" | tr -d ' ')
test "$track_count" -eq 26 || fail "expected 26 canonical tracks, found $track_count"
cmp -s "$tracks" "$expected_tracks" || fail "stable 26-track order changed"

runtime_row_count=$(awk -F '\t' '$2 ~ /^V010\.RUNTIME\.([0-9]|1[0-7])$/ { count++ } END { print count + 0 }' "$rows")
test "$runtime_row_count" -eq 18 ||
  fail "expected 18 canonical common runtime rows, found $runtime_row_count"
runtime_wrong_owner=$(awk -F '\t' '
$2 ~ /^V010\.RUNTIME\.([0-9]|1[0-7])$/ && $1 != "TRACK.RUNTIME" { print $2; exit }
' "$rows")
test -z "$runtime_wrong_owner" ||
  fail "common runtime row has wrong owner: $runtime_wrong_owner"
generation_runtime_row=$(awk -F '\t' '
$1 == "TRACK.GENERATION" && $2 ~ /^V010\.RUNTIME\.([0-9]|1[0-7])$/ { print $2; exit }
' "$rows")
test -z "$generation_runtime_row" ||
  fail "generation retains migrated common runtime row: $generation_runtime_row"
awk -F '\t' '
$1 == "TRACK.RUNTIME" && $2 == "V010.RUNTIME.1" &&
    $3 == "milestone" && $4 == "complete" { found = 1 }
END { exit found ? 0 : 1 }
' "$rows" || fail "V010.RUNTIME.1 is not the completed common runtime milestone"

for required_id in \
  MODEL.CLASS.QWEN.0 \
  MODEL.CLASS.GEMMA.0 \
  TENSOR.COLLECTION.QWEN.0 \
  TENSOR.COLLECTION.GEMMA.0 \
  OWI.TARGETS.QWEN.0 \
  OWI.TARGETS.GEMMA.0 \
  MODELS.SOURCE.MAP.HANDOFF.0 \
  V010.GGUF.QTYPE.ABI.0 \
  V010.GGUF.ARTIFACT.ABI.0 \
  V010.PROJECT.RECOVERY.1 \
  V010.PROJECT.COMPILATION.0 \
  V010.DOCS.README.COMPILATION.0 \
  V010.MODEL.TRANSFORM.IR.0 \
  POST010.COMPILATION.RUNTIME.BINDING.0
do
  grep -F "$(printf '\t%s\t' "$required_id")" "$rows" >/dev/null ||
    fail "missing required recovered ID: $required_id"
done

grep -F 'Qwen, Gemma, and dense/common work already implemented remains active' "$project" >/dev/null ||
  fail "multi-family engineering scope is missing"
grep -F 'DeepSeek-V4-Flash is the only model whose complete source-to-text chain closes' "$project" >/dev/null ||
  fail "exclusive v0.1 release target is missing"

grep -F '### 9.4 TRACK.COMPILATION' "$project" >/dev/null ||
  fail "compilation track is missing"
grep -F '| `V010.DOCS.README.COMPILATION.0` | project | `complete` |' "$project" >/dev/null ||
  fail "README compilation milestone is not complete"
grep -F '| `V010.MODEL.TRANSFORM.IR.0` | DeepSeek + common plan | `complete` |' "$project" >/dev/null ||
  fail "transformation IR milestone is not complete"
grep -F '| `V010.QUANT.2` | common + DeepSeek roles | `complete` |' "$project" >/dev/null ||
  fail "quantization is not complete"
grep -F '| `V010.GGUF.WRITER.1` | common | `complete` |' "$project" >/dev/null ||
  fail "GGUF writer is not complete"
grep -F '| `V010.ARTIFACT.EMIT.DEEPSEEK.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "complete DeepSeek artifact emission is not complete"
grep -F '| `V010.GGUF.ROUNDTRIP.1` | common + DeepSeek artifact | `complete` |' "$project" >/dev/null ||
  fail "GGUF roundtrip is not complete"
grep -F '| `V010.ARTIFACT.SUPPORT.CUTOVER.0` | common | `complete` |' "$project" >/dev/null ||
  fail "complete-artifact support cutover is not complete"
grep -F '| `V010.ARTIFACT.MATERIALIZE.0` | common | `complete` |' "$project" >/dev/null ||
  fail "artifact materialization is not complete"
grep -F '| `V010.ARTIFACT.MATERIALIZE.DEEPSEEK.0` | DeepSeek / DGX Spark | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek materialization is not complete"
grep -F '| `V010.RUNTIME.DESCRIPTOR.GGUF.0` | common | `complete` |' "$project" >/dev/null ||
  fail "common runtime descriptor is not complete"
grep -F '| `V010.RUNTIME.DESCRIPTOR.DEEPSEEK.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek runtime descriptor is not complete"
grep -F '| `V010.REPO.SEMANTIC.COMPRESSION.0` | project | `complete` |' "$project" >/dev/null ||
  fail "repository semantic compression is not complete"
grep -F '| `V010.REPO.C.CANONICALIZATION.0` | project | `complete` |' "$project" >/dev/null ||
  fail "repository C canonicalization is not complete"
grep -F '| `V010.GRAPH.DEEPSEEK.ATTENTION.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek attention execution is not complete"
grep -F '| `V010.CLI.GRAPH.0` | DeepSeek + common operator | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek attention operator reachability is not complete"
grep -F '| `V010.RUNTIME.1` | common | `complete` |' "$project" >/dev/null ||
  fail "common runtime model/session plane is not complete"
grep -F '| `V010.RUNTIME.DEEPSEEK.KV.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek KV is not complete"
grep -F '| `V010.RUNTIME.DEEPSEEK.KV.0` | DeepSeek | `complete` | Own exact session-local DeepSeek state for every attention class and layer through checked CPU/CUDA residency, all-layer transactional publication, causal read-after-write, capacity/position continuity, clear/reuse, invalidation, and release. | V010.RUNTIME.1 | current |' "$project" >/dev/null ||
  fail "DeepSeek KV does not depend on the common runtime milestone"
grep -F '| `V010.RUNTIME.DEEPSEEK.PREFILL.0` | DeepSeek | `complete` | Admit exact activation chunks for all 43 attention layers, execute them on CPU or GB10 CUDA, and atomically publish persistent state and committed-prefix progression without claiming prompt or full-model prefill. | V010.RUNTIME.DEEPSEEK.KV.0 | current |' "$project" >/dev/null ||
  fail "DeepSeek activation prefill is not complete"
grep -F '| `V010.RUNTIME.DEEPSEEK.MOE.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek MoE is not complete"
grep -F '| `V010.GRAPH.DEEPSEEK.TRANSFORMER.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek transformer composition is not complete"
grep -F '| `V010.RUNTIME.DEEPSEEK.DECODE.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek repeated decode is not complete"
grep -F '| `V010.RUNTIME.DEEPSEEK.LOGITS.0` | DeepSeek | `complete` |' "$project" >/dev/null ||
  fail "DeepSeek logits is not complete"
grep -F '| `V010.RUNTIME.SAMPLING.0` | common | `active` |' "$project" >/dev/null ||
  fail "sampling is not active after logits closure"
grep -F '| V010.MODEL.TRANSFORM.IR.0 | recovered/promoted |' "$project" >/dev/null ||
  fail "quantization does not depend on the transformation IR"

sequence=$(awk '
/^```text$/ { block = ""; in_block = 1; next }
in_block && /^```$/ {
  if (block ~ /V010\.DOCS\.README\.COMPILATION\.0/ &&
      block ~ /V010\.MODEL\.TRANSFORM\.IR\.0/ &&
      block ~ /V010\.QUANT\.2/ &&
      block ~ /V010\.GGUF\.WRITER\.1/ &&
      block ~ /V010\.REPO\.SEMANTIC\.COMPRESSION\.0/ &&
      block ~ /V010\.REPO\.C\.CANONICALIZATION\.0/ &&
      block ~ /V010\.GRAPH\.DEEPSEEK\.ATTENTION\.0/ &&
      block ~ /V010\.CLI\.GRAPH\.0/ &&
      block ~ /V010\.RUNTIME\.1:/ &&
      block ~ /V010\.RUNTIME\.DEEPSEEK\.KV\.0/) {
    print block
    exit
  }
  in_block = 0
  next
}
in_block { block = block $0 "\n" }
' "$project")
test -n "$sequence" || fail "compilation critical-path block is missing"
printf '%s' "$sequence" | awk '
BEGIN { expected = 1 }
/V010\.DOCS\.README\.COMPILATION\.0/ { if (expected != 1) exit 1; expected = 2 }
/V010\.MODEL\.TRANSFORM\.IR\.0/ { if (expected != 2) exit 1; expected = 3 }
/V010\.QUANT\.2/ { if (expected != 3) exit 1; expected = 4 }
/V010\.GGUF\.WRITER\.1/ { if (expected != 4) exit 1; expected = 5 }
/V010\.REPO\.SEMANTIC\.COMPRESSION\.0/ { if (expected != 5) exit 1; expected = 6 }
/V010\.REPO\.C\.CANONICALIZATION\.0/ { if (expected != 6) exit 1; expected = 7 }
/V010\.GRAPH\.DEEPSEEK\.ATTENTION\.0/ { if (expected != 7) exit 1; expected = 8 }
/V010\.CLI\.GRAPH\.0/ { if (expected != 8) exit 1; expected = 9 }
/V010\.RUNTIME\.1:/ { if (expected != 9) exit 1; expected = 10 }
/V010\.RUNTIME\.DEEPSEEK\.KV\.0/ { if (expected != 10) exit 1; expected = 11 }
END { exit expected == 11 ? 0 : 1 }
' || fail "compilation/repository/attention critical-path order is invalid"

echo "project ledger: ok (tracks=$track_count recovered=$recovered_count ids=$row_count milestones=$milestone_count active=$active_id)"
