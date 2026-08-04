#!/bin/sh
#
# YVEX - quant-job CLI smoke test
#
# File: tests/cli/quant_job.sh
# Layer: test

set -eu

. tests/support/cleanup.sh

YVEX_BIN=${YVEX_BIN:-./yvex}
OUT_DIR=${YVEX_TEST_OUT_DIR:-build/tests/quant-job-cli}

fail() {
    printf 'FAIL: %s\n' "$1" >&2
    exit 1
}

yvex_test_cleanup "$OUT_DIR"
mkdir -p "$OUT_DIR/native"

printf 'fake-template\n' > "$OUT_DIR/template.gguf"
printf '#!/bin/sh\nexit 0\n' > "$OUT_DIR/tool"
chmod +x "$OUT_DIR/tool"

"$YVEX_BIN" compile quant job create \
  --name test-job \
  --arch deepseek4 \
  --tool external \
  --tool-path "$OUT_DIR/tool" \
  --native-source "$OUT_DIR/native" \
  --template "$OUT_DIR/template.gguf" \
  --out-gguf "$OUT_DIR/out.gguf" \
  --log "$OUT_DIR/job.log" \
  --status ready \
  --command "test command" \
  --out "$OUT_DIR/job.json" > "$OUT_DIR/create.out" 2> "$OUT_DIR/create.err" || fail "create failed"

test -f "$OUT_DIR/job.json" || fail "manifest missing"
grep 'compile quant job: written' "$OUT_DIR/create.out" >/dev/null || fail "missing create heading"
grep 'tool_exists: yes' "$OUT_DIR/create.out" >/dev/null || fail "missing tool exists"
grep 'source_exists: yes' "$OUT_DIR/create.out" >/dev/null || fail "missing source exists"
grep 'template_exists: yes' "$OUT_DIR/create.out" >/dev/null || fail "missing template exists"
grep 'output_exists: no' "$OUT_DIR/create.out" >/dev/null || fail "missing output exists"
grep 'status: quant-job-written' "$OUT_DIR/create.out" >/dev/null || fail "missing create status"

"$YVEX_BIN" compile quant job inspect --manifest "$OUT_DIR/job.json" > "$OUT_DIR/inspect.out" 2> "$OUT_DIR/inspect.err" || fail "inspect failed"
grep 'compile quant job: inspect' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect heading"
grep 'status: quant-job-manifest' "$OUT_DIR/inspect.out" >/dev/null || fail "missing inspect status"

"$YVEX_BIN" compile quant job validate --manifest "$OUT_DIR/job.json" > "$OUT_DIR/validate.out" 2> "$OUT_DIR/validate.err" || fail "validate failed"
grep 'compile quant job: validate' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate heading"
grep 'status: quant-job-valid' "$OUT_DIR/validate.out" >/dev/null || fail "missing validate status"

"$YVEX_BIN" compile quant job --help > "$OUT_DIR/help.out" 2> "$OUT_DIR/help.err" || fail "help failed"
grep 'usage: yvex compile quant job' "$OUT_DIR/help.out" >/dev/null || fail "missing help"
