#!/usr/bin/env sh
set -eu

test -x build/tests/test

out="${TMPDIR:-/tmp}/yvex-gguf-artifact-abi.$$"
trap 'rm -f "$out"' EXIT

build/tests/test >"$out" 2>&1
grep -nF 'test: gguf_artifact_abi' "$out" >/dev/null

# The operational reader and layout owner are the only structural authority.
# A report-only parser projection must not return as a second admission path.
grep -nF 'int yvex_artifact_read_at(' include/yvex/artifact.h >/dev/null
grep -nF 'int yvex_gguf_open_ex(' include/yvex/gguf.h >/dev/null
test ! -e src/gguf/report.c
if rg -nF 'yvex_gguf_artifact_abi_report_build' \
  src include tests/unit tests/live >/dev/null; then
  echo 'artifact ABI guard: report-only structural authority returned' >&2
  exit 1
fi
grep -nF 'yvex_gguf_open_ex(gguf' tests/unit/gguf_artifact_abi.c >/dev/null
grep -nF 'yvex_gguf_layout_validate(artifact, gguf' \
  tests/unit/gguf_artifact_abi.c >/dev/null

if grep -nE '\b(fseek|ftell|fread)\b' src/artifact/core.c >/dev/null; then
  echo 'artifact ABI guard: whole-file stdio path returned' >&2
  exit 1
fi
if grep -nF 'yvex_artifact_data' src/gguf/*.c >/dev/null; then
  echo 'artifact ABI guard: GGUF parser depends on whole-file artifact data' >&2
  exit 1
fi
if grep -nF 'strstr' src/gguf/reader.c >/dev/null; then
  echo 'artifact ABI guard: reader failure policy depends on error strings' >&2
  exit 1
fi
if grep -nE 'for \([^;]+;[^;]+< i;' src/artifact/integrity.c >/dev/null; then
  echo 'artifact ABI guard: integrity restored quadratic duplicate detection' >&2
  exit 1
fi
if grep -R -nF 'required_key_count' src/gguf include/yvex >/dev/null; then
  echo 'artifact ABI guard: synthetic required-key count returned' >&2
  exit 1
fi

grep -nF '72317ull' tests/unit/gguf_artifact_abi.c >/dev/null
grep -nF '129280ull' tests/unit/gguf_artifact_abi.c >/dev/null
grep -nF '160ull * 1024ull * 1024ull * 1024ull' \
  tests/unit/gguf_artifact_abi.c >/dev/null
