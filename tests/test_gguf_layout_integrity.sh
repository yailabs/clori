#!/usr/bin/env sh
set -eu

# One domain owner validates ordered padded layout. Consumers call it directly
# instead of reconstructing range policy or routing through a report projection.
test -f include/yvex/gguf.h
test -f src/gguf/layout_integrity.c
grep -nF 'int yvex_gguf_layout_validate(' include/yvex/gguf.h >/dev/null
grep -nF 'yvex_gguf_layout_validate(artifact, gguf' \
  src/artifact/integrity.c >/dev/null
test ! -e src/gguf/report.c
grep -nF 'yvex_gguf_layout_validate(artifact, gguf' \
  tests/unit/gguf_artifact_abi.c >/dev/null
# Runtime stays compilation-free: the binding carries the sealed complete
# admission, then runtime authenticates that admission on its retained handle
# before constructing the GGUF and tensor views. Family-specific admission
# remains in the preparation plane checked immediately below.
grep -nF '&model->admission, &binding_failure, err)' \
  src/runtime/core.c >/dev/null
grep -nF 'yvex_artifact_admission_authenticate(' src/runtime/core.c >/dev/null
grep -nF 'yvex_gguf_open(&model->gguf, model->artifact' \
  src/runtime/core.c >/dev/null
grep -nF 'yvex_tensor_table_from_gguf(&model->tensors, model->gguf' \
  src/runtime/core.c >/dev/null
grep -nF 'yvex_artifact_integrity_validate' \
  src/model/artifacts/gate.c >/dev/null
grep -nF 'yvex_gguf_layout_validate(artifact, gguf' \
  src/model/core.c >/dev/null

grep -nF '(out->alignment & (out->alignment - 1u)) != 0u' \
  src/gguf/layout_integrity.c >/dev/null
grep -nF 'tensor_payload_bytes_read = 0ull' \
  src/gguf/layout_integrity.c >/dev/null
grep -nF '72317ull' tests/unit/gguf_layout_integrity.c >/dev/null
grep -nF '24u' tests/unit/gguf_layout_integrity.c >/dev/null
grep -nF 'YVEX_GGUF_LAYOUT_NONCANONICAL_TAIL' \
  tests/unit/gguf_layout_integrity.c >/dev/null

if grep -nF 'strstr' src/gguf/layout_integrity.c \
  src/artifact/integrity.c >/dev/null; then
  echo 'GGUF layout guard: semantic policy depends on error strings' >&2
  exit 1
fi
if grep -nE '\b(qsort|heapsort|mergesort)\s*\(' \
  src/gguf/layout_integrity.c >/dev/null; then
  echo 'GGUF layout guard: directory order is being normalized before admission' >&2
  exit 1
fi
if grep -nF 'yvex_artifact_identity_read(path' \
  src/artifact/integrity.c >/dev/null; then
  echo 'GGUF layout guard: integrity reopened the path for implicit hashing' >&2
  exit 1
fi
grep -nF 'yvex_artifact_identity_read_open' \
  src/artifact/integrity.c >/dev/null
