#!/usr/bin/env sh
set -eu

test -f include/yvex/qtype.h

grep -nF 'af97976c7810cdabb1863172f31c432dab767de7' \
  include/yvex/qtype.h docs/reference/verified-inference.md >/dev/null
grep -nF 'yvex_gguf_qtype_tensor_storage' src/gguf/qtype.c >/dev/null
grep -nF 'yvex_gguf_qtype_tensor_storage' src/gguf/core.c >/dev/null
grep -nF 'yvex_gguf_qtype_tensor_storage' \
  src/gguf/layout_integrity.c >/dev/null
test ! -e src/gguf/report.c
grep -nF 'layout.raw_tensor_bytes == 36ull' \
  tests/unit/gguf_qtype_abi.c >/dev/null
grep -nF 'yvex_dtype_tensor_storage_bytes' \
  src/model/core.c \
  src/artifact/integrity.c \
  src/graph/memory_plan.c >/dev/null
grep -nF 'yvex_gguf_qtype_geometry_find' src/gguf/conversion.c >/dev/null
grep -nF 'yvex_gguf_qtype_geometry_find_by_name' src/gguf/tools.c >/dev/null
if grep -nE 'block_elems|block_bytes|scalar_bytes|is_supported_for_storage_accounting' \
    include/yvex/model.h src/model/core.c; then
  echo 'dtype owner must project canonical GGUF storage geometry' >&2
  exit 1
fi

if grep -nF 'ceil-block' src/gguf/qtype.c src/model/core.c; then
  echo 'partial block padding must not re-enter qtype accounting' >&2
  exit 1
fi

if grep -nF 'static const char *ggml_type_name' src/gguf/core.c; then
  echo 'GGUF parser must use the canonical qtype identity owner' >&2
  exit 1
fi

if grep -nF 'qtype_to_ggml' src/gguf/conversion.c; then
  echo 'conversion must not recreate a qtype ID/width registry' >&2
  exit 1
fi

if grep -nF 'yvex_dtype_storage_bytes' \
    src/artifact/integrity.c src/graph/memory_plan.c; then
  echo 'multidimensional consumers must not use the single-row compatibility API' >&2
  exit 1
fi
