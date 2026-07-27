#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

sh tests/test_source_ownership.sh
sh tests/test_repository_layout.sh
sh tests/test_architecture_boundaries.sh

required_paths='
src/cli/main.c
src/core/status.c
include/yvex/internal/source.h
include/yvex/internal/source_payload.h
include/yvex/internal/families/deepseek_v4.h
src/model/families/deepseek_v4.c
src/model/compilation/ir.c
src/model/compilation/binding.c
src/runtime/descriptor.c
src/graph/private.h
src/graph/attention.c
src/graph/numeric.c
src/graph/families/deepseek_v4.c
src/backend/cuda/families/deepseek_v4.c
src/gguf/reader.c
src/gguf/writer.c
src/artifact/materialize.c
src/runtime/core.c
'
for path in $required_paths; do
  test -f "$path" || {
    echo "source layout: missing canonical owner: $path" >&2
    exit 1
  }
done

forbidden_paths='
src/model/architecture
src/model/target/deepseek_tensor_coverage.c
src/model/target/deepseek_gguf_map.c
src/model/compilation/deepseek_transform_ir.c
src/graph/deepseek_attention_plan.c
src/graph/deepseek_attention_execute.c
src/graph/deepseek_attention_sink.c
src/graph/deepseek_attention_numeric.c
src/graph/deepseek_attention_internal.c
src/backend/cuda/deepseek_attention.c
'
for path in $forbidden_paths; do
  test ! -e "$path" || {
    echo "source layout: superseded owner remains: $path" >&2
    exit 1
  }
done

if find src include -type f \( -name 'yvex_*.c' -o -name 'yvex_*.h' -o -name 'yvex_*.cu' \) -print | grep .; then
  echo "source layout: project prefix leaked into an owned filename" >&2
  exit 1
fi

if find src -type f \( -name '*_internal.c' -o -name '*_private.c' -o -name '*_internal.h' \) -print | grep .; then
  echo "source layout: implementation-phase file remains" >&2
  exit 1
fi

if grep -RInE '#include[[:space:]]+["<].*src/cli/' \
    src/core src/model src/graph src/runtime src/backend; then
  echo "source layout: domain owner depends on CLI" >&2
  exit 1
fi

if grep -RInE '#include[[:space:]]+["<].*tests/' src include; then
  echo "source layout: production owner depends on tests" >&2
  exit 1
fi

grep -nF '$(OBJ_DIR)/%.o: %.c' Makefile >/dev/null
grep -nF '@mkdir -p $(@D)' Makefile >/dev/null

# Every independently focused runtime owner must remain inside the aggregate
# used by both sanitizer builds. Derive the set so a later owner cannot add a
# green focused target while silently escaping ASan/UBSan coverage.
runtime_aggregate=$(awk '
  /^test-runtime: \$\(TEST_RUNNER\)$/ { body = 1; next }
  body && /^[^[:space:]#][^=]*:/ { exit }
  body { print }
' Makefile)
focused_runtime_filters=$(awk '
  /^test-runtime-[[:alnum:]_-]+: \$\(TEST_RUNNER\)$/ { focused = 1; next }
  focused && /^\tYVEX_TEST_FILTER=runtime_[[:alnum:]_-]+ \$\(TEST_RUNNER\)$/ {
    line = $0
    sub(/^\tYVEX_TEST_FILTER=/, "", line)
    sub(/ \$\(TEST_RUNNER\)$/, "", line)
    print line
    focused = 0
    next
  }
  focused { focused = 0 }
' Makefile | sort -u)
for filter in $focused_runtime_filters; do
  printf '%s\n' "$runtime_aggregate" |
    grep -F "YVEX_TEST_FILTER=$filter \$(TEST_RUNNER)" >/dev/null || {
      echo "source layout: runtime sanitizer aggregate omits $filter" >&2
      exit 1
    }
done

echo "source layout: ok canonical_owners=19 superseded_owners=0"
