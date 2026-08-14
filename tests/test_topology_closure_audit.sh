#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

sh tests/test_source_ownership.sh
sh tests/test_repository_layout.sh
sh tests/test_architecture_boundaries.sh
sh tests/test_surface.sh

test -f src/graph/attention.c
test -f src/graph/numeric.c
test -f src/graph/families/deepseek_v4.c
test -f src/backend/cuda/attention.c
test -f tests/reference/deepseek_attention.c

if grep -RIn 'yvex_test_attention_reference_' src include; then
  echo "topology closure: production attention calls the test reference" >&2
  exit 1
fi

if grep -En 'yvex_attention_[A-Za-z0-9_]*[[:space:]]*\(' \
    tests/reference/deepseek_attention.c; then
  echo "topology closure: independent reference calls production attention arithmetic" >&2
  exit 1
fi
if grep -En -- '->(cpu_|cuda_|rolling_state_|history_validate)' \
    tests/reference/deepseek_attention.c; then
  echo "topology closure: independent reference dispatches a production attention path" >&2
  exit 1
fi

if grep -En \
    'cpu_(probe|first_token|chunk)_execute|yvex_attention_[A-Za-z0-9_]*_cpu|yvex_test_attention_reference_|yvex_quant_[A-Za-z0-9_]*reference' \
    src/backend/cuda/attention.c; then
  echo "topology closure: CUDA attention contains a host numeric fallback" >&2
  exit 1
fi
if grep -RIn -E 'Fallback embedded PTX|\.visible[[:space:]]+\.entry' \
    src/backend/cuda --include='*.c'; then
  echo "topology closure: production C contains fallback PTX" >&2
  exit 1
fi

if find src -type f -path '*/families/*' | grep -E '/deepseek[^/]*_(plan|execute|sink|numeric|internal|reference|report)\.(c|h|cu)$'; then
  echo "topology closure: family phase fragmentation remains" >&2
  exit 1
fi

family_files=$(find src -type f -path '*/families/deepseek_v4.*' | wc -l | tr -d ' ')
test "$family_files" -eq 2 || {
  echo "topology closure: DeepSeek family projection count is $family_files/2" >&2
  exit 1
}

grep -F '.cuda_execution_ready = 1,' src/graph/plan.c >/dev/null || {
  echo "topology closure: attention CUDA readiness is not admitted" >&2
  exit 1
}
grep -F '.full_execution_ready = 1,' src/graph/plan.c >/dev/null || {
  echo "topology closure: attention full execution is not admitted" >&2
  exit 1
}
grep -F 'descriptor->summary.generation_ready = 0;' src/runtime/descriptor.c >/dev/null || {
  echo "topology closure: runtime generation readiness was promoted" >&2
  exit 1
}
for fact in \
  'attention_execution_supported=%d' \
  'attention_cuda_execution_ready=%d' \
  'runtime_generation_ready=%d'; do
  grep -F "$fact" tests/live/attention_deepseek.c >/dev/null || {
    echo "topology closure: missing live refusal projection: $fact" >&2
    exit 1
  }
done

echo "topology closure: ok hard=1 advisory=0 family_files=$family_files"
