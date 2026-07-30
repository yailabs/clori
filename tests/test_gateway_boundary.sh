#!/bin/sh
# Purpose: prove OpenAI syntax stays in the server adapter and owns no runtime authority.
set -eu

adapter=${YVEX_OPENAI_ADAPTER:-build/tests/openai_adapter}
test -x "$adapter"
test -d src/server/openai
test ! -d src/gateway/openai
test "$(find src/server/openai -maxdepth 1 -type f -name '*.c' | wc -l | tr -d ' ')" = 5
test -z "$(rg -l '(^|[[:space:]])int[[:space:]]+main[[:space:]]*\(' src/server/openai || true)"

if rg -n '#include <yvex/internal/(runtime|transformer|generation|decode|logits|sampling)\.h>' \
    src/server/openai >/dev/null; then
    echo 'OpenAI adapter boundary: direct engine header dependency found' >&2
    exit 1
fi

for object in build/obj/src/server/openai/*.o; do
    test -f "$object"
    if nm -u "$object" | grep -E \
      'yvex_(runtime_model_open|artifact_materialize|runtime_transformer|runtime_generation|backend_cuda)' \
      >/dev/null; then
        echo "OpenAI adapter boundary: direct engine symbol in $object" >&2
        exit 1
    fi
done

! rg -n '^gateway:|^dev-tools:|^package-dev:|YVEX_OPENAI_BIN|YVEX_DEV_BIN' Makefile \
    >/dev/null
! rg -n '^src/gateway/openai/' config/source_owners.tsv >/dev/null

echo 'OpenAI adapter boundary: in-process server owner; protocol-only execution path'
