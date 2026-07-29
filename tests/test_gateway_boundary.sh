#!/bin/sh
# Purpose: prove the application gateway cannot enter model or engine ownership.
set -eu

binary=${YVEX_OPENAI_BIN:-./yvex-openai}
test -x "$binary"

if nm "$binary" | grep -E \
  'yvex_(artifact|backend|runtime_(model|transformer|generation|sampling|logits)|gguf|graph)_' \
  >/dev/null; then
  echo "gateway boundary: inference-engine symbol entered yvex-openai" >&2
  exit 1
fi

if ldd "$binary" 2>/dev/null | grep -E 'libcuda|libcudart|libyvex' >/dev/null; then
  echo "gateway boundary: inference-engine library entered yvex-openai" >&2
  exit 1
fi

awk -F '\t' '$1 == "src/gateway/openai/main.c" && $4 == "entrypoint" { found = 1 }
  END { exit !found }' config/source_owners.tsv

echo "gateway boundary: client protocol only; model/artifact/CUDA symbols absent"
