#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

test -f include/yvex/api.h
test -f src/cli/main.c
test -f src/cli/input/private.h
test -f src/cli/render/private.h

for retired_header in runtime generation metrics; do
  test ! -e "include/yvex/$retired_header.h" || {
    echo "surface: retired diagnostic ABI returned: $retired_header" >&2
    exit 1
  }
done

retired_symbol_pattern='^yvex_(engine_|session_|kv_cache_|decode_step_summary_init$|logits_|sampling_strategy_name$|metrics_|metric_phase_name$|profile_write_json$|trace_)'
if find include/yvex -maxdepth 1 -type f -name '*.h' -exec \
    grep -HnE 'yvex_(engine_|session_|kv_cache_|decode_step_summary_init([^[:alnum:]_]|$)|logits_|sampling_strategy_name([^[:alnum:]_]|$)|metrics_|metric_phase_name([^[:alnum:]_]|$)|profile_write_json([^[:alnum:]_]|$)|trace_)' \
    {} +; then
  echo "surface: retired diagnostic symbol returned to the installed ABI" >&2
  exit 1
fi

archive=${YVEX_LIB:-build/lib/libyvex.a}
if test -f "$archive" &&
    nm -g --defined-only "$archive" |
      awk '{print $NF}' |
      grep -E "$retired_symbol_pattern"; then
  echo "surface: retired diagnostic symbol returned to the static library" >&2
  exit 1
fi

test ! -d src/generation || {
  echo "surface: superseded generation runtime spine returned" >&2
  exit 1
}

for surface in backend graph model_artifacts model_target source; do
  test -f "src/cli/commands/$surface.c" || {
    echo "surface: missing command adapter: $surface" >&2
    exit 1
  }
  test -f "src/cli/input/$surface.c" || {
    echo "surface: missing typed input owner: $surface" >&2
    exit 1
  }
done

for surface in backend graph model_artifacts model_target source; do
  test -f "src/cli/render/$surface.c" || {
    echo "surface: missing typed render owner: $surface" >&2
    exit 1
  }
done

test -f src/cli/io/out.c || {
  echo "surface: missing canonical CLI byte writer" >&2
  exit 1
}
grep -q '^void yvex_cli_json_begin' src/cli/io/out.c || {
  echo "surface: CLI byte writer lost JSON projection" >&2
  exit 1
}

if grep -RInE '(^|[^a-zA-Z_])(printf|fprintf|vprintf|vfprintf|puts|fputs|putchar|perror)[[:space:]]*\(' \
    src/cli/commands src/cli/input; then
  echo "surface: command/input owner writes operator output" >&2
  exit 1
fi

if grep -RInE '\b(argc|argv)\b|usage:[[:space:]]*yvex' \
    src/model src/graph src/runtime src/backend src/source src/artifact src/gguf; then
  echo "surface: domain owner contains CLI grammar" >&2
  exit 1
fi

if find . -maxdepth 1 -type f \( -name 'yvex_*.c' -o -name 'yvex_*_private.h' \) -print | grep .; then
  echo "surface: forbidden root compatibility owner" >&2
  exit 1
fi

echo "surface topology: ok typed-command-input-render triples=5 writers=1"
