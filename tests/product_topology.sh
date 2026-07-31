#!/bin/sh
# Verifies that builds and packages expose only the two admitted product executables.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
YVEXD_BIN=${YVEXD_BIN:-./yvexd}
BUILD_DIR=${BUILD_DIR:-build}

test -x "$YVEX_BIN"
test -x "$YVEXD_BIN"
test ! -e ./yvex-dev
test ! -e ./yvex-openai
test ! -e "$BUILD_DIR/package/product/bin/yvex-dev"
test ! -e "$BUILD_DIR/package/product/bin/yvex-openai"
test ! -e "$BUILD_DIR/package/developer"
test "$(find "$BUILD_DIR/package/product/bin" -maxdepth 1 -type f -perm /111 \
    -printf '%f\n' | LC_ALL=C sort | tr '\n' ' ')" = 'yvex yvexd '

for binary in "$YVEX_BIN" "$YVEXD_BIN"; do
    test "$(nm "$binary" | awk '$NF == "main" { count++ } END { print count + 0 }')" = 1
done
test "$(rg -l '(^|[[:space:]])int[[:space:]]+main[[:space:]]*\(' src/cli src/daemon \
    | LC_ALL=C sort | tr '\n' ' ')" = 'src/cli/main.c src/daemon/yvexd.c '
! rg -n '^gateway:|^dev-tools:|^package-dev:|YVEX_OPENAI_BIN|YVEX_DEV_BIN' Makefile \
    >/dev/null
printf 'test: product_topology yvex+yvexd only\n'
