#!/bin/sh
# Verifies that one executable owns commands, clients, and the foreground server mode.
set -eu

YVEX_BIN=${YVEX_BIN:-./yvex}
BUILD_DIR=${BUILD_DIR:-build}

test -x "$YVEX_BIN"
test ! -e ./yvexd
test ! -e ./yvex-dev
test ! -e ./yvex-openai
test ! -e "$BUILD_DIR/package/product/bin/yvex-dev"
test ! -e "$BUILD_DIR/package/product/bin/yvex-openai"
test ! -e "$BUILD_DIR/package/developer"
test "$(find "$BUILD_DIR/package/product/bin" -maxdepth 1 -type f -perm /111 \
    -printf '%f\n' | LC_ALL=C sort | tr '\n' ' ')" = 'yvex '

test "$(nm "$YVEX_BIN" | awk '$NF == "main" { count++ } END { print count + 0 }')" = 1
test "$(rg -l '(^|[[:space:]])int[[:space:]]+main[[:space:]]*\(' src/cli src/server \
    | LC_ALL=C sort | tr '\n' ' ')" = 'src/cli/main.c '
test "$(nm "$YVEX_BIN" | awk '$NF == "yvex_cli_server_dispatch" { count++ } END { print count + 0 }')" = 1
! rg -n '^gateway:|^dev-tools:|^package-dev:|YVEX_OPENAI_BIN|YVEX_DEV_BIN' Makefile \
    >/dev/null
printf 'test: product_topology single yvex command/server binary\n'
