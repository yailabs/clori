#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

fail() {
    printf 'repository-layout: %s\n' "$1" >&2
    exit 1
}

# The canonical clean target accepts only owned build roots and never follows
# a caller-controlled symlink. No recipe may use recursive deletion as cleanup.
make_cleanup=$(awk '
    /^[^[:space:]#][^=]*:/ { target = $0 }
    /^\t/ && /rm[[:space:]]+([^[:space:]]+[[:space:]]+)*(--recursive|-[^[:space:]]*[rR][^[:space:]]*)/ && target !~ /^clean[[:space:]]*:/ {
        print FNR ":" target ":" $0
    }
' Makefile)
if [ -n "$make_cleanup" ]; then
    printf '%s\n' "$make_cleanup" >&2
    fail "recursive cleanup outside the canonical clean target"
fi
if make -s BUILD_DIR=/ clean >/dev/null 2>&1; then
    fail "clean target accepts the filesystem root"
fi
rg -q 'if test "\$\$build_dir" = build; then' Makefile ||
    fail "external BUILD_DIR cleanup may remove repository-root executables"
clean_probe=$(mktemp -d /tmp/yvex-clean-guard.XXXXXX)
clean_external=$(mktemp -d /tmp/clean-guard-external.XXXXXX)
ln -s "$clean_external" "$clean_probe/build"
if make -s BUILD_DIR="$clean_probe/build" clean >/dev/null 2>&1; then
    fail "clean target follows a caller-controlled build symlink"
fi
unlink "$clean_probe/build"
rmdir "$clean_probe" "$clean_external"

# Operator attention and its sanitizer wrappers own mkdtemp roots and must
# never grow broad deletion shortcuts as the command surface evolves.
if rg -n --glob '*attention*.sh' \
    'rm[[:space:]]+([^[:space:]]+[[:space:]]+)*(--recursive|-[^[:space:]]*[rR][^[:space:]]*)' tests; then
    fail "attention operator or sanitizer script uses recursive rm"
fi

# Commit-bound benchmark provenance is generated as an explicit object
# dependency. A process-wide CPPFLAGS define would leave incremental builds
# carrying the previous HEAD after a commit.
if rg -n 'CPPFLAGS.*YVEX_BUILD_COMMIT' Makefile; then
    fail "build commit provenance is an untracked process-wide compiler define"
fi
rg -q '^BUILD_COMMIT_HEADER[[:space:]]*:=' Makefile ||
    fail "build commit provenance header is missing"
rg -q '^YVEX_BUILD_SOURCE_STATE[[:space:]]+[?]=' Makefile ||
    fail "build source-state provenance is missing"
rg -q '^YVEX_BUILD_SOURCE_DELTA_IDENTITY[[:space:]]+[?]=' Makefile ||
    fail "exact dirty source-delta provenance is missing"
rg -q '^YVEX_BUILD_SOURCE_TREE[[:space:]]+[?]=' Makefile ||
    fail "exact source-tree provenance is missing"
rg -q '^YVEX_BUILD_IDENTITY[[:space:]]+[?]=' Makefile ||
    fail "compiler/link/CUDA build provenance is missing"
rg -q '^YVEX_BUILD_SOURCE_ROOT[[:space:]]+[?]=' Makefile ||
    fail "external benchmark path boundary lacks a generated source root"
rg -q '^\$\(OBJ_DIR\)/src/cli/commands/graph\.o: \$\(BUILD_COMMIT_HEADER\)' Makefile ||
    fail "operator object does not depend on build commit provenance"
rg -q '^\$\(OBJ_DIR\)/src/runtime/benchmark\.o: \$\(BUILD_COMMIT_HEADER\)' Makefile ||
    fail "runtime benchmark object does not depend on build commit provenance"
rg -q '^\$\(BUILD_COMMIT_HEADER\): FORCE' Makefile ||
    fail "build commit provenance does not revalidate on incremental builds"
for field in YVEX_BUILD_SOURCE_STATE YVEX_BUILD_SOURCE_DELTA_IDENTITY \
             YVEX_BUILD_IDENTITY; do
    rg -q "$field" src/runtime/benchmark.c ||
        fail "runtime benchmark records do not consume generated $field provenance"
done
rg -q 'YVEX_BUILD_SOURCE_TREE' src/runtime/evidence.c ||
    fail "execution qualification records do not consume source-tree provenance"
rg -q 'YVEX_BUILD_SOURCE_ROOT' src/cli/commands/graph.c ||
    fail "operator benchmark paths do not consume the generated source-root boundary"
if rg -n 'git[[:space:]]+(status|diff|rev-parse)' src/runtime src/cli/commands/graph.c; then
    fail "runtime benchmark provenance must not inspect the repository at execution time"
fi

# Production membership is handwritten once in the ownership manifest. Make
# consumes only the deterministic projection, so a new file cannot bypass
# ownership admission through a wildcard or a second source list.
rg -q '^SOURCE_OWNER_MANIFEST := config/source_owners.tsv$' Makefile ||
    fail "canonical production source manifest is not configured"
rg -q '^SOURCE_MANIFEST_GENERATOR := tools/generate_source_manifest.py$' Makefile ||
    fail "source build projection has no canonical generator"
rg -q '^include \$\(SOURCE_MANIFEST_MK\)$' Makefile ||
    fail "Make does not consume the generated source projection"
if rg -n '\$\(wildcard[[:space:]]+(src|include)/|^[A-Z0-9_]+[[:space:]]*[:?+]?=[[:space:]]*(src|include)/.*[.](c|cu|h)\b' Makefile; then
    fail "Makefile repeats or wildcard-admits production membership"
fi
make -s check-source-manifest >/dev/null ||
    fail "generated source projection is stale or nondeterministic"

# Nested package/native-CUDA compositions may request the same generated image
# manifest concurrently. Publication must expose either the previous complete
# manifest or the next complete manifest, never a truncated intermediate file.
for target in '$(CUDA_PTX_INC)' '$(CUDA_CUBIN_INC)'; do
    recipe=$(awk -v target="$target" '
        index($0, target ":") == 1 { active = 1 }
        active && /^[^[:space:]#][^=]*:/ && index($0, target ":") != 1 { exit }
        active { print }
    ' Makefile)
    printf '%s\n' "$recipe" | grep -F 'tmp="$@.tmp.$$$$"' >/dev/null ||
        fail "$target generation lacks a process-unique staging path"
    printf '%s\n' "$recipe" | grep -F 'mv "$$tmp" "$@"' >/dev/null ||
        fail "$target generation is not atomically published"
done

for obsolete_target in cli server test-client test-cli-cutover \
                       test-openai-agent test-runtime-benchmark-chart \
                       test-runtime-sessions test-runtime-turns \
                       test-runtime-telemetry; do
    if rg -q "^${obsolete_target}:" Makefile; then
        fail "historical Make target returned: ${obsolete_target}"
    fi
done

provenance_root=$(mktemp -d /tmp/yvex-build-provenance.XXXXXX)
provenance_header="$provenance_root/generated/build_commit.h"
cleanup_provenance() {
    rm -f "$provenance_header"
    rmdir "$provenance_root/generated" 2>/dev/null || :
    rmdir "$provenance_root" 2>/dev/null || :
}
trap cleanup_provenance EXIT HUP INT TERM
for source_state in clean dirty; do
    make -s BUILD_DIR="$provenance_root" \
        YVEX_BUILD_COMMIT=0123456789abcdef0123456789abcdef01234567 \
        YVEX_BUILD_SOURCE_TREE=cccccccccccccccccccccccccccccccccccccccc \
        YVEX_BUILD_SOURCE_STATE="$source_state" \
        YVEX_BUILD_SOURCE_DELTA_IDENTITY=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
        YVEX_BUILD_IDENTITY=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
        YVEX_BUILD_SOURCE_ROOT=/tmp/yvex-provenance-source \
        "$provenance_header"
    rg -q '^#define YVEX_BUILD_COMMIT "0123456789abcdef0123456789abcdef01234567"$' \
        "$provenance_header" || fail "generated build provenance lost the exact commit"
    rg -q '^#define YVEX_BUILD_SOURCE_TREE "cccccccccccccccccccccccccccccccccccccccc"$' \
        "$provenance_header" || fail "generated build provenance lost the exact source tree"
    rg -q "^#define YVEX_BUILD_SOURCE_STATE \"$source_state\"$" "$provenance_header" ||
        fail "generated build provenance did not retain $source_state source state"
    rg -q '^#define YVEX_BUILD_SOURCE_DELTA_IDENTITY "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"$' \
        "$provenance_header" || fail "generated provenance lost the exact source delta"
    rg -q '^#define YVEX_BUILD_IDENTITY "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"$' \
        "$provenance_header" || fail "generated provenance lost the exact build identity"
    rg -q '^#define YVEX_BUILD_SOURCE_ROOT "/tmp/yvex-provenance-source"$' \
        "$provenance_header" || fail "generated provenance lost the source-root boundary"
done

# Target-specific flags are implementation-local and are already represented by
# the source tree.  They must not change the invocation-wide build identity
# according to which build_commit.h consumer Make happens to visit first.
base_build_identity=$(make --no-print-directory -s \
    BUILD_DIR="$provenance_root" print-build-identity)
make --no-print-directory -s BUILD_DIR="$provenance_root" \
    --eval='.PHONY: provenance-target-context' \
    --eval='provenance-target-context: CPPFLAGS += -DYVEX_TARGET_LOCAL_ONLY=1' \
    --eval="provenance-target-context: $provenance_header" \
    provenance-target-context
header_build_identity=$(sed -n \
    's/^#define YVEX_BUILD_IDENTITY "\([0-9a-f]*\)"$/\1/p' \
    "$provenance_header")
test "$header_build_identity" = "$base_build_identity" ||
    fail "target traversal order changes the invocation-wide build identity"
cleanup_provenance
trap - EXIT HUP INT TERM

build_identity=$(make --no-print-directory -s print-build-identity)
material_identity=$(make --no-print-directory -s print-build-identity \
    YVEX_BUILD_SOURCE_DELTA_IDENTITY=dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd)
runtime_noise_identity=$(YVEX_IGNORED_RUNTIME_NOISE=changed \
    make --no-print-directory -s print-build-identity)
test "$build_identity" != "$material_identity" ||
    fail "material source mutation does not change build identity"
test "$build_identity" = "$runtime_noise_identity" ||
    fail "irrelevant runtime environment noise changes build identity"

python3 tests/c_structure.py check layout
