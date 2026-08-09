#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

if rg -n '#include[[:space:]]+"src/graph/private\.h"' src/cli; then
    echo "architecture: CLI depends on graph-private ABI" >&2
    exit 1
fi

printf '%s\n' '#include <yvex/internal/graph.h>' 'int main(void) { return 0; }' |
    "${CC:-cc}" -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L \
        -Iinclude -I. -std=c11 -Wall -Wextra -pedantic -Werror \
        -x c -fsyntax-only -

fail() {
    printf 'architecture: %s\n' "$1" >&2
    exit 1
}

family_compare_pattern='(strcmp|strncmp|strcasecmp|strncasecmp|strstr|strcasestr)'
family_name_pattern='(deepseek|qwen|gemma|llama|kimi|mamba)'
family_branch_pattern="(${family_compare_pattern}[^;]*${family_name_pattern}|${family_name_pattern}[^;]*${family_compare_pattern})"
runtime_planning_include_pattern='#include[[:space:]]+[<"]yvex/internal/(compilation|source|source_payload|gguf_writer)[.]h[>"]'
runtime_planning_call_pattern='yvex_(source_payload_[A-Za-z0-9_]*|transform_[A-Za-z0-9_]*|quant_plan_[A-Za-z0-9_]*|gguf_writer_[A-Za-z0-9_]*)[[:space:]]*\('
runtime_planning_symbol_pattern='^yvex_(source_payload_[A-Za-z0-9_]*|transform_[A-Za-z0-9_]*|quant_plan_[A-Za-z0-9_]*|gguf_writer_[A-Za-z0-9_]*)$'
runtime_family_dispatch_pattern='(yvex_runtime_family_adapter|[.]adapter->graph|'
runtime_family_dispatch_pattern="${runtime_family_dispatch_pattern}"'[.]adapter[[:space:]]*=|'
runtime_family_dispatch_pattern="${runtime_family_dispatch_pattern}"'yvex_graph_execution_(find|at)[[:space:]]*\()'
fallback_ptx_pattern='(fallback_ptx|ptx_fallback|"[[:space:]]*[.]version[[:space:]]+[0-9])'
cuda_cpu_fallback_pattern='(cpu_chunk_execute|rolling_state_step_cpu|yvex_backend_open_cpu(_impl)?|yvex_quant_cpu_[A-Za-z0-9_]*|yvex_attention_[A-Za-z0-9_]*_cpu)[[:space:]]*\('
deprecated_digest_hash_pattern='yvex_sha256_[A-Za-z0-9_]*[[:space:]]*\([^;]*\boutput_digest\b'
backend_digest_alias_pattern='\boutput_digest\b[^;]*\b(cpu_output_digest|cuda_output_digest)\b|\b(cpu_output_digest|cuda_output_digest)\b[^;]*\boutput_digest\b'
cli_family_helper_pattern='yvex_(source_is_release_target|model_register_deepseek_v4)[[:space:]]*\('
cli_family_abi_pattern='(#include[[:space:]]+[<"]yvex/internal/families/|yvex_[A-Za-z0-9_]*(deepseek|qwen|gemma|llama|kimi|mamba)[A-Za-z0-9_]*|YVEX_[A-Z0-9_]*(DEEPSEEK|QWEN|GEMMA|LLAMA|KIMI|MAMBA)[A-Z0-9_]*)'
cli_preparation_call_pattern='yvex_(source_payload_[A-Za-z0-9_]*|transform_[A-Za-z0-9_]*|quant_plan_[A-Za-z0-9_]*|gguf_writer_[A-Za-z0-9_]*|materialization_(plan|session)_[A-Za-z0-9_]*|runtime_descriptor_build[A-Za-z0-9_]*|artifact_physical_compatibility_[A-Za-z0-9_]*)[[:space:]]*\('
family_preparation_callback_pattern='^static[[:space:]]+int[[:space:]]+prepare_deepseek_runtime_binding[[:space:]]*\('
family_preparation_leak_pattern='(yvex_(model_register_deepseek_v4|graph_lower_deepseek_v4|artifact_admit_deepseek|runtime_descriptor_build_deepseek|quant_plan_build_deepseek_profile)[[:space:]]*\(|YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME)'
cli_runtime_lifecycle_pattern='yvex_runtime_(model_(open|close|summary_copy|view_get)|session_(open|close|summary_copy|view_get)|residency_(prepare|close|snapshot|invalidate))[[:space:]]*\('
recursive_cleanup_pattern='(^|[;&|()[:space:]])(command[[:space:]]+)?r'\
'm[[:space:]]+([^#;]*[[:space:]])?(-[[:alpha:]]*[rR][[:alpha:]]*|--recursive)([[:space:]]|$)'
recursive_cleanup_call_pattern='(system|popen)[[:space:]]*\([^;]*(rm[[:space:]]+-[[:alpha:]]*[rR]|rm[[:space:]]+--recursive)'
conversation_literal_pattern='(<think>|</think>|｜DSML｜|<tool_result>|<｜(User|Assistant|latest_reminder)｜>)'
family_runtime_context_pattern='(context_capacity|requested_session_context|admitted_execution_maximum|'
family_runtime_context_pattern="${family_runtime_context_pattern}"'per_(session|request)_maximum|physical_state_pool_tokens)'
moe_family_registry_pattern='yvex_graph_moe_family_(at|find)[[:space:]]*\('

# Every expression used as a hard gate carries positive and negative probes.
# This catches regex drift before a repository scan can produce false comfort.
printf '%s\n' 'if (strcmp(target, "deepseek") == 0) dispatch();' |
    rg -i "$family_branch_pattern" >/dev/null ||
    fail "family-string branch guard misses a direct target comparison"
if printf '%s\n' 'if (adapter->family_id == request->family_id) dispatch();' |
    rg -i "$family_branch_pattern" >/dev/null; then
    fail "family-string branch guard rejects typed adapter dispatch"
fi
printf '%s\n' 'yvex_source_is_release_target(target);' |
    rg "$cli_family_helper_pattern" >/dev/null ||
    fail "CLI family-helper guard misses source release-target policy"
printf '%s\n' 'yvex_model_register_deepseek_v4();' |
    rg "$cli_family_helper_pattern" >/dev/null ||
    fail "CLI family-helper guard misses direct family-model registration"
if printf '%s\n' 'adapter->preparation_model();' |
    rg "$cli_family_helper_pattern" >/dev/null; then
    fail "CLI family-helper guard rejects typed adapter preparation"
fi
printf '%s\n' '#include <yvex/internal/families/deepseek_v4.h>' |
    rg -i "$cli_family_abi_pattern" >/dev/null ||
    fail "CLI family-ABI guard misses a direct family header"
printf '%s\n' 'yvex_runtime_descriptor_build_deepseek(&descriptor);' |
    rg "$cli_preparation_call_pattern" >/dev/null ||
    fail "CLI preparation guard misses direct family/compiler planning"
if printf '%s\n' 'preparation->prepare_runtime_binding(&request, &result, &err);' |
    rg -i "$cli_family_abi_pattern|$cli_preparation_call_pattern" >/dev/null; then
    fail "CLI preparation guard rejects typed family preparation dispatch"
fi
printf '%s\n' 'yvex_runtime_model_open(&model, &request, &failure, &err);' |
    rg "$cli_runtime_lifecycle_pattern" >/dev/null ||
    fail "CLI lifecycle guard misses direct runtime-model ownership"
if printf '%s\n' 'yvex_graph_attention_operator_execute(&request, &result, &cleanup, &err);' |
    rg "$cli_runtime_lifecycle_pattern" >/dev/null; then
    fail "CLI lifecycle guard rejects the canonical production operator"
fi
printf '%s\n' '<think>' |
    rg "$conversation_literal_pattern" >/dev/null ||
    fail "conversation-literal guard misses a source-authored control token"
if printf '%s\n' 'conversation->thinking_start' |
    rg "$conversation_literal_pattern" >/dev/null; then
    fail "conversation-literal guard rejects a typed family fact"
fi
printf '%s\n' 'options.context_capacity = 4096;' |
    rg "$family_runtime_context_pattern" >/dev/null ||
    fail "family context-capacity guard misses a runtime-selected capacity"
if printf '%s\n' 'model.maximum_context = 1048576;' |
    rg "$family_runtime_context_pattern" >/dev/null; then
    fail "family context-capacity guard rejects a semantic model maximum"
fi
printf '%s\n' 'yvex_graph_moe_family_at(index);' |
    rg "$moe_family_registry_pattern" >/dev/null ||
    fail "MoE family-registry guard misses a global compiler-policy lookup"
printf '%s\n' '#include <yvex/internal/source_payload.h>' |
    rg "$runtime_planning_include_pattern" >/dev/null ||
    fail "runtime planning-dependency guard misses source payload ownership"
printf '%s\n' 'yvex_quant_plan_build_explicit();' |
    rg "$runtime_planning_call_pattern" >/dev/null ||
    fail "runtime planning-dependency guard misses quant-plan construction"
if printf '%s\n' 'yvex_quant_f16_decode(bits);' |
    rg "$runtime_planning_call_pattern" >/dev/null; then
    fail "runtime planning-dependency guard rejects the canonical scalar codec"
fi
printf '%s\n' 'yvex_gguf_writer_plan_release' |
    rg "$runtime_planning_symbol_pattern" >/dev/null ||
    fail "runtime link-dependency guard misses a writer-planning symbol"
if printf '%s\n' 'yvex_quant_f16_decode' |
    rg "$runtime_planning_symbol_pattern" >/dev/null; then
    fail "runtime link-dependency guard rejects the canonical scalar codec"
fi
printf '%s\n' 'model.adapter->graph();' |
    rg "$runtime_family_dispatch_pattern" >/dev/null ||
    fail "runtime family-dispatch guard misses a concrete adapter callback"
printf '%s\n' 'yvex_graph_execution_find(0, 0, target);' |
    rg "$runtime_family_dispatch_pattern" >/dev/null ||
    fail "runtime family-dispatch guard misses a concrete execution registry lookup"
if printf '%s\n' 'model.graph;' |
    rg "$runtime_family_dispatch_pattern" >/dev/null; then
    fail "runtime family-dispatch guard rejects the generic graph capability"
fi
printf '%s\n' 'static const char fallback_ptx[] = ".version 8.0";' |
    rg -i "$fallback_ptx_pattern" >/dev/null ||
    fail "fallback-PTX guard misses an embedded production blob"
printf '%s\n' 'family->cpu_chunk_execute(plan);' |
    rg "$cuda_cpu_fallback_pattern" >/dev/null ||
    fail "CUDA fallback guard misses CPU attention dispatch"
if printf '%s\n' 'cuda_kernel_launch(plan);' |
    rg "$cuda_cpu_fallback_pattern" >/dev/null; then
    fail "CUDA fallback guard rejects device-only dispatch"
fi
printf '%s\n' 'yvex_sha256_update_text(&hash, result->output_digest);' |
    rg "$deprecated_digest_hash_pattern" >/dev/null ||
    fail "digest guard misses legacy output_digest identity input"
if printf '%s\n' 'yvex_sha256_update_text(&hash, result->tensor_output_digest);' |
    rg "$deprecated_digest_hash_pattern" >/dev/null; then
    fail "digest guard rejects canonical tensor_output_digest"
fi
printf '%s\n' 'copy(output_digest, cuda_output_digest);' |
    rg "$backend_digest_alias_pattern" >/dev/null ||
    fail "digest guard misses backend-specific legacy semantics"
for unsafe_flags in '-rf unsafe' '-fr unsafe' '-f -r unsafe' '--recursive unsafe'; do
    printf '%s%s\n' 'rm ' "$unsafe_flags" | rg "$recursive_cleanup_pattern" >/dev/null ||
        fail "destructive-cleanup guard misses recursive flags: $unsafe_flags"
done
if printf '%s%s\n' 'rm ' '-f owned-file' | rg "$recursive_cleanup_pattern" >/dev/null; then
    fail "destructive-cleanup guard rejects non-recursive file removal"
fi
printf '%s\n' 'system("rm -rf /tmp/fixture");' |
    rg "$recursive_cleanup_call_pattern" >/dev/null ||
    fail "destructive-cleanup guard misses a C shell deletion"
if printf '%s\n' 'unlink(owned_path);' |
    rg "$recursive_cleanup_call_pattern" >/dev/null; then
    fail "destructive-cleanup guard rejects narrow owned cleanup"
fi

# Common runtime and operator owners dispatch through typed family adapters.
# Family names may occur in recipes, evidence, and rendered facts, but never as
# string-comparison control flow in the family-neutral execution plane.

[ ! -d src/runtime/families ] ||
    fail "runtime must remain family-neutral; concrete family runtime directories are forbidden"
if find src/runtime -type f \( -name '*deepseek*' -o -name '*qwen*' -o -name '*gemma*' \) |
    rg .; then
    fail "runtime contains a concrete family source"
fi

# A family may project facts at three irreducible boundaries: model intake,
# graph composition, and a fused backend lowering. These same-basename files
# are distinct owners; any fourth DeepSeek implementation would instead create
# the parallel family runtime that this DAG forbids.
deepseek_family_sources=$(find src -path '*/families/deepseek_v4.c' -type f | sort)
expected_deepseek_family_sources='src/backend/cuda/families/deepseek_v4.c
src/graph/families/deepseek_v4.c
src/model/families/deepseek_v4.c'
[ "$deepseek_family_sources" = "$expected_deepseek_family_sources" ] ||
    fail "DeepSeek must have exactly model, graph, and fused-CUDA family projections"
while IFS= read -r source; do
    awk -F '\t' -v source="$source" '$1 == source && $4 == "family" { found = 1 } END { exit !found }' \
        config/source_owners.tsv ||
        fail "family projection lacks a distinct machine-readable family owner: $source"
done <<EOF
$expected_deepseek_family_sources
EOF

if find src include -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' \) \
        ! -path 'src/model/families/*' -print0 |
    xargs -0 rg -n "$conversation_literal_pattern"; then
    fail "source-authored conversation literal escaped the model-family projection"
fi

if find src -path '*/families/*' -type f \
        \( -name '*.c' -o -name '*.h' -o -name '*.cu' \) -print0 |
    xargs -0 rg -n "$family_runtime_context_pattern"; then
    fail "a family projection owns runtime-selected context capacity"
fi

family_neutral_sources=$(
    {
        find src/runtime -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' \)
        find src/backend -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' \)
        find src/backend/cuda -maxdepth 1 -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' \)
        printf '%s\n' src/artifact/materialize.c src/cli/commands/graph.c src/cli/input/graph.c
    } | LC_ALL=C sort -u
)
while IFS= read -r source; do
    [ -f "$source" ] || fail "family-neutral scan input is missing: $source"
    if rg -n -i "$family_branch_pattern" "$source"; then
        fail "family-neutral owner branches on a family or target name: $source"
    fi
done <<EOF
$family_neutral_sources
EOF
if rg -n "$cli_family_helper_pattern" src/cli/commands/graph.c; then
    fail "common graph CLI bypasses typed runtime-family preparation facts"
fi
if rg -n -i "$cli_family_abi_pattern" src/cli/commands/graph.c; then
    fail "common graph CLI imports or names a concrete family ABI"
fi
if rg -n "$cli_preparation_call_pattern" src/cli/commands/graph.c; then
    fail "common graph CLI directly constructs compiler preparation truth"
fi
if rg -n "$cli_runtime_lifecycle_pattern" src/cli/commands/graph.c; then
    fail "common graph CLI owns runtime model, session, or residency lifecycle"
fi

# Generic attention state iterates sealed family-projected components. Class
# names, family constants, and runtime dependencies would recreate policy in
# the lifecycle owner and make the next state consumer unsafe.
[ ! -e src/runtime/state.c ] ||
    fail "attention state lifecycle remains under the runtime owner"
if rg -n '(YVEX_DEEPSEEK|YVEX_ATTENTION_CLASS_(SWA|CSA|HCA)|families/deepseek)' \
    src/graph/state.c include/yvex/internal/graph_state.h; then
    fail "generic attention state contains concrete family policy"
fi
if rg -n '#include[[:space:]]+[<"]yvex/internal/runtime[.]h[>"]' \
    include/yvex/internal/graph_state.h src/graph/state.c; then
    fail "graph state ABI depends on the downstream runtime owner"
fi
if rg -n 'yvex_runtime_attention_(state|capacity)' \
    src/graph/state.c include/yvex/internal/graph_state.h; then
    fail "generic attention state retains the obsolete runtime-owned ABI"
fi

# DeepSeek cold preparation is intentionally composed by the admitted model
# artifact gate.  Its callback remains file-local; common runtime and CLI code
# can reach it only through the typed family-preparation registry.
if [ "$(rg -c "$family_preparation_callback_pattern" src/model/artifacts/gate.c)" -ne 1 ]; then
    fail "DeepSeek cold preparation callback is missing or no longer file-local"
fi
if rg -n "$family_preparation_leak_pattern" src/runtime src/cli/commands/graph.c; then
    fail "family-specific cold preparation leaked into common runtime/CLI owners"
fi
if rg -n "$moe_family_registry_pattern" src include; then
    fail "generic MoE compilation retains a concrete family registry"
fi
preparation_callback_owners=$(
    rg -l 'prepare_deepseek_runtime_binding' src include | LC_ALL=C sort
)
if [ "$preparation_callback_owners" != 'src/model/artifacts/gate.c' ]; then
    printf '%s\n' "$preparation_callback_owners" >&2
    fail "DeepSeek cold preparation callback escaped its admitted composition owner"
fi

# Runtime consumes an immutable runtime binding. Source verification,
# Transformation-IR construction, quant planning, and writer planning belong
# to the preparation plane and may not re-enter a runtime translation unit.
if rg -n "$runtime_planning_include_pattern" src/runtime; then
    fail "runtime translation unit includes a source/compiler planning owner"
fi
if rg -n "$runtime_planning_call_pattern" src/runtime; then
    fail "runtime translation unit calls a source/compiler planning owner"
fi
runtime_objects=$(find build/obj/src/runtime -type f -name '*.o' 2>/dev/null | LC_ALL=C sort)
[ -n "$runtime_objects" ] || fail "runtime object inventory is unavailable"
runtime_planning_symbols=$(
    nm -u $runtime_objects | awk '{ print $NF }' |
        rg "$runtime_planning_symbol_pattern" || true
)
if [ -n "$runtime_planning_symbols" ]; then
    printf '%s\n' "$runtime_planning_symbols" >&2
    fail "runtime objects link source/compiler planning symbols"
fi
rg -n '^test-runtime-attention-live:' Makefile >/dev/null ||
    fail "runtime attention session/oracle evidence has no canonical target"
rg -nF 'YVEX_ATTENTION_RUNTIME_BINDING="$$binding" $(ATTENTION_LIVE_RUNNER)' Makefile >/dev/null ||
    fail "runtime attention evidence may silently omit its immutable binding"

# CUDA production accepts only the generated kernels.cu bundle. A local PTX
# blob or call into a CPU numerical owner is a fallback even when hidden behind
# an otherwise successful CUDA dispatch.
if rg -n -i "$fallback_ptx_pattern" src/backend/cuda; then
    fail "CUDA backend contains fallback or embedded PTX"
fi
if rg -n "$cuda_cpu_fallback_pattern" src/backend/cuda; then
    fail "CUDA backend calls a CPU numerical owner"
fi

# The split digest contract has one owner per fact. Tensor/state bytes belong to
# the embedded graph probe result; runtime owns only path evidence and the full
# execution identity. The runtime result must embed that probe instead of
# duplicating its fields.
for digest_field in tensor_output_digest state_delta_digest; do
    rg -w "$digest_field" include/yvex/internal/graph.h >/dev/null ||
        fail "graph probe lacks canonical digest field: $digest_field"
done
rg -w 'yvex_attention_probe_result[[:space:]]+probe' include/yvex/internal/runtime.h >/dev/null ||
    fail "runtime result does not embed the canonical graph probe result"
for digest_field in execution_evidence_digest execution_identity; do
    rg -w "$digest_field" include/yvex/internal/runtime.h >/dev/null ||
        fail "runtime result lacks canonical digest field: $digest_field"
done
if rg -n "$deprecated_digest_hash_pattern" src/runtime src/cli; then
    fail "deprecated output_digest contributes to a semantic identity"
fi
if rg -n "$backend_digest_alias_pattern" src/runtime src/cli; then
    fail "deprecated output_digest acquired backend-specific semantics"
fi
if rg -n '"output_digest"' src/cli/render/graph.c; then
    fail "deprecated output_digest remains exposed by the operator surface"
fi
if rg -w output_digest include/yvex/internal/runtime.h src/runtime/graph.c >/dev/null; then
    fail "deprecated output_digest remains in the runtime result contract"
fi

# Typed attention facts remain at their renderer owner. A key may appear in
# distinct report projections, but no historical .def catalog shadows them.
attention_render_fields=$(sed -nE \
    's/.*ATTENTION_(FIELD|TIMING|BENCHMARK_FIELD)\("([^"]+)".*/\2/p' \
    src/cli/render/graph.c | wc -l)
[ "$attention_render_fields" -gt 0 ] || fail "attention renderer has no typed fields"

# The command system has one reviewable source and one generated immutable
# projection. No route table, slash catalog, or retired executable is a second
# semantic authority.
[ -f config/operator/registry.json ] || fail "canonical operator registry is missing"
[ -f tools/generate_operator_registry.py ] || fail "operator registry generator is missing"
[ -f build/generated/operator/registry.c ] || fail "generated operator descriptors are missing"
[ -f build/obj/generated/operator/registry.o ] || fail "compiled operator descriptors are missing"
[ ! -d src/cli/catalog ] || fail "orphan CLI catalogs remain"
registry_sources=$(find config -type f -name '*registry*.json' -path '*/operator/*' | wc -l)
[ "$registry_sources" -eq 1 ] || fail "operator command registry source count is not one"
if rg -n 'offline_routes|runtime_routes|command_routes' src/cli; then
    fail "handwritten command route table remains"
fi
if rg -n 'strcmp\([^,]+,[[:space:]]*"/(help|status|runtime|model|memory|sessions|session|new|attach|detach|reset|close|cancel)' \
    src/cli; then
    fail "independent slash-command semantic parser remains"
fi
if rg -n 'yvex-dev|yvex-openai|"eval"|"bench"' config/operator/registry.json; then
    fail "operator registry exposes retired or unavailable products"
fi
if nm -u build/obj/generated/operator/registry.o | grep . >/dev/null; then
    fail "generated operator descriptors depend on executable behavior"
fi
if rg -n 'yvex_(artifact|backend|generation|graph|protocol|runtime|server)_|malloc\(|fopen\(' \
    build/generated/operator/registry.c; then
    fail "generated operator descriptors contain domain logic or resource behavior"
fi

for reachability_contract in \
    '### Executable reachability' \
    'production API directly' \
    'operator_command_available' \
    'cli_applicability=not_applicable'
do
    rg -F "$reachability_contract" AGENTS.md >/dev/null ||
        fail "executable-reachability contract is incomplete: $reachability_contract"
done

pending_identity_pattern='pending-payload-'\
'(plan|byte)-identity'
if rg -n "$pending_identity_pattern" src include; then
    fail "selected artifact admission retains a placeholder payload identity"
fi

# C owners must never hide broad deletion behind a shell call. The runtime,
# attention, CUDA, live, and reference test surfaces are included explicitly so
# lifecycle evidence cannot escape the same ownership rule as production.
cleanup_c_sources=$(
    {
        find src include tests/reference tests/live tests/cli tests/unit/cuda \
            -type f \( -name '*.c' -o -name '*.h' -o -name '*.cu' \)
        find tests/unit -maxdepth 1 -type f \
            \( -name '*runtime*.c' -o -name '*attention*.c' -o -name '*graph*.c' \)
    } | LC_ALL=C sort -u
)
while IFS= read -r source; do
    [ -f "$source" ] || fail "cleanup scan input is missing: $source"
    if rg -n "$recursive_cleanup_call_pattern" "$source"; then
        fail "production/runtime evidence contains broad shell cleanup: $source"
    fi
done <<EOF
$cleanup_c_sources
EOF

maintained_scripts=$(
    {
        git ls-files '*.sh'
        git ls-files --others --exclude-standard '*.sh'
    } | LC_ALL=C sort -u | while IFS= read -r script; do
        [ -f "$script" ] && printf '%s\n' "$script"
    done
)
[ -n "$maintained_scripts" ] || fail "maintained-script inventory is empty"
while IFS= read -r script; do
    if rg -n "$recursive_cleanup_pattern" "$script"; then
        fail "maintained script contains recursive rm cleanup: $script"
    fi
done <<EOF
$maintained_scripts
EOF

# Exercise cleanup ownership without exposing a real repository, model, or
# external path to deletion. Dangerous path checks use the validation-only API.
. tests/support/cleanup.sh
cleanup_probe=$(mktemp -d "${TMPDIR:-/tmp}/yvex-cleanup-guard.XXXXXX")
cleanup_external=$(mktemp -d "${TMPDIR:-/tmp}/cleanup-external.XXXXXX")
printf 'preserve\n' >"$cleanup_external/preserve"
mkdir -p "$cleanup_probe/nested"
printf 'remove\n' >"$cleanup_probe/nested/file"
ln -s "$cleanup_external" "$cleanup_probe/external-link"
yvex_test_cleanup "$cleanup_probe"
[ ! -e "$cleanup_probe" ] || fail "owned cleanup root survived deletion"
[ -f "$cleanup_external/preserve" ] || fail "cleanup followed a descendant symlink"
yvex_test_cleanup "$cleanup_probe" || fail "owned cleanup is not idempotent"

cleanup_link="${TMPDIR:-/tmp}/yvex-cleanup-link.$$"
ln -s "$cleanup_external" "$cleanup_link"
if yvex_test_cleanup_validate "$cleanup_link" >/dev/null 2>&1; then
    fail "cleanup admits a symlink root"
fi
unlink "$cleanup_link"

for refused_path in '' / "$PWD" "$HOME/lab/models" "$cleanup_external" \
    'build/tests/owned/../../..'; do
    if yvex_test_cleanup_validate "$refused_path" >/dev/null 2>&1; then
        fail "cleanup admits an unsafe path: ${refused_path:-<empty>}"
    fi
done

status_probe="${TMPDIR:-/tmp}/yvex-cleanup-status.$$"
set +e
yvex_test_cleanup_preserving_status 37 "$status_probe"
preserved_status=$?
yvex_test_cleanup_preserving_status 0 "$cleanup_external" >/dev/null 2>&1
cleanup_failure_status=$?
set -e
[ "$preserved_status" -eq 37 ] || fail "cleanup disguised the original test status"
[ "$cleanup_failure_status" -ne 0 ] || fail "cleanup failure was reported as success"
find "$cleanup_external" -xdev -depth -mindepth 1 -delete
rmdir "$cleanup_external"

if rg -n 'tests/reference/deepseek_attention|yvex_test_attention_reference_' \
    src include; then
    fail "production source references the test-only attention oracle"
fi

# The oracle may inspect immutable plans and decode admitted inputs. It may not
# call production equation, state-transition, selection, or reduction owners.
oracle_source=tests/reference/deepseek_attention.c
if rg -n \
    '(cpu_chunk_execute|cuda_token_execute|rolling_state_step_cpu)|yvex_attention_(activation_apply|compute_round|csa_select|hadamard_cpu|history_validate|output_project|reduce_chunk|rms_norm|rope_apply|topk_select|unit_rms_norm)[[:space:]]*\(' \
    "$oracle_source"; then
    fail "attention oracle calls a production numeric or composition owner"
fi

if rg -n "$runtime_family_dispatch_pattern" src/runtime include/yvex/internal/runtime.h; then
    fail "runtime retains a family adapter or model-name execution callback"
fi

for product in "${YVEX_LIB:-build/lib/libyvex.a}" "${YVEX_BIN:-./yvex}"; do
    [ -f "$product" ] || fail "required production product is missing: $product"
    if nm -A "$product" | rg 'yvex_test_attention_reference_'; then
        fail "production product links the test-only attention oracle: $product"
    fi
done

# The unified ELF may contain offline engine commands, but runtime-facing dispatch
# remains one protocol-only object lane and the product retains one process entrypoint.
client_lane=${YVEX_CLIENT_LANE_OBJ:-build/obj/src/cli/io/client.o}
[ -f "$client_lane" ] || fail "runtime-client lane object is missing: $client_lane"
if nm -u "$client_lane" | rg \
    'yvex_(runtime_model_open|artifact_materialize|runtime_transformer|runtime_generation_operator_execute|backend_cuda)'; then
    fail "runtime-client lane gained an engine dependency"
fi
product=${YVEX_BIN:-./yvex}
[ -x "$product" ] || fail "role product is missing: $product"
main_count=$(nm "$product" | awk '$NF == "main" { count++ } END { print count + 0 }')
[ "$main_count" -eq 1 ] || fail "role product does not own exactly one main: $product"
nm "$product" | rg 'yvex_cli_server_dispatch' >/dev/null ||
    fail "yvex does not contain its foreground server entrypoint"
[ ! -e ./yvexd ] || fail "retired hidden server executable remains"
[ ! -d src/gateway/openai ] || fail "retired standalone OpenAI source owner remains"
if rg -n '(^|[[:space:]])int[[:space:]]+main[[:space:]]*\(' src/server/openai; then
    fail "in-process OpenAI adapter contains a process entrypoint"
fi

# Link-time dependency admission keeps the independent oracle limited to checked
# allocation, immutable family facts, payload reads, descriptor lookup, and digests.
reference_objects=${YVEX_REFERENCE_OBJS:-build/obj/tests/reference/deepseek_attention.o}
for object in $reference_objects; do
    [ -f "$object" ] || fail "attention oracle object is missing: $object"
    unexpected=$(
        nm -u "$object" | awk '{ print $NF }' |
        while IFS= read -r symbol; do
            case "$symbol" in
                yvex_core_allocate|\
                yvex_error_clear|\
                yvex_materialization_session_read|yvex_model_register_deepseek_v4|\
                yvex_runtime_descriptor_find_role|yvex_sha256_*)
                    ;;
                yvex_*)
                    printf '%s\n' "$symbol"
                    ;;
            esac
        done
    )
    if [ -n "$unexpected" ]; then
        printf '%s\n' "$unexpected" >&2
        fail "attention oracle gained a production algorithm dependency: $object"
    fi
done

# The attention quality matrix is one complete rule table: every CPU/CUDA
# execution-mode family is either admitted or has an exact typed refusal.
quality_matrix=config/attention_quality.tsv
[ -f "$quality_matrix" ] || fail "attention quality matrix is missing"
quality_matrix_error=$(
    awk -F '\t' '
        NR == 1 {
            if (NF != 24 || $1 != "scope" || $4 != "backend" ||
                $7 != "capability_supported" || $22 != "evidence_identity" ||
                $24 != "reason") print "invalid header"
            next
        }
        {
            if (NF != 24) print "invalid field count on row " NR
            if ($1 != "*" || $2 != "*" || $3 != "*" || $6 != "*")
                print "matrix rule is not exhaustive on row " NR
            if ($4 != "cpu" && $4 != "cuda") print "invalid backend on row " NR
            if ($5 != "eager" && $5 != "piecewise" && $5 != "full")
                print "invalid mode on row " NR
            if ($7 != "0" && $7 != "1") print "invalid support bit on row " NR
            if (length($22) != 64) print "invalid evidence identity on row " NR
            key[$4 ":" $5]++
        }
        END {
            expected["cpu:eager"]; expected["cpu:piecewise"]; expected["cpu:full"]
            expected["cuda:eager"]; expected["cuda:piecewise"]; expected["cuda:full"]
            for (item in expected)
                if (key[item] != 1) print "missing or duplicate rule " item
        }
    ' "$quality_matrix"
)
[ -z "$quality_matrix_error" ] || {
    printf '%s\n' "$quality_matrix_error" >&2
    fail "attention quality matrix is incomplete"
}
quality_matrix_identity=$(sha256sum "$quality_matrix" | awk '{ print $1 }')
rg -F "\"$quality_matrix_identity\"" src/runtime/graph.c >/dev/null ||
    fail "runtime qualification does not bind the exact quality matrix identity"

python3 tests/c_structure.py check architecture
