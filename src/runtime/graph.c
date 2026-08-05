/* Coordinates one session's attention candidate transaction with reusable backend resources.
 * Publication waits until execution, validation, and residency succeed across every owner. */
#include <yvex/internal/core.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/benchmark.h>
#include <yvex/internal/graph.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/runtime_prefill.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int runtime_refuse(yvex_error *err, yvex_status status, const char *where, const char *message) {
    if (err && yvex_error_is_set(err)) return yvex_error_code(err);
    yvex_error_set(err, status, where, message);
    return status;
}
static double runtime_seconds(unsigned long long nanoseconds) {
    return (double)nanoseconds / 1000000000.0;
}
static int runtime_attention_is_graph_action(yvex_runtime_operator_action action) {
    return action >= YVEX_RUNTIME_OPERATOR_CAPTURE && action <= YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE;
}
int yvex_graph_attention_operator_selection_validate(
    const yvex_graph_attention_operator_request *request, yvex_error *err) {
    if (!request || (request->operation_scope != YVEX_RUNTIME_SCOPE_ATTENTION_CORE &&
                     request->operation_scope != YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE &&
                     request->operation_scope != YVEX_RUNTIME_SCOPE_RELEASE_ATTENTION_SET))
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention",
                              "attention operation scope is invalid");
    if (request->operation_scope == YVEX_RUNTIME_SCOPE_RELEASE_ATTENTION_SET &&
        (request->scope != YVEX_ATTENTION_PROBE_SCOPE_FULL || request->select_layer ||
         request->select_selection_key))
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.selection",
                              "release attention set requires complete unfiltered layer coverage");
    return YVEX_OK;
}
typedef struct {
    yvex_status status;
    const char *where, *reason;
} runtime_attention_request_rule;
static const runtime_attention_request_rule runtime_attention_request_rules[] = {
    {YVEX_ERR_INVALID_ARG, "runtime.attention", "artifact and immutable runtime binding are required"},
     {YVEX_ERR_INVALID_ARG, "runtime.attention", "canonical probe and quick/full scope are required"},
    {YVEX_ERR_UNSUPPORTED, "runtime.attention", "attention backend must be cpu or cuda"},
     {YVEX_ERR_INVALID_ARG, "runtime.attention", "attention execution mode is invalid"},
    {YVEX_ERR_INVALID_ARG, "runtime.attention", "attention trace or required-mode policy is invalid"},
     {YVEX_ERR_BOUNDS, "runtime.attention.capture_bucket", "attention capture bucket exceeds its bound"},
    {YVEX_ERR_INVALID_ARG, "runtime.attention", "attention operator action is invalid"},
     {YVEX_ERR_BOUNDS, "runtime.attention", "attention token, repeat, or warmup count exceeds its bound"},
    {YVEX_ERR_INVALID_ARG, "runtime.attention.selection", "one valid layer or selection key is required"},
     {YVEX_ERR_UNSUPPORTED, "runtime.attention.state",
     "history tokens require state exercise or component benchmark"},
    {YVEX_ERR_INVALID_ARG, "runtime.attention.phase", "state exercise requires two rows and one un-warmed run"},
    {YVEX_ERR_UNSUPPORTED, "runtime.attention.phase",
     "state exercise requires one explicit eager backend"},
    {YVEX_ERR_UNSUPPORTED, "runtime.attention.compare", "comparison requires one un-warmed state position"},
     {YVEX_ERR_UNSUPPORTED, "runtime.attention", "CUDA graph actions require a CUDA session"},
    {YVEX_ERR_UNSUPPORTED, "runtime.attention", "CUDA graph actions require graph mode"},
     {YVEX_ERR_UNSUPPORTED, "runtime.attention", "CPU attention supports eager execution only"},
    {YVEX_ERR_UNSUPPORTED, "runtime.attention", "mixed and speculative phases are not admitted"},
     {YVEX_ERR_INVALID_ARG, "runtime.attention", "attention decode requires one activation row"},
};
static int runtime_attention_request_validate(const yvex_graph_attention_operator_request *request, yvex_error *err) {
    unsigned int rule;
    if (!request || !request->target)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention", "attention target is required");
    if (!yvex_runtime_family_adapter_find(request->target)) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "runtime.attention",
                        "unsupported attention target: %s", request->target);
        return YVEX_ERR_UNSUPPORTED;
    }
    const int rejected[] = {
        !request->artifact_path || !request->runtime_binding_path || !request->artifact_path[0] ||
            !request->runtime_binding_path[0],
        request->probe != YVEX_ATTENTION_PROBE_CANONICAL_V2 ||
            (request->scope != YVEX_ATTENTION_PROBE_SCOPE_QUICK &&
             request->scope != YVEX_ATTENTION_PROBE_SCOPE_FULL),
        !request->compare_backends && request->backend != YVEX_BACKEND_KIND_CPU &&
            request->backend != YVEX_BACKEND_KIND_CUDA,
        (unsigned int)request->mode > (unsigned int)YVEX_RUNTIME_MODE_AUTO,
        (unsigned int)request->trace_policy > (unsigned int)YVEX_RUNTIME_TRACE_FULL ||
            (request->require_mode != 0 && request->require_mode != 1),
        request->capture_bucket && strlen(request->capture_bucket) >= YVEX_RUNTIME_CAPTURE_BUCKET_CAP,
        (unsigned int)request->operator_action > (unsigned int)YVEX_RUNTIME_OPERATOR_RESIDENCY_INSPECT,
        !request->token_count || !request->repeat || request->repeat > 10000ull ||
            request->warmup > 10000ull,
        (request->select_layer && request->layer_count != 1ull) ||
            (request->select_selection_key && !request->selection_key) ||
            (request->select_layer && request->select_selection_key),
        request->history_tokens &&
            request->operator_action != YVEX_RUNTIME_OPERATOR_STATE_EXERCISE &&
            request->operator_action != YVEX_RUNTIME_OPERATOR_BENCHMARK,
        request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_EXERCISE &&
            (request->token_count < 2ull || request->warmup || request->repeat != 1ull),
        request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_EXERCISE &&
            (request->compare_backends || request->mode != YVEX_RUNTIME_MODE_EAGER),
        request->compare_backends && (request->warmup || request->repeat != 1ull),
        runtime_attention_is_graph_action(request->operator_action) &&
            request->backend != YVEX_BACKEND_KIND_CUDA,
        runtime_attention_is_graph_action(request->operator_action) &&
            request->mode == YVEX_RUNTIME_MODE_EAGER,
        !request->compare_backends && request->backend == YVEX_BACKEND_KIND_CPU &&
            request->mode != YVEX_RUNTIME_MODE_EAGER && request->mode != YVEX_RUNTIME_MODE_AUTO,
        request->phase != YVEX_RUNTIME_PHASE_ATTENTION_PREFILL &&
            request->phase != YVEX_RUNTIME_PHASE_ATTENTION_DECODE,
        request->phase == YVEX_RUNTIME_PHASE_ATTENTION_DECODE && request->token_count > 1ull,
    };
    for (rule = 0u; rule < sizeof(rejected) / sizeof(rejected[0]); ++rule)
        if (rejected[rule])
            return runtime_refuse(err, runtime_attention_request_rules[rule].status,
                                  runtime_attention_request_rules[rule].where,
                                  runtime_attention_request_rules[rule].reason);
    return yvex_graph_attention_operator_selection_validate(request, err);
}
static const yvex_graph_attention_operator_result runtime_attention_result_default = {
    .status = "refused", .command = "graph attention execute", .target = "unavailable",
    .family = "unavailable", .backend = "unavailable", .scope = "unavailable",
    .input_class = "unavailable", .execution_class = "unavailable", .weights_class = "unavailable",
    .probe = {.first_failing_layer = YVEX_ATTENTION_NO_LAYER,
              .first_failing_coordinate = YVEX_ATTENTION_NO_LAYER}, .production_api_available = 1,
    .internal_live_runner_available = 1, .operator_command_available = 1,
};
static const char *const runtime_attention_action_names[] = {
    "execute", "plan", "state inspect", "state validate", "state exercise", "capture", "replay",
    "cuda-graph list", "cuda-graph inspect", "cuda-graph warmup", "cuda-graph update",
    "cuda-graph invalidate", "cuda-graph release", "trace", "profile", "benchmark",
    "qualify", "capabilities", "residency inspect"};
static const char *const runtime_attention_phase_names[] = {"prefill", "decode", "mixed", "verify"};
static const char *const runtime_attention_mode_names[] = {"eager", "piecewise", "full", "auto"};
static const char *const runtime_attention_auto_mode_reasons[] = {
    "auto-selected eager mode; CUDA graph modes are unavailable",
    "auto-selected admitted CUDA piecewise graph mode after full refusal",
    "auto-selected admitted CUDA full graph mode"};
static const char *const runtime_attention_scope_names[] = {"core", "envelope", "release-attention-set"};
static const char *const runtime_attention_trace_names[] = {"none", "summary", "stages", "full"};
static const yvex_attention_evidence_level runtime_attention_evidence_levels[] = {
    YVEX_ATTENTION_EVIDENCE_NONE, YVEX_ATTENTION_EVIDENCE_SUMMARY,
    YVEX_ATTENTION_EVIDENCE_STAGES, YVEX_ATTENTION_EVIDENCE_FULL};
static const char *const runtime_attention_comparison_stage_names[] = {
    "none", "output", "publication", "raw_kv", "compressed_geometry", "compressed_kv",
    "compressed_positions", "indexer_emission_geometry", "indexer_emission_kv",
    "indexer_emission_positions", "main_geometry", "main_kv", "main_score",
    "indexer_rolling_geometry", "indexer_rolling_kv", "indexer_rolling_score"};
static const char *const runtime_attention_trace_stage_names[] = {
    "input", "q_low", "query", "raw_kv", "compressed_kv", "indexer_kv",
    "index_query", "index_weights", "attention_values", "core_output", "envelope_output"};
typedef struct {
    size_t values, rows, width, fallback_width;
} runtime_attention_trace_field;
typedef yvex_attention_publication attn_pub;
enum { ATTN_PUB_TOKENS = offsetof(attn_pub, token_count) };
static const runtime_attention_trace_field runtime_attention_trace_fields[] = {
    {offsetof(attn_pub, input), ATTN_PUB_TOKENS, offsetof(attn_pub, envelope_output_width),
     offsetof(attn_pub, hidden_width)},
    {offsetof(attn_pub, q_low), ATTN_PUB_TOKENS, offsetof(attn_pub, q_rank), 0u},
     {offsetof(attn_pub, query), ATTN_PUB_TOKENS, offsetof(attn_pub, query_width), 0u},
    {offsetof(attn_pub, raw_kv), ATTN_PUB_TOKENS, offsetof(attn_pub, kv_width), 0u},
     {offsetof(attn_pub, compressed_kv), offsetof(attn_pub, compressed_count),
     offsetof(attn_pub, compressed_stride), 0u},
    {offsetof(attn_pub, indexer_kv), offsetof(attn_pub, indexer_count), offsetof(attn_pub, indexer_stride), 0u},
     {offsetof(attn_pub, index_query), ATTN_PUB_TOKENS, offsetof(attn_pub, index_query_stride), 0u},
     {offsetof(attn_pub, index_weights), ATTN_PUB_TOKENS, offsetof(attn_pub, index_weight_stride), 0u},
     {offsetof(attn_pub, attention_values), ATTN_PUB_TOKENS, offsetof(attn_pub, query_width), 0u},
     {offsetof(attn_pub, core_output), ATTN_PUB_TOKENS, offsetof(attn_pub, core_output_width), 0u},
     {offsetof(attn_pub, envelope_output), ATTN_PUB_TOKENS, offsetof(attn_pub, envelope_output_width), 0u},
};
static const char *runtime_attention_name(const char *const *names, unsigned int maximum, unsigned int value) {
    return value <= maximum ? names[value] : "invalid";
}
static void runtime_attention_name_copy(char *out, size_t capacity, const char *const *names,
                                        unsigned int maximum, unsigned int value) {
    yvex_core_text_copy(out, capacity, runtime_attention_name(names, maximum, value));
}
static void runtime_attention_u64_raise(unsigned long long *maximum, unsigned long long candidate) {
    if (candidate > *maximum) *maximum = candidate;
}
typedef struct {
    yvex_sha256 hash;
    yvex_runtime_trace_policy policy;
    unsigned long long stage_count, value_count;
} runtime_attention_trace;
static int runtime_attention_trace_span(runtime_attention_trace *trace, const char *name,
                                        const void *values, unsigned long long rows,
                                        unsigned long long width, size_t element_size) {
    unsigned long long count, next;
    if (!rows || !width) return YVEX_OK;
    if (!trace || !name || !values || !element_size || !yvex_core_u64_mul(rows, width, &count) ||
        count > (unsigned long long)(SIZE_MAX / element_size) ||
        !yvex_core_u64_add(trace->value_count, count, &next) || !yvex_sha256_update_text(&trace->hash, name) ||
        !yvex_sha256_update_u64(&trace->hash, rows) || !yvex_sha256_update_u64(&trace->hash, width) ||
        !yvex_sha256_update(&trace->hash, values, (size_t)count * element_size))
        return YVEX_ERR_STATE;
    trace->value_count = next;
    trace->stage_count++;
    return YVEX_OK;
}
static int runtime_attention_trace_rolling(runtime_attention_trace *trace, const char *name,
                                           const yvex_attention_rolling_state_output *state) {
    int rc;
    if (!state || !state->present) return YVEX_OK;
    if (!yvex_sha256_update_text(&trace->hash, name) ||
        !yvex_sha256_update_u64(&trace->hash, state->schema_version) ||
        !yvex_sha256_update_u64(&trace->hash, state->kind) ||
        !yvex_sha256_update_u64(&trace->hash, state->layer_index) ||
        !yvex_sha256_update_u64(&trace->hash, state->next_token_position) ||
        !yvex_sha256_update_u64(&trace->hash, state->ratio) ||
        !yvex_sha256_update_u64(&trace->hash, state->previous_fill) ||
        !yvex_sha256_update_u64(&trace->hash, state->current_fill) ||
        !yvex_sha256_update_u64(&trace->hash, state->cursor) ||
        !yvex_sha256_update_u64(&trace->hash, state->overlap) ||
        !yvex_sha256_update_u64(&trace->hash, state->rotated))
        return YVEX_ERR_STATE;
    rc = runtime_attention_trace_span(trace, "rolling_kv", state->kv_state,
                                      state->kv_state_extent, 1ull, sizeof(float));
    if (rc == YVEX_OK)
        rc = runtime_attention_trace_span(trace, "rolling_score", state->score_state,
                                          state->score_state_extent, 1ull, sizeof(float));
    return rc;
}
static int runtime_attention_trace_capture(void *context, yvex_backend_kind backend,
                                           const yvex_attention_publication *publication, yvex_error *err) {
    runtime_attention_trace *trace = (runtime_attention_trace *)context;
    const unsigned char *base = (const unsigned char *)publication;
    unsigned int first, index;
    int rc = YVEX_OK;
    if (!trace || !publication || !publication->complete)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.trace",
                              "trace callback requires a complete production publication");
    if (!yvex_sha256_update_u64(&trace->hash, backend) ||
        !yvex_sha256_update_u64(&trace->hash, publication->layer_index))
        rc = YVEX_ERR_STATE;
    first = trace->policy == YVEX_RUNTIME_TRACE_FULL ? 0u :
            trace->policy == YVEX_RUNTIME_TRACE_STAGES ? 1u : 9u;
    for (index = first; rc == YVEX_OK &&
                        index < sizeof(runtime_attention_trace_fields) /
                                    sizeof(runtime_attention_trace_fields[0]); ++index) {
        const runtime_attention_trace_field *field = &runtime_attention_trace_fields[index];
        const float *values = *(float *const *)(base + field->values);
        unsigned long long rows = *(const unsigned long long *)(base + field->rows);
        unsigned long long width = *(const unsigned long long *)(base + field->width);
        if (!width && field->fallback_width)
            width = *(const unsigned long long *)(base + field->fallback_width);
        rc = runtime_attention_trace_span(
            trace, runtime_attention_trace_stage_names[index], values, rows, width, sizeof(float));
    }
    if (trace->policy == YVEX_RUNTIME_TRACE_SUMMARY && rc == YVEX_OK)
        rc = runtime_attention_trace_span(trace, "raw_kv", publication->raw_kv,
                                          publication->token_count, publication->kv_width,
                                          sizeof(float));
    if (trace->policy == YVEX_RUNTIME_TRACE_FULL && rc == YVEX_OK)
        rc = runtime_attention_trace_span(trace, "compressed_positions",
            publication->compressed_positions, publication->compressed_count, 1ull,
            sizeof(unsigned long long));
    if (trace->policy == YVEX_RUNTIME_TRACE_FULL && rc == YVEX_OK)
        rc = runtime_attention_trace_span(trace, "indexer_positions",
            publication->indexer_positions, publication->indexer_count, 1ull,
            sizeof(unsigned long long));
    if (trace->policy == YVEX_RUNTIME_TRACE_FULL && rc == YVEX_OK)
        rc = runtime_attention_trace_span(trace, "topk_positions", publication->topk_positions,
            publication->token_count, publication->topk_stride, sizeof(unsigned long long));
    if (trace->policy == YVEX_RUNTIME_TRACE_FULL && rc == YVEX_OK && publication->topk_counts)
        rc = runtime_attention_trace_span(trace, "topk_counts", publication->topk_counts,
                                          publication->token_count, 1ull,
                                          sizeof(unsigned long long));
    if (trace->policy >= YVEX_RUNTIME_TRACE_STAGES && rc == YVEX_OK)
        rc = runtime_attention_trace_rolling(trace, "main_rolling", &publication->next_main_rolling_state);
    if (trace->policy >= YVEX_RUNTIME_TRACE_STAGES && rc == YVEX_OK)
        rc = runtime_attention_trace_rolling(trace, "indexer_rolling", &publication->next_indexer_rolling_state);
    if (rc != YVEX_OK)
        return runtime_refuse(err, rc, "runtime.attention.trace", "production trace evidence hashing failed");
    return YVEX_OK;
}
static int runtime_attention_trace_begin(runtime_attention_trace *trace,
                                         const yvex_graph_attention_operator_request *request,
                                         const char *selected_mode, yvex_error *err) {
    memset(trace, 0, sizeof(*trace));
    trace->policy = request->trace_policy;
    yvex_sha256_init(&trace->hash);
    if (!yvex_sha256_update_text(&trace->hash, "yvex.runtime.attention.execution-evidence.v3") ||
        !yvex_sha256_update_u64(&trace->hash, request->operator_action) ||
        !yvex_sha256_update_u64(&trace->hash, request->backend) ||
        !yvex_sha256_update_u64(&trace->hash, request->phase) ||
        !yvex_sha256_update_u64(&trace->hash, request->trace_policy) ||
        !yvex_sha256_update_text(&trace->hash, selected_mode))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.trace", "identity failed");
    return YVEX_OK;
}
static int runtime_attention_trace_finish(runtime_attention_trace *trace,
                                          yvex_graph_attention_operator_result *result, yvex_error *err) {
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    if (!trace || !result || !result->execution_descriptor_identity[0] ||
        !yvex_sha256_update_text(&trace->hash, result->execution_descriptor_identity) ||
        !yvex_sha256_update_text(&trace->hash, result->probe.tensor_output_digest) ||
        !yvex_sha256_update_text(&trace->hash, result->probe.state_delta_digest) ||
        !yvex_sha256_update_text(&trace->hash, result->probe.attention_execution_identity) ||
        !yvex_sha256_update_u64(&trace->hash, result->execution_dispatch_count) ||
        !yvex_sha256_final(&trace->hash, digest))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.trace",
                              "execution evidence identity finalization failed");
    yvex_sha256_hex(digest, result->execution_evidence_digest);
    result->trace_stage_count = trace->stage_count; result->trace_value_count = trace->value_count;
    return YVEX_OK;
}
static void runtime_attention_result_initialize(const yvex_graph_attention_operator_request *request,
                                                yvex_graph_attention_operator_result *result) {
    const yvex_runtime_family_adapter *adapter;
    *result = runtime_attention_result_default;
    if (!request) return;
    adapter = request->target ? yvex_runtime_family_adapter_find(request->target) : NULL;
    (void)snprintf(result->command, sizeof(result->command), "graph attention %s",
                   runtime_attention_name(runtime_attention_action_names, YVEX_RUNTIME_OPERATOR_RESIDENCY_INSPECT,
                                          (unsigned int)request->operator_action));
    if (request->target)
        yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    if (adapter) {
        yvex_core_text_copy(result->family, sizeof(result->family), adapter->family_name);
        if (request->probe == YVEX_ATTENTION_PROBE_CANONICAL_V2)
            yvex_core_text_copy(result->input_class, sizeof(result->input_class), "canonical_attention_probe");
    }
    if (request->compare_backends)
        yvex_core_text_copy(result->backend, sizeof(result->backend), "compare");
    else if (request->backend == YVEX_BACKEND_KIND_CPU ||
             request->backend == YVEX_BACKEND_KIND_CUDA)
        yvex_core_text_copy(result->backend, sizeof(result->backend), yvex_backend_kind_name(request->backend));
    if (request->scope == YVEX_ATTENTION_PROBE_SCOPE_QUICK ||
        request->scope == YVEX_ATTENTION_PROBE_SCOPE_FULL)
        yvex_core_text_copy(result->scope, sizeof(result->scope),
                            request->scope == YVEX_ATTENTION_PROBE_SCOPE_FULL ? "full" : "quick");
    if (request->artifact_path)
        yvex_core_text_copy(result->artifact_path, sizeof(result->artifact_path), request->artifact_path);
    if (request->runtime_binding_path)
        yvex_core_text_copy(result->runtime_binding_path, sizeof(result->runtime_binding_path),
                            request->runtime_binding_path);
    runtime_attention_name_copy(result->phase, sizeof(result->phase), runtime_attention_phase_names,
                                YVEX_RUNTIME_PHASE_ATTENTION_SPECULATIVE_VERIFY, request->phase);
    runtime_attention_name_copy(result->requested_mode, sizeof(result->requested_mode),
                                runtime_attention_mode_names, YVEX_RUNTIME_MODE_AUTO, request->mode);
    runtime_attention_name_copy(result->trace_policy, sizeof(result->trace_policy),
                                runtime_attention_trace_names, YVEX_RUNTIME_TRACE_FULL,
                                request->trace_policy);
    runtime_attention_name_copy(result->operation_scope, sizeof(result->operation_scope),
                                runtime_attention_scope_names, YVEX_RUNTIME_SCOPE_RELEASE_ATTENTION_SET,
                                request->operation_scope);
    if (request->operator_action == YVEX_RUNTIME_OPERATOR_BENCHMARK ||
        request->operator_action == YVEX_RUNTIME_OPERATOR_PROFILE ||
        request->operator_action == YVEX_RUNTIME_OPERATOR_QUALIFY)
        yvex_core_text_copy(result->benchmark_scope, sizeof(result->benchmark_scope),
                            "attention_component");
    yvex_core_text_copy(result->attention_class, sizeof(result->attention_class),
                        request->attention_class ? request->attention_class : "mixed");
    result->requested_token_count = request->token_count; result->requested_history_tokens = request->history_tokens;
    result->requested_layer_start = request->layer_start;
    result->requested_layer_count = request->select_layer ? request->layer_count : 0ull;
}
static int runtime_attention_result_bind(const yvex_runtime_model *model,
                                         yvex_graph_attention_operator_result *result, yvex_error *err) {
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_runtime_family_adapter *adapter = view ? view->adapter : NULL;
    yvex_runtime_model_summary model_summary;
    const yvex_runtime_binding_summary *binding = view ? view->binding : NULL;
    const yvex_artifact_physical_compatibility *compatibility =
        binding ? &binding->physical_compatibility : NULL;
    const yvex_materialization_summary *materialization =
        yvex_materialization_session_summary(view ? view->materialization : NULL);
    const yvex_runtime_descriptor_summary *descriptor =
        yvex_runtime_descriptor_summary_get(view ? view->descriptor : NULL);
    const yvex_attention_summary *attention = adapter && adapter->graph()
        ? adapter->graph()->plan_summary(view->attention) : NULL;
    if (!adapter || yvex_runtime_model_summary_copy(model, &model_summary, err) != YVEX_OK ||
        !model_summary.sealed || !model_summary.valid || !binding ||
        !compatibility || !compatibility->physical_payload_compatible ||
        !materialization || !materialization->committed || !descriptor || !attention)
        return 0;
    (void)snprintf(result->source_snapshot_identity,
                   sizeof(result->source_snapshot_identity), "%016llx",
                   binding->source_snapshot_identity);
    yvex_runtime_identity_copy(result->payload_identity, binding->payload_identity);
    yvex_runtime_identity_copy(result->artifact_identity, binding->artifact_identity);
    yvex_runtime_identity_copy(result->artifact_transform_identity, binding->artifact_transform_identity);
    yvex_runtime_identity_copy(result->logical_transform_identity, binding->logical_transform_identity);
    yvex_runtime_identity_copy(result->materialization_identity, materialization->plan_identity);
    yvex_runtime_identity_copy(result->logical_model_identity, descriptor->logical_model_identity);
    yvex_runtime_identity_copy(result->runtime_numeric_identity, descriptor->runtime_numeric_identity);
    yvex_runtime_identity_copy(result->runtime_descriptor_identity, descriptor->runtime_descriptor_identity);
    yvex_runtime_identity_copy(result->attention_plan_identity, attention->attention_plan_identity);
    yvex_runtime_identity_copy(result->runtime_binding_identity, binding->identity);
    yvex_runtime_identity_copy(result->runtime_model_identity, model_summary.runtime_model_identity);
    yvex_runtime_identity_copy(result->semantic_graph_identity, binding->semantic_graph_identity);
    yvex_runtime_identity_copy(result->executable_graph_identity, binding->executable_graph_identity);
    yvex_runtime_identity_copy(result->current_writer_plan_identity, compatibility->writer_plan_identity);
    yvex_runtime_identity_copy(result->payload_plan_identity, compatibility->payload_plan_identity);
    yvex_runtime_identity_copy(result->payload_byte_identity, compatibility->payload_byte_identity);
    result->main_layers_total = attention->layer_count;
    result->bindings_total = attention->required_binding_count;
    memcpy(&result->physical_payload_compatible, &compatibility->physical_payload_compatible, 8u * sizeof(int));
    result->capabilities = model_summary.capabilities;
    yvex_core_text_copy(result->execution_class, sizeof(result->execution_class), "production");
    yvex_core_text_copy(result->weights_class, sizeof(result->weights_class),
                                "admitted_external_artifact");
    result->artifact_hash_passes = model_summary.artifact_hash_passes;
    result->artifact_bytes_hashed = model_summary.artifact_bytes_hashed;
    result->artifact_identity_verified = model_summary.artifact_hash_passes == 1ull;
    memcpy(&result->runtime_model_builds, &model_summary.runtime_model_builds, 4u * sizeof(unsigned long long));
    memcpy(result->lifecycle_seconds, model_summary.lifecycle_seconds, sizeof(result->lifecycle_seconds));
    return 1;
}
static void runtime_attention_result_refuse(yvex_graph_attention_operator_result *result, int status,
                                            const yvex_error *err) {
    const char *message = err ? yvex_error_message(err) : "";
    const char *where = err ? yvex_error_where(err) : "";
    yvex_status code = err && yvex_error_is_set(err) ? yvex_error_code(err) : (yvex_status)status;
    result->completed = 0;
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->failure_code, sizeof(result->failure_code), yvex_status_name(code));
    yvex_core_text_copy(result->failure_where, sizeof(result->failure_where),
                                where[0] ? where : "runtime.attention");
    (void)snprintf(result->reason, sizeof(result->reason), "%s %s: %s",
                   yvex_status_name(code), result->failure_where,
                   message[0] ? message : "attention execution refused");
}
static int runtime_attention_execution_identity(
    const yvex_graph_attention_operator_request *request, const yvex_graph_attention_operator_result *result,
    char output[YVEX_SHA256_HEX_CAP]) {
    const char *prefix[] = {
        "yvex.runtime.attention-execution.v2", result->runtime_model_identity,
        result->runtime_binding_identity, result->execution_descriptor_identity};
    const char *suffix[] = {
        result->selected_mode, result->capture_bucket, result->cuda_launch_graph_identity,
        result->cuda_graph_exec_identity, result->probe.tensor_output_digest,
        result->probe.state_delta_digest, result->execution_evidence_digest};
    const unsigned long long values[] = {
        request->backend, request->phase, request->mode, request->operation_scope,
        request->operator_action, request->scope,
        request->token_count ? request->token_count : 1ull,
        request->repeat ? request->repeat : 1ull, request->maximum_host_bytes,
        request->maximum_device_bytes, (unsigned long long)request->require_mode};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t index;
    yvex_sha256_init(&hash);
    for (index = 0u; index < sizeof(prefix) / sizeof(prefix[0]); ++index)
        if (!yvex_sha256_update_text(&hash, prefix[index])) return 0;
    for (index = 0u; index < sizeof(values) / sizeof(values[0]); ++index)
        if (!yvex_sha256_update_u64(&hash, values[index])) return 0;
    for (index = 0u; index < sizeof(suffix) / sizeof(suffix[0]); ++index)
        if (!yvex_sha256_update_text(&hash, suffix[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
static int runtime_attention_execution_descriptor_identity(
    const yvex_graph_attention_operator_request *request, const yvex_runtime_model *model,
    const yvex_runtime_execution_session *session,
    const yvex_graph_attention_capacity_plan *capacity,
    const yvex_graph_attention_operator_result *result,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err) {
    const yvex_runtime_model_view *model_view = yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(session);
    const yvex_runtime_binding_summary *binding = model_view ? model_view->binding : NULL;
    yvex_runtime_residency_summary residency_storage;
    yvex_runtime_session_summary session_storage;
    yvex_graph_attention_state_summary state_storage;
    const yvex_runtime_residency_summary *residency = &residency_storage;
    const yvex_runtime_session_summary *session_summary = &session_storage;
    const yvex_graph_attention_state_summary *state = &state_storage;
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    char expected_workspace_identity[YVEX_SHA256_HEX_CAP];
    unsigned int component;
    if (!request || !binding || !capacity_summary || !session_view || !result->selected_mode[0] ||
        yvex_runtime_residency_snapshot(
            model_view->residency, &residency_storage, NULL, NULL, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(session, &session_storage, err) != YVEX_OK ||
        !session_view->attention_state_provider || session_view->attention_state_provider->summary(
            session_view->attention_state_provider->context, &state_storage, err) != YVEX_OK ||
        !state->sealed || state->invalidated)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.descriptor",
            "sealed model, session, state, residency, and mode facts are required");
    if (!capacity_summary->selected_layer_count)
        return runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.selection",
                              "selected attention layer or class is unavailable");
    if (yvex_runtime_workspace_identity_compute(
            result->runtime_model_identity, session_summary->backend,
            request->maximum_host_bytes, request->maximum_device_bytes,
            session_summary->workspace_bytes, session_summary->host_workspace_bytes,
            session_summary->device_workspace_bytes ? capacity_summary->identity : NULL,
            expected_workspace_identity, err) != YVEX_OK ||
        strcmp(expected_workspace_identity, session_summary->workspace_identity) != 0)
        return runtime_refuse(err, YVEX_ERR_FORMAT, "runtime.attention.workspace",
                              "session workspace identity disagrees with execution facts");
    {
        yvex_runtime_execution_descriptor_facts facts = {
            .schema_version = YVEX_RUNTIME_EXECUTION_DESCRIPTOR_SCHEMA_V2,
            .runtime_model_identity = result->runtime_model_identity, .runtime_binding_identity = binding->identity,
            .artifact_identity = binding->artifact_identity, .selected_mode = result->selected_mode,
            .runtime_numeric_identity = binding->runtime_numeric_identity, .capture_bucket = result->capture_bucket,
            .runtime_descriptor_identity = binding->runtime_descriptor_identity, .request_count = 1ull,
            .semantic_graph_identity = binding->semantic_graph_identity, .device_kind = session_summary->backend,
            .executable_graph_identity = binding->executable_graph_identity, .probe = request->probe,
            .residency_identity = residency->residency_identity, .probe_scope = request->scope,
            .workspace_identity = session_summary->workspace_identity,
            .capacity_plan_identity = capacity_summary->identity,
            .state_layout_identity = state->state_layout_identity, .trace_policy = request->trace_policy,
            .family_adapter_id = binding->family_adapter_id, .family_adapter_version = binding->family_adapter_version,
            .operation_scope = request->operation_scope, .phase = request->phase,
            .backend = request->backend, .requested_mode = request->mode,
            .compare_backends = request->compare_backends, .token_count = request->token_count,
            .start_position = request->history_tokens, .layer_start = capacity_summary->first_layer,
            .layer_count = capacity_summary->selected_layer_count,
            .selection_key = capacity_summary->selected_layer_count == 1ull
                                 ? yvex_graph_attention_capacity_plan_layer(
                                       capacity, capacity_summary->first_layer)->recipe.selection_key
                                 : ULLONG_MAX,
            .binding_count = capacity_summary->selected_binding_count,
            .maximum_compression_ratio = capacity_summary->maximum_compression_ratio,
            .maximum_topk_capacity = capacity_summary->maximum_topk_capacity,
            .maximum_host_bytes = request->maximum_host_bytes, .maximum_device_bytes = request->maximum_device_bytes,
            .residency_generation = session_summary->residency_generation,
            .resident_binding_count = session_summary->resident_binding_count,
            .resident_encoded_bytes = session_summary->resident_encoded_bytes,
            .workspace_bytes = session_summary->workspace_bytes, .state_allocated_bytes = state->allocated_bytes,
            .workspace_generation = session_summary->workspace_generation,
            .prepared_state_layers = state->prepared_layer_count, .state_generation = state->generation,
            .device_index = session_summary->device_index,
            .compute_capability_major = session_summary->compute_capability_major,
            .compute_capability_minor = session_summary->compute_capability_minor,
            .total_device_bytes = session_summary->total_device_bytes};
        for (component = 0u; component < YVEX_ATTENTION_STATE_BINDING_COUNT;
             ++component) {
            facts.state_component_entries[component] = state->components[component].entry_count;
            facts.state_component_capacities[component] =
                capacity_summary->components[component].capacity;
        }
        memcpy(facts.qtype_binding_counts, residency->qtype_binding_counts,
               sizeof(facts.qtype_binding_counts));
        memcpy(facts.qtype_bytes, residency->qtype_bytes, sizeof(facts.qtype_bytes));
        return yvex_runtime_execution_descriptor_identity_compute(&facts, output, err);
    }
}
static int runtime_attention_action_dispatches(yvex_runtime_operator_action action) {
    return action == YVEX_RUNTIME_OPERATOR_EXECUTE ||
           (action >= YVEX_RUNTIME_OPERATOR_STATE_EXERCISE &&
            action <= YVEX_RUNTIME_OPERATOR_QUALIFY);
}
static int runtime_attention_state_summary_publish(
    const yvex_runtime_execution_session *session, int validate,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    yvex_graph_attention_state_summary summary;
    yvex_runtime_state_residency_summary residency;
    if (!view || !view->attention_state_provider || view->attention_state_provider->summary(
            view->attention_state_provider->context, &summary, err) != YVEX_OK ||
        !view->state_residency ||
        yvex_runtime_state_residency_summary_copy(
            view->state_residency, &residency, err) != YVEX_OK ||
        !summary.sealed || !summary.persistent || summary.transaction_active ||
        !residency.sealed || residency.invalidated)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.state",
                              "sealed idle persistent attention state is required");
    yvex_runtime_identity_copy(result->state_layout_identity, summary.state_layout_identity);
    yvex_runtime_identity_copy(result->state_content_identity, summary.state_content_identity);
    yvex_runtime_identity_copy(result->state_residency_identity, residency.layout_identity);
    memcpy(&result->state_layer_count, &summary.layer_count, 2u * sizeof(unsigned long long));
    result->state_allocated_bytes = summary.allocated_bytes;
    result->state_capacity = summary.capacity;
    result->state_committed_sequence_length = summary.committed_sequence_length;
    result->state_next_position = summary.next_position;
    result->state_generation = summary.generation;
    result->state_residency_generation = residency.generation;
    result->state_device_bytes = residency.device_bytes;
    result->state_upload_bytes = residency.upload_bytes;
    result->state_upload_count = residency.upload_count;
    memcpy(&result->state_commit_count, &summary.commit_count, 4u * sizeof(unsigned long long));
    result->state_sealed = summary.sealed; result->state_persistent = summary.persistent;
    result->state_position_consistent = summary.position_consistent; result->state_cuda_ready = residency.cuda_ready;
    result->state_transaction_active = summary.transaction_active;
    result->state_validation_passed = validate; return YVEX_OK;
}
static int runtime_attention_mode_configure(const yvex_graph_attention_operator_request *request,
    yvex_runtime_execution_session *session, yvex_graph_attention_operator_result *result,
    yvex_runtime_execution_mode *selected_mode, yvex_error *err) {
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    yvex_backend *backend = view ? view->backend : NULL;
    const yvex_runtime_capabilities *capabilities = &result->capabilities;
    yvex_runtime_execution_mode selected = YVEX_RUNTIME_MODE_EAGER;
    yvex_backend_cuda_graph_capability capability;
    const char *reason = "explicit execution mode";
    const int cuda = request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA;
    char capture_bucket[YVEX_RUNTIME_CAPTURE_BUCKET_CAP];
    int graph_admitted, rc;
    if (!cuda) {
        if (request->capture_bucket)
            return runtime_refuse(err, YVEX_ERR_UNSUPPORTED, "runtime.attention.capture_bucket",
                                  "CPU eager attention has no CUDA capture bucket");
        yvex_core_text_copy(capture_bucket, sizeof(capture_bucket), "not-applicable");
        reason = request->mode == YVEX_RUNTIME_MODE_AUTO ? "auto-selected CPU eager mode"
                                                         : "explicit eager mode";
        goto publish;
    }
    if ((request->mode == YVEX_RUNTIME_MODE_PIECEWISE &&
         !capabilities->cuda_piecewise_graph_implemented) ||
        (request->mode == YVEX_RUNTIME_MODE_FULL && !capabilities->cuda_full_graph_implemented))
        return runtime_refuse(err, YVEX_ERR_UNSUPPORTED, "runtime.attention.mode",
                              request->mode == YVEX_RUNTIME_MODE_FULL
                                  ? "full CUDA graph execution is not implemented"
                                  : "piecewise CUDA graph execution is not implemented");
    selected = request->mode == YVEX_RUNTIME_MODE_AUTO ? YVEX_RUNTIME_MODE_EAGER : request->mode;
    if (request->mode == YVEX_RUNTIME_MODE_AUTO) {
        rc = yvex_backend_cuda_graph_query(backend, &capability, err);
        if (rc != YVEX_OK) return rc;
        graph_admitted = capability.state == YVEX_BACKEND_CUDA_GRAPH_OPEN &&
                         capability.edge_inventory_available && capability.async_memory_available &&
                         capability.async_copy_available && capability.pinned_host_memory_available;
        if (graph_admitted && capabilities->cuda_full_graph_implemented &&
            !getenv("YVEX_TEST_RUNTIME_AUTO_DISABLE_FULL"))
            selected = YVEX_RUNTIME_MODE_FULL;
        else if (graph_admitted && capabilities->cuda_piecewise_graph_implemented &&
                 !getenv("YVEX_TEST_RUNTIME_AUTO_DISABLE_PIECEWISE"))
            selected = YVEX_RUNTIME_MODE_PIECEWISE;
        reason = runtime_attention_auto_mode_reasons[selected];
    }
    if (runtime_attention_is_graph_action(request->operator_action) &&
        selected == YVEX_RUNTIME_MODE_EAGER)
        return runtime_refuse(err, YVEX_ERR_UNSUPPORTED, "runtime.attention.mode",
                              "CUDA graph action requires an admitted graph mode");
    if (selected == YVEX_RUNTIME_MODE_EAGER) {
        yvex_core_text_copy(capture_bucket, sizeof(capture_bucket), "not-applicable");
        if (request->capture_bucket)
            return runtime_refuse(err, YVEX_ERR_UNSUPPORTED, "runtime.attention.capture_bucket",
                                  "CUDA eager attention has no capture bucket");
    } else if (request->phase == YVEX_RUNTIME_PHASE_ATTENTION_DECODE) {
        yvex_core_text_copy(capture_bucket, sizeof(capture_bucket), "decode-1");
    } else if (snprintf(capture_bucket, sizeof(capture_bucket), "prefill-%llu",
                        request->token_count) <= 0) {
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.capture_bucket",
                              "prefill capture bucket construction failed");
    }
    if (request->capture_bucket && strcmp(request->capture_bucket, capture_bucket) != 0)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.capture_bucket",
                              "capture bucket must match exact phase/token geometry");
publish:
    yvex_core_text_copy(result->selected_mode, sizeof(result->selected_mode),
                        runtime_attention_mode_names[selected]);
    yvex_core_text_copy(result->selection_reason, sizeof(result->selection_reason), reason);
    yvex_core_text_copy(result->capture_bucket, sizeof(result->capture_bucket), capture_bucket);
    *selected_mode = selected;
    if (cuda && request->require_mode && request->mode == YVEX_RUNTIME_MODE_AUTO &&
        selected != YVEX_RUNTIME_MODE_FULL)
        return runtime_refuse(err, YVEX_ERR_UNSUPPORTED, "runtime.attention.mode",
                              "required AUTO mode could not select the full CUDA graph");
    return YVEX_OK;
}
static int runtime_attention_mode_bind_descriptor(const yvex_graph_attention_operator_request *request,
    yvex_runtime_execution_session *session, const yvex_graph_attention_operator_result *result,
    yvex_runtime_execution_mode mode, const yvex_graph_attention_capacity_plan *capacity,
    yvex_error *err) {
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    const yvex_graph_attention_capacity_summary *summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    yvex_backend_cuda_attention_mode selected;
    if (!request->compare_backends && request->backend != YVEX_BACKEND_KIND_CUDA)
        return YVEX_OK;
    if (!view || !summary)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.capacity",
                              "sealed capacity plan is required for backend configuration");
    selected = mode == YVEX_RUNTIME_MODE_FULL ? YVEX_BACKEND_CUDA_ATTENTION_FULL
        : mode == YVEX_RUNTIME_MODE_PIECEWISE ? YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE
                                              : YVEX_BACKEND_CUDA_ATTENTION_EAGER;
    return yvex_backend_cuda_attention_configure(
        view->backend, selected, result->execution_descriptor_identity, result->capture_bucket,
        summary->components[YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY].maximum_capacity,
        summary->components[YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY].maximum_capacity,
        summary->components[YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY].maximum_capacity, err);
}
static int runtime_attention_graph_summary(yvex_runtime_execution_session *session,
                                           int require_execution,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    yvex_backend_cuda_attention_graph_summary summary;
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    yvex_backend *backend = view ? view->backend : NULL;
    int rc;
    if (!backend || backend->kind != YVEX_BACKEND_KIND_CUDA) return YVEX_OK;
    rc = yvex_backend_cuda_attention_graph_summary_get(backend, &summary, err);
    if (rc != YVEX_OK) return rc;
    if (require_execution && summary.selected_mode != YVEX_BACKEND_CUDA_ATTENTION_EAGER &&
        (!summary.graph_count || !summary.capture_count || !summary.replay_count ||
         !summary.kernel_node_count ||
         (summary.selected_mode == YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE &&
          summary.piece_count < 2ull)))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.cuda_graph",
                              "requested CUDA graph mode produced incomplete graph evidence");
    yvex_runtime_identity_copy(result->cuda_launch_graph_identity, summary.launch_graph_identity);
    yvex_runtime_identity_copy(result->cuda_graph_exec_identity, summary.graph_exec_identity);
    yvex_core_text_copy(result->capture_bucket, sizeof(result->capture_bucket), summary.capture_bucket);
    (void)snprintf(result->cuda_driver, sizeof(result->cuda_driver), "%d", summary.driver_version);
    yvex_runtime_identity_copy(result->cuda_build_identity, summary.cuda_build_identity);
    memcpy(&result->cuda_graph_count, &summary.graph_count, 5u * sizeof(unsigned long long));
    memcpy(&result->cuda_graph_launch_count, &summary.launch_count, 5u * sizeof(unsigned long long));
    memcpy(&result->cuda_graph_update_count, &summary.update_count, 2u * sizeof(unsigned long long));
    memcpy(&result->cuda_graph_capture_elapsed_ns, &summary.capture_elapsed_ns, 4u * sizeof(unsigned long long));
    result->cuda_graph_invalidation_count = summary.invalidation_count;
    return YVEX_OK;
}
static int runtime_attention_progress(const yvex_graph_attention_operator_request *request,
                                      yvex_runtime_lifecycle_phase phase, unsigned long long completed,
                                      unsigned long long total, yvex_error *err) {
    if (request->progress && !request->progress(request->progress_context, phase, completed, total))
        return runtime_refuse(err, YVEX_ERR_CANCELLED, "runtime.attention",
                              "attention operator lifecycle was cancelled");
    return YVEX_OK;
}
static int runtime_attention_graph_lifecycle_partition(yvex_graph_attention_operator_result *result, yvex_error *err) {
    const double nested = result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_CAPTURE] +
        result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_INSTANTIATE];
    double *enclosing;
    if (nested == 0.0) return YVEX_OK;
    enclosing = result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP] > 0.0
                    ? &result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP]
                    : &result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_EXECUTION];
    if (*enclosing + 1.0e-9 < nested)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.timing",
                              "CUDA Graph subphase timing exceeds its enclosing dispatch");
    *enclosing = *enclosing > nested ? *enclosing - nested : 0.0;
    return YVEX_OK;
}
typedef struct {
    const yvex_attention_state_provider *provider;
    yvex_runtime_state_residency *residency;
    yvex_attention_operation_scope operation_scope;
    yvex_sha256 output_hash;
    unsigned long long output_values;
    unsigned long long active_layer_ordinal;
    int hash_output;
    int layer_active;
    char last_delta_identity[YVEX_SHA256_HEX_CAP];
} runtime_attention_state_bridge;
static int runtime_attention_state_begin(
    void *context, unsigned long long layer_ordinal, const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *initial_history,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation, const yvex_attention_history_view **history,
    yvex_attention_failure *failure, yvex_error *err) {
    runtime_attention_state_bridge *bridge = (runtime_attention_state_bridge *)context;
    int rc;
    if (!bridge || !bridge->provider || !bridge->provider->begin)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.state",
                              "valid state bridge and bounded token range are required");
    rc = bridge->provider->begin(
        bridge->provider->context, layer_ordinal, layer, initial_history,
        token_position, token_count, cancellation, history, failure, err);
    if (rc == YVEX_OK) {
        bridge->active_layer_ordinal = layer_ordinal;
        bridge->layer_active = 1;
    }
    return rc;
}
static int runtime_attention_state_hash_output(
    runtime_attention_state_bridge *bridge, const yvex_attention_publication *publication) {
    const float *values;
    unsigned long long width, count, index;
    values = bridge->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                 ? publication->envelope_output : publication->core_output;
    width = bridge->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                ? publication->envelope_output_width : publication->core_output_width;
    if (!values || !width || !yvex_core_u64_mul(publication->token_count, width, &count))
        return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        if (!isfinite(values[index]))
            return 0;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(&bridge->output_hash, (unsigned long long)bits))
            return 0;
    }
    bridge->output_values += count;
    return 1;
}
static int runtime_attention_state_stage(
    void *context, const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation, char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err) {
    runtime_attention_state_bridge *bridge = (runtime_attention_state_bridge *)context;
    int rc;
    if (!bridge || !bridge->provider || !bridge->provider->stage ||
        !bridge->layer_active || !publication ||
        (bridge->hash_output && !runtime_attention_state_hash_output(bridge, publication)))
        return runtime_refuse(err, YVEX_ERR_FORMAT, "runtime.attention.state",
                              "complete attention output publication is required");
    rc = bridge->provider->stage(bridge->provider->context, publication, cancellation,
        state_delta_identity, failure, err);
    if (rc == YVEX_OK && bridge->residency)
        rc = yvex_runtime_state_residency_stage(
            bridge->residency, bridge->provider,
            bridge->active_layer_ordinal, err);
    if (rc == YVEX_OK && state_delta_identity)
        yvex_runtime_identity_copy(bridge->last_delta_identity, state_delta_identity);
    if (rc == YVEX_OK) bridge->layer_active = 0;
    return rc;
}
static int runtime_attention_state_abort(
    void *context, yvex_attention_failure *failure, yvex_error *err) {
    runtime_attention_state_bridge *bridge = (runtime_attention_state_bridge *)context;
    int rc;
    if (!bridge || !bridge->provider || !bridge->provider->abort)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.state",
                              "attention state bridge is required for abort");
    rc = bridge->provider->abort(bridge->provider->context, failure, err);
    bridge->layer_active = 0;
    return rc;
}
static yvex_attention_probe_state_provider
runtime_attention_state_provider(runtime_attention_state_bridge *bridge) {
    yvex_attention_probe_state_provider provider;
    memset(&provider, 0, sizeof(provider));
    provider.context = bridge; provider.begin = runtime_attention_state_begin;
    provider.stage = runtime_attention_state_stage; provider.abort = runtime_attention_state_abort;
    return provider;
}
int yvex_runtime_session_prepare_attention_scope_state(
    yvex_runtime_execution_session *session, yvex_runtime_model *model,
    yvex_tensor_scope scope, const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err) {
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(session);
    const yvex_runtime_model_view *model_view = yvex_runtime_model_view_get(model);
    const yvex_attention_state_provider *provider = session_view
        ? (scope == YVEX_TENSOR_SCOPE_DRAFT
               ? session_view->draft_attention_state_provider
               : session_view->attention_state_provider)
        : NULL;
    const yvex_attention_plan *attention = model_view
        ? (scope == YVEX_TENSOR_SCOPE_DRAFT
               ? model_view->draft_attention : model_view->attention)
        : NULL;
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    const yvex_graph_attention_capacity_summary *capacity_summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    unsigned long long index, prepared = 0ull;
    if ((scope != YVEX_TENSOR_SCOPE_GLOBAL && scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        !session_view || session_view->model != model || !provider || !provider->prepare ||
        !attention_summary || !capacity_summary ||
        strcmp(attention_summary->attention_plan_identity,
               capacity_summary->attention_plan_identity) != 0)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.capacity",
                              "matching attention and capacity plans are required");
    for (index = 0ull; index < capacity_summary->layer_count; ++index) {
        const yvex_graph_attention_capacity_layer *layer_capacity =
            yvex_graph_attention_capacity_plan_layer(capacity, index);
        const yvex_attention_layer_plan *layer = yvex_attention_plan_layer_at(attention, index);
        yvex_attention_probe_history *seed = NULL;
        const yvex_attention_history_view *initial = NULL;
        int rc;
        if (!layer || !layer_capacity)
            return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.capacity",
                                  "capacity-plan layer lookup failed");
        if (!layer_capacity->selected) continue;
        if (layer_capacity->recipe.initial_position) {
            rc = yvex_attention_probe_history_open(&seed, layer, attention_summary,
                layer_capacity->recipe.initial_position, &initial, err);
            if (rc != YVEX_OK) return rc;
        }
        rc = provider->prepare(
            provider->context, index, &layer_capacity->recipe, initial, failure, err);
        yvex_attention_probe_history_close(&seed);
        if (rc != YVEX_OK) return rc;
        prepared++;
    }
    if (prepared != capacity_summary->selected_layer_count)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.capacity",
                              "capacity-plan selection was not prepared completely");
    {
        yvex_runtime_model_failure model_failure;
        return yvex_runtime_session_prepare_persistent_scope_state(
            session, scope, capacity, &model_failure, err);
    }
}
int yvex_runtime_session_prepare_attention_probe_state(
    yvex_runtime_execution_session *session, yvex_runtime_model *model,
    const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err)
{
    return yvex_runtime_session_prepare_attention_scope_state(
        session, model, YVEX_TENSOR_SCOPE_GLOBAL, capacity, failure, err);
}
typedef struct {
    yvex_attention_state_provider state_provider;
    int state_provider_ready;
    runtime_attention_state_bridge bridge;
    yvex_attention_probe_state_provider provider;
    char tensor_digest[YVEX_SHA256_HEX_CAP];
    char state_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long start_position;
    unsigned long long payload_bytes_read, kernel_launches, peak_device_bytes;
    unsigned long long bindings_executed, topk_selected;
} runtime_attention_phase_lane;
typedef struct {
    unsigned long long layer_ordinal, selection_key;
    yvex_attention_class attention_class;
    unsigned long long compression_ratio;
    runtime_attention_phase_lane lane[2];
} runtime_attention_phase_pair;
typedef struct {
    runtime_attention_phase_pair *pairs;
    unsigned long long pair_count;
} runtime_attention_phase_context;
static int runtime_attention_phase_lane_close(runtime_attention_phase_lane *lane, yvex_error *err);
static int runtime_attention_phase_lane_evidence_begin(
    runtime_attention_phase_lane *lane, unsigned long long layer_ordinal,
    yvex_attention_operation_scope operation_scope, yvex_error *err) {
    if (!lane || !lane->state_provider_ready)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                              "phase state provider is unavailable");
    memset(&lane->payload_bytes_read, 0, 5u * sizeof(lane->payload_bytes_read));
    lane->tensor_digest[0] = '\0'; lane->state_identity[0] = '\0';
    lane->bridge.output_values = 0ull; lane->bridge.last_delta_identity[0] = '\0';
    yvex_sha256_init(&lane->bridge.output_hash);
    if (!yvex_sha256_update_text(&lane->bridge.output_hash,
                                 "yvex.runtime.attention.phase-output.v1") ||
        !yvex_sha256_update_u64(&lane->bridge.output_hash, layer_ordinal) ||
        !yvex_sha256_update_u64(&lane->bridge.output_hash, (unsigned long long)operation_scope))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                              "phase output identity initialization failed");
    return YVEX_OK;
}
static int runtime_attention_phase_lane_open(
    runtime_attention_phase_lane *lane, const yvex_graph_family_api *graph,
    const yvex_attention_plan *plan, unsigned long long layer_ordinal,
    const yvex_attention_state_recipe *recipe,
    yvex_attention_operation_scope operation_scope, unsigned long long maximum_host_bytes,
    yvex_attention_failure *failure, yvex_error *err) {
    const yvex_attention_layer_plan *layer = yvex_attention_plan_layer_at(plan, layer_ordinal);
    const yvex_attention_summary *summary = yvex_attention_plan_summary(plan);
    yvex_attention_probe_history *seed = NULL;
    const yvex_attention_history_view *initial = NULL;
    int rc;
    memset(lane, 0, sizeof(*lane));
    if (!layer || !summary || !recipe || recipe->layer_index != layer_ordinal)
        return runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.phase",
                              "phase lane recipe is unavailable or stale");
    rc = yvex_attention_state_provider_open_persistent(
        graph, plan, maximum_host_bytes, &lane->state_provider, failure, err);
    if (rc != YVEX_OK) return rc;
    lane->state_provider_ready = 1;
    if (recipe->initial_position) {
        rc = yvex_attention_probe_history_open(
            &seed, layer, summary, recipe->initial_position, &initial, err);
        if (rc != YVEX_OK) return rc;
    }
    rc = lane->state_provider.prepare(
        lane->state_provider.context, layer_ordinal, recipe, initial, failure, err);
    yvex_attention_probe_history_close(&seed);
    if (rc != YVEX_OK) return rc;
    lane->bridge.provider = &lane->state_provider;
    lane->start_position = recipe->initial_position;
    lane->bridge.operation_scope = operation_scope;
    lane->bridge.hash_output = 1;
    lane->provider = runtime_attention_state_provider(&lane->bridge);
    return YVEX_OK;
}
static int runtime_attention_phase_lane_close(runtime_attention_phase_lane *lane, yvex_error *err) {
    int rc;
    if (!lane) return YVEX_OK;
    if (lane->state_provider_ready && getenv("YVEX_TEST_RUNTIME_PHASE_CLEANUP_FAILURE"))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase.cleanup",
                              "phase provider cleanup fault injected before release");
    if (lane->state_provider_ready && !lane->state_provider.release)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase.cleanup",
                              "phase provider release operation is unavailable");
    if (lane->state_provider_ready) {
        rc = lane->state_provider.release(&lane->state_provider.context, err);
        if (rc != YVEX_OK) return rc;
    }
    memset(lane, 0, sizeof(*lane));
    yvex_error_clear(err);
    return YVEX_OK;
}
static int runtime_attention_phase_lane_execute(
    runtime_attention_phase_lane *lane, yvex_runtime_model *model,
    const yvex_graph_family_api *graph, const yvex_attention_probe_request *base_request,
    unsigned long long layer_ordinal, unsigned long long token_count,
    int decode_steps, yvex_attention_failure *failure, yvex_error *err) {
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    yvex_attention_probe_request request = *base_request;
    unsigned long long step, steps = decode_steps ? token_count : 1ull;
    int rc = YVEX_OK;
    request.select_layer = 1;
    request.layer_ordinal = layer_ordinal;
    request.select_position = 1;
    request.state_provider = &lane->provider;
    for (step = 0ull; rc == YVEX_OK && step < steps; ++step) {
        yvex_attention_probe_result probe;
        memset(&probe, 0, sizeof(probe));
        probe.first_failing_layer = YVEX_ATTENTION_NO_LAYER;
        probe.first_failing_coordinate = YVEX_ATTENTION_NO_LAYER;
        request.token_position = lane->start_position + (decode_steps ? step : 0ull);
        request.token_count = decode_steps ? 1ull : token_count;
        rc = yvex_attention_probe_execute(graph, view ? view->attention : NULL, NULL,
            view ? view->materialization : NULL,
            view ? view->descriptor : NULL, &request, &probe, failure, err);
        if (rc != YVEX_OK)
            break;
        rc = lane->state_provider.commit(lane->state_provider.context, failure, err);
        if (rc != YVEX_OK)
            break;
        lane->payload_bytes_read += probe.payload_bytes_read;
        lane->kernel_launches += probe.kernel_launches;
        runtime_attention_u64_raise(&lane->peak_device_bytes, probe.peak_device_bytes);
        lane->bindings_executed = probe.bindings_executed;
        runtime_attention_u64_raise(&lane->topk_selected, probe.topk_selected);
    }
    if (rc == YVEX_OK) {
        unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
        if (!yvex_sha256_final(&lane->bridge.output_hash, digest))
            rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                                "phase output identity finalization failed");
        else {
            yvex_sha256_hex(digest, lane->tensor_digest);
            rc = lane->state_provider.identity(lane->state_provider.context, layer_ordinal,
                lane->state_identity, err);
        }
    }
    if (rc != YVEX_OK) {
        yvex_attention_failure primary_failure = failure ? *failure : (yvex_attention_failure){0};
        yvex_attention_failure cleanup_failure = {0};
        yvex_error primary_error;
        yvex_error cleanup_error;
        int primary_error_set = err && yvex_error_is_set(err);
        int cleanup_rc;
        if (err)
            primary_error = *err;
        yvex_error_clear(&cleanup_error);
        cleanup_rc = runtime_attention_state_abort(
            &lane->bridge, &cleanup_failure, &cleanup_error);
        if (cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (failure)
                *failure = cleanup_failure;
            if (err)
                *err = cleanup_error;
        } else {
            if (failure)
                *failure = primary_failure;
            if (err) {
                if (primary_error_set)
                    *err = primary_error;
                else
                    yvex_error_set(err, (yvex_status)rc, "runtime.attention.phase",
                                   primary_failure.reason ? primary_failure.reason
                                       : "attention phase execution failed before rollback");
            }
        }
    }
    return rc;
}
static int runtime_attention_phase_context_release(void **owner, yvex_error *err) {
    runtime_attention_phase_context *context = owner ? *owner : NULL;
    yvex_error first, current;
    unsigned long long pair;
    int rc = YVEX_OK, close_rc;
    if (!context) return YVEX_OK;
    for (pair = 0ull; pair < context->pair_count; ++pair) {
        for (unsigned int lane = 0u; lane < 2u; ++lane) {
            yvex_error_clear(&current);
            close_rc = runtime_attention_phase_lane_close(
                &context->pairs[pair].lane[lane], &current);
            if (rc == YVEX_OK && close_rc != YVEX_OK) {
                rc = close_rc;
                first = current;
            }
        }
    }
    if (rc != YVEX_OK) {
        if (err) *err = first;
        return rc;
    }
    free(context->pairs);
    free(context);
    *owner = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
static int runtime_attention_phase_context_open(
    runtime_attention_phase_context *context, const yvex_graph_family_api *graph,
    const yvex_attention_plan *plan, const yvex_graph_attention_capacity_plan *capacity,
    yvex_attention_operation_scope operation_scope, unsigned long long start_position,
    unsigned long long token_count,
    unsigned long long maximum_host_bytes, yvex_attention_failure *failure, yvex_error *err) {
    const yvex_graph_attention_capacity_summary *summary =
        yvex_graph_attention_capacity_plan_summary(capacity);
    unsigned long long layer, pair = 0ull;
    int seen[YVEX_ATTENTION_CLASS_HCA + 1] = {0}, rc = YVEX_OK;
    unsigned int lane;
    memset(context, 0, sizeof(*context));
    if (!summary || !summary->selected_layer_count ||
        summary->selected_layer_count > (unsigned long long)(SIZE_MAX / sizeof(*context->pairs)))
        return runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.phase",
                              "selected phase-equivalence recipes are unavailable");
    context->pair_count = summary->selected_layer_count > (unsigned long long)YVEX_ATTENTION_CLASS_HCA + 1ull
                              ? (unsigned long long)YVEX_ATTENTION_CLASS_HCA + 1ull : summary->selected_layer_count;
    context->pairs = calloc((size_t)context->pair_count, sizeof(*context->pairs));
    if (!context->pairs)
        return runtime_refuse(err, YVEX_ERR_NOMEM, "runtime.attention.phase",
                              "phase-equivalence lane allocation failed");
    for (layer = 0ull; rc == YVEX_OK && layer < summary->layer_count; ++layer) {
        const yvex_graph_attention_capacity_layer *selected =
            yvex_graph_attention_capacity_plan_layer(capacity, layer);
        const yvex_attention_layer_plan *layer_plan;
        yvex_attention_state_recipe phase_recipe;
        runtime_attention_phase_pair *entry;
        if (!selected || !selected->selected) continue;
        layer_plan = yvex_attention_plan_layer_at(plan, layer);
        if (!layer_plan) {
            rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                                "selected attention layer is unavailable");
            break;
        }
        if ((unsigned int)layer_plan->attention_class > YVEX_ATTENTION_CLASS_HCA ||
            seen[layer_plan->attention_class])
            continue;
        seen[layer_plan->attention_class] = 1;
        entry = &context->pairs[pair++];
        entry->layer_ordinal = layer;
        entry->selection_key = selected->recipe.selection_key;
        entry->attention_class = layer_plan->attention_class; entry->compression_ratio = layer_plan->compression_ratio;
        phase_recipe = selected->recipe; phase_recipe.initial_position = start_position;
        if (!yvex_core_u64_add(start_position, token_count, &phase_recipe.final_position) ||
            yvex_attention_state_recipe_seal(&phase_recipe, err) != YVEX_OK) {
            rc = runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.phase",
                                "phase-equivalence recipe range is invalid");
            break;
        }
        for (lane = 0u; rc == YVEX_OK && lane < 2u; ++lane)
            rc = runtime_attention_phase_lane_open(
                &entry->lane[lane], graph, plan, layer, &phase_recipe,
                operation_scope, maximum_host_bytes, failure, err);
    }
    if (rc == YVEX_OK && pair != context->pair_count)
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                            "selected phase-equivalence recipes changed during preparation");
    return rc;
}
static int runtime_attention_phase_equivalence(
    yvex_runtime_execution_session *session, yvex_runtime_model *model,
    const yvex_graph_family_api *graph, const yvex_attention_probe_request *base_request,
    runtime_attention_phase_context *context,
    unsigned long long start_position, unsigned long long token_count,
    yvex_runtime_model_failure *model_failure,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    yvex_sha256 output_hash, state_hash;
    yvex_attention_failure failure;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long pair_index;
    unsigned int lane;
    int acquired = 0, rc;
    if (!context || !context->pairs || !context->pair_count || token_count < 2ull)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.phase",
                              "phase equivalence requires at least two activation rows");
    yvex_sha256_init(&output_hash);
    yvex_sha256_init(&state_hash);
    if (!yvex_sha256_update_text(&output_hash, "yvex.runtime.attention.phase-output-set.v1") ||
        !yvex_sha256_update_text(&state_hash, "yvex.runtime.attention.phase-state-set.v1"))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                              "phase aggregate identity initialization failed");
    rc = yvex_runtime_session_begin(session, model_failure, err);
    acquired = rc == YVEX_OK;
    for (pair_index = 0ull;
         rc == YVEX_OK && pair_index < context->pair_count; ++pair_index) {
        runtime_attention_phase_pair *pair = &context->pairs[pair_index];
        runtime_attention_phase_lane *chunk = &pair->lane[0];
        runtime_attention_phase_lane *decode = &pair->lane[1];
        unsigned long long layer_ordinal = pair->layer_ordinal;
        for (lane = 0u; rc == YVEX_OK && lane < 2u; ++lane)
            rc = runtime_attention_phase_lane_evidence_begin(
                &pair->lane[lane], layer_ordinal, base_request->operation_scope, err);
        for (lane = 0u; rc == YVEX_OK && lane < 2u; ++lane)
            rc = runtime_attention_phase_lane_execute(
                &pair->lane[lane], model, graph, base_request, layer_ordinal, token_count,
                lane != 0u, &failure, err);
        if (rc == YVEX_OK && (strcmp(chunk->tensor_digest, decode->tensor_digest) != 0 ||
             strcmp(chunk->state_identity, decode->state_identity) != 0))
            rc = runtime_refuse(err, YVEX_ERR_FORMAT, "runtime.attention.phase",
                                "prefill and ordered decode publications disagree");
        if (rc == YVEX_OK && (!yvex_sha256_update_u64(&output_hash, pair->selection_key) ||
             !yvex_sha256_update_u64(&output_hash, layer_ordinal) ||
             !yvex_sha256_update_text(&output_hash, chunk->tensor_digest) ||
             !yvex_sha256_update_u64(&state_hash, pair->selection_key) ||
             !yvex_sha256_update_u64(&state_hash, layer_ordinal) ||
             !yvex_sha256_update_text(&state_hash, chunk->state_identity)))
            rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                                "phase aggregate identity update failed");
        if (rc == YVEX_OK) {
            result->probe.layers_executed++;
            if (pair->attention_class == YVEX_ATTENTION_CLASS_SWA)
                result->probe.swa_layers_executed++;
            else if (pair->attention_class == YVEX_ATTENTION_CLASS_CSA)
                result->probe.csa_layers_executed++;
            else if (pair->attention_class == YVEX_ATTENTION_CLASS_HCA) {
                result->probe.hca_layers_executed++;
                result->probe.hca_ratio = pair->compression_ratio;
            } else
                rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                                    "selected attention class is unsupported");
        }
        if (rc == YVEX_OK) {
            result->probe.bindings_executed += chunk->bindings_executed;
            result->probe.comparison_values += chunk->bridge.output_values;
            result->probe.payload_bytes_read +=
                chunk->payload_bytes_read + decode->payload_bytes_read;
            result->probe.kernel_launches += chunk->kernel_launches + decode->kernel_launches;
            runtime_attention_u64_raise(&result->probe.peak_device_bytes, chunk->peak_device_bytes);
            runtime_attention_u64_raise(&result->probe.peak_device_bytes, decode->peak_device_bytes);
            runtime_attention_u64_raise(&result->probe.topk_selected, chunk->topk_selected);
        }
    }
    if (rc == YVEX_OK && (!yvex_sha256_final(&output_hash, digest)))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                            "phase output-set identity finalization failed");
    if (rc == YVEX_OK)
        yvex_sha256_hex(digest, result->probe.tensor_output_digest);
    if (rc == YVEX_OK && !yvex_sha256_final(&state_hash, digest))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                            "phase state-set identity finalization failed");
    if (rc == YVEX_OK)
        yvex_sha256_hex(digest, result->probe.state_delta_digest);
    if (rc == YVEX_OK)
        rc = yvex_runtime_model_validate(model, model_failure, err);
    if (acquired) rc = yvex_runtime_session_finish(session, rc, err);
    if (rc == YVEX_OK) {
        yvex_sha256 evidence_hash;
        result->probe.comparison_available = 1;
        result->probe.comparison_passed = 1;
        result->probe.bitwise_equality_observed = 1;
        result->probe.comparison_finite_values = result->probe.comparison_values;
        if (base_request->compare_backends) {
            yvex_runtime_identity_copy(result->probe.cpu_output_digest,
                                            result->probe.tensor_output_digest);
            yvex_runtime_identity_copy(result->probe.cuda_output_digest,
                                            result->probe.tensor_output_digest);
        } else if (base_request->backend == YVEX_BACKEND_KIND_CPU) {
            yvex_runtime_identity_copy(result->probe.cpu_output_digest,
                                            result->probe.tensor_output_digest);
        } else {
            yvex_runtime_identity_copy(result->probe.cuda_output_digest,
                                            result->probe.tensor_output_digest);
        }
        yvex_sha256_init(&evidence_hash);
        if (!yvex_sha256_update_text(
                &evidence_hash, "yvex.runtime.attention.phase-equivalence.v2") ||
            !yvex_sha256_update_text(&evidence_hash, result->probe.tensor_output_digest) ||
            !yvex_sha256_update_text(&evidence_hash, result->probe.state_delta_digest) ||
            !yvex_sha256_update_u64(&evidence_hash, start_position) ||
            !yvex_sha256_update_u64(&evidence_hash, token_count) ||
            !yvex_sha256_final(&evidence_hash, digest))
            return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.phase",
                                  "phase evidence identity finalization failed");
        yvex_sha256_hex(digest, result->probe.attention_execution_identity);
        yvex_runtime_identity_copy(result->execution_evidence_digest,
                                        result->probe.attention_execution_identity);
    }
    return rc;
}
static int runtime_attention_batch_validate(yvex_runtime_execution_session *session,
    const yvex_attention_probe_request *request, yvex_error *err) {
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    const yvex_attention_state_provider *provider = view
        ? (request->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
               ? view->draft_attention_state_provider
               : view->attention_state_provider)
        : NULL;
    yvex_graph_attention_state_summary summary;
    if (!request->state_provider || !provider ||
        provider->summary(provider->context, &summary, err) != YVEX_OK)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.state",
                              "runtime attention state batch is unavailable");
    if (request->compare_backends ? summary.transaction_active : (!summary.transaction_active ||
                                     summary.candidate_active || !summary.staged_layer_count))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.state",
                              "runtime attention state batch is incomplete");
    return YVEX_OK;
}
int yvex_runtime_attention_probe_execute(yvex_runtime_execution_session *session,
    yvex_runtime_model *model, const yvex_attention_probe_request *request,
    yvex_attention_probe_result *result, yvex_runtime_model_failure *model_failure,
    yvex_error *err) {
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(session);
    yvex_attention_probe_request execution;
    yvex_attention_probe_state_provider state_provider;
    yvex_attention_probe_result probe = {0};
    yvex_attention_failure failure = {0};
    yvex_runtime_session_summary session_summary = {0};
    yvex_error execution_error;
    int acquired = 0, staged, rc, validation_rc;
    const yvex_attention_plan *attention;
    const yvex_attention_state_provider *persistent_state;
    yvex_runtime_state_residency *state_residency;
    if (!request || !result || !view || !view->binding || !view->adapter ||
        !view->adapter->graph || !session_view || session_view->model != model ||
        (request->tensor_scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         request->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT) ||
        !session_view->backend ||
        (yvex_backend_kind_of(session_view->backend) == YVEX_BACKEND_KIND_CUDA &&
         !session_view->attention_workspace) ||
        request->backend_context ||
        request->workspace || request->state_provider || request->logical_model_identity ||
        ((request->activation_view || request->device_view)
             ? request->probe != YVEX_ATTENTION_PROBE_UNSPECIFIED ||
                   !yvex_sha256_hex_valid(request->input_identity)
             : request->probe != YVEX_ATTENTION_PROBE_CANONICAL_V2) ||
        (request->compare_backends
             ? yvex_backend_kind_of(session_view->backend) != YVEX_BACKEND_KIND_CUDA
             : yvex_backend_kind_of(session_view->backend) != request->backend))
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention.execute",
                              "matching runtime owners and a semantic-only probe are required");
    attention = request->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                    ? view->draft_attention : view->attention;
    persistent_state = request->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                           ? session_view->draft_attention_state_provider
                           : session_view->attention_state_provider;
    state_residency = request->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                          ? session_view->draft_state_residency
                          : session_view->state_residency;
    if (!attention || !persistent_state)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.execute",
                              "selected attention stack has no runtime state authority");
    execution = *request;
    runtime_attention_state_bridge bridge = {
        .provider = persistent_state, .residency = state_residency,
        .operation_scope = execution.operation_scope};
    state_provider = runtime_attention_state_provider(&bridge);
    execution.backend_context = session_view->backend;
    execution.workspace = session_view->attention_workspace;
    execution.state_provider = &state_provider;
    execution.logical_model_identity = view->binding->logical_model_identity;
    probe.first_failing_layer = probe.first_failing_coordinate = YVEX_ATTENTION_NO_LAYER;
    staged = request->transaction_disposition == YVEX_ATTENTION_TRANSACTION_STAGE;
    rc = staged ? yvex_runtime_session_summary_copy(session, &session_summary, err)
                : yvex_runtime_session_begin(session, model_failure, err);
    if (rc == YVEX_OK && staged && (!session_summary.open || !session_summary.busy))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.execute",
                            "staged attention requires an owned session transaction");
    acquired = rc == YVEX_OK && !staged;
    if (rc == YVEX_OK)
        rc = yvex_attention_execute(
            view->adapter->graph(), attention, NULL, view->materialization,
            view->descriptor, &execution, &probe, &failure, err);
    if (rc != YVEX_OK && err && !yvex_error_is_set(err))
        yvex_error_set(err, (yvex_status)rc, "runtime.attention.execute",
                       failure.reason ? failure.reason
                                      : "production attention execution failed without a diagnostic");
    if (rc == YVEX_OK)
        rc = runtime_attention_batch_validate(session, &execution, err);
    if (rc != YVEX_OK && err) execution_error = *err;
    validation_rc = acquired ? yvex_runtime_model_validate(model, model_failure, err) : YVEX_OK;
    if (validation_rc != YVEX_OK) rc = validation_rc;
    else if (rc != YVEX_OK && err) *err = execution_error;
    if (acquired)
        rc = yvex_runtime_session_finish_scope(
            session, request->tensor_scope, request->transaction_disposition,
            rc, err);
    if (rc == YVEX_OK) *result = probe;
    return rc;
}
static int runtime_attention_state_probe(yvex_runtime_execution_session *session, yvex_runtime_model *model,
    const yvex_attention_probe_request *base, unsigned long long tokens, unsigned long long position,
    int perturb, yvex_attention_probe_result *result, yvex_runtime_model_failure *failure, yvex_error *err) {
    yvex_attention_probe_request request = *base;
    request.backend_context = NULL; request.workspace = NULL;
    request.logical_model_identity = NULL;
    request.token_count = tokens; request.token_position = position;
    request.perturb_input = perturb;
    return yvex_runtime_attention_probe_execute(session, model, &request, result, failure, err);
}
static int runtime_attention_operator_dispatch(
    const yvex_graph_attention_operator_request *request,
    yvex_runtime_execution_session *session, yvex_runtime_model *model,
    const yvex_graph_family_api *graph, yvex_attention_probe_request *probe_request,
    runtime_attention_phase_context *phase_context,
    unsigned long long first, unsigned long long count,
    unsigned long long warmup, unsigned long long total, double *samples, double *device_samples,
    yvex_runtime_model_failure *model_failure,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    unsigned long long offset;
    int rc = YVEX_OK;
    if (!runtime_attention_action_dispatches(request->operator_action)) return YVEX_OK;
    for (offset = 0ull; rc == YVEX_OK && offset < count; ++offset) {
        const unsigned long long index = first + offset;
        const unsigned long long sample_index = index >= warmup ? index - warmup : 0ull;
        double elapsed = 0.0, device_elapsed = 0.0;
        double *measurement = samples && index >= warmup ? &samples[sample_index] : &elapsed;
        rc = runtime_attention_progress(request, YVEX_RUNTIME_LIFECYCLE_EXECUTION, index, total, err);
        if (rc != YVEX_OK) break;
        result->execution_dispatch_count++;
        if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_EXERCISE) {
            const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
            yvex_graph_attention_state_summary state;
            yvex_attention_probe_result fresh, continuation, alternative;
            unsigned long long prefix = request->history_tokens ? request->history_tokens : 1ull;
            unsigned long long current = request->history_tokens ? probe_request->token_count
                                                                 : probe_request->token_count - 1ull;
            const unsigned long long started = yvex_core_monotonic_ns();
            if (!view || !view->attention_state_provider ||
                view->attention_state_provider->summary(
                    view->attention_state_provider->context, &state, err) != YVEX_OK ||
                !state.position_consistent)
                rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.state",
                                    "state exercise requires one consistent sequence position");
            if (rc == YVEX_OK)
                rc = runtime_attention_phase_equivalence(session, model, graph, probe_request, phase_context,
                    request->history_tokens,
                    probe_request->token_count, model_failure, result, err);
            probe_request->select_position = 1;
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, 1ull, 0ull, 0,
                                                   &fresh, model_failure, err);
            if (rc == YVEX_OK)
                rc = yvex_runtime_session_reset_persistent_state(session, model_failure, err);
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, prefix, 0ull, 0,
                                                   &result->probe, model_failure, err);
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, current, prefix, 0,
                                                   &continuation, model_failure, err);
            if (rc == YVEX_OK)
                rc = yvex_runtime_session_reset_persistent_state(session, model_failure, err);
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, prefix, 0ull, 1,
                                                   &result->probe, model_failure, err);
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, current, prefix, 0,
                                                   &alternative, model_failure, err);
            if (rc == YVEX_OK &&
                strcmp(continuation.tensor_output_digest, alternative.tensor_output_digest) == 0)
                rc = runtime_refuse(err, YVEX_ERR_FORMAT, "runtime.attention.state", "history was not causal");
            if (rc == YVEX_OK)
                rc = yvex_runtime_session_reset_persistent_state(session, model_failure, err);
            if (rc == YVEX_OK)
                rc = runtime_attention_state_probe(session, model, probe_request, 1ull, 0ull, 0,
                                                   &result->probe, model_failure, err);
            if (rc == YVEX_OK &&
                (strcmp(fresh.tensor_output_digest, result->probe.tensor_output_digest) != 0 ||
                 strcmp(fresh.state_delta_digest, result->probe.state_delta_digest) != 0))
                rc = runtime_refuse(err, YVEX_ERR_FORMAT, "runtime.attention.state", "clear disagrees with fresh");
            if (rc == YVEX_OK) {
                result->probe = continuation;
                result->state_read_after_write_verified = 1;
                result->state_clear_reuse_verified = 1;
            }
            *measurement = started ? runtime_seconds(yvex_core_monotonic_ns() - started) : 0.0;
        } else if (rc == YVEX_OK) {
            yvex_attention_probe_result probe;
            unsigned long long started;
            if (!yvex_core_u64_mul(index, probe_request->token_count, &started) ||
                !yvex_core_u64_add(request->history_tokens, started, &probe_request->token_position)) {
                rc = runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.state",
                                    "attention state token position overflowed");
                break;
            }
            started = yvex_core_monotonic_ns();
            rc = yvex_runtime_attention_probe_execute(
                session, model, probe_request, &probe, model_failure, err);
            *measurement = runtime_seconds(yvex_core_monotonic_ns() - started);
            if (rc == YVEX_OK) {
                result->probe = probe;
                device_elapsed = runtime_seconds(probe.cuda_device_execution_elapsed_ns);
                yvex_core_text_copy(result->first_failing_stage,
                    sizeof(result->first_failing_stage),
                    runtime_attention_name(runtime_attention_comparison_stage_names,
                        YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_SCORE,
                        (unsigned int)probe.first_failing_stage));
            }
            if (rc == YVEX_OK && device_samples && index >= warmup)
                device_samples[sample_index] = device_elapsed;
        }
        if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE &&
            index == 0ull) {
            const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
            rc = yvex_backend_cuda_attention_graph_registry_apply(
                view ? view->backend : NULL, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_UPDATE,
                &result->cuda_graph_registry_affected_count, err);
        }
        if (index >= warmup)
            result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_EXECUTION] += *measurement;
    }
    if (rc == YVEX_OK && first + count == total && samples)
        rc = yvex_runtime_benchmark_samples_finish(samples, device_samples, total - warmup,
            request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA, result, err);
    if (rc == YVEX_OK && first + count == total)
        rc = runtime_attention_progress(request, YVEX_RUNTIME_LIFECYCLE_EXECUTION, total, total, err);
    return rc;
}
typedef struct {
    yvex_runtime_model_summary model;
    yvex_materialization_access_summary materialization;
    yvex_runtime_residency_summary residency;
    yvex_runtime_session_summary session;
    yvex_backend_memory_stats backend;
    yvex_backend_host_workspace_summary host_workspace;
    yvex_attention_workspace_summary graph_workspace;
    yvex_core_allocation_epoch host_allocations;
} runtime_attention_warm_snapshot;
typedef struct {
    unsigned long long before, after;
} runtime_warm_counter;
enum {
    RUNTIME_WARM_HOST_ALLOCATIONS = 7,
    RUNTIME_WARM_HOST_REALLOCS = 8,
    RUNTIME_WARM_STABLE_COUNTERS = 10,
    RUNTIME_WARM_H2D_BYTES = 10,
    RUNTIME_WARM_D2H_BYTES = 11,
    RUNTIME_WARM_MONOTONIC_COUNTERS = 12,
    RUNTIME_WARM_COUNTERS = 15
};
static int runtime_attention_warm_snapshot_take(
    yvex_runtime_model *model, yvex_runtime_execution_session *session,
    runtime_attention_warm_snapshot *out, yvex_error *err) {
    const yvex_runtime_model_view *model_view = yvex_runtime_model_view_get(model);
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(session);
    const yvex_attention_workspace_summary *workspace = session_view
        ? yvex_attention_workspace_summary_get(session_view->attention_workspace) : NULL;
    memset(out, 0, sizeof(*out));
    if (!model_view || !session_view || !workspace ||
        yvex_runtime_model_summary_copy(model, &out->model, err) != YVEX_OK ||
        yvex_materialization_session_access_summary(model_view->materialization, &out->materialization, err) !=
            YVEX_OK ||
        yvex_runtime_residency_snapshot(model_view->residency, &out->residency, NULL, NULL, err) != YVEX_OK ||
        yvex_runtime_session_summary_copy(session, &out->session, err) != YVEX_OK ||
        yvex_backend_get_memory_stats(session_view->backend, &out->backend, err) != YVEX_OK ||
        !yvex_backend_host_workspace_summary_get(session_view->backend, &out->host_workspace))
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.warm",
                              "runtime warm-path counters could not be sampled");
    out->graph_workspace = *workspace;
    yvex_core_allocation_epoch_snapshot(&out->host_allocations);
    return YVEX_OK;
}
static int runtime_attention_warm_snapshot_publish(const runtime_attention_warm_snapshot *before,
    const runtime_attention_warm_snapshot *after,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    const runtime_warm_counter counters[RUNTIME_WARM_COUNTERS] = {
        {before->model.artifact_hash_passes, after->model.artifact_hash_passes},
{before->materialization.artifact_read_calls, after->materialization.artifact_read_calls},
        {before->residency.cold_artifact_read_calls, after->residency.cold_artifact_read_calls},
{before->backend.allocation_events, after->backend.allocation_events},
        {before->backend.release_events, after->backend.release_events},
{before->residency.cuda_upload_bytes, after->residency.cuda_upload_bytes},
        {before->host_workspace.allocation_count, after->host_workspace.allocation_count},
{before->host_allocations.allocation_events, after->host_allocations.allocation_events},
        {before->host_allocations.reallocation_events, after->host_allocations.reallocation_events},
{before->host_allocations.release_events, after->host_allocations.release_events},
        {before->backend.h2d_bytes, after->backend.h2d_bytes}, {before->backend.d2h_bytes, after->backend.d2h_bytes},
{before->graph_workspace.capacity_bytes, after->graph_workspace.capacity_bytes},
{before->session.residency_generation, after->session.residency_generation},
        {before->session.workspace_generation, after->session.workspace_generation} };
unsigned int index, regressed = before->host_allocations.overflowed ||
                                    after->host_allocations.overflowed;
    unsigned long long host_allocations, host_reallocations;
    int transitioned = 0;
    for (index = 0u; index < RUNTIME_WARM_COUNTERS; ++index) {
        regressed |= index < RUNTIME_WARM_MONOTONIC_COUNTERS &&
                     counters[index].after < counters[index].before;
        transitioned |= (index < RUNTIME_WARM_STABLE_COUNTERS ||
                          index >= RUNTIME_WARM_MONOTONIC_COUNTERS) &&
                         counters[index].after != counters[index].before;
    }
    if (regressed)
        return runtime_refuse(
            err, YVEX_ERR_STATE, "runtime.attention.warm", "runtime warm-path counters regressed");
    host_allocations = counters[RUNTIME_WARM_HOST_ALLOCATIONS].after - counters[RUNTIME_WARM_HOST_ALLOCATIONS].before;
    host_reallocations = counters[RUNTIME_WARM_HOST_REALLOCS].after - counters[RUNTIME_WARM_HOST_REALLOCS].before;
    if (host_allocations > ULLONG_MAX - host_reallocations)
        return runtime_refuse(
            err, YVEX_ERR_BOUNDS, "runtime.attention.warm", "warm host allocation evidence overflowed");
    if (transitioned)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.warm",
                              "steady-state runtime performed a forbidden resource transition");
    result->warm_artifact_hash_passes = 0ull;
    result->warm_weight_artifact_reads = 0ull;
    result->warm_weight_upload_bytes = 0ull;
    result->warm_host_allocations = 0ull;
    result->warm_device_allocations = 0ull;
    result->warm_device_frees = 0ull;
    result->warm_h2d_bytes = counters[RUNTIME_WARM_H2D_BYTES].after - counters[RUNTIME_WARM_H2D_BYTES].before;
    result->warm_d2h_bytes = counters[RUNTIME_WARM_D2H_BYTES].after - counters[RUNTIME_WARM_D2H_BYTES].before;
    return YVEX_OK;
}
static int runtime_attention_session_publish(const yvex_runtime_execution_session *session,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    yvex_runtime_session_summary summary;
    if (yvex_runtime_session_summary_copy(session, &summary, err) != YVEX_OK)
        return runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention",
                              "runtime session summary disappeared before publication");
    yvex_runtime_identity_copy(result->residency_identity, summary.residency_identity);
    yvex_runtime_identity_copy(result->workspace_identity, summary.workspace_identity);
    memcpy(&result->resident_binding_count, &summary.resident_binding_count, 4u * sizeof(unsigned long long));
    result->workspace_bytes = summary.workspace_bytes;
    memcpy(&result->pinned_host_bytes, &summary.host_workspace_bytes, 2u * sizeof(unsigned long long));
    result->pinned_host_residency = summary.host_workspace_pinned;
    memcpy(&result->upload_bytes, &summary.upload_bytes, 2u * sizeof(unsigned long long));
    result->capabilities = summary.capabilities;
    yvex_core_text_copy(result->probe.cuda_device, sizeof(result->probe.cuda_device), summary.device_name);
    result->probe.cuda_compute_capability_major = summary.compute_capability_major;
    result->probe.cuda_compute_capability_minor = summary.compute_capability_minor;
    result->attention_cuda_execution_ready = summary.capabilities.cuda_prefill_eager_ready &&
        summary.capabilities.cuda_decode_eager_ready;
    return YVEX_OK;
}
static int runtime_attention_registry_publish(const yvex_graph_attention_operator_request *request,
    yvex_runtime_execution_session *session,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    const yvex_runtime_session_view *view = yvex_runtime_session_view_get(session);
    yvex_backend *backend = view ? view->backend : NULL;
    yvex_backend_cuda_attention_graph_entry entry;
    unsigned long long count = 0ull, affected = 0ull;
    const int registry_action = request->operator_action >= YVEX_RUNTIME_OPERATOR_GRAPH_LIST &&
                                request->operator_action <= YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE;
    int rc = YVEX_OK;
    if (runtime_attention_is_graph_action(request->operator_action))
        yvex_core_text_copy(result->cuda_graph_registry_scope,
                            sizeof(result->cuda_graph_registry_scope), "ephemeral-process-session");
    if (registry_action)
        rc = yvex_backend_cuda_attention_graph_registry_count(backend, &count, err);
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT && !count)
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.cuda_graph",
                            "process-local CUDA graph registry is empty");
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT)
        rc = yvex_backend_cuda_attention_graph_registry_get(backend, 0ull, &entry, err);
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_INVALIDATE)
        rc = yvex_backend_cuda_attention_graph_registry_apply(
            backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_INVALIDATE, &affected, err);
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE)
        rc = yvex_backend_cuda_attention_graph_registry_apply(
            backend, YVEX_BACKEND_CUDA_GRAPH_REGISTRY_RELEASE, &affected, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_graph_summary(
            session, runtime_attention_action_dispatches(request->operator_action) &&
                         request->operator_action != YVEX_RUNTIME_OPERATOR_GRAPH_INVALIDATE &&
                         request->operator_action != YVEX_RUNTIME_OPERATOR_GRAPH_RELEASE,
            result, err);
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE &&
        (!result->cuda_graph_update_count || result->cuda_graph_update_pending_count ||
         !result->cuda_graph_last_update_elapsed_ns))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.cuda_graph",
                            "CUDA graph update did not complete through production replay");
    if (rc == YVEX_OK && registry_action)
        rc = yvex_backend_cuda_attention_graph_registry_count(backend, &result->cuda_graph_registry_count, err);
    if (rc == YVEX_OK && affected)
        result->cuda_graph_registry_affected_count = affected;
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_INSPECT) {
        result->cuda_graph_registry_index = entry.index;
        result->cuda_graph_entry_state = entry.graph.state;
        result->cuda_graph_entry_reason = entry.graph.reason;
        result->cuda_graph_entry_capture_mode = entry.graph.capture_mode;
        result->cuda_graph_entry_uploaded = entry.graph.uploaded;
        result->cuda_graph_entry_update_requested = entry.update_requested;
        yvex_core_text_copy(result->cuda_graph_entry_compatibility_identity,
                            sizeof(result->cuda_graph_entry_compatibility_identity),
                            entry.compatibility_identity);
    }
    return rc;
}
static int runtime_attention_publish_result(const yvex_graph_attention_operator_request *request,
    yvex_runtime_execution_session *session, runtime_attention_trace *trace,
    unsigned long long warmup, unsigned long long repeat, yvex_graph_attention_operator_result *result,
    int rc, yvex_error *err) {
    unsigned long long started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = runtime_attention_progress(request, YVEX_RUNTIME_LIFECYCLE_PUBLICATION, 0ull, 1ull, err);
    if (rc == YVEX_OK) rc = runtime_attention_registry_publish(request, session, result, err);
    if (rc == YVEX_OK) rc = runtime_attention_session_publish(session, result, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_state_summary_publish(
            session, request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_VALIDATE, result, err);
    result->repeat_count = repeat;
    result->warmup_count = warmup;
    if (rc == YVEX_OK) rc = runtime_attention_trace_finish(trace, result, err);
    if (rc == YVEX_OK && !runtime_attention_execution_identity(request, result, result->execution_identity))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention",
                            "runtime execution identity serialization failed");
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_PUBLICATION] += runtime_seconds(
        yvex_core_monotonic_ns() - started);
    if (rc == YVEX_OK)
        rc = runtime_attention_progress(request, YVEX_RUNTIME_LIFECYCLE_PUBLICATION, 1ull, 1ull, err);
    return rc;
}
static int runtime_attention_cleanup(yvex_runtime_cleanup_lease **lease, double *samples,
    yvex_graph_attention_operator_result *result, int status, yvex_error *err) {
    unsigned long long started = yvex_core_monotonic_ns();
    yvex_error cleanup;
    int cleanup_rc;
    yvex_error_clear(&cleanup);
    cleanup_rc = yvex_runtime_cleanup_lease_close(lease, &cleanup);
    if (cleanup_rc != YVEX_OK) {
        status = cleanup_rc;
        if (err) *err = cleanup;
    }
    free(samples);
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_CLEANUP] +=
        runtime_seconds(yvex_core_monotonic_ns() - started);
    return status;
}
static int runtime_attention_open(const yvex_graph_attention_operator_request *request,
    const yvex_runtime_family_adapter *adapter, yvex_runtime_cleanup_lease **cleanup,
    yvex_runtime_model **model, yvex_runtime_execution_session **session,
    yvex_runtime_execution_mode *selected_mode, yvex_runtime_model_failure *failure,
    yvex_graph_attention_operator_result *result, yvex_error *err) {
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    unsigned long long started;
    int rc;
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = adapter->target_id;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    model_request.progress = request->progress;
    model_request.progress_context = request->progress_context;
    rc = yvex_runtime_cleanup_lease_acquire(
        cleanup, &model_request, NULL, model, NULL, failure, err);
    if (rc == YVEX_OK && !runtime_attention_result_bind(*model, result, err))
        rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention",
                            "model summary is incomplete");
    session_request.backend = request->compare_backends ? YVEX_BACKEND_KIND_CUDA : request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK)
        rc = runtime_attention_progress(
            request, YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN, 0ull, 1ull, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_cleanup_lease_session_open(
            *cleanup, &session_request, session, failure, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_mode_configure(request, *session, result, selected_mode, err);
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN] +=
        runtime_seconds(yvex_core_monotonic_ns() - started);
    if (rc == YVEX_OK)
        rc = runtime_attention_progress(
            request, YVEX_RUNTIME_LIFECYCLE_BACKEND_OPEN, 1ull, 1ull, err);
    return rc;
}
static int runtime_attention_capacity_build(const yvex_graph_attention_operator_request *request,
    const yvex_runtime_model *model, const yvex_graph_family_api *graph,
    unsigned long long execution_count, yvex_graph_attention_capacity_plan **out, yvex_error *err) {
    const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
    yvex_graph_attention_capacity_request facts;
    unsigned long long exercise_capacity;
    memset(&facts, 0, sizeof(facts));
    facts.scope = request->scope;
    facts.history_tokens = request->history_tokens;
    facts.start_position = request->history_tokens;
    facts.token_count = request->token_count;
    facts.execution_count = execution_count;
    if (request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_EXERCISE) {
        if (!yvex_core_u64_add(request->history_tokens, request->token_count,
                               &exercise_capacity))
            return runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.state",
                                  "persistent state exercise capacity overflowed");
        facts.history_tokens = facts.start_position = 0ull;
        facts.token_count = exercise_capacity;
        facts.execution_count = 1ull;
        facts.use_requested_position = 1;
    }
    facts.layer_start = request->layer_start;
    facts.selection_key = request->selection_key;
    facts.select_layer = request->select_layer;
    facts.select_selection_key = request->select_selection_key;
    return yvex_graph_attention_capacity_plan_build(
        out, graph, view ? view->attention : NULL, &facts, err);
}
static const unsigned long long runtime_qualification_counters[] = {
    1ull, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull, 1ull, 1ull, 1ull, 1ull};
static const char runtime_quality_matrix_identity[] =
    "f420309313d355606543bcc50cd4cff44ada2be6c06723d328dc70b6db5e3673";
static const char qualify_where[] = "runtime.attention.qualification";
static int runtime_attention_qualify(yvex_graph_attention_operator_result *result, yvex_error *err) {
    static const char status[YVEX_RUNTIME_QUALITY_STATUS_COUNT][24] = {
        "pass", "pass", "pass", "available", "pass", "pass", "not_measured"};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const int structural = memcmp(&result->artifact_hash_passes, runtime_qualification_counters,
                                  sizeof(runtime_qualification_counters)) == 0 &&
        !result->warm_weight_artifact_reads && !result->warm_weight_upload_bytes &&
        !result->warm_host_allocations && !result->warm_device_allocations &&
        !result->warm_device_frees && result->capabilities.attention_benchmark_ready;
    if (!structural)
        return runtime_refuse(err, YVEX_ERR_STATE, qualify_where, "runtime structural invariant failed");
    memcpy(result->quality_status, status, sizeof(status));
    yvex_runtime_identity_copy(result->quality_matrix_identity, runtime_quality_matrix_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.attention.qualification.v1") ||
        !yvex_sha256_update_text(&hash, result->execution_identity) ||
        !yvex_sha256_update_text(&hash, result->quality_matrix_identity) ||
        !yvex_sha256_update_text(&hash, result->quality_status[YVEX_RUNTIME_QUALITY_QUALIFICATION]) ||
        !yvex_sha256_final(&hash, digest))
        return runtime_refuse(err, YVEX_ERR_STATE, qualify_where, "qualification identity construction failed");
    yvex_sha256_hex(digest, result->qualification_identity);
    return YVEX_OK;
}
int yvex_graph_attention_operator_execute(const yvex_graph_attention_operator_request *request,
    yvex_graph_attention_operator_result *result, yvex_runtime_cleanup_lease **retained_cleanup, yvex_error *err) {
    const yvex_runtime_family_adapter *adapter;
    const yvex_graph_family_api *graph;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_graph_attention_capacity_plan *capacity = NULL;
    yvex_runtime_model_failure model_failure;
    yvex_attention_probe_request probe_request;
    runtime_attention_phase_context *phase_context = NULL;
    runtime_attention_trace trace;
    runtime_attention_warm_snapshot warm_before, warm_after;
    yvex_core_execution_observation observation_before, observation_after, observation_delta;
    double *samples = NULL, *device_samples = NULL;
    unsigned long long warmup, repeat, preparation_dispatches, automatic_preparation = 0ull;
    unsigned long long measurement_start = 0ull, phase_started, dispatch_count;
    unsigned long long selected_layer = YVEX_ATTENTION_NO_LAYER, total_started = yvex_core_monotonic_ns();
    yvex_runtime_execution_mode selected_mode = YVEX_RUNTIME_MODE_EAGER;
    int rc;
    if (request && request->activation_input_path)
        return yvex_runtime_activation_prefill_operator_execute(request, result, retained_cleanup, err);
    if (!result || !retained_cleanup || *retained_cleanup)
        return runtime_refuse(err, YVEX_ERR_INVALID_ARG, "runtime.attention", "result and cleanup are required");
    runtime_attention_result_initialize(request, result);
    rc = runtime_attention_request_validate(request, err);
    if (rc != YVEX_OK) {
        runtime_attention_result_refuse(result, rc, err);
        return rc;
    }
    yvex_core_execution_observation_snapshot(&observation_before);
    adapter = yvex_runtime_family_adapter_find(request->target);
    graph = adapter ? adapter->graph() : NULL;
    repeat = request->repeat ? request->repeat : 1ull;
    warmup = request->warmup;
    if (!warmup && (request->operator_action == YVEX_RUNTIME_OPERATOR_REPLAY ||
                    (request->operator_action >= YVEX_RUNTIME_OPERATOR_GRAPH_WARMUP &&
                     request->operator_action <= YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE) ||
                    request->operator_action == YVEX_RUNTIME_OPERATOR_QUALIFY))
        warmup = request->operator_action == YVEX_RUNTIME_OPERATOR_GRAPH_UPDATE ? 2ull : 1ull;
    if ((request->operator_action == YVEX_RUNTIME_OPERATOR_BENCHMARK ||
         request->operator_action == YVEX_RUNTIME_OPERATOR_PROFILE) &&
        repeat <= (unsigned long long)(SIZE_MAX / (2u * sizeof(*samples)))) {
        samples = (double *)calloc((size_t)(repeat * 2ull), sizeof(*samples));
        if (!samples) {
            rc = runtime_refuse(err, YVEX_ERR_NOMEM, "runtime.attention", "sample allocation failed");
            runtime_attention_result_refuse(result, rc, err);
            return rc;
        }
        device_samples = samples + repeat;
    }
    memset(&model_failure, 0, sizeof(model_failure));
    rc = runtime_attention_open(request, adapter, &cleanup, &model, &session, &selected_mode,
                                &model_failure, result, err);
    phase_started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK && samples && !warmup && repeat &&
        (request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA) &&
        strcmp(result->selected_mode, "eager") != 0)
        automatic_preparation = 1ull;
    if (rc == YVEX_OK && (!yvex_core_u64_add(warmup, automatic_preparation, &measurement_start) ||
         !yvex_core_u64_add(measurement_start, repeat, &dispatch_count)))
        rc = runtime_refuse(err, YVEX_ERR_BOUNDS, "runtime.attention.state", "extent overflowed");
    if (rc == YVEX_OK) rc = runtime_attention_capacity_build(request, model, graph, dispatch_count, &capacity, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_trace_begin(&trace, request, result->selected_mode, err);
    if (rc == YVEX_OK &&
        (runtime_attention_action_dispatches(request->operator_action) ||
         request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_INSPECT ||
         request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_VALIDATE)) {
        yvex_attention_failure state_failure;
        rc = yvex_runtime_session_prepare_attention_probe_state(session, model, capacity, &state_failure, err);
    }
    if (rc == YVEX_OK && (request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA))
        rc = yvex_runtime_session_prepare_attention_workspace(
            session, selected_mode, request->operation_scope,
            runtime_attention_evidence_levels[request->trace_policy], capacity, 0ull,
            &model_failure, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_execution_descriptor_identity(
            request, model, session, capacity, result, result->execution_descriptor_identity, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_mode_bind_descriptor(request, session, result, selected_mode, capacity, err);
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_WORKSPACE_PREPARE] +=
        runtime_seconds(yvex_core_monotonic_ns() - phase_started);
    memset(&probe_request, 0, sizeof(probe_request));
    probe_request.backend = request->backend;
    probe_request.probe = request->probe;
    probe_request.scope = request->scope;
    probe_request.operation_scope = request->operation_scope == YVEX_RUNTIME_SCOPE_ATTENTION_CORE
                                        ? YVEX_ATTENTION_OPERATION_CORE : YVEX_ATTENTION_OPERATION_ENVELOPE;
    probe_request.token_count = request->token_count;
    probe_request.evidence_level = runtime_attention_evidence_levels[request->trace_policy];
    if (rc == YVEX_OK && request->select_layer)
        selected_layer = request->layer_start;
    else if (rc == YVEX_OK && request->select_selection_key)
        selected_layer = yvex_graph_attention_capacity_plan_summary(capacity)->first_layer;
    probe_request.select_layer = selected_layer != YVEX_ATTENTION_NO_LAYER;
    probe_request.layer_ordinal = selected_layer;
    probe_request.compare_backends = request->compare_backends;
    probe_request.cancel_requested = request->cancel_requested;
    probe_request.cancel_context = request->cancel_context;
    if (runtime_attention_action_dispatches(request->operator_action) &&
        request->operator_action != YVEX_RUNTIME_OPERATOR_STATE_EXERCISE)
        probe_request.select_position = request->scope != YVEX_ATTENTION_PROBE_SCOPE_QUICK;
    if (request->trace_policy != YVEX_RUNTIME_TRACE_NONE) {
        probe_request.evidence = runtime_attention_trace_capture;
        probe_request.evidence_context = &trace;
    }
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_STATE_EXERCISE) {
        const yvex_runtime_model_view *view = yvex_runtime_model_view_get(model);
        yvex_attention_failure phase_failure;
        probe_request.logical_model_identity = view && view->binding ? view->binding->logical_model_identity : NULL;
        probe_request.backend_context = yvex_runtime_session_view_get(session)->backend;
        probe_request.workspace = yvex_runtime_session_view_get(session)->attention_workspace;
        phase_context = calloc(1u, sizeof(*phase_context));
        if (!phase_context)
            rc = runtime_refuse(err, YVEX_ERR_NOMEM, "runtime.attention.phase",
                                "phase cleanup owner allocation failed");
        if (rc == YVEX_OK) {
            rc = yvex_runtime_cleanup_lease_adopt(
                cleanup, phase_context, runtime_attention_phase_context_release, err);
            if (rc != YVEX_OK) { free(phase_context); phase_context = NULL; }
        }
        if (rc == YVEX_OK)
            rc = runtime_attention_phase_context_open(
                phase_context, graph, view ? view->attention : NULL, capacity,
                probe_request.operation_scope, request->history_tokens,
                probe_request.token_count, request->maximum_host_bytes, &phase_failure, err);
    }
    preparation_dispatches = runtime_attention_action_dispatches(request->operator_action) ? measurement_start : 0ull;
    if (!samples && preparation_dispatches == 0ull && repeat > 0ull && (request->compare_backends ||
        request->backend == YVEX_BACKEND_KIND_CUDA) && strcmp(result->selected_mode, "eager") != 0)
        preparation_dispatches = 1ull;
    phase_started = yvex_core_monotonic_ns();
    if (rc == YVEX_OK && preparation_dispatches)
        rc = runtime_attention_operator_dispatch(
            request, session, model, graph, &probe_request, phase_context, 0ull,
            preparation_dispatches, measurement_start, dispatch_count, samples, device_samples,
            &model_failure, result, err);
    if (preparation_dispatches && measurement_start)
        result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_WARMUP] +=
            runtime_seconds(yvex_core_monotonic_ns() - phase_started);
    if (rc == YVEX_OK)
        rc = runtime_attention_warm_snapshot_take(model, session, &warm_before, err);
    if (rc == YVEX_OK && getenv("YVEX_TEST_RUNTIME_WARM_HOST_ALLOCATION")) {
        void *injected = malloc(1u);
        if (!injected)
            rc = runtime_refuse(
                err, YVEX_ERR_NOMEM, "runtime.attention.warm", "warm allocation injection failed");
        free(injected);
    }
    if (rc == YVEX_OK && preparation_dispatches < dispatch_count)
        rc = runtime_attention_operator_dispatch(
            request, session, model, graph, &probe_request, phase_context, preparation_dispatches,
            dispatch_count - preparation_dispatches, measurement_start, dispatch_count, samples,
            device_samples, &model_failure, result, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_warm_snapshot_take(model, session, &warm_after, err);
    if (rc == YVEX_OK)
        rc = runtime_attention_warm_snapshot_publish(&warm_before, &warm_after, result, err);
    if (rc == YVEX_OK) {
        yvex_core_execution_observation_snapshot(&observation_after);
        if (!yvex_core_execution_observation_delta(&observation_before, &observation_after, &observation_delta))
            rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.observation",
                                "execution planning observation counters regressed or overflowed");
    }
    if (rc == YVEX_OK) {
        result->runtime_source_headers_read = observation_delta.source_headers_read;
        result->runtime_source_payload_bytes_read = observation_delta.source_payload_bytes_read;
        result->runtime_transform_plans_built = observation_delta.transform_plans_built;
        result->runtime_quant_plans_built = observation_delta.quant_plans_built;
        result->runtime_writer_plans_built = observation_delta.writer_plans_built;
        if (result->runtime_source_headers_read || result->runtime_source_payload_bytes_read ||
            result->runtime_transform_plans_built || result->runtime_quant_plans_built ||
            result->runtime_writer_plans_built)
            rc = runtime_refuse(err, YVEX_ERR_STATE, "runtime.attention.observation",
                                "runtime execution entered an upstream source or planning owner");
    }
    rc = runtime_attention_publish_result(request, session, &trace, warmup, repeat, result, rc, err);
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_CAPTURE] =
        runtime_seconds(result->cuda_graph_capture_elapsed_ns);
    result->lifecycle_seconds[YVEX_RUNTIME_LIFECYCLE_GRAPH_INSTANTIATE] =
        runtime_seconds(result->cuda_graph_instantiate_elapsed_ns);
    if (rc == YVEX_OK)
        rc = runtime_attention_graph_lifecycle_partition(result, err);
    yvex_graph_attention_capacity_plan_close(&capacity);
    rc = runtime_attention_cleanup(&cleanup, samples, result, rc, err);
    if (cleanup) *retained_cleanup = cleanup;
    if (rc == YVEX_OK && request->operator_action == YVEX_RUNTIME_OPERATOR_QUALIFY)
        rc = runtime_attention_qualify(result, err);
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        result->reason[0] = '\0';
        yvex_error_clear(err);
    } else
        runtime_attention_result_refuse(result, rc, err);
    result->total_seconds = (double)(yvex_core_monotonic_ns() - total_started) / 1.0e9;
    return rc;
}
