/*
 * Provide reusable state and selection mechanisms to graph-family recipes.
 *
 * History is immutable and causal selection is deterministic. Protocol truth does not promote
 * complete attention, persistent KV, or generation.
 */
#include "src/graph/private.h"
typedef struct {
    unsigned long long kv_stride, score_stride, kv_extent, score_extent;
    float *kv, *score;
} attention_rolling_output_storage;
_Static_assert(sizeof(attention_rolling_output_storage) ==
                   offsetof(yvex_attention_rolling_state_output, overlap) -
                       offsetof(yvex_attention_rolling_state_output, kv_state_stride),
               "rolling output storage must remain one contiguous representation");
_Static_assert(sizeof(yvex_attention_rolling_state_view) ==
                   sizeof(yvex_attention_rolling_state_output),
               "rolling view/output layouts must remain representation-compatible");
_Static_assert(offsetof(yvex_attention_rolling_state_view, attention_plan_identity) ==
                   offsetof(yvex_attention_rolling_state_output, attention_plan_identity),
               "rolling view/output identity offsets must remain compatible");
_Static_assert((int)YVEX_BACKEND_ATTENTION_PHASE_PREFILL == (int)YVEX_EXECUTION_PHASE_PREFILL,
               "backend prefill phase must match execution ABI");
_Static_assert((int)YVEX_BACKEND_ATTENTION_PHASE_DECODE == (int)YVEX_EXECUTION_PHASE_DECODE,
               "backend decode phase must match execution ABI");
_Static_assert((int)YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_DRAFT ==
                   (int)YVEX_EXECUTION_PHASE_DRAFT,
               "backend draft phase must match execution ABI");
_Static_assert((int)YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_VERIFY ==
                   (int)YVEX_EXECUTION_PHASE_VERIFY,
               "backend verify phase must match execution ABI");
_Static_assert((int)YVEX_BACKEND_ATTENTION_PHASE_MIXED == (int)YVEX_EXECUTION_PHASE_MIXED,
               "backend mixed phase must match execution ABI");

static int attention_refuse(yvex_attention_failure *failure, yvex_attention_failure_code code,
                            unsigned long long layer, unsigned long long expected,
                            unsigned long long actual, yvex_error *error, int status,
                            const char *reason) {
    return yvex_attention_reject(failure, code, NULL, layer, YVEX_TENSOR_ROLE_UNKNOWN, expected,
                                 actual, error, status, reason);
}

static int attention_history_refuse(const yvex_attention_layer_plan *layer,
                                    unsigned long long expected, unsigned long long actual,
                                    yvex_attention_failure *failure, yvex_error *err,
                                    yvex_status status, const char *reason) {
    return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY,
                            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, expected, actual,
                            err, status, reason);
}
/*
 * Sample one borrowed cooperative-cancellation predicate at a named safe point.
 *
 * Optional borrowed predicate, layer identity, stable safe-point text, and diagnostics.
 */
int yvex_attention_cancel_check(const yvex_attention_cancellation *cancellation,
                                unsigned long long layer_index, const char *safe_point,
                                yvex_attention_failure *failure, yvex_error *err) {
    if (!cancellation)
        return YVEX_OK;
    if (!cancellation->requested)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                layer_index, 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
                                "attention cancellation requires a borrowed predicate");
    if (cancellation->requested(cancellation->context))
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_CANCELLED, layer_index,
                                0ull, 1ull, err, YVEX_ERR_CANCELLED,
                                safe_point ? safe_point : "attention execution cancelled");
    return YVEX_OK;
}
/*
 * Validate one family-selected attention class before payload access or state mutation.
 *
 * Reads geometry only and clears no admitted identity or state.
 */
int yvex_attention_class_geometry_validate(const yvex_attention_layer_plan *layer,
                                           unsigned long long csa_ratio,
                                           unsigned long long hca_ratio,
                                           yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long query_width, indexer_width = 0ull, output_group_width;
    unsigned long long output_width, output_low_width;
    int valid = layer && layer->sliding_window && layer->query_heads && layer->kv_heads &&
                layer->head_dimension && layer->hidden_dimension &&
                layer->rope_head_dimension <= layer->head_dimension &&
                layer->query_heads % layer->kv_heads == 0ull && layer->output_groups &&
                layer->query_heads % layer->output_groups == 0ull && layer->output_lora_rank &&
                yvex_core_u64_mul(layer->query_heads, layer->head_dimension, &query_width) &&
                yvex_core_u64_mul(layer->query_heads / layer->output_groups, layer->head_dimension,
                                  &output_group_width) &&
                yvex_core_u64_mul(layer->output_groups, output_group_width, &output_width) &&
                output_width == query_width &&
                yvex_core_u64_mul(layer->output_groups, layer->output_lora_rank, &output_low_width);
    if (!valid)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                                err, YVEX_ERR_FORMAT,
                                "attention common head, position, or output geometry is invalid");
    switch (layer->attention_class) {
    case YVEX_ATTENTION_CLASS_SWA:
        valid = layer->compression_ratio == 0ull && !layer->compressor_required &&
                !layer->indexer_required && !layer->indexer_heads &&
                !layer->indexer_head_dimension && !layer->indexer_topk &&
                !layer->sparse_topk.required && !layer->sparse_topk.k;
        break;
    case YVEX_ATTENTION_CLASS_CSA:
        valid = csa_ratio && layer->compression_ratio == csa_ratio && layer->compressor_required &&
                layer->indexer_required && layer->indexer_heads && layer->indexer_head_dimension &&
                yvex_core_u64_mul(layer->indexer_heads, layer->indexer_head_dimension,
                                  &indexer_width) &&
                indexer_width && layer->indexer_topk && layer->sparse_topk.required &&
                layer->sparse_topk.k == layer->indexer_topk;
        break;
    case YVEX_ATTENTION_CLASS_HCA:
        valid = hca_ratio && layer->compression_ratio == hca_ratio && layer->compressor_required &&
                !layer->indexer_required && !layer->indexer_heads &&
                !layer->indexer_head_dimension && !layer->indexer_topk &&
                !layer->sparse_topk.required && !layer->sparse_topk.k;
        break;
    default:
        valid = 0;
        break;
    }
    if (!valid)
        return attention_refuse(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, layer->layer_index,
            layer->attention_class == YVEX_ATTENTION_CLASS_HCA ? hca_ratio : csa_ratio,
            layer->compression_ratio, err, YVEX_ERR_FORMAT,
            "attention class geometry does not match the family contract");
    return yvex_attention_accept(failure, err);
}

void yvex_attention_envelope_workspace_release(yvex_attention_envelope_workspace *workspace)
{
    if (!workspace) return;
    if (!workspace->workspace) {
        free(workspace->residual);
        free(workspace->linear_mixes);
        free(workspace->scale);
        free(workspace->base);
        free(workspace->post);
        free(workspace->combination);
        free(workspace->norm_weights);
    }
    memset(workspace, 0, sizeof(*workspace));
}

int yvex_attention_envelope_scratch_elements(const yvex_attention_layer_plan *layer,
                                             unsigned long long token_count,
                                             unsigned long long *elements)
{
    unsigned long long per_token, stream_square, total;
    if (!layer || !token_count || !elements || !layer->residual_stream_count ||
        !layer->mhc_mixing_rows ||
        !yvex_core_u64_mul(layer->residual_stream_count, layer->residual_stream_count,
                           &stream_square) ||
        !yvex_core_u64_add(layer->mhc_mixing_rows, layer->residual_stream_count,
                           &per_token) ||
        !yvex_core_u64_add(per_token, stream_square, &per_token) ||
        !yvex_core_u64_add(per_token, layer->residual_expanded_width, &per_token) ||
        !yvex_core_u64_mul(per_token, token_count, &total) ||
        !yvex_core_u64_add(total, layer->mhc_base_width, &total) ||
        !yvex_core_u64_add(total, layer->mhc_scale_width, &total) ||
        !yvex_core_u64_add(total, layer->attention_input_norm_width, &total))
        return 0;
    *elements = total;
    return 1;
}

static int attention_envelope_reject(const yvex_attention_layer_plan *layer,
                                     yvex_attention_failure_code code,
                                     const yvex_runtime_tensor_binding *binding,
                                     yvex_tensor_role role, unsigned long long expected,
                                     unsigned long long actual, yvex_attention_failure *failure,
                                     yvex_error *err, yvex_status status, const char *reason)
{
    return yvex_attention_reject(
        failure, code, binding, layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, role,
        expected, actual, err, status, reason);
}

static int attention_envelope_bindings_validate(
    const yvex_attention_layer_plan *layer, const yvex_runtime_tensor_binding *function,
    const yvex_runtime_tensor_binding *base, const yvex_runtime_tensor_binding *scale,
    const yvex_runtime_tensor_binding *norm, yvex_attention_failure *failure, yvex_error *err)
{
    unsigned long long base_elements, scale_elements, norm_elements;
    if (!layer || !function || !base || !scale || !norm || !function->binding || !base->binding ||
        !scale->binding || !norm->binding)
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, 4ull, 0ull, failure, err,
            YVEX_ERR_FORMAT, "attention envelope requires mHC and input-norm bindings");
    if (function->qtype != YVEX_GGUF_QTYPE_F32 || base->qtype != YVEX_GGUF_QTYPE_F32 ||
        scale->qtype != YVEX_GGUF_QTYPE_F32)
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_QTYPE, function,
            layer->mhc_function_role, YVEX_GGUF_QTYPE_F32,
            function->qtype, failure, err, YVEX_ERR_UNSUPPORTED,
            "attention envelope mHC bindings require exact F32 storage");
    if (!yvex_core_u64_mul(base->binding->row_count, base->binding->row_width,
                           &base_elements) ||
        !yvex_core_u64_mul(scale->binding->row_count, scale->binding->row_width,
                           &scale_elements) ||
        !yvex_core_u64_mul(norm->binding->row_count, norm->binding->row_width,
                           &norm_elements) ||
        function->binding->row_count != layer->mhc_mixing_rows ||
        function->binding->row_width != layer->mhc_mixing_columns ||
        base_elements != layer->mhc_base_width ||
        scale_elements != layer->mhc_scale_width ||
        norm_elements != layer->attention_input_norm_width)
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, function,
            layer->mhc_function_role, layer->mhc_mixing_columns,
            function->binding->row_width, failure, err, YVEX_ERR_FORMAT,
            "attention envelope binding shapes do not match the immutable plan");
    return YVEX_OK;
}

static int attention_envelope_bind(
    const yvex_runtime_descriptor *descriptor, const yvex_attention_layer_plan *layer,
    const yvex_runtime_tensor_binding **function, const yvex_runtime_tensor_binding **base,
    const yvex_runtime_tensor_binding **scale, const yvex_runtime_tensor_binding **norm,
    yvex_attention_failure *failure, yvex_error *err)
{
    *function = yvex_attention_binding_find(descriptor, layer->mhc_function_role, layer);
    *base = yvex_attention_binding_find(descriptor, layer->mhc_base_role, layer);
    *scale = yvex_attention_binding_find(descriptor, layer->mhc_scale_role, layer);
    *norm = yvex_attention_binding_find(descriptor, layer->attention_input_norm_role, layer);
    return attention_envelope_bindings_validate(
        layer, *function, *base, *scale, *norm, failure, err);
}

int yvex_attention_envelope_prepare(
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_layer_plan *layer, const float *expanded_input,
    unsigned long long token_count, unsigned long long input_stride, float *core_input,
    unsigned long long core_stride, yvex_attention_envelope_workspace *workspace,
    yvex_attention_scratch_budget *scratch, yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_runtime_tensor_binding *function, *base, *scale, *norm;
    yvex_attention_mhc_pre_args pre;
    unsigned long long rows = 0ull, token;
    int rc = YVEX_OK;
    if (!session || !descriptor || !layer || !expanded_input || !token_count || !core_input ||
        !workspace || workspace->linear_mixes || input_stride < layer->residual_expanded_width ||
        core_stride < layer->hidden_dimension)
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, failure, err,
            YVEX_ERR_INVALID_ARG, "attention envelope preparation arguments are invalid");
    rc = attention_envelope_bind(descriptor, layer, &function, &base, &scale, &norm,
                                 failure, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(token_count, layer->residual_expanded_width,
                           &workspace->residual_elements) ||
        !yvex_core_u64_mul(token_count, layer->mhc_mixing_rows, &workspace->mix_elements) ||
        !yvex_core_u64_mul(token_count, layer->residual_stream_count,
                           &workspace->post_elements) ||
        !yvex_core_u64_mul(workspace->post_elements, layer->residual_stream_count,
                           &workspace->combination_elements))
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, function,
            layer->mhc_function_role, ULLONG_MAX, token_count,
            failure, err, YVEX_ERR_BOUNDS, "attention envelope workspace geometry overflowed");
    workspace->residual_stride = layer->residual_expanded_width;
    workspace->workspace = scratch ? scratch->workspace : NULL;
    workspace->residual = yvex_attention_scratch_calloc(
        scratch, workspace->residual_elements, sizeof(float));
    workspace->linear_mixes = yvex_attention_scratch_calloc(
        scratch, workspace->mix_elements, sizeof(float));
    workspace->scale = yvex_attention_scratch_calloc(
        scratch, layer->mhc_scale_width, sizeof(float));
    workspace->base = yvex_attention_scratch_calloc(
        scratch, layer->mhc_base_width, sizeof(float));
    workspace->post = yvex_attention_scratch_calloc(
        scratch, workspace->post_elements, sizeof(float));
    workspace->combination = yvex_attention_scratch_calloc(
        scratch, workspace->combination_elements, sizeof(float));
    workspace->norm_weights = yvex_attention_scratch_calloc(
        scratch, layer->attention_input_norm_width, sizeof(float));
    if (!workspace->residual || !workspace->linear_mixes || !workspace->scale ||
        !workspace->base || !workspace->post || !workspace->combination ||
        !workspace->norm_weights) {
        unsigned long long expected = workspace->mix_elements;
        yvex_attention_envelope_workspace_release(workspace);
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, function,
            layer->mhc_function_role, expected, 0ull,
            failure, err, YVEX_ERR_NOMEM, "attention envelope workspace allocation failed");
    }
    for (token = 0ull; token < token_count; ++token) {
        const float *source = expanded_input + token * input_stride;
        float *residual = workspace->residual + token * workspace->residual_stride;
        unsigned long long lane;
        memcpy(residual, source, (size_t)workspace->residual_stride * sizeof(*residual));
        for (lane = 0ull; lane < workspace->residual_stride; ++lane)
            if (!isfinite(residual[lane])) {
                rc = attention_envelope_reject(
                    layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                    YVEX_TENSOR_ROLE_UNKNOWN, 1ull, token, failure, err,
                    YVEX_ERR_FORMAT, "attention envelope residual input is non-finite");
                goto done;
            }
        if (!yvex_attention_compute_round(layer->compute_contract, residual,
                                          workspace->residual_stride)) {
            rc = attention_envelope_reject(
                layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_TENSOR_ROLE_UNKNOWN, workspace->residual_stride,
                token, failure, err, YVEX_ERR_FORMAT,
                "attention envelope residual input cannot reach its BF16 boundary");
            goto done;
        }
    }
    rc = yvex_attention_dot_batch(
        session, function, 0ull, workspace->residual, token_count,
        workspace->residual_stride,
        layer->residual_expanded_width, layer->mhc_mixing_rows, workspace->linear_mixes,
        layer->mhc_mixing_rows, &rows, scratch, result, failure, err);
    if (rc == YVEX_OK && rows != layer->mhc_mixing_rows)
        rc = attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, function,
            layer->mhc_function_role, layer->mhc_mixing_rows, rows,
            failure, err, YVEX_ERR_FORMAT, "attention envelope mHC projection is incomplete");
    if (rc == YVEX_OK)
        rc = yvex_attention_decode_flat(session, scale, workspace->scale,
                                        layer->mhc_scale_width, scratch, result, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_attention_decode_flat(session, base, workspace->base,
                                        layer->mhc_base_width, scratch, result, failure, err);
    if (rc == YVEX_OK)
        rc = yvex_attention_decode_flat(session, norm, workspace->norm_weights,
                                        layer->attention_input_norm_width, scratch, result,
                                        failure, err);
    pre = (yvex_attention_mhc_pre_args){
        layer, workspace->residual, workspace->linear_mixes, workspace->scale, workspace->base,
        token_count, workspace->residual_stride, layer->mhc_mixing_rows, core_input, workspace->post,
        workspace->combination, core_stride, layer->residual_stream_count,
        layer->residual_stream_count * layer->residual_stream_count};
    if (rc == YVEX_OK) rc = yvex_attention_mhc_pre(&pre, failure, err);
    for (token = 0ull; rc == YVEX_OK && token < token_count; ++token) {
        float *row = core_input + token * core_stride;
        if (!yvex_attention_rms_norm(row, layer->attention_input_norm_width,
                                     workspace->norm_weights, layer->rms_norm_epsilon) ||
            !yvex_attention_compute_round(layer->compute_contract, row,
                                          layer->attention_input_norm_width))
            rc = attention_envelope_reject(
                layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, norm,
                layer->attention_input_norm_role,
                layer->attention_input_norm_width, token, failure, err, YVEX_ERR_FORMAT,
                "attention envelope input normalization failed");
    }
done:
    if (rc != YVEX_OK) yvex_attention_envelope_workspace_release(workspace);
    return rc;
}

int yvex_attention_envelope_finish(
    const yvex_attention_layer_plan *layer, const float *core_output,
    unsigned long long core_stride, unsigned long long token_count,
    const yvex_attention_envelope_workspace *workspace, float *envelope_output,
    unsigned long long envelope_stride, yvex_attention_failure *failure, yvex_error *err)
{
    yvex_attention_mhc_post_args post;
    if (!workspace || !workspace->residual || !workspace->post || !workspace->combination)
        return attention_envelope_reject(
            layer, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, NULL,
            YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, 1ull, 0ull,
            failure, err, YVEX_ERR_STATE, "attention envelope coefficients are absent");
    post = (yvex_attention_mhc_post_args){
        layer, core_output, workspace->residual, workspace->post, workspace->combination,
        token_count, core_stride, workspace->residual_stride, layer->residual_stream_count,
        layer->residual_stream_count * layer->residual_stream_count,
        envelope_output, envelope_stride};
    return yvex_attention_mhc_post(&post, failure, err);
}

int yvex_attention_csa_select(
    const yvex_attention_layer_plan *layer, const yvex_attention_history_view *history,
    const float *current_indexer, unsigned long long current_indexer_count,
    unsigned long long current_indexer_stride, const unsigned long long *current_indexer_positions,
    const float *index_query, const float *index_weights, unsigned long long query_position,
    unsigned long long *selected, unsigned long long *selected_count,
    unsigned long long *valid_count, yvex_attention_scratch_budget *scratch,
    yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long total;
    float *scores = NULL;
    unsigned long long *ordinals = NULL, *valid_indexes = NULL, *ranked = NULL;
    unsigned long long candidate, valid = 0ull, ranked_count = 0ull;
    size_t base_reserved = 0u, ranked_reserved = 0u;
    int rc = YVEX_OK;
    if (selected_count)
        *selected_count = 0ull;
    if (valid_count)
        *valid_count = 0ull;
    if (!layer || !history || !index_query || !index_weights || !selected || !selected_count ||
        !valid_count)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                                err, YVEX_ERR_INVALID_ARG,
                                "CSA selection requires history, query, weights, and outputs");
    if (history->indexer_entry_count > ULLONG_MAX - current_indexer_count)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                layer->layer_index, ULLONG_MAX, history->indexer_entry_count, err,
                                YVEX_ERR_BOUNDS, "CSA candidate count overflowed");
    total = history->indexer_entry_count + current_indexer_count;
    if (!total)
        return YVEX_OK;
    if (!yvex_attention_scratch_reserve(
            scratch, total, sizeof(*scores) + sizeof(*ordinals) + sizeof(*valid_indexes),
            &base_reserved))
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
                                layer->layer_index, scratch ? scratch->limit_bytes : 0ull,
                                scratch ? (unsigned long long)scratch->live_bytes : 0ull, err,
                                YVEX_ERR_BOUNDS,
                                "CSA selection exceeds the attention scratch budget");
    scores = (float *)yvex_attention_scratch_calloc(scratch, total, sizeof(*scores));
    ordinals = (unsigned long long *)yvex_attention_scratch_calloc(
        scratch, total, sizeof(*ordinals));
    valid_indexes = (unsigned long long *)yvex_attention_scratch_calloc(
        scratch, total, sizeof(*valid_indexes));
    if (!scores || !ordinals || !valid_indexes) {
        rc = attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION,
                              layer->layer_index, total, 0ull, err, YVEX_ERR_NOMEM,
                              "CSA selection scratch allocation failed");
        goto cleanup;
    }
    for (candidate = 0ull; candidate < total; ++candidate) {
        const float *key = attention_segment_row(
            history->indexer_kv, history->indexer_entry_count, history->indexer_kv_stride,
            current_indexer, current_indexer_count, current_indexer_stride, candidate);
        unsigned long long position = attention_segment_position(
            history->indexer_positions, history->indexer_entry_count, current_indexer_positions,
            current_indexer_count, candidate);
        unsigned long long head;
        double score = 0.0;
        if (!key || position == ULLONG_MAX || position > query_position ||
            position > ULLONG_MAX - layer->compression_ratio + 1ull ||
            position + layer->compression_ratio - 1ull > query_position)
            continue;
        for (head = 0ull; head < layer->indexer_heads; ++head) {
            const float *q = index_query + head * layer->indexer_head_dimension;
            unsigned long long lane;
            double dot = 0.0;
            for (lane = 0ull; lane < layer->indexer_head_dimension; ++lane) {
                if (!isfinite(q[lane]) || !isfinite(key[lane]))
                    goto numeric;
                dot += (double)q[lane] * (double)key[lane];
            }
            if (dot < 0.0)
                dot = 0.0;
            if (!isfinite(index_weights[head]))
                goto numeric;
            score += dot * (double)index_weights[head];
        }
        score *= 1.0 / sqrt((double)layer->indexer_head_dimension);
        score *= 1.0 / sqrt((double)layer->indexer_heads);
        if (!isfinite(score))
            goto numeric;
        scores[valid] = (float)score;
        ordinals[valid] = position;
        valid_indexes[valid] = candidate;
        valid++;
    }
    if (valid) {
        unsigned long long ranked_capacity = attention_min_u64(valid, layer->sparse_topk.k);
        unsigned long long index;
        if (!yvex_attention_scratch_reserve(scratch, ranked_capacity, sizeof(*ranked),
                                            &ranked_reserved)) {
            rc = attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
                                  layer->layer_index, scratch ? scratch->limit_bytes : 0ull,
                                  scratch ? (unsigned long long)scratch->live_bytes : 0ull, err,
                                  YVEX_ERR_BOUNDS,
                                  "CSA ranked selection exceeds the attention scratch budget");
            goto cleanup;
        }
        ranked = (unsigned long long *)yvex_attention_scratch_calloc(
            scratch, ranked_capacity, sizeof(*ranked));
        if (!ranked) {
            rc = attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION,
                                  layer->layer_index, valid, 0ull, err, YVEX_ERR_NOMEM,
                                  "CSA ranked-selection scratch allocation failed");
            goto cleanup;
        }
        rc = yvex_attention_topk_select(scores, ordinals, valid, layer->sparse_topk.k, ranked,
                                        &ranked_count, scratch, failure, err);
        if (rc != YVEX_OK)
            goto cleanup;
        for (index = 0ull; index < ranked_count; ++index)
            selected[index] = valid_indexes[ranked[index]];
        *selected_count = ranked_count;
    }
    *valid_count = valid;
cleanup:
    yvex_attention_scratch_free(scratch, ranked);
    yvex_attention_scratch_free(scratch, scores);
    yvex_attention_scratch_free(scratch, ordinals);
    yvex_attention_scratch_free(scratch, valid_indexes);
    attention_scratch_release(scratch, ranked_reserved);
    attention_scratch_release(scratch, base_reserved);
    return rc;
numeric:
    rc = attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                          layer->layer_index, 1ull, candidate, err, YVEX_ERR_FORMAT,
                          "CSA index scoring produced non-finite values");
    goto cleanup;
}

int yvex_attention_rolling_geometry(const yvex_attention_layer_plan *layer,
                                    yvex_attention_rolling_kind kind, unsigned long long *ratio,
                                    unsigned long long *head_dim, unsigned long long *state_width,
                                    unsigned long long *state_slots, int *overlap, int *rotated) {
    unsigned long long coeff;
    if (!layer || !ratio || !head_dim || !state_width || !state_slots || !overlap || !rotated ||
        layer->compression_ratio == 0ull)
        return 0;
    if (kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN) {
        *ratio = layer->compression_ratio;
        *head_dim = layer->head_dimension;
        *overlap = layer->attention_class == YVEX_ATTENTION_CLASS_CSA ? 1 : 0;
        *rotated = 0;
    } else if (kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER) {
        if (layer->attention_class != YVEX_ATTENTION_CLASS_CSA)
            return 0;
        *ratio = layer->compression_ratio;
        *head_dim = layer->indexer_head_dimension;
        *overlap = 1;
        *rotated = 1;
    } else {
        return 0;
    }
    coeff = *overlap ? 2ull : 1ull;
    if (!yvex_core_u64_mul(*head_dim, coeff, state_width) ||
        !yvex_core_u64_mul(*ratio, coeff, state_slots))
        return 0;
    return *ratio != 0ull && *head_dim != 0ull;
}

static int attention_rolling_active_values_are_finite(
    const float *kv_state, const float *score_state, unsigned long long kv_stride,
    unsigned long long score_stride, unsigned long long head_dim, unsigned long long ratio,
    unsigned long long previous_fill, unsigned long long current_fill, int overlap) {
    unsigned long long slot;
    unsigned long long lane;
    if (!kv_state || !score_state)
        return 0;
    /* The previous half participates in the first overlap emission even
     * before it contains a completed group.  Its fail-closed score sentinel
     * is therefore part of the admitted state, not unused storage. */
    if (overlap && previous_fill == 0ull) {
        for (slot = 0ull; slot < ratio; ++slot) {
            for (lane = 0ull; lane < head_dim; ++lane) {
                float score = score_state[slot * score_stride + lane];
                if (!isinf(score) || !signbit(score))
                    return 0;
            }
        }
    }
    for (slot = 0ull; overlap && slot < previous_fill; ++slot) {
        for (lane = 0ull; lane < head_dim; ++lane) {
            if (!isfinite(kv_state[slot * kv_stride + lane]) ||
                !isfinite(score_state[slot * score_stride + lane]))
                return 0;
        }
    }
    for (slot = 0ull; slot < current_fill; ++slot) {
        unsigned long long base = overlap ? ratio + slot : slot;
        unsigned long long lane_offset = overlap ? head_dim : 0ull;
        for (lane = 0ull; lane < head_dim; ++lane) {
            if (!isfinite(kv_state[base * kv_stride + lane + lane_offset]) ||
                !isfinite(score_state[base * score_stride + lane + lane_offset]))
                return 0;
        }
    }
    return 1;
}

static int attention_rolling_state_validate(const yvex_attention_layer_plan *layer,
                                            const yvex_attention_rolling_state_view *state,
                                            yvex_attention_rolling_kind kind,
                                            yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long ratio, head_dim, state_width, state_slots, required_extent;
    unsigned long long expected_cursor, expected_previous_fill;
    int overlap, rotated;
    if (!layer || !state || !state->present)
        return attention_history_refuse(layer, 1ull, 0ull, failure, err, YVEX_ERR_INVALID_ARG,
                                        "DeepSeek attention rolling state is missing");
    if (!yvex_attention_rolling_geometry(layer, kind, &ratio, &head_dim, &state_width, &state_slots,
                                         &overlap, &rotated))
        return attention_history_refuse(
            layer, 1ull, 0ull, failure, err, YVEX_ERR_UNSUPPORTED,
            "DeepSeek attention rolling state is not used by this class");
    if (state->schema_version != YVEX_DEEPSEEK_ATTENTION_ROLLING_STATE_SCHEMA_V1 ||
        state->kind != kind || state->attention_class != layer->attention_class ||
        state->layer_index != layer->layer_index || state->ratio != ratio ||
        state->head_dimension != head_dim || state->state_width != state_width ||
        state->state_slots != state_slots || state->overlap != overlap || state->rotated != rotated)
        return attention_history_refuse(
            layer, state_width, state->state_width, failure, err, YVEX_ERR_FORMAT,
            "DeepSeek attention rolling state identity or geometry mismatch");
    if (state->cursor >= ratio || state->previous_fill > ratio || state->current_fill > ratio ||
        (!overlap && state->previous_fill))
        return attention_history_refuse(
            layer, ratio, state->cursor, failure, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention rolling state cursor or fill is invalid");
    expected_cursor = state->next_token_position % ratio;
    expected_previous_fill = overlap && state->next_token_position >= ratio ? ratio : 0ull;
    if (state->cursor != expected_cursor || state->current_fill != expected_cursor ||
        state->previous_fill != expected_previous_fill)
        return attention_history_refuse(
            layer, expected_cursor, state->cursor, failure, err, YVEX_ERR_STATE,
            "DeepSeek attention rolling state does not match its token position");
    if (state->kv_state_stride < state_width || state->score_state_stride < state_width ||
        !state->kv_state || !state->score_state)
        return attention_history_refuse(
            layer, state_width, state->kv_state_stride, failure, err, YVEX_ERR_FORMAT,
            "DeepSeek attention rolling state storage is incomplete");
    if (!yvex_core_u64_mul(state_slots, state->kv_state_stride, &required_extent) ||
        state->kv_state_extent < required_extent ||
        !yvex_core_u64_mul(state_slots, state->score_state_stride, &required_extent) ||
        state->score_state_extent < required_extent)
        return attention_history_refuse(
            layer, state_slots, state->kv_state_extent, failure, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention rolling state extent is too small");
    if (!attention_rolling_active_values_are_finite(
            state->kv_state, state->score_state, state->kv_state_stride, state->score_state_stride,
            head_dim, ratio, state->previous_fill, state->current_fill, overlap))
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                                layer->layer_index, 1ull, 0ull, err, YVEX_ERR_FORMAT,
                                "DeepSeek attention rolling state contains non-finite active values");
    return yvex_attention_accept(failure, err);
}
/*
 * Require the unique complete local and compressed history prefix.
 *
 * Returns typed missing, extra, unordered, or overflow refusal.
 */
static int attention_history_positions_validate(const yvex_attention_layer_plan *layer,
                                                const yvex_attention_history_view *history,
                                                yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long local_capacity, expected_local, local_start, i;
    unsigned long long expected_compressed = 0ull;
    if (!layer->sliding_window)
        return attention_history_refuse(
            layer, 1ull, 0ull, failure, err, YVEX_ERR_FORMAT,
            "DeepSeek attention history requires a nonzero sliding window");
    local_capacity = layer->sliding_window -
                     (layer->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT ? 0ull : 1ull);
    expected_local = history->token_count < local_capacity ? history->token_count : local_capacity;
    local_start = history->token_count - expected_local;
    if (history->local_tail_count != expected_local)
        return attention_history_refuse(
            layer, expected_local, history->local_tail_count, failure, err,
            history->local_tail_count > expected_local ? YVEX_ERR_BOUNDS : YVEX_ERR_FORMAT,
            "DeepSeek attention local history is not the complete window suffix");
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
        if (!layer->compression_ratio)
            return attention_history_refuse(
                layer, 1ull, 0ull, failure, err, YVEX_ERR_FORMAT,
                "compressed attention history requires a nonzero ratio");
        expected_compressed = history->token_count / layer->compression_ratio;
    }
    if (history->compressed_entry_count != expected_compressed)
        return attention_history_refuse(
            layer, expected_compressed, history->compressed_entry_count, failure, err,
            YVEX_ERR_FORMAT,
            "DeepSeek attention compressed history is not the complete prefix");
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        history->indexer_entry_count != expected_compressed)
        return attention_history_refuse(
            layer, expected_compressed, history->indexer_entry_count, failure, err,
            YVEX_ERR_FORMAT, "CSA indexer history is not the complete compressed prefix");
    for (i = 0ull; i < expected_local; ++i) {
        if (history->local_positions[i] != local_start + i)
            return attention_history_refuse(
                layer, local_start + i, history->local_positions[i], failure, err,
                YVEX_ERR_FORMAT, "local history does not contain the exact contiguous suffix");
    }
    for (i = 0ull; i < expected_compressed; ++i) {
        unsigned long long expected_position = 0ull;
        if (!yvex_core_u64_mul(i, layer->compression_ratio, &expected_position) ||
            history->compressed_positions[i] != expected_position ||
            (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
             history->indexer_positions[i] != expected_position))
            return attention_history_refuse(
                layer, expected_position, history->compressed_positions[i], failure, err,
                YVEX_ERR_FORMAT,
                "compressed history does not contain every completed ratio group");
    }
    return yvex_attention_accept(failure, err);
}

static void attention_rolling_copy_state(const yvex_attention_rolling_state_view *before,
                                         yvex_attention_rolling_state_output *after) {
    unsigned long long slot;
    for (slot = 0ull; slot < before->state_slots; ++slot) {
        memcpy(after->kv_state + slot * after->kv_state_stride,
               before->kv_state + slot * before->kv_state_stride,
               (size_t)(before->state_width * sizeof(float)));
        memcpy(after->score_state + slot * after->score_state_stride,
               before->score_state + slot * before->score_state_stride,
               (size_t)(before->state_width * sizeof(float)));
    }
}

static int attention_rolling_emit(const yvex_attention_rolling_state_output *state,
                                  unsigned long long head_dim, unsigned long long ratio,
                                  int overlap, float *compressed_out,
                                  unsigned long long compressed_out_count) {
    unsigned long long lane;
    unsigned long long slot;
    if (!state || !compressed_out || compressed_out_count < head_dim)
        return 0;
    for (lane = 0ull; lane < head_dim; ++lane) {
        double max_score = -HUGE_VAL;
        double denom = 0.0;
        double value = 0.0;
        for (slot = 0ull; slot < ratio; ++slot) {
            double score = state->score_state[slot * state->score_state_stride + lane];
            if (overlap) {
                double score2 =
                    state
                        ->score_state[(ratio + slot) * state->score_state_stride + lane + head_dim];
                if (score2 > max_score)
                    max_score = score2;
            }
            if (score > max_score)
                max_score = score;
        }
        for (slot = 0ull; slot < ratio; ++slot) {
            double score = state->score_state[slot * state->score_state_stride + lane];
            double weight = exp(score - max_score);
            denom += weight;
            value += weight * (double)state->kv_state[slot * state->kv_state_stride + lane];
            if (overlap) {
                double score2 =
                    state
                        ->score_state[(ratio + slot) * state->score_state_stride + lane + head_dim];
                double weight2 = exp(score2 - max_score);
                denom += weight2;
                value +=
                    weight2 *
                    (double)
                        state->kv_state[(ratio + slot) * state->kv_state_stride + lane + head_dim];
            }
        }
        if (!isfinite(denom) || denom <= 0.0 || !isfinite(value))
            return 0;
        compressed_out[lane] = (float)(value / denom);
    }
    return 1;
}

int yvex_attention_rolling_state_step_cpu(const yvex_attention_layer_plan *layer,
                                          const yvex_attention_rolling_state_view *before,
                                          const float *token_kv, const float *token_score,
                                          const float *ape_row,
                                          yvex_attention_rolling_state_output *after,
                                          float *compressed_out,
                                          unsigned long long compressed_out_count, int *emitted,
                                          yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long ratio, head_dim, state_width, state_slots, required_extent;
    unsigned long long slot, lane;
    attention_rolling_output_storage storage;
    int overlap, rotated, rc;
    if (emitted)
        *emitted = 0;
    rc = attention_rolling_state_validate(
        layer, before, before ? before->kind : YVEX_DEEPSEEK_ATTENTION_ROLLING_NONE, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (!token_kv || !token_score || !ape_row || !after)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                layer->layer_index, 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
                                "DeepSeek attention rolling transition requires token vectors and output state");
    if (before->next_token_position == ULLONG_MAX)
        return attention_history_refuse(
            layer, ULLONG_MAX, before->next_token_position, failure, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention rolling token position would overflow");
    if (!yvex_attention_rolling_geometry(layer, before->kind, &ratio, &head_dim, &state_width,
                                         &state_slots, &overlap, &rotated))
        return attention_history_refuse(
            layer, 1ull, 0ull, failure, err, YVEX_ERR_UNSUPPORTED,
            "DeepSeek attention rolling transition lacks geometry");
    if (!after->kv_state || !after->score_state || after->kv_state_stride < state_width ||
        after->score_state_stride < state_width ||
        !yvex_core_u64_mul(state_slots, after->kv_state_stride, &required_extent) ||
        after->kv_state_extent < required_extent ||
        !yvex_core_u64_mul(state_slots, after->score_state_stride, &required_extent) ||
        after->score_state_extent < required_extent)
        return attention_history_refuse(
            layer, state_width, after ? after->kv_state_stride : 0ull, failure, err,
            YVEX_ERR_FORMAT, "DeepSeek attention rolling output storage is incomplete");
    memcpy(&storage, &after->kv_state_stride, sizeof(storage));
    for (lane = 0ull; lane < state_width; ++lane) {
        if (!isfinite(token_kv[lane]) || !isfinite(token_score[lane]) || !isfinite(ape_row[lane]))
            return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                                    layer->layer_index, 1ull, lane, err, YVEX_ERR_FORMAT,
                                    "DeepSeek attention rolling transition input is non-finite");
    }
    memcpy(after, before, sizeof(*after));
    after->next_token_position = before->next_token_position + 1ull;
    after->cursor = (before->cursor + 1ull) % ratio;
    memcpy(&after->kv_state_stride, &storage, sizeof(storage));
    attention_rolling_copy_state(before, after);
    slot = overlap ? ratio + before->cursor : before->cursor;
    for (lane = 0ull; lane < state_width; ++lane) {
        after->kv_state[slot * after->kv_state_stride + lane] = token_kv[lane];
        after->score_state[slot * after->score_state_stride + lane] =
            token_score[lane] + ape_row[lane];
    }
    if (after->current_fill < before->cursor + 1ull)
        after->current_fill = before->cursor + 1ull;
    if (after->cursor == 0ull) {
        if (!attention_rolling_emit(after, head_dim, ratio, overlap, compressed_out,
                                    compressed_out_count))
            return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                                    layer->layer_index, head_dim, compressed_out_count, err,
                                    YVEX_ERR_FORMAT,
                                    "DeepSeek attention rolling compression emitted invalid values");
        if (overlap) {
            for (slot = 0ull; slot < ratio; ++slot) {
                memcpy(after->kv_state + slot * after->kv_state_stride,
                       after->kv_state + (ratio + slot) * after->kv_state_stride,
                       (size_t)(state_width * sizeof(float)));
                memcpy(after->score_state + slot * after->score_state_stride,
                       after->score_state + (ratio + slot) * after->score_state_stride,
                       (size_t)(state_width * sizeof(float)));
            }
            after->previous_fill = ratio;
        } else {
            after->previous_fill = 0ull;
        }
        after->current_fill = 0ull;
        after->cursor = 0ull;
        if (emitted)
            *emitted = 1;
    }
    return yvex_attention_accept(failure, err);
}

int yvex_attention_history_validate(const yvex_attention_layer_plan *layer,
                                    const yvex_attention_history_view *history,
                                    yvex_attention_failure *failure, yvex_error *err) {
    int rc;
    if (!layer || !history)
        return attention_refuse(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                                err, YVEX_ERR_INVALID_ARG,
                                "DeepSeek attention history validation requires layer and history");
    if (!history->immutable)
        return attention_history_refuse(layer, 1ull, 0ull, failure, err, YVEX_ERR_STATE,
                                        "DeepSeek attention history view must be immutable");
    if (history->local_tail_count && (!history->local_kv || !history->local_positions ||
                                      history->local_kv_stride < layer->head_dimension))
        return attention_history_refuse(
            layer, layer->head_dimension, history->local_kv_stride, failure, err, YVEX_ERR_FORMAT,
            "DeepSeek attention local history lacks raw KV storage");
    if (history->compressed_entry_count &&
        (!history->compressed_kv || !history->compressed_positions ||
         history->compressed_kv_stride < layer->head_dimension))
        return attention_history_refuse(
            layer, layer->head_dimension, history->compressed_kv_stride, failure, err,
            YVEX_ERR_FORMAT, "DeepSeek attention compressed history lacks KV storage");
    if (history->indexer_entry_count &&
        (!history->indexer_kv || !history->indexer_positions ||
         history->indexer_kv_stride < layer->indexer_head_dimension))
        return attention_history_refuse(
            layer, layer->indexer_head_dimension, history->indexer_kv_stride, failure, err,
            YVEX_ERR_FORMAT, "DeepSeek attention indexer history lacks KV storage");
    rc = attention_history_positions_validate(layer, history, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA &&
        (history->compressed_entry_count || history->indexer_entry_count))
        return attention_history_refuse(
            layer, 0ull, history->compressed_entry_count + history->indexer_entry_count, failure,
            err, YVEX_ERR_FORMAT,
            "SWA history may not carry compressed or indexer entries");
    if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA && history->indexer_entry_count)
        return attention_history_refuse(
            layer, 0ull, history->indexer_entry_count, failure, err, YVEX_ERR_FORMAT,
            "HCA history may not carry CSA indexer entries");
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA) {
        if (history->main_rolling_state.present || history->indexer_rolling_state.present)
            return attention_history_refuse(
                layer, 0ull, 1ull, failure, err, YVEX_ERR_FORMAT,
                "SWA history may not carry compressor rolling state");
    } else {
        rc = attention_rolling_state_validate(layer, &history->main_rolling_state,
                                              YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN, failure, err);
        if (rc != YVEX_OK)
            return rc;
        if (history->main_rolling_state.next_token_position != history->token_count)
            return attention_history_refuse(
                layer, history->token_count, history->main_rolling_state.next_token_position,
                failure, err, YVEX_ERR_STATE, "main rolling state token position is stale");
        if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
            rc = attention_rolling_state_validate(layer, &history->indexer_rolling_state,
                                                  YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER, failure,
                                                  err);
            if (rc != YVEX_OK)
                return rc;
            if (history->indexer_rolling_state.next_token_position != history->token_count)
                return attention_history_refuse(
                    layer, history->token_count, history->indexer_rolling_state.next_token_position,
                    failure, err, YVEX_ERR_STATE, "indexer rolling state token position is stale");
        } else if (history->indexer_rolling_state.present) {
            return attention_history_refuse(
                layer, 0ull, 1ull, failure, err, YVEX_ERR_FORMAT,
                "HCA history may not carry indexer rolling state");
        }
    }
    return yvex_attention_accept(failure, err);
}
/* Execution composes admitted generic operations without owning their math. */
#define YVEX_ATTENTION_PI 3.14159265358979323846264338327950288
enum {
    ATTENTION_PROBE_CPU = 0,
    ATTENTION_PROBE_CUDA,
    ATTENTION_PROBE_BACKEND_COUNT,
    ATTENTION_PROBE_CLASS_COUNT = 3,
    ATTENTION_PROBE_OWNED_BUFFERS = 6
};
static const double attention_comparison_absolute_tolerance = 5.0e-4;
static const double attention_comparison_relative_tolerance = 5.0e-4;
static const char *const probe_output_domain = "yvex.attention.tensor-output.v2";
static const char *const probe_state_domain = "yvex.attention.state-delta.v2";
struct yvex_attention_probe_history {
    yvex_attention_history_view view;
    void *owned[ATTENTION_PROBE_OWNED_BUFFERS];
    yvex_attention_workspace *workspace;
};
typedef struct {
    yvex_sha256 output_hash[ATTENTION_PROBE_BACKEND_COUNT];
    yvex_sha256 state_hash[ATTENTION_PROBE_BACKEND_COUNT];
    yvex_sha256 committed_state_hash;
    double squared_error;
    int committed_state_available;
} attention_probe_metrics;
typedef struct {
    yvex_attention_cpu_result evidence;
    yvex_attention_publication publication;
} attention_probe_backend;
typedef struct {
    const char *text;
    unsigned long long number;
} attention_probe_identity_field;
static const attention_probe_identity_field attention_comparison_identity_fields[] = {
    {NULL, 3ull},     /* schema */
    {NULL, 5ull},     /* tolerance numerator */
    {NULL, 10000ull}, /* tolerance denominator */
    {NULL, 1ull},     /* scale by max(abs(reference), 1) */
    {NULL, 1ull},     /* reject non-finite values */
    {NULL, 1ull},     /* RMSE covers finite values only */
    {NULL, 1ull},     /* observe bitwise equality with memcmp */
    {NULL, 0ull},     /* bitwise equality is not required */
    {NULL, 1ull},     /* traverse layer then coordinate */
    {NULL, 1ull},     /* compare state-delta numerics */
    {NULL, 1ull},     /* require exact state geometry */
    {NULL, 1ull},     /* require exact state positions */
};
typedef struct {
    unsigned long long count;
    unsigned long long width;
    unsigned long long seed;
    unsigned long long extent;
    float *values;
} attention_probe_segment;
typedef struct {
    const yvex_graph_execution_api *family;
    const yvex_attention_plan *plan;
    const void *family_ir;
    yvex_materialization_session *session;
    const yvex_runtime_descriptor *descriptor;
    const yvex_attention_probe_request *request;
    const yvex_attention_summary *summary;
    yvex_attention_probe_result candidate;
    yvex_attention_failure *failure;
    yvex_error *error;
    yvex_backend *cuda_backend;
    attention_probe_metrics metrics;
} attention_probe_context;

static int attention_probe_fail(yvex_error *error, int code, const char *message) {
    yvex_error_set(error, code, "attention.probe", message);
    return code;
}

static void attention_probe_history_release(yvex_attention_probe_history *history) {
    unsigned int index;
    if (!history)
        return;
    if (!history->workspace)
        for (index = 0u; index < ATTENTION_PROBE_OWNED_BUFFERS; ++index)
            free(history->owned[index]);
    memset(history, 0, sizeof(*history));
}

static void attention_probe_fill(float *values, float *scores, unsigned long long count,
                                 unsigned long long seed) {
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        unsigned long long code = (index * 37ull + seed * 19ull + 11ull) % 257ull;
        values[index] = (float)((long long)code - 128ll) / 257.0f;
        if (scores)
            scores[index] = values[index] * 0.5f;
    }
}

static void *attention_probe_calloc(yvex_attention_workspace *workspace,
                                    unsigned long long count, size_t width) {
    if (!count)
        return NULL;
    return workspace ? yvex_attention_workspace_calloc(workspace, count, width)
                     : yvex_attention_calloc_array(count, width);
}
/*
 * Construct one release-safe rolling state at its exact probe position.
 *
 * Admitted layer geometry, rolling kind, position, and plan identity. Bounded attention history,
 * never persistent KV.
 */
static int attention_probe_rolling_init(yvex_attention_probe_history *history,
                                        yvex_attention_rolling_state_view *state,
                                        const yvex_attention_layer_plan *layer,
                                        yvex_attention_rolling_kind kind,
                                        unsigned long long position, const char *plan_identity) {
    yvex_attention_failure failure = {0};
    yvex_error error;
    float *values = NULL, *scores = NULL;
    unsigned long long offset, extent;
    unsigned int owner = kind == YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER ? 4u : 2u;
    int rc;
    yvex_error_clear(&error);
    rc = yvex_attention_rolling_storage_acquire(
        layer, kind, position, history->workspace, &values, &scores, state,
        &failure, &error);
    history->owned[owner] = values;
    history->owned[owner + 1u] = scores;
    if (rc != YVEX_OK)
        return 0;
    state->current_fill = state->cursor;
    state->previous_fill = state->overlap && position >= state->ratio ? state->ratio : 0ull;
    extent = state->previous_fill * state->state_width;
    if (extent)
        attention_probe_fill(values, scores, extent,
                             layer->layer_index + (unsigned long long)kind + 17ull);
    offset = state->overlap ? state->ratio * state->state_width : 0ull;
    extent = state->current_fill * state->state_width;
    if (extent)
        attention_probe_fill(values + offset, scores + offset, extent,
                             layer->layer_index + (unsigned long long)kind + 31ull);
    yvex_core_text_copy(state->attention_plan_identity, sizeof(state->attention_plan_identity), plan_identity);
    return 1;
}
/*
 * Build real-geometry local, compressed, indexer, and rolling probe history.
 *
 * Admitted layer, summary, and causal position. Deterministic attention probe state, not prompt,
 * prefill, or KV.
 */
static int attention_probe_history_init(yvex_attention_probe_history *history,
                                        const yvex_attention_layer_plan *layer,
                                        const yvex_attention_summary *summary,
                                        unsigned long long position) {
    attention_probe_segment segments[3];
    unsigned long long value_count = 0ull, position_count, offset = 0ull, index;
    unsigned long long *positions, *local_positions, *compressed_positions;
    float *values;
    unsigned int segment;
    if (!history || !layer || !summary || !layer->sliding_window ||
        (layer->attention_class != YVEX_ATTENTION_CLASS_SWA && !layer->compression_ratio))
        return 0;
    {
        yvex_attention_workspace *workspace = history->workspace;
        memset(history, 0, sizeof(*history));
        history->workspace = workspace;
    }
    {
        unsigned long long local_capacity =
            layer->sliding_window -
            (layer->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT ? 0ull : 1ull);
        segments[0] = (attention_probe_segment){
        position < local_capacity ? position : local_capacity,
        layer->head_dimension, layer->layer_index + 101ull, 0ull, NULL};
    }
    segments[1] = (attention_probe_segment){
        layer->attention_class == YVEX_ATTENTION_CLASS_SWA
            ? 0ull
            : position / layer->compression_ratio,
        layer->head_dimension, layer->layer_index + 211ull, 0ull, NULL};
    segments[2] = (attention_probe_segment){
        layer->attention_class == YVEX_ATTENTION_CLASS_CSA ? segments[1].count : 0ull,
        layer->indexer_head_dimension, layer->layer_index + 307ull, 0ull, NULL};
    for (segment = 0u; segment < 3u; ++segment) {
        if (!yvex_core_u64_mul(segments[segment].count, segments[segment].width,
                               &segments[segment].extent) ||
            !yvex_core_u64_add(value_count, segments[segment].extent, &value_count))
            return 0;
    }
    if (!yvex_core_u64_add(segments[0].count, segments[1].count, &position_count))
        return 0;
    values = attention_probe_calloc(history->workspace, value_count, sizeof(*values));
    positions = attention_probe_calloc(history->workspace, position_count, sizeof(*positions));
    history->owned[0] = values;
    history->owned[1] = positions;
    if ((value_count && !values) || (position_count && !positions))
        goto fail;
    for (segment = 0u; segment < 3u; ++segment) {
        segments[segment].values = segments[segment].count ? values + offset : NULL;
        if (segments[segment].count)
            attention_probe_fill(segments[segment].values, NULL, segments[segment].extent,
                                 segments[segment].seed);
        offset += segments[segment].extent;
    }
    local_positions = positions;
    compressed_positions = segments[0].count ? positions + segments[0].count : positions;
    for (index = 0ull; index < segments[0].count; ++index)
        local_positions[index] = position - segments[0].count + index;
    for (index = 0ull; index < segments[1].count; ++index)
            compressed_positions[index] = index * layer->compression_ratio;
    history->view = (yvex_attention_history_view){
        .immutable = 1,
        .token_count = position,
        .local_tail_count = segments[0].count,
        .local_kv = segments[0].values,
        .local_positions = segments[0].count ? local_positions : NULL,
        .local_kv_stride = segments[0].count ? segments[0].width : 0ull,
        .compressed_entry_count = segments[1].count,
        .compressed_kv = segments[1].values,
        .compressed_positions = segments[1].count ? compressed_positions : NULL,
        .compressed_kv_stride = segments[1].count ? segments[1].width : 0ull,
        .indexer_entry_count = segments[2].count,
        .indexer_kv = segments[2].values,
        .indexer_positions = segments[2].count ? compressed_positions : NULL,
        .indexer_kv_stride = segments[2].count ? segments[2].width : 0ull,
    };
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA &&
        !attention_probe_rolling_init(history, &history->view.main_rolling_state, layer,
                                      YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN, position,
                                      summary->attention_plan_identity))
        goto fail;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        !attention_probe_rolling_init(history, &history->view.indexer_rolling_state, layer,
                                      YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER, position,
                                      summary->attention_plan_identity))
        goto fail;
    return 1;
fail:
    attention_probe_history_release(history);
    return 0;
}
/*
 * Construct one independently owned canonical probe history for a runtime seed.
 *
 * Admitted layer, plan summary, and causal position.
 */
int yvex_attention_probe_history_open(yvex_attention_probe_history **out,
    const yvex_attention_layer_plan *layer, const yvex_attention_summary *summary,
    unsigned long long position, const yvex_attention_history_view **view, yvex_error *err) {
    yvex_attention_probe_history *history;
    if (out) *out = NULL;
    if (view) *view = NULL;
    if (!out || !view || !position || !layer || !summary)
        return attention_probe_fail(err, YVEX_ERR_INVALID_ARG,
                                    "canonical probe history arguments are invalid");
    history = calloc(1u, sizeof(*history));
    if (!history || !attention_probe_history_init(history, layer, summary, position)) {
        free(history);
        return attention_probe_fail(err, YVEX_ERR_NOMEM,
                                    "canonical probe history allocation failed");
    }
    *out = history;
    *view = &history->view;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_attention_probe_history_close(yvex_attention_probe_history **history) {
    if (!history || !*history) return;
    attention_probe_history_release(*history);
    free(*history);
    *history = NULL;
}

static int attention_probe_identity(const char *domain,
                                    const attention_probe_identity_field *fields,
                                    size_t field_count, char output[YVEX_SHA256_HEX_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    size_t index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain))
        return 0;
    for (index = 0u; index < field_count; ++index) {
        int ok = fields[index].text ? yvex_sha256_update_text(&hash, fields[index].text)
                                    : yvex_sha256_update_u64(&hash, fields[index].number);
        if (!ok)
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int attention_probe_comparison_identity(char output[YVEX_SHA256_HEX_CAP]) {
    return attention_probe_identity(
        "yvex.attention.cpu-cuda.comparison.v3", attention_comparison_identity_fields,
        sizeof(attention_comparison_identity_fields) /
            sizeof(attention_comparison_identity_fields[0]),
        output);
}

static int attention_publication_identity(
    const attention_probe_context *context,
    const yvex_attention_layer_plan *layer,
    yvex_attention_publication *publication)
{
    const attention_probe_identity_field fields[] = {
        {context->summary->attention_plan_identity, 0ull},
        {context->request->logical_model_identity, 0ull},
        {context->request->input_identity, 0ull},
        {NULL, layer->layer_index},
        {NULL, context->request->operation_scope},
        {NULL, publication->token_position},
        {NULL, publication->token_count},
        {NULL, context->request->candidate_block_visible},
        {NULL, context->request->retain_prefix_checkpoints}};
    publication->token_ids = context->request->token_ids;
    return attention_probe_identity(
        "yvex.graph.attention.publication.v1", fields,
        sizeof(fields) / sizeof(fields[0]), publication->execution_identity);
}

static int attention_probe_compare(attention_probe_context *context,
                                   const yvex_attention_publication *cpu,
                                   const yvex_attention_publication *cuda) {
    int rc = yvex_attention_publication_compare(
        cpu, cuda, attention_comparison_absolute_tolerance,
        attention_comparison_relative_tolerance, &context->candidate,
        &context->metrics.squared_error, context->error);
    if (rc == YVEX_ERR_FORMAT)
        return attention_probe_fail(context->error, YVEX_ERR_FORMAT,
                                    "CPU/CUDA attention output or state comparison failed");
    return rc;
}
/*
 * Discard staged state without erasing the failure that required rollback.
 *
 * An abort failure supersedes the primary error because cleanup ownership is unresolved.
 */
static int attention_probe_state_abort(attention_probe_context *context, int primary_status)
{
    const yvex_attention_probe_state_provider *provider = context->request->state_provider;
    yvex_attention_failure primary_failure =
        context->failure ? *context->failure : (yvex_attention_failure){0};
    yvex_attention_failure cleanup_failure = {0};
    yvex_error primary_error = context->error ? *context->error : (yvex_error){0};
    yvex_error cleanup_error = {0};
    int rc;
    if (!provider)
        return YVEX_OK;
    rc = provider->abort(provider->context, &cleanup_failure, &cleanup_error);
    if (rc != YVEX_OK) {
        if (context->failure)
            *context->failure = cleanup_failure;
        if (context->error)
            *context->error = cleanup_error;
        return rc;
    }
    if (primary_status != YVEX_OK) {
        if (context->failure)
            *context->failure = primary_failure;
        if (context->error) {
            if (yvex_error_is_set(&primary_error))
                *context->error = primary_error;
            else
                yvex_error_set(context->error, (yvex_status)primary_status,
                               "graph.attention.execute",
                               primary_failure.reason ? primary_failure.reason
                                                      : "attention execution failed before rollback");
        }
    } else {
        if (context->failure)
            memset(context->failure, 0, sizeof(*context->failure));
        yvex_error_clear(context->error);
    }
    return YVEX_OK;
}

static int attention_probe_input_open(
    attention_probe_context *context, unsigned long long layer_ordinal,
    const yvex_attention_layer_plan *layer, unsigned long long position,
    unsigned long long token_count, unsigned long long input_width,
    const float **input, float **owned_input, unsigned long long *input_stride)
{
    unsigned long long input_count, token;
    int rc;
    if (!yvex_core_u64_mul(token_count, input_width, &input_count))
        return attention_probe_fail(context->error, YVEX_ERR_BOUNDS,
                                    "attention input extent overflowed");
    if (context->request->activation_view) {
        rc = context->request->activation_view(
            context->request->activation_context, layer_ordinal, token_count,
            input, input_stride, context->error);
        if (rc != YVEX_OK) return rc;
        if (!*input || *input_stride != input_width)
            return attention_probe_fail(
                context->error, YVEX_ERR_FORMAT,
                "activation input geometry disagrees with the admitted layer");
        return YVEX_OK;
    }
    if (context->request->device_view) {
        *input = NULL;
        *input_stride = input_width;
        return YVEX_OK;
    }
    *owned_input = attention_probe_calloc(
        context->request->workspace, input_count, sizeof(**owned_input));
    *input = *owned_input;
    if (!*input)
        return attention_probe_fail(
            context->error,
            context->request->workspace ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            context->request->workspace
                ? "canonical attention workspace capacity is insufficient"
                : "canonical attention input allocation failed");
    for (token = 0ull; token < token_count; ++token)
        attention_probe_fill(
            *owned_input + token * input_width, NULL, input_width,
            layer->layer_index + position + token + 1009ull +
                (unsigned long long)context->request->perturb_input);
    return YVEX_OK;
}

static int attention_probe_device_open(
    attention_probe_context *context, unsigned long long layer_ordinal,
    unsigned long long token_count, const yvex_device_tensor **input,
    yvex_device_tensor **output)
{
    int rc;
    if (!context->request->device_view) return YVEX_OK;
    rc = context->request->device_view(context->request->activation_context,
        layer_ordinal, token_count, input, output, context->error);
    if (rc != YVEX_OK) return rc;
    return *input && *output ? YVEX_OK : attention_probe_fail(
        context->error, YVEX_ERR_FORMAT, "device activation views are incomplete");
}

static int attention_probe_backend_execute(
    attention_probe_context *context, yvex_attention_cpu_options *options,
    const yvex_attention_layer_plan *layer, attention_probe_backend *run,
    unsigned int index)
{
    const int cuda = index == ATTENTION_PROBE_CUDA;
    int rc;
    options->publication = &run->publication;
    rc = cuda ? context->family->cuda_token_execute(
                    context->plan, context->family_ir, context->session,
                    context->descriptor, context->cuda_backend, options,
                    &run->evidence, context->failure, context->error)
              : context->family->cpu_chunk_execute(
                    context->plan, context->family_ir, context->session,
                    context->descriptor, options, &run->evidence,
                    context->failure, context->error);
    options->publication = NULL;
    if (rc != YVEX_OK) return rc;
    if (!(cuda ? run->evidence.cuda_executed : run->evidence.executed) ||
        !attention_publication_identity(context, layer, &run->publication) ||
        !yvex_attention_publication_hash_update(
            &context->metrics.output_hash[index],
            &context->metrics.state_hash[index], &run->publication))
        return attention_probe_fail(
            context->error, YVEX_ERR_STATE,
            cuda ? "CUDA attention publication was incomplete"
                 : "CPU attention publication was incomplete");
    if (context->request->evidence) {
        rc = context->request->evidence(
            context->request->evidence_context,
            cuda ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU,
            &run->publication, context->error);
        if (rc != YVEX_OK) return rc;
    }
    if (!yvex_core_u64_add(context->candidate.payload_bytes_read,
                           run->evidence.payload_bytes_read,
                           &context->candidate.payload_bytes_read) ||
        !yvex_core_u64_add(context->candidate.kernel_launches,
                           run->evidence.cuda_kernel_launches,
                           &context->candidate.kernel_launches) ||
        !yvex_core_u64_add(context->candidate.tensor_core_launches,
                           run->evidence.cuda_tensor_core_launches,
                           &context->candidate.tensor_core_launches) ||
        !yvex_core_u64_add(context->candidate.h2d_bytes,
                           run->evidence.cuda_h2d_bytes,
                           &context->candidate.h2d_bytes) ||
        !yvex_core_u64_add(context->candidate.d2h_bytes,
                           run->evidence.cuda_d2h_bytes,
                           &context->candidate.d2h_bytes) ||
        !yvex_core_u64_add(context->candidate.d2d_bytes,
                           run->evidence.cuda_d2d_bytes,
                           &context->candidate.d2d_bytes) ||
        !yvex_core_u64_add(context->candidate.stream_synchronizations,
                           run->evidence.cuda_stream_synchronizations,
                           &context->candidate.stream_synchronizations) ||
        !yvex_core_u64_add(context->candidate.device_synchronizations,
                           run->evidence.cuda_device_synchronizations,
                           &context->candidate.device_synchronizations) ||
        !yvex_core_u64_add(context->candidate.cuda_device_execution_elapsed_ns,
                           run->evidence.cuda_device_execution_elapsed_ns,
                           &context->candidate.cuda_device_execution_elapsed_ns))
        return attention_probe_fail(context->error, YVEX_ERR_BOUNDS,
                                    "attention execution counter overflowed");
    if (run->evidence.cuda_peak_device_bytes > context->candidate.peak_device_bytes)
        context->candidate.peak_device_bytes = run->evidence.cuda_peak_device_bytes;
    if (run->evidence.topk_selected > context->candidate.topk_selected)
        context->candidate.topk_selected = run->evidence.topk_selected;
    if (cuda && yvex_execution_memory_facts_merge(
                    &context->candidate.memory, &run->evidence.memory,
                    context->error) != YVEX_OK)
        return yvex_error_code(context->error);
    return YVEX_OK;
}
/*
 * Execute one real-geometry layer through each requested production backend.
 *
 * Admitted operator context, layer, and deterministic position.
 */
static int attention_probe_layer_execute(
    attention_probe_context *context, unsigned long long layer_ordinal,
    const yvex_attention_layer_plan *layer, unsigned long long position) {
    yvex_attention_cpu_options options;
    yvex_attention_cancellation cancellation = {
        context->request->cancel_requested, context->request->cancel_context};
    attention_probe_backend backend[ATTENTION_PROBE_BACKEND_COUNT] = {0};
    yvex_attention_probe_history history = {0};
    const yvex_attention_history_view *execution_history = NULL;
    unsigned long long token_count = context->request->token_count ? context->request->token_count : 1ull;
    unsigned long long input_width =
        context->request->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
            ? layer->residual_expanded_width : layer->hidden_dimension;
    unsigned long long input_stride = input_width;
    unsigned long long workspace_mark = 0ull;
    const float *input = NULL;
    const yvex_device_tensor *device_input = NULL;
    yvex_device_tensor *device_output = NULL;
    float *owned_input = NULL;
    unsigned int index;
    int state_started = 0, rc = YVEX_OK;
    if (context->request->workspace)
        workspace_mark = yvex_attention_workspace_mark(context->request->workspace);
    rc = attention_probe_input_open(
        context, layer_ordinal, layer, position, token_count, input_width,
        &input, &owned_input, &input_stride);
    if (rc != YVEX_OK) goto cleanup;
    rc = attention_probe_device_open(
        context, layer_ordinal, token_count, &device_input, &device_output);
    if (rc != YVEX_OK) goto cleanup;
    history.workspace = context->request->workspace;
    if (position && !context->request->state_provider &&
        !attention_probe_history_init(&history, layer, context->summary, position)) {
        rc = attention_probe_fail(
            context->error,
            context->request->workspace ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            context->request->workspace
                ? "canonical attention history exceeds prepared workspace capacity"
                : "canonical attention history allocation failed");
        goto cleanup;
    }
    execution_history = position ? &history.view : NULL;
    if (context->request->state_provider) {
        const yvex_attention_probe_state_provider *provider =
            context->request->state_provider;
        if (!provider->context || !provider->begin || !provider->stage || !provider->abort) {
            rc = attention_probe_fail(context->error, YVEX_ERR_INVALID_ARG,
                                      "attention state provider contract is incomplete");
            goto cleanup;
        }
        rc = provider->begin(provider->context, layer_ordinal, layer,
                             execution_history, position, token_count,
                             context->request->cancel_requested ? &cancellation : NULL,
                             &execution_history, context->failure, context->error);
        if (rc != YVEX_OK) goto cleanup;
        state_started = 1;
    }
    context->family->cpu_options_default(&options);
    options.operation_scope = context->request->operation_scope;
    options.execution_phase = context->request->execution_phase;
    options.logical_model_identity = context->request->logical_model_identity;
    options.layer_index = layer->layer_index;
    options.token_position = position;
    options.token_count = token_count;
    options.input = input;
    options.input_stride = input_stride;
    options.device_input = device_input;
    options.device_output = device_output;
    options.history = execution_history;
    options.evidence_level = context->request->evidence_level;
    options.execution_class = context->request->execution_class;
    options.workspace = context->request->workspace;
    options.cancellation = context->request->cancel_requested ? &cancellation : NULL;
    options.candidate_block_visible = context->request->candidate_block_visible;
    options.retain_prefix_checkpoints = context->request->retain_prefix_checkpoints;
    for (index = 0u; index < ATTENTION_PROBE_BACKEND_COUNT; ++index) {
        attention_probe_backend *run = &backend[index];
        int cuda = index == ATTENTION_PROBE_CUDA;
        if (!context->request->compare_backends &&
            context->request->backend != (cuda ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU))
            continue;
        rc = attention_probe_backend_execute(context, &options, layer, run, index);
        if (rc != YVEX_OK) goto cleanup;
    }
    if (context->request->compare_backends) {
        rc = attention_probe_compare(context, &backend[ATTENTION_PROBE_CPU].publication,
                                     &backend[ATTENTION_PROBE_CUDA].publication);
        if (rc != YVEX_OK) {
            if (!yvex_error_is_set(context->error))
                (void)attention_probe_fail(context->error, YVEX_ERR_FORMAT,
                                           "CPU/CUDA attention output geometry disagrees");
            goto cleanup;
        }
        if (state_started) {
            rc = attention_probe_state_abort(context, YVEX_OK);
            if (rc == YVEX_OK) state_started = 0;
        }
    }
    if (context->request->state_provider && !context->request->compare_backends) {
        const unsigned int committed_backend =
            context->request->backend == YVEX_BACKEND_KIND_CPU
                ? ATTENTION_PROBE_CPU
                : ATTENTION_PROBE_CUDA;
        char state_delta_identity[YVEX_SHA256_HEX_CAP];
        rc = context->request->state_provider->stage(
            context->request->state_provider->context,
            &backend[committed_backend].publication,
            context->request->cancel_requested ? &cancellation : NULL,
            state_delta_identity, context->failure, context->error);
        if (rc != YVEX_OK) goto cleanup;
        state_started = 0;
        if (!yvex_sha256_hex_valid(state_delta_identity) ||
            !yvex_sha256_update_u64(&context->metrics.committed_state_hash,
                                    layer->layer_index) ||
            !yvex_sha256_update_text(&context->metrics.committed_state_hash,
                                     state_delta_identity)) {
            rc = attention_probe_fail(context->error, YVEX_ERR_STATE,
                                      "attention state provider returned an invalid identity");
            goto cleanup;
        }
        context->metrics.committed_state_available = 1;
    }
    context->candidate.bindings_executed += layer->required_binding_count;
    context->candidate.layers_executed++;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA)
        context->candidate.hca_ratio = layer->compression_ratio;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA)
        ++context->candidate.swa_layers_executed;
    else if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA)
        ++context->candidate.csa_layers_executed;
    else
        ++context->candidate.hca_layers_executed;
cleanup:
    if (state_started) {
        int abort_rc = attention_probe_state_abort(context, rc);
        if (abort_rc != YVEX_OK)
            rc = abort_rc;
    }
    for (index = 0u; index < ATTENTION_PROBE_BACKEND_COUNT; ++index)
        yvex_attention_execution_trace_release(&backend[index].publication);
    attention_probe_history_release(&history);
    if (context->request->workspace) {
        yvex_error rewind_error;
        yvex_error_clear(&rewind_error);
        if (yvex_attention_workspace_rewind(
                context->request->workspace, workspace_mark, &rewind_error) != YVEX_OK &&
            rc == YVEX_OK) {
            rc = attention_probe_fail(context->error, YVEX_ERR_STATE,
                                      "attention workspace rewind failed after publication");
        }
    } else {
        free(owned_input);
    }
    return rc;
}
/*
 * Resolve one deep canonical class position plus a bounded repeat offset.
 *
 * Publishes one causal position without changing plan state. Malformed class geometry or overflow
 * refuses.
 */
int yvex_attention_probe_position_resolve(const yvex_attention_layer_plan *layer, int class_selected,
    unsigned long long offset, unsigned long long *position, yvex_error *err) {
    unsigned long long base, count;
    if (!layer || !position)
        return attention_probe_fail(err, YVEX_ERR_INVALID_ARG,
                                    "canonical probe position arguments are invalid");
    if (class_selected)
        base = layer->attention_class == YVEX_ATTENTION_CLASS_SWA ? 0ull : 1ull;
    else if (layer->attention_class == YVEX_ATTENTION_CLASS_SWA)
        base = layer->sliding_window;
    else if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        if (!yvex_core_u64_add(layer->indexer_topk, 1ull, &count) ||
            !yvex_core_u64_mul(layer->compression_ratio, count, &base))
            return attention_probe_fail(err, YVEX_ERR_BOUNDS,
                                        "canonical CSA probe position overflowed");
    } else if (layer->attention_class == YVEX_ATTENTION_CLASS_HCA && layer->compression_ratio)
        base = layer->compression_ratio - 1ull;
    else
        return attention_probe_fail(err, YVEX_ERR_FORMAT,
                                    "canonical probe position geometry is invalid");
    if (!yvex_core_u64_add(base, offset, position))
        return attention_probe_fail(err, YVEX_ERR_BOUNDS,
                                    "canonical probe position overflowed");
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Finalize output, execution, and comparison-contract identities.
 *
 * Comparison or identity refusal leaves the result incomplete. Execution evidence is not artifact
 * or generation identity.
 */
static int attention_probe_finalize(attention_probe_context *context) {
    yvex_attention_probe_result *result = &context->candidate;
    const yvex_attention_probe_request *request = context->request;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char *backend_digest[] = {result->cpu_output_digest, result->cuda_output_digest};
    char *backend_state_digest[] = {result->cpu_state_delta_digest,
                                    result->cuda_state_delta_digest};
    attention_probe_identity_field fields[19];
    const char *selected_digest;
    unsigned int index;
    if (!attention_probe_comparison_identity(result->comparison_contract_identity))
        return attention_probe_fail(context->error, YVEX_ERR_STATE,
                                    "comparison contract identity encoding failed");
    result->comparison_rmse =
        result->comparison_finite_values
            ? sqrt(context->metrics.squared_error / (double)result->comparison_finite_values)
            : 0.0;
    result->comparison_passed =
        request->compare_backends && result->comparison_values != 0ull &&
        result->comparison_finite_values == result->comparison_values &&
        result->comparison_nonfinite_values == 0ull &&
        isfinite(result->comparison_maximum_absolute_error) &&
        isfinite(result->comparison_maximum_relative_error) && isfinite(result->comparison_rmse) &&
        result->first_failing_layer == YVEX_ATTENTION_NO_LAYER;
    for (index = 0u; index < ATTENTION_PROBE_BACKEND_COUNT; ++index) {
        int selected = request->compare_backends ||
                       request->backend == (index ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU);
        if (selected && !yvex_sha256_final(&context->metrics.output_hash[index], digest))
            goto identity_failure;
        if (selected)
            yvex_sha256_hex(digest, backend_digest[index]);
        if (selected && !yvex_sha256_final(&context->metrics.state_hash[index], digest))
            goto identity_failure;
        if (selected)
            yvex_sha256_hex(digest, backend_state_digest[index]);
    }
    if (context->metrics.committed_state_available) {
        if (!yvex_sha256_final(&context->metrics.committed_state_hash, digest))
            goto identity_failure;
        yvex_sha256_hex(digest, result->state_delta_digest);
    }
    if (request->compare_backends) {
        result->output_digest_equal =
            strcmp(result->cpu_output_digest, result->cuda_output_digest) == 0;
        result->state_delta_digest_equal =
            strcmp(result->cpu_state_delta_digest, result->cuda_state_delta_digest) == 0;
        if (result->output_digest_equal)
            yvex_core_text_copy(result->tensor_output_digest,
                                sizeof(result->tensor_output_digest),
                                result->cpu_output_digest);
        if (result->state_delta_digest_equal && !context->metrics.committed_state_available)
            yvex_core_text_copy(result->state_delta_digest,
                                sizeof(result->state_delta_digest),
                                result->cpu_state_delta_digest);
        if (!result->comparison_passed)
            return attention_probe_fail(context->error, YVEX_ERR_FORMAT,
                                        "CPU/CUDA comparison contract failed");
    } else {
        index = request->backend == YVEX_BACKEND_KIND_CPU ? ATTENTION_PROBE_CPU
                                                          : ATTENTION_PROBE_CUDA;
        selected_digest = backend_digest[index];
        yvex_core_text_copy(result->tensor_output_digest, sizeof(result->tensor_output_digest), selected_digest);
        if (!context->metrics.committed_state_available) {
            selected_digest = backend_state_digest[index];
            yvex_core_text_copy(result->state_delta_digest, sizeof(result->state_delta_digest), selected_digest);
        }
    }
    fields[0] = (attention_probe_identity_field){context->summary->attention_plan_identity, 0ull};
    fields[1] = (attention_probe_identity_field){request->logical_model_identity, 0ull};
    fields[2] = (attention_probe_identity_field){
        request->input_identity, request->input_identity
                                     ? 0ull
                                     : YVEX_ATTENTION_PROBE_CANONICAL_V2};
    fields[3] = (attention_probe_identity_field){NULL, request->scope};
    fields[4] = (attention_probe_identity_field){NULL, request->operation_scope};
    fields[5] = (attention_probe_identity_field){NULL, request->token_count};
    fields[6] = (attention_probe_identity_field){
        NULL, request->select_layer ? request->layer_ordinal : YVEX_ATTENTION_NO_LAYER};
    fields[7] = (attention_probe_identity_field){
        NULL, request->select_position ? request->token_position : YVEX_ATTENTION_NO_LAYER};
    fields[8] = (attention_probe_identity_field){
        NULL, request->compare_backends ? 2ull : (unsigned long long)request->backend};
    fields[9] = (attention_probe_identity_field){NULL, result->layers_executed};
    fields[10] = (attention_probe_identity_field){NULL, result->bindings_executed};
    fields[11] = (attention_probe_identity_field){result->cpu_output_digest, 0ull};
    fields[12] = (attention_probe_identity_field){result->cuda_output_digest, 0ull};
    fields[13] = (attention_probe_identity_field){result->cpu_state_delta_digest, 0ull};
    fields[14] = (attention_probe_identity_field){result->cuda_state_delta_digest, 0ull};
    fields[15] = (attention_probe_identity_field){result->state_delta_digest, 0ull};
    fields[16] = (attention_probe_identity_field){NULL, request->candidate_block_visible};
    fields[17] = (attention_probe_identity_field){NULL, request->execution_phase};
    fields[18] = (attention_probe_identity_field){NULL, request->execution_class};
    if (!attention_probe_identity("yvex.attention.operator.execution.v6", fields, 19u,
                                  result->attention_execution_identity))
        goto identity_failure;
    return YVEX_OK;
identity_failure:
    return attention_probe_fail(context->error, YVEX_ERR_STATE,
                                "attention execution identity encoding failed");
}

static int attention_probe_cuda_open(attention_probe_context *context) {
    yvex_backend_options options = {.kind = YVEX_BACKEND_KIND_CUDA};
    yvex_backend_capability_result capability = {0};
    yvex_backend_device_info device = {0};
    int rc = YVEX_OK;
    if (context->request->backend_context)
        context->cuda_backend = context->request->backend_context;
    else
        rc = yvex_backend_open(&context->cuda_backend, &options, context->error);
    if (rc == YVEX_OK)
        rc = yvex_backend_query_capability(context->cuda_backend,
                                           YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &capability,
                                           context->error);
    if (rc == YVEX_OK && capability.state != YVEX_BACKEND_CAPABILITY_SUPPORTED) {
        rc = attention_probe_fail(context->error, YVEX_ERR_UNSUPPORTED,
                                  "device-complete encoded attention is unavailable");
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_get_device_info(context->cuda_backend, &device, context->error);
    if (rc == YVEX_OK) {
        yvex_core_text_copy(context->candidate.cuda_device,
                            sizeof(context->candidate.cuda_device),
                            device.name ? device.name : "unknown");
        context->candidate.cuda_compute_capability_major = device.compute_capability_major;
        context->candidate.cuda_compute_capability_minor = device.compute_capability_minor;
    }
    return rc;
}

static void attention_probe_comparison_publish(yvex_attention_probe_result *result,
                                               const yvex_attention_probe_result *candidate) {
    size_t identities = offsetof(yvex_attention_probe_result, cuda_device) -
                        offsetof(yvex_attention_probe_result, cpu_output_digest);
    size_t counts = offsetof(yvex_attention_probe_result, cuda_compute_capability_major) -
                    offsetof(yvex_attention_probe_result, comparison_values);
    size_t metrics = sizeof(*result) -
                     offsetof(yvex_attention_probe_result, comparison_maximum_absolute_error);
    memcpy(&result->cpu_output_digest, &candidate->cpu_output_digest, identities);
    memcpy(&result->comparison_values, &candidate->comparison_values, counts);
    memcpy(&result->comparison_maximum_absolute_error,
           &candidate->comparison_maximum_absolute_error, metrics);
    result->comparison_passed = 0;
}

int yvex_attention_execute(
    const yvex_graph_execution_api *family, const yvex_attention_plan *plan,
    const void *family_ir, yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_execution_request *request,
    yvex_attention_probe_result *result,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_probe_context context = {.family = family,
                                       .plan = plan,
                                       .family_ir = family_ir,
                                       .session = session,
                                       .descriptor = descriptor,
                                       .request = request,
                                       .failure = failure,
                                       .error = err};
    unsigned long long index;
    int selected[ATTENTION_PROBE_CLASS_COUNT] = {0};
    int workspace_started = 0, rc = YVEX_OK;
    if (!request ||
        (((!request->activation_view && !request->device_view) &&
          request->probe != YVEX_ATTENTION_PROBE_CANONICAL_V2) ||
         ((request->activation_view || request->device_view) &&
          (request->probe != YVEX_ATTENTION_PROBE_UNSPECIFIED ||
           !yvex_sha256_hex_valid(request->input_identity))) ||
         (request->device_view && !request->activation_view &&
          (request->backend != YVEX_BACKEND_KIND_CUDA ||
           request->compare_backends ||
           request->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL))) ||
        (request->state_provider &&
         (request->activation_view || request->device_view) &&
         !request->token_ids) ||
        (request->backend != YVEX_BACKEND_KIND_CPU &&
         request->backend != YVEX_BACKEND_KIND_CUDA) ||
        (request->scope != YVEX_ATTENTION_PROBE_SCOPE_QUICK &&
         request->scope != YVEX_ATTENTION_PROBE_SCOPE_FULL) ||
        (request->operation_scope != YVEX_ATTENTION_OPERATION_CORE &&
         request->operation_scope != YVEX_ATTENTION_OPERATION_ENVELOPE) ||
        (request->execution_phase != YVEX_EXECUTION_PHASE_PREFILL &&
         request->execution_phase != YVEX_EXECUTION_PHASE_DECODE &&
         request->execution_phase != YVEX_EXECUTION_PHASE_MIXED &&
         request->execution_phase != YVEX_EXECUTION_PHASE_DRAFT &&
         request->execution_phase != YVEX_EXECUTION_PHASE_VERIFY) ||
        request->execution_class > YVEX_EXECUTION_CLASS_FORENSIC_REFERENCE)
        return attention_probe_fail(err, YVEX_ERR_INVALID_ARG, "canonical V2 probe request is invalid");
    if (!family || !plan || !session || !descriptor || !result ||
        !family->cpu_options_default || !family->cpu_chunk_execute || !family->cuda_token_execute)
        return attention_probe_fail(err, YVEX_ERR_INVALID_ARG,
                                    "sealed attention owners and execution API are required");
    context.summary = yvex_attention_plan_summary(plan);
    if (!context.summary || !context.summary->full_execution_ready ||
        !context.summary->cpu_reference_ready ||
        ((request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA) &&
         !context.summary->cuda_execution_ready))
        return attention_probe_fail(err, YVEX_ERR_UNSUPPORTED,
                                    "requested production attention capability is not admitted");
    if ((unsigned int)request->evidence_level >
            (unsigned int)YVEX_ATTENTION_EVIDENCE_FULL ||
        (request->select_layer &&
         request->layer_ordinal >= yvex_attention_plan_layer_count(plan)) ||
        (request->select_position && !request->select_layer && !request->state_provider))
        return attention_probe_fail(err, YVEX_ERR_BOUNDS,
                                    "selected attention probe layer or position is invalid");
    memset(&context.candidate, 0, sizeof(context.candidate));
    context.candidate.comparison_available = request->compare_backends;
    context.candidate.first_failing_layer = YVEX_ATTENTION_NO_LAYER;
    context.candidate.first_failing_coordinate = YVEX_ATTENTION_NO_LAYER;
    context.candidate.bitwise_equality_observed = request->compare_backends;
    yvex_sha256_init(&context.metrics.committed_state_hash);
    if (!yvex_sha256_update_text(&context.metrics.committed_state_hash,
                                 "yvex.attention.committed-state-delta.v1"))
        return attention_probe_fail(err, YVEX_ERR_STATE,
                                    "attention state identity initialization failed");
    for (index = 0ull; index < ATTENTION_PROBE_BACKEND_COUNT; ++index) {
        yvex_sha256_init(&context.metrics.output_hash[index]);
        yvex_sha256_init(&context.metrics.state_hash[index]);
        if (!yvex_sha256_update_text(&context.metrics.output_hash[index], probe_output_domain) ||
            !yvex_sha256_update_text(&context.metrics.state_hash[index], probe_state_domain))
            return attention_probe_fail(err, YVEX_ERR_STATE,
                                        "attention output identity initialization failed");
    }
    if (request->workspace) {
        rc = yvex_attention_workspace_begin(request->workspace, err);
        if (rc != YVEX_OK)
            return rc;
        workspace_started = 1;
    }
    if (request->compare_backends || request->backend == YVEX_BACKEND_KIND_CUDA) {
        rc = attention_probe_cuda_open(&context);
        if (rc != YVEX_OK)
            goto cleanup;
    }
    for (index = 0ull; index < yvex_attention_plan_layer_count(plan); ++index) {
        const yvex_attention_layer_plan *layer = yvex_attention_plan_layer_at(plan, index);
        int class_selected;
        int include;
        unsigned long long position = request->token_position;
        if (!layer) {
            rc = attention_probe_fail(err, YVEX_ERR_STATE,
                                      "attention layer disappeared during traversal");
            goto cleanup;
        }
        class_selected = (unsigned int)layer->attention_class >= ATTENTION_PROBE_CLASS_COUNT ||
                         selected[layer->attention_class];
        include = request->select_layer ? index == request->layer_ordinal
                                        : request->scope == YVEX_ATTENTION_PROBE_SCOPE_FULL ||
                                              !class_selected;
        if (!include)
            continue;
        if ((unsigned int)layer->attention_class < ATTENTION_PROBE_CLASS_COUNT)
            selected[layer->attention_class] = 1;
        if (!request->select_position)
            rc = yvex_attention_probe_position_resolve(
                layer, class_selected, request->token_position, &position, err);
        if (rc == YVEX_OK)
            rc = attention_probe_layer_execute(&context, index, layer, position);
        if (rc != YVEX_OK) {
            if (rc == YVEX_ERR_FORMAT && request->compare_backends)
                attention_probe_comparison_publish(result, &context.candidate);
            goto cleanup;
        }
    }
    if ((request->select_layer && context.candidate.layers_executed != 1ull) ||
        (!request->select_layer && request->scope == YVEX_ATTENTION_PROBE_SCOPE_QUICK &&
         (context.candidate.layers_executed != 3ull || !selected[0] || !selected[1] ||
          !selected[2])) ||
        (!request->select_layer && request->scope == YVEX_ATTENTION_PROBE_SCOPE_FULL &&
         (context.candidate.layers_executed != context.summary->layer_count ||
          context.candidate.bindings_executed != context.summary->required_binding_count))) {
        rc = attention_probe_fail(err, YVEX_ERR_STATE,
                                  "requested attention scope did not execute completely");
        goto cleanup;
    }
    rc = attention_probe_finalize(&context);
    if (rc == YVEX_ERR_FORMAT && request->compare_backends)
        attention_probe_comparison_publish(result, &context.candidate);
cleanup:
    if (!request->backend_context)
        yvex_backend_close(context.cuda_backend);
    if (workspace_started) {
        yvex_error workspace_error;
        int workspace_rc;
        yvex_error_clear(&workspace_error);
        if (rc != YVEX_OK)
            (void)yvex_attention_workspace_rewind(
                request->workspace, 0ull, &workspace_error);
        workspace_rc = yvex_attention_workspace_finish(
            request->workspace, &workspace_error);
        if (workspace_rc != YVEX_OK && rc == YVEX_OK)
            rc = attention_probe_fail(
                err, YVEX_ERR_STATE,
                "attention workspace retained spans after execution publication");
    }
    if (rc != YVEX_OK && request->state_provider) {
        int abort_rc = attention_probe_state_abort(&context, rc);
        if (abort_rc != YVEX_OK)
            rc = abort_rc;
    }
    if (rc == YVEX_OK) {
        *result = context.candidate;
        yvex_error_clear(err);
    }
    return rc;
}

int yvex_attention_probe_execute(
    const yvex_graph_execution_api *family, const yvex_attention_plan *plan,
    const void *family_ir, yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_probe_request *request,
    yvex_attention_probe_result *result,
    yvex_attention_failure *failure, yvex_error *err)
{
    if (!request || request->activation_view || request->input_identity ||
        request->token_ids)
        return attention_probe_fail(
            err, YVEX_ERR_INVALID_ARG,
            "canonical probe cannot borrow an activation input");
    return yvex_attention_execute(family, plan, family_ir, session, descriptor,
                                  request, result, failure, err);
}
