/*
 * Execute one DeepSeek-selected MoE layer with canonical qtype kernels on CUDA.
 *
 * Production weights remain directly addressable and success follows device completion. Backend
 * execution consumes a typed MoE job and never reconstructs model topology.
 */
#include <yvex/internal/moe.h>
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#define MOE_CUDA_BLOCK 256u
#define MOE_CUDA_ROWS_PER_BLOCK 8u
#define MOE_CUDA_Q8_BYTES 292ull
struct yvex_backend_moe_execution {
    yvex_backend *backend;
    yvex_cuda_backend_state *state;
    const yvex_cuda_attention_operations *ops;
    yvex_cuda_work work;
    yvex_backend_attention_failure failure;
    const yvex_moe_layer_job *job;
    CUdeviceptr status, expanded, normalized, post, combination;
    CUdeviceptr mix, scale, base, logits, scores, selected, weights;
    CUdeviceptr gate, up, intermediate, expert, routed, shared, combined, weight_buffer, route_aux;
    size_t weight_buffer_bytes, route_aux_bytes;
    int host_status, finished, grouped_selected;
    unsigned long long h2d, d2h, subviews, uploads, downloads, direct_weights;
    unsigned long long d2d, device_synchronizations, started_ns, ingress_ns, routing_ns;
    unsigned long long routed_ns, shared_ns, synchronization_ns;
};
typedef struct {
    yvex_backend *backend;
    yvex_cuda_backend_state *state;
    const yvex_cuda_attention_operations *ops;
    yvex_cuda_work work;
    yvex_backend_attention_failure failure;
    CUdeviceptr status, expanded, normalized, post, combination, mix, scale, base;
    CUdeviceptr logits, scores, selected, weights, tokens, order, unique;
    CUdeviceptr input_q8, routed_intermediate, routed_intermediate_q8;
    CUdeviceptr routed_pairs, routed, shared_intermediate, shared_intermediate_q8;
    CUdeviceptr shared_pairs, shared, combined;
    int host_status;
    unsigned long long host_unique, h2d, d2h, d2d, downloads;
    unsigned long long started_ns, synchronization_ns;
    unsigned long long stream_synchronizations, device_synchronizations;
} moe_cuda_batch;
static int moe_cuda_add_selected(yvex_backend_moe_execution *execution, yvex_error *err);

static int moe_cuda_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "cuda.moe", reason);
    return status;
}

static int moe_cuda_mhc_shared_bytes(unsigned long long streams,
                                     unsigned int *shared_bytes,
                                     yvex_error *err)
{
    unsigned long long count, bytes;
    if (!shared_bytes ||
        !yvex_core_u64_add(streams, 1ull + MOE_CUDA_BLOCK, &count) ||
        !yvex_core_u64_mul(count, sizeof(double), &bytes) || bytes > UINT_MAX)
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA mHC shared geometry exceeds launch bounds");
    *shared_bytes = (unsigned int)bytes;
    return YVEX_OK;
}

static int moe_cuda_q8_eligible(unsigned int qtype)
{
    return yvex_cuda_q8_activation_eligible(qtype);
}

static int moe_cuda_workspace_add(unsigned long long *total,
                                  unsigned long long count,
                                  unsigned long long width)
{
    unsigned long long bytes, aligned;
    if (!total || !count || !width || !yvex_core_u64_mul(count, width, &bytes) ||
        !yvex_core_u64_add(bytes, 255ull, &aligned))
        return 0;
    aligned &= ~255ull;
    return yvex_core_u64_add(*total, aligned, total);
}

static int moe_cuda_grid(unsigned long long tasks, unsigned int tasks_per_block,
                         unsigned int *grid)
{
    unsigned long long blocks;
    if (!tasks || !tasks_per_block || !grid) return 0;
    blocks = tasks / tasks_per_block + (tasks % tasks_per_block != 0ull);
    if (!blocks || blocks > UINT_MAX) return 0;
    *grid = (unsigned int)blocks;
    return 1;
}

static int moe_cuda_rows_q8(const yvex_moe_layer_plan *layer,
                            const yvex_moe_weight_view weights[3],
                            unsigned long long intermediate_width)
{
    return layer && weights && layer->hidden_width % 256ull == 0ull &&
           intermediate_width % 256ull == 0ull &&
           moe_cuda_q8_eligible(weights[0].qtype) &&
           moe_cuda_q8_eligible(weights[1].qtype) &&
           moe_cuda_q8_eligible(weights[2].qtype);
}

static int moe_cuda_plan_q8(const yvex_moe_layer_plan *layer,
                            unsigned int base,
                            unsigned long long intermediate_width)
{
    return layer && layer->hidden_width % 256ull == 0ull &&
           intermediate_width % 256ull == 0ull &&
           moe_cuda_q8_eligible(layer->qtypes[base]) &&
           moe_cuda_q8_eligible(layer->qtypes[base + 1u]) &&
           moe_cuda_q8_eligible(layer->qtypes[base + 2u]);
}

static int moe_cuda_rows_workspace_required(const yvex_moe_layer_plan *layer,
                                             unsigned long long row_count,
                                             unsigned long long *bytes,
                                             yvex_error *err)
{
    unsigned long long total = 0ull, pairs, expanded, hidden, post, combination;
    unsigned long long mix, logits, routed_intermediate, shared_intermediate;
    unsigned long long pair_outputs;
    unsigned long long input_q8, routed_q8, shared_q8;
    unsigned long long maximum_input = 0ull;
    int routed_path_q8, shared_path_q8;
    if (bytes) *bytes = 0ull;
    if (!layer || !bytes || !row_count || !layer->hidden_width ||
        !layer->residual_streams || !layer->expanded_width ||
        !layer->mhc_mixing_rows || !layer->routed_experts ||
        !layer->experts_per_token || !layer->expert_intermediate_width ||
        !layer->shared_intermediate_width || layer->shared_experts != 1ull ||
        !yvex_core_u64_mul(row_count, layer->experts_per_token, &pairs) ||
        !yvex_core_u64_mul(row_count, layer->expanded_width, &expanded) ||
        !yvex_core_u64_mul(row_count, layer->hidden_width, &hidden) ||
        !yvex_core_u64_mul(row_count, layer->residual_streams, &post) ||
        !yvex_core_u64_mul(post, layer->residual_streams, &combination) ||
        !yvex_core_u64_mul(row_count, layer->mhc_mixing_rows, &mix) ||
        !yvex_core_u64_mul(row_count, layer->routed_experts, &logits) ||
        !yvex_core_u64_mul(pairs, layer->expert_intermediate_width,
                           &routed_intermediate) ||
        !yvex_core_u64_mul(row_count, layer->shared_intermediate_width,
                           &shared_intermediate) ||
        !yvex_core_u64_mul(pairs, layer->hidden_width, &pair_outputs))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE workspace geometry is invalid");
    routed_path_q8 = moe_cuda_plan_q8(
        layer, YVEX_MOE_WEIGHT_ROUTED_GATE, layer->expert_intermediate_width);
    shared_path_q8 = moe_cuda_plan_q8(
        layer, YVEX_MOE_WEIGHT_SHARED_GATE, layer->shared_intermediate_width);
    if (moe_cuda_q8_eligible(layer->qtypes[YVEX_MOE_WEIGHT_MHC_FUNCTION]) &&
        layer->expanded_width % 256ull == 0ull)
        maximum_input = layer->expanded_width;
    if ((routed_path_q8 || shared_path_q8 ||
         (moe_cuda_q8_eligible(layer->qtypes[YVEX_MOE_WEIGHT_ROUTER]) &&
          layer->hidden_width % 256ull == 0ull)) &&
        layer->hidden_width > maximum_input)
        maximum_input = layer->hidden_width;
    if (!yvex_core_u64_mul(row_count, maximum_input / 256ull, &input_q8) ||
        !yvex_core_u64_mul(pairs, layer->expert_intermediate_width / 256ull,
                           &routed_q8) ||
        !yvex_core_u64_mul(row_count, layer->shared_intermediate_width / 256ull,
                           &shared_q8))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE workspace geometry is invalid");
#define ADD(count_, width_)                                                                \
    do { if (!moe_cuda_workspace_add(&total, (count_), (width_))) goto overflow; } while (0)
    ADD(1ull, sizeof(int));
    ADD(expanded, sizeof(float));
    ADD(hidden, sizeof(float));
    ADD(post, sizeof(float));
    ADD(combination, sizeof(float));
    ADD(mix, sizeof(float));
    ADD(3ull, sizeof(float));
    ADD(layer->mhc_mixing_rows, sizeof(float));
    ADD(logits, sizeof(float));
    ADD(logits, sizeof(float));
    ADD(pairs, sizeof(unsigned long long));
    ADD(pairs, sizeof(float));
    ADD(row_count, sizeof(unsigned int));
    ADD(pairs, sizeof(unsigned long long));
    ADD(1ull, sizeof(unsigned long long));
    if (input_q8) ADD(input_q8, MOE_CUDA_Q8_BYTES);
    ADD(routed_intermediate, sizeof(float));
    if (routed_path_q8 && routed_q8) ADD(routed_q8, MOE_CUDA_Q8_BYTES);
    ADD(pair_outputs, sizeof(float));
    ADD(hidden, sizeof(float));
    ADD(shared_intermediate, sizeof(float));
    if (shared_path_q8 && shared_q8) ADD(shared_q8, MOE_CUDA_Q8_BYTES);
    ADD(hidden, sizeof(float));
    ADD(hidden, sizeof(float));
    ADD(hidden, sizeof(float));
#undef ADD
    *bytes = total;
    yvex_error_clear(err);
    return YVEX_OK;
overflow:
    return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                           "CUDA width-N MoE workspace extent overflowed");
}

static yvex_backend_attention_weight moe_cuda_weight(const yvex_moe_weight_view *weight)
{
    yvex_backend_attention_weight out = {0};
    if (!weight) return out;
    out.encoded = weight->encoded;
    out.encoded_bytes = weight->encoded_bytes;
    out.row_bytes = weight->row_bytes;
    out.row_width = weight->row_width;
    out.row_count = weight->row_count;
    out.qtype = weight->qtype;
    out.present = weight->encoded && weight->encoded_bytes != 0u;
    return out;
}

static int moe_cuda_allocate(yvex_backend_moe_execution *execution, CUdeviceptr *out,
                             size_t bytes, const void *source, int zero,
                             const char *stage, yvex_error *err)
{
    return execution->ops->allocate(&execution->work, out, bytes, source, zero, stage,
                                    &execution->failure, err);
}

static int moe_cuda_upload(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight,
                           const char *stage, yvex_error *err)
{
    int rc = YVEX_OK;
    if (!weight || !weight->encoded || !weight->encoded_bytes ||
        weight->encoded_bytes > execution->weight_buffer_bytes)
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "selected MoE encoded weight exceeds its stable staging range");
    rc = execution->ops->initialize(&execution->work, execution->weight_buffer,
                                    weight->encoded_bytes, weight->encoded, 0, stage,
                                    &execution->failure, err);
    if (rc == YVEX_OK) {
        execution->h2d += weight->encoded_bytes;
        execution->uploads++;
    }
    return rc;
}

static int moe_cuda_weight_address(yvex_backend_moe_execution *execution,
                                   const yvex_moe_weight_view *weight,
                                   CUdeviceptr *device, const char *stage,
                                   yvex_error *err)
{
    int rc;
    if (!device) return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                                        "CUDA MoE weight address output is required");
    *device = 0ull;
    if (weight && weight->device_address) {
        *device = (CUdeviceptr)weight->device_address;
        execution->direct_weights++;
        return YVEX_OK;
    }
    rc = moe_cuda_upload(execution, weight, stage, err);
    if (rc == YVEX_OK) *device = execution->weight_buffer;
    return rc;
}

static int moe_cuda_matvec(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight, CUdeviceptr input,
                           CUdeviceptr output, int round_bf16,
                           const char *stage, yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    CUdeviceptr device_weight = 0ull;
    int rc = moe_cuda_weight_address(execution, weight, &device_weight, stage, err);
    return rc == YVEX_OK
               ? execution->ops->matvec(&execution->work, &encoded,
                                         device_weight, 0ull,
                                         weight->row_count, 1ull, input, output, round_bf16,
                                         execution->status, stage, &execution->failure, err)
               : rc;
}

static int moe_cuda_decode(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight, CUdeviceptr output,
                           const char *stage, yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    CUdeviceptr device_weight = 0ull;
    int rc = moe_cuda_weight_address(execution, weight, &device_weight, stage, err);
    return rc == YVEX_OK
               ? execution->ops->decode(&execution->work, &encoded,
                                         device_weight, 0ull,
                                         weight->row_width, output, execution->status,
                                         stage, &execution->failure, err)
               : rc;
}

static int moe_cuda_sync_status(yvex_backend_moe_execution *execution,
                                const char *stage, yvex_error *err)
{
    unsigned long long started = 0ull, completed = 0ull;
    int rc = execution->ops->download(
        &execution->work, &execution->host_status, execution->status,
        sizeof(execution->host_status), stage, &execution->failure, err);
    if (rc == YVEX_OK) {
        execution->downloads++;
        execution->d2h += sizeof(execution->host_status);
        started = yvex_core_monotonic_ns();
        rc = yvex_cuda_synchronize(execution->backend,
            YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, stage, err);
        completed = yvex_core_monotonic_ns();
        execution->device_synchronizations++;
        if (completed > started) execution->synchronization_ns += completed - started;
    }
    if (rc == YVEX_OK && execution->host_status)
        rc = moe_cuda_refuse(err, YVEX_ERR_BACKEND,
                             "CUDA MoE kernel reported invalid or non-finite numerics");
    return rc;
}

static int moe_cuda_ranges(yvex_backend_moe_execution *execution, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = execution->job->layer;
    size_t hidden = (size_t)layer->hidden_width * sizeof(float);
    size_t expanded = (size_t)layer->expanded_width * sizeof(float);
    unsigned long long grouped_count = layer->expert_intermediate_width *
                                       layer->experts_per_token;
    size_t intermediate;
    size_t routed = (size_t)layer->routed_experts * sizeof(float);
    size_t selected = (size_t)layer->experts_per_token * sizeof(unsigned long long);
    size_t selected_weights = (size_t)layer->experts_per_token * sizeof(float);
    if (grouped_count < layer->shared_intermediate_width)
        grouped_count = layer->shared_intermediate_width;
    if (grouped_count > SIZE_MAX / sizeof(float))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA MoE intermediate extent overflowed");
    intermediate = (size_t)grouped_count * sizeof(float);
    int rc = moe_cuda_allocate(execution, &execution->status, sizeof(int), NULL, 1,
                               "cuda.moe.status", err);
#define RANGE(member_, bytes_, source_, zero_, stage_)                                      \
    do {                                                                                    \
        if (rc == YVEX_OK)                                                                  \
            rc = moe_cuda_allocate(execution, &execution->member_, (bytes_), (source_),     \
                                   (zero_), (stage_), err);                                  \
    } while (0)
    RANGE(expanded, expanded, execution->job->device_input ? NULL : execution->job->expanded_input,
          0, "cuda.moe.input");
    RANGE(normalized, hidden, NULL, 1, "cuda.moe.normalized");
    RANGE(post, (size_t)layer->residual_streams * sizeof(float), NULL, 1, "cuda.moe.post");
    RANGE(combination, (size_t)layer->residual_streams * layer->residual_streams * sizeof(float),
          NULL, 1, "cuda.moe.combination");
    RANGE(mix, (size_t)layer->mhc_mixing_rows * sizeof(float), NULL, 1, "cuda.moe.mix");
    RANGE(scale, 3u * sizeof(float), NULL, 1, "cuda.moe.scale");
    RANGE(base, (size_t)layer->mhc_mixing_rows * sizeof(float), NULL, 1, "cuda.moe.base");
    RANGE(logits, routed, NULL, 1, "cuda.moe.logits");
    RANGE(scores, routed, NULL, 1, "cuda.moe.scores");
    RANGE(selected, selected, NULL, 1, "cuda.moe.selected");
    RANGE(weights, selected_weights, NULL, 1, "cuda.moe.weights");
    RANGE(gate, intermediate, NULL, 1, "cuda.moe.gate");
    RANGE(up, intermediate, NULL, 1, "cuda.moe.up");
    RANGE(intermediate, intermediate, NULL, 1, "cuda.moe.intermediate");
    RANGE(expert, hidden, NULL, 1, "cuda.moe.expert");
    RANGE(routed, hidden, NULL, 1, "cuda.moe.routed");
    RANGE(shared, hidden, NULL, 1, "cuda.moe.shared");
    RANGE(combined, hidden, NULL, 1, "cuda.moe.combined");
    RANGE(weight_buffer, execution->weight_buffer_bytes, NULL, 1, "cuda.moe.weight-buffer");
    RANGE(route_aux, execution->route_aux_bytes, NULL, 1, "cuda.moe.route-aux");
#undef RANGE
    if (rc == YVEX_OK && execution->job->device_input) {
        CUstream stream = yvex_cuda_launch_stream(execution->backend);
        CUresult copied = stream && execution->state->driver.cuMemcpyDtoDAsync_v2
                              ? execution->state->driver.cuMemcpyDtoDAsync_v2(
                                    execution->expanded,
                                    (CUdeviceptr)execution->job->device_input->data,
                                    expanded, stream)
                              : !stream ? execution->state->driver.cuMemcpyDtoD_v2(
                                    execution->expanded,
                                    (CUdeviceptr)execution->job->device_input->data,
                                    expanded) : (CUresult)1;
        rc = yvex_cuda_status(&execution->state->driver, copied,
                              "cuda.moe.device-input", err);
        if (rc == YVEX_OK) execution->d2d += expanded;
    }
    if (rc == YVEX_OK && !execution->job->device_input) execution->h2d += expanded;
    return rc;
}

static int moe_cuda_prepare_input(yvex_backend_moe_execution *execution, yvex_error *err)
{
    const yvex_moe_layer_job *job = execution->job;
    const yvex_moe_layer_plan *layer = job->layer;
    yvex_backend_attention_weight norm = moe_cuda_weight(&job->weights[YVEX_MOE_WEIGHT_FFN_NORM]);
    CUdeviceptr norm_weight = 0ull;
    unsigned long long streams = layer->residual_streams, width = layer->hidden_width;
    unsigned int shared_bytes = 0u;
    int rc = moe_cuda_mhc_shared_bytes(streams, &shared_bytes, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_matvec(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                             execution->expanded, execution->mix, 0,
                             "cuda.moe.mhc-function", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_decode(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                             execution->scale, "cuda.moe.mhc-scale", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_decode(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_BASE],
                             execution->base, "cuda.moe.mhc-base", err);
    if (rc == YVEX_OK) {
        unsigned long long one = 1ull;
        void *params[] = {
            &execution->expanded, &execution->mix, &execution->scale, &execution->base,
            &streams, &width, (void *)&layer->mhc_mixing_rows,
            (void *)&layer->mhc_sinkhorn_iterations, (void *)&layer->rms_epsilon,
            (void *)&layer->mhc_epsilon, (void *)&layer->mhc_post_multiplier,
            &execution->normalized, &execution->post, &execution->combination,
            &one, &execution->status};
        rc = execution->ops->launch(
            &execution->work, execution->state->deepseek_mhc_pre_function,
            1u, MOE_CUDA_BLOCK, shared_bytes, params, "cuda.moe.mhc-pre",
            &execution->failure, err);
    }
    if (rc == YVEX_OK &&
        (rc = moe_cuda_weight_address(execution, &job->weights[YVEX_MOE_WEIGHT_FFN_NORM],
                                      &norm_weight, "cuda.moe.norm-weight", err)) == YVEX_OK)
        rc = execution->ops->weighted_norm(
            &execution->work, execution->normalized, layer->hidden_width, 1ull, &norm,
            norm_weight, layer->rms_epsilon, execution->status,
            "cuda.moe.ffn-norm", &execution->failure, err);
    return rc;
}

static int moe_cuda_route(yvex_backend_moe_execution *execution,
                          yvex_moe_layer_result *result, yvex_error *err)
{
    const yvex_moe_layer_job *job = execution->job;
    const yvex_moe_layer_plan *layer = job->layer;
    const yvex_moe_weight_view *aux = &job->weights[
        layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID
            ? YVEX_MOE_WEIGHT_ROUTER_TABLE : YVEX_MOE_WEIGHT_ROUTER_BIAS];
    unsigned int router_class = (unsigned int)layer->router_class;
    int normalize = layer->normalize_topk_probabilities;
    CUdeviceptr hash = 0ull, bias = 0ull;
    int rc = moe_cuda_matvec(execution, &job->weights[YVEX_MOE_WEIGHT_ROUTER],
                             execution->normalized, execution->logits, 0,
                             "cuda.moe.router-projection", err);
    if (rc != YVEX_OK) return rc;
    if (layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID) {
        unsigned long long host_selected[YVEX_MOE_MAX_SELECTED];
        unsigned long long rank;
        const unsigned char *row;
        if (!job->token_id_present || job->token_id >= layer->hash_table_rows ||
            aux->row_bytes < layer->experts_per_token * sizeof(int32_t))
            return moe_cuda_refuse(err, YVEX_ERR_BOUNDS, "CUDA hash router token ID is invalid");
        row = aux->encoded + (aux->row_count == 1ull
                                  ? 0ull
                                  : (unsigned long long)job->token_id * aux->row_bytes);
        for (rank = 0ull; rank < layer->experts_per_token; ++rank) {
            int32_t value;
            memcpy(&value, row + rank * sizeof(value), sizeof(value));
            host_selected[rank] = value < 0 ? ULLONG_MAX : (unsigned long long)value;
        }
        rc = execution->ops->initialize(&execution->work, execution->route_aux,
                                        (size_t)layer->experts_per_token * sizeof(*host_selected),
                                        host_selected, 0, "cuda.moe.hash-row",
                                        &execution->failure, err);
        hash = execution->route_aux;
        execution->h2d += layer->experts_per_token * sizeof(*host_selected);
    } else {
        if (aux->device_address) {
            bias = (CUdeviceptr)aux->device_address;
            execution->direct_weights++;
        } else {
            rc = execution->ops->initialize(&execution->work, execution->route_aux,
                                            aux->encoded_bytes, aux->encoded, 0,
                                            "cuda.moe.router-bias", &execution->failure, err);
            bias = execution->route_aux;
            execution->h2d += aux->encoded_bytes;
            execution->uploads++;
        }
    }
    if (rc == YVEX_OK) {
        void *params[] = {
            &execution->logits, &bias, &hash, &router_class,
            (void *)&layer->routed_experts, (void *)&layer->experts_per_token,
            &normalize, (void *)&layer->routed_scaling_factor, &execution->scores,
            &execution->selected, &execution->weights, &execution->status};
        rc = execution->ops->launch(&execution->work, execution->state->moe_route_function,
                                    1u, 1u, 0u, params, "cuda.moe.route",
                                    &execution->failure, err);
    }
    result->router.selected_count = layer->experts_per_token;
#define DOWNLOAD(target_, source_, bytes_, stage_)                                         \
    do {                                                                                   \
        if (rc == YVEX_OK)                                                                 \
            rc = execution->ops->download(&execution->work, (target_), (source_),          \
                                          (bytes_), (stage_), &execution->failure, err);    \
        if (rc == YVEX_OK) { execution->d2h += (bytes_); execution->downloads++; }          \
    } while (0)
    if (!job->device_output || job->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL) {
        DOWNLOAD(result->router.router_logits, execution->logits,
                 (size_t)layer->routed_experts * sizeof(float), "cuda.moe.logits-download");
        DOWNLOAD(result->router.router_scores, execution->scores,
                 (size_t)layer->routed_experts * sizeof(float), "cuda.moe.scores-download");
    }
    if (!execution->grouped_selected) {
        DOWNLOAD(result->router.selected_experts, execution->selected,
                 (size_t)layer->experts_per_token * sizeof(unsigned long long),
                 "cuda.moe.selected-download");
        DOWNLOAD(result->router.selected_weights, execution->weights,
                 (size_t)layer->experts_per_token * sizeof(float),
                 "cuda.moe.weights-download");
    }
#undef DOWNLOAD
    return rc == YVEX_OK && !execution->grouped_selected
               ? moe_cuda_sync_status(execution, "cuda.moe.route-sync", err) : rc;
}

int yvex_backend_moe_begin(yvex_backend_moe_execution **out, yvex_backend *backend,
                           const yvex_moe_layer_job *job,
                           yvex_moe_layer_result *result, yvex_error *err)
{
    yvex_backend_moe_execution *execution;
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    unsigned long long routed_subview, maximum;
    int rc;
    if (out) *out = NULL;
    if (!out || !backend || !job || !layer || !result || !job->expanded_input ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA ||
        !result->combined_output || result->combined_capacity < layer->hidden_width ||
        !result->post || result->post_capacity < layer->residual_streams ||
        !result->combination ||
        result->combination_capacity < layer->residual_streams * layer->residual_streams)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG, "CUDA MoE begin arguments are invalid");
    if ((job->device_input || job->device_output) &&
        (!backend_tensor_owner_is(backend, job->device_input) ||
         !backend_tensor_owner_is(backend, job->device_output) ||
         !backend_tensor_f32_elements(job->device_input, layer->expanded_width) ||
         !backend_tensor_f32_elements(job->device_output, layer->expanded_width)))
        return moe_cuda_refuse(err, YVEX_ERR_FORMAT,
                               "CUDA MoE device activation views are incompatible");
    execution = (yvex_backend_moe_execution *)calloc(1u, sizeof(*execution));
    if (!execution) return moe_cuda_refuse(err, YVEX_ERR_NOMEM, "CUDA MoE owner allocation failed");
    execution->backend = backend;
    execution->state = yvex_cuda_state(backend);
    execution->ops = yvex_cuda_attention_operations_get();
    execution->job = job;
    execution->started_ns = yvex_core_monotonic_ns();
    execution->work.backend = backend;
    execution->work.state = execution->state;
    execution->work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    execution->grouped_selected = job->device_output &&
        job->evidence_level != YVEX_ATTENTION_EVIDENCE_FULL &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE].device_address &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_UP].device_address &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_DOWN].device_address;
    if (!execution->state || !execution->state->moe_route_function ||
        !execution->state->q8_quantize_function ||
        !execution->state->moe_swiglu_function || !execution->state->moe_accumulate_function) {
        rc = moe_cuda_refuse(err, YVEX_ERR_UNSUPPORTED, "CUDA MoE kernel bundle is unavailable");
        goto fail;
    }
    routed_subview = job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE].encoded_bytes /
                     layer->routed_experts;
    maximum = job->weights[YVEX_MOE_WEIGHT_SHARED_GATE].encoded_bytes;
    if (job->weights[YVEX_MOE_WEIGHT_SHARED_UP].encoded_bytes > maximum)
        maximum = job->weights[YVEX_MOE_WEIGHT_SHARED_UP].encoded_bytes;
    if (job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN].encoded_bytes > maximum)
        maximum = job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN].encoded_bytes;
    if (routed_subview > maximum) maximum = routed_subview;
    execution->weight_buffer_bytes = (size_t)maximum;
    execution->route_aux_bytes = (size_t)layer->routed_experts * sizeof(float);
    if (execution->route_aux_bytes < (size_t)layer->experts_per_token * sizeof(unsigned long long))
        execution->route_aux_bytes = (size_t)layer->experts_per_token * sizeof(unsigned long long);
    backend_workspace_reset(backend);
    rc = moe_cuda_ranges(execution, err);
    if (rc == YVEX_OK) rc = moe_cuda_prepare_input(execution, err);
    if (rc == YVEX_OK) {
        unsigned long long routed_started = yvex_core_monotonic_ns();
        execution->ingress_ns = routed_started - execution->started_ns;
        rc = moe_cuda_route(execution, result, err);
        if (rc == YVEX_OK) execution->routing_ns = yvex_core_monotonic_ns() - routed_started;
    }
    if (rc == YVEX_OK && execution->grouped_selected)
        rc = moe_cuda_add_selected(execution, err);
    if (rc != YVEX_OK) goto fail;
    *out = execution;
    return YVEX_OK;
fail:
    (void)yvex_backend_moe_close(&execution, NULL);
    return rc;
}

static int moe_cuda_add_selected(yvex_backend_moe_execution *execution, yvex_error *err)
{
    const yvex_moe_layer_job *job = execution ? execution->job : NULL;
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    const yvex_moe_weight_view *gate, *up, *down;
    unsigned long long gate_expert_bytes, up_expert_bytes, down_expert_bytes;
    unsigned long long hidden_blocks, intermediate_blocks, quantize_tasks;
    unsigned long long started;
    int q8_input;
    int rc = YVEX_OK;
    if (!execution || !layer || execution->finished || !execution->grouped_selected ||
        !execution->state->moe_grouped_up_function ||
        !execution->state->moe_grouped_down_function)
        return moe_cuda_refuse(err, YVEX_ERR_UNSUPPORTED,
                               "CUDA grouped selected-expert execution is unavailable");
    gate = &job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE];
    up = &job->weights[YVEX_MOE_WEIGHT_ROUTED_UP];
    down = &job->weights[YVEX_MOE_WEIGHT_ROUTED_DOWN];
    if (!gate->row_bytes || !up->row_bytes || !down->row_bytes ||
        gate->row_count != layer->routed_experts * layer->expert_intermediate_width ||
        up->row_count != gate->row_count ||
        down->row_count != layer->routed_experts * layer->hidden_width ||
        gate->row_width != layer->hidden_width || up->row_width != layer->hidden_width ||
        down->row_width != layer->expert_intermediate_width)
        return moe_cuda_refuse(err, YVEX_ERR_FORMAT,
                               "CUDA grouped selected-expert geometry is incompatible");
    gate_expert_bytes = gate->row_bytes * layer->expert_intermediate_width;
    up_expert_bytes = up->row_bytes * layer->expert_intermediate_width;
    down_expert_bytes = down->row_bytes * layer->hidden_width;
    q8_input = layer->hidden_width % 256ull == 0ull &&
               layer->expert_intermediate_width % 256ull == 0ull &&
               moe_cuda_q8_eligible(gate->qtype) && moe_cuda_q8_eligible(up->qtype) &&
               moe_cuda_q8_eligible(down->qtype);
    hidden_blocks = q8_input ? layer->hidden_width / 256ull : layer->hidden_width;
    intermediate_blocks = q8_input ? layer->expert_intermediate_width / 256ull :
                          layer->expert_intermediate_width;
    quantize_tasks = intermediate_blocks * layer->experts_per_token;
    if (q8_input && (hidden_blocks > UINT_MAX || quantize_tasks > UINT_MAX))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA grouped Q8 activation grid exceeds launch bounds");
    started = yvex_core_monotonic_ns();
    if (q8_input) {
        unsigned long long one = 1ull;
        void *params[] = {&execution->gate, &execution->normalized,
                          (void *)&layer->hidden_width, &one, &execution->status};
        rc = execution->ops->launch(&execution->work,
            execution->state->q8_quantize_function, (unsigned int)hidden_blocks,
            MOE_CUDA_BLOCK, 0u, params, "cuda.moe.input-q8",
            &execution->failure, err);
    }
    if (rc == YVEX_OK) {
        CUdeviceptr gate_address = (CUdeviceptr)gate->device_address;
        CUdeviceptr up_address = (CUdeviceptr)up->device_address;
        unsigned int gate_qtype = gate->qtype, up_qtype = up->qtype;
        unsigned long long rows = layer->experts_per_token *
                                  layer->expert_intermediate_width;
        unsigned long long blocks = (rows + MOE_CUDA_ROWS_PER_BLOCK - 1ull) /
                                    MOE_CUDA_ROWS_PER_BLOCK;
        void *params[] = {&gate_address, (void *)&gate->row_bytes, &gate_expert_bytes,
                          &gate_qtype, &up_address, (void *)&up->row_bytes, &up_expert_bytes,
                          &up_qtype, &execution->selected,
                          (void *)&layer->experts_per_token, (void *)&layer->routed_experts,
                          q8_input ? &execution->gate : &execution->normalized,
                          &hidden_blocks, &q8_input,
                          (void *)&layer->expert_intermediate_width,
                          (void *)&layer->activation_limit, &execution->intermediate,
                          &execution->status};
        if (blocks > UINT_MAX)
            return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                                   "CUDA grouped gate/up grid exceeds launch bounds");
        rc = execution->ops->launch(&execution->work,
            execution->state->moe_grouped_up_function, (unsigned int)blocks,
            MOE_CUDA_BLOCK, 0u, params,
            "cuda.moe.grouped-up", &execution->failure, err);
    }
    if (rc == YVEX_OK && q8_input) {
        void *params[] = {&execution->up, &execution->intermediate,
                          (void *)&layer->expert_intermediate_width,
                          (void *)&layer->experts_per_token, &execution->status};
        rc = execution->ops->launch(&execution->work,
            execution->state->q8_quantize_function, (unsigned int)quantize_tasks,
            MOE_CUDA_BLOCK, 0u, params, "cuda.moe.intermediate-q8",
            &execution->failure, err);
    }
    if (rc == YVEX_OK) {
        CUdeviceptr down_address = (CUdeviceptr)down->device_address;
        unsigned int qtype = down->qtype;
        void *params[] = {&down_address, (void *)&down->row_bytes, &down_expert_bytes,
                          &qtype, &execution->selected, &execution->weights,
                          (void *)&layer->experts_per_token, (void *)&layer->routed_experts,
                          q8_input ? &execution->up : &execution->intermediate,
                          &intermediate_blocks, &q8_input,
                          (void *)&layer->hidden_width, &execution->routed,
                          &execution->status};
        unsigned long long blocks = (layer->hidden_width + MOE_CUDA_ROWS_PER_BLOCK - 1ull) /
                                    MOE_CUDA_ROWS_PER_BLOCK;
        if (blocks > UINT_MAX)
            return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                                   "CUDA grouped down grid exceeds launch bounds");
        rc = execution->ops->launch(&execution->work,
            execution->state->moe_grouped_down_function,
            (unsigned int)blocks, MOE_CUDA_BLOCK, 0u, params, "cuda.moe.grouped-down",
            &execution->failure, err);
    }
    if (rc == YVEX_OK) {
        execution->direct_weights += 3ull;
        execution->subviews += layer->experts_per_token * 3ull;
        execution->routed_ns += yvex_core_monotonic_ns() - started;
    }
    return rc;
}

int yvex_backend_moe_add_expert(yvex_backend_moe_execution *execution,
                                const yvex_moe_weight_view *gate,
                                const yvex_moe_weight_view *up,
                                const yvex_moe_weight_view *down, float route_weight,
                                int shared, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = execution && execution->job
                                           ? execution->job->layer : NULL;
    unsigned long long width;
    unsigned int grid;
    unsigned long long started = yvex_core_monotonic_ns();
    int rc;
    if (!execution || !layer || !gate || !up || !down || execution->finished ||
        gate->row_count != up->row_count || down->row_width != gate->row_count ||
        down->row_count != layer->hidden_width)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG, "CUDA MoE expert geometry is invalid");
    width = gate->row_count;
    rc = moe_cuda_matvec(execution, gate, execution->normalized, execution->gate, 0,
                         "cuda.moe.expert-gate", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_matvec(execution, up, execution->normalized, execution->up, 0,
                             "cuda.moe.expert-up", err);
    grid = (unsigned int)((width + MOE_CUDA_BLOCK - 1ull) / MOE_CUDA_BLOCK);
    if (rc == YVEX_OK) {
        void *params[] = {&execution->gate, &execution->up, &width,
                          (void *)&layer->activation_limit, &execution->intermediate,
                          &execution->status};
        rc = execution->ops->launch(&execution->work, execution->state->moe_swiglu_function,
                                    grid, MOE_CUDA_BLOCK, 0u, params, "cuda.moe.swiglu",
                                    &execution->failure, err);
    }
    if (rc == YVEX_OK)
        rc = moe_cuda_matvec(execution, down, execution->intermediate,
                             execution->expert, 1, "cuda.moe.expert-down", err);
    grid = (unsigned int)((layer->hidden_width + MOE_CUDA_BLOCK - 1ull) / MOE_CUDA_BLOCK);
    if (rc == YVEX_OK) {
        float weight = shared ? 1.0f : route_weight;
        CUdeviceptr aggregate = shared ? execution->shared : execution->routed;
        void *params[] = {&execution->expert, (void *)&layer->hidden_width, &weight,
                          &aggregate, &execution->status};
        rc = execution->ops->launch(&execution->work, execution->state->moe_accumulate_function,
                                    grid, MOE_CUDA_BLOCK, 0u, params, "cuda.moe.accumulate",
                                    &execution->failure, err);
    }
    if (rc == YVEX_OK) {
        execution->subviews += shared ? 0ull : 3ull;
        if (shared) execution->shared_ns += yvex_core_monotonic_ns() - started;
        else execution->routed_ns += yvex_core_monotonic_ns() - started;
    }
    return rc;
}
/*
 * Publish one complete CUDA MoE output.
 *
 * Backend transaction.
 */
int yvex_backend_moe_finish(yvex_backend_moe_execution *execution,
                            yvex_moe_layer_result *result, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = execution && execution->job
                                           ? execution->job->layer : NULL;
    unsigned long long activation_elements, activation_bytes;
    int rc;
    if (!execution || !layer || !result || execution->finished)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG, "CUDA MoE finish is invalid");
    if (!result->routed_output || result->routed_capacity < layer->hidden_width ||
        !result->shared_output || result->shared_capacity < layer->hidden_width)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                               "CUDA MoE routed/shared publication capacity is invalid");
    {
        float weight = 1.0f;
        unsigned int grid = (unsigned int)((layer->hidden_width + MOE_CUDA_BLOCK - 1ull) /
                                            MOE_CUDA_BLOCK);
        void *routed_params[] = {&execution->routed, (void *)&layer->hidden_width, &weight,
                                 &execution->combined, &execution->status};
        void *shared_params[] = {&execution->shared, (void *)&layer->hidden_width, &weight,
                                 &execution->combined, &execution->status};
        rc = execution->ops->launch(&execution->work, execution->state->moe_accumulate_function,
                                    grid, MOE_CUDA_BLOCK, 0u, routed_params,
                                    "cuda.moe.combine-routed", &execution->failure, err);
        if (rc == YVEX_OK)
            rc = execution->ops->launch(&execution->work, execution->state->moe_accumulate_function,
                                        grid, MOE_CUDA_BLOCK, 0u, shared_params,
                                        "cuda.moe.combine-shared", &execution->failure, err);
    }
    if (rc == YVEX_OK)
        rc = execution->ops->round_bf16(&execution->work, execution->combined,
                                    layer->hidden_width, execution->status,
                                    "cuda.moe.output-round", &execution->failure, err);
    if (rc == YVEX_OK && execution->job->device_output) {
        unsigned long long streams = layer->residual_streams, width = layer->hidden_width;
        unsigned long long one = 1ull;
        unsigned long long count = streams * width;
        unsigned int grid = (unsigned int)((count + MOE_CUDA_BLOCK - 1ull) / MOE_CUDA_BLOCK);
        CUdeviceptr output = (CUdeviceptr)execution->job->device_output->data;
        void *params[] = {&execution->combined, &execution->expanded, &execution->post,
                          &execution->combination, &streams, &width, &output, &one,
                          &execution->status};
        rc = execution->ops->launch(&execution->work,
            execution->state->deepseek_mhc_post_function, grid, MOE_CUDA_BLOCK, 0u,
            params, "cuda.moe.deferred-post", &execution->failure, err);
    }
#define DOWNLOAD(target_, source_, bytes_, stage_)                                         \
    do {                                                                                   \
        if (rc == YVEX_OK)                                                                 \
            rc = execution->ops->download(&execution->work, (target_), (source_),          \
                                          (bytes_), (stage_), &execution->failure, err);    \
        if (rc == YVEX_OK) execution->d2h += (bytes_);                                     \
    } while (0)
    if (execution->grouped_selected) {
        DOWNLOAD(result->router.selected_experts, execution->selected,
                 (size_t)layer->experts_per_token * sizeof(unsigned long long),
                 "cuda.moe.selected-publication");
        DOWNLOAD(result->router.selected_weights, execution->weights,
                 (size_t)layer->experts_per_token * sizeof(float),
                 "cuda.moe.weights-publication");
    }
    if (!execution->job->device_output ||
        execution->job->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL) {
        DOWNLOAD(result->combined_output, execution->combined,
                 (size_t)layer->hidden_width * sizeof(float), "cuda.moe.output-download");
        DOWNLOAD(result->routed_output, execution->routed,
                 (size_t)layer->hidden_width * sizeof(float), "cuda.moe.routed-download");
        DOWNLOAD(result->shared_output, execution->shared,
                 (size_t)layer->hidden_width * sizeof(float), "cuda.moe.shared-download");
        DOWNLOAD(result->post, execution->post,
                 (size_t)layer->residual_streams * sizeof(float), "cuda.moe.post-download");
        DOWNLOAD(result->combination, execution->combination,
                 (size_t)layer->residual_streams * layer->residual_streams * sizeof(float),
                 "cuda.moe.combination-download");
    }
#undef DOWNLOAD
    if (rc == YVEX_OK) rc = moe_cuda_sync_status(execution, "cuda.moe.finish-sync", err);
    if (rc != YVEX_OK) return rc;
    if (execution->job->device_input && execution->job->device_output) {
        if (!yvex_core_u64_mul(layer->expanded_width, 2ull, &activation_elements) ||
            !yvex_core_u64_mul(activation_elements, sizeof(float), &activation_bytes))
            return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                                   "CUDA MoE compulsory activation extent overflowed");
        result->memory.activation_bytes = activation_bytes;
        result->memory.temporary_bytes = execution->work.peak_bytes;
    }
    result->expert_subviews_accessed += execution->subviews;
    result->host_to_device_bytes += execution->h2d;
    result->device_to_host_bytes += execution->d2h;
    result->kernel_launches += execution->work.launches;
    result->upload_count += execution->uploads;
    result->download_count += execution->downloads;
    result->cache_hits += execution->direct_weights;
    result->cache_misses += execution->uploads;
    result->device_to_device_bytes += execution->d2d;
    result->device_synchronizations += execution->device_synchronizations;
    result->synchronization_ns += execution->synchronization_ns;
    result->ingress_ns += execution->ingress_ns;
    result->routing_ns += execution->routing_ns;
    result->routed_ns += execution->routed_ns;
    result->shared_ns += execution->shared_ns;
    result->total_ns += yvex_core_monotonic_ns() - execution->started_ns;
    execution->finished = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Release one CUDA MoE transaction.
 *
 * Retains retryable ownership.
 */
int yvex_backend_moe_close(yvex_backend_moe_execution **execution, yvex_error *err)
{
    int rc;
    if (!execution || !*execution) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    rc = yvex_cuda_work_cleanup(&(*execution)->work, err);
    if (rc == YVEX_OK) {
        free(*execution);
        *execution = NULL;
        yvex_error_clear(err);
    }
    return rc;
}

static int moe_cuda_batch_allocate(moe_cuda_batch *batch, CUdeviceptr *out,
                                   unsigned long long count, size_t width,
                                   int zero, const char *stage, yvex_error *err)
{
    unsigned long long bytes;
    if (!batch || !out || !count || !width ||
        !yvex_core_u64_mul(count, (unsigned long long)width, &bytes) || bytes > SIZE_MAX)
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE range overflowed");
    return batch->ops->allocate(&batch->work, out, (size_t)bytes, NULL, zero,
                                stage, &batch->failure, err);
}

static int moe_cuda_batch_matvec(moe_cuda_batch *batch,
                                 const yvex_moe_weight_view *weight,
                                 unsigned long long input_rows,
                                 CUdeviceptr input, CUdeviceptr output,
                                 int output_bf16, const char *stage,
                                 yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    if (!weight || !weight->device_address)
        return moe_cuda_refuse(err, YVEX_ERR_STATE,
                               "CUDA width-N MoE requires resident weights");
    return batch->ops->matvec(
        &batch->work, &encoded, (CUdeviceptr)weight->device_address, 0ull,
        weight->row_count, input_rows, input, output, output_bf16,
        batch->status, stage, &batch->failure, err);
}

static int moe_cuda_batch_decode(moe_cuda_batch *batch,
                                 const yvex_moe_weight_view *weight,
                                 CUdeviceptr output, const char *stage,
                                 yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    if (!weight || !weight->device_address)
        return moe_cuda_refuse(err, YVEX_ERR_STATE,
                               "CUDA width-N MoE requires resident weights");
    return batch->ops->decode(
        &batch->work, &encoded, (CUdeviceptr)weight->device_address, 0ull,
        weight->row_width, output, batch->status, stage, &batch->failure, err);
}

static int moe_cuda_rows_geometry(const yvex_moe_layer_job *job,
                                  const yvex_moe_row_batch *rows,
                                  const yvex_moe_row_batch_output *output,
                                  unsigned long long *pairs, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    unsigned long long expanded, hidden, post, combination;
    unsigned long long routed_intermediate_rows, routed_down_rows;
    unsigned long long slot;
    if (pairs) *pairs = 0ull;
    if (!job || !layer || !rows || !output || !pairs ||
        rows->schema_version != YVEX_MOE_ROW_BATCH_SCHEMA_V1 || !rows->row_count ||
        rows->row_stride != rows->row_width || rows->row_width != layer->expanded_width ||
        !rows->device_rows || !rows->device_outputs || !rows->token_ids ||
        !rows->token_ids_present || rows->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE ||
        job->evidence_level == YVEX_ATTENTION_EVIDENCE_FULL || layer->shared_experts != 1ull ||
        !yvex_core_u64_mul(rows->row_count, layer->experts_per_token, pairs) ||
        !yvex_core_u64_mul(rows->row_count, layer->expanded_width, &expanded) ||
        !yvex_core_u64_mul(rows->row_count, layer->hidden_width, &hidden) ||
        !yvex_core_u64_mul(rows->row_count, layer->residual_streams, &post) ||
        !yvex_core_u64_mul(post, layer->residual_streams, &combination) ||
        !yvex_core_u64_mul(layer->routed_experts,
                           layer->expert_intermediate_width,
                           &routed_intermediate_rows) ||
        !yvex_core_u64_mul(layer->routed_experts, layer->hidden_width,
                           &routed_down_rows) ||
        !backend_tensor_f32_elements(rows->device_rows, expanded) ||
        !backend_tensor_f32_elements(rows->device_outputs, expanded) ||
        ((job->device_completion &&
          (!job->device_completion->defer ||
           !job->device_completion->host_status ||
           !job->device_completion->host_unique_experts)) ||
         (!job->device_completion &&
          (!output->selected_experts || !output->selected_weights ||
           output->selection_capacity < *pairs))) ||
        !output->combined_rows ||
        output->combined_capacity < hidden || !output->routed_rows ||
        output->routed_capacity < hidden || !output->shared_rows ||
        output->shared_capacity < hidden || !output->post_rows ||
        output->post_capacity < post || !output->combination_rows ||
        output->combination_capacity < combination)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                               "CUDA width-N MoE job geometry is incompatible");
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
        if (layer->tensor_ids[slot] != YVEX_MOE_NO_TENSOR &&
            !job->weights[slot].device_address)
            return moe_cuda_refuse(err, YVEX_ERR_STATE,
                                   "CUDA width-N MoE weight residency is incomplete");
    if (job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE].row_count != routed_intermediate_rows ||
        job->weights[YVEX_MOE_WEIGHT_ROUTED_UP].row_count != routed_intermediate_rows ||
        job->weights[YVEX_MOE_WEIGHT_ROUTED_DOWN].row_count != routed_down_rows ||
        job->weights[YVEX_MOE_WEIGHT_SHARED_GATE].row_count !=
            layer->shared_intermediate_width ||
        job->weights[YVEX_MOE_WEIGHT_SHARED_UP].row_count !=
            layer->shared_intermediate_width ||
        job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN].row_count != layer->hidden_width)
        return moe_cuda_refuse(err, YVEX_ERR_FORMAT,
                               "CUDA width-N MoE expert packs are incompatible");
    return YVEX_OK;
}

static int moe_cuda_batch_ranges(moe_cuda_batch *batch,
                                 const yvex_moe_layer_job *job,
                                 const yvex_moe_row_batch *rows,
                                 unsigned long long pairs,
                                 int routed_q8, int shared_q8,
                                 yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    unsigned long long expanded, hidden, post, combination, mix, logits;
    unsigned long long routed_intermediate, shared_intermediate, pair_outputs;
    unsigned long long maximum_input = 0ull, input_q8 = 0ull;
    unsigned long long routed_q8_count = 0ull, shared_q8_count = 0ull;
    int rc;
    if (moe_cuda_q8_eligible(job->weights[YVEX_MOE_WEIGHT_MHC_FUNCTION].qtype) &&
        layer->expanded_width % 256ull == 0ull)
        maximum_input = layer->expanded_width;
    if ((routed_q8 || shared_q8 ||
         (moe_cuda_q8_eligible(job->weights[YVEX_MOE_WEIGHT_ROUTER].qtype) &&
          layer->hidden_width % 256ull == 0ull)) &&
        layer->hidden_width > maximum_input)
        maximum_input = layer->hidden_width;
    if (maximum_input &&
        !yvex_core_u64_mul(rows->row_count, maximum_input / 256ull, &input_q8))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE Q8 input range overflowed");
    if (!yvex_core_u64_mul(rows->row_count, layer->expanded_width, &expanded) ||
        !yvex_core_u64_mul(rows->row_count, layer->hidden_width, &hidden) ||
        !yvex_core_u64_mul(rows->row_count, layer->residual_streams, &post) ||
        !yvex_core_u64_mul(post, layer->residual_streams, &combination) ||
        !yvex_core_u64_mul(rows->row_count, layer->mhc_mixing_rows, &mix) ||
        !yvex_core_u64_mul(rows->row_count, layer->routed_experts, &logits) ||
        !yvex_core_u64_mul(pairs, layer->expert_intermediate_width,
                           &routed_intermediate) ||
        !yvex_core_u64_mul(rows->row_count, layer->shared_intermediate_width,
                           &shared_intermediate) ||
        !yvex_core_u64_mul(pairs, layer->hidden_width, &pair_outputs) ||
        (routed_q8 && !yvex_core_u64_mul(
            pairs, layer->expert_intermediate_width / 256ull,
            &routed_q8_count)) ||
        (shared_q8 && !yvex_core_u64_mul(
            rows->row_count, layer->shared_intermediate_width / 256ull,
            &shared_q8_count)))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE range geometry overflowed");
#define RANGE(member_, count_, width_, zero_, stage_)                                      \
    do {                                                                                   \
        if (rc == YVEX_OK)                                                                 \
            rc = moe_cuda_batch_allocate(batch, &batch->member_, (count_), (width_),       \
                                         (zero_), (stage_), err);                          \
    } while (0)
    rc = moe_cuda_batch_allocate(batch, &batch->status, 1ull, sizeof(int), 1,
                                 "cuda.moe.rows.status", err);
    RANGE(expanded, expanded, sizeof(float), 0, "cuda.moe.rows.expanded");
    RANGE(normalized, hidden, sizeof(float), 1, "cuda.moe.rows.normalized");
    RANGE(post, post, sizeof(float), 1, "cuda.moe.rows.post");
    RANGE(combination, combination, sizeof(float), 1, "cuda.moe.rows.combination");
    RANGE(mix, mix, sizeof(float), 1, "cuda.moe.rows.mix");
    RANGE(scale, 3ull, sizeof(float), 1, "cuda.moe.rows.scale");
    RANGE(base, layer->mhc_mixing_rows, sizeof(float), 1, "cuda.moe.rows.base");
    RANGE(logits, logits, sizeof(float), 1, "cuda.moe.rows.logits");
    RANGE(scores, logits, sizeof(float), 1, "cuda.moe.rows.scores");
    RANGE(selected, pairs, sizeof(unsigned long long), 1, "cuda.moe.rows.selected");
    RANGE(weights, pairs, sizeof(float), 1, "cuda.moe.rows.weights");
    RANGE(tokens, rows->row_count, sizeof(unsigned int), 0, "cuda.moe.rows.tokens");
    RANGE(order, pairs, sizeof(unsigned long long), 1, "cuda.moe.rows.order");
    RANGE(unique, 1ull, sizeof(unsigned long long), 1, "cuda.moe.rows.unique");
    if (rc == YVEX_OK && input_q8)
        RANGE(input_q8, input_q8, MOE_CUDA_Q8_BYTES, 1, "cuda.moe.rows.input-q8");
    RANGE(routed_intermediate, routed_intermediate, sizeof(float), 1,
          "cuda.moe.rows.routed-intermediate");
    if (rc == YVEX_OK && routed_q8)
        RANGE(routed_intermediate_q8, routed_q8_count,
              MOE_CUDA_Q8_BYTES, 1, "cuda.moe.rows.routed-q8");
    RANGE(routed_pairs, pair_outputs, sizeof(float), 1, "cuda.moe.rows.routed-pairs");
    RANGE(routed, hidden, sizeof(float), 1, "cuda.moe.rows.routed");
    RANGE(shared_intermediate, shared_intermediate, sizeof(float), 1,
          "cuda.moe.rows.shared-intermediate");
    if (rc == YVEX_OK && shared_q8)
        RANGE(shared_intermediate_q8, shared_q8_count,
              MOE_CUDA_Q8_BYTES, 1, "cuda.moe.rows.shared-q8");
    RANGE(shared_pairs, hidden, sizeof(float), 1, "cuda.moe.rows.shared-pairs");
    RANGE(shared, hidden, sizeof(float), 1, "cuda.moe.rows.shared");
    RANGE(combined, hidden, sizeof(float), 1, "cuda.moe.rows.combined");
#undef RANGE
    if (rc == YVEX_OK && input_q8) {
        batch->work.q8_input = batch->input_q8;
        batch->work.q8_capacity = input_q8 * MOE_CUDA_Q8_BYTES;
    }
    return rc;
}

static int moe_cuda_batch_copy_input(moe_cuda_batch *batch,
                                     const yvex_moe_row_batch *rows,
                                     unsigned long long bytes,
                                     yvex_error *err)
{
    CUstream stream = yvex_cuda_launch_stream(batch->backend);
    CUresult copied = stream && batch->state->driver.cuMemcpyDtoDAsync_v2
                          ? batch->state->driver.cuMemcpyDtoDAsync_v2(
                                batch->expanded, (CUdeviceptr)rows->device_rows->data,
                                (size_t)bytes, stream)
                          : !stream ? batch->state->driver.cuMemcpyDtoD_v2(
                                batch->expanded, (CUdeviceptr)rows->device_rows->data,
                                (size_t)bytes) : (CUresult)1;
    int rc = yvex_cuda_status(&batch->state->driver, copied,
                              "cuda.moe.rows.input-copy", err);
    if (rc == YVEX_OK) batch->d2d += bytes;
    return rc;
}

static int moe_cuda_batch_prepare(moe_cuda_batch *batch,
                                  const yvex_moe_layer_job *job,
                                  const yvex_moe_row_batch *rows,
                                  yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    yvex_backend_attention_weight norm =
        moe_cuda_weight(&job->weights[YVEX_MOE_WEIGHT_FFN_NORM]);
    unsigned long long streams = layer->residual_streams, width = layer->hidden_width;
    unsigned long long expanded_bytes;
    unsigned int shared_bytes = 0u;
    int rc;
    if (!yvex_core_u64_mul(rows->row_count, layer->expanded_width, &expanded_bytes) ||
        !yvex_core_u64_mul(expanded_bytes, sizeof(float), &expanded_bytes) ||
        expanded_bytes > SIZE_MAX)
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE input bytes overflowed");
    rc = moe_cuda_mhc_shared_bytes(streams, &shared_bytes, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_copy_input(batch, rows, expanded_bytes, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_matvec(batch, &job->weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                                   rows->row_count, batch->expanded, batch->mix, 0,
                                   "cuda.moe.rows.mhc-function", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_decode(batch, &job->weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                                   batch->scale, "cuda.moe.rows.mhc-scale", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_decode(batch, &job->weights[YVEX_MOE_WEIGHT_MHC_BASE],
                                   batch->base, "cuda.moe.rows.mhc-base", err);
    if (rc == YVEX_OK) {
        void *params[] = {
            &batch->expanded, &batch->mix, &batch->scale, &batch->base,
            &streams, &width, (void *)&layer->mhc_mixing_rows,
            (void *)&layer->mhc_sinkhorn_iterations, (void *)&layer->rms_epsilon,
            (void *)&layer->mhc_epsilon, (void *)&layer->mhc_post_multiplier,
            &batch->normalized, &batch->post, &batch->combination,
            (void *)&rows->row_count, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->deepseek_mhc_pre_function,
            (unsigned int)rows->row_count, MOE_CUDA_BLOCK, shared_bytes, params,
            "cuda.moe.rows.mhc-pre", &batch->failure, err);
    }
    if (rc == YVEX_OK)
        rc = batch->ops->weighted_norm(
            &batch->work, batch->normalized, layer->hidden_width, rows->row_count,
            &norm, (CUdeviceptr)job->weights[YVEX_MOE_WEIGHT_FFN_NORM].device_address,
            layer->rms_epsilon, batch->status, "cuda.moe.rows.ffn-norm",
            &batch->failure, err);
    return rc;
}

static int moe_cuda_batch_route(moe_cuda_batch *batch,
                                const yvex_moe_layer_job *job,
                                const yvex_moe_row_batch *rows,
                                unsigned long long pairs,
                                yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    const yvex_moe_weight_view *aux = &job->weights[
        layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID
            ? YVEX_MOE_WEIGHT_ROUTER_TABLE : YVEX_MOE_WEIGHT_ROUTER_BIAS];
    CUdeviceptr hash = layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID
                           ? (CUdeviceptr)aux->device_address : 0ull;
    CUdeviceptr bias = layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE
                           ? (CUdeviceptr)aux->device_address : 0ull;
    unsigned int router_class = (unsigned int)layer->router_class;
    unsigned long long token_bytes;
    int normalize = layer->normalize_topk_probabilities;
    int rc = moe_cuda_batch_matvec(
        batch, &job->weights[YVEX_MOE_WEIGHT_ROUTER], rows->row_count,
        batch->normalized, batch->logits, 0, "cuda.moe.rows.router-projection", err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_mul(rows->row_count, sizeof(*rows->token_ids), &token_bytes) ||
         token_bytes > SIZE_MAX))
        rc = moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                             "CUDA width-N MoE token upload overflowed");
    if (rc == YVEX_OK) {
        rc = batch->ops->initialize(
            &batch->work, batch->tokens, (size_t)token_bytes, rows->token_ids, 0,
            "cuda.moe.rows.token-ids", &batch->failure, err);
        if (rc == YVEX_OK) batch->h2d += token_bytes;
    }
    if (rc == YVEX_OK) {
        void *params[] = {
            &batch->logits, &bias, &hash, &batch->tokens, &router_class,
            (void *)&rows->row_count, (void *)&layer->routed_experts,
            (void *)&layer->experts_per_token, (void *)&layer->hash_table_rows,
            (void *)&layer->hash_table_columns, (void *)&aux->row_bytes,
            &normalize, (void *)&layer->routed_scaling_factor, &batch->scores,
            &batch->selected, &batch->weights, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_route_rows_function,
            (unsigned int)rows->row_count, MOE_CUDA_BLOCK, 0u, params,
            "cuda.moe.rows.route", &batch->failure, err);
    }
    if (rc == YVEX_OK) {
        void *params[] = {
            &batch->selected, (void *)&rows->row_count,
            (void *)&layer->experts_per_token, (void *)&layer->routed_experts,
            &batch->order, &batch->unique, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_pair_order_function,
            1u, MOE_CUDA_BLOCK, 0u,
            params, "cuda.moe.rows.pair-order", &batch->failure, err);
    }
    (void)pairs;
    return rc;
}

static int moe_cuda_batch_quantize(moe_cuda_batch *batch, CUdeviceptr output,
                                   CUdeviceptr input, unsigned long long width,
                                   unsigned long long rows, const char *stage,
                                   yvex_error *err)
{
    unsigned long long blocks = width / 256ull, tasks;
    if (!blocks || !yvex_core_u64_mul(blocks, rows, &tasks) || tasks > UINT_MAX)
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE Q8 geometry is invalid");
    {
        void *params[] = {&output, &input, &width, &rows, &batch->status};
        return batch->ops->launch(
            &batch->work, batch->state->q8_quantize_function,
            (unsigned int)tasks, MOE_CUDA_BLOCK, 0u, params, stage,
            &batch->failure, err);
    }
}

static int moe_cuda_batch_experts(moe_cuda_batch *batch,
                                  const yvex_moe_layer_job *job,
                                  const yvex_moe_row_batch *rows,
                                  unsigned long long pairs, int routed,
                                  int q8, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    unsigned int base = routed ? YVEX_MOE_WEIGHT_ROUTED_GATE
                               : YVEX_MOE_WEIGHT_SHARED_GATE;
    const yvex_moe_weight_view *gate = &job->weights[base];
    const yvex_moe_weight_view *up = &job->weights[base + 1u];
    const yvex_moe_weight_view *down = &job->weights[base + 2u];
    unsigned long long count = routed ? pairs : rows->row_count;
    unsigned long long topk = routed ? layer->experts_per_token : 1ull;
    unsigned long long experts = routed ? layer->routed_experts : layer->shared_experts;
    unsigned long long intermediate_width = routed ? layer->expert_intermediate_width
                                                    : layer->shared_intermediate_width;
    unsigned long long gate_expert_bytes, up_expert_bytes, down_expert_bytes;
    unsigned long long input_extent = q8 ? layer->hidden_width / 256ull
                                         : layer->hidden_width;
    unsigned long long intermediate_extent = q8 ? intermediate_width / 256ull
                                                : intermediate_width;
    unsigned long long up_tasks, down_tasks, reduce_tasks;
    unsigned int up_grid, down_grid, reduce_grid;
    CUdeviceptr selected = routed ? batch->selected : 0ull;
    CUdeviceptr weights = routed ? batch->weights : 0ull;
    CUdeviceptr order = routed ? batch->order : 0ull;
    CUdeviceptr intermediate = routed ? batch->routed_intermediate
                                      : batch->shared_intermediate;
    CUdeviceptr intermediate_q8 = routed ? batch->routed_intermediate_q8
                                         : batch->shared_intermediate_q8;
    CUdeviceptr pair_outputs = routed ? batch->routed_pairs : batch->shared_pairs;
    CUdeviceptr aggregate = routed ? batch->routed : batch->shared;
    int rc;
    if (!yvex_core_u64_mul(gate->row_bytes, intermediate_width, &gate_expert_bytes) ||
        !yvex_core_u64_mul(up->row_bytes, intermediate_width, &up_expert_bytes) ||
        !yvex_core_u64_mul(down->row_bytes, layer->hidden_width, &down_expert_bytes) ||
        !yvex_core_u64_mul(count, intermediate_width, &up_tasks) ||
        !yvex_core_u64_mul(count, layer->hidden_width, &down_tasks) ||
        !yvex_core_u64_mul(rows->row_count, layer->hidden_width, &reduce_tasks) ||
        !moe_cuda_grid(up_tasks, MOE_CUDA_ROWS_PER_BLOCK, &up_grid) ||
        !moe_cuda_grid(down_tasks, MOE_CUDA_ROWS_PER_BLOCK, &down_grid) ||
        !moe_cuda_grid(reduce_tasks, MOE_CUDA_BLOCK, &reduce_grid))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE expert grid exceeds launch bounds");
    if (q8)
        rc = moe_cuda_batch_quantize(batch, batch->input_q8, batch->normalized,
                                     layer->hidden_width, rows->row_count,
                                     routed ? "cuda.moe.rows.routed-input-q8"
                                            : "cuda.moe.rows.shared-input-q8", err);
    else rc = YVEX_OK;
    if (rc == YVEX_OK) {
        CUdeviceptr gate_address = (CUdeviceptr)gate->device_address;
        CUdeviceptr up_address = (CUdeviceptr)up->device_address;
        unsigned int gate_qtype = gate->qtype, up_qtype = up->qtype;
        void *params[] = {
            &gate_address, (void *)&gate->row_bytes, &gate_expert_bytes, &gate_qtype,
            &up_address, (void *)&up->row_bytes, &up_expert_bytes, &up_qtype,
            &selected, &order, &count, &topk, &experts,
            q8 ? &batch->input_q8 : &batch->normalized, &input_extent, &q8,
            &intermediate_width, (void *)&layer->activation_limit,
            &intermediate, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_grouped_up_rows_function,
            up_grid, MOE_CUDA_BLOCK, 0u, params,
            routed ? "cuda.moe.rows.routed-up" : "cuda.moe.rows.shared-up",
            &batch->failure, err);
    }
    if (rc == YVEX_OK && q8)
        rc = moe_cuda_batch_quantize(
            batch, intermediate_q8, intermediate, intermediate_width, count,
            routed ? "cuda.moe.rows.routed-intermediate-q8"
                   : "cuda.moe.rows.shared-intermediate-q8", err);
    if (rc == YVEX_OK) {
        CUdeviceptr down_address = (CUdeviceptr)down->device_address;
        unsigned int qtype = down->qtype;
        void *params[] = {
            &down_address, (void *)&down->row_bytes, &down_expert_bytes, &qtype,
            &selected, &weights, &order, &count, &topk, &experts,
            q8 ? &intermediate_q8 : &intermediate, &intermediate_extent, &q8,
            (void *)&layer->hidden_width, &pair_outputs, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_grouped_down_rows_function,
            down_grid, MOE_CUDA_BLOCK, 0u, params,
            routed ? "cuda.moe.rows.routed-down" : "cuda.moe.rows.shared-down",
            &batch->failure, err);
    }
    if (rc == YVEX_OK) {
        void *params[] = {
            &pair_outputs, (void *)&rows->row_count, &topk,
            (void *)&layer->hidden_width, &aggregate, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_reduce_rows_function,
            reduce_grid, MOE_CUDA_BLOCK, 0u, params,
            routed ? "cuda.moe.rows.routed-reduce" : "cuda.moe.rows.shared-reduce",
            &batch->failure, err);
    }
    return rc;
}

static int moe_cuda_batch_publish(moe_cuda_batch *batch,
                                  const yvex_moe_layer_job *job,
                                  const yvex_moe_row_batch *rows,
                                  const yvex_moe_row_batch_output *output,
                                  unsigned long long pairs,
                                  yvex_moe_row_batch_result *result,
                                  yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    unsigned long long hidden, expanded, activation_bytes, selected_bytes, weight_bytes;
    unsigned long long started, completed, cache_hits = 0ull, slot;
    unsigned int hidden_grid, expanded_grid;
    int deferred = job->device_completion != NULL;
    int rc, device_wide = 0;
    if (!yvex_core_u64_mul(rows->row_count, layer->hidden_width, &hidden) ||
        !yvex_core_u64_mul(rows->row_count, layer->expanded_width, &expanded) ||
        !yvex_core_u64_mul(expanded, 2ull * sizeof(float), &activation_bytes) ||
        !yvex_core_u64_mul(pairs, sizeof(*output->selected_experts), &selected_bytes) ||
        !yvex_core_u64_mul(pairs, sizeof(*output->selected_weights), &weight_bytes) ||
        selected_bytes > SIZE_MAX || weight_bytes > SIZE_MAX ||
        !moe_cuda_grid(hidden, MOE_CUDA_BLOCK, &hidden_grid) ||
        !moe_cuda_grid(expanded, MOE_CUDA_BLOCK, &expanded_grid))
        return moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                               "CUDA width-N MoE publication grid is invalid");
    {
        void *params[] = {&batch->routed, &batch->shared, &hidden,
                          &batch->combined, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->moe_combine_rows_function,
            hidden_grid, MOE_CUDA_BLOCK, 0u, params, "cuda.moe.rows.combine",
            &batch->failure, err);
    }
    if (rc == YVEX_OK) {
        CUdeviceptr destination = (CUdeviceptr)rows->device_outputs->data;
        unsigned long long streams = layer->residual_streams, width = layer->hidden_width;
        void *params[] = {
            &batch->combined, &batch->expanded, &batch->post, &batch->combination,
            &streams, &width, &destination, (void *)&rows->row_count, &batch->status};
        rc = batch->ops->launch(
            &batch->work, batch->state->deepseek_mhc_post_function,
            expanded_grid, MOE_CUDA_BLOCK, 0u, params, "cuda.moe.rows.mhc-post",
            &batch->failure, err);
    }
#define DOWNLOAD(target_, source_, bytes_, stage_)                                         \
    do {                                                                                   \
        if (rc == YVEX_OK)                                                                 \
            rc = batch->ops->download(&batch->work, (target_), (source_), (bytes_),        \
                                      (stage_), &batch->failure, err);                     \
        if (rc == YVEX_OK) { batch->d2h += (bytes_); batch->downloads++; }                 \
    } while (0)
    if (deferred) {
        DOWNLOAD(job->device_completion->host_unique_experts, batch->unique,
                 sizeof(*job->device_completion->host_unique_experts),
                 "cuda.moe.rows.unique-download");
        DOWNLOAD(job->device_completion->host_status, batch->status,
                 sizeof(*job->device_completion->host_status),
                 "cuda.moe.rows.status-download");
    } else {
        DOWNLOAD(output->selected_experts, batch->selected,
                 (size_t)selected_bytes, "cuda.moe.rows.selected-download");
        DOWNLOAD(output->selected_weights, batch->weights,
                 (size_t)weight_bytes, "cuda.moe.rows.weights-download");
        DOWNLOAD(&batch->host_unique, batch->unique, sizeof(batch->host_unique),
                 "cuda.moe.rows.unique-download");
        DOWNLOAD(&batch->host_status, batch->status, sizeof(batch->host_status),
                 "cuda.moe.rows.status-download");
    }
#undef DOWNLOAD
    if (rc == YVEX_OK && !deferred) {
        started = yvex_core_monotonic_ns();
        rc = yvex_cuda_launch_synchronize(
            batch->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
            &device_wide, "cuda.moe.rows.finish-sync", err);
        completed = yvex_core_monotonic_ns();
        if (completed > started) batch->synchronization_ns += completed - started;
        if (rc == YVEX_OK) {
            batch->device_synchronizations += (unsigned long long)device_wide;
            batch->stream_synchronizations += (unsigned long long)!device_wide;
        }
    }
    if (rc == YVEX_OK && !deferred && batch->host_status)
        rc = moe_cuda_refuse(err, YVEX_ERR_BACKEND,
                             "CUDA width-N MoE reported invalid or non-finite numerics");
    if (rc == YVEX_OK) {
        for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot)
            cache_hits += layer->tensor_ids[slot] != YVEX_MOE_NO_TENSOR;
        result->schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
        result->completed = !deferred;
        result->device_completion_pending = deferred;
        result->execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
        result->row_count = rows->row_count;
        result->row_expert_pairs = pairs;
        result->unique_experts = deferred ? 0ull : batch->host_unique;
        result->grouped_expert_operations = pairs;
        result->shared_expert_operations = rows->row_count;
        result->h2d_bytes = batch->h2d;
        result->d2h_bytes = batch->d2h;
        result->d2d_bytes = batch->d2d;
        result->kernel_launches = batch->work.launches;
        result->upload_count = batch->h2d != 0ull;
        result->download_count = batch->downloads;
        result->cache_hits = cache_hits;
        result->stream_synchronizations = batch->stream_synchronizations;
        result->device_synchronizations = batch->device_synchronizations;
        result->synchronization_ns = batch->synchronization_ns;
        result->memory.activation_bytes = activation_bytes;
        result->memory.temporary_bytes = batch->work.peak_bytes;
        result->memory.measured_operations = (unsigned long long)!deferred;
        result->memory.complete = !deferred;
        result->total_ns = yvex_core_monotonic_ns() - batch->started_ns;
    }
    return rc;
}

static int moe_cuda_rows_active_bytes(const yvex_moe_layer_job *job,
                                      const yvex_moe_row_batch *rows,
                                      unsigned long long *base_bytes,
                                      unsigned long long *per_expert_bytes)
{
    const yvex_moe_layer_plan *layer = job->layer;
    unsigned long long slot, base = 0ull, per_expert = 0ull, value;
    if (!base_bytes || !per_expert_bytes) return 0;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        const yvex_moe_weight_view *weight = &job->weights[slot];
        if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR ||
            (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
             slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN))
            continue;
        value = weight->encoded_bytes;
        if (slot == YVEX_MOE_WEIGHT_ROUTER_TABLE &&
            !yvex_core_u64_mul(rows->row_count, weight->row_bytes, &value))
            return 0;
        if (!yvex_core_u64_add(base, value, &base)) return 0;
    }
    for (slot = YVEX_MOE_WEIGHT_ROUTED_GATE;
         slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN; ++slot) {
        const yvex_moe_weight_view *weight = &job->weights[slot];
        if (!layer->routed_experts ||
            weight->encoded_bytes % layer->routed_experts != 0ull ||
            !yvex_core_u64_add(per_expert,
                               weight->encoded_bytes / layer->routed_experts,
                               &per_expert))
            return 0;
    }
    *base_bytes = base;
    *per_expert_bytes = per_expert;
    return 1;
}

static int moe_cuda_execute_rows(yvex_backend *backend,
                                 const yvex_moe_layer_job *job,
                                 const yvex_moe_row_batch *rows,
                                 const yvex_moe_row_batch_output *output,
                                 yvex_moe_row_batch_result *result,
                                 yvex_error *err)
{
    moe_cuda_batch batch = {0};
    const yvex_moe_layer_plan *layer = job ? job->layer : NULL;
    yvex_moe_weight_view routed_weights[3], shared_weights[3];
    unsigned long long pairs = 0ull, required = 0ull, active_bytes = 0ull;
    unsigned long long active_base = 0ull, active_per_expert = 0ull;
    int routed_q8, shared_q8, rc, cleanup_rc;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !job || !rows || !output || !result ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA ||
        !backend_tensor_owner_is(backend, rows->device_rows) ||
        !backend_tensor_owner_is(backend, rows->device_outputs) ||
        moe_cuda_rows_geometry(job, rows, output, &pairs, err) != YVEX_OK)
        return yvex_error_code(err) == YVEX_OK
                   ? moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                                     "CUDA width-N MoE owners are invalid")
                   : yvex_error_code(err);
    if (job->cancel_requested && job->cancel_requested(job->cancel_context))
        return moe_cuda_refuse(err, YVEX_ERR_CANCELLED,
                               "CUDA width-N MoE was cancelled before launch");
    memcpy(routed_weights, &job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE],
           sizeof(routed_weights));
    memcpy(shared_weights, &job->weights[YVEX_MOE_WEIGHT_SHARED_GATE],
           sizeof(shared_weights));
    routed_q8 = moe_cuda_rows_q8(layer, routed_weights,
                                 layer->expert_intermediate_width);
    shared_q8 = moe_cuda_rows_q8(layer, shared_weights,
                                 layer->shared_intermediate_width);
    rc = moe_cuda_rows_workspace_required(layer, rows->row_count, &required, err);
    if (rc != YVEX_OK) return rc;
    if (!backend->workspace_device_tensor || backend->workspace_bytes < required)
        return moe_cuda_refuse(err, YVEX_ERR_NOMEM,
                               "CUDA width-N MoE workspace was not preflighted");
    batch.backend = backend;
    batch.state = yvex_cuda_state(backend);
    batch.ops = yvex_cuda_attention_operations_get();
    batch.work.backend = backend;
    batch.work.state = batch.state;
    batch.work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    batch.started_ns = yvex_core_monotonic_ns();
    backend_workspace_reset(backend);
    rc = moe_cuda_batch_ranges(&batch, job, rows, pairs, routed_q8, shared_q8, err);
    if (rc == YVEX_OK) rc = moe_cuda_batch_prepare(&batch, job, rows, err);
    if (rc == YVEX_OK) rc = moe_cuda_batch_route(&batch, job, rows, pairs, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_experts(&batch, job, rows, pairs, 1, routed_q8, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_experts(&batch, job, rows, pairs, 0, shared_q8, err);
    if (rc == YVEX_OK)
        rc = moe_cuda_batch_publish(&batch, job, rows, output, pairs, result, err);
    if (rc == YVEX_OK &&
        !moe_cuda_rows_active_bytes(job, rows, &active_base, &active_per_expert))
        rc = moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                             "CUDA width-N MoE active-byte accounting overflowed");
    if (rc == YVEX_OK && !result->device_completion_pending &&
        (!yvex_core_u64_mul(active_per_expert, result->unique_experts,
                            &active_bytes) ||
         !yvex_core_u64_add(active_base, active_bytes, &active_bytes)))
        rc = moe_cuda_refuse(err, YVEX_ERR_BOUNDS,
                             "CUDA width-N MoE active-byte total overflowed");
    if (rc == YVEX_OK) {
        result->active_weight_base_bytes = active_base;
        result->active_weight_per_unique_expert_bytes = active_per_expert;
        result->encoded_bytes_read = active_bytes;
        result->memory.active_weight_bytes = active_bytes;
    }
    cleanup_rc = yvex_cuda_work_cleanup(&batch.work, rc == YVEX_OK ? err : NULL);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK) rc = cleanup_rc;
    if (rc == YVEX_OK) rows->device_outputs->is_written = 1;
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}

static int moe_cuda_complete_rows(yvex_backend *backend,
                                  int barrier_observed,
                                  yvex_moe_row_batch_result *result,
                                  yvex_error *err)
{
    unsigned long long started = 0ull, completed;
    int device_wide = 0, rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!backend || !result ||
        (barrier_observed != 0 && barrier_observed != 1) ||
        yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA)
        return moe_cuda_refuse(err, YVEX_ERR_INVALID_ARG,
                               "CUDA MoE completion owner is invalid");
    if (!barrier_observed) {
        started = yvex_core_monotonic_ns();
        rc = yvex_cuda_launch_synchronize(
            backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.moe.rows.phase-sync", err);
    }
    completed = yvex_core_monotonic_ns();
    if (rc == YVEX_OK) {
        result->schema_version = YVEX_MOE_ROW_BATCH_SCHEMA_V1;
        result->completed = 1;
        result->execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
        result->device_synchronizations =
            (unsigned long long)(!barrier_observed && device_wide);
        result->stream_synchronizations =
            (unsigned long long)(!barrier_observed && !device_wide);
        result->synchronization_ns =
            !barrier_observed && completed > started ? completed - started : 0ull;
        yvex_error_clear(err);
    }
    return rc;
}

static const yvex_backend_moe_operations moe_cuda_row_operations = {
    moe_cuda_rows_workspace_required,
    moe_cuda_execute_rows,
    moe_cuda_complete_rows
};

const yvex_backend_moe_operations *yvex_backend_moe_operations_get(
    const yvex_backend *backend)
{
    const yvex_cuda_backend_state *state;
    if (!backend || yvex_backend_kind_of(backend) != YVEX_BACKEND_KIND_CUDA ||
        !(state = yvex_cuda_state(backend)) || !state->moe_route_rows_function ||
        !state->moe_pair_order_function || !state->moe_grouped_up_rows_function ||
        !state->moe_grouped_down_rows_function || !state->moe_reduce_rows_function ||
        !state->moe_combine_rows_function || !state->deepseek_mhc_pre_function ||
        !state->deepseek_mhc_post_function)
        return NULL;
    return &moe_cuda_row_operations;
}
