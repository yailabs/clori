/* Owner: CUDA MoE execution.
 * Owns: selected-expert device staging, admitted MoE kernel launches, synchronization, and cleanup.
 * Does not own: family routing policy, artifact addressing, runtime sessions, CPU fallback, or CLI evidence.
 * Invariants: CUDA requests execute all numerical stages on device and upload selected expert subviews only.
 * Boundary: backend execution consumes a typed MoE job and never reconstructs model topology.
 * Purpose: execute one DeepSeek-selected MoE layer with canonical qtype kernels on CUDA.
 * Inputs: admitted encoded fixed/selected weights, expanded activation, and exact family plan facts.
 * Effects: uses stable workspace ranges, transfers selected bytes, and publishes output after synchronization.
 * Failure: copy, launch, status, or cleanup failure publishes no successful result. */
#include <yvex/internal/moe.h>

#include "src/backend/cuda/private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define MOE_CUDA_BLOCK 256u

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
    int host_status, finished;
    unsigned long long h2d, d2h, subviews, uploads;
};

/* Purpose: publish one CUDA MoE refusal through the typed backend owner. */
static int moe_cuda_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "cuda.moe", reason);
    return status;
}

/* Purpose: project one graph weight view into the existing encoded CUDA matvec ABI. */
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

/* Purpose: allocate one stable range. Inputs: workspace, extent, and optional source.
 * Effects: advances workspace. Failure: typed backend error. Boundary: CUDA MoE scratch. */
static int moe_cuda_allocate(yvex_backend_moe_execution *execution, CUdeviceptr *out,
                             size_t bytes, const void *source, int zero,
                             const char *stage, yvex_error *err)
{
    return execution->ops->allocate(&execution->work, out, bytes, source, zero, stage,
                                    &execution->failure, err);
}

/* Purpose: upload one encoded weight. Inputs: admitted view and stable staging range.
 * Effects: replaces staged bytes. Failure: typed bound/transfer error. Boundary: CUDA MoE weight. */
static int moe_cuda_upload(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight,
                           const char *stage, yvex_error *err)
{
    int rc;
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

/* Purpose: execute one encoded matrix-vector product from the reusable weight range. */
static int moe_cuda_matvec(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight, CUdeviceptr input,
                           CUdeviceptr output, int round_bf16,
                           const char *stage, yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    int rc = moe_cuda_upload(execution, weight, stage, err);
    return rc == YVEX_OK
               ? execution->ops->matvec(&execution->work, &encoded,
                                         execution->weight_buffer, 0ull,
                                         weight->row_count, input, output, round_bf16,
                                         execution->status, stage, &execution->failure, err)
               : rc;
}

/* Purpose: decode one coefficient vector. Inputs: encoded view and output range.
 * Effects: writes device coefficients. Failure: typed CUDA error. Boundary: CUDA qtype execution. */
static int moe_cuda_decode(yvex_backend_moe_execution *execution,
                           const yvex_moe_weight_view *weight, CUdeviceptr output,
                           const char *stage, yvex_error *err)
{
    yvex_backend_attention_weight encoded = moe_cuda_weight(weight);
    int rc = moe_cuda_upload(execution, weight, stage, err);
    return rc == YVEX_OK
               ? execution->ops->decode(&execution->work, &encoded,
                                         execution->weight_buffer, 0ull,
                                         weight->row_width, output, execution->status,
                                         stage, &execution->failure, err)
               : rc;
}

/* Purpose: synchronize one CUDA MoE phase and reject device status atomically. */
static int moe_cuda_sync_status(yvex_backend_moe_execution *execution,
                                const char *stage, yvex_error *err)
{
    int rc = execution->ops->download(&execution->work, &execution->host_status,
                                      execution->status, sizeof(execution->host_status),
                                      stage, &execution->failure, err);
    if (rc == YVEX_OK)
        rc = yvex_cuda_synchronize(execution->backend,
                                   YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
                                   stage, err);
    execution->d2h += sizeof(execution->host_status);
    if (rc == YVEX_OK && execution->host_status)
        rc = moe_cuda_refuse(err, YVEX_ERR_BACKEND,
                             "CUDA MoE kernel reported invalid or non-finite numerics");
    return rc;
}

/* Purpose: allocate per-layer stable ranges. Inputs: sealed job and workspace.
 * Effects: fixes device addresses. Failure: typed capacity error. Boundary: CUDA MoE workspace. */
static int moe_cuda_ranges(yvex_backend_moe_execution *execution, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = execution->job->layer;
    size_t hidden = (size_t)layer->hidden_width * sizeof(float);
    size_t expanded = (size_t)layer->expanded_width * sizeof(float);
    size_t intermediate = (size_t)layer->shared_intermediate_width * sizeof(float);
    size_t routed = (size_t)layer->routed_experts * sizeof(float);
    size_t selected = (size_t)layer->experts_per_token * sizeof(unsigned long long);
    size_t selected_weights = (size_t)layer->experts_per_token * sizeof(float);
    int rc = moe_cuda_allocate(execution, &execution->status, sizeof(int), NULL, 1,
                               "cuda.moe.status", err);
#define RANGE(member_, bytes_, source_, zero_, stage_)                                      \
    do {                                                                                    \
        if (rc == YVEX_OK)                                                                  \
            rc = moe_cuda_allocate(execution, &execution->member_, (bytes_), (source_),     \
                                   (zero_), (stage_), err);                                  \
    } while (0)
    RANGE(expanded, expanded, execution->job->expanded_input, 0, "cuda.moe.input");
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
    if (rc == YVEX_OK) execution->h2d += expanded;
    return rc;
}

/* Purpose: execute mHC FFN ingress and RMS norm. Inputs: typed layer weights and activation.
 * Effects: writes device input state. Failure: typed kernel error. Boundary: CUDA MoE preparation. */
static int moe_cuda_prepare_input(yvex_backend_moe_execution *execution, yvex_error *err)
{
    const yvex_moe_layer_job *job = execution->job;
    const yvex_moe_layer_plan *layer = job->layer;
    yvex_backend_attention_weight norm = moe_cuda_weight(&job->weights[YVEX_MOE_WEIGHT_FFN_NORM]);
    unsigned long long streams = layer->residual_streams, width = layer->hidden_width;
    int rc = moe_cuda_matvec(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_FUNCTION],
                             execution->expanded, execution->mix, 0,
                             "cuda.moe.mhc-function", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_decode(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_SCALE],
                             execution->scale, "cuda.moe.mhc-scale", err);
    if (rc == YVEX_OK)
        rc = moe_cuda_decode(execution, &job->weights[YVEX_MOE_WEIGHT_MHC_BASE],
                             execution->base, "cuda.moe.mhc-base", err);
    if (rc == YVEX_OK) {
        void *params[] = {
            &execution->expanded, &execution->mix, &execution->scale, &execution->base,
            &streams, &width, (void *)&layer->mhc_mixing_rows,
            (void *)&layer->mhc_sinkhorn_iterations, (void *)&layer->rms_epsilon,
            (void *)&layer->mhc_epsilon, (void *)&layer->mhc_post_multiplier,
            &execution->normalized, &execution->post, &execution->combination,
            &execution->status};
        rc = execution->ops->launch(&execution->work, execution->state->deepseek_mhc_pre_function,
                                    1u, 1u, 0u, params, "cuda.moe.mhc-pre",
                                    &execution->failure, err);
    }
    if (rc == YVEX_OK &&
        (rc = moe_cuda_upload(execution, &job->weights[YVEX_MOE_WEIGHT_FFN_NORM],
                              "cuda.moe.norm-weight", err)) == YVEX_OK)
        rc = execution->ops->weighted_norm(
            &execution->work, execution->normalized, layer->hidden_width, &norm,
            execution->weight_buffer, layer->rms_epsilon, execution->status,
            "cuda.moe.ffn-norm", &execution->failure, err);
    return rc;
}

/* Purpose: execute routing. Inputs: prepared state and typed router policy.
 * Effects: publishes selection after sync. Failure: typed numeric error. Boundary: CUDA MoE router. */
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
        rc = execution->ops->initialize(&execution->work, execution->route_aux,
                                        aux->encoded_bytes, aux->encoded, 0,
                                        "cuda.moe.router-bias", &execution->failure, err);
        bias = execution->route_aux;
        execution->h2d += aux->encoded_bytes;
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
        if (rc == YVEX_OK) execution->d2h += (bytes_);                                     \
    } while (0)
    DOWNLOAD(result->router.router_logits, execution->logits,
             (size_t)layer->routed_experts * sizeof(float), "cuda.moe.logits-download");
    DOWNLOAD(result->router.router_scores, execution->scores,
             (size_t)layer->routed_experts * sizeof(float), "cuda.moe.scores-download");
    DOWNLOAD(result->router.selected_experts, execution->selected,
             (size_t)layer->experts_per_token * sizeof(unsigned long long),
             "cuda.moe.selected-download");
    DOWNLOAD(result->router.selected_weights, execution->weights,
             (size_t)layer->experts_per_token * sizeof(float), "cuda.moe.weights-download");
#undef DOWNLOAD
    return rc == YVEX_OK ? moe_cuda_sync_status(execution, "cuda.moe.route-sync", err) : rc;
}

/* Purpose: begin one CUDA MoE layer. Inputs: admitted backend job and result storage.
 * Effects: owns work and publishes routing. Failure: closes partial work. Boundary: backend lifecycle. */
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
    execution = (yvex_backend_moe_execution *)calloc(1u, sizeof(*execution));
    if (!execution) return moe_cuda_refuse(err, YVEX_ERR_NOMEM, "CUDA MoE owner allocation failed");
    execution->backend = backend;
    execution->state = yvex_cuda_state(backend);
    execution->ops = yvex_cuda_attention_operations_get();
    execution->job = job;
    execution->work.backend = backend;
    execution->work.state = execution->state;
    execution->work.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    if (!execution->state || !execution->state->moe_route_function ||
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
    if (rc == YVEX_OK) rc = moe_cuda_route(execution, result, err);
    if (rc != YVEX_OK) goto fail;
    *out = execution;
    return YVEX_OK;
fail:
    (void)yvex_backend_moe_close(&execution, NULL);
    return rc;
}

/* Purpose: execute one selected expert. Inputs: exact gate/up/down views and route weight.
 * Effects: accumulates device output. Failure: typed kernel error. Boundary: backend MoE compute. */
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
    }
    return rc;
}

/* Purpose: publish one complete CUDA MoE output. Inputs: complete execution and result storage.
 * Effects: downloads typed result. Failure: publishes no success. Boundary: backend transaction. */
int yvex_backend_moe_finish(yvex_backend_moe_execution *execution,
                            yvex_moe_layer_result *result, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = execution && execution->job
                                           ? execution->job->layer : NULL;
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
#define DOWNLOAD(target_, source_, bytes_, stage_)                                         \
    do {                                                                                   \
        if (rc == YVEX_OK)                                                                 \
            rc = execution->ops->download(&execution->work, (target_), (source_),          \
                                          (bytes_), (stage_), &execution->failure, err);    \
        if (rc == YVEX_OK) execution->d2h += (bytes_);                                     \
    } while (0)
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
#undef DOWNLOAD
    if (rc == YVEX_OK) rc = moe_cuda_sync_status(execution, "cuda.moe.finish-sync", err);
    if (rc != YVEX_OK) return rc;
    result->expert_subviews_accessed += execution->subviews;
    result->host_to_device_bytes += execution->h2d;
    result->device_to_host_bytes += execution->d2h;
    result->kernel_launches += execution->work.launches;
    result->upload_count += execution->uploads;
    execution->finished = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Purpose: release one CUDA MoE transaction. Inputs: owned execution handle.
 * Effects: frees work on success. Failure: retains retryable ownership. Boundary: backend cleanup. */
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
