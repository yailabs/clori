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
static int moe_cuda_add_selected(yvex_backend_moe_execution *execution, yvex_error *err);

static int moe_cuda_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "cuda.moe", reason);
    return status;
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
                                         weight->row_count, input, output, round_bf16,
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
        (rc = moe_cuda_weight_address(execution, &job->weights[YVEX_MOE_WEIGHT_FFN_NORM],
                                      &norm_weight, "cuda.moe.norm-weight", err)) == YVEX_OK)
        rc = execution->ops->weighted_norm(
            &execution->work, execution->normalized, layer->hidden_width, &norm,
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
        (gate->qtype == YVEX_GGUF_QTYPE_IQ2_XXS || gate->qtype == YVEX_GGUF_QTYPE_Q2_K ||
         gate->qtype == YVEX_GGUF_QTYPE_Q8_0) &&
        (up->qtype == YVEX_GGUF_QTYPE_IQ2_XXS || up->qtype == YVEX_GGUF_QTYPE_Q2_K ||
         up->qtype == YVEX_GGUF_QTYPE_Q8_0) &&
        (down->qtype == YVEX_GGUF_QTYPE_IQ2_XXS || down->qtype == YVEX_GGUF_QTYPE_Q2_K ||
         down->qtype == YVEX_GGUF_QTYPE_Q8_0);
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
        unsigned long long count = streams * width;
        unsigned int grid = (unsigned int)((count + MOE_CUDA_BLOCK - 1ull) / MOE_CUDA_BLOCK);
        CUdeviceptr output = (CUdeviceptr)execution->job->device_output->data;
        void *params[] = {&execution->combined, &execution->expanded, &execution->post,
                          &execution->combination, &streams, &width, &output,
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
