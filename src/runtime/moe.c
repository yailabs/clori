/* Owner: runtime MoE execution.
 * Owns: immutable MoE context, selected-weight access, CPU/CUDA dispatch, cancellation, and publication.
 * Does not own: family policy, artifact admission, graph numerics, persistent KV, transformer composition, or CLI I/O.
 * Invariants: one context binds one model/session and reads only fixed or selected expert ranges.
 * Boundary: runtime coordinates admitted graph/backend owners without reconstructing compiler or family truth.
 * Purpose: execute complete token-local MoE blocks through reusable session-owned resources.
 * Inputs: sealed runtime model/session, typed activation input, and bounded execution options.
 * Effects: reads admitted ranges, reuses scratch/workspace, and publishes only complete typed results.
 * Failure: cancellation, identity drift, resource, numerical, or cleanup failure publishes no partial result. */
#include <yvex/internal/moe.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    unsigned char *data;
    unsigned long long capacity;
} moe_byte_buffer;
struct yvex_runtime_moe_context {
    yvex_runtime_model *model;
    yvex_runtime_execution_session *session;
    const yvex_runtime_model_view *model_view;
    const yvex_runtime_session_view *session_view;
    yvex_moe_plan *plan;
    yvex_runtime_moe_options options;
    moe_byte_buffer fixed[YVEX_MOE_WEIGHT_COUNT];
    moe_byte_buffer selected[3];
    float *scratch, *normalized, *post, *combination;
    float *expert, *routed, *shared, *combined;
    unsigned long long hidden_capacity, residual_capacity;
    float *candidate_combined, *candidate_post, *candidate_combination;
    unsigned long long candidate_tokens, candidate_layers;
    yvex_device_tensor *device_workspace;
    unsigned long long host_bytes, execution_count;
    int workspace_owned, workspace_ready, busy, invalidated;
    pthread_mutex_t mutex;
    int mutex_ready;
};
/* Purpose: publish one stable runtime MoE refusal. */
static int runtime_moe_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.moe", reason);
    return status;
}
/* Purpose: resolve one exact materialization binding named by the immutable plan. */
static const yvex_materialized_tensor_binding *runtime_moe_binding(
    const yvex_runtime_moe_context *context, const yvex_moe_layer_plan *layer,
    yvex_moe_weight_slot slot)
{
    unsigned long long tensor_id = layer && slot < YVEX_MOE_WEIGHT_COUNT
                                       ? layer->tensor_ids[slot] : YVEX_MOE_NO_TENSOR;
    return tensor_id == YVEX_MOE_NO_TENSOR ? NULL
        : yvex_materialization_session_tensor_at(context->model_view->materialization, tensor_id);
}
/* Purpose: derive one encoded row extent from an admitted materialization binding. */
static int runtime_moe_row_bytes(const yvex_materialized_tensor_binding *binding,
                                 unsigned long long *out)
{
    return binding && out && binding->row_count &&
           binding->encoded_bytes % binding->row_count == 0ull &&
           ((*out = binding->encoded_bytes / binding->row_count) != 0ull);
}
/* Purpose: project binding geometry plus caller-owned bytes into the generic graph weight ABI.
 * Inputs: binding/bytes. Effects: fills view. Failure: typed. Boundary: borrowed encoded bytes. */
static int runtime_moe_weight(yvex_moe_weight_view *out,
                              const yvex_materialized_tensor_binding *binding,
                              const unsigned char *bytes, unsigned long long encoded_bytes,
                              unsigned long long row_count, unsigned long long expert,
                              unsigned long long device_address,
                              yvex_error *err)
{
    unsigned long long row_bytes, expected;
    if (!out || !binding || !bytes || !encoded_bytes || !row_count ||
        !runtime_moe_row_bytes(binding, &row_bytes) ||
        !yvex_core_u64_mul(row_bytes, row_count, &expected) ||
        expected != encoded_bytes || expected > binding->encoded_bytes)
        return runtime_moe_refuse(err, YVEX_ERR_FORMAT, "MoE weight view geometry is invalid");
    memset(out, 0, sizeof(*out));
    out->tensor_id = binding->tensor_id;
    out->expert_index = expert;
    out->role = binding->role;
    out->qtype = binding->qtype;
    out->encoded = bytes;
    out->encoded_bytes = (size_t)encoded_bytes;
    out->row_bytes = row_bytes;
    out->row_width = binding->row_width;
    out->row_count = row_count;
    out->device_address = device_address;
    return YVEX_OK;
}
/* Purpose: copy one exact artifact range into a prepared reusable host buffer.
 * Inputs: context/range. Effects: reads bytes. Failure: typed. Boundary: admitted open artifact. */
static int runtime_moe_read(yvex_runtime_moe_context *context,
                            const yvex_materialized_tensor_binding *binding,
                            unsigned long long offset, unsigned long long bytes,
                            moe_byte_buffer *buffer, yvex_error *err)
{
    yvex_materialization_failure failure;
    if (!binding || !buffer || !buffer->data || bytes > buffer->capacity ||
        bytes > (unsigned long long)SIZE_MAX)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                  "MoE selected weight exceeds prepared host storage");
    memset(&failure, 0, sizeof(failure));
    return yvex_materialization_session_read(context->model_view->materialization, binding,
                                              offset, buffer->data, (size_t)bytes,
                                              &failure, err);
}
/* Purpose: borrow one CUDA-addressable resident range or copy it for CPU execution.
 * Inputs: admitted binding subrange and the CPU fallback buffer owned by this context.
 * Effects: publishes an immutable host view plus its optional mapped CUDA address.
 * Failure: range, residency, or read failure publishes no borrowed view.
 * Boundary: CUDA execution never restages model-resident bytes through a host scratch copy. */
static int runtime_moe_access(yvex_runtime_moe_context *context,
                              const yvex_materialized_tensor_binding *binding,
                              unsigned long long offset, unsigned long long bytes,
                              moe_byte_buffer *buffer, const unsigned char **data,
                              unsigned long long *device_address, yvex_error *err)
{
    const unsigned char *resident = NULL;
    unsigned long long resident_bytes = 0ull;
    if (data) *data = NULL;
    if (device_address) *device_address = 0ull;
    if (!context || !binding || !bytes || !data || !device_address ||
        offset > binding->encoded_bytes || bytes > binding->encoded_bytes - offset)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                  "MoE encoded subrange is invalid");
    if (yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA) {
        int rc = runtime_moe_read(context, binding, offset, bytes, buffer, err);
        if (rc == YVEX_OK) *data = buffer->data;
        return rc;
    }
    if (yvex_runtime_residency_binding_view(context->model_view->residency, binding,
                                             &resident, &resident_bytes, err) != YVEX_OK ||
        offset > resident_bytes || bytes > resident_bytes - offset)
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "MoE resident binding range is unavailable");
    *data = resident + offset;
    if (yvex_backend_resident_resolve(context->session_view->backend, *data,
                                      bytes, device_address) != YVEX_BACKEND_RESIDENT_HIT) {
        *data = NULL;
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "MoE resident bytes are not CUDA-addressable");
    }
    return YVEX_OK;
}
/* Purpose: calculate exact reusable host buffer maxima without reading payload bytes.
 * Inputs: context plan. Effects: records bounds. Failure: typed. Boundary: checked geometry. */
static int runtime_moe_buffer_plan(yvex_runtime_moe_context *context, yvex_error *err)
{
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(context->plan);
    unsigned long long layer_index, slot, total = sizeof(*context), scratch_count, scratch_bytes;
    int cuda = yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA;
    for (layer_index = 0ull; layer_index < summary->layer_count; ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(context->plan, layer_index);
        if (layer->hidden_width > context->hidden_capacity)
            context->hidden_capacity = layer->hidden_width;
        if (layer->residual_streams > context->residual_capacity)
            context->residual_capacity = layer->residual_streams;
        for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
            const yvex_materialized_tensor_binding *binding;
            unsigned long long bytes;
            if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR) continue;
            binding = runtime_moe_binding(context, layer, (yvex_moe_weight_slot)slot);
            if (!binding) return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                                    "MoE plan binding disappeared");
            if (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
                slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN) {
                if (!binding->expert_count || binding->encoded_bytes % binding->expert_count)
                    return runtime_moe_refuse(err, YVEX_ERR_FORMAT,
                                              "MoE expert aggregate cannot form exact subviews");
                bytes = binding->encoded_bytes / binding->expert_count;
                if (bytes > context->selected[slot - YVEX_MOE_WEIGHT_ROUTED_GATE].capacity)
                    context->selected[slot - YVEX_MOE_WEIGHT_ROUTED_GATE].capacity = bytes;
                continue;
            }
            bytes = slot == YVEX_MOE_WEIGHT_ROUTER_TABLE
                        ? binding->encoded_bytes / binding->row_count
                        : binding->encoded_bytes;
            if (bytes > context->fixed[slot].capacity) context->fixed[slot].capacity = bytes;
        }
    }
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        if (!context->fixed[slot].capacity || cuda) continue;
        if (!yvex_core_u64_add(total, context->fixed[slot].capacity, &total)) goto overflow;
        context->fixed[slot].data = (unsigned char *)malloc((size_t)context->fixed[slot].capacity);
        if (!context->fixed[slot].data) goto allocation;
    }
    for (slot = 0ull; slot < 3ull; ++slot) {
        if (cuda) continue;
        if (!yvex_core_u64_add(total, context->selected[slot].capacity, &total)) goto overflow;
        context->selected[slot].data = (unsigned char *)malloc((size_t)context->selected[slot].capacity);
        if (!context->selected[slot].data) goto allocation;
    }
    if (!context->hidden_capacity || !context->residual_capacity ||
        !yvex_core_u64_mul(context->hidden_capacity, 5ull, &scratch_count) ||
        !yvex_core_u64_add(scratch_count, context->residual_capacity, &scratch_count) ||
        !yvex_core_u64_mul(context->residual_capacity, context->residual_capacity,
                           &scratch_bytes) ||
        !yvex_core_u64_add(scratch_count, scratch_bytes, &scratch_count) ||
        !yvex_core_u64_mul(scratch_count, sizeof(float), &scratch_bytes) ||
        !yvex_core_u64_add(total, scratch_bytes, &total))
        goto overflow;
    if (context->options.maximum_host_bytes && total > context->options.maximum_host_bytes)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE host buffers exceed their budget");
    context->scratch = (float *)calloc((size_t)scratch_count, sizeof(float));
    if (!context->scratch) goto allocation;
    context->normalized = context->scratch;
    context->expert = context->normalized + context->hidden_capacity;
    context->routed = context->expert + context->hidden_capacity;
    context->shared = context->routed + context->hidden_capacity;
    context->combined = context->shared + context->hidden_capacity;
    context->post = context->combined + context->hidden_capacity;
    context->combination = context->post + context->residual_capacity;
    context->host_bytes = total;
    return YVEX_OK;
overflow:
    return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE host buffer extent overflowed");
allocation:
    return runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE reusable host buffer allocation failed");
}
/* Purpose: attach one exact stable CUDA workspace when the session has none.
 * Inputs: context. Effects: allocates device state. Failure: typed. Boundary: backend owner. */
static int runtime_moe_cuda_workspace(yvex_runtime_moe_context *context, yvex_error *err)
{
    yvex_runtime_session_summary session_summary;
    yvex_backend_tensor_desc descriptor = {0};
    int rc;
    if (yvex_runtime_session_summary_copy(context->session, &session_summary, err) != YVEX_OK)
        return yvex_error_code(err);
    if (session_summary.device_workspace_bytes) {
        if (session_summary.device_workspace_bytes < YVEX_MOE_CUDA_WORKSPACE_BYTES)
            return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                      "existing CUDA workspace is too small for MoE");
        context->workspace_ready = 1;
        return YVEX_OK;
    }
    if (context->options.maximum_device_bytes &&
        YVEX_MOE_CUDA_WORKSPACE_BYTES > context->options.maximum_device_bytes)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE CUDA workspace exceeds its budget");
    descriptor.name = "moe_workspace";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = YVEX_MOE_CUDA_WORKSPACE_BYTES / sizeof(float);
    descriptor.bytes = YVEX_MOE_CUDA_WORKSPACE_BYTES;
    rc = yvex_backend_tensor_alloc(context->session_view->backend, &descriptor,
                                   &context->device_workspace, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_workspace_attach(context->session_view->backend,
                                           context->device_workspace, 1ull, err);
    if (rc == YVEX_OK) {
        context->workspace_owned = 1;
        context->workspace_ready = 1;
    }
    return rc;
}
/* Purpose: load fixed per-layer weights and retain aggregate metadata without full expert reads.
 * Inputs: layer/token. Effects: reads ranges. Failure: typed. Boundary: fixed bindings only. */
static int runtime_moe_load_layer(yvex_runtime_moe_context *context,
                                  const yvex_moe_layer_plan *layer,
                                  unsigned int token_id, yvex_moe_layer_job *job,
                                  unsigned long long *bytes_read, yvex_error *err)
{
    unsigned long long slot;
    memset(job, 0, sizeof(*job));
    job->layer = layer;
    job->token_id = token_id;
    job->token_id_present = 1;
    job->cancel_requested = context->options.cancel_requested;
    job->cancel_context = context->options.cancel_context;
    job->evidence_level = context->options.evidence_level;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        const yvex_materialized_tensor_binding *binding;
        const unsigned char *data = NULL;
        unsigned long long device_address = 0ull;
        unsigned long long offset = 0ull, bytes, rows;
        if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR) continue;
        binding = runtime_moe_binding(context, layer, (yvex_moe_weight_slot)slot);
        if (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
            slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN) {
            if (yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA &&
                runtime_moe_access(context, binding, 0ull, binding->encoded_bytes,
                                   &context->fixed[slot], &data, &device_address, err) != YVEX_OK)
                return yvex_error_code(err);
            job->weights[slot].tensor_id = binding->tensor_id;
            job->weights[slot].role = binding->role;
            job->weights[slot].qtype = binding->qtype;
            job->weights[slot].encoded = data;
            job->weights[slot].encoded_bytes = (size_t)binding->encoded_bytes;
            job->weights[slot].row_width = binding->row_width;
            job->weights[slot].row_count = binding->row_count;
            job->weights[slot].row_bytes = binding->encoded_bytes / binding->row_count;
            job->weights[slot].device_address = device_address;
            continue;
        }
        rows = binding->row_count;
        bytes = binding->encoded_bytes;
        if (slot == YVEX_MOE_WEIGHT_ROUTER_TABLE) {
            if ((unsigned long long)token_id >= layer->hash_table_rows)
                return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                          "MoE token ID exceeds hash router table");
            bytes = binding->encoded_bytes / binding->row_count;
            offset = (unsigned long long)token_id * bytes;
            rows = 1ull;
        }
        if (runtime_moe_access(context, binding, offset, bytes, &context->fixed[slot],
                               &data, &device_address, err) != YVEX_OK ||
            runtime_moe_weight(&job->weights[slot], binding, data, bytes, rows,
                               YVEX_MOE_NO_TENSOR, device_address, err) != YVEX_OK)
            return yvex_error_code(err);
        *bytes_read += bytes;
    }
    return YVEX_OK;
}
/* Purpose: read exactly one routed expert's three physical subviews.
 * Inputs: expert ordinal. Effects: reads selected bytes. Failure: typed. Boundary: selected expert. */
static int runtime_moe_load_expert(yvex_runtime_moe_context *context,
                                   const yvex_moe_layer_plan *layer,
                                   unsigned long long expert,
                                   yvex_moe_weight_view views[3],
                                   unsigned long long *bytes_read, yvex_error *err)
{
    unsigned long long index;
    for (index = 0ull; index < 3ull; ++index) {
        yvex_moe_weight_slot slot = (yvex_moe_weight_slot)(YVEX_MOE_WEIGHT_ROUTED_GATE + index);
        const yvex_materialized_tensor_binding *binding = runtime_moe_binding(context, layer, slot);
        yvex_materialized_expert_subview subview;
        yvex_materialization_failure failure;
        const unsigned char *data = NULL;
        unsigned long long device_address = 0ull;
        unsigned long long offset, rows;
        memset(&failure, 0, sizeof(failure));
        if (yvex_materialization_session_expert_subview(
                context->model_view->materialization, binding, expert, &subview,
                &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        offset = subview.absolute_offset - binding->absolute_offset;
        rows = binding->row_count / binding->expert_count;
        if (runtime_moe_access(context, binding, offset, subview.encoded_bytes,
                               &context->selected[index], &data, &device_address, err) != YVEX_OK ||
            runtime_moe_weight(&views[index], binding, data, subview.encoded_bytes,
                               rows, expert, device_address, err) != YVEX_OK)
            return yvex_error_code(err);
        *bytes_read += subview.encoded_bytes;
    }
    return YVEX_OK;
}
/* Purpose: hash one exact typed byte span for transactional result publication. */
static int runtime_moe_digest(const void *data, size_t bytes,
                              char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update(&hash, data, bytes) || !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
/* Purpose: publish finite combined values through the exact BF16 RNE boundary. */
static int runtime_moe_round(float *values, unsigned long long count)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        values[index] = yvex_quant_bf16_decode(yvex_quant_bf16_encode(values[index]));
    }
    return 1;
}
/* Purpose: execute one admitted layer on CPU using only generic graph primitives.
 * Inputs: context/job. Effects: stages output. Failure: typed. Boundary: no publication. */
static int runtime_moe_layer_cpu(yvex_runtime_moe_context *context,
                                 const yvex_moe_layer_job *job,
                                 yvex_moe_layer_result *result, yvex_error *err)
{
    const yvex_moe_layer_plan *layer = job->layer;
    yvex_moe_weight_view selected[3];
    unsigned long long rank, lane, bytes_read = 0ull;
    int rc = yvex_moe_ffn_prepare_cpu(job, context->normalized, context->post,
                                      context->combination, err);
    if (rc == YVEX_OK)
        rc = yvex_moe_route_cpu(job, context->normalized, &result->router, err);
    memset(context->routed, 0, (size_t)layer->hidden_width * sizeof(float));
    memset(context->shared, 0, (size_t)layer->hidden_width * sizeof(float));
    for (rank = 0ull; rc == YVEX_OK && rank < result->router.selected_count; ++rank) {
        if (job->cancel_requested && job->cancel_requested(job->cancel_context))
            rc = runtime_moe_refuse(err, YVEX_ERR_CANCELLED, "MoE execution was cancelled");
        if (rc == YVEX_OK)
            rc = runtime_moe_load_expert(context, layer,
                                         result->router.selected_experts[rank], selected,
                                         &bytes_read, err);
        if (rc == YVEX_OK)
            rc = yvex_moe_expert_cpu(layer, &selected[0], &selected[1], &selected[2],
                                     context->normalized, context->expert, err);
        if (rc == YVEX_OK)
            for (lane = 0ull; lane < layer->hidden_width; ++lane)
                context->routed[lane] += context->expert[lane] *
                                         result->router.selected_weights[rank];
    }
    if (rc == YVEX_OK)
        rc = yvex_moe_expert_cpu(layer, &job->weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                                 &job->weights[YVEX_MOE_WEIGHT_SHARED_UP],
                                 &job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN],
                                 context->normalized, context->shared, err);
    if (rc != YVEX_OK) return rc;
    for (lane = 0ull; lane < layer->hidden_width; ++lane)
        context->combined[lane] = context->routed[lane] + context->shared[lane];
    if (!runtime_moe_round(context->combined, layer->hidden_width))
        return runtime_moe_refuse(err, YVEX_ERR_FORMAT, "MoE combined output is non-finite");
    memcpy(result->combined_output, context->combined,
           (size_t)layer->hidden_width * sizeof(float));
    memcpy(result->routed_output, context->routed,
           (size_t)layer->hidden_width * sizeof(float));
    memcpy(result->shared_output, context->shared,
           (size_t)layer->hidden_width * sizeof(float));
    memcpy(result->post, context->post, (size_t)layer->residual_streams * sizeof(float));
    memcpy(result->combination, context->combination,
           (size_t)layer->residual_streams * layer->residual_streams * sizeof(float));
    result->expert_subviews_accessed = result->router.selected_count * 3ull;
    result->encoded_bytes_read += bytes_read;
    return YVEX_OK;
}
/* Purpose: execute one admitted layer on CUDA and expose only selected expert subviews.
 * Inputs: context/job. Effects: stages device output. Failure: typed. Boundary: no CPU fallback. */
static int runtime_moe_layer_cuda(yvex_runtime_moe_context *context,
                                  const yvex_moe_layer_job *job,
                                  yvex_moe_layer_result *result, yvex_error *err)
{
    yvex_backend_moe_execution *execution = NULL;
    yvex_moe_weight_view selected[3];
    unsigned long long rank, bytes_read = 0ull;
    int rc = yvex_backend_moe_begin(&execution, context->session_view->backend, job, result, err);
    if (rc == YVEX_OK && job->device_output &&
        job->evidence_level != YVEX_ATTENTION_EVIDENCE_FULL &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE].device_address &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_UP].device_address &&
        job->weights[YVEX_MOE_WEIGHT_ROUTED_DOWN].device_address) {
        bytes_read += result->router.selected_count *
            (job->weights[YVEX_MOE_WEIGHT_ROUTED_GATE].encoded_bytes +
             job->weights[YVEX_MOE_WEIGHT_ROUTED_UP].encoded_bytes +
             job->weights[YVEX_MOE_WEIGHT_ROUTED_DOWN].encoded_bytes) /
            job->layer->routed_experts;
        rank = result->router.selected_count;
    } else for (rank = 0ull; rc == YVEX_OK && rank < result->router.selected_count; ++rank) {
        if (job->cancel_requested && job->cancel_requested(job->cancel_context))
            rc = runtime_moe_refuse(err, YVEX_ERR_CANCELLED, "CUDA MoE execution was cancelled");
        if (rc == YVEX_OK)
            rc = runtime_moe_load_expert(context, job->layer,
                                         result->router.selected_experts[rank], selected,
                                         &bytes_read, err);
        if (rc == YVEX_OK)
            rc = yvex_backend_moe_add_expert(execution, &selected[0], &selected[1],
                                             &selected[2], result->router.selected_weights[rank],
                                             0, err);
    }
    if (rc == YVEX_OK)
        rc = yvex_backend_moe_add_expert(
            execution, &job->weights[YVEX_MOE_WEIGHT_SHARED_GATE],
            &job->weights[YVEX_MOE_WEIGHT_SHARED_UP],
            &job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN], 1.0f, 1, err);
    if (rc == YVEX_OK) rc = yvex_backend_moe_finish(execution, result, err);
    {
        yvex_error cleanup;
        int cleanup_rc = yvex_backend_moe_close(&execution, &cleanup);
        if (rc == YVEX_OK && cleanup_rc != YVEX_OK) {
            rc = cleanup_rc;
            if (err) *err = cleanup;
        }
    }
    result->encoded_bytes_read += bytes_read;
    return rc;
}
/* Purpose: execute one layer into context-owned staging and derive layer digests.
 * Inputs: context/layer/input. Effects: stages result. Failure: typed. Boundary: no publication. */
static int runtime_moe_layer_owned(yvex_runtime_moe_context *context,
                                   unsigned long long layer_index,
                                   const float *expanded_input,
                                   const yvex_device_tensor *device_input,
                                   yvex_device_tensor *device_output, unsigned int token_id,
                                   int token_id_present, yvex_moe_layer_result *result,
                                   yvex_error *err)
{
    const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(context->plan, layer_index);
    yvex_moe_layer_job job;
    unsigned long long fixed_bytes = 0ull, slot;
    int rc;
    memset(result, 0, sizeof(*result));
    if (!layer || !expanded_input || !token_id_present ||
        (context->options.cancel_requested &&
         context->options.cancel_requested(context->options.cancel_context)))
        return runtime_moe_refuse(err, !layer || !expanded_input || !token_id_present
                                           ? YVEX_ERR_INVALID_ARG : YVEX_ERR_CANCELLED,
                                  "MoE layer request is invalid or cancelled");
    rc = runtime_moe_load_layer(context, layer, token_id, &job, &fixed_bytes, err);
    job.expanded_input = expanded_input;
    job.device_input = device_input;
    job.device_output = device_output;
    result->combined_output = context->combined;
    result->combined_capacity = context->hidden_capacity;
    result->routed_output = context->routed;
    result->routed_capacity = context->hidden_capacity;
    result->shared_output = context->shared;
    result->shared_capacity = context->hidden_capacity;
    result->post = context->post;
    result->post_capacity = context->residual_capacity;
    result->combination = context->combination;
    result->combination_capacity = context->residual_capacity * context->residual_capacity;
    if (rc == YVEX_OK &&
        yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA &&
        !context->workspace_ready)
        rc = runtime_moe_cuda_workspace(context, err);
    if (rc == YVEX_OK)
        rc = yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA
                 ? runtime_moe_layer_cuda(context, &job, result, err)
                 : runtime_moe_layer_cpu(context, &job, result, err);
    if (rc != YVEX_OK) {
        memset(result, 0, sizeof(*result));
        return rc;
    }
    result->encoded_bytes_read += fixed_bytes;
    if (yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA)
        result->cache_misses = result->expert_subviews_accessed;
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        unsigned long long count;
        if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR ||
            job.weights[slot].qtype >= YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP) continue;
        count = slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
                        slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN
                    ? result->router.selected_count : 1ull;
        result->qtype_counts[job.weights[slot].qtype] += count;
    }
    if (yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA &&
        device_output && context->options.evidence_level != YVEX_ATTENTION_EVIDENCE_FULL) {
        if (!yvex_moe_router_result_identity(&result->router, layer->routed_experts,
                                              result->routing_digest))
            return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                      "MoE routing selection identity derivation failed");
    } else if (!runtime_moe_digest(
                   context->routed, (size_t)layer->hidden_width * sizeof(float),
                   result->routed_digest) ||
               !runtime_moe_digest(
                   context->shared, (size_t)layer->hidden_width * sizeof(float),
                   result->shared_digest) ||
               !runtime_moe_digest(
                   result->combined_output, (size_t)layer->hidden_width * sizeof(float),
                   result->combined_digest) ||
               !yvex_moe_router_result_identity(
                   &result->router, layer->routed_experts, result->routing_digest))
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE result digest derivation failed");
    return YVEX_OK;
}
/* Purpose: allocate one immutable MoE context and all reusable session resources.
 * Inputs: model/session/options. Effects: allocates context. Failure: typed cleanup. Boundary: session. */
int yvex_runtime_moe_context_open(yvex_runtime_moe_context **out, yvex_runtime_model *model,
                                  yvex_runtime_execution_session *session,
                                  const yvex_runtime_moe_options *options, yvex_error *err)
{
    yvex_runtime_moe_context *context;
    const yvex_moe_plan_summary *summary;
    int rc;
    if (out) *out = NULL;
    if (!out || !model || !session || !options)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE context arguments are required");
    context = (yvex_runtime_moe_context *)calloc(1u, sizeof(*context));
    if (!context) return runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    if (!context->model_view || !context->session_view || context->session_view->model != model ||
        !context->model_view->binding->capabilities.moe_plan_ready ||
        !context->model_view->binding->capabilities.moe_router_ready ||
        !context->model_view->binding->capabilities.moe_routed_expert_ready ||
        !context->model_view->binding->capabilities.moe_shared_expert_ready ||
        !context->model_view->binding->capabilities.moe_block_ready ||
        pthread_mutex_init(&context->mutex, NULL) != 0) {
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE model/session ownership is invalid");
        goto fail;
    }
    context->mutex_ready = 1;
    rc = yvex_moe_plan_build(&context->plan, context->model_view->adapter->adapter_id,
                             context->model_view->adapter->adapter_version,
                             context->model_view->materialization,
                             context->model_view->descriptor,
                             context->model_view->attention, err);
    summary = yvex_moe_plan_summary_get(context->plan);
    if (rc == YVEX_OK && (!summary ||
        strcmp(summary->moe_plan_identity, context->model_view->binding->moe_plan_identity) != 0))
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "runtime binding MoE plan is stale");
    if (rc == YVEX_OK) rc = runtime_moe_buffer_plan(context, err);
    if (rc == YVEX_OK && !options->defer_cuda_workspace &&
        yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA)
        rc = runtime_moe_cuda_workspace(context, err);
    if (rc != YVEX_OK) goto fail;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    (void)yvex_runtime_moe_context_close(&context, NULL);
    return rc;
}
/* Purpose: borrow the context's immutable MoE plan.
 * Inputs: context. Effects: none. Failure: null. Boundary: borrowed context lifetime. */
const yvex_moe_plan *yvex_runtime_moe_context_plan(const yvex_runtime_moe_context *context)
{
    return context ? context->plan : NULL;
}
/* Purpose: reserve one all-request publication arena once and reuse compatible extents.
 * Inputs: exact extents. Effects: allocates staging. Failure: typed. Boundary: context-owned. */
static int runtime_moe_publication_prepare(yvex_runtime_moe_context *context,
                                           unsigned long long layers,
                                           unsigned long long tokens,
                                           yvex_error *err)
{
    unsigned long long rows, combined_count, post_count, combination_count, total, bytes;
    const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(context->plan, 0ull);
    if (!layer || !yvex_core_u64_mul(layers, tokens, &rows) ||
        !yvex_core_u64_mul(rows, layer->hidden_width, &combined_count) ||
        !yvex_core_u64_mul(rows, layer->residual_streams, &post_count) ||
        !yvex_core_u64_mul(post_count, layer->residual_streams, &combination_count))
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE publication extent overflowed");
    if (context->candidate_combined)
        return layers <= context->candidate_layers && tokens <= context->candidate_tokens
                   ? YVEX_OK
                   : runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                        "MoE request exceeds the sealed publication capacity");
    if (!yvex_core_u64_add(combined_count, post_count, &total) ||
        !yvex_core_u64_add(total, combination_count, &total) ||
        !yvex_core_u64_mul(total, sizeof(float), &bytes) ||
        (context->options.maximum_host_bytes &&
         (context->host_bytes > context->options.maximum_host_bytes ||
          bytes > context->options.maximum_host_bytes - context->host_bytes)))
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE publication exceeds host budget");
    context->candidate_combined = (float *)calloc((size_t)combined_count, sizeof(float));
    context->candidate_post = (float *)calloc((size_t)post_count, sizeof(float));
    context->candidate_combination = (float *)calloc((size_t)combination_count, sizeof(float));
    if (!context->candidate_combined || !context->candidate_post ||
        !context->candidate_combination)
        return runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE publication allocation failed");
    context->candidate_layers = layers;
    context->candidate_tokens = tokens;
    context->host_bytes += bytes;
    return YVEX_OK;
}
/* Purpose: execute every ordered layer/token and publish only after complete success.
 * Inputs: context/input/output. Effects: atomic publication. Failure: typed rollback. Boundary: MoE. */
int yvex_runtime_moe_execute(yvex_runtime_moe_context *context,
                             const yvex_moe_input *input,
                             yvex_runtime_moe_output *output,
                             yvex_runtime_moe_result *result, yvex_error *err)
{
    const yvex_moe_input_summary *input_summary = yvex_moe_input_summary_get(input);
    const yvex_moe_plan_summary *plan = yvex_moe_plan_summary_get(context ? context->plan : NULL);
    const unsigned int *tokens = yvex_moe_input_token_ids(input);
    yvex_runtime_model_failure failure;
    yvex_sha256 output_hash, route_hash, routed_hash, shared_hash, execution_hash;
    unsigned long long layer_index, token_index, output_count, post_count, combination_count;
    unsigned long long qtype_index;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const yvex_moe_layer_plan *first_layer;
    int rc, session_begun = 0, locked = 0;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !input_summary || !plan || !output || !result || !tokens ||
        yvex_moe_input_validate(input, context->plan, context->model_view->binding, err) != YVEX_OK)
        return yvex_error_is_set(err) ? yvex_error_code(err)
                                      : runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                                           "MoE execution arguments are invalid");
    first_layer = yvex_moe_plan_layer_at(context->plan, 0ull);
    if (!first_layer ||
        !yvex_core_u64_mul(plan->layer_count, input_summary->token_count, &output_count) ||
        !yvex_core_u64_mul(output_count, first_layer->hidden_width, &output_count) ||
        !yvex_core_u64_mul(plan->layer_count, input_summary->token_count, &post_count) ||
        !yvex_core_u64_mul(post_count, first_layer->residual_streams, &post_count) ||
        !yvex_core_u64_mul(post_count, first_layer->residual_streams, &combination_count) ||
        !output->combined_outputs || output->combined_capacity < output_count ||
        !output->post || output->post_capacity < post_count ||
        !output->combination || output->combination_capacity < combination_count)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE output capacity is insufficient");
    if (pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context lock failed");
    locked = 1;
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context is busy or invalidated");
    }
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    locked = 0;
    rc = runtime_moe_publication_prepare(context, plan->layer_count,
                                         input_summary->token_count, err);
    memset(&failure, 0, sizeof(failure));
    if (rc == YVEX_OK) {
        rc = yvex_runtime_session_begin(context->session, &failure, err);
        session_begun = rc == YVEX_OK;
    }
    yvex_sha256_init(&output_hash); yvex_sha256_init(&route_hash);
    yvex_sha256_init(&routed_hash); yvex_sha256_init(&shared_hash);
    for (layer_index = 0ull; rc == YVEX_OK && layer_index < plan->layer_count; ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(context->plan, layer_index);
        const float *layer_values;
        unsigned long long stride;
        rc = yvex_moe_input_layer_view(input, layer_index, &layer_values, &stride, err);
        for (token_index = 0ull; rc == YVEX_OK && token_index < input_summary->token_count;
             ++token_index) {
            yvex_moe_layer_result staged;
            unsigned long long row = layer_index * input_summary->token_count + token_index;
            rc = runtime_moe_layer_owned(context, layer_index,
                                         layer_values + token_index * stride, NULL, NULL,
                                         tokens[token_index], 1, &staged, err);
            if (rc != YVEX_OK) break;
            memcpy(context->candidate_combined + row * layer->hidden_width,
                   staged.combined_output, (size_t)layer->hidden_width * sizeof(float));
            memcpy(context->candidate_post + row * layer->residual_streams,
                   staged.post, (size_t)layer->residual_streams * sizeof(float));
            memcpy(context->candidate_combination +
                       row * layer->residual_streams * layer->residual_streams,
                   staged.combination,
                   (size_t)layer->residual_streams * layer->residual_streams * sizeof(float));
            (void)yvex_sha256_update_text(&route_hash, staged.routing_digest);
            (void)yvex_sha256_update_text(&routed_hash, staged.routed_digest);
            (void)yvex_sha256_update_text(&shared_hash, staged.shared_digest);
            result->layers_executed++;
            result->hash_router_executions +=
                layer->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID;
            result->learned_router_executions +=
                layer->router_class == YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
            result->routed_expert_executions += staged.router.selected_count;
            result->shared_expert_executions += layer->shared_experts;
            result->expert_subviews_accessed += staged.expert_subviews_accessed;
            result->encoded_bytes_read += staged.encoded_bytes_read;
            result->host_to_device_bytes += staged.host_to_device_bytes;
            result->device_to_host_bytes += staged.device_to_host_bytes;
            result->kernel_launches += staged.kernel_launches;
            result->upload_count += staged.upload_count;
            result->download_count += staged.download_count;
            result->cache_hits += staged.cache_hits;
            result->cache_misses += staged.cache_misses;
            result->stream_synchronizations += staged.stream_synchronizations;
            result->device_synchronizations += staged.device_synchronizations;
            result->device_to_device_bytes += staged.device_to_device_bytes;
            result->ingress_ns += staged.ingress_ns;
            result->routing_ns += staged.routing_ns;
            result->routed_ns += staged.routed_ns;
            result->shared_ns += staged.shared_ns;
            result->post_ns += staged.post_ns;
            result->total_ns += staged.total_ns;
            result->synchronization_ns += staged.synchronization_ns;
            for (qtype_index = 0ull; qtype_index < YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP;
                 ++qtype_index)
                result->qtype_counts[qtype_index] += staged.qtype_counts[qtype_index];
        }
    }
    if (rc == YVEX_OK)
        (void)yvex_sha256_update(&output_hash, context->candidate_combined,
                                 (size_t)output_count * sizeof(float));
    if (session_begun) {
        int finish_rc = yvex_runtime_session_finish(context->session, rc, err);
        if (rc == YVEX_OK && finish_rc != YVEX_OK) rc = finish_rc;
    }
    if (rc == YVEX_OK) {
        memcpy(output->combined_outputs, context->candidate_combined,
               (size_t)output_count * sizeof(float));
        memcpy(output->post, context->candidate_post, (size_t)post_count * sizeof(float));
        memcpy(output->combination, context->candidate_combination,
               (size_t)combination_count * sizeof(float));
        result->completed = 1;
        result->token_count = input_summary->token_count;
        yvex_runtime_identity_copy(result->input_identity, input_summary->input_identity);
    }
    if (rc == YVEX_OK && yvex_sha256_final(&output_hash, digest))
        yvex_sha256_hex(digest, result->combined_output_digest);
    if (rc == YVEX_OK && yvex_sha256_final(&route_hash, digest))
        yvex_sha256_hex(digest, result->routing_digest);
    if (rc == YVEX_OK && yvex_sha256_final(&routed_hash, digest))
        yvex_sha256_hex(digest, result->routed_digest);
    if (rc == YVEX_OK && yvex_sha256_final(&shared_hash, digest))
        yvex_sha256_hex(digest, result->shared_digest);
    if (rc == YVEX_OK) {
        yvex_sha256_init(&execution_hash);
        (void)yvex_sha256_update_text(&execution_hash, "yvex.runtime.moe-execution.v1");
        (void)yvex_sha256_update_text(&execution_hash, plan->moe_plan_identity);
        (void)yvex_sha256_update_text(&execution_hash, input_summary->input_identity);
        (void)yvex_sha256_update_u64(&execution_hash,
                                     yvex_backend_kind_of(context->session_view->backend));
        (void)yvex_sha256_update_text(&execution_hash, result->combined_output_digest);
        if (!yvex_sha256_final(&execution_hash, digest)) rc = YVEX_ERR_STATE;
        else yvex_sha256_hex(digest, result->execution_identity);
    }
    if (!locked && pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}
/* Purpose: execute one in-memory layer for the future transformer consumer.
 * Inputs: typed layer/input. Effects: publishes result. Failure: typed rollback. Boundary: token-local. */
static int runtime_moe_execute_layer_mode(yvex_runtime_moe_context *context,
                                          unsigned long long layer_index,
                                          const float *expanded_input,
                                          const yvex_device_tensor *device_input,
                                          yvex_device_tensor *device_output, unsigned int token_id,
                                          int token_id_present, int manage_session,
                                          yvex_moe_layer_result *result, yvex_error *err)
{
    yvex_runtime_model_failure failure;
    yvex_moe_layer_result staged;
    float *combined, *post, *routed, *shared, *combination;
    unsigned long long combined_capacity, routed_capacity, shared_capacity;
    unsigned long long post_capacity, combination_capacity;
    const yvex_moe_layer_plan *layer = context
        ? yvex_moe_plan_layer_at(context->plan, layer_index) : NULL;
    int rc, session_begun = 0;
    if (!context || !layer || !result || !result->combined_output ||
        result->combined_capacity < layer->hidden_width || !result->post ||
        result->post_capacity < layer->residual_streams || !result->combination ||
        result->combination_capacity < layer->residual_streams * layer->residual_streams ||
        !result->routed_output || result->routed_capacity < layer->hidden_width ||
        !result->shared_output || result->shared_capacity < layer->hidden_width)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "MoE single-layer output capacity is invalid");
    if (pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context lock failed");
    if (context->busy || context->invalidated) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context is busy or invalidated");
    }
    combined = result->combined_output;
    routed = result->routed_output;
    shared = result->shared_output;
    post = result->post;
    combination = result->combination;
    combined_capacity = result->combined_capacity;
    routed_capacity = result->routed_capacity;
    shared_capacity = result->shared_capacity;
    post_capacity = result->post_capacity;
    combination_capacity = result->combination_capacity;
    context->busy = 1;
    (void)pthread_mutex_unlock(&context->mutex);
    memset(&failure, 0, sizeof(failure));
    rc = manage_session ? yvex_runtime_session_begin(context->session, &failure, err) : YVEX_OK;
    session_begun = manage_session && rc == YVEX_OK;
    if (rc == YVEX_OK)
        rc = runtime_moe_layer_owned(context, layer_index, expanded_input,
                                     device_input, device_output, token_id,
                                     token_id_present, &staged, err);
    if (session_begun) {
        int finish_rc = yvex_runtime_session_finish(context->session, rc, err);
        if (rc == YVEX_OK && finish_rc != YVEX_OK) rc = finish_rc;
    }
    if (rc == YVEX_OK) {
        *result = staged;
        result->combined_output = combined; result->combined_capacity = combined_capacity;
        result->routed_output = routed; result->routed_capacity = routed_capacity;
        result->shared_output = shared; result->shared_capacity = shared_capacity;
        result->post = post; result->post_capacity = post_capacity;
        result->combination = combination; result->combination_capacity = combination_capacity;
        memcpy(combined, staged.combined_output, (size_t)layer->hidden_width * sizeof(float));
        memcpy(routed, staged.routed_output, (size_t)layer->hidden_width * sizeof(float));
        memcpy(shared, staged.shared_output, (size_t)layer->hidden_width * sizeof(float));
        memcpy(post, staged.post, (size_t)layer->residual_streams * sizeof(float));
        memcpy(combination, staged.combination,
               (size_t)layer->residual_streams * layer->residual_streams * sizeof(float));
    }
    if (pthread_mutex_lock(&context->mutex) == 0) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}
/* Purpose: execute one layer as a complete standalone session operation.
 * Inputs: admitted context/layer/input. Effects: owns session begin/finish and output publication.
 * Failure: aborts its session operation. Boundary: standalone component consumer. */
int yvex_runtime_moe_execute_layer(yvex_runtime_moe_context *context,
                                   unsigned long long layer_index,
                                   const float *expanded_input, unsigned int token_id,
                                   int token_id_present, yvex_moe_layer_result *result,
                                   yvex_error *err)
{
    return runtime_moe_execute_layer_mode(context, layer_index, expanded_input, NULL, NULL, token_id,
                                          token_id_present, 1, result, err);
}
/* Purpose: execute one layer inside an already acquired transformer transaction.
 * Inputs: admitted context/layer/input and session ownership held by the caller.
 * Effects: publishes token-local output without finishing the outer state transaction.
 * Failure: leaves rollback to the outer owner. Boundary: transformer composition only. */
int yvex_runtime_moe_execute_layer_borrowed(yvex_runtime_moe_context *context,
                                            unsigned long long layer_index,
                                            const float *expanded_input, unsigned int token_id,
                                            int token_id_present,
                                            yvex_moe_layer_result *result, yvex_error *err)
{
    const yvex_runtime_session_view *view = context ? context->session_view : NULL;
    yvex_runtime_session_summary summary;
    if (!view || yvex_runtime_session_summary_copy(context->session, &summary, err) != YVEX_OK ||
        !summary.busy)
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "borrowed MoE execution requires an acquired runtime session");
    return runtime_moe_execute_layer_mode(context, layer_index, expanded_input, NULL, NULL, token_id,
                                          token_id_present, 0, result, err);
}
/* Purpose: execute one borrowed layer while retaining its activation on the CUDA device.
 * Inputs: acquired session, host evidence view, and matching device input/output tensors.
 * Effects: publishes host evidence and the next device-resident residual atomically.
 * Failure: leaves outer rollback and device-output publication to the transformer owner.
 * Boundary: no session finish, attention state mutation, or CPU numerical fallback. */
int yvex_runtime_moe_execute_layer_device_borrowed(
    yvex_runtime_moe_context *context, unsigned long long layer_index,
    const float *expanded_input, const yvex_device_tensor *device_input,
    yvex_device_tensor *device_output, unsigned int token_id, int token_id_present,
    yvex_moe_layer_result *result, yvex_error *err)
{
    yvex_runtime_session_summary summary;
    if (!context || !device_input || !device_output ||
        yvex_runtime_session_summary_copy(context->session, &summary, err) != YVEX_OK ||
        !summary.busy)
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "device MoE execution requires an acquired session");
    return runtime_moe_execute_layer_mode(
        context, layer_index, expanded_input, device_input, device_output,
        token_id, token_id_present, 0, result, err);
}
/* Purpose: reset transient publication bytes without changing immutable plans or session state.
 * Inputs: idle context. Effects: clears staging. Failure: typed. Boundary: no KV mutation. */
int yvex_runtime_moe_context_reset(yvex_runtime_moe_context *context, yvex_error *err)
{
    const yvex_moe_layer_plan *layer;
    if (!context || pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE context reset is invalid");
    if (context->busy) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "busy MoE context cannot reset");
    }
    layer = yvex_moe_plan_layer_at(context->plan, 0ull);
    if (context->candidate_combined && layer)
        memset(context->candidate_combined, 0,
               (size_t)context->candidate_layers * context->candidate_tokens *
                   layer->hidden_width * sizeof(float));
    context->execution_count = 0ull;
    (void)pthread_mutex_unlock(&context->mutex);
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: release one context after detaching its exact backend workspace.
 * Inputs: context owner. Effects: frees resources. Failure: typed/retryable. Boundary: session. */
int yvex_runtime_moe_context_close(yvex_runtime_moe_context **context, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->busy)
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "busy MoE context cannot close");
    if ((*context)->workspace_owned) {
        yvex_backend_workspace_detach((*context)->session_view->backend);
        rc = yvex_backend_tensor_release((*context)->session_view->backend,
                                         &(*context)->device_workspace, err);
        if (rc != YVEX_OK) return rc;
        (*context)->workspace_owned = 0;
    }
    for (index = 0ull; index < YVEX_MOE_WEIGHT_COUNT; ++index)
        free((*context)->fixed[index].data);
    for (index = 0ull; index < 3ull; ++index) free((*context)->selected[index].data);
    free((*context)->scratch);
    free((*context)->candidate_combined);
    free((*context)->candidate_post);
    free((*context)->candidate_combination);
    yvex_moe_plan_close(&(*context)->plan);
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Purpose: adapt the typed cleanup lease to the opaque MoE context lifecycle. */
static int runtime_moe_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_moe_context_close((yvex_runtime_moe_context **)opaque, err);
}
/* Purpose: project one typed operator refusal without reclassifying its domain status. */
static void runtime_moe_operator_refuse(yvex_moe_operator_result *result,
                                        const yvex_error *err)
{
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err) ? yvex_error_message(err)
                                                     : "MoE execution refused");
}
/* Purpose: execute the production MoE API through one complete operator-owned lifecycle.
 * Inputs: operator request. Effects: opens/executes/closes. Failure: typed. Boundary: no rendering. */
int yvex_runtime_moe_operator_execute(const yvex_moe_operator_request *request,
                                      yvex_moe_operator_result *result,
                                      yvex_runtime_cleanup_lease **retained_cleanup,
                                      yvex_error *err)
{
    yvex_runtime_model_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_moe_options options = {0};
    yvex_moe_input_limits limits = {0};
    yvex_runtime_model_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_runtime_model *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_moe_context *context = NULL;
    yvex_moe_input *input = NULL;
    yvex_runtime_moe_output output = {0};
    const yvex_runtime_model_view *model_view;
    const yvex_moe_plan_summary *plan;
    const yvex_moe_layer_plan *first_layer;
    const yvex_moe_input_summary *input_summary;
    yvex_error primary;
    unsigned long long rows, combined_count, post_count, combination_count, total;
    int rc, cleanup_rc, adopted = 0;
    if (result) memset(result, 0, sizeof(*result));
    if (!request || !result || !retained_cleanup || *retained_cleanup ||
        !request->target || !request->artifact_path || !request->runtime_binding_path ||
        !request->input_path || (request->backend != YVEX_BACKEND_KIND_CPU &&
                                 request->backend != YVEX_BACKEND_KIND_CUDA)) {
        rc = runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                "complete typed MoE operator arguments are required");
        if (result) runtime_moe_operator_refuse(result, err);
        return rc;
    }
    yvex_core_text_copy(result->command, sizeof(result->command), "graph moe execute");
    yvex_core_text_copy(result->target, sizeof(result->target), request->target);
    yvex_core_text_copy(result->backend, sizeof(result->backend),
                        request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
    model_request.artifact_path = request->artifact_path;
    model_request.runtime_binding_path = request->runtime_binding_path;
    model_request.target_id = request->target;
    model_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.backend = request->backend;
    session_request.maximum_host_bytes = request->maximum_host_bytes;
    session_request.maximum_device_bytes = request->maximum_device_bytes;
    rc = yvex_runtime_cleanup_lease_acquire(&cleanup, &model_request, &session_request,
                                            &model, &session, &failure, err);
    limits.maximum_file_bytes = request->maximum_host_bytes
                                    ? request->maximum_host_bytes : 1ull << 30u;
    if (rc == YVEX_OK)
        rc = yvex_moe_input_open_file(&input, request->input_path, &limits, err);
    options.maximum_host_bytes = request->maximum_host_bytes;
    options.maximum_device_bytes = request->maximum_device_bytes;
    options.cancel_requested = request->cancel_requested;
    options.cancel_context = request->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_moe_context_open(&context, model, session, &options, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(cleanup, context, runtime_moe_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    model_view = yvex_runtime_model_view_get(model);
    plan = yvex_moe_plan_summary_get(yvex_runtime_moe_context_plan(context));
    first_layer = yvex_moe_plan_layer_at(yvex_runtime_moe_context_plan(context), 0ull);
    input_summary = yvex_moe_input_summary_get(input);
    if (rc == YVEX_OK &&
        (!model_view || !plan || !first_layer || !input_summary ||
         !yvex_core_u64_mul(plan->layer_count, input_summary->token_count, &rows) ||
         !yvex_core_u64_mul(rows, first_layer->hidden_width, &combined_count) ||
         !yvex_core_u64_mul(rows, first_layer->residual_streams, &post_count) ||
         !yvex_core_u64_mul(post_count, first_layer->residual_streams,
                            &combination_count) ||
         !yvex_core_u64_add(combined_count, post_count, &total) ||
         !yvex_core_u64_add(total, combination_count, &total) || total > SIZE_MAX / sizeof(float)))
        rc = runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE operator output extent overflowed");
    if (rc == YVEX_OK) {
        output.combined_outputs = (float *)calloc((size_t)combined_count, sizeof(float));
        output.post = (float *)calloc((size_t)post_count, sizeof(float));
        output.combination = (float *)calloc((size_t)combination_count, sizeof(float));
        output.combined_capacity = combined_count;
        output.post_capacity = post_count;
        output.combination_capacity = combination_count;
        if (!output.combined_outputs || !output.post || !output.combination)
            rc = runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE operator output allocation failed");
    }
    if (rc == YVEX_OK)
        rc = yvex_runtime_moe_execute(context, input, &output, &result->execution, err);
    if (rc == YVEX_OK) {
        yvex_core_text_copy(result->command, sizeof(result->command), "graph moe execute");
        yvex_core_text_copy(result->target, sizeof(result->target), request->target);
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->adapter->family_name);
        yvex_core_text_copy(result->backend, sizeof(result->backend),
                            request->backend == YVEX_BACKEND_KIND_CUDA ? "cuda" : "cpu");
        yvex_runtime_identity_copy(result->artifact_identity,
                                   model_view->binding->artifact_identity);
        yvex_runtime_identity_copy(result->runtime_binding_identity,
                                   model_view->binding->identity);
        yvex_runtime_identity_copy(result->runtime_descriptor_identity,
                                   model_view->binding->runtime_descriptor_identity);
        yvex_runtime_identity_copy(result->runtime_numeric_identity,
                                   model_view->binding->runtime_numeric_identity);
        yvex_runtime_identity_copy(result->moe_plan_identity, plan->moe_plan_identity);
        result->layer_count = plan->layer_count;
        result->token_count = input_summary->token_count;
        result->hash_router_count = plan->hash_router_layer_count;
        result->learned_router_count = plan->learned_router_layer_count;
        result->routed_experts = plan->routed_experts;
        result->shared_experts = plan->shared_experts;
        result->experts_per_token = plan->experts_per_token;
        result->moe_plan_ready = result->moe_router_ready = 1;
        result->moe_routed_expert_ready = result->moe_shared_expert_ready = 1;
        result->moe_block_ready = 1;
    }
    free(output.combined_outputs);
    free(output.post);
    free(output.combination);
    yvex_moe_input_close(&input);
    primary = err ? *err : (yvex_error){0};
    if (!adopted && context) {
        cleanup_rc = yvex_runtime_moe_context_close(&context, err);
        if (rc == YVEX_OK && cleanup_rc != YVEX_OK) rc = cleanup_rc;
    }
    cleanup_rc = yvex_runtime_cleanup_lease_close(&cleanup, err);
    if (cleanup_rc != YVEX_OK) rc = cleanup_rc;
    else if (rc != YVEX_OK && err) *err = primary;
    if (cleanup) *retained_cleanup = cleanup;
    if (rc == YVEX_OK) {
        result->completed = 1;
        yvex_core_text_copy(result->status, sizeof(result->status), "complete");
        yvex_error_clear(err);
    } else {
        runtime_moe_operator_refuse(result, err);
    }
    return rc;
}
