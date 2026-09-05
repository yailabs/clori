/*
 * Execute complete MoE blocks through reusable session-owned resources.
 *
 * One context binds one model/session and reads only fixed or selected expert ranges. Runtime
 * coordinates admitted graph/backend owners without reconstructing compiler or family truth.
 */
#include <yvex/internal/moe.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/runtime.h>
#include "src/runtime/private.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    unsigned char *data;
    unsigned long long capacity;
} moe_byte_buffer;
struct yvex_runtime_moe_context {
    yvex_model_engine *model;
    yvex_runtime_execution_session *session;
    const yvex_model_engine_view *model_view;
    const yvex_runtime_session_view *session_view;
    const yvex_moe_plan *plan;
    yvex_runtime_moe_options options;
    moe_byte_buffer fixed[YVEX_MOE_WEIGHT_COUNT];
    moe_byte_buffer selected[3];
    float *scratch, *normalized, *post, *combination;
    float *expert, *routed, *shared, *combined;
    unsigned long long hidden_capacity, residual_capacity;
    unsigned long long device_workspace_bytes;
    yvex_moe_device_completion_slot *pending_layers;
    unsigned long long pending_layer_capacity, pending_layer_count;
    unsigned long long pending_first_layer;
    int pending_active;
    float *candidate_combined, *candidate_post, *candidate_combination;
    unsigned long long candidate_tokens, candidate_layers;
    yvex_device_tensor *device_workspace;
    unsigned long long host_bytes, execution_count;
    int workspace_owned, workspace_ready, busy, invalidated;
    pthread_mutex_t mutex;
    int mutex_ready;
};

static int runtime_moe_refuse(yvex_error *err, yvex_status status, const char *reason)
{
    yvex_error_set(err, status, "runtime.moe", reason);
    return status;
}

static const yvex_materialized_tensor_binding *runtime_moe_binding(
    const yvex_runtime_moe_context *context, const yvex_moe_layer_plan *layer,
    yvex_moe_weight_slot slot)
{
    unsigned long long tensor_id = layer && slot < YVEX_MOE_WEIGHT_COUNT
                                       ? layer->tensor_ids[slot] : YVEX_MOE_NO_TENSOR;
    return tensor_id == YVEX_MOE_NO_TENSOR ? NULL
        : yvex_materialization_session_tensor_at(context->model_view->materialization, tensor_id);
}

static int runtime_moe_activation(
    const yvex_runtime_moe_context *context,
    const yvex_materialized_tensor_binding *binding,
    unsigned long long population,
    yvex_execution_activation_class *activation,
    yvex_engine_implementation *implementation, yvex_error *err)
{
    const yvex_physical_execution_ir *physical = context && context->model_view
        ? context->model_view->physical_execution : NULL;
    const yvex_physical_execution_decision *decision = physical && binding
        ? yvex_physical_execution_ir_decision_at(physical, binding->tensor_id) : NULL;
    const yvex_engine_implementation_record *record =
        context && context->session && binding
            ? runtime_specialization_tensor(
                  context->session->specialization, physical, binding->tensor_id)
            : NULL;
    const yvex_runtime_execution_profile *profile =
        context ? context->options.execution_profile : NULL;
    int degraded = profile &&
        profile->moe_resolution == YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    if (!decision || !record || !population || !activation || !implementation)
        return runtime_moe_refuse(
            err, YVEX_ERR_INVALID_ARG, "MoE physical activation owners are unavailable");
    if (decision->terminal_tensor_id != binding->tensor_id ||
        decision->role != binding->role || decision->canonical_qtype != binding->qtype ||
        decision->canonical_row_width != binding->row_width ||
        decision->canonical_row_count != binding->row_count)
        return runtime_moe_refuse(
            err, YVEX_ERR_STATE, "MoE physical execution decision is stale");
    *activation = degraded ? record->fallback_activation : record->activation;
    /* Runtime validates the compiled operation but never promotes a total row population into
     * expert-compatible width. Wide regimes are selected only from the sealed worklist policy
     * after routing has produced real same-expert buckets. */
    *implementation = degraded ? record->fallback_implementation : record->implementation;
    return YVEX_OK;
}

static int runtime_moe_worklist_contract(
    yvex_runtime_moe_context *context, const yvex_moe_layer_plan *layer,
    const yvex_moe_row_batch *rows, const yvex_runtime_session_summary *session,
    yvex_execution_batch_source *source, yvex_execution_batch *batch,
    yvex_expert_worklist_policy *policy, yvex_error *err)
{
    const yvex_physical_execution_ir *physical = context->model_view->physical_execution;
    const yvex_attention_state_provider *provider =
        layer->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
            ? context->session_view->draft_attention_state_provider
            : context->session_view->attention_state_provider;
    const yvex_engine_implementation_record *decisions[3];
    yvex_model_engine_summary model;
    yvex_graph_attention_state_summary state = {0};
    unsigned long long slot, next_execution;
    if (!physical || !provider || !provider->summary || !source ||
        !rows->execution_source_count || !rows->execution_sources ||
        !rows->execution_rows ||
        yvex_model_engine_summary_copy(context->model, &model, err) != YVEX_OK)
        return yvex_error_code(err) == YVEX_OK
                   ? runtime_moe_refuse(err, YVEX_ERR_STATE,
                                        "expert worklist identity owners are unavailable")
                   : yvex_error_code(err);
    memset(batch, 0, sizeof(*batch));
    batch->schema_version = YVEX_EXECUTION_BATCH_SCHEMA_V2;
    batch->provenance = rows->provenance;
    batch->phase = rows->phase;
    batch->row_count = rows->row_count;
    batch->source_count = rows->execution_source_count;
    batch->engine_generation = model.engine_generation;
    if (rows->execution_source_count == 1ull) {
        if (provider->summary(provider->context, &state, err) != YVEX_OK ||
            !yvex_core_u64_add(session->execution_count, 1ull,
                               &next_execution))
            return yvex_error_code(err) == YVEX_OK
                       ? runtime_moe_refuse(
                             err, YVEX_ERR_STATE,
                             "expert worklist source generation is unavailable")
                       : yvex_error_code(err);
        *source = rows->execution_sources[0];
        source->execution_generation = next_execution;
        source->state_generation = state.generation;
        batch->sources = source;
    } else {
        batch->sources = rows->execution_sources;
    }
    batch->rows = rows->execution_rows;
    yvex_runtime_identity_copy(batch->execution_profile_identity,
                               rows->execution_profile_identity);
    yvex_runtime_identity_copy(batch->operation_identity, layer->layer_identity);
    if (yvex_execution_batch_seal(batch, err) != YVEX_OK) return yvex_error_code(err);
    for (slot = 0ull; slot < 3ull; ++slot) {
        const yvex_materialized_tensor_binding *binding = runtime_moe_binding(
            context, layer, (yvex_moe_weight_slot)(YVEX_MOE_WEIGHT_ROUTED_GATE + slot));
        decisions[slot] = binding
                              ? runtime_specialization_tensor(
                                    context->session->specialization,
                                    physical, binding->tensor_id)
                              : NULL;
        if (!decisions[slot] ||
            decisions[slot]->schema_version != YVEX_ENGINE_SPECIALIZATION_SCHEMA_V1 ||
            !decisions[slot]->worklist_width_mask ||
            (slot && (decisions[slot]->worklist_width_mask !=
                          decisions[0]->worklist_width_mask ||
                      decisions[slot]->matrix_tile_minimum !=
                          decisions[0]->matrix_tile_minimum ||
                      decisions[slot]->implementation !=
                          decisions[0]->implementation)))
            return runtime_moe_refuse(
                err, YVEX_ERR_STATE,
                "compiled routed-expert worklist policies disagree");
    }
    memset(policy, 0, sizeof(*policy));
    policy->schema_version = YVEX_EXPERT_WORKLIST_POLICY_SCHEMA_V2;
    policy->supported_width_mask = decisions[0]->worklist_width_mask;
    policy->matrix_tile_minimum = decisions[0]->matrix_tile_minimum;
    policy->row_implementation = decisions[0]->implementation;
    policy->matrix_implementation = decisions[0]->matrix_tile_minimum
                                      ? YVEX_ENGINE_IMPLEMENTATION_DEVICE_MATRIX_TILE
                                      : YVEX_ENGINE_IMPLEMENTATION_COUNT;
    return yvex_expert_worklist_policy_seal(policy, err);
}

static int runtime_moe_row_bytes(const yvex_materialized_tensor_binding *binding,
                                 unsigned long long *out)
{
    return binding && out && binding->row_count &&
           binding->encoded_bytes % binding->row_count == 0ull &&
           ((*out = binding->encoded_bytes / binding->row_count) != 0ull);
}

static int runtime_moe_weight(yvex_moe_weight_view *out,
                              const yvex_materialized_tensor_binding *binding,
                              const unsigned char *bytes, unsigned long long encoded_bytes,
                              unsigned long long row_count, unsigned long long expert,
                              unsigned long long device_address,
                              yvex_execution_layout_class layout,
                              yvex_execution_activation_class activation,
                              yvex_engine_implementation implementation,
                              yvex_error *err)
{
    unsigned long long row_bytes, expected;
    if (!out || !binding || !bytes || !encoded_bytes || !row_count ||
        !runtime_moe_row_bytes(binding, &row_bytes) ||
        !yvex_core_u64_mul(row_bytes, row_count, &expected) ||
        expected != encoded_bytes || expected > binding->encoded_bytes ||
        encoded_bytes > (unsigned long long)SIZE_MAX ||
        layout > YVEX_EXECUTION_LAYOUT_EXPERT_MAJOR ||
        implementation >= YVEX_ENGINE_IMPLEMENTATION_COUNT)
        return runtime_moe_refuse(err, YVEX_ERR_FORMAT, "MoE weight view geometry is invalid");
    memset(out, 0, sizeof(*out));
    out->tensor_id = binding->tensor_id;
    out->expert_index = expert;
    out->role = binding->role;
    out->qtype = binding->qtype;
    out->layout = layout;
    out->activation = activation;
    out->implementation = implementation;
    out->encoded = bytes;
    out->encoded_bytes = (size_t)encoded_bytes;
    out->row_bytes = row_bytes;
    out->row_width = binding->row_width;
    out->row_count = row_count;
    out->device_address = device_address;
    return YVEX_OK;
}

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
/*
 * Borrow one CUDA-addressable resident range or copy it for CPU execution.
 *
 * Admitted binding subrange and the CPU fallback buffer owned by this context. CUDA execution
 * never restages model-resident bytes through a host scratch copy.
 */
static int runtime_moe_access(yvex_runtime_moe_context *context,
                              const yvex_materialized_tensor_binding *binding,
                              unsigned long long offset, unsigned long long bytes,
                              moe_byte_buffer *buffer, const unsigned char **data,
                              unsigned long long *device_address,
                              yvex_execution_layout_class *layout, yvex_error *err)
{
    const unsigned char *resident = NULL;
    unsigned long long resident_bytes = 0ull;
    if (data) *data = NULL;
    if (device_address) *device_address = 0ull;
    if (layout) *layout = YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
    if (!context || !binding || !bytes || !data || !device_address ||
        !layout ||
        offset > binding->encoded_bytes || bytes > binding->encoded_bytes - offset)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                  "MoE encoded subrange is invalid");
    if (yvex_backend_kind_of(context->session_view->backend) != YVEX_BACKEND_KIND_CUDA) {
        int rc = runtime_moe_read(context, binding, offset, bytes, buffer, err);
        if (rc == YVEX_OK) {
            *data = buffer->data;
        }
        return rc;
    }
    if (yvex_runtime_private_residency_execution_view(
            context->model_view->residency, binding, &resident, &resident_bytes,
            layout, err) != YVEX_OK)
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "MoE resident binding range is unavailable");
    if (offset > resident_bytes || bytes > resident_bytes - offset)
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

static int runtime_moe_buffer_plan(yvex_runtime_moe_context *context, yvex_error *err)
{
    const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(context->plan);
    const yvex_backend_moe_operations *backend_operations =
        yvex_backend_moe_operations_get(context->session_view->backend);
    unsigned long long layer_index, slot, total = sizeof(*context), scratch_count, scratch_bytes;
    int cuda = yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA;
    for (layer_index = 0ull; layer_index < summary->layer_count; ++layer_index) {
        const yvex_moe_layer_plan *layer = yvex_moe_plan_layer_at(context->plan, layer_index);
        if (layer->hidden_width > context->hidden_capacity)
            context->hidden_capacity = layer->hidden_width;
        if (layer->residual_streams > context->residual_capacity)
            context->residual_capacity = layer->residual_streams;
        if (backend_operations) {
            unsigned long long workspace_bytes;
            if (backend_operations->workspace_required(
                    layer, context->options.row_capacity, &workspace_bytes, err) != YVEX_OK)
                return yvex_error_code(err) == YVEX_OK
                           ? runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                               "MoE row capacity overflowed")
                           : yvex_error_code(err);
            if (workspace_bytes > context->device_workspace_bytes)
                context->device_workspace_bytes = workspace_bytes;
        }
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
    if (cuda) {
        const yvex_moe_plan_summary *target =
            yvex_moe_plan_summary_get(context->model_view->moe);
        const yvex_moe_plan_summary *draft =
            yvex_moe_plan_summary_get(context->model_view->draft_moe);
        unsigned long long layers = target ? target->layer_count : 0ull;
        if (draft && draft->layer_count > layers) layers = draft->layer_count;
        if (!layers) goto overflow;
        context->pending_layer_capacity = layers;
    }
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

static int runtime_moe_cuda_workspace(yvex_runtime_moe_context *context, yvex_error *err)
{
    yvex_runtime_session_summary session_summary;
    yvex_backend_tensor_desc descriptor = {0};
    int rc;
    if (yvex_runtime_session_summary_copy(context->session, &session_summary, err) != YVEX_OK)
        return yvex_error_code(err);
    if (session_summary.device_workspace_bytes) {
        if (session_summary.device_workspace_bytes < context->device_workspace_bytes)
            return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                      "existing CUDA workspace is too small for MoE");
        context->workspace_ready = 1;
        return YVEX_OK;
    }
    if (!context->device_workspace_bytes)
        return runtime_moe_refuse(err, YVEX_ERR_UNSUPPORTED,
                                  "CUDA MoE workspace geometry is unavailable");
    if (context->options.maximum_device_bytes &&
        context->device_workspace_bytes > context->options.maximum_device_bytes)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS, "MoE CUDA workspace exceeds its budget");
    descriptor.name = "moe_workspace";
    descriptor.dtype = YVEX_DTYPE_F32;
    descriptor.rank = 1u;
    descriptor.dims[0] = context->device_workspace_bytes / sizeof(float);
    descriptor.bytes = context->device_workspace_bytes;
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

static int runtime_moe_load_layer(yvex_runtime_moe_context *context,
                                  const yvex_moe_layer_plan *layer,
                                  unsigned int token_id, unsigned long long input_rows,
                                  int complete_router_table,
                                  yvex_moe_layer_job *job,
                                  unsigned long long *bytes_read, yvex_error *err)
{
    unsigned long long slot, routed_population;
    memset(job, 0, sizeof(*job));
    job->layer = layer;
    job->token_id = token_id;
    job->token_id_present = 1;
    job->cancel_requested = context->options.cancel_requested;
    job->cancel_context = context->options.cancel_context;
    job->evidence_level = context->options.evidence_level;
    job->eager_execution = context->options.eager_execution;
    if (!input_rows || !yvex_core_u64_mul(input_rows, layer->experts_per_token,
                                          &routed_population))
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                  "MoE row-kernel population overflowed");
    for (slot = 0ull; slot < YVEX_MOE_WEIGHT_COUNT; ++slot) {
        const yvex_materialized_tensor_binding *binding;
        const unsigned char *data = NULL;
        unsigned long long device_address = 0ull;
        yvex_execution_layout_class layout = YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
        unsigned long long offset = 0ull, bytes, rows, population = 1ull;
        yvex_execution_activation_class activation;
        yvex_engine_implementation implementation;
        if (layer->tensor_ids[slot] == YVEX_MOE_NO_TENSOR) continue;
        binding = runtime_moe_binding(context, layer, (yvex_moe_weight_slot)slot);
        if (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE && slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN)
            population = routed_population;
        else if (slot >= YVEX_MOE_WEIGHT_SHARED_GATE && slot <= YVEX_MOE_WEIGHT_SHARED_DOWN)
            population = input_rows;
        if (runtime_moe_activation(
                context, binding, population, &activation, &implementation, err) != YVEX_OK)
            return yvex_error_code(err);
        if (slot >= YVEX_MOE_WEIGHT_ROUTED_GATE &&
            slot <= YVEX_MOE_WEIGHT_ROUTED_DOWN) {
            if (yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA &&
                runtime_moe_access(context, binding, 0ull, binding->encoded_bytes,
                                   &context->fixed[slot], &data, &device_address,
                                   &layout, err) != YVEX_OK)
                return yvex_error_code(err);
            job->weights[slot].tensor_id = binding->tensor_id;
            job->weights[slot].expert_index = YVEX_MOE_NO_TENSOR;
            job->weights[slot].role = binding->role;
            job->weights[slot].qtype = binding->qtype;
            job->weights[slot].layout = layout;
            job->weights[slot].activation = activation;
            job->weights[slot].implementation = implementation;
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
            if (!complete_router_table &&
                (unsigned long long)token_id >= layer->hash_table_rows)
                return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                          "MoE token ID exceeds hash router table");
            if (!complete_router_table) {
                bytes = binding->encoded_bytes / binding->row_count;
                offset = (unsigned long long)token_id * bytes;
                rows = 1ull;
            }
        }
        if (runtime_moe_access(context, binding, offset, bytes, &context->fixed[slot],
                               &data, &device_address, &layout, err) != YVEX_OK ||
            runtime_moe_weight(&job->weights[slot], binding, data, bytes, rows,
                               YVEX_MOE_NO_TENSOR, device_address, layout, activation,
                               implementation, err) != YVEX_OK)
            return yvex_error_code(err);
        *bytes_read += bytes;
    }
    return YVEX_OK;
}

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
        yvex_execution_layout_class layout = YVEX_EXECUTION_LAYOUT_CANONICAL_ROW;
        unsigned long long offset, rows;
        yvex_execution_activation_class activation;
        yvex_engine_implementation implementation;
        memset(&failure, 0, sizeof(failure));
        if (yvex_materialization_session_expert_subview(
                context->model_view->materialization, binding, expert, &subview,
                &failure, err) != YVEX_OK)
            return yvex_error_code(err);
        offset = subview.absolute_offset - binding->absolute_offset;
        rows = binding->row_count / binding->expert_count;
        if (runtime_moe_activation(
                context, binding, 1ull, &activation, &implementation, err) != YVEX_OK)
            return yvex_error_code(err);
        if (runtime_moe_access(context, binding, offset, subview.encoded_bytes,
                               &context->selected[index], &data, &device_address,
                               &layout, err) != YVEX_OK ||
            runtime_moe_weight(&views[index], binding, data, subview.encoded_bytes,
                               rows, expert, device_address, layout, activation,
                               implementation, err) != YVEX_OK)
            return yvex_error_code(err);
        *bytes_read += subview.encoded_bytes;
    }
    return YVEX_OK;
}

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

static int runtime_moe_round(float *values, unsigned long long count)
{
    unsigned long long index;
    for (index = 0ull; index < count; ++index) {
        if (!isfinite(values[index])) return 0;
        values[index] = yvex_quant_bf16_decode(yvex_quant_bf16_encode(values[index]));
    }
    return 1;
}

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
                                     context->normalized, result->router.selected_weights[rank], context->expert, err);
        if (rc == YVEX_OK)
            for (lane = 0ull; lane < layer->hidden_width; ++lane)
                context->routed[lane] += context->expert[lane];
    }
    if (rc == YVEX_OK)
        rc = yvex_moe_expert_cpu(layer, &job->weights[YVEX_MOE_WEIGHT_SHARED_GATE],
                                 &job->weights[YVEX_MOE_WEIGHT_SHARED_UP],
                                 &job->weights[YVEX_MOE_WEIGHT_SHARED_DOWN],
                                 context->normalized, 1.0f, context->shared, err);
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
    rc = runtime_moe_load_layer(context, layer, token_id, 1ull, 0, &job,
                                &fixed_bytes, err);
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
    if (result->memory.activation_bytes || result->memory.temporary_bytes) {
        unsigned long long activation = result->memory.activation_bytes;
        unsigned long long temporary = result->memory.temporary_bytes;
        memset(&result->memory, 0, sizeof(result->memory));
        if (yvex_execution_memory_facts_add(
                &result->memory, result->encoded_bytes_read, 0ull, activation,
                temporary, 1ull, 0ull, err) != YVEX_OK)
            return yvex_error_code(err);
    } else if (yvex_execution_memory_facts_add(
                   &result->memory, 0ull, 0ull, 0ull, 0ull,
                   0ull, 1ull, err) != YVEX_OK) return yvex_error_code(err);
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

int yvex_runtime_moe_context_open(yvex_runtime_moe_context **out, yvex_model_engine *model,
                                  yvex_runtime_execution_session *session,
                                  const yvex_runtime_moe_options *options,
                                  unsigned long long *device_workspace_bytes,
                                  yvex_error *err)
{
    yvex_runtime_moe_context *context;
    const yvex_moe_plan_summary *summary;
    int rc;
    if (out) *out = NULL;
    if (device_workspace_bytes) *device_workspace_bytes = 0ull;
    if (!out || !model || !session || !options ||
        (options->eager_execution != 0 && options->eager_execution != 1) ||
        (options->tensor_scope != YVEX_TENSOR_SCOPE_GLOBAL &&
         options->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT))
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE context arguments are required");
    context = (yvex_runtime_moe_context *)calloc(1u, sizeof(*context));
    if (!context) return runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_model_engine_view_get(model);
    context->session_view = yvex_runtime_session_view_get(session);
    context->options = *options;
    if (!context->options.row_capacity) context->options.row_capacity = 1ull;
    if (!context->model_view || !context->session_view || context->session_view->engine != model ||
        !context->model_view->binding->capabilities.moe_plan_ready ||
        !context->model_view->binding->capabilities.moe_router_ready ||
        !context->model_view->binding->capabilities.moe_routed_expert_ready ||
        !context->model_view->binding->capabilities.moe_shared_expert_ready ||
        !context->model_view->binding->capabilities.moe_block_ready ||
        (options->execution_profile &&
         !runtime_execution_profile_matches(options->execution_profile,
                                            model, session)) ||
        pthread_mutex_init(&context->mutex, NULL) != 0) {
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE model/session ownership is invalid");
        goto fail;
    }
    context->mutex_ready = 1;
    context->plan = options->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                        ? context->model_view->draft_moe
                        : context->model_view->moe;
    summary = yvex_moe_plan_summary_get(context->plan);
    rc = YVEX_OK;
    if (!summary ||
        strcmp(summary->moe_plan_identity,
               options->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                   ? context->model_view->binding->draft_moe_plan_identity
                   : context->model_view->binding->moe_plan_identity) != 0)
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "runtime binding MoE plan is stale");
    if (rc == YVEX_OK) rc = runtime_moe_buffer_plan(context, err);
    if (rc == YVEX_OK && !options->defer_device_workspace &&
        yvex_backend_kind_of(context->session_view->backend) == YVEX_BACKEND_KIND_CUDA)
        rc = runtime_moe_cuda_workspace(context, err);
    if (rc != YVEX_OK) goto fail;
    if (device_workspace_bytes) *device_workspace_bytes = context->device_workspace_bytes;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;
fail:
    (void)yvex_runtime_moe_context_close(&context, NULL);
    return rc;
}
/*
 * Borrow the context's immutable MoE plan.
 *
 * Borrowed context lifetime.
 */
const yvex_moe_plan *yvex_runtime_moe_context_plan(const yvex_runtime_moe_context *context)
{
    return context ? context->plan : NULL;
}

int yvex_runtime_moe_host_workspace_bind(yvex_runtime_moe_context *context,
                                         yvex_error *err)
{
    unsigned long long bytes;
    void *slots = NULL;
    if (!context || !context->pending_layer_capacity ||
        !yvex_core_u64_mul(context->pending_layer_capacity,
                           sizeof(*context->pending_layers), &bytes) ||
        yvex_backend_host_workspace_reserve(
            context->session_view->backend, bytes,
            _Alignof(yvex_moe_device_completion_slot), &slots) !=
            YVEX_BACKEND_RESIDENT_HIT)
        return runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                  "MoE completion ledger is not admitted");
    context->pending_layers = slots;
    yvex_error_clear(err);
    return YVEX_OK;
}

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
/*
 * Execute every ordered layer/token and publish only after complete success.
 *
 * Atomic publication. Typed rollback.
 */
int yvex_runtime_moe_execute(yvex_runtime_moe_context *context,
                             const yvex_moe_input *input,
                             yvex_runtime_moe_output *output,
                             yvex_runtime_moe_result *result, yvex_error *err)
{
    const yvex_moe_input_summary *input_summary = yvex_moe_input_summary_get(input);
    const yvex_moe_plan_summary *plan = yvex_moe_plan_summary_get(context ? context->plan : NULL);
    const unsigned int *tokens = yvex_moe_input_token_ids(input);
    yvex_model_engine_failure failure;
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
            if (yvex_execution_memory_facts_merge(
                    &result->memory, &staged.memory, err) != YVEX_OK) {
                rc = yvex_error_code(err);
                break;
            }
            result->host_to_device_bytes += staged.host_to_device_bytes;
            result->device_to_host_bytes += staged.device_to_host_bytes;
            result->kernel_launches += staged.kernel_launches;
            result->upload_count += staged.upload_count;
            result->download_count += staged.download_count;
            result->cache_hits += staged.cache_hits;
            result->cache_misses += staged.cache_misses;
            result->queue_synchronizations += staged.queue_synchronizations;
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
/*
 * Execute one in-memory layer for the future transformer consumer.
 *
 * Typed rollback.
 */
static int runtime_moe_execute_layer_mode(yvex_runtime_moe_context *context,
                                          unsigned long long layer_index,
                                          const float *expanded_input,
                                          const yvex_device_tensor *device_input,
                                          yvex_device_tensor *device_output, unsigned int token_id,
                                          int token_id_present, int manage_session,
                                          yvex_moe_layer_result *result, yvex_error *err)
{
    yvex_model_engine_failure failure;
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

int yvex_runtime_moe_execute_layer(yvex_runtime_moe_context *context,
                                   unsigned long long layer_index,
                                   const float *expanded_input, unsigned int token_id,
                                   int token_id_present, yvex_moe_layer_result *result,
                                   yvex_error *err)
{
    return runtime_moe_execute_layer_mode(context, layer_index, expanded_input, NULL, NULL, token_id,
                                          token_id_present, 1, result, err);
}
static int runtime_moe_batch_account(yvex_moe_row_batch_result *batch,
                                     const yvex_moe_layer_result *row,
                                     unsigned char *seen,
                                     unsigned long long seen_count)
{
    unsigned long long rank;
    batch->row_expert_pairs += row->router.selected_count;
    batch->grouped_expert_operations += row->router.selected_count;
    batch->expert_subviews_accessed += row->expert_subviews_accessed;
    batch->encoded_bytes_read += row->encoded_bytes_read;
    batch->h2d_bytes += row->host_to_device_bytes;
    batch->d2h_bytes += row->device_to_host_bytes;
    batch->d2d_bytes += row->device_to_device_bytes;
    batch->kernel_launches += row->kernel_launches;
    batch->upload_count += row->upload_count;
    batch->download_count += row->download_count;
    batch->cache_hits += row->cache_hits;
    batch->cache_misses += row->cache_misses;
    batch->queue_synchronizations += row->queue_synchronizations;
    batch->device_synchronizations += row->device_synchronizations;
    batch->total_ns += row->total_ns;
    batch->synchronization_ns += row->synchronization_ns;
    if (yvex_execution_memory_facts_merge(
            &batch->memory, &row->memory, NULL) != YVEX_OK)
        return 0;
    for (rank = 0ull; rank < row->router.selected_count; ++rank) {
        unsigned long long expert = row->router.selected_experts[rank];
        if (expert >= seen_count) return 0;
        if (!seen[expert]) {
            seen[expert] = 1u;
            batch->unique_experts++;
        }
    }
    return 1;
}

static int runtime_moe_batch_identity(const yvex_moe_plan_summary *plan,
                                      const yvex_moe_layer_plan *layer,
                                      const yvex_moe_row_batch *batch,
                                      yvex_moe_row_batch_result *result)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long row;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.moe-row-batch.v2") ||
        !yvex_sha256_update_text(&hash, plan->moe_plan_identity) ||
        !yvex_sha256_update_text(&hash, layer->layer_identity) ||
        !yvex_sha256_update_text(
            &hash, result->execution_profile_available
                       ? result->execution_profile_identity : "uncompiled-reference") ||
        !yvex_sha256_update_u64(&hash, result->execution_class) ||
        !yvex_sha256_update_u64(&hash, result->row_count) ||
        !yvex_sha256_update_u64(&hash, result->row_expert_pairs) ||
        !yvex_sha256_update_u64(&hash, result->unique_experts) ||
        !yvex_sha256_update_u64(&hash, result->worklists.worklist_count) ||
        !yvex_sha256_update_u64(&hash, result->worklists.bucket_count) ||
        !yvex_sha256_update_u64(&hash,
                                result->worklists.maximum_bucket_population) ||
        !yvex_sha256_update_u64(&hash,
                                result->worklists.matrix_tile_eligible_pairs) ||
        !yvex_sha256_update_u64(&hash,
                                result->worklists.matrix_tile_executed_pairs) ||
        !yvex_sha256_update_u64(&hash, result->device_completion_pending) ||
        !yvex_sha256_update_u64(&hash, result->active_weight_base_bytes) ||
        !yvex_sha256_update_u64(
            &hash, result->active_weight_per_unique_expert_bytes) ||
        !yvex_sha256_update_text(&hash, result->routing_digest) ||
        (result->execution_batch_identity[0] &&
         !yvex_sha256_update_text(&hash, result->execution_batch_identity)) ||
        (result->expert_worklist_identity[0] &&
         !yvex_sha256_update_text(&hash, result->expert_worklist_identity)))
        return 0;
    for (row = 0ull; row < batch->row_count; ++row)
        if (!yvex_sha256_update_u64(&hash, batch->token_ids[row])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, result->execution_identity);
    return 1;
}

static int runtime_moe_device_routing_identity(
    const yvex_moe_plan_summary *plan, const yvex_moe_layer_plan *layer,
    const yvex_moe_row_batch *batch,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long row;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.runtime.moe-device-routing.v1") ||
        !yvex_sha256_update_text(&hash, plan->moe_plan_identity) ||
        !yvex_sha256_update_text(&hash, layer->layer_identity) ||
        !yvex_sha256_update_text(
            &hash, batch->execution_profile_identity
                       ? batch->execution_profile_identity : "uncompiled-reference") ||
        !yvex_sha256_update_u64(&hash, batch->row_count) ||
        !yvex_sha256_update_u64(&hash, batch->row_width) ||
        !yvex_sha256_update_u64(&hash, layer->experts_per_token))
        return 0;
    for (row = 0ull; row < batch->row_count; ++row)
        if (!yvex_sha256_update_u64(&hash, batch->token_ids[row])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_moe_row_routing_identity(
    const yvex_runtime_moe_context *context, unsigned long long layer_index,
    const yvex_moe_row_batch *batch, char output[YVEX_SHA256_HEX_CAP],
    yvex_error *err)
{
    const yvex_moe_plan_summary *plan = context
        ? yvex_moe_plan_summary_get(context->plan) : NULL;
    const yvex_moe_layer_plan *layer = context
        ? yvex_moe_plan_layer_at(context->plan, layer_index) : NULL;
    if (output) output[0] = '\0';
    if (!plan || !layer || !batch || !output || !batch->token_ids ||
        !batch->row_count || !batch->token_ids_present ||
        !runtime_moe_device_routing_identity(plan, layer, batch, output))
        return runtime_moe_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "MoE row routing identity owners are unavailable");
    yvex_error_clear(err);
    return YVEX_OK;
}

static int runtime_moe_batch_result_seal(
    const yvex_moe_plan_summary *plan, const yvex_moe_layer_plan *layer,
    const yvex_moe_row_batch *batch, const yvex_execution_batch *execution_batch,
    const yvex_expert_worklist_policy *worklist_policy,
    yvex_moe_row_batch_result *result, yvex_error *err)
{
    result->schema_version = YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V4;
    result->completed = !result->device_completion_pending;
    result->execution_class = batch->execution_class;
    result->row_count = batch->row_count;
    result->shared_expert_operations = batch->row_count * layer->shared_experts;
    result->execution_profile_available = batch->execution_profile_identity != NULL;
    if (batch->execution_profile_identity)
        yvex_runtime_identity_copy(result->execution_profile_identity,
                                   batch->execution_profile_identity);
    if (batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE) {
        yvex_runtime_identity_copy(result->execution_batch_identity,
                                   execution_batch->identity);
        yvex_runtime_identity_copy(result->worklist_policy_identity,
                                   worklist_policy->identity);
        if (yvex_expert_worklist_routing_identity(
                result->execution_batch_identity, result->worklist_policy_identity,
                result->routing_digest, result->expert_worklist_identity, err) != YVEX_OK)
            return yvex_error_code(err);
    }
    if (!runtime_moe_batch_identity(plan, layer, batch, result))
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "ordered MoE execution identity failed");
    return YVEX_OK;
}

static int runtime_moe_transaction_begin(
    yvex_runtime_moe_context *context, const yvex_moe_layer_plan *layer,
    const yvex_moe_row_batch *batch,
    const yvex_backend_moe_operations *operations, yvex_error *err)
{
    if (batch->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE ||
        context->pending_active)
        return YVEX_OK;
    if ((!batch->complete_after_operation && layer->ordinal != 0ull) ||
        !context->pending_layers || !context->pending_layer_capacity ||
        !operations->complete_rows)
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "MoE row transaction cannot begin");
    memset(context->pending_layers, 0,
           (size_t)context->pending_layer_capacity *
               sizeof(*context->pending_layers));
    context->pending_layer_count = 0ull;
    context->pending_first_layer = layer->ordinal;
    context->pending_active = 1;
    return YVEX_OK;
}

/*
 * Admit width-N as one ordered MoE operation while the portable implementation remains
 * token-local. This boundary prevents Transformer and future backends from defining batching as
 * repeated one-row calls; a grouped kernel can replace this adapter without changing semantics.
 */
static int runtime_moe_execute_layer_rows(yvex_runtime_moe_context *context, unsigned long long layer_index,
    const yvex_moe_row_batch *batch, const yvex_moe_row_batch_output *output,
    yvex_moe_row_batch_result *result, yvex_error *err)
{
    const yvex_moe_plan_summary *plan = context ? yvex_moe_plan_summary_get(context->plan) : NULL;
    const yvex_moe_layer_plan *layer = context
        ? yvex_moe_plan_layer_at(context->plan, layer_index) : NULL;
    yvex_runtime_session_summary session;
    const yvex_backend_moe_operations *backend_operations = context
        ? yvex_backend_moe_operations_get(context->session_view->backend) : NULL;
    yvex_moe_layer_job batch_job;
    yvex_execution_batch_source execution_source;
    yvex_execution_batch execution_batch;
    yvex_expert_worklist_policy worklist_policy;
    yvex_moe_device_completion completion = {0};
    yvex_moe_device_completion_slot *pending = NULL;
    yvex_moe_row_batch_output staged_output;
    yvex_sha256 routing_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES], *seen = NULL;
    unsigned long long row, hidden_count, residual_count, combination_count, fixed_bytes = 0ull;
    int rc = YVEX_OK, locked = 0, deferred;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !plan || !layer || !batch || !output || !result ||
        batch->schema_version != YVEX_MOE_ROW_BATCH_SCHEMA_V1 || !batch->row_count ||
        batch->provenance > YVEX_EXECUTION_BATCH_COMPILED_COMPATIBLE ||
        batch->phase >= YVEX_EXECUTION_PHASE_COUNT ||
        (batch->complete_after_operation != 0 &&
         batch->complete_after_operation != 1) ||
        batch->row_width != layer->expanded_width ||
        batch->row_stride < batch->row_width || !batch->expanded_rows || !batch->token_ids ||
        !batch->token_ids_present || batch->row_count > context->options.row_capacity ||
        (batch->execution_class != YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE &&
         batch->execution_class != YVEX_EXECUTION_CLASS_DEVICE_NATIVE) ||
        (context->options.execution_profile &&
         ((context->options.execution_profile->moe_resolution == YVEX_EXECUTION_RESOLUTION_EXACT) !=
              (batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE) ||
          !batch->execution_profile_identity ||
          strcmp(batch->execution_profile_identity,
                 context->options.execution_profile->identity) != 0)) ||
        (!context->options.execution_profile && batch->execution_profile_identity) ||
        (batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE &&
         (!backend_operations || !batch->device_rows || !batch->device_outputs)) ||
        !yvex_core_u64_mul(batch->row_count, layer->hidden_width, &hidden_count) ||
        !yvex_core_u64_mul(batch->row_count, layer->residual_streams, &residual_count) ||
        !yvex_core_u64_mul(residual_count, layer->residual_streams, &combination_count) ||
        !output->combined_rows || output->combined_capacity < hidden_count ||
        !output->routed_rows || output->routed_capacity < hidden_count ||
        !output->shared_rows || output->shared_capacity < hidden_count ||
        !output->post_rows || output->post_capacity < residual_count ||
        !output->combination_rows || output->combination_capacity < combination_count ||
        ((batch->device_rows == NULL) != (batch->device_outputs == NULL)) ||
        yvex_runtime_session_summary_copy(context->session, &session, err) != YVEX_OK ||
        !session.busy || !layer->routed_experts || layer->routed_experts > SIZE_MAX)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "ordered MoE row batch or execution profile is invalid");
    if (batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE &&
        runtime_moe_worklist_contract(context, layer, batch, &session, &execution_source,
                                      &execution_batch, &worklist_policy, err) != YVEX_OK)
        return yvex_error_code(err);
    if (pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context lock failed");
    locked = 1;
    if (context->busy || context->invalidated) {
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE context is busy or invalidated");
        goto done;
    }
    context->busy = 1;
    rc = runtime_moe_transaction_begin(context, layer, batch,
                                       backend_operations, err);
    if (rc != YVEX_OK) goto done;
    deferred = batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE &&
               context->pending_active;
    if (!deferred && batch->device_outputs) batch->device_outputs->is_written = 0;
    if ((context->pending_active && !deferred) ||
        (deferred &&
         (context->pending_first_layer >
                  ULLONG_MAX - context->pending_layer_count ||
          layer->ordinal !=
              context->pending_first_layer + context->pending_layer_count ||
          layer->ordinal >= context->pending_layer_capacity ||
          context->pending_layers[layer->ordinal].pending))) {
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE, "MoE deferred layer ownership is invalid");
        goto done;
    }
    if (!deferred)
        seen = yvex_core_calloc((size_t)layer->routed_experts, sizeof(*seen));
    if (!deferred && !seen) {
        rc = runtime_moe_refuse(err, YVEX_ERR_NOMEM, "MoE batch expert-set allocation failed");
        goto done;
    }
    yvex_sha256_init(&routing_hash);
    (void)yvex_sha256_update_text(&routing_hash, "yvex.runtime.moe-row-routing.v1");
    if (batch->execution_class == YVEX_EXECUTION_CLASS_DEVICE_NATIVE) {
        staged_output = *output;
        rc = runtime_moe_load_layer(
            context, layer, batch->token_ids[0], batch->row_count, 1,
            &batch_job, &fixed_bytes, err);
        if (rc == YVEX_OK) {
            batch_job.expanded_input = batch->expanded_rows;
            batch_job.device_input = batch->device_rows;
            batch_job.device_output = batch->device_outputs;
            batch_job.execution_batch = &execution_batch;
            batch_job.worklist_policy = &worklist_policy;
            if (deferred) {
                pending = &context->pending_layers[layer->ordinal];
                completion.defer = 1;
                completion.host = pending;
                batch_job.device_completion = &completion;
            }
            rc = backend_operations->execute_rows(
                context->session_view->backend, &batch_job, batch, &staged_output,
                result, err);
        }
        if (rc == YVEX_OK) {
            if (!result->device_completion_pending || result->completed ||
                !result->active_weight_per_unique_expert_bytes ||
                !runtime_moe_device_routing_identity(
                    plan, layer, batch, result->routing_digest))
                rc = runtime_moe_refuse(
                    err, YVEX_ERR_STATE,
                    "CUDA width-N MoE deferred publication is incomplete");
            else {
                pending->row_count = batch->row_count;
                pending->row_expert_pairs = result->row_expert_pairs;
                pending->routed_experts = layer->routed_experts;
                pending->active_base_bytes = result->active_weight_base_bytes;
                pending->active_per_expert_bytes =
                    result->active_weight_per_unique_expert_bytes;
                pending->activation_bytes = result->memory.activation_bytes;
                pending->temporary_bytes = result->memory.temporary_bytes;
                pending->pending = 1;
                context->pending_layer_count++;
            }
        }
    } else for (row = 0ull; row < batch->row_count && rc == YVEX_OK; ++row) {
        yvex_moe_layer_result staged;
        yvex_device_tensor device_input, device_output;
        const yvex_device_tensor *device_input_ptr = NULL;
        yvex_device_tensor *device_output_ptr = NULL;
        if (batch->device_rows &&
            (!yvex_backend_tensor_f32_subview(batch->device_rows, row * batch->row_width,
                                              batch->row_width, &device_input) ||
             !yvex_backend_tensor_f32_subview(batch->device_outputs, row * batch->row_width,
                                              batch->row_width, &device_output))) {
            rc = runtime_moe_refuse(err, YVEX_ERR_BOUNDS,
                                    "ordered MoE device row view is invalid");
            break;
        }
        if (batch->device_rows) {
            device_input_ptr = &device_input;
            device_output_ptr = &device_output;
        }
        rc = runtime_moe_layer_owned(
            context, layer_index, batch->expanded_rows + row * batch->row_stride,
            device_input_ptr, device_output_ptr, batch->token_ids[row], 1, &staged, err);
        if (rc != YVEX_OK) break;
        memcpy(output->combined_rows + row * layer->hidden_width, staged.combined_output,
               (size_t)layer->hidden_width * sizeof(float));
        memcpy(output->routed_rows + row * layer->hidden_width, staged.routed_output,
               (size_t)layer->hidden_width * sizeof(float));
        memcpy(output->shared_rows + row * layer->hidden_width, staged.shared_output,
               (size_t)layer->hidden_width * sizeof(float));
        memcpy(output->post_rows + row * layer->residual_streams, staged.post,
               (size_t)layer->residual_streams * sizeof(float));
        memcpy(output->combination_rows + row * layer->residual_streams * layer->residual_streams,
               staged.combination,
               (size_t)layer->residual_streams * layer->residual_streams * sizeof(float));
        if (!runtime_moe_batch_account(result, &staged, seen,
                                       layer->routed_experts)) {
            rc = runtime_moe_refuse(err, YVEX_ERR_STATE,
                                    "MoE batch selected an expert outside its plan");
            break;
        }
        if (!yvex_sha256_update_text(&routing_hash, staged.routing_digest))
            rc = runtime_moe_refuse(err, YVEX_ERR_STATE,
                                    "ordered MoE routing identity update failed");
    }
    if (rc == YVEX_OK && !deferred && !yvex_sha256_final(&routing_hash, digest))
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE,
                                "ordered MoE routing identity finalization failed");
    if (rc == YVEX_OK && !deferred && batch->device_outputs) batch->device_outputs->is_written = 1;
    if (rc == YVEX_OK && !deferred)
        yvex_sha256_hex(digest, result->routing_digest);
    if (rc == YVEX_OK)
        rc = runtime_moe_batch_result_seal(
            plan, layer, batch, &execution_batch, &worklist_policy, result, err);
done:
    yvex_core_free(seen);
    if (locked) {
        context->busy = 0;
        if (rc == YVEX_OK) context->execution_count++;
        (void)pthread_mutex_unlock(&context->mutex);
    }
    if (rc != YVEX_OK) memset(result, 0, sizeof(*result));
    return rc;
}

static int runtime_moe_rows_complete(yvex_runtime_moe_context *context,
                                     int barrier_observed,
                                     yvex_moe_row_batch_result *result,
                                     yvex_error *err)
{
    const yvex_backend_moe_operations *operations;
    yvex_moe_row_batch_result backend = {0};
    unsigned long long index, found = 0ull;
    int rc = YVEX_OK;
    if (result) memset(result, 0, sizeof(*result));
    if (!context || !result || (barrier_observed != 0 && barrier_observed != 1) ||
        pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "MoE row completion owner is invalid");
    if (context->busy) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "busy MoE rows cannot complete");
    }
    if (!context->pending_active) {
        result->schema_version = YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V4;
        result->completed = 1;
        (void)pthread_mutex_unlock(&context->mutex);
        yvex_error_clear(err);
        return YVEX_OK;
    }
    operations = yvex_backend_moe_operations_get(context->session_view->backend);
    if (!operations || !operations->complete_rows) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "pending MoE rows have no completion owner");
    }
    rc = operations->complete_rows(
        context->session_view->backend, barrier_observed, &backend, err);
    for (index = 0ull; rc == YVEX_OK &&
                         index < context->pending_layer_capacity; ++index) {
        yvex_moe_device_completion_slot *pending = &context->pending_layers[index];
        yvex_expert_worklist_observation observation;
        unsigned long long active;
        if (!pending->pending) continue;
        found++;
        if (pending->status) {
            yvex_error_setf(
                err, YVEX_ERR_BACKEND, "runtime.moe",
                "deferred CUDA MoE layer %llu reported device status %d",
                index, pending->status);
            rc = YVEX_ERR_BACKEND;
            break;
        }
        observation = pending->worklist;
        if (observation.schema_version !=
                YVEX_EXPERT_WORKLIST_OBSERVATION_SCHEMA_V1 ||
            observation.worklist_count != 1ull ||
            observation.pair_count != pending->row_expert_pairs ||
            !pending->worklist.bucket_count ||
            pending->worklist.bucket_count > pending->routed_experts ||
            pending->worklist.bucket_count > pending->row_expert_pairs ||
            !pending->worklist.maximum_bucket_population ||
            pending->worklist.matrix_tile_eligible_pairs >
                pending->row_expert_pairs ||
            pending->worklist.narrow_pairs !=
                pending->row_expert_pairs -
                    pending->worklist.matrix_tile_eligible_pairs ||
            !yvex_core_u64_mul(pending->active_per_expert_bytes,
                               pending->worklist.bucket_count, &active) ||
            !yvex_core_u64_add(active, pending->active_base_bytes, &active) ||
            !yvex_core_u64_add(result->unique_experts,
                               pending->worklist.bucket_count,
                               &result->unique_experts) ||
            !yvex_core_u64_add(result->row_expert_pairs,
                               pending->row_expert_pairs,
                               &result->row_expert_pairs) ||
            yvex_expert_worklist_observation_add(
                &result->worklists, &observation, err) != YVEX_OK ||
            !yvex_core_u64_add(result->encoded_bytes_read, active,
                               &result->encoded_bytes_read) ||
            yvex_execution_memory_facts_add(
                &result->memory, active, 0ull, pending->activation_bytes,
                pending->temporary_bytes, 1ull, 0ull, err) != YVEX_OK) {
            rc = runtime_moe_refuse(
                err, YVEX_ERR_STATE,
                "deferred CUDA MoE physical facts are invalid");
            break;
        }
    }
    if (rc == YVEX_OK && found != context->pending_layer_count)
        rc = runtime_moe_refuse(err, YVEX_ERR_STATE,
                                "deferred CUDA MoE layer count diverged");
    if (rc == YVEX_OK) {
        result->schema_version = YVEX_MOE_ROW_BATCH_RESULT_SCHEMA_V4;
        result->completed = 1;
        result->execution_class = YVEX_EXECUTION_CLASS_DEVICE_NATIVE;
        result->queue_synchronizations = backend.queue_synchronizations;
        result->device_synchronizations = backend.device_synchronizations;
        result->synchronization_ns = backend.synchronization_ns;
    } else {
        context->invalidated = 1;
    }
    memset(context->pending_layers, 0,
           (size_t)context->pending_layer_capacity *
               sizeof(*context->pending_layers));
    context->pending_layer_count = 0ull;
    context->pending_first_layer = 0ull;
    context->pending_active = 0;
    (void)pthread_mutex_unlock(&context->mutex);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_moe_rows(yvex_runtime_moe_context *context,
                          const yvex_moe_rows_request *request,
                          yvex_moe_row_batch_result *result,
                          yvex_error *err)
{
    if (!request)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                                  "MoE row operation is required");
    if (request->operation == YVEX_MOE_ROWS_EXECUTE)
        return runtime_moe_execute_layer_rows(
            context, request->layer_index, request->batch, request->output,
            result, err);
    if (request->operation == YVEX_MOE_ROWS_COMPLETE)
        return runtime_moe_rows_complete(
            context, request->barrier_observed, result, err);
    return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG,
                              "MoE row operation is invalid");
}

int yvex_runtime_moe_context_reset(yvex_runtime_moe_context *context, yvex_error *err)
{
    const yvex_moe_layer_plan *layer;
    if (!context || pthread_mutex_lock(&context->mutex) != 0)
        return runtime_moe_refuse(err, YVEX_ERR_INVALID_ARG, "MoE context reset is invalid");
    if (context->busy || context->pending_active) {
        (void)pthread_mutex_unlock(&context->mutex);
        return runtime_moe_refuse(err, YVEX_ERR_STATE,
                                  "busy or pending MoE context cannot reset");
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

int yvex_runtime_moe_context_close(yvex_runtime_moe_context **context, yvex_error *err)
{
    unsigned long long index;
    int rc = YVEX_OK;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    if ((*context)->busy || (*context)->pending_active)
        return runtime_moe_refuse(
            err, YVEX_ERR_STATE, "busy or pending MoE context cannot close");
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
    if ((*context)->mutex_ready) (void)pthread_mutex_destroy(&(*context)->mutex);
    free(*context);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int runtime_moe_cleanup(void **opaque, yvex_error *err)
{
    return yvex_runtime_moe_context_close((yvex_runtime_moe_context **)opaque, err);
}

static void runtime_moe_operator_refuse(yvex_moe_operator_result *result,
                                        const yvex_error *err)
{
    yvex_core_text_copy(result->status, sizeof(result->status), "refused");
    yvex_core_text_copy(result->reason, sizeof(result->reason),
                        err && yvex_error_is_set(err) ? yvex_error_message(err)
                                                     : "MoE execution refused");
}

int yvex_runtime_moe_operator_execute(const yvex_moe_operator_request *request,
                                      yvex_moe_operator_result *result,
                                      yvex_runtime_cleanup_lease **retained_cleanup,
                                      yvex_error *err)
{
    yvex_model_engine_open_request model_request = {0};
    yvex_runtime_session_open_request session_request = {0};
    yvex_runtime_moe_options options = {0};
    yvex_moe_input_limits limits = {0};
    yvex_model_engine_failure failure = {0};
    yvex_runtime_cleanup_lease *cleanup = NULL;
    yvex_model_engine *model = NULL;
    yvex_runtime_execution_session *session = NULL;
    yvex_runtime_moe_context *context = NULL;
    yvex_moe_input *input = NULL;
    yvex_runtime_moe_output output = {0};
    const yvex_model_engine_view *model_view;
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
    yvex_core_text_copy(result->command, sizeof(result->command), "execute moe");
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
        rc = yvex_runtime_moe_context_open(&context, model, session, &options, NULL, err);
    if (rc == YVEX_OK) {
        rc = yvex_runtime_cleanup_lease_adopt(cleanup, context, runtime_moe_cleanup, err);
        adopted = rc == YVEX_OK;
    }
    model_view = yvex_model_engine_view_get(model);
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
        yvex_core_text_copy(result->command, sizeof(result->command), "execute moe");
        yvex_core_text_copy(result->target, sizeof(result->target), request->target);
        yvex_core_text_copy(result->family, sizeof(result->family),
                            model_view->target_id);
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
