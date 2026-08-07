/*
 * A generation context admits one model/session pair and owns the finite composition of lower
 * transformer, logits, sampling, tokenizer, and speculative resources. Construction publishes
 * nothing until every lower plan and lifecycle primitive is ready; close drains an admitted turn
 * before releasing those resources in reverse dependency order.
 */
#include "src/runtime/private.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <build_commit.h>
#include <yvex/internal/backend.h>
#include <yvex/internal/core.h>
#include <yvex/internal/execution.h>
#include <yvex/internal/moe.h>

static int generation_context_refuse(yvex_error *err, yvex_status status,
                                     const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}

static int generation_device_stochastic(
    const yvex_runtime_generation_context *context,
    const yvex_backend *backend)
{
    return context && context->options.backend == YVEX_BACKEND_KIND_CUDA &&
           context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION &&
           context->options.sampling_policy.strategy ==
               YVEX_SAMPLING_STRATEGY_STOCHASTIC &&
           yvex_backend_sampling_operations_get(backend) != NULL;
}

static int generation_device_selection(
    const yvex_runtime_generation_context *context,
    const yvex_backend *backend)
{
    return context && context->options.backend == YVEX_BACKEND_KIND_CUDA &&
           context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION &&
           (context->options.sampling_policy.strategy ==
                YVEX_SAMPLING_STRATEGY_GREEDY ||
            generation_device_stochastic(context, backend));
}

typedef struct {
    yvex_execution_state_class_request classes[YVEX_MODEL_STATE_CLASS_COUNT];
    unsigned long long candidate_bytes_per_token;
} generation_capacity_geometry;

static unsigned long long generation_capacity_gcd(unsigned long long left,
                                                   unsigned long long right)
{
    while (right) {
        unsigned long long remainder = left % right;
        left = right;
        right = remainder;
    }
    return left;
}

static int generation_capacity_lcm(unsigned long long left,
                                   unsigned long long right,
                                   unsigned long long *result)
{
    unsigned long long divisor;
    if (!left || !right || !result) return 0;
    divisor = generation_capacity_gcd(left, right);
    return yvex_core_u64_mul(left / divisor, right, result);
}

static int generation_capacity_periodic_add(
    yvex_execution_state_class_request *state, unsigned long long period,
    unsigned long long bytes)
{
    unsigned long long common, existing, added;
    if (!state || !period || !bytes) return 0;
    if (!state->bytes_per_block) {
        state->extent = YVEX_EXECUTION_STATE_EXTENT_CONTEXT;
        state->logical_block_tokens = period;
        state->bytes_per_block = bytes;
        return 1;
    }
    if (state->extent != YVEX_EXECUTION_STATE_EXTENT_CONTEXT ||
        !generation_capacity_lcm(state->logical_block_tokens, period, &common) ||
        !yvex_core_u64_mul(state->bytes_per_block,
                           common / state->logical_block_tokens, &existing) ||
        !yvex_core_u64_mul(bytes, common / period, &added) ||
        !yvex_core_u64_add(existing, added, &state->bytes_per_block)) return 0;
    state->logical_block_tokens = common;
    return 1;
}

static int generation_capacity_fixed_add(
    yvex_execution_state_class_request *state, unsigned long long tokens,
    unsigned long long bytes_per_token)
{
    if (!state || !tokens || !bytes_per_token) return 0;
    if (!state->bytes_per_block) {
        state->extent = YVEX_EXECUTION_STATE_EXTENT_FIXED;
        state->logical_block_tokens = 1ull;
        state->fixed_tokens_per_sequence = tokens;
        state->bytes_per_block = bytes_per_token;
        return 1;
    }
    return state->extent == YVEX_EXECUTION_STATE_EXTENT_FIXED &&
           state->fixed_tokens_per_sequence == tokens &&
           yvex_core_u64_add(state->bytes_per_block, bytes_per_token,
                             &state->bytes_per_block);
}

static int generation_capacity_component_bytes(
    const yvex_attention_state_component_recipe *component,
    unsigned long long bank_count, unsigned long long *bytes)
{
    unsigned long long values;
    if (!component || !bytes || !bank_count) return 0;
    if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
        if (!yvex_core_u64_mul(component->value_width, sizeof(float), &values) ||
            !yvex_core_u64_add(values, sizeof(unsigned long long), &values) ||
            !yvex_core_u64_mul(values, bank_count, bytes)) return 0;
        return 1;
    }
    if (!yvex_core_u64_add(component->rolling.kv_state_extent,
                           component->rolling.score_state_extent, &values) ||
        !yvex_core_u64_mul(values, sizeof(float), &values) ||
        !yvex_core_u64_mul(values, bank_count, bytes)) return 0;
    return 1;
}

static int generation_capacity_target_component(
    generation_capacity_geometry *geometry,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_component_recipe *component)
{
    yvex_model_state_class state_class;
    unsigned long long bytes, candidate;
    if (!generation_capacity_component_bytes(component, 2ull, &bytes) ||
        !generation_capacity_component_bytes(component, 1ull, &candidate) ||
        !yvex_core_u64_add(geometry->candidate_bytes_per_token, candidate,
                           &geometry->candidate_bytes_per_token)) return 0;
    switch (component->binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        if (component->capacity && component->binding ==
                                       YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
            if (!yvex_core_u64_mul(bytes, 2ull, &bytes)) return 0;
        }
        return generation_capacity_fixed_add(
            &geometry->classes[YVEX_MODEL_STATE_SWA_RING],
            component->capacity, bytes);
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        state_class = layer->attention_class == YVEX_ATTENTION_CLASS_HCA
                          ? YVEX_MODEL_STATE_HCA_HISTORY
                          : YVEX_MODEL_STATE_COMPRESSED_HISTORY;
        return generation_capacity_periodic_add(
            &geometry->classes[state_class], layer->compression_ratio, bytes);
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        return generation_capacity_periodic_add(
            &geometry->classes[YVEX_MODEL_STATE_INDEXER_HISTORY],
            layer->compression_ratio, bytes);
    case YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING:
        return generation_capacity_fixed_add(
            &geometry->classes[YVEX_MODEL_STATE_MAIN_ROLLING], 1ull, bytes);
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING:
        return generation_capacity_fixed_add(
            &geometry->classes[YVEX_MODEL_STATE_INDEXER_ROLLING], 1ull, bytes);
    default: return 0;
    }
}

static int generation_capacity_plan_accumulate(
    generation_capacity_geometry *geometry,
    const yvex_attention_plan *attention,
    const yvex_graph_attention_capacity_plan *capacity, int draft)
{
    unsigned long long layer_index;
    for (layer_index = 0ull; layer_index < yvex_attention_plan_layer_count(attention);
         ++layer_index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(attention, layer_index);
        const yvex_graph_attention_capacity_layer *capacity_layer =
            yvex_graph_attention_capacity_plan_layer(capacity, layer_index);
        unsigned int component_index;
        if (!layer || !capacity_layer || !capacity_layer->selected) return 0;
        for (component_index = 0u;
             component_index < capacity_layer->recipe.component_count;
             ++component_index) {
            const yvex_attention_state_component_recipe *component =
                &capacity_layer->recipe.components[component_index];
            unsigned long long bytes;
            if (!draft) {
                if (!generation_capacity_target_component(
                        geometry, layer, component)) return 0;
                continue;
            }
            if (component->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY ||
                component->binding != YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY ||
                !generation_capacity_component_bytes(component, 2ull, &bytes) ||
                !yvex_core_u64_mul(bytes, 2ull, &bytes) ||
                !generation_capacity_fixed_add(
                    &geometry->classes[YVEX_MODEL_STATE_DRAFT_PERSISTENT],
                    component->capacity, bytes)) return 0;
        }
    }
    return 1;
}

static void generation_capacity_geometry_initialize(
    generation_capacity_geometry *geometry,
    const yvex_model_execution_descriptor *model)
{
    unsigned long long index;
    memset(geometry, 0, sizeof(*geometry));
    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        yvex_execution_state_class_request *state = &geometry->classes[index];
        if (!(model->persistent_state_class_mask &
              YVEX_MODEL_STATE_CLASS_BIT(index))) continue;
        state->state_class = (yvex_model_state_class)index;
        state->alignment_bytes = 256ull;
        state->kernel_tile_tokens = 1ull;
        state->promotion_granularity_tokens = 1ull;
        state->page_table_entry_bytes = 16ull;
    }
}

static int generation_capacity_graph_geometry(
    yvex_runtime_generation_context *context,
    const yvex_model_execution_descriptor *model,
    generation_capacity_geometry *geometry, yvex_error *err)
{
    const yvex_graph_family_api *graph = context->model_view->adapter->graph();
    const yvex_attention_plan *plans[2] = {
        context->model_view->attention, context->model_view->draft_attention};
    unsigned long long plan_index;
    generation_capacity_geometry_initialize(geometry, model);
    for (plan_index = 0ull; plan_index < 2ull; ++plan_index) {
        yvex_graph_attention_capacity_request request = {0};
        yvex_graph_attention_capacity_plan *capacity = NULL;
        int rc;
        if (!plans[plan_index]) {
            if (!plan_index ||
                (model->persistent_state_class_mask &
                 YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_DRAFT_PERSISTENT)))
                return generation_context_refuse(
                    err, YVEX_ERR_STATE,
                    "model state geometry requires an unavailable attention plan");
            continue;
        }
        request.scope = YVEX_ATTENTION_PROBE_SCOPE_FULL;
        request.history_tokens = 0ull;
        request.start_position = 0ull;
        request.token_count = context->options.context_capacity;
        request.execution_count = 1ull;
        request.use_requested_position = 1;
        rc = yvex_graph_attention_capacity_plan_build(
            &capacity, graph, plans[plan_index], &request, err);
        if (rc == YVEX_OK &&
            !generation_capacity_plan_accumulate(
                geometry, plans[plan_index], capacity, plan_index != 0ull))
            rc = generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "state-class geometry cannot represent the admitted graph plan");
        yvex_graph_attention_capacity_plan_close(&capacity);
        if (rc != YVEX_OK) return rc;
    }
    if (model->persistent_state_class_mask &
        YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_RESIDUAL_MIXING)) {
        unsigned long long bytes;
        if (!yvex_core_u64_mul(model->residual_streams, model->hidden_width, &bytes) ||
            !yvex_core_u64_mul(bytes, sizeof(float), &bytes) ||
            !generation_capacity_fixed_add(
                &geometry->classes[YVEX_MODEL_STATE_RESIDUAL_MIXING], 1ull, bytes))
            return generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "residual state geometry overflowed");
    }
    if (model->persistent_state_class_mask &
        YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_CANDIDATE_DELTA)) {
        yvex_execution_state_class_request *candidate =
            &geometry->classes[YVEX_MODEL_STATE_CANDIDATE_DELTA];
        candidate->extent = YVEX_EXECUTION_STATE_EXTENT_CANDIDATE;
        candidate->logical_block_tokens = 1ull;
        candidate->bytes_per_block = geometry->candidate_bytes_per_token;
        candidate->promotion_granularity_tokens =
            model->verification_width_maximum;
    }
    if (model->persistent_state_class_mask &
        YVEX_MODEL_STATE_CLASS_BIT(YVEX_MODEL_STATE_PREFIX_CHECKPOINT)) {
        yvex_execution_state_class_request *prefix =
            &geometry->classes[YVEX_MODEL_STATE_PREFIX_CHECKPOINT];
        prefix->extent = YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET;
        prefix->logical_block_tokens = 1ull;
        prefix->bytes_per_block = 16ull;
        prefix->kernel_tile_tokens = model->verification_width_maximum;
        prefix->shared = 1;
        prefix->copy_on_write = 1;
    }
    return YVEX_OK;
}

static int generation_capacity_hardware(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    const yvex_runtime_session_view *view =
        yvex_runtime_session_view_get(context->session);
    yvex_backend_device_info device;
    yvex_backend_cuda_attention_graph_summary cuda = {0};
    long pages, page_bytes;
    unsigned long long total;
    int rc;
    if (!view || !view->backend ||
        yvex_backend_get_device_info(view->backend, &device, err) != YVEX_OK ||
        (page_bytes = sysconf(_SC_PAGESIZE)) <= 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "memory-admission hardware facts are unavailable");
    if (device.kind == YVEX_BACKEND_KIND_CUDA) {
        total = device.total_memory_bytes;
    } else {
        pages = sysconf(_SC_PHYS_PAGES);
        if (pages <= 0 ||
            !yvex_core_u64_mul((unsigned long long)pages,
                               (unsigned long long)page_bytes, &total))
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "host memory extent is unavailable");
    }
    memset(&context->hardware_profile, 0, sizeof(context->hardware_profile));
    context->hardware_profile.schema_version =
        YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1;
    context->hardware_profile.backend = device.kind;
    context->hardware_profile.admitted_fact_mask =
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_MEMORY) |
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_PAGING);
    context->hardware_profile.device_index = device.device_index;
    context->hardware_profile.compute_major = device.compute_capability_major;
    context->hardware_profile.compute_minor = device.compute_capability_minor;
    context->hardware_profile.total_memory_bytes = total;
    context->hardware_profile.usable_memory_bytes = total;
    context->hardware_profile.host_page_bytes = (unsigned long long)page_bytes;
    context->hardware_profile.device_page_bytes = (unsigned long long)page_bytes;
    context->hardware_profile.unified_addressing = device.unified_addressing;
    context->hardware_profile.coherent_host_memory = device.managed_memory;
    if (device.kind == YVEX_BACKEND_KIND_CUDA) {
        if (yvex_backend_cuda_attention_graph_summary_get(
                view->backend, &cuda, err) != YVEX_OK ||
            !cuda.kernel_bundle_architecture[0] ||
            !yvex_sha256_hex_valid(cuda.cuda_build_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "kernel-bundle hardware facts are unavailable");
        rc = yvex_backend_bandwidth_probe(
            view->backend, &context->bandwidth_evidence, err);
        if (rc != YVEX_OK) return rc;
        context->hardware_profile.sustainable_read_bytes_per_second =
            context->bandwidth_evidence.sustainable_read_bytes_per_second;
        context->hardware_profile.sustainable_copy_bytes_per_second =
            context->bandwidth_evidence.sustainable_copy_bytes_per_second;
        context->hardware_profile.admitted_fact_mask |=
            YVEX_EXECUTION_HARDWARE_FACT_BIT(
                YVEX_EXECUTION_HARDWARE_FACT_BANDWIDTH);
        context->hardware_profile.native_architecture_code =
            cuda.kernel_bundle_native;
        if (cuda.kernel_bundle_native)
            context->hardware_profile.admitted_fact_mask |=
                YVEX_EXECUTION_HARDWARE_FACT_BIT(
                    YVEX_EXECUTION_HARDWARE_FACT_NATIVE_CODE);
        (void)snprintf(context->hardware_profile.name,
                       sizeof(context->hardware_profile.name),
                       "cuda-%s", cuda.kernel_bundle_architecture);
    } else {
        yvex_core_text_copy(context->hardware_profile.name,
                            sizeof(context->hardware_profile.name),
                            "cpu-memory");
    }
    return yvex_execution_hardware_profile_seal(
        &context->hardware_profile, err);
}

static int generation_capacity_workload(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime =
        yvex_runtime_descriptor_summary_get(context->model_view->descriptor);
    const yvex_model_execution_descriptor *model =
        runtime && runtime->model_execution.schema_version
            ? &runtime->model_execution
            : NULL;
    const yvex_speculation_family_policy *speculation =
        context->speculation
            ? yvex_runtime_speculation_policy_get(context->speculation)
            : NULL;
    memset(&context->workload_profile, 0, sizeof(context->workload_profile));
    context->workload_profile.schema_version =
        YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1;
    context->workload_profile.kind =
        YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    context->workload_profile.minimum_session_context =
        context->options.context_capacity;
    context->workload_profile.requested_session_context =
        context->options.context_capacity;
    context->workload_profile.concurrent_sequences = 1ull;
    context->workload_profile.logical_batch_tokens =
        context->options.prefill_chunk_tokens;
    context->workload_profile.prefill_chunk_tokens =
        context->options.prefill_chunk_tokens;
    context->workload_profile.attention_microbatch_rows =
        context->options.prefill_chunk_tokens;
    context->workload_profile.moe_row_tile =
        context->options.prefill_chunk_tokens;
    context->workload_profile.output_head_rows =
        speculation ? speculation->block_size + 1ull
                    : (context->options.mode == YVEX_GENERATION_MODE_DSPARK && model
                           ? model->verification_width_maximum
                           : 1ull);
    context->workload_profile.system_reserve_bytes =
        YVEX_EXECUTION_MINIMUM_SYSTEM_RESERVE;
    context->workload_profile.latency_priority = 1;
    yvex_core_text_copy(context->workload_profile.name,
                        sizeof(context->workload_profile.name),
                        "interactive-latency");
    return yvex_execution_workload_profile_seal(
        &context->workload_profile, err);
}

static int generation_sampling_workspace(
    const yvex_runtime_generation_context *context, yvex_backend *backend,
    const yvex_model_execution_descriptor *model,
    unsigned long long *workspace, yvex_error *err)
{
    const yvex_backend_sampling_operations *operations;
    unsigned long long selection = 0ull, speculation = 0ull;
    if (workspace) *workspace = 0ull;
    if (!context || !workspace)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "device sampling workspace owner is unavailable");
    if (!generation_device_stochastic(context, backend)) return YVEX_OK;
    operations = yvex_backend_sampling_operations_get(backend);
    if (!model || !operations || !operations->workspace_required ||
        operations->workspace_required(
            model->vocabulary_size, &selection, err) != YVEX_OK)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "device stochastic workspace geometry is unavailable");
    if (context->options.mode == YVEX_GENERATION_MODE_DSPARK &&
        (!model->proposal_width ||
         !operations->speculation_workspace_required ||
         operations->speculation_workspace_required(
             model->vocabulary_size, model->proposal_width,
             &speculation, err) != YVEX_OK))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "device stochastic speculation workspace geometry is unavailable");
    *workspace = speculation > selection ? speculation : selection;
    return YVEX_OK;
}

static int generation_capacity_build(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *runtime =
        yvex_runtime_descriptor_summary_get(context->model_view->descriptor);
    const yvex_model_execution_descriptor *model =
        runtime && runtime->model_execution.schema_version
            ? &runtime->model_execution
            : NULL;
    yvex_runtime_residency_summary residency;
    generation_capacity_geometry geometry;
    yvex_execution_state_class_request states[YVEX_MODEL_STATE_CLASS_COUNT];
    yvex_execution_capacity_plan_request request = {0};
    unsigned long long workspace, sampling_workspace = 0ull, index, count = 0ull;
    if (generation_capacity_hardware(context, err) != YVEX_OK) return yvex_error_code(err);
    if (!model) return YVEX_OK;
    if (generation_capacity_workload(context, err) != YVEX_OK) return yvex_error_code(err);
    if (context->options.context_capacity > model->maximum_context)
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "requested context exceeds the model-authored maximum");
    if (generation_capacity_graph_geometry(context, model, &geometry, err) != YVEX_OK ||
        yvex_runtime_residency_snapshot(context->model_view->residency, &residency,
                                        NULL, NULL, err) != YVEX_OK)
        return yvex_error_code(err);
    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        if (!(model->persistent_state_class_mask &
              YVEX_MODEL_STATE_CLASS_BIT(index))) continue;
        if (!geometry.classes[index].bytes_per_block)
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "model-authored state class lacks physical geometry");
        states[count++] = geometry.classes[index];
    }
    if (!yvex_core_u64_mul(context->options.prefill_chunk_tokens,
                           model->hidden_width, &workspace) ||
        !yvex_core_u64_mul(workspace, sizeof(float) * 8ull, &workspace) ||
        !workspace)
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "execution workspace geometry overflowed");
    if (generation_sampling_workspace(
            context, yvex_runtime_session_view_get(context->session)->backend,
            model, &sampling_workspace, err) != YVEX_OK)
        return yvex_error_code(err);
    if (sampling_workspace > workspace) workspace = sampling_workspace;
    request.schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    request.model = model;
    request.hardware = &context->hardware_profile;
    request.workload = &context->workload_profile;
    request.model_bytes = residency.encoded_bytes;
    request.state_classes = states;
    request.state_class_count = count;
    request.workspace_bytes = workspace;
    request.scheduler_bytes = 1024ull * 1024ull;
    request.graph_bytes = 1024ull * 1024ull;
    if (!request.model_bytes)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "resident model byte extent is unavailable for capacity admission");
    return yvex_execution_capacity_plan_build(
        &request, &context->capacity_plan, err);
}

static int generation_stops_open(yvex_runtime_generation_context *context,
                                 unsigned long long vocabulary_size,
                                 yvex_error *err)
{
    unsigned long long index;
    if (!context->options.additional_stop_token_count) return YVEX_OK;
    if (!context->options.additional_stop_token_ids ||
        context->options.additional_stop_token_count > SIZE_MAX / sizeof(unsigned int))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "additional stop-token extent is invalid");
    context->additional_stops = yvex_core_calloc(
        (size_t)context->options.additional_stop_token_count,
        sizeof(*context->additional_stops));
    if (!context->additional_stops)
        return generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "additional stop-token allocation failed");
    for (index = 0ull; index < context->options.additional_stop_token_count; ++index) {
        unsigned int token = context->options.additional_stop_token_ids[index];
        unsigned long long scan;
        if (token >= vocabulary_size)
            return generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "additional stop token is outside vocabulary");
        for (scan = 0ull; scan < index; ++scan)
            if (context->additional_stops[scan] == token)
                return generation_context_refuse(
                    err, YVEX_ERR_FORMAT,
                    "additional stop tokens contain a duplicate");
        context->additional_stops[index] = token;
    }
    context->options.additional_stop_token_ids = context->additional_stops;
    return YVEX_OK;
}

static int generation_execution_profile_build(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    const yvex_physical_execution_summary *physical =
        yvex_physical_execution_ir_summary(context->model_view->physical_execution);
    const yvex_runtime_binding_summary *binding = context->model_view->binding;
    const yvex_runtime_session_view *session_view = yvex_runtime_session_view_get(context->session);
    yvex_runtime_session_summary session;
    yvex_compiled_execution_profile_request request = {0};
    yvex_backend_cuda_attention_graph_summary cuda = {0};
    yvex_backend_cuda_graph_capability graph = {0};
    const char *kernel_bundle = YVEX_BUILD_IDENTITY;
    char hardware[YVEX_EXECUTION_TEXT_CAP];
    int rc;

    if (!physical || !binding || !session_view || !session_view->backend ||
        yvex_runtime_session_summary_copy(context->session, &session, err) != YVEX_OK)
        return generation_context_refuse(
            err, YVEX_ERR_STATE, "execution profile owners are unavailable");
    if (context->options.backend == YVEX_BACKEND_KIND_CUDA) {
        rc = yvex_backend_cuda_attention_graph_summary_get(
            session_view->backend, &cuda, err);
        if (rc != YVEX_OK || !yvex_sha256_hex_valid(cuda.cuda_build_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE, "CUDA kernel bundle identity is unavailable");
        rc = yvex_backend_cuda_graph_query(session_view->backend, &graph, err);
        if (rc != YVEX_OK)
            return generation_context_refuse(
                err, YVEX_ERR_STATE, "CUDA graph capability is unavailable");
        kernel_bundle = cuda.cuda_build_identity;
        (void)snprintf(hardware, sizeof(hardware), "portable-cuda-sm%d%d",
                       session.compute_capability_major,
                       session.compute_capability_minor);
    } else {
        yvex_core_text_copy(hardware, sizeof(hardware), "portable-cpu");
    }
    request.schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V1;
    request.logical_model_identity = binding->logical_model_identity;
    request.physical_variant_identity = binding->profile_identity;
    request.physical_execution_identity = physical->identity;
    request.artifact_identity = binding->artifact_identity;
    request.materialization_identity = binding->materialization_identity;
    request.runtime_binding_identity = binding->identity;
    request.kernel_bundle_identity = kernel_bundle;
    request.hardware_profile = hardware;
    request.backend = context->options.backend;
    request.device_index = session.device_index;
    request.compute_major = session.compute_capability_major;
    request.compute_minor = session.compute_capability_minor;
    request.context_capacity = context->options.context_capacity;
    request.generation_mode = context->options.mode == YVEX_GENERATION_MODE_DSPARK
                                  ? YVEX_EXECUTION_GENERATION_SPECULATIVE
                                  : YVEX_EXECUTION_GENERATION_TARGET_ONLY;
    request.workload = YVEX_EXECUTION_WORKLOAD_INTERACTIVE;
    request.evidence = context->options.evidence_profile;
    request.execution_class = YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.host_stochastic_reference =
        context->options.sampling_policy.strategy != YVEX_SAMPLING_STRATEGY_GREEDY &&
        !generation_device_stochastic(context, session_view->backend);
    request.token_local_moe_reference =
        context->options.backend != YVEX_BACKEND_KIND_CUDA ||
        context->options.evidence_profile != YVEX_EXECUTION_EVIDENCE_PRODUCTION ||
        yvex_backend_moe_operations_get(session_view->backend) == NULL;
    request.eager_attention_reference =
        context->options.backend != YVEX_BACKEND_KIND_CUDA ||
        context->options.mode == YVEX_GENERATION_MODE_DSPARK ||
        !binding->capabilities.cuda_full_graph_implemented ||
        graph.state != YVEX_BACKEND_CUDA_GRAPH_OPEN ||
        !graph.edge_inventory_available || !graph.async_memory_available ||
        !graph.async_copy_available || !graph.pinned_host_memory_available;
    return yvex_compiled_execution_profile_seal(
        &request, &context->execution_profile, err);
}

static int generation_plan_build(yvex_runtime_generation_context *context,
                                 yvex_error *err)
{
    yvex_runtime_model_summary model;
    const yvex_transformer_plan_summary *transformer;
    const yvex_runtime_logits_plan_summary *logits;
    const yvex_tokenizer_plan_summary *tokenizer;
    yvex_runtime_generation_plan_summary plan = {0};
    if (yvex_runtime_model_summary_copy(context->model, &model, err) != YVEX_OK)
        return yvex_error_code(err);
    transformer = yvex_transformer_plan_summary_get(
        yvex_runtime_transformer_context_plan(context->transformer));
    logits = yvex_runtime_logits_plan_summary_get(context->logits);
    tokenizer = yvex_tokenizer_plan_summary_get(context->tokenizer);
    if (!transformer || !logits || !tokenizer || !tokenizer->sealed ||
        !tokenizer->runtime_bound ||
        tokenizer->vocabulary_size != transformer->vocabulary_size ||
        tokenizer->vocabulary_size != logits->vocabulary_size ||
        strcmp(transformer->transformer_plan_identity,
               logits->transformer_plan_identity) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lower-owner plans are incompatible");
    plan.schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V5;
    plan.backend = context->options.backend;
    plan.mode = context->options.mode;
    plan.context_capacity = context->options.context_capacity;
    plan.prefill_chunk_tokens = context->options.prefill_chunk_tokens;
    plan.maximum_new_tokens = context->options.maximum_new_tokens;
    plan.maximum_output_bytes = context->options.maximum_output_bytes;
    plan.trace_policy = (unsigned int)context->options.trace_policy;
    plan.evidence_profile = context->execution_profile.evidence;
    plan.execution_class = context->execution_profile.execution_class;
    yvex_runtime_identity_copy(plan.runtime_model_identity,
                               model.runtime_model_identity);
    yvex_runtime_identity_copy(plan.runtime_binding_identity,
                               context->model_view->binding->identity);
    yvex_runtime_identity_copy(plan.runtime_descriptor_identity,
                               model.runtime_descriptor_identity);
    yvex_runtime_identity_copy(plan.tokenizer_plan_identity,
                               tokenizer->tokenizer_plan_identity);
    yvex_runtime_identity_copy(plan.prompt_policy_identity,
                               tokenizer->prompt_policy_identity);
    yvex_runtime_identity_copy(plan.transformer_plan_identity,
                               transformer->transformer_plan_identity);
    yvex_runtime_identity_copy(plan.logits_plan_identity,
                               logits->output_head_plan_identity);
    yvex_runtime_identity_copy(plan.sampling_policy_identity,
                               context->options.sampling_policy.policy_identity);
    yvex_runtime_identity_copy(plan.kernel_bundle_identity,
                               context->execution_profile.kernel_bundle_identity);
    yvex_runtime_identity_copy(plan.execution_profile_identity,
                               context->execution_profile.identity);
    yvex_runtime_identity_copy(plan.workload_profile_identity, context->workload_profile.identity);
    yvex_core_text_copy(plan.hardware_profile, sizeof(plan.hardware_profile),
                        context->execution_profile.hardware_profile);
    if (context->speculation) {
        const yvex_speculation_family_policy *policy =
            yvex_runtime_speculation_policy_get(context->speculation);
        if (!policy || !yvex_sha256_hex_valid(policy->policy_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "DSpark policy identity is unavailable");
        yvex_runtime_identity_copy(plan.speculation_policy_identity,
                                   policy->policy_identity);
    }
    if (!yvex_runtime_generation_stop_identity(
            tokenizer, context->additional_stops,
            context->options.additional_stop_token_count,
            plan.stop_policy_identity) ||
        !yvex_runtime_generation_plan_identity(
            &plan, plan.generation_plan_identity))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation plan identity derivation failed");
    context->plan = plan;
    return YVEX_OK;
}

static int generation_execution_owners_open(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_options *options,
    const yvex_runtime_logits_plan_summary **logits_plan, yvex_error *err)
{
    yvex_runtime_transformer_options transformer = {0};
    yvex_runtime_logits_options logits = {0};
    yvex_runtime_sampling_options sampling = {0};
    yvex_runtime_speculation_options speculation = {0};
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(context->session);
    const yvex_runtime_descriptor_summary *runtime =
        yvex_runtime_descriptor_summary_get(context->model_view->descriptor);
    const yvex_model_execution_descriptor *model =
        runtime && runtime->model_execution.schema_version
            ? &runtime->model_execution
            : NULL;
    unsigned long long sampling_workspace = 0ull;
    int device_selection = session_view &&
        generation_device_selection(context, session_view->backend);
    int rc;

    context->device_selection = device_selection;

    if (generation_sampling_workspace(
            context, session_view ? session_view->backend : NULL, model,
            &sampling_workspace, err) != YVEX_OK)
        return yvex_error_code(err);

    transformer.maximum_host_bytes = options->maximum_host_bytes;
    transformer.maximum_device_bytes = options->maximum_device_bytes;
    transformer.context_capacity = options->context_capacity;
    transformer.workspace_token_capacity = options->prefill_chunk_tokens;
    transformer.minimum_device_workspace_bytes = sampling_workspace;
    if (options->mode == YVEX_GENERATION_MODE_DSPARK &&
        transformer.workspace_token_capacity < YVEX_SPECULATION_MAX_BLOCK + 2ull)
        transformer.workspace_token_capacity = YVEX_SPECULATION_MAX_BLOCK + 2ull;
    transformer.cancel_requested = options->cancel_requested;
    transformer.cancel_context = options->cancel_context;
    transformer.evidence_level =
        runtime_attention_evidence(options->evidence_profile);
    transformer.device_hidden_output =
        device_selection && options->mode == YVEX_GENERATION_MODE_TARGET_ONLY;
    transformer.execution_profile = &context->execution_profile;
    transformer.shape_registry = context->execution_shapes;
    rc = yvex_runtime_transformer_context_open(
        &context->transformer, context->model, context->session, &transformer, err);
    logits.maximum_rows = options->mode == YVEX_GENERATION_MODE_DSPARK
                              ? YVEX_SPECULATION_MAX_BLOCK + 1ull
                              : 1ull;
    logits.maximum_host_bytes = options->maximum_host_bytes;
    logits.maximum_device_bytes = options->maximum_device_bytes;
    logits.evidence_profile = options->evidence_profile;
    logits.device_selection = device_selection;
    logits.execution_profile = &context->execution_profile;
    logits.cancel_requested = options->cancel_requested;
    logits.cancel_context = options->cancel_context;
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_open(
            &context->logits, context->model, context->session,
            yvex_runtime_transformer_context_plan(context->transformer),
            &logits, err);
    *logits_plan = rc == YVEX_OK
                       ? yvex_runtime_logits_plan_summary_get(context->logits)
                       : NULL;
    if (rc == YVEX_OK && !*logits_plan)
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE, "runtime logits plan is unavailable");
    if (rc != YVEX_OK) return rc;
    sampling.maximum_vocabulary_size = (*logits_plan)->vocabulary_size;
    sampling.maximum_rows = logits.maximum_rows;
    sampling.maximum_host_bytes = options->maximum_host_bytes;
    sampling.device_selection = device_selection;
    sampling.cancel_requested = options->cancel_requested;
    sampling.cancel_context = options->cancel_context;
    rc = yvex_runtime_sampling_context_open(
        &context->sampling, *logits_plan, &context->options.sampling_policy,
        &sampling, err);
    if (rc != YVEX_OK || options->mode != YVEX_GENERATION_MODE_DSPARK)
        return rc;
    speculation.backend = options->backend;
    speculation.context_capacity = options->context_capacity;
    speculation.maximum_host_bytes = options->maximum_host_bytes;
    speculation.maximum_device_bytes = options->maximum_device_bytes;
    speculation.cancel_requested = options->cancel_requested;
    speculation.cancel_context = options->cancel_context;
    speculation.execution_profile = &context->execution_profile;
    speculation.shape_registry = context->execution_shapes;
    return yvex_runtime_speculation_context_open(
        &context->speculation, context->model, context->session,
        context->transformer, context->logits, context->sampling,
        &context->options.sampling_policy, &speculation, err);
}

int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_model_failure state_failure = {0};
    yvex_tokenizer_decode_options decoder_options = {0};
    const yvex_runtime_logits_plan_summary *logits_plan = NULL;
    unsigned long long hidden_bytes, logits_bytes;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !model || !session || !options ||
        options->schema_version != YVEX_RUNTIME_GENERATION_SCHEMA_V5 ||
        (options->backend != YVEX_BACKEND_KIND_CPU &&
         options->backend != YVEX_BACKEND_KIND_CUDA) ||
        options->mode > YVEX_GENERATION_MODE_DSPARK ||
        !options->context_capacity || !options->prefill_chunk_tokens ||
        !options->maximum_new_tokens || !options->maximum_output_bytes ||
        options->trace_policy > YVEX_RUNTIME_TRACE_FULL ||
        options->evidence_profile > YVEX_EXECUTION_EVIDENCE_FORENSIC)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete bounded generation options are required");
    context = yvex_core_calloc(1u, sizeof(*context));
    if (!context)
        return generation_context_refuse(
            err, YVEX_ERR_NOMEM,
            "generation context allocation failed");
    context->model = model;
    context->session = session;
    context->model_view = yvex_runtime_model_view_get(model);
    context->tokenizer = context->model_view ? context->model_view->tokenizer : NULL;
    context->options = *options;
    atomic_init(&context->lifecycle, 0u);
    atomic_init(&context->admission_failures, 0ull);
    if (!context->model_view || !context->tokenizer ||
        !yvex_runtime_session_view_get(session) ||
        yvex_runtime_session_view_get(session)->model != model) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation model, session, and tokenizer are not paired");
        goto failure;
    }
    rc = generation_stops_open(
        context, yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_runtime_sampling_policy_seal(
        &context->options.sampling_policy,
        yvex_tokenizer_vocab_size(context->tokenizer), err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_execution_profile_build(context, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_capacity_build(context, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_runtime_session_configure_persistent_pages(
        session, &context->capacity_plan, &state_failure, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_execution_shape_registry_open(
        &context->execution_shapes, 128ull, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_execution_owners_open(
        context, options, &logits_plan, err);
    if (rc != YVEX_OK) goto failure;
    if (!context->workload_profile.schema_version)
        rc = generation_capacity_workload(context, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_plan_build(context, err);
    if (rc != YVEX_OK) goto failure;
    decode_options.maximum_steps = options->maximum_new_tokens;
    decode_options.cancel_requested = options->cancel_requested;
    decode_options.cancel_context = options->cancel_context;
    rc = yvex_runtime_decode_context_open(
        &context->decode, context->transformer, session, &decode_options, err);
    if (rc != YVEX_OK) goto failure;
    decoder_options.skip_special_tokens = 1;
    decoder_options.require_complete_utf8 = 1;
    decoder_options.cancelled = options->cancel_requested;
    decoder_options.cancel_context = options->cancel_context;
    rc = yvex_tokenizer_decoder_open(
        &context->decoder, context->tokenizer, &decoder_options, err);
    if (rc != YVEX_OK) goto failure;
    rc = yvex_token_sequence_open(
        &context->sequence, options->maximum_new_tokens, err);
    if (rc != YVEX_OK) goto failure;
    context->hidden_count = logits_plan->hidden_width;
    context->logits_count = logits_plan->vocabulary_size;
    if ((context->speculation &&
         !yvex_core_u64_mul(context->hidden_count,
                            YVEX_SPECULATION_MAX_BLOCK + 2ull,
                            &context->hidden_count)) ||
        !yvex_core_u64_mul(context->hidden_count, sizeof(float), &hidden_bytes) ||
        !yvex_core_u64_mul(context->logits_count, sizeof(float), &logits_bytes)) {
        rc = generation_context_refuse(err, YVEX_ERR_NOMEM,
            "generation-local workspace geometry overflowed");
        goto failure;
    }
    context->workspace_bytes = hidden_bytes;
    if ((!context->device_selection &&
         !yvex_core_u64_add(context->workspace_bytes, logits_bytes,
                            &context->workspace_bytes)) ||
        context->workspace_bytes > SIZE_MAX ||
        (options->maximum_host_bytes &&
         context->workspace_bytes > options->maximum_host_bytes)) {
        rc = generation_context_refuse(err, YVEX_ERR_NOMEM,
            "generation-local workspace exceeds its budget");
        goto failure;
    }
    context->hidden = yvex_core_calloc((size_t)context->hidden_count, sizeof(float));
    if (!context->device_selection)
        context->logits_row = yvex_core_calloc((size_t)context->logits_count, sizeof(float));
    if (!context->hidden ||
        (!context->device_selection && !context->logits_row)) {
        rc = generation_context_refuse(err, YVEX_ERR_NOMEM,
            "generation-local workspace allocation failed");
        goto failure;
    }
    if (pthread_mutex_init(&context->drain_mutex, NULL) != 0) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lifecycle mutex initialization failed");
        goto failure;
    }
    context->drain_mutex_ready = 1;
    if (pthread_cond_init(&context->drain_condition, NULL) != 0) {
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation lifecycle condition initialization failed");
        goto failure;
    }
    context->drain_condition_ready = 1;
    context->continuation_allowed = 1;
    *out = context;
    yvex_error_clear(err);
    return YVEX_OK;

failure:
    if (context) {
        yvex_error cleanup;
        yvex_error_clear(&cleanup);
        yvex_token_sequence_close(&context->sequence);
        yvex_tokenizer_decoder_close(&context->decoder);
        (void)yvex_runtime_decode_context_close(&context->decode, &cleanup);
        (void)yvex_runtime_speculation_context_close(
            &context->speculation, &cleanup);
        (void)yvex_runtime_sampling_context_close(&context->sampling, &cleanup);
        (void)yvex_runtime_logits_context_close(&context->logits, &cleanup);
        (void)yvex_runtime_transformer_context_close(
            &context->transformer, &cleanup);
        yvex_execution_shape_registry_close(&context->execution_shapes);
        if (context->drain_condition_ready)
            (void)pthread_cond_destroy(&context->drain_condition);
        if (context->drain_mutex_ready)
            (void)pthread_mutex_destroy(&context->drain_mutex);
        yvex_core_free(context->logits_row);
        yvex_core_free(context->hidden);
        yvex_core_free(context->additional_stops);
        yvex_core_free(context);
    }
    return rc;
}

const yvex_runtime_generation_plan_summary *yvex_runtime_generation_plan_summary_get(
    const yvex_runtime_generation_context *context)
{
    return context ? &context->plan : NULL;
}

int yvex_runtime_generation_context_close(
    yvex_runtime_generation_context **context, yvex_error *err)
{
    yvex_runtime_generation_context *owner;
    unsigned int observed, desired;
    int rc;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    owner = *context;
    observed = atomic_load_explicit(&owner->lifecycle, memory_order_acquire);
    while (!(observed & YVEX_GENERATION_LIFECYCLE_CLOSING)) {
        desired = observed | YVEX_GENERATION_LIFECYCLE_CLOSING;
        if (atomic_compare_exchange_weak_explicit(
                &owner->lifecycle, &observed, desired,
                memory_order_acq_rel, memory_order_acquire))
            break;
    }
    if (owner->drain_mutex_ready) {
        if (pthread_mutex_lock(&owner->drain_mutex) != 0)
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "generation close drain lock failed");
        while (atomic_load_explicit(&owner->lifecycle, memory_order_acquire) &
               YVEX_GENERATION_LIFECYCLE_ACTIVE) {
            if (!owner->drain_condition_ready ||
                pthread_cond_wait(&owner->drain_condition,
                                  &owner->drain_mutex) != 0) {
                (void)pthread_mutex_unlock(&owner->drain_mutex);
                return generation_context_refuse(
                    err, YVEX_ERR_STATE,
                    "generation close drain failed");
            }
        }
        (void)pthread_mutex_unlock(&owner->drain_mutex);
    }
    rc = yvex_runtime_speculation_context_close(&owner->speculation, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_close(&owner->sampling, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_logits_context_close(&owner->logits, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_decode_context_close(&owner->decode, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_transformer_context_close(&owner->transformer, err);
    if (rc != YVEX_OK) return rc;
    yvex_execution_shape_registry_close(&owner->execution_shapes);
    yvex_tokenizer_decoder_close(&owner->decoder);
    yvex_token_sequence_close(&owner->sequence);
    if (owner->drain_condition_ready &&
        pthread_cond_destroy(&owner->drain_condition) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation close condition cleanup failed");
    owner->drain_condition_ready = 0;
    if (owner->drain_mutex_ready &&
        pthread_mutex_destroy(&owner->drain_mutex) != 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation close mutex cleanup failed");
    owner->drain_mutex_ready = 0;
    atomic_store_explicit(&owner->lifecycle,
                          YVEX_GENERATION_LIFECYCLE_CLOSED,
                          memory_order_release);
    yvex_core_free(owner->logits_row);
    yvex_core_free(owner->hidden);
    yvex_core_free(owner->additional_stops);
    memset(owner, 0, sizeof(*owner));
    yvex_core_free(owner);
    *context = NULL;
    yvex_error_clear(err);
    return YVEX_OK;
}
