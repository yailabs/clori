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
#include <yvex/internal/runtime_state_store.h>

static int generation_context_refuse(yvex_error *err, yvex_status status,
                                     const char *reason)
{
    yvex_error_set(err, status, "runtime.generation", reason);
    return status;
}

static int generation_options_valid(
    const yvex_runtime_generation_options *options)
{
    return options &&
           options->schema_version == YVEX_RUNTIME_GENERATION_SCHEMA_V5 &&
           (options->backend == YVEX_BACKEND_KIND_CPU ||
            options->backend == YVEX_BACKEND_KIND_CUDA) &&
           options->mode <= YVEX_GENERATION_MODE_DSPARK &&
           options->workload_kind <= YVEX_EXECUTION_WORKLOAD_FULL_MODEL_RESEARCH &&
           options->context_capacity && options->prefill_chunk_tokens &&
           options->maximum_new_tokens && options->maximum_output_bytes &&
           options->trace_policy <= YVEX_RUNTIME_TRACE_FULL &&
           options->evidence_profile <= YVEX_EXECUTION_EVIDENCE_FORENSIC &&
           (options->continuous_batching == 0 ||
            options->continuous_batching == 1);
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

typedef struct {
    const char *identity;
    unsigned long long maximum_context, hidden_width, vocabulary_size;
    unsigned long long residual_streams, candidate_width;
} generation_semantic_capacity;

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
    const yvex_attention_layer_plan *layers, unsigned long long layer_count,
    const yvex_graph_attention_capacity_plan *capacity, int draft)
{
    unsigned long long layer_index;
    for (layer_index = 0ull; layer_index < layer_count; ++layer_index) {
        const yvex_attention_layer_plan *layer = &layers[layer_index];
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
    generation_capacity_geometry *geometry)
{
    unsigned long long index;
    memset(geometry, 0, sizeof(*geometry));
    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        yvex_execution_state_class_request *state = &geometry->classes[index];
        state->state_class = (yvex_model_state_class)index;
        state->alignment_bytes = 256ull;
        state->kernel_tile_tokens = 1ull;
        state->promotion_granularity_tokens = 1ull;
        state->page_table_entry_bytes = 16ull;
    }
}

static int generation_capacity_graph_geometry(
    yvex_runtime_generation_context *context,
    const generation_semantic_capacity *semantic,
    generation_capacity_geometry *geometry,
    yvex_graph_attention_capacity_plan **workspace_capacity, yvex_error *err)
{
    const yvex_runtime_binding *binding =
        context && context->model_view
            ? context->model_view->compiled_binding : NULL;
    const yvex_attention_summary *summaries[2];
    const yvex_attention_layer_plan *layers[2];
    unsigned long long layer_counts[2];
    unsigned long long plan_index;
    *workspace_capacity = NULL;
    if (!binding)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled attention geometry is unavailable");
    summaries[0] = &binding->attention;
    summaries[1] = binding->summary.draft_layer_count
                       ? &binding->draft_attention : NULL;
    layers[0] = binding->layers;
    layers[1] = binding->draft_layers;
    layer_counts[0] = binding->summary.layer_count;
    layer_counts[1] = binding->summary.draft_layer_count;
    generation_capacity_geometry_initialize(geometry);
    for (plan_index = 0ull; plan_index < 2ull; ++plan_index) {
        yvex_graph_attention_capacity_request request = {0};
        yvex_graph_attention_capacity_plan *capacity = NULL;
        int rc;
        if (!summaries[plan_index] || !layers[plan_index] ||
            !layer_counts[plan_index]) {
            if (!plan_index)
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
        rc = yvex_graph_attention_capacity_plan_build_compiled(
            &capacity, summaries[plan_index], layers[plan_index],
            layer_counts[plan_index], &request, err);
        if (rc == YVEX_OK &&
            !generation_capacity_plan_accumulate(
                geometry, layers[plan_index], layer_counts[plan_index],
                capacity, plan_index != 0ull))
            rc = generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "state-class geometry cannot represent the admitted graph plan");
        if (rc == YVEX_OK && !plan_index) {
            *workspace_capacity = capacity;
            capacity = NULL;
        }
        yvex_graph_attention_capacity_plan_close(&capacity);
        if (rc != YVEX_OK) return rc;
    }
    if (semantic->residual_streams > 1ull) {
        unsigned long long bytes;
        if (!yvex_core_u64_mul(semantic->residual_streams,
                               semantic->hidden_width, &bytes) ||
            !yvex_core_u64_mul(bytes, sizeof(float), &bytes) ||
            !generation_capacity_fixed_add(
                &geometry->classes[YVEX_MODEL_STATE_RESIDUAL_MIXING], 1ull, bytes))
            return generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "residual state geometry overflowed");
    }
    if (geometry->candidate_bytes_per_token) {
        yvex_execution_state_class_request *candidate =
            &geometry->classes[YVEX_MODEL_STATE_CANDIDATE_DELTA];
        candidate->extent = YVEX_EXECUTION_STATE_EXTENT_CANDIDATE;
        candidate->logical_block_tokens = 1ull;
        candidate->bytes_per_block = geometry->candidate_bytes_per_token;
        candidate->promotion_granularity_tokens =
            semantic->candidate_width;
    }
    if (geometry->candidate_bytes_per_token) {
        yvex_execution_state_class_request *prefix =
            &geometry->classes[YVEX_MODEL_STATE_PREFIX_CHECKPOINT];
        prefix->extent = YVEX_EXECUTION_STATE_EXTENT_PREFIX_BUDGET;
        prefix->logical_block_tokens = 1ull;
        prefix->bytes_per_block = 16ull;
        prefix->kernel_tile_tokens = semantic->candidate_width;
        prefix->shared = 1;
        prefix->copy_on_write = 1;
    }
    return YVEX_OK;
}

static int generation_capacity_hardware(
    yvex_runtime_generation_context *context, yvex_backend *backend,
    yvex_runtime_weight_placement placement, int resident,
    unsigned long long *live_available, yvex_error *err)
{
    yvex_backend_device_info device = {0};
    yvex_backend_cuda_attention_graph_summary cuda = {0};
    const char *placement_name = "host";
    long page_bytes;
    unsigned long long system_total, total, available, reserve_basis;
    int process_limited, shared_system_domain = 0, rc;
    if (!context || !live_available ||
        (context->options.backend == YVEX_BACKEND_KIND_CUDA && !backend) ||
        (backend && yvex_backend_get_device_info(backend, &device, err) != YVEX_OK) ||
        (page_bytes = sysconf(_SC_PAGESIZE)) <= 0)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "memory-admission hardware facts are unavailable");
    if (!backend) device.kind = YVEX_BACKEND_KIND_CPU;
    if (device.kind != context->options.backend)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "capacity backend differs from the requested execution backend");
    if (!yvex_runtime_private_memory_capacity(
            &system_total, &available, &process_limited))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "live process memory capacity is unavailable");
    if (device.kind == YVEX_BACKEND_KIND_CUDA) {
        shared_system_domain =
            device.unified_addressing && device.total_memory_bytes == system_total &&
            ((placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED &&
              device.managed_memory) ||
             (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED &&
              (resident ||
               yvex_backend_resident_map_readonly_supported(backend))));
        total = shared_system_domain ? system_total : device.total_memory_bytes;
        if (!shared_system_domain && device.free_memory_bytes < available)
            available = device.free_memory_bytes;
    } else {
        total = system_total;
    }
    *live_available = available;
    memset(&context->hardware_profile, 0, sizeof(context->hardware_profile));
    context->hardware_profile.schema_version =
        YVEX_EXECUTION_HARDWARE_PROFILE_SCHEMA_V1;
    context->hardware_profile.backend = device.kind;
    context->hardware_profile.admitted_fact_mask =
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_MEMORY) |
        YVEX_EXECUTION_HARDWARE_FACT_BIT(YVEX_EXECUTION_HARDWARE_FACT_PAGING);
    context->hardware_profile.device_index = backend ? device.device_index : 0;
    context->hardware_profile.compute_major = device.compute_capability_major;
    context->hardware_profile.compute_minor = device.compute_capability_minor;
    context->hardware_profile.total_memory_bytes = total;
    context->hardware_profile.usable_memory_bytes = total;
    if (device.kind == YVEX_BACKEND_KIND_CUDA &&
        context->options.maximum_device_bytes &&
        context->options.maximum_device_bytes <
            context->hardware_profile.usable_memory_bytes)
        context->hardware_profile.usable_memory_bytes =
            context->options.maximum_device_bytes;
    if ((device.kind == YVEX_BACKEND_KIND_CPU ||
         placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED ||
         placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) &&
        context->options.maximum_host_bytes &&
        context->options.maximum_host_bytes <
            context->hardware_profile.usable_memory_bytes)
        context->hardware_profile.usable_memory_bytes =
            context->options.maximum_host_bytes;
    reserve_basis = system_total;
    if (context->options.maximum_host_bytes &&
        context->options.maximum_host_bytes < reserve_basis)
        reserve_basis = context->options.maximum_host_bytes;
    context->system_capacity_bytes = reserve_basis;
    context->system_reserve_bytes =
        yvex_runtime_private_system_reserve(reserve_basis);
    context->hardware_profile.host_page_bytes = (unsigned long long)page_bytes;
    context->hardware_profile.device_page_bytes = (unsigned long long)page_bytes;
    context->hardware_profile.unified_addressing = device.unified_addressing;
    context->hardware_profile.coherent_host_memory = shared_system_domain;
    if (device.kind == YVEX_BACKEND_KIND_CUDA && resident) {
        if (yvex_backend_cuda_attention_graph_summary_get(
                backend, &cuda, err) != YVEX_OK ||
            !cuda.kernel_bundle_architecture[0] ||
            !yvex_sha256_hex_valid(cuda.cuda_build_identity))
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "kernel-bundle hardware facts are unavailable");
        rc = yvex_backend_bandwidth_probe(
            backend, &context->bandwidth_evidence, err);
        if (rc != YVEX_OK) return rc;
        if (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_CUDA_MANAGED) {
            context->hardware_profile.sustainable_read_bytes_per_second =
                context->bandwidth_evidence.sustainable_read_bytes_per_second;
            placement_name = "managed";
        } else if (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_ARTIFACT_MAPPED) {
            context->hardware_profile.sustainable_read_bytes_per_second =
                context->bandwidth_evidence.sustainable_coherent_host_bytes_per_second;
            placement_name = "artifact-map";
        } else if (placement == YVEX_RUNTIME_WEIGHT_PLACEMENT_HOST_LOCKED) {
            context->hardware_profile.sustainable_read_bytes_per_second =
                context->bandwidth_evidence.sustainable_coherent_host_bytes_per_second;
            placement_name = "hostmap";
        } else {
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "model residency has no admitted bandwidth placement");
        }
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
                       "cuda-%s-%s", cuda.kernel_bundle_architecture,
                       placement_name);
    } else if (device.kind == YVEX_BACKEND_KIND_CUDA) {
        yvex_core_text_copy(context->hardware_profile.name,
                            sizeof(context->hardware_profile.name),
                            "cuda-capacity-preflight");
    } else {
        yvex_core_text_copy(context->hardware_profile.name,
                            sizeof(context->hardware_profile.name),
                            "cpu-memory");
    }
    (void)process_limited;
    return yvex_execution_hardware_profile_seal(
        &context->hardware_profile, err);
}

static int generation_capacity_workload(
    yvex_runtime_generation_context *context, yvex_error *err)
{
    static const char *const names[] = {
        "interactive-latency", "balanced-serving", "long-context",
        "deep-context", "full-model-research"
    };
    const yvex_speculation_family_policy *speculation = NULL;
    if (context && context->options.mode == YVEX_GENERATION_MODE_DSPARK &&
        context->model_view && context->model_view->compiled_binding &&
        !yvex_runtime_binding_policies(
            context->model_view->compiled_binding, NULL, NULL, &speculation))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled speculation workload geometry is unavailable");
    memset(&context->workload_profile, 0, sizeof(context->workload_profile));
    context->workload_profile.schema_version =
        YVEX_EXECUTION_WORKLOAD_PROFILE_SCHEMA_V1;
    context->workload_profile.kind = context->options.workload_kind;
    context->workload_profile.minimum_session_context =
        context->options.context_capacity;
    context->workload_profile.requested_session_context =
        context->options.context_capacity;
    context->workload_profile.concurrent_sequences =
        context->options.concurrent_sequences;
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
                    : 1ull;
    if (!context->system_capacity_bytes || !context->system_reserve_bytes)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "runtime system-reserve capacity is unavailable");
    context->workload_profile.system_reserve_bytes =
        context->system_reserve_bytes;
    context->workload_profile.latency_priority =
        context->options.workload_kind ==
        YVEX_EXECUTION_WORKLOAD_INTERACTIVE_LATENCY;
    context->workload_profile.continuous_batching =
        context->options.continuous_batching;
    yvex_core_text_copy(context->workload_profile.name,
                        sizeof(context->workload_profile.name),
                        names[context->options.workload_kind]);
    return yvex_execution_workload_profile_seal(
        &context->workload_profile, err);
}

static int generation_physical_row_capacity(
    const yvex_runtime_generation_context *context,
    unsigned long long *capacity, yvex_error *err)
{
    const yvex_speculation_family_policy *speculation = NULL;
    unsigned long long draft_width = 0ull;
    if (capacity) *capacity = 0ull;
    if (!context || !capacity || !context->options.prefill_chunk_tokens ||
        !context->model_view || !context->model_view->compiled_binding ||
        !yvex_runtime_binding_policies(
            context->model_view->compiled_binding, NULL, NULL, &speculation))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled physical row geometry is unavailable");
    *capacity = context->options.prefill_chunk_tokens;
    if (context->options.mode == YVEX_GENERATION_MODE_DSPARK) {
        if (!speculation ||
            !yvex_core_u64_add(speculation->block_size, 2ull, &draft_width))
            return generation_context_refuse(
                err, YVEX_ERR_BOUNDS,
                "compiled speculative execution width is invalid");
        if (draft_width > *capacity) *capacity = draft_width;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int generation_shape_registry_capacity(
    const yvex_runtime_generation_context *context,
    unsigned long long *capacity, yvex_error *err)
{
    unsigned long long width, classes;
    if (capacity) *capacity = 0ull;
    if (generation_physical_row_capacity(context, &width, err) != YVEX_OK)
        return yvex_error_code(err);
    /* Workspace generations replace, rather than multiply, this admitted class envelope. */
    if (width > YVEX_EXECUTION_SHAPE_MAX_WIDTH ||
        !yvex_core_u64_mul(width, YVEX_EXECUTION_PHASE_COUNT, &classes) ||
        !yvex_core_u64_mul(classes, YVEX_EXECUTION_CONTEXT_NEAR_CAPACITY + 1ull,
                           &classes) ||
        !yvex_core_u64_mul(classes, 2ull, &classes) ||
        !yvex_core_u64_mul(classes, 2ull, &classes))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "compiled execution-shape envelope exceeds bounded geometry");
    *capacity = classes;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int generation_sampling_workspace(
    const yvex_runtime_generation_context *context, yvex_backend *backend,
    unsigned long long vocabulary_size, unsigned long long proposal_width,
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
    if (!vocabulary_size || !operations || !operations->workspace_required ||
        operations->workspace_required(
            vocabulary_size, &selection, err) != YVEX_OK)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "device stochastic workspace geometry is unavailable");
    if (context->options.mode == YVEX_GENERATION_MODE_DSPARK &&
        (!proposal_width ||
         !operations->speculation_workspace_required ||
         operations->speculation_workspace_required(
             vocabulary_size, proposal_width,
             &speculation, err) != YVEX_OK))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "device stochastic speculation workspace geometry is unavailable");
    *workspace = speculation > selection ? speculation : selection;
    return YVEX_OK;
}

static int generation_moe_workspace(
    const yvex_runtime_generation_context *context, yvex_backend *backend,
    unsigned long long target_rows, unsigned long long draft_rows,
    unsigned long long *workspace, yvex_error *err)
{
    const yvex_backend_moe_operations *operations;
    unsigned long long layer_max = 0ull, maximum_layers = 0ull;
    unsigned long long completion_bytes;
    unsigned int draft;
    if (workspace) *workspace = 0ull;
    if (!context || !workspace || !target_rows)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "compiled MoE workspace facts are incomplete");
    if (context->options.backend != YVEX_BACKEND_KIND_CUDA) return YVEX_OK;
    operations = yvex_backend_moe_operations_get(backend);
    for (draft = 0u; draft < 2u; ++draft) {
        const yvex_moe_plan *plan = yvex_compiled_model_plan_moe(
            context->model_view->compiled_plan, draft != 0u);
        const yvex_moe_plan_summary *summary = yvex_moe_plan_summary_get(plan);
        unsigned long long rows = draft ? draft_rows : target_rows;
        unsigned long long index;
        if (!summary || !rows) continue;
        if (!operations || !operations->workspace_required)
            return generation_context_refuse(
                err, YVEX_ERR_UNSUPPORTED,
                "CUDA MoE workspace capability is unavailable");
        if (summary->layer_count > maximum_layers)
            maximum_layers = summary->layer_count;
        for (index = 0ull; index < summary->layer_count; ++index) {
            unsigned long long bytes;
            int rc = operations->workspace_required(
                yvex_moe_plan_layer_at(plan, index), rows, &bytes, err);
            if (rc != YVEX_OK) return rc;
            if (bytes > layer_max) layer_max = bytes;
        }
    }
    if (!maximum_layers) return YVEX_OK;
    if (!yvex_core_u64_mul(
            maximum_layers, sizeof(yvex_moe_device_completion_slot),
            &completion_bytes) || !layer_max ||
        !yvex_core_u64_add(layer_max, completion_bytes, workspace))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "CUDA MoE workspace extent overflowed");
    return YVEX_OK;
}

static int generation_attention_workspace(
    const yvex_runtime_generation_context *context,
    const yvex_graph_attention_capacity_plan *capacity,
    unsigned long long physical_rows, unsigned long long *workspace,
    yvex_error *err)
{
    const yvex_runtime_binding *binding =
        context ? context->model_view->compiled_binding : NULL;
    const yvex_attention_summary *summaries[2];
    const yvex_attention_layer_plan *layers[2];
    unsigned long long layer_counts[2], plan_count, plan_index;
    int deferred = context &&
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
        runtime_attention_evidence(context->options.evidence_profile) ==
            YVEX_ATTENTION_EVIDENCE_NONE;
    if (workspace) *workspace = 0ull;
    if (!binding || !capacity || !physical_rows || !workspace)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "compiled attention workspace facts are incomplete");
    summaries[0] = &binding->attention;
    summaries[1] = &binding->draft_attention;
    layers[0] = binding->layers;
    layers[1] = binding->draft_layers;
    layer_counts[0] = binding->summary.layer_count;
    layer_counts[1] = binding->summary.draft_layer_count;
    plan_count = context->options.mode == YVEX_GENERATION_MODE_DSPARK ? 2ull : 1ull;
    for (plan_index = 0ull; plan_index < plan_count; ++plan_index) {
        yvex_graph_attention_capacity_plan *owned_capacity = NULL;
        const yvex_graph_attention_capacity_plan *selected_capacity = capacity;
        unsigned int mode;
        int rc = YVEX_OK;
        if (!summaries[plan_index] || !layers[plan_index] ||
            !layer_counts[plan_index])
            return generation_context_refuse(
                err, YVEX_ERR_STATE,
                "compiled attention workspace plan is unavailable");
        if (plan_index) {
            yvex_graph_attention_capacity_request request = {
                .scope = YVEX_ATTENTION_PROBE_SCOPE_FULL,
                .token_count = context->options.context_capacity,
                .execution_count = 1ull,
                .use_requested_position = 1,
            };
            rc = yvex_graph_attention_capacity_plan_build_compiled(
                &owned_capacity, summaries[plan_index], layers[plan_index],
                layer_counts[plan_index], &request, err);
            selected_capacity = owned_capacity;
        }
        for (mode = YVEX_ATTENTION_EXECUTION_EAGER;
             rc == YVEX_OK && mode <= YVEX_ATTENTION_EXECUTION_FULL; ++mode) {
            unsigned long long bytes;
            rc = yvex_runtime_private_attention_workspace_required(
                summaries[plan_index], layers[plan_index],
                layer_counts[plan_index], selected_capacity,
                (yvex_attention_execution_mode)mode,
                YVEX_ATTENTION_OPERATION_ENVELOPE,
                runtime_attention_evidence(context->options.evidence_profile),
                physical_rows, deferred, &bytes, err);
            if (rc == YVEX_OK && bytes > *workspace) *workspace = bytes;
        }
        yvex_graph_attention_capacity_plan_close(&owned_capacity);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

static int generation_capacity_build_for(
    yvex_runtime_generation_context *context, yvex_backend *backend,
    yvex_runtime_weight_placement placement, unsigned long long model_bytes,
    int model_resident, unsigned long long transient_bytes,
    yvex_graph_attention_capacity_plan **workspace_capacity,
    unsigned long long *required_out, unsigned long long *available_out,
    yvex_error *err)
{
    const yvex_transformer_plan_summary *transformer =
        yvex_transformer_plan_summary_get(
            yvex_compiled_model_plan_transformer(
                context->model_view->compiled_plan, 0));
    const yvex_speculation_family_policy *speculation = NULL;
    generation_semantic_capacity semantic;
    yvex_compiled_context_envelope context_envelope;
    generation_capacity_geometry geometry;
    yvex_execution_state_class_request states[YVEX_MODEL_STATE_CLASS_COUNT];
    yvex_execution_capacity_plan_request request = {0};
    unsigned long long workspace, sampling_workspace = 0ull;
    unsigned long long attention_workspace = 0ull, moe_workspace = 0ull;
    unsigned long long physical_rows, draft_rows = 0ull, index, count = 0ull;
    unsigned long long graph_bytes, scheduler_bytes, live_available, live_required;
    int rc;
    if (required_out) *required_out = 0ull;
    if (available_out) *available_out = 0ull;
    if (generation_capacity_hardware(
            context, backend, placement, model_resident,
            &live_available, err) != YVEX_OK)
        return yvex_error_code(err);
    if (available_out) *available_out = live_available;
    memset(&semantic, 0, sizeof(semantic));
    if (!context->model_view->binding ||
        !context->model_view->compiled_plan ||
        !yvex_runtime_binding_policies(
            context->model_view->compiled_binding, NULL, NULL, &speculation))
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "generation requires one sealed semantic model and compiled execution plan");
    semantic.identity = context->model_view->binding->model_execution_identity;
    semantic.maximum_context =
        context->model_view->binding->semantic_maximum_context;
    if (!yvex_sha256_hex_valid(semantic.identity) ||
        !semantic.maximum_context)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled semantic context capability is unavailable");
    if (context->options.context_capacity > semantic.maximum_context)
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "requested context exceeds the model-authored semantic maximum");
    if (!transformer)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled transformer geometry is unavailable");
    semantic.hidden_width = transformer->hidden_width;
    semantic.vocabulary_size = transformer->vocabulary_size;
    semantic.residual_streams = transformer->residual_streams;
    semantic.candidate_width = speculation && speculation->block_size
                                   ? speculation->block_size + 1ull : 1ull;
    if (generation_capacity_workload(context, err) != YVEX_OK) return yvex_error_code(err);
    if (yvex_compiled_model_plan_context_envelope(
            context->model_view->compiled_plan, semantic.identity,
            semantic.maximum_context, &context_envelope, err) != YVEX_OK ||
        yvex_compiled_context_envelope_admit(
            &context_envelope, context->options.context_capacity,
            context->options.mode == YVEX_GENERATION_MODE_DSPARK, err) != YVEX_OK)
        return yvex_error_code(err);
    if (generation_capacity_graph_geometry(
            context, &semantic, &geometry, workspace_capacity, err) != YVEX_OK)
        return yvex_error_code(err);
    if (generation_physical_row_capacity(
            context, &physical_rows, err) != YVEX_OK)
        return yvex_error_code(err);
    if (context->options.mode == YVEX_GENERATION_MODE_DSPARK &&
        !yvex_core_u64_add(semantic.candidate_width, 1ull, &draft_rows))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "compiled draft workspace row extent overflowed");
    for (index = 0ull; index < YVEX_MODEL_STATE_CLASS_COUNT; ++index) {
        if (!geometry.classes[index].bytes_per_block) continue;
        states[count++] = geometry.classes[index];
        request.semantic_state_class_mask |= YVEX_MODEL_STATE_CLASS_BIT(index);
    }
    if (!count)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "compiled graph exposes no persistent-state geometry");
    if (!yvex_core_u64_mul(context->options.prefill_chunk_tokens,
                           semantic.hidden_width, &workspace) ||
        !yvex_core_u64_mul(workspace, sizeof(float) * 8ull, &workspace) ||
        !workspace)
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "execution workspace geometry overflowed");
    if (generation_sampling_workspace(
            context, backend, semantic.vocabulary_size,
            semantic.candidate_width ? semantic.candidate_width - 1ull : 0ull,
            &sampling_workspace, err) != YVEX_OK)
        return yvex_error_code(err);
    context->sampling_workspace_bytes = sampling_workspace;
    if (sampling_workspace > workspace) workspace = sampling_workspace;
    if (generation_attention_workspace(
            context, *workspace_capacity, physical_rows,
            &attention_workspace, err) != YVEX_OK ||
        generation_moe_workspace(
            context, backend, context->options.prefill_chunk_tokens,
            draft_rows, &moe_workspace, err) != YVEX_OK)
        return yvex_error_code(err);
    if (attention_workspace > workspace) workspace = attention_workspace;
    if (moe_workspace > workspace) workspace = moe_workspace;
    request.schema_version = YVEX_EXECUTION_CAPACITY_PLAN_SCHEMA_V1;
    request.model_execution_identity = semantic.identity;
    request.semantic_maximum_context = semantic.maximum_context;
    request.candidate_width = semantic.candidate_width;
    request.hardware = &context->hardware_profile;
    request.workload = &context->workload_profile;
    request.model_bytes = model_bytes;
    request.state_classes = states;
    request.state_class_count = count;
    if (!yvex_core_u64_mul(workspace,
                           context->workload_profile.concurrent_sequences,
                           &request.workspace_bytes) ||
        !yvex_core_u64_mul(1024ull * 1024ull,
                           context->workload_profile.concurrent_sequences,
                           &graph_bytes) ||
        !yvex_core_u64_mul(1024ull * 1024ull,
                           context->workload_profile.concurrent_sequences,
                           &scheduler_bytes))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "multi-sequence fixed-resource accounting overflowed");
    request.scheduler_bytes = scheduler_bytes;
    request.graph_bytes = graph_bytes;
    if (!request.model_bytes)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "resident model byte extent is unavailable for capacity admission");
    rc = yvex_execution_capacity_plan_build(
        &request, &context->capacity_plan, err);
    if (rc != YVEX_OK) return rc;
    if (context->capacity_plan.required_bytes < request.model_bytes)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "capacity plan does not cover resident model bytes");
    live_required = context->capacity_plan.required_bytes;
    if (model_resident)
        live_required -= request.model_bytes;
    else if (!yvex_core_u64_add(live_required, transient_bytes,
                                &live_required))
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            "transient admission peak accounting overflowed");
    if (required_out) *required_out = live_required;
    if (live_required > live_available)
        return generation_context_refuse(
            err, YVEX_ERR_BOUNDS,
            model_resident
                ? "live process memory cannot preserve the admitted runtime reserve"
                : "pre-residency peak cannot preserve the admitted runtime reserve");
    return YVEX_OK;
}

static int generation_capacity_build(
    yvex_runtime_generation_context *context,
    yvex_graph_attention_capacity_plan **workspace_capacity, yvex_error *err)
{
    const yvex_runtime_session_view *session =
        context ? yvex_runtime_session_view_get(context->session) : NULL;
    yvex_runtime_residency_summary residency = {0};
    unsigned long long model_bytes;
    if (!context || !context->model_view || !context->model_view->residency ||
        !session || !session->backend ||
        yvex_runtime_residency_snapshot(
            context->model_view->residency, &residency,
            NULL, NULL, err) != YVEX_OK || !residency.encoded_bytes)
        return generation_context_refuse(
            err, YVEX_ERR_STATE,
            "model residency placement facts are unavailable");
    model_bytes = residency.cuda_addressable_bytes
                      ? residency.cuda_addressable_bytes
                      : residency.host_resident_bytes
                            ? residency.host_resident_bytes
                            : residency.artifact_backed_bytes
                                  ? residency.artifact_backed_bytes
                                  : residency.encoded_bytes;
    return generation_capacity_build_for(
        context, session->backend, residency.placement,
        model_bytes, 1, 0ull, workspace_capacity,
        NULL, NULL, err);
}

int yvex_runtime_private_generation_capacity_preflight(
    const yvex_runtime_binding *binding, yvex_backend *backend,
    const yvex_runtime_generation_options *options,
    unsigned long long *required_bytes, unsigned long long *available_bytes,
    yvex_error *err)
{
    yvex_runtime_generation_context context = {0};
    yvex_runtime_model_view view = {0};
    yvex_graph_attention_capacity_plan *workspace_capacity = NULL;
    yvex_runtime_weight_placement placement;
    unsigned long long transient_bytes, model_bytes;
    int rc;
    if (required_bytes) *required_bytes = 0ull;
    if (available_bytes) *available_bytes = 0ull;
    if (!binding || !generation_options_valid(options) ||
        !required_bytes || !available_bytes ||
        !runtime_binding_maximum_tensor_bytes(
            binding, &transient_bytes))
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "complete startup capacity facts are required");
    view.binding = &binding->summary;
    view.compiled_binding = binding;
    view.compiled_plan = binding->plan;
    context.model_view = &view;
    context.options = *options;
    if (!context.options.concurrent_sequences)
        context.options.concurrent_sequences = 1ull;
    if (context.options.prefill_chunk_tokens > context.options.context_capacity)
        context.options.prefill_chunk_tokens = context.options.context_capacity;
    rc = yvex_runtime_private_weight_placement_select(
        binding, options->backend, backend, &placement, err);
    if (rc == YVEX_OK)
        rc = yvex_runtime_private_residency_backing_bytes(
            binding, backend, placement, &model_bytes, err);
    if (rc == YVEX_OK)
        rc = generation_capacity_build_for(
            &context, backend, placement, model_bytes,
            0, transient_bytes, &workspace_capacity, required_bytes,
            available_bytes, err);
    yvex_graph_attention_capacity_plan_close(&workspace_capacity);
    return rc;
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
        (void)snprintf(hardware, sizeof(hardware), "%s-cuda-sm%d%d",
                       cuda.kernel_bundle_native ? "native" : "portable",
                       session.compute_capability_major,
                       session.compute_capability_minor);
    } else {
        yvex_core_text_copy(hardware, sizeof(hardware), "portable-cpu");
    }
    request.schema_version = YVEX_COMPILED_EXECUTION_PROFILE_SCHEMA_V2;
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
    request.execution_class =
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
                cuda.kernel_bundle_native &&
                context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION
            ? YVEX_EXECUTION_CLASS_DEVICE_NATIVE
            : YVEX_EXECUTION_CLASS_PORTABLE_REFERENCE;
    request.sampling_resolution =
        context->options.sampling_policy.strategy == YVEX_SAMPLING_STRATEGY_GREEDY ||
                generation_device_stochastic(context, session_view->backend)
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.moe_resolution =
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
                cuda.kernel_bundle_native &&
                context->options.evidence_profile == YVEX_EXECUTION_EVIDENCE_PRODUCTION &&
                yvex_backend_moe_operations_get(session_view->backend) != NULL
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
    request.attention_resolution =
        context->options.backend == YVEX_BACKEND_KIND_CUDA &&
                binding->capabilities.cuda_full_graph_implemented &&
                graph.state == YVEX_BACKEND_CUDA_GRAPH_OPEN &&
                graph.edge_inventory_available && graph.async_memory_available &&
                graph.async_copy_available && graph.pinned_host_memory_available
            ? YVEX_EXECUTION_RESOLUTION_EXACT
            : YVEX_EXECUTION_RESOLUTION_COMPATIBLE_DEGRADED;
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
    const yvex_runtime_logits_plan_summary **logits_plan,
    unsigned long long *workspace_bytes, yvex_error *err)
{
    yvex_runtime_transformer_options transformer = {0};
    yvex_runtime_logits_options logits = {0};
    yvex_runtime_sampling_options sampling = {0};
    yvex_runtime_speculation_options speculation = {0};
    const yvex_runtime_session_view *session_view =
        yvex_runtime_session_view_get(context->session);
    unsigned long long draft_workspace = 0ull;
    int device_selection = session_view &&
        generation_device_selection(context, session_view->backend);
    int rc;

    context->device_selection = device_selection;
    *workspace_bytes = 0ull;

    transformer.maximum_host_bytes = options->maximum_host_bytes;
    transformer.maximum_device_bytes = options->maximum_device_bytes;
    transformer.context_capacity = options->context_capacity;
    transformer.workspace_token_capacity = options->prefill_chunk_tokens;
    transformer.minimum_device_workspace_bytes = context->sampling_workspace_bytes;
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
        &context->transformer, context->model, context->session, &transformer,
        workspace_bytes, err);
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
    speculation.prefill_chunk_tokens = options->prefill_chunk_tokens;
    speculation.maximum_host_bytes = options->maximum_host_bytes;
    speculation.maximum_device_bytes = options->maximum_device_bytes;
    speculation.cancel_requested = options->cancel_requested;
    speculation.cancel_context = options->cancel_context;
    speculation.execution_profile = &context->execution_profile;
    speculation.shape_registry = context->execution_shapes;
    rc = yvex_runtime_speculation_context_open(
        &context->speculation, context->model, context->session,
        context->transformer, context->logits, context->sampling,
        &context->options.sampling_policy, &speculation, &draft_workspace, err);
    if (rc == YVEX_OK && draft_workspace > *workspace_bytes)
        *workspace_bytes = draft_workspace;
    return rc;
}

int yvex_runtime_generation_context_summary_copy(
    const yvex_runtime_generation_context *context,
    yvex_runtime_generation_context_summary *summary, yvex_error *err)
{
    yvex_runtime_generation_context *mutable =
        (yvex_runtime_generation_context *)context;
    yvex_runtime_sampling_context_summary sampling;
    yvex_token_sequence_summary sequence;
    unsigned int lifecycle;
    int rc;
    if (!context || !summary)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "generation context and snapshot output are required");
    rc = yvex_runtime_private_generation_enter(mutable, err);
    if (rc != YVEX_OK) return rc;
    memset(summary, 0, sizeof(*summary));
    rc = yvex_runtime_sampling_context_snapshot(context->sampling, &sampling, err);
    if (rc == YVEX_OK)
        rc = yvex_token_sequence_summary_get(context->sequence, &sequence, err);
    lifecycle = atomic_load_explicit(&context->lifecycle, memory_order_acquire);
    if (rc == YVEX_OK) {
        summary->schema_version = YVEX_RUNTIME_GENERATION_SCHEMA_V3;
        summary->open = !(lifecycle & YVEX_GENERATION_LIFECYCLE_CLOSING);
        summary->busy = 0;
        summary->closing =
            (lifecycle & YVEX_GENERATION_LIFECYCLE_CLOSING) != 0u;
        summary->execution_count = context->execution_count;
        summary->failure_count = context->failure_count +
            atomic_load_explicit(&context->admission_failures,
                                 memory_order_relaxed);
        summary->cancellation_count = context->cancellation_count;
        summary->token_capacity = context->options.maximum_new_tokens;
        summary->text_capacity = context->options.maximum_output_bytes;
        summary->workspace_bytes = context->workspace_bytes +
                                   sampling.workspace_bytes;
        summary->concurrent_sequences = context->capacity_plan.concurrent_sequences;
        summary->capacity_required_bytes = context->capacity_plan.required_bytes;
        summary->capacity_unreserved_bytes = context->capacity_plan.unreserved_bytes;
        summary->continuous_batching =
            context->workload_profile.continuous_batching;
        yvex_runtime_identity_copy(summary->generation_plan_identity,
                                   context->plan.generation_plan_identity);
        yvex_runtime_identity_copy(summary->capacity_plan_identity,
                                   context->capacity_plan.identity);
        yvex_runtime_identity_copy(summary->token_sequence_identity,
                                   sequence.state_identity);
        yvex_runtime_identity_copy(summary->rng_state_identity,
                                   sampling.rng_state_identity);
    }
    yvex_runtime_private_generation_leave(mutable, rc, 0);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_generation_context_open(
    yvex_runtime_generation_context **out, yvex_runtime_model *model,
    yvex_runtime_execution_session *session,
    const yvex_runtime_generation_options *options, yvex_error *err)
{
    yvex_runtime_generation_context *context = NULL;
    yvex_runtime_decode_options decode_options = {0};
    yvex_runtime_model_failure state_failure = {0};
    yvex_runtime_model_failure workspace_failure = {0};
    yvex_graph_attention_capacity_plan *workspace_capacity = NULL;
    yvex_tokenizer_decode_options decoder_options = {0};
    const yvex_runtime_logits_plan_summary *logits_plan = NULL;
    unsigned long long hidden_bytes, logits_bytes, execution_workspace = 0ull;
    unsigned long long physical_rows = 0ull;
    unsigned long long shape_capacity = 0ull;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!out || !model || !session || !generation_options_valid(options))
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
    if (!context->options.concurrent_sequences)
        context->options.concurrent_sequences = 1ull;
    if (context->options.prefill_chunk_tokens > context->options.context_capacity)
        context->options.prefill_chunk_tokens = context->options.context_capacity;
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
    rc = generation_capacity_build(context, &workspace_capacity, err);
    if (rc != YVEX_OK) goto failure;
    if (context->capacity_plan.schema_version) {
        rc = yvex_runtime_session_configure_persistent_pages(
            session, &context->capacity_plan, &state_failure, err);
        if (rc != YVEX_OK) goto failure;
    }
    rc = generation_shape_registry_capacity(context, &shape_capacity, err);
    if (rc == YVEX_OK)
        rc = yvex_execution_shape_registry_open(
            &context->execution_shapes, shape_capacity, err);
    if (rc != YVEX_OK) goto failure;
    rc = generation_execution_owners_open(
        context, &context->options, &logits_plan, &execution_workspace, err);
    if (rc == YVEX_OK)
        rc = generation_physical_row_capacity(context, &physical_rows, err);
    if (rc != YVEX_OK) goto failure;
    if (context->options.backend == YVEX_BACKEND_KIND_CUDA)
        rc = yvex_runtime_session_prepare_attention_workspace(
            context->session,
            context->execution_profile.attention_resolution !=
                YVEX_EXECUTION_RESOLUTION_EXACT
                ? YVEX_RUNTIME_MODE_EAGER : YVEX_RUNTIME_MODE_FULL,
            YVEX_RUNTIME_SCOPE_ATTENTION_ENVELOPE, YVEX_ATTENTION_EVIDENCE_NONE,
            workspace_capacity, physical_rows, execution_workspace,
            &workspace_failure, err);
    yvex_graph_attention_capacity_plan_close(&workspace_capacity);
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
    yvex_graph_attention_capacity_plan_close(&workspace_capacity);
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

static int generation_checkpoint_identity(
    const yvex_runtime_generation_checkpoint *checkpoint,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_sha256_init(&hash);
    if (!checkpoint ||
        !yvex_sha256_update_text(&hash,
                                 "yvex.runtime.generation.checkpoint.v1") ||
        !yvex_sha256_update_u64(&hash, checkpoint->schema_version) ||
        !yvex_sha256_update_text(&hash,
                                 checkpoint->generation_plan_identity) ||
        !yvex_sha256_update_text(&hash,
                                 checkpoint->sampling.checkpoint_identity) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_runtime_generation_context_checkpoint(
    yvex_runtime_generation_context *context,
    yvex_runtime_generation_checkpoint *checkpoint, yvex_error *err)
{
    int rc;
    if (checkpoint) memset(checkpoint, 0, sizeof(*checkpoint));
    if (!context || !checkpoint)
        return generation_context_refuse(
            err, YVEX_ERR_INVALID_ARG,
            "generation checkpoint output is required");
    rc = yvex_runtime_private_generation_enter(context, err);
    if (rc != YVEX_OK) return rc;
    checkpoint->schema_version = YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1;
    yvex_runtime_identity_copy(checkpoint->generation_plan_identity,
                               context->plan.generation_plan_identity);
    rc = yvex_runtime_sampling_context_checkpoint(
        context->sampling, &checkpoint->sampling, err);
    if (rc == YVEX_OK &&
        !generation_checkpoint_identity(checkpoint,
                                        checkpoint->checkpoint_identity))
        rc = generation_context_refuse(
            err, YVEX_ERR_STATE, "generation checkpoint identity failed");
    if (rc != YVEX_OK) memset(checkpoint, 0, sizeof(*checkpoint));
    yvex_runtime_private_generation_leave(context, rc, 0);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_runtime_generation_context_restore(
    yvex_runtime_generation_context *context,
    const yvex_runtime_generation_checkpoint *checkpoint, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    int rc;
    if (!context || !checkpoint)
        return generation_context_refuse(err, YVEX_ERR_INVALID_ARG,
                                         "generation checkpoint is required");
    rc = yvex_runtime_private_generation_enter(context, err);
    if (rc != YVEX_OK) return rc;
    if (checkpoint->schema_version !=
            YVEX_RUNTIME_GENERATION_CHECKPOINT_SCHEMA_V1 ||
        strcmp(checkpoint->generation_plan_identity,
               context->plan.generation_plan_identity) != 0 ||
        !generation_checkpoint_identity(checkpoint, identity) ||
        strcmp(identity, checkpoint->checkpoint_identity) != 0)
        rc = generation_context_refuse(
            err, YVEX_ERR_FORMAT,
            "generation checkpoint is incompatible or corrupt");
    if (rc == YVEX_OK)
        rc = yvex_runtime_sampling_context_restore(
            context->sampling, &checkpoint->sampling, err);
    yvex_runtime_private_generation_leave(context, rc, 0);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
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
