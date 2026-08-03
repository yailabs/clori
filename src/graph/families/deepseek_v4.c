/*
 * Compose admitted generic graph mechanisms for the DeepSeek-V4 schedule.
 *
 * Planning reads no payload; incomplete attention remains unsupported. Family evidence.
 */
#include "src/graph/private.h"
#include <yvex/internal/families/deepseek_v4.h>
#include <yvex/internal/graph_state.h>
#include <yvex/internal/moe.h>
#include <yvex/internal/runtime.h>
#include <yvex/internal/logits.h>
#include <yvex/internal/transformer.h>
enum { DEEPSEEK_ATTENTION_CSA_RATIO = 4, DEEPSEEK_ATTENTION_HCA_RATIO = 128 };
static const yvex_attention_cpu_options cpu_options_template = {
    .token_count = 1ull, .local_history_tokens = 4ull, .compressed_history_tokens = 8ull,
    .max_q_b_rows = 128ull, .max_kv_rows = 512ull,
    .max_compressor_rows = 32ull, .max_indexer_rows = 64ull,
    .scratch_limit_bytes = 64ull * 1024ull * 1024ull, .evidence_level = YVEX_ATTENTION_EVIDENCE_FULL
};
static void graph_cpu_options_default(yvex_attention_cpu_options *options)
{
    if (!options) return;
    *options = cpu_options_template;
}
static int graph_recipe_identity(const void *context, char output[65])
{
    return yvex_model_register_deepseek_v4()->transform.architecture_identity(
        (const yvex_deepseek_v4_ir *)context, output);
}
static int graph_execution_admit(
    const yvex_attention_plan *plan, const void *family_ir,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options, const char *cancel_stage,
    const yvex_attention_layer_plan **layer, yvex_attention_failure *failure,
    yvex_error *err)
{
    char derived_identity[65];
    const char *logical_identity = options ? options->logical_model_identity : NULL;
    if (family_ir) {
        if (!graph_recipe_identity(family_ir, derived_identity))
            logical_identity = NULL;
        else if (logical_identity && strcmp(logical_identity, derived_identity) != 0)
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DESCRIPTOR, NULL,
                YVEX_DEEPSEEK_V4_IR_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
                0ull, err, YVEX_ERR_STATE,
                "explicit and architecture-derived logical model identities disagree");
        else
            logical_identity = derived_identity;
    }
    if (!yvex_sha256_hex_valid(logical_identity))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_DEEPSEEK_V4_IR_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention requires an identity-bearing family execution request");
    return yvex_attention_execution_admit(
        plan, logical_identity, session, descriptor, options, cancel_stage,
        DEEPSEEK_ATTENTION_CSA_RATIO, DEEPSEEK_ATTENTION_HCA_RATIO, layer,
        failure, err);
}
static int graph_history_admit(
    const yvex_attention_layer_plan *layer, const yvex_attention_cpu_options *options,
    int initial_state_supported, yvex_attention_history_view *out,
    yvex_attention_failure *failure, yvex_error *err)
{
    int rc;
    memset(out, 0, sizeof(*out));
    if (options->history) *out = *options->history;
    if (out->token_count != options->token_position)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, NULL,
            layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, options->token_position,
            out->token_count, err, YVEX_ERR_STATE,
            "DeepSeek attention history is not contiguous");
    if (!options->history && options->token_position)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, NULL,
            layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, options->token_position,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention requires complete history after the first token");
    if (options->history) {
        rc = yvex_attention_history_validate(layer, out, failure, err);
        if (rc != YVEX_OK) return rc;
    } else if (!initial_state_supported &&
               layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, NULL,
            layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_STATE, "compressed CUDA attention requires explicit rolling history");
    }
    out->immutable = 1;
    out->token_count = options->token_position;
    return YVEX_OK;
}
static int graph_recipe_project(const yvex_deepseek_v4_layer_spec *layer,
                                unsigned long long ordinal, yvex_tensor_scope scope,
                                unsigned long long predictor,
                                yvex_attention_layer_plan *out)
{
    if (!layer || !out) return 0;
    memset(out, 0, sizeof(*out));
    out->ordinal = ordinal;
    out->layer_index = layer->layer_index;
    out->predictor_index = predictor;
    out->tensor_scope = scope;
    out->attention_class = layer->attention_class;
    out->compute_contract = layer->compute_contract;
    out->compression_ratio = layer->compression_ratio;
    out->sliding_window = layer->kv.sliding_window;
    out->query_heads = layer->query_heads;
    out->kv_heads = layer->kv_heads;
    out->head_dimension = layer->head_dimension;
    out->rope_head_dimension = layer->rope_head_dimension;
    out->query_lora_rank = layer->query_lora_rank;
    out->output_lora_rank = layer->output_lora_rank;
    out->output_groups = layer->output_groups;
    out->output_group_input_width = layer->output_group_input_width;
    out->hidden_dimension = layer->tensors.q_a_columns;
    out->indexer_heads = layer->indexer_heads;
    out->indexer_head_dimension = layer->indexer_head_dimension;
    out->indexer_topk = layer->indexer_topk;
    out->compressor_ape_columns = layer->tensors.compressor_ape_columns;
    out->indexer_ape_columns = layer->tensors.indexer_ape_columns;
    out->rms_norm_epsilon = layer->rms_norm_epsilon;
    out->residual_stream_count = layer->mhc.residual_streams;
    out->residual_stream_width = layer->mhc.stream_width;
    out->residual_expanded_width = layer->mhc.expanded_width;
    out->mhc_mixing_rows = layer->mhc.mixing_rows;
    out->mhc_mixing_columns = layer->mhc.mixing_columns;
    out->mhc_base_width = layer->mhc.base_width;
    out->mhc_scale_width = layer->mhc.scale_width;
    out->mhc_sinkhorn_iterations = layer->mhc.sinkhorn_iterations;
    out->mhc_epsilon = layer->mhc.epsilon;
    out->mhc_residual_post_multiplier = layer->mhc.residual_post_multiplier;
    out->mhc_entry_policy = (unsigned int)layer->mhc.entry;
    out->mhc_attention_pre_and_post = layer->mhc.attention_pre_and_post;
    out->attention_input_norm_required = layer->attention_input_norm.required;
    out->attention_input_norm_width = layer->attention_input_norm.width;
    out->attention_input_norm_role = YVEX_TENSOR_ROLE_ATTENTION_NORM;
    out->mhc_function_role = YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION;
    out->mhc_base_role = YVEX_TENSOR_ROLE_HC_ATTENTION_BASE;
    out->mhc_scale_role = YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE;
    out->compressor_required = layer->compressor_required;
    out->indexer_required = layer->indexer_required;
    out->position = layer->position;
    out->attention_kv_activation = layer->attention_kv_activation;
    out->compressor_activation = layer->compressor_activation;
    out->compressor_rotated_activation = layer->compressor_rotated_activation;
    out->indexer_query_activation = layer->indexer_query_activation;
    out->sparse_topk = layer->sparse_topk;
    return 1;
}
static int graph_recipe_layer(const void *context, unsigned long long index,
                              yvex_attention_layer_plan *out)
{
    const yvex_deepseek_v4_layer_spec *layer =
        yvex_model_register_deepseek_v4()->ir.layer_at(
            (const yvex_deepseek_v4_ir *)context, index);
    return graph_recipe_project(layer, index, YVEX_TENSOR_SCOPE_MAIN_LAYER,
                                YVEX_ATTENTION_NO_TENSOR_INDEX, out);
}
static int graph_recipe_draft_layer(const void *context, unsigned long long index,
                                    yvex_attention_layer_plan *out)
{
    const yvex_deepseek_v4_auxiliary_spec *draft =
        yvex_model_register_deepseek_v4()->ir.auxiliary_at(
            (const yvex_deepseek_v4_ir *)context, index);
    return draft && graph_recipe_project(&draft->layer, index, YVEX_TENSOR_SCOPE_DRAFT,
                                         draft->predictor_index, out);
}
static int graph_plan_build(yvex_attention_plan **out, const void *family_ir,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_deepseek_v4_ir *ir = (const yvex_deepseek_v4_ir *)family_ir;
    const yvex_deepseek_v4_model_spec *model = ir
        ? yvex_model_register_deepseek_v4()->ir.model(ir) : NULL;
    yvex_attention_recipe recipe;
    if (!model) return yvex_attention_plan_build(out, NULL, session, descriptor, failure, err);
    recipe = (yvex_attention_recipe){
        .context = ir, .layer_count = yvex_model_register_deepseek_v4()->ir.layer_count(ir),
        .auxiliary_layer_count = model->auxiliary_layer_count,
        .swa_layer_count = model->swa_layer_count, .csa_layer_count = model->csa_layer_count,
        .hca_layer_count = model->hca_layer_count,
        .tensor_scope = YVEX_TENSOR_SCOPE_MAIN_LAYER,
        .identity = graph_recipe_identity, .layer = graph_recipe_layer};
    return yvex_attention_plan_build(out, &recipe, session, descriptor, failure, err);
}
static int graph_draft_plan_build(yvex_attention_plan **out, const void *family_ir,
    const yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_deepseek_v4_ir *ir = (const yvex_deepseek_v4_ir *)family_ir;
    const yvex_deepseek_v4_model_spec *model =
        ir ? yvex_model_register_deepseek_v4()->ir.model(ir) : NULL;
    yvex_attention_recipe recipe;
    if (!model || !model->dspark.present)
        return yvex_attention_plan_build(out, NULL, session, descriptor, failure, err);
    recipe = (yvex_attention_recipe){
        .context = ir, .layer_count = model->dspark.draft_layer_count,
        .auxiliary_layer_count = model->dspark.draft_layer_count,
        .swa_layer_count = model->dspark.draft_layer_count,
        .tensor_scope = YVEX_TENSOR_SCOPE_DRAFT,
        .identity = graph_recipe_identity, .layer = graph_recipe_draft_layer};
    return yvex_attention_plan_build(out, &recipe, session, descriptor, failure, err);
}
static int graph_selection_key_resolve(const char *token,
                                       unsigned long long *selection_key,
                                       yvex_error *err)
{
    unsigned long long key = 0ull;
    if (token && strcmp(token, "swa") == 0)
        key = (unsigned long long)YVEX_ATTENTION_CLASS_SWA + 1ull;
    else if (token && strcmp(token, "csa") == 0)
        key = (unsigned long long)YVEX_ATTENTION_CLASS_CSA + 1ull;
    else if (token && strcmp(token, "hca") == 0)
        key = (unsigned long long)YVEX_ATTENTION_CLASS_HCA + 1ull;
    if (!selection_key || !key) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "graph.deepseek.selection_key",
                       "DeepSeek attention selection requires swa, csa, or hca");
        return YVEX_ERR_INVALID_ARG;
    }
    *selection_key = key;
    yvex_error_clear(err);
    return YVEX_OK;
}
enum { CPU_ROLLING_MAIN = 0, CPU_ROLLING_INDEXER, CPU_ROLLING_COUNT };
typedef struct {
    const yvex_runtime_tensor_binding *weights[4];
    yvex_attention_component_span state[2], emission;
    yvex_attention_rolling_state_view before, current;
    yvex_attention_rolling_state_output after;
    float *initial_state[2], *projected[2], *checkpoints[2], *ape, *norm;
    unsigned long long *positions, emitted;
} cpu_rolling_stage;
typedef struct {
    yvex_attention_rolling_kind kind;
    yvex_tensor_role weight_roles[4];
    yvex_tensor_role output_role, norm_role;
    yvex_attention_component_kind state_components[2], emission_component;
    int activate_complete_output;
} cpu_rolling_recipe;
static const cpu_rolling_recipe cpu_rolling_recipes[CPU_ROLLING_COUNT] = {
    {YVEX_DEEPSEEK_ATTENTION_ROLLING_MAIN,
        {YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE,
         YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM},
        YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM,
        {YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_KV_STATE, YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE},
        YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV,
        0
    },
    {YVEX_DEEPSEEK_ATTENTION_ROLLING_INDEXER,
        {YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE,
         YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM},
        YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM,
        {YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV_STATE,
         YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE},
        YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV,
        1
    }
};
typedef struct {
    const yvex_attention_plan *plan;
    const void *family_ir;
    yvex_materialization_session *session;
    const yvex_runtime_descriptor *descriptor;
    yvex_attention_cpu_options defaults;
    const yvex_attention_cpu_options *opts;
    yvex_attention_cpu_result *result;
    yvex_attention_failure *failure;
    yvex_error *err;
    const yvex_attention_layer_plan *layer_plan;
    const yvex_runtime_tensor_binding *q_a, *q_a_norm, *q_b, *kv, *kv_norm;
    const yvex_runtime_tensor_binding *sinks, *out_a, *out_b;
    const yvex_runtime_tensor_binding *index_q_binding, *index_weight_binding;
    yvex_attention_memory_sink sink;
    yvex_attention_state_transaction transaction;
    yvex_attention_history_view history;
    yvex_attention_component_span output_span, envelope_span, raw_kv_span;
    yvex_attention_envelope_workspace envelope;
    cpu_rolling_stage rolling[CPU_ROLLING_COUNT];
    const yvex_attention_component_span *committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT];
    const char *committed_identity;
    unsigned long long layer_index, token_count, hidden_width, q_rank;
    unsigned long long query_width, kv_width, rows, token;
    unsigned long long hidden_elements, q_low_elements, query_elements, kv_elements;
    unsigned long long scratch_elements, scratch_term;
    yvex_attention_scratch_budget scratch;
    float *hidden, *q_low, *q_norm_weights, *query, *kv_norm_weights;
    float *sink_values, *attention_values;
    float *index_query, *index_weights;
    unsigned long long *trace_topk_counts, *trace_topk_positions;
    unsigned long long trace_topk_stride;
    int rc;
} cpu_chunk_context;
static int cpu_chunk_reject(cpu_chunk_context *context,
                            yvex_attention_failure_code code,
                            const yvex_runtime_tensor_binding *binding, yvex_tensor_role role,
                            unsigned long long expected,
                            unsigned long long actual, yvex_status status,
                            const char *reason)
{
    return yvex_attention_reject(context->failure, code, binding,
                                 context->layer_index, role, expected, actual,
                                 context->err, status, reason);
}
static int cpu_chunk_round(cpu_chunk_context *context, float *values, unsigned long long count, const char *stage)
{
    if (yvex_attention_compute_round(context->layer_plan->compute_contract,
                                     values, count))
        return YVEX_OK;
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, count, 0ull, YVEX_ERR_FORMAT, stage);
}
static int cpu_chunk_scratch_reserve(cpu_chunk_context *context,
                                     unsigned long long count, size_t element_size, const char *reason)
{
    size_t bytes;
    if (!yvex_attention_scratch_reserve(
            &context->scratch, count, element_size, &bytes))
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN,
            context->opts->scratch_limit_bytes,
            (unsigned long long)context->scratch.live_bytes,
            YVEX_ERR_BOUNDS, reason);
    return YVEX_OK;
}
static int cpu_chunk_rolling_active(const cpu_chunk_context *context, unsigned int index)
{
    return index == CPU_ROLLING_MAIN
        ? context->layer_plan->attention_class != YVEX_ATTENTION_CLASS_SWA
        : context->layer_plan->attention_class == YVEX_ATTENTION_CLASS_CSA;
}
static yvex_attention_rolling_state_view *cpu_chunk_history_state(cpu_chunk_context *context, unsigned int index)
{
    return index == CPU_ROLLING_MAIN
        ? &context->history.main_rolling_state
        : &context->history.indexer_rolling_state;
}
static int cpu_chunk_rolling_initialize(cpu_chunk_context *context, unsigned int index)
{
    const cpu_rolling_recipe *recipe = &cpu_rolling_recipes[index];
    cpu_rolling_stage *stage = &context->rolling[index];
    unsigned long long ratio, head_dimension, state_width, state_slots, extent;
    int overlap, rotated;
    int rc;
    if (!cpu_chunk_rolling_active(context, index)) return YVEX_OK;
    if (!yvex_attention_rolling_geometry(
            context->layer_plan, recipe->kind, &ratio, &head_dimension,
            &state_width, &state_slots, &overlap, &rotated) ||
        !yvex_core_u64_mul(state_width, state_slots, &extent) ||
        !yvex_core_u64_mul(extent, 2ull, &extent))
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX, state_slots, YVEX_ERR_BOUNDS,
            "attention rolling scratch geometry overflowed");
    rc = cpu_chunk_scratch_reserve(
        context, extent, sizeof(float),
        "attention rolling scratch budget exceeded");
    if (rc != YVEX_OK) return rc;
    rc = yvex_attention_rolling_storage_acquire(
        context->layer_plan, recipe->kind, context->opts->token_position,
        context->scratch.workspace,
        &stage->initial_state[0], &stage->initial_state[1], &stage->before,
        context->failure, context->err);
    if (rc == YVEX_OK) *cpu_chunk_history_state(context, index) = stage->before;
    return rc;
}
static int cpu_chunk_admit(cpu_chunk_context *context)
{
attention_result_reset(context->result);
if (!context->opts) {
    graph_cpu_options_default(&context->defaults);
    context->opts = &context->defaults;
}
context->token_count = context->opts->token_count ? context->opts->token_count : 1ull;
context->scratch.limit_bytes = context->opts->scratch_limit_bytes;
context->scratch.workspace = context->opts->workspace;
if (!context->plan || !context->session ||
    !context->descriptor || !context->result || !context->opts->input)
    return yvex_attention_reject(
        context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
        YVEX_DEEPSEEK_V4_IR_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
        0ull, context->err, YVEX_ERR_INVALID_ARG,
        "DeepSeek attention CPU chunk requires plan, session, descriptor, explicit input, and result");
context->layer_index = context->opts->layer_index;
context->rc = graph_execution_admit(
    context->plan, context->family_ir, context->session, context->descriptor,
    context->opts,
    "attention CPU execution cancelled before mutation",
    &context->layer_plan, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->q_a = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_A, context->layer_plan);
context->q_a_norm = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM, context->layer_plan);
context->q_b = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_Q_B, context->layer_plan);
context->kv = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_KV, context->layer_plan);
context->kv_norm = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_KV_NORM, context->layer_plan);
context->sinks = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_SINKS, context->layer_plan);
context->out_a = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_OUT_A, context->layer_plan);
context->out_b = yvex_attention_binding_find(
    context->descriptor, YVEX_TENSOR_ROLE_ATTENTION_OUT_B, context->layer_plan);
if (!context->q_a || !context->q_a_norm || !context->q_b || !context->kv ||
    !context->kv_norm || !context->sinks || !context->out_a ||
    !context->out_b)
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, 8ull, 0ull, YVEX_ERR_FORMAT,
        "DeepSeek attention CPU chunk requires complete Q/KV/sink/output bindings");
context->hidden_width = context->q_a->binding->row_width;
context->q_rank = context->q_a->binding->row_count;
context->query_width = context->q_b->binding->row_count;
context->kv_width = context->kv->binding->row_count;
if (context->q_rank != context->q_b->binding->row_width ||
    context->query_width != context->layer_plan->query_heads * context->layer_plan->head_dimension ||
    context->kv_width != context->layer_plan->head_dimension ||
    context->hidden_width != context->layer_plan->hidden_dimension)
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, context->q_b,
        YVEX_TENSOR_ROLE_ATTENTION_Q_B,
        context->layer_plan->query_heads * context->layer_plan->head_dimension,
        context->query_width, YVEX_ERR_FORMAT,
        "DeepSeek attention CPU chunk tensor dimensions do not match plan");
if (!yvex_core_u64_mul(context->token_count, context->hidden_width,
                               &context->hidden_elements) ||
    !yvex_core_u64_mul(context->token_count, context->q_rank, &context->q_low_elements) ||
    !yvex_core_u64_mul(context->token_count, context->query_width,
                               &context->query_elements) ||
    !yvex_core_u64_mul(context->token_count, context->kv_width, &context->kv_elements) ||
    !yvex_core_u64_add(context->hidden_elements, context->q_low_elements,
                               &context->scratch_elements) ||
    !yvex_core_u64_add(context->scratch_elements, context->query_elements,
                               &context->scratch_elements) ||
    !yvex_core_u64_add(context->scratch_elements, context->kv_elements,
                               &context->scratch_elements) ||
    !yvex_core_u64_add(context->scratch_elements, context->query_elements,
                               &context->scratch_elements) ||
    !yvex_core_u64_add(context->q_rank, context->kv_width, &context->scratch_term) ||
    !yvex_core_u64_add(context->scratch_term, context->layer_plan->query_heads,
                               &context->scratch_term) ||
    !yvex_core_u64_add(context->scratch_elements, context->scratch_term,
                      &context->scratch_elements))
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, context->opts->scratch_limit_bytes,
        ULLONG_MAX, YVEX_ERR_BOUNDS,
        "DeepSeek attention CPU chunk scratch geometry overflowed");
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE) {
    unsigned long long envelope_elements;
    if (!yvex_attention_envelope_scratch_elements(
            context->layer_plan, context->token_count, &envelope_elements) ||
        !yvex_core_u64_add(context->scratch_elements, envelope_elements,
                           &context->scratch_elements))
        return cpu_chunk_reject(context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
            NULL, YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX, 0ull, YVEX_ERR_BOUNDS,
            "attention envelope scratch geometry overflowed");
}
context->rc = cpu_chunk_scratch_reserve(
    context, context->scratch_elements, sizeof(float),
    "DeepSeek attention CPU chunk scratch budget exceeded");
if (context->rc != YVEX_OK) return context->rc;
context->rc = graph_history_admit(
    context->layer_plan, context->opts, 1, &context->history,
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
if (context->opts->history) {
    context->rolling[CPU_ROLLING_MAIN].before =
        context->history.main_rolling_state;
    context->rolling[CPU_ROLLING_INDEXER].before =
        context->history.indexer_rolling_state;
}
if (!context->opts->history) {
    unsigned int rolling_index;
    for (rolling_index = 0u; rolling_index < CPU_ROLLING_COUNT;
         ++rolling_index) {
        context->rc = cpu_chunk_rolling_initialize(context, rolling_index);
        if (context->rc != YVEX_OK) return context->rc;
    }
}
{
    yvex_attention_memory_sink_options sink_options = {
        .fail_acquire_kind = YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT,
        .fail_seal_kind = YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT,
        .workspace = context->scratch.workspace,
    };
    context->rc = yvex_attention_memory_sink_init(
        &context->sink, &sink_options, context->failure, context->err);
}
if (context->rc != YVEX_OK) return context->rc;
if (!yvex_attention_transaction_scratch_elements(
        context->layer_plan, &context->history, context->opts->operation_scope,
        context->opts->token_position, context->token_count,
        &context->scratch_elements))
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX, 0ull, YVEX_ERR_BOUNDS,
        "attention transaction staging geometry overflowed");
context->rc = cpu_chunk_scratch_reserve(
    context, context->scratch_elements, sizeof(float),
    "attention transaction staging exceeds the scratch budget");
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_state_transaction_begin_scope(
    &context->sink, context->layer_plan, &context->history,
    context->opts->operation_scope, context->opts->token_position,
    context->token_count, &context->transaction, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_state_transaction_acquire(
    &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT,
    &context->output_span, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_state_transaction_acquire(
    &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV,
    &context->raw_kv_span, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE) {
    context->rc = yvex_attention_state_transaction_acquire(
        &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT,
        &context->envelope_span, context->failure, context->err);
    if (context->rc != YVEX_OK) return context->rc;
}
    return YVEX_OK;
}
static int cpu_chunk_input_prepare(cpu_chunk_context *context)
{
context->hidden = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->hidden_elements, sizeof(float));
context->q_low = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->q_low_elements, sizeof(float));
context->q_norm_weights = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->q_rank, sizeof(float));
context->query = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->query_elements, sizeof(float));
context->kv_norm_weights = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->kv_width, sizeof(float));
context->sink_values = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->layer_plan->query_heads, sizeof(float));
context->attention_values = (float *)yvex_attention_scratch_calloc(
    &context->scratch, context->query_elements, sizeof(float));
if (!context->hidden || !context->q_low || !context->q_norm_weights || !context->query || !context->kv_norm_weights ||
    !context->sink_values || !context->attention_values) {
    context->rc = cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, context->scratch.live_bytes, 0ull,
        YVEX_ERR_NOMEM, "failed to allocate DeepSeek CPU chunk scratch");
    return context->rc;
}
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE)
    return yvex_attention_envelope_prepare(
        context->session, context->descriptor, context->layer_plan,
        context->opts->input, context->token_count, context->opts->input_stride,
        context->hidden, context->hidden_width, &context->envelope,
        &context->scratch, context->result, context->failure, context->err);
for (context->token = 0ull; context->token < context->token_count; ++context->token) {
    unsigned long long lane;
    const float *source = context->opts->input +
        context->token * context->opts->input_stride;
    for (lane = 0ull; lane < context->hidden_width; ++lane) {
        if (!isfinite(source[lane])) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_TENSOR_ROLE_UNKNOWN, context->hidden_width, lane, YVEX_ERR_FORMAT,
                "DeepSeek attention input contains non-finite values");
            return context->rc;
        }
    }
    memcpy(context->hidden + context->token * context->hidden_width, source,
           (size_t)context->hidden_width * sizeof(*context->hidden));
}
context->rc = cpu_chunk_round(
    context, context->hidden, context->hidden_elements,
    "DeepSeek attention input could not enter the BF16 compute contract");
return context->rc;
}
static int cpu_chunk_project(cpu_chunk_context *context)
{
context->rc = yvex_attention_cancel_check(
    context->opts->cancellation, context->layer_index,
    "attention CPU execution cancelled before payload access",
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = cpu_chunk_input_prepare(context);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_dot_batch(
    context->session, context->q_a, 0ull, context->hidden,
    context->token_count, context->hidden_width, context->hidden_width,
    context->q_rank, context->q_low, context->q_rank, &context->rows,
    &context->scratch, context->result, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = cpu_chunk_round(
    context, context->q_low, context->q_low_elements,
    "DeepSeek attention Q-A projection could not publish BF16 values");
if (context->rc != YVEX_OK) return context->rc;
if (context->rows != context->q_rank) {
    context->rc = cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, context->q_a,
        YVEX_TENSOR_ROLE_ATTENTION_Q_A, context->q_rank, context->rows, YVEX_ERR_FORMAT,
        "DeepSeek attention CPU chunk Q-A projection is incomplete");
    return context->rc;
}
context->rc = yvex_attention_decode_row(
    context->session, context->q_a_norm, 0ull, context->q_norm_weights,
    context->q_rank, &context->scratch, context->result, context->failure,
    context->err);
if (context->rc != YVEX_OK) return context->rc;
for (context->token = 0ull; context->token < context->token_count; ++context->token) {
    if (!yvex_attention_rms_norm(context->q_low + context->token * context->q_rank, context->q_rank,
                                  context->q_norm_weights,
                                  context->layer_plan->rms_norm_epsilon)) {
        context->rc = cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, context->q_a_norm,
            YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM, context->q_rank, context->token,
            YVEX_ERR_FORMAT,
            "DeepSeek attention CPU chunk Q norm produced invalid values");
        return context->rc;
    }
    context->rc = cpu_chunk_round(
        context, context->q_low + context->token * context->q_rank,
        context->q_rank,
        "DeepSeek attention Q norm could not publish BF16 values");
    if (context->rc != YVEX_OK) return context->rc;
}
context->rc = yvex_attention_dot_batch(
    context->session, context->q_b, 0ull, context->q_low,
    context->token_count, context->q_rank, context->q_rank,
    context->query_width,
    context->query, context->query_width, &context->rows,
    &context->scratch, context->result, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = cpu_chunk_round(
    context, context->query, context->query_elements,
    "DeepSeek attention Q-B projection could not publish BF16 values");
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_dot_batch(
    context->session, context->kv, 0ull, context->hidden,
    context->token_count, context->hidden_width, context->hidden_width,
    context->kv_width, (float *)context->raw_kv_span.data,
    context->raw_kv_span.stride, &context->rows,
    &context->scratch, context->result, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = cpu_chunk_round(
    context, (float *)context->raw_kv_span.data,
    context->raw_kv_span.expected_elements,
    "DeepSeek attention KV projection could not publish BF16 values");
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_decode_row(
    context->session, context->kv_norm, 0ull, context->kv_norm_weights,
    context->kv_width, &context->scratch, context->result, context->failure,
    context->err);
if (context->rc != YVEX_OK) return context->rc;
for (context->token = 0ull; context->token < context->token_count; ++context->token) {
    unsigned long long head;
    for (head = 0ull; head < context->layer_plan->query_heads; ++head) {
        if (!yvex_attention_unit_rms_norm(
                context->query + context->token * context->query_width +
                    head * context->layer_plan->head_dimension,
                context->layer_plan->head_dimension, context->layer_plan->rms_norm_epsilon)) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, context->q_b,
                YVEX_TENSOR_ROLE_ATTENTION_Q_B, context->layer_plan->head_dimension,
                head, YVEX_ERR_FORMAT,
                "DeepSeek attention query head RMS norm failed");
            return context->rc;
        }
        context->rc = cpu_chunk_round(
            context,
            context->query + context->token * context->query_width +
                head * context->layer_plan->head_dimension,
            context->layer_plan->head_dimension,
            "DeepSeek attention query normalization could not publish BF16 values");
        if (context->rc != YVEX_OK) return context->rc;
        if (!yvex_attention_rope_apply(
                context->query + context->token * context->query_width +
                    head * context->layer_plan->head_dimension,
                context->layer_plan->head_dimension, context->layer_plan->rope_head_dimension,
                context->opts->token_position + context->token, &context->layer_plan->position, 0)) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, context->q_b,
                YVEX_TENSOR_ROLE_ATTENTION_Q_B, 1ull, head, YVEX_ERR_FORMAT,
                "DeepSeek attention query RoPE/YaRN application failed");
            return context->rc;
        }
        context->rc = cpu_chunk_round(
            context,
            context->query + context->token * context->query_width +
                head * context->layer_plan->head_dimension,
            context->layer_plan->head_dimension,
            "DeepSeek attention query RoPE could not publish BF16 values");
        if (context->rc != YVEX_OK) return context->rc;
    }
    if (!yvex_attention_rms_norm(
            (float *)context->raw_kv_span.data + context->token * context->raw_kv_span.stride,
            context->kv_width, context->kv_norm_weights, context->layer_plan->rms_norm_epsilon)) {
        context->rc = cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, context->kv_norm,
            YVEX_TENSOR_ROLE_ATTENTION_KV_NORM, context->kv_width, context->token,
            YVEX_ERR_FORMAT,
            "DeepSeek attention CPU chunk KV norm produced invalid values");
        return context->rc;
    }
    context->rc = cpu_chunk_round(
        context,
        (float *)context->raw_kv_span.data +
            context->token * context->raw_kv_span.stride,
        context->kv_width,
        "DeepSeek attention KV normalization could not publish BF16 values");
    if (context->rc != YVEX_OK) return context->rc;
    if (!yvex_attention_rope_apply(
            (float *)context->raw_kv_span.data + context->token * context->raw_kv_span.stride,
            context->kv_width, context->layer_plan->rope_head_dimension,
            context->opts->token_position + context->token, &context->layer_plan->position, 0)) {
        context->rc = cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, context->kv,
            YVEX_TENSOR_ROLE_ATTENTION_KV, 1ull, context->token, YVEX_ERR_FORMAT,
            "DeepSeek attention KV RoPE/YaRN application failed");
        return context->rc;
    }
    context->rc = cpu_chunk_round(
        context,
        (float *)context->raw_kv_span.data +
            context->token * context->raw_kv_span.stride,
        context->kv_width,
        "DeepSeek attention KV RoPE could not publish BF16 values");
    if (context->rc != YVEX_OK) return context->rc;
    if (context->kv_width > context->layer_plan->rope_head_dimension) {
        context->rc = yvex_attention_activation_apply(
            &context->layer_plan->attention_kv_activation,
            (float *)context->raw_kv_span.data + context->token * context->raw_kv_span.stride,
            context->kv_width - context->layer_plan->rope_head_dimension, context->layer_index,
            YVEX_TENSOR_ROLE_ATTENTION_KV, &context->scratch,
            context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
    }
}
context->rc = yvex_attention_decode_flat(
    context->session, context->sinks, context->sink_values,
    context->layer_plan->query_heads, &context->scratch, context->result,
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
    return YVEX_OK;
}
static int cpu_chunk_emission_prepare(cpu_chunk_context *context, unsigned int index)
{
    const cpu_rolling_recipe *recipe = &cpu_rolling_recipes[index];
    cpu_rolling_stage *stage = &context->rolling[index];
    if (!context->transaction.components[recipe->emission_component].required)
        return YVEX_OK;
    context->rc = yvex_attention_state_transaction_acquire(
        &context->transaction, recipe->emission_component, &stage->emission,
        context->failure, context->err);
    if (context->rc != YVEX_OK) return context->rc;
    context->rc = cpu_chunk_scratch_reserve(
        context, stage->emission.dims[0], sizeof(*stage->positions),
        index == CPU_ROLLING_MAIN
            ? "attention compressed-position scratch exceeds its budget"
            : "attention indexer-position scratch exceeds its budget");
    if (context->rc != YVEX_OK) return context->rc;
    stage->positions = yvex_attention_scratch_calloc(
        &context->scratch, stage->emission.dims[0], sizeof(*stage->positions));
    if (stage->positions) return YVEX_OK;
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
        recipe->output_role, stage->emission.dims[0], 0ull, YVEX_ERR_NOMEM,
        index == CPU_ROLLING_MAIN
            ? "DeepSeek attention compressed-position allocation failed"
            : "DeepSeek attention indexer-position allocation failed");
}
static int cpu_chunk_rolling_prepare(cpu_chunk_context *context, unsigned int index)
{
    const cpu_rolling_recipe *recipe = &cpu_rolling_recipes[index];
    cpu_rolling_stage *stage = &context->rolling[index];
    unsigned int item;
    if (!cpu_chunk_rolling_active(context, index)) return YVEX_OK;
    for (item = 0u; item < 4u; ++item) {
        stage->weights[item] = yvex_attention_binding_find(
            context->descriptor, recipe->weight_roles[item],
            context->layer_plan);
        if (!stage->weights[item])
            return cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING,
                NULL, recipe->output_role, 4ull, 0ull, YVEX_ERR_FORMAT,
                "attention rolling stage requires all encoded bindings");
    }
    for (item = 0u; item < 2u; ++item) {
        context->rc = yvex_attention_state_transaction_acquire(
            &context->transaction, recipe->state_components[item],
            &stage->state[item], context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
    }
    if (context->opts->retain_prefix_checkpoints &&
        !yvex_attention_candidate_checkpoints_open(
            &context->scratch, context->token_count, &stage->before,
            &stage->checkpoints[0], &stage->checkpoints[1]))
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, context->opts->scratch_limit_bytes,
            context->scratch.live_bytes, YVEX_ERR_BOUNDS,
            "attention rolling checkpoints exceed their budget");
    if (index == CPU_ROLLING_INDEXER) {
        context->rc = cpu_chunk_emission_prepare(context, index);
        if (context->rc != YVEX_OK) return context->rc;
    }
    if (!yvex_core_u64_mul(context->token_count, stage->before.state_width,
                           &context->scratch_term))
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX, context->token_count,
            YVEX_ERR_BOUNDS, "attention rolling scratch geometry overflowed");
    context->rc = cpu_chunk_scratch_reserve(
        context, context->scratch_term, 2u * sizeof(float),
        "attention rolling token scratch exceeds its budget");
    if (context->rc == YVEX_OK)
        context->rc = cpu_chunk_scratch_reserve(
            context, stage->before.state_width, sizeof(float),
            "attention rolling APE scratch exceeds its budget");
    if (context->rc == YVEX_OK)
        context->rc = cpu_chunk_scratch_reserve(
            context, stage->before.head_dimension, sizeof(float),
            "attention rolling norm scratch exceeds its budget");
    if (context->rc != YVEX_OK) return context->rc;
    stage->projected[0] = yvex_attention_scratch_calloc(
        &context->scratch, context->scratch_term, sizeof(float));
    stage->projected[1] = yvex_attention_scratch_calloc(
        &context->scratch, context->scratch_term, sizeof(float));
    stage->ape = yvex_attention_scratch_calloc(
        &context->scratch, stage->before.state_width, sizeof(float));
    stage->norm = yvex_attention_scratch_calloc(
        &context->scratch, stage->before.head_dimension, sizeof(float));
    if (!stage->projected[0] || !stage->projected[1] || !stage->ape ||
        !stage->norm)
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, context->scratch_term, 0ull,
            YVEX_ERR_NOMEM, "attention rolling scratch allocation failed");
    for (item = 0u; item < 2u; ++item) {
        context->rc = yvex_attention_dot_batch(
            context->session, stage->weights[item], 0ull, context->hidden,
            context->token_count, context->hidden_width,
            context->hidden_width, stage->before.state_width,
            stage->projected[item], stage->before.state_width, &context->rows,
            &context->scratch, context->result, context->failure,
            context->err);
        if (context->rc != YVEX_OK) return context->rc;
    }
    context->rc = yvex_attention_decode_row(
        context->session, stage->weights[3], 0ull, stage->norm,
        stage->before.head_dimension, &context->scratch, context->result,
        context->failure, context->err);
    if (context->rc != YVEX_OK || index == CPU_ROLLING_INDEXER)
        return context->rc;
    return cpu_chunk_emission_prepare(context, index);
}
static int cpu_chunk_rolling_step(cpu_chunk_context *context, unsigned int index)
{
    const cpu_rolling_recipe *recipe = &cpu_rolling_recipes[index];
    cpu_rolling_stage *stage = &context->rolling[index];
    const yvex_attention_activation_policy *activation =
        index == CPU_ROLLING_MAIN
            ? &context->layer_plan->compressor_activation
            : &context->layer_plan->compressor_rotated_activation;
    unsigned long long width = stage->before.head_dimension;
    unsigned int item;
    if (!cpu_chunk_rolling_active(context, index)) return YVEX_OK;
    stage->current = stage->before;
    for (context->token = 0ull; context->token < context->token_count;
         ++context->token) {
        unsigned long long position;
        int emitted = 0;
        float *output = stage->emission.data
            ? (float *)stage->emission.data +
                  stage->emitted * stage->emission.stride
            : stage->projected[0] +
                  context->token * stage->before.state_width;
        context->rc = yvex_attention_decode_row(
            context->session, stage->weights[2],
            (context->opts->token_position + context->token) %
                context->layer_plan->compression_ratio,
            stage->ape, stage->before.state_width, &context->scratch,
            context->result, context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
        memset(&stage->after, 0, sizeof(stage->after));
        stage->after.kv_state = (float *)stage->state[0].data;
        stage->after.score_state = (float *)stage->state[1].data;
        stage->after.kv_state_stride = stage->before.kv_state_stride;
        stage->after.score_state_stride = stage->before.score_state_stride;
        stage->after.kv_state_extent = stage->before.kv_state_extent;
        stage->after.score_state_extent = stage->before.score_state_extent;
        context->rc = yvex_attention_rolling_state_step_cpu(
            context->layer_plan, &stage->current,
            stage->projected[0] +
                context->token * stage->before.state_width,
            stage->projected[1] +
                context->token * stage->before.state_width,
            stage->ape, &stage->after, output, width, &emitted,
            context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
        if (context->opts->retain_prefix_checkpoints)
            yvex_attention_candidate_checkpoint_capture(
                stage->checkpoints[0], stage->checkpoints[1], context->token,
                &stage->after);
        if (emitted) {
            position = context->opts->token_position + context->token + 1ull -
                       context->layer_plan->compression_ratio;
            context->rc = cpu_chunk_round(
                context, output, width,
                "attention rolling output could not enter BF16 before normalization");
            if (context->rc != YVEX_OK) return context->rc;
            if (!yvex_attention_rms_norm(
                    output, width, stage->norm,
                    context->layer_plan->rms_norm_epsilon))
                return cpu_chunk_reject(
                    context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                    stage->weights[3], recipe->norm_role, width,
                    stage->emitted, YVEX_ERR_FORMAT,
                    "attention rolling normalization geometry is invalid");
            context->rc = cpu_chunk_round(
                context, output, width,
                "attention rolling norm could not publish BF16 values");
            if (context->rc != YVEX_OK) return context->rc;
            if (!yvex_attention_rope_apply(
                    output, width, context->layer_plan->rope_head_dimension,
                    position, &context->layer_plan->position, 0))
                return cpu_chunk_reject(
                    context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                    recipe->output_role, 1ull, context->token,
                    YVEX_ERR_FORMAT, "attention rolling RoPE/YaRN failed");
            context->rc = cpu_chunk_round(
                context, output, width,
                "attention rolling RoPE could not publish BF16 values");
            if (context->rc != YVEX_OK) return context->rc;
            if (recipe->activate_complete_output ||
                width > context->layer_plan->rope_head_dimension) {
                unsigned long long active = recipe->activate_complete_output
                    ? width : width - context->layer_plan->rope_head_dimension;
                context->rc = yvex_attention_activation_apply(
                    activation, output, active, context->layer_index,
                    recipe->output_role, &context->scratch, context->failure,
                    context->err);
                if (context->rc != YVEX_OK) return context->rc;
            }
            if (!stage->positions ||
                stage->emitted >= stage->emission.dims[0])
                return cpu_chunk_reject(
                    context, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
                    NULL, recipe->output_role, stage->emission.dims[0],
                    stage->emitted, YVEX_ERR_BOUNDS,
                    "attention rolling stage emitted beyond planned positions");
            stage->positions[stage->emitted++] = position;
        }
        yvex_attention_rolling_output_bind(&stage->after, &stage->current);
    }
    if (stage->emission.data) {
        context->rc = yvex_attention_state_transaction_seal(
            &context->transaction, recipe->emission_component,
            stage->emission.expected_elements, context->failure,
            context->err);
        if (context->rc != YVEX_OK) return context->rc;
    }
    for (item = 0u; item < 2u; ++item) {
        context->rc = yvex_attention_state_transaction_seal(
            &context->transaction, recipe->state_components[item],
            stage->state[item].expected_elements, context->failure,
            context->err);
        if (context->rc != YVEX_OK) return context->rc;
    }
    return YVEX_OK;
}
static int cpu_chunk_index_query(cpu_chunk_context *context)
{
    if (context->layer_plan->attention_class != YVEX_ATTENTION_CLASS_CSA)
        return YVEX_OK;
    context->index_q_binding = yvex_attention_binding_find(
            context->descriptor, YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
            context->layer_plan);
        context->index_weight_binding = yvex_attention_binding_find(
            context->descriptor, YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
            context->layer_plan);
        if (!context->index_q_binding || !context->index_weight_binding ||
            context->index_q_binding->binding->row_width != context->q_rank ||
            context->index_q_binding->binding->row_count !=
                context->layer_plan->indexer_heads *
                    context->layer_plan->indexer_head_dimension ||
            context->index_weight_binding->binding->row_width != context->hidden_width ||
            context->index_weight_binding->binding->row_count !=
                context->layer_plan->indexer_heads) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING,
                context->index_q_binding,
                YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
                context->layer_plan->indexer_heads *
                    context->layer_plan->indexer_head_dimension,
                context->index_q_binding ? context->index_q_binding->binding->row_count
                                : 0ull,
                YVEX_ERR_FORMAT,
                "DeepSeek CSA index query/weight bindings do not match the plan");
            return context->rc;
        }
        if (!yvex_core_u64_mul(
                context->token_count,
                context->layer_plan->indexer_heads *
                    context->layer_plan->indexer_head_dimension,
                &context->scratch_term)) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                context->index_q_binding,
                YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B, ULLONG_MAX,
                context->token_count, YVEX_ERR_BOUNDS,
                "DeepSeek CSA index query extent overflowed");
            return context->rc;
        }
        context->rc = cpu_chunk_scratch_reserve(
            context, context->scratch_term, sizeof(float),
            "attention index-query scratch exceeds its budget");
        if (context->rc != YVEX_OK) return context->rc;
        context->index_query = (float *)yvex_attention_scratch_calloc(
            &context->scratch, context->scratch_term,
            sizeof(*context->index_query));
        if (!yvex_core_u64_mul(context->token_count,
                               context->layer_plan->indexer_heads,
                               &context->scratch_term))
            return cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
                context->index_weight_binding,
                YVEX_TENSOR_ROLE_INDEXER_PROJECTION, ULLONG_MAX,
                context->token_count, YVEX_ERR_BOUNDS,
                "attention index-weight scratch geometry overflowed");
        context->rc = cpu_chunk_scratch_reserve(
            context, context->scratch_term, sizeof(float),
            "attention index-weight scratch exceeds its budget");
        if (context->rc != YVEX_OK) return context->rc;
        context->index_weights = (float *)yvex_attention_scratch_calloc(
            &context->scratch, context->scratch_term,
            sizeof(*context->index_weights));
        if (!context->index_query || !context->index_weights) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION,
                context->index_q_binding,
                YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B, context->scratch_term,
                0ull, YVEX_ERR_NOMEM,
                "DeepSeek CSA index query/weight allocation failed");
            return context->rc;
        }
        context->rc = yvex_attention_dot_batch(
            context->session, context->index_q_binding, 0ull, context->q_low, context->token_count, context->q_rank,
            context->q_rank,
            context->layer_plan->indexer_heads *
                context->layer_plan->indexer_head_dimension,
            context->index_query,
            context->layer_plan->indexer_heads *
                context->layer_plan->indexer_head_dimension,
            &context->rows, &context->scratch, context->result,
            context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
        if (context->rows != context->layer_plan->indexer_heads *
                        context->layer_plan->indexer_head_dimension) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                context->index_q_binding,
                YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
                context->layer_plan->indexer_heads *
                    context->layer_plan->indexer_head_dimension,
                context->rows, YVEX_ERR_FORMAT,
                "DeepSeek CSA index query projection is incomplete");
            return context->rc;
        }
        context->rc = cpu_chunk_round(
            context, context->index_query,
            context->token_count * context->layer_plan->indexer_heads *
                context->layer_plan->indexer_head_dimension,
            "DeepSeek index query projection could not publish BF16 values");
        if (context->rc != YVEX_OK) return context->rc;
        context->rc = yvex_attention_dot_batch(
            context->session, context->index_weight_binding, 0ull, context->hidden, context->token_count,
            context->hidden_width, context->hidden_width, context->layer_plan->indexer_heads,
            context->index_weights, context->layer_plan->indexer_heads,
            &context->rows, &context->scratch, context->result,
            context->failure, context->err);
        if (context->rc != YVEX_OK) return context->rc;
        if (context->rows != context->layer_plan->indexer_heads) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                context->index_weight_binding,
                YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
                context->layer_plan->indexer_heads, context->rows, YVEX_ERR_FORMAT,
                "DeepSeek CSA index weight projection is incomplete");
            return context->rc;
        }
        context->rc = cpu_chunk_round(
            context, context->index_weights,
            context->token_count * context->layer_plan->indexer_heads,
            "DeepSeek index weight projection could not publish BF16 values");
        if (context->rc != YVEX_OK) return context->rc;
        for (context->token = 0ull; context->token < context->token_count; ++context->token) {
            unsigned long long head;
            for (head = 0ull; head < context->layer_plan->indexer_heads; ++head) {
                float *head_query = context->index_query +
                    context->token * context->layer_plan->indexer_heads *
                        context->layer_plan->indexer_head_dimension +
                    head * context->layer_plan->indexer_head_dimension;
                if (!yvex_attention_rope_apply(
                        head_query,
                        context->layer_plan->indexer_head_dimension,
                        context->layer_plan->rope_head_dimension,
                        context->opts->token_position + context->token,
                        &context->layer_plan->position, 0)) {
                    context->rc = cpu_chunk_reject(
                        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                        context->index_q_binding,
                        YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B, 1ull,
                        head, YVEX_ERR_FORMAT,
                        "DeepSeek CSA index query RoPE/YaRN failed");
                    return context->rc;
                }
                context->rc = cpu_chunk_round(
                    context, head_query,
                    context->layer_plan->indexer_head_dimension,
                    "DeepSeek index query RoPE could not publish BF16 values");
                if (context->rc != YVEX_OK) return context->rc;
                context->rc = yvex_attention_activation_apply(
                    &context->layer_plan->indexer_query_activation, head_query,
                    context->layer_plan->indexer_head_dimension, context->layer_index,
                    YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B,
                    &context->scratch, context->failure, context->err);
                if (context->rc != YVEX_OK) return context->rc;
            }
        }
    return YVEX_OK;
}
static int cpu_chunk_trace_prepare(cpu_chunk_context *context)
{
if ((context->opts->publication || context->opts->trace) &&
    context->layer_plan->attention_class == YVEX_ATTENTION_CLASS_CSA) {
    unsigned long long compressed_total;
    unsigned long long topk_extent;
    if (!yvex_core_u64_add(context->history.compressed_entry_count,
                                   context->rolling[CPU_ROLLING_MAIN].emitted,
                                   &compressed_total) ||
        !yvex_core_u64_mul(
            context->token_count,
            attention_min_u64(compressed_total,
                               context->layer_plan->sparse_topk.k),
            &topk_extent)) {
        context->rc = cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX,
            context->rolling[CPU_ROLLING_MAIN].emitted,
            YVEX_ERR_BOUNDS,
            "DeepSeek CSA trace top-k extent overflowed");
        return context->rc;
    }
    context->trace_topk_stride = attention_min_u64(
        compressed_total, context->layer_plan->sparse_topk.k);
    if (context->trace_topk_stride) {
        context->rc = cpu_chunk_scratch_reserve(
            context, context->token_count,
            sizeof(*context->trace_topk_counts),
            "attention top-k count scratch exceeds its budget");
        if (context->rc == YVEX_OK)
            context->rc = cpu_chunk_scratch_reserve(
                context, topk_extent,
                sizeof(*context->trace_topk_positions),
                "attention top-k position scratch exceeds its budget");
        if (context->rc != YVEX_OK) return context->rc;
        context->trace_topk_counts =
            (unsigned long long *)yvex_attention_scratch_calloc(
                &context->scratch, context->token_count,
                sizeof(*context->trace_topk_counts));
        context->trace_topk_positions =
            (unsigned long long *)yvex_attention_scratch_calloc(
                &context->scratch, topk_extent,
                sizeof(*context->trace_topk_positions));
        if (!context->trace_topk_counts || !context->trace_topk_positions) {
            context->rc = cpu_chunk_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
                YVEX_TENSOR_ROLE_UNKNOWN, topk_extent, 0ull, YVEX_ERR_NOMEM,
                "DeepSeek CSA trace top-k allocation failed");
            return context->rc;
        }
    }
}
    return YVEX_OK;
}
static int cpu_chunk_reduce_commit(cpu_chunk_context *context)
{
context->rc = yvex_attention_reduce_chunk(
    context->layer_plan, context->query, &context->history,
    (const float *)context->raw_kv_span.data, context->raw_kv_span.stride,
    context->rolling[CPU_ROLLING_MAIN].emission.data
        ? (const float *)context->rolling[CPU_ROLLING_MAIN].emission.data : NULL,
    context->rolling[CPU_ROLLING_MAIN].emitted,
    context->rolling[CPU_ROLLING_MAIN].emission.stride,
    context->rolling[CPU_ROLLING_MAIN].positions,
    context->rolling[CPU_ROLLING_INDEXER].emission.data
        ? (const float *)context->rolling[CPU_ROLLING_INDEXER].emission.data : NULL,
    context->rolling[CPU_ROLLING_INDEXER].emitted,
    context->rolling[CPU_ROLLING_INDEXER].emission.stride,
    context->rolling[CPU_ROLLING_INDEXER].positions,
    context->index_query,
    context->layer_plan->indexer_heads * context->layer_plan->indexer_head_dimension,
    context->index_weights, context->layer_plan->indexer_heads, context->sink_values, context->token_count,
    context->opts->token_position, context->attention_values,
    context->opts->candidate_block_visible, context->trace_topk_counts,
    context->trace_topk_positions, context->trace_topk_stride,
    &context->scratch, context->result, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_output_project(
    context->session, context->out_a, context->out_b,
    context->attention_values, context->token_count, context->query_width,
    context->layer_plan->output_groups, context->layer_plan->output_group_input_width,
    context->layer_plan->output_lora_rank, context->hidden_width,
    context->layer_plan->compute_contract, (float *)context->output_span.data,
    context->output_span.stride, &context->scratch, context->result,
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE) {
    context->rc = yvex_attention_envelope_finish(
        context->layer_plan, (const float *)context->output_span.data,
        context->output_span.stride, context->token_count, &context->envelope,
        (float *)context->envelope_span.data, context->envelope_span.stride,
        context->failure, context->err);
    if (context->rc != YVEX_OK) return context->rc;
}
context->rc = yvex_attention_cancel_check(
    context->opts->cancellation, context->layer_index,
    "attention CPU execution cancelled before state commit",
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_state_transaction_seal(
    &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT,
    context->output_span.expected_elements, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE) {
    context->rc = yvex_attention_state_transaction_seal(
        &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT,
        context->envelope_span.expected_elements, context->failure, context->err);
    if (context->rc != YVEX_OK) return context->rc;
}
context->rc = yvex_attention_state_transaction_seal(
    &context->transaction, YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV,
    context->raw_kv_span.expected_elements, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = yvex_attention_state_transaction_commit(
    &context->transaction, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
    return YVEX_OK;
}
static int cpu_chunk_capture(cpu_chunk_context *context)
{
const yvex_attention_component_span *const *committed = context->committed;
unsigned int component;
for (component = 0u; component < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT;
     ++component)
    context->committed[component] =
        yvex_attention_memory_sink_committed_component(
            &context->sink, (yvex_attention_component_kind)component);
context->committed_identity = yvex_attention_memory_sink_identity(&context->sink);
if (!committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT] ||
    !committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT]->data ||
    !committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV] ||
    !committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV]->data ||
    !context->committed_identity) {
    context->rc = cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, YVEX_ERR_STATE,
        "DeepSeek attention CPU chunk commit did not publish output identity");
    return context->rc;
}
if (context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE &&
    (!committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT] ||
     !committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT]->data))
    return cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, YVEX_ERR_STATE,
        "attention envelope commit did not publish expanded output");
if ((context->opts->publication || context->opts->trace) &&
    !yvex_attention_trace_capture(
        context->opts->publication ? context->opts->publication
                                   : context->opts->trace,
        context->layer_index, context->layer_plan->attention_class,
        context->opts->token_position, context->token_count, context->hidden_width, context->q_rank,
        context->query_width, context->kv_width, context->hidden, context->q_low, context->query,
        (const float *)committed[
            YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV]->data,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV]->data
            : NULL,
        context->rolling[CPU_ROLLING_MAIN].emitted,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV]
            ? committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV]->stride : 0ull,
        context->rolling[CPU_ROLLING_MAIN].positions,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV]->data : NULL,
        context->rolling[CPU_ROLLING_INDEXER].emitted,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV]
            ? committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV]->stride : 0ull,
        context->rolling[CPU_ROLLING_INDEXER].positions,
        context->index_query,
        context->layer_plan->indexer_heads *
            context->layer_plan->indexer_head_dimension,
        context->index_weights, context->layer_plan->indexer_heads, context->attention_values,
        (const float *)committed[
            YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT]->data,
        context->trace_topk_counts,
        context->trace_topk_positions, context->trace_topk_stride,
        context->rolling[CPU_ROLLING_MAIN].after.present
            ? &context->rolling[CPU_ROLLING_MAIN].after : NULL,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_KV_STATE]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_KV_STATE]->data : NULL,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE]->data : NULL,
        context->rolling[CPU_ROLLING_INDEXER].after.present
            ? &context->rolling[CPU_ROLLING_INDEXER].after : NULL,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV_STATE]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV_STATE]->data : NULL,
        committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE]
            ? (const float *)committed[
                  YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE]->data : NULL,
        context->opts->retain_prefix_checkpoints &&
                context->layer_plan->attention_class != YVEX_ATTENTION_CLASS_SWA
            ? context->token_count : 0ull,
        context->rolling[CPU_ROLLING_MAIN].checkpoints[0],
        context->rolling[CPU_ROLLING_MAIN].checkpoints[1],
        context->rolling[CPU_ROLLING_INDEXER].checkpoints[0],
        context->rolling[CPU_ROLLING_INDEXER].checkpoints[1],
        context->opts->evidence_level, context->opts->workspace)) {
    context->rc = cpu_chunk_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
        YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, YVEX_ERR_NOMEM,
        "DeepSeek attention execution trace capture failed");
    return context->rc;
}
if (context->opts->publication || context->opts->trace) {
    yvex_attention_execution_trace *trace = context->opts->publication
        ? context->opts->publication : context->opts->trace;
    if (!yvex_attention_trace_outputs_attach(
            trace, committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT],
            committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT]))
        return cpu_chunk_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, YVEX_ERR_NOMEM,
            "attention envelope output trace attachment failed");
    trace->prefix_addressable = context->opts->retain_prefix_checkpoints;
}
    return YVEX_OK;
}
static void cpu_chunk_publish(cpu_chunk_context *context)
{
const yvex_attention_component_span *const *committed = context->committed;
context->result->executed = 1;
context->result->full_attention = 1;
context->result->cuda_executed = 0;
context->result->operation_scope = context->opts->operation_scope;
context->result->layer_index = context->layer_index;
context->result->attention_class = context->layer_plan->attention_class;
context->result->token_position = context->opts->token_position;
context->result->q_a_rows = context->q_rank;
context->result->q_b_rows = context->query_width;
context->result->kv_rows = context->kv_width;
context->result->local_entries = context->history.local_tail_count + context->token_count;
context->result->compressed_entries =
    context->rolling[CPU_ROLLING_MAIN].emitted;
context->result->state_raw_entries = context->token_count;
context->result->state_compressed_entries =
    context->rolling[CPU_ROLLING_MAIN].emitted;
context->result->state_indexer_entries =
    context->rolling[CPU_ROLLING_INDEXER].emitted;
context->result->q_projection_checksum = yvex_attention_checksum(context->q_low,
                                                   context->token_count * context->q_rank);
context->result->kv_projection_checksum =
    yvex_attention_checksum(
        (const float *)committed[
            YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV]->data,
        committed[
            YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV]->expected_elements);
context->result->rope_checksum = yvex_attention_checksum(context->query, context->token_count * context->query_width);
context->result->attention_checksum =
    yvex_attention_checksum(context->attention_values, context->token_count * context->query_width);
yvex_attention_result_outputs_publish(
    context->result, context->opts->operation_scope,
    committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT],
    committed[YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT]);
    yvex_core_text_copy(context->result->output_identity,
                        sizeof(context->result->output_identity),
                        context->committed_identity);
yvex_error_clear(context->err);
if (context->failure) memset(context->failure, 0, sizeof(*context->failure));
}
static int graph_cpu_chunk_execute(const yvex_attention_plan *plan, const void *family_ir,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options, yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err)
{
    cpu_chunk_context context;
    unsigned int rolling_index, item;
    int rc;
    memset(&context, 0, sizeof(context));
    context.plan = plan;
    context.family_ir = family_ir;
    context.session = session;
    context.descriptor = descriptor;
    context.opts = options;
    context.result = result;
    context.failure = failure;
    context.err = err;
    rc = cpu_chunk_admit(&context);
    if (rc == YVEX_OK) rc = cpu_chunk_project(&context);
    if (rc == YVEX_OK) rc = cpu_chunk_rolling_prepare(&context, CPU_ROLLING_MAIN);
    if (rc == YVEX_OK) rc = cpu_chunk_rolling_step(&context, CPU_ROLLING_MAIN);
    if (rc == YVEX_OK) rc = cpu_chunk_rolling_prepare(&context, CPU_ROLLING_INDEXER);
    if (rc == YVEX_OK) rc = cpu_chunk_rolling_step(&context, CPU_ROLLING_INDEXER);
    if (rc == YVEX_OK) rc = cpu_chunk_index_query(&context);
    if (rc == YVEX_OK) rc = cpu_chunk_trace_prepare(&context);
    if (rc == YVEX_OK) rc = cpu_chunk_reduce_commit(&context);
    if (rc == YVEX_OK) rc = cpu_chunk_capture(&context);
    if (rc == YVEX_OK) cpu_chunk_publish(&context);
    if (rc != YVEX_OK &&
        context.transaction.status == YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN) {
        yvex_attention_failure cleanup_failure;
        yvex_error cleanup_error;
        yvex_error_clear(&cleanup_error);
        if (yvex_attention_state_transaction_abort(
                &context.transaction, &cleanup_failure, &cleanup_error) != YVEX_OK)
            rc = yvex_attention_reject(
                context.failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_CLEANUP, NULL,
                context.layer_index, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull,
                context.err, YVEX_ERR_STATE,
                "attention CPU transaction rollback failed");
    }
    yvex_attention_memory_sink_release(&context.sink);
    yvex_attention_envelope_workspace_release(&context.envelope);
    yvex_attention_scratch_free(&context.scratch, context.hidden);
    yvex_attention_scratch_free(&context.scratch, context.q_low);
    yvex_attention_scratch_free(&context.scratch, context.q_norm_weights);
    yvex_attention_scratch_free(&context.scratch, context.query);
    yvex_attention_scratch_free(&context.scratch, context.kv_norm_weights);
    yvex_attention_scratch_free(&context.scratch, context.sink_values);
    yvex_attention_scratch_free(&context.scratch, context.attention_values);
    for (rolling_index = 0u; rolling_index < CPU_ROLLING_COUNT;
         ++rolling_index) {
        for (item = 0u; item < 2u; ++item) {
            yvex_attention_scratch_free(
                &context.scratch, context.rolling[rolling_index].initial_state[item]);
            yvex_attention_scratch_free(
                &context.scratch, context.rolling[rolling_index].projected[item]);
        }
        yvex_attention_scratch_free(&context.scratch,
                                    context.rolling[rolling_index].ape);
        yvex_attention_scratch_free(&context.scratch,
                                    context.rolling[rolling_index].norm);
        yvex_attention_scratch_free(&context.scratch,
                                    context.rolling[rolling_index].positions);
    }
    yvex_attention_scratch_free(&context.scratch, context.index_query);
    yvex_attention_scratch_free(&context.scratch, context.index_weights);
    yvex_attention_scratch_free(&context.scratch, context.trace_topk_counts);
    yvex_attention_scratch_free(&context.scratch, context.trace_topk_positions);
    if (rc != YVEX_OK) attention_result_reset(result);
    return rc;
}
#include <yvex/internal/backend.h>
typedef struct {
    yvex_tensor_role role;
    yvex_backend_attention_weight_slot slot;
    unsigned int class_mask, envelope_only;
} attention_cuda_role;
enum {
    ROLE_CSA = 2u, ROLE_COMPRESSED = 6u, ROLE_ALL = 7u
};
#define R(role_name, slot_name, mask) {role_name, slot_name, mask, 0u}
#define E(role_name, slot_name) {role_name, slot_name, ROLE_ALL, 1u}
static const attention_cuda_role cuda_roles[] = {
    R(YVEX_TENSOR_ROLE_ATTENTION_Q_A, YVEX_BACKEND_ATTENTION_WEIGHT_Q_A, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM, YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_Q_B, YVEX_BACKEND_ATTENTION_WEIGHT_Q_B, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_KV, YVEX_BACKEND_ATTENTION_WEIGHT_KV, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_KV_NORM, YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_SINKS, YVEX_BACKEND_ATTENTION_WEIGHT_SINKS, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_OUT_A, YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_OUT_B, YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B, ROLE_ALL),
    R(YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV, YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_KV, ROLE_COMPRESSED),
    R(YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE, YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_GATE, ROLE_COMPRESSED),
    R(YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE, YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_APE, ROLE_COMPRESSED),
    R(YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM, YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_NORM, ROLE_COMPRESSED),
    R(YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_KV, ROLE_CSA),
    R(YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_GATE, ROLE_CSA),
    R(YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_APE, ROLE_CSA),
    R(YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_NORM, ROLE_CSA),
    R(YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q, ROLE_CSA),
    R(YVEX_TENSOR_ROLE_INDEXER_PROJECTION, YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION, ROLE_CSA),
    E(YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION),
    E(YVEX_TENSOR_ROLE_HC_ATTENTION_BASE, YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE),
    E(YVEX_TENSOR_ROLE_HC_ATTENTION_SCALE, YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE),
    E(YVEX_TENSOR_ROLE_ATTENTION_NORM, YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM),
};
#undef E
#undef R
static const yvex_attention_failure_code cuda_failure_map[] = {
    YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND, YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, YVEX_DEEPSEEK_ATTENTION_FAILURE_CANCELLED,
    YVEX_DEEPSEEK_ATTENTION_FAILURE_CLEANUP
};
static int cuda_token_prepare(attention_cuda_context *context)
{
if (context->result) memset(context->result, 0, sizeof(*context->result));
memset(&context->weights, 0, sizeof(context->weights));
memset(&context->job, 0, sizeof(context->job));
memset(&context->cuda_output, 0, sizeof(context->cuda_output));
memset(&context->trace, 0, sizeof(context->trace));
memset(&context->empty_history, 0, sizeof(context->empty_history));
if (!context->opts) {
    graph_cpu_options_default(&context->defaults);
    context->opts = &context->defaults;
}
if (!context->plan || !context->session ||
    !context->descriptor || !context->backend || !context->result ||
    !context->opts->input || context->opts->input_stride == 0ull)
    return yvex_attention_cuda_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
        1ull, 0ull, YVEX_ERR_INVALID_ARG,
        "CUDA attention requires an explicit input phase and backend");
context->token_count = context->opts->token_count ? context->opts->token_count : 1ull;
context->rc = graph_execution_admit(
    context->plan, context->family_ir, context->session, context->descriptor,
    context->opts,
    "CUDA attention cancelled before graph dispatch",
    &context->layer, context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->rc = graph_history_admit(
    context->layer, context->opts, 0, &context->empty_history,
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
context->history = &context->empty_history;
context->rc = yvex_attention_cuda_trace_open(
    &context->trace, context->layer, context->opts->operation_scope,
    context->history, context->opts->token_position, context->token_count,
    context->opts->evidence_level, context->opts->retain_prefix_checkpoints,
    context->opts->workspace,
    context->opts->scratch_limit_bytes, &context->trace_bytes,
    context->failure, context->err);
if (context->rc != YVEX_OK) return context->rc;
if (context->trace.input) {
    unsigned long long token;
    unsigned long long width = context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
                                   ? context->layer->residual_expanded_width
                                   : context->layer->hidden_dimension;
    for (token = 0ull; token < context->token_count; ++token)
        memcpy(context->trace.input + token * width,
               context->opts->input + token * context->opts->input_stride,
               (size_t)width * sizeof(float));
}
    return YVEX_OK;
}
static int cuda_token_project(attention_cuda_context *context)
{
    yvex_backend_host_workspace_summary workspace;
if (context->layer->compute_contract !=
    YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1)
    return yvex_attention_cuda_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
        YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
        context->layer->compute_contract, YVEX_ERR_UNSUPPORTED,
        "CUDA attention compute contract has no backend projection");
context->job.compute_contract =
    YVEX_BACKEND_ATTENTION_COMPUTE_BF16_F32_RNE_V1;
context->job.schema = YVEX_BACKEND_ATTENTION_JOB_SCHEMA;
context->job.phase = context->token_count == 1ull
    ? YVEX_BACKEND_ATTENTION_PHASE_DECODE
    : context->opts->candidate_block_visible
          ? YVEX_BACKEND_ATTENTION_PHASE_SPECULATIVE_DRAFT
          : YVEX_BACKEND_ATTENTION_PHASE_PREFILL;
context->job.candidate_block_visible = context->opts->candidate_block_visible;
context->job.retain_prefix_checkpoints = context->opts->retain_prefix_checkpoints;
context->job.operation_scope = context->opts->operation_scope ==
        YVEX_ATTENTION_OPERATION_ENVELOPE
    ? YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
    : YVEX_BACKEND_ATTENTION_SCOPE_CORE;
switch (context->layer->attention_class) {
case YVEX_ATTENTION_CLASS_SWA:
    context->job.attention_class = YVEX_BACKEND_ATTENTION_SWA;
    break;
case YVEX_ATTENTION_CLASS_CSA:
    context->job.attention_class = YVEX_BACKEND_ATTENTION_CSA;
    break;
case YVEX_ATTENTION_CLASS_HCA:
    context->job.attention_class = YVEX_BACKEND_ATTENTION_HCA;
    break;
default:
    return yvex_attention_cuda_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, 1ull,
        context->layer->attention_class, YVEX_ERR_FORMAT,
        "CUDA attention class has no backend representation");
}
context->role_mask = 1u << context->layer->attention_class;
context->job.token_position = context->opts->token_position;
context->job.token_count = context->token_count;
context->job.input_stride = context->opts->input_stride;
context->job.hidden_width = context->layer->hidden_dimension;
context->job.q_rank = context->layer->query_lora_rank;
context->job.query_heads = context->layer->query_heads;
context->job.head_dimension = context->layer->head_dimension;
context->job.kv_width = context->layer->head_dimension;
context->job.sliding_window = context->layer->sliding_window;
context->job.compression_ratio = context->layer->compression_ratio;
context->job.output_groups = context->layer->output_groups;
context->job.output_group_input_width = context->layer->output_group_input_width;
context->job.output_rank = context->layer->output_lora_rank;
context->job.indexer_heads = context->layer->indexer_heads;
context->job.indexer_head_dimension = context->layer->indexer_head_dimension;
context->job.indexer_topk = context->layer->sparse_topk.k;
context->job.evidence_level = (unsigned int)context->opts->evidence_level;
context->job.residual_stream_count = context->layer->residual_stream_count;
context->job.residual_stream_width = context->layer->residual_stream_width;
context->job.residual_expanded_width = context->layer->residual_expanded_width;
context->job.mhc_mixing_rows = context->layer->mhc_mixing_rows;
context->job.mhc_sinkhorn_iterations = context->layer->mhc_sinkhorn_iterations;
context->job.rms_epsilon = context->layer->rms_norm_epsilon;
context->job.mhc_epsilon = context->layer->mhc_epsilon;
context->job.mhc_residual_post_multiplier =
    context->layer->mhc_residual_post_multiplier;
context->job.position.theta = context->layer->position.theta;
context->job.position.scaling_factor = context->layer->position.scaling_factor;
context->job.position.original_context = context->layer->position.original_context;
context->job.position.beta_fast = context->layer->position.beta_fast;
context->job.position.beta_slow = context->layer->position.beta_slow;
context->job.position.rope_dimensions = context->layer->rope_head_dimension;
yvex_attention_cuda_activation_project(
    &context->layer->attention_kv_activation, &context->job.attention_kv_activation);
yvex_attention_cuda_activation_project(
    &context->layer->compressor_activation, &context->job.compressor_activation);
yvex_attention_cuda_activation_project(
    &context->layer->compressor_rotated_activation,
    &context->job.compressor_rotated_activation);
yvex_attention_cuda_activation_project(
    &context->layer->indexer_query_activation, &context->job.indexer_query_activation);
context->job.input = context->opts->input;
context->job.device_input = context->opts->device_input;
context->job.device_output = context->opts->device_output;
context->job.local_kv = context->history->local_kv;
context->job.local_positions = context->history->local_positions;
context->job.local_count = context->history->local_tail_count;
context->job.local_stride = context->history->local_kv_stride;
context->job.compressed_kv = context->history->compressed_kv;
context->job.compressed_positions = context->history->compressed_positions;
context->job.compressed_count = context->history->compressed_entry_count;
context->job.compressed_stride = context->history->compressed_kv_stride;
context->job.indexer_kv = context->history->indexer_kv;
context->job.indexer_positions = context->history->indexer_positions;
context->job.indexer_count = context->history->indexer_entry_count;
context->job.indexer_stride = context->history->indexer_kv_stride;
if (context->layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
    unsigned long long end = context->opts->token_position + context->token_count;
    context->compressed_capacity = end / context->layer->compression_ratio -
                                   context->opts->token_position /
                                       context->layer->compression_ratio;
    context->indexer_capacity = context->layer->attention_class == YVEX_ATTENTION_CLASS_CSA
                                    ? context->compressed_capacity : 0ull;
}
if (context->history->main_rolling_state.present)
    (void)yvex_attention_cuda_rolling_project(
        &context->history->main_rolling_state, &context->job.main_rolling);
if (context->history->indexer_rolling_state.present)
    (void)yvex_attention_cuda_rolling_project(
        &context->history->indexer_rolling_state, &context->job.indexer_rolling);
if (context->opts->cancellation) {
    context->cancellation.requested = context->opts->cancellation->requested;
    context->cancellation.context = context->opts->cancellation->context;
    context->job.cancellation = &context->cancellation;
}
if (!yvex_backend_host_workspace_summary_get(context->backend, &workspace) ||
    !workspace.attached || !workspace.owned || !workspace.pinned)
    return yvex_attention_cuda_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, 1ull, 0ull,
        YVEX_ERR_STATE,
        "CUDA attention requires its sealed session-owned pinned staging workspace");
context->job.max_host_bytes = workspace.capacity;
context->job.max_device_bytes = 1024ull * 1024ull * 1024ull;
for (context->i = 0u; context->i < sizeof(cuda_roles) / sizeof(cuda_roles[0]);
     ++context->i) {
    const attention_cuda_role *role = &cuda_roles[context->i];
    if (!(role->class_mask & context->role_mask)) continue;
    if (role->envelope_only &&
        context->opts->operation_scope != YVEX_ATTENTION_OPERATION_ENVELOPE)
        continue;
    context->rc = yvex_attention_cuda_role_load(
        context->session, context->descriptor, context->layer,
        role->role, role->slot, &context->weights, &context->job,
        context->failure, context->err);
    if (context->rc != YVEX_OK) return context->rc;
}
    return YVEX_OK;
}
static int cuda_token_dispatch(attention_cuda_context *context)
{
yvex_attention_failure_code graph_failure =
    YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND;
yvex_error graph_error;
unsigned long long hidden, input, q_low, query, raw, index_query, index_weights, envelope, topk;
if (!yvex_core_u64_mul(context->token_count, context->trace.hidden_width, &hidden) ||
    !yvex_core_u64_mul(context->token_count, context->trace.hidden_width, &input) ||
    !yvex_core_u64_mul(context->token_count, context->trace.q_rank, &q_low) ||
    !yvex_core_u64_mul(context->token_count, context->trace.query_width, &query) ||
    !yvex_core_u64_mul(context->token_count, context->trace.kv_width, &raw) ||
    !yvex_core_u64_mul(context->token_count, context->trace.index_query_stride, &index_query) ||
    !yvex_core_u64_mul(context->token_count, context->trace.index_weight_stride, &index_weights) ||
    !yvex_core_u64_mul(context->token_count, context->trace.envelope_output_width, &envelope) ||
    !yvex_core_u64_mul(context->token_count, context->trace.topk_stride, &topk))
    return yvex_attention_cuda_reject(
        context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, ULLONG_MAX,
        context->token_count, YVEX_ERR_BOUNDS,
        "CUDA attention publication geometry overflowed");
#define FS(field, values, count) \
    context->cuda_output.field = (yvex_backend_float_span){values, values ? (count) : 0ull}
#define US(field, values, count) \
    context->cuda_output.field = (yvex_backend_u64_span){values, values ? (count) : 0ull}
FS(core_input, context->trace.input, input);
FS(q_low, context->trace.q_low, q_low);
FS(query, context->trace.query, query);
FS(raw_kv, context->trace.raw_kv, raw);
FS(compressed_kv, context->trace.compressed_kv,
   context->compressed_capacity * context->trace.compressed_stride);
FS(indexer_kv, context->trace.indexer_kv,
   context->indexer_capacity * context->trace.indexer_stride);
FS(index_query, context->trace.index_query, index_query);
FS(index_weights, context->trace.index_weights, index_weights);
FS(attention_values, context->trace.attention_values, query);
FS(output, context->trace.output, hidden);
FS(envelope_output, context->trace.envelope_output, envelope);
US(compressed_positions, context->trace.compressed_positions, context->compressed_capacity);
US(indexer_positions, context->trace.indexer_positions, context->indexer_capacity);
US(topk_counts, context->trace.topk_counts, context->token_count);
US(topk_positions, context->trace.topk_positions, topk);
FS(main_kv_state, context->trace.main_rolling_kv_checkpoints,
   context->history->main_rolling_state.kv_state_extent *
       (context->trace.rolling_checkpoint_count ? context->trace.rolling_checkpoint_count : 1ull));
FS(main_score_state, context->trace.main_rolling_score_checkpoints,
   context->history->main_rolling_state.score_state_extent *
       (context->trace.rolling_checkpoint_count ? context->trace.rolling_checkpoint_count : 1ull));
FS(indexer_kv_state, context->trace.indexer_rolling_kv_checkpoints,
   context->history->indexer_rolling_state.kv_state_extent *
       (context->trace.rolling_checkpoint_count ? context->trace.rolling_checkpoint_count : 1ull));
FS(indexer_score_state, context->trace.indexer_rolling_score_checkpoints,
   context->history->indexer_rolling_state.score_state_extent *
       (context->trace.rolling_checkpoint_count ? context->trace.rolling_checkpoint_count : 1ull));
#undef US
#undef FS
context->rc = yvex_backend_attention_execute(
    context->backend, &context->job, &context->cuda_output, &context->cuda_failure, context->err);
if (context->rc != YVEX_OK) {
    if ((size_t)context->cuda_failure.code <
        sizeof(cuda_failure_map) / sizeof(cuda_failure_map[0]))
        graph_failure = cuda_failure_map[context->cuda_failure.code];
    context->rc = yvex_attention_reject(context->failure, graph_failure, NULL,
        context->layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
        context->cuda_failure.expected, context->cuda_failure.actual, &graph_error,
        (yvex_status)context->rc,
        context->cuda_failure.stage ? context->cuda_failure.stage :
            "CUDA attention backend execution failed");
    return context->rc;
}
    return YVEX_OK;
}
static int graph_cuda_request_execute(const yvex_attention_plan *plan, const void *family_ir,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    yvex_backend *backend, const yvex_attention_cpu_options *options,
    yvex_attention_cpu_result *result, yvex_attention_failure *failure, yvex_error *err)
{
    attention_cuda_context context;
    int rc;
    memset(&context, 0, sizeof(context));
    context.plan = plan;
    context.family_ir = family_ir;
    context.session = session;
    context.descriptor = descriptor;
    context.backend = backend;
    context.opts = options;
    context.result = result;
    context.failure = failure;
    context.err = err;
    rc = cuda_token_prepare(&context);
    if (rc == YVEX_OK) rc = cuda_token_project(&context);
    if (rc == YVEX_OK) rc = cuda_token_dispatch(&context);
    if (rc == YVEX_OK) rc = yvex_attention_cuda_publish(&context);
    yvex_attention_cuda_weights_release(&context.weights);
    yvex_attention_execution_trace_release(&context.trace);
    if (rc != YVEX_OK && context.result)
        memset(context.result, 0, sizeof(*context.result));
    return rc;
}
static const yvex_graph_family_api deepseek_graph_api = {
    .plan_build = graph_plan_build,
    .draft_plan_build = graph_draft_plan_build,
    .plan_close = yvex_attention_plan_close,
    .plan_summary = yvex_attention_plan_summary,
    .plan_layer_count = yvex_attention_plan_layer_count,
    .plan_layer_at = yvex_attention_plan_layer_at,
    .selection_key_resolve = graph_selection_key_resolve,
    .state_recipe = yvex_attention_state_recipe_build,
    .workspace_recipe = yvex_attention_workspace_recipe_build,
    .history_validate = yvex_attention_history_validate,
    .rolling_state_step_cpu = yvex_attention_rolling_state_step_cpu,
    .cpu_options_default = graph_cpu_options_default,
    .publication_release = yvex_attention_execution_trace_release,
    .execution_trace_release = yvex_attention_execution_trace_release,
    .cuda_token_execute = graph_cuda_request_execute,
    .cpu_chunk_execute = graph_cpu_chunk_execute
};
const yvex_graph_family_api *yvex_graph_lower_deepseek_v4(void)
{
    return &deepseek_graph_api;
}
static int deepseek_moe_layer(unsigned long long index, const yvex_runtime_descriptor_summary *runtime,
                              const yvex_attention_layer_plan *attention, yvex_moe_layer_plan *out,
                              yvex_error *err)
{
    if (!runtime || !attention || !out || attention->ordinal != index ||
        (attention->tensor_scope == YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         (attention->layer_index >= runtime->layer_count ||
          attention->predictor_index != YVEX_MATERIALIZATION_NO_INDEX)) ||
        (attention->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT &&
         (attention->predictor_index >= runtime->draft_layer_count ||
          attention->layer_index < runtime->layer_count)) ||
        (attention->tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         attention->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.family.deepseek.moe",
                       "DeepSeek MoE projection requires one ordered admitted layer");
        return YVEX_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->schema_version = YVEX_MOE_PLAN_SCHEMA_V1; out->ordinal = index;
    out->layer_index = attention->layer_index;
    out->predictor_index = attention->predictor_index;
    out->tensor_scope = attention->tensor_scope;
    out->router_class = attention->tensor_scope == YVEX_TENSOR_SCOPE_MAIN_LAYER &&
                                attention->layer_index < 3ull
                            ? YVEX_MOE_ROUTER_HASH_TOKEN_ID :
                                      YVEX_MOE_ROUTER_LEARNED_HIDDEN_STATE;
    out->scoring = YVEX_MOE_SCORING_SQRT_SOFTPLUS; out->topk_policy = YVEX_MOE_TOPK_NOAUX_TC;
    out->activation = YVEX_MOE_ACTIVATION_SILU; out->hidden_width = attention->hidden_dimension;
    out->residual_streams = attention->residual_stream_count; out->expanded_width = attention->residual_expanded_width;
    out->mhc_mixing_rows = attention->mhc_mixing_rows;
    out->mhc_sinkhorn_iterations = attention->mhc_sinkhorn_iterations;
    out->rms_epsilon = attention->rms_norm_epsilon; out->mhc_epsilon = attention->mhc_epsilon;
    out->mhc_post_multiplier = attention->mhc_residual_post_multiplier;
    out->routed_experts = runtime->routed_experts; out->shared_experts = 1ull;
    out->experts_per_token = runtime->experts_per_token;
    out->expert_intermediate_width = out->shared_intermediate_width = 2048ull;
    out->hash_table_rows = runtime->vocabulary_size; out->hash_table_columns = out->experts_per_token;
    out->correction_bias_width = out->routed_experts; out->routed_scaling_factor = 1.5;
    out->activation_limit = 10.0;
    out->requires_token_ids = out->router_class == YVEX_MOE_ROUTER_HASH_TOKEN_ID;
    out->requires_correction_bias = !out->requires_token_ids; out->normalize_topk_probabilities = 1;
    yvex_error_clear(err);
    return YVEX_OK;
}
static const yvex_moe_family_api deepseek_moe_api = {
    .adapter_id = 0x44535634ull, .adapter_version = 7ull, .project_layer = deepseek_moe_layer};
const yvex_moe_family_api *yvex_graph_moe_family_at(unsigned long long index) {
    return index == 0ull ? &deepseek_moe_api : NULL;
}
static int deepseek_mixer_capability(yvex_sequence_mixer_semantics semantics,
                                     yvex_runtime_mixer_capability *out)
{
    if (!out) return 0;
    out->family = YVEX_SEQUENCE_MIXER_SOFTMAX_ATTENTION;
    out->semantics = semantics;
    out->state = YVEX_RUNTIME_MIXER_NOT_IMPLEMENTED;
    out->reason = "sequence-mixer semantics are not implemented by this family adapter";
    if (semantics == YVEX_SEQUENCE_MIXER_SLIDING_WINDOW ||
        semantics == YVEX_SEQUENCE_MIXER_COMPRESSED_SPARSE ||
        semantics == YVEX_SEQUENCE_MIXER_HIERARCHICAL_COMPRESSED) {
        out->state = YVEX_RUNTIME_MIXER_SUPPORTED;
        out->reason = "admitted DeepSeek attention semantics";
    } else if (semantics >= YVEX_SEQUENCE_MIXER_DELTANET) {
        out->state = YVEX_RUNTIME_MIXER_NOT_ADMITTED;
        out->reason = "sequence-mixer family is outside the admitted DeepSeek softmax adapter";
    }
    return 1;
}
static int deepseek_execution_capabilities(yvex_runtime_capabilities *out)
{
    if (!out) return 0;
    *out = (yvex_runtime_capabilities){
        .attention_semantics_ready = 1, .attention_core_ready = 1,
        .attention_envelope_ready = 1, .cpu_prefill_eager_ready = 1,
        .cpu_decode_eager_ready = 1, .cuda_eager_implemented = 1,
        .cuda_piecewise_graph_implemented = 1, .cuda_full_graph_implemented = 1,
        .attention_state_delta_ready = 1, .attention_operator_ready = 1,
        .attention_trace_ready = 1, .attention_profile_ready = 1,
        .attention_benchmark_ready = 1, .moe_plan_ready = 1,
        .moe_router_ready = 1, .moe_routed_expert_ready = 1,
        .moe_shared_expert_ready = 1, .moe_block_ready = 1, .transformer_ready = 1,
        .output_head_binding_ready = 1, .output_head_projection_ready = 1,
        .logits_cpu_ready = 1, .logits_cuda_ready = 1, .logits_prefill_ready = 1,
        .logits_decode_ready = 1, .logits_full_vocabulary_ready = 1,
        .logits_hidden_contract_ready = 1, .logits_partial_progress_ready = 1,
        .logits_ready = 1};
    return yvex_runtime_capabilities_contract_valid(out);
}
static int deepseek_transformer_policy(yvex_transformer_family_policy *out)
{
    if (!out) return 0;
    *out = (yvex_transformer_family_policy){
        .schema_version = YVEX_TRANSFORMER_PLAN_SCHEMA_V2,
        .initial_policy = YVEX_TRANSFORMER_INITIAL_REPEAT_STREAMS,
        .final_policy = YVEX_TRANSFORMER_FINAL_SIGMOID_MHC_RMS,
        .residual_streams = 4ull, .hidden_width = 4096ull,
        .expanded_width = 16384ull, .maximum_context = 1048576ull,
        .sinkhorn_iterations = 20ull, .mhc_epsilon = 1e-6,
        .output_norm_epsilon = 1e-6, .attention_then_moe = 1,
        .deferred_ffn_post = 1, .final_norm_after_head = 1};
    return 1;
}
static int deepseek_logits_policy(yvex_logits_family_policy *out)
{
    if (!out) return 0;
    *out = (yvex_logits_family_policy){
        .schema_version = YVEX_RUNTIME_LOGITS_SCHEMA_V1,
        .separate_output_head = 1, .tied_output_head = 0, .output_head_bias = 0};
    return 1;
}
static int deepseek_speculation_policy(yvex_speculation_family_policy *out)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (!out) return 0;
    *out = (yvex_speculation_family_policy){
        .schema_version = YVEX_SPECULATION_FAMILY_POLICY_SCHEMA_V1,
        .block_size = 5ull, .noise_token_id = 128799ull,
        .target_feature_layer_count = 3ull,
        .target_feature_layers = {40ull, 41ull, 42ull},
        .target_feature_width = 4096ull,
        .concatenated_feature_width = 12288ull,
        .draft_layer_count = 3ull, .markov_rank = 256ull,
        .accepted_prefix_maximum = 5ull,
        .feature_projection_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_PROJECTION,
        .feature_norm_role = YVEX_TENSOR_ROLE_DRAFT_FEATURE_NORM,
        .output_norm_role = YVEX_TENSOR_ROLE_DRAFT_OUTPUT_NORM,
        .markov_embedding_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_EMBEDDING,
        .markov_output_role = YVEX_TENSOR_ROLE_DRAFT_MARKOV_OUTPUT,
        .confidence_role = YVEX_TENSOR_ROLE_DRAFT_CONFIDENCE,
        .parallel_block_backbone = 1, .sequential_markov = 1,
        .confidence_available = 1, .shares_embedding = 1,
        .shares_output_head = 1, .target_verification_required = 1};
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.deepseek-v4.dspark.policy.v1") ||
        !yvex_sha256_update_u64(&hash, out->block_size) ||
        !yvex_sha256_update_u64(&hash, out->noise_token_id) ||
        !yvex_sha256_update_u64(&hash, out->target_feature_layer_count) ||
        !yvex_sha256_update_u64(&hash, out->target_feature_width) ||
        !yvex_sha256_update_u64(&hash, out->concatenated_feature_width) ||
        !yvex_sha256_update_u64(&hash, out->draft_layer_count) ||
        !yvex_sha256_update_u64(&hash, out->markov_rank))
        return 0;
    for (index = 0ull; index < out->target_feature_layer_count; ++index)
        if (!yvex_sha256_update_u64(&hash, out->target_feature_layers[index])) return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, out->policy_identity);
    return 1;
}
static const yvex_runtime_family_adapter deepseek_adapter = {
    .schema_version = YVEX_RUNTIME_FAMILY_ADAPTER_SCHEMA_V2,
    .adapter_id = 0x44535634ull, .adapter_version = 7ull,
    .target_id = "deepseek4-v4-flash-dspark",
    .family_name = "deepseek-v4-flash-dspark",
    .operator_family_key = "deepseek",
    .operator_artifact_filename = YVEX_SELECTED_DEEPSEEK_ARTIFACT_FILENAME,
    .logical_transform_identity = YVEX_SELECTED_DEEPSEEK_TRANSFORM_IDENTITY,
    .mixer_family = YVEX_SEQUENCE_MIXER_SOFTMAX_ATTENTION,
    .mixer_capability = deepseek_mixer_capability,
    .graph = yvex_graph_lower_deepseek_v4,
    .execution_capabilities = deepseek_execution_capabilities,
    .transformer_policy = deepseek_transformer_policy,
    .logits_policy = deepseek_logits_policy,
    .speculation_policy = deepseek_speculation_policy};
const yvex_runtime_family_adapter *yvex_runtime_family_at(unsigned long long index)
{
    return index == 0ull ? &deepseek_adapter : NULL;
}
