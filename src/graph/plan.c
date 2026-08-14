/*
 * Derive bounded executable requirements from admitted model and runtime facts.
 *
 * Planning is deterministic, immutable after build, and reads zero payload bytes. Plan and
 * capability facts are not graph execution or runtime support.
 */
#include "src/graph/private.h"

#include <yvex/internal/backend.h>
#include <yvex/internal/graph_state.h>
#include <stdlib.h>
#include <string.h>

static void state_recipe_history_add(
    yvex_attention_state_recipe *recipe, yvex_attention_state_binding binding,
    unsigned long long capacity, unsigned long long width)
{
    yvex_attention_state_component_recipe *component =
        &recipe->components[recipe->component_count];
    *component = (yvex_attention_state_component_recipe){
        .schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1,
        .ordinal = recipe->component_count++,
        .kind = YVEX_ATTENTION_STATE_COMPONENT_HISTORY,
        .binding = binding, .capacity = capacity, .value_width = width};
}

static int state_recipe_rolling_add(
    yvex_attention_state_recipe *recipe, const yvex_attention_layer_plan *layer,
    yvex_attention_rolling_kind kind, yvex_attention_state_binding binding)
{
    yvex_attention_state_component_recipe *component =
        &recipe->components[recipe->component_count];
    unsigned long long ratio, head, width, slots, extent;
    unsigned long long previous_fill, current_fill;
    int overlap, rotated;
    if (!yvex_attention_rolling_geometry(layer, kind, &ratio, &head, &width,
                                         &slots, &overlap, &rotated) ||
        !yvex_core_u64_mul(width, slots, &extent))
        return 0;
    previous_fill = overlap && recipe->initial_position >= ratio ? ratio : 0ull;
    current_fill = recipe->initial_position % ratio;
    *component = (yvex_attention_state_component_recipe){
        .schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1,
        .ordinal = recipe->component_count++,
        .kind = YVEX_ATTENTION_STATE_COMPONENT_ROLLING,
        .binding = binding, .capacity = slots, .value_width = width,
        .rolling = {
            .present = 1, .schema_version = YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1,
            .kind = kind, .attention_class = layer->attention_class,
            .layer_index = layer->layer_index,
            .next_token_position = recipe->initial_position,
            .ratio = ratio, .head_dimension = head, .state_width = width,
            .state_slots = slots, .cursor = current_fill,
            .previous_fill = previous_fill, .current_fill = current_fill,
            .kv_state_stride = width, .score_state_stride = width,
            .kv_state_extent = extent, .score_state_extent = extent,
            .overlap = overlap, .rotated = rotated}};
    yvex_core_text_copy(component->rolling.attention_plan_identity,
                        sizeof(component->rolling.attention_plan_identity),
                        recipe->attention_plan_identity);
    return 1;
}

int yvex_attention_state_recipe_build(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe_request *request,
    yvex_attention_state_recipe *recipe, yvex_attention_failure *failure,
    yvex_error *err)
{
    unsigned long long local_capacity, compressed_capacity = 0ull;
    if (!layer || !request || !recipe || !request->attention_plan_identity ||
        request->final_position < request->initial_position || !layer->sliding_window)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_HISTORY, NULL,
            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
            "attention state recipe requires complete bounded facts");
    memset(recipe, 0, sizeof(*recipe));
    recipe->schema_version = YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1;
    recipe->layer_index = layer->layer_index;
    recipe->selection_key = (unsigned long long)layer->attention_class + 1ull;
    recipe->initial_position = request->initial_position;
    recipe->final_position = request->final_position;
    yvex_core_text_copy(recipe->attention_plan_identity,
                        sizeof(recipe->attention_plan_identity),
                        request->attention_plan_identity);
    /* A draft block reads the complete retained window before adding its ephemeral
     * candidate rows. Target attention reserves one slot for the current row. */
    local_capacity = layer->sliding_window -
                     (layer->tensor_scope == YVEX_TENSOR_SCOPE_DRAFT ? 0ull : 1ull);
    if (request->final_position < local_capacity) local_capacity = request->final_position;
    state_recipe_history_add(recipe, YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY,
                             local_capacity, layer->head_dimension);
    if (layer->compressor_required) {
        if (!layer->compression_ratio) goto malformed;
        compressed_capacity = request->final_position / layer->compression_ratio;
        state_recipe_history_add(
            recipe, YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY,
            compressed_capacity, layer->head_dimension);
        if (!state_recipe_rolling_add(
                recipe, layer, YVEX_ATTENTION_ROLLING_MAIN,
                YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING))
            goto malformed;
    }
    if (layer->indexer_required) {
        state_recipe_history_add(
            recipe, YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY,
            compressed_capacity, layer->indexer_head_dimension);
        if (!state_recipe_rolling_add(
                recipe, layer, YVEX_ATTENTION_ROLLING_INDEXER,
                YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING))
            goto malformed;
    }
    return yvex_attention_state_recipe_seal(recipe, err);
malformed:
    return yvex_attention_reject(
        failure, YVEX_ATTENTION_FAILURE_HISTORY, NULL, layer->layer_index,
        YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err, YVEX_ERR_FORMAT,
        "attention rolling state recipe is malformed");
}

static const yvex_attention_state_component_recipe *state_recipe_component(
    const yvex_attention_state_recipe *recipe,
    yvex_attention_state_binding binding)
{
    unsigned int index;
    for (index = 0u; index < recipe->component_count; ++index)
        if (recipe->components[index].binding == binding) return &recipe->components[index];
    return NULL;
}

typedef struct workspace_recipe_slot {
    yvex_attention_workspace_component_kind kind;
    yvex_attention_workspace_lifetime lifetime;
    int scales_with_tokens;
} workspace_recipe_slot;

static void workspace_recipe_add(
    yvex_attention_workspace_recipe *recipe,
    const workspace_recipe_slot *slot,
    unsigned long long count, unsigned long long width)
{
    if (!count) return;
    recipe->components[recipe->component_count] = (yvex_attention_workspace_component){
        .schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1,
        .ordinal = recipe->component_count++, .kind = slot->kind,
        .lifetime = slot->lifetime,
        .element_count = count, .element_width = width, .alignment = 256ull,
        .scales_with_tokens = slot->scales_with_tokens};
}

static const workspace_recipe_slot workspace_recipe_layout[] = {
    {YVEX_ATTENTION_WORKSPACE_INGRESS, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 1},
    {YVEX_ATTENTION_WORKSPACE_LOCAL_VALUES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_LOCAL_POSITIONS, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_COMPRESSED_VALUES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_COMPRESSED_POSITIONS, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_VALUES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_POSITIONS, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_VALUES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_SCORES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_VALUES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_SCORES, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_STATE_COUNTER, YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE, 0},
    {YVEX_ATTENTION_WORKSPACE_STATUS, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_SELECTION_COUNTER, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_ENVELOPE_OUTPUT, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_Q_LOW, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_QUERY, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_RAW_KV, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_ATTENTION_VALUES, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_OUTPUT, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_ENVELOPE_STAGING, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_COMPRESSED_EMISSION, YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 1},
    {YVEX_ATTENTION_WORKSPACE_COMPRESSED_EMISSION_POSITION,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 1},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_EMISSION, YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 1},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_EMISSION_POSITION,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 1},
    {YVEX_ATTENTION_WORKSPACE_INDEX_QUERY, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_INDEX_WEIGHTS, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_TOPK_INDICES, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_CANDIDATE_VALUES,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 0},
    {YVEX_ATTENTION_WORKSPACE_MAIN_ROLLING_CANDIDATE_SCORES,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_CANDIDATE_VALUES,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 0},
    {YVEX_ATTENTION_WORKSPACE_INDEXER_ROLLING_CANDIDATE_SCORES,
     YVEX_ATTENTION_WORKSPACE_STATE_DELTA, 0},
    {YVEX_ATTENTION_WORKSPACE_CORE_INPUT_EVIDENCE,
     YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_OUTPUT_LOW, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_TOPK_POSITIONS, YVEX_ATTENTION_WORKSPACE_EXECUTION, 1},
    {YVEX_ATTENTION_WORKSPACE_TOPK_SCORES, YVEX_ATTENTION_WORKSPACE_EXECUTION, 0},
    {YVEX_ATTENTION_WORKSPACE_TOPK_VALID_INDICES, YVEX_ATTENTION_WORKSPACE_EXECUTION, 0}
};

/*
 * Derive one generic attention-workspace recipe from sealed state facts.
 *
 * Seals deterministic component extents. Immutable planning owns no backend byte layout or runtime
 * allocation.
 */
int yvex_attention_workspace_recipe_build(
    const yvex_attention_layer_plan *layer,
    const yvex_attention_state_recipe *state, yvex_attention_execution_mode mode,
    yvex_attention_operation_scope scope, yvex_attention_evidence_level evidence_level,
    unsigned long long token_capacity, yvex_attention_workspace_recipe *recipe,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_state_component_recipe *local, *compressed, *indexer, *main, *index;
    yvex_attention_state_recipe state_copy;
    unsigned long long query, index_query = 0ull, head_bytes, index_bytes = 0ull;
    unsigned long long input_bytes, residual_bytes, output_low, candidates = 0ull, selected = 0ull;
    unsigned long long counts[sizeof(workspace_recipe_layout) / sizeof(workspace_recipe_layout[0])];
    unsigned long long widths[sizeof(workspace_recipe_layout) / sizeof(workspace_recipe_layout[0])] = {0ull};
    size_t slot;
    if (!layer || !state || !recipe || !token_capacity) goto malformed;
    state_copy = *state;
    if (yvex_attention_state_recipe_seal(&state_copy, err) != YVEX_OK ||
        layer->layer_index != state->layer_index || mode > YVEX_ATTENTION_EXECUTION_FULL ||
        scope > YVEX_ATTENTION_OPERATION_RELEASE_SET || evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        !yvex_core_u64_mul(layer->query_heads, layer->head_dimension, &query) ||
        !yvex_core_u64_mul(layer->head_dimension, sizeof(float), &head_bytes) ||
        !yvex_core_u64_mul(layer->hidden_dimension, sizeof(float), &input_bytes) ||
        !yvex_core_u64_mul(layer->residual_expanded_width, sizeof(float), &residual_bytes) ||
        !yvex_core_u64_mul(layer->output_groups, layer->output_lora_rank, &output_low) ||
        (layer->indexer_required &&
         (!yvex_core_u64_mul(layer->indexer_heads, layer->indexer_head_dimension,
                             &index_query) ||
          !yvex_core_u64_mul(layer->indexer_head_dimension, sizeof(float), &index_bytes))))
        goto malformed;
    local = state_recipe_component(state, YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY);
    compressed = state_recipe_component(state, YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY);
    indexer = state_recipe_component(state, YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY);
    main = state_recipe_component(state, YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING);
    index = state_recipe_component(state, YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING);
    if (indexer && !yvex_core_u64_add(indexer->capacity, 1ull, &candidates)) goto malformed;
    selected = candidates < layer->indexer_topk ? candidates : layer->indexer_topk;
    if (layer->indexer_required && !selected) selected = 1ull;
    memset(recipe, 0, sizeof(*recipe));
    recipe->schema_version = YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1;
    recipe->layer_index = layer->layer_index;
    recipe->mode = mode;
    recipe->scope = scope;
    recipe->evidence_level = evidence_level;
    recipe->token_capacity = token_capacity;
    yvex_core_text_copy(recipe->state_recipe_identity,
                        sizeof(recipe->state_recipe_identity), state->identity);
    counts[0] = 1ull;
    if (local &&
        !yvex_core_u64_add(local->capacity, token_capacity, &counts[1]))
        goto malformed;
    counts[2] = counts[1];
    counts[3] = counts[4] = compressed ? compressed->capacity : 0ull;
    counts[5] = counts[6] = indexer ? indexer->capacity : 0ull;
    counts[7] = main ? main->rolling.kv_state_extent : 0ull;
    counts[8] = main ? main->rolling.score_state_extent : 0ull;
    counts[9] = index ? index->rolling.kv_state_extent : 0ull;
    counts[10] = index ? index->rolling.score_state_extent : 0ull;
    counts[11] = 2ull;
    counts[12] = 1ull;
    counts[13] = layer->indexer_required ? 2ull : 0ull;
    counts[14] = scope != YVEX_ATTENTION_OPERATION_CORE ? layer->hidden_dimension : 0ull;
    counts[15] = layer->query_lora_rank;
    counts[16] = query;
    counts[17] = layer->head_dimension;
    counts[18] = query;
    counts[19] = layer->hidden_dimension;
    counts[20] = scope != YVEX_ATTENTION_OPERATION_CORE
                     ? layer->residual_expanded_width : 0ull;
    counts[21] = main ? layer->head_dimension : 0ull;
    counts[22] = main ? 1ull : 0ull;
    counts[23] = index ? layer->indexer_head_dimension : 0ull;
    counts[24] = index ? 1ull : 0ull;
    counts[25] = index ? index_query : 0ull;
    counts[26] = index ? layer->indexer_heads : 0ull;
    counts[27] = index ? selected : 0ull;
    counts[28] = main ? main->rolling.kv_state_extent : 0ull;
    counts[29] = main ? main->rolling.score_state_extent : 0ull;
    counts[30] = index ? index->rolling.kv_state_extent : 0ull;
    counts[31] = index ? index->rolling.score_state_extent : 0ull;
    counts[32] = evidence_level == YVEX_ATTENTION_EVIDENCE_FULL ? layer->hidden_dimension : 0ull;
    counts[33] = output_low;
    counts[34] = selected;
    counts[35] = counts[36] = candidates;
    widths[0] = scope == YVEX_ATTENTION_OPERATION_CORE ? input_bytes : residual_bytes;
    widths[1] = head_bytes;
    widths[2] = sizeof(unsigned long long);
    widths[3] = head_bytes;
    widths[4] = sizeof(unsigned long long);
    widths[5] = index_bytes;
    widths[6] = sizeof(unsigned long long);
    for (slot = 7u; slot < 11u; ++slot) widths[slot] = sizeof(float);
    widths[11] = widths[13] = widths[22] = widths[24] = widths[27] =
        sizeof(unsigned long long);
    widths[12] = sizeof(int);
    for (slot = 14u; slot < 34u; ++slot)
        if (!widths[slot]) widths[slot] = sizeof(float);
    widths[34] = widths[36] = sizeof(unsigned long long);
    widths[35] = sizeof(float);
    for (slot = 0u;
         slot < sizeof(workspace_recipe_layout) / sizeof(workspace_recipe_layout[0]);
         ++slot)
        workspace_recipe_add(recipe, &workspace_recipe_layout[slot],
                             counts[slot], widths[slot]);
    return yvex_attention_workspace_recipe_seal(recipe, err);
malformed:
    return yvex_attention_reject(
        failure, YVEX_ATTENTION_FAILURE_SCRATCH, NULL,
        layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
        YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err, YVEX_ERR_FORMAT,
        "attention workspace recipe is malformed");
}

static void attention_failure_set(
    yvex_attention_failure *failure,
    yvex_attention_failure_code code,
    const yvex_runtime_tensor_binding *binding,
    unsigned long long layer_index,
    yvex_tensor_role role,
    unsigned long long expected,
    unsigned long long actual,
    const char *reason)
{
    if (!failure) return;
    memset(failure, 0, sizeof(*failure));
    failure->code = code;
    failure->layer_index = layer_index;
    failure->role = role;
    failure->expected = expected;
    failure->actual = actual;
    failure->reason = reason;
    if (binding && binding->binding)
        yvex_core_text_copy(failure->tensor_name, sizeof(failure->tensor_name), binding->binding->name);
}

int yvex_attention_reject(yvex_attention_failure *failure,
                            yvex_attention_failure_code code,
                            const yvex_runtime_tensor_binding *binding,
                            unsigned long long layer_index,
                            yvex_tensor_role role,
                            unsigned long long expected,
                            unsigned long long actual,
                            yvex_error *err,
                            yvex_status err_code,
                            const char *reason)
{
    attention_failure_set(failure, code, binding, layer_index, role, expected,
                          actual, reason);
    yvex_error_set(err, err_code, "yvex_deepseek_attention", reason);
    return err_code;
}

int yvex_attention_accept(yvex_attention_failure *failure, yvex_error *err)
{
    yvex_error_clear(err);
    if (failure) memset(failure, 0, sizeof(*failure));
    return YVEX_OK;
}

int yvex_attention_checked_size(unsigned long long count,
                                  unsigned long long width,
                                  size_t *out)
{
    unsigned long long bytes;

    if (!out || !yvex_core_u64_mul(count, width, &bytes) ||
        bytes > (unsigned long long)SIZE_MAX)
        return 0;
    *out = (size_t)bytes;
    return 1;
}

void *yvex_attention_calloc_array(unsigned long long count,
                                    unsigned long long width)
{
    size_t bytes;

    if (!yvex_attention_checked_size(count, width, &bytes)) return NULL;
    return calloc(1u, bytes);
}

int yvex_attention_context_validate(
    const yvex_attention_plan *plan,
    const char *logical_identity,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_attention_summary *attention;
    const yvex_runtime_descriptor_summary *runtime;
    const yvex_materialization_summary *materialization;

    if (!plan || !logical_identity || !logical_identity[0] || !session ||
        !descriptor)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_INVALID_ARG,
            "attention execution identity validation requires all owners");
    attention = yvex_attention_plan_summary(plan);
    runtime = yvex_runtime_descriptor_summary_get(descriptor);
    materialization = yvex_materialization_session_summary(session);
    if (!attention || !runtime || !materialization ||
        !materialization->committed)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "attention execution requires sealed identity-bearing owners");
    if (strcmp(logical_identity, runtime->logical_model_identity) != 0 ||
        strcmp(logical_identity, attention->logical_model_identity) != 0 ||
        strcmp(runtime->runtime_numeric_identity,
               attention->runtime_numeric_identity) != 0 ||
        strcmp(runtime->runtime_descriptor_identity,
               attention->runtime_descriptor_identity) != 0 ||
        strcmp(materialization->plan_identity,
               runtime->materialization_plan_identity) != 0 ||
        strcmp(materialization->plan_identity,
               attention->materialization_plan_identity) != 0)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "attention execution refused a stale or mismatched identity chain");
    return YVEX_OK;
}
typedef enum {
    PLAN_HASH_INT,
    PLAN_HASH_UINT,
    PLAN_HASH_U64,
    PLAN_HASH_F64
} plan_hash_kind;

typedef struct {
    size_t offset;
    plan_hash_kind kind;
} plan_hash_field;

#define PLAN_FIELD(type, member, kind) {offsetof(type, member), kind}
static const plan_hash_field plan_layer_fields[] = {
    PLAN_FIELD(yvex_attention_layer_plan, ordinal, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, layer_index, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, predictor_index, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, tensor_scope, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, attention_class, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, compute_contract, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, compression_ratio, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, sliding_window, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, query_heads, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, kv_heads, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, head_dimension, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, rope_head_dimension, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, query_lora_rank, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, output_lora_rank, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, output_groups, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, output_group_input_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, hidden_dimension, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, indexer_heads, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, indexer_head_dimension, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, indexer_topk, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, compressor_ape_columns, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, indexer_ape_columns, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, rms_norm_epsilon, PLAN_HASH_F64),
    PLAN_FIELD(yvex_attention_layer_plan, residual_stream_count, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, residual_stream_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, residual_expanded_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_mixing_rows, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_mixing_columns, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_base_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_scale_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_sinkhorn_iterations, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, attention_input_norm_width, PLAN_HASH_U64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_epsilon, PLAN_HASH_F64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_residual_post_multiplier, PLAN_HASH_F64),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_entry_policy, PLAN_HASH_UINT),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_attention_pre_and_post, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, attention_input_norm_required, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, attention_input_norm_role, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_function_role, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_base_role, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, mhc_scale_role, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, compressor_required, PLAN_HASH_INT),
    PLAN_FIELD(yvex_attention_layer_plan, indexer_required, PLAN_HASH_INT),
};

static const yvex_attention_summary attention_ready_summary = {
    .status = YVEX_ATTENTION_STATUS_EXECUTION_READY,
    .history_contract_ready = 1,
    .state_delta_contract_ready = 1,
    .cpu_reference_ready = 1,
    .cuda_execution_ready = 1,
    .full_execution_ready = 1,
};

#undef PLAN_FIELD
void yvex_attention_plan_close(yvex_attention_plan *plan);

/*
 * Allocate the common immutable storage used by built and imported plans.
 *
 * Allocation performs no family discovery, binding, or identity work.
 */
static int attention_plan_allocate(
    yvex_attention_plan **out, unsigned long long layer_count,
    const char *plan_reason, const char *layer_reason,
    yvex_attention_failure *failure, yvex_error *err)
{
    yvex_attention_plan *plan;
    size_t layer_bytes;

    *out = NULL;
    if (!yvex_attention_checked_size(layer_count, sizeof(*plan->layers),
                                     &layer_bytes))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            (unsigned long long)SIZE_MAX, layer_count, err, YVEX_ERR_BOUNDS,
            "attention layer count exceeds platform capacity");
    plan = (yvex_attention_plan *)calloc(1u, sizeof(*plan));
    if (plan)
        plan->layers = (yvex_attention_layer_plan *)calloc(1u, layer_bytes);
    if (!plan || !plan->layers) {
        yvex_attention_plan_close(plan);
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            plan ? layer_count : sizeof(*plan), 0ull, err, YVEX_ERR_NOMEM,
            plan ? layer_reason : plan_reason);
    }
    plan->layer_count = layer_count;
    *out = plan;
    return YVEX_OK;
}

/*
 * Bind the runtime identity chain and recipe counts into a ready plan summary.
 *
 * Bounded identity copies always terminate their destinations. This projection neither computes an
 * identity nor reads payload bytes.
 */
static void attention_summary_initialize(
    yvex_attention_summary *summary, const yvex_runtime_descriptor_summary *runtime,
    const yvex_attention_recipe *recipe)
{
    *summary = attention_ready_summary;
    yvex_core_text_copy(summary->artifact_identity, sizeof(summary->artifact_identity),
                        runtime->artifact_identity);
    yvex_core_text_copy(summary->materialization_plan_identity,
                        sizeof(summary->materialization_plan_identity),
                        runtime->materialization_plan_identity);
    yvex_core_text_copy(summary->logical_model_identity,
                        sizeof(summary->logical_model_identity),
                        runtime->logical_model_identity);
    yvex_core_text_copy(summary->runtime_descriptor_identity,
                        sizeof(summary->runtime_descriptor_identity),
                        runtime->runtime_descriptor_identity);
    yvex_core_text_copy(summary->runtime_numeric_identity,
                        sizeof(summary->runtime_numeric_identity),
                        runtime->runtime_numeric_identity);
    summary->layer_count = recipe->layer_count;
    summary->auxiliary_layer_count = recipe->auxiliary_layer_count;
    summary->tensor_scope = recipe->tensor_scope;
    summary->swa_layer_count = recipe->swa_layer_count;
    summary->csa_layer_count = recipe->csa_layer_count;
    summary->hca_layer_count = recipe->hca_layer_count;
}

static void plan_hash_f64(yvex_sha256 *hash, double value)
{
    unsigned long long bits = 0ull;
    memcpy(&bits, &value, sizeof(bits));
    (void)attention_hash_u64(hash, bits);
}

/*
 * Serialize an ordered semantic field catalog without hashing native object storage.
 *
 * Bounded in-memory hash updates have no recoverable failure at this legacy boundary. Offsets
 * select declared fields only; padding and pointers never enter identities.
 */
static void plan_hash_fields(yvex_sha256 *hash, const void *object,
                             const plan_hash_field *fields, size_t count)
{
    const unsigned char *bytes = (const unsigned char *)object;
    size_t index;

    for (index = 0u; index < count; ++index) {
        unsigned long long value = 0ull;

        if (fields[index].kind == PLAN_HASH_F64) {
            double floating;
            memcpy(&floating, bytes + fields[index].offset, sizeof(floating));
            plan_hash_f64(hash, floating);
            continue;
        }
        if (fields[index].kind == PLAN_HASH_U64)
            memcpy(&value, bytes + fields[index].offset, sizeof(value));
        else if (fields[index].kind == PLAN_HASH_UINT) {
            unsigned int narrow;
            memcpy(&narrow, bytes + fields[index].offset, sizeof(narrow));
            value = narrow;
        } else {
            int narrow;
            memcpy(&narrow, bytes + fields[index].offset, sizeof(narrow));
            value = (unsigned long long)narrow;
        }
        (void)attention_hash_u64(hash, value);
    }
}

/* Derive the deterministic compute identity identity from explicit semantic fields. */
int yvex_attention_plan_identity_compute(
    const yvex_attention_summary *summary,
    const yvex_attention_layer_plan *layers,
    unsigned long long layer_count,
    char output[YVEX_ATTENTION_IDENTITY_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long i;

    if (!summary || !layers || !layer_count || layer_count != summary->layer_count || !output)
        return 0;
    yvex_sha256_init(&hash);
    (void)attention_hash_text(&hash, "yvex.deepseek.attention.plan.v4");
    (void)attention_hash_text(&hash, summary->artifact_identity);
    (void)attention_hash_text(&hash, summary->materialization_plan_identity);
    (void)attention_hash_text(&hash, summary->logical_model_identity);
    (void)attention_hash_text(&hash, summary->runtime_descriptor_identity);
    (void)attention_hash_text(&hash, summary->runtime_numeric_identity);
    (void)attention_hash_u64(&hash, summary->layer_count);
    (void)attention_hash_u64(&hash, summary->tensor_scope);
    (void)attention_hash_u64(&hash, summary->required_envelope_binding_count);
    for (i = 0ull; i < layer_count; ++i) {
        const yvex_attention_layer_plan *layer = &layers[i];
        plan_hash_fields(&hash, layer, plan_layer_fields,
                         sizeof(plan_layer_fields) / sizeof(plan_layer_fields[0]));
        (void)yvex_model_position_identity_update(&hash, &layer->position);
        (void)yvex_model_activation_identity_update(&hash, &layer->attention_kv_activation);
        (void)yvex_model_activation_identity_update(&hash, &layer->compressor_activation);
        (void)yvex_model_activation_identity_update(
            &hash, &layer->compressor_rotated_activation);
        (void)yvex_model_activation_identity_update(&hash, &layer->indexer_query_activation);
        (void)yvex_model_topk_identity_update(&hash, &layer->sparse_topk);
        (void)attention_hash_u64(&hash, layer->required_binding_count);
        (void)attention_hash_u64(&hash, layer->qtype_compute_refusal_count);
        (void)attention_hash_u64(&hash, layer->payload_bytes_bound);
    }
    (void)yvex_sha256_final(&hash, digest);
    yvex_sha256_hex(digest, output);
    return 1;
}

static int attention_envelope_binding_count(
    const yvex_attention_plan *plan, const yvex_runtime_descriptor *descriptor,
    unsigned long long *count, yvex_attention_failure_code failure_code,
    const char *reason, yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_runtime_descriptor_summary *summary =
        yvex_runtime_descriptor_summary_get(descriptor);
    unsigned long long index;

    *count = 0ull;
    for (index = 0ull; summary && index < summary->tensor_count; ++index) {
        const yvex_runtime_tensor_binding *binding =
            yvex_runtime_descriptor_tensor_at(descriptor, index);
        if (yvex_attention_plan_binding_classify(plan, binding) ==
                YVEX_ATTENTION_BINDING_ENVELOPE &&
            !yvex_core_u64_add(*count, 1ull, count))
            return yvex_attention_reject(
                failure, failure_code, binding, YVEX_ATTENTION_NO_LAYER,
                binding ? binding->role : YVEX_TENSOR_ROLE_UNKNOWN,
                ULLONG_MAX, index, err, YVEX_ERR_BOUNDS, reason);
    }
    return YVEX_OK;
}

static int attention_bind_role(
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_layer_plan *layer_plan,
    yvex_attention_summary *summary,
    yvex_tensor_scope scope,
    unsigned long long layer_index,
    unsigned long long predictor_index,
    yvex_tensor_role role,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_runtime_tensor_binding *binding =
        yvex_runtime_descriptor_find_role(
            descriptor, role, scope, layer_index, predictor_index);
    const yvex_quant_numeric_capability *capability;
    unsigned long long total;

    if (!layer_plan || !summary ||
        !yvex_core_u64_add(layer_plan ? layer_plan->required_binding_count : 0ull,
                           1ull, &total))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DIMENSION, binding,
            layer_index, role, ULLONG_MAX,
            layer_plan ? layer_plan->required_binding_count : 0ull,
            err, YVEX_ERR_BOUNDS,
            "DeepSeek attention binding count overflowed");
    layer_plan->required_binding_count = total;
    if (!binding || !binding->binding) {
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_MISSING_BINDING, binding,
            layer_index, role, 1ull, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention binding is missing from runtime descriptor");
    }
    if (!binding->binding->encoded_bytes || !binding->binding->rank) {
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DIMENSION, binding,
            layer_index, role, 1ull, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention binding has empty shape or byte range");
    }
    capability = yvex_quant_numeric_capability_at(binding->qtype);
    if (!capability || !capability->identity_known ||
        !capability->storage_admitted ||
        !capability->dedicated_cpu_compute_available) {
        layer_plan->qtype_compute_refusal_count++;
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_QTYPE, binding,
            layer_index, role, 1ull, 0ull, err, YVEX_ERR_UNSUPPORTED,
            "DeepSeek attention binding qtype lacks admitted CPU row compute");
    }
    if (binding->qtype >= YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_QTYPE, binding,
            layer_index, role, YVEX_RUNTIME_DESCRIPTOR_QTYPE_CAP,
            binding->qtype, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention binding qtype is outside the canonical identity space");
    if (!yvex_core_u64_add(summary->qtype_binding_counts[binding->qtype],
                           1ull, &total))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DIMENSION, binding,
            layer_index, role, ULLONG_MAX,
            summary->qtype_binding_counts[binding->qtype], err,
            YVEX_ERR_BOUNDS,
            "DeepSeek attention qtype binding count overflowed");
    summary->qtype_binding_counts[binding->qtype] = total;
    if (!yvex_core_u64_add(layer_plan->payload_bytes_bound,
                           binding->binding->encoded_bytes, &total))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DIMENSION, binding,
            layer_index, role, ULLONG_MAX,
            binding->binding->encoded_bytes, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention payload byte count overflowed");
    layer_plan->payload_bytes_bound = total;
    return YVEX_OK;
}

enum {
    ATTENTION_BASE_ROLE_COUNT = 8,
    ATTENTION_COMPRESSOR_ROLE_COUNT = 12,
    ATTENTION_REQUIRED_ROLE_COUNT = 18
};

static const yvex_tensor_role attention_required_roles[] = {
    YVEX_TENSOR_ROLE_ATTENTION_SINKS, YVEX_TENSOR_ROLE_ATTENTION_Q_A_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_KV_NORM, YVEX_TENSOR_ROLE_ATTENTION_Q_A,
    YVEX_TENSOR_ROLE_ATTENTION_Q_B, YVEX_TENSOR_ROLE_ATTENTION_KV,
    YVEX_TENSOR_ROLE_ATTENTION_OUT_A, YVEX_TENSOR_ROLE_ATTENTION_OUT_B,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_APE, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_NORM,
    YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_GATE, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_APE, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_NORM,
    YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_GATE, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV,
    YVEX_TENSOR_ROLE_INDEXER_ATTENTION_Q_B, YVEX_TENSOR_ROLE_INDEXER_PROJECTION,
};
_Static_assert(sizeof(attention_required_roles) / sizeof(attention_required_roles[0]) ==
                   ATTENTION_REQUIRED_ROLE_COUNT,
               "attention role table must preserve the complete binding order");

/*
 * Reconstruct an immutable attention plan from authenticated runtime-binding records.
 *
 * Allocates independently owned plan storage and computes its identity. Stale identity chains,
 * malformed layers, or accounting mismatch publish no plan.
 */
int yvex_attention_plan_import(yvex_attention_plan **out,
                               const yvex_attention_summary *summary,
                               const yvex_attention_layer_plan *layers,
                               unsigned long long layer_count,
                               const yvex_materialization_session *session,
                               const yvex_runtime_descriptor *descriptor,
                               yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_materialization_summary *materialization;
    const yvex_runtime_descriptor_summary *runtime;
    yvex_attention_plan *plan;
    char expected_identity[YVEX_ATTENTION_IDENTITY_CAP];
    char computed_identity[YVEX_ATTENTION_IDENTITY_CAP];
    unsigned long long i, required = 0ull, envelope = 0ull, payload = 0ull, refused = 0ull;
    unsigned long long swa = 0ull, csa = 0ull, hca = 0ull;
    int rc;

    if (out) *out = NULL;
    materialization = yvex_materialization_session_summary(session);
    runtime = yvex_runtime_descriptor_summary_get(descriptor);
    if (!out || !summary || !layers || !layer_count || !materialization ||
        !materialization->committed || !runtime ||
        runtime->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY ||
        summary->status != YVEX_ATTENTION_STATUS_EXECUTION_READY ||
        summary->layer_count != layer_count ||
        strcmp(summary->artifact_identity, runtime->artifact_identity) != 0 ||
        strcmp(summary->materialization_plan_identity, materialization->plan_identity) != 0 ||
        strcmp(summary->runtime_descriptor_identity,
               runtime->runtime_descriptor_identity) != 0 ||
        strcmp(summary->runtime_numeric_identity, runtime->runtime_numeric_identity) != 0)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            summary ? summary->layer_count : 0ull, layer_count, err,
            YVEX_ERR_INVALID_ARG, "runtime binding attention records are incomplete or stale");
    rc = attention_plan_allocate(
        &plan, layer_count, "runtime binding attention plan allocation failed",
        "runtime binding attention layer allocation failed", failure, err);
    if (rc != YVEX_OK) return rc;
    plan->summary = *summary;
    memcpy(plan->layers, layers, (size_t)layer_count * sizeof(*plan->layers));
    yvex_core_text_copy(expected_identity, sizeof(expected_identity), summary->attention_plan_identity);
    for (i = 0ull; i < layer_count; ++i) {
        const yvex_attention_layer_plan *layer = &plan->layers[i];
        if (layer->ordinal != i || layer->tensor_scope != summary->tensor_scope ||
            layer->tensor_scope == YVEX_TENSOR_SCOPE_GLOBAL ||
            layer->compute_contract != YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 ||
            !yvex_core_u64_add(required, layer->required_binding_count, &required) ||
            !yvex_core_u64_add(payload, layer->payload_bytes_bound, &payload) ||
            !yvex_core_u64_add(refused, layer->qtype_compute_refusal_count, &refused)) {
            yvex_attention_plan_close(plan);
            return yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL, i,
                YVEX_TENSOR_ROLE_UNKNOWN, i, layer->ordinal, err, YVEX_ERR_FORMAT,
                "runtime binding attention layer record is invalid");
        }
        swa += layer->attention_class == YVEX_ATTENTION_CLASS_SWA;
        csa += layer->attention_class == YVEX_ATTENTION_CLASS_CSA;
        hca += layer->attention_class == YVEX_ATTENTION_CLASS_HCA;
    }
    rc = attention_envelope_binding_count(
        plan, descriptor, &envelope,
        YVEX_ATTENTION_FAILURE_ARCHITECTURE,
        "runtime binding envelope count overflowed", failure, err);
    if (rc != YVEX_OK) {
        yvex_attention_plan_close(plan);
        return rc;
    }
    if (required != summary->required_binding_count || payload != summary->payload_bytes_bound ||
        refused != summary->qtype_compute_refusal_count || swa != summary->swa_layer_count ||
        csa != summary->csa_layer_count || hca != summary->hca_layer_count ||
        envelope != summary->required_envelope_binding_count) {
        yvex_attention_plan_close(plan);
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            summary->required_binding_count, required, err, YVEX_ERR_FORMAT,
            "runtime binding attention summary disagrees with its layers");
    }
    if (!yvex_attention_plan_identity_compute(&plan->summary, plan->layers,
                                              plan->layer_count, computed_identity) ||
        strcmp(computed_identity, expected_identity) != 0) {
        yvex_attention_plan_close(plan);
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_FORMAT, "runtime binding attention identity disagrees with its layers");
    }
    *out = plan;
    return yvex_attention_accept(failure, err);
}

static int attention_bind_required_layer_roles(
    const yvex_runtime_descriptor *descriptor,
    const yvex_attention_layer_plan *layer,
    yvex_attention_layer_plan *layer_plan,
    yvex_attention_summary *summary,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    yvex_tensor_scope scope = layer->tensor_scope;
    unsigned long long layer_index = layer->layer_index;
    unsigned int index;
    int rc;

    for (index = 0u; index < ATTENTION_REQUIRED_ROLE_COUNT; ++index) {
        if (index >= ATTENTION_BASE_ROLE_COUNT &&
            index < ATTENTION_COMPRESSOR_ROLE_COUNT && !layer->compressor_required)
            continue;
        if (index >= ATTENTION_COMPRESSOR_ROLE_COUNT && !layer->indexer_required)
            continue;
        rc = attention_bind_role(descriptor, layer_plan, summary, scope, layer_index,
                                 layer->predictor_index,
                                 attention_required_roles[index], failure, err);
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}

typedef struct {
    const yvex_semantic_model_ir *model;
    const yvex_semantic_attention_layer *layers;
    unsigned long long layer_count;
} semantic_attention_recipe;

static int attention_plan_build(
    yvex_attention_plan **out, const yvex_attention_recipe *recipe,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err);

static int semantic_attention_identity(const void *context, char output[65])
{
    const semantic_attention_recipe *recipe = context;
    const yvex_semantic_model_ir_summary *summary =
        recipe ? yvex_semantic_model_ir_summary_get(recipe->model) : NULL;

    if (!summary || !yvex_sha256_hex_valid(summary->logical_model_identity))
        return 0;
    yvex_core_text_copy(output, 65u, summary->logical_model_identity);
    return 1;
}

static int semantic_attention_layer(
    const void *context, unsigned long long index,
    yvex_attention_layer_plan *output)
{
    const semantic_attention_recipe *recipe = context;
    const yvex_semantic_attention_layer *input;

    if (!recipe || !output || index >= recipe->layer_count) return 0;
    input = &recipe->layers[index];
    *output = (yvex_attention_layer_plan){
        .ordinal = input->ordinal,
        .layer_index = input->layer_index,
        .predictor_index = input->predictor_index,
        .tensor_scope = input->tensor_scope,
        .attention_class = input->attention_class,
        .compute_contract = input->compute_contract,
        .compression_ratio = input->compression_ratio,
        .sliding_window = input->sliding_window,
        .query_heads = input->query_heads,
        .kv_heads = input->kv_heads,
        .head_dimension = input->head_dimension,
        .rope_head_dimension = input->rope_head_dimension,
        .query_lora_rank = input->query_lora_rank,
        .output_lora_rank = input->output_lora_rank,
        .output_groups = input->output_groups,
        .output_group_input_width = input->output_group_input_width,
        .hidden_dimension = input->hidden_dimension,
        .indexer_heads = input->indexer_heads,
        .indexer_head_dimension = input->indexer_head_dimension,
        .indexer_topk = input->indexer_topk,
        .compressor_ape_columns = input->compressor_ape_columns,
        .indexer_ape_columns = input->indexer_ape_columns,
        .rms_norm_epsilon = input->rms_norm_epsilon,
        .residual_stream_count = input->residual_stream_count,
        .residual_stream_width = input->residual_stream_width,
        .residual_expanded_width = input->residual_expanded_width,
        .mhc_mixing_rows = input->mhc_mixing_rows,
        .mhc_mixing_columns = input->mhc_mixing_columns,
        .mhc_base_width = input->mhc_base_width,
        .mhc_scale_width = input->mhc_scale_width,
        .mhc_sinkhorn_iterations = input->mhc_sinkhorn_iterations,
        .attention_input_norm_width = input->attention_input_norm_width,
        .mhc_epsilon = input->mhc_epsilon,
        .mhc_residual_post_multiplier = input->mhc_residual_post_multiplier,
        .mhc_entry_policy = input->mhc_entry_policy,
        .mhc_attention_pre_and_post = input->mhc_attention_pre_and_post,
        .attention_input_norm_required = input->attention_input_norm_required,
        .attention_input_norm_role = input->attention_input_norm_role,
        .mhc_function_role = input->mhc_function_role,
        .mhc_base_role = input->mhc_base_role,
        .mhc_scale_role = input->mhc_scale_role,
        .compressor_required = input->compressor_required,
        .indexer_required = input->indexer_required,
        .position = input->position,
        .attention_kv_activation = input->attention_kv_activation,
        .compressor_activation = input->compressor_activation,
        .compressor_rotated_activation = input->compressor_rotated_activation,
        .indexer_query_activation = input->indexer_query_activation,
        .sparse_topk = input->sparse_topk};
    return 1;
}

int yvex_attention_plan_build_semantic(
    yvex_attention_plan **out, const yvex_semantic_model_ir *semantic_model,
    yvex_tensor_scope tensor_scope,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_semantic_model_ir_summary *semantic =
        yvex_semantic_model_ir_summary_get(semantic_model);
    const yvex_model_execution_descriptor *execution =
        semantic ? &semantic->execution_descriptor : NULL;
    semantic_attention_recipe context = {0};
    yvex_attention_recipe recipe = {0};

    if (out) *out = NULL;
    if (!out || !semantic_model || !session || !descriptor ||
        (tensor_scope != YVEX_TENSOR_SCOPE_MAIN_LAYER &&
         tensor_scope != YVEX_TENSOR_SCOPE_DRAFT))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull,
            err, YVEX_ERR_INVALID_ARG,
            "semantic attention planning requires model, scope, session, and descriptor");
    if (!execution || !yvex_semantic_model_ir_attention_view(
                          semantic_model, tensor_scope, &context.layers,
                          &context.layer_count))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull,
            err, YVEX_ERR_FORMAT,
            "semantic model has no admitted attention topology for this scope");
    context.model = semantic_model;
    recipe = (yvex_attention_recipe){
        .context = &context,
        .layer_count = context.layer_count,
        .auxiliary_layer_count = execution->draft_layer_count,
        .swa_layer_count = tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                               ? context.layer_count : execution->swa_layers,
        .csa_layer_count = tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                               ? 0ull : execution->csa_layers,
        .hca_layer_count = tensor_scope == YVEX_TENSOR_SCOPE_DRAFT
                               ? 0ull : execution->hca_layers,
        .tensor_scope = tensor_scope,
        .identity = semantic_attention_identity,
        .layer = semantic_attention_layer};
    return attention_plan_build(
        out, &recipe, session, descriptor, failure, err);
}

static int attention_plan_build(
    yvex_attention_plan **out,
    const yvex_attention_recipe *recipe,
    const yvex_materialization_session *session,
    const yvex_runtime_descriptor *descriptor,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    const yvex_materialization_summary *materialization;
    const yvex_runtime_descriptor_summary *runtime;
    yvex_attention_plan *plan;
    unsigned long long layer_count;
    unsigned long long i;
    unsigned long long total;
    char logical_identity[YVEX_TRANSFORM_IR_IDENTITY_CAP];
    int rc;

    if (out) *out = NULL;
    if (!out || !recipe || !recipe->context || !recipe->identity ||
        !recipe->layer || !recipe->layer_count || !session || !descriptor)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_INVALID_ARG,
            "DeepSeek attention plan requires IR, materialization session, and descriptor");
    runtime = yvex_runtime_descriptor_summary_get(descriptor);
    materialization = yvex_materialization_session_summary(session);
    if (!materialization || !materialization->committed ||
        materialization->status != YVEX_MATERIALIZATION_STATUS_COMMITTED)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_MATERIALIZATION, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention requires committed materialization");
    if (!runtime || runtime->status != YVEX_RUNTIME_DESCRIPTOR_STATUS_READY)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention requires a ready runtime descriptor");
    if (!runtime->runtime_numeric_identity[0] ||
        runtime->runtime_numeric_schema_version == 0u)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention requires runtime numeric descriptor facts");
    if (!recipe->identity(recipe->context, logical_identity) ||
        strcmp(logical_identity, runtime->logical_model_identity) != 0 ||
        strcmp(materialization->plan_identity,
               runtime->materialization_plan_identity) != 0)
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_DESCRIPTOR, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention refused a stale runtime descriptor identity");
    layer_count = recipe->layer_count;
    rc = attention_plan_allocate(
        &plan, layer_count, "failed to allocate DeepSeek attention plan",
        "failed to allocate DeepSeek attention layer plans", failure, err);
    if (rc != YVEX_OK) return rc;
    attention_summary_initialize(&plan->summary, runtime, recipe);

    for (i = 0ull; i < layer_count; ++i) {
        yvex_attention_layer_plan *layer = &plan->layers[i];
        if (!recipe->layer(recipe->context, i, layer)) {
            yvex_attention_plan_close(plan);
            return yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL,
                i, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
                YVEX_ERR_FORMAT, "DeepSeek attention layer is missing");
        }
        if (layer->ordinal != i || layer->tensor_scope != recipe->tensor_scope ||
            layer->tensor_scope == YVEX_TENSOR_SCOPE_GLOBAL) {
            rc = yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_ARCHITECTURE, NULL,
                layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, i, layer->ordinal,
                err, YVEX_ERR_FORMAT,
                "attention layer scope or ordinal is inconsistent");
            yvex_attention_plan_close(plan);
            return rc;
        }
        if (layer->compute_contract !=
            YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1) {
            rc = yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_NUMERIC, NULL,
                layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
                YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1,
                layer->compute_contract, err, YVEX_ERR_UNSUPPORTED,
                "DeepSeek attention compute contract is unsupported");
            yvex_attention_plan_close(plan);
            return rc;
        }
        rc = attention_bind_required_layer_roles(
            descriptor, layer, layer, &plan->summary, failure, err);
        if (rc != YVEX_OK) {
            yvex_attention_plan_close(plan);
            return rc;
        }
        if (!yvex_core_u64_add(plan->summary.required_binding_count,
                               plan->layers[i].required_binding_count,
                               &total)) {
            rc = yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_DIMENSION, NULL,
                i, YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX,
                plan->layers[i].required_binding_count, err, YVEX_ERR_BOUNDS,
                "DeepSeek attention summary binding count overflowed");
            yvex_attention_plan_close(plan);
            return rc;
        }
        plan->summary.required_binding_count = total;
        if (!yvex_core_u64_add(plan->summary.qtype_compute_refusal_count,
                               plan->layers[i].qtype_compute_refusal_count,
                               &total) ||
            !yvex_core_u64_add(plan->summary.payload_bytes_bound,
                               plan->layers[i].payload_bytes_bound,
                               &plan->summary.payload_bytes_bound)) {
            rc = yvex_attention_reject(
                failure, YVEX_ATTENTION_FAILURE_DIMENSION, NULL,
                i, YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX,
                plan->layers[i].payload_bytes_bound, err, YVEX_ERR_BOUNDS,
                "DeepSeek attention summary accounting overflowed");
            yvex_attention_plan_close(plan);
            return rc;
        }
        plan->summary.qtype_compute_refusal_count = total;
    }
    rc = attention_envelope_binding_count(
        plan, descriptor, &plan->summary.required_envelope_binding_count,
        YVEX_ATTENTION_FAILURE_DIMENSION,
        "attention envelope binding count overflowed", failure, err);
    if (rc != YVEX_OK) {
        yvex_attention_plan_close(plan);
        return rc;
    }
    (void)yvex_attention_plan_identity_compute(
        &plan->summary, plan->layers, plan->layer_count,
        plan->summary.attention_plan_identity);
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_attention_plan_close(yvex_attention_plan *plan)
{
    if (!plan) return;
    free(plan->layers);
    free(plan);
}

const yvex_attention_summary *yvex_attention_plan_summary(
    const yvex_attention_plan *plan)
{
    return plan ? &plan->summary : NULL;
}

unsigned long long yvex_attention_plan_layer_count(
    const yvex_attention_plan *plan)
{
    return plan ? plan->layer_count : 0ull;
}

const yvex_attention_layer_plan *yvex_attention_plan_layer_at(
    const yvex_attention_plan *plan,
    unsigned long long index)
{
    if (!plan || index >= plan->layer_count) return NULL;
    return &plan->layers[index];
}

const yvex_attention_layer_plan *yvex_attention_plan_layer_find(
    const yvex_attention_plan *plan, unsigned long long layer_index)
{
    unsigned long long index;

    if (!plan) return NULL;
    for (index = 0ull; index < plan->layer_count; ++index)
        if (plan->layers[index].layer_index == layer_index)
            return &plan->layers[index];
    return NULL;
}

static const yvex_attention_layer_plan *attention_plan_find_layer(
    const yvex_attention_plan *plan, const yvex_runtime_tensor_binding *binding)
{
    unsigned long long index;

    if (!plan) return NULL;
    for (index = 0ull; index < plan->layer_count; ++index)
        if (plan->layers[index].tensor_scope == binding->scope &&
            plan->layers[index].layer_index == binding->layer_index &&
            plan->layers[index].predictor_index == binding->predictor_index)
            return &plan->layers[index];
    return NULL;
}

yvex_attention_binding_class yvex_attention_plan_binding_classify(
    const yvex_attention_plan *plan, const yvex_runtime_tensor_binding *binding)
{
    const yvex_attention_layer_plan *layer;

    if (!binding || !binding->binding || !plan ||
        binding->scope != plan->summary.tensor_scope)
        return YVEX_ATTENTION_BINDING_NOT_REQUIRED;
    layer = attention_plan_find_layer(plan, binding);
    if (!layer) return YVEX_ATTENTION_BINDING_NOT_REQUIRED;
    for (unsigned int index = 0u; index < ATTENTION_REQUIRED_ROLE_COUNT; ++index) {
        if (binding->role != attention_required_roles[index]) continue;
        if (index < ATTENTION_BASE_ROLE_COUNT ||
            (index < ATTENTION_COMPRESSOR_ROLE_COUNT && layer->compressor_required) ||
            (index >= ATTENTION_COMPRESSOR_ROLE_COUNT && layer->indexer_required))
            return YVEX_ATTENTION_BINDING_CORE;
        return YVEX_ATTENTION_BINDING_NOT_REQUIRED;
    }
    if (layer->attention_input_norm_required &&
        binding->role == layer->attention_input_norm_role)
        return YVEX_ATTENTION_BINDING_ENVELOPE;
    if (layer->mhc_attention_pre_and_post &&
        (binding->role == layer->mhc_function_role ||
         binding->role == layer->mhc_base_role ||
         binding->role == layer->mhc_scale_role))
        return YVEX_ATTENTION_BINDING_ENVELOPE;
    return YVEX_ATTENTION_BINDING_NOT_REQUIRED;
}

static int backend_allowed(const char *name)
{
    return strcmp(name, "cpu") == 0 || strcmp(name, "none") == 0 || strcmp(name, "cuda") == 0;
}

static yvex_backend_kind backend_kind_from_name(const char *name)
{
    return strcmp(name, "cuda") == 0 ? YVEX_BACKEND_KIND_CUDA : YVEX_BACKEND_KIND_CPU;
}

typedef struct {
    size_t plan_offset;
    yvex_backend_operation_variant variants[3];
    size_t variant_count;
} plan_backend_capability;

static const plan_backend_capability plan_backend_capabilities[] = {
    {offsetof(yvex_plan, backend_tensor_alloc),
     {YVEX_BACKEND_VARIANT_TENSOR_ALLOC, YVEX_BACKEND_VARIANT_TENSOR_ZERO}, 2u},
    {offsetof(yvex_plan, backend_tensor_read_write),
     {YVEX_BACKEND_VARIANT_TENSOR_WRITE, YVEX_BACKEND_VARIANT_TENSOR_READ,
      YVEX_BACKEND_VARIANT_TENSOR_COPY}, 3u},
    {offsetof(yvex_plan, backend_op_embed), {YVEX_BACKEND_VARIANT_EMBED_F32_TO_F32}, 1u},
    {offsetof(yvex_plan, backend_op_matmul), {YVEX_BACKEND_VARIANT_MATMUL_F32}, 1u},
    {offsetof(yvex_plan, backend_op_mlp), {YVEX_BACKEND_VARIANT_MLP_DENSE_F32}, 1u},
    {offsetof(yvex_plan, backend_op_rms_norm),
     {YVEX_BACKEND_VARIANT_RMS_NORM_F32_WEIGHT_F32}, 1u},
    {offsetof(yvex_plan, backend_op_rope), {YVEX_BACKEND_VARIANT_ROPE_F32}, 1u},
    {offsetof(yvex_plan, backend_op_attention),
     {YVEX_BACKEND_VARIANT_ATTENTION_CAUSAL_F32}, 1u},
};

static int plan_backend_variant_supported(
    const yvex_backend *backend, yvex_backend_operation_variant variant)
{
    yvex_backend_capability_result capability;
    yvex_error err;
    yvex_error_clear(&err);
    return yvex_backend_query_capability(
               backend, variant, &capability, &err) == YVEX_OK &&
           capability.state == YVEX_BACKEND_CAPABILITY_SUPPORTED;
}

static void plan_backend_capabilities_fill(yvex_plan *plan, const yvex_backend *backend)
{
    size_t capability, variant;

    for (capability = 0u;
         capability < sizeof(plan_backend_capabilities) / sizeof(plan_backend_capabilities[0]);
         ++capability) {
        int supported = 1;
        for (variant = 0u; variant < plan_backend_capabilities[capability].variant_count;
             ++variant)
            supported &= plan_backend_variant_supported(
                backend, plan_backend_capabilities[capability].variants[variant]);
        memcpy((unsigned char *)plan + plan_backend_capabilities[capability].plan_offset,
               &supported, sizeof(supported));
    }
}

static int fill_backend_status(yvex_plan *plan, const char *backend_name, yvex_error *err)
{
    yvex_backend *backend = NULL;
    yvex_backend_options backend_options;
    int rc;

    if (strcmp(backend_name, "cpu") == 0 || strcmp(backend_name, "cuda") == 0) {
        memset(&backend_options, 0, sizeof(backend_options));
        backend_options.kind = backend_kind_from_name(backend_name);
        rc = yvex_backend_open(&backend, &backend_options, err);
        if (rc == YVEX_ERR_UNSUPPORTED && strcmp(backend_name, "cuda") == 0) {
            plan->backend_status = yvex_core_strdup("unavailable");
            yvex_error_clear(err);
            return plan->backend_status ? YVEX_OK : YVEX_ERR_NOMEM;
        }
        if (rc != YVEX_OK) {
            return rc;
        }
        plan->backend_status = yvex_core_strdup(
            yvex_backend_status_of(backend) == YVEX_BACKEND_STATUS_CONTEXT_READY
                ? "context-available" : "available");
        plan_backend_capabilities_fill(plan, backend);
        yvex_backend_close(backend);
    } else {
        plan->backend_status = yvex_core_strdup("not-selected");
    }

    if (!plan->backend_status) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_plan_create", "failed to copy backend status");
        return YVEX_ERR_NOMEM;
    }
    return YVEX_OK;
}

/* Construct plan create with checked geometry and explicit ownership. */
int yvex_plan_create(yvex_plan **out,
                     const yvex_model_descriptor *model,
                     const yvex_tensor_table *tensors,
                     const yvex_plan_options *options,
                     yvex_error *err)
{
    yvex_plan *plan;
    yvex_graph_build_options graph_options;
    const char *backend_name = "cpu";
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_plan_create", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!model || !tensors) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_plan_create", "model and tensors are required");
        return YVEX_ERR_INVALID_ARG;
    }

    memset(&graph_options, 0, sizeof(graph_options));
    graph_options.sequence_length = 1;
    graph_options.context_length = yvex_model_context_length(model) > 0
                                       ? yvex_model_context_length(model)
                                       : 1;
    graph_options.include_prefill_path = 1;

    if (options) {
        if (options->sequence_length > 0) {
            graph_options.sequence_length = options->sequence_length;
        }
        if (options->context_length > 0) {
            graph_options.context_length = options->context_length;
        }
        if (options->backend_name) {
            backend_name = options->backend_name;
        }
    }

    if (!backend_allowed(backend_name)) {
        yvex_error_setf(err, YVEX_ERR_UNSUPPORTED, "yvex_plan_create",
                        "backend label unsupported in graph planner: %s", backend_name);
        return YVEX_ERR_UNSUPPORTED;
    }

    plan = (yvex_plan *)calloc(1, sizeof(*plan));
    if (!plan) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_plan_create", "failed to allocate plan");
        return YVEX_ERR_NOMEM;
    }
    plan->backend_name = yvex_core_strdup(backend_name);
    if (!plan->backend_name) {
        yvex_plan_close(plan);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_plan_create", "failed to copy backend label");
        return YVEX_ERR_NOMEM;
    }
    rc = fill_backend_status(plan, backend_name, err);
    if (rc != YVEX_OK) {
        yvex_plan_close(plan);
        return rc;
    }

    rc = yvex_graph_build_for_model(&plan->graph, model, tensors, &graph_options, err);
    if (rc == YVEX_OK) {
        rc = yvex_memory_plan_from_graph(&plan->memory, plan->graph, tensors, err);
    }
    if (rc != YVEX_OK) {
        yvex_plan_close(plan);
        return rc;
    }

    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_plan_close(yvex_plan *plan)
{
    if (!plan) {
        return;
    }
    free(plan->backend_name);
    free(plan->backend_status);
    yvex_memory_plan_close(plan->memory);
    yvex_graph_close(plan->graph);
    free(plan);
}

const yvex_graph *yvex_plan_graph(const yvex_plan *plan)
{
    return plan ? plan->graph : NULL;
}

const yvex_memory_plan *yvex_plan_memory(const yvex_plan *plan)
{
    return plan ? plan->memory : NULL;
}

static int add_checked(unsigned long long a,
                       unsigned long long b,
                       unsigned long long *out,
                       yvex_error *err)
{
    if (a > ULLONG_MAX - b) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_memory_plan_from_graph", "memory byte total overflow");
        return YVEX_ERR_BOUNDS;
    }
    *out = a + b;
    return YVEX_OK;
}

static int compute_activation_peak(const yvex_graph *graph,
                                   unsigned long long *out,
                                   yvex_error *err)
{
    unsigned long long peak = 0;
    unsigned long long i;

    for (i = 0; i < yvex_graph_value_count(graph); ++i) {
        const yvex_graph_value_info *value = yvex_graph_value_at(graph, i);
        unsigned long long bytes;
        int rc;

        if (!value || value->kind != YVEX_VALUE_ACTIVATION) {
            continue;
        }
        rc = yvex_dtype_tensor_storage_bytes(value->dtype,
                                             value->dims,
                                             value->rank,
                                             &bytes,
                                             err);
        if (rc == YVEX_ERR_UNSUPPORTED) {
            yvex_error_clear(err);
            continue;
        }
        if (rc != YVEX_OK) {
            return rc;
        }
        if (bytes > peak) {
            peak = bytes;
        }
    }

    *out = peak;
    return YVEX_OK;
}

int yvex_memory_plan_from_graph(yvex_memory_plan **out,
                                const yvex_graph *graph,
                                const yvex_tensor_table *tensors,
                                yvex_error *err)
{
    yvex_memory_plan *plan;
    unsigned long long i;
    int rc;

    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_memory_plan_from_graph", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;

    if (!graph || !tensors) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_memory_plan_from_graph",
                       "graph and tensors are required");
        return YVEX_ERR_INVALID_ARG;
    }

    plan = (yvex_memory_plan *)calloc(1, sizeof(*plan));
    if (!plan) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_memory_plan_from_graph",
                       "failed to allocate memory plan");
        return YVEX_ERR_NOMEM;
    }

    for (i = 0; i < yvex_tensor_table_count(tensors); ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(tensors, i);
        if (!tensor) {
            continue;
        }
        if (tensor->storage_bytes == 0) {
            plan->summary.model_tensor_bytes_unknown_count += 1u;
        } else {
            rc = add_checked(plan->summary.model_tensor_bytes_known,
                             tensor->storage_bytes,
                             &plan->summary.model_tensor_bytes_known,
                             err);
            if (rc != YVEX_OK) {
                yvex_memory_plan_close(plan);
                return rc;
            }
        }
    }

    rc = compute_activation_peak(graph, &plan->summary.activation_peak_bytes, err);
    if (rc != YVEX_OK) {
        yvex_memory_plan_close(plan);
        return rc;
    }

    rc = add_checked(plan->summary.model_tensor_bytes_known,
                     plan->summary.activation_peak_bytes,
                     &plan->summary.total_known_bytes,
                     err);
    if (rc != YVEX_OK) {
        yvex_memory_plan_close(plan);
        return rc;
    }
    switch (yvex_graph_status_of(graph)) {
    case YVEX_GRAPH_STATUS_BUILT:
        plan->status = YVEX_MEMORY_PLAN_ESTIMATED;
        break;
    case YVEX_GRAPH_STATUS_PARTIAL:
        plan->status = YVEX_MEMORY_PLAN_PARTIAL;
        break;
    case YVEX_GRAPH_STATUS_UNSUPPORTED:
    case YVEX_GRAPH_STATUS_INVALID:
        plan->status = YVEX_MEMORY_PLAN_UNSUPPORTED;
        break;
    case YVEX_GRAPH_STATUS_EMPTY:
    default:
        plan->status = YVEX_MEMORY_PLAN_EMPTY;
        break;
    }

    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_memory_plan_close(yvex_memory_plan *plan)
{
    free(plan);
}

yvex_memory_plan_status yvex_memory_plan_status_of(const yvex_memory_plan *plan)
{
    return plan ? plan->status : YVEX_MEMORY_PLAN_EMPTY;
}

int yvex_memory_plan_get_summary(const yvex_memory_plan *plan,
                                 yvex_memory_plan_summary *out,
                                 yvex_error *err)
{
    if (!plan || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_memory_plan_get_summary",
                       "plan and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = plan->summary;
    yvex_error_clear(err);
    return YVEX_OK;
}
