/*
 * Centralize allocation, publication, rollback, and evidence ownership for attention execution.
 *
 * Active state is transactional; committed views outlive no owning sink or trace. Bounded
 * execution memory does not establish complete attention or runtime support.
 */
#include "src/graph/private.h"
#include <yvex/internal/graph_state.h>
#include <limits.h>
#include <stdlib.h>
struct yvex_graph_attention_capacity_plan {
    yvex_graph_attention_capacity_summary summary;
    yvex_graph_attention_capacity_request request;
    yvex_graph_attention_capacity_layer *layers;
};

static int capacity_reject(yvex_error *err, yvex_status status, const char *reason) {
    yvex_error_set(err, status, "graph.attention.capacity", reason);
    return status;
}

static int capacity_add(unsigned long long value, unsigned long long *total,
                        unsigned long long *maximum) {
    if (!yvex_core_u64_add(*total, value, total)) return 0;
    if (value > *maximum) *maximum = value;
    return 1;
}

static int capacity_key_seen(const yvex_graph_attention_capacity_plan *plan,
                             unsigned long long count, unsigned long long key) {
    unsigned long long index;
    for (index = 0ull; index < count; ++index)
        if (plan->layers[index].selected &&
            plan->layers[index].recipe.selection_key == key)
            return 1;
    return 0;
}

static int capacity_hash_u64s(yvex_sha256 *hash, const unsigned long long *fields,
                              size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, fields[index])) return 0;
    return 1;
}
/*
 * Hash all immutable selection, extent, and per-layer capacity facts in canonical order.
 *
 * Complete unpublished capacity plan and fixed identity output. Writes a SHA-256 identity over
 * canonical scalar fields only. SHA serialization failure returns false without publishing a
 * partial identity. Never hashes pointers, allocation layout, state bytes, or backend resources.
 */
static int capacity_identity(
    const yvex_graph_attention_capacity_plan *plan,
    char output[YVEX_SHA256_HEX_CAP]) {
    const yvex_graph_attention_capacity_request *request = &plan->request;
    const yvex_graph_attention_capacity_summary *summary = &plan->summary;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    const unsigned long long request_fields[] = {
        summary->schema_version, request->scope,
        request->history_tokens, request->start_position, request->token_count,
        request->execution_count, request->layer_start, request->selection_key,
        (unsigned long long)request->select_layer,
        (unsigned long long)request->select_selection_key,
        (unsigned long long)request->use_requested_position,
        summary->maximum_token_count, summary->selected_binding_count};
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.capacity-plan.v2") ||
        !yvex_sha256_update_text(&hash, summary->attention_plan_identity) ||
        !capacity_hash_u64s(&hash, request_fields,
                            sizeof(request_fields) / sizeof(request_fields[0])))
        return 0;
    for (index = 0ull; index < summary->layer_count; ++index) {
        const yvex_graph_attention_capacity_layer *layer = &plan->layers[index];
        const unsigned long long fields[] = {
            layer->layer_ordinal, (unsigned long long)layer->selected};
        if (!capacity_hash_u64s(&hash, fields, sizeof(fields) / sizeof(fields[0])) ||
            (layer->selected &&
             !yvex_sha256_update_text(&hash, layer->recipe.identity)))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}
/*
 * Derive one immutable capacity plan before state, workspace, descriptor, or backend mutation.
 *
 * Owns one independently queryable per-layer plan with a deterministic identity. Malformed
 * selection, geometry, overflow, or allocation publishes no plan.
 */
int yvex_graph_attention_capacity_plan_build(
    yvex_graph_attention_capacity_plan **out, const yvex_attention_plan *attention,
    const yvex_graph_attention_capacity_request *request, yvex_error *err) {
    const yvex_attention_summary *attention_summary = yvex_attention_plan_summary(attention);
    yvex_graph_attention_capacity_plan *plan;
    unsigned long long bytes, extent, index;
    if (!out || *out || !attention ||
        !attention_summary || !request ||
        attention_summary->status != YVEX_ATTENTION_STATUS_EXECUTION_READY ||
        !yvex_sha256_hex_valid(attention_summary->attention_plan_identity) ||
        !request->token_count || !request->execution_count ||
        request->history_tokens != request->start_position ||
        (unsigned int)request->scope > (unsigned int)YVEX_ATTENTION_PROBE_SCOPE_FULL ||
        (request->select_layer && request->select_selection_key))
        return capacity_reject(err, YVEX_ERR_INVALID_ARG,
                               "complete sealed capacity-plan facts are required");
    if (!yvex_core_u64_mul(request->token_count, request->execution_count, &extent) ||
        !yvex_core_u64_mul(yvex_attention_plan_layer_count(attention),
                           (unsigned long long)sizeof(*plan->layers), &bytes) ||
        bytes > (unsigned long long)SIZE_MAX)
        return capacity_reject(err, YVEX_ERR_BOUNDS,
                               "attention capacity-plan extent overflowed");
    plan = calloc(1u, sizeof(*plan));
    if (!plan)
        return capacity_reject(err, YVEX_ERR_NOMEM,
                               "attention capacity plan allocation failed");
    plan->summary.layer_count = yvex_attention_plan_layer_count(attention);
    plan->layers = calloc((size_t)plan->summary.layer_count, sizeof(*plan->layers));
    if (!plan->layers) {
        free(plan);
        return capacity_reject(err, YVEX_ERR_NOMEM,
                               "attention capacity layer allocation failed");
    }
    plan->request = *request;
    plan->summary.schema_version = YVEX_GRAPH_ATTENTION_CAPACITY_SCHEMA_V1;
    plan->summary.first_layer = ULLONG_MAX;
    plan->summary.maximum_token_count = request->token_count;
    yvex_core_text_copy(plan->summary.attention_plan_identity,
                        sizeof(plan->summary.attention_plan_identity),
                        attention_summary->attention_plan_identity);
    for (index = 0ull; index < plan->summary.layer_count; ++index) {
        const yvex_attention_layer_plan *layer =
            yvex_attention_plan_layer_at(attention, index);
        yvex_graph_attention_capacity_layer *capacity = &plan->layers[index];
        yvex_attention_state_recipe_request recipe_request;
        yvex_attention_state_recipe recipe;
        yvex_attention_failure failure;
        unsigned long long initial = request->start_position, final;
        unsigned int component;
        int rc, selected;
        capacity->layer_ordinal = index;
        if (!layer) {
            yvex_graph_attention_capacity_plan_close(&plan);
            return capacity_reject(err, YVEX_ERR_FORMAT,
                                   "attention capacity layer is malformed");
        }
        if (request->scope == YVEX_ATTENTION_PROBE_SCOPE_QUICK &&
            !request->use_requested_position) {
            rc = yvex_attention_probe_position_resolve(
                layer, 0, request->start_position, &initial, err);
            if (rc != YVEX_OK) {
                yvex_graph_attention_capacity_plan_close(&plan);
                return rc;
            }
        }
        if (!yvex_core_u64_add(initial, extent, &final)) {
            yvex_graph_attention_capacity_plan_close(&plan);
            return capacity_reject(err, YVEX_ERR_BOUNDS,
                                   "attention capacity final position overflowed");
        }
        memset(&recipe_request, 0, sizeof(recipe_request));
        recipe_request.layer_ordinal = index;
        recipe_request.initial_position = initial;
        recipe_request.final_position = final;
        recipe_request.attention_plan_identity =
            attention_summary->attention_plan_identity;
        memset(&recipe, 0, sizeof(recipe));
        memset(&failure, 0, sizeof(failure));
        rc = yvex_attention_state_recipe_build(
            layer, &recipe_request, &recipe, &failure, err);
        if (rc != YVEX_OK ||
            yvex_attention_state_recipe_seal(&recipe, err) != YVEX_OK) {
            yvex_graph_attention_capacity_plan_close(&plan);
            return rc != YVEX_OK ? rc : yvex_error_code(err);
        }
        selected = request->select_layer
                       ? index == request->layer_start
                       : request->select_selection_key
                             ? recipe.selection_key == request->selection_key &&
                                   !plan->summary.selected_layer_count
                             : request->scope == YVEX_ATTENTION_PROBE_SCOPE_FULL ||
                                   !capacity_key_seen(
                                       plan, index, recipe.selection_key);
        capacity->selected = selected;
        if (!selected) continue;
        capacity->recipe = recipe;
        for (component = 0u; component < recipe.component_count; ++component) {
            const yvex_attention_state_component_recipe *item =
                &recipe.components[component];
            yvex_graph_attention_component_capacity *aggregate =
                &plan->summary.components[item->binding];
            unsigned long long extent_count;
            int extent_ok =
                item->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY
                    ? yvex_core_u64_mul(
                          item->capacity, item->value_width, &extent_count)
                    : yvex_core_u64_add(
                          item->rolling.kv_state_extent,
                          item->rolling.score_state_extent, &extent_count);
            if (!extent_ok ||
                !capacity_add(item->capacity, &aggregate->capacity,
                              &aggregate->maximum_capacity) ||
                !capacity_add(extent_count, &aggregate->value_extent,
                              &aggregate->maximum_value_extent)) {
                yvex_graph_attention_capacity_plan_close(&plan);
                return capacity_reject(
                    err, YVEX_ERR_BOUNDS,
                    "attention component capacity aggregate overflowed");
            }
        }
        if (!plan->summary.selected_layer_count)
            plan->summary.first_layer = index;
        plan->summary.selected_layer_count++;
        if (!yvex_core_u64_add(plan->summary.selected_binding_count,
                               layer->required_binding_count,
                               &plan->summary.selected_binding_count)) {
            yvex_graph_attention_capacity_plan_close(&plan);
            return capacity_reject(
                err, YVEX_ERR_BOUNDS,
                "selected attention binding count overflowed");
        }
        if (layer->compression_ratio > plan->summary.maximum_compression_ratio)
            plan->summary.maximum_compression_ratio = layer->compression_ratio;
        if (layer->sparse_topk.k > plan->summary.maximum_topk_capacity)
            plan->summary.maximum_topk_capacity = layer->sparse_topk.k;
    }
    if (!plan->summary.selected_layer_count ||
        !capacity_identity(plan, plan->summary.identity)) {
        yvex_graph_attention_capacity_plan_close(&plan);
        return capacity_reject(
            err, YVEX_ERR_STATE,
            "attention capacity selection or identity is incomplete");
    }
    *out = plan;
    yvex_error_clear(err);
    return YVEX_OK;
}
/* Borrow one sealed capacity summary without transferring plan ownership. */
const yvex_graph_attention_capacity_summary *
yvex_graph_attention_capacity_plan_summary(
    const yvex_graph_attention_capacity_plan *plan) {
    return plan ? &plan->summary : NULL;
}

const yvex_graph_attention_capacity_layer *
yvex_graph_attention_capacity_plan_layer(
    const yvex_graph_attention_capacity_plan *plan,
    unsigned long long layer_ordinal) {
    return plan && layer_ordinal < plan->summary.layer_count
               ? &plan->layers[layer_ordinal] : NULL;
}
/*
 * Release one capacity plan through idempotent pointer ownership.
 *
 * Clears caller ownership and releases the per-layer decisions and plan. Null and already closed
 * ownership are harmless.
 */
void yvex_graph_attention_capacity_plan_close(
    yvex_graph_attention_capacity_plan **plan_ptr) {
    yvex_graph_attention_capacity_plan *plan;
    if (!plan_ptr || !*plan_ptr) return;
    plan = *plan_ptr;
    *plan_ptr = NULL;
    free(plan->layers);
    memset(plan, 0, sizeof(*plan));
    free(plan);
}
static int attention_rolling_view_init(const yvex_attention_layer_plan *layer,
                                       yvex_attention_rolling_kind kind,
                                       unsigned long long next_token_position, float *kv_state,
                                       float *score_state, yvex_attention_rolling_state_view *out);
static const yvex_attention_memory_sink_options memory_sink_defaults = {
    .fail_acquire_kind = YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT,
    .fail_seal_kind = YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT};

static int attention_memory_reject(
    yvex_attention_failure *failure, yvex_attention_failure_code code,
    unsigned long long layer, unsigned long long expected, unsigned long long actual,
    yvex_error *err, yvex_status status, const char *reason)
{
    return yvex_attention_reject(failure, code, NULL, layer, YVEX_TENSOR_ROLE_UNKNOWN,
                                 expected, actual, err, status, reason);
}

static int attention_workspace_component_hash(
    yvex_sha256 *hash, const yvex_attention_workspace_component *component)
{
    return yvex_sha256_update_text(
               hash, "yvex.graph.attention.workspace-component.v1") &&
           yvex_sha256_update_u64(hash, component->schema_version) &&
           yvex_sha256_update_u64(hash, component->ordinal) &&
           yvex_sha256_update_u64(hash, component->kind) &&
           yvex_sha256_update_u64(hash, component->lifetime) &&
           yvex_sha256_update_u64(hash, component->element_count) &&
           yvex_sha256_update_u64(hash, component->element_width) &&
           yvex_sha256_update_u64(hash, component->alignment) &&
           yvex_sha256_update_u64(
               hash, (unsigned long long)component->scales_with_tokens);
}
/*
 * Seal one family-projected workspace recipe before backend lowering.
 *
 * Malformed or unhashable facts return a typed error without trusted identity.
 */
int yvex_attention_workspace_recipe_seal(yvex_attention_workspace_recipe *recipe,
                                         yvex_error *err)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long seen = 0ull;
    unsigned int index;
    if (!recipe ||
        recipe->schema_version != YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1 ||
        !recipe->token_capacity || !recipe->component_count ||
        recipe->component_count > YVEX_ATTENTION_WORKSPACE_COMPONENT_CAP ||
        recipe->mode > YVEX_ATTENTION_EXECUTION_FULL ||
        recipe->scope > YVEX_ATTENTION_OPERATION_RELEASE_SET ||
        recipe->evidence_level > YVEX_ATTENTION_EVIDENCE_FULL ||
        !yvex_sha256_hex_valid(recipe->state_recipe_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.workspace.recipe",
                       "complete bounded workspace recipe facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.workspace-recipe.v1") ||
        !yvex_sha256_update_u64(&hash, recipe->schema_version) ||
        !yvex_sha256_update_u64(&hash, recipe->layer_index) ||
        !yvex_sha256_update_u64(&hash, recipe->mode) ||
        !yvex_sha256_update_u64(&hash, recipe->scope) ||
        !yvex_sha256_update_u64(&hash, recipe->evidence_level) ||
        !yvex_sha256_update_u64(&hash, recipe->token_capacity) ||
        !yvex_sha256_update_u64(&hash, recipe->component_count) ||
        !yvex_sha256_update_text(&hash, recipe->state_recipe_identity))
        goto identity_failure;
    for (index = 0u; index < recipe->component_count; ++index) {
        yvex_attention_workspace_component *component = &recipe->components[index];
        unsigned long long bit;
        if (component->schema_version != YVEX_ATTENTION_WORKSPACE_RECIPE_SCHEMA_V1 ||
            component->ordinal != index ||
            component->kind > YVEX_ATTENTION_WORKSPACE_TOPK_VALID_INDICES ||
            component->lifetime > YVEX_ATTENTION_WORKSPACE_GRAPH_STABLE ||
            !component->element_count || !component->element_width ||
            !component->alignment ||
            (component->alignment & (component->alignment - 1ull)) != 0ull ||
            (component->scales_with_tokens != 0 &&
             component->scales_with_tokens != 1)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.attention.workspace.recipe",
                           "workspace component recipe is malformed");
            return YVEX_ERR_FORMAT;
        }
        bit = 1ull << (unsigned int)component->kind;
        if (seen & bit) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.attention.workspace.recipe",
                           "workspace component kind is duplicated");
            return YVEX_ERR_FORMAT;
        }
        seen |= bit;
        if (!attention_workspace_component_hash(&hash, component))
            goto identity_failure;
    }
    if (!yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, recipe->identity);
    yvex_error_clear(err);
    return YVEX_OK;
identity_failure:
    recipe->identity[0] = '\0';
    yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.workspace.recipe",
                   "workspace recipe identity could not be sealed");
    return YVEX_ERR_STATE;
}
typedef enum {
    ATTENTION_TRACE_INPUT,
    ATTENTION_TRACE_Q_LOW,
    ATTENTION_TRACE_QUERY,
    ATTENTION_TRACE_RAW_KV,
    ATTENTION_TRACE_COMPRESSED_KV,
    ATTENTION_TRACE_INDEXER_KV,
    ATTENTION_TRACE_INDEX_QUERY,
    ATTENTION_TRACE_INDEX_WEIGHTS,
    ATTENTION_TRACE_VALUES,
    ATTENTION_TRACE_OUTPUT,
    ATTENTION_TRACE_ENVELOPE_OUTPUT,
    ATTENTION_TRACE_MAIN_KV,
    ATTENTION_TRACE_MAIN_SCORE,
    ATTENTION_TRACE_INDEX_KV,
    ATTENTION_TRACE_INDEX_SCORE,
    ATTENTION_TRACE_FLOAT_COUNT
} attention_trace_float_field;
typedef enum {
    ATTENTION_TRACE_COMPRESSED_POSITIONS,
    ATTENTION_TRACE_INDEXER_POSITIONS,
    ATTENTION_TRACE_TOPK_COUNTS,
    ATTENTION_TRACE_TOPK_POSITIONS,
    ATTENTION_TRACE_U64_COUNT
} attention_trace_u64_field;
static const size_t attention_trace_float_offsets[] = {
    offsetof(yvex_attention_publication, input),
    offsetof(yvex_attention_publication, q_low),
    offsetof(yvex_attention_publication, query),
    offsetof(yvex_attention_publication, raw_kv),
    offsetof(yvex_attention_publication, compressed_kv),
    offsetof(yvex_attention_publication, indexer_kv),
    offsetof(yvex_attention_publication, index_query),
    offsetof(yvex_attention_publication, index_weights),
    offsetof(yvex_attention_publication, attention_values),
    offsetof(yvex_attention_publication, output),
    offsetof(yvex_attention_publication, envelope_output),
    offsetof(yvex_attention_publication, main_rolling_kv_checkpoints),
    offsetof(yvex_attention_publication, main_rolling_score_checkpoints),
    offsetof(yvex_attention_publication, indexer_rolling_kv_checkpoints),
    offsetof(yvex_attention_publication, indexer_rolling_score_checkpoints),
};
static const size_t attention_trace_u64_offsets[] = {
    offsetof(yvex_attention_publication, compressed_positions),
    offsetof(yvex_attention_publication, indexer_positions),
    offsetof(yvex_attention_publication, topk_counts),
    offsetof(yvex_attention_publication, topk_positions),
};
typedef struct {
    size_t view_offset, output_offset, width;
} attention_rolling_field;
static const attention_rolling_field attention_rolling_fields[] = {
    {offsetof(yvex_attention_rolling_state_view, present),
     offsetof(yvex_attention_rolling_state_output, present), sizeof(int)},
    {offsetof(yvex_attention_rolling_state_view, schema_version),
     offsetof(yvex_attention_rolling_state_output, schema_version), sizeof(unsigned int)},
    {offsetof(yvex_attention_rolling_state_view, kind),
     offsetof(yvex_attention_rolling_state_output, kind), sizeof(yvex_attention_rolling_kind)},
    {offsetof(yvex_attention_rolling_state_view, attention_class),
     offsetof(yvex_attention_rolling_state_output, attention_class), sizeof(yvex_attention_class)},
    {offsetof(yvex_attention_rolling_state_view, layer_index),
     offsetof(yvex_attention_rolling_state_output, layer_index), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, next_token_position),
     offsetof(yvex_attention_rolling_state_output, next_token_position), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, ratio),
     offsetof(yvex_attention_rolling_state_output, ratio), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, head_dimension),
     offsetof(yvex_attention_rolling_state_output, head_dimension), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, state_width),
     offsetof(yvex_attention_rolling_state_output, state_width), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, state_slots),
     offsetof(yvex_attention_rolling_state_output, state_slots), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, previous_fill),
     offsetof(yvex_attention_rolling_state_output, previous_fill), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, current_fill),
     offsetof(yvex_attention_rolling_state_output, current_fill), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, cursor),
     offsetof(yvex_attention_rolling_state_output, cursor), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, kv_state_stride),
     offsetof(yvex_attention_rolling_state_output, kv_state_stride), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, score_state_stride),
     offsetof(yvex_attention_rolling_state_output, score_state_stride), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, kv_state_extent),
     offsetof(yvex_attention_rolling_state_output, kv_state_extent), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, score_state_extent),
     offsetof(yvex_attention_rolling_state_output, score_state_extent), sizeof(unsigned long long)},
    {offsetof(yvex_attention_rolling_state_view, kv_state),
     offsetof(yvex_attention_rolling_state_output, kv_state), sizeof(float *)},
    {offsetof(yvex_attention_rolling_state_view, score_state),
     offsetof(yvex_attention_rolling_state_output, score_state), sizeof(float *)},
    {offsetof(yvex_attention_rolling_state_view, overlap),
     offsetof(yvex_attention_rolling_state_output, overlap), sizeof(int)},
    {offsetof(yvex_attention_rolling_state_view, rotated),
     offsetof(yvex_attention_rolling_state_output, rotated), sizeof(int)},
    {offsetof(yvex_attention_rolling_state_view, attention_plan_identity),
     offsetof(yvex_attention_rolling_state_output, attention_plan_identity),
     YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP},
};

static void attention_rolling_transfer(void *destination, const void *source, int output_to_view)
{
    unsigned int i;
    for (i = 0u; i < sizeof(attention_rolling_fields) / sizeof(attention_rolling_fields[0]); ++i) {
        const attention_rolling_field *field = &attention_rolling_fields[i];
        size_t destination_offset = output_to_view ? field->view_offset : field->output_offset;
        size_t source_offset = output_to_view ? field->output_offset : field->view_offset;
        memcpy((unsigned char *)destination + destination_offset,
               (const unsigned char *)source + source_offset, field->width);
    }
}

static float **attention_trace_float_slot(yvex_attention_publication *trace,
                                          attention_trace_float_field field)
{
    return (float **)(void *)((unsigned char *)trace + attention_trace_float_offsets[field]);
}

static unsigned long long **attention_trace_u64_slot(yvex_attention_publication *trace,
                                                     attention_trace_u64_field field)
{
    return (unsigned long long **)(void *)((unsigned char *)trace +
                                           attention_trace_u64_offsets[field]);
}

static int attention_storage_copy(void **out, unsigned long long count, const void *source,
                                  unsigned long long source_count, size_t width,
                                  yvex_attention_workspace *workspace)
{
    void *copy;
    size_t bytes;
    if (!out || count > source_count || (count && (!source || !width)) ||
        !yvex_attention_checked_size(count, width, &bytes)) return 0;
    *out = NULL;
    if (!count) return 1;
    copy = workspace ? yvex_attention_workspace_calloc(workspace, count, width)
                     : yvex_attention_calloc_array(count, width);
    if (!copy) return 0;
    memcpy(copy, source, bytes);
    *out = copy;
    return 1;
}

static int attention_trace_storage_open(
    yvex_attention_publication *trace,
    const unsigned long long float_counts[ATTENTION_TRACE_FLOAT_COUNT],
    const float *const float_sources[ATTENTION_TRACE_FLOAT_COUNT],
    const unsigned long long u64_counts[ATTENTION_TRACE_U64_COUNT],
    const unsigned long long *const u64_sources[ATTENTION_TRACE_U64_COUNT],
    yvex_attention_workspace *workspace, unsigned long long limit_bytes,
    unsigned long long *owned_bytes)
{
    unsigned long long total = 0ull, bytes;
    unsigned int i;
    for (i = 0u; i < ATTENTION_TRACE_FLOAT_COUNT; ++i)
        if (!yvex_core_u64_mul(float_counts[i], sizeof(float), &bytes) ||
            !yvex_core_u64_add(total, bytes, &total))
            return YVEX_ERR_BOUNDS;
    for (i = 0u; i < ATTENTION_TRACE_U64_COUNT; ++i)
        if (!yvex_core_u64_mul(u64_counts[i], sizeof(unsigned long long), &bytes) ||
            !yvex_core_u64_add(total, bytes, &total))
            return YVEX_ERR_BOUNDS;
    if (total > limit_bytes || total > (unsigned long long)SIZE_MAX)
        return YVEX_ERR_BOUNDS;
    for (i = 0u; i < ATTENTION_TRACE_FLOAT_COUNT; ++i) {
        float **slot = attention_trace_float_slot(trace, (attention_trace_float_field)i);
        if (float_sources && float_counts[i] && !float_sources[i])
            return YVEX_ERR_INVALID_ARG;
        *slot = NULL;
        if (!float_counts[i]) continue;
        *slot = workspace
            ? yvex_attention_workspace_calloc(workspace, float_counts[i], sizeof(**slot))
            : yvex_attention_calloc_array(float_counts[i], sizeof(**slot));
        if (!*slot) return YVEX_ERR_NOMEM;
        if (float_sources)
            memcpy(*slot, float_sources[i], (size_t)float_counts[i] * sizeof(**slot));
    }
    for (i = 0u; i < ATTENTION_TRACE_U64_COUNT; ++i) {
        unsigned long long **slot = attention_trace_u64_slot(trace, (attention_trace_u64_field)i);
        if (u64_sources && u64_counts[i] && !u64_sources[i])
            return YVEX_ERR_INVALID_ARG;
        *slot = NULL;
        if (!u64_counts[i]) continue;
        *slot = workspace
            ? yvex_attention_workspace_calloc(workspace, u64_counts[i], sizeof(**slot))
            : yvex_attention_calloc_array(u64_counts[i], sizeof(**slot));
        if (!*slot) return YVEX_ERR_NOMEM;
        if (u64_sources)
            memcpy(*slot, u64_sources[i], (size_t)u64_counts[i] * sizeof(**slot));
    }
    *owned_bytes = total;
    return YVEX_OK;
}
typedef struct {
    unsigned long long input_width, compressed_capacity, indexer_capacity;
    unsigned long long main_kv_extent, main_score_extent;
    unsigned long long index_kv_extent, index_score_extent, topk_capacity;
    unsigned long long rolling_rows;
} attention_trace_shape;

static int attention_trace_counts(
    const yvex_attention_publication *trace, const attention_trace_shape *shape,
    unsigned long long floats[ATTENTION_TRACE_FLOAT_COUNT],
    unsigned long long integers[ATTENTION_TRACE_U64_COUNT])
{
    const unsigned long long widths[ATTENTION_TRACE_FLOAT_COUNT] = {
        shape->input_width, trace->q_rank, trace->query_width, trace->kv_width,
        trace->compressed_stride, trace->indexer_stride, trace->index_query_stride,
        trace->index_weight_stride, trace->query_width, trace->hidden_width,
        trace->envelope_output_width, 1ull, 1ull, 1ull, 1ull};
    const unsigned long long rows[ATTENTION_TRACE_FLOAT_COUNT] = {
        trace->token_count, trace->token_count, trace->token_count, trace->token_count,
        shape->compressed_capacity, shape->indexer_capacity, trace->token_count,
        trace->token_count, trace->token_count, trace->token_count, trace->token_count,
        shape->main_kv_extent, shape->main_score_extent,
        shape->index_kv_extent, shape->index_score_extent};
    unsigned int i;
    memset(integers, 0, sizeof(*integers) * ATTENTION_TRACE_U64_COUNT);
    for (i = 0u; i < ATTENTION_TRACE_FLOAT_COUNT; ++i) {
        unsigned long long row_count = rows[i];
        if (i >= ATTENTION_TRACE_MAIN_KV &&
            !yvex_core_u64_mul(row_count, shape->rolling_rows, &row_count))
            return 0;
        if (!yvex_core_u64_mul(row_count, widths[i], &floats[i])) return 0;
    }
    integers[ATTENTION_TRACE_COMPRESSED_POSITIONS] = shape->compressed_capacity;
    integers[ATTENTION_TRACE_INDEXER_POSITIONS] = shape->indexer_capacity;
    integers[ATTENTION_TRACE_TOPK_COUNTS] = shape->topk_capacity ? trace->token_count : 0ull;
    return yvex_core_u64_mul(trace->token_count, shape->topk_capacity,
                             &integers[ATTENTION_TRACE_TOPK_POSITIONS]);
}

static int attention_trace_counts_filter(
    yvex_attention_evidence_level level,
    unsigned long long floats[ATTENTION_TRACE_FLOAT_COUNT],
    unsigned long long integers[ATTENTION_TRACE_U64_COUNT])
{
    if ((unsigned int)level > (unsigned int)YVEX_ATTENTION_EVIDENCE_FULL)
        return 0;
    if (level < YVEX_ATTENTION_EVIDENCE_FULL)
        floats[ATTENTION_TRACE_INPUT] = 0ull;
    if (level < YVEX_ATTENTION_EVIDENCE_STAGES) {
        floats[ATTENTION_TRACE_Q_LOW] = 0ull;
        floats[ATTENTION_TRACE_QUERY] = 0ull;
        floats[ATTENTION_TRACE_INDEX_QUERY] = 0ull;
        floats[ATTENTION_TRACE_INDEX_WEIGHTS] = 0ull;
        floats[ATTENTION_TRACE_VALUES] = 0ull;
    }
    if (level < YVEX_ATTENTION_EVIDENCE_FULL) {
        integers[ATTENTION_TRACE_TOPK_COUNTS] = 0ull;
        integers[ATTENTION_TRACE_TOPK_POSITIONS] = 0ull;
    }
    return 1;
}

static int attention_trace_layer_describe(
    yvex_attention_publication *trace, const yvex_attention_layer_plan *layer,
    yvex_attention_operation_scope scope, unsigned long long position,
    unsigned long long tokens, int index_enabled, unsigned long long topk_capacity)
{
    memset(trace, 0, sizeof(*trace));
    trace->layer_index = layer->layer_index;
    trace->attention_class = layer->attention_class;
    trace->token_position = position;
    trace->token_count = tokens;
    trace->hidden_width = trace->core_output_width = layer->hidden_dimension;
    trace->envelope_output_width = scope == YVEX_ATTENTION_OPERATION_ENVELOPE
        ? layer->residual_expanded_width : 0ull;
    trace->q_rank = layer->query_lora_rank;
    trace->kv_width = trace->compressed_stride = layer->head_dimension;
    trace->indexer_stride = layer->indexer_head_dimension;
    trace->index_weight_stride = layer->indexer_heads;
    trace->topk_stride = topk_capacity;
    return yvex_core_u64_mul(layer->query_heads, layer->head_dimension, &trace->query_width) &&
        (!index_enabled || yvex_core_u64_mul(layer->indexer_heads, layer->indexer_head_dimension,
                                             &trace->index_query_stride));
}

static void attention_trace_rolling_bind(yvex_attention_publication *trace,
                                         unsigned long long rows)
{
    if (trace->next_main_rolling_state.present) {
        trace->next_main_rolling_state.kv_state =
            trace->main_rolling_kv_checkpoints +
            (rows - 1ull) * trace->next_main_rolling_state.kv_state_extent;
        trace->next_main_rolling_state.score_state =
            trace->main_rolling_score_checkpoints +
            (rows - 1ull) * trace->next_main_rolling_state.score_state_extent;
    }
    if (trace->next_indexer_rolling_state.present) {
        trace->next_indexer_rolling_state.kv_state =
            trace->indexer_rolling_kv_checkpoints +
            (rows - 1ull) * trace->next_indexer_rolling_state.kv_state_extent;
        trace->next_indexer_rolling_state.score_state =
            trace->indexer_rolling_score_checkpoints +
            (rows - 1ull) * trace->next_indexer_rolling_state.score_state_extent;
    }
}

void yvex_attention_cuda_weights_release(attention_cuda_weights *weights) {
    unsigned int i;
    if (!weights)
        return;
    for (i = 0u; i < YVEX_BACKEND_ATTENTION_WEIGHT_COUNT; ++i) {
        free(weights->owned[i]);
        weights->owned[i] = NULL;
    }
    weights->payload_bytes_read = 0ull;
}

static int attention_cuda_load_weight(yvex_materialization_session *session,
                                      const yvex_runtime_tensor_binding *runtime_binding,
                                      yvex_backend_attention_weight_slot slot,
                                      attention_cuda_weights *owned,
                                      yvex_backend_attention_job *job,
                                      yvex_attention_failure *failure, yvex_error *err) {
    const yvex_materialized_tensor_binding *binding;
    yvex_backend_attention_weight *weight;
    unsigned long long blocks;
    unsigned long long row_bytes;
    unsigned long long expected;
    const unsigned char *resident = NULL;
    yvex_materialization_failure materialization_failure;
    int rc;
    if (!session || !runtime_binding || !runtime_binding->binding || !owned || !job ||
        slot >= YVEX_BACKEND_ATTENTION_WEIGHT_COUNT)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, runtime_binding,
            YVEX_ATTENTION_NO_LAYER,
            runtime_binding ? runtime_binding->role : YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG, "CUDA attention weight load requires a typed binding and slot");
    binding = runtime_binding->binding;
    if (!binding->row_width || !binding->row_count || !binding->block_size ||
        !binding->bytes_per_block || binding->row_width % binding->block_size != 0ull ||
        binding->encoded_bytes > (unsigned long long)SIZE_MAX)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role,
                                     binding->block_size, binding->row_width, err, YVEX_ERR_BOUNDS,
                                     "CUDA attention encoded tensor geometry is invalid");
    blocks = binding->row_width / binding->block_size;
    if (!yvex_core_u64_mul(blocks, binding->bytes_per_block, &row_bytes) ||
        !yvex_core_u64_mul(row_bytes, binding->row_count, &expected) ||
        expected != binding->encoded_bytes)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role,
                                     binding->encoded_bytes, expected, err, YVEX_ERR_FORMAT,
                                     "CUDA attention encoded tensor range is not row-exact");
    memset(&materialization_failure, 0, sizeof(materialization_failure));
    rc = yvex_materialization_session_borrow(
        session, binding, 0ull, (size_t)binding->encoded_bytes, &resident,
        &materialization_failure, err);
    if (rc == YVEX_ERR_INVALID_ARG) {
        yvex_error_clear(err);
        owned->owned[slot] = (unsigned char *)malloc((size_t)binding->encoded_bytes);
        if (!owned->owned[slot])
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, runtime_binding,
                binding->layer_index, binding->role, binding->encoded_bytes, 0ull, err,
                YVEX_ERR_NOMEM, "CUDA attention encoded weight allocation failed");
        rc = yvex_materialization_session_read(
            session, binding, 0ull, owned->owned[slot], (size_t)binding->encoded_bytes,
            NULL, err);
        resident = owned->owned[slot];
        if (rc == YVEX_OK &&
            !yvex_core_u64_add(owned->payload_bytes_read, binding->encoded_bytes,
                               &owned->payload_bytes_read))
            rc = YVEX_ERR_BOUNDS;
    }
    if (rc != YVEX_OK)
        return yvex_attention_reject(
            failure, rc == YVEX_ERR_BOUNDS ? YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION
                                           : YVEX_DEEPSEEK_ATTENTION_FAILURE_READ,
            runtime_binding, binding->layer_index, binding->role, binding->encoded_bytes,
            0ull, err, (yvex_status)rc,
            rc == YVEX_ERR_BOUNDS ? "CUDA attention payload-byte accounting overflowed"
                                  : "CUDA attention failed to borrow resident encoded weight");
    weight = &job->weights[slot];
    weight->encoded = resident;
    weight->encoded_bytes = (size_t)binding->encoded_bytes;
    weight->row_bytes = row_bytes;
    weight->row_width = binding->row_width;
    weight->row_count = binding->row_count;
    weight->qtype = binding->qtype;
    weight->present = 1;
    return YVEX_OK;
}

int yvex_attention_cuda_role_load(yvex_materialization_session *session,
                                  const yvex_runtime_descriptor *descriptor,
                                  const yvex_attention_layer_plan *layer, yvex_tensor_role role,
                                  yvex_backend_attention_weight_slot slot,
                                  attention_cuda_weights *owned, yvex_backend_attention_job *job,
                                  yvex_attention_failure *failure, yvex_error *err) {
    const yvex_runtime_tensor_binding *binding =
        yvex_attention_binding_find(descriptor, role, layer);
    if (!binding)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_MISSING_BINDING, NULL,
                                     layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
                                     role, 1ull, 0ull, err, YVEX_ERR_FORMAT,
                                     "CUDA attention required typed role binding is absent");
    return attention_cuda_load_weight(session, binding, slot, owned, job, failure, err);
}

void yvex_attention_cuda_activation_project(const yvex_attention_activation_policy *source,
                                            yvex_backend_attention_activation *out) {
    memset(out, 0, sizeof(*out));
    if (!source || !source->required)
        return;
    out->required = 1;
    out->block_width = source->block_width;
    out->quantization = (unsigned int)source->quantization;
    out->hadamard = source->pre_transform == YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2;
}

int yvex_attention_cuda_rolling_project(const yvex_attention_rolling_state_view *source,
                                        yvex_backend_attention_rolling *out) {
    if (!source || !source->present || !out)
        return 0;
    *out = (yvex_backend_attention_rolling){
        .present = 1,
        .next_token_position = source->next_token_position,
        .ratio = source->ratio,
        .head_dimension = source->head_dimension,
        .state_width = source->state_width,
        .state_slots = source->state_slots,
        .cursor = source->cursor,
        .previous_fill = source->previous_fill,
        .current_fill = source->current_fill,
        .kv_state = source->kv_state,
        .kv_state_capacity = source->kv_state_extent,
        .score_state = source->score_state,
        .score_state_capacity = source->score_state_extent,
        .overlap = source->overlap,
    };
    return 1;
}

static void cuda_rolling_commit(const yvex_attention_rolling_state_view *before,
                                unsigned long long token_position,
                                unsigned long long token_count,
                                yvex_attention_rolling_state_output *after) {
    float *kv_state;
    float *score_state;
    unsigned long long cursor, previous_fill, current_fill, token;
    if (!before || !before->present || !after || !token_count)
        return;
    kv_state = after->kv_state;
    score_state = after->score_state;
    attention_rolling_transfer(after, before, 0);
    after->kv_state = kv_state;
    after->score_state = score_state;
    cursor = before->cursor;
    previous_fill = before->previous_fill;
    current_fill = before->current_fill;
    for (token = 0ull; token < token_count; ++token) {
        int emitted = ((token_position + token + 1ull) % before->ratio) == 0ull;
        previous_fill = emitted ? (before->overlap ? before->ratio : 0ull) : previous_fill;
        current_fill = emitted ? 0ull : (current_fill < cursor + 1ull ? cursor + 1ull : current_fill);
        cursor = emitted ? 0ull : (cursor + 1ull) % before->ratio;
    }
    after->next_token_position = token_position + token_count;
    after->previous_fill = previous_fill;
    after->current_fill = current_fill;
    after->cursor = cursor;
    after->kv_state_stride = before->state_width;
    after->score_state_stride = before->state_width;
    if (!yvex_core_u64_mul(before->state_width, before->state_slots, &after->kv_state_extent))
        after->kv_state_extent = 0ull;
    after->score_state_extent = after->kv_state_extent;
}

static int cuda_output_identity(const yvex_attention_plan *plan,
                                const yvex_attention_execution_trace *trace,
                                char out[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    const yvex_attention_summary *summary = yvex_attention_plan_summary(plan);
    const float *values;
    unsigned long long count;
    unsigned long long width;
    unsigned long long envelope;
    size_t bytes;
    envelope = trace && trace->envelope_output_width ? 1ull : 0ull;
    values = envelope ? trace->envelope_output : (trace ? trace->output : NULL);
    width = envelope ? trace->envelope_output_width : (trace ? trace->hidden_width : 0ull);
    if (!summary || !trace || !trace->complete || !out || !values || !width ||
        !yvex_core_u64_mul(trace->token_count, width, &count) ||
        !yvex_attention_checked_size(count, sizeof(float), &bytes))
        return 0;
    yvex_sha256_init(&hash);
    if (!attention_hash_text(&hash, "yvex.deepseek.attention.cuda.output.v2") ||
        !attention_hash_text(&hash, summary->attention_plan_identity) ||
        !attention_hash_u64(&hash, trace->layer_index) ||
        !attention_hash_u64(&hash, trace->token_position) ||
        !attention_hash_u64(&hash, trace->token_count) ||
        !attention_hash_u64(&hash, envelope) ||
        !attention_hash_u64(&hash, width) ||
        !yvex_sha256_update(&hash, values, bytes) || !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, out);
    return 1;
}

int yvex_attention_cuda_reject(
    attention_cuda_context *context, yvex_attention_failure_code code,
    unsigned long long expected, unsigned long long actual,
    yvex_status status, const char *reason)
{
    return yvex_attention_reject(
        context->failure, code, NULL,
        context->layer ? context->layer->layer_index : context->opts->layer_index,
        YVEX_TENSOR_ROLE_UNKNOWN, expected, actual, context->err, status, reason);
}

static int attention_cuda_memory_facts(attention_cuda_context *context)
{
    const yvex_backend_attention_job *job = &context->job;
    unsigned long long state_elements = 0ull, state_positions, activation_elements;
    unsigned long long value, state_bytes, position_bytes, activation_bytes;
    const yvex_backend_attention_rolling *rolling[] = {
        &job->main_rolling, &job->indexer_rolling};
    unsigned int index;
    if (!job->device_input || !job->device_output)
        return yvex_execution_memory_facts_add(
                   &context->result->memory, 0ull, 0ull, 0ull, 0ull,
                   0ull, 1ull, context->err) == YVEX_OK;
#define ADD_STATE(count_, stride_)                                                \
    do {                                                                          \
        if (!yvex_core_u64_mul((count_), (stride_), &value) ||                    \
            !yvex_core_u64_add(state_elements, value, &state_elements)) return 0; \
    } while (0)
    ADD_STATE(job->local_count, job->local_stride);
    ADD_STATE(job->compressed_count, job->compressed_stride);
    ADD_STATE(job->indexer_count, job->indexer_stride);
    for (index = 0u; index < sizeof(rolling) / sizeof(rolling[0]); ++index)
        if (rolling[index]->present &&
            (!yvex_core_u64_add(state_elements, rolling[index]->kv_state_capacity,
                                &state_elements) ||
             !yvex_core_u64_add(state_elements, rolling[index]->score_state_capacity,
                                &state_elements))) return 0;
#undef ADD_STATE
    if (!yvex_core_u64_add(job->local_count, job->compressed_count, &state_positions) ||
        !yvex_core_u64_add(state_positions, job->indexer_count, &state_positions) ||
        !yvex_core_u64_mul(state_elements, sizeof(float), &state_bytes) ||
        !yvex_core_u64_mul(state_positions, sizeof(unsigned long long), &position_bytes) ||
        !yvex_core_u64_add(state_bytes, position_bytes, &state_bytes) ||
        !yvex_core_u64_mul(job->token_count,
                           job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                               ? job->residual_expanded_width : job->hidden_width,
                           &activation_elements) ||
        !yvex_core_u64_mul(activation_elements, 2ull, &activation_elements) ||
        !yvex_core_u64_mul(activation_elements, sizeof(float), &activation_bytes)) return 0;
    return yvex_execution_memory_facts_add(
               &context->result->memory, context->weights.payload_bytes_read,
               state_bytes, activation_bytes, context->cuda_output.peak_device_bytes,
               1ull, 0ull, context->err) == YVEX_OK;
}
/*
 * Publish CUDA evidence and rolling state after complete device execution.
 *
 * Accounting or identity refusal publishes no candidate state.
 */
int yvex_attention_cuda_publish(attention_cuda_context *context)
{
    unsigned long long q_low, query, raw, output, envelope;
    unsigned long long token;
    if (context->cuda_output.tokens_executed != context->token_count ||
        !yvex_core_u64_mul(context->token_count, context->trace.q_rank, &q_low) ||
        !yvex_core_u64_mul(context->token_count, context->trace.query_width, &query) ||
        !yvex_core_u64_mul(context->token_count, context->trace.kv_width, &raw) ||
        !yvex_core_u64_mul(context->token_count, context->trace.hidden_width, &output) ||
        !yvex_core_u64_mul(context->token_count, context->trace.envelope_output_width,
                           &envelope))
        return yvex_attention_cuda_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND,
            context->token_count, context->cuda_output.tokens_executed,
            YVEX_ERR_STATE,
            "CUDA attention backend did not complete the admitted token range");
    if (!attention_cuda_memory_facts(context))
        return yvex_attention_cuda_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
            ULLONG_MAX, context->token_count, YVEX_ERR_BOUNDS,
            "CUDA attention compulsory memory extent overflowed");
    context->trace.compressed_count = context->cuda_output.compressed_count;
    context->trace.indexer_count = context->cuda_output.indexer_count;
    context->trace.device_state_staged = context->cuda_output.device_state_staged;
    context->trace.device_state_staged_bytes = context->cuda_output.device_state_staged_bytes;
    context->trace.compressed_stride = context->trace.compressed_count
                                           ? context->layer->head_dimension : 0ull;
    context->trace.indexer_stride = context->trace.indexer_count
                                        ? context->layer->indexer_head_dimension : 0ull;
    if (context->layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        unsigned long long storage_stride = context->trace.topk_stride;
        unsigned long long semantic_stride = context->cuda_output.topk_count;
        if (semantic_stride > storage_stride)
            return yvex_attention_cuda_reject(
                context, YVEX_DEEPSEEK_ATTENTION_FAILURE_BACKEND,
                storage_stride, semantic_stride, YVEX_ERR_BOUNDS,
                "CUDA attention top-k publication exceeded its storage stride");
        if (context->trace.topk_positions && semantic_stride < storage_stride)
            for (token = 1ull; token < context->token_count; ++token)
                memmove(context->trace.topk_positions + token * semantic_stride,
                        context->trace.topk_positions + token * storage_stride,
                        (size_t)semantic_stride * sizeof(*context->trace.topk_positions));
        context->trace.topk_stride = semantic_stride;
    }
    context->trace.complete = 1;
    context->result->executed = 1;
    context->result->full_attention = 1;
    context->result->cuda_executed = 1;
    context->result->operation_scope = context->opts->operation_scope;
    context->result->layer_index = context->layer->layer_index;
    context->result->attention_class = context->layer->attention_class;
    context->result->token_position = context->opts->token_position;
    context->result->q_a_rows = context->layer->query_lora_rank;
    context->result->q_b_rows = context->layer->query_heads * context->layer->head_dimension;
    context->result->kv_rows = context->layer->head_dimension;
    context->result->topk_candidates = context->cuda_output.valid_candidate_count;
    context->result->topk_selected = context->cuda_output.topk_count;
    context->result->local_entries =
        context->history->local_tail_count + context->token_count;
    context->result->compressed_entries = context->cuda_output.compressed_count;
    context->result->payload_bytes_read = context->weights.payload_bytes_read;
    context->result->state_raw_entries = context->token_count;
    context->result->state_compressed_entries = context->cuda_output.compressed_count;
    context->result->state_indexer_entries = context->cuda_output.indexer_count;
    context->result->cuda_kernel_launches = context->cuda_output.kernel_launches;
    context->result->cuda_h2d_bytes = context->cuda_output.h2d_bytes;
    context->result->cuda_d2h_bytes = context->cuda_output.d2h_bytes;
    context->result->cuda_d2d_bytes = context->cuda_output.d2d_bytes;
    context->result->cuda_stream_synchronizations =
        context->cuda_output.stream_synchronizations;
    context->result->cuda_device_synchronizations =
        context->cuda_output.device_synchronizations;
    context->result->cuda_device_execution_elapsed_ns =
        context->cuda_output.device_execution_elapsed_ns;
    if (!yvex_core_u64_add(context->trace_bytes,
                           context->cuda_output.peak_host_bytes,
                           &context->result->cuda_peak_host_bytes))
        return yvex_attention_cuda_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
            context->opts->scratch_limit_bytes, ULLONG_MAX, YVEX_ERR_BOUNDS,
            "CUDA attention peak host accounting overflowed");
    context->result->cuda_peak_device_bytes = context->cuda_output.peak_device_bytes;
    context->result->cuda_host_workspace_capacity =
        context->cuda_output.host_workspace_capacity;
    context->result->cuda_host_workspace_used = context->cuda_output.host_workspace_used;
    context->result->cuda_host_workspace_peak = context->cuda_output.host_workspace_peak;
    context->result->cuda_host_workspace_allocations =
        context->cuda_output.host_workspace_allocation_count;
    context->result->cuda_host_workspace_reused = context->cuda_output.host_workspace_reused;
    context->result->q_projection_checksum = context->trace.q_low
        ? yvex_attention_checksum(context->trace.q_low, q_low) : 0.0;
    context->result->kv_projection_checksum =
        yvex_attention_checksum(context->trace.raw_kv, raw);
    context->result->rope_checksum = context->trace.query
        ? yvex_attention_checksum(context->trace.query, query) : 0.0;
    context->result->attention_checksum = context->trace.attention_values
        ? yvex_attention_checksum(context->trace.attention_values, query) : 0.0;
    context->result->core_output_checksum = context->trace.evidence_level
        ? yvex_attention_checksum(context->trace.output, output) : 0.0;
    context->result->envelope_output_checksum = context->trace.evidence_level
        ? yvex_attention_checksum(context->trace.envelope_output, envelope) : 0.0;
    context->result->output_checksum =
        context->opts->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
            ? context->result->envelope_output_checksum
            : context->result->core_output_checksum;
    if (context->trace.evidence_level && !cuda_output_identity(
            context->plan, &context->trace, context->result->output_identity))
        return yvex_attention_cuda_reject(
            context, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
            1ull, 0ull, YVEX_ERR_STATE,
            "CUDA attention output identity construction failed");
    if (context->history->main_rolling_state.present)
        cuda_rolling_commit(
            &context->history->main_rolling_state, context->opts->token_position,
            context->token_count, &context->trace.next_main_rolling_state);
    if (context->history->indexer_rolling_state.present)
        cuda_rolling_commit(
            &context->history->indexer_rolling_state, context->opts->token_position,
            context->token_count, &context->trace.next_indexer_rolling_state);
    if (context->opts->publication || context->opts->trace) {
        yvex_attention_publication *publication = context->opts->publication
            ? context->opts->publication : context->opts->trace;
        *publication = context->trace;
        memset(&context->trace, 0, sizeof(context->trace));
    }
    if (context->failure) memset(context->failure, 0, sizeof(*context->failure));
    yvex_error_clear(context->err);
    return YVEX_OK;
}
/*
 * Allocate one complete multi-token CUDA publication before dispatch.
 *
 * Owns bounded storage for every device-produced trace component. Overflow or allocation failure
 * releases all partial storage.
 */
int yvex_attention_cuda_trace_open(yvex_attention_publication *trace,
                                   const yvex_attention_layer_plan *layer,
                                   yvex_attention_operation_scope scope,
                                   const yvex_attention_history_view *history,
                                   unsigned long long token_position,
                                   unsigned long long token_count,
                                   yvex_attention_evidence_level evidence_level,
                                   int retain_prefix_checkpoints,
                                   yvex_attention_workspace *workspace,
                                   unsigned long long limit_bytes,
                                   unsigned long long *owned_bytes,
                                   yvex_attention_failure *failure, yvex_error *err)
{
    unsigned long long floats[ATTENTION_TRACE_FLOAT_COUNT] = {0ull};
    unsigned long long integers[ATTENTION_TRACE_U64_COUNT] = {0ull};
    unsigned long long end, compressed = 0ull, indexer = 0ull;
    unsigned long long main_extent = 0ull, index_extent = 0ull, topk = 0ull;
    attention_trace_shape shape;
    unsigned long long input_width;
    if (!trace || !layer || !history || !owned_bytes || !token_count ||
        !yvex_core_u64_add(token_position, token_count, &end))
        goto invalid;
    input_width = scope == YVEX_ATTENTION_OPERATION_ENVELOPE
        ? layer->residual_expanded_width : layer->hidden_dimension;
    *owned_bytes = 0ull;
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA) {
        compressed = end / layer->compression_ratio - token_position / layer->compression_ratio;
        if (!yvex_core_u64_mul(history->main_rolling_state.state_width,
                               history->main_rolling_state.state_slots, &main_extent))
            goto fail;
    }
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA) {
        indexer = compressed;
        topk = layer->sparse_topk.k;
        if (!yvex_core_u64_mul(history->indexer_rolling_state.state_width,
                               history->indexer_rolling_state.state_slots, &index_extent))
            goto fail;
    }
    if (!attention_trace_layer_describe(
            trace, layer, scope, token_position, token_count, 1,
            topk))
        goto fail;
    trace->evidence_level = (unsigned int)evidence_level;
    trace->prefix_addressable = retain_prefix_checkpoints;
    trace->rolling_checkpoint_count = retain_prefix_checkpoints && main_extent
                                          ? token_count : 0ull;
    trace->workspace = workspace;
    shape = (attention_trace_shape){input_width, compressed, indexer, main_extent, main_extent,
                                    index_extent, index_extent, trace->topk_stride,
                                    trace->rolling_checkpoint_count
                                        ? trace->rolling_checkpoint_count : 1ull};
    if (!attention_trace_counts(trace, &shape, floats, integers) ||
        !attention_trace_counts_filter(evidence_level, floats, integers))
        goto fail;
    {
        int rc = attention_trace_storage_open(trace, floats, NULL, integers, NULL, workspace,
                                              limit_bytes, owned_bytes);
        if (rc != YVEX_OK) {
            yvex_attention_execution_trace_release(trace);
            return attention_memory_reject(
                failure, rc == YVEX_ERR_NOMEM ? YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION
                                              : YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
                layer->layer_index, limit_bytes, *owned_bytes, err, (yvex_status)rc,
                rc == YVEX_ERR_NOMEM ? "CUDA attention trace allocation failed"
                                     : "CUDA attention trace exceeds its host budget");
        }
    }
    trace->owned = 1;
    trace->core_output = trace->output;
    if (history->main_rolling_state.present)
        attention_rolling_transfer(&trace->next_main_rolling_state,
                                   &history->main_rolling_state, 0);
    if (history->indexer_rolling_state.present)
        attention_rolling_transfer(&trace->next_indexer_rolling_state,
                                   &history->indexer_rolling_state, 0);
    attention_trace_rolling_bind(
        trace, trace->rolling_checkpoint_count
                   ? trace->rolling_checkpoint_count : 1ull);
    return YVEX_OK;
invalid:
    return attention_memory_reject(
        failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
        layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, 0ull, err,
        YVEX_ERR_INVALID_ARG, "CUDA attention trace requires history and a bounded token range");
fail:
    yvex_attention_execution_trace_release(trace);
    return attention_memory_reject(
        failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
        layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, limit_bytes, 0ull, err,
        YVEX_ERR_BOUNDS, "CUDA attention trace geometry exceeds its host budget");
}

int yvex_attention_rolling_storage_acquire(const yvex_attention_layer_plan *layer,
                                           yvex_attention_rolling_kind kind,
                                           unsigned long long token_position,
                                           yvex_attention_workspace *workspace,
                                           float **kv_state_out, float **score_state_out,
                                           yvex_attention_rolling_state_view *view_out,
                                           yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long ratio;
    unsigned long long head_dim;
    unsigned long long state_width;
    unsigned long long state_slots;
    unsigned long long extent;
    float *kv_state = NULL;
    float *score_state = NULL;
    yvex_attention_rolling_state_view view;
    int overlap;
    int rotated;
    int rc;
    unsigned long long mark = workspace ? yvex_attention_workspace_mark(workspace) : 0ull;
    if (kv_state_out)
        *kv_state_out = NULL;
    if (score_state_out)
        *score_state_out = NULL;
    if (view_out)
        memset(view_out, 0, sizeof(*view_out));
    if (!layer || !kv_state_out || !score_state_out || !view_out ||
        !yvex_attention_rolling_geometry(layer, kind, &ratio, &head_dim, &state_width, &state_slots,
                                         &overlap, &rotated) ||
        !yvex_core_u64_mul(state_slots, state_width, &extent))
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY,
            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention could not derive empty rolling-state geometry");
    kv_state = workspace
        ? (float *)yvex_attention_workspace_calloc(workspace, extent, sizeof(float))
        : (float *)yvex_attention_calloc_array(extent, sizeof(float));
    score_state = workspace
        ? (float *)yvex_attention_workspace_calloc(workspace, extent, sizeof(float))
        : (float *)yvex_attention_calloc_array(extent, sizeof(float));
    if (!kv_state || !score_state) {
        rc = attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, layer->layer_index,
            extent, 0ull, err, YVEX_ERR_NOMEM,
            "DeepSeek attention empty rolling-state allocation failed");
        goto fail;
    }
    if (!attention_rolling_view_init(layer, kind, token_position, kv_state, score_state, &view)) {
        rc = attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, layer->layer_index,
            extent, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention empty rolling-state initialization failed");
        goto fail;
    }
    *kv_state_out = kv_state;
    *score_state_out = score_state;
    *view_out = view;
    return YVEX_OK;
fail:
    if (workspace) {
        yvex_error rewind_error;
        yvex_error_clear(&rewind_error);
        (void)yvex_attention_workspace_rewind(workspace, mark, &rewind_error);
    } else {
        free(kv_state);
        free(score_state);
    }
    return rc;
}

static int attention_rolling_view_init(const yvex_attention_layer_plan *layer,
                                       yvex_attention_rolling_kind kind,
                                       unsigned long long next_token_position, float *kv_state,
                                       float *score_state, yvex_attention_rolling_state_view *out) {
    unsigned long long ratio;
    unsigned long long head_dim;
    unsigned long long state_width;
    unsigned long long state_slots;
    unsigned long long extent;
    unsigned long long i;
    int overlap;
    int rotated;
    if (!layer || !kv_state || !score_state || !out ||
        !yvex_attention_rolling_geometry(layer, kind, &ratio, &head_dim, &state_width, &state_slots,
                                         &overlap, &rotated) ||
        !yvex_core_u64_mul(state_slots, state_width, &extent))
        return 0;
    for (i = 0ull; i < extent; ++i) {
        kv_state[i] = 0.0f;
        score_state[i] = -INFINITY;
    }
    memset(out, 0, sizeof(*out));
    out->present = 1;
    out->schema_version = YVEX_DEEPSEEK_ATTENTION_ROLLING_STATE_SCHEMA_V1;
    out->kind = kind;
    out->attention_class = layer->attention_class;
    out->layer_index = layer->layer_index;
    out->next_token_position = next_token_position;
    out->ratio = ratio;
    out->head_dimension = head_dim;
    out->state_width = state_width;
    out->state_slots = state_slots;
    out->cursor = next_token_position % ratio;
    out->kv_state_stride = state_width;
    out->score_state_stride = state_width;
    out->kv_state_extent = extent;
    out->score_state_extent = extent;
    out->kv_state = kv_state;
    out->score_state = score_state;
    out->overlap = overlap;
    out->rotated = rotated;
    return 1;
}

void yvex_attention_rolling_output_bind(const yvex_attention_rolling_state_output *out,
                                        yvex_attention_rolling_state_view *view) {
    memset(view, 0, sizeof(*view));
    attention_rolling_transfer(view, out, 1);
}

void yvex_attention_execution_trace_release(yvex_attention_execution_trace *trace) {
    yvex_attention_workspace *workspace;
    unsigned int i;
    if (!trace)
        return;
    workspace = trace->workspace;
    if (!workspace) {
        for (i = 0u; i < ATTENTION_TRACE_FLOAT_COUNT; ++i)
            free(*attention_trace_float_slot(trace, (attention_trace_float_field)i));
        for (i = 0u; i < ATTENTION_TRACE_U64_COUNT; ++i)
            free(*attention_trace_u64_slot(trace, (attention_trace_u64_field)i));
    }
    memset(trace, 0, sizeof(*trace));
}
/*
 * Distinguish committed core and envelope output in one owned trace.
 *
 * Evidence ownership only; no attention or residual arithmetic is performed here.
 */
int yvex_attention_trace_outputs_attach(yvex_attention_execution_trace *trace,
                                        const yvex_attention_component_span *core,
                                        const yvex_attention_component_span *envelope)
{
    unsigned long long envelope_width = envelope ? envelope->dims[1] : 0ull;
    if (!trace || !trace->owned || !trace->complete || !trace->output || !core || !core->data ||
        core->dims[1] != trace->hidden_width ||
        (envelope && (!envelope->data || !envelope_width)))
        return 0;
    trace->core_output = trace->output;
    trace->core_output_width = core->dims[1];
    if (!envelope) return 1;
    if (!attention_storage_copy((void **)&trace->envelope_output,
                                envelope->expected_elements, envelope->data,
                                envelope->expected_elements, sizeof(float), trace->workspace)) {
        yvex_attention_execution_trace_release(trace);
        return 0;
    }
    trace->envelope_output_width = envelope_width;
    return 1;
}

void yvex_attention_result_outputs_publish(yvex_attention_cpu_result *result,
                                           yvex_attention_operation_scope scope,
                                           const yvex_attention_component_span *core,
                                           const yvex_attention_component_span *envelope)
{
    result->core_output_checksum = core && core->data
        ? yvex_attention_checksum(core->data, core->expected_elements) : 0.0;
    result->envelope_output_checksum = envelope && envelope->data
        ? yvex_attention_checksum(envelope->data, envelope->expected_elements) : 0.0;
    result->output_checksum = scope == YVEX_ATTENTION_OPERATION_ENVELOPE
        ? result->envelope_output_checksum : result->core_output_checksum;
}

int yvex_attention_trace_capture(
    yvex_attention_execution_trace *trace, unsigned long long layer_index,
    yvex_attention_class attention_class, unsigned long long token_position,
    unsigned long long token_count, unsigned long long hidden_width, unsigned long long q_rank,
    unsigned long long query_width, unsigned long long kv_width, const float *input,
    const float *q_low, const float *query, const float *raw_kv, const float *compressed_kv,
    unsigned long long compressed_count, unsigned long long compressed_stride,
    const unsigned long long *compressed_positions, const float *indexer_kv,
    unsigned long long indexer_count, unsigned long long indexer_stride,
    const unsigned long long *indexer_positions, const float *index_query,
    unsigned long long index_query_stride, const float *index_weights,
    unsigned long long index_weight_stride, const float *attention_values, const float *output,
    const unsigned long long *topk_counts, const unsigned long long *topk_positions,
    unsigned long long topk_stride, const yvex_attention_rolling_state_output *main_state,
    const float *main_state_kv, const float *main_state_score,
    const yvex_attention_rolling_state_output *index_state, const float *index_state_kv,
    const float *index_state_score, unsigned long long rolling_checkpoint_count,
    const float *main_checkpoint_kv, const float *main_checkpoint_score,
    const float *index_checkpoint_kv, const float *index_checkpoint_score,
    yvex_attention_evidence_level evidence_level,
    yvex_attention_workspace *workspace) {
    unsigned long long floats[ATTENTION_TRACE_FLOAT_COUNT] = {0ull};
    unsigned long long integers[ATTENTION_TRACE_U64_COUNT] = {0ull};
    const float *float_sources[ATTENTION_TRACE_FLOAT_COUNT] = {
        input, q_low, query, raw_kv, compressed_kv, indexer_kv, index_query,
        index_weights, attention_values, output, NULL,
        rolling_checkpoint_count ? main_checkpoint_kv : main_state_kv,
        rolling_checkpoint_count ? main_checkpoint_score : main_state_score,
        rolling_checkpoint_count ? index_checkpoint_kv : index_state_kv,
        rolling_checkpoint_count ? index_checkpoint_score : index_state_score};
    const unsigned long long *u64_sources[ATTENTION_TRACE_U64_COUNT] = {
        compressed_positions, indexer_positions, topk_counts, topk_positions};
    attention_trace_shape shape;
    unsigned long long owned_bytes;
    if (!trace || trace->owned || !token_count || !hidden_width || !q_rank || !query_width ||
        !kv_width || (compressed_count && !compressed_positions) ||
        (indexer_count && !indexer_positions) || (topk_stride && (!topk_counts || !topk_positions)))
        return 0;
    memset(trace, 0, sizeof(*trace));
    trace->owned = 1;
    trace->workspace = workspace;
    trace->evidence_level = (unsigned int)evidence_level;
    trace->layer_index = layer_index;
    trace->attention_class = attention_class;
    trace->token_position = token_position;
    trace->token_count = token_count;
    trace->hidden_width = hidden_width;
    trace->q_rank = q_rank;
    trace->query_width = query_width;
    trace->kv_width = kv_width;
    trace->compressed_count = compressed_count;
    trace->compressed_stride = compressed_stride;
    trace->indexer_count = indexer_count;
    trace->indexer_stride = indexer_stride;
    trace->index_query_stride = index_query_stride;
    trace->index_weight_stride = index_weight_stride;
    trace->topk_stride = topk_stride;
    trace->prefix_addressable = rolling_checkpoint_count != 0ull;
    trace->rolling_checkpoint_count = rolling_checkpoint_count;
    if (main_state && main_state->present) {
        trace->next_main_rolling_state = *main_state;
        trace->next_main_rolling_state.kv_state = NULL;
        trace->next_main_rolling_state.score_state = NULL;
    }
    if (index_state && index_state->present) {
        trace->next_indexer_rolling_state = *index_state;
        trace->next_indexer_rolling_state.kv_state = NULL;
        trace->next_indexer_rolling_state.score_state = NULL;
    }
    shape = (attention_trace_shape){
        hidden_width, compressed_count, indexer_count,
        main_state && main_state->present ? main_state->kv_state_extent : 0ull,
        main_state && main_state->present ? main_state->score_state_extent : 0ull,
        index_state && index_state->present ? index_state->kv_state_extent : 0ull,
        index_state && index_state->present ? index_state->score_state_extent : 0ull,
        topk_stride, rolling_checkpoint_count ? rolling_checkpoint_count : 1ull};
    if (!attention_trace_counts(trace, &shape, floats, integers) ||
        !attention_trace_counts_filter(evidence_level, floats, integers))
        goto fail;
    /* The committed envelope is attached below; do not allocate an unused duplicate span. */
    floats[ATTENTION_TRACE_ENVELOPE_OUTPUT] = 0ull;
    if (attention_trace_storage_open(trace, floats, float_sources, integers, u64_sources,
                                     workspace, ULLONG_MAX, &owned_bytes) != YVEX_OK)
        goto fail;
    attention_trace_rolling_bind(
        trace, rolling_checkpoint_count ? rolling_checkpoint_count : 1ull);
    trace->complete = 1;
    return 1;
fail:
    yvex_attention_execution_trace_release(trace);
    return 0;
}

static void attention_component_release(yvex_attention_component_span *span) {
    if (!span)
        return;
    if (!span->workspace) free(span->data);
    memset(span, 0, sizeof(*span));
}

static void attention_transaction_release_staging(yvex_attention_state_transaction *transaction) {
    unsigned int i;
    if (!transaction)
        return;
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i)
        attention_component_release(&transaction->components[i]);
}
/* Construct component prepare f32 with checked geometry and explicit ownership. */
static int attention_component_prepare_f32(yvex_attention_component_span *span,
                                           yvex_attention_component_kind kind, unsigned int rank,
                                           unsigned long long dim0, unsigned long long dim1,
                                           unsigned long long stride,
                                           unsigned long long position_start,
                                           unsigned long long position_count) {
    unsigned long long elements;
    unsigned long long bytes;
    if (!span || rank == 0u || rank > 4u || dim0 == 0ull || (rank > 1u && dim1 == 0ull) ||
        stride < (rank > 1u ? dim1 : dim0))
        return 0;
    if (rank > 1u) {
        if (!yvex_core_u64_mul(dim0, stride, &elements))
            return 0;
    } else {
        elements = dim0;
    }
    if (!yvex_core_u64_mul(elements, sizeof(float), &bytes) || bytes > (unsigned long long)SIZE_MAX)
        return 0;
    memset(span, 0, sizeof(*span));
    span->kind = kind;
    span->storage = YVEX_DEEPSEEK_ATTENTION_COMPONENT_STORAGE_F32;
    span->rank = rank;
    span->dims[0] = dim0;
    span->dims[1] = rank > 1u ? dim1 : 0ull;
    span->stride = stride;
    span->expected_elements = elements;
    span->byte_extent = bytes;
    span->position_start = position_start;
    span->position_count = position_count;
    span->required = 1;
    return 1;
}

static int attention_emission_count(unsigned long long token_position,
                                    unsigned long long token_count, unsigned long long ratio,
                                    unsigned long long *out) {
    unsigned long long end;
    if (!out || ratio == 0ull || !yvex_core_u64_add(token_position, token_count, &end))
        return 0;
    *out = end / ratio - token_position / ratio;
    return 1;
}
typedef enum {
    ATTENTION_LAYOUT_ALWAYS,
    ATTENTION_LAYOUT_ENVELOPE,
    ATTENTION_LAYOUT_COMPRESSED,
    ATTENTION_LAYOUT_COMPRESSED_EMISSION,
    ATTENTION_LAYOUT_CSA,
    ATTENTION_LAYOUT_CSA_EMISSION
} attention_component_admission;
typedef enum {
    ATTENTION_EXTENT_TOKEN_HIDDEN,
    ATTENTION_EXTENT_TOKEN_KV,
    ATTENTION_EXTENT_TOKEN_ENVELOPE,
    ATTENTION_EXTENT_MAIN_KV,
    ATTENTION_EXTENT_MAIN_SCORE,
    ATTENTION_EXTENT_COMPRESSED_KV,
    ATTENTION_EXTENT_INDEX_KV,
    ATTENTION_EXTENT_INDEX_SCORE,
    ATTENTION_EXTENT_INDEXER_KV
} attention_component_extent;
typedef struct {
    yvex_attention_component_kind kind;
    attention_component_admission admission;
    attention_component_extent extent;
    const char *failure;
} attention_component_spec;
static const attention_component_spec attention_component_specs[] = {
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_ATTENTION_OUTPUT, ATTENTION_LAYOUT_ALWAYS,
     ATTENTION_EXTENT_TOKEN_HIDDEN, "DeepSeek attention state transaction output extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_RAW_LOCAL_KV, ATTENTION_LAYOUT_ALWAYS,
     ATTENTION_EXTENT_TOKEN_KV, "DeepSeek attention state transaction output extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_ENVELOPE_OUTPUT, ATTENTION_LAYOUT_ENVELOPE,
     ATTENTION_EXTENT_TOKEN_ENVELOPE, "attention envelope transaction extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_KV_STATE, ATTENTION_LAYOUT_COMPRESSED,
     ATTENTION_EXTENT_MAIN_KV, "DeepSeek attention main rolling-state extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE, ATTENTION_LAYOUT_COMPRESSED,
     ATTENTION_EXTENT_MAIN_SCORE, "DeepSeek attention main rolling-state extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_COMPRESSED_MAIN_KV,
     ATTENTION_LAYOUT_COMPRESSED_EMISSION, ATTENTION_EXTENT_COMPRESSED_KV,
     "DeepSeek attention compressed KV extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV_STATE, ATTENTION_LAYOUT_CSA,
     ATTENTION_EXTENT_INDEX_KV, "DeepSeek attention indexer rolling-state extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE, ATTENTION_LAYOUT_CSA,
     ATTENTION_EXTENT_INDEX_SCORE, "DeepSeek attention indexer rolling-state extent overflowed"},
    {YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_KV, ATTENTION_LAYOUT_CSA_EMISSION,
     ATTENTION_EXTENT_INDEXER_KV, "DeepSeek attention indexer KV extent overflowed"},
};
/* Derive every transaction component from one canonical typed layout table. */
static int attention_transaction_layout(
    const yvex_attention_layer_plan *layer, const yvex_attention_history_view *history,
    yvex_attention_operation_scope scope, unsigned long long position,
    unsigned long long tokens, yvex_attention_component_span *spans,
    unsigned long long *elements, const char **failure, unsigned long long *expected)
{
    unsigned long long emitted = 0ull, total = 0ull;
    unsigned int i;
    memset(spans, 0, sizeof(*spans) * YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT);
    if (layer->attention_class != YVEX_ATTENTION_CLASS_SWA &&
        !attention_emission_count(position, tokens, layer->compression_ratio, &emitted)) {
        if (failure) *failure = "DeepSeek attention compression emission count overflowed";
        if (expected) *expected = layer->compression_ratio;
        return 0;
    }
    for (i = 0u; i < sizeof(attention_component_specs) / sizeof(attention_component_specs[0]); ++i) {
        const attention_component_spec *spec = &attention_component_specs[i];
        const yvex_attention_rolling_state_view *rolling = &history->main_rolling_state;
        unsigned long long dim0 = tokens, dim1 = layer->hidden_dimension;
        unsigned long long stride = dim1, positions = tokens;
        int admitted = spec->admission == ATTENTION_LAYOUT_ALWAYS ||
            (spec->admission == ATTENTION_LAYOUT_ENVELOPE &&
             scope == YVEX_ATTENTION_OPERATION_ENVELOPE) ||
            ((spec->admission == ATTENTION_LAYOUT_COMPRESSED ||
              spec->admission == ATTENTION_LAYOUT_COMPRESSED_EMISSION) &&
             layer->attention_class != YVEX_ATTENTION_CLASS_SWA) ||
            ((spec->admission == ATTENTION_LAYOUT_CSA ||
              spec->admission == ATTENTION_LAYOUT_CSA_EMISSION) &&
             layer->attention_class == YVEX_ATTENTION_CLASS_CSA);
        if (!admitted || ((spec->admission == ATTENTION_LAYOUT_COMPRESSED_EMISSION ||
                           spec->admission == ATTENTION_LAYOUT_CSA_EMISSION) && !emitted))
            continue;
        if (spec->extent == ATTENTION_EXTENT_TOKEN_KV ||
            spec->extent == ATTENTION_EXTENT_COMPRESSED_KV)
            dim1 = stride = layer->head_dimension;
        else if (spec->extent == ATTENTION_EXTENT_TOKEN_ENVELOPE)
            dim1 = stride = layer->residual_expanded_width;
        else if (spec->extent == ATTENTION_EXTENT_INDEXER_KV)
            dim1 = stride = layer->indexer_head_dimension;
        if (spec->extent == ATTENTION_EXTENT_COMPRESSED_KV ||
            spec->extent == ATTENTION_EXTENT_INDEXER_KV)
            dim0 = positions = emitted;
        if (spec->extent == ATTENTION_EXTENT_MAIN_KV ||
            spec->extent == ATTENTION_EXTENT_MAIN_SCORE ||
            spec->extent == ATTENTION_EXTENT_INDEX_KV ||
            spec->extent == ATTENTION_EXTENT_INDEX_SCORE) {
            if (spec->extent == ATTENTION_EXTENT_INDEX_KV ||
                spec->extent == ATTENTION_EXTENT_INDEX_SCORE)
                rolling = &history->indexer_rolling_state;
            dim0 = rolling->state_slots;
            dim1 = rolling->state_width;
            stride = spec->extent == ATTENTION_EXTENT_MAIN_KV ||
                    spec->extent == ATTENTION_EXTENT_INDEX_KV
                ? rolling->kv_state_stride : rolling->score_state_stride;
        }
        if (!attention_component_prepare_f32(&spans[spec->kind], spec->kind, 2u, dim0, dim1,
                                             stride, position, positions) ||
            !yvex_core_u64_add(total, spans[spec->kind].expected_elements, &total)) {
            if (failure) *failure = spec->failure;
            if (expected) *expected = dim0;
            return 0;
        }
    }
    if (elements) *elements = total;
    return 1;
}
/*
 * Derive exact transaction-owned F32 staging for one attention request.
 *
 * Invalid state or arithmetic overflow leaves the output zero.
 */
int yvex_attention_transaction_scratch_elements(
    const yvex_attention_layer_plan *layer, const yvex_attention_history_view *history,
    yvex_attention_operation_scope scope, unsigned long long token_position,
    unsigned long long token_count, unsigned long long *elements)
{
    yvex_attention_component_span spans[YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT];
    if (elements) *elements = 0ull;
    if (!layer || !history || !elements || !token_count ||
        (scope != YVEX_ATTENTION_OPERATION_CORE && scope != YVEX_ATTENTION_OPERATION_ENVELOPE))
        return 0;
    return attention_transaction_layout(layer, history, scope, token_position, token_count,
                                        spans, elements, NULL, NULL);
}

static int attention_transaction_allocate_staging(yvex_attention_state_transaction *transaction,
                                                  yvex_attention_failure *failure,
                                                  yvex_error *err) {
    unsigned int i;
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i) {
        yvex_attention_component_span *span = &transaction->components[i];
        if (!span->required)
            continue;
        span->workspace = transaction->sink ? transaction->sink->workspace : NULL;
        span->data = span->workspace
            ? yvex_attention_workspace_calloc(span->workspace, span->byte_extent, 1ull)
            : calloc(1u, (size_t)span->byte_extent);
        if (!span->data) {
            attention_transaction_release_staging(transaction);
            return attention_memory_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, transaction->layer_index,
                span->byte_extent, 0ull, err, YVEX_ERR_NOMEM,
                "DeepSeek attention state transaction staging allocation failed");
        }
    }
    return YVEX_OK;
}
/* Derive the deterministic transaction identity identity from explicit semantic fields. */
static int attention_transaction_identity(const yvex_attention_state_transaction *transaction,
                                          char out[YVEX_DEEPSEEK_ATTENTION_IDENTITY_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int i;
    if (!transaction || !out)
        return 0;
    yvex_sha256_init(&hash);
    if (!attention_hash_text(&hash, "yvex.deepseek.attention.delta.v1") ||
        !attention_hash_u64(&hash, transaction->layer_index) ||
        !attention_hash_u64(&hash, transaction->attention_class) ||
        !attention_hash_u64(&hash, transaction->token_position) ||
        !attention_hash_u64(&hash, transaction->token_count) ||
        !attention_hash_text(&hash, transaction->previous_state_identity))
        return 0;
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i) {
        const yvex_attention_component_span *span = &transaction->components[i];
        if (!span->required)
            continue;
        if (!attention_hash_u64(&hash, span->kind) ||
            !attention_hash_u64(&hash, span->rank) ||
            !attention_hash_u64(&hash, span->dims[0]) ||
            !attention_hash_u64(&hash, span->dims[1]) ||
            !attention_hash_u64(&hash, span->stride) ||
            !attention_hash_u64(&hash, span->produced_elements) ||
            !attention_hash_u64(&hash, span->byte_extent) ||
            !yvex_sha256_update(&hash, span->data, (size_t)span->byte_extent))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, out);
    return 1;
}

static int attention_component_values_are_finite(const yvex_attention_component_span *span) {
    const float *values;
    unsigned long long i;
    if (!span || span->storage != YVEX_DEEPSEEK_ATTENTION_COMPONENT_STORAGE_F32 || !span->data)
        return 0;
    if (span->kind == YVEX_DEEPSEEK_ATTENTION_COMPONENT_MAIN_SCORE_STATE ||
        span->kind == YVEX_DEEPSEEK_ATTENTION_COMPONENT_INDEXER_SCORE_STATE)
        return 1;
    values = (const float *)span->data;
    for (i = 0ull; i < span->expected_elements; ++i) {
        if (!isfinite(values[i]))
            return 0;
    }
    return 1;
}

int yvex_attention_memory_sink_init(yvex_attention_memory_sink *sink,
                                    const yvex_attention_memory_sink_options *options,
                                    yvex_attention_failure *failure, yvex_error *err) {
    if (!sink)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, YVEX_ATTENTION_NO_LAYER,
            1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
            "DeepSeek attention memory sink requires caller storage");
    memset(sink, 0, sizeof(*sink));
    sink->initialized = 1;
    sink->options = options ? *options : memory_sink_defaults;
    sink->workspace = sink->options.workspace;
    return yvex_attention_accept(failure, err);
}

void yvex_attention_memory_sink_release(yvex_attention_memory_sink *sink) {
    unsigned int i;
    if (!sink)
        return;
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i)
        attention_component_release(&sink->committed[i]);
    memset(sink, 0, sizeof(*sink));
}

int yvex_attention_state_transaction_begin_scope(
    yvex_attention_memory_sink *sink, const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *history, yvex_attention_operation_scope scope,
    unsigned long long token_position, unsigned long long token_count,
    yvex_attention_state_transaction *transaction, yvex_attention_failure *failure,
    yvex_error *err) {
    const char *layout_failure = NULL;
    unsigned long long layout_expected = 0ull;
    int rc;
    if (transaction)
        memset(transaction, 0, sizeof(*transaction));
    if (!sink || !sink->initialized || !layer || !history || !transaction || token_count == 0ull ||
        (scope != YVEX_ATTENTION_OPERATION_CORE && scope != YVEX_ATTENTION_OPERATION_ENVELOPE))
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER, 1ull, token_count, err,
            YVEX_ERR_INVALID_ARG,
            "DeepSeek attention state transaction requires sink, layer, history, and tokens");
    if (sink->options.fail_begin)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, layer->layer_index, 1ull, 0ull,
            err, YVEX_ERR_STATE, "DeepSeek attention memory sink injected begin failure");
    rc = yvex_attention_history_validate(layer, history, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (history->token_count != token_position)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, layer->layer_index,
            history->token_count, token_position, err, YVEX_ERR_STATE,
            "DeepSeek attention state transaction token position is not contiguous");
    if (token_position > ULLONG_MAX - token_count)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, layer->layer_index, ULLONG_MAX,
            token_position, err, YVEX_ERR_BOUNDS,
            "DeepSeek attention state transaction token range overflows");
    transaction->status = YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN;
    transaction->sink = sink;
    transaction->layer_index = layer->layer_index;
    transaction->attention_class = layer->attention_class;
    transaction->token_position = token_position;
    transaction->token_count = token_count;
    yvex_core_text_copy(transaction->previous_state_identity,
                        sizeof(transaction->previous_state_identity),
                        sink->has_committed ? sink->committed_identity : "initial");
    if (!attention_transaction_layout(layer, history, scope, token_position, token_count,
                                      transaction->components, NULL, &layout_failure,
                                      &layout_expected))
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, layer->layer_index,
            layout_expected, 0ull, err, YVEX_ERR_BOUNDS,
            layout_failure ? layout_failure : "DeepSeek attention state layout overflowed");
    rc = attention_transaction_allocate_staging(transaction, failure, err);
    if (rc != YVEX_OK)
        return rc;
    return yvex_attention_accept(failure, err);
}

int yvex_attention_state_transaction_acquire(yvex_attention_state_transaction *transaction,
                                             yvex_attention_component_kind kind,
                                             yvex_attention_component_span *out,
                                             yvex_attention_failure *failure, yvex_error *err) {
    yvex_attention_component_span *span;
    if (out)
        memset(out, 0, sizeof(*out));
    if (!transaction || transaction->status != YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN ||
        kind >= YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
            transaction ? transaction->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN,
            transaction ? transaction->status : YVEX_DEEPSEEK_ATTENTION_TRANSACTION_EMPTY, err,
            YVEX_ERR_STATE, "DeepSeek attention state transaction acquire requires begun state");
    span = &transaction->components[kind];
    if (!span->required || !span->data || span->acquired)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, span->acquired, err, YVEX_ERR_STATE,
            "DeepSeek attention state component is absent or already acquired");
    if (transaction->sink && transaction->sink->options.fail_acquire_kind == kind)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention memory sink injected acquire failure");
    span->acquired = 1;
    if (out)
        *out = *span;
    return yvex_attention_accept(failure, err);
}

int yvex_attention_state_transaction_seal(yvex_attention_state_transaction *transaction,
                                          yvex_attention_component_kind kind,
                                          unsigned long long produced_elements,
                                          yvex_attention_failure *failure, yvex_error *err) {
    yvex_attention_component_span *span;
    if (!transaction || transaction->status != YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN ||
        kind >= YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
            transaction ? transaction->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN,
            transaction ? transaction->status : YVEX_DEEPSEEK_ATTENTION_TRANSACTION_EMPTY, err,
            YVEX_ERR_STATE, "DeepSeek attention state transaction seal requires begun state");
    span = &transaction->components[kind];
    if (!span->required || !span->acquired || span->sealed ||
        produced_elements != span->expected_elements)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            span->expected_elements, produced_elements, err,
            YVEX_ERR_BOUNDS, "DeepSeek attention state component seal has wrong element count");
    if (transaction->sink && transaction->sink->options.fail_seal_kind == kind)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention memory sink injected seal failure");
    if (!attention_component_values_are_finite(span))
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention state component contains non-finite values");
    span->produced_elements = produced_elements;
    span->written = 1;
    span->sealed = 1;
    return yvex_attention_accept(failure, err);
}

int yvex_attention_state_transaction_commit(yvex_attention_state_transaction *transaction,
                                            yvex_attention_failure *failure, yvex_error *err) {
    yvex_attention_component_span next[YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT];
    unsigned int i;
    memset(next, 0, sizeof(next));
    if (!transaction || transaction->status != YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN ||
        !transaction->sink || !transaction->sink->initialized)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
            transaction ? transaction->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN,
            transaction ? transaction->status : YVEX_DEEPSEEK_ATTENTION_TRANSACTION_EMPTY, err,
            YVEX_ERR_STATE, "DeepSeek attention state transaction commit requires begun state");
    if (transaction->sink->options.fail_commit)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention memory sink injected commit failure");
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i) {
        const yvex_attention_component_span *span = &transaction->components[i];
        if (!span->required)
            continue;
        if (!span->acquired || !span->written || !span->sealed ||
            span->produced_elements != span->expected_elements)
            return attention_memory_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
                span->expected_elements,
                span->produced_elements, err, YVEX_ERR_STATE,
                "DeepSeek attention state transaction cannot commit incomplete component");
    }
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i) {
        const yvex_attention_component_span *span = &transaction->components[i];
        if (!span->required)
            continue;
        next[i] = *span;
        if (!attention_storage_copy(&next[i].data, span->expected_elements, span->data,
                                    span->expected_elements, sizeof(float), next[i].workspace)) {
            unsigned int j;
            for (j = 0u; j < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++j)
                attention_component_release(&next[j]);
            return attention_memory_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, transaction->layer_index,
                span->byte_extent, 0ull, err, YVEX_ERR_NOMEM,
                "DeepSeek attention state commit allocation failed");
        }
    }
    if (!attention_transaction_identity(transaction, transaction->transaction_identity)) {
        for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i)
            attention_component_release(&next[i]);
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_FORMAT,
            "DeepSeek attention state transaction identity failed");
    }
    for (i = 0u; i < YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT; ++i) {
        attention_component_release(&transaction->sink->committed[i]);
        transaction->sink->committed[i] = next[i];
        memset(&next[i], 0, sizeof(next[i]));
    }
    yvex_core_text_copy(transaction->sink->committed_identity,
                        sizeof(transaction->sink->committed_identity),
                        transaction->transaction_identity);
    transaction->sink->has_committed = 1;
    attention_transaction_release_staging(transaction);
    transaction->status = YVEX_DEEPSEEK_ATTENTION_TRANSACTION_COMMITTED;
    return yvex_attention_accept(failure, err);
}

int yvex_attention_state_transaction_abort(yvex_attention_state_transaction *transaction,
                                           yvex_attention_failure *failure, yvex_error *err) {
    if (!transaction || transaction->status != YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
            transaction ? transaction->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_DEEPSEEK_ATTENTION_TRANSACTION_BEGUN,
            transaction ? transaction->status : YVEX_DEEPSEEK_ATTENTION_TRANSACTION_EMPTY, err,
            YVEX_ERR_STATE, "DeepSeek attention state transaction abort requires begun state");
    attention_transaction_release_staging(transaction);
    transaction->status = YVEX_DEEPSEEK_ATTENTION_TRANSACTION_ABORTED;
    if (transaction->sink && transaction->sink->options.fail_abort)
        return attention_memory_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA, transaction->layer_index,
            1ull, 0ull, err, YVEX_ERR_STATE,
            "DeepSeek attention memory sink injected abort failure");
    return yvex_attention_accept(failure, err);
}

const yvex_attention_component_span *
yvex_attention_memory_sink_committed_component(const yvex_attention_memory_sink *sink,
                                               yvex_attention_component_kind kind) {
    if (!sink || !sink->initialized || !sink->has_committed ||
        kind >= YVEX_DEEPSEEK_ATTENTION_COMPONENT_COUNT || !sink->committed[kind].required)
        return NULL;
    return &sink->committed[kind];
}

const char *yvex_attention_memory_sink_identity(const yvex_attention_memory_sink *sink) {
    if (!sink || !sink->initialized || !sink->has_committed)
        return NULL;
    return sink->committed_identity;
}
