/*
 * Seal attention-state geometry independently from mutable provider storage.
 *
 * Recipe identity binds component ordering and rolling geometry, never runtime pointers or
 * persistent values. Families can therefore project state requirements before a session exists.
 */
#include <yvex/internal/graph_state.h>

#include <stdint.h>
#include <string.h>

#include <yvex/internal/core.h>

#include "src/graph/private.h"

static int recipe_hash_u64s(yvex_sha256 *hash,
                            const unsigned long long *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}

static int recipe_history(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component,
    const float **values, const unsigned long long **positions,
    unsigned long long *count);

const yvex_attention_rolling_state_view *yvex_graph_state_rolling_view(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding)
{
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &view->main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &view->indexer_rolling_state;
    return NULL;
}

int yvex_graph_state_history_project(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component,
    yvex_graph_state_history_span *out)
{
    const float *values;
    const unsigned long long *positions;
    unsigned long long count;
    if (!view || !component || !out ||
        !recipe_history(view, component, &values, &positions, &count))
        return 0;
    *out = (yvex_graph_state_history_span){
        values, positions, count, component->value_width};
    return 1;
}

static int recipe_hash_floats(yvex_sha256 *hash, const float *values,
                              unsigned long long count)
{
    unsigned long long index;
    if ((count && !values) || !yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, &values[index], sizeof(bits));
        if (!yvex_sha256_update_u64(hash, (unsigned long long)bits)) return 0;
    }
    return 1;
}

static int recipe_rolling_layout_hash(
    yvex_sha256 *hash, const yvex_attention_rolling_state_view *view)
{
    const unsigned long long fields[] = {
        view->schema_version, (unsigned long long)view->kind,
        view->layer_index, view->ratio, view->head_dimension, view->state_width,
        view->state_slots, view->kv_state_stride, view->score_state_stride,
        view->kv_state_extent, view->score_state_extent,
        (unsigned long long)view->overlap, (unsigned long long)view->rotated};
    if (!yvex_sha256_update_u64(hash, (unsigned long long)view->present)) return 0;
    return !view->present ||
           (recipe_hash_u64s(hash, fields, sizeof(fields) / sizeof(fields[0])) &&
            yvex_sha256_update_text(hash, view->attention_plan_identity));
}

static int recipe_history(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component,
    const float **values, const unsigned long long **positions,
    unsigned long long *count)
{
    if (component->binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
        *values = view->local_kv;
        *positions = view->local_positions;
        *count = view->local_tail_count;
    } else if (component->binding ==
               YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) {
        *values = view->compressed_kv;
        *positions = view->compressed_positions;
        *count = view->compressed_entry_count;
    } else if (component->binding ==
               YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) {
        *values = view->indexer_kv;
        *positions = view->indexer_positions;
        *count = view->indexer_entry_count;
    } else {
        return 0;
    }
    return 1;
}

int yvex_graph_state_initial_identity(
    const yvex_attention_history_view *view,
    const yvex_attention_state_recipe *recipe, const char *plan_identity,
    char output[YVEX_SHA256_HEX_CAP])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    if (!view || !recipe || !yvex_sha256_hex_valid(plan_identity) || !output)
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state.v3") ||
        !yvex_sha256_update_text(&hash, plan_identity) ||
        !yvex_sha256_update_text(&hash, recipe->identity) ||
        !yvex_sha256_update_u64(&hash, view->token_count) ||
        !yvex_sha256_update_u64(&hash, recipe->component_count))
        return 0;
    for (index = 0u; index < recipe->component_count; ++index) {
        const yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            const float *values;
            const unsigned long long *positions;
            unsigned long long count, value_count, position;
            if (!recipe_history(view, component, &values, &positions, &count) ||
                !yvex_core_u64_mul(count, component->value_width, &value_count) ||
                !recipe_hash_floats(&hash, values, value_count))
                return 0;
            for (position = 0ull; position < count; ++position)
                if (!yvex_sha256_update_u64(&hash, positions[position])) return 0;
        } else {
            const yvex_attention_rolling_state_view *rolling =
                yvex_graph_state_rolling_view(view, component->binding);
            if (!rolling || !recipe_rolling_layout_hash(&hash, rolling) ||
                !yvex_sha256_update_u64(&hash, rolling->next_token_position) ||
                !yvex_sha256_update_u64(&hash, rolling->previous_fill) ||
                !yvex_sha256_update_u64(&hash, rolling->current_fill) ||
                !yvex_sha256_update_u64(&hash, rolling->cursor) ||
                !recipe_hash_floats(&hash, rolling->kv_state,
                                    rolling->kv_state_extent) ||
                !recipe_hash_floats(&hash, rolling->score_state,
                                    rolling->score_state_extent))
                return 0;
        }
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

int yvex_graph_state_advance_identity(
    const char *prior_identity, const yvex_attention_state_recipe *recipe,
    const char *plan_identity, const yvex_attention_publication *publication,
    char output[YVEX_SHA256_HEX_CAP])
{
    const int token_backed = publication && publication->token_ids != NULL;
    const unsigned long long rows = token_backed ? publication->token_count : 1ull;
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    char current[YVEX_SHA256_HEX_CAP];
    unsigned long long row;
    if (!yvex_sha256_hex_valid(prior_identity) || !recipe ||
        !yvex_sha256_hex_valid(plan_identity) || !publication || !output ||
        (!token_backed && !yvex_sha256_hex_valid(publication->execution_identity)))
        return 0;
    yvex_core_text_copy(current, sizeof(current), prior_identity);
    for (row = 0ull; row < rows; ++row) {
        yvex_sha256_init(&hash);
        if (!yvex_sha256_update_text(
                &hash, token_backed ? "yvex.graph.attention.state.v5"
                                    : "yvex.graph.attention.state.v4") ||
            !yvex_sha256_update_text(&hash, plan_identity) ||
            !yvex_sha256_update_text(&hash, recipe->identity) ||
            !yvex_sha256_update_text(&hash, current) ||
            !(token_backed
                  ? yvex_sha256_update_u64(
                        &hash, publication->token_position + row) &&
                        yvex_sha256_update_u64(&hash,
                                               publication->token_ids[row])
                  : yvex_sha256_update_text(
                        &hash, publication->execution_identity) &&
                        yvex_sha256_update_u64(&hash,
                                               publication->token_position +
                                                   publication->token_count)) ||
            !yvex_sha256_final(&hash, digest))
            return 0;
        yvex_sha256_hex(digest, current);
    }
    yvex_core_text_copy(output, YVEX_SHA256_HEX_CAP, current);
    return 1;
}

static int recipe_component_shape_valid(
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_state_component_recipe *component)
{
    const yvex_attention_rolling_state_view *rolling = &component->rolling;
    if (component->schema_version != YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1 ||
        component->binding >= YVEX_ATTENTION_STATE_BINDING_COUNT ||
        component->kind > YVEX_ATTENTION_STATE_COMPONENT_ROLLING)
        return 0;
    if (component->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY)
        return component->binding <= YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY &&
               component->value_width && !rolling->present && !rolling->kv_state &&
               !rolling->score_state;
    return component->binding >= YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING &&
           rolling->present &&
           rolling->schema_version == YVEX_ATTENTION_ROLLING_STATE_SCHEMA_V1 &&
           rolling->kind != YVEX_ATTENTION_ROLLING_NONE &&
           rolling->layer_index == recipe->layer_index &&
           rolling->next_token_position == recipe->initial_position &&
           rolling->kv_state_extent && rolling->score_state_extent &&
           rolling->kv_state_stride && rolling->score_state_stride &&
           !rolling->kv_state && !rolling->score_state &&
           strcmp(rolling->attention_plan_identity,
                  recipe->attention_plan_identity) == 0;
}

static int recipe_component_hash(
    yvex_sha256 *hash,
    const yvex_attention_state_component_recipe *component)
{
    const yvex_attention_rolling_state_view *rolling = &component->rolling;
    const unsigned long long fields[] = {
        component->schema_version, component->ordinal, component->kind,
        component->binding, component->capacity, component->value_width,
        (unsigned long long)rolling->present, rolling->schema_version,
        rolling->kind, rolling->layer_index, rolling->next_token_position,
        rolling->ratio, rolling->head_dimension, rolling->state_width,
        rolling->state_slots, rolling->previous_fill, rolling->current_fill,
        rolling->cursor, rolling->kv_state_stride, rolling->score_state_stride,
        rolling->kv_state_extent, rolling->score_state_extent,
        (unsigned long long)rolling->overlap,
        (unsigned long long)rolling->rotated};
    return yvex_sha256_update_text(
               hash, "yvex.graph.attention.state-component.v1") &&
           recipe_hash_u64s(hash, fields, sizeof(fields) / sizeof(fields[0])) &&
           yvex_sha256_update_text(hash, rolling->attention_plan_identity);
}

static int recipe_seal_unchecked(yvex_attention_state_recipe *recipe,
                                 yvex_error *err)
{
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    unsigned int seen = 0u;
    yvex_sha256 hash;
    if (!recipe || recipe->schema_version != YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1 ||
        !recipe->component_count ||
        recipe->component_count > YVEX_ATTENTION_STATE_COMPONENT_CAP ||
        recipe->final_position < recipe->initial_position ||
        !yvex_sha256_hex_valid(recipe->attention_plan_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "graph.attention.state.recipe",
                       "complete bounded state recipe facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash,
                                 "yvex.graph.attention.state-recipe.v1") ||
        !yvex_sha256_update_u64(&hash, recipe->schema_version) ||
        !yvex_sha256_update_u64(&hash, recipe->layer_index) ||
        !yvex_sha256_update_u64(&hash, recipe->selection_key) ||
        !yvex_sha256_update_u64(&hash, recipe->initial_position) ||
        !yvex_sha256_update_u64(&hash, recipe->final_position) ||
        !yvex_sha256_update_u64(&hash, recipe->component_count) ||
        !yvex_sha256_update_text(&hash, recipe->attention_plan_identity))
        goto identity_failure;
    for (index = 0u; index < recipe->component_count; ++index) {
        yvex_attention_state_component_recipe *component =
            &recipe->components[index];
        unsigned int bit;
        if (component->ordinal != index ||
            !recipe_component_shape_valid(recipe, component)) {
            yvex_error_set(err, YVEX_ERR_FORMAT,
                           "graph.attention.state.recipe",
                           "state component recipe shape is malformed");
            return YVEX_ERR_FORMAT;
        }
        bit = 1u << (unsigned int)component->binding;
        if (seen & bit) {
            yvex_error_set(err, YVEX_ERR_FORMAT,
                           "graph.attention.state.recipe",
                           "state component binding is duplicated");
            return YVEX_ERR_FORMAT;
        }
        seen |= bit;
        if (!recipe_component_hash(&hash, component)) goto identity_failure;
    }
    if (!yvex_sha256_final(&hash, digest)) goto identity_failure;
    yvex_sha256_hex(digest, recipe->identity);
    yvex_error_clear(err);
    return YVEX_OK;

identity_failure:
    recipe->identity[0] = '\0';
    yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.recipe",
                   "state recipe identity could not be sealed");
    return YVEX_ERR_STATE;
}

int yvex_attention_state_recipe_seal(yvex_attention_state_recipe *recipe,
                                     yvex_error *err)
{
    yvex_attention_state_recipe candidate;
    int rc;
    if (!recipe) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG,
                       "graph.attention.state.recipe",
                       "state recipe is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!recipe->identity[0]) return recipe_seal_unchecked(recipe, err);
    if (!yvex_sha256_hex_valid(recipe->identity)) goto mismatch;
    candidate = *recipe;
    candidate.identity[0] = '\0';
    rc = recipe_seal_unchecked(&candidate, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(recipe->identity, candidate.identity) != 0) goto mismatch;
    yvex_error_clear(err);
    return YVEX_OK;

mismatch:
    yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.recipe",
                   "state recipe identity does not match its fields");
    return YVEX_ERR_STATE;
}
