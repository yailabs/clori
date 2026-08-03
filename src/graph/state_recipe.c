/*
 * Seal attention-state geometry independently from mutable provider storage.
 *
 * Recipe identity binds component ordering and rolling geometry, never runtime pointers or
 * persistent values. Families can therefore project state requirements before a session exists.
 */
#include <yvex/internal/graph_state.h>

#include <string.h>

#include <yvex/internal/core.h>

static int recipe_hash_u64s(yvex_sha256 *hash,
                            const unsigned long long *values, size_t count)
{
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
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
