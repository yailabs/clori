/*
 * Retain bounded typed persistent state across phase-neutral attention executions.
 *
 * Storage follows sealed component recipes and committed history is immutable in a transaction.
 * Runtime retains an opaque provider handle and supplies optional backend residency.
 */
#include <yvex/internal/graph_state.h>
#include <yvex/internal/core.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct {
    yvex_attention_state_component_recipe recipe;
    float *values;
    unsigned long long *positions;
    float *auxiliary;
} state_component_storage;
typedef struct {
    yvex_attention_history_view view;
    state_component_storage components[YVEX_ATTENTION_STATE_BINDING_COUNT];
    char state_identity[YVEX_SHA256_HEX_CAP];
} attention_state_bank;
typedef struct {
    yvex_attention_layer_plan plan;
    yvex_attention_state_recipe recipe;
    attention_state_bank bank[2];
    unsigned long long allocated_bytes;
    unsigned int committed_bank;
    int prepared, staged;
} attention_layer_state;
typedef struct {
    unsigned long long layer_index, token_position, token_count, next_position;
    unsigned long long component_entries[YVEX_ATTENTION_STATE_BINDING_COUNT];
    char prior_state_identity[YVEX_SHA256_HEX_CAP];
    char candidate_state_identity[YVEX_SHA256_HEX_CAP];
    char state_delta_identity[YVEX_SHA256_HEX_CAP];
    int requires_commit;
} attention_state_delta;
typedef struct {
    attention_layer_state *layer;
    unsigned long long layer_ordinal, token_position, token_count, applied_tokens, staged_count;
    unsigned long long batch_position, batch_token_count;
    unsigned long long prepared_commit_count, prepared_generation, prepared_next_position;
    int active, candidate_active, failed, publication_prepared;
    yvex_attention_cancellation cancellation;
    int cancellation_bound;
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char prepared_content_identity[YVEX_SHA256_HEX_CAP];
    attention_state_delta delta;
} attention_state_transaction;
typedef struct {
    const float *values;
    const unsigned long long *positions;
    unsigned long long count, width;
} state_history_span;

static int state_hash_u64s(yvex_sha256 *hash, const unsigned long long *values,
                           size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}

static int state_component_recipe_shape_valid(
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_state_component_recipe *component) {
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

static int state_component_recipe_hash(
    yvex_sha256 *hash,
    const yvex_attention_state_component_recipe *component) {
    const yvex_attention_rolling_state_view *rolling = &component->rolling;
    const unsigned long long fields[] = {
        component->schema_version, component->ordinal, component->kind,
        component->binding, component->capacity, component->value_width,
        (unsigned long long)rolling->present, rolling->schema_version,
        rolling->kind, rolling->layer_index,
        rolling->next_token_position, rolling->ratio, rolling->head_dimension,
        rolling->state_width, rolling->state_slots, rolling->previous_fill,
        rolling->current_fill, rolling->cursor, rolling->kv_state_stride,
        rolling->score_state_stride, rolling->kv_state_extent,
        rolling->score_state_extent, (unsigned long long)rolling->overlap,
        (unsigned long long)rolling->rotated};
    return yvex_sha256_update_text(hash,
                                   "yvex.graph.attention.state-component.v1") &&
           state_hash_u64s(hash, fields, sizeof(fields) / sizeof(fields[0])) &&
           yvex_sha256_update_text(hash, rolling->attention_plan_identity);
}
/*
 * Seal one family-projected recipe using only canonical component fields.
 *
 * Malformed or unhashable facts return a typed error with no trusted recipe identity.
 */
static int state_recipe_seal_unchecked(yvex_attention_state_recipe *recipe,
                                       yvex_error *err) {
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    unsigned int seen = 0u;
    yvex_sha256 hash;
    if (!recipe || recipe->schema_version != YVEX_ATTENTION_STATE_RECIPE_SCHEMA_V1 ||
        !recipe->component_count ||
        recipe->component_count > YVEX_ATTENTION_STATE_COMPONENT_CAP ||
        recipe->final_position < recipe->initial_position ||
        !yvex_sha256_hex_valid(recipe->attention_plan_identity)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.state.recipe",
                       "complete bounded state recipe facts are required");
        return YVEX_ERR_INVALID_ARG;
    }
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state-recipe.v1") ||
        !yvex_sha256_update_u64(&hash, recipe->schema_version) ||
        !yvex_sha256_update_u64(&hash, recipe->layer_index) ||
        !yvex_sha256_update_u64(&hash, recipe->selection_key) ||
        !yvex_sha256_update_u64(&hash, recipe->initial_position) ||
        !yvex_sha256_update_u64(&hash, recipe->final_position) ||
        !yvex_sha256_update_u64(&hash, recipe->component_count) ||
        !yvex_sha256_update_text(&hash, recipe->attention_plan_identity))
        goto identity_failure;
    for (index = 0u; index < recipe->component_count; ++index) {
        yvex_attention_state_component_recipe *component = &recipe->components[index];
        unsigned int bit;
        if (component->ordinal != index ||
            !state_component_recipe_shape_valid(recipe, component)) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.attention.state.recipe",
                           "state component recipe shape is malformed");
            return YVEX_ERR_FORMAT;
        }
        bit = 1u << (unsigned int)component->binding;
        if (seen & bit) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "graph.attention.state.recipe",
                           "state component binding is duplicated");
            return YVEX_ERR_FORMAT;
        }
        seen |= bit;
        if (!state_component_recipe_hash(&hash, component))
            goto identity_failure;
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
/*
 * Seal an unpublished recipe or independently validate a sealed one.
 *
 * Mutable unpublished or identity-bearing family recipe and typed error output.
 */
int yvex_attention_state_recipe_seal(yvex_attention_state_recipe *recipe,
                                     yvex_error *err) {
    yvex_attention_state_recipe candidate;
    int rc;
    if (!recipe) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.state.recipe",
                       "state recipe is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!recipe->identity[0]) return state_recipe_seal_unchecked(recipe, err);
    if (!yvex_sha256_hex_valid(recipe->identity)) goto mismatch;
    candidate = *recipe;
    candidate.identity[0] = '\0';
    rc = state_recipe_seal_unchecked(&candidate, err);
    if (rc != YVEX_OK) return rc;
    if (strcmp(recipe->identity, candidate.identity) != 0) goto mismatch;
    yvex_error_clear(err);
    return YVEX_OK;
mismatch:
    yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.recipe",
                   "state recipe identity does not match its fields");
    return YVEX_ERR_STATE;
}
typedef struct {
    const yvex_graph_family_api *family;
    const yvex_attention_plan *plan;
    attention_layer_state *layers;
    unsigned long long layer_count, maximum_host_bytes;
    yvex_graph_attention_state_summary summary;
    attention_state_transaction transaction;
    pthread_mutex_t mutex;
    int mutex_ready;
} attention_state;
static const yvex_graph_attention_state_summary initial_state_summary = {
    .schema_version = YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V2,
    .sealed = 1, .persistent = 1, .position_consistent = 1,
    .generation = 1ull};
static void state_close(attention_state **state_ptr);

static state_history_span state_history_project(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component) {
    switch (component->binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        return (state_history_span){view->local_kv, view->local_positions,
                                    view->local_tail_count, component->value_width};
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        return (state_history_span){view->compressed_kv,
                                    view->compressed_positions,
                                    view->compressed_entry_count,
                                    component->value_width};
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        return (state_history_span){view->indexer_kv, view->indexer_positions,
                                    view->indexer_entry_count,
                                    component->value_width};
    default: return (state_history_span){0};
    }
}

static const yvex_attention_rolling_state_view *state_rolling_view(
    const yvex_attention_history_view *view,
    yvex_attention_state_binding binding) {
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &view->main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &view->indexer_rolling_state;
    return NULL;
}

static int state_reject(yvex_attention_failure *failure, unsigned long long layer,
                        unsigned long long expected, unsigned long long actual,
                        const char *reason, yvex_status status, yvex_error *err) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = status == YVEX_ERR_CANCELLED ? YVEX_ATTENTION_FAILURE_CANCELLED
                                                     : YVEX_ATTENTION_FAILURE_STATE_DELTA;
        failure->layer_index = layer;
        failure->role = YVEX_TENSOR_ROLE_UNKNOWN;
        failure->expected = expected;
        failure->actual = actual;
        failure->reason = reason;
    }
    yvex_error_setf(err, status, "graph.attention.state",
                    "%s (layer=%llu expected=%llu actual=%llu)",
                    reason, layer, expected, actual);
    return status;
}

static int state_lock(attention_state *state, unsigned long long layer,
                      yvex_attention_failure *failure, yvex_error *err) {
    if (state && state->mutex_ready && pthread_mutex_lock(&state->mutex) == 0)
        return YVEX_OK;
    return state_reject(failure, layer, 1ull, 0ull,
                        "attention state synchronization is unavailable", YVEX_ERR_STATE, err);
}

static int state_enter(attention_state *state, unsigned long long layer,
                       unsigned long long actual, const char *invalid_reason,
                       yvex_attention_failure *failure, yvex_error *err) {
    if (!state)
        return state_reject(failure, layer, 1ull, actual, invalid_reason,
                            YVEX_ERR_INVALID_ARG, err);
    return state_lock(state, layer, failure, err);
}

static int state_unlock_result(attention_state *state, int rc,
                               yvex_attention_failure *failure, yvex_error *err) {
    (void)pthread_mutex_unlock(&state->mutex);
    if (rc != YVEX_OK) return rc;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

static int state_transaction_result(attention_state *state, int rc,
                                    yvex_attention_failure *failure, yvex_error *err) {
    if (rc != YVEX_OK && state->transaction.active) state->transaction.failed = 1;
    return state_unlock_result(state, rc, failure, err);
}

static int state_cancel_check(attention_state *state,
                              const yvex_attention_cancellation *cancellation,
                              unsigned long long layer, const char *reason,
                              yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long next;
    if (!cancellation) return YVEX_OK;
    if (!cancellation->requested)
        return state_reject(failure, layer, 1ull, 0ull,
                            "attention state cancellation predicate is missing", YVEX_ERR_INVALID_ARG, err);
    if (cancellation->requested(cancellation->context)) {
        if (!yvex_core_u64_add(state->summary.cancellation_count, 1ull, &next)) {
            state->summary.invalidated = 1;
            return state_reject(failure, layer, ULLONG_MAX, 1ull,
                                "attention state cancellation count overflowed", YVEX_ERR_BOUNDS, err);
        }
        state->summary.cancellation_count = next;
        return state_reject(failure, layer, 0ull, 1ull, reason, YVEX_ERR_CANCELLED, err);
    }
    return YVEX_OK;
}

static int state_allocate(void **out, unsigned long long count, size_t width,
                          unsigned long long *accounted) {
    unsigned long long bytes;
    *out = NULL;
    if (!count) return 1;
    if (!yvex_core_u64_mul(count, (unsigned long long)width, &bytes) ||
        bytes > (unsigned long long)SIZE_MAX ||
        !yvex_core_u64_add(*accounted, bytes, accounted))
        return 0;
    *out = calloc((size_t)count, width);
    return *out != NULL;
}

static void state_bank_release(attention_state_bank *bank) {
    unsigned int index;
    if (!bank) return;
    for (index = 0u; index < YVEX_ATTENTION_STATE_BINDING_COUNT; ++index) {
        free(bank->components[index].values);
        free(bank->components[index].positions);
        free(bank->components[index].auxiliary);
    }
    memset(bank, 0, sizeof(*bank));
}
static int state_bank_identity(attention_state_bank *bank,
                               const attention_layer_state *layer,
                               const char *plan_identity);
/*
 * Return one prepared bank to its canonical empty state without reallocating storage.
 *
 * Prepared bank, immutable layer geometry, and sealed attention-plan identity. Clears every owned
 * span, resets dynamic counters, and replaces the state identity. Canonical identity failure
 * returns false after clearing the unusable bank. Allocation geometry, borrowed pointers, and
 * layout identity remain unchanged.
 */
static int state_bank_reset(attention_state_bank *bank,
                            const attention_layer_state *layer,
                            const char *plan_identity) {
    unsigned int index;
    bank->view.token_count = bank->view.local_tail_count =
        bank->view.compressed_entry_count = bank->view.indexer_entry_count = 0ull;
    for (index = 0u; index < layer->recipe.component_count; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        state_component_storage *storage = &bank->components[recipe->binding];
        unsigned long long count, element;
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            if (!yvex_core_u64_mul(recipe->capacity, recipe->value_width, &count))
                return 0;
            if (count)
                memset(storage->values, 0, (size_t)count * sizeof(float));
            if (recipe->capacity)
                memset(storage->positions, 0,
                       (size_t)recipe->capacity * sizeof(unsigned long long));
            continue;
        }
        {
            yvex_attention_rolling_state_view *view =
                (yvex_attention_rolling_state_view *)
                    state_rolling_view(&bank->view, recipe->binding);
            if (!view) return 0;
            memset(storage->values, 0,
                   (size_t)view->kv_state_extent * sizeof(float));
            for (element = 0ull; element < view->score_state_extent; ++element)
                storage->auxiliary[element] = -INFINITY;
            view->next_token_position = 0ull;
            view->previous_fill = view->current_fill = view->cursor = 0ull;
        }
    }
    return state_bank_identity(bank, layer, plan_identity);
}
/*
 * Bind one bank's owned arrays into its immutable history view.
 *
 * Publishes an in-process view only and transfers no ownership.
 */
static void state_bank_bind(attention_state_bank *bank,
                            const attention_layer_state *layer) {
    unsigned int index;
    for (index = 0u; index < layer->recipe.component_count; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        state_component_storage *storage = &bank->components[recipe->binding];
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_ROLLING) {
            yvex_attention_rolling_state_view *view =
                (yvex_attention_rolling_state_view *)
                    state_rolling_view(&bank->view, recipe->binding);
            view->kv_state = storage->values;
            view->score_state = storage->auxiliary;
        } else if (recipe->binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY) {
            bank->view.local_kv = storage->values;
            bank->view.local_positions = storage->positions;
            bank->view.local_kv_stride = recipe->value_width;
        } else if (recipe->binding ==
                   YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY) {
            bank->view.compressed_kv = storage->values;
            bank->view.compressed_positions = storage->positions;
            bank->view.compressed_kv_stride = recipe->value_width;
        } else if (recipe->binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY) {
            bank->view.indexer_kv = storage->values;
            bank->view.indexer_positions = storage->positions;
            bank->view.indexer_kv_stride = recipe->value_width;
        }
    }
    bank->view.immutable = 1;
}

static int state_bank_open(attention_state_bank *bank,
                           attention_layer_state *layer,
                           unsigned long long *bytes,
                           yvex_attention_failure *failure, yvex_error *err) {
    unsigned int index;
    int rc = YVEX_OK;
    memset(bank, 0, sizeof(*bank));
    for (index = 0u; index < layer->recipe.component_count && rc == YVEX_OK; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        state_component_storage *storage = &bank->components[recipe->binding];
        unsigned long long count, element;
        storage->recipe = *recipe;
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            if (!yvex_core_u64_mul(recipe->capacity, recipe->value_width, &count) ||
                !state_allocate((void **)&storage->values, count, sizeof(float), bytes) ||
                !state_allocate((void **)&storage->positions, recipe->capacity,
                                sizeof(unsigned long long), bytes))
                rc = state_reject(failure, layer->plan.layer_index, 1ull, 0ull,
                                  "attention history component allocation failed",
                                  YVEX_ERR_NOMEM, err);
            continue;
        }
        if (!state_allocate((void **)&storage->values,
                            recipe->rolling.kv_state_extent, sizeof(float), bytes) ||
            !state_allocate((void **)&storage->auxiliary,
                            recipe->rolling.score_state_extent, sizeof(float), bytes)) {
            rc = state_reject(failure, layer->plan.layer_index, 1ull, 0ull,
                              "attention rolling component allocation failed",
                              YVEX_ERR_NOMEM, err);
            continue;
        }
        *(yvex_attention_rolling_state_view *)
             state_rolling_view(&bank->view, recipe->binding) = recipe->rolling;
        for (element = 0ull; element < recipe->rolling.score_state_extent; ++element)
            storage->auxiliary[element] = -INFINITY;
    }
    if (rc != YVEX_OK) {
        state_bank_release(bank);
        return rc;
    }
    state_bank_bind(bank, layer);
    return YVEX_OK;
}

static int state_bank_transfer(attention_state_bank *bank,
                               const attention_layer_state *layer,
                               const yvex_attention_history_view *source,
                               int validate_storage) {
    yvex_attention_history_view allocated = bank->view;
    unsigned int index;
    if (!source) return 1;
    bank->view = *source;
    for (index = 0u; index < layer->recipe.component_count; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        state_component_storage *storage = &bank->components[recipe->binding];
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            state_history_span span = state_history_project(source, recipe);
            unsigned long long count;
            if (validate_storage &&
                (span.count > recipe->capacity ||
                 !yvex_core_u64_mul(span.count, span.width, &count) ||
                 (count && !span.values) ||
                 (span.count && !span.positions)))
                return 0;
            (void)yvex_core_u64_mul(span.count, span.width, &count);
            if (count)
                memcpy(storage->values, span.values,
                       (size_t)count * sizeof(float));
            if (span.count)
                memcpy(storage->positions, span.positions,
                       (size_t)span.count * sizeof(unsigned long long));
            continue;
        }
        {
            const yvex_attention_rolling_state_view *input =
                state_rolling_view(source, recipe->binding);
            const yvex_attention_rolling_state_view *target =
                state_rolling_view(&allocated, recipe->binding);
            yvex_attention_rolling_state_view *output =
                (yvex_attention_rolling_state_view *)
                    state_rolling_view(&bank->view, recipe->binding);
            if (!input->present) {
                if (!validate_storage)
                    *output = *state_rolling_view(&allocated, recipe->binding);
                continue;
            }
            if (validate_storage &&
                (!storage->values || !storage->auxiliary || !input->kv_state ||
                 !input->score_state || input->kv_state_extent != target->kv_state_extent ||
                 input->score_state_extent != target->score_state_extent))
                return 0;
            *output = *input;
            memcpy(storage->values, input->kv_state,
                   (size_t)input->kv_state_extent * sizeof(float));
            memcpy(storage->auxiliary, input->score_state,
                   (size_t)input->score_state_extent * sizeof(float));
        }
    }
    state_bank_bind(bank, layer);
    return 1;
}

static void state_bank_copy(attention_state_bank *destination,
                            const attention_state_bank *source,
    const attention_layer_state *layer) {
    (void)state_bank_transfer(destination, layer, &source->view, 0);
    memmove(destination->state_identity, source->state_identity, sizeof(destination->state_identity));
}

static int state_hash_float(yvex_sha256 *hash, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return yvex_sha256_update_u64(hash, (unsigned long long)bits);
}

static int state_hash_floats(yvex_sha256 *hash, const float *values,
                             unsigned long long count) {
    unsigned long long index;
    if (count && !values) return 0;
    if (!yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!state_hash_float(hash, values[index])) return 0;
    return 1;
}

static int state_rolling_layout_hash(
    yvex_sha256 *hash, const yvex_attention_rolling_state_view *view) {
    const unsigned long long fields[] = {
        view->schema_version, (unsigned long long)view->kind,
        view->layer_index, view->ratio,
        view->head_dimension, view->state_width, view->state_slots, view->kv_state_stride,
        view->score_state_stride, view->kv_state_extent, view->score_state_extent,
        (unsigned long long)view->overlap, (unsigned long long)view->rotated};
    if (!yvex_sha256_update_u64(hash, (unsigned long long)view->present)) return 0;
    if (!view->present) return 1;
    return state_hash_u64s(hash, fields, sizeof(fields) / sizeof(fields[0])) &&
           yvex_sha256_update_text(hash, view->attention_plan_identity);
}
/*
 * Compute one semantic state identity from explicit history fields and values.
 *
 * Complete bank, immutable layer geometry, and sealed attention-plan identity. Replaces the bank
 * identity with a deterministic canonical digest. Excludes pointers, bank selection, allocation
 * capacity, and transaction counters.
 */
static int state_bank_identity(attention_state_bank *bank,
                               const attention_layer_state *layer,
                               const char *plan_identity) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long count, position;
    unsigned int index;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state.v3") ||
        !yvex_sha256_update_text(&hash, plan_identity) ||
        !yvex_sha256_update_text(&hash, layer->recipe.identity) ||
        !yvex_sha256_update_u64(&hash, bank->view.token_count) ||
        !yvex_sha256_update_u64(&hash, layer->recipe.component_count))
        return 0;
    for (index = 0u; index < layer->recipe.component_count; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY) {
            state_history_span span = state_history_project(&bank->view, recipe);
            if (!yvex_core_u64_mul(span.count, span.width, &count) ||
                !state_hash_floats(&hash, span.values, count))
                return 0;
            for (position = 0ull; position < span.count; ++position)
                if (!yvex_sha256_update_u64(&hash, span.positions[position]))
                    return 0;
        } else {
            const yvex_attention_rolling_state_view *view =
                state_rolling_view(&bank->view, recipe->binding);
            if (!view || !state_rolling_layout_hash(&hash, view) ||
                !yvex_sha256_update_u64(&hash, view->next_token_position) ||
                !yvex_sha256_update_u64(&hash, view->previous_fill) ||
                !yvex_sha256_update_u64(&hash, view->current_fill) ||
                !yvex_sha256_update_u64(&hash, view->cursor) ||
                !state_hash_floats(&hash, view->kv_state, view->kv_state_extent) ||
                !state_hash_floats(&hash, view->score_state,
                                   view->score_state_extent))
                return 0;
        }
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, bank->state_identity);
    return 1;
}
/*
 * Compute provider layout identity from plan geometry and one optional unpublished layer.
 *
 * Writes one complete deterministic digest only after every field hashes successfully. Missing
 * plan facts or hash failure leaves provider ownership and summary unchanged. Layout identity
 * excludes history values, pointers, and allocation addresses.
 */
static int state_layout_identity(const attention_state *state,
                                 unsigned long long candidate_index,
                                 const attention_layer_state *candidate,
                                 char output[YVEX_SHA256_HEX_CAP]) {
    const yvex_attention_summary *summary = yvex_attention_plan_summary(state->plan);
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (!summary) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state-layout.v2") ||
        !yvex_sha256_update_u64(&hash, YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V2) ||
        !yvex_sha256_update_text(&hash, summary->attention_plan_identity) ||
        !yvex_sha256_update_u64(&hash, state->layer_count))
        return 0;
    for (index = 0ull; index < state->layer_count; ++index) {
        const attention_layer_state *layer =
            candidate && index == candidate_index ? candidate : &state->layers[index];
        const unsigned long long fields[] = {
            layer->plan.layer_index, (unsigned long long)layer->prepared};
        if (!state_hash_u64s(&hash, fields, sizeof(fields) / sizeof(fields[0])))
            return 0;
        if (layer->prepared &&
            !yvex_sha256_update_text(&hash, layer->recipe.identity))
            return 0;
    }
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int state_content_identity(const attention_state *state,
                                  unsigned long long candidate_index,
                                  const attention_layer_state *candidate,
                                  int staged,
                                  const char *layout_identity,
                                  char output[YVEX_SHA256_HEX_CAP]) {
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index, count = 0ull;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.persistent-state.v1") ||
        !yvex_sha256_update_text(&hash, layout_identity))
        return 0;
    for (index = 0ull; index < state->layer_count; ++index) {
        const attention_layer_state *layer =
            candidate && index == candidate_index ? candidate : &state->layers[index];
        const attention_state_bank *bank;
        if (!layer->prepared) continue;
        bank = &layer->bank[staged && layer->staged
                                ? 1u - layer->committed_bank
                                : layer->committed_bank];
        if (!yvex_sha256_update_u64(&hash, index) ||
            !yvex_sha256_update_text(&hash, bank->state_identity))
            return 0;
        ++count;
    }
    if (!yvex_sha256_update_u64(&hash, count) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    return 1;
}

static int state_rolling_apply(yvex_attention_rolling_state_view *view,
                               float *kv, float *score,
                               const yvex_attention_rolling_state_output *output) {
    if (!output->present) return !view->present;
    if (!view->present || !kv || !score || !output->kv_state ||
        !output->score_state || output->kv_state_extent != view->kv_state_extent ||
        output->score_state_extent != view->score_state_extent)
        return 0;
    memcpy(kv, output->kv_state, (size_t)output->kv_state_extent * sizeof(float));
    memcpy(score, output->score_state, (size_t)output->score_state_extent * sizeof(float));
    yvex_attention_rolling_output_bind(output, view);
    view->kv_state = kv;
    view->score_state = score;
    return 1;
}

static const yvex_attention_rolling_state_output *state_publication_rolling(
    const yvex_attention_publication *publication,
    yvex_attention_state_binding binding) {
    if (binding == YVEX_ATTENTION_STATE_BINDING_MAIN_ROLLING)
        return &publication->next_main_rolling_state;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_ROLLING)
        return &publication->next_indexer_rolling_state;
    return NULL;
}

static const yvex_attention_state_component_recipe *state_component_recipe_find(
    const attention_layer_state *layer, yvex_attention_state_binding binding) {
    unsigned int index;
    for (index = 0u; index < layer->recipe.component_count; ++index)
        if (layer->recipe.components[index].binding == binding)
            return &layer->recipe.components[index];
    return NULL;
}

static state_history_span state_publication_history(
    const yvex_attention_publication *publication,
    yvex_attention_state_binding binding) {
    switch (binding) {
    case YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY:
        return (state_history_span){publication->raw_kv, NULL,
                                    publication->token_count,
                                    publication->kv_width};
    case YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY:
        return (state_history_span){publication->compressed_kv,
                                    publication->compressed_positions,
                                    publication->compressed_count,
                                    publication->compressed_stride};
    case YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY:
        return (state_history_span){publication->indexer_kv,
                                    publication->indexer_positions,
                                    publication->indexer_count,
                                    publication->indexer_stride};
    default: return (state_history_span){0};
    }
}

static unsigned long long *state_history_count(
    yvex_attention_history_view *view, yvex_attention_state_binding binding) {
    if (binding == YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY)
        return &view->local_tail_count;
    if (binding == YVEX_ATTENTION_STATE_BINDING_COMPRESSED_HISTORY)
        return &view->compressed_entry_count;
    if (binding == YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY)
        return &view->indexer_entry_count;
    return NULL;
}
/*
 * Preflight one publication before mutating the candidate bank.
 *
 * Active transaction and one complete production publication.
 */
static int state_publication_validate(const attention_state_transaction *transaction,
                                      const yvex_attention_publication *publication) {
    const attention_layer_state *layer = transaction->layer;
    const attention_state_bank *candidate = &layer->bank[1u - layer->committed_bank];
    unsigned long long remaining = transaction->token_count - transaction->applied_tokens;
    unsigned long long next, expected_next;
    unsigned int binding;
    if (!publication ||
        !yvex_core_u64_add(candidate->view.token_count, publication->token_count, &next) ||
        !yvex_core_u64_add(publication->token_position, publication->token_count, &expected_next) ||
        !publication->complete ||
        publication->layer_index != layer->plan.layer_index ||
        publication->token_position != candidate->view.token_count ||
        !publication->token_count || publication->token_count > remaining ||
        next != expected_next)
        return 0;
    for (binding = 0u; binding <= YVEX_ATTENTION_STATE_BINDING_INDEXER_HISTORY;
         ++binding) {
        const yvex_attention_state_component_recipe *recipe =
            state_component_recipe_find(layer, (yvex_attention_state_binding)binding);
        state_history_span span = state_publication_history(
            publication, (yvex_attention_state_binding)binding);
        unsigned long long current = recipe
            ? state_history_project(&candidate->view, recipe).count : 0ull;
        if (!recipe) {
            if (span.count && binding != YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY)
                return 0;
            continue;
        }
        if (span.count && recipe->capacity &&
            (!span.values || span.width < recipe->value_width ||
             (binding != YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY &&
              !span.positions)))
            return 0;
        if (span.positions &&
            (!yvex_core_u64_add(current, span.count, &current) ||
             current > recipe->capacity))
            return 0;
    }
    return 1;
}

static void state_history_append(attention_state_bank *bank,
                                 const attention_layer_state *layer,
                                 const yvex_attention_publication *publication) {
    unsigned int component;
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_component_storage *storage = &bank->components[recipe->binding];
        state_history_span span;
        unsigned long long *count, row;
        if (recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY || !recipe->capacity)
            continue;
        span = state_publication_history(publication, recipe->binding);
        count = state_history_count(&bank->view, recipe->binding);
        for (row = 0ull; row < span.count; ++row) {
            if (!span.positions && *count == recipe->capacity) {
                memmove(storage->values, storage->values + recipe->value_width,
                        (size_t)((*count - 1ull) * recipe->value_width) * sizeof(float));
                memmove(storage->positions, storage->positions + 1,
                        (size_t)(*count - 1ull) * sizeof(*storage->positions));
                --*count;
            }
            memcpy(storage->values + *count * recipe->value_width,
                   span.values + row * span.width,
                   (size_t)recipe->value_width * sizeof(float));
            storage->positions[*count] = span.positions
                                             ? span.positions[row]
                                             : publication->token_position + row;
            ++*count;
        }
    }
}
/*
 * Derive one candidate delta identity from prior and complete candidate state.
 *
 * Active transaction with fully applied token publications. Fills candidate counters and replaces
 * its state-delta identity. Hash failure returns false and leaves the transaction uncommittable.
 * Identity names the proposed change but does not publish the candidate bank.
 */
static int state_delta_identity(attention_state_transaction *transaction) {
    const attention_layer_state *layer = transaction->layer;
    const attention_state_bank *candidate = &layer->bank[1u - layer->committed_bank];
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned int index;
    transaction->delta.next_position = candidate->view.token_count;
    for (index = 0u; index < layer->recipe.component_count; ++index) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[index];
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY)
            transaction->delta.component_entries[recipe->binding] =
                state_history_project(&candidate->view, recipe).count;
    }
    yvex_core_text_copy(transaction->delta.candidate_state_identity,
                        sizeof(transaction->delta.candidate_state_identity),
                        candidate->state_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state-delta.v2") ||
        !yvex_sha256_update_text(&hash, transaction->state_layout_identity) ||
        !yvex_sha256_update_text(&hash, transaction->delta.prior_state_identity) ||
        !yvex_sha256_update_text(&hash, transaction->delta.candidate_state_identity) ||
        !yvex_sha256_update_u64(&hash, transaction->layer_ordinal) ||
        !yvex_sha256_update_u64(&hash, transaction->token_position) ||
        !yvex_sha256_update_u64(&hash, transaction->applied_tokens) ||
        !state_hash_u64s(&hash, transaction->delta.component_entries,
                         YVEX_ATTENTION_STATE_BINDING_COUNT) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, transaction->delta.state_delta_identity);
    return 1;
}
/*
 * Open one empty session-local provider without preparing heavyweight layer banks.
 *
 * Admitted family graph ABI, sealed plan, memory budget, and output ownership slot. Invalid
 * owners, allocation, plan lookup, or identity failure releases all partial state.
 */
static int state_open(
    attention_state **out, const yvex_graph_family_api *family,
    const yvex_attention_plan *plan, unsigned long long maximum_host_bytes,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state *state;
    unsigned long long index, count;
    if (out) *out = NULL;
    count = yvex_attention_plan_layer_count(plan);
    if (!out || !family || !family->state_recipe || !family->history_validate ||
        !plan || !count)
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                            "sealed attention plan and family state ABI are required",
                            YVEX_ERR_INVALID_ARG, err);
    state = (attention_state *)calloc(1u, sizeof(*state));
    if (!state)
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                            "attention state provider allocation failed",
                            YVEX_ERR_NOMEM, err);
    state->layers = (attention_layer_state *)calloc((size_t)count, sizeof(*state->layers));
    if (!state->layers || pthread_mutex_init(&state->mutex, NULL) != 0) {
        free(state->layers);
        free(state);
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, count, 0ull,
                            "attention state provider initialization failed", YVEX_ERR_NOMEM, err);
    }
    state->mutex_ready = 1;
    state->family = family;
    state->plan = plan;
    state->layer_count = count;
    state->maximum_host_bytes = maximum_host_bytes;
    state->summary = initial_state_summary;
    state->summary.layer_count = count;
    for (index = 0ull; index < count; ++index) {
        const yvex_attention_layer_plan *layer = yvex_attention_plan_layer_at(plan, index);
        if (!layer) {
            state_close(&state);
            return state_reject(failure, index, 1ull, 0ull,
                                "attention state layer lookup failed", YVEX_ERR_STATE, err);
        }
        state->layers[index].plan = *layer;
    }
    if (!state_layout_identity(state, ULLONG_MAX, NULL,
                               state->summary.state_layout_identity)) {
        state_close(&state);
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                            "attention state layout identity failed", YVEX_ERR_STATE, err);
    }
    *out = state;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}
/*
 * Prepare two allocation-stable banks for one layer and optional immutable prior state.
 *
 * Atomically installs equal committed/candidate banks and updates layout identity.
 */
static int state_prepare(
    attention_state *state, unsigned long long layer_index,
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_history_view *initial_history,
    yvex_attention_failure *failure, yvex_error *err) {
    const yvex_attention_summary *summary;
    attention_layer_state candidate;
    unsigned long long bank, bytes = 0ull, total;
    int rc = YVEX_OK;
    rc = state_enter(state, layer_index, layer_index,
                     "attention state prepare arguments are invalid", failure, err);
    if (rc != YVEX_OK) return rc;
    if (!recipe || layer_index >= state->layer_count) {
        if (state->transaction.active) state->transaction.failed = 1;
        rc = state_reject(failure, layer_index, state->layer_count, layer_index,
                          "attention state prepare arguments are invalid", YVEX_ERR_INVALID_ARG, err);
        goto done;
    }
    if (state->transaction.active || state->summary.cancelled ||
        state->summary.invalidated) {
        if (state->transaction.active) state->transaction.failed = 1;
        rc = state_reject(failure, layer_index, 0ull, 1ull,
                          state->summary.invalidated
                              ? "attention state provider is invalidated"
                              : state->summary.cancelled
                                    ? "attention state provider is cancelled"
                                    : "attention state transaction is active",
                          YVEX_ERR_STATE, err);
        goto done;
    }
    if (state->layers[layer_index].prepared) {
        rc = state_reject(failure, layer_index, 0ull, 1ull,
                          "attention state layer is already prepared", YVEX_ERR_STATE, err);
        goto done;
    }
    memset(&candidate, 0, sizeof(candidate));
    candidate.plan = state->layers[layer_index].plan;
    candidate.recipe = *recipe;
    if (recipe->layer_index != candidate.plan.layer_index ||
        strcmp(recipe->attention_plan_identity,
               yvex_attention_plan_summary(state->plan)->attention_plan_identity) != 0 ||
        yvex_attention_state_recipe_seal(&candidate.recipe, err) != YVEX_OK) {
        rc = state_reject(failure, layer_index, candidate.plan.layer_index,
                          recipe->layer_index,
                          "attention state recipe does not match its sealed layer",
                          YVEX_ERR_FORMAT, err);
        goto done;
    }
    if (initial_history) {
        rc = state->family->history_validate(&candidate.plan, initial_history,
                                             failure, err);
        if (rc != YVEX_OK) goto done;
    }
    summary = yvex_attention_plan_summary(state->plan);
    for (bank = 0ull; bank < 2ull && rc == YVEX_OK; ++bank) {
        rc = state_bank_open(&candidate.bank[bank], &candidate,
                             &bytes, failure, err);
        if (rc == YVEX_OK && initial_history &&
            !state_bank_transfer(&candidate.bank[bank], &candidate,
                                 initial_history, 1))
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "initial attention history could not be copied", YVEX_ERR_FORMAT, err);
    }
    candidate.prepared = rc == YVEX_OK;
    for (bank = 0ull; bank < 2ull && rc == YVEX_OK; ++bank)
        if (!state_bank_identity(&candidate.bank[bank], &candidate,
                                 summary->attention_plan_identity))
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "initial attention state identity failed", YVEX_ERR_STATE, err);
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(state->summary.allocated_bytes, bytes, &total) ||
         (state->maximum_host_bytes && total > state->maximum_host_bytes)))
        rc = state_reject(failure, layer_index, state->maximum_host_bytes, total,
                          "attention state host budget exceeded", YVEX_ERR_BOUNDS, err);
    if (rc == YVEX_OK) {
        unsigned long long prepared_next, generation_next;
        char layout_identity[YVEX_SHA256_HEX_CAP];
        char content_identity[YVEX_SHA256_HEX_CAP];
        unsigned long long capacity_next =
            state->summary.prepared_layer_count
                ? (recipe->final_position < state->summary.capacity
                       ? recipe->final_position : state->summary.capacity)
                : recipe->final_position;
        int position_consistent =
            state->summary.position_consistent &&
            (!state->summary.prepared_layer_count ||
             state->summary.next_position == recipe->initial_position);
        if (!yvex_core_u64_add(state->summary.prepared_layer_count, 1ull,
                               &prepared_next) ||
            !yvex_core_u64_add(state->summary.generation, 1ull,
                               &generation_next)) {
            rc = state_reject(failure, layer_index, ULLONG_MAX, 1ull,
                              "attention state capacity accounting overflowed", YVEX_ERR_BOUNDS, err);
            goto done;
        }
        candidate.allocated_bytes = bytes;
        if (!state_layout_identity(state, layer_index, &candidate, layout_identity)) {
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "prepared state layout identity failed", YVEX_ERR_STATE, err);
            goto done;
        }
        if (!state_content_identity(state, layer_index, &candidate, 0,
                                    layout_identity, content_identity)) {
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "prepared state content identity failed",
                              YVEX_ERR_STATE, err);
            goto done;
        }
        state->layers[layer_index] = candidate;
        state->summary.prepared_layer_count = prepared_next;
        state->summary.generation = generation_next;
        state->summary.allocated_bytes = total;
        state->summary.capacity = capacity_next;
        state->summary.position_consistent = position_consistent;
        if (prepared_next == 1ull) {
            state->summary.committed_sequence_length = recipe->initial_position;
            state->summary.next_position = recipe->initial_position;
        }
        yvex_core_text_copy(state->summary.state_layout_identity,
                            sizeof(state->summary.state_layout_identity),
                            layout_identity);
        yvex_core_text_copy(state->summary.state_content_identity,
                            sizeof(state->summary.state_content_identity),
                            content_identity);
        memset(&candidate, 0, sizeof(candidate));
    }
    if (rc != YVEX_OK) {
        state_bank_release(&candidate.bank[0]);
        state_bank_release(&candidate.bank[1]);
    }
done:
    return state_unlock_result(state, rc, failure, err);
}

static const yvex_attention_history_view *state_view(
    const attention_state *state, unsigned long long layer_index,
    yvex_attention_state_view_kind kind) {
    attention_state *mutable_state = (attention_state *)state;
    const attention_layer_state *layer;
    const yvex_attention_history_view *view = NULL;
    if (!state || layer_index >= state->layer_count || !state->mutex_ready ||
        pthread_mutex_lock(&mutable_state->mutex) != 0)
        return NULL;
    layer = &state->layers[layer_index];
    if (!state->summary.invalidated && !state->summary.cancelled &&
        layer->prepared && kind == YVEX_ATTENTION_STATE_VIEW_COMMITTED)
        view = &layer->bank[layer->committed_bank].view;
    else if (!state->summary.invalidated && !state->summary.cancelled &&
             layer->prepared && kind == YVEX_ATTENTION_STATE_VIEW_CANDIDATE &&
             (!state->transaction.active ||
              (!state->transaction.failed &&
               (layer->staged ||
                (state->transaction.candidate_active &&
                 state->transaction.layer == layer)))))
        view = &layer->bank[1u - layer->committed_bank].view;
    (void)pthread_mutex_unlock(&mutable_state->mutex);
    return view;
}

static int state_summary_add(unsigned long long entries,
                             unsigned long long capacity,
                             yvex_graph_attention_state_component_summary *summary) {
    if (!yvex_core_u64_add(summary->entry_count, entries,
                           &summary->entry_count) ||
        !yvex_core_u64_add(summary->capacity, capacity,
                           &summary->capacity))
        return 0;
    if (capacity > summary->maximum_capacity)
        summary->maximum_capacity = capacity;
    return 1;
}
/*
 * Copy the canonical identity of one committed session-local layer state.
 *
 * Synchronized provider, prepared layer ordinal, and fixed identity output. Copies identity bytes
 * while holding the provider lifecycle lock. Identity covers persistent attention history without
 * pointers or backend placement.
 */
static int state_identity_copy(
    attention_state *state, unsigned long long layer_index,
    char output[YVEX_SHA256_HEX_CAP], yvex_error *err) {
    attention_layer_state *layer;
    if (output) output[0] = '\0';
    if (!state || !output || layer_index >= state->layer_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.state.identity",
                       "prepared attention state and identity output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!state->mutex_ready || pthread_mutex_lock(&state->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.identity",
                       "attention state synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    layer = &state->layers[layer_index];
    if (!layer->prepared || state->transaction.active || state->summary.invalidated ||
        state->summary.cancelled) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.identity",
                       "valid committed prepared attention state is required");
        return state_unlock_result(state, YVEX_ERR_STATE, NULL, err);
    }
    yvex_core_text_copy(output, YVEX_SHA256_HEX_CAP, layer->bank[layer->committed_bank].state_identity);
    return state_unlock_result(state, YVEX_OK, NULL, err);
}
/*
 * Copy synchronized state lifecycle, capacity, and committed-entry facts.
 *
 * Malformed ownership or aggregate overflow publishes no partial snapshot.
 */
static int state_summary_copy(
    const attention_state *state,
    yvex_graph_attention_state_summary *out, yvex_error *err) {
    attention_state *mutable_state = (attention_state *)state;
    unsigned long long index;
    if (!state || !out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.state.summary",
                       "attention state and summary output are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!state->mutex_ready || pthread_mutex_lock(&mutable_state->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.summary",
                       "attention state synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    *out = state->summary;
    out->transaction_active = state->transaction.active;
    out->candidate_active = state->transaction.candidate_active;
    out->abort_required = state->transaction.failed;
    out->staged_layer_count = state->transaction.staged_count;
    for (index = 0ull; index < state->layer_count; ++index) {
        const attention_layer_state *layer = &state->layers[index];
        const yvex_attention_history_view *view;
        unsigned int component;
        if (!layer->prepared) continue;
        view = &layer->bank[layer->committed_bank].view;
        for (component = 0u; component < layer->recipe.component_count;
             ++component) {
            const yvex_attention_state_component_recipe *recipe =
                &layer->recipe.components[component];
            unsigned long long entries;
            if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY)
                entries = state_history_project(view, recipe).count;
            else
                entries = state_rolling_view(view, recipe->binding)->present
                              ? 1ull : 0ull;
            if (!state_summary_add(entries, recipe->capacity,
                                   &out->components[recipe->binding])) {
                memset(out, 0, sizeof(*out));
                yvex_error_set(err, YVEX_ERR_BOUNDS,
                               "graph.attention.state.summary",
                               "attention state aggregate accounting overflowed");
                return state_unlock_result(mutable_state, YVEX_ERR_BOUNDS,
                                           NULL, err);
            }
        }
    }
    if (state->transaction.active && !state->transaction.candidate_active &&
        !state->transaction.failed && state->transaction.staged_count &&
        state->transaction.staged_count == state->summary.prepared_layer_count) {
        if (!yvex_core_u64_add(state->summary.generation, 1ull,
                               &out->staged_generation) ||
            !yvex_core_u64_add(state->transaction.batch_position,
                               state->transaction.batch_token_count,
                               &out->staged_next_position) ||
            out->staged_next_position > state->summary.capacity ||
            !state_content_identity(
                state, ULLONG_MAX, NULL, 1,
                state->summary.state_layout_identity,
                out->staged_state_content_identity)) {
            memset(out, 0, sizeof(*out));
            yvex_error_set(err, YVEX_ERR_BOUNDS,
                           "graph.attention.state.summary",
                           "staged attention state summary overflowed");
            return state_unlock_result(mutable_state, YVEX_ERR_BOUNDS,
                                       NULL, err);
        }
        out->staged_batch_complete = 1;
    }
    return state_unlock_result(mutable_state, YVEX_OK, NULL, err);
}

static void state_candidate_clear(attention_state_transaction *transaction) {
    transaction->layer = NULL;
    transaction->layer_ordinal = transaction->token_position =
        transaction->token_count = transaction->applied_tokens = 0ull;
    transaction->candidate_active = 0;
    memset(&transaction->delta, 0, sizeof(transaction->delta));
}
/*
 * Start one allocation-free layer candidate inside a provider-wide batch.
 *
 * The batch remains private until one atomic publish after complete graph execution.
 */
static int state_begin(
    attention_state *state, unsigned long long layer_index,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_layer_state *layer;
    attention_state_bank *committed, *candidate;
    int rc = YVEX_OK;
    rc = state_enter(state, layer_index, token_count,
                     "attention state transaction arguments are invalid", failure, err);
    if (rc != YVEX_OK) return rc;
    if (layer_index >= state->layer_count || !token_count) {
        if (state->transaction.active) state->transaction.failed = 1;
        rc = state_reject(failure, layer_index, state->layer_count, layer_index,
                          "attention state transaction arguments are invalid", YVEX_ERR_INVALID_ARG, err);
        goto done;
    }
    layer = &state->layers[layer_index];
    if (!layer->prepared || state->transaction.candidate_active ||
        state->transaction.failed || layer->staged || state->summary.cancelled ||
        state->summary.invalidated || !state->summary.position_consistent) {
        rc = state_reject(failure, layer_index, 0ull, 1ull,
                          state->summary.invalidated
                              ? "attention state provider is invalidated"
                              : state->summary.cancelled
                                    ? "attention state provider is cancelled"
                              : !layer->prepared
                                    ? "attention state layer is not prepared"
                              : state->transaction.candidate_active
                                    ? "another attention state candidate is active"
                              : state->transaction.failed
                                    ? "attention state batch is already failed"
                              : layer->staged
                                    ? "attention state layer is already staged"
                              : !state->summary.position_consistent
                                    ? "attention state layer positions disagree"
                                    : "attention state layer is not transaction-ready",
                          state->summary.invalidated
                              ? YVEX_ERR_STATE
                              : state->summary.cancelled ? YVEX_ERR_CANCELLED : YVEX_ERR_STATE,
                          err);
        goto done;
    }
    if (token_position > ULLONG_MAX - token_count ||
        token_position + token_count > layer->recipe.final_position ||
        (state->transaction.active &&
         (state->transaction.batch_position != token_position ||
          state->transaction.batch_token_count != token_count))) {
        if (state->transaction.active) state->transaction.failed = 1;
        rc = state_reject(
            failure, layer_index, layer->recipe.final_position,
            token_position > ULLONG_MAX - token_count
                ? ULLONG_MAX : token_position + token_count,
            state->transaction.active
                ? "attention state layer range differs inside one model transaction"
                : "attention state append exceeds its sealed capacity",
            state->transaction.active ? YVEX_ERR_STATE : YVEX_ERR_BOUNDS, err);
        goto done;
    }
    rc = state_cancel_check(state, cancellation, layer_index,
                            "attention state cancelled before begin", failure, err);
    if (rc != YVEX_OK) goto done;
    if (state->transaction.active &&
        (state->transaction.cancellation_bound != (cancellation != NULL) ||
         (cancellation &&
          (state->transaction.cancellation.requested != cancellation->requested ||
           state->transaction.cancellation.context != cancellation->context)))) {
        state->transaction.failed = 1;
        rc = state_reject(failure, layer_index, 1ull, 0ull,
                          "attention state cancellation view changed inside one batch", YVEX_ERR_STATE, err);
        goto done;
    }
    committed = &layer->bank[layer->committed_bank];
    if (committed->view.token_count != token_position) {
        rc = state_reject(failure, layer_index, committed->view.token_count, token_position,
                          "attention state position is not contiguous", YVEX_ERR_STATE, err);
        goto done;
    }
    candidate = &layer->bank[1u - layer->committed_bank];
    state_bank_copy(candidate, committed, layer);
    if (!state->transaction.active) {
        memset(&state->transaction, 0, sizeof(state->transaction));
        state->transaction.active = 1;
        state->transaction.batch_position = token_position;
        state->transaction.batch_token_count = token_count;
        if (cancellation) {
            state->transaction.cancellation = *cancellation;
            state->transaction.cancellation_bound = 1;
        }
    }
    state->transaction.candidate_active = 1;
    state->transaction.layer = layer;
    state->transaction.layer_ordinal = layer_index;
    state->transaction.token_position = token_position;
    state->transaction.token_count = token_count;
    state->transaction.delta.layer_index = layer_index;
    state->transaction.delta.token_position = token_position;
    state->transaction.delta.token_count = token_count;
    state->transaction.delta.requires_commit = 1;
    yvex_core_text_copy(state->transaction.state_layout_identity,
                        sizeof(state->transaction.state_layout_identity),
                        state->summary.state_layout_identity);
    yvex_core_text_copy(state->transaction.delta.prior_state_identity,
                        sizeof(state->transaction.delta.prior_state_identity),
                        committed->state_identity);
done:
    return state_transaction_result(state, rc, failure, err);
}

/* Resolve a successful preflight without another fallible operation. */
static void state_publish_prepared(attention_state *state) {
    attention_state_transaction *transaction;
    unsigned long long index;
    if (!state) return;
    transaction = &state->transaction;
    if (!transaction->publication_prepared) return;
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        if (!layer->staged) continue;
        layer->committed_bank = 1u - layer->committed_bank;
        layer->staged = 0;
    }
    state->summary.commit_count = transaction->prepared_commit_count;
    state->summary.generation = transaction->prepared_generation;
    state->summary.committed_sequence_length = transaction->prepared_next_position;
    state->summary.next_position = transaction->prepared_next_position;
    yvex_core_text_copy(state->summary.state_content_identity,
                        sizeof(state->summary.state_content_identity),
                        transaction->prepared_content_identity);
    memset(transaction, 0, sizeof(*transaction));
    (void)pthread_mutex_unlock(&state->mutex);
}

static int state_apply(
    attention_state *state,
    const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char delta_identity_output[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state_transaction *transaction;
    attention_layer_state *layer;
    attention_state_bank *candidate;
    unsigned int component;
    int rc = YVEX_OK;
    if (delta_identity_output) delta_identity_output[0] = '\0';
    rc = state_enter(state, YVEX_ATTENTION_NO_LAYER, 0ull,
                     "attention state provider is required", failure, err);
    if (rc != YVEX_OK) return rc;
    transaction = &state->transaction;
    if (!transaction->active || !transaction->candidate_active || transaction->failed ||
        !state_publication_validate(transaction, publication)) {
        if (transaction->active) transaction->failed = 1;
        rc = state_reject(failure,
                          transaction->candidate_active ? transaction->layer_ordinal
                                              : YVEX_ATTENTION_NO_LAYER,
                          transaction->candidate_active ? transaction->token_count : 1ull,
                          publication ? publication->token_count : 0ull,
                          "attention publication does not match the active candidate",
                          YVEX_ERR_FORMAT, err);
        goto done;
    }
    rc = state_cancel_check(state, cancellation, transaction->layer_ordinal,
                            "attention state cancelled before candidate apply", failure, err);
    if (rc != YVEX_OK) {
        transaction->failed = 1;
        goto done;
    }
    layer = transaction->layer;
    candidate = &layer->bank[1u - layer->committed_bank];
    state_history_append(candidate, layer, publication);
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_component_storage *storage;
        const yvex_attention_rolling_state_output *output;
        if (recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_ROLLING) continue;
        storage = &candidate->components[recipe->binding];
        output = state_publication_rolling(publication, recipe->binding);
        if (!output ||
            !state_rolling_apply((yvex_attention_rolling_state_view *)
                                     state_rolling_view(&candidate->view,
                                                        recipe->binding),
                                 storage->values, storage->auxiliary, output)) {
            transaction->failed = 1;
            rc = state_reject(failure, transaction->layer_ordinal, 1ull, 0ull,
                              "attention rolling state delta is incomplete",
                              YVEX_ERR_FORMAT, err);
            goto done;
        }
    }
    candidate->view.token_count += publication->token_count;
    transaction->applied_tokens += publication->token_count;
    rc = state->family->history_validate(&layer->plan, &candidate->view, failure, err);
    if (rc != YVEX_OK) {
        transaction->failed = 1;
        goto done;
    }
    if (!state_bank_identity(candidate, layer,
                             yvex_attention_plan_summary(state->plan)->attention_plan_identity) ||
        !state_delta_identity(transaction)) {
        transaction->failed = 1;
        rc = state_reject(failure, transaction->layer_ordinal, 1ull, 0ull,
                          "attention candidate state identity failed", YVEX_ERR_STATE, err);
        goto done;
    }
    if (delta_identity_output)
        yvex_core_text_copy(delta_identity_output, YVEX_SHA256_HEX_CAP,
                            transaction->delta.state_delta_identity);
    if (transaction->applied_tokens == transaction->token_count) {
        unsigned long long next;
        if (!yvex_core_u64_add(transaction->staged_count, 1ull, &next)) {
            transaction->failed = 1;
            rc = state_reject(failure, transaction->layer_ordinal, ULLONG_MAX, 1ull,
                              "attention state staged-layer count overflowed",
                              YVEX_ERR_BOUNDS, err);
            goto done;
        }
        transaction->staged_count = next;
        transaction->layer->staged = 1;
        state_candidate_clear(transaction);
    }
done:
    return state_transaction_result(state, rc, failure, err);
}
/* Validate publication while retaining the provider lock, so a session can
 * preflight every participating state owner before any bank becomes visible. */
static int state_prepare_publish(
    attention_state *state,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state_transaction *transaction;
    unsigned long long index, staged = 0ull, commit_next, generation_next;
    unsigned long long next_position = ULLONG_MAX;
    char content_identity[YVEX_SHA256_HEX_CAP];
    int injected;
    int rc = YVEX_OK;
    rc = state_enter(state, YVEX_ATTENTION_NO_LAYER, 0ull,
                     "attention state provider is required", failure, err);
    if (rc != YVEX_OK) return rc;
    transaction = &state->transaction;
    for (index = 0ull; index < state->layer_count; ++index)
        if (state->layers[index].staged) ++staged;
    injected = getenv("YVEX_TEST_RUNTIME_STATE_PUBLISH_FAILURE") != NULL;
    if (!transaction->active || transaction->candidate_active || transaction->failed ||
        transaction->publication_prepared ||
        !transaction->staged_count || staged != transaction->staged_count ||
        staged != state->summary.prepared_layer_count ||
        transaction->batch_position != state->summary.next_position ||
        state->summary.cancelled || state->summary.invalidated || injected) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, transaction->staged_count, staged,
            injected
                ? "attention state publication fault was injected"
                : "attention state batch is incomplete and cannot publish",
            YVEX_ERR_STATE, err);
        goto done;
    }
    if (!yvex_core_u64_add(state->summary.commit_count, staged, &commit_next)) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, ULLONG_MAX, staged,
                          "attention state commit count overflowed", YVEX_ERR_BOUNDS, err);
        goto done;
    }
    if (!yvex_core_u64_add(state->summary.generation, 1ull, &generation_next) ||
        !yvex_core_u64_add(transaction->batch_position,
                           transaction->batch_token_count, &next_position) ||
        next_position > state->summary.capacity ||
        !state_content_identity(state, ULLONG_MAX, NULL, 1,
                                state->summary.state_layout_identity,
                                content_identity)) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER,
                          state->summary.capacity, next_position,
                          "attention state publication identity or position failed",
                          YVEX_ERR_BOUNDS, err);
        goto done;
    }
    rc = state_cancel_check(state,
                            transaction->cancellation_bound ? &transaction->cancellation : NULL,
                            YVEX_ATTENTION_NO_LAYER, "attention state cancelled before publication",
                            failure, err);
    if (rc != YVEX_OK) goto done;
    transaction->prepared_commit_count = commit_next;
    transaction->prepared_generation = generation_next;
    transaction->prepared_next_position = next_position;
    yvex_core_text_copy(transaction->prepared_content_identity,
                        sizeof(transaction->prepared_content_identity), content_identity);
    transaction->publication_prepared = 1;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
done:
    return state_transaction_result(state, rc, failure, err);
}

static void state_cancel_prepared_publish(attention_state *state) {
    if (!state || !state->transaction.publication_prepared) return;
    state->transaction.publication_prepared = 0;
    state->transaction.prepared_commit_count = 0ull;
    state->transaction.prepared_generation = 0ull;
    state->transaction.prepared_next_position = 0ull;
    state->transaction.prepared_content_identity[0] = '\0';
    (void)pthread_mutex_unlock(&state->mutex);
}

static int state_publish(attention_state *state,
                         yvex_attention_failure *failure, yvex_error *err) {
    int rc = state_prepare_publish(state, failure, err);
    if (rc == YVEX_OK) state_publish_prepared(state);
    return rc;
}

static int state_abort(
    attention_state *state,
    yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long index, next;
    if (!state)
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                            "attention state provider is required", YVEX_ERR_INVALID_ARG, err);
    if (!state->mutex_ready || getenv("YVEX_TEST_RUNTIME_STATE_ABORT_FAILURE") ||
        pthread_mutex_lock(&state->mutex) != 0)
        return state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                            "attention state abort synchronization failed", YVEX_ERR_STATE, err);
    if (!state->transaction.active) {
        return state_unlock_result(state, YVEX_OK, failure, err);
    }
    if (!yvex_core_u64_add(state->summary.abort_count, 1ull, &next)) {
        state->summary.invalidated = 1;
        return state_unlock_result(
            state, state_reject(failure, state->transaction.layer_ordinal,
                                ULLONG_MAX, 1ull,
                                "attention state abort counter overflowed",
                                YVEX_ERR_BOUNDS, err), failure, err);
    }
    for (index = 0ull; index < state->layer_count; ++index)
        state->layers[index].staged = 0;
    memset(&state->transaction, 0, sizeof(state->transaction));
    state->summary.abort_count = next;
    return state_unlock_result(state, YVEX_OK, failure, err);
}

static int state_reset(
    attention_state *state,
    yvex_attention_failure *failure, yvex_error *err) {
    const yvex_attention_summary *plan_summary;
    unsigned long long index, generation, reset_count;
    char content_identity[YVEX_SHA256_HEX_CAP];
    int rc = YVEX_OK;
    rc = state_enter(state, YVEX_ATTENTION_NO_LAYER, 0ull,
                     "attention state provider is required", failure, err);
    if (rc != YVEX_OK) return rc;
    if (state->transaction.active || state->summary.cancelled || state->summary.invalidated) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, 0ull, 1ull,
                          "only an idle valid attention state may reset", YVEX_ERR_STATE, err);
        goto done;
    }
    if (!yvex_core_u64_add(state->summary.generation, 1ull, &generation) ||
        !yvex_core_u64_add(state->summary.reset_count, 1ull, &reset_count)) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, ULLONG_MAX, 1ull,
                          "attention state reset accounting overflowed", YVEX_ERR_BOUNDS, err);
        goto done;
    }
    plan_summary = yvex_attention_plan_summary(state->plan);
    if (!plan_summary) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                          "attention state reset lost its plan authority", YVEX_ERR_STATE, err);
        goto done;
    }
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        unsigned int bank;
        if (!layer->prepared) continue;
        for (bank = 0u; bank < 2u; ++bank)
            if (!state_bank_reset(&layer->bank[bank], layer,
                                  plan_summary->attention_plan_identity)) {
                state->summary.invalidated = 1;
                rc = state_reject(failure, index, 1ull, 0ull,
                                  "attention state reset identity failed", YVEX_ERR_STATE, err);
                goto done;
            }
        layer->committed_bank = 0u;
        layer->staged = 0;
    }
    if (!state_content_identity(state, ULLONG_MAX, NULL, 0,
                                state->summary.state_layout_identity,
                                content_identity)) {
        state->summary.invalidated = 1;
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                          "attention state reset content identity failed",
                          YVEX_ERR_STATE, err);
        goto done;
    }
    memset(&state->transaction, 0, sizeof(state->transaction));
    state->summary.generation = generation;
    state->summary.reset_count = reset_count;
    state->summary.committed_sequence_length = 0ull;
    state->summary.next_position = 0ull;
    state->summary.position_consistent = 1;
    yvex_core_text_copy(state->summary.state_content_identity,
                        sizeof(state->summary.state_content_identity),
                        content_identity);
done:
    return state_unlock_result(state, rc, failure, err);
}
/*
 * Poison every candidate and permanently invalidate one provider generation.
 *
 * Missing ownership or generation overflow fails closed.
 */
static int state_invalidate(attention_state *state, yvex_error *err) {
    unsigned long long next;
    if (!state) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.attention.state.invalidate",
                       "attention state is required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (!state->mutex_ready || getenv("YVEX_TEST_RUNTIME_STATE_INVALIDATE_FAILURE") ||
        pthread_mutex_lock(&state->mutex) != 0) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.attention.state.invalidate",
                       "attention state synchronization is unavailable");
        return YVEX_ERR_STATE;
    }
    if (state->summary.invalidated) {
        return state_unlock_result(state, YVEX_OK, NULL, err);
    }
    if (!yvex_core_u64_add(state->summary.generation, 1ull, &next)) {
        state->summary.invalidated = state->summary.cancelled = 1;
        if (state->transaction.active) state->transaction.failed = 1;
        yvex_error_set(err, YVEX_ERR_BOUNDS, "graph.attention.state.invalidate",
                       "attention state generation overflowed");
        return state_unlock_result(state, YVEX_ERR_BOUNDS, NULL, err);
    }
    state->summary.generation = next;
    state->summary.invalidated = state->summary.cancelled = 1;
    if (state->transaction.active) state->transaction.failed = 1;
    return state_unlock_result(state, YVEX_OK, NULL, err);
}

static void state_close(attention_state **state_ptr) {
    attention_state *state;
    unsigned long long layer, bank;
    if (!state_ptr || !*state_ptr) return;
    state = *state_ptr;
    if (state->mutex_ready && pthread_mutex_lock(&state->mutex) != 0) return;
    *state_ptr = NULL;
    for (layer = 0ull; layer < state->layer_count; ++layer)
        for (bank = 0ull; bank < 2ull; ++bank)
            state_bank_release(&state->layers[layer].bank[bank]);
    free(state->layers);
    if (state->mutex_ready) {
        (void)pthread_mutex_unlock(&state->mutex);
        (void)pthread_mutex_destroy(&state->mutex);
    }
    memset(state, 0, sizeof(*state));
    free(state);
}

static int provider_persistent_prepare(void *context, unsigned long long layer_index,
                                      const yvex_attention_state_recipe *recipe,
                                      const yvex_attention_history_view *initial_history,
                                      yvex_attention_failure *failure, yvex_error *err) {
    return state_prepare((attention_state *)context, layer_index, recipe, initial_history,
                         failure, err);
}

static int provider_persistent_summary(void *context, yvex_graph_attention_state_summary *out,
                                      yvex_error *err) {
    return state_summary_copy((const attention_state *)context, out, err);
}

static const yvex_attention_history_view *provider_persistent_view(
    void *context, unsigned long long layer_index, yvex_attention_state_view_kind kind) {
    return state_view((const attention_state *)context, layer_index, kind);
}

static int provider_persistent_identity(void *context, unsigned long long layer_index,
                                       char output[YVEX_SHA256_HEX_CAP], yvex_error *err) {
    return state_identity_copy((attention_state *)context, layer_index, output, err);
}
/*
 * Begin one default state candidate and return its immutable committed prior.
 *
 * Opens one candidate transaction and borrows its committed history.
 */
static int provider_persistent_begin(void *context, unsigned long long layer_ordinal,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *initial_history,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation,
    const yvex_attention_history_view **history,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state *state = (attention_state *)context;
    const yvex_attention_history_view *committed;
    int rc;
    if (history) *history = NULL;
    if (!state || !layer || !history || !token_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.state.provider.begin",
                       "valid provider, layer, history output, and token range are required");
        return YVEX_ERR_INVALID_ARG;
    }
    (void)initial_history;
    committed = state_view(state, layer_ordinal, YVEX_ATTENTION_STATE_VIEW_COMMITTED);
    if (!committed) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.state.provider.begin",
                       "committed attention state is unavailable");
        return YVEX_ERR_STATE;
    }
    rc = state_begin(state, layer_ordinal, token_position, token_count, cancellation,
                     failure, err);
    if (rc == YVEX_OK) *history = committed;
    return rc;
}

static int provider_persistent_stage(void *context,
    const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char state_delta_identity[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err) {
    return state_apply((attention_state *)context, publication, cancellation,
                       state_delta_identity, failure, err);
}

static int provider_persistent_prepare_commit(
    void *context, yvex_attention_failure *failure, yvex_error *err) {
    return state_prepare_publish((attention_state *)context, failure, err);
}

static void provider_persistent_publish_commit(void *context) {
    state_publish_prepared((attention_state *)context);
}

static void provider_persistent_cancel_commit(void *context) {
    state_cancel_prepared_publish((attention_state *)context);
}

static int provider_persistent_commit(void *context, yvex_attention_failure *failure,
                                     yvex_error *err) {
    return state_publish((attention_state *)context, failure, err);
}

static int provider_persistent_abort(void *context, yvex_attention_failure *failure,
                                    yvex_error *err) {
    return state_abort((attention_state *)context, failure, err);
}

static int provider_persistent_reset(void *context, yvex_attention_failure *failure,
                                    yvex_error *err) {
    return state_reset((attention_state *)context, failure, err);
}

static int provider_persistent_invalidate(void *context, yvex_error *err) {
    return state_invalidate((attention_state *)context, err);
}

static int provider_persistent_release(void **context, yvex_error *err) {
    attention_state *state;
    if (!context || !*context) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    state = (attention_state *)*context;
    state_close(&state);
    *context = state;
    if (state) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.state.provider.release",
                       "persistent attention state cleanup is incomplete");
        return YVEX_ERR_STATE;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_attention_state_provider_open_persistent(
    const yvex_graph_family_api *family, const yvex_attention_plan *plan,
    unsigned long long maximum_host_bytes, yvex_attention_state_provider *out,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state *state = NULL;
    int rc;
    if (out) memset(out, 0, sizeof(*out));
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.state.provider.open",
                       "attention state provider output is required");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = state_open(&state, family, plan, maximum_host_bytes, failure, err);
    if (rc != YVEX_OK) return rc;
    *out = (yvex_attention_state_provider){
        .schema_version = YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V3,
        .context = state,
        .prepare = provider_persistent_prepare,
        .summary = provider_persistent_summary,
        .view = provider_persistent_view,
        .identity = provider_persistent_identity,
        .begin = provider_persistent_begin,
        .stage = provider_persistent_stage,
        .prepare_commit = provider_persistent_prepare_commit,
        .publish_commit = provider_persistent_publish_commit,
        .cancel_commit = provider_persistent_cancel_commit,
        .commit = provider_persistent_commit,
        .abort = provider_persistent_abort,
        .reset = provider_persistent_reset,
        .invalidate = provider_persistent_invalidate,
        .release = provider_persistent_release,
    };
    yvex_error_clear(err);
    return YVEX_OK;
}
