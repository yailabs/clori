/*
 * Retain recipe-bound persistent state with transaction-private candidate banks.
 * Runtime supplies optional backend residency without changing logical publication.
 */
#include <yvex/internal/graph_state.h>
#include <yvex/internal/candidate.h>
#include <yvex/internal/core.h>
#include "src/graph/private.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef yvex_graph_state_component_storage state_component_storage;
typedef struct {
    yvex_attention_history_view view;
    state_component_storage components[YVEX_ATTENTION_STATE_BINDING_COUNT];
    char state_identity[YVEX_SHA256_HEX_CAP];
} attention_state_bank;
typedef struct {
    yvex_attention_layer_plan plan;
    yvex_attention_state_recipe recipe;
    attention_state_bank bank[2];
    unsigned int committed_bank;
    int prepared, staged, banks_synchronized;
    yvex_attention_candidate_delta **candidate_deltas;
    unsigned long long candidate_delta_count, candidate_delta_capacity;
    unsigned long long candidate_delta_limit, candidate_delta_bytes;
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
    int prefix_selected;
    unsigned long long selected_prefix_count, extension_token_count;
    yvex_attention_cancellation cancellation;
    int cancellation_bound;
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char prepared_content_identity[YVEX_SHA256_HEX_CAP];
    attention_state_delta delta;
} attention_state_transaction;
typedef yvex_graph_state_history_span state_history_span;
static int state_hash_u64s(yvex_sha256 *hash, const unsigned long long *values,
                           size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}
typedef struct {
    const yvex_graph_family_api *family;
    const yvex_attention_plan *plan;
    attention_layer_state *layers;
    unsigned long long layer_count, maximum_host_bytes;
    yvex_graph_state_page_pool *page_pool;
    yvex_execution_capacity_plan capacity_plan;
    yvex_graph_attention_state_summary summary;
    attention_state_transaction transaction;
    pthread_mutex_t mutex;
    int mutex_ready, paging_configured;
} attention_state;
static const yvex_graph_attention_state_summary initial_state_summary = {
    .schema_version = YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V4,
    .sealed = 1, .persistent = 1, .position_consistent = 1,
    .generation = 1ull};
static void state_close(attention_state **state_ptr);
#define state_rolling_view yvex_graph_state_rolling_view
static state_history_span state_history_project(
    const yvex_attention_history_view *view,
    const yvex_attention_state_component_recipe *component)
{
    state_history_span span = {0};
    (void)yvex_graph_state_history_project(view, component, &span);
    return span;
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
static void state_bank_release(attention_state_bank *bank) {
    if (!bank) return;
    yvex_graph_state_bank_pages_close(bank->components);
    memset(bank, 0, sizeof(*bank));
}
static int state_bank_initial_identity(attention_state_bank *bank,
                                       const attention_layer_state *layer,
                                       const char *plan_identity);
static int state_bank_reset(attention_state_bank *bank,
                            const attention_layer_state *layer,
                            const char *plan_identity, yvex_error *err) {
    if (yvex_graph_state_bank_pages_reset(
            bank->components, &bank->view, &layer->recipe, err) != YVEX_OK)
        return 0;
    yvex_graph_state_bank_pages_bind(
        bank->components, &bank->view, &layer->recipe);
    return state_bank_initial_identity(bank, layer, plan_identity);
}
static int state_bank_open(attention_state *state,
                           attention_state_bank *bank,
                           attention_layer_state *layer,
                           yvex_attention_failure *failure, yvex_error *err) {
    yvex_error cause;
    yvex_status status;

    memset(bank, 0, sizeof(*bank));
    if (yvex_graph_state_bank_pages_open(
            state->page_pool,
            state->paging_configured ? &state->capacity_plan : NULL,
            yvex_attention_plan_summary(state->plan), &layer->plan,
            &layer->recipe, bank->components, &bank->view, err) != YVEX_OK) {
        cause = err ? *err : (yvex_error){0};
        status = yvex_error_is_set(&cause)
                     ? (yvex_status)yvex_error_code(&cause)
                     : YVEX_ERR_NOMEM;
        if (failure) {
            memset(failure, 0, sizeof(*failure));
            failure->code = YVEX_ATTENTION_FAILURE_STATE_DELTA;
            failure->layer_index = layer->plan.layer_index;
            failure->role = YVEX_TENSOR_ROLE_UNKNOWN;
            failure->expected = 1ull;
            failure->reason = "attention state page-store allocation failed";
        }
        yvex_error_setf(
            err, status, "graph.attention.state",
            "attention state page-store allocation failed (layer=%llu): %s",
            layer->plan.layer_index,
            yvex_error_is_set(&cause) ? yvex_error_message(&cause)
                                      : "allocation owner returned no detail");
        return status;
    }
    yvex_graph_state_bank_pages_bind(
        bank->components, &bank->view, &layer->recipe);
    return YVEX_OK;
}
static void state_bank_rolling_copy(attention_state_bank *destination,
                                    const attention_state_bank *source,
                                    const attention_layer_state *layer)
{
    unsigned int component;
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_component_storage *target;
        const state_component_storage *origin;
        yvex_attention_rolling_state_view *target_view;
        const yvex_attention_rolling_state_view *source_view;
        if (recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_ROLLING) continue;
        target = &destination->components[recipe->binding];
        origin = &source->components[recipe->binding];
        target_view = (yvex_attention_rolling_state_view *)state_rolling_view(
            &destination->view, recipe->binding);
        source_view = state_rolling_view(&source->view, recipe->binding);
        memcpy(target->values, origin->values,
               (size_t)source_view->kv_state_extent * sizeof(*target->values));
        memcpy(target->auxiliary, origin->auxiliary,
               (size_t)source_view->score_state_extent * sizeof(*target->auxiliary));
        *target_view = *source_view;
    }
}
static unsigned long long state_candidate_delta_tokens(
    const attention_layer_state *layer)
{
    unsigned int index;
    unsigned long long count = 0ull;
    for (index = 0u; index < layer->candidate_delta_count; ++index)
        count += yvex_attention_candidate_delta_token_count(
            layer->candidate_deltas[index]);
    return count;
}
static void state_bank_restore_candidate(
    attention_state_bank *candidate, const attention_state_bank *committed,
    attention_layer_state *layer)
{
    unsigned long long added = state_candidate_delta_tokens(layer);
    unsigned int component;
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        state_component_storage *target = &candidate->components[recipe->binding];
        const state_component_storage *source =
            &committed->components[recipe->binding];
        unsigned long long row, count, start;
        if (recipe->kind != YVEX_ATTENTION_STATE_COMPONENT_HISTORY ||
            recipe->binding != YVEX_ATTENTION_STATE_BINDING_LOCAL_HISTORY)
            continue;
        count = committed->view.local_tail_count;
        start = source->start;
        for (row = 0ull; row < added; ++row) {
            unsigned long long slot;
            if (count != recipe->capacity) {
                slot = (start + count) % recipe->capacity;
                ++count;
                continue;
            }
            /* Only overwritten committed ring slots require rollback. */
            start = (start + 1ull) % recipe->capacity;
            slot = (start + recipe->capacity - 1ull) % recipe->capacity;
            memcpy(target->values + slot * recipe->value_width,
                   source->values + slot * recipe->value_width,
                   (size_t)recipe->value_width * sizeof(*target->values));
            memcpy(target->values + (slot + recipe->capacity) * recipe->value_width,
                   source->values + (slot + recipe->capacity) * recipe->value_width,
                   (size_t)recipe->value_width * sizeof(*target->values));
            target->positions[slot] = source->positions[slot];
            target->positions[slot + recipe->capacity] =
                source->positions[slot + recipe->capacity];
        }
    }
    candidate->view = committed->view;
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        candidate->components[recipe->binding].start =
            committed->components[recipe->binding].start;
    }
    state_bank_rolling_copy(candidate, committed, layer);
    yvex_graph_state_bank_pages_bind(
        candidate->components, &candidate->view, &layer->recipe);
    memcpy(candidate->state_identity, committed->state_identity,
           sizeof(candidate->state_identity));
    layer->banks_synchronized = 1;
}
static int state_bank_initial_identity(attention_state_bank *bank,
                                       const attention_layer_state *layer,
                                       const char *plan_identity) {
    return yvex_graph_state_initial_identity(
        &bank->view, &layer->recipe, plan_identity, bank->state_identity);
}
static int state_bank_advance_identity(
    attention_state_bank *bank, const attention_layer_state *layer,
    const char *plan_identity, const yvex_attention_publication *publication)
{
    yvex_attention_publication adjusted = *publication;
    if (!publication->token_ids)
        adjusted.token_position = bank->view.token_count - publication->token_count;
    return yvex_graph_state_advance_identity(
        bank->state_identity, &layer->recipe, plan_identity, &adjusted,
        bank->state_identity);
}
/* Layout identity covers geometry but never mutable history values or addresses. */
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
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.state-layout.v3") ||
        !yvex_sha256_update_u64(&hash, YVEX_GRAPH_ATTENTION_STATE_SCHEMA_V4) ||
        !yvex_sha256_update_text(&hash, summary->attention_plan_identity) ||
        !yvex_sha256_update_text(
            &hash, state->paging_configured
                       ? state->capacity_plan.identity
                       : "unpaged-reference") ||
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
static int state_history_prepare_deltas(
    attention_state_bank *bank, const attention_layer_state *layer,
    yvex_error *err)
{
    yvex_attention_publication aggregate = {0};
    const yvex_attention_publication *single[1] = {&aggregate};
    unsigned long long index;
    for (index = 0ull; index < layer->candidate_delta_count; ++index) {
        const yvex_attention_publication *publication =
            yvex_attention_candidate_delta_publication(
                layer->candidate_deltas[index]);
        if (!publication ||
            !yvex_core_u64_add(aggregate.token_count,
                               publication->token_count,
                               &aggregate.token_count) ||
            !yvex_core_u64_add(aggregate.compressed_count,
                               publication->compressed_count,
                               &aggregate.compressed_count) ||
            !yvex_core_u64_add(aggregate.indexer_count,
                               publication->indexer_count,
                               &aggregate.indexer_count))
            return 0;
        if (publication->compressed_positions)
            aggregate.compressed_positions =
                (unsigned long long *)(uintptr_t)1u;
        if (publication->indexer_positions)
            aggregate.indexer_positions =
                (unsigned long long *)(uintptr_t)1u;
    }
    return yvex_graph_state_pages_prepare_publications(
               &bank->view, bank->components, &layer->recipe,
               single, 1ull, err) == YVEX_OK;
}

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
            unsigned long long slot = *count;
            if (!span.positions && *count == recipe->capacity) {
                storage->start = (storage->start + 1ull) % recipe->capacity;
                slot = (storage->start + recipe->capacity - 1ull) %
                       recipe->capacity;
            } else {
                slot = (storage->start + slot) % recipe->capacity;
                ++*count;
            }
            memcpy(storage->values + slot * recipe->value_width,
                   span.values + row * span.width,
                   (size_t)recipe->value_width * sizeof(float));
            storage->positions[slot] = span.positions
                                           ? span.positions[row]
                                           : publication->token_position + row;
            if (!span.positions) {
                memcpy(storage->values +
                           (slot + recipe->capacity) * recipe->value_width,
                       storage->values + slot * recipe->value_width,
                       (size_t)recipe->value_width * sizeof(float));
                storage->positions[slot + recipe->capacity] =
                    storage->positions[slot];
                bank->view.local_kv = storage->values +
                                      storage->start * recipe->value_width;
                bank->view.local_positions = storage->positions + storage->start;
            }
        }
    }
}
static int state_bank_apply_publication(
    attention_state *state, attention_layer_state *layer,
    attention_state_bank *candidate,
    const yvex_attention_publication *publication,
    yvex_attention_failure *failure, yvex_error *err)
{
    unsigned int component;
    const yvex_attention_publication *publications[1] = {publication};
    if (yvex_graph_state_pages_prepare_publications(
            &candidate->view, candidate->components, &layer->recipe,
            publications, 1ull, err) != YVEX_OK)
        return state_reject(
            failure, layer->plan.layer_index, 1ull, 0ull,
            "attention candidate pages could not be committed",
            yvex_error_is_set(err) ? (yvex_status)yvex_error_code(err)
                                   : YVEX_ERR_BOUNDS,
            err);
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
            !state_rolling_apply(
                (yvex_attention_rolling_state_view *)state_rolling_view(
                    &candidate->view, recipe->binding),
                storage->values, storage->auxiliary, output))
            return state_reject(
                failure, layer->plan.layer_index, 1ull, 0ull,
                "attention rolling state delta is incomplete",
                YVEX_ERR_FORMAT, err);
    }
    candidate->view.token_count += publication->token_count;
    {
        int rc = state->family->history_validate(
            &layer->plan, &candidate->view, failure, err);
        if (rc != YVEX_OK) return rc;
    }
    if (!state_bank_advance_identity(
            candidate, layer,
            yvex_attention_plan_summary(state->plan)->attention_plan_identity,
            publication))
        return state_reject(
            failure, layer->plan.layer_index, 1ull, 0ull,
            "attention candidate state identity failed", YVEX_ERR_STATE, err);
    return YVEX_OK;
}
static int state_layer_delta_projection_validate(
    const attention_layer_state *layer, const attention_state_bank *committed,
    yvex_error *err)
{
    unsigned int index;
    unsigned long long expected = committed->view.token_count;
    for (index = 0u; index < layer->candidate_delta_count; ++index) {
        const yvex_attention_publication *publication =
            yvex_attention_candidate_delta_publication(
                layer->candidate_deltas[index]);
        if (!publication || !publication->token_count ||
            publication->token_position != expected ||
            !yvex_core_u64_add(expected, publication->token_count, &expected))
            return 0;
    }
    yvex_error_clear(err);
    return 1;
}
static void state_bank_mirror_committed(attention_layer_state *layer)
{
    attention_state_bank *committed = &layer->bank[layer->committed_bank];
    attention_state_bank *candidate = &layer->bank[1u - layer->committed_bank];
    unsigned int component;
    unsigned long long index;
    for (index = 0u; index < layer->candidate_delta_count; ++index) {
        const yvex_attention_publication *publication =
            yvex_attention_candidate_delta_publication(
                layer->candidate_deltas[index]);
        state_history_append(committed, layer, publication);
        committed->view.token_count += publication->token_count;
    }
    for (component = 0u; component < layer->recipe.component_count; ++component) {
        const yvex_attention_state_component_recipe *recipe =
            &layer->recipe.components[component];
        if (recipe->kind == YVEX_ATTENTION_STATE_COMPONENT_HISTORY)
            committed->components[recipe->binding].start =
                candidate->components[recipe->binding].start;
    }
    state_bank_rolling_copy(committed, candidate, layer);
    yvex_graph_state_bank_pages_bind(
        committed->components, &committed->view, &layer->recipe);
    memcpy(committed->state_identity, candidate->state_identity,
           sizeof(committed->state_identity));
    layer->banks_synchronized = 1;
}
/* Naming a candidate delta does not make its state visible. */
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
    if (yvex_graph_state_page_pool_open(
            &state->page_pool, maximum_host_bytes, err) != YVEX_OK) {
        state_close(&state);
        return state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, maximum_host_bytes, 0ull,
            "attention state page-pool initialization failed",
            yvex_error_is_set(err) ? (yvex_status)yvex_error_code(err)
                                   : YVEX_ERR_NOMEM,
            err);
    }
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
static int state_configure_pages(
    attention_state *state, const yvex_execution_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err)
{
    char layout_identity[YVEX_SHA256_HEX_CAP];
    char content_identity[YVEX_SHA256_HEX_CAP];
    unsigned long long generation;
    int rc = state_enter(
        state, YVEX_ATTENTION_NO_LAYER, capacity ? capacity->state_class_count : 0ull,
        "attention state paging configuration is invalid", failure, err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_graph_state_capacity_plan_valid(capacity)) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
            "sealed execution capacity plan is required for state paging",
            YVEX_ERR_INVALID_ARG, err);
        goto done;
    }
    if (state->paging_configured &&
        strcmp(state->capacity_plan.identity, capacity->identity) == 0)
        goto done;
    if (state->paging_configured || state->summary.prepared_layer_count ||
        state->transaction.active) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, 0ull, 1ull,
            "state page geometry cannot change after storage admission",
            YVEX_ERR_STATE, err);
        goto done;
    }
    if (!yvex_core_u64_add(state->summary.generation, 1ull, &generation)) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, ULLONG_MAX, 1ull,
            "state page configuration generation overflowed",
            YVEX_ERR_BOUNDS, err);
        goto done;
    }
    state->capacity_plan = *capacity;
    state->paging_configured = 1;
    if (!state_layout_identity(
            state, ULLONG_MAX, NULL, layout_identity) ||
        !state_content_identity(
            state, ULLONG_MAX, NULL, 0, layout_identity,
            content_identity)) {
        memset(&state->capacity_plan, 0, sizeof(state->capacity_plan));
        state->paging_configured = 0;
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
            "state page layout identity construction failed",
            YVEX_ERR_STATE, err);
        goto done;
    }
    state->paging_configured = 0;
    memset(&state->capacity_plan, 0, sizeof(state->capacity_plan));
    if (yvex_graph_state_page_pool_bind_capacity(
            state->page_pool, capacity, err) != YVEX_OK) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER,
            capacity->state_pool_bytes, 0ull,
            "state page-pool budget could not be admitted",
            (yvex_status)yvex_error_code(err), err);
        goto done;
    }
    state->capacity_plan = *capacity;
    state->paging_configured = 1;
    state->summary.generation = generation;
    state->summary.paged = 1;
    state->summary.paging_configured = 1;
    yvex_core_text_copy(
        state->summary.capacity_plan_identity,
        sizeof(state->summary.capacity_plan_identity), capacity->identity);
    yvex_core_text_copy(
        state->summary.state_layout_identity,
        sizeof(state->summary.state_layout_identity), layout_identity);
    yvex_core_text_copy(
        state->summary.state_content_identity,
        sizeof(state->summary.state_content_identity), content_identity);
done:
    return state_unlock_result(state, rc, failure, err);
}
static int state_prepare(
    attention_state *state, unsigned long long layer_index,
    const yvex_attention_state_recipe *recipe,
    const yvex_attention_history_view *initial_history,
    yvex_attention_failure *failure, yvex_error *err) {
    const yvex_attention_summary *summary;
    attention_layer_state candidate;
    unsigned long long bank;
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
    candidate.candidate_delta_limit =
        recipe->final_position - recipe->initial_position;
    if (!candidate.candidate_delta_limit) {
        rc = state_reject(
            failure, layer_index, 1ull, 0ull,
            "attention state candidate range is empty", YVEX_ERR_BOUNDS, err);
        goto done;
    }
    for (bank = 0ull; bank < 2ull && rc == YVEX_OK; ++bank) {
        rc = state_bank_open(state, &candidate.bank[bank], &candidate,
                             failure, err);
        if (rc == YVEX_OK && initial_history &&
            yvex_graph_state_bank_pages_transfer(
                candidate.bank[bank].components,
                &candidate.bank[bank].view, &candidate.recipe,
                initial_history, 1, err) != YVEX_OK)
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "initial attention history could not be copied", YVEX_ERR_FORMAT, err);
    }
    candidate.prepared = rc == YVEX_OK;
    for (bank = 0ull; bank < 2ull && rc == YVEX_OK; ++bank)
        if (!state_bank_initial_identity(&candidate.bank[bank], &candidate,
                                         summary->attention_plan_identity))
            rc = state_reject(failure, layer_index, 1ull, 0ull,
                              "initial attention state identity failed", YVEX_ERR_STATE, err);
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
        candidate.banks_synchronized = 1;
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
        free(candidate.candidate_deltas);
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
                state->transaction.prefix_selected ||
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
static int state_summary_copy(
    const attention_state *state,
    yvex_graph_attention_state_summary *out, yvex_error *err) {
    attention_state *mutable_state = (attention_state *)state;
    yvex_graph_state_page_summary pages;
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
    if (yvex_graph_state_page_pool_summary(
            state->page_pool, &pages, err) != YVEX_OK) {
        memset(out, 0, sizeof(*out));
        return state_unlock_result(
            mutable_state, (int)yvex_error_code(err), NULL, err);
    }
    out->allocated_bytes = pages.allocated_bytes;
    out->virtual_bytes = pages.virtual_bytes;
    out->resident_bytes = pages.resident_bytes;
    out->page_table_bytes = pages.metadata_bytes;
    out->page_count = pages.page_count;
    out->resident_page_count = pages.resident_page_count;
    out->page_commit_count = pages.page_commit_count;
    out->page_release_count = pages.page_release_count;
    out->transaction_active = state->transaction.active;
    out->candidate_active = state->transaction.candidate_active;
    out->abort_required = state->transaction.failed;
    out->staged_layer_count = state->transaction.staged_count;
    out->prefix_selected = state->transaction.prefix_selected;
    out->extension_ready = state->transaction.prefix_selected &&
                           state->transaction.extension_token_count != 0ull;
    out->selected_prefix_count = state->transaction.selected_prefix_count;
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
    } else if (out->extension_ready &&
               yvex_core_u64_add(state->transaction.batch_position,
                                 state->transaction.selected_prefix_count,
                                 &out->staged_next_position) &&
               yvex_core_u64_add(state->summary.generation, 1ull,
                                 &out->staged_generation)) {
        out->staged_state_content_identity[0] = '\0';
    }
    return state_unlock_result(mutable_state, YVEX_OK, NULL, err);
}
static void state_candidate_deltas_close(attention_state *state)
{
    unsigned long long index;
    unsigned long long delta;
    if (!state) return;
    for (index = 0ull; index < state->layer_count; ++index) {
        for (delta = 0u;
             delta < state->layers[index].candidate_delta_count; ++delta)
            yvex_attention_candidate_delta_close(
                &state->layers[index].candidate_deltas[delta]);
        state->layers[index].candidate_delta_count = 0u;
    }
}
static void state_candidate_clear(attention_state_transaction *transaction) {
    transaction->layer = NULL;
    transaction->layer_ordinal = transaction->token_position =
        transaction->token_count = transaction->applied_tokens = 0ull;
    transaction->candidate_active = 0;
    memset(&transaction->delta, 0, sizeof(transaction->delta));
}
/* The provider-wide batch remains private until complete graph publication. */
static int state_begin(
    attention_state *state, unsigned long long layer_index,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_layer_state *layer;
    attention_state_bank *committed, *candidate;
    int extending;
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
    extending = state->transaction.active && state->transaction.prefix_selected;
    if (!layer->prepared || state->transaction.candidate_active ||
        state->transaction.failed || layer->staged || state->summary.cancelled ||
        state->summary.invalidated || !state->summary.position_consistent ||
        (!extending && !layer->banks_synchronized) ||
        (!extending && layer->candidate_delta_count)) {
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
                              : !layer->banks_synchronized
                                    ? "attention state banks are not synchronized"
                              : layer->candidate_delta_count
                                    ? "attention state prefix selection is unresolved"
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
         ((!extending &&
           (state->transaction.batch_position != token_position ||
            state->transaction.batch_token_count != token_count)) ||
          (extending &&
           (token_position != state->transaction.batch_position +
                                  state->transaction.selected_prefix_count ||
            token_count != state->transaction.extension_token_count))))) {
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
    candidate = &layer->bank[1u - layer->committed_bank];
    if ((!extending && committed->view.token_count != token_position) ||
        (extending && candidate->view.token_count != token_position)) {
        rc = state_reject(
            failure, layer_index,
            extending ? candidate->view.token_count : committed->view.token_count,
            token_position, "attention state position is not contiguous",
            YVEX_ERR_STATE, err);
        goto done;
    }
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
static void state_publish_prepared(attention_state *state) {
    attention_state_transaction *transaction;
    unsigned long long index;
    if (!state) return;
    transaction = &state->transaction;
    if (!transaction->publication_prepared) return;
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        if (!layer->staged) continue;
        state_bank_mirror_committed(layer);
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
    state_candidate_deltas_close(state);
    memset(transaction, 0, sizeof(*transaction));
    (void)pthread_mutex_unlock(&state->mutex);
}
static int state_candidate_delta_reserve(
    attention_state *state, attention_layer_state *layer, yvex_error *err)
{
    if (layer->candidate_delta_count < layer->candidate_delta_capacity)
        return 1;
    return yvex_graph_state_pointer_table_reserve(
               state->page_pool, (void ***)&layer->candidate_deltas,
               &layer->candidate_delta_capacity,
               &layer->candidate_delta_bytes,
               layer->candidate_delta_limit, err) == YVEX_OK;
}
static int state_apply(
    attention_state *state,
    const yvex_attention_publication *publication,
    const yvex_attention_cancellation *cancellation,
    char delta_identity_output[YVEX_SHA256_HEX_CAP],
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state_transaction *transaction;
    attention_layer_state *layer;
    attention_state_bank *committed, *candidate;
    yvex_attention_candidate_delta *candidate_delta = NULL;
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
    committed = &layer->bank[layer->committed_bank];
    candidate = &layer->bank[1u - layer->committed_bank];
    if (!state_candidate_delta_reserve(state, layer, err) ||
        yvex_attention_candidate_delta_open(
            &candidate_delta, publication, err) != YVEX_OK) {
        transaction->failed = 1;
        rc = state_reject(failure, transaction->layer_ordinal, 1ull, 0ull,
                          "bounded attention state delta could not be retained",
                          (yvex_status)yvex_error_code(err), err);
        goto done;
    }
    layer->candidate_deltas[layer->candidate_delta_count++] = candidate_delta;
    candidate_delta = NULL;
    rc = state_bank_apply_publication(
        state, layer, candidate, publication, failure, err);
    if (rc != YVEX_OK) {
        transaction->failed = 1;
        state_bank_restore_candidate(candidate, committed, layer);
        goto done;
    }
    layer->banks_synchronized = 0;
    transaction->applied_tokens += publication->token_count;
    if (!state_delta_identity(transaction)) {
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
    yvex_attention_candidate_delta_close(&candidate_delta);
    return state_transaction_result(state, rc, failure, err);
}
static int state_select_prefix(
    attention_state *state, unsigned long long prefix_count,
    unsigned long long extension_count,
    yvex_attention_failure *failure, yvex_error *err)
{
    attention_state_transaction *transaction;
    unsigned long long index, staged = 0ull, final_count;
    int rc = YVEX_OK;
    rc = state_enter(state, YVEX_ATTENTION_NO_LAYER, prefix_count,
                     "attention candidate prefix owner is required", failure, err);
    if (rc != YVEX_OK) return rc;
    transaction = &state->transaction;
    if (!yvex_core_u64_add(prefix_count, extension_count, &final_count) ||
        !transaction->active || transaction->candidate_active ||
        transaction->failed || transaction->prefix_selected ||
        !prefix_count || prefix_count > transaction->batch_token_count ||
        extension_count > 1ull ||
        final_count > state->summary.capacity ||
        transaction->batch_position > state->summary.capacity - final_count) {
        transaction->failed = transaction->active;
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER, transaction->batch_token_count,
            final_count, "accepted attention prefix is not admissible",
            YVEX_ERR_STATE, err);
        goto done;
    }
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        if (layer->staged) ++staged;
        if (layer->staged && layer->candidate_delta_count != 1u) {
            rc = state_reject(
                failure, index, 1ull, 0ull,
                "staged attention layer has no prefix-addressable delta",
                YVEX_ERR_STATE, err);
            goto failed;
        }
    }
    if (staged != state->summary.prepared_layer_count ||
        staged != transaction->staged_count) {
        rc = state_reject(
            failure, YVEX_ATTENTION_NO_LAYER,
            state->summary.prepared_layer_count, staged,
            "attention prefix selection requires a complete target stack",
            YVEX_ERR_STATE, err);
        goto failed;
    }
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        attention_state_bank *committed = &layer->bank[layer->committed_bank];
        attention_state_bank *candidate = &layer->bank[1u - layer->committed_bank];
        yvex_attention_candidate_delta *selected_delta = NULL;
        yvex_attention_publication publication;
        if (!layer->staged) continue;
        rc = yvex_attention_candidate_delta_project(
            layer->candidate_deltas[0], &committed->view, prefix_count,
            &publication, err);
        if (rc != YVEX_OK) goto failed;
        if (yvex_attention_candidate_delta_open(
                &selected_delta, &publication, err) != YVEX_OK) {
            rc = yvex_error_code(err);
            goto failed;
        }
        state_bank_restore_candidate(candidate, committed, layer);
        rc = state_bank_apply_publication(
            state, layer, candidate, &publication, failure, err);
        if (rc != YVEX_OK) {
            yvex_attention_candidate_delta_close(&selected_delta);
            state_bank_restore_candidate(candidate, committed, layer);
            goto failed;
        }
        yvex_attention_candidate_delta_close(&layer->candidate_deltas[0]);
        layer->candidate_delta_count = 0u;
        layer->candidate_deltas[layer->candidate_delta_count++] = selected_delta;
        layer->banks_synchronized = 0;
        layer->staged = extension_count == 0ull;
    }
    transaction->prefix_selected = 1;
    transaction->selected_prefix_count = prefix_count;
    transaction->extension_token_count = extension_count;
    transaction->batch_token_count = final_count;
    transaction->staged_count = extension_count ? 0ull : staged;
    yvex_error_clear(err);
    goto done;
failed:
    transaction->failed = 1;
done:
    return state_transaction_result(state, rc, failure, err);
}
/* Retain the lock while every participant preflights the same publication. */
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
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        if (!layer->staged) continue;
        ++staged;
        if (!layer->candidate_delta_count ||
            !state_layer_delta_projection_validate(
                layer, &layer->bank[layer->committed_bank], err) ||
            !state_history_prepare_deltas(
                &layer->bank[layer->committed_bank], layer, err)) {
            rc = state_reject(
                failure, index, 1ull, layer->candidate_delta_count,
                "attention state delta cannot preflight its committed pages",
                yvex_error_is_set(err) ? (yvex_status)yvex_error_code(err)
                                       : YVEX_ERR_STATE,
                err);
            goto done;
        }
    }
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
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        if (!layer->banks_synchronized)
            state_bank_restore_candidate(
                &layer->bank[1u - layer->committed_bank],
                &layer->bank[layer->committed_bank], layer);
        layer->staged = 0;
    }
    state_candidate_deltas_close(state);
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
                                  plan_summary->attention_plan_identity,
                                  err)) {
                state->summary.invalidated = 1;
                rc = state_reject(failure, index, 1ull, 0ull,
                                  "attention state reset identity failed", YVEX_ERR_STATE, err);
                goto done;
            }
        layer->committed_bank = 0u;
        layer->staged = 0;
        layer->banks_synchronized = 1;
        {
            unsigned long long delta;
            for (delta = 0u; delta < layer->candidate_delta_count; ++delta)
                yvex_attention_candidate_delta_close(
                    &layer->candidate_deltas[delta]);
            layer->candidate_delta_count = 0u;
        }
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
static int state_restore(
    attention_state *state, const yvex_attention_state_checkpoint *checkpoint,
    yvex_attention_failure *failure, yvex_error *err)
{
    unsigned long long index, generation;
    char content_identity[YVEX_SHA256_HEX_CAP];
    int rc;
    rc = state_enter(state, YVEX_ATTENTION_NO_LAYER,
                     checkpoint ? checkpoint->layer_count : 0ull,
                     "attention state checkpoint is invalid", failure, err);
    if (rc != YVEX_OK) return rc;
    if (!checkpoint ||
        checkpoint->schema_version != YVEX_ATTENTION_STATE_CHECKPOINT_SCHEMA_V1 ||
        checkpoint->layer_count != state->layer_count || !checkpoint->layers ||
        !checkpoint->layer_identities || state->transaction.active ||
        state->summary.invalidated || state->summary.cancelled ||
        state->summary.prepared_layer_count != state->layer_count ||
        checkpoint->committed_sequence_length > state->summary.capacity ||
        strcmp(checkpoint->state_layout_identity,
               state->summary.state_layout_identity) != 0 ||
        strcmp(checkpoint->capacity_plan_identity,
               state->summary.capacity_plan_identity) != 0 ||
        !yvex_sha256_hex_valid(checkpoint->state_content_identity)) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, state->layer_count,
                          checkpoint ? checkpoint->layer_count : 0ull,
                          "state checkpoint is incompatible with this provider",
                          YVEX_ERR_FORMAT, err);
        goto done;
    }
    if (!yvex_core_u64_add(state->summary.generation, 1ull, &generation)) {
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, ULLONG_MAX, 1ull,
                          "checkpoint restore generation overflowed",
                          YVEX_ERR_BOUNDS, err);
        goto done;
    }
    for (index = 0ull; index < state->layer_count; ++index) {
        attention_layer_state *layer = &state->layers[index];
        attention_state_bank *candidate = &layer->bank[1u - layer->committed_bank];
        const yvex_attention_history_view *view = &checkpoint->layers[index];
        rc = view->token_count == checkpoint->committed_sequence_length
                 ? state->family->history_validate(&layer->plan, view, failure, err)
                 : YVEX_ERR_FORMAT;
        if (rc == YVEX_OK)
            rc = yvex_graph_state_bank_pages_transfer(
                candidate->components, &candidate->view, &layer->recipe,
                view, 1, err);
        if (rc == YVEX_OK &&
            !yvex_sha256_hex_valid(checkpoint->layer_identities[index]))
            rc = YVEX_ERR_FORMAT;
        if (rc != YVEX_OK) {
            unsigned long long dirty;
            for (dirty = 0ull; dirty <= index; ++dirty)
                state->layers[dirty].banks_synchronized = 0;
            if (!yvex_error_is_set(err))
                yvex_error_set(err, YVEX_ERR_FORMAT,
                               "graph.attention.state.restore",
                               "checkpoint layer identity is invalid");
            break;
        }
        yvex_core_text_copy(candidate->state_identity,
                            sizeof(candidate->state_identity),
                            checkpoint->layer_identities[index]);
    }
    if (rc != YVEX_OK) goto done;
    for (index = 0ull; index < state->layer_count; ++index)
        state->layers[index].committed_bank =
            1u - state->layers[index].committed_bank;
    if (!state_content_identity(state, ULLONG_MAX, NULL, 0,
                                state->summary.state_layout_identity,
                                content_identity) ||
        strcmp(content_identity, checkpoint->state_content_identity) != 0) {
        for (index = 0ull; index < state->layer_count; ++index)
            state->layers[index].committed_bank =
                1u - state->layers[index].committed_bank;
        for (index = 0ull; index < state->layer_count; ++index)
            state->layers[index].banks_synchronized = 0;
        rc = state_reject(failure, YVEX_ATTENTION_NO_LAYER, 1ull, 0ull,
                          "checkpoint aggregate identity is invalid",
                          YVEX_ERR_FORMAT, err);
        goto done;
    }
    state_candidate_deltas_close(state);
    memset(&state->transaction, 0, sizeof(state->transaction));
    for (index = 0ull; index < state->layer_count; ++index)
        state->layers[index].banks_synchronized = 0;
    state->summary.generation = generation;
    state->summary.committed_sequence_length =
        checkpoint->committed_sequence_length;
    state->summary.next_position = checkpoint->committed_sequence_length;
    state->summary.position_consistent = 1;
    yvex_core_text_copy(state->summary.state_content_identity,
                        sizeof(state->summary.state_content_identity),
                        content_identity);
done:
    return state_unlock_result(state, rc, failure, err);
}
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
    state_candidate_deltas_close(state);
    return state_unlock_result(state, YVEX_OK, NULL, err);
}
static void state_close(attention_state **state_ptr) {
    attention_state *state;
    unsigned long long layer, bank;
    if (!state_ptr || !*state_ptr) return;
    state = *state_ptr;
    if (state->mutex_ready && pthread_mutex_lock(&state->mutex) != 0) return;
    *state_ptr = NULL;
    for (layer = 0ull; layer < state->layer_count; ++layer) {
        unsigned long long delta;
        for (delta = 0u;
             delta < state->layers[layer].candidate_delta_count; ++delta)
            yvex_attention_candidate_delta_close(
                &state->layers[layer].candidate_deltas[delta]);
        for (bank = 0ull; bank < 2ull; ++bank)
            state_bank_release(&state->layers[layer].bank[bank]);
        yvex_graph_state_page_pool_release(
            state->page_pool, state->layers[layer].candidate_delta_bytes);
        free(state->layers[layer].candidate_deltas);
    }
    free(state->layers);
    yvex_graph_state_page_pool_close(&state->page_pool);
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
static int provider_persistent_configure_pages(
    void *context, const yvex_execution_capacity_plan *capacity,
    yvex_attention_failure *failure, yvex_error *err)
{
    return state_configure_pages(
        (attention_state *)context, capacity, failure, err);
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
static int provider_persistent_begin(void *context, unsigned long long layer_ordinal,
    const yvex_attention_layer_plan *layer,
    const yvex_attention_history_view *initial_history,
    unsigned long long token_position, unsigned long long token_count,
    const yvex_attention_cancellation *cancellation,
    const yvex_attention_history_view **history,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_state *state = (attention_state *)context;
    int rc;
    if (history) *history = NULL;
    if (!state || !layer || !history || !token_count) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.state.provider.begin",
                       "valid provider, layer, history output, and token range are required");
        return YVEX_ERR_INVALID_ARG;
    }
    (void)initial_history;
    rc = state_begin(state, layer_ordinal, token_position, token_count, cancellation,
                     failure, err);
    if (rc == YVEX_OK)
        *history = state_view(
            state, layer_ordinal, YVEX_ATTENTION_STATE_VIEW_CANDIDATE);
    if (rc == YVEX_OK && !*history) {
        yvex_error_set(err, YVEX_ERR_STATE, "graph.state.provider.begin",
                       "candidate attention state is unavailable");
        rc = YVEX_ERR_STATE;
    }
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
static int provider_persistent_select_prefix(
    void *context, unsigned long long prefix_count,
    unsigned long long extension_count, yvex_attention_failure *failure,
    yvex_error *err)
{
    return state_select_prefix((attention_state *)context, prefix_count,
                               extension_count, failure, err);
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
static int provider_persistent_restore(
    void *context, const yvex_attention_state_checkpoint *checkpoint,
    yvex_attention_failure *failure, yvex_error *err)
{
    return state_restore((attention_state *)context, checkpoint, failure, err);
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
        .schema_version = YVEX_ATTENTION_STATE_PROVIDER_SCHEMA_V6,
        .context = state,
        .configure_pages = provider_persistent_configure_pages,
        .prepare = provider_persistent_prepare,
        .summary = provider_persistent_summary,
        .view = provider_persistent_view,
        .identity = provider_persistent_identity,
        .begin = provider_persistent_begin,
        .stage = provider_persistent_stage,
        .select_prefix = provider_persistent_select_prefix,
        .prepare_commit = provider_persistent_prepare_commit,
        .publish_commit = provider_persistent_publish_commit,
        .cancel_commit = provider_persistent_cancel_commit,
        .commit = provider_persistent_commit,
        .abort = provider_persistent_abort,
        .reset = provider_persistent_reset,
        .restore = provider_persistent_restore,
        .invalidate = provider_persistent_invalidate,
        .release = provider_persistent_release,
    };
    yvex_error_clear(err);
    return YVEX_OK;
}
