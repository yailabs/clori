/* Coordinate graph-owned shared backing across two isolated runtime sessions. */
#include "src/runtime/private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/runtime_prefix.h>

struct yvex_runtime_session_prefix {
    unsigned int schema_version;
    yvex_attention_state_prefix *target, *draft;
    yvex_attention_state_recipe *target_recipes, *draft_recipes;
    unsigned long long target_recipe_count, draft_recipe_count;
    yvex_execution_capacity_plan target_capacity, draft_capacity;
    char runtime_model_identity[YVEX_SHA256_HEX_CAP];
    yvex_runtime_session_prefix_summary summary;
};

static int prefix_refuse(yvex_runtime_model_failure *failure,
                         yvex_status status, const char *reason,
                         yvex_error *err)
{
    yvex_runtime_private_failure_record(
        failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "session-prefix", 1ull,
        0ull, reason);
    yvex_error_set(err, status, "runtime.session-prefix", reason);
    return status;
}

static int prefix_identity(
    const yvex_runtime_session_prefix *prefix,
    yvex_runtime_session_prefix_summary *summary)
{
    yvex_attention_state_prefix_summary target = {0}, draft = {0};
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    yvex_error err;
    unsigned long long shared, mapped, references, layer;

    if (!prefix || prefix->schema_version !=
                       YVEX_RUNTIME_SESSION_PREFIX_SCHEMA_V1 ||
        !prefix->target ||
        yvex_attention_state_prefix_summary_copy(prefix->target, &target,
                                                 &err) != YVEX_OK ||
        !yvex_sha256_hex_valid(prefix->runtime_model_identity) ||
        prefix->target_recipe_count != target.layer_count ||
        (target.layer_count && !prefix->target_recipes) ||
        strcmp(target.capacity_plan_identity,
               prefix->target_capacity.identity) != 0)
        return 0;
    shared = target.shared_bytes;
    mapped = target.mapped_bytes;
    references = target.reference_count;
    if (prefix->draft) {
        if (yvex_attention_state_prefix_summary_copy(prefix->draft, &draft,
                                                     &err) != YVEX_OK ||
            target.committed_sequence_length !=
                draft.committed_sequence_length ||
            prefix->draft_recipe_count != draft.layer_count ||
            (draft.layer_count && !prefix->draft_recipes) ||
            strcmp(draft.capacity_plan_identity,
                   prefix->draft_capacity.identity) != 0 ||
            !yvex_core_u64_add(shared, draft.shared_bytes, &shared) ||
            !yvex_core_u64_add(mapped, draft.mapped_bytes, &mapped))
            return 0;
        if (draft.reference_count < references)
            references = draft.reference_count;
    }
    memset(summary, 0, sizeof(*summary));
    summary->schema_version = YVEX_RUNTIME_SESSION_PREFIX_SCHEMA_V1;
    summary->scope_count = prefix->draft ? 2ull : 1ull;
    summary->committed_sequence_length =
        target.committed_sequence_length;
    summary->shared_bytes = shared;
    summary->mapped_bytes = mapped;
    summary->reference_count = references;
    yvex_runtime_identity_copy(summary->runtime_model_identity,
                               prefix->runtime_model_identity);
    yvex_runtime_identity_copy(summary->target_prefix_identity,
                               target.prefix_identity);
    if (prefix->draft)
        yvex_runtime_identity_copy(summary->draft_prefix_identity,
                                   draft.prefix_identity);
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash,
                                 "yvex.runtime.session-prefix.v1") ||
        !yvex_sha256_update_u64(&hash, summary->schema_version) ||
        !yvex_sha256_update_u64(&hash, summary->scope_count) ||
        !yvex_sha256_update_u64(
            &hash, summary->committed_sequence_length) ||
        !yvex_sha256_update_u64(&hash, summary->shared_bytes) ||
        !yvex_sha256_update_u64(&hash, summary->mapped_bytes) ||
        !yvex_sha256_update_text(&hash, summary->runtime_model_identity) ||
        !yvex_sha256_update_text(&hash, summary->target_prefix_identity) ||
        !yvex_sha256_update_text(&hash, summary->draft_prefix_identity))
        return 0;
    for (layer = 0ull; layer < prefix->target_recipe_count; ++layer)
        if (!yvex_sha256_hex_valid(prefix->target_recipes[layer].identity) ||
            !yvex_sha256_update_text(
                &hash, prefix->target_recipes[layer].identity))
            return 0;
    for (layer = 0ull; layer < prefix->draft_recipe_count; ++layer)
        if (!yvex_sha256_hex_valid(prefix->draft_recipes[layer].identity) ||
            !yvex_sha256_update_text(
                &hash, prefix->draft_recipes[layer].identity))
            return 0;
    if (!yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, summary->prefix_identity);
    return 1;
}

static int prefix_recipes_capture(
    yvex_attention_state_provider *provider, unsigned long long layer_count,
    yvex_attention_state_recipe **recipes, yvex_error *err)
{
    unsigned long long layer;
    if (recipes) *recipes = NULL;
    if (!provider || !provider->recipe || !recipes || !layer_count ||
        layer_count > SIZE_MAX / sizeof(**recipes))
        return prefix_refuse(NULL, YVEX_ERR_STATE,
                             "source prefix recipes are unavailable", err);
    *recipes = calloc((size_t)layer_count, sizeof(**recipes));
    if (!*recipes)
        return prefix_refuse(NULL, YVEX_ERR_NOMEM,
                             "source prefix recipe allocation failed", err);
    for (layer = 0ull; layer < layer_count; ++layer) {
        const yvex_attention_state_recipe *recipe =
            provider->recipe(provider->context, layer);
        if (!recipe || !yvex_sha256_hex_valid(recipe->identity)) {
            free(*recipes);
            *recipes = NULL;
            return prefix_refuse(NULL, YVEX_ERR_STATE,
                                 "source prefix recipe is invalid", err);
        }
        (*recipes)[layer] = *recipe;
    }
    return YVEX_OK;
}

static int prefix_provider_capture(
    yvex_attention_state_provider *provider, unsigned long long maximum_bytes,
    yvex_execution_capacity_plan *capacity,
    yvex_attention_state_prefix **prefix,
    yvex_attention_state_recipe **recipes,
    unsigned long long *recipe_count,
    yvex_attention_state_prefix_summary *summary, yvex_error *err)
{
    const yvex_execution_capacity_plan *source;
    yvex_attention_failure attention_failure = {0};
    int rc;

    if (!provider || !provider->context || !provider->capacity ||
        !provider->prefix_capture || !capacity || !prefix || !recipes ||
        !recipe_count || !summary ||
        !(source = provider->capacity(provider->context)) ||
        yvex_execution_capacity_plan_validate(source, err) != YVEX_OK)
        return prefix_refuse(NULL, YVEX_ERR_STATE,
                             "paged provider capacity is unavailable", err);
    *capacity = *source;
    rc = provider->prefix_capture(provider->context, maximum_bytes, prefix,
                                  &attention_failure, err);
    if (rc == YVEX_OK)
        rc = yvex_attention_state_prefix_summary_copy(*prefix, summary, err);
    if (rc == YVEX_OK)
        rc = prefix_recipes_capture(provider, summary->layer_count, recipes,
                                    err);
    if (rc == YVEX_OK) *recipe_count = summary->layer_count;
    if (rc == YVEX_OK &&
        strcmp(summary->capacity_plan_identity, capacity->identity) != 0)
        rc = prefix_refuse(NULL, YVEX_ERR_FORMAT,
                           "captured prefix capacity identity changed", err);
    return rc;
}

int yvex_runtime_session_prefix_capture(
    yvex_runtime_execution_session *source,
    unsigned long long maximum_shared_bytes,
    yvex_runtime_session_prefix **out,
    yvex_runtime_session_prefix_summary *summary,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    yvex_runtime_session_prefix *prefix = NULL;
    yvex_attention_state_prefix_summary target = {0}, draft = {0};
    yvex_runtime_model_summary model = {0};
    unsigned long long remaining;
    int rc = YVEX_OK, draft_pristine = 0;

    if (out) *out = NULL;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!source || !out || !summary || !maximum_shared_bytes ||
        !source->lifecycle_mutex_ready ||
        pthread_mutex_lock(&source->lifecycle_mutex) != 0)
        return prefix_refuse(failure, YVEX_ERR_INVALID_ARG,
                             "idle source session and byte budget are required",
                             err);
    if (!source->summary.open || source->summary.busy || source->closing ||
        source->summary.invalidated ||
        !source->attention_state_provider_ready) {
        rc = prefix_refuse(failure, YVEX_ERR_STATE,
                           "source session cannot publish a prefix", err);
        goto done;
    }
    prefix = calloc(1u, sizeof(*prefix));
    if (!prefix) {
        rc = prefix_refuse(failure, YVEX_ERR_NOMEM,
                           "session prefix allocation failed", err);
        goto done;
    }
    prefix->schema_version = YVEX_RUNTIME_SESSION_PREFIX_SCHEMA_V1;
    rc = yvex_runtime_model_summary_copy(source->model, &model, err);
    if (rc != YVEX_OK || !yvex_sha256_hex_valid(model.runtime_model_identity)) {
        rc = prefix_refuse(failure, YVEX_ERR_STATE,
                           "source runtime model identity is unavailable", err);
        goto done;
    }
    yvex_runtime_identity_copy(prefix->runtime_model_identity,
                               model.runtime_model_identity);
    rc = prefix_provider_capture(
        &source->attention_state_provider, maximum_shared_bytes,
        &prefix->target_capacity, &prefix->target,
        &prefix->target_recipes, &prefix->target_recipe_count,
        &target, err);
    if (rc != YVEX_OK) goto done;
    if (target.shared_bytes > maximum_shared_bytes) {
        rc = prefix_refuse(failure, YVEX_ERR_BOUNDS,
                           "target prefix exceeded the shared byte budget",
                           err);
        goto done;
    }
    remaining = maximum_shared_bytes - target.shared_bytes;
    if (source->draft_attention_state_provider_ready) {
        rc = yvex_runtime_private_attention_state_pristine(
            &source->draft_attention_state_provider, &draft_pristine, err);
        if (rc != YVEX_OK) goto done;
        if (!draft_pristine) {
            if (!remaining) {
                rc = prefix_refuse(
                    failure, YVEX_ERR_BOUNDS,
                    "target prefix exhausted the shared byte budget", err);
                goto done;
            }
            rc = prefix_provider_capture(
                &source->draft_attention_state_provider, remaining,
                &prefix->draft_capacity, &prefix->draft,
                &prefix->draft_recipes, &prefix->draft_recipe_count,
                &draft, err);
            if (rc != YVEX_OK) goto done;
            if (target.committed_sequence_length !=
                draft.committed_sequence_length) {
                rc = prefix_refuse(failure, YVEX_ERR_STATE,
                                   "target and draft prefixes diverged", err);
                goto done;
            }
        }
    }
    if (!prefix_identity(prefix, &prefix->summary) ||
        prefix->summary.shared_bytes > maximum_shared_bytes) {
        rc = prefix_refuse(failure, YVEX_ERR_FORMAT,
                           "session prefix identity could not seal", err);
        goto done;
    }
    *summary = prefix->summary;
    *out = prefix;
    prefix = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
done:
    (void)pthread_mutex_unlock(&source->lifecycle_mutex);
    yvex_runtime_session_prefix_close(&prefix);
    return rc;
}

static int prefix_provider_attach(
    yvex_attention_state_provider *provider,
    const yvex_execution_capacity_plan *capacity,
    const yvex_attention_state_prefix *prefix,
    const yvex_attention_state_recipe *recipes,
    unsigned long long recipe_count, yvex_error *err)
{
    yvex_attention_failure attention_failure = {0};
    yvex_attention_state_prefix_summary prefix_summary = {0};
    yvex_graph_attention_state_summary state_summary = {0};
    unsigned long long layer;
    int rc;

    if (!provider || !provider->context || !provider->configure_pages ||
        !provider->prepare || !provider->summary ||
        !provider->prefix_attach || !capacity ||
        !prefix || !recipes || !recipe_count ||
        yvex_attention_state_prefix_summary_copy(
            prefix, &prefix_summary, err) != YVEX_OK ||
        recipe_count != prefix_summary.layer_count)
        return prefix_refuse(NULL, YVEX_ERR_STATE,
                             "destination prefix provider is unavailable", err);
    rc = provider->configure_pages(provider->context, capacity,
                                   &attention_failure, err);
    for (layer = 0ull; rc == YVEX_OK && layer < recipe_count; ++layer)
        rc = provider->prepare(provider->context, layer, &recipes[layer],
                               NULL, &attention_failure, err);
    if (rc == YVEX_OK)
        rc = provider->summary(provider->context, &state_summary, err);
    if (rc == YVEX_OK &&
        state_summary.prepared_layer_count != prefix_summary.layer_count)
        rc = prefix_refuse(NULL, YVEX_ERR_FORMAT,
                           "destination prefix layer coverage is incompatible",
                           err);
    if (rc == YVEX_OK &&
        state_summary.committed_sequence_length != recipes[0].initial_position)
        rc = prefix_refuse(NULL, YVEX_ERR_FORMAT,
                           "destination prefix preparation is not pristine",
                           err);
    if (rc == YVEX_OK &&
        strcmp(state_summary.state_layout_identity,
               prefix_summary.state_layout_identity) != 0)
        rc = prefix_refuse(NULL, YVEX_ERR_FORMAT,
                           "destination prefix state layout is incompatible",
                           err);
    if (rc == YVEX_OK &&
        strcmp(state_summary.capacity_plan_identity,
               prefix_summary.capacity_plan_identity) != 0)
        rc = prefix_refuse(NULL, YVEX_ERR_FORMAT,
                           "destination prefix capacity is incompatible", err);
    if (rc == YVEX_OK)
        rc = provider->prefix_attach(provider->context, prefix,
                                     &attention_failure, err);
    return rc;
}

int yvex_runtime_session_prefix_attach(
    yvex_runtime_execution_session *destination,
    const yvex_runtime_session_prefix *prefix,
    yvex_runtime_session_prefix_summary *summary,
    yvex_runtime_model_failure *failure, yvex_error *err)
{
    yvex_runtime_model_summary model = {0};
    yvex_runtime_session_prefix_summary current = {0};
    int rc, draft_pristine = 0;

    if (summary) memset(summary, 0, sizeof(*summary));
    if (!destination || !prefix || !summary ||
        !destination->lifecycle_mutex_ready ||
        pthread_mutex_lock(&destination->lifecycle_mutex) != 0)
        return prefix_refuse(failure, YVEX_ERR_INVALID_ARG,
                             "empty destination session is required", err);
    rc = prefix_identity(prefix, &current) ? YVEX_OK : YVEX_ERR_FORMAT;
    if (rc == YVEX_OK)
        rc = yvex_runtime_model_summary_copy(destination->model, &model, err);
    if (rc == YVEX_OK && !prefix->draft &&
        destination->draft_attention_state_provider_ready)
        rc = yvex_runtime_private_attention_state_pristine(
            &destination->draft_attention_state_provider,
            &draft_pristine, err);
    if (rc != YVEX_OK || !destination->summary.open ||
        destination->summary.busy || destination->closing ||
        destination->summary.invalidated || destination->state_residency ||
        destination->draft_state_residency ||
        !destination->attention_state_provider_ready ||
        (prefix->draft &&
         !destination->draft_attention_state_provider_ready) ||
        (!prefix->draft &&
         destination->draft_attention_state_provider_ready &&
         !draft_pristine) ||
        strcmp(current.runtime_model_identity,
               model.runtime_model_identity) != 0) {
        rc = prefix_refuse(failure, YVEX_ERR_STATE,
                           "prefix and destination session are incompatible",
                           err);
        goto done;
    }
    rc = prefix_provider_attach(&destination->attention_state_provider,
                                &prefix->target_capacity, prefix->target,
                                prefix->target_recipes,
                                prefix->target_recipe_count, err);
    if (rc == YVEX_OK && prefix->draft)
        rc = prefix_provider_attach(
            &destination->draft_attention_state_provider,
            &prefix->draft_capacity, prefix->draft,
            prefix->draft_recipes, prefix->draft_recipe_count, err);
    if (rc != YVEX_OK) {
        yvex_error cleanup;
        destination->summary.invalidated = 1;
        (void)yvex_runtime_private_session_invalidate(destination, 1,
                                                      &cleanup);
        yvex_runtime_private_failure_record(
            failure, YVEX_RUNTIME_MODEL_FAILURE_GRAPH, "session-prefix",
            1ull, 0ull, "prefix attachment failed atomically");
        goto done;
    }
    if (!prefix_identity(prefix, summary)) {
        rc = prefix_refuse(failure, YVEX_ERR_FORMAT,
                           "attached prefix identity changed", err);
        goto done;
    }
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
done:
    (void)pthread_mutex_unlock(&destination->lifecycle_mutex);
    return rc;
}

void yvex_runtime_session_prefix_close(yvex_runtime_session_prefix **owner)
{
    yvex_runtime_session_prefix *prefix = owner ? *owner : NULL;
    if (!prefix) return;
    *owner = NULL;
    yvex_attention_state_prefix_close(&prefix->draft);
    yvex_attention_state_prefix_close(&prefix->target);
    free(prefix->draft_recipes);
    free(prefix->target_recipes);
    memset(prefix, 0, sizeof(*prefix));
    free(prefix);
}
