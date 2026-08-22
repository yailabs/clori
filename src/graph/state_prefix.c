/* Immutable prefix backing is shared; each attached provider owns only its writable COW tail. */
#include "src/graph/private.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

struct yvex_attention_state_prefix {
    unsigned int schema_version;
    unsigned long long layer_count, committed_sequence_length;
    yvex_graph_state_bank_prefix **layers;
    char (*layer_identities)[YVEX_SHA256_HEX_CAP];
    unsigned long long shared_bytes, mapped_bytes;
    char state_layout_identity[YVEX_SHA256_HEX_CAP];
    char state_content_identity[YVEX_SHA256_HEX_CAP];
    char capacity_plan_identity[YVEX_SHA256_HEX_CAP];
    char identity[YVEX_SHA256_HEX_CAP];
};

static int prefix_reject(yvex_attention_failure *failure,
                         unsigned long long layer, yvex_status status,
                         const char *reason, yvex_error *err)
{
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = YVEX_ATTENTION_FAILURE_STATE_DELTA;
        failure->layer_index = layer;
        failure->role = YVEX_TENSOR_ROLE_UNKNOWN;
        failure->reason = reason;
    }
    yvex_error_set(err, status, "graph.state.prefix", reason);
    return status;
}

static int prefix_add(unsigned long long left, unsigned long long right,
                      unsigned long long *result)
{
    return yvex_core_u64_add(left, right, result);
}

static int prefix_identity(const yvex_attention_state_prefix *prefix,
                           char output[YVEX_SHA256_HEX_CAP],
                           unsigned long long *shared_bytes,
                           unsigned long long *mapped_bytes,
                           unsigned long long *references)
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long layer, shared = 0ull, mapped = 0ull;
    unsigned long long minimum_references = ULLONG_MAX;

    if (!prefix || !prefix->layers || !prefix->layer_identities ||
        prefix->schema_version != YVEX_ATTENTION_STATE_PREFIX_SCHEMA_V1 ||
        !prefix->layer_count ||
        !yvex_sha256_hex_valid(prefix->state_layout_identity) ||
        !yvex_sha256_hex_valid(prefix->state_content_identity) ||
        !yvex_sha256_hex_valid(prefix->capacity_plan_identity))
        return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, "yvex.graph.attention.prefix.v1") ||
        !yvex_sha256_update_u64(&hash, prefix->schema_version) ||
        !yvex_sha256_update_u64(&hash, prefix->layer_count) ||
        !yvex_sha256_update_u64(&hash, prefix->committed_sequence_length) ||
        !yvex_sha256_update_text(&hash, prefix->state_layout_identity) ||
        !yvex_sha256_update_text(&hash, prefix->state_content_identity) ||
        !yvex_sha256_update_text(&hash, prefix->capacity_plan_identity))
        return 0;
    for (layer = 0ull; layer < prefix->layer_count; ++layer) {
        unsigned long long layer_shared, layer_mapped, layer_references;
        const char *bank_identity = NULL;
        if (!prefix->layers[layer] ||
            !yvex_sha256_hex_valid(prefix->layer_identities[layer]))
            return 0;
        yvex_graph_state_bank_prefix_summary(
            prefix->layers[layer], &layer_shared, &layer_mapped,
            &layer_references, &bank_identity);
        if (!bank_identity || !yvex_sha256_hex_valid(bank_identity) ||
            !layer_references ||
            !prefix_add(shared, layer_shared, &shared) ||
            !prefix_add(mapped, layer_mapped, &mapped) ||
            !yvex_sha256_update_u64(&hash, layer) ||
            !yvex_sha256_update_text(&hash, bank_identity) ||
            !yvex_sha256_update_text(&hash, prefix->layer_identities[layer]))
            return 0;
        if (layer_references < minimum_references)
            minimum_references = layer_references;
    }
    if (!yvex_sha256_update_u64(&hash, shared) ||
        !yvex_sha256_update_u64(&hash, mapped) ||
        !yvex_sha256_final(&hash, digest))
        return 0;
    yvex_sha256_hex(digest, output);
    if (shared_bytes) *shared_bytes = shared;
    if (mapped_bytes) *mapped_bytes = mapped;
    if (references)
        *references = minimum_references == ULLONG_MAX ? 1ull
                                                       : minimum_references;
    return 1;
}

static int prefix_measure(attention_state *state, unsigned long long *bytes,
                          yvex_error *err)
{
    unsigned long long layer, total = 0ull;
    for (layer = 0ull; layer < state->layer_count; ++layer) {
        attention_layer_state *entry = &state->layers[layer];
        unsigned long long current;
        int rc;
        if (!entry->prepared)
            return prefix_reject(NULL, layer, YVEX_ERR_STATE,
                                 "state prefix layer is not prepared", err);
        rc = yvex_graph_state_bank_prefix_measure(
            entry->bank[entry->committed_bank].components,
            &entry->recipe, &current, err);
        if (rc != YVEX_OK) return rc;
        if (!prefix_add(total, current, &total))
            return prefix_reject(NULL, layer, YVEX_ERR_BOUNDS,
                                 "state prefix byte measurement failed", err);
    }
    *bytes = total;
    return YVEX_OK;
}

static int prefix_capture_banks(attention_state *state,
                                yvex_attention_state_prefix *prefix,
                                yvex_error *err)
{
    unsigned long long layer;
    for (layer = 0ull; layer < state->layer_count; ++layer) {
        attention_layer_state *entry = &state->layers[layer];
        attention_state_bank *committed =
            &entry->bank[entry->committed_bank];
        if (yvex_graph_state_bank_prefix_capture(
                &prefix->layers[layer], committed->components,
                &committed->view, &entry->recipe, err) != YVEX_OK)
            return (int)yvex_error_code(err);
        yvex_core_text_copy(prefix->layer_identities[layer],
                            YVEX_SHA256_HEX_CAP,
                            committed->state_identity);
    }
    return YVEX_OK;
}

static int prefix_preflight_banks(
    attention_state *state, const yvex_attention_state_prefix *prefix,
    yvex_error *err)
{
    unsigned long long layer;
    for (layer = 0ull; layer < state->layer_count; ++layer) {
        attention_layer_state *entry = &state->layers[layer];
        unsigned int bank;
        for (bank = 0u; bank < 2u; ++bank)
            if (yvex_graph_state_bank_prefix_compatible(
                    prefix->layers[layer], entry->bank[bank].components,
                    &entry->recipe, err) != YVEX_OK)
                return (int)yvex_error_code(err);
    }
    return YVEX_OK;
}

static int prefix_attach_banks(
    attention_state *state, const yvex_attention_state_prefix *prefix,
    yvex_error *err)
{
    unsigned long long layer;
    for (layer = 0ull; layer < state->layer_count; ++layer) {
        attention_layer_state *entry = &state->layers[layer];
        unsigned int bank;
        for (bank = 0u; bank < 2u; ++bank) {
            attention_state_bank *target = &entry->bank[bank];
            if (yvex_graph_state_bank_prefix_attach(
                    prefix->layers[layer], target->components, &target->view,
                    &entry->recipe, err) != YVEX_OK)
                return (int)yvex_error_code(err);
            yvex_core_text_copy(target->state_identity,
                                sizeof(target->state_identity),
                                prefix->layer_identities[layer]);
        }
        entry->committed_bank = 0u;
        entry->staged = entry->completion_pending = 0;
        entry->banks_synchronized = 1;
    }
    return YVEX_OK;
}

int yvex_graph_attention_state_prefix_capture(
    void *context, unsigned long long maximum_bytes,
    yvex_attention_state_prefix **out, yvex_attention_failure *failure,
    yvex_error *err)
{
    attention_state *state = context;
    yvex_attention_state_prefix *prefix = NULL;
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned long long measured = 0ull, shared, mapped;
    int rc = YVEX_OK;
    if (out) *out = NULL;
    if (!state || !out || !maximum_bytes || !state->mutex_ready ||
        pthread_mutex_lock(&state->mutex) != 0)
        return prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                             YVEX_ERR_INVALID_ARG,
                             "valid idle state and prefix budget are required",
                             err);
    if (state->transaction.active || state->summary.cancelled ||
        state->summary.invalidated || !state->summary.position_consistent ||
        state->summary.prepared_layer_count != state->layer_count) {
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           YVEX_ERR_STATE,
                           "only complete idle state may become a prefix",
                           err);
        goto done;
    }
    rc = prefix_measure(state, &measured, err);
    if (rc != YVEX_OK) {
        rc = prefix_reject(
            failure, YVEX_ATTENTION_NO_LAYER,
            yvex_error_is_set(err) ? (yvex_status)yvex_error_code(err)
                                   : YVEX_ERR_STATE,
            rc == YVEX_ERR_UNSUPPORTED
                ? "state prefix capture requires paged storage"
                : "state prefix byte measurement failed",
            err);
        goto done;
    }
    if (measured > maximum_bytes) {
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           YVEX_ERR_BOUNDS,
                           "state prefix exceeds its admitted budget", err);
        goto done;
    }
    prefix = calloc(1u, sizeof(*prefix));
    if (!prefix) goto nomem;
    prefix->layers = calloc((size_t)state->layer_count,
                            sizeof(*prefix->layers));
    prefix->layer_identities = calloc(
        (size_t)state->layer_count, sizeof(*prefix->layer_identities));
    if (!prefix->layers || !prefix->layer_identities) goto nomem;
    prefix->schema_version = YVEX_ATTENTION_STATE_PREFIX_SCHEMA_V1;
    prefix->layer_count = state->layer_count;
    prefix->committed_sequence_length =
        state->summary.committed_sequence_length;
    yvex_core_text_copy(prefix->state_layout_identity,
                        sizeof(prefix->state_layout_identity),
                        state->summary.state_layout_identity);
    yvex_core_text_copy(prefix->state_content_identity,
                        sizeof(prefix->state_content_identity),
                        state->summary.state_content_identity);
    yvex_core_text_copy(prefix->capacity_plan_identity,
                        sizeof(prefix->capacity_plan_identity),
                        state->summary.capacity_plan_identity);
    if (prefix_capture_banks(state, prefix, err) != YVEX_OK ||
        !prefix_identity(prefix, identity, &shared, &mapped, NULL) ||
        shared != measured || shared > maximum_bytes ||
        prefix_preflight_banks(state, prefix, err) != YVEX_OK ||
        prefix_attach_banks(state, prefix, err) != YVEX_OK) {
        state->summary.invalidated = 1;
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           yvex_error_is_set(err)
                               ? (yvex_status)yvex_error_code(err)
                               : YVEX_ERR_STATE,
                           "state prefix capture could not publish backing",
                           err);
        goto done;
    }
    prefix->shared_bytes = shared;
    prefix->mapped_bytes = mapped;
    yvex_core_text_copy(prefix->identity, sizeof(prefix->identity), identity);
    *out = prefix;
    prefix = NULL;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    goto done;
nomem:
    rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER, YVEX_ERR_NOMEM,
                       "state prefix owner allocation failed", err);
done:
    (void)pthread_mutex_unlock(&state->mutex);
    yvex_attention_state_prefix_close(&prefix);
    return rc;
}

int yvex_graph_attention_state_prefix_attach(
    void *context, const yvex_attention_state_prefix *prefix,
    yvex_attention_failure *failure, yvex_error *err)
{
    attention_state *state = context;
    yvex_attention_state_prefix_summary summary;
    unsigned long long generation;
    char content_identity[YVEX_SHA256_HEX_CAP];
    int rc = YVEX_OK;
    if (!state || !prefix || !state->mutex_ready ||
        yvex_attention_state_prefix_summary_copy(prefix, &summary, err) !=
            YVEX_OK ||
        pthread_mutex_lock(&state->mutex) != 0)
        return prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                             YVEX_ERR_INVALID_ARG,
                             "valid prefix and destination state are required",
                             err);
    /* A freshly prepared provider starts at its recipe-authored origin, which
     * need not be token zero.  Attaching replaces both committed and candidate
     * banks, so admit only that pristine origin rather than assuming zero. */
    if (state->transaction.active || state->summary.cancelled ||
        state->summary.invalidated ||
        !state->summary.position_consistent || !state->layer_count ||
        state->summary.prepared_layer_count != state->layer_count ||
        !state->layers[0].prepared ||
        state->summary.committed_sequence_length !=
            state->layers[0].recipe.initial_position ||
        state->summary.next_position !=
            state->layers[0].recipe.initial_position ||
        summary.layer_count != state->layer_count ||
        summary.committed_sequence_length > state->summary.capacity ||
        strcmp(summary.state_layout_identity,
               state->summary.state_layout_identity) != 0 ||
        strcmp(summary.capacity_plan_identity,
               state->summary.capacity_plan_identity) != 0 ||
        !prefix_add(state->summary.generation, 1ull, &generation) ||
        prefix_preflight_banks(state, prefix, err) != YVEX_OK) {
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           YVEX_ERR_FORMAT,
                           "prefix is incompatible with destination state",
                           err);
        goto done;
    }
    if (prefix_attach_banks(state, prefix, err) != YVEX_OK) {
        state->summary.invalidated = 1;
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           (yvex_status)yvex_error_code(err),
                           "prefix backing attachment failed", err);
        goto done;
    }
    yvex_graph_attention_state_candidate_clear(state);
    memset(&state->transaction, 0, sizeof(state->transaction));
    if (!yvex_graph_attention_state_content_identity(
            state, state->summary.state_layout_identity, content_identity) ||
        strcmp(content_identity, summary.state_content_identity) != 0) {
        state->summary.invalidated = 1;
        rc = prefix_reject(failure, YVEX_ATTENTION_NO_LAYER,
                           YVEX_ERR_FORMAT,
                           "attached prefix content identity is invalid", err);
        goto done;
    }
    state->summary.generation = generation;
    state->summary.committed_sequence_length =
        summary.committed_sequence_length;
    state->summary.next_position = summary.committed_sequence_length;
    state->summary.position_consistent = 1;
    yvex_core_text_copy(state->summary.state_content_identity,
                        sizeof(state->summary.state_content_identity),
                        content_identity);
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
done:
    (void)pthread_mutex_unlock(&state->mutex);
    return rc;
}

int yvex_attention_state_prefix_summary_copy(
    const yvex_attention_state_prefix *prefix,
    yvex_attention_state_prefix_summary *summary, yvex_error *err)
{
    char identity[YVEX_SHA256_HEX_CAP];
    unsigned long long shared, mapped, references;
    if (summary) memset(summary, 0, sizeof(*summary));
    if (!prefix || !summary ||
        !prefix_identity(prefix, identity, &shared, &mapped, &references) ||
        shared != prefix->shared_bytes || mapped != prefix->mapped_bytes ||
        strcmp(identity, prefix->identity) != 0)
        return prefix_reject(NULL, YVEX_ATTENTION_NO_LAYER,
                             YVEX_ERR_FORMAT,
                             "state prefix integrity validation failed", err);
    summary->schema_version = prefix->schema_version;
    summary->layer_count = prefix->layer_count;
    summary->committed_sequence_length = prefix->committed_sequence_length;
    summary->shared_bytes = shared;
    summary->mapped_bytes = mapped;
    summary->reference_count = references;
    yvex_core_text_copy(summary->state_layout_identity,
                        sizeof(summary->state_layout_identity),
                        prefix->state_layout_identity);
    yvex_core_text_copy(summary->state_content_identity,
                        sizeof(summary->state_content_identity),
                        prefix->state_content_identity);
    yvex_core_text_copy(summary->capacity_plan_identity,
                        sizeof(summary->capacity_plan_identity),
                        prefix->capacity_plan_identity);
    yvex_core_text_copy(summary->prefix_identity,
                        sizeof(summary->prefix_identity), prefix->identity);
    yvex_error_clear(err);
    return YVEX_OK;
}

void yvex_attention_state_prefix_close(yvex_attention_state_prefix **owner)
{
    yvex_attention_state_prefix *prefix = owner ? *owner : NULL;
    unsigned long long layer;
    if (!prefix) return;
    *owner = NULL;
    for (layer = 0ull; layer < prefix->layer_count; ++layer)
        yvex_graph_state_bank_prefix_close(&prefix->layers[layer]);
    free(prefix->layers);
    free(prefix->layer_identities);
    memset(prefix, 0, sizeof(*prefix));
    free(prefix);
}
