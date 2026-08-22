/*
 * Preserve prefix-addressable state deltas after target verification.
 *
 * Acceptance is known only after the full target has produced all candidate distributions. This
 * owner retains the small state delta needed to promote that accepted prefix without executing
 * target rows again.
 */
#include <yvex/internal/candidate.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <yvex/internal/core.h>

#include "src/graph/private.h"

typedef enum {
    CANDIDATE_STORAGE_RAW_KV = 0,
    CANDIDATE_STORAGE_COMPRESSED_KV,
    CANDIDATE_STORAGE_COMPRESSED_POSITIONS,
    CANDIDATE_STORAGE_INDEXER_KV,
    CANDIDATE_STORAGE_INDEXER_POSITIONS,
    CANDIDATE_STORAGE_TOKEN_IDS,
    CANDIDATE_STORAGE_MAIN_KV_CHECKPOINTS,
    CANDIDATE_STORAGE_MAIN_SCORE_CHECKPOINTS,
    CANDIDATE_STORAGE_INDEXER_KV_CHECKPOINTS,
    CANDIDATE_STORAGE_INDEXER_SCORE_CHECKPOINTS,
    CANDIDATE_STORAGE_COUNT
} candidate_storage_slot;

struct yvex_attention_candidate_delta {
    unsigned int schema_version;
    yvex_attention_publication publication;
    void *storage[CANDIDATE_STORAGE_COUNT];
    size_t storage_capacity[CANDIDATE_STORAGE_COUNT];
};

static int candidate_refuse(yvex_error *err, yvex_status status,
                            const char *reason)
{
    yvex_error_set(err, status, "graph.attention.candidate", reason);
    return status;
}

static int candidate_store(yvex_attention_candidate_delta *delta,
                           candidate_storage_slot slot, void **out,
                           const void *source, unsigned long long count,
                           size_t width)
{
    size_t bytes;
    *out = NULL;
    if (!count) return 1;
    if (!delta || slot >= CANDIDATE_STORAGE_COUNT || !source || !width ||
        count > SIZE_MAX / width)
        return 0;
    bytes = (size_t)count * width;
    if (bytes > delta->storage_capacity[slot]) return 0;
    memmove(delta->storage[slot], source, bytes);
    *out = delta->storage[slot];
    return 1;
}

static int candidate_storage_bytes(unsigned long long count,
                                   unsigned long long width,
                                   size_t element_size, size_t *out)
{
    unsigned long long elements;
    if (out) *out = 0u;
    if (!out || !element_size ||
        !yvex_core_u64_mul(count, width, &elements) ||
        elements > SIZE_MAX / element_size)
        return 0;
    *out = (size_t)elements * element_size;
    return 1;
}

/*
 * A one-row decode may publish compressed state only on a later ratio boundary. Reserve that
 * bounded periodic form with the first delta so a warmed execution does not allocate merely
 * because the current token crosses the boundary. New storage is adopted only after every grow
 * succeeds, preserving an existing reusable delta on allocation failure.
 */
static int candidate_storage_prepare(
    yvex_attention_candidate_delta *delta,
    const yvex_attention_publication *publication)
{
    size_t required[CANDIDATE_STORAGE_COUNT] = {0u};
    void *replacement[CANDIDATE_STORAGE_COUNT] = {0};
    unsigned long long compressed_width = publication->compressed_stride
        ? publication->compressed_stride
        : (publication->next_main_rolling_state.present
               ? publication->next_main_rolling_state.head_dimension
               : 0ull);
    unsigned long long indexer_width = publication->indexer_stride
        ? publication->indexer_stride
        : (publication->next_indexer_rolling_state.present
               ? publication->next_indexer_rolling_state.head_dimension
               : 0ull);
    unsigned long long compressed_rows = compressed_width
                                             ? publication->token_count
                                             : publication->compressed_count;
    unsigned long long indexer_rows = indexer_width
                                          ? publication->token_count
                                          : publication->indexer_count;
    unsigned long long rolling_rows = publication->prefix_addressable
                                          ? publication->rolling_checkpoint_count
                                          : 1ull;
    unsigned int slot;
    int valid =
        candidate_storage_bytes(publication->token_count,
                                publication->kv_width, sizeof(float),
                                &required[CANDIDATE_STORAGE_RAW_KV]) &&
        candidate_storage_bytes(compressed_rows, compressed_width, sizeof(float),
                                &required[CANDIDATE_STORAGE_COMPRESSED_KV]) &&
        candidate_storage_bytes(compressed_rows, 1ull,
                                sizeof(*publication->compressed_positions),
                                &required[CANDIDATE_STORAGE_COMPRESSED_POSITIONS]) &&
        candidate_storage_bytes(indexer_rows, indexer_width,
                                sizeof(float),
                                &required[CANDIDATE_STORAGE_INDEXER_KV]) &&
        candidate_storage_bytes(indexer_rows, 1ull,
                                sizeof(*publication->indexer_positions),
                                &required[CANDIDATE_STORAGE_INDEXER_POSITIONS]) &&
        candidate_storage_bytes(publication->token_ids
                                    ? publication->token_count
                                    : 0ull,
                                1ull, sizeof(*publication->token_ids),
                                &required[CANDIDATE_STORAGE_TOKEN_IDS]) &&
        candidate_storage_bytes(publication->next_main_rolling_state.present
                                    ? rolling_rows
                                    : 0ull,
                                publication->next_main_rolling_state.kv_state_extent,
                                sizeof(float),
                                &required[CANDIDATE_STORAGE_MAIN_KV_CHECKPOINTS]) &&
        candidate_storage_bytes(publication->next_main_rolling_state.present
                                    ? rolling_rows
                                    : 0ull,
                                publication->next_main_rolling_state.score_state_extent,
                                sizeof(float),
                                &required[CANDIDATE_STORAGE_MAIN_SCORE_CHECKPOINTS]) &&
        candidate_storage_bytes(publication->next_indexer_rolling_state.present
                                    ? rolling_rows
                                    : 0ull,
                                publication->next_indexer_rolling_state.kv_state_extent,
                                sizeof(float),
                                &required[CANDIDATE_STORAGE_INDEXER_KV_CHECKPOINTS]) &&
        candidate_storage_bytes(publication->next_indexer_rolling_state.present
                                    ? rolling_rows
                                    : 0ull,
                                publication->next_indexer_rolling_state.score_state_extent,
                                sizeof(float),
                                &required[CANDIDATE_STORAGE_INDEXER_SCORE_CHECKPOINTS]);
    if (!valid ||
        (publication->compressed_count &&
         (!publication->compressed_kv || !publication->compressed_positions ||
          !publication->compressed_stride)) ||
        (publication->indexer_count &&
         (!publication->indexer_kv || !publication->indexer_positions ||
          !publication->indexer_stride)))
        return 0;
    for (slot = 0u; slot < CANDIDATE_STORAGE_COUNT; ++slot) {
        if (required[slot] <= delta->storage_capacity[slot]) continue;
        replacement[slot] = malloc(required[slot]);
        if (!replacement[slot]) {
            unsigned int release;
            for (release = 0u; release < CANDIDATE_STORAGE_COUNT; ++release)
                free(replacement[release]);
            return 0;
        }
    }
    for (slot = 0u; slot < CANDIDATE_STORAGE_COUNT; ++slot) {
        if (!replacement[slot]) continue;
        free(delta->storage[slot]);
        delta->storage[slot] = replacement[slot];
        delta->storage_capacity[slot] = required[slot];
    }
    return 1;
}

static int candidate_store_floats(
    yvex_attention_candidate_delta *delta, candidate_storage_slot slot,
    float **out, const float *source, unsigned long long rows,
    unsigned long long width)
{
    unsigned long long count;
    return yvex_core_u64_mul(rows, width, &count) &&
           candidate_store(delta, slot, (void **)out, source, count,
                           sizeof(**out));
}

int yvex_attention_candidate_checkpoints_open(
    yvex_attention_scratch_budget *scratch, unsigned long long rows,
    const yvex_attention_rolling_state_view *rolling, float **kv,
    float **score)
{
    size_t kv_bytes, score_bytes;
    unsigned long long kv_count, score_count;
    if (!scratch || !rolling || !rolling->present || !rows || !kv || !score ||
        !yvex_core_u64_mul(rows, rolling->kv_state_extent, &kv_count) ||
        !yvex_core_u64_mul(rows, rolling->score_state_extent, &score_count) ||
        !yvex_attention_scratch_reserve(scratch, kv_count, sizeof(**kv), &kv_bytes) ||
        !yvex_attention_scratch_reserve(scratch, score_count, sizeof(**score),
                                        &score_bytes))
        return 0;
    *kv = yvex_attention_scratch_calloc(scratch, kv_count, sizeof(**kv));
    *score = yvex_attention_scratch_calloc(scratch, score_count, sizeof(**score));
    return *kv && *score;
}

void yvex_attention_candidate_checkpoint_capture(
    float *kv, float *score, unsigned long long row,
    const yvex_attention_rolling_state_output *rolling)
{
    memcpy(kv + row * rolling->kv_state_extent, rolling->kv_state,
           (size_t)rolling->kv_state_extent * sizeof(*kv));
    memcpy(score + row * rolling->score_state_extent, rolling->score_state,
           (size_t)rolling->score_state_extent * sizeof(*score));
}

static void candidate_storage_release(yvex_attention_candidate_delta *delta)
{
    unsigned int slot;
    if (!delta) return;
    for (slot = 0u; slot < CANDIDATE_STORAGE_COUNT; ++slot)
        free(delta->storage[slot]);
    memset(delta, 0, sizeof(*delta));
}

static int candidate_copy_rolling(
    yvex_attention_candidate_delta *delta,
    const yvex_attention_publication *source,
    const yvex_attention_rolling_state_output *rolling,
    const float *kv, const float *score, float **target_kv,
    float **target_score, candidate_storage_slot kv_slot,
    candidate_storage_slot score_slot)
{
    unsigned long long rows = source->rolling_checkpoint_count;
    if (!rolling->present) return !kv && !score;
    return rows && rolling->kv_state_extent && rolling->score_state_extent &&
           kv && score &&
           candidate_store_floats(delta, kv_slot, target_kv, kv, rows,
                                  rolling->kv_state_extent) &&
           candidate_store_floats(delta, score_slot, target_score, score, rows,
                                  rolling->score_state_extent);
}

static int candidate_copy_rolling_final(
    yvex_attention_candidate_delta *delta,
    const yvex_attention_rolling_state_output *rolling, float **target_kv,
    float **target_score, candidate_storage_slot kv_slot,
    candidate_storage_slot score_slot)
{
    if (!rolling->present) return 1;
    return rolling->kv_state && rolling->score_state &&
           candidate_store_floats(delta, kv_slot, target_kv,
                                  rolling->kv_state, 1ull,
                                  rolling->kv_state_extent) &&
           candidate_store_floats(delta, score_slot, target_score,
                                  rolling->score_state, 1ull,
                                  rolling->score_state_extent);
}

static int candidate_delta_assign(
    yvex_attention_candidate_delta **owner,
    const yvex_attention_publication *publication, yvex_error *err)
{
    yvex_attention_candidate_delta *delta;
    yvex_attention_publication *copy;
    int allocated = 0;
    if (!owner || !publication || !publication->complete ||
        publication->device_completion_pending ||
        !publication->token_count ||
        !publication->raw_kv || !publication->kv_width ||
        (publication->prefix_addressable &&
         (publication->next_main_rolling_state.present ||
          publication->next_indexer_rolling_state.present) &&
         publication->rolling_checkpoint_count != publication->token_count))
        return candidate_refuse(
            err, YVEX_ERR_FORMAT,
            "prefix-addressable attention publication is incomplete");
    delta = *owner;
    if (!delta) {
        delta = calloc(1u, sizeof(*delta));
        if (!delta)
            return candidate_refuse(err, YVEX_ERR_NOMEM,
                                    "candidate delta allocation failed");
        allocated = 1;
    }
    if (!candidate_storage_prepare(delta, publication)) {
        if (allocated) {
            candidate_storage_release(delta);
            free(delta);
        }
        return candidate_refuse(err, YVEX_ERR_NOMEM,
                                "candidate delta storage allocation failed");
    }
    delta->schema_version = YVEX_ATTENTION_CANDIDATE_SCHEMA_V1;
    copy = &delta->publication;
    *copy = *publication;
    copy->owned = 0;
    copy->workspace = NULL;
    copy->input = copy->q_low = copy->query = NULL;
    copy->index_query = copy->index_weights = NULL;
    copy->attention_values = copy->output = NULL;
    copy->core_output = copy->envelope_output = NULL;
    copy->topk_counts = copy->topk_positions = NULL;
    copy->raw_kv = copy->compressed_kv = copy->indexer_kv = NULL;
    copy->compressed_positions = copy->indexer_positions = NULL;
    copy->token_ids = NULL;
    copy->main_rolling_kv_checkpoints = NULL;
    copy->main_rolling_score_checkpoints = NULL;
    copy->indexer_rolling_kv_checkpoints = NULL;
    copy->indexer_rolling_score_checkpoints = NULL;
    copy->next_main_rolling_state.kv_state = NULL;
    copy->next_main_rolling_state.score_state = NULL;
    copy->next_indexer_rolling_state.kv_state = NULL;
    copy->next_indexer_rolling_state.score_state = NULL;
    if (!candidate_store_floats(
            delta, CANDIDATE_STORAGE_RAW_KV, &copy->raw_kv,
            publication->raw_kv, publication->token_count,
            publication->kv_width) ||
        !candidate_store_floats(
            delta, CANDIDATE_STORAGE_COMPRESSED_KV, &copy->compressed_kv,
            publication->compressed_kv, publication->compressed_count,
            publication->compressed_stride) ||
        !candidate_store(
            delta, CANDIDATE_STORAGE_COMPRESSED_POSITIONS,
            (void **)&copy->compressed_positions,
            publication->compressed_positions, publication->compressed_count,
            sizeof(*copy->compressed_positions)) ||
        !candidate_store_floats(
            delta, CANDIDATE_STORAGE_INDEXER_KV, &copy->indexer_kv,
            publication->indexer_kv, publication->indexer_count,
            publication->indexer_stride) ||
        !candidate_store(
            delta, CANDIDATE_STORAGE_INDEXER_POSITIONS,
            (void **)&copy->indexer_positions, publication->indexer_positions,
            publication->indexer_count, sizeof(*copy->indexer_positions)) ||
        (publication->token_ids &&
         !candidate_store(
             delta, CANDIDATE_STORAGE_TOKEN_IDS, (void **)&copy->token_ids,
             publication->token_ids, publication->token_count,
             sizeof(*copy->token_ids))) ||
        !(publication->prefix_addressable
              ? candidate_copy_rolling(
                    delta, publication,
                    &publication->next_main_rolling_state,
                    publication->main_rolling_kv_checkpoints,
                    publication->main_rolling_score_checkpoints,
                    &copy->main_rolling_kv_checkpoints,
                    &copy->main_rolling_score_checkpoints,
                    CANDIDATE_STORAGE_MAIN_KV_CHECKPOINTS,
                    CANDIDATE_STORAGE_MAIN_SCORE_CHECKPOINTS)
              : candidate_copy_rolling_final(
                    delta, &publication->next_main_rolling_state,
                    &copy->main_rolling_kv_checkpoints,
                    &copy->main_rolling_score_checkpoints,
                    CANDIDATE_STORAGE_MAIN_KV_CHECKPOINTS,
                    CANDIDATE_STORAGE_MAIN_SCORE_CHECKPOINTS)) ||
        !(publication->prefix_addressable
              ? candidate_copy_rolling(
                    delta, publication,
                    &publication->next_indexer_rolling_state,
                    publication->indexer_rolling_kv_checkpoints,
                    publication->indexer_rolling_score_checkpoints,
                    &copy->indexer_rolling_kv_checkpoints,
                    &copy->indexer_rolling_score_checkpoints,
                    CANDIDATE_STORAGE_INDEXER_KV_CHECKPOINTS,
                    CANDIDATE_STORAGE_INDEXER_SCORE_CHECKPOINTS)
              : candidate_copy_rolling_final(
                    delta, &publication->next_indexer_rolling_state,
                    &copy->indexer_rolling_kv_checkpoints,
                    &copy->indexer_rolling_score_checkpoints,
                    CANDIDATE_STORAGE_INDEXER_KV_CHECKPOINTS,
                    CANDIDATE_STORAGE_INDEXER_SCORE_CHECKPOINTS))) {
        if (allocated) {
            candidate_storage_release(delta);
            free(delta);
        }
        return candidate_refuse(err, YVEX_ERR_NOMEM,
                                "candidate delta storage allocation failed");
    }
    if (!publication->prefix_addressable) {
        if (copy->next_main_rolling_state.present) {
            copy->next_main_rolling_state.kv_state =
                copy->main_rolling_kv_checkpoints;
            copy->next_main_rolling_state.score_state =
                copy->main_rolling_score_checkpoints;
        }
        if (copy->next_indexer_rolling_state.present) {
            copy->next_indexer_rolling_state.kv_state =
                copy->indexer_rolling_kv_checkpoints;
            copy->next_indexer_rolling_state.score_state =
                copy->indexer_rolling_score_checkpoints;
        }
    }
    *owner = delta;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_attention_candidate_delta_open(
    yvex_attention_candidate_delta **out,
    const yvex_attention_publication *publication, yvex_error *err)
{
    return candidate_delta_assign(out, publication, err);
}

/*
 * Compressed positions name the first token in a completed ratio group, not
 * the token that completed it.  Prefix selection must therefore use the
 * rolling-state ratio and prefix extent; comparing the stored position with
 * the prefix end would expose an emission produced by a later candidate row.
 */
static int candidate_emission_count(
    const yvex_attention_history_view *committed,
    const yvex_attention_rolling_state_view *rolling,
    unsigned long long prefix_count, unsigned long long available,
    unsigned long long *selected)
{
    unsigned long long end;
    if (!committed || !rolling || !rolling->present || !rolling->ratio ||
        !selected ||
        !yvex_core_u64_add(committed->token_count, prefix_count, &end))
        return 0;
    *selected = end / rolling->ratio -
                committed->token_count / rolling->ratio;
    return *selected <= available;
}

static void candidate_rolling_project(
    const yvex_attention_rolling_state_view *before,
    const float *kv_checkpoints, const float *score_checkpoints,
    unsigned long long prefix_count,
    yvex_attention_rolling_state_output *after)
{
    unsigned long long token, cursor, previous_fill, current_fill;
    memcpy(after, before, sizeof(*after));
    after->kv_state = (float *)(kv_checkpoints +
        (prefix_count - 1ull) * before->kv_state_extent);
    after->score_state = (float *)(score_checkpoints +
        (prefix_count - 1ull) * before->score_state_extent);
    cursor = before->cursor;
    previous_fill = before->previous_fill;
    current_fill = before->current_fill;
    for (token = 0ull; token < prefix_count; ++token) {
        int emitted = ((before->next_token_position + token + 1ull) %
                       before->ratio) == 0ull;
        previous_fill = emitted ? (before->overlap ? before->ratio : 0ull)
                                : previous_fill;
        current_fill = emitted ? 0ull
                               : (current_fill < cursor + 1ull
                                      ? cursor + 1ull : current_fill);
        cursor = emitted ? 0ull : (cursor + 1ull) % before->ratio;
    }
    after->next_token_position = before->next_token_position + prefix_count;
    after->previous_fill = previous_fill;
    after->current_fill = current_fill;
    after->cursor = cursor;
}

int yvex_attention_candidate_delta_project(
    const yvex_attention_candidate_delta *delta,
    const yvex_attention_history_view *committed,
    unsigned long long prefix_count, yvex_attention_publication *out,
    yvex_error *err)
{
    const yvex_attention_publication *source = delta ? &delta->publication : NULL;
    unsigned long long end;
    if (out) memset(out, 0, sizeof(*out));
    if (!source)
        return candidate_refuse(err, YVEX_ERR_INVALID_ARG,
                                "candidate prefix owner is required");
    if (!committed)
        return candidate_refuse(err, YVEX_ERR_INVALID_ARG,
                                "candidate prefix committed base is required");
    if (!out)
        return candidate_refuse(err, YVEX_ERR_INVALID_ARG,
                                "candidate prefix output is required");
    if (committed->token_count != source->token_position)
        return candidate_refuse(err, YVEX_ERR_STATE,
                                "candidate prefix does not continue its committed base");
    if (!prefix_count || prefix_count > source->token_count)
        return candidate_refuse(err, YVEX_ERR_BOUNDS,
                                "candidate prefix exceeds the retained verification extent");
    if (!source->prefix_addressable && prefix_count != source->token_count)
        return candidate_refuse(
            err, YVEX_ERR_UNSUPPORTED,
            "non-prefix publication cannot project a shorter state prefix");
    if (!yvex_core_u64_add(source->token_position, prefix_count, &end))
        return candidate_refuse(err, YVEX_ERR_BOUNDS,
                                "candidate prefix position overflowed");
    *out = *source;
    out->owned = 0;
    out->prefix_addressable = 0;
    out->token_count = prefix_count;
    if (source->next_main_rolling_state.present &&
        (!committed->main_rolling_state.present ||
         !candidate_emission_count(
             committed, &committed->main_rolling_state, prefix_count,
             source->compressed_count, &out->compressed_count)))
        return candidate_refuse(
            err, YVEX_ERR_STATE,
            "compressed prefix checkpoints disagree with the committed base");
    if (source->next_indexer_rolling_state.present &&
        (!committed->indexer_rolling_state.present ||
         !candidate_emission_count(
             committed, &committed->indexer_rolling_state, prefix_count,
             source->indexer_count, &out->indexer_count)))
        return candidate_refuse(
            err, YVEX_ERR_STATE,
            "indexer prefix checkpoints disagree with the committed base");
    if (source->prefix_addressable && source->next_main_rolling_state.present) {
        if (!committed->main_rolling_state.present)
            return candidate_refuse(err, YVEX_ERR_STATE,
                                    "main rolling prefix has no committed base");
        candidate_rolling_project(
            &committed->main_rolling_state,
            source->main_rolling_kv_checkpoints,
            source->main_rolling_score_checkpoints, prefix_count,
            &out->next_main_rolling_state);
    }
    if (source->prefix_addressable && source->next_indexer_rolling_state.present) {
        if (!committed->indexer_rolling_state.present)
            return candidate_refuse(err, YVEX_ERR_STATE,
                                    "indexer rolling prefix has no committed base");
        candidate_rolling_project(
            &committed->indexer_rolling_state,
            source->indexer_rolling_kv_checkpoints,
            source->indexer_rolling_score_checkpoints, prefix_count,
            &out->next_indexer_rolling_state);
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

unsigned long long yvex_attention_candidate_delta_token_count(
    const yvex_attention_candidate_delta *delta)
{
    return delta ? delta->publication.token_count : 0ull;
}

const yvex_attention_publication *yvex_attention_candidate_delta_publication(
    const yvex_attention_candidate_delta *delta)
{
    return delta && delta->schema_version == YVEX_ATTENTION_CANDIDATE_SCHEMA_V1
               ? &delta->publication
               : NULL;
}

void yvex_attention_candidate_delta_close(
    yvex_attention_candidate_delta **delta)
{
    if (!delta || !*delta) return;
    candidate_storage_release(*delta);
    free(*delta);
    *delta = NULL;
}
