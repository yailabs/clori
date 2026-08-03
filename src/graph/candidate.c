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

struct yvex_attention_candidate_delta {
    unsigned int schema_version;
    yvex_attention_publication publication;
};

static int candidate_refuse(yvex_error *err, yvex_status status,
                            const char *reason)
{
    yvex_error_set(err, status, "graph.attention.candidate", reason);
    return status;
}

static int candidate_copy(void **out, const void *source,
                          unsigned long long count, size_t width)
{
    size_t bytes;
    void *copy;
    *out = NULL;
    if (!count) return 1;
    if (!source || !width || count > SIZE_MAX / width) return 0;
    bytes = (size_t)count * width;
    copy = calloc(1u, bytes);
    if (!copy) return 0;
    memcpy(copy, source, bytes);
    *out = copy;
    return 1;
}

static int candidate_copy_floats(float **out, const float *source,
                                 unsigned long long rows,
                                 unsigned long long width)
{
    unsigned long long count;
    return yvex_core_u64_mul(rows, width, &count) &&
           candidate_copy((void **)out, source, count, sizeof(**out));
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

static void candidate_release_publication(yvex_attention_publication *publication)
{
    if (!publication) return;
    free(publication->raw_kv);
    free(publication->compressed_kv);
    free(publication->compressed_positions);
    free(publication->indexer_kv);
    free(publication->indexer_positions);
    free(publication->main_rolling_kv_checkpoints);
    free(publication->main_rolling_score_checkpoints);
    free(publication->indexer_rolling_kv_checkpoints);
    free(publication->indexer_rolling_score_checkpoints);
    memset(publication, 0, sizeof(*publication));
}

static int candidate_copy_rolling(
    const yvex_attention_publication *source,
    const yvex_attention_rolling_state_output *rolling,
    const float *kv, const float *score, float **target_kv,
    float **target_score)
{
    unsigned long long rows = source->rolling_checkpoint_count;
    if (!rolling->present) return !kv && !score;
    return rows && rolling->kv_state_extent && rolling->score_state_extent &&
           kv && score &&
           candidate_copy_floats(target_kv, kv, rows,
                                 rolling->kv_state_extent) &&
           candidate_copy_floats(target_score, score, rows,
                                 rolling->score_state_extent);
}

static int candidate_copy_rolling_final(
    const yvex_attention_rolling_state_output *rolling, float **target_kv,
    float **target_score)
{
    if (!rolling->present) return 1;
    return rolling->kv_state && rolling->score_state &&
           candidate_copy_floats(target_kv, rolling->kv_state, 1ull,
                                 rolling->kv_state_extent) &&
           candidate_copy_floats(target_score, rolling->score_state, 1ull,
                                 rolling->score_state_extent);
}

int yvex_attention_candidate_delta_open(
    yvex_attention_candidate_delta **out,
    const yvex_attention_publication *publication, yvex_error *err)
{
    yvex_attention_candidate_delta *delta;
    yvex_attention_publication *copy;
    if (out) *out = NULL;
    if (!out || !publication || !publication->complete ||
        !publication->token_count ||
        !publication->raw_kv || !publication->kv_width ||
        (publication->prefix_addressable &&
         (publication->next_main_rolling_state.present ||
          publication->next_indexer_rolling_state.present) &&
         publication->rolling_checkpoint_count != publication->token_count))
        return candidate_refuse(
            err, YVEX_ERR_FORMAT,
            "prefix-addressable attention publication is incomplete");
    delta = calloc(1u, sizeof(*delta));
    if (!delta)
        return candidate_refuse(err, YVEX_ERR_NOMEM,
                                "candidate delta allocation failed");
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
    copy->main_rolling_kv_checkpoints = NULL;
    copy->main_rolling_score_checkpoints = NULL;
    copy->indexer_rolling_kv_checkpoints = NULL;
    copy->indexer_rolling_score_checkpoints = NULL;
    copy->next_main_rolling_state.kv_state = NULL;
    copy->next_main_rolling_state.score_state = NULL;
    copy->next_indexer_rolling_state.kv_state = NULL;
    copy->next_indexer_rolling_state.score_state = NULL;
    if (!candidate_copy_floats(&copy->raw_kv, publication->raw_kv,
                               publication->token_count,
                               publication->kv_width) ||
        !candidate_copy_floats(&copy->compressed_kv,
                               publication->compressed_kv,
                               publication->compressed_count,
                               publication->compressed_stride) ||
        !candidate_copy((void **)&copy->compressed_positions,
                        publication->compressed_positions,
                        publication->compressed_count,
                        sizeof(*copy->compressed_positions)) ||
        !candidate_copy_floats(&copy->indexer_kv,
                               publication->indexer_kv,
                               publication->indexer_count,
                               publication->indexer_stride) ||
        !candidate_copy((void **)&copy->indexer_positions,
                        publication->indexer_positions,
                        publication->indexer_count,
                        sizeof(*copy->indexer_positions)) ||
        !(publication->prefix_addressable
              ? candidate_copy_rolling(
                    publication, &publication->next_main_rolling_state,
                    publication->main_rolling_kv_checkpoints,
                    publication->main_rolling_score_checkpoints,
                    &copy->main_rolling_kv_checkpoints,
                    &copy->main_rolling_score_checkpoints)
              : candidate_copy_rolling_final(
                    &publication->next_main_rolling_state,
                    &copy->main_rolling_kv_checkpoints,
                    &copy->main_rolling_score_checkpoints)) ||
        !(publication->prefix_addressable
              ? candidate_copy_rolling(
                    publication, &publication->next_indexer_rolling_state,
                    publication->indexer_rolling_kv_checkpoints,
                    publication->indexer_rolling_score_checkpoints,
                    &copy->indexer_rolling_kv_checkpoints,
                    &copy->indexer_rolling_score_checkpoints)
              : candidate_copy_rolling_final(
                    &publication->next_indexer_rolling_state,
                    &copy->indexer_rolling_kv_checkpoints,
                    &copy->indexer_rolling_score_checkpoints))) {
        candidate_release_publication(copy);
        free(delta);
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
    *out = delta;
    yvex_error_clear(err);
    return YVEX_OK;
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
    candidate_release_publication(&(*delta)->publication);
    memset(*delta, 0, sizeof(**delta));
    free(*delta);
    *delta = NULL;
}
