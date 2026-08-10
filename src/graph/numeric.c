/*
 * Provide deterministic scalar mechanisms shared by graph construction and family execution.
 *
 * Runtime policy comes from immutable plans and reference algorithms remain test-owned.
 * Reproducible primitives and fixture proofs do not establish complete attention support.
 */
#include "src/graph/private.h"

static const double attention_pi =
    3.14159265358979323846264338327950288;

typedef enum {
    ATTENTION_COMPARE_INT,
    ATTENTION_COMPARE_UINT,
    ATTENTION_COMPARE_U64,
    ATTENTION_COMPARE_CLASS,
    ATTENTION_COMPARE_ROLLING_KIND,
    ATTENTION_COMPARE_TEXT
} attention_compare_field_kind;

typedef struct {
    size_t offset, extent;
    attention_compare_field_kind kind;
} attention_compare_field;

typedef enum {
    ATTENTION_COMPARE_GEOMETRY,
    ATTENTION_COMPARE_F32,
    ATTENTION_COMPARE_POSITIONS
} attention_compare_step_kind;

enum {
    ATTENTION_COMPARE_PRODUCT = 1u,
    ATTENTION_COMPARE_SKIP_EMPTY = 2u,
    ATTENTION_COMPARE_REQUIRE_STRIDE = 4u,
    ATTENTION_COMPARE_REQUIRE_PRESENT = 8u,
    ATTENTION_COMPARE_STOP_ON_MISMATCH = 16u,
    ATTENTION_COMPARE_NEGATIVE_INFINITY_SENTINEL = 32u,
    ATTENTION_COMPARE_GROUPS = 5u
};

typedef struct {
    attention_compare_step_kind kind;
    yvex_attention_comparison_stage stage;
    unsigned int group, flags;
    size_t base_offset, data_offset, count_offset, stride_offset;
    const attention_compare_field *fields;
    size_t field_count;
    const char *overflow_message;
} attention_compare_step;

typedef struct {
    yvex_attention_state_comparison result;
    unsigned char compatible[ATTENTION_COMPARE_GROUPS];
} attention_compare_cursor;

static const attention_compare_field attention_publication_fields[] = {
    {offsetof(yvex_attention_publication, complete),
     sizeof(((yvex_attention_publication *)0)->complete), ATTENTION_COMPARE_INT},
    {offsetof(yvex_attention_publication, layer_index),
     sizeof(((yvex_attention_publication *)0)->layer_index), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_publication, attention_class),
     sizeof(((yvex_attention_publication *)0)->attention_class), ATTENTION_COMPARE_CLASS},
    {offsetof(yvex_attention_publication, token_position),
     sizeof(((yvex_attention_publication *)0)->token_position), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_publication, token_count),
     sizeof(((yvex_attention_publication *)0)->token_count), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_publication, kv_width),
     sizeof(((yvex_attention_publication *)0)->kv_width), ATTENTION_COMPARE_U64},
};
static const attention_compare_field attention_compressed_fields[] = {
    {offsetof(yvex_attention_publication, compressed_count),
     sizeof(((yvex_attention_publication *)0)->compressed_count), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_publication, compressed_stride),
     sizeof(((yvex_attention_publication *)0)->compressed_stride), ATTENTION_COMPARE_U64},
};
static const attention_compare_field attention_indexer_fields[] = {
    {offsetof(yvex_attention_publication, indexer_count),
     sizeof(((yvex_attention_publication *)0)->indexer_count), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_publication, indexer_stride),
     sizeof(((yvex_attention_publication *)0)->indexer_stride), ATTENTION_COMPARE_U64},
};
static const attention_compare_field attention_rolling_fields[] = {
    {offsetof(yvex_attention_rolling_state_output, present),
     sizeof(((yvex_attention_rolling_state_output *)0)->present), ATTENTION_COMPARE_INT},
    {offsetof(yvex_attention_rolling_state_output, schema_version),
     sizeof(((yvex_attention_rolling_state_output *)0)->schema_version), ATTENTION_COMPARE_UINT},
    {offsetof(yvex_attention_rolling_state_output, kind),
     sizeof(((yvex_attention_rolling_state_output *)0)->kind), ATTENTION_COMPARE_ROLLING_KIND},
    {offsetof(yvex_attention_rolling_state_output, attention_class),
     sizeof(((yvex_attention_rolling_state_output *)0)->attention_class),
     ATTENTION_COMPARE_CLASS},
    {offsetof(yvex_attention_rolling_state_output, layer_index),
     sizeof(((yvex_attention_rolling_state_output *)0)->layer_index), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, next_token_position),
     sizeof(((yvex_attention_rolling_state_output *)0)->next_token_position),
     ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, ratio),
     sizeof(((yvex_attention_rolling_state_output *)0)->ratio), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, head_dimension),
     sizeof(((yvex_attention_rolling_state_output *)0)->head_dimension), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, state_width),
     sizeof(((yvex_attention_rolling_state_output *)0)->state_width), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, state_slots),
     sizeof(((yvex_attention_rolling_state_output *)0)->state_slots), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, kv_state_stride),
     sizeof(((yvex_attention_rolling_state_output *)0)->kv_state_stride), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, score_state_stride),
     sizeof(((yvex_attention_rolling_state_output *)0)->score_state_stride),
     ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, kv_state_extent),
     sizeof(((yvex_attention_rolling_state_output *)0)->kv_state_extent), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, score_state_extent),
     sizeof(((yvex_attention_rolling_state_output *)0)->score_state_extent),
     ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, previous_fill),
     sizeof(((yvex_attention_rolling_state_output *)0)->previous_fill), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, current_fill),
     sizeof(((yvex_attention_rolling_state_output *)0)->current_fill), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, cursor),
     sizeof(((yvex_attention_rolling_state_output *)0)->cursor), ATTENTION_COMPARE_U64},
    {offsetof(yvex_attention_rolling_state_output, overlap),
     sizeof(((yvex_attention_rolling_state_output *)0)->overlap), ATTENTION_COMPARE_INT},
    {offsetof(yvex_attention_rolling_state_output, rotated),
     sizeof(((yvex_attention_rolling_state_output *)0)->rotated), ATTENTION_COMPARE_INT},
    {offsetof(yvex_attention_rolling_state_output, attention_plan_identity),
     sizeof(((yvex_attention_rolling_state_output *)0)->attention_plan_identity),
     ATTENTION_COMPARE_TEXT},
};

static const attention_compare_step attention_state_steps[] = {
    {ATTENTION_COMPARE_GEOMETRY, YVEX_ATTENTION_COMPARISON_STAGE_PUBLICATION, 0u,
     ATTENTION_COMPARE_STOP_ON_MISMATCH, 0u, 0u, 0u, 0u, attention_publication_fields,
     sizeof(attention_publication_fields) / sizeof(attention_publication_fields[0]), NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_RAW_KV, 0u,
     ATTENTION_COMPARE_PRODUCT, 0u, offsetof(yvex_attention_publication, raw_kv),
     offsetof(yvex_attention_publication, token_count),
     offsetof(yvex_attention_publication, kv_width), NULL, 0u,
     "attention state raw KV geometry overflowed"},
    {ATTENTION_COMPARE_GEOMETRY, YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_GEOMETRY, 1u, 0u,
     0u, 0u, 0u, 0u, attention_compressed_fields,
     sizeof(attention_compressed_fields) / sizeof(attention_compressed_fields[0]), NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_KV, 1u,
     ATTENTION_COMPARE_PRODUCT | ATTENTION_COMPARE_SKIP_EMPTY |
         ATTENTION_COMPARE_REQUIRE_STRIDE,
     0u, offsetof(yvex_attention_publication, compressed_kv),
     offsetof(yvex_attention_publication, compressed_count),
     offsetof(yvex_attention_publication, compressed_stride), NULL, 0u,
     "attention state emission geometry overflowed"},
    {ATTENTION_COMPARE_POSITIONS, YVEX_ATTENTION_COMPARISON_STAGE_COMPRESSED_POSITIONS, 1u,
     ATTENTION_COMPARE_SKIP_EMPTY, 0u,
     offsetof(yvex_attention_publication, compressed_positions),
     offsetof(yvex_attention_publication, compressed_count), 0u, NULL, 0u, NULL},
    {ATTENTION_COMPARE_GEOMETRY, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_GEOMETRY, 2u,
     0u, 0u, 0u, 0u, 0u, attention_indexer_fields,
     sizeof(attention_indexer_fields) / sizeof(attention_indexer_fields[0]), NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_KV, 2u,
     ATTENTION_COMPARE_PRODUCT | ATTENTION_COMPARE_SKIP_EMPTY |
         ATTENTION_COMPARE_REQUIRE_STRIDE,
     0u, offsetof(yvex_attention_publication, indexer_kv),
     offsetof(yvex_attention_publication, indexer_count),
     offsetof(yvex_attention_publication, indexer_stride), NULL, 0u,
     "attention state emission geometry overflowed"},
    {ATTENTION_COMPARE_POSITIONS, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_EMISSION_POSITIONS, 2u,
     ATTENTION_COMPARE_SKIP_EMPTY, 0u, offsetof(yvex_attention_publication, indexer_positions),
     offsetof(yvex_attention_publication, indexer_count), 0u, NULL, 0u, NULL},
    {ATTENTION_COMPARE_GEOMETRY, YVEX_ATTENTION_COMPARISON_STAGE_MAIN_GEOMETRY, 3u, 0u,
     offsetof(yvex_attention_publication, next_main_rolling_state), 0u, 0u, 0u,
     attention_rolling_fields,
     sizeof(attention_rolling_fields) / sizeof(attention_rolling_fields[0]), NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_MAIN_KV, 3u,
     ATTENTION_COMPARE_REQUIRE_PRESENT,
     offsetof(yvex_attention_publication, next_main_rolling_state),
     offsetof(yvex_attention_rolling_state_output, kv_state),
     offsetof(yvex_attention_rolling_state_output, kv_state_extent), 0u, NULL, 0u, NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_MAIN_SCORE, 3u,
     ATTENTION_COMPARE_REQUIRE_PRESENT | ATTENTION_COMPARE_NEGATIVE_INFINITY_SENTINEL,
     offsetof(yvex_attention_publication, next_main_rolling_state),
     offsetof(yvex_attention_rolling_state_output, score_state),
     offsetof(yvex_attention_rolling_state_output, score_state_extent), 0u, NULL, 0u, NULL},
    {ATTENTION_COMPARE_GEOMETRY, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_GEOMETRY, 4u,
     0u, offsetof(yvex_attention_publication, next_indexer_rolling_state), 0u, 0u, 0u,
     attention_rolling_fields,
     sizeof(attention_rolling_fields) / sizeof(attention_rolling_fields[0]), NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_KV, 4u,
     ATTENTION_COMPARE_REQUIRE_PRESENT,
     offsetof(yvex_attention_publication, next_indexer_rolling_state),
     offsetof(yvex_attention_rolling_state_output, kv_state),
     offsetof(yvex_attention_rolling_state_output, kv_state_extent), 0u, NULL, 0u, NULL},
    {ATTENTION_COMPARE_F32, YVEX_ATTENTION_COMPARISON_STAGE_INDEXER_ROLLING_SCORE, 4u,
     ATTENTION_COMPARE_REQUIRE_PRESENT | ATTENTION_COMPARE_NEGATIVE_INFINITY_SENTINEL,
     offsetof(yvex_attention_publication, next_indexer_rolling_state),
     offsetof(yvex_attention_rolling_state_output, score_state),
     offsetof(yvex_attention_rolling_state_output, score_state_extent), 0u, NULL, 0u, NULL},
};

static int attention_compare_field_equal(const unsigned char *left, const unsigned char *right,
                                         const attention_compare_field *field)
{
    left += field->offset;
    right += field->offset;
    if (field->kind == ATTENTION_COMPARE_TEXT)
        return strncmp((const char *)left, (const char *)right, field->extent) == 0;
    return memcmp(left, right, field->extent) == 0;
}

static void attention_compare_fail(attention_compare_cursor *cursor,
                                   yvex_attention_comparison_stage stage,
                                   unsigned long long coordinate, int exact_failure)
{
    cursor->result.numeric.within_tolerance = 0;
    if (exact_failure) cursor->result.numeric.bitwise_equal = 0;
    if (exact_failure > 1) cursor->result.geometry_equal = 0;
    if (cursor->result.first_failing_stage == YVEX_ATTENTION_COMPARISON_STAGE_NONE &&
        cursor->result.numeric.first_failing_coordinate == ULLONG_MAX) {
        cursor->result.first_failing_stage = stage;
        cursor->result.numeric.first_failing_coordinate = coordinate;
    }
}

/*
 * Accumulate one F32 span under the canonical finite/tolerance contract.
 *
 * Malformed storage or aggregate-count overflow returns a typed failure.
 */
static int attention_compare_f32(const float *left, const float *right,
                                 unsigned long long count, double absolute_tolerance,
                                 double relative_tolerance, unsigned int flags,
                                 yvex_attention_comparison_stage stage,
                                 attention_compare_cursor *cursor, yvex_error *err)
{
    unsigned long long index, total, sentinel_count = 0ull;

    if (!left || !right || !count || count > SIZE_MAX / sizeof(*left)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "graph.f32.compare",
                       "finite tolerances and representable non-empty F32 ranges are required");
        return YVEX_ERR_INVALID_ARG;
    }
    cursor->result.numeric.bitwise_equal &=
        memcmp(left, right, (size_t)count * sizeof(*left)) == 0;
    for (index = 0ull; index < count; ++index) {
        double left_value = left[index], right_value = right[index];
        double absolute = fabs(left_value - right_value);
        double scale = fmax(fabs(left_value), fabs(right_value));
        double relative = scale > 0.0 ? absolute / scale : 0.0;

        if (!isfinite(left_value) || !isfinite(right_value)) {
            if ((flags & ATTENTION_COMPARE_NEGATIVE_INFINITY_SENTINEL) &&
                isinf(left_value) && signbit(left_value) &&
                isinf(right_value) && signbit(right_value)) {
                sentinel_count++;
                continue;
            }
            cursor->result.numeric.nonfinite_value_count++;
            attention_compare_fail(cursor, stage, index, 0);
            continue;
        }
        cursor->result.numeric.finite_value_count++;
        cursor->result.numeric.maximum_absolute_error =
            fmax(cursor->result.numeric.maximum_absolute_error, absolute);
        cursor->result.numeric.maximum_relative_error =
            fmax(cursor->result.numeric.maximum_relative_error, relative);
        cursor->result.numeric.squared_error_sum += absolute * absolute;
        if (!isfinite(absolute_tolerance + relative_tolerance * scale) ||
            absolute > absolute_tolerance + relative_tolerance * scale)
            attention_compare_fail(cursor, stage, index, 0);
    }
    if (!yvex_core_u64_add(cursor->result.numeric.value_count,
                           count - sentinel_count, &total)) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "attention.state.compare",
                       "aggregate attention state comparison counts overflowed");
        return YVEX_ERR_BOUNDS;
    }
    cursor->result.numeric.value_count = total;
    return YVEX_OK;
}

/*
 * Execute the canonical ordered state-comparison descriptor table.
 *
 * Malformed storage or arithmetic overflow stops before publishing caller output.
 */
static int attention_compare_state(const yvex_attention_publication *left,
                                   const yvex_attention_publication *right,
                                   double absolute_tolerance, double relative_tolerance,
                                   attention_compare_cursor *cursor, yvex_error *err)
{
    size_t index;

    for (index = 0u; index < sizeof(attention_state_steps) / sizeof(attention_state_steps[0]);
         ++index) {
        const attention_compare_step *step = &attention_state_steps[index];
        const unsigned char *left_base = (const unsigned char *)left + step->base_offset;
        const unsigned char *right_base = (const unsigned char *)right + step->base_offset;
        unsigned long long count = 0ull, item;
        size_t field;
        int rc;

        if (!cursor->compatible[step->group]) continue;
        if (step->kind == ATTENTION_COMPARE_GEOMETRY) {
            for (field = 0u; field < step->field_count; ++field)
                if (!attention_compare_field_equal(left_base, right_base, &step->fields[field]))
                    break;
            if (field != step->field_count) {
                cursor->compatible[step->group] = 0u;
                attention_compare_fail(cursor, step->stage, ULLONG_MAX, 2);
                if (step->flags & ATTENTION_COMPARE_STOP_ON_MISMATCH) break;
            }
            continue;
        }
        if ((step->flags & ATTENTION_COMPARE_REQUIRE_PRESENT) &&
            !*(const int *)(left_base + offsetof(yvex_attention_rolling_state_output, present)))
            continue;
        count = *(const unsigned long long *)(left_base + step->count_offset);
        if ((step->flags & ATTENTION_COMPARE_SKIP_EMPTY) && !count) continue;
        if (step->flags & ATTENTION_COMPARE_PRODUCT) {
            item = *(const unsigned long long *)(left_base + step->stride_offset);
            if (((step->flags & ATTENTION_COMPARE_REQUIRE_STRIDE) && !item) ||
                !yvex_core_u64_mul(count, item, &count)) {
                yvex_error_set(err, YVEX_ERR_BOUNDS, "attention.state.compare",
                               step->overflow_message);
                return YVEX_ERR_BOUNDS;
            }
        }
        if (step->kind == ATTENTION_COMPARE_F32) {
            const float *left_values =
                *(float *const *)(left_base + step->data_offset);
            const float *right_values =
                *(float *const *)(right_base + step->data_offset);
            rc = attention_compare_f32(left_values, right_values, count, absolute_tolerance,
                                       relative_tolerance, step->flags, step->stage, cursor, err);
            if (rc != YVEX_OK) return rc;
        } else {
            const unsigned long long *left_values =
                *(unsigned long long *const *)(left_base + step->data_offset);
            const unsigned long long *right_values =
                *(unsigned long long *const *)(right_base + step->data_offset);
            if (!left_values || !right_values) {
                yvex_error_set(err, YVEX_ERR_INVALID_ARG, "attention.state.compare",
                               "non-empty attention state positions require two ranges");
                return YVEX_ERR_INVALID_ARG;
            }
            for (item = 0ull; item < count; ++item)
                if (left_values[item] != right_values[item]) {
                    attention_compare_fail(cursor, step->stage, item, 1);
                    break;
                }
        }
    }
    return YVEX_OK;
}

/* Compare two finite F32 ranges under one deterministic numeric contract. */
int yvex_graph_f32_compare(const float *left, const float *right,
                           unsigned long long count, double absolute_tolerance,
                           double relative_tolerance, yvex_graph_f32_comparison *out,
                           yvex_error *err)
{
    attention_compare_cursor cursor = {.result = {.geometry_equal = 1}};
    int rc;

    if (!out || !isfinite(absolute_tolerance) || !isfinite(relative_tolerance) ||
        absolute_tolerance < 0.0 || relative_tolerance < 0.0)
        return attention_compare_f32(NULL, NULL, 0ull, 0.0, 0.0, 0u,
                                     YVEX_ATTENTION_COMPARISON_STAGE_NONE, &cursor, err);
    cursor.result.numeric.within_tolerance = cursor.result.numeric.bitwise_equal = 1;
    cursor.result.numeric.first_failing_coordinate = ULLONG_MAX;
    rc = attention_compare_f32(left, right, count, absolute_tolerance, relative_tolerance, 0u,
                               YVEX_ATTENTION_COMPARISON_STAGE_NONE, &cursor, err);
    if (rc == YVEX_OK) *out = cursor.result.numeric;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

/*
 * Compare the complete attention-owned candidate state delta across production paths.
 *
 * Malformed ranges or overflow preserve caller output and return typed failure.
 */
int yvex_attention_state_compare(const yvex_attention_publication *left,
                                 const yvex_attention_publication *right,
                                 double absolute_tolerance, double relative_tolerance,
                                 yvex_attention_state_comparison *out, yvex_error *err)
{
    attention_compare_cursor cursor = {.result = {.geometry_equal = 1},
                                       .compatible = {1u, 1u, 1u, 1u, 1u}};
    int rc;

    if (!left || !right || !out || !isfinite(absolute_tolerance) ||
        !isfinite(relative_tolerance) || absolute_tolerance < 0.0 || relative_tolerance < 0.0) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "attention.state.compare",
                       "two publications and finite non-negative tolerances are required");
        return YVEX_ERR_INVALID_ARG;
    }
    cursor.result.numeric.within_tolerance = cursor.result.numeric.bitwise_equal = 1;
    cursor.result.numeric.first_failing_coordinate = ULLONG_MAX;
    rc = attention_compare_state(left, right, absolute_tolerance, relative_tolerance, &cursor, err);
    if (rc == YVEX_OK) *out = cursor.result;
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_attention_publication_compare(
    const yvex_attention_publication *left,
    const yvex_attention_publication *right,
    double absolute_tolerance, double relative_tolerance,
    yvex_attention_probe_result *result, double *squared_error,
    yvex_error *err)
{
    yvex_graph_f32_comparison output;
    yvex_attention_state_comparison state;
    unsigned long long left_width, right_width, count;
    const float *left_values, *right_values;
    int rc;
    if (!left || !right || !result || !squared_error)
        return YVEX_ERR_INVALID_ARG;
    left_width = left->envelope_output_width
                     ? left->envelope_output_width : left->hidden_width;
    right_width = right->envelope_output_width
                      ? right->envelope_output_width : right->hidden_width;
    left_values = left->envelope_output_width
                      ? left->envelope_output : left->output;
    right_values = right->envelope_output_width
                       ? right->envelope_output : right->output;
    if (!left_values || !right_values || !left->complete ||
        !right->complete || left->layer_index != right->layer_index ||
        left->token_count != right->token_count ||
        left_width != right_width ||
        !yvex_core_u64_mul(left->token_count, left_width, &count))
        return YVEX_ERR_FORMAT;
    rc = yvex_graph_f32_compare(
        left_values, right_values, count, absolute_tolerance,
        relative_tolerance, &output, err);
    if (rc == YVEX_OK)
        rc = yvex_attention_state_compare(
            left, right, absolute_tolerance, relative_tolerance, &state, err);
    if (rc != YVEX_OK) return rc;
    result->bitwise_equality_observed &=
        output.bitwise_equal && state.geometry_equal &&
        state.numeric.bitwise_equal;
    result->comparison_maximum_absolute_error =
        fmax(result->comparison_maximum_absolute_error,
             fmax(output.maximum_absolute_error,
                  state.numeric.maximum_absolute_error));
    result->comparison_maximum_relative_error =
        fmax(result->comparison_maximum_relative_error,
             fmax(output.maximum_relative_error,
                  state.numeric.maximum_relative_error));
    *squared_error +=
        output.squared_error_sum + state.numeric.squared_error_sum;
    result->comparison_output_values += count;
    result->comparison_state_values += state.numeric.value_count;
    result->comparison_values += count + state.numeric.value_count;
    result->comparison_finite_values +=
        output.finite_value_count + state.numeric.finite_value_count;
    result->comparison_nonfinite_values +=
        output.nonfinite_value_count + state.numeric.nonfinite_value_count;
    if (result->first_failing_layer == YVEX_ATTENTION_NO_LAYER &&
        !output.within_tolerance) {
        result->first_failing_layer = left->layer_index;
        result->first_failing_coordinate = output.first_failing_coordinate;
        result->first_failing_stage = YVEX_ATTENTION_COMPARISON_STAGE_OUTPUT;
    } else if (result->first_failing_layer == YVEX_ATTENTION_NO_LAYER &&
               (!state.geometry_equal ||
                !state.numeric.within_tolerance)) {
        result->first_failing_layer = left->layer_index;
        result->first_failing_coordinate =
            state.numeric.first_failing_coordinate;
        result->first_failing_stage = state.first_failing_stage;
    }
    return result->first_failing_layer == left->layer_index
               ? YVEX_ERR_FORMAT : YVEX_OK;
}

static const float *attention_hash_output_values(
    const yvex_attention_publication *publication, unsigned long long *width)
{
    if (!publication || !width) return NULL;
    if (publication->envelope_output_width) {
        *width = publication->envelope_output_width;
        return publication->envelope_output;
    }
    *width = publication->hidden_width;
    return publication->output;
}

static int attention_hash_floats(yvex_sha256 *hash, const float *values,
                                 unsigned long long count)
{
    unsigned long long index;

    if ((count && !values) || !yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        memcpy(&bits, values + index, sizeof(bits));
        if (!yvex_sha256_update_u64(hash, (unsigned long long)bits)) return 0;
    }
    return 1;
}

static int attention_hash_u64s(yvex_sha256 *hash, const unsigned long long *values,
                               unsigned long long count)
{
    unsigned long long index;

    if ((count && !values) || !yvex_sha256_update_u64(hash, count)) return 0;
    for (index = 0ull; index < count; ++index)
        if (!yvex_sha256_update_u64(hash, values[index])) return 0;
    return 1;
}

/*
 * Hash ordered typed scalar fields without serializing object padding.
 *
 * Numeric identity hashes declared values, never native object representation.
 */
static int attention_hash_fields(yvex_sha256 *hash, const void *object,
                                 const attention_compare_field *fields, size_t count)
{
    const unsigned char *bytes = (const unsigned char *)object;
    size_t index;

    for (index = 0u; index < count; ++index) {
        unsigned long long value = 0ull;
        if (fields[index].kind == ATTENTION_COMPARE_TEXT) {
            if (!yvex_sha256_update_text(hash, (const char *)bytes + fields[index].offset))
                return 0;
            continue;
        }
        if (fields[index].kind == ATTENTION_COMPARE_U64)
            memcpy(&value, bytes + fields[index].offset, sizeof(value));
        else if (fields[index].kind == ATTENTION_COMPARE_UINT) {
            unsigned int narrow;
            memcpy(&narrow, bytes + fields[index].offset, sizeof(narrow));
            value = narrow;
        } else {
            int narrow;
            memcpy(&narrow, bytes + fields[index].offset, sizeof(narrow));
            value = (unsigned long long)narrow;
        }
        if (!yvex_sha256_update_u64(hash, value)) return 0;
    }
    return 1;
}

static int attention_hash_rolling(yvex_sha256 *hash,
                                  const yvex_attention_rolling_state_output *state)
{
    if (!state || !yvex_sha256_update_u64(hash, (unsigned long long)state->present)) return 0;
    if (!state->present) return 1;
    return state->kv_state && state->score_state &&
           attention_hash_fields(
               hash, state, attention_rolling_fields + 1u,
               sizeof(attention_rolling_fields) / sizeof(attention_rolling_fields[0]) - 1u) &&
           attention_hash_floats(hash, state->kv_state, state->kv_state_extent) &&
           attention_hash_floats(hash, state->score_state, state->score_state_extent);
}

/*
 * Append one complete publication to distinct output and state evidence digests.
 *
 * Incomplete geometry, missing storage, overflow, or hash refusal returns false.
 */
int yvex_attention_publication_hash_update(
    yvex_sha256 *output_hash, yvex_sha256 *state_hash,
    const yvex_attention_publication *publication)
{
    unsigned long long output_count, output_width, raw_count, compressed_count, indexer_count;
    const float *output_values = attention_hash_output_values(publication, &output_width);

    if (!output_hash || !state_hash || !publication || !publication->complete ||
        !output_values || !publication->raw_kv ||
        !yvex_core_u64_mul(publication->token_count, output_width, &output_count) ||
        !yvex_core_u64_mul(publication->token_count, publication->kv_width, &raw_count) ||
        !yvex_core_u64_mul(publication->compressed_count, publication->compressed_stride,
                           &compressed_count) ||
        !yvex_core_u64_mul(publication->indexer_count, publication->indexer_stride,
                           &indexer_count))
        return 0;
    if (publication->evidence_level == YVEX_ATTENTION_EVIDENCE_NONE)
        return yvex_sha256_hex_valid(publication->execution_identity) &&
               yvex_sha256_update_text(output_hash,
                                       "yvex.attention.output.semantic.v1") &&
               yvex_sha256_update_text(output_hash,
                                       publication->execution_identity) &&
               yvex_sha256_update_u64(output_hash, output_count) &&
               yvex_sha256_update_text(state_hash,
                                       "yvex.attention.state.semantic.v1") &&
               yvex_sha256_update_text(state_hash,
                                       publication->execution_identity) &&
               yvex_sha256_update_u64(state_hash, raw_count) &&
               yvex_sha256_update_u64(state_hash, compressed_count) &&
               yvex_sha256_update_u64(state_hash, indexer_count);
    return attention_hash_fields(output_hash, publication,
                                 attention_publication_fields + 1u, 4u) &&
           yvex_sha256_update_u64(output_hash, output_width) &&
           attention_hash_floats(output_hash, output_values, output_count) &&
           attention_hash_fields(state_hash, publication,
                                 attention_publication_fields + 1u, 5u) &&
           attention_hash_floats(state_hash, publication->raw_kv, raw_count) &&
           yvex_sha256_update_u64(state_hash, publication->compressed_stride) &&
           attention_hash_floats(state_hash, publication->compressed_kv, compressed_count) &&
           attention_hash_u64s(state_hash, publication->compressed_positions,
                               publication->compressed_count) &&
           yvex_sha256_update_u64(state_hash, publication->indexer_stride) &&
           attention_hash_floats(state_hash, publication->indexer_kv, indexer_count) &&
           attention_hash_u64s(state_hash, publication->indexer_positions,
                               publication->indexer_count) &&
           attention_hash_rolling(state_hash, &publication->next_main_rolling_state) &&
           attention_hash_rolling(state_hash, &publication->next_indexer_rolling_state);
}

/* Apply the identity-bearing DeepSeek activation storage boundary. */
int yvex_attention_compute_round(yvex_attention_compute_contract contract,
                                 float *values,
                                 unsigned long long count)
{
    unsigned long long i;

    if (contract != YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1 ||
        !values || !count)
        return 0;
    for (i = 0ull; i < count; ++i) {
        if (!isfinite(values[i])) return 0;
        values[i] = yvex_quant_bf16_decode(yvex_quant_bf16_encode(values[i]));
    }
    return 1;
}

static int attention_rms_norm_apply(float *values, unsigned long long count,
                                    const float *weights, double epsilon)
{
    unsigned long long i;
    double mean = 0.0;
    double inv;

    if (!values || count == 0ull || !isfinite(epsilon) || epsilon <= 0.0)
        return 0;
    for (i = 0ull; i < count; ++i) {
        double v = values[i];
        if (!isfinite(v) || (weights && !isfinite(weights[i]))) return 0;
        mean += v * v;
    }
    mean /= (double)count;
    inv = 1.0 / sqrt(mean + epsilon);
    if (!isfinite(inv)) return 0;
    for (i = 0ull; i < count; ++i) {
        double v = (double)values[i] * inv;
        if (weights) v *= (double)weights[i];
        if (!isfinite(v)) return 0;
        values[i] = (float)v;
    }
    return 1;
}

/*
 * Apply RMS normalization with one learned weight per value.
 *
 * Weight ownership and attention composition remain with the caller.
 */
int yvex_attention_rms_norm(float *values, unsigned long long count,
                            const float *weights, double epsilon)
{
    return weights && attention_rms_norm_apply(values, count, weights, epsilon);
}

int yvex_attention_unit_rms_norm(float *values,
                                 unsigned long long count,
                                 double epsilon)
{
    return attention_rms_norm_apply(values, count, NULL, epsilon);
}

static int attention_mhc_geometry(const yvex_attention_layer_plan *layer)
{
    unsigned long long expanded;
    unsigned long long rows;

    return layer && layer->mhc_attention_pre_and_post &&
           layer->residual_stream_count > 0ull && layer->residual_stream_width > 0ull &&
           yvex_core_u64_mul(layer->residual_stream_count, layer->residual_stream_width,
                             &expanded) &&
           expanded == layer->residual_expanded_width &&
           yvex_core_u64_add(layer->residual_stream_count, 2ull, &rows) &&
           yvex_core_u64_mul(rows, layer->residual_stream_count, &rows) &&
           rows == layer->mhc_mixing_rows &&
           layer->mhc_mixing_columns == expanded && layer->mhc_base_width == rows &&
           layer->mhc_scale_width == 3ull && layer->mhc_sinkhorn_iterations > 0ull &&
           isfinite(layer->rms_norm_epsilon) && layer->rms_norm_epsilon > 0.0 &&
           isfinite(layer->mhc_epsilon) && layer->mhc_epsilon > 0.0 &&
           isfinite(layer->mhc_residual_post_multiplier) &&
           layer->mhc_residual_post_multiplier > 0.0;
}

static double attention_sigmoid(double value)
{
    if (value >= 0.0) {
        double inverse = exp(-value);
        return 1.0 / (1.0 + inverse);
    }
    {
        double direct = exp(value);
        return direct / (1.0 + direct);
    }
}

static int attention_mhc_sinkhorn(float *matrix, unsigned long long streams,
                                  unsigned long long iterations, double epsilon)
{
    unsigned long long row, column, iteration;

    for (row = 0ull; row < streams; ++row) {
        double maximum = -(double)INFINITY;
        double total = 0.0;
        for (column = 0ull; column < streams; ++column)
            maximum = fmax(maximum, matrix[row * streams + column]);
        for (column = 0ull; column < streams; ++column) {
            double value = exp((double)matrix[row * streams + column] - maximum);
            matrix[row * streams + column] = (float)value;
            total += value;
        }
        if (!isfinite(total) || total <= 0.0) return 0;
        for (column = 0ull; column < streams; ++column)
            matrix[row * streams + column] =
                (float)((double)matrix[row * streams + column] / total + epsilon);
    }
    for (iteration = 0ull; iteration < iterations; ++iteration) {
        if (iteration != 0ull) {
            for (row = 0ull; row < streams; ++row) {
                double total = 0.0;
                for (column = 0ull; column < streams; ++column)
                    total += matrix[row * streams + column];
                if (!isfinite(total)) return 0;
                for (column = 0ull; column < streams; ++column)
                    matrix[row * streams + column] =
                        (float)((double)matrix[row * streams + column] / (total + epsilon));
            }
        }
        for (column = 0ull; column < streams; ++column) {
            double total = 0.0;
            for (row = 0ull; row < streams; ++row)
                total += matrix[row * streams + column];
            if (!isfinite(total)) return 0;
            for (row = 0ull; row < streams; ++row)
                matrix[row * streams + column] =
                    (float)((double)matrix[row * streams + column] / (total + epsilon));
        }
    }
    return 1;
}

int yvex_attention_mhc_pre(const yvex_attention_mhc_pre_args *args,
                           yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_attention_layer_plan *layer = args ? args->layer : NULL;
    unsigned long long token, stream, lane;

    if (!args || !attention_mhc_geometry(layer) || !args->residual ||
        !args->linear_mixes || !args->scale || !args->base || !args->collapsed ||
        !args->post || !args->combination || !args->token_count ||
        args->residual_stride < layer->residual_expanded_width ||
        args->mix_stride < layer->mhc_mixing_rows ||
        args->collapsed_stride < layer->residual_stream_width ||
        args->post_stride < layer->residual_stream_count ||
        args->combination_stride <
            layer->residual_stream_count * layer->residual_stream_count)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, 1ull, 0ull, err, YVEX_ERR_BOUNDS,
            "attention mHC ingress geometry is invalid");
    for (lane = 0ull; lane < layer->mhc_scale_width; ++lane)
        if (!isfinite(args->scale[lane])) goto numeric;
    for (lane = 0ull; lane < layer->mhc_base_width; ++lane)
        if (!isfinite(args->base[lane])) goto numeric;
    for (token = 0ull; token < args->token_count; ++token) {
        const float *residual = args->residual + token * args->residual_stride;
        const float *mix = args->linear_mixes + token * args->mix_stride;
        float *collapsed = args->collapsed + token * args->collapsed_stride;
        float *post = args->post + token * args->post_stride;
        float *combination = args->combination + token * args->combination_stride;
        double squares = 0.0;
        double inverse;

        memset(collapsed, 0, (size_t)layer->residual_stream_width * sizeof(*collapsed));
        for (lane = 0ull; lane < layer->residual_expanded_width; ++lane) {
            if (!isfinite(residual[lane])) goto numeric;
            squares += (double)residual[lane] * (double)residual[lane];
        }
        inverse = 1.0 / sqrt(squares / (double)layer->residual_expanded_width +
                             layer->rms_norm_epsilon);
        if (!isfinite(inverse)) goto numeric;
        for (stream = 0ull; stream < layer->residual_stream_count; ++stream) {
            unsigned long long target;
            double pre_value = attention_sigmoid(
                (double)mix[stream] * inverse * (double)args->scale[0] +
                (double)args->base[stream]);
            post[stream] = (float)(layer->mhc_residual_post_multiplier * attention_sigmoid(
                (double)mix[layer->residual_stream_count + stream] * inverse *
                    (double)args->scale[1] +
                (double)args->base[layer->residual_stream_count + stream]));
            for (target = 0ull; target < layer->residual_stream_count; ++target) {
                unsigned long long index = 2ull * layer->residual_stream_count +
                                           stream * layer->residual_stream_count + target;
                combination[stream * layer->residual_stream_count + target] =
                    (float)((double)mix[index] * inverse * (double)args->scale[2] +
                            (double)args->base[index]);
            }
            for (lane = 0ull; lane < layer->residual_stream_width; ++lane)
                collapsed[lane] += (float)((pre_value + layer->mhc_epsilon) *
                    (double)residual[stream * layer->residual_stream_width + lane]);
        }
        if (!attention_mhc_sinkhorn(combination, layer->residual_stream_count,
                                    layer->mhc_sinkhorn_iterations, layer->mhc_epsilon) ||
            !yvex_attention_compute_round(layer->compute_contract, collapsed,
                                          layer->residual_stream_width))
            goto numeric;
    }
    return yvex_attention_accept(failure, err);
numeric:
    return yvex_attention_reject(
        failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
        layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
        YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, 1ull, 0ull, err, YVEX_ERR_FORMAT,
        "attention mHC ingress produced non-finite values");
}

int yvex_attention_mhc_post(const yvex_attention_mhc_post_args *args,
                            yvex_attention_failure *failure, yvex_error *err)
{
    const yvex_attention_layer_plan *layer = args ? args->layer : NULL;
    unsigned long long token, target, lane;

    if (!args || !attention_mhc_geometry(layer) || !args->core_output || !args->residual ||
        !args->post || !args->combination || !args->envelope_output || !args->token_count ||
        args->core_stride < layer->residual_stream_width ||
        args->residual_stride < layer->residual_expanded_width ||
        args->post_stride < layer->residual_stream_count ||
        args->combination_stride <
            layer->residual_stream_count * layer->residual_stream_count ||
        args->envelope_stride < layer->residual_expanded_width)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, 1ull, 0ull, err, YVEX_ERR_BOUNDS,
            "attention mHC egress geometry is invalid");
    for (token = 0ull; token < args->token_count; ++token) {
        const float *core = args->core_output + token * args->core_stride;
        const float *residual = args->residual + token * args->residual_stride;
        const float *post = args->post + token * args->post_stride;
        const float *combination = args->combination + token * args->combination_stride;
        float *output = args->envelope_output + token * args->envelope_stride;
        for (target = 0ull; target < layer->residual_stream_count; ++target) {
            for (lane = 0ull; lane < layer->residual_stream_width; ++lane) {
                unsigned long long source;
                double value = (double)post[target] * (double)core[lane];
                for (source = 0ull; source < layer->residual_stream_count; ++source)
                    value += (double)combination[source * layer->residual_stream_count + target] *
                             (double)residual[source * layer->residual_stream_width + lane];
                if (!isfinite(value)) goto numeric;
                output[target * layer->residual_stream_width + lane] = (float)value;
            }
        }
        if (!yvex_attention_compute_round(layer->compute_contract, output,
                                          layer->residual_expanded_width))
            goto numeric;
    }
    return yvex_attention_accept(failure, err);
numeric:
    return yvex_attention_reject(
        failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
        layer ? layer->layer_index : YVEX_ATTENTION_NO_LAYER,
        YVEX_TENSOR_ROLE_HC_ATTENTION_FUNCTION, 1ull, 0ull, err, YVEX_ERR_FORMAT,
        "attention mHC egress produced non-finite values");
}

static double attention_yarn_frequency(
    const yvex_attention_position_policy *position,
    unsigned long long pair,
    unsigned long long rope_dims)
{
    double exponent;
    double frequency;

    if (!position || rope_dims < 2ull || position->theta <= 1ull) return 0.0;
    exponent = (double)(pair * 2ull) / (double)rope_dims;
    frequency = 1.0 / pow((double)position->theta, exponent);
    if (position->original_context && position->scaling_factor) {
        double denominator = 2.0 * log((double)position->theta);
        double low_dim = (double)rope_dims *
            log((double)position->original_context /
                ((double)position->beta_fast * 2.0 * attention_pi)) /
            denominator;
        double high_dim = (double)rope_dims *
            log((double)position->original_context /
                ((double)position->beta_slow * 2.0 * attention_pi)) /
            denominator;
        double low = floor(low_dim);
        double high = ceil(high_dim);
        double lane = (double)pair;
        double ramp;
        double smooth;

        if (low < 0.0) low = 0.0;
        if (high > (double)rope_dims - 1.0)
            high = (double)rope_dims - 1.0;
        if (low == high) high += 0.001;
        ramp = (lane - low) / (high - low);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        smooth = 1.0 - ramp;
        frequency = frequency / (double)position->scaling_factor *
                        (1.0 - smooth) +
                    frequency * smooth;
    }
    return frequency;
}

int yvex_attention_rope_apply(
    float *values,
    unsigned long long count,
    unsigned long long rope_dims,
    unsigned long long token_position,
    const yvex_attention_position_policy *position,
    int inverse)
{
    unsigned long long start;
    unsigned long long i;

    if (!values || count == 0ull || rope_dims < 2ull || rope_dims > count ||
        !position || position->theta <= 1ull)
        return 0;
    if (rope_dims & 1ull) rope_dims--;
    start = count - rope_dims;
    for (i = 0ull; i < rope_dims; i += 2ull) {
        double frequency = attention_yarn_frequency(position, i / 2ull,
                                                    rope_dims);
        double angle = (double)token_position * frequency;
        double c = cos(angle);
        double s = inverse ? -sin(angle) : sin(angle);
        double x = values[start + i];
        double y = values[start + i + 1ull];
        if (!isfinite(x) || !isfinite(y) || !isfinite(c) || !isfinite(s))
            return 0;
        values[start + i] = (float)(x * c - y * s);
        values[start + i + 1ull] = (float)(x * s + y * c);
    }
    return 1;
}

typedef struct {
    float score;
    unsigned long long ordinal;
    unsigned long long index;
} attention_topk_candidate;

static int attention_score_equal(float left, float right)
{
    if (left == 0.0f && right == 0.0f) return 1;
    return left == right;
}

static int attention_candidate_before(const attention_topk_candidate *left,
                                      const attention_topk_candidate *right)
{
    if (!attention_score_equal(left->score, right->score))
        return left->score > right->score;
    return left->ordinal < right->ordinal;
}

static int attention_candidate_ordinal_compare(const void *left,
                                               const void *right)
{
    const attention_topk_candidate *a =
        (const attention_topk_candidate *)left;
    const attention_topk_candidate *b =
        (const attention_topk_candidate *)right;

    if (a->ordinal < b->ordinal) return -1;
    if (a->ordinal > b->ordinal) return 1;
    return 0;
}

static int attention_candidate_rank_compare(const void *left,
                                            const void *right)
{
    const attention_topk_candidate *a =
        (const attention_topk_candidate *)left;
    const attention_topk_candidate *b =
        (const attention_topk_candidate *)right;

    if (attention_candidate_before(a, b)) return -1;
    if (attention_candidate_before(b, a)) return 1;
    return 0;
}

static float attention_power_of_two_ceil(float value)
{
    int exponent;
    float mantissa;

    if (!isfinite(value) || value <= 0.0f) return 0.0f;
    mantissa = frexpf(value, &exponent);
    if (mantissa > 0.5f) exponent++;
    return ldexpf(1.0f, exponent - 1);
}

int yvex_attention_hadamard_cpu(
    const float *input,
    unsigned long long length,
    float scale,
    int reject_nonfinite,
    float *output,
    yvex_attention_scratch_budget *budget,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    float *scratch = NULL;
    unsigned long long padded_length;
    unsigned long long i;
    unsigned long long step;
    size_t scratch_bytes = 0u;
    int rc;

    if (!input || !output || length == 0ull ||
        !yvex_core_power_of_two_capacity(length, 1ull, 1ull, 1ull, &padded_length)) {
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_INVALID_ARG,
            "Hadamard CPU requires non-empty input and output");
    }
    if (!yvex_attention_scratch_reserve(
            budget, padded_length, sizeof(*scratch), &scratch_bytes))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            budget ? budget->limit_bytes : 0ull,
            budget ? (unsigned long long)budget->live_bytes : 0ull,
            err, YVEX_ERR_BOUNDS,
            "Hadamard CPU scratch budget exceeded");
    scratch = (float *)yvex_attention_scratch_calloc(
        budget, padded_length, sizeof(*scratch));
    if (!scratch)
        rc = yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            padded_length, 0ull, err, YVEX_ERR_NOMEM,
            "Hadamard CPU scratch allocation failed");
    if (!scratch) {
        attention_scratch_release(budget, scratch_bytes);
        return rc;
    }
    for (i = 0ull; i < length; ++i) {
        if (reject_nonfinite && !isfinite(input[i])) {
            yvex_attention_scratch_free(budget, scratch);
            attention_scratch_release(budget, scratch_bytes);
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
                0ull, err, YVEX_ERR_FORMAT,
                "Hadamard CPU refuses non-finite input");
        }
        scratch[i] = input[i];
    }
    for (step = 1ull; step < padded_length; step *= 2ull) {
        unsigned long long block;
        for (block = 0ull; block < padded_length; block += step * 2ull) {
            unsigned long long lane;
            for (lane = 0ull; lane < step; ++lane) {
                float left = scratch[block + lane];
                float right = scratch[block + lane + step];
                scratch[block + lane] = left + right;
                scratch[block + lane + step] = left - right;
            }
        }
    }
    for (i = 0ull; i < length; ++i)
        output[i] = scratch[i] * scale;
    yvex_attention_scratch_free(budget, scratch);
    attention_scratch_release(budget, scratch_bytes);
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_attention_topk_select(
    const float *scores,
    const unsigned long long *ordinals,
    unsigned long long candidate_count,
    unsigned long long k,
    unsigned long long *selected_indices,
    unsigned long long *selected_count,
    yvex_attention_scratch_budget *scratch,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    attention_topk_candidate *candidates;
    unsigned long long i;
    unsigned long long out_count;

    if (selected_count) *selected_count = 0ull;
    if (!scores || !ordinals || !selected_indices || !selected_count) {
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
            0ull, err, YVEX_ERR_INVALID_ARG,
            "top-k selection requires scores, ordinals, and output");
    }
    if (candidate_count == 0ull || k == 0ull) {
        yvex_error_clear(err);
        return YVEX_OK;
    }
    candidates = (attention_topk_candidate *)yvex_attention_scratch_calloc(
        scratch, candidate_count, sizeof(*candidates));
    if (!candidates)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
            candidate_count, 0ull, err, YVEX_ERR_NOMEM,
            "top-k candidate allocation failed");
    for (i = 0ull; i < candidate_count; ++i) {
        if (!isfinite(scores[i])) {
            yvex_attention_scratch_free(scratch, candidates);
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
                candidate_count, i, err, YVEX_ERR_FORMAT,
                "top-k refuses non-finite score");
        }
        candidates[i].score = scores[i];
        candidates[i].ordinal = ordinals[i];
        candidates[i].index = i;
    }
    qsort(candidates, (size_t)candidate_count, sizeof(*candidates),
          attention_candidate_ordinal_compare);
    for (i = 1ull; i < candidate_count; ++i) {
        if (candidates[i - 1ull].ordinal == candidates[i].ordinal) {
            yvex_attention_scratch_free(scratch, candidates);
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN,
                candidate_count, i, err, YVEX_ERR_FORMAT,
                "top-k refuses duplicate candidate ordinal");
        }
    }
    qsort(candidates, (size_t)candidate_count, sizeof(*candidates),
          attention_candidate_rank_compare);
    out_count = attention_min_u64(candidate_count, k);
    for (i = 0ull; i < out_count; ++i)
        selected_indices[i] = candidates[i].index;
    *selected_count = out_count;
    yvex_attention_scratch_free(scratch, candidates);
    yvex_error_clear(err);
    return YVEX_OK;
}

static unsigned char attention_ue8m0_encode_scale(float value)
{
    int exponent;
    float mantissa;

    if (!isfinite(value) || value <= 0.0f) return 0xffu;
    mantissa = frexpf(value, &exponent);
    if (mantissa > 0.5f) exponent++;
    exponent += 126;
    if (exponent < 0) return 0u;
    if (exponent > 254) return 254u;
    return (unsigned char)exponent;
}

static float attention_ue8m0_decode_scale(unsigned char code)
{
    return yvex_quant_e8m0_decode(code);
}

static float attention_fp8_e4m3fn_decode(unsigned char code);

static unsigned char attention_fp8_e4m3fn_encode(float value)
{
    static const float finite_max = 448.0f;
    float magnitude = fabsf(value);
    float best_error = INFINITY;
    unsigned char best = 0u;
    unsigned int code;
    int negative = signbit(value);

    if (!isfinite(value)) return negative ? 0xffu : 0x7fu;
    if (magnitude > finite_max) magnitude = finite_max;
    for (code = 0u; code < 0x7fu; ++code) {
        float decoded = attention_fp8_e4m3fn_decode(
            (unsigned char)code);
        float error = fabsf(decoded - magnitude);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = (unsigned char)code;
        }
    }
    return negative ? (unsigned char)(best | 0x80u) : best;
}

static float attention_fp8_e4m3fn_decode(unsigned char code)
{
    return yvex_quant_fp8_e4m3fn_decode(code);
}

typedef unsigned char (*attention_fake_encode_fn)(float value);
typedef float (*attention_fake_decode_fn)(unsigned char code);

static int attention_fake_quant_block(
    const float *input, unsigned long long count, float *dequantized,
    unsigned char *codes, unsigned char *scale_code, float finite_max,
    float minimum_amax, int clamp_normalized, int clear_scale_first,
    attention_fake_encode_fn encode, attention_fake_decode_fn decode,
    const char *argument_reason, const char *nonfinite_reason,
    const char *scale_reason, yvex_attention_failure *failure,
    yvex_error *err)
{
    unsigned long long i;
    float amax = minimum_amax;
    float scale;

    if (!input || !dequantized || !codes || !scale_code || !count ||
        !encode || !decode)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull,
            err, YVEX_ERR_INVALID_ARG, argument_reason);
    if (clear_scale_first) *scale_code = 0u;
    for (i = 0ull; i < count; ++i) {
        float magnitude;
        if (!isfinite(input[i]))
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, count, i,
                err, YVEX_ERR_FORMAT, nonfinite_reason);
        magnitude = fabsf(input[i]);
        if (magnitude > amax) amax = magnitude;
    }
    *scale_code = attention_ue8m0_encode_scale(
        attention_power_of_two_ceil(amax / finite_max));
    scale = attention_ue8m0_decode_scale(*scale_code);
    if (!isfinite(scale) || scale <= 0.0f)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, count, 0ull,
            err, YVEX_ERR_FORMAT, scale_reason);
    for (i = 0ull; i < count; ++i) {
        float normalized = input[i] / scale;
        if (clamp_normalized && normalized > finite_max) normalized = finite_max;
        if (clamp_normalized && normalized < -finite_max) normalized = -finite_max;
        codes[i] = encode(normalized);
        dequantized[i] = yvex_quant_bf16_decode(
            yvex_quant_bf16_encode(decode(codes[i]) * scale));
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_attention_fp8_fake_quant_block(
    const float *input,
    unsigned long long count,
    float *dequantized,
    unsigned char *codes,
    unsigned char *scale_code,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    return attention_fake_quant_block(
        input, count, dequantized, codes, scale_code, 448.0f, 1.0e-4f,
        1, 0, attention_fp8_e4m3fn_encode, attention_fp8_e4m3fn_decode,
        "FP8 fake quant requires input, output, code, and scale buffers",
        "FP8 fake quant refuses non-finite activation",
        "FP8 fake quant produced invalid UE8M0 scale", failure, err);
}

static unsigned char attention_fp4_e2m1_encode(float value)
{
    static const float values[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    float magnitude = fabsf(value);
    float best_error;
    unsigned char best = 0u;
    unsigned int code;

    if (isnan(value)) return signbit(value) ? 0x8u : 0u;
    if (magnitude > 6.0f) magnitude = 6.0f;
    best_error = fabsf(magnitude - values[0]);
    for (code = 1u; code < 8u; ++code) {
        float error = fabsf(magnitude - values[code]);
        if (error < best_error ||
            (error == best_error && !(code & 1u) && (best & 1u))) {
            best_error = error;
            best = (unsigned char)code;
        }
    }
    return signbit(value) ? (unsigned char)(best | 0x8u) : best;
}

static float attention_fp4_e2m1_decode(unsigned char code)
{
    static const float values[] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    unsigned char magnitude = (unsigned char)(code & 0x7u);
    float value = values[magnitude];

    return (code & 0x8u) ? -value : value;
}

int yvex_attention_fp4_fake_quant_block(
    const float *input,
    unsigned long long count,
    float *dequantized,
    unsigned char *codes,
    unsigned char *scale_code,
    yvex_attention_failure *failure,
    yvex_error *err)
{
    return attention_fake_quant_block(
        input, count, dequantized, codes, scale_code, 6.0f,
        6.0f * ldexpf(1.0f, -126), 0, 1,
        attention_fp4_e2m1_encode, attention_fp4_e2m1_decode,
        "FP4 fake quant requires input, output, code, and scale buffers",
        "FP4 fake quant refuses non-finite activation",
        "FP4 fake quant produced invalid UE8M0 scale", failure, err);
}

static int graph_numeric_reject(yvex_error *err, yvex_status status, const char *message)
{
    yvex_error_set(err, status, "graph.numeric", message);
    return status;
}

int yvex_graph_conv1d_output_length(const yvex_graph_conv1d_geometry *geometry,
                                    unsigned long long *output_length, yvex_error *err)
{
    unsigned long long dilated_kernel;
    unsigned long long padded;
    unsigned long long result;

    if (output_length) *output_length = 0ull;
    if (!geometry || !output_length || !geometry->batch || !geometry->input_channels ||
        !geometry->output_channels || !geometry->input_length || !geometry->kernel_size ||
        !geometry->stride || !geometry->dilation)
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "Conv1D geometry requires nonzero extents");
    if (!yvex_core_u64_mul(geometry->kernel_size - 1ull, geometry->dilation,
                           &dilated_kernel) ||
        !yvex_core_u64_add(dilated_kernel, 1ull, &dilated_kernel))
        return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                    "Conv1D dilated kernel extent overflowed");
    if (geometry->transposed) {
        unsigned long long expanded;
        unsigned long long removed;

        if (geometry->output_padding >= geometry->stride)
            return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                        "transposed Conv1D output padding exceeds its stride");
        if (!yvex_core_u64_mul(geometry->input_length - 1ull, geometry->stride,
                               &expanded) ||
            !yvex_core_u64_mul(geometry->padding, 2ull, &removed) ||
            !yvex_core_u64_add(expanded, dilated_kernel, &result) ||
            !yvex_core_u64_add(result, geometry->output_padding, &result) || result <= removed)
            return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                        "transposed Conv1D output extent is invalid");
        result -= removed;
    } else {
        unsigned long long added;

        if (geometry->output_padding)
            return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                        "ordinary Conv1D has no output padding");
        if (!yvex_core_u64_mul(geometry->padding, 2ull, &added) ||
            !yvex_core_u64_add(geometry->input_length, added, &padded) ||
            padded < dilated_kernel)
            return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                        "Conv1D kernel exceeds padded input extent");
        result = (padded - dilated_kernel) / geometry->stride + 1ull;
    }
    *output_length = result;
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_conv1d_extents(const yvex_graph_conv1d_geometry *geometry,
                                unsigned long long output_length,
                                unsigned long long *input_elements,
                                unsigned long long *weight_elements,
                                unsigned long long *output_elements, yvex_error *err)
{
    unsigned long long input_rows;
    unsigned long long weight_rows;
    unsigned long long output_rows;

    if (!yvex_core_u64_mul(geometry->batch, geometry->input_channels, &input_rows) ||
        !yvex_core_u64_mul(input_rows, geometry->input_length, input_elements) ||
        !yvex_core_u64_mul(geometry->input_channels, geometry->output_channels,
                           &weight_rows) ||
        !yvex_core_u64_mul(weight_rows, geometry->kernel_size, weight_elements) ||
        !yvex_core_u64_mul(geometry->batch, geometry->output_channels, &output_rows) ||
        !yvex_core_u64_mul(output_rows, output_length, output_elements))
        return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                    "Conv1D tensor extent overflowed");
    return YVEX_OK;
}

static int graph_conv1d_scale(const float *weight, unsigned long long base,
                              unsigned long long count, const float *gain,
                              unsigned long long gain_index, float *scale,
                              yvex_error *err)
{
    double squared = 0.0;
    unsigned long long index;

    *scale = 1.0f;
    if (!gain) return YVEX_OK;
    for (index = 0ull; index < count; ++index) {
        double value = weight[base + index];
        squared += value * value;
    }
    if (!isfinite(squared) || squared <= 0.0 || !isfinite(gain[gain_index]))
        return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                    "Conv1D weight normalization is not finite");
    *scale = gain[gain_index] / sqrtf((float)squared);
    if (!isfinite(*scale))
        return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                    "Conv1D weight normalization scale is not finite");
    return YVEX_OK;
}

static int graph_conv1d_forward(const yvex_graph_conv1d_geometry *geometry,
                                const float *input, const float *weight,
                                const float *bias, const float *gain,
                                unsigned long long output_length, float *output,
                                yvex_error *err)
{
    unsigned long long batch;
    unsigned long long output_channel;

    for (output_channel = 0ull; output_channel < geometry->output_channels;
         ++output_channel) {
        unsigned long long weight_base = output_channel * geometry->input_channels *
                                         geometry->kernel_size;
        float scale;
        int rc = graph_conv1d_scale(
            weight, weight_base, geometry->input_channels * geometry->kernel_size,
            gain, output_channel, &scale, err);
        if (rc != YVEX_OK) return rc;
        for (batch = 0ull; batch < geometry->batch; ++batch) {
            unsigned long long output_position;
            for (output_position = 0ull; output_position < output_length;
                 ++output_position) {
                float sum = bias ? bias[output_channel] : 0.0f;
                unsigned long long input_channel;
                for (input_channel = 0ull; input_channel < geometry->input_channels;
                     ++input_channel) {
                    unsigned long long kernel;
                    for (kernel = 0ull; kernel < geometry->kernel_size; ++kernel) {
                        unsigned long long projected = output_position * geometry->stride +
                                                       kernel * geometry->dilation;
                        unsigned long long input_position;
                        unsigned long long input_index;
                        unsigned long long weight_index;
                        if (projected < geometry->padding) continue;
                        input_position = projected - geometry->padding;
                        if (input_position >= geometry->input_length) continue;
                        input_index = (batch * geometry->input_channels + input_channel) *
                                      geometry->input_length + input_position;
                        weight_index = weight_base + input_channel * geometry->kernel_size +
                                       kernel;
                        sum += input[input_index] * weight[weight_index] * scale;
                    }
                }
                if (!isfinite(sum))
                    return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                                "Conv1D produced a non-finite value");
                output[(batch * geometry->output_channels + output_channel) * output_length +
                       output_position] = sum;
            }
        }
    }
    return YVEX_OK;
}

static int graph_conv1d_transposed(const yvex_graph_conv1d_geometry *geometry,
                                   const float *input, const float *weight,
                                   const float *bias, const float *gain,
                                   unsigned long long output_length, float *output,
                                   yvex_error *err)
{
    unsigned long long output_elements = geometry->batch * geometry->output_channels *
                                         output_length;
    unsigned long long index;
    unsigned long long batch;

    for (index = 0ull; index < output_elements; ++index)
        output[index] = bias ? bias[(index / output_length) % geometry->output_channels] : 0.0f;
    for (batch = 0ull; batch < geometry->batch; ++batch) {
        unsigned long long input_channel;
        for (input_channel = 0ull; input_channel < geometry->input_channels; ++input_channel) {
            unsigned long long weight_base = input_channel * geometry->output_channels *
                                             geometry->kernel_size;
            float scale;
            int rc = graph_conv1d_scale(
                weight, weight_base, geometry->output_channels * geometry->kernel_size,
                gain, input_channel, &scale, err);
            if (rc != YVEX_OK) return rc;
            for (index = 0ull; index < geometry->input_length; ++index) {
                float value = input[(batch * geometry->input_channels + input_channel) *
                                    geometry->input_length + index] * scale;
                unsigned long long output_channel;
                for (output_channel = 0ull; output_channel < geometry->output_channels;
                     ++output_channel) {
                    unsigned long long kernel;
                    for (kernel = 0ull; kernel < geometry->kernel_size; ++kernel) {
                        unsigned long long projected = index * geometry->stride +
                                                       kernel * geometry->dilation;
                        unsigned long long output_position;
                        unsigned long long output_index;
                        unsigned long long weight_index;
                        if (projected < geometry->padding) continue;
                        output_position = projected - geometry->padding;
                        if (output_position >= output_length) continue;
                        output_index = (batch * geometry->output_channels + output_channel) *
                                       output_length + output_position;
                        weight_index = weight_base + output_channel * geometry->kernel_size +
                                       kernel;
                        output[output_index] += value * weight[weight_index];
                    }
                }
            }
        }
    }
    for (index = 0ull; index < output_elements; ++index)
        if (!isfinite(output[index]))
            return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                        "transposed Conv1D produced a non-finite value");
    return YVEX_OK;
}

int yvex_graph_conv1d_f32(const yvex_graph_conv1d_geometry *geometry,
                          const float *input, unsigned long long input_count,
                          const float *weight, unsigned long long weight_count,
                          const float *bias, unsigned long long bias_count,
                          const float *gain, unsigned long long gain_count,
                          float *output, unsigned long long output_count,
                          yvex_error *err)
{
    unsigned long long output_length;
    unsigned long long expected_input;
    unsigned long long expected_weight;
    unsigned long long expected_output;
    unsigned long long normalized_channels;
    int rc;

    if (!geometry || !input || !weight || !output)
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "Conv1D requires geometry, input, weight, and output");
    rc = yvex_graph_conv1d_output_length(geometry, &output_length, err);
    if (rc != YVEX_OK) return rc;
    rc = graph_conv1d_extents(geometry, output_length, &expected_input,
                              &expected_weight, &expected_output, err);
    if (rc != YVEX_OK) return rc;
    normalized_channels = geometry->transposed ? geometry->input_channels :
                                                 geometry->output_channels;
    if (input_count != expected_input || weight_count != expected_weight ||
        output_count != expected_output || (bias && bias_count != geometry->output_channels) ||
        (!bias && bias_count) || (gain && gain_count != normalized_channels) ||
        (!gain && gain_count))
        return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                    "Conv1D tensor extents differ from geometry");
    rc = geometry->transposed ?
             graph_conv1d_transposed(geometry, input, weight, bias, gain,
                                     output_length, output, err) :
             graph_conv1d_forward(geometry, input, weight, bias, gain,
                                  output_length, output, err);
    if (rc == YVEX_OK) yvex_error_clear(err);
    return rc;
}

int yvex_graph_alias_snake_f32(const float *input, unsigned long long batch,
                               unsigned long long channels, unsigned long long length,
                               const float *alpha_log, const float *beta_log,
                               const float up_filter[12], const float down_filter[12],
                               float *output, float *scratch,
                               unsigned long long scratch_count, yvex_error *err)
{
    unsigned long long doubled;
    unsigned long long groups;
    unsigned long long total_elements;
    unsigned long long padded_length;
    unsigned long long group;

    if (!input || !batch || !channels || !length || !alpha_log || !beta_log ||
        !up_filter || !down_filter || !output || !scratch ||
        !yvex_core_u64_mul(batch, channels, &groups) ||
        !yvex_core_u64_mul(groups, length, &total_elements) ||
        total_elements > (unsigned long long)SIZE_MAX / sizeof(float) ||
        !yvex_core_u64_mul(length, 2ull, &doubled) ||
        !yvex_core_u64_add(length, 10ull, &padded_length) ||
        padded_length > (~0ull - 15ull) / 2ull || scratch_count < doubled)
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "alias-free SnakeBeta requires complete bounded buffers");
    for (group = 0ull; group < groups; ++group) {
        unsigned long long position;
        unsigned long long channel = group % channels;
        float alpha = expf(alpha_log[channel]);
        float beta = expf(beta_log[channel]);

        if (!isfinite(alpha) || !isfinite(beta) || beta <= 0.0f)
            return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                        "SnakeBeta parameters are not finite");
        for (position = 0ull; position < doubled; ++position) {
            unsigned long long raw_position = position + 15ull;
            float value = 0.0f;
            unsigned long long padded_position;
            for (padded_position = 0ull; padded_position < padded_length;
                 ++padded_position) {
                unsigned long long projected = padded_position * 2ull;
                unsigned long long kernel;
                unsigned long long source_position;
                if (raw_position < projected || raw_position - projected >= 12ull) continue;
                kernel = raw_position - projected;
                source_position = padded_position < 5ull ? 0ull : padded_position - 5ull;
                if (source_position >= length) source_position = length - 1ull;
                value += input[group * length + source_position] * up_filter[kernel];
            }
            value *= 2.0f;
            value += sinf(alpha * value) * sinf(alpha * value) / (beta + 1.0e-9f);
            if (!isfinite(value))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "SnakeBeta produced a non-finite value");
            scratch[position] = value;
        }
        for (position = 0ull; position < length; ++position) {
            float value = 0.0f;
            unsigned long long kernel;
            for (kernel = 0ull; kernel < 12ull; ++kernel) {
                unsigned long long padded = position * 2ull + kernel;
                unsigned long long source_position = padded < 5ull ? 0ull : padded - 5ull;
                if (source_position >= doubled) source_position = doubled - 1ull;
                value += scratch[source_position] * down_filter[kernel];
            }
            if (!isfinite(value))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "SnakeBeta downsampling produced a non-finite value");
            output[group * length + position] = value;
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Project source-layout [out, in] F32 weights without inventing a physical transpose. */
int yvex_graph_linear_source_f32(
    const float *input, unsigned long long input_count, unsigned long long rows,
    unsigned long long input_width, const float *weight,
    unsigned long long weight_count, const float *bias,
    unsigned long long bias_count, unsigned long long output_width,
    float *output, unsigned long long output_count, yvex_error *err)
{
    unsigned long long expected_input, expected_weight, expected_output;
    unsigned long long row, column;

    if (!input || !rows || !input_width || !weight || !output || !output_width ||
        !yvex_core_u64_mul(rows, input_width, &expected_input) ||
        !yvex_core_u64_mul(output_width, input_width, &expected_weight) ||
        !yvex_core_u64_mul(rows, output_width, &expected_output) ||
        expected_input > (unsigned long long)SIZE_MAX / sizeof(float) ||
        expected_weight > (unsigned long long)SIZE_MAX / sizeof(float) ||
        expected_output > (unsigned long long)SIZE_MAX / sizeof(float) ||
        expected_input != input_count || expected_weight != weight_count ||
        expected_output != output_count || (bias && bias_count != output_width) ||
        (!bias && bias_count))
        return graph_numeric_reject(err, YVEX_ERR_BOUNDS,
                                    "linear F32 extents differ from source geometry");
    for (row = 0ull; row < rows; ++row) {
        const float *input_row = input + row * input_width;
        for (column = 0ull; column < output_width; ++column) {
            const float *weight_row = weight + column * input_width;
            float sum = bias ? bias[column] : 0.0f;
            unsigned long long inner;

            for (inner = 0ull; inner < input_width; ++inner)
                sum += input_row[inner] * weight_row[inner];
            if (!isfinite(sum))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "linear F32 produced a non-finite value");
            output[row * output_width + column] = sum;
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Apply the source LayerNorm contract in place over independently normalized rows. */
int yvex_graph_layer_norm_f32(float *values, unsigned long long rows,
                              unsigned long long width, const float *weight,
                              const float *bias, double epsilon, yvex_error *err)
{
    unsigned long long row;

    if (!values || !rows || !width || !weight || !bias || !isfinite(epsilon) ||
        epsilon <= 0.0 || rows > ULLONG_MAX / width ||
        rows * width > (unsigned long long)SIZE_MAX / sizeof(float))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "LayerNorm requires bounded rows, affine values, and epsilon");
    for (row = 0ull; row < rows; ++row) {
        float *current = values + row * width;
        double mean = 0.0, variance = 0.0, inverse;
        unsigned long long index;

        for (index = 0ull; index < width; ++index) {
            if (!isfinite(current[index]) || !isfinite(weight[index]) ||
                !isfinite(bias[index]))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "LayerNorm input or affine value is non-finite");
            mean += current[index];
        }
        mean /= (double)width;
        for (index = 0ull; index < width; ++index) {
            double centered = (double)current[index] - mean;
            variance += centered * centered;
        }
        inverse = 1.0 / sqrt(variance / (double)width + epsilon);
        if (!isfinite(inverse))
            return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                        "LayerNorm variance is not finite");
        for (index = 0ull; index < width; ++index) {
            double normalized = ((double)current[index] - mean) * inverse;
            double result = normalized * weight[index] + bias[index];
            if (!isfinite(result))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "LayerNorm produced a non-finite value");
            current[index] = (float)result;
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_silu_gate_f32(const float *fused, unsigned long long rows,
                             unsigned long long width, float *output,
                             yvex_error *err)
{
    unsigned long long row, index;

    if (!fused || !rows || !width || !output || width > ULLONG_MAX / 2ull ||
        rows > ULLONG_MAX / (width * 2ull) ||
        rows * width > (unsigned long long)SIZE_MAX / sizeof(float))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "gated SiLU requires bounded fused rows and output");
    for (row = 0ull; row < rows; ++row) {
        const float *gate = fused + row * width * 2ull;
        const float *value = gate + width;
        for (index = 0ull; index < width; ++index) {
            float result = gate[index] / (1.0f + expf(-gate[index])) * value[index];
            if (!isfinite(result))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "gated SiLU produced a non-finite value");
            output[row * width + index] = result;
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Execute full noncausal attention over interleaved [head, Q/K/V, width] rows. */
int yvex_graph_full_attention_f32(
    const float *qkv, unsigned long long rows, unsigned long long heads,
    unsigned long long head_width, float *output, float *scratch,
    unsigned long long scratch_count, yvex_error *err)
{
    unsigned long long hidden, qkv_width, output_elements, qkv_elements;
    unsigned long long query, head;
    float scale;

    if (!qkv || !rows || !heads || !head_width || !output || !scratch ||
        !yvex_core_u64_mul(heads, head_width, &hidden) ||
        !yvex_core_u64_mul(hidden, 3ull, &qkv_width) ||
        !yvex_core_u64_mul(rows, hidden, &output_elements) ||
        !yvex_core_u64_mul(rows, qkv_width, &qkv_elements) ||
        output_elements > (unsigned long long)SIZE_MAX / sizeof(float) ||
        qkv_elements > (unsigned long long)SIZE_MAX / sizeof(float) ||
        scratch_count < rows)
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "full attention requires bounded QKV geometry and scratch");
    scale = 1.0f / sqrtf((float)head_width);
    memset(output, 0, (size_t)output_elements * sizeof(*output));
    for (query = 0ull; query < rows; ++query) {
        for (head = 0ull; head < heads; ++head) {
            const float *q = qkv + query * qkv_width + head * head_width * 3ull;
            float maximum = -INFINITY, sum = 0.0f;
            unsigned long long key, coordinate;

            for (key = 0ull; key < rows; ++key) {
                const float *k = qkv + key * qkv_width +
                                 head * head_width * 3ull + head_width;
                float score = 0.0f;
                for (coordinate = 0ull; coordinate < head_width; ++coordinate)
                    score += q[coordinate] * k[coordinate];
                scratch[key] = score * scale;
                if (scratch[key] > maximum) maximum = scratch[key];
            }
            for (key = 0ull; key < rows; ++key) {
                scratch[key] = expf(scratch[key] - maximum);
                sum += scratch[key];
            }
            if (!isfinite(sum) || sum <= 0.0f)
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "full attention softmax is not finite");
            for (key = 0ull; key < rows; ++key) {
                const float *value = qkv + key * qkv_width +
                                     head * head_width * 3ull + head_width * 2ull;
                float probability = scratch[key] / sum;
                float *destination = output + query * hidden + head * head_width;
                for (coordinate = 0ull; coordinate < head_width; ++coordinate)
                    destination[coordinate] += probability * value[coordinate];
            }
        }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_interleaved_qk_norm_f32(
    float *qkv, unsigned long long rows, unsigned long long heads,
    unsigned long long head_width, double epsilon, yvex_error *err)
{
    unsigned long long qkv_width, row, head;
    if (!qkv || !rows || !heads || !head_width || !isfinite(epsilon) || epsilon <= 0.0 ||
        !yvex_core_u64_mul(heads, head_width, &qkv_width) ||
        !yvex_core_u64_mul(qkv_width, 3ull, &qkv_width))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "interleaved Q/K normalization requires bounded geometry");
    for (row = 0ull; row < rows; ++row)
        for (head = 0ull; head < heads; ++head) {
            float *base = qkv + row * qkv_width + head * head_width * 3ull;
            if (!yvex_attention_unit_rms_norm(base, head_width, epsilon) ||
                !yvex_attention_unit_rms_norm(base + head_width, head_width, epsilon))
                return graph_numeric_reject(
                    err, YVEX_ERR_FORMAT,
                    "interleaved Q/K normalization produced a non-finite value");
        }
    yvex_error_clear(err);
    return YVEX_OK;
}

static int graph_rope_half_f32(
    float *values, unsigned long long heads, unsigned long long head_width,
    unsigned long long rotary_width, const float *cosines,
    const float *sines, yvex_error *err)
{
    unsigned long long head, lane, half = rotary_width / 2ull;
    if (!values || !heads || !head_width || !rotary_width || rotary_width > head_width ||
        rotary_width % 2ull || !cosines || !sines)
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "half-split RoPE requires bounded even geometry");
    for (head = 0ull; head < heads; ++head)
        for (lane = 0ull; lane < half; ++lane) {
            float *value = values + head * head_width;
            float first = value[lane], second = value[half + lane];
            float rotated_first = first * cosines[lane] - second * sines[lane];
            float rotated_second = second * cosines[lane] + first * sines[lane];
            if (!isfinite(rotated_first) || !isfinite(rotated_second))
                return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                            "half-split RoPE produced a non-finite value");
            value[lane] = rotated_first;
            value[half + lane] = rotated_second;
        }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_rope_3d_row_f32(
    unsigned long long token, unsigned long long frames,
    unsigned long long height, unsigned long long width,
    unsigned long long frequencies, float base,
    float *cosines, float *sines, yvex_error *err)
{
    const float tau = 6.28318530717958647692f;
    float coordinates[3] = {0.0f, 0.0f, 0.0f};
    unsigned long long plane, patches, coordinate, frequency;
    if (!frames || !height || !width || !frequencies || !isfinite(base) || base <= 0.0f ||
        !cosines || !sines || !yvex_core_u64_mul(height, width, &plane) ||
        !yvex_core_u64_mul(frames, plane, &patches))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "3D RoPE row requires bounded normalized geometry");
    if (token < patches) {
        unsigned long long spatial = token % plane;
        coordinates[0] = 2.0f * ((float)(token / plane) + 0.5f) / (float)frames - 1.0f;
        coordinates[1] = 2.0f * ((float)(spatial / width) + 0.5f) / (float)height - 1.0f;
        coordinates[2] = 2.0f * ((float)(spatial % width) + 0.5f) / (float)width - 1.0f;
    }
    for (coordinate = 0ull; coordinate < 3ull; ++coordinate)
        for (frequency = 0ull; frequency < frequencies; ++frequency) {
            unsigned long long index = coordinate * frequencies + frequency;
            float angle = tau * coordinates[coordinate] *
                          powf(base, -(float)frequency / (float)frequencies);
            cosines[index] = cosf(angle);
            sines[index] = sinf(angle);
        }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_rope_3d_interleaved_qk_f32(
    float *qkv, unsigned long long rows, unsigned long long frames,
    unsigned long long height, unsigned long long width,
    unsigned long long heads, unsigned long long head_width,
    unsigned long long frequencies, float base, yvex_error *err)
{
    float cosines[96], sines[96];
    unsigned long long rotary_width, qkv_width, row, head, kind;
    if (!qkv || !rows || !heads || !head_width || !frequencies || frequencies > 32ull ||
        !yvex_core_u64_mul(frequencies, 6ull, &rotary_width) ||
        rotary_width > head_width || !yvex_core_u64_mul(heads, head_width, &qkv_width) ||
        !yvex_core_u64_mul(qkv_width, 3ull, &qkv_width))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "interleaved 3D RoPE requires bounded Q/K geometry");
    for (row = 0ull; row < rows; ++row) {
        int rc = yvex_graph_rope_3d_row_f32(
            row, frames, height, width, frequencies, base, cosines, sines, err);
        if (rc != YVEX_OK) return rc;
        for (head = 0ull; head < heads; ++head)
            for (kind = 0ull; kind < 2ull; ++kind) {
                float *values = qkv + row * qkv_width + head * head_width * 3ull +
                                kind * head_width;
                rc = graph_rope_half_f32(
                    values, 1ull, head_width, rotary_width, cosines, sines, err);
                if (rc != YVEX_OK) return rc;
            }
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_graph_scaled_residual_f32(
    float *hidden, const float *delta, const float *scale,
    unsigned long long rows, unsigned long long width, yvex_error *err)
{
    unsigned long long count, index;
    if (!hidden || !delta || !scale || !rows || !width ||
        !yvex_core_u64_mul(rows, width, &count))
        return graph_numeric_reject(err, YVEX_ERR_INVALID_ARG,
                                    "scaled residual requires bounded nonempty tensors");
    for (index = 0ull; index < count; ++index) {
        float value = hidden[index] + delta[index] * scale[index % width];
        if (!isfinite(value))
            return graph_numeric_reject(err, YVEX_ERR_FORMAT,
                                        "scaled residual produced a non-finite value");
        hidden[index] = value;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}
