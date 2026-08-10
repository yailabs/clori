/*
 * Centralize reusable graph state and payload-bound attention computation.
 *
 * Graph facts and positioned reads preserve admitted descriptor geometry. Graph construction and
 * bounded operations are not transformer or generation support.
 */
#include "src/graph/private.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int graph_hash_f32_values(yvex_sha256 *hash, const float *values,
                                 unsigned long long count)
{
    unsigned long long index;
    if ((count && !values) || count > SIZE_MAX / sizeof(float)) return 0;
    for (index = 0ull; index < count; ++index) {
        uint32_t bits;
        unsigned char bytes[4];
        if (!isfinite(values[index])) return 0;
        memcpy(&bits, values + index, sizeof(bits));
        bytes[0] = (unsigned char)(bits >> 24u);
        bytes[1] = (unsigned char)(bits >> 16u);
        bytes[2] = (unsigned char)(bits >> 8u);
        bytes[3] = (unsigned char)bits;
        if (!yvex_sha256_update(hash, bytes, sizeof(bytes))) return 0;
    }
    return 1;
}

int yvex_graph_f32_execution_identity(
    const char *domain, const char *artifact_identity,
    const unsigned long long *geometry, unsigned long long geometry_count,
    const float *input, unsigned long long input_count,
    const float *output, unsigned long long output_count, char identity[65])
{
    yvex_sha256 hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long index;
    if (!domain || !domain[0] || !yvex_sha256_hex_valid(artifact_identity) ||
        !geometry || !geometry_count || !input || !input_count || !output ||
        !output_count || !identity) return 0;
    yvex_sha256_init(&hash);
    if (!yvex_sha256_update_text(&hash, domain) ||
        !yvex_sha256_update_text(&hash, artifact_identity)) return 0;
    for (index = 0ull; index < geometry_count; ++index)
        if (!yvex_sha256_update_u64_be(&hash, geometry[index])) return 0;
    if (!graph_hash_f32_values(&hash, input, input_count) ||
        !graph_hash_f32_values(&hash, output, output_count) ||
        !yvex_sha256_final(&hash, digest)) return 0;
    yvex_sha256_hex(digest, identity);
    return 1;
}
struct yvex_attention_workspace {
    unsigned char *arena;
    yvex_attention_workspace_summary summary;
};

/*
 * Derive the reusable numerical graph arena from family scratch policy.
 *
 * Refuses absent policy or overflow.
 */
int yvex_attention_workspace_capacity_resolve(
    const yvex_graph_family_api *family, const yvex_attention_plan *plan,
    unsigned long long *arena_bytes, yvex_error *err)
{
    const yvex_attention_summary *summary;
    yvex_attention_cpu_options options;
    unsigned long long required_arena;
    if (arena_bytes) *arena_bytes = 0ull;
    if (!family || !plan || !arena_bytes || !family->plan_summary ||
        !family->cpu_options_default) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "attention.workspace",
                       "sealed plan and family workspace policy are required");
        return YVEX_ERR_INVALID_ARG;
    }
    summary = family->plan_summary(plan);
    memset(&options, 0, sizeof(options));
    family->cpu_options_default(&options);
    if (!summary || !summary->full_execution_ready || !options.scratch_limit_bytes ||
        !yvex_core_u64_mul(options.scratch_limit_bytes, 3ull, &required_arena)) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.workspace",
                       "attention plan has no bounded executable workspace policy");
        return YVEX_ERR_STATE;
    }
    *arena_bytes = required_arena;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Allocate the one reusable host arena during cold session preparation.
 *
 * Exclusive output ownership and a checked nonzero byte capacity.
 */
int yvex_attention_workspace_open(yvex_attention_workspace **out,
                                  unsigned long long capacity_bytes, yvex_error *err)
{
    yvex_attention_workspace *workspace;
    if (out) *out = NULL;
    if (!out || !capacity_bytes || capacity_bytes > (unsigned long long)SIZE_MAX) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "attention.workspace",
                       "attention workspace capacity is invalid");
        return YVEX_ERR_BOUNDS;
    }
    workspace = (yvex_attention_workspace *)calloc(1u, sizeof(*workspace));
    if (workspace) workspace->arena = (unsigned char *)calloc(1u, (size_t)capacity_bytes);
    if (!workspace || !workspace->arena) {
        free(workspace);
        yvex_error_set(err, YVEX_ERR_NOMEM, "attention.workspace",
                       "attention workspace cold allocation failed");
        return YVEX_ERR_NOMEM;
    }
    workspace->summary.capacity_bytes = capacity_bytes;
    workspace->summary.open = 1;
    *out = workspace;
    yvex_error_clear(err);
    return YVEX_OK;
}

/* Begin one exclusive execution lifetime without resizing storage. */
int yvex_attention_workspace_begin(yvex_attention_workspace *workspace, yvex_error *err)
{
    if (!workspace || !workspace->summary.open || workspace->summary.active) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.workspace",
                       "attention workspace is absent or already active");
        return YVEX_ERR_STATE;
    }
    workspace->summary.used_bytes = 0ull;
    workspace->summary.active = 1;
    workspace->summary.generation++;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Borrow the current nested allocation boundary without mutation.
 *
 * Returns its current byte cursor without changing ownership.
 */
unsigned long long yvex_attention_workspace_mark(const yvex_attention_workspace *workspace)
{
    return workspace && workspace->summary.active ? workspace->summary.used_bytes : ULLONG_MAX;
}

/*
 * Retire one nested lifetime after every borrowed span is consumed.
 *
 * Callers must publish output and state evidence before rewinding.
 */
int yvex_attention_workspace_rewind(yvex_attention_workspace *workspace,
                                    unsigned long long mark, yvex_error *err)
{
    if (!workspace || !workspace->summary.active || mark > workspace->summary.used_bytes) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.workspace",
                       "attention workspace rewind boundary is invalid");
        return YVEX_ERR_STATE;
    }
    workspace->summary.used_bytes = mark;
    workspace->summary.rewind_count++;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Finish one execution only when no borrowed arena spans remain.
 *
 * Ends exclusive execution ownership without freeing storage. Cold close, not finish, releases the
 * process-lifetime arena.
 */
int yvex_attention_workspace_finish(yvex_attention_workspace *workspace, yvex_error *err)
{
    if (!workspace || !workspace->summary.active || workspace->summary.used_bytes) {
        yvex_error_set(err, YVEX_ERR_STATE, "attention.workspace",
                       "attention workspace still owns live execution spans");
        return YVEX_ERR_STATE;
    }
    workspace->summary.active = 0;
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Acquire one aligned zeroed range without heap fallback or cursor drift.
 *
 * Overflow or exhaustion increments refusal evidence and returns null.
 */
void *yvex_attention_workspace_calloc(yvex_attention_workspace *workspace,
                                      unsigned long long count, unsigned long long width)
{
    const unsigned long long alignment = (unsigned long long)_Alignof(max_align_t);
    unsigned long long bytes, aligned, next;
    void *result;
    if (!workspace || !workspace->summary.active || !count || !width ||
        !yvex_core_u64_mul(count, width, &bytes) ||
        !yvex_core_u64_add(workspace->summary.used_bytes, alignment - 1ull, &aligned))
        return NULL;
    aligned -= aligned % alignment;
    if (!yvex_core_u64_add(aligned, bytes, &next) ||
        next > workspace->summary.capacity_bytes) {
        workspace->summary.capacity_failure_count++;
        return NULL;
    }
    result = workspace->arena + aligned;
    memset(result, 0, (size_t)bytes);
    workspace->summary.used_bytes = next;
    if (next > workspace->summary.peak_bytes) workspace->summary.peak_bytes = next;
    workspace->summary.acquisition_count++;
    return result;
}

/*
 * Borrow immutable arena counters for runtime accounting.
 *
 * Runtime may project counters but cannot mutate graph ownership.
 */
const yvex_attention_workspace_summary *yvex_attention_workspace_summary_get(
    const yvex_attention_workspace *workspace)
{
    return workspace ? &workspace->summary : NULL;
}

/* Release the cold arena allocation through exclusive pointer ownership. */
void yvex_attention_workspace_close(yvex_attention_workspace **workspace_ptr)
{
    yvex_attention_workspace *workspace;
    if (!workspace_ptr || !*workspace_ptr) return;
    workspace = *workspace_ptr;
    *workspace_ptr = NULL;
    free(workspace->arena);
    memset(workspace, 0, sizeof(*workspace));
    free(workspace);
}

/*
 * Allocate scratch from an attached arena or cold compatibility heap.
 *
 * Overflow, capacity exhaustion, or allocation failure returns null.
 */
void *yvex_attention_scratch_calloc(yvex_attention_scratch_budget *budget,
                                    unsigned long long count, unsigned long long width)
{
    return budget && budget->workspace
               ? yvex_attention_workspace_calloc(budget->workspace, count, width)
               : yvex_attention_calloc_array(count, width);
}

void yvex_attention_scratch_free(yvex_attention_scratch_budget *budget, void *allocation)
{
    if (!budget || !budget->workspace) free(allocation);
}

int yvex_attention_execution_admit(
    const yvex_attention_plan *plan, const char *logical_identity,
    yvex_materialization_session *session, const yvex_runtime_descriptor *descriptor,
    const yvex_attention_cpu_options *options, const char *cancel_stage,
    unsigned long long csa_ratio, unsigned long long hca_ratio,
    const yvex_attention_layer_plan **layer, yvex_attention_failure *failure,
    yvex_error *err)
{
    int rc = yvex_attention_context_validate(
        plan, logical_identity, session, descriptor, failure, err);
    if (rc != YVEX_OK) return rc;
    if (!options || !layer || (options->publication && options->trace) ||
        (options->publication && options->publication->owned) ||
        (options->trace && options->trace->owned))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            options ? options->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_TENSOR_ROLE_UNKNOWN, 0ull, 1ull, err, YVEX_ERR_STATE,
            "attention publication must be singular and released before reuse");
    /* A draft plan starts at the target layer coordinate where its feature tap
     * lives; that coordinate is not its zero-based ordinal inside the draft
     * plan. Execution selects the semantic coordinate, while traversal remains
     * ordinal-based. */
    *layer = yvex_attention_plan_layer_find(plan, options->layer_index);
    if (!*layer)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ARCHITECTURE, NULL,
            options->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_BOUNDS, "attention execution layer is absent");
    if (options->candidate_block_visible &&
        ((*layer)->tensor_scope != YVEX_TENSOR_SCOPE_DRAFT ||
         (*layer)->attention_class != YVEX_ATTENTION_CLASS_SWA ||
         options->token_count < 2ull))
        return yvex_attention_reject(
            failure, YVEX_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            (*layer)->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, 2ull,
            options->token_count, err, YVEX_ERR_INVALID_ARG,
            "candidate-block visibility requires multi-token draft SWA execution");
    if (options->operation_scope != YVEX_ATTENTION_OPERATION_CORE &&
        options->operation_scope != YVEX_ATTENTION_OPERATION_ENVELOPE)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            options->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
            YVEX_ATTENTION_OPERATION_ENVELOPE, options->operation_scope, err,
            YVEX_ERR_FORMAT, "attention operation scope is invalid");
    if (options->input_stride <
        (options->operation_scope == YVEX_ATTENTION_OPERATION_ENVELOPE
             ? (*layer)->residual_expanded_width : (*layer)->hidden_dimension))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            options->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
            (*layer)->hidden_dimension, options->input_stride, err,
            YVEX_ERR_BOUNDS, "attention input stride is shorter than its scope");
    rc = yvex_attention_class_geometry_validate(
        *layer, csa_ratio, hca_ratio, failure, err);
    return rc == YVEX_OK
               ? yvex_attention_cancel_check(options->cancellation,
                                              options->layer_index, cancel_stage,
                                              failure, err)
               : rc;
}
static void graph_value_clear(yvex_graph_value_info *value);
static void graph_op_clear(yvex_graph_op_info *op);
static void graph_missing_clear(yvex_graph_missing_required *missing);
static int graph_add_value(yvex_graph *graph, yvex_value_kind kind, const char *name,
                           unsigned int rank, const unsigned long long *dims, yvex_dtype dtype,
                           yvex_residency residency, const char *source_tensor_name,
                           unsigned int *out_id, yvex_error *err);
static int graph_add_op(yvex_graph *graph, yvex_op_kind kind, yvex_op_status status,
                        const char *name, const unsigned int *inputs, unsigned int input_count,
                        const unsigned int *outputs, unsigned int output_count, const char *reason,
                        yvex_error *err);
static int graph_add_missing(yvex_graph *graph, yvex_tensor_role role, const char *reason,
                             yvex_error *err);
typedef struct {
    const yvex_attention_layer_plan *layer;
    const yvex_attention_history_view *history;
    const float *query;
    const float *current_kv;
    const float *current_compressed;
    const float *current_indexer;
    const float *index_query;
    const float *index_weights;
    const float *sinks;
    const unsigned long long *current_compressed_positions;
    const unsigned long long *current_indexer_positions;
    float *out;
    unsigned long long current_kv_stride;
    unsigned long long current_compressed_count;
    unsigned long long current_compressed_stride;
    unsigned long long current_indexer_count;
    unsigned long long current_indexer_stride;
    unsigned long long index_query_stride;
    unsigned long long index_weight_stride;
    unsigned long long token_count;
    unsigned long long token_position;
    unsigned long long compressed_total;
    unsigned long long *trace_topk_counts;
    unsigned long long *trace_topk_positions;
    unsigned long long trace_topk_stride;
    yvex_attention_scratch_budget *scratch;
    yvex_attention_cpu_result *result;
    yvex_attention_failure *failure;
    yvex_error *err;
    double scale;
    int candidate_block_visible;
} attention_reduce_context;

static int reduce_visit_row(const attention_reduce_context *context, const float *query,
                            const float *row, double maximum, double *score_or_denominator,
                            float *destination, int accumulate) {
    unsigned long long lane;
    double score = 0.0;
    if (!context || !query || !row || !score_or_denominator)
        return 0;
    for (lane = 0ull; lane < context->layer->head_dimension; ++lane) {
        if (!isfinite(query[lane]) || !isfinite(row[lane]))
            return 0;
        score += (double)query[lane] * (double)row[lane];
    }
    score *= context->scale;
    if (!isfinite(score))
        return 0;
    if (!accumulate) {
        if (score > *score_or_denominator)
            *score_or_denominator = score;
        return 1;
    }
    score = exp(score - maximum);
    if (!isfinite(score) || !destination)
        return 0;
    *score_or_denominator += score;
    for (lane = 0ull; lane < context->layer->head_dimension; ++lane)
        destination[lane] += (float)(score * row[lane]);
    return 1;
}

static int reduce_local_rows(const attention_reduce_context *context, unsigned long long token,
                             const float *query, double maximum, double *state, float *destination,
                             int accumulate) {
    unsigned long long absolute = context->token_position + token;
    unsigned long long first = absolute + 1ull > context->layer->sliding_window
                                   ? absolute + 1ull - context->layer->sliding_window
                                   : 0ull;
    unsigned long long candidate;
    if (context->candidate_block_visible) {
        for (candidate = 0ull; candidate < context->history->local_tail_count; ++candidate) {
            const float *row = context->history->local_kv +
                               candidate * context->history->local_kv_stride;
            if (!reduce_visit_row(context, query, row, maximum, state, destination,
                                  accumulate))
                return 0;
        }
        for (candidate = 0ull; candidate < context->token_count; ++candidate) {
            const float *row = context->current_kv +
                               candidate * context->current_kv_stride;
            if (!reduce_visit_row(context, query, row, maximum, state, destination,
                                  accumulate))
                return 0;
        }
        return 1;
    }
    for (candidate = 0ull; candidate < context->history->local_tail_count; ++candidate) {
        unsigned long long position = context->history->local_positions[candidate];
        const float *row;
        if (position < first || position > absolute)
            continue;
        row = context->history->local_kv + candidate * context->history->local_kv_stride;
        if (!reduce_visit_row(context, query, row, maximum, state, destination, accumulate))
            return 0;
    }
    for (candidate = 0ull; candidate <= token; ++candidate) {
        unsigned long long position = context->token_position + candidate;
        const float *row = context->current_kv + candidate * context->current_kv_stride;
        if (position >= first &&
            !reduce_visit_row(context, query, row, maximum, state, destination, accumulate))
            return 0;
    }
    return 1;
}

static int reduce_hca_candidate_valid(const attention_reduce_context *context,
                                      unsigned long long candidate, unsigned long long absolute) {
    unsigned long long position = attention_segment_position(
        context->history->compressed_positions, context->history->compressed_entry_count,
        context->current_compressed_positions, context->current_compressed_count, candidate);
    return position != ULLONG_MAX && position <= absolute &&
           position <= ULLONG_MAX - context->layer->compression_ratio + 1ull &&
           position + context->layer->compression_ratio - 1ull <= absolute;
}

static int reduce_compressed_rows(const attention_reduce_context *context,
                                  unsigned long long absolute, const unsigned long long *selected,
                                  unsigned long long selected_count, const float *query,
                                  double maximum, double *state, float *destination,
                                  int accumulate) {
    unsigned long long count = context->layer->attention_class == YVEX_ATTENTION_CLASS_HCA
                                   ? context->compressed_total
                                   : selected_count;
    unsigned long long candidate;
    for (candidate = 0ull; candidate < count; ++candidate) {
        unsigned long long index = context->layer->attention_class == YVEX_ATTENTION_CLASS_HCA
                                       ? candidate
                                       : selected[candidate];
        const float *row;
        if (context->layer->attention_class == YVEX_ATTENTION_CLASS_HCA &&
            !reduce_hca_candidate_valid(context, index, absolute))
            continue;
        row = attention_segment_row(
            context->history->compressed_kv, context->history->compressed_entry_count,
            context->history->compressed_kv_stride, context->current_compressed,
            context->current_compressed_count, context->current_compressed_stride, index);
        if (!reduce_visit_row(context, query, row, maximum, state, destination, accumulate))
            return 0;
    }
    return 1;
}

static int reduce_head(const attention_reduce_context *context, unsigned long long token,
                       unsigned long long head, const unsigned long long *selected,
                       unsigned long long selected_count) {
    unsigned long long absolute = context->token_position + token;
    const float *query = context->query +
                         token * context->layer->query_heads * context->layer->head_dimension +
                         head * context->layer->head_dimension;
    float *destination = context->out +
                         token * context->layer->query_heads * context->layer->head_dimension +
                         head * context->layer->head_dimension;
    double maximum = (double)context->sinks[head];
    double denominator;
    unsigned long long lane;
    if (!isfinite(maximum) ||
        !reduce_local_rows(context, token, query, maximum, &maximum, NULL, 0) ||
        !reduce_compressed_rows(context, absolute, selected, selected_count, query, maximum,
                                &maximum, NULL, 0))
        return 0;
    denominator = exp((double)context->sinks[head] - maximum);
    if (!isfinite(denominator))
        return 0;
    memset(destination, 0, (size_t)context->layer->head_dimension * sizeof(*destination));
    if (!reduce_local_rows(context, token, query, maximum, &denominator, destination, 1) ||
        !reduce_compressed_rows(context, absolute, selected, selected_count, query, maximum,
                                &denominator, destination, 1) ||
        !isfinite(denominator) || denominator <= 0.0)
        return 0;
    for (lane = 0ull; lane < context->layer->head_dimension; ++lane)
        destination[lane] = (float)((double)destination[lane] / denominator);
    if (!yvex_attention_compute_round(context->layer->compute_contract, destination,
                                      context->layer->head_dimension) ||
        !yvex_attention_rope_apply(destination, context->layer->head_dimension,
                                   context->layer->rope_head_dimension, absolute,
                                   &context->layer->position, 1))
        return 0;
    return yvex_attention_compute_round(context->layer->compute_contract, destination,
                                        context->layer->head_dimension);
}

static int reduce_select(const attention_reduce_context *context, unsigned long long token,
                         unsigned long long *selected, unsigned long long *selected_count) {
    unsigned long long absolute = context->token_position + token;
    unsigned long long valid_count = 0ull;
    unsigned long long candidate;
    int rc = YVEX_OK;
    *selected_count = 0ull;
    if (context->layer->attention_class == YVEX_ATTENTION_CLASS_CSA && context->compressed_total) {
        rc = yvex_attention_csa_select(
            context->layer, context->history, context->current_indexer,
            context->current_indexer_count, context->current_indexer_stride,
            context->current_indexer_positions,
            context->index_query + token * context->index_query_stride,
            context->index_weights + token * context->index_weight_stride, absolute, selected,
            selected_count, &valid_count, context->scratch, context->failure, context->err);
        if (rc != YVEX_OK)
            return rc;
    } else if (context->layer->attention_class == YVEX_ATTENTION_CLASS_HCA) {
        for (candidate = 0ull; candidate < context->compressed_total; ++candidate)
            if (reduce_hca_candidate_valid(context, candidate, absolute))
                valid_count++;
        *selected_count = valid_count;
    }
    if (context->result) {
        if (valid_count > context->result->topk_candidates)
            context->result->topk_candidates = valid_count;
        if (*selected_count > context->result->topk_selected)
            context->result->topk_selected = *selected_count;
    }
    if (context->layer->attention_class != YVEX_ATTENTION_CLASS_CSA || !context->trace_topk_counts)
        return YVEX_OK;
    if (*selected_count > context->trace_topk_stride ||
        (*selected_count && (!selected || !context->trace_topk_positions)))
        return yvex_attention_reject(context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_STATE_DELTA,
                                     NULL, context->layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
                                     context->trace_topk_stride, *selected_count, context->err,
                                     YVEX_ERR_BOUNDS,
                                     "CSA trace top-k extent is smaller than selection");
    context->trace_topk_counts[token] = *selected_count;
    for (candidate = 0ull; candidate < *selected_count; ++candidate)
        context->trace_topk_positions[token * context->trace_topk_stride + candidate] =
            attention_segment_position(context->history->compressed_positions,
                                            context->history->compressed_entry_count,
                                            context->current_compressed_positions,
                                            context->current_compressed_count, selected[candidate]);
    return YVEX_OK;
}

static int reduce_context_validate(attention_reduce_context *context) {
    unsigned long long query_width;
    unsigned long long last_position;

    if (!context->layer || !context->query || !context->history || !context->current_kv ||
        !context->sinks || !context->out || !context->token_count)
        return yvex_attention_reject(
            context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, NULL,
            context->layer ? context->layer->layer_index : YVEX_ATTENTION_NO_LAYER,
            YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, context->err, YVEX_ERR_INVALID_ARG,
            "sparse attention requires plan, history, query, KV, and output");
    if (!yvex_core_u64_mul(context->layer->query_heads, context->layer->head_dimension,
                           &query_width) ||
        context->current_kv_stride < context->layer->head_dimension ||
        !yvex_core_u64_add(context->token_position, context->token_count - 1ull, &last_position))
        return yvex_attention_reject(
            context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
            context->layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX,
            context->token_position, context->err, YVEX_ERR_BOUNDS,
            "attention query width, KV stride, or position extent overflowed");
    if (!yvex_core_u64_add(context->history->compressed_entry_count,
                           context->current_compressed_count, &context->compressed_total))
        return yvex_attention_reject(context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     NULL, context->layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
                                     ULLONG_MAX, context->history->compressed_entry_count,
                                     context->err, YVEX_ERR_BOUNDS,
                                     "sparse attention compressed candidate count overflowed");
    if (context->current_compressed_count &&
        (!context->current_compressed || !context->current_compressed_positions ||
         context->current_compressed_stride < context->layer->head_dimension))
        return yvex_attention_reject(
            context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, NULL,
            context->layer->layer_index, YVEX_TENSOR_ROLE_ATTENTION_COMPRESSOR_KV,
            context->layer->head_dimension, context->current_compressed_stride, context->err,
            YVEX_ERR_FORMAT, "current compressed attention entries are incomplete");
    if (context->layer->attention_class == YVEX_ATTENTION_CLASS_CSA &&
        (context->current_indexer_count != context->current_compressed_count ||
         (context->current_indexer_count &&
          (!context->current_indexer || !context->current_indexer_positions ||
           !context->index_query || !context->index_weights ||
           context->current_indexer_stride < context->layer->indexer_head_dimension))))
        return yvex_attention_reject(
            context->failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_HISTORY, NULL,
            context->layer->layer_index, YVEX_TENSOR_ROLE_INDEXER_COMPRESSOR_KV,
            context->current_compressed_count, context->current_indexer_count, context->err,
            YVEX_ERR_FORMAT, "CSA current compressed and indexer entries are not paired");
    context->scale = 1.0 / sqrt((double)context->layer->head_dimension);
    return YVEX_OK;
}

int yvex_attention_reduce_chunk(
    const yvex_attention_layer_plan *layer, const float *query,
    const yvex_attention_history_view *history, const float *current_kv,
    unsigned long long current_kv_stride, const float *current_compressed,
    unsigned long long current_compressed_count, unsigned long long current_compressed_stride,
    const unsigned long long *current_compressed_positions, const float *current_indexer,
    unsigned long long current_indexer_count, unsigned long long current_indexer_stride,
    const unsigned long long *current_indexer_positions, const float *index_query,
    unsigned long long index_query_stride, const float *index_weights,
    unsigned long long index_weight_stride, const float *sinks, unsigned long long token_count,
    unsigned long long token_position, float *out, int candidate_block_visible,
    unsigned long long *trace_topk_counts,
    unsigned long long *trace_topk_positions, unsigned long long trace_topk_stride,
    yvex_attention_scratch_budget *scratch, yvex_attention_cpu_result *result,
    yvex_attention_failure *failure, yvex_error *err) {
    attention_reduce_context context = {layer,
                                        history,
                                        query,
                                        current_kv,
                                        current_compressed,
                                        current_indexer,
                                        index_query,
                                        index_weights,
                                        sinks,
                                        current_compressed_positions,
                                        current_indexer_positions,
                                        out,
                                        current_kv_stride,
                                        current_compressed_count,
                                        current_compressed_stride,
                                        current_indexer_count,
                                        current_indexer_stride,
                                        index_query_stride,
                                        index_weight_stride,
                                        token_count,
                                        token_position,
                                        0ull,
                                        trace_topk_counts,
                                        trace_topk_positions,
                                        trace_topk_stride,
                                        scratch,
                                        result,
                                        failure,
                                        err,
                                        0.0,
                                        candidate_block_visible};
    unsigned long long *selected = NULL;
    size_t selected_reserved = 0u;
    unsigned long long token;
    int rc = reduce_context_validate(&context);
    if (rc != YVEX_OK)
        return rc;
    if (layer->attention_class == YVEX_ATTENTION_CLASS_CSA && context.compressed_total) {
        unsigned long long selected_capacity =
            attention_min_u64(context.compressed_total, layer->sparse_topk.k);
        if (!yvex_attention_scratch_reserve(scratch, selected_capacity, sizeof(*selected),
                                            &selected_reserved))
            return yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL, layer->layer_index,
                YVEX_TENSOR_ROLE_UNKNOWN, scratch ? scratch->limit_bytes : 0ull,
                scratch ? (unsigned long long)scratch->live_bytes : 0ull, err, YVEX_ERR_BOUNDS,
                "sparse selection exceeds the attention scratch budget");
        selected = (unsigned long long *)yvex_attention_scratch_calloc(
            scratch, selected_capacity, sizeof(*selected));
        if (!selected) {
            attention_scratch_release(scratch, selected_reserved);
            return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
                                         layer->layer_index, YVEX_TENSOR_ROLE_UNKNOWN,
                                         context.compressed_total, 0ull, err, YVEX_ERR_NOMEM,
                                         "sparse attention selection allocation failed");
        }
    }
    for (token = 0ull; token < token_count; ++token) {
        unsigned long long selected_count = 0ull;
        unsigned long long head;
        rc = reduce_select(&context, token, selected, &selected_count);
        if (rc != YVEX_OK)
            break;
        for (head = 0ull; head < layer->query_heads; ++head) {
            if (!reduce_head(&context, token, head, selected, selected_count)) {
                rc = yvex_attention_reject(
                    failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL, layer->layer_index,
                    YVEX_TENSOR_ROLE_UNKNOWN, 1ull, token, err, YVEX_ERR_FORMAT,
                    "sparse attention score, softmax, or reduction became non-finite");
                break;
            }
        }
        if (rc != YVEX_OK)
            break;
    }
    yvex_attention_scratch_free(scratch, selected);
    attention_scratch_release(scratch, selected_reserved);
    return rc;
}

int yvex_attention_output_project(
    yvex_materialization_session *session, const yvex_runtime_tensor_binding *out_a,
    const yvex_runtime_tensor_binding *out_b, const float *attention_values,
    unsigned long long token_count, unsigned long long attention_stride,
    unsigned long long group_count, unsigned long long group_input_width, unsigned long long rank,
    unsigned long long hidden_width, yvex_attention_compute_contract compute_contract, float *out,
    unsigned long long output_stride, yvex_attention_scratch_budget *scratch,
    yvex_attention_cpu_result *result, yvex_attention_failure *failure, yvex_error *err) {
    float *low = NULL;
    unsigned long long attention_width;
    unsigned long long low_stride;
    unsigned long long low_elements;
    unsigned long long group;
    unsigned long long rows;
    size_t low_bytes = 0u;
    int rc;
    if (!session || !out_a || !out_b || !attention_values || !out || !token_count || !group_count ||
        !group_input_width || !rank || !hidden_width || output_stride < hidden_width ||
        compute_contract != YVEX_ATTENTION_COMPUTE_BF16_F32_RNE_V1)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, out_a,
            YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_ATTENTION_OUT_A, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG,
            "attention batch output projection requires bindings and buffers");
    if (!yvex_core_u64_mul(group_count, group_input_width, &attention_width) ||
        attention_stride < attention_width || !yvex_core_u64_mul(group_count, rank, &low_stride) ||
        !yvex_core_u64_mul(token_count, low_stride, &low_elements))
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, out_a,
                                     out_a->layer_index, YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
                                     ULLONG_MAX, attention_stride, err, YVEX_ERR_BOUNDS,
                                     "attention output projection geometry overflowed");
    if (!yvex_attention_scratch_reserve(scratch, low_elements, sizeof(float), &low_bytes))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, out_a, out_a->layer_index,
            YVEX_TENSOR_ROLE_ATTENTION_OUT_A, scratch ? scratch->limit_bytes : 0ull,
            scratch ? (unsigned long long)scratch->live_bytes : 0ull, err, YVEX_ERR_BOUNDS,
            "attention output projection scratch budget exceeded");
    low = (float *)yvex_attention_scratch_calloc(scratch, low_elements, sizeof(float));
    if (!low)
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, out_a,
                                   out_a->layer_index, YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
                                   low_elements, 0ull, err, YVEX_ERR_NOMEM,
                                   "failed to allocate attention batch output scratch");
    if (!low)
        goto cleanup;
    for (group = 0ull; group < group_count; ++group) {
        rc = yvex_attention_dot_batch(session, out_a, group * rank,
                                      attention_values + group * group_input_width, token_count,
                                      attention_stride, group_input_width, rank, low + group * rank,
                                      low_stride, &rows, scratch, result, failure, err);
        if (rc != YVEX_OK)
            goto cleanup;
        if (rows != rank) {
            rc = yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, out_a, out_a->layer_index,
                YVEX_TENSOR_ROLE_ATTENTION_OUT_A, rank, rows, err, YVEX_ERR_FORMAT,
                "attention batch output A projection did not produce full group rank");
            goto cleanup;
        }
    }
    if (!yvex_attention_compute_round(compute_contract, low, low_elements)) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, out_a,
                                   out_a->layer_index, YVEX_TENSOR_ROLE_ATTENTION_OUT_A,
                                   low_elements, 0ull, err, YVEX_ERR_FORMAT,
                                   "attention output A projection could not publish BF16 values");
        goto cleanup;
    }
    rc = yvex_attention_dot_batch(session, out_b, 0ull, low, token_count, low_stride, low_stride,
                                  hidden_width, out, output_stride, &rows, scratch, result, failure,
                                  err);
    if (rc == YVEX_OK && rows != hidden_width)
        rc = yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, out_b, out_b->layer_index,
            YVEX_TENSOR_ROLE_ATTENTION_OUT_B, hidden_width, rows, err, YVEX_ERR_FORMAT,
            "attention batch output B projection did not produce full hidden width");
    if (rc == YVEX_OK) {
        unsigned long long token;
        for (token = 0ull; token < token_count; ++token)
            if (!yvex_attention_compute_round(compute_contract, out + token * output_stride,
                                              hidden_width)) {
                rc = yvex_attention_reject(
                    failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, out_b, out_b->layer_index,
                    YVEX_TENSOR_ROLE_ATTENTION_OUT_B, hidden_width, token, err, YVEX_ERR_FORMAT,
                    "attention output B projection could not publish BF16 values");
                break;
            }
    }
cleanup:
    yvex_attention_scratch_free(scratch, low);
    attention_scratch_release(scratch, low_bytes);
    return rc;
}

int yvex_attention_decode_flat(yvex_materialization_session *session,
                               const yvex_runtime_tensor_binding *runtime_binding, float *out,
                               unsigned long long expected_elements,
                               yvex_attention_scratch_budget *scratch,
                               yvex_attention_cpu_result *result, yvex_attention_failure *failure,
                               yvex_error *err) {
    const yvex_materialized_tensor_binding *binding;
    unsigned long long row;
    unsigned long long total;
    int rc;
    if (!session || !runtime_binding || !runtime_binding->binding || !out)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, runtime_binding,
            YVEX_ATTENTION_NO_LAYER,
            runtime_binding ? runtime_binding->role : YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG, "attention flat decode requires session, binding, and output");
    binding = runtime_binding->binding;
    if (!yvex_core_u64_mul(binding->row_width, binding->row_count, &total) ||
        total != expected_elements)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role,
                                     expected_elements, total, err, YVEX_ERR_BOUNDS,
                                     "attention flat decode expected element count mismatch");
    for (row = 0ull; row < binding->row_count; ++row) {
        rc =
            yvex_attention_decode_row(session, runtime_binding, row, out + row * binding->row_width,
                                      binding->row_width, scratch, result, failure, err);
        if (rc != YVEX_OK)
            return rc;
    }
    return YVEX_OK;
}

int yvex_attention_activation_apply(const yvex_attention_activation_policy *policy, float *values,
                                    unsigned long long count, unsigned long long layer_index,
                                    yvex_tensor_role role, yvex_attention_scratch_budget *budget,
                                    yvex_attention_failure *failure, yvex_error *err) {
    float *scratch = NULL;
    float *block_out = NULL;
    unsigned char *codes = NULL;
    unsigned long long block_width;
    unsigned long long offset;
    size_t scratch_bytes = 0u;
    size_t block_out_bytes = 0u;
    size_t codes_bytes = 0u;
    int rc = YVEX_OK;
    if (!policy || !policy->required)
        return YVEX_OK;
    if (!values || count == 0ull || policy->block_width == 0ull)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                     NULL, layer_index, role, 1ull, 0ull, err, YVEX_ERR_INVALID_ARG,
                                     "runtime activation policy requires values and block width");
    if (policy->block_axis != YVEX_ATTENTION_AXIS_FINAL_DIMENSION)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                                     layer_index, role, YVEX_ATTENTION_AXIS_FINAL_DIMENSION,
                                     (unsigned long long)policy->block_axis, err, YVEX_ERR_FORMAT,
                                     "runtime activation policy axis is unsupported");
    if (policy->pre_transform == YVEX_ATTENTION_TRANSFORM_DAO_FHT_V1_1_0_POST2) {
        float scale = 1.0f / sqrtf((float)count);
        if (!yvex_attention_scratch_reserve(budget, count, sizeof(*scratch), &scratch_bytes))
            return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
                                         layer_index, role, budget ? budget->limit_bytes : 0ull,
                                         budget ? (unsigned long long)budget->live_bytes : 0ull,
                                         err, YVEX_ERR_BOUNDS,
                                         "runtime activation transform scratch budget exceeded");
        scratch = (float *)yvex_attention_scratch_calloc(budget, count, sizeof(*scratch));
        if (!scratch) {
            attention_scratch_release(budget, scratch_bytes);
            return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
                                         layer_index, role, count, 0ull, err, YVEX_ERR_NOMEM,
                                         "runtime activation Hadamard scratch allocation failed");
        }
        rc = yvex_attention_hadamard_cpu(values, count, scale, 1, scratch, budget, failure, err);
        if (rc != YVEX_OK)
            goto cleanup;
        memcpy(values, scratch, (size_t)count * sizeof(*values));
    } else if (policy->pre_transform != YVEX_ATTENTION_TRANSFORM_NONE) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL,
                                   layer_index, role, 0ull,
                                   (unsigned long long)policy->pre_transform, err, YVEX_ERR_FORMAT,
                                   "runtime activation transform is unsupported");
        goto cleanup;
    }
    block_width = policy->block_width;
    if (count % block_width != 0ull) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
                                   layer_index, role, block_width, count, err, YVEX_ERR_BOUNDS,
                                   "runtime activation policy requires exact block divisibility");
        goto cleanup;
    }
    if (!yvex_attention_scratch_reserve(budget, block_width, sizeof(*block_out),
                                        &block_out_bytes)) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
                                   layer_index, role, budget ? budget->limit_bytes : 0ull,
                                   budget ? (unsigned long long)budget->live_bytes : 0ull, err,
                                   YVEX_ERR_BOUNDS,
                                   "runtime activation dequantized block scratch budget exceeded");
        goto cleanup;
    }
    block_out = (float *)yvex_attention_scratch_calloc(
        budget, block_width, sizeof(*block_out));
    if (!block_out) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
                                   layer_index, role, block_width, 0ull, err, YVEX_ERR_NOMEM,
                                   "runtime activation dequantized block allocation failed");
        goto cleanup;
    }
    if (!yvex_attention_scratch_reserve(budget, block_width, sizeof(*codes), &codes_bytes)) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, NULL,
                                   layer_index, role, budget ? budget->limit_bytes : 0ull,
                                   budget ? (unsigned long long)budget->live_bytes : 0ull, err,
                                   YVEX_ERR_BOUNDS,
                                   "runtime activation code block scratch budget exceeded");
        goto cleanup;
    }
    codes = (unsigned char *)yvex_attention_scratch_calloc(
        budget, block_width, sizeof(*codes));
    if (!codes) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION, NULL,
                                   layer_index, role, block_width, 0ull, err, YVEX_ERR_NOMEM,
                                   "runtime activation code block allocation failed");
        goto cleanup;
    }
    for (offset = 0ull; offset < count; offset += block_width) {
        unsigned char scale_code = 0u;
        if (policy->quantization == YVEX_ATTENTION_QUANT_FP8_E4M3_UE8M0_FAKE_DEQUANT) {
            rc = yvex_attention_fp8_fake_quant_block(values + offset, block_width, block_out, codes,
                                                     &scale_code, failure, err);
        } else if (policy->quantization == YVEX_ATTENTION_QUANT_FP4_E2M1_UE8M0_FAKE_DEQUANT) {
            rc = yvex_attention_fp4_fake_quant_block(values + offset, block_width, block_out, codes,
                                                     &scale_code, failure, err);
        } else {
            rc = yvex_attention_reject(
                failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, NULL, layer_index, role, 0ull,
                (unsigned long long)policy->quantization, err, YVEX_ERR_FORMAT,
                "runtime activation quantization is unsupported");
        }
        if (rc != YVEX_OK)
            goto cleanup;
        memcpy(values + offset, block_out, (size_t)block_width * sizeof(*values));
    }
    yvex_error_clear(err);
cleanup:
    yvex_attention_scratch_free(budget, scratch);
    yvex_attention_scratch_free(budget, block_out);
    yvex_attention_scratch_free(budget, codes);
    attention_scratch_release(budget, codes_bytes);
    attention_scratch_release(budget, block_out_bytes);
    attention_scratch_release(budget, scratch_bytes);
    return rc;
}

int yvex_attention_scratch_reserve(yvex_attention_scratch_budget *budget, unsigned long long count,
                                   size_t element_size, size_t *bytes_out) {
    size_t bytes;
    size_t next;
    if (!budget || !bytes_out || !yvex_attention_checked_size(count, element_size, &bytes) ||
        budget->live_bytes > SIZE_MAX - bytes)
        return 0;
    next = budget->live_bytes + bytes;
    if (budget->limit_bytes && (unsigned long long)next > budget->limit_bytes)
        return 0;
    budget->live_bytes = next;
    if (next > budget->peak_bytes)
        budget->peak_bytes = next;
    *bytes_out = bytes;
    return 1;
}

const yvex_runtime_tensor_binding *
yvex_attention_binding_find(const yvex_runtime_descriptor *descriptor, yvex_tensor_role role,
                            const yvex_attention_layer_plan *layer) {
    if (!layer) return NULL;
    return yvex_runtime_descriptor_find_role(descriptor, role, layer->tensor_scope,
                                             layer->layer_index, layer->predictor_index);
}

static int attention_row_geometry(const yvex_materialized_tensor_binding *binding,
                                unsigned long long *row_bytes_out,
                                unsigned long long *row_count_out, yvex_attention_failure *failure,
                                yvex_error *err) {
    unsigned long long blocks;
    unsigned long long row_bytes;
    unsigned long long encoded_bytes = 0ull;
    if (!binding || !row_bytes_out || !row_count_out)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT,
                                     NULL, YVEX_ATTENTION_NO_LAYER, YVEX_TENSOR_ROLE_UNKNOWN, 1ull,
                                     0ull, err, YVEX_ERR_INVALID_ARG,
                                     "attention row-byte calculation requires binding and outputs");
    if (!binding->row_width || !binding->row_count || !binding->block_size ||
        !binding->bytes_per_block || binding->row_width % binding->block_size != 0ull)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
                                     binding->layer_index, binding->role, binding->block_size,
                                     binding->row_width, err, YVEX_ERR_FORMAT,
                                     "attention binding row geometry is invalid");
    blocks = binding->row_width / binding->block_size;
    if (!yvex_core_u64_mul(blocks, binding->bytes_per_block, &row_bytes) || row_bytes == 0ull ||
        row_bytes > (unsigned long long)SIZE_MAX)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
                                     binding->layer_index, binding->role, ULLONG_MAX, row_bytes,
                                     err, YVEX_ERR_BOUNDS,
                                     "attention binding row byte size overflowed");
    if (!yvex_core_u64_mul(row_bytes, binding->row_count, &encoded_bytes))
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL,
                                     binding->layer_index, binding->role, ULLONG_MAX,
                                     binding->encoded_bytes, err, YVEX_ERR_BOUNDS,
                                     "attention binding encoded byte geometry overflowed");
    if (binding->encoded_bytes != encoded_bytes)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, NULL, binding->layer_index,
            binding->role, encoded_bytes, binding->encoded_bytes, err, YVEX_ERR_BOUNDS,
            "attention binding encoded bytes do not exactly match declared rows");
    *row_bytes_out = row_bytes;
    *row_count_out = binding->row_count;
    return YVEX_OK;
}

static int attention_payload_account(yvex_attention_cpu_result *result, unsigned long long bytes,
                                   const yvex_runtime_tensor_binding *binding,
                                   yvex_attention_failure *failure, yvex_error *err) {
    unsigned long long total;
    if (!result)
        return YVEX_OK;
    if (!yvex_core_u64_add(result->payload_bytes_read, bytes, &total))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, binding,
            binding && binding->binding ? binding->binding->layer_index : YVEX_ATTENTION_NO_LAYER,
            binding ? binding->role : YVEX_TENSOR_ROLE_UNKNOWN, ULLONG_MAX, bytes, err,
            YVEX_ERR_BOUNDS, "attention payload byte accounting overflowed");
    result->payload_bytes_read = total;
    return YVEX_OK;
}

int yvex_attention_decode_row(yvex_materialization_session *session,
                              const yvex_runtime_tensor_binding *runtime_binding,
                              unsigned long long row_index, float *out,
                              unsigned long long out_elements,
                              yvex_attention_scratch_budget *scratch,
                              yvex_attention_cpu_result *result, yvex_attention_failure *failure,
                              yvex_error *err) {
    const yvex_materialized_tensor_binding *binding;
    unsigned long long row_bytes;
    unsigned long long row_count;
    unsigned long long block_count;
    unsigned long long block;
    unsigned char *encoded = NULL;
    size_t encoded_scratch = 0u;
    int rc;
    if (!session || !runtime_binding || !runtime_binding->binding || !out)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, runtime_binding,
            YVEX_ATTENTION_NO_LAYER,
            runtime_binding ? runtime_binding->role : YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG, "attention row decode requires session, binding, and output");
    binding = runtime_binding->binding;
    rc = attention_row_geometry(binding, &row_bytes, &row_count, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (row_index >= row_count || out_elements != binding->row_width)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role,
                                     binding->row_width, out_elements, err, YVEX_ERR_BOUNDS,
                                     "attention row decode output shape does not match tensor row");
    if (!yvex_attention_scratch_reserve(scratch, row_bytes, 1u, &encoded_scratch))
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH, runtime_binding, binding->layer_index,
            binding->role, scratch ? scratch->limit_bytes : 0ull,
            scratch ? (unsigned long long)scratch->live_bytes : 0ull, err, YVEX_ERR_BOUNDS,
            "attention encoded row scratch budget exceeded");
    encoded = (unsigned char *)yvex_attention_scratch_calloc(scratch, row_bytes, 1ull);
    if (!encoded)
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION,
                                   runtime_binding, binding->layer_index, binding->role, row_bytes,
                                   0ull, err, YVEX_ERR_NOMEM,
                                   "failed to allocate attention encoded row scratch");
    if (!encoded) {
        attention_scratch_release(scratch, encoded_scratch);
        return rc;
    }
    rc = yvex_materialization_session_read(session, binding, row_index * row_bytes, encoded,
                                           (size_t)row_bytes, NULL, err);
    if (rc != YVEX_OK) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_READ, runtime_binding,
                                   binding->layer_index, binding->role, row_bytes, 0ull, err, rc,
                                   "failed to read attention encoded row");
        goto cleanup;
    }
    rc = attention_payload_account(result, row_bytes, runtime_binding, failure, err);
    if (rc != YVEX_OK)
        goto cleanup;
    block_count = binding->row_width / binding->block_size;
    for (block = 0ull; block < block_count; ++block) {
        yvex_quant_failure qfailure;
        rc = yvex_quant_decode_block(
            binding->qtype, encoded + (size_t)(block * binding->bytes_per_block),
            (size_t)binding->bytes_per_block, out + (block * binding->block_size),
            binding->block_size, &qfailure, err);
        if (rc != YVEX_OK) {
            rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                                       runtime_binding, binding->layer_index, binding->role,
                                       binding->block_size, qfailure.actual, err, rc,
                                       "failed to decode attention qtype row");
            goto cleanup;
        }
    }
cleanup:
    yvex_attention_scratch_free(scratch, encoded);
    attention_scratch_release(scratch, encoded_scratch);
    return rc;
}

int yvex_attention_dot_batch(yvex_materialization_session *session,
                             const yvex_runtime_tensor_binding *runtime_binding,
                             unsigned long long start_row, const float *vectors,
                             unsigned long long token_count, unsigned long long vector_stride,
                             unsigned long long vector_len, unsigned long long max_rows, float *out,
                             unsigned long long output_stride, unsigned long long *rows_out,
                             yvex_attention_scratch_budget *scratch,
                             yvex_attention_cpu_result *result, yvex_attention_failure *failure,
                             yvex_error *err) {
    const yvex_materialized_tensor_binding *binding;
    unsigned long long row_bytes;
    unsigned long long row_count;
    unsigned long long rows;
    unsigned long long row;
    unsigned char *encoded = NULL;
    size_t encoded_scratch = 0u;
    int rc;
    if (!session || !runtime_binding || !runtime_binding->binding || !vectors || !out ||
        !rows_out || token_count == 0ull || vector_stride < vector_len)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_INVALID_ARGUMENT, runtime_binding,
            YVEX_ATTENTION_NO_LAYER,
            runtime_binding ? runtime_binding->role : YVEX_TENSOR_ROLE_UNKNOWN, 1ull, 0ull, err,
            YVEX_ERR_INVALID_ARG,
            "attention batch dot rows require session, binding, vectors, and output");
    binding = runtime_binding->binding;
    rc = attention_row_geometry(binding, &row_bytes, &row_count, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (vector_len != binding->row_width)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, runtime_binding,
            binding->layer_index, binding->role, binding->row_width, vector_len, err,
            YVEX_ERR_BOUNDS, "attention batch dot vector length does not match tensor row width");
    if (start_row >= row_count)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role,
                                     row_count, start_row, err, YVEX_ERR_BOUNDS,
                                     "attention batch dot row range starts beyond tensor rows");
    rows =
        attention_min_u64(max_rows ? max_rows : row_count - start_row, row_count - start_row);
    if (rows == 0ull)
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION,
                                     runtime_binding, binding->layer_index, binding->role, 1ull,
                                     0ull, err, YVEX_ERR_BOUNDS,
                                     "attention batch dot requires at least one output row");
    if (output_stride < rows)
        return yvex_attention_reject(
            failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_DIMENSION, runtime_binding,
            binding->layer_index, binding->role, rows, output_stride, err, YVEX_ERR_BOUNDS,
            "attention batch dot output stride is smaller than produced rows");
    if (!yvex_attention_scratch_reserve(scratch, row_bytes, 1u, &encoded_scratch)) {
        attention_scratch_release(scratch, encoded_scratch);
        return yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_SCRATCH,
                                     runtime_binding, binding->layer_index, binding->role,
                                     scratch ? scratch->limit_bytes : 0ull,
                                     scratch ? (unsigned long long)scratch->live_bytes : 0ull, err,
                                     YVEX_ERR_BOUNDS, "attention dot row scratch budget exceeded");
    }
    encoded = (unsigned char *)yvex_attention_scratch_calloc(scratch, row_bytes, 1ull);
    if (!encoded) {
        rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_ALLOCATION,
                                   runtime_binding, binding->layer_index, binding->role, row_bytes,
                                   0ull, err, YVEX_ERR_NOMEM,
                                   "failed to allocate attention batch dot row scratch");
        goto cleanup;
    }
    for (row = 0ull; row < rows; ++row) {
        yvex_quant_failure qfailure;
        unsigned long long token;
        rc = yvex_materialization_session_read(session, binding, (start_row + row) * row_bytes,
                                               encoded, (size_t)row_bytes, NULL, err);
        if (rc != YVEX_OK) {
            rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_READ,
                                       runtime_binding, binding->layer_index, binding->role,
                                       row_bytes, 0ull, err, rc,
                                       "failed to read attention batch dot encoded row");
            goto cleanup;
        }
        rc = attention_payload_account(result, row_bytes, runtime_binding, failure, err);
        if (rc != YVEX_OK)
            goto cleanup;
        for (token = 0ull; token < token_count; ++token) {
            const float *vector = vectors + token * vector_stride;
            float *dst = out + token * output_stride + row;
            rc = yvex_quant_cpu_dot(binding->qtype, encoded, (size_t)row_bytes, vector, vector_len,
                                    dst, &qfailure, err);
            if (rc != YVEX_OK) {
                rc = yvex_attention_reject(failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC,
                                           runtime_binding, binding->layer_index, binding->role,
                                           vector_len, qfailure.actual, err, rc,
                                           "attention batch encoded row dot failed");
                goto cleanup;
            }
            if (!isfinite(*dst)) {
                rc = yvex_attention_reject(
                    failure, YVEX_DEEPSEEK_ATTENTION_FAILURE_NUMERIC, runtime_binding,
                    binding->layer_index, binding->role, 1ull, row, err, YVEX_ERR_FORMAT,
                    "attention batch encoded row dot produced non-finite output");
                goto cleanup;
            }
        }
    }
    *rows_out = rows;
cleanup:
    yvex_attention_scratch_free(scratch, encoded);
    attention_scratch_release(scratch, encoded_scratch);
    return rc;
}

double yvex_attention_checksum(const float *values, unsigned long long count) {
    unsigned long long i;
    double sum = 0.0;
    if (!values)
        return 0.0;
    for (i = 0ull; i < count; ++i) {
        double weight = (double)((i % 29ull) + 1ull) / 29.0;
        sum += (double)values[i] * weight;
    }
    return sum;
}

void yvex_graph_close(yvex_graph *graph) {
    unsigned long long i;
    if (!graph)
        return;
    free(graph->architecture);
    free(graph->model_name);
    for (i = 0; i < graph->value_count; ++i) {
        graph_value_clear(&graph->values[i]);
    }
    for (i = 0; i < graph->op_count; ++i) {
        graph_op_clear(&graph->ops[i]);
    }
    for (i = 0; i < graph->missing_count; ++i) {
        graph_missing_clear(&graph->missing[i]);
    }
    free(graph->values);
    free(graph->ops);
    free(graph->edges);
    free(graph->missing);
    free(graph);
}

yvex_graph_status yvex_graph_status_of(const yvex_graph *graph) {
    return graph ? graph->status : YVEX_GRAPH_STATUS_EMPTY;
}

unsigned long long yvex_graph_value_count(const yvex_graph *graph) {
    return graph ? graph->value_count : 0;
}

unsigned long long yvex_graph_op_count(const yvex_graph *graph) {
    return graph ? graph->op_count : 0;
}

unsigned long long yvex_graph_missing_required_count(const yvex_graph *graph) {
    return graph ? graph->missing_count : 0;
}

const yvex_graph_value_info *yvex_graph_value_at(const yvex_graph *graph,
                                                 unsigned long long index) {
    if (!graph || index >= graph->value_count)
        return NULL;
    return &graph->values[index];
}

const yvex_graph_op_info *yvex_graph_op_at(const yvex_graph *graph, unsigned long long index) {
    if (!graph || index >= graph->op_count)
        return NULL;
    return &graph->ops[index];
}

const yvex_graph_missing_required *yvex_graph_missing_required_at(const yvex_graph *graph,
                                                                  unsigned long long index) {
    if (!graph || index >= graph->missing_count)
        return NULL;
    return &graph->missing[index];
}

static const yvex_tensor_info *find_role(const yvex_tensor_table *table, yvex_tensor_role role) {
    unsigned long long i;
    for (i = 0; i < yvex_tensor_table_count(table); ++i) {
        const yvex_tensor_info *tensor = yvex_tensor_table_at(table, i);
        if (tensor && tensor->role == role) {
            return tensor;
        }
    }
    return NULL;
}

static int add_required_diagnostics(yvex_graph *graph, const yvex_tensor_info *token_embedding,
                                    const yvex_tensor_info *output_norm,
                                    const yvex_tensor_info *output_head, yvex_error *err) {
    int rc;
    if (!token_embedding) {
        rc = graph_add_missing(graph, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING,
                               "required for token embedding", err);
        if (rc != YVEX_OK) {
            return rc;
        }
    }
    if (!output_norm) {
        rc = graph_add_missing(graph, YVEX_TENSOR_ROLE_OUTPUT_NORM,
                               "required for final normalization", err);
        if (rc != YVEX_OK) {
            return rc;
        }
    }
    if (!output_head) {
        rc = graph_add_missing(graph, YVEX_TENSOR_ROLE_OUTPUT_HEAD, "required for logits", err);
        if (rc != YVEX_OK) {
            return rc;
        }
    }
    return YVEX_OK;
}

static int add_embedding_path(yvex_graph *graph, const yvex_tensor_info *token_embedding,
                              unsigned long long sequence_length, yvex_error *err) {
    unsigned long long token_shape[1];
    unsigned long long hidden_shape[2];
    unsigned int token_ids_value;
    unsigned int weight_value;
    unsigned int hidden_value;
    unsigned int inputs[2];
    unsigned int outputs[1];
    int rc;
    if (!token_embedding) {
        return YVEX_OK;
    }
    if (token_embedding->rank < 2 || token_embedding->dims[0] == 0) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_graph_build_for_model",
                       "token embedding tensor requires rank >= 2 and non-zero hidden dimension");
        return YVEX_ERR_FORMAT;
    }
    token_shape[0] = sequence_length;
    rc = graph_add_value(graph, YVEX_VALUE_TOKEN_IDS, "token_ids", 1, token_shape, YVEX_DTYPE_I32,
                         YVEX_RESIDENCY_HOST, NULL, &token_ids_value, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    rc = graph_add_value(graph, YVEX_VALUE_WEIGHT, token_embedding->name, token_embedding->rank,
                         token_embedding->dims, token_embedding->dtype, YVEX_RESIDENCY_HOST,
                         token_embedding->name, &weight_value, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    hidden_shape[0] = sequence_length;
    hidden_shape[1] = token_embedding->dims[0];
    rc = graph_add_value(graph, YVEX_VALUE_ACTIVATION, "hidden", 2, hidden_shape,
                         token_embedding->dtype, YVEX_RESIDENCY_HOST, NULL, &hidden_value, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    inputs[0] = token_ids_value;
    inputs[1] = weight_value;
    outputs[0] = hidden_value;
    return graph_add_op(graph, YVEX_OP_EMBED, YVEX_OP_STATUS_PLANNED, "embed", inputs, 2, outputs,
                        1, "", err);
}
/* Construct build for model with checked geometry and explicit ownership. */
int yvex_graph_build_for_model(yvex_graph **out, const yvex_model_descriptor *model,
                               const yvex_tensor_table *tensors,
                               const yvex_graph_build_options *options, yvex_error *err) {
    yvex_graph *graph;
    const yvex_tensor_info *token_embedding;
    const yvex_tensor_info *output_norm;
    const yvex_tensor_info *output_head;
    unsigned long long sequence_length = 1;
    unsigned long long context_length = 1;
    int rc;
    if (!out) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_graph_build_for_model", "out is required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = NULL;
    if (!model || !tensors) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_graph_build_for_model",
                       "model and tensors are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (options) {
        if (options->sequence_length > 0) {
            sequence_length = options->sequence_length;
        }
        if (options->context_length > 0) {
            context_length = options->context_length;
        }
        (void)options->include_decode_step;
        (void)options->include_prefill_path;
    }
    if ((!options || options->context_length == 0) && yvex_model_context_length(model) > 0) {
        context_length = yvex_model_context_length(model);
    }
    graph = (yvex_graph *)calloc(1, sizeof(*graph));
    if (!graph) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_build_for_model",
                       "failed to allocate graph");
        return YVEX_ERR_NOMEM;
    }
    graph->status = YVEX_GRAPH_STATUS_EMPTY;
    graph->sequence_length = sequence_length;
    graph->context_length = context_length;
    graph->architecture = yvex_core_strdup(yvex_arch_name(yvex_model_arch(model)));
    graph->model_name = yvex_core_strdup(yvex_model_name(model));
    if (!graph->architecture || !graph->model_name) {
        yvex_graph_close(graph);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_build_for_model",
                       "failed to copy graph labels");
        return YVEX_ERR_NOMEM;
    }
    token_embedding = find_role(tensors, YVEX_TENSOR_ROLE_TOKEN_EMBEDDING);
    output_norm = find_role(tensors, YVEX_TENSOR_ROLE_OUTPUT_NORM);
    output_head = find_role(tensors, YVEX_TENSOR_ROLE_OUTPUT_HEAD);
    rc = add_embedding_path(graph, token_embedding, sequence_length, err);
    if (rc != YVEX_OK) {
        yvex_graph_close(graph);
        return rc;
    }
    rc = add_required_diagnostics(graph, token_embedding, output_norm, output_head, err);
    if (rc != YVEX_OK) {
        yvex_graph_close(graph);
        return rc;
    }
    if (!token_embedding) {
        graph->status = YVEX_GRAPH_STATUS_UNSUPPORTED;
    } else if (graph->missing_count > 0) {
        graph->status = YVEX_GRAPH_STATUS_PARTIAL;
    } else {
        graph->status = YVEX_GRAPH_STATUS_BUILT;
    }
    *out = graph;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_shape_product(const unsigned long long *dims, unsigned int rank, unsigned long long *out,
                       yvex_error *err) {
    unsigned long long product = 1;
    unsigned int i;
    if (!out || (rank > 0 && !dims)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_shape_product",
                       "dims and out are required");
        return YVEX_ERR_INVALID_ARG;
    }
    *out = 0;
    if (rank == 0 || rank > YVEX_GRAPH_MAX_DIMS) {
        yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_shape_product", "rank is out of range");
        return YVEX_ERR_FORMAT;
    }
    for (i = 0; i < rank; ++i) {
        if (dims[i] == 0) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_shape_product",
                           "dimension must be non-zero");
            return YVEX_ERR_FORMAT;
        }
        if (product > ULLONG_MAX / dims[i]) {
            yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_shape_product",
                           "dimension product overflow");
            return YVEX_ERR_BOUNDS;
        }
        product *= dims[i];
    }
    *out = product;
    yvex_error_clear(err);
    return YVEX_OK;
}

int yvex_shape_equal(const unsigned long long *a, unsigned int a_rank, const unsigned long long *b,
                     unsigned int b_rank) {
    unsigned int i;
    if (a_rank != b_rank || a_rank > YVEX_GRAPH_MAX_DIMS || !a || !b) {
        return 0;
    }
    for (i = 0; i < a_rank; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

int yvex_shape_copy(unsigned long long *dst, unsigned int dst_cap, const unsigned long long *src,
                    unsigned int src_rank, yvex_error *err) {
    unsigned int i;
    if (!dst || (src_rank > 0 && !src)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_shape_copy", "src and dst are required");
        return YVEX_ERR_INVALID_ARG;
    }
    if (src_rank > dst_cap || src_rank > YVEX_GRAPH_MAX_DIMS) {
        yvex_error_set(err, YVEX_ERR_BOUNDS, "yvex_shape_copy", "destination shape is too small");
        return YVEX_ERR_BOUNDS;
    }
    for (i = 0; i < src_rank; ++i) {
        if (src[i] == 0) {
            yvex_error_set(err, YVEX_ERR_FORMAT, "yvex_shape_copy", "dimension must be non-zero");
            return YVEX_ERR_FORMAT;
        }
        dst[i] = src[i];
    }
    for (; i < dst_cap; ++i) {
        dst[i] = 0;
    }
    yvex_error_clear(err);
    return YVEX_OK;
}

static void graph_value_clear(yvex_graph_value_info *value) {
    if (!value) {
        return;
    }
    free((char *)value->name);
    free((char *)value->source_tensor_name);
    memset(value, 0, sizeof(*value));
}

static void graph_op_clear(yvex_graph_op_info *op) {
    if (!op) {
        return;
    }
    free((char *)op->name);
    free((char *)op->reason);
    memset(op, 0, sizeof(*op));
}

static void graph_missing_clear(yvex_graph_missing_required *missing) {
    if (!missing) {
        return;
    }
    free((char *)missing->role_name);
    free((char *)missing->reason);
    memset(missing, 0, sizeof(*missing));
}
/*
 * Grow one graph-owned row table with shared checked capacity semantics.
 *
 * Overflow or allocation refusal leaves the original table and capacity unchanged.
 */
static int reserve_rows(void *rows, unsigned long long *capacity, unsigned long long need,
                        size_t row_bytes, const char *where, const char *overflow_message,
                        const char *allocation_message, void **out, yvex_error *err) {
    void *grown;
    unsigned long long next_cap;

    if (*capacity >= need) {
        *out = rows;
        return YVEX_OK;
    }
    next_cap = *capacity == 0 ? 4 : *capacity;
    while (next_cap < need) {
        if (next_cap > ULLONG_MAX / 2u) {
            next_cap = need;
            break;
        }
        next_cap *= 2u;
    }
    if (!row_bytes || next_cap > (unsigned long long)(SIZE_MAX / row_bytes)) {
        yvex_error_set(err, YVEX_ERR_NOMEM, where, overflow_message);
        return YVEX_ERR_NOMEM;
    }
    grown = realloc(rows, (size_t)next_cap * row_bytes);
    if (!grown) {
        yvex_error_set(err, YVEX_ERR_NOMEM, where, allocation_message);
        return YVEX_ERR_NOMEM;
    }
    memset((unsigned char *)grown + (size_t)*capacity * row_bytes, 0,
           (size_t)(next_cap - *capacity) * row_bytes);
    *capacity = next_cap;
    *out = grown;
    return YVEX_OK;
}

static int reserve_ops(yvex_graph *graph, unsigned long long need, yvex_error *err) {
    unsigned long long op_capacity = graph->op_cap, edge_capacity = graph->op_cap;
    void *rows = NULL;
    int rc = reserve_rows(graph->ops, &op_capacity, need, sizeof(*graph->ops),
                          "yvex_graph_add_op", "op table too large",
                          "failed to grow op table", &rows, err);
    if (rc != YVEX_OK) return rc;
    graph->ops = (yvex_graph_op_info *)rows;
    rc = reserve_rows(graph->edges, &edge_capacity, need, sizeof(*graph->edges),
                      "yvex_graph_add_op", "op table too large",
                      "failed to grow op edge table", &rows, err);
    if (rc != YVEX_OK) return rc;
    graph->edges = (yvex_graph_op_edges *)rows;
    graph->op_cap = op_capacity;
    return YVEX_OK;
}

static int graph_add_value(yvex_graph *graph, yvex_value_kind kind, const char *name,
                           unsigned int rank, const unsigned long long *dims, yvex_dtype dtype,
                           yvex_residency residency, const char *source_tensor_name,
                           unsigned int *out_id, yvex_error *err) {
    yvex_graph_value_info *value;
    unsigned int i;
    int rc;
    if (!graph || !name || rank > YVEX_GRAPH_MAX_DIMS || (rank > 0 && !dims)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_graph_add_value",
                       "invalid value arguments");
        return YVEX_ERR_INVALID_ARG;
    }
    {
        void *rows = NULL;
        rc = reserve_rows(graph->values, &graph->value_cap, graph->value_count + 1u,
                          sizeof(*graph->values), "yvex_graph_add_value",
                          "value table too large", "failed to grow value table",
                          &rows, err);
        if (rc == YVEX_OK)
            graph->values = (yvex_graph_value_info *)rows;
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    value = &graph->values[graph->value_count];
    value->id = (unsigned int)graph->value_count;
    value->kind = kind;
    value->name = yvex_core_strdup(name);
    if (!value->name) {
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_add_value", "failed to copy value name");
        return YVEX_ERR_NOMEM;
    }
    value->rank = rank;
    for (i = 0; i < rank; ++i) {
        value->dims[i] = dims[i];
    }
    value->dtype = dtype;
    value->residency = residency;
    if (source_tensor_name) {
        value->source_tensor_name = yvex_core_strdup(source_tensor_name);
        if (!value->source_tensor_name) {
            graph_value_clear(value);
            yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_add_value",
                           "failed to copy source tensor");
            return YVEX_ERR_NOMEM;
        }
    }
    if (out_id) {
        *out_id = value->id;
    }
    graph->value_count += 1u;
    return YVEX_OK;
}

static int graph_add_op(yvex_graph *graph, yvex_op_kind kind, yvex_op_status status,
                        const char *name, const unsigned int *inputs, unsigned int input_count,
                        const unsigned int *outputs, unsigned int output_count, const char *reason,
                        yvex_error *err) {
    yvex_graph_op_info *op;
    yvex_graph_op_edges *edges;
    unsigned int i;
    int rc;
    if (!graph || !name || input_count > 4u || output_count > 4u || (input_count > 0 && !inputs) ||
        (output_count > 0 && !outputs)) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_graph_add_op", "invalid op arguments");
        return YVEX_ERR_INVALID_ARG;
    }
    rc = reserve_ops(graph, graph->op_count + 1u, err);
    if (rc != YVEX_OK) {
        return rc;
    }
    op = &graph->ops[graph->op_count];
    edges = &graph->edges[graph->op_count];
    op->id = (unsigned int)graph->op_count;
    op->kind = kind;
    op->status = status;
    op->name = yvex_core_strdup(name);
    op->reason = yvex_core_strdup(reason ? reason : "");
    if (!op->name || !op->reason) {
        graph_op_clear(op);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_add_op", "failed to copy op text");
        return YVEX_ERR_NOMEM;
    }
    op->input_count = input_count;
    op->output_count = output_count;
    for (i = 0; i < input_count; ++i) {
        edges->input_ids[i] = inputs[i];
    }
    for (i = 0; i < output_count; ++i) {
        edges->output_ids[i] = outputs[i];
    }
    graph->op_count += 1u;
    return YVEX_OK;
}

static int graph_add_missing(yvex_graph *graph, yvex_tensor_role role, const char *reason,
                             yvex_error *err) {
    yvex_graph_missing_required *missing;
    int rc;
    if (!graph) {
        yvex_error_set(err, YVEX_ERR_INVALID_ARG, "yvex_graph_add_missing", "graph is required");
        return YVEX_ERR_INVALID_ARG;
    }
    {
        void *rows = NULL;
        rc = reserve_rows(graph->missing, &graph->missing_cap,
                          graph->missing_count + 1u, sizeof(*graph->missing),
                          "yvex_graph_add_missing", "missing table too large",
                          "failed to grow missing table", &rows, err);
        if (rc == YVEX_OK)
            graph->missing = (yvex_graph_missing_required *)rows;
    }
    if (rc != YVEX_OK) {
        return rc;
    }
    missing = &graph->missing[graph->missing_count];
    missing->role = role;
    missing->role_name = yvex_core_strdup(yvex_tensor_role_name(role));
    missing->reason = yvex_core_strdup(reason ? reason : "");
    if (!missing->role_name || !missing->reason) {
        graph_missing_clear(missing);
        yvex_error_set(err, YVEX_ERR_NOMEM, "yvex_graph_add_missing",
                       "failed to copy missing diagnostic");
        return YVEX_ERR_NOMEM;
    }
    graph->missing_count += 1u;
    return YVEX_OK;
}
