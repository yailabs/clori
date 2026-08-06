/*
 * Execute admitted encoded DeepSeek attention requests through generated CUDA kernels.
 *
 * Weights stay encoded; numerical work uses generated PTX; publication follows device completion.
 * Device-complete request execution is not persistent KV or a decode loop.
 */
#include <yvex/internal/backend.h>
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
enum { YVEX_CUDA_ATTN_BLOCK = 256u, YVEX_CUDA_ATTN_TRANSFERS = 20u, YVEX_CUDA_ATTN_UPLOADS = 16u };
enum { YVEX_CUDA_ATTN_INITIALIZERS = 64u, YVEX_CUDA_ATTN_CSA_RATIO = 4u, YVEX_CUDA_ATTN_HCA_RATIO = 128u };
enum { ROLL_MAIN = 0, ROLL_INDEX, ROLL_COUNT };
typedef struct {
    CUdeviceptr kv, score, ape, before_kv, before_score, after_kv, after_score, value, positions, scratch_value;
    unsigned long long extent, value_count, value_extent;
} attn_rolling_run;
typedef yvex_cuda_work attn_resources;
typedef yvex_cuda_attention_transfer attn_transfer;
typedef yvex_cuda_attention_upload attn_upload;
typedef struct {
    CUdeviceptr *device; const void *source; CUdeviceptr device_source;
    size_t bytes; int zero; const char *stage;
} attn_initializer;
typedef enum {
    CUDA_DIM_ONE = 0, CUDA_DIM_HIDDEN, CUDA_DIM_Q_RANK, CUDA_DIM_QUERY_WIDTH, CUDA_DIM_KV_WIDTH,
    CUDA_DIM_QUERY_HEADS, CUDA_DIM_LOW_COUNT, CUDA_DIM_OUTPUT_GROUP_INPUT, CUDA_DIM_MAIN_STATE_WIDTH,
    CUDA_DIM_MAIN_RATIO, CUDA_DIM_HEAD, CUDA_DIM_INDEX_STATE_WIDTH, CUDA_DIM_INDEX_RATIO,
    CUDA_DIM_INDEX_HEAD, CUDA_DIM_INDEX_QUERY, CUDA_DIM_INDEX_HEADS, CUDA_DIM_MHC_ROWS,
    CUDA_DIM_RESIDUAL_EXPANDED, CUDA_DIM_MHC_BASE, CUDA_DIM_MHC_SCALE
} attn_dimension;
typedef struct {
    yvex_backend_attention_weight_slot slot; attn_dimension rows, width; unsigned int class_mask;
} attn_weight_shape;
static const attn_weight_shape attn_weight_shapes[] = {
    {YVEX_BACKEND_ATTENTION_WEIGHT_Q_A, CUDA_DIM_Q_RANK, CUDA_DIM_HIDDEN, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM, CUDA_DIM_ONE, CUDA_DIM_Q_RANK, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_Q_B, CUDA_DIM_QUERY_WIDTH, CUDA_DIM_Q_RANK, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_KV, CUDA_DIM_KV_WIDTH, CUDA_DIM_HIDDEN, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM, CUDA_DIM_ONE, CUDA_DIM_KV_WIDTH, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_SINKS, CUDA_DIM_ONE, CUDA_DIM_QUERY_HEADS, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A, CUDA_DIM_LOW_COUNT, CUDA_DIM_OUTPUT_GROUP_INPUT, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B, CUDA_DIM_HIDDEN, CUDA_DIM_LOW_COUNT, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_KV, CUDA_DIM_MAIN_STATE_WIDTH, CUDA_DIM_HIDDEN, 6u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_GATE, CUDA_DIM_MAIN_STATE_WIDTH, CUDA_DIM_HIDDEN, 6u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_APE, CUDA_DIM_MAIN_RATIO, CUDA_DIM_MAIN_STATE_WIDTH, 6u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_NORM, CUDA_DIM_ONE, CUDA_DIM_HEAD, 6u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_KV, CUDA_DIM_INDEX_STATE_WIDTH, CUDA_DIM_HIDDEN, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_GATE, CUDA_DIM_INDEX_STATE_WIDTH, CUDA_DIM_HIDDEN, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_APE, CUDA_DIM_INDEX_RATIO, CUDA_DIM_INDEX_STATE_WIDTH, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_NORM, CUDA_DIM_ONE, CUDA_DIM_INDEX_HEAD, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q, CUDA_DIM_INDEX_QUERY, CUDA_DIM_Q_RANK, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION, CUDA_DIM_INDEX_HEADS, CUDA_DIM_HIDDEN, 2u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION, CUDA_DIM_MHC_ROWS, CUDA_DIM_RESIDUAL_EXPANDED, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE, CUDA_DIM_ONE, CUDA_DIM_MHC_BASE, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE, CUDA_DIM_ONE, CUDA_DIM_MHC_SCALE, 7u},
    {YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM, CUDA_DIM_ONE, CUDA_DIM_HIDDEN, 7u}
};
typedef struct {
    attn_resources resources;
    const yvex_cuda_attention_operations *ops;
    yvex_backend *backend; yvex_cuda_backend_state *state;
    yvex_backend_attention_job request;
    yvex_backend_attention_job *job; yvex_backend_attention_output *output;
    yvex_backend_attention_failure *failure; yvex_error *err;
    unsigned char *host_stage; size_t host_stage_bytes;
    attn_transfer transfers[YVEX_CUDA_ATTN_TRANSFERS]; size_t transfer_count;
    attn_upload uploads[YVEX_CUDA_ATTN_UPLOADS]; size_t upload_count;
    attn_initializer initializers[YVEX_CUDA_ATTN_INITIALIZERS]; size_t initializer_count;
    CUdeviceptr weight[YVEX_BACKEND_ATTENTION_WEIGHT_COUNT];
    CUdeviceptr input, core_input, q_low, query, raw_kv;
    CUdeviceptr sinks, attention, low, final_output, envelope_output;
    CUdeviceptr mhc_mix, mhc_scale, mhc_base, mhc_post, mhc_combination;
    attn_rolling_run rolling[ROLL_COUNT];
    CUdeviceptr index_query, index_weights, history_local, history_local_positions;
    CUdeviceptr history_compressed, history_compressed_positions;
    CUdeviceptr history_indexer, history_indexer_positions;
    CUdeviceptr selected, selected_positions, selected_count, valid_count;
    CUdeviceptr topk_scores, valid_indexes, device_status;
    unsigned long long query_width, candidate_capacity, topk_capacity, low_count;
    unsigned long long local_extent, compressed_extent, history_index_extent;
    unsigned long long local_storage_extent, compressed_storage_extent, indexer_storage_extent;
    unsigned long long local_capacity, compressed_capacity, indexer_capacity;
    unsigned long long index_query_extent, emission_position, topk_count, staged_valid_count;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes, device_execution_elapsed_ns;
    unsigned long long stream_synchronizations, device_synchronizations;
    int *staged_status; unsigned long long *staged_selected_count, *staged_candidate_count;
    int host_workspace_reused, host_status;
    unsigned long long ordinal, phase_start_position, input_extent;
    unsigned long long phase_compressed_count, phase_indexer_count;
    unsigned long long initial_local_count, initial_compressed_count, initial_indexer_count;
    unsigned long long phase_local_storage_count, phase_compressed_storage_count;
    unsigned long long phase_indexer_storage_count;
    CUdeviceptr phase_input, phase_core_input, phase_q_low, phase_query, phase_raw_kv;
    CUdeviceptr phase_attention, phase_low, phase_output, phase_envelope_output;
    CUdeviceptr phase_mhc_mix, phase_mhc_scale, phase_mhc_base, phase_mhc_post;
    CUdeviceptr phase_mhc_combination, phase_index_query, phase_index_weights;
    CUdeviceptr phase_selected, phase_selected_positions, phase_selected_count, phase_valid_count;
    CUdeviceptr phase_local, phase_local_positions, phase_compressed, phase_compressed_positions;
    CUdeviceptr phase_indexer, phase_indexer_positions, phase_main_kv, phase_main_score;
    CUdeviceptr phase_index_kv, phase_index_score;
    CUdeviceptr phase_new_compressed, phase_new_compressed_positions;
    CUdeviceptr phase_new_indexer, phase_new_indexer_positions;
    unsigned long long *phase_topk_counts, *phase_valid_counts;
} attn_run;
typedef enum {
    EXT_ONE = 0, EXT_INPUT, EXT_CORE, EXT_RAW_KV, EXT_Q_LOW, EXT_QUERY, EXT_LOW, EXT_ENVELOPE, EXT_MHC_MIX,
    EXT_MHC_SCALE, EXT_MHC_POST, EXT_MHC_COMBINATION, EXT_TOKENS, EXT_LOCAL,
    EXT_LOCAL_POSITIONS, EXT_COMPRESSED, EXT_COMPRESSED_POSITIONS, EXT_INDEXER,
    EXT_INDEXER_POSITIONS, EXT_MAIN_STATE, EXT_INDEX_STATE,
    EXT_INDEX_QUERY, EXT_INDEX_WEIGHTS, EXT_SELECTED, EXT_CANDIDATES, EXT_QUERY_HEADS, EXT_PHASE_COMPRESSED,
    EXT_PHASE_COMPRESSED_POSITIONS, EXT_PHASE_INDEXER, EXT_PHASE_INDEXER_POSITIONS,
    EXT_MAIN_ROLLING, EXT_INDEX_ROLLING, EXT_MAIN_PUBLICATION, EXT_INDEX_PUBLICATION,
    EXT_LOCAL_USED, EXT_COMPRESSED_USED, EXT_INDEXER_USED,
    EXT_MAIN_WIDTH, EXT_MAIN_HEAD, EXT_INDEX_WIDTH, EXT_INDEX_HEAD, EXT_MAIN_POSITION, EXT_INDEX_POSITION
} attn_extent_kind;
typedef enum {
    SRC_NONE = 0, SRC_INPUT, SRC_LOCAL, SRC_LOCAL_POSITIONS, SRC_COMPRESSED, SRC_COMPRESSED_POSITIONS,
    SRC_INDEXER, SRC_INDEXER_POSITIONS, SRC_MAIN_KV, SRC_MAIN_SCORE, SRC_INDEX_KV, SRC_INDEX_SCORE
} attn_source_kind;
static int attn_stage_layout(attn_run *run, unsigned char *base, size_t *total);
static int attn_extent(const attn_run *run, attn_extent_kind kind, unsigned long long *out);
static const void *attn_allocation_source(const attn_run *run, attn_source_kind source);
static int attn_graph_mode(const attn_run *run) {
    return run->state->attention_graph_configured &&
           run->state->attention_mode != YVEX_BACKEND_CUDA_ATTENTION_EAGER;
}
static int attn_run_fail(attn_run *run, yvex_backend_attention_failure_code code,
                                   const char *stage, unsigned long long expected, unsigned long long actual,
                                   yvex_status status, const char *message) {
    return run->ops->fail(
        run->failure, code, stage, expected, actual, run->err, status, message);
}
static int attn_account_d2d(attn_run *run, size_t bytes, const char *stage) {
    if (yvex_core_u64_add(run->d2d_bytes, bytes, &run->d2d_bytes)) return YVEX_OK;
    return attn_run_fail(run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage,
                         ULLONG_MAX, bytes, YVEX_ERR_BOUNDS,
                         "CUDA attention D2D accounting overflowed");
}
static int attn_cancel(attn_run *run, const char *stage, int device_work_pending) {
    return run->ops->cancel(
        run->backend, run->job, stage, device_work_pending,
        run->failure, run->err);
}
static int attn_transfer_add(attn_run *run, CUdeviceptr *device, void *output,
                                       unsigned long long output_capacity, unsigned long long capacity,
                                       unsigned long long *used, size_t width, const char *stage) {
    size_t bytes;
    attn_transfer *transfer;
    if (!capacity) return YVEX_OK;
    if (run->transfer_count >= YVEX_CUDA_ATTN_TRANSFERS ||
        !output || (!used && output_capacity < capacity) || (used && !output_capacity) ||
        !yvex_cuda_work_checked_bytes(capacity, width, &bytes) ||
        (uintptr_t)output > UINTPTR_MAX - bytes)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.output", capacity,
            output_capacity, YVEX_ERR_BOUNDS,
            "CUDA attention output span is absent, invalid, or undersized");
    transfer = &run->transfers[run->transfer_count++];
    *transfer = (attn_transfer){device, output, NULL, capacity, output_capacity, used, width, stage};
    return run->ops->account_transfer(
        capacity, width, &run->d2h_bytes, stage, run->failure, run->err);
}
typedef enum { SPAN_FLOAT = 0, SPAN_U64 } attn_span_kind;
typedef struct {
    size_t device_offset, output_offset, used_offset;
    attn_extent_kind extent;
    attn_span_kind span;
    unsigned int class_mask, evidence;
    int envelope_only;
    const char *stage;
} attn_transfer_spec;
#define T(device, output, used, extent, span, classes, evidence, envelope, label) \
    {offsetof(attn_run, device), offsetof(yvex_backend_attention_output, output), \
     used, extent, span, classes, evidence, envelope, "cuda.deepseek_attention.copy." label}
static const attn_transfer_spec attn_transfers[] = {
    T(phase_core_input, core_input, SIZE_MAX, EXT_CORE, SPAN_FLOAT, 7u, 3u, 0, "core_input"),
    T(phase_q_low, q_low, SIZE_MAX, EXT_Q_LOW, SPAN_FLOAT, 7u, 2u, 0, "q_low"),
    T(phase_query, query, SIZE_MAX, EXT_QUERY, SPAN_FLOAT, 7u, 2u, 0, "query"),
    T(phase_raw_kv, raw_kv, SIZE_MAX, EXT_RAW_KV, SPAN_FLOAT, 7u, 0u, 0, "raw_kv"),
    T(phase_attention, attention_values, SIZE_MAX, EXT_QUERY, SPAN_FLOAT, 7u, 2u, 0, "attention"),
    T(phase_output, output, SIZE_MAX, EXT_CORE, SPAN_FLOAT, 7u, 1u, 0, "output"),
    T(phase_envelope_output, envelope_output, SIZE_MAX, EXT_ENVELOPE,
      SPAN_FLOAT, 7u, 1u, 1, "envelope_output"),
    T(phase_new_compressed, compressed_kv, SIZE_MAX, EXT_PHASE_COMPRESSED, SPAN_FLOAT, 6u, 0u, 0, "compressed"),
    T(phase_new_compressed_positions, compressed_positions,
      offsetof(attn_run, phase_compressed_count), EXT_PHASE_COMPRESSED_POSITIONS,
      SPAN_U64, 6u, 0u, 0, "compressed_positions"),
    T(phase_new_indexer, indexer_kv, SIZE_MAX, EXT_PHASE_INDEXER, SPAN_FLOAT, 2u, 0u, 0, "indexer"),
    T(phase_new_indexer_positions, indexer_positions,
      offsetof(attn_run, phase_indexer_count), EXT_PHASE_INDEXER_POSITIONS,
      SPAN_U64, 2u, 0u, 0, "indexer_positions"),
    T(phase_index_query, index_query, SIZE_MAX, EXT_INDEX_QUERY, SPAN_FLOAT, 2u, 2u, 0, "index_query"),
    T(phase_index_weights, index_weights, SIZE_MAX, EXT_INDEX_WEIGHTS, SPAN_FLOAT, 2u, 2u, 0, "index_weights"),
    T(phase_selected_positions, topk_positions, SIZE_MAX, EXT_SELECTED, SPAN_U64, 2u, 3u, 0, "topk_positions"),
    T(rolling[ROLL_MAIN].after_kv, main_kv_state, SIZE_MAX,
      EXT_MAIN_PUBLICATION, SPAN_FLOAT, 6u, 0u, 0, "main_kv_state"),
    T(rolling[ROLL_MAIN].after_score, main_score_state, SIZE_MAX,
      EXT_MAIN_PUBLICATION, SPAN_FLOAT, 6u, 0u, 0, "main_score_state"),
    T(rolling[ROLL_INDEX].after_kv, indexer_kv_state, SIZE_MAX,
      EXT_INDEX_PUBLICATION, SPAN_FLOAT, 2u, 0u, 0, "index_kv_state"),
    T(rolling[ROLL_INDEX].after_score, indexer_score_state, SIZE_MAX,
      EXT_INDEX_PUBLICATION, SPAN_FLOAT, 2u, 0u, 0, "index_score_state")
};
#undef T
static int attn_transfer_plan(attn_run *run) {
    size_t index;
    int rc;
    rc = run->ops->account_transfer(
        1ull, sizeof(int), &run->d2h_bytes,
        "cuda.deepseek_attention.copy.status", run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->account_transfer(
            run->job->token_count, sizeof(unsigned long long),
            &run->d2h_bytes, "cuda.deepseek_attention.copy.counts",
            run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->account_transfer(
            run->job->token_count, sizeof(unsigned long long),
            &run->d2h_bytes, "cuda.deepseek_attention.copy.counts",
            run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    for (index = 0u; index < sizeof(attn_transfers) /
                                  sizeof(attn_transfers[0]); ++index) {
        const attn_transfer_spec *spec = &attn_transfers[index];
        CUdeviceptr *device;
        unsigned long long *used = NULL, extent;
        void *data;
        unsigned long long capacity;
        if (!(spec->class_mask & (1u << run->job->attention_class)) ||
            run->job->evidence_level < spec->evidence ||
            (spec->envelope_only &&
             run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE))
            continue;
        if (!attn_extent(run, spec->extent, &extent))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, spec->stage,
                ULLONG_MAX, 0ull, YVEX_ERR_BOUNDS,
                "CUDA attention publication extent overflowed");
        device = (CUdeviceptr *)((unsigned char *)run + spec->device_offset);
        if (spec->span == SPAN_U64) {
            yvex_backend_u64_span *span = (yvex_backend_u64_span *)
                ((unsigned char *)run->output + spec->output_offset);
            data = span->data;
            capacity = span->capacity;
        } else {
            yvex_backend_float_span *span = (yvex_backend_float_span *)
                ((unsigned char *)run->output + spec->output_offset);
            data = span->data;
            capacity = span->capacity;
        }
        if (spec->used_offset != SIZE_MAX)
            used = (unsigned long long *)((unsigned char *)run + spec->used_offset);
        rc = attn_transfer_add(
            run, device, data, capacity, extent, used,
            spec->span == SPAN_U64 ? sizeof(unsigned long long) : sizeof(float),
            spec->stage);
        if (rc != YVEX_OK) return rc;
    }
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
        ((run->output->topk_counts.data &&
          run->output->topk_counts.capacity < run->job->token_count) ||
         (run->output->valid_candidate_counts.data &&
          run->output->valid_candidate_counts.capacity < run->job->token_count)))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.copy.phase_counts", run->job->token_count,
            run->output->topk_counts.capacity, YVEX_ERR_BOUNDS,
            "CUDA attention per-token count spans are undersized");
    return YVEX_OK;
}
typedef struct {
    size_t device_offset;
    attn_source_kind source;
    attn_extent_kind capacity, used;
    size_t width;
    int generated;
    const char *stage;
} attn_upload_spec;
#define U(device, source, capacity, used, width, generated, label) \
    {offsetof(attn_run, device), source, capacity, used, width, generated, \
     "cuda.deepseek_attention.upload." label}
static const attn_upload_spec attn_uploads[] = {
    U(phase_input, SRC_INPUT, EXT_INPUT, EXT_INPUT, sizeof(float), 0, "input"),
    U(phase_local, SRC_LOCAL, EXT_LOCAL, EXT_LOCAL_USED, sizeof(float), 0, "local_history"),
    U(phase_local_positions, SRC_LOCAL_POSITIONS, EXT_LOCAL_POSITIONS,
      EXT_LOCAL_POSITIONS, sizeof(uint64_t), 1, "local_positions"),
    U(phase_compressed, SRC_COMPRESSED, EXT_COMPRESSED, EXT_COMPRESSED_USED, sizeof(float), 0, "compressed_history"),
    U(phase_compressed_positions, SRC_COMPRESSED_POSITIONS,
      EXT_COMPRESSED_POSITIONS, EXT_COMPRESSED_POSITIONS, sizeof(uint64_t), 1,
      "compressed_positions"),
    U(phase_indexer, SRC_INDEXER, EXT_INDEXER, EXT_INDEXER_USED, sizeof(float), 0, "indexer_history"),
    U(phase_indexer_positions, SRC_INDEXER_POSITIONS, EXT_INDEXER_POSITIONS,
      EXT_INDEXER_POSITIONS, sizeof(uint64_t), 1, "indexer_positions"),
    U(phase_main_kv, SRC_MAIN_KV, EXT_MAIN_STATE, EXT_MAIN_ROLLING, sizeof(float), 0, "main_kv_state"),
    U(phase_main_score, SRC_MAIN_SCORE, EXT_MAIN_STATE, EXT_MAIN_ROLLING, sizeof(float), 0, "main_score_state"),
    U(phase_index_kv, SRC_INDEX_KV, EXT_INDEX_STATE, EXT_INDEX_ROLLING, sizeof(float), 0, "index_kv_state"),
    U(phase_index_score, SRC_INDEX_SCORE, EXT_INDEX_STATE, EXT_INDEX_ROLLING, sizeof(float), 0, "index_score_state")
};
#undef U
static int attn_upload_plan(attn_run *run) {
    size_t index;
    for (index = 0u; index < sizeof(attn_uploads) /
                                  sizeof(attn_uploads[0]); ++index) {
        const attn_upload_spec *spec = &attn_uploads[index];
        attn_upload upload;
        size_t bytes;
        if (spec->device_offset == offsetof(attn_run, phase_input) && run->job->device_input)
            continue;
        if (!attn_extent(run, spec->capacity, &upload.count) ||
            !attn_extent(run, spec->used, &upload.used))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, spec->stage,
                ULLONG_MAX, 0ull, YVEX_ERR_BOUNDS,
                "CUDA attention graph input extent overflowed");
        if (!upload.count) continue;
        upload.device = (CUdeviceptr *)((unsigned char *)run + spec->device_offset);
        upload.source = attn_allocation_source(run, spec->source);
        upload.staged = NULL;
        upload.width = spec->width;
        upload.generated = spec->generated;
        upload.stage = spec->stage;
        if ((upload.used && !upload.source && !spec->generated) ||
            upload.used > upload.count ||
            run->upload_count >= YVEX_CUDA_ATTN_UPLOADS ||
            !yvex_cuda_work_checked_bytes(upload.count, upload.width, &bytes))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
                upload.stage, upload.count, upload.used, YVEX_ERR_BOUNDS,
                "CUDA attention graph input is invalid");
        run->uploads[run->upload_count++] = upload;
    }
    return YVEX_OK;
}
static int attn_alias_validate(attn_run *run) {
    int verdict = run->ops->validate_alias(
        run->job, run->transfers, run->transfer_count, run->local_extent,
        run->compressed_extent, run->history_index_extent,
        run->rolling[ROLL_MAIN].extent, run->rolling[ROLL_INDEX].extent);
    if (verdict > 0) return YVEX_OK;
    return attn_run_fail(
        run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
        verdict ? "cuda.deepseek_attention.validate.alias"
                : "cuda.deepseek_attention.validate.alias.range",
        verdict ? 0ull : 1ull, verdict ? 1ull : 0ull,
        verdict ? YVEX_ERR_INVALID_ARG : YVEX_ERR_BOUNDS,
        verdict ? "CUDA attention output aliases another output or immutable input"
                : "CUDA attention host range is not representable");
}
static unsigned long long attn_dimension_value(const attn_run *run,
                                                         attn_dimension dimension) {
    const unsigned long long values[] = {
        1ull, run->job->hidden_width, run->job->q_rank, run->query_width, run->job->kv_width,
        run->job->query_heads, run->low_count, run->job->output_group_input_width,
        run->job->main_rolling.state_width, run->job->main_rolling.ratio, run->job->head_dimension,
        run->job->indexer_rolling.state_width, run->job->indexer_rolling.ratio,
        run->job->indexer_rolling.head_dimension, run->index_query_extent,
        run->job->indexer_heads, run->job->mhc_mixing_rows,
        run->job->residual_expanded_width, run->job->mhc_mixing_rows, 3ull
    };
    return (unsigned int)dimension < sizeof(values) / sizeof(values[0]) ? values[dimension] : 0ull;
}
static int attn_validate_derived(attn_run *run) {
    unsigned long long ratio = run->job->compression_ratio, candidate_count = 0ull, group_width = 0ull;
    size_t i;
    int rc;
    if (run->job->token_position == ULLONG_MAX)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.position", ULLONG_MAX - 1ull,
            run->job->token_position, YVEX_ERR_BOUNDS,
            "CUDA attention token position cannot advance safely");
    if (run->job->kv_width != run->job->head_dimension ||
        !run->job->sliding_window || !run->job->output_groups ||
        !run->job->output_group_input_width || !run->job->output_rank ||
        !yvex_core_u64_mul(run->job->output_groups,
                           run->job->output_group_input_width, &group_width) ||
        group_width != run->query_width || run->job->rms_epsilon <= 0.0 ||
        run->job->position.theta <= 1ull ||
        !run->job->position.scaling_factor ||
        (run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA
             ? run->job->position.original_context != 0ull
             : run->job->position.original_context == 0ull) ||
        !run->job->position.beta_slow ||
        run->job->position.beta_fast <= run->job->position.beta_slow ||
        !run->job->position.rope_dimensions ||
        (run->job->position.rope_dimensions & 1ull) ||
        run->job->position.rope_dimensions > run->job->head_dimension ||
        run->job->local_count > run->job->sliding_window)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.geometry", run->query_width,
            group_width, YVEX_ERR_FORMAT,
            "CUDA attention interdependent geometry is invalid");
    if ((run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA && ratio != 0ull) ||
        (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
         ratio != YVEX_CUDA_ATTN_CSA_RATIO) ||
        (run->job->attention_class == YVEX_BACKEND_ATTENTION_HCA &&
         ratio != YVEX_CUDA_ATTN_HCA_RATIO))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.ratio",
            run->job->attention_class == YVEX_BACKEND_ATTENTION_HCA
                ? YVEX_CUDA_ATTN_HCA_RATIO
                : run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
                    ? YVEX_CUDA_ATTN_CSA_RATIO : 0ull,
            ratio, YVEX_ERR_FORMAT,
            "CUDA attention class/compression ratio is invalid");
    if (run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE &&
        (!run->job->residual_stream_count || !run->job->residual_stream_width ||
         run->job->residual_stream_width != run->job->hidden_width ||
         !yvex_core_u64_mul(run->job->residual_stream_count,
                            run->job->residual_stream_width, &group_width) ||
         group_width != run->job->residual_expanded_width ||
         run->job->residual_stream_count > ULLONG_MAX - 2ull ||
         (run->job->residual_stream_count + 2ull) >
             ULLONG_MAX / run->job->residual_stream_count ||
         (run->job->residual_stream_count + 2ull) *
                 run->job->residual_stream_count != run->job->mhc_mixing_rows ||
         !run->job->mhc_sinkhorn_iterations || run->job->mhc_epsilon <= 0.0 ||
         run->job->mhc_residual_post_multiplier <= 0.0))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.envelope",
            run->job->residual_expanded_width, group_width, YVEX_ERR_FORMAT,
            "CUDA attention envelope geometry is invalid");
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
        run->job->compressed_stride != run->job->head_dimension)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.compressed_stride",
            run->job->head_dimension, run->job->compressed_stride,
            YVEX_ERR_FORMAT,
            "CUDA compressed attention phase stride is invalid");
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA) {
        rc = run->ops->validate_rolling(
            run->job, &run->job->main_rolling, ratio, run->job->head_dimension,
            run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA,
            &run->rolling[ROLL_MAIN].extent,
            "cuda.deepseek_attention.validate.main_state", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
        run->rolling[ROLL_MAIN].value_count =
            ((run->job->token_position + 1ull) % ratio) == 0ull ? 1ull : 0ull;
        if (run->rolling[ROLL_MAIN].value_count)
            run->emission_position = run->job->token_position + 1ull - ratio;
        if (!yvex_core_u64_mul(run->rolling[ROLL_MAIN].value_count,
                               run->job->head_dimension,
                               &run->rolling[ROLL_MAIN].value_extent))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.deepseek_attention.validate.compressed_extent",
                ULLONG_MAX, run->job->head_dimension, YVEX_ERR_BOUNDS,
                "CUDA attention compressed output extent overflowed");
    } else if (run->job->main_rolling.present || run->job->indexer_rolling.present ||
               run->job->compressed_count || run->job->indexer_count) {
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.swa_state", 0ull, 1ull,
            YVEX_ERR_FORMAT,
            "CUDA SWA cannot consume compressed rolling state");
    }
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA) {
        if (!run->job->indexer_heads || !run->job->indexer_head_dimension ||
            !run->job->indexer_topk ||
            run->job->position.rope_dimensions >
                run->job->indexer_head_dimension ||
            run->job->indexer_stride != run->job->indexer_head_dimension)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
                "cuda.deepseek_attention.validate.index_geometry",
                run->job->head_dimension, run->job->compressed_stride,
                YVEX_ERR_FORMAT,
                "CUDA CSA indexer and compressed geometry is invalid");
        rc = run->ops->validate_rolling(
            run->job, &run->job->indexer_rolling, ratio,
            run->job->indexer_head_dimension, 1,
            &run->rolling[ROLL_INDEX].extent,
            "cuda.deepseek_attention.validate.index_state", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
        run->rolling[ROLL_INDEX].value_count =
            run->rolling[ROLL_MAIN].value_count;
        if (!yvex_core_u64_mul(run->rolling[ROLL_INDEX].value_count,
                               run->job->indexer_head_dimension,
                               &run->rolling[ROLL_INDEX].value_extent))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.deepseek_attention.validate.indexer_extent",
                ULLONG_MAX, run->job->indexer_head_dimension, YVEX_ERR_BOUNDS,
                "CUDA attention indexer output extent overflowed");
        if (!yvex_core_u64_add(run->indexer_capacity, 1ull,
                               &candidate_count))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.deepseek_attention.validate.candidates", ULLONG_MAX,
                run->job->indexer_count, YVEX_ERR_BOUNDS,
                "CUDA attention candidate count overflowed");
        run->candidate_capacity = candidate_count ? candidate_count : 1ull;
        run->topk_capacity = run->job->indexer_topk < candidate_count
            ? run->job->indexer_topk : candidate_count;
        if (!run->topk_capacity) run->topk_capacity = 1ull;
    } else if (run->job->indexer_rolling.present || run->job->indexer_count) {
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.index_state", 0ull, 1ull,
            YVEX_ERR_FORMAT,
            "CUDA non-CSA attention cannot consume indexer state");
    }
    for (i = 0u;
         i < sizeof(attn_weight_shapes) /
                 sizeof(attn_weight_shapes[0]);
         ++i) {
        const attn_weight_shape *shape = &attn_weight_shapes[i];
        if (!(shape->class_mask & (1u << run->job->attention_class))) continue;
        if (shape->slot >= YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION &&
            run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
            continue;
        rc = run->ops->validate_weight(
            &run->job->weights[shape->slot],
            attn_dimension_value(run, shape->rows),
            attn_dimension_value(run, shape->width), run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    rc = run->ops->validate_activation(
        &run->job->attention_kv_activation,
        run->job->kv_width - run->job->position.rope_dimensions,
        "cuda.deepseek_attention.validate.kv_activation", run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = run->ops->validate_activation(
            &run->job->compressor_activation,
            run->job->head_dimension - run->job->position.rope_dimensions,
            "cuda.deepseek_attention.validate.compressor_activation",
            run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = run->ops->validate_activation(
            &run->job->compressor_rotated_activation,
            run->job->indexer_head_dimension,
            "cuda.deepseek_attention.validate.compressor_rotated_activation",
            run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->validate_activation(
            &run->job->indexer_query_activation,
            run->job->indexer_head_dimension,
            "cuda.deepseek_attention.validate.indexer_activation",
            run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = attn_transfer_plan(run);
    if (rc == YVEX_OK) rc = attn_upload_plan(run);
    if (rc == YVEX_OK) rc = attn_alias_validate(run);
    return rc;
}
static const void *attn_upload_source(const attn_run *run, const CUdeviceptr *target, int *generated) {
    size_t i;
    for (i = 0u; i < run->upload_count; ++i) {
        if (run->uploads[i].device != target) continue;
        *generated = run->uploads[i].generated;
        return run->uploads[i].staged;
    }
    return NULL;
}
/*
 * Allocate typed device values.
 *
 * Cleanup remains transaction-owned.
 */
static int attn_alloc_values(attn_run *run, CUdeviceptr *target,
                             unsigned long long count, size_t width,
                             const void *source, int zero, const char *stage) {
    const void *stable_source = source, *planned_source;
    unsigned long long resident_address = 0ull;
    CUdeviceptr device_source = 0u;
    size_t bytes;
    int captured = attn_graph_mode(run);
    int generated = 0, resident, rc;
    if (*target) return YVEX_OK;
    planned_source = attn_upload_source(run, target, &generated);
    if (!yvex_cuda_work_checked_bytes(count, (unsigned long long)width, &bytes))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage, ULLONG_MAX,
            count, YVEX_ERR_BOUNDS,
            "CUDA attention allocation size overflowed");
    if (target == &run->phase_input)
        device_source = yvex_cuda_activation_pointer(run->backend, run->job->device_input);
    resident = source && !device_source && !generated
        ? yvex_backend_state_residency_resolve(run->backend, source, bytes, &resident_address)
        : YVEX_BACKEND_RESIDENT_MISS;
    if (resident == YVEX_BACKEND_RESIDENT_INVALID)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            YVEX_ERR_STATE, "persistent device state mapping is invalid");
    if (resident == YVEX_BACKEND_RESIDENT_HIT)
        device_source = (CUdeviceptr)resident_address;
    if (source && !device_source) {
        rc = run->ops->account_transfer(
            count, width, &run->h2d_bytes, stage, run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    if (!device_source) {
        if (planned_source)
            stable_source = planned_source;
        else if (source)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
                YVEX_ERR_STATE, "CUDA graph input has no stable pinned staging slot");
    }
    if (captured && !yvex_cuda_capture_active(run->backend)) {
        rc = run->ops->allocate(&run->resources, target, bytes, NULL, 0,
                                           stage, run->failure, run->err);
        if (rc != YVEX_OK || run->resources.prepare_only) return rc;
        if (run->initializer_count >= YVEX_CUDA_ATTN_INITIALIZERS)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage,
                YVEX_CUDA_ATTN_INITIALIZERS, run->initializer_count,
                YVEX_ERR_BOUNDS, "CUDA graph initializer inventory is full");
        run->initializers[run->initializer_count++] =
            (attn_initializer){
                target, device_source ? NULL : stable_source, device_source,
                bytes, device_source ? 0 : zero, stage};
        return YVEX_OK;
    }
    rc = run->ops->allocate(&run->resources, target, bytes,
                            device_source ? NULL : stable_source,
                            device_source ? 0 : zero, stage,
                            run->failure, run->err);
    if (rc != YVEX_OK || run->resources.prepare_only || !device_source)
        return rc;
    {
        CUstream stream = yvex_cuda_launch_stream(run->backend);
        CUresult copy = stream && run->state->driver.cuMemcpyDtoDAsync_v2
                            ? run->state->driver.cuMemcpyDtoDAsync_v2(
                                  *target, device_source, bytes, stream)
                            : !stream
                                  ? run->state->driver.cuMemcpyDtoD_v2(*target, device_source, bytes)
                                  : (CUresult)1;
        rc = yvex_cuda_status(&run->state->driver, copy, stage, run->err);
        return rc == YVEX_OK ? attn_account_d2d(run, bytes, stage) : rc;
    }
}
typedef struct {
    size_t target_offset;
    attn_extent_kind extent;
    attn_source_kind source;
    size_t width;
    int zero, envelope_only;
    const char *stage;
} attn_allocation_spec;
#define A(field, extent, source, width, zero, envelope, label) \
    {offsetof(attn_run, field), extent, source, width, zero, envelope, \
     "cuda.deepseek_attention.alloc." label}
static const attn_allocation_spec attn_allocations[] = {
    A(device_status, EXT_ONE, SRC_NONE, sizeof(int), 1, 0, "status"),
    A(phase_input, EXT_INPUT, SRC_INPUT, 0u, 0, 0, "input"),
    A(phase_core_input, EXT_CORE, SRC_NONE, 0u, 1, 1, "core_input"),
    A(phase_q_low, EXT_Q_LOW, SRC_NONE, 0u, 1, 0, "q_low"),
    A(phase_query, EXT_QUERY, SRC_NONE, 0u, 1, 0, "query"),
    A(sinks, EXT_QUERY_HEADS, SRC_NONE, 0u, 1, 0, "sinks"),
    A(phase_attention, EXT_QUERY, SRC_NONE, 0u, 1, 0, "attention"),
    A(phase_low, EXT_LOW, SRC_NONE, 0u, 1, 0, "output_low"),
    A(phase_output, EXT_CORE, SRC_NONE, 0u, 1, 0, "output"),
    A(phase_envelope_output, EXT_ENVELOPE, SRC_NONE, 0u, 1, 1, "envelope_output"),
    A(phase_mhc_mix, EXT_MHC_MIX, SRC_NONE, 0u, 1, 1, "mhc_mix"),
    A(phase_mhc_scale, EXT_MHC_SCALE, SRC_NONE, 0u, 1, 1, "mhc_scale"),
    A(phase_mhc_base, EXT_MHC_MIX, SRC_NONE, 0u, 1, 1, "mhc_base"),
    A(phase_mhc_post, EXT_MHC_POST, SRC_NONE, 0u, 1, 1, "mhc_post"),
    A(phase_mhc_combination, EXT_MHC_COMBINATION, SRC_NONE, 0u, 1, 1, "mhc_combination"),
    A(phase_selected_count, EXT_TOKENS, SRC_NONE, sizeof(unsigned long long), 1, 0, "selected_count"),
    A(phase_valid_count, EXT_TOKENS, SRC_NONE, sizeof(unsigned long long), 1, 0, "valid_count"),
    A(phase_local, EXT_LOCAL, SRC_LOCAL, 0u, 0, 0, "local_history"),
    A(phase_local_positions, EXT_LOCAL_POSITIONS, SRC_LOCAL_POSITIONS,
      sizeof(uint64_t), 0, 0, "local_positions"),
    A(phase_compressed, EXT_COMPRESSED, SRC_COMPRESSED, 0u, 0, 0, "compressed_history"),
    A(phase_compressed_positions, EXT_COMPRESSED_POSITIONS,
      SRC_COMPRESSED_POSITIONS, sizeof(uint64_t), 0, 0, "compressed_positions"),
    A(phase_indexer, EXT_INDEXER, SRC_INDEXER, 0u, 0, 0, "indexer_history"),
    A(phase_indexer_positions, EXT_INDEXER_POSITIONS, SRC_INDEXER_POSITIONS,
      sizeof(uint64_t), 0, 0, "indexer_positions"),
    A(phase_main_kv, EXT_MAIN_STATE, SRC_MAIN_KV, 0u, 0, 0, "main_kv_state"),
    A(phase_main_score, EXT_MAIN_STATE, SRC_MAIN_SCORE, 0u, 0, 0, "main_score_state"),
    A(phase_index_kv, EXT_INDEX_STATE, SRC_INDEX_KV, 0u, 0, 0, "index_kv_state"),
    A(phase_index_score, EXT_INDEX_STATE, SRC_INDEX_SCORE, 0u, 0, 0, "index_score_state"),
    A(rolling[ROLL_MAIN].kv, EXT_MAIN_WIDTH, SRC_NONE, 0u, 1, 0, "main_kv"),
    A(rolling[ROLL_MAIN].score, EXT_MAIN_WIDTH, SRC_NONE, 0u, 1, 0, "main_score"),
    A(rolling[ROLL_MAIN].ape, EXT_MAIN_WIDTH, SRC_NONE, 0u, 1, 0, "main_ape"),
    A(rolling[ROLL_MAIN].scratch_value, EXT_MAIN_HEAD, SRC_NONE, 0u, 1, 0, "main_value"),
    A(rolling[ROLL_MAIN].positions, EXT_MAIN_POSITION, SRC_NONE, sizeof(unsigned long long), 1, 0, "main_position"),
    A(rolling[ROLL_INDEX].kv, EXT_INDEX_WIDTH, SRC_NONE, 0u, 1, 0, "index_kv"),
    A(rolling[ROLL_INDEX].score, EXT_INDEX_WIDTH, SRC_NONE, 0u, 1, 0, "index_score"),
    A(rolling[ROLL_INDEX].ape, EXT_INDEX_WIDTH, SRC_NONE, 0u, 1, 0, "index_ape"),
    A(rolling[ROLL_INDEX].scratch_value, EXT_INDEX_HEAD, SRC_NONE, 0u, 1, 0, "index_value"),
    A(rolling[ROLL_INDEX].positions, EXT_INDEX_POSITION, SRC_NONE, sizeof(unsigned long long), 1, 0, "index_position"),
    A(phase_index_query, EXT_INDEX_QUERY, SRC_NONE, 0u, 1, 0, "index_query"),
    A(phase_index_weights, EXT_INDEX_WEIGHTS, SRC_NONE, 0u, 1, 0, "index_weights"),
    A(phase_selected, EXT_SELECTED, SRC_NONE, sizeof(unsigned long long), 1, 0, "selected"),
    A(phase_selected_positions, EXT_SELECTED, SRC_NONE, sizeof(unsigned long long), 1, 0, "selected_positions"),
    A(topk_scores, EXT_CANDIDATES, SRC_NONE, 0u, 1, 0, "topk_scores"),
    A(valid_indexes, EXT_CANDIDATES, SRC_NONE, sizeof(unsigned long long), 1, 0, "valid_indexes")
};
#undef A
static int attn_extent(const attn_run *run,
                                 attn_extent_kind kind,
                                 unsigned long long *out) {
    unsigned long long left = run->job->token_count, right = 1ull;
    switch (kind) {
    case EXT_ONE: left = 1ull; break;
    case EXT_INPUT: right = run->input_extent; break;
    case EXT_CORE: right = run->job->hidden_width; break;
    case EXT_RAW_KV: right = run->job->kv_width; break;
    case EXT_Q_LOW: right = run->job->q_rank; break;
    case EXT_QUERY: right = run->query_width; break;
    case EXT_LOW: right = run->low_count; break;
    case EXT_ENVELOPE: right = run->job->residual_expanded_width; break;
    case EXT_MHC_MIX: right = run->job->mhc_mixing_rows; break;
    case EXT_MHC_SCALE: right = 3ull; break;
    case EXT_MHC_POST: right = run->job->residual_stream_count; break;
    case EXT_MHC_COMBINATION:
        if (!yvex_core_u64_mul(run->job->residual_stream_count,
                               run->job->residual_stream_count, &right)) return 0;
        break;
    case EXT_TOKENS: break;
    case EXT_LOCAL: *out = run->local_storage_extent; return 1;
    case EXT_LOCAL_POSITIONS: *out = run->phase_local_storage_count; return 1;
    case EXT_COMPRESSED: *out = run->compressed_storage_extent; return 1;
    case EXT_COMPRESSED_POSITIONS: *out = run->phase_compressed_storage_count; return 1;
    case EXT_INDEXER: *out = run->indexer_storage_extent; return 1;
    case EXT_INDEXER_POSITIONS: *out = run->phase_indexer_storage_count; return 1;
    case EXT_MAIN_STATE:
        if (left == ULLONG_MAX) return 0;
        ++left; right = run->rolling[ROLL_MAIN].extent; break;
    case EXT_INDEX_STATE:
        if (left == ULLONG_MAX) return 0;
        ++left; right = run->rolling[ROLL_INDEX].extent; break;
    case EXT_INDEX_QUERY: right = run->index_query_extent; break;
    case EXT_INDEX_WEIGHTS: right = run->job->indexer_heads; break;
    case EXT_SELECTED: right = run->topk_capacity; break;
    case EXT_CANDIDATES: left = run->candidate_capacity; break;
    case EXT_QUERY_HEADS: left = run->job->query_heads; break;
    case EXT_PHASE_COMPRESSED:
        left = run->phase_compressed_count; right = run->job->head_dimension; break;
    case EXT_PHASE_COMPRESSED_POSITIONS: left = run->phase_compressed_count; break;
    case EXT_PHASE_INDEXER:
        left = run->phase_indexer_count; right = run->job->indexer_head_dimension; break;
    case EXT_PHASE_INDEXER_POSITIONS: left = run->phase_indexer_count; break;
    case EXT_MAIN_ROLLING: left = run->rolling[ROLL_MAIN].extent; break;
    case EXT_INDEX_ROLLING: left = run->rolling[ROLL_INDEX].extent; break;
    case EXT_MAIN_PUBLICATION:
        left = run->rolling[ROLL_MAIN].extent;
        right = run->job->retain_prefix_checkpoints ? run->job->token_count : 1ull;
        break;
    case EXT_INDEX_PUBLICATION:
        left = run->rolling[ROLL_INDEX].extent;
        right = run->job->retain_prefix_checkpoints ? run->job->token_count : 1ull;
        break;
    case EXT_LOCAL_USED: left = run->local_extent; break;
    case EXT_COMPRESSED_USED: left = run->compressed_extent; break;
    case EXT_INDEXER_USED: left = run->history_index_extent; break;
    case EXT_MAIN_WIDTH: left = run->job->main_rolling.state_width; break;
    case EXT_MAIN_HEAD: left = run->job->main_rolling.head_dimension; break;
    case EXT_INDEX_WIDTH: left = run->job->indexer_rolling.state_width; break;
    case EXT_INDEX_HEAD: left = run->job->indexer_rolling.head_dimension; break;
    case EXT_MAIN_POSITION: left = run->job->main_rolling.present ? 1ull : 0ull; break;
    case EXT_INDEX_POSITION: left = run->job->indexer_rolling.present ? 1ull : 0ull; break;
    default: return 0;
    }
    return yvex_core_u64_mul(left, right, out);
}
static const void *attn_allocation_source(
    const attn_run *run, attn_source_kind source) {
    const void *values[] = {
        NULL, run->job->input, run->job->local_kv, run->job->local_positions,
        run->job->compressed_kv, run->job->compressed_positions,
        run->job->indexer_kv, run->job->indexer_positions,
        run->job->main_rolling.kv_state, run->job->main_rolling.score_state,
        run->job->indexer_rolling.kv_state, run->job->indexer_rolling.score_state
    };
    return (unsigned int)source < sizeof(values) / sizeof(values[0]) ? values[source] : NULL;
}
/*
 * Execute one interval from the immutable device-allocation catalog.
 *
 * Admitted run and a bounded catalog interval. Acquires request-lifetime device ranges through the
 * resource owner.
 */
static int attn_allocations_execute(attn_run *run,
                                              size_t first, size_t count) {
    size_t index;
    for (index = first; index < first + count; ++index) {
        const attn_allocation_spec *spec = &attn_allocations[index];
        CUdeviceptr *target = (CUdeviceptr *)((unsigned char *)run + spec->target_offset);
        const void *source;
        unsigned long long extent;
        int rc;
        if (!attn_extent(run, spec->extent, &extent))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, spec->stage,
                ULLONG_MAX, 0ull, YVEX_ERR_BOUNDS,
                "CUDA attention request arena extent overflowed");
        if (spec->envelope_only &&
            run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
            extent = 0ull;
        source = attn_allocation_source(run, spec->source);
        if (extent && (rc = attn_alloc_values(
                run, target, extent, spec->width ? spec->width : sizeof(float),
                source, spec->zero, spec->stage)) != YVEX_OK)
            return rc;
    }
    return YVEX_OK;
}
static int attn_prepare(attn_run *run) {
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    unsigned long long phase_end, local_storage, compressed_end, indexer_end;
    unsigned long long device_input_elements, device_output_elements;
    int rc;
    if (!run->backend || !run->state)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.execute", 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "CUDA attention backend is required");
    rc = backend_dispatch_admit(run->backend, "cuda.deepseek_attention.execute", run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->validate_job(
        run->job, run->output, run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    run->phase_start_position = run->job->token_position;
    run->input_extent = run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                            ? run->job->residual_expanded_width : run->job->hidden_width;
    if ((run->job->device_input || run->job->device_output) &&
        (!yvex_core_u64_mul(run->job->token_count, run->input_extent,
                            &device_input_elements) ||
         !yvex_core_u64_mul(
             run->job->token_count,
             run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                 ? run->job->residual_expanded_width : run->job->hidden_width,
             &device_output_elements) ||
         !yvex_cuda_activation_views_valid(
             run->backend, run->job->device_input, device_input_elements,
             run->job->device_output, device_output_elements)))
        return attn_run_fail(run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.validate.device-activation", 1ull, 0ull,
            YVEX_ERR_FORMAT, "CUDA attention device activation views are incompatible");
    run->initial_local_count = run->job->local_count;
    run->initial_compressed_count = run->job->compressed_count;
    run->initial_indexer_count = run->job->indexer_count;
    run->local_capacity = attn_graph_mode(run)
                              ? run->state->attention_local_capacity
                              : run->job->sliding_window -
                                    (run->job->candidate_block_visible ? 0ull : 1ull);
    run->compressed_capacity = run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA
        ? 0ull : (attn_graph_mode(run)
                      ? run->state->attention_compressed_capacity
                      : (run->job->token_position + run->job->token_count) /
                            run->job->compression_ratio);
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
        !run->compressed_capacity)
        run->compressed_capacity = 1ull;
    run->indexer_capacity = run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
        ? (attn_graph_mode(run)
               ? run->state->attention_indexer_capacity : run->compressed_capacity)
        : 0ull;
    if (!yvex_core_u64_add(run->job->token_position, run->job->token_count, &phase_end) ||
        !yvex_core_u64_add(
            attn_graph_mode(run) ? run->local_capacity : run->initial_local_count,
            run->job->token_count, &local_storage))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.validate.phase", ULLONG_MAX,
            run->job->token_count, YVEX_ERR_BOUNDS,
            "CUDA attention request range overflowed");
    run->phase_compressed_count = run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA
        ? 0ull : phase_end / run->job->compression_ratio - run->initial_compressed_count;
    run->phase_indexer_count = run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
        ? run->phase_compressed_count : 0ull;
    if (!yvex_core_u64_add(run->initial_compressed_count, run->phase_compressed_count,
                           &compressed_end) ||
        !yvex_core_u64_add(run->initial_indexer_count, run->phase_indexer_count,
                           &indexer_end))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.validate.history_extent", ULLONG_MAX,
            run->phase_compressed_count, YVEX_ERR_BOUNDS,
            "CUDA attention request history extent overflowed");
    run->phase_local_storage_count = local_storage;
    run->phase_compressed_storage_count =
        run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA ? 0ull :
        attn_graph_mode(run) ? run->compressed_capacity :
        compressed_end ? compressed_end : 1ull;
    run->phase_indexer_storage_count =
        run->job->attention_class != YVEX_BACKEND_ATTENTION_CSA ? 0ull :
        attn_graph_mode(run) && run->indexer_capacity ? run->indexer_capacity :
        indexer_end ? indexer_end : 1ull;
    if (run->initial_local_count > run->local_capacity)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.capacity.local", run->local_capacity,
            run->initial_local_count, YVEX_ERR_BOUNDS,
            "CUDA attention local history exceeds the selected execution shape");
    if (compressed_end > run->compressed_capacity)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.capacity.compressed", run->compressed_capacity,
            compressed_end, YVEX_ERR_BOUNDS,
            "CUDA attention compressed history exceeds the selected execution shape");
    if (indexer_end > run->indexer_capacity)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.capacity.indexer", run->indexer_capacity,
            indexer_end, YVEX_ERR_BOUNDS,
            "CUDA attention indexer history exceeds the selected execution shape");
    {
        struct {
            unsigned long long left, right, *result;
        } extents[] = {
            {run->job->query_heads, run->job->head_dimension, &run->query_width},
            {run->job->output_groups, run->job->output_rank, &run->low_count},
            {run->job->local_count, run->job->local_stride, &run->local_extent},
            {run->job->compressed_count, run->job->compressed_stride,
             &run->compressed_extent},
            {run->job->indexer_count, run->job->indexer_stride,
             &run->history_index_extent},
            {run->phase_local_storage_count, run->job->local_stride,
             &run->local_storage_extent},
            {run->phase_compressed_storage_count, run->job->compressed_stride,
             &run->compressed_storage_extent},
            {run->phase_indexer_storage_count, run->job->indexer_stride,
             &run->indexer_storage_extent},
            {run->job->indexer_heads, run->job->indexer_head_dimension,
             &run->index_query_extent}
        };
        size_t index;
        for (index = 0u; index < sizeof(extents) / sizeof(extents[0]); ++index)
            if (!yvex_core_u64_mul(
                    extents[index].left, extents[index].right,
                    extents[index].result))
                return attn_run_fail(
                    run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                    "cuda.deepseek_attention.validate.extent", ULLONG_MAX, 0ull,
                    YVEX_ERR_BOUNDS,
                    "CUDA attention logical extent overflowed");
    }
    rc = attn_validate_derived(run);
    if (rc != YVEX_OK) return rc;
    if (!attn_stage_layout(run, NULL, &run->host_stage_bytes) ||
        !run->host_stage_bytes)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.validate.host_layout", ULLONG_MAX, 0ull,
            YVEX_ERR_BOUNDS,
            "CUDA attention host staging layout is not representable");
    if (run->host_stage_bytes > run->job->max_host_bytes) {
        rc = attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.validate.host_budget",
            run->job->max_host_bytes, run->host_stage_bytes,
            YVEX_ERR_NOMEM,
            "CUDA attention host staging exceeds its sealed workspace recipe");
        yvex_error_setf(
            run->err, YVEX_ERR_NOMEM,
            "cuda.deepseek_attention.validate.host_budget",
            "CUDA attention needs %zu host-staging bytes but the sealed recipe provides %llu",
            run->host_stage_bytes, run->job->max_host_bytes);
        return rc;
    }
    rc = attn_cancel(
        run, "cuda.deepseek_attention.cancel.before_dispatch", 0);
    if (rc != YVEX_OK) return rc;
    rc = yvex_cuda_require_capability(
        run->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
        "cuda.deepseek_attention.capability", run->err);
    if (rc != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.deepseek_attention.capability", 1ull, 0ull,
            (yvex_status)rc,
            "CUDA DeepSeek attention kernel bundle is not admitted");
    rc = injected && strcmp(injected, "context") == 0
        ? YVEX_ERR_BACKEND
        : yvex_cuda_set_current(
              run->backend, "cuda.deepseek_attention.context", run->err);
    if (rc != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.deepseek_attention.context", 1ull, 0ull,
            (yvex_status)rc, "CUDA attention context activation failed");
    run->resources.backend = run->backend;
    run->resources.state = run->state;
    run->resources.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    run->resources.budget = run->job->max_device_bytes;
    /* Full evidence trades the production activation codec for canonical-order
     * arithmetic so the independent oracle can compare exact stage values. */
    run->resources.forensic_numeric = run->job->evidence_level == 3u;
    return YVEX_OK;
}
static int attn_allocate_base(attn_run *run) {
    const size_t allocation_count =
        sizeof(attn_allocations) / sizeof(attn_allocations[0]);
    unsigned int slot;
    int rc;
    rc = attn_allocations_execute(run, 0u, 1u);
    if (rc != YVEX_OK) return rc;
    for (slot = 0u; slot < YVEX_BACKEND_ATTENTION_WEIGHT_COUNT; ++slot) {
        const yvex_backend_attention_weight *host = &run->job->weights[slot];
        unsigned long long resident_address = 0ull;
        int resident;
        if (!host->present) continue;
        resident = yvex_backend_resident_resolve(
            run->backend, host->encoded, host->encoded_bytes, &resident_address);
        if (resident == YVEX_BACKEND_RESIDENT_HIT) {
            run->weight[slot] = (CUdeviceptr)resident_address;
            continue;
        }
        if (resident == YVEX_BACKEND_RESIDENT_INVALID ||
            run->backend->resident_host_base)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_COPY,
                "cuda.deepseek_attention.resident.weight", host->encoded_bytes, 0ull,
                YVEX_ERR_STATE, "CUDA resident weight mapping is incomplete or invalid");
        rc = run->ops->allocate(&run->resources, &run->weight[slot],
                                     host->encoded_bytes, host->encoded, 0,
                                     "cuda.deepseek_attention.alloc.weight",
                                     run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    rc = attn_allocations_execute(run, 1u, allocation_count - 1u);
    if (rc != YVEX_OK) return rc;
    if (run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE)
        run->phase_core_input = run->phase_input;
    run->phase_raw_kv = run->phase_local +
        run->initial_local_count * run->job->local_stride * sizeof(float);
    run->phase_new_compressed = run->phase_compressed +
        run->initial_compressed_count * run->job->compressed_stride * sizeof(float);
    run->phase_new_compressed_positions = run->phase_compressed_positions +
        run->initial_compressed_count * sizeof(unsigned long long);
    run->phase_new_indexer = run->phase_indexer +
        run->initial_indexer_count * run->job->indexer_stride * sizeof(float);
    run->phase_new_indexer_positions = run->phase_indexer_positions +
        run->initial_indexer_count * sizeof(unsigned long long);
    return YVEX_OK;
}
/*
 * Bind one token ordinal to disjoint request arenas and prior device state.
 *
 * Updates only launch parameters; all numerical dependencies remain device-resident.
 */
static void attn_phase_bind(attn_run *run, unsigned long long ordinal) {
    yvex_backend_attention_job *job = &run->request;
    unsigned long long position = run->phase_start_position + ordinal;
    unsigned long long local_before = run->initial_local_count + ordinal;
    unsigned long long local_count = position < job->sliding_window - 1ull
                                         ? position : job->sliding_window - 1ull;
    unsigned long long local_offset = local_before - local_count;
    unsigned long long compressed_count = job->attention_class == YVEX_BACKEND_ATTENTION_SWA
                                              ? 0ull : position / job->compression_ratio;
    unsigned long long emit = job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
                              (position + 1ull) % job->compression_ratio == 0ull;
    unsigned long long float_size = sizeof(float), u64_size = sizeof(unsigned long long);
    unsigned long long main_extent = run->rolling[ROLL_MAIN].extent;
    unsigned long long index_extent = run->rolling[ROLL_INDEX].extent;
    run->ordinal = ordinal;
    if (job->candidate_block_visible) {
        local_count = run->initial_local_count + job->token_count;
        local_offset = 0ull;
    }
    job->token_position = position;
    job->local_count = local_count;
    job->compressed_count = compressed_count;
    job->indexer_count = job->attention_class == YVEX_BACKEND_ATTENTION_CSA
                             ? compressed_count : 0ull;
    run->input = run->phase_input + ordinal * run->input_extent * float_size;
    run->core_input = job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
        ? run->phase_core_input + ordinal * job->hidden_width * float_size : run->input;
    run->q_low = run->phase_q_low + ordinal * job->q_rank * float_size;
    run->query = run->phase_query + ordinal * run->query_width * float_size;
    run->raw_kv = run->phase_raw_kv + ordinal * job->kv_width * float_size;
    run->attention = run->phase_attention + ordinal * run->query_width * float_size;
    run->low = run->phase_low + ordinal * run->low_count * float_size;
    run->final_output = run->phase_output + ordinal * job->hidden_width * float_size;
    run->envelope_output = run->phase_envelope_output +
        ordinal * job->residual_expanded_width * float_size;
    run->mhc_mix = run->phase_mhc_mix + ordinal * job->mhc_mixing_rows * float_size;
    run->mhc_scale = run->phase_mhc_scale + ordinal * 3ull * float_size;
    run->mhc_base = run->phase_mhc_base + ordinal * job->mhc_mixing_rows * float_size;
    run->mhc_post = run->phase_mhc_post + ordinal * job->residual_stream_count * float_size;
    run->mhc_combination = run->phase_mhc_combination +
        ordinal * job->residual_stream_count * job->residual_stream_count * float_size;
    run->index_query = run->phase_index_query + ordinal * run->index_query_extent * float_size;
    run->index_weights = run->phase_index_weights + ordinal * job->indexer_heads * float_size;
    run->selected = run->phase_selected + ordinal * run->topk_capacity * u64_size;
    run->selected_positions = run->phase_selected_positions +
        ordinal * run->topk_capacity * u64_size;
    run->selected_count = run->phase_selected_count + ordinal * u64_size;
    run->valid_count = run->phase_valid_count + ordinal * u64_size;
    run->history_local = run->phase_local + local_offset * job->local_stride * float_size;
    run->history_local_positions = run->phase_local_positions + local_offset * u64_size;
    run->history_compressed = run->phase_compressed;
    run->history_compressed_positions = run->phase_compressed_positions;
    run->history_indexer = run->phase_indexer;
    run->history_indexer_positions = run->phase_indexer_positions;
    if (job->attention_class == YVEX_BACKEND_ATTENTION_SWA) return;
    job->main_rolling.next_token_position = position;
    job->main_rolling.cursor = position % job->compression_ratio;
    job->main_rolling.current_fill = job->main_rolling.cursor;
    job->main_rolling.previous_fill = job->main_rolling.overlap &&
        position >= job->compression_ratio ? job->compression_ratio : 0ull;
    run->rolling[ROLL_MAIN].before_kv =
        run->phase_main_kv + ordinal * main_extent * float_size;
    run->rolling[ROLL_MAIN].before_score =
        run->phase_main_score + ordinal * main_extent * float_size;
    run->rolling[ROLL_MAIN].after_kv =
        run->phase_main_kv + (ordinal + 1ull) * main_extent * float_size;
    run->rolling[ROLL_MAIN].after_score =
        run->phase_main_score + (ordinal + 1ull) * main_extent * float_size;
    run->rolling[ROLL_MAIN].value =
        run->phase_compressed + compressed_count * job->compressed_stride * float_size;
    run->rolling[ROLL_MAIN].positions =
        run->phase_compressed_positions + compressed_count * u64_size;
    run->rolling[ROLL_MAIN].value_count = emit;
    run->rolling[ROLL_MAIN].value_extent = emit * job->head_dimension;
    run->emission_position = emit ? position + 1ull - job->compression_ratio : 0ull;
    if (job->attention_class != YVEX_BACKEND_ATTENTION_CSA) return;
    job->indexer_rolling.next_token_position = position;
    job->indexer_rolling.cursor = position % job->compression_ratio;
    job->indexer_rolling.current_fill = job->indexer_rolling.cursor;
    job->indexer_rolling.previous_fill = position >= job->compression_ratio
                                            ? job->compression_ratio : 0ull;
    run->rolling[ROLL_INDEX].before_kv =
        run->phase_index_kv + ordinal * index_extent * float_size;
    run->rolling[ROLL_INDEX].before_score =
        run->phase_index_score + ordinal * index_extent * float_size;
    run->rolling[ROLL_INDEX].after_kv =
        run->phase_index_kv + (ordinal + 1ull) * index_extent * float_size;
    run->rolling[ROLL_INDEX].after_score =
        run->phase_index_score + (ordinal + 1ull) * index_extent * float_size;
    run->rolling[ROLL_INDEX].value =
        run->phase_indexer + compressed_count * job->indexer_stride * float_size;
    run->rolling[ROLL_INDEX].positions =
        run->phase_indexer_positions + compressed_count * u64_size;
    run->rolling[ROLL_INDEX].value_count = emit;
    run->rolling[ROLL_INDEX].value_extent = emit * job->indexer_head_dimension;
}
static int attn_envelope_pre(attn_run *run) {
    unsigned long long streams = run->job->residual_stream_count, width = run->job->residual_stream_width;
    unsigned int shared_bytes;
    int rc;
    if (run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
        return YVEX_OK;
    if (streams > UINT_MAX / sizeof(double) - 1ull - YVEX_CUDA_ATTN_BLOCK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.mhc_pre", UINT_MAX, streams,
            YVEX_ERR_BOUNDS, "CUDA mHC shared geometry exceeds launch bounds");
    shared_bytes = (unsigned int)((streams + 1ull + YVEX_CUDA_ATTN_BLOCK) *
                                  sizeof(double));
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION], 0ull,
        run->job->mhc_mixing_rows, 1ull, run->input, run->mhc_mix, 0,
        run->device_status, "cuda.deepseek_attention.mhc_function",
        run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->decode(
            &run->resources,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE], 0ull, 3ull,
            run->mhc_scale, run->device_status,
            "cuda.deepseek_attention.mhc_scale", run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->decode(
            &run->resources,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE], 0ull,
            run->job->mhc_mixing_rows, run->mhc_base, run->device_status,
            "cuda.deepseek_attention.mhc_base", run->failure, run->err);
    if (rc == YVEX_OK) {
        unsigned long long one = 1ull;
        void *params[] = {
            &run->input, &run->mhc_mix, &run->mhc_scale, &run->mhc_base,
            &streams, &width, (void *)&run->job->mhc_mixing_rows,
            (void *)&run->job->mhc_sinkhorn_iterations,
            (void *)&run->job->rms_epsilon, (void *)&run->job->mhc_epsilon,
            (void *)&run->job->mhc_residual_post_multiplier, &run->core_input,
            &run->mhc_post, &run->mhc_combination, &one, &run->device_status
        };
        rc = run->ops->launch(
            &run->resources, run->state->deepseek_mhc_pre_function, 1u,
            YVEX_CUDA_ATTN_BLOCK, shared_bytes, params,
            "cuda.deepseek_attention.mhc_pre", run->failure, run->err);
    }
    if (rc == YVEX_OK)
        rc = run->ops->weighted_norm(
            &run->resources, run->core_input, run->job->hidden_width, 1ull,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM],
            run->job->rms_epsilon, run->device_status,
            "cuda.deepseek_attention.input_norm", run->failure, run->err);
    return rc;
}
static int attn_project(attn_run *run) {
    int rc = YVEX_OK;
    if (run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE)
        rc = run->ops->round_bf16(
            &run->resources, run->core_input, run->job->hidden_width,
            run->device_status, "cuda.deepseek_attention.input_round",
            run->failure, run->err);
    if (rc == YVEX_OK) rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A], 0ull,
        run->job->q_rank, 1ull, run->core_input, run->q_low, 1, run->device_status,
        "cuda.deepseek_attention.q_a", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->weighted_norm(
        &run->resources, run->q_low, run->job->q_rank, 1ull,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM],
        run->job->rms_epsilon, run->device_status,
        "cuda.deepseek_attention.q_norm", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_B],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_B], 0ull,
        run->query_width, 1ull, run->q_low, run->query, 1, run->device_status,
        "cuda.deepseek_attention.q_b", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_KV],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_KV], 0ull,
        run->job->kv_width, 1ull, run->core_input, run->raw_kv, 1, run->device_status,
        "cuda.deepseek_attention.kv", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->weighted_norm(
        &run->resources, run->raw_kv, run->job->kv_width, 1ull,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM],
        run->job->rms_epsilon, run->device_status,
        "cuda.deepseek_attention.kv_norm", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->unit_norm(
        &run->resources, run->query, run->job->query_heads,
        run->job->head_dimension, run->job->rms_epsilon, run->device_status,
        "cuda.deepseek_attention.query_unit_norm", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->query, run->job->query_heads,
        run->job->head_dimension, run->job->token_position,
        &run->job->position, 0, run->device_status,
        "cuda.deepseek_attention.query_rope", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->raw_kv, 1ull, run->job->kv_width,
        run->job->token_position, &run->job->position, 0,
        run->device_status, "cuda.deepseek_attention.kv_rope",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    if (run->job->kv_width > run->job->position.rope_dimensions) {
        rc = run->ops->activation(
            &run->resources, run->raw_kv, 1ull,
            run->job->kv_width - run->job->position.rope_dimensions,
            &run->job->attention_kv_activation, run->device_status,
            "cuda.deepseek_attention.kv_activation", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    return run->ops->decode(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_SINKS],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_SINKS], 0ull,
        run->job->query_heads, run->sinks, run->device_status,
        "cuda.deepseek_attention.sinks", run->failure, run->err);
}
static int attn_rolling_execute(attn_run *run, unsigned int kind) {
    const int index = kind == ROLL_INDEX;
    const yvex_backend_attention_rolling *rolling = index
        ? &run->job->indexer_rolling : &run->job->main_rolling;
    attn_rolling_run *device = &run->rolling[kind];
    yvex_backend_attention_weight_slot base = index
        ? YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_KV
        : YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_KV;
    const yvex_backend_attention_activation *activation = index
        ? &run->job->compressor_rotated_activation
        : &run->job->compressor_activation;
    const char *stage = index ? "cuda.deepseek_attention.index_rolling"
                              : "cuda.deepseek_attention.main_rolling";
    unsigned long long activation_width = rolling->head_dimension;
    int emit, rc;
    rc = run->ops->matvec(
        &run->resources, &run->job->weights[base], run->weight[base], 0ull,
        rolling->state_width, 1ull, run->core_input, device->kv, 0, run->device_status,
        stage, run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->matvec(
            &run->resources, &run->job->weights[base + 1],
            run->weight[base + 1], 0ull, rolling->state_width, 1ull, run->core_input,
            device->score, 0, run->device_status, stage, run->failure,
            run->err);
    if (rc == YVEX_OK)
        rc = run->ops->decode(
            &run->resources, &run->job->weights[base + 2],
            run->weight[base + 2], run->job->token_position % rolling->ratio,
            rolling->state_width, device->ape, run->device_status, stage,
            run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    emit = device->value_count != 0ull;
    if (!emit && attn_graph_mode(run)) device->value = device->scratch_value;
    {
        void *params[] = {
            &device->before_kv, &device->before_score, &device->kv,
            &device->score, &device->ape, &device->after_kv,
            &device->after_score, &device->value, (void *)&rolling->ratio,
            (void *)&rolling->head_dimension, (void *)&rolling->state_width,
            (void *)&rolling->state_slots, (void *)&rolling->cursor,
            (void *)&rolling->overlap, &emit, &run->device_status
        };
        rc = run->ops->launch(
            &run->resources, run->state->deepseek_rolling_function, 1u,
            YVEX_CUDA_ATTN_BLOCK, 0u, params, stage, run->failure, run->err);
    }
    if (rc != YVEX_OK || (!emit && !attn_graph_mode(run))) return rc;
    rc = run->ops->weighted_norm(
        &run->resources, device->value, rolling->head_dimension, 1ull,
        &run->job->weights[base + 3], run->weight[base + 3],
        run->job->rms_epsilon, run->device_status, stage, run->failure,
        run->err);
    if (rc == YVEX_OK)
        rc = run->ops->rope(
            &run->resources, device->value, 1ull, rolling->head_dimension,
            run->emission_position, &run->job->position, 0,
            run->device_status, stage, run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    if (!index) {
        if (activation_width <= run->job->position.rope_dimensions)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
                stage, run->job->position.rope_dimensions + 1ull,
                activation_width, YVEX_ERR_FORMAT,
                "CUDA main compressor requires non-RoPE dimensions");
        activation_width -= run->job->position.rope_dimensions;
    }
    return run->ops->activation(
        &run->resources, device->value, 1ull, activation_width, activation,
        run->device_status, stage, run->failure, run->err);
}
static int attn_index_topk(attn_run *run) {
    int rc;
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q], 0ull,
        run->index_query_extent, 1ull, run->q_low, run->index_query,
        1, run->device_status, "cuda.deepseek_attention.index_query",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION], 0ull,
        run->job->indexer_heads, 1ull, run->core_input, run->index_weights,
        1, run->device_status, "cuda.deepseek_attention.index_weights",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->index_query, run->job->indexer_heads,
        run->job->indexer_head_dimension, run->job->token_position,
        &run->job->position, 0, run->device_status,
        "cuda.deepseek_attention.index_query_rope",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->activation(
        &run->resources, run->index_query, run->job->indexer_heads,
        run->job->indexer_head_dimension, &run->job->indexer_query_activation,
        run->device_status, "cuda.deepseek_attention.index_query_activation",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    {
        void *params[] = {
            &run->index_query, &run->index_weights, &run->history_indexer,
            &run->history_indexer_positions, (void *)&run->job->indexer_count,
            (void *)&run->job->indexer_stride,
            &run->rolling[ROLL_INDEX].value,
            &run->rolling[ROLL_INDEX].positions,
            &run->rolling[ROLL_INDEX].value_count,
            (void *)&run->job->indexer_head_dimension,
            (void *)&run->job->indexer_heads,
            (void *)&run->job->indexer_head_dimension,
            (void *)&run->job->compression_ratio,
            (void *)&run->job->token_position, (void *)&run->job->indexer_topk,
            &run->selected, &run->selected_positions, &run->selected_count,
            &run->valid_count, &run->topk_scores, &run->valid_indexes,
            &run->device_status
        };
        return run->ops->launch(
            &run->resources, run->state->deepseek_topk_function, 1u,
            YVEX_CUDA_ATTN_BLOCK, YVEX_CUDA_ATTN_BLOCK * sizeof(double),
            params, "cuda.deepseek_attention.topk", run->failure, run->err);
    }
}
static int attn_compress(attn_run *run) {
    int rc;
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA) return YVEX_OK;
    rc = attn_rolling_execute(run, ROLL_MAIN);
    if (rc != YVEX_OK ||
        run->job->attention_class != YVEX_BACKEND_ATTENTION_CSA) return rc;
    rc = attn_rolling_execute(run, ROLL_INDEX);
    return rc == YVEX_OK ? attn_index_topk(run) : rc;
}
static int attn_reduce(attn_run *run) {
    unsigned long long group;
    unsigned int attention_class = (unsigned int)run->job->attention_class;
    int rc;
    {
        void *params[] = {
            &run->query, &run->history_local, &run->history_local_positions,
            (void *)&run->job->local_count, (void *)&run->job->local_stride,
            &run->raw_kv, (void *)&run->job->kv_width,
            &run->history_compressed, &run->history_compressed_positions,
            (void *)&run->job->compressed_count,
            (void *)&run->job->compressed_stride,
            &run->rolling[ROLL_MAIN].value,
            &run->rolling[ROLL_MAIN].positions,
            &run->rolling[ROLL_MAIN].value_count,
            (void *)&run->job->head_dimension, &run->selected,
            &run->selected_count, &run->sinks,
            (void *)&run->job->query_heads, (void *)&run->job->head_dimension,
            (void *)&run->job->sliding_window,
            (void *)&run->job->compression_ratio, &attention_class,
            (void *)&run->job->token_position,
            (void *)&run->job->candidate_block_visible, &run->attention,
            &run->device_status
        };
        rc = run->ops->launch(
            &run->resources, run->state->deepseek_reduce_function,
            (unsigned int)run->job->query_heads, YVEX_CUDA_ATTN_BLOCK,
            YVEX_CUDA_ATTN_BLOCK * sizeof(double), params,
            "cuda.deepseek_attention.reduce", run->failure, run->err);
    }
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->attention, run->job->query_heads,
        run->job->head_dimension, run->job->token_position,
        &run->job->position, 1, run->device_status,
        "cuda.deepseek_attention.output_inverse_rope",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    for (group = 0ull; group < run->job->output_groups; ++group) {
        CUdeviceptr group_input =
            run->attention +
            group * run->job->output_group_input_width * sizeof(float);
        CUdeviceptr group_output =
            run->low + group * run->job->output_rank * sizeof(float);
        rc = run->ops->matvec(
            &run->resources,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A],
            group * run->job->output_rank, run->job->output_rank, 1ull,
            group_input, group_output, 1, run->device_status,
            "cuda.deepseek_attention.output_a", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    return run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B], 0ull,
        run->job->hidden_width, 1ull, run->low, run->final_output,
        1, run->device_status, "cuda.deepseek_attention.output_b",
        run->failure, run->err);
}
/* Apply optional mHC residual egress to completed core output without host materialization. */
static int attn_envelope_post(attn_run *run) {
    unsigned long long expanded = run->job->residual_expanded_width;
    unsigned long long streams = run->job->residual_stream_count;
    unsigned long long width = run->job->residual_stream_width;
    unsigned int grid;
    if (run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
        return YVEX_OK;
    if (!expanded || expanded > UINT_MAX * (unsigned long long)YVEX_CUDA_ATTN_BLOCK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.deepseek_attention.mhc_post", UINT_MAX, expanded,
            YVEX_ERR_BOUNDS, "CUDA attention mHC egress extent is invalid");
    grid = (unsigned int)((expanded + YVEX_CUDA_ATTN_BLOCK - 1ull) /
                          YVEX_CUDA_ATTN_BLOCK);
    {
        unsigned long long one = 1ull;
        void *params[] = {
            &run->final_output, &run->input, &run->mhc_post,
            &run->mhc_combination, &streams, &width,
            &run->envelope_output, &one, &run->device_status
        };
        return run->ops->launch(
            &run->resources, run->state->deepseek_mhc_post_function, grid,
            YVEX_CUDA_ATTN_BLOCK, 0u, params,
            "cuda.deepseek_attention.mhc_post", run->failure, run->err);
    }
}
typedef struct {
    attn_run *run; unsigned int first, last;
} attn_graph_piece;
static int (*const attn_kernel_stages[YVEX_CUDA_ATTENTION_STAGE_COUNT])(attn_run *) = {
        attn_envelope_pre, attn_project, attn_compress,
        attn_reduce, attn_envelope_post
};
static int attn_initializers_enqueue(attn_run *run) {
    size_t i;
    int rc;
    for (i = 0u; i < run->initializer_count; ++i) {
        attn_initializer *init = &run->initializers[i];
        if (init->device_source) {
            CUstream stream = yvex_cuda_launch_stream(run->backend);
            rc = stream && run->state->driver.cuMemcpyDtoDAsync_v2
                     ? yvex_cuda_status(&run->state->driver,
                                        run->state->driver.cuMemcpyDtoDAsync_v2(
                                            *init->device, init->device_source,
                                            init->bytes, stream),
                                        init->stage, run->err)
                     : YVEX_ERR_UNSUPPORTED;
            if (rc == YVEX_ERR_UNSUPPORTED)
                yvex_error_set(run->err, YVEX_ERR_UNSUPPORTED, init->stage,
                               "captured device-state copy is unavailable");
        } else {
            rc = run->ops->initialize(
                &run->resources, *init->device, init->bytes, init->source,
                init->zero, init->stage, run->failure, run->err);
        }
        if (rc != YVEX_OK) return rc;
    }
    return YVEX_OK;
}
static int attn_downloads_enqueue(attn_run *run) {
    size_t i;
    int rc = run->ops->download(
        &run->resources, run->staged_status, run->device_status,
        sizeof(*run->staged_status), "cuda.deepseek_attention.copy.status",
        run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->download(
            &run->resources, run->staged_selected_count, run->phase_selected_count,
            (size_t)run->job->token_count * sizeof(*run->staged_selected_count),
            "cuda.deepseek_attention.copy.selected_count", run->failure,
            run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->download(
            &run->resources, run->staged_candidate_count, run->phase_valid_count,
            (size_t)run->job->token_count * sizeof(*run->staged_candidate_count),
            "cuda.deepseek_attention.copy.valid_count", run->failure, run->err);
    for (i = 0u; rc == YVEX_OK && i < run->transfer_count; ++i) {
        attn_transfer *transfer = &run->transfers[i];
        size_t bytes;
        if (!yvex_cuda_work_checked_bytes(
                transfer->capacity, transfer->width, &bytes))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, transfer->stage,
                transfer->capacity, 0ull, YVEX_ERR_BOUNDS,
                "CUDA attention output staging extent overflowed");
        rc = run->ops->download(
            &run->resources, transfer->staged, *transfer->device, bytes,
            transfer->stage, run->failure, run->err);
    }
    return rc;
}
static int attn_graph_enqueue(void *opaque, int enqueue_kernels, yvex_error *err) {
    attn_graph_piece *piece = (attn_graph_piece *)opaque;
    unsigned long long token;
    unsigned int stage;
    int rc = YVEX_OK;
    (void)err;
    piece->run->resources.prepare_only = !enqueue_kernels;
    for (token = 0ull; rc == YVEX_OK && token < piece->run->job->token_count; ++token) {
        attn_phase_bind(piece->run, token);
        for (stage = piece->first; rc == YVEX_OK && stage < piece->last; ++stage)
            if (cuda_attention_piece_active(
                    piece->run->job->operation_scope,
                    piece->run->job->attention_class,
                    (yvex_cuda_attention_stage)stage))
                rc = attn_kernel_stages[stage](piece->run);
    }
    piece->run->resources.prepare_only = 0;
    return rc;
}
/* State promotion changes source addresses, so refresh copies on the graph stream before replay. */
static int attn_graph_prepare(void *opaque, yvex_error *err) {
    attn_graph_piece *piece = (attn_graph_piece *)opaque;
    (void)err;
    if (piece->first == 0u ||
        (piece->run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE &&
         piece->first == YVEX_CUDA_ATTENTION_STAGE_PROJECT))
        return attn_initializers_enqueue(piece->run);
    return YVEX_OK;
}
static int attn_graph_execute(attn_run *run, unsigned int first, unsigned int last) {
    attn_graph_piece piece = {run, first, last};
    yvex_backend_cuda_graph_info info;
    size_t initializer;
    char identity[160];
    int measure_device_time = run->job->evidence_level != 0u, rc;
    if (yvex_cuda_attention_graph_key(run->backend, run->job, first, last,
                                      identity, run->err) != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.deepseek_attention.graph.identity", 1ull, 0ull,
            YVEX_ERR_STATE, "CUDA attention graph compatibility identity is incomplete");
    rc = yvex_cuda_graph_execute(run->backend, identity, attn_graph_prepare, attn_graph_enqueue,
        &piece, measure_device_time ? YVEX_CUDA_GRAPH_EXECUTION_MEASURE_DEVICE_TIME :
            YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
                YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION, &info, run->err);
    if (rc != YVEX_OK && run->failure &&
        run->failure->code == YVEX_BACKEND_ATTENTION_FAILURE_NONE) {
        run->failure->code = YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH;
        run->failure->stage = "cuda.deepseek_attention.graph.execute";
        run->failure->expected = 1ull;
        run->failure->actual = 0ull;
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->device_execution_elapsed_ns,
                            info.last_device_elapsed_ns,
                            &run->device_execution_elapsed_ns)))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.graph.timing", ULLONG_MAX,
            info.last_device_elapsed_ns, YVEX_ERR_BOUNDS,
            "CUDA attention graph device timing overflowed");
    if (rc == YVEX_OK) {
        run->resources.launches += info.inventory.kernel_node_count;
        if (!info.completion_pending)
            run->stream_synchronizations += 1ull +
                (unsigned long long)(measure_device_time && run->state->timing_ready);
        if (first == 0u ||
            (run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE &&
             first == YVEX_CUDA_ATTENTION_STAGE_PROJECT))
            for (initializer = 0u; rc == YVEX_OK && initializer < run->initializer_count;
                 ++initializer)
                if (run->initializers[initializer].device_source)
                    rc = attn_account_d2d(run, run->initializers[initializer].bytes,
                                          run->initializers[initializer].stage);
    }
    return rc;
}
static int attn_numerical_execute(attn_run *run) {
    const unsigned int first_end = YVEX_CUDA_ATTENTION_STAGE_COMPRESS;
    const unsigned int last_begin = YVEX_CUDA_ATTENTION_STAGE_REDUCE;
    unsigned long long token;
    unsigned int stage;
    int measure_device_time = run->job->evidence_level != 0u, rc = YVEX_OK;
    if (!run->state->attention_graph_configured ||
        run->state->attention_mode == YVEX_BACKEND_CUDA_ATTENTION_EAGER) {
        if (measure_device_time) rc = yvex_cuda_timing(
                run->backend, yvex_cuda_launch_stream(run->backend), YVEX_CUDA_TIMING_BEGIN, NULL,
                "cuda.deepseek_attention.eager.timing.begin", run->err);
        for (token = 0ull; rc == YVEX_OK && token < run->job->token_count; ++token) {
            attn_phase_bind(run, token);
            for (stage = 0u; rc == YVEX_OK && stage < first_end; ++stage)
                rc = attn_kernel_stages[stage](run);
        }
        for (token = 0ull; rc == YVEX_OK && token < run->job->token_count; ++token) {
            attn_phase_bind(run, token);
            rc = attn_kernel_stages[YVEX_CUDA_ATTENTION_STAGE_COMPRESS](run);
        }
        for (token = 0ull; rc == YVEX_OK && token < run->job->token_count; ++token) {
            attn_phase_bind(run, token);
            for (stage = last_begin; rc == YVEX_OK && stage < YVEX_CUDA_ATTENTION_STAGE_COUNT;
                 ++stage)
                rc = attn_kernel_stages[stage](run);
        }
        if (rc == YVEX_OK && measure_device_time)
            rc = yvex_cuda_timing(
                run->backend, yvex_cuda_launch_stream(run->backend), YVEX_CUDA_TIMING_FINISH,
                &run->device_execution_elapsed_ns,
                "cuda.deepseek_attention.eager.timing.finish", run->err);
        else if (rc != YVEX_OK && measure_device_time)
            (void)yvex_cuda_timing(run->backend, NULL, YVEX_CUDA_TIMING_DISCARD,
                                   NULL, NULL, NULL);
        if (rc != YVEX_OK && run->failure &&
            run->failure->code == YVEX_BACKEND_ATTENTION_FAILURE_NONE) {
            run->failure->code = YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE;
            run->failure->stage = "cuda.deepseek_attention.eager.timing";
            run->failure->expected = 1ull;
            run->failure->actual = 0ull;
        }
        if (rc == YVEX_OK)
            rc = attn_cancel(
                run, "cuda.deepseek_attention.cancel.after_request", 1);
        return rc;
    }
    if (run->state->attention_mode == YVEX_BACKEND_CUDA_ATTENTION_FULL &&
        !run->job->candidate_block_visible) {
        rc = attn_graph_execute(run, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT);
        return rc == YVEX_OK
                   ? attn_cancel(
                         run, "cuda.deepseek_attention.cancel.after_full_graph", 1)
                   : rc;
    }
    if (run->state->attention_mode != YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE &&
        !(run->state->attention_mode == YVEX_BACKEND_CUDA_ATTENTION_FULL &&
          run->job->candidate_block_visible))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.deepseek_attention.graph.mode", YVEX_BACKEND_CUDA_ATTENTION_FULL,
            run->state->attention_mode, YVEX_ERR_UNSUPPORTED,
            "CUDA attention execution mode is unavailable");
    /*
     * Parallel draft attention needs every candidate key before any query is
     * reduced. A monolithic per-token graph would read later candidate slots
     * before their projections ran, so split this request at the projection
     * boundary even when ordinary attention selected full capture.
     */
    rc = attn_graph_execute(run, 0u, first_end);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = attn_graph_execute(
            run, YVEX_CUDA_ATTENTION_STAGE_COMPRESS,
            YVEX_CUDA_ATTENTION_STAGE_COMPRESS + 1u);
    if (rc == YVEX_OK)
        rc = attn_graph_execute(run, last_begin, YVEX_CUDA_ATTENTION_STAGE_COUNT);
    if (rc == YVEX_OK)
        rc = attn_cancel(
            run, "cuda.deepseek_attention.cancel.after_graph_piece", 1);
    return rc;
}
static int attn_synchronize(attn_run *run) {
    unsigned long long expected_topk, output_elements, token;
    unsigned long long output_width;
    CUdeviceptr output_source;
    size_t index;
    int device_wide = 0, rc = YVEX_OK;
    output_width = run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                       ? run->job->residual_expanded_width : run->job->hidden_width;
    output_source = run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                        ? run->phase_envelope_output : run->phase_output;
    if (run->job->device_output &&
        !yvex_core_u64_mul(run->job->token_count, output_width, &output_elements))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.copy.device-output", ULLONG_MAX,
            run->job->token_count, YVEX_ERR_BOUNDS,
            "CUDA attention device output extent overflowed");
    if (run->job->device_output)
        rc = yvex_cuda_activation_copy(run->backend, output_source,
            run->job->device_output, output_elements,
            "cuda.deepseek_attention.copy.device-output", run->err);
    if (rc == YVEX_OK && run->job->device_output) {
        size_t bytes;
        if (!yvex_cuda_work_checked_bytes(output_elements, sizeof(float), &bytes))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.deepseek_attention.copy.device-output", ULLONG_MAX,
                run->job->token_count, YVEX_ERR_BOUNDS,
                "CUDA attention device output accounting overflowed");
        rc = attn_account_d2d(run, bytes, "cuda.deepseek_attention.copy.device-output");
    }
    if (rc == YVEX_OK && run->job->retain_prefix_checkpoints) {
        run->rolling[ROLL_MAIN].after_kv = run->phase_main_kv +
            run->rolling[ROLL_MAIN].extent * sizeof(float);
        run->rolling[ROLL_MAIN].after_score = run->phase_main_score +
            run->rolling[ROLL_MAIN].extent * sizeof(float);
        run->rolling[ROLL_INDEX].after_kv = run->phase_index_kv +
            run->rolling[ROLL_INDEX].extent * sizeof(float);
        run->rolling[ROLL_INDEX].after_score = run->phase_index_score +
            run->rolling[ROLL_INDEX].extent * sizeof(float);
    }
    if (rc == YVEX_OK) rc = attn_downloads_enqueue(run);
    if (rc == YVEX_OK)
        rc = yvex_cuda_launch_synchronize(
            run->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED, &device_wide,
            "cuda.deepseek_attention.synchronize", run->err);
    if (rc == YVEX_OK) {
        run->stream_synchronizations +=
            (unsigned long long)(!attn_graph_mode(run) &&
                                 run->job->evidence_level != 0u && run->state->timing_ready) +
            (unsigned long long)!device_wide;
        run->device_synchronizations += (unsigned long long)device_wide;
    }
    if (rc != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE,
            "cuda.deepseek_attention.synchronize", 1ull, 0ull,
            (yvex_status)rc, "CUDA attention completion failed");
    run->host_status = *run->staged_status;
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA) {
        run->topk_count = 0ull;
        run->staged_valid_count = 0ull;
        for (token = 0ull; token < run->job->token_count; ++token) {
            unsigned long long selected = run->staged_selected_count[token];
            unsigned long long valid = run->staged_candidate_count[token];
            expected_topk = valid < run->job->indexer_topk ? valid : run->job->indexer_topk;
            if (valid > run->candidate_capacity || selected != expected_topk)
                return attn_run_fail(
                    run, YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
                    "cuda.deepseek_attention.copy.topk", expected_topk, selected,
                    YVEX_ERR_BOUNDS,
                    "CUDA attention top-k counts violate candidate geometry");
            if (selected > run->topk_count) run->topk_count = selected;
            if (valid > run->staged_valid_count) run->staged_valid_count = valid;
        }
    }
    for (index = 0u; index < run->transfer_count; ++index)
        if (run->transfers[index].used &&
            (*run->transfers[index].used > run->transfers[index].capacity ||
             *run->transfers[index].used > run->transfers[index].output_capacity))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_COPY,
                run->transfers[index].stage, run->transfers[index].output_capacity,
                *run->transfers[index].used, YVEX_ERR_BOUNDS,
                "CUDA attention logical output exceeded its publication span");
    if (run->host_status != 0)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_NUMERIC,
            "cuda.deepseek_attention.numeric", 0ull,
            (unsigned long long)run->host_status, YVEX_ERR_FORMAT,
            "CUDA attention device numerical stage refused its input");
    return YVEX_OK;
}
static int attn_stage_layout(attn_run *run, unsigned char *base, size_t *total) {
    unsigned long long csa_tokens =
        run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
            ? run->job->token_count : 0ull;
    return run->ops->stage_layout(
        base, run->uploads, run->upload_count, run->transfers,
        run->transfer_count, csa_tokens, &run->staged_status,
        &run->staged_selected_count, &run->staged_candidate_count, total);
}
static int attn_stage_allocate(attn_run *run) {
    const char *injected = getenv("YVEX_TEST_CUDA_ATTENTION_FAILURE");
    size_t actual = run->host_stage_bytes;
    int rc = run->ops->stage_acquire(
        run->backend, run->host_stage_bytes, attn_graph_mode(run),
        injected && strcmp(injected, "host-staging") == 0,
        &run->host_stage, &run->host_workspace_reused,
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    if (attn_stage_layout(run, run->host_stage, &actual) &&
        actual == run->host_stage_bytes)
        return YVEX_OK;
    return attn_run_fail(
        run, YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
        "cuda.deepseek_attention.host_stage.layout", run->host_stage_bytes,
        actual, YVEX_ERR_BOUNDS, "CUDA attention host staging layout drifted");
}
static int attn_stage_inputs(attn_run *run) {
    size_t i;
    for (i = 0u; i < run->upload_count; ++i) {
        attn_upload *upload = &run->uploads[i];
        size_t bytes, used_bytes;
        if (!upload->staged ||
            !yvex_cuda_work_checked_bytes(upload->count, upload->width, &bytes) ||
            !yvex_cuda_work_checked_bytes(upload->used, upload->width, &used_bytes))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, upload->stage,
                upload->count, 0ull, YVEX_ERR_BOUNDS,
                "CUDA attention pinned input staging is invalid");
        memset(upload->staged, 0, bytes);
        if (upload->device == &run->phase_input) {
            unsigned long long token;
            for (token = 0ull; token < run->job->token_count; ++token)
                memcpy((float *)upload->staged + token * run->input_extent,
                       run->job->input + token * run->job->input_stride,
                       (size_t)run->input_extent * sizeof(float));
        } else if (upload->device == &run->phase_local_positions) {
            unsigned long long token;
            if (run->initial_local_count)
                memcpy(upload->staged, run->job->local_positions,
                       (size_t)run->initial_local_count * sizeof(unsigned long long));
            for (token = 0ull; token < run->job->token_count; ++token)
                ((unsigned long long *)upload->staged)[run->initial_local_count + token] =
                    run->job->token_position + token;
        } else if (upload->device == &run->phase_compressed_positions ||
                   upload->device == &run->phase_indexer_positions) {
            const unsigned long long initial = upload->device == &run->phase_compressed_positions
                                                   ? run->initial_compressed_count
                                                   : run->initial_indexer_count;
            const unsigned long long *positions = upload->device == &run->phase_compressed_positions
                                                      ? run->job->compressed_positions
                                                      : run->job->indexer_positions;
            unsigned long long token, emitted = initial;
            if (initial) memcpy(upload->staged, positions, (size_t)initial * sizeof(*positions));
            for (token = 0ull; token < run->job->token_count; ++token) {
                unsigned long long position = run->job->token_position + token;
                if ((position + 1ull) % run->job->compression_ratio == 0ull)
                    ((unsigned long long *)upload->staged)[emitted++] =
                        position + 1ull - run->job->compression_ratio;
            }
        } else if (used_bytes) {
            memcpy(upload->staged, upload->source, used_bytes);
        }
    }
    return YVEX_OK;
}
/* Publish the fully admitted transaction as the sole caller-visible commit. */
static int attn_publish(attn_run *run) {
    yvex_backend_host_workspace_summary workspace;
    unsigned long long backend_h2d, backend_d2h;
    size_t i;
    if (!yvex_core_u64_add(
            run->backend->stats.h2d_bytes, run->h2d_bytes, &backend_h2d) ||
        !yvex_core_u64_add(
            run->backend->stats.d2h_bytes, run->d2h_bytes, &backend_d2h))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.deepseek_attention.transfer.account", ULLONG_MAX,
            run->h2d_bytes > run->d2h_bytes ? run->h2d_bytes : run->d2h_bytes,
            YVEX_ERR_BOUNDS,
            "CUDA attention cumulative transfer accounting overflowed");
    for (i = 0u; i < run->transfer_count; ++i) {
        const attn_transfer *transfer = &run->transfers[i];
        unsigned long long count = transfer->used ?
            *transfer->used : transfer->capacity;
        if (count) memcpy(transfer->output, transfer->staged,
                          (size_t)count * transfer->width);
    }
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
        run->output->topk_counts.data) {
        memcpy(run->output->topk_counts.data, run->staged_selected_count,
               (size_t)run->job->token_count * sizeof(*run->staged_selected_count));
    }
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA &&
        run->output->valid_candidate_counts.data) {
        memcpy(run->output->valid_candidate_counts.data, run->staged_candidate_count,
               (size_t)run->job->token_count * sizeof(*run->staged_candidate_count));
    }
    run->output->tokens_executed = run->job->token_count;
    run->output->compressed_count = run->phase_compressed_count;
    run->output->indexer_count = run->phase_indexer_count;
    run->output->topk_count = run->topk_count;
    run->output->valid_candidate_count = run->staged_valid_count;
    run->output->host_bytes = 0ull;
    run->output->peak_host_bytes = run->host_stage_bytes;
    run->output->device_bytes = 0ull;
    run->output->peak_device_bytes = run->resources.peak_bytes;
    run->output->kernel_launches = run->resources.launches;
    run->output->h2d_bytes = run->h2d_bytes;
    run->output->d2h_bytes = run->d2h_bytes;
    run->output->d2d_bytes = run->d2d_bytes;
    run->output->stream_synchronizations = run->stream_synchronizations;
    run->output->device_synchronizations = run->device_synchronizations;
    run->output->device_execution_elapsed_ns = run->device_execution_elapsed_ns;
    run->backend->stats.h2d_bytes = backend_h2d;
    run->backend->stats.d2h_bytes = backend_d2h;
    if (yvex_backend_host_workspace_summary_get(run->backend, &workspace)) {
        run->output->host_workspace_capacity = workspace.capacity;
        run->output->host_workspace_used = workspace.used;
        run->output->host_workspace_peak = workspace.peak;
        run->output->host_workspace_allocation_count = workspace.allocation_count;
        run->output->host_workspace_reused = run->host_workspace_reused;
    }
    return YVEX_OK;
}
typedef struct {
    int (*execute)(attn_run *run); const char *cancel_stage; int pending_device_work;
} attn_transaction_phase;
static const attn_transaction_phase attn_transaction[] = {
    {attn_stage_allocate, NULL, 0}, {attn_stage_inputs, NULL, 0},
    {attn_allocate_base, "cuda.deepseek_attention.cancel.after_allocation", 0},
    {attn_numerical_execute, "cuda.deepseek_attention.cancel.before_commit", 1},
    {attn_synchronize, "cuda.deepseek_attention.cancel.after_copy", 0}
};
int yvex_backend_attention_execute(yvex_backend *backend, const yvex_backend_attention_job *job,
                                   yvex_backend_attention_output *output,
                                   yvex_backend_attention_failure *failure, yvex_error *err) {
    attn_run run = {.backend = backend, .state = yvex_cuda_state(backend),
                    .ops = yvex_cuda_attention_operations_get(), .topk_capacity = 1ull,
                    .output = output, .failure = failure, .err = err, .candidate_capacity = 1ull};
    yvex_error cleanup_error;
    size_t phase;
    int cleanup_rc, rc;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (job) {
        run.request = *job;
        run.job = &run.request;
    }
    rc = attn_prepare(&run);
    if (rc != YVEX_OK) return rc;
    backend_workspace_reset(backend);
    backend_host_workspace_reset(backend);
    for (phase = 0u, rc = YVEX_OK;
         rc == YVEX_OK && phase < sizeof(attn_transaction) /
                                      sizeof(attn_transaction[0]);
         ++phase) {
        rc = attn_transaction[phase].execute(&run);
        if (rc == YVEX_OK && attn_transaction[phase].cancel_stage)
            rc = attn_cancel(
                &run, attn_transaction[phase].cancel_stage,
                attn_transaction[phase].pending_device_work);
    }
    cleanup_rc = yvex_cuda_work_cleanup(&run.resources, &cleanup_error);
    if (rc == YVEX_OK && cleanup_rc != YVEX_OK)
        rc = attn_run_fail(
            &run, YVEX_BACKEND_ATTENTION_FAILURE_CLEANUP,
            "cuda.deepseek_attention.cleanup", 0ull,
            run.state->deferred_release_bytes, (yvex_status)cleanup_rc,
            "CUDA attention temporary cleanup failed");
    if (rc == YVEX_OK) rc = attn_cancel(&run, "cuda.deepseek_attention.cancel.publish", 0);
    if (rc == YVEX_OK) rc = attn_publish(&run);
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}
