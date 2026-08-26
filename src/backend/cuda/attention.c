/* Execute admitted encoded attention with device-complete numerical work and publication. */
#include <yvex/internal/backend.h>
#include "src/backend/cuda/private.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
enum { YVEX_CUDA_ATTN_BLOCK = 256u, YVEX_CUDA_ATTN_TRANSFERS = 20u, YVEX_CUDA_ATTN_UPLOADS = 16u };
enum { YVEX_CUDA_ATTN_INITIALIZERS = 64u };
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
    const yvex_cuda_attention_configuration *configuration;
    yvex_backend_attention_job request;
    yvex_backend_attention_job *job; yvex_backend_attention_output *output;
    yvex_backend_attention_failure *failure; yvex_error *err;
    yvex_backend_attention_completion *completion;
    unsigned char *host_stage; size_t host_stage_bytes, download_stage_bytes;
    attn_transfer transfers[YVEX_CUDA_ATTN_TRANSFERS]; size_t transfer_count;
    attn_upload uploads[YVEX_CUDA_ATTN_UPLOADS]; size_t upload_count;
    attn_initializer initializers[YVEX_CUDA_ATTN_INITIALIZERS]; size_t initializer_count;
    CUdeviceptr weight[YVEX_BACKEND_ATTENTION_WEIGHT_COUNT];
    CUdeviceptr input, core_input, q_low, query, raw_kv, download_stage;
    CUdeviceptr sinks, attention;
    attn_rolling_run rolling[ROLL_COUNT];
    CUdeviceptr index_query, index_weights, history_local, history_local_positions;
    CUdeviceptr history_compressed, history_compressed_positions;
    CUdeviceptr history_indexer, history_indexer_positions;
    CUdeviceptr selected, selected_positions, selected_count, valid_count;
    CUdeviceptr topk_scores, valid_indexes, device_status;
    unsigned long long query_width, candidate_capacity, topk_capacity, low_count;
    unsigned long long local_extent, compressed_extent, history_index_extent;
    unsigned long long local_storage_extent, compressed_storage_extent, indexer_storage_extent;
    unsigned long long local_capacity, local_ring_capacity, compressed_capacity, indexer_capacity;
    unsigned long long index_query_extent, emission_position;
    unsigned long long h2d_bytes, d2h_bytes, d2d_bytes, device_execution_elapsed_ns;
    unsigned long long stream_synchronizations, device_synchronizations;
    int *staged_status; unsigned long long *staged_selected_count, *staged_candidate_count;
    int host_workspace_reused;
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
    return run->configuration && run->configuration->mode != YVEX_BACKEND_CUDA_ATTENTION_EAGER;
}
static int attn_run_fail(attn_run *run, yvex_backend_attention_failure_code code,
                                   const char *stage, unsigned long long expected, unsigned long long actual,
                                   yvex_status status, const char *message) {
    return run->ops->fail(run->failure, code, stage, expected, actual, run->err, status, message);
}
static int attn_account_d2d(attn_run *run, size_t bytes, const char *stage) {
    if (yvex_core_u64_add(run->d2d_bytes, bytes, &run->d2d_bytes)) return YVEX_OK;
    return attn_run_fail(run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage,
                         ULLONG_MAX, bytes, YVEX_ERR_BOUNDS,
                         "CUDA attention D2D accounting overflowed");
}
static int attn_device_copy(attn_run *run, CUdeviceptr target, CUdeviceptr source,
                            size_t bytes, const char *stage) {
    CUstream stream = yvex_cuda_launch_stream(run->backend);
    CUresult copied = stream && run->state->driver.cuMemcpyDtoDAsync_v2
                          ? run->state->driver.cuMemcpyDtoDAsync_v2(target, source, bytes, stream)
                          : !stream ? run->state->driver.cuMemcpyDtoD_v2(target, source, bytes)
                                    : (CUresult)1;
    int rc = yvex_cuda_status(&run->state->driver, copied, stage, run->err);
    return rc == YVEX_OK ? attn_account_d2d(run, bytes, stage) : rc;
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
            "cuda.attention.validate.output", capacity,
            output_capacity, YVEX_ERR_BOUNDS,
            "CUDA attention output span is absent, invalid, or undersized");
    transfer = &run->transfers[run->transfer_count++];
    *transfer = (attn_transfer){device, output, NULL, capacity, output_capacity, used, width, stage};
    return YVEX_OK;
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
     used, extent, span, classes, evidence, envelope, "cuda.attention.copy." label}
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
    int rc = YVEX_OK;
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
            "cuda.attention.copy.phase_counts", run->job->token_count,
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
     "cuda.attention.upload." label}
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
        verdict ? "cuda.attention.validate.alias"
                : "cuda.attention.validate.alias.range",
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
            "cuda.attention.validate.position", ULLONG_MAX - 1ull,
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
            "cuda.attention.validate.geometry", run->query_width,
            group_width, YVEX_ERR_FORMAT,
            "CUDA attention interdependent geometry is invalid");
    if ((run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA && ratio != 0ull) ||
        (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA && !ratio))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.ratio",
            run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA ? 0ull : 1ull,
            ratio, YVEX_ERR_FORMAT,
            "CUDA attention class requires a compatible compression ratio");
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
            "cuda.attention.validate.envelope",
            run->job->residual_expanded_width, group_width, YVEX_ERR_FORMAT,
            "CUDA attention envelope geometry is invalid");
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
        run->job->compressed_stride != run->job->head_dimension)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.compressed_stride",
            run->job->head_dimension, run->job->compressed_stride,
            YVEX_ERR_FORMAT,
            "CUDA compressed attention phase stride is invalid");
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA) {
        rc = run->ops->validate_rolling(
            run->job, &run->job->main_rolling, ratio, run->job->head_dimension,
            run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA,
            &run->rolling[ROLL_MAIN].extent,
            "cuda.attention.validate.main_state", run->failure, run->err);
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
                "cuda.attention.validate.compressed_extent",
                ULLONG_MAX, run->job->head_dimension, YVEX_ERR_BOUNDS,
                "CUDA attention compressed output extent overflowed");
    } else if (run->job->main_rolling.present || run->job->indexer_rolling.present ||
               run->job->compressed_count || run->job->indexer_count) {
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.swa_state", 0ull, 1ull,
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
                "cuda.attention.validate.index_geometry",
                run->job->head_dimension, run->job->compressed_stride,
                YVEX_ERR_FORMAT,
                "CUDA CSA indexer and compressed geometry is invalid");
        rc = run->ops->validate_rolling(
            run->job, &run->job->indexer_rolling, ratio,
            run->job->indexer_head_dimension, 1,
            &run->rolling[ROLL_INDEX].extent,
            "cuda.attention.validate.index_state", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
        run->rolling[ROLL_INDEX].value_count =
            run->rolling[ROLL_MAIN].value_count;
        if (!yvex_core_u64_mul(run->rolling[ROLL_INDEX].value_count,
                               run->job->indexer_head_dimension,
                               &run->rolling[ROLL_INDEX].value_extent))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.attention.validate.indexer_extent",
                ULLONG_MAX, run->job->indexer_head_dimension, YVEX_ERR_BOUNDS,
                "CUDA attention indexer output extent overflowed");
        if (!yvex_core_u64_add(run->indexer_capacity, 1ull,
                               &candidate_count))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.attention.validate.candidates", ULLONG_MAX,
                run->job->indexer_count, YVEX_ERR_BOUNDS,
                "CUDA attention candidate count overflowed");
        run->candidate_capacity = candidate_count ? candidate_count : 1ull;
        run->topk_capacity = run->job->indexer_topk < candidate_count
            ? run->job->indexer_topk : candidate_count;
        if (!run->topk_capacity) run->topk_capacity = 1ull;
    } else if (run->job->indexer_rolling.present || run->job->indexer_count) {
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.validate.index_state", 0ull, 1ull,
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
        "cuda.attention.validate.kv_activation", run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = run->ops->validate_activation(
            &run->job->compressor_activation,
            run->job->head_dimension - run->job->position.rope_dimensions,
            "cuda.attention.validate.compressor_activation",
            run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = run->ops->validate_activation(
            &run->job->compressor_rotated_activation,
            run->job->indexer_head_dimension,
            "cuda.attention.validate.compressor_rotated_activation",
            run->failure, run->err);
    if (rc == YVEX_OK && run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA)
        rc = run->ops->validate_activation(
            &run->job->indexer_query_activation,
            run->job->indexer_head_dimension,
            "cuda.attention.validate.indexer_activation",
            run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = attn_transfer_plan(run);
    if (rc == YVEX_OK) rc = attn_upload_plan(run);
    if (rc == YVEX_OK) rc = attn_alias_validate(run);
    return rc;
}
static const attn_upload *attn_upload_find(const attn_run *run, const CUdeviceptr *target) {
    for (size_t i = 0u; i < run->upload_count; ++i)
        if (run->uploads[i].device == target) return &run->uploads[i];
    return NULL;
}
static int attn_alloc_values(attn_run *run, CUdeviceptr *target,
                             unsigned long long count, size_t width,
                             const void *source, int zero, const char *stage) {
    const attn_upload *upload;
    const void *stable_source = source;
    unsigned long long resident_address = 0ull;
    CUdeviceptr device_source = 0u;
    size_t bytes, visible_bytes;
    int captured = attn_graph_mode(run), persistent_input, resident, rc;
    if (*target) return YVEX_OK;
    upload = attn_upload_find(run, target);
    if (!yvex_cuda_work_checked_bytes(count, (unsigned long long)width, &bytes))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, stage, ULLONG_MAX,
            count, YVEX_ERR_BOUNDS,
            "CUDA attention allocation size overflowed");
    if (target == &run->phase_input)
        device_source = yvex_cuda_activation_pointer(run->backend, run->job->device_input);
    visible_bytes = upload && upload->used ? (size_t)(upload->used * upload->width) : 1u;
    /* Transient candidate capacity is wider; only committed ring rows may alias residency. */
    persistent_input = upload && (target == &run->phase_compressed || target == &run->phase_indexer ||
        (target == &run->phase_local && run->initial_local_count + run->job->token_count <= run->local_ring_capacity));
    resident = source && !device_source && upload && !upload->generated && persistent_input
        ? yvex_backend_state_residency_resolve(run->backend, source, visible_bytes,
                                               &resident_address)
        : YVEX_BACKEND_RESIDENT_MISS;
    if (resident == YVEX_BACKEND_RESIDENT_INVALID)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, stage, bytes, 0ull,
            YVEX_ERR_STATE, "persistent device state mapping is invalid");
    if (resident == YVEX_BACKEND_RESIDENT_HIT) {
        *target = (CUdeviceptr)resident_address;
        return YVEX_OK;
    }
    if (source && !device_source) {
        rc = run->ops->account_transfer(
            count, width, &run->h2d_bytes, stage, run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    if (!device_source) {
        if (upload && upload->staged) stable_source = upload->staged;
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
    return attn_device_copy(run, *target, device_source, bytes, stage);
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
     "cuda.attention.alloc." label}
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
    case EXT_INDEX_STATE:
        if (!run->job->retain_prefix_checkpoints) left = 1ull;
        else if (left == ULLONG_MAX) return 0;
        else ++left;
        right = run->rolling[kind == EXT_MAIN_STATE ? ROLL_MAIN : ROLL_INDEX].extent;
        break;
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
            "cuda.attention.execute", 1ull, 0ull,
            YVEX_ERR_INVALID_ARG, "CUDA attention backend is required");
    rc = backend_dispatch_admit(run->backend, "cuda.attention.execute", run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->validate_job(
        run->job, run->output, run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    run->configuration = yvex_cuda_attention_configuration_active(run->state, run->job->phase);
    if (run->state->attention_configuration_count && !run->configuration)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET, "cuda.attention.capacity.phase",
            run->job->phase, YVEX_BACKEND_ATTENTION_PHASE_COUNT, YVEX_ERR_BOUNDS,
            "CUDA attention phase has no active capacity configuration");
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
            "cuda.attention.validate.device-activation", 1ull, 0ull,
            YVEX_ERR_FORMAT, "CUDA attention device activation views are incompatible");
    run->initial_local_count = run->job->local_count;
    run->initial_compressed_count = run->job->compressed_count;
    run->initial_indexer_count = run->job->indexer_count;
    run->local_capacity = attn_graph_mode(run)
                              ? yvex_cuda_attention_local_capacity(run->configuration, run->job, 0)
                              : run->job->sliding_window;
    run->local_ring_capacity = attn_graph_mode(run) ?
        yvex_cuda_attention_local_capacity(run->configuration, run->job, 1) :
        run->job->sliding_window - (run->job->candidate_block_visible ? 0ull : 1ull);
    run->compressed_capacity = run->job->attention_class == YVEX_BACKEND_ATTENTION_SWA
        ? 0ull : (attn_graph_mode(run)
                      ? run->configuration->compressed_capacity
                      : (run->job->token_position + run->job->token_count) /
                            run->job->compression_ratio);
    if (run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA &&
        !run->compressed_capacity)
        run->compressed_capacity = 1ull;
    run->indexer_capacity = run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
        ? (attn_graph_mode(run)
               ? run->configuration->indexer_capacity : run->compressed_capacity)
        : 0ull;
    if (!yvex_core_u64_add(run->job->token_position, run->job->token_count, &phase_end) ||
        !yvex_core_u64_add(
            attn_graph_mode(run) ? run->local_capacity : run->initial_local_count,
            run->job->token_count, &local_storage))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.validate.phase", ULLONG_MAX,
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
            "cuda.attention.validate.history_extent", ULLONG_MAX,
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
            "cuda.attention.capacity.local", run->local_capacity,
            run->initial_local_count, YVEX_ERR_BOUNDS,
            "CUDA attention local history exceeds configured capacity");
    if (compressed_end > run->compressed_capacity)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.capacity.compressed", run->compressed_capacity,
            compressed_end, YVEX_ERR_BOUNDS,
            "CUDA attention compressed history exceeds configured capacity");
    if (indexer_end > run->indexer_capacity)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.capacity.indexer", run->indexer_capacity,
            indexer_end, YVEX_ERR_BOUNDS,
            "CUDA attention indexer history exceeds configured capacity");
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
                    "cuda.attention.validate.extent", ULLONG_MAX, 0ull,
                    YVEX_ERR_BOUNDS,
                    "CUDA attention logical extent overflowed");
    }
    rc = attn_validate_derived(run);
    if (rc != YVEX_OK) return rc;
    if (!attn_stage_layout(run, NULL, &run->host_stage_bytes) ||
        !run->host_stage_bytes)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.validate.host_layout", ULLONG_MAX, 0ull,
            YVEX_ERR_BOUNDS,
            "CUDA attention host staging layout is not representable");
    if (run->host_stage_bytes > run->job->max_host_bytes) {
        rc = attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.validate.host_budget",
            run->job->max_host_bytes, run->host_stage_bytes,
            YVEX_ERR_NOMEM,
            "CUDA attention host staging exceeds its sealed workspace recipe");
        yvex_error_setf(
            run->err, YVEX_ERR_NOMEM,
            "cuda.attention.validate.host_budget",
            "CUDA attention needs %zu host-staging bytes but the sealed recipe provides %llu",
            run->host_stage_bytes, run->job->max_host_bytes);
        return rc;
    }
    rc = attn_cancel(
        run, "cuda.attention.cancel.before_dispatch", 0);
    if (rc != YVEX_OK) return rc;
    rc = yvex_cuda_require_capability(
        run->backend, YVEX_BACKEND_VARIANT_ATTENTION_ENCODED,
        "cuda.attention.capability", run->err);
    if (rc != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.capability", 1ull, 0ull,
            (yvex_status)rc,
            "CUDA encoded-attention kernel bundle is not admitted");
    rc = injected && strcmp(injected, "context") == 0
        ? YVEX_ERR_BACKEND
        : yvex_cuda_set_current(
              run->backend, "cuda.attention.context", run->err);
    if (rc != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.context", 1ull, 0ull,
            (yvex_status)rc, "CUDA attention context activation failed");
    run->resources.backend = run->backend;
    run->resources.state = run->state;
    run->resources.variant = YVEX_BACKEND_VARIANT_ATTENTION_ENCODED;
    run->resources.budget = run->job->max_device_bytes;
    run->resources.activation_q8 = run->job->native_execution && run->job->evidence_level < 3u &&
        run->state->kernel_bundle_native &&
        run->state->qtype_tensorcore_rows_function != NULL;
    /* Full evidence selects canonical-order arithmetic for the stage oracles. */
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
                "cuda.attention.resident.weight", host->encoded_bytes, 0ull,
                YVEX_ERR_STATE, "CUDA resident weight mapping is incomplete or invalid");
        rc = run->ops->allocate(&run->resources, &run->weight[slot],
                                     host->encoded_bytes, host->encoded, 0,
                                     "cuda.attention.alloc.weight",
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
static void attn_rolling_bind(attn_run *run, unsigned int slot, CUdeviceptr kv, CUdeviceptr score,
                              unsigned long long extent, unsigned long long ordinal) {
    unsigned long long before = run->job->retain_prefix_checkpoints ? ordinal : 0ull;
    unsigned long long after = run->job->retain_prefix_checkpoints ? ordinal + 1ull : 0ull;
    unsigned long long bytes = extent * sizeof(float);
    run->rolling[slot].before_kv = kv + before * bytes;
    run->rolling[slot].before_score = score + before * bytes;
    run->rolling[slot].after_kv = kv + after * bytes;
    run->rolling[slot].after_score = score + after * bytes;
}
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
    /* The uploaded no-checkpoint state is transaction-private, so in-place mutation preserves
     * rollback; checkpointed execution keeps each historical state version distinct. */
    attn_rolling_bind(run, ROLL_MAIN, run->phase_main_kv, run->phase_main_score,
                      run->rolling[ROLL_MAIN].extent, ordinal);
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
    attn_rolling_bind(run, ROLL_INDEX, run->phase_index_kv, run->phase_index_score,
                      run->rolling[ROLL_INDEX].extent, ordinal);
    run->rolling[ROLL_INDEX].value =
        run->phase_indexer + compressed_count * job->indexer_stride * float_size;
    run->rolling[ROLL_INDEX].positions =
        run->phase_indexer_positions + compressed_count * u64_size;
    run->rolling[ROLL_INDEX].value_count = emit;
    run->rolling[ROLL_INDEX].value_extent = emit * job->indexer_head_dimension;
}
static int attn_envelope_pre(attn_run *run) {
    unsigned long long streams = run->job->residual_stream_count, width = run->job->residual_stream_width;
    unsigned long long rows = run->job->token_count;
    unsigned int shared_bytes;
    int rc;
    if (run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE) return YVEX_OK;
    if (run->ordinal) return YVEX_OK;
    if (rows > UINT_MAX ||
        streams > UINT_MAX / sizeof(double) - 1ull - YVEX_CUDA_ATTN_BLOCK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.mhc_pre", UINT_MAX, rows,
            YVEX_ERR_BOUNDS, "CUDA mHC shared geometry exceeds launch bounds");
    shared_bytes = (unsigned int)((streams + 1ull + YVEX_CUDA_ATTN_BLOCK) *
                                  sizeof(double));
    rc = run->ops->matvec(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_FUNCTION], 0ull,
        run->job->mhc_mixing_rows, rows, run->phase_input, run->phase_mhc_mix, 0,
        run->device_status, "cuda.attention.mhc_function",
        run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->decode(
            &run->resources,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_SCALE], 0ull, 3ull,
            run->phase_mhc_scale, run->device_status,
            "cuda.attention.mhc_scale", run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->decode(
            &run->resources,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_MHC_BASE], 0ull,
            run->job->mhc_mixing_rows, run->phase_mhc_base, run->device_status,
            "cuda.attention.mhc_base", run->failure, run->err);
    if (rc == YVEX_OK) {
        void *params[] = {
            &run->phase_input, &run->phase_mhc_mix, &run->phase_mhc_scale,
            &run->phase_mhc_base,
            &streams, &width, (void *)&run->job->mhc_mixing_rows,
            (void *)&run->job->mhc_sinkhorn_iterations,
            (void *)&run->job->rms_epsilon, (void *)&run->job->mhc_epsilon,
            (void *)&run->job->mhc_residual_post_multiplier, &run->phase_core_input,
            &run->phase_mhc_post, &run->phase_mhc_combination, &rows,
            &run->device_status
        };
        rc = run->ops->launch(
            &run->resources, run->state->residual_mhc_pre_function,
            (unsigned int)rows,
            YVEX_CUDA_ATTN_BLOCK, shared_bytes, params,
            "cuda.attention.mhc_pre", run->failure, run->err);
    }
    if (rc == YVEX_OK)
        rc = run->ops->weighted_norm(
            &run->resources, run->phase_core_input, run->job->hidden_width, rows,
            &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM],
            run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INPUT_NORM],
            run->job->rms_epsilon, run->device_status,
            "cuda.attention.input_norm", run->failure, run->err);
    return rc;
}
static int attn_project(attn_run *run) {
    unsigned long long rows = run->job->token_count, query_vectors, core_values;
    int rc = YVEX_OK;
    if (!yvex_core_u64_mul(rows, run->job->query_heads, &query_vectors) ||
        !yvex_core_u64_mul(rows, run->job->hidden_width, &core_values))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.project.rows", ULLONG_MAX, rows,
            YVEX_ERR_BOUNDS, "CUDA attention projection row geometry overflowed");
    if (run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_CORE)
        rc = run->ops->round_bf16(
            &run->resources, run->core_input, core_values, run->device_status,
            "cuda.attention.input_round", run->failure, run->err);
    if (rc == YVEX_OK) rc = run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A], 0ull,
        run->job->q_rank, rows, run->core_input, run->q_low, 1, run->device_status,
        "cuda.attention.q_a", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->weighted_norm(
        &run->resources, run->q_low, run->job->q_rank, rows,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_A_NORM],
        run->job->rms_epsilon, run->device_status, "cuda.attention.q_norm",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_Q_B],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_Q_B], 0ull,
        run->query_width, rows, run->q_low, run->query, 1, run->device_status,
        "cuda.attention.q_b", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_KV],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_KV], 0ull,
        run->job->kv_width, rows, run->core_input, run->raw_kv, 1, run->device_status,
        "cuda.attention.kv", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->weighted_norm(
        &run->resources, run->raw_kv, run->job->kv_width, rows,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_KV_NORM],
        run->job->rms_epsilon, run->device_status, "cuda.attention.kv_norm",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->unit_norm(
        &run->resources, run->query, query_vectors, run->job->head_dimension,
        run->job->rms_epsilon, run->device_status,
        "cuda.attention.query_unit_norm", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->phase_query, run->job->query_heads, rows, run->job->head_dimension,
        run->phase_start_position,
        &run->job->position, 0, run->device_status,
        "cuda.attention.query_rope", run->failure, run->err);
    if (rc == YVEX_OK)
        rc = run->ops->rope(
            &run->resources, run->phase_raw_kv, 1ull, rows, run->job->kv_width,
            run->phase_start_position,
            &run->job->position, 0, run->device_status,
            "cuda.attention.kv_rope", run->failure, run->err);
    if (rc == YVEX_OK && run->job->kv_width > run->job->position.rope_dimensions)
        rc = run->ops->activation(
            &run->resources, run->phase_raw_kv, rows,
            run->job->kv_width - run->job->position.rope_dimensions,
            run->job->kv_width, &run->job->attention_kv_activation, run->device_status,
            "cuda.attention.kv_activation",
            run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    return run->ops->decode(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_SINKS],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_SINKS], 0ull,
        run->job->query_heads, run->sinks, run->device_status,
        "cuda.attention.sinks", run->failure, run->err);
}
static int attn_rolling_execute(attn_run *run, unsigned int kind) {
    const int index = kind == ROLL_INDEX;
    const yvex_backend_attention_rolling *rolling = index ? &run->job->indexer_rolling
                                                          : &run->job->main_rolling;
    attn_rolling_run *device = &run->rolling[kind];
    yvex_backend_attention_weight_slot base = index ? YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_KV
                                                     : YVEX_BACKEND_ATTENTION_WEIGHT_MAIN_KV;
    const yvex_backend_attention_activation *activation = index
        ? &run->job->compressor_rotated_activation : &run->job->compressor_activation;
    const char *stage = index ? "cuda.attention.index_rolling" : "cuda.attention.main_rolling";
    unsigned long long activation_width = rolling->head_dimension, rows = rolling->state_width;
    int emit, rc;
    if (run->job->weights[base].qtype == YVEX_GGUF_QTYPE_BF16 &&
        run->job->weights[base + 1].qtype == YVEX_GGUF_QTYPE_BF16 &&
        rows <= UINT_MAX * 8ull - 7ull) {
        unsigned int grid = (unsigned int)((rows + 7ull) / 8ull);
        void *params[] = {
            &run->weight[base], &run->job->weights[base].row_bytes,
            &run->weight[base + 1], &run->job->weights[base + 1].row_bytes,
            &run->job->weights[base].row_width, &rows,
            &run->core_input, &device->kv, &device->score, &run->device_status};
        rc = run->ops->launch(&run->resources, run->state->attention_bf16_pair_function,
                              grid, 256u, 0u, params, stage, run->failure, run->err);
    } else {
        rc = run->ops->matvec(&run->resources, &run->job->weights[base], run->weight[base],
            0ull, rolling->state_width, 1ull, run->core_input, device->kv, 0,
            run->device_status, stage, run->failure, run->err);
        if (rc == YVEX_OK)
            rc = run->ops->matvec(&run->resources, &run->job->weights[base + 1],
                run->weight[base + 1], 0ull, rolling->state_width, 1ull,
                run->core_input, device->score, 0, run->device_status, stage,
                run->failure, run->err);
    }
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
            &run->resources, run->state->attention_rolling_state_function, 1u,
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
            &run->resources, device->value, 1ull, 1ull, rolling->head_dimension,
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
        &run->resources, device->value, 1ull, activation_width,
        activation_width, activation,
        run->device_status, stage, run->failure, run->err);
}
static int attn_index_prepare(attn_run *run) {
    unsigned long long rows = run->job->token_count, index_vectors;
    int rc;
    if (!yvex_core_u64_mul(rows, run->job->indexer_heads, &index_vectors))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.index.rows", ULLONG_MAX, rows,
            YVEX_ERR_BOUNDS, "CUDA attention index row geometry overflowed");
    rc = run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_Q], 0ull,
        run->index_query_extent, rows, run->phase_q_low, run->phase_index_query,
        1, run->device_status, "cuda.attention.index_query", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_INDEX_PROJECTION], 0ull,
        run->job->indexer_heads, rows, run->phase_core_input, run->phase_index_weights,
        1, run->device_status, "cuda.attention.index_weights",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->rope(
        &run->resources, run->phase_index_query, run->job->indexer_heads, rows,
        run->job->indexer_head_dimension, run->phase_start_position,
        &run->job->position, 0, run->device_status, "cuda.attention.index_query_rope",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    rc = run->ops->activation(
        &run->resources, run->phase_index_query, index_vectors,
        run->job->indexer_head_dimension, run->job->indexer_head_dimension,
        &run->job->indexer_query_activation,
        run->device_status, "cuda.attention.index_query_activation",
        run->failure, run->err);
    return rc;
}
static int attn_index_topk(attn_run *run) {
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
            &run->resources, run->state->attention_topk_function, 1u,
            YVEX_CUDA_ATTN_BLOCK, YVEX_CUDA_ATTN_BLOCK * sizeof(double),
            params, "cuda.attention.topk", run->failure, run->err);
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
    unsigned long long low_width, batch_blocks;
    unsigned int attention_class = (unsigned int)run->job->attention_class;
    int rc;
    if (!run->ordinal) {
        if (!yvex_core_u64_mul(run->job->token_count, run->job->query_heads,
                               &batch_blocks) || batch_blocks > UINT_MAX)
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
                "cuda.attention.reduce.rows", UINT_MAX, run->job->token_count,
                YVEX_ERR_BOUNDS, "CUDA attention row grid exceeds launch bounds");
        void *params[] = {
            &run->phase_query, &run->phase_local, &run->phase_local_positions,
            &run->initial_local_count, (void *)&run->job->local_stride,
            &run->phase_compressed, &run->phase_compressed_positions,
            (void *)&run->job->compressed_stride,
            &run->phase_selected, &run->phase_selected_count,
            &run->topk_capacity, &run->sinks,
            (void *)&run->job->query_heads, (void *)&run->job->head_dimension,
            (void *)&run->job->sliding_window,
            (void *)&run->job->compression_ratio, &attention_class,
            &run->phase_start_position, (void *)&run->job->token_count,
            (void *)&run->job->candidate_block_visible, &run->attention,
            &run->device_status
        };
        CUfunction reduce = run->resources.activation_q8
            ? run->state->attention_reduce_native_function
            : run->state->attention_reduce_function;
        unsigned int reduce_shared = run->resources.activation_q8
            ? 0u : YVEX_CUDA_ATTN_BLOCK * sizeof(double);
        rc = run->ops->launch(
            &run->resources, reduce,
            (unsigned int)batch_blocks, YVEX_CUDA_ATTN_BLOCK,
            reduce_shared, params,
            "cuda.attention.reduce", run->failure, run->err);
        if (rc != YVEX_OK) return rc;
    }
    if (run->ordinal + 1ull < run->job->token_count) return YVEX_OK;
    rc = run->ops->rope(
        &run->resources, run->phase_attention, run->job->query_heads,
        run->job->token_count, run->job->head_dimension,
        run->phase_start_position, &run->job->position, 1,
        run->device_status, "cuda.attention.output_inverse_rope",
        run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    if (!yvex_core_u64_mul(run->job->output_groups, run->job->output_rank,
                           &low_width))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.output_a.rows", ULLONG_MAX, run->job->output_groups,
            YVEX_ERR_BOUNDS, "CUDA attention output-A row geometry overflowed");
    rc = run->ops->matvec_grouped(
        &run->resources,
        &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_A],
        run->job->output_groups, run->job->output_rank,
        run->job->token_count, run->phase_attention, run->query_width,
        run->phase_low, low_width, 1, run->device_status,
        "cuda.attention.output_a", run->failure, run->err);
    if (rc != YVEX_OK) return rc;
    return run->ops->matvec(
        &run->resources, &run->job->weights[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B],
        run->weight[YVEX_BACKEND_ATTENTION_WEIGHT_OUT_B], 0ull,
        run->job->hidden_width, run->job->token_count, run->phase_low,
        run->phase_output, 1, run->device_status, "cuda.attention.output_b",
        run->failure, run->err);
}
/* Apply optional mHC residual egress to completed core output without host materialization. */
static int attn_envelope_post(attn_run *run) {
    unsigned long long expanded = run->job->residual_expanded_width;
    unsigned long long streams = run->job->residual_stream_count, total;
    unsigned long long width = run->job->residual_stream_width;
    unsigned int grid;
    if (run->job->operation_scope != YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE)
        return YVEX_OK;
    if (run->ordinal) return YVEX_OK;
    if (!expanded || !yvex_core_u64_mul(expanded, run->job->token_count, &total) ||
        total > UINT_MAX * (unsigned long long)YVEX_CUDA_ATTN_BLOCK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.mhc_post", UINT_MAX, run->job->token_count,
            YVEX_ERR_BOUNDS, "CUDA attention mHC egress extent is invalid");
    grid = (unsigned int)((total + YVEX_CUDA_ATTN_BLOCK - 1ull) / YVEX_CUDA_ATTN_BLOCK);
    {
        void *params[] = {
            &run->phase_output, &run->phase_input, &run->phase_mhc_post,
            &run->phase_mhc_combination, &streams, &width, &run->phase_envelope_output,
            (void *)&run->job->token_count, &run->device_status
        };
        return run->ops->launch(
            &run->resources, run->state->residual_mhc_post_function, grid,
            YVEX_CUDA_ATTN_BLOCK, 0u, params,
            "cuda.attention.mhc_post", run->failure, run->err);
    }
}
typedef struct {
    attn_run *run; unsigned int first, last;
} attn_graph_piece;
static int (*const attn_kernel_stages[YVEX_CUDA_ATTENTION_STAGE_COUNT])(attn_run *) = {
    attn_envelope_pre, attn_project, attn_compress, attn_reduce, attn_envelope_post
};
static int attn_stages_execute(attn_run *run, unsigned int first, unsigned int last) {
    unsigned long long token;
    unsigned int stage;
    int rc = YVEX_OK;
    for (stage = first; rc == YVEX_OK && stage < last; ++stage) {
        if (!cuda_attention_piece_active(
                run->job->operation_scope, run->job->attention_class,
                (yvex_cuda_attention_stage)stage))
            continue;
        if (stage == YVEX_CUDA_ATTENTION_STAGE_PROJECT) {
            attn_phase_bind(run, 0ull);
            rc = attn_kernel_stages[stage](run);
            continue;
        }
        if (stage == YVEX_CUDA_ATTENTION_STAGE_COMPRESS &&
            run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA) {
            rc = attn_index_prepare(run);
            if (rc != YVEX_OK) continue;
        }
        for (token = 0ull; rc == YVEX_OK && token < run->job->token_count; ++token) {
            attn_phase_bind(run, token);
            rc = attn_kernel_stages[stage](run);
        }
    }
    return rc;
}
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
    attn_transfer spans[YVEX_CUDA_ATTN_TRANSFERS + 3u] = {
        {&run->device_status, run->staged_status, NULL, 1ull, 1ull, NULL,
         sizeof(*run->staged_status), "cuda.attention.copy.status"}};
    size_t count = 1u, i;
    int rc = run->ops->allocate(
        &run->resources, &run->download_stage, run->download_stage_bytes,
        NULL, 1, "cuda.attention.copy.packed.allocate", run->failure, run->err);
    if (run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA) {
        spans[count++] = (attn_transfer){&run->phase_selected_count, run->staged_selected_count,
            NULL, run->job->token_count, run->job->token_count, NULL,
            sizeof(*run->staged_selected_count), "cuda.attention.copy.selected_count"};
        spans[count++] = (attn_transfer){&run->phase_valid_count, run->staged_candidate_count,
            NULL, run->job->token_count, run->job->token_count, NULL,
            sizeof(*run->staged_candidate_count), "cuda.attention.copy.valid_count"};
    }
    for (i = 0u; i < run->transfer_count; ++i)
        spans[count++] = run->transfers[i];
    for (i = 0u; rc == YVEX_OK && i < count; ++i) {
        attn_transfer *span = &spans[i];
        const unsigned char *host = span->staged ? span->staged : span->output;
        size_t bytes, offset = (size_t)(host - run->host_stage);
        if (!yvex_cuda_work_checked_bytes(span->capacity, span->width, &bytes))
            return attn_run_fail(run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, span->stage,
                                 span->capacity, 0ull, YVEX_ERR_BOUNDS,
                                 "CUDA attention output staging extent overflowed");
        if (offset > run->download_stage_bytes ||
            bytes > run->download_stage_bytes - offset)
            return attn_run_fail(run, YVEX_BACKEND_ATTENTION_FAILURE_COPY, span->stage,
                                 run->download_stage_bytes, offset, YVEX_ERR_BOUNDS,
                                 "CUDA attention packed publication layout drifted");
        rc = attn_device_copy(run, run->download_stage + offset, *span->device,
                              bytes, "cuda.attention.copy.packed.stage");
    }
    if (rc == YVEX_OK)
        rc = run->ops->download(&run->resources, run->host_stage, run->download_stage,
                                run->download_stage_bytes, "cuda.attention.copy.packed",
                                run->failure, run->err);
    return rc;
}
static int attn_graph_enqueue(void *opaque, int enqueue_kernels, yvex_error *err) {
    attn_graph_piece *piece = (attn_graph_piece *)opaque;
    int rc;
    (void)err;
    piece->run->resources.prepare_only = !enqueue_kernels;
    rc = attn_stages_execute(piece->run, piece->first, piece->last);
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
    int measure_device_time = run->job->measure_device_time, rc;
    if (yvex_cuda_attention_graph_key(run->backend, run->job, first, last,
                                      identity, run->err) != YVEX_OK)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.graph.identity", 1ull, 0ull,
            YVEX_ERR_STATE, "CUDA attention graph compatibility identity is incomplete");
    rc = yvex_cuda_graph_execute(run->backend, identity, attn_graph_prepare, attn_graph_enqueue,
        &piece, measure_device_time ? YVEX_CUDA_GRAPH_EXECUTION_MEASURE_DEVICE_TIME :
            YVEX_CUDA_GRAPH_EXECUTION_SHARED_LAUNCH_STREAM |
                YVEX_CUDA_GRAPH_EXECUTION_DEFER_COMPLETION, &info, run->err);
    if (rc != YVEX_OK && run->failure &&
        run->failure->code == YVEX_BACKEND_ATTENTION_FAILURE_NONE) {
        run->failure->code = YVEX_BACKEND_ATTENTION_FAILURE_LAUNCH;
        run->failure->stage = "cuda.attention.graph.execute";
        run->failure->expected = 1ull;
        run->failure->actual = 0ull;
    }
    if (rc == YVEX_OK &&
        (!yvex_core_u64_add(run->device_execution_elapsed_ns,
                            info.last_device_elapsed_ns,
                            &run->device_execution_elapsed_ns)))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.graph.timing", ULLONG_MAX,
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
    static const unsigned int passes[][2] = {
        {0u, YVEX_CUDA_ATTENTION_STAGE_COMPRESS},
        {YVEX_CUDA_ATTENTION_STAGE_COMPRESS, YVEX_CUDA_ATTENTION_STAGE_COMPRESS + 1u},
        {YVEX_CUDA_ATTENTION_STAGE_REDUCE, YVEX_CUDA_ATTENTION_STAGE_COUNT}};
    unsigned int pass;
    int measure_device_time = run->job->measure_device_time, rc = YVEX_OK;
    if (!run->configuration ||
        run->configuration->mode == YVEX_BACKEND_CUDA_ATTENTION_EAGER) {
        if (measure_device_time) rc = yvex_cuda_timing(
                run->backend, yvex_cuda_launch_stream(run->backend), YVEX_CUDA_TIMING_BEGIN, NULL,
                "cuda.attention.eager.timing.begin", run->err);
        for (pass = 0u; rc == YVEX_OK && pass < 3u; ++pass)
            rc = attn_stages_execute(run, passes[pass][0], passes[pass][1]);
        if (rc == YVEX_OK && measure_device_time)
            rc = yvex_cuda_timing(
                run->backend, yvex_cuda_launch_stream(run->backend), YVEX_CUDA_TIMING_FINISH,
                &run->device_execution_elapsed_ns,
                "cuda.attention.eager.timing.finish", run->err);
        else if (rc != YVEX_OK && measure_device_time)
            (void)yvex_cuda_timing(run->backend, NULL, YVEX_CUDA_TIMING_DISCARD,
                                   NULL, NULL, NULL);
        if (rc != YVEX_OK && run->failure &&
            run->failure->code == YVEX_BACKEND_ATTENTION_FAILURE_NONE) {
            run->failure->code = YVEX_BACKEND_ATTENTION_FAILURE_SYNCHRONIZE;
            run->failure->stage = "cuda.attention.eager.timing";
            run->failure->expected = 1ull;
            run->failure->actual = 0ull;
        }
        if (rc == YVEX_OK)
            rc = attn_cancel(
                run, "cuda.attention.cancel.after_request", 1);
        return rc;
    }
    if (run->configuration->mode == YVEX_BACKEND_CUDA_ATTENTION_FULL &&
        !run->job->candidate_block_visible) {
        rc = attn_graph_execute(run, 0u, YVEX_CUDA_ATTENTION_STAGE_COUNT);
        return rc == YVEX_OK
                   ? attn_cancel(
                         run, "cuda.attention.cancel.after_full_graph", 1)
                   : rc;
    }
    if (run->configuration->mode != YVEX_BACKEND_CUDA_ATTENTION_PIECEWISE &&
        !(run->configuration->mode == YVEX_BACKEND_CUDA_ATTENTION_FULL &&
          run->job->candidate_block_visible))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_CAPABILITY,
            "cuda.attention.graph.mode", YVEX_BACKEND_CUDA_ATTENTION_FULL,
            run->configuration->mode, YVEX_ERR_UNSUPPORTED,
            "CUDA attention execution mode is unavailable");
    /* Draft needs every projection before reduction, so candidate capture retains this split. */
    rc = attn_graph_execute(run, 0u, YVEX_CUDA_ATTENTION_STAGE_COMPRESS);
    if (rc == YVEX_OK && run->job->attention_class != YVEX_BACKEND_ATTENTION_SWA)
        rc = attn_graph_execute(
            run, YVEX_CUDA_ATTENTION_STAGE_COMPRESS,
            YVEX_CUDA_ATTENTION_STAGE_COMPRESS + 1u);
    if (rc == YVEX_OK)
        rc = attn_graph_execute(
            run, YVEX_CUDA_ATTENTION_STAGE_REDUCE, YVEX_CUDA_ATTENTION_STAGE_COUNT);
    if (rc == YVEX_OK)
        rc = attn_cancel(
            run, "cuda.attention.cancel.after_graph_piece", 1);
    return rc;
}
static int attn_completion_prepare(attn_run *run)
{
    yvex_backend_host_workspace_summary workspace;
    yvex_backend_attention_completion *completion = run->completion;
    size_t index;
    if (!completion || run->transfer_count > YVEX_BACKEND_ATTENTION_COMPLETION_TRANSFER_CAP)
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
            "cuda.attention.complete.prepare",
            YVEX_BACKEND_ATTENTION_COMPLETION_TRANSFER_CAP,
            run->transfer_count, YVEX_ERR_BOUNDS,
            "CUDA attention completion capacity is invalid");
    completion->output = *run->output;
    completion->output.tokens_executed = run->job->token_count;
    completion->output.compressed_count = run->phase_compressed_count;
    completion->output.indexer_count = run->phase_indexer_count;
    completion->output.host_bytes = 0ull;
    completion->output.peak_host_bytes = run->host_stage_bytes;
    completion->output.device_bytes = 0ull;
    completion->output.peak_device_bytes = run->resources.peak_bytes;
    completion->output.kernel_launches = run->resources.launches;
    completion->output.tensor_core_launches = run->resources.tensor_core_launches;
    completion->output.h2d_bytes = run->h2d_bytes;
    completion->output.d2h_bytes = run->d2h_bytes;
    completion->output.d2d_bytes = run->d2d_bytes;
    completion->output.stream_synchronizations = run->stream_synchronizations;
    completion->output.device_synchronizations = run->device_synchronizations;
    completion->output.device_execution_elapsed_ns = run->device_execution_elapsed_ns;
    if (yvex_backend_host_workspace_summary_get(run->backend, &workspace)) {
        completion->output.host_workspace_capacity = workspace.capacity;
        completion->output.host_workspace_used = workspace.used;
        completion->output.host_workspace_peak = workspace.peak;
        completion->output.host_workspace_allocation_count = workspace.allocation_count;
        completion->output.host_workspace_reused = run->host_workspace_reused;
    }
    completion->host_status = run->staged_status;
    completion->host_selected_counts = run->staged_selected_count;
    completion->host_candidate_counts = run->staged_candidate_count;
    completion->attention_class = run->job->attention_class;
    completion->token_count = run->job->token_count;
    completion->indexer_topk = run->job->indexer_topk;
    completion->candidate_capacity = run->candidate_capacity;
    completion->transfer_count = (unsigned int)run->transfer_count;
    for (index = 0u; index < run->transfer_count; ++index) {
        const attn_transfer *source = &run->transfers[index];
        completion->transfers[index] = (yvex_backend_attention_completion_transfer){
            source->output, source->staged, source->capacity,
            source->output_capacity,
            source->used ? *source->used : source->capacity,
            source->width, source->stage};
    }
    completion->pending = 1;
    return YVEX_OK;
}

static int attn_synchronize(attn_run *run) {
    yvex_cuda_attention_state_sources sources;
    unsigned long long output_elements;
    unsigned long long output_width;
    size_t state_bytes = 0u;
    CUdeviceptr output_source;
    int state_staged = 0, rc = YVEX_OK;
    output_width = run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                       ? run->job->residual_expanded_width : run->job->hidden_width;
    output_source = run->job->operation_scope == YVEX_BACKEND_ATTENTION_SCOPE_ENVELOPE
                        ? run->phase_envelope_output : run->phase_output;
    if (run->job->device_output &&
        !yvex_core_u64_mul(run->job->token_count, output_width, &output_elements))
        return attn_run_fail(
            run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
            "cuda.attention.copy.device-output", ULLONG_MAX,
            run->job->token_count, YVEX_ERR_BOUNDS,
            "CUDA attention device output extent overflowed");
    if (run->job->device_output)
        rc = yvex_cuda_activation_copy(run->backend, output_source,
            run->job->device_output, output_elements,
            "cuda.attention.copy.device-output", run->err);
    if (rc == YVEX_OK && run->job->device_output) {
        size_t bytes;
        if (!yvex_cuda_work_checked_bytes(output_elements, sizeof(float), &bytes))
            return attn_run_fail(
                run, YVEX_BACKEND_ATTENTION_FAILURE_BUDGET,
                "cuda.attention.copy.device-output", ULLONG_MAX,
                run->job->token_count, YVEX_ERR_BOUNDS,
                "CUDA attention device output accounting overflowed");
        rc = attn_account_d2d(run, bytes, "cuda.attention.copy.device-output");
    }
    if (rc == YVEX_OK && run->job->evidence_level == 0u &&
        !run->job->retain_prefix_checkpoints) {
        sources = (yvex_cuda_attention_state_sources){
            run->phase_local, run->phase_local_positions, run->phase_compressed,
            run->phase_compressed_positions, run->phase_indexer,
            run->phase_indexer_positions, run->phase_main_kv, run->phase_main_score,
            run->phase_index_kv, run->phase_index_score, run->initial_local_count,
            run->initial_compressed_count, run->initial_indexer_count,
            run->phase_compressed_count, run->phase_indexer_count, run->local_ring_capacity,
            run->rolling[ROLL_MAIN].extent, run->rolling[ROLL_INDEX].extent};
        rc = run->ops->state_stage(
            run->backend, run->job, &sources, &state_bytes, &state_staged, run->err);
        if (rc == YVEX_OK && state_staged)
            rc = attn_account_d2d(
                run, state_bytes, "cuda.attention.state.stage");
        if (rc == YVEX_OK && state_staged) {
            run->output->device_state_staged = 1;
            run->output->device_state_staged_bytes = (unsigned long long)state_bytes;
        }
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
    if (rc == YVEX_OK) rc = attn_completion_prepare(run);
    if (rc == YVEX_OK && !run->completion->defer)
        rc = yvex_backend_attention_complete(
            run->backend, run->completion, 0, run->err);
    if (rc != YVEX_OK && run->failure && run->completion->failure.code)
        *run->failure = run->completion->failure;
    if (rc == YVEX_OK) *run->output = run->completion->output;
    return rc;
}
static int attn_stage_layout(attn_run *run, unsigned char *base, size_t *total) {
    unsigned long long csa_tokens =
        run->job->attention_class == YVEX_BACKEND_ATTENTION_CSA
            ? run->job->token_count : 0ull;
    return run->ops->stage_layout(
        base, run->uploads, run->upload_count, run->transfers,
        run->transfer_count, csa_tokens, &run->staged_status,
        &run->staged_selected_count, &run->staged_candidate_count, total,
        &run->download_stage_bytes);
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
        actual == run->host_stage_bytes && run->download_stage_bytes) {
        run->d2h_bytes = run->download_stage_bytes;
        return YVEX_OK;
    }
    return attn_run_fail(
        run, YVEX_BACKEND_ATTENTION_FAILURE_ALLOCATION,
        "cuda.attention.host_stage.layout", run->host_stage_bytes,
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
typedef struct {
    int (*execute)(attn_run *run); const char *cancel_stage; int pending_device_work;
} attn_transaction_phase;
static const attn_transaction_phase attn_transaction[] = {
    {attn_stage_allocate, NULL, 0}, {attn_stage_inputs, NULL, 0},
    {attn_allocate_base, "cuda.attention.cancel.after_allocation", 0},
    {attn_numerical_execute, "cuda.attention.cancel.before_commit", 1},
    {attn_synchronize, "cuda.attention.cancel.after_copy", 0}
};
int yvex_backend_attention_execute(yvex_backend *backend, const yvex_backend_attention_job *job,
                                   yvex_backend_attention_output *output,
                                   yvex_backend_attention_failure *failure, yvex_error *err) {
    yvex_backend_attention_completion local_completion = {.stack_start = 1};
    attn_run run = {.backend = backend, .state = yvex_cuda_state(backend),
                    .ops = yvex_cuda_attention_operations_get(), .topk_capacity = 1ull,
                    .output = output, .failure = failure, .err = err, .candidate_capacity = 1ull};
    yvex_error cleanup_error, primary_error;
    yvex_backend_attention_failure primary_failure;
    size_t phase;
    int cleanup_rc, rc;
    if (failure) memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (job) {
        run.request = *job;
        run.job = &run.request;
    }
    run.completion = job && job->device_completion
                         ? job->device_completion : &local_completion;
    if (run.completion != &local_completion) {
        int defer = run.completion->defer;
        int stack_start = run.completion->stack_start;
        if (!defer || run.completion->pending)
            return run.ops->fail(
                failure, YVEX_BACKEND_ATTENTION_FAILURE_INVALID_ARGUMENT,
                "cuda.attention.complete.admit", 1ull, 0ull, err,
                YVEX_ERR_INVALID_ARG,
                "CUDA attention deferred completion is not reusable");
        memset(run.completion, 0, sizeof(*run.completion));
        run.completion->defer = defer;
        run.completion->stack_start = stack_start;
    }
    rc = attn_prepare(&run);
    if (rc != YVEX_OK) return rc;
    backend_workspace_reset(backend);
    if (run.completion->stack_start) backend_host_workspace_reset(backend);
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
            "cuda.attention.cleanup", 0ull,
            run.state->deferred_release_bytes, (yvex_status)cleanup_rc,
            "CUDA attention temporary cleanup failed");
    if (rc == YVEX_OK && !run.completion->defer)
        rc = attn_cancel(&run, "cuda.attention.cancel.publish", 0);
    if (rc != YVEX_OK && run.completion->pending) {
        int completion_rc;
        primary_error = err ? *err : (yvex_error){0};
        primary_failure = failure ? *failure : (yvex_backend_attention_failure){0};
        completion_rc = yvex_backend_attention_complete(
            backend, run.completion, 0, &cleanup_error);
        if (completion_rc != YVEX_OK && !yvex_error_is_set(&primary_error)) {
            if (err) *err = cleanup_error;
            if (failure) *failure = run.completion->failure;
            rc = completion_rc;
        } else {
            if (err) *err = primary_error;
            if (failure) *failure = primary_failure;
        }
    }
    if (rc == YVEX_OK) {
        if (failure) memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}
