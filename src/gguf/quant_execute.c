/*
 * Execute sealed physical decisions through bounded source and sink transactions.
 *
 * Inputs resolve through the immutable binding; no full weight or full terminal is retained;
 * per-terminal sink transactions are exact. This produces writer-ready chunks but never creates a
 * GGUF file.
 */
#include <limits.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>

typedef struct quant_executor quant_executor;

typedef struct {
    yvex_quant_metrics metrics;
    unsigned long long output_chunks;
    unsigned long long encoded_bytes;
    unsigned long long reference_elements;
} quant_terminal_result;

typedef struct {
    quant_executor *executor;
    const yvex_quant_decision *decision;
    const yvex_quant_output_sink *sink;
    unsigned char *output;
    size_t output_capacity;
    size_t output_used;
    unsigned long long output_offset;
    float block[YVEX_QUANT_Q2_K_ELEMENTS];
    unsigned int block_elements;
    unsigned int block_used;
    unsigned long long element_index;
    unsigned long long row_element;
    unsigned long long calibration_entry_ordinal;
    int calibration_bound;
    yvex_quant_metrics metrics;
    unsigned long long output_chunks;
} quant_emitter;

typedef struct {
    int (*consume)(void *context, const unsigned char *bytes, size_t byte_count,
                   unsigned long long logical_offset, yvex_quant_failure *failure, yvex_error *err);
    void *context;
    quant_executor *executor;
    const yvex_quant_decision *decision;
    unsigned long long source_index;
    unsigned long long next_offset;
    int consumer_status;
    yvex_quant_failure *failure;
    yvex_error *err;
} quant_source_sink;

typedef struct {
    quant_executor *executor;
    unsigned char *bytes;
    size_t capacity;
    size_t used;
} quant_collector;

typedef struct {
    quant_emitter *emitter;
    yvex_native_dtype dtype;
    unsigned int scalar_bytes;
    unsigned char partial[8];
    unsigned int partial_used;
} quant_scalar_stream;

typedef struct {
    quant_emitter *emitter;
    yvex_native_dtype dtype;
    unsigned int scalar_bytes;
    unsigned long long source_index;
    unsigned char partial[8];
    unsigned int partial_used;
} quant_exact_copy_stream;

typedef struct {
    quant_emitter *emitter;
    unsigned long long source_index;
    unsigned char partial[8];
    unsigned int partial_used;
} quant_cast_stream;

typedef struct {
    quant_emitter *emitter;
    const unsigned char *scales;
    size_t scale_bytes;
    unsigned long long rows;
    unsigned long long columns;
    unsigned long long scale_rows;
    unsigned long long scale_columns;
    unsigned long long block_rows;
    unsigned long long block_columns;
    unsigned long long element_index;
} quant_scale_pair_stream;

typedef struct {
    quant_emitter *emitter;
    const unsigned char *scales;
    size_t scale_bytes;
    unsigned char packed[16];
    unsigned int packed_used;
    unsigned long long group_index;
} quant_expert_stream;

typedef struct {
    quant_executor *executor;
    unsigned int worker_index;
} quant_worker;

struct quant_executor {
    const yvex_quant_plan *plan;
    const yvex_transform_ir *ir;
    const yvex_transform_binding *binding;
    yvex_source_payload_session *session;
    const yvex_quant_output_sink *sink;
    yvex_quant_executor_options options;
    pthread_mutex_t mutex;
    int mutex_initialized;
    unsigned long long next_terminal;
    atomic_int start;
    atomic_int stop;
    int failure_set;
    yvex_quant_failure failure;
    yvex_error error;
    size_t owned_bytes;
    yvex_quant_execution_summary summary;
};

static void *quant_executor_default_allocate(size_t size, void *context) {
    (void)context;
    return malloc(size);
}

static void quant_executor_default_release(void *allocation, void *context) {
    (void)context;
    free(allocation);
}

static int quant_executor_default_thread_create(pthread_t *thread, void *(*entry)(void *),
                                                void *argument, void *context) {
    (void)context;
    return pthread_create(thread, NULL, entry, argument);
}

static int quant_execute_fail(yvex_quant_failure *failure, yvex_quant_failure_code code,
                              const yvex_quant_decision *decision, unsigned long long source,
                              unsigned long long row, unsigned long long block,
                              unsigned long long expected, unsigned long long actual,
                              yvex_error *err, int status, const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = decision ? decision->terminal_ordinal : ULLONG_MAX;
        failure->source_index = source;
        failure->row_index = row;
        failure->block_index = block;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = decision ? decision->qtype : UINT_MAX;
        failure->operation = decision ? decision->operation : YVEX_TRANSFORM_OP_COUNT;
    }
    yvex_error_set(err, (yvex_status)status, "quant.execute", message);
    return status;
}

/*
 * Prove complete imatrix identity and per-terminal coverage before any payload read.
 *
 * Identity, name, expert geometry, or coverage mismatch refuses the executor.
 */
static int quant_imatrix_plan_validate(const yvex_quant_plan *plan,
                                       const yvex_quant_plan_summary *plan_summary,
                                       const yvex_imatrix_data *imatrix,
                                       yvex_quant_failure *failure, yvex_error *err) {
    yvex_imatrix_data_summary data_summary;
    unsigned long long ordinal;
    int rc;

    if (!plan_summary->calibration_required)
        return YVEX_OK;
    if (!imatrix)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_REQUIRED, NULL,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
                                  YVEX_ERR_INVALID_ARG,
                                  "calibrated physical plan requires admitted imatrix data");
    rc = yvex_imatrix_data_validate(imatrix, err);
    if (rc == YVEX_OK)
        rc = yvex_imatrix_data_get_summary(imatrix, &data_summary, err);
    if (rc != YVEX_OK || !data_summary.complete ||
        strcmp(data_summary.imatrix_identity, plan_summary->imatrix_identity) != 0)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY, NULL,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
                                  YVEX_ERR_FORMAT,
                                  "imatrix identity does not match the sealed physical plan");
    for (ordinal = 0u; ordinal < plan_summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(plan, ordinal);
        yvex_imatrix_entry_summary entry = {0};
        unsigned long long expected = 0u;

        if (!decision || !decision->policy_requires_imatrix)
            continue;
        if (!decision->physical_tensor_name[0] || decision->physical_expert_count == 0u ||
            decision->row_count % decision->physical_expert_count != 0u ||
            !yvex_core_u64_mul(decision->physical_expert_count, decision->row_width, &expected) ||
            yvex_imatrix_data_find(imatrix, decision->physical_tensor_name, &entry, err) !=
                YVEX_OK ||
            entry.value_count != expected)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY, decision, ULLONG_MAX,
                ULLONG_MAX, ULLONG_MAX, expected, entry.value_count, err, YVEX_ERR_FORMAT,
                "imatrix entry does not cover the physical expert-row geometry");
    }
    return YVEX_OK;
}

static int quant_cancellation_requested(const yvex_quant_cancellation *cancellation) {
    return cancellation &&
           atomic_load_explicit(&cancellation->requested, memory_order_acquire) != 0;
}

void yvex_quant_executor_options_default(yvex_quant_executor_options *options) {
    if (!options)
        return;
    memset(options, 0, sizeof(*options));
    options->worker_count = 1u;
    options->source_chunk_bytes = YVEX_SOURCE_PAYLOAD_DEFAULT_CHUNK_BYTES;
    options->output_chunk_bytes = 1024u * 1024u;
    options->maximum_owned_bytes = 64u * 1024u * 1024u;
    options->allocate = quant_executor_default_allocate;
    options->release = quant_executor_default_release;
    options->thread_create = quant_executor_default_thread_create;
}

static int quant_executor_cancelled(const quant_executor *executor) {
    return quant_cancellation_requested(executor->options.cancellation);
}

static int quant_executor_stopping(const quant_executor *executor) {
    return atomic_load_explicit(&executor->stop, memory_order_acquire) ||
           quant_executor_cancelled(executor);
}

static unsigned long long quant_scaled_fact(double value) {
    double scaled;

    if (!(value > 0.0))
        return 0u;
    scaled = value * 1000000000.0;
    if (!isfinite(scaled) || scaled >= (double)ULLONG_MAX)
        return ULLONG_MAX;
    return (unsigned long long)scaled;
}

static void *quant_executor_allocate(quant_executor *executor, size_t bytes, int *budget_exceeded) {
    void *allocation;

    if (budget_exceeded)
        *budget_exceeded = 0;
    if (!bytes)
        return NULL;
    pthread_mutex_lock(&executor->mutex);
    if (bytes > executor->options.maximum_owned_bytes ||
        executor->owned_bytes > executor->options.maximum_owned_bytes - bytes) {
        if (budget_exceeded)
            *budget_exceeded = 1;
        pthread_mutex_unlock(&executor->mutex);
        return NULL;
    }
    executor->owned_bytes += bytes;
    if (executor->owned_bytes > executor->summary.peak_owned_bytes)
        executor->summary.peak_owned_bytes = executor->owned_bytes;
    pthread_mutex_unlock(&executor->mutex);
    allocation = executor->options.allocate(bytes, executor->options.context);
    if (!allocation) {
        pthread_mutex_lock(&executor->mutex);
        executor->owned_bytes -= bytes;
        pthread_mutex_unlock(&executor->mutex);
    }
    return allocation;
}

static void quant_executor_release(quant_executor *executor, void *allocation, size_t bytes) {
    if (!allocation)
        return;
    executor->options.release(allocation, executor->options.context);
    pthread_mutex_lock(&executor->mutex);
    if (executor->owned_bytes >= bytes)
        executor->owned_bytes -= bytes;
    else
        executor->owned_bytes = 0u;
    pthread_mutex_unlock(&executor->mutex);
}

static unsigned int quant_source_scalar_bytes(yvex_native_dtype dtype) {
    switch (dtype) {
    case YVEX_NATIVE_DTYPE_F32:
    case YVEX_NATIVE_DTYPE_I32:
        return 4u;
    case YVEX_NATIVE_DTYPE_F16:
    case YVEX_NATIVE_DTYPE_BF16:
        return 2u;
    case YVEX_NATIVE_DTYPE_F8_E4M3:
    case YVEX_NATIVE_DTYPE_F8_E8M0:
    case YVEX_NATIVE_DTYPE_I8:
        return 1u;
    case YVEX_NATIVE_DTYPE_I64:
        return 8u;
    default:
        return 0u;
    }
}

static int quant_source_begin(void *opaque, const yvex_source_payload_plan_summary *summary) {
    quant_source_sink *sink = (quant_source_sink *)opaque;
    return !sink || !summary || summary->range_count != 1u;
}

/*
 * Forward one exact monotonic source chunk to a transformation consumer.
 *
 * Records cancellation, peer stop, callback refusal, or offset overflow.
 */
static int quant_source_chunk(void *opaque, const yvex_source_payload_chunk *chunk,
                              const unsigned char *bytes) {
    quant_source_sink *sink = (quant_source_sink *)opaque;

    if (!sink || !chunk || !bytes || chunk->byte_length == 0u ||
        chunk->logical_offset != sink->next_offset)
        return 1;
    if (quant_executor_stopping(sink->executor)) {
        int cancelled = quant_executor_cancelled(sink->executor);
        sink->consumer_status = quant_execute_fail(
            sink->failure, cancelled ? YVEX_QUANT_FAILURE_CANCELLED : YVEX_QUANT_FAILURE_WORKER,
            sink->decision, sink->source_index, ULLONG_MAX, ULLONG_MAX, 0u, sink->next_offset,
            sink->err, YVEX_ERR_STATE,
            cancelled ? "quantization execution was cancelled between source chunks"
                      : "quantization source delivery stopped after a peer failure");
        return 1;
    }
    sink->consumer_status = sink->consume(sink->context, bytes, chunk->byte_length,
                                          chunk->logical_offset, sink->failure, sink->err);
    if (sink->consumer_status != YVEX_OK)
        return 1;
    if (!yvex_core_u64_add(sink->next_offset, chunk->byte_length, &sink->next_offset))
        return 1;
    return 0;
}

static int quant_source_commit(void *opaque, const yvex_source_payload_stream_result *result) {
    quant_source_sink *sink = (quant_source_sink *)opaque;
    return !sink || !result || !result->complete || result->aborted ||
           sink->consumer_status != YVEX_OK || sink->next_offset != result->delivered_logical_bytes;
}

static void quant_source_abort(void *opaque, const yvex_source_payload_failure *failure,
                               const yvex_source_payload_stream_result *result) {
    (void)opaque;
    (void)failure;
    (void)result;
}

static int quant_read_source(quant_executor *executor, const yvex_quant_decision *decision,
                             unsigned long long source_index,
                             int (*consume)(void *context, const unsigned char *bytes,
                                            size_t byte_count, unsigned long long logical_offset,
                                            yvex_quant_failure *failure, yvex_error *err),
                             void *context, yvex_quant_failure *failure, yvex_error *err,
                             unsigned long long *bytes_read, unsigned long long *chunks_read) {
    const yvex_source_payload_range *range =
        yvex_transform_binding_range_at(executor->binding, source_index);
    yvex_source_payload_plan *plan = NULL;
    yvex_source_payload_failure payload_failure;
    yvex_source_payload_stream_result result;
    yvex_source_payload_sink payload_sink;
    quant_source_sink source_sink;
    unsigned long long tensor_index;
    int rc;

    if (bytes_read)
        *bytes_read = 0u;
    if (chunks_read)
        *chunks_read = 0u;
    if (!range || !consume)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, decision,
                                  source_index, ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
                                  YVEX_ERR_INVALID_ARG,
                                  "bound source range and consumer are required");
    if (quant_executor_stopping(executor)) {
        int cancelled = quant_executor_cancelled(executor);
        return quant_execute_fail(
            failure, cancelled ? YVEX_QUANT_FAILURE_CANCELLED : YVEX_QUANT_FAILURE_WORKER, decision,
            source_index, ULLONG_MAX, ULLONG_MAX, 0u, 0u, err, YVEX_ERR_STATE,
            cancelled ? "quantization execution was cancelled"
                      : "quantization source read stopped after a peer failure");
    }
    tensor_index = range->source_tensor_index;
    rc = yvex_source_payload_plan_build(&plan, executor->session, &tensor_index, 1u,
                                        executor->options.source_chunk_bytes, 4096u,
                                        &payload_failure, err);
    if (rc != YVEX_OK)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_PAYLOAD_NOT_READABLE, decision,
                                  source_index, ULLONG_MAX, ULLONG_MAX, 1u, payload_failure.code,
                                  err, rc, "source range plan construction failed");
    memset(&source_sink, 0, sizeof(source_sink));
    source_sink.consume = consume;
    source_sink.context = context;
    source_sink.executor = executor;
    source_sink.decision = decision;
    source_sink.source_index = source_index;
    source_sink.failure = failure;
    source_sink.err = err;
    memset(&payload_sink, 0, sizeof(payload_sink));
    payload_sink.begin = quant_source_begin;
    payload_sink.chunk = quant_source_chunk;
    payload_sink.commit = quant_source_commit;
    payload_sink.abort = quant_source_abort;
    payload_sink.context = &source_sink;
    memset(&result, 0, sizeof(result));
    rc = yvex_source_payload_session_stream(executor->session, plan, &payload_sink, &result,
                                            &payload_failure, err);
    yvex_source_payload_plan_close(plan);
    if (bytes_read)
        *bytes_read = result.physical_bytes_read;
    if (chunks_read)
        *chunks_read = result.chunks_completed;
    if (source_sink.consumer_status != YVEX_OK) {
        if (failure) {
            if (failure->terminal_ordinal == ULLONG_MAX)
                failure->terminal_ordinal = decision->terminal_ordinal;
            if (failure->source_index == ULLONG_MAX)
                failure->source_index = source_index;
            if (failure->qtype == UINT_MAX)
                failure->qtype = decision->qtype;
            if (failure->operation == YVEX_TRANSFORM_OP_COUNT)
                failure->operation = decision->operation;
        }
        return source_sink.consumer_status;
    }
    if (rc != YVEX_OK) {
        yvex_quant_failure_code code =
            payload_failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHORT_READ
                ? YVEX_QUANT_FAILURE_SOURCE_SHORT_READ
            : payload_failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_SHARD_DRIFT
                ? YVEX_QUANT_FAILURE_SOURCE_DRIFT
            : payload_failure.code == YVEX_SOURCE_PAYLOAD_FAILURE_CANCELLED
                ? YVEX_QUANT_FAILURE_CANCELLED
                : YVEX_QUANT_FAILURE_PAYLOAD_NOT_READABLE;
        return quant_execute_fail(failure, code, decision, source_index, ULLONG_MAX, ULLONG_MAX,
                                  range->byte_length, result.delivered_logical_bytes, err, rc,
                                  "trusted source range delivery failed");
    }
    if (!result.complete || !result.committed || result.aborted ||
        result.delivered_logical_bytes != range->byte_length)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SOURCE_SHORT_READ, decision,
                                  source_index, ULLONG_MAX, ULLONG_MAX, range->byte_length,
                                  result.delivered_logical_bytes, err, YVEX_ERR_IO,
                                  "source range did not complete exactly");
    return YVEX_OK;
}

static int quant_collect_chunk(void *opaque, const unsigned char *bytes, size_t byte_count,
                               unsigned long long logical_offset, yvex_quant_failure *failure,
                               yvex_error *err) {
    quant_collector *collector = (quant_collector *)opaque;

    (void)failure;
    (void)err;
    if (!collector || !bytes || logical_offset != collector->used ||
        byte_count > collector->capacity - collector->used)
        return YVEX_ERR_BOUNDS;
    memcpy(collector->bytes + collector->used, bytes, byte_count);
    collector->used += byte_count;
    return YVEX_OK;
}

static int quant_collect_source(quant_executor *executor, const yvex_quant_decision *decision,
                                unsigned long long source_index, unsigned char **out,
                                size_t *out_bytes, yvex_quant_failure *failure, yvex_error *err,
                                unsigned long long *payload_bytes, unsigned long long *chunks) {
    const yvex_source_payload_range *range =
        yvex_transform_binding_range_at(executor->binding, source_index);
    quant_collector collector;
    int budget_exceeded = 0;
    int rc;

    *out = NULL;
    *out_bytes = 0u;
    if (!range || range->byte_length == 0u || range->byte_length > SIZE_MAX)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, decision, source_index,
                                  ULLONG_MAX, ULLONG_MAX, SIZE_MAX, range ? range->byte_length : 0u,
                                  err, YVEX_ERR_BOUNDS,
                                  "collected side-input byte count is not representable");
    memset(&collector, 0, sizeof(collector));
    collector.executor = executor;
    collector.capacity = (size_t)range->byte_length;
    collector.bytes =
        (unsigned char *)quant_executor_allocate(executor, collector.capacity, &budget_exceeded);
    if (!collector.bytes)
        return quant_execute_fail(
            failure,
            budget_exceeded ? YVEX_QUANT_FAILURE_RESOURCE_BUDGET : YVEX_QUANT_FAILURE_ALLOCATION,
            decision, source_index, ULLONG_MAX, ULLONG_MAX, collector.capacity,
            executor->options.maximum_owned_bytes, err, YVEX_ERR_NOMEM,
            budget_exceeded ? "side-input allocation exceeded the executor budget"
                            : "side-input allocation failed");
    rc = quant_read_source(executor, decision, source_index, quant_collect_chunk, &collector,
                           failure, err, payload_bytes, chunks);
    if (rc != YVEX_OK || collector.used != collector.capacity) {
        quant_executor_release(executor, collector.bytes, collector.capacity);
        if (rc != YVEX_OK)
            return rc;
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SOURCE_SHORT_READ, decision,
                                  source_index, ULLONG_MAX, ULLONG_MAX, collector.capacity,
                                  collector.used, err, YVEX_ERR_IO,
                                  "collected source side input is incomplete");
    }
    *out = collector.bytes;
    *out_bytes = collector.capacity;
    return YVEX_OK;
}

/*
 * Deliver the current output chunk exactly once.
 *
 * Preserves abortable state on cancellation, overflow, or sink refusal. Performs bounded sink
 * delivery but never commits a terminal.
 */
static int quant_emitter_flush(quant_emitter *emitter, yvex_quant_failure *failure,
                               yvex_error *err) {
    if (emitter->output_used == 0u)
        return YVEX_OK;
    if (quant_executor_stopping(emitter->executor)) {
        int cancelled = quant_executor_cancelled(emitter->executor);
        return quant_execute_fail(
            failure, cancelled ? YVEX_QUANT_FAILURE_CANCELLED : YVEX_QUANT_FAILURE_WORKER,
            emitter->decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 0u, emitter->output_offset, err,
            YVEX_ERR_STATE,
            cancelled ? "quantization execution was cancelled before output delivery"
                      : "quantization output stopped before delivery after a peer failure");
    }
    if (emitter->output_chunks == ULLONG_MAX)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  emitter->output_chunks, err, YVEX_ERR_BOUNDS,
                                  "encoded output chunk accounting overflowed");
    if (!emitter->sink->deliver_chunk ||
        emitter->sink->deliver_chunk(emitter->sink->context, emitter->decision,
                                     emitter->output_offset, emitter->output,
                                     emitter->output_used) != 0)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SINK_SHORT_WRITE, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, emitter->output_used, 0u, err,
                                  YVEX_ERR_IO, "quantized output sink refused a bounded chunk");
    if (!yvex_core_u64_add(emitter->output_offset, emitter->output_used, &emitter->output_offset))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  emitter->output_offset, err, YVEX_ERR_BOUNDS,
                                  "encoded output offset overflowed");
    emitter->output_used = 0u;
    emitter->output_chunks++;
    if (quant_executor_stopping(emitter->executor)) {
        int cancelled = quant_executor_cancelled(emitter->executor);
        return quant_execute_fail(
            failure, cancelled ? YVEX_QUANT_FAILURE_CANCELLED : YVEX_QUANT_FAILURE_WORKER,
            emitter->decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 0u, emitter->output_offset, err,
            YVEX_ERR_STATE,
            cancelled ? "quantization execution was cancelled after bounded output delivery"
                      : "quantization output stopped after delivery on a peer failure");
    }
    return YVEX_OK;
}

static int quant_emitter_append(quant_emitter *emitter, const unsigned char *bytes,
                                size_t byte_count, yvex_quant_failure *failure, yvex_error *err) {
    int rc;

    if (!bytes || byte_count == 0u || byte_count > emitter->output_capacity)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, emitter->output_capacity,
                                  byte_count, err, YVEX_ERR_BOUNDS,
                                  "encoded block exceeds the bounded output chunk");
    if (emitter->output_used > emitter->output_capacity - byte_count) {
        rc = quant_emitter_flush(emitter, failure, err);
        if (rc != YVEX_OK)
            return rc;
    }
    memcpy(emitter->output + emitter->output_used, bytes, byte_count);
    emitter->output_used += byte_count;
    return YVEX_OK;
}

static int quant_emitter_bound(const quant_emitter *emitter, const float *reference,
                               const float *reconstructed, const unsigned char *encoded,
                               yvex_quant_failure *failure, yvex_error *err) {
    double maximum_error = 0.0;
    double squared = 0.0;
    double reference_squared = 0.0;
    unsigned int index;

    for (index = 0u; index < emitter->block_elements; ++index) {
        double absolute = fabs((double)reconstructed[index] - reference[index]);
        double magnitude = fabs((double)reference[index]);
        if (absolute > maximum_error)
            maximum_error = absolute;
        squared += absolute * absolute;
        reference_squared += magnitude * magnitude;
    }
    if (emitter->decision->qtype == YVEX_GGUF_QTYPE_Q8_0) {
        unsigned short bits = (unsigned short)encoded[0] | ((unsigned short)encoded[1] << 8);
        double scale = yvex_quant_f16_decode(bits);
        if (!isfinite(scale) || maximum_error > scale * 1.01 + 1e-7)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_NUMERIC_BOUND, emitter->decision, ULLONG_MAX,
                (emitter->element_index - 1u) / emitter->decision->row_width,
                emitter->element_index / emitter->block_elements - 1u, quant_scaled_fact(scale),
                quant_scaled_fact(maximum_error), err, YVEX_ERR_FORMAT,
                "Q8_0 reconstruction exceeded its block-scale error bound");
    } else if (emitter->decision->qtype == YVEX_GGUF_QTYPE_Q2_K) {
        double rmse = sqrt(squared / emitter->block_elements);
        double reference_rms = sqrt(reference_squared / emitter->block_elements);
        float global_scale =
            yvex_quant_f16_decode((unsigned short)encoded[80] | ((unsigned short)encoded[81] << 8));
        float global_minimum =
            yvex_quant_f16_decode((unsigned short)encoded[82] | ((unsigned short)encoded[83] << 8));

        for (index = 0u; index < emitter->block_elements; ++index) {
            unsigned int subblock = index / 16u;
            float scale = global_scale * (float)(encoded[subblock] & 0x0fu);
            float minimum = global_minimum * (float)(encoded[subblock] >> 4);
            double low = -(double)minimum;
            double high = 3.0 * (double)scale - (double)minimum;
            double value = reference[index];
            double allowed = value < low    ? low - value
                             : value > high ? value - high
                                            : (double)scale * 0.5;
            double actual = fabs((double)reconstructed[index] - value);
            double rounding_slack = 1e-6 * (1.0 + fabs(value) + fabs((double)reconstructed[index]));

            if (actual > allowed + rounding_slack)
                return quant_execute_fail(
                    failure, YVEX_QUANT_FAILURE_NUMERIC_BOUND, emitter->decision, ULLONG_MAX,
                    (emitter->element_index - 1u) / emitter->decision->row_width,
                    emitter->element_index / emitter->block_elements - 1u,
                    quant_scaled_fact(allowed + rounding_slack), quant_scaled_fact(actual), err,
                    YVEX_ERR_FORMAT, "Q2_K code is not the nearest bounded affine value");
        }
        if (rmse > reference_rms + 1e-6)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_NUMERIC_BOUND, emitter->decision, ULLONG_MAX,
                (emitter->element_index - 1u) / emitter->decision->row_width,
                emitter->element_index / emitter->block_elements - 1u,
                quant_scaled_fact(reference_rms + 1e-6), quant_scaled_fact(rmse), err,
                YVEX_ERR_FORMAT, "Q2_K block is not better than the zero reconstruction baseline");
    }
    return YVEX_OK;
}

static int quant_emitter_encode(quant_emitter *emitter, yvex_quant_failure *failure,
                                yvex_error *err) {
    unsigned char encoded[YVEX_QUANT_Q2_K_BYTES];
    float reconstructed[YVEX_QUANT_Q2_K_ELEMENTS];
    float calibration[YVEX_QUANT_Q2_K_ELEMENTS];
    const float *weights = NULL;
    size_t encoded_bytes = 0u;
    int rc;

    if (emitter->calibration_bound) {
        unsigned long long block_start = emitter->element_index - emitter->block_elements;
        unsigned long long row = block_start / emitter->decision->row_width;
        unsigned long long rows_per_expert =
            emitter->decision->row_count / emitter->decision->physical_expert_count;
        unsigned long long expert = row / rows_per_expert;
        unsigned long long value_offset = expert * emitter->decision->row_width +
                                          block_start % emitter->decision->row_width;
        rc = yvex_imatrix_data_read(emitter->executor->options.imatrix,
                                    emitter->calibration_entry_ordinal, value_offset,
                                    calibration, emitter->block_elements, err);
        if (rc != YVEX_OK)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY,
                                      emitter->decision, ULLONG_MAX, row,
                                      block_start / emitter->block_elements, value_offset, 0u,
                                      err, rc, "imatrix block read violated sealed coverage");
        weights = calibration;
    }
    rc = weights ? yvex_quant_encode_block_weighted(
                       emitter->decision->qtype, emitter->block, weights,
                       emitter->block_elements, encoded, sizeof(encoded), &encoded_bytes,
                       failure, err)
                 : yvex_quant_encode_block(emitter->decision->qtype, emitter->block,
                                           emitter->block_elements, encoded, sizeof(encoded),
                                           &encoded_bytes, failure, err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_quant_decode_block(emitter->decision->qtype, encoded, encoded_bytes, reconstructed,
                                 emitter->block_elements, failure, err);
    if (rc != YVEX_OK)
        return rc;
    rc = quant_emitter_bound(emitter, emitter->block, reconstructed, encoded, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (!yvex_quant_metrics_update(&emitter->metrics, emitter->block, reconstructed, weights,
                                   emitter->block_elements))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  emitter->metrics.element_count, err, YVEX_ERR_BOUNDS,
                                  "numeric metric accounting overflowed");
    return quant_emitter_append(emitter, encoded, encoded_bytes, failure, err);
}

static int quant_emitter_value(quant_emitter *emitter, float value, yvex_quant_failure *failure,
                               yvex_error *err) {
    int rc;

    if (emitter->element_index >= emitter->decision->element_count ||
        emitter->row_element >= emitter->decision->row_width)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW, emitter->decision, ULLONG_MAX, ULLONG_MAX,
            ULLONG_MAX, emitter->decision->element_count, emitter->element_index, err,
            YVEX_ERR_BOUNDS, "transformed element exceeded the sealed terminal geometry");
    if (!isfinite(value))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_NONFINITE, emitter->decision,
                                  ULLONG_MAX, emitter->element_index / emitter->decision->row_width,
                                  emitter->element_index / emitter->block_elements, 0u,
                                  emitter->element_index, err, YVEX_ERR_FORMAT,
                                  "non-finite transformed value is forbidden by the profile");
    emitter->block[emitter->block_used++] = value;
    emitter->element_index++;
    emitter->row_element++;
    if (emitter->block_used == emitter->block_elements) {
        rc = quant_emitter_encode(emitter, failure, err);
        if (rc != YVEX_OK)
            return rc;
        emitter->block_used = 0u;
    }
    if (emitter->row_element == emitter->decision->row_width) {
        if (emitter->block_used != 0u)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_ROW_DIVISIBILITY, emitter->decision, ULLONG_MAX,
                emitter->element_index / emitter->decision->row_width - 1u, ULLONG_MAX,
                emitter->block_elements, emitter->block_used, err, YVEX_ERR_BOUNDS,
                "qtype block crossed a physical row boundary");
        emitter->row_element = 0u;
    }
    return YVEX_OK;
}

/* Begin one terminal transaction with bounded staging. */
static int quant_emitter_open(quant_emitter *emitter, quant_executor *executor,
                              const yvex_quant_decision *decision, yvex_quant_failure *failure,
                              yvex_error *err) {
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(decision->qtype);
    int budget_exceeded = 0;

    memset(emitter, 0, sizeof(*emitter));
    emitter->executor = executor;
    emitter->decision = decision;
    emitter->sink = executor->sink;
    if (!geometry || !geometry->block_size || geometry->block_size > YVEX_QUANT_Q2_K_ELEMENTS)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNKNOWN_QTYPE, decision, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, YVEX_QUANT_Q2_K_ELEMENTS,
                                  geometry ? geometry->block_size : 0u, err, YVEX_ERR_UNSUPPORTED,
                                  "selected qtype has no executable block geometry");
    emitter->block_elements = geometry->block_size;
    if (decision->policy_requires_imatrix) {
        yvex_imatrix_entry_summary entry;
        int rc = yvex_imatrix_data_find(executor->options.imatrix,
                                        decision->physical_tensor_name, &entry, err);
        if (rc != YVEX_OK)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_CALIBRATION_IDENTITY,
                                      decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 1u, 0u,
                                      err, rc, "sealed imatrix entry disappeared before execution");
        emitter->calibration_entry_ordinal = entry.ordinal;
        emitter->calibration_bound = 1;
    }
    emitter->output_capacity = executor->options.output_chunk_bytes;
    emitter->output = (unsigned char *)quant_executor_allocate(executor, emitter->output_capacity,
                                                               &budget_exceeded);
    if (!emitter->output)
        return quant_execute_fail(
            failure,
            budget_exceeded ? YVEX_QUANT_FAILURE_RESOURCE_BUDGET : YVEX_QUANT_FAILURE_ALLOCATION,
            decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, emitter->output_capacity,
            executor->options.maximum_owned_bytes, err,
            budget_exceeded ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            budget_exceeded ? "bounded output buffer exceeded the executor budget"
                            : "bounded output buffer allocation failed");
    yvex_quant_metrics_init(&emitter->metrics);
    if (!emitter->sink->begin_terminal ||
        emitter->sink->begin_terminal(emitter->sink->context, decision) != 0) {
        quant_executor_release(executor, emitter->output, emitter->output_capacity);
        emitter->output = NULL;
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SINK_PROTOCOL, decision, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, 1u, 0u, err, YVEX_ERR_IO,
                                  "output sink refused terminal begin");
    }
    return YVEX_OK;
}

static void quant_emitter_close(quant_emitter *emitter) {
    if (!emitter)
        return;
    quant_executor_release(emitter->executor, emitter->output, emitter->output_capacity);
    emitter->output = NULL;
}

/*
 * Finish exact terminal delivery and commit its sink transaction.
 *
 * Refuses incomplete geometry, byte mismatch, flush failure, or commit refusal.
 */
static int quant_emitter_finish(quant_emitter *emitter, quant_terminal_result *result,
                                yvex_quant_failure *failure, yvex_error *err) {
    int rc;

    if (emitter->block_used != 0u || emitter->row_element != 0u ||
        emitter->element_index != emitter->decision->element_count)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, emitter->decision,
                                  ULLONG_MAX, emitter->element_index / emitter->decision->row_width,
                                  ULLONG_MAX, emitter->decision->element_count,
                                  emitter->element_index, err, YVEX_ERR_FORMAT,
                                  "transformed terminal element accounting is incomplete");
    rc = quant_emitter_flush(emitter, failure, err);
    if (rc != YVEX_OK)
        return rc;
    if (emitter->output_offset != emitter->decision->encoded_bytes)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_SINK_SHORT_WRITE, emitter->decision, ULLONG_MAX, ULLONG_MAX,
            ULLONG_MAX, emitter->decision->encoded_bytes, emitter->output_offset, err, YVEX_ERR_IO,
            "terminal encoded byte count differs from its sealed decision");
    if (!emitter->sink->commit_terminal ||
        emitter->sink->commit_terminal(emitter->sink->context, emitter->decision,
                                       emitter->output_offset) != 0)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SINK_PROTOCOL, emitter->decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, emitter->output_offset, 0u,
                                  err, YVEX_ERR_IO, "output sink refused exact terminal commit");
    memset(result, 0, sizeof(*result));
    result->metrics = emitter->metrics;
    result->output_chunks = emitter->output_chunks;
    result->encoded_bytes = emitter->output_offset;
    result->reference_elements = emitter->metrics.element_count;
    return YVEX_OK;
}

static int quant_scalar_consume(void *opaque, const unsigned char *bytes, size_t byte_count,
                                unsigned long long logical_offset, yvex_quant_failure *failure,
                                yvex_error *err) {
    quant_scalar_stream *stream = (quant_scalar_stream *)opaque;
    size_t offset = 0u;

    (void)logical_offset;
    while (offset < byte_count) {
        size_t take = stream->scalar_bytes - stream->partial_used;
        float value;
        int rc;

        if (take > byte_count - offset)
            take = byte_count - offset;
        memcpy(stream->partial + stream->partial_used, bytes + offset, take);
        stream->partial_used += (unsigned int)take;
        offset += take;
        if (stream->partial_used != stream->scalar_bytes)
            continue;
        rc = yvex_quant_source_scalar_decode(stream->dtype, stream->partial, &value, failure, err);
        if (rc != YVEX_OK)
            return rc;
        rc = quant_emitter_value(stream->emitter, value, failure, err);
        if (rc != YVEX_OK)
            return rc;
        stream->partial_used = 0u;
    }
    return YVEX_OK;
}

/*
 * Validate and forward a complete exact-scalar batch.
 *
 * Refuses invalid scalar data, overflow, or sink rejection.
 */
static int quant_exact_copy_batch(quant_exact_copy_stream *stream, const unsigned char *bytes,
                                  unsigned long long elements, yvex_quant_failure *failure,
                                  yvex_error *err) {
    unsigned long long byte_count;
    unsigned long long index;
    int rc;

    if (!stream || !bytes || !elements ||
        !yvex_core_u64_mul(elements, stream->scalar_bytes, &byte_count) || byte_count > SIZE_MAX ||
        stream->emitter->element_index > stream->emitter->decision->element_count ||
        elements > stream->emitter->decision->element_count - stream->emitter->element_index)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW, stream ? stream->emitter->decision : NULL,
            stream ? stream->source_index : ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
            stream ? stream->emitter->decision->element_count : 0u,
            stream ? stream->emitter->element_index + elements : elements, err, YVEX_ERR_BOUNDS,
            "exact scalar batch exceeds the terminal element geometry");
    for (index = 0u; index < elements; ++index) {
        float value;
        yvex_quant_failure scalar_failure;
        yvex_error scalar_error;

        rc = yvex_quant_source_scalar_decode(stream->dtype, bytes + index * stream->scalar_bytes,
                                             &value, &scalar_failure, &scalar_error);
        if (rc != YVEX_OK)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION,
                                      stream->emitter->decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                      1u, stream->dtype, err, rc,
                                      "exact scalar source dtype has no canonical decoder");
        if (!isfinite(value))
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_NONFINITE, stream->emitter->decision,
                stream->source_index,
                (stream->emitter->element_index + index) / stream->emitter->decision->row_width,
                ULLONG_MAX, 0u, stream->emitter->element_index + index, err, YVEX_ERR_FORMAT,
                "non-finite exact scalar is forbidden by the profile");
    }
    if (stream->emitter->metrics.element_count > ULLONG_MAX - elements ||
        stream->emitter->metrics.finite_count > ULLONG_MAX - elements)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW,
                                  stream->emitter->decision, stream->source_index, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, stream->emitter->metrics.element_count,
                                  err, YVEX_ERR_BOUNDS, "exact scalar accounting overflowed");
    rc = quant_emitter_append(stream->emitter, bytes, (size_t)byte_count, failure, err);
    if (rc != YVEX_OK)
        return rc;
    stream->emitter->metrics.element_count += elements;
    stream->emitter->metrics.finite_count += elements;
    stream->emitter->element_index += elements;
    {
        unsigned long long row_width = stream->emitter->decision->row_width;
        unsigned long long remainder = elements % row_width;
        stream->emitter->row_element = remainder >= row_width - stream->emitter->row_element
                                           ? remainder - (row_width - stream->emitter->row_element)
                                           : stream->emitter->row_element + remainder;
    }
    return YVEX_OK;
}

static int quant_exact_copy_consume(void *opaque, const unsigned char *bytes, size_t byte_count,
                                    unsigned long long logical_offset, yvex_quant_failure *failure,
                                    yvex_error *err) {
    quant_exact_copy_stream *stream = (quant_exact_copy_stream *)opaque;
    size_t offset = 0u;
    unsigned long long expected_offset = 0u;
    int rc;

    if (!stream || !bytes || stream->scalar_bytes == 0u ||
        stream->scalar_bytes > sizeof(stream->partial) ||
        !yvex_core_u64_mul(stream->emitter->element_index, stream->scalar_bytes, &expected_offset) ||
        !yvex_core_u64_add(expected_offset, stream->partial_used, &expected_offset) ||
        logical_offset != expected_offset)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SOURCE_SHORT_READ,
                                  stream ? stream->emitter->decision : NULL, ULLONG_MAX, ULLONG_MAX,
                                  ULLONG_MAX, expected_offset, logical_offset, err, YVEX_ERR_IO,
                                  "exact scalar source chunks are not logically contiguous");
    if (stream->partial_used != 0u) {
        size_t take = stream->scalar_bytes - stream->partial_used;
        if (take > byte_count)
            take = byte_count;
        memcpy(stream->partial + stream->partial_used, bytes, take);
        stream->partial_used += (unsigned int)take;
        offset += take;
        if (stream->partial_used == stream->scalar_bytes) {
            rc = quant_exact_copy_batch(stream, stream->partial, 1u, failure, err);
            if (rc != YVEX_OK)
                return rc;
            stream->partial_used = 0u;
        }
    }
    while (byte_count - offset >= stream->scalar_bytes) {
        size_t take = byte_count - offset;
        size_t maximum = stream->emitter->output_capacity -
                         stream->emitter->output_capacity % stream->scalar_bytes;
        unsigned long long elements;

        if (take > maximum)
            take = maximum;
        take -= take % stream->scalar_bytes;
        elements = take / stream->scalar_bytes;
        rc = quant_exact_copy_batch(stream, bytes + offset, elements, failure, err);
        if (rc != YVEX_OK)
            return rc;
        offset += take;
    }
    if (offset < byte_count) {
        stream->partial_used = (unsigned int)(byte_count - offset);
        memcpy(stream->partial, bytes + offset, stream->partial_used);
    }
    return YVEX_OK;
}

/*
 * Narrow streamed I64 values to little-endian I32 after complete range checks.
 *
 * Rejects partial, out-of-range, overflow, decode, or sink conditions before commit.
 */
static int quant_cast_consume(void *opaque, const unsigned char *bytes, size_t byte_count,
                              unsigned long long logical_offset, yvex_quant_failure *failure,
                              yvex_error *err) {
    quant_cast_stream *stream = (quant_cast_stream *)opaque;
    size_t offset = 0u;

    (void)logical_offset;
    while (offset < byte_count) {
        size_t take = sizeof(stream->partial) - stream->partial_used;
        int64_t exact_value;
        int32_t value;
        unsigned char encoded[4];
        unsigned long long element;
        int rc;

        if (take > byte_count - offset)
            take = byte_count - offset;
        memcpy(stream->partial + stream->partial_used, bytes + offset, take);
        stream->partial_used += (unsigned int)take;
        offset += take;
        if (stream->partial_used != sizeof(stream->partial))
            continue;
        element = stream->emitter->element_index;
        rc = yvex_quant_source_i64_decode(stream->partial, &exact_value, NULL, NULL);
        if (rc != YVEX_OK)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_CAST_RANGE, stream->emitter->decision,
                stream->source_index, element / stream->emitter->decision->row_width, element, 1u,
                0u, err, rc, "I64 source scalar could not be decoded exactly");
        if (exact_value < INT32_MIN || exact_value > INT32_MAX)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_CAST_RANGE, stream->emitter->decision,
                stream->source_index, element / stream->emitter->decision->row_width, element,
                INT32_MAX, (unsigned long long)exact_value, err, YVEX_ERR_BOUNDS,
                "I64 value is outside the complete I32 publication range");
        if (element >= stream->emitter->decision->element_count ||
            stream->emitter->metrics.element_count == ULLONG_MAX ||
            stream->emitter->metrics.finite_count == ULLONG_MAX)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW,
                                      stream->emitter->decision, stream->source_index, ULLONG_MAX,
                                      element, stream->emitter->decision->element_count, element,
                                      err, YVEX_ERR_BOUNDS,
                                      "checked-cast element accounting exceeded the terminal");
        value = (int32_t)exact_value;
        encoded[0] = (unsigned char)((uint32_t)value & 0xffu);
        encoded[1] = (unsigned char)(((uint32_t)value >> 8) & 0xffu);
        encoded[2] = (unsigned char)(((uint32_t)value >> 16) & 0xffu);
        encoded[3] = (unsigned char)(((uint32_t)value >> 24) & 0xffu);
        rc = quant_emitter_append(stream->emitter, encoded, sizeof(encoded), failure, err);
        if (rc != YVEX_OK)
            return rc;
        stream->emitter->element_index++;
        stream->emitter->row_element++;
        stream->emitter->metrics.element_count++;
        stream->emitter->metrics.finite_count++;
        if (stream->emitter->row_element == stream->emitter->decision->row_width)
            stream->emitter->row_element = 0u;
        stream->partial_used = 0u;
    }
    return YVEX_OK;
}

/*
 * Combine streamed FP8 weights with their admitted E8M0 scale grid.
 *
 * Rejects index overflow, malformed grids, non-finite codes, or emission refusal.
 */
static int quant_scale_pair_consume(void *opaque, const unsigned char *bytes, size_t byte_count,
                                    unsigned long long logical_offset, yvex_quant_failure *failure,
                                    yvex_error *err) {
    quant_scale_pair_stream *stream = (quant_scale_pair_stream *)opaque;
    size_t offset;

    (void)logical_offset;
    for (offset = 0u; offset < byte_count; ++offset) {
        unsigned long long row = stream->element_index / stream->columns;
        unsigned long long column = stream->element_index % stream->columns;
        unsigned long long scale_index;
        unsigned long long scale_row_offset;
        float value;
        float scale;
        int rc;

        if (!yvex_core_u64_mul(row / stream->block_rows, stream->scale_columns, &scale_row_offset) ||
            !yvex_core_u64_add(scale_row_offset, column / stream->block_columns, &scale_index))
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW,
                                      stream->emitter->decision, ULLONG_MAX, row, ULLONG_MAX,
                                      ULLONG_MAX, stream->element_index, err, YVEX_ERR_BOUNDS,
                                      "FP8 scale-grid index arithmetic overflowed");
        if (scale_index >= stream->scale_bytes)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_FP8_SCALE_PAIR,
                                      stream->emitter->decision, ULLONG_MAX, row, scale_index,
                                      stream->scale_bytes, scale_index, err, YVEX_ERR_BOUNDS,
                                      "FP8 scale index exceeded the bound grid");
        value = yvex_quant_fp8_e4m3fn_decode(bytes[offset]);
        scale = yvex_quant_e8m0_decode(stream->scales[scale_index]);
        if (!isfinite(value) || !isfinite(scale))
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_NONFINITE,
                                      stream->emitter->decision, ULLONG_MAX, row, scale_index, 0u,
                                      bytes[offset], err, YVEX_ERR_FORMAT,
                                      "FP8 weight or E8M0 scale decoded as non-finite");
        rc = quant_emitter_value(stream->emitter, value * scale, failure, err);
        if (rc != YVEX_OK)
            return rc;
        stream->element_index++;
    }
    return YVEX_OK;
}

static int quant_expert_consume(void *opaque, const unsigned char *bytes, size_t byte_count,
                                unsigned long long logical_offset, yvex_quant_failure *failure,
                                yvex_error *err) {
    quant_expert_stream *stream = (quant_expert_stream *)opaque;
    size_t offset = 0u;

    (void)logical_offset;
    while (offset < byte_count) {
        size_t take = sizeof(stream->packed) - stream->packed_used;
        float values[YVEX_QUANT_MXFP4_ELEMENTS];
        unsigned int value;
        int rc;

        if (take > byte_count - offset)
            take = byte_count - offset;
        memcpy(stream->packed + stream->packed_used, bytes + offset, take);
        stream->packed_used += (unsigned int)take;
        offset += take;
        if (stream->packed_used != sizeof(stream->packed))
            continue;
        if (stream->group_index >= stream->scale_bytes)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, stream->emitter->decision, ULLONG_MAX,
                ULLONG_MAX, stream->group_index, stream->scale_bytes, stream->group_index, err,
                YVEX_ERR_BOUNDS, "expert MXFP4 scale index exceeded the bound tensor");
        rc = yvex_quant_source_mxfp4_decode(stream->packed, stream->scales[stream->group_index],
                                            values, failure, err);
        if (rc != YVEX_OK)
            return rc;
        for (value = 0u; value < YVEX_QUANT_MXFP4_ELEMENTS; ++value) {
            rc = quant_emitter_value(stream->emitter, values[value], failure, err);
            if (rc != YVEX_OK)
                return rc;
        }
        stream->packed_used = 0u;
        stream->group_index++;
    }
    return YVEX_OK;
}

/*
 * Execute one IR IDENTITY terminal as exact copy, conversion, or selected quantization.
 *
 * Rejects arity, dtype, or partial-scalar violations and propagates transaction failure.
 */
static int quant_execute_identity(quant_executor *executor, const yvex_quant_decision *decision,
                                  const yvex_transform_node *node, quant_emitter *emitter,
                                  yvex_quant_failure *failure, yvex_error *err,
                                  unsigned long long *payload_bytes, unsigned long long *chunks,
                                  unsigned long long *sources) {
    const yvex_transform_value *input = yvex_transform_ir_node_input_at(executor->ir, node, 0u);
    const yvex_transform_source_value *source =
        input ? yvex_transform_ir_source_at(executor->ir, input->source_index) : NULL;
    quant_scalar_stream stream;
    quant_exact_copy_stream exact;
    int exact_copy;
    int rc;

    if (!source || node->input_count != 1u)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, decision,
                                  input ? input->source_index : ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  1u, node->input_count, err, YVEX_ERR_FORMAT,
                                  "IDENTITY requires exactly one bound source value");
    memset(&stream, 0, sizeof(stream));
    stream.emitter = emitter;
    stream.dtype = source->source_dtype;
    stream.scalar_bytes = quant_source_scalar_bytes(source->source_dtype);
    if (!stream.scalar_bytes || stream.scalar_bytes > sizeof(stream.partial))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, decision,
                                  input->source_index, ULLONG_MAX, ULLONG_MAX, 1u,
                                  source->source_dtype, err, YVEX_ERR_UNSUPPORTED,
                                  "IDENTITY source dtype has no canonical scalar decoder");
    exact_copy =
        (source->source_dtype == YVEX_NATIVE_DTYPE_F32 && decision->qtype == YVEX_GGUF_QTYPE_F32) ||
        (source->source_dtype == YVEX_NATIVE_DTYPE_F16 && decision->qtype == YVEX_GGUF_QTYPE_F16) ||
        (source->source_dtype == YVEX_NATIVE_DTYPE_BF16 &&
         decision->qtype == YVEX_GGUF_QTYPE_BF16) ||
        (source->source_dtype == YVEX_NATIVE_DTYPE_I32 && decision->qtype == YVEX_GGUF_QTYPE_I32);
    memset(&exact, 0, sizeof(exact));
    exact.emitter = emitter;
    exact.dtype = source->source_dtype;
    exact.scalar_bytes = stream.scalar_bytes;
    exact.source_index = input->source_index;
    rc = quant_read_source(executor, decision, input->source_index,
                           exact_copy ? quant_exact_copy_consume : quant_scalar_consume,
                           exact_copy ? (void *)&exact : (void *)&stream, failure, err,
                           payload_bytes, chunks);
    if (rc == YVEX_OK && (exact_copy ? exact.partial_used : stream.partial_used) != 0u)
        rc = quant_execute_fail(failure, YVEX_QUANT_FAILURE_SOURCE_SHORT_READ, decision,
                                input->source_index, ULLONG_MAX, ULLONG_MAX, stream.scalar_bytes,
                                exact_copy ? exact.partial_used : stream.partial_used, err,
                                YVEX_ERR_IO, "source scalar ended at a partial element");
    if (rc == YVEX_OK)
        *sources = 1u;
    return rc;
}

static int quant_execute_scale_pair(quant_executor *executor, const yvex_quant_decision *decision,
                                    const yvex_transform_node *node, quant_emitter *emitter,
                                    yvex_quant_failure *failure, yvex_error *err,
                                    unsigned long long *payload_bytes, unsigned long long *chunks,
                                    unsigned long long *sources) {
    const yvex_transform_value *weight_value =
        yvex_transform_ir_node_input_at(executor->ir, node, 0u);
    const yvex_transform_value *scale_value =
        yvex_transform_ir_node_input_at(executor->ir, node, 1u);
    const yvex_transform_source_value *weight =
        weight_value ? yvex_transform_ir_source_at(executor->ir, weight_value->source_index) : NULL;
    const yvex_transform_source_value *scale =
        scale_value ? yvex_transform_ir_source_at(executor->ir, scale_value->source_index) : NULL;
    unsigned char *scale_bytes = NULL;
    size_t scale_size = 0u;
    quant_scale_pair_stream stream;
    unsigned long long scale_payload = 0u;
    unsigned long long scale_chunks = 0u;
    unsigned long long weight_payload = 0u;
    unsigned long long weight_chunks = 0u;
    unsigned long long weight_elements;
    unsigned long long scale_elements;
    int rc;

    if (node->input_count != 2u || !weight || !scale ||
        weight->source_dtype != YVEX_NATIVE_DTYPE_F8_E4M3 ||
        scale->source_dtype != YVEX_NATIVE_DTYPE_F8_E8M0 || weight->shape.rank != 2u ||
        scale->shape.rank != 2u || weight->shape.dims[0] == 0u || weight->shape.dims[1] == 0u ||
        node->scale_block_rows == 0u || node->scale_block_columns == 0u)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_FP8_SCALE_PAIR, decision,
                                  weight_value ? weight_value->source_index : ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, 2u, node->input_count, err,
                                  YVEX_ERR_FORMAT,
                                  "scale-paired decode geometry or source dtypes are malformed");
    if (!yvex_core_u64_mul(weight->shape.dims[0], weight->shape.dims[1], &weight_elements) ||
        !yvex_core_u64_mul(scale->shape.dims[0], scale->shape.dims[1], &scale_elements))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_ELEMENT_OVERFLOW, decision,
                                  weight_value->source_index, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  0u, err, YVEX_ERR_BOUNDS,
                                  "FP8 weight or scale-grid element geometry overflowed");
    if (scale->shape.dims[0] != (weight->shape.dims[0] - 1u) / node->scale_block_rows + 1u ||
        scale->shape.dims[1] != (weight->shape.dims[1] - 1u) / node->scale_block_columns + 1u)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_FP8_SCALE_PAIR, decision,
                                  scale_value->source_index, ULLONG_MAX, ULLONG_MAX,
                                  weight_elements, scale_elements, err, YVEX_ERR_FORMAT,
                                  "FP8 scale grid does not cover the exact weight shape");
    rc = quant_collect_source(executor, decision, scale_value->source_index, &scale_bytes,
                              &scale_size, failure, err, &scale_payload, &scale_chunks);
    if (rc != YVEX_OK)
        return rc;
    memset(&stream, 0, sizeof(stream));
    stream.emitter = emitter;
    stream.scales = scale_bytes;
    stream.scale_bytes = scale_size;
    stream.rows = weight->shape.dims[0];
    stream.columns = weight->shape.dims[1];
    stream.scale_rows = scale->shape.dims[0];
    stream.scale_columns = scale->shape.dims[1];
    stream.block_rows = node->scale_block_rows;
    stream.block_columns = node->scale_block_columns;
    rc = quant_read_source(executor, decision, weight_value->source_index, quant_scale_pair_consume,
                           &stream, failure, err, &weight_payload, &weight_chunks);
    quant_executor_release(executor, scale_bytes, scale_size);
    if (rc != YVEX_OK)
        return rc;
    if (stream.element_index != decision->element_count)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_FP8_SCALE_PAIR, decision, weight_value->source_index,
            ULLONG_MAX, ULLONG_MAX, decision->element_count, stream.element_index, err,
            YVEX_ERR_FORMAT, "FP8 weight element count differs from the terminal plan");
    if (!yvex_core_u64_add(scale_payload, weight_payload, payload_bytes) ||
        !yvex_core_u64_add(scale_chunks, weight_chunks, chunks))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, decision,
                                  weight_value->source_index, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  scale_payload, err, YVEX_ERR_BOUNDS,
                                  "scale-pair source accounting overflowed");
    *sources = 2u;
    return YVEX_OK;
}

/*
 * Execute one checked I64-to-I32 terminal.
 *
 * Numeric narrowing remains transactional and cannot publish a partial terminal.
 */
static int quant_execute_checked_cast(quant_executor *executor, const yvex_quant_decision *decision,
                                      const yvex_transform_node *node, quant_emitter *emitter,
                                      yvex_quant_failure *failure, yvex_error *err,
                                      unsigned long long *payload_bytes, unsigned long long *chunks,
                                      unsigned long long *sources) {
    const yvex_transform_value *input = yvex_transform_ir_node_input_at(executor->ir, node, 0u);
    const yvex_transform_source_value *source =
        input ? yvex_transform_ir_source_at(executor->ir, input->source_index) : NULL;
    const yvex_source_payload_range *range =
        input ? yvex_transform_binding_range_at(executor->binding, input->source_index) : NULL;
    quant_cast_stream stream;
    unsigned long long expected_bytes = ULLONG_MAX;
    int rc;

    if (node->input_count != 1u || !source || source->source_dtype != YVEX_NATIVE_DTYPE_I64 ||
        decision->qtype != YVEX_GGUF_QTYPE_I32)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, decision,
                                  input ? input->source_index : ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  1u, node->input_count, err, YVEX_ERR_FORMAT,
                                  "checked cast requires one I64 input and I32 output");
    if (!yvex_core_u64_mul(decision->element_count, 8u, &expected_bytes) || !range ||
        range->byte_length != expected_bytes)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_CAST_RANGE, decision,
                                  input->source_index, ULLONG_MAX, ULLONG_MAX, expected_bytes,
                                  range ? range->byte_length : 0u, err, YVEX_ERR_FORMAT,
                                  "I64 source byte geometry is inconsistent");
    memset(&stream, 0, sizeof(stream));
    stream.emitter = emitter;
    stream.source_index = input->source_index;
    rc = quant_read_source(executor, decision, input->source_index, quant_cast_consume, &stream,
                           failure, err, payload_bytes, chunks);
    if (rc != YVEX_OK)
        return rc;
    if (stream.partial_used != 0u || emitter->element_index != decision->element_count)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_SOURCE_SHORT_READ, decision,
                                  input->source_index, ULLONG_MAX, ULLONG_MAX,
                                  decision->element_count, emitter->element_index, err, YVEX_ERR_IO,
                                  "I64 source ended at an incomplete checked cast");
    *sources = 1u;
    return YVEX_OK;
}

static int quant_execute_experts(quant_executor *executor, const yvex_quant_decision *decision,
                                 const yvex_transform_node *node, quant_emitter *emitter,
                                 yvex_quant_failure *failure, yvex_error *err,
                                 unsigned long long *payload_bytes, unsigned long long *chunks,
                                 unsigned long long *sources) {
    unsigned long long expert;
    unsigned long long total_payload = 0u;
    unsigned long long total_chunks = 0u;

    if (node->expert_count == 0u || node->expert_count > ULLONG_MAX / 2u ||
        node->input_count != node->expert_count * 2u || node->packing_factor != 2u ||
        node->scale_group_width != 32u)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, decision, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
            node->expert_count * 2u, node->input_count, err, YVEX_ERR_FORMAT,
            "expert aggregation fan-in or source block geometry is malformed");
    for (expert = 0u; expert < node->expert_count; ++expert) {
        const yvex_transform_value *weight_value =
            yvex_transform_ir_node_input_at(executor->ir, node, expert * 2u);
        const yvex_transform_value *scale_value =
            yvex_transform_ir_node_input_at(executor->ir, node, expert * 2u + 1u);
        const yvex_transform_source_value *weight =
            weight_value ? yvex_transform_ir_source_at(executor->ir, weight_value->source_index)
                         : NULL;
        const yvex_transform_source_value *scale =
            scale_value ? yvex_transform_ir_source_at(executor->ir, scale_value->source_index)
                        : NULL;
        unsigned char *scale_bytes = NULL;
        size_t scale_size = 0u;
        unsigned long long scale_payload = 0u;
        unsigned long long scale_chunks = 0u;
        unsigned long long weight_payload = 0u;
        unsigned long long weight_chunks = 0u;
        quant_expert_stream stream;
        int rc;

        if (!weight || !scale || weight->expert_index != expert || scale->expert_index != expert ||
            weight->source_dtype != YVEX_NATIVE_DTYPE_I8 ||
            scale->source_dtype != YVEX_NATIVE_DTYPE_F8_E8M0)
            return quant_execute_fail(
                failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, decision,
                weight_value ? weight_value->source_index : ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                expert, weight ? weight->expert_index : ULLONG_MAX, err, YVEX_ERR_FORMAT,
                "expert weight/scale inputs are not in canonical expert order");
        rc = quant_collect_source(executor, decision, scale_value->source_index, &scale_bytes,
                                  &scale_size, failure, err, &scale_payload, &scale_chunks);
        if (rc != YVEX_OK)
            return rc;
        memset(&stream, 0, sizeof(stream));
        stream.emitter = emitter;
        stream.scales = scale_bytes;
        stream.scale_bytes = scale_size;
        rc = quant_read_source(executor, decision, weight_value->source_index, quant_expert_consume,
                               &stream, failure, err, &weight_payload, &weight_chunks);
        quant_executor_release(executor, scale_bytes, scale_size);
        if (rc != YVEX_OK)
            return rc;
        if (stream.packed_used != 0u || stream.group_index != scale_size)
            return quant_execute_fail(failure, YVEX_QUANT_FAILURE_MXFP4_BLOCK, decision,
                                      weight_value->source_index, ULLONG_MAX, stream.group_index,
                                      scale_size, stream.group_index, err, YVEX_ERR_FORMAT,
                                      "expert packed weights and E8M0 scale groups diverge");
        {
            unsigned long long pair_payload;
            unsigned long long pair_chunks;
            if (!yvex_core_u64_add(scale_payload, weight_payload, &pair_payload) ||
                !yvex_core_u64_add(scale_chunks, weight_chunks, &pair_chunks) ||
                !yvex_core_u64_add(total_payload, pair_payload, &total_payload) ||
                !yvex_core_u64_add(total_chunks, pair_chunks, &total_chunks))
                return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, decision,
                                          weight_value->source_index, ULLONG_MAX, ULLONG_MAX,
                                          ULLONG_MAX, total_payload, err, YVEX_ERR_BOUNDS,
                                          "expert source accounting overflowed");
        }
    }
    *payload_bytes = total_payload;
    *chunks = total_chunks;
    *sources = node->input_count;
    return YVEX_OK;
}

/*
 * Execute exactly one sealed terminal operation inside one sink transaction.
 *
 * Aborts an opened terminal, releases staging, and never reports a partial commit.
 */
static int quant_execute_terminal(quant_executor *executor, unsigned long long ordinal,
                                  quant_terminal_result *result, unsigned long long *payload_bytes,
                                  unsigned long long *chunks, unsigned long long *sources,
                                  yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_decision *decision = yvex_quant_plan_decision_at(executor->plan, ordinal);
    const yvex_transform_value *terminal =
        yvex_transform_binding_terminal_at(executor->binding, ordinal);
    const yvex_transform_node *node =
        yvex_transform_binding_terminal_operation(executor->binding, ordinal);
    quant_emitter emitter;
    int opened = 0;
    int rc;

    memset(result, 0, sizeof(*result));
    *payload_bytes = 0u;
    *chunks = 0u;
    *sources = 0u;
    if (!decision || !terminal || !node || decision->terminal_value_id != terminal->id ||
        decision->node_id != node->id)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_MISSING_DECISION, decision, ULLONG_MAX, ULLONG_MAX,
            ULLONG_MAX, ordinal, decision ? decision->terminal_ordinal : ULLONG_MAX, err,
            YVEX_ERR_FORMAT, "sealed terminal decision does not match the bound operation");
    rc = quant_emitter_open(&emitter, executor, decision, failure, err);
    if (rc != YVEX_OK)
        return rc;
    opened = 1;
    switch (node->kind) {
    case YVEX_TRANSFORM_OP_IDENTITY:
        rc = quant_execute_identity(executor, decision, node, &emitter, failure, err, payload_bytes,
                                    chunks, sources);
        break;
    case YVEX_TRANSFORM_OP_DECODE_SCALE_PAIR:
        rc = quant_execute_scale_pair(executor, decision, node, &emitter, failure, err,
                                      payload_bytes, chunks, sources);
        break;
    case YVEX_TRANSFORM_OP_CHECKED_CAST:
        rc = quant_execute_checked_cast(executor, decision, node, &emitter, failure, err,
                                        payload_bytes, chunks, sources);
        break;
    case YVEX_TRANSFORM_OP_EXPERT_AGGREGATE:
        rc = quant_execute_experts(executor, decision, node, &emitter, failure, err, payload_bytes,
                                   chunks, sources);
        break;
    default:
        rc = quant_execute_fail(failure, YVEX_QUANT_FAILURE_UNSUPPORTED_OPERATION, decision,
                                ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 4u, node->kind, err,
                                YVEX_ERR_UNSUPPORTED,
                                "Transformation IR operation has no byte executor");
        break;
    }
    if (rc == YVEX_OK)
        rc = quant_emitter_finish(&emitter, result, failure, err);
    if (rc != YVEX_OK && opened && executor->sink->abort_terminal)
        executor->sink->abort_terminal(executor->sink->context, decision, failure,
                                       emitter.output_offset);
    quant_emitter_close(&emitter);
    return rc;
}

static int quant_double_merge(double *target, double source) {
    double merged;

    if (!target || !isfinite(*target) || !isfinite(source))
        return 0;
    merged = *target + source;
    if (!isfinite(merged))
        return 0;
    *target = merged;
    return 1;
}

static int quant_metrics_merge(yvex_quant_metrics *target, const yvex_quant_metrics *source) {
    yvex_quant_metrics merged;

    if (!target || !source)
        return 0;
    merged = *target;
    if (!yvex_core_u64_add(merged.element_count, source->element_count, &merged.element_count) ||
        !yvex_core_u64_add(merged.finite_count, source->finite_count, &merged.finite_count) ||
        !yvex_core_u64_add(merged.nonfinite_count, source->nonfinite_count, &merged.nonfinite_count) ||
        !isfinite(source->maximum_absolute_error) ||
        !quant_double_merge(&merged.absolute_error_sum, source->absolute_error_sum) ||
        !quant_double_merge(&merged.squared_error_sum, source->squared_error_sum) ||
        !quant_double_merge(&merged.relative_error_sum, source->relative_error_sum) ||
        !quant_double_merge(&merged.reference_squared_sum, source->reference_squared_sum) ||
        !quant_double_merge(&merged.dot_reference, source->dot_reference) ||
        !quant_double_merge(&merged.dot_reconstructed, source->dot_reconstructed))
        return 0;
    if (source->maximum_absolute_error > merged.maximum_absolute_error)
        merged.maximum_absolute_error = source->maximum_absolute_error;
    *target = merged;
    return 1;
}

/*
 * Account one committed terminal by qtype, role, payload, chunks, and metrics.
 *
 * Reports overflow or invalid identity without changing the prior summary.
 */
static int quant_summary_record_terminal(quant_executor *executor,
                                         const yvex_quant_decision *decision,
                                         const quant_terminal_result *result,
                                         unsigned long long payload_bytes,
                                         unsigned long long chunks, unsigned long long sources,
                                         yvex_quant_failure *failure, yvex_error *err) {
    yvex_quant_execution_summary next;

    if (!executor || !decision || !result ||
        decision->qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID ||
        (unsigned int)decision->role >= YVEX_TENSOR_ROLE_COUNT)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, decision,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, 1u, 0u, err,
                                  YVEX_ERR_INVALID_ARG,
                                  "committed terminal has no bounded summary identity");
    next = executor->summary;
    if (!yvex_core_u64_add(next.terminals_executed, 1u, &next.terminals_executed) ||
        !yvex_core_u64_add(next.committed_terminals, 1u, &next.committed_terminals) ||
        !yvex_core_u64_add(next.source_values_consumed, sources, &next.source_values_consumed) ||
        !yvex_core_u64_add(next.source_ranges_read, sources, &next.source_ranges_read) ||
        !yvex_core_u64_add(next.payload_bytes_read, payload_bytes, &next.payload_bytes_read) ||
        !yvex_core_u64_add(next.source_chunks, chunks, &next.source_chunks) ||
        !yvex_core_u64_add(next.output_chunks, result->output_chunks, &next.output_chunks) ||
        !yvex_core_u64_add(next.encoded_output_bytes, result->encoded_bytes,
                       &next.encoded_output_bytes) ||
        !yvex_core_u64_add(next.reference_decode_elements, result->reference_elements,
                       &next.reference_decode_elements) ||
        !yvex_core_u64_add(next.qtype_tensor_counts[decision->qtype], 1u,
                       &next.qtype_tensor_counts[decision->qtype]) ||
        !yvex_core_u64_add(next.qtype_output_bytes[decision->qtype], result->encoded_bytes,
                       &next.qtype_output_bytes[decision->qtype]) ||
        !yvex_core_u64_add(next.role_tensor_counts[decision->role], 1u,
                       &next.role_tensor_counts[decision->role]))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_BYTE_OVERFLOW, decision, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, next.encoded_output_bytes,
                                  err, YVEX_ERR_BOUNDS,
                                  "committed terminal aggregate accounting overflowed");
    if (!quant_metrics_merge(&next.qtype_metrics[decision->qtype], &result->metrics) ||
        !quant_metrics_merge(&next.role_metrics[decision->role], &result->metrics))
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_NUMERIC_BOUND, decision, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, result->metrics.element_count,
                                  err, YVEX_ERR_BOUNDS,
                                  "committed terminal numeric metric accounting overflowed");
    executor->summary = next;
    return YVEX_OK;
}

/* Select and retain the deterministic primary worker failure. */
static void quant_record_failure(quant_executor *executor, const yvex_quant_failure *failure,
                                 const yvex_error *err) {
    int replace;

    pthread_mutex_lock(&executor->mutex);
    atomic_store_explicit(&executor->stop, 1, memory_order_release);
    replace = !executor->failure_set ||
              ((executor->failure.code == YVEX_QUANT_FAILURE_CANCELLED ||
                executor->failure.code == YVEX_QUANT_FAILURE_WORKER) &&
               failure->code != YVEX_QUANT_FAILURE_CANCELLED &&
               failure->code != YVEX_QUANT_FAILURE_WORKER) ||
              (failure->code != YVEX_QUANT_FAILURE_CANCELLED &&
               failure->code != YVEX_QUANT_FAILURE_WORKER &&
               executor->failure.code != YVEX_QUANT_FAILURE_CANCELLED &&
               executor->failure.code != YVEX_QUANT_FAILURE_WORKER &&
               failure->terminal_ordinal < executor->failure.terminal_ordinal);
    if (replace) {
        executor->failure = *failure;
        executor->error = *err;
        executor->failure_set = 1;
    }
    if (failure->code == YVEX_QUANT_FAILURE_CANCELLED)
        executor->summary.cancellations++;
    else if (failure->code == YVEX_QUANT_FAILURE_SOURCE_SHORT_READ)
        executor->summary.short_reads++;
    else if (failure->code == YVEX_QUANT_FAILURE_SOURCE_DRIFT)
        executor->summary.payload_drifts++;
    else if (failure->code == YVEX_QUANT_FAILURE_SINK_SHORT_WRITE ||
             failure->code == YVEX_QUANT_FAILURE_SINK_PROTOCOL)
        executor->summary.sink_failures++;
    else if (failure->code == YVEX_QUANT_FAILURE_NUMERIC_BOUND)
        executor->summary.numeric_bound_violations++;
    pthread_mutex_unlock(&executor->mutex);
}

/*
 * Claim and execute independent terminals until completion or global stop.
 *
 * Publishes the deterministic primary failure and requests peer termination.
 */
static void *quant_worker_main(void *opaque) {
    quant_worker *worker = (quant_worker *)opaque;
    quant_executor *executor = worker->executor;

    (void)worker->worker_index;
    while (!atomic_load_explicit(&executor->start, memory_order_acquire) &&
           !atomic_load_explicit(&executor->stop, memory_order_acquire))
        sched_yield();
    for (;;) {
        unsigned long long ordinal;
        const yvex_quant_decision *decision;
        quant_terminal_result result;
        yvex_quant_failure failure;
        yvex_error error;
        unsigned long long payload_bytes;
        unsigned long long chunks;
        unsigned long long sources;
        int rc;

        pthread_mutex_lock(&executor->mutex);
        if (atomic_load_explicit(&executor->stop, memory_order_acquire) ||
            executor->next_terminal >= executor->summary.terminal_decisions) {
            pthread_mutex_unlock(&executor->mutex);
            break;
        }
        ordinal = executor->next_terminal++;
        executor->summary.terminals_attempted++;
        pthread_mutex_unlock(&executor->mutex);
        if (quant_cancellation_requested(executor->options.cancellation)) {
            decision = yvex_quant_plan_decision_at(executor->plan, ordinal);
            quant_execute_fail(&failure, YVEX_QUANT_FAILURE_CANCELLED, decision, ULLONG_MAX,
                               ULLONG_MAX, ULLONG_MAX, 0u, 0u, &error, YVEX_ERR_STATE,
                               "quantization execution was cancelled");
            quant_record_failure(executor, &failure, &error);
            break;
        }
        rc = quant_execute_terminal(executor, ordinal, &result, &payload_bytes, &chunks, &sources,
                                    &failure, &error);
        if (rc != YVEX_OK) {
            quant_record_failure(executor, &failure, &error);
            break;
        }
        decision = yvex_quant_plan_decision_at(executor->plan, ordinal);
        pthread_mutex_lock(&executor->mutex);
        rc = quant_summary_record_terminal(executor, decision, &result, payload_bytes, chunks,
                                           sources, &failure, &error);
        pthread_mutex_unlock(&executor->mutex);
        if (rc != YVEX_OK) {
            quant_record_failure(executor, &failure, &error);
            break;
        }
    }
    return NULL;
}

int yvex_quant_execute(const yvex_quant_plan *plan, const yvex_quant_output_sink *sink,
                       const yvex_quant_executor_options *requested_options,
                       yvex_quant_execution_summary *summary, yvex_quant_failure *failure,
                       yvex_error *err) {
    const yvex_quant_plan_summary *plan_summary = yvex_quant_plan_summary_get(plan);
    const yvex_transform_binding *binding = yvex_quant_plan_binding(plan);
    yvex_transform_failure transform_failure;
    yvex_quant_executor_options options;
    quant_executor executor;
    quant_worker *workers = NULL;
    pthread_t *threads = NULL;
    size_t worker_bytes;
    size_t thread_bytes;
    size_t required_output_bytes = 0u;
    int worker_budget_exceeded = 0;
    int thread_budget_exceeded = 0;
    unsigned int started = 0u;
    unsigned int index;
    int rc;

    if (summary)
        memset(summary, 0, sizeof(*summary));
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    if (!plan || !plan_summary || !plan_summary->complete || !binding || !sink ||
        !sink->begin_terminal || !sink->deliver_chunk || !sink->commit_terminal ||
        !sink->abort_terminal || !summary)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, NULL, ULLONG_MAX,
                                  ULLONG_MAX, ULLONG_MAX, 1u, 0u, err, YVEX_ERR_INVALID_ARG,
                                  "sealed plan, complete sink protocol, and summary are required");
    yvex_quant_executor_options_default(&options);
    if (requested_options)
        options = *requested_options;
    if (!options.allocate)
        options.allocate = quant_executor_default_allocate;
    if (!options.release)
        options.release = quant_executor_default_release;
    if (!options.thread_create)
        options.thread_create = quant_executor_default_thread_create;
    if (options.worker_count != 0u && options.output_chunk_bytes <= SIZE_MAX / options.worker_count)
        required_output_bytes = options.output_chunk_bytes * options.worker_count;
    if (options.worker_count == 0u || options.worker_count > 64u ||
        options.source_chunk_bytes < YVEX_SOURCE_PAYLOAD_MIN_CHUNK_BYTES ||
        options.source_chunk_bytes > YVEX_SOURCE_PAYLOAD_MAX_CHUNK_BYTES ||
        options.output_chunk_bytes < 4096u || options.output_chunk_bytes > 16u * 1024u * 1024u ||
        required_output_bytes == 0u || options.maximum_owned_bytes < required_output_bytes)
        return quant_execute_fail(
            failure, YVEX_QUANT_FAILURE_RESOURCE_BUDGET, NULL, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
            required_output_bytes ? required_output_bytes : SIZE_MAX, options.maximum_owned_bytes,
            err, YVEX_ERR_BOUNDS, "executor worker, chunk, or memory budget is invalid");
    rc = quant_imatrix_plan_validate(plan, plan_summary, options.imatrix, failure, err);
    if (rc != YVEX_OK)
        return rc;
    rc = yvex_transform_binding_readable_validate(binding, &transform_failure, err);
    if (rc != YVEX_OK)
        return quant_execute_fail(failure,
                                  transform_failure.code ==
                                          YVEX_TRANSFORM_FAILURE_PAYLOAD_IDENTITY_MISMATCH
                                      ? YVEX_QUANT_FAILURE_PAYLOAD_IDENTITY
                                      : YVEX_QUANT_FAILURE_PAYLOAD_NOT_READABLE,
                                  NULL, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX,
                                  YVEX_SOURCE_PAYLOAD_STATE_READY, transform_failure.actual, err,
                                  rc, "transform binding is not readable at the byte boundary");
    memset(&executor, 0, sizeof(executor));
    executor.plan = plan;
    executor.ir = yvex_quant_plan_transform_ir(plan);
    executor.binding = binding;
    executor.session = yvex_transform_binding_payload_session(binding);
    executor.sink = sink;
    executor.options = options;
    executor.summary.terminal_decisions = plan_summary->decision_count;
    executor.summary.configured_memory_budget = options.maximum_owned_bytes;
    executor.summary.configured_workers = options.worker_count;
    atomic_init(&executor.start, 0);
    atomic_init(&executor.stop, 0);
    if (!executor.ir || !executor.session || pthread_mutex_init(&executor.mutex, NULL) != 0)
        return quant_execute_fail(failure, YVEX_QUANT_FAILURE_WORKER, NULL, ULLONG_MAX, ULLONG_MAX,
                                  ULLONG_MAX, 1u, 0u, err, YVEX_ERR_STATE,
                                  "executor binding or worker mutex initialization failed");
    executor.mutex_initialized = 1;
    worker_bytes = (size_t)options.worker_count * sizeof(*workers);
    thread_bytes =
        options.worker_count > 1u ? (size_t)(options.worker_count - 1u) * sizeof(*threads) : 0u;
    workers =
        (quant_worker *)quant_executor_allocate(&executor, worker_bytes, &worker_budget_exceeded);
    if (thread_bytes)
        threads =
            (pthread_t *)quant_executor_allocate(&executor, thread_bytes, &thread_budget_exceeded);
    if (!workers || (thread_bytes && !threads)) {
        int budget_exceeded = worker_budget_exceeded || thread_budget_exceeded;
        rc = quant_execute_fail(
            failure,
            budget_exceeded ? YVEX_QUANT_FAILURE_RESOURCE_BUDGET : YVEX_QUANT_FAILURE_ALLOCATION,
            NULL, ULLONG_MAX, ULLONG_MAX, ULLONG_MAX, worker_bytes + thread_bytes, 0u, err,
            budget_exceeded ? YVEX_ERR_BOUNDS : YVEX_ERR_NOMEM,
            budget_exceeded ? "executor worker state exceeded the memory budget"
                            : "executor worker-state allocation failed");
        goto cleanup;
    }
    rc = YVEX_OK;
    memset(workers, 0, worker_bytes);
    if (threads)
        memset(threads, 0, thread_bytes);
    for (index = 0u; index < options.worker_count; ++index) {
        workers[index].executor = &executor;
        workers[index].worker_index = index;
    }
    for (index = 1u; index < options.worker_count; ++index) {
        if (options.thread_create(&threads[index - 1u], quant_worker_main, &workers[index],
                                  options.context) != 0) {
            pthread_mutex_lock(&executor.mutex);
            atomic_store_explicit(&executor.stop, 1, memory_order_release);
            executor.summary.worker_failures++;
            pthread_mutex_unlock(&executor.mutex);
            rc = quant_execute_fail(failure, YVEX_QUANT_FAILURE_WORKER, NULL, ULLONG_MAX,
                                    ULLONG_MAX, ULLONG_MAX, options.worker_count - 1u, started, err,
                                    YVEX_ERR_STATE, "executor worker thread startup failed");
            goto join;
        }
        started++;
    }
    executor.summary.workers_started = started + 1u;
    atomic_store_explicit(&executor.start, 1, memory_order_release);
    (void)quant_worker_main(&workers[0]);
join:
    atomic_store_explicit(&executor.start, 1, memory_order_release);
    for (index = 0u; index < started; ++index)
        (void)pthread_join(threads[index], NULL);
    if (rc != YVEX_OK)
        goto cleanup;
    if (executor.failure_set) {
        *failure = executor.failure;
        *err = executor.error;
        rc = yvex_error_code(err);
        if (rc == YVEX_OK)
            rc = YVEX_ERR_STATE;
        goto cleanup;
    }
    if (executor.summary.terminals_executed != plan_summary->decision_count ||
        executor.summary.source_values_consumed != plan_summary->source_value_count ||
        executor.summary.encoded_output_bytes != plan_summary->encoded_bytes) {
        rc = quant_execute_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, NULL, ULLONG_MAX,
                                ULLONG_MAX, ULLONG_MAX, plan_summary->decision_count,
                                executor.summary.terminals_executed, err, YVEX_ERR_FORMAT,
                                "complete-plan execution accounting is incomplete");
        goto cleanup;
    }
    executor.summary.complete = 1;
    rc = YVEX_OK;
cleanup:
    executor.summary.aborted_terminals =
        executor.summary.terminals_attempted >= executor.summary.committed_terminals
            ? executor.summary.terminals_attempted - executor.summary.committed_terminals
            : 0u;
    if (threads)
        quant_executor_release(&executor, threads, thread_bytes);
    if (workers)
        quant_executor_release(&executor, workers, worker_bytes);
    *summary = executor.summary;
    if (executor.mutex_initialized)
        pthread_mutex_destroy(&executor.mutex);
    if (rc == YVEX_OK) {
        if (failure)
            memset(failure, 0, sizeof(*failure));
        yvex_error_clear(err);
    }
    return rc;
}
