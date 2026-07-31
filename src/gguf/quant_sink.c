/*
 * Enforce terminal transactions and derive schedule-independent quantized-byte digests.
 *
 * No payload byte is retained; a terminal digest publishes only on exact-size commit; final
 * identity follows canonical terminal order. Execution digests prove byte production, not GGUF
 * emission.
 */
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>

typedef enum {
    QUANT_DIGEST_EMPTY = 0,
    QUANT_DIGEST_ACTIVE,
    QUANT_DIGEST_COMMITTED,
    QUANT_DIGEST_ABORTED
} quant_digest_state;

typedef struct {
    quant_digest_state state;
    unsigned int qtype;
    unsigned long long expected_bytes;
    unsigned long long delivered_bytes;
    unsigned long long chunks;
    yvex_sha256 hash;
    char digest[YVEX_SHA256_HEX_BYTES];
} quant_digest_record;

struct yvex_quant_digest_sink {
    pthread_mutex_t mutex;
    int mutex_initialized;
    const yvex_quant_plan *plan;
    char plan_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    char payload_identity[YVEX_QUANT_PLAN_IDENTITY_CAP];
    quant_digest_record *records;
    yvex_quant_digest_summary summary;
    yvex_quant_allocate_fn allocate;
    yvex_quant_release_fn release;
    void *allocator_context;
    int finalized;
};

static void *quant_sink_default_allocate(size_t size, void *context) {
    (void)context;
    return malloc(size);
}

static void quant_sink_default_release(void *allocation, void *context) {
    (void)context;
    free(allocation);
}

static int quant_sink_fail(yvex_quant_failure *failure, yvex_quant_failure_code code,
                           unsigned long long terminal, unsigned long long expected,
                           unsigned long long actual, yvex_error *err, int status,
                           const char *message) {
    if (failure) {
        memset(failure, 0, sizeof(*failure));
        failure->code = code;
        failure->terminal_ordinal = terminal;
        failure->source_index = ULLONG_MAX;
        failure->row_index = ULLONG_MAX;
        failure->block_index = ULLONG_MAX;
        failure->expected = expected;
        failure->actual = actual;
        failure->qtype = UINT_MAX;
        failure->operation = YVEX_TRANSFORM_OP_COUNT;
    }
    yvex_error_set(err, (yvex_status)status, "quant.sink", message);
    return status;
}

static int quant_digest_begin(void *opaque, const yvex_quant_decision *decision) {
    yvex_quant_digest_sink *sink = (yvex_quant_digest_sink *)opaque;
    quant_digest_record *record;
    int refused = 0;

    if (!sink || !decision || decision->terminal_ordinal >= sink->summary.terminal_count)
        return 1;
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->finalized || record->state != QUANT_DIGEST_EMPTY ||
        decision != yvex_quant_plan_decision_at(sink->plan, decision->terminal_ordinal) ||
        decision->qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID) {
        refused = 1;
    } else {
        record->state = QUANT_DIGEST_ACTIVE;
        record->qtype = decision->qtype;
        record->expected_bytes = decision->encoded_bytes;
        yvex_sha256_init(&record->hash);
    }
    pthread_mutex_unlock(&sink->mutex);
    return refused;
}

/*
 * Hash one monotonic bounded output chunk for an active terminal.
 *
 * Protocol, overflow, oversize, or hash refusal leaves counters unchanged.
 */
static int quant_digest_chunk(void *opaque, const yvex_quant_decision *decision,
                              unsigned long long output_offset, const unsigned char *bytes,
                              size_t byte_count) {
    yvex_quant_digest_sink *sink = (yvex_quant_digest_sink *)opaque;
    quant_digest_record *record;
    int refused = 0;

    if (!sink || !decision || !bytes || byte_count == 0u ||
        decision->terminal_ordinal >= sink->summary.terminal_count)
        return 1;
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->finalized || record->state != QUANT_DIGEST_ACTIVE ||
        output_offset != record->delivered_bytes ||
        ULLONG_MAX - record->delivered_bytes < byte_count || record->chunks == ULLONG_MAX ||
        record->delivered_bytes + byte_count > record->expected_bytes ||
        !yvex_sha256_update(&record->hash, bytes, byte_count)) {
        refused = 1;
    } else {
        record->delivered_bytes += byte_count;
        record->chunks++;
    }
    pthread_mutex_unlock(&sink->mutex);
    return refused;
}

/*
 * Commit one terminal only after exact-size byte delivery and hash finalization.
 *
 * Terminal commit is not aggregate execution finalization.
 */
static int quant_digest_commit(void *opaque, const yvex_quant_decision *decision,
                               unsigned long long delivered_bytes) {
    yvex_quant_digest_sink *sink = (yvex_quant_digest_sink *)opaque;
    quant_digest_record *record;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    int refused = 0;

    if (!sink || !decision || decision->terminal_ordinal >= sink->summary.terminal_count)
        return 1;
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (sink->finalized || record->state != QUANT_DIGEST_ACTIVE ||
        delivered_bytes != record->delivered_bytes || delivered_bytes != record->expected_bytes ||
        sink->summary.committed_terminals == ULLONG_MAX ||
        ULLONG_MAX - sink->summary.output_chunks < record->chunks ||
        ULLONG_MAX - sink->summary.encoded_bytes < delivered_bytes ||
        ULLONG_MAX - sink->summary.qtype_bytes[record->qtype] < delivered_bytes ||
        !yvex_sha256_final(&record->hash, digest)) {
        refused = 1;
    } else {
        yvex_sha256_hex(digest, record->digest);
        record->state = QUANT_DIGEST_COMMITTED;
        sink->summary.committed_terminals++;
        sink->summary.output_chunks += record->chunks;
        sink->summary.encoded_bytes += delivered_bytes;
        sink->summary.qtype_bytes[record->qtype] += delivered_bytes;
    }
    pthread_mutex_unlock(&sink->mutex);
    return refused;
}

static void quant_digest_abort(void *opaque, const yvex_quant_decision *decision,
                               const yvex_quant_failure *failure,
                               unsigned long long delivered_bytes) {
    yvex_quant_digest_sink *sink = (yvex_quant_digest_sink *)opaque;
    quant_digest_record *record;

    (void)failure;
    (void)delivered_bytes;
    if (!sink || !decision || decision->terminal_ordinal >= sink->summary.terminal_count)
        return;
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[decision->terminal_ordinal];
    if (record->state == QUANT_DIGEST_ACTIVE || record->state == QUANT_DIGEST_EMPTY) {
        memset(&record->hash, 0, sizeof(record->hash));
        record->state = QUANT_DIGEST_ABORTED;
        if (sink->summary.aborted_terminals != ULLONG_MAX)
            sink->summary.aborted_terminals++;
    }
    pthread_mutex_unlock(&sink->mutex);
}

/*
 * Create a digest sink using the canonical heap allocator.
 *
 * Output owner slot, sealed plan, matching payload identity, and diagnostics. Allocates bounded
 * per-terminal hash state while borrowing the plan. Invalid plan/identity or allocation failure
 * leaves the output null.
 */
int yvex_quant_digest_sink_create(yvex_quant_digest_sink **out, const yvex_quant_plan *plan,
                                  const char *payload_identity, yvex_quant_failure *failure,
                                  yvex_error *err) {
    return yvex_quant_digest_sink_create_with_allocator(out, plan, payload_identity, NULL, failure,
                                                        err);
}

/*
 * Construct a digest sink through an explicit fault-injectable allocator seam.
 *
 * Output slot, sealed plan, payload identity, optional allocator, and diagnostics. Owns sink,
 * record array, and mutex; borrows the sealed plan for its lifetime. Invalid lifecycle, identity
 * mismatch, allocation, or mutex failure unwinds completely. Custom allocation changes ownership
 * mechanics, never digest semantics.
 */
int yvex_quant_digest_sink_create_with_allocator(yvex_quant_digest_sink **out,
                                                 const yvex_quant_plan *plan,
                                                 const char *payload_identity,
                                                 const yvex_quant_sink_allocator *allocator,
                                                 yvex_quant_failure *failure, yvex_error *err) {
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    yvex_quant_digest_sink *sink;
    yvex_quant_allocate_fn allocate =
        allocator && allocator->allocate ? allocator->allocate : quant_sink_default_allocate;
    yvex_quant_release_fn release =
        allocator && allocator->release ? allocator->release : quant_sink_default_release;
    void *allocator_context = allocator ? allocator->context : NULL;
    size_t record_bytes;

    if (out)
        *out = NULL;
    if (!out || !plan || !summary || !summary->complete || !payload_identity ||
        summary->terminal_count > SIZE_MAX / sizeof(quant_digest_record) ||
        (allocator && (!allocator->allocate || !allocator->release)))
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX, 1u, 0u,
                               err, YVEX_ERR_INVALID_ARG,
                               "sealed plan and matching payload identity are required");
    if (strcmp(payload_identity, summary->required_payload_identity) != 0)
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_PAYLOAD_IDENTITY, ULLONG_MAX, 1u, 0u,
                               err, YVEX_ERR_FORMAT,
                               "digest sink payload identity does not match the sealed plan");
    sink = (yvex_quant_digest_sink *)allocate(sizeof(*sink), allocator_context);
    if (!sink)
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_ALLOCATION, ULLONG_MAX, sizeof(*sink),
                               0u, err, YVEX_ERR_NOMEM, "digest sink allocation failed");
    memset(sink, 0, sizeof(*sink));
    sink->allocate = allocate;
    sink->release = release;
    sink->allocator_context = allocator_context;
    record_bytes = (size_t)summary->terminal_count * sizeof(*sink->records);
    sink->records = (quant_digest_record *)allocate(record_bytes, allocator_context);
    if (sink->records)
        memset(sink->records, 0, record_bytes);
    if (!sink->records || pthread_mutex_init(&sink->mutex, NULL) != 0) {
        if (sink->records)
            release(sink->records, allocator_context);
        release(sink, allocator_context);
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_ALLOCATION, ULLONG_MAX,
                               summary->terminal_count, 0u, err, YVEX_ERR_NOMEM,
                               "digest sink record or mutex allocation failed");
    }
    sink->mutex_initialized = 1;
    sink->plan = plan;
    sink->summary.terminal_count = summary->terminal_count;
    memcpy(sink->plan_identity, summary->profile_identity, sizeof(sink->plan_identity));
    memcpy(sink->payload_identity, payload_identity, sizeof(sink->payload_identity));
    *out = sink;
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Release every sink-owned resource and null the caller owner slot.
 *
 * Destroys the mutex, releases records and sink storage, and clears ownership.
 */
void yvex_quant_digest_sink_release(yvex_quant_digest_sink **sink_address) {
    yvex_quant_digest_sink *sink;
    yvex_quant_release_fn release;
    void *allocator_context;

    if (!sink_address || !*sink_address)
        return;
    sink = *sink_address;
    *sink_address = NULL;
    release = sink->release ? sink->release : quant_sink_default_release;
    allocator_context = sink->allocator_context;
    if (sink->mutex_initialized)
        pthread_mutex_destroy(&sink->mutex);
    release(sink->records, allocator_context);
    memset(sink, 0, sizeof(*sink));
    release(sink, allocator_context);
}

/*
 * Expose the digest sink through the transactional quant output protocol.
 *
 * The adapter borrows the sink and retains no independent ownership.
 */
void yvex_quant_digest_sink_adapter(yvex_quant_digest_sink *sink, yvex_quant_output_sink *out) {
    if (!out)
        return;
    memset(out, 0, sizeof(*out));
    if (!sink)
        return;
    out->begin_terminal = quant_digest_begin;
    out->deliver_chunk = quant_digest_chunk;
    out->commit_terminal = quant_digest_commit;
    out->abort_terminal = quant_digest_abort;
    out->context = sink;
}

/*
 * Seal the canonical aggregate execution identity after every terminal commits.
 *
 * Aborted, missing, repeated, or identity-encoding failure publishes no complete summary.
 * Execution identity remains distinct from a serialized artifact identity.
 */
int yvex_quant_digest_sink_finalize(yvex_quant_digest_sink *sink, yvex_quant_digest_summary *out,
                                    yvex_quant_failure *failure, yvex_error *err) {
    yvex_sha256 hash;
    yvex_sha256 payload_hash;
    unsigned char digest[YVEX_SHA256_DIGEST_BYTES];
    unsigned long long ordinal = 0u;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!sink || !out)
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX, 1u, 0u,
                               err, YVEX_ERR_INVALID_ARG,
                               "digest sink and summary output are required");
    pthread_mutex_lock(&sink->mutex);
    if (sink->finalized || sink->summary.committed_terminals != sink->summary.terminal_count ||
        sink->summary.aborted_terminals != 0u) {
        unsigned long long committed = sink->summary.committed_terminals;
        pthread_mutex_unlock(&sink->mutex);
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, ULLONG_MAX,
                               sink->summary.terminal_count, committed, err, YVEX_ERR_STATE,
                               "all terminal transactions must commit before finalization");
    }
    yvex_sha256_init(&hash);
    yvex_sha256_init(&payload_hash);
    if (!yvex_sha256_update_text(&hash, "yvex.quant.execution.v1") ||
        !yvex_sha256_update_text(&hash, sink->plan_identity) ||
        !yvex_sha256_update_text(&hash, sink->payload_identity) ||
        !yvex_sha256_update_u64(&hash, YVEX_QUANT_NUMERIC_CONTRACT_VERSION) ||
        !yvex_sha256_update_u64(&hash, sink->summary.terminal_count) ||
        !yvex_sha256_update_text(&payload_hash, "yvex.quant.payload.bytes.v1") ||
        !yvex_sha256_update_u64(&payload_hash, sink->summary.terminal_count))
        goto encoding_failure;
    for (ordinal = 0u; ordinal < sink->summary.terminal_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(sink->plan, ordinal);
        quant_digest_record *record = &sink->records[ordinal];
        if (!decision || record->state != QUANT_DIGEST_COMMITTED ||
            !yvex_sha256_update_u64(&hash, ordinal) ||
            !yvex_sha256_update_text(&hash, decision->decision_identity) ||
            !yvex_sha256_update_u64(&hash, record->delivered_bytes) ||
            !yvex_sha256_update_text(&hash, record->digest) ||
            !yvex_sha256_update_u64(&payload_hash, ordinal) ||
            !yvex_sha256_update_u64(&payload_hash, decision->qtype) ||
            !yvex_sha256_update_u64(&payload_hash, record->delivered_bytes) ||
            !yvex_sha256_update_text(&payload_hash, record->digest))
            goto encoding_failure;
    }
    if (!yvex_sha256_final(&hash, digest))
        goto encoding_failure;
    yvex_sha256_hex(digest, sink->summary.execution_identity);
    if (!yvex_sha256_final(&payload_hash, digest))
        goto encoding_failure;
    yvex_sha256_hex(digest, sink->summary.payload_byte_identity);
    sink->summary.complete = 1;
    sink->finalized = 1;
    *out = sink->summary;
    pthread_mutex_unlock(&sink->mutex);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
encoding_failure:
    pthread_mutex_unlock(&sink->mutex);
    return quant_sink_fail(failure, YVEX_QUANT_FAILURE_DIGEST_MISMATCH, ULLONG_MAX,
                           sink->summary.terminal_count, ordinal, err, YVEX_ERR_BOUNDS,
                           "aggregate execution identity encoding failed");
}

/* Compare a complete observed execution digest with one expected identity. */
int yvex_quant_digest_summary_validate(const yvex_quant_digest_summary *summary,
                                       const char *expected_execution_identity,
                                       yvex_quant_failure *failure, yvex_error *err) {
    if (!summary || !summary->complete || !yvex_sha256_hex_valid(summary->execution_identity) ||
        !yvex_sha256_hex_valid(expected_execution_identity))
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ULLONG_MAX, 1u, 0u,
                               err, YVEX_ERR_INVALID_ARG,
                               "complete observed and expected execution identities are required");
    if (strcmp(summary->execution_identity, expected_execution_identity) != 0)
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_DIGEST_MISMATCH, ULLONG_MAX,
                               summary->terminal_count, 0u, err, YVEX_ERR_FORMAT,
                               "aggregate quantized-output identity mismatch");
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Copy one committed terminal digest into a pointer-free caller-owned value.
 *
 * The accessor exposes no raw encoded bytes and transfers no sink ownership.
 */
int yvex_quant_digest_sink_terminal_at(yvex_quant_digest_sink *sink, unsigned long long ordinal,
                                       yvex_quant_terminal_digest *out, yvex_quant_failure *failure,
                                       yvex_error *err) {
    quant_digest_record *record;

    if (out)
        memset(out, 0, sizeof(*out));
    if (!sink || !out || ordinal >= sink->summary.terminal_count)
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INVALID_ARGUMENT, ordinal,
                               sink ? sink->summary.terminal_count : 0u, ordinal, err,
                               YVEX_ERR_INVALID_ARG,
                               "digest terminal ordinal and output must be valid");
    pthread_mutex_lock(&sink->mutex);
    record = &sink->records[ordinal];
    if (record->state != QUANT_DIGEST_COMMITTED) {
        pthread_mutex_unlock(&sink->mutex);
        return quant_sink_fail(failure, YVEX_QUANT_FAILURE_INCOMPLETE, ordinal, 1u, record->state,
                               err, YVEX_ERR_STATE,
                               "terminal digest is unavailable before exact commit");
    }
    out->terminal_ordinal = ordinal;
    out->qtype = record->qtype;
    out->delivered_bytes = record->delivered_bytes;
    out->chunks = record->chunks;
    memcpy(out->sha256, record->digest, sizeof(out->sha256));
    out->committed = 1;
    pthread_mutex_unlock(&sink->mutex);
    if (failure)
        memset(failure, 0, sizeof(*failure));
    yvex_error_clear(err);
    return YVEX_OK;
}

/*
 * Calculate exact sink-owned bytes for resource-budget evidence.
 *
 * Invalid state or size overflow yields zero.
 */
size_t yvex_quant_digest_sink_owned_bytes(const yvex_quant_digest_sink *sink) {
    const yvex_quant_plan_summary *summary;
    size_t records;

    if (!sink || !sink->plan)
        return 0u;
    summary = yvex_quant_plan_summary_get(sink->plan);
    if (!summary || summary->terminal_count > SIZE_MAX / sizeof(*sink->records))
        return 0u;
    records = (size_t)summary->terminal_count * sizeof(*sink->records);
    return records > SIZE_MAX - sizeof(*sink) ? 0u : sizeof(*sink) + records;
}
