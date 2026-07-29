/* Owner: core.status (core).
 * Owns: status names, error text, version facts, process-local logging, checked arithmetic, canonical string
 *   ownership, and deterministic index hashing.
 * Does not own: unrelated subsystem policy or unsupported higher-stage claims.
 * Invariants: errors remain typed, copied text is terminated, arithmetic is checked, hashes use explicit byte
 *   encodings, and logging stays process-local.
 * Boundary: core status utilities do not infer model, backend, or runtime capability.
 * Purpose: provide status, error, checked scalar, owned-string, index-hash, and logging primitives.
 * Inputs: typed core values, borrowed byte ranges, and caller-owned outputs.
 * Effects: mutates only declared core state or caller-owned outputs.
 * Failure: returns typed refusal or null allocation with caller-owned state defined. */

#define YVEX_CORE_ALLOCATION_IMPLEMENTATION 1
#include <yvex/internal/core.h>
#include <limits.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const char yvex_empty_string[] = "";
static _Thread_local yvex_core_allocation_epoch allocation_epoch;
static _Thread_local yvex_core_execution_observation execution_observation;

/* Purpose: account one thread-local allocation event without wrapping the counter.
 * Inputs: epoch-owned counter.
 * Effects: increments the counter or latches overflow.
 * Failure: cannot fail; overflow remains observable in the epoch.
 * Boundary: records YVEX-owned allocator activity without changing allocation policy. */
static void allocation_epoch_advance(unsigned long long *counter)
{
    if (*counter == ULLONG_MAX)
        allocation_epoch.overflowed = 1;
    else
        ++*counter;
}

/* Purpose: snapshot exact YVEX allocator events on the current execution thread.
 * Inputs: optional caller-owned output.
 * Effects: copies thread-local facts when an output is present.
 * Failure: a null output is a harmless no-op.
 * Boundary: does not allocate, reset, or aggregate across threads. */
void yvex_core_allocation_epoch_snapshot(yvex_core_allocation_epoch *out)
{
    if (out) *out = allocation_epoch;
}

/* Purpose: execute one libc allocation operation through the observed YVEX owner.
 * Inputs: typed operation, optional owned pointer, and exact element geometry.
 * Effects: allocates, resizes, or releases storage and advances the matching TLS counter.
 * Failure: invalid operations or allocator failure return null without a success event.
 * Boundary: this is the sole non-public allocation symbol consumed by YVEX-owned code. */
void *yvex_core_allocate(yvex_core_allocation_operation operation,
                         void *pointer, size_t count, size_t size)
{
    void *result = NULL;
    switch (operation) {
    case YVEX_CORE_ALLOCATE_MALLOC:
        result = malloc(size);
        if (result) allocation_epoch_advance(&allocation_epoch.allocation_events);
        break;
    case YVEX_CORE_ALLOCATE_CALLOC:
        result = calloc(count, size);
        if (result) allocation_epoch_advance(&allocation_epoch.allocation_events);
        break;
    case YVEX_CORE_ALLOCATE_REALLOC:
        result = realloc(pointer, size);
        if (result) allocation_epoch_advance(&allocation_epoch.reallocation_events);
        else if (pointer && !size) allocation_epoch_advance(&allocation_epoch.release_events);
        break;
    case YVEX_CORE_ALLOCATE_FREE:
        if (pointer) allocation_epoch_advance(&allocation_epoch.release_events);
        free(pointer);
        break;
    }
    return result;
}

/* Purpose: record one observed upstream execution-planning event on the current thread.
 * Inputs: typed event and exact count.
 * Effects: advances one fact or latches observation overflow.
 * Failure: an invalid event kind fails closed by latching overflow.
 * Boundary: observation never changes the behavior of the observed owner. */
void yvex_core_execution_observation_record(
    yvex_core_execution_observation_kind kind, unsigned long long amount)
{
    unsigned long long *counter = NULL;
    switch (kind) {
    case YVEX_CORE_OBSERVE_SOURCE_HEADER:
        counter = &execution_observation.source_headers_read;
        break;
    case YVEX_CORE_OBSERVE_SOURCE_PAYLOAD_BYTES:
        counter = &execution_observation.source_payload_bytes_read;
        break;
    case YVEX_CORE_OBSERVE_TRANSFORM_PLAN:
        counter = &execution_observation.transform_plans_built;
        break;
    case YVEX_CORE_OBSERVE_QUANT_PLAN:
        counter = &execution_observation.quant_plans_built;
        break;
    case YVEX_CORE_OBSERVE_WRITER_PLAN:
        counter = &execution_observation.writer_plans_built;
        break;
    case YVEX_CORE_OBSERVE_COUNT:
        break;
    }
    if (!counter || *counter > ULLONG_MAX - amount)
        execution_observation.overflowed = 1;
    else
        *counter += amount;
}

/* Purpose: snapshot upstream execution-planning activity on the current thread.
 * Inputs: optional caller-owned output.
 * Effects: copies the current thread-local observation facts.
 * Failure: a null output is a harmless no-op.
 * Boundary: does not reset counters or inspect subsystem state. */
void yvex_core_execution_observation_snapshot(
    yvex_core_execution_observation *out)
{
    if (out) *out = execution_observation;
}

/* Purpose: derive monotonic observed work between two snapshots.
 * Inputs: before/after facts and required caller-owned output.
 * Effects: writes the exact field-wise delta on success.
 * Failure: null, overflowed, or regressed facts return false.
 * Boundary: compares observation counters without inspecting subsystem state. */
int yvex_core_execution_observation_delta(
    const yvex_core_execution_observation *before,
    const yvex_core_execution_observation *after,
    yvex_core_execution_observation *out)
{
    if (!before || !after || !out || before->overflowed || after->overflowed)
        return 0;
    if (after->source_headers_read < before->source_headers_read ||
        after->source_payload_bytes_read < before->source_payload_bytes_read ||
        after->transform_plans_built < before->transform_plans_built ||
        after->quant_plans_built < before->quant_plans_built ||
        after->writer_plans_built < before->writer_plans_built)
        return 0;
    memset(out, 0, sizeof(*out));
    out->source_headers_read = after->source_headers_read - before->source_headers_read;
    out->source_payload_bytes_read =
        after->source_payload_bytes_read - before->source_payload_bytes_read;
    out->transform_plans_built =
        after->transform_plans_built - before->transform_plans_built;
    out->quant_plans_built = after->quant_plans_built - before->quant_plans_built;
    out->writer_plans_built = after->writer_plans_built - before->writer_plans_built;
    return 1;
}

/* Purpose: copy borrowed text into one bounded terminated destination.
 * Inputs: optional source plus caller-owned destination and capacity.
 * Effects: writes a truncated copy and one terminating null byte when capacity permits.
 * Failure: null destination or zero capacity is a harmless no-op.
 * Boundary: copies bytes without parsing or assigning domain semantics. */
void yvex_core_text_copy(char *dst, size_t cap, const char *src)
{
    if (!dst || cap == 0) {
        return;
    }

    if (!src) {
        src = "";
    }

    snprintf(dst, cap, "%s", src);
    dst[cap - 1] = '\0';
}

/* Purpose: duplicate nullable text through one checked ownership mechanism.
 * Inputs: borrowed text; null is the canonical empty string.
 * Effects: returns one caller-owned allocation.
 * Failure: allocation or length overflow returns null without side effects.
 * Boundary: copying bytes does not validate or reinterpret domain text. */
char *yvex_core_strdup(const char *text)
{
    size_t length;
    char *copy;

    if (!text) {
        text = "";
    }
    length = strlen(text);
    if (length == SIZE_MAX) {
        return NULL;
    }
    copy = (char *)yvex_core_malloc(length + 1u);
    if (copy) {
        memcpy(copy, text, length + 1u);
    }
    return copy;
}

/* Purpose: add two 64-bit geometry facts without wraparound.
 * Inputs: immutable operands and required caller-owned output.
 * Effects: writes the exact sum only on success.
 * Failure: null output or overflow returns false without writing output.
 * Boundary: generic checked arithmetic; it owns no domain accounting policy. */
int yvex_core_u64_add(unsigned long long left,
                      unsigned long long right,
                      unsigned long long *out)
{
    if (!out || left > ULLONG_MAX - right) {
        return 0;
    }
    *out = left + right;
    return 1;
}

/* Purpose: multiply two 64-bit geometry facts without wraparound.
 * Inputs: immutable operands and required caller-owned output.
 * Effects: writes the exact product only on success.
 * Failure: null output or overflow returns false without writing output.
 * Boundary: generic checked arithmetic; it owns no domain accounting policy. */
int yvex_core_u64_mul(unsigned long long left,
                      unsigned long long right,
                      unsigned long long *out)
{
    if (!out || (left != 0ull && right > ULLONG_MAX / left)) {
        return 0;
    }
    *out = left * right;
    return 1;
}

/* Purpose: read one monotonic timestamp for lifecycle and benchmark evidence.
 * Inputs: process monotonic clock.
 * Effects: none beyond reading the operating-system clock.
 * Failure: unavailable clock returns zero without changing caller state.
 * Boundary: time is operational evidence and never enters semantic identities. */
unsigned long long yvex_core_monotonic_ns(void)
{
    struct timespec value;
    return clock_gettime(CLOCK_MONOTONIC, &value) == 0
               ? (unsigned long long)value.tv_sec * 1000000000ull +
                     (unsigned long long)value.tv_nsec
               : 0ull;
}

/* Purpose: Release or reset owned error clear state (`yvex_error_clear`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
void yvex_error_clear(yvex_error *err)
{
    if (!err) {
        return;
    }

    err->code = YVEX_OK;
    err->where[0] = '\0';
    err->message[0] = '\0';
}

/* Purpose: Compute error set for its core invariant (`yvex_error_set`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
void yvex_error_set(yvex_error *err, yvex_status code, const char *where, const char *message)
{
    if (!err) {
        return;
    }

    err->code = code;
    yvex_core_text_copy(err->where, YVEX_ERROR_WHERE_CAP, where ? where : "unknown");
    yvex_core_text_copy(err->message, YVEX_ERROR_MESSAGE_CAP, message);
}

/* Purpose: Compute error setf for its core invariant (`yvex_error_setf`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
void yvex_error_setf(yvex_error *err, yvex_status code, const char *where, const char *fmt, ...)
{
    va_list ap;

    if (!err) {
        return;
    }

    err->code = code;
    yvex_core_text_copy(err->where, YVEX_ERROR_WHERE_CAP, where ? where : "unknown");

    if (!fmt) {
        err->message[0] = '\0';
        return;
    }

    va_start(ap, fmt);
    vsnprintf(err->message, YVEX_ERROR_MESSAGE_CAP, fmt, ap);
    va_end(ap);
    err->message[YVEX_ERROR_MESSAGE_CAP - 1] = '\0';
}

/* Purpose: Compute error is set for its core invariant (`yvex_error_is_set`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_error_is_set(const yvex_error *err)
{
    if (!err) {
        return 0;
    }

    return err->code != YVEX_OK;
}

/* Purpose: Compute error code for its core invariant (`yvex_error_code`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
yvex_status yvex_error_code(const yvex_error *err)
{
    if (!err) {
        return YVEX_OK;
    }

    return err->code;
}

/* Purpose: Compute error where for its core invariant (`yvex_error_where`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_error_where(const yvex_error *err)
{
    if (!err) {
        return yvex_empty_string;
    }

    return err->where;
}

/* Purpose: Compute error message for its core invariant (`yvex_error_message`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_error_message(const yvex_error *err)
{
    if (!err) {
        return yvex_empty_string;
    }

    return err->message;
}

/* Purpose: Compute log level name for its core invariant (`yvex_log_level_name`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_log_level_name(yvex_log_level level)
{
    switch (level) {
    case YVEX_LOG_ERROR:
        return "error";
    case YVEX_LOG_WARN:
        return "warn";
    case YVEX_LOG_INFO:
        return "info";
    case YVEX_LOG_DEBUG:
        return "debug";
    case YVEX_LOG_TRACE:
        return "trace";
    default:
        return "unknown";
    }
}

/* Purpose: Compute log domain name for its core invariant (`yvex_log_domain_name`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_log_domain_name(yvex_log_domain domain)
{
    switch (domain) {
    case YVEX_LOG_CORE:
        return "core";
    case YVEX_LOG_CLI:
        return "cli";
    default:
        return "unknown";
    }
}

/* Purpose: Compute status name for its core invariant (`yvex_status_name`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_status_name(yvex_status status)
{
    switch (status) {
    case YVEX_OK:
        return "YVEX_OK";
    case YVEX_ERR:
        return "YVEX_ERR";
    case YVEX_ERR_NOMEM:
        return "YVEX_ERR_NOMEM";
    case YVEX_ERR_IO:
        return "YVEX_ERR_IO";
    case YVEX_ERR_FORMAT:
        return "YVEX_ERR_FORMAT";
    case YVEX_ERR_UNSUPPORTED:
        return "YVEX_ERR_UNSUPPORTED";
    case YVEX_ERR_BACKEND:
        return "YVEX_ERR_BACKEND";
    case YVEX_ERR_BOUNDS:
        return "YVEX_ERR_BOUNDS";
    case YVEX_ERR_STATE:
        return "YVEX_ERR_STATE";
    case YVEX_ERR_CANCELLED:
        return "YVEX_ERR_CANCELLED";
    case YVEX_ERR_INVALID_ARG:
        return "YVEX_ERR_INVALID_ARG";
    case YVEX_ERR_TIMEOUT:
        return "YVEX_ERR_TIMEOUT";
    default:
        return "YVEX_STATUS_UNKNOWN";
    }
}

/* Purpose: Compute status is ok for its core invariant (`yvex_status_is_ok`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_status_is_ok(yvex_status status)
{
    return status == YVEX_OK;
}

/* Purpose: Compute status is error for its core invariant (`yvex_status_is_error`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_status_is_error(yvex_status status)
{
    return status != YVEX_OK;
}

/* Purpose: Compute version string for its core invariant (`yvex_version_string`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
const char *yvex_version_string(void)
{
    return "0.1.0";
}

/* Purpose: Compute version major for its core invariant (`yvex_version_major`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_version_major(void)
{
    return YVEX_VERSION_MAJOR;
}

/* Purpose: Compute version minor for its core invariant (`yvex_version_minor`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_version_minor(void)
{
    return YVEX_VERSION_MINOR;
}

/* Purpose: Compute version patch for its core invariant (`yvex_version_patch`).
 * Inputs: Borrowed typed facts.
 * Effects: Mutates declared CLI state only.
 * Failure: Typed refusal; outputs remain defined.
 * Boundary: Core mechanism only. */
int yvex_version_patch(void)
{
    return YVEX_VERSION_PATCH;
}
