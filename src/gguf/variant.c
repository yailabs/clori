/* Owner: gguf.physical_variant serialization (TRACK.COMPILATION).
 * Owns: canonical pointer-free physical-variant plan bytes, transactional publication, and exact revalidation.
 * Does not own: policy resolution, IR construction, quant execution, GGUF writing, materialization, or CLI output.
 * Invariants: serialization includes every ordered terminal decision and excludes paths, pointers, padding, and time.
 * Boundary: an external plan is executable only after regeneration from admitted logical/policy facts and byte match.
 * Purpose: preserve one sealed physical decision set across planning and emission without creating a second planner.
 * Inputs: an immutable complete quant plan and an explicit regular-file path.
 * Effects: writes or validates one bounded canonical plan file.
 * Failure: partial temporary files are removed and mismatched files never admit a plan. */
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/quant_numeric.h>

#define VARIANT_FILE_MAX_BYTES (8u * 1024u * 1024u)

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} variant_buffer;

/* Purpose: publish one stable physical-variant file refusal.
 * Inputs: error sink, status, and semantic message.
 * Effects: replaces only caller diagnostic state.
 * Failure: returns the supplied status.
 * Boundary: diagnostics own no file or plan resources. */
static int variant_fail(yvex_error *err, yvex_status status, const char *message) {
    yvex_error_set(err, status, "quant.variant_file", message);
    return status;
}

/* Purpose: reserve bounded serialization capacity with checked geometric growth.
 * Inputs: mutable buffer and required total byte count.
 * Effects: may replace the buffer allocation while retaining existing bytes.
 * Failure: budget, arithmetic, or allocation failure returns false.
 * Boundary: capacity is serialization workspace and never semantic identity. */
static int variant_buffer_reserve(variant_buffer *buffer, size_t required) {
    size_t capacity;
    char *next;

    if (!buffer || required > VARIANT_FILE_MAX_BYTES)
        return 0;
    if (required <= buffer->capacity)
        return 1;
    capacity = buffer->capacity ? buffer->capacity : 4096u;
    while (capacity < required) {
        if (capacity > VARIANT_FILE_MAX_BYTES / 2u) {
            capacity = VARIANT_FILE_MAX_BYTES;
            break;
        }
        capacity *= 2u;
    }
    next = (char *)realloc(buffer->data, capacity);
    if (!next)
        return 0;
    buffer->data = next;
    buffer->capacity = capacity;
    return 1;
}

/* Purpose: append one checked formatted record to canonical plan bytes.
 * Inputs: mutable buffer and format arguments containing admitted scalar/text facts.
 * Effects: appends one record and maintains a trailing NUL outside semantic length.
 * Failure: formatting, bound, or allocation failure returns false.
 * Boundary: caller supplies canonical field order and escaping-free identity text. */
static int variant_buffer_append(variant_buffer *buffer, const char *format, ...) {
    va_list arguments;
    va_list copy;
    int count;
    size_t required;

    va_start(arguments, format);
    va_copy(copy, arguments);
    count = vsnprintf(NULL, 0u, format, copy);
    va_end(copy);
    if (count < 0 || (size_t)count > SIZE_MAX - buffer->length - 1u) {
        va_end(arguments);
        return 0;
    }
    required = buffer->length + (size_t)count + 1u;
    if (!variant_buffer_reserve(buffer, required)) {
        va_end(arguments);
        return 0;
    }
    if (vsnprintf(buffer->data + buffer->length, required - buffer->length, format, arguments) !=
        count) {
        va_end(arguments);
        return 0;
    }
    va_end(arguments);
    buffer->length += (size_t)count;
    return 1;
}

/* Purpose: append one complete terminal decision using explicit scalar fields.
 * Inputs: buffer and immutable decision in canonical ordinal order.
 * Effects: appends one newline-terminated decision record.
 * Failure: invalid rank or workspace exhaustion returns false.
 * Boundary: native structure bytes and pointer values never enter serialization. */
static int variant_decision_append(variant_buffer *buffer, const yvex_quant_decision *decision) {
    unsigned int axis;

    if (!decision || decision->rank > YVEX_GGUF_QTYPE_MAX_DIMS ||
        !variant_buffer_append(
            buffer,
            "decision=%llu,%llu,%llu,%u,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%d,%u,%u,%d,%llu,%u,%d,%s,%s,%llu",
            decision->terminal_ordinal, decision->terminal_value_id, decision->node_id,
            (unsigned int)decision->role, (unsigned int)decision->collection,
            (unsigned int)decision->scope, (unsigned int)decision->operation, decision->qtype,
            decision->row_width, decision->row_count, decision->element_count,
            decision->encoded_bytes, decision->approximation, (unsigned int)decision->calibration,
            decision->numeric_contract_version, decision->policy_bound,
            decision->policy_rule_ordinal, decision->policy_priority,
            decision->policy_requires_imatrix, decision->policy_rule_identity,
            decision->decision_identity, decision->physical_expert_count))
        return 0;
    for (axis = 0u; axis < decision->rank; ++axis)
        if (!variant_buffer_append(buffer, ",%llu", decision->dims[axis]))
            return 0;
    return variant_buffer_append(buffer, ",%u,%s,%s\n", decision->row_axis,
                                 decision->physical_tensor_name, decision->policy_label);
}

/* Purpose: serialize one sealed physical plan in its canonical pointer-free field order.
 * Inputs: complete immutable plan and output buffer owner.
 * Effects: allocates and fills the exact external plan bytes.
 * Failure: incomplete plan, malformed decision, or workspace failure publishes no output bytes.
 * Boundary: serialization records decisions and identities but never resolves policy. */
static int variant_serialize(const yvex_quant_plan *plan, variant_buffer *buffer,
                             yvex_error *err) {
    const yvex_quant_plan_summary *summary = yvex_quant_plan_summary_get(plan);
    unsigned long long ordinal;

    memset(buffer, 0, sizeof(*buffer));
    if (!summary || !summary->complete || summary->state != YVEX_QUANT_PLAN_SEALED ||
        summary->decision_count == 0u ||
        !variant_buffer_append(buffer, "yvex.physical_variant_plan.v%u\n",
                               YVEX_PHYSICAL_VARIANT_FILE_SCHEMA_VERSION) ||
        !variant_buffer_append(buffer, "plan_schema=%u\n", summary->schema_version) ||
        !variant_buffer_append(buffer, "profile_name=%s\n", summary->profile_name) ||
        !variant_buffer_append(buffer, "profile_identity=%s\n", summary->profile_identity) ||
        !variant_buffer_append(buffer, "policy_identity=%s\n", summary->policy_identity) ||
        !variant_buffer_append(buffer, "imatrix_identity=%s\n", summary->imatrix_identity) ||
        !variant_buffer_append(buffer, "physical_variant_identity=%s\n",
                               summary->physical_variant_identity) ||
        !variant_buffer_append(buffer, "payload_plan_identity=%s\n",
                               summary->payload_plan_identity) ||
        !variant_buffer_append(buffer, "transform_identity=%s\n", summary->transform_identity) ||
        !variant_buffer_append(buffer, "required_payload_identity=%s\n",
                               summary->required_payload_identity) ||
        !variant_buffer_append(buffer, "source_snapshot_identity=%llu\n",
                               summary->source_snapshot_identity) ||
        !variant_buffer_append(buffer, "mapping_identity=%llu\n", summary->mapping_identity) ||
        !variant_buffer_append(buffer, "decision_count=%llu\n", summary->decision_count) ||
        !variant_buffer_append(buffer, "encoded_bytes=%llu\n", summary->encoded_bytes)) {
        free(buffer->data);
        memset(buffer, 0, sizeof(*buffer));
        return variant_fail(err, YVEX_ERR_BOUNDS,
                            "complete bounded physical plan serialization failed");
    }
    for (ordinal = 0u; ordinal < summary->decision_count; ++ordinal) {
        const yvex_quant_decision *decision = yvex_quant_plan_decision_at(plan, ordinal);
        if (!decision || decision->terminal_ordinal != ordinal ||
            !variant_decision_append(buffer, decision)) {
            free(buffer->data);
            memset(buffer, 0, sizeof(*buffer));
            return variant_fail(err, YVEX_ERR_FORMAT,
                                "physical plan has an incomplete ordered decision");
        }
    }
    return YVEX_OK;
}

/* Purpose: transactionally publish one canonical physical-variant plan file.
 * Inputs: destination path, sealed plan, and error sink.
 * Effects: delegates durable no-replace publication to the canonical core file lifecycle.
 * Failure: destination conflict or typed file-lifecycle failure removes temporary state.
 * Boundary: the plan owner and all borrowed logical facts remain unchanged. */
int yvex_quant_plan_file_write(const char *path, const yvex_quant_plan *plan, yvex_error *err) {
    variant_buffer buffer;
    yvex_core_file_result result;
    int rc;

    if (!path || !path[0] || strlen(path) >= YVEX_PATH_CAP)
        return variant_fail(err, YVEX_ERR_INVALID_ARG, "bounded output path is required");
    rc = variant_serialize(plan, &buffer, err);
    if (rc != YVEX_OK)
        return rc;
    memset(&result, 0, sizeof(result));
    rc = yvex_core_file_publish_noreplace(path, buffer.data, buffer.length, NULL, NULL, NULL,
                                          &result, err);
    free(buffer.data);
    return rc;
}

/* Purpose: validate one external plan as the exact serialization of a regenerated sealed plan.
 * Inputs: regular-file path, independently rebuilt plan, and error sink.
 * Effects: reads bounded bytes and compares the complete canonical representation.
 * Failure: symlink, drift, truncation, trailing bytes, or any field mutation refuses.
 * Boundary: validation never imports decisions without their logical/policy authority. */
int yvex_quant_plan_file_validate(const char *path, const yvex_quant_plan *plan, yvex_error *err) {
    variant_buffer expected;
    yvex_core_file_result result;
    unsigned char *actual = NULL;
    size_t actual_count = 0u;
    int rc;

    if (!path || !path[0])
        return variant_fail(err, YVEX_ERR_INVALID_ARG, "physical plan path is required");
    rc = variant_serialize(plan, &expected, err);
    if (rc != YVEX_OK)
        return rc;
    memset(&result, 0, sizeof(result));
    rc = yvex_core_file_read_snapshot(path, VARIANT_FILE_MAX_BYTES, &actual, &actual_count,
                                      &result, err);
    if (rc != YVEX_OK)
        goto done;
    if (actual_count != expected.length || memcmp(actual, expected.data, expected.length) != 0) {
        rc = variant_fail(err, YVEX_ERR_FORMAT,
                          "physical plan bytes do not match regenerated policy decisions");
        goto done;
    }
    rc = YVEX_OK;
    yvex_error_clear(err);

done:
    free(actual);
    free(expected.data);
    return rc;
}
