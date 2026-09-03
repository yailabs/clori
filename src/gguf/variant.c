/*
 * Preserve one sealed physical decision set across planning and emission without creating a second
 * planner.
 *
 * Serialization includes every ordered terminal decision and excludes paths, pointers, padding,
 * and time. An external plan is executable only after regeneration from admitted logical/policy
 * facts and byte match.
 */
#include <stdarg.h>
#include <errno.h>
#include <limits.h>
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

static int variant_fail(yvex_error *err, yvex_status status, const char *message) {
    yvex_error_set(err, status, "quant.variant_file", message);
    return status;
}

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

/*
 * Append one checked formatted record to canonical plan bytes.
 *
 * Caller supplies canonical field order and escaping-free identity text.
 */
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

static int variant_next_line(const unsigned char *bytes, size_t count,
                             size_t *cursor, const char **line,
                             size_t *line_count)
{
    size_t begin, end;

    if (!bytes || !cursor || !line || !line_count || *cursor >= count)
        return 0;
    begin = *cursor;
    for (end = begin; end < count && bytes[end] != '\n'; ++end) {
        if (bytes[end] == '\0' || bytes[end] == '\r') return 0;
    }
    if (end >= count) return 0;
    *line = (const char *)bytes + begin;
    *line_count = end - begin;
    *cursor = end + 1u;
    return 1;
}

static int variant_text_line(const char *line, size_t count, const char *key,
                             char *out, size_t capacity)
{
    size_t key_count = strlen(key), value_count;

    if (!line || !out || capacity == 0u || count < key_count + 1u ||
        memcmp(line, key, key_count) != 0 || line[key_count] != '=')
        return 0;
    value_count = count - key_count - 1u;
    if (value_count >= capacity) return 0;
    memcpy(out, line + key_count + 1u, value_count);
    out[value_count] = '\0';
    return 1;
}

static int variant_u64_line(const char *line, size_t count, const char *key,
                            unsigned long long *out)
{
    char value[32], *end = NULL;
    unsigned long long parsed;

    if (!out || !variant_text_line(line, count, key, value, sizeof(value)))
        return 0;
    if (value[0] < '0' || value[0] > '9') return 0;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || !end || *end) return 0;
    *out = parsed;
    return 1;
}

static int variant_probe_header(const unsigned char *bytes, size_t count,
                                size_t *cursor,
                                yvex_quant_plan_file_summary *summary)
{
    static const char magic[] = "yvex.physical_variant_plan.v1";
    const char *line;
    size_t line_count;
    unsigned long long plan_schema;

#define NEXT_LINE() variant_next_line(bytes, count, cursor, &line, &line_count)
    if (!NEXT_LINE() || line_count != sizeof(magic) - 1u ||
        memcmp(line, magic, sizeof(magic) - 1u) != 0 ||
        !NEXT_LINE() || !variant_u64_line(line, line_count, "plan_schema", &plan_schema) ||
        plan_schema > UINT_MAX ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "profile_name",
                                            summary->profile_name,
                                            sizeof(summary->profile_name)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "profile_identity",
                                            summary->profile_identity,
                                            sizeof(summary->profile_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "policy_identity",
                                            summary->policy_identity,
                                            sizeof(summary->policy_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "imatrix_identity",
                                            summary->imatrix_identity,
                                            sizeof(summary->imatrix_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count,
                                            "physical_variant_identity",
                                            summary->physical_variant_identity,
                                            sizeof(summary->physical_variant_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "payload_plan_identity",
                                            summary->payload_plan_identity,
                                            sizeof(summary->payload_plan_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count, "transform_identity",
                                            summary->transform_identity,
                                            sizeof(summary->transform_identity)) ||
        !NEXT_LINE() || !variant_text_line(line, line_count,
                                            "required_payload_identity",
                                            summary->required_payload_identity,
                                            sizeof(summary->required_payload_identity)) ||
        !NEXT_LINE() || !variant_u64_line(line, line_count,
                                           "source_snapshot_identity",
                                           &summary->source_snapshot_identity) ||
        !NEXT_LINE() || !variant_u64_line(line, line_count, "mapping_identity",
                                           &summary->mapping_identity) ||
        !NEXT_LINE() || !variant_u64_line(line, line_count, "decision_count",
                                           &summary->decision_count) ||
        !NEXT_LINE() || !variant_u64_line(line, line_count, "encoded_bytes",
                                           &summary->encoded_bytes))
        return 0;
#undef NEXT_LINE
    summary->file_schema_version = YVEX_PHYSICAL_VARIANT_FILE_SCHEMA_VERSION;
    summary->plan_schema_version = (unsigned int)plan_schema;
    return summary->plan_schema_version != 0u && summary->decision_count != 0ull &&
           summary->encoded_bytes != 0ull &&
           yvex_sha256_hex_valid(summary->profile_identity) &&
           (!summary->policy_identity[0] ||
            yvex_sha256_hex_valid(summary->policy_identity)) &&
           (!summary->imatrix_identity[0] ||
            !strcmp(summary->imatrix_identity, "none") ||
            yvex_sha256_hex_valid(summary->imatrix_identity)) &&
           yvex_sha256_hex_valid(summary->physical_variant_identity) &&
           yvex_sha256_hex_valid(summary->payload_plan_identity) &&
           yvex_sha256_hex_valid(summary->transform_identity) &&
           yvex_sha256_hex_valid(summary->required_payload_identity);
}

static int variant_probe_decisions(const unsigned char *bytes, size_t count,
                                   size_t cursor,
                                   unsigned long long expected_count)
{
    const char *line;
    size_t line_count;
    unsigned long long ordinal = 0ull;

    while (variant_next_line(bytes, count, &cursor, &line, &line_count)) {
        char value[32], *end = NULL;
        size_t digit_count = 0u;
        unsigned long long parsed;

        if (line_count <= sizeof("decision=") - 1u ||
            memcmp(line, "decision=", sizeof("decision=") - 1u) != 0)
            return 0;
        while (sizeof("decision=") - 1u + digit_count < line_count &&
               line[sizeof("decision=") - 1u + digit_count] >= '0' &&
               line[sizeof("decision=") - 1u + digit_count] <= '9')
            digit_count++;
        if (!digit_count || digit_count >= sizeof(value) ||
            sizeof("decision=") - 1u + digit_count >= line_count ||
            line[sizeof("decision=") - 1u + digit_count] != ',')
            return 0;
        memcpy(value, line + sizeof("decision=") - 1u, digit_count);
        value[digit_count] = '\0';
        errno = 0;
        parsed = strtoull(value, &end, 10);
        if (errno || !end || *end || parsed != ordinal) return 0;
        ordinal++;
    }
    return cursor == count && ordinal == expected_count;
}

int yvex_quant_plan_file_probe(const char *path,
                               yvex_quant_plan_file_summary *summary,
                               yvex_error *err)
{
    yvex_core_file_result result;
    unsigned char *bytes = NULL;
    size_t count = 0u, cursor = 0u;
    int rc;

    if (!path || !path[0] || !summary)
        return variant_fail(err, YVEX_ERR_INVALID_ARG,
                            "physical plan path and summary are required");
    memset(summary, 0, sizeof(*summary));
    memset(&result, 0, sizeof(result));
    rc = yvex_core_file_read_snapshot(path, VARIANT_FILE_MAX_BYTES, &bytes,
                                      &count, &result, err);
    if (rc == YVEX_OK && !variant_probe_header(bytes, count, &cursor, summary))
        rc = variant_fail(err, YVEX_ERR_FORMAT,
                          "physical plan identity header is malformed");
    if (rc == YVEX_OK &&
        !variant_probe_decisions(bytes, count, cursor,
                                 summary->decision_count))
        rc = variant_fail(err, YVEX_ERR_FORMAT,
                          "physical plan decision rows are malformed");
    free(bytes);
    if (rc == YVEX_OK) {
        summary->complete = 1;
        yvex_error_clear(err);
    }
    return rc;
}

static int variant_physical_headers_equal(
    const yvex_quant_plan_file_summary *creation,
    const yvex_quant_plan_file_summary *current)
{
    return creation && current && creation->complete && current->complete &&
           creation->file_schema_version == current->file_schema_version &&
           creation->plan_schema_version == current->plan_schema_version &&
           !strcmp(creation->profile_name, current->profile_name) &&
           !strcmp(creation->policy_identity, current->policy_identity) &&
           !strcmp(creation->imatrix_identity, current->imatrix_identity) &&
           !strcmp(creation->payload_plan_identity,
                   current->payload_plan_identity) &&
           !strcmp(creation->required_payload_identity,
                   current->required_payload_identity) &&
           creation->source_snapshot_identity == current->source_snapshot_identity &&
           creation->mapping_identity == current->mapping_identity &&
           creation->decision_count == current->decision_count &&
           creation->encoded_bytes == current->encoded_bytes &&
           !strcmp(creation->profile_identity,
                   creation->physical_variant_identity) &&
           !strcmp(current->profile_identity,
                   current->physical_variant_identity);
}

int yvex_quant_plan_file_validate_physical_equivalence(
    const char *path, const yvex_quant_plan *plan,
    yvex_quant_plan_file_summary *creation, yvex_error *err)
{
    yvex_quant_plan_file_summary actual_summary = {0}, expected_summary = {0};
    variant_buffer expected;
    yvex_core_file_result result;
    unsigned char *actual = NULL;
    size_t actual_count = 0u, actual_cursor = 0u, expected_cursor = 0u;
    int rc;

    if (creation) memset(creation, 0, sizeof(*creation));
    if (!path || !path[0] || !plan)
        return variant_fail(err, YVEX_ERR_INVALID_ARG,
                            "physical plan path and current plan are required");
    rc = variant_serialize(plan, &expected, err);
    if (rc != YVEX_OK) return rc;
    memset(&result, 0, sizeof(result));
    rc = yvex_core_file_read_snapshot(path, VARIANT_FILE_MAX_BYTES, &actual,
                                      &actual_count, &result, err);
    if (rc != YVEX_OK) goto done;
    if (!variant_probe_header(actual, actual_count, &actual_cursor,
                              &actual_summary) ||
        !variant_probe_header((const unsigned char *)expected.data,
                              expected.length, &expected_cursor,
                              &expected_summary) ||
        !variant_probe_decisions(actual, actual_count, actual_cursor,
                                 actual_summary.decision_count) ||
        !variant_probe_decisions((const unsigned char *)expected.data,
                                 expected.length, expected_cursor,
                                 expected_summary.decision_count)) {
        rc = variant_fail(err, YVEX_ERR_FORMAT,
                          "physical plan creation or current decisions are malformed");
        goto done;
    }
    actual_summary.complete = 1;
    expected_summary.complete = 1;
    if (!variant_physical_headers_equal(&actual_summary, &expected_summary) ||
        actual_count - actual_cursor != expected.length - expected_cursor ||
        memcmp(actual + actual_cursor, expected.data + expected_cursor,
               expected.length - expected_cursor) != 0) {
        rc = variant_fail(
            err, YVEX_ERR_FORMAT,
            "physical plan decisions do not match current policy and source semantics");
        goto done;
    }
    if (creation) *creation = actual_summary;
    rc = YVEX_OK;
    yvex_error_clear(err);

done:
    free(actual);
    free(expected.data);
    return rc;
}
