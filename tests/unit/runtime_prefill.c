/*
 * Exercises the tensor-file boundary before target-scale runtime execution. Canonical facts
 * admit identically from memory and file, while malformed files refuse. Focused internal-ABI
 * coverage; no fixture enters production objects.
 */
#define _GNU_SOURCE
#include "tests/test.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <yvex/internal/runtime_prefill.h>

typedef struct {
    yvex_runtime_activation_input_summary summary;
    yvex_runtime_activation_layer_record records[3];
    float payload[18];
} prefill_fixture;

static void prefill_identity(char output[YVEX_SHA256_HEX_CAP],
                             unsigned int value)
{
    (void)snprintf(output, YVEX_SHA256_HEX_CAP, "%064x", value);
}

static void prefill_fixture_open(prefill_fixture *fixture)
{
    const unsigned long long widths[] = {2ull, 3ull, 4ull};
    unsigned long long index, value, offset = 0ull;

    memset(fixture, 0, sizeof(*fixture));
    fixture->summary.schema_version =
        YVEX_RUNTIME_ACTIVATION_INPUT_SCHEMA_V1;
    fixture->summary.operation_scope = YVEX_ATTENTION_OPERATION_CORE;
    fixture->summary.token_start = 7ull;
    fixture->summary.token_count = 2ull;
    fixture->summary.layer_count = 3ull;
    prefill_identity(fixture->summary.logical_model_identity, 1u);
    prefill_identity(fixture->summary.runtime_numeric_identity, 2u);
    prefill_identity(fixture->summary.runtime_descriptor_identity, 3u);
    prefill_identity(fixture->summary.attention_plan_identity, 4u);
    for (index = 0ull; index < 3ull; ++index) {
        yvex_runtime_activation_layer_record *record =
            &fixture->records[index];
        record->ordinal = index;
        record->layer_index = index * 2ull;
        record->width = record->stride = widths[index];
        record->payload_offset = offset;
        record->payload_bytes =
            fixture->summary.token_count * widths[index] * sizeof(float);
        prefill_identity(record->layer_identity, (unsigned int)(10ull + index));
        offset += record->payload_bytes;
    }
    fixture->summary.payload_bytes = offset;
    for (value = 0ull; value < 18ull; ++value)
        fixture->payload[value] = (float)(value + 1ull) / 16.0f;
}

static int prefill_test_memory(void)
{
    prefill_fixture fixture, changed;
    yvex_runtime_activation_input *input = NULL;
    const yvex_runtime_activation_input_summary *summary;
    const float *view = NULL;
    unsigned long long stride = 0ull;
    yvex_error err;
    int rc;

    prefill_fixture_open(&fixture);
    rc = yvex_runtime_activation_input_seal(
        &fixture.summary, fixture.records, fixture.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "activation fixture seals");
    YVEX_TEST_ASSERT(
        yvex_sha256_hex_valid(fixture.summary.payload_digest) &&
            yvex_sha256_hex_valid(fixture.summary.input_identity),
        "sealed activation identities are canonical");
    rc = yvex_runtime_activation_input_open_memory(
        &input, &fixture.summary, fixture.records, fixture.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && input, "sealed memory input opens");
    summary = yvex_runtime_activation_input_summary_get(input);
    YVEX_TEST_ASSERT(
        summary && summary->token_start == 7ull &&
            summary->token_count == 2ull && summary->layer_count == 3ull,
        "memory input preserves typed summary");
    rc = yvex_runtime_activation_input_view(
        input, 1ull, 1ull, 1ull, &view, &stride, &err);
    YVEX_TEST_ASSERT(
        rc == YVEX_OK && stride == 3ull && view &&
            view[0] == fixture.payload[7],
        "checked view resolves the exact layer token");
    rc = yvex_runtime_activation_input_view(
        input, 1ull, 2ull, 1ull, &view, &stride, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS, "out-of-range token view refuses");
    yvex_runtime_activation_input_close(&input);
    yvex_runtime_activation_input_close(&input);

    changed = fixture;
    changed.records[2].layer_index = changed.records[0].layer_index;
    rc = yvex_runtime_activation_input_seal(
        &changed.summary, changed.records, changed.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "duplicate layer refuses");
    changed = fixture;
    changed.records[1].stride++;
    rc = yvex_runtime_activation_input_seal(
        &changed.summary, changed.records, changed.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "ambiguous stride refuses");
    changed = fixture;
    changed.payload[3] = NAN;
    rc = yvex_runtime_activation_input_seal(
        &changed.summary, changed.records, changed.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "non-finite input refuses");
    return 0;
}

static int prefill_append_byte(const char *path)
{
    const unsigned char value = 0xa5u;
    int fd = open(path, O_WRONLY | O_APPEND | O_CLOEXEC);
    int ok = fd >= 0 && write(fd, &value, 1u) == 1;
    if (fd >= 0 && close(fd) != 0) ok = 0;
    return ok;
}

/* Prove secure bounded file admission, drift refusal, and cleanup. */
static int prefill_test_file(void)
{
    prefill_fixture fixture;
    yvex_runtime_activation_input *input = NULL;
    yvex_runtime_activation_input_limits limits;
    char root[] = "/tmp/yvex-runtime-prefill.XXXXXX";
    char path[512], link_path[512], trailing_path[512];
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(mkdtemp(root) != NULL, "prefill temporary root opens");
    YVEX_TEST_ASSERT(
        snprintf(path, sizeof(path), "%s/input%s", root,
                 YVEX_RUNTIME_ACTIVATION_INPUT_SUFFIX) < (int)sizeof(path) &&
            snprintf(link_path, sizeof(link_path), "%s/link%s", root,
                     YVEX_RUNTIME_ACTIVATION_INPUT_SUFFIX) <
                (int)sizeof(link_path) &&
            snprintf(trailing_path, sizeof(trailing_path), "%s/trailing%s",
                     root, YVEX_RUNTIME_ACTIVATION_INPUT_SUFFIX) <
                (int)sizeof(trailing_path),
        "prefill fixture paths fit");
    prefill_fixture_open(&fixture);
    rc = yvex_runtime_activation_input_seal(
        &fixture.summary, fixture.records, fixture.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "file activation fixture seals");
    rc = yvex_runtime_activation_input_write(
        path, &fixture.summary, fixture.records, fixture.payload, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "activation tensor file publishes");
    limits.maximum_file_bytes = 64ull * 1024ull;
    rc = yvex_runtime_activation_input_open_file(
        &input, path, &limits, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && input, "bounded tensor file admits");
    YVEX_TEST_ASSERT(
        yvex_runtime_activation_input_validate(input, &err) == YVEX_OK,
        "unchanged open input validates");
    YVEX_TEST_ASSERT(prefill_append_byte(path), "open input can be drifted");
    YVEX_TEST_ASSERT(
        yvex_runtime_activation_input_validate(input, &err) == YVEX_ERR_STATE,
        "open-handle snapshot detects drift");
    yvex_runtime_activation_input_close(&input);

    rc = yvex_runtime_activation_input_write(
        trailing_path, &fixture.summary, fixture.records, fixture.payload,
        &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "trailing fixture publishes");
    YVEX_TEST_ASSERT(prefill_append_byte(trailing_path),
                     "trailing fixture extends");
    rc = yvex_runtime_activation_input_open_file(
        &input, trailing_path, &limits, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT && !input,
                     "trailing bytes refuse admission");
    YVEX_TEST_ASSERT(symlink(trailing_path, link_path) == 0,
                     "activation symlink fixture opens");
    rc = yvex_runtime_activation_input_open_file(
        &input, link_path, &limits, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_IO && !input,
                     "activation symlink refuses");
    limits.maximum_file_bytes = 32ull;
    rc = yvex_runtime_activation_input_open_file(
        &input, trailing_path, &limits, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_BOUNDS && !input,
                     "activation resource budget refuses");
    YVEX_TEST_ASSERT(
        unlink(link_path) == 0 && unlink(trailing_path) == 0 &&
            unlink(path) == 0 && rmdir(root) == 0,
        "activation fixtures clean exactly");
    return 0;
}

int yvex_test_runtime_prefill(void)
{
    if (prefill_test_memory() != 0) return 1;
    if (prefill_test_file() != 0) return 1;
    return 0;
}
