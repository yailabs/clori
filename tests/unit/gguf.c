/*
 * Exercises GGUF probing, fixed header parsing, metadata parsing, tensor directory parsing,
 * lookup helpers, and malformed fixture handling.
 */
#include <string.h>

#include <yvex/artifact.h>
#include <yvex/gguf.h>

#include "tests/test.h"

static int open_fixture(const char *path, yvex_artifact **out)
{
    yvex_artifact_options options;
    yvex_error err;
    int rc;

    memset(&options, 0, sizeof(options));
    options.path = path;
    options.readonly = 1;

    rc = yvex_artifact_open(out, &options, &err);
    if (rc != YVEX_OK) {
        fprintf(stderr, "open fixture failed: %s: %s\n", yvex_error_where(&err), yvex_error_message(&err));
        return 1;
    }
    return 0;
}

static int expect_gguf_open_status(const char *path, int expected_status)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture(path, &artifact) == 0, "open malformed fixture");
    rc = yvex_gguf_open(&gguf, artifact, &err);
    if (rc != expected_status) {
        fprintf(stderr, "expected %d for %s, got %d: %s: %s\n",
                expected_status, path, rc, yvex_error_where(&err), yvex_error_message(&err));
        yvex_gguf_close(gguf);
        yvex_artifact_close(artifact);
        return 1;
    }
    YVEX_TEST_ASSERT(gguf == NULL, "failed open leaves gguf null");
    yvex_artifact_close(artifact);
    return 0;
}

static int test_valid_minimal(void)
{
    yvex_artifact *artifact;
    yvex_gguf *gguf = NULL;
    yvex_gguf_header header;
    yvex_gguf_probe probe;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/valid-minimal.gguf", &artifact) == 0, "open valid fixture");

    rc = yvex_gguf_read_header(artifact, &header, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid header parses");
    YVEX_TEST_ASSERT(header.version == 3, "valid version");
    YVEX_TEST_ASSERT(header.tensor_count == 0, "valid tensor count");
    YVEX_TEST_ASSERT(header.metadata_count == 0, "valid metadata count");

    rc = yvex_gguf_probe_file(artifact, &probe, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid probe ok");
    YVEX_TEST_ASSERT(probe.is_gguf == 1, "valid probe is gguf");
    YVEX_TEST_ASSERT(probe.header.version == 3, "probe version");

    rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid minimal directory parses");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_count(gguf) == 0, "minimal metadata count");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_count(gguf) == 0, "minimal tensor count");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_data_offset(gguf) == 32ull,
                     "minimal tensor-data boundary uses default alignment");

    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_bad_magic(void)
{
    yvex_artifact *artifact;
    yvex_gguf_header header;
    yvex_gguf_probe probe;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/bad-magic.gguf", &artifact) == 0, "open bad magic fixture");

    rc = yvex_gguf_probe_file(artifact, &probe, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "bad magic probe ok");
    YVEX_TEST_ASSERT(probe.is_gguf == 0, "bad magic probe unknown");

    rc = yvex_gguf_read_header(artifact, &header, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "bad magic read format error");

    yvex_artifact_close(artifact);
    return 0;
}

static int test_short_header(void)
{
    yvex_artifact *artifact;
    yvex_gguf_header header;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/short-header.gguf", &artifact) == 0, "open short fixture");
    rc = yvex_gguf_read_header(artifact, &header, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_FORMAT, "short header format error");
    yvex_artifact_close(artifact);
    return 0;
}

static int test_unsupported_version(void)
{
    yvex_artifact *artifact;
    yvex_gguf_header header;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/unsupported-version.gguf", &artifact) == 0, "open unsupported fixture");
    rc = yvex_gguf_read_header(artifact, &header, &err);
    YVEX_TEST_ASSERT(rc == YVEX_ERR_UNSUPPORTED, "unsupported version");
    yvex_artifact_close(artifact);
    return 0;
}

static int test_valid_metadata_and_tensors(void)
{
    yvex_artifact *artifact;
    yvex_gguf *gguf = NULL;
    const yvex_gguf_header *header;
    const yvex_gguf_value *value;
    const yvex_gguf_tensor_info *tensor;
    const char *text;
    unsigned long long text_len;
    unsigned long long u64;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/valid-metadata-tensors.gguf", &artifact) == 0, "open GGUF parser valid fixture");

    rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "GGUF parser valid fixture parses");

    header = yvex_gguf_header_view(gguf);
    YVEX_TEST_ASSERT(header != NULL, "header view exists");
    YVEX_TEST_ASSERT(header->version == 3, "GGUF parser version");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_count(gguf) == 5, "GGUF parser metadata count");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_count(gguf) == 1, "GGUF parser tensor count");
    YVEX_TEST_ASSERT(yvex_gguf_alignment(gguf) == 32, "alignment is 32");
    YVEX_TEST_ASSERT((yvex_gguf_tensor_data_offset(gguf) % 32) == 0, "tensor data offset aligned");

    value = yvex_gguf_metadata_find(gguf, "general.architecture");
    YVEX_TEST_ASSERT(value != NULL, "find architecture");
    YVEX_TEST_ASSERT(yvex_gguf_value_type_of(value) == YVEX_GGUF_VALUE_STRING, "architecture is string");
    YVEX_TEST_ASSERT(yvex_gguf_value_as_string(value, &text, &text_len) == YVEX_OK, "architecture string accessor");
    YVEX_TEST_ASSERT(text_len == 5, "architecture string len");
    YVEX_TEST_ASSERT(strncmp(text, "llama", 5) == 0, "architecture string value");

    value = yvex_gguf_metadata_find(gguf, "general.name");
    YVEX_TEST_ASSERT(value != NULL, "find name");
    YVEX_TEST_ASSERT(yvex_gguf_value_as_string(value, &text, &text_len) == YVEX_OK, "name string accessor");
    YVEX_TEST_ASSERT_STREQ(text, "yvex-test", "name string value");

    value = yvex_gguf_metadata_find(gguf, "llama.context_length");
    YVEX_TEST_ASSERT(value != NULL, "find context length");
    YVEX_TEST_ASSERT(yvex_gguf_value_as_u64(value, &u64) == YVEX_OK, "context u64 accessor");
    YVEX_TEST_ASSERT(u64 == 4096, "context value");

    value = yvex_gguf_metadata_find(gguf, "general.file_type");
    YVEX_TEST_ASSERT(value != NULL, "find file type");
    YVEX_TEST_ASSERT(yvex_gguf_value_as_u64(value, &u64) == YVEX_OK, "file type accessor");
    YVEX_TEST_ASSERT(u64 == 0, "file type value");

    YVEX_TEST_ASSERT(yvex_gguf_metadata_find(gguf, "missing.key") == NULL, "missing metadata returns null");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_key(gguf, 999) == NULL, "metadata key out of range null");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_value(gguf, 999) == NULL, "metadata value out of range null");

    tensor = yvex_gguf_tensor_at(gguf, 0);
    YVEX_TEST_ASSERT(tensor != NULL, "tensor 0 exists");
    YVEX_TEST_ASSERT_STREQ(tensor->name, "token_embd.weight", "tensor name");
    YVEX_TEST_ASSERT(tensor->rank == 2, "tensor rank");
    YVEX_TEST_ASSERT(tensor->dims[0] == 4, "tensor dim 0");
    YVEX_TEST_ASSERT(tensor->dims[1] == 8, "tensor dim 1");
    YVEX_TEST_ASSERT(tensor->ggml_type == 0, "tensor raw type");
    YVEX_TEST_ASSERT_STREQ(tensor->ggml_type_name, "F32", "tensor type name");
    YVEX_TEST_ASSERT(tensor->relative_offset == 0, "tensor relative offset");
    YVEX_TEST_ASSERT(tensor->absolute_offset == yvex_gguf_tensor_data_offset(gguf), "tensor absolute offset");

    tensor = yvex_gguf_tensor_find(gguf, "token_embd.weight");
    YVEX_TEST_ASSERT(tensor != NULL, "tensor find works");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_find(gguf, "missing.weight") == NULL, "missing tensor returns null");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_at(gguf, 999) == NULL, "tensor out of range null");

    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_malformed_metadata(void)
{
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-unknown-type.gguf", YVEX_ERR_UNSUPPORTED) == 0, "unknown metadata type fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-string-oob.gguf", YVEX_ERR_FORMAT) == 0, "metadata string out of bounds fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-array-oob.gguf", YVEX_ERR_BOUNDS) == 0, "metadata array out of bounds fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-nested.gguf", YVEX_ERR_BOUNDS) == 0, "truncated nested array fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-bool-invalid.gguf", YVEX_ERR_FORMAT) == 0, "invalid bool fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/metadata-empty-key.gguf", YVEX_ERR_FORMAT) == 0, "empty key fails");
    return 0;
}

static int test_malformed_tensors(void)
{
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-name-out-of-bounds.gguf", YVEX_ERR_FORMAT) == 0, "tensor name out of bounds fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-rank-zero.gguf", YVEX_ERR_FORMAT) == 0, "tensor rank zero fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-rank-unsupported.gguf", YVEX_ERR_FORMAT) == 0, "tensor rank unsupported fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-dim-zero.gguf", YVEX_ERR_FORMAT) == 0, "tensor dim zero fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-dim-overflow.gguf", YVEX_ERR_BOUNDS) == 0, "tensor dim overflow fails");
    YVEX_TEST_ASSERT(expect_gguf_open_status("tests/fixtures/gguf/tensor-offset-misaligned.gguf", YVEX_ERR_FORMAT) == 0, "tensor offset misaligned fails");
    return 0;
}

static int test_nonaddressable_tensor_range_is_exposed(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    const yvex_gguf_tensor_info *tensor;
    yvex_error err;
    int rc;

    YVEX_TEST_ASSERT(open_fixture("tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf",
                                  &artifact) == 0,
                     "open nonaddressable tensor fixture");
    rc = yvex_gguf_open(&gguf, artifact, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "structural reader exposes nonaddressable range fact");
    tensor = yvex_gguf_tensor_at(gguf, 0ull);
    YVEX_TEST_ASSERT(tensor != NULL, "nonaddressable tensor exists");
    YVEX_TEST_ASSERT(tensor->range_addressable == 0, "tensor range is explicitly nonaddressable");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

int yvex_test_gguf(void)
{
    if (test_valid_minimal() != 0) {
        return 1;
    }
    if (test_bad_magic() != 0) {
        return 1;
    }
    if (test_short_header() != 0) {
        return 1;
    }
    if (test_unsupported_version() != 0) {
        return 1;
    }
    if (test_valid_metadata_and_tensors() != 0) {
        return 1;
    }
    if (test_malformed_metadata() != 0) {
        return 1;
    }
    if (test_malformed_tensors() != 0) {
        return 1;
    }
    if (test_nonaddressable_tensor_range_is_exposed() != 0) {
        return 1;
    }
    return 0;
}
