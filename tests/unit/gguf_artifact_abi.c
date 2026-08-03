/* Exercises the operational file-backed GGUF reader and layout ABI directly. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "tests/test.h"
#include <yvex/internal/gguf.h>

#define READER_TEST_DIR "build/tests/gguf-reader"

static int write_bytes(FILE *fp, const unsigned char *bytes, size_t count)
{
    return fwrite(bytes, 1u, count, fp) == count;
}

static int write_u8(FILE *fp, unsigned int value)
{
    unsigned char byte = (unsigned char)value;
    return write_bytes(fp, &byte, 1u);
}

static int write_u16(FILE *fp, unsigned int value)
{
    unsigned char bytes[2];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    return write_bytes(fp, bytes, sizeof(bytes));
}

static int write_u32(FILE *fp, unsigned int value)
{
    unsigned char bytes[4];
    bytes[0] = (unsigned char)(value & 0xffu);
    bytes[1] = (unsigned char)((value >> 8) & 0xffu);
    bytes[2] = (unsigned char)((value >> 16) & 0xffu);
    bytes[3] = (unsigned char)((value >> 24) & 0xffu);
    return write_bytes(fp, bytes, sizeof(bytes));
}

static int write_u64(FILE *fp, unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int i;
    for (i = 0u; i < 8u; ++i) {
        bytes[i] = (unsigned char)((value >> (8u * i)) & 0xffull);
    }
    return write_bytes(fp, bytes, sizeof(bytes));
}

static int write_string_bytes(FILE *fp, const void *data, size_t len)
{
    return write_u64(fp, (unsigned long long)len) &&
           (len == 0u || fwrite(data, 1u, len, fp) == len);
}

static int write_string(FILE *fp, const char *text)
{
    return write_string_bytes(fp, text, strlen(text));
}

static int write_header(FILE *fp,
                        unsigned long long tensor_count,
                        unsigned long long metadata_count)
{
    return write_u32(fp, YVEX_GGUF_MAGIC) &&
           write_u32(fp, 3u) &&
           write_u64(fp, tensor_count) &&
           write_u64(fp, metadata_count);
}

static int write_padding(FILE *fp, unsigned int alignment)
{
    off_t pos = ftello(fp);
    unsigned int remainder;
    unsigned int count;
    unsigned char zero[64] = {0};
    if (pos < 0 || alignment == 0u || alignment > sizeof(zero)) return 0;
    remainder = (unsigned int)((unsigned long long)pos % alignment);
    count = remainder == 0u ? 0u : alignment - remainder;
    return count == 0u || write_bytes(fp, zero, count);
}

static int write_metadata_scalar(FILE *fp, unsigned int type)
{
    switch (type) {
    case YVEX_GGUF_VALUE_UINT8:
    case YVEX_GGUF_VALUE_INT8:
    case YVEX_GGUF_VALUE_BOOL:
        return write_u8(fp, type == YVEX_GGUF_VALUE_BOOL ? 1u : 7u);
    case YVEX_GGUF_VALUE_UINT16:
    case YVEX_GGUF_VALUE_INT16:
        return write_u16(fp, 7u);
    case YVEX_GGUF_VALUE_UINT32:
    case YVEX_GGUF_VALUE_INT32:
    case YVEX_GGUF_VALUE_FLOAT32:
        return write_u32(fp, 7u);
    case YVEX_GGUF_VALUE_STRING:
        return write_string(fp, "x");
    case YVEX_GGUF_VALUE_ARRAY:
        return write_u32(fp, YVEX_GGUF_VALUE_UINT8) &&
               write_u64(fp, 1ull) && write_u8(fp, 7u);
    case YVEX_GGUF_VALUE_UINT64:
    case YVEX_GGUF_VALUE_INT64:
    case YVEX_GGUF_VALUE_FLOAT64:
        return write_u64(fp, 7ull);
    default:
        return 0;
    }
}

static int prepare_test_dir(void)
{
    (void)mkdir("build", 0777);
    (void)mkdir("build/tests", 0777);
    return mkdir(READER_TEST_DIR, 0777) == 0 || access(READER_TEST_DIR, F_OK) == 0;
}

static int open_reader(const char *path,
                       const yvex_gguf_reader_options *reader_options,
                       yvex_artifact **artifact,
                       yvex_gguf **gguf,
                       yvex_gguf_parse_result *result)
{
    yvex_artifact_options artifact_options;
    yvex_error err;
    int rc;
    memset(&artifact_options, 0, sizeof(artifact_options));
    artifact_options.path = path;
    artifact_options.readonly = 1;
    *artifact = NULL;
    *gguf = NULL;
    yvex_error_clear(&err);
    rc = yvex_artifact_open(artifact, &artifact_options, &err);
    if (rc != YVEX_OK) return rc;
    return yvex_gguf_open_ex(gguf, *artifact, reader_options, result, &err);
}

static int expect_typed_refusal(const char *path,
                                yvex_gguf_parse_code code,
                                yvex_gguf_parse_section section);

static int expect_layout_refusal(const char *path, yvex_gguf_layout_code code)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result parse_result;
    yvex_gguf_layout_result layout;
    yvex_error err;
    int rc;
    memset(&layout, 0, sizeof(layout));
    rc = open_reader(path, NULL, &artifact, &gguf, &parse_result);
    if (rc == YVEX_OK) {
        yvex_error_clear(&err);
        rc = yvex_gguf_layout_validate(artifact, gguf, &layout, &err);
    }
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return rc != YVEX_OK && layout.code == code;
}

static int test_valid_fixture(void)
{
    const char *path = "tests/fixtures/gguf/valid-metadata-tensors.gguf";
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    yvex_gguf_layout_result layout;
    const yvex_gguf_header *header;
    const yvex_gguf_reader_stats *stats;
    const yvex_gguf_tensor_info *tensor;
    yvex_error err;
    int rc = open_reader(path, NULL, &artifact, &gguf, &result);

    YVEX_TEST_ASSERT(rc == YVEX_OK, "valid GGUF fixture opens");
    header = yvex_gguf_header_view(gguf);
    stats = yvex_gguf_reader_stats_view(gguf);
    tensor = yvex_gguf_tensor_at(gguf, 0ull);
    YVEX_TEST_ASSERT(header && header->version == 3u, "container version");
    YVEX_TEST_ASSERT(header->metadata_count == 5ull, "metadata count");
    YVEX_TEST_ASSERT(header->tensor_count == 1ull, "tensor count");
    YVEX_TEST_ASSERT(tensor && tensor->rank == 2u, "tensor rank");
    YVEX_TEST_ASSERT(stats && stats->payload_bytes_read == 0ull,
                     "reader reads no payload bytes");
    YVEX_TEST_ASSERT(stats->structural_bytes_read > 0ull,
                     "reader records structural bytes");
    yvex_error_clear(&err);
    rc = yvex_gguf_layout_validate(artifact, gguf, &layout, &err);
    YVEX_TEST_ASSERT(rc == YVEX_OK && layout.accepted,
                     "canonical layout accepts valid fixture");
    YVEX_TEST_ASSERT(layout.tensors_validated == 1ull,
                     "layout validates every tensor");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return 0;
}

static int test_invalid_magic(void)
{
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/bad-magic.gguf",
                                          YVEX_GGUF_PARSE_INVALID_MAGIC,
                                          YVEX_GGUF_PARSE_SECTION_CONTAINER),
                     "bad magic has typed refusal");
    return 0;
}

static int test_unsupported_version(void)
{
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/unsupported-version.gguf",
                                          YVEX_GGUF_PARSE_UNSUPPORTED_VERSION,
                                          YVEX_GGUF_PARSE_SECTION_CONTAINER),
                     "unsupported version has typed refusal");
    return 0;
}

static int test_malformed_metadata(void)
{
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/metadata-unknown-type.gguf",
                                          YVEX_GGUF_PARSE_UNSUPPORTED_METADATA_TYPE,
                                          YVEX_GGUF_PARSE_SECTION_METADATA),
                     "unknown metadata type has typed refusal");
    return 0;
}

static int test_tensor_info_refusal(void)
{
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/tensor-rank-unsupported.gguf",
                                          YVEX_GGUF_PARSE_INVALID_RANK,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "unsupported rank has typed refusal");
    return 0;
}

static int test_range_refusal(void)
{
    YVEX_TEST_ASSERT(expect_layout_refusal(
                         "tests/fixtures/gguf/tensor-offset-out-of-bounds.gguf",
                         YVEX_GGUF_LAYOUT_FIRST_OFFSET_NOT_ZERO),
                     "out-of-bounds tensor offset has typed layout refusal");
    return 0;
}

static int test_missing_file(void)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;

    YVEX_TEST_ASSERT(open_reader("tests/fixtures/gguf/missing.gguf", NULL,
                                 &artifact, &gguf, &result) != YVEX_OK,
                     "missing file refuses before parsing");
    YVEX_TEST_ASSERT(artifact == NULL && gguf == NULL,
                     "missing file leaves no reader ownership");
    return 0;
}

static int write_all_metadata_types_fixture(const char *path)
{
    static const char *type_names[] = {
        "uint8", "int8", "uint16", "int16", "uint32", "int32",
        "float32", "bool", "string", "array", "uint64", "int64", "float64"
    };
    FILE *fp = fopen(path, "wb");
    unsigned int type;
    char key[96];
    int ok = fp != NULL;
    if (!fp) return 0;
    ok = ok && write_header(fp, 0ull, 26ull);
    for (type = 0u; ok && type <= (unsigned int)YVEX_GGUF_VALUE_FLOAT64; ++type) {
        (void)snprintf(key, sizeof(key), "reader.scalar.%s", type_names[type]);
        ok = write_string(fp, key) && write_u32(fp, type) &&
             write_metadata_scalar(fp, type);
    }
    for (type = 0u; ok && type <= (unsigned int)YVEX_GGUF_VALUE_FLOAT64; ++type) {
        (void)snprintf(key, sizeof(key), "reader.array.%s", type_names[type]);
        ok = write_string(fp, key) && write_u32(fp, YVEX_GGUF_VALUE_ARRAY) &&
             write_u32(fp, type) && write_u64(fp, 1ull) &&
             write_metadata_scalar(fp, type);
    }
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_all_metadata_types_and_nested_arrays(void)
{
    const char *path = READER_TEST_DIR "/metadata-types.gguf";
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    unsigned int type;
    int rc;
    remove(path);
    YVEX_TEST_ASSERT(write_all_metadata_types_fixture(path), "write metadata type fixture");
    rc = open_reader(path, NULL, &artifact, &gguf, &result);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "all metadata types parse");
    YVEX_TEST_ASSERT(yvex_gguf_metadata_count(gguf) == 26ull, "metadata type count");
    for (type = 0u; type <= (unsigned int)YVEX_GGUF_VALUE_FLOAT64; ++type) {
        const yvex_gguf_value *scalar = yvex_gguf_metadata_value(gguf, type);
        const yvex_gguf_value *array = yvex_gguf_metadata_value(gguf, 13ull + type);
        yvex_gguf_array_info info;
        YVEX_TEST_ASSERT(yvex_gguf_value_type_of(scalar) == (yvex_gguf_value_type)type,
                         "scalar metadata type preserved");
        YVEX_TEST_ASSERT(yvex_gguf_value_array_info(array, &info) == YVEX_OK,
                         "array metadata accessor");
        YVEX_TEST_ASSERT(info.element_type == (yvex_gguf_value_type)type,
                         "array element type preserved");
        YVEX_TEST_ASSERT(yvex_gguf_value_array_at(array, 0ull) != NULL,
                         "array element retained");
    }
    YVEX_TEST_ASSERT(yvex_gguf_value_type_of(NULL) == YVEX_GGUF_VALUE_INVALID,
                     "null value does not synthesize array type");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    remove(path);
    return 0;
}

static int write_duplicate_metadata_fixture(const char *path)
{
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 0ull, 2ull) &&
         write_string(fp, "reader.duplicate") && write_u32(fp, YVEX_GGUF_VALUE_UINT32) && write_u32(fp, 1u) &&
         write_string(fp, "reader.duplicate") && write_u32(fp, YVEX_GGUF_VALUE_UINT32) && write_u32(fp, 2u);
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int write_tensor_record(FILE *fp,
                               const char *name,
                               unsigned int qtype,
                               unsigned long long relative_offset)
{
    return write_string(fp, name) && write_u32(fp, 1u) && write_u64(fp, 1ull) &&
           write_u32(fp, qtype) && write_u64(fp, relative_offset);
}

static int write_duplicate_tensor_fixture(const char *path)
{
    FILE *fp = fopen(path, "wb");
    unsigned char payload[4] = {0};
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 2ull, 0ull) &&
         write_tensor_record(fp, "reader.weight", YVEX_GGUF_QTYPE_F32, 0ull) &&
         write_tensor_record(fp, "reader.weight", YVEX_GGUF_QTYPE_F32, 0ull) &&
         write_padding(fp, 32u) && write_bytes(fp, payload, sizeof(payload));
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int expect_typed_refusal(const char *path,
                                yvex_gguf_parse_code code,
                                yvex_gguf_parse_section section)
{
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    int rc = open_reader(path, NULL, &artifact, &gguf, &result);
    int ok = rc != YVEX_OK && gguf == NULL && result.code == code &&
             result.section == section;
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    return ok;
}

static int test_duplicate_identifiers(void)
{
    const char *metadata_path = READER_TEST_DIR "/duplicate-metadata.gguf";
    const char *tensor_path = READER_TEST_DIR "/duplicate-tensor.gguf";
    remove(metadata_path);
    remove(tensor_path);
    YVEX_TEST_ASSERT(write_duplicate_metadata_fixture(metadata_path),
                     "write duplicate metadata fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(metadata_path,
                                          YVEX_GGUF_PARSE_DUPLICATE_METADATA_KEY,
                                          YVEX_GGUF_PARSE_SECTION_METADATA),
                     "duplicate metadata key is typed");
    YVEX_TEST_ASSERT(write_duplicate_tensor_fixture(tensor_path),
                     "write duplicate tensor fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(tensor_path,
                                          YVEX_GGUF_PARSE_DUPLICATE_TENSOR_NAME,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "duplicate tensor name is typed");
    remove(metadata_path);
    remove(tensor_path);
    return 0;
}

static int write_embedded_null_key_fixture(const char *path)
{
    const unsigned char key[3] = {'a', 0u, 'b'};
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 0ull, 1ull) && write_string_bytes(fp, key, sizeof(key)) &&
         write_u32(fp, YVEX_GGUF_VALUE_UINT32) && write_u32(fp, 1u);
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int write_embedded_null_tensor_fixture(const char *path)
{
    const unsigned char name[3] = {'a', 0u, 'b'};
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 1ull, 0ull) &&
         write_string_bytes(fp, name, sizeof(name)) && write_u32(fp, 1u) &&
         write_u64(fp, 1ull) && write_u32(fp, YVEX_GGUF_QTYPE_F32) &&
         write_u64(fp, 0ull);
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int write_empty_tensor_fixture(const char *path)
{
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 1ull, 0ull) && write_string(fp, "") &&
         write_u32(fp, 1u) && write_u64(fp, 1ull) &&
         write_u32(fp, YVEX_GGUF_QTYPE_F32) && write_u64(fp, 0ull);
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_identifier_safety(void)
{
    const char *key_path = READER_TEST_DIR "/embedded-null-key.gguf";
    const char *tensor_path = READER_TEST_DIR "/embedded-null-tensor.gguf";
    const char *empty_path = READER_TEST_DIR "/empty-tensor.gguf";
    remove(key_path);
    remove(tensor_path);
    remove(empty_path);
    YVEX_TEST_ASSERT(write_embedded_null_key_fixture(key_path),
                     "write embedded-null key fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(key_path, YVEX_GGUF_PARSE_MALFORMED_KEY,
                                          YVEX_GGUF_PARSE_SECTION_METADATA),
                     "embedded-null key is refused");
    YVEX_TEST_ASSERT(write_embedded_null_tensor_fixture(tensor_path),
                     "write embedded-null tensor fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(tensor_path,
                                          YVEX_GGUF_PARSE_MALFORMED_TENSOR_NAME,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "embedded-null tensor name is refused");
    remove(key_path);
    remove(tensor_path);
    YVEX_TEST_ASSERT(write_empty_tensor_fixture(empty_path),
                     "write empty tensor-name fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(empty_path,
                                          YVEX_GGUF_PARSE_EMPTY_TENSOR_NAME,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "empty tensor name has a distinct typed refusal");
    remove(empty_path);
    return 0;
}

static int write_invalid_utf8_fixture(const char *path)
{
    const unsigned char invalid_utf8[2] = {0xc0u, 0xafu};
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 0ull, 1ull) && write_string(fp, "reader.text") &&
         write_u32(fp, YVEX_GGUF_VALUE_STRING) &&
         write_string_bytes(fp, invalid_utf8, sizeof(invalid_utf8));
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_malformed_utf8_value(void)
{
    const char *path = READER_TEST_DIR "/invalid-utf8.gguf";
    remove(path);
    YVEX_TEST_ASSERT(write_invalid_utf8_fixture(path), "write invalid UTF-8 fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(path, YVEX_GGUF_PARSE_MALFORMED_STRING,
                                          YVEX_GGUF_PARSE_SECTION_METADATA),
                     "invalid UTF-8 metadata string is typed");
    remove(path);
    return 0;
}

static int write_sparse_large_fixture(const char *path,
                                      unsigned long long sparse_size)
{
    FILE *fp = fopen(path, "wb+");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 1ull, 1ull) &&
         write_string(fp, "general.alignment") &&
         write_u32(fp, YVEX_GGUF_VALUE_UINT32) && write_u32(fp, 32u) &&
         write_string(fp, "reader.weight") && write_u32(fp, 2u) &&
         write_u64(fp, 32ull) && write_u64(fp, 2ull) &&
         write_u32(fp, YVEX_GGUF_QTYPE_Q4_0) && write_u64(fp, 0ull) &&
         write_padding(fp, 32u) && fflush(fp) == 0 &&
         ftruncate(fileno(fp), (off_t)sparse_size) == 0;
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_sparse_large_file_reads_only_structure(void)
{
    const char *path = READER_TEST_DIR "/sparse-large.gguf";
    const unsigned long long sparse_size = 160ull * 1024ull * 1024ull * 1024ull;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    const yvex_gguf_reader_stats *stats;
    const yvex_gguf_tensor_info *tensor;
    int rc;
    remove(path);
    YVEX_TEST_ASSERT(write_sparse_large_fixture(path, sparse_size), "write sparse large GGUF");
    rc = open_reader(path, NULL, &artifact, &gguf, &result);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "sparse large GGUF parses");
    YVEX_TEST_ASSERT(yvex_artifact_size(artifact) == sparse_size, "64-bit sparse file size");
    YVEX_TEST_ASSERT(yvex_artifact_data(artifact) == NULL, "structural artifact is not mapped");
    stats = yvex_gguf_reader_stats_view(gguf);
    YVEX_TEST_ASSERT(stats != NULL, "reader stats exist");
    YVEX_TEST_ASSERT(stats->structural_bytes_read == stats->directory_end_offset,
                     "one canonical structural pass");
    YVEX_TEST_ASSERT(stats->structural_bytes_read < 4096ull,
                     "structural bytes independent of sparse payload");
    YVEX_TEST_ASSERT(stats->payload_bytes_read == 0ull, "sparse payload bytes are zero");
    tensor = yvex_gguf_tensor_at(gguf, 0ull);
    YVEX_TEST_ASSERT(tensor && tensor->storage_bytes == 36ull,
                     "sparse tensor uses canonical qtype geometry");
    YVEX_TEST_ASSERT(tensor->range_addressable == 1, "sparse tensor range addressable");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    remove(path);
    return 0;
}

static int write_scale_fixture(const char *path,
                               unsigned long long tensor_count,
                               unsigned long long token_count)
{
    FILE *fp = fopen(path, "wb");
    unsigned long long i;
    unsigned char payload[4] = {0};
    char name[64];
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, tensor_count, 1ull) &&
         write_string(fp, "tokenizer.ggml.tokens") &&
         write_u32(fp, YVEX_GGUF_VALUE_ARRAY) &&
         write_u32(fp, YVEX_GGUF_VALUE_STRING) && write_u64(fp, token_count);
    for (i = 0ull; ok && i < token_count; ++i) ok = write_string(fp, "");
    for (i = 0ull; ok && i < tensor_count; ++i) {
        (void)snprintf(name, sizeof(name), "reader.tensor_%05llu", i);
        ok = write_tensor_record(fp, name, YVEX_GGUF_QTYPE_F32, 0ull);
    }
    ok = ok && write_padding(fp, 32u) && write_bytes(fp, payload, sizeof(payload));
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_target_scale_directory(void)
{
    const char *path = READER_TEST_DIR "/target-scale.gguf";
    const unsigned long long tensor_count = 72317ull;
    const unsigned long long token_count = 129280ull;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    const yvex_gguf_reader_stats *stats;
    const yvex_gguf_value *tokens;
    yvex_gguf_array_info info;
    int rc;
    remove(path);
    YVEX_TEST_ASSERT(write_scale_fixture(path, tensor_count, token_count),
                     "write target-scale directory");
    rc = open_reader(path, NULL, &artifact, &gguf, &result);
    YVEX_TEST_ASSERT(rc == YVEX_OK, "target-scale directory parses");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_count(gguf) == tensor_count,
                     "target-scale tensor count");
    tokens = yvex_gguf_metadata_find(gguf, "tokenizer.ggml.tokens");
    YVEX_TEST_ASSERT(yvex_gguf_value_array_info(tokens, &info) == YVEX_OK,
                     "target-scale tokenizer array exists");
    YVEX_TEST_ASSERT(info.count == token_count, "target-scale tokenizer count");
    YVEX_TEST_ASSERT(yvex_gguf_tensor_find(gguf, "reader.tensor_69186") != NULL,
                     "target-scale hash lookup");
    stats = yvex_gguf_reader_stats_view(gguf);
    YVEX_TEST_ASSERT(stats && stats->payload_bytes_read == 0ull,
                     "target-scale reader reads zero payload");
    YVEX_TEST_ASSERT(stats->structural_bytes_read == stats->directory_end_offset,
                     "target-scale directory parsed once");
    yvex_gguf_close(gguf);
    yvex_artifact_close(artifact);
    remove(path);
    return 0;
}

static int test_resource_budget_refusal(void)
{
    yvex_gguf_reader_options options;
    yvex_artifact *artifact = NULL;
    yvex_gguf *gguf = NULL;
    yvex_gguf_parse_result result;
    int rc;
    yvex_gguf_reader_options_default(&options);
    options.max_metadata_entries = 1ull;
    rc = open_reader("tests/fixtures/gguf/valid-metadata-tensors.gguf",
                     &options, &artifact, &gguf, &result);
    YVEX_TEST_ASSERT(rc != YVEX_OK, "metadata budget refuses");
    YVEX_TEST_ASSERT(gguf == NULL, "budget refusal leaves view null");
    YVEX_TEST_ASSERT(result.code == YVEX_GGUF_PARSE_RESOURCE_LIMIT,
                     "budget refusal is typed");
    YVEX_TEST_ASSERT(result.section == YVEX_GGUF_PARSE_SECTION_RESOURCE,
                     "budget refusal section");
    yvex_artifact_close(artifact);
    return 0;
}

static int write_truncated_tensor_fixture(const char *path)
{
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 1ull, 0ull) && write_string(fp, "reader.weight") &&
         write_u32(fp, 2u) && write_u64(fp, 32ull);
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_typed_truncation_boundaries(void)
{
    const char *path = READER_TEST_DIR "/truncated-tensor.gguf";
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/short-header.gguf",
                                          YVEX_GGUF_PARSE_SHORT_READ,
                                          YVEX_GGUF_PARSE_SECTION_CONTAINER),
                     "container truncation typed");
    YVEX_TEST_ASSERT(expect_typed_refusal(
                         "tests/fixtures/gguf/metadata-string-oob.gguf",
                         YVEX_GGUF_PARSE_MALFORMED_STRING,
                         YVEX_GGUF_PARSE_SECTION_METADATA),
                     "metadata truncation typed");
    remove(path);
    YVEX_TEST_ASSERT(write_truncated_tensor_fixture(path), "write tensor truncation fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(path, YVEX_GGUF_PARSE_SHORT_READ,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "tensor directory truncation typed");
    remove(path);
    return 0;
}

static int write_alignment_fixture(const char *path,
                                   unsigned int value_type,
                                   unsigned int alignment)
{
    FILE *fp = fopen(path, "wb");
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 0ull, 1ull) &&
         write_string(fp, "general.alignment") &&
         write_u32(fp, value_type);
    if (ok) {
        ok = value_type == YVEX_GGUF_VALUE_UINT64
                 ? write_u64(fp, (unsigned long long)alignment)
                 : write_u32(fp, alignment);
    }
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int write_qtype_or_offset_fixture(const char *path,
                                         unsigned int qtype,
                                         unsigned long long relative_offset)
{
    FILE *fp = fopen(path, "wb");
    unsigned char payload[4] = {0};
    int ok;
    if (!fp) return 0;
    ok = write_header(fp, 1ull, 0ull) &&
         write_tensor_record(fp, "reader.weight", qtype, relative_offset) &&
         write_padding(fp, 32u) && write_bytes(fp, payload, sizeof(payload));
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static int test_typed_tensor_and_range_refusals(void)
{
    const char *alignment_path = READER_TEST_DIR "/invalid-alignment.gguf";
    const char *alignment_type_path = READER_TEST_DIR "/invalid-alignment-type.gguf";
    const char *alignment_24_path = READER_TEST_DIR "/alignment-24.gguf";
    const char *qtype_path = READER_TEST_DIR "/refused-qtype.gguf";
    const char *offset_path = READER_TEST_DIR "/offset-overflow.gguf";
    unsigned long long aligned_max = ULLONG_MAX & ~31ull;
    remove(alignment_path);
    remove(alignment_type_path);
    remove(alignment_24_path);
    remove(qtype_path);
    remove(offset_path);
    YVEX_TEST_ASSERT(write_alignment_fixture(alignment_path,
                                              YVEX_GGUF_VALUE_UINT32, 12u),
                     "write invalid alignment fixture");
    YVEX_TEST_ASSERT(expect_layout_refusal(
                         alignment_path, YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO),
                     "non-power-of-two alignment is layout-typed");
    YVEX_TEST_ASSERT(write_alignment_fixture(alignment_type_path,
                                              YVEX_GGUF_VALUE_UINT64, 32u),
                     "write invalid alignment type fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(alignment_type_path,
                                          YVEX_GGUF_PARSE_INVALID_ALIGNMENT,
                                          YVEX_GGUF_PARSE_SECTION_METADATA),
                     "invalid alignment type is typed");
    YVEX_TEST_ASSERT(write_alignment_fixture(alignment_24_path,
                                              YVEX_GGUF_VALUE_UINT32, 24u),
                     "write non-power-of-two alignment fixture");
    {
        yvex_artifact *artifact = NULL;
        yvex_gguf *gguf = NULL;
        yvex_gguf_parse_result result;
        int rc = open_reader(alignment_24_path, NULL, &artifact, &gguf, &result);
        yvex_gguf_layout_result layout;
        yvex_error err;
        YVEX_TEST_ASSERT(rc == YVEX_OK, "structural reader preserves alignment value");
        YVEX_TEST_ASSERT(yvex_gguf_alignment(gguf) == 24u,
                         "non-power-of-two alignment preserved");
        yvex_error_clear(&err);
        rc = yvex_gguf_layout_validate(artifact, gguf, &layout, &err);
        YVEX_TEST_ASSERT(rc != YVEX_OK &&
                         layout.code == YVEX_GGUF_LAYOUT_ALIGNMENT_NOT_POWER_OF_TWO,
                         "alignment 24 is refused by global admission");
        yvex_gguf_close(gguf);
        yvex_artifact_close(artifact);
    }
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/tensor-rank-zero.gguf",
                                          YVEX_GGUF_PARSE_INVALID_RANK,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "invalid rank is typed");
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/tensor-dim-zero.gguf",
                                          YVEX_GGUF_PARSE_INVALID_DIMENSION,
                                          YVEX_GGUF_PARSE_SECTION_TENSOR_INFO),
                     "invalid dimension is typed");
    YVEX_TEST_ASSERT(expect_typed_refusal("tests/fixtures/gguf/tensor-dim-overflow.gguf",
                                          YVEX_GGUF_PARSE_ROW_BYTE_OVERFLOW,
                                          YVEX_GGUF_PARSE_SECTION_QTYPE),
                     "qtype row-byte overflow is distinct from offset overflow");
    YVEX_TEST_ASSERT(write_qtype_or_offset_fixture(qtype_path,
                                                   YVEX_GGUF_QTYPE_Q4_2_REMOVED,
                                                   0ull),
                     "write refused qtype fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(qtype_path, YVEX_GGUF_PARSE_REFUSED_QTYPE,
                                          YVEX_GGUF_PARSE_SECTION_QTYPE),
                     "refused qtype is typed");
    YVEX_TEST_ASSERT(write_qtype_or_offset_fixture(offset_path,
                                                   YVEX_GGUF_QTYPE_F32,
                                                   aligned_max),
                     "write offset overflow fixture");
    YVEX_TEST_ASSERT(expect_typed_refusal(offset_path,
                                          YVEX_GGUF_PARSE_OFFSET_OVERFLOW,
                                          YVEX_GGUF_PARSE_SECTION_RANGE),
                     "absolute offset overflow is typed");
    remove(alignment_path);
    remove(alignment_type_path);
    remove(alignment_24_path);
    remove(qtype_path);
    remove(offset_path);
    return 0;
}

static int test_view_lifetime_and_repeated_cleanup(void)
{
    unsigned int i;
    for (i = 0u; i < 64u; ++i) {
        yvex_artifact *artifact = NULL;
        yvex_gguf *gguf = NULL;
        yvex_gguf_parse_result result;
        const yvex_gguf_value *value;
        int rc = open_reader("tests/fixtures/gguf/valid-metadata-tensors.gguf",
                             NULL, &artifact, &gguf, &result);
        YVEX_TEST_ASSERT(rc == YVEX_OK, "repeated reader open");
        yvex_artifact_close(artifact);
        value = yvex_gguf_metadata_find(gguf, "general.architecture");
        YVEX_TEST_ASSERT(value != NULL, "view owns accessor storage after file close");
        yvex_gguf_close(gguf);
    }
    yvex_gguf_close(NULL);
    return 0;
}

int yvex_test_gguf_artifact_abi(void)
{
    YVEX_TEST_ASSERT(prepare_test_dir(), "prepare GGUF reader test directory");
    if (test_valid_fixture() != 0) return 1;
    if (test_invalid_magic() != 0) return 1;
    if (test_unsupported_version() != 0) return 1;
    if (test_malformed_metadata() != 0) return 1;
    if (test_tensor_info_refusal() != 0) return 1;
    if (test_range_refusal() != 0) return 1;
    if (test_missing_file() != 0) return 1;
    if (test_all_metadata_types_and_nested_arrays() != 0) return 1;
    if (test_duplicate_identifiers() != 0) return 1;
    if (test_identifier_safety() != 0) return 1;
    if (test_malformed_utf8_value() != 0) return 1;
    if (test_sparse_large_file_reads_only_structure() != 0) return 1;
    if (test_target_scale_directory() != 0) return 1;
    if (test_resource_budget_refusal() != 0) return 1;
    if (test_typed_truncation_boundaries() != 0) return 1;
    if (test_typed_tensor_and_range_refusals() != 0) return 1;
    if (test_view_lifetime_and_repeated_cleanup() != 0) return 1;
    return 0;
}
