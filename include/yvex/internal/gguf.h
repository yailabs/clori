/*
 * GGUF parsing, metadata projection, and structural equivalence share container-owned types here.
 * Parsed structure remains distinct from complete-artifact admission.
 */
#ifndef INCLUDE_YVEX_INTERNAL_GGUF_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_GGUF_H_INCLUDED

#include <stddef.h>
#include <stdint.h>
#include <yvex/artifact.h>
#include <yvex/core.h>
#include <yvex/gguf.h>
#include <yvex/model.h>
#include <yvex/qtype.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf_writer.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/internal/source.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bounded GGUF-owned JSON document cursor shared by quantization metadata owners. */
typedef struct {
    yvex_json cursor;
    char *buffer;
    const char *path;
    const char *context;
    yvex_error *err;
} yvex_gguf_json;
int yvex_gguf_json_open(yvex_gguf_json *json,
                        const char *path,
                        const char *context,
                        yvex_error *err);
void yvex_gguf_json_close(yvex_gguf_json *json);
int yvex_gguf_json_fail(yvex_gguf_json *json, const char *message);
int yvex_gguf_json_expect(yvex_gguf_json *json, char expected);
char *yvex_gguf_json_string(yvex_gguf_json *json);
int yvex_gguf_json_skip(yvex_gguf_json *json);
int yvex_gguf_json_member(yvex_gguf_json *json, char **key, int *complete);
void yvex_gguf_json_optional_comma(yvex_gguf_json *json);
typedef int (*yvex_gguf_json_array_item_fn)(yvex_gguf_json *json, void *context);
int yvex_gguf_json_array(yvex_gguf_json *json,
                         yvex_gguf_json_array_item_fn item,
                         void *context,
                         const char *malformed,
                         const char *unterminated);

/* Decode one admitted canonical little-endian unsigned 16-bit field. */
static inline unsigned short gguf_u16le_load(const unsigned char *bytes)
{
    return (unsigned short)((unsigned short)bytes[0] | ((unsigned short)bytes[1] << 8));
}

/* Decode one admitted canonical little-endian unsigned 32-bit field. */
static inline unsigned int gguf_u32le_load(const unsigned char *bytes)
{
    return (unsigned int)bytes[0] | ((unsigned int)bytes[1] << 8) |
           ((unsigned int)bytes[2] << 16) | ((unsigned int)bytes[3] << 24);
}

/* Interpret one portable two's-complement I32 bit pattern without narrowing. */
static inline int32_t gguf_i32_from_u32(uint32_t value)
{
    return value <= (uint32_t)INT32_MAX ? (int32_t)value
                                        : -1 - (int32_t)(UINT32_MAX - value);
}

/* Private parser failure. */
void yvex_gguf_parse_result_reset(yvex_gguf_parse_result *result);
int yvex_gguf_reader_fail(yvex_gguf_parse_result *result,
                          yvex_gguf_parse_code code,
                          yvex_gguf_parse_section section,
                          unsigned long long byte_offset,
                          unsigned long long record_index,
                          yvex_error *err,
                          const char *where,
                          const char *reason);
/* Map. */
#define YVEX_GGUF_MAPPING_REFERENCE_COMMIT \
    "e920c523e3b8a0163fe498af5bf90df35ff51d25"
#define YVEX_GGUF_DSPARK_EXTENSION_VERSION 1u
#define YVEX_GGUF_NO_FORCED_QTYPE (~0u)
int yvex_gguf_name_map_resolve(yvex_tensor_role role,
                               int draft_extension,
                               unsigned long long layer_index,
                               unsigned long long predictor_index,
                               char *out,
                               size_t out_cap,
                               yvex_gguf_name_provenance *provenance,
                               const char **reason);
int yvex_gguf_layout_map_shape_supported(yvex_tensor_role role,
                                         unsigned int qtype,
                                         unsigned int rank,
                                         const unsigned long long *dims,
                                         const char **reason);

/* Tokenizer Metadata. */
#define YVEX_GGUF_TOKENIZER_SHA256_CAP 65u
typedef enum {
    YVEX_GGUF_TOKENIZER_OK = 0,
    YVEX_GGUF_TOKENIZER_INVALID_ARGUMENT,
    YVEX_GGUF_TOKENIZER_SOURCE_IDENTITY,
    YVEX_GGUF_TOKENIZER_RESOURCE_LIMIT,
    YVEX_GGUF_TOKENIZER_ALLOCATION,
    YVEX_GGUF_TOKENIZER_MALFORMED_JSON,
    YVEX_GGUF_TOKENIZER_UNSUPPORTED_JSON,
    YVEX_GGUF_TOKENIZER_DUPLICATE_TOKEN_ID,
    YVEX_GGUF_TOKENIZER_MISSING_TOKEN_ID,
    YVEX_GGUF_TOKENIZER_TOKEN_MISMATCH,
    YVEX_GGUF_TOKENIZER_CARDINALITY,
    YVEX_GGUF_TOKENIZER_SPECIAL_TOKEN
} yvex_gguf_tokenizer_code;
typedef struct {
    yvex_gguf_tokenizer_code code;
    unsigned long long record_index;
    unsigned long long expected;
    unsigned long long actual;
    char field[64];
} yvex_gguf_tokenizer_failure;
typedef struct {
    unsigned long long tokenizer_json_bytes;
    unsigned long long tokenizer_config_bytes;
    unsigned long long token_count;
    unsigned long long merge_count;
    unsigned long long added_token_count;
    unsigned long long owned_bytes;
    unsigned long long decoded_string_bytes;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    unsigned int pad_token_id;
    int add_bos_token;
    int add_eos_token;
    int chat_template_present;
    char pre_tokenizer[32];
    char tokenizer_json_sha256[YVEX_GGUF_TOKENIZER_SHA256_CAP];
    char tokenizer_config_sha256[YVEX_GGUF_TOKENIZER_SHA256_CAP];
    char tokenizer_json_git_oid[41];
    char tokenizer_config_git_oid[41];
    int complete;
} yvex_gguf_tokenizer_summary;
typedef struct yvex_gguf_tokenizer_metadata yvex_gguf_tokenizer_metadata;
int yvex_gguf_tokenizer_metadata_load(
    yvex_gguf_tokenizer_metadata **out,
    const yvex_source_verification *verification,
    unsigned long long expected_vocab_size,
    const char *pre_tokenizer,
    size_t maximum_owned_bytes,
    yvex_gguf_tokenizer_failure *failure,
    yvex_error *err);
void yvex_gguf_tokenizer_metadata_release(
    yvex_gguf_tokenizer_metadata **metadata);
const yvex_gguf_tokenizer_summary *yvex_gguf_tokenizer_summary_get(
    const yvex_gguf_tokenizer_metadata *metadata);
int yvex_gguf_tokenizer_token_at(
    const yvex_gguf_tokenizer_metadata *metadata,
    unsigned long long index,
    const unsigned char **bytes,
    size_t *byte_count,
    int *token_type);
int yvex_gguf_tokenizer_merge_at(
    const yvex_gguf_tokenizer_metadata *metadata,
    unsigned long long index,
    const unsigned char **bytes,
    size_t *byte_count);
int yvex_gguf_tokenizer_raw_json(
    const yvex_gguf_tokenizer_metadata *metadata,
    const unsigned char **bytes,
    size_t *byte_count);
int yvex_gguf_tokenizer_raw_config(
    const yvex_gguf_tokenizer_metadata *metadata,
    const unsigned char **bytes,
    size_t *byte_count);

/* Roundtrip. */
typedef enum {
    YVEX_GGUF_ROUNDTRIP_OK = 0,
    YVEX_GGUF_ROUNDTRIP_INVALID_ARGUMENT,
    YVEX_GGUF_ROUNDTRIP_ARTIFACT_OPEN,
    YVEX_GGUF_ROUNDTRIP_READER_REFUSAL,
    YVEX_GGUF_ROUNDTRIP_LAYOUT_REFUSAL,
    YVEX_GGUF_ROUNDTRIP_HEADER_DIVERGENCE,
    YVEX_GGUF_ROUNDTRIP_METADATA_DIVERGENCE,
    YVEX_GGUF_ROUNDTRIP_TENSOR_DIVERGENCE,
    YVEX_GGUF_ROUNDTRIP_PREFIX_DIVERGENCE,
    YVEX_GGUF_ROUNDTRIP_PAYLOAD_DIGEST,
    YVEX_GGUF_ROUNDTRIP_ARTIFACT_DIGEST,
    YVEX_GGUF_ROUNDTRIP_SHORT_READ,
    YVEX_GGUF_ROUNDTRIP_NONZERO_PADDING,
    YVEX_GGUF_ROUNDTRIP_TOKENIZER_INCOMPLETE,
    YVEX_GGUF_ROUNDTRIP_FILE_DRIFT,
    YVEX_GGUF_ROUNDTRIP_ALLOCATION
} yvex_gguf_roundtrip_code;
typedef struct {
    yvex_gguf_roundtrip_code code;
    unsigned long long metadata_index;
    unsigned long long tensor_index;
    unsigned long long expected;
    unsigned long long actual;
    unsigned long long file_offset;
    char name[YVEX_GGUF_WRITER_NAME_CAP];
} yvex_gguf_roundtrip_failure;
typedef struct yvex_gguf_roundtrip_summary {
    unsigned long long file_bytes;
    unsigned long long bytes_hashed;
    unsigned long long prefix_bytes_verified;
    unsigned long long payload_bytes_verified;
    unsigned long long padding_bytes_verified;
    unsigned long long metadata_count;
    unsigned long long tensor_count;
    unsigned long long terminals_verified;
    unsigned long long read_calls;
    unsigned long long peak_owned_bytes;
    unsigned long long file_device;
    unsigned long long file_inode;
    long long file_mtime_seconds;
    long long file_mtime_nanoseconds;
    long long file_ctime_seconds;
    long long file_ctime_nanoseconds;
    char artifact_identity[YVEX_SHA256_HEX_CAP];
    char payload_byte_identity[YVEX_SHA256_HEX_CAP];
    int tokenizer_complete;
    int reader_accepted;
    int layout_accepted;
    int payload_accepted;
    int snapshot_stable;
    int complete;
} yvex_gguf_roundtrip_summary;
typedef void (*yvex_gguf_roundtrip_progress_fn)(
    void *context,
    const yvex_gguf_roundtrip_summary *summary,
    unsigned long long planned_file_bytes);
typedef struct {
    size_t verification_chunk_bytes;
    yvex_gguf_roundtrip_progress_fn progress;
    void *progress_context;
} yvex_gguf_roundtrip_options;
void yvex_gguf_roundtrip_options_default(
    yvex_gguf_roundtrip_options *options);
int yvex_gguf_roundtrip_validate(
    const char *path,
    const yvex_gguf_writer_plan *writer_plan,
    yvex_quant_digest_sink *digest_sink,
    const yvex_gguf_roundtrip_options *options,
    yvex_gguf_roundtrip_summary *out,
    yvex_gguf_roundtrip_failure *failure,
    yvex_error *err);
#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_GGUF_H_INCLUDED */
