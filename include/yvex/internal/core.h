/*
 * Cross-subsystem leaf mechanisms live here: checked arithmetic, hashing, bounded JSON, paths,
 * and deterministic indexes. Consumers assign domain meaning to their results.
 */
#ifndef INCLUDE_YVEX_INTERNAL_CORE_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_CORE_H_INCLUDED

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <yvex/core.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bounded JSON parsing. */
#define YVEX_JSON_KEY_CAP 1024u
typedef struct {
    const char *cursor;
    const char *end;
    unsigned int depth;
} yvex_json;
typedef enum {
    YVEX_JSON_ITEM_ERROR = -1,
    YVEX_JSON_ITEM_END = 0,
    YVEX_JSON_ITEM_READY = 1
} yvex_json_item;
typedef enum {
    YVEX_JSON_COLLECTION_OBJECT = 1,
    YVEX_JSON_COLLECTION_ARRAY = 2
} yvex_json_collection;
typedef struct {
    yvex_json *json;
    size_t count;
    char closing;
    int complete;
    int trailing_separator;
} yvex_json_iter;
void yvex_json_init(yvex_json *json, const char *data, size_t length);
char *yvex_read_bounded_file(const char *path,
                             size_t cap,
                             size_t *length,
                             yvex_error *err);
int yvex_core_file_read_text_prefix(const char *path,
                                    char *buffer,
                                    size_t capacity);
void yvex_json_space(yvex_json *json);
int yvex_json_iter_begin(yvex_json *json,
                         yvex_json_iter *iter,
                         yvex_json_collection collection);
yvex_json_item yvex_json_object_member(yvex_json_iter *iter,
                                       char *key,
                                       size_t capacity);
yvex_json_item yvex_json_array_value(yvex_json_iter *iter);
int yvex_json_string(yvex_json *json, char *out, size_t cap);
char *yvex_json_string_dup(yvex_json *json, size_t cap);
const char *yvex_json_probe_field_value(const char *text, const char *key);
int yvex_json_probe_string_field(const char *text,
                                 const char *key,
                                 char *out,
                                 size_t capacity);
const void *yvex_core_keyed_row_find(const void *rows,
                                     size_t count,
                                     size_t stride,
                                     size_t key_offset,
                                     const char *key);
int yvex_json_skip_value(yvex_json *json);
int yvex_json_complete(yvex_json *json);
int yvex_json_u64(yvex_json *json, unsigned long long *out);
int yvex_json_bool(yvex_json *json, int *out);
int yvex_json_number_text(yvex_json *json, char *out, size_t cap);
int yvex_json_u64_array(yvex_json *json,
                        unsigned long long *values,
                        size_t cap,
                        unsigned long long *count);

/* SHA-256 and identity encodings. */
#define YVEX_SHA256_DIGEST_BYTES 32u
#define YVEX_SHA256_HEX_BYTES 65u
typedef struct {
    uint32_t state[8];
    uint64_t length;
    unsigned char block[64];
    size_t used;
    int finalized;
} yvex_sha256;
void yvex_sha256_init(yvex_sha256 *context);
int yvex_sha256_update(yvex_sha256 *context, const void *data, size_t length);
int yvex_sha256_final(yvex_sha256 *context,
                      unsigned char digest[YVEX_SHA256_DIGEST_BYTES]);
void yvex_sha256_hex(const unsigned char digest[YVEX_SHA256_DIGEST_BYTES],
                     char output[YVEX_SHA256_HEX_BYTES]);
int yvex_sha256_hex_valid(const char *text);
int yvex_sha256_update_u64(yvex_sha256 *context, unsigned long long value);
int yvex_sha256_update_text(yvex_sha256 *context, const char *text);

/* Append one integer through the retained big-endian identity encoding. */
static inline int yvex_sha256_update_u64_be(yvex_sha256 *context,
                                             unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int index;

    for (index = 0u; index < sizeof(bytes); ++index)
        bytes[sizeof(bytes) - 1u - index] =
            (unsigned char)((value >> (index * 8u)) & 0xffu);
    return yvex_sha256_update(context, bytes, sizeof(bytes));
}

/* Process-local deterministic helpers shared by diagnostic owners. */
char *yvex_core_strdup(const char *text);

typedef struct {
    unsigned long long allocation_events, reallocation_events, release_events;
    int overflowed;
} yvex_core_allocation_epoch;
typedef enum {
    YVEX_CORE_ALLOCATE_MALLOC = 0,
    YVEX_CORE_ALLOCATE_CALLOC,
    YVEX_CORE_ALLOCATE_REALLOC,
    YVEX_CORE_ALLOCATE_FREE
} yvex_core_allocation_operation;
void yvex_core_allocation_epoch_snapshot(yvex_core_allocation_epoch *out);
void *yvex_core_allocate(yvex_core_allocation_operation operation,
                         void *pointer, size_t count, size_t size);

/* Route malloc through the single observed core allocation ABI. */
static inline void *yvex_core_malloc(size_t size)
{
    return yvex_core_allocate(YVEX_CORE_ALLOCATE_MALLOC, NULL, 1u, size);
}
/* Route calloc through the single observed core allocation ABI. */
static inline void *yvex_core_calloc(size_t count, size_t size)
{
    return yvex_core_allocate(YVEX_CORE_ALLOCATE_CALLOC, NULL, count, size);
}
/* Route realloc through the single observed core allocation ABI. */
static inline void *yvex_core_realloc(void *pointer, size_t size)
{
    return yvex_core_allocate(YVEX_CORE_ALLOCATE_REALLOC, pointer, 1u, size);
}
/* Route free through the single observed core allocation ABI. */
static inline void yvex_core_free(void *pointer)
{
    (void)yvex_core_allocate(YVEX_CORE_ALLOCATE_FREE, pointer, 0u, 0u);
}

typedef enum {
    YVEX_CORE_OBSERVE_SOURCE_HEADER = 0,
    YVEX_CORE_OBSERVE_SOURCE_PAYLOAD_BYTES,
    YVEX_CORE_OBSERVE_TRANSFORM_PLAN,
    YVEX_CORE_OBSERVE_QUANT_PLAN,
    YVEX_CORE_OBSERVE_WRITER_PLAN,
    YVEX_CORE_OBSERVE_COUNT
} yvex_core_execution_observation_kind;
typedef struct {
    unsigned long long source_headers_read, source_payload_bytes_read;
    unsigned long long transform_plans_built, quant_plans_built;
    unsigned long long writer_plans_built;
    int overflowed;
} yvex_core_execution_observation;
void yvex_core_execution_observation_record(
    yvex_core_execution_observation_kind kind, unsigned long long amount);
void yvex_core_execution_observation_snapshot(
    yvex_core_execution_observation *out);
int yvex_core_execution_observation_delta(
    const yvex_core_execution_observation *before,
    const yvex_core_execution_observation *after,
    yvex_core_execution_observation *out);

#ifndef YVEX_CORE_ALLOCATION_IMPLEMENTATION
#define malloc(size) yvex_core_malloc(size)
#define calloc(count, size) yvex_core_calloc((count), (size))
#define realloc(pointer, size) yvex_core_realloc((pointer), (size))
#define free(pointer) yvex_core_free(pointer)
#endif

/* Bounded byte storage shared by canonical serialization owners. */
typedef struct {
    unsigned char *data;
    size_t count;
    size_t capacity;
    size_t maximum;
    size_t initial_capacity;
} yvex_core_bytes;

/*
 * Reserve one bounded byte arena without exposing partial growth.
 *
 * Overflow, budget, or allocation failure preserves count and content.
 */
static inline int yvex_core_bytes_reserve(yvex_core_bytes *bytes,
                                          size_t additional)
{
    size_t required;
    size_t capacity;
    unsigned char *grown;

    if (!bytes || additional > SIZE_MAX - bytes->count) return 0;
    required = bytes->count + additional;
    if (required > bytes->maximum) return 0;
    if (required <= bytes->capacity) return 1;
    capacity = bytes->capacity ? bytes->capacity : bytes->initial_capacity;
    if (!capacity) capacity = 1u;
    while (capacity < required) {
        if (capacity > bytes->maximum / 2u) {
            capacity = bytes->maximum;
            break;
        }
        capacity *= 2u;
    }
    if (capacity < required) return 0;
    grown = (unsigned char *)realloc(bytes->data, capacity);
    if (!grown) return 0;
    bytes->data = grown;
    bytes->capacity = capacity;
    return 1;
}

/* Append one exact borrowed span without advancing count on failure. */
static inline int yvex_core_bytes_append(yvex_core_bytes *bytes,
                                         const void *data,
                                         size_t count)
{
    if ((!data && count) || !yvex_core_bytes_reserve(bytes, count)) return 0;
    if (count) memcpy(bytes->data + bytes->count, data, count);
    bytes->count += count;
    return 1;
}

/*
 * Append canonical zero bytes without advancing count on failure.
 *
 * Initialized arena and exact padding count. Callers retain padding geometry and meaning.
 */
static inline int yvex_core_bytes_append_zero(yvex_core_bytes *bytes,
                                              size_t count)
{
    if (!yvex_core_bytes_reserve(bytes, count)) return 0;
    if (count) memset(bytes->data + bytes->count, 0, count);
    bytes->count += count;
    return 1;
}

/* Mix one borrowed byte range into a non-authoritative FNV-1a index hash. */
static inline unsigned long long yvex_core_hash_mix_bytes(unsigned long long hash,
                                                          const void *data,
                                                          size_t length)
{
    const unsigned char *bytes = (const unsigned char *)data;
    size_t index;

    if (!bytes && length != 0u) return hash;
    for (index = 0u; index < length; ++index) {
        hash ^= (unsigned long long)bytes[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

/* Mix one unsigned integer through its canonical little-endian bytes. */
static inline unsigned long long yvex_core_hash_mix_u64(unsigned long long hash,
                                                        unsigned long long value)
{
    unsigned char bytes[8];
    unsigned int index;

    for (index = 0u; index < sizeof(bytes); ++index)
        bytes[index] = (unsigned char)((value >> (index * 8u)) & 0xffull);
    return yvex_core_hash_mix_bytes(hash, bytes, sizeof(bytes));
}

/* Derive one nonzero deterministic process-local index hash. */
static inline unsigned long long yvex_core_index_hash(const char *text)
{
    const char *key = text ? text : "";
    unsigned long long hash =
        yvex_core_hash_mix_bytes(1469598103934665603ull, key, strlen(key));

    return hash ? hash : 1ull;
}
int yvex_core_u64_add(unsigned long long left,
                      unsigned long long right,
                      unsigned long long *out);
int yvex_core_u64_mul(unsigned long long left,
                      unsigned long long right,
                      unsigned long long *out);
void yvex_core_text_copy(char *destination, size_t capacity, const char *source);

/* Derive a checked power-of-two capacity for one bounded load ratio. */
static inline int yvex_core_power_of_two_capacity(unsigned long long count,
                                                   unsigned long long minimum,
                                                   unsigned long long load_numerator,
                                                   unsigned long long load_denominator,
                                                   unsigned long long *out)
{
    unsigned long long capacity = minimum, scaled, required;

    if (!out || !minimum || (minimum & (minimum - 1ull)) || !load_numerator ||
        !load_denominator || (count && load_denominator > ULLONG_MAX / count))
        return 0;
    scaled = count * load_denominator;
    required = scaled / load_numerator + (scaled % load_numerator != 0ull);
    while (capacity < required) {
        if (capacity > ULLONG_MAX / 2ull) return 0;
        capacity *= 2ull;
    }
    *out = capacity;
    return 1;
}
unsigned long long yvex_core_monotonic_ns(void);
int yvex_core_mkdir_parent(const char *path,
                           const char *where,
                           yvex_error *err);
void yvex_core_timestamp_utc(char *out, size_t capacity);

/* Safe file lifecycle shared by content-addressed repository owners. */
typedef enum {
    YVEX_CORE_FILE_STAGE_NONE = 0,
    YVEX_CORE_FILE_STAGE_ARGUMENT,
    YVEX_CORE_FILE_STAGE_PATH,
    YVEX_CORE_FILE_STAGE_CREATE,
    YVEX_CORE_FILE_STAGE_WRITE,
    YVEX_CORE_FILE_STAGE_FILE_SYNC,
    YVEX_CORE_FILE_STAGE_FILE_CLOSE,
    YVEX_CORE_FILE_STAGE_CONFLICT,
    YVEX_CORE_FILE_STAGE_PUBLISH,
    YVEX_CORE_FILE_STAGE_DIRECTORY_SYNC,
    YVEX_CORE_FILE_STAGE_OPEN,
    YVEX_CORE_FILE_STAGE_BOUNDS,
    YVEX_CORE_FILE_STAGE_ALLOCATION,
    YVEX_CORE_FILE_STAGE_READ,
    YVEX_CORE_FILE_STAGE_DRIFT,
    YVEX_CORE_FILE_STAGE_TEMPORARY_UNLINK,
    YVEX_CORE_FILE_STAGE_VALIDATE
} yvex_core_file_stage;
typedef enum {
    YVEX_CORE_FILE_CLEANUP_NONE = 0,
    YVEX_CORE_FILE_CLEANUP_FILE_CLOSE,
    YVEX_CORE_FILE_CLEANUP_TEMPORARY_UNLINK,
    YVEX_CORE_FILE_CLEANUP_DESTINATION_UNLINK,
    YVEX_CORE_FILE_CLEANUP_DIRECTORY_SYNC
} yvex_core_file_cleanup_stage;
typedef struct {
    int inject_file_close_failure;
    int inject_temporary_unlink_failure;
    int inject_cleanup_unlink_failure;
} yvex_core_file_faults;
typedef struct {
    yvex_core_file_stage stage;
    int system_error;
    unsigned long long expected, actual;
    yvex_core_file_cleanup_stage cleanup_stage;
    int cleanup_system_error;
} yvex_core_file_result;
typedef int (*yvex_core_file_validator)(int descriptor, size_t count,
                                        void *context, yvex_error *err);
int yvex_core_file_publish_noreplace(const char *path, const void *data, size_t count,
                                     const yvex_core_file_faults *faults,
                                     yvex_core_file_validator validator,
                                     void *validator_context,
                                     yvex_core_file_result *result, yvex_error *err);
int yvex_core_file_read_snapshot(const char *path, size_t maximum_bytes,
                                 unsigned char **data, size_t *count,
                                 yvex_core_file_result *result, yvex_error *err);
int yvex_core_file_read_descriptor_snapshot(
    int descriptor, size_t expected_count, unsigned char **data,
    yvex_core_file_result *result, yvex_error *err);

/* Deterministic shard indexes. */
typedef enum {
    YVEX_SHARD_INDEX_OK = 0,
    YVEX_SHARD_INDEX_INVALID,
    YVEX_SHARD_INDEX_BUDGET,
    YVEX_SHARD_INDEX_DUPLICATE_ID,
    YVEX_SHARD_INDEX_DUPLICATE_KEY,
    YVEX_SHARD_INDEX_NONCANONICAL_ORDER
} yvex_shard_index_result;
typedef struct {
    unsigned long long canonical_id;
    const char *canonical_key;
} yvex_shard_index_entry;
typedef struct {
    const yvex_shard_index_entry *entries;
    unsigned long long count;
} yvex_shard_index;
yvex_shard_index_result yvex_shard_index_init(
    yvex_shard_index *index,
    const yvex_shard_index_entry *entries,
    unsigned long long count,
    unsigned long long maximum_entries);
const yvex_shard_index_entry *yvex_shard_index_at(
    const yvex_shard_index *index,
    unsigned long long canonical_id);
const yvex_shard_index_entry *yvex_shard_index_find(
    const yvex_shard_index *index,
    const char *canonical_key,
    unsigned long long *comparisons);
void yvex_shard_index_reset(yvex_shard_index *index);

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_CORE_H_INCLUDED */
