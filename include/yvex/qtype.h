/* Qtype identity, storage geometry, codec support, and executable backend support are separate
 * facts. This ABI owns the stable identity and geometry vocabulary shared across those gates. */
#ifndef YVEX_QTYPE_H
#define YVEX_QTYPE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Qtype identity and geometry. */
#define YVEX_GGUF_QTYPE_ABI_UPSTREAM_COMMIT \
    "af97976c7810cdabb1863172f31c432dab767de7"
#define YVEX_GGUF_QTYPE_ABI_ON_DISK_MAX_ID 39u
#define YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID 42u
#define YVEX_GGUF_QTYPE_MAX_DIMS 4u

typedef enum {
    YVEX_GGUF_QTYPE_F32 = 0,
    YVEX_GGUF_QTYPE_F16 = 1,
    YVEX_GGUF_QTYPE_Q4_0 = 2,
    YVEX_GGUF_QTYPE_Q4_1 = 3,
    YVEX_GGUF_QTYPE_Q4_2_REMOVED = 4,
    YVEX_GGUF_QTYPE_Q4_3_REMOVED = 5,
    YVEX_GGUF_QTYPE_Q5_0 = 6,
    YVEX_GGUF_QTYPE_Q5_1 = 7,
    YVEX_GGUF_QTYPE_Q8_0 = 8,
    YVEX_GGUF_QTYPE_Q8_1 = 9,
    YVEX_GGUF_QTYPE_Q2_K = 10,
    YVEX_GGUF_QTYPE_Q3_K = 11,
    YVEX_GGUF_QTYPE_Q4_K = 12,
    YVEX_GGUF_QTYPE_Q5_K = 13,
    YVEX_GGUF_QTYPE_Q6_K = 14,
    YVEX_GGUF_QTYPE_Q8_K = 15,
    YVEX_GGUF_QTYPE_IQ2_XXS = 16,
    YVEX_GGUF_QTYPE_IQ2_XS = 17,
    YVEX_GGUF_QTYPE_IQ3_XXS = 18,
    YVEX_GGUF_QTYPE_IQ1_S = 19,
    YVEX_GGUF_QTYPE_IQ4_NL = 20,
    YVEX_GGUF_QTYPE_IQ3_S = 21,
    YVEX_GGUF_QTYPE_IQ2_S = 22,
    YVEX_GGUF_QTYPE_IQ4_XS = 23,
    YVEX_GGUF_QTYPE_I8 = 24,
    YVEX_GGUF_QTYPE_I16 = 25,
    YVEX_GGUF_QTYPE_I32 = 26,
    YVEX_GGUF_QTYPE_I64 = 27,
    YVEX_GGUF_QTYPE_F64 = 28,
    YVEX_GGUF_QTYPE_IQ1_M = 29,
    YVEX_GGUF_QTYPE_BF16 = 30,
    YVEX_GGUF_QTYPE_Q4_0_4_4_REMOVED = 31,
    YVEX_GGUF_QTYPE_Q4_0_4_8_REMOVED = 32,
    YVEX_GGUF_QTYPE_Q4_0_8_8_REMOVED = 33,
    YVEX_GGUF_QTYPE_TQ1_0 = 34,
    YVEX_GGUF_QTYPE_TQ2_0 = 35,
    YVEX_GGUF_QTYPE_IQ4_NL_4_4_REMOVED = 36,
    YVEX_GGUF_QTYPE_IQ4_NL_4_8_REMOVED = 37,
    YVEX_GGUF_QTYPE_IQ4_NL_8_8_REMOVED = 38,
    YVEX_GGUF_QTYPE_MXFP4 = 39,
    YVEX_GGUF_QTYPE_NVFP4_OUTSIDE_BASELINE = 40,
    YVEX_GGUF_QTYPE_Q1_0_OUTSIDE_BASELINE = 41,
    YVEX_GGUF_QTYPE_Q2_0_OUTSIDE_BASELINE = 42
} yvex_gguf_qtype_id;

typedef enum {
    YVEX_GGUF_QTYPE_IDENTITY_ADMITTED = 0,
    YVEX_GGUF_QTYPE_IDENTITY_REMOVED = 1,
    YVEX_GGUF_QTYPE_IDENTITY_RESERVED = 2,
    YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE = 3,
    YVEX_GGUF_QTYPE_IDENTITY_UNKNOWN = 4
} yvex_gguf_qtype_identity_status;

typedef enum {
    YVEX_GGUF_QTYPE_STORAGE_UNKNOWN = 0,
    YVEX_GGUF_QTYPE_STORAGE_SCALAR_FLOAT = 1,
    YVEX_GGUF_QTYPE_STORAGE_SCALAR_INTEGER = 2,
    YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED = 3
} yvex_gguf_qtype_storage_class;

typedef enum {
    YVEX_GGUF_QTYPE_STORAGE_OK = 0,
    YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT = 1,
    YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID = 2,
    YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID = 3,
    YVEX_GGUF_QTYPE_STORAGE_RESERVED_ID = 4,
    YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE = 5,
    YVEX_GGUF_QTYPE_STORAGE_GEOMETRY_UNAVAILABLE = 6,
    YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK = 7,
    YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION = 8,
    YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH = 9,
    YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW = 10,
    YVEX_GGUF_QTYPE_STORAGE_ROW_BYTE_OVERFLOW = 11,
    YVEX_GGUF_QTYPE_STORAGE_ROW_COUNT_OVERFLOW = 12,
    YVEX_GGUF_QTYPE_STORAGE_TOTAL_BYTE_OVERFLOW = 13,
    YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH = 14
} yvex_gguf_qtype_storage_status;

typedef struct {
    unsigned int qtype;
    const char *name;
    yvex_gguf_qtype_identity_status identity_status;
    yvex_gguf_qtype_storage_class storage_class;
    unsigned int block_size;
    unsigned int bytes_per_block;
    unsigned int scalar_width;
    unsigned int numeric_capability_reserved;
    const char *reason;
} yvex_gguf_qtype_geometry;

typedef struct {
    yvex_gguf_qtype_storage_status status;
    unsigned int qtype;
    unsigned int rank;
    unsigned long long row_width;
    unsigned long long row_count;
    unsigned long long element_count;
    unsigned long long row_bytes;
    unsigned long long total_bytes;
    unsigned long long actual_bytes;
    const char *reason;
} yvex_gguf_qtype_storage_result;

size_t yvex_gguf_qtype_geometry_count(void);
const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_at(size_t index);
const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_find(unsigned int qtype);
const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_find_by_name(const char *name);

const char *yvex_gguf_qtype_name(unsigned int qtype);
const char *yvex_gguf_qtype_identity_status_name(yvex_gguf_qtype_identity_status status);
const char *yvex_gguf_qtype_storage_status_name(yvex_gguf_qtype_storage_status status);

int yvex_gguf_qtype_supported_for_storage(unsigned int qtype, const char **reason);

yvex_gguf_qtype_storage_status yvex_gguf_qtype_tensor_storage(
    unsigned int qtype,
    const unsigned long long *dims,
    unsigned int rank,
    yvex_gguf_qtype_storage_result *out);

yvex_gguf_qtype_storage_status yvex_gguf_qtype_validate_tensor_storage(
    unsigned int qtype,
    const unsigned long long *dims,
    unsigned int rank,
    unsigned long long actual_storage_bytes,
    yvex_gguf_qtype_storage_result *out);

/* The compatibility form interprets element_count as one logical row, never a tensor. */
int yvex_gguf_qtype_storage_bytes(unsigned int qtype,
                                  unsigned long long row_element_count,
                                  unsigned long long *out,
                                  const char **reason);

/* Qtype capability projection. */
typedef struct {
    unsigned int ggml_type;
    int policy_supported;
    int emit_supported;
    int quantize_supported;
    int compute_supported;
    const char *notes;
} yvex_qtype_support_info;

const yvex_qtype_support_info *yvex_qtype_support_by_name(const char *qtype);
unsigned long long yvex_qtype_support_count(void);
const yvex_qtype_support_info *yvex_qtype_support_at(unsigned long long index);
const char *yvex_qtype_support_name(const yvex_qtype_support_info *info);
int yvex_qtype_support_storage_supported(const yvex_qtype_support_info *info);

#ifdef __cplusplus
}
#endif

#endif /* YVEX_QTYPE_H */
