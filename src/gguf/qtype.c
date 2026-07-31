/*
 * Define pinned qtype identity, row geometry, storage accounting, and refusal facts.
 *
 * Dimension zero is the GGML row width; block rows divide exactly; storage admission never implies
 * a decoder, emitter, quantizer, or compute kernel. This owner reads tensor geometry only and
 * never reads tensor payload bytes.
 */
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include <yvex/internal/core.h>
#include <yvex/internal/gguf.h>
#include <yvex/internal/quant_numeric.h>

static const yvex_gguf_qtype_geometry qtype_geometry[] = {
    {0u, "F32", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_FLOAT, 1u, 4u, 4u,
     0, "storage geometry admitted by pinned GGUF ABI"},
    {1u, "F16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_FLOAT, 1u, 2u, 2u,
     0, "storage geometry admitted by pinned GGUF ABI"},
    {2u, "Q4_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     18u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {3u, "Q4_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     20u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {4u, "Q4_2", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u, 0u, 0,
     "qtype ID was removed from the pinned GGUF ABI"},
    {5u, "Q4_3", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u, 0u, 0,
     "qtype ID was removed from the pinned GGUF ABI"},
    {6u, "Q5_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     22u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {7u, "Q5_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     24u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {8u, "Q8_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     34u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {9u, "Q8_1", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     36u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {10u, "Q2_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     84u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {11u, "Q3_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     110u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {12u, "Q4_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     144u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {13u, "Q5_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     176u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {14u, "Q6_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     210u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {15u, "Q8_K", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     292u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {16u, "IQ2_XXS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED,
     256u, 66u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {17u, "IQ2_XS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED,
     256u, 74u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {18u, "IQ3_XXS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED,
     256u, 98u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {19u, "IQ1_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     50u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {20u, "IQ4_NL", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     18u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {21u, "IQ3_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     110u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {22u, "IQ2_S", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     82u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {23u, "IQ4_XS", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED,
     256u, 136u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {24u, "I8", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_INTEGER, 1u, 1u,
     1u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {25u, "I16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_INTEGER, 1u, 2u,
     2u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {26u, "I32", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_INTEGER, 1u, 4u,
     4u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {27u, "I64", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_INTEGER, 1u, 8u,
     8u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {28u, "F64", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_FLOAT, 1u, 8u,
     8u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {29u, "IQ1_M", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     56u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {30u, "BF16", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_SCALAR_FLOAT, 1u, 2u,
     2u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {31u, "Q4_0_4_4", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u, 0u,
     0, "qtype ID was removed from the pinned GGUF ABI"},
    {32u, "Q4_0_4_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u, 0u,
     0, "qtype ID was removed from the pinned GGUF ABI"},
    {33u, "Q4_0_8_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u, 0u,
     0, "qtype ID was removed from the pinned GGUF ABI"},
    {34u, "TQ1_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     54u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {35u, "TQ2_0", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 256u,
     66u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {36u, "IQ4_NL_4_4", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u,
     0u, 0, "qtype ID was removed from the pinned GGUF ABI"},
    {37u, "IQ4_NL_4_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u,
     0u, 0, "qtype ID was removed from the pinned GGUF ABI"},
    {38u, "IQ4_NL_8_8", YVEX_GGUF_QTYPE_IDENTITY_REMOVED, YVEX_GGUF_QTYPE_STORAGE_UNKNOWN, 0u, 0u,
     0u, 0, "qtype ID was removed from the pinned GGUF ABI"},
    {39u, "MXFP4", YVEX_GGUF_QTYPE_IDENTITY_ADMITTED, YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 32u,
     17u, 0u, 0, "storage geometry admitted by pinned GGUF ABI"},
    {40u, "NVFP4", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE,
     YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 64u, 36u, 0u, 0,
     "qtype identity is newer than the pinned GGUF on-disk baseline"},
    {41u, "Q1_0", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE,
     YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 128u, 18u, 0u, 0,
     "qtype identity is newer than the pinned GGUF on-disk baseline"},
    {42u, "Q2_0", YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE,
     YVEX_GGUF_QTYPE_STORAGE_BLOCK_QUANTIZED, 64u, 18u, 0u, 0,
     "qtype identity is newer than the pinned GGUF on-disk baseline"}};

_Static_assert(sizeof(qtype_geometry) / sizeof(qtype_geometry[0]) == 43u,
               "pinned qtype registry must account for IDs 0 through 42");

static const char *const identity_status_names[] = {
    "admitted", "removed", "reserved", "outside-baseline", "unknown",
};

/* Expose the complete pinned qtype identity-table cardinality. */
size_t yvex_gguf_qtype_geometry_count(void) {
    return sizeof(qtype_geometry) / sizeof(qtype_geometry[0]);
}

const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_at(size_t index) {
    if (index >= yvex_gguf_qtype_geometry_count())
        return NULL;
    return &qtype_geometry[index];
}

/* Borrow one immutable geometry row by exact numeric qtype identity. */
const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_find(unsigned int qtype) {
    if (qtype > YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID)
        return NULL;
    return &qtype_geometry[qtype];
}

/*
 * Resolve an immutable geometry row by its exact canonical name.
 *
 * Names are projected from the numeric identity registry, never reparsed elsewhere.
 */
const yvex_gguf_qtype_geometry *yvex_gguf_qtype_geometry_find_by_name(const char *name) {
    size_t i;

    if (!name)
        return NULL;
    for (i = 0u; i < yvex_gguf_qtype_geometry_count(); ++i) {
        if (strcmp(qtype_geometry[i].name, name) == 0)
            return &qtype_geometry[i];
    }
    return NULL;
}

/* Project the canonical pinned name for one numeric qtype identity. */
const char *yvex_gguf_qtype_name(unsigned int qtype) {
    const yvex_gguf_qtype_geometry *geometry = yvex_gguf_qtype_geometry_find(qtype);

    return geometry ? geometry->name : "UNKNOWN";
}

/*
 * Render qtype identity admission as stable diagnostic text.
 *
 * Identity-status enum. Identity status remains distinct from storage and codec support.
 */
const char *yvex_gguf_qtype_identity_status_name(yvex_gguf_qtype_identity_status status) {
    return status >= YVEX_GGUF_QTYPE_IDENTITY_ADMITTED && status <= YVEX_GGUF_QTYPE_IDENTITY_UNKNOWN
               ? identity_status_names[status]
               : identity_status_names[YVEX_GGUF_QTYPE_IDENTITY_UNKNOWN];
}

typedef struct {
    const char *name;
    const char *reason;
} storage_status_text;

static const storage_status_text storage_status_texts[] = {
    {"ok", "GGUF tensor row geometry accepted"},
    {"invalid-argument", "qtype storage requires dimensions and an output result"},
    {"unknown-qtype-id", "unknown GGUF qtype ID"},
    {"removed-qtype-id", "GGUF qtype ID was removed from the pinned on-disk ABI"},
    {"reserved-qtype-id", "GGUF qtype ID is reserved by the pinned on-disk ABI"},
    {"outside-on-disk-baseline", "GGUF qtype ID is outside the pinned on-disk ABI"},
    {"storage-geometry-unavailable", "admitted GGUF qtype has no storage geometry"},
    {"invalid-rank", "GGUF tensor rank is outside the admitted range"},
    {"invalid-dimension", "GGUF tensor dimension is zero"},
    {"row-width-block-mismatch", "GGUF row width is not divisible by qtype block width"},
    {"element-count-overflow", "GGUF tensor element count overflows"},
    {"row-byte-overflow", "GGUF qtype row byte count overflows"},
    {"row-count-overflow", "GGUF tensor row count overflows"},
    {"total-byte-overflow", "GGUF tensor total byte count overflows"},
    {"expected-actual-storage-mismatch",
     "GGUF tensor storage bytes do not match qtype row geometry"},
};

const char *yvex_gguf_qtype_storage_status_name(yvex_gguf_qtype_storage_status status) {
    return status >= YVEX_GGUF_QTYPE_STORAGE_OK &&
                   (size_t)status < sizeof(storage_status_texts) / sizeof(storage_status_texts[0])
               ? storage_status_texts[status].name
               : "unknown-storage-status";
}

static yvex_gguf_qtype_storage_status
qtype_admission_status(const yvex_gguf_qtype_geometry *geometry) {
    if (!geometry)
        return YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID;
    switch (geometry->identity_status) {
    case YVEX_GGUF_QTYPE_IDENTITY_ADMITTED:
        break;
    case YVEX_GGUF_QTYPE_IDENTITY_REMOVED:
        return YVEX_GGUF_QTYPE_STORAGE_REMOVED_ID;
    case YVEX_GGUF_QTYPE_IDENTITY_RESERVED:
        return YVEX_GGUF_QTYPE_STORAGE_RESERVED_ID;
    case YVEX_GGUF_QTYPE_IDENTITY_OUTSIDE_BASELINE:
        return YVEX_GGUF_QTYPE_STORAGE_OUTSIDE_BASELINE;
    case YVEX_GGUF_QTYPE_IDENTITY_UNKNOWN:
        return YVEX_GGUF_QTYPE_STORAGE_UNKNOWN_ID;
    }
    if (geometry->storage_class == YVEX_GGUF_QTYPE_STORAGE_UNKNOWN || geometry->block_size == 0u ||
        geometry->bytes_per_block == 0u) {
        return YVEX_GGUF_QTYPE_STORAGE_GEOMETRY_UNAVAILABLE;
    }
    return YVEX_GGUF_QTYPE_STORAGE_OK;
}

static const char *storage_status_reason(yvex_gguf_qtype_storage_status status) {
    return status >= YVEX_GGUF_QTYPE_STORAGE_OK &&
                   (size_t)status < sizeof(storage_status_texts) / sizeof(storage_status_texts[0])
               ? storage_status_texts[status].reason
               : "unknown GGUF qtype storage refusal";
}

static yvex_gguf_qtype_storage_status set_storage_status(yvex_gguf_qtype_storage_result *out,
                                                         yvex_gguf_qtype_storage_status status) {
    if (out) {
        out->status = status;
        out->reason = storage_status_reason(status);
    }
    return status;
}

/*
 * Calculate exact contiguous tensor storage from row-aware qtype geometry.
 *
 * Qtype identity, dimensions, rank, and writable result. Invalid identity/rank/dimension, block
 * mismatch, or overflow refuses distinctly.
 */
yvex_gguf_qtype_storage_status yvex_gguf_qtype_tensor_storage(unsigned int qtype,
                                                              const unsigned long long *dims,
                                                              unsigned int rank,
                                                              yvex_gguf_qtype_storage_result *out) {
    const yvex_gguf_qtype_geometry *geometry;
    yvex_gguf_qtype_storage_status status;
    unsigned long long blocks;
    unsigned int i;

    if (!out)
        return YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    out->qtype = qtype;
    out->rank = rank;
    out->status = YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT;
    out->reason = storage_status_reason(out->status);
    if (!dims)
        return out->status;

    geometry = yvex_gguf_qtype_geometry_find(qtype);
    status = qtype_admission_status(geometry);
    if (status != YVEX_GGUF_QTYPE_STORAGE_OK) {
        return set_storage_status(out, status);
    }
    if (rank == 0u || rank > YVEX_GGUF_QTYPE_MAX_DIMS) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_INVALID_RANK);
    }
    for (i = 0u; i < rank; ++i) {
        if (dims[i] == 0ull) {
            return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_INVALID_DIMENSION);
        }
    }

    out->row_width = dims[0];
    if ((out->row_width % geometry->block_size) != 0ull) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_ROW_BLOCK_MISMATCH);
    }
    blocks = out->row_width / geometry->block_size;
    if (!yvex_core_u64_mul(blocks, geometry->bytes_per_block, &out->row_bytes)) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_ROW_BYTE_OVERFLOW);
    }

    out->row_count = 1ull;
    for (i = 1u; i < rank; ++i) {
        if (!yvex_core_u64_mul(out->row_count, dims[i], &out->row_count)) {
            return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_ROW_COUNT_OVERFLOW);
        }
    }
    if (!yvex_core_u64_mul(out->row_width, out->row_count, &out->element_count)) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_ELEMENT_COUNT_OVERFLOW);
    }
    if (!yvex_core_u64_mul(out->row_bytes, out->row_count, &out->total_bytes)) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_TOTAL_BYTE_OVERFLOW);
    }
    return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_OK);
}

yvex_gguf_qtype_storage_status
yvex_gguf_qtype_validate_tensor_storage(unsigned int qtype, const unsigned long long *dims,
                                        unsigned int rank, unsigned long long actual_storage_bytes,
                                        yvex_gguf_qtype_storage_result *out) {
    yvex_gguf_qtype_storage_status status;

    status = yvex_gguf_qtype_tensor_storage(qtype, dims, rank, out);
    if (!out || status != YVEX_GGUF_QTYPE_STORAGE_OK)
        return status;
    out->actual_bytes = actual_storage_bytes;
    if (out->total_bytes != actual_storage_bytes) {
        return set_storage_status(out, YVEX_GGUF_QTYPE_STORAGE_EXPECTED_ACTUAL_MISMATCH);
    }
    return status;
}

/*
 * Report whether one pinned qtype has structurally admitted storage geometry.
 *
 * Every non-admitted identity returns false with its canonical reason.
 */
int yvex_gguf_qtype_supported_for_storage(unsigned int qtype, const char **reason) {
    yvex_gguf_qtype_storage_status status =
        qtype_admission_status(yvex_gguf_qtype_geometry_find(qtype));

    if (reason)
        *reason = storage_status_reason(status);
    return status == YVEX_GGUF_QTYPE_STORAGE_OK;
}

/*
 * Calculate exact encoded bytes for one logical row through the compatibility ABI.
 *
 * Null output, invalid qtype, partial block, or overflow returns false.
 */
int yvex_gguf_qtype_storage_bytes(unsigned int qtype, unsigned long long row_element_count,
                                  unsigned long long *out, const char **reason) {
    yvex_gguf_qtype_storage_result result;
    yvex_gguf_qtype_storage_status status;

    if (!out) {
        if (reason)
            *reason = storage_status_reason(YVEX_GGUF_QTYPE_STORAGE_INVALID_ARGUMENT);
        return 0;
    }
    *out = 0ull;
    status = yvex_gguf_qtype_tensor_storage(qtype, &row_element_count, 1u, &result);
    if (reason)
        *reason = result.reason;
    if (status != YVEX_GGUF_QTYPE_STORAGE_OK)
        return 0;
    *out = result.total_bytes;
    return 1;
}
