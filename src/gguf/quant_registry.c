/*
 * Project one canonical numeric and compute capability row for every pinned qtype ID.
 *
 * Table ordinal equals the canonical GGUF qtype ID; removed and post-baseline identities always
 * refuse deterministically. An implemented primitive is not materialization or runtime support.
 */
#include <string.h>
#include <yvex/internal/quant_numeric.h>
#include <yvex/qtype.h>

#define SCALAR_CLASSES                                                                             \
    (YVEX_TRANSFORM_PHYSICAL_F32 | YVEX_TRANSFORM_PHYSICAL_F16 | YVEX_TRANSFORM_PHYSICAL_BF16 |    \
     YVEX_TRANSFORM_PHYSICAL_I32)
#define REAL_TRANSFORMS                                                                            \
    (YVEX_QUANT_TRANSFORM_IDENTITY | YVEX_QUANT_TRANSFORM_SCALE_PAIR | YVEX_QUANT_TRANSFORM_EXPERT)
#define CAP(q, storage, enc, dec, cpu, cuda, cal, classes, transforms, refusal)                    \
    {                                                                                              \
        {q,                                                                                        \
         1,                                                                                        \
         storage,                                                                                  \
         enc,                                                                                      \
         dec,                                                                                      \
         cpu,                                                                                      \
         cuda,                                                                                     \
         cal,                                                                                      \
         classes,                                                                                  \
         transforms,                                                                               \
         YVEX_QUANT_NUMERIC_CONTRACT_VERSION,                                                      \
         enc,                                                                                      \
         refusal},                                                                                 \
        {                                                                                          \
            q, storage, enc, enc, cpu, "canonical TRACK.QUANT numeric capability projection"       \
        }                                                                                          \
    }
#define ADMITTED_NO_CODEC(q)                                                                       \
    CAP(q, 1, 0, 0, 0, 0, YVEX_QUANT_CALIBRATION_UNSUPPORTED, 0u, 0u,                              \
        YVEX_QUANT_REFUSAL_ENCODER_UNAVAILABLE)
#define REMOVED(q)                                                                                 \
    CAP(q, 0, 0, 0, 0, 0, YVEX_QUANT_CALIBRATION_UNSUPPORTED, 0u, 0u,                              \
        YVEX_QUANT_REFUSAL_REMOVED_IDENTITY)
#define OUTSIDE(q)                                                                                 \
    CAP(q, 0, 0, 0, 0, 0, YVEX_QUANT_CALIBRATION_UNSUPPORTED, 0u, 0u,                              \
        YVEX_QUANT_REFUSAL_OUTSIDE_PINNED_BASELINE)

typedef struct {
    yvex_quant_numeric_capability capability;
    yvex_qtype_support_info compatibility;
} quant_registry_row;

static const quant_registry_row quant_registry[] = {
    CAP(YVEX_GGUF_QTYPE_F32, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_F32, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    CAP(YVEX_GGUF_QTYPE_F16, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_F16, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q4_0),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q4_1),
    REMOVED(YVEX_GGUF_QTYPE_Q4_2_REMOVED),
    REMOVED(YVEX_GGUF_QTYPE_Q4_3_REMOVED),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q5_0),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q5_1),
    CAP(YVEX_GGUF_QTYPE_Q8_0, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_QUANTIZED, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q8_1),
    CAP(YVEX_GGUF_QTYPE_Q2_K, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_OPTIONAL,
        YVEX_TRANSFORM_PHYSICAL_QUANTIZED, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q3_K),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q4_K),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q5_K),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q6_K),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_Q8_K),
    CAP(YVEX_GGUF_QTYPE_IQ2_XXS, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_REQUIRED,
        YVEX_TRANSFORM_PHYSICAL_QUANTIZED, YVEX_QUANT_TRANSFORM_EXPERT,
        YVEX_QUANT_REFUSAL_NONE),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ2_XS),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ3_XXS),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ1_S),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ4_NL),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ3_S),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ2_S),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ4_XS),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_I8),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_I16),
    CAP(YVEX_GGUF_QTYPE_I32, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_I32,
        YVEX_QUANT_TRANSFORM_CHECKED_CAST | YVEX_QUANT_TRANSFORM_IDENTITY, YVEX_QUANT_REFUSAL_NONE),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_I64),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_F64),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_IQ1_M),
    CAP(YVEX_GGUF_QTYPE_BF16, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_BF16, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    REMOVED(YVEX_GGUF_QTYPE_Q4_0_4_4_REMOVED),
    REMOVED(YVEX_GGUF_QTYPE_Q4_0_4_8_REMOVED),
    REMOVED(YVEX_GGUF_QTYPE_Q4_0_8_8_REMOVED),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_TQ1_0),
    ADMITTED_NO_CODEC(YVEX_GGUF_QTYPE_TQ2_0),
    REMOVED(YVEX_GGUF_QTYPE_IQ4_NL_4_4_REMOVED),
    REMOVED(YVEX_GGUF_QTYPE_IQ4_NL_4_8_REMOVED),
    REMOVED(YVEX_GGUF_QTYPE_IQ4_NL_8_8_REMOVED),
    CAP(YVEX_GGUF_QTYPE_MXFP4, 1, 1, 1, 1, 1, YVEX_QUANT_CALIBRATION_NONE,
        YVEX_TRANSFORM_PHYSICAL_QUANTIZED, REAL_TRANSFORMS, YVEX_QUANT_REFUSAL_NONE),
    OUTSIDE(YVEX_GGUF_QTYPE_NVFP4_OUTSIDE_BASELINE),
    OUTSIDE(YVEX_GGUF_QTYPE_Q1_0_OUTSIDE_BASELINE),
    OUTSIDE(YVEX_GGUF_QTYPE_Q2_0_OUTSIDE_BASELINE)};

typedef char
    quant_capability_count_must_cover_all_ids[sizeof(quant_registry) / sizeof(quant_registry[0]) ==
                                                      YVEX_GGUF_QTYPE_ABI_UPSTREAM_MAX_ID + 1u
                                                  ? 1
                                                  : -1];

static const char *const calibration_names[] = {"none", "optional", "required", "unsupported"};
/* Resolve the immutable numeric capability for one exact qtype identity. */
const yvex_quant_numeric_capability *yvex_quant_numeric_capability_at(unsigned int qtype) {
    if (qtype >= sizeof(quant_registry) / sizeof(quant_registry[0]))
        return NULL;
    return &quant_registry[qtype].capability;
}

/*
 * Resolve a numeric capability by the canonical geometry-owned qtype name.
 *
 * Name lookup never creates a second qtype identity authority.
 */
const yvex_quant_numeric_capability *yvex_quant_numeric_capability_by_name(const char *name) {
    const yvex_gguf_qtype_geometry *geometry;

    if (!name)
        return NULL;
    geometry = yvex_gguf_qtype_geometry_find_by_name(name);
    return geometry ? yvex_quant_numeric_capability_at(geometry->qtype) : NULL;
}

static unsigned int numeric_capability_count(void) {
    return (unsigned int)(sizeof(quant_registry) / sizeof(quant_registry[0]));
}

const yvex_qtype_support_info *yvex_qtype_support_by_name(const char *qtype) {
    const yvex_gguf_qtype_geometry *geometry;

    if (!qtype)
        return NULL;
    geometry = yvex_gguf_qtype_geometry_find_by_name(qtype);
    return geometry && geometry->qtype < numeric_capability_count()
               ? &quant_registry[geometry->qtype].compatibility
               : NULL;
}

unsigned long long yvex_qtype_support_count(void) {
    return numeric_capability_count();
}

const yvex_qtype_support_info *yvex_qtype_support_at(unsigned long long index) {
    return index < numeric_capability_count() ? &quant_registry[index].compatibility : NULL;
}

const char *yvex_qtype_support_name(const yvex_qtype_support_info *info) {
    return info ? yvex_gguf_qtype_name(info->ggml_type) : "UNKNOWN";
}

int yvex_qtype_support_storage_supported(const yvex_qtype_support_info *info) {
    const yvex_quant_numeric_capability *capability =
        info ? yvex_quant_numeric_capability_at(info->ggml_type) : NULL;
    return capability && capability->storage_admitted;
}

const char *yvex_quant_calibration_name(yvex_quant_calibration_requirement requirement) {
    return requirement <= YVEX_QUANT_CALIBRATION_UNSUPPORTED ? calibration_names[requirement]
                                                              : "unknown";
}
