/*
 * Immutable artifact lowering is a compiler product. These records describe the sealed mapping
 * from semantic transform terminals to container operands without owning family policy or payload
 * resources.
 */
#ifndef INCLUDE_YVEX_INTERNAL_ARTIFACT_LOWERING_H_INCLUDED
#define INCLUDE_YVEX_INTERNAL_ARTIFACT_LOWERING_H_INCLUDED

#include <stddef.h>
#include <yvex/gguf.h>
#include <yvex/model.h>
#include <yvex/source.h>

#ifdef __cplusplus
extern "C" {
#endif

#define YVEX_ARTIFACT_LOWERING_NO_INDEX (~0ull)
#define YVEX_ARTIFACT_LOWERING_AGGREGATED_AXIS (~0u)
#define YVEX_ARTIFACT_LOWERING_POLICY_SCHEMA_V1 1u
#define YVEX_ARTIFACT_LOWERING_METADATA_CAP 64u

typedef unsigned int yvex_artifact_lowering_transform;
enum {
    YVEX_ARTIFACT_LOWERING_TRANSFORM_DIRECT = 0,
    YVEX_ARTIFACT_LOWERING_TRANSFORM_FP8_E4M3_E8M0,
    YVEX_ARTIFACT_LOWERING_TRANSFORM_EXPERT_MXFP4,
    YVEX_ARTIFACT_LOWERING_TRANSFORM_I64_TO_I32
};

typedef unsigned int yvex_artifact_lowering_contribution_kind;
enum {
    YVEX_ARTIFACT_LOWERING_CONTRIBUTION_PRIMARY = 0,
    YVEX_ARTIFACT_LOWERING_CONTRIBUTION_SCALE,
    YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_WEIGHT,
    YVEX_ARTIFACT_LOWERING_CONTRIBUTION_EXPERT_SCALE,
    YVEX_ARTIFACT_LOWERING_CONTRIBUTION_ROUTING_TABLE
};

typedef unsigned int yvex_artifact_lowering_metadata_type;
enum {
    YVEX_ARTIFACT_LOWERING_METADATA_STRING = 0,
    YVEX_ARTIFACT_LOWERING_METADATA_U64,
    YVEX_ARTIFACT_LOWERING_METADATA_F64,
    YVEX_ARTIFACT_LOWERING_METADATA_BOOL,
    YVEX_ARTIFACT_LOWERING_METADATA_U64_ARRAY,
    YVEX_ARTIFACT_LOWERING_METADATA_F64_ARRAY
};

typedef unsigned int yvex_artifact_lowering_failure_code;
enum {
    YVEX_ARTIFACT_LOWERING_FAILURE_NONE = 0,
    YVEX_ARTIFACT_LOWERING_FAILURE_INVALID_ARGUMENT,
    YVEX_ARTIFACT_LOWERING_FAILURE_ARCHITECTURE,
    YVEX_ARTIFACT_LOWERING_FAILURE_COVERAGE_ROW,
    YVEX_ARTIFACT_LOWERING_FAILURE_MISSING_SOURCE,
    YVEX_ARTIFACT_LOWERING_FAILURE_DUPLICATE_SOURCE,
    YVEX_ARTIFACT_LOWERING_FAILURE_SOURCE_DTYPE,
    YVEX_ARTIFACT_LOWERING_FAILURE_EXPERT_SEQUENCE,
    YVEX_ARTIFACT_LOWERING_FAILURE_NAME,
    YVEX_ARTIFACT_LOWERING_FAILURE_DUPLICATE_NAME,
    YVEX_ARTIFACT_LOWERING_FAILURE_LAYOUT,
    YVEX_ARTIFACT_LOWERING_FAILURE_METADATA,
    YVEX_ARTIFACT_LOWERING_FAILURE_ACCOUNTING,
    YVEX_ARTIFACT_LOWERING_FAILURE_ARITHMETIC_OVERFLOW,
    YVEX_ARTIFACT_LOWERING_FAILURE_ALLOCATION,
    YVEX_ARTIFACT_LOWERING_FAILURE_TRANSFORM_IR,
    YVEX_ARTIFACT_LOWERING_FAILURE_LOWERING_DIVERGENCE,
    YVEX_ARTIFACT_LOWERING_FAILURE_MAPPING_IDENTITY
};

typedef struct yvex_artifact_lowering_failure {
    yvex_artifact_lowering_failure_code code;
    yvex_tensor_role role;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_index, expected, actual;
    char source_name[256];
    char emitted_name[192];
} yvex_artifact_lowering_failure;

typedef struct yvex_artifact_lowering_contribution {
    char source_name[256];
    yvex_native_dtype source_dtype;
    unsigned int source_rank;
    unsigned long long source_dims[2];
    yvex_artifact_lowering_contribution_kind kind;
    unsigned long long source_row_index, descriptor_index, expert_index;
} yvex_artifact_lowering_contribution;

typedef struct yvex_artifact_lowering_descriptor {
    yvex_tensor_role role;
    yvex_tensor_collection collection;
    yvex_tensor_scope scope;
    unsigned long long layer_index, predictor_index, expert_count;
    char emitted_name[192];
    yvex_artifact_lowering_transform transform;
    yvex_gguf_name_provenance name_provenance;
    unsigned int forced_qtype, logical_rank;
    unsigned long long logical_dims[YVEX_TENSOR_MAX_DIMS];
    unsigned int source_axis_for_logical[YVEX_TENSOR_MAX_DIMS];
    unsigned long long contribution_offset, contribution_count, identity;
} yvex_artifact_lowering_descriptor;

typedef struct yvex_artifact_lowering_metadata {
    char key[128];
    yvex_artifact_lowering_metadata_type type;
    char string_value[192];
    unsigned long long u64_value;
    double f64_value;
    int bool_value;
    unsigned long long array_values[64];
    double f64_array_values[64];
    unsigned int array_count;
} yvex_artifact_lowering_metadata;

/*
 * Families project exact container policy before generic lowering begins. The policy is borrowed
 * only for construction; the resulting immutable map owns copies of its metadata values.
 */
typedef struct yvex_artifact_lowering_policy {
    unsigned int schema_version;
    unsigned long long source_contribution_count;
    unsigned long long descriptor_count;
    unsigned long long trunk_descriptor_count;
    unsigned long long draft_descriptor_count;
    unsigned long long pinned_standard_count;
    unsigned long long extension_count;
    unsigned long long trunk_collection_counts[YVEX_TENSOR_COLLECTION_COUNT];
    const yvex_artifact_lowering_metadata *metadata;
    unsigned long long metadata_count;
} yvex_artifact_lowering_policy;

typedef struct yvex_artifact_lowering_summary {
    unsigned long long source_contribution_count, descriptor_count, trunk_descriptor_count;
    unsigned long long draft_descriptor_count, pinned_standard_count, semantic_standard_count;
    unsigned long long extension_count;
    unsigned long long collection_counts[YVEX_TENSOR_COLLECTION_COUNT];
    unsigned long long metadata_count, header_scan_count, payload_bytes_read, source_identity;
    unsigned long long coverage_identity, mapping_identity;
    int complete;
} yvex_artifact_lowering_summary;

typedef void *(*yvex_artifact_lowering_allocate_fn)(size_t, void *);
typedef void (*yvex_artifact_lowering_release_fn)(void *, void *);
typedef struct yvex_artifact_lowering_allocator {
    yvex_artifact_lowering_allocate_fn allocate;
    yvex_artifact_lowering_release_fn release;
    void *context;
} yvex_artifact_lowering_allocator;

typedef struct yvex_artifact_lowering_map yvex_artifact_lowering_map;
struct yvex_transform_ir;

typedef struct yvex_artifact_lowering_api {
    int (*build)(yvex_artifact_lowering_map **out,
                 const struct yvex_transform_ir *transform_ir,
                 const yvex_artifact_lowering_policy *policy,
                 yvex_artifact_lowering_failure *failure,
                 yvex_error *err);
    int (*build_with_allocator)(yvex_artifact_lowering_map **out,
                                const struct yvex_transform_ir *transform_ir,
                                const yvex_artifact_lowering_policy *policy,
                                const yvex_artifact_lowering_allocator *allocator,
                                yvex_artifact_lowering_failure *failure,
                                yvex_error *err);
    void (*close)(yvex_artifact_lowering_map *map);
    const yvex_artifact_lowering_summary *(*summary)(
        const yvex_artifact_lowering_map *map);
    const yvex_artifact_lowering_descriptor *(*descriptor_at)(
        const yvex_artifact_lowering_map *map, unsigned long long index);
    const yvex_artifact_lowering_contribution *(*contribution_at)(
        const yvex_artifact_lowering_map *map, unsigned long long index);
    const yvex_artifact_lowering_descriptor *(*find_source)(
        const yvex_artifact_lowering_map *map, const char *source_name);
    const yvex_artifact_lowering_descriptor *(*find_emitted)(
        const yvex_artifact_lowering_map *map, const char *emitted_name);
    const yvex_artifact_lowering_descriptor *(*find_role)(
        const yvex_artifact_lowering_map *map, yvex_tensor_role role,
        yvex_tensor_scope scope, unsigned long long layer_index,
        unsigned long long predictor_index);
    const yvex_artifact_lowering_metadata *(*metadata_at)(
        const yvex_artifact_lowering_map *map, unsigned long long index);
    const yvex_artifact_lowering_metadata *(*metadata_find)(
        const yvex_artifact_lowering_map *map, const char *key);
    const char *(*transform_name)(yvex_artifact_lowering_transform transform);
    const char *(*failure_name)(yvex_artifact_lowering_failure_code code);
} yvex_artifact_lowering_api;

extern const yvex_artifact_lowering_api yvex_artifact_lowering_operations;

#ifdef __cplusplus
}
#endif

#endif /* INCLUDE_YVEX_INTERNAL_ARTIFACT_LOWERING_H_INCLUDED */
